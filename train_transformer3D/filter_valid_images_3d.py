"""
筛选能提取到3D关键点和姿态的图片
使用InsightFace提取2D关键点，然后通过PnP算法估计3D关键点和姿态
保存特征、3D关键点、姿态和路径到JSON文件
"""
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm
import logging
import numpy as np

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils import get_insightface_detector
from train_transformer3D.utils_3d import get_3d_landmarks_and_pose
from feature_extractor import DINOv2FeatureExtractor
from train_transformer.utils_seed import set_seed
from PIL import Image

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_valid_images_3d(
    video_dir: str,
    face_dir: str,
    output_file: str = 'train_transformer3D/valid_images_3d.json',
    use_cpu: bool = False,
    dinov2_model_name: str = 'dinov2_vitb14',
    base_dir: str = None
):
    """
    筛选能提取到3D关键点和姿态的图片，并保存到JSON
    
    Args:
        video_dir: 视频帧图片目录
        face_dir: 正面图片目录
        output_file: 输出文件路径
        use_cpu: 是否使用CPU
        dinov2_model_name: DINOv2模型名称，默认'dinov2_vitb14'（768维）
        base_dir: 基础目录，用于计算相对路径（如果为None，使用output_file的父目录）
    """
    video_dir = Path(video_dir)
    face_dir = Path(face_dir)
    output_file = Path(output_file)
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 确定基础目录（用于计算相对路径）
    if base_dir is None:
        base_dir = output_file.parent
    else:
        base_dir = Path(base_dir)
    
    # 初始化检测器和特征提取器
    logger.info("初始化InsightFace检测器...")
    detector = get_insightface_detector(use_cpu=use_cpu)
    
    logger.info(f"初始化DINOv2特征提取器（模型: {dinov2_model_name}）...")
    model_dir = os.environ.get('DINOV2_MODEL_DIR', None)
    if model_dir:
        logger.info(f"  使用模型目录: {model_dir}")
    
    device = 'cpu' if use_cpu else None
    feature_extractor = DINOv2FeatureExtractor(
        model_name=dinov2_model_name,
        resize_to_96=False,
        device=device,
        model_dir=model_dir
    )
    
    # 获取所有正面图片的人名
    face_names = set()
    face_images = {}
    for face_file in face_dir.glob('*.jpg'):
        person_name = face_file.stem
        face_names.add(person_name)
        face_images[person_name] = face_file
    
    logger.info(f"找到正面图片人名: {len(face_names)} 个")
    
    # 存储有效数据（包含特征、3D关键点和姿态）
    valid_data = {}
    
    # 遍历每个人
    for person_name in face_names:
        logger.info(f"\n处理 {person_name}...")
        
        # 检查正面图是否能检测到3D关键点和姿态
        face_image_path = face_images[person_name]
        
        # 提取3D关键点和姿态
        face_landmarks_2d, face_landmarks_3d, face_box, face_euler_angles, face_rotation_matrix = \
            get_3d_landmarks_and_pose(detector, str(face_image_path))
        
        if face_landmarks_2d is None or face_landmarks_3d is None:
            logger.warning(f"正面图无法提取3D关键点: {face_image_path}")
            continue
        
        # 提取DINOv2特征
        try:
            face_features = feature_extractor.extract_features(str(face_image_path))
            logger.info(f"  ✓ 正面图特征提取成功，维度: {len(face_features)}")
        except Exception as e:
            logger.warning(f"正面图特征提取失败: {face_image_path}, 错误: {e}")
            continue
        
        # 计算相对路径
        face_image_path_rel = face_image_path.relative_to(base_dir) if base_dir else face_image_path
        face_image_path_str = str(face_image_path_rel).replace('\\', '/')
        
        # 获取视频帧图片目录
        video_person_dir = video_dir / person_name
        if not video_person_dir.exists():
            logger.warning(f"视频目录不存在: {video_person_dir}")
            continue
        
        # 获取所有视频帧图片
        video_images = sorted(list(video_person_dir.glob('*.jpg')))
        logger.info(f"找到 {len(video_images)} 张视频帧图片")
        
        # 筛选能提取到3D关键点和姿态的图片
        valid_video_data = []
        for video_image_path in tqdm(video_images, desc=f"处理 {person_name}"):
            # 提取3D关键点和姿态
            video_landmarks_2d, video_landmarks_3d, video_box, video_euler_angles, video_rotation_matrix = \
                get_3d_landmarks_and_pose(detector, str(video_image_path))
            
            if video_landmarks_2d is None or video_landmarks_3d is None:
                continue
            
            # 提取DINOv2特征
            try:
                video_features = feature_extractor.extract_features(str(video_image_path))
            except Exception as e:
                logger.debug(f"特征提取失败: {video_image_path}, 错误: {e}")
                continue
            
            # 计算相对路径
            video_image_path_rel = video_image_path.relative_to(base_dir) if base_dir else video_image_path
            video_image_path_str = str(video_image_path_rel).replace('\\', '/')
            
            # 保存数据
            video_item = {
                'image_path': video_image_path_str,
                'features': video_features.tolist(),
                'landmarks_2d': video_landmarks_2d.tolist(),  # [5, 2]
                'landmarks_3d': video_landmarks_3d.tolist(),  # [5, 3]
                'box': video_box.tolist(),  # [4]
            }
            
            # 添加姿态信息
            if video_euler_angles is not None:
                video_item['euler_angles'] = video_euler_angles.tolist()  # [yaw, pitch, roll] 度
            if video_rotation_matrix is not None:
                video_item['rotation_matrix'] = video_rotation_matrix.tolist()  # [3, 3]
            
            valid_video_data.append(video_item)
        
        logger.info(f"{person_name}: {len(valid_video_data)}/{len(video_images)} 张图片有效")
        
        if len(valid_video_data) > 0:
            person_data = {
                'face_image_path': face_image_path_str,
                'face_features': face_features.tolist(),
                'face_landmarks_2d': face_landmarks_2d.tolist(),  # [5, 2]
                'face_landmarks_3d': face_landmarks_3d.tolist(),  # [5, 3]
                'face_box': face_box.tolist(),  # [4]
                'video_data': valid_video_data,
                'total_video_images': len(video_images),
                'valid_video_images': len(valid_video_data),
                'feature_dim': len(face_features),
                'dinov2_model': dinov2_model_name
            }
            
            # 添加正面图的姿态信息
            if face_euler_angles is not None:
                person_data['face_euler_angles'] = face_euler_angles.tolist()  # [yaw, pitch, roll] 度
            if face_rotation_matrix is not None:
                person_data['face_rotation_matrix'] = face_rotation_matrix.tolist()  # [3, 3]
            
            valid_data[person_name] = person_data
    
    # 保存到文件
    logger.info(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    total_persons = len(valid_data)
    total_valid_images = sum(info['valid_video_images'] for info in valid_data.values())
    total_video_images = sum(info['total_video_images'] for info in valid_data.values())
    
    logger.info(f"\n统计信息:")
    logger.info(f"  总人数: {total_persons}")
    logger.info(f"  总视频帧数: {total_video_images}")
    logger.info(f"  有效视频帧数: {total_valid_images}")
    logger.info(f"  有效率: {total_valid_images/total_video_images*100:.2f}%" if total_video_images > 0 else "  有效率: 0%")
    
    # 检查特征维度一致性
    feature_dims = set()
    for info in valid_data.values():
        feature_dims.add(info['feature_dim'])
    
    if len(feature_dims) > 1:
        logger.warning(f"⚠️  发现不同的特征维度: {feature_dims}")
    else:
        logger.info(f"✓ 所有特征维度一致: {list(feature_dims)[0]}")
    
    logger.info(f"\n✓ 处理完成！")
    
    return valid_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='筛选能提取到3D关键点和姿态的图片')
    parser.add_argument('--video_dir', type=str,
                       default=r'/root/face000/train/datas/video',
                       help='视频帧图片目录')
    parser.add_argument('--face_dir', type=str,
                       default=r'/root/face000/train/datas/face',
                       help='正面图片目录')
    parser.add_argument('--output_file', type=str,
                       default='train_transformer3D/valid_images_3d.json',
                       help='输出文件路径')
    parser.add_argument('--base_dir', type=str,
                       default=None,
                       help='基础目录，用于计算相对路径')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用CPU（默认使用GPU）')
    parser.add_argument('--dinov2_model', type=str,
                       default='dinov2_vitb14',
                       help='DINOv2模型名称（默认dinov2_vitb14）')
    
    args = parser.parse_args()
    
    filter_valid_images_3d(
        video_dir=args.video_dir,
        face_dir=args.face_dir,
        output_file=args.output_file,
        use_cpu=args.use_cpu,
        dinov2_model_name=args.dinov2_model,
        base_dir=args.base_dir
    )
