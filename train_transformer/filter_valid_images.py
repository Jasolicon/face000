"""
筛选能提取到关键点和DINOv2特征的图片
保存特征、关键点和路径到JSON文件，供dataset使用
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

from utils import get_insightface_detector, get_insightface_landmarks, landmarks_to_3d, calculate_spherical_angle
from feature_extractor import DINOv2FeatureExtractor
from train_transformer.utils_seed import set_seed
from PIL import Image

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_valid_images(
    video_dir: str,
    face_dir: str,
    output_file: str = 'train_transformer/valid_images.json',
    use_cpu: bool = False,
    dinov2_model_name: str = 'dinov2_vitb14',
    base_dir: str = None
):
    """
    筛选能提取到关键点和DINOv2特征的图片，并保存特征和关键点到JSON
    
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
    device = 'cpu' if use_cpu else None
    feature_extractor = DINOv2FeatureExtractor(
        model_name=dinov2_model_name,
        resize_to_96=False,
        device=device
    )
    
    # 获取所有正面图片的人名
    face_names = set()
    face_images = {}
    for face_file in face_dir.glob('*.jpg'):
        person_name = face_file.stem
        face_names.add(person_name)
        face_images[person_name] = face_file
    
    logger.info(f"找到正面图片人名: {face_names}")
    
    # 存储有效数据（包含特征和关键点）
    valid_data = {}
    
    # 遍历每个人
    for person_name in face_names:
        logger.info(f"\n处理 {person_name}...")
        
        # 检查正面图是否能检测到关键点并提取特征
        face_image_path = face_images[person_name]
        
        # 检测关键点
        face_landmarks, face_box = get_insightface_landmarks(detector, str(face_image_path))
        if face_landmarks is None or face_box is None:
            logger.warning(f"正面图无法检测到关键点: {face_image_path}")
            continue
        
        # 提取DINOv2特征
        try:
            face_features = feature_extractor.extract_features(str(face_image_path))
            logger.info(f"  ✓ 正面图特征提取成功，维度: {len(face_features)}")
        except Exception as e:
            logger.warning(f"正面图特征提取失败: {face_image_path}, 错误: {e}")
            continue
        
        # 获取图片尺寸（用于计算球面角）
        face_img = Image.open(face_image_path)
        face_width, face_height = face_img.size
        
        # 转换为3D坐标
        face_landmarks_3d = landmarks_to_3d(face_landmarks, face_box, face_width, face_height)
        
        # 计算相对路径
        face_image_path_rel = face_image_path.relative_to(base_dir) if base_dir else face_image_path
        face_image_path_str = str(face_image_path_rel).replace('\\', '/')  # 统一使用正斜杠
        
        # 获取视频帧图片目录
        video_person_dir = video_dir / person_name
        if not video_person_dir.exists():
            logger.warning(f"视频目录不存在: {video_person_dir}")
            continue
        
        # 获取所有视频帧图片
        video_images = sorted(list(video_person_dir.glob('*.jpg')))
        logger.info(f"找到 {len(video_images)} 张视频帧图片")
        
        # 筛选能检测到关键点并提取特征的图片
        valid_video_data = []
        for video_image_path in tqdm(video_images, desc=f"处理 {person_name}"):
            # 检测关键点
            video_landmarks, video_box = get_insightface_landmarks(detector, str(video_image_path))
            if video_landmarks is None or video_box is None:
                continue
            
            # 提取DINOv2特征
            try:
                video_features = feature_extractor.extract_features(str(video_image_path))
            except Exception as e:
                logger.debug(f"特征提取失败: {video_image_path}, 错误: {e}")
                continue
            
            # 获取图片尺寸
            video_img = Image.open(video_image_path)
            video_width, video_height = video_img.size
            
            # 转换为3D坐标
            video_landmarks_3d = landmarks_to_3d(video_landmarks, video_box, video_width, video_height)
            
            # 计算球面角
            angles, avg_angle = calculate_spherical_angle(
                face_landmarks_3d,
                video_landmarks_3d,
                face_landmarks,
                video_landmarks
            )
            
            # 计算相对路径
            video_image_path_rel = video_image_path.relative_to(base_dir) if base_dir else video_image_path
            video_image_path_str = str(video_image_path_rel).replace('\\', '/')  # 统一使用正斜杠
            
            # 保存数据（特征转换为列表以便JSON序列化）
            valid_video_data.append({
                'image_path': video_image_path_str,
                'features': video_features.tolist(),  # numpy数组转列表
                'landmarks_2d': video_landmarks.tolist(),  # [5, 2]
                'landmarks_3d': video_landmarks_3d.tolist(),  # [5, 3]
                'box': video_box.tolist(),  # [4]
                'spherical_angles': angles.tolist(),  # [5]
                'avg_angle': float(avg_angle) if avg_angle is not None else 0.0
            })
        
        logger.info(f"{person_name}: {len(valid_video_data)}/{len(video_images)} 张图片有效")
        
        if len(valid_video_data) > 0:
            valid_data[person_name] = {
                'face_image_path': face_image_path_str,
                'face_features': face_features.tolist(),  # numpy数组转列表
                'face_landmarks_2d': face_landmarks.tolist(),  # [5, 2]
                'face_landmarks_3d': face_landmarks_3d.tolist(),  # [5, 3]
                'face_box': face_box.tolist(),  # [4]
                'video_data': valid_video_data,
                'total_video_images': len(video_images),
                'valid_video_images': len(valid_video_data),
                'feature_dim': len(face_features),  # 特征维度
                'dinov2_model': dinov2_model_name  # 使用的模型
            }
    
    # 保存到文件
    logger.info(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    total_persons = len(valid_data)
    total_valid_images = sum(info['valid_video_images'] for info in valid_data.values())
    total_video_images = sum(info['total_video_images'] for info in valid_data.values())
    
    # 检查特征维度一致性
    feature_dims = [info['feature_dim'] for info in valid_data.values()]
    if len(set(feature_dims)) > 1:
        logger.warning(f"⚠️  检测到不同的特征维度: {set(feature_dims)}")
    else:
        logger.info(f"✓ 所有特征维度一致: {feature_dims[0] if feature_dims else 'N/A'}")
    
    logger.info("\n" + "=" * 70)
    logger.info("筛选完成！")
    logger.info("=" * 70)
    logger.info(f"总人数: {total_persons}")
    logger.info(f"总视频帧图片: {total_video_images}")
    logger.info(f"有效视频帧图片: {total_valid_images}")
    logger.info(f"有效率: {total_valid_images/total_video_images*100:.2f}%" if total_video_images > 0 else "N/A")
    logger.info(f"使用的DINOv2模型: {dinov2_model_name}")
    logger.info(f"特征维度: {feature_dims[0] if feature_dims else 'N/A'}")
    logger.info(f"结果已保存到: {output_file}")
    logger.info(f"基础目录（相对路径基准）: {base_dir}")
    
    return valid_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='筛选能提取到关键点和DINOv2特征的图片')
    parser.add_argument('--video_dir', type=str,
                       default=r'/root/face000/train/datas/video',
                       help='视频帧图片目录')
    parser.add_argument('--face_dir', type=str,
                       default=r'/root/face000/train/datas/face',
                       help='正面图片目录')
    parser.add_argument('--output_file', type=str,
                       default='train_transformer/valid_images.json',
                       help='输出文件路径')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='基础目录，用于计算相对路径（默认使用output_file的父目录）')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用CPU')
    parser.add_argument('--dinov2_model', type=str, default='dinov2_vitb14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2模型名称（默认: dinov2_vitb14，768维）')
    
    args = parser.parse_args()
    
    # 如果base_dir未指定，使用output_file的父目录
    if args.base_dir is None:
        output_file_path = Path(args.output_file)
        args.base_dir = str(output_file_path.parent)
    
    filter_valid_images(
        video_dir=args.video_dir,
        face_dir=args.face_dir,
        output_file=args.output_file,
        use_cpu=args.use_cpu,
        dinov2_model_name=args.dinov2_model,
        base_dir=args.base_dir
    )

