"""
处理视频目录：将视频拆成帧，提取特征和关键点，保存到数据库格式
文件名就是人名
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import logging
from PIL import Image

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from feature_manager import FeatureManager
from feature_extractor import DINOv2FeatureExtractor
from utils import (
    get_insightface_detector, 
    get_insightface_landmarks,
    landmarks_to_3d,
    calculate_spherical_angle
)
from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path, output_dir=None, frame_interval=1, max_frames=None):
    """
    从视频中提取帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录（如果为None，则在视频同目录下创建frames文件夹）
        frame_interval: 帧间隔（每隔多少帧提取一帧）
        max_frames: 最大提取帧数
    
    Returns:
        frame_paths: 提取的帧图片路径列表
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_frames"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"视频信息: {video_path.name}")
    logger.info(f"  总帧数: {total_frames}, 帧率: {fps:.2f} fps, 分辨率: {width}x{height}")
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据间隔提取帧
        if frame_count % frame_interval == 0:
            # 保存帧
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = output_dir / frame_filename
            
            # 使用PIL保存（支持中文路径）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save(frame_path, quality=95)
            
            frame_paths.append(frame_path)
            extracted_count += 1
            
            # 如果达到最大帧数，停止
            if max_frames is not None and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    logger.info(f"✓ 已提取 {len(frame_paths)} 帧到 {output_dir}")
    return frame_paths


def process_video_directory(
    video_dir,
    output_base_dir='C:\\Codes\\face000\\train\\datas',
    output_features_dir=None,
    frame_interval=1,
    max_frames_per_video=None,
    use_cpu=False,
    dinov2_model_name='dinov2_vitb14',
    save_frames=True,
    save_frontal_face=True,
    save_features=True,
    save_only_frontal_features=False
):
    """
    处理视频目录：将视频拆成帧，提取特征和关键点，保存到数据库格式
    
    Args:
        video_dir: 视频目录路径（文件名就是人名）
        output_base_dir: 输出基础目录（face和video的父目录）
        output_features_dir: 特征存储目录
        frame_interval: 帧间隔（每隔多少帧提取一帧）
        max_frames_per_video: 每个视频最大提取帧数
        use_cpu: 是否使用CPU
        dinov2_model_name: DINOv2模型名称
        save_frames: 是否保存帧图片（如果为False，只提取特征和关键点，不保存帧）
        save_frontal_face: 是否保存正脸图片到face目录
        save_features: 是否保存特征到数据库（默认True）
        save_only_frontal_features: 是否只保存正脸特征（不保存视频帧特征，避免污染特征库）
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {video_dir}")
    
    # 创建输出目录结构
    output_base_dir = Path(output_base_dir)
    face_dir = output_base_dir / 'face'
    video_dir_output = output_base_dir / 'video'
    face_dir.mkdir(parents=True, exist_ok=True)
    video_dir_output.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("处理视频目录：提取特征和关键点")
    logger.info("=" * 70)
    logger.info(f"视频目录: {video_dir}")
    logger.info(f"输出基础目录: {output_base_dir}")
    logger.info(f"  正脸目录: {face_dir}")
    logger.info(f"  视频帧目录: {video_dir_output}")
    if output_features_dir:
        logger.info(f"特征存储目录: {output_features_dir}")
    else:
        logger.info(f"特征存储: 不保存特征")
    logger.info(f"帧间隔: {frame_interval}")
    logger.info(f"最大帧数/视频: {max_frames_per_video or '无限制'}")
    logger.info(f"只保存正脸特征: {save_only_frontal_features}")
    
    # 初始化组件
    logger.info("\n初始化组件...")
    logger.info("  初始化InsightFace检测器...")
    detector = get_insightface_detector(use_cpu=use_cpu)
    
    feature_extractor = None
    feature_manager = None
    
    if save_features:
        logger.info(f"  初始化DINOv2特征提取器（模型: {dinov2_model_name}）...")
        device = 'cpu' if use_cpu else None
        feature_extractor = DINOv2FeatureExtractor(
            model_name=dinov2_model_name,
            resize_to_96=False,
            device=device
        )
        
        if output_features_dir:
            logger.info("  初始化特征管理器...")
            feature_manager = FeatureManager(storage_dir=output_features_dir)
        else:
            logger.warning("  警告: save_features=True 但未指定 output_features_dir，将不保存特征")
            save_features = False
    
    # 查找所有视频文件（包括MTS格式）
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mts', '.m2ts']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'*{ext}'))
        video_files.extend(video_dir.glob(f'*{ext.upper()}'))
    
    if len(video_files) == 0:
        logger.warning(f"未找到视频文件（支持的格式: {', '.join(video_extensions)}）")
        return
    
    logger.info(f"\n找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    total_features = 0
    success_videos = 0
    
    for video_file in tqdm(video_files, desc="处理视频"):
        person_name = video_file.stem  # 文件名（不含扩展名）就是人名
        logger.info(f"\n处理视频: {video_file.name} (人名: {person_name})")
        
        try:
            # 创建视频帧输出目录
            person_video_dir = video_dir_output / person_name
            person_video_dir.mkdir(parents=True, exist_ok=True)
            
            # 提取帧
            if save_frames:
                # 直接保存到video目录
                frame_paths = extract_frames_from_video(
                    video_file,
                    output_dir=person_video_dir,
                    frame_interval=frame_interval,
                    max_frames=max_frames_per_video
                )
            else:
                # 不保存帧，直接处理视频（使用临时文件）
                frame_paths = []
                cap = cv2.VideoCapture(str(video_file))
                frame_count = 0
                extracted_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        # 临时保存帧到video目录
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # 保存到临时文件（在video目录中）
                        temp_frame_path = person_video_dir / f"temp_{person_name}_frame_{frame_count:06d}.jpg"
                        pil_image.save(temp_frame_path, quality=95)
                        frame_paths.append(temp_frame_path)
                        
                        extracted_count += 1
                        if max_frames_per_video is not None and extracted_count >= max_frames_per_video:
                            break
                    
                    frame_count += 1
                
                cap.release()
            
            if len(frame_paths) == 0:
                logger.warning(f"  未提取到帧，跳过")
                continue
            
            logger.info(f"  提取到 {len(frame_paths)} 帧")
            
            # 处理每一帧，同时寻找最正脸的帧
            valid_frames = 0
            frontal_frame_info = None  # (frame_path, avg_angle, landmarks, box)
            min_avg_angle = float('inf')
            
            for frame_path in tqdm(frame_paths, desc=f"  处理 {person_name} 的帧", leave=False):
                try:
                    # 检测关键点
                    landmarks, box = get_insightface_landmarks(detector, str(frame_path))
                    if landmarks is None or box is None:
                        logger.debug(f"    无法检测到关键点: {frame_path.name}")
                        continue
                    
                    # 获取图片尺寸
                    img = Image.open(frame_path)
                    img_width, img_height = img.size
                    
                    # 转换为3D坐标
                    landmarks_3d = landmarks_to_3d(landmarks, box, img_width, img_height)
                    
                    # 计算角度（相对于标准正面，使用第一个有效帧作为参考）
                    if frontal_frame_info is None:
                        # 第一个有效帧作为参考
                        reference_landmarks_3d = landmarks_3d.copy()
                        reference_landmarks_2d = landmarks.copy()
                        reference_box = box.copy()
                        reference_img_size = (img_width, img_height)
                        avg_angle = 0.0  # 第一个帧作为参考，角度为0
                    else:
                        # 计算与参考的角度
                        angles, avg_angle = calculate_spherical_angle(
                            reference_landmarks_3d,
                            landmarks_3d,
                            reference_landmarks_2d,
                            landmarks
                        )
                        avg_angle = abs(avg_angle) if avg_angle is not None else float('inf')
                    
                    # 更新最正脸的帧（角度最小的）
                    if avg_angle < min_avg_angle:
                        min_avg_angle = avg_angle
                        frontal_frame_info = (frame_path, avg_angle, landmarks, box)
                    
                    # 如果帧不在video目录，移动到video目录
                    if save_frames and not str(frame_path).startswith(str(person_video_dir)):
                        frame_filename = frame_path.name
                        target_frame_path = person_video_dir / frame_filename
                        import shutil
                        shutil.move(frame_path, target_frame_path)
                        frame_path = target_frame_path
                    
                    # 提取特征（如果需要）
                    if save_features and not save_only_frontal_features:
                        features = feature_extractor.extract_features(str(frame_path))
                        
                        # 保存特征到数据库（使用相对路径）
                        relative_path = frame_path.relative_to(output_base_dir) if output_base_dir.exists() else str(frame_path)
                        feature_manager.save_feature(
                            feature_vector=features,
                            image_path=str(relative_path).replace('\\', '/'),  # 统一使用正斜杠
                            person_name=person_name
                        )
                    
                    valid_frames += 1
                    total_features += 1
                    
                except Exception as e:
                    logger.warning(f"    处理帧失败 {frame_path.name}: {e}")
                    continue
            
            # 保存最正脸的帧到face目录
            if save_frontal_face and frontal_frame_info is not None:
                frontal_frame_path, frontal_angle, frontal_landmarks, frontal_box = frontal_frame_info
                frontal_face_path = face_dir / f"{person_name}.jpg"
                
                try:
                    # 复制最正脸的帧到face目录
                    import shutil
                    shutil.copy2(frontal_frame_path, frontal_face_path)
                    logger.info(f"  ✓ 保存正脸图片: {frontal_face_path} (角度: {frontal_angle:.2f}°)")
                    
                    # 保存正脸的特征到数据库（如果启用）
                    if save_features and feature_manager is not None:
                        frontal_features = feature_extractor.extract_features(str(frontal_face_path))
                        relative_path = frontal_face_path.relative_to(output_base_dir) if output_base_dir.exists() else str(frontal_face_path)
                        feature_manager.save_feature(
                            feature_vector=frontal_features,
                            image_path=str(relative_path).replace('\\', '/'),
                            person_name=person_name
                        )
                        total_features += 1
                except Exception as e:
                    logger.warning(f"  保存正脸图片失败: {e}")
            
            # 如果不保存帧，删除临时文件
            if not save_frames:
                for frame_path in frame_paths:
                    if frame_path.exists() and 'temp_' in str(frame_path):
                        frame_path.unlink()
            
            
            logger.info(f"  ✓ {person_name}: 成功处理 {valid_frames}/{len(frame_paths)} 帧")
            success_videos += 1
            
        except Exception as e:
            logger.error(f"  处理视频失败 {video_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 统计信息
    logger.info("\n" + "=" * 70)
    logger.info("处理完成！")
    logger.info(f"  总视频数: {len(video_files)}")
    logger.info(f"  成功处理: {success_videos}")
    logger.info(f"  总特征数: {total_features}")
    logger.info(f"  特征存储目录: {output_features_dir}")
    logger.info("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='处理视频目录：提取特征和关键点，保存到数据库格式')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='视频目录路径（文件名就是人名）')
    parser.add_argument('--output_base_dir', type=str, default='C:\\Codes\\face000\\train\\datas',
                       help='输出基础目录（face和video的父目录，默认: C:\\Codes\\face000\\train\\datas）')
    parser.add_argument('--output_features_dir', type=str, default=None,
                       help='特征存储目录（默认: None，不保存特征。如果指定，会保存特征到该目录）')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='帧间隔（每隔多少帧提取一帧，默认1表示每帧都提取）')
    parser.add_argument('--max_frames_per_video', type=int, default=None,
                       help='每个视频最大提取帧数（默认无限制）')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用CPU（默认使用GPU）')
    parser.add_argument('--dinov2_model', type=str, default='dinov2_vitb14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2模型名称（默认: dinov2_vitb14，768维）')
    parser.add_argument('--no_save_frames', action='store_true',
                       help='不保存帧图片（只提取特征和关键点，节省空间）')
    parser.add_argument('--no_save_frontal', action='store_true',
                       help='不保存正脸图片到face目录')
    parser.add_argument('--no_save_features', action='store_true',
                       help='不保存特征到数据库（默认会保存）')
    parser.add_argument('--save_only_frontal_features', action='store_true',
                       help='只保存正脸特征，不保存视频帧特征（避免污染特征库）')
    
    args = parser.parse_args()
    
    process_video_directory(
        video_dir=args.video_dir,
        output_base_dir=args.output_base_dir,
        output_features_dir=args.output_features_dir,
        frame_interval=args.frame_interval,
        max_frames_per_video=args.max_frames_per_video,
        use_cpu=args.use_cpu,
        dinov2_model_name=args.dinov2_model,
        save_frames=not args.no_save_frames,
        save_frontal_face=not args.no_save_frontal,
        save_features=not args.no_save_features,
        save_only_frontal_features=args.save_only_frontal_features
    )


if __name__ == '__main__':
    main()

