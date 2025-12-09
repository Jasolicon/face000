"""
筛选能提取到关键点的图片
保存图片路径到文件，供dataset使用
"""
import sys
from pathlib import Path
import json
from tqdm import tqdm
import logging

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils import get_insightface_detector, get_insightface_landmarks
from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def filter_valid_images(
    video_dir: str,
    face_dir: str,
    output_file: str = 'train_transformer/valid_images.json',
    use_cpu: bool = False
):
    """
    筛选能提取到关键点的图片
    
    Args:
        video_dir: 视频帧图片目录
        face_dir: 正面图片目录
        output_file: 输出文件路径
        use_cpu: 是否使用CPU
    """
    video_dir = Path(video_dir)
    face_dir = Path(face_dir)
    output_file = Path(output_file)
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化检测器
    logger.info("初始化InsightFace检测器...")
    detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 获取所有正面图片的人名
    face_names = set()
    face_images = {}
    for face_file in face_dir.glob('*.jpg'):
        person_name = face_file.stem
        face_names.add(person_name)
        face_images[person_name] = str(face_file)
    
    logger.info(f"找到正面图片人名: {face_names}")
    
    # 存储有效图片路径
    valid_images = {}
    
    # 遍历每个人
    for person_name in face_names:
        logger.info(f"\n处理 {person_name}...")
        
        # 检查正面图是否能检测到关键点
        face_image_path = face_images[person_name]
        face_landmarks, face_box = get_insightface_landmarks(detector, face_image_path)
        
        if face_landmarks is None or face_box is None:
            logger.warning(f"正面图无法检测到关键点: {face_image_path}")
            continue
        
        # 获取视频帧图片目录
        video_person_dir = video_dir / person_name
        if not video_person_dir.exists():
            logger.warning(f"视频目录不存在: {video_person_dir}")
            continue
        
        # 获取所有视频帧图片
        video_images = sorted(list(video_person_dir.glob('*.jpg')))
        logger.info(f"找到 {len(video_images)} 张视频帧图片")
        
        # 筛选能检测到关键点的图片
        valid_video_images = []
        for video_image_path in tqdm(video_images, desc=f"检测 {person_name}"):
            video_landmarks, video_box = get_insightface_landmarks(detector, str(video_image_path))
            
            if video_landmarks is not None and video_box is not None:
                valid_video_images.append(str(video_image_path))
        
        logger.info(f"{person_name}: {len(valid_video_images)}/{len(video_images)} 张图片有效")
        
        if len(valid_video_images) > 0:
            valid_images[person_name] = {
                'face_image_path': face_image_path,
                'video_images': valid_video_images,
                'total_video_images': len(video_images),
                'valid_video_images': len(valid_video_images)
            }
    
    # 保存到文件
    logger.info(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(valid_images, f, indent=2, ensure_ascii=False)
    
    # 统计信息
    total_persons = len(valid_images)
    total_valid_images = sum(info['valid_video_images'] for info in valid_images.values())
    total_video_images = sum(info['total_video_images'] for info in valid_images.values())
    
    logger.info("\n" + "=" * 70)
    logger.info("筛选完成！")
    logger.info("=" * 70)
    logger.info(f"总人数: {total_persons}")
    logger.info(f"总视频帧图片: {total_video_images}")
    logger.info(f"有效视频帧图片: {total_valid_images}")
    logger.info(f"有效率: {total_valid_images/total_video_images*100:.2f}%" if total_video_images > 0 else "N/A")
    logger.info(f"结果已保存到: {output_file}")
    
    return valid_images


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='筛选能提取到关键点的图片')
    parser.add_argument('--video_dir', type=str,
                       default=r'C:\Codes\face000\train\datas\video',
                       help='视频帧图片目录')
    parser.add_argument('--face_dir', type=str,
                       default=r'C:\Codes\face000\train\datas\face',
                       help='正面图片目录')
    parser.add_argument('--output_file', type=str,
                       default='train_transformer/valid_images.json',
                       help='输出文件路径')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用CPU')
    
    args = parser.parse_args()
    
    filter_valid_images(
        video_dir=args.video_dir,
        face_dir=args.face_dir,
        output_file=args.output_file,
        use_cpu=args.use_cpu
    )

