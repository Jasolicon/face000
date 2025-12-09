"""
预处理视频：将视频拆分成帧
在训练前运行此脚本，将视频转换为帧图片，加快训练速度
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录和train目录到路径
train_dir = Path(__file__).parent
project_root = train_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(train_dir))

# 使用相对导入（从train目录内运行）
from utils import batch_extract_videos, extract_video_frames


def main():
    parser = argparse.ArgumentParser(description='预处理视频：将视频拆分成帧')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--frame_interval', type=int, default=1, help='帧间隔（每隔多少帧提取一帧）')
    parser.add_argument('--max_frames', type=int, default=None, help='每个视频最大提取帧数')
    parser.add_argument('--image_format', type=str, default='jpg', choices=['jpg', 'png'], help='图像格式')
    parser.add_argument('--force_rebuild', action='store_true', help='强制重新提取（即使文件夹已存在）')
    parser.add_argument('--single_video', type=str, default=None, help='处理单个视频文件（可选）')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("视频预处理：将视频拆分成帧")
    print("=" * 70)
    
    if args.single_video:
        # 处理单个视频
        video_path = Path(args.single_video)
        if not video_path.exists():
            print(f"错误: 视频文件不存在: {video_path}")
            return
        
        print(f"\n处理单个视频: {video_path.name}")
        try:
            frame_paths = extract_video_frames(
                video_path=str(video_path),
                output_dir=None,  # 使用视频文件名作为文件夹名
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
                image_format=args.image_format
            )
            print(f"✓ 完成！已保存 {len(frame_paths)} 帧")
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 批量处理
        print(f"\n数据目录: {args.data_dir}")
        print(f"帧间隔: {args.frame_interval}")
        print(f"最大帧数: {args.max_frames or '无限制'}")
        print(f"图像格式: {args.image_format}")
        print(f"强制重建: {args.force_rebuild}")
        print()
        
        results = batch_extract_videos(
            data_dir=args.data_dir,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            image_format=args.image_format,
            force_rebuild=args.force_rebuild
        )
        
        # 统计信息
        total_videos = len(results)
        success_videos = len([v for v in results.values() if v])
        total_frames = sum(len(frames) for frames in results.values())
        
        print("\n" + "=" * 70)
        print("处理统计:")
        print(f"  总视频数: {total_videos}")
        print(f"  成功处理: {success_videos}")
        print(f"  总帧数: {total_frames}")
        print("=" * 70)


if __name__ == '__main__':
    main()

