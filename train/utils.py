"""
训练工具函数
"""
import cv2
import os
from pathlib import Path
from typing import Optional, List
import numpy as np
from tqdm import tqdm


def extract_video_frames(
    video_path: str,
    output_dir: Optional[str] = None,
    frame_interval: int = 1,
    max_frames: Optional[int] = None,
    image_format: str = 'jpg',
    prefix: str = 'frame'
) -> List[str]:
    """
    将视频拆分成帧并保存到文件夹
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录，如果为None则使用视频文件名（不含扩展名）作为文件夹名
        frame_interval: 帧间隔（每隔多少帧提取一帧，1表示每帧都提取）
        max_frames: 最大提取帧数，如果为None则提取所有帧
        image_format: 图像格式（jpg, png等）
        prefix: 帧文件名前缀（建议使用ASCII字符，如'frame'，避免中文乱码）
        
    Returns:
        保存的帧图片路径列表
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 确定输出目录
    if output_dir is None:
        # 使用视频文件名（不含扩展名）作为文件夹名
        video_stem = video_path.stem
        output_dir = video_path.parent / video_stem
    else:
        output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {video_path.name}")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  分辨率: {width}x{height}")
    print(f"  输出目录: {output_dir}")
    
    # 计算要提取的帧
    frames_to_extract = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据间隔提取帧
        if frame_count % frame_interval == 0:
            frames_to_extract.append((frame_count, frame))
            extracted_count += 1
            
            # 如果达到最大帧数，停止
            if max_frames is not None and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    # 保存帧
    saved_paths = []
    print(f"正在保存 {len(frames_to_extract)} 帧...")
    
    for idx, (frame_num, frame) in enumerate(tqdm(frames_to_extract, desc="提取帧")):
        # 生成文件名（确保prefix只包含ASCII字符，避免中文乱码）
        # 如果prefix包含非ASCII字符，自动使用'frame'作为前缀
        safe_prefix = prefix
        try:
            # 检查是否包含非ASCII字符
            prefix.encode('ascii')
        except (UnicodeEncodeError, AttributeError):
            # 如果包含非ASCII字符，直接使用'frame'作为前缀
            safe_prefix = 'frame'
            if prefix != 'frame':
                print(f"警告: prefix '{prefix}' 包含非ASCII字符，已自动使用 'frame' 作为前缀")
        
        frame_filename = f"{safe_prefix}_{frame_num:06d}.{image_format}"
        frame_path = output_dir / frame_filename
        
        # 保存图像（确保路径使用正确的编码）
        try:
            # 尝试使用cv2.imwrite保存
            # 注意：cv2.imwrite在某些系统上可能无法正确处理中文路径
            frame_path_str = str(frame_path)
            success = cv2.imwrite(frame_path_str, frame)
            
            if not success:
                # 如果cv2.imwrite失败，尝试使用PIL（PIL对中文路径支持更好）
                from PIL import Image
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.save(frame_path_str, quality=95, format=image_format.upper())
        except Exception as e:
            print(f"保存帧失败 {frame_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        saved_paths.append(str(frame_path))
    
    print(f"✓ 完成！已保存 {len(saved_paths)} 帧到 {output_dir}")
    
    return saved_paths


def batch_extract_videos(
    data_dir: str,
    frame_interval: int = 1,
    max_frames: Optional[int] = None,
    image_format: str = 'jpg',
    force_rebuild: bool = False
) -> dict:
    """
    批量处理视频，将视频拆分成帧
    
    Args:
        data_dir: 数据目录
        frame_interval: 帧间隔
        max_frames: 每个视频最大提取帧数
        image_format: 图像格式
        force_rebuild: 是否强制重新提取（即使文件夹已存在）
        
    Returns:
        处理结果字典 {视频路径: [帧路径列表]}
    """
    data_dir = Path(data_dir)
    results = {}
    
    # 支持的视频格式
    video_exts = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    
    # 查找所有视频文件
    video_files = []
    for ext in video_exts:
        video_files.extend(list(data_dir.glob(f'*{ext}')))
        # 也查找子目录中的视频
        video_files.extend(list(data_dir.glob(f'*/*{ext}')))
    
    if len(video_files) == 0:
        print(f"警告: 在 {data_dir} 中未找到视频文件")
        return results
    
    print(f"找到 {len(video_files)} 个视频文件")
    print("=" * 70)
    
    for video_path in video_files:
        # 确定输出目录（使用视频文件名作为文件夹名）
        video_stem = video_path.stem
        output_dir = video_path.parent / video_stem
        
        # 检查是否已存在
        if output_dir.exists() and not force_rebuild:
            frame_files = list(output_dir.glob(f'*.{image_format}'))
            if len(frame_files) > 0:
                print(f"跳过 {video_path.name} (已存在 {len(frame_files)} 帧)")
                results[str(video_path)] = [str(f) for f in frame_files]
                continue
        
        try:
            # 提取帧
            frame_paths = extract_video_frames(
                video_path=str(video_path),
                output_dir=str(output_dir),
                frame_interval=frame_interval,
                max_frames=max_frames,
                image_format=image_format
            )
            results[str(video_path)] = frame_paths
        except Exception as e:
            print(f"✗ 处理失败 {video_path.name}: {e}")
            results[str(video_path)] = []
    
    print("=" * 70)
    print(f"批量处理完成！成功处理 {len([v for v in results.values() if v])} 个视频")
    
    return results


def get_video_frames_dir(video_path: str) -> Optional[Path]:
    """
    获取视频对应的帧文件夹路径
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        帧文件夹路径，如果不存在则返回None
    """
    video_path = Path(video_path)
    frames_dir = video_path.parent / video_path.stem
    
    if frames_dir.exists() and frames_dir.is_dir():
        return frames_dir
    
    return None

