"""
在图片上绘制关键点并保存
从 valid_images_normalized.json 读取关键点数据，绘制到对应图片上
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
from tqdm import tqdm

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))

# 关键点名称（InsightFace 5点关键点）
LANDMARK_NAMES = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角']
LANDMARK_COLORS = ['red', 'blue', 'green', 'yellow', 'orange']


def draw_landmarks_on_image(image_path: str, landmarks_2d: list, box: list = None, 
                           output_path: str = None, person_name: str = None):
    """
    在图片上绘制关键点
    
    Args:
        image_path: 图片路径
        landmarks_2d: 关键点坐标 [[x1, y1], [x2, y2], ...] 或 [[x1, y1, x2, y2, ...]]
        box: 边界框 [x1, y1, x2, y2]（可选）
        output_path: 输出路径（可选，默认覆盖原图）
        person_name: 人名（用于显示）
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # 转换关键点格式
        if isinstance(landmarks_2d, list):
            landmarks = np.array(landmarks_2d)
        else:
            landmarks = landmarks_2d
        
        # 如果是扁平数组，reshape为 [5, 2]
        if landmarks.ndim == 1:
            landmarks = landmarks.reshape(-1, 2)
        
        # 确保是 [5, 2] 格式
        if landmarks.shape[0] != 5:
            print(f"警告: 关键点数量不是5个，实际是 {landmarks.shape[0]} 个")
            return False
        
        # 绘制边界框（如果有）
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='cyan', width=2)
        
        # 绘制关键点
        point_radius = max(3, min(img.width, img.height) // 100)  # 根据图片大小调整点的大小
        
        for i, landmark in enumerate(landmarks):
            # 获取坐标
            if len(landmark) >= 2:
                x, y = float(landmark[0]), float(landmark[1])
            else:
                continue
            
            # 绘制点
            x, y = int(x), int(y)
            
            # 确保坐标在图片范围内
            if x < 0 or x >= img.width or y < 0 or y >= img.height:
                continue
            
            # 绘制大圆（填充）
            color_map = {
                'red': (255, 0, 0),
                'blue': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (255, 255, 0),
                'orange': (255, 165, 0)
            }
            color = color_map.get(LANDMARK_COLORS[i % len(LANDMARK_COLORS)], (255, 0, 0))
            
            draw.ellipse(
                [x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                fill=color,
                outline='white',
                width=1
            )
            
            # 绘制标签（可选，如果图片不太小）
            if img.width > 200 and img.height > 200:
                try:
                    # 尝试使用默认字体
                    font = ImageFont.load_default()
                    label = f"{i+1}"  # 显示序号
                    # 计算文本位置（在点的右上方）
                    text_x = x + point_radius + 2
                    text_y = y - point_radius - 2
                    # 绘制文本背景
                    bbox = draw.textbbox((text_x, text_y), label, font=font)
                    draw.rectangle(bbox, fill='black', outline='white', width=1)
                    draw.text((text_x, text_y), label, fill='white', font=font)
                except:
                    pass
        
        # 添加标题（如果有）
        if person_name:
            try:
                font = ImageFont.load_default()
                title = f"{person_name}"
                # 在图片顶部绘制标题背景
                bbox = draw.textbbox((10, 10), title, font=font)
                draw.rectangle(
                    [bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5],
                    fill='black',
                    outline='white',
                    width=2
                )
                draw.text((10, 10), title, fill='white', font=font)
            except:
                pass
        
        # 保存图片
        if output_path is None:
            output_path = image_path
        
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"处理图片失败 {image_path}: {e}")
        return False


def visualize_landmarks_from_json(
    json_file: str = 'train_transformer/valid_images_normalized.json',
    base_dir: str = None,
    output_dir: str = 'landmark_visualizations',
    max_persons: int = None,
    max_video_frames: int = 5
):
    """
    从JSON文件读取关键点数据，绘制到图片上
    
    Args:
        json_file: JSON文件路径
        base_dir: 基础目录（用于解析相对路径）
        output_dir: 输出目录
        max_persons: 最多处理的人数（None表示全部）
        max_video_frames: 每个人最多处理的视频帧数
    """
    # 读取JSON文件
    print(f"读取JSON文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确定基础目录
    if base_dir is None:
        base_dir = Path(json_file).parent.parent
    else:
        base_dir = Path(base_dir)
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"基础目录: {base_dir}")
    print(f"输出目录: {output_dir}")
    print(f"找到 {len(data)} 个人")
    print(f"JSON文件路径: {Path(json_file).absolute()}")
    
    # 统计
    total_faces = 0
    total_video_frames = 0
    success_faces = 0
    success_video_frames = 0
    
    # 处理每个人
    person_names = list(data.keys())
    if max_persons:
        person_names = person_names[:max_persons]
    
    for person_name in tqdm(person_names, desc="处理人员"):
        person_data = data[person_name]
        
        # 处理正面图
        if 'face_image_path' in person_data and 'face_landmarks_2d' in person_data:
            face_path = base_dir / person_data['face_image_path']
            landmarks_2d = person_data['face_landmarks_2d']
            face_box = person_data.get('face_box', None)
            
            if face_path.exists():
                # 输出路径
                output_face_path = output_dir / f"{person_name}_face.jpg"
                
                if draw_landmarks_on_image(
                    str(face_path),
                    landmarks_2d,
                    face_box,
                    str(output_face_path),
                    person_name=f"{person_name}_正面"
                ):
                    success_faces += 1
                total_faces += 1
            else:
                print(f"警告: 正面图不存在: {face_path}")
        
        # 处理视频帧
        if 'video_data' in person_data:
            video_data = person_data['video_data']
            
            # 限制处理的视频帧数
            if max_video_frames:
                video_data = video_data[:max_video_frames]
            
            for idx, frame_data in enumerate(video_data):
                if 'image_path' in frame_data and 'landmarks_2d' in frame_data:
                    frame_path = base_dir / frame_data['image_path']
                    landmarks_2d = frame_data['landmarks_2d']
                    frame_box = frame_data.get('box', None)
                    
                    if frame_path.exists():
                        # 输出路径
                        output_frame_path = output_dir / f"{person_name}_video_{idx:03d}.jpg"
                        
                        if draw_landmarks_on_image(
                            str(frame_path),
                            landmarks_2d,
                            frame_box,
                            str(output_frame_path),
                            person_name=f"{person_name}_视频帧{idx}"
                        ):
                            success_video_frames += 1
                        total_video_frames += 1
    
    # 打印统计信息
    print("\n" + "="*70)
    print("处理完成！")
    print("="*70)
    print(f"正面图: {success_faces}/{total_faces} 成功")
    print(f"视频帧: {success_video_frames}/{total_video_frames} 成功")
    print(f"输出目录: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='在图片上绘制关键点')
    parser.add_argument('--json_file', type=str,
                       default='train_transformer/valid_images_normalized.json',
                       help='JSON文件路径')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='基础目录（用于解析相对路径，默认使用JSON文件父目录的父目录）')
    parser.add_argument('--output_dir', type=str, default='landmark_visualizations',
                       help='输出目录')
    parser.add_argument('--max_persons', type=int, default=None,
                       help='最多处理的人数（None表示全部）')
    parser.add_argument('--max_video_frames', type=int, default=5,
                       help='每个人最多处理的视频帧数')
    
    args = parser.parse_args()
    
    visualize_landmarks_from_json(
        json_file=args.json_file,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        max_persons=args.max_persons,
        max_video_frames=args.max_video_frames
    )
