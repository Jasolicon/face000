"""
使用 InsightFace 检测人脸关键点，并计算与参考图像的球面角度偏离
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pathlib import Path
from insightface.app import FaceAnalysis
import math
import logging

# 设置日志级别为 ERROR，以屏蔽不必要的输出
logging.getLogger('insightface').setLevel(logging.ERROR)

def get_insightface_landmarks(detector, image_path):
    """
    使用 InsightFace 检测人脸关键点
    
    Args:
        detector: FaceAnalysis 检测器
        image_path: 图像路径
        
    Returns:
        landmarks: 关键点坐标 [5, 2] (左眼、右眼、鼻子、左嘴角、右嘴角)
        box: 人脸边界框 [x1, y1, x2, y2]
    """
    # 读取图像（使用 cv2，insightface 需要 numpy 数组）
    img = cv2.imread(str(image_path))
    if img is None:
        # 如果 cv2 读取失败（中文路径），使用 PIL 读取
        pil_img = Image.open(image_path).convert('RGB')
        img = np.array(pil_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # InsightFace 检测人脸和关键点
    faces = detector.get(img)
    
    if len(faces) == 0:
        return None, None
    
    # 使用第一个人脸
    face = faces[0]
    
    # InsightFace 返回的关键点格式：[5, 2] - (左眼、右眼、鼻子、左嘴角、右嘴角)
    landmarks = face.kps  # [5, 2]
    
    # 边界框格式：[x1, y1, x2, y2]
    box = face.bbox  # [x1, y1, x2, y2]
    
    return landmarks, box

def landmarks_to_3d(landmarks, box, img_width, img_height):
    """
    将2D关键点转换为3D坐标（假设人脸在球面上）
    
    Args:
        landmarks: 2D关键点 [5, 2]
        box: 人脸边界框 [x1, y1, x2, y2]
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        landmarks_3d: 3D关键点 [5, 3]
    """
    # 计算人脸中心
    face_center_x = (box[0] + box[2]) / 2
    face_center_y = (box[1] + box[3]) / 2
    face_width = box[2] - box[0]
    face_height = box[3] - box[1]
    
    # 归一化坐标到 [-1, 1] 范围
    landmarks_normalized = landmarks.copy()
    landmarks_normalized[:, 0] = (landmarks[:, 0] - face_center_x) / (face_width / 2)
    landmarks_normalized[:, 1] = (landmarks[:, 1] - face_center_y) / (face_height / 2)
    
    # 转换为3D坐标（假设在单位球面上）
    # 使用球面坐标：x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)
    landmarks_3d = np.zeros((5, 3))
    
    for i in range(5):
        x_norm = landmarks_normalized[i, 0]
        y_norm = landmarks_normalized[i, 1]
        
        # 计算球面坐标
        # 假设人脸在球面上，z坐标基于距离中心的距离
        r = np.sqrt(x_norm**2 + y_norm**2)
        if r > 1.0:
            r = 1.0
        
        # 计算角度
        theta = math.acos(1 - r)  # 从中心到边缘的角度
        phi = math.atan2(y_norm, x_norm) if x_norm != 0 else 0
        
        # 转换为3D坐标
        landmarks_3d[i, 0] = math.sin(theta) * math.cos(phi)
        landmarks_3d[i, 1] = math.sin(theta) * math.sin(phi)
        landmarks_3d[i, 2] = math.cos(theta)
    
    return landmarks_3d

def calculate_spherical_angle(landmarks1_3d, landmarks2_3d, landmarks1_2d=None, landmarks2_2d=None):
    """
    计算两组关键点之间的球面角度偏离（带正负号）
    
    Args:
        landmarks1_3d: 参考图像的3D关键点 [5, 3]
        landmarks2_3d: 当前图像的3D关键点 [5, 3]
        landmarks1_2d: 参考图像的2D关键点 [5, 2]（用于判断正负）
        landmarks2_2d: 当前图像的2D关键点 [5, 2]（用于判断正负）
        
    Returns:
        angles: 每个关键点的角度偏离（度，带正负号）[5]
               正数表示抬头，负数表示低头
        avg_angle: 平均角度偏离（度，带正负号）
    """
    angles = []
    
    for i in range(5):
        # 计算两个3D向量之间的角度
        v1 = landmarks1_3d[i]
        v2 = landmarks2_3d[i]
        
        # 归一化
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # 计算点积
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # 计算角度（弧度转度）
        angle_magnitude = math.degrees(math.acos(dot_product))
        
        # 判断正负：根据y坐标变化
        # 如果提供了2D关键点，使用y坐标判断
        if landmarks1_2d is not None and landmarks2_2d is not None:
            # y坐标减小（向上移动）= 抬头 = 正数
            # y坐标增大（向下移动）= 低头 = 负数
            y_diff = landmarks2_2d[i, 1] - landmarks1_2d[i, 1]
            
            # 如果y坐标减小（向上），角度为正；如果y坐标增大（向下），角度为负
            if y_diff < 0:
                # 向上移动，抬头，正数
                angle = angle_magnitude
            elif y_diff > 0:
                # 向下移动，低头，负数
                angle = -angle_magnitude
            else:
                # 没有垂直移动，角度为0或很小的正数
                angle = angle_magnitude if angle_magnitude > 0.1 else 0.0
        else:
            # 如果没有2D关键点，使用3D向量的y分量判断
            # y分量减小 = 抬头 = 正数
            y_diff_3d = v2_norm[1] - v1_norm[1]
            if y_diff_3d < -0.01:  # 向上
                angle = angle_magnitude
            elif y_diff_3d > 0.01:  # 向下
                angle = -angle_magnitude
            else:
                angle = angle_magnitude if angle_magnitude > 0.1 else 0.0
        
        angles.append(angle)
    
    avg_angle = np.mean(angles)
    return np.array(angles), avg_angle

def visualize_landmarks(image_path, landmarks, box, save_path=None):
    """
    在图像上绘制关键点和边界框
    
    Args:
        image_path: 图像路径
        landmarks: 关键点 [5, 2]
        box: 边界框 [x1, y1, x2, y2]
        save_path: 保存路径（可选）
    """
    # 使用 PIL 读取图像（支持中文路径）
    pil_img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试加载中文字体（支持跨平台）
    try:
        from font_utils import get_chinese_font_pil
        font = get_chinese_font_pil(20)
    except ImportError:
        # 如果 font_utils 不可用，使用旧方法
        try:
            # Windows 系统字体
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)  # 黑体
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)  # 微软雅黑
            except:
                font = ImageFont.load_default()  # 默认字体（可能不支持中文）
    
    # 绘制边界框（绿色）
    draw.rectangle([(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))], 
                   outline=(0, 255, 0), width=2)
    
    # 关键点名称
    landmark_names = ['左眼', '右眼', '鼻子', '左嘴角', '右嘴角']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # 绘制关键点
    for i, (point, name, color) in enumerate(zip(landmarks, landmark_names, colors)):
        x, y = int(point[0]), int(point[1])
        # 绘制圆点
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=color)
        # 绘制文本（使用 PIL 的 ImageDraw，支持中文）
        draw.text((x+10, y-10), name, fill=color, font=font)
    
    # 转换为 numpy 数组
    img_rgb = np.array(pil_img)
    
    if save_path:
        # 使用 PIL 保存（支持中文路径）
        pil_save = Image.fromarray(img_rgb)
        pil_save.save(str(save_path))
    
    return img_rgb

def main():
    # 设置路径
    video_dir = Path(r'C:\Codes\face000\train\datas\video\袁润东')
    reference_image = video_dir / '袁润东frame_000000.jpg'
    output_dir = Path(r'C:\Codes\face000\landmark_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化 InsightFace 检测器
    print("正在初始化 InsightFace 检测器...")
    try:
        # 尝试使用 GPU
        detector = FaceAnalysis(name='buffalo_l')
        detector.prepare(ctx_id=0, det_size=(640, 640))
        print("使用设备: GPU (CUDA)")
    except Exception as e:
        # 如果 GPU 失败，使用 CPU
        print(f"GPU 初始化失败，使用 CPU: {e}")
        detector = FaceAnalysis(name='buffalo_l')
        detector.prepare(ctx_id=-1, det_size=(640, 640))
        print("使用设备: CPU")
    
    # 获取前15张图片
    image_files = sorted(video_dir.glob('*.jpg'))[:-1]
    print(f"找到 {len(image_files)} 张图片")
    
    # 检测参考图像的关键点
    print(f"\n检测参考图像: {reference_image.name}")
    ref_landmarks, ref_box = get_insightface_landmarks(detector, reference_image)
    
    if ref_landmarks is None:
        print("❌ 参考图像中未检测到人脸")
        return
    
    # 获取参考图像尺寸
    ref_img = Image.open(reference_image)
    ref_width, ref_height = ref_img.size
    
    # 转换为3D坐标
    ref_landmarks_3d = landmarks_to_3d(ref_landmarks, ref_box, ref_width, ref_height)
    
    # 可视化参考图像
    ref_img_vis = visualize_landmarks(reference_image, ref_landmarks, ref_box, 
                                      output_dir / 'reference_landmarks.jpg')
    
    # 处理每张图片
    results = []
    
    for img_path in image_files:
        print(f"\n处理: {img_path.name}")
        
        # 检测关键点
        landmarks, box = get_insightface_landmarks(detector, img_path)
        
        if landmarks is None:
            print(f"  ⚠️ 未检测到人脸")
            results.append({
                'image': img_path.name,
                'landmarks': None,
                'angles': None,
                'avg_angle': None
            })
            continue
        
        # 获取图像尺寸
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # 转换为3D坐标
        landmarks_3d = landmarks_to_3d(landmarks, box, img_width, img_height)
        
        # 计算球面角度偏离（带正负号）
        angles, avg_angle = calculate_spherical_angle(
            ref_landmarks_3d, landmarks_3d, 
            ref_landmarks, landmarks  # 传入2D关键点用于判断正负
        )
        
        direction = "抬头" if avg_angle > 0 else "低头" if avg_angle < 0 else "平视"
        print(f"  平均角度偏离: {avg_angle:+.2f}° ({direction})")
        print(f"  各关键点角度偏离:")
        landmark_names = ['左眼', '右眼', '鼻子', '左嘴角', '右嘴角']
        for name, angle in zip(landmark_names, angles):
            direction_point = "抬头" if angle > 0 else "低头" if angle < 0 else "平视"
            print(f"    {name}: {angle:+.2f}° ({direction_point})")
        
        # 可视化
        img_vis = visualize_landmarks(img_path, landmarks, box)
        
        # 保存可视化结果（使用 PIL 保存，支持中文路径）
        save_path = output_dir / f'{img_path.stem}_landmarks.jpg'
        pil_save = Image.fromarray(img_vis)
        pil_save.save(str(save_path))
        
        results.append({
            'image': img_path.name,
            'landmarks': landmarks,
            'angles': angles,
            'avg_angle': avg_angle
        })
    
    # 保存结果摘要
    print("\n" + "="*60)
    print("结果摘要")
    print("="*60)
    print(f"{'图片':<30} {'平均角度偏离(°)':<20} {'方向':<10}")
    print("-"*60)
    
    for result in results:
        if result['avg_angle'] is not None:
            direction = "抬头" if result['avg_angle'] > 0 else "低头" if result['avg_angle'] < 0 else "平视"
            print(f"{result['image']:<30} {result['avg_angle']:+.2f}{'':<15} {direction:<10}")
        else:
            print(f"{result['image']:<30} {'未检测到人脸':<20} {'-':<10}")
    
    # 保存详细结果到文件
    output_file = output_dir / 'landmark_angles.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("关键点球面角度偏离分析\n")
        f.write("="*60 + "\n")
        f.write(f"参考图像: {reference_image.name}\n\n")
        
        for result in results:
            f.write(f"\n图片: {result['image']}\n")
            if result['avg_angle'] is not None:
                direction = "抬头" if result['avg_angle'] > 0 else "低头" if result['avg_angle'] < 0 else "平视"
                f.write(f"  平均角度偏离: {result['avg_angle']:+.2f}° ({direction})\n")
                f.write(f"  各关键点角度偏离:\n")
                landmark_names = ['左眼', '右眼', '鼻子', '左嘴角', '右嘴角']
                for name, angle in zip(landmark_names, result['angles']):
                    direction_point = "抬头" if angle > 0 else "低头" if angle < 0 else "平视"
                    f.write(f"    {name}: {angle:+.2f}° ({direction_point})\n")
            else:
                f.write(f"  未检测到人脸\n")
    
    print(f"\n结果已保存到: {output_file}")
    print(f"可视化图像已保存到: {output_dir}")

if __name__ == "__main__":
    main()

