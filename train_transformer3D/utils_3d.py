"""
3D关键点和姿态提取工具
使用InsightFace提取2D关键点，然后通过PnP算法估计3D关键点和姿态（yaw, pitch, roll）
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from insightface.app import FaceAnalysis
import math
import logging
import os
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils import get_insightface_detector

# 设置日志级别
logging.getLogger('insightface').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# 标准3D人脸模型（基于平均人脸，单位：毫米）
# 这些是5个关键点在3D空间中的标准位置（正面，yaw=0°）
STANDARD_3D_LANDMARKS = np.array([
    [-36.0, 13.0, -42.0],   # 左眼中心
    [36.0, 13.0, -42.0],    # 右眼中心
    [0.0, 0.0, 0.0],        # 鼻尖（作为原点）
    [-25.0, -30.0, -42.0], # 左嘴角
    [25.0, -30.0, -42.0]   # 右嘴角
], dtype=np.float32)


def estimate_pose_from_landmarks(landmarks_2d, box, img_width, img_height, 
                                 model_3d=STANDARD_3D_LANDMARKS):
    """
    使用PnP算法从2D关键点估计3D姿态
    
    Args:
        landmarks_2d: 2D关键点 [5, 2]
        box: 人脸边界框 [x1, y1, x2, y2]
        img_width: 图像宽度
        img_height: 图像高度
        model_3d: 标准3D关键点模型 [5, 3]
    
    Returns:
        landmarks_3d: 估计的3D关键点 [5, 3]（相对于相机坐标系）
        rotation_vector: 旋转向量 [3]（用于cv2.Rodrigues）
        translation_vector: 平移向量 [3]
        euler_angles: 欧拉角 (yaw, pitch, roll) 度
        rotation_matrix: 旋转矩阵 [3, 3]
    """
    # 计算相机内参（简化假设）
    # 假设焦距 = 图像宽度（经验值）
    focal_length = img_width
    center_x = img_width / 2.0
    center_y = img_height / 2.0
    
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # 畸变系数（假设无畸变）
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # 使用PnP算法求解姿态
    # 注意：SOLVEPNP_ITERATIVE需要至少6个点，但我们只有5个点
    # 使用EPnP算法（只需要4个点）或UPnP算法
    # 对于5个点，使用SOLVEPNP_EPNP（推荐）或SOLVEPNP_UPNP
    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_3d,           # 3D模型点 [5, 3]
            landmarks_2d,       # 2D图像点 [5, 2]
            camera_matrix,      # 相机内参
            dist_coeffs,        # 畸变系数
            flags=cv2.SOLVEPNP_EPNP  # EPnP算法，只需要4个点
        )
    except cv2.error as e:
        # 如果EPnP失败，尝试UPnP
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_3d,
                landmarks_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_UPNP  # UPnP算法，也只需要4个点
            )
        except cv2.error:
            # 如果都失败，返回None
            return None, None, None, None, None
    
    if not success:
        return None, None, None, None, None
    
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # 计算欧拉角（yaw, pitch, roll）
    # 从旋转矩阵提取欧拉角
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                   rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        yaw = math.atan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0
    
    # 转换为度
    yaw_deg = math.degrees(yaw)
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)
    
    euler_angles = np.array([yaw_deg, pitch_deg, roll_deg])
    
    # 将标准3D模型点转换到当前姿态
    # landmarks_3d = R @ model_3d.T + t
    landmarks_3d = (rotation_matrix @ model_3d.T + translation_vector).T
    
    return landmarks_3d, rotation_vector, translation_vector, euler_angles, rotation_matrix


def get_3d_landmarks_and_pose(detector, image_path):
    """
    使用InsightFace提取2D关键点，然后估计3D关键点和姿态
    
    Args:
        detector: InsightFace检测器
        image_path: 图像路径（可以是str或PIL Image）
    
    Returns:
        landmarks_2d: 2D关键点 [5, 2] 或 None
        landmarks_3d: 3D关键点 [5, 3] 或 None
        box: 人脸边界框 [x1, y1, x2, y2] 或 None
        euler_angles: 欧拉角 (yaw, pitch, roll) 度 或 None
        rotation_matrix: 旋转矩阵 [3, 3] 或 None
    """
    # 读取图像
    if isinstance(image_path, Image.Image):
        img_rgb = np.array(image_path.convert('RGB'))
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_width, img_height = image_path.size
    else:
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_width, img_height = pil_img.size
        except Exception as e:
            img = cv2.imread(str(image_path))
            if img is None:
                return None, None, None, None, None
            img_height, img_width = img.shape[:2]
    
    # InsightFace检测人脸和关键点
    faces = detector.get(img)
    
    if len(faces) == 0:
        return None, None, None, None, None
    
    # 使用第一个人脸
    face = faces[0]
    landmarks_2d = face.kps  # [5, 2]
    box = face.bbox  # [x1, y1, x2, y2]
    
    # 估计3D关键点和姿态
    landmarks_3d, _, _, euler_angles, rotation_matrix = estimate_pose_from_landmarks(
        landmarks_2d, box, img_width, img_height
    )
    
    if landmarks_3d is None:
        return landmarks_2d, None, box, None, None
    
    return landmarks_2d, landmarks_3d, box, euler_angles, rotation_matrix


def visualize_3d_landmarks_and_pose(image, landmarks_2d, landmarks_3d, 
                                    euler_angles, box, save_path=None):
    """
    可视化3D关键点和姿态
    
    Args:
        image: 输入图像（PIL Image或numpy数组）
        landmarks_2d: 2D关键点 [5, 2]
        landmarks_3d: 3D关键点 [5, 3]
        euler_angles: 欧拉角 (yaw, pitch, roll) 度
        box: 人脸边界框
        save_path: 保存路径（可选）
    """
    from PIL import ImageDraw, ImageFont
    
    if isinstance(image, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        img = image.copy()
    
    draw = ImageDraw.Draw(img)
    
    # 绘制边界框
    if box is not None:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline='red', width=2)
    
    # 绘制2D关键点
    landmark_names = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角']
    colors = ['blue', 'blue', 'green', 'red', 'red']
    
    for i, (name, color) in enumerate(zip(landmark_names, colors)):
        x, y = landmarks_2d[i]
        draw.ellipse([x-3, y-3, x+3, y+3], fill=color, outline='black')
        draw.text((x+5, y-10), name, fill=color)
    
    # 显示姿态信息
    if euler_angles is not None:
        yaw, pitch, roll = euler_angles
        text = f"Yaw: {yaw:.1f}°\nPitch: {pitch:.1f}°\nRoll: {roll:.1f}°"
        draw.text((10, 10), text, fill='yellow')
    
    if save_path:
        img.save(save_path)
    
    return img
