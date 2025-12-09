"""
人脸检测模块 - 使用 facenet_pytorch
"""
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import cv2


class FaceDetector:
    """人脸检测器类"""
    
    def __init__(self, device=None, image_size=160, min_face_ratio=0.5):
        """
        初始化人脸检测器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')，如果为None则自动选择
            image_size: 检测到的人脸图像尺寸
            min_face_ratio: 最小面部占比阈值（0-1之间），低于此值的人脸将被过滤
                           默认0.5表示面部区域应至少占检测框的50%
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.image_size = image_size
        self.min_face_ratio = min_face_ratio
        # 初始化MTCNN人脸检测器
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device
        )
    
    def calculate_face_ratio(self, box, img_width, img_height):
        """
        计算面部占比（相对于检测框）
        
        方法：假设面部区域在检测框的下75%部分（头发在上25%部分）
        这是一个启发式方法，适用于大多数正面人脸
        
        Args:
            box: 人脸边界框 [x1, y1, x2, y2]
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            face_ratio: 面部占比（0-1之间）
            hair_ratio: 头发占比（0-1之间）
        """
        x1, y1, x2, y2 = box.astype(float)
        
        # 确保坐标在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # 计算检测框的尺寸
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        if box_area == 0:
            return 0.0, 0.0
        
        # 假设面部区域在检测框的下75%部分
        # 头发区域在上25%部分
        # 这是一个简化的模型，适用于大多数正面人脸
        hair_height_ratio = 0.25  # 头发占检测框高度的25%
        face_height_ratio = 0.75  # 面部占检测框高度的75%
        
        # 计算头发区域面积（上25%）
        hair_area = box_width * (box_height * hair_height_ratio)
        
        # 计算面部区域面积（下75%）
        face_area = box_width * (box_height * face_height_ratio)
        
        # 计算占比
        face_ratio = face_area / box_area
        hair_ratio = hair_area / box_area
        
        return face_ratio, hair_ratio
    
    def detect_faces(self, image_path, filter_by_face_ratio=True):
        """
        检测图像中的人脸
        
        Args:
            image_path: 图像路径或PIL Image对象
            filter_by_face_ratio: 是否根据面部占比过滤人脸（默认True）
            
        Returns:
            faces: 检测到的人脸列表，每个元素是一个PIL Image对象
            boxes: 人脸边界框列表，格式为 [x1, y1, x2, y2]
            probs: 检测置信度列表
        """
        # 加载图像
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path
        
        # 检测人脸
        boxes, probs = self.mtcnn.detect(img)
        
        faces = []
        filtered_boxes = []
        filtered_probs = []
        
        if boxes is not None:
            # 提取每个人脸并计算面部占比
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.width, x2)
                y2 = min(img.height, y2)
                
                # 计算面部占比
                if filter_by_face_ratio:
                    face_ratio, hair_ratio = self.calculate_face_ratio(
                        box, img.width, img.height
                    )
                    
                    # 如果面部占比太小，跳过这个人脸
                    if face_ratio < self.min_face_ratio:
                        print(f"  过滤人脸 {i+1}: 面部占比 {face_ratio:.2%} < 阈值 {self.min_face_ratio:.2%} "
                              f"(头发占比: {hair_ratio:.2%})")
                        continue
                
                # 裁剪人脸
                face = img.crop((x1, y1, x2, y2))
                faces.append(face)
                filtered_boxes.append(box)
                if probs is not None and i < len(probs):
                    filtered_probs.append(probs[i])
        
        # 返回过滤后的结果
        if len(faces) == 0:
            return [], None, None
        
        return faces, np.array(filtered_boxes) if filtered_boxes else None, np.array(filtered_probs) if filtered_probs else None
    
    def extract_aligned_face(self, image_path):
        """
        提取对齐后的人脸图像（用于特征提取）
        
        Args:
            image_path: 图像路径或PIL Image对象
            
        Returns:
            aligned_face: 对齐后的人脸PIL Image，如果未检测到人脸则返回None
            box: 人脸边界框
            prob: 检测置信度
        """
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path
        
        # 提取对齐的人脸
        aligned_face = self.mtcnn(img)
        
        if aligned_face is not None:
            # 获取检测框和置信度
            boxes, probs = self.mtcnn.detect(img)
            if boxes is not None and len(boxes) > 0:
                return aligned_face, boxes[0], probs[0] if probs is not None else 1.0
        
        return None, None, None
    
    def draw_boxes(self, image_path, boxes, save_path=None):
        """
        在图像上绘制人脸边界框
        
        Args:
            image_path: 图像路径
            boxes: 人脸边界框列表
            save_path: 保存路径，如果为None则只返回图像
            
        Returns:
            img_with_boxes: 绘制了边界框的图像
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        return img_rgb

