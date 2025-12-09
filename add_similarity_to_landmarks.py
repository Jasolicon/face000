"""
使用 InsightFace 和 DINOv2 提取特征，计算相似度并添加到 landmark_angles.txt
"""
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from insightface.app import FaceAnalysis
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import logging
import re

# 设置日志级别
logging.getLogger('insightface').setLevel(logging.ERROR)

class DINOv2FeatureExtractor:
    """使用 DINOv2 提取图像特征"""
    
    def __init__(self, model_name='dinov2_vitb14', device=None):
        """
        初始化 DINOv2 特征提取器
        
        Args:
            model_name: DINOv2 模型名称 ('dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitl14', 'dinov2_vitg14')
            device: 计算设备
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 尝试加载 DINOv2 模型（如果失败，尝试其他模型名称）
        model_names_to_try = [model_name, 'dinov2_vitb14', 'dinov2_vits14', 'vit_base_patch16_224']
        self.model = None
        
        for try_name in model_names_to_try:
            try:
                print(f"  尝试加载 DINOv2 模型: {try_name}...")
                self.model = timm.create_model(
                    try_name,
                    pretrained=True,
                    num_classes=0,  # 移除分类头
                )
                self.model.eval()
                self.model.to(self.device)
                print(f"  ✓ 成功加载模型: {try_name}")
                break
            except Exception as e:
                print(f"  ✗ 模型 {try_name} 加载失败: {e}")
                continue
        
        if self.model is None:
            raise RuntimeError("无法加载任何 DINOv2 模型，请检查 timm 库和模型名称")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """
        提取图像特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            features: 归一化的特征向量 (numpy array)
        """
        # 使用 PIL 读取（支持中文路径）
        img = Image.open(image_path).convert('RGB')
        
        # 预处理
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(img_tensor)
            # L2归一化
            features = F.normalize(features, p=2, dim=1)
        
        # 转换为numpy数组
        features = features.cpu().numpy().flatten()
        
        return features

class InsightFaceFeatureExtractor:
    """使用 InsightFace 提取人脸特征"""
    
    def __init__(self, use_cpu=False):
        """
        初始化 InsightFace 特征提取器
        
        Args:
            use_cpu: 是否使用 CPU
        """
        self.detector = FaceAnalysis(name='buffalo_l')
        try:
            if use_cpu:
                self.detector.prepare(ctx_id=-1, det_size=(640, 640))
            else:
                self.detector.prepare(ctx_id=0, det_size=(640, 640))
        except:
            self.detector.prepare(ctx_id=-1, det_size=(640, 640))
    
    def extract_features(self, image_path):
        """
        提取人脸特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            features: 归一化的特征向量 (numpy array)，如果未检测到人脸返回 None
        """
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            # 如果 cv2 读取失败（中文路径），使用 PIL 读取
            pil_img = Image.open(image_path).convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 检测人脸并提取特征
        faces = self.detector.get(img)
        
        if len(faces) == 0:
            return None
        
        # 使用第一个人脸的特征
        face = faces[0]
        features = face.normed_embedding  # 已经归一化的特征向量
        
        return features

def cosine_similarity(feat1, feat2):
    """计算余弦相似度"""
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def arcface_similarity(feat1, feat2):
    """
    计算 ArcFace 相似度（实际上就是余弦相似度，但 ArcFace 特征已经归一化）
    """
    # ArcFace 特征已经归一化，直接计算点积即可
    return np.dot(feat1, feat2)

def parse_landmark_file(file_path):
    """
    解析 landmark_angles.txt 文件，提取图片名称和对应的行号
    
    Returns:
        image_lines: {image_name: [line_numbers]} 字典
    """
    image_lines = {}
    current_image = None
    current_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # 检查是否是图片行
        if line.startswith('图片:'):
            # 保存上一个图片的信息
            if current_image is not None:
                image_lines[current_image] = current_lines
            
            # 提取图片名称
            match = re.search(r'图片:\s*(.+)', line)
            if match:
                current_image = match.group(1).strip()
                current_lines = [i]
            else:
                current_image = None
                current_lines = []
        elif current_image is not None:
            # 属于当前图片的行
            current_lines.append(i)
    
    # 保存最后一个图片
    if current_image is not None:
        image_lines[current_image] = current_lines
    
    return image_lines, lines

def update_landmark_file(file_path, similarity_data):
    """
    更新 landmark_angles.txt 文件，添加相似度信息
    
    Args:
        file_path: 文件路径
        similarity_data: {image_name: {'insightface_cosine': float, 'insightface_arcface': float, 
                                      'dinov2_cosine': float, 'dinov2_arcface': float}} 字典
    """
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 为每个图片添加相似度信息
    for image_name, data in similarity_data.items():
        # 查找该图片的记录结束位置（下一个"图片:"之前或文件末尾）
        pattern = rf'(图片:\s*{re.escape(image_name)}.*?)(?=\n图片:|\Z)'
        
        def replace_func(match):
            original_text = match.group(1)
            # 检查是否已经添加了相似度信息
            if '特征相似度:' in original_text:
                return original_text  # 已经添加过，不重复添加
            
            # 添加相似度信息
            sim_text = "\n  特征相似度:\n"
            
            # InsightFace 相似度
            if data.get('insightface_cosine') is not None:
                if_cos = data['insightface_cosine']
                if_arc = data['insightface_arcface']
                marker_if = " ⭐" if if_cos > 0.9 or if_arc > 0.9 else ""
                sim_text += f"    InsightFace - Cosine: {if_cos:.4f}, ArcFace: {if_arc:.4f}{marker_if}\n"
            else:
                sim_text += f"    InsightFace - 未检测到人脸\n"
            
            # DINOv2 相似度
            if data.get('dinov2_cosine') is not None:
                dino_cos = data['dinov2_cosine']
                dino_arc = data['dinov2_arcface']
                marker_dino = " ⭐" if dino_cos > 0.9 or dino_arc > 0.9 else ""
                sim_text += f"    DINOv2 - Cosine: {dino_cos:.4f}, ArcFace: {dino_arc:.4f}{marker_dino}\n"
            else:
                sim_text += f"    DINOv2 - 提取失败\n"
            
            return original_text + sim_text
        
        content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # 设置路径
    standard_image = Path(r'C:\Codes\face000\train\datas\face\袁润东.jpg')
    video_dir = Path(r'C:\Codes\face000\train\datas\video\袁润东')
    landmark_file = Path(r'C:\Codes\face000\landmark_analysis_mtcnn\landmark_angles.txt')
    
    print("="*60)
    print("特征相似度计算")
    print("="*60)
    
    # 初始化特征提取器
    print("\n初始化特征提取器...")
    print("  初始化 InsightFace...")
    insightface_extractor = InsightFaceFeatureExtractor(use_cpu=True)
    
    print("  初始化 DINOv2...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    dinov2_extractor = DINOv2FeatureExtractor(model_name='dinov2_vitb14', device=device)
    
    # 提取标准图像的特征
    print(f"\n提取标准图像特征: {standard_image.name}")
    
    # InsightFace 特征
    print("  提取 InsightFace 特征...")
    standard_if_feat = insightface_extractor.extract_features(standard_image)
    if standard_if_feat is None:
        print("  ⚠️ 标准图像中未检测到人脸（InsightFace）")
        standard_if_feat = None
    else:
        print(f"  ✓ InsightFace 特征维度: {len(standard_if_feat)}")
    
    # DINOv2 特征
    print("  提取 DINOv2 特征...")
    try:
        standard_dino_feat = dinov2_extractor.extract_features(standard_image)
        print(f"  ✓ DINOv2 特征维度: {len(standard_dino_feat)}")
    except Exception as e:
        print(f"  ❌ DINOv2 特征提取失败: {e}")
        standard_dino_feat = None
    
    # 获取所有图片文件
    image_files = sorted(video_dir.glob('*.jpg'))
    print(f"\n找到 {len(image_files)} 张图片")
    
    # 解析现有文件
    print(f"\n解析现有文件: {landmark_file}")
    image_lines, _ = parse_landmark_file(landmark_file)
    print(f"  找到 {len(image_lines)} 个图片记录")
    
    # 计算相似度
    similarity_data = {}
    
    print(f"\n开始计算相似度...")
    for idx, img_path in enumerate(image_files, 1):
        image_name = img_path.name
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_name}")
        
        data = {}
        
        # InsightFace 相似度
        if_feat = insightface_extractor.extract_features(img_path)
        if if_feat is not None and standard_if_feat is not None:
            if_cosine = cosine_similarity(standard_if_feat, if_feat)
            if_arcface = arcface_similarity(standard_if_feat, if_feat)
            data['insightface_cosine'] = if_cosine
            data['insightface_arcface'] = if_arcface
            marker = " ⭐" if if_cosine > 0.9 or if_arcface > 0.9 else ""
            print(f"  InsightFace - Cosine: {if_cosine:.4f}, ArcFace: {if_arcface:.4f}{marker}")
        else:
            data['insightface_cosine'] = None
            data['insightface_arcface'] = None
            print(f"  InsightFace - 未检测到人脸")
        
        # DINOv2 相似度
        try:
            dino_feat = dinov2_extractor.extract_features(img_path)
            if dino_feat is not None and standard_dino_feat is not None:
                dino_cosine = cosine_similarity(standard_dino_feat, dino_feat)
                dino_arcface = arcface_similarity(standard_dino_feat, dino_feat)
                data['dinov2_cosine'] = dino_cosine
                data['dinov2_arcface'] = dino_arcface
                marker = " ⭐" if dino_cosine > 0.9 or dino_arcface > 0.9 else ""
                print(f"  DINOv2 - Cosine: {dino_cosine:.4f}, ArcFace: {dino_arcface:.4f}{marker}")
            else:
                data['dinov2_cosine'] = None
                data['dinov2_arcface'] = None
                print(f"  DINOv2 - 特征提取失败")
        except Exception as e:
            data['dinov2_cosine'] = None
            data['dinov2_arcface'] = None
            print(f"  DINOv2 - 错误: {e}")
        
        similarity_data[image_name] = data
    
    # 更新文件
    print(f"\n更新文件: {landmark_file}")
    update_landmark_file(landmark_file, similarity_data)
    print("  ✓ 文件更新完成")
    
    # 统计信息
    print(f"\n统计信息:")
    high_sim_count_if = sum(1 for d in similarity_data.values() 
                            if d.get('insightface_cosine') and 
                            (d['insightface_cosine'] > 0.9 or d.get('insightface_arcface', 0) > 0.9))
    high_sim_count_dino = sum(1 for d in similarity_data.values() 
                              if d.get('dinov2_cosine') and 
                              (d['dinov2_cosine'] > 0.9 or d.get('dinov2_arcface', 0) > 0.9))
    
    print(f"  InsightFace 高相似度 (>0.9): {high_sim_count_if}/{len(similarity_data)}")
    print(f"  DINOv2 高相似度 (>0.9): {high_sim_count_dino}/{len(similarity_data)}")
    
    print("\n完成！")

if __name__ == "__main__":
    main()

