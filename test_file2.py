"""
测试文件2 - 视频第一帧人脸检测、特征比对和中文标注
使用 DINOv2 提取特征，与 features 目录中的特征对比（阈值0.25）
在图片上标注关键点和角度
"""
import os

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import timm
from feature_manager import FeatureManager
from feature_matcher import FaceMatcher
from utils import (
    get_insightface_detector, 
    get_insightface_landmarks,
    detect_landmarks_and_calculate_angle,
    draw_landmarks_on_image,
    landmarks_to_3d,
    calculate_spherical_angle,
    deduplicate_recognition_results,
    draw_recognition_results
)
from train_transformer.load_transformer import load_transformer_model, correct_features_with_transformer


class DINOv2FeatureExtractor:
    """使用 DINOv2 提取图像特征"""
    
    def __init__(self, model_name='dinov2_vitb14', device=None):
        """
        初始化 DINOv2 特征提取器
        
        Args:
            model_name: DINOv2 模型名称
            device: 计算设备
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 默认使用 DINOv2 ViT-B/14，如果失败则使用 vit_base_patch16_224 作为备用
        self.use_dinov2 = False
        
        try:
            print(f"  尝试从 torch.hub 加载 DINOv2 ViT-B/14（默认模型）...")
            # 使用 torch.hub 加载 DINOv2 模型
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            self.use_dinov2 = True
            print(f"  ✓ 成功加载 DINOv2 ViT-B/14 模型")
        except Exception as e:
            print(f"  ✗ 无法加载 DINOv2 ViT-B/14: {str(e)}")
            print(f"  提示: 运行 'python download_dinov2.py' 可以提前下载模型")
            print(f"  将使用备用模型: vit_base_patch16_224")
            self.use_dinov2 = False
            
            # 使用备用模型
            model_name_to_use = 'vit_base_patch16_224'
            try:
                self.model = timm.create_model(
                    model_name_to_use,
                    pretrained=True,
                    num_classes=0,  # 移除分类头
                )
                self.model.eval()
                self.model.to(self.device)
                print(f"  ✓ 成功加载备用模型: {model_name_to_use}")
            except Exception as e2:
                error_msg = f"无法加载模型 {model_name_to_use}: {str(e2)}"
                error_msg += "\n提示: timm 会自动缓存下载的模型，首次下载后后续使用会从缓存加载"
                raise RuntimeError(error_msg)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """
        提取图像特征
        
        Args:
            image: PIL Image对象或图像路径
            
        Returns:
            features: 归一化的特征向量 (numpy array)
        """
        # 加载图像
        if isinstance(image, str) or isinstance(image, Path):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB') if hasattr(image, 'convert') else image
        
        # 预处理
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            if self.use_dinov2:
                # DINOv2 模型，需要特殊处理输出
                output = self.model(img_tensor)
                # DINOv2 返回的是字典，包含 'x_norm_clstoken' 或直接是特征向量
                if isinstance(output, dict):
                    if 'x_norm_clstoken' in output:
                        features = output['x_norm_clstoken']
                    elif 'x_prenorm' in output:
                        # 如果是 prenorm，取 CLS token (第一个 token)
                        features = output['x_prenorm'][:, 0]
                    else:
                        # 取第一个值
                        features = list(output.values())[0]
                        if features.dim() > 2:
                            # 如果是序列，取 CLS token
                            features = features[:, 0]
                else:
                    # 直接返回特征向量
                    features = output
                    if features.dim() > 2:
                        # 如果是序列，取 CLS token
                        features = features[:, 0]
            else:
                # 标准模型（vit_base_patch16_224）
                features = self.model(img_tensor)
            
            # L2归一化
            features = F.normalize(features, p=2, dim=1)
        
        # 转换为numpy数组
        features = features.cpu().numpy().flatten()
        
        return features

def get_chinese_font(font_size=20):
    """
    获取中文字体（支持跨平台）
    
    Args:
        font_size: 字体大小
        
    Returns:
        font: PIL字体对象
    """
    try:
        from font_utils import get_chinese_font_pil
        return get_chinese_font_pil(font_size)
    except ImportError:
        # 如果 font_utils 不可用，使用旧方法
        # 尝试使用系统字体
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyhbd.ttc',  # 微软雅黑 Bold
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except:
                    continue
        
        # 如果找不到字体，使用默认字体（可能不支持中文）
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except:
            return ImageFont.load_default()


def draw_boxes_with_names(image, boxes, names, probs=None):
    """
    在图像上绘制检测框和中文人名
    
    Args:
        image: PIL Image对象
        boxes: 人脸边界框列表 [[x1, y1, x2, y2], ...]
        names: 人名列表
        probs: 置信度列表（可选）
        
    Returns:
        annotated_image: 标注后的PIL Image对象
    """
    # 创建可绘制的图像副本
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 获取中文字体
    font = get_chinese_font(font_size=24)
    small_font = get_chinese_font(font_size=16)
    
    for i, (box, name) in enumerate(zip(boxes, names)):
        x1, y1, x2, y2 = box.astype(int)
        
        # 扩展检测框25%
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 扩展25%意味着宽度和高度各乘以1.25
        new_width = width * 1.25
        new_height = height * 1.25
        
        # 计算新的边界框坐标
        new_x1 = int(center_x - new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)
        
        # 确保边界框不超出图像范围
        img_width, img_height = image.size
        new_x1 = max(0, min(new_x1, img_width - 1))
        new_y1 = max(0, min(new_y1, img_height - 1))
        new_x2 = max(0, min(new_x2, img_width - 1))
        new_y2 = max(0, min(new_y2, img_height - 1))
        
        # 绘制扩展后的检测框
        draw.rectangle([new_x1, new_y1, new_x2, new_y2], outline=(0, 255, 0), width=3)
        
        # 准备文本
        if probs is not None and i < len(probs):
            text = f"{name} ({probs[i]:.2f})"
        else:
            text = name
        
        # 计算文本位置（在框的上方）
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 绘制文本背景（使用扩展后的边界框坐标）
        text_x = new_x1
        text_y = new_y1 - text_height - 5
        if text_y < 0:
            text_y = new_y2 + 5
        
        # 绘制半透明背景
        overlay = Image.new('RGBA', draw_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
            fill=(0, 0, 0, 180)
        )
        draw_image = Image.alpha_composite(draw_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(draw_image)
        
        # 绘制文本
        draw.text((text_x, text_y), text, fill=(255, 255, 0), font=font)
    
    return draw_image


def process_video_frame(video_path, features_dir='features', output_path=None, similarity_threshold=0.25, 
                       reference_image_path=None, use_cpu=False, frame_number=0, 
                       transformer_model_path=None):
    """
    处理视频指定帧：检测人脸、比对特征、标注结果
    
    Args:
        video_path: 视频文件路径
        features_dir: 特征存储目录
        output_path: 输出图像路径，如果为None则不保存
        similarity_threshold: 相似度阈值（默认0.25）
        reference_image_path: 参考图像路径（用于计算角度，可选）
        use_cpu: 是否使用 CPU
        frame_number: 要处理的帧号（从0开始，默认0表示第一帧）
        transformer_model_path: Transformer模型路径（用于特征矫正，可选）
        
    Returns:
        annotated_image: 标注后的图像
        results: 检测和比对结果
    """
    print("=" * 70)
    print(f"视频第 {frame_number} 帧人脸检测和特征比对")
    print("=" * 70)
    
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return None, None
    
    print(f"视频路径: {video_path}")
    
    # 读取视频指定帧
    print(f"\n正在读取视频第 {frame_number} 帧...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return None, None
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  视频总帧数: {total_frames}")
    
    # 检查帧号是否有效
    if frame_number < 0:
        print(f"  警告: 帧号 {frame_number} 无效，使用第 0 帧")
        frame_number = 0
    elif frame_number >= total_frames:
        print(f"  警告: 帧号 {frame_number} 超出范围（总帧数: {total_frames}），使用最后一帧")
        frame_number = total_frames - 1
    
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"错误: 无法读取视频第 {frame_number} 帧")
        return None, None
    
    print(f"  成功读取第 {frame_number} 帧")
    
    # 转换为PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    print(f"图像尺寸: {pil_image.size}")
    
    # 初始化组件
    print("\n正在初始化 DINOv2 特征提取器...")
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f"  使用设备: {device}")
    feature_extractor = DINOv2FeatureExtractor(device=device)
    
    # 加载Transformer模型（如果提供）
    transformer_model = None
    if transformer_model_path and Path(transformer_model_path).exists():
        print("\n正在加载Transformer模型...")
        try:
            transformer_model, _ = load_transformer_model(transformer_model_path, device=device)
            print("  ✓ Transformer模型已加载，将用于特征矫正")
        except Exception as e:
            print(f"  ⚠️ Transformer模型加载失败: {e}")
            print("  将跳过特征矫正步骤")
    else:
        if transformer_model_path:
            print(f"\n⚠️ Transformer模型文件不存在: {transformer_model_path}")
            print("  将跳过特征矫正步骤")
        else:
            print("\n未提供Transformer模型路径，将跳过特征矫正步骤")
    
    # 加载特征数据库
    print("\n正在加载特征数据库...")
    feature_manager = FeatureManager(storage_dir=features_dir)
    feature_count = feature_manager.get_count()
    print(f"  已加载 {feature_count} 个特征")
    
    if feature_count > 0:
        features, metadata = feature_manager.get_all_features()
        if features is not None and len(features) > 0:
            db_feature_dim = features.shape[1]
            print(f"  数据库特征维度: {db_feature_dim}")
    
    print("\n正在初始化特征比对器...")
    print(f"  相似度阈值: {similarity_threshold}")
    print(f"  说明: 只有余弦相似度 >= {similarity_threshold} 的匹配才会被接受")
    face_matcher = FaceMatcher(
        feature_manager=feature_manager,
        similarity_threshold=similarity_threshold
    )
    
    # 初始化 InsightFace 检测器（用于关键点检测）
    print("\n正在初始化 InsightFace 检测器（用于关键点检测）...")
    insightface_detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 如果提供了参考图像，加载参考关键点
    reference_landmarks_3d = None
    reference_landmarks_2d = None
    reference_box = None
    reference_img_size = None
    if reference_image_path and os.path.exists(reference_image_path):
        print(f"\n正在加载参考图像: {reference_image_path}")
        ref_landmarks, ref_box = get_insightface_landmarks(insightface_detector, reference_image_path)
        if ref_landmarks is not None:
            ref_img = Image.open(reference_image_path)
            ref_width, ref_height = ref_img.size
            reference_landmarks_3d = landmarks_to_3d(ref_landmarks, ref_box, ref_width, ref_height)
            reference_landmarks_2d = ref_landmarks
            reference_box = ref_box
            reference_img_size = (ref_width, ref_height)
            print("  ✓ 参考关键点加载成功")
        else:
            print("  ⚠️ 参考图像中未检测到人脸")
    
    # 使用 InsightFace 检测人脸
    print("\n正在使用 InsightFace 检测人脸...")
    # 将 PIL 图像转换为 numpy 数组（BGR格式，InsightFace 需要）
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 检测人脸
    detected_faces = insightface_detector.get(img_bgr)
    
    if len(detected_faces) == 0:
        print("未检测到人脸")
        return pil_image, []
    
    print(f"检测到 {len(detected_faces)} 个人脸")
    print("  说明: 将使用 InsightFace 裁剪的人脸区域进行特征提取")
    
    # 提取每个人脸的特征并比对
    print("\n正在提取特征并比对...")
    print("  流程: InsightFace 裁剪的人脸区域 → DINOv2 提取特征 → 与DINO特征库比对")
    results = []
    matched_names = []
    matched_probs = []
    
    for i, face_info in enumerate(detected_faces):
        print(f"  处理第 {i+1}/{len(detected_faces)} 个人脸...")
        
        # 获取边界框
        box = face_info.bbox  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 获取置信度（如果有）
        prob = face_info.det_score if hasattr(face_info, 'det_score') else 1.0
        
        # 裁剪人脸区域
        # 确保坐标在图像范围内
        img_width, img_height = pil_image.size
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # 裁剪人脸
        face_crop = pil_image.crop((x1, y1, x2, y2))
        
        print(f"    人脸尺寸: {face_crop.size}, 边界框: [{x1}, {y1}, {x2}, {y2}], 置信度: {prob:.4f}")
        
        try:
            # 检查人脸图像是否有效（只检查是否为空，不再检查大小）
            face_width, face_height = face_crop.size
            if face_width <= 0 or face_height <= 0:
                print(f"    警告: 人脸图像无效 ({face_crop.size})，跳过处理")
                matched_names.append('图像无效')
                matched_probs.append(0.0)
                results.append({
                    'box': [x1, y1, x2, y2],
                    'prob': prob,
                    'name': '图像无效',
                    'similarity': 0.0,
                    'match': None,
                    'landmarks': None,
                    'angles': None,
                    'avg_angle': None
                })
                continue
            
            # 使用 DINOv2 提取特征（从裁剪的人脸区域）
            features = feature_extractor.extract_features(face_crop)
            print(f"    特征提取成功，特征维度: {len(features)} (使用DINOv2提取InsightFace裁剪的人脸区域)")
            
            # 使用Transformer矫正特征（如果模型已加载且提供了参考图像）
            if transformer_model is not None and reference_image_path and reference_landmarks_3d is not None:
                # 使用已检测到的关键点计算角度
                if hasattr(face_info, 'kps') and face_info.kps is not None:
                    video_landmarks = face_info.kps
                    video_box = box
                    video_width = x2 - x1
                    video_height = y2 - y1
                    
                    # 转换为3D坐标
                    video_landmarks_3d = landmarks_to_3d(video_landmarks, video_box, video_width, video_height)
                    
                    # 计算球面角
                    angles, _ = calculate_spherical_angle(
                        reference_landmarks_3d,
                        video_landmarks_3d,
                        reference_landmarks_2d,
                        video_landmarks
                    )
                    
                    # 使用Transformer矫正特征
                    features = correct_features_with_transformer(
                        transformer_model, 
                        features, 
                        angles, 
                        device=device
                    )
                    print(f"    ✓ 已使用Transformer矫正特征（平均角度: {angles.mean():.2f}°）")
                else:
                    print(f"    ⚠️ 无法获取关键点，跳过Transformer矫正")
            elif transformer_model is not None:
                print(f"    ⚠️ 未提供参考图像，跳过Transformer矫正")
            
            # 检查特征数据库是否为空
            if feature_count == 0:
                print(f"    提示: 特征数据库为空，无法进行比对")
                matched_names.append('数据库为空')
                matched_probs.append(0.0)
                results.append({
                    'box': [x1, y1, x2, y2],
                    'prob': prob,
                    'name': '数据库为空',
                    'similarity': 0.0,
                    'match': None,
                    'landmarks': None,
                    'angles': None,
                    'avg_angle': None
                })
                continue
            
            # 检查特征维度是否匹配
            if feature_count > 0:
                features_db, _ = feature_manager.get_all_features()
                if features_db is not None:
                    db_dim = features_db.shape[1]
                    query_dim = len(features)
                    if db_dim != query_dim:
                        print(f"    错误: 特征维度不匹配！")
                        print(f"      查询特征维度: {query_dim} (DINOv2)")
                        print(f"      数据库特征维度: {db_dim}")
                        matched_names.append('维度不匹配')
                        matched_probs.append(0.0)
                        results.append({
                            'box': [x1, y1, x2, y2],
                            'prob': prob,
                            'name': '维度不匹配',
                            'similarity': 0.0,
                            'match': None,
                            'landmarks': None,
                            'angles': None,
                            'avg_angle': None
                        })
                        continue
            
            # 比对特征（使用余弦相似度，阈值0.25）
            # 直接计算余弦相似度，不使用归一化版本
            features_db, metadata_db = feature_manager.get_all_features()
            if features_db is not None and len(features_db) > 0:
                # 打印特征库信息（仅第一次）
                if not hasattr(process_video_frame, '_printed_db_info'):
                    print(f"\n  [调试] 特征库信息:")
                    print(f"    总特征数: {len(features_db)}")
                    unique_names = set(m.get('person_name', 'Unknown') for m in metadata_db)
                    print(f"    唯一人员数: {len(unique_names)}")
                    print(f"    人员列表: {sorted(unique_names)}")
                    process_video_frame._printed_db_info = True
                
                # 计算与所有特征的余弦相似度
                query_feat_norm = features / (np.linalg.norm(features) + 1e-8)
                db_feats_norm = features_db / (np.linalg.norm(features_db, axis=1, keepdims=True) + 1e-8)
                cosine_similarities = np.dot(db_feats_norm, query_feat_norm)
                
                # 找到最高相似度
                best_idx = np.argmax(cosine_similarities)
                best_similarity = float(cosine_similarities[best_idx])
                
                # 打印前5个最高相似度（用于调试）
                top5_indices = np.argsort(cosine_similarities)[::-1][:5]
                print(f"    [调试] 前5个最高相似度:")
                for rank, idx in enumerate(top5_indices, 1):
                    name = metadata_db[idx].get('person_name', 'Unknown')
                    sim = cosine_similarities[idx]
                    print(f"      {rank}. {name}: {sim:.4f}")
                
                # 检查是否超过阈值
                if best_similarity >= similarity_threshold:
                    matches = [{
                        'index': int(best_idx),
                        'similarity': best_similarity,
                        'metadata': metadata_db[best_idx]
                    }]
                else:
                    matches = []
            else:
                matches = []
            
            # 检测关键点并计算角度（如果提供了参考图像）
            # InsightFace 已经检测到了关键点，直接使用
            landmarks = None
            angles = None
            avg_angle = None
            
            if hasattr(face_info, 'kps') and face_info.kps is not None:
                landmarks = face_info.kps  # [5, 2]
                
                if reference_landmarks_3d is not None:
                    # 计算角度
                    try:
                        # 获取当前图像尺寸
                        img_width, img_height = pil_image.size
                        
                        # 转换为3D坐标
                        landmarks_3d = landmarks_to_3d(landmarks, box, img_width, img_height)
                        
                        # 计算角度
                        angles, avg_angle = calculate_spherical_angle(
                            reference_landmarks_3d, landmarks_3d,
                            reference_landmarks_2d, landmarks
                        )
                        
                        direction = "抬头" if avg_angle > 0 else "低头" if avg_angle < 0 else "平视"
                        print(f"    关键点检测成功，平均角度: {avg_angle:+.2f}° ({direction})")
                    except Exception as e:
                        print(f"    角度计算出错: {e}")
                else:
                    print(f"    关键点检测成功（未提供参考图像，无法计算角度）")
            
            if matches:
                best_match = matches[0]
                name = best_match['metadata'].get('person_name', '未知')
                similarity = best_match['similarity']
                matched_names.append(name)
                matched_probs.append(similarity)
                
                results.append({
                    'box': [x1, y1, x2, y2],
                    'prob': prob,
                    'name': name,
                    'similarity': similarity,
                    'match': best_match,
                    'landmarks': landmarks,
                    'angles': angles,
                    'avg_angle': avg_angle
                })
                print(f"    匹配: {name} (余弦相似度: {similarity:.4f})")
            else:
                matched_names.append('未识别')
                matched_probs.append(0.0)
                results.append({
                    'box': [x1, y1, x2, y2],
                    'prob': prob,
                    'name': '未识别',
                    'similarity': 0.0,
                    'match': None,
                    'landmarks': landmarks,
                    'angles': angles,
                    'avg_angle': avg_angle
                })
                print(f"    未找到匹配（余弦相似度低于阈值 {similarity_threshold}）")
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"    特征提取失败: {error_msg}")
            print(f"    错误详情:")
            traceback.print_exc()
            matched_names.append('检测失败')
            matched_probs.append(0.0)
            results.append({
                'box': [x1, y1, x2, y2],
                'prob': prob,
                'name': '检测失败',
                'similarity': 0.0,
                'match': None,
                'landmarks': None,
                'angles': None,
                'avg_angle': None
            })
    
    # 处理识别结果：确保每个姓名只分配给置信度最高的检测框
    print("\n正在处理识别结果（去重）...")
    print(f"  [调试] 去重前: 检测到 {len(results)} 个人脸")
    recognized_before = sum(1 for r in results if r.get('name', '未识别') not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'})
    print(f"  [调试] 去重前: 成功识别 {recognized_before} 个人脸")
    processed_results = deduplicate_recognition_results(results, confidence_key='similarity')
    recognized_after = sum(1 for r in processed_results if r.get('name', '未识别') not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'})
    print(f"  [调试] 去重后: 成功识别 {recognized_after} 个人脸")
    unique_names_after = set(r.get('name', '未识别') for r in processed_results if r.get('name', '未识别') not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'})
    print(f"  [调试] 去重后: 识别出的人员: {sorted(unique_names_after)}")
    
    # 更新 matched_names 和 matched_probs
    matched_names = [r['name'] for r in processed_results]
    matched_probs = [r.get('similarity', 0.0) for r in processed_results]
    
    # 在图像上绘制检测框、姓名、关键点和角度
    print("\n正在绘制检测结果...")
    annotated_image = draw_recognition_results(
        pil_image,
        processed_results,
        show_landmarks=True,
        show_angles=True
    )
    
    # 保存结果
    if output_path:
        annotated_image.save(output_path)
        print(f"\n结果已保存到: {output_path}")
    else:
        # 默认保存路径
        output_path = 'output_annotated.jpg'
        annotated_image.save(output_path)
        print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    
    return annotated_image, results


def main():
    """主函数"""
    # 视频路径
    video_path = r"D:\Code\face000\datas\camera\5-6班人脸视频cut\刘子源\281.jpg"
    # video_path = r"C:\AIXLAB\DATA\video\101_2025-10-27-09-30-04_classroom.mp4"
    
    # 特征存储目录
    features_dir = 'features_224'
    
    # 输出路径
    output_path = 'video_frame_result224_t.jpg'
    
    # 相似度阈值设置（使用0.25）
    similarity_threshold = 0.25
    
    # 参考图像路径（用于计算角度）
    reference_image_path = r'C:\Codes\face000\train\datas\face\袁润东.jpg'
    
    # Transformer模型路径（用于特征矫正）
    transformer_model_path = 'train_transformer/checkpoints/best_model.pth'  # 可以修改为其他检查点
    
    # 要处理的帧号（从0开始，0表示第一帧）
    frame_number = 120  # 可以修改这个值来处理不同的帧
    
    # 处理视频指定帧
    annotated_image, results = process_video_frame(
        video_path=video_path,
        features_dir=features_dir,
        output_path=output_path,
        similarity_threshold=similarity_threshold,
        reference_image_path=reference_image_path,
        use_cpu=False,
        frame_number=frame_number,
        transformer_model_path=transformer_model_path
    )
    
    if annotated_image is not None:
        print(f"\n检测到 {len(results)} 个人脸")
        for i, result in enumerate(results, 1):
            print(f"\n人脸 {i}:")
            print(f"  位置: {result['box']}")
            print(f"  人脸检测置信度: {result['prob']:.4f} (InsightFace检测)")
            print(f"  姓名: {result['name']}")
            if result['match']:
                print(f"  特征比对相似度: {result['similarity']:.4f} (与特征库的余弦相似度)")
            elif result['name'] == '未识别':
                print(f"  特征比对相似度: {result.get('similarity', 0.0):.4f} (低于阈值 {similarity_threshold})")
            if result.get('avg_angle') is not None:
                direction = "抬头" if result['avg_angle'] > 0 else "低头" if result['avg_angle'] < 0 else "平视"
                print(f"  平均角度: {result['avg_angle']:+.2f}° ({direction})")
    else:
        print("处理失败")


if __name__ == '__main__':
    main()

