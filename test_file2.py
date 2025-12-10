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
        
        # 默认使用 DINOv2 ViT-B/14（768维），如果失败则使用 vit_base_patch16_224 作为备用
        # 注意：Transformer模型是用768维训练的，所以测试时必须使用dinov2_vitb14
        self.model_name = model_name
        self.use_dinov2 = False
        
        try:
            print(f"  尝试从 torch.hub 加载 DINOv2 {model_name}...")
            # 使用 torch.hub 加载 DINOv2 模型
            self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            self.use_dinov2 = True
            feature_dim = 384 if model_name == 'dinov2_vits14' else 768 if model_name == 'dinov2_vitb14' else 1024 if model_name == 'dinov2_vitl14' else 1536
            print(f"  ✓ 成功加载 DINOv2 {model_name} 模型（特征维度: {feature_dim}）")
            if model_name != 'dinov2_vitb14':
                print(f"  ⚠️  警告: Transformer模型是用768维训练的，当前使用{feature_dim}维模型可能导致维度不匹配！")
        except Exception as e:
            print(f"  ✗ 无法加载 DINOv2 {model_name}: {str(e)}")
            print(f"  提示: 运行 'python download_dinov2.py --model {model_name}' 可以提前下载模型")
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
    # 注意：必须使用 dinov2_vitb14 (768维)，因为Transformer模型是用768维训练的
    feature_extractor = DINOv2FeatureExtractor(model_name='dinov2_vitb14', device=device)
    
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
            # 注意：大脸可能需要更好的预处理，确保裁剪区域质量
            original_features = feature_extractor.extract_features(face_crop)
            face_width, face_height = face_crop.size
            print(f"    特征提取成功，特征维度: {len(original_features)}, 人脸尺寸: {face_width}x{face_height}")
            
            # 初始化矫正后的特征（默认使用原始特征）
            corrected_features = original_features.copy()
            use_transformer = False
            angles = None
            
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
                    corrected_features = correct_features_with_transformer(
                        transformer_model, 
                        original_features, 
                        angles, 
                        device=device
                    )
                    use_transformer = True
                    print(f"    ✓ 已使用Transformer矫正特征（平均角度: {angles.mean():.2f}°）")
                else:
                    print(f"    ⚠️ 无法获取关键点，跳过Transformer矫正")
            elif transformer_model is not None:
                print(f"    ⚠️ 未提供参考图像，跳过Transformer矫正")
            
            # 使用矫正后的特征进行后续处理
            features = corrected_features
            
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
            
            # 比对特征：同时使用原始特征和矫正后特征的相似度
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
                
                # 计算原始特征的相似度
                original_feat_norm = original_features / (np.linalg.norm(original_features) + 1e-8)
                db_feats_norm = features_db / (np.linalg.norm(features_db, axis=1, keepdims=True) + 1e-8)
                original_similarities = np.dot(db_feats_norm, original_feat_norm)
                
                # 计算矫正后特征的相似度
                corrected_feat_norm = corrected_features / (np.linalg.norm(corrected_features) + 1e-8)
                corrected_similarities = np.dot(db_feats_norm, corrected_feat_norm)
                
                # 综合相似度：加权平均或取最大值
                if use_transformer:
                    if similarity_fusion == 'max':
                        # 取最大值：选择更可信的相似度
                        combined_similarities = np.maximum(original_similarities, corrected_similarities)
                    elif similarity_fusion == 'weighted':
                        # 加权平均：原始0.3，矫正后0.7
                        combined_similarities = original_similarities * 0.3 + corrected_similarities * 0.7
                    else:
                        # 默认使用最大值
                        combined_similarities = np.maximum(original_similarities, corrected_similarities)
                    
                    print(f"    [调试] 原始特征最高相似度: {original_similarities.max():.4f}")
                    print(f"    [调试] 矫正后特征最高相似度: {corrected_similarities.max():.4f}")
                else:
                    # 如果没有使用Transformer，只使用原始特征
                    combined_similarities = original_similarities
                
                # 找到最高综合相似度
                best_idx = np.argmax(combined_similarities)
                best_similarity = float(combined_similarities[best_idx])
                original_sim = float(original_similarities[best_idx])
                corrected_sim = float(corrected_similarities[best_idx]) if use_transformer else original_sim
                
                # 打印前5个最高相似度（用于调试）
                top5_indices = np.argsort(combined_similarities)[::-1][:5]
                print(f"    [调试] 前5个最高综合相似度:")
                for rank, idx in enumerate(top5_indices, 1):
                    name = metadata_db[idx].get('person_name', 'Unknown')
                    sim = combined_similarities[idx]
                    orig_sim = original_similarities[idx]
                    corr_sim = corrected_similarities[idx] if use_transformer else orig_sim
                    if use_transformer:
                        print(f"      {rank}. {name}: 综合={sim:.4f} (原始={orig_sim:.4f}, 矫正={corr_sim:.4f})")
                    else:
                        print(f"      {rank}. {name}: {sim:.4f}")
                
                # 检查是否超过阈值
                if best_similarity >= similarity_threshold:
                    matches = [{
                        'index': int(best_idx),
                        'similarity': best_similarity,
                        'original_similarity': original_sim,
                        'corrected_similarity': corrected_sim if use_transformer else original_sim,
                        'metadata': metadata_db[best_idx]
                    }]
                    print(f"    ✓ 匹配成功: {metadata_db[best_idx].get('person_name', '未知')} (综合相似度: {best_similarity:.4f} >= 阈值: {similarity_threshold})")
                else:
                    matches = []
                    print(f"    ✗ 匹配失败: 最高综合相似度 {best_similarity:.4f} < 阈值 {similarity_threshold}")
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
                
                # 计算检测框大小（用于调试）
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                results.append({
                    'box': [x1, y1, x2, y2],
                    'prob': prob,
                    'name': name,
                    'similarity': similarity,
                    'match': best_match,
                    'landmarks': landmarks,
                    'angles': angles,
                    'avg_angle': avg_angle,
                    'box_area': box_area  # 添加检测框面积用于调试
                })
                angle_info = f", 角度: {avg_angle:+.2f}°" if avg_angle is not None else ""
                print(f"    匹配: {name} (相似度: {similarity:.4f}, 框大小: {box_width}x{box_height}={box_area}{angle_info})")
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
    
    # 打印去重前的详细信息
    print(f"\n  [调试] 去重前详细信息:")
    for i, r in enumerate(results):
        name = r.get('name', '未识别')
        if name not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'}:
            box = r.get('box', [0, 0, 0, 0])
            box_area = r.get('box_area', (box[2]-box[0])*(box[3]-box[1]))
            similarity = r.get('similarity', 0.0)
            avg_angle = r.get('avg_angle')
            angle_str = f", 角度: {avg_angle:+.2f}°" if avg_angle is not None else ""
            print(f"    框 {i}: {name} - 相似度: {similarity:.4f}, 框大小: {box_area}{angle_str}")
    
    processed_results = deduplicate_recognition_results(results, confidence_key='similarity')
    recognized_after = sum(1 for r in processed_results if r.get('name', '未识别') not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'})
    print(f"\n  [调试] 去重后: 成功识别 {recognized_after} 个人脸")
    unique_names_after = set(r.get('name', '未识别') for r in processed_results if r.get('name', '未识别') not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'})
    print(f"  [调试] 去重后: 识别出的人员: {sorted(unique_names_after)}")
    
    # 打印去重后的详细信息
    print(f"\n  [调试] 去重后详细信息:")
    for i, r in enumerate(processed_results):
        name = r.get('name', '未识别')
        box = r.get('box', [0, 0, 0, 0])
        box_area = r.get('box_area', (box[2]-box[0])*(box[3]-box[1]))
        similarity = r.get('similarity', 0.0)
        avg_angle = r.get('avg_angle')
        angle_str = f", 角度: {avg_angle:+.2f}°" if avg_angle is not None else ""
        status = "✓" if name not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'} else "✗"
        print(f"    {status} 框 {i}: {name} - 相似度: {similarity:.4f}, 框大小: {box_area}{angle_str}")
    
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


def process_video_segment(video_path, features_dir='features', output_path=None, similarity_threshold=0.25, 
                         reference_image_path=None, use_cpu=False, start_frame=0, duration_seconds=5,
                         transformer_model_path=None, similarity_fusion='max'):
    """
    处理视频片段：检测人脸、比对特征、标注结果，并保存为视频
    
    Args:
        video_path: 视频文件路径
        features_dir: 特征存储目录
        output_path: 输出视频路径，如果为None则自动生成
        similarity_threshold: 相似度阈值（默认0.25）
        reference_image_path: 参考图像路径（用于计算角度，可选）
        use_cpu: 是否使用 CPU
        start_frame: 起始帧号（从0开始，默认0）
        duration_seconds: 处理时长（秒，默认5秒）
        transformer_model_path: Transformer模型路径（用于特征矫正，可选）
        
    Returns:
        output_path: 输出视频路径
    """
    print("=" * 70)
    print(f"处理视频片段：从第 {start_frame} 帧开始，持续 {duration_seconds} 秒")
    print("=" * 70)
    
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return None
    
    print(f"视频路径: {video_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件")
        return None
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n视频信息:")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  分辨率: {width}x{height}")
    
    # 计算要处理的帧数
    frames_to_process = int(fps * duration_seconds)
    end_frame = min(start_frame + frames_to_process, total_frames)
    actual_duration = (end_frame - start_frame) / fps if fps > 0 else 0
    
    print(f"\n处理范围:")
    print(f"  起始帧: {start_frame}")
    print(f"  结束帧: {end_frame}")
    print(f"  处理帧数: {end_frame - start_frame}")
    print(f"  实际时长: {actual_duration:.2f} 秒")
    
    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 初始化组件（只初始化一次，提高效率）
    print("\n正在初始化组件...")
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f"  使用设备: {device}")
    
    # DINOv2 特征提取器
    print("  初始化 DINOv2 特征提取器...")
    feature_extractor = DINOv2FeatureExtractor(model_name='dinov2_vitb14', device=device)
    
    # Transformer模型
    transformer_model = None
    if transformer_model_path and Path(transformer_model_path).exists():
        print("  加载Transformer模型...")
        try:
            transformer_model, _ = load_transformer_model(transformer_model_path, device=device)
            print("  ✓ Transformer模型已加载")
        except Exception as e:
            print(f"  ⚠️ Transformer模型加载失败: {e}")
    else:
        if transformer_model_path:
            print(f"  ⚠️ Transformer模型文件不存在: {transformer_model_path}")
    
    # 特征数据库
    print("  加载特征数据库...")
    feature_manager = FeatureManager(storage_dir=features_dir)
    feature_count = feature_manager.get_count()
    print(f"  ✓ 已加载 {feature_count} 个特征")
    
    # InsightFace 检测器
    print("  初始化 InsightFace 检测器...")
    insightface_detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 参考图像和关键点
    reference_landmarks_3d = None
    reference_landmarks_2d = None
    reference_box = None
    reference_img_size = None
    if reference_image_path and os.path.exists(reference_image_path):
        print(f"  加载参考图像: {reference_image_path}")
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
    
    # 创建视频写入器
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f'video_segment_{video_name}_frame{start_frame}_{duration_seconds}s.mp4'
    
    print(f"\n输出视频路径: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"错误: 无法创建输出视频文件: {output_path}")
        cap.release()
        return None
    
    # 处理每一帧
    print(f"\n开始处理 {end_frame - start_frame} 帧...")
    frame_count = 0
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            print(f"  警告: 无法读取第 {frame_idx} 帧，停止处理")
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  处理进度: {frame_count}/{end_frame - start_frame} 帧 ({frame_count/(end_frame-start_frame)*100:.1f}%)")
        
        # 转换为PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 检测人脸
        img_array = np.array(pil_image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        detected_faces = insightface_detector.get(img_bgr)
        
        if len(detected_faces) == 0:
            # 没有检测到人脸，直接写入原帧
            out.write(frame)
            continue
        
        # 处理每个人脸
        results = []
        for face_info in detected_faces:
            box = face_info.bbox
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # 裁剪人脸区域
            img_width, img_height = pil_image.size
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            face_crop = pil_image.crop((x1, y1, x2, y2))
            
            try:
                # 提取原始特征
                original_features = feature_extractor.extract_features(face_crop)
                corrected_features = original_features.copy()
                use_transformer = False
                
                # Transformer矫正特征
                if transformer_model is not None and reference_landmarks_3d is not None:
                    if hasattr(face_info, 'kps') and face_info.kps is not None:
                        video_landmarks = face_info.kps
                        video_width = x2 - x1
                        video_height = y2 - y1
                        video_landmarks_3d = landmarks_to_3d(video_landmarks, box, video_width, video_height)
                        angles, _ = calculate_spherical_angle(
                            reference_landmarks_3d, video_landmarks_3d,
                            reference_landmarks_2d, video_landmarks
                        )
                        corrected_features = correct_features_with_transformer(
                            transformer_model, original_features, angles, device=device
                        )
                        use_transformer = True
                
                # 比对特征：使用原始特征和矫正后特征的融合相似度
                landmarks = face_info.kps if hasattr(face_info, 'kps') and face_info.kps is not None else None
                angles = None
                avg_angle = None
                
                if reference_landmarks_3d is not None and landmarks is not None:
                    landmarks_3d = landmarks_to_3d(landmarks, box, img_width, img_height)
                    angles, avg_angle = calculate_spherical_angle(
                        reference_landmarks_3d, landmarks_3d,
                        reference_landmarks_2d, landmarks
                    )
                
                prob = face_info.det_score if hasattr(face_info, 'det_score') else 1.0
                name = '未识别'
                similarity = 0.0
                match = None
                
                if feature_count > 0:
                    features_db, metadata_db = feature_manager.get_all_features()
                    if features_db is not None and len(features_db) > 0:
                        # 计算原始特征的相似度
                        original_feat_norm = original_features / (np.linalg.norm(original_features) + 1e-8)
                        db_feats_norm = features_db / (np.linalg.norm(features_db, axis=1, keepdims=True) + 1e-8)
                        original_similarities = np.dot(db_feats_norm, original_feat_norm)
                        
                        # 计算矫正后特征的相似度
                        corrected_feat_norm = corrected_features / (np.linalg.norm(corrected_features) + 1e-8)
                        corrected_similarities = np.dot(db_feats_norm, corrected_feat_norm)
                        
                        # 融合相似度
                        if use_transformer:
                            if similarity_fusion == 'max':
                                combined_similarities = np.maximum(original_similarities, corrected_similarities)
                            elif similarity_fusion == 'weighted':
                                combined_similarities = original_similarities * 0.3 + corrected_similarities * 0.7
                            else:
                                combined_similarities = np.maximum(original_similarities, corrected_similarities)
                        else:
                            combined_similarities = original_similarities
                        
                        best_idx = np.argmax(combined_similarities)
                        best_similarity = float(combined_similarities[best_idx])
                        original_sim = float(original_similarities[best_idx])
                        corrected_sim = float(corrected_similarities[best_idx]) if use_transformer else original_sim
                        
                        if best_similarity >= similarity_threshold:
                            name = metadata_db[best_idx].get('person_name', '未知')
                            similarity = best_similarity
                            match = {
                                'index': int(best_idx),
                                'similarity': best_similarity,
                                'original_similarity': original_sim,
                                'corrected_similarity': corrected_sim,
                                'metadata': metadata_db[best_idx]
                            }
                
                # 计算检测框大小
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                results.append({
                    'box': [x1, y1, x2, y2],
                    'prob': prob,
                    'name': name,
                    'similarity': similarity,
                    'match': match,
                    'landmarks': landmarks,
                    'angles': angles,
                    'avg_angle': avg_angle,
                    'box_area': box_area  # 添加检测框面积用于去重
                })
            except Exception as e:
                # 处理失败，跳过这个人脸
                continue
        
        # 去重处理
        processed_results = deduplicate_recognition_results(results, confidence_key='similarity')
        
        # 绘制结果
        annotated_image = draw_recognition_results(
            pil_image,
            processed_results,
            show_landmarks=True,
            show_angles=True
        )
        
        # 转换为BGR格式并写入视频
        annotated_array = np.array(annotated_image)
        annotated_bgr = cv2.cvtColor(annotated_array, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"\n" + "=" * 70)
    print(f"处理完成！")
    print(f"  处理帧数: {frame_count}")
    print(f"  输出视频: {output_path}")
    print("=" * 70)
    
    return output_path


def main():
    """主函数"""
    # 视频路径
    video_path = r'C:\AIXLAB\video_batch_deployment\temp_data\垂杨柳上课视频 (2).mp4'
    # video_path = r"C:\AIXLAB\DATA\video\101_2025-10-27-09-30-04_classroom.mp4"
    
    # 特征存储目录
    features_dir = 'features_224'
    
    # 输出路径（视频文件）
    output_path = 'video_segment_result224_t.mp4'
    
    # 相似度阈值设置（使用0.25）
    similarity_threshold = 0.25
    
    # 参考图像路径（用于计算角度）
    reference_image_path = r'C:\Codes\face000\train\datas\face\袁润东.jpg'
    
    # Transformer模型路径（用于特征矫正）
    transformer_model_path = 'train_transformer/checkpoints/best_model.pth'  # 可以修改为其他检查点
    
    # 起始帧号（从0开始，0表示第一帧）
    start_frame = 120  # 可以修改这个值来处理不同的起始位置
    
    # 处理时长（秒）
    duration_seconds = 2  # 处理5秒视频
    
    # 相似度融合方式：'max'（取最大值）或 'weighted'（加权平均，原始0.3+矫正0.7）
    similarity_fusion = 'max'  # 默认使用最大值，选择更可信的相似度
    
    # 处理视频片段
    output_video_path = process_video_segment(
        video_path=video_path,
        features_dir=features_dir,
        output_path=output_path,
        similarity_threshold=similarity_threshold,
        reference_image_path=reference_image_path,
        use_cpu=False,
        start_frame=start_frame,
        duration_seconds=duration_seconds,
        transformer_model_path=transformer_model_path,
        similarity_fusion=similarity_fusion
    )
    
    if output_video_path:
        print(f"\n✓ 视频已保存到: {output_video_path}")
    else:
        print("\n✗ 处理失败")


if __name__ == '__main__':
    main()

