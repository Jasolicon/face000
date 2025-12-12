"""
3D增强的人脸检测和识别实用函数
使用InsightFace提取特征和3D关键点，通过TransformerDecoderOnly3D模型矫正特征
与正面人脸特征库比对，并进行去重处理
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
import logging

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils import (
    get_insightface_detector,
    deduplicate_recognition_results,
    draw_recognition_results
)
from feature_manager import FeatureManager
from train_transformer3D.utils_3d import get_3d_landmarks_and_pose
from train_transformer3D.models_3d import TransformerDecoderOnly3D
from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_3d_model(model_path, device=None, **model_kwargs):
    """
    加载训练好的3D Transformer模型
    
    Args:
        model_path: 模型检查点路径
        device: 计算设备（默认自动选择）
        **model_kwargs: 模型参数（如果检查点中没有保存）
    
    Returns:
        model: 加载的模型
        device: 使用的设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    logger.info(f"加载3D模型: {model_path}")
    logger.info(f"使用设备: {device}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数（从检查点或kwargs）
    if 'args' in checkpoint:
        args = checkpoint['args']
        d_model = args.get('d_model', model_kwargs.get('d_model', 512))
        nhead = args.get('nhead', model_kwargs.get('nhead', 8))
        num_layers = args.get('num_layers', model_kwargs.get('num_layers', 4))
        dim_feedforward = args.get('dim_feedforward', model_kwargs.get('dim_feedforward', 2048))
        dropout = args.get('dropout', model_kwargs.get('dropout', 0.1))
        num_keypoints = args.get('num_keypoints', model_kwargs.get('num_keypoints', 5))
        pose_dim = args.get('pose_dim', model_kwargs.get('pose_dim', 3))
        use_spatial_attention = args.get('use_spatial_attention', model_kwargs.get('use_spatial_attention', False))
        use_pose_attention = args.get('use_pose_attention', model_kwargs.get('use_pose_attention', False))
    else:
        # 使用默认值或kwargs
        d_model = model_kwargs.get('d_model', 512)
        nhead = model_kwargs.get('nhead', 8)
        num_layers = model_kwargs.get('num_layers', 4)
        dim_feedforward = model_kwargs.get('dim_feedforward', 2048)
        dropout = model_kwargs.get('dropout', 0.1)
        num_keypoints = model_kwargs.get('num_keypoints', 5)
        pose_dim = model_kwargs.get('pose_dim', 3)
        use_spatial_attention = model_kwargs.get('use_spatial_attention', False)
        use_pose_attention = model_kwargs.get('use_pose_attention', False)
    
    # 创建模型
    model = TransformerDecoderOnly3D(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_keypoints=num_keypoints,
        pose_dim=pose_dim,
        use_spatial_attention=use_spatial_attention,
        use_pose_attention=use_pose_attention,
        use_angle_pe=True,
        use_angle_conditioning=True
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    logger.info("✓ 3D模型加载成功")
    logger.info(f"  模型参数: d_model={d_model}, num_keypoints={num_keypoints}, pose_dim={pose_dim}")
    
    return model, device


def correct_features_with_3d_model(model, features, keypoints_3d, pose, device):
    """
    使用3D模型矫正特征
    
    Args:
        model: TransformerDecoderOnly3D模型
        features: 输入特征 [feature_dim] 或 [batch, feature_dim]
        keypoints_3d: 3D关键点 [num_keypoints, 3] 或 [batch, num_keypoints, 3]
        pose: 姿态向量 [pose_dim] 或 [batch, pose_dim] (欧拉角)
        device: 计算设备
    
    Returns:
        corrected_features: 矫正后的特征
    """
    # 转换为tensor
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if isinstance(keypoints_3d, np.ndarray):
        keypoints_3d = torch.from_numpy(keypoints_3d).float()
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose).float()
    
    # 确保是2D tensor [batch, ...]
    if features.dim() == 1:
        features = features.unsqueeze(0)
    if keypoints_3d.dim() == 2:
        keypoints_3d = keypoints_3d.unsqueeze(0)
    if pose.dim() == 1:
        pose = pose.unsqueeze(0)
    
    # 移动到设备
    features = features.to(device)
    keypoints_3d = keypoints_3d.to(device)
    pose = pose.to(device)
    
    # 前向传播
    with torch.no_grad():
        # angles用于兼容性（使用pose）
        angles = pose.clone()
        corrected_features = model(
            src=features,
            angles=angles,
            keypoints_3d=keypoints_3d,
            pose=pose,
            return_residual=False  # 返回完整特征，不是残差
        )
    
    # 转换回numpy
    if corrected_features.dim() == 2 and corrected_features.size(0) == 1:
        corrected_features = corrected_features.squeeze(0)
    corrected_features = corrected_features.cpu().numpy()
    
    # L2归一化
    norm = np.linalg.norm(corrected_features)
    if norm > 0:
        corrected_features = corrected_features / norm
    
    return corrected_features


def detect_faces_3d(
    input_path,
    features_dir='features',
    model_path=None,
    similarity_threshold=0.25,
    frame_number=None,
    use_cpu=False,
    output_path=None,
    show_landmarks=True,
    show_angles=True
):
    """
    完整的人脸检测和识别函数（3D增强版本）
    
    Args:
        input_path: 输入路径（视频文件或图片文件）
        features_dir: 正面人脸特征库目录
        model_path: 训练好的3D模型路径（可选，如果提供则进行特征矫正）
        similarity_threshold: 相似度阈值（默认0.25）
        frame_number: 视频帧号（仅对视频有效，None表示第一帧）
        use_cpu: 是否使用CPU
        output_path: 输出图像路径（可选）
        show_landmarks: 是否显示关键点
        show_angles: 是否显示角度信息
    
    Returns:
        annotated_image: 标注后的图像（PIL Image）
        results: 检测和识别结果列表
    """
    logger.info("=" * 70)
    logger.info("3D增强的人脸检测和识别")
    logger.info("=" * 70)
    
    # 检查输入文件
    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        return None, None
    
    # 判断是视频还是图片
    is_video = input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    # 读取图像
    if is_video:
        logger.info(f"读取视频: {input_path}")
        if frame_number is None:
            frame_number = 0
        logger.info(f"处理第 {frame_number} 帧")
        
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error("无法打开视频文件")
            return None, None
        
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"视频总帧数: {total_frames}")
        
        # 检查帧号
        if frame_number < 0:
            frame_number = 0
        elif frame_number >= total_frames:
            frame_number = total_frames - 1
        
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logger.error(f"无法读取视频第 {frame_number} 帧")
            return None, None
        
        # 转换为PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
    else:
        logger.info(f"读取图片: {input_path}")
        try:
            pil_image = Image.open(input_path).convert('RGB')
        except Exception as e:
            logger.error(f"无法读取图片: {e}")
            return None, None
    
    logger.info(f"图像尺寸: {pil_image.size}")
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载3D模型（如果提供）
    model = None
    if model_path and Path(model_path).exists():
        try:
            model, device = load_3d_model(model_path, device=device)
            logger.info("✓ 3D模型已加载，将用于特征矫正")
        except Exception as e:
            logger.warning(f"3D模型加载失败: {e}")
            logger.warning("将跳过特征矫正步骤")
            model = None
    elif model_path:
        logger.warning(f"3D模型文件不存在: {model_path}")
        logger.warning("将跳过特征矫正步骤")
    
    # 加载特征数据库
    logger.info(f"加载特征数据库: {features_dir}")
    feature_manager = FeatureManager(storage_dir=features_dir)
    feature_count = feature_manager.get_count()
    logger.info(f"已加载 {feature_count} 个特征")
    
    if feature_count == 0:
        logger.warning("特征数据库为空，无法进行比对")
    
    # 初始化InsightFace检测器
    logger.info("初始化InsightFace检测器...")
    detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 检测人脸
    logger.info("检测人脸...")
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    detected_faces = detector.get(img_bgr)
    
    if len(detected_faces) == 0:
        logger.warning("未检测到人脸")
        return pil_image, []
    
    logger.info(f"检测到 {len(detected_faces)} 个人脸")
    
    # 处理每个人脸
    logger.info("提取特征并比对...")
    results = []
    
    for i, face_info in enumerate(detected_faces):
        logger.info(f"处理第 {i+1}/{len(detected_faces)} 个人脸...")
        
        # 获取边界框
        box = face_info.bbox  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # 获取置信度
        prob = face_info.det_score if hasattr(face_info, 'det_score') else 1.0
        
        # 提取特征（InsightFace的normed_embedding，512维）
        try:
            features = face_info.normed_embedding  # [512] 已归一化
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            logger.info(f"  特征提取成功，维度: {len(features)}")
        except Exception as e:
            logger.error(f"  特征提取失败: {e}")
            results.append({
                'box': [x1, y1, x2, y2],
                'prob': prob,
                'name': '特征提取失败',
                'similarity': 0.0,
                'match': None,
                'landmarks': None,
                'angles': None,
                'avg_angle': None
            })
            continue
        
        # 提取3D关键点和姿态
        landmarks_2d = None
        landmarks_3d = None
        pose = None
        
        if hasattr(face_info, 'kps') and face_info.kps is not None:
            landmarks_2d = face_info.kps  # [5, 2]
            
            # 使用utils_3d提取3D关键点和姿态
            # 直接使用已检测到的关键点，避免重复检测
            try:
                from train_transformer3D.utils_3d import estimate_pose_from_landmarks
                img_width, img_height = pil_image.size
                
                landmarks_3d_result, _, _, euler_angles, _ = estimate_pose_from_landmarks(
                    landmarks_2d, box, img_width, img_height
                )
                
                if landmarks_3d_result is not None and euler_angles is not None:
                    landmarks_3d = landmarks_3d_result
                    pose = euler_angles  # [3] (yaw, pitch, roll)
                    logger.info(f"  3D关键点和姿态提取成功")
                    logger.info(f"    姿态: yaw={pose[0]:.2f}°, pitch={pose[1]:.2f}°, roll={pose[2]:.2f}°")
                else:
                    logger.warning(f"  无法提取3D关键点和姿态")
            except Exception as e:
                logger.warning(f"  3D关键点和姿态提取失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # 使用3D模型矫正特征（如果模型已加载且关键点可用）
        if model is not None and landmarks_3d is not None and pose is not None:
            try:
                features = correct_features_with_3d_model(
                    model, features, landmarks_3d, pose, device
                )
                logger.info(f"  ✓ 已使用3D模型矫正特征")
            except Exception as e:
                logger.warning(f"  特征矫正失败: {e}")
        elif model is not None:
            logger.warning(f"  缺少3D关键点或姿态，跳过特征矫正")
        
        # 与特征库比对
        name = '未识别'
        similarity = 0.0
        match = None
        
        if feature_count > 0:
            try:
                # 获取所有特征
                features_db, metadata_db = feature_manager.get_all_features()
                
                if features_db is not None and len(features_db) > 0:
                    # 检查特征维度
                    db_dim = features_db.shape[1]
                    query_dim = len(features)
                    
                    if db_dim != query_dim:
                        logger.warning(f"  特征维度不匹配: 查询={query_dim}, 数据库={db_dim}")
                        results.append({
                            'box': [x1, y1, x2, y2],
                            'prob': prob,
                            'name': '维度不匹配',
                            'similarity': 0.0,
                            'match': None,
                            'landmarks': landmarks_2d,
                            'angles': None,
                            'avg_angle': None
                        })
                        continue
                    
                    # 计算余弦相似度
                    query_feat_norm = features / (np.linalg.norm(features) + 1e-8)
                    db_feats_norm = features_db / (np.linalg.norm(features_db, axis=1, keepdims=True) + 1e-8)
                    cosine_similarities = np.dot(db_feats_norm, query_feat_norm)
                    
                    # 找到最高相似度
                    best_idx = np.argmax(cosine_similarities)
                    best_similarity = float(cosine_similarities[best_idx])
                    
                    # 检查是否超过阈值
                    if best_similarity >= similarity_threshold:
                        name = metadata_db[best_idx].get('person_name', '未知')
                        similarity = best_similarity
                        match = {
                            'index': int(best_idx),
                            'similarity': best_similarity,
                            'metadata': metadata_db[best_idx]
                        }
                        logger.info(f"  匹配: {name} (相似度: {similarity:.4f})")
                    else:
                        logger.info(f"  未找到匹配 (最高相似度: {best_similarity:.4f} < 阈值: {similarity_threshold})")
            except Exception as e:
                logger.error(f"  特征比对失败: {e}")
        
        # 计算平均角度（如果有姿态）
        avg_angle = None
        if pose is not None:
            # 使用yaw角度的绝对值作为平均角度
            avg_angle = float(np.abs(pose[0]))  # yaw角度
        
        # 保存结果
        results.append({
            'box': [x1, y1, x2, y2],
            'prob': prob,
            'name': name,
            'similarity': similarity,
            'match': match,
            'landmarks': landmarks_2d,
            'angles': pose,
            'avg_angle': avg_angle
        })
    
    # 去重处理
    logger.info("处理识别结果（去重）...")
    processed_results = deduplicate_recognition_results(results, confidence_key='similarity')
    
    # 绘制结果
    logger.info("绘制检测结果...")
    annotated_image = draw_recognition_results(
        pil_image,
        processed_results,
        show_landmarks=show_landmarks,
        show_angles=show_angles
    )
    
    # 保存结果
    if output_path:
        annotated_image.save(output_path)
        logger.info(f"结果已保存到: {output_path}")
    
    logger.info("=" * 70)
    logger.info("处理完成！")
    logger.info("=" * 70)
    
    return annotated_image, processed_results


def main():
    """主函数 - 使用示例"""
    # 输入路径（可以是视频或图片）
    input_path = r"D:\Code\face000\datas\camera\5-6班人脸视频cut\刘子源\281.jpg"
    # input_path = r"C:\AIXLAB\DATA\video\101_2025-10-27-09-30-04_classroom.mp4"  # 视频示例
    
    # 特征库目录
    features_dir = 'features_224'
    
    # 3D模型路径（可选）
    model_path = 'train_transformer3D/checkpoints/best_model.pth'
    
    # 相似度阈值
    similarity_threshold = 0.25
    
    # 视频帧号（仅对视频有效）
    frame_number = 0  # None表示第一帧
    
    # 输出路径
    output_path = 'face_detection_3d_result.jpg'
    
    # 执行检测
    annotated_image, results = detect_faces_3d(
        input_path=input_path,
        features_dir=features_dir,
        model_path=model_path,
        similarity_threshold=similarity_threshold,
        frame_number=frame_number,
        use_cpu=False,
        output_path=output_path,
        show_landmarks=True,
        show_angles=True
    )
    
    # 打印结果
    if annotated_image is not None:
        logger.info(f"\n检测到 {len(results)} 个人脸")
        for i, result in enumerate(results, 1):
            logger.info(f"\n人脸 {i}:")
            logger.info(f"  位置: {result['box']}")
            logger.info(f"  检测置信度: {result['prob']:.4f}")
            logger.info(f"  姓名: {result['name']}")
            logger.info(f"  相似度: {result['similarity']:.4f}")
            if result.get('avg_angle') is not None:
                logger.info(f"  平均角度: {result['avg_angle']:.2f}°")
    else:
        logger.error("处理失败")


if __name__ == '__main__':
    main()
