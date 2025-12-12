"""
提取3D关键点和姿态，并进行对齐和归一化
1. 处理正面图：提取3D关键点，计算平均标准位置
2. 处理视频帧：提取3D关键点，对齐到标准坐标系
3. 保存特征、关键点和元数据
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils import get_insightface_detector
from train_transformer3D.utils_3d import get_3d_landmarks_and_pose
from train_transformer.utils_seed import set_seed
from PIL import Image
import cv2

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Landmark3DAligner:
    """
    3D关键点对齐和归一化器
    """
    
    def __init__(self):
        self.standard_center = None  # 标准中心点 [3]
        self.standard_scale = None   # 标准尺度
        self.standard_landmarks = None  # 标准关键点位置 [5, 3]
    
    def compute_standard_from_faces(self, all_face_landmarks_3d: List[np.ndarray]) -> Dict:
        """
        从所有正面图计算标准坐标系
        
        Args:
            all_face_landmarks_3d: 所有正面图的3D关键点列表，每个 [5, 3]
        
        Returns:
            dict: 包含标准中心点、尺度、标准关键点的字典
        """
        if len(all_face_landmarks_3d) == 0:
            raise ValueError("没有正面图关键点数据")
        
        # 转换为numpy数组 [N, 5, 3]
        all_landmarks = np.array(all_face_landmarks_3d)
        logger.info(f"收集到 {len(all_landmarks)} 个正面图，关键点形状: {all_landmarks.shape}")
        
        # 计算每个关键点的平均位置 [5, 3]
        mean_landmarks = np.mean(all_landmarks, axis=0)
        
        # 计算中心点（所有关键点的平均位置）
        self.standard_center = np.mean(mean_landmarks, axis=0)  # [3]
        
        # 计算相对于中心点的标准坐标
        self.standard_landmarks = mean_landmarks - self.standard_center  # [5, 3]
        
        # 计算标准尺度（使用标准坐标的最大距离）
        distances = np.linalg.norm(self.standard_landmarks, axis=1)
        max_distance = np.max(distances)
        self.standard_scale = 1.0 / (max_distance + 1e-8)
        
        logger.info(f"标准中心点: {self.standard_center}")
        logger.info(f"标准尺度: {self.standard_scale:.6f}")
        logger.info(f"最大距离: {max_distance:.2f}")
        
        return {
            'standard_center': self.standard_center.tolist(),
            'standard_scale': float(self.standard_scale),
            'standard_landmarks': self.standard_landmarks.tolist(),
            'mean_landmarks': mean_landmarks.tolist(),
            'num_samples': len(all_landmarks)
        }
    
    def align_and_normalize(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        对齐和归一化3D关键点
        
        Args:
            landmarks_3d: 3D关键点 [5, 3]
        
        Returns:
            aligned_normalized: 对齐和归一化后的关键点 [5, 3]
        """
        if self.standard_center is None or self.standard_scale is None:
            raise ValueError("请先调用 compute_standard_from_faces 计算标准坐标系")
        
        # 对齐：减去标准中心点
        aligned = landmarks_3d - self.standard_center
        
        # 归一化：乘以标准尺度
        normalized = aligned * self.standard_scale
        
        return normalized


def extract_face_data(
    face_dir: str,
    output_dir: str,
    use_cpu: bool = False
) -> Dict:
    """
    提取正面图的3D关键点、姿态和特征（使用InsightFace提取特征）
    
    Args:
        face_dir: 正面图目录
        output_dir: 输出目录
        use_cpu: 是否使用CPU
    
    Returns:
        dict: 包含标准坐标系信息的字典
    """
    face_dir = Path(face_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("处理正面图")
    logger.info("=" * 70)
    
    # 初始化检测器（InsightFace同时用于检测和特征提取）
    logger.info("初始化InsightFace检测器（用于关键点和特征提取）...")
    detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 获取所有正面图
    face_images = sorted(list(face_dir.glob('*.jpg')))
    logger.info(f"找到 {len(face_images)} 张正面图")
    
    # 检查是否已有处理结果，加载已处理的人员列表
    front_metadata_path = output_dir / 'front_metadata.json'
    processed_persons = set()
    existing_data = None
    
    # 检查是否需要强制重新处理（通过检查函数参数或全局变量）
    force_reprocess = getattr(extract_face_data, '_force_reprocess', False)
    
    if front_metadata_path.exists() and not force_reprocess:
        logger.info(f"发现已有处理结果: {front_metadata_path}")
        try:
            with open(front_metadata_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            # 获取已处理的人员列表
            if 'metadata' in existing_data:
                processed_persons = {meta.get('person_name') for meta in existing_data['metadata']}
                logger.info(f"已处理 {len(processed_persons)} 个人员: {sorted(processed_persons)}")
        except Exception as e:
            logger.warning(f"无法加载已有数据: {e}，将重新处理所有数据")
            existing_data = None
            processed_persons = set()
    
    # 过滤出需要处理的人员（跳过已完整提取的）
    images_to_process = []
    skipped_count = 0
    
    for face_image_path in face_images:
        person_name = face_image_path.stem
        if person_name in processed_persons:
            skipped_count += 1
            logger.debug(f"跳过已处理的人员: {person_name}")
        else:
            images_to_process.append(face_image_path)
    
    if skipped_count > 0:
        logger.info(f"跳过 {skipped_count} 个已完整提取的人员")
    logger.info(f"需要处理 {len(images_to_process)} 张正面图")
    
    # 存储数据
    all_face_features = []
    all_face_landmarks_3d = []
    all_face_landmarks_2d = []
    all_face_poses = []
    all_face_boxes = []
    face_metadata = []
    
    # 如果已有数据，加载已有数据
    if existing_data is not None:
        try:
            front_feature_path = output_dir / 'front_feature.npy'
            front_keypoints_path = output_dir / 'front_keypoints.npy'
            
            if front_feature_path.exists() and front_keypoints_path.exists():
                existing_features = np.load(front_feature_path)
                existing_keypoints = np.load(front_keypoints_path)
                
                # 只保留已处理人员的数据
                existing_metadata = existing_data.get('metadata', [])
                existing_poses = existing_data.get('poses', [])
                existing_boxes = existing_data.get('boxes', [])
                existing_landmarks_2d = existing_data.get('landmarks_2d', [])
                existing_landmarks_3d_original = existing_data.get('landmarks_3d_original', [])
                
                for idx, meta in enumerate(existing_metadata):
                    person_name = meta.get('person_name')
                    if person_name in processed_persons:
                        all_face_features.append(existing_features[idx])
                        all_face_landmarks_3d.append(np.array(existing_landmarks_3d_original[idx]))
                        all_face_landmarks_2d.append(np.array(existing_landmarks_2d[idx]))
                        all_face_poses.append(np.array(existing_poses[idx]))
                        all_face_boxes.append(np.array(existing_boxes[idx]))
                        face_metadata.append(meta)
                
                logger.info(f"加载了 {len(all_face_features)} 个已有人员的数据")
        except Exception as e:
            logger.warning(f"加载已有数据失败: {e}，将重新处理")
            all_face_features = []
            all_face_landmarks_3d = []
            all_face_landmarks_2d = []
            all_face_poses = []
            all_face_boxes = []
            face_metadata = []
    
    # 记录失败的图片
    failed_images = {
        'no_face_detected': [],      # 未检测到人脸
        'no_landmarks': [],           # 无法提取关键点
        'read_error': [],             # 读取错误
        'feature_error': []           # 特征提取错误
    }
    
    # 处理每张正面图
    for face_image_path in tqdm(images_to_process, desc="处理正面图"):
        person_name = face_image_path.stem
        
        # 读取图像用于特征提取
        try:
            pil_img = Image.open(face_image_path).convert('RGB')
            img = np.array(pil_img)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"无法读取图像: {face_image_path}, 错误: {e}")
            failed_images['read_error'].append({
                'path': str(face_image_path),
                'person_name': person_name,
                'error': str(e)
            })
            continue
        
        # 使用InsightFace检测人脸（同时获取关键点和特征）
        faces = detector.get(img_bgr)
        if len(faces) == 0:
            logger.warning(f"未检测到人脸: {face_image_path}")
            failed_images['no_face_detected'].append({
                'path': str(face_image_path),
                'person_name': person_name
            })
            continue
        
        face = faces[0]
        
        # 提取特征（InsightFace已经归一化的特征）
        # normed_embedding已经是L2归一化的512维特征向量
        try:
            features = face.normed_embedding  # [512] 已归一化，范围通常在[-1, 1]
            
            # 确保是numpy数组
            if not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            # 验证特征维度
            if len(features) != 512:
                logger.warning(f"特征维度错误: {face_image_path}, 维度={len(features)}")
                failed_images['feature_error'].append({
                    'path': str(face_image_path),
                    'person_name': person_name,
                    'error': f'特征维度错误: {len(features)}'
                })
                continue
        except Exception as e:
            logger.warning(f"特征提取失败: {face_image_path}, 错误: {e}")
            failed_images['feature_error'].append({
                'path': str(face_image_path),
                'person_name': person_name,
                'error': str(e)
            })
            continue
        
        # 提取3D关键点和姿态
        landmarks_2d, landmarks_3d, box, euler_angles, rotation_matrix = \
            get_3d_landmarks_and_pose(detector, str(face_image_path))
        
        if landmarks_2d is None or landmarks_3d is None:
            logger.warning(f"无法提取3D关键点: {face_image_path}")
            failed_images['no_landmarks'].append({
                'path': str(face_image_path),
                'person_name': person_name
            })
            continue
        
        # 保存数据
        all_face_features.append(features)
        all_face_landmarks_3d.append(landmarks_3d)
        all_face_landmarks_2d.append(landmarks_2d)
        all_face_poses.append(euler_angles if euler_angles is not None else np.array([0.0, 0.0, 0.0]))
        all_face_boxes.append(box)
        
        face_metadata.append({
            'person_name': person_name,
            'image_path': str(face_image_path),
            'image_size': Image.open(face_image_path).size
        })
    
    logger.info(f"成功处理 {len(all_face_features)} 张正面图")
    
    # 转换为numpy数组
    # InsightFace的normed_embedding已经是归一化的512维特征向量
    face_features = np.array(all_face_features)  # [N, 512] (InsightFace特征维度)
    face_landmarks_3d = np.array(all_face_landmarks_3d)  # [N, 5, 3]
    face_landmarks_2d = np.array(all_face_landmarks_2d)  # [N, 5, 2]
    face_poses = np.array(all_face_poses)  # [N, 3]
    face_boxes = np.array(all_face_boxes)  # [N, 4]
    
    logger.info(f"特征维度: {face_features.shape[1]} (InsightFace: 512维)")
    
    # 计算标准坐标系
    logger.info("\n计算标准坐标系...")
    aligner = Landmark3DAligner()
    standard_info = aligner.compute_standard_from_faces(all_face_landmarks_3d)
    
    # 对齐和归一化所有正面图关键点
    logger.info("对齐和归一化正面图关键点...")
    aligned_normalized_landmarks = []
    for landmarks_3d in face_landmarks_3d:
        aligned_normalized = aligner.align_and_normalize(landmarks_3d)
        aligned_normalized_landmarks.append(aligned_normalized)
    
    aligned_normalized_landmarks = np.array(aligned_normalized_landmarks)  # [N, 5, 3]
    
    # 保存数据
    logger.info(f"\n保存数据到: {output_dir}")
    
    # 保存特征
    front_feature_path = output_dir / 'front_feature.npy'
    np.save(front_feature_path, face_features)
    logger.info(f"✓ 保存特征: {front_feature_path} (形状: {face_features.shape})")
    
    # 保存对齐和归一化后的关键点
    front_keypoints_path = output_dir / 'front_keypoints.npy'
    np.save(front_keypoints_path, aligned_normalized_landmarks)
    logger.info(f"✓ 保存关键点: {front_keypoints_path} (形状: {aligned_normalized_landmarks.shape})")
    
    # 保存元数据
    front_metadata = {
        'num_samples': len(face_features),
        'feature_dim': face_features.shape[1],
        'standard_info': standard_info,
        'metadata': face_metadata,
        'poses': face_poses.tolist(),
        'boxes': face_boxes.tolist(),
        'landmarks_2d': face_landmarks_2d.tolist(),
        'landmarks_3d_original': face_landmarks_3d.tolist()  # 保存原始关键点
    }
    
    front_metadata_path = output_dir / 'front_metadata.json'
    with open(front_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(front_metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 保存元数据: {front_metadata_path}")
    
    # 保存失败记录
    failed_summary = {
        'total_images': len(face_images),
        'successful': len(face_features),
        'failed': len(face_images) - len(face_features),
        'failed_details': failed_images
    }
    
    failed_log_path = output_dir / 'front_failed_images.json'
    with open(failed_log_path, 'w', encoding='utf-8') as f:
        json.dump(failed_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 保存失败记录: {failed_log_path}")
    
    # 打印失败统计
    logger.info(f"\n正面图处理统计:")
    logger.info(f"  总数: {len(face_images)}")
    logger.info(f"  跳过（已处理）: {skipped_count}")
    logger.info(f"  本次处理: {len(images_to_process)}")
    logger.info(f"  成功: {len(face_features)}")
    logger.info(f"  失败: {len(images_to_process) - (len(face_features) - len(processed_persons))}")
    if len(failed_images['no_face_detected']) > 0:
        logger.info(f"    - 未检测到人脸: {len(failed_images['no_face_detected'])}")
    if len(failed_images['no_landmarks']) > 0:
        logger.info(f"    - 无法提取关键点: {len(failed_images['no_landmarks'])}")
    if len(failed_images['read_error']) > 0:
        logger.info(f"    - 读取错误: {len(failed_images['read_error'])}")
    if len(failed_images['feature_error']) > 0:
        logger.info(f"    - 特征提取错误: {len(failed_images['feature_error'])}")
    
    # 保存对齐器信息（用于视频帧处理）
    return {
        'aligner': aligner,
        'standard_info': standard_info,
        'num_faces': len(face_features),
        'failed_images': failed_summary
    }


def extract_video_data(
    video_dir: str,
    output_dir: str,
    aligner: Landmark3DAligner,
    use_cpu: bool = False
):
    """
    提取视频帧的3D关键点、姿态和特征，并对齐到标准坐标系（使用InsightFace提取特征）
    
    Args:
        video_dir: 视频帧目录
        output_dir: 输出目录
        aligner: 对齐器（包含标准坐标系信息）
        use_cpu: 是否使用CPU
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("处理视频帧")
    logger.info("=" * 70)
    
    # 初始化检测器（InsightFace同时用于检测和特征提取）
    logger.info("初始化InsightFace检测器（用于关键点和特征提取）...")
    detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 获取所有人员目录
    person_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir()])
    logger.info(f"找到 {len(person_dirs)} 个人的视频数据")
    
    # 检查是否已有处理结果，加载已处理的人员列表
    video_metadata_path = output_dir / 'video_metadata.json'
    processed_persons_video = set()
    existing_video_data = None
    
    # 检查是否需要强制重新处理
    force_reprocess = getattr(extract_video_data, '_force_reprocess', False)
    
    if video_metadata_path.exists() and not force_reprocess:
        logger.info(f"发现已有视频处理结果: {video_metadata_path}")
        try:
            with open(video_metadata_path, 'r', encoding='utf-8') as f:
                existing_video_data = json.load(f)
            
            # 获取已处理的人员列表
            if 'metadata' in existing_video_data:
                processed_persons_video = {meta.get('person_name') for meta in existing_video_data['metadata']}
                logger.info(f"已处理 {len(processed_persons_video)} 个人的视频数据: {sorted(processed_persons_video)}")
        except Exception as e:
            logger.warning(f"无法加载已有视频数据: {e}，将重新处理所有数据")
            existing_video_data = None
            processed_persons_video = set()
    
    # 构建已处理帧的映射（用于检查完整性）- 必须在过滤之前构建
    processed_frames_map = {}  # {person_name: set(frame_indices)}
    if existing_video_data is not None and 'metadata' in existing_video_data:
        for meta in existing_video_data['metadata']:
            person_name = meta.get('person_name')
            frame_index = meta.get('frame_index', '')
            if person_name not in processed_frames_map:
                processed_frames_map[person_name] = set()
            processed_frames_map[person_name].add(frame_index)
    
    # 过滤出需要处理的人员（检查是否完整提取）
    persons_to_process = []
    skipped_persons = []
    partially_processed = {}  # 记录部分处理的人员及其已处理的帧
    
    for person_dir in person_dirs:
        person_name = person_dir.name
        video_images = sorted(list(person_dir.glob('*.jpg')))
        total_frames = len(video_images)
        
        # 检查该人员是否已完整提取
        if person_name in processed_persons_video:
            # 检查是否所有帧都已处理
            processed_frames = processed_frames_map.get(person_name, set())
            expected_frames = {img.stem for img in video_images}
            
            if processed_frames == expected_frames and len(processed_frames) == total_frames:
                # 完整提取，跳过
                skipped_persons.append(person_name)
                logger.debug(f"跳过已完整提取的人员视频: {person_name} ({total_frames}帧)")
            elif len(processed_frames) > 0:
                # 部分提取，记录需要继续处理的帧
                remaining_frames = expected_frames - processed_frames
                partially_processed[person_name] = {
                    'person_dir': person_dir,
                    'processed_frames': processed_frames,
                    'remaining_frames': remaining_frames,
                    'total_frames': total_frames,
                    'processed_count': len(processed_frames)
                }
                logger.info(f"发现部分处理的人员: {person_name} (已处理: {len(processed_frames)}/{total_frames})")
            else:
                # 在列表中但实际没有数据，需要重新处理
                persons_to_process.append(person_dir)
                logger.debug(f"人员 {person_name} 在列表中但无数据，将重新处理")
        else:
            # 新人员，需要处理
            persons_to_process.append(person_dir)
    
    if len(skipped_persons) > 0:
        logger.info(f"跳过 {len(skipped_persons)} 个已完整提取的人员视频")
    if len(partially_processed) > 0:
        logger.info(f"发现 {len(partially_processed)} 个部分处理的人员，将继续处理剩余帧")
    logger.info(f"需要处理 {len(persons_to_process)} 个新人员的视频数据")
    
    # 存储所有视频帧数据
    all_video_features = []
    all_video_landmarks_3d = []
    all_video_landmarks_2d = []
    all_video_poses = []
    all_video_boxes = []
    video_metadata = []
    
    # 如果已有数据，加载已有数据
    if existing_video_data is not None:
        try:
            video_feature_path = output_dir / 'video_feature.npy'
            video_keypoints_path = output_dir / 'video_keypoints.npy'
            
            if video_feature_path.exists() and video_keypoints_path.exists():
                existing_features = np.load(video_feature_path)
                existing_keypoints = np.load(video_keypoints_path)
                
                # 加载已有数据：包括完整处理的人员和部分处理的人员的已处理帧
                existing_metadata = existing_video_data.get('metadata', [])
                existing_poses = existing_video_data.get('poses', [])
                existing_boxes = existing_video_data.get('boxes', [])
                existing_landmarks_2d = existing_video_data.get('landmarks_2d', [])
                
                # 构建完整处理的人员集合（排除部分处理的人员）
                fully_processed_persons = set()
                partially_processed_persons = set(partially_processed.keys())
                
                for person_dir in person_dirs:
                    person_name = person_dir.name
                    if person_name in processed_persons_video and person_name not in partially_processed_persons:
                        video_images = sorted(list(person_dir.glob('*.jpg')))
                        processed_frames = processed_frames_map.get(person_name, set())
                        expected_frames = {img.stem for img in video_images}
                        # 只有完全匹配才认为是完整处理
                        if processed_frames == expected_frames and len(processed_frames) == len(video_images):
                            fully_processed_persons.add(person_name)
                
                # 加载完整处理的人员的所有数据
                for idx, meta in enumerate(existing_metadata):
                    person_name = meta.get('person_name')
                    frame_index = meta.get('frame_index', '')
                    
                    if person_name in fully_processed_persons:
                        # 完整处理的人员：加载所有数据
                        all_video_features.append(existing_features[idx])
                        all_video_landmarks_3d.append(existing_keypoints[idx])
                        all_video_landmarks_2d.append(np.array(existing_landmarks_2d[idx]))
                        all_video_poses.append(np.array(existing_poses[idx]))
                        all_video_boxes.append(np.array(existing_boxes[idx]))
                        video_metadata.append(meta)
                    elif person_name in partially_processed_persons:
                        # 部分处理的人员：只加载已处理的帧（这些帧会在后续继续处理时保留）
                        # 注意：部分处理的人员的已处理帧会在处理剩余帧时一起保存
                        # 这里先加载，后续处理剩余帧后会合并
                        processed_frames = processed_frames_map.get(person_name, set())
                        if frame_index in processed_frames:
                            all_video_features.append(existing_features[idx])
                            all_video_landmarks_3d.append(existing_keypoints[idx])
                            all_video_landmarks_2d.append(np.array(existing_landmarks_2d[idx]))
                            all_video_poses.append(np.array(existing_poses[idx]))
                            all_video_boxes.append(np.array(existing_boxes[idx]))
                            video_metadata.append(meta)
                
                logger.info(f"加载了 {len(all_video_features)} 个已有视频数据（完整处理: {len(fully_processed_persons)}, 部分处理: {len(partially_processed_persons)}）")
        except Exception as e:
            logger.warning(f"加载已有视频数据失败: {e}，将重新处理")
            all_video_features = []
            all_video_landmarks_3d = []
            all_video_landmarks_2d = []
            all_video_poses = []
            all_video_boxes = []
            video_metadata = []
    
    # 记录所有失败的图片（按类型分类）
    all_failed_videos = {
        'no_face_detected': [],
        'no_landmarks': [],
        'read_error': [],
        'feature_error': []
    }
    
    # 处理每个人的视频帧（新人员）
    for person_dir in persons_to_process:
        person_name = person_dir.name
        logger.info(f"\n处理 {person_name}...")
        
        # 获取该人的所有视频帧
        video_images = sorted(list(person_dir.glob('*.jpg')))
        logger.info(f"  找到 {len(video_images)} 张视频帧")
        
        # 处理所有帧
        frames_to_process = video_images
        
        person_video_features = []
        person_video_landmarks_3d = []
        person_video_landmarks_2d = []
        person_video_poses = []
        person_video_boxes = []
        person_metadata = []
        
        # 记录该人的失败图片
        person_failed = {
            'no_face_detected': 0,
            'no_landmarks': 0,
            'read_error': 0,
            'feature_error': 0
        }
        
        for video_image_path in tqdm(frames_to_process, desc=f"  处理 {person_name}"):
            # 读取图像用于特征提取
            try:
                pil_img = Image.open(video_image_path).convert('RGB')
                img = np.array(pil_img)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.debug(f"无法读取图像: {video_image_path}, 错误: {e}")
                person_failed['read_error'] += 1
                all_failed_videos['read_error'].append({
                    'path': str(video_image_path),
                    'person_name': person_name,
                    'error': str(e)
                })
                continue
            
            # 使用InsightFace检测人脸（同时获取关键点和特征）
            faces = detector.get(img_bgr)
            if len(faces) == 0:
                person_failed['no_face_detected'] += 1
                all_failed_videos['no_face_detected'].append({
                    'path': str(video_image_path),
                    'person_name': person_name
                })
                continue
            
            face = faces[0]
            
            # 提取特征（InsightFace已经归一化的特征）
            # normed_embedding已经是L2归一化的512维特征向量
            try:
                features = face.normed_embedding  # [512] 已归一化，范围通常在[-1, 1]
                
                # 确保是numpy数组
                if not isinstance(features, np.ndarray):
                    features = np.array(features, dtype=np.float32)
                
                # 验证特征维度
                if len(features) != 512:
                    logger.debug(f"特征维度错误: {video_image_path}, 维度={len(features)}")
                    person_failed['feature_error'] += 1
                    all_failed_videos['feature_error'].append({
                        'path': str(video_image_path),
                        'person_name': person_name,
                        'error': f'特征维度错误: {len(features)}'
                    })
                    continue
            except Exception as e:
                logger.debug(f"特征提取失败: {video_image_path}, 错误: {e}")
                person_failed['feature_error'] += 1
                all_failed_videos['feature_error'].append({
                    'path': str(video_image_path),
                    'person_name': person_name,
                    'error': str(e)
                })
                continue
            
            # 提取3D关键点和姿态
            landmarks_2d, landmarks_3d, box, euler_angles, rotation_matrix = \
                get_3d_landmarks_and_pose(detector, str(video_image_path))
            
            if landmarks_2d is None or landmarks_3d is None:
                person_failed['no_landmarks'] += 1
                all_failed_videos['no_landmarks'].append({
                    'path': str(video_image_path),
                    'person_name': person_name
                })
                continue
            
            # 对齐和归一化关键点
            aligned_normalized = aligner.align_and_normalize(landmarks_3d)
            
            # 保存数据
            person_video_features.append(features)
            person_video_landmarks_3d.append(aligned_normalized)
            person_video_landmarks_2d.append(landmarks_2d)
            person_video_poses.append(euler_angles if euler_angles is not None else np.array([0.0, 0.0, 0.0]))
            person_video_boxes.append(box)
            
            person_metadata.append({
                'person_name': person_name,
                'image_path': str(video_image_path),
                'image_size': Image.open(video_image_path).size,
                'frame_index': video_image_path.stem
            })
        
        total_failed = sum(person_failed.values())
        logger.info(f"  {person_name}: 成功处理 {len(person_video_features)}/{len(video_images)} 张视频帧")
        if total_failed > 0:
            logger.info(f"    失败: {total_failed} (未检测到人脸: {person_failed['no_face_detected']}, "
                       f"无法提取关键点: {person_failed['no_landmarks']}, "
                       f"读取错误: {person_failed['read_error']}, "
                       f"特征错误: {person_failed['feature_error']})")
        
        # 添加到总列表
        all_video_features.extend(person_video_features)
        all_video_landmarks_3d.extend(person_video_landmarks_3d)
        all_video_landmarks_2d.extend(person_video_landmarks_2d)
        all_video_poses.extend(person_video_poses)
        all_video_boxes.extend(person_video_boxes)
        video_metadata.extend(person_metadata)
    
    # 处理部分处理的人员（继续处理剩余帧）
    for person_name, info in partially_processed.items():
        person_dir = info['person_dir']
        remaining_frames = info['remaining_frames']
        processed_count = info['processed_count']
        total_frames = info['total_frames']
        
        logger.info(f"\n继续处理 {person_name} (已处理: {processed_count}/{total_frames})...")
        
        # 获取需要处理的帧
        all_video_images = sorted(list(person_dir.glob('*.jpg')))
        frames_to_process = [img for img in all_video_images if img.stem in remaining_frames]
        logger.info(f"  需要处理剩余 {len(frames_to_process)} 张视频帧")
        
        person_video_features = []
        person_video_landmarks_3d = []
        person_video_landmarks_2d = []
        person_video_poses = []
        person_video_boxes = []
        person_metadata = []
        
        # 记录该人的失败图片
        person_failed = {
            'no_face_detected': 0,
            'no_landmarks': 0,
            'read_error': 0,
            'feature_error': 0
        }
        
        for video_image_path in tqdm(frames_to_process, desc=f"  继续处理 {person_name}"):
            # 读取图像用于特征提取
            try:
                pil_img = Image.open(video_image_path).convert('RGB')
                img = np.array(pil_img)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.debug(f"无法读取图像: {video_image_path}, 错误: {e}")
                person_failed['read_error'] += 1
                all_failed_videos['read_error'].append({
                    'path': str(video_image_path),
                    'person_name': person_name,
                    'error': str(e)
                })
                continue
            
            # 使用InsightFace检测人脸（同时获取关键点和特征）
            faces = detector.get(img_bgr)
            if len(faces) == 0:
                person_failed['no_face_detected'] += 1
                all_failed_videos['no_face_detected'].append({
                    'path': str(video_image_path),
                    'person_name': person_name
                })
                continue
            
            face = faces[0]
            
            # 提取特征（InsightFace已经归一化的特征）
            try:
                features = face.normed_embedding  # [512] 已归一化
                
                # 确保是numpy数组
                if not isinstance(features, np.ndarray):
                    features = np.array(features, dtype=np.float32)
                
                # 验证特征维度
                if len(features) != 512:
                    logger.debug(f"特征维度错误: {video_image_path}, 维度={len(features)}")
                    person_failed['feature_error'] += 1
                    all_failed_videos['feature_error'].append({
                        'path': str(video_image_path),
                        'person_name': person_name,
                        'error': f'特征维度错误: {len(features)}'
                    })
                    continue
            except Exception as e:
                logger.debug(f"特征提取失败: {video_image_path}, 错误: {e}")
                person_failed['feature_error'] += 1
                all_failed_videos['feature_error'].append({
                    'path': str(video_image_path),
                    'person_name': person_name,
                    'error': str(e)
                })
                continue
            
            # 提取3D关键点和姿态
            landmarks_2d, landmarks_3d, box, euler_angles, rotation_matrix = \
                get_3d_landmarks_and_pose(detector, str(video_image_path))
            
            if landmarks_2d is None or landmarks_3d is None:
                person_failed['no_landmarks'] += 1
                all_failed_videos['no_landmarks'].append({
                    'path': str(video_image_path),
                    'person_name': person_name
                })
                continue
            
            # 对齐和归一化关键点
            aligned_normalized = aligner.align_and_normalize(landmarks_3d)
            
            # 保存数据
            person_video_features.append(features)
            person_video_landmarks_3d.append(aligned_normalized)
            person_video_landmarks_2d.append(landmarks_2d)
            person_video_poses.append(euler_angles if euler_angles is not None else np.array([0.0, 0.0, 0.0]))
            person_video_boxes.append(box)
            
            person_metadata.append({
                'person_name': person_name,
                'image_path': str(video_image_path),
                'image_size': Image.open(video_image_path).size,
                'frame_index': video_image_path.stem
            })
        
        total_failed = sum(person_failed.values())
        logger.info(f"  {person_name}: 继续处理成功 {len(person_video_features)}/{len(frames_to_process)} 张视频帧")
        if total_failed > 0:
            logger.info(f"    失败: {total_failed}")
        
        # 添加到总列表
        all_video_features.extend(person_video_features)
        all_video_landmarks_3d.extend(person_video_landmarks_3d)
        all_video_landmarks_2d.extend(person_video_landmarks_2d)
        all_video_poses.extend(person_video_poses)
        all_video_boxes.extend(person_video_boxes)
        video_metadata.extend(person_metadata)
        
        person_video_features = []
        person_video_landmarks_3d = []
        person_video_landmarks_2d = []
        person_video_poses = []
        person_video_boxes = []
        person_metadata = []
        
        # 记录该人的失败图片
        person_failed = {
            'no_face_detected': 0,
            'no_landmarks': 0,
            'read_error': 0,
            'feature_error': 0
        }
        
        for video_image_path in tqdm(frames_to_process, desc=f"  处理 {person_name}"):
            # 读取图像用于特征提取
            try:
                pil_img = Image.open(video_image_path).convert('RGB')
                img = np.array(pil_img)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.debug(f"无法读取图像: {video_image_path}, 错误: {e}")
                person_failed['read_error'] += 1
                all_failed_videos['read_error'].append({
                    'path': str(video_image_path),
                    'person_name': person_name,
                    'error': str(e)
                })
                continue
            
            # 使用InsightFace检测人脸（同时获取关键点和特征）
            faces = detector.get(img_bgr)
            if len(faces) == 0:
                person_failed['no_face_detected'] += 1
                all_failed_videos['no_face_detected'].append({
                    'path': str(video_image_path),
                    'person_name': person_name
                })
                continue
            
            face = faces[0]
            
            # 提取特征（InsightFace已经归一化的特征）
            # normed_embedding已经是L2归一化的512维特征向量
            try:
                features = face.normed_embedding  # [512] 已归一化，范围通常在[-1, 1]
                
                # 确保是numpy数组
                if not isinstance(features, np.ndarray):
                    features = np.array(features, dtype=np.float32)
                
                # 验证特征维度
                if len(features) != 512:
                    logger.debug(f"特征维度错误: {video_image_path}, 维度={len(features)}")
                    person_failed['feature_error'] += 1
                    all_failed_videos['feature_error'].append({
                        'path': str(video_image_path),
                        'person_name': person_name,
                        'error': f'特征维度错误: {len(features)}'
                    })
                    continue
            except Exception as e:
                logger.debug(f"特征提取失败: {video_image_path}, 错误: {e}")
                person_failed['feature_error'] += 1
                all_failed_videos['feature_error'].append({
                    'path': str(video_image_path),
                    'person_name': person_name,
                    'error': str(e)
                })
                continue
            
            # 提取3D关键点和姿态
            landmarks_2d, landmarks_3d, box, euler_angles, rotation_matrix = \
                get_3d_landmarks_and_pose(detector, str(video_image_path))
            
            if landmarks_2d is None or landmarks_3d is None:
                person_failed['no_landmarks'] += 1
                all_failed_videos['no_landmarks'].append({
                    'path': str(video_image_path),
                    'person_name': person_name
                })
                continue
            
            # 对齐和归一化关键点
            aligned_normalized = aligner.align_and_normalize(landmarks_3d)
            
            # 保存数据
            person_video_features.append(features)
            person_video_landmarks_3d.append(aligned_normalized)
            person_video_landmarks_2d.append(landmarks_2d)
            person_video_poses.append(euler_angles if euler_angles is not None else np.array([0.0, 0.0, 0.0]))
            person_video_boxes.append(box)
            
            person_metadata.append({
                'person_name': person_name,
                'image_path': str(video_image_path),
                'image_size': Image.open(video_image_path).size,
                'frame_index': video_image_path.stem
            })
        
        total_failed = sum(person_failed.values())
        logger.info(f"  {person_name}: 成功处理 {len(person_video_features)}/{len(video_images)} 张视频帧")
        if total_failed > 0:
            logger.info(f"    失败: {total_failed} (未检测到人脸: {person_failed['no_face_detected']}, "
                       f"无法提取关键点: {person_failed['no_landmarks']}, "
                       f"读取错误: {person_failed['read_error']}, "
                       f"特征错误: {person_failed['feature_error']})")
        
        # 添加到总列表
        all_video_features.extend(person_video_features)
        all_video_landmarks_3d.extend(person_video_landmarks_3d)
        all_video_landmarks_2d.extend(person_video_landmarks_2d)
        all_video_poses.extend(person_video_poses)
        all_video_boxes.extend(person_video_boxes)
        video_metadata.extend(person_metadata)
    
    logger.info(f"\n总共成功处理 {len(all_video_features)} 张视频帧")
    
    # 转换为numpy数组
    # InsightFace的normed_embedding已经是归一化的512维特征向量
    video_features = np.array(all_video_features)  # [N, 512] (InsightFace特征维度)
    video_landmarks_3d = np.array(all_video_landmarks_3d)  # [N, 5, 3]
    video_landmarks_2d = np.array(all_video_landmarks_2d)  # [N, 5, 2]
    video_poses = np.array(all_video_poses)  # [N, 3]
    video_boxes = np.array(all_video_boxes)  # [N, 4]
    
    logger.info(f"特征维度: {video_features.shape[1]} (InsightFace: 512维)")
    
    # 保存失败记录
    # 统计总视频帧数（所有人员目录，包括已处理和本次处理的）
    total_video_images_all = sum(len(list(person_dir.glob('*.jpg'))) for person_dir in person_dirs)
    # 本次处理的视频帧数
    total_video_images_processed = sum(len(list(person_dir.glob('*.jpg'))) for person_dir in persons_to_process)
    # 总成功数（包括已加载的）
    total_successful = len(video_features)
    
    failed_summary = {
        'total_images': total_video_images_all,
        'skipped_persons': len(skipped_persons),
        'processed_persons': len(persons_to_process),
        'successful': total_successful,
        'failed': total_video_images_processed - (total_successful - len([m for m in (existing_video_data.get('metadata', []) if existing_video_data else []) if m.get('person_name') in processed_persons_video])),
        'failed_details': all_failed_videos
    }
    
    failed_log_path = output_dir / 'video_failed_images.json'
    with open(failed_log_path, 'w', encoding='utf-8') as f:
        json.dump(failed_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 保存失败记录: {failed_log_path}")
    
    # 保存数据
    logger.info(f"\n保存数据到: {output_dir}")
    
    # 保存特征
    video_feature_path = output_dir / 'video_feature.npy'
    np.save(video_feature_path, video_features)
    logger.info(f"✓ 保存特征: {video_feature_path} (形状: {video_features.shape})")
    
    # 保存对齐和归一化后的关键点
    video_keypoints_path = output_dir / 'video_keypoints.npy'
    np.save(video_keypoints_path, video_landmarks_3d)
    logger.info(f"✓ 保存关键点: {video_keypoints_path} (形状: {video_landmarks_3d.shape})")
    
    # 保存元数据
    video_metadata_dict = {
        'num_samples': len(video_features),
        'feature_dim': video_features.shape[1],
        'metadata': video_metadata,
        'poses': video_poses.tolist(),
        'boxes': video_boxes.tolist(),
        'landmarks_2d': video_landmarks_2d.tolist()
    }
    
    video_metadata_path = output_dir / 'video_metadata.json'
    with open(video_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(video_metadata_dict, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 保存元数据: {video_metadata_path}")
    
    # 打印详细失败统计
    existing_count = len([m for m in (existing_video_data.get('metadata', []) if existing_video_data else []) if m.get('person_name') in processed_persons_video])
    new_processed_count = total_successful - existing_count
    failed_count = total_video_images_processed - new_processed_count
    
    logger.info(f"\n视频帧处理统计:")
    logger.info(f"  总数: {total_video_images_all}")
    logger.info(f"  跳过（已处理）: {len(skipped_persons)} 个人")
    logger.info(f"  本次处理: {len(persons_to_process)} 个人")
    logger.info(f"  成功总数: {total_successful} (已有: {existing_count}, 新增: {new_processed_count})")
    logger.info(f"  失败: {failed_count}")
    if failed_count > 0 and total_video_images_processed > 0:
        logger.info(f"  成功率: {new_processed_count/total_video_images_processed*100:.2f}%")
        logger.info(f"  失败详情:")
        if len(all_failed_videos['no_face_detected']) > 0:
            logger.info(f"    - 未检测到人脸: {len(all_failed_videos['no_face_detected'])}")
        if len(all_failed_videos['no_landmarks']) > 0:
            logger.info(f"    - 无法提取关键点: {len(all_failed_videos['no_landmarks'])}")
        if len(all_failed_videos['read_error']) > 0:
            logger.info(f"    - 读取错误: {len(all_failed_videos['read_error'])}")
        if len(all_failed_videos['feature_error']) > 0:
            logger.info(f"    - 特征提取错误: {len(all_failed_videos['feature_error'])}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='提取3D关键点和姿态，并进行对齐和归一化')
    parser.add_argument('--face_dir', type=str,
                       default='train/datas/face',
                       help='正面图目录')
    parser.add_argument('--video_dir', type=str,
                       default='train/datas/video',
                       help='视频帧目录')
    parser.add_argument('--output_dir', type=str,
                       default='train/datas/file',
                       help='输出目录')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用CPU（默认使用GPU）')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='强制重新处理所有数据（忽略已处理的数据）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置强制重新处理标志
    if args.force_reprocess:
        extract_face_data._force_reprocess = True
        extract_video_data._force_reprocess = True
        logger.info("⚠️  强制重新处理模式：将忽略已处理的数据")
    else:
        extract_face_data._force_reprocess = False
        extract_video_data._force_reprocess = False
    
    # 第一步：处理正面图
    face_result = extract_face_data(
        face_dir=args.face_dir,
        output_dir=output_dir,
        use_cpu=args.use_cpu
    )
    
    # 第二步：处理视频帧（使用正面图的标准坐标系）
    extract_video_data(
        video_dir=args.video_dir,
        output_dir=output_dir,
        aligner=face_result['aligner'],
        use_cpu=args.use_cpu
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("处理完成！")
    logger.info("=" * 70)
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"正面图数量: {face_result['num_faces']}")
    logger.info(f"标准坐标系信息已保存在 front_metadata.json 中")


if __name__ == "__main__":
    main()
