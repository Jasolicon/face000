"""
3D增强的Transformer训练数据集（新格式，推荐使用）
从npy和json文件读取对齐和归一化后的数据

这是当前正在使用的数据集类，被 train_3d.py 使用。
数据格式：
- front_feature.npy, front_keypoints.npy, front_metadata.json
- video_feature.npy, video_keypoints.npy, video_metadata.json

所有数据已经过对齐和归一化处理，适用于生产环境训练。
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Optional, Tuple
import logging

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置随机种子（不依赖外部模块）
def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Aligned3DFaceDataset(Dataset):
    """
    从对齐和归一化后的npy文件读取数据的3D数据集
    
    数据格式：
    - front_feature.npy: 正面图特征 [N, feature_dim]
    - front_keypoints.npy: 正面图对齐归一化后的关键点 [N, 5, 3]（已弃用，现在从原始数据重新归一化）
    - front_metadata.json: 正面图元数据（包含原始关键点 landmarks_3d_original）
    - video_feature.npy: 视频帧特征 [M, feature_dim]
    - video_keypoints.npy: 视频帧对齐归一化后的关键点 [M, 5, 3]（已弃用，现在从原始数据重新归一化）
    - video_metadata.json: 视频帧元数据（包含原始关键点 landmarks_3d_original）
    
    注意：现在使用每张图片自己的中心点进行归一化，而不是使用全局标准中心点。
    """
    
    def __init__(
        self,
        data_dir: str = 'train/datas/file',
        load_in_memory: bool = True,
        use_self_center: bool = True,
        min_yaw_angle: float = None,
        max_yaw_angle: float = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录（包含front_*.npy和video_*.npy文件）
            load_in_memory: 是否将所有数据加载到内存（默认True，速度快但占用内存）
            use_self_center: 是否使用每张图片自己的中心点进行归一化（默认True，推荐）
            min_yaw_angle: 最小yaw角度阈值（度），如果设置，只保留yaw角度 >= min_yaw_angle 的样本
            max_yaw_angle: 最大yaw角度阈值（度），如果设置，只保留yaw角度 <= max_yaw_angle 的样本
        """
        self.data_dir = Path(data_dir)
        self.load_in_memory = load_in_memory
        self.use_self_center = use_self_center
        self.min_yaw_angle = min_yaw_angle
        self.max_yaw_angle = max_yaw_angle
        
        # 检查文件是否存在
        required_files = [
            'front_feature.npy',
            'front_keypoints.npy',
            'front_metadata.json',
            'video_feature.npy',
            'video_keypoints.npy',
            'video_metadata.json'
        ]
        
        for file_name in required_files:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        logger.info(f"加载数据目录: {self.data_dir}")
        
        # 加载元数据
        logger.info("加载元数据...")
        with open(self.data_dir / 'front_metadata.json', 'r', encoding='utf-8') as f:
            self.front_metadata = json.load(f)
        
        with open(self.data_dir / 'video_metadata.json', 'r', encoding='utf-8') as f:
            self.video_metadata = json.load(f)
        
        # 获取标准尺度（用于归一化）
        if 'standard_info' in self.front_metadata:
            self.standard_scale = self.front_metadata['standard_info'].get('standard_scale', 1.0)
        else:
            logger.warning("未找到标准尺度信息，使用默认值1.0")
            self.standard_scale = 1.0
        
        # 加载原始关键点（用于重新归一化）
        if 'landmarks_3d_original' in self.front_metadata and len(self.front_metadata['landmarks_3d_original']) > 0:
            self.front_landmarks_3d_original = np.array(self.front_metadata['landmarks_3d_original'], dtype=np.float32)  # [N, 5, 3]
            # 验证数据长度是否匹配
            if len(self.front_landmarks_3d_original) == len(self.front_metadata['metadata']):
                logger.info(f"  正面图原始关键点: {self.front_landmarks_3d_original.shape}")
            else:
                logger.warning(f"正面图原始关键点数量不匹配: {len(self.front_landmarks_3d_original)} != {len(self.front_metadata['metadata'])}, 将使用已归一化的关键点")
                self.front_landmarks_3d_original = None
        else:
            logger.warning("未找到正面图原始关键点或数据为空，将使用已归一化的关键点")
            self.front_landmarks_3d_original = None
        
        if 'landmarks_3d_original' in self.video_metadata and len(self.video_metadata['landmarks_3d_original']) > 0:
            self.video_landmarks_3d_original = np.array(self.video_metadata['landmarks_3d_original'], dtype=np.float32)  # [M, 5, 3]
            # 验证数据长度是否匹配
            if len(self.video_landmarks_3d_original) == len(self.video_metadata['metadata']):
                logger.info(f"  视频帧原始关键点: {self.video_landmarks_3d_original.shape}")
            else:
                logger.warning(f"视频帧原始关键点数量不匹配: {len(self.video_landmarks_3d_original)} != {len(self.video_metadata['metadata'])}, 将使用已归一化的关键点")
                self.video_landmarks_3d_original = None
        else:
            logger.warning("未找到视频帧原始关键点或数据为空，将使用已归一化的关键点")
            self.video_landmarks_3d_original = None
        
        # 加载数据
        if load_in_memory:
            logger.info("加载所有数据到内存...")
            self.front_features = np.load(self.data_dir / 'front_feature.npy')
            self.front_keypoints = np.load(self.data_dir / 'front_keypoints.npy')
            self.video_features = np.load(self.data_dir / 'video_feature.npy')
            self.video_keypoints = np.load(self.data_dir / 'video_keypoints.npy')
            
            logger.info(f"  正面图特征: {self.front_features.shape}")
            logger.info(f"  正面图关键点: {self.front_keypoints.shape}")
            logger.info(f"  视频帧特征: {self.video_features.shape}")
            logger.info(f"  视频帧关键点: {self.video_keypoints.shape}")
        else:
            # 只保存路径，需要时再加载
            self.front_feature_path = self.data_dir / 'front_feature.npy'
            self.front_keypoints_path = self.data_dir / 'front_keypoints.npy'
            self.video_feature_path = self.data_dir / 'video_feature.npy'
            self.video_keypoints_path = self.data_dir / 'video_keypoints.npy'
            
            # 获取数据形状（用于验证）
            front_features_sample = np.load(self.front_feature_path, mmap_mode='r')
            self.front_features_shape = front_features_sample.shape
            self.video_features_shape = np.load(self.video_feature_path, mmap_mode='r').shape
        
        # 加载姿态数据（从元数据中）
        self.front_poses = np.array(self.front_metadata['poses'], dtype=np.float32)  # [N, 3]
        self.video_poses = np.array(self.video_metadata['poses'], dtype=np.float32)  # [M, 3]
        
        # 构建样本索引映射
        # 通过person_name建立正面图和视频帧的对应关系
        logger.info("构建样本索引映射...")
        self.samples = self._build_samples()
        logger.info(f"构建了 {len(self.samples)} 个训练样本（过滤前）")
        
        # 根据角度过滤样本
        if self.min_yaw_angle is not None or self.max_yaw_angle is not None:
            self.samples = self._filter_samples_by_angle(self.samples)
            logger.info(f"角度过滤后剩余 {len(self.samples)} 个训练样本")
            if self.min_yaw_angle is not None:
                logger.info(f"  最小yaw角度阈值: {self.min_yaw_angle}°")
            if self.max_yaw_angle is not None:
                logger.info(f"  最大yaw角度阈值: {self.max_yaw_angle}°")
        
        # 验证数据一致性
        self._validate_data()
    
    def _normalize_landmarks_with_self_center(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """
        使用图片自己的中心点归一化关键点
        
        Args:
            landmarks_3d: 原始3D关键点 [5, 3]
        
        Returns:
            normalized: 归一化后的关键点 [5, 3]
        """
        if self.use_self_center:
            # 计算该图片自己的中心点（5个关键点的平均值）
            image_center = np.mean(landmarks_3d, axis=0)  # [3]
            # 计算相对坐标
            relative_landmarks = landmarks_3d - image_center  # [5, 3]
            # 归一化（使用标准尺度）
            normalized = relative_landmarks * self.standard_scale  # [5, 3]
        else:
            # 使用全局标准中心点（不推荐）
            if 'standard_info' in self.front_metadata:
                standard_center = np.array(self.front_metadata['standard_info'].get('standard_center', [0, 0, 0]))
                relative_landmarks = landmarks_3d - standard_center
                normalized = relative_landmarks * self.standard_scale
            else:
                # 如果没有标准中心点，直接使用自己的中心点
                image_center = np.mean(landmarks_3d, axis=0)
                relative_landmarks = landmarks_3d - image_center
                normalized = relative_landmarks * self.standard_scale
        
        return normalized
    
    def _build_samples(self) -> List[Dict]:
        """
        构建训练样本列表
        通过person_name建立正面图和视频帧的对应关系
        
        Returns:
            samples: 样本列表，每个样本包含：
                - front_idx: 正面图索引
                - video_idx: 视频帧索引
                - person_name: 人名
        """
        samples = []
        
        # 创建正面图person_name到索引的映射
        front_name_to_idx = {}
        for idx, meta in enumerate(self.front_metadata['metadata']):
            person_name = meta['person_name']
            if person_name not in front_name_to_idx:
                front_name_to_idx[person_name] = []
            front_name_to_idx[person_name].append(idx)
        
        # 创建视频帧person_name到索引的映射
        video_name_to_indices = {}
        for idx, meta in enumerate(self.video_metadata['metadata']):
            person_name = meta['person_name']
            if person_name not in video_name_to_indices:
                video_name_to_indices[person_name] = []
            video_name_to_indices[person_name].append(idx)
        
        # 建立对应关系
        for person_name in front_name_to_idx.keys():
            if person_name not in video_name_to_indices:
                logger.warning(f"正面图 {person_name} 没有对应的视频帧数据")
                continue
            
            front_indices = front_name_to_idx[person_name]
            video_indices = video_name_to_indices[person_name]
            
            # 每个正面图对应所有该人的视频帧
            for front_idx in front_indices:
                for video_idx in video_indices:
                    samples.append({
                        'front_idx': front_idx,
                        'video_idx': video_idx,
                        'person_name': person_name
                    })
        
        return samples
    
    def _filter_samples_by_angle(self, samples: List[Dict]) -> List[Dict]:
        """
        根据yaw角度过滤样本
        
        Args:
            samples: 样本列表
        
        Returns:
            filtered_samples: 过滤后的样本列表
        """
        if self.min_yaw_angle is None and self.max_yaw_angle is None:
            return samples
        
        filtered_samples = []
        for sample in samples:
            video_idx = sample['video_idx']
            # 获取该视频帧的yaw角度
            yaw_angle = self.video_poses[video_idx, 0]  # yaw是第一个元素
            abs_yaw_angle = abs(yaw_angle)  # 使用绝对值，因为左右转都应该保留
            
            # 检查是否在角度范围内
            # min_yaw_angle: 只保留 |yaw| >= min_yaw_angle 的样本（排除接近正面的图片）
            if self.min_yaw_angle is not None and abs_yaw_angle < abs(self.min_yaw_angle):
                continue
            # max_yaw_angle: 只保留 |yaw| <= max_yaw_angle 的样本
            if self.max_yaw_angle is not None and abs_yaw_angle > abs(self.max_yaw_angle):
                continue
            
            filtered_samples.append(sample)
        
        return filtered_samples
    
    def _validate_data(self):
        """验证数据一致性"""
        # 检查特征维度
        if self.load_in_memory:
            front_feature_dim = self.front_features.shape[1]
            video_feature_dim = self.video_features.shape[1]
        else:
            front_feature_dim = self.front_features_shape[1]
            video_feature_dim = self.video_features_shape[1]
        
        if front_feature_dim != video_feature_dim:
            raise ValueError(f"特征维度不一致: 正面图={front_feature_dim}, 视频帧={video_feature_dim}")
        
        # 检查关键点形状
        if self.load_in_memory:
            if self.front_keypoints.shape[1:] != (5, 3):
                raise ValueError(f"正面图关键点形状错误: {self.front_keypoints.shape}, 应为 [N, 5, 3]")
            if self.video_keypoints.shape[1:] != (5, 3):
                raise ValueError(f"视频帧关键点形状错误: {self.video_keypoints.shape}, 应为 [M, 5, 3]")
        
        # 检查姿态形状
        if self.front_poses.shape[1] != 3:
            raise ValueError(f"正面图姿态形状错误: {self.front_poses.shape}, 应为 [N, 3]")
        if self.video_poses.shape[1] != 3:
            raise ValueError(f"视频帧姿态形状错误: {self.video_poses.shape}, 应为 [M, 3]")
        
        logger.info("✓ 数据验证通过")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Args:
            idx: 样本索引
        
        Returns:
            sample: 包含以下键的字典：
                - src: 视频帧特征 [feature_dim]（侧面特征，用于GAN训练）
                - tgt: 正面图特征 [feature_dim]（正面特征，用于GAN训练）
                - keypoints_3d: 视频帧3D关键点 [5, 3]（已对齐和归一化，侧面）
                - pose: 视频帧姿态向量 [3] (欧拉角: yaw, pitch, roll，侧面)
                - angles: 角度（兼容性，使用pose）[3]（侧面）
                - front_keypoints_3d: 正面图3D关键点 [5, 3]（已对齐和归一化，用于GAN训练）
                - front_pose: 正面图姿态向量 [3]（用于GAN训练，通常接近[0,0,0]）
                - front_angles: 正面角度 [3]（兼容性，用于GAN训练）
        """
        sample_info = self.samples[idx]
        front_idx = sample_info['front_idx']
        video_idx = sample_info['video_idx']
        
        # 加载数据
        if self.load_in_memory:
            # 从内存读取
            src = self.video_features[video_idx]  # [feature_dim]
            tgt = self.front_features[front_idx]  # [feature_dim]
            pose = self.video_poses[video_idx]  # [3]
            front_pose = self.front_poses[front_idx]  # [3]
        else:
            # 从文件读取（使用内存映射）
            video_features = np.load(self.video_feature_path, mmap_mode='r')
            front_features = np.load(self.front_feature_path, mmap_mode='r')
            
            src = video_features[video_idx]
            tgt = front_features[front_idx]
            pose = self.video_poses[video_idx]
            front_pose = self.front_poses[front_idx]
        
        # 处理关键点：使用原始关键点，用图片自己的中心点归一化
        if self.video_landmarks_3d_original is not None and len(self.video_landmarks_3d_original) > video_idx:
            # 使用原始关键点，重新归一化（使用图片自己的中心点）
            video_landmarks_3d_orig = self.video_landmarks_3d_original[video_idx]  # [5, 3]
            keypoints_3d = self._normalize_landmarks_with_self_center(video_landmarks_3d_orig)  # [5, 3]
        else:
            # 如果没有原始关键点，使用已归一化的关键点（向后兼容）
            if self.load_in_memory:
                keypoints_3d = self.video_keypoints[video_idx]  # [5, 3]
            else:
                video_keypoints = np.load(self.video_keypoints_path, mmap_mode='r')
                keypoints_3d = video_keypoints[video_idx]
        
        if self.front_landmarks_3d_original is not None and len(self.front_landmarks_3d_original) > front_idx:
            # 使用原始关键点，重新归一化（使用图片自己的中心点）
            front_landmarks_3d_orig = self.front_landmarks_3d_original[front_idx]  # [5, 3]
            front_keypoints_3d = self._normalize_landmarks_with_self_center(front_landmarks_3d_orig)  # [5, 3]
        else:
            # 如果没有原始关键点，使用已归一化的关键点（向后兼容）
            if self.load_in_memory:
                front_keypoints_3d = self.front_keypoints[front_idx]  # [5, 3]
            else:
                front_keypoints = np.load(self.front_keypoints_path, mmap_mode='r')
                front_keypoints_3d = front_keypoints[front_idx]
        
        # 转换为torch tensor
        src = torch.from_numpy(src).float()
        tgt = torch.from_numpy(tgt).float()
        keypoints_3d = torch.from_numpy(keypoints_3d).float()
        front_keypoints_3d = torch.from_numpy(front_keypoints_3d).float()
        pose = torch.from_numpy(pose).float()
        front_pose = torch.from_numpy(front_pose).float()
        
        # angles用于兼容性（使用pose）
        angles = pose.clone()
        # 正面角度（用于GAN训练）
        front_angles = front_pose.clone()
        
        return {
            'src': src,  # 侧面特征（视频帧）
            'tgt': tgt,  # 正面特征
            'keypoints_3d': keypoints_3d,  # 侧面关键点（视频帧）
            'pose': pose,  # 侧面姿态（视频帧）
            'angles': angles,  # 侧面角度（视频帧，兼容性）
            'front_keypoints_3d': front_keypoints_3d,  # 正面关键点（用于GAN训练）
            'front_pose': front_pose,  # 正面姿态（用于GAN训练）
            'front_angles': front_angles,  # 正面角度（用于GAN训练，兼容性）
            'person_name': sample_info['person_name'],
            'front_idx': front_idx,
            'video_idx': video_idx
        }
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            dict: 统计信息
        """
        stats = {
            'num_samples': len(self.samples),
            'num_front_images': len(self.front_metadata['metadata']),
            'num_video_images': len(self.video_metadata['metadata']),
            'feature_dim': self.front_features.shape[1] if self.load_in_memory else self.front_features_shape[1],
            'num_keypoints': 5,
            'keypoint_dim': 3,
            'pose_dim': 3,
            'unique_persons': len(set(s['person_name'] for s in self.samples))
        }
        
        return stats


def split_dataset_by_person(
    dataset: Aligned3DFaceDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.3,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    按person_name分割数据集，确保同一人的数据不会同时出现在不同集合中
    
    Args:
        dataset: 数据集实例
        train_ratio: 训练集比例（默认0.6）
        val_ratio: 验证集比例（默认0.3）
        test_ratio: 测试集比例（默认0.1）
        random_seed: 随机种子
    
    Returns:
        train_indices: 训练集样本索引
        val_indices: 验证集样本索引
        test_indices: 测试集样本索引
    """
    
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为1.0，当前为: {total_ratio}")
    
    # 按person_name分组
    person_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        person_name = sample['person_name']
        person_to_indices[person_name].append(idx)
    
    # 获取所有person_name并打乱
    person_names = list(person_to_indices.keys())
    random.seed(random_seed)
    random.shuffle(person_names)
    
    # 计算每个集合的人数
    total_persons = len(person_names)
    num_train_persons = int(total_persons * train_ratio)
    num_val_persons = int(total_persons * val_ratio)
    # num_test_persons = total_persons - num_train_persons - num_val_persons
    
    # 分割person_name
    train_persons = person_names[:num_train_persons]
    val_persons = person_names[num_train_persons:num_train_persons + num_val_persons]
    test_persons = person_names[num_train_persons + num_val_persons:]
    
    # 收集每个集合的样本索引
    train_indices = []
    val_indices = []
    test_indices = []
    
    for person_name in train_persons:
        train_indices.extend(person_to_indices[person_name])
    
    for person_name in val_persons:
        val_indices.extend(person_to_indices[person_name])
    
    for person_name in test_persons:
        test_indices.extend(person_to_indices[person_name])
    
    # 打乱每个集合内的样本（保持person_name分组）
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    logger.info(f"数据集分割完成:")
    logger.info(f"  总人数: {total_persons}")
    logger.info(f"  训练集: {len(train_persons)} 人, {len(train_indices)} 个样本 ({len(train_indices)/len(dataset.samples)*100:.1f}%)")
    logger.info(f"  验证集: {len(val_persons)} 人, {len(val_indices)} 个样本 ({len(val_indices)/len(dataset.samples)*100:.1f}%)")
    logger.info(f"  测试集: {len(test_persons)} 人, {len(test_indices)} 个样本 ({len(test_indices)/len(dataset.samples)*100:.1f}%)")
    
    return train_indices, val_indices, test_indices


def triplet_collate_fn(batch):
    """
    自定义collate函数，专门为三元组损失优化
    确保person_name等字符串字段被正确收集为列表
    
    Args:
        batch: 批次数据列表
        
    Returns:
        batched_data: 批处理后的数据字典
    """
    # 收集所有字段
    batched = {}
    
    # Tensor字段：自动堆叠
    tensor_fields = ['src', 'tgt', 'keypoints_3d', 'pose', 'angles', 
                     'front_keypoints_3d', 'front_pose', 'front_angles']
    for field in tensor_fields:
        if field in batch[0]:
            batched[field] = torch.stack([item[field] for item in batch])
    
    # 字符串字段：收集为列表
    string_fields = ['person_name']
    for field in string_fields:
        if field in batch[0]:
            batched[field] = [item[field] for item in batch]
    
    # 整数字段：收集为列表或tensor
    int_fields = ['front_idx', 'video_idx']
    for field in int_fields:
        if field in batch[0]:
            batched[field] = torch.tensor([item[field] for item in batch], dtype=torch.long)
    
    return batched


def create_dataloader(
    data_dir: str = 'train/datas/file',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_in_memory: bool = True,
    use_triplet_collate: bool = True  # 三元组损失专用collate函数
) -> DataLoader:
    """
    创建数据加载器（不分割，返回完整数据集）
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载器工作进程数
        pin_memory: 是否使用pin_memory
        load_in_memory: 是否将所有数据加载到内存
        use_triplet_collate: 是否使用三元组损失专用的collate函数（默认True）
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = Aligned3DFaceDataset(
        data_dir=data_dir,
        load_in_memory=load_in_memory
    )
    
    # 选择collate函数
    collate_fn = triplet_collate_fn if use_triplet_collate else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 丢弃最后一个不完整的批次
        persistent_workers=num_workers > 0,  # 保持worker进程
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    return dataloader


def create_train_val_test_dataloaders(
    data_dir: str = 'train/datas/file',
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_in_memory: bool = True,
    train_ratio: float = 0.6,
    val_ratio: float = 0.3,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    min_yaw_angle: float = None,
    max_yaw_angle: float = None,
    use_triplet_collate: bool = True  # 三元组损失专用collate函数
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练集、验证集和测试集数据加载器（按person_name分割）
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        pin_memory: 是否使用pin_memory
        load_in_memory: 是否将所有数据加载到内存
        train_ratio: 训练集比例（默认0.6）
        val_ratio: 验证集比例（默认0.3）
        test_ratio: 测试集比例（默认0.1）
        random_seed: 随机种子
        min_yaw_angle: 最小yaw角度阈值（度），如果设置，只保留yaw角度 >= min_yaw_angle 的样本
        max_yaw_angle: 最大yaw角度阈值（度），如果设置，只保留yaw角度 <= max_yaw_angle 的样本
        use_triplet_collate: 是否使用三元组损失专用的collate函数（默认True）
    
    Returns:
        train_loader: 训练集数据加载器
        val_loader: 验证集数据加载器
        test_loader: 测试集数据加载器
    """
    
    # 创建完整数据集
    full_dataset = Aligned3DFaceDataset(
        data_dir=data_dir,
        load_in_memory=load_in_memory,
        min_yaw_angle=min_yaw_angle,
        max_yaw_angle=max_yaw_angle
    )
    
    # 按person_name分割
    train_indices, val_indices, test_indices = split_dataset_by_person(
        full_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 选择collate函数
    collate_fn = triplet_collate_fn if use_triplet_collate else None
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 验证集不丢弃最后一个batch
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # 测试集不丢弃最后一个batch
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    print("=" * 70)
    print("测试 Aligned3DFaceDataset 数据集")
    print("=" * 70)
    
    # 检查数据目录是否存在
    data_dir = 'train/datas/file'
    if not Path(data_dir).exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
        print("请先运行 extract_and_align_3d_data.py 生成数据文件")
    else:
        # 创建数据集
        print(f"\n创建数据集（数据目录: {data_dir}）...")
        try:
            dataset = Aligned3DFaceDataset(data_dir=data_dir, load_in_memory=True)
            
            # 打印统计信息
            stats = dataset.get_statistics()
            print(f"\n数据集统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # 获取一个样本
            print("\n获取第一个样本...")
            sample = dataset[0]
            
            print(f"样本键: {sample.keys()}")
            print(f"src形状: {sample['src'].shape}")
            print(f"tgt形状: {sample['tgt'].shape}")
            print(f"keypoints_3d形状: {sample['keypoints_3d'].shape}")
            print(f"pose形状: {sample['pose'].shape}")
            print(f"angles形状: {sample['angles'].shape}")
            print(f"front_keypoints_3d形状: {sample['front_keypoints_3d'].shape}")
            print(f"front_pose形状: {sample['front_pose'].shape}")
            print(f"person_name: {sample['person_name']}")
            
            # 创建数据加载器
            print("\n创建数据加载器...")
            dataloader = create_dataloader(
                data_dir=data_dir,
                batch_size=4,
                shuffle=False,
                num_workers=0,  # Windows上使用0
                load_in_memory=True
            )
            
            # 获取一个批次
            print("\n获取第一个批次...")
            batch = next(iter(dataloader))
            
            print(f"批次键: {batch.keys()}")
            print(f"src批次形状: {batch['src'].shape}")
            print(f"tgt批次形状: {batch['tgt'].shape}")
            print(f"keypoints_3d批次形状: {batch['keypoints_3d'].shape}")
            print(f"pose批次形状: {batch['pose'].shape}")
            
            print("\n" + "=" * 70)
            print("测试完成！")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
