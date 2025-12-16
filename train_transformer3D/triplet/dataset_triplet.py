"""
三元组损失专用数据集
模仿 dataset_3d.py 的结构，专门为三元组损失优化

数据格式：
- front_feature.npy, front_keypoints.npy, front_metadata.json
- video_feature.npy, video_keypoints.npy, video_metadata.json

所有数据已经过对齐和归一化处理，适用于三元组损失训练。
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
sys.path.append(str(Path(__file__).parent.parent.parent))

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


class TripletFaceDataset3D(Dataset):
    """
    三元组损失专用的3D数据集
    从对齐和归一化后的npy文件读取数据
    
    数据格式：
    - front_feature.npy: 正面图特征 [N, feature_dim]
    - front_keypoints.npy: 正面图对齐归一化后的关键点 [N, 5, 3]
    - front_metadata.json: 正面图元数据（包含原始关键点 landmarks_3d_original）
    - video_feature.npy: 视频帧特征 [M, feature_dim]
    - video_keypoints.npy: 视频帧对齐归一化后的关键点 [M, 5, 3]
    - video_metadata.json: 视频帧元数据（包含原始关键点 landmarks_3d_original）
    
    所有数据已经过对齐和归一化处理，适用于三元组损失训练。
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
        初始化三元组数据集
        
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
            self.front_landmarks_3d_original = np.array(self.front_metadata['landmarks_3d_original'], dtype=np.float32)
            if len(self.front_landmarks_3d_original) == len(self.front_metadata['metadata']):
                logger.info(f"  正面图原始关键点: {self.front_landmarks_3d_original.shape}")
            else:
                logger.warning(f"正面图原始关键点数量不匹配，将使用已归一化的关键点")
                self.front_landmarks_3d_original = None
        else:
            logger.warning("未找到正面图原始关键点或数据为空，将使用已归一化的关键点")
            self.front_landmarks_3d_original = None
        
        if 'landmarks_3d_original' in self.video_metadata and len(self.video_metadata['landmarks_3d_original']) > 0:
            self.video_landmarks_3d_original = np.array(self.video_metadata['landmarks_3d_original'], dtype=np.float32)
            if len(self.video_landmarks_3d_original) == len(self.video_metadata['metadata']):
                logger.info(f"  视频帧原始关键点: {self.video_landmarks_3d_original.shape}")
            else:
                logger.warning(f"视频帧原始关键点数量不匹配，将使用已归一化的关键点")
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
        self.front_poses = np.array(self.front_metadata['poses'], dtype=np.float32)
        self.video_poses = np.array(self.video_metadata['poses'], dtype=np.float32)
        
        # 构建样本索引映射
        logger.info("构建样本索引映射...")
        self.samples = self._build_samples()
        logger.info(f"构建了 {len(self.samples)} 个训练样本（过滤前）")
        
        # 根据角度过滤样本
        if self.min_yaw_angle is not None or self.max_yaw_angle is not None:
            self.samples = self._filter_samples_by_angle(self.samples)
            logger.info(f"角度过滤后剩余 {len(self.samples)} 个训练样本")
        
        # 验证数据一致性
        self._validate_data()
    
    def _normalize_landmarks_with_self_center(self, landmarks_3d: np.ndarray) -> np.ndarray:
        """使用图片自己的中心点归一化关键点"""
        if self.use_self_center:
            image_center = np.mean(landmarks_3d, axis=0)
            relative_landmarks = landmarks_3d - image_center
            normalized = relative_landmarks * self.standard_scale
        else:
            if 'standard_info' in self.front_metadata:
                standard_center = np.array(self.front_metadata['standard_info'].get('standard_center', [0, 0, 0]))
                relative_landmarks = landmarks_3d - standard_center
                normalized = relative_landmarks * self.standard_scale
            else:
                image_center = np.mean(landmarks_3d, axis=0)
                relative_landmarks = landmarks_3d - image_center
                normalized = relative_landmarks * self.standard_scale
        
        return normalized
    
    def _build_samples(self) -> List[Dict]:
        """构建训练样本列表，通过person_name建立正面图和视频帧的对应关系"""
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
        """根据yaw角度过滤样本"""
        if self.min_yaw_angle is None and self.max_yaw_angle is None:
            return samples
        
        filtered_samples = []
        for sample in samples:
            video_idx = sample['video_idx']
            yaw_angle = self.video_poses[video_idx, 0]
            abs_yaw_angle = abs(yaw_angle)
            
            if self.min_yaw_angle is not None and abs_yaw_angle < abs(self.min_yaw_angle):
                continue
            if self.max_yaw_angle is not None and abs_yaw_angle > abs(self.max_yaw_angle):
                continue
            
            filtered_samples.append(sample)
        
        return filtered_samples
    
    def _validate_data(self):
        """验证数据一致性"""
        if self.load_in_memory:
            front_feature_dim = self.front_features.shape[1]
            video_feature_dim = self.video_features.shape[1]
        else:
            front_feature_dim = self.front_features_shape[1]
            video_feature_dim = self.video_features_shape[1]
        
        if front_feature_dim != video_feature_dim:
            raise ValueError(f"特征维度不一致: 正面图={front_feature_dim}, 视频帧={video_feature_dim}")
        
        logger.info("✓ 数据验证通过")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本"""
        sample_info = self.samples[idx]
        front_idx = sample_info['front_idx']
        video_idx = sample_info['video_idx']
        
        # 加载数据
        if self.load_in_memory:
            src = self.video_features[video_idx]
            tgt = self.front_features[front_idx]
            pose = self.video_poses[video_idx]
            front_pose = self.front_poses[front_idx]
        else:
            video_features = np.load(self.video_feature_path, mmap_mode='r')
            front_features = np.load(self.front_feature_path, mmap_mode='r')
            
            src = video_features[video_idx]
            tgt = front_features[front_idx]
            pose = self.video_poses[video_idx]
            front_pose = self.front_poses[front_idx]
        
        # 处理关键点
        if self.video_landmarks_3d_original is not None and len(self.video_landmarks_3d_original) > video_idx:
            video_landmarks_3d_orig = self.video_landmarks_3d_original[video_idx]
            keypoints_3d = self._normalize_landmarks_with_self_center(video_landmarks_3d_orig)
        else:
            if self.load_in_memory:
                keypoints_3d = self.video_keypoints[video_idx]
            else:
                video_keypoints = np.load(self.video_keypoints_path, mmap_mode='r')
                keypoints_3d = video_keypoints[video_idx]
        
        if self.front_landmarks_3d_original is not None and len(self.front_landmarks_3d_original) > front_idx:
            front_landmarks_3d_orig = self.front_landmarks_3d_original[front_idx]
            front_keypoints_3d = self._normalize_landmarks_with_self_center(front_landmarks_3d_orig)
        else:
            if self.load_in_memory:
                front_keypoints_3d = self.front_keypoints[front_idx]
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
        
        angles = pose.clone()
        front_angles = front_pose.clone()
        
        return {
            'src': src,
            'tgt': tgt,
            'keypoints_3d': keypoints_3d,
            'pose': pose,
            'angles': angles,
            'front_keypoints_3d': front_keypoints_3d,
            'front_pose': front_pose,
            'front_angles': front_angles,
            'person_name': sample_info['person_name'],
            'front_idx': front_idx,
            'video_idx': video_idx
        }
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
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
    dataset: TripletFaceDataset3D,
    train_ratio: float = 0.6,
    val_ratio: float = 0.3,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """按person_name分割数据集，确保同一人的数据不会同时出现在不同集合中"""
    
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
    
    # 打乱每个集合内的样本
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    logger.info(f"数据集分割完成:")
    logger.info(f"  总人数: {total_persons}")
    logger.info(f"  训练集: {len(train_persons)} 人, {len(train_indices)} 个样本")
    logger.info(f"  验证集: {len(val_persons)} 人, {len(val_indices)} 个样本")
    logger.info(f"  测试集: {len(test_persons)} 人, {len(test_indices)} 个样本")
    
    return train_indices, val_indices, test_indices


def create_triplet_dataloader(
    data_dir: str = 'train/datas/file',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_in_memory: bool = True
) -> DataLoader:
    """创建三元组数据加载器（不分割，返回完整数据集）"""
    dataset = TripletFaceDataset3D(
        data_dir=data_dir,
        load_in_memory=load_in_memory
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=triplet_collate_fn  # 使用三元组专用的collate函数
    )
    
    return dataloader


def create_triplet_train_val_test_dataloaders(
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
    max_yaw_angle: float = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练集、验证集和测试集数据加载器（按person_name分割）"""
    
    # 创建完整数据集
    full_dataset = TripletFaceDataset3D(
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
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=triplet_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=triplet_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=triplet_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    print("=" * 70)
    print("测试 TripletFaceDataset3D 数据集")
    print("=" * 70)
    
    data_dir = 'train/datas/file'
    if not Path(data_dir).exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
    else:
        try:
            dataset = TripletFaceDataset3D(data_dir=data_dir, load_in_memory=True)
            
            stats = dataset.get_statistics()
            print(f"\n数据集统计信息:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            sample = dataset[0]
            print(f"\n样本键: {sample.keys()}")
            print(f"person_name: {sample['person_name']}")
            
            print("\n" + "=" * 70)
            print("测试完成！")
            print("=" * 70)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()

