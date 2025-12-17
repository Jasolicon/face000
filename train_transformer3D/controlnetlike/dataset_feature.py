"""
模型1的数据集：特征转换ControlNet
数据格式：
- 输入：特征 + 姿势
- 控制：目标角度
- 输出：目标角度的特征（保持身份一致性）

数据来源：与triplet数据集相同，但需要配对不同角度的数据
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

from train_transformer3D.utils_seed import set_seed

set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureControlDataset(Dataset):
    """
    特征转换ControlNet数据集
    
    数据格式：
    - front_feature.npy: 正面图特征 [N, feature_dim]
    - front_keypoints.npy: 正面图关键点 [N, 5, 3]
    - front_metadata.json: 正面图元数据
    - video_feature.npy: 视频帧特征 [M, feature_dim]
    - video_keypoints.npy: 视频帧关键点 [M, 5, 3]
    - video_metadata.json: 视频帧元数据
    
    每个样本包含：
    - source_features: 源特征 [feature_dim]
    - source_pose: 源姿势 [pose_dim]
    - target_angle: 目标角度（控制角度）[pose_dim]
    - target_features: 目标角度的特征 [feature_dim]（ground truth）
    - person_name: 人员名称（用于身份一致性验证）
    """
    
    def __init__(
        self,
        data_dir: str,
        max_samples: Optional[int] = None,
        min_angle_diff: float = 5.0,  # 最小角度差异（度）
        max_angle_diff: float = 90.0  # 最大角度差异（度）
    ):
        """
        Args:
            data_dir: 数据目录路径
            max_samples: 最大样本数（None表示使用所有数据）
            min_angle_diff: 最小角度差异（用于筛选有效样本）
            max_angle_diff: 最大角度差异
        """
        self.data_dir = Path(data_dir)
        self.min_angle_diff = min_angle_diff
        self.max_angle_diff = max_angle_diff
        
        # 加载数据
        print(f"加载数据从: {self.data_dir}")
        
        # 加载正面图数据
        front_feature_path = self.data_dir / 'front_feature.npy'
        front_keypoints_path = self.data_dir / 'front_keypoints.npy'
        front_metadata_path = self.data_dir / 'front_metadata.json'
        
        if not front_feature_path.exists():
            raise FileNotFoundError(f"找不到文件: {front_feature_path}")
        if not front_keypoints_path.exists():
            raise FileNotFoundError(f"找不到文件: {front_keypoints_path}")
        if not front_metadata_path.exists():
            raise FileNotFoundError(f"找不到文件: {front_metadata_path}")
        
        self.front_features = np.load(front_feature_path)  # [N, feature_dim]
        self.front_keypoints = np.load(front_keypoints_path)  # [N, 5, 3]
        with open(front_metadata_path, 'r', encoding='utf-8') as f:
            self.front_metadata = json.load(f)
        
        # 加载视频帧数据
        video_feature_path = self.data_dir / 'video_feature.npy'
        video_keypoints_path = self.data_dir / 'video_keypoints.npy'
        video_metadata_path = self.data_dir / 'video_metadata.json'
        
        if not video_feature_path.exists():
            raise FileNotFoundError(f"找不到文件: {video_feature_path}")
        if not video_keypoints_path.exists():
            raise FileNotFoundError(f"找不到文件: {video_keypoints_path}")
        if not video_metadata_path.exists():
            raise FileNotFoundError(f"找不到文件: {video_metadata_path}")
        
        self.video_features = np.load(video_feature_path)  # [M, feature_dim]
        self.video_keypoints = np.load(video_keypoints_path)  # [M, 5, 3]
        with open(video_metadata_path, 'r', encoding='utf-8') as f:
            self.video_metadata = json.load(f)
        
        print(f"正面图数量: {len(self.front_features)}")
        print(f"视频帧数量: {len(self.video_features)}")
        
        # 构建样本对
        self.samples = self._build_samples()
        
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"总样本数: {len(self.samples)}")
    
    def _extract_pose_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        从关键点提取姿势（欧拉角）
        简化实现：使用关键点的几何关系估算角度
        
        Args:
            keypoints: 关键点 [5, 3]
        Returns:
            pose: 姿势（欧拉角）[3]
        """
        # 简化实现：使用关键点的位置关系估算角度
        # 实际应该使用更复杂的3D姿态估计方法
        
        # 计算yaw（左右转头）
        # 使用左右眼关键点的x坐标差异
        if keypoints.shape[0] >= 2:
            left_eye = keypoints[0, 0]  # 左眼x坐标
            right_eye = keypoints[1, 0]  # 右眼x坐标
            yaw = np.arctan2(right_eye - left_eye, 1.0) * 180 / np.pi
        else:
            yaw = 0.0
        
        # 计算pitch（上下点头）
        # 使用鼻子和眼睛的y坐标差异
        if keypoints.shape[0] >= 3:
            nose = keypoints[2, 1]  # 鼻子y坐标
            eye_y = (keypoints[0, 1] + keypoints[1, 1]) / 2  # 眼睛平均y坐标
            pitch = np.arctan2(nose - eye_y, 1.0) * 180 / np.pi
        else:
            pitch = 0.0
        
        # roll（头部旋转）
        if keypoints.shape[0] >= 2:
            roll = np.arctan2(keypoints[1, 1] - keypoints[0, 1], keypoints[1, 0] - keypoints[0, 0]) * 180 / np.pi
        else:
            roll = 0.0
        
        return np.array([yaw, pitch, roll])
    
    def _calculate_angle_diff(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """计算两个姿势的角度差异（度）"""
        diff = pose1 - pose2
        angle_diff = np.linalg.norm(diff)
        return angle_diff
    
    def _build_samples(self) -> List[Dict]:
        """
        构建样本对
        每个样本包含：源特征+姿势 -> 目标角度 -> 目标特征
        """
        samples = []
        
        # 按person_name组织数据
        front_by_person = defaultdict(list)
        video_by_person = defaultdict(list)
        
        # 获取真正的元数据列表
        # front_metadata 是一个字典，包含 'metadata' 键，值是元数据列表
        if isinstance(self.front_metadata, dict) and 'metadata' in self.front_metadata:
            front_metadata_list = self.front_metadata['metadata']
        else:
            # 如果直接是列表，则使用它
            front_metadata_list = self.front_metadata if isinstance(self.front_metadata, list) else []
        
        if isinstance(self.video_metadata, dict) and 'metadata' in self.video_metadata:
            video_metadata_list = self.video_metadata['metadata']
        else:
            video_metadata_list = self.video_metadata if isinstance(self.video_metadata, list) else []
        
        # 组织正面图数据
        for i, metadata in enumerate(front_metadata_list):
            if isinstance(metadata, dict):
                person_name = metadata.get('person_name', f'person_{i}')
            else:
                person_name = f'person_{i}'
            front_by_person[person_name].append({
                'index': i,
                'feature': self.front_features[i],
                'keypoints': self.front_keypoints[i],
                'metadata': metadata
            })
        
        # 组织视频帧数据
        for i, metadata in enumerate(video_metadata_list):
            if isinstance(metadata, dict):
                person_name = metadata.get('person_name', f'person_{i}')
            else:
                person_name = f'person_{i}'
            video_by_person[person_name].append({
                'index': i,
                'feature': self.video_features[i],
                'keypoints': self.video_keypoints[i],
                'metadata': metadata
            })
        
        # 为每个人构建样本对
        for person_name in front_by_person.keys():
            if person_name not in video_by_person:
                continue
            
            front_samples = front_by_person[person_name]
            video_samples = video_by_person[person_name]
            
            # 为每个正面图找到匹配的视频帧
            for front_sample in front_samples:
                front_feature = front_sample['feature']
                front_keypoints = front_sample['keypoints']
                front_pose = self._extract_pose_from_keypoints(front_keypoints)
                
                # 为每个视频帧创建样本
                for video_sample in video_samples:
                    video_feature = video_sample['feature']
                    video_keypoints = video_sample['keypoints']
                    video_pose = self._extract_pose_from_keypoints(video_keypoints)
                    
                    # 计算角度差异
                    angle_diff = self._calculate_angle_diff(front_pose, video_pose)
                    
                    # 筛选有效样本
                    if self.min_angle_diff <= angle_diff <= self.max_angle_diff:
                        samples.append({
                            'source_features': front_feature.astype(np.float32),
                            'source_pose': front_pose.astype(np.float32),
                            'target_angle': video_pose.astype(np.float32),
                            'target_features': video_feature.astype(np.float32),
                            'person_name': person_name,
                            'front_idx': front_sample['index'],
                            'video_idx': video_sample['index']
                        })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            dict包含：
                - source_features: 源特征 [feature_dim]
                - source_pose: 源姿势 [pose_dim]
                - target_angle: 目标角度（控制角度）[pose_dim]
                - target_features: 目标角度的特征 [feature_dim]
                - person_name: 人员名称
        """
        sample = self.samples[idx]
        
        return {
            'source_features': torch.from_numpy(sample['source_features']),
            'source_pose': torch.from_numpy(sample['source_pose']),
            'target_angle': torch.from_numpy(sample['target_angle']),
            'target_features': torch.from_numpy(sample['target_features']),
            'person_name': sample['person_name'],
            'front_idx': sample['front_idx'],
            'video_idx': sample['video_idx']
        }


def feature_control_collate_fn(batch):
    """
    自定义collate函数
    """
    batched = {}
    
    # Tensor字段
    tensor_fields = ['source_features', 'source_pose', 'target_angle', 'target_features']
    for field in tensor_fields:
        if field in batch[0]:
            batched[field] = torch.stack([item[field] for item in batch])
    
    # 字符串字段
    string_fields = ['person_name']
    for field in string_fields:
        if field in batch[0]:
            batched[field] = [item[field] for item in batch]
    
    # 整数字段
    int_fields = ['front_idx', 'video_idx']
    for field in int_fields:
        if field in batch[0]:
            batched[field] = torch.tensor([item[field] for item in batch], dtype=torch.long)
    
    return batched


def create_feature_control_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_samples: Optional[int] = None,
    min_angle_diff: float = 5.0,
    max_angle_diff: float = 90.0
) -> DataLoader:
    """
    创建特征转换ControlNet的数据加载器
    """
    dataset = FeatureControlDataset(
        data_dir=data_dir,
        max_samples=max_samples,
        min_angle_diff=min_angle_diff,
        max_angle_diff=max_angle_diff
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_control_collate_fn
    )
    
    return dataloader


def create_feature_control_train_val_test_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    pin_memory: bool = True,
    max_samples: Optional[int] = None,
    min_angle_diff: float = 5.0,
    max_angle_diff: float = 90.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试数据加载器（按person_name分割）
    """
    # 创建完整数据集
    full_dataset = FeatureControlDataset(
        data_dir=data_dir,
        max_samples=max_samples,
        min_angle_diff=min_angle_diff,
        max_angle_diff=max_angle_diff
    )
    
    # 按person_name分割
    person_names = list(set([sample['person_name'] for sample in full_dataset.samples]))
    random.shuffle(person_names)
    
    n_total = len(person_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_persons = set(person_names[:n_train])
    val_persons = set(person_names[n_train:n_train + n_val])
    test_persons = set(person_names[n_train + n_val:])
    
    # 创建索引
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, sample in enumerate(full_dataset.samples):
        person_name = sample['person_name']
        if person_name in train_persons:
            train_indices.append(i)
        elif person_name in val_persons:
            val_indices.append(i)
        elif person_name in test_persons:
            test_indices.append(i)
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"训练集: {len(train_dataset)} 样本 ({len(train_persons)} 人)")
    print(f"验证集: {len(val_dataset)} 样本 ({len(val_persons)} 人)")
    print(f"测试集: {len(test_dataset)} 样本 ({len(test_persons)} 人)")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_control_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_control_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_control_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """测试数据集"""
    print("=" * 70)
    print("特征转换ControlNet数据集测试")
    print("=" * 70)
    
    # 创建数据集
    data_dir = 'train/datas/file'
    dataset = FeatureControlDataset(data_dir=data_dir, max_samples=100)
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试一个样本
    sample = dataset[0]
    print(f"\n样本字段:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    # 测试数据加载器
    dataloader = create_feature_control_dataloader(
        data_dir=data_dir,
        batch_size=4,
        shuffle=False
    )
    
    print(f"\n数据加载器批次大小: {dataloader.batch_size}")
    
    # 测试一个批次
    batch = next(iter(dataloader))
    print(f"\n批次字段:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
