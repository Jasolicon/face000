"""
模型1的数据集：FeatureControlNetDataset
用于特征空间的角度转换训练

数据格式：
- 输入：源特征 + 源姿势 + 目标姿势
- 输出：目标角度的特征（保持身份一致）
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.utils_seed import set_seed

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureControlNetDataset(Dataset):
    """
    FeatureControlNet数据集
    
    为每个样本提供：
    - 源特征和源姿势
    - 目标姿势（控制信号）
    - 目标特征（监督信号，同一人的不同角度）
    """
    
    def __init__(
        self,
        data_dir: str = 'train/datas/file',
        load_in_memory: bool = True,
        min_angle_diff: float = 5.0,  # 最小角度差异（度）
        max_samples_per_person: int = None  # 每个人的最大样本数
    ):
        """
        Args:
            data_dir: 数据目录
            load_in_memory: 是否加载到内存
            min_angle_diff: 最小角度差异（度），用于筛选有效的训练样本
            max_samples_per_person: 每个人的最大样本数（用于平衡数据集）
        """
        self.data_dir = Path(data_dir)
        self.load_in_memory = load_in_memory
        self.min_angle_diff = min_angle_diff
        
        # 检查文件
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
        with open(self.data_dir / 'front_metadata.json', 'r', encoding='utf-8') as f:
            self.front_metadata = json.load(f)
        
        with open(self.data_dir / 'video_metadata.json', 'r', encoding='utf-8') as f:
            self.video_metadata = json.load(f)
        
        # 加载特征和姿势
        if self.load_in_memory:
            logger.info("加载特征到内存...")
            self.front_features = np.load(self.data_dir / 'front_feature.npy')  # [N, 512]
            self.video_features = np.load(self.data_dir / 'video_feature.npy')  # [M, 512]
            logger.info(f"  正面特征: {self.front_features.shape}")
            logger.info(f"  视频特征: {self.video_features.shape}")
        else:
            self.front_feature_path = self.data_dir / 'front_feature.npy'
            self.video_feature_path = self.data_dir / 'video_feature.npy'
        
        # 提取姿势信息
        self.front_poses = self._extract_poses(self.front_metadata['metadata'])  # [N, 3]
        self.video_poses = self._extract_poses(self.video_metadata['metadata'])  # [M, 3]
        
        # 按人员分组
        self.person_groups = self._group_by_person()
        
        # 构建样本
        self.samples = self._build_samples(max_samples_per_person)
        
        logger.info(f"数据集大小: {len(self.samples)} 个样本")
        logger.info(f"人员数量: {len(self.person_groups)}")
    
    def _extract_poses(self, metadata: List[Dict]) -> np.ndarray:
        """从元数据中提取姿势（欧拉角）"""
        poses = []
        for item in metadata:
            if 'pose' in item:
                pose = item['pose']  # [yaw, pitch, roll]
            elif 'euler_angles' in item:
                pose = item['euler_angles']
            else:
                # 如果没有姿势信息，使用默认值
                pose = [0.0, 0.0, 0.0]
            poses.append(pose)
        return np.array(poses, dtype=np.float32)
    
    def _group_by_person(self) -> Dict[str, Dict]:
        """按人员分组"""
        groups = defaultdict(lambda: {'front': [], 'video': []})
        
        # 正面图
        for idx, item in enumerate(self.front_metadata['metadata']):
            person_name = item.get('person_name', 'unknown')
            groups[person_name]['front'].append({
                'idx': idx,
                'pose': self.front_poses[idx],
                'metadata': item
            })
        
        # 视频帧
        for idx, item in enumerate(self.video_metadata['metadata']):
            person_name = item.get('person_name', 'unknown')
            groups[person_name]['video'].append({
                'idx': idx,
                'pose': self.video_poses[idx],
                'metadata': item
            })
        
        return dict(groups)
    
    def _calculate_angle_diff(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """计算两个姿势之间的角度差异（度）"""
        # 计算欧拉角的差异
        diff = np.abs(pose1 - pose2)
        # 主要考虑yaw角度（第一个维度）
        angle_diff = np.sqrt(np.sum(diff ** 2))  # L2距离
        return angle_diff
    
    def _build_samples(self, max_samples_per_person: Optional[int] = None) -> List[Dict]:
        """构建训练样本"""
        samples = []
        
        for person_name, group in self.person_groups.items():
            front_items = group['front']
            video_items = group['video']
            
            # 合并所有样本
            all_items = []
            for item in front_items:
                all_items.append({**item, 'type': 'front'})
            for item in video_items:
                all_items.append({**item, 'type': 'video'})
            
            if len(all_items) < 2:
                continue  # 至少需要2个样本才能配对
            
            # 生成配对样本
            person_samples = []
            for i, src_item in enumerate(all_items):
                for j, tgt_item in enumerate(all_items):
                    if i == j:
                        continue  # 跳过自己
                    
                    # 计算角度差异
                    angle_diff = self._calculate_angle_diff(src_item['pose'], tgt_item['pose'])
                    
                    if angle_diff < self.min_angle_diff:
                        continue  # 角度差异太小，跳过
                    
                    # 确定索引
                    if src_item['type'] == 'front':
                        src_idx = src_item['idx']
                        src_type = 'front'
                    else:
                        src_idx = src_item['idx']
                        src_type = 'video'
                    
                    if tgt_item['type'] == 'front':
                        tgt_idx = tgt_item['idx']
                        tgt_type = 'front'
                    else:
                        tgt_idx = tgt_item['idx']
                        tgt_type = 'video'
                    
                    person_samples.append({
                        'person_name': person_name,
                        'src_idx': src_idx,
                        'src_type': src_type,
                        'src_pose': src_item['pose'],
                        'tgt_idx': tgt_idx,
                        'tgt_type': tgt_type,
                        'tgt_pose': tgt_item['pose'],
                        'angle_diff': angle_diff
                    })
            
            # 限制每个人员的样本数
            if max_samples_per_person and len(person_samples) > max_samples_per_person:
                # 按角度差异排序，选择差异较大的样本
                person_samples.sort(key=lambda x: x['angle_diff'], reverse=True)
                person_samples = person_samples[:max_samples_per_person]
            
            samples.extend(person_samples)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Returns:
            sample: 包含以下键的字典：
                - features: 源特征 [512]
                - source_pose: 源姿势 [3]
                - target_pose: 目标姿势 [3]
                - target_features: 目标特征 [512]（监督信号）
                - person_name: 人员名称（字符串）
        """
        sample_info = self.samples[idx]
        
        # 加载源特征
        if sample_info['src_type'] == 'front':
            if self.load_in_memory:
                features = self.front_features[sample_info['src_idx']]
            else:
                front_features = np.load(self.front_feature_path, mmap_mode='r')
                features = front_features[sample_info['src_idx']]
        else:
            if self.load_in_memory:
                features = self.video_features[sample_info['src_idx']]
            else:
                video_features = np.load(self.video_feature_path, mmap_mode='r')
                features = video_features[sample_info['src_idx']]
        
        # 加载目标特征
        if sample_info['tgt_type'] == 'front':
            if self.load_in_memory:
                target_features = self.front_features[sample_info['tgt_idx']]
            else:
                front_features = np.load(self.front_feature_path, mmap_mode='r')
                target_features = front_features[sample_info['tgt_idx']]
        else:
            if self.load_in_memory:
                target_features = self.video_features[sample_info['tgt_idx']]
            else:
                video_features = np.load(self.video_feature_path, mmap_mode='r')
                target_features = video_features[sample_info['tgt_idx']]
        
        # 转换为torch tensor
        features = torch.from_numpy(features).float()
        target_features = torch.from_numpy(target_features).float()
        source_pose = torch.from_numpy(sample_info['src_pose']).float()
        target_pose = torch.from_numpy(sample_info['tgt_pose']).float()
        
        return {
            'features': features,
            'source_pose': source_pose,
            'target_pose': target_pose,
            'target_features': target_features,
            'person_name': sample_info['person_name']
        }


def feature_controlnet_collate_fn(batch: List[Dict]) -> Dict:
    """自定义collate函数"""
    batched = {}
    
    # 张量字段
    tensor_fields = ['features', 'source_pose', 'target_pose', 'target_features']
    for field in tensor_fields:
        if field in batch[0]:
            batched[field] = torch.stack([item[field] for item in batch])
    
    # 字符串字段
    string_fields = ['person_name']
    for field in string_fields:
        if field in batch[0]:
            batched[field] = [item[field] for item in batch]
    
    return batched


def create_feature_controlnet_dataloaders(
    data_dir: str = 'train/datas/file',
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_in_memory: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    min_angle_diff: float = 5.0,
    max_samples_per_person: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否使用pin_memory
        load_in_memory: 是否加载到内存
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        min_angle_diff: 最小角度差异
        max_samples_per_person: 每个人的最大样本数
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # 创建完整数据集
    full_dataset = FeatureControlNetDataset(
        data_dir=data_dir,
        load_in_memory=load_in_memory,
        min_angle_diff=min_angle_diff,
        max_samples_per_person=max_samples_per_person
    )
    
    # 按人员分割数据集（确保同一人不会同时出现在训练和验证集）
    person_names = list(set([s['person_name'] for s in full_dataset.samples]))
    np.random.shuffle(person_names)
    
    n_total = len(person_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_persons = set(person_names[:n_train])
    val_persons = set(person_names[n_train:n_train + n_val])
    test_persons = set(person_names[n_train + n_val:])
    
    # 创建索引
    train_indices = [i for i, s in enumerate(full_dataset.samples) if s['person_name'] in train_persons]
    val_indices = [i for i, s in enumerate(full_dataset.samples) if s['person_name'] in val_persons]
    test_indices = [i for i, s in enumerate(full_dataset.samples) if s['person_name'] in test_persons]
    
    # 创建子集
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    logger.info(f"数据集分割:")
    logger.info(f"  训练集: {len(train_dataset)} 样本 ({len(train_persons)} 人)")
    logger.info(f"  验证集: {len(val_dataset)} 样本 ({len(val_persons)} 人)")
    logger.info(f"  测试集: {len(test_dataset)} 样本 ({len(test_persons)} 人)")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_controlnet_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_controlnet_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=feature_controlnet_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """测试数据集"""
    print("=" * 70)
    print("FeatureControlNetDataset 测试")
    print("=" * 70)
    
    # 创建数据集
    dataset = FeatureControlNetDataset(
        data_dir='train/datas/file',
        load_in_memory=True,
        min_angle_diff=5.0
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本键: {sample.keys()}")
        print(f"  特征形状: {sample['features'].shape}")
        print(f"  源姿势形状: {sample['source_pose'].shape}")
        print(f"  目标姿势形状: {sample['target_pose'].shape}")
        print(f"  目标特征形状: {sample['target_features'].shape}")
        print(f"  人员名称: {sample['person_name']}")
    
    # 测试数据加载器
    train_loader, val_loader, test_loader = create_feature_controlnet_dataloaders(
        data_dir='train/datas/file',
        batch_size=4,
        num_workers=0
    )
    
    print(f"\n数据加载器:")
    print(f"  训练集: {len(train_loader)} 批次")
    print(f"  验证集: {len(val_loader)} 批次")
    print(f"  测试集: {len(test_loader)} 批次")
    
    # 测试一个批次
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"\n批次键: {batch.keys()}")
        print(f"  特征形状: {batch['features'].shape}")
        print(f"  源姿势形状: {batch['source_pose'].shape}")
        print(f"  目标姿势形状: {batch['target_pose'].shape}")
        print(f"  目标特征形状: {batch['target_features'].shape}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

