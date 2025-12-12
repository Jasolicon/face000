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
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import logging

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.utils_seed import set_seed

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Aligned3DFaceDataset(Dataset):
    """
    从对齐和归一化后的npy文件读取数据的3D数据集
    
    数据格式：
    - front_feature.npy: 正面图特征 [N, feature_dim]
    - front_keypoints.npy: 正面图对齐归一化后的关键点 [N, 5, 3]
    - front_metadata.json: 正面图元数据
    - video_feature.npy: 视频帧特征 [M, feature_dim]
    - video_keypoints.npy: 视频帧对齐归一化后的关键点 [M, 5, 3]
    - video_metadata.json: 视频帧元数据
    """
    
    def __init__(
        self,
        data_dir: str = 'train/datas/file',
        load_in_memory: bool = True
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录（包含front_*.npy和video_*.npy文件）
            load_in_memory: 是否将所有数据加载到内存（默认True，速度快但占用内存）
        """
        self.data_dir = Path(data_dir)
        self.load_in_memory = load_in_memory
        
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
        logger.info(f"构建了 {len(self.samples)} 个训练样本")
        
        # 验证数据一致性
        self._validate_data()
    
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
                - src: 视频帧特征 [feature_dim]
                - tgt: 正面图特征 [feature_dim]
                - keypoints_3d: 视频帧3D关键点 [5, 3]（已对齐和归一化）
                - pose: 视频帧姿态向量 [3] (欧拉角: yaw, pitch, roll)
                - angles: 角度（兼容性，使用pose）[3]
                - front_keypoints_3d: 正面图3D关键点 [5, 3]（可选，已对齐和归一化）
                - front_pose: 正面图姿态向量 [3]（可选）
        """
        sample_info = self.samples[idx]
        front_idx = sample_info['front_idx']
        video_idx = sample_info['video_idx']
        
        # 加载数据
        if self.load_in_memory:
            # 从内存读取
            src = self.video_features[video_idx]  # [feature_dim]
            tgt = self.front_features[front_idx]  # [feature_dim]
            keypoints_3d = self.video_keypoints[video_idx]  # [5, 3]
            front_keypoints_3d = self.front_keypoints[front_idx]  # [5, 3]
            pose = self.video_poses[video_idx]  # [3]
            front_pose = self.front_poses[front_idx]  # [3]
        else:
            # 从文件读取（使用内存映射）
            video_features = np.load(self.video_feature_path, mmap_mode='r')
            front_features = np.load(self.front_feature_path, mmap_mode='r')
            video_keypoints = np.load(self.video_keypoints_path, mmap_mode='r')
            front_keypoints = np.load(self.front_keypoints_path, mmap_mode='r')
            
            src = video_features[video_idx]
            tgt = front_features[front_idx]
            keypoints_3d = video_keypoints[video_idx]
            front_keypoints_3d = front_keypoints[front_idx]
            pose = self.video_poses[video_idx]
            front_pose = self.front_poses[front_idx]
        
        # 转换为torch tensor
        src = torch.from_numpy(src).float()
        tgt = torch.from_numpy(tgt).float()
        keypoints_3d = torch.from_numpy(keypoints_3d).float()
        front_keypoints_3d = torch.from_numpy(front_keypoints_3d).float()
        pose = torch.from_numpy(pose).float()
        front_pose = torch.from_numpy(front_pose).float()
        
        # angles用于兼容性（使用pose）
        angles = pose.clone()
        
        return {
            'src': src,
            'tgt': tgt,
            'keypoints_3d': keypoints_3d,
            'pose': pose,
            'angles': angles,
            'front_keypoints_3d': front_keypoints_3d,  # 可选：正面图关键点
            'front_pose': front_pose,  # 可选：正面图姿态
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


def create_dataloader(
    data_dir: str = 'train/datas/file',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    load_in_memory: bool = True
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载器工作进程数
        pin_memory: 是否使用pin_memory
        load_in_memory: 是否将所有数据加载到内存
    
    Returns:
        DataLoader: 数据加载器
    """
    dataset = Aligned3DFaceDataset(
        data_dir=data_dir,
        load_in_memory=load_in_memory
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 丢弃最后一个不完整的批次
    )
    
    return dataloader


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
