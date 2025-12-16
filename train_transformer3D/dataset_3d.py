"""
3D增强的Transformer训练数据集（旧格式）
支持3D关键点和姿态数据

⚠️ 注意：这是旧版本的数据集类，使用单个JSON文件格式。
   新版本请使用 dataset.py 中的 Aligned3DFaceDataset，它使用更高效的.npy格式
   并且数据已经对齐和归一化。

保留此文件用于：
1. 兼容旧的 valid_images_3d.json 数据格式
2. 作为参考实现
3. 临时测试用途

生产环境请使用 dataset.py
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional
import logging

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.utils_seed import set_seed

# 设置随机种子
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerFaceDataset3D(Dataset):
    """
    3D增强的Transformer训练数据集
    
    输入：
    - DINOv2特征（768维）：从不同角度图片提取
    - 3D关键点 [5, 3]：5个关键点的3D坐标
    - 姿态向量 [3]：欧拉角 (yaw, pitch, roll)
    
    输出：
    - 正面图特征（768维）：正面图对应的特征
    """
    
    def __init__(
        self,
        valid_images_3d_file: str = 'train_transformer3D/valid_images_3d.json',
        cache_features: bool = True
    ):
        """
        初始化3D数据集
        
        Args:
            valid_images_3d_file: 包含3D关键点和姿态的JSON文件路径
            cache_features: 是否缓存特征（默认True）
        """
        self.valid_images_3d_file = Path(valid_images_3d_file)
        self.cache_features = cache_features
        
        # 加载数据
        if not self.valid_images_3d_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.valid_images_3d_file}")
        
        logger.info(f"加载3D数据文件: {self.valid_images_3d_file}")
        with open(self.valid_images_3d_file, 'r', encoding='utf-8') as f:
            self.valid_data = json.load(f)
        
        logger.info(f"加载了 {len(self.valid_data)} 个人的数据")
        
        # 构建样本列表
        self.samples = self._build_samples()
        logger.info(f"构建了 {len(self.samples)} 个训练样本")
        
        # 特征缓存
        self.feature_cache = {} if cache_features else None
        
        # 检查数据格式
        self._validate_data_format()
    
    def _validate_data_format(self):
        """验证数据格式"""
        if len(self.samples) == 0:
            raise ValueError("没有有效的训练样本")
        
        # 检查第一个样本
        sample = self.samples[0]
        person_name = sample['person_name']
        person_data = self.valid_data[person_name]
        
        # 检查必需字段
        required_fields = [
            'face_features', 'face_landmarks_3d', 'face_euler_angles',
            'video_data'
        ]
        for field in required_fields:
            if field not in person_data:
                raise ValueError(f"缺少必需字段: {field}")
        
        # 检查video_data中的字段
        if len(person_data['video_data']) > 0:
            video_item = person_data['video_data'][0]
            required_video_fields = [
                'features', 'landmarks_3d', 'euler_angles'
            ]
            for field in required_video_fields:
                if field not in video_item:
                    raise ValueError(f"video_data中缺少必需字段: {field}")
        
        logger.info("✓ 数据格式验证通过")
    
    def _build_samples(self) -> List[Dict]:
        """
        构建训练样本列表
        
        Returns:
            samples: 样本列表，每个样本包含：
                - person_name: 人名
                - video_index: 视频帧索引
                - face_features: 正面图特征
                - video_features: 视频帧特征
                - landmarks_3d: 3D关键点
                - pose: 姿态向量（欧拉角）
        """
        samples = []
        
        for person_name, person_data in self.valid_data.items():
            # 获取正面图特征
            face_features = np.array(person_data['face_features'], dtype=np.float32)
            
            # 获取正面图的3D关键点和姿态（作为参考）
            face_landmarks_3d = np.array(person_data['face_landmarks_3d'], dtype=np.float32)
            face_euler_angles = np.array(person_data.get('face_euler_angles', [0.0, 0.0, 0.0]), dtype=np.float32)
            
            # 遍历视频帧
            for video_index, video_item in enumerate(person_data['video_data']):
                video_features = np.array(video_item['features'], dtype=np.float32)
                video_landmarks_3d = np.array(video_item['landmarks_3d'], dtype=np.float32)
                video_euler_angles = np.array(video_item.get('euler_angles', [0.0, 0.0, 0.0]), dtype=np.float32)
                
                sample = {
                    'person_name': person_name,
                    'video_index': video_index,
                    'face_features': face_features,
                    'video_features': video_features,
                    'landmarks_3d': video_landmarks_3d,
                    'pose': video_euler_angles,
                    'face_landmarks_3d': face_landmarks_3d,  # 保留正面图关键点（可选）
                    'face_pose': face_euler_angles  # 保留正面图姿态（可选）
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本
        
        Returns:
            sample: 包含以下键的字典：
                - src: 视频帧特征 [feature_dim]
                - tgt: 正面图特征 [feature_dim]
                - keypoints_3d: 3D关键点 [num_keypoints, 3]
                - pose: 姿态向量 [3]
                - angles: 角度（兼容性，使用pose）[3]
        """
        sample = self.samples[idx]
        
        # 转换为torch tensor
        src = torch.from_numpy(sample['video_features']).float()
        tgt = torch.from_numpy(sample['face_features']).float()
        keypoints_3d = torch.from_numpy(sample['landmarks_3d']).float()
        pose = torch.from_numpy(sample['pose']).float()
        
        # angles用于兼容性（使用pose）
        angles = pose.clone()
        
        return {
            'src': src,
            'tgt': tgt,
            'keypoints_3d': keypoints_3d,
            'pose': pose,
            'angles': angles,
            'person_name': sample['person_name'],
            'video_index': sample['video_index']
        }


def create_dataloader_3d(
    valid_images_3d_file: str = 'train_transformer3D/valid_images_3d.json',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_features: bool = True
) -> DataLoader:
    """
    创建3D数据加载器
    
    Args:
        valid_images_3d_file: 包含3D关键点和姿态的JSON文件路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载器工作进程数
        pin_memory: 是否使用pin_memory
        cache_features: 是否缓存特征
        
    Returns:
        DataLoader: 数据加载器
    """
    dataset = TransformerFaceDataset3D(
        valid_images_3d_file=valid_images_3d_file,
        cache_features=cache_features
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
    print("测试 TransformerFaceDataset3D 数据集")
    print("=" * 70)
    
    # 检查数据文件是否存在
    data_file = 'train_transformer3D/valid_images_3d.json'
    if not Path(data_file).exists():
        print(f"⚠️  数据文件不存在: {data_file}")
        print("请先运行 filter_valid_images_3d.py 生成数据文件")
    else:
        # 创建数据集
        print(f"\n创建数据集（数据文件: {data_file}）...")
        try:
            dataset = TransformerFaceDataset3D(valid_images_3d_file=data_file)
            
            print(f"数据集大小: {len(dataset)}")
            
            # 获取一个样本
            print("\n获取第一个样本...")
            sample = dataset[0]
            
            print(f"样本键: {sample.keys()}")
            print(f"src形状: {sample['src'].shape}")
            print(f"tgt形状: {sample['tgt'].shape}")
            print(f"keypoints_3d形状: {sample['keypoints_3d'].shape}")
            print(f"pose形状: {sample['pose'].shape}")
            print(f"angles形状: {sample['angles'].shape}")
            
            # 创建数据加载器
            print("\n创建数据加载器...")
            dataloader = create_dataloader_3d(
                valid_images_3d_file=data_file,
                batch_size=4,
                shuffle=False,
                num_workers=0  # Windows上使用0
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
