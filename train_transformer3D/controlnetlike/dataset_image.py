"""
模型2的数据集：图像生成ControlNet
数据格式：
- 输入：图片
- 控制：目标姿势
- 输出：目标角度的图片

需要从原始图片数据加载，或从特征重建图片
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
from PIL import Image
import cv2

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


class ImageControlDataset(Dataset):
    """
    图像生成ControlNet数据集
    
    数据格式：
    - 需要原始图片路径和元数据
    - 每个样本包含：
        - source_image: 源图像 [3, H, W]
        - target_pose: 目标姿势（控制姿势）[pose_dim]
        - target_image: 目标角度的图像 [3, H, W]（ground truth）
        - person_name: 人员名称
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 112,
        max_samples: Optional[int] = None,
        min_angle_diff: float = 5.0,
        max_angle_diff: float = 90.0,
        image_dir: Optional[str] = None  # 图片目录（如果与data_dir不同）
    ):
        """
        Args:
            data_dir: 数据目录路径（包含metadata和keypoints）
            image_size: 图像尺寸
            max_samples: 最大样本数
            min_angle_diff: 最小角度差异
            max_angle_diff: 最大角度差异
            image_dir: 图片目录（如果为None，使用data_dir）
        """
        self.data_dir = Path(data_dir)
        self.image_dir = Path(image_dir) if image_dir else self.data_dir
        self.image_size = image_size
        self.min_angle_diff = min_angle_diff
        self.max_angle_diff = max_angle_diff
        
        # 加载元数据
        front_metadata_path = self.data_dir / 'front_metadata.json'
        video_metadata_path = self.data_dir / 'video_metadata.json'
        
        if not front_metadata_path.exists():
            raise FileNotFoundError(f"找不到文件: {front_metadata_path}")
        if not video_metadata_path.exists():
            raise FileNotFoundError(f"找不到文件: {video_metadata_path}")
        
        with open(front_metadata_path, 'r', encoding='utf-8') as f:
            self.front_metadata = json.load(f)
        with open(video_metadata_path, 'r', encoding='utf-8') as f:
            self.video_metadata = json.load(f)
        
        # 加载关键点（用于提取姿势）
        front_keypoints_path = self.data_dir / 'front_keypoints.npy'
        video_keypoints_path = self.data_dir / 'video_keypoints.npy'
        
        if not front_keypoints_path.exists():
            raise FileNotFoundError(f"找不到文件: {front_keypoints_path}")
        if not video_keypoints_path.exists():
            raise FileNotFoundError(f"找不到文件: {video_keypoints_path}")
        
        self.front_keypoints = np.load(front_keypoints_path)  # [N, 5, 3]
        self.video_keypoints = np.load(video_keypoints_path)  # [M, 5, 3]
        
        print(f"正面图数量: {len(self.front_metadata)}")
        print(f"视频帧数量: {len(self.video_metadata)}")
        
        # 构建样本对
        self.samples = self._build_samples()
        
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"总样本数: {len(self.samples)}")
    
    def _extract_pose_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """从关键点提取姿势（欧拉角）"""
        if keypoints.shape[0] >= 2:
            left_eye = keypoints[0, 0]
            right_eye = keypoints[1, 0]
            yaw = np.arctan2(right_eye - left_eye, 1.0) * 180 / np.pi
        else:
            yaw = 0.0
        
        if keypoints.shape[0] >= 3:
            nose = keypoints[2, 1]
            eye_y = (keypoints[0, 1] + keypoints[1, 1]) / 2
            pitch = np.arctan2(nose - eye_y, 1.0) * 180 / np.pi
        else:
            pitch = 0.0
        
        if keypoints.shape[0] >= 2:
            roll = np.arctan2(keypoints[1, 1] - keypoints[0, 1], keypoints[1, 0] - keypoints[0, 0]) * 180 / np.pi
        else:
            roll = 0.0
        
        return np.array([yaw, pitch, roll])
    
    def _calculate_angle_diff(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """计算两个姿势的角度差异"""
        diff = pose1 - pose2
        return np.linalg.norm(diff)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        加载图像并预处理
        
        Args:
            image_path: 图像路径
        Returns:
            image: 预处理后的图像 [3, H, W]
        """
        if not os.path.exists(image_path):
            # 如果图片不存在，返回零图像
            return torch.zeros(3, self.image_size, self.image_size)
        
        try:
            # 使用OpenCV加载（支持更多格式）
            img = cv2.imread(image_path)
            if img is None:
                return torch.zeros(3, self.image_size, self.image_size)
            
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            # 转换为tensor并归一化到[-1, 1]
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            img = img / 255.0 * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            
            return img
        except Exception as e:
            logger.warning(f"加载图像失败 {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _build_samples(self) -> List[Dict]:
        """构建样本对"""
        samples = []
        
        # 按person_name组织数据
        front_by_person = defaultdict(list)
        video_by_person = defaultdict(list)
        
        # 组织正面图数据
        for i, metadata in enumerate(self.front_metadata):
            person_name = metadata.get('person_name', f'person_{i}')
            image_path = metadata.get('image_path', '')
            if not image_path:
                # 尝试从其他字段获取路径
                image_path = metadata.get('file_path', '')
            
            front_by_person[person_name].append({
                'index': i,
                'image_path': image_path,
                'keypoints': self.front_keypoints[i],
                'metadata': metadata
            })
        
        # 组织视频帧数据
        for i, metadata in enumerate(self.video_metadata):
            person_name = metadata.get('person_name', f'person_{i}')
            image_path = metadata.get('image_path', '')
            if not image_path:
                image_path = metadata.get('file_path', '')
            
            video_by_person[person_name].append({
                'index': i,
                'image_path': image_path,
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
                front_keypoints = front_sample['keypoints']
                front_pose = self._extract_pose_from_keypoints(front_keypoints)
                front_image_path = front_sample['image_path']
                
                # 构建完整路径
                if not os.path.isabs(front_image_path):
                    front_image_path = self.image_dir / front_image_path
                else:
                    front_image_path = Path(front_image_path)
                
                for video_sample in video_samples:
                    video_keypoints = video_sample['keypoints']
                    video_pose = self._extract_pose_from_keypoints(video_keypoints)
                    video_image_path = video_sample['image_path']
                    
                    # 构建完整路径
                    if not os.path.isabs(video_image_path):
                        video_image_path = self.image_dir / video_image_path
                    else:
                        video_image_path = Path(video_image_path)
                    
                    # 计算角度差异
                    angle_diff = self._calculate_angle_diff(front_pose, video_pose)
                    
                    # 筛选有效样本
                    if self.min_angle_diff <= angle_diff <= self.max_angle_diff:
                        samples.append({
                            'source_image_path': str(front_image_path),
                            'target_image_path': str(video_image_path),
                            'target_pose': video_pose.astype(np.float32),
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
                - source_image: 源图像 [3, H, W]
                - target_pose: 目标姿势（控制姿势）[pose_dim]
                - target_image: 目标角度的图像 [3, H, W]
                - person_name: 人员名称
        """
        sample = self.samples[idx]
        
        # 加载图像
        source_image = self._load_image(sample['source_image_path'])
        target_image = self._load_image(sample['target_image_path'])
        
        return {
            'source_image': source_image,
            'target_pose': torch.from_numpy(sample['target_pose']),
            'target_image': target_image,
            'person_name': sample['person_name'],
            'front_idx': sample['front_idx'],
            'video_idx': sample['video_idx']
        }


def image_control_collate_fn(batch):
    """自定义collate函数"""
    batched = {}
    
    # Tensor字段
    tensor_fields = ['source_image', 'target_pose', 'target_image']
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


def create_image_control_dataloader(
    data_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 112,
    max_samples: Optional[int] = None,
    image_dir: Optional[str] = None,
    min_angle_diff: float = 5.0,
    max_angle_diff: float = 90.0
) -> DataLoader:
    """创建图像生成ControlNet的数据加载器"""
    dataset = ImageControlDataset(
        data_dir=data_dir,
        image_size=image_size,
        max_samples=max_samples,
        image_dir=image_dir,
        min_angle_diff=min_angle_diff,
        max_angle_diff=max_angle_diff
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=image_control_collate_fn
    )
    
    return dataloader


def create_image_control_train_val_test_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    pin_memory: bool = True,
    image_size: int = 112,
    max_samples: Optional[int] = None,
    image_dir: Optional[str] = None,
    min_angle_diff: float = 5.0,
    max_angle_diff: float = 90.0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试数据加载器（按person_name分割）"""
    # 创建完整数据集
    full_dataset = ImageControlDataset(
        data_dir=data_dir,
        image_size=image_size,
        max_samples=max_samples,
        image_dir=image_dir,
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
        collate_fn=image_control_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=image_control_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=image_control_collate_fn
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """测试数据集"""
    print("=" * 70)
    print("图像生成ControlNet数据集测试")
    print("=" * 70)
    
    # 创建数据集
    data_dir = 'train/datas/file'
    dataset = ImageControlDataset(data_dir=data_dir, max_samples=10)
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本字段:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
