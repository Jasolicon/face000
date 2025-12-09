"""
Transformer训练数据集
- 输入：不同角度图片的DINOv2特征 + 球面角（作为位置编码）
- 输出：features_224中的对应特征（正面图特征）
"""
import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json
from typing import List, Dict, Tuple, Optional
import logging

# 添加父目录到路径，以便导入模块
sys.path.append(str(Path(__file__).parent.parent))

from feature_extractor import DINOv2FeatureExtractor
from feature_manager import FeatureManager
from utils import (
    get_insightface_detector,
    get_insightface_landmarks,
    landmarks_to_3d,
    calculate_spherical_angle
)
from train_transformer.utils_seed import set_seed

# 设置随机种子（数据集初始化时）
set_seed(42)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerFaceDataset(Dataset):
    """
    Transformer训练数据集
    
    输入：
    - DINOv2特征（768维）：从不同角度图片提取
    - 球面角（5维）：5个关键点的角度，作为位置编码
    
    输出：
    - features_224特征（768维）：正面图对应的特征
    """
    
    def __init__(
        self,
        features_224_dir: str = r'C:\Codes\face000\features_224',
        video_dir: str = r'C:\Codes\face000\train\datas\video',
        face_dir: str = r'C:\Codes\face000\train\datas\face',
        valid_images_file: str = None,
        use_cpu: bool = False,
        cache_features: bool = True
    ):
        """
        初始化数据集
        
        Args:
            features_224_dir: features_224特征库目录
            video_dir: 视频帧图片目录
            face_dir: 正面图片目录
            use_cpu: 是否使用CPU
            cache_features: 是否缓存提取的特征
        """
        self.features_224_dir = Path(features_224_dir)
        self.video_dir = Path(video_dir)
        self.face_dir = Path(face_dir)
        self.valid_images_file = valid_images_file
        self.use_cpu = use_cpu
        
        # 初始化特征提取器和检测器
        logger.info("初始化DINOv2特征提取器...")
        self.feature_extractor = DINOv2FeatureExtractor(resize_to_96=False, device=None if not use_cpu else 'cpu')
        
        logger.info("初始化InsightFace检测器...")
        self.detector = get_insightface_detector(use_cpu=use_cpu)
        
        # 加载features_224特征库
        logger.info(f"加载特征库: {self.features_224_dir}")
        self.feature_manager = FeatureManager(storage_dir=str(self.features_224_dir))
        self.features_224, self.metadata_224 = self.feature_manager.get_all_features()
        
        if self.features_224 is None or len(self.features_224) == 0:
            raise ValueError(f"特征库 {self.features_224_dir} 为空或不存在")
        
        logger.info(f"加载了 {len(self.features_224)} 个特征")
        
        # 构建样本列表
        if self.valid_images_file and Path(self.valid_images_file).exists():
            logger.info(f"✓ 从文件加载有效图片列表: {self.valid_images_file}")
            self.samples = self._build_samples_from_file()
            logger.info(f"✓ 使用筛选后的文件，共 {len(self.samples)} 个训练样本")
        else:
            if self.valid_images_file:
                logger.warning(f"⚠️ 指定的有效图片文件不存在: {self.valid_images_file}")
            logger.info("扫描目录构建样本列表（未使用筛选文件）...")
            self.samples = self._build_samples()
            logger.info(f"构建了 {len(self.samples)} 个训练样本")
        
        # 特征缓存
        self.cache_features = cache_features
        self.feature_cache = {} if cache_features else None
    
    def _build_samples(self) -> List[Dict]:
        """
        构建训练样本列表
        
        Returns:
            samples: 样本列表，每个样本包含：
                - video_image_path: 视频帧图片路径
                - face_image_path: 正面图片路径
                - person_name: 人名
                - feature_224_index: features_224中的索引
        """
        samples = []
        
        # 获取所有正面图片的人名
        face_names = set()
        for face_file in self.face_dir.glob('*.jpg'):
            person_name = face_file.stem  # 去掉扩展名
            face_names.add(person_name)
        
        logger.info(f"找到正面图片人名: {face_names}")
        
        # 获取features_224中的人名
        features_224_names = {}
        for idx, meta in enumerate(self.metadata_224):
            person_name = meta.get('person_name')
            if person_name:
                if person_name not in features_224_names:
                    features_224_names[person_name] = []
                features_224_names[person_name].append({
                    'index': idx,
                    'image_path': meta.get('image_path', ''),
                    'metadata': meta
                })
        
        logger.info(f"features_224中的人名: {list(features_224_names.keys())}")
        
        # 找到共有的人名
        common_names = face_names.intersection(set(features_224_names.keys()))
        logger.info(f"共有的人名: {common_names}")
        
        if len(common_names) == 0:
            raise ValueError("没有找到共有的人名！请检查数据路径和特征库。")
        
        # 遍历共有的人名
        for person_name in common_names:
            # 获取该人的正面图片路径
            face_image_path = self.face_dir / f"{person_name}.jpg"
            if not face_image_path.exists():
                logger.warning(f"正面图片不存在: {face_image_path}")
                continue
            
            # 获取该人在features_224中的特征（可能有多个）
            person_features_224 = features_224_names[person_name]
            
            # 尝试找到匹配正面图的特征（优先匹配路径）
            feature_224_index = None
            face_image_path_str = str(face_image_path)
            face_image_name = Path(face_image_path).name
            
            # 首先尝试路径匹配
            face_image_path_normalized = face_image_path_str.replace('\\', '/')
            for feat_info in person_features_224:
                feat_path = feat_info['image_path']
                feat_path_normalized = feat_path.replace('\\', '/')
                feat_name = Path(feat_path).name
                # 检查路径是否匹配（支持完整路径或文件名匹配）
                if (feat_path_normalized == face_image_path_normalized or 
                    feat_name == face_image_name):
                    feature_224_index = feat_info['index']
                    logger.info(f"找到匹配正面图的特征（路径匹配）: {feat_path}")
                    break
            
            # 如果路径匹配失败，使用该人的第一个特征（假设是同一个人，特征应该相似）
            if feature_224_index is None:
                if len(person_features_224) > 0:
                    feature_224_index = person_features_224[0]['index']
                    logger.info(f"使用该人在features_224中的第一个特征（索引: {feature_224_index}）")
                    logger.info(f"正面图路径: {face_image_path_str}")
                    logger.info(f"features_224中的特征路径: {person_features_224[0]['image_path']}")
                else:
                    feature_224_index = None
                    logger.warning(f"该人在features_224中没有特征，将实时提取正面图特征")
            
            # 获取该人的视频帧图片目录
            video_person_dir = self.video_dir / person_name
            if not video_person_dir.exists():
                logger.warning(f"视频目录不存在: {video_person_dir}")
                continue
            
            # 遍历所有视频帧图片
            video_images = sorted(list(video_person_dir.glob('*.jpg')))
            logger.info(f"{person_name}: 找到 {len(video_images)} 张视频帧图片")
            
            for video_image_path in video_images:
                samples.append({
                    'video_image_path': str(video_image_path),
                    'face_image_path': str(face_image_path),
                    'person_name': person_name,
                    'feature_224_index': feature_224_index  # 可能为None，表示需要实时提取
                })
        
        return samples
    
    def _build_samples_from_file(self) -> List[Dict]:
        """
        从文件加载有效图片列表并构建训练样本
        
        Returns:
            samples: 样本列表
        """
        samples = []
        
        # 加载有效图片文件
        with open(self.valid_images_file, 'r', encoding='utf-8') as f:
            valid_images = json.load(f)
        
        logger.info(f"从文件加载了 {len(valid_images)} 个人的数据")
        
        # 获取features_224中的人名
        features_224_names = {}
        for idx, meta in enumerate(self.metadata_224):
            person_name = meta.get('person_name')
            if person_name:
                if person_name not in features_224_names:
                    features_224_names[person_name] = []
                features_224_names[person_name].append({
                    'index': idx,
                    'image_path': meta.get('image_path', ''),
                    'metadata': meta
                })
        
        # 遍历有效图片数据
        for person_name, person_data in valid_images.items():
            # 检查该人是否在features_224中
            if person_name not in features_224_names:
                logger.warning(f"{person_name} 不在features_224中，跳过")
                continue
            
            face_image_path = person_data['face_image_path']
            video_images = person_data['video_images']
            
            # 获取该人在features_224中的特征
            person_features_224 = features_224_names[person_name]
            
            # 尝试找到匹配正面图的特征
            feature_224_index = None
            face_image_name = Path(face_image_path).name
            
            # 首先尝试路径匹配
            face_image_path_normalized = face_image_path.replace('\\', '/')
            for feat_info in person_features_224:
                feat_path = feat_info['image_path']
                feat_path_normalized = feat_path.replace('\\', '/')
                feat_name = Path(feat_path).name
                if (feat_path_normalized == face_image_path_normalized or 
                    feat_name == face_image_name):
                    feature_224_index = feat_info['index']
                    logger.debug(f"找到匹配正面图的特征（路径匹配）: {feat_path}")
                    break
            
            # 如果路径匹配失败，使用该人的第一个特征
            if feature_224_index is None:
                if len(person_features_224) > 0:
                    feature_224_index = person_features_224[0]['index']
                    logger.debug(f"使用该人在features_224中的第一个特征（索引: {feature_224_index}）")
                else:
                    feature_224_index = None
                    logger.warning(f"该人在features_224中没有特征，将实时提取正面图特征")
            
            # 添加样本
            for video_image_path in video_images:
                samples.append({
                    'video_image_path': video_image_path,
                    'face_image_path': face_image_path,
                    'person_name': person_name,
                    'feature_224_index': feature_224_index
                })
            
            logger.info(f"{person_name}: 添加了 {len(video_images)} 个样本")
        
        return samples
    
    def _extract_dinov2_features(self, image_path: str) -> np.ndarray:
        """
        提取DINOv2特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            features: DINOv2特征向量 (768维)
        """
        # 检查缓存
        if self.cache_features and image_path in self.feature_cache:
            return self.feature_cache[image_path]
        
        # 提取特征
        features = self.feature_extractor.extract_features(image_path)
        
        # 缓存
        if self.cache_features:
            self.feature_cache[image_path] = features
        
        return features
    
    def _calculate_spherical_angles(
        self,
        video_image_path: str,
        face_image_path: str,
        skip_warning: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        计算视频帧图片与正面图片的球面角
        
        Args:
            video_image_path: 视频帧图片路径
            face_image_path: 正面图片路径
            skip_warning: 是否跳过警告（如果使用筛选后的文件，可以跳过）
            
        Returns:
            angles: 5个关键点的球面角 (5维)
            avg_angle: 平均角度（可选）
        """
        # 检测正面图片的关键点
        face_landmarks, face_box = get_insightface_landmarks(self.detector, face_image_path)
        if face_landmarks is None or face_box is None:
            if not skip_warning:
                logger.warning(f"无法检测正面图片关键点: {face_image_path}")
            return np.zeros(5), None
        
        # 检测视频帧图片的关键点
        video_landmarks, video_box = get_insightface_landmarks(self.detector, video_image_path)
        if video_landmarks is None or video_box is None:
            if not skip_warning:
                logger.warning(f"无法检测视频帧图片关键点: {video_image_path}")
            return np.zeros(5), None
        
        # 获取图片尺寸
        face_img = Image.open(face_image_path)
        video_img = Image.open(video_image_path)
        face_width, face_height = face_img.size
        video_width, video_height = video_img.size
        
        # 转换为3D坐标
        face_landmarks_3d = landmarks_to_3d(face_landmarks, face_box, face_width, face_height)
        video_landmarks_3d = landmarks_to_3d(video_landmarks, video_box, video_width, video_height)
        
        # 计算球面角
        angles, avg_angle = calculate_spherical_angle(
            face_landmarks_3d,
            video_landmarks_3d,
            face_landmarks,
            video_landmarks
        )
        
        return angles, avg_angle
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            sample: 包含以下键的字典：
                - 'input_features': DINOv2特征 (768维)
                - 'position_encoding': 球面角 (5维)
                - 'target_features': features_224特征 (768维)
                - 'person_name': 人名（字符串）
        """
        sample_info = self.samples[idx]
        
        # 提取DINOv2特征（输入）
        input_features = self._extract_dinov2_features(sample_info['video_image_path'])
        
        # 计算球面角（位置编码）
        # 如果使用了筛选后的文件，跳过警告（因为已经筛选过了）
        use_valid_file = (self.valid_images_file is not None and Path(self.valid_images_file).exists())
        angles, _ = self._calculate_spherical_angles(
            sample_info['video_image_path'],
            sample_info['face_image_path'],
            skip_warning=use_valid_file
        )
        
        # 如果使用了筛选文件但仍然检测不到关键点，记录错误（但不输出警告）
        if use_valid_file and np.allclose(angles, 0):
            logger.error(f"筛选文件中的图片无法检测关键点（这不应该发生）: {sample_info['video_image_path']}")
        
        # 获取features_224特征（目标）
        # 如果feature_224_index为None，直接从正面图提取特征
        if sample_info['feature_224_index'] is not None:
            target_features = self.features_224[sample_info['feature_224_index']]
        else:
            # 实时从正面图提取特征
            logger.debug(f"实时提取正面图特征: {sample_info['face_image_path']}")
            target_features = self._extract_dinov2_features(sample_info['face_image_path'])
        
        # 计算残差：正面特征 - 角度特征
        residual = target_features - input_features
        
        # 转换为torch tensor
        return {
            'input_features': torch.FloatTensor(input_features),  # [768] 角度图特征
            'position_encoding': torch.FloatTensor(angles),  # [5] 球面角
            'target_features': torch.FloatTensor(target_features),  # [768] 正面图特征（用于验证）
            'target_residual': torch.FloatTensor(residual),  # [768] 残差（训练目标）
            'person_name': sample_info['person_name']
        }


def create_dataloader(
    dataset: TransformerFaceDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    创建DataLoader
    
    Args:
        dataset: TransformerFaceDataset实例
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        pin_memory: 是否使用pin_memory
        
    Returns:
        dataloader: DataLoader实例
    """
    def collate_fn(batch):
        """
        自定义collate函数，处理批次数据
        """
        input_features = torch.stack([item['input_features'] for item in batch])  # [B, 768]
        position_encoding = torch.stack([item['position_encoding'] for item in batch])  # [B, 5]
        target_features = torch.stack([item['target_features'] for item in batch])  # [B, 768] 用于验证
        target_residual = torch.stack([item['target_residual'] for item in batch])  # [B, 768] 残差（训练目标）
        person_names = [item['person_name'] for item in batch]  # [B]
        
        return {
            'input_features': input_features,
            'position_encoding': position_encoding,
            'target_features': target_features,
            'target_residual': target_residual,
            'person_names': person_names
        }
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据集
    print("=" * 70)
    print("测试 TransformerFaceDataset")
    print("=" * 70)
    
    # 创建数据集
    dataset = TransformerFaceDataset(
        features_224_dir=r'C:\Codes\face000\features_224',
        video_dir=r'C:\Codes\face000\train\datas\video',
        face_dir=r'C:\Codes\face000\train\datas\face',
        use_cpu=False,
        cache_features=True
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取一个样本
    print("\n获取第一个样本...")
    sample = dataset[0]
    print(f"输入特征形状: {sample['input_features'].shape}")
    print(f"位置编码形状: {sample['position_encoding'].shape}")
    print(f"目标特征形状: {sample['target_features'].shape}")
    print(f"人名: {sample['person_name']}")
    print(f"位置编码值: {sample['position_encoding'].numpy()}")
    
    # 创建DataLoader
    print("\n创建DataLoader...")
    dataloader = create_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 测试一个批次
    print("\n测试一个批次...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n批次 {batch_idx}:")
        print(f"  输入特征形状: {batch['input_features'].shape}")
        print(f"  位置编码形状: {batch['position_encoding'].shape}")
        print(f"  目标特征形状: {batch['target_features'].shape}")
        print(f"  人名: {batch['person_names']}")
        print(f"  位置编码值:\n{batch['position_encoding'].numpy()}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

