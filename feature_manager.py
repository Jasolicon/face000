"""
特征保存和管理模块
"""
import numpy as np
import json
import os
from pathlib import Path
import pickle


class FeatureManager:
    """特征管理器，负责特征的保存、加载和管理"""
    
    def __init__(self, storage_dir='features'):
        """
        初始化特征管理器
        
        Args:
            storage_dir: 特征存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.features_file = self.storage_dir / 'features.npy'
        self.metadata_file = self.storage_dir / 'metadata.json'
        
        # 加载已有数据
        self.features = None
        self.metadata = []
        self.load_features()
    
    def save_feature(self, feature_vector, image_path, person_id=None, person_name=None):
        """
        保存单个特征向量
        
        Args:
            feature_vector: 特征向量 (numpy array)
            image_path: 原始图像路径
            person_id: 人员ID（可选）
            person_name: 人员姓名（可选）
        """
        feature_vector = np.array(feature_vector).flatten()
        
        # 创建元数据
        metadata_entry = {
            'image_path': str(image_path),
            'person_id': person_id,
            'person_name': person_name,
            'feature_dim': len(feature_vector),
            'extractor_type': None  # 将在保存时设置
        }
        
        # 添加到特征矩阵
        if self.features is None:
            self.features = feature_vector.reshape(1, -1)
        else:
            # 检查特征维度是否一致
            if self.features.shape[1] != len(feature_vector):
                raise ValueError(
                    f"特征维度不匹配: 已有特征维度为 {self.features.shape[1]}, "
                    f"新特征维度为 {len(feature_vector)}"
                )
            self.features = np.vstack([self.features, feature_vector])
        
        # 添加元数据
        metadata_entry['index'] = len(self.metadata)
        self.metadata.append(metadata_entry)
        
        # 保存到文件
        self._save_to_disk()
    
    def save_batch_features(self, feature_vectors, image_paths, person_ids=None, person_names=None):
        """
        批量保存特征向量
        
        Args:
            feature_vectors: 特征向量列表或矩阵 (numpy array)
            image_paths: 图像路径列表
            person_ids: 人员ID列表（可选）
            person_names: 人员姓名列表（可选）
        """
        feature_vectors = np.array(feature_vectors)
        
        if person_ids is None:
            person_ids = [None] * len(image_paths)
        if person_names is None:
            person_names = [None] * len(image_paths)
        
        for i, (feature, img_path, pid, pname) in enumerate(
            zip(feature_vectors, image_paths, person_ids, person_names)
        ):
            self.save_feature(feature, img_path, pid, pname)
    
    def load_features(self):
        """从磁盘加载特征和元数据"""
        if self.features_file.exists():
            self.features = np.load(self.features_file)
        
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
    
    def _save_to_disk(self):
        """保存特征和元数据到磁盘"""
        if self.features is not None:
            np.save(self.features_file, self.features)
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def get_feature(self, index):
        """
        根据索引获取特征向量
        
        Args:
            index: 特征索引
            
        Returns:
            feature: 特征向量
            metadata: 对应的元数据
        """
        if self.features is None or index >= len(self.features):
            raise IndexError(f"索引 {index} 超出范围")
        
        return self.features[index], self.metadata[index]
    
    def get_all_features(self):
        """
        获取所有特征和元数据
        
        Returns:
            features: 所有特征矩阵
            metadata: 所有元数据列表
        """
        return self.features, self.metadata
    
    def delete_feature(self, index):
        """
        删除指定索引的特征
        
        Args:
            index: 要删除的特征索引
        """
        if self.features is None or index >= len(self.features):
            raise IndexError(f"索引 {index} 超出范围")
        
        # 删除特征
        self.features = np.delete(self.features, index, axis=0)
        
        # 删除元数据
        del self.metadata[index]
        
        # 更新索引
        for i, meta in enumerate(self.metadata):
            meta['index'] = i
        
        # 保存到磁盘
        self._save_to_disk()
    
    def clear_all(self):
        """清空所有特征和元数据"""
        self.features = None
        self.metadata = []
        self._save_to_disk()
    
    def get_count(self):
        """获取已保存的特征数量"""
        if self.features is None:
            return 0
        return len(self.features)

