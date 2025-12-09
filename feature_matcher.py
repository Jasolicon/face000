"""
特征比对模块
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import torch


class FeatureMatcher:
    """特征比对器，用于计算特征相似度"""
    
    def __init__(self, similarity_metric='cosine'):
        """
        初始化特征比对器
        
        Args:
            similarity_metric: 相似度度量方法 ('cosine' 或 'euclidean')
        """
        self.similarity_metric = similarity_metric
    
    def compute_similarity(self, feature1, feature2):
        """
        计算两个特征向量的相似度
        
        Args:
            feature1: 特征向量1 (numpy array)
            feature2: 特征向量2 (numpy array)
            
        Returns:
            similarity: 相似度分数 (0-1之间，值越大越相似)
        """
        feature1 = np.array(feature1).flatten()
        feature2 = np.array(feature2).flatten()
        
        if len(feature1) != len(feature2):
            raise ValueError("特征向量维度不匹配")
        
        if self.similarity_metric == 'cosine':
            # 余弦相似度
            similarity = np.dot(feature1, feature2) / (
                np.linalg.norm(feature1) * np.linalg.norm(feature2)
            )
            # 将相似度归一化到0-1范围（余弦相似度范围是-1到1）
            similarity = (similarity + 1) / 2
        elif self.similarity_metric == 'euclidean':
            # 欧氏距离（转换为相似度）
            distance = np.linalg.norm(feature1 - feature2)
            # 使用指数函数将距离转换为相似度
            similarity = np.exp(-distance)
        else:
            raise ValueError(f"不支持的相似度度量方法: {self.similarity_metric}")
        
        return float(similarity)
    
    def find_best_match(self, query_feature, feature_matrix, top_k=1):
        """
        在特征矩阵中找到与查询特征最相似的top_k个特征
        
        Args:
            query_feature: 查询特征向量 (numpy array)
            feature_matrix: 特征矩阵 (numpy array, shape: [n_samples, feature_dim])
            top_k: 返回最相似的top_k个结果
            
        Returns:
            indices: 最相似特征的索引列表
            similarities: 对应的相似度分数列表
        """
        query_feature = np.array(query_feature).flatten()
        feature_matrix = np.array(feature_matrix)
        
        if len(query_feature) != feature_matrix.shape[1]:
            raise ValueError("特征维度不匹配")
        
        # 计算所有相似度
        similarities = []
        for i in range(len(feature_matrix)):
            sim = self.compute_similarity(query_feature, feature_matrix[i])
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 获取top_k个最相似的索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_similarities = similarities[top_indices]
        
        return top_indices.tolist(), top_similarities.tolist()
    
    def match_with_threshold(self, query_feature, feature_matrix, threshold=0.7):
        """
        使用阈值进行特征匹配
        
        Args:
            query_feature: 查询特征向量
            feature_matrix: 特征矩阵
            threshold: 相似度阈值
            
        Returns:
            matches: 匹配结果列表，每个元素为 (index, similarity)
        """
        query_feature = np.array(query_feature).flatten()
        feature_matrix = np.array(feature_matrix)
        
        matches = []
        for i in range(len(feature_matrix)):
            sim = self.compute_similarity(query_feature, feature_matrix[i])
            if sim >= threshold:
                matches.append((i, sim))
        
        # 按相似度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches


class FaceMatcher:
    """人脸特征比对器（封装了完整的比对流程）"""
    
    def __init__(self, feature_manager, matcher=None, similarity_threshold=0.7):
        """
        初始化人脸比对器
        
        Args:
            feature_manager: FeatureManager实例
            matcher: FeatureMatcher实例，如果为None则创建默认实例
            similarity_threshold: 相似度阈值
        """
        self.feature_manager = feature_manager
        self.matcher = matcher if matcher is not None else FeatureMatcher()
        self.similarity_threshold = similarity_threshold
    
    def match_face(self, query_feature, top_k=5):
        """
        比对查询特征与已保存的特征
        
        Args:
            query_feature: 查询特征向量
            top_k: 返回最相似的top_k个结果
            
        Returns:
            results: 匹配结果列表，每个元素包含：
                - index: 特征索引
                - similarity: 相似度分数
                - metadata: 对应的元数据
        """
        features, metadata = self.feature_manager.get_all_features()
        
        if features is None or len(features) == 0:
            return []
        
        # 找到最佳匹配
        indices, similarities = self.matcher.find_best_match(
            query_feature, features, top_k=top_k
        )
        
        # 构建结果
        results = []
        for idx, sim in zip(indices, similarities):
            if sim >= self.similarity_threshold:
                result = {
                    'index': int(idx),
                    'similarity': float(sim),
                    'metadata': metadata[idx]
                }
                results.append(result)
        
        return results
    
    def identify_person(self, query_feature):
        """
        识别人员身份（返回最相似的一个）
        
        Args:
            query_feature: 查询特征向量
            
        Returns:
            result: 识别结果，包含相似度最高的匹配信息，如果没有匹配则返回None
        """
        matches = self.match_face(query_feature, top_k=1)
        
        if matches:
            return matches[0]
        return None

