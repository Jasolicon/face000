"""
Transformer训练数据集
- 输入：不同角度图片的DINOv2特征 + 球面角（作为位置编码）
- 输出：features_224中的对应特征（正面图特征）
"""
import os
import sys
from pathlib import Path

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # 禁用 hf_transfer
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时
    os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '5'    # 重试5次

# 尝试导入 setup_mirrors（如果存在）
try:
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from setup_mirrors import setup_all_mirrors
    setup_all_mirrors()
except ImportError:
    pass  # 如果 setup_mirrors 不存在，使用上面的默认设置

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
        features_224_dir: str = r'root/face000/features_224',
        video_dir: str = r'root/face000/train/datas/video',
        face_dir: str = r'root/face000/train/datas/face',
        valid_images_file: str = None,
        use_cpu: bool = False,
        cache_features: bool = True,
        dinov2_model_name: str = 'dinov2_vitb14'
    ):
        """
        初始化数据集
        
        Args:
            features_224_dir: features_224特征库目录
            video_dir: 视频帧图片目录
            face_dir: 正面图片目录
            use_cpu: 是否使用CPU
            cache_features: 是否缓存提取的特征
            dinov2_model_name: DINOv2模型名称，默认'dinov2_vitb14'（768维）
                - 'dinov2_vits14': 小模型，384维
                - 'dinov2_vitb14': 中等模型，768维（推荐）
                - 'dinov2_vitl14': 大模型，1024维
                - 'dinov2_vitg14': 超大模型，1536维
        """
        self.features_224_dir = Path(features_224_dir)
        self.video_dir = Path(video_dir)
        self.face_dir = Path(face_dir)
        self.valid_images_file = valid_images_file
        self.use_cpu = use_cpu
        
        # 初始化特征提取器和检测器
        logger.info(f"初始化DINOv2特征提取器（模型: {dinov2_model_name}）...")
        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=dinov2_model_name,
            resize_to_96=False,
            device=None if not use_cpu else 'cpu'
        )
        
        logger.info("初始化InsightFace检测器...")
        self.detector = get_insightface_detector(use_cpu=use_cpu)
        
        # 加载features_224特征库
        logger.info(f"加载特征库: {self.features_224_dir}")
        self.feature_manager = FeatureManager(storage_dir=str(self.features_224_dir))
        self.features_224, self.metadata_224 = self.feature_manager.get_all_features()
        
        if self.features_224 is None or len(self.features_224) == 0:
            raise ValueError(f"特征库 {self.features_224_dir} 为空或不存在")
        
        logger.info(f"加载了 {len(self.features_224)} 个特征")
        
        # 自动检测特征维度（小模型是384，中等模型是768）
        # 通过从features_224获取一个特征来检测维度
        try:
            # 从features_224获取一个特征来检测维度
            if self.features_224 is not None and len(self.features_224) > 0:
                test_feature = self.features_224[0]
                self.feature_dim = test_feature.shape[0] if hasattr(test_feature, 'shape') else len(test_feature)
                logger.info(f"从features_224检测到特征维度: {self.feature_dim}")
            else:
                # 如果features_224不可用，使用默认值（根据模型类型）
                # dinov2_vits14是384，dinov2_vitb14是768，dinov2_vitl14是1024，dinov2_vitg14是1536
                model_name = getattr(self.feature_extractor, 'model_name', dinov2_model_name)
                feature_dims = {
                    'dinov2_vits14': 384,
                    'dinov2_vitb14': 768,
                    'dinov2_vitl14': 1024,
                    'dinov2_vitg14': 1536
                }
                self.feature_dim = feature_dims.get(model_name, 768)  # 默认768维
                logger.info(f"使用默认特征维度: {self.feature_dim} (模型: {model_name})")
        except Exception as e:
            logger.warning(f"无法检测特征维度，使用默认值384: {e}")
            self.feature_dim = 384  # 小模型默认维度
        
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
        
        # 检查目录是否存在
        if not self.face_dir.exists():
            raise ValueError(f"正面图片目录不存在: {self.face_dir}\n"
                           f"请检查路径是否正确，或使用 --face_dir 参数指定正确的路径")
        
        if not self.video_dir.exists():
            raise ValueError(f"视频帧目录不存在: {self.video_dir}\n"
                           f"请检查路径是否正确，或使用 --video_dir 参数指定正确的路径")
        
        # 获取所有正面图片的人名
        face_names = set()
        face_files = list(self.face_dir.glob('*.jpg'))
        
        if len(face_files) == 0:
            logger.warning(f"⚠️  在 {self.face_dir} 中未找到任何 .jpg 文件")
            logger.warning(f"   目录内容: {list(self.face_dir.iterdir())[:10] if self.face_dir.exists() else '目录不存在'}")
            logger.warning(f"   提示: 请确保正面图片文件名格式为: <人名>.jpg")
        else:
            logger.info(f"在 {self.face_dir} 中找到 {len(face_files)} 个 .jpg 文件")
        
        for face_file in face_files:
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
            # 提供详细的诊断信息
            logger.error("=" * 70)
            logger.error("❌ 错误: 没有找到共有的人名！")
            logger.error("=" * 70)
            logger.error(f"\n正面图片目录: {self.face_dir}")
            logger.error(f"找到的正面图片人名 ({len(face_names)} 个): {sorted(face_names)[:10]}...")
            logger.error(f"\nfeatures_224 中的人名 ({len(features_224_names)} 个): {sorted(features_224_names.keys())[:10]}...")
            
            # 检查是否有部分匹配（大小写、空格等）
            face_names_lower = {name.lower().strip() for name in face_names}
            features_224_names_lower = {name.lower().strip() for name in features_224_names.keys()}
            common_names_lower = face_names_lower.intersection(features_224_names_lower)
            
            if len(common_names_lower) > 0:
                logger.warning(f"\n⚠️  发现 {len(common_names_lower)} 个可能匹配的人名（忽略大小写和空格）:")
                logger.warning(f"   {sorted(common_names_lower)[:10]}")
                logger.warning(f"   提示: 可能是文件名大小写或空格问题，请检查文件名")
            
            logger.error(f"\n解决方案:")
            logger.error(f"1. 检查正面图片目录路径是否正确: {self.face_dir}")
            logger.error(f"2. 确保正面图片文件名格式为: <人名>.jpg")
            logger.error(f"3. 确保文件名与 features_224 中的人名完全匹配（包括大小写）")
            logger.error(f"4. 检查视频帧目录: {self.video_dir}")
            logger.error(f"5. 可以使用 --valid_images_file 参数指定筛选后的图片列表")
            logger.error("=" * 70)
            
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
        从文件加载有效数据（包含特征和关键点）并构建训练样本
        
        Returns:
            samples: 样本列表，每个样本包含预提取的特征和关键点
        """
        samples = []
        
        # 加载有效数据文件
        with open(self.valid_images_file, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        
        logger.info(f"从文件加载了 {len(valid_data)} 个人的数据")
        
        # 检查数据格式：新格式（包含特征）还是旧格式（只包含路径）
        is_new_format = False
        if valid_data and len(valid_data) > 0:
            first_person_data = list(valid_data.values())[0]
            if 'face_features' in first_person_data and 'video_data' in first_person_data:
                is_new_format = True
                logger.info("✓ 检测到新格式数据（包含预提取的特征和关键点）")
            else:
                logger.info("⚠️  检测到旧格式数据（只包含路径），将使用旧方式读取")
        
        if is_new_format:
            # 新格式：直接使用预提取的特征和关键点
            for person_name, person_data in valid_data.items():
                # 获取基础目录（用于解析相对路径）
                base_dir = Path(self.valid_images_file).parent
                
                # 获取正面图特征和关键点
                face_features = np.array(person_data['face_features'])
                face_landmarks_2d = np.array(person_data['face_landmarks_2d'])
                face_landmarks_3d = np.array(person_data['face_landmarks_3d'])
                face_box = np.array(person_data['face_box'])
                
                # 检查特征维度
                feature_dim = person_data.get('feature_dim', len(face_features))
                dinov2_model = person_data.get('dinov2_model', 'dinov2_vitb14')
                
                # 获取该人在features_224中的特征（用于目标特征）
                # 尝试找到匹配的特征
                feature_224_index = None
                face_image_path = person_data['face_image_path']
                
                # 解析相对路径为绝对路径（如果需要）
                if not Path(face_image_path).is_absolute():
                    face_image_path_abs = base_dir / face_image_path
                else:
                    face_image_path_abs = Path(face_image_path)
                
                # 在features_224中查找匹配的特征
                features_224_names = {}
                for idx, meta in enumerate(self.metadata_224):
                    meta_person_name = meta.get('person_name')
                    if meta_person_name == person_name:
                        if person_name not in features_224_names:
                            features_224_names[person_name] = []
                        features_224_names[person_name].append({
                            'index': idx,
                            'image_path': meta.get('image_path', ''),
                            'metadata': meta
                        })
                
                if person_name in features_224_names:
                    person_features_224 = features_224_names[person_name]
                    # 尝试路径匹配
                    face_image_name = Path(face_image_path_abs).name
                    face_image_path_normalized = str(face_image_path_abs).replace('\\', '/')
                    for feat_info in person_features_224:
                        feat_path = feat_info['image_path']
                        feat_path_normalized = feat_path.replace('\\', '/')
                        feat_name = Path(feat_path).name
                        if (feat_path_normalized == face_image_path_normalized or 
                            feat_name == face_image_name):
                            feature_224_index = feat_info['index']
                            break
                    
                    # 如果路径匹配失败，使用第一个特征
                    if feature_224_index is None and len(person_features_224) > 0:
                        feature_224_index = person_features_224[0]['index']
                else:
                    logger.warning(f"{person_name} 不在features_224中，将使用预提取的正面图特征")
                
                # 遍历视频数据
                video_data_list = person_data['video_data']
                for video_item in video_data_list:
                    # 解析相对路径
                    video_image_path = video_item['image_path']
                    if not Path(video_image_path).is_absolute():
                        video_image_path_abs = base_dir / video_image_path
                    else:
                        video_image_path_abs = Path(video_image_path)
                    
                    # 获取预提取的特征和关键点
                    video_features = np.array(video_item['features'])
                    video_landmarks_2d = np.array(video_item['landmarks_2d'])
                    video_landmarks_3d = np.array(video_item['landmarks_3d'])
                    video_box = np.array(video_item['box'])
                    spherical_angles = np.array(video_item['spherical_angles'])
                    
                    samples.append({
                        'person_name': person_name,
                        'video_image_path': str(video_image_path_abs),  # 保留路径用于调试
                        'face_image_path': str(face_image_path_abs),  # 保留路径用于调试
                        'feature_224_index': feature_224_index,
                        # 预提取的特征和关键点
                        'video_features': video_features,
                        'face_features': face_features,
                        'spherical_angles': spherical_angles,
                        'video_landmarks_2d': video_landmarks_2d,
                        'video_landmarks_3d': video_landmarks_3d,
                        'face_landmarks_2d': face_landmarks_2d,
                        'face_landmarks_3d': face_landmarks_3d,
                        'video_box': video_box,
                        'face_box': face_box,
                        'use_precomputed': True  # 标记使用预计算数据
                    })
                
                logger.info(f"{person_name}: 添加了 {len(video_data_list)} 个样本（使用预提取特征）")
        else:
            # 旧格式：只包含路径，需要实时提取（保持向后兼容）
            logger.warning("使用旧格式数据，将实时提取特征（建议重新运行 filter_valid_images.py 生成新格式）")
            
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
            for person_name, person_data in valid_data.items():
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
                        'feature_224_index': feature_224_index,
                        'use_precomputed': False  # 标记需要实时计算
                    })
                
                logger.info(f"{person_name}: 添加了 {len(video_images)} 个样本（实时提取特征）")
        
        return samples
    
    def _extract_dinov2_features(self, image_path: str) -> np.ndarray:
        """
        提取DINOv2特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            features: DINOv2特征向量 (384维或768维，取决于模型)
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
                - 'input_features': DINOv2特征 (384维或768维，取决于模型)
                - 'position_encoding': 球面角 (5维)
                - 'target_features': features_224特征 (768维)
                - 'target_residual': 残差特征
                - 'person_name': 人名（字符串）
        """
        sample_info = self.samples[idx]
        
        # 检查是否使用预计算的特征和关键点
        use_precomputed = sample_info.get('use_precomputed', False)
        
        if use_precomputed:
            # 使用预计算的特征和关键点（新格式）
            input_features = sample_info['video_features']
            angles = sample_info['spherical_angles']
            
            # 获取features_224特征（目标）
            if sample_info['feature_224_index'] is not None:
                target_features = self.features_224[sample_info['feature_224_index']]
            else:
                # 使用预提取的正面图特征
                target_features = sample_info['face_features']
        else:
            # 实时提取特征和关键点（旧格式或未使用筛选文件）
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
        features_224_dir=r'root/face000/features_224',
        video_dir=r'root/face000/train\datas\video',
        face_dir=r'root/face000/train\datas\face',
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

