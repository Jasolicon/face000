"""
多角度人脸识别数据集
支持正脸图片 + 多角度视频
"""
import os
import sys
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_detector import FaceDetector

# 导入特征管理器（用于保存DINO特征）
try:
    from feature_manager import FeatureManager
except ImportError:
    FeatureManager = None

# 导入工具函数
try:
    # 尝试从train目录导入
    from utils import get_video_frames_dir
except ImportError:
    # 如果失败，尝试从train.utils导入
    try:
        from train.utils import get_video_frames_dir
    except ImportError:
        # 最后尝试直接加载文件
        import importlib.util
        utils_path = Path(__file__).parent / 'utils.py'
        if utils_path.exists():
            spec = importlib.util.spec_from_file_location("train_utils", utils_path)
            utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils)
            get_video_frames_dir = utils.get_video_frames_dir
        else:
            # 如果没有utils，创建一个简单的实现
            def get_video_frames_dir(video_path: str):
                video_path = Path(video_path)
                frames_dir = video_path.parent / video_path.stem
                if frames_dir.exists() and frames_dir.is_dir():
                    return frames_dir
                return None


class MultiAngleFaceDataset(Dataset):
    """多角度人脸识别数据集"""
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        num_frames_per_video: int = 10,
        augment: bool = True,
        device: str = 'cpu',
        save_features: bool = False,
        feature_storage_dir: Optional[str] = None,
        negative_ratio: float = 0.5
    ):
        """
        初始化数据集
        
        支持两种数据格式：
        1. 文件名对应模式：
           datas/
           ├── 张三.jpg          # 正脸图片
           ├── 张三.mp4          # 视频
           ├── 李四.jpg
           ├── 李四.mp4
           └── ...
        
        2. 目录模式：
           datas/
           ├── person_001/
           │   ├── face.jpg 或 front.jpg
           │   └── video.mp4
           └── ...
        
        Args:
            data_dir: 数据目录
            image_size: 图像尺寸
            num_frames_per_video: 每个视频提取的帧数
            augment: 是否使用数据增强
            device: 计算设备
            save_features: 是否保存提取的特征（兼容DINOFeatureExtractor）
            feature_storage_dir: 特征存储目录
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_frames_per_video = num_frames_per_video
        self.augment = augment
        self.device = device
        self.negative_ratio = negative_ratio  # 负样本比例（0-1之间）
        
        # 初始化人脸检测器
        self.face_detector = FaceDetector(device=device)
        
        # 特征保存相关（兼容DINOFeatureExtractor）
        self.save_features = save_features
        self.feature_storage_dir = Path(feature_storage_dir) if feature_storage_dir else None
        if self.save_features and FeatureManager is not None and self.feature_storage_dir:
            self.feature_manager = FeatureManager(storage_dir=str(self.feature_storage_dir))
            print(f"特征保存已启用，存储目录: {self.feature_storage_dir}")
        else:
            self.feature_manager = None
        
        # 加载数据
        self.samples = self._load_samples()
        
        # 创建人员ID到索引的映射
        self.person_ids = sorted(set([s['person_id'] for s in self.samples]))
        self.person_to_idx = {pid: idx for idx, pid in enumerate(self.person_ids)}
        self.num_classes = len(self.person_ids)
        
        # 配对信息（由train.py设置）
        self.train_pairs = []
        self.val_pairs = []
        
        # 图像增强（仅在训练时使用）
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.augment_transform = None
        
        print(f"数据集加载完成:")
        print(f"  总样本数: {len(self.samples)}")
        print(f"  人员数: {self.num_classes}")
        print(f"  图像尺寸: {image_size}x{image_size}")
        
        # 验证样本路径
        if len(self.samples) > 0:
            print(f"\n验证样本路径...")
            valid_samples = 0
            for sample in self.samples:
                front_path = Path(sample['front_image'])
                if front_path.exists():
                    valid_samples += 1
                else:
                    print(f"  ⚠️ 路径不存在: {sample['front_image']}")
            print(f"  有效样本数: {valid_samples}/{len(self.samples)}")
    
    def _load_samples(self) -> List[dict]:
        """加载所有样本
        
        支持三种数据格式：
        
        格式1: 文件名对应模式
        datas/
        ├── 张三.jpg          # 正脸图片
        ├── 张三.mp4          # 视频
        └── ...
        
        格式2: 目录模式
        datas/
        ├── person_001/
        │   ├── face.jpg
        │   └── video.mp4
        └── ...
        
        格式3: face/ 和 video/ 子目录模式（推荐）
        支持两种子格式：
        
        格式3a: video目录下是视频文件
        datas/
        ├── face/
        │   ├── 张三.jpg
        │   └── 李四.jpg
        └── video/
            ├── 张三.mp4
            └── 李四.mp4
        
        格式3b: video目录下是人名文件夹（推荐）
        datas/
        ├── face/
        │   ├── 张三.jpg
        │   └── 李四.jpg
        └── video/
            ├── 张三/
            │   ├── frame_001.jpg
            │   └── frame_002.jpg
            └── 李四/
                ├── frame_001.jpg
                └── frame_002.jpg
        """
        samples = []
        
        # 支持的图片格式
        image_exts = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        # 支持的视频格式
        video_exts = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        
        # 检查是否是格式3：face/ 和 video/ 子目录模式
        face_dir = self.data_dir / 'face'
        video_dir = self.data_dir / 'video'
        
        if face_dir.exists() and face_dir.is_dir():
            # 格式3: face/ 和 video/ 子目录
            print("检测到 face/ 和 video/ 子目录格式")
            
            # 收集face目录中的图片
            face_files = {}
            for ext in image_exts:
                for face_file in face_dir.glob(f'*{ext}'):
                    stem = face_file.stem
                    face_files[stem] = face_file
            
            # 收集video目录中的内容（支持两种格式）
            video_files = {}  # 存储视频文件路径或文件夹路径
            video_folders = {}  # 存储人名文件夹路径
            
            if video_dir.exists():
                # 方式1: video目录下是视频文件（.mp4等）
                for ext in video_exts:
                    for video_file in video_dir.glob(f'*{ext}'):
                        stem = video_file.stem
                        video_files[stem] = str(video_file)
                
                # 方式2: video目录下是人名文件夹，文件夹里是jpg图片
                for person_folder in video_dir.iterdir():
                    if person_folder.is_dir():
                        person_name = person_folder.name
                        # 检查文件夹中是否有图片文件
                        has_images = False
                        for ext in image_exts:
                            if list(person_folder.glob(f'*{ext}')):
                                has_images = True
                                break
                        if has_images:
                            video_folders[person_name] = str(person_folder)
            
            # 合并video_files和video_folders（优先使用文件夹）
            all_video_data = {**video_files, **video_folders}
            
            # 匹配同名的图片和视频/文件夹
            matched_names = set(face_files.keys()) & set(all_video_data.keys())
            for name in sorted(matched_names):
                video_path = all_video_data[name]
                # 判断是文件夹还是视频文件
                is_folder = Path(video_path).is_dir()
                samples.append({
                    'person_id': name,  # 使用文件名作为person_id（人名）
                    'front_image': str(face_files[name]),
                    'video_path': video_path,
                    'is_front_only': False,
                    'is_video_folder': is_folder  # 标记是否为文件夹
                })
            
            # 处理只有图片没有视频的情况
            image_only = set(face_files.keys()) - set(all_video_data.keys())
            for name in sorted(image_only):
                samples.append({
                    'person_id': name,
                    'front_image': str(face_files[name]),
                    'video_path': None,
                    'is_front_only': True,
                    'is_video_folder': False
                })
        
        else:
            # 格式1和格式2：文件模式或目录模式
            # 收集所有文件
            image_files = {}
            video_files = {}
            
            # 遍历数据目录
            for item in sorted(self.data_dir.iterdir()):
                if item.is_file():
                    # 文件模式：文件名对应
                    stem = item.stem  # 文件名（不含扩展名）
                    suffix = item.suffix.lower()
                    
                    if suffix in [ext.lower() for ext in image_exts]:
                        # 正脸图片
                        if stem not in image_files:
                            image_files[stem] = item
                    elif suffix in [ext.lower() for ext in video_exts]:
                        # 视频文件
                        if stem not in video_files:
                            video_files[stem] = item
                
                elif item.is_dir() and item.name not in ['face', 'video']:
                    # 目录模式：person_xxx/ 或 人名/（排除face和video目录）
                    person_id = item.name
                    
                    # 查找正脸图片（face.jpg 或 front.jpg）
                    front_image_path = None
                    for face_name in ['face', 'front']:
                        for ext in image_exts:
                            face_path = item / f'{face_name}{ext}'
                            if face_path.exists():
                                front_image_path = face_path
                                break
                        if front_image_path:
                            break
                    
                    # 查找视频文件（video.mp4）
                    video_path = None
                    for video_name in ['video']:
                        for ext in video_exts:
                            video_file = item / f'{video_name}{ext}'
                            if video_file.exists():
                                video_path = video_file
                                break
                        if video_path:
                            break
                    
                    if front_image_path:
                        samples.append({
                            'person_id': person_id,
                            'front_image': str(front_image_path),
                            'video_path': str(video_path) if video_path else None,
                            'is_front_only': video_path is None
                        })
            
            # 处理文件模式（文件名对应）
            # 匹配同名的图片和视频
            matched_names = set(image_files.keys()) & set(video_files.keys())
            
            for name in sorted(matched_names):
                samples.append({
                    'person_id': name,  # 使用文件名作为person_id（人名）
                    'front_image': str(image_files[name]),
                    'video_path': str(video_files[name]),
                    'is_front_only': False
                })
            
            # 处理只有图片没有视频的情况
            image_only = set(image_files.keys()) - set(video_files.keys())
            for name in sorted(image_only):
                samples.append({
                    'person_id': name,
                    'front_image': str(image_files[name]),
                    'video_path': None,
                    'is_front_only': True
                })
        
        # 去重
        seen = set()
        unique_samples = []
        for sample in samples:
            key = (sample['person_id'], sample['front_image'])
            if key not in seen:
                seen.add(key)
                unique_samples.append(sample)
        
        return unique_samples
    
    def _get_video_frames(self, video_path: str) -> List:
        """获取视频对应的帧图片路径列表或numpy数组列表
        
        支持两种格式：
        1. 视频文件路径（.mp4等）- 从视频中提取或使用已提取的帧
        2. 文件夹路径（人名文件夹，包含jpg图片）- 直接读取文件夹中的图片
        
        Args:
            video_path: 视频文件路径或文件夹路径
            
        Returns:
            帧图片路径列表（List[str]）或numpy数组列表（List[np.ndarray]）
        """
        video_path_obj = Path(video_path)
        
        # 如果是文件夹（人名文件夹格式），直接读取文件夹中的图片
        if video_path_obj.is_dir():
            frame_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                frame_files.extend(sorted(video_path_obj.glob(f'*{ext}')))
            if len(frame_files) > 0:
                return [str(f) for f in frame_files]
            return []
        
        # 如果是视频文件，使用原有逻辑
        # 首先检查是否有已提取的帧文件夹
        frames_dir = get_video_frames_dir(video_path)
        
        if frames_dir and frames_dir.exists():
            # 使用已提取的帧图片（返回字符串路径列表）
            frame_files = sorted(frames_dir.glob('*.jpg')) + sorted(frames_dir.glob('*.png'))
            if len(frame_files) > 0:
                return [str(f) for f in frame_files]
        
        # 如果没有已提取的帧，从视频中提取（兼容旧方法，返回numpy数组列表）
        return self._extract_video_frames_fallback(video_path)
    
    def _extract_video_frames_fallback(self, video_path: str) -> List[np.ndarray]:
        """从视频中提取帧（备用方法，如果帧文件夹不存在）"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []
        
        # 均匀采样帧
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def _detect_and_crop_face(self, image: np.ndarray) -> Optional[Image.Image]:
        """检测并裁剪人脸
        
        使用 MTCNN (facenet_pytorch) 进行人脸检测
        
        Returns:
            如果检测到人脸，返回裁剪后的人脸图像
            如果未检测到人脸，返回 None（调用者会使用原图）
        """
        try:
            pil_image = Image.fromarray(image)
            
            # 检测人脸（使用MTCNN）
            faces, boxes, probs = self.face_detector.detect_faces(pil_image)
            
            if len(faces) == 0:
                return None
            
            # 使用第一个人脸
            return faces[0]
        except Exception as e:
            print(f"❌ 人脸检测失败: {e}")
            return None
    
    def _load_and_preprocess_image(self, image_path: str, apply_augment: bool = False) -> Optional[torch.Tensor]:
        """加载并预处理图像
        
        如果检测不到人脸，直接使用原图（DINO可以直接处理整张图片）
        
        Args:
            image_path: 图像路径
            apply_augment: 是否应用图像增强
        """
        try:
            # 确保 image_path 是字符串或 Path 对象
            if not isinstance(image_path, (str, Path)):
                image_path = str(image_path)
            image_path = str(Path(image_path).resolve())
            
            # 检查文件是否存在
            if not Path(image_path).exists():
                print(f"❌ 图像文件不存在: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            
            # 尝试检测并裁剪人脸
            face = self._detect_and_crop_face(np.array(image))
            if face is None:
                # 如果检测不到人脸，直接使用原图（DINO可以直接处理整张图片）
                print(f"⚠️ 未检测到人脸，使用原图: {image_path}")
                face = image
            
            # Resize
            face = face.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            # 应用图像增强（如果启用，在转换为tensor之前）
            if apply_augment and self.augment_transform is not None:
                face = self.augment_transform(face)
            
            # 转换为tensor（确保是float32类型）
            face_array = np.array(face).astype(np.float32) / 255.0
            
            # 归一化 (ImageNet标准，兼容DINO)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            face_array = (face_array - mean) / std
            
            # 转换为tensor [C, H, W]，确保是float32类型
            face_tensor = torch.from_numpy(face_array).permute(2, 0, 1).float()
            
            return face_tensor
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_video_frame(self, video_path: str, person_id: str = None) -> Optional[torch.Tensor]:
        """从视频中加载一帧（优先使用已提取的帧图片）
        
        Args:
            video_path: 视频路径
            person_id: 人员ID，用于从划分的帧列表中选择
        """
        # 如果设置了帧划分，且该类别有划分信息，使用划分的帧
        if person_id and hasattr(self, 'person_frame_split') and person_id in self.person_frame_split:
            split_info = self.person_frame_split[person_id]
            if self.is_train:
                frame_paths = split_info.get('train_frames', [])
            else:
                frame_paths = split_info.get('val_frames', [])
            
            if len(frame_paths) == 0:
                # 如果划分的帧列表为空，回退到原始方法
                frame_paths = self._get_video_frames(video_path)
        else:
            # 没有划分信息，使用原始方法
            frame_paths = self._get_video_frames(video_path)
        
        if len(frame_paths) == 0:
            return None
        
        # 随机选择一帧
        if isinstance(frame_paths[0], str):
            # 从文件路径加载
            frame_path = random.choice(frame_paths)
            try:
                # 确保路径是字符串
                frame_path = str(Path(frame_path).resolve())
                if not Path(frame_path).exists():
                    return None
                frame_image = Image.open(frame_path).convert('RGB')
                frame_array = np.array(frame_image)
            except Exception as e:
                print(f"加载帧图片失败 {frame_path}: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # 从numpy数组加载（备用方法）
            frame = random.choice(frame_paths)
            frame_array = frame
        
        # 尝试检测并裁剪人脸
        face = self._detect_and_crop_face(frame_array)
        if face is None:
            # 如果检测不到人脸，直接使用原图（DINO可以直接处理整张图片）
            face = Image.fromarray(frame_array)
        
        # Resize
        face = face.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # 转换为tensor（确保是float32类型）
        face_array = np.array(face).astype(np.float32) / 255.0
        
        # 归一化 (ImageNet标准，兼容DINO)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_array = (face_array - mean) / std
        
        # 转换为tensor [C, H, W]，确保是float32类型
        face_tensor = torch.from_numpy(face_array).permute(2, 0, 1).float()
        
        return face_tensor
    
    def __len__(self) -> int:
        # 如果使用配对模式，返回配对数量；否则返回样本数
        if hasattr(self, 'train_pairs') and len(self.train_pairs) > 0:
            return len(self.train_pairs)  # 训练时使用train_pairs
        elif hasattr(self, 'val_pairs') and len(self.val_pairs) > 0:
            return len(self.val_pairs)  # 验证时使用val_pairs
        return len(self.samples)
    
    def get_pair(self, pair: dict) -> dict:
        """从配对中加载图像对
        
        Args:
            pair: 配对字典，包含 'front_image', 'angle_image', 'person_id', 'is_positive'
        
        Returns:
            包含 'front_image', 'angle_image', 'person_id', 'person_idx', 'label', 'pair_label' 的字典
        """
        front_image_path = pair['front_image']
        angle_image_path = pair['angle_image']
        person_id = pair['person_id']
        is_positive = pair['is_positive']
        
        person_idx = self.person_to_idx.get(person_id, 0)
        
        # 加载正脸图像
        front_tensor = self._load_and_preprocess_image(front_image_path, apply_augment=self.augment)
        if front_tensor is None:
            # 如果加载失败，返回默认tensor
            default_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return {
                'front_image': default_tensor,
                'angle_image': default_tensor,
                'person_id': person_id,
                'person_idx': person_idx,
                'label': person_idx,
                'pair_label': 1 if is_positive else 0
            }
        
        # 加载角度图像
        if angle_image_path == front_image_path:
            # 如果是自对比，使用正脸图像
            angle_tensor = front_tensor.clone()
        else:
            angle_tensor = self._load_and_preprocess_image(angle_image_path, apply_augment=self.augment)
            if angle_tensor is None:
                # 如果加载失败，使用正脸图像
                angle_tensor = front_tensor.clone()
        
        return {
            'front_image': front_tensor,
            'angle_image': angle_tensor,
            'person_id': person_id,
            'person_idx': person_idx,
            'label': person_idx,  # 用于ArcFace的类别标签
            'pair_label': 1 if is_positive else 0  # 正负样本对标签（1=正样本，0=负样本）
        }
    
    def __getitem__(self, idx: int) -> dict:
        """获取一个训练样本
        
        支持正样本对（同一人的正脸+多角度）和负样本对（不同人的组合）
        确保一个epoch遍历完所有训练样本，但角度图随机匹配
        """
        # 使用索引确保遍历完所有样本（DataLoader的shuffle会打乱顺序）
        sample_idx = idx % len(self.samples)
        max_attempts = len(self.samples)  # 最多尝试所有样本
        
        for attempt in range(max_attempts):
            # 使用索引选择样本（确保遍历完所有样本）
            current_idx = (sample_idx + attempt) % len(self.samples)
            sample = self.samples[current_idx]
            person_id = sample['person_id']
            person_idx = self.person_to_idx[person_id]
            
            # 加载正脸图像
            front_tensor = self._load_and_preprocess_image(sample['front_image'])
            if front_tensor is None:
                # 如果加载失败，尝试下一个样本
                print(f"⚠️ 样本加载失败: {sample['person_id']} - {sample['front_image']}")
                continue
            
            # 决定是正样本对还是负样本对
            is_positive = random.random() > self.negative_ratio
            
            if is_positive:
                # 正样本对：同一人的正脸+多角度
                if sample['is_front_only']:
                    # 如果没有视频，使用正脸图像（自对比）
                    angle_tensor = front_tensor.clone()
                else:
                    angle_tensor = self._load_video_frame(sample['video_path'], person_id=person_id)
                    if angle_tensor is None:
                        # 如果视频加载失败，使用正脸图像
                        angle_tensor = front_tensor.clone()
                label = 1  # 正样本对标签
            else:
                # 负样本对：不同人的正脸+多角度
                # 随机选择另一个不同的人
                other_idx = random.randint(0, len(self.samples) - 1)
                while other_idx == current_idx or self.samples[other_idx]['person_id'] == person_id:
                    other_idx = random.randint(0, len(self.samples) - 1)
                
                other_sample = self.samples[other_idx]
                # 加载另一个人的多角度图像
                if other_sample['is_front_only']:
                    angle_tensor = self._load_and_preprocess_image(other_sample['front_image'])
                    if angle_tensor is None:
                        # 如果加载失败，尝试下一个样本
                        continue
                else:
                    angle_tensor = self._load_video_frame(other_sample['video_path'], person_id=other_sample['person_id'])
                    if angle_tensor is None:
                        # 如果视频加载失败，使用另一个人的正脸图像
                        angle_tensor = self._load_and_preprocess_image(other_sample['front_image'])
                        if angle_tensor is None:
                            # 如果还是失败，尝试下一个样本
                            continue
                label = 0  # 负样本对标签
            
            # 成功加载，返回数据
            return {
                'front_image': front_tensor,
                'angle_image': angle_tensor,
                'person_id': person_id,
                'person_idx': person_idx,
                'label': person_idx,  # 用于ArcFace的类别标签
                'pair_label': label  # 正负样本对标签（1=正样本，0=负样本）
            }
        
        # 如果所有样本都加载失败，返回一个默认的tensor（避免崩溃）
        print(f"\n❌ 警告: 所有样本加载失败，返回默认tensor")
        print(f"   总样本数: {len(self.samples)}")
        print(f"   请检查:")
        print(f"   1. 图像文件是否存在")
        print(f"   2. 图像中是否包含可检测的人脸")
        print(f"   3. 人脸检测器是否正常工作")
        print(f"   样本路径示例:")
        for i, sample in enumerate(self.samples[:3]):  # 只显示前3个
            print(f"     - {sample['person_id']}: {sample['front_image']}")
        default_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        return {
            'front_image': default_tensor,
            'angle_image': default_tensor,
            'person_id': 0,
            'person_idx': 0,
            'label': 0,
            'pair_label': 0
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    shuffle: bool = True,
    device: str = 'cpu'
) -> DataLoader:
    """创建数据加载器"""
    dataset = MultiAngleFaceDataset(
        data_dir=data_dir,
        image_size=image_size,
        device=device
    )
    print(len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
        drop_last=True
    )
    
    return dataloader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataloader = create_dataloader(
        data_dir=r'C:\Codes\face000\train\datas',
        batch_size=1,
        num_workers=0,
        image_size=224,
        shuffle=True,
        device='cuda'
    )
    fig = plt.figure(figsize=(10, 10))
    print("=" * 60)
    print("数据集诊断信息:")
    print(f"  DataLoader 长度: {len(dataloader)}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Drop last: {dataloader.drop_last}")
    print(f"  数据集长度: {len(dataloader.dataset)}")
    if hasattr(dataloader.dataset, 'samples'):
        print(f"  原始样本数: {len(dataloader.dataset.samples)}")
        print(f"  人员数: {dataloader.dataset.num_classes}")
        print(f"  人员ID列表: {dataloader.dataset.person_ids}")
        print(f"\n  样本详情:")
        for i, sample in enumerate(dataloader.dataset.samples[:10]):  # 只显示前10个
            print(f"    样本 {i}: person_id={sample['person_id']}, front_image={sample['front_image']}")
            if 'video_path' in sample:
                print(f"              video_path={sample.get('video_path', 'None')}")
    if hasattr(dataloader.dataset, 'train_pairs'):
        print(f"  训练配对数: {len(dataloader.dataset.train_pairs)}")
    if hasattr(dataloader.dataset, 'val_pairs'):
        print(f"  验证配对数: {len(dataloader.dataset.val_pairs)}")
    print("=" * 60)
    print()
    
    for idx, data in enumerate(dataloader):
        print(idx)
        print(f"front_image: {data['front_image'].shape}")
        print(f"angle_image: {data['angle_image'].shape}")
        print(f"person_id: {data['person_id']}")
        print(f"person_idx: {data['person_idx']}")
        print(f"label: {data['label']}")
        print(f"pair_label: {data['pair_label']}")
        
        # 处理 front_image：移除 batch 维度（如果有），转换为 numpy，然后 transpose
        img = data['front_image']
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # 如果有多余的 batch 维度，移除它
        if img.ndim == 4:  # [B, C, H, W]
            img = img[0]  # 取第一个样本 [C, H, W]
        elif img.ndim == 3:  # [C, H, W]
            pass  # 已经是正确的形状
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        # 从 [C, H, W] 转换为 [H, W, C]
        img = img.transpose(1, 2, 0)
        # 反归一化（ImageNet 标准归一化）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        img = (img * 255.0).astype(np.uint8)
        
        # 处理 angle_image：同样的操作
        img_angle = data['angle_image']
        if isinstance(img_angle, torch.Tensor):
            img_angle = img_angle.cpu().numpy()
        if img_angle.ndim == 4:  # [B, C, H, W]
            img_angle = img_angle[0]  # 取第一个样本 [C, H, W]
        elif img_angle.ndim == 3:  # [C, H, W]
            pass  # 已经是正确的形状
        else:
            raise ValueError(f"Unexpected image shape: {img_angle.shape}")
        img_angle = img_angle.transpose(1, 2, 0)
        # 反归一化
        img_angle = img_angle * std + mean
        img_angle = np.clip(img_angle, 0, 1)
        img_angle = (img_angle * 255.0).astype(np.uint8)
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img)
        ax.set_title(f"front_image_{idx}")
        ax.axis('off')
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(img_angle)
        ax.set_title(f"angle_image_{idx}")
        ax.axis('off')
        
        # 保存图像而不是显示（避免非交互式环境警告）
        output_path = Path(__file__).parent.parent / 'checkpoints' / f'sample_{idx}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  已保存样本图像: {output_path}")
        plt.close(fig)  # 关闭图形以释放内存
        
        if idx == 5:
            break
        plt.close()