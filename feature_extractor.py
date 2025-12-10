"""
图像特征提取模块 - 使用 DINO
"""
# 在导入任何可能使用 HuggingFace 的库之前设置镜像
import os
import sys
from pathlib import Path

# 设置镜像环境变量（必须在导入 timm 之前）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # 禁用 hf_transfer
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时
    os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '5'    # 重试5次

# 尝试导入 setup_mirrors（如果存在）
try:
    # 添加项目根目录到路径
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from setup_mirrors import setup_all_mirrors
    setup_all_mirrors()
except ImportError:
    pass  # 如果 setup_mirrors 不存在，使用上面的默认设置

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import timm
import numpy as np


class DINOFeatureExtractor:
    """使用DINO模型提取图像特征"""
    
    def __init__(self, model_name='vit_base_patch16_224', device=None):
        """
        初始化DINO特征提取器
        
        Args:
            model_name: DINO模型名称
            device: 计算设备 ('cuda' 或 'cpu')，如果为None则自动选择
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载DINO模型
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # 移除分类头，只使用特征提取
        )
        self.model.eval()
        self.model.to(self.device)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """
        提取图像特征
        
        Args:
            image: PIL Image对象或图像路径
            
        Returns:
            features: 特征向量 (numpy array)
        """
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image
        
        # 预处理
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(img_tensor)
            # L2归一化
            features = nn.functional.normalize(features, p=2, dim=1)
        
        # 转换为numpy数组
        features = features.cpu().numpy().flatten()
        
        return features
    
    def extract_batch_features(self, images):
        """
        批量提取特征
        
        Args:
            images: PIL Image对象列表或图像路径列表
            
        Returns:
            features: 特征矩阵 (numpy array, shape: [batch_size, feature_dim])
        """
        batch_tensors = []
        
        for image in images:
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            else:
                img = image
            
            img_tensor = self.transform(img)
            batch_tensors.append(img_tensor)
        
        # 堆叠为批次
        batch = torch.stack(batch_tensors).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(batch)
            # L2归一化
            features = nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()


class DINOv2FeatureExtractor:
    """使用 DINOv2 提取图像特征"""
    
    def __init__(self, model_name='dinov2_vits14', device=None, resize_to_96=True, model_dir=None):
        """
        初始化 DINOv2 特征提取器
        
        Args:
            model_name: DINOv2 模型名称 ('dinov2_vits14' 小模型, 'dinov2_vitb14' 中等模型)
            device: 计算设备 ('cuda' 或 'cpu')，如果为None则自动选择
            resize_to_96: 是否在提取特征前先缩放到 96*96
            model_dir: 模型文件目录（如果提供，将从该目录加载模型，避免重复下载）
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.resize_to_96 = resize_to_96
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else None
        
        # 设置模型下载镜像（在加载模型前）
        try:
            from model_utils import setup_model_mirrors
            setup_model_mirrors()
        except ImportError:
            # 如果 model_utils 不可用，直接设置环境变量
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 默认使用 DINOv2 ViT-S/14（小模型），如果失败则使用 vit_base_patch16_224 作为备用
        self.use_dinov2 = False
        
        # 首先尝试从指定目录加载模型（如果提供）
        if self.model_dir is not None:
            model_path = self.model_dir / f"{model_name}.pth"
            if model_path.exists():
                try:
                    print(f"  从本地路径加载 DINOv2 {model_name}: {model_path}")
                    # 先创建模型结构
                    self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)
                    # 加载权重
                    state_dict = torch.load(model_path, map_location='cpu')
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.model.to(self.device)
                    self.use_dinov2 = True
                    print(f"  ✓ 成功从本地路径加载 DINOv2 {model_name} 模型")
                    if model_name == 'dinov2_vits14':
                        print(f"  提示: 使用小模型，特征维度为384（而非768）")
                except Exception as e:
                    print(f"  ⚠️  从本地路径加载失败: {e}")
                    print(f"  将尝试从 torch.hub 加载...")
                    self.use_dinov2 = False
        
        # 如果本地加载失败或未提供路径，尝试使用 torch.hub 加载
        if not self.use_dinov2:
            try:
                print(f"  尝试从 torch.hub 加载 DINOv2 {model_name}...")
                # 如果指定了model_dir，设置torch.hub的缓存目录
                if self.model_dir is not None:
                    # 设置torch.hub的下载目录
                    original_hub_dir = torch.hub.get_dir()
                    torch.hub.set_dir(str(self.model_dir))
                    try:
                        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
                    finally:
                        # 恢复原始目录
                        torch.hub.set_dir(original_hub_dir)
                else:
                    # 使用默认缓存
                    self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
                
                self.model.eval()
                self.model.to(self.device)
                self.use_dinov2 = True
                print(f"  ✓ 成功加载 DINOv2 {model_name} 模型")
                if self.model_dir:
                    print(f"  模型保存位置: {self.model_dir}")
                else:
                    print(f"  模型缓存位置: ~/.cache/torch/hub/checkpoints/")
                
                # 如果指定了model_dir，保存模型到该目录
                if self.model_dir is not None:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = self.model_dir / f"{model_name}.pth"
                    if not model_path.exists():
                        print(f"  保存模型到: {model_path}")
                        torch.save(self.model.state_dict(), model_path)
                        print(f"  ✓ 模型已保存到: {model_path}")
                
                # 注意：小模型(dinov2_vits14)的特征维度是384，中等模型(dinov2_vitb14)是768
                if model_name == 'dinov2_vits14':
                    print(f"  提示: 使用小模型，特征维度为384（而非768）")
            except Exception as e:
                print(f"  ✗ 无法加载 DINOv2 {model_name}: {str(e)}")
                print(f"  提示: 运行 'python download_dinov2.py --model {model_name} --save_dir <路径>' 可以提前下载模型")
                print(f"  将使用备用模型: vit_base_patch16_224")
                self.use_dinov2 = False
        
        # 如果 DINOv2 加载失败，使用备用模型
        if not self.use_dinov2:
            model_name_to_use = 'vit_base_patch16_224'
            print(f"  加载备用模型: {model_name_to_use}...")
            print(f"  提示: 如果下载失败，请检查网络连接或使用镜像")
            print(f"  当前 HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
            
            try:
                # 确保使用镜像（在调用 timm.create_model 前）
                if 'HF_ENDPOINT' not in os.environ:
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                
                # timm.create_model 会自动使用缓存，如果模型已下载过，不会重新下载
                # timm 会通过 huggingface_hub 下载，使用 HF_ENDPOINT 环境变量
                self.model = timm.create_model(
                    model_name_to_use,
                    pretrained=True,
                    num_classes=0,  # 移除分类头
                )
                self.model.eval()
                self.model.to(self.device)
                print(f"  ✓ 成功加载备用模型: {model_name_to_use}")
                print(f"  模型缓存位置: timm 会自动缓存到系统缓存目录")
                print(f"  (Windows: %USERPROFILE%\\.cache\\torch\\hub\\ 或 %USERPROFILE%\\.cache\\huggingface\\)")
            except Exception as e:
                error_msg = f"无法加载模型 {model_name_to_use}: {str(e)}"
                error_msg += "\n提示: timm 会自动缓存下载的模型，首次下载后后续使用会从缓存加载"
                raise RuntimeError(error_msg)
        
        # 图像预处理
        # DINOv2 ViT-B/14 的标准输入是 518x518，但也可以使用 224x224
        # 如果 resize_to_96=True，先缩放到 96*96，然后再缩放到目标尺寸
        if self.use_dinov2:
            # DINOv2 模型，使用 518x518 或 224x224
            target_size = 224  # 可以使用 224 或 518
            if self.resize_to_96:
                self.transform = transforms.Compose([
                    transforms.Resize((96, 96)),  # 先缩放到 96*96
                    transforms.Resize((target_size, target_size)),  # 再缩放到目标尺寸
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((target_size, target_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            # 备用模型 vit_base_patch16_224，使用 224x224
            if self.resize_to_96:
                self.transform = transforms.Compose([
                    transforms.Resize((96, 96)),  # 先缩放到 96*96
                    transforms.Resize((224, 224)),  # 再缩放到 224*224
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
    
    def extract_features(self, image):
        """
        提取图像特征
        
        Args:
            image: PIL Image对象或图像路径
            
        Returns:
            features: 归一化的特征向量 (numpy array)
        """
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB') if hasattr(image, 'convert') else image
        
        # 预处理（如果 resize_to_96=True，会先缩放到 96*96）
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            if self.use_dinov2:
                # DINOv2 模型，需要特殊处理输出
                output = self.model(img_tensor)
                # DINOv2 返回的是字典，包含 'x_norm_clstoken' 或直接是特征向量
                if isinstance(output, dict):
                    if 'x_norm_clstoken' in output:
                        features = output['x_norm_clstoken']
                    elif 'x_prenorm' in output:
                        # 如果是 prenorm，取 CLS token (第一个 token)
                        features = output['x_prenorm'][:, 0]
                    else:
                        # 取第一个值
                        features = list(output.values())[0]
                        if features.dim() > 2:
                            # 如果是序列，取 CLS token
                            features = features[:, 0]
                else:
                    # 直接返回特征向量
                    features = output
                    if features.dim() > 2:
                        # 如果是序列，取 CLS token
                        features = features[:, 0]
            else:
                # 标准模型（vit_base_patch16_224）
                features = self.model(img_tensor)
            
            # L2归一化
            features = nn.functional.normalize(features, p=2, dim=1)
        
        # 转换为numpy数组
        features = features.cpu().numpy().flatten()
        
        return features
    
    def extract_batch_features(self, images):
        """
        批量提取特征
        
        Args:
            images: PIL Image对象列表或图像路径列表
            
        Returns:
            features: 特征矩阵 (numpy array, shape: [batch_size, feature_dim])
        """
        batch_tensors = []
        
        for image in images:
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            else:
                img = image.convert('RGB') if hasattr(image, 'convert') else image
            
            img_tensor = self.transform(img)
            batch_tensors.append(img_tensor)
        
        # 堆叠为批次
        batch = torch.stack(batch_tensors).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(batch)
            # L2归一化
            features = nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()


class FaceNetFeatureExtractor:
    """使用FaceNet提取人脸特征（作为备选方案）"""
    
    def __init__(self, device=None):
        """
        初始化FaceNet特征提取器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')，如果为None则自动选择
        """
        from facenet_pytorch import InceptionResnetV1
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载预训练的FaceNet模型
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.model.to(self.device)
    
    def extract_features(self, face_image):
        """
        提取人脸特征
        
        Args:
            face_image: PIL Image对象或torch.Tensor (已经对齐的人脸图像)
            
        Returns:
            features: 特征向量 (numpy array)
        """
        # 如果输入是PIL Image，需要转换为tensor
        if isinstance(face_image, Image.Image):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            face_tensor = transform(face_image).unsqueeze(0).to(self.device)
        else:
            face_tensor = face_image.to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(face_tensor)
            # L2归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy().flatten()


class ArcFaceFeatureExtractor:
    """使用ArcFace提取人脸特征"""
    
    def __init__(self, model_name='r50', device=None):
        """
        初始化ArcFace特征提取器
        
        Args:
            model_name: ArcFace模型名称 ('r50', 'r100', 'r34'等)
            device: 计算设备 ('cuda' 或 'cpu')，如果为None则自动选择
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_name = model_name
        
        # 设置模型下载镜像（在加载模型前）
        try:
            from model_utils import setup_model_mirrors
            setup_model_mirrors()
        except ImportError:
            # 如果 model_utils 不可用，直接设置环境变量
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 尝试使用insightface（推荐）
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            # 尝试不同的模型名称（按优先级）
            model_names_to_try = [
                'buffalo_l',      # 最新版本推荐模型
                'buffalo_s',      # 小版本
                'buffalo_m',      # 中等版本
                'antelopev2',     # 另一个可用模型
                model_name,       # 用户指定的模型
            ]
            
            self.app = None
            self.use_insightface = False
            
            for try_model_name in model_names_to_try:
                try:
                    print(f"  尝试加载模型: {try_model_name}...")
                    # 检测可用的providers
                    import onnxruntime as ort
                    available_providers = ort.get_available_providers()
                    
                    # 根据可用providers和设备选择
                    if self.device.type == 'cuda' and 'CUDAExecutionProvider' in available_providers:
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    else:
                        providers = ['CPUExecutionProvider']
                    
                    # 初始化ArcFace模型
                    self.app = FaceAnalysis(name=try_model_name, providers=providers)
                    self.app.prepare(ctx_id=0 if self.device.type == 'cuda' and 'CUDAExecutionProvider' in available_providers else -1, det_size=(640, 640))
                    self.use_insightface = True
                    print(f"  ✓ 成功加载insightface模型: {try_model_name}")
                    print(f"  使用providers: {providers}")
                    break
                except Exception as e:
                    print(f"  ✗ 模型 {try_model_name} 加载失败: {str(e)}")
                    continue
            
            if not self.use_insightface:
                raise Exception("所有insightface模型加载失败，将使用fallback实现")
                
        except (ImportError, Exception) as e:
            # 如果insightface不可用或加载失败，使用fallback实现
            if isinstance(e, ImportError):
                print(f"  警告: insightface未安装，使用fallback实现")
                print(f"  建议: 安装insightface以获得更好的ArcFace支持: pip install insightface onnxruntime")
            else:
                print(f"  警告: insightface模型加载失败，使用fallback实现")
                print(f"  原因: {str(e)}")
                print(f"  提示: 将使用timm模型作为替代")
            try:
                # 使用timm中的arcface模型
                self.model = timm.create_model(
                    'tf_efficientnet_b0.ns_jft_in1k',  # 使用timm中的模型作为替代
                    pretrained=True,
                    num_classes=0,  # 移除分类头
                )
                self.model.eval()
                self.model.to(self.device)
                self.use_insightface = False
                
                # 图像预处理
                self.transform = transforms.Compose([
                    transforms.Resize((112, 112)),  # ArcFace标准输入尺寸
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            except Exception as e:
                # 如果都不可用，使用简化的ArcFace实现
                print(f"  警告: 无法加载ArcFace模型，使用简化实现: {str(e)}")
                self.use_insightface = False
                self.model = None
                self.transform = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
    
    def extract_features(self, face_image):
        """
        提取人脸特征
        
        Args:
            face_image: PIL Image对象或numpy array (人脸图像)
            
        Returns:
            features: 特征向量 (numpy array, 512维)
        """
        if self.use_insightface:
            # 使用insightface提取特征
            import torch
            import cv2
            
            # 处理不同类型的输入
            if isinstance(face_image, torch.Tensor):
                # 如果是Tensor，转换为numpy array
                # 确保Tensor在CPU上并detach
                if face_image.is_cuda:
                    face_image = face_image.cpu()
                face_image = face_image.detach()
                
                # 转换为numpy array
                if face_image.dim() == 3:
                    # [C, H, W] -> [H, W, C]
                    img_array = face_image.permute(1, 2, 0).numpy()
                elif face_image.dim() == 4:
                    # [B, C, H, W] -> [H, W, C]
                    img_array = face_image[0].permute(1, 2, 0).numpy()
                else:
                    img_array = face_image.numpy()
                
                # 确保是numpy array类型
                if not isinstance(img_array, np.ndarray):
                    img_array = np.array(img_array)
                
                # 确保数据类型正确
                img_array = img_array.astype(np.float32)
                
                # 归一化处理（MTCNN返回的tensor通常在[-1, 1]或[0, 1]范围）
                if img_array.min() < 0:
                    # 假设是[-1, 1]范围，转换到[0, 255]
                    img_array = ((img_array + 1) / 2 * 255).astype(np.uint8)
                elif img_array.max() <= 1.0:
                    # 假设是[0, 1]范围
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    # 已经在[0, 255]范围，直接转换类型
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                # 确保是3通道图像
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]  # 去掉alpha通道
                elif len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    raise ValueError(f"不支持的图像形状: {img_array.shape}")
                
                # 转换为BGR格式（insightface需要）
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif isinstance(face_image, Image.Image):
                # 如果是PIL Image，转换为numpy array
                img_array = np.array(face_image)
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                # 假设已经是numpy array，确保格式正确
                if isinstance(face_image, np.ndarray):
                    img_bgr = face_image.copy()
                    if len(img_bgr.shape) == 2:
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
                    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
                        # 假设是RGB，转换为BGR
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
                else:
                    # 尝试转换为numpy array
                    img_bgr = np.array(face_image)
                    if len(img_bgr.shape) == 2:
                        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
            
            # 最终检查：确保img_bgr是numpy array且格式正确
            if not isinstance(img_bgr, np.ndarray):
                raise TypeError(f"无法将输入转换为numpy array，类型: {type(img_bgr)}")
            
            # 确保是uint8类型
            if img_bgr.dtype != np.uint8:
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
            
            # 对于已经裁剪的人脸，使用insightface提取特征
            # 方法：直接使用recognition模型的session进行推理
            try:
                # 将图像resize到112x112（ArcFace标准输入）
                img_resized = cv2.resize(img_bgr, (112, 112))
                
                # 直接使用recognition模型提取特征（优先方法）
                if hasattr(self.app, 'models') and 'recognition' in self.app.models:
                    rec_model = self.app.models['recognition']
                    
                    # 准备输入：转换为模型期望的格式
                    # insightface的recognition模型期望 [1, 3, 112, 112] 格式，RGB，归一化到[-1, 1]
                    face_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    face_img = face_img.astype(np.float32)
                    face_img = (face_img / 255.0 - 0.5) / 0.5  # 归一化到[-1, 1]
                    face_img = np.transpose(face_img, (2, 0, 1))  # HWC -> CHW
                    face_img = np.expand_dims(face_img, axis=0)  # 添加batch维度 [1, 3, 112, 112]
                    
                    # 使用onnxruntime session直接推理（推荐方法）
                    if hasattr(rec_model, 'session'):
                        try:
                            # 获取输入名称
                            input_name = rec_model.session.get_inputs()[0].name
                            # 运行推理
                            outputs = rec_model.session.run(None, {input_name: face_img})
                            features = outputs[0][0]  # 获取第一个输出的第一个样本
                            
                            # L2归一化
                            features = features.flatten()
                            features = features / np.linalg.norm(features)
                            return features
                        except Exception as e_session:
                            # session推理失败，尝试其他方法
                            raise ValueError(f"session推理失败: {str(e_session)}")
                    else:
                        raise ValueError("recognition模型没有session属性")
                else:
                    raise ValueError("无法访问recognition模型")
                    
            except Exception as e:
                # 如果直接使用recognition模型失败，尝试使用FaceAnalysis的get方法
                # 注意：对于已裁剪的人脸，get方法可能无法检测到人脸
                # 所以我们需要创建一个包含人脸的完整图像，或者使用其他方法
                try:
                    # 方法：将已裁剪的人脸图像放在一个更大的背景中
                    # 这样可以确保get方法能够检测到人脸
                    h, w = img_bgr.shape[:2]
                    # 创建一个更大的图像（至少640x640，insightface的默认检测尺寸）
                    min_size = 640
                    if h < min_size or w < min_size:
                        # 创建一个更大的背景
                        scale = max(min_size / h, min_size / w) * 1.2
                        new_h, new_w = int(h * scale), int(w * scale)
                        # 创建白色背景
                        large_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
                        # 将人脸图像放在中心
                        y_offset = (new_h - h) // 2
                        x_offset = (new_w - w) // 2
                        large_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_bgr
                        img_bgr = large_img
                    
                    # 使用FaceAnalysis的get方法
                    faces = self.app.get(img_bgr)
                    if len(faces) > 0:
                        features = faces[0].normed_embedding
                        features = features / np.linalg.norm(features)
                        return features
                    else:
                        raise ValueError(f"特征提取失败: {str(e)}")
                except Exception as e2:
                    raise ValueError(f"特征提取失败: {str(e)}, 回退方法也失败: {str(e2)}")
        else:
            # 使用timm模型或简化实现
            import torch
            
            # 处理不同类型的输入
            if isinstance(face_image, torch.Tensor):
                # 如果是Tensor，转换为PIL Image
                if face_image.is_cuda:
                    face_image = face_image.cpu()
                face_image = face_image.detach()
                
                # 转换为numpy array
                if face_image.dim() == 3:
                    # [C, H, W] -> [H, W, C]
                    img_array = face_image.permute(1, 2, 0).numpy()
                elif face_image.dim() == 4:
                    # [B, C, H, W] -> [H, W, C]
                    img_array = face_image[0].permute(1, 2, 0).numpy()
                else:
                    img_array = face_image.numpy()
                
                # 归一化处理
                if img_array.min() < 0:
                    # [-1, 1] -> [0, 255]
                    img_array = ((img_array + 1) / 2 * 255).astype(np.uint8)
                elif img_array.max() <= 1.0:
                    # [0, 1] -> [0, 255]
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                
                # 转换为PIL Image
                img = Image.fromarray(img_array).convert('RGB')
            elif isinstance(face_image, Image.Image):
                img = face_image.convert('RGB')
            elif isinstance(face_image, np.ndarray):
                # 如果是numpy array
                if len(face_image.shape) == 2:
                    img = Image.fromarray(face_image).convert('RGB')
                else:
                    img = Image.fromarray(face_image).convert('RGB')
            else:
                # 尝试转换为PIL Image
                try:
                    img = Image.fromarray(np.array(face_image)).convert('RGB')
                except:
                    raise TypeError(f"无法处理输入类型: {type(face_image)}")
            
            # 预处理
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            if self.model is not None:
                # 使用timm模型提取特征
                with torch.no_grad():
                    features = self.model(img_tensor)
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                return features.cpu().numpy().flatten()
            else:
                # 简化实现：使用ResNet作为backbone
                # 这里可以使用一个简单的特征提取网络
                import torchvision.models as models
                if not hasattr(self, '_backup_model'):
                    # 使用ResNet50作为backbone
                    backbone = models.resnet50(pretrained=True)
                    # 移除最后的分类层
                    self._backup_model = torch.nn.Sequential(*list(backbone.children())[:-1])
                    self._backup_model.eval()
                    self._backup_model.to(self.device)
                
                with torch.no_grad():
                    features = self._backup_model(img_tensor)
                    features = features.view(features.size(0), -1)
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                # 降维到512（如果特征维度不是512）
                features_np = features.cpu().numpy().flatten()
                if len(features_np) != 512:
                    # 使用PCA或线性层降维（这里简化处理，直接截断或填充）
                    if len(features_np) > 512:
                        features_np = features_np[:512]
                    else:
                        # 填充到512
                        features_np = np.pad(features_np, (0, 512 - len(features_np)), 'constant')
                return features_np

