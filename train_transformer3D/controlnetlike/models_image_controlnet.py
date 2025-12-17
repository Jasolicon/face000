"""
模型2：图像生成ControlNet
输入：图片
控制：另一个姿势特征
输出：该角度的图片

参考ControlNet设计：
- 主网络：冻结的预训练图像生成网络（或使用特征到图像的生成器）
- 控制分支：可训练的网络，接收控制姿势
- 零卷积：将控制信号注入主网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.utils_seed import set_seed

set_seed(42)


class ZeroConv1d(nn.Module):
    """
    零卷积层（Zero Convolution）for 1D features
    ControlNet的核心：初始时输出为零，不干扰主网络
    训练过程中逐渐学习控制信号
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # 初始化为零
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch, in_channels, length] 或 [batch, in_channels]
        Returns:
            output: 输出特征 [batch, out_channels, length] 或 [batch, out_channels]
        """
        if x.dim() == 2:
            # [batch, in_channels] -> [batch, in_channels, 1]
            x = x.unsqueeze(-1)
            output = self.conv(x)  # [batch, out_channels, 1]
            output = output.squeeze(-1)  # [batch, out_channels]
        else:
            output = self.conv(x)
        return output


class ZeroConv2d(nn.Module):
    """
    零卷积层（Zero Convolution）for 2D features
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        # 初始化为零
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ImageEncoder(nn.Module):
    """
    图像编码器：将图像编码为特征
    使用InsightFace的冻结卷积backbone（去掉全连接部分）
    InsightFace通常使用ResNet作为backbone，我们使用timm中的ResNet
    """
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 512,
        image_size: int = 112,  # InsightFace标准尺寸
        use_insightface: bool = True,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        self.use_insightface = use_insightface
        self.freeze_backbone = freeze_backbone
        
        if use_insightface:
            # 尝试使用真正的InsightFace预训练模型
            # 方案1：尝试使用insightface库（ONNX模型）
            insightface_loaded = False
            try:
                import insightface
                from insightface.app import FaceAnalysis
                import onnxruntime as ort
                
                # 尝试加载InsightFace模型
                try:
                    # 检测可用的providers
                    available_providers = ort.get_available_providers()
                    if torch.cuda.is_available() and 'CUDAExecutionProvider' in available_providers:
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                        ctx_id = 0
                    else:
                        providers = ['CPUExecutionProvider']
                        ctx_id = -1
                    
                    # 初始化InsightFace模型
                    self.insightface_app = FaceAnalysis(
                        name='buffalo_l',  # 或 'buffalo_s', 'buffalo_m'
                        providers=providers
                    )
                    self.insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                    
                    # 获取backbone模型（recognition model）
                    # InsightFace的recognition model就是backbone + 全连接层
                    # 我们需要提取backbone部分
                    # 注意：InsightFace的ONNX模型结构较复杂，这里我们使用一个变通方案
                    # 在forward中直接调用InsightFace提取特征
                    
                    insightface_loaded = True
                    print("✓ 成功加载InsightFace模型（ONNX格式）")
                    print("  注意：将在forward中直接使用InsightFace提取特征")
                    
                    # 特征投影层（InsightFace输出512维，直接使用或投影）
                    # InsightFace已经输出512维特征，如果feature_dim也是512，可以直接使用
                    if feature_dim == 512:
                        self.feature_proj = nn.Identity()  # 直接使用，无需投影
                    else:
                        self.feature_proj = nn.Sequential(
                            nn.Linear(512, feature_dim),
                            nn.BatchNorm1d(feature_dim),
                            nn.ReLU(inplace=True)
                        )
                    
                except Exception as e:
                    print(f"⚠️ InsightFace ONNX模型加载失败: {e}")
                    print("  将尝试使用timm的ResNet50（ImageNet预训练）作为替代")
                    insightface_loaded = False
            
            except ImportError:
                print("⚠️ insightface库未安装")
                print("  将使用timm的ResNet50（ImageNet预训练）作为替代")
                print("  如需使用真正的InsightFace预训练，请安装: pip install insightface onnxruntime")
                insightface_loaded = False
            
            # 方案2：如果InsightFace加载失败，使用timm的ResNet50（ImageNet预训练）
            if not insightface_loaded:
                try:
                    import timm
                    # 使用ResNet50作为backbone（ImageNet预训练，不是InsightFace预训练）
                    self.backbone = timm.create_model(
                        'resnet50',
                        pretrained=True,  # ImageNet预训练
                        num_classes=0,  # 移除分类头，只使用backbone
                        global_pool='avg'  # 全局平均池化
                    )
                    backbone_output_dim = 2048  # ResNet50的输出维度
                    print("✓ 使用ResNet50作为backbone（ImageNet预训练，冻结卷积部分）")
                    print("  注意：这是ImageNet预训练，不是InsightFace预训练")
                    
                    # 冻结backbone参数
                    if freeze_backbone:
                        for param in self.backbone.parameters():
                            param.requires_grad = False
                        print("✓ Backbone已冻结")
                    
                    # 特征投影层
                    self.feature_proj = nn.Sequential(
                        nn.Linear(backbone_output_dim, feature_dim),
                        nn.BatchNorm1d(feature_dim),
                        nn.ReLU(inplace=True)
                    )
                    
                except ImportError:
                    print("⚠️ timm未安装，使用自定义CNN编码器")
                    print("  建议安装: pip install timm")
                    self.use_insightface = False
                    self._init_custom_encoder()
                except Exception as e:
                    print(f"⚠️ 加载backbone失败: {e}")
                    print("  使用自定义CNN编码器作为备用")
                    self.use_insightface = False
                    self._init_custom_encoder()
        else:
            self._init_custom_encoder()
        
        if not self.use_insightface:
            self._init_weights()
    
    def _init_custom_encoder(self):
        """初始化自定义CNN编码器（备用方案）"""
        # CNN编码器
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第四层
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 全局平均池化
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 特征投影
        self.feature_proj = nn.Linear(512, self.feature_dim)
    
    def _init_weights(self):
        """初始化权重（仅用于自定义编码器）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: 输入图像 [batch, 3, H, W]（已归一化到[-1, 1]）
        Returns:
            features: 图像特征 [batch, feature_dim]
        """
        if self.use_insightface:
            # 检查是否使用真正的InsightFace模型（ONNX）
            if hasattr(self, 'insightface_app'):
                # 使用真正的InsightFace模型提取特征
                batch_size = image.size(0)
                features_list = []
                
                # 将输入从[-1, 1]转换到[0, 255]（BGR格式，InsightFace需要）
                image_normalized = (image + 1) / 2.0  # [-1, 1] -> [0, 1]
                image_normalized = image_normalized * 255.0  # [0, 1] -> [0, 255]
                
                # 转换为numpy并处理每个样本
                for i in range(batch_size):
                    img_tensor = image_normalized[i]  # [3, H, W]
                    
                    # 转换为numpy array [H, W, 3] BGR格式
                    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                    img_np = img_np[:, :, ::-1]  # RGB -> BGR
                    img_np = img_np.astype(np.uint8)
                    
                    # 使用InsightFace提取特征
                    faces = self.insightface_app.get(img_np)
                    if len(faces) > 0:
                        # 使用第一个人脸的特征（已经归一化的512维特征）
                        face_features = faces[0].normed_embedding  # [512]
                    else:
                        # 如果没有检测到人脸，使用零向量
                        face_features = np.zeros(512, dtype=np.float32)
                    
                    features_list.append(face_features)
                
                # 转换为tensor
                features_np = np.stack(features_list, axis=0)  # [batch, 512]
                features_tensor = torch.from_numpy(features_np).float().to(image.device)
                
                # 特征投影（如果需要）
                features = self.feature_proj(features_tensor)  # [batch, feature_dim]
                
            elif hasattr(self, 'backbone'):
                # 使用timm的ResNet50（ImageNet预训练）
                # 注意：timm模型通常期望ImageNet归一化，但我们的输入是[-1, 1]
                # 需要转换归一化
                # ImageNet归一化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # 我们的输入: [-1, 1] -> [0, 1] -> ImageNet归一化
                
                # 将输入从[-1, 1]转换到[0, 1]
                image_normalized = (image + 1) / 2.0
                
                # ImageNet归一化
                mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
                image_normalized = (image_normalized - mean) / std
                
                # 通过backbone提取特征（冻结的卷积部分）
                with torch.set_grad_enabled(not self.freeze_backbone):
                    backbone_features = self.backbone(image_normalized)  # [batch, 2048]
                
                # 特征投影
                features = self.feature_proj(backbone_features)  # [batch, feature_dim]
            else:
                # 如果都没有，使用自定义编码器
                x = self.encoder(image)  # [batch, 512, 1, 1]
                x = x.view(x.size(0), -1)  # [batch, 512]
                features = self.feature_proj(x)  # [batch, feature_dim]
        else:
            # 使用自定义编码器
            # CNN编码
            x = self.encoder(image)  # [batch, 512, 1, 1]
            x = x.view(x.size(0), -1)  # [batch, 512]
            
            # 特征投影
            features = self.feature_proj(x)  # [batch, feature_dim]
        
        return features


class PoseExtractor(nn.Module):
    """
    姿势提取器：从图像特征中提取姿势
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3
    ):
        super().__init__()
        self.pose_dim = pose_dim
        
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, pose_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 图像特征 [batch, feature_dim]
        Returns:
            pose: 提取的姿势 [batch, pose_dim]
        """
        return self.pose_head(features)


class ImageControlBranch(nn.Module):
    """
    图像控制分支：接收控制姿势，生成控制信号
    """
    def __init__(
        self,
        control_pose_dim: int = 3,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        self.control_pose_dim = control_pose_dim
        self.feature_dim = feature_dim
        
        # 姿势编码器
        self.pose_encoder = nn.Sequential(
            nn.Linear(control_pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 控制信号生成层
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        self.control_layers = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, feature_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        image_features: torch.Tensor,
        source_pose: torch.Tensor,
        target_pose: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            image_features: 图像特征 [batch, feature_dim]
            source_pose: 源姿势 [batch, pose_dim]
            target_pose: 目标姿势（控制姿势）[batch, pose_dim]
        Returns:
            control_signal: 控制信号 [batch, feature_dim]
        """
        # 计算姿势差异
        pose_diff = target_pose - source_pose  # [batch, pose_dim]
        
        # 编码姿势差异
        pose_features = self.pose_encoder(pose_diff)  # [batch, hidden_dim]
        
        # 融合图像特征和姿势特征
        combined = pose_features + self.pose_encoder(target_pose)  # [batch, hidden_dim]
        
        # 生成控制信号
        control_hidden = self.control_layers(combined)  # [batch, hidden_dim]
        control_signal = self.output_proj(control_hidden)  # [batch, feature_dim]
        
        return control_signal


class ImageGenerator(nn.Module):
    """
    图像生成器：从特征生成图像
    使用转置卷积（反卷积）生成图像
    """
    def __init__(
        self,
        feature_dim: int = 512,
        image_size: int = 112,
        in_channels: int = 3,
        freeze: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        # 特征到空间特征的投影
        spatial_size = image_size // 16  # 经过4次下采样
        self.spatial_proj = nn.Linear(feature_dim, 512 * spatial_size * spatial_size)
        self.spatial_size = spatial_size
        
        # 转置卷积生成器
        self.generator = nn.Sequential(
            # 第一层：512 -> 256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第二层：256 -> 128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第三层：128 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 第四层：64 -> 3
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 [batch, feature_dim]
        Returns:
            image: 生成的图像 [batch, 3, H, W]
        """
        # 投影到空间特征
        spatial_features = self.spatial_proj(features)  # [batch, 512 * spatial_size * spatial_size]
        spatial_features = spatial_features.view(
            features.size(0), 512, self.spatial_size, self.spatial_size
        )  # [batch, 512, spatial_size, spatial_size]
        
        # 生成图像
        image = self.generator(spatial_features)  # [batch, 3, image_size, image_size]
        
        return image


class ImageControlNet(nn.Module):
    """
    图像生成ControlNet
    
    架构：
    1. 图像编码器：将图像编码为特征
    2. 姿势提取器：从图像特征中提取姿势
    3. 控制分支：接收控制姿势，生成控制信号
    4. 零卷积：将控制信号注入主网络
    5. 图像生成器：从特征生成图像
    """
    def __init__(
        self,
        feature_dim: int = 512,
        pose_dim: int = 3,
        image_size: int = 112,
        in_channels: int = 3,
        num_control_layers: int = 3,
        freeze_generator: bool = False,  # 是否冻结生成器
        use_insightface: bool = True,    # 是否使用InsightFace backbone
        freeze_backbone: bool = True     # 是否冻结backbone参数
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        self.image_size = image_size
        
        # 图像编码器（使用InsightFace的冻结卷积backbone）
        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
            feature_dim=feature_dim,
            image_size=image_size,
            use_insightface=use_insightface,  # 使用InsightFace backbone
            freeze_backbone=freeze_backbone   # 冻结backbone参数
        )
        
        # 姿势提取器
        self.pose_extractor = PoseExtractor(
            feature_dim=feature_dim,
            pose_dim=pose_dim
        )
        
        # 控制分支
        self.control_branch = ImageControlBranch(
            control_pose_dim=pose_dim,
            feature_dim=feature_dim,
            hidden_dim=256,
            num_layers=num_control_layers
        )
        
        # 零卷积
        self.zero_conv = ZeroConv1d(feature_dim, feature_dim)
        
        # 图像生成器
        self.image_generator = ImageGenerator(
            feature_dim=feature_dim,
            image_size=image_size,
            in_channels=in_channels,
            freeze=freeze_generator
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def extract_pose_from_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        从图像中提取姿势特征
        
        Args:
            image: 输入图像 [batch, 3, H, W]
        Returns:
            pose: 提取的姿势 [batch, pose_dim]
        """
        # 编码图像
        image_features = self.image_encoder(image)  # [batch, feature_dim]
        
        # 提取姿势
        pose = self.pose_extractor(image_features)  # [batch, pose_dim]
        
        return pose
    
    def forward(
        self,
        image: torch.Tensor,  # 源图像 [batch, 3, H, W]
        target_pose: torch.Tensor,  # 目标姿势 [batch, pose_dim]
        return_control_signal: bool = False,
        return_source_pose: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            image: 源图像 [batch, 3, H, W]
            target_pose: 目标姿势（控制姿势）[batch, pose_dim]
            return_control_signal: 是否返回控制信号
            return_source_pose: 是否返回提取的源姿势
            
        Returns:
            output_image: 目标角度的图像 [batch, 3, H, W]
            control_signal: 控制信号 [batch, feature_dim]（如果return_control_signal=True）
            source_pose: 提取的源姿势 [batch, pose_dim]（如果return_source_pose=True）
        """
        # ========== 1. 从图像中提取特征和姿势 ==========
        image_features = self.image_encoder(image)  # [batch, feature_dim]
        source_pose = self.pose_extractor(image_features)  # [batch, pose_dim]
        
        # ========== 2. 控制分支生成控制信号 ==========
        control_signal = self.control_branch(image_features, source_pose, target_pose)  # [batch, feature_dim]
        
        # ========== 3. 零卷积处理控制信号 ==========
        control_output = self.zero_conv(control_signal.unsqueeze(-1)).squeeze(-1)  # [batch, feature_dim]
        
        # ========== 4. 主网络处理 ==========
        # 主网络接收图像特征
        main_features = image_features  # [batch, feature_dim]
        
        # 融合控制信号
        fused_features = main_features + control_output  # [batch, feature_dim]
        
        # ========== 5. 生成图像 ==========
        output_image = self.image_generator(fused_features)  # [batch, 3, H, W]
        
        # 将输出范围从 [-1, 1] 转换到 [0, 1]（如果需要）
        # output_image = (output_image + 1) / 2
        
        result = (output_image,)
        if return_control_signal:
            result = result + (control_signal,)
        if return_source_pose:
            result = result + (source_pose,)
        
        if len(result) == 1:
            return result[0], None, None
        elif len(result) == 2:
            return result[0], result[1], None
        else:
            return result
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        if self.image_generator.parameters().__next__().requires_grad:
            # 生成器可训练
            return list(self.parameters())
        else:
            # 只返回控制分支和零卷积的参数
            trainable_params = []
            trainable_params.extend(list(self.image_encoder.parameters()))
            trainable_params.extend(list(self.pose_extractor.parameters()))
            trainable_params.extend(list(self.control_branch.parameters()))
            trainable_params.extend(list(self.zero_conv.parameters()))
            return trainable_params


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("图像生成ControlNet测试")
    print("=" * 70)
    
    # 创建模型
    model = ImageControlNet(
        feature_dim=512,
        pose_dim=3,
        image_size=112,
        in_channels=3,
        num_control_layers=3,
        freeze_generator=False
    )
    
    # 测试前向传播
    batch_size = 4
    image = torch.randn(batch_size, 3, 112, 112)
    target_pose = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  图像: {image.shape}")
    print(f"  目标姿势: {target_pose.shape}")
    
    # 前向传播
    output_image, control_signal, source_pose = model(
        image=image,
        target_pose=target_pose,
        return_control_signal=True,
        return_source_pose=True
    )
    
    print(f"\n输出形状:")
    print(f"  输出图像: {output_image.shape}")
    print(f"  控制信号: {control_signal.shape}")
    print(f"  源姿势: {source_pose.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
