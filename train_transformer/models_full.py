"""
完整模型定义 - 包含图像编码器和Transformer
输入为原始图像，输出为转换后的特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.models import SimpleTransformerEncoder, AnglePositionalEncoding
import os


class ImageEncoder(nn.Module):
    """
    图像编码器 - 支持ResNet50和DINOv2
    """
    
    def __init__(
        self,
        encoder_type: str = 'resnet50',  # 'resnet50' 或 'dinov2'
        feature_dim: int = 768,
        freeze_backbone: bool = False,
        dinov2_model_name: str = 'dinov2_vitb14',
        device: Optional[torch.device] = None
    ):
        """
        初始化图像编码器
        
        Args:
            encoder_type: 编码器类型 ('resnet50' 或 'dinov2')
            feature_dim: 输出特征维度
            freeze_backbone: 是否冻结backbone（用于迁移学习）
            dinov2_model_name: DINOv2模型名称（仅当encoder_type='dinov2'时使用）
            device: 计算设备
        """
        super(ImageEncoder, self).__init__()
        
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if encoder_type == 'resnet50':
            # ResNet50编码器
            resnet = models.resnet50(pretrained=True)
            # 移除最后的分类层和全局平均池化层
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # 保留到conv层
            # 添加全局平均池化和投影层
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            # ResNet50的最后一层输出是2048维，需要投影到目标维度
            self.projection = nn.Linear(2048, feature_dim)
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # 图像预处理（ImageNet标准化）
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
        elif encoder_type == 'dinov2':
            # DINOv2编码器
            # 直接使用torch.hub加载DINOv2模型（更高效）
            import torch.hub
            
            # 设置模型下载镜像
            try:
                from model_utils import setup_model_mirrors
                setup_model_mirrors()
            except ImportError:
                if 'HF_ENDPOINT' not in os.environ:
                    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            
            # 加载DINOv2模型
            try:
                self.backbone = torch.hub.load('facebookresearch/dinov2', dinov2_model_name, pretrained=True)
                self.backbone.eval()
            except Exception as e:
                raise RuntimeError(f"无法加载DINOv2模型 {dinov2_model_name}: {e}")
            
            # DINOv2已经输出正确维度，但可能需要投影层
            dinov2_dims = {
                'dinov2_vits14': 384,
                'dinov2_vitb14': 768,
                'dinov2_vitl14': 1024,
                'dinov2_vitg14': 1536
            }
            dinov2_dim = dinov2_dims.get(dinov2_model_name, 768)
            
            if dinov2_dim != feature_dim:
                self.projection = nn.Linear(dinov2_dim, feature_dim)
            else:
                self.projection = nn.Identity()
            
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # DINOv2预处理（224x224，ImageNet标准化）
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # 初始化投影层
        if hasattr(self, 'projection') and isinstance(self.projection, nn.Linear):
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像 [B, C, H, W]（已归一化）
            
        Returns:
            features: 特征向量 [B, feature_dim]
        """
        if self.encoder_type == 'resnet50':
            # ResNet50编码
            x = self.backbone(images)  # [B, 2048, H', W']
            x = self.global_pool(x)  # [B, 2048, 1, 1]
            x = x.view(x.size(0), -1)  # [B, 2048]
            features = self.projection(x)  # [B, feature_dim]
            # L2归一化
            features = F.normalize(features, p=2, dim=1)
            
        elif self.encoder_type == 'dinov2':
            # DINOv2编码
            with torch.set_grad_enabled(self.training):
                output = self.backbone(images)
                # DINOv2输出处理
                if isinstance(output, dict):
                    if 'x_norm_clstoken' in output:
                        features = output['x_norm_clstoken']
                    elif 'x_prenorm' in output:
                        features = output['x_prenorm'][:, 0]  # CLS token
                    else:
                        # 尝试获取第一个值
                        features = list(output.values())[0]
                        if features.dim() > 2:
                            features = features[:, 0]  # 取CLS token
                else:
                    features = output
                    if features.dim() > 2:
                        # 如果是序列输出，取CLS token（第一个token）
                        features = features[:, 0]
                
                # 投影到目标维度
                features = self.projection(features)
                # L2归一化
                features = F.normalize(features, p=2, dim=1)
        
        return features
    
    def extract_features(self, image) -> torch.Tensor:
        """
        从PIL Image或路径提取特征（推理模式）
        
        Args:
            image: PIL Image对象或图像路径
            
        Returns:
            features: 特征向量 [feature_dim]
        """
        self.eval()
        
        # 加载图像
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image
        
        # 预处理
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.forward(img_tensor)
        
        return features[0]  # 返回单个特征向量


class FullMultiAngleFaceModel(nn.Module):
    """
    完整的多角度人脸特征转换模型
    包含图像编码器和Transformer
    输入：多角度图像和正面图像（原始图像）
    输出：转换后的特征
    """
    
    def __init__(
        self,
        encoder_type: str = 'resnet50',  # 'resnet50' 或 'dinov2'
        feature_dim: int = 768,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_angle_pe: bool = True,
        use_angle_conditioning: bool = True,
        freeze_encoder: bool = False,
        dinov2_model_name: str = 'dinov2_vitb14',
        device: Optional[torch.device] = None
    ):
        """
        初始化完整模型
        
        Args:
            encoder_type: 图像编码器类型 ('resnet50' 或 'dinov2')
            feature_dim: 特征维度（编码器输出维度）
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            use_angle_pe: 是否使用角度位置编码
            use_angle_conditioning: 是否使用角度条件归一化
            freeze_encoder: 是否冻结图像编码器
            dinov2_model_name: DINOv2模型名称（仅当encoder_type='dinov2'时使用）
            device: 计算设备
        """
        super(FullMultiAngleFaceModel, self).__init__()
        
        self.encoder_type = encoder_type
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 图像编码器（用于提取多角度图像特征）
        self.image_encoder = ImageEncoder(
            encoder_type=encoder_type,
            feature_dim=feature_dim,
            freeze_backbone=freeze_encoder,
            dinov2_model_name=dinov2_model_name,
            device=self.device
        )
        
        # 正面图像编码器（可选，可以与多角度编码器共享权重）
        # 如果共享权重，可以节省参数和显存
        self.face_encoder = self.image_encoder  # 共享编码器
        
        # 特征维度对齐（如果需要）
        if feature_dim != d_model:
            self.feature_projection = nn.Linear(feature_dim, d_model)
        else:
            self.feature_projection = nn.Identity()
        
        # Transformer（用于特征转换）
        self.transformer = SimpleTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_angle_pe=use_angle_pe,
            use_angle_conditioning=use_angle_conditioning,
            angle_dim=5
        )
        
        # 输出投影层（可选）
        self.output_projection = nn.Linear(d_model, d_model)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        if isinstance(self.feature_projection, nn.Linear):
            nn.init.xavier_uniform_(self.feature_projection.weight)
            nn.init.zeros_(self.feature_projection.bias)
        
        if isinstance(self.output_projection, nn.Linear):
            nn.init.xavier_uniform_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        multi_angle_images: torch.Tensor,
        face_images: torch.Tensor,
        angles: torch.Tensor,
        return_residual: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            multi_angle_images: 多角度图像 [B, C, H, W]（已归一化）
            face_images: 正面图像 [B, C, H, W]（已归一化，用于计算目标特征）
            angles: 球面角 [B, 5]
            return_residual: 是否返回残差（True）或完整特征（False）
            
        Returns:
            predicted_features: 预测的特征 [B, d_model]
            target_features: 目标特征（正面图像特征）[B, d_model]
        """
        # 提取多角度图像特征
        input_features = self.image_encoder(multi_angle_images)  # [B, feature_dim]
        input_features = self.feature_projection(input_features)  # [B, d_model]
        
        # 提取正面图像特征（目标特征）
        target_features = self.face_encoder(face_images)  # [B, feature_dim]
        target_features = self.feature_projection(target_features)  # [B, d_model]
        
        # Transformer转换
        residual = self.transformer(input_features, angles, return_residual=return_residual)  # [B, d_model]
        
        # 应用残差得到矫正后的特征
        if return_residual:
            corrected_features = input_features + residual
        else:
            corrected_features = residual
        
        # 输出投影
        predicted_features = self.output_projection(corrected_features)
        
        return predicted_features, target_features
    
    def predict(
        self,
        multi_angle_image: torch.Tensor,
        angles: torch.Tensor
    ) -> torch.Tensor:
        """
        预测模式（只需要多角度图像和角度）
        
        Args:
            multi_angle_image: 多角度图像 [C, H, W] 或 [1, C, H, W]（已归一化）
            angles: 球面角 [5] 或 [1, 5]
            
        Returns:
            predicted_features: 预测的特征 [d_model]
        """
        self.eval()
        
        # 添加batch维度
        if multi_angle_image.dim() == 3:
            multi_angle_image = multi_angle_image.unsqueeze(0)
        if angles.dim() == 1:
            angles = angles.unsqueeze(0)
        
        # 提取特征
        input_features = self.image_encoder(multi_angle_image)  # [1, feature_dim]
        input_features = self.feature_projection(input_features)  # [1, d_model]
        
        # Transformer转换
        residual = self.transformer(input_features, angles, return_residual=True)  # [1, d_model]
        
        # 应用残差
        corrected_features = input_features + residual
        
        # 输出投影
        predicted_features = self.output_projection(corrected_features)
        
        return predicted_features[0]  # 返回单个特征向量


if __name__ == "__main__":
    # 测试模型
    print("=" * 70)
    print("测试 FullMultiAngleFaceModel")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试ResNet50版本
    print("\n" + "-" * 70)
    print("测试 ResNet50 编码器")
    print("-" * 70)
    
    model_resnet = FullMultiAngleFaceModel(
        encoder_type='resnet50',
        feature_dim=768,
        d_model=768,
        nhead=8,
        num_layers=4,
        freeze_encoder=False,
        device=device
    ).to(device)
    
    print(f"ResNet50模型参数数量: {sum(p.numel() for p in model_resnet.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model_resnet.parameters() if p.requires_grad):,}")
    
    # 创建测试数据
    batch_size = 2
    multi_angle_images = torch.randn(batch_size, 3, 224, 224).to(device)
    face_images = torch.randn(batch_size, 3, 224, 224).to(device)
    angles = torch.randn(batch_size, 5).to(device)
    
    print(f"\n输入形状:")
    print(f"  多角度图像: {multi_angle_images.shape}")
    print(f"  正面图像: {face_images.shape}")
    print(f"  角度: {angles.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    model_resnet.eval()
    with torch.no_grad():
        predicted_features, target_features = model_resnet(
            multi_angle_images, face_images, angles
        )
    
    print(f"输出形状:")
    print(f"  预测特征: {predicted_features.shape}")
    print(f"  目标特征: {target_features.shape}")
    
    # 测试DINOv2版本（如果可用）
    print("\n" + "-" * 70)
    print("测试 DINOv2 编码器")
    print("-" * 70)
    
    try:
        model_dinov2 = FullMultiAngleFaceModel(
            encoder_type='dinov2',
            feature_dim=768,
            d_model=768,
            nhead=8,
            num_layers=4,
            dinov2_model_name='dinov2_vitb14',
            freeze_encoder=False,
            device=device
        ).to(device)
        
        print(f"DINOv2模型参数数量: {sum(p.numel() for p in model_dinov2.parameters()):,}")
        print(f"可训练参数数量: {sum(p.numel() for p in model_dinov2.parameters() if p.requires_grad):,}")
        
        # 前向传播
        print("\n执行前向传播...")
        model_dinov2.eval()
        with torch.no_grad():
            predicted_features, target_features = model_dinov2(
                multi_angle_images, face_images, angles
            )
        
        print(f"输出形状:")
        print(f"  预测特征: {predicted_features.shape}")
        print(f"  目标特征: {target_features.shape}")
        
    except Exception as e:
        print(f"⚠️  DINOv2测试失败: {e}")
        print("   这可能是正常的，如果DINOv2模型未下载")
    
    # 对比参数量
    print("\n" + "=" * 70)
    print("参数量对比:")
    print("=" * 70)
    resnet_params = sum(p.numel() for p in model_resnet.parameters())
    print(f"ResNet50版本: {resnet_params:,}")
    
    try:
        dinov2_params = sum(p.numel() for p in model_dinov2.parameters())
        print(f"DINOv2版本:  {dinov2_params:,}")
        print(f"差异: {dinov2_params - resnet_params:,} ({(dinov2_params - resnet_params) / resnet_params * 100:.1f}%)")
    except:
        pass
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

