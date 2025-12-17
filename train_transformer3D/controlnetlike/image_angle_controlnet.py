"""
模型2：ImageAngleControlNet（图片角度控制）
输入图片，获取图片姿势特征，受另一个姿势特征调控，获得该角度的图片

参考 ControlNet 架构：
- 图片编码器：提取图片特征
- 姿势提取器：从图片中提取姿势特征
- 姿势控制网络：处理目标姿势（ControlNet-like）
- 图片解码器：生成目标角度的图片
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Optional, Tuple

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.utils_seed import set_seed
from train_transformer3D.controlnetlike.utils import ZeroLinear, ZeroConv1d

# 设置随机种子
set_seed(42)


class ImageEncoder(nn.Module):
    """
    图片编码器：提取图片特征
    使用轻量级CNN架构
    """
    def __init__(self, image_size=224, feature_dim=512):
        super().__init__()
        self.image_size = image_size
        self.feature_dim = feature_dim
        
        # 使用ResNet-like架构
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差块
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 投影到特征维度
        self.fc = nn.Linear(512, feature_dim)
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        """创建残差层"""
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(ResidualBlock(in_channels if i == 0 else out_channels, 
                                       out_channels, stride))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: 输入图片 [batch, 3, H, W]
        Returns:
            features: 图片特征 [batch, feature_dim]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 下采样投影
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class PoseExtractor(nn.Module):
    """
    姿势提取器：从图片特征中提取姿势特征
    """
    def __init__(self, feature_dim=512, pose_dim=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.pose_dim = pose_dim
        
        self.pose_extractor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, pose_dim)
        )
    
    def forward(self, image_features):
        """
        Args:
            image_features: 图片特征 [batch, feature_dim]
        Returns:
            pose: 提取的姿势 [batch, pose_dim]
        """
        return self.pose_extractor(image_features)


class PoseControlBlock(nn.Module):
    """
    姿势控制块（ControlNet-like）
    处理目标姿势，生成控制信号
    """
    def __init__(self, pose_dim=3, hidden_dim=512, num_layers=3):
        super().__init__()
        self.pose_dim = pose_dim
        self.hidden_dim = hidden_dim
        
        # 姿势编码器
        layers = []
        in_dim = pose_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        
        self.pose_encoder = nn.Sequential(*layers)
        
        # 零卷积输出（ControlNet核心）
        self.zero_conv = ZeroLinear(hidden_dim, hidden_dim)
    
    def forward(self, target_pose):
        """
        Args:
            target_pose: 目标姿势 [batch, pose_dim]
        Returns:
            control_signal: 控制信号 [batch, hidden_dim]
        """
        encoded = self.pose_encoder(target_pose)
        control = self.zero_conv(encoded)
        return control


class ImageDecoder(nn.Module):
    """
    图片解码器：从特征生成图片
    使用转置卷积
    """
    def __init__(self, feature_dim=512, image_size=224):
        super().__init__()
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        # 初始投影
        self.fc = nn.Linear(feature_dim, 512 * 7 * 7)  # 假设特征图大小为7x7
        
        # 转置卷积层
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, features):
        """
        Args:
            features: 特征 [batch, feature_dim]
        Returns:
            image: 生成的图片 [batch, 3, H, W]
        """
        x = self.fc(features)
        x = x.view(x.size(0), 512, 7, 7)
        
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.deconv5(x)
        x = self.tanh(x)  # 输出范围[-1, 1]
        
        # 调整到目标尺寸
        if x.size(-1) != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), 
                            mode='bilinear', align_corners=False)
        
        return x


class ImageAngleControlNet(nn.Module):
    """
    图片角度控制网络
    
    输入：
    - src_image: 源图片 [batch, 3, H, W]
    - target_pose: 目标姿势 [batch, pose_dim]（控制信号）
    
    输出：
    - target_image: 目标角度图片 [batch, 3, H, W]
    """
    def __init__(
        self,
        image_size: int = 224,
        pose_dim: int = 3,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        num_control_layers: int = 3,
        num_fusion_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            image_size: 图片尺寸
            pose_dim: 姿势维度（默认3，欧拉角）
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            num_control_layers: 控制网络层数
            num_fusion_layers: 融合网络层数
            dropout: Dropout比率
        """
        super().__init__()
        self.image_size = image_size
        self.pose_dim = pose_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # ========== 1. 图片编码器 ==========
        self.image_encoder = ImageEncoder(
            image_size=image_size,
            feature_dim=feature_dim
        )
        
        # ========== 2. 姿势提取器 ==========
        self.pose_extractor = PoseExtractor(
            feature_dim=feature_dim,
            pose_dim=pose_dim
        )
        
        # ========== 3. 姿势控制网络（ControlNet-like） ==========
        self.pose_control = PoseControlBlock(
            pose_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=num_control_layers
        )
        
        # ========== 4. 特征融合网络 ==========
        # 输入：图片特征 + 控制信号
        fusion_input_dim = feature_dim + hidden_dim
        
        fusion_layers = []
        in_dim = fusion_input_dim
        for i in range(num_fusion_layers):
            fusion_layers.append(nn.Linear(in_dim, hidden_dim))
            fusion_layers.append(nn.LayerNorm(hidden_dim))
            fusion_layers.append(nn.ReLU(inplace=True))
            fusion_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.fusion_network = nn.Sequential(*fusion_layers)
        
        # ========== 5. 图片解码器 ==========
        self.image_decoder = ImageDecoder(
            feature_dim=hidden_dim,
            image_size=image_size
        )
        
        # ========== 6. 零卷积输出（ControlNet核心） ==========
        self.zero_conv_output = ZeroLinear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        src_image: torch.Tensor,
        target_pose: torch.Tensor,
        return_extracted_pose: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            src_image: 源图片 [batch, 3, H, W]
            target_pose: 目标姿势 [batch, pose_dim]（控制信号）
            return_extracted_pose: 是否返回提取的姿势
        
        Returns:
            target_image: 目标角度图片 [batch, 3, H, W]
            extracted_pose: 从源图片提取的姿势 [batch, pose_dim]（如果return_extracted_pose=True）
        """
        batch_size = src_image.size(0)
        
        # ========== 1. 编码图片 ==========
        image_features = self.image_encoder(src_image)  # [batch, feature_dim]
        
        # ========== 2. 提取姿势 ==========
        extracted_pose = self.pose_extractor(image_features)  # [batch, pose_dim]
        
        # ========== 3. 姿势控制网络（ControlNet-like） ==========
        control_signal = self.pose_control(target_pose)  # [batch, hidden_dim]
        
        # ========== 4. 特征融合 ==========
        # 拼接图片特征和控制信号
        fused = torch.cat([
            image_features,    # [batch, feature_dim]
            control_signal     # [batch, hidden_dim]
        ], dim=1)  # [batch, feature_dim + hidden_dim]
        
        # 通过融合网络
        fused_features = self.fusion_network(fused)  # [batch, hidden_dim]
        
        # ========== 5. 生成目标图片 ==========
        # 零卷积输出（ControlNet核心：训练初期不影响主网络）
        fused_features_delta = self.zero_conv_output(fused_features)  # [batch, hidden_dim]
        
        # 残差连接
        final_features = fused_features + fused_features_delta
        
        # 解码为图片
        target_image = self.image_decoder(final_features)  # [batch, 3, H, W]
        
        # 归一化到[0, 1]范围（从tanh的[-1, 1]转换）
        target_image = (target_image + 1) / 2
        
        if return_extracted_pose:
            return target_image, extracted_pose
        else:
            return target_image, None


if __name__ == "__main__":
    """测试代码"""
    print("=" * 70)
    print("ImageAngleControlNet 测试")
    print("=" * 70)
    
    # 创建模型
    model = ImageAngleControlNet(
        image_size=224,
        pose_dim=3,
        feature_dim=512,
        hidden_dim=512
    )
    
    # 测试前向传播
    batch_size = 4
    src_image = torch.randn(batch_size, 3, 224, 224)
    target_pose = torch.randn(batch_size, 3)
    
    print(f"\n输入形状:")
    print(f"  源图片: {src_image.shape}")
    print(f"  目标姿势: {target_pose.shape}")
    
    # 前向传播
    target_image, extracted_pose = model(
        src_image, target_pose, return_extracted_pose=True
    )
    
    print(f"\n输出形状:")
    print(f"  目标图片: {target_image.shape}")
    print(f"  提取的姿势: {extracted_pose.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

