"""
使用CLIP编码姿态信息的模块
将姿态角度转换为文本描述，通过CLIP编码器获得语义丰富的姿态表示
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
from pathlib import Path

# 尝试导入CLIP
CLIP_AVAILABLE = False
try:
    import clip
    # 检查是否有load函数（确保是OpenAI的CLIP库）
    if hasattr(clip, 'load'):
        CLIP_AVAILABLE = True
    else:
        print("警告: 导入的clip模块不是OpenAI的CLIP库，缺少load函数")
except ImportError:
    print("警告: CLIP未安装，无法使用CLIP姿态编码器。请安装: pip install ftfy regex tqdm")
except Exception as e:
    print(f"警告: CLIP导入失败: {e}")


def pose_to_text(yaw, pitch, roll):
    """
    将姿态角度转换为文本描述
    
    Args:
        yaw: 左右转头角度（度）
        pitch: 上下抬头角度（度）
        roll: 头部倾斜角度（度）
    
    Returns:
        text: 文本描述
    """
    # Yaw: 左右转头
    if yaw < -45:
        yaw_desc = "left profile view"
    elif yaw < -15:
        yaw_desc = "left three-quarter view"
    elif yaw < 15:
        yaw_desc = "frontal view"
    elif yaw < 45:
        yaw_desc = "right three-quarter view"
    else:
        yaw_desc = "right profile view"
    
    # Pitch: 上下抬头
    if pitch < -20:
        pitch_desc = "looking down"
    elif pitch < 20:
        pitch_desc = "level gaze"
    else:
        pitch_desc = "looking up"
    
    # Roll: 头部倾斜
    if abs(roll) < 10:
        roll_desc = "upright"
    elif roll > 0:
        roll_desc = "tilted right"
    else:
        roll_desc = "tilted left"
    
    return f"{yaw_desc}, {pitch_desc}, {roll_desc}"


def pose_to_text_continuous(yaw, pitch, roll):
    """
    将姿态角度转换为连续文本描述（保留数值精度）
    
    Args:
        yaw: 左右转头角度（度）
        pitch: 上下抬头角度（度）
        roll: 头部倾斜角度（度）
    
    Returns:
        text: 文本描述
    """
    return f"face rotated {yaw:.1f} degrees horizontally, {pitch:.1f} degrees vertically, tilted {roll:.1f} degrees"


class CLIPPoseEncoder(nn.Module):
    """
    使用CLIP编码姿态信息
    
    将姿态角度转换为文本描述，通过CLIP文本编码器获得语义丰富的姿态表示
    """
    
    def __init__(self, 
                 output_dim: int,
                 clip_model_name: str = 'ViT-B/32',
                 device: str = 'cuda',
                 freeze_clip: bool = True,
                 use_continuous_text: bool = False):
        """
        Args:
            output_dim: 输出维度（通常为 dim * 3，用于Q、K、V）
            clip_model_name: CLIP模型名称（'ViT-B/32', 'ViT-B/16', 'ViT-L/14'等）
            device: 设备
            freeze_clip: 是否冻结CLIP参数（只训练投影层）
            use_continuous_text: 是否使用连续数值文本描述
        """
        super().__init__()
        
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP未正确安装或导入失败。\n"
                "请安装OpenAI的CLIP库:\n"
                "  pip install ftfy regex tqdm\n"
                "  pip install git+https://github.com/openai/CLIP.git\n"
                "或使用国内镜像:\n"
                "  pip install git+https://gitee.com/mirrors/CLIP.git"
            )
        
        self.output_dim = output_dim
        self.device = device
        self.freeze_clip = freeze_clip
        self.use_continuous_text = use_continuous_text
        
        # 加载CLIP模型
        print(f"加载CLIP模型: {clip_model_name}")
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        except AttributeError as e:
            raise ImportError(
                f"CLIP库导入失败: {e}\n"
                "请确保安装的是OpenAI的CLIP库:\n"
                "  pip install git+https://github.com/openai/CLIP.git"
            )
        
        if freeze_clip:
            # 冻结CLIP参数
            for param in self.clip_model.parameters():
                param.requires_grad = False
            print("CLIP参数已冻结，只训练投影层")
        
        # CLIP文本编码器输出维度
        self.clip_text_dim = self.clip_model.text_projection.shape[1]  # 通常是512
        
        # 投影层：将CLIP编码投影到模型需要的维度
        self.proj = nn.Sequential(
            nn.Linear(self.clip_text_dim, self.clip_text_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.clip_text_dim, output_dim)
        )
        
        # 缓存文本token（可选，用于加速）
        self._text_cache = {}
        
    def _pose_to_text(self, yaw, pitch, roll):
        """根据配置选择文本描述方式"""
        if self.use_continuous_text:
            return pose_to_text_continuous(yaw, pitch, roll)
        else:
            return pose_to_text(yaw, pitch, roll)
    
    def forward(self, pose_angles):
        """
        Args:
            pose_angles: [batch, 3] (yaw, pitch, roll) 或 [batch, 3] tensor
        
        Returns:
            pose_emb: [batch, output_dim]
        """
        batch_size = pose_angles.shape[0]
        device = pose_angles.device
        
        # 将姿态角度转换为文本描述
        texts = []
        for i in range(batch_size):
            yaw = pose_angles[i, 0].item()
            pitch = pose_angles[i, 1].item()
            roll = pose_angles[i, 2].item()
            text = self._pose_to_text(yaw, pitch, roll)
            texts.append(text)
        
        # 使用CLIP编码文本
        text_tokens = clip.tokenize(texts).to(device)
        
        if self.freeze_clip:
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)  # [batch, clip_text_dim]
        else:
            text_features = self.clip_model.encode_text(text_tokens)  # [batch, clip_text_dim]
        
        # 归一化（CLIP通常需要归一化）
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 投影到模型需要的维度
        pose_emb = self.proj(text_features)  # [batch, output_dim]
        
        return pose_emb


class HybridPoseEncoder(nn.Module):
    """
    混合姿态编码器：结合CLIP语义编码和数值编码
    
    可能比单独使用CLIP或MLP效果更好
    """
    
    def __init__(self,
                 output_dim: int,
                 clip_model_name: str = 'ViT-B/32',
                 device: str = 'cuda',
                 freeze_clip: bool = True,
                 mlp_hidden_dim: int = 64):
        """
        Args:
            output_dim: 输出维度
            clip_model_name: CLIP模型名称
            device: 设备
            freeze_clip: 是否冻结CLIP参数
            mlp_hidden_dim: MLP隐藏层维度
        """
        super().__init__()
        
        # CLIP编码器
        self.clip_encoder = CLIPPoseEncoder(
            output_dim=output_dim // 2,  # CLIP占一半
            clip_model_name=clip_model_name,
            device=device,
            freeze_clip=freeze_clip
        )
        
        # MLP编码器（数值编码）
        self.mlp_encoder = nn.Sequential(
            nn.Linear(3, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim // 2)  # MLP占一半
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, pose_angles):
        """
        Args:
            pose_angles: [batch, 3]
        
        Returns:
            pose_emb: [batch, output_dim]
        """
        # CLIP语义编码
        clip_emb = self.clip_encoder(pose_angles)  # [batch, output_dim // 2]
        
        # MLP数值编码
        mlp_emb = self.mlp_encoder(pose_angles)  # [batch, output_dim // 2]
        
        # 拼接
        combined = torch.cat([clip_emb, mlp_emb], dim=-1)  # [batch, output_dim]
        
        # 融合
        pose_emb = self.fusion(combined)  # [batch, output_dim]
        
        return pose_emb


# 测试代码
if __name__ == "__main__":
    if not CLIP_AVAILABLE:
        print("CLIP未安装，跳过测试")
        exit(0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试CLIP编码器
    print("\n测试CLIP姿态编码器:")
    encoder = CLIPPoseEncoder(output_dim=512 * 3, device=device)
    
    # 创建测试数据
    batch_size = 4
    pose_angles = torch.tensor([
        [45.0, 10.0, -5.0],   # 右转，轻微上仰，左倾
        [-30.0, -15.0, 0.0],  # 左转，下看，正
        [0.0, 0.0, 0.0],      # 正面
        [-60.0, 25.0, 10.0]   # 大角度左转，上仰，右倾
    ], device=device)
    
    # 前向传播
    with torch.no_grad():
        pose_emb = encoder(pose_angles)
    
    print(f"输入姿态角度形状: {pose_angles.shape}")
    print(f"输出编码形状: {pose_emb.shape}")
    print(f"输出编码范围: [{pose_emb.min().item():.4f}, {pose_emb.max().item():.4f}]")
    
    # 测试混合编码器
    print("\n测试混合姿态编码器:")
    hybrid_encoder = HybridPoseEncoder(output_dim=512 * 3, device=device)
    
    with torch.no_grad():
        hybrid_emb = hybrid_encoder(pose_angles)
    
    print(f"混合编码器输出形状: {hybrid_emb.shape}")
    print(f"混合编码器输出范围: [{hybrid_emb.min().item():.4f}, {hybrid_emb.max().item():.4f}]")
    
    print("\n测试完成！")
