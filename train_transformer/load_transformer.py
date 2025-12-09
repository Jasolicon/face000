"""
加载训练好的Transformer模型
"""
import torch
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.models import SimpleTransformerEncoder


def load_transformer_model(checkpoint_path, device=None):
    """
    加载训练好的Transformer模型
    
    Args:
        checkpoint_path: 模型检查点路径
        device: 计算设备（如果为None则自动选择）
        
    Returns:
        model: 加载的模型
        device: 使用的设备
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"加载Transformer模型: {checkpoint_path}")
    print(f"使用设备: {device}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置（从检查点或使用默认值）
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # 检查旧模型是否包含角度条件归一化的参数
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    has_angle_conditioning = any('angle_conditioned_norm' in str(key) for key in (state_dict.keys() if isinstance(state_dict, dict) else []))
    
    # 创建模型（使用默认配置，应该与训练时一致）
    # 如果旧模型没有use_angle_conditioning参数，则不使用角度条件归一化以保持兼容性
    model = SimpleTransformerEncoder(
        d_model=768,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        use_angle_pe=True,
        use_angle_conditioning=has_angle_conditioning,  # 根据检查点自动判断
        angle_dim=5
    )
    
    # 加载权重
    try:
        if isinstance(model_state, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        if not has_angle_conditioning:
            print("  ⚠️ 检测到旧模型（无角度条件归一化），将使用兼容模式")
    except Exception as e:
        print(f"  ⚠️ 模型加载警告: {e}")
        print("  尝试使用strict=False模式...")
        if isinstance(model_state, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Transformer模型加载成功")
    
    return model, device


def correct_features_with_transformer(model, features, angles, device=None):
    """
    使用Transformer模型矫正特征（应用残差）
    
    Args:
        model: Transformer模型
        features: DINOv2特征 [batch_size, 768] 或 [768]
        angles: 球面角 [batch_size, 5] 或 [5]
        device: 计算设备
        
    Returns:
        corrected_features: 矫正后的特征 [batch_size, 768] 或 [768]
    """
    if device is None:
        device = next(model.parameters()).device
    
    # 转换为torch tensor
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features)
    if not isinstance(angles, torch.Tensor):
        angles = torch.FloatTensor(angles)
    
    # 添加batch维度（如果需要）
    if features.dim() == 1:
        features = features.unsqueeze(0)  # [1, 768]
    if angles.dim() == 1:
        angles = angles.unsqueeze(0)  # [1, 5]
    
    # 移动到设备
    features = features.to(device)
    angles = angles.to(device)
    
    # 前向传播：预测残差
    with torch.no_grad():
        predicted_residual = model(features, angles, return_residual=True)
    
    # 应用残差：矫正后的特征 = 原始特征 + 预测的残差
    corrected_features = features + predicted_residual
    
    # 移除batch维度（如果输入是单个特征）
    if corrected_features.dim() == 2 and corrected_features.size(0) == 1:
        corrected_features = corrected_features.squeeze(0)
    
    # 转换为numpy（如果需要）
    if isinstance(features, torch.Tensor):
        corrected_features = corrected_features.cpu().numpy()
    
    return corrected_features

