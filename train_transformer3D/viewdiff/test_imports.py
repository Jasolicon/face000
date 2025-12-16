"""
测试ViewDiff模块的导入和基本功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

print("=" * 70)
print("测试ViewDiff模块导入...")
print("=" * 70)

try:
    # 测试导入
    from train_transformer3D.viewdiff import (
        LoRALinearLayer,
        PoseConditionedLoRAAttention,
        LightweightFaceProjectionLayer,
        CrossViewAttention,
        PriorPreservationLoss,
        EnhancedTransformerDecoderOnly3D,
        EnhancedTransformerWithPrior
    )
    print("✓ 所有模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("测试LoRA线性层...")
print("=" * 70)

try:
    lora_layer = LoRALinearLayer(in_features=512, out_features=512, rank=4, alpha=1.0)
    x = torch.randn(4, 512)
    out = lora_layer(x)
    print(f"✓ LoRA层测试通过: 输入形状 {x.shape} -> 输出形状 {out.shape}")
except Exception as e:
    print(f"✗ LoRA层测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试姿态条件LoRA注意力...")
print("=" * 70)

try:
    pose_lora_attn = PoseConditionedLoRAAttention(
        feature_dim=512,
        pose_dim=3,
        num_heads=8,
        rank=4,
        alpha=1.0
    )
    x = torch.randn(4, 512)
    pose = torch.randn(4, 3)
    out = pose_lora_attn(x, pose)
    print(f"✓ 姿态LoRA注意力测试通过: 输入形状 {x.shape} -> 输出形状 {out.shape}")
except Exception as e:
    print(f"✗ 姿态LoRA注意力测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试增强版Transformer模型...")
print("=" * 70)

try:
    model = EnhancedTransformerDecoderOnly3D(
        d_model=512,
        nhead=8,
        num_layers=2,  # 减少层数以加快测试
        use_lora_attention=True,
        use_projection_layer=True,
        use_cross_view=False
    )
    
    # 测试前向传播
    src = torch.randn(2, 512)
    angles = torch.randn(2, 3)
    keypoints_3d = torch.randn(2, 5, 3)
    pose = torch.randn(2, 3)
    
    output = model(
        src=src,
        angles=angles,
        keypoints_3d=keypoints_3d,
        pose=pose,
        return_residual=True
    )
    
    print(f"✓ 增强版模型测试通过:")
    print(f"  输入形状: {src.shape}")
    print(f"  输出形状: {output.shape}")
    
    # 测试参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in model.get_lora_parameters())
    
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  LoRA参数: {lora_params:,}")
    print(f"  LoRA参数占比: {lora_params/trainable_params*100:.2f}%")
    
except Exception as e:
    print(f"✗ 增强版模型测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("所有测试完成！")
print("=" * 70)

