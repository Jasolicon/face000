"""
对比不同模型配置的参数量
"""
import torch
from train_transformer.models import SimpleTransformerEncoder
from train_transformer.models_lightweight import LightweightTransformer, AngleConditionedMLP, ResidualMLP

def count_parameters(model):
    """计算模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

print("=" * 80)
print("模型参数量对比")
print("=" * 80)

# 1. 8层标准Transformer
print("\n1. 8层标准Transformer:")
print("   配置: d_model=768, nhead=8, num_layers=8, dim_feedforward=2048")
m1 = SimpleTransformerEncoder(
    d_model=768,
    nhead=8,
    num_layers=8,
    dim_feedforward=2048,
    use_angle_pe=True,
    use_angle_conditioning=True,
    angle_dim=5
)
p1_total, p1_trainable = count_parameters(m1)
print(f"   总参数量: {p1_total:,}")
print(f"   可训练参数: {p1_trainable:,}")
print(f"   模型大小: {p1_total * 4 / 1024 / 1024:.2f} MB")

# 2. 4层标准Transformer（之前默认）
print("\n2. 4层标准Transformer (之前默认):")
print("   配置: d_model=768, nhead=8, num_layers=4, dim_feedforward=2048")
m2 = SimpleTransformerEncoder(
    d_model=768,
    nhead=8,
    num_layers=4,
    dim_feedforward=2048,
    use_angle_pe=True,
    use_angle_conditioning=True,
    angle_dim=5
)
p2_total, p2_trainable = count_parameters(m2)
print(f"   总参数量: {p2_total:,}")
print(f"   可训练参数: {p2_trainable:,}")
print(f"   模型大小: {p2_total * 4 / 1024 / 1024:.2f} MB")

# 3. 轻量级Transformer（现在默认）
print("\n3. 轻量级Transformer (现在默认):")
print("   配置: d_model=768, nhead=4, num_layers=2, dim_feedforward=1024")
m3 = LightweightTransformer(
    d_model=768,
    nhead=4,
    num_layers=2,
    dim_feedforward=1024,
    use_angle_conditioning=True,
    angle_dim=5
)
p3_total, p3_trainable = count_parameters(m3)
print(f"   总参数量: {p3_total:,}")
print(f"   可训练参数: {p3_trainable:,}")
print(f"   模型大小: {p3_total * 4 / 1024 / 1024:.2f} MB")

# 4. MLP模型
print("\n4. MLP模型:")
print("   配置: input_dim=768, hidden_dims=[512, 512, 768], output_dim=768")
m4 = AngleConditionedMLP(
    input_dim=768,
    hidden_dims=[512, 512, 768],
    output_dim=768,
    use_angle_conditioning=True,
    angle_dim=5
)
p4_total, p4_trainable = count_parameters(m4)
print(f"   总参数量: {p4_total:,}")
print(f"   可训练参数: {p4_trainable:,}")
print(f"   模型大小: {p4_total * 4 / 1024 / 1024:.2f} MB")

# 5. 残差MLP
print("\n5. 残差MLP:")
print("   配置: input_dim=768, hidden_dim=512, num_layers=3, output_dim=768")
m5 = ResidualMLP(
    input_dim=768,
    hidden_dim=512,
    num_layers=3,
    output_dim=768,
    use_angle_conditioning=True,
    angle_dim=5
)
p5_total, p5_trainable = count_parameters(m5)
print(f"   总参数量: {p5_total:,}")
print(f"   可训练参数: {p5_trainable:,}")
print(f"   模型大小: {p5_total * 4 / 1024 / 1024:.2f} MB")

# 对比结果
print("\n" + "=" * 80)
print("对比结果:")
print("=" * 80)
print(f"\n相对于8层Transformer:")
print(f"  4层Transformer:     {p1_total/p2_total:.2f}x 倍 (减少 {(1-p2_total/p1_total)*100:.1f}%)")
print(f"  轻量级Transformer:  {p1_total/p3_total:.2f}x 倍 (减少 {(1-p3_total/p1_total)*100:.1f}%)")
print(f"  MLP模型:            {p1_total/p4_total:.2f}x 倍 (减少 {(1-p4_total/p1_total)*100:.1f}%)")
print(f"  残差MLP:             {p1_total/p5_total:.2f}x 倍 (减少 {(1-p5_total/p1_total)*100:.1f}%)")

print(f"\n相对于4层Transformer:")
print(f"  轻量级Transformer:  {p2_total/p3_total:.2f}x 倍 (减少 {(1-p3_total/p2_total)*100:.1f}%)")
print(f"  MLP模型:            {p2_total/p4_total:.2f}x 倍 (减少 {(1-p4_total/p2_total)*100:.1f}%)")
print(f"  残差MLP:             {p2_total/p5_total:.2f}x 倍 (减少 {(1-p5_total/p2_total)*100:.1f}%)")

print(f"\n相对于轻量级Transformer:")
print(f"  MLP模型:            {p3_total/p4_total:.2f}x 倍 (减少 {(1-p4_total/p3_total)*100:.1f}%)")
print(f"  残差MLP:             {p3_total/p5_total:.2f}x 倍 (减少 {(1-p5_total/p3_total)*100:.1f}%)")

print("\n" + "=" * 80)

