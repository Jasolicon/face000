"""
SENet在一维向量上的使用示例

演示如何将SENet应用于1D特征向量（如512维的InsightFace特征）
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.triplet.models_senet_triplet import SEBlock, DualBranchSENet

def example_1_basic_usage():
    """示例1：基础使用 - SEBlock处理1D向量"""
    print("=" * 70)
    print("示例1：SEBlock基础使用")
    print("=" * 70)
    
    # 创建SE Block（方案1：每个样本独立计算，推荐）
    se_block = SEBlock(channels=512, reduction=16, use_batch_stat=False)
    
    # 输入：1D特征向量 [batch, channels]
    batch_size = 4
    x = torch.randn(batch_size, 512)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输入范围: [{x.min():.4f}, {x.max():.4f}]")
    
    # 前向传播
    output = se_block(x)
    
    print(f"\n输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    
    # 获取权重（用于分析）
    with torch.no_grad():
        weights = se_block.fc(x)  # [batch, channels]
        print(f"\n权重形状: {weights.shape}")
        print(f"权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"权重均值: {weights.mean():.4f}")
    
    print("\n✓ 示例1完成\n")


def example_2_comparison():
    """示例2：对比两种方案"""
    print("=" * 70)
    print("示例2：对比两种方案")
    print("=" * 70)
    
    # 创建两个SE Block
    se_block_1 = SEBlock(channels=512, reduction=16, use_batch_stat=False)  # 方案1
    se_block_2 = SEBlock(channels=512, reduction=16, use_batch_stat=True)     # 方案2
    
    # 输入
    x = torch.randn(4, 512)
    
    # 前向传播
    output_1 = se_block_1(x)
    output_2 = se_block_2(x)
    
    # 获取权重
    with torch.no_grad():
        weights_1 = se_block_1.fc(x)  # [batch, channels] - 每个样本不同
        weights_2 = se_block_2.fc(x.mean(dim=0, keepdim=True)).expand(4, -1)  # [batch, channels] - 所有样本相同
    
    print(f"\n方案1（独立计算）:")
    print(f"  权重形状: {weights_1.shape}")
    print(f"  权重是否相同: {torch.allclose(weights_1[0], weights_1[1])}")
    print(f"  权重标准差: {weights_1.std():.4f}")
    
    print(f"\n方案2（Batch统计）:")
    print(f"  权重形状: {weights_2.shape}")
    print(f"  权重是否相同: {torch.allclose(weights_2[0], weights_2[1])}")
    print(f"  权重标准差: {weights_2.std():.4f}")
    
    print("\n✓ 示例2完成\n")


def example_3_visualize_weights():
    """示例3：可视化通道权重"""
    print("=" * 70)
    print("示例3：可视化通道权重")
    print("=" * 70)
    
    # 创建SE Block
    se_block = SEBlock(channels=512, reduction=16, use_batch_stat=False)
    
    # 输入：单个样本
    x = torch.randn(1, 512)
    
    # 获取权重
    with torch.no_grad():
        weights = se_block.fc(x).squeeze(0).numpy()  # [512]
    
    # 找出最重要的通道
    top_k = 10
    top_indices = np.argsort(weights)[-top_k:][::-1]
    
    print(f"\nTop {top_k} 重要通道:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. 通道 {idx}: 权重 = {weights[idx]:.4f}")
    
    # 可视化（如果matplotlib可用）
    try:
        plt.figure(figsize=(12, 4))
        plt.plot(weights, alpha=0.7, label='所有通道权重')
        plt.scatter(top_indices, weights[top_indices], color='red', s=50, 
                   label=f'Top {top_k} 重要通道', zorder=5)
        plt.xlabel('通道索引')
        plt.ylabel('权重值')
        plt.title('SENet通道权重分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = 'train_transformer3D/triplet/se_weights_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 权重可视化已保存到: {save_path}")
        plt.close()
    except Exception as e:
        print(f"\n⚠️  可视化失败: {e}")
    
    print("\n✓ 示例3完成\n")


def example_4_dual_branch():
    """示例4：双分支SENet使用"""
    print("=" * 70)
    print("示例4：双分支SENet使用")
    print("=" * 70)
    
    # 创建双分支SENet
    dual_branch = DualBranchSENet(
        feature_dim=512,
        reduction=16,
        fusion_alpha=0.7,
        learnable_fusion=True,
        use_batch_stat=False  # 使用方案1（推荐）
    )
    
    # 输入
    x = torch.randn(4, 512)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    front_features, identity_branch, pose_branch = dual_branch(x, return_branches=True)
    
    print(f"\n输出形状:")
    print(f"  正面特征: {front_features.shape}")
    print(f"  身份分支: {identity_branch.shape}")
    print(f"  姿态分支: {pose_branch.shape}")
    
    # 获取融合权重
    fusion_alpha = dual_branch.get_fusion_alpha()
    print(f"\n融合权重:")
    print(f"  α (身份权重): {fusion_alpha:.4f}")
    print(f"  1-α (姿态权重): {1 - fusion_alpha:.4f}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in dual_branch.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    print("\n✓ 示例4完成\n")


def example_5_identity_vs_pose():
    """示例5：分析身份分支和姿态分支的差异"""
    print("=" * 70)
    print("示例5：分析身份分支和姿态分支的差异")
    print("=" * 70)
    
    # 创建双分支SENet
    dual_branch = DualBranchSENet(
        feature_dim=512,
        reduction=16,
        fusion_alpha=0.7,
        learnable_fusion=True,
        use_batch_stat=False
    )
    
    # 输入
    x = torch.randn(4, 512)
    
    # 前向传播
    front_features, identity_branch, pose_branch = dual_branch(x, return_branches=True)
    
    # 计算与原始输入的相似度
    with torch.no_grad():
        # 身份分支应该与原始输入更相似（因为有残差连接）
        identity_sim = F.cosine_similarity(identity_branch, x, dim=1)
        pose_sim = F.cosine_similarity(pose_branch, x, dim=1)
        
        print(f"\n与原始输入的余弦相似度:")
        print(f"  身份分支: {identity_sim.mean():.4f} ± {identity_sim.std():.4f}")
        print(f"  姿态分支: {pose_sim.mean():.4f} ± {pose_sim.std():.4f}")
        
        # 计算两个分支的差异
        branch_diff = F.mse_loss(identity_branch, pose_branch)
        print(f"\n分支差异 (MSE): {branch_diff.item():.4f}")
    
    print("\n✓ 示例5完成\n")


if __name__ == "__main__":
    import torch.nn.functional as F
    
    print("\n" + "=" * 70)
    print("SENet在一维向量上的使用示例")
    print("=" * 70 + "\n")
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_comparison()
    example_3_visualize_weights()
    example_4_dual_branch()
    example_5_identity_vs_pose()
    
    print("=" * 70)
    print("所有示例完成！")
    print("=" * 70)

