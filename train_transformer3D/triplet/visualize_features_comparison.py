"""
特征可视化脚本（增强版）：比较原始特征和模型生成特征
包括：
1. 原始侧面 vs 原始正面（ground truth对比）
2. 原始侧面 vs 模型生成的正面（模型输出对比）
3. 模型生成的正面 vs 原始正面（模型质量评估）
"""
import os
import sys
from pathlib import Path

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '5'

# 尝试导入 setup_mirrors（如果存在）
try:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from setup_mirrors import setup_all_mirrors
    setup_all_mirrors()
except ImportError:
    pass

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse

# 导入原有的可视化函数
from visualize_features import (
    extract_features,
    compute_feature_similarity
)

# 设置matplotlib中文字体
try:
    from font_utils import setup_chinese_font_matplotlib
    setup_chinese_font_matplotlib()
except ImportError:
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                break
            except:
                continue
    except:
        plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def visualize_comparison(
    side_features_orig,      # 原始侧面特征 [N, 512]
    front_features_orig,      # 原始正面特征 [N, 512] (ground truth)
    front_features_pred,      # 模型生成的正面特征 [N, 512]
    similarity_dict_orig,     # 原始特征之间的相似性
    similarity_dict_pred,     # 模型输出与原始侧面的相似性
    similarity_dict_quality,  # 模型输出与原始正面的相似性（质量评估）
    save_dir='train_transformer3D/triplet/feature_visualizations',
    top_k=50
):
    """
    可视化三组特征的对比
    
    Args:
        side_features_orig: 原始侧面特征 [N, 512]
        front_features_orig: 原始正面特征 [N, 512]
        front_features_pred: 模型生成的正面特征 [N, 512]
        similarity_dict_orig: 原始特征相似性字典
        similarity_dict_pred: 模型输出相似性字典
        similarity_dict_quality: 模型质量评估相似性字典
        save_dir: 保存目录
        top_k: 显示前k个最相似/最不相似的维度
    """
    os.makedirs(save_dir, exist_ok=True)
    
    N, D = side_features_orig.shape
    
    # ========== 图1: 三组特征相关性对比 ==========
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 提取相关性数据
    corrs_orig = [item['correlation'] for item in similarity_dict_orig['dimension_correlations']]
    corrs_pred = [item['correlation'] for item in similarity_dict_pred['dimension_correlations']]
    corrs_quality = [item['correlation'] for item in similarity_dict_quality['dimension_correlations']]
    
    # 子图1: 原始特征相关性分布
    axes[0, 0].hist(corrs_orig, bins=50, alpha=0.7, color='blue', edgecolor='black', label='原始特征')
    axes[0, 0].axvline(np.mean(corrs_orig), color='red', linestyle='--', linewidth=2,
                      label=f'均值: {np.mean(corrs_orig):.3f}')
    axes[0, 0].set_xlabel('Pearson相关系数', fontsize=12)
    axes[0, 0].set_ylabel('频数', fontsize=12)
    axes[0, 0].set_title('原始侧面 vs 原始正面\n相关性分布', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 模型输出相关性分布
    axes[0, 1].hist(corrs_pred, bins=50, alpha=0.7, color='green', edgecolor='black', label='模型输出')
    axes[0, 1].axvline(np.mean(corrs_pred), color='red', linestyle='--', linewidth=2,
                      label=f'均值: {np.mean(corrs_pred):.3f}')
    axes[0, 1].set_xlabel('Pearson相关系数', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].set_title('原始侧面 vs 模型生成正面\n相关性分布', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 模型质量相关性分布
    axes[0, 2].hist(corrs_quality, bins=50, alpha=0.7, color='purple', edgecolor='black', label='模型质量')
    axes[0, 2].axvline(np.mean(corrs_quality), color='red', linestyle='--', linewidth=2,
                      label=f'均值: {np.mean(corrs_quality):.3f}')
    axes[0, 2].set_xlabel('Pearson相关系数', fontsize=12)
    axes[0, 2].set_ylabel('频数', fontsize=12)
    axes[0, 2].set_title('模型生成正面 vs 原始正面\n相关性分布（质量评估）', fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 子图4: 相关性对比（折线图）
    dims_orig = [item['dimension'] for item in similarity_dict_orig['dimension_correlations']]
    dims_pred = [item['dimension'] for item in similarity_dict_pred['dimension_correlations']]
    dims_quality = [item['dimension'] for item in similarity_dict_quality['dimension_correlations']]
    
    axes[1, 0].plot(range(D), corrs_orig, linewidth=1, alpha=0.7, color='blue', label='原始特征')
    axes[1, 0].plot(range(D), corrs_pred, linewidth=1, alpha=0.7, color='green', label='模型输出')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('维度索引（按原始特征相关性排序）', fontsize=12)
    axes[1, 0].set_ylabel('Pearson相关系数', fontsize=12)
    axes[1, 0].set_title('相关性对比', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图5: 相关性差异（模型输出 - 原始特征）
    corrs_diff = np.array(corrs_pred) - np.array(corrs_orig)
    axes[1, 1].plot(range(D), corrs_diff, linewidth=1, alpha=0.7, color='orange')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 1].fill_between(range(D), corrs_diff, 0, alpha=0.3, color='orange')
    axes[1, 1].set_xlabel('维度索引', fontsize=12)
    axes[1, 1].set_ylabel('相关性差异', fontsize=12)
    axes[1, 1].set_title('模型输出相关性 - 原始特征相关性\n(正值表示模型增强了相似性)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 子图6: 模型质量相关性排序
    axes[1, 2].plot(range(D), corrs_quality, linewidth=1, alpha=0.7, color='purple')
    axes[1, 2].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1, 2].fill_between(range(D), corrs_quality, 0, alpha=0.3, color='purple')
    axes[1, 2].set_xlabel('维度索引（按质量相关性排序）', fontsize=12)
    axes[1, 2].set_ylabel('Pearson相关系数', fontsize=12)
    axes[1, 2].set_title('模型质量相关性排序\n(越高表示模型生成越接近真实正面)', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_comparison_correlation.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_comparison_correlation.png")
    
    # ========== 图2: 特征分布对比（代表性维度） ==========
    # 选择最相似的3个和最不相似的3个维度
    top_dims_orig = [item['dimension'] for item in similarity_dict_orig['dimension_correlations'][:3]]
    bottom_dims_orig = [item['dimension'] for item in similarity_dict_orig['dimension_correlations'][-3:]]
    selected_dims = top_dims_orig + bottom_dims_orig
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, dim in enumerate(selected_dims):
        ax = axes[idx]
        
        # 绘制三个分布
        ax.hist(side_features_orig[:, dim], bins=50, alpha=0.5, 
               label='原始侧面', color='blue', density=True)
        ax.hist(front_features_orig[:, dim], bins=50, alpha=0.5, 
               label='原始正面', color='red', density=True)
        ax.hist(front_features_pred[:, dim], bins=50, alpha=0.5, 
               label='模型生成正面', color='green', density=True)
        
        # 获取相关性信息
        corr_orig_item = next(item for item in similarity_dict_orig['dimension_correlations'] 
                             if item['dimension'] == dim)
        corr_pred_item = next(item for item in similarity_dict_pred['dimension_correlations'] 
                             if item['dimension'] == dim)
        corr_quality_item = next(item for item in similarity_dict_quality['dimension_correlations'] 
                               if item['dimension'] == dim)
        
        title = f'维度 {dim}\n原始: {corr_orig_item["correlation"]:.3f}, '
        title += f'模型输出: {corr_pred_item["correlation"]:.3f}, '
        title += f'质量: {corr_quality_item["correlation"]:.3f}'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('特征值', fontsize=10)
        ax.set_ylabel('密度', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_comparison_distribution.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_comparison_distribution.png")
    
    # ========== 图3: 散点图对比 ==========
    # 选择最相似和最不相似的维度
    top_dim = similarity_dict_orig['dimension_correlations'][0]['dimension']
    bottom_dim = similarity_dict_orig['dimension_correlations'][-1]['dimension']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 最相似维度：原始侧面 vs 原始正面
    axes[0, 0].scatter(side_features_orig[:, top_dim], front_features_orig[:, top_dim], 
                      alpha=0.5, s=10, color='blue')
    axes[0, 0].plot([side_features_orig[:, top_dim].min(), side_features_orig[:, top_dim].max()],
                   [side_features_orig[:, top_dim].min(), side_features_orig[:, top_dim].max()],
                   'r--', linewidth=2, label='y=x')
    corr_orig = similarity_dict_orig['dimension_correlations'][0]['correlation']
    axes[0, 0].set_xlabel(f'原始侧面特征 (维度 {top_dim})', fontsize=12)
    axes[0, 0].set_ylabel(f'原始正面特征 (维度 {top_dim})', fontsize=12)
    axes[0, 0].set_title(f'最相似维度 {top_dim}\n原始特征对比 (相关系数: {corr_orig:.3f})', 
                        fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 最相似维度：原始侧面 vs 模型生成正面
    corr_pred_item = next(item for item in similarity_dict_pred['dimension_correlations'] 
                         if item['dimension'] == top_dim)
    axes[0, 1].scatter(side_features_orig[:, top_dim], front_features_pred[:, top_dim], 
                      alpha=0.5, s=10, color='green')
    axes[0, 1].plot([side_features_orig[:, top_dim].min(), side_features_orig[:, top_dim].max()],
                   [side_features_orig[:, top_dim].min(), side_features_orig[:, top_dim].max()],
                   'r--', linewidth=2, label='y=x')
    axes[0, 1].set_xlabel(f'原始侧面特征 (维度 {top_dim})', fontsize=12)
    axes[0, 1].set_ylabel(f'模型生成正面特征 (维度 {top_dim})', fontsize=12)
    axes[0, 1].set_title(f'最相似维度 {top_dim}\n模型输出对比 (相关系数: {corr_pred_item["correlation"]:.3f})', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 最相似维度：模型生成正面 vs 原始正面
    corr_quality_item = next(item for item in similarity_dict_quality['dimension_correlations'] 
                            if item['dimension'] == top_dim)
    axes[0, 2].scatter(front_features_pred[:, top_dim], front_features_orig[:, top_dim], 
                      alpha=0.5, s=10, color='purple')
    axes[0, 2].plot([front_features_pred[:, top_dim].min(), front_features_pred[:, top_dim].max()],
                   [front_features_pred[:, top_dim].min(), front_features_pred[:, top_dim].max()],
                   'r--', linewidth=2, label='y=x')
    axes[0, 2].set_xlabel(f'模型生成正面特征 (维度 {top_dim})', fontsize=12)
    axes[0, 2].set_ylabel(f'原始正面特征 (维度 {top_dim})', fontsize=12)
    axes[0, 2].set_title(f'最相似维度 {top_dim}\n质量评估 (相关系数: {corr_quality_item["correlation"]:.3f})', 
                        fontsize=14, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 最不相似维度：重复上述三个对比
    corr_orig_bottom = next(item for item in similarity_dict_orig['dimension_correlations'] 
                           if item['dimension'] == bottom_dim)['correlation']
    corr_pred_bottom = next(item for item in similarity_dict_pred['dimension_correlations'] 
                           if item['dimension'] == bottom_dim)['correlation']
    corr_quality_bottom = next(item for item in similarity_dict_quality['dimension_correlations'] 
                              if item['dimension'] == bottom_dim)['correlation']
    
    axes[1, 0].scatter(side_features_orig[:, bottom_dim], front_features_orig[:, bottom_dim], 
                      alpha=0.5, s=10, color='blue')
    axes[1, 0].plot([side_features_orig[:, bottom_dim].min(), side_features_orig[:, bottom_dim].max()],
                   [side_features_orig[:, bottom_dim].min(), side_features_orig[:, bottom_dim].max()],
                   'r--', linewidth=2, label='y=x')
    axes[1, 0].set_xlabel(f'原始侧面特征 (维度 {bottom_dim})', fontsize=12)
    axes[1, 0].set_ylabel(f'原始正面特征 (维度 {bottom_dim})', fontsize=12)
    axes[1, 0].set_title(f'最不相似维度 {bottom_dim}\n原始特征对比 (相关系数: {corr_orig_bottom:.3f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(side_features_orig[:, bottom_dim], front_features_pred[:, bottom_dim], 
                      alpha=0.5, s=10, color='green')
    axes[1, 1].plot([side_features_orig[:, bottom_dim].min(), side_features_orig[:, bottom_dim].max()],
                   [side_features_orig[:, bottom_dim].min(), side_features_orig[:, bottom_dim].max()],
                   'r--', linewidth=2, label='y=x')
    axes[1, 1].set_xlabel(f'原始侧面特征 (维度 {bottom_dim})', fontsize=12)
    axes[1, 1].set_ylabel(f'模型生成正面特征 (维度 {bottom_dim})', fontsize=12)
    axes[1, 1].set_title(f'最不相似维度 {bottom_dim}\n模型输出对比 (相关系数: {corr_pred_bottom:.3f})', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].scatter(front_features_pred[:, bottom_dim], front_features_orig[:, bottom_dim], 
                      alpha=0.5, s=10, color='purple')
    axes[1, 2].plot([front_features_pred[:, bottom_dim].min(), front_features_pred[:, bottom_dim].max()],
                   [front_features_pred[:, bottom_dim].min(), front_features_pred[:, bottom_dim].max()],
                   'r--', linewidth=2, label='y=x')
    axes[1, 2].set_xlabel(f'模型生成正面特征 (维度 {bottom_dim})', fontsize=12)
    axes[1, 2].set_ylabel(f'原始正面特征 (维度 {bottom_dim})', fontsize=12)
    axes[1, 2].set_title(f'最不相似维度 {bottom_dim}\n质量评估 (相关系数: {corr_quality_bottom:.3f})', 
                        fontsize=14, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_comparison_scatter.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_comparison_scatter.png")
    
    # ========== 保存统计信息 ==========
    stats_file = os.path.join(save_dir, 'feature_comparison_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("特征对比统计信息\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"总维度数: {D}\n")
        f.write(f"样本数: {N}\n\n")
        
        f.write("1. 原始特征对比（原始侧面 vs 原始正面）:\n")
        f.write(f"  平均相关系数: {np.mean(corrs_orig):.4f}\n")
        f.write(f"  中位数相关系数: {np.median(corrs_orig):.4f}\n")
        f.write(f"  标准差: {np.std(corrs_orig):.4f}\n\n")
        
        f.write("2. 模型输出对比（原始侧面 vs 模型生成正面）:\n")
        f.write(f"  平均相关系数: {np.mean(corrs_pred):.4f}\n")
        f.write(f"  中位数相关系数: {np.median(corrs_pred):.4f}\n")
        f.write(f"  标准差: {np.std(corrs_pred):.4f}\n\n")
        
        f.write("3. 模型质量评估（模型生成正面 vs 原始正面）:\n")
        f.write(f"  平均相关系数: {np.mean(corrs_quality):.4f}\n")
        f.write(f"  中位数相关系数: {np.median(corrs_quality):.4f}\n")
        f.write(f"  标准差: {np.std(corrs_quality):.4f}\n\n")
        
        f.write("4. 模型改进分析:\n")
        corrs_improvement = np.array(corrs_pred) - np.array(corrs_orig)
        f.write(f"  平均相关性改进: {np.mean(corrs_improvement):.4f}\n")
        f.write(f"  改进的维度数: {np.sum(corrs_improvement > 0)} / {D}\n")
        f.write(f"  改进比例: {np.sum(corrs_improvement > 0) / D * 100:.2f}%\n\n")
        
        f.write("5. 模型质量分析:\n")
        f.write(f"  高质量维度数（质量相关性 > 0.8）: {np.sum(np.array(corrs_quality) > 0.8)} / {D}\n")
        f.write(f"  高质量比例: {np.sum(np.array(corrs_quality) > 0.8) / D * 100:.2f}%\n")
    
    print(f"✓ 已保存: feature_comparison_statistics.txt")


def main():
    parser = argparse.ArgumentParser(description='可视化原始特征和模型生成特征的对比')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='分析的样本数量')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--model_type', type=str, default='simple',
                       choices=['simple', 'transformer'],
                       help='模型类型')
    parser.add_argument('--image_dim', type=int, default=512,
                       help='图像特征维度')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, 
                       default='train_transformer3D/triplet/feature_visualizations',
                       help='保存目录')
    parser.add_argument('--top_k', type=int, default=50,
                       help='显示前k个最相似/最不相似的维度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("加载数据...")
    from train_transformer3D.triplet.dataset_triplet import (
        TripletFaceDataset3D,
        triplet_collate_fn
    )
    from torch.utils.data import DataLoader, Subset
    
    dataset = TripletFaceDataset3D(
        data_dir=args.data_dir,
        load_in_memory=True
    )
    
    # 限制样本数量
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = list(range(min(args.num_samples, len(dataset))))
        dataset = Subset(dataset, indices)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=triplet_collate_fn,
        pin_memory=True
    )
    
    # 创建模型
    print(f"加载模型: {args.model_path}")
    if args.model_type == 'simple':
        from train_transformer3D.triplet.models_simple_triplet import SimpleTripletNetwork
        model = SimpleTripletNetwork(
            image_dim=args.image_dim,
            pose_dim=3,
            hidden_dim=1024,
            num_layers=3
        ).to(device)
    else:
        from train_transformer3D.triplet.models_3d_triplet import TransformerDecoderOnly3D_Triplet
        model = TransformerDecoderOnly3D_Triplet(
            d_model=args.image_dim,
            nhead=8,
            num_layers=4
        ).to(device)
    
    # 加载权重（PyTorch 2.6+需要设置weights_only=False）
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("✓ 模型加载完成")
    
    # 提取特征
    print("\n提取特征...")
    side_features_orig, front_features_orig, front_features_pred, identity_features, poses, person_names = extract_features(
        model, dataloader, device, args.model_type
    )
    
    if front_features_orig is None:
        print("⚠️  警告: 数据集中没有原始正面特征（tgt），无法进行完整对比")
        print("   将只比较原始侧面和模型生成的正面特征")
        front_features_orig = front_features_pred  # 使用模型输出作为占位符
    
    print(f"✓ 提取完成: {len(side_features_orig)} 个样本")
    print(f"  原始侧面特征形状: {side_features_orig.shape}")
    print(f"  原始正面特征形状: {front_features_orig.shape}")
    print(f"  模型生成正面特征形状: {front_features_pred.shape}")
    
    # 计算相似性
    print("\n计算特征相似性...")
    similarity_dict_orig = compute_feature_similarity(side_features_orig, front_features_orig)
    similarity_dict_pred = compute_feature_similarity(side_features_orig, front_features_pred)
    similarity_dict_quality = compute_feature_similarity(front_features_pred, front_features_orig)
    
    print(f"✓ 相似性计算完成")
    print(f"  原始特征平均相关系数: {np.mean([item['correlation'] for item in similarity_dict_orig['dimension_correlations']]):.4f}")
    print(f"  模型输出平均相关系数: {np.mean([item['correlation'] for item in similarity_dict_pred['dimension_correlations']]):.4f}")
    print(f"  模型质量平均相关系数: {np.mean([item['correlation'] for item in similarity_dict_quality['dimension_correlations']]):.4f}")
    
    # 可视化
    print("\n生成对比可视化图片...")
    visualize_comparison(
        side_features_orig,
        front_features_orig,
        front_features_pred,
        similarity_dict_orig,
        similarity_dict_pred,
        similarity_dict_quality,
        save_dir=args.save_dir,
        top_k=args.top_k
    )
    
    print(f"\n✓ 所有对比可视化图片已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()

