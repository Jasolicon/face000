"""
特征可视化脚本：分析正面特征和侧面特征的分布相似性
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
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.triplet.dataset_triplet import (
    TripletFaceDataset3D,
    triplet_collate_fn
)
from torch.utils.data import DataLoader, Subset
from train_transformer3D.triplet.models_simple_triplet import SimpleTripletNetwork
from train_transformer3D.triplet.models_3d_triplet import TransformerDecoderOnly3D_Triplet

# 设置matplotlib中文字体
import platform
import matplotlib.font_manager as fm

def setup_chinese_font():
    """设置matplotlib中文字体（改进版，确保中文正确显示）"""
    system = platform.system()
    
    # 根据操作系统选择中文字体
    if system == 'Windows':
        font_candidates = [
            'Microsoft YaHei',      # 微软雅黑（Windows默认）
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong',             # 仿宋
            'Microsoft JhengHei'    # 微软正黑体
        ]
    elif system == 'Darwin':  # macOS
        font_candidates = [
            'PingFang SC',
            'STHeiti',
            'Arial Unicode MS',
            'STSong',
            'STKaiti'
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Noto Serif CJK SC',
            'Source Han Sans CN',
            'DejaVu Sans'
        ]
    
    # 尝试设置字体
    font_set = False
    selected_font = None
    
    for font_name in font_candidates:
        try:
            # 先检查字体是否存在
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if font_name not in available_fonts:
                continue
            
            # 设置字体
            plt.rcParams['font.sans-serif'] = [font_name] + [f for f in plt.rcParams['font.sans-serif'] if f != font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 清除matplotlib字体缓存（如果需要）
            try:
                import matplotlib
                matplotlib.font_manager._rebuild()
            except:
                pass
            
            # 测试字体是否可用（通过创建测试图）
            test_fig = plt.figure(figsize=(1, 1))
            test_ax = test_fig.add_subplot(111)
            test_ax.text(0.5, 0.5, '测试中文', fontsize=10)
            plt.close(test_fig)
            
            font_set = True
            selected_font = font_name
            print(f"✓ 已设置 matplotlib 中文字体: {font_name}")
            break
        except Exception as e:
            continue
    
    # 如果预设字体都不可用，尝试查找系统可用的中文字体
    if not font_set:
        try:
            # 获取所有可用字体
            fonts = [f.name for f in fm.fontManager.ttflist]
            
            # 查找中文字体关键词
            chinese_keywords = [
                'hei', 'song', 'kai', 'fang', 'yahei', 'simhei', 'simsun',
                'wqy', 'noto', 'source', 'droid', 'uming', 'ukai', 'pingfang'
            ]
            
            chinese_fonts = [
                f for f in fonts 
                if any(keyword in f.lower() for keyword in chinese_keywords)
            ]
            
            if chinese_fonts:
                # 优先选择包含中文关键词的字体
                preferred_fonts = [
                    f for f in chinese_fonts 
                    if any(kw in f.lower() for kw in ['yahei', 'simhei', 'hei', 'noto', 'wqy', 'source'])
                ]
                selected_font = preferred_fonts[0] if preferred_fonts else chinese_fonts[0]
                
                plt.rcParams['font.sans-serif'] = [selected_font] + [f for f in plt.rcParams['font.sans-serif'] if f != selected_font]
                plt.rcParams['axes.unicode_minus'] = False
                font_set = True
                print(f"✓ 已设置 matplotlib 中文字体: {selected_font}")
            else:
                print("⚠️  警告: 未找到中文字体，中文可能显示为方块")
                print("   提示: 在 Windows 上通常已安装 Microsoft YaHei")
                print("   在 Linux 上可以安装: sudo apt-get install fonts-wqy-microhei")
                plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"⚠️  警告: 设置中文字体失败: {e}")
            plt.rcParams['axes.unicode_minus'] = False
    
    # 确保负号正常显示
    plt.rcParams['axes.unicode_minus'] = False
    
    # 强制设置字体属性（确保中文显示）
    if selected_font:
        try:
            # 设置字体属性
            prop = fm.FontProperties(fname=None, family=selected_font)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [selected_font]
        except:
            pass
    
    return font_set

# 初始化中文字体
try:
    from font_utils import setup_chinese_font_matplotlib
    result = setup_chinese_font_matplotlib()
    if not result:
        setup_chinese_font()
except ImportError:
    setup_chinese_font()
except Exception as e:
    print(f"⚠️  字体设置异常: {e}")
    setup_chinese_font()

# 设置seaborn样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


def extract_features(model, dataloader, device, model_type='simple'):
    """
    从数据集中提取特征（包括原始特征和模型生成的特征）
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        model_type: 模型类型 ('simple' 或 'transformer')
        
    Returns:
        side_features_orig: 原始侧面特征 [N, 512] (ground truth)
        front_features_orig: 原始正面特征 [N, 512] (ground truth)
        front_features_pred: 模型生成的正面特征 [N, 512] (模型输出)
        identity_features: 身份特征 [N, 512]
        poses: 姿势 [N, 3]
        person_names: 人员名称列表
    """
    model.eval()
    
    side_features_orig_list = []
    front_features_orig_list = []
    front_features_pred_list = []
    identity_features_list = []
    poses_list = []
    person_names_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='提取特征'):
            src = batch['src'].to(device)  # 原始侧面特征
            tgt = batch.get('tgt', None)  # 原始正面特征（ground truth）
            if tgt is not None:
                tgt = tgt.to(device)
            pose = batch['pose'].to(device)
            person_names = batch['person_name']
            
            # 模型生成的正面特征
            if model_type == 'simple':
                front_features_pred, identity_features, front_pose = model(
                    src=src,
                    pose=pose,
                    return_identity_features=True,
                    return_front_pose=True
                )
            else:  # transformer
                identity_features, residual = model(
                    src=src,
                    angles=pose,
                    pose=pose,
                    return_residual=False
                )
                front_features_pred = src + residual if residual is not None else src
            
            side_features_orig_list.append(src.cpu())
            front_features_pred_list.append(front_features_pred.cpu())
            identity_features_list.append(identity_features.cpu())
            poses_list.append(pose.cpu())
            person_names_list.extend(person_names)
            
            # 原始正面特征（如果有）
            if tgt is not None:
                front_features_orig_list.append(tgt.cpu())
            else:
                # 如果没有tgt，使用None占位
                front_features_orig_list.append(None)
    
    side_features_orig = torch.cat(side_features_orig_list, dim=0).numpy()
    front_features_pred = torch.cat(front_features_pred_list, dim=0).numpy()
    identity_features = torch.cat(identity_features_list, dim=0).numpy()
    poses = torch.cat(poses_list, dim=0).numpy()
    
    # 处理原始正面特征（可能为None）
    if front_features_orig_list[0] is not None:
        front_features_orig = torch.cat(front_features_orig_list, dim=0).numpy()
    else:
        front_features_orig = None
    
    return side_features_orig, front_features_orig, front_features_pred, identity_features, poses, person_names_list


def compute_feature_similarity(side_features, front_features):
    """
    计算特征相似性
    
    Args:
        side_features: 侧面特征 [N, 512]
        front_features: 正面特征 [N, 512]
        
    Returns:
        similarity_dict: 相似性字典
    """
    N, D = side_features.shape
    
    # 1. 逐维度相关性（Pearson相关系数）
    dimension_correlations = []
    for d in range(D):
        corr, p_value = pearsonr(side_features[:, d], front_features[:, d])
        dimension_correlations.append({
            'dimension': d,
            'correlation': corr,
            'p_value': p_value
        })
    
    dimension_correlations = sorted(dimension_correlations, 
                                   key=lambda x: abs(x['correlation']), 
                                   reverse=True)
    
    # 2. 逐维度统计相似性（均值、方差）
    side_mean = np.mean(side_features, axis=0)
    front_mean = np.mean(front_features, axis=0)
    side_std = np.std(side_features, axis=0)
    front_std = np.std(front_features, axis=0)
    
    # 均值差异
    mean_diff = np.abs(side_mean - front_mean)
    # 方差比率
    std_ratio = np.minimum(side_std, front_std) / (np.maximum(side_std, front_std) + 1e-8)
    
    # 3. 余弦相似度（逐维度）
    side_norm = side_features / (np.linalg.norm(side_features, axis=1, keepdims=True) + 1e-8)
    front_norm = front_features / (np.linalg.norm(front_features, axis=1, keepdims=True) + 1e-8)
    cosine_sim_per_dim = np.mean(side_norm * front_norm, axis=0)
    
    # 4. KL散度（逐维度，假设高斯分布）
    kl_divs = []
    for d in range(D):
        side_dist = stats.norm(side_mean[d], side_std[d] + 1e-8)
        front_dist = stats.norm(front_mean[d], front_std[d] + 1e-8)
        # 简化的KL散度估计
        kl = 0.5 * (np.log((front_std[d] + 1e-8) / (side_std[d] + 1e-8)) + 
                   (side_std[d]**2 + (side_mean[d] - front_mean[d])**2) / 
                   (front_std[d]**2 + 1e-8) - 1)
        kl_divs.append(kl)
    
    kl_divs = np.array(kl_divs)
    
    return {
        'dimension_correlations': dimension_correlations,
        'mean_diff': mean_diff,
        'std_ratio': std_ratio,
        'cosine_sim_per_dim': cosine_sim_per_dim,
        'kl_divs': kl_divs,
        'side_mean': side_mean,
        'front_mean': front_mean,
        'side_std': side_std,
        'front_std': front_std
    }


def visualize_features(
    side_features_orig, 
    front_features_orig,
    front_features_pred, 
    similarity_dict_orig,  # 原始特征之间的相似性
    similarity_dict_pred,  # 模型输出与原始侧面的相似性
    similarity_dict_quality=None,  # 模型输出与原始正面的相似性（质量评估）
    save_dir='train_transformer3D/triplet/feature_visualizations',
    top_k=50
):
    """
    可视化特征分布和相似性（支持三组特征对比）
    
    Args:
        side_features_orig: 原始侧面特征 [N, 512]
        front_features_orig: 原始正面特征 [N, 512] (ground truth，可能为None)
        front_features_pred: 模型生成的正面特征 [N, 512]
        similarity_dict_orig: 原始特征相似性字典（可能为None）
        similarity_dict_pred: 模型输出相似性字典
        similarity_dict_quality: 模型质量评估相似性字典（可能为None）
        save_dir: 保存目录
        top_k: 显示前k个最相似/最不相似的维度
    """
    os.makedirs(save_dir, exist_ok=True)
    
    N, D = side_features_orig.shape
    
    # 使用模型输出的相似性作为主要参考（因为总是存在）
    similarity_dict = similarity_dict_pred
    if similarity_dict_orig is not None:
        similarity_dict = similarity_dict_orig  # 优先使用原始特征相似性
    
    # 获取最相似和最不相似的维度
    top_correlations = similarity_dict['dimension_correlations'][:top_k]
    bottom_correlations = similarity_dict['dimension_correlations'][-top_k:]
    
    top_dims = [item['dimension'] for item in top_correlations]
    bottom_dims = [item['dimension'] for item in bottom_correlations]
    
    # ========== 图1: 相关性热力图（前k个维度） ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 最相似的k个维度相关性
    top_corr_matrix = np.zeros((top_k, top_k))
    for i, dim_i in enumerate(top_dims):
        for j, dim_j in enumerate(top_dims):
            if i == j:
                top_corr_matrix[i, j] = 1.0
            else:
                corr, _ = pearsonr(
                    np.concatenate([side_features_orig[:, dim_i], front_features_pred[:, dim_i]]),
                    np.concatenate([side_features_orig[:, dim_j], front_features_pred[:, dim_j]])
                )
                top_corr_matrix[i, j] = corr
    
    sns.heatmap(top_corr_matrix, ax=axes[0, 0], cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={'label': '相关系数'})
    axes[0, 0].set_title(f'最相似的{top_k}个维度相关性矩阵', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('维度索引')
    axes[0, 0].set_ylabel('维度索引')
    
    # 子图2: 相关性分布直方图
    all_correlations = [item['correlation'] for item in similarity_dict['dimension_correlations']]
    axes[0, 1].hist(all_correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(np.mean(all_correlations), color='red', linestyle='--', 
                      linewidth=2, label=f'均值: {np.mean(all_correlations):.3f}')
    axes[0, 1].set_xlabel('Pearson相关系数', fontsize=12)
    axes[0, 1].set_ylabel('频数', fontsize=12)
    axes[0, 1].set_title('所有维度相关性分布', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 最相似维度的特征值分布对比（三组特征）
    dim_idx = top_dims[0]
    axes[1, 0].hist(side_features_orig[:, dim_idx], bins=50, alpha=0.5, 
                   label='原始侧面', color='blue', density=True)
    if front_features_orig is not None:
        axes[1, 0].hist(front_features_orig[:, dim_idx], bins=50, alpha=0.5, 
                       label='原始正面', color='red', density=True)
    axes[1, 0].hist(front_features_pred[:, dim_idx], bins=50, alpha=0.5, 
                   label='模型生成正面', color='green', density=True)
    axes[1, 0].set_xlabel(f'特征值 (维度 {dim_idx})', fontsize=12)
    axes[1, 0].set_ylabel('密度', fontsize=12)
    corr_text = f'原始: {similarity_dict_orig["dimension_correlations"][0]["correlation"]:.3f}' if similarity_dict_orig else ''
    corr_text += f', 模型: {similarity_dict_pred["dimension_correlations"][0]["correlation"]:.3f}'
    axes[1, 0].set_title(f'最相似维度 {dim_idx} 的特征分布\n({corr_text})', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 最不相似维度的特征值分布对比（三组特征）
    dim_idx = bottom_dims[0]
    axes[1, 1].hist(side_features_orig[:, dim_idx], bins=50, alpha=0.5, 
                   label='原始侧面', color='blue', density=True)
    if front_features_orig is not None:
        axes[1, 1].hist(front_features_orig[:, dim_idx], bins=50, alpha=0.5, 
                       label='原始正面', color='red', density=True)
    axes[1, 1].hist(front_features_pred[:, dim_idx], bins=50, alpha=0.5, 
                   label='模型生成正面', color='green', density=True)
    axes[1, 1].set_xlabel(f'特征值 (维度 {dim_idx})', fontsize=12)
    axes[1, 1].set_ylabel('密度', fontsize=12)
    corr_text = f'原始: {similarity_dict_orig["dimension_correlations"][-1]["correlation"]:.3f}' if similarity_dict_orig else ''
    corr_text += f', 模型: {similarity_dict_pred["dimension_correlations"][-1]["correlation"]:.3f}'
    axes[1, 1].set_title(f'最不相似维度 {dim_idx} 的特征分布\n({corr_text})', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_correlation_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_correlation_analysis.png")
    
    # ========== 图2: 逐维度相似性分析 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 子图1: 相关性排序
    dims = [item['dimension'] for item in similarity_dict['dimension_correlations']]
    corrs = [item['correlation'] for item in similarity_dict['dimension_correlations']]
    
    axes[0, 0].plot(range(D), corrs, linewidth=1, alpha=0.7, color='blue')
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0, 0].fill_between(range(D), corrs, 0, alpha=0.3, color='blue')
    axes[0, 0].set_xlabel('维度索引（按相关性排序）', fontsize=12)
    axes[0, 0].set_ylabel('Pearson相关系数', fontsize=12)
    axes[0, 0].set_title('所有维度相关性排序', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 均值差异
    sorted_mean_diff = np.sort(similarity_dict['mean_diff'])[::-1]
    axes[0, 1].bar(range(min(50, D)), sorted_mean_diff[:50], alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('维度索引（按均值差异排序）', fontsize=12)
    axes[0, 1].set_ylabel('均值差异', fontsize=12)
    axes[0, 1].set_title('前50个维度均值差异', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 方差比率
    sorted_std_ratio = np.sort(similarity_dict['std_ratio'])
    axes[1, 0].bar(range(min(50, D)), sorted_std_ratio[:50], alpha=0.7, color='green')
    axes[1, 0].set_xlabel('维度索引（按方差比率排序）', fontsize=12)
    axes[1, 0].set_ylabel('方差比率', fontsize=12)
    axes[1, 0].set_title('前50个维度方差比率（越接近1越相似）', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 余弦相似度
    sorted_cosine = np.sort(similarity_dict['cosine_sim_per_dim'])[::-1]
    axes[1, 1].bar(range(min(50, D)), sorted_cosine[:50], alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('维度索引（按余弦相似度排序）', fontsize=12)
    axes[1, 1].set_ylabel('余弦相似度', fontsize=12)
    axes[1, 1].set_title('前50个维度余弦相似度', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_similarity_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_similarity_analysis.png")
    
    # ========== 图3: 特征分布对比（多个维度） ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 选择6个代表性维度：最相似的3个和最不相似的3个
    selected_dims = top_dims[:3] + bottom_dims[:3]
    
    for idx, dim in enumerate(selected_dims):
        ax = axes[idx]
        ax.hist(side_features_orig[:, dim], bins=50, alpha=0.5, 
               label='原始侧面', color='blue', density=True)
        if front_features_orig is not None:
            ax.hist(front_features_orig[:, dim], bins=50, alpha=0.5, 
                   label='原始正面', color='red', density=True)
        ax.hist(front_features_pred[:, dim], bins=50, alpha=0.5, 
               label='模型生成正面', color='green', density=True)
        
        corr_orig_item = next((item for item in similarity_dict_orig['dimension_correlations'] 
                               if item['dimension'] == dim), None) if similarity_dict_orig else None
        corr_pred_item = next(item for item in similarity_dict_pred['dimension_correlations'] 
                             if item['dimension'] == dim)
        
        title = f'维度 {dim}\n'
        if corr_orig_item:
            title += f'原始: {corr_orig_item["correlation"]:.3f}, '
        title += f'模型: {corr_pred_item["correlation"]:.3f}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('特征值', fontsize=10)
        ax.set_ylabel('密度', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distribution_comparison.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_distribution_comparison.png")
    
    # ========== 图4: 特征值散点图（最相似和最不相似维度） ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 最相似维度：原始侧面 vs 模型生成正面
    dim = top_dims[0]
    axes[0].scatter(side_features_orig[:, dim], front_features_pred[:, dim], 
                   alpha=0.5, s=10, color='green', label='模型输出')
    if front_features_orig is not None:
        axes[0].scatter(side_features_orig[:, dim], front_features_orig[:, dim], 
                       alpha=0.3, s=5, color='red', label='原始正面')
    axes[0].plot([side_features_orig[:, dim].min(), side_features_orig[:, dim].max()],
                 [side_features_orig[:, dim].min(), side_features_orig[:, dim].max()],
                 'r--', linewidth=2, label='y=x')
    corr_pred_item = next(item for item in similarity_dict_pred['dimension_correlations'] 
                         if item['dimension'] == dim)
    corr_orig_item = next((item for item in similarity_dict_orig['dimension_correlations'] 
                          if item['dimension'] == dim), None) if similarity_dict_orig else None
    title = f'最相似维度 {dim}\n模型输出: {corr_pred_item["correlation"]:.3f}'
    if corr_orig_item:
        title += f', 原始: {corr_orig_item["correlation"]:.3f}'
    axes[0].set_xlabel(f'原始侧面特征 (维度 {dim})', fontsize=12)
    axes[0].set_ylabel(f'正面特征 (维度 {dim})', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 最不相似维度：原始侧面 vs 模型生成正面
    dim = bottom_dims[0]
    axes[1].scatter(side_features_orig[:, dim], front_features_pred[:, dim], 
                   alpha=0.5, s=10, color='green', label='模型输出')
    if front_features_orig is not None:
        axes[1].scatter(side_features_orig[:, dim], front_features_orig[:, dim], 
                       alpha=0.3, s=5, color='red', label='原始正面')
    axes[1].plot([side_features_orig[:, dim].min(), side_features_orig[:, dim].max()],
                 [side_features_orig[:, dim].min(), side_features_orig[:, dim].max()],
                 'r--', linewidth=2, label='y=x')
    corr_pred_item = next(item for item in similarity_dict_pred['dimension_correlations'] 
                         if item['dimension'] == dim)
    corr_orig_item = next((item for item in similarity_dict_orig['dimension_correlations'] 
                          if item['dimension'] == dim), None) if similarity_dict_orig else None
    title = f'最不相似维度 {dim}\n模型输出: {corr_pred_item["correlation"]:.3f}'
    if corr_orig_item:
        title += f', 原始: {corr_orig_item["correlation"]:.3f}'
    axes[1].set_xlabel(f'原始侧面特征 (维度 {dim})', fontsize=12)
    axes[1].set_ylabel(f'正面特征 (维度 {dim})', fontsize=12)
    axes[1].set_title(title, fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_scatter_plots.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_scatter_plots.png")
    
    # ========== 图5: 特征热力图（所有维度） ==========
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始侧面特征热力图（按相关性排序）
    side_sorted = side_features_orig[:, dims]
    im1 = axes[0].imshow(side_sorted.T, aspect='auto', cmap='viridis', 
                        interpolation='nearest', vmin=-3, vmax=3)
    axes[0].set_xlabel('样本索引', fontsize=12)
    axes[0].set_ylabel('维度索引（按相关性排序）', fontsize=12)
    axes[0].set_title('原始侧面特征热力图', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='特征值')
    
    # 模型生成正面特征热力图（按相关性排序）
    front_sorted = front_features_pred[:, dims]
    im2 = axes[1].imshow(front_sorted.T, aspect='auto', cmap='viridis', 
                         interpolation='nearest', vmin=-3, vmax=3)
    axes[1].set_xlabel('样本索引', fontsize=12)
    axes[1].set_ylabel('维度索引（按相关性排序）', fontsize=12)
    axes[1].set_title('模型生成正面特征热力图', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='特征值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_heatmaps.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✓ 已保存: feature_heatmaps.png")
    
    # ========== 保存统计信息 ==========
    stats_file = os.path.join(save_dir, 'feature_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("特征分布相似性统计\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"总维度数: {D}\n")
        f.write(f"样本数: {N}\n\n")
        
        f.write("相关性统计:\n")
        corrs = [item['correlation'] for item in similarity_dict['dimension_correlations']]
        f.write(f"  平均相关系数: {np.mean(corrs):.4f}\n")
        f.write(f"  中位数相关系数: {np.median(corrs):.4f}\n")
        f.write(f"  标准差: {np.std(corrs):.4f}\n")
        f.write(f"  最大相关系数: {max(corrs):.4f} (维度 {dims[corrs.index(max(corrs))]})\n")
        f.write(f"  最小相关系数: {min(corrs):.4f} (维度 {dims[corrs.index(min(corrs))]})\n\n")
        
        if similarity_dict_orig is not None:
            corrs_orig = [item['correlation'] for item in similarity_dict_orig['dimension_correlations']]
            f.write("原始特征对比（原始侧面 vs 原始正面）:\n")
            f.write(f"  平均相关系数: {np.mean(corrs_orig):.4f}\n")
            f.write(f"  中位数相关系数: {np.median(corrs_orig):.4f}\n\n")
        
        corrs_pred = [item['correlation'] for item in similarity_dict_pred['dimension_correlations']]
        f.write("模型输出对比（原始侧面 vs 模型生成正面）:\n")
        f.write(f"  平均相关系数: {np.mean(corrs_pred):.4f}\n")
        f.write(f"  中位数相关系数: {np.median(corrs_pred):.4f}\n\n")
        
        if similarity_dict_quality is not None:
            corrs_quality = [item['correlation'] for item in similarity_dict_quality['dimension_correlations']]
            f.write("模型质量评估（模型生成正面 vs 原始正面）:\n")
            f.write(f"  平均相关系数: {np.mean(corrs_quality):.4f}\n")
            f.write(f"  中位数相关系数: {np.median(corrs_quality):.4f}\n\n")
        
        f.write(f"最相似的{top_k}个维度:\n")
        for i, item in enumerate(top_correlations[:10]):
            f.write(f"  {i+1}. 维度 {item['dimension']}: 相关系数 = {item['correlation']:.4f}\n")
        
        f.write(f"\n最不相似的{top_k}个维度:\n")
        for i, item in enumerate(bottom_correlations[:10]):
            f.write(f"  {i+1}. 维度 {item['dimension']}: 相关系数 = {item['correlation']:.4f}\n")
    
    print(f"✓ 已保存: feature_statistics.txt")


def main():
    parser = argparse.ArgumentParser(description='可视化正面特征和侧面特征的分布相似性')
    
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
    dataset = TripletFaceDataset3D(
        data_dir=args.data_dir,
        load_in_memory=True
    )
    
    # 限制样本数量
    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = list(range(min(args.num_samples, len(dataset))))
        from torch.utils.data import Subset
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
        model = SimpleTripletNetwork(
            image_dim=args.image_dim,
            pose_dim=3,
            hidden_dim=1024,
            num_layers=3
        ).to(device)
    else:
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
    
    print(f"✓ 提取完成: {len(side_features_orig)} 个样本")
    print(f"  原始侧面特征形状: {side_features_orig.shape}")
    if front_features_orig is not None:
        print(f"  原始正面特征形状: {front_features_orig.shape}")
    print(f"  模型生成正面特征形状: {front_features_pred.shape}")
    print(f"  身份特征形状: {identity_features.shape}")
    
    # 计算相似性（三组对比）
    print("\n计算特征相似性...")
    
    # 1. 原始特征之间的相似性（如果有原始正面特征）
    if front_features_orig is not None:
        similarity_dict_orig = compute_feature_similarity(side_features_orig, front_features_orig)
        print(f"  原始特征平均相关系数: {np.mean([item['correlation'] for item in similarity_dict_orig['dimension_correlations']]):.4f}")
    else:
        similarity_dict_orig = None
        print("  ⚠️  没有原始正面特征，跳过原始特征对比")
    
    # 2. 模型输出与原始侧面的相似性
    similarity_dict_pred = compute_feature_similarity(side_features_orig, front_features_pred)
    print(f"  模型输出平均相关系数: {np.mean([item['correlation'] for item in similarity_dict_pred['dimension_correlations']]):.4f}")
    
    # 3. 模型输出与原始正面的相似性（质量评估，如果有原始正面特征）
    if front_features_orig is not None:
        similarity_dict_quality = compute_feature_similarity(front_features_pred, front_features_orig)
        print(f"  模型质量平均相关系数: {np.mean([item['correlation'] for item in similarity_dict_quality['dimension_correlations']]):.4f}")
    else:
        similarity_dict_quality = None
    
    print(f"✓ 相似性计算完成")
    
    # 可视化
    print("\n生成可视化图片...")
    visualize_features(
        side_features_orig,
        front_features_orig if front_features_orig is not None else front_features_pred,  # 如果没有原始正面，使用模型输出作为占位符
        front_features_pred,
        similarity_dict_orig if similarity_dict_orig is not None else similarity_dict_pred,
        similarity_dict_pred,
        similarity_dict_quality,
        save_dir=args.save_dir,
        top_k=args.top_k
    )
    
    print(f"\n✓ 所有可视化图片已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()

