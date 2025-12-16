"""
计算每个人的每张图与正面图特征的相似度，并绘制相似度变化折线图
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_features_and_metadata(data_dir: str = 'train/datas/file'):
    """加载所有特征和元数据"""
    data_dir = Path(data_dir)
    
    logger.info(f"加载数据目录: {data_dir}")
    
    # 加载正面图数据
    front_features = np.load(data_dir / 'front_feature.npy')  # [N, 512]
    with open(data_dir / 'front_metadata.json', 'r', encoding='utf-8') as f:
        front_metadata = json.load(f)
    
    front_person_names = [meta['person_name'] for meta in front_metadata['metadata']]
    front_poses = np.array(front_metadata['poses'])  # [N, 3] (yaw, pitch, roll)
    front_file_paths = [meta['image_path'] for meta in front_metadata['metadata']]
    
    # 加载视频帧数据
    video_features = np.load(data_dir / 'video_feature.npy')  # [M, 512]
    with open(data_dir / 'video_metadata.json', 'r', encoding='utf-8') as f:
        video_metadata = json.load(f)
    
    video_person_names = [meta['person_name'] for meta in video_metadata['metadata']]
    video_poses = np.array(video_metadata['poses'])  # [M, 3]
    video_file_paths = [meta['image_path'] for meta in video_metadata['metadata']]
    
    logger.info(f"正面图数量: {len(front_features)}")
    logger.info(f"视频帧数量: {len(video_features)}")
    logger.info(f"总人数: {len(set(front_person_names))}")
    
    return {
        'front': {
            'features': front_features,
            'person_names': front_person_names,
            'poses': front_poses,
            'file_paths': front_file_paths
        },
        'video': {
            'features': video_features,
            'person_names': video_person_names,
            'poses': video_poses,
            'file_paths': video_file_paths
        }
    }


def calculate_similarity_to_frontal(person_data):
    """
    计算每个人的每张图与正面图特征的相似度
    
    Args:
        person_data: 包含该人所有数据的字典
    
    Returns:
        similarities: 相似度列表
        angles: 对应的角度列表（用于排序）
        indices: 对应的索引列表
    """
    frontal_feature = person_data['frontal_feature']  # [512]
    all_features = person_data['all_features']  # [N, 512]
    all_poses = person_data['all_poses']  # [N, 3]
    
    # 计算余弦相似度
    # cosine_similarity 需要 2D 数组，所以需要 reshape
    similarities = cosine_similarity(
        all_features,
        frontal_feature.reshape(1, -1)
    ).flatten()  # [N]
    
    # 获取 yaw 角度用于排序
    yaw_angles = all_poses[:, 0]  # [N]
    
    # 按 yaw 角度排序
    sorted_indices = np.argsort(yaw_angles)
    
    return similarities[sorted_indices], yaw_angles[sorted_indices], sorted_indices


def organize_data_by_person(data):
    """按人员组织数据"""
    person_to_data = defaultdict(lambda: {
        'frontal_features': [],
        'frontal_indices': [],
        'video_features': [],
        'video_poses': [],
        'video_indices': []
    })
    
    # 组织正面图数据
    for idx, (person_name, feature, pose) in enumerate(zip(
        data['front']['person_names'],
        data['front']['features'],
        data['front']['poses']
    )):
        person_to_data[person_name]['frontal_features'].append(feature)
        person_to_data[person_name]['frontal_indices'].append(idx)
    
    # 组织视频帧数据
    for idx, (person_name, feature, pose) in enumerate(zip(
        data['video']['person_names'],
        data['video']['features'],
        data['video']['poses']
    )):
        person_to_data[person_name]['video_features'].append(feature)
        person_to_data[person_name]['video_poses'].append(pose)
        person_to_data[person_name]['video_indices'].append(idx)
    
    # 转换为 numpy 数组
    person_data_organized = {}
    for person_name, person_data in person_to_data.items():
        if len(person_data['frontal_features']) == 0:
            logger.warning(f"{person_name}: 没有正面图数据，跳过")
            continue
        
        # 计算正面特征的平均值（如果有多个正面图）
        frontal_features = np.array(person_data['frontal_features'])  # [N_front, 512]
        frontal_feature = np.mean(frontal_features, axis=0)  # [512] - 平均正面特征
        
        # 合并所有特征（正面图 + 视频帧）
        video_features = np.array(person_data['video_features']) if len(person_data['video_features']) > 0 else np.empty((0, 512))
        all_features = np.vstack([frontal_features, video_features])  # [N_total, 512]
        
        # 合并所有姿态
        frontal_poses = np.array([data['front']['poses'][i] for i in person_data['frontal_indices']])
        video_poses = np.array(person_data['video_poses']) if len(person_data['video_poses']) > 0 else np.empty((0, 3))
        all_poses = np.vstack([frontal_poses, video_poses])  # [N_total, 3]
        
        person_data_organized[person_name] = {
            'frontal_feature': frontal_feature,
            'all_features': all_features,
            'all_poses': all_poses,
            'n_frontal': len(frontal_features),
            'n_video': len(video_features),
            'n_total': len(all_features)
        }
    
    return person_data_organized


def plot_similarity_by_person(person_data_organized, output_path='similarity_by_person.png', max_persons=None):
    """
    绘制每个人的相似度变化折线图（子图）
    
    Args:
        person_data_organized: 按人员组织的数据
        output_path: 输出图片路径
        max_persons: 最多显示的人数（如果为None，显示所有人）
    """
    logger.info("计算相似度并绘制图表...")
    
    # 选择要显示的人员
    person_names = sorted(person_data_organized.keys())
    if max_persons is not None:
        person_names = person_names[:max_persons]
    
    n_persons = len(person_names)
    
    # 计算子图布局（尽量接近正方形）
    n_cols = int(np.ceil(np.sqrt(n_persons)))
    n_rows = int(np.ceil(n_persons / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    
    # 如果只有一个人，axes 不是数组
    if n_persons == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 为每个人绘制子图
    for idx, person_name in enumerate(person_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        person_data = person_data_organized[person_name]
        
        # 计算相似度
        similarities, yaw_angles, sorted_indices = calculate_similarity_to_frontal(person_data)
        
        # 绘制折线图
        ax.plot(yaw_angles, similarities, 'o-', linewidth=1.5, markersize=3, alpha=0.7)
        ax.set_xlabel('Yaw角度 (°)', fontsize=9)
        ax.set_ylabel('相似度', fontsize=9)
        ax.set_title(f'{person_name}\n(正面:{person_data["n_frontal"]}, 视频:{person_data["n_video"]})', 
                     fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # 添加统计信息
        mean_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        ax.text(0.02, 0.98, f'平均: {mean_sim:.3f}\n最小: {min_sim:.3f}\n最大: {max_sim:.3f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 隐藏多余的子图
    for idx in range(n_persons, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('每个人的特征与正面图特征的相似度变化（按Yaw角度排序）', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"相似度图表已保存到: {output_path}")
    plt.close()


def print_similarity_statistics(person_data_organized):
    """打印相似度统计信息"""
    logger.info("\n" + "=" * 70)
    logger.info("相似度统计信息")
    logger.info("=" * 70)
    
    all_mean_similarities = []
    all_min_similarities = []
    all_max_similarities = []
    
    for person_name, person_data in sorted(person_data_organized.items()):
        similarities, yaw_angles, _ = calculate_similarity_to_frontal(person_data)
        
        mean_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        std_sim = np.std(similarities)
        
        all_mean_similarities.append(mean_sim)
        all_min_similarities.append(min_sim)
        all_max_similarities.append(max_sim)
        
        logger.info(f"\n{person_name}:")
        logger.info(f"  样本数: {person_data['n_total']} (正面:{person_data['n_frontal']}, 视频:{person_data['n_video']})")
        logger.info(f"  平均相似度: {mean_sim:.4f} ± {std_sim:.4f}")
        logger.info(f"  相似度范围: [{min_sim:.4f}, {max_sim:.4f}]")
        logger.info(f"  Yaw角度范围: [{np.min(yaw_angles):.1f}°, {np.max(yaw_angles):.1f}°]")
    
    logger.info(f"\n【整体统计】")
    logger.info(f"  平均相似度（所有人平均）: {np.mean(all_mean_similarities):.4f} ± {np.std(all_mean_similarities):.4f}")
    logger.info(f"  最小相似度（所有人平均）: {np.mean(all_min_similarities):.4f} ± {np.std(all_min_similarities):.4f}")
    logger.info(f"  最大相似度（所有人平均）: {np.mean(all_max_similarities):.4f} ± {np.std(all_max_similarities):.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='计算并可视化每个人的特征与正面图特征的相似度')
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                        help='数据目录')
    parser.add_argument('--output_path', type=str, default='similarity_by_person.png',
                        help='输出图片路径')
    parser.add_argument('--max_persons', type=int, default=None,
                        help='最多显示的人数（如果为None，显示所有人）')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("特征相似度可视化工具")
    logger.info("=" * 70)
    
    try:
        # 1. 加载数据
        data = load_features_and_metadata(args.data_dir)
        
        # 2. 按人员组织数据
        logger.info("\n按人员组织数据...")
        person_data_organized = organize_data_by_person(data)
        logger.info(f"共 {len(person_data_organized)} 个人的数据")
        
        # 3. 打印统计信息
        print_similarity_statistics(person_data_organized)
        
        # 4. 绘制相似度图表
        plot_similarity_by_person(
            person_data_organized,
            output_path=args.output_path,
            max_persons=args.max_persons
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("完成！")
        logger.info(f"图表已保存到: {args.output_path}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
