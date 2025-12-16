"""
基于姿态信息的特征聚类可视化工具
根据yaw, pitch, roll角度对特征进行聚类和可视化
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
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
    front_features = np.load(data_dir / 'front_feature.npy')
    with open(data_dir / 'front_metadata.json', 'r', encoding='utf-8') as f:
        front_metadata = json.load(f)
    
    front_person_names = [meta['person_name'] for meta in front_metadata['metadata']]
    front_poses = np.array(front_metadata['poses'])  # [N, 3] (yaw, pitch, roll)
    
    # 加载视频帧数据
    video_features = np.load(data_dir / 'video_feature.npy')
    with open(data_dir / 'video_metadata.json', 'r', encoding='utf-8') as f:
        video_metadata = json.load(f)
    
    video_person_names = [meta['person_name'] for meta in video_metadata['metadata']]
    video_poses = np.array(video_metadata['poses'])  # [M, 3]
    video_file_paths = [meta['image_path'] for meta in video_metadata['metadata']]
    
    # 合并所有数据
    all_features = np.vstack([front_features, video_features])
    all_labels = front_person_names + video_person_names
    all_poses = np.vstack([front_poses, video_poses])
    all_file_paths = [meta['image_path'] for meta in front_metadata['metadata']] + video_file_paths
    all_types = ['face'] * len(front_features) + ['video'] * len(video_features)
    
    logger.info(f"总特征数量: {len(all_features)}")
    logger.info(f"总人数: {len(set(all_labels))}")
    logger.info(f"姿态范围: yaw=[{np.min(all_poses[:, 0]):.1f}°, {np.max(all_poses[:, 0]):.1f}°], "
                f"pitch=[{np.min(all_poses[:, 1]):.1f}°, {np.max(all_poses[:, 1]):.1f}°], "
                f"roll=[{np.min(all_poses[:, 2]):.1f}°, {np.max(all_poses[:, 2]):.1f}°]")
    
    return all_features, all_labels, all_poses, all_file_paths, all_types


def classify_by_pose(poses, yaw_threshold=15, pitch_threshold=15, roll_threshold=10):
    """
    根据姿态角度对样本进行分类
    
    Args:
        poses: 姿态角度 [N, 3] (yaw, pitch, roll)
        yaw_threshold: yaw角度阈值（度）
        pitch_threshold: pitch角度阈值（度）
        roll_threshold: roll角度阈值（度）
    
    Returns:
        pose_labels: 姿态标签列表 [N]
        pose_categories: 姿态类别字典
    """
    yaw = poses[:, 0]
    pitch = poses[:, 1]
    roll = poses[:, 2]
    
    pose_labels = []
    pose_categories = {
        '正面': {'yaw_range': (-yaw_threshold, yaw_threshold), 
                'pitch_range': (-pitch_threshold, pitch_threshold),
                'roll_range': (-roll_threshold, roll_threshold)},
        '左转': {'yaw_range': (-90, -yaw_threshold)},
        '右转': {'yaw_range': (yaw_threshold, 90)},
        '上仰': {'pitch_range': (pitch_threshold, 90)},
        '下俯': {'pitch_range': (-90, -pitch_threshold)},
        '左倾斜': {'roll_range': (-90, -roll_threshold)},
        '右倾斜': {'roll_range': (roll_threshold, 90)},
    }
    
    for i in range(len(poses)):
        y, p, r = yaw[i], pitch[i], roll[i]
        
        # 判断主要姿态
        if (abs(y) < yaw_threshold and abs(p) < pitch_threshold and abs(r) < roll_threshold):
            label = '正面'
        elif y < -yaw_threshold:
            if abs(p) > pitch_threshold:
                label = f'左转{"上仰" if p > 0 else "下俯"}'
            else:
                label = '左转'
        elif y > yaw_threshold:
            if abs(p) > pitch_threshold:
                label = f'右转{"上仰" if p > 0 else "下俯"}'
            else:
                label = '右转'
        elif p > pitch_threshold:
            label = '上仰'
        elif p < -pitch_threshold:
            label = '下俯'
        elif r < -roll_threshold:
            label = '左倾斜'
        elif r > roll_threshold:
            label = '右倾斜'
        else:
            label = '其他'
        
        pose_labels.append(label)
    
    return pose_labels, pose_categories


def cluster_by_pose_groups(features, pose_labels, n_clusters_per_group=3):
    """
    对每个姿态组内的特征进行聚类
    
    Args:
        features: 特征矩阵 [N, feature_dim]
        pose_labels: 姿态标签 [N]
        n_clusters_per_group: 每个姿态组内的聚类数量
    
    Returns:
        cluster_labels: 聚类标签 [N]
        group_cluster_info: 每个组的聚类信息
    """
    unique_poses = sorted(set(pose_labels))
    cluster_labels = np.zeros(len(features), dtype=int)
    group_cluster_info = {}
    
    current_cluster_id = 0
    
    for pose_group in unique_poses:
        mask = np.array([pl == pose_group for pl in pose_labels])
        group_features = features[mask]
        
        if len(group_features) < n_clusters_per_group:
            # 如果样本太少，不聚类
            n_clusters = 1
        else:
            n_clusters = min(n_clusters_per_group, len(group_features))
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            group_clusters = kmeans.fit_predict(group_features)
            group_clusters = group_clusters + current_cluster_id
        else:
            group_clusters = np.full(len(group_features), current_cluster_id)
        
        cluster_labels[mask] = group_clusters
        
        group_cluster_info[pose_group] = {
            'n_samples': len(group_features),
            'n_clusters': n_clusters,
            'cluster_ids': list(range(current_cluster_id, current_cluster_id + n_clusters))
        }
        
        current_cluster_id += n_clusters
        
        logger.info(f"  姿态组 '{pose_group}': {len(group_features)} 个样本, {n_clusters} 个聚类")
    
    return cluster_labels, group_cluster_info


def reduce_dimensions(features, method='tsne', n_components=2, random_state=42):
    """降维到2D用于可视化"""
    logger.info(f"使用 {method.upper()} 降维到 {n_components}D...")
    
    if method.lower() == 'tsne':
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=30,
            max_iter=1000,
            verbose=1
        )
    elif method.lower() == 'pca':
        reducer = PCA(
            n_components=n_components,
            random_state=random_state
        )
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    reduced_features = reducer.fit_transform(features)
    logger.info(f"降维完成: {features.shape} -> {reduced_features.shape}")
    
    return reduced_features


def create_interactive_plot_by_pose(
    reduced_features,
    pose_labels,
    cluster_labels,
    person_names,
    file_paths,
    poses,
    data_types,
    output_path='feature_clusters_by_pose.html',
    title='基于姿态的特征聚类可视化'
):
    """创建基于姿态的交互式图表"""
    logger.info("创建基于姿态的交互式图表...")
    
    # 准备悬停文本
    hover_texts = []
    for i, (person_name, file_path, data_type, pose) in enumerate(zip(person_names, file_paths, data_types, poses)):
        file_name = Path(file_path).name
        yaw, pitch, roll = pose[0], pose[1], pose[2]
        hover_text = f"<b>文件名:</b> {file_name}<br>"
        hover_text += f"<b>人员:</b> {person_name}<br>"
        hover_text += f"<b>类型:</b> {data_type}<br>"
        hover_text += f"<b>姿态:</b> {pose_labels[i]}<br>"
        hover_text += f"<b>Yaw:</b> {yaw:.1f}°<br>"
        hover_text += f"<b>Pitch:</b> {pitch:.1f}°<br>"
        hover_text += f"<b>Roll:</b> {roll:.1f}°<br>"
        hover_text += f"<b>聚类:</b> {cluster_labels[i]}<br>"
        hover_text += f"<b>完整路径:</b> {file_path}"
        hover_texts.append(hover_text)
    
    # 按姿态分组
    unique_poses = sorted(set(pose_labels))
    n_pose_groups = len(unique_poses)
    colors = px.colors.qualitative.Set3[:n_pose_groups] if n_pose_groups <= 12 else px.colors.qualitative.Alphabet[:n_pose_groups]
    
    fig = go.Figure()
    
    # 为每个姿态组添加散点图
    for pose_idx, pose_group in enumerate(unique_poses):
        mask = np.array([pl == pose_group for pl in pose_labels])
        group_features = reduced_features[mask]
        group_hover_texts = [hover_texts[i] for i in range(len(hover_texts)) if mask[i]]
        
        # 区分face和video
        group_types = [data_types[i] for i in range(len(data_types)) if mask[i]]
        face_mask = np.array([gt == 'face' for gt in group_types])
        video_mask = ~face_mask
        
        # Face点
        if np.any(face_mask):
            fig.add_trace(go.Scatter(
                x=group_features[face_mask, 0],
                y=group_features[face_mask, 1],
                mode='markers',
                name=f'{pose_group} (Face)',
                text=[group_hover_texts[i] for i in range(len(group_hover_texts)) if face_mask[i]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=8,
                    color=colors[pose_idx % len(colors)],
                    opacity=0.7,
                    symbol='circle',
                    line=dict(width=1, color='white')
                )
            ))
        
        # Video点
        if np.any(video_mask):
            fig.add_trace(go.Scatter(
                x=group_features[video_mask, 0],
                y=group_features[video_mask, 1],
                mode='markers',
                name=f'{pose_group} (Video)',
                text=[group_hover_texts[i] for i in range(len(group_hover_texts)) if video_mask[i]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=8,
                    color=colors[pose_idx % len(colors)],
                    opacity=0.7,
                    symbol='triangle-up',
                    line=dict(width=1, color='white')
                )
            ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title='维度 1',
        yaxis_title='维度 2',
        hovermode='closest',
        width=1400,
        height=900,
        template='plotly_white',
        legend=dict(
            title='姿态类别',
            itemsizing='constant'
        )
    )
    
    fig.write_html(output_path)
    logger.info(f"交互式图表已保存到: {output_path}")
    
    return fig


def create_matplotlib_plot_by_pose(
    reduced_features,
    pose_labels,
    cluster_labels,
    person_names,
    file_paths,
    poses,
    data_types,
    output_path='feature_clusters_by_pose.png',
    title='基于姿态的特征聚类可视化'
):
    """创建基于姿态的静态图表"""
    logger.info("创建基于姿态的静态图表...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    unique_poses = sorted(set(pose_labels))
    n_pose_groups = len(unique_poses)
    colors = plt.cm.tab20(np.linspace(0, 1, n_pose_groups))
    
    # 为每个姿态组绘制散点
    for pose_idx, pose_group in enumerate(unique_poses):
        mask = np.array([pl == pose_group for pl in pose_labels])
        group_features = reduced_features[mask]
        group_types = [data_types[i] for i in range(len(data_types)) if mask[i]]
        
        face_mask = np.array([gt == 'face' for gt in group_types])
        video_mask = ~face_mask
        
        if np.any(face_mask):
            ax.scatter(
                group_features[face_mask, 0],
                group_features[face_mask, 1],
                c=[colors[pose_idx]],
                marker='o',
                s=50,
                alpha=0.6,
                label=f'{pose_group} (Face)',
                edgecolors='black',
                linewidths=0.5
            )
        
        if np.any(video_mask):
            ax.scatter(
                group_features[video_mask, 0],
                group_features[video_mask, 1],
                c=[colors[pose_idx]],
                marker='^',
                s=50,
                alpha=0.6,
                label=f'{pose_group} (Video)',
                edgecolors='black',
                linewidths=0.5
            )
    
    ax.set_xlabel('维度 1', fontsize=12)
    ax.set_ylabel('维度 2', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"静态图表已保存到: {output_path}")
    plt.close()


def analyze_pose_distribution(pose_labels, poses):
    """分析姿态分布"""
    logger.info("\n" + "=" * 70)
    logger.info("姿态分布分析")
    logger.info("=" * 70)
    
    from collections import Counter
    pose_counts = Counter(pose_labels)
    
    logger.info("\n各姿态类别的样本数量:")
    for pose_group, count in sorted(pose_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(pose_labels) * 100
        logger.info(f"  {pose_group}: {count} ({percentage:.1f}%)")
    
    # 分析角度分布
    yaw = poses[:, 0]
    pitch = poses[:, 1]
    roll = poses[:, 2]
    
    logger.info(f"\n角度统计:")
    logger.info(f"  Yaw:   平均={np.mean(yaw):.1f}°, 标准差={np.std(yaw):.1f}°, 范围=[{np.min(yaw):.1f}°, {np.max(yaw):.1f}°]")
    logger.info(f"  Pitch: 平均={np.mean(pitch):.1f}°, 标准差={np.std(pitch):.1f}°, 范围=[{np.min(pitch):.1f}°, {np.max(pitch):.1f}°]")
    logger.info(f"  Roll:  平均={np.mean(roll):.1f}°, 标准差={np.std(roll):.1f}°, 范围=[{np.min(roll):.1f}°, {np.max(roll):.1f}°]")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于姿态的特征聚类可视化工具')
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                        help='数据目录')
    parser.add_argument('--yaw_threshold', type=float, default=15,
                        help='Yaw角度阈值（度）')
    parser.add_argument('--pitch_threshold', type=float, default=15,
                        help='Pitch角度阈值（度）')
    parser.add_argument('--roll_threshold', type=float, default=10,
                        help='Roll角度阈值（度）')
    parser.add_argument('--n_clusters_per_group', type=int, default=3,
                        help='每个姿态组内的聚类数量')
    parser.add_argument('--reduce_method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='降维方法')
    parser.add_argument('--output_html', type=str, default='feature_clusters_by_pose.html',
                        help='交互式图表输出路径')
    parser.add_argument('--output_png', type=str, default='feature_clusters_by_pose.png',
                        help='静态图表输出路径')
    parser.add_argument('--skip_static', action='store_true',
                        help='跳过静态图表生成')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("基于姿态的特征聚类可视化工具")
    logger.info("=" * 70)
    
    try:
        # 1. 加载数据
        all_features, all_labels, all_poses, all_file_paths, all_types = load_features_and_metadata(args.data_dir)
        
        # 2. 根据姿态分类
        logger.info("\n根据姿态角度分类...")
        pose_labels, pose_categories = classify_by_pose(
            all_poses,
            yaw_threshold=args.yaw_threshold,
            pitch_threshold=args.pitch_threshold,
            roll_threshold=args.roll_threshold
        )
        
        # 3. 分析姿态分布
        analyze_pose_distribution(pose_labels, all_poses)
        
        # 4. 对每个姿态组进行聚类
        logger.info("\n对每个姿态组进行聚类...")
        cluster_labels, group_cluster_info = cluster_by_pose_groups(
            all_features,
            pose_labels,
            n_clusters_per_group=args.n_clusters_per_group
        )
        
        logger.info(f"\n总共创建了 {len(set(cluster_labels))} 个聚类")
        
        # 5. 降维
        reduced_features = reduce_dimensions(
            all_features,
            method=args.reduce_method,
            n_components=2,
            random_state=args.random_state
        )
        
        # 6. 创建交互式图表
        create_interactive_plot_by_pose(
            reduced_features,
            pose_labels,
            cluster_labels,
            all_labels,
            all_file_paths,
            all_poses,
            all_types,
            output_path=args.output_html,
            title=f'基于姿态的特征聚类可视化 ({args.reduce_method.upper()}, {len(set(cluster_labels))} 个聚类)'
        )
        
        # 7. 创建静态图表
        if not args.skip_static:
            create_matplotlib_plot_by_pose(
                reduced_features,
                pose_labels,
                cluster_labels,
                all_labels,
                all_file_paths,
                all_poses,
                all_types,
                output_path=args.output_png,
                title=f'基于姿态的特征聚类可视化 ({args.reduce_method.upper()}, {len(set(cluster_labels))} 个聚类)'
            )
        
        logger.info("\n" + "=" * 70)
        logger.info("完成！")
        logger.info(f"交互式图表: {args.output_html}")
        if not args.skip_static:
            logger.info(f"静态图表: {args.output_png}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
