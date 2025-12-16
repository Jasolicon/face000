"""
特征聚类可视化工具
将所有特征按照人数进行聚类，并使用交互式图表展示
鼠标悬停时显示文件名
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
    """
    加载所有特征和元数据
    
    Args:
        data_dir: 数据目录
    
    Returns:
        all_features: 所有特征 [N, feature_dim]
        all_labels: 所有标签（person_name）[N]
        all_file_paths: 所有文件路径 [N]
        all_types: 所有类型（'face' 或 'video'）[N]
    """
    data_dir = Path(data_dir)
    
    logger.info(f"加载数据目录: {data_dir}")
    
    # 检查文件是否存在
    required_files = [
        'front_feature.npy',
        'front_metadata.json',
        'video_feature.npy',
        'video_metadata.json'
    ]
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 加载正面图数据
    logger.info("加载正面图数据...")
    front_features = np.load(data_dir / 'front_feature.npy')  # [N, 512]
    with open(data_dir / 'front_metadata.json', 'r', encoding='utf-8') as f:
        front_metadata = json.load(f)
    
    front_person_names = [meta['person_name'] for meta in front_metadata['metadata']]
    front_file_paths = [meta['image_path'] for meta in front_metadata['metadata']]
    
    logger.info(f"  正面图数量: {len(front_features)}")
    logger.info(f"  正面图人数: {len(set(front_person_names))}")
    
    # 加载视频帧数据
    logger.info("加载视频帧数据...")
    video_features = np.load(data_dir / 'video_feature.npy')  # [M, 512]
    with open(data_dir / 'video_metadata.json', 'r', encoding='utf-8') as f:
        video_metadata = json.load(f)
    
    video_person_names = [meta['person_name'] for meta in video_metadata['metadata']]
    video_file_paths = [meta['image_path'] for meta in video_metadata['metadata']]
    
    logger.info(f"  视频帧数量: {len(video_features)}")
    logger.info(f"  视频帧人数: {len(set(video_person_names))}")
    
    # 合并所有数据
    all_features = np.vstack([front_features, video_features])  # [N+M, 512]
    all_labels = front_person_names + video_person_names
    all_file_paths = front_file_paths + video_file_paths
    all_types = ['face'] * len(front_features) + ['video'] * len(video_features)
    
    logger.info(f"总特征数量: {len(all_features)}")
    logger.info(f"总人数: {len(set(all_labels))}")
    logger.info(f"特征维度: {all_features.shape[1]}")
    
    return all_features, all_labels, all_file_paths, all_types


def reduce_dimensions(features, method='tsne', n_components=2, random_state=42):
    """
    降维到2D用于可视化
    
    Args:
        features: 特征矩阵 [N, feature_dim]
        method: 降维方法 ('tsne' 或 'pca')
        n_components: 降维后的维度（默认2）
        random_state: 随机种子
    
    Returns:
        reduced_features: 降维后的特征 [N, n_components]
    """
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


def perform_clustering(features, person_names=None, n_clusters=None, method='kmeans', random_state=42):
    """
    对特征进行聚类
    
    Args:
        features: 特征矩阵 [N, feature_dim] 或降维后的特征 [N, 2]
        person_names: 人名列表 [N]（如果提供，将根据人数自动确定聚类数）
        n_clusters: 聚类数量（如果为None且提供了person_names，则根据人数自动确定）
        method: 聚类方法（默认 'kmeans'）
        random_state: 随机种子
    
    Returns:
        cluster_labels: 聚类标签 [N]
        n_clusters: 实际使用的聚类数量
    """
    if n_clusters is None:
        if person_names is not None:
            # 根据人数自动确定聚类数
            unique_persons = len(set(person_names))
            n_clusters = unique_persons
            logger.info(f"根据人数自动确定聚类数量: {n_clusters} (共 {unique_persons} 人)")
        else:
            # 如果未指定人数，使用特征数量的一半作为聚类数
            n_clusters = max(2, min(50, features.shape[0] // 10))
            logger.info(f"自动确定聚类数量: {n_clusters}")
    
    logger.info(f"使用 {method} 进行聚类，聚类数量: {n_clusters}...")
    
    if method.lower() == 'kmeans':
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
    else:
        raise ValueError(f"不支持的聚类方法: {method}")
    
    cluster_labels = clusterer.fit_predict(features)
    
    logger.info(f"聚类完成，共 {len(set(cluster_labels))} 个聚类")
    
    return cluster_labels, n_clusters


def create_interactive_plot(
    reduced_features,
    cluster_labels,
    person_names,
    file_paths,
    data_types,
    output_path='feature_clusters.html',
    title='特征聚类可视化'
):
    """
    创建交互式图表（使用 Plotly）
    鼠标悬停时显示文件名
    
    Args:
        reduced_features: 降维后的特征 [N, 2]
        cluster_labels: 聚类标签 [N]
        person_names: 人名列表 [N]
        file_paths: 文件路径列表 [N]
        data_types: 数据类型列表 [N] ('face' 或 'video')
        output_path: 输出HTML文件路径
        title: 图表标题
    """
    logger.info("创建交互式图表...")
    
    # 准备悬停文本
    hover_texts = []
    for i, (person_name, file_path, data_type) in enumerate(zip(person_names, file_paths, data_types)):
        file_name = Path(file_path).name
        hover_text = f"<b>文件名:</b> {file_name}<br>"
        hover_text += f"<b>人员:</b> {person_name}<br>"
        hover_text += f"<b>类型:</b> {data_type}<br>"
        hover_text += f"<b>聚类:</b> {cluster_labels[i]}<br>"
        hover_text += f"<b>完整路径:</b> {file_path}"
        hover_texts.append(hover_text)
    
    # 创建颜色映射（按聚类）
    n_clusters = len(set(cluster_labels))
    colors = px.colors.qualitative.Set3[:n_clusters] if n_clusters <= 12 else px.colors.qualitative.Alphabet[:n_clusters]
    
    # 创建图表
    fig = go.Figure()
    
    # 为每个聚类添加一个散点图
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_features = reduced_features[mask]
        cluster_hover_texts = [hover_texts[i] for i in range(len(hover_texts)) if mask[i]]
        cluster_person_names = [person_names[i] for i in range(len(person_names)) if mask[i]]
        
        fig.add_trace(go.Scatter(
            x=cluster_features[:, 0],
            y=cluster_features[:, 1],
            mode='markers',
            name=f'聚类 {cluster_id}',
            text=cluster_hover_texts,
            hovertemplate='%{text}<extra></extra>',
            marker=dict(
                size=8,
                color=colors[cluster_id % len(colors)],
                opacity=0.7,
                line=dict(width=1, color='white')
            )
        ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),
        xaxis_title='维度 1',
        yaxis_title='维度 2',
        hovermode='closest',
        width=1200,
        height=800,
        template='plotly_white',
        legend=dict(
            title='聚类',
            itemsizing='constant'
        )
    )
    
    # 保存为HTML
    fig.write_html(output_path)
    logger.info(f"交互式图表已保存到: {output_path}")
    
    return fig


def create_matplotlib_plot(
    reduced_features,
    cluster_labels,
    person_names,
    file_paths,
    data_types,
    output_path='feature_clusters.png',
    title='特征聚类可视化'
):
    """
    创建静态图表（使用 Matplotlib）
    
    Args:
        reduced_features: 降维后的特征 [N, 2]
        cluster_labels: 聚类标签 [N]
        person_names: 人名列表 [N]
        file_paths: 文件路径列表 [N]
        data_types: 数据类型列表 [N]
        output_path: 输出图片路径
        title: 图表标题
    """
    logger.info("创建静态图表...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 为每个聚类绘制散点
    n_clusters = len(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_features = reduced_features[mask]
        
        # 区分 face 和 video
        face_mask = np.array([data_types[i] == 'face' for i in range(len(data_types)) if mask[i]])
        video_mask = ~face_mask
        
        if np.any(face_mask):
            ax.scatter(
                cluster_features[face_mask, 0],
                cluster_features[face_mask, 1],
                c=[colors[cluster_id]],
                marker='o',
                s=50,
                alpha=0.6,
                label=f'聚类 {cluster_id} (Face)' if cluster_id == 0 else '',
                edgecolors='black',
                linewidths=0.5
            )
        
        if np.any(video_mask):
            ax.scatter(
                cluster_features[video_mask, 0],
                cluster_features[video_mask, 1],
                c=[colors[cluster_id]],
                marker='^',
                s=50,
                alpha=0.6,
                label=f'聚类 {cluster_id} (Video)' if cluster_id == 0 else '',
                edgecolors='black',
                linewidths=0.5
            )
    
    ax.set_xlabel('维度 1', fontsize=12)
    ax.set_ylabel('维度 2', fontsize=12)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"静态图表已保存到: {output_path}")
    plt.close()


def analyze_clustering_quality(cluster_labels, person_names):
    """
    分析聚类质量（按人员分组）
    
    Args:
        cluster_labels: 聚类标签 [N]
        person_names: 人名列表 [N]
    """
    logger.info("\n分析聚类质量...")
    
    # 统计每个聚类中的人员分布
    cluster_to_persons = defaultdict(set)
    person_to_clusters = defaultdict(set)
    
    for cluster_id, person_name in zip(cluster_labels, person_names):
        cluster_to_persons[cluster_id].add(person_name)
        person_to_clusters[person_name].add(cluster_id)
    
    # 计算每个聚类的主要人员
    logger.info("\n各聚类的人员分布:")
    for cluster_id in sorted(cluster_to_persons.keys()):
        persons = cluster_to_persons[cluster_id]
        logger.info(f"  聚类 {cluster_id}: {len(persons)} 人 - {sorted(persons)}")
    
    # 计算每个人员所在的聚类
    logger.info("\n各人员的聚类分布:")
    for person_name in sorted(person_to_clusters.keys()):
        clusters = person_to_clusters[person_name]
        if len(clusters) > 1:
            logger.info(f"  {person_name}: 分布在 {len(clusters)} 个聚类中 - {sorted(clusters)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征聚类可视化工具')
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                        help='数据目录（包含front_*.npy和video_*.npy文件）')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='聚类数量（如果为None，则自动确定）')
    parser.add_argument('--reduce_method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='降维方法（tsne 或 pca）')
    parser.add_argument('--cluster_method', type=str, default='kmeans', choices=['kmeans'],
                        help='聚类方法')
    parser.add_argument('--output_html', type=str, default='feature_clusters.html',
                        help='交互式图表输出路径（HTML）')
    parser.add_argument('--output_png', type=str, default='feature_clusters.png',
                        help='静态图表输出路径（PNG）')
    parser.add_argument('--skip_static', action='store_true',
                        help='跳过静态图表生成')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("特征聚类可视化工具")
    logger.info("=" * 70)
    
    try:
        # 1. 加载数据
        all_features, all_labels, all_file_paths, all_types = load_features_and_metadata(args.data_dir)
        
        # 2. 降维
        reduced_features = reduce_dimensions(
            all_features,
            method=args.reduce_method,
            n_components=2,
            random_state=args.random_state
        )
        
        # 3. 聚类（按人数）
        cluster_labels, n_clusters = perform_clustering(
            all_features,  # 使用原始特征进行聚类（更准确）
            person_names=all_labels,  # 传入人名列表，用于自动确定聚类数
            n_clusters=args.n_clusters,
            method=args.cluster_method,
            random_state=args.random_state
        )
        
        # 4. 分析聚类质量
        analyze_clustering_quality(cluster_labels, all_labels)
        
        # 5. 创建交互式图表（支持鼠标悬停显示文件名）
        create_interactive_plot(
            reduced_features,
            cluster_labels,
            all_labels,
            all_file_paths,
            all_types,
            output_path=args.output_html,
            title=f'特征聚类可视化 ({args.reduce_method.upper()}, {n_clusters} 个聚类)'
        )
        
        # 6. 创建静态图表（可选）
        if not args.skip_static:
            create_matplotlib_plot(
                reduced_features,
                cluster_labels,
                all_labels,
                all_file_paths,
                all_types,
                output_path=args.output_png,
                title=f'特征聚类可视化 ({args.reduce_method.upper()}, {n_clusters} 个聚类)'
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
