"""
可视化两张图片的特征差异
分别使用 features_224（无resize_to_96）和 features_96（有resize_to_96）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from pathlib import Path
from feature_extractor import DINOv2FeatureExtractor
from feature_manager import FeatureManager
import torch
import platform

# 配置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()
    
    # 根据操作系统选择中文字体
    if system == 'Windows':
        # Windows系统常用中文字体
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        font_candidates = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        font_candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 尝试设置字体
    font_set = False
    for font_name in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            # 测试字体是否可用
            test_fig = plt.figure(figsize=(1, 1))
            test_ax = test_fig.add_subplot(111)
            test_ax.text(0.5, 0.5, '测试', fontsize=10)
            plt.close(test_fig)
            font_set = True
            print(f"✓ 已设置中文字体: {font_name}")
            break
        except Exception:
            continue
    
    if not font_set:
        # 如果所有字体都不可用，尝试查找系统可用的中文字体
        try:
            fonts = [f.name for f in fm.fontManager.ttflist]
            chinese_fonts = [f for f in fonts if any(keyword in f.lower() for keyword in ['hei', 'song', 'kai', 'fang', 'yahei', 'simhei', 'simsun'])]
            if chinese_fonts:
                plt.rcParams['font.sans-serif'] = [chinese_fonts[0]]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✓ 已设置中文字体: {chinese_fonts[0]}")
            else:
                print("⚠️ 警告: 未找到中文字体，中文可能显示为方块")
                plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"⚠️ 警告: 设置中文字体失败: {e}")
            plt.rcParams['axes.unicode_minus'] = False

# 初始化中文字体
setup_chinese_font()

def extract_features_for_image(image_path, resize_to_96=False, device=None):
    """
    提取图像特征
    
    Args:
        image_path: 图像路径（支持中文路径）
        resize_to_96: 是否先缩放到96*96
        device: 计算设备
        
    Returns:
        features: 特征向量
    """
    # 确保路径是字符串格式（支持中文）
    image_path_str = str(image_path)
    extractor = DINOv2FeatureExtractor(resize_to_96=resize_to_96, device=device)
    features = extractor.extract_features(image_path_str)
    return features

def load_features_from_db(storage_dir):
    """
    从特征库加载特征
    
    Args:
        storage_dir: 特征库目录
        
    Returns:
        features: 特征矩阵
        metadata: 元数据列表
    """
    manager = FeatureManager(storage_dir=storage_dir)
    features, metadata = manager.get_all_features()
    return features, metadata

def find_image_in_db(image_path, storage_dir):
    """
    在特征库中查找图像的特征
    
    Args:
        image_path: 图像路径
        storage_dir: 特征库目录
        
    Returns:
        features: 特征向量，如果未找到返回None
        index: 特征索引，如果未找到返回None
    """
    features_db, metadata = load_features_from_db(storage_dir)
    if features_db is None or metadata is None:
        return None, None
    
    # 查找匹配的图像路径
    image_path_str = str(image_path)
    for i, meta in enumerate(metadata):
        if meta.get('image_path') == image_path_str:
            return features_db[i], i
    
    return None, None

def visualize_feature_comparison(image1_path, image2_path, 
                                storage_dir_224='features_224',
                                storage_dir_96='features_96',
                                output_dir='feature_visualization'):
    """
    可视化两张图片的特征差异
    
    Args:
        image1_path: 第一张图片路径
        image2_path: 第二张图片路径
        storage_dir_224: features_224目录（无resize_to_96）
        storage_dir_96: features_96目录（有resize_to_96）
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("特征差异可视化")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载图像（使用PIL，支持中文路径）
    print(f"\n加载图像...")
    print(f"  图像1: {image1_path}")
    print(f"  图像2: {image2_path}")
    
    # 使用PIL读取图像，支持中文路径
    img1 = Image.open(str(image1_path)).convert('RGB')
    img2 = Image.open(str(image2_path)).convert('RGB')
    
    print(f"  图像1尺寸: {img1.size}")
    print(f"  图像2尺寸: {img2.size}")
    
    # 提取特征（features_224 - 无resize_to_96）
    print(f"\n提取特征（features_224 - 无resize_to_96）...")
    print("  提取图像1特征...")
    feat1_224 = extract_features_for_image(image1_path, resize_to_96=False, device=device)
    print(f"    特征维度: {len(feat1_224)}")
    
    print("  提取图像2特征...")
    feat2_224 = extract_features_for_image(image2_path, resize_to_96=False, device=device)
    print(f"    特征维度: {len(feat2_224)}")
    
    # 提取特征（features_96 - 有resize_to_96）
    print(f"\n提取特征（features_96 - 有resize_to_96）...")
    print("  提取图像1特征...")
    feat1_96 = extract_features_for_image(image1_path, resize_to_96=True, device=device)
    print(f"    特征维度: {len(feat1_96)}")
    
    print("  提取图像2特征...")
    feat2_96 = extract_features_for_image(image2_path, resize_to_96=True, device=device)
    print(f"    特征维度: {len(feat2_96)}")
    
    # 计算相似度
    def cosine_similarity(feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    sim_224 = cosine_similarity(feat1_224, feat2_224)
    sim_96 = cosine_similarity(feat1_96, feat2_96)
    
    print(f"\n相似度对比:")
    print(f"  features_224 (无resize_to_96): {sim_224:.4f}")
    print(f"  features_96 (有resize_to_96): {sim_96:.4f}")
    
    # 计算特征差异
    diff_224 = np.abs(feat1_224 - feat2_224)
    diff_96 = np.abs(feat1_96 - feat2_96)
    
    print(f"\n特征差异统计:")
    print(f"  features_224:")
    print(f"    平均差异: {np.mean(diff_224):.6f}")
    print(f"    最大差异: {np.max(diff_224):.6f}")
    print(f"    最小差异: {np.min(diff_224):.6f}")
    print(f"    标准差: {np.std(diff_224):.6f}")
    print(f"  features_96:")
    print(f"    平均差异: {np.mean(diff_96):.6f}")
    print(f"    最大差异: {np.max(diff_96):.6f}")
    print(f"    最小差异: {np.min(diff_96):.6f}")
    print(f"    标准差: {np.std(diff_96):.6f}")
    
    # 可视化
    print(f"\n生成可视化图像...")
    
    # 创建大图
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 显示原始图像
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(img1)
    ax1.set_title(f'图像1\n{Path(image1_path).name}', fontsize=10)
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 4, 2)
    ax2.imshow(img2)
    ax2.set_title(f'图像2\n{Path(image2_path).name}', fontsize=10)
    ax2.axis('off')
    
    # 2. features_224 特征向量对比
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(feat1_224[:500], label='图像1', alpha=0.7, linewidth=0.5)
    ax3.plot(feat2_224[:500], label='图像2', alpha=0.7, linewidth=0.5)
    ax3.set_title(f'features_224 特征向量对比\n(前500维, 相似度: {sim_224:.4f})', fontsize=10)
    ax3.set_xlabel('特征维度')
    ax3.set_ylabel('特征值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(diff_224[:500], color='red', linewidth=0.5)
    ax4.set_title(f'features_224 特征差异\n(前500维)', fontsize=10)
    ax4.set_xlabel('特征维度')
    ax4.set_ylabel('|特征1 - 特征2|')
    ax4.grid(True, alpha=0.3)
    
    # 3. features_96 特征向量对比
    ax5 = plt.subplot(3, 4, 5)
    ax5.plot(feat1_96[:500], label='图像1', alpha=0.7, linewidth=0.5)
    ax5.plot(feat2_96[:500], label='图像2', alpha=0.7, linewidth=0.5)
    ax5.set_title(f'features_96 特征向量对比\n(前500维, 相似度: {sim_96:.4f})', fontsize=10)
    ax5.set_xlabel('特征维度')
    ax5.set_ylabel('特征值')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(3, 4, 6)
    ax6.plot(diff_96[:500], color='red', linewidth=0.5)
    ax6.set_title(f'features_96 特征差异\n(前500维)', fontsize=10)
    ax6.set_xlabel('特征维度')
    ax6.set_ylabel('|特征1 - 特征2|')
    ax6.grid(True, alpha=0.3)
    
    # 4. 特征差异分布直方图
    ax7 = plt.subplot(3, 4, 7)
    ax7.hist(diff_224, bins=50, alpha=0.7, label='features_224', color='blue')
    ax7.hist(diff_96, bins=50, alpha=0.7, label='features_96', color='orange')
    ax7.set_title('特征差异分布', fontsize=10)
    ax7.set_xlabel('特征差异值')
    ax7.set_ylabel('频数')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 5. 相似度对比
    ax8 = plt.subplot(3, 4, 8)
    categories = ['features_224\n(无resize_to_96)', 'features_96\n(有resize_to_96)']
    similarities = [sim_224, sim_96]
    colors = ['blue', 'orange']
    bars = ax8.bar(categories, similarities, color=colors, alpha=0.7)
    ax8.set_ylim([0, 1])
    ax8.set_ylabel('余弦相似度')
    ax8.set_title('相似度对比', fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{sim:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 6. 特征差异统计对比
    ax9 = plt.subplot(3, 4, 9)
    stats_224 = [np.mean(diff_224), np.max(diff_224), np.std(diff_224)]
    stats_96 = [np.mean(diff_96), np.max(diff_96), np.std(diff_96)]
    x = np.arange(3)
    width = 0.35
    ax9.bar(x - width/2, stats_224, width, label='features_224', alpha=0.7, color='blue')
    ax9.bar(x + width/2, stats_96, width, label='features_96', alpha=0.7, color='orange')
    ax9.set_ylabel('差异值')
    ax9.set_title('特征差异统计对比', fontsize=10)
    ax9.set_xticks(x)
    ax9.set_xticklabels(['平均差异', '最大差异', '标准差'])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # 7. 特征向量热力图（features_224）
    ax10 = plt.subplot(3, 4, 10)
    # 将特征向量reshape为2D用于显示
    feat1_2d_224 = feat1_224[:768].reshape(24, 32) if len(feat1_224) >= 768 else feat1_224.reshape(-1, 1)
    feat2_2d_224 = feat2_224[:768].reshape(24, 32) if len(feat2_224) >= 768 else feat2_224.reshape(-1, 1)
    diff_2d_224 = np.abs(feat1_2d_224 - feat2_2d_224)
    im1 = ax10.imshow(diff_2d_224, cmap='hot', aspect='auto')
    ax10.set_title('features_224 特征差异热力图', fontsize=10)
    plt.colorbar(im1, ax=ax10)
    
    # 8. 特征向量热力图（features_96）
    ax11 = plt.subplot(3, 4, 11)
    feat1_2d_96 = feat1_96[:768].reshape(24, 32) if len(feat1_96) >= 768 else feat1_96.reshape(-1, 1)
    feat2_2d_96 = feat2_96[:768].reshape(24, 32) if len(feat2_96) >= 768 else feat2_96.reshape(-1, 1)
    diff_2d_96 = np.abs(feat1_2d_96 - feat2_2d_96)
    im2 = ax11.imshow(diff_2d_96, cmap='hot', aspect='auto')
    ax11.set_title('features_96 特征差异热力图', fontsize=10)
    plt.colorbar(im2, ax=ax11)
    
    # 9. 特征值分布对比
    ax12 = plt.subplot(3, 4, 12)
    ax12.scatter(feat1_224[:200], feat2_224[:200], alpha=0.5, s=10, label='features_224', color='blue')
    ax12.scatter(feat1_96[:200], feat2_96[:200], alpha=0.5, s=10, label='features_96', color='orange')
    ax12.plot([-1, 1], [-1, 1], 'r--', linewidth=1, alpha=0.5)  # 对角线
    ax12.set_xlabel('图像1特征值')
    ax12.set_ylabel('图像2特征值')
    ax12.set_title('特征值散点图对比\n(前200维)', fontsize=10)
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = output_dir / 'feature_difference_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  可视化结果已保存到: {output_path}")
    plt.close()
    
    # 保存详细数据到文本文件
    output_txt = output_dir / 'feature_difference_stats.txt'
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("特征差异分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"图像1: {image1_path}\n")
        f.write(f"图像2: {image2_path}\n\n")
        f.write(f"features_224 (无resize_to_96):\n")
        f.write(f"  特征维度: {len(feat1_224)}\n")
        f.write(f"  余弦相似度: {sim_224:.6f}\n")
        f.write(f"  平均差异: {np.mean(diff_224):.6f}\n")
        f.write(f"  最大差异: {np.max(diff_224):.6f}\n")
        f.write(f"  最小差异: {np.min(diff_224):.6f}\n")
        f.write(f"  标准差: {np.std(diff_224):.6f}\n\n")
        f.write(f"features_96 (有resize_to_96):\n")
        f.write(f"  特征维度: {len(feat1_96)}\n")
        f.write(f"  余弦相似度: {sim_96:.6f}\n")
        f.write(f"  平均差异: {np.mean(diff_96):.6f}\n")
        f.write(f"  最大差异: {np.max(diff_96):.6f}\n")
        f.write(f"  最小差异: {np.min(diff_96):.6f}\n")
        f.write(f"  标准差: {np.std(diff_96):.6f}\n\n")
        f.write(f"相似度差异: {abs(sim_224 - sim_96):.6f}\n")
        f.write(f"平均差异变化: {np.mean(diff_96) - np.mean(diff_224):.6f}\n")
    
    print(f"  统计结果已保存到: {output_txt}")
    
    print("\n" + "=" * 70)
    print("可视化完成！")
    print("=" * 70)

def main():
    # 图像路径
    image1_path = r'C:\Codes\face000\train\datas\face\柴懿珈.jpg'
    image2_path = r'C:\Codes\face000\train\datas\video\柴懿珈\柴懿珈frame_000081.jpg'
    
    # 特征库目录
    storage_dir_224 = 'features_224'
    storage_dir_96 = 'features_96'
    
    # 输出目录
    output_dir = 'feature_visualization'
    
    # 执行可视化
    visualize_feature_comparison(
        image1_path=image1_path,
        image2_path=image2_path,
        storage_dir_224=storage_dir_224,
        storage_dir_96=storage_dir_96,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()

