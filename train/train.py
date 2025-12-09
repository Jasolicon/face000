"""
多角度人脸识别模型训练
损失函数：交叉熵 + ArcFace + 对比损失
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, Optional, List
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

# 添加项目根目录和train目录到路径
train_dir = Path(__file__).parent
project_root = train_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(train_dir))

# 使用相对导入（从train目录内运行）
from dataset import create_dataloader, MultiAngleFaceDataset
from model import MultiAngleFaceModel, ContrastiveLoss


class CombinedLoss(nn.Module):
    """组合损失函数：ArcFace损失 + 对比损失 + 一致性损失"""
    
    def __init__(
        self,
        use_arcface: bool = True,
        use_contrastive: bool = True,
        arcface_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        ce_weight: float = 1.0
    ):
        """
        初始化组合损失
        
        Args:
            use_arcface: 是否使用ArcFace损失
            use_contrastive: 是否使用对比损失
            arcface_weight: ArcFace损失权重
            contrastive_weight: 对比损失权重
            ce_weight: 交叉熵损失权重
        """
        super().__init__()
        self.use_arcface = use_arcface
        self.use_contrastive = use_contrastive
        self.arcface_weight = arcface_weight
        self.contrastive_weight = contrastive_weight
        self.ce_weight = ce_weight
        
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 对比损失
        if use_contrastive:
            self.contrastive_loss = ContrastiveLoss()
    
    def forward(
        self,
        front_features: torch.Tensor,
        angle_features: torch.Tensor,
        front_logits: Optional[torch.Tensor],
        angle_logits: Optional[torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        
        Args:
            front_features: 正脸特征 [B, D]
            angle_features: 多角度特征 [B, D]
            front_logits: 正脸分类logits [B, num_classes]
            angle_logits: 多角度分类logits [B, num_classes]
            labels: 标签 [B]
            
        Returns:
            损失字典
        """
        losses = {}
        total_loss = 0.0
        
        # ArcFace损失（使用交叉熵计算ArcFace logits的损失）
        if self.use_arcface and front_logits is not None and angle_logits is not None:
            # ArcFace logits已经应用了margin和scale，直接使用交叉熵计算损失
            arcface_front = self.ce_loss(front_logits, labels)
            arcface_angle = self.ce_loss(angle_logits, labels)
            arcface_loss = (arcface_front + arcface_angle) / 2.0
            losses['arcface_loss'] = arcface_loss
            # 为了兼容性，也保留ce_loss名称
            losses['ce_loss'] = arcface_loss
            total_loss += self.arcface_weight * arcface_loss
        else:
            # 如果ArcFace损失未计算，记录原因
            zero_loss = torch.tensor(0.0, device=front_features.device, requires_grad=False)
            losses['arcface_loss'] = zero_loss
            losses['ce_loss'] = zero_loss  # 兼容性
            if not self.use_arcface:
                if not hasattr(self, '_arcface_warning_shown'):
                    print(f"\n⚠️  警告: ArcFace损失为0，因为未启用ArcFace (use_arcface=False)")
                    print(f"   提示: 训练时请添加 --use_arcface 参数以启用ArcFace损失")
                    self._arcface_warning_shown = True
            elif front_logits is None or angle_logits is None:
                if not hasattr(self, '_arcface_warning_shown'):
                    print(f"\n⚠️  警告: ArcFace损失为0，因为模型未返回logits")
                    print(f"   原因: front_logits={front_logits is None}, angle_logits={angle_logits is None}")
                    print(f"   提示: 检查模型是否正确创建了arcface_head (num_classes不应为None)")
                    self._arcface_warning_shown = True
        
        # 对比损失（使用pair_label来判断正负样本对）
        if self.use_contrastive:
            # 需要pair_label来判断正负样本对
            # 但这里只有labels，我们需要从batch中获取pair_label
            # 暂时使用labels来判断（同一label为正样本，不同label为负样本）
            contrastive_loss = self.contrastive_loss(front_features, angle_features, labels)
            losses['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss
        
        # 特征一致性损失（同一人的正脸和多角度特征应该相似）
        cosine_sim = F.cosine_similarity(front_features, angle_features, dim=1)
        consistency_loss = 1.0 - cosine_sim.mean()
        losses['consistency_loss'] = consistency_loss
        total_loss += 0.3 * consistency_loss
        
        losses['total_loss'] = total_loss
        return losses


def train_epoch(
    model: MultiAngleFaceModel,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    
    total_losses = {
        'total_loss': 0.0,
        'arcface_loss': 0.0,
        'ce_loss': 0.0,  # 兼容性，与arcface_loss相同
        'contrastive_loss': 0.0,
        'consistency_loss': 0.0
    }
    
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 确保数据类型和设备正确
        front_images = batch['front_image'].float().to(device)
        angle_images = batch['angle_image'].float().to(device)
        labels = batch['label'].long().to(device)
        
        # 前向传播（DINO孪生网络）
        front_features, front_logits, _ = model(front_images, labels=labels)
        angle_features, angle_logits, similarity = model(angle_images, labels=labels, images2=front_images)
        
        # 调试信息：检查logits是否为None
        if batch_idx == 0 and epoch == 0:
            print(f"\n调试信息:")
            print(f"  use_arcface: {criterion.use_arcface}")
            print(f"  front_logits is None: {front_logits is None}")
            print(f"  angle_logits is None: {angle_logits is None}")
            if front_logits is not None:
                print(f"  front_logits shape: {front_logits.shape}")
            if angle_logits is not None:
                print(f"  angle_logits shape: {angle_logits.shape}")
            print(f"  labels shape: {labels.shape}, unique labels: {torch.unique(labels)}")
            print(f"  model.num_classes: {model.num_classes}")
            print(f"  model.arcface_head is None: {model.arcface_head is None}")
        
        # 计算损失
        losses = criterion(
            front_features,
            angle_features,
            front_logits,
            angle_logits,
            labels
        )
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录损失
        for key, value in losses.items():
            total_losses[key] += value.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'arcface': f"{losses.get('arcface_loss', losses.get('ce_loss', 0)):.4f}",
            'contrastive': f"{losses.get('contrastive_loss', 0):.4f}"
        })
        
        # 记录到tensorboard
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            for key, value in losses.items():
                writer.add_scalar(f'train/{key}', value.item(), global_step)
    
    # 计算平均损失
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    return avg_losses


def plot_losses(
    train_loss: List[float],
    train_ce_loss: List[float],
    train_contrastive_loss: List[float],
    train_consistency_loss: List[float],
    val_accuracy: List[float],
    val_similarity: List[float],
    save_path: Path
):
    """绘制损失曲线图"""
    epochs = range(1, len(train_loss) + 1)
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('训练损失和验证指标', fontsize=16, fontweight='bold')
    
    # 1. 总损失曲线
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, 'b-', label='训练总损失', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title('训练总损失', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. 各项损失曲线
    ax2 = axes[0, 1]
    if any(train_ce_loss):
        ax2.plot(epochs, train_ce_loss, 'r-', label='ArcFace损失', linewidth=2)
    if any(train_contrastive_loss):
        ax2.plot(epochs, train_contrastive_loss, 'g-', label='对比损失', linewidth=2)
    if any(train_consistency_loss):
        ax2.plot(epochs, train_consistency_loss, 'm-', label='一致性损失', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('损失值', fontsize=12)
    ax2.set_title('各项损失分解', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 3. 验证准确率
    ax3 = axes[1, 0]
    ax3.plot(epochs, val_accuracy, 'g-', label='验证准确率', linewidth=2, marker='o', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('准确率', fontsize=12)
    ax3.set_title('验证准确率', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_ylim([0, 1])
    
    # 4. 验证相似度
    ax4 = axes[1, 1]
    ax4.plot(epochs, val_similarity, 'orange', label='平均相似度', linewidth=2, marker='s', markersize=4)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('相似度', fontsize=12)
    ax4.set_title('验证平均相似度', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 损失图已保存: {save_path}")


def validate(
    model: MultiAngleFaceModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """验证"""
    model.eval()
    
    all_similarities = []
    all_pair_labels = []
    total_similarity = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            # 确保数据类型和设备正确
            front_images = batch['front_image'].float().to(device)
            angle_images = batch['angle_image'].float().to(device)
            labels = batch['label'].long().to(device)
            pair_labels = batch['pair_label'].long().to(device)  # 正负样本对标签（1=正样本，0=负样本）
            
            # 提取特征（DINO孪生网络）
            front_features, _, _ = model(front_images, labels=None)
            angle_features, _, similarity = model(angle_images, labels=None, images2=front_images)
            
            # 计算相似度
            cosine_sim = F.cosine_similarity(front_features, angle_features, dim=1)  # [B]
            
            # 收集所有相似度和标签（用于统计分析）
            all_similarities.append(cosine_sim.cpu())
            all_pair_labels.append(pair_labels.cpu())
            
            total_similarity += cosine_sim.sum().item()
            total_samples += len(pair_labels)
    
    # 合并所有batch的数据
    all_similarities = torch.cat(all_similarities, dim=0).numpy()
    all_pair_labels = torch.cat(all_pair_labels, dim=0).numpy()
    
    # 计算平均相似度
    avg_similarity = total_similarity / total_samples if total_samples > 0 else 0.0
    
    # 统计分析
    positive_sim = all_similarities[all_pair_labels == 1]
    negative_sim = all_similarities[all_pair_labels == 0]
    num_positive = len(positive_sim)
    num_negative = len(negative_sim)
    
    # 计算正负样本对的平均相似度
    avg_positive_sim = positive_sim.mean() if len(positive_sim) > 0 else 0.0
    avg_negative_sim = negative_sim.mean() if len(negative_sim) > 0 else 0.0
    
    # 使用固定阈值0.85计算准确率
    threshold = 0.85
    predicted = (all_similarities > threshold).astype(int)
    final_accuracy = (predicted == all_pair_labels).sum() / len(all_pair_labels) if len(all_pair_labels) > 0 else 0.0
    
    # 计算精确率、召回率、F1分数（用于统计）
    tp = ((predicted == 1) & (all_pair_labels == 1)).sum()
    fp = ((predicted == 1) & (all_pair_labels == 0)).sum()
    fn = ((predicted == 0) & (all_pair_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': final_accuracy,  # 使用固定阈值0.85的准确率
        'avg_similarity': avg_similarity,
        'threshold': threshold,  # 固定阈值0.85
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_positive_sim': avg_positive_sim,
        'avg_negative_sim': avg_negative_sim,
        'num_positive': num_positive,
        'num_negative': num_negative
    }
    
    # 计算相似度分布统计
    positive_sim_above_threshold = (positive_sim > threshold).sum() if len(positive_sim) > 0 else 0
    negative_sim_above_threshold = (negative_sim > threshold).sum() if len(negative_sim) > 0 else 0
    positive_sim_below_threshold = len(positive_sim) - positive_sim_above_threshold
    negative_sim_below_threshold = len(negative_sim) - negative_sim_above_threshold
    
    # 计算相似度分位数（用于诊断）
    positive_sim_median = np.median(positive_sim) if len(positive_sim) > 0 else 0.0
    negative_sim_median = np.median(negative_sim) if len(negative_sim) > 0 else 0.0
    positive_sim_std = positive_sim.std() if len(positive_sim) > 0 else 0.0
    negative_sim_std = negative_sim.std() if len(negative_sim) > 0 else 0.0
    
    # 打印详细统计信息（每个epoch都打印，便于诊断）
    print(f"\n验证集统计信息 (Epoch {epoch}):")
    print(f"  总样本数: {total_samples}")
    print(f"  正样本对数: {num_positive} ({num_positive/total_samples*100:.1f}%)")
    print(f"  负样本对数: {num_negative} ({num_negative/total_samples*100:.1f}%)")
    print(f"  正样本对相似度: 平均={avg_positive_sim:.4f}, 中位数={positive_sim_median:.4f}, 标准差={positive_sim_std:.4f}")
    print(f"  负样本对相似度: 平均={avg_negative_sim:.4f}, 中位数={negative_sim_median:.4f}, 标准差={negative_sim_std:.4f}")
    print(f"  相似度差异: {avg_positive_sim - avg_negative_sim:.4f}")
    print(f"  使用阈值: {threshold:.3f}")
    print(f"  正样本对中相似度>{threshold}: {positive_sim_above_threshold}/{num_positive} ({positive_sim_above_threshold/num_positive*100:.1f}%)")
    print(f"  负样本对中相似度>{threshold}: {negative_sim_above_threshold}/{num_negative} ({negative_sim_above_threshold/num_negative*100:.1f}%)")
    print(f"  准确率: {final_accuracy:.4f}")
    print(f"  精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    
    # 如果准确率很低，给出诊断建议
    if final_accuracy < 0.6:
        print(f"\n  ⚠️  诊断信息:")
        if num_negative > num_positive * 5:
            print(f"    - 负样本对数量远多于正样本对（{num_negative/num_positive:.1f}倍），可能导致准确率偏低")
        if avg_positive_sim - avg_negative_sim < 0.1:
            print(f"    - 正负样本对相似度差异太小（{avg_positive_sim - avg_negative_sim:.4f}），模型难以区分")
        if negative_sim_above_threshold > num_negative * 0.3:
            print(f"    - 太多负样本对相似度高于阈值（{negative_sim_above_threshold/num_negative*100:.1f}%），可能需要提高阈值")
        if positive_sim_below_threshold > num_positive * 0.3:
            print(f"    - 太多正样本对相似度低于阈值（{positive_sim_below_threshold/num_positive*100:.1f}%），可能需要降低阈值")
    
    if writer is not None:
        writer.add_scalar('val/accuracy', final_accuracy, epoch)
        writer.add_scalar('val/avg_similarity', avg_similarity, epoch)
        writer.add_scalar('val/threshold', threshold, epoch)
        writer.add_scalar('val/precision', precision, epoch)
        writer.add_scalar('val/recall', recall, epoch)
        writer.add_scalar('val/f1', f1, epoch)
        writer.add_scalar('val/avg_positive_sim', avg_positive_sim, epoch)
        writer.add_scalar('val/avg_negative_sim', avg_negative_sim, epoch)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练多角度人脸识别模型')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--image_size', type=int, default=224, help='图像尺寸')
    parser.add_argument('--feature_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    # Windows上multiprocessing使用spawn方式，需要序列化数据集
    # 如果数据集包含不可序列化的对象（如CUDA设备），会导致EOFError
    # 在Windows上默认使用单进程（num_workers=0）
    default_num_workers = 0 if platform.system() == 'Windows' else 4
    parser.add_argument('--num_workers', type=int, default=default_num_workers, help='数据加载线程数（Windows上建议使用0）')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--use_arcface', action='store_true', help='使用ArcFace损失（推荐启用）')
    parser.add_argument('--use_contrastive', action='store_true', default=True, help='使用对比损失')
    parser.add_argument('--freeze_backbone', action='store_true', help='冻结DINO backbone')
    parser.add_argument('--save_features', action='store_true', help='保存提取的特征（兼容DINOFeatureExtractor）')
    parser.add_argument('--feature_storage_dir', type=str, default='features', help='特征存储目录')
    
    args = parser.parse_args()
    
    # Windows上强制使用num_workers=0，避免序列化问题
    if platform.system() == 'Windows' and args.num_workers > 0:
        print(f"⚠️  警告: Windows上使用num_workers > 0可能导致EOFError")
        print(f"   已自动将num_workers设置为0（单进程模式）")
        print(f"   如需使用多进程，请确保数据集可序列化（移除CUDA设备等）")
        args.num_workers = 0
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("加载数据集...")
    dataset = MultiAngleFaceDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        device=str(device),
        save_features=args.save_features,
        feature_storage_dir=args.feature_storage_dir
    )
    
    # 按类别（人员）7:3划分训练集和验证集
    # 1. 先读取每个类别的所有多角度数据（视频帧）
    # 2. 对每个类别，按排序每4张取1张（降采样）
    # 3. 创建配对：正脸图 + 角度图（正样本或负样本）
    # 4. 按7:3划分配对到训练集和验证集
    # 5. 分别打乱训练集和验证集
    print("按类别划分训练集和验证集（7:3）...")
    print("正在读取每个类别的多角度数据...")
    
    # 按人员ID分组，并读取每个类别的所有视频帧，然后7:3划分
    person_data = {}  # {person_id: {'samples': [indices], 'front_images': [paths], 'train_frames': [frame_paths], 'val_frames': [frame_paths]}}
    
    for idx, sample in enumerate(dataset.samples):
        person_id = sample['person_id']
        if person_id not in person_data:
            person_data[person_id] = {
                'samples': [],
                'front_images': [],  # 正脸图片路径
                'train_frames': [],  # 训练集帧（7部分）
                'val_frames': []  # 验证集帧（3部分）
            }
        person_data[person_id]['samples'].append(idx)
        person_data[person_id]['front_images'].append(sample['front_image'])
        
        # 读取该样本的所有视频帧，然后7:3划分
        if not sample.get('is_front_only', False) and 'video_path' in sample:
            video_frames = dataset._get_video_frames(sample['video_path'])
            if isinstance(video_frames, list) and len(video_frames) > 0:
                # 如果是路径列表，先排序，然后每4张取1张（降采样）
                if isinstance(video_frames[0], str):
                    # 排序（按文件名）
                    sorted_frames = sorted(video_frames)
                    # 每4张取1张（降采样）
                    downsampled_frames = sorted_frames[::4]
                    
                    # 对帧进行7:3划分（先打乱，再划分）
                    random.shuffle(downsampled_frames)
                    num_train_frames = max(1, int(len(downsampled_frames) * 0.7))
                    train_frames = downsampled_frames[:num_train_frames]
                    val_frames = downsampled_frames[num_train_frames:]
                    
                    person_data[person_id]['train_frames'].extend(train_frames)
                    person_data[person_id]['val_frames'].extend(val_frames)
                    
                    print(f"  {person_id}: 读取到 {len(sorted_frames)} 个视频帧，降采样后 {len(downsampled_frames)} 个")
                    print(f"    训练集帧: {len(train_frames)}, 验证集帧: {len(val_frames)}")
                else:
                    print(f"  ⚠️ {person_id}: 视频帧是numpy数组，无法预处理（需要先提取帧到文件夹）")
            else:
                print(f"  ⚠️ {person_id}: 未读取到视频帧")
    
    print(f"\n读取完成，共 {len(person_data)} 个类别")
    for person_id, data in person_data.items():
        print(f"  {person_id}: {len(data['samples'])} 个样本, {len(data['front_images'])} 张正脸图")
        print(f"    训练集帧: {len(data['train_frames'])}, 验证集帧: {len(data['val_frames'])}")
    
    # 收集所有训练集帧（混合所有person_id的7部分帧）
    all_train_frames = []  # [(angle_frame_path, person_id), ...]
    for person_id, data in person_data.items():
        for frame in data['train_frames']:
            all_train_frames.append((frame, person_id))
    
    # 收集所有验证集帧（混合所有person_id的3部分帧）
    all_val_frames = []  # [(angle_frame_path, person_id), ...]
    for person_id, data in person_data.items():
        for frame in data['val_frames']:
            all_val_frames.append((frame, person_id))
    
    print(f"\n混合后的训练集帧总数: {len(all_train_frames)}")
    print(f"混合后的验证集帧总数: {len(all_val_frames)}")
    
    # 收集所有正脸图
    all_front_images = []  # [(front_image_path, person_id), ...]
    for person_id, data in person_data.items():
        front_images = data['front_images']
        for front_image in front_images:
            all_front_images.append((front_image, person_id))
    
    print(f"正脸图总数: {len(all_front_images)}")
    
    # 创建训练集配对：从混合的训练集帧中任取一张与正面配对
    # 每个正脸图与训练集帧配对，可以是同一人（正样本）或不同人（负样本）
    train_pairs = []
        for front_image, front_person_id in all_front_images:
        # 随机选择一个训练集帧（可以是同一人的，也可以是不同人的）
        angle_frame, angle_person_id = random.choice(all_train_frames)
            is_positive = (angle_person_id == front_person_id)
            
        train_pairs.append({
                'front_image': front_image,
                'angle_image': angle_frame,
                'person_id': front_person_id,  # 使用正脸图所属的person_id
                'is_positive': is_positive
            })
    
    # 创建验证集配对：从混合的验证集帧中任取一张与正面配对
    val_pairs = []
    for front_image, front_person_id in all_front_images:
        # 随机选择一个验证集帧（可以是同一人的，也可以是不同人的）
        angle_frame, angle_person_id = random.choice(all_val_frames)
        is_positive = (angle_person_id == front_person_id)
        
        val_pairs.append({
            'front_image': front_image,
            'angle_image': angle_frame,
            'person_id': front_person_id,  # 使用正脸图所属的person_id
            'is_positive': is_positive
        })
    
    # 打乱训练集和验证集
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    
    print(f"\n创建配对完成:")
    print(f"  训练集配对: {len(train_pairs)} 个")
    print(f"  验证集配对: {len(val_pairs)} 个")
    
    # 分别打乱训练集和验证集
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    
    print(f"训练集: {len(train_pairs)} 个配对 ({len(set(p['person_id'] for p in train_pairs))} 个类别)")
    print(f"验证集: {len(val_pairs)} 个配对 ({len(set(p['person_id'] for p in val_pairs))} 个类别)")
    
    # 检查是否所有类别都在训练集和验证集中
    train_person_ids = set(p['person_id'] for p in train_pairs)
    val_person_ids = set(p['person_id'] for p in val_pairs)
    all_person_ids_set = set(person_data.keys())
    
    if train_person_ids != all_person_ids_set:
        print(f"⚠️  警告: 训练集缺少类别: {all_person_ids_set - train_person_ids}")
    if val_person_ids != all_person_ids_set:
        print(f"⚠️  警告: 验证集缺少类别: {all_person_ids_set - val_person_ids}")
    
    # 将配对信息传递给dataset
    dataset.train_pairs = train_pairs
    dataset.val_pairs = val_pairs
    
    # 创建训练集和验证集数据集
    # 使用配对数据集，直接使用预先准备好的配对
    class PairDataset(torch.utils.data.Dataset):
        """配对数据集，使用预先准备好的配对"""
        def __init__(self, base_dataset, pairs):
            self.base_dataset = base_dataset
            self.pairs = pairs
        
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            pair = self.pairs[idx]
            return self.base_dataset.get_pair(pair)
    
    train_dataset = PairDataset(dataset, train_pairs)
    val_dataset = PairDataset(dataset, val_pairs)
    
    print(f"训练集数据集长度: {len(train_dataset)}")
    print(f"验证集数据集长度: {len(val_dataset)}")
    
    # 设置训练集使用增强
    train_dataset.base_dataset.augment = args.augment if hasattr(args, 'augment') else True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # 每个batch 16对图片
        shuffle=True,  # 打乱顺序，但确保遍历完所有训练样本
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  # 不丢弃最后一个不完整的batch，确保使用所有训练数据
    )
    
    # 检查验证集是否为空
    if len(val_pairs) == 0:
        print("⚠️  警告: 验证集为空，无法进行验证")
        print("   将跳过验证步骤，只进行训练")
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=True  # 不丢弃最后一个不完整的batch，确保使用所有验证数据
        )
    
    # 创建模型（DINO孪生网络）
    print("创建模型...")
    num_classes_for_model = dataset.num_classes if args.use_arcface else None
    print(f"  使用ArcFace: {args.use_arcface}")
    print(f"  类别数: {dataset.num_classes}")
    print(f"  模型num_classes: {num_classes_for_model}")
    model = MultiAngleFaceModel(
        dino_model_name='vit_base_patch16_224',
        feature_dim=args.feature_dim,  # 投影后的特征维度（默认512）
        num_classes=num_classes_for_model,
        freeze_backbone=args.freeze_backbone if hasattr(args, 'freeze_backbone') else False
    ).to(device)
    
    # 验证模型配置
    if args.use_arcface:
        if model.arcface_head is None:
            print(f"  ❌ 错误: 启用了ArcFace但模型未创建arcface_head")
            print(f"     num_classes={num_classes_for_model}, dataset.num_classes={dataset.num_classes}")
            raise ValueError("模型配置错误：启用了ArcFace但num_classes为None")
        else:
            print(f"  ✓ ArcFace分类头已创建，输出维度: {model.arcface_head.out_features}")
    else:
        print(f"  ⚠️  警告: 未启用ArcFace，ArcFace损失将不会计算")
        print(f"    提示: 如需使用ArcFace损失，请添加 --use_arcface 参数")
    
    # 损失函数
    criterion = CombinedLoss(
        use_arcface=args.use_arcface,
        use_contrastive=args.use_contrastive
    )
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=output_dir / 'logs')
    
    # 记录损失历史（用于绘图）
    train_loss_history: List[float] = []
    train_ce_loss_history: List[float] = []
    train_contrastive_loss_history: List[float] = []
    train_consistency_loss_history: List[float] = []
    val_accuracy_history: List[float] = []
    val_similarity_history: List[float] = []
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 验证（如果验证集不为空）
        if val_loader is not None and len(val_dataset) > 0:
            val_metrics = validate(model, val_loader, device, epoch, writer)
        else:
            val_metrics = {'accuracy': 0.0, 'avg_similarity': 0.0}
        
        # 更新学习率
        scheduler.step()
        
        # 记录损失历史（优先使用arcface_loss，兼容ce_loss）
        train_loss_history.append(train_losses['total_loss'])
        arcface_loss_value = train_losses.get('arcface_loss', train_losses.get('ce_loss', 0.0))
        train_ce_loss_history.append(arcface_loss_value)
        train_contrastive_loss_history.append(train_losses.get('contrastive_loss', 0.0))
        train_consistency_loss_history.append(train_losses.get('consistency_loss', 0.0))
        val_accuracy_history.append(val_metrics['accuracy'])
        val_similarity_history.append(val_metrics['avg_similarity'])
        
        # 打印日志
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  训练损失: {train_losses['total_loss']:.4f}")
        print(f"    - ArcFace损失: {arcface_loss_value:.4f}")
        print(f"    - 对比损失: {train_losses.get('contrastive_loss', 0.0):.4f}")
        print(f"    - 一致性损失: {train_losses.get('consistency_loss', 0.0):.4f}")
        print(f"  验证准确率: {val_metrics['accuracy']:.4f} (阈值={val_metrics.get('threshold', 0.85):.3f})")
        print(f"  平均相似度: {val_metrics['avg_similarity']:.4f}")
        if 'avg_positive_sim' in val_metrics:
            print(f"  正样本对相似度: {val_metrics['avg_positive_sim']:.4f}, 负样本对相似度: {val_metrics['avg_negative_sim']:.4f}")
        
        # 绘制并保存损失图
        plot_losses(
            train_loss_history,
            train_ce_loss_history,
            train_contrastive_loss_history,
            train_consistency_loss_history,
            val_accuracy_history,
            val_similarity_history,
            output_dir / 'loss_curves.png'
        )
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': dataset.num_classes
            }
            torch.save(checkpoint, output_dir / 'best_model.ckpt')
            print(f"  ✓ 保存最佳模型 (准确率: {best_acc:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.ckpt')
    
    writer.close()
    print("训练完成！")


if __name__ == '__main__':
    main()

