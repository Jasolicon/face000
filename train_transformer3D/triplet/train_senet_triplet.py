"""
SENet三元组网络训练脚本
使用SENetTripletNetwork，集成身份保护损失
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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time
from tqdm import tqdm
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.triplet.dataset_triplet import (
    TripletFaceDataset3D, 
    create_triplet_train_val_test_dataloaders
)
from train_transformer3D.triplet.models_senet_triplet import SENetTripletNetwork
from train_transformer3D.triplet.models_utils import set_seed, set_deterministic_mode

# 导入三元组损失
try:
    from train_transformer3D.triplet.angle_aware_loss import AngleAwareTripletLoss
    TRIPLET_LOSS_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 AngleAwareTripletLoss")
    TRIPLET_LOSS_AVAILABLE = False
    AngleAwareTripletLoss = None

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("警告: 混合精度训练不可用（需要PyTorch >= 1.6）")

# 设置随机种子
set_seed(42)

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

# 身份相关维度（从特征分析报告中获取的高相似维度）
IDENTITY_DIMS = [60, 312, 459, 217, 115, 74, 350, 113, 305, 149,  # Top 10
                 42, 88, 156, 201, 267, 298, 341, 378, 412, 445]  # 扩展Top 20


def compute_identity_preserve_loss(front_features, side_features, identity_dims, weight=0.3):
    """
    计算身份保护损失：保护高相似维度
    
    Args:
        front_features: 模型生成的正面特征 [batch, feature_dim]
        side_features: 原始侧面特征 [batch, feature_dim]
        identity_dims: 身份相关维度列表
        weight: 损失权重
    
    Returns:
        loss: 身份保护损失
    """
    if len(identity_dims) == 0:
        return torch.tensor(0.0, device=front_features.device)
    
    # 提取身份相关维度
    front_identity = front_features[:, identity_dims]  # [batch, len(identity_dims)]
    side_identity = side_features[:, identity_dims]    # [batch, len(identity_dims)]
    
    # 计算MSE损失
    identity_loss = F.mse_loss(front_identity, side_identity)
    
    return weight * identity_loss


def plot_training_curves(
    train_metrics_history,
    val_metrics_history,
    save_path=None
):
    """绘制训练曲线"""
    if not train_metrics_history or not val_metrics_history:
        print("警告: 数据为空，无法绘制曲线")
        return
    
    epochs = range(1, len(train_metrics_history) + 1)
    train_total_losses = [m['total_loss'] for m in train_metrics_history]
    train_triplet_losses = [m['triplet_loss'] for m in train_metrics_history]
    train_identity_losses = [m.get('identity_loss', 0.0) for m in train_metrics_history]
    val_total_losses = [m['total_loss'] for m in val_metrics_history]
    val_triplet_losses = [m['triplet_loss'] for m in val_metrics_history]
    val_cosine_sims = [m.get('cosine_sim', 0.0) for m in val_metrics_history]
    fusion_alphas = [m.get('fusion_alpha', 0.7) for m in train_metrics_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 总损失和余弦相似度
    ax1 = axes[0, 0]
    color_train = 'blue'
    color_val = 'red'
    color_sim = 'green'
    ax1.set_ylabel('总损失', color='black', fontsize=12)
    line1 = ax1.plot(epochs, train_total_losses, label='训练总损失', marker='o', 
                     linewidth=2, markersize=4, color=color_train, alpha=0.8)
    line2 = ax1.plot(epochs, val_total_losses, label='验证总损失', marker='s', 
                     linewidth=2, markersize=4, color=color_val, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([1, len(epochs)])
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('余弦相似度', color=color_sim, fontsize=12)
    line3 = ax2.plot(epochs, val_cosine_sims, label='验证余弦相似度', marker='^', 
                      linewidth=2, markersize=4, color=color_sim, alpha=0.8, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color_sim)
    ax2.set_ylim([0, 1])
    
    lines = [line1[0], line2[0], line3[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
    ax1.set_title('总损失与验证余弦相似度', fontsize=14, fontweight='bold')
    
    # 子图2: 三元组损失
    ax3 = axes[0, 1]
    color_train_triplet = 'purple'
    color_val_triplet = 'orange'
    ax3.set_ylabel('三元组损失', color='black', fontsize=12)
    line4 = ax3.plot(epochs, train_triplet_losses, label='训练三元组损失', marker='o', 
                     linewidth=2, markersize=4, color=color_train_triplet, alpha=0.8)
    line5 = ax3.plot(epochs, val_triplet_losses, label='验证三元组损失', marker='s', 
                     linewidth=2, markersize=4, color=color_val_triplet, alpha=0.8)
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax3.set_title('三元组损失', fontsize=14, fontweight='bold')
    
    # 子图3: 身份保护损失
    ax4 = axes[1, 0]
    color_identity = 'brown'
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('身份保护损失', color='black', fontsize=12)
    line6 = ax4.plot(epochs, train_identity_losses, label='训练身份保护损失', marker='o', 
                     linewidth=2, markersize=4, color=color_identity, alpha=0.8)
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax4.set_title('身份保护损失', fontsize=14, fontweight='bold')
    
    # 子图4: 融合权重
    ax5 = axes[1, 1]
    color_alpha = 'teal'
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('融合权重 (α)', color='black', fontsize=12)
    line7 = ax5.plot(epochs, fusion_alphas, label='身份分支权重 (α)', marker='o', 
                     linewidth=2, markersize=4, color=color_alpha, alpha=0.8)
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='平衡线 (0.5)')
    ax5.tick_params(axis='y', labelcolor='black')
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax5.set_title('SENet融合权重变化', fontsize=14, fontweight='bold')
    
    plt.suptitle('SENet三元组模型训练曲线', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        default_path = 'train_transformer3D/triplet/senet_training_curves.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {default_path}")
    
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                use_amp=False, gradient_accumulation_steps=1, scaler=None, writer=None,
                identity_preserve_weight=0.3):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_reconstruction_loss = 0.0
    total_identity_loss = 0.0
    num_batches = 0
    num_triplets_list = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        src = batch['src'].to(device)
        pose = batch['pose'].to(device)
        person_names = batch['person_name']
        
        # 将person_name转换为数值标签
        unique_names = sorted(list(set(person_names)))
        name_to_label = {name: idx for idx, name in enumerate(unique_names)}
        labels = torch.tensor([name_to_label[name] for name in person_names], 
                             dtype=torch.long, device=device)
        
        # 梯度累积：每 accumulation_steps 步清零一次
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast():
                front_features, identity_features, front_pose = model(
                    src=src,
                    pose=pose,
                    return_identity_features=True,
                    return_front_pose=True
                )
                
                # 计算三元组损失
                loss, loss_dict = criterion(
                    features=identity_features,
                    labels=labels,
                    angles=pose,
                    features_orig=src
                )
                
                # 计算身份保护损失
                identity_loss = compute_identity_preserve_loss(
                    front_features, src, IDENTITY_DIMS, weight=identity_preserve_weight
                )
                
                # 总损失
                total_batch_loss = loss + identity_loss
                
                # 检查损失异常
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf")
                    continue
                
                # 梯度累积：除以累积步数
                total_batch_loss = total_batch_loss / gradient_accumulation_steps
                
                # 反向传播
                scaler.scale(total_batch_loss).backward()
                
                # 梯度累积：每 accumulation_steps 步更新一次
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
        else:
            # 标准精度训练
            front_features, identity_features, front_pose = model(
                src=src,
                pose=pose,
                return_identity_features=True,
                return_front_pose=True
            )
            
            # 计算三元组损失
            loss, loss_dict = criterion(
                features=identity_features,
                labels=labels,
                angles=pose,
                features_orig=src
            )
            
            # 计算身份保护损失
            identity_loss = compute_identity_preserve_loss(
                front_features, src, IDENTITY_DIMS, weight=identity_preserve_weight
            )
            
            # 总损失
            total_batch_loss = loss + identity_loss
            
            # 检查损失异常
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf")
                continue
            
            # 梯度累积：除以累积步数
            total_batch_loss = total_batch_loss / gradient_accumulation_steps
            
            # 反向传播
            total_batch_loss.backward()
            
            # 梯度累积：每 accumulation_steps 步更新一次
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 记录
        total_loss += loss_dict.get('total_loss', loss.item()) * gradient_accumulation_steps
        total_triplet_loss += loss_dict.get('triplet_loss', 0.0)
        total_reconstruction_loss += loss_dict.get('reconstruction_loss', 0.0)
        total_identity_loss += identity_loss.item() * gradient_accumulation_steps
        num_triplets_list.append(loss_dict.get('num_triplets', 0))
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item() * gradient_accumulation_steps:.4f}',
            'triplet': f'{loss_dict.get("triplet_loss", 0.0):.4f}',
            'identity': f'{identity_loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', total_batch_loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/TripletLoss', loss_dict.get('triplet_loss', 0.0), global_step)
            writer.add_scalar('Train/IdentityLoss', identity_loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_triplet_loss = total_triplet_loss / num_batches if num_batches > 0 else 0.0
    avg_reconstruction_loss = total_reconstruction_loss / num_batches if num_batches > 0 else 0.0
    avg_identity_loss = total_identity_loss / num_batches if num_batches > 0 else 0.0
    avg_num_triplets = np.mean(num_triplets_list) if num_triplets_list else 0.0
    
    # 获取融合权重
    fusion_alpha = model.get_fusion_alpha()
    
    return {
        'total_loss': avg_loss,
        'triplet_loss': avg_triplet_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'identity_loss': avg_identity_loss,
        'num_triplets': avg_num_triplets,
        'fusion_alpha': fusion_alpha
    }


def validate(model, dataloader, criterion, device, use_amp=False):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='验证'):
            src = batch['src'].to(device)
            pose = batch['pose'].to(device)
            person_names = batch['person_name']
            tgt = batch.get('tgt', None)  # 原始正面特征（如果有）
            if tgt is not None:
                tgt = tgt.to(device)
            
            # 将person_name转换为数值标签
            unique_names = sorted(list(set(person_names)))
            name_to_label = {name: idx for idx, name in enumerate(unique_names)}
            labels = torch.tensor([name_to_label[name] for name in person_names], 
                                 dtype=torch.long, device=device)
            
            # 前向传播
            if use_amp:
                with autocast():
                    front_features, identity_features, front_pose = model(
                        src=src,
                        pose=pose,
                        return_identity_features=True,
                        return_front_pose=True
                    )
                    
                    loss, loss_dict = criterion(
                        features=identity_features,
                        labels=labels,
                        angles=pose,
                        features_orig=src
                    )
            else:
                front_features, identity_features, front_pose = model(
                    src=src,
                    pose=pose,
                    return_identity_features=True,
                    return_front_pose=True
                )
                
                loss, loss_dict = criterion(
                    features=identity_features,
                    labels=labels,
                    angles=pose,
                    features_orig=src
                )
            
            # 计算余弦相似度（如果有目标特征）
            if tgt is not None:
                tgt_norm = F.normalize(tgt, p=2, dim=1)
                identity_norm = F.normalize(identity_features, p=2, dim=1)
                cosine_sim = (tgt_norm * identity_norm).sum(dim=1).mean().item()
            else:
                cosine_sim = 0.0
            
            total_loss += loss_dict.get('total_loss', loss.item())
            total_triplet_loss += loss_dict.get('triplet_loss', 0.0)
            total_cosine_sim += cosine_sim
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_triplet_loss = total_triplet_loss / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        'triplet_loss': avg_triplet_loss,
        'cosine_sim': avg_cosine_sim
    }


def main():
    parser = argparse.ArgumentParser(description='训练SENet三元组网络')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--image_dim', type=int, default=512,
                       help='图像特征维度')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿势维度')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='全连接层数量')
    parser.add_argument('--se_reduction', type=int, default=16,
                       help='SE Block压缩比例')
    parser.add_argument('--fusion_alpha', type=float, default=0.7,
                       help='身份分支初始权重')
    parser.add_argument('--use_batch_stat', action='store_true',
                       help='SE Block是否使用batch统计（默认：每个样本独立）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--identity_preserve_weight', type=float, default=0.3,
                       help='身份保护损失权重')
    
    # 三元组损失参数
    parser.add_argument('--margin', type=float, default=0.3,
                       help='三元组损失margin')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='角度权重alpha')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='角度权重beta')
    parser.add_argument('--angle_threshold', type=float, default=30.0,
                       help='角度阈值（度）')
    
    # 其他参数
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/triplet/checkpoints_senet',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/triplet/logs_senet',
                       help='TensorBoard日志目录')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存模型的间隔（epoch）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建数据加载器
    print("\n加载数据...")
    train_loader, val_loader, test_loader = create_triplet_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = SENetTripletNetwork(
        image_dim=args.image_dim,
        pose_dim=args.pose_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        se_reduction=args.se_reduction,
        fusion_alpha=args.fusion_alpha,
        learnable_fusion=True,
        use_batch_stat=args.use_batch_stat
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"初始融合权重: {model.get_fusion_alpha():.4f}")
    
    # 创建损失函数
    if not TRIPLET_LOSS_AVAILABLE:
        raise RuntimeError("无法导入 AngleAwareTripletLoss")
    
    criterion = AngleAwareTripletLoss(
        margin=args.margin,
        alpha=args.alpha,
        beta=args.beta,
        angle_threshold=args.angle_threshold
    )
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = None
    if args.use_amp and AMP_AVAILABLE:
        scaler = GradScaler()
        print("使用混合精度训练")
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    train_metrics_history = []
    val_metrics_history = []
    
    if args.resume:
        print(f"\n从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_metrics_history = checkpoint.get('train_metrics_history', [])
        val_metrics_history = checkpoint.get('val_metrics_history', [])
        print(f"从 Epoch {start_epoch} 恢复训练")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_amp=args.use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scaler=scaler,
            writer=writer,
            identity_preserve_weight=args.identity_preserve_weight
        )
        train_metrics_history.append(train_metrics)
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, use_amp=args.use_amp)
        val_metrics_history.append(val_metrics)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValLoss', val_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValCosineSim', val_metrics['cosine_sim'], epoch)
        writer.add_scalar('Epoch/FusionAlpha', train_metrics['fusion_alpha'], epoch)
        
        # 打印进度
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, "
              f"三元组: {train_metrics['triplet_loss']:.4f}, "
              f"身份保护: {train_metrics['identity_loss']:.4f}, "
              f"融合权重: {train_metrics['fusion_alpha']:.4f}")
        print(f"  验证 - 总损失: {val_metrics['total_loss']:.4f}, "
              f"三元组: {val_metrics['triplet_loss']:.4f}, "
              f"余弦相似度: {val_metrics['cosine_sim']:.4f}")
        
        # 保存最佳模型
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_metrics_history': train_metrics_history,
                'val_metrics_history': val_metrics_history,
                'args': vars(args)
            }, best_model_path)
            print(f"  ✓ 保存最佳模型: {best_model_path}")
        
        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_metrics_history': train_metrics_history,
                'val_metrics_history': val_metrics_history,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✓ 保存检查点: {checkpoint_path}")
            
            # 绘制训练曲线
            plot_path = os.path.join(args.save_dir, f'training_curves_epoch_{epoch+1}.png')
            plot_training_curves(train_metrics_history, val_metrics_history, plot_path)
    
    # 最终保存
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'train_metrics_history': train_metrics_history,
        'val_metrics_history': val_metrics_history,
        'args': vars(args)
    }, final_model_path)
    print(f"\n✓ 保存最终模型: {final_model_path}")
    
    # 保存训练历史
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_metrics': train_metrics_history,
            'val_metrics': val_metrics_history,
            'args': vars(args)
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存训练历史: {history_path}")
    
    # 绘制最终训练曲线
    final_plot_path = os.path.join(args.save_dir, 'final_training_curves.png')
    plot_training_curves(train_metrics_history, val_metrics_history, final_plot_path)
    
    writer.close()
    print("\n训练完成！")


if __name__ == '__main__':
    main()

