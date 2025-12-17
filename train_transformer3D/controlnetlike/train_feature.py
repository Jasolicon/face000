"""
模型1训练脚本：FeatureAngleControlNet
特征角度控制网络训练
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.controlnetlike.feature_angle_controlnet import FeatureAngleControlNet
from train_transformer3D.controlnetlike.dataset_feature import (
    create_feature_control_train_val_test_dataloaders
)
from train_transformer3D.utils_seed import set_seed, set_deterministic_mode
from train_transformer3D.losses import MSELoss, CosineSimilarityLoss, CombinedLoss

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


def plot_training_curves(train_metrics_history, val_metrics_history, save_path=None):
    """绘制训练曲线"""
    if not train_metrics_history or not val_metrics_history:
        print("警告: 数据为空，无法绘制曲线")
        return
    
    epochs = range(1, len(train_metrics_history) + 1)
    train_losses = [m['total_loss'] for m in train_metrics_history]
    train_recon_losses = [m.get('reconstruction_loss', 0.0) for m in train_metrics_history]
    train_identity_losses = [m.get('identity_loss', 0.0) for m in train_metrics_history]
    val_losses = [m['total_loss'] for m in val_metrics_history]
    val_recon_losses = [m.get('reconstruction_loss', 0.0) for m in val_metrics_history]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 子图1: 总损失
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, label='训练总损失', marker='o', linewidth=2, markersize=4, color='blue', alpha=0.8)
    ax1.plot(epochs, val_losses, label='验证总损失', marker='s', linewidth=2, markersize=4, color='red', alpha=0.8)
    ax1.set_ylabel('总损失', fontsize=12)
    ax1.set_title('总损失', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 子图2: 组件损失
    ax2 = axes[1]
    ax2.plot(epochs, train_recon_losses, label='训练重建损失', marker='o', linewidth=2, markersize=4, color='green', alpha=0.8)
    ax2.plot(epochs, train_identity_losses, label='训练身份损失', marker='^', linewidth=2, markersize=4, color='purple', alpha=0.8)
    ax2.plot(epochs, val_recon_losses, label='验证重建损失', marker='s', linewidth=2, markersize=4, color='orange', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('组件损失', fontsize=12)
    ax2.set_title('组件损失', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('特征角度控制网络训练曲线', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        default_path = 'train_transformer3D/controlnetlike/feature_training_curves.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {default_path}")
    
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                use_amp=False, gradient_accumulation_steps=1, scaler=None, writer=None,
                identity_loss_weight=0.1):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_identity_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        src_features = batch['src_features'].to(device)
        src_pose = batch['src_pose'].to(device)
        target_angle = batch['target_angle'].to(device)
        target_features = batch['target_features'].to(device)
        
        # 梯度累积
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast():
                pred_features, identity_features = model(
                    src_features, src_pose, target_angle, return_identity=True
                )
                
                # 重建损失
                recon_loss = criterion(pred_features, target_features)
                
                # 身份一致性损失
                identity_loss = model.compute_identity_loss(
                    src_features, src_pose, target_angle, num_samples=3
                )
                
                # 总损失
                total_batch_loss = recon_loss + identity_loss_weight * identity_loss
                
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf")
                    continue
                
                total_batch_loss = total_batch_loss / gradient_accumulation_steps
                
                scaler.scale(total_batch_loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
        else:
            pred_features, identity_features = model(
                src_features, src_pose, target_angle, return_identity=True
            )
            
            # 重建损失
            recon_loss = criterion(pred_features, target_features)
            
            # 身份一致性损失
            identity_loss = model.compute_identity_loss(
                src_features, src_pose, target_angle, num_samples=3
            )
            
            # 总损失
            total_batch_loss = recon_loss + identity_loss_weight * identity_loss
            
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf")
                continue
            
            total_batch_loss = total_batch_loss / gradient_accumulation_steps
            
            total_batch_loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 记录
        total_loss += total_batch_loss.item() * gradient_accumulation_steps
        total_recon_loss += recon_loss.item()
        total_identity_loss += identity_loss.item()
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item() * gradient_accumulation_steps:.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'identity': f'{identity_loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', total_batch_loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/ReconLoss', recon_loss.item(), global_step)
            writer.add_scalar('Train/IdentityLoss', identity_loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_identity_loss = total_identity_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'identity_loss': avg_identity_loss
    }


def validate(model, dataloader, criterion, device, use_amp=False, identity_loss_weight=0.1):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_identity_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='验证'):
            src_features = batch['src_features'].to(device)
            src_pose = batch['src_pose'].to(device)
            target_angle = batch['target_angle'].to(device)
            target_features = batch['target_features'].to(device)
            
            if use_amp:
                with autocast():
                    pred_features, identity_features = model(
                        src_features, src_pose, target_angle, return_identity=True
                    )
                    recon_loss = criterion(pred_features, target_features)
                    identity_loss = model.compute_identity_loss(
                        src_features, src_pose, target_angle, num_samples=3
                    )
            else:
                pred_features, identity_features = model(
                    src_features, src_pose, target_angle, return_identity=True
                )
                recon_loss = criterion(pred_features, target_features)
                identity_loss = model.compute_identity_loss(
                    src_features, src_pose, target_angle, num_samples=3
                )
            
            total_loss += (recon_loss.item() + identity_loss_weight * identity_loss.item())
            total_recon_loss += recon_loss.item()
            total_identity_loss += identity_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_identity_loss = total_identity_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'identity_loss': avg_identity_loss
    }


def main():
    parser = argparse.ArgumentParser(description='训练特征角度控制网络')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='特征维度')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿势维度')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='隐藏层维度')
    parser.add_argument('--identity_dim', type=int, default=256,
                       help='身份特征维度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--identity_loss_weight', type=float, default=0.1,
                       help='身份损失权重')
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['mse', 'cosine', 'combined'],
                       help='损失函数类型')
    
    # 其他参数
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/controlnetlike/checkpoints_feature',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/controlnetlike/logs_feature',
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
    train_loader, val_loader, test_loader = create_feature_control_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = FeatureAngleControlNet(
        feature_dim=args.feature_dim,
        pose_dim=args.pose_dim,
        hidden_dim=args.hidden_dim,
        identity_dim=args.identity_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 创建损失函数
    if args.loss_type == 'mse':
        criterion = MSELoss()
    elif args.loss_type == 'cosine':
        criterion = CosineSimilarityLoss()
    else:
        criterion = CombinedLoss(mse_weight=0.5, cosine_weight=0.5)
    
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
            identity_loss_weight=args.identity_loss_weight
        )
        train_metrics_history.append(train_metrics)
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion, device, 
            use_amp=args.use_amp,
            identity_loss_weight=args.identity_loss_weight
        )
        val_metrics_history.append(val_metrics)
        
        # 更新学习率
        scheduler.step()
        
        # TensorBoard
        writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValLoss', val_metrics['total_loss'], epoch)
        
        # 打印进度
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, "
              f"重建: {train_metrics['reconstruction_loss']:.4f}, "
              f"身份: {train_metrics['identity_loss']:.4f}")
        print(f"  验证 - 总损失: {val_metrics['total_loss']:.4f}, "
              f"重建: {val_metrics['reconstruction_loss']:.4f}, "
              f"身份: {val_metrics['identity_loss']:.4f}")
        
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

