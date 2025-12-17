"""
模型2训练脚本：图像生成ControlNet
训练模型从图片生成目标角度的图片，受姿势控制
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

# 尝试导入 setup_mirrors
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
import torchvision.utils as vutils

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.controlnetlike.models_image_controlnet import ImageControlNet
from train_transformer3D.controlnetlike.dataset_image import (
    create_image_control_train_val_test_dataloaders
)
from train_transformer3D.utils_seed import set_seed, set_deterministic_mode

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("警告: 混合精度训练不可用")

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
    train_losses = [m['total_loss'] for m in train_metrics_history]
    train_image_losses = [m['image_loss'] for m in train_metrics_history]
    train_pose_losses = [m.get('pose_loss', 0.0) for m in train_metrics_history]
    val_losses = [m['total_loss'] for m in val_metrics_history]
    val_image_losses = [m['image_loss'] for m in val_metrics_history]
    val_psnrs = [m.get('psnr', 0.0) for m in val_metrics_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 子图1: 总损失
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, label='训练总损失', marker='o', linewidth=2, markersize=4)
    ax1.plot(epochs, val_losses, label='验证总损失', marker='s', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('总损失', fontsize=12)
    ax1.set_title('总损失', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 图像损失
    ax2 = axes[0, 1]
    ax2.plot(epochs, train_image_losses, label='训练图像损失', marker='o', linewidth=2, markersize=4)
    ax2.plot(epochs, val_image_losses, label='验证图像损失', marker='s', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('图像损失', fontsize=12)
    ax2.set_title('图像损失', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 姿势损失
    ax3 = axes[1, 0]
    ax3.plot(epochs, train_pose_losses, label='训练姿势损失', marker='o', linewidth=2, markersize=4)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('姿势损失', fontsize=12)
    ax3.set_title('姿势预测损失', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: PSNR
    ax4 = axes[1, 1]
    ax4.plot(epochs, val_psnrs, label='验证PSNR', marker='^', linewidth=2, markersize=4, color='green')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('PSNR (dB)', fontsize=12)
    ax4.set_title('验证PSNR', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('图像生成ControlNet训练曲线', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        default_path = 'train_transformer3D/controlnetlike/image_controlnet_training_curves.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {default_path}")
    
    plt.close()


def compute_psnr(pred, target):
    """计算PSNR"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def train_epoch(
    model, dataloader, criterion, optimizer, device, epoch,
    use_amp=False, gradient_accumulation_steps=1, scaler=None, writer=None,
    pose_loss_weight=0.1
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_image_loss = 0.0
    total_pose_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        source_image = batch['source_image'].to(device)
        target_pose = batch['target_pose'].to(device)
        target_image = batch['target_image'].to(device)
        
        # 梯度累积
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast():
                output_image, control_signal, source_pose = model(
                    image=source_image,
                    target_pose=target_pose,
                    return_control_signal=True,
                    return_source_pose=True
                )
                
                # 计算图像损失
                image_loss = criterion(output_image, target_image)
                
                # 计算姿势预测损失（辅助任务）
                # 使用真实目标姿势（如果有ground truth pose）
                pose_loss = F.mse_loss(source_pose, target_pose) * pose_loss_weight
                
                # 总损失
                total_batch_loss = image_loss + pose_loss
                
                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf")
                    continue
                
                total_batch_loss = total_batch_loss / gradient_accumulation_steps
                
                # 反向传播
                scaler.scale(total_batch_loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
        else:
            output_image, control_signal, source_pose = model(
                image=source_image,
                target_pose=target_pose,
                return_control_signal=True,
                return_source_pose=True
            )
            
            # 计算图像损失
            image_loss = criterion(output_image, target_image)
            
            # 计算姿势预测损失（辅助任务）
            pose_loss = F.mse_loss(source_pose, target_pose) * pose_loss_weight
            
            # 总损失
            total_batch_loss = image_loss + pose_loss
            
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf")
                continue
            
            total_batch_loss = total_batch_loss / gradient_accumulation_steps
            
            # 反向传播
            total_batch_loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 记录
        total_loss += total_batch_loss.item() * gradient_accumulation_steps
        total_image_loss += image_loss.item()
        total_pose_loss += pose_loss.item()
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{total_batch_loss.item() * gradient_accumulation_steps:.4f}',
            'img': f'{image_loss.item():.4f}',
            'pose': f'{pose_loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', total_batch_loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/ImageLoss', image_loss.item(), global_step)
            writer.add_scalar('Train/PoseLoss', pose_loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_image_loss = total_image_loss / num_batches if num_batches > 0 else 0.0
    avg_pose_loss = total_pose_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        'image_loss': avg_image_loss,
        'pose_loss': avg_pose_loss
    }


def validate(model, dataloader, criterion, device, use_amp=False, pose_loss_weight=0.1):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_image_loss = 0.0
    total_psnr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='验证'):
            source_image = batch['source_image'].to(device)
            target_pose = batch['target_pose'].to(device)
            target_image = batch['target_image'].to(device)
            
            # 前向传播
            if use_amp:
                with autocast():
                    output_image, _, source_pose = model(
                        image=source_image,
                        target_pose=target_pose,
                        return_control_signal=False,
                        return_source_pose=True
                    )
                    
                    image_loss = criterion(output_image, target_image)
                    pose_loss = F.mse_loss(source_pose, target_pose) * pose_loss_weight
                    total_batch_loss = image_loss + pose_loss
            else:
                output_image, _, source_pose = model(
                    image=source_image,
                    target_pose=target_pose,
                    return_control_signal=False,
                    return_source_pose=True
                )
                
                image_loss = criterion(output_image, target_image)
                pose_loss = F.mse_loss(source_pose, target_pose) * pose_loss_weight
                total_batch_loss = image_loss + pose_loss
            
            # 计算PSNR
            # 将图像范围从[-1, 1]转换到[0, 1]用于PSNR计算
            output_norm = (output_image + 1) / 2
            target_norm = (target_image + 1) / 2
            psnr = compute_psnr(output_norm, target_norm)
            
            total_loss += total_batch_loss.item()
            total_image_loss += image_loss.item()
            total_psnr += psnr
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_image_loss = total_image_loss / num_batches if num_batches > 0 else 0.0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        'image_loss': avg_image_loss,
        'psnr': avg_psnr
    }


def main():
    parser = argparse.ArgumentParser(description='训练图像生成ControlNet')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录路径')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='图片目录（如果与data_dir不同）')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小（图像生成需要更多内存）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--image_size', type=int, default=112,
                       help='图像尺寸')
    
    # 模型参数
    parser.add_argument('--feature_dim', type=int, default=512,
                       help='特征维度')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿势维度')
    parser.add_argument('--num_control_layers', type=int, default=3,
                       help='控制分支层数')
    parser.add_argument('--freeze_generator', action='store_true',
                       help='是否冻结图像生成器')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--pose_loss_weight', type=float, default=0.1,
                       help='姿势预测损失权重')
    
    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='mse',
                       choices=['mse', 'l1', 'combined'],
                       help='损失函数类型')
    
    # 其他参数
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/controlnetlike/checkpoints_image',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/controlnetlike/logs_image',
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
    train_loader, val_loader, test_loader = create_image_control_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        image_dir=args.image_dir
    )
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = ImageControlNet(
        feature_dim=args.feature_dim,
        pose_dim=args.pose_dim,
        image_size=args.image_size,
        in_channels=3,
        num_control_layers=args.num_control_layers,
        freeze_generator=args.freeze_generator
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 创建损失函数
    if args.loss_type == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_type == 'l1':
        criterion = nn.L1Loss()
    else:  # combined
        criterion = nn.MSELoss()  # 简化实现
    
    # 创建优化器
    if args.freeze_generator:
        trainable_params = model.get_trainable_parameters()
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
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
            pose_loss_weight=args.pose_loss_weight
        )
        train_metrics_history.append(train_metrics)
        
        # 验证
        val_metrics = validate(
            model, val_loader, criterion, device,
            use_amp=args.use_amp,
            pose_loss_weight=args.pose_loss_weight
        )
        val_metrics_history.append(val_metrics)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/TrainImageLoss', train_metrics['image_loss'], epoch)
        writer.add_scalar('Epoch/TrainPoseLoss', train_metrics['pose_loss'], epoch)
        writer.add_scalar('Epoch/ValLoss', val_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValImageLoss', val_metrics['image_loss'], epoch)
        writer.add_scalar('Epoch/ValPSNR', val_metrics['psnr'], epoch)
        
        # 定期可视化生成的图像（每10个epoch）
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 获取一个验证批次
                val_batch = next(iter(val_loader))
                source_image = val_batch['source_image'][:8].to(device)
                target_pose = val_batch['target_pose'][:8].to(device)
                target_image = val_batch['target_image'][:8].to(device)
                
                # 生成预测图像
                output_image, _, source_pose = model(
                    image=source_image,
                    target_pose=target_pose,
                    return_control_signal=False,
                    return_source_pose=True
                )
                
                # 将图像从[-1, 1]转换到[0, 1]用于可视化
                source_vis = (source_image + 1) / 2
                target_vis = (target_image + 1) / 2
                output_vis = (output_image + 1) / 2
                
                # 创建图像网格（每行：源图像、目标图像、生成图像）
                # 合并图像：源、目标、生成
                image_grid = torch.cat([
                    source_vis,
                    target_vis,
                    output_vis
                ], dim=0)  # [24, 3, H, W]
                
                # 创建网格图像
                grid = vutils.make_grid(image_grid, nrow=8, normalize=False, padding=2)
                writer.add_image('Images/Comparison', grid, epoch)
                
                # 单独可视化每个样本（前4个）
                for i in range(min(4, source_image.size(0))):
                    sample_grid = torch.stack([
                        source_vis[i],
                        target_vis[i],
                        output_vis[i]
                    ], dim=0)
                    grid_single = vutils.make_grid(sample_grid, nrow=3, normalize=False, padding=2)
                    writer.add_image(f'Images/Sample_{i}', grid_single, epoch)
                
                # 可视化姿势预测
                pose_diff = target_pose - source_pose
                writer.add_histogram('Pose/Target', target_pose, epoch)
                writer.add_histogram('Pose/Source', source_pose, epoch)
                writer.add_histogram('Pose/Difference', pose_diff, epoch)
            
            model.train()
        
        # 打印进度
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, "
              f"图像: {train_metrics['image_loss']:.4f}, "
              f"姿势: {train_metrics['pose_loss']:.4f}")
        print(f"  验证 - 总损失: {val_metrics['total_loss']:.4f}, "
              f"图像: {val_metrics['image_loss']:.4f}, "
              f"PSNR: {val_metrics['psnr']:.2f} dB")
        
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

