"""
简单三元组网络训练脚本
使用SimpleTripletNetwork代替复杂的Transformer网络
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
from train_transformer3D.triplet.models_simple_triplet import SimpleTripletNetwork
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
    val_total_losses = [m['total_loss'] for m in val_metrics_history]
    val_triplet_losses = [m['triplet_loss'] for m in val_metrics_history]
    val_cosine_sims = [m['cosine_sim'] for m in val_metrics_history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 子图1：总损失和余弦相似度
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
    
    ax2_twin = ax1.twinx()
    ax2_twin.set_ylabel('余弦相似度', color=color_sim, fontsize=12)
    line3 = ax2_twin.plot(epochs, val_cosine_sims, label='验证余弦相似度', marker='^', 
                          linewidth=2, markersize=4, color=color_sim, alpha=0.8, linestyle='-.')
    ax2_twin.tick_params(axis='y', labelcolor=color_sim)
    ax2_twin.set_ylim([0, 1])
    
    lines = [line1[0], line2[0], line3[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)
    ax1.set_title('总损失与验证余弦相似度', fontsize=14, fontweight='bold')
    
    # 子图2：三元组损失
    color_train_triplet = 'purple'
    color_val_triplet = 'orange'
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('三元组损失', color='black', fontsize=12)
    line4 = ax2.plot(epochs, train_triplet_losses, label='训练三元组损失', marker='o', 
                     linewidth=2, markersize=4, color=color_train_triplet, alpha=0.8)
    line5 = ax2.plot(epochs, val_triplet_losses, label='验证三元组损失', marker='s', 
                     linewidth=2, markersize=4, color=color_val_triplet, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.set_title('三元组损失', fontsize=14, fontweight='bold')
    
    plt.suptitle('简单三元组网络训练曲线', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        default_path = 'simple_triplet_training_curves.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {default_path}")
    
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                use_amp=False, gradient_accumulation_steps=1, scaler=None, writer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_reconstruction_loss = 0.0
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
            
            # 检查损失异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf: {loss.item()}")
                continue
            
            # 梯度累积：除以累积步数
            loss = loss / gradient_accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积：每 accumulation_steps 步更新一次
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
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
            
            # 检查损失异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf: {loss.item()}")
                continue
            
            # 梯度累积：除以累积步数
            loss = loss / gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积：每 accumulation_steps 步更新一次
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # 记录
        total_loss += loss_dict.get('total_loss', loss.item()) * gradient_accumulation_steps
        total_triplet_loss += loss_dict.get('triplet_loss', 0.0)
        total_reconstruction_loss += loss_dict.get('reconstruction_loss', 0.0)
        num_triplets_list.append(loss_dict.get('num_triplets', 0))
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'triplet': f'{loss_dict.get("triplet_loss", 0.0):.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/TripletLoss', loss_dict.get('triplet_loss', 0.0), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_triplet_loss = total_triplet_loss / num_batches if num_batches > 0 else 0.0
    avg_reconstruction_loss = total_reconstruction_loss / num_batches if num_batches > 0 else 0.0
    avg_num_triplets = np.mean(num_triplets_list) if num_triplets_list else 0.0
    
    return {
        'total_loss': avg_loss,
        'triplet_loss': avg_triplet_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'num_triplets': avg_num_triplets
    }


def validate(model, dataloader, criterion, device, use_amp=False):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_triplet_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            # 移动到设备
            src = batch['src'].to(device)
            tgt = batch.get('tgt', None)
            if tgt is not None:
                tgt = tgt.to(device)
            pose = batch['pose'].to(device)
            person_names = batch['person_name']
            
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
                tgt_norm = nn.functional.normalize(tgt, p=2, dim=1)
                identity_norm = nn.functional.normalize(identity_features, p=2, dim=1)
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
    parser = argparse.ArgumentParser(description='训练简单三元组网络')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--image_dim', type=int, default=512,
                       help='图像特征维度（InsightFace: 512）')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿势维度（欧拉角: 3）')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='全连接层数量')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    
    # 三元组损失参数
    parser.add_argument('--margin', type=float, default=0.2,
                       help='三元组损失margin')
    parser.add_argument('--alpha', type=float, default=2.0,
                       help='角度权重参数alpha')
    parser.add_argument('--beta', type=float, default=1.5,
                       help='角度权重参数beta')
    parser.add_argument('--angle_threshold', type=float, default=30.0,
                       help='角度差异阈值（度）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/logs_simple_triplet',
                       help='TensorBoard日志目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建数据加载器
    print("加载数据...")
    train_loader, val_loader, test_loader = create_triplet_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("创建简单三元组网络...")
    model = SimpleTripletNetwork(
        image_dim=args.image_dim,
        pose_dim=args.pose_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建损失函数
    if not TRIPLET_LOSS_AVAILABLE:
        raise RuntimeError("无法导入 AngleAwareTripletLoss，请检查 angle_aware_loss.py")
    
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
    
    # 混合精度训练
    use_amp = args.use_amp and device.type == 'cuda' and AMP_AVAILABLE
    scaler = None
    if use_amp:
        scaler = GradScaler()
        print("✓ 已启用混合精度训练（FP16）")
    else:
        print("使用标准精度训练")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"从 Epoch {start_epoch} 继续训练")
    
    # 训练历史
    train_metrics_history = []
    val_metrics_history = []
    
    # 训练循环
    print("\n开始训练...")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_amp=use_amp,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scaler=scaler,
            writer=writer
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, use_amp=use_amp)
        
        # 记录历史
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # 记录到TensorBoard
        writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValLoss', val_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValCosineSim', val_metrics['cosine_sim'], epoch)
        
        # 打印结果
        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(f"  训练损失: {train_metrics['total_loss']:.4f}")
        print(f"  验证损失: {val_metrics['total_loss']:.4f}")
        print(f"  验证余弦相似度: {val_metrics['cosine_sim']:.4f}")
        
        # 保存最佳模型
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            best_model_path = os.path.join(args.save_dir, 'best_model_simple_triplet.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ 保存最佳模型到 {best_model_path}")
        
        # 定期保存检查点和绘制曲线
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}_simple_triplet.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics,
                'args': vars(args)
            }
            if scaler is not None:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ 保存检查点到 {checkpoint_path}")
            
            # 绘制训练曲线
            plot_path = os.path.join(args.log_dir, f'training_curves_epoch_{epoch + 1}.png')
            plot_training_curves(train_metrics_history, val_metrics_history, plot_path)
    
    writer.close()
    
    # 保存训练历史
    history_path = Path(args.log_dir) / 'simple_triplet_training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_metrics_history': train_metrics_history,
            'val_metrics_history': val_metrics_history,
            'total_epochs': len(train_metrics_history),
            'best_val_loss': best_val_loss
        }, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")
    
    # 最终绘制
    final_plot_path = os.path.join(args.log_dir, 'final_training_curves.png')
    plot_training_curves(train_metrics_history, val_metrics_history, final_plot_path)
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

