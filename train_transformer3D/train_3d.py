"""
3D增强的Transformer模型训练脚本
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
    project_root = Path(__file__).parent.parent
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
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.dataset import Aligned3DFaceDataset, create_dataloader
from train_transformer3D.models_3d import TransformerDecoderOnly3D
from train_transformer.losses import CosineSimilarityLoss, MSELoss, CombinedLoss
try:
    from train_transformer.angle_aware_loss import AngleAwareTripletLoss
except ImportError:
    AngleAwareTripletLoss = None
from train_transformer.utils_seed import set_seed, set_deterministic_mode

# 设置随机种子
set_seed(42)


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        keypoints_3d = batch['keypoints_3d'].to(device)
        pose = batch['pose'].to(device)
        angles = batch['angles'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(
            src=src,
            angles=angles,
            keypoints_3d=keypoints_3d,
            pose=pose,
            return_residual=False
        )
        
        # 计算损失
        loss = criterion(output, tgt)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            # 移动到设备
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            keypoints_3d = batch['keypoints_3d'].to(device)
            pose = batch['pose'].to(device)
            angles = batch['angles'].to(device)
            
            # 前向传播
            output = model(
                src=src,
                angles=angles,
                keypoints_3d=keypoints_3d,
                pose=pose,
                return_residual=False
            )
            
            # 计算损失
            loss = criterion(output, tgt)
            
            # 计算余弦相似度
            cosine_sim = torch.nn.functional.cosine_similarity(output, tgt, dim=1).mean()
            
            total_loss += loss.item()
            total_cosine_sim += cosine_sim.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_cosine_sim


def main():
    parser = argparse.ArgumentParser(description='训练3D增强的Transformer模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str,
                       default='train/datas/file',
                       help='数据目录（包含front_*.npy和video_*.npy文件）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512,
                       help='模型维度（InsightFace特征维度：512）')
    parser.add_argument('--nhead', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    parser.add_argument('--num_keypoints', type=int, default=5,
                       help='3D关键点数量')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿态维度')
    parser.add_argument('--use_spatial_attention', action='store_true',
                       help='使用空间注意力融合')
    parser.add_argument('--use_pose_attention', action='store_true',
                       help='使用姿态条件注意力')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['mse', 'cosine', 'combined'],
                       help='损失函数类型')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备（cuda/cpu）')
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/logs',
                       help='TensorBoard日志目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性模式')
    
    args = parser.parse_args()
    
    # 设置确定性模式
    if args.deterministic:
        set_deterministic_mode()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print(f"加载数据目录: {args.data_dir}")
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        load_in_memory=True
    )
    
    # 创建验证集（使用相同数据，但shuffle=False）
    val_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        load_in_memory=True
    )
    
    # 创建模型
    print("创建模型...")
    model = TransformerDecoderOnly3D(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_keypoints=args.num_keypoints,
        pose_dim=args.pose_dim,
        use_spatial_attention=args.use_spatial_attention,
        use_pose_attention=args.use_pose_attention,
        use_angle_pe=True,
        use_angle_conditioning=True
    )
    
    model = model.to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    if args.loss_type == 'mse':
        criterion = MSELoss()
    elif args.loss_type == 'cosine':
        criterion = CosineSimilarityLoss()
    else:  # combined
        criterion = CombinedLoss(mse_weight=0.5, cosine_weight=0.5)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"从epoch {start_epoch}恢复训练")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        # 验证
        val_loss, val_cosine_sim = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录到tensorboard
        writer.add_scalar('Train/AvgLoss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/CosineSimilarity', val_cosine_sim, epoch)
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证余弦相似度: {val_cosine_sim:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ 保存最佳模型: {checkpoint_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ 保存检查点: {checkpoint_path}")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    
    writer.close()


if __name__ == "__main__":
    main()
