"""
通用人脸姿态不变网络训练脚本
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.dataset import create_train_val_test_dataloaders
from train_transformer3D.models_universal import UniversalFaceTransformer
from train_transformer3D.losses_universal import UniversalFaceLoss
from train_transformer3D.utils_seed import set_seed


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, 
                use_amp=False, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {
        'total': 0.0,
        'id_similarity': 0.0,
        'pose': 0.0,
        'ortho': 0.0,
        'contrast': 0.0,
        'reconstruction': 0.0,
        'pose_consistency': 0.0,
        'similarity_protection': 0.0  # 新增：相似度保护损失
    }
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        src = batch['src'].to(device)  # 侧面特征
        tgt = batch['tgt'].to(device)  # 正面特征
        pose = batch['pose'].to(device)  # 姿态角度
        person_names = batch['person_name']
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = model(src, pose, mode='train')
                
                # 准备目标
                targets = {
                    'tgt_features': tgt,
                    'pose_labels': pose,
                    'person_names': person_names
                }
                
                # 计算损失
                losses = criterion(outputs, targets, model)
                loss = losses['total']
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 前向传播
            outputs = model(src, pose, mode='train')
            
            # 准备目标
            targets = {
                'tgt_features': tgt,
                'pose_labels': pose,
                'person_names': person_names
            }
            
            # 计算损失
            losses = criterion(outputs, targets, model)
            loss = losses['total']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item() if isinstance(losses[key], torch.Tensor) else losses[key]
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'id': f'{losses.get("id_similarity", 0).item() if isinstance(losses.get("id_similarity", 0), torch.Tensor) else losses.get("id_similarity", 0):.4f}',
            'pose': f'{losses.get("pose", 0).item() if isinstance(losses.get("pose", 0), torch.Tensor) else losses.get("pose", 0):.4f}'
        })
        
        # 记录到TensorBoard（改进：降低记录频率，减少"时间倒流"视觉效果）
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            
            # 高方差损失：降低记录频率（减少视觉混乱）
            high_variance_losses = ['contrast', 'pose_consistency']
            
            # 普通损失：每10个batch记录一次
            if batch_idx % 10 == 0 or batch_idx == len(dataloader) - 1:
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor):
                        # 高方差损失：每50个batch记录一次
                        if any(hv in key.lower() for hv in high_variance_losses):
                            if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
                                writer.add_scalar(f'Train/Loss_{key}', value.item(), global_step)
                        else:
                            writer.add_scalar(f'Train/Loss_{key}', value.item(), global_step)
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for key in loss_components:
        loss_components[key] = loss_components[key] / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, loss_components


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_cosine_sim = 0.0
    loss_components = {
        'total': 0.0,
        'id_similarity': 0.0,
        'pose': 0.0,
        'ortho': 0.0,
        'contrast': 0.0,
        'reconstruction': 0.0,
        'similarity_protection': 0.0  # 新增：相似度保护损失
    }
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            # 移动到设备
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            pose = batch['pose'].to(device)
            person_names = batch['person_name']
            
            # 前向传播
            outputs = model(src, pose, mode='train')
            
            # 准备目标
            targets = {
                'tgt_features': tgt,
                'pose_labels': pose,
                'person_names': person_names
            }
            
            # 计算损失
            losses = criterion(outputs, targets, model)
            loss = losses['total']
            
            # 计算余弦相似度（使用512维特征）
            # 将id_features投影回feat_dim空间，然后与tgt计算余弦相似度
            tgt_features = tgt  # [batch, feat_dim=512]
            # 将id_features投影回512维
            id_features_512 = criterion.id_to_feat_proj(outputs['id_features'])  # [batch, id_dim=256] -> [batch, feat_dim=512]
            # 归一化
            tgt_features_norm = F.normalize(tgt_features, dim=1)  # [batch, 512]
            id_features_512_norm = F.normalize(id_features_512, dim=1)  # [batch, 512]
            # 计算余弦相似度
            cosine_sim = F.cosine_similarity(id_features_512_norm, tgt_features_norm, dim=1).mean()
            
            total_loss += loss.item()
            total_cosine_sim += cosine_sim.item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item() if isinstance(losses[key], torch.Tensor) else losses[key]
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    for key in loss_components:
        loss_components[key] = loss_components[key] / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_cosine_sim, loss_components


def main():
    parser = argparse.ArgumentParser(description='训练通用人脸姿态不变网络')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=512,
                       help='特征维度（InsightFace: 512）')
    parser.add_argument('--id_dim', type=int, default=256,
                       help='身份特征维度')
    parser.add_argument('--pose_dim', type=int, default=128,
                       help='姿态特征维度')
    parser.add_argument('--num_pose_bins', type=int, default=36,
                       help='姿态原型数量')
    parser.add_argument('--transformer_depth', type=int, default=6,
                       help='Transformer深度')
    parser.add_argument('--transformer_heads', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--transformer_mlp_dim', type=int, default=1024,
                       help='Transformer MLP维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    parser.add_argument('--use_clip_pose_encoder', action='store_true',
                       help='使用CLIP编码姿态信息（实验性功能）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    
    # 损失权重（改进：调整默认权重，更重视身份相似度）
    parser.add_argument('--lambda_id', type=float, default=3.0,
                       help='身份相似度损失权重（提高以保护身份信息，从2.0提升到3.0）')
    parser.add_argument('--lambda_pose', type=float, default=0.2,
                       help='姿态估计损失权重（降低以避免冲突，从0.3降到0.2）')
    parser.add_argument('--lambda_ortho', type=float, default=0.05,
                       help='正交约束损失权重（降低以避免过度约束）')
    parser.add_argument('--lambda_contrast', type=float, default=0.1,
                       help='对比学习损失权重（降低以减少波动，从0.2降到0.1）')
    parser.add_argument('--lambda_reconstruction', type=float, default=0.5,
                       help='重建损失权重（提高以保护特征质量）')
    parser.add_argument('--lambda_similarity_protection', type=float, default=1.0,
                       help='相似度保护损失权重（增加以稳定训练，从0.5提升到1.0）')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比学习温度参数（从0.07增加到0.1，使对比学习更平滑）')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/checkpoints_universal',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/logs_universal',
                       help='TensorBoard日志目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--min_yaw_angle', type=float, default=None,
                       help='最小yaw角度阈值（度）')
    parser.add_argument('--max_yaw_angle', type=float, default=None,
                       help='最大yaw角度阈值（度）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志目录（带时间戳）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path(args.log_dir)
    base_log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base_log_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard日志目录: {log_dir}")
    
    # 创建数据加载器
    print(f"加载数据目录: {args.data_dir}")
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        load_in_memory=True,
        train_ratio=0.6,
        val_ratio=0.3,
        test_ratio=0.1,
        random_seed=42,
        min_yaw_angle=args.min_yaw_angle,
        max_yaw_angle=args.max_yaw_angle
    )
    
    print(f"✓ 训练集: {len(train_loader.dataset)} 个样本")
    print(f"✓ 验证集: {len(val_loader.dataset)} 个样本")
    print(f"✓ 测试集: {len(test_loader.dataset)} 个样本")
    
    # 创建模型
    print("创建模型...")
    model = UniversalFaceTransformer(
        feat_dim=args.feat_dim,
        id_dim=args.id_dim,
        pose_dim=args.pose_dim,
        num_pose_bins=args.num_pose_bins,
        transformer_depth=args.transformer_depth,
        transformer_heads=args.transformer_heads,
        transformer_mlp_dim=args.transformer_mlp_dim,
        dropout=args.dropout,
        use_clip_pose_encoder=args.use_clip_pose_encoder,
        device=str(device)
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    criterion = UniversalFaceLoss(
        lambda_id=args.lambda_id,
        lambda_pose=args.lambda_pose,
        lambda_ortho=args.lambda_ortho,
        lambda_contrast=args.lambda_contrast,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_similarity_protection=args.lambda_similarity_protection,
        temperature=args.temperature,
        id_dim=args.id_dim,
        feat_dim=args.feat_dim
    ).to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器（改进：使用ReduceLROnPlateau，根据验证集表现自适应调整）
    # 注意：需要在验证后调用 scheduler.step(val_cosine_sim)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # 监控余弦相似度（越大越好）
        factor=0.5,  # 学习率减半
        patience=10,  # 10个epoch没有改善就降低学习率
        min_lr=1e-6  # 最小学习率
    )   
    
    # 混合精度训练
    use_amp = args.use_mixed_precision and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_cosine = 0.0
    
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"从epoch {start_epoch}恢复训练")
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss, train_loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            use_amp=use_amp, scaler=scaler
        )
        
        # 验证
        val_loss, val_cosine_sim, val_loss_components = validate(
            model, val_loader, criterion, device
        )
        
        # 学习率调度（改进：使用ReduceLROnPlateau，根据验证集余弦相似度调整）
        scheduler.step(val_cosine_sim)  # 传入验证集余弦相似度
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到TensorBoard
        writer.add_scalar('Train/AvgLoss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/CosineSimilarity', val_cosine_sim, epoch)
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        for key in train_loss_components:
            writer.add_scalar(f'Train/Loss_{key}', train_loss_components[key], epoch)
        for key in val_loss_components:
            writer.add_scalar(f'Val/Loss_{key}', val_loss_components[key], epoch)
        
        # 打印信息
        print(f"\n训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证余弦相似度: {val_cosine_sim:.4f}")
        print(f"学习率: {current_lr:.6f}")
        print(f"\n训练损失组件:")
        for key, value in train_loss_components.items():
            print(f"  {key}: {value:.6f}")
        
        # 保存最佳模型
        if val_cosine_sim > best_val_cosine:
            best_val_cosine = val_cosine_sim
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_cosine': best_val_cosine,
                'args': vars(args)
            }, save_dir / 'best_model.pth')
            print(f"✓ 保存最佳模型 (验证余弦相似度: {best_val_cosine:.4f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_cosine': best_val_cosine,
                'args': vars(args)
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # 测试
    print("\n" + "="*70)
    print("测试模型...")
    print("="*70)
    test_loss, test_cosine_sim, test_loss_components = validate(
        model, test_loader, criterion, device
    )
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试余弦相似度: {test_cosine_sim:.4f}")
    
    writer.close()
    print(f"\n训练完成！最佳验证余弦相似度: {best_val_cosine:.4f}")
    print(f"模型保存在: {save_dir}")


if __name__ == "__main__":
    main()
