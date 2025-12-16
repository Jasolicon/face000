"""
增强版3D Transformer训练脚本
集成LoRA注意力、3D投影层、跨视角注意力和先验保护
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
    create_triplet_dataloader, 
    create_triplet_train_val_test_dataloaders
)
from train_transformer3D.viewdiff import (
    EnhancedTransformerDecoderOnly3D,
    EnhancedTransformerWithPrior
)
from train_transformer3D.triplet.models_utils import set_seed, set_deterministic_mode

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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                use_amp=False, gradient_accumulation_steps=1, scaler=None, writer=None):
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
        
        # 梯度累积：每 accumulation_steps 步清零一次
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast():
                output = model(
                    src=src,
                    angles=angles,
                    keypoints_3d=keypoints_3d,
                    pose=pose,
                    return_residual=True
                )
                
                loss = criterion(output, tgt)
            
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
            output = model(
                src=src,
                angles=angles,
                keypoints_3d=keypoints_3d,
                pose=pose,
                return_residual=True
            )
            
            loss = criterion(output, tgt)
            
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
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {'total_loss': avg_loss}


def validate(model, dataloader, criterion, device, use_amp=False):
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
            if use_amp:
                with autocast():
                    output = model(
                        src=src,
                        angles=angles,
                        keypoints_3d=keypoints_3d,
                        pose=pose,
                        return_residual=True
                    )
                    loss = criterion(output, tgt)
            else:
                output = model(
                    src=src,
                    angles=angles,
                    keypoints_3d=keypoints_3d,
                    pose=pose,
                    return_residual=True
                )
                loss = criterion(output, tgt)
            
            # 计算余弦相似度
            tgt_norm = F.normalize(tgt, p=2, dim=1)
            output_norm = F.normalize(output, p=2, dim=1)
            cosine_sim = (tgt_norm * output_norm).sum(dim=1).mean().item()
            
            total_loss += loss.item()
            total_cosine_sim += cosine_sim
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    
    return {
        'total_loss': avg_loss,
        'cosine_sim': avg_cosine_sim
    }


def main():
    parser = argparse.ArgumentParser(description='训练增强版3D Transformer模型（ViewDiff风格）')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
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
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout比率')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿态维度')
    
    # ViewDiff增强参数
    parser.add_argument('--use_lora_attention', action='store_true',
                       help='使用LoRA注意力')
    parser.add_argument('--use_projection_layer', action='store_true',
                       help='使用3D投影层')
    parser.add_argument('--use_cross_view', action='store_true',
                       help='使用跨视角注意力')
    parser.add_argument('--n_views', type=int, default=5,
                       help='视角数量（如果使用跨视角）')
    parser.add_argument('--rank', type=int, default=4,
                       help='LoRA秩')
    parser.add_argument('--lora_alpha', type=float, default=1.0,
                       help='LoRA缩放因子')
    parser.add_argument('--use_prior_preservation', action='store_true',
                       help='使用先验保护训练')
    parser.add_argument('--lambda_prior', type=float, default=0.1,
                       help='先验保护权重')
    parser.add_argument('--base_model_path', type=str, default=None,
                       help='基础模型路径（用于先验保护）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--lora_lr', type=float, default=1e-3,
                       help='LoRA参数学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    parser.add_argument('--use_amp', action='store_true',
                       help='使用混合精度训练')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/logs_enhanced',
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
    print("创建增强版模型...")
    model = EnhancedTransformerDecoderOnly3D(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_keypoints=5,
        pose_dim=args.pose_dim,
        use_lora_attention=args.use_lora_attention,
        use_projection_layer=args.use_projection_layer,
        use_cross_view=args.use_cross_view,
        n_views=args.n_views,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        use_pose_attention=not args.use_lora_attention  # 如果不用LoRA，使用原始注意力
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建基础模型（用于先验保护）
    base_model = None
    if args.use_prior_preservation:
        if args.base_model_path and Path(args.base_model_path).exists():
            print(f"加载基础模型: {args.base_model_path}")
            base_model = EnhancedTransformerDecoderOnly3D(
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                use_lora_attention=False,
                use_projection_layer=False
            ).to(device)
            checkpoint = torch.load(args.base_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                base_model.load_state_dict(checkpoint)
            base_model.eval()
        else:
            print("警告: 未提供基础模型路径，先验保护将使用当前模型初始化")
            base_model = EnhancedTransformerDecoderOnly3D(
                d_model=args.d_model,
                nhead=args.nhead,
                num_layers=args.num_layers,
                use_lora_attention=False,
                use_projection_layer=False
            ).to(device)
            # 复制当前模型权重作为基础
            base_model.load_state_dict(model.state_dict())
            base_model.eval()
    
    # 包装带先验保护的模型
    if args.use_prior_preservation and base_model is not None:
        model_with_prior = EnhancedTransformerWithPrior(
            model=model,
            base_model=base_model,
            lambda_prior=args.lambda_prior
        )
        print(f"✓ 已启用先验保护训练 (lambda={args.lambda_prior})")
    else:
        model_with_prior = None
    
    # 创建损失函数
    criterion = nn.MSELoss()
    
    # 创建优化器（可单独优化LoRA参数）
    if args.use_lora_attention:
        all_params = model.get_trainable_parameters(include_base=True)
        lora_params = model.get_lora_parameters()
        
        # 分离LoRA参数和基础参数
        base_params = [p for p in all_params if p not in lora_params]
        
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': args.lr},
            {'params': lora_params, 'lr': args.lora_lr}
        ], weight_decay=args.weight_decay)
        print(f"✓ 使用分层学习率: 基础参数={args.lr}, LoRA参数={args.lora_lr}")
    else:
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
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"从 Epoch {start_epoch} 继续训练")
    
    # 训练历史
    train_loss_history = []
    val_loss_history = []
    val_cosine_sim_history = []
    
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
        train_loss_history.append(train_metrics['total_loss'])
        val_loss_history.append(val_metrics['total_loss'])
        val_cosine_sim_history.append(val_metrics['cosine_sim'])
        
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
            
            best_model_path = os.path.join(args.save_dir, 'best_model_enhanced.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ 保存最佳模型到 {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}_enhanced.pth')
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
    
    writer.close()
    
    # 保存训练历史
    history_path = Path(args.log_dir) / 'enhanced_training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'val_cosine_sim_history': val_cosine_sim_history,
            'total_epochs': len(train_loss_history),
            'best_val_loss': best_val_loss,
            'final_val_cosine_sim': val_cosine_sim_history[-1] if val_cosine_sim_history else 0.0
        }, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

