"""
Transformer模型训练脚本
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.dataset import TransformerFaceDataset, create_dataloader
from train_transformer.models import SimpleTransformerEncoder
from train_transformer.models_lightweight import AngleConditionedMLP, ResidualMLP, LightweightTransformer
from train_transformer.losses import CosineSimilarityLoss, MSELoss, CombinedLoss, ResidualAndFinalLoss
from train_transformer.utils_seed import set_seed, set_deterministic_mode

# 配置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()
    
    if system == 'Windows':
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == 'Darwin':  # macOS
        font_candidates = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        font_candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    font_set = False
    for font_name in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cosine_loss = 0.0
    total_mse_loss = 0.0
    total_final_cosine_loss = 0.0  # 初始化最终特征余弦损失
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        input_features = batch['input_features'].to(device)
        angles = batch['position_encoding'].to(device)
        target_residual = batch['target_residual'].to(device)  # 残差（训练目标）
        target_features = batch['target_features'].to(device)  # 完整特征（用于验证）
        
        # 前向传播：模型预测残差
        optimizer.zero_grad()
        predicted_residual = model(input_features, angles, return_residual=True)
        
        # 应用残差得到矫正后的特征
        corrected_features = input_features + predicted_residual
        
        # 计算损失（改进版：同时优化残差和最终特征）
        if isinstance(criterion, ResidualAndFinalLoss):
            # 使用改进的损失函数：同时优化残差和最终特征
            loss_dict = criterion(
                predicted_residual, target_residual,
                corrected_features, target_features
            )
            loss = loss_dict['total_loss']
            cosine_loss = (loss_dict['residual_cosine_loss'] + loss_dict['final_cosine_loss']).item() / 2
            mse_loss = (loss_dict['residual_mse_loss'] + loss_dict['final_mse_loss']).item() / 2
            final_cosine_loss = loss_dict['final_cosine_loss'].item()
        elif isinstance(criterion, CombinedLoss):
            # 传统损失函数：只优化残差
            loss_dict = criterion(predicted_residual, target_residual)
            loss = loss_dict['total_loss']
            cosine_loss = loss_dict['cosine_loss'].item()
            mse_loss = loss_dict['mse_loss'].item()
            final_cosine_loss = 0.0
        else:
            # 单一损失函数
            loss = criterion(predicted_residual, target_residual)
            cosine_loss = 0.0
            mse_loss = 0.0
            final_cosine_loss = 0.0
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        total_cosine_loss += cosine_loss
        total_mse_loss += mse_loss
        total_final_cosine_loss += final_cosine_loss  # 总是累加（如果不是ResidualAndFinalLoss则为0）
        num_batches += 1
        
        # 更新进度条
        if isinstance(criterion, ResidualAndFinalLoss):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'residual': f'{loss_dict["residual_loss"].item():.4f}',
                'final': f'{loss_dict["final_loss"].item():.4f}',
                'final_cos': f'{final_cosine_loss:.4f}'
            })
        elif isinstance(criterion, CombinedLoss):
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cosine': f'{cosine_loss:.4f}',
                'mse': f'{mse_loss:.4f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_cosine_loss = total_cosine_loss / num_batches if num_batches > 0 else 0.0
    avg_mse_loss = total_mse_loss / num_batches if num_batches > 0 else 0.0
    avg_final_cosine_loss = total_final_cosine_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'cosine_loss': avg_cosine_loss,
        'mse_loss': avg_mse_loss,
        'final_cosine_loss': avg_final_cosine_loss
    }


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_cosine_loss = 0.0
    total_mse_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_features = batch['input_features'].to(device)
            angles = batch['position_encoding'].to(device)
            target_residual = batch['target_residual'].to(device)  # 残差（训练目标）
            target_features = batch['target_features'].to(device)  # 完整特征（用于验证）
            
            # 前向传播：模型预测残差
            predicted_residual = model(input_features, angles, return_residual=True)
            
            # 应用残差得到矫正后的特征
            corrected_features = input_features + predicted_residual
            
            # 计算损失（改进版：同时优化残差和最终特征）
            if isinstance(criterion, ResidualAndFinalLoss):
                # 使用改进的损失函数：同时优化残差和最终特征
                loss_dict = criterion(
                    predicted_residual, target_residual,
                    corrected_features, target_features
                )
                loss = loss_dict['total_loss']
                cosine_loss = (loss_dict['residual_cosine_loss'] + loss_dict['final_cosine_loss']).item() / 2
                mse_loss = (loss_dict['residual_mse_loss'] + loss_dict['final_mse_loss']).item() / 2
            elif isinstance(criterion, CombinedLoss):
                # 传统损失函数：只优化残差
                loss_dict = criterion(predicted_residual, target_residual)
                loss = loss_dict['total_loss']
                cosine_loss = loss_dict['cosine_loss'].item()
                mse_loss = loss_dict['mse_loss'].item()
            else:
                # 单一损失函数
                loss = criterion(predicted_residual, target_residual)
                cosine_loss = 0.0
                mse_loss = 0.0
            
            # 计算余弦相似度（矫正后的特征 vs 目标特征）
            pred_norm = nn.functional.normalize(corrected_features, p=2, dim=1)
            target_norm = nn.functional.normalize(target_features, p=2, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean().item()
            
            total_loss += loss.item()
            total_cosine_loss += cosine_loss
            total_mse_loss += mse_loss
            total_cosine_sim += cosine_sim
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_cosine_loss = total_cosine_loss / num_batches if num_batches > 0 else 0.0
    avg_mse_loss = total_mse_loss / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'cosine_loss': avg_cosine_loss,
        'mse_loss': avg_mse_loss,
        'cosine_similarity': avg_cosine_sim
    }


def plot_training_curves(train_losses, val_losses, train_cosine_losses=None, 
                         val_cosine_losses=None, train_mse_losses=None, 
                         val_mse_losses=None, val_cosine_sims=None, save_path=None):
    """绘制训练曲线（支持中文）"""
    num_metrics = 1
    if train_cosine_losses is not None:
        num_metrics += 2
    if val_cosine_sims is not None:
        num_metrics += 1
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]
    
    idx = 0
    
    # 总损失
    ax = axes[idx]
    ax.plot(train_losses, label='训练损失', marker='o', linewidth=2)
    ax.plot(val_losses, label='验证损失', marker='s', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('损失', fontsize=12)
    ax.set_title('训练和验证损失', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    idx += 1
    
    # 余弦相似度损失
    if train_cosine_losses is not None:
        ax = axes[idx]
        ax.plot(train_cosine_losses, label='训练余弦损失', marker='o', linewidth=2)
        if val_cosine_losses is not None:
            ax.plot(val_cosine_losses, label='验证余弦损失', marker='s', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('余弦损失', fontsize=12)
        ax.set_title('余弦相似度损失', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        idx += 1
    
    # MSE损失
    if train_mse_losses is not None:
        ax = axes[idx]
        ax.plot(train_mse_losses, label='训练MSE损失', marker='o', linewidth=2)
        if val_mse_losses is not None:
            ax.plot(val_mse_losses, label='验证MSE损失', marker='s', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MSE损失', fontsize=12)
        ax.set_title('MSE损失', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        idx += 1
    
    # 验证余弦相似度
    if val_cosine_sims is not None:
        ax = axes[idx]
        ax.plot(val_cosine_sims, label='验证余弦相似度', marker='s', linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('余弦相似度', fontsize=12)
        ax.set_title('验证余弦相似度', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def save_checkpoint(model, optimizer, epoch, loss, save_path, best=False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best': best
    }
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--features_224_dir', type=str, 
                       default=r'C:\Codes\face000\features_224',
                       help='features_224特征库目录')
    parser.add_argument('--video_dir', type=str,
                       default=r'C:\Codes\face000\train\datas\video',
                       help='视频帧图片目录')
    parser.add_argument('--face_dir', type=str,
                       default=r'C:\Codes\face000\train\datas\face',
                       help='正面图片目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--loss_type', type=str, default='residual_and_final',
                       choices=['cosine', 'mse', 'combined', 'residual_and_final'],
                       help='损失函数类型')
    parser.add_argument('--cosine_weight', type=float, default=0.5,
                       help='余弦损失权重（仅combined和residual_and_final损失）')
    parser.add_argument('--mse_weight', type=float, default=0.5,
                       help='MSE损失权重（仅combined和residual_and_final损失）')
    parser.add_argument('--residual_weight', type=float, default=0.5,
                       help='残差损失权重（仅residual_and_final损失）')
    parser.add_argument('--final_weight', type=float, default=0.5,
                       help='最终特征损失权重（仅residual_and_final损失）')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='是否使用学习率调度器')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='学习率调度器类型')
    parser.add_argument('--d_model', type=int, default=768,
                       help='模型维度')
    parser.add_argument('--nhead', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='编码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout比率')
    parser.add_argument('--model_type', type=str, default='lightweight_transformer',
                       choices=['transformer', 'lightweight_transformer', 'mlp', 'residual_mlp'],
                       help='模型类型：transformer(标准), lightweight_transformer(轻量级Transformer), mlp(MLP), residual_mlp(残差MLP)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 768],
                       help='MLP隐藏层维度列表（仅用于mlp类型）')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='ResidualMLP隐藏层维度（仅用于residual_mlp类型）')
    parser.add_argument('--num_residual_blocks', type=int, default=3,
                       help='ResidualMLP残差块数量（仅用于residual_mlp类型）')
    parser.add_argument('--save_dir', type=str, default='train_transformer/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer/logs',
                       help='日志保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--use_cpu', action='store_true',
                       help='使用CPU训练')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--valid_images_file', type=str, 
                       default='train_transformer/valid_images.json',
                       help='有效图片列表文件路径（如果提供，将从文件读取图片路径）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--deterministic', action='store_true',
                       help='启用确定性模式（完全可重复，但可能影响性能）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.deterministic:
        set_deterministic_mode(args.seed, deterministic=True)
    else:
        set_seed(args.seed)
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device('cpu' if args.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("创建数据集...")
    dataset = TransformerFaceDataset(
        features_224_dir=args.features_224_dir,
        video_dir=args.video_dir,
        face_dir=args.face_dir,
        valid_images_file=args.valid_images_file,
        use_cpu=args.use_cpu,
        cache_features=True
    )
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(dataset_size), [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 创建子数据集
    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 创建DataLoader（直接使用DataLoader，不使用create_dataloader）
    def collate_fn(batch):
        input_features = torch.stack([item['input_features'] for item in batch])
        position_encoding = torch.stack([item['position_encoding'] for item in batch])
        target_features = torch.stack([item['target_features'] for item in batch])
        target_residual = torch.stack([item['target_residual'] for item in batch])
        person_names = [item['person_name'] for item in batch]
        return {
            'input_features': input_features,
            'position_encoding': position_encoding,
            'target_features': target_features,
            'target_residual': target_residual,
            'person_names': person_names
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # 创建模型
    print("创建模型...")
    print(f"模型类型: {args.model_type}")
    
    if args.model_type == 'transformer':
        # 标准Transformer
        model = SimpleTransformerEncoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            use_angle_pe=True,
            use_angle_conditioning=True,
            angle_dim=5
        ).to(device)
    elif args.model_type == 'lightweight_transformer':
        # 轻量级Transformer（默认）
        model = LightweightTransformer(
            d_model=args.d_model,
            nhead=4,  # 减少注意力头数
            num_layers=2,  # 减少层数
            dim_feedforward=1024,  # 减少前馈网络维度
            dropout=args.dropout,
            use_angle_conditioning=True,
            angle_dim=5
        ).to(device)
    elif args.model_type == 'mlp':
        # MLP模型
        model = AngleConditionedMLP(
            input_dim=args.d_model,
            hidden_dims=args.hidden_dims,
            output_dim=args.d_model,
            use_angle_conditioning=True,
            angle_dim=5,
            dropout=args.dropout
        ).to(device)
    elif args.model_type == 'residual_mlp':
        # 残差MLP模型
        model = ResidualMLP(
            input_dim=args.d_model,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_residual_blocks,
            output_dim=args.d_model,
            use_angle_conditioning=True,
            angle_dim=5,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"未知的模型类型: {args.model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建损失函数
    if args.loss_type == 'cosine':
        criterion = CosineSimilarityLoss()
    elif args.loss_type == 'mse':
        criterion = MSELoss()
    elif args.loss_type == 'combined':
        criterion = CombinedLoss(
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight
        )
    else:  # residual_and_final（改进版）
        criterion = ResidualAndFinalLoss(
            residual_weight=args.residual_weight,
            final_weight=args.final_weight,
            cosine_weight=args.cosine_weight,
            mse_weight=args.mse_weight
        )
        print(f"使用改进的损失函数：残差权重={args.residual_weight}, 最终特征权重={args.final_weight}")
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = None
    if args.use_scheduler:
        if args.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        elif args.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=args.epochs // 3, gamma=0.5
            )
        else:  # plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        print(f"使用学习率调度器: {args.scheduler_type}")
    else:
        # 默认使用ReduceLROnPlateau（不通过参数控制）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        print(f"从epoch {start_epoch}恢复训练")
    
    # 训练历史
    train_losses = []
    val_losses = []
    train_cosine_losses = []
    val_cosine_losses = []
    train_mse_losses = []
    val_mse_losses = []
    val_cosine_sims = []
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        train_losses.append(train_metrics['loss'])
        if train_metrics['cosine_loss'] > 0:
            train_cosine_losses.append(train_metrics['cosine_loss'])
        if train_metrics['mse_loss'] > 0:
            train_mse_losses.append(train_metrics['mse_loss'])
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        val_losses.append(val_metrics['loss'])
        if val_metrics['cosine_loss'] > 0:
            val_cosine_losses.append(val_metrics['cosine_loss'])
        if val_metrics['mse_loss'] > 0:
            val_mse_losses.append(val_metrics['mse_loss'])
        if val_metrics['cosine_similarity'] > 0:
            val_cosine_sims.append(val_metrics['cosine_similarity'])
        
        # 学习率调度
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
        if train_metrics['cosine_loss'] > 0:
            writer.add_scalar('CosineLoss/Train', train_metrics['cosine_loss'], epoch)
        if val_metrics['cosine_loss'] > 0:
            writer.add_scalar('CosineLoss/Val', val_metrics['cosine_loss'], epoch)
        if val_metrics['cosine_similarity'] > 0:
            writer.add_scalar('CosineSimilarity/Val', val_metrics['cosine_similarity'], epoch)
        if train_metrics.get('final_cosine_loss', 0) > 0:
            writer.add_scalar('FinalCosineLoss/Train', train_metrics['final_cosine_loss'], epoch)
        
        # 打印指标
        print(f"\n训练损失: {train_metrics['loss']:.6f}")
        if train_metrics['cosine_loss'] > 0:
            print(f"训练余弦损失: {train_metrics['cosine_loss']:.6f}")
        if train_metrics['mse_loss'] > 0:
            print(f"训练MSE损失: {train_metrics['mse_loss']:.6f}")
        if train_metrics.get('final_cosine_loss', 0) > 0:
            print(f"训练最终特征余弦损失: {train_metrics['final_cosine_loss']:.6f}")
        print(f"验证损失: {val_metrics['loss']:.6f}")
        if val_metrics['cosine_loss'] > 0:
            print(f"验证余弦损失: {val_metrics['cosine_loss']:.6f}")
        if val_metrics['mse_loss'] > 0:
            print(f"验证MSE损失: {val_metrics['mse_loss']:.6f}")
        if val_metrics['cosine_similarity'] > 0:
            print(f"验证余弦相似度: {val_metrics['cosine_similarity']:.6f}")
        
        # 保存检查点
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        save_checkpoint(model, optimizer, epoch, val_metrics['loss'], checkpoint_path)
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_path = save_dir / 'best_model.pth'
            save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_model_path, best=True)
            print(f"✓ 保存最佳模型 (验证损失: {best_val_loss:.6f})")
        
        # 每10个epoch保存一次训练曲线
        if (epoch + 1) % 10 == 0:
            plot_path = log_dir / f'training_curves_epoch_{epoch+1}.png'
            plot_training_curves(
                train_losses, val_losses,
                train_cosine_losses if len(train_cosine_losses) > 0 else None,
                val_cosine_losses if len(val_cosine_losses) > 0 else None,
                train_mse_losses if len(train_mse_losses) > 0 else None,
                val_mse_losses if len(val_mse_losses) > 0 else None,
                val_cosine_sims if len(val_cosine_sims) > 0 else None,
                save_path=str(plot_path)
            )
    
    # 保存最终训练曲线
    final_plot_path = log_dir / 'training_curves_final.png'
    plot_training_curves(
        train_losses, val_losses,
        train_cosine_losses if len(train_cosine_losses) > 0 else None,
        val_cosine_losses if len(val_cosine_losses) > 0 else None,
        train_mse_losses if len(train_mse_losses) > 0 else None,
        val_mse_losses if len(val_mse_losses) > 0 else None,
        val_cosine_sims if len(val_cosine_sims) > 0 else None,
        save_path=str(final_plot_path)
    )
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_cosine_losses': train_cosine_losses,
        'val_cosine_losses': val_cosine_losses,
        'train_mse_losses': train_mse_losses,
        'val_mse_losses': val_mse_losses,
        'val_cosine_sims': val_cosine_sims
    }
    history_path = log_dir / 'training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n训练历史已保存到: {history_path}")
    
    writer.close()
    print("\n训练完成！")


if __name__ == "__main__":
    main()

