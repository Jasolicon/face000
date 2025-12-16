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

from train_transformer3D.dataset import create_train_val_test_dataloaders
from train_transformer3D.models_3d import TransformerDecoderOnly3D
from train_transformer3D.models_3d_fulltransformer import TransformerEncoderDecoder3D
from train_transformer3D.models_angle_warping import FinalRecommendedModel
from train_transformer3D.models_3d_clip import TransformerDecoderOnly3D_CLIP
from train_transformer3D.losses import CosineSimilarityLoss, MSELoss, CombinedLoss
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("警告: 混合精度训练不可用（需要PyTorch >= 1.6）")
from train_transformer3D.utils_seed import set_seed, set_deterministic_mode

# 设置随机种子
set_seed(42)

# 设置matplotlib中文字体
try:
    from font_utils import setup_chinese_font_matplotlib
    setup_chinese_font_matplotlib()
except ImportError:
    # 如果 font_utils 不可用，使用旧方法
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


def plot_training_curves(train_losses, val_losses, val_cosine_sims, save_path=None):
    """
    绘制训练曲线：在一张图片中同时绘制损失图和相似度变化图（使用双Y轴）
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        val_cosine_sims: 验证余弦相似度列表
        save_path: 保存路径
    """
    # 检查数据是否为空
    if not train_losses or not val_losses or not val_cosine_sims:
        print("警告: 数据为空，无法绘制曲线")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    # 创建单个图，使用双Y轴
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左Y轴：绘制损失曲线
    color_train = 'blue'
    color_val = 'red'
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('损失值', fontsize=14, fontweight='bold', color='black')
    line1 = ax1.plot(epochs, train_losses, label='训练损失', marker='o', linewidth=2.5, 
                     markersize=5, color=color_train, alpha=0.8)
    line2 = ax1.plot(epochs, val_losses, label='验证损失', marker='s', linewidth=2.5, 
                     markersize=5, color=color_val, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([1, len(epochs)])
    
    # 右Y轴：绘制相似度曲线
    ax2 = ax1.twinx()  # 共享X轴，创建右Y轴
    color_sim = 'green'
    ax2.set_ylabel('余弦相似度', fontsize=14, fontweight='bold', color=color_sim)
    line3 = ax2.plot(epochs, val_cosine_sims, label='验证余弦相似度', marker='^', 
                     linewidth=2.5, markersize=5, color=color_sim, alpha=0.8, linestyle='-.')
    ax2.tick_params(axis='y', labelcolor=color_sim)
    ax2.set_ylim([0, 1])
    
    # 合并图例（修复：plot返回列表，需要提取第一个元素）
    lines = [line1[0], line2[0], line3[0]]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)
    
    # 设置标题
    plt.title('训练损失和验证相似度变化曲线', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        default_path = 'training_curves.png'
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {default_path}")
    
    plt.close()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None, 
                use_amp=False, scaler=None):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        writer: TensorBoard写入器
        use_amp: 是否使用混合精度训练
        scaler: 梯度缩放器（混合精度训练时使用）
    """
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
        
        if use_amp and scaler is not None:
            # 混合精度训练
            with autocast():
                output = model(
                    src=src,
                    angles=angles,
                    keypoints_3d=keypoints_3d,
                    pose=pose,
                    return_residual=False
                )
                loss = criterion(output, tgt)
            
            # 检查损失异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf: {loss.item()}")
                continue
            
            # 反向传播（混合精度）
            scaler.scale(loss).backward()
            
            # 梯度裁剪（需要先unscale）
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
                return_residual=False
            )
            loss = criterion(output, tgt)
            
            # 检查损失异常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: Epoch {epoch}, Batch {batch_idx} 损失为 NaN 或 Inf: {loss.item()}")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # 记录
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
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
            
            # 计算损失（损失函数现在返回tensor）
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
    # 兼容旧版本参数名
    parser.add_argument('--valid_images_3d_file', type=str, default=None,
                       help='[已废弃] 旧版本参数，请使用 --data_dir')
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
                       help='[已废弃] 解码器层数，请使用 --num_decoder_layers')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout比率（Transformer推荐0.15，根据过拟合情况调整）')
    parser.add_argument('--num_keypoints', type=int, default=5,
                       help='3D关键点数量')
    parser.add_argument('--pose_dim', type=int, default=3,
                       help='姿态维度')
    parser.add_argument('--use_spatial_attention', action='store_true',
                       help='使用空间注意力融合')
    parser.add_argument('--use_pose_attention', action='store_true',
                       help='使用姿态条件注意力')
    parser.add_argument('--model_type', type=str, default='decoder_only',
                       choices=['decoder_only', 'encoder_decoder', 'angle_warping'],
                       help='模型类型：decoder_only（仅解码器）、encoder_decoder（编码器-解码器）或angle_warping（角度条件仿射变换）')
    parser.add_argument('--num_encoder_layers', type=int, default=4,
                       help='编码器层数（encoder_decoder模式）')
    parser.add_argument('--num_decoder_layers', type=int, default=4,
                       help='解码器层数（encoder_decoder模式，或decoder_only模式的层数）')
    parser.add_argument('--use_pose_pe', action='store_true', default=True,
                       help='使用姿态位置编码（encoder_decoder模式，默认True，使用--use_pose_pe启用）')
    parser.add_argument('--no_pose_pe', action='store_false', dest='use_pose_pe',
                       help='禁用姿态位置编码')
    
    # 角度条件仿射变换模型参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='隐藏层维度（angle_warping模式）')
    parser.add_argument('--num_basis', type=int, default=32,
                       help='特征基向量数量（angle_warping模式）')
    parser.add_argument('--use_basis', action='store_true', default=True,
                       help='使用特征基重建（angle_warping模式）')
    parser.add_argument('--no_basis', action='store_false', dest='use_basis',
                       help='禁用特征基重建')
    parser.add_argument('--use_refinement', action='store_true', default=True,
                       help='使用残差细化（angle_warping模式）')
    parser.add_argument('--no_refinement', action='store_false', dest='use_refinement',
                       help='禁用残差细化')
    parser.add_argument('--use_attention_refine', action='store_true', default=True,
                       help='使用自注意力精修（angle_warping模式）')
    parser.add_argument('--no_attention_refine', action='store_false', dest='use_attention_refine',
                       help='禁用自注意力精修')
    parser.add_argument('--num_attention_layers', type=int, default=1,
                       help='自注意力层数（angle_warping模式）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数（已优化：从100增加到150，突破平台期）')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='峰值学习率（配合warmup使用，Transformer推荐1e-4到2e-4）')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='学习率预热轮数（Transformer最佳实践：5-10 epochs，已优化为10）')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                       help='预热起始学习率（从很小的值开始）')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'step'],
                       help='学习率调度器类型（推荐cosine，Transformer最佳实践）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减（Transformer推荐1e-4）')
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['mse', 'cosine', 'combined'],
                       help='损失函数类型')
    parser.add_argument('--mse_weight', type=float, default=0.1,
                       help='MSE损失权重（combined模式，默认0.1，更关注相似度）')
    parser.add_argument('--cosine_weight', type=float, default=0.9,
                       help='余弦损失权重（combined模式，默认0.9，更关注特征方向）')
    # ReduceLROnPlateau参数（当scheduler_type='plateau'时使用）
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                       help='学习率衰减因子（plateau模式，默认0.5）')
    parser.add_argument('--scheduler_patience', type=int, default=7,
                       help='学习率调度器耐心值（plateau模式，默认7，给模型更多时间）')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7,
                       help='学习率最小值')
    # Cosine Annealing参数
    parser.add_argument('--cosine_eta_min', type=float, default=1e-5,
                       help='Cosine退火最小学习率（已优化：从1e-6提高到1e-5，保持探索能力）')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='使用混合精度训练（FP16，加速训练）')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                       help='早停耐心值（验证损失连续N个epoch不改善则停止，0表示禁用）')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-5,
                       help='早停最小改善阈值')
    
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
    parser.add_argument('--min_yaw_angle', type=float, default=None,
                        help='最小yaw角度阈值（度），如果设置，只保留 |yaw| >= min_yaw_angle 的样本（排除接近正面的图片）。例如：--min_yaw_angle 15 表示只保留yaw角度绝对值>=15°的样本')
    parser.add_argument('--max_yaw_angle', type=float, default=None,
                        help='最大yaw角度阈值（度），如果设置，只保留 |yaw| <= max_yaw_angle 的样本')
    
    args = parser.parse_args()
    
    # 兼容旧版本参数：如果提供了 valid_images_3d_file，转换为 data_dir
    if args.valid_images_3d_file is not None:
        import warnings
        warnings.warn(
            f"--valid_images_3d_file 参数已废弃，请使用 --data_dir。"
            f"当前将忽略 --valid_images_3d_file={args.valid_images_3d_file}，使用 --data_dir={args.data_dir}",
            DeprecationWarning
        )
    
    # 设置确定性模式
    if args.deterministic:
        set_deterministic_mode()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每次训练创建带时间戳的子目录，避免不同运行的数据叠加
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path(args.log_dir)
    base_log_dir.mkdir(parents=True, exist_ok=True)
    log_dir = base_log_dir / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard日志目录: {log_dir}")
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 根据设备类型决定是否使用pin_memory
    pin_memory = True  # 默认值，将在创建dataloader时根据设备调整
    
    # 打印CUDA详细信息
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        if device.type == 'cuda':
            print(f"当前GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("警告: CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 创建数据加载器（按6:3:1分割）
    print(f"加载数据目录: {args.data_dir}")
    print("按person_name分割数据集: 训练集60% / 验证集30% / 测试集10%")
    # 根据设备类型决定是否使用pin_memory
    use_pin_memory = pin_memory and device.type == 'cuda'
    
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
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
    
    # 显示角度过滤信息
    if args.min_yaw_angle is not None or args.max_yaw_angle is not None:
        print(f"\n角度过滤设置:")
        if args.min_yaw_angle is not None:
            print(f"  最小yaw角度阈值: {args.min_yaw_angle}° (只保留 |yaw| >= {abs(args.min_yaw_angle)}° 的样本)")
        if args.max_yaw_angle is not None:
            print(f"  最大yaw角度阈值: {args.max_yaw_angle}° (只保留 |yaw| <= {abs(args.max_yaw_angle)}° 的样本)")
    
    # 创建模型
    print("创建模型...")
    if args.model_type == 'encoder_decoder':
        print("使用编码器-解码器架构（TransformerEncoderDecoder3D）")
        model = TransformerEncoderDecoder3D(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_keypoints=args.num_keypoints,
            pose_dim=args.pose_dim,
            use_pose_pe=args.use_pose_pe,
            use_angle_conditioning=True
        )
    elif args.model_type == 'angle_warping':
        print("使用角度条件仿射变换架构（FinalRecommendedModel）")
        model = FinalRecommendedModel(
            d_model=args.d_model,
            hidden_dim=args.hidden_dim,
            num_basis=args.num_basis,
            use_basis=args.use_basis,
            use_refinement=args.use_refinement,
            use_attention_refine=args.use_attention_refine,
            num_attention_layers=args.num_attention_layers
        )
    elif args.model_type == 'decoder_only_clip':
        print("使用CLIP增强的解码器架构（TransformerDecoderOnly3D_CLIP）")
        model = TransformerDecoderOnly3D_CLIP(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            num_keypoints=args.num_keypoints,
            pose_dim=args.pose_dim,
            use_pose_attention=args.use_pose_attention,
            use_angle_pe=True,
            use_angle_conditioning=True,
            use_clip_pose_encoder=True,  # 使用CLIP编码姿态
            device=str(device)
        )
    else:  # decoder_only
        print("使用仅解码器架构（TransformerDecoderOnly3D）")
        model = TransformerDecoderOnly3D(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_decoder_layers,  # 使用num_decoder_layers参数
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
        # 优化：提高余弦相似度权重，更关注特征方向而非数值差异
        criterion = CombinedLoss(mse_weight=args.mse_weight, cosine_weight=args.cosine_weight)
        print(f"使用组合损失: MSE权重={args.mse_weight:.2f}, 余弦权重={args.cosine_weight:.2f}")
    
    # 创建优化器（AdamW是Transformer的标准选择）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,  # 峰值学习率，实际会通过warmup逐步增加
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),  # Transformer标准配置
        eps=1e-8
    )
    
    # 学习率调度器（Transformer最佳实践：Warmup + Cosine Annealing）
    scheduler = None
    warmup_scheduler = None
    
    # 1. Warmup阶段：线性增加学习率（Transformer训练的关键技巧）
    if args.warmup_epochs > 0:
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                # 线性从warmup_lr增加到lr
                return args.warmup_lr / args.lr + (1 - args.warmup_lr / args.lr) * (epoch / args.warmup_epochs)
            else:
                return 1.0
        
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        print(f"✓ 启用学习率预热: {args.warmup_epochs} epochs (从 {args.warmup_lr:.2e} 到 {args.lr:.2e})")
    
    # 2. 主调度器：Warmup后的学习率衰减策略
    if args.scheduler_type == 'cosine':
        # Cosine Annealing（Transformer推荐，平滑衰减）
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),  # 在warmup之后生效
            eta_min=args.cosine_eta_min
        )
        print(f"✓ 使用Cosine退火调度器: T_max={args.epochs - args.warmup_epochs}, eta_min={args.cosine_eta_min:.2e}")
        
        # 组合warmup和cosine调度器
        if warmup_scheduler is not None and args.warmup_epochs > 0:
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )
            print(f"✓ 组合调度器: Warmup ({args.warmup_epochs} epochs) → Cosine Annealing")
        else:
            scheduler = main_scheduler
            
    elif args.scheduler_type == 'plateau':
        # ReduceLROnPlateau（基于验证损失）
        main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.scheduler_min_lr,
            verbose=True
        )
        print(f"✓ 使用Plateau调度器: factor={args.scheduler_factor}, patience={args.scheduler_patience}")
        
        # 组合warmup和plateau调度器
        if warmup_scheduler is not None and args.warmup_epochs > 0:
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )
            print(f"✓ 组合调度器: Warmup ({args.warmup_epochs} epochs) → Plateau")
        else:
            scheduler = main_scheduler
            
    elif args.scheduler_type == 'step':
        # StepLR（固定步长衰减）
        step_size = max(1, (args.epochs - args.warmup_epochs) // 3)
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=args.scheduler_factor
        )
        print(f"✓ 使用Step调度器: step_size={step_size}, gamma={args.scheduler_factor}")
        
        if warmup_scheduler is not None and args.warmup_epochs > 0:
            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[args.warmup_epochs]
            )
            print(f"✓ 组合调度器: Warmup ({args.warmup_epochs} epochs) → Step")
        else:
            scheduler = main_scheduler
    else:
        # 默认：只有warmup，没有后续衰减
        scheduler = warmup_scheduler if warmup_scheduler is not None else None
        if scheduler is None:
            print("⚠️  警告: 未启用任何学习率调度器")
    
    # 混合精度训练
    use_amp = args.use_mixed_precision and AMP_AVAILABLE and device.type == 'cuda'
    scaler = None
    if use_amp:
        scaler = GradScaler()
        print("✓ 启用混合精度训练（FP16）")
    else:
        if args.use_mixed_precision:
            print("⚠️  混合精度训练请求但不可用（需要CUDA和PyTorch >= 1.6）")
    
    # 早停机制
    early_stopping_enabled = args.early_stopping_patience > 0
    if early_stopping_enabled:
        print(f"✓ 启用早停机制: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
    best_val_loss_for_early_stop = float('inf')
    epochs_without_improvement = 0
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 训练历史记录
    train_loss_history = []
    val_loss_history = []
    val_cosine_sim_history = []
    
    if args.resume:
        print(f"恢复训练: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # 恢复历史记录（如果存在）
            train_loss_history = checkpoint.get('train_loss_history', [])
            val_loss_history = checkpoint.get('val_loss_history', [])
            val_cosine_sim_history = checkpoint.get('val_cosine_sim_history', [])
            # 恢复学习率调度器状态（如果存在）
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("✓ 恢复学习率调度器状态")
                except Exception as e:
                    print(f"⚠️  恢复学习率调度器状态失败: {e}，将使用新的调度器")
            # 恢复混合精度scaler状态（如果存在）
            if use_amp and scaler is not None and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✓ 恢复混合精度scaler状态")
            # 恢复早停状态（如果存在）
            if early_stopping_enabled and 'best_val_loss_for_early_stop' in checkpoint:
                best_val_loss_for_early_stop = checkpoint['best_val_loss_for_early_stop']
                epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
                print(f"✓ 恢复早停状态: best_val_loss={best_val_loss_for_early_stop:.4f}, epochs_without_improvement={epochs_without_improvement}")
            print(f"从epoch {start_epoch}恢复训练")
        except Exception as e:
            print(f"❌ 恢复训练失败: {e}")
            print("将从头开始训练")
            start_epoch = 0
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            use_amp=use_amp, scaler=scaler
        )
        
        # 验证
        val_loss, val_cosine_sim = validate(model, val_loader, criterion, device)
        
        # 测试（仅在最后一个epoch或每10个epoch）
        if (epoch + 1) == args.epochs or (epoch + 1) % 10 == 0:
            test_loss, test_cosine_sim = validate(model, test_loader, criterion, device)
            writer.add_scalar('Test/Loss', test_loss, epoch)
            writer.add_scalar('Test/CosineSimilarity', test_cosine_sim, epoch)
            print(f"测试损失: {test_loss:.4f}")
            print(f"测试余弦相似度: {test_cosine_sim:.4f}")
        
        # 学习率调度（根据调度器类型选择）
        if scheduler is not None:
            if args.scheduler_type == 'plateau':
                # Plateau调度器需要验证损失
                scheduler.step(val_loss)
            else:
                # Cosine/Step/Warmup调度器按epoch更新
                scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到tensorboard
        writer.add_scalar('Train/AvgLoss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/CosineSimilarity', val_cosine_sim, epoch)
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        print(f"\n训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证余弦相似度: {val_cosine_sim:.4f}")
        print(f"当前学习率: {current_lr:.2e}")
        
        # 计算训练/验证损失比（用于判断过拟合）
        if val_loss > 0:
            overfit_ratio = train_loss / val_loss
            if overfit_ratio < 0.8:
                print(f"⚠️  可能过拟合: 训练损失/验证损失 = {overfit_ratio:.3f} (建议 > 0.9)")
            elif overfit_ratio > 1.2:
                print(f"ℹ️  可能欠拟合: 训练损失/验证损失 = {overfit_ratio:.3f} (建议 < 1.1)")
        
        # 早停检查
        if early_stopping_enabled:
            if val_loss < best_val_loss_for_early_stop - args.early_stopping_min_delta:
                best_val_loss_for_early_stop = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n{'='*70}")
                print(f"早停触发: 验证损失连续 {args.early_stopping_patience} 个epoch未改善")
                print(f"最佳验证损失: {best_val_loss_for_early_stop:.4f}")
                print(f"{'='*70}")
                break
        
        # 记录历史
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_cosine_sim_history.append(val_cosine_sim)
        
        # 绘制并保存训练曲线（每5个epoch或最后一个epoch）
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            plot_path = log_dir / f'training_curves_epoch_{epoch+1}.png'
            plot_training_curves(
                train_loss_history,
                val_loss_history,
                val_cosine_sim_history,
                save_path=str(plot_path)
            )
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = save_dir / 'best_model.pth'
            try:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': vars(args),
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'val_cosine_sim_history': val_cosine_sim_history
                }
                # 保存学习率调度器状态
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                # 保存混合精度scaler状态
                if use_amp and scaler is not None:
                    checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                # 保存早停状态
                if early_stopping_enabled:
                    checkpoint_data['best_val_loss_for_early_stop'] = best_val_loss_for_early_stop
                    checkpoint_data['epochs_without_improvement'] = epochs_without_improvement
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"✓ 保存最佳模型: {checkpoint_path}")
            except Exception as e:
                print(f"❌ 保存最佳模型失败: {e}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            try:
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': vars(args),
                    'train_loss_history': train_loss_history,
                    'val_loss_history': val_loss_history,
                    'val_cosine_sim_history': val_cosine_sim_history
                }
                # 保存学习率调度器状态
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                # 保存混合精度scaler状态
                if use_amp and scaler is not None:
                    checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                # 保存早停状态
                if early_stopping_enabled:
                    checkpoint_data['best_val_loss_for_early_stop'] = best_val_loss_for_early_stop
                    checkpoint_data['epochs_without_improvement'] = epochs_without_improvement
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"✓ 保存检查点: {checkpoint_path}")
            except Exception as e:
                print(f"❌ 保存检查点失败: {e}")
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    
    # 保存最终训练曲线
    final_plot_path = log_dir / 'training_curves_final.png'
    plot_training_curves(
        train_loss_history,
        val_loss_history,
        val_cosine_sim_history,
        save_path=str(final_plot_path)
    )
    
    # 保存训练历史到JSON文件
    history_path = log_dir / 'training_history.json'
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
    
    writer.close()


if __name__ == "__main__":
    main()
