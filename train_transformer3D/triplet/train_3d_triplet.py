"""
3D增强的Transformer模型训练脚本 - 三元组损失版本
使用角度感知三元组损失训练模型，强化跨角度身份识别能力
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
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_transformer3D.triplet.dataset_triplet import TripletFaceDataset3D, create_triplet_dataloader, create_triplet_train_val_test_dataloaders
from train_transformer3D.triplet.models_3d_triplet import TransformerDecoderOnly3D_Triplet
from train_transformer3D.triplet.models_utils import set_seed, set_deterministic_mode

# 导入三元组损失（从triplet目录）
try:
    from train_transformer3D.triplet.angle_aware_loss import AngleAwareTripletLoss
    TRIPLET_LOSS_AVAILABLE = True
except ImportError:
    print("警告: 无法导入 AngleAwareTripletLoss，请确保 angle_aware_loss.py 存在")
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


def plot_triplet_training_curves(
    train_losses, 
    val_losses, 
    train_triplet_losses,
    val_triplet_losses,
    val_cosine_sims,
    save_path=None
):
    """
    绘制三元组训练曲线：包含总损失、三元组损失和余弦相似度
    
    Args:
        train_losses: 训练总损失列表
        val_losses: 验证总损失列表
        train_triplet_losses: 训练三元组损失列表
        val_triplet_losses: 验证三元组损失列表
        val_cosine_sims: 验证余弦相似度列表
        save_path: 保存路径
    """
    # 检查数据是否为空
    if not train_losses or not val_losses:
        print("警告: 数据为空，无法绘制曲线")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    # 创建子图：2行1列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== 第一个子图：总损失和余弦相似度 ==========
    # 左Y轴：绘制总损失曲线
    color_train = 'blue'
    color_val = 'red'
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('总损失值', fontsize=14, fontweight='bold', color='black')
    line1 = ax1.plot(epochs, train_losses, label='训练总损失', marker='o', linewidth=2.5, 
                     markersize=5, color=color_train, alpha=0.8)
    line2 = ax1.plot(epochs, val_losses, label='验证总损失', marker='s', linewidth=2.5, 
                     markersize=5, color=color_val, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([1, len(epochs)])
    
    # 右Y轴：绘制相似度曲线
    ax1_twin = ax1.twinx()
    color_sim = 'green'
    ax1_twin.set_ylabel('余弦相似度', fontsize=14, fontweight='bold', color=color_sim)
    if val_cosine_sims:
        line3 = ax1_twin.plot(epochs, val_cosine_sims, label='验证余弦相似度', marker='^', 
                             linewidth=2.5, markersize=5, color=color_sim, alpha=0.8, linestyle='-.')
        ax1_twin.tick_params(axis='y', labelcolor=color_sim)
        ax1_twin.set_ylim([0, 1])
        lines1 = [line1[0], line2[0], line3[0]]
    else:
        lines1 = [line1[0], line2[0]]
    
    labels1 = [l.get_label() for l in lines1]
    ax1.legend(lines1, labels1, loc='upper left', fontsize=11, framealpha=0.9)
    ax1.set_title('总损失和验证相似度变化曲线', fontsize=14, fontweight='bold', pad=10)
    
    # ========== 第二个子图：三元组损失 ==========
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('三元组损失值', fontsize=14, fontweight='bold', color='black')
    if train_triplet_losses:
        line4 = ax2.plot(epochs, train_triplet_losses, label='训练三元组损失', marker='o', 
                        linewidth=2.5, markersize=5, color='purple', alpha=0.8)
    if val_triplet_losses:
        line5 = ax2.plot(epochs, val_triplet_losses, label='验证三元组损失', marker='s', 
                        linewidth=2.5, markersize=5, color='orange', alpha=0.8)
    
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([1, len(epochs)])
    
    lines2 = []
    if train_triplet_losses:
        lines2.append(line4[0])
    if val_triplet_losses:
        lines2.append(line5[0])
    
    if lines2:
        labels2 = [l.get_label() for l in lines2]
        ax2.legend(lines2, labels2, loc='upper left', fontsize=11, framealpha=0.9)
    ax2.set_title('三元组损失变化曲线', fontsize=14, fontweight='bold', pad=10)
    
    plt.suptitle('三元组损失训练曲线', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        default_path = 'triplet_training_curves.png'
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
    num_triplets_list = []
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        keypoints_3d = batch['keypoints_3d'].to(device)
        pose = batch['pose'].to(device)
        angles = batch['angles'].to(device)
        
        # 获取身份标签（从person_name）
        # 使用自定义collate函数后，person_name是字符串列表
        person_names = batch.get('person_name', None)
        if person_names is None:
            print(f"警告: Batch {batch_idx} 没有 person_name，跳过")
            continue
        
        # 验证person_names格式
        if not isinstance(person_names, list):
            print(f"警告: Batch {batch_idx} person_name 格式错误: {type(person_names)}，跳过")
            continue
        
        if len(person_names) == 0:
            print(f"警告: Batch {batch_idx} person_name 为空，跳过")
            continue
        
        # 将人名转换为数字标签
        unique_names = list(set(person_names))
        if len(unique_names) < 2:
            # 三元组损失需要至少2个不同身份
            print(f"警告: Batch {batch_idx} 只有 {len(unique_names)} 个身份，跳过（三元组需要>=2）")
            continue
        
        name_to_label = {name: idx for idx, name in enumerate(unique_names)}
        labels = torch.tensor([name_to_label[name] for name in person_names], 
                              device=device, dtype=torch.long)
        
        # 梯度累积：每 accumulation_steps 步清零一次
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # 前向传播
        if use_amp and scaler is not None:
            with autocast():
                # 模型返回身份特征（用于三元组损失）
                identity_features, residual = model(
                    src=src,
                    angles=angles,
                    keypoints_3d=keypoints_3d,
                    pose=pose,
                    return_residual=False  # 三元组损失只需要身份特征
                )
                
                # 计算三元组损失
                loss, loss_dict = criterion(
                    features=identity_features,
                    labels=labels,
                    angles=pose,  # 使用pose作为角度
                    features_orig=src  # 原始侧面特征（用于重建损失）
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
            # 模型返回身份特征（用于三元组损失）
            identity_features, residual = model(
                src=src,
                angles=angles,
                keypoints_3d=keypoints_3d,
                pose=pose,
                return_residual=False  # 三元组损失只需要身份特征
            )
            
            # 计算三元组损失
            loss, loss_dict = criterion(
                features=identity_features,
                labels=labels,
                angles=pose,  # 使用pose作为角度
                features_orig=src  # 原始侧面特征（用于重建损失）
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
                # 梯度裁剪
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
            'triplets': f'{loss_dict.get("num_triplets", 0)}',
            'lr': f'{current_lr:.2e}'
        })
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/TripletLoss', loss_dict.get('triplet_loss', 0.0), global_step)
            writer.add_scalar('Train/ReconstructionLoss', loss_dict.get('reconstruction_loss', 0.0), global_step)
            writer.add_scalar('Train/NumTriplets', loss_dict.get('num_triplets', 0), global_step)
            writer.add_scalar('Train/LearningRate', current_lr, global_step)
            if 'avg_pos_dist' in loss_dict:
                writer.add_scalar('Train/AvgPosDist', loss_dict['avg_pos_dist'], global_step)
            if 'avg_neg_dist' in loss_dict:
                writer.add_scalar('Train/AvgNegDist', loss_dict['avg_neg_dist'], global_step)
    
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
    total_reconstruction_loss = 0.0
    total_cosine_sim = 0.0
    num_triplets_list = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            # 移动到设备
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            keypoints_3d = batch['keypoints_3d'].to(device)
            pose = batch['pose'].to(device)
            angles = batch['angles'].to(device)
            
            # 获取身份标签
            person_names = batch.get('person_name', None)
            if person_names is None or not isinstance(person_names, list) or len(person_names) == 0:
                continue
            
            # 将人名转换为数字标签
            unique_names = list(set(person_names))
            if len(unique_names) < 2:
                # 三元组损失需要至少2个不同身份
                continue
            
            name_to_label = {name: idx for idx, name in enumerate(unique_names)}
            labels = torch.tensor([name_to_label[name] for name in person_names], 
                                  device=device, dtype=torch.long)
            
            # 前向传播
            if use_amp:
                with autocast():
                    identity_features, residual = model(
                        src=src,
                        angles=angles,
                        keypoints_3d=keypoints_3d,
                        pose=pose,
                        return_residual=False
                    )
                    
                    loss, loss_dict = criterion(
                        features=identity_features,
                        labels=labels,
                        angles=pose,
                        features_orig=src
                    )
            else:
                identity_features, residual = model(
                    src=src,
                    angles=angles,
                    keypoints_3d=keypoints_3d,
                    pose=pose,
                    return_residual=False
                )
                
                loss, loss_dict = criterion(
                    features=identity_features,
                    labels=labels,
                    angles=pose,
                    features_orig=src
                )
            
            # 计算余弦相似度（身份特征 vs 目标特征）
            tgt_norm = nn.functional.normalize(tgt, p=2, dim=1)
            identity_norm = nn.functional.normalize(identity_features, p=2, dim=1)
            cosine_sim = (tgt_norm * identity_norm).sum(dim=1).mean().item()
            
            total_loss += loss_dict.get('total_loss', loss.item())
            total_triplet_loss += loss_dict.get('triplet_loss', 0.0)
            total_reconstruction_loss += loss_dict.get('reconstruction_loss', 0.0)
            total_cosine_sim += cosine_sim
            num_triplets_list.append(loss_dict.get('num_triplets', 0))
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_triplet_loss = total_triplet_loss / num_batches if num_batches > 0 else 0.0
    avg_reconstruction_loss = total_reconstruction_loss / num_batches if num_batches > 0 else 0.0
    avg_cosine_sim = total_cosine_sim / num_batches if num_batches > 0 else 0.0
    avg_num_triplets = np.mean(num_triplets_list) if num_triplets_list else 0.0
    
    return {
        'total_loss': avg_loss,
        'triplet_loss': avg_triplet_loss,
        'reconstruction_loss': avg_reconstruction_loss,
        'cosine_sim': avg_cosine_sim,
        'num_triplets': avg_num_triplets
    }


def main():
    parser = argparse.ArgumentParser(description='训练3D增强的Transformer模型（三元组损失版本）')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str,
                       default='train/datas/file',
                       help='数据目录（包含front_*.npy和video_*.npy文件）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小（建议32-64，三元组损失需要足够样本）')
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
    parser.add_argument('--identity_dim', type=int, default=512,
                       help='身份特征维度')
    
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
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                       help='禁用混合精度训练')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/logs',
                       help='TensorBoard日志目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备（cuda/cpu）')
    
    args = parser.parse_args()
    
    # 检查三元组损失是否可用
    if not TRIPLET_LOSS_AVAILABLE:
        print("错误: 无法导入 AngleAwareTripletLoss")
        print("请确保 train_transformer3D/triplet/angle_aware_loss.py 存在")
        return
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 创建数据加载器（使用三元组损失专用的collate函数）
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
    print("创建模型...")
    model = TransformerDecoderOnly3D_Triplet(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pose_dim=args.pose_dim,
        identity_dim=args.identity_dim,
        return_identity_features=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 创建损失函数
    print("创建三元组损失函数...")
    criterion = AngleAwareTripletLoss(
        margin=args.margin,
        alpha=args.alpha,
        beta=args.beta,
        angle_threshold=args.angle_threshold
    )
    print(f"三元组损失参数: margin={args.margin}, alpha={args.alpha}, beta={args.beta}, angle_threshold={args.angle_threshold}")
    
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
    
    # 创建日志目录（用于保存绘图）
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练历史记录
    train_loss_history = []
    val_loss_history = []
    train_triplet_loss_history = []
    val_triplet_loss_history = []
    val_cosine_sim_history = []
    
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
        
        # 记录到TensorBoard
        writer.add_scalar('Epoch/TrainLoss', train_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/TrainTripletLoss', train_metrics['triplet_loss'], epoch)
        writer.add_scalar('Epoch/ValLoss', val_metrics['total_loss'], epoch)
        writer.add_scalar('Epoch/ValTripletLoss', val_metrics['triplet_loss'], epoch)
        writer.add_scalar('Epoch/ValCosineSim', val_metrics['cosine_sim'], epoch)
        
        # 打印结果
        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(f"  训练损失: {train_metrics['total_loss']:.4f} "
              f"(三元组: {train_metrics['triplet_loss']:.4f}, "
              f"重建: {train_metrics['reconstruction_loss']:.4f}, "
              f"三元组数: {train_metrics['num_triplets']:.1f})")
        print(f"  验证损失: {val_metrics['total_loss']:.4f} "
              f"(三元组: {val_metrics['triplet_loss']:.4f}, "
              f"重建: {val_metrics['reconstruction_loss']:.4f}, "
              f"余弦相似度: {val_metrics['cosine_sim']:.4f}, "
              f"三元组数: {val_metrics['num_triplets']:.1f})")
        
        # 记录历史
        train_loss_history.append(train_metrics['total_loss'])
        val_loss_history.append(val_metrics['total_loss'])
        train_triplet_loss_history.append(train_metrics['triplet_loss'])
        val_triplet_loss_history.append(val_metrics['triplet_loss'])
        val_cosine_sim_history.append(val_metrics['cosine_sim'])
        
        # 绘制并保存训练曲线（每10个epoch或最后一个epoch）
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            plot_path = log_dir / f'triplet_training_curves_epoch_{epoch+1}.png'
            plot_triplet_training_curves(
                train_loss_history,
                val_loss_history,
                train_triplet_loss_history,
                val_triplet_loss_history,
                val_cosine_sim_history,
                save_path=str(plot_path)
            )
        
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
            
            best_model_path = os.path.join(args.save_dir, 'best_model_triplet.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  ✓ 保存最佳模型到 {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}_triplet.pth')
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
    
    # 保存最终训练曲线
    final_plot_path = log_dir / 'triplet_training_curves_final.png'
    plot_triplet_training_curves(
        train_loss_history,
        val_loss_history,
        train_triplet_loss_history,
        val_triplet_loss_history,
        val_cosine_sim_history,
        save_path=str(final_plot_path)
    )
    
    # 保存训练历史到JSON文件
    history_path = log_dir / 'triplet_training_history.json'
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'train_triplet_loss_history': train_triplet_loss_history,
            'val_triplet_loss_history': val_triplet_loss_history,
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

