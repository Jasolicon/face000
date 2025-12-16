"""
单向GAN训练脚本
结合Transformer生成器和GAN判别器的训练
专注于侧面→正面特征转换
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import json
import sys
from datetime import datetime

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.cyclegan import CycleGAN
from train_transformer3D.dataset import create_train_val_test_dataloaders
from train_transformer.losses import CosineSimilarityLoss, MSELoss, CombinedLoss

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("警告: 混合精度训练不可用（需要PyTorch >= 1.6）")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard不可用")


def create_adversarial_loss(use_lsgan: bool = True):
    """
    创建对抗损失函数
    
    Args:
        use_lsgan: 是否使用LSGAN（最小二乘GAN），否则使用BCE
        
    Returns:
        loss_fn: 损失函数
    """
    if use_lsgan:
        # LSGAN: 最小二乘损失，更稳定
        def lsgan_loss(pred, target_is_real):
            if target_is_real:
                return torch.mean((pred - 1.0) ** 2)
            else:
                return torch.mean(pred ** 2)
        return lsgan_loss
    else:
        # BCE损失
        criterion = nn.BCEWithLogitsLoss()
        def bce_loss(pred, target_is_real):
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return criterion(pred, target)
        return bce_loss


def train_gan_epoch(
    model: CycleGAN,
    dataloader,
    optimizer_G: optim.Optimizer,
    optimizer_D_B: optim.Optimizer,
    adversarial_loss_fn,
    pair_loss_fn: nn.Module,
    identity_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    lambda_pair: float = 10.0,
    lambda_identity: float = 0.5,
    lambda_adversarial: float = 1.0,
    use_amp: bool = False,
    scaler: GradScaler = None,
    writer = None,
    use_lsgan: bool = True
):
    """
    训练一个epoch的单向GAN
    
    Args:
        model: GAN模型（只使用G_AB和D_B）
        dataloader: 数据加载器
        optimizer_G: 生成器优化器（G_AB）
        optimizer_D_B: 判别器B优化器（判断正面特征）
        adversarial_loss_fn: 对抗损失函数
        pair_loss_fn: 配对损失函数（生成的正面特征与真实正面特征的差异）
        identity_loss_fn: 身份损失函数
        device: 设备
        epoch: 当前epoch
        lambda_pair: 配对损失权重
        lambda_identity: 身份损失权重
        lambda_adversarial: 对抗损失权重
        use_amp: 是否使用混合精度
        scaler: 梯度缩放器
        writer: TensorBoard writer
    """
    model.train()
    
    # 统计
    total_loss_G = 0.0
    total_loss_D_B = 0.0
    total_loss_pair = 0.0
    total_loss_identity = 0.0
    total_loss_adv_G = 0.0
    
    # 准确率统计
    total_correct_D_B = 0
    total_samples_D_B = 0
    
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        side_features = batch['src'].to(device)  # 侧面特征
        front_features = batch['tgt'].to(device)  # 正面特征
        keypoints_3d = batch['keypoints_3d'].to(device)  # 侧面关键点
        pose = batch['pose'].to(device)  # 侧面姿态
        angles = batch['angles'].to(device)
        
        # 获取正面关键点和姿态（用于身份损失）
        front_keypoints_3d = batch.get('front_keypoints_3d', keypoints_3d).to(device)  # 正面关键点
        front_pose = batch.get('front_pose', pose).to(device)  # 正面姿态（通常接近[0,0,0]）
        front_angles = batch.get('front_angles', front_pose.clone()).to(device)  # 正面角度（如果dataset提供则使用，否则从pose克隆）
        
        batch_size = side_features.shape[0]
        
        # ========== 训练判别器 ==========
        # 判别器需要区分真实和生成的正面特征
        
        # 生成假特征：侧面→正面
        with torch.no_grad():
            fake_front = model.G_AB(side_features, angles, keypoints_3d, pose, return_residual=False)
        
        # 训练判别器B（正面特征）
        optimizer_D_B.zero_grad()
        if use_amp and scaler is not None:
            with autocast():
                # 真实正面特征
                pred_real_B = model.D_B(front_features)
                loss_D_B_real = adversarial_loss_fn(pred_real_B, target_is_real=True)
                
                # 生成正面特征
                pred_fake_B = model.D_B(fake_front.detach())
                loss_D_B_fake = adversarial_loss_fn(pred_fake_B, target_is_real=False)
                
                loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
                
                # 计算判别器B准确率（混合精度分支）
                if not use_lsgan:
                    # BCE损失：使用sigmoid后，阈值是0.5
                    pred_real_B_prob = torch.sigmoid(pred_real_B)
                    pred_fake_B_prob = torch.sigmoid(pred_fake_B)
                    correct_real_B = (pred_real_B_prob > 0.5).sum().item()
                    correct_fake_B = (pred_fake_B_prob < 0.5).sum().item()
                else:
                    # LSGAN：输出接近1为真实，接近0为生成
                    # 阈值应该是0，而不是0.5（因为输出未经过sigmoid）
                    correct_real_B = (pred_real_B > 0).sum().item()
                    correct_fake_B = (pred_fake_B < 0).sum().item()
                total_correct_D_B += correct_real_B + correct_fake_B
                total_samples_D_B += pred_real_B.shape[0] + pred_fake_B.shape[0]
            
            scaler.scale(loss_D_B).backward()
            scaler.step(optimizer_D_B)
        else:
            pred_real_B = model.D_B(front_features)
            loss_D_B_real = adversarial_loss_fn(pred_real_B, target_is_real=True)
            
            pred_fake_B = model.D_B(fake_front.detach())
            loss_D_B_fake = adversarial_loss_fn(pred_fake_B, target_is_real=False)
            
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            
            # 计算判别器B准确率（用于监控）
            if not use_lsgan:
                # BCE损失：使用sigmoid后，阈值是0.5
                pred_real_B_prob = torch.sigmoid(pred_real_B)
                pred_fake_B_prob = torch.sigmoid(pred_fake_B)
                correct_real_B = (pred_real_B_prob > 0.5).sum().item()
                correct_fake_B = (pred_fake_B_prob < 0.5).sum().item()
            else:
                # LSGAN：输出接近1为真实，接近0为生成
                # 阈值应该是0，而不是0.5（因为输出未经过sigmoid）
                correct_real_B = (pred_real_B > 0).sum().item()
                correct_fake_B = (pred_fake_B < 0).sum().item()
            
            total_correct_D_B += correct_real_B + correct_fake_B
            total_samples_D_B += pred_real_B.shape[0] + pred_fake_B.shape[0]
            loss_D_B.backward()
            optimizer_D_B.step()
        
        # ========== 训练生成器 ==========
        optimizer_G.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                # 生成假特征：侧面→正面
                fake_front = model.G_AB(side_features, angles, keypoints_3d, pose, return_residual=False)
                
                # 对抗损失：生成器希望判别器认为生成的正面特征是真实的
                pred_fake_B = model.D_B(fake_front)
                loss_adv_G = adversarial_loss_fn(pred_fake_B, target_is_real=True)
                
                # 配对损失：生成的正面特征应该接近真实的正面特征
                loss_pair = pair_loss_fn(fake_front, front_features)
                
                # 身份损失：正面→正面应该保持不变（可选）
                if lambda_identity > 0:
                    id_front = model.G_AB(front_features, front_angles, front_keypoints_3d, front_pose, return_residual=False)
                    loss_identity = identity_loss_fn(id_front, front_features)
                else:
                    loss_identity = torch.tensor(0.0, device=device)
                
                # 总损失
                loss_G = (
                    lambda_adversarial * loss_adv_G +
                    lambda_pair * loss_pair +
                    lambda_identity * loss_identity
                )
            
            scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(model.G_AB.parameters(), max_norm=1.0)
            scaler.step(optimizer_G)
            scaler.update()
        else:
            # 生成假特征：侧面→正面
            fake_front = model.G_AB(side_features, angles, keypoints_3d, pose, return_residual=False)
            
            # 对抗损失：生成器希望判别器认为生成的正面特征是真实的
            pred_fake_B = model.D_B(fake_front)
            loss_adv_G = adversarial_loss_fn(pred_fake_B, target_is_real=True)
            
            # 配对损失：生成的正面特征应该接近真实的正面特征
            loss_pair = pair_loss_fn(fake_front, front_features)
            
            # 身份损失
            if lambda_identity > 0:
                id_front = model.G_AB(front_features, front_angles, front_keypoints_3d, front_pose, return_residual=False)
                loss_identity = identity_loss_fn(id_front, front_features)
            else:
                loss_identity = torch.tensor(0.0, device=device)
            
            # 总损失
            loss_G = (
                lambda_adversarial * loss_adv_G +
                lambda_pair * loss_pair +
                lambda_identity * loss_identity
            )
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(model.G_AB.parameters(), max_norm=1.0)
            optimizer_G.step()
        
        # 统计
        total_loss_G += loss_G.item()
        total_loss_D_B += loss_D_B.item()
        total_loss_pair += loss_pair.item()
        total_loss_identity += loss_identity.item()
        total_loss_adv_G += loss_adv_G.item()
        num_batches += 1
        
        # 计算当前batch的准确率（用于进度条显示）
        with torch.no_grad():
            # 重新获取预测值用于准确率计算（不计算梯度）
            pred_real_B_display = model.D_B(front_features)
            pred_fake_B_display = model.D_B(fake_front.detach())
            
            # 计算准确率用于进度条显示
            if not use_lsgan:
                # BCE损失：使用sigmoid后，阈值是0.5
                pred_real_B_prob = torch.sigmoid(pred_real_B_display)
                pred_fake_B_prob = torch.sigmoid(pred_fake_B_display)
                batch_acc_D_B = ((pred_real_B_prob > 0.5).sum() + (pred_fake_B_prob < 0.5).sum()).item() / (pred_real_B_display.shape[0] + pred_fake_B_display.shape[0])
            else:
                # LSGAN：输出接近1为真实，接近0为生成
                # 阈值应该是0，而不是0.5（因为输出未经过sigmoid）
                batch_acc_D_B = ((pred_real_B_display > 0).sum() + (pred_fake_B_display < 0).sum()).item() / (pred_real_B_display.shape[0] + pred_fake_B_display.shape[0])
        
        # 更新进度条
        pbar.set_postfix({
            'G': f'{loss_G.item():.4f}',
            'D_B': f'{loss_D_B.item():.4f}',
            'Pair': f'{loss_pair.item():.4f}',
            'Acc_B': f'{batch_acc_D_B:.2%}'
        })
    
    # 平均损失
    avg_loss_G = total_loss_G / num_batches
    avg_loss_D_B = total_loss_D_B / num_batches
    avg_loss_pair = total_loss_pair / num_batches
    avg_loss_identity = total_loss_identity / num_batches
    avg_loss_adv_G = total_loss_adv_G / num_batches
    
    # 计算平均准确率
    avg_acc_D_B = total_correct_D_B / total_samples_D_B if total_samples_D_B > 0 else 0.0
    
    # 记录到TensorBoard
    if writer is not None:
        global_step = epoch * len(dataloader) + num_batches
        writer.add_scalar('Train/Loss_G', avg_loss_G, global_step)
        writer.add_scalar('Train/Loss_D_B', avg_loss_D_B, global_step)
        writer.add_scalar('Train/Loss_Pair', avg_loss_pair, global_step)
        writer.add_scalar('Train/Loss_Identity', avg_loss_identity, global_step)
        writer.add_scalar('Train/Loss_Adv_G', avg_loss_adv_G, global_step)
        writer.add_scalar('Train/Acc_D_B', avg_acc_D_B, global_step)
    
    return {
        'loss_G': avg_loss_G,
        'loss_D_B': avg_loss_D_B,
        'loss_pair': avg_loss_pair,
        'loss_identity': avg_loss_identity,
        'loss_adv_G': avg_loss_adv_G,
        'acc_D_B': avg_acc_D_B
    }


def validate_gan(
    model: CycleGAN,
    dataloader,
    adversarial_loss_fn,
    pair_loss_fn: nn.Module,
    identity_loss_fn: nn.Module,
    device: torch.device,
    lambda_pair: float = 10.0,
    lambda_identity: float = 0.5,
    lambda_adversarial: float = 1.0,
    use_lsgan: bool = True
):
    """
    验证GAN模型
    """
    model.eval()
    
    total_loss_G = 0.0
    total_loss_D_B = 0.0
    total_loss_pair = 0.0
    
    # 准确率统计
    total_correct_D_B = 0
    total_samples_D_B = 0
    
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            side_features = batch['src'].to(device)
            front_features = batch['tgt'].to(device)
            keypoints_3d = batch['keypoints_3d'].to(device)
            pose = batch['pose'].to(device)
            angles = batch['angles'].to(device)
            
            # 获取正面关键点和姿态（用于身份损失）
            front_keypoints_3d = batch.get('front_keypoints_3d', keypoints_3d).to(device)
            front_pose = batch.get('front_pose', pose).to(device)
            front_angles = batch.get('front_angles', front_pose.clone()).to(device)
            
            # 生成假特征：侧面→正面
            fake_front = model.G_AB(side_features, angles, keypoints_3d, pose, return_residual=False)
            
            # 判别器B损失
            pred_real_B = model.D_B(front_features)
            pred_fake_B = model.D_B(fake_front)
            loss_D_B = (adversarial_loss_fn(pred_real_B, target_is_real=True) +
                       adversarial_loss_fn(pred_fake_B, target_is_real=False)) * 0.5
            
            # 计算判别器B准确率
            if not use_lsgan:
                pred_real_B_prob = torch.sigmoid(pred_real_B)
                pred_fake_B_prob = torch.sigmoid(pred_fake_B)
                correct_real_B = (pred_real_B_prob > 0.5).sum().item()
                correct_fake_B = (pred_fake_B_prob < 0.5).sum().item()
            else:
                correct_real_B = (pred_real_B > 0).sum().item()
                correct_fake_B = (pred_fake_B < 0).sum().item()
            total_correct_D_B += correct_real_B + correct_fake_B
            total_samples_D_B += pred_real_B.shape[0] + pred_fake_B.shape[0]
            
            # 生成器损失
            loss_adv_G = adversarial_loss_fn(pred_fake_B, target_is_real=True)
            loss_pair = pair_loss_fn(fake_front, front_features)
            
            loss_G = lambda_adversarial * loss_adv_G + lambda_pair * loss_pair
            
            total_loss_G += loss_G.item()
            total_loss_D_B += loss_D_B.item()
            total_loss_pair += loss_pair.item()
            num_batches += 1
    
    # 计算平均准确率
    avg_acc_D_B = total_correct_D_B / total_samples_D_B if total_samples_D_B > 0 else 0.0
    
    return {
        'loss_G': total_loss_G / num_batches,
        'loss_D_B': total_loss_D_B / num_batches,
        'loss_pair': total_loss_pair / num_batches,
        'acc_D_B': avg_acc_D_B
    }


def main():
    parser = argparse.ArgumentParser(description='单向GAN训练脚本（侧面→正面特征转换）')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小（GAN训练建议较小，16-32）')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--generator_type', type=str, default='decoder_only',
                       choices=['decoder_only', 'encoder_decoder', 'angle_warping'],
                       help='生成器类型')
    parser.add_argument('--discriminator_type', type=str, default='patch',
                       choices=['simple', 'patch'],
                       help='判别器类型')
    parser.add_argument('--d_model', type=int, default=512,
                       help='模型维度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr_G', type=float, default=4e-4,
                       help='生成器学习率（建议是判别器的2-4倍）')
    parser.add_argument('--lr_D', type=float, default=1e-4,
                       help='判别器学习率（建议是生成器的1/2-1/4）')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Adam beta1参数')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Adam beta2参数')
    
    # 损失权重
    parser.add_argument('--lambda_pair', type=float, default=10.0,
                       help='配对损失权重（生成的正面特征与真实正面特征的差异）')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                       help='身份损失权重（0表示禁用）')
    parser.add_argument('--lambda_adversarial', type=float, default=2.0,
                       help='对抗损失权重（建议1.0-3.0，如果判别器太强可以增加）')
    parser.add_argument('--use_lsgan', action='store_true', default=True,
                       help='使用LSGAN损失（否则使用BCE）')
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备')
    parser.add_argument('--save_dir', type=str, default='train_transformer3D/gan_checkpoints',
                       help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='train_transformer3D/gan_logs',
                       help='TensorBoard日志目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 数据加载器
    print("加载数据...")
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        load_in_memory=True,
        train_ratio=0.6,
        val_ratio=0.3,
        test_ratio=0.1,
        random_seed=42
    )
    
    print(f"✓ 训练集: {len(train_loader.dataset)} 个样本")
    print(f"✓ 验证集: {len(val_loader.dataset)} 个样本")
    print(f"✓ 测试集: {len(test_loader.dataset)} 个样本")
    
    # 创建模型
    print("创建单向GAN模型...")
    model = CycleGAN(
        generator_type=args.generator_type,
        discriminator_type=args.discriminator_type,
        d_model=args.d_model,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        num_keypoints=5,
        pose_dim=3
    )
    
    model = model.to(device)
    
    print(f"G_AB 参数量: {sum(p.numel() for p in model.G_AB.parameters()):,}")
    print(f"D_B 参数量: {sum(p.numel() for p in model.D_B.parameters()):,}")
    print(f"总参数量: {sum(p.numel() for p in model.G_AB.parameters()) + sum(p.numel() for p in model.D_B.parameters()):,}")
    
    # 创建优化器
    optimizer_G = optim.Adam(
        model.G_AB.parameters(),
        lr=args.lr_G,
        betas=(args.beta1, args.beta2)
    )
    
    optimizer_D_B = optim.Adam(
        model.D_B.parameters(),
        lr=args.lr_D,
        betas=(args.beta1, args.beta2)
    )
    
    # 损失函数
    adversarial_loss_fn = create_adversarial_loss(use_lsgan=args.use_lsgan)
    pair_loss_fn = CombinedLoss(mse_weight=0.5, cosine_weight=0.5)
    identity_loss_fn = CombinedLoss(mse_weight=0.5, cosine_weight=0.5)
    
    # 混合精度
    use_amp = args.use_mixed_precision and AMP_AVAILABLE
    scaler = GradScaler() if use_amp else None
    
    if use_amp:
        print("✓ 启用混合精度训练")
    
    # TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"✓ TensorBoard日志: {log_dir}")
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
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
        train_metrics = train_gan_epoch(
            model=model,
            dataloader=train_loader,
            optimizer_G=optimizer_G,
            optimizer_D_B=optimizer_D_B,
            adversarial_loss_fn=adversarial_loss_fn,
            pair_loss_fn=pair_loss_fn,
            identity_loss_fn=identity_loss_fn,
            device=device,
            epoch=epoch,
            lambda_pair=args.lambda_pair,
            lambda_identity=args.lambda_identity,
            lambda_adversarial=args.lambda_adversarial,
            use_amp=use_amp,
            scaler=scaler,
            writer=writer,
            use_lsgan=args.use_lsgan
        )
        
        print(f"\n训练损失:")
        print(f"  生成器: {train_metrics['loss_G']:.4f}")
        print(f"  判别器B: {train_metrics['loss_D_B']:.4f} (准确率: {train_metrics['acc_D_B']:.2%})")
        print(f"  配对损失: {train_metrics['loss_pair']:.4f}")
        print(f"  身份损失: {train_metrics['loss_identity']:.4f}")
        
        # 验证
        val_metrics = validate_gan(
            model=model,
            dataloader=val_loader,
            adversarial_loss_fn=adversarial_loss_fn,
            pair_loss_fn=pair_loss_fn,
            identity_loss_fn=identity_loss_fn,
            device=device,
            lambda_pair=args.lambda_pair,
            lambda_identity=args.lambda_identity,
            lambda_adversarial=args.lambda_adversarial,
            use_lsgan=args.use_lsgan
        )
        
        print(f"\n验证损失:")
        print(f"  生成器: {val_metrics['loss_G']:.4f}")
        print(f"  判别器B: {val_metrics['loss_D_B']:.4f} (准确率: {val_metrics['acc_D_B']:.2%})")
        print(f"  配对损失: {val_metrics['loss_pair']:.4f}")
        
        # 记录到TensorBoard
        if writer is not None:
            writer.add_scalar('Val/Loss_G', val_metrics['loss_G'], epoch)
            writer.add_scalar('Val/Loss_D_B', val_metrics['loss_D_B'], epoch)
            writer.add_scalar('Val/Loss_Pair', val_metrics['loss_pair'], epoch)
            writer.add_scalar('Val/Acc_D_B', val_metrics['acc_D_B'], epoch)
        
        # 保存最佳模型
        if val_metrics['loss_G'] < best_val_loss:
            best_val_loss = val_metrics['loss_G']
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
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
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }, checkpoint_path)
            print(f"✓ 保存检查点: {checkpoint_path}")
    
    if writer is not None:
        writer.close()
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
