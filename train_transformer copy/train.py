"""
Transformer模型训练脚本
"""
import os
import sys
from pathlib import Path

# 在导入任何可能使用 HuggingFace 的库之前设置镜像
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # 禁用 hf_transfer
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时
    os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '5'    # 重试5次

# 尝试导入 setup_mirrors（如果存在）
try:
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from setup_mirrors import setup_all_mirrors
    setup_all_mirrors()
except ImportError:
    pass  # 如果 setup_mirrors 不存在，使用上面的默认设置

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
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.dataset import TransformerFaceDataset, create_dataloader
from train_transformer.models import SimpleTransformerEncoder
from train_transformer.models_lightweight import AngleConditionedMLP, ResidualMLP, LightweightTransformer
try:
    from train_transformer.models_decoder_only import TransformerDecoderOnly
except ImportError:
    TransformerDecoderOnly = None
from train_transformer.losses import CosineSimilarityLoss, MSELoss, CombinedLoss, ResidualAndFinalLoss
try:
    from train_transformer.angle_aware_loss import AngleAwareTripletLoss
except ImportError:
    AngleAwareTripletLoss = None
from train_transformer.utils_seed import set_seed, set_deterministic_mode

# 配置matplotlib中文字体
def setup_chinese_font():
    """设置matplotlib中文字体（支持跨平台）"""
    try:
        from font_utils import setup_chinese_font_matplotlib
        setup_chinese_font_matplotlib()
    except ImportError:
        # 如果 font_utils 不可用，使用旧方法
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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                use_amp=False, gradient_accumulation_steps=1, scaler=None):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        use_amp: 是否使用混合精度训练
        gradient_accumulation_steps: 梯度累积步数
        scaler: GradScaler（用于混合精度）
    """
    model.train()
    total_loss = 0.0
    total_cosine_loss = 0.0
    total_mse_loss = 0.0
    total_final_cosine_loss = 0.0  # 初始化最终特征余弦损失
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    optimizer.zero_grad()  # 在epoch开始时清零梯度
    
    for batch_idx, batch in enumerate(pbar):
        input_features = batch['input_features'].to(device, non_blocking=True)
        angles = batch['position_encoding'].to(device, non_blocking=True)
        target_residual = batch['target_residual'].to(device, non_blocking=True)  # 残差（训练目标）
        target_features = batch['target_features'].to(device, non_blocking=True)  # 完整特征（用于验证）
        
        # 前向传播：模型预测残差（使用混合精度）
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                predicted_residual = model(input_features, angles, return_residual=True)
                # 应用残差得到矫正后的特征
                corrected_features = input_features + predicted_residual
                
                # 计算损失
                if isinstance(criterion, ResidualAndFinalLoss):
                    loss_dict = criterion(
                        predicted_residual, target_residual,
                        corrected_features, target_features
                    )
                    loss = loss_dict['total_loss']
                    cosine_loss = (loss_dict['residual_cosine_loss'] + loss_dict['final_cosine_loss']).item() / 2
                    mse_loss = (loss_dict['residual_mse_loss'] + loss_dict['final_mse_loss']).item() / 2
                    final_cosine_loss = loss_dict['final_cosine_loss'].item()
                elif isinstance(criterion, CombinedLoss):
                    loss_dict = criterion(predicted_residual, target_residual)
                    loss = loss_dict['total_loss']
                    cosine_loss = loss_dict['cosine_loss'].item()
                    mse_loss = loss_dict['mse_loss'].item()
                    final_cosine_loss = 0.0
                elif isinstance(criterion, AngleAwareTripletLoss):
                    # 角度感知三元组损失需要身份标签
                    person_names = batch.get('person_names', None)
                    if person_names is not None:
                        # 将人名转换为数字标签
                        unique_names = list(set(person_names))
                        name_to_label = {name: idx for idx, name in enumerate(unique_names)}
                        labels = torch.tensor([name_to_label[name] for name in person_names], device=device)
                        
                        # 使用矫正后的特征和原始输入特征
                        loss, loss_dict = criterion(
                            features=corrected_features,
                            labels=labels,
                            angles=angles,
                            features_orig=input_features
                        )
                        cosine_loss = 0.0
                        mse_loss = 0.0
                        final_cosine_loss = 0.0
                    else:
                        # 如果没有person_names，回退到MSE损失
                        loss = F.mse_loss(corrected_features, target_features)
                        loss_dict = {'total_loss': loss.item(), 'num_triplets': 0}
                        cosine_loss = 0.0
                        mse_loss = loss.item()
                        final_cosine_loss = 0.0
                else:
                    loss = criterion(predicted_residual, target_residual)
                    cosine_loss = 0.0
                    mse_loss = 0.0
                    final_cosine_loss = 0.0
                
                # 梯度累积：除以累积步数
                loss = loss / gradient_accumulation_steps
        else:
            # 不使用混合精度
            predicted_residual = model(input_features, angles, return_residual=True)
            # 应用残差得到矫正后的特征
            corrected_features = input_features + predicted_residual
            
            # 计算损失
            if isinstance(criterion, ResidualAndFinalLoss):
                loss_dict = criterion(
                    predicted_residual, target_residual,
                    corrected_features, target_features
                )
                loss = loss_dict['total_loss']
                cosine_loss = (loss_dict['residual_cosine_loss'] + loss_dict['final_cosine_loss']).item() / 2
                mse_loss = (loss_dict['residual_mse_loss'] + loss_dict['final_mse_loss']).item() / 2
                final_cosine_loss = loss_dict['final_cosine_loss'].item()
            elif isinstance(criterion, CombinedLoss):
                loss_dict = criterion(predicted_residual, target_residual)
                loss = loss_dict['total_loss']
                cosine_loss = loss_dict['cosine_loss'].item()
                mse_loss = loss_dict['mse_loss'].item()
                final_cosine_loss = 0.0
            elif isinstance(criterion, AngleAwareTripletLoss):
                # 角度感知三元组损失需要身份标签
                person_names = batch.get('person_names', None)
                if person_names is not None:
                    # 将人名转换为数字标签
                    unique_names = list(set(person_names))
                    name_to_label = {name: idx for idx, name in enumerate(unique_names)}
                    labels = torch.tensor([name_to_label[name] for name in person_names], device=device)
                    
                    # 使用矫正后的特征和原始输入特征
                    loss, loss_dict = criterion(
                        features=corrected_features,
                        labels=labels,
                        angles=angles,
                        features_orig=input_features
                    )
                    cosine_loss = 0.0
                    mse_loss = 0.0
                    final_cosine_loss = 0.0
                else:
                    # 如果没有person_names，回退到MSE损失
                    loss = F.mse_loss(corrected_features, target_features)
                    loss_dict = {'total_loss': loss.item(), 'num_triplets': 0}
                    cosine_loss = 0.0
                    mse_loss = loss.item()
                    final_cosine_loss = 0.0
            else:
                loss = criterion(predicted_residual, target_residual)
                cosine_loss = 0.0
                mse_loss = 0.0
                final_cosine_loss = 0.0
            
            # 梯度累积：除以累积步数
            loss = loss / gradient_accumulation_steps
        
        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积：每 accumulation_steps 步更新一次
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                # 梯度裁剪（在scaler中）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
        
        # 记录损失（需要乘以累积步数以恢复真实损失值）
        total_loss += loss.item() * gradient_accumulation_steps
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


def validate(model, dataloader, criterion, device, use_amp=False):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        use_amp: 是否使用混合精度（验证时也可以使用以加速）
    """
    model.eval()
    total_loss = 0.0
    total_cosine_loss = 0.0
    total_mse_loss = 0.0
    total_cosine_sim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            input_features = batch['input_features'].to(device, non_blocking=True)
            angles = batch['position_encoding'].to(device, non_blocking=True)
            target_residual = batch['target_residual'].to(device, non_blocking=True)  # 残差（训练目标）
            target_features = batch['target_features'].to(device, non_blocking=True)  # 完整特征（用于验证）
            
            # 前向传播：模型预测残差（使用混合精度以加速）
            if use_amp:
                with torch.cuda.amp.autocast():
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
            else:
                # 不使用混合精度
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
                elif isinstance(criterion, AngleAwareTripletLoss):
                    # 角度感知三元组损失需要身份标签
                    person_names = batch.get('person_names', None)
                    if person_names is not None:
                        unique_names = list(set(person_names))
                        name_to_label = {name: idx for idx, name in enumerate(unique_names)}
                        labels = torch.tensor([name_to_label[name] for name in person_names], device=device)
                        loss, loss_dict = criterion(
                            features=corrected_features,
                            labels=labels,
                            angles=angles,
                            features_orig=input_features
                        )
                        cosine_loss = 0.0
                        mse_loss = 0.0
                    else:
                        loss = F.mse_loss(corrected_features, target_features)
                        cosine_loss = 0.0
                        mse_loss = loss.item()
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


def save_checkpoint(model, optimizer, epoch, loss, save_path, best=False, max_retries=3, model_type=None):
    """
    保存模型检查点（带错误处理和重试机制）
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 损失值
        save_path: 保存路径
        best: 是否为最佳模型
        max_retries: 最大重试次数
    """
    import shutil
    
    save_path = Path(save_path)
    
    # 确保保存目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查磁盘空间（至少需要500MB）
    try:
        stat = shutil.disk_usage(save_path.parent)
        free_space_gb = stat.free / (1024**3)
        if free_space_gb < 0.5:
            print(f"⚠️  警告: 磁盘空间不足 ({free_space_gb:.2f} GB)，可能无法保存模型")
    except Exception as e:
        print(f"⚠️  无法检查磁盘空间: {e}")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best': best,
        'model_type': model_type  # 保存模型类型信息
    }
    
    # 先保存到临时文件，然后重命名（原子操作）
    temp_path = save_path.with_suffix('.tmp')
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            # 如果临时文件存在，先删除
            if temp_path.exists():
                temp_path.unlink()
            
            # 保存到临时文件
            torch.save(checkpoint, temp_path)
            
            # 原子操作：重命名临时文件为最终文件
            if save_path.exists():
                save_path.unlink()  # 删除旧文件
            temp_path.rename(save_path)
            
            # 验证文件是否成功保存
            if save_path.exists() and save_path.stat().st_size > 0:
                file_size_mb = save_path.stat().st_size / (1024**2)
                print(f"✓ 模型已保存到: {save_path} ({file_size_mb:.2f} MB)")
                return True
            else:
                raise RuntimeError("保存的文件为空或不存在")
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                print(f"⚠️  保存失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"   等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 最后一次尝试失败
                error_msg = f"❌ 保存模型失败 (已重试 {max_retries} 次): {e}\n"
                error_msg += f"   保存路径: {save_path}\n"
                error_msg += f"   请检查:\n"
                error_msg += f"   1. 磁盘空间是否充足\n"
                error_msg += f"   2. 目录是否有写入权限\n"
                error_msg += f"   3. 磁盘是否有I/O错误\n"
                
                # 尝试保存到备用位置
                try:
                    backup_path = save_path.parent / f"{save_path.stem}_backup{save_path.suffix}"
                    torch.save(checkpoint, backup_path)
                    error_msg += f"   已尝试保存到备用位置: {backup_path}\n"
                except Exception as backup_e:
                    error_msg += f"   备用保存也失败: {backup_e}\n"
                
                print(error_msg)
                raise RuntimeError(error_msg)
    
    return False


def main():
    parser = argparse.ArgumentParser(description='训练Transformer模型')
    parser.add_argument('--features_224_dir', type=str, 
                       default=r'/root/face000/features_224',
                       help='features_224特征库目录')
    parser.add_argument('--video_dir', type=str,
                       default=r'/root/face000/train/datas/video',
                       help='视频帧图片目录')
    parser.add_argument('--face_dir', type=str,
                       default=r'/root/face000/train/datas/face',
                       help='正面图片目录')
    parser.add_argument('--dinov2_model', type=str,
                       default='dinov2_vitb14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='DINOv2模型名称（默认: dinov2_vitb14，768维）')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小（24GB显卡推荐128，可根据显存调整）')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='梯度累积步数（用于模拟更大的batch_size）')
    # use_amp 通过 --no_amp 控制（默认启用）
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='学习率（推荐2e-4，可根据模型大小调整）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减（推荐1e-4，防止过拟合）')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='学习率预热轮数（前N个epoch线性增加学习率）')
    parser.add_argument('--warmup_lr', type=float, default=1e-6,
                       help='预热起始学习率')
    parser.add_argument('--loss_type', type=str, default='residual_and_final',
                       choices=['cosine', 'mse', 'combined', 'residual_and_final'],
                       help='损失函数类型')
    parser.add_argument('--cosine_weight', type=float, default=0.5,
                       help='余弦损失权重（仅combined和residual_and_final损失）')
    parser.add_argument('--mse_weight', type=float, default=0.5,
                       help='MSE损失权重（仅combined和residual_and_final损失）')
    parser.add_argument('--residual_weight', type=float, default=0.6,
                       help='残差损失权重（仅residual_and_final损失，推荐0.6-0.7）')
    parser.add_argument('--final_weight', type=float, default=0.4,
                       help='最终特征损失权重（仅residual_and_final损失，推荐0.3-0.4）')
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                       help='是否使用学习率调度器（默认启用）')
    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='学习率调度器类型（推荐cosine）')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6,
                       help='Cosine调度器最小学习率')
    parser.add_argument('--scheduler_patience', type=int, default=10,
                       help='ReduceLROnPlateau调度器的patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                       help='ReduceLROnPlateau调度器的衰减因子')
    parser.add_argument('--d_model', type=int, default=None,
                       help='模型维度（如果为None，将自动从数据集检测）')
    parser.add_argument('--nhead', type=int, default=8,
                       help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='编码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout比率（推荐0.15，防止过拟合）')
    parser.add_argument('--model_type', type=str, default='lightweight_transformer',
                       choices=['transformer', 'lightweight_transformer', 'decoder_only', 'mlp', 'residual_mlp'],
                       help='模型类型：transformer(标准编码器), lightweight_transformer(轻量级编码器), decoder_only(仅解码器), mlp(MLP), residual_mlp(残差MLP)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=None,
                       help='MLP隐藏层维度列表（仅用于mlp类型，如果为None将根据d_model自动设置）')
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
    parser.add_argument('--memory_fraction', type=float, default=None,
                       help='GPU显存使用比例（0.0-1.0），None表示使用全部显存（24GB显卡建议0.9）')
    parser.add_argument('--allow_tf32', action='store_true', default=True,
                       help='允许使用TF32（TensorFloat-32）加速计算')
    parser.add_argument('--no_amp', action='store_true',
                       help='禁用混合精度训练（默认启用）')
    
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
    
    # 检查保存目录的磁盘空间和权限
    try:
        import shutil
        stat = shutil.disk_usage(save_dir)
        free_space_gb = stat.free / (1024**3)
        total_space_gb = stat.total / (1024**3)
        print(f"保存目录: {save_dir}")
        print(f"磁盘空间: {free_space_gb:.2f} GB / {total_space_gb:.2f} GB (可用/总计)")
        if free_space_gb < 1.0:
            print(f"⚠️  警告: 可用磁盘空间不足 ({free_space_gb:.2f} GB)，建议至少保留 1 GB")
        
        # 测试写入权限
        test_file = save_dir / '.write_test'
        try:
            test_file.write_text('test')
            test_file.unlink()
            print(f"✓ 保存目录有写入权限")
        except Exception as e:
            print(f"❌ 保存目录无写入权限: {e}")
            raise
    except Exception as e:
        print(f"⚠️  无法检查保存目录: {e}")
    
    # 设备
    device = torch.device('cpu' if args.use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")
    
    # GPU显存管理配置
    if device.type == 'cuda':
        # 设置显存使用比例（如果指定）
        if args.memory_fraction is not None:
            if 0.0 < args.memory_fraction <= 1.0:
                torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
                print(f"✓ 已设置GPU显存使用比例: {args.memory_fraction*100:.1f}%")
            else:
                print(f"⚠️  警告: memory_fraction必须在(0.0, 1.0]范围内，已忽略")
        
        # 设置TF32（如果支持）
        if args.allow_tf32:
            # PyTorch 1.12+支持TF32
            if hasattr(torch.backends.cuda, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"✓ 已启用TF32加速")
        
        # 显示GPU信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"GPU: {gpu_name}")
            print(f"总显存: {total_memory:.2f} GB")
            
            # 注意：共享显存是Windows系统自动管理的，PyTorch无法直接控制
            # 当专用显存不足时，系统会自动使用共享显存，但性能会显著下降
            print(f"提示: 如果显存不足，建议减小batch_size或使用混合精度训练")
    
    # 创建数据集
    print("创建数据集...")
    print(f"使用 DINOv2 模型: {args.dinov2_model}")
    dataset = TransformerFaceDataset(
        features_224_dir=args.features_224_dir,
        video_dir=args.video_dir,
        face_dir=args.face_dir,
        valid_images_file=args.valid_images_file,
        use_cpu=args.use_cpu,
        cache_features=True,
        dinov2_model_name=args.dinov2_model
    )
    
    # 自动检测特征维度（如果未指定）
    # 根据DINOv2模型确定特征维度
    feature_dims = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536
    }
    expected_feature_dim = feature_dims.get(args.dinov2_model, 768)
    
    if args.d_model is None:
        # 优先使用数据集检测到的维度
        if hasattr(dataset, 'feature_dim'):
            args.d_model = dataset.feature_dim
            print(f"✓ 自动检测到特征维度: {args.d_model} (从数据集)")
        else:
            # 使用模型对应的维度
            args.d_model = expected_feature_dim
            print(f"✓ 自动设置模型维度为: {args.d_model} (根据DINOv2模型 {args.dinov2_model})")
    else:
        # 检查维度是否匹配
        if args.d_model != expected_feature_dim:
            print(f"⚠️  警告: 指定的d_model ({args.d_model}) 与DINOv2模型维度 ({expected_feature_dim}) 不匹配")
            print(f"   建议使用: --d_model {expected_feature_dim}")
        else:
            print(f"使用指定的特征维度: {args.d_model}")
    
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
    elif args.model_type == 'decoder_only':
        # 仅解码器模型
        if TransformerDecoderOnly is None:
            raise ImportError("无法导入 TransformerDecoderOnly，请确保 models_decoder_only.py 存在")
        model = TransformerDecoderOnly(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            use_angle_pe=True,
            use_angle_conditioning=True,
            angle_dim=5
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
    elif args.loss_type == 'angle_aware_triplet':
        if AngleAwareTripletLoss is None:
            raise ImportError("无法导入 AngleAwareTripletLoss，请确保 angle_aware_loss.py 存在")
        criterion = AngleAwareTripletLoss(
            margin=0.2,
            alpha=2.0,
            beta=1.5,
            angle_threshold=30.0
        )
        print(f"使用角度感知三元组损失：margin=0.2, alpha=2.0, beta=1.5")
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
    
    # 混合精度训练（FP16）
    use_amp = not args.no_amp  # 默认启用，除非指定 --no_amp
    scaler = None
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("✓ 已启用混合精度训练（FP16）")
        print(f"  这将节省约50%的显存并加速训练")
    elif use_amp and device.type == 'cpu':
        print("⚠️  警告: CPU不支持混合精度训练，已禁用")
        use_amp = False
    else:
        use_amp = False
    
    # 更新args以便传递给train_epoch
    args.use_amp = use_amp
    
    # 学习率调度器（带Warmup）
    scheduler = None
    warmup_scheduler = None
    
    if args.warmup_epochs > 0:
        # Warmup阶段：线性增加学习率
        def warmup_lambda(epoch):
            if epoch < args.warmup_epochs:
                return args.warmup_lr / args.lr + (1 - args.warmup_lr / args.lr) * (epoch / args.warmup_epochs)
            else:
                return 1.0
        
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        print(f"✓ 启用学习率预热: {args.warmup_epochs} epochs (从 {args.warmup_lr:.2e} 到 {args.lr:.2e})")
    
    if args.use_scheduler:
        if args.scheduler_type == 'cosine':
            # Cosine退火调度器（在warmup之后生效）
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.scheduler_eta_min
            )
            # 组合warmup和cosine调度器
            if warmup_scheduler is not None and args.warmup_epochs > 0:
                from torch.optim.lr_scheduler import SequentialLR
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[args.warmup_epochs]
                )
            else:
                scheduler = main_scheduler
        elif args.scheduler_type == 'step':
            main_scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, (args.epochs - args.warmup_epochs) // 3), gamma=args.scheduler_factor
            )
            if warmup_scheduler is not None and args.warmup_epochs > 0:
                from torch.optim.lr_scheduler import SequentialLR
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[args.warmup_epochs]
                )
            else:
                scheduler = main_scheduler
        else:  # plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=args.scheduler_factor, 
                patience=args.scheduler_patience, verbose=True
            )
        print(f"✓ 使用学习率调度器: {args.scheduler_type}")
    else:
        # 默认使用ReduceLROnPlateau（不通过参数控制）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.scheduler_factor, 
            patience=args.scheduler_patience, verbose=True
        )
        print(f"✓ 使用默认学习率调度器: ReduceLROnPlateau")
    
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
        val_metrics = validate(model, val_loader, criterion, device, use_amp=args.use_amp)
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
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"当前学习率: {current_lr:.2e}")
        
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
        
        # 只保存最佳模型（不保存每个epoch的检查点，节省磁盘空间）
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            try:
                # 文件名包含模型类型标注
                model_type_suffix = args.model_type.replace('_', '-')  # decoder_only -> decoder-only
                best_model_path = save_dir / f'best_model_{model_type_suffix}.pth'
                if save_checkpoint(model, optimizer, epoch, val_metrics['loss'], best_model_path, best=True, model_type=args.model_type):
                    print(f"✓ 保存最佳模型 (验证损失: {best_val_loss:.6f}, Epoch {epoch+1})")
            except Exception as e:
                print(f"❌ 保存最佳模型失败: {e}")
                print("   训练将继续，但最佳模型未保存")
        
        # 每10个epoch保存一次训练曲线
        if (epoch + 1) % 10 == 0:
            model_type_suffix = args.model_type.replace('_', '-')
            plot_path = log_dir / f'training_curves_{model_type_suffix}_epoch_{epoch+1}.png'
            plot_training_curves(
                train_losses, val_losses,
                train_cosine_losses if len(train_cosine_losses) > 0 else None,
                val_cosine_losses if len(val_cosine_losses) > 0 else None,
                train_mse_losses if len(train_mse_losses) > 0 else None,
                val_mse_losses if len(val_mse_losses) > 0 else None,
                val_cosine_sims if len(val_cosine_sims) > 0 else None,
                save_path=str(plot_path)
            )
    
    # 保存最终训练曲线（包含模型类型标注）
    model_type_suffix = args.model_type.replace('_', '-')
    final_plot_path = log_dir / f'training_curves_{model_type_suffix}_final.png'
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

