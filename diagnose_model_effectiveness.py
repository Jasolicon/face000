"""
诊断模型有效性工具
比较原始特征与模型输出特征的相似度，判断模型是否真的在学习
"""
import os
import sys
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_data(model_path, data_dir='train/datas/file', model_type='decoder_only'):
    """加载模型和数据"""
    import sys
    sys.path.append(str(Path(__file__).parent))
    
    from train_transformer3D.dataset import create_train_val_test_dataloaders
    from train_transformer3D.models_3d import TransformerDecoderOnly3D
    from train_transformer3D.models_3d_fulltransformer import TransformerEncoderDecoder3D
    from train_transformer3D.models_angle_warping import FinalRecommendedModel
    from train_transformer3D.models_universal import UniversalFaceTransformer
    from train_transformer3D.losses_universal import UniversalFaceLoss
    
    logger.info(f"加载模型: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    logger.info("加载数据...")
    train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=0,
        load_in_memory=True
    )
    
    # 初始化criterion（用于universal模型的投影层）
    criterion = None
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从checkpoint读取模型配置
    # checkpoint中保存了'args'字典，包含所有训练参数
    if 'args' in checkpoint:
        args_dict = checkpoint['args']
        # 优先使用命令行参数，如果没有则使用checkpoint中的
        if model_type == 'decoder_only' and 'model_type' in args_dict:
            model_type = args_dict.get('model_type', model_type)
        logger.info(f"从checkpoint读取模型类型: {model_type}")
    elif 'model_type' in checkpoint:
        if model_type == 'decoder_only':  # 只有默认值时才使用checkpoint中的
            model_type = checkpoint['model_type']
        args_dict = {}
    else:
        # 尝试从state_dict推断模型类型
        state_dict_keys = list(checkpoint['model_state_dict'].keys())
        if 'ortho_proj' in state_dict_keys[0] or 'pose_encoder' in state_dict_keys[0]:
            model_type = 'universal'
        elif 'spatial_fusion' in state_dict_keys[0] or 'pose_attention' in state_dict_keys[0]:
            model_type = 'decoder_only'
        elif 'encoder' in state_dict_keys[0]:
            model_type = 'encoder_decoder'
        else:
            model_type = 'decoder_only'  # 默认
        args_dict = {}
        logger.warning(f"checkpoint中未找到model_type，推断为: {model_type}")
    
    # 获取模型参数（从checkpoint的args中读取）
    if model_type == 'decoder_only':
        d_model = args_dict.get('d_model', 512)
        nhead = args_dict.get('nhead', 8)
        num_layers = args_dict.get('num_decoder_layers', args_dict.get('num_layers', 4))  # 优先使用num_decoder_layers
        dim_feedforward = args_dict.get('dim_feedforward', 2048)
        dropout = args_dict.get('dropout', 0.1)
        num_keypoints = args_dict.get('num_keypoints', 5)
        pose_dim = args_dict.get('pose_dim', 3)
        use_spatial_attention = args_dict.get('use_spatial_attention', True)
        use_pose_attention = args_dict.get('use_pose_attention', True)
        
        logger.info(f"创建模型: decoder_only")
        logger.info(f"  参数: d_model={d_model}, nhead={nhead}, num_layers={num_layers}, "
                   f"dim_feedforward={dim_feedforward}, dropout={dropout}, num_keypoints={num_keypoints}")
        
        model = TransformerDecoderOnly3D(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_keypoints=num_keypoints,
            pose_dim=pose_dim,
            use_spatial_attention=use_spatial_attention,
            use_pose_attention=use_pose_attention,
            use_angle_pe=True,
            use_angle_conditioning=True
        )
    elif model_type == 'encoder_decoder':
        d_model = args_dict.get('d_model', 512)
        nhead = args_dict.get('nhead', 8)
        num_encoder_layers = args_dict.get('num_encoder_layers', 3)
        num_decoder_layers = args_dict.get('num_decoder_layers', 3)
        dim_feedforward = args_dict.get('dim_feedforward', 2048)
        dropout = args_dict.get('dropout', 0.1)
        num_keypoints = args_dict.get('num_keypoints', 5)
        
        logger.info(f"创建模型: encoder_decoder")
        logger.info(f"  参数: d_model={d_model}, num_encoder_layers={num_encoder_layers}, num_decoder_layers={num_decoder_layers}")
        
        model = TransformerEncoderDecoder3D(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_keypoints=num_keypoints
        )
    elif model_type == 'angle_warping':
        d_model = args_dict.get('d_model', 512)
        hidden_dim = args_dict.get('hidden_dim', 256)
        num_basis = args_dict.get('num_basis', 32)
        use_basis = args_dict.get('use_basis', True)
        use_refinement = args_dict.get('use_refinement', True)
        use_attention_refine = args_dict.get('use_attention_refine', True)
        num_attention_layers = args_dict.get('num_attention_layers', 2)
        
        logger.info(f"创建模型: angle_warping")
        logger.info(f"  参数: d_model={d_model}, hidden_dim={hidden_dim}, num_basis={num_basis}")
        
        model = FinalRecommendedModel(
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_basis=num_basis,
            use_basis=use_basis,
            use_refinement=use_refinement,
            use_attention_refine=use_attention_refine,
            num_attention_layers=num_attention_layers
        )
    elif model_type == 'universal':
        feat_dim = args_dict.get('feat_dim', 512)
        id_dim = args_dict.get('id_dim', 256)
        pose_dim = args_dict.get('pose_dim', 128)
        num_pose_bins = args_dict.get('num_pose_bins', 36)
        transformer_depth = args_dict.get('transformer_depth', 4)
        transformer_heads = args_dict.get('transformer_heads', 8)
        transformer_mlp_dim = args_dict.get('transformer_mlp_dim', 1024)
        dropout = args_dict.get('dropout', 0.1)
        
        logger.info(f"创建模型: universal")
        logger.info(f"  参数: feat_dim={feat_dim}, id_dim={id_dim}, pose_dim={pose_dim}, "
                   f"transformer_depth={transformer_depth}, transformer_heads={transformer_heads}")
        
        model = UniversalFaceTransformer(
            feat_dim=feat_dim,
            id_dim=id_dim,
            pose_dim=pose_dim,
            num_pose_bins=num_pose_bins,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )
        
        # 创建损失函数以获取投影层（用于将id_features投影回512维）
        criterion = UniversalFaceLoss(
            lambda_id=args_dict.get('lambda_id', 2.0),
            lambda_pose=args_dict.get('lambda_pose', 0.3),
            lambda_ortho=args_dict.get('lambda_ortho', 0.05),
            lambda_contrast=args_dict.get('lambda_contrast', 0.2),
            lambda_reconstruction=args_dict.get('lambda_reconstruction', 0.5),
            id_dim=id_dim,
            feat_dim=feat_dim
        ).to(device)
        
        # 尝试从checkpoint加载损失函数的投影层权重（如果存在）
        if 'criterion_state_dict' in checkpoint:
            try:
                criterion.load_state_dict(checkpoint['criterion_state_dict'], strict=False)
                logger.info("✓ 损失函数投影层权重加载成功")
            except:
                logger.warning("损失函数投影层权重加载失败，将使用Xavier初始化的投影层")
                import torch.nn.init as init
                init.xavier_uniform_(criterion.id_to_feat_proj.weight)
        else:
            # 如果没有保存损失函数权重，使用Xavier初始化投影层（比随机初始化更好）
            import torch.nn.init as init
            init.xavier_uniform_(criterion.id_to_feat_proj.weight)
            logger.warning("checkpoint中未找到损失函数权重，使用Xavier初始化的投影层（可能不够准确）")
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 尝试加载state_dict，如果失败则使用strict=False
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info("✓ 模型状态字典加载成功（严格模式）")
    except RuntimeError as e:
        logger.warning(f"严格加载失败: {str(e)[:200]}...")
        logger.info("尝试使用strict=False加载（忽略不匹配的键）...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            logger.warning(f"缺失的键数量: {len(missing_keys)}")
            if len(missing_keys) <= 20:
                logger.warning(f"缺失的键: {missing_keys}")
            else:
                logger.warning(f"缺失的键（前20个）: {missing_keys[:20]}")
        if unexpected_keys:
            logger.warning(f"意外的键数量: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 20:
                logger.warning(f"意外的键: {unexpected_keys}")
            else:
                logger.warning(f"意外的键（前20个）: {unexpected_keys[:20]}")
        logger.info("✓ 模型状态字典加载完成（非严格模式）")
    model.to(device)
    model.eval()
    
    logger.info(f"模型加载完成，设备: {device}")
    
    # 对于universal模型，返回模型、数据加载器、设备和损失函数（用于投影）
    if model_type == 'universal':
        return model, val_loader, device, criterion
    else:
        return model, val_loader, device, None


def diagnose_model_effectiveness(model, dataloader, device, max_samples=1000, criterion=None, model_type='decoder_only'):
    """
    诊断模型有效性
    比较原始特征和模型输出特征与正面特征的相似度
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最大样本数
        criterion: 损失函数（用于universal模型的投影层）
        model_type: 模型类型
    """
    logger.info("开始诊断模型有效性...")
    
    model.eval()
    if criterion is not None:
        criterion.eval()
    
    results = {
        'original_similarities': [],
        'model_similarities': [],
        'improvements': [],
        'yaw_angles': [],
        'person_names': []
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break
            
            src = batch['src'].to(device)  # 侧面特征
            tgt = batch['tgt'].to(device)  # 正面特征
            pose = batch['pose'].to(device)
            person_names = batch['person_name']
            
            # 模型前向传播（根据模型类型使用不同的接口）
            if model_type == 'universal':
                # Universal模型使用 features 和 pose_angles
                outputs = model(features=src, pose_angles=pose, mode='inference')
                # Universal模型返回字典，包含id_features (256维)
                # 需要投影回512维才能与tgt比较
                if criterion is not None:
                    id_features_512 = criterion.id_to_feat_proj(outputs['id_features'])  # [batch, 256] -> [batch, 512]
                    output = id_features_512
                else:
                    # 如果没有损失函数，直接使用id_features（可能不准确）
                    logger.warning("未提供损失函数，直接使用id_features计算相似度（可能不准确）")
                    output = outputs['id_features']
            else:
                # 其他模型使用原来的接口
                keypoints_3d = batch['keypoints_3d'].to(device)
                angles = batch['angles'].to(device)
                output = model(
                    src=src,
                    angles=angles,
                    keypoints_3d=keypoints_3d,
                    pose=pose,
                    return_residual=False
                )
            
            # 计算原始特征与正面特征的相似度
            original_sim = torch.nn.functional.cosine_similarity(src, tgt, dim=1)
            
            # 计算模型输出与正面特征的相似度
            model_sim = torch.nn.functional.cosine_similarity(output, tgt, dim=1)
            
            # 计算改进量
            improvements = model_sim - original_sim
            
            # 获取yaw角度
            yaw = pose[:, 0].cpu().numpy()
            abs_yaw = np.abs(yaw)
            
            # 过滤逻辑：
            # 1. |yaw| >= 15° 的样本全部保留
            # 2. |yaw| < 15° 的样本随机保留10%
            large_angle_mask = abs_yaw >= 15.0
            small_angle_mask = abs_yaw < 15.0
            
            # 对于小角度样本，随机保留10%
            if np.sum(small_angle_mask) > 0:
                small_angle_indices = np.where(small_angle_mask)[0]
                # 随机选择10%的小角度样本
                keep_ratio = 0.1
                n_keep = max(1, int(len(small_angle_indices) * keep_ratio))
                np.random.seed(42)  # 固定随机种子以保证可复现
                kept_indices = np.random.choice(small_angle_indices, size=n_keep, replace=False)
                small_angle_keep_mask = np.zeros(len(yaw), dtype=bool)
                small_angle_keep_mask[kept_indices] = True
            else:
                small_angle_keep_mask = np.zeros(len(yaw), dtype=bool)
            
            # 合并掩码：大角度全部保留 + 小角度随机保留
            valid_mask = large_angle_mask | small_angle_keep_mask
            
            if np.sum(valid_mask) > 0:
                # 只保存符合条件的样本
                results['original_similarities'].extend(original_sim.cpu().numpy()[valid_mask])
                results['model_similarities'].extend(model_sim.cpu().numpy()[valid_mask])
                results['improvements'].extend(improvements.cpu().numpy()[valid_mask])
                results['yaw_angles'].extend(yaw[valid_mask])
                # person_names 是列表，需要手动过滤
                person_names_list = list(person_names)
                results['person_names'].extend([person_names_list[i] for i in range(len(person_names_list)) if valid_mask[i]])
            
            sample_count += np.sum(valid_mask)
    
    # 转换为numpy数组
    for key in results:
        if key != 'person_names':
            results[key] = np.array(results[key])
    
    # 统计大角度和小角度样本数量
    yaw_array = np.array(results['yaw_angles'])
    large_angle_count = np.sum(np.abs(yaw_array) >= 15.0)
    small_angle_count = np.sum(np.abs(yaw_array) < 15.0)
    logger.info(f"处理了 {sample_count} 个样本（大角度 |yaw| >= 15°: {large_angle_count} 个，小角度 |yaw| < 15°: {small_angle_count} 个，随机保留10%）")
    
    return results


def analyze_results(results):
    """分析诊断结果"""
    logger.info("\n" + "=" * 70)
    logger.info("模型有效性诊断结果")
    logger.info("=" * 70)
    
    original_sim = results['original_similarities']
    model_sim = results['model_similarities']
    improvements = results['improvements']
    yaw_angles = results['yaw_angles']
    
    # 统计大角度和小角度样本数量
    yaw_array = np.array(yaw_angles)
    large_angle_count = np.sum(np.abs(yaw_array) >= 15.0)
    small_angle_count = np.sum(np.abs(yaw_array) < 15.0)
    
    logger.info(f"\n【整体统计】（大角度 |yaw| >= 15°: {large_angle_count} 个，小角度 |yaw| < 15°: {small_angle_count} 个）")
    logger.info(f"原始特征平均相似度: {np.mean(original_sim):.4f} ± {np.std(original_sim):.4f}")
    logger.info(f"模型输出平均相似度: {np.mean(model_sim):.4f} ± {np.std(model_sim):.4f}")
    logger.info(f"平均改进量: {np.mean(improvements):.4f} ± {np.std(improvements):.4f}")
    
    # 改进率
    improvement_rate = np.mean(improvements > 0) * 100
    logger.info(f"改进率: {improvement_rate:.1f}% 的样本得到了改进")
    
    # 按角度分析
    logger.info(f"\n【按角度分析】")
    angle_ranges = [
        (-90, -45, "大角度左转 (-90° ~ -45°)"),
        (-45, -15, "中角度左转 (-45° ~ -15°)"),
        (-15, 15, "接近正面 (-15° ~ 15°)"),
        (15, 45, "中角度右转 (15° ~ 45°)"),
        (45, 90, "大角度右转 (45° ~ 90°)")
    ]
    
    for min_angle, max_angle, label in angle_ranges:
        mask = (yaw_angles >= min_angle) & (yaw_angles < max_angle)
        if np.sum(mask) > 0:
            orig_sim_range = np.mean(original_sim[mask])
            model_sim_range = np.mean(model_sim[mask])
            improvement_range = np.mean(improvements[mask])
            logger.info(f"{label}:")
            logger.info(f"  原始相似度: {orig_sim_range:.4f}")
            logger.info(f"  模型相似度: {model_sim_range:.4f}")
            logger.info(f"  改进量: {improvement_range:.4f} ({'✓' if improvement_range > 0 else '✗'})")
    
    # 诊断结论
    logger.info(f"\n【诊断结论】")
    if np.mean(improvements) > 0.01:
        logger.info("✓ 模型正在学习：平均改进量 > 0.01")
    elif np.mean(improvements) > 0:
        logger.info("⚠ 模型有轻微改进，但改进量很小")
    else:
        logger.info("✗ 模型没有改进，甚至可能变差")
        logger.info("  可能的原因：")
        logger.info("    1. 模型架构不适合这个任务")
        logger.info("    2. 训练不充分或学习率设置不当")
        logger.info("    3. 损失函数权重不平衡")
        logger.info("    4. 数据质量问题")
        logger.info("    5. 模型容量不足或过拟合")
    
    if improvement_rate < 50:
        logger.info(f"⚠ 只有 {improvement_rate:.1f}% 的样本得到改进，模型可能没有有效学习")
    
    return {
        'mean_original_sim': np.mean(original_sim),
        'mean_model_sim': np.mean(model_sim),
        'mean_improvement': np.mean(improvements),
        'improvement_rate': improvement_rate
    }


def plot_diagnosis_results(results, output_path='model_effectiveness_diagnosis.png'):
    """绘制诊断结果图表（大角度全部保留，小角度随机保留10%）"""
    logger.info("绘制诊断结果图表（大角度全部保留，小角度随机保留10%）...")
    
    original_sim = results['original_similarities']
    model_sim = results['model_similarities']
    improvements = results['improvements']
    yaw_angles = results['yaw_angles']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 相似度对比散点图
    ax1 = axes[0, 0]
    ax1.scatter(original_sim, model_sim, alpha=0.5, s=10)
    ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='无改进线')
    ax1.set_xlabel('原始特征相似度', fontsize=12)
    ax1.set_ylabel('模型输出相似度', fontsize=12)
    ax1.set_title('相似度对比（点在红线上方表示改进）', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. 改进量分布直方图
    ax2 = axes[0, 1]
    ax2.hist(improvements, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='无改进线')
    ax2.axvline(np.mean(improvements), color='green', linestyle='--', linewidth=2, 
                label=f'平均改进: {np.mean(improvements):.4f}')
    ax2.set_xlabel('改进量 (模型相似度 - 原始相似度)', fontsize=12)
    ax2.set_ylabel('样本数量', fontsize=12)
    ax2.set_title('改进量分布', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 按角度分组的改进量
    ax3 = axes[1, 0]
    angle_ranges = [
        (-90, -45, "大左转"),
        (-45, -15, "中左转"),
        (-15, 15, "正面"),
        (15, 45, "中右转"),
        (45, 90, "大右转")
    ]
    
    range_labels = []
    range_improvements = []
    for min_angle, max_angle, label in angle_ranges:
        mask = (yaw_angles >= min_angle) & (yaw_angles < max_angle)
        if np.sum(mask) > 0:
            range_labels.append(label)
            range_improvements.append(np.mean(improvements[mask]))
    
    colors = ['green' if imp > 0 else 'red' for imp in range_improvements]
    ax3.bar(range_labels, range_improvements, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('角度范围', fontsize=12)
    ax3.set_ylabel('平均改进量', fontsize=12)
    ax3.set_title('不同角度下的平均改进量', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. 相似度随角度变化
    ax4 = axes[1, 1]
    # 按角度排序
    sorted_indices = np.argsort(yaw_angles)
    sorted_yaw = yaw_angles[sorted_indices]
    sorted_orig = original_sim[sorted_indices]
    sorted_model = model_sim[sorted_indices]
    
    ax4.plot(sorted_yaw, sorted_orig, 'o-', label='原始特征', alpha=0.6, markersize=2, linewidth=1)
    ax4.plot(sorted_yaw, sorted_model, 's-', label='模型输出', alpha=0.6, markersize=2, linewidth=1)
    ax4.set_xlabel('Yaw角度 (°)', fontsize=12)
    ax4.set_ylabel('相似度', fontsize=12)
    ax4.set_title('相似度随角度变化', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('模型有效性诊断', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"诊断图表已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='诊断模型有效性')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--data_dir', type=str, default='train/datas/file',
                        help='数据目录')
    parser.add_argument('--model_type', type=str, default='decoder_only',
                        choices=['decoder_only', 'encoder_decoder', 'angle_warping', 'universal'],
                        help='模型类型')
    parser.add_argument('--max_samples', type=int, default=1000,
                        help='最多处理的样本数')
    parser.add_argument('--output_path', type=str, default='model_effectiveness_diagnosis.png',
                        help='输出图表路径')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("模型有效性诊断工具")
    logger.info("=" * 70)
    
    try:
        # 1. 加载模型和数据
        load_result = load_model_and_data(
            args.model_path,
            args.data_dir,
            args.model_type
        )
        
        if len(load_result) == 4:
            model, val_loader, device, criterion = load_result
        else:
            model, val_loader, device = load_result
            criterion = None
        
        # 2. 诊断模型有效性
        results = diagnose_model_effectiveness(
            model, val_loader, device, args.max_samples, criterion, args.model_type
        )
        
        # 3. 分析结果
        stats = analyze_results(results)
        
        # 4. 绘制诊断图表
        plot_diagnosis_results(results, args.output_path)
        
        logger.info("\n" + "=" * 70)
        logger.info("诊断完成！")
        logger.info(f"图表已保存到: {args.output_path}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
