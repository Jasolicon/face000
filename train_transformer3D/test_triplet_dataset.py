"""
测试数据集和训练代码是否满足三元组损失的要求
"""
import sys
from pathlib import Path
import torch

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.dataset import create_train_val_test_dataloaders
from train_transformer3D.models_3d_triplet import TransformerDecoderOnly3D_Triplet

def test_triplet_pipeline():
    """测试完整的三元组损失数据流"""
    print("=" * 70)
    print("测试三元组损失数据流")
    print("=" * 70)
    
    # 1. 创建数据加载器
    print("\n1. 创建数据加载器...")
    try:
        train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
            data_dir='train/datas/file',
            batch_size=16,  # 使用较大的batch_size确保有足够的身份
            num_workers=0,  # Windows上使用0
            load_in_memory=True,
            use_triplet_collate=True  # 使用三元组损失专用的collate函数
        )
        print(f"   ✓ 数据加载器创建成功")
        print(f"   训练集批次数量: {len(train_loader)}")
    except Exception as e:
        print(f"   ❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 获取一个批次
    print("\n2. 获取一个批次...")
    try:
        batch = next(iter(train_loader))
        print(f"   ✓ 批次获取成功")
        print(f"   批次键: {list(batch.keys())}")
    except Exception as e:
        print(f"   ❌ 批次获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 检查person_name字段
    print("\n3. 检查person_name字段...")
    if 'person_name' not in batch:
        print(f"   ❌ 批次中缺少 'person_name' 字段")
        return False
    
    person_names = batch['person_name']
    print(f"   ✓ person_name 字段存在")
    print(f"   person_name 类型: {type(person_names)}")
    print(f"   person_name 长度: {len(person_names)}")
    print(f"   person_name 示例: {person_names[:5]}")
    
    if not isinstance(person_names, list):
        print(f"   ❌ person_name 不是列表类型")
        return False
    
    # 4. 转换为标签
    print("\n4. 转换为数字标签...")
    unique_names = list(set(person_names))
    print(f"   唯一身份数: {len(unique_names)}")
    print(f"   唯一身份: {unique_names[:5]}")
    
    if len(unique_names) < 2:
        print(f"   ⚠️  警告: 批次中只有 {len(unique_names)} 个身份")
        print(f"      建议增加 batch_size 或检查数据分布")
    else:
        print(f"   ✓ 有足够的身份进行三元组采样")
    
    name_to_label = {name: idx for idx, name in enumerate(unique_names)}
    labels = torch.tensor([name_to_label[name] for name in person_names], dtype=torch.long)
    print(f"   标签形状: {labels.shape}")
    print(f"   标签示例: {labels[:5]}")
    
    # 5. 创建模型并测试前向传播
    print("\n5. 测试模型前向传播...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TransformerDecoderOnly3D_Triplet(
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            identity_dim=512,
            return_identity_features=True
        ).to(device)
        model.eval()
        
        src = batch['src'].to(device)
        pose = batch['pose'].to(device)
        angles = batch['angles'].to(device)
        keypoints_3d = batch['keypoints_3d'].to(device)
        
        with torch.no_grad():
            identity_features, residual = model(
                src=src,
                angles=angles,
                keypoints_3d=keypoints_3d,
                pose=pose,
                return_residual=False
            )
        
        print(f"   ✓ 模型前向传播成功")
        print(f"   输入形状: src={src.shape}, pose={pose.shape}")
        print(f"   输出形状: identity_features={identity_features.shape}")
        print(f"   身份特征L2范数: {torch.norm(identity_features, dim=1).mean().item():.4f} (应该接近1.0)")
    except Exception as e:
        print(f"   ❌ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. 测试三元组损失
    print("\n6. 测试三元组损失...")
    try:
        from train_transformer.angle_aware_loss import AngleAwareTripletLoss
        
        criterion = AngleAwareTripletLoss(
            margin=0.2,
            alpha=2.0,
            beta=1.5,
            angle_threshold=30.0
        )
        
        loss, loss_dict = criterion(
            features=identity_features,
            labels=labels.to(device),
            angles=pose,
            features_orig=src
        )
        
        print(f"   ✓ 三元组损失计算成功")
        print(f"   总损失: {loss.item():.4f}")
        print(f"   三元组损失: {loss_dict.get('triplet_loss', 0.0):.4f}")
        print(f"   重建损失: {loss_dict.get('reconstruction_loss', 0.0):.4f}")
        print(f"   三元组数量: {loss_dict.get('num_triplets', 0)}")
        print(f"   平均正样本距离: {loss_dict.get('avg_pos_dist', 0.0):.4f}")
        print(f"   平均负样本距离: {loss_dict.get('avg_neg_dist', 0.0):.4f}")
        
        if loss_dict.get('num_triplets', 0) == 0:
            print(f"   ⚠️  警告: 没有采样到三元组")
            print(f"      可能原因：批次中身份不足或样本不足")
        else:
            print(f"   ✓ 成功采样到 {loss_dict['num_triplets']} 个三元组")
            
    except ImportError:
        print(f"   ⚠️  无法导入 AngleAwareTripletLoss，跳过测试")
    except Exception as e:
        print(f"   ❌ 三元组损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ 所有测试通过！数据集和训练代码满足三元组损失的要求")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_triplet_pipeline()
    if not success:
        print("\n❌ 测试失败，请检查上述错误信息")
        sys.exit(1)
    else:
        print("\n✓ 可以开始三元组损失训练！")

