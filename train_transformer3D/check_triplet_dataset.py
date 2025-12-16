"""
检查数据集是否满足三元组损失的输入要求
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer3D.dataset import Aligned3DFaceDataset, create_train_val_test_dataloaders
import torch

def check_triplet_requirements():
    """检查数据集是否满足三元组损失的要求"""
    print("=" * 70)
    print("检查数据集是否满足三元组损失的输入要求")
    print("=" * 70)
    
    # 1. 检查数据目录
    data_dir = 'train/datas/file'
    print(f"\n1. 检查数据目录: {data_dir}")
    if not Path(data_dir).exists():
        print(f"   ❌ 数据目录不存在: {data_dir}")
        return False
    print(f"   ✓ 数据目录存在")
    
    # 2. 创建数据集
    print(f"\n2. 创建数据集...")
    try:
        dataset = Aligned3DFaceDataset(data_dir=data_dir, load_in_memory=True)
        print(f"   ✓ 数据集创建成功")
        print(f"   样本数量: {len(dataset)}")
    except Exception as e:
        print(f"   ❌ 数据集创建失败: {e}")
        return False
    
    # 3. 检查单个样本是否包含 person_name
    print(f"\n3. 检查单个样本...")
    try:
        sample = dataset[0]
        print(f"   样本键: {list(sample.keys())}")
        
        if 'person_name' not in sample:
            print(f"   ❌ 样本中缺少 'person_name' 字段")
            return False
        print(f"   ✓ 样本包含 'person_name' 字段")
        print(f"   person_name 类型: {type(sample['person_name'])}")
        print(f"   person_name 值: {sample['person_name']}")
        
        # 检查其他必需字段
        required_fields = ['src', 'tgt', 'pose', 'angles']
        for field in required_fields:
            if field not in sample:
                print(f"   ❌ 样本中缺少 '{field}' 字段")
                return False
        print(f"   ✓ 所有必需字段都存在")
        
    except Exception as e:
        print(f"   ❌ 获取样本失败: {e}")
        return False
    
    # 4. 检查数据加载器（批处理）
    print(f"\n4. 检查数据加载器（批处理）...")
    try:
        train_loader, val_loader, test_loader = create_train_val_test_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            num_workers=0,  # Windows上使用0
            load_in_memory=True
        )
        print(f"   ✓ 数据加载器创建成功")
        print(f"   训练集批次数量: {len(train_loader)}")
        print(f"   验证集批次数量: {len(val_loader)}")
        print(f"   测试集批次数量: {len(test_loader)}")
        
        # 获取一个批次
        batch = next(iter(train_loader))
        print(f"\n   批次键: {list(batch.keys())}")
        
        if 'person_name' not in batch:
            print(f"   ❌ 批次中缺少 'person_name' 字段")
            return False
        print(f"   ✓ 批次包含 'person_name' 字段")
        
        person_names = batch['person_name']
        print(f"   person_names 类型: {type(person_names)}")
        print(f"   person_names 长度: {len(person_names)}")
        print(f"   person_names 示例: {person_names[:3]}")
        
        # 检查是否可以转换为标签
        unique_names = list(set(person_names))
        name_to_label = {name: idx for idx, name in enumerate(unique_names)}
        labels = torch.tensor([name_to_label[name] for name in person_names])
        print(f"   ✓ 可以转换为数字标签")
        print(f"   唯一身份数: {len(unique_names)}")
        print(f"   标签形状: {labels.shape}")
        print(f"   标签示例: {labels[:3]}")
        
    except Exception as e:
        print(f"   ❌ 数据加载器检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 检查三元组采样要求
    print(f"\n5. 检查三元组采样要求...")
    try:
        # 检查每个批次是否有足够的身份（至少2个不同身份）
        batch = next(iter(train_loader))
        person_names = batch['person_name']
        unique_names = list(set(person_names))
        
        if len(unique_names) < 2:
            print(f"   ⚠️  警告: 批次中只有 {len(unique_names)} 个不同身份")
            print(f"      三元组损失需要至少2个不同身份才能采样负样本")
            print(f"      建议增加 batch_size 或使用更大的数据集")
        else:
            print(f"   ✓ 批次中有 {len(unique_names)} 个不同身份，满足三元组采样要求")
        
        # 检查每个身份是否有多个样本（用于采样正样本）
        name_counts = {}
        for name in person_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        single_sample_names = [name for name, count in name_counts.items() if count == 1]
        if len(single_sample_names) > 0:
            print(f"   ⚠️  警告: 批次中有 {len(single_sample_names)} 个身份只有1个样本")
            print(f"      这些身份无法在当前批次中采样正样本（需要同身份的不同样本）")
            print(f"      建议增加 batch_size")
        else:
            print(f"   ✓ 所有身份都有多个样本，可以采样正样本")
        
    except Exception as e:
        print(f"   ❌ 三元组采样检查失败: {e}")
        return False
    
    # 6. 检查角度信息
    print(f"\n6. 检查角度信息...")
    try:
        batch = next(iter(train_loader))
        pose = batch['pose']
        print(f"   ✓ 姿态信息存在")
        print(f"   pose 形状: {pose.shape}")
        print(f"   pose 范围: yaw=[{pose[:, 0].min():.2f}, {pose[:, 0].max():.2f}], "
              f"pitch=[{pose[:, 1].min():.2f}, {pose[:, 1].max():.2f}], "
              f"roll=[{pose[:, 2].min():.2f}, {pose[:, 2].max():.2f}]")
        
    except Exception as e:
        print(f"   ❌ 角度信息检查失败: {e}")
        return False
    
    # 7. 统计信息
    print(f"\n7. 数据集统计信息...")
    stats = dataset.get_statistics()
    print(f"   总样本数: {stats['num_samples']}")
    print(f"   唯一身份数: {stats['unique_persons']}")
    print(f"   平均每个身份的样本数: {stats['num_samples'] / stats['unique_persons']:.1f}")
    
    if stats['unique_persons'] < 2:
        print(f"   ❌ 数据集只有 {stats['unique_persons']} 个身份，无法进行三元组训练")
        return False
    
    print(f"\n" + "=" * 70)
    print("✓ 数据集满足三元组损失的输入要求！")
    print("=" * 70)
    print("\n建议:")
    print("  1. 确保 batch_size >= 16，以便每个批次有足够的身份和样本")
    print("  2. 如果某些身份样本较少，考虑使用更大的 batch_size")
    print("  3. 三元组损失会自动处理无法采样的情况（返回零损失）")
    
    return True


if __name__ == "__main__":
    success = check_triplet_requirements()
    if not success:
        print("\n❌ 数据集不满足三元组损失的要求，请检查上述错误信息")
        sys.exit(1)
    else:
        print("\n✓ 所有检查通过，可以开始三元组损失训练！")

