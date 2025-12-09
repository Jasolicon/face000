"""
测试数据集
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from train_transformer.dataset import TransformerFaceDataset, create_dataloader

if __name__ == "__main__":
    print("=" * 70)
    print("测试 TransformerFaceDataset")
    print("=" * 70)
    
    try:
        # 创建数据集
        print("\n创建数据集...")
        dataset = TransformerFaceDataset(
            features_224_dir=r'C:\Codes\face000\features_224',
            video_dir=r'C:\Codes\face000\train\datas\video',
            face_dir=r'C:\Codes\face000\train\datas\face',
            use_cpu=False,
            cache_features=True
        )
        
        print(f"\n数据集大小: {len(dataset)}")
        
        # 获取一个样本
        print("\n获取第一个样本...")
        sample = dataset[0]
        print(f"输入特征形状: {sample['input_features'].shape}")
        print(f"位置编码形状: {sample['position_encoding'].shape}")
        print(f"目标特征形状: {sample['target_features'].shape}")
        print(f"目标残差形状: {sample['target_residual'].shape}")
        print(f"人名: {sample['person_name']}")
        print(f"位置编码值: {sample['position_encoding'].numpy()}")
        print(f"残差统计: min={sample['target_residual'].min():.4f}, max={sample['target_residual'].max():.4f}, mean={sample['target_residual'].mean():.4f}")
        
        # 创建DataLoader
        print("\n创建DataLoader...")
        dataloader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        # 测试一个批次
        print("\n测试一个批次...")
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n批次 {batch_idx}:")
            print(f"  输入特征形状: {batch['input_features'].shape}")
            print(f"  位置编码形状: {batch['position_encoding'].shape}")
            print(f"  目标特征形状: {batch['target_features'].shape}")
            print(f"  目标残差形状: {batch['target_residual'].shape}")
            print(f"  人名: {batch['person_names']}")
            print(f"  位置编码值:\n{batch['position_encoding'].numpy()}")
            print(f"  残差统计: min={batch['target_residual'].min():.4f}, max={batch['target_residual'].max():.4f}, mean={batch['target_residual'].mean():.4f}")
            
            if batch_idx >= 2:  # 只测试前3个批次
                break
        
        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

