"""
批量更新现有特征数据库中的姓名信息
从图像文件名中提取姓名并更新metadata
"""
import os
import json
from pathlib import Path
from feature_manager import FeatureManager


def update_person_names_from_filename(storage_dir='features'):
    """
    从图像文件名提取姓名并更新metadata
    
    Args:
        storage_dir: 特征存储目录
    """
    print("=" * 70)
    print("批量更新姓名信息")
    print("=" * 70)
    
    manager = FeatureManager(storage_dir=storage_dir)
    features, metadata = manager.get_all_features()
    
    if features is None or len(metadata) == 0:
        print("特征数据库为空，无需更新")
        return
    
    print(f"\n找到 {len(metadata)} 条记录")
    print("开始更新姓名信息...")
    print("-" * 70)
    
    updated_count = 0
    skipped_count = 0
    
    for i, meta in enumerate(metadata):
        image_path = meta.get('image_path', '')
        current_name = meta.get('person_name')
        
        if not image_path:
            print(f"  记录 {i}: 缺少image_path，跳过")
            skipped_count += 1
            continue
        
        # 从文件名提取姓名（不含扩展名）
        filename = os.path.basename(image_path)
        extracted_name = os.path.splitext(filename)[0]
        
        # 如果当前姓名为None或空，则更新
        if current_name is None or current_name == '':
            meta['person_name'] = extracted_name
            updated_count += 1
            print(f"  {i+1}/{len(metadata)}: {extracted_name} (从 {filename} 提取)")
        else:
            skipped_count += 1
            print(f"  {i+1}/{len(metadata)}: {current_name} (已有姓名，跳过)")
    
    # 保存更新后的元数据
    if updated_count > 0:
        print("\n" + "-" * 70)
        print(f"更新完成: 更新了 {updated_count} 条记录，跳过 {skipped_count} 条记录")
        print("正在保存...")
        
        # 保存到文件
        metadata_file = Path(storage_dir) / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"已保存到: {metadata_file}")
    else:
        print("\n没有需要更新的记录")
    
    print("=" * 70)


def main():
    """主函数"""
    import sys
    
    storage_dir = 'features'
    if len(sys.argv) > 1:
        storage_dir = sys.argv[1]
    
    update_person_names_from_filename(storage_dir=storage_dir)


if __name__ == '__main__':
    main()

