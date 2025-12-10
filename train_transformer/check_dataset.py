"""
检查数据集配置的脚本
用于诊断数据路径和文件匹配问题
"""
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from feature_manager import FeatureManager
import json

def check_dataset_config(
    features_224_dir: str,
    video_dir: str,
    face_dir: str
):
    """
    检查数据集配置
    
    Args:
        features_224_dir: 特征库目录
        video_dir: 视频帧目录
        face_dir: 正面图片目录
    """
    print("=" * 70)
    print("数据集配置检查")
    print("=" * 70)
    
    features_224_path = Path(features_224_dir)
    video_path = Path(video_dir)
    face_path = Path(face_dir)
    
    # 1. 检查目录是否存在
    print("\n1. 检查目录是否存在:")
    print(f"   features_224_dir: {features_224_path}")
    print(f"   ✓ 存在" if features_224_path.exists() else f"   ✗ 不存在")
    
    print(f"   video_dir: {video_path}")
    print(f"   ✓ 存在" if video_path.exists() else f"   ✗ 不存在")
    
    print(f"   face_dir: {face_path}")
    print(f"   ✓ 存在" if face_path.exists() else f"   ✗ 不存在")
    
    # 2. 检查 features_224
    print("\n2. 检查 features_224 特征库:")
    if features_224_path.exists():
        try:
            feature_manager = FeatureManager(storage_dir=str(features_224_path))
            features, metadata = feature_manager.get_all_features()
            
            if features is None or len(features) == 0:
                print(f"   ✗ 特征库为空")
            else:
                print(f"   ✓ 找到 {len(features)} 个特征")
                
                # 提取人名
                person_names = set()
                for meta in metadata:
                    person_name = meta.get('person_name')
                    if person_name:
                        person_names.add(person_name)
                
                print(f"   ✓ 找到 {len(person_names)} 个不同的人名")
                print(f"   前10个人名: {sorted(list(person_names))[:10]}")
        except Exception as e:
            print(f"   ✗ 加载特征库失败: {e}")
    else:
        print(f"   ✗ 目录不存在，无法检查")
    
    # 3. 检查正面图片
    print("\n3. 检查正面图片目录:")
    if face_path.exists():
        face_files = list(face_path.glob('*.jpg'))
        print(f"   ✓ 找到 {len(face_files)} 个 .jpg 文件")
        
        if len(face_files) > 0:
            face_names = {f.stem for f in face_files}
            print(f"   ✓ 找到 {len(face_names)} 个不同的人名")
            print(f"   前10个人名: {sorted(list(face_names))[:10]}")
            
            # 显示文件列表
            print(f"\n   文件列表（前10个）:")
            for f in face_files[:10]:
                print(f"     - {f.name}")
        else:
            print(f"   ⚠️  目录为空或没有 .jpg 文件")
            print(f"   目录内容: {list(face_path.iterdir())[:10]}")
    else:
        print(f"   ✗ 目录不存在")
    
    # 4. 检查视频帧目录
    print("\n4. 检查视频帧目录:")
    if video_path.exists():
        person_dirs = [d for d in video_path.iterdir() if d.is_dir()]
        print(f"   ✓ 找到 {len(person_dirs)} 个人物目录")
        
        if len(person_dirs) > 0:
            print(f"   前10个人物目录:")
            for person_dir in person_dirs[:10]:
                video_files = list(person_dir.glob('*.jpg'))
                print(f"     - {person_dir.name}: {len(video_files)} 张图片")
    else:
        print(f"   ✗ 目录不存在")
    
    # 5. 匹配检查
    print("\n5. 人名匹配检查:")
    if features_224_path.exists() and face_path.exists():
        try:
            # 获取 features_224 中的人名
            feature_manager = FeatureManager(storage_dir=str(features_224_path))
            _, metadata = feature_manager.get_all_features()
            features_224_names = {meta.get('person_name') for meta in metadata if meta.get('person_name')}
            
            # 获取正面图片中的人名
            face_files = list(face_path.glob('*.jpg'))
            face_names = {f.stem for f in face_files}
            
            # 找到共有的人名
            common_names = face_names.intersection(features_224_names)
            
            print(f"   features_224 中的人名: {len(features_224_names)} 个")
            print(f"   正面图片中的人名: {len(face_names)} 个")
            print(f"   共有的人名: {len(common_names)} 个")
            
            if len(common_names) > 0:
                print(f"   ✓ 匹配成功！")
                print(f"   匹配的人名（前10个）: {sorted(list(common_names))[:10]}")
            else:
                print(f"   ✗ 没有匹配的人名")
                
                # 检查大小写问题
                face_names_lower = {name.lower().strip() for name in face_names}
                features_224_names_lower = {name.lower().strip() for name in features_224_names}
                common_names_lower = face_names_lower.intersection(features_224_names_lower)
                
                if len(common_names_lower) > 0:
                    print(f"\n   ⚠️  发现 {len(common_names_lower)} 个可能匹配的人名（忽略大小写）:")
                    print(f"   {sorted(list(common_names_lower))[:10]}")
                    print(f"   提示: 可能是文件名大小写问题")
                
                # 显示不匹配的示例
                print(f"\n   不匹配示例（正面图片中但不在 features_224 中，前5个）:")
                unmatched = face_names - features_224_names
                for name in sorted(list(unmatched))[:5]:
                    print(f"     - {name}")
                
                print(f"\n   不匹配示例（features_224 中但不在正面图片中，前5个）:")
                unmatched = features_224_names - face_names
                for name in sorted(list(unmatched))[:5]:
                    print(f"     - {name}")
        except Exception as e:
            print(f"   ✗ 匹配检查失败: {e}")
    
    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检查数据集配置')
    parser.add_argument('--features_224_dir', type=str,
                       default=r'/root/face000/features_224',
                       help='features_224特征库目录')
    parser.add_argument('--video_dir', type=str,
                       default=r'/root/face000/train/datas/video',
                       help='视频帧图片目录')
    parser.add_argument('--face_dir', type=str,
                       default=r'/root/face000/train/datas/face',
                       help='正面图片目录')
    
    args = parser.parse_args()
    
    check_dataset_config(
        features_224_dir=args.features_224_dir,
        video_dir=args.video_dir,
        face_dir=args.face_dir
    )

