"""
计算标准关键点坐标
从数据集中提取所有正面图关键点，计算标准坐标系统
"""
import json
import numpy as np
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))


def calculate_standard_landmarks(valid_images_file: str = 'train_transformer/valid_images.json'):
    """
    计算标准关键点坐标
    
    Args:
        valid_images_file: valid_images.json文件路径
        
    Returns:
        standard_landmarks: 标准关键点坐标 [5, 2]
        center: 中心点坐标 [2]
        statistics: 统计信息
    """
    print("=" * 70)
    print("计算标准关键点坐标")
    print("=" * 70)
    
    # 加载数据
    print(f"\n正在加载数据: {valid_images_file}")
    print("  这可能需要一些时间，因为文件可能很大...")
    import sys
    sys.stdout.flush()
    
    try:
        with open(valid_images_file, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        print(f"✓ 加载完成，共 {len(valid_data)} 个人的数据")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 收集所有正面图关键点
    print("\n正在收集正面图关键点...")
    all_face_landmarks = []
    
    for idx, (person_name, person_data) in enumerate(valid_data.items()):
        if (idx + 1) % 10 == 0:
            print(f"  处理进度: {idx + 1}/{len(valid_data)}", end='\r')
            sys.stdout.flush()
        
        if 'face_landmarks_2d' in person_data:
            landmarks_2d = np.array(person_data['face_landmarks_2d'])
            if landmarks_2d.shape == (5, 2):  # 确保格式正确
                all_face_landmarks.append(landmarks_2d)
    
    print(f"\n✓ 收集完成")
    
    if len(all_face_landmarks) == 0:
        raise ValueError("未找到有效的正面图关键点数据")
    
    print(f"\n收集到 {len(all_face_landmarks)} 个正面图关键点")
    
    # 转换为numpy数组 [N, 5, 2]
    all_landmarks = np.array(all_face_landmarks)
    print(f"关键点数组形状: {all_landmarks.shape}")
    
    # 计算每个关键点的平均位置 [5, 2]
    # 对第一个维度（样本数）求平均
    mean_landmarks = np.mean(all_landmarks, axis=0)
    
    print(f"\n每个关键点的平均位置:")
    landmark_names = ['左眼', '右眼', '鼻尖', '左嘴角', '右嘴角']
    for i, name in enumerate(landmark_names):
        print(f"  {name}: ({mean_landmarks[i, 0]:.2f}, {mean_landmarks[i, 1]:.2f})")
    
    # 计算所有关键点的中心点（作为对齐点）
    # 中心点 = 所有关键点的平均位置
    center = np.mean(mean_landmarks, axis=0)
    
    print(f"\n中心点（对齐点）: ({center[0]:.2f}, {center[1]:.2f})")
    
    # 计算相对于中心点的标准坐标
    standard_landmarks = mean_landmarks - center
    
    print(f"\n标准坐标（相对于中心点）:")
    for i, name in enumerate(landmark_names):
        print(f"  {name}: ({standard_landmarks[i, 0]:.2f}, {standard_landmarks[i, 1]:.2f})")
    
    # 计算统计信息
    # 每个关键点的标准差
    std_landmarks = np.std(all_landmarks, axis=0)
    
    print(f"\n每个关键点的标准差（变异性）:")
    for i, name in enumerate(landmark_names):
        std_x = std_landmarks[i, 0]
        std_y = std_landmarks[i, 1]
        print(f"  {name}: X={std_x:.2f}, Y={std_y:.2f}")
    
    # 计算每个关键点到中心点的平均距离
    distances_to_center = []
    for landmarks in all_landmarks:
        # 计算每个样本的中心点
        sample_center = np.mean(landmarks, axis=0)
        # 计算每个关键点到中心点的距离
        distances = np.linalg.norm(landmarks - sample_center, axis=1)
        distances_to_center.append(distances)
    
    distances_to_center = np.array(distances_to_center)  # [N, 5]
    mean_distances = np.mean(distances_to_center, axis=0)  # [5]
    
    print(f"\n每个关键点到中心点的平均距离:")
    for i, name in enumerate(landmark_names):
        print(f"  {name}: {mean_distances[i]:.2f} 像素")
    
    # 计算标准坐标的统计信息
    standard_distances = np.linalg.norm(standard_landmarks, axis=1)
    
    print(f"\n标准坐标到原点的距离:")
    for i, name in enumerate(landmark_names):
        print(f"  {name}: {standard_distances[i]:.2f} 像素")
    
    # 构建统计信息字典
    statistics = {
        'num_samples': len(all_face_landmarks),
        'center': center.tolist(),
        'mean_landmarks': mean_landmarks.tolist(),
        'standard_landmarks': standard_landmarks.tolist(),
        'std_landmarks': std_landmarks.tolist(),
        'mean_distances_to_center': mean_distances.tolist(),
        'standard_distances': standard_distances.tolist(),
        'landmark_names': landmark_names
    }
    
    # 保存结果
    output_file = Path(valid_images_file).parent / 'standard_landmarks.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_file}")
    
    # 输出标准坐标（用于直接使用）
    print("\n" + "=" * 70)
    print("标准坐标（可直接使用）")
    print("=" * 70)
    print("\n标准关键点坐标（相对于中心点）:")
    print("standard_landmarks = np.array([")
    for i, name in enumerate(landmark_names):
        x, y = standard_landmarks[i]
        comma = "," if i < 4 else ""
        print(f"    [{x:8.4f}, {y:8.4f}],  # {name}{comma}")
    print("])")
    
    print(f"\n中心点坐标:")
    print(f"center = np.array([{center[0]:.4f}, {center[1]:.4f}])")
    
    print(f"\n平均关键点坐标（绝对坐标）:")
    print("mean_landmarks = np.array([")
    for i, name in enumerate(landmark_names):
        x, y = mean_landmarks[i]
        comma = "," if i < 4 else ""
        print(f"    [{x:8.4f}, {y:8.4f}],  # {name}{comma}")
    print("])")
    
    print("\n" + "=" * 70)
    print("计算完成！")
    print("=" * 70)
    
    return standard_landmarks, center, statistics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算标准关键点坐标')
    parser.add_argument('--valid_images_file', type=str,
                       default='train_transformer/valid_images.json',
                       help='valid_images.json文件路径')
    
    args = parser.parse_args()
    
    # 确保输出立即刷新
    import sys
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    try:
        standard_landmarks, center, statistics = calculate_standard_landmarks(
            valid_images_file=args.valid_images_file
        )
        sys.stdout.flush()
    except Exception as e:
        import traceback
        print(f"\n错误: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()

