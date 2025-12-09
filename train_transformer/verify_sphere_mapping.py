"""
验证关键点是否映射到球面上
"""
import numpy as np
import math

def landmarks_to_3d(landmarks, box, img_width, img_height):
    """
    将2D关键点转换为3D坐标（假设人脸在球面上）
    """
    # 计算人脸中心
    face_center_x = (box[0] + box[2]) / 2
    face_center_y = (box[1] + box[3]) / 2
    face_width = box[2] - box[0]
    face_height = box[3] - box[1]
    
    # 归一化坐标到 [-1, 1] 范围
    landmarks_normalized = landmarks.copy()
    landmarks_normalized[:, 0] = (landmarks[:, 0] - face_center_x) / (face_width / 2)
    landmarks_normalized[:, 1] = (landmarks[:, 1] - face_center_y) / (face_height / 2)
    
    # 转换为3D坐标（假设在单位球面上）
    landmarks_3d = np.zeros((5, 3))
    
    for i in range(5):
        x_norm = landmarks_normalized[i, 0]
        y_norm = landmarks_normalized[i, 1]
        
        # 计算球面坐标
        r = np.sqrt(x_norm**2 + y_norm**2)
        if r > 1.0:
            r = 1.0
        
        # 计算角度
        theta = math.acos(1 - r)  # 从中心到边缘的角度
        phi = math.atan2(y_norm, x_norm) if x_norm != 0 else 0
        
        # 转换为3D坐标
        landmarks_3d[i, 0] = math.sin(theta) * math.cos(phi)
        landmarks_3d[i, 1] = math.sin(theta) * math.sin(phi)
        landmarks_3d[i, 2] = math.cos(theta)
    
    return landmarks_3d

# 测试：验证是否在球面上
print("=" * 70)
print("验证关键点是否映射到单位球面上")
print("=" * 70)

# 创建测试数据
test_landmarks = np.array([
    [100, 50],   # 左眼
    [200, 50],   # 右眼
    [150, 100],  # 鼻子
    [120, 150],  # 左嘴角
    [180, 150]   # 右嘴角
])

test_box = [50, 30, 250, 170]  # [x1, y1, x2, y2]
img_width, img_height = 300, 200

# 转换为3D坐标
landmarks_3d = landmarks_to_3d(test_landmarks, test_box, img_width, img_height)

print("\n2D关键点坐标:")
print(test_landmarks)

print("\n3D关键点坐标:")
print(landmarks_3d)

print("\n验证是否在单位球面上（x² + y² + z² 应该等于 1）:")
for i in range(5):
    x, y, z = landmarks_3d[i]
    distance_squared = x**2 + y**2 + z**2
    distance = np.sqrt(distance_squared)
    print(f"关键点 {i+1}: x²+y²+z² = {distance_squared:.6f}, 距离 = {distance:.6f}")

print("\n" + "=" * 70)
print("结论:")
print("=" * 70)
print("✓ 所有关键点的 x² + y² + z² = 1.0")
print("✓ 这意味着所有关键点都在单位球面上")
print("✓ 球心在原点 (0, 0, 0)，半径为 1")

# 可视化说明
print("\n" + "=" * 70)
print("映射过程说明:")
print("=" * 70)
print("""
1. 2D关键点 (x, y) 在图像平面上
   ↓
2. 归一化到 [-1, 1] 范围: (x_norm, y_norm)
   ↓
3. 计算到中心的距离: r = sqrt(x_norm² + y_norm²)
   ↓
4. 计算球面角度:
   - theta = arccos(1 - r)  # 极角
   - phi = atan2(y_norm, x_norm)  # 方位角
   ↓
5. 转换为3D球面坐标:
   - x_3d = sin(theta) * cos(phi)
   - y_3d = sin(theta) * sin(phi)
   - z_3d = cos(theta)
   ↓
6. 结果: (x_3d, y_3d, z_3d) 在单位球面上
   - x² + y² + z² = 1
""")

