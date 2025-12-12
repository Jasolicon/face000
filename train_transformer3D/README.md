# train_transformer3D

使用3D关键点和姿态进行Transformer训练的模块。

## 主要功能

### 1. 3D关键点和姿态提取

使用 **InsightFace** 提取2D关键点，然后通过 **PnP算法** 估计3D关键点和姿态（yaw, pitch, roll）。

### 2. 核心文件

- `utils_3d.py`: 3D关键点和姿态提取工具
  - `get_3d_landmarks_and_pose()`: 提取3D关键点和姿态
  - `estimate_pose_from_landmarks()`: 使用PnP算法估计姿态
  - `visualize_3d_landmarks_and_pose()`: 可视化3D关键点和姿态

- `filter_valid_images_3d.py`: 筛选和提取3D数据
  - 从视频帧和正面图中提取3D关键点和姿态
  - 保存到JSON文件供训练使用

## 使用方法

### 提取3D关键点和姿态

```bash
python train_transformer3D/filter_valid_images_3d.py \
    --video_dir /path/to/video \
    --face_dir /path/to/face \
    --output_file train_transformer3D/valid_images_3d.json
```

### 输出格式

JSON文件包含以下信息：

```json
{
  "人名": {
    "face_image_path": "相对路径/正面图.jpg",
    "face_features": [768维特征向量],
    "face_landmarks_2d": [[5, 2]],
    "face_landmarks_3d": [[5, 3]],
    "face_euler_angles": [yaw, pitch, roll],  // 度
    "face_rotation_matrix": [[3, 3]],
    "video_data": [
      {
        "image_path": "相对路径/视频帧.jpg",
        "features": [768维特征向量],
        "landmarks_2d": [[5, 2]],
        "landmarks_3d": [[5, 3]],
        "euler_angles": [yaw, pitch, roll],  // 度
        "rotation_matrix": [[3, 3]]
      }
    ]
  }
}
```

## 3D关键点和姿态说明

### 3D关键点

- 使用PnP算法从2D关键点估计3D位置
- 基于标准3D人脸模型（平均人脸）
- 输出：5个关键点的3D坐标 [5, 3]

### 姿态（Euler Angles）

- **Yaw**: 左右旋转角度（-90°到90°）
- **Pitch**: 上下旋转角度（-90°到90°）
- **Roll**: 平面旋转角度（-180°到180°）

### 旋转矩阵

- 3x3旋转矩阵，表示人脸的3D旋转
- 可用于精确的3D变换

## 与2D方法的区别

| 特性 | 2D方法 | 3D方法 |
|------|--------|--------|
| 关键点 | 2D坐标 [5, 2] | 3D坐标 [5, 3] |
| 角度信息 | 球面角（近似） | 欧拉角（精确） |
| 旋转表示 | 立体投影 | 旋转矩阵 |
| 精度 | 中等 | 高 |

## 注意事项

1. **标准3D模型**: 使用平均人脸的3D关键点位置作为参考
2. **相机参数**: 使用简化的相机内参（焦距=图像宽度）
3. **精度**: PnP算法需要至少4个非共线点，5个关键点足够
4. **姿态范围**: 极端角度（>90°）可能估计不准确
