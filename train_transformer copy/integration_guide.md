# Transformer模型集成指南

## 一、集成流程

将训练好的Transformer模型集成到图片检测流程中，完整流程如下：

```
1. InsightFace人脸检测
   ↓
2. 计算球面角（与参考正面图的球面角）
   ↓
3. DINOv2提取特征（从检测到的人脸区域）
   ↓
4. Transformer特征矫正（使用训练好的模型）
   ↓
5. 相似度计算（与特征库比对）
```

## 二、已创建的文件

### 1. **`train_transformer/load_transformer.py`**
- `load_transformer_model()`: 加载训练好的Transformer模型
- `correct_features_with_transformer()`: 使用Transformer矫正特征

### 2. **修改了 `test_file2.py`**
- 添加了 `transformer_model_path` 参数
- 在特征提取后添加了Transformer特征矫正步骤

## 三、使用方法

### 基本使用（不使用Transformer）
```python
annotated_image, results = process_video_frame(
    video_path=video_path,
    features_dir='features_224',
    similarity_threshold=0.25,
    reference_image_path=reference_image_path,
    frame_number=120
)
```

### 使用Transformer特征矫正
```python
annotated_image, results = process_video_frame(
    video_path=video_path,
    features_dir='features_224',
    similarity_threshold=0.25,
    reference_image_path=reference_image_path,
    frame_number=120,
    transformer_model_path='train_transformer/checkpoints/best_model.pth'  # 添加这个参数
)
```

## 四、工作流程详解

### 步骤1：InsightFace人脸检测
```python
detected_faces = insightface_detector.get(img_bgr)
# 获取边界框和关键点
box = face_info.bbox
landmarks = face_info.kps
```

### 步骤2：计算球面角
```python
# 与参考正面图的关键点计算球面角
angles, avg_angle = calculate_spherical_angle(
    reference_landmarks_3d,
    video_landmarks_3d,
    reference_landmarks_2d,
    video_landmarks
)
```

### 步骤3：DINOv2提取特征
```python
# 裁剪人脸区域
face_crop = pil_image.crop((x1, y1, x2, y2))
# 提取特征
features = feature_extractor.extract_features(face_crop)  # [768]
```

### 步骤4：Transformer特征矫正（新增）
```python
if transformer_model is not None:
    # 使用Transformer矫正特征
    corrected_features = correct_features_with_transformer(
        transformer_model,
        features,      # [768] DINOv2特征
        angles,        # [5] 球面角
        device=device
    )
    features = corrected_features  # 使用矫正后的特征
```

### 步骤5：相似度计算
```python
# 与特征库中的所有特征计算余弦相似度
cosine_similarities = np.dot(db_feats_norm, query_feat_norm)
best_similarity = np.max(cosine_similarities)
```

## 五、代码位置

### Transformer模型加载
- 文件：`train_transformer/load_transformer.py`
- 函数：`load_transformer_model()`, `correct_features_with_transformer()`

### 检测流程集成
- 文件：`test_file2.py`
- 函数：`process_video_frame()`
- 集成位置：第452-477行（DINOv2特征提取后）

## 六、注意事项

1. **参考图像必需**：Transformer特征矫正需要参考图像来计算球面角
2. **模型路径**：确保Transformer模型路径正确
3. **特征维度**：Transformer输入和输出都是768维，与DINOv2特征维度一致
4. **设备一致性**：确保Transformer模型和特征提取器使用相同的设备（CPU/GPU）

## 七、效果预期

使用Transformer特征矫正后：
- **提高相似度**：不同角度的特征经过矫正后，与正面图特征的相似度应该提高
- **提高识别准确率**：特别是对于侧脸、低头、抬头等非正面角度
- **角度不变性**：模型学习到了角度与特征的关系，能够将不同角度的特征映射到正面图特征空间

