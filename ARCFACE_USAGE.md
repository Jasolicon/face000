# ArcFace使用指南

## 概述

ArcFace是一种先进的人脸识别模型，在某些场景下比FaceNet表现更好。本系统已集成ArcFace支持，可以与DINO和FaceNet一起使用。

## 安装依赖

ArcFace需要安装`insightface`库：

```bash
pip install insightface onnxruntime
```

或者安装完整依赖：

```bash
pip install -r requirements.txt
```

## 特征维度

- **ArcFace**: 512维（与FaceNet相同）
- **DINO**: 768维
- **FaceNet**: 512维

## 使用方法

### 方法1: 在test_file2.py中使用ArcFace

修改`test_file2.py`的`main()`函数：

```python
# 强制使用ArcFace
use_arcface = True

annotated_image, results = process_video_frame(
    video_path=video_path,
    features_dir='features_arcface',  # 使用单独的目录
    output_path=output_path,
    similarity_threshold=0.6,
    use_arcface=use_arcface  # 启用ArcFace
)
```

### 方法2: 使用test_arcface.py示例

直接运行ArcFace示例：

```bash
python test_arcface.py
```

### 方法3: 自动检测

如果不指定`use_arcface`，系统会自动检测特征数据库使用的提取器类型：

- 如果数据库是512维，且元数据标记为`arcface`，则自动使用ArcFace
- 否则使用FaceNet（默认）

## 完整流程

### 1. 使用ArcFace注册特征

```python
from face_detector import FaceDetector
from feature_extractor import ArcFaceFeatureExtractor
from feature_manager import FeatureManager

# 初始化
face_detector = FaceDetector()
feature_extractor = ArcFaceFeatureExtractor()
feature_manager = FeatureManager(storage_dir='features_arcface')

# 检测人脸并提取特征
aligned_face, box, prob = face_detector.extract_aligned_face('image.jpg')
features = feature_extractor.extract_features(aligned_face)

# 保存特征（需要在metadata中标记extractor_type）
feature_manager.save_feature(features, 'image.jpg', person_name='张三')
```

### 2. 使用ArcFace进行识别

```python
from test_file2 import process_video_frame

results = process_video_frame(
    video_path='video.mp4',
    features_dir='features_arcface',
    use_arcface=True
)
```

## 工作流程

1. **MTCNN检测人脸** - 使用MTCNN检测视频帧中的人脸
2. **裁剪人脸区域** - 从原始图像中裁剪出RGB人脸区域
3. **ArcFace提取特征** - 使用ArcFace对裁剪的人脸区域提取512维特征
4. **特征比对** - 与已保存的ArcFace特征库进行比对
5. **标注结果** - 在图像上绘制检测框和识别的人名

## 模型选择

ArcFace支持多种模型：

- `r50` - ResNet50（默认，推荐）
- `r100` - ResNet100（更准确但更慢）
- `r34` - ResNet34（更快但准确率略低）

在初始化时指定：

```python
extractor = ArcFaceFeatureExtractor(model_name='r100')
```

## 与FaceNet和DINO的对比

| 特性 | ArcFace | FaceNet | DINO |
|------|---------|---------|------|
| 特征维度 | 512 | 512 | 768 |
| 准确率 | 高 | 中-高 | 中 |
| 速度 | 中 | 快 | 慢 |
| 适用场景 | 人脸识别 | 人脸识别 | 图像检索 |
| 输入尺寸 | 112x112 | 160x160 | 224x224 |

## 注意事项

1. **特征库分离**: 建议为不同提取器使用不同的特征目录
   - `features_arcface` - ArcFace特征
   - `features_facenet` - FaceNet特征
   - `features_dino` - DINO特征

2. **维度匹配**: ArcFace和FaceNet都是512维，但特征空间不同，不能混用

3. **模型下载**: 首次使用insightface时，会自动下载模型文件（约100MB）

4. **GPU加速**: 如果有CUDA GPU，ArcFace会自动使用GPU加速

## 故障排除

### 问题1: insightface未安装

**错误**: `ImportError: No module named 'insightface'`

**解决**: 
```bash
pip install insightface onnxruntime
```

### 问题2: 模型下载失败

**错误**: 模型下载超时或失败

**解决**: 
- 检查网络连接
- 手动下载模型文件
- 使用代理

### 问题3: 特征维度不匹配

**错误**: `ValueError: 特征维度不匹配`

**解决**: 
- 确保注册和识别使用相同的特征提取器
- 检查特征数据库的维度

## 示例代码

完整示例请参考 `test_arcface.py`

