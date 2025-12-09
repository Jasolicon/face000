# 人脸检测和特征提取系统

基于 `facenet_pytorch` 和 `DINO` 的人脸检测、图像特征提取、特征保存和比对系统。

## 功能特性

- ✅ **人脸检测**: 使用 MTCNN 进行高精度人脸检测
- ✅ **特征提取**: 
  - 使用 DINO 模型提取图像特征（默认）
  - 使用 FaceNet 提取人脸特征（可选）
- ✅ **特征管理**: 特征向量的保存、加载和管理
- ✅ **特征比对**: 基于余弦相似度的特征匹配和人员识别

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
face000/
├── face_detector.py          # 人脸检测模块
├── feature_extractor.py       # 特征提取模块（DINO和FaceNet）
├── feature_manager.py         # 特征保存和管理模块
├── feature_matcher.py         # 特征比对模块
├── main.py                    # 主程序和使用示例
├── requirements.txt           # 项目依赖
└── README.md                  # 说明文档
```

## 使用方法

### 基本使用

```python
from main import FaceRecognitionSystem

# 初始化系统（使用DINO提取图像特征）
system = FaceRecognitionSystem(use_facenet=False, storage_dir='features')

# 注册图像到数据库
system.register_image(
    'path/to/image.jpg',
    person_id='001',
    person_name='张三'
)

# 识别图像
matches = system.identify_image('path/to/query_image.jpg', top_k=5)

# 查看匹配结果
for match in matches:
    print(f"相似度: {match['similarity']:.4f}")
    print(f"人员: {match['metadata']['person_name']}")
```

### 使用FaceNet提取人脸特征

```python
# 初始化系统（使用FaceNet提取人脸特征）
system = FaceRecognitionSystem(use_facenet=True, storage_dir='features')

# 注册人脸图像
system.register_image('face_image.jpg', person_name='李四')

# 识别人脸
matches = system.identify_image('query_face.jpg')
```

### 批量注册图像

```python
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
person_names = ['张三', '李四', '王五']

results = system.register_batch(image_paths, person_names=person_names)
for img_path, success, message in results:
    print(f"{img_path}: {message}")
```

### 直接使用各个模块

#### 人脸检测

```python
from face_detector import FaceDetector

detector = FaceDetector()
faces, boxes, probs = detector.detect_faces('image.jpg')
```

#### 特征提取

```python
from feature_extractor import DINOFeatureExtractor

extractor = DINOFeatureExtractor()
features = extractor.extract_features('image.jpg')
```

#### 特征管理

```python
from feature_manager import FeatureManager

manager = FeatureManager(storage_dir='features')
manager.save_feature(features, 'image.jpg', person_name='张三')
features, metadata = manager.get_all_features()
```

#### 特征比对

```python
from feature_matcher import FeatureMatcher, FaceMatcher
from feature_manager import FeatureManager

manager = FeatureManager()
matcher = FaceMatcher(manager, similarity_threshold=0.7)
matches = matcher.match_face(query_features, top_k=5)
```

## 运行示例

```bash
python main.py
```

## 模块说明

### FaceDetector (face_detector.py)
- 使用 MTCNN 进行人脸检测
- 支持单张图像和批量检测
- 提供人脸对齐功能（用于FaceNet）

### FeatureExtractor (feature_extractor.py)
- **DINOFeatureExtractor**: 使用 DINO 模型提取图像特征
- **FaceNetFeatureExtractor**: 使用 FaceNet 提取人脸特征

### FeatureManager (feature_manager.py)
- 特征的保存和加载
- 元数据管理（图像路径、人员信息等）
- 支持批量操作

### FeatureMatcher (feature_matcher.py)
- 基于余弦相似度的特征比对
- 支持top-k检索和阈值匹配
- 提供完整的人脸识别流程

## 注意事项

1. **GPU支持**: 如果系统有CUDA支持的GPU，程序会自动使用GPU加速
2. **特征维度**: DINO和FaceNet提取的特征维度不同，不能混用
3. **相似度阈值**: 默认阈值为0.7，可根据实际需求调整
4. **图像格式**: 支持常见的图像格式（JPG, PNG等）

## 依赖说明

- `torch`: PyTorch深度学习框架
- `facenet-pytorch`: 人脸检测和识别
- `timm`: 预训练模型库（包含DINO）
- `Pillow`: 图像处理
- `numpy`: 数值计算
- `opencv-python`: 图像处理
- `transformers`: 模型加载（DINO可能需要）

## 许可证

本项目仅供学习和研究使用。

