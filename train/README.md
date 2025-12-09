# 多角度人脸识别训练系统

## 应用场景分析

### 场景：只有正脸图片，需要识别多角度人脸

**推荐方案：特征对比（Feature Comparison）**

#### 为什么特征对比更适合？

1. **灵活性**：
   - 不需要为每个人训练单独的模型
   - 可以动态添加新的人脸（只需保存正脸特征）
   - 支持few-shot学习（只需少量正脸样本）

2. **效率**：
   - 训练一次，适用于所有人
   - 推理时只需提取特征并对比
   - 计算成本低

3. **可扩展性**：
   - 新增人员只需添加正脸特征
   - 不需要重新训练模型
   - 适合大规模部署

4. **准确性**：
   - 通过训练学习角度不变性
   - 特征空间中对齐不同角度的人脸
   - 可以结合ArcFace等先进损失函数

#### 识别模型 vs 特征对比

| 特性 | 识别模型 | 特征对比 |
|------|---------|---------|
| 训练方式 | 为每个人训练 | 训练一次通用模型 |
| 新增人员 | 需要重新训练 | 只需添加特征 |
| Few-shot | 困难 | 容易 |
| 计算效率 | 低 | 高 |
| **适用场景** | **固定人员库** | **动态人员库（推荐）** |

**结论**：对于"只有正脸图片，识别多角度人脸"的场景，**特征对比方案更合适**。

## 模型架构

### 核心思想
学习一个角度不变的特征空间，使得：
- 同一人的正脸和多角度人脸在特征空间中距离近
- 不同人的特征在特征空间中距离远

### 架构设计

```
输入: 正脸图像 + 多角度人脸图像
  ↓
CLIP Vision Encoder (预训练)
  ↓
角度不变性模块 (Angle-Invariant Module)
  ↓
特征投影层 (Feature Projection)
  ↓
输出: 512维特征向量
```

### 损失函数

1. **ArcFace损失**：确保类内紧凑、类间分离
2. **对比损失**：正脸-多角度对拉近，不同人拉远
3. **角度一致性损失**：同一人的不同角度特征对齐

## 数据格式

```
train/
├── datas/
│   ├── person_001/
│   │   ├── front.jpg          # 正脸图片
│   │   └── video.mp4          # 多角度转头视频
│   ├── person_002/
│   │   ├── front.jpg
│   │   └── video.mp4
│   └── ...
```

## 使用方法

### 1. 准备数据

将数据按以下结构组织：
```
datas/
├── person_001/
│   ├── front.jpg
│   └── video.mp4
├── person_002/
│   ├── front.jpg
│   └── video.mp4
```

### 2. 训练模型

```bash
python train/train.py --data_dir train/datas --epochs 50
```

### 3. 使用模型

```python
from train.model import MultiAngleFaceModel

# 加载模型
model = MultiAngleFaceModel.load_from_checkpoint('checkpoints/best.ckpt')

# 提取特征
front_feature = model.extract_feature(front_image)
angle_feature = model.extract_feature(angle_image)

# 对比特征
similarity = cosine_similarity(front_feature, angle_feature)
```

