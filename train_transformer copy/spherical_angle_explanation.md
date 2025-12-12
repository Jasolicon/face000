# 球面角计算原理详解

## 一、概述

球面角用于衡量人脸在不同角度下，关键点相对于正面图的角度偏离。计算过程分为三个主要步骤：

1. **2D关键点检测**：使用 InsightFace 检测5个关键点
2. **2D到3D转换**：将2D关键点投影到3D球面坐标系
3. **角度计算**：计算两个3D向量之间的夹角，并判断正负号

---

## 二、详细计算步骤

### 步骤1：2D关键点检测

使用 InsightFace 检测人脸，获取5个关键点的2D坐标：

```python
landmarks_2d = [
    [x_left_eye, y_left_eye],      # 左眼
    [x_right_eye, y_right_eye],     # 右眼
    [x_nose, y_nose],               # 鼻子
    [x_left_mouth, y_left_mouth],   # 左嘴角
    [x_right_mouth, y_right_mouth]  # 右嘴角
]  # 形状: [5, 2]
```

同时获取人脸边界框：
```python
box = [x1, y1, x2, y2]  # 左上角和右下角坐标
```

---

### 步骤2：2D到3D球面坐标转换

#### 2.1 归一化坐标

将2D关键点坐标归一化到 `[-1, 1]` 范围：

```python
# 计算人脸中心
face_center_x = (box[0] + box[2]) / 2
face_center_y = (box[1] + box[3]) / 2
face_width = box[2] - box[0]
face_height = box[3] - box[1]

# 归一化到 [-1, 1]
x_norm = (landmark_x - face_center_x) / (face_width / 2)
y_norm = (landmark_y - face_center_y) / (face_height / 2)
```

**含义**：
- `x_norm = 0`：关键点在人脸中心（水平方向）
- `x_norm = ±1`：关键点在人脸边缘（水平方向）
- `y_norm = 0`：关键点在人脸中心（垂直方向）
- `y_norm = ±1`：关键点在人脸边缘（垂直方向）

#### 2.2 计算球面坐标

假设人脸在单位球面上，使用球面坐标系转换：

```python
# 计算到中心的距离
r = sqrt(x_norm² + y_norm²)  # 范围: [0, 1]

# 限制在单位圆内
if r > 1.0:
    r = 1.0

# 计算球面角度
theta = arccos(1 - r)  # 从球心到关键点的极角 [0, π]
phi = atan2(y_norm, x_norm)  # 方位角 [-π, π]
```

**几何意义**：
- `theta`：从球心到关键点的极角
  - `theta = 0`：在球心（人脸中心）
  - `theta = π/2`：在赤道（人脸边缘）
  - `theta = π`：在对面（理论上不会出现）

- `phi`：在水平面上的方位角
  - `phi = 0`：正右方
  - `phi = π/2`：正下方
  - `phi = -π/2`：正上方

#### 2.3 转换为3D笛卡尔坐标

```python
# 球面坐标转3D笛卡尔坐标
x_3d = sin(theta) * cos(phi)
y_3d = sin(theta) * sin(phi)
z_3d = cos(theta)

landmarks_3d = [x_3d, y_3d, z_3d]  # 形状: [5, 3]
```

**3D坐标含义**：
- `x_3d`：水平方向（左右）
- `y_3d`：垂直方向（上下）
- `z_3d`：深度方向（前后）

所有3D关键点都在单位球面上：`x² + y² + z² = 1`

---

### 步骤3：计算球面角度偏离

#### 3.1 计算角度大小

对于每个关键点，计算参考图像和当前图像的3D向量之间的夹角：

```python
# 获取两个3D向量
v1 = landmarks1_3d[i]  # 参考图像（正面图）的3D关键点
v2 = landmarks2_3d[i]  # 当前图像（不同角度）的3D关键点

# 归一化向量
v1_norm = v1 / ||v1||
v2_norm = v2 / ||v2||

# 计算点积
dot_product = v1_norm · v2_norm  # 范围: [-1, 1]

# 计算角度（弧度）
angle_radians = arccos(dot_product)  # 范围: [0, π]

# 转换为角度（度）
angle_magnitude = degrees(angle_radians)  # 范围: [0, 180°]
```

**几何意义**：
- `angle = 0°`：两个向量完全一致（无角度偏离）
- `angle = 90°`：两个向量垂直
- `angle = 180°`：两个向量完全相反（理论上不会出现）

#### 3.2 判断正负号（抬头/低头）

角度的大小表示偏离程度，但我们需要知道是"抬头"还是"低头"，因此需要判断正负号：

**方法1：使用2D关键点的y坐标变化**（优先）

```python
# 计算y坐标差值
y_diff = landmarks2_2d[i, 1] - landmarks1_2d[i, 1]

if y_diff < 0:
    # y坐标减小 → 向上移动 → 抬头 → 正数
    angle = +angle_magnitude
elif y_diff > 0:
    # y坐标增大 → 向下移动 → 低头 → 负数
    angle = -angle_magnitude
else:
    # 没有垂直移动
    angle = angle_magnitude if angle_magnitude > 0.1 else 0.0
```

**方法2：使用3D向量的y分量变化**（备用）

```python
# 计算3D向量的y分量差值
y_diff_3d = v2_norm[1] - v1_norm[1]

if y_diff_3d < -0.01:  # y分量减小 → 抬头
    angle = +angle_magnitude
elif y_diff_3d > 0.01:  # y分量增大 → 低头
    angle = -angle_magnitude
else:
    angle = angle_magnitude if angle_magnitude > 0.1 else 0.0
```

**正负号规则**：
- **正数（+）**：抬头（向上看）
- **负数（-）**：低头（向下看）
- **0**：无垂直移动或角度很小

#### 3.3 计算平均角度

```python
# 对5个关键点的角度求平均
angles = [angle_0, angle_1, angle_2, angle_3, angle_4]  # [5]
avg_angle = mean(angles)
```

---

## 三、完整计算流程示例

### 示例：计算正面图与侧脸图的球面角

**输入**：
- 正面图关键点：`landmarks_front_2d = [[100, 50], [200, 50], [150, 100], [120, 150], [180, 150]]`
- 侧脸图关键点：`landmarks_side_2d = [[80, 60], [180, 55], [130, 105], [110, 160], [170, 155]]`

**步骤1：转换为3D坐标**

正面图：
```python
# 归一化
x_norm = (100 - 150) / 50 = -1.0  # 左眼在左边缘
y_norm = (50 - 100) / 50 = -1.0   # 左眼在上边缘

# 球面坐标
r = sqrt((-1.0)² + (-1.0)²) = sqrt(2) ≈ 1.414 → 限制为 1.0
theta = arccos(1 - 1.0) = arccos(0) = π/2
phi = atan2(-1.0, -1.0) = -3π/4

# 3D坐标
x_3d = sin(π/2) * cos(-3π/4) = 1 * (-√2/2) = -√2/2
y_3d = sin(π/2) * sin(-3π/4) = 1 * (-√2/2) = -√2/2
z_3d = cos(π/2) = 0
```

侧脸图：
```python
# 类似计算...
# 假设得到：v2 = [x_3d_side, y_3d_side, z_3d_side]
```

**步骤2：计算角度**

```python
# 归一化向量
v1_norm = normalize([-√2/2, -√2/2, 0])
v2_norm = normalize([x_3d_side, y_3d_side, z_3d_side])

# 点积
dot_product = v1_norm · v2_norm = 0.7（假设值）

# 角度大小
angle_magnitude = arccos(0.7) ≈ 45.6°

# 判断正负号
y_diff = 60 - 50 = 10 > 0  # y坐标增大 → 向下移动
angle = -45.6°  # 低头
```

**输出**：
- `angles = [-45.6°, -42.3°, -38.1°, -40.2°, -43.5°]`（5个关键点的角度）
- `avg_angle = -41.9°`（平均角度，表示整体向下低头约42度）

---

## 四、关键点说明

### 5个关键点的含义

1. **左眼**：左眼中心点
2. **右眼**：右眼中心点
3. **鼻子**：鼻尖
4. **左嘴角**：左嘴角
5. **右嘴角**：右嘴角

### 角度值的典型范围

- **正面图（参考）**：所有角度为 `0°`
- **轻微抬头**：`+5°` 到 `+15°`
- **明显抬头**：`+15°` 到 `+30°`
- **轻微低头**：`-5°` 到 `-15°`
- **明显低头**：`-15°` 到 `-30°`
- **极端角度**：`±30°` 以上（较少见）

---

## 五、在Transformer中的应用

### 作为位置编码

球面角在Transformer中作为**位置编码**使用：

```python
# 输入
input_features = DINOv2_features  # [batch_size, 768]
angles = spherical_angles          # [batch_size, 5]

# 位置编码
position_encoding = AnglePositionalEncoding(d_model=768, angle_dim=5)
pe = position_encoding(angles)  # [batch_size, 768]

# 添加到输入特征
enhanced_features = input_features + pe  # [batch_size, 768]
```

**作用**：
- 告诉模型当前输入图像的角度信息
- 帮助模型理解不同角度下的特征差异
- 引导模型将不同角度的特征映射到正面图特征

---

## 六、数学公式总结

### 2D到3D转换

```
x_norm = (x - center_x) / (width / 2)
y_norm = (y - center_y) / (height / 2)
r = sqrt(x_norm² + y_norm²)
theta = arccos(1 - r)
phi = atan2(y_norm, x_norm)

x_3d = sin(theta) * cos(phi)
y_3d = sin(theta) * sin(phi)
z_3d = cos(theta)
```

### 角度计算

```
v1_norm = v1 / ||v1||
v2_norm = v2 / ||v2||
dot_product = v1_norm · v2_norm
angle = arccos(dot_product)

if y_diff < 0:
    angle = +angle  # 抬头
else:
    angle = -angle  # 低头
```

---

## 七、优势与局限性

### 优势

1. **几何直观**：基于3D几何，符合人脸在3D空间中的运动规律
2. **角度不变性**：不受图像缩放、平移影响
3. **方向信息**：正负号提供抬头/低头方向
4. **多关键点**：5个关键点提供丰富的角度信息

### 局限性

1. **假设简化**：假设人脸在球面上，实际人脸是复杂3D形状
2. **2D投影**：从2D图像推断3D信息存在误差
3. **极端角度**：极端角度下关键点可能被遮挡或检测失败
4. **光照影响**：光照变化可能影响关键点检测精度

---

## 八、代码位置

相关代码位于：
- `utils.py`：
  - `landmarks_to_3d()`：2D到3D转换
  - `calculate_spherical_angle()`：角度计算
- `train_transformer/models.py`：
  - `AnglePositionalEncoding`：角度位置编码模块

