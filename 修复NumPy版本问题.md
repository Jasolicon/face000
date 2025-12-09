# 修复 NumPy 版本兼容性问题

## 问题描述

错误信息：
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
```

**原因**: PyTorch 2.2.2 使用 NumPy 1.x 编译，与 NumPy 2.x 不兼容。

## 解决方案

### 方案 1: 降级 NumPy（推荐）

```bash
# 卸载当前版本
pip uninstall numpy -y

# 安装兼容版本
pip install 'numpy>=1.24.0,<2.0.0'

# 或安装特定版本（推荐）
pip install numpy==1.26.4
```

### 方案 2: 使用 requirements.txt

```bash
# 重新安装所有依赖（会自动安装正确的 NumPy 版本）
pip install -r requirements.txt --force-reinstall numpy
```

### 方案 3: 使用 conda

```bash
conda install "numpy<2.0.0"
```

## 验证修复

```python
import numpy as np
print(f"NumPy 版本: {np.__version__}")
# 应该显示: 1.26.4 或类似的 1.x 版本
```

## 其他问题修复

### 1. 安装 diffusers

```bash
pip install diffusers>=0.21.0,<1.0.0
```

### 2. 安装 mediapipe（可选）

```bash
pip install mediapipe
```

### 3. 完整安装

```bash
# 修复 NumPy
pip install 'numpy>=1.24.0,<2.0.0'

# 安装缺失的依赖
pip install diffusers>=0.21.0,<1.0.0
pip install mediapipe  # 可选
```

## 快速修复脚本

创建 `fix_numpy.py`:

```python
import subprocess
import sys

print("正在修复 NumPy 版本...")

# 卸载当前版本
subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"])

# 安装兼容版本
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])

# 验证
import numpy as np
print(f"✓ NumPy 版本: {np.__version__}")
```

运行：
```bash
python fix_numpy.py
```

## 版本兼容性

| PyTorch | NumPy | 状态 |
|---------|-------|------|
| 2.2.2 | 1.24.0 - 1.26.4 | ✅ 兼容 |
| 2.2.2 | 2.0.0+ | ❌ 不兼容 |

## 注意事项

1. **NumPy 2.0+ 是重大更新**，许多库尚未完全支持
2. **PyTorch 2.2.2 明确要求 NumPy < 2.0.0**
3. **建议使用 NumPy 1.26.4**（稳定且兼容）

## 完整修复步骤

```bash
# 1. 修复 NumPy
pip install 'numpy>=1.24.0,<2.0.0'

# 2. 安装缺失依赖
pip install diffusers>=0.21.0,<1.0.0

# 3. 验证
python -c "import numpy; import torch; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}')"
```

