# Linux 中文字体安装说明

## 概述

为了在 Linux 系统上正确显示中文，需要安装中文字体。本项目已支持自动检测和使用常见的中文字体。

## 自动字体检测

项目中的 `font_utils.py` 会自动检测和使用以下 Linux 中文字体：

- **文泉驿微米黑** (WenQuanYi Micro Hei)
- **文泉驿正黑** (WenQuanYi Zen Hei)
- **Noto Sans CJK** (思源黑体)
- **Noto Serif CJK** (思源宋体)
- **Droid Sans Fallback**
- **文鼎字体** (AR PL UMing/UKai)

## 安装方法

### Ubuntu/Debian 系统

```bash
# 安装文泉驿字体
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# 或安装 Noto 字体（推荐，支持更多语言）
sudo apt-get install fonts-noto-cjk fonts-noto-cjk-extra

# 刷新字体缓存
fc-cache -fv
```

### CentOS/RHEL 系统

```bash
# 安装文泉驿字体
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts

# 或安装 Noto 字体
sudo yum install google-noto-cjk-fonts

# 刷新字体缓存
fc-cache -fv
```

### 手动安装字体

如果系统包管理器不可用，可以手动安装：

1. **下载字体文件**：
   - 文泉驿字体: https://github.com/anthonyfok/fonts-wqy-microhei
   - Noto 字体: https://www.google.com/get/noto/

2. **安装到系统目录**：
   ```bash
   # 创建字体目录
   sudo mkdir -p /usr/share/fonts/truetype/chinese
   
   # 复制字体文件
   sudo cp *.ttf /usr/share/fonts/truetype/chinese/
   
   # 设置权限
   sudo chmod 644 /usr/share/fonts/truetype/chinese/*
   
   # 刷新字体缓存
   sudo fc-cache -fv
   ```

3. **或安装到用户目录**：
   ```bash
   # 创建用户字体目录
   mkdir -p ~/.fonts
   
   # 复制字体文件
   cp *.ttf ~/.fonts/
   
   # 刷新字体缓存
   fc-cache -fv
   ```

## 验证字体安装

### 方法 1: 使用 fontconfig

```bash
# 列出所有中文字体
fc-list :lang=zh

# 查找特定字体
fc-list | grep -i "wqy\|noto\|droid"
```

### 方法 2: 使用项目工具

```bash
# 运行字体工具测试
python font_utils.py
```

这会显示：
- 操作系统信息
- PIL/Pillow 字体加载状态
- Matplotlib 字体设置状态
- 可用的中文字体列表

## 常见问题

### Q1: 字体安装后仍然显示方块

**解决方案**：
1. 确保刷新了字体缓存：`fc-cache -fv`
2. 重启 Python 进程（如果正在运行）
3. 检查字体是否正确安装：`fc-list :lang=zh`

### Q2: 找不到字体文件

**解决方案**：
1. 检查字体文件路径是否正确
2. 确保字体文件有读取权限
3. 尝试使用 `fontconfig` 查找字体：`fc-list | grep -i "fontname"`

### Q3: Docker 容器中字体问题

**解决方案**：
1. 在 Dockerfile 中安装字体：
   ```dockerfile
   RUN apt-get update && \
       apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei && \
       fc-cache -fv
   ```

2. 或挂载字体目录：
   ```yaml
   volumes:
     - /usr/share/fonts:/usr/share/fonts:ro
   ```

### Q4: 字体路径问题

项目会自动检测以下常见路径：
- `/usr/share/fonts/`
- `/usr/share/fonts/truetype/`
- `~/.fonts/`
- `~/.local/share/fonts/`

如果字体在其他位置，可以：
1. 创建符号链接到上述目录
2. 或修改 `font_utils.py` 添加自定义路径

## 推荐配置

### 最小配置（仅显示中文）

```bash
sudo apt-get install fonts-wqy-microhei
fc-cache -fv
```

### 完整配置（支持多种字体）

```bash
sudo apt-get install \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    fonts-noto-cjk \
    fonts-noto-cjk-extra
fc-cache -fv
```

## 测试

安装字体后，运行以下命令测试：

```bash
# 测试字体工具
python font_utils.py

# 测试训练脚本（会生成包含中文的图表）
python train_transformer/train.py --epochs 1
```

如果图表中的中文正常显示，说明字体配置成功。

