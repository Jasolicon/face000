# 多姿态人脸生成系统 Dockerfile
# 基于 PyTorch 2.2.2 + CUDA 11.8

FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 复制 requirements.txt
COPY requirements.txt /app/requirements.txt

# 安装 Python 依赖
# 注意：基础镜像已包含 torch 2.2.2
# 先安装 torchvision（需要与 torch 版本匹配，使用 CUDA 11.8 版本）
RUN pip install torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖（pip 会自动跳过已安装的 torch）
RUN pip install -r requirements.txt

# 复制项目文件
COPY . /app/

# 创建必要的目录
RUN mkdir -p /app/features \
    /app/features_arcface \
    /app/generated_poses \
    /app/output

# 设置权限
RUN chmod -R 755 /app

# 暴露端口（如果需要 Web 服务）
# EXPOSE 8000

# 默认命令
CMD ["python", "--version"]

