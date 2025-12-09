#!/bin/bash
# Docker 快速启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}多姿态人脸生成系统 - Docker 启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}警告: docker-compose 未安装，将使用 docker 命令${NC}"
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

# 检查 GPU 支持（Linux）
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}检测到 NVIDIA GPU${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}未检测到 NVIDIA GPU，将使用 CPU 模式${NC}"
    GPU_AVAILABLE=false
fi

# 创建必要的目录
echo -e "${GREEN}创建必要的目录...${NC}"
mkdir -p features features_arcface generated_poses output input

# 构建镜像
echo -e "${GREEN}构建 Docker 镜像...${NC}"
if [ "$USE_COMPOSE" = true ]; then
    docker-compose build
else
    docker build -t face-gen:latest .
fi

# 启动容器
echo -e "${GREEN}启动容器...${NC}"
if [ "$USE_COMPOSE" = true ]; then
    docker-compose up -d
    echo -e "${GREEN}容器已启动！${NC}"
    echo -e "${YELLOW}使用以下命令进入容器:${NC}"
    echo "  docker-compose exec face-gen bash"
    echo ""
    echo -e "${YELLOW}查看日志:${NC}"
    echo "  docker-compose logs -f"
    echo ""
    echo -e "${YELLOW}停止容器:${NC}"
    echo "  docker-compose down"
else
    if [ "$GPU_AVAILABLE" = true ]; then
        docker run --gpus all -it --rm \
            -v $(pwd)/features:/app/features \
            -v $(pwd)/features_arcface:/app/features_arcface \
            -v $(pwd)/generated_poses:/app/generated_poses \
            -v $(pwd)/output:/app/output \
            -v $(pwd)/input:/app/input:ro \
            face-gen:latest \
            bash
    else
        docker run -it --rm \
            -v $(pwd)/features:/app/features \
            -v $(pwd)/features_arcface:/app/features_arcface \
            -v $(pwd)/generated_poses:/app/generated_poses \
            -v $(pwd)/output:/app/output \
            -v $(pwd)/input:/app/input:ro \
            face-gen:latest \
            bash
    fi
fi

echo -e "${GREEN}完成！${NC}"

