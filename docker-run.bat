@echo off
REM Docker 快速启动脚本 (Windows)

echo ========================================
echo 多姿态人脸生成系统 - Docker 启动脚本
echo ========================================

REM 检查 Docker 是否安装
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker 未安装
    echo 请先安装 Docker Desktop: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM 检查 Docker Compose 是否安装
where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 警告: docker-compose 未安装，将使用 docker 命令
    set USE_COMPOSE=false
) else (
    set USE_COMPOSE=true
)

REM 创建必要的目录
echo 创建必要的目录...
if not exist features mkdir features
if not exist features_arcface mkdir features_arcface
if not exist generated_poses mkdir generated_poses
if not exist output mkdir output
if not exist input mkdir input

REM 构建镜像
echo 构建 Docker 镜像...
if "%USE_COMPOSE%"=="true" (
    docker-compose build
) else (
    docker build -t face-gen:latest .
)

REM 启动容器
echo 启动容器...
if "%USE_COMPOSE%"=="true" (
    docker-compose up -d
    echo 容器已启动！
    echo.
    echo 使用以下命令进入容器:
    echo   docker-compose exec face-gen bash
    echo.
    echo 查看日志:
    echo   docker-compose logs -f
    echo.
    echo 停止容器:
    echo   docker-compose down
) else (
    docker run -it --rm ^
        -v "%CD%\features:/app/features" ^
        -v "%CD%\features_arcface:/app/features_arcface" ^
        -v "%CD%\generated_poses:/app/generated_poses" ^
        -v "%CD%\output:/app/output" ^
        -v "%CD%\input:/app/input:ro" ^
        face-gen:latest ^
        bash
)

echo 完成！
pause

