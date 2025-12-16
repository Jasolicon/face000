@echo off
REM TensorBoard启动脚本（Windows）
REM 使用方法：双击此文件，或在命令行中运行

echo 正在启动TensorBoard...
echo 日志目录: train_transformer3D/gan_logs
echo.

REM 方法1：使用python -m tensorboard.main（推荐）
python -m tensorboard.main --logdir train_transformer3D/gan_logs --port 6006

REM 如果方法1失败，尝试方法2
if errorlevel 1 (
    echo.
    echo 方法1失败，尝试使用tensorboard命令...
    tensorboard --logdir train_transformer3D/gan_logs --port 6006
)

REM 如果都失败，提示用户
if errorlevel 1 (
    echo.
    echo ========================================
    echo 错误：无法启动TensorBoard
    echo ========================================
    echo.
    echo 请尝试以下方法：
    echo 1. 安装TensorBoard: pip install tensorboard
    echo 2. 使用Python模块: python -m tensorboard.main --logdir train_transformer3D/gan_logs
    echo 3. 检查Python环境是否正确
    echo.
    pause
)
