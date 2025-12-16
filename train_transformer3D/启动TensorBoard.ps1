# TensorBoard启动脚本（PowerShell）
# 使用方法：在PowerShell中运行 .\启动TensorBoard.ps1

Write-Host "正在启动TensorBoard..." -ForegroundColor Green
Write-Host "日志目录: train_transformer3D/gan_logs" -ForegroundColor Cyan
Write-Host ""

# 方法1：使用python -m tensorboard.main（推荐）
Write-Host "尝试方法1: python -m tensorboard.main" -ForegroundColor Yellow
try {
    python -m tensorboard.main --logdir train_transformer3D/gan_logs --port 6006
    exit 0
} catch {
    Write-Host "方法1失败，尝试方法2..." -ForegroundColor Yellow
}

# 方法2：使用tensorboard命令
try {
    tensorboard --logdir train_transformer3D/gan_logs --port 6006
    exit 0
} catch {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "错误：无法启动TensorBoard" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "请尝试以下方法：" -ForegroundColor Yellow
    Write-Host "1. 安装TensorBoard: pip install tensorboard" -ForegroundColor White
    Write-Host "2. 使用Python模块: python -m tensorboard.main --logdir train_transformer3D/gan_logs" -ForegroundColor White
    Write-Host "3. 检查Python环境是否正确" -ForegroundColor White
    Write-Host ""
    Write-Host "如果使用conda环境，请先激活环境：" -ForegroundColor Yellow
    Write-Host "   conda activate <your_env_name>" -ForegroundColor White
    Write-Host ""
    Read-Host "按Enter键退出"
    exit 1
}
