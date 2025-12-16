# Git 历史清理脚本 - 移除大文件
# 警告：这个脚本会重写 Git 历史，请确保已经备份！

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "Git 历史清理脚本" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "这个脚本将从 Git 历史中移除以下大文件：" -ForegroundColor Cyan
Write-Host "  - train/datas/video/**/*.jpg (视频帧图片)" -ForegroundColor White
Write-Host "  - train_transformer*/checkpoint/best_model.pth (模型文件)" -ForegroundColor White
Write-Host "  - train_transformer*/valid_images*.json (大JSON文件)" -ForegroundColor White
Write-Host "  - train_transformer3D/gan_checkpoints/best_model.pth (GAN模型)" -ForegroundColor White
Write-Host "  - feature_clusters*.html (HTML可视化文件)" -ForegroundColor White
Write-Host ""
Write-Host "警告：这个操作会重写 Git 历史！" -ForegroundColor Red
Write-Host "建议先备份仓库或创建新分支！" -ForegroundColor Red
Write-Host ""

$confirm = Read-Host "是否继续？(yes/no)"
if ($confirm -ne "yes") {
    Write-Host "操作已取消" -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "步骤 1: 使用 git filter-branch 移除大文件..." -ForegroundColor Yellow
Write-Host "（这可能需要几分钟，请耐心等待）" -ForegroundColor Yellow
Write-Host ""

# 使用 git filter-branch 移除大文件
# 注意：在 Windows PowerShell 中，需要使用单引号来避免转义问题
$filterCmd = 'git filter-branch --force --index-filter "git rm --cached --ignore-unmatch -r train/datas/video train_transformer/checkpoint/best_model.pth train_transformer copy/checkpoint/best_model.pth train_transformer/valid_images.json train_transformer/valid_images_normalized.json train_transformer copy/valid_images.json train_transformer copy/valid_images_normalized.json train_transformer3D/gan_checkpoints/best_model.pth feature_clusters.html feature_clusters_by_pose.html" --prune-empty --tag-name-filter cat -- --all'

try {
    Invoke-Expression $filterCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n步骤 2: 清理引用和优化仓库..." -ForegroundColor Yellow
        git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
        git reflog expire --expire=now --all
        git gc --prune=now --aggressive
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "清理完成！" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "现在可以尝试推送：" -ForegroundColor Yellow
        Write-Host "  git push origin main --force" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "注意：由于重写了历史，需要使用 --force 推送" -ForegroundColor Yellow
        Write-Host "如果远程仓库有保护，可能需要先取消保护" -ForegroundColor Yellow
    } else {
        Write-Host "`n清理过程中出现错误！" -ForegroundColor Red
        Write-Host "错误代码: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n发生异常：" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
