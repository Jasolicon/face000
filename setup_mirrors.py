"""
模型镜像环境变量初始化脚本
在程序启动时运行此脚本，确保所有模型下载都使用镜像
"""
import os
import sys

def setup_all_mirrors():
    """
    设置所有模型下载镜像环境变量
    应该在导入任何可能下载模型的库之前调用
    """
    # HuggingFace 镜像（用于 timm、huggingface_hub 等）
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("✓ 已设置 HuggingFace 镜像: https://hf-mirror.com")
    
    # 设置 huggingface_hub 相关环境变量
    # 禁用 hf_transfer（如果镜像不支持）
    if 'HF_HUB_ENABLE_HF_TRANSFER' not in os.environ:
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    
    # 设置超时时间（避免连接超时）
    if 'HF_HUB_DOWNLOAD_TIMEOUT' not in os.environ:
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时
    
    # 设置重试次数
    if 'HF_HUB_DOWNLOAD_RETRIES' not in os.environ:
        os.environ['HF_HUB_DOWNLOAD_RETRIES'] = '5'
    
    return {
        'HF_ENDPOINT': os.environ.get('HF_ENDPOINT'),
        'HF_HUB_ENABLE_HF_TRANSFER': os.environ.get('HF_HUB_ENABLE_HF_TRANSFER'),
        'HF_HUB_DOWNLOAD_TIMEOUT': os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT'),
        'HF_HUB_DOWNLOAD_RETRIES': os.environ.get('HF_HUB_DOWNLOAD_RETRIES')
    }


# 自动执行（当作为模块导入时）
if __name__ != '__main__':
    setup_all_mirrors()


if __name__ == "__main__":
    print("=" * 70)
    print("模型镜像环境变量设置")
    print("=" * 70)
    
    mirrors = setup_all_mirrors()
    
    print("\n当前镜像设置:")
    for key, value in mirrors.items():
        if value:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: 未设置")
    
    print("\n提示:")
    print("1. 在 Python 脚本开头导入此模块以自动设置镜像:")
    print("   import setup_mirrors  # 会自动设置镜像")
    print("2. 或手动设置环境变量:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("3. 对于 timm 库，确保在导入 timm 之前设置 HF_ENDPOINT")

