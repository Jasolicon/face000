"""
模型加载工具函数
支持设置镜像环境变量以加速模型下载
"""
import os
import sys

def setup_model_mirrors():
    """
    设置模型下载镜像环境变量
    用于加速模型下载（特别是在中国大陆）
    """
    # HuggingFace 镜像（用于 DINOv2、timm 等通过 HuggingFace 下载的模型）
    if 'HF_ENDPOINT' not in os.environ:
        # 使用 hf-mirror.com 镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("✓ 已设置 HuggingFace 镜像: https://hf-mirror.com")
    
    # 设置 huggingface_hub 相关环境变量
    # 确保 huggingface_hub 使用镜像
    if 'HF_HUB_ENABLE_HF_TRANSFER' not in os.environ:
        # 禁用 hf_transfer（如果镜像不支持）
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    
    # 设置超时时间（避免连接超时）
    if 'HF_HUB_DOWNLOAD_TIMEOUT' not in os.environ:
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时
    
    # PyTorch Hub 镜像（如果需要）
    # 注意：torch.hub 默认使用 GitHub，可以通过设置代理或使用镜像
    # 这里主要设置 HF_ENDPOINT，因为 DINOv2 可能通过 HuggingFace 下载
    
    # 其他可能需要的镜像设置
    # GITHUB_MIRROR（如果 torch.hub 需要）
    if 'GITHUB_MIRROR' not in os.environ:
        # 可以设置 GitHub 镜像（如果需要）
        # os.environ['GITHUB_MIRROR'] = 'https://ghproxy.com'
        pass
    
    return {
        'HF_ENDPOINT': os.environ.get('HF_ENDPOINT'),
        'GITHUB_MIRROR': os.environ.get('GITHUB_MIRROR')
    }


def setup_torch_hub_mirror():
    """
    设置 PyTorch Hub 镜像（如果需要）
    
    注意：torch.hub 主要从 GitHub 下载，可以通过以下方式加速：
    1. 设置代理
    2. 使用 GitHub 镜像
    3. 手动下载模型到缓存目录
    """
    # 如果设置了代理，torch.hub 会自动使用
    # 这里主要确保环境变量已设置
    
    # 可以设置 GitHub 代理（如果需要）
    # 例如使用 ghproxy.com
    if 'GITHUB_MIRROR' not in os.environ:
        # 不强制设置，让用户根据需要配置
        pass


if __name__ == "__main__":
    print("=" * 70)
    print("模型镜像环境变量设置")
    print("=" * 70)
    
    mirrors = setup_model_mirrors()
    
    print("\n当前镜像设置:")
    for key, value in mirrors.items():
        if value:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: 未设置")
    
    print("\n提示:")
    print("1. HuggingFace 镜像已自动设置（如果未设置）")
    print("2. 如需使用其他镜像，可以设置环境变量:")
    print("   export HF_ENDPOINT=https://your-mirror.com")
    print("3. 对于 torch.hub，可以通过代理或手动下载加速")

