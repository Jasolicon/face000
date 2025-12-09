"""
下载 DINOv2 模型的脚本
"""
import torch

def download_dinov2_vitb14():
    """
    下载 DINOv2 ViT-B/14 模型
    模型会自动缓存到 ~/.cache/torch/hub/checkpoints/
    """
    print("=" * 70)
    print("下载 DINOv2 ViT-B/14 模型")
    print("=" * 70)
    
    print("\n正在从 Facebook Research 下载 DINOv2 ViT-B/14 模型...")
    print("模型将自动缓存，后续使用无需重新下载")
    
    try:
        # 使用 torch.hub 加载 DINOv2 模型
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model.eval()
        
        print("\n✓ 模型下载成功！")
        print(f"模型类型: {type(model)}")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试模型
        print("\n正在测试模型...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ 模型测试成功！")
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        
        print("\n" + "=" * 70)
        print("模型已下载并缓存，可以在代码中使用")
        print("=" * 70)
        
        return model
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n提示:")
        print("1. 确保网络连接正常")
        print("2. 确保已安装 torch 和 torchvision")
        print("3. 如果下载中断，可以重新运行此脚本")
        raise

if __name__ == "__main__":
    download_dinov2_vitb14()

