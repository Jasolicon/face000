"""
下载 DINOv2 模型的脚本
支持下载小模型 (dinov2_vits14) 和中等模型 (dinov2_vitb14)
支持下载到指定位置
"""
import torch
import argparse
import os
from pathlib import Path
import shutil

# 设置模型下载镜像（在导入 torch.hub 前）
try:
    from model_utils import setup_model_mirrors
    setup_model_mirrors()
except ImportError:
    # 如果 model_utils 不可用，直接设置环境变量
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_dinov2(model_name='dinov2_vits14', save_dir=None):
    """
    下载 DINOv2 模型
    
    Args:
        model_name: 模型名称
            - 'dinov2_vits14': 小模型 (ViT-S/14, 特征维度384, ~22M参数)
            - 'dinov2_vitb14': 中等模型 (ViT-B/14, 特征维度768, ~86M参数)
            - 'dinov2_vitl14': 大模型 (ViT-L/14, 特征维度1024, ~300M参数)
            - 'dinov2_vitg14': 超大模型 (ViT-G/14, 特征维度1536, ~1.1B参数)
        save_dir: 保存模型的目录（如果为None，则只下载到默认缓存位置）
    
    模型会自动缓存到 ~/.cache/torch/hub/checkpoints/
    如果指定了save_dir，模型文件也会复制到指定目录
    """
    model_info = {
        'dinov2_vits14': {'name': 'ViT-S/14', 'dim': 384, 'size': '小模型'},
        'dinov2_vitb14': {'name': 'ViT-B/14', 'dim': 768, 'size': '中等模型'},
        'dinov2_vitl14': {'name': 'ViT-L/14', 'dim': 1024, 'size': '大模型'},
        'dinov2_vitg14': {'name': 'ViT-G/14', 'dim': 1536, 'size': '超大模型'},
    }
    
    if model_name not in model_info:
        print(f"❌ 未知的模型名称: {model_name}")
        print(f"支持的模型: {', '.join(model_info.keys())}")
        return None
    
    info = model_info[model_name]
    print("=" * 70)
    print(f"下载 DINOv2 {info['name']} ({info['size']})")
    print("=" * 70)
    
    print(f"\n模型信息:")
    print(f"  - 名称: {info['name']}")
    print(f"  - 特征维度: {info['dim']}")
    print(f"  - 类型: {info['size']}")
    
    print(f"\n正在从 Facebook Research 下载 DINOv2 {info['name']} 模型...")
    print("模型将自动缓存，后续使用无需重新下载")
    
    try:
        # 使用 torch.hub 加载 DINOv2 模型
        model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)
        model.eval()
        
        print("\n✓ 模型下载成功！")
        print(f"模型类型: {type(model)}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"模型参数数量: {param_count:,}")
        print(f"模型大小: {param_count * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # 测试模型
        print("\n正在测试模型...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ 模型测试成功！")
        print(f"输入形状: {dummy_input.shape}")
        if isinstance(output, dict):
            if 'x_norm_clstoken' in output:
                print(f"输出形状: {output['x_norm_clstoken'].shape} (特征维度: {output['x_norm_clstoken'].shape[1]})")
            else:
                print(f"输出类型: dict, 键: {list(output.keys())}")
        else:
            print(f"输出形状: {output.shape} (特征维度: {output.shape[1]})")
        
        # 如果指定了保存目录，将模型保存到指定位置
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 保存模型文件
            model_file = save_path / f"{model_name}.pth"
            print(f"\n正在保存模型到: {model_file}")
            torch.save(model.state_dict(), model_file)
            print(f"✓ 模型已保存到: {model_file}")
            
            # 尝试找到并复制torch.hub下载的原始文件
            try:
                import torch.hub
                hub_dir = torch.hub.get_dir()
                checkpoints_dir = Path(hub_dir) / "checkpoints"
                
                # 查找可能的模型文件
                possible_files = list(checkpoints_dir.glob(f"*{model_name}*"))
                if not possible_files:
                    # 尝试查找dinov2相关的文件
                    possible_files = list(checkpoints_dir.glob("*dinov2*"))
                
                if possible_files:
                    # 复制找到的文件
                    for src_file in possible_files:
                        dst_file = save_path / src_file.name
                        if not dst_file.exists():
                            shutil.copy2(src_file, dst_file)
                            print(f"✓ 已复制模型文件: {src_file.name}")
                else:
                    print(f"⚠️  未找到torch.hub下载的原始模型文件")
                    print(f"   模型已保存为state_dict格式: {model_file}")
            except Exception as e:
                print(f"⚠️  无法复制原始模型文件: {e}")
                print(f"   模型已保存为state_dict格式: {model_file}")
        
        print("\n" + "=" * 70)
        print("模型已下载并缓存，可以在代码中使用")
        if save_dir:
            print(f"模型已保存到: {save_dir}")
        print("=" * 70)
        
        return model
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n提示:")
        print("1. 确保网络连接正常")
        print("2. 确保已安装 torch 和 torchvision")
        print("3. 如果下载中断，可以重新运行此脚本")
        raise

def download_all_models(save_dir=None):
    """
    下载所有常用模型（小模型和768维模型）
    
    Args:
        save_dir: 保存模型的目录
    """
    models_to_download = ['dinov2_vits14', 'dinov2_vitb14']
    
    print("=" * 70)
    print("批量下载 DINOv2 模型")
    print("=" * 70)
    print(f"\n将下载以下模型:")
    print(f"  1. dinov2_vits14 - 小模型 (384维)")
    print(f"  2. dinov2_vitb14 - 768维模型")
    print()
    
    if save_dir:
        print(f"模型将保存到: {save_dir}")
    else:
        print("模型将只下载到默认缓存位置")
    
    print("\n" + "=" * 70)
    
    downloaded_models = {}
    for i, model_name in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] 下载 {model_name}...")
        print("-" * 70)
        try:
            model = download_dinov2(model_name, save_dir=save_dir)
            downloaded_models[model_name] = model
        except Exception as e:
            print(f"❌ 下载 {model_name} 失败: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("批量下载完成")
    print("=" * 70)
    print(f"\n成功下载 {len(downloaded_models)}/{len(models_to_download)} 个模型")
    
    if save_dir:
        print(f"\n所有模型文件已保存到: {save_dir}")
    
    return downloaded_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载 DINOv2 模型')
    parser.add_argument('--model', type=str, default='dinov2_vits14',
                       choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                       help='要下载的模型名称（默认: dinov2_vits14 小模型，384维）')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存模型的目录（如果指定，模型会保存到该目录）')
    parser.add_argument('--all', action='store_true',
                       help='下载小模型和768维模型（dinov2_vits14 和 dinov2_vitb14）')
    args = parser.parse_args()
    
    # 如果指定了 --all，下载所有常用模型
    if args.all:
        download_all_models(save_dir=args.save_dir)
    else:
        # 默认下载小模型（dinov2_vits14）
        download_dinov2(args.model, save_dir=args.save_dir)

