"""
下载 InsightFace 模型的脚本
支持断点续传和重试机制
"""
import os
import sys
import time
from pathlib import Path

# 设置模型下载镜像（在加载模型前）
try:
    from model_utils import setup_model_mirrors
    setup_model_mirrors()
except ImportError:
    # 如果 model_utils 不可用，直接设置环境变量
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_insightface_model(model_name='buffalo_l', max_retries=5, retry_delay=5):
    """
    下载 InsightFace 模型，支持重试和断点续传
    
    Args:
        model_name: 模型名称（默认: buffalo_l）
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒），每次重试会翻倍
    """
    print("=" * 70)
    print(f"下载 InsightFace 模型: {model_name}")
    print("=" * 70)
    
    from insightface.app import FaceAnalysis
    
    for attempt in range(max_retries):
        try:
            print(f"\n尝试 {attempt + 1}/{max_retries}...")
            
            # 尝试初始化检测器（会自动下载模型）
            print("正在初始化检测器（如果模型不存在会自动下载）...")
            detector = FaceAnalysis(name=model_name)
            
            # 准备检测器（使用CPU模式，避免GPU相关问题）
            print("正在准备检测器...")
            detector.prepare(ctx_id=-1, det_size=(640, 640))
            
            print("\n✓ 模型下载并初始化成功！")
            print(f"模型位置: ~/.insightface/models/{model_name}/")
            
            # 测试检测器
            print("\n正在测试检测器...")
            import numpy as np
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            faces = detector.get(test_image)
            print(f"✓ 检测器测试成功（检测到 {len(faces)} 个人脸）")
            
            print("\n" + "=" * 70)
            print("模型已准备就绪，可以在代码中使用")
            print("=" * 70)
            
            return detector
            
        except Exception as e:
            error_msg = str(e)
            is_download_error = any(keyword in error_msg.lower() for keyword in [
                'incompleteread', 'connection broken', 'chunkedencoding',
                'connection error', 'timeout', 'download', 'protocol error'
            ])
            
            if is_download_error and attempt < max_retries - 1:
                print(f"\n⚠️  下载失败（尝试 {attempt + 1}/{max_retries}）")
                print(f"   错误: {error_msg[:200]}")
                print(f"   等待 {retry_delay} 秒后重试...")
                print(f"   提示: InsightFace 支持断点续传，已下载的部分不会丢失")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
                continue
            else:
                print(f"\n❌ 下载失败: {error_msg}")
                if is_download_error:
                    print("\n解决方案:")
                    print("1. 检查网络连接")
                    print("2. 多次运行此脚本（支持断点续传）")
                    print("3. 手动下载模型:")
                    print("   - 模型目录: ~/.insightface/models/buffalo_l/")
                    print("   - 下载地址: https://github.com/deepinsight/insightface/releases")
                    print("   - 或设置环境变量指定模型路径:")
                    print("     export INSIGHTFACE_ROOT=/path/to/models")
                    print("4. 使用代理（如果网络受限）")
                raise


def check_model_exists(model_name='buffalo_l'):
    """检查模型是否已存在"""
    home = Path.home()
    model_dir = home / '.insightface' / 'models' / model_name
    
    if model_dir.exists():
        files = list(model_dir.glob('*'))
        if files:
            print(f"✓ 模型已存在: {model_dir}")
            print(f"  文件数量: {len(files)}")
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")
            return True
    
    print(f"✗ 模型不存在: {model_dir}")
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载 InsightFace 模型')
    parser.add_argument('--model', type=str, default='buffalo_l',
                       choices=['buffalo_l', 'buffalo_s', 'buffalo_m'],
                       help='模型名称（默认: buffalo_l）')
    parser.add_argument('--max_retries', type=int, default=5,
                       help='最大重试次数（默认: 5）')
    parser.add_argument('--retry_delay', type=int, default=5,
                       help='初始重试延迟（秒，默认: 5）')
    parser.add_argument('--check', action='store_true',
                       help='只检查模型是否存在，不下载')
    
    args = parser.parse_args()
    
    if args.check:
        check_model_exists(args.model)
    else:
        # 先检查模型是否已存在
        if check_model_exists(args.model):
            print("\n模型已存在，跳过下载")
            print("如需重新下载，请删除模型目录后重试")
        else:
            download_insightface_model(
                model_name=args.model,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )

