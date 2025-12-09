"""
随机种子设置工具
确保所有随机操作的可重复性
"""
import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    设置所有随机种子，确保结果可重复
    
    Args:
        seed: 随机种子值（默认42）
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # PyTorch的确定性模式（可能影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ 随机种子已设置为: {seed}")


def set_deterministic_mode(seed: int = 42, deterministic: bool = True):
    """
    设置确定性模式（完全可重复，但可能影响性能）
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式
    """
    set_seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("✓ 已启用确定性模式（可能影响性能）")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("✓ 已启用性能优化模式（非完全确定性）")

