"""
随机种子和确定性模式设置工具
统一提供 set_seed 和 set_deterministic_mode 函数
"""
import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """
    设置随机种子，确保结果可复现
    
    Args:
        seed: 随机种子值，默认为42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置Python哈希随机化（如果可用）
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_deterministic_mode():
    """
    设置确定性模式，确保结果完全可复现
    注意：这可能会降低性能
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 注意：torch.use_deterministic_algorithms(True) 在某些操作上可能不支持
    # 如果遇到错误，可以注释掉下面这行
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

