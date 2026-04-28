"""
NPU兼容性适配层
用于自动检测和切换GPU/NPU设备
"""

import os
import torch

# 检测NPU是否可用
def is_npu_available():
    """检测NPU设备是否可用"""
    try:
        import torch_npu
        return torch.npu.is_available()
    except ImportError:
        return False

# 检测CUDA是否可用
def is_cuda_available():
    """检测CUDA设备是否可用"""
    return torch.cuda.is_available()

# 获取设备类型
def get_device_type():
    """
    自动检测并返回可用的设备类型
    优先级: NPU > CUDA > CPU
    """
    if is_npu_available():
        return "npu"
    elif is_cuda_available():
        return "cuda"
    else:
        return "cpu"

# 获取设备数量
def get_device_count(device_type=None):
    """获取指定类型设备的数量"""
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "npu":
        return torch.npu.device_count()
    elif device_type == "cuda":
        return torch.cuda.device_count()
    else:
        return 1

# 设置当前设备
def set_device(device_id, device_type=None):
    """
    设置当前使用的设备
    
    Args:
        device_id: 设备ID
        device_type: 设备类型 ('npu', 'cuda', 'cpu')
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "npu":
        torch.npu.set_device(device_id)
    elif device_type == "cuda":
        torch.cuda.set_device(device_id)
    # CPU不需要设置设备

# 获取torch.device对象
def get_device(device_id=0, device_type=None):
    """
    获取torch.device对象
    
    Args:
        device_id: 设备ID
        device_type: 设备类型,如果为None则自动检测
    
    Returns:
        torch.device对象
    """
    if device_type is None:
        device_type = get_device_type()
    
    return torch.device(f"{device_type}:{device_id}")

# 张量转移到设备
def to_device(tensor, device=None):
    """
    将张量转移到指定设备
    
    Args:
        tensor: torch.Tensor对象
        device: 目标设备,如果为None则自动选择
    
    Returns:
        转移后的张量
    """
    if device is None:
        device = get_device()
    
    return tensor.to(device)

# 获取分布式后端
def get_dist_backend(device_type=None):
    """
    获取分布式训练的后端类型
    
    Args:
        device_type: 设备类型
    
    Returns:
        分布式后端字符串
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "npu":
        return "hccl"  # 华为集合通信库
    elif device_type == "cuda":
        return "nccl"  # NVIDIA集合通信库
    else:
        return "gloo"  # CPU通用后端

# 初始化设备环境
def init_device_env(device_type=None, device_id=0):
    """
    初始化设备环境
    
    Args:
        device_type: 设备类型
        device_id: 设备ID
    
    Returns:
        device_type, device对象
    """
    if device_type is None:
        device_type = get_device_type()
    
    # 设置设备
    if device_type in ["npu", "cuda"]:
        set_device(device_id, device_type)
    
    device = get_device(device_id, device_type)
    
    # 打印设备信息
    print(f"Using device: {device}")
    if device_type == "npu":
        print(f"NPU device count: {get_device_count(device_type)}")
    elif device_type == "cuda":
        print(f"CUDA device count: {get_device_count(device_type)}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device_id)}")
    
    return device_type, device

# 设备同步
def device_synchronize(device_type=None):
    """
    同步设备,等待所有操作完成
    
    Args:
        device_type: 设备类型
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "npu":
        torch.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()

# 清空设备缓存
def empty_cache(device_type=None):
    """
    清空设备缓存
    
    Args:
        device_type: 设备类型
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "npu":
        torch.npu.empty_cache()
    elif device_type == "cuda":
        torch.cuda.empty_cache()

# 获取设备内存信息
def get_memory_info(device_type=None, device_id=0):
    """
    获取设备内存信息
    
    Args:
        device_type: 设备类型
        device_id: 设备ID
    
    Returns:
        字典包含已分配内存和缓存内存
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == "npu":
        return {
            "allocated": torch.npu.memory_allocated(device_id),
            "cached": torch.npu.memory_reserved(device_id)
        }
    elif device_type == "cuda":
        return {
            "allocated": torch.cuda.memory_allocated(device_id),
            "cached": torch.cuda.memory_reserved(device_id)
        }
    else:
        return {
            "allocated": 0,
            "cached": 0
        }

# 设备兼容性装饰器
def device_compatible(func):
    """
    装饰器,自动处理设备兼容性
    """
    def wrapper(*args, **kwargs):
        # 自动检测设备类型
        if 'device' in kwargs and kwargs['device'] == 'cuda':
            # 如果指定了cuda但不可用,尝试切换到npu
            if not is_cuda_available() and is_npu_available():
                kwargs['device'] = 'npu'
                print(f"Warning: CUDA not available, switching to NPU")
        return func(*args, **kwargs)
    return wrapper

# 打印设备信息
def print_device_info():
    """打印详细的设备信息"""
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    
    # NPU信息
    print(f"\nNPU Available: {is_npu_available()}")
    if is_npu_available():
        print(f"NPU Device Count: {torch.npu.device_count()}")
        for i in range(torch.npu.device_count()):
            print(f"  NPU {i}: {torch.npu.get_device_name(i)}")
    
    # CUDA信息
    print(f"\nCUDA Available: {is_cuda_available()}")
    if is_cuda_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  CUDA {i}: {torch.cuda.get_device_name(i)}")
    
    # 当前选择
    print(f"\nSelected Device Type: {get_device_type()}")
    print(f"Distributed Backend: {get_dist_backend()}")
    print("=" * 50)

if __name__ == "__main__":
    # 测试设备检测
    print_device_info()

