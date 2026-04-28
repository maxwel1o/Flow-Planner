"""
NPU算子封装层
封装PyTorch操作以支持NPU
"""

import torch
from npu_compat import get_device_type, to_device

class NPUOps:
    """NPU算子封装类"""
    
    @staticmethod
    def softmax(x, dim=-1):
        """Softmax操作"""
        return torch.softmax(x, dim=dim)
    
    @staticmethod
    def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
        """Layer Normalization"""
        return torch.layer_norm(x, normalized_shape, weight, bias, eps)
    
    @staticmethod
    def gelu(x):
        """GELU激活函数"""
        return torch.nn.functional.gelu(x)
    
    @staticmethod
    def silu(x):
        """SiLU激活函数 (Swish)"""
        return torch.nn.functional.silu(x)
    
    @staticmethod
    def attention(q, k, v, mask=None, scale=None):
        """
        缩放点积注意力
        
        Args:
            q: Query张量 (B, H, N, D)
            k: Key张量 (B, H, N, D)
            v: Value张量 (B, H, N, D)
            mask: 注意力掩码
            scale: 缩放因子
        
        Returns:
            注意力输出
        """
        if scale is None:
            scale = 1.0 / (q.size(-1) ** 0.5)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn = torch.softmax(scores, dim=-1)
        
        # 加权求和
        output = torch.matmul(attn, v)
        
        return output

# 流匹配相关算子
class FlowMatchingOps:
    """流匹配相关算子封装"""
    
    @staticmethod
    def affine_path_sample(x_0, x_1, t, alpha_t=None, sigma_t=None):
        """
        仿射路径采样
        
        Args:
            x_0: 噪声样本
            x_1: 数据样本
            t: 时间参数 [0, 1]
            alpha_t: alpha函数
            sigma_t: sigma函数
        
        Returns:
            x_t: 插值样本
        """
        if alpha_t is None:
            # CondOT路径: alpha_t = t
            alpha_t = t
        if sigma_t is None:
            # CondOT路径: sigma_t = 1 - t
            sigma_t = 1 - t
        
        x_t = alpha_t * x_1 + sigma_t * x_0
        return x_t
    
    @staticmethod
    def velocity_to_target(velocity, x_t, t, path):
        """
        将速度场转换为目标预测
        
        Args:
            velocity: 速度场预测
            x_t: 当前样本
            t: 时间参数
            path: 路径对象
        
        Returns:
            目标预测
        """
        # 使用路径对象的转换方法
        return path.velocity_to_target(velocity, x_t, t)
    
    @staticmethod
    def cfg_velocity(velocity_cond, velocity_uncond, cfg_weight):
        """
        无分类器引导
        
        Args:
            velocity_cond: 条件速度
            velocity_uncond: 无条件速度
            cfg_weight: CFG权重
        
        Returns:
            引导后的速度
        """
        return velocity_uncond + cfg_weight * (velocity_cond - velocity_uncond)

# 数据处理相关算子
class DataProcessOps:
    """数据处理算子封装"""
    
    @staticmethod
    def normalize(data, mean, std):
        """
        数据归一化
        
        Args:
            data: 输入数据
            mean: 均值
            std: 标准差
        
        Returns:
            归一化后的数据
        """
        device = data.device
        return (data - mean.to(device)) / std.to(device)
    
    @staticmethod
    def denormalize(data, mean, std):
        """
        数据反归一化
        
        Args:
            data: 归一化数据
            mean: 均值
            std: 标准差
        
        Returns:
            原始尺度的数据
        """
        device = data.device
        return data * std.to(device) + mean.to(device)
    
    @staticmethod
    def rotation_matrix_2d(theta):
        """
        2D旋转矩阵
        
        Args:
            theta: 旋转角度
        
        Returns:
            旋转矩阵 (2, 2)
        """
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        rotation_matrix = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1)
        ], dim=-2)
        
        return rotation_matrix
    
    @staticmethod
    def transform_points(points, translation, rotation):
        """
        点变换
        
        Args:
            points: 点坐标 (..., 2)
            translation: 平移向量 (2,)
            rotation: 旋转角度或旋转矩阵
        
        Returns:
            变换后的点
        """
        if rotation.dim() == 0:
            # rotation是角度,构建旋转矩阵
            rotation_matrix = DataProcessOps.rotation_matrix_2d(rotation)
        else:
            rotation_matrix = rotation
        
        # 应用旋转和平移
        transformed = torch.matmul(points, rotation_matrix.transpose(-2, -1))
        transformed = transformed + translation
        
        return transformed

# 分布式训练相关
class DistributedOps:
    """分布式训练算子封装"""
    
    @staticmethod
    def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None):
        """
        全局归约操作
        
        Args:
            tensor: 输入张量
            op: 归约操作类型
            group: 进程组
        
        Returns:
            归约后的张量
        """
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor
    
    @staticmethod
    def all_gather(tensor_list, tensor, group=None):
        """
        全局收集操作
        
        Args:
            tensor_list: 输出张量列表
            tensor: 输入张量
            group: 进程组
        """
        if torch.distributed.is_initialized():
            torch.distributed.all_gather(tensor_list, tensor, group=group)
    
    @staticmethod
    def barrier():
        """同步屏障"""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

# 模型相关算子
class ModelOps:
    """模型相关算子封装"""
    
    @staticmethod
    def gradient_checkpointing(func, *args, **kwargs):
        """
        梯度检查点(用于节省内存)
        
        Args:
            func: 要执行的函数
            args: 函数参数
            kwargs: 函数关键字参数
        
        Returns:
            函数输出
        """
        from torch.utils.checkpoint import checkpoint
        return checkpoint(func, *args, **kwargs)
    
    @staticmethod
    def mixed_precision_forward(model, input, enabled=True):
        """
        混合精度前向传播
        
        Args:
            model: 模型
            input: 输入
            enabled: 是否启用混合精度
        
        Returns:
            模型输出
        """
        device_type = get_device_type()
        
        if enabled and device_type in ['cuda', 'npu']:
            with torch.autocast(device_type=device_type):
                return model(input)
        else:
            return model(input)

# 工具函数
def get_ops(device_type=None):
    """
    获取设备相关的算子
    
    Args:
        device_type: 设备类型
    
    Returns:
        算子对象
    """
    if device_type is None:
        device_type = get_device_type()
    
    return {
        'npu': NPUOps,
        'cuda': NPUOps,  # CUDA使用相同的算子
        'cpu': NPUOps
    }[device_type]

if __name__ == "__main__":
    # 测试算子
    print("Testing NPU Ops...")
    
    # 测试注意力
    q = torch.randn(2, 4, 10, 64)
    k = torch.randn(2, 4, 10, 64)
    v = torch.randn(2, 4, 10, 64)
    
    output = NPUOps.attention(q, k, v)
    print(f"Attention output shape: {output.shape}")
    
    # 测试归一化
    data = torch.randn(10, 3)
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([1.0, 1.0, 1.0])
    
    normalized = DataProcessOps.normalize(data, mean, std)
    print(f"Normalized data shape: {normalized.shape}")
    
    print("All tests passed!")

