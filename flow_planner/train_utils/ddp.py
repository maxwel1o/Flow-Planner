import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
import subprocess

# 检测设备类型
def get_device_type():
    """检测可用的设备类型: npu, cuda, cpu"""
    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass
    
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def get_device_count(device_type=None):
    """获取设备数量"""
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'npu':
        return torch.npu.device_count()
    elif device_type == 'cuda':
        return torch.cuda.device_count()
    return 1

def set_device(device_id, device_type=None):
    """设置当前设备"""
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'npu':
        torch.npu.set_device(device_id)
    elif device_type == 'cuda':
        torch.cuda.set_device(device_id)


def ddp_setup_universal(verbose=False, cfg=None):
       # 检测设备类型
       device_type = get_device_type()
       
       if not cfg.ddp.distributed:
              print(f"do not use ddp, train on {device_type.upper()} 0")
              return 0, 0, 1
              
       if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
              rank = int(os.environ["RANK"])
              world_size = int(os.environ['WORLD_SIZE'])
              gpu = int(os.environ['LOCAL_RANK'])
              os.environ['MASTER_PORT'] = str(getattr(cfg.ddp, 'port', '29529'))
              os.environ["MASTER_ADDR"] = str(getattr(cfg.ddp, 'addr', 'localhost'))
       elif 'SLURM_PROCID' in os.environ:
              rank = int(os.environ['SLURM_PROCID'])
              gpu = rank % get_device_count(device_type)
              world_size = int(os.environ['SLURM_NTASKS'])
              node_list = os.environ['SLURM_NODELIST']
              num_devices = get_device_count(device_type)
              addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
              os.environ['MASTER_PORT'] = str(cfg.port)
              os.environ['MASTER_ADDR'] = addr
       else:
              print("Not using DDP mode")
              return 0, 0, 1

       os.environ['WORLD_SIZE'] = str(world_size)
       os.environ['LOCAL_RANK'] = str(gpu)
       os.environ['RANK'] = str(rank)  

       # 使用设备无关的set_device
       set_device(gpu, device_type)
       
       dist_backend = cfg.ddp.init_process_group
       dist_url = "env://"
       print('| distributed init (rank {}): {}, device {} ({})'.format(rank, dist_url, gpu, device_type.upper()), flush=True)
       print(f'| dist backend={dist_backend} world_size={world_size} rank={rank}')
       init_process_group(backend=dist_backend, world_size=world_size, rank=rank)
       torch.distributed.barrier()
       if verbose:
              setup_for_distributed(rank == 0)
       return rank, gpu, world_size


def setup_for_distributed(is_master):
       """
       This function disables printing when not in master process
       """
       import builtins as __builtin__
       builtin_print = __builtin__.print

       def print(*args, **kwargs):
              force = kwargs.pop('force', False)
              if is_master or force:
                     builtin_print(*args, **kwargs)

       __builtin__.print = print


def get_world_size():
       if not is_dist_avail_and_initialized():
              return 1
       return dist.get_world_size()


def get_rank():
       if not is_dist_avail_and_initialized():
              return 0
       return dist.get_rank()


def get_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    else:
        return model


def is_dist_avail_and_initialized():
       if not dist.is_available():
              return False
       if not dist.is_initialized():
              return False
       return True



def reduce_and_average_losses(loss_dict, device):
       torch.distributed.barrier()
       world_size = dist.get_world_size()
       for key in loss_dict.keys():
              loss_tensor = torch.tensor([loss_dict[key].item()]).to(device)
              dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
              loss_dict[key] = loss_tensor.item() / world_size
       return loss_dict

def gather_tensor(tensor, dst_rank=0):
    world_size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.gather(tensor, gather_list=tensor_list if get_rank() == dst_rank else None, dst=dst_rank)
    return tensor_list

