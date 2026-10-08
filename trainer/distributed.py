from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training with torchrun requires CUDA devices.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend)
    return local_rank


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def reduce_mean(value, device):
    if not is_distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()
