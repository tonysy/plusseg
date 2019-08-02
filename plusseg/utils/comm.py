# Copyright (c) ShanghaiTech PLUS Lab. All Rights Reserved.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time

import torch
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when using distributed training
    """

    if not dist.is_available():
        return 
    if not dist.is_initialized():
        return 
    world_size = dist.get_world_size()
    if world_size == 1:
        return 
    dist.barrier()

def all_gather(data):
    """
    Run all_gather on arbitray picklable data(not necessarily tensors)

    Args:
        data: any picklable object
    Returns:
        list[data: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range()]