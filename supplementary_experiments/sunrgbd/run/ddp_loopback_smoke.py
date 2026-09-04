#!/usr/bin/env python3
"""Verify single-node NCCL rendezvous through the loopback address."""

import json
import os

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    value = torch.tensor(float(dist.get_rank() + 1), device='cuda')
    dist.all_reduce(value)
    if dist.get_rank() == 0:
        print(json.dumps({
            'status': 'complete',
            'world_size': dist.get_world_size(),
            'all_reduce_sum': value.item(),
            'master_addr': os.environ.get('MASTER_ADDR'),
            'master_port': os.environ.get('MASTER_PORT'),
        }), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
