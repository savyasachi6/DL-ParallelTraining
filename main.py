import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import FullyShardedDataParallel as FairScaleFSDP
import deepspeed
import argparse
from data import get_dataloaders
from model import SimpleResNet
from train import train_model
from utils import setup_distributed

if __name__ == "__main__":
    # Force unbuffered output for better logging in distributed environments
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    parser = argparse.ArgumentParser(description="Run distributed training with different parallel modes")
    parser.add_argument('--mode', type=str, default='DDP', choices=['DP', 'DDP', 'FSDP', 'FairScaleFSDP', 'DeepSpeed'], help="Parallel training mode")
    args = parser.parse_args()

    parallel_mode = args.mode
    rank, local_rank, device = setup_distributed()
    if parallel_mode == 'DP':
        world_size = 1
    else:
        world_size = int(os.environ['SLURM_NTASKS']) if 'SLURM_NTASKS' in os.environ else 1
    trainloader, testloader = get_dataloaders(rank, world_size=world_size)

    if parallel_mode == 'DP':
        model = SimpleResNet().to(device)
        model = nn.DataParallel(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    elif parallel_mode == 'DDP':
        model = SimpleResNet().to(device)
        model = DDP(model, device_ids=[local_rank])
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    elif parallel_mode == 'FSDP':
        model = SimpleResNet().to(device)
        model = FSDP(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    elif parallel_mode == 'FairScaleFSDP':
        model = SimpleResNet().to(device)
        model = FairScaleFSDP(model)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
    elif parallel_mode == 'DeepSpeed':
        ds_config = {
            "train_micro_batch_size_per_gpu": 256,
            "zero_optimization": {"stage": 2},
            #"fp16": {"enabled": True}
        }
        model = SimpleResNet()
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            optimizer=optim.Adam(model.parameters(), lr=0.001),
            dist_init_required=False
        )
        criterion = nn.CrossEntropyLoss()

    train_model(model, trainloader, optimizer, criterion, parallel_mode=parallel_mode, rank=rank, local_rank=local_rank)

    if parallel_mode != 'DP' and world_size > 1:
        dist.destroy_process_group()
