import os
import torch
import torch.distributed as dist
import subprocess
from datetime import timedelta

def setup_distributed():
    """
    Initialize the distributed training environment.
    
    Returns:
        tuple: (rank, local_rank, device)
    """
    try:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        print(f"SLURM_PROCID={rank}, SLURM_LOCALID={local_rank}, SLURM_NTASKS={world_size}")
    except KeyError:
        print("SLURM variables not found, falling back to single-process mode")
        rank = 0
        local_rank = 0
        world_size = 1

    os.environ['LOCAL_RANK'] = str(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(f"Rank {rank}: Set device to {device}, GPU: {torch.cuda.get_device_name(local_rank)}")

    if world_size > 1:
        if 'SLURM_NODELIST' in os.environ:
            node_list = os.environ['SLURM_NODELIST']
            master_addr = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list]).decode().splitlines()[0]
            os.environ['MASTER_ADDR'] = master_addr
            print(f"Rank {rank}: Set MASTER_ADDR to {master_addr} from SLURM_NODELIST")
        else:
            os.environ['MASTER_ADDR'] = 'localhost'
            print(f"Rank {rank}: SLURM_NODELIST not set, defaulting MASTER_ADDR to 'localhost'")

        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
            print(f"Rank {rank}: MASTER_PORT not set by SLURM, defaulting to '29500'")

        print(f"Rank {rank}: Initializing process group with world_size={world_size}, MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=60)
        )
        print(f"Rank {rank}: Process group initialized")
    else:
        print(f"Rank {rank}: Single-process mode, skipping process group initialization")

    return rank, local_rank, device