# Distributed Training with PyTorch

This project demonstrates distributed training of a ResNet model on the CIFAR-10 dataset using various parallelization techniques in PyTorch, including DataParallel (DP), DistributedDataParallel (DDP), FullyShardedDataParallel (FSDP), FairScale's FSDP, and DeepSpeed. It also monitors GPU metrics (utilization, memory usage, and power consumption) and generates plots.

## Prerequisites

- **Python**: 3.8 or higher
- **Hardware**: NVIDIA GPUs with CUDA support
- **Software**:
  - PyTorch
  - torchvision
  - fairscale
  - deepspeed
  - matplotlib
  - Slurm workload manager (for distributed modes other than DP)
  - NCCL (for distributed communication)

## Installation

1. **Clone the Repository**:
   ```bash
   cd distributed_training

#Run bash file