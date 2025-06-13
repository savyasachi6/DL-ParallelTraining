# Distributed Training with PyTorch

This project demonstrates distributed training of a ResNet model on the CIFAR-10 dataset using various parallelization techniques in PyTorch, including DataParallel (DP), DistributedDataParallel (DDP), FullyShardedDataParallel (FSDP), FairScale's FSDP, and DeepSpeed. It also monitors GPU metrics (utilization, memory usage, and power consumption) and generates plots.

# 🧠 DL-ParallelTraining: Benchmarking Multi-GPU Strategies in PyTorch

This repository benchmarks and compares various **multi-GPU training strategies** for deep learning using PyTorch. We explore performance across **DataParallel**, **DistributedDataParallel (DDP)**, **Fully Sharded Data Parallel (FSDP)**, **FairScale FSDP**, and **DeepSpeed (ZeRO-1/2/3)** with real-world metrics.

📖 _Accompanies blog series:_
- Part 1: [Foundations & Concepts](https://bit.ly/4kxRrcO)
- Part 2: [Benchmarks & Metrics](https://bit.ly/3FXQxqU)

---

## 🚀 Project Overview

We trained a custom ResNet-50 on the **CIFAR-10** dataset under five distributed training setups:

| Strategy         | GPU Util (%) | Final Loss | Memory (GB) | Power (W) | Training Time (s) |
|------------------|---------------|------------|--------------|-----------|--------------------|
| DeepSpeed        | 44.76         | 2.0575     | 13.86        | 87.72     | 149.78             |
| DDP              | 56.67         | 2.0111     | 13.33        | 112.84    | 18.72              |
| FairScale FSDP   | 59.68         | 2.0077     | 13.20        | 112.45    | 19.99              |
| FSDP             | 52.68         | 2.0396     | 13.25        | 105.02    | 21.12              |
| DataParallel     | 13.32         | 2.2167     | 3.67         | 68.47     | 68.98              |

Each strategy was evaluated for:
- 🧠 **Training loss per epoch**
- 📊 **GPU utilization**
- 💾 **Memory usage**
- ⚡ **Power consumption**
- ⏱️ **Training time**

---

## 🏗️ Setup & Requirements

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install DeepSpeed (optional, for DeepSpeed benchmarking)
DS_BUILD_OPS=1 pip install deepspeed


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



#Run bash file