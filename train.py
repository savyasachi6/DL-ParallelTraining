import torch
import time
from torch.cuda.amp import autocast
import threading
import csv
from metrics import get_gpu_utilization, get_gpu_memory_usage, get_gpu_power, plot_metrics, sample_metrics

def train_model(model, trainloader, optimizer, criterion, epochs=50, parallel_mode="None", rank=0, local_rank=0):
    """
    Train the model with the specified parallel mode, monitoring GPU metrics.

    Args:
        model: The neural network model
        trainloader: Training data loader
        optimizer: Optimizer for training
        criterion: Loss function
        epochs (int): Number of epochs to train
        parallel_mode (str): Parallel training mode ("DP", "DDP", "DeepSpeed", or "None")
        rank (int): Rank of the current process in distributed mode
        local_rank (int): Local rank of the process on the node

    Returns:
        float: Total training time (on rank 0, 0 otherwise)
    """
    print(f"Rank {rank}: Starting training with mode {parallel_mode}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") if parallel_mode != "DP" else torch.device("cuda")
    if parallel_mode != "DeepSpeed":
        model.to(device)
    model.train()
    scaler = torch.amp.GradScaler('cuda') if parallel_mode != "DeepSpeed" else None
    torch.cuda.empty_cache()

    # Determine GPUs to monitor based on parallel mode
    if parallel_mode == "DP":
        num_gpus = torch.cuda.device_count()
        gpus_to_monitor = list(range(num_gpus))
    else:
        num_gpus = 1  # Each process monitors its own GPU
        gpus_to_monitor = [local_rank]

    world_size = torch.distributed.get_world_size() if parallel_mode != "DP" and torch.distributed.is_initialized() else 1

    # Initialize metric storage on rank 0 only
    if rank == 0:
        epoch_times = []
        epoch_losses = []
        gpu_utilizations = [[] for _ in range(world_size if parallel_mode != "DP" else num_gpus)]
        memory_usages = [[] for _ in range(world_size if parallel_mode != "DP" else num_gpus)]
        power_usages = [[] for _ in range(world_size if parallel_mode != "DP" else num_gpus)]
        actual_power_samples = [[] for _ in range(world_size if parallel_mode != "DP" else num_gpus)]

    for epoch in range(epochs):
        if parallel_mode != "DP" and hasattr(trainloader, 'sampler'):
            trainloader.sampler.set_epoch(epoch)

        start_time = time.time()

        # Start GPU metric sampling in a background thread
        util_samples = [[] for _ in gpus_to_monitor]
        mem_samples = [[] for _ in gpus_to_monitor]
        power_samples = [[] for _ in gpus_to_monitor]
        stop_event = threading.Event()
        sampling_thread = threading.Thread(target=sample_metrics, args=(stop_event, util_samples, mem_samples, power_samples, gpus_to_monitor))
        sampling_thread.start()

        # Training loop
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            if parallel_mode == "DeepSpeed":
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                model.backward(loss)
                model.step()
            else:
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            running_loss += loss.item()
            if i % 10 == 0 and rank == 0:
                print(f"Rank {rank}: [{parallel_mode}] Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
                print(f"Rank {rank}: GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB")

        # Compute local average loss
        avg_loss_local = running_loss / len(trainloader) if len(trainloader) > 0 else 0

        # Compute global average loss in distributed mode
        if parallel_mode != "DP" and torch.distributed.is_initialized():
            loss_tensor = torch.tensor(avg_loss_local, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_loss_global = loss_tensor.item() / world_size
        else:
            avg_loss_global = avg_loss_local

        if rank == 0:
            print(f"Rank {rank}: [{parallel_mode}] Epoch {epoch+1}, Loss: {avg_loss_global:.4f}")

        # Synchronize processes in distributed mode
        if parallel_mode != "DP" and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Stop sampling and compute averages
        stop_event.set()
        sampling_thread.join()

        avg_utils = [sum(samples) / len(samples) if samples else 0 for samples in util_samples]
        avg_mems = [sum(samples) / len(samples) if samples else 0 for samples in mem_samples]
        avg_powers = [sum(samples) / len(samples) if samples else 0 for samples in power_samples]

        # Handle metrics based on parallel mode
        if parallel_mode == "DP":
            if rank == 0:
                for gpu_id in range(num_gpus):
                    gpu_utilizations[gpu_id].append(avg_utils[gpu_id])
                    memory_usages[gpu_id].append(avg_mems[gpu_id])
                    power_usages[gpu_id].append(avg_powers[gpu_id])
                    actual_power_samples[gpu_id].append(power_samples[gpu_id])
        elif torch.distributed.is_initialized():
            # Gather metrics from all processes to rank 0
            metrics_tensor = torch.tensor([avg_utils[0], avg_mems[0], avg_powers[0]], device=device)
            all_metrics = [torch.zeros_like(metrics_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(all_metrics, metrics_tensor)
            # Gather actual power samples
            local_power_samples = power_samples[0]
            power_sample_len = len(local_power_samples)
            max_len = torch.tensor([power_sample_len], device=device)
            torch.distributed.all_reduce(max_len, op=torch.distributed.ReduceOp.MAX)
            max_len = max_len.item()
            padded_power_samples = local_power_samples + [0] * (max_len - power_sample_len)
            power_tensor = torch.tensor(padded_power_samples, device=device)
            all_power_samples = [torch.zeros_like(power_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(all_power_samples, power_tensor)
            if rank == 0:
                for r in range(world_size):
                    util, mem, power = all_metrics[r]
                    gpu_utilizations[r].append(util.item())
                    memory_usages[r].append(mem.item())
                    power_usages[r].append(power.item())
                    actual_power_samples[r].append([s.item() for s in all_power_samples[r][:power_sample_len]])

        if rank == 0:
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            epoch_losses.append(avg_loss_global)

    if rank == 0:
        training_time = sum(epoch_times)
        print(f"Rank {rank}: [{parallel_mode}] Total Training Time: {training_time:.2f}s")

        # Compute overall averages
        avg_gpu_utils = [sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0 for gpu_utils in gpu_utilizations]
        avg_power_utils = [sum(power_utils) / len(power_utils) if power_utils else 0 for power_utils in power_usages]

        # Save metrics to CSV
        with open(f"{parallel_mode}_metrics.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["Epoch", "Training Time (s)", "Loss"]
            for gpu_id in range(len(gpu_utilizations)):
                header.extend([f"GPU{gpu_id} Util (%)", f"GPU{gpu_id} Memory (GB)", f"GPU{gpu_id} Avg Power (W)"])
            writer.writerow(header)
            for epoch in range(epochs):
                row = [epoch + 1, epoch_times[epoch], epoch_losses[epoch]]
                for gpu_id in range(len(gpu_utilizations)):
                    row.extend([gpu_utilizations[gpu_id][epoch], memory_usages[gpu_id][epoch], power_usages[gpu_id][epoch]])
                writer.writerow(row)
            # Add average row
            avg_row = ["Average", training_time / epochs, sum(epoch_losses) / epochs]
            for gpu_id in range(len(gpu_utilizations)):
                avg_row.extend([avg_gpu_utils[gpu_id], sum(memory_usages[gpu_id]) / epochs, avg_power_utils[gpu_id]])
            writer.writerow(avg_row)
        print(f"Metrics saved to {parallel_mode}_metrics.csv")

        # Generate plots
        plot_metrics(epochs, epoch_times, memory_usages, gpu_utilizations, power_usages, avg_gpu_utils, avg_power_utils, actual_power_samples, parallel_mode, len(gpu_utilizations), epoch_losses)

    return training_time if rank == 0 else 0
