import subprocess
import threading
import time
import matplotlib.pyplot as plt
import os

def get_gpu_utilization(gpu_id):
    """Get GPU utilization percentage for a specific GPU."""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', str(gpu_id)])
        return float(result.decode('utf-8').strip())
    except Exception as e:
        print(f"Error getting GPU utilization for GPU {gpu_id}: {e}")
        return 0.0

def get_gpu_memory_usage(gpu_id):
    """Get GPU memory usage in GB for a specific GPU."""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu_id)])
        return float(result.decode('utf-8').strip()) / 1024  # Convert MB to GB
    except Exception as e:
        print(f"Error getting GPU memory usage for GPU {gpu_id}: {e}")
        return 0.0

def get_gpu_power(gpu_id):
    """Get GPU power consumption in watts for a specific GPU."""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits', '-i', str(gpu_id)])
        return float(result.decode('utf-8').strip())
    except Exception as e:
        print(f"Error getting GPU power for GPU {gpu_id}: {e}")
        return 0.0

def sample_metrics(stop_event, util_samples, mem_samples, power_samples, gpus_to_monitor, rate=1.0):
    """
    Sample GPU metrics periodically until stopped.

    Args:
        stop_event (threading.Event): Event to signal stopping
        util_samples (list of lists): Utilization samples per GPU
        mem_samples (list of lists): Memory samples per GPU
        power_samples (list of lists): Power samples per GPU
        gpus_to_monitor (list): List of GPU IDs to monitor
        rate (float): Sampling interval in seconds (default: 1.0)
    """
    while not stop_event.is_set():
        for gpu_id in gpus_to_monitor:
            util = get_gpu_utilization(gpu_id)
            mem = get_gpu_memory_usage(gpu_id)
            power = get_gpu_power(gpu_id)
            idx = gpus_to_monitor.index(gpu_id)
            util_samples[idx].append(util)
            mem_samples[idx].append(mem)
            power_samples[idx].append(power)
        time.sleep(rate)

def plot_metrics(epochs, times, memory_usages, gpu_utilizations, power_usages, avg_gpu_utilizations, avg_power_usages, actual_power_samples, parallel_mode, num_gpus, losses):
    """
    Plot combined training metrics over all epochs.

    Args:
        epochs (int): Number of epochs
        times (list): Training time per epoch
        memory_usages (list of lists): Memory usage per GPU per epoch
        gpu_utilizations (list of lists): GPU utilization per GPU per epoch
        power_usages (list of lists): Power usage per GPU per epoch
        avg_gpu_utilizations (list): Average GPU utilization per GPU over all epochs
        avg_power_usages (list): Average power usage per GPU over all epochs
        actual_power_samples (list of lists of lists): Actual power samples per GPU per epoch (not plotted individually)
        parallel_mode (str): Parallel training mode
        num_gpus (int): Number of GPUs
        losses (list): Loss per epoch
    """
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    epoch_nums = list(range(1, epochs + 1))
    colors = ['green', 'blue', 'red', 'purple']

    # Training Time Bar Plot
    plt.figure(figsize=(10, 5))
    plt.bar(epoch_nums, times, color='blue', label='Training Time (seconds)')
    plt.xlabel("Epoch")
    plt.ylabel("Training Time (seconds)")
    plt.title(f"Training Time per Epoch - {parallel_mode}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{parallel_mode}_training_time_bar.png'))
    plt.close()

    # Memory Usage Line Plot
    plt.figure(figsize=(10, 5))
    for gpu_id in range(num_gpus):
        plt.plot(epoch_nums, memory_usages[gpu_id], marker='o', color=colors[gpu_id % len(colors)], label=f'GPU {gpu_id} Memory (GB)')
    plt.xlabel("Epoch")
    plt.ylabel("GPU Memory Usage (GB)")
    plt.title(f"GPU Memory Usage per Epoch - {parallel_mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{parallel_mode}_memory_usage_line.png'))
    plt.close()

    # GPU Utilization Scatter Plot
    plt.figure(figsize=(10, 5))
    for gpu_id in range(num_gpus):
        plt.scatter(epoch_nums, gpu_utilizations[gpu_id], color=colors[gpu_id % len(colors)], label=f'GPU {gpu_id} Utilization (%)')
        plt.axhline(avg_gpu_utilizations[gpu_id], color=colors[gpu_id % len(colors)], linestyle='--', label=f'GPU {gpu_id} Avg: {avg_gpu_utilizations[gpu_id]:.2f}%')
    plt.xlabel("Epoch")
    plt.ylabel("GPU Utilization (%)")
    plt.title(f"GPU Utilization per Epoch - {parallel_mode}")
    plt.legend()
    plt.grid(True)  # Added grid for clarity
    plt.savefig(os.path.join(plots_dir, f'{parallel_mode}_gpu_utilization_scatter.png'))
    plt.close()

    # Average Power Usage Line Plot
    plt.figure(figsize=(10, 5))
    for gpu_id in range(num_gpus):
        plt.plot(epoch_nums, power_usages[gpu_id], marker='s', color=colors[gpu_id % len(colors)], label=f'GPU {gpu_id} Avg Power (W)')
        plt.axhline(avg_power_usages[gpu_id], color=colors[gpu_id % len(colors)], linestyle='--', label=f'GPU {gpu_id} Avg: {avg_power_usages[gpu_id]:.2f}W')
    plt.xlabel("Epoch")
    plt.ylabel("Average Power Usage (W)")
    plt.title(f"Average GPU Power Usage per Epoch - {parallel_mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{parallel_mode}_power_usage_line.png'))
    plt.close()

    # Loss Line Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_nums, losses, marker='o', color='red', label='Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss per Epoch - {parallel_mode}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{parallel_mode}_loss_line.png'))
    plt.close()

    print(f"Plots saved in '{plots_dir}' folder for mode {parallel_mode}")
