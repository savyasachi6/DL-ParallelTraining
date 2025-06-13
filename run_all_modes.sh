#!/bin/bash

# Define the modes to run
MODES=("DP" "DDP" "FSDP" "FairScaleFSDP" "DeepSpeed")

# Python script to execute
SCRIPT="main.py"

# Resource parameters
NODES=2
TASKS_PER_NODE=4  # Use 4 tasks per node for distributed modes to utilize all GPUs
GPUS_PER_NODE=4   # Each node has 4 GPUs (based on your nvidia-smi output)

# Ensure the plots folder exists
PLOTS_DIR="plots"
if [ ! -d "$PLOTS_DIR" ]; then
    mkdir -p "$PLOTS_DIR"
    echo "Created $PLOTS_DIR folder"
fi

# Loop over each mode and execute the appropriate srun command
for MODE in "${MODES[@]}"; do
    echo "------------------------------------------"
    echo "Running mode: $MODE"
    echo "------------------------------------------"

    if [ "$MODE" == "DP" ]; then
        # DP mode: Single process, multiple GPUs on one node
        srun --nodes=1 --ntasks-per-node=1 --gres=gpu:$GPUS_PER_NODE python $SCRIPT --mode $MODE > "${MODE}_output.log" 2>&1
    else
        # Distributed modes: Multiple nodes, 4 tasks per node to use all GPUs
        srun --nodes=$NODES --ntasks-per-node=$TASKS_PER_NODE --gres=gpu:$GPUS_PER_NODE python $SCRIPT --mode $MODE > "${MODE}_output.log" 2>&1
    fi

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Mode $MODE completed successfully."
        echo "Check ${MODE}_output.log for training details and '$PLOTS_DIR' for plots."
    else
        echo "Mode $MODE failed. Check ${MODE}_output.log for error details."
    fi

    # Brief delay to ensure resources are freed up between runs
    sleep 5
done

echo "------------------------------------------"
echo "All modes have been executed."
echo "Plots are saved in the '$PLOTS_DIR' folder."