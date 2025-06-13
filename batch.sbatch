#!/bin/bash
#SBATCH -J run_all_modes         # Job name
#SBATCH -o job_output_%j.log     # Output log file
#SBATCH -e job_error_%j.log      # Error log file
#SBATCH -N 2                     # Request 2 nodes
#SBATCH -n 8                     # Request 8 total tasks (4 per node)
#SBATCH --ntasks-per-node=4      # Explicitly set 4 tasks per node
#SBATCH -p rtx                   # RTX partition (4 GPUs per node)
#SBATCH -t 08:00:00              # Max execution time (4 hours)

# Load necessary modules
module load python3/3.9.2
module load cuda/12.2

source /home1/10504/lfw24/pytorchenv/bin/activate

MODES=("DP" "DDP" "FSDP" "FairScaleFSDP" "DeepSpeed")
SCRIPT="main.py"

PLOTS_DIR="plots"
if [ ! -d "$PLOTS_DIR" ]; then
    mkdir -p "$PLOTS_DIR"
    echo "Created $PLOTS_DIR folder"
fi

for MODE in "${MODES[@]}"; do
    echo "------------------------------------------"
    echo "Running mode: $MODE"
    echo "------------------------------------------"

    if [ "$MODE" == "DP" ]; then
        # DP mode: Single process, multiple GPUs on one node
        srun --nodes=1 --ntasks=1 python $SCRIPT --mode $MODE > "${MODE}_output.log" 2>&1
    else
        # Distributed modes: 2 nodes, 4 tasks per node (8 total GPUs)
        srun --nodes=2 --ntasks=8 python $SCRIPT --mode $MODE > "${MODE}_output.log" 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo "Mode $MODE completed successfully."
        echo "Check ${MODE}_output.log for training details and '$PLOTS_DIR' for plots."
    else
        echo "Mode $MODE failed. Check ${MODE}_output.log for error details."
    fi
    sleep 5
done

echo "------------------------------------------"
echo "All modes have been executed."
echo "Plots are saved in the '$PLOTS_DIR' folder."
