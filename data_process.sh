#!/bin/bash

###################################
# User Configuration Section
###################################

# nuPlan data paths
NUPLAN_DATA_PATH="/workspace/Flow-Planner/flow_planner/data/dataset/nuplan/data/cache/mini"
NUPLAN_MAP_PATH="/workspace/Flow-Planner/flow_planner/data/dataset/nuplan/maps"

# Output path
TRAIN_SET_PATH="./cache"

# Number of scenarios to process
TOTAL_SCENARIOS=10000

# Number of parallel workers (set to number of CPU cores)
NUM_WORKERS=4

###################################
# Data Processing Script
###################################

echo "=========================================="
echo "Starting data processing..."
echo "=========================================="
echo "nuPlan data path: $NUPLAN_DATA_PATH"
echo "nuPlan map path: $NUPLAN_MAP_PATH"
echo "Output path: $TRAIN_SET_PATH"
echo "Total scenarios: $TOTAL_SCENARIOS"
echo "Number of workers: $NUM_WORKERS"
echo "=========================================="

python data_process.py \
    --data_path $NUPLAN_DATA_PATH \
    --map_path $NUPLAN_MAP_PATH \
    --save_path $TRAIN_SET_PATH \
    --total_scenarios $TOTAL_SCENARIOS \
    --num_workers $NUM_WORKERS

echo "=========================================="
echo "Data processing completed!"
echo "=========================================="

