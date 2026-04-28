export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=null
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT=/workspace/Flow-Planner
export SAVE_DIR=/workspace/Flow-Planner/save
export TENSORBOARD_LOG_PATH=/workspace/Flow-Planner/tensorboard_logs
export TRAINING_DATA=/workspace/Flow-Planner/cache
export TRAINING_JSON=/workspace/Flow-Planner/diffusion_planner_training.json
export TORCH_LOGS="dynamic,recompiles"

python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone flow_planner/trainer.py --config-name flow_planner_npu
