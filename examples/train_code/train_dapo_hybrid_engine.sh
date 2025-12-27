set -x

pkill -9 -f ray
ray stop
rm -rf /tmp/ray
eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate mm-eureka-hl
# echo "conda activate mm-eureka-hl"
conda activate openrlhf
unset http_proxy
unset https_proxy

# 
MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/model/Qwen3-4B'
# DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/code/CodeHacker-Plus/special_judge/batch_001.json'
DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/code/CodeHacker-Plus/train.json'
# OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_12k_32b'
# OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_12k_32b_all_bs32'
# OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_12k_A3B'
OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_24k_4B_fix'

EXP_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/code/OpenRLHF

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"


cd $EXP_ROOT
rm -rf /tmp/ray

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200

# OUTPUT_DIR='/absolute/path/to/output/dir'

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD

JOB_ARGS=($(python get_job_args_lc.py))
echo "JOB_ARGS: ${JOB_ARGS[@]}"

NNODES="${JOB_ARGS[0]}"
GPUS_PER_NODE="${JOB_ARGS[1]}"
MASTER_ADDR="${JOB_ARGS[2]}"
MASTER_PORT="${JOB_ARGS[3]}"
NODE_RANK="${JOB_ARGS[4]}"

echo "NNODES: ${NNODES}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NODE_RANK: ${NODE_RANK}"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# # python3 -m openrlhf.cli.train_ppo_ray \
# if [ "$NODE_RANK" -eq 0 ]; then
#    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8 
#    sleep 30 
#    RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
#    --working-dir $WORKING_DIR \
#    -- python3 -m openrlhf.cli.train_ppo_ray \
#    --ref_num_nodes 1 \
#    --ref_num_gpus_per_node 8 \
#    --actor_num_nodes 1 \
#    --actor_num_gpus_per_node 8 \
#    --vllm_num_engines 8 \
#    --vllm_tensor_parallel_size 1 \
#    --colocate_all_models \
#    --vllm_gpu_memory_utilization 0.6 \
#    --init_kl_coef 1e-3 \
#    --gamma 1.0 \
#    --use_kl_loss \
#    --kl_estimator k3 \
#    --advantage_estimator group_norm \
#    --dynamic_filtering \
#    --dynamic_filtering_reward_range 0 1 \
#    --eps_clip_low_high 0.2 0.3 \
#    --pretrain ${MODEL_PATH} \
#    --remote_rm_url /openrlhf/examples/python/reward_func.py \
#    --save_path ${OUTPUT_DIR} \
#    --ckpt_path "${OUTPUT_DIR}/ckpt" \
#    --save_steps 20 \
#    --save_hf_ckpt \
#    --micro_train_batch_size 8 \
#    --train_batch_size 128 \
#    --micro_rollout_batch_size 16 \
#    --rollout_batch_size 128 \
#    --n_samples_per_prompt 8 \
#    --max_epochs 1 \
#    --prompt_max_len 1024 \
#    --max_samples 20000 \
#    --generate_max_len 1024 \
#    --zero_stage 3 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --prompt_data OpenRLHF/prompt-collection-v0.1 \
#    --input_key context_messages \
#    --apply_chat_template \
#    --gradient_checkpointing \
#    --packing_samples \
#    --vllm_sync_backend nccl \
#    --enforce_eager \
#    --vllm_enable_sleep \
#    --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
#    --deepspeed_enable_sleep
# fi

# ray stop
# # You could also try
# #   --kl_estimator k2 \



if [ "$NODE_RANK" -eq 0 ]; then
   # 1. 启动 Ray Head
   # 显式加上 --include-dashboard=true 确保尝试启动面板
   ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8 --include-dashboard=true
   
   echo "Checking if Ray Dashboard is up on port $RAY_DASHBOARD_PORT..."

   # 2. 智能等待端口开启 (最多等 60秒)
   for i in {1..30}; do
       if netstat -tuln | grep -q ":$RAY_DASHBOARD_PORT "; then
           echo "Ray Dashboard is listening on port $RAY_DASHBOARD_PORT!"
           break
       fi
       echo "Waiting for Dashboard... ($i/30)"
       sleep 2
   done

   # 3. 关键修复：临时取消代理设置，防止 requests 走代理访问 localhost
   # 保存旧代理设置（如果需要）
   OLD_HTTP_PROXY=$http_proxy
   OLD_HTTPS_PROXY=$https_proxy
   
   unset http_proxy
   unset https_proxy
   export no_proxy="127.0.0.1,localhost,$MASTER_ADDR"

   # 4. 提交任务
   # 使用 --address 显式指定，并添加重试逻辑
   echo "Submitting job to http://127.0.0.1:$RAY_DASHBOARD_PORT"
   
   RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
   --working-dir $WORKING_DIR \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.8 \
   --gamma 1.0 \
   --advantage_estimator group_norm \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0 1 \
   --eps_clip_low_high 0.2 0.3 \
   --pretrain ${MODEL_PATH} \
   --remote_rm_url ./examples/python/reward_func_code.py \
   --save_path ${OUTPUT_DIR} \
   --ckpt_path "${OUTPUT_DIR}/ckpt" \
   --save_steps 2 \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 32 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 2048 \
   --max_samples 20000 \
   --generate_max_len 24000 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data ${DATA_PATH} \
   --input_key description \
   --label_key id \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
   --deepspeed_enable_sleep | tee ${OUTPUT_DIR}/training.log

   # 恢复代理（如果后续脚本需要）
   export http_proxy=$OLD_HTTP_PROXY
   export https_proxy=$OLD_HTTPS_PROXY
fi