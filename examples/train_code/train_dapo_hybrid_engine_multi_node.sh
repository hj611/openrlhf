set -x

pkill -9 -f ray
ray stop
rm -rf /tmp/ray

eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate openrlhf
echo "conda activate openrlhf"

# 导入环境变量（如果有的话）
if [ -f /workdir/export_gid_index.sh ]; then
    source /workdir/export_gid_index.sh
fi

unset http_proxy
unset https_proxy

# ============================================================================
# 配置路径
# ============================================================================
# MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/model/Qwen3-8B'
# MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/model/Qwen3-30B-A3B-Thinking-2507'
# MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/model/Qwen3-32B'
MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/model/Qwen3-4B'
# DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/code/CodeHacker-Plus/special_judge/batch_001.json'
DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/code/CodeHacker-Plus/train.json'
# OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_12k_32b'
# OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_12k_32b_all_bs32'
# OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_12k_A3B'
OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code_24k_4B'

EXP_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/code/OpenRLHF

cd $EXP_ROOT

# ============================================================================
# 环境变量配置
# ============================================================================
export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_SOCKET_IFNAME=eth0
export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD

# ============================================================================
# 获取分布式训练参数
# ============================================================================
JOB_ARGS=($(python get_job_args_lc.py))
echo "JOB_ARGS: ${JOB_ARGS[@]}"

NNODES="${JOB_ARGS[0]}"
GPUS_PER_NODE="${JOB_ARGS[1]}"
MASTER_ADDR="${JOB_ARGS[2]}"
MASTER_PORT="${JOB_ARGS[3]}"
NODE_RANK="${JOB_ARGS[4]}"

echo "============================================================"
echo "分布式训练配置:"
echo "  NNODES: ${NNODES}"
echo "  GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "  MASTER_PORT: ${MASTER_PORT}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  RAY_MASTER_PORT: ${RAY_MASTER_PORT}"
echo "  RAY_DASHBOARD_PORT: ${RAY_DASHBOARD_PORT}"
echo "============================================================"

# 创建输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# ============================================================================
# 启动 Ray 集群
# ============================================================================

if [ "$NODE_RANK" -eq 0 ]; then
    # ========================================================================
    # Master 节点：启动 Ray Head
    # ========================================================================
    echo "🚀 启动 Ray Head 节点 (Node Rank: $NODE_RANK)"
    
    ray start --head \
        --port=$RAY_MASTER_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$RAY_DASHBOARD_PORT \
        --num-gpus $GPUS_PER_NODE \
        --include-dashboard=true
    
    echo "✓ Ray Head 已启动"
    echo "  Dashboard: http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}"
    echo "  Ray Address: ${MASTER_ADDR}:${RAY_MASTER_PORT}"
    
    # 等待 Dashboard 启动
    echo "⏳ 等待 Ray Dashboard 启动..."
    for i in {1..30}; do
        if netstat -tuln | grep -q ":$RAY_DASHBOARD_PORT "; then
            echo "✓ Ray Dashboard 已就绪 (端口 $RAY_DASHBOARD_PORT)"
            break
        fi
        echo "  等待中... ($i/30)"
        sleep 2
    done
    
    # 等待 worker 节点连接（如果是多机训练）
    if [ "$NNODES" -gt 1 ]; then
        echo "⏳ 等待 $((NNODES - 1)) 个 worker 节点连接..."
        sleep 30
        
        # 检查连接的节点数
        CONNECTED_NODES=$(ray status | grep "Total:" | awk '{print $2}')
        echo "✓ 当前连接节点数: $CONNECTED_NODES / $NNODES"
    fi
    
    # ========================================================================
    # 提交训练任务
    # ========================================================================
    echo ""
    echo "============================================================"
    echo "📤 提交训练任务到 Ray 集群"
    echo "============================================================"
    
    # 临时取消代理（避免访问 localhost 走代理）
    OLD_HTTP_PROXY=$http_proxy
    OLD_HTTPS_PROXY=$https_proxy
    unset http_proxy
    unset https_proxy
    export no_proxy="127.0.0.1,localhost,$MASTER_ADDR"
    
    RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
        --working-dir $WORKING_DIR \
        -- python3 -m openrlhf.cli.train_ppo_ray \
        --ref_num_nodes $NNODES \
        --ref_num_gpus_per_node $GPUS_PER_NODE \
        --actor_num_nodes $NNODES \
        --actor_num_gpus_per_node $GPUS_PER_NODE \
        --vllm_num_engines 8 \
        --vllm_tensor_parallel_size 2 \
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
        --deepspeed_enable_sleep \
        | tee ${OUTPUT_DIR}/training.log
    
    # 恢复代理设置 | tee ${OUTPUT_DIR}/training.log
    export http_proxy=$OLD_HTTP_PROXY
    export https_proxy=$OLD_HTTPS_PROXY
    
    echo ""
    echo "============================================================"
    echo "✅ 训练任务已提交"
    echo "============================================================"
    
else
    # ========================================================================
    # Worker 节点：连接到 Ray Head
    # ========================================================================
    echo "🔗 启动 Ray Worker 节点 (Node Rank: $NODE_RANK)"
    echo "  连接到 Master: ${MASTER_ADDR}:${RAY_MASTER_PORT}"
    
    # 等待 master 节点启动
    echo "⏳ 等待 Master 节点启动..."
    sleep 30
    
    # 启动 worker 节点（阻塞模式）
    ray start \
        --address="${MASTER_ADDR}:${RAY_MASTER_PORT}" \
        --num-gpus $GPUS_PER_NODE \
        --block
    
    echo "✓ Ray Worker 已连接到集群"
fi

# ============================================================================
# 清理（仅在 master 节点执行）
# ============================================================================
if [ "$NODE_RANK" -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "🧹 训练完成，清理 Ray 集群"
    echo "============================================================"
    
    # 等待一段时间确保日志写入完成
    sleep 10
    
    # 停止 Ray
    ray stop
    
    echo "✅ 所有任务完成"
fi