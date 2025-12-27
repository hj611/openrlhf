set -x

pkill -9 -f ray
 

eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate mm-eureka-hl
echo "conda activate mm-eureka-hl"

ls /workdir/export_gid_index.sh
source /workdir/export_gid_index.sh
 
# MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/qwen25vl-fix/stage3-all-from-stage2All-distill-osworld-opencua-drag-3img-7b'  ###TODO-1
MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/qwen25vl-fix/stage4-from-stage3-distill-osworld-claude-3img-think-click-drag-all-7b-1126'
 
# DATA_PATH='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/data/AndroidControl/Android_control_onlyAction_fix_previous_action_R1.jsonl,/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/jiangshu/data/output_train_R1_stage_1.jsonl'
# DATA_PROB='0.5,0.5'
# DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/UI/ZeroGUI-train/data_gen/UITAS15_4o_distill/train_json_drag_userPrompt_3img_r1_train_all_5w.jsonl'  ###TODO-2
DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/ZeroGUI-train/data/claude-sandbox-newVM-45-gen/claude_computer_use/screenshot/claude-4.5-openai/qwen25_abs_coord_937664pix/r1/3imgs_think_interleaved_r1_train.jsonl'
# DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/UI/ZeroGUI-train/data_gen/UITAS15_4o_distill/eval/uitars15_2node_global_step15_attemp5/train_json_drag_r1.jsonl'
 
OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/Ultron7b-distill-osworld-claude-3img-12k-multiAC'  ###TODO-3

EXP_ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/R1/src/MM-EUREKA_qwen

cd $EXP_ROOT

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_SOCKET_IFNAME=eth0

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

if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 30


if [ "$NODE_RANK" -eq 0 ]; then
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
  --working-dir $WORKING_DIR \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --remote_rm_url examples/scripts/reward_func_qwen_instruct_agent_addTypeParam_drag_3img_zero_all_multiAC.py \
  --actor_num_nodes 2 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 8 \
  --vllm_tensor_parallel_size 2 \
  --vllm_gpu_memory_utilization 0.95 \
  --ring_attn_size 1 \
  --pretrain ${MODEL_PATH} \
  --save_path ${OUTPUT_DIR} \
  --micro_train_batch_size 1 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 64 \
  --temperature 1.0 \
  --n_samples_per_prompt 8 \
  --lambd 1.0 \
  --gamma 1.0 \
  --max_epochs 1 \
  --num_episodes 1 \
  --prompt_max_len 12000 \
  --max_samples 100000 \
  --generate_max_len 4096 \
  --advantage_estimator group_norm \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.0 \
  --prompt_data ${DATA_PATH} \
  --disable_fast_tokenizer \
  --input_key message \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --save_steps 50 \
  --ckpt_path "${OUTPUT_DIR}/ckpt" \
  --max_ckpt_num 1 \
  --save_hf_ckpt \
  --freeze_prefix visual \
  --enable_accuracy_filter \
  --accuracy_lower_bound 0.1 \
  --accuracy_upper_bound 0.95 \
  --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
  --load_checkpoint | tee ${OUTPUT_DIR}/training.log
fi

ray stop

# --prompt_data_probs ${DATA_PROB}\
# sh /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/LLaMA-Factory-0512/examples/train25/20250811/train_agent_stage3_qwen25vl_72b_all_from_stage2os_distill_opencup_drag_3img_hl.sh