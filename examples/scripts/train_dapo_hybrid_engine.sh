set -x

pkill -9 -f ray
# ray stop
rm -rf /tmp/ray
eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate mm-eureka-hl
# echo "conda activate mm-eureka-hl"
conda activate openrlhf


MODEL_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_data/agent/model/QwenVL25_3B/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743'  ###TODO-1
 
DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/UI/ZeroGUI-train/data_gen/UITAS15_4o_distill/train_json_drag_userPrompt_3img_r1_train_all_5w.jsonl'  ###TODO-2
# DATA_PATH='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/UI/ZeroGUI-train/data_gen/UITAS15_4o_distill/eval/uitars15_2node_global_step15_attemp5/train_json_drag_r1.jsonl'
 
OUTPUT_DIR='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/output/agent/mm-eureka-out/code'  ###TODO-3

EXP_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/huangjing/proj/code/OpenRLHF

set -x

python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --dynamic_filtering \
   --dynamic_filtering_reward_range 0 1 \
   --eps_clip_low_high 0.2 0.3 \
   --pretrain ${MODEL_PATH} \
   --remote_rm_url /openrlhf/examples/python/reward_func.py \
   --save_path ${OUTPUT_DIR} \
   --ckpt_path "${OUTPUT_DIR}/ckpt" \
   --save_steps 20 \
   --save_hf_ckpt \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 20000 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --gradient_checkpointing \
   --packing_samples \
   --vllm_sync_backend nccl \
   --enforce_eager \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep

# You could also try
#   --kl_estimator k2 \
