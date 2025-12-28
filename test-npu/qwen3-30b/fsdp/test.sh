

# WORKSPACE_HOME and DATA_HOME support custom path configuration.
WORKSPACE_HOME=$pwd
DATA_HOME=$pwd

sp_size=1
tp_size=8
dp_size=2
enable_dp_attention=True
sync_mode=sync
train_prompt_bsz=16
train_prompt_mini_bsz=16

max_prompt_length=$((1024 * 2))
max_response_length=$((10))

CKPTS_DIR=/ckpt/Qwen3-30B-MoE-save
model_path=/ckpt/Qwen3-30B-MoE
train_data=/dapo-math-17k.parquet
valid_data=/dapo-math-17k.parquet


use_dynamic_bsz=False
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    data.val_files=$valid_data \
    data.return_raw_chat=True \
    data.train_batch_size=$train_prompt_bsz \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_prompt_mini_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tp_size \
    actor_rollout_ref.rollout.data_parallel_size=$dp_size \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_dp_attention=$enable_dp_attention \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.mode=$sync_mode \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.deepep_mode="auto" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_a2a_backend="none" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend="ascend" \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph="False" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.nccl_timeout=3600 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name='verl_grpo_example_2k_32k' \
    trainer.experiment_name='qwen3_30b_function_rm' \
    trainer.n_gpus_per_node=$NPUS_PER_NODE \
    trainer.nnodes=$NNODES  \
    trainer.save_freq=1000 \
    trainer.test_freq=10000 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    trainer.device=npu $@ 
