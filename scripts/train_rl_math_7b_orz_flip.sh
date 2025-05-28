export N_GPUS=8
export BASE_MODEL='/root/Qwen2.5-7B'

export RANDOM_FLIP=$1
export EXPERIMENT_NAME=math-qwen-7b-orz-group-flip-${RANDOM_FLIP}
export VLLM_ATTENTION_BACKEND=XFORMERS

math_test_path="orz_qwen_ins/math_test.parquet"
aime_test_path="orz_qwen_ins/aime_test.parquet"
gpqa_test_path="orz_qwen_ins/gpqa_test.parquet"

python3 -m verl.trainer.main_ppo \
    data.train_files="orz_qwen_ins/train.parquet" \
    data.val_files=[$math_test_path,$aime_test_path,$gpqa_test_path] \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=768 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.flip_prob=$RANDOM_FLIP \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.use_ref_policy=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=0 \
    critic.optim.lr=5e-6 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=16 \
    trainer.critic_warmup=20 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl_example_orz' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 2>&1 | tee verl_demo_${EXPERIMENT_NAME}.log