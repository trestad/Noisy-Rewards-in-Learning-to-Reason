export N_GPUS=8
export BASE_MODEL='/root/Qwen2.5-7B'

export RM_PATH=$1
export EXPERIMENT_NAME=help-think-answer-sumup-qwen-7b-${RM_PATH}
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    data.train_files="help-think/train.parquet" \
    data.val_files="help-think/val.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1536 \
    data.max_response_length=6144 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.use_ref_policy=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=0 \
    reward_model.enable=True \
    reward_model.model.path=$RM_PATH \
    reward_model.micro_batch_size=8 \
    critic.optim.lr=5e-6 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=8 \
    trainer.critic_warmup=20 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl_example_help_think' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    +trainer.val_before_train=True \
    trainer.total_epochs=1 2>&1 | tee verl_demo_${EXPERIMENT_NAME}.log






export EXPERIMENT_NAME=help-think-answer-sumup-qwen-7b-${step}-0.1-rpr
export VLLM_ATTENTION_BACKEND=XFORMERS
export RM_PATH=${step}

python3 -m verl.trainer.main_ppo \
    data.train_files="help-think/train.parquet" \
    data.val_files=""help-think/val.parquet"" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=1536 \
    data.max_response_length=6144 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.use_ref_policy=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=0 \
    reward_model.enable=True \
    reward_model.model.path=$RM_PATH \
    reward_model.micro_batch_size=8 \
    critic.optim.lr=5e-6 \
    critic.model.path=$BASE_MODEL \
    critic.ppo_micro_batch_size=8 \
    trainer.critic_warmup=20 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl_example_help_think' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    +trainer.val_before_train=True \
    actor_rollout_ref.rollout.add_format_reward_when_wrong=True \
    trainer.total_epochs=1 2>&1 | tee verl_demo_${EXPERIMENT_NAME}.log
