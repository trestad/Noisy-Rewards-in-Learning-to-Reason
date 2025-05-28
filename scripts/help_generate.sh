export N_GPUS=8
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL=$1
export OUTPUT_PATH=$2

python3 -m verl.trainer.main_generation \
data.path="help-think/val.parquet" \
data.n_samples=1 \
data.batch_size=48 \
data.prompt_key=prompt \
data.output_path=$OUTPUT_PATH \
model.path=$MODEL \
+model.trust_remote_code=True \
trainer.n_gpus_per_node=$N_GPUS \
rollout.prompt_length=1536 \
rollout.response_length=6144 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.5 \