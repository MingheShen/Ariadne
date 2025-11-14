export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SWIFT_DISTRIBUTED_BACKEND=nccl
export GLOO_SOCKET_IFNAME=lo
export MAX_PIXELS=65536
export NPROC_PER_NODE=8
export OMP_NUM_THREADS=1
# export WANDB_BASE_URL=https://api.bandw.top

swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --reward_funcs external_r1v_acc external_r1v_format format\
    --reward_weights 1 0.1 0.1 \
    --torch_dtype bfloat16 \
    --dataset train_am.jsonl \
    --external_plugins plugin.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --max_steps 100000 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir GRPO_MAZE \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 8 \
    --temperature 1. \
    --repetition_penalty 1.1 \
    --system 'prompt.txt' \
    --deepspeed zero3 \
    --log_completions false \
    --train_type full \
    --report_to wandb \
