tokenizer_project=tokenizer_training
tokenizer=detok_BB
token_channels=128

tokenizer_exp_name="$1"
num_register_tokens=0

force_one_d_seq=0
exp_name=ditddt_xl-${tokenizer_exp_name}
load_from=work_dirs/gen_model_training/$exp_name/checkpoints/epoch_0079.pth

project=gen_model_training
batch_size=128
epochs=100

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

echo "[INFO] nnodes=${WORLD_SIZE}, node_rank=${RANK}, nproc_per_node=${NPROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"
global_batch=$(( batch_size * WORLD_SIZE * NPROC_PER_NODE ))
echo "[INFO] per-GPU batch=${batch_size}, global batch=${global_batch}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${WORLD_SIZE:-1}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29501}" \
    main_diffusion.py \
    --project $project --exp_name $exp_name \
    --batch_size $batch_size --epochs $epochs \
    --pretrained_model_name_or_path "" \
    --token_channels $token_channels \
    --num_register_tokens $num_register_tokens \
    --tokenizer $tokenizer \
    --use_ema_tokenizer \
    --collect_tokenizer_stats --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --overwrite_stats \
    --load_tokenizer_from work_dirs/tokenizer_training/$tokenizer_exp_name/checkpoints/epoch_0199.pth \
    --load_from $load_from \
    --model DiTDDT_xl \
    --ditdh_sched \
    --force_one_d_seq $force_one_d_seq \
    --num_sampling_steps 50 \
    --cfg_list 1.0 \
    --evaluate \
    --keep_eval_folder \
    --eval_bsz 256 \
    --num_images 50000 \
    --data_path ./data/imagenet/train