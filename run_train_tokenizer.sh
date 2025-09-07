project=tokenizer_training
exp_name='detokBB-g3.0-m0.7-200ep-auxdinov3siglip'
batch_size=64
data_path=./data/imagenet/train
model=detok_BB
aux_model_type="dinov3,siglip"
epochs=200
discriminator_start_epoch=100
gamma=0.3
mask_ratio=0.7

GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-$GPUS_PER_NODE}

torchrun \
  --nnodes="${NUM_NODES:-1}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK:-0}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29500}" \
  main_reconstruction.py \
  --project "${project}" --exp_name "${exp_name}" --auto_resume \
  --batch_size "${batch_size}" --model "${model}" \
  --aux_model_type "${aux_model_type}" \
  --gamma "${gamma}" --mask_ratio "${mask_ratio}" \
  --online_eval \
  --epochs "${epochs}" --discriminator_start_epoch "${discriminator_start_epoch}" \
  --data_path "${data_path}"