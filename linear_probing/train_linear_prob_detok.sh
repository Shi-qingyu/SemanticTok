batch_size=1024


export PYTHONPATH=.
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
    linear_probing/linear_prob_detok.py \
    --model detok_BB \
    --token_channels 16 \
    --checkpoint_path "released_model/detok-BB-gamm3.0-m0.7-decoder_tuned.pth" \
    --batch_size $batch_size \
    --epochs 50 \
    --print_freq 50 \
    --eval_freq 5 \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/linear_prob \