batch_size=512


export PYTHONPATH=.
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
    linear_probing/detok.py \
    --model detok_BB \
    --pretrained_model_name_or_path "" \
    --token_channels 16 \
    --last_layer_feature \
    --checkpoint_path "work_dirs/tokenizer_training/detokBB-ch16-g3.0-m0.00.7random-auxdinov2transformerreconstruction/checkpoints/latest.pth" \
    --batch_size $batch_size \
    --epochs 1 \
    --print_freq 50 \
    --eval_freq 1 \
    --data_path ./data/imagenet/train \
    --output_dir ./work_dirs/linear_prob