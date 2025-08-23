## Preparation


### Installation

Create and activate conda environment:
```bash
conda create -n detok python=3.10 -y && conda activate detok
pip install -r requirements.txt
```

### Dataset
Create `data/` folder by `mkdir data/` then download [ImageNet](http://image-net.org/download) dataset. You can either:
1. Download it directly to `data/imagenet/` 
2. Create a symbolic link: `ln -s /path/to/your/imagenet data/imagenet`


### Download Required Files

Create data directory and download required files:
```bash
mkdir data/
# Download everything from huggingface
huggingface-cli download jjiaweiyang/l-DeTok --local-dir released_model
mv released_model/train.txt data/
mv released_model/val.txt data/
mv released_model/fid_stats data/
mv released_model/imagenet-val-prc.zip ./data/

# Unzip: imagenet-val-prc.zip for precision & recall evaluation
python -m zipfile -e ./data/imagenet-val-prc.zip ./data/
```

### Data Organization

Your data directory should be organized as follows:
```
data/
├── fid_stats/          # FID statistics
│   ├── adm_in256_stats.npz
│   └── val_fid_statistics_file.npz
├── imagenet/           # ImageNet dataset (or symlink)
│   ├── train/
│   └── val/
├── imagenet-val-prc/   # Precision-recall data
├── train.txt           # Training file list
└── val.txt             # Validation file list
```

## Training

### 1. Tokenizer Training

Train DeTok tokenizer with denoising:
```bash
project=tokenizer_training
exp_name=detokBB-g3.0-m0.7-200ep
batch_size=32  # global batch size = batch_size x num_nodes x 8 = 1024
num_nodes=4    # adjust for your multi-node setup

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_reconstruction.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --model detok_BB \
    --gamma 3.0 --mask_ratio 0.7 --random_mask_ratio \
    --use_aux_decoder --aux_loss_weight 1.0 \
    --online_eval \
    --epochs 200 --discriminator_start_epoch 100 \
    --data_path ./data/imagenet/train
```

### 2. Generative Model Training

Train SiT-base (100 epochs):
```bash
tokenizer_project=tokenizer_training
tokenizer_exp_name=detokBB-g3.0-m0.7-200ep-decoder_ft-100ep
project=gen_model_training
exp_name=sit_base-${tokenizer_exp_name}
batch_size=32
num_nodes=4
epochs=100

torchrun --nproc_per_node=8 --nnodes=$num_nodes --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    main_diffusion.py \
    --project $project --exp_name $exp_name --auto_resume \
    --batch_size $batch_size --epochs $epochs --use_aligned_schedule \
    --tokenizer detok_BB --use_ema_tokenizer --collect_tokenizer_stats \
    --stats_key $tokenizer_exp_name --stats_cache_path work_dirs/stats.pkl \
    --load_tokenizer_from work_dirs/$tokenizer_project/$tokenizer_exp_name/checkpoints/latest.pth \
    --model SiT_base \
    --num_sampling_steps 250 --cfg 1.6 \
    --cfg_list 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 \
    --vis_freq 50 --eval_bsz 256 \
    --data_path ./data/imagenet/train
```