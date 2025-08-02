"""
DeTok: Reconstruction model training script.
"""

import argparse
import datetime
import logging
import sys
import os
import time
import json

import torch
import torch.distributed

import models

import utils.distributed as dist
import utils.misc as misc
from utils.logger import setup_logging
from utils.builders import (
    create_reconstruction_model,
    create_val_dataloader,
)
from utils.train_utils import evaluate_tokenizer

# performance optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("DeTok")


def setup(args: argparse.Namespace):
    """setup distributed training, logging, and experiment configuration"""
    dist.enable_distributed()
    global logger

    if args.exp_name is None:
        args.exp_name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_exp"

    base_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base_dir
    args.eval_dir = os.path.join(base_dir, "eval")

    global_rank, world_size = dist.get_global_rank(), dist.get_world_size()
    args.world_size = world_size
    args.global_bsz = args.batch_size * world_size

    misc.fix_random_seeds(args.seed + global_rank)

    wandb_logger = None
    if global_rank == 0:
        for path in [args.log_dir, args.eval_dir]:
            os.makedirs(path, exist_ok=True)

        setup_logging(output=args.log_dir, name="DeTok", rank0_log_only=True)
        logger.info(f"Logging to {args.log_dir}")
        json_config = json.dumps(args.__dict__, indent=4, sort_keys=True)
        logger.info(json_config)

        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        json_path = os.path.join(args.log_dir, f"args_{time_str}.json")
        with open(json_path, "w") as f:
            json.dump(args.__dict__, f, indent=4)
        logger.info(f"Args saved to {json_path}")

    return wandb_logger


def create_reconstruction_model(args):
    logger.info("Creating reconstruction models.")
    if args.model in models.VAE_models:
        model = models.VAE_models[args.model](
            load_ckpt=not getattr(args, "no_load_ckpt", False),
            gamma=args.gamma,
        )
    elif args.model in models.DeTok_models:
        model = models.DeTok_models[args.model](
            img_size=args.img_size,
            patch_size=args.patch_size,
            token_channels=args.token_channels,
            mask_ratio=args.mask_ratio,
            gamma=args.gamma,
        )
    else:
        raise ValueError(f"Unsupported model {args.model}")

    model.cuda()
    logger.info("====Model=====")
    logger.info(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    return model


def main(args: argparse.Namespace) -> int:
    global logger
    wandb_logger = setup(args)

    # initialize data loaders for evaluation only
    data_loader_val = create_val_dataloader(args)

    # initialize model
    model = create_reconstruction_model(args)

    # load from checkpoint before wrapping with DDP
    if args.load_from:
        logger.info(f"Loading checkpoint from {args.load_from}")
        checkpoint = torch.load(args.load_from, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning("No checkpoint provided. Using randomly initialized model.")

    # setup distributed evaluation
    if dist.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[dist.get_global_rank()],
            output_device=dist.get_global_rank(),
            find_unused_parameters=False
        )
        logger.info(f"Model wrapped with DDP on rank {dist.get_global_rank()}")

    # get models without DDP wrapper for evaluation
    model_wo_ddp = model.module if hasattr(model, "module") else model

    # evaluation (only main model, no EMA)
    logger.info("Starting evaluation...")
    torch.cuda.empty_cache()
    
    logger.info("Evaluating main model...")
    evaluate_tokenizer(
        args, model_wo_ddp, None, data_loader_val, 0, wandb_logger, use_ema=False
    )

    logger.info("Evaluation completed.")
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Reconstruction model evaluation", add_help=False)

    # model parameters
    parser.add_argument("--model", default="detok_BB", type=str)
    parser.add_argument("--token_channels", default=16, type=int)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument("--mask_ratio", default=0.0, type=float)
    parser.add_argument("--gamma", default=0.0, type=float, help="noise standard deviation for training")
    parser.add_argument("--use_additive_noise", action="store_true")

    parser.add_argument("--vis_only", action="store_true")

    # checkpoint parameters
    parser.add_argument("--load_from", type=str, required=True, help="load from pretrained model checkpoint")

    # evaluation parameters
    parser.add_argument("--num_images", default=50000, type=int, help="Number of images to evaluate on")
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/val_fid_statistics_file.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--eval_bsz", type=int, default=256)

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs/eval")
    
    # dataset parameters
    parser.add_argument("--data_path", default="./data/imagenet/val", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # system parameters
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")

    # wandb parameters
    parser.add_argument("--project", default="lDeTok_eval", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
