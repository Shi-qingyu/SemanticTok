import argparse
import sys
import logging
import os
import time
import tqdm

import torch
from torch import nn
import torchvision.transforms as transforms
from transformers import Dinov2Model, Dinov2Config, AutoImageProcessor
from PIL import Image

from utils.loader import ListDataset, center_crop_arr
from utils.logger import MetricLogger, SmoothedValue
import utils.distributed as dist
import utils.misc as misc

logger = logging.getLogger("DINOv2 Linear Probing")


class DINOv2LinearProbing(nn.Module):
    def __init__(self, model_name: str, num_classes: int, use_cls_token: bool = True, freeze_backbone: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.freeze_backbone = freeze_backbone
        
        # Map model names to transformers model identifiers
        model_mapping = {
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
            'dinov2_vitg14': 'facebook/dinov2-giant',
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Unsupported model: {model_name}. Choose from {list(model_mapping.keys())}")
        
        # Load DINOv2 model using transformers
        model_id = model_mapping[model_name]
        try:
            self.backbone = Dinov2Model.from_pretrained(model_id)
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            logger.info(f"Successfully loaded {model_id} from transformers")
        except Exception as e:
            logger.warning(f"Failed to load {model_id} from transformers: {e}")
            logger.info("Falling back to torch.hub loading...")
            # Fallback to torch.hub
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
            self.image_processor = None
            logger.info(f"Successfully loaded {model_name} from torch.hub")
        
        # Get feature dimension from config
        if hasattr(self.backbone, 'config'):
            feature_dim = self.backbone.config.hidden_size
        else:
            # Fallback for torch.hub models
            if hasattr(self.backbone, 'embed_dim'):
                feature_dim = self.backbone.embed_dim
            elif hasattr(self.backbone, 'num_features'):
                feature_dim = self.backbone.num_features
            else:
                # Default dimensions for different DINOv2 models
                model_dims = {
                    'dinov2_vits14': 384,
                    'dinov2_vitb14': 768,
                    'dinov2_vitl14': 1024,
                    'dinov2_vitg14': 1536,
                }
                feature_dim = model_dims.get(model_name, 768)
        
        logger.info(f"Using DINOv2 model: {model_name} ({model_id}) with feature dimension: {feature_dim}")
        logger.info(f"Classification method: {'CLS token' if use_cls_token else 'Pooled patch tokens'}")
        
        # Create classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            logger.info("Backbone frozen, only training classification head")
        else:
            logger.info("Fine-tuning entire model")
            
        # Initialize classifier
        nn.init.trunc_normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_backbone):
            # Check if we're using transformers or torch.hub model
            if self.image_processor is not None:
                # Using transformers model
                outputs = self.backbone(pixel_values=x)
                
                if self.use_cls_token:
                    # Use CLS token for classification
                    features = outputs.last_hidden_state[:, 0]  # [B, hidden_size]
                else:
                    # Use patch tokens with global average pooling
                    patch_tokens = outputs.last_hidden_state[:, 1:]  # [B, num_patches, hidden_size]
                    features = patch_tokens.mean(dim=1)  # [B, hidden_size]
            else:
                # Using torch.hub model (fallback)
                if self.use_cls_token:
                    # Use CLS token for classification
                    features = self.backbone(x)  # [B, feature_dim]
                else:
                    # Use patch tokens with global average pooling
                    # Get all tokens (including CLS token)
                    features = self.backbone.forward_features(x)  # [B, num_patches + 1, feature_dim]
                    # Remove CLS token and pool patch tokens
                    patch_tokens = features[:, 1:]  # [B, num_patches, feature_dim]
                    features = patch_tokens.mean(dim=1)  # [B, feature_dim]
        
        # Classification
        logits = self.classifier(features)
        return logits


def setup(args: argparse.Namespace):
    """setup distributed training, logging, and experiment configuration"""
    dist.enable_distributed()
    
    if args.exp_name is None:
        method = "cls" if args.use_cls_token else "pool"
        freeze_str = "frozen" if args.freeze_backbone else "finetune"
        args.exp_name = f"dinov2_{args.model_name}_{method}_{freeze_str}_{time.strftime('%Y%m%d_%H%M')}"

    base_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base_dir
    
    global_rank, world_size = dist.get_global_rank(), dist.get_world_size()
    args.world_size = world_size
    args.global_bsz = args.batch_size * world_size
    
    misc.fix_random_seeds(args.seed + global_rank)
    
    if global_rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(base_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    logger.info(f"Distributed setup: rank {global_rank}/{world_size}")
    return global_rank == 0


def create_train_dataloader(args, image_processor=None):
    """Create training dataloader"""
    if image_processor is not None:
        # Use transformers image processor
        def transform_fn(pil_image):
            # Apply basic augmentations first
            if torch.rand(1) < 0.5:  # Random horizontal flip
                pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Use the image processor
            processed = image_processor(pil_image, return_tensors="pt")
            return processed["pixel_values"].squeeze(0)
        
        transform_train = transform_fn
    else:
        # Fallback to torchvision transforms
        transform_train = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])
    
    dataset_train = ListDataset(
        args.data_path,
        data_list="data/train.txt",
        transform=transform_train,
        loader_name="img_loader",
        return_label=True,
        should_flip=False,
    )
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=True,
    )

    logger.info(f"Train dataset size: {len(dataset_train)}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    return data_loader_train


def create_val_dataloader(args, image_processor=None):
    """Create validation dataloader"""
    if image_processor is not None:
        # Use transformers image processor
        def transform_fn(pil_image):
            processed = image_processor(pil_image, return_tensors="pt")
            return processed["pixel_values"].squeeze(0)
        
        transform_val = transform_fn
    else:
        # Fallback to torchvision transforms
        transform_val = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ])
    
    dataset_val = ListDataset(
        args.data_path.replace("train", "val"),
        data_list="data/val.txt",
        transform=transform_val,
        loader_name="img_loader",
        return_label=True,
        should_flip=False,
    )
    
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=False,
    )

    logger.info(f"Val dataset size: {len(dataset_val)}")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_val


def evaluate(model, data_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, disable=dist.get_global_rank() != 0):
            img = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            # Validate and clamp labels
            labels = labels.long()
            labels = torch.clamp(labels, 0, model.num_classes - 1)
            
            logits = model(img)
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
    
    # Aggregate across all processes
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(total_correct_tensor)
        torch.distributed.all_reduce(total_samples_tensor)
    
    accuracy = total_correct_tensor.item() / total_samples_tensor.item()
    return accuracy


def get_lr_scheduler(optimizer, args, num_training_steps):
    """Create learning rate scheduler"""
    if args.lr_sched == "cosine":
        warmup_steps = int(args.warmup_rate * num_training_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    else:
        return None


def main(args: argparse.Namespace) -> int:
    is_main_process = setup(args)
    
    # Create model
    device = torch.device(f"cuda:{dist.get_global_rank()}")
    model = DINOv2LinearProbing(
        model_name=args.model_name,
        num_classes=args.num_classes,
        use_cls_token=args.use_cls_token,
        freeze_backbone=args.freeze_backbone
    )
    
    # Create dataloaders with image processor
    data_loader_train = create_train_dataloader(args, model.image_processor)
    data_loader_val = create_val_dataloader(args, model.image_processor)
    
    # Move model to device
    model = model.to(device)
    
    # Wrap model with DistributedDataParallel
    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[dist.get_global_rank()],
            find_unused_parameters=False
        )

    # Get trainable parameters
    if args.freeze_backbone:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Training {len(trainable_params)} parameters (classifier only)")
    else:
        trainable_params = model.parameters()
        logger.info("Training all parameters (full fine-tuning)")
    
    # Setup optimizer
    if args.lr is None:
        args.lr = args.blr * args.global_bsz / 256
    
    if args.freeze_backbone:
        # Use higher learning rate for linear probing
        optimizer = torch.optim.SGD(
            trainable_params, 
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        # Use lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.lr * 0.1,  # Lower LR for fine-tuning
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay
        )

    # Setup learning rate scheduler
    num_training_steps = len(data_loader_train) * args.epochs
    scheduler = get_lr_scheduler(optimizer, args, num_training_steps)

    loss_fn = nn.CrossEntropyLoss()
    
    # Setup metric logger
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=50, fmt='{value:.4f}'))

    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
            
        model.train()

        for step, batch in enumerate(data_loader_train):
            img = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            # Validate labels
            labels = labels.long()
            
            # Debug: Check for invalid labels
            if step == 0 and epoch == 0:
                logger.info(f"Label range: min={labels.min().item()}, max={labels.max().item()}")
                logger.info(f"Number of classes: {args.num_classes}")
                invalid_labels = (labels < 0) | (labels >= args.num_classes)
                if invalid_labels.any():
                    logger.warning(f"Found {invalid_labels.sum().item()} invalid labels out of {labels.size(0)}")
                    logger.warning(f"Invalid label values: {labels[invalid_labels].unique()}")

            logits = model(img)
            loss = loss_fn(logits, labels)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Aggregate metrics across processes
            loss_reduced = dist.all_reduce_mean(loss)
            acc_reduced = dist.all_reduce_mean(acc)
            
            metric_logger.update(
                loss=loss_reduced,
                acc=acc_reduced,
                lr=optimizer.param_groups[0]["lr"]
            )
            
            if is_main_process and step % args.print_freq == 0:
                logger.info(
                    f"Epoch: [{epoch}/{args.epochs}] Step: [{step}/{len(data_loader_train)}] "
                    f"Loss: {loss_reduced:.4f} Acc: {acc_reduced:.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        # Evaluation
        if epoch % args.eval_freq == 0:
            val_acc = evaluate(model, data_loader_val, device)
            if is_main_process:
                logger.info(f"Epoch {epoch} - Validation accuracy: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_acc:
                    best_acc = val_acc
                    checkpoint = {
                        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'accuracy': val_acc,
                        'args': args,
                    }
                    torch.save(checkpoint, os.path.join(args.log_dir, 'best_model.pth'))
                    logger.info(f"New best model saved with accuracy: {best_acc:.4f}")
    
    if is_main_process:
        logger.info(f"Training completed. Best validation accuracy: {best_acc:.4f}")
    
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("DINOv2 Linear Probing", add_help=False)

    # distributed training parameters
    parser.add_argument("--local-rank", "--local_rank", default=0, type=int, help="Local rank for distributed training")

    # basic training parameters
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU for training")

    # model parameters
    parser.add_argument("--model_name", default="dinov2_vitb14", type=str, 
                       choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                       help="DINOv2 model variant")
    parser.add_argument("--use_cls_token", action="store_true", 
                       help="Use CLS token for classification (default: use pooled patch tokens)")
    parser.add_argument("--freeze_backbone", action="store_true", 
                       help="Freeze backbone and only train classifier (linear probing)")
    parser.add_argument("--img_size", default=224, type=int)

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs/dinov2_linear_prob")
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=5)

    # evaluation parameters
    parser.add_argument("--eval_bsz", type=int, default=256)

    # optimization parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-1, help="Base learning rate")
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for AdamW optimizer")

    # dataset parameters
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # system parameters
    parser.add_argument("--seed", default=0, type=int)

    # wandb parameters
    parser.add_argument("--project", default="dinov2_linear_prob", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)
