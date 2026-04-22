"""
LSEC-Net entry point.

Usage:
  python main.py --mode debug    --data_root ./archive/Dataset_BUSI_with_GT
  python main.py --mode train    --data_root ./archive/Dataset_BUSI_with_GT --folds 1
  python main.py --mode train    --data_root ./archive/Dataset_BUSI_with_GT --folds 5
  python main.py --mode evaluate --data_root ./archive/Dataset_BUSI_with_GT \\
                 --checkpoint proposed_fold0.pth proposed_fold1.pth ...
"""

import argparse
from datetime import datetime
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import (
    build_file_list,
    download_busi_dataset,
    make_splits,
    BUSIDataset,
    get_transforms,
)
from models.lsec_net import LSECNet
from losses.losses import LSECLoss
from metrics.metrics import evaluate_model, aggregate_results, print_result_table
from trainer import train_and_evaluate, run_fold, compute_class_weights


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Mode: debug
# ─────────────────────────────────────────────

def mode_debug(args, device):
    print("\n" + "="*50)
    print("  MODE: DEBUG  (fold 0, 3 epochs, verbose per-batch)")
    print("="*50)

    file_list = build_file_list(args.data_root)
    print(f"Total samples : {len(file_list)}")

    train_val, test_set, folds = make_splits(file_list)

    train_data = [train_val[i] for i in folds[0][0]]
    val_data   = [train_val[i] for i in folds[0][1]]

    train_loader = DataLoader(
        BUSIDataset(train_data, get_transforms('train')),
        batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        BUSIDataset(val_data, get_transforms('val')),
        batch_size=8, shuffle=False, num_workers=2)

    train_labels  = [x[2] for x in train_data]
    class_weights = compute_class_weights(train_labels, 3).to(device)

    model     = LSECNet(num_classes=3).to(device)
    criterion = LSECLoss(lambda1=1.0, lambda2=0.3, class_weights=class_weights)

    # Sanity check: shapes + NaN
    print("\n  [Sanity] Forward pass …")
    batch = next(iter(train_loader))
    with torch.no_grad():
        logits, feat = model(batch['image'].to(device))
        cam          = model.get_cam(feat, batch['label'].to(device))
    print(f"  logits: {tuple(logits.shape)} | feat: {tuple(feat.shape)} | CAM: {tuple(cam.shape)}")
    print(f"  CAM range [{cam.min():.4f}, {cam.max():.4f}] | NaN: {cam.isnan().any().item()}")

    print("\n  [Debug] Training 3 epochs …")
    run_fold(model, criterion, train_loader, val_loader,
             epochs=3, device=device, fold_idx=0, debug=True)

    print("\n  Pipeline OK.")


# ─────────────────────────────────────────────
# Mode: train
# ─────────────────────────────────────────────

def mode_train(args, device):
    run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or (
        f'{run_stamp}_train_{args.variant}_folds{args.folds}_ep{args.epochs}'
    )
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)
    args.output_dir = run_dir

    print("\n" + "="*50)
    print(f"  MODE: TRAIN  |  variant={args.variant}  |  folds={args.folds}  |  epochs={args.epochs}")
    print(f"  RUN DIR: {run_dir}")
    print("="*50)

    with open(os.path.join(run_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, default=str)

    file_list = build_file_list(args.data_root)
    print(f"Total samples : {len(file_list)}")

    train_val, test_set, folds = make_splits(file_list)

    run_a = args.variant in ('A', 'both')
    run_b = args.variant in ('B', 'both')

    baseline_results = proposed_results = None

    # ── Variant A: Baseline (L_cls only) ────────────────────────
    if run_a:
        print("\n" + "★"*50)
        print("  VARIANT A — GradCAM Baseline  (L_cls only)")
        print("★"*50)
        baseline_results = train_and_evaluate(
            train_val, test_set, folds,
            use_mask_loss=False,
            n_folds_to_run=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            variant_name='baseline',
            tta=args.tta,
            output_dir=run_dir,
            aug=args.aug,
            label_smoothing=args.label_smoothing,
            mixup_prob=0.0 if args.no_mixup else args.mixup_prob,
            mixup_alpha=args.mixup_alpha,
            class_weight_mode=args.class_weight_mode,
            dropout=args.dropout,
            warmup_epochs=args.warmup_epochs,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            warmup_lr=args.warmup_lr,
            calibrate_logits=args.calibrate_logits,
            sampler=args.sampler,
            mask_lambda1=args.mask_lambda1,
            mask_lambda2=args.mask_lambda2,
            seed=args.seed,
        )

    # ── Variant B: LSEC-Net (L_cls + L_align + L_out) ───────────
    if run_b:
        print("\n" + "★"*50)
        print("  VARIANT B — LSEC-Net Proposed  (L_cls + L_align + L_out)")
        print("★"*50)
        proposed_results = train_and_evaluate(
            train_val, test_set, folds,
            use_mask_loss=True,
            n_folds_to_run=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            variant_name='proposed',
            tta=args.tta,
            output_dir=run_dir,
            aug=args.aug,
            label_smoothing=args.label_smoothing,
            mixup_prob=0.0 if args.no_mixup else args.mixup_prob,
            mixup_alpha=args.mixup_alpha,
            class_weight_mode=args.class_weight_mode,
            dropout=args.dropout,
            warmup_epochs=args.warmup_epochs,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            warmup_lr=args.warmup_lr,
            calibrate_logits=args.calibrate_logits,
            sampler=args.sampler,
            mask_lambda1=args.mask_lambda1,
            mask_lambda2=args.mask_lambda2,
            seed=args.seed,
        )

    # ── Final comparison (only meaningful when both ran + multiple folds) ──
    if run_a and run_b and args.folds > 1:
        print("\n" + "="*68)
        print("  FINAL COMPARISON")
        print("="*68)
        print_result_table(baseline_results, proposed_results)


# ─────────────────────────────────────────────
# Mode: evaluate
# ─────────────────────────────────────────────

def mode_evaluate(args, device):
    print("\n" + "="*50)
    print("  MODE: EVALUATE")
    print("="*50)

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluate mode")

    file_list = build_file_list(args.data_root)
    _, test_set, _ = make_splits(file_list)

    test_loader = DataLoader(
        BUSIDataset(test_set, get_transforms('val')),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    fold_results = []

    for ckpt_path in args.checkpoint:
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Not found: {ckpt_path}")
            continue

        model = LSECNet(num_classes=3).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n  Checkpoint : {ckpt_path}")
        output_dir = os.path.dirname(ckpt_path) or None
        res = evaluate_model(
            model_name, model, test_loader, device,
            save_cm=True, tta=args.tta, output_dir=output_dir)
        fold_results.append(res)

    if len(fold_results) > 1:
        print(f"\n{'='*50}")
        print(f"  Aggregated across {len(fold_results)} checkpoints")
        print(f"{'='*50}")
        aggregate_results(fold_results)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='LSEC-Net — train / debug / evaluate')
    p.add_argument('--mode',       required=True,
                   choices=['debug', 'train', 'evaluate'])
    p.add_argument('--data_root',  default='/workspace/Dataset_BUSI_with_GT',
                   help='Path to Dataset_BUSI_with_GT directory')
    p.add_argument('--download_dataset', action='store_true',
                   help='Download BUSI from KaggleHub into --download_dir before running')
    p.add_argument('--download_dir', default='/workspace',
                   help='Directory where KaggleHub should download BUSI')
    p.add_argument('--force_download', action='store_true',
                   help='Force KaggleHub to download BUSI again')
    p.add_argument('--variant',    choices=['A', 'B', 'both'], default='both',
                   help='Variant to train: A=baseline (L_cls only), B=proposed (L_cls+L_align+L_out), both=run both')
    p.add_argument('--tta',        action='store_true',
                   help='Enable Test-Time Augmentation during final evaluation')
    p.add_argument('--folds',      type=int, default=1,
                   help='Folds to run: 1 for early signal, 5 for full CV (train mode)')
    p.add_argument('--epochs',     type=int, default=40,
                   help='Max epochs per fold')
    p.add_argument('--batch_size', type=int, default=16,
                   help='Batch size (default 16; BUSI ~400 train samples → stable gradient)')
    p.add_argument('--runs_dir', default='runs',
                   help='Parent directory for datetime-named train outputs')
    p.add_argument('--run_name', default=None,
                   help='Optional explicit run folder name inside --runs_dir')
    p.add_argument('--aug', choices=['default', 'light', 'none'], default='default',
                   help='Training augmentation strength')
    p.add_argument('--label_smoothing', type=float, default=0.1,
                   help='CrossEntropy label smoothing')
    p.add_argument('--no_mixup', action='store_true',
                   help='Disable mixup for baseline ablations')
    p.add_argument('--mixup_prob', type=float, default=0.5,
                   help='Mixup probability when mixup is enabled')
    p.add_argument('--mixup_alpha', type=float, default=0.2,
                   help='Beta distribution alpha for mixup')
    p.add_argument('--class_weight_mode', choices=['inverse', 'none'], default='inverse',
                   help='Class weighting strategy for CrossEntropy')
    p.add_argument('--dropout', type=float, default=0.3,
                   help='Classifier dropout probability')
    p.add_argument('--warmup_epochs', type=int, default=5,
                   help='Frozen-backbone warmup epochs')
    p.add_argument('--warmup_lr', type=float, default=1e-3,
                   help='Head learning rate during frozen-backbone warmup')
    p.add_argument('--head_lr', type=float, default=1e-4,
                   help='Head learning rate after unfreezing')
    p.add_argument('--backbone_lr', type=float, default=5e-5,
                   help='Backbone learning rate after unfreezing')
    p.add_argument('--calibrate_logits', action='store_true',
                   help='Tune class logit offsets on validation fold before test evaluation')
    p.add_argument('--sampler', choices=['shuffle', 'balanced'], default='shuffle',
                   help='Training sampler: regular shuffle or class-balanced sampling')
    p.add_argument('--mask_lambda1', type=float, default=1.0,
                   help='CAM/mask Dice alignment loss weight for Variant B')
    p.add_argument('--mask_lambda2', type=float, default=0.3,
                   help='CAM outside-mask loss weight for Variant B')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for splits, augmentation, dataloader workers, and torch')
    p.add_argument('--checkpoint', nargs='+', default=None,
                   help='Checkpoint .pth path(s) for evaluate mode')
    return p.parse_args()


def main():
    args   = parse_args()
    set_seed(args.seed)
    if args.download_dataset:
        args.data_root = download_busi_dataset(
            download_dir=args.download_dir,
            force_download=args.force_download,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    if args.mode == 'debug':
        mode_debug(args, device)
    elif args.mode == 'train':
        mode_train(args, device)
    elif args.mode == 'evaluate':
        mode_evaluate(args, device)


if __name__ == '__main__':
    main()
