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
import os
import torch
from torch.utils.data import DataLoader

from data.dataset import build_file_list, make_splits, BUSIDataset, get_transforms
from models.lsec_net import LSECNet
from losses.losses import LSECLoss
from metrics.metrics import evaluate_model, aggregate_results, print_result_table
from trainer import train_and_evaluate, run_fold, compute_class_weights


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
    print("\n" + "="*50)
    print(f"  MODE: TRAIN  |  folds={args.folds}  |  epochs={args.epochs}")
    print("="*50)

    file_list = build_file_list(args.data_root)
    print(f"Total samples : {len(file_list)}")

    train_val, test_set, folds = make_splits(file_list)

    # ── Variant A: Baseline (L_cls only) ────────────────────────
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
    )

    # ── Variant B: LSEC-Net (L_cls + L_align + L_out) ───────────
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
    )

    # ── Final comparison (only meaningful with multiple folds) ───
    if args.folds > 1:
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
        res = evaluate_model(model_name, model, test_loader, device, save_cm=True)
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
    p.add_argument('--folds',      type=int, default=1,
                   help='Folds to run: 1 for early signal, 5 for full CV (train mode)')
    p.add_argument('--epochs',     type=int, default=40,
                   help='Max epochs per fold')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--checkpoint', nargs='+', default=None,
                   help='Checkpoint .pth path(s) for evaluate mode')
    return p.parse_args()


def main():
    args   = parse_args()
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
