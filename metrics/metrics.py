import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error,
    roc_auc_score, confusion_matrix,
)

from data.dataset import CLASS_NAMES, XAI_ACC_THRESHOLD

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# XAI Metrics
# ─────────────────────────────────────────────

def pointing_game(cam, mask):
    """Max activation pixel falls inside GT mask → hit=1, miss=0."""
    B         = cam.shape[0]
    cam_flat  = cam.view(B, -1)
    mask_flat = mask.view(B, -1)
    max_idx   = cam_flat.argmax(dim=1)
    hit       = mask_flat[torch.arange(B), max_idx]
    return hit.float().mean().item()


def soft_iou(cam, mask):
    """Soft intersection-over-union between CAM and mask."""
    i = (cam * mask).sum(dim=[1, 2, 3])
    u = (cam + mask - cam * mask).sum(dim=[1, 2, 3])
    return (i / (u + 1e-8)).mean().item()


def inside_ratio(cam, mask):
    """Fraction of total CAM energy that falls inside the mask."""
    inside = (cam * mask).sum(dim=[1, 2, 3])
    total  = cam.sum(dim=[1, 2, 3])
    return (inside / (total + 1e-8)).mean().item()


def compute_xai_metrics(cam, mask):
    cam, mask = cam.detach().cpu(), mask.cpu()
    return {
        'pointing_game': pointing_game(cam, mask),
        'soft_iou':      soft_iou(cam, mask),
        'inside_ratio':  inside_ratio(cam, mask),
    }


# ─────────────────────────────────────────────
# Classification + XAI combined evaluation
# ─────────────────────────────────────────────

_TTA_AUGS = [
    lambda x: x,
    TF.hflip,
    TF.vflip,
]


def evaluate_model(model_name, model, loader, device, save_cm=True, tta=False):
    """
    Full evaluation pass:
      - Accuracy, Precision, Recall, F1 (weighted + macro), MAE, RMSE, AUC
      - Pointing Game, Soft IoU, Inside Ratio (gated: accuracy >= XAI_ACC_THRESHOLD)
      - Confusion matrix saved as PNG if save_cm=True
      - tta=True: averages softmax over hflip / vflip / original (XAI uses original feat)

    Returns dict of all metrics.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    xai_cams, xai_masks = [], []

    with torch.no_grad():
        for batch in loader:
            img    = batch['image'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

            if tta:
                tta_probs = []
                for aug in _TTA_AUGS:
                    logits, _ = model(aug(img))
                    tta_probs.append(torch.softmax(logits, 1))
                probs = torch.stack(tta_probs).mean(0)
                preds = probs.argmax(1)
                # XAI uses original-image features for interpretability consistency
                _, feat = model(img)
            else:
                logits, feat = model(img)
                preds  = logits.argmax(1)
                probs  = torch.softmax(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            non_normal = (labels != 0)
            if non_normal.any():
                cam = model.get_cam(feat[non_normal], labels[non_normal])
                xai_cams.append(cam.cpu())
                xai_masks.append(mask[non_normal].cpu())

    y_true      = np.array(all_labels)
    y_pred      = np.array(all_preds)
    y_pred_prob = np.array(all_probs)

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w      = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    mae       = mean_absolute_error(y_true, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_true, y_pred))

    try:
        auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
    except ValueError:
        auc = float('nan')

    print(f"\n  Model     : {model_name}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}  (weighted)")
    print(f"  Recall    : {recall:.4f}  (weighted)")
    print(f"  F1        : {f1_w:.4f}  (weighted)  |  F1 macro: {f1_macro:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  RMSE      : {rmse:.4f}")
    print(f"  AUC       : {auc:.4f}  (OvR)")

    if accuracy >= XAI_ACC_THRESHOLD and xai_cams:
        xai = compute_xai_metrics(torch.cat(xai_cams), torch.cat(xai_masks))
        print(f"  Pointing Game : {xai['pointing_game']:.4f}")
        print(f"  Soft IoU      : {xai['soft_iou']:.4f}")
        print(f"  Inside Ratio  : {xai['inside_ratio']:.4f}")
    else:
        print(f"  [XAI SKIP] accuracy {accuracy:.4f} < {XAI_ACC_THRESHOLD}")
        xai = {'pointing_game': None, 'soft_iou': None, 'inside_ratio': None}

    if save_cm:
        _save_confusion_matrix(model_name, y_true, y_pred)

    return {
        'accuracy':    accuracy,
        'precision':   precision,
        'recall':      recall,
        'f1_weighted': f1_w,
        'f1_macro':    f1_macro,
        'mae':         mae,
        'rmse':        rmse,
        'auc':         auc,
        **xai,
    }


def _save_confusion_matrix(model_name, y_true, y_pred):
    cm_arr = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    fname = f'Confusion_{model_name.replace(" ", "_")}.png'
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  CM saved  : {fname}")


# ─────────────────────────────────────────────
# Aggregation (across folds)
# ─────────────────────────────────────────────

XAI_KEYS = ('pointing_game', 'soft_iou', 'inside_ratio')


def aggregate_results(fold_results):
    """Compute mean ± std across folds. XAI keys skip None values."""
    keys = [k for k in fold_results[0].keys() if k != 'confusion_matrix']
    agg  = {}

    print(f"\n  {'Metric':<22} {'Mean':>10} {'Std':>10}  Folds")
    print('  ' + '-' * 52)
    for k in keys:
        vals = [r[k] for r in fold_results if r[k] is not None]
        if not vals:
            print(f"  {k:<22}: N/A (no fold passed XAI threshold)")
            agg[k] = {'mean': None, 'std': None}
            continue
        suffix = f'  [{len(vals)}/5]' if k in XAI_KEYS else ''
        print(f"  {k:<22}: {np.mean(vals):.4f}  ±{np.std(vals):.4f}{suffix}")
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return agg


def print_result_table(baseline, proposed):
    """Side-by-side comparison table for baseline vs proposed."""
    metrics = ['accuracy', 'f1_macro', 'auc',
               'pointing_game', 'soft_iou', 'inside_ratio']

    print(f"\n  {'Metric':<22} {'Baseline':>22} {'LSEC-Net (ours)':>22}")
    print('  ' + '-' * 68)
    for m in metrics:
        if m not in baseline or m not in proposed:
            continue
        b = baseline[m]
        p = proposed[m]
        b_str = f"{b['mean']:.4f} ± {b['std']:.4f}" if b['mean'] is not None else 'N/A'
        p_str = f"{p['mean']:.4f} ± {p['std']:.4f}" if p['mean'] is not None else 'N/A'
        print(f"  {m:<22} {b_str:>22} {p_str:>22}")
