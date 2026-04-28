"""
evaluate_busbra.py — Cross-dataset XAI evaluation on BUS-BRA

Loads a trained LSEC-Net checkpoint (trained on BUSI) and evaluates
XAI localization quality (Pointing Game, Soft IoU, Inside Ratio)
on the BUS-BRA dataset without any retraining.

Usage:
  # With data already downloaded:
  python evaluate_busbra.py --checkpoint runs/fold0_best.pth --data_root ./archive/BUSBRA/BUSBRA

  # Auto-download via KaggleHub:
  python evaluate_busbra.py --checkpoint runs/fold0_best.pth --download

  # Multiple checkpoints (reports mean ± std):
  python evaluate_busbra.py --checkpoint runs/fold0.pth runs/fold1.pth --data_root ./archive/BUSBRA/BUSBRA

  # Filter by pathology:
  python evaluate_busbra.py --checkpoint runs/fold0.pth --pathology benign
  python evaluate_busbra.py --checkpoint runs/fold0.pth --pathology malignant
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from models.lsec_net import LSECNet
from metrics.metrics import compute_xai_metrics, aggregate_results

BUSBRA_KAGGLE_HANDLE = 'orvile/bus-bra-a-breast-ultrasound-dataset'
BUSBRA_CSV_FILENAME  = 'bus_data.csv'

# BUS-BRA only has benign/malignant — map to BUSI class indices
LABEL_MAP = {'benign': 1, 'malignant': 2}


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class BUSBRADataset(Dataset):
    """
    Reads BUS-BRA from:
      <root>/Images/bus_XXXX-{l/r/s}.png
      <root>/Masks/mask_XXXX-{l/r/s}.png
      <root>/bus_data.csv  (columns: ID, Pathology, ...)

    Returns the same dict format as BUSIDataset:
      {'image': [3,224,224], 'mask': [1,224,224], 'label': long}

    Labels follow BUSI CLASS_MAP: benign=1, malignant=2.
    """

    def __init__(self, root, pathology=None):
        """
        Args:
            root:       path to BUSBRA root (contains Images/, Masks/, bus_data.csv)
            pathology:  None (all) | 'benign' | 'malignant' — filter subset
        """
        csv_path = os.path.join(root, BUSBRA_CSV_FILENAME)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{BUSBRA_CSV_FILENAME} not found in {root}")

        df = pd.read_csv(csv_path)
        df['Pathology'] = df['Pathology'].str.strip().str.lower()

        if pathology is not None:
            pathology = pathology.strip().lower()
            if pathology not in LABEL_MAP:
                raise ValueError(f"pathology must be one of {list(LABEL_MAP)}, got '{pathology}'")
            df = df[df['Pathology'] == pathology].reset_index(drop=True)

        # Drop rows with unknown pathology
        df = df[df['Pathology'].isin(LABEL_MAP)].reset_index(drop=True)

        self.samples = []
        for _, row in df.iterrows():
            img_path  = os.path.join(root, 'Images', f"{row['ID']}.png")
            mask_name = row['ID'].replace('bus_', 'mask_')
            mask_path = os.path.join(root, 'Masks', f"{mask_name}.png")
            label     = LABEL_MAP[row['Pathology']]
            self.samples.append((img_path, mask_path, label))

        self._transform = _val_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        img, mask = self._transform(img, mask)
        return {
            'image': img,
            'mask':  mask,
            'label': torch.tensor(label, dtype=torch.long),
        }


def _val_transform(img, mask):
    img  = TF.resize(img,  [224, 224])
    mask = TF.resize(mask, [224, 224],
                     interpolation=TF.InterpolationMode.NEAREST)
    img  = TF.to_tensor(img)
    img  = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mask = torch.from_numpy(
        (np.array(mask, dtype=np.float32) > 128).astype(np.float32)
    ).unsqueeze(0)
    return img, mask


# ─────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────

def download_busbra(download_dir='/workspace'):
    """
    Downloads BUS-BRA from KaggleHub into download_dir and returns the root
    folder that contains bus_data.csv.
    KaggleHub cache is redirected to download_dir via KAGGLE_CACHE_FOLDER env var.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download BUS-BRA.\n"
            "Install with: pip install kagglehub"
        ) from exc

    os.makedirs(download_dir, exist_ok=True)
    os.environ['KAGGLE_CACHE_FOLDER'] = download_dir

    print(f"Downloading BUS-BRA via KaggleHub into {download_dir} ...")
    path = kagglehub.dataset_download(BUSBRA_KAGGLE_HANDLE)
    print(f"Path to dataset files: {path}")

    for search_root in (path, download_dir):
        for dirpath, _, files in os.walk(search_root):
            if BUSBRA_CSV_FILENAME in files:
                print(f"BUS-BRA root found: {dirpath}")
                return dirpath

    raise FileNotFoundError(
        f"Downloaded BUS-BRA but could not locate {BUSBRA_CSV_FILENAME}.\n"
        f"KaggleHub path: {path}\n"
        f"Search root  : {download_dir}"
    )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_xai(model, loader, device, cam_method='intrinsic'):
    model.eval()
    cams, masks = [], []

    for batch in loader:
        img    = batch['image'].to(device)
        mask   = batch['mask'].to(device)
        labels = batch['label'].to(device)

        if cam_method == 'intrinsic':
            with torch.no_grad():
                logits, feat = model(img)
                cam = model.get_explanation(feat, logits, labels, method=cam_method)
        else:
            model.zero_grad(set_to_none=True)
            _, feat = model(img)
            logits = model.classifier(model.dropout(model.gap(feat).flatten(1)))
            cam = model.get_explanation(feat, logits, labels, method=cam_method)

        cams.append(cam.detach().cpu())
        masks.append(mask.detach().cpu())

    return compute_xai_metrics(torch.cat(cams), torch.cat(masks))


def evaluate_cls(model, loader, device):
    """
    Classification eval on BUS-BRA (benign=1 vs malignant=2).
    Uses the 3-class BUSI head — predicting normal(0) counts as wrong.

    Metrics are computed to be directly comparable with evaluate_model() on BUSI:
      - accuracy  : same formula (accuracy_score)
      - f1_macro  : macro over labels [1,2] only — 2-class average (note in report)
      - auc       : OvR over labels [1,2] using prob[:,1:] — consistent with BUSI OvR
      - pct_pred_normal : fraction predicted as normal (cross-domain diagnostic)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            img    = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(img)
            preds  = logits.argmax(1)
            probs  = torch.softmax(logits, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)    # shape [N, 3]

    accuracy   = float(accuracy_score(y_true, y_pred))
    f1         = float(f1_score(y_true, y_pred, labels=[1, 2],
                                average='macro', zero_division=0))
    pct_normal = float((y_pred == 0).mean())

    try:
        # OvR over the two present classes — consistent with BUSI multi_class='ovr'
        # y_prob[:, 1:] gives [P(benign), P(malignant)]; labels=[1,2]
        auc = float(roc_auc_score(
            y_true, y_prob[:, 1:],
            multi_class='ovr', labels=[1, 2],
        ))
    except ValueError:
        auc = float('nan')

    return {
        'cls_accuracy':    accuracy,
        'cls_f1_macro':    f1,
        'cls_auc':         auc,
        'pct_pred_normal': pct_normal,
    }


def run(args, device):
    # ── resolve data root ──────────────────────
    data_root = args.data_root
    if data_root is None or not os.path.exists(os.path.join(data_root, BUSBRA_CSV_FILENAME)):
        if args.download:
            data_root = download_busbra(args.download_dir)
        else:
            print(
                "[ERROR] BUS-BRA data not found.\n"
                f"  Tried: {data_root}\n"
                "  Pass --download to auto-download via KaggleHub, or\n"
                "  pass --data_root <path> pointing to the folder with bus_data.csv."
            )
            sys.exit(1)

    # ── dataset ───────────────────────────────
    dataset = BUSBRADataset(data_root, pathology=args.pathology)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pathology_tag = args.pathology or 'all'
    print(f"\n{'='*60}")
    print("  BUS-BRA Evaluation (XAI + Classification)")
    print(f"  Data root  : {data_root}")
    print(f"  Pathology  : {pathology_tag}  ({len(dataset)} samples)")
    print(f"{'='*60}")

    # ── checkpoint loop ───────────────────────
    fold_results = []

    for ckpt_path in args.checkpoint:
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Not found: {ckpt_path}")
            continue

        model = LSECNet(num_classes=3, pretrained=False).to(device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)

        print(f"\n  Checkpoint : {ckpt_path}")

        xai = evaluate_xai(model, loader, device, cam_method=args.cam_method)
        cls = evaluate_cls(model, loader, device)

        print(f"  [XAI]  Pointing Game : {xai['pointing_game']:.4f}")
        print(f"  [XAI]  Soft IoU      : {xai['soft_iou']:.4f}")
        print(f"  [XAI]  Inside Ratio  : {xai['inside_ratio']:.4f}")
        print(f"  [XAI]  AUPRC         : {xai['auprc']:.4f}")
        print(f"  [CLS]  Accuracy      : {cls['cls_accuracy']:.4f}")
        print(f"  [CLS]  F1 macro      : {cls['cls_f1_macro']:.4f}")
        print(f"  [CLS]  AUC           : {cls['cls_auc']:.4f}")
        print(f"  [CLS]  % pred normal : {cls['pct_pred_normal']:.4f}")

        fold_results.append({**xai, **cls})

    if not fold_results:
        print("\n  No valid checkpoints found.")
        sys.exit(1)

    if len(fold_results) > 1:
        print(f"\n{'='*60}")
        print(f"  Aggregated across {len(fold_results)} checkpoints")
        print(f"{'='*60}")
        aggregate_results(fold_results)


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Cross-dataset XAI evaluation of LSEC-Net on BUS-BRA'
    )
    p.add_argument(
        '--checkpoint', nargs='+', required=True,
        help='Path(s) to trained LSEC-Net .pth checkpoint(s)'
    )
    p.add_argument(
        '--data_root', default='./archive/BUSBRA/BUSBRA',
        help='Path to BUS-BRA root folder (must contain bus_data.csv, Images/, Masks/)'
    )
    p.add_argument(
        '--download', action='store_true',
        help='Auto-download BUS-BRA via KaggleHub if data_root is missing'
    )
    p.add_argument(
        '--download_dir', default='./archive',
        help='Directory for KaggleHub download (used with --download)'
    )
    p.add_argument(
        '--pathology', choices=['benign', 'malignant'], default=None,
        help='Evaluate on a specific pathology subset (default: all)'
    )
    p.add_argument(
        '--batch_size', type=int, default=32,
        help='Inference batch size'
    )
    p.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker count'
    )
    p.add_argument(
        '--device', default=None,
        help='cuda / cpu (auto-detect if omitted)'
    )
    p.add_argument(
        '--cam_method', choices=['intrinsic', 'gradcam', 'gradcampp'], default='intrinsic',
        help='CAM method for XAI evaluation'
    )
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"  Device: {device}")
    run(args, device)
