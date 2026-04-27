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

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from models.lsec_net import LSECNet
from metrics.metrics import compute_xai_metrics, aggregate_results

BUSBRA_KAGGLE_HANDLE = 'orvile/bus-bra-a-breast-ultrasound-dataset'

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
        csv_path = os.path.join(root, 'bus_data.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"bus_data.csv not found in {root}")

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
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download BUS-BRA.\n"
            "Install with: pip install kagglehub"
        ) from exc

    os.makedirs(download_dir, exist_ok=True)
    print(f"Downloading BUS-BRA via KaggleHub into {download_dir} ...")
    path = kagglehub.dataset_download(
        BUSBRA_KAGGLE_HANDLE,
        output_dir=download_dir,
    )
    print(f"Path to dataset files: {path}")

    # Walk to find the folder that contains bus_data.csv
    for dirpath, _, files in os.walk(path):
        if 'bus_data.csv' in files:
            print(f"BUS-BRA root found: {dirpath}")
            return dirpath

    for dirpath, _, files in os.walk(download_dir):
        if 'bus_data.csv' in files:
            print(f"BUS-BRA root found: {dirpath}")
            return dirpath

    raise FileNotFoundError(
        f"Downloaded BUS-BRA but could not locate bus_data.csv. "
        f"KaggleHub path: {path}"
    )


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_xai(model, loader, device):
    model.eval()
    cams, masks = [], []

    with torch.no_grad():
        for batch in loader:
            img    = batch['image'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

            _, feat = model(img)
            cam = model.get_cam(feat, labels)

            cams.append(cam.cpu())
            masks.append(mask.cpu())

    return compute_xai_metrics(torch.cat(cams), torch.cat(masks))


def run(args, device):
    # ── resolve data root ──────────────────────
    data_root = args.data_root
    if data_root is None or not os.path.exists(os.path.join(data_root, 'bus_data.csv')):
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
    print(f"  BUS-BRA XAI Evaluation")
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
        # support raw state_dict or wrapped checkpoint
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        model.load_state_dict(state)

        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n  Checkpoint : {ckpt_path}")

        xai = evaluate_xai(model, loader, device)

        print(f"  Pointing Game : {xai['pointing_game']:.4f}")
        print(f"  Soft IoU      : {xai['soft_iou']:.4f}")
        print(f"  Inside Ratio  : {xai['inside_ratio']:.4f}")

        fold_results.append(xai)

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
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device(
        args.device if args.device
        else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"  Device: {device}")
    run(args, device)
