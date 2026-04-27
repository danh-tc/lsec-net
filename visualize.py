"""
Generate qualitative CAM comparison figures for LSEC-Net reports.

Example:
  python visualize.py \
    --baseline_checkpoint runs/<run>/baseline_fold0.pth \
    --proposed_checkpoint runs/<run>/proposed_fold0.pth \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --output_dir runs/<run>/visualizations/fold0
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from data.dataset import CLASS_NAMES, build_file_list, make_splits
from models.lsec_net import LSECNet


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_state_dict(path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    return state


def load_model(path, device):
    model = LSECNet(num_classes=3, pretrained=False).to(device)
    model.load_state_dict(load_state_dict(path, device))
    model.eval()
    return model


def find_splits_json(*checkpoint_paths):
    for ckpt_path in checkpoint_paths:
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
        splits_path = os.path.join(ckpt_dir, 'splits.json')
        if os.path.exists(splits_path):
            return splits_path
    return None


def load_test_set(args):
    if args.splits_json:
        splits_path = args.splits_json
    else:
        splits_path = find_splits_json(args.baseline_checkpoint, args.proposed_checkpoint)

    if splits_path:
        with open(splits_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Splits loaded: {splits_path}")
        return [tuple(x) for x in data['test_set']]

    print(f"[WARN] splits.json not found. Re-splitting with seed={args.seed}.")
    file_list = build_file_list(args.data_root)
    _, test_set, _ = make_splits(file_list, seed=args.seed)
    return test_set


def normalize_for_model(img):
    img = TF.resize(img, [224, 224])
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, IMAGENET_MEAN.flatten().tolist(), IMAGENET_STD.flatten().tolist())


def load_sample(sample):
    img_path, mask_path, label = sample
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    img_tensor = normalize_for_model(img).unsqueeze(0)
    mask_np = np.array(TF.resize(mask, [224, 224], interpolation=TF.InterpolationMode.NEAREST))
    mask_np = (mask_np > 128).astype(np.float32)
    img_np = np.array(TF.resize(img, [224, 224]))
    return img_tensor, img_np, mask_np, int(label)


def predict_cam(model, img_tensor, device, cam_method):
    img_tensor = img_tensor.to(device)
    if cam_method == 'intrinsic':
        with torch.no_grad():
            logits, feat = model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
            conf = float(probs[0, pred].item())
            cam = model.get_explanation(
                feat, logits, torch.tensor([pred], device=device), method=cam_method)
    else:
        model.zero_grad(set_to_none=True)
        logits, feat = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        conf = float(probs[0, pred].item())
        cam = model.get_explanation(
            feat, logits, torch.tensor([pred], device=device), method=cam_method)
    return pred, conf, cam.squeeze().detach().cpu().numpy()


def apply_heatmap(img_np, cam_np, alpha=0.5):
    heatmap = plt.get_cmap('jet')(cam_np)[..., :3]
    image = img_np.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * image + alpha * heatmap, 0.0, 1.0)


def draw_mask_contour(ax, mask_np, color='lime'):
    ax.contour(mask_np, levels=[0.5], colors=[color], linewidths=1.2)


def filter_samples(test_set, class_filter):
    if class_filter == 'all':
        return test_set
    if class_filter == 'non-normal':
        return [x for x in test_set if x[2] != 0]
    label_map = {'normal': 0, 'benign': 1, 'malignant': 2}
    return [x for x in test_set if x[2] == label_map[class_filter]]


def select_samples(test_set, args):
    samples = filter_samples(test_set, args.class_filter)
    if not samples:
        raise ValueError(f"No samples found for class_filter={args.class_filter}")

    if args.random:
        rng = random.Random(args.seed)
        rng.shuffle(samples)

    return samples[:args.num_samples]


def save_comparison(sample, idx, baseline_model, proposed_model, device, output_dir, cam_method):
    img_tensor, img_np, mask_np, label = load_sample(sample)
    base_pred, base_conf, base_cam = predict_cam(baseline_model, img_tensor, device, cam_method)
    prop_pred, prop_conf, prop_cam = predict_cam(proposed_model, img_tensor, device, cam_method)
    diff = prop_cam - base_cam

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_np)
    draw_mask_contour(axes[0], mask_np)
    axes[0].set_title(f"Original + GT\nlabel={CLASS_NAMES[label]}")

    axes[1].imshow(apply_heatmap(img_np, base_cam))
    draw_mask_contour(axes[1], mask_np)
    axes[1].set_title(f"Baseline {cam_method}\npred={CLASS_NAMES[base_pred]} ({base_conf:.2f})")

    axes[2].imshow(apply_heatmap(img_np, prop_cam))
    draw_mask_contour(axes[2], mask_np)
    axes[2].set_title(f"LSEC-Net {cam_method}\npred={CLASS_NAMES[prop_pred]} ({prop_conf:.2f})")

    im = axes[3].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    draw_mask_contour(axes[3], mask_np, color='black')
    axes[3].set_title("CAM diff\nproposed - baseline")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis('off')

    img_name = os.path.splitext(os.path.basename(sample[0]))[0]
    safe_name = img_name.replace(' ', '_').replace('(', '').replace(')', '')
    out_path = os.path.join(output_dir, f"{cam_method}_compare_{idx:02d}_{safe_name}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize baseline vs LSEC-Net CAM overlays')
    parser.add_argument('--baseline_checkpoint', required=True, help='Path to baseline .pth checkpoint')
    parser.add_argument('--proposed_checkpoint', required=True, help='Path to proposed .pth checkpoint')
    parser.add_argument('--data_root', default='/workspace/busi_data/Dataset_BUSI_with_GT',
                        help='BUSI Dataset_BUSI_with_GT directory')
    parser.add_argument('--splits_json', default=None,
                        help='Optional splits.json path. Defaults to checkpoint folder splits.json.')
    parser.add_argument('--output_dir', default='visualizations',
                        help='Directory where comparison PNGs will be saved')
    parser.add_argument('--num_samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--class_filter',
                        choices=['all', 'non-normal', 'normal', 'benign', 'malignant'],
                        default='non-normal',
                        help='Which test samples to visualize')
    parser.add_argument('--random', action='store_true', help='Randomly sample instead of taking first matches')
    parser.add_argument('--seed', type=int, default=42, help='Seed for split fallback and random sampling')
    parser.add_argument('--device', default=None, help='cuda / cpu. Auto-detect if omitted')
    parser.add_argument('--cam_method', choices=['intrinsic', 'gradcam', 'gradcampp'],
                        default='intrinsic', help='CAM method to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Device: {device}")

    test_set = load_test_set(args)
    samples = select_samples(test_set, args)
    print(f"Selected samples: {len(samples)} ({args.class_filter})")

    baseline_model = load_model(args.baseline_checkpoint, device)
    proposed_model = load_model(args.proposed_checkpoint, device)

    saved = []
    for idx, sample in enumerate(samples):
        out_path = save_comparison(
            sample, idx, baseline_model, proposed_model, device, args.output_dir,
            args.cam_method,
        )
        saved.append(out_path)
        print(f"Saved: {out_path}")

    print(f"\nDone. Saved {len(saved)} figure(s) to {args.output_dir}")


if __name__ == '__main__':
    main()
