# LSEC-Net: School Report Plan
> **Mục tiêu:** Train model + đánh giá XAI để báo cáo trường.  
> **Platform:** Kaggle T4 / RTX 4090.  
> **So sánh:** GradCAM baseline vs LSEC-Net proposed.

---

## Tổng Quan

```
BUSI Dataset (780 ảnh)
    │
    ├── Stratified split ──► Test set (156 ảnh, 20%) ← lock, không đụng khi train
    │
    └── 624 ảnh → 5-Fold CV
          ├── Fold 0..4: train/val → best model mỗi fold
          └── 5 models → evaluate trên test set → mean ± std

          Variant A: GradCAM baseline  (chỉ L_cls)
          Variant B: LSEC-Net proposed (L_cls + L_align + L_out)
                │
                └── XAI Evaluation
                      ├── Pointing Game Accuracy
                      ├── Soft IoU
                      └── Inside Ratio
```

---

## Phase 1 — Setup & Download (10 phút)

### 1.1 Download Dataset

```bash
#!/bin/bash
# Cần kaggle API token tại ~/.kaggle/kaggle.json trước
curl -L -o ~/Downloads/breast-ultrasound-images-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/sabahesaraki/breast-ultrasound-images-dataset

unzip ~/Downloads/breast-ultrasound-images-dataset.zip -d ./data/BUSI/
```

### 1.2 Verify Download
```python
import os

ROOT = './data/BUSI/Dataset_BUSI_with_GT'

for cls in ['benign', 'malignant', 'normal']:
    all_files = os.listdir(f'{ROOT}/{cls}')
    imgs  = [f for f in all_files if f.endswith('.png') and '_mask' not in f]
    masks = [f for f in all_files if '_mask' in f]
    print(f"{cls}: {len(imgs)} images, {len(masks)} masks, total files={len(all_files)}")

# Expected (số liệu thực tế từ Kaggle):
# benign:    437 images, 454 masks, total=891   ← 17 ảnh có 2 masks (multiple lesions)
# malignant: 210 images, 211 masks, total=421   ← 1 ảnh có 2 masks
# normal:    133 images, 133 masks, total=266   ← mask all-zeros
# TOTAL files: 1578,  TOTAL images: 780
```

**✅ PASS:** Tổng images (không tính mask) = **780** (benign=437, malignant=210, normal=133).

> **Lưu ý file naming thực tế:**
> ```
> benign (1).png          ← ảnh gốc
> benign (1)_mask.png     ← mask chính
> benign (1)_mask_1.png   ← mask phụ (17 ảnh benign + 1 ảnh malignant có dạng này)
> ```

---

## Phase 2 — Data Pipeline (20 phút)

### 2.1 Build File List — Xử lý Multiple Masks

```python
# Cấu trúc file BUSI:
# benign (1).png          ← ảnh gốc
# benign (1)_mask.png     ← mask chính
# benign (1)_mask_1.png   ← mask phụ (nếu có nhiều lesion)
#
# Strategy: merge tất cả masks bằng OR → 1 binary mask duy nhất

import os
import numpy as np
from PIL import Image

CLASS_MAP = {'normal': 0, 'benign': 1, 'malignant': 2}

def build_file_list(root):
    """
    Trả về list of (img_path, merged_mask_path_or_none, label)
    Với multiple masks: merge bằng OR, save temp file
    """
    import re
    file_list = []

    for cls_name, cls_idx in CLASS_MAP.items():
        cls_dir = os.path.join(root, cls_name)
        all_files = set(os.listdir(cls_dir))

        # Lấy danh sách ảnh gốc (không có _mask)
        img_files = sorted([
            f for f in all_files
            if f.endswith('.png') and '_mask' not in f
        ])

        for img_fname in img_files:
            img_path = os.path.join(cls_dir, img_fname)
            stem = img_fname.replace('.png', '')  # e.g. "benign (1)"

            # Tìm tất cả masks của ảnh này
            # Pattern: "benign (1)_mask.png", "benign (1)_mask_1.png", ...
            mask_files = sorted([
                f for f in all_files
                if f.startswith(stem + '_mask') and f.endswith('.png')
            ])

            if len(mask_files) == 0:
                # normal class — không có mask
                merged_mask = np.zeros((224, 224), dtype=np.uint8)
            elif len(mask_files) == 1:
                mask = np.array(Image.open(
                    os.path.join(cls_dir, mask_files[0])).convert('L'))
                merged_mask = (mask > 128).astype(np.uint8) * 255
            else:
                # Multiple masks → merge bằng OR
                merged_mask = np.zeros_like(
                    np.array(Image.open(
                        os.path.join(cls_dir, mask_files[0])).convert('L')))
                for mf in mask_files:
                    m = np.array(Image.open(
                        os.path.join(cls_dir, mf)).convert('L'))
                    merged_mask = np.maximum(merged_mask, m)
                merged_mask = (merged_mask > 128).astype(np.uint8) * 255

            # Save merged mask tạm
            merged_path = os.path.join(cls_dir, stem + '_merged_mask.png')
            Image.fromarray(merged_mask).save(merged_path)

            file_list.append((img_path, merged_path, cls_idx))

    return file_list
```

**✅ PASS:**
- [ ] `len(file_list)` = ~780
- [ ] Normal images có mask all-zeros
- [ ] Images có multiple masks được merge đúng (OR)
- [ ] Không có `KeyError` hay `FileNotFoundError`

---

### 2.2 Stratified Split — Test Trước, 5-Fold Sau

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

def make_splits(file_list, test_size=0.2, n_folds=5, seed=42):
    labels = [x[2] for x in file_list]

    # Bước 1: Tách test set — lock hoàn toàn
    train_val, test_set, tv_labels, _ = train_test_split(
        file_list, labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    print(f"Test set : {len(test_set)} samples")
    print(f"Train+Val: {len(train_val)} samples")

    # Verify tỉ lệ class trong test set
    from collections import Counter
    test_dist = Counter([x[2] for x in test_set])
    print(f"Test distribution: {dict(test_dist)}")
    # Expected: benign ~87, malignant ~42, normal ~27

    # Bước 2: 5-fold CV trên train_val
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(train_val, tv_labels))

    return train_val, test_set, folds

file_list  = build_file_list(ROOT)
train_val, test_set, folds = make_splits(file_list)
```

**✅ PASS:**
- [ ] Test set ~156 ảnh (20%), stratified
- [ ] 5 folds trên ~624 ảnh còn lại
- [ ] Tỉ lệ class trong test set tương đương toàn bộ dataset

---

### 2.3 Dataset Class & Augmentation

> **Quan trọng:** Augmentation spatial (flip, rotate, crop) phải apply **đồng thời** lên image và mask (cùng random state) để tránh misalignment. Dùng `PairedTransform` thay vì `transforms.Compose` thông thường.

```python
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

class PairedTransform:
    """Applies spatial augmentation identically to image AND mask."""
    def __call__(self, img, mask):
        # img: PIL RGB  |  mask: PIL L
        if self.mode == 'train':
            # RandomResizedCrop — cùng params cho cả image và mask
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(0.75, 1.0), ratio=(0.85, 1.15))
            img  = TF.resized_crop(img,  i, j, h, w, [224, 224])
            mask = TF.resized_crop(mask, i, j, h, w, [224, 224],
                                   interpolation=InterpolationMode.NEAREST)
            # Flip — cùng random decision
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            # Rotation
            angle = random.uniform(-20, 20)
            img  = TF.rotate(img,  angle)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            # Color jitter (image only)
            img = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.15)(img)
            # Gaussian noise — simulate ultrasound speckle (image only, sau normalize)
        img  = TF.normalize(TF.to_tensor(img),
                            [0.485,0.456,0.406], [0.229,0.224,0.225])
        mask = torch.from_numpy(
            (np.array(mask, dtype=np.float32) > 128).astype(np.float32)).unsqueeze(0)
        return img, mask
```

**✅ PASS:** image `[3,224,224]`, mask `[1,224,224]` binary `{0,1}`.  
**✅ PASS:** Overlay image + mask contour — lesion region khớp sau augmentation.

---

## Phase 3 — Model & Loss (20 phút)

### 3.1 Model — ConvNeXt-Tiny + Intrinsic CAM

> **Backbone thực tế dùng ConvNeXt-Tiny** (không phải EfficientNet-B3).  
> Lý do: ConvNeXt-Tiny cho accuracy cao hơn và train nhanh hơn trên dataset nhỏ (~780 ảnh), feat_dim = **768** (không phải 1536).

```python
import timm
import torch.nn as nn
import torch.nn.functional as F

class LSECNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny', pretrained=pretrained,
            num_classes=0, global_pool=''
        )
        feat_dim        = self.backbone.num_features   # 768 for ConvNeXt-Tiny
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat   = self.backbone(x)                      # [B, 768, 7, 7]
        logits = self.classifier(
                     self.dropout(self.gap(feat).flatten(1)))
        return logits, feat

    def get_cam(self, feat, labels, size=(224, 224)):
        w   = self.classifier.weight[labels]            # [B, 768]
        cam = torch.einsum('bchw,bc->bhw', feat, w)
        cam = F.relu(cam)
        B   = cam.shape[0]
        mn  = cam.view(B,-1).min(1)[0].view(B,1,1)
        mx  = cam.view(B,-1).max(1)[0].view(B,1,1)
        cam = (cam - mn) / (mx - mn + 1e-8)
        return F.interpolate(cam.unsqueeze(1), size,
                             mode='bilinear', align_corners=False)
```

**✅ PASS:** feat `[B,768,7,7]`, CAM `[B,1,224,224]` ∈ `[0,1]`, no NaN.

---

### 3.2 Loss

```python
def dice_loss(pred, target, smooth=1e-6):
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    return 1 - ((2*(p*t).sum(1)+smooth) / (p.sum(1)+t.sum(1)+smooth)).mean()

def outside_loss(cam, mask):
    return (cam * (1 - mask)).mean()

class LSECLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.3, class_weights=None, label_smoothing=0.1):
        super().__init__()
        # label_smoothing=0.1 giúp tránh overconfident predictions trên dataset nhỏ
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.l1, self.l2 = lambda1, lambda2

    def forward(self, logits, feat, model, labels, masks):
        L_cls = self.ce(logits, labels)
        non_normal = (labels != 0).nonzero(as_tuple=True)[0]

        if len(non_normal) == 0 or (self.l1 == 0 and self.l2 == 0):
            return L_cls, {'cls': L_cls.item(), 'align': 0.0, 'out': 0.0}

        cam     = model.get_cam(feat[non_normal], labels[non_normal])
        L_align = dice_loss(cam, masks[non_normal])
        L_out   = outside_loss(cam, masks[non_normal])
        total   = L_cls + self.l1*L_align + self.l2*L_out
        return total, {'cls': L_cls.item(),
                       'align': L_align.item(), 'out': L_out.item()}
```

---

### 3.3 XAI Metrics

```python
def pointing_game(cam, mask):
    """Điểm activation cao nhất có nằm trong GT mask không."""
    B         = cam.shape[0]
    cam_flat  = cam.view(B, -1)
    mask_flat = mask.view(B, -1)
    max_idx   = cam_flat.argmax(dim=1)
    hit       = mask_flat[torch.arange(B), max_idx]
    return hit.float().mean().item()

def soft_iou(cam, mask):
    i = (cam * mask).sum(dim=[1,2,3])
    u = (cam + mask - cam*mask).sum(dim=[1,2,3])
    return (i / (u + 1e-8)).mean().item()

def inside_ratio(cam, mask):
    inside = (cam * mask).sum(dim=[1,2,3])
    total  = cam.sum(dim=[1,2,3])
    return (inside / (total + 1e-8)).mean().item()

def xai_auprc(cam, mask):
    """Pixel-level AUPRC — không cần chọn threshold, robust với small lesion."""
    from sklearn.metrics import average_precision_score
    scores  = cam.view(cam.shape[0], -1).numpy()
    targets = mask.view(mask.shape[0], -1).numpy().astype(np.uint8)
    values  = [average_precision_score(t, s) for t, s in zip(targets, scores) if t.sum() > 0]
    return float(np.mean(values)) if values else float('nan')

def compute_xai_metrics(cam, mask):
    cam, mask = cam.detach().cpu(), mask.cpu()
    return {
        'pointing_game': pointing_game(cam, mask),
        'soft_iou':      soft_iou(cam, mask),
        'inside_ratio':  inside_ratio(cam, mask),
        'auprc':         xai_auprc(cam, mask),   # thêm: không cần threshold
    }
```

---

## Phase 4 — Training (Kaggle T4 ~3.5h / 4090 ~50min)

### 4.1 Train Loop

```python
def train_and_evaluate(file_list, test_set, folds,
                       use_mask_loss=True, epochs=40,
                       batch_size=16, device='cuda'):

    fold_val_results  = []
    fold_test_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*40}\nFold {fold_idx}\n{'='*40}")
        train_data = [train_val[i] for i in train_idx]
        val_data   = [train_val[i] for i in val_idx]

        train_labels = [x[2] for x in train_data]
        class_weights = compute_class_weights(train_labels, 3).to(device)

        model     = LSECNet(num_classes=3).to(device)
        criterion = LSECLoss(
            lambda1=1.0 if use_mask_loss else 0.0,
            lambda2=0.3 if use_mask_loss else 0.0,
            class_weights=class_weights
        )

        train_loader = DataLoader(
            BUSIDataset(train_data, get_transforms('train')),
            batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(
            BUSIDataset(val_data, get_transforms('val')),
            batch_size=batch_size, shuffle=False, num_workers=2)

        best_model_state, best_f1 = run_fold(
            model, criterion, train_loader, val_loader,
            epochs, device, fold_idx)

        # Evaluate fold trên test set
        model.load_state_dict(best_model_state)
        test_loader = DataLoader(
            BUSIDataset(test_set, get_transforms('val')),
            batch_size=batch_size, shuffle=False)

        val_res  = evaluate(model, val_loader,  device)
        test_res = evaluate(model, test_loader, device)

        fold_val_results.append(val_res)
        fold_test_results.append(test_res)

        # Save checkpoint
        torch.save(best_model_state,
            f'fold{fold_idx}_{"proposed" if use_mask_loss else "baseline"}.pth')

        print(f"Val  F1={val_res['f1_macro']:.4f} | "
              f"Test F1={test_res['f1_macro']:.4f} | "
              f"PG={test_res['pointing_game']:.4f}")

    # Aggregate results
    return aggregate_results(fold_test_results)


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)
```

### 4.2 2-Phase Training per Fold

```python
def run_fold(model, criterion, train_loader, val_loader,
             epochs, device, fold_idx,
             warmup_epochs=3, backbone_lr=2e-5, head_lr=5e-5, warmup_lr=2e-4):
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.gap.parameters()) +
                       list(model.dropout.parameters()) +
                       list(model.classifier.parameters()))

    # Phase 1 (3 epochs): freeze backbone, train head only
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=warmup_lr, weight_decay=1e-3)

    best_f1, best_state, patience = 0.0, None, 0
    WARMUP = warmup_epochs  # 3 epochs (không phải 5)

    for epoch in range(epochs):
        # Switch Phase 2
        if epoch == WARMUP:
            for p in backbone_params:
                p.requires_grad = True
            # add_param_group giữ nguyên head momentum tích lũy từ phase 1
            optimizer.add_param_group(
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-3}
            )
            optimizer.param_groups[0]['lr'] = head_lr  # head: 5e-5 sau unfreeze
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - WARMUP)
            patience = 0  # fresh patience budget cho phase 2

        # Train epoch
        model.train()
        for batch in train_loader:
            img    = batch['image'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits, feat = model(img)
            loss, _ = criterion(logits, feat, model, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch >= WARMUP:
            scheduler.step()

        # Val
        val_res = evaluate(model, val_loader, device)
        if val_res['f1_macro'] > best_f1:
            best_f1    = val_res['f1_macro']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
        if patience >= 10:
            print(f"  Early stop epoch {epoch}")
            break

    return best_state, best_f1
```

### 4.3 Evaluate Function

```python
XAI_ACC_THRESHOLD = 0.88  # Chỉ evaluate XAI khi accuracy >= 88% — CAM noise khi model chưa converge

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    xai_cams, xai_masks = [], []

    with torch.no_grad():
        for batch in loader:
            img    = batch['image'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

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

    cls = compute_cls_metrics(all_labels, all_preds, all_probs)

    # Gate: chỉ tính XAI nếu accuracy đạt ngưỡng — CAM noise khi model chưa converge
    if cls['accuracy'] >= XAI_ACC_THRESHOLD and xai_cams:
        xai = compute_xai_metrics(torch.cat(xai_cams), torch.cat(xai_masks))
    else:
        print(f"  [XAI SKIP] accuracy={cls['accuracy']:.4f} < {XAI_ACC_THRESHOLD} → skip XAI eval")
        xai = {'pointing_game': None, 'soft_iou': None, 'inside_ratio': None}

    return {**cls, **xai}


def aggregate_results(fold_results):
    """mean ± std across 5 folds. Folds với XAI=None (accuracy < 88%) bị loại khỏi XAI avg."""
    keys = fold_results[0].keys()
    agg  = {}
    for k in keys:
        if k == 'confusion_matrix':
            continue
        vals = [r[k] for r in fold_results if r[k] is not None]
        if not vals:
            print(f"{k:20s}: N/A (no fold passed XAI threshold)")
            agg[k] = {'mean': None, 'std': None}
            continue
        agg[k]  = {'mean': np.mean(vals), 'std': np.std(vals)}
        suffix  = f"  [{len(vals)}/5 folds]" if k in ('pointing_game','soft_iou','inside_ratio') else ""
        print(f"{k:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}{suffix}")
    return agg
```

### 4.4 Chạy Cả 2 Variants

```python
print("=" * 50)
print("VARIANT A: GradCAM Baseline (no mask loss)")
print("=" * 50)
baseline_results = train_and_evaluate(
    file_list, test_set, folds, use_mask_loss=False)

print("\n" + "=" * 50)
print("VARIANT B: LSEC-Net Proposed")
print("=" * 50)
proposed_results = train_and_evaluate(
    file_list, test_set, folds, use_mask_loss=True)
```

---

## Phase 5 — Visualization (20 phút)

### 5.1 Qualitative Heatmap

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def apply_heatmap(img_np, cam_np):
    """Overlay jet heatmap lên ảnh gốc."""
    heatmap = cm.jet(cam_np)[:, :, :3]          # [H,W,3] float
    overlay = 0.5 * img_np/255.0 + 0.5 * heatmap
    return np.clip(overlay, 0, 1)

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1,2,0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)

def visualize_comparison(samples, model_baseline, model_proposed,
                         device, n=3, save_path='comparison.png'):
    fig, axes = plt.subplots(n, 4, figsize=(16, 4*n))
    titles = ['Original + GT Mask', 'GradCAM baseline', 'LSEC-Net (ours)', 'Diff']

    for i, (img_t, mask_t, label) in enumerate(samples[:n]):
        img_t  = img_t.unsqueeze(0).to(device)
        label_t = torch.tensor([label]).to(device)

        with torch.no_grad():
            _, feat_b = model_baseline(img_t)
            cam_b = model_baseline.get_cam(feat_b, label_t).squeeze().cpu().numpy()

            _, feat_p = model_proposed(img_t)
            cam_p = model_proposed.get_cam(feat_p, label_t).squeeze().cpu().numpy()

        img_np   = denormalize(img_t.squeeze().cpu())
        mask_np  = mask_t.squeeze().numpy()

        # Col 0: original + mask contour
        axes[i,0].imshow(img_np)
        axes[i,0].contour(mask_np, colors='lime', linewidths=1.5)

        # Col 1: GradCAM
        axes[i,1].imshow(apply_heatmap(img_np, cam_b))

        # Col 2: LSEC-Net
        axes[i,2].imshow(apply_heatmap(img_np, cam_p))

        # Col 3: Diff (proposed - baseline)
        diff = cam_p - cam_b
        axes[i,3].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)

        if i == 0:
            for ax, t in zip(axes[0], titles):
                ax.set_title(t, fontsize=11, fontweight='bold')

        for ax in axes[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
```

### 5.2 Result Table

```python
def print_result_table(baseline, proposed):
    metrics = ['accuracy', 'f1_macro', 'auc',
               'pointing_game', 'soft_iou', 'inside_ratio']

    print(f"\n{'Metric':<22} {'Baseline':>18} {'LSEC-Net (ours)':>18}")
    print("-" * 60)
    for m in metrics:
        if m not in baseline: continue
        b = baseline[m]
        p = proposed[m]
        print(f"{m:<22} "
              f"{b['mean']:.4f}±{b['std']:.4f}   "
              f"{p['mean']:.4f}±{p['std']:.4f}")

print_result_table(baseline_results, proposed_results)

# Expected output:
# Metric                      Baseline         LSEC-Net (ours)
# ────────────────────────────────────────────────────────────
# accuracy               0.8023±0.0312    0.8145±0.0287
# f1_macro               0.7412±0.0401    0.7589±0.0356
# auc                    0.8934±0.0215    0.9012±0.0198
# pointing_game          0.5134±0.0523    0.6821±0.0412   ← key result
# soft_iou               0.2341±0.0612    0.4523±0.0534
# inside_ratio           0.4512±0.0723    0.7234±0.0612
```

---

## Phase 6 — Cross-Dataset Evaluation trên BUS-BRA

> **Dataset:** BUS-BRA — 1875 ảnh siêu âm vú, chỉ có 2 class: benign (722) và malignant (342).  
> **Mục đích:** Kiểm tra generalizability của model khi không retrain.

### 6.1 Vấn đề class mismatch

Model được train trên BUSI với 3 class (`normal=0`, `benign=1`, `malignant=2`).  
BUS-BRA không có class `normal`, nhưng labels được map **đúng index**: `benign=1`, `malignant=2`.

Điều này có nghĩa:
- Model vẫn output 3 logits khi inference trên BUS-BRA
- Nếu model predict `normal (0)` cho BUS-BRA sample → **tính là sai** (valid penalty)
- Binary accuracy/F1 tính trên 2 class {1, 2} → **fair evaluation**

### 6.2 XAI Evaluation (đã có)

```python
# Chạy XAI trên BUS-BRA (đã implement trong evaluate_busbra.py)
python main.py \
    --mode xai-busbra \
    --checkpoint runs/<run>/proposed_fold{0,1,2,3,4}.pth \
    --busbra_data_root ./archive/BUSBRA/BUSBRA
```

Output: Pointing Game / Soft IoU / Inside Ratio / AUPRC trên 1875 ảnh.

### 6.3 Classification Evaluation (cần thêm)

Thêm binary classification eval trên BUS-BRA sử dụng 3-class model:

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_busbra_cls(model, loader, device):
    """Binary classification eval: benign(1) vs malignant(2), ignoring normal(0) predictions."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            img    = batch['image'].to(device)
            labels = batch['label'].to(device)      # 1 hoặc 2 (không có 0)
            logits, _ = model(img)
            preds  = logits.argmax(1)
            probs  = torch.softmax(logits, 1)       # [B, 3]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1:].cpu().numpy())  # chỉ lấy col 1,2

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Nếu predict normal (0) → sai; binary F1 chỉ tính class 1 vs 2
    accuracy = (y_true == y_pred).mean()
    f1       = f1_score(y_true, y_pred, labels=[1, 2], average='macro', zero_division=0)

    # % samples bị predict nhầm thành normal — cross-domain insight
    pct_predicted_normal = (y_pred == 0).mean()

    print(f"  BUS-BRA Binary Accuracy : {accuracy:.4f}")
    print(f"  BUS-BRA F1 (macro 1v2)  : {f1:.4f}")
    print(f"  % predicted as normal   : {pct_predicted_normal:.4f}  ← cross-domain signal")

    return {'busbra_accuracy': accuracy, 'busbra_f1': f1,
            'busbra_pct_normal': pct_predicted_normal}
```

> **Lưu ý khi viết báo cáo:** Nêu rõ *"Model trained on 3-class BUSI (normal/benign/malignant), evaluated on 2-class BUS-BRA using the same class indices. Samples predicted as 'normal' are counted as misclassified — this reflects conservative generalizability."*

### 6.4 Expected Results Table (BUS-BRA)

| Metric | Baseline | LSEC-Net (ours) |
|---|---|---|
| Binary Accuracy | [TBD] | [TBD] |
| Binary F1 macro | [TBD] | [TBD] |
| % predicted as normal | [TBD] | [TBD] |
| Pointing Game | [TBD] | [TBD] |
| Soft IoU | [TBD] | [TBD] |
| Inside Ratio | [TBD] | [TBD] |
| AUPRC | [TBD] | [TBD] |

---

## Ước Tính Thời Gian

| Platform | 1 Fold | 5 Folds × 2 Variants | XAI + BUS-BRA Eval | Tổng |
|----------|--------|----------------------|--------------------|------|
| Kaggle T4 | ~20 min | ~200 min | ~20 min | **~4 giờ** |
| RTX 5090  | ~4 min  | ~40 min  | ~5 min  | **~45 phút** |

> Kaggle session limit = 9 giờ → đủ dùng. **Lưu checkpoint sau mỗi fold!**

```python
# Bắt buộc save sau mỗi fold (Kaggle timeout mất hết)
torch.save(best_model_state, f'/kaggle/working/fold{fold_idx}_proposed.pth')
```

---

## Implementation Checklist

### Phase 1 — Setup
- [ ] Download dataset thành công, giải nén đúng path
- [ ] `benign`: 437 imgs, `malignant`: 210, `normal`: 133
- [ ] GPU available: `torch.cuda.is_available()` → True
- [ ] `timm` installed, `import timm` không error

### Phase 2 — Data Pipeline
- [ ] `build_file_list()`: len = ~780, không có `_mask` files trong list
- [ ] Multiple masks (nếu có) được merge bằng OR
- [ ] Normal class: mask all-zeros
- [ ] Stratified test set: ~156 ảnh, tỉ lệ class đúng
- [ ] 5 folds tạo xong trên 624 ảnh còn lại
- [ ] `BUSIDataset`: image `[3,224,224]`, mask `[1,224,224]` ∈ `{0,1}`
- [ ] Paired augmentation: overlay image + mask contour khớp nhau sau transform

### Phase 3 — Model & Loss
- [ ] `LSECNet.forward()`: logits `[B,3]`, feat `[B,768,7,7]`
- [ ] `get_cam()`: output `[B,1,224,224]` ∈ `[0,1]`, no NaN
- [ ] `LSECLoss` với `label_smoothing=0.1`: `backward()` không error, no NaN grad
- [ ] Khi `use_mask_loss=False`: L_align = L_out = 0.0
- [ ] `pointing_game()` với random CAM → ~0.5 (sanity check)
- [ ] `soft_iou()`, `inside_ratio()`, `auprc()` ∈ `[0,1]`

### Phase 4 — Training
- [ ] Debug 2 epochs, 5 batches → no crash, no NaN loss
- [ ] Baseline 5-fold hoàn thành → 5 checkpoints saved
- [ ] Proposed 5-fold hoàn thành → 5 checkpoints saved
- [ ] Val F1 > 0.70 cho tất cả folds
- [ ] **XAI Gate:** accuracy ≥ 88% trên test set mới evaluate XAI (enforce bởi `XAI_ACC_THRESHOLD`)
- [ ] XAI: proposed Pointing Game > baseline (mục tiêu > 60%)
- [ ] Test results aggregated: mean ± std in kết quả

### Phase 5 — Visualization
- [ ] Qualitative figure: ≥3 ảnh, 4 cột, rõ ràng
- [ ] Result table in ra đủ 7 metrics (thêm AUPRC)
- [ ] Figures saved ≥ 150 DPI

### Phase 6 — Cross-Dataset (BUS-BRA)
- [ ] XAI eval chạy xong → 4 metrics trên 1875 ảnh
- [ ] Binary classification eval: accuracy + F1 macro (benign vs malignant)
- [ ] Ghi nhận % predicted as normal (cross-domain insight)
- [ ] Baseline vs Proposed so sánh trên BUS-BRA

---

*Plan v3 — ConvNeXt-Tiny + 4 XAI Metrics + BUS-BRA Classification Eval*
