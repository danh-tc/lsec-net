# LSEC-Net

ConvNeXt-Tiny + Intrinsic CAM với mask-supervision loss cho bài toán phân loại ung thư vú (BUSI dataset, 3 class).

So sánh **Variant A** (CAM baseline) vs **Variant B** (LSEC-Net proposed) trên 6 metrics: Accuracy, F1, AUC, Pointing Game, Soft IoU, Inside Ratio.

---

## Dataset

BUSI — 780 ảnh siêu âm vú  
`normal (0)` / `benign (1)` / `malignant (2)`

```
/workspace/busi_data/
└── Dataset_BUSI_with_GT/
    ├── benign/       437 images, 454 masks
    ├── malignant/    210 images, 211 masks
    └── normal/       133 images, 133 masks
```

Dataset có thể được tải tự động từ KaggleHub:

```bash
python main.py \
    --mode debug \
    --download_dataset \
    --download_dir /workspace/busi_data
```

---

## Cấu trúc project

```
lsec-net/
├── data/
│   ├── __init__.py
│   └── dataset.py          # build_file_list, make_splits, BUSIDataset, get_transforms
├── models/
│   ├── __init__.py
│   └── lsec_net.py         # LSECNet (ConvNeXt-Tiny + CAM head)
├── losses/
│   ├── __init__.py
│   └── losses.py           # dice_loss, outside_loss, LSECLoss
├── metrics/
│   ├── __init__.py
│   └── metrics.py          # XAI metrics + evaluate_model + aggregate_results
├── trainer.py              # run_fold, train_and_evaluate
├── main.py                 # entry point — tất cả modes
├── evaluate_busbra.py      # standalone script (backward compat, dùng trực tiếp vẫn được)
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Modes

Tất cả modes đều chạy qua `main.py` với flag `--mode`:

```
debug       → kiểm tra pipeline nhanh
train       → train model + eval classification sau mỗi fold
evaluate    → load checkpoint → eval classification (không cần retrain)
xai         → load checkpoint → XAI metrics trên BUSI test set
xai-busbra  → load checkpoint → XAI metrics trên BUS-BRA (cross-dataset)
visualize.py → tạo hình so sánh CAM baseline vs LSEC-Net cho báo cáo
```

**Workflow chuẩn:**

```
debug → train --folds 1 → train --folds 5 → xai → xai-busbra
                                           └──→ evaluate (nếu cần)
```

---

### 1. `debug` — Kiểm tra pipeline

Chạy fold 0, 3 epochs, in loss từng batch. Dùng để xác nhận forward pass OK trước khi train thật.

```bash
python main.py \
    --mode debug \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT
```

---

### 2. `train` — Train model

Chạy K-fold CV cho Variant A/B. Eval classification tự động sau mỗi fold.  
`splits.json` được lưu trong run directory để các mode `xai` / `xai-busbra` dùng lại đúng test set.

**Bước 1 — Early signal (fold 0):**

```bash
python main.py \
    --mode train \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --folds 1
```

Nếu `accuracy >= 0.88` sau fold 0 → tiếp tục bước 2.

**Bước 2 — Full 5-fold CV:**

```bash
python main.py \
    --mode train \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --folds 5
```

**Output:**
```
runs/<run_name>/
├── args.json
├── splits.json                       ← test set full tuples (path, mask, label) dùng bởi xai mode
├── baseline_fold0.pth ... fold4.pth
├── proposed_fold0.pth  ... fold4.pth
├── baseline_fold_results.json
├── proposed_fold_results.json
└── Confusion_*.png
```

**Comparison table cuối (khi `--variant both --folds 5`):**
```
Metric                       Baseline         LSEC-Net (ours)
────────────────────────────────────────────────────────────
accuracy               0.8800±0.0210    0.8980±0.0180
f1_macro               0.8320±0.0290    0.8610±0.0250
auc                    0.9410±0.0150    0.9560±0.0130
pointing_game          0.6200±0.0480    0.7350±0.0390  [5/5]
soft_iou               0.3450±0.0550    0.5210±0.0490  [5/5]
inside_ratio           0.5630±0.0640    0.7580±0.0560  [5/5]
```

---

### 3. `evaluate` — Đánh giá classification từ checkpoint

Load checkpoint đã có, evaluate **classification only** trên BUSI test set.  
Dùng khi muốn re-evaluate checkpoint trên máy khác hoặc sau khi tune xong.

```bash
# 1 checkpoint
python main.py \
    --mode evaluate \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --checkpoint runs/<run>/proposed_fold0.pth

# Nhiều checkpoint → aggregate mean ± std
python main.py \
    --mode evaluate \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --checkpoint runs/<run>/proposed_fold{0,1,2,3,4}.pth
```

**Output:** Accuracy / F1 / AUC / Confusion matrix PNG.

---

### 4. `xai` — Đánh giá XAI trên BUSI test set

Tính Pointing Game / Soft IoU / Inside Ratio trên **cùng test set lúc train**.

- Nếu `splits.json` tồn tại trong cùng thư mục với checkpoint → load trực tiếp (đảm bảo exact same test set).
- Nếu không có → fallback re-split với `--seed` (cần `--data_root`).

```bash
# 1 checkpoint
python main.py \
    --mode xai \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --checkpoint runs/<run>/proposed_fold0.pth

# Nhiều checkpoint → aggregate
python main.py \
    --mode xai \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --checkpoint runs/<run>/proposed_fold{0,1,2,3,4}.pth
```

---

### 5. `xai-busbra` — Cross-dataset XAI trên BUS-BRA

Inference model trained trên BUSI, evaluate XAI localization trên **BUS-BRA** (1875 ảnh, benign/malignant). Không cần BUSI data hay splits.

```bash
# Data đã có sẵn
python main.py \
    --mode xai-busbra \
    --checkpoint runs/<run>/proposed_fold0.pth \
    --busbra_data_root ./archive/BUSBRA/BUSBRA

# Auto-download qua KaggleHub
python main.py \
    --mode xai-busbra \
    --checkpoint runs/<run>/proposed_fold0.pth \
    --busbra_download \
    --busbra_download_dir ./archive

# Nhiều checkpoint + filter pathology
python main.py \
    --mode xai-busbra \
    --checkpoint runs/<run>/proposed_fold{0,1,2,3,4}.pth \
    --busbra_data_root ./archive/BUSBRA/BUSBRA \
    --pathology malignant
```

---

### 6. `visualize.py` — Vẽ heatmap so sánh

Tạo figure gồm 4 cột: ảnh gốc + GT mask, baseline CAM, LSEC-Net CAM, và chênh lệch CAM.  
Script tự dùng `splits.json` trong thư mục checkpoint nếu có, nên hình lấy đúng test set lúc train.

```bash
python visualize.py \
    --baseline_checkpoint runs/<run>/baseline_fold0.pth \
    --proposed_checkpoint runs/<run>/proposed_fold0.pth \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --output_dir runs/<run>/visualizations/fold0 \
    --num_samples 6 \
    --class_filter non-normal
```

Lọc riêng class:

```bash
python visualize.py \
    --baseline_checkpoint runs/<run>/baseline_fold0.pth \
    --proposed_checkpoint runs/<run>/proposed_fold0.pth \
    --output_dir runs/<run>/visualizations/malignant_fold0 \
    --num_samples 6 \
    --class_filter malignant
```

---

## Arguments

### Chung (tất cả modes)

| Argument | Default | Mô tả |
|---|---|---|
| `--mode` | — | `debug` / `train` / `evaluate` / `xai` / `xai-busbra` (bắt buộc) |
| `--data_root` | `/workspace/Dataset_BUSI_with_GT` | Path đến BUSI dataset |
| `--checkpoint` | `None` | Checkpoint `.pth` (bắt buộc với evaluate / xai / xai-busbra) |
| `--batch_size` | `16` | Batch size |
| `--seed` | `42` | Random seed (splits, aug, torch) |
| `--tta` | `False` | Test-Time Augmentation |

### Train

| Argument | Default | Mô tả |
|---|---|---|
| `--variant` | `both` | `A` / `B` / `both` |
| `--folds` | `1` | Số folds (1 = early signal, 5 = full CV) |
| `--epochs` | `50` | Max epochs mỗi fold |
| `--aug` | `default` | `default` / `light` / `none` |
| `--label_smoothing` | `0.1` | CrossEntropy label smoothing |
| `--no_mixup` | `False` | Tắt mixup |
| `--dropout` | `0.3` | Classifier dropout |
| `--warmup_epochs` | `3` | Epochs frozen-backbone warmup |
| `--warmup_lr` | `2e-4` | Head LR trong warmup |
| `--head_lr` | `5e-5` | Head LR sau unfreeze |
| `--backbone_lr` | `2e-5` | Backbone LR sau unfreeze |
| `--sampler` | `balanced` | `shuffle` / `balanced` |
| `--calibrate_logits` | `False` | Tune logit bias trên val trước eval |
| `--mask_lambda1` | `1.0` | Dice alignment loss weight (Variant B) |
| `--mask_lambda2` | `0.3` | Outside-mask loss weight (Variant B) |
| `--backbone_weights` | `None` | Pretrained backbone `.pth` (strict=False) |
| `--runs_dir` | `runs` | Thư mục chứa run outputs |
| `--run_name` | `None` | Tên run folder (auto-generate nếu bỏ trống) |

### Download BUSI

| Argument | Default | Mô tả |
|---|---|---|
| `--download_dataset` | `False` | Tải BUSI từ KaggleHub |
| `--download_dir` | `/workspace` | Thư mục tải vào |
| `--force_download` | `False` | Ép tải lại |

### xai-busbra

| Argument | Default | Mô tả |
|---|---|---|
| `--busbra_data_root` | `./archive/BUSBRA/BUSBRA` | Path đến BUS-BRA |
| `--busbra_download` | `False` | Auto-download BUS-BRA |
| `--busbra_download_dir` | `./archive` | Thư mục tải BUS-BRA |
| `--pathology` | `None` (all) | `benign` / `malignant` |

---

## Model

**Backbone:** ConvNeXt-Tiny pretrained ImageNet-1k (via `timm`). Feature map: `[B, 768, 7, 7]`.

### Variant A — CAM Baseline

Chỉ dùng `L_cls`. XAI bằng intrinsic CAM (classifier weights).

### Variant B — LSEC-Net Proposed

| Loss | Công thức | Mục đích |
|---|---|---|
| `L_cls` | CrossEntropy | Phân loại |
| `L_align` (λ=1.0) | Dice(CAM, mask) | CAM overlap với GT mask |
| `L_out` (λ=0.3) | mean(CAM × (1−mask)) | Phạt CAM tràn ra ngoài |

### Training strategy

- **Phase 1** (3 epochs): Backbone frozen, train head với `warmup_lr=2e-4`
- **Phase 2** (47 epochs): Unfreeze backbone, CosineAnnealingLR, early stopping patience=10

---

## Hardware

Tested on RTX 5090.

| | 1 fold (1 variant) | Full (5 fold × 2 variant) |
|---|---|---|
| RTX 5090 | ~4 min | ~40 min |
| Kaggle T4 | ~25 min | ~4 giờ |
