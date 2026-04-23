# LSEC-Net

ConvNeXt-Tiny + Intrinsic CAM với mask-supervision loss cho bài toán phân loại ung thư vú (BUSI dataset, 3 class).

So sánh **Variant A** (GradCAM baseline) vs **Variant B** (LSEC-Net proposed) trên 6 metrics: Accuracy, F1, AUC, Pointing Game, Soft IoU, Inside Ratio.

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

Sau khi tải xong, các lần chạy sau có thể dùng trực tiếp:

```bash
--data_root /workspace/busi_data/Dataset_BUSI_with_GT
```

> Không nên dùng `--download_dir /workspace` vì KaggleHub yêu cầu output dir rỗng.

---

## Cấu trúc project

```
lsec-net/
├── data/
│   ├── __init__.py
│   └── dataset.py        # build_file_list, make_splits, BUSIDataset, get_transforms
├── models/
│   ├── __init__.py
│   └── lsec_net.py       # LSECNet (ConvNeXt-Tiny + CAM head)
├── losses/
│   ├── __init__.py
│   └── losses.py         # dice_loss, outside_loss, LSECLoss
├── metrics/
│   ├── __init__.py
│   └── metrics.py        # XAI metrics + evaluate_model + aggregate_results
├── trainer.py            # run_fold, train_and_evaluate
├── main.py               # entry point — argparse + mode dispatch
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

### 1. `debug` — Kiểm tra pipeline, không mất thời gian

Chạy fold 0, chỉ 3 epochs, in loss từng batch.  
Dùng để xác nhận: forward pass OK, loss giảm, không NaN, không crash.

```bash
python main.py \
    --mode debug \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT
```

Nếu chưa tải dataset, chạy debug kèm download:

```bash
python main.py \
    --mode debug \
    --download_dataset \
    --download_dir /workspace/busi_data
```

**Khi nào dùng:** Lần đầu chạy, sau khi thay đổi model/loss, khi bị lỗi lạ.

---

### 2. `train` — Train model

Chạy cả 2 variants (baseline + proposed) với 5-fold CV.  
Dùng `--folds 1` trước để lấy tín hiệu sớm, rồi `--folds 5` cho kết quả đầy đủ.

**Bước 1 — Early signal (fold 0 only):**

```bash
python main.py \
    --mode train \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --folds 1
```

Output sau fold 0:
- Nếu `accuracy >= 0.88` → in `[Fold 0 OK]`, tiếp tục step 2
- Nếu `accuracy < 0.88`  → in `[WARNING]`, nên debug trước khi chạy thêm

**Bước 2 — Full 5-fold CV:**

```bash
python main.py \
    --mode train \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --folds 5
```

**Checkpoints được lưu:**
```
baseline_fold0.pth  baseline_fold1.pth  ...  baseline_fold4.pth
proposed_fold0.pth  proposed_fold1.pth  ...  proposed_fold4.pth
```

**Output cuối — Comparison table:**
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

> **Lưu ý XAI:** Pointing Game, Soft IoU, Inside Ratio chỉ được tính khi  
> `accuracy >= 0.88`. Nếu một fold không đạt ngưỡng → hiển thị `[N/5]` thay vì `[5/5]`.

---

### 3. `evaluate` — Đánh giá checkpoint đã train

Load 1 hoặc nhiều checkpoint, đánh giá trên locked test set.

**Đánh giá 1 checkpoint:**

```bash
python main.py \
    --mode evaluate \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --checkpoint proposed_fold0.pth
```

**Đánh giá tất cả 5 folds và aggregate:**

```bash
python main.py \
    --mode evaluate \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT \
    --checkpoint proposed_fold0.pth proposed_fold1.pth proposed_fold2.pth proposed_fold3.pth proposed_fold4.pth
```

**Output cho mỗi checkpoint:**
- Accuracy / Precision / Recall / F1 (weighted + macro)
- MAE / RMSE / AUC (multi-class OvR)
- Pointing Game / Soft IoU / Inside Ratio (nếu accuracy >= 0.88)
- Confusion matrix PNG: `Confusion_proposed_fold0.png`

---

## Arguments

| Argument | Default | Mô tả |
|---|---|---|
| `--mode` | — | `debug` / `train` / `evaluate` (bắt buộc) |
| `--data_root` | `/workspace/Dataset_BUSI_with_GT` | Path đến thư mục dataset |
| `--download_dataset` | `False` | Tải BUSI từ KaggleHub trước khi chạy |
| `--download_dir` | `/workspace` | Thư mục để KaggleHub tải dataset vào |
| `--force_download` | `False` | Ép KaggleHub tải lại dataset |
| `--variant` | `both` | `A` / `B` / `both` |
| `--folds` | `1` | Số folds cần chạy (1–5) |
| `--epochs` | `50` | Số epochs tối đa mỗi fold |
| `--batch_size` | `16` | Batch size |
| `--aug` | `default` | Augmentation strength: `default` / `light` / `none` |
| `--label_smoothing` | `0.1` | CrossEntropy label smoothing |
| `--no_mixup` | `False` | Tắt mixup cho baseline |
| `--dropout` | `0.3` | Classifier dropout |
| `--warmup_epochs` | `3` | Epochs frozen-backbone warmup |
| `--warmup_lr` | `2e-4` | Head LR trong warmup |
| `--head_lr` | `5e-5` | Head LR sau khi unfreeze |
| `--backbone_lr` | `2e-5` | Backbone LR sau khi unfreeze |
| `--sampler` | `balanced` | `shuffle` hoặc `balanced` (WeightedRandomSampler) |
| `--calibrate_logits` | `False` | Tune logit bias trên val trước khi eval test |
| `--tta` | `False` | Test-Time Augmentation |
| `--backbone_weights` | `None` | Path pretrained backbone `.pth` (load strict=False) |
| `--checkpoint` | `None` | Checkpoint `.pth` path(s) cho evaluate mode |

Nếu CUDA không khả dụng, có thể test flow bằng CPU:

```bash
CUDA_VISIBLE_DEVICES="" python main.py \
    --mode debug \
    --data_root /workspace/busi_data/Dataset_BUSI_with_GT
```

---

## Workflow đề xuất

```
1. debug          → pipeline OK?
        ↓ yes
2. train --folds 1 → fold 0 accuracy >= 88%?
        ↓ yes
3. train --folds 5 → lấy kết quả đầy đủ
        ↓
4. evaluate --checkpoint proposed_fold*.pth  → report cuối
```

---

## Backbone

Dùng **ConvNeXt-Tiny** pretrained trên ImageNet-1k (via `timm`).  
Feature map output: `[B, 768, 7, 7]` với input 224×224.

**Tại sao ConvNeXt-Tiny thay vì ResNet50:**
- Kernel 7×7 bắt texture tốt hơn cho ảnh siêu âm (speckle pattern)
- LayerNorm ổn định hơn BatchNorm với batch size nhỏ
- Accuracy cao hơn ~3-5% với cùng số parameters (~28M)
- CAM vẫn tương thích đầy đủ — `feat = [B, 768, 7, 7]`, công thức `einsum('bchw,bc->bhw')` không đổi

### Variant A — GradCAM Baseline

Chỉ dùng `L_cls`. XAI bằng Grad-CAM post-hoc sau khi train xong.

### Variant B — LSEC-Net Proposed

Train Intrinsic CAM trực tiếp với mask supervision:

| Component | Công thức | Mục đích |
|---|---|---|
| `L_cls` | CrossEntropy | Phân loại đúng class |
| `L_align` (λ=1.0) | Dice Loss(CAM, mask) | CAM overlap với GT mask |
| `L_out` (λ=0.3) | mean(CAM × (1−mask)) | Phạt CAM tràn ra ngoài mask |

CAM được tính differentiable trong quá trình train → attention map học được lesion boundary từ GT mask.

---

## Training strategy

**2-phase training:**
- **Phase 1** (warmup, 3 epochs): Backbone frozen, chỉ train head với `warmup_lr=2e-4`
- **Phase 2** (fine-tune, 47 epochs): Backbone unfreeze với `backbone_lr=2e-5`, head `head_lr=5e-5`, CosineAnnealingLR

Early stopping patience=10, monitored trên val F1-macro. Patience reset khi bước vào Phase 2 để đảm bảo backbone có đủ budget để adapt.

**Class imbalance:** `--sampler balanced` (WeightedRandomSampler) + class-weighted CrossEntropy.

---

## Hardware

Tested on RTX 5090. Thời gian ước tính:

| | 1 fold (1 variant) | Full (5 fold × 2 variant) |
|---|---|---|
| RTX 5090 | ~4 min | ~40 min |
| Kaggle T4 | ~25 min | ~4 giờ |
