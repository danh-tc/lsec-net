# LSEC-Net

EfficientNet-B3 + Intrinsic CAM với mask-supervision loss cho bài toán phân loại ung thư vú (BUSI dataset, 3 class).

So sánh **Variant A** (GradCAM baseline) vs **Variant B** (LSEC-Net proposed) trên 6 metrics: Accuracy, F1, AUC, Pointing Game, Soft IoU, Inside Ratio.

---

## Dataset

BUSI — 780 ảnh siêu âm vú  
`normal (0)` / `benign (1)` / `malignant (2)`

```
archive/
└── Dataset_BUSI_with_GT/
    ├── benign/       437 images, 454 masks
    ├── malignant/    210 images, 211 masks
    └── normal/       133 images, 133 masks
```

---

## Cấu trúc project

```
lsec-net/
├── data/
│   ├── __init__.py
│   └── dataset.py        # build_file_list, make_splits, BUSIDataset, get_transforms
├── models/
│   ├── __init__.py
│   └── lsec_net.py       # LSECNet (EfficientNet-B3 + CAM head)
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
python lsec_net.py \
    --mode debug \
    --data_root ./archive/Dataset_BUSI_with_GT
```

**Khi nào dùng:** Lần đầu chạy, sau khi thay đổi model/loss, khi bị lỗi lạ.

---

### 2. `train` — Train model

Chạy cả 2 variants (baseline + proposed) với 5-fold CV.  
Dùng `--folds 1` trước để lấy tín hiệu sớm, rồi `--folds 5` cho kết quả đầy đủ.

**Bước 1 — Early signal (fold 0 only):**

```bash
python lsec_net.py \
    --mode train \
    --data_root ./archive/Dataset_BUSI_with_GT \
    --folds 1 \
    --epochs 40 \
    --batch_size 32
```

Output sau fold 0:
- Nếu `accuracy >= 0.88` → in `[Fold 0 OK]`, tiếp tục step 2
- Nếu `accuracy < 0.88`  → in `[WARNING]`, nên debug trước khi chạy thêm

**Bước 2 — Full 5-fold CV:**

```bash
python lsec_net.py \
    --mode train \
    --data_root ./archive/Dataset_BUSI_with_GT \
    --folds 5 \
    --epochs 40 \
    --batch_size 32
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
accuracy               0.8023±0.0312    0.8145±0.0287
f1_macro               0.7412±0.0401    0.7589±0.0356
auc                    0.8934±0.0215    0.9012±0.0198
pointing_game          0.5134±0.0523    0.6821±0.0412  [5/5]
soft_iou               0.2341±0.0612    0.4523±0.0534  [5/5]
inside_ratio           0.4512±0.0723    0.7234±0.0612  [5/5]
```

> **Lưu ý XAI:** Pointing Game, Soft IoU, Inside Ratio chỉ được tính khi  
> `accuracy >= 0.88`. Nếu một fold không đạt ngưỡng → hiển thị `[N/5]` thay vì `[5/5]`.

---

### 3. `evaluate` — Đánh giá checkpoint đã train

Load 1 hoặc nhiều checkpoint, đánh giá trên locked test set.

**Đánh giá 1 checkpoint:**

```bash
python lsec_net.py \
    --mode evaluate \
    --data_root ./archive/Dataset_BUSI_with_GT \
    --checkpoint proposed_fold0.pth
```

**Đánh giá tất cả 5 folds và aggregate:**

```bash
python lsec_net.py \
    --mode evaluate \
    --data_root ./archive/Dataset_BUSI_with_GT \
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
| `--folds` | `1` | Số folds cần chạy (1–5) |
| `--epochs` | `40` | Số epochs tối đa mỗi fold |
| `--batch_size` | `32` | Batch size |
| `--checkpoint` | `None` | Path checkpoint(s) cho evaluate mode |

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

## Loss

| Component | Công thức | Mục đích |
|---|---|---|
| `L_cls` | CrossEntropy | Phân loại đúng class |
| `L_align` (λ=1.0) | Dice Loss(CAM, mask) | CAM overlap với GT mask |
| `L_out` (λ=0.3) | mean(CAM × (1−mask)) | Phạt CAM tràn ra ngoài mask |

Baseline chỉ dùng `L_cls`. LSEC-Net dùng cả 3.

---

## Hardware

Tested on RTX 5090. Thời gian ước tính:

| | 1 fold (1 variant) | Full (5 fold × 2 variant) |
|---|---|---|
| RTX 5090 | ~3 min | ~30 min |
| Kaggle T4 | ~20 min | ~3.5 giờ |
