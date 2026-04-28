# LSEC-Net

ConvNeXt-Tiny + Intrinsic CAM với mask-supervision loss cho bài toán phân loại ung thư vú (BUSI dataset, 3 class).

So sánh **Variant A** (CAM baseline, chỉ `L_cls`) vs **Variant B** (LSEC-Net proposed, `L_cls + L_align + L_out`) trên classification metrics và XAI metrics: Accuracy, F1, AUC, Pointing Game, Soft IoU, Inside Ratio, AUPRC.

---

## Dataset

**BUSI** — 780 ảnh siêu âm vú, 3 class:
```
Dataset_BUSI_with_GT/
├── benign/       437 images, 454 masks
├── malignant/    210 images, 211 masks
└── normal/       133 images, 133 masks
```

**BUS-BRA** — 1875 ảnh, 2 class (benign/malignant, không có normal):
```
BUSBRA/
├── bus_data.csv
├── Images/
└── Masks/
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cấu trúc project

```
lsec-net/
├── data/
│   └── dataset.py          # build_file_list, make_splits, BUSIDataset, download_busi_dataset
├── models/
│   └── lsec_net.py         # LSECNet (ConvNeXt-Tiny + CAM head)
├── losses/
│   └── losses.py           # dice_loss, outside_loss, LSECLoss
├── metrics/
│   └── metrics.py          # XAI metrics + evaluate_model + aggregate_results
├── trainer.py              # run_fold, train_and_evaluate
├── main.py                 # entry point — tất cả modes
├── evaluate_busbra.py      # BUS-BRA dataset + XAI + classification eval
├── visualize.py            # CAM comparison figures
├── plot_history.py         # training curves + paired t-test
└── requirements.txt
```

---

## Modes

```
debug       → kiểm tra pipeline nhanh (forward pass + 3 epochs)
train       → train 5-fold CV, lưu checkpoint + results
evaluate    → load checkpoint → classification metrics
xai         → load checkpoint → XAI metrics trên BUSI test set
xai-busbra  → load checkpoint → XAI + classification trên BUS-BRA (cross-dataset)
```

---

## Workflow đầy đủ để lấy data cho báo cáo

### Setup Kaggle API (cần cho download tự động)

```bash
mkdir -p ~/.kaggle
# Tạo file từ kaggle.com → Account → Create New API Token
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 1. Debug — kiểm tra pipeline

```bash
# Download BUSI + chạy debug cùng lúc
python main.py \
    --mode debug \
    --download_dataset \
    --download_dir /workspace/data
```

`--download_dataset` tải BUSI vào `/workspace/data` và tự cập nhật `--data_root`.  
Nếu thấy `Pipeline OK` → tiếp tục.

### 2. Train 5-fold, cả 2 variants

```bash
python main.py \
    --mode train \
    --data_root /workspace/data/Dataset_BUSI_with_GT \
    --folds 5 \
    --variant both \
    --epochs 50 \
    --run_name final_run
```

Output lưu vào `runs/final_run/`. Đặt biến môi trường để dùng lại:

```bash
RUN=runs/final_run
DATA=/workspace/data/Dataset_BUSI_with_GT
```

### 3. XAI trên BUSI test set

```bash
# Baseline
python main.py --mode xai --data_root $DATA \
    --checkpoint \
        $RUN/baseline_fold0.pth \
        $RUN/baseline_fold1.pth \
        $RUN/baseline_fold2.pth \
        $RUN/baseline_fold3.pth \
        $RUN/baseline_fold4.pth

# Proposed
python main.py --mode xai --data_root $DATA \
    --checkpoint \
        $RUN/proposed_fold0.pth \
        $RUN/proposed_fold1.pth \
        $RUN/proposed_fold2.pth \
        $RUN/proposed_fold3.pth \
        $RUN/proposed_fold4.pth
```

Kết quả lưu cạnh checkpoint:
```
runs/final_run/xai_busi_intrinsic_results.json
runs/final_run/xai_busi_intrinsic_results.csv
```

### 4. BUS-BRA — XAI + Classification (cross-dataset)

```bash
# Download BUS-BRA lần đầu (bỏ --busbra_download những lần sau)
python main.py --mode xai-busbra \
    --busbra_download \
    --busbra_download_dir /workspace/data \
    --checkpoint $RUN/proposed_fold0.pth

# Đặt biến sau khi biết path (in ra trong log: "BUS-BRA root found: ...")
BUSBRA=/workspace/data/<path-in-log>

# Baseline — tất cả folds
python main.py --mode xai-busbra --busbra_data_root $BUSBRA \
    --checkpoint \
        $RUN/baseline_fold0.pth \
        $RUN/baseline_fold1.pth \
        $RUN/baseline_fold2.pth \
        $RUN/baseline_fold3.pth \
        $RUN/baseline_fold4.pth

# Proposed — tất cả folds
python main.py --mode xai-busbra --busbra_data_root $BUSBRA \
    --checkpoint \
        $RUN/proposed_fold0.pth \
        $RUN/proposed_fold1.pth \
        $RUN/proposed_fold2.pth \
        $RUN/proposed_fold3.pth \
        $RUN/proposed_fold4.pth
```

Output mỗi checkpoint in:
```
[XAI]  Pointing Game : 0.xxxx
[XAI]  Soft IoU      : 0.xxxx
[XAI]  Inside Ratio  : 0.xxxx
[XAI]  AUPRC         : 0.xxxx
[CLS]  Accuracy      : 0.xxxx
[CLS]  F1 macro      : 0.xxxx   ← macro over benign/malignant (2 class)
[CLS]  AUC           : 0.xxxx   ← OvR benign vs malignant
[CLS]  % pred normal : 0.xxxx   ← cross-domain diagnostic
```

Kết quả lưu cạnh checkpoint:
```
runs/final_run/xai_busbra_intrinsic_results.json
runs/final_run/xai_busbra_intrinsic_results.csv
```

### 5. Per-class BUS-BRA (benign vs malignant riêng)

```bash
# Proposed — benign only
python main.py --mode xai-busbra --busbra_data_root $BUSBRA \
    --pathology benign \
    --checkpoint \
        $RUN/proposed_fold0.pth \
        $RUN/proposed_fold1.pth \
        $RUN/proposed_fold2.pth \
        $RUN/proposed_fold3.pth \
        $RUN/proposed_fold4.pth

# Proposed — malignant only
python main.py --mode xai-busbra --busbra_data_root $BUSBRA \
    --pathology malignant \
    --checkpoint \
        $RUN/proposed_fold0.pth \
        $RUN/proposed_fold1.pth \
        $RUN/proposed_fold2.pth \
        $RUN/proposed_fold3.pth \
        $RUN/proposed_fold4.pth
```

### 6. CAM comparison figures

```bash
# Non-normal samples (benign + malignant)
python visualize.py \
    --baseline_checkpoint $RUN/baseline_fold0.pth \
    --proposed_checkpoint $RUN/proposed_fold0.pth \
    --data_root $DATA \
    --output_dir $RUN/visualizations \
    --num_samples 6 \
    --class_filter non-normal

# Malignant only (thường compelling hơn cho báo cáo)
python visualize.py \
    --baseline_checkpoint $RUN/baseline_fold0.pth \
    --proposed_checkpoint $RUN/proposed_fold0.pth \
    --data_root $DATA \
    --output_dir $RUN/visualizations/malignant \
    --num_samples 6 \
    --class_filter malignant
```

Output: `runs/final_run/visualizations/intrinsic_compare_XX_<name>.png`

### 7. Training curves

```bash
python plot_history.py curves --run_dir $RUN
```

Output: `runs/final_run/plots/training_curves.png`

### 8. Paired t-test (statistical significance)

```bash
python plot_history.py stats --run_dir $RUN
```

Output in ra terminal:
```
  Metric               Base mean  Prop mean     Diff    p-value  Sig?
  [XAI] pointing_game     0.xxxx     0.xxxx   +0.xxx     0.xxx   **
  [XAI] soft_iou          0.xxxx     0.xxxx   +0.xxx     0.xxx   **
  ...
  Significance: * p<0.05   ** p<0.01   *** p<0.001
```

---

## Output structure sau khi chạy xong

```
runs/final_run/
├── args.json                              ← hyperparams đã dùng
├── splits.json                            ← test set cố định
│
├── baseline_fold{0-4}.pth                 ← 5 checkpoints baseline
├── proposed_fold{0-4}.pth                 ← 5 checkpoints proposed
│
├── baseline_fold{0-4}_history.json        ← loss/metrics từng epoch
├── proposed_fold{0-4}_history.json
│
├── baseline_aggregate_results.json        ← mean ± std BUSI classification
├── proposed_aggregate_results.json
│
├── Confusion_baseline_fold{0-4}.png       ← confusion matrix
├── Confusion_proposed_fold{0-4}.png
│
├── xai_busi_intrinsic_results.json        ← XAI trên BUSI test set
├── xai_busi_intrinsic_results.csv
│
├── xai_busbra_intrinsic_results.json      ← XAI + CLS trên BUS-BRA
├── xai_busbra_intrinsic_results.csv
│
├── visualizations/
│   └── intrinsic_compare_XX_<name>.png   ← CAM comparison figures
│
└── plots/
    └── training_curves.png               ← loss / F1 / PG per epoch
```

### Mapping file → báo cáo

| Nội dung báo cáo | File |
|---|---|
| Table: BUSI classification (Acc/F1/AUC) | `*_aggregate_results.json` |
| Table: BUSI XAI (PG/IoU/Ratio/AUPRC) | `xai_busi_intrinsic_results.json` |
| Table: BUS-BRA XAI + classification | `xai_busbra_intrinsic_results.json` |
| Table: Per-class BUS-BRA | terminal output bước 5 |
| Statistical significance | terminal output bước 8 |
| Figure: CAM comparison | `visualizations/*.png` |
| Figure: Training curves | `plots/training_curves.png` |
| Figure: Confusion matrix | `Confusion_*.png` |

---

## Model

**Backbone:** ConvNeXt-Tiny pretrained ImageNet-1k (via `timm`). Feature map: `[B, 768, 7, 7]`.

### Variant A — CAM Baseline

Chỉ dùng `L_cls`. XAI bằng intrinsic CAM (classifier weights).

### Variant B — LSEC-Net Proposed

| Loss | Công thức | Mục đích |
|---|---|---|
| `L_cls` | CrossEntropy + label smoothing 0.1 | Phân loại |
| `L_align` (λ=1.0) | Dice(CAM, mask) | CAM overlap với GT mask |
| `L_out` (λ=0.3) | mean(CAM × (1−mask)) | Phạt CAM tràn ra ngoài mask |

Chỉ áp dụng `L_align` và `L_out` cho benign/malignant — normal class không có lesion.

### Training strategy

- **Phase 1** (3 epochs): Backbone frozen, train head với `warmup_lr=2e-4`
- **Phase 2** (47 epochs): Unfreeze backbone, CosineAnnealingLR, early stopping patience=10

---

## Arguments

### Chung (tất cả modes)

| Argument | Default | Mô tả |
|---|---|---|
| `--mode` | — | `debug` / `train` / `evaluate` / `xai` / `xai-busbra` (bắt buộc) |
| `--data_root` | `/workspace/Dataset_BUSI_with_GT` | Path đến BUSI dataset |
| `--checkpoint` | `None` | Checkpoint `.pth` (bắt buộc với evaluate / xai / xai-busbra) |
| `--batch_size` | `16` | Batch size |
| `--seed` | `42` | Random seed |
| `--tta` | `False` | Test-Time Augmentation |
| `--cam_method` | `intrinsic` | `intrinsic` / `gradcam` / `gradcampp` |

### Download BUSI

| Argument | Default | Mô tả |
|---|---|---|
| `--download_dataset` | `False` | Tải BUSI từ KaggleHub trước khi chạy |
| `--download_dir` | `/workspace` | Thư mục tải vào (dùng persistent storage trên RunPod) |
| `--force_download` | `False` | Ép tải lại dù đã có |

### Train

| Argument | Default | Mô tả |
|---|---|---|
| `--variant` | `both` | `A` / `B` / `both` |
| `--folds` | `1` | Số folds (1 = early signal, 5 = full CV) |
| `--epochs` | `50` | Max epochs mỗi fold |
| `--aug` | `default` | `default` / `light` / `none` |
| `--label_smoothing` | `0.1` | CrossEntropy label smoothing |
| `--no_mixup` | `False` | Tắt mixup (baseline only) |
| `--dropout` | `0.3` | Classifier dropout |
| `--warmup_epochs` | `3` | Epochs frozen-backbone warmup |
| `--warmup_lr` | `2e-4` | Head LR trong warmup |
| `--head_lr` | `5e-5` | Head LR sau unfreeze |
| `--backbone_lr` | `2e-5` | Backbone LR sau unfreeze |
| `--sampler` | `balanced` | `shuffle` / `balanced` |
| `--mask_lambda1` | `1.0` | Dice alignment loss weight (Variant B) |
| `--mask_lambda2` | `0.3` | Outside-mask loss weight (Variant B) |
| `--run_name` | `None` | Tên run folder (auto-generate nếu bỏ trống) |

### xai-busbra

| Argument | Default | Mô tả |
|---|---|---|
| `--busbra_data_root` | `./archive/BUSBRA/BUSBRA` | Path đến BUS-BRA root |
| `--busbra_download` | `False` | Auto-download BUS-BRA lần đầu |
| `--busbra_download_dir` | `./archive` | Thư mục tải BUS-BRA |
| `--pathology` | `None` (all) | `benign` / `malignant` — filter subset |

---

## Hardware

Tested on RTX 5090.

| | 1 fold (1 variant) | Full (5 fold × 2 variant) |
|---|---|---|
| RTX 5090 | ~4 min | ~40 min |
| RTX 4090 | ~8 min | ~80 min |
| Kaggle T4 | ~25 min | ~4 giờ |
