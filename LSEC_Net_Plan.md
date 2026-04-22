# LSEC-Net: Lesion-Supervised Explainable Classification Network
## Research Plan & TODO List

> **Target:** Computers in Biology and Medicine (Scopus Q1, IF ~7)  
> **Timeline:** ~3 tuần từ ngày bắt đầu  
> **Core idea:** Train classifier với mask supervision trong loss → CAM phải overlap GT lesion mask → vừa classify đúng, vừa "nhìn đúng chỗ"

---

## 📌 Tóm Tắt Paper (1 đoạn)

Các paper hiện tại (DNBCD, EfficientNet + Grad-CAM) dùng XAI post-hoc: train classifier trước, visualize heatmap sau. Không có cơ chế nào đảm bảo heatmap align với GT lesion. Paper này đề xuất dùng **intrinsic CAM từ GAP** (differentiable) kết hợp **mask alignment loss** để buộc model học "nhìn đúng chỗ" trong lúc train — không phải sau khi train xong. Evaluate định lượng XAI quality bằng IoU/Dice/Inside Ratio vs GT mask. Test trên BUSI (3-class) và BUS-BRA (generalizability, 2-class).

---

## 🗂️ Phase 0: Setup & Literature (Ngày 0)

### 0.1 Environment Setup
- [ ] Tạo virtual environment Python 3.10+
- [ ] Cài packages: `torch torchvision timm scikit-learn pandas numpy matplotlib seaborn tqdm`
- [ ] Verify CUDA available
- [ ] Setup project folder structure:

```
lsec_net/
├── data/
│   ├── BUSI/
│   │   ├── benign/          # image + mask
│   │   ├── malignant/       # image + mask
│   │   └── normal/          # image + mask (mask rỗng)
│   └── BUS-BRA/
│       ├── images/
│       ├── masks/
│       └── folds/           # standardized 5-fold splits
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── cam_utils.py
│   ├── metrics.py
│   └── train.py
├── configs/
│   ├── busi_config.yaml
│   └── busbra_config.yaml
├── experiments/
│   ├── busi/
│   └── busbra/
├── notebooks/
│   └── analysis.ipynb
└── README.md
```

### 0.2 Dataset Download
- [ ] Download BUSI: https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
- [ ] Download BUS-BRA: https://zenodo.org/records/8231412
- [ ] Verify số lượng ảnh BUSI: 437 benign + 210 malignant + 133 normal = **780 ảnh**
- [ ] Verify số lượng ảnh BUS-BRA: 722 benign + 342 malignant = **1875 ảnh** (không có normal)
- [ ] Kiểm tra GT mask: benign/malignant có mask lesion, normal có mask rỗng (all zeros)
- [ ] Kiểm tra BUS-BRA 5-fold splits file đi kèm

### 0.3 Literature References Cần Cite
- [ ] Lưu lại 5 paper chính làm related work:
  - DNBCD (Alom et al., Sci. Reports 2025) — baseline chính
  - EfficientNet-B7 + Grad-CAM (Latha et al., BMC Med Imaging 2024)
  - CAM gốc (Zhou et al., CVPR 2016) — technical foundation
  - "More than just a heatmap" (Frontiers Med. Tech. 2025) — support gap argument
  - HyFormer-Net (arXiv 2025) — differentiate với joint seg+cls approach
  - Multi-dataset benchmark (arXiv 2025) — reference cho BUS-BRA usage

---

## 🗂️ Phase 1: Data Pipeline (Ngày 1)

### 1.1 BUSI Dataset Class
- [ ] Viết `BUSIDataset(Dataset)`:
  - Load image + mask theo class folder
  - Resize về 224×224
  - Normalize: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (ImageNet)
  - Return: `{image, mask, label}` với label: 0=normal, 1=benign, 2=malignant
  - Normal class: mask = tensor zeros (224×224)
- [ ] Tạo 5-fold splits cho BUSI (stratified by class, **không phải random**):
  ```python
  from sklearn.model_selection import StratifiedKFold
  # Đảm bảo mỗi fold giữ nguyên tỉ lệ class
  ```
- [ ] Verify: in ra class distribution của từng fold

### 1.2 BUS-BRA Dataset Class
- [ ] Viết `BUSBRADataset(Dataset)`:
  - Load theo standardized 5-fold splits file có sẵn
  - Label: 0=benign, 1=malignant (2 class, không có normal)
  - Cùng resize/normalize như BUSI
- [ ] **Không tự tạo splits** — dùng file splits có sẵn để reproducibility

### 1.3 Augmentation
- [ ] Chỉ apply cho training set:
  - RandomHorizontalFlip(p=0.5)
  - RandomVerticalFlip(p=0.5)
  - RandomRotation(degrees=15)
  - ColorJitter(brightness=0.2, contrast=0.2)
- [ ] **Không augment validation/test set**
- [ ] Augmentation phải apply **đồng thời** lên image và mask (cùng random seed)

### 1.4 DataLoader
- [ ] Batch size: 16 (BUSI nhỏ, cần batch nhỏ)
- [ ] num_workers: 4
- [ ] pin_memory: True nếu có GPU
- [ ] Verify: plot 5 sample images + masks để kiểm tra alignment

---

## 🗂️ Phase 2: Model Architecture (Ngày 1-2)

### 2.1 Backbone
- [ ] Dùng **EfficientNet-B3** (pretrained ImageNet):
  ```python
  import timm
  backbone = timm.create_model('efficientnet_b3', pretrained=True, 
                                num_classes=0, global_pool='')
  # num_classes=0 và global_pool='' để lấy feature maps raw
  ```
  - Lý do chọn B3: balance giữa B0 (yếu) và B7 (quá lớn cho 780 ảnh)
  - Output feature map: [B, 1536, 7, 7] với input 224×224

### 2.2 Full Model
- [ ] Viết class `LSECNet(nn.Module)`:

```python
class LSECNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Backbone: EfficientNet-B3, bỏ classifier head
        self.backbone = timm.create_model('efficientnet_b3', 
                                           pretrained=True,
                                           num_classes=0, 
                                           global_pool='')
        self.feat_dim = 1536  # output channels của B3
        
        # GAP
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.feat_dim, num_classes)
    
    def forward(self, x):
        # Feature maps: [B, C, H, W]
        feat_maps = self.backbone(x)           # [B, 1536, 7, 7]
        
        # GAP → classification
        pooled = self.gap(feat_maps).flatten(1)  # [B, 1536]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)          # [B, num_classes]
        
        return logits, feat_maps  # trả về cả feature maps để tính CAM
```

### 2.3 CAM Generation
- [ ] Viết `generate_cam(feat_maps, classifier_weights, class_idx, target_size)`:

```python
def generate_cam(feat_maps, classifier_weights, class_idx, target_size=(224, 224)):
    """
    feat_maps: [B, C, H, W] - raw feature maps từ backbone
    classifier_weights: [num_classes, C] - weights của Linear layer
    class_idx: predicted class index (hoặc GT class khi train)
    target_size: output size để so sánh với GT mask
    
    Returns: cam [B, 1, H_out, W_out] normalized về [0,1]
    """
    # Lấy weights của class dự đoán
    weights = classifier_weights[class_idx]  # [C]
    
    # Tính weighted sum: CAM = sum_k(w_k * f_k)
    # feat_maps: [B, C, H, W], weights: [C]
    cam = torch.einsum('bchw,c->bhw', feat_maps, weights)  # [B, H, W]
    cam = F.relu(cam)  # chỉ giữ positive activation
    
    # Normalize về [0,1] per sample
    cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
    cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    
    # Upsample về kích thước mask
    cam = F.interpolate(cam.unsqueeze(1), size=target_size, 
                        mode='bilinear', align_corners=False)  # [B, 1, H, W]
    return cam
```

**Lưu ý quan trọng:** Dùng predicted class khi inference, dùng GT class khi tính loss để tránh gradient issue.

### 2.4 Training Phases
- [ ] **Phase 1 (Warmup, 5 epochs):** Freeze backbone, chỉ train classifier head → L_cls only
- [ ] **Phase 2 (Full, 35 epochs):** Unfreeze toàn bộ, train với L_total

---

## 🗂️ Phase 3: Loss Functions (Ngày 2)

### 3.1 L_cls — Classification Loss
```python
L_cls = nn.CrossEntropyLoss(weight=class_weights)
# class_weights: tính từ inverse frequency để handle imbalance
# BUSI: normal(133) < malignant(210) < benign(437)
```
- [ ] Tính class weights: `w_i = N / (n_classes * n_i)`
- [ ] Verify class weights trước khi train

### 3.2 L_align — CAM-Mask Alignment Loss
```python
def dice_loss(cam, gt_mask, smooth=1e-6):
    """
    cam: [B, 1, H, W] normalized [0,1]
    gt_mask: [B, 1, H, W] binary {0,1}
    """
    intersection = (cam * gt_mask).sum(dim=[1,2,3])
    dice = (2 * intersection + smooth) / \
           (cam.sum(dim=[1,2,3]) + gt_mask.sum(dim=[1,2,3]) + smooth)
    return 1 - dice.mean()

L_align = dice_loss(cam, gt_mask)
```

### 3.3 L_outside — Outside Penalty Loss
```python
def outside_loss(cam, gt_mask):
    """
    Penalize activation bên ngoài GT mask
    cam: [B, 1, H, W]
    gt_mask: [B, 1, H, W] binary {0,1}
    """
    outside_region = 1 - gt_mask  # vùng NGOÀI mask
    outside_activation = cam * outside_region
    return outside_activation.mean()

L_outside = outside_loss(cam, gt_mask)
```

### 3.4 Total Loss & Normal Class Handling
```python
def compute_loss(logits, feat_maps, classifier_weights, 
                 gt_labels, gt_masks, lambda1=1.0, lambda2=0.3):
    
    L_cls = criterion_cls(logits, gt_labels)
    
    # Chỉ tính mask loss cho non-Normal samples
    # Normal class (label=0 trong BUSI): mask rỗng → skip
    non_normal_mask = (gt_labels != NORMAL_CLASS_IDX)
    
    if non_normal_mask.sum() > 0:
        # Lấy subset non-normal
        feat_non_normal = feat_maps[non_normal_mask]
        mask_non_normal = gt_masks[non_normal_mask]
        labels_non_normal = gt_labels[non_normal_mask]
        
        # Generate CAM với GT class (không phải predicted)
        cam = generate_cam(feat_non_normal, classifier_weights, 
                          labels_non_normal, target_size=(224,224))
        
        L_align = dice_loss(cam, mask_non_normal)
        L_out = outside_loss(cam, mask_non_normal)
        
        L_total = L_cls + lambda1 * L_align + lambda2 * L_out
    else:
        L_total = L_cls
    
    return L_total, L_cls, L_align, L_out
```

- [ ] Test loss với dummy data trước khi train
- [ ] Verify gradient flow: `loss.backward()` không có None gradient

---

## 🗂️ Phase 4: Evaluation Metrics (Ngày 2)

### 4.1 Classification Metrics
- [ ] Accuracy (overall)
- [ ] F1-score macro (quan trọng vì imbalanced)
- [ ] AUC per class (one-vs-rest)
- [ ] Confusion matrix

### 4.2 XAI Metrics (Điểm Novel)
```python
def compute_xai_metrics(cam, gt_mask, threshold=0.5):
    """
    cam: [B, 1, H, W] normalized [0,1]
    gt_mask: [B, 1, H, W] binary {0,1}
    
    Returns dict với các metrics
    """
    # Binary CAM bằng threshold
    cam_binary = (cam > threshold).float()
    
    # IoU
    intersection = (cam_binary * gt_mask).sum(dim=[1,2,3])
    union = (cam_binary + gt_mask).clamp(0,1).sum(dim=[1,2,3])
    iou = (intersection / (union + 1e-8)).mean()
    
    # Dice
    dice = (2 * intersection / 
            (cam_binary.sum(dim=[1,2,3]) + gt_mask.sum(dim=[1,2,3]) + 1e-8)).mean()
    
    # Inside Ratio: bao nhiêu % activation của CAM nằm trong mask
    cam_total = cam.sum(dim=[1,2,3])
    cam_inside = (cam * gt_mask).sum(dim=[1,2,3])
    inside_ratio = (cam_inside / (cam_total + 1e-8)).mean()
    
    # Outside Ratio: bao nhiêu % activation nằm ngoài mask
    outside_ratio = 1 - inside_ratio
    
    return {
        'iou': iou.item(),
        'dice': dice.item(), 
        'inside_ratio': inside_ratio.item(),
        'outside_ratio': outside_ratio.item()
    }
```

- [ ] Compute XAI metrics **chỉ trên benign và malignant** (normal không có lesion)
- [ ] Report per-class XAI metrics: benign riêng, malignant riêng
- [ ] **Soft IoU** (không threshold) cũng report để handle blurry CAM

---

## 🗂️ Phase 5: Training (Ngày 3-5)

### 5.1 Config BUSI
```yaml
# busi_config.yaml
dataset: BUSI
num_classes: 3
classes: [normal, benign, malignant]
normal_class_idx: 0

model:
  backbone: efficientnet_b3
  pretrained: true
  dropout: 0.3

training:
  batch_size: 16
  epochs_warmup: 5      # freeze backbone
  epochs_full: 35       # unfreeze all
  lr_warmup: 1e-3       # chỉ train head
  lr_full: 1e-4         # full model
  lr_backbone_factor: 0.1  # backbone LR = lr_full * 0.1
  optimizer: AdamW
  scheduler: CosineAnnealingLR
  weight_decay: 1e-4

loss:
  lambda1: 1.0   # L_align weight
  lambda2: 0.3   # L_outside weight

eval:
  xai_threshold: 0.5
  
folds: 5
seed: 42
```

### 5.2 Training Loop
- [ ] Implement `train_one_epoch()`:
  - Forward pass → logits, feat_maps
  - Compute loss (handle normal class)
  - Backward + optimizer step
  - Log: L_total, L_cls, L_align, L_outside per batch
- [ ] Implement `validate()`:
  - Compute classification metrics
  - Compute XAI metrics trên non-normal samples
  - Save best model theo **F1-macro** (không phải accuracy vì imbalanced)
- [ ] Early stopping: patience=10 epochs

### 5.3 BUSI 5-Fold Training
- [ ] Train 4 baseline variants × 5 folds:
  - **B1:** EfficientNet-B3 + Grad-CAM (post-hoc, không có mask loss)
  - **B2:** EfficientNet-B3 + CAM intrinsic (no mask loss, λ1=λ2=0)
  - **B3:** Ours + chỉ L_align (λ1=1.0, λ2=0)
  - **B4:** Ours đầy đủ (λ1=1.0, λ2=0.3) ← **Proposed**
- [ ] Lưu results vào CSV: `results/busi_results.csv`
- [ ] Lưu best model weights per fold

**Lưu ý Grad-CAM baseline:** Grad-CAM cần backward pass riêng, không train được với mask loss. B1 là pure post-hoc: train xong mới compute Grad-CAM để evaluate XAI metrics.

### 5.4 BUS-BRA 5-Fold Training
- [ ] Dùng **cùng architecture** như BUSI, chỉ thay `num_classes=2`
- [ ] Dùng standardized 5-fold splits từ BUS-BRA dataset
- [ ] Train: B1 (Grad-CAM baseline) và B4 (Proposed)
- [ ] Lưu results: `results/busbra_results.csv`

### 5.5 Ablation — Lambda Sensitivity
- [ ] Train thêm 3 variants với λ khác nhau (trên BUSI, fold 1 only):

| λ1 | λ2 | Note |
|---|---|---|
| 0 | 0 | No mask loss |
| 1.0 | 0 | Only align |
| 0 | 0.3 | Only outside |
| 1.0 | 0.3 | **Full (proposed)** |
| 0.5 | 0.3 | λ1 sensitivity |
| 2.0 | 0.3 | λ1 sensitivity |
| 1.0 | 0.1 | λ2 sensitivity |
| 1.0 | 0.5 | λ2 sensitivity |

---

## 🗂️ Phase 6: Analysis & Figures (Ngày 6-7)

### 6.1 Qualitative Analysis — Key Figure
- [ ] Chọn 3 loại cases để visualize:
  - **Correct class + Correct heatmap** (proposed wins)
  - **Correct class + Wrong heatmap** (baseline fails — như DNBCD Fig.28)
  - **Normal class** (heatmap phải blank/suppressed)
- [ ] Plot: Original | GT Mask | Grad-CAM | CAM Intrinsic | Ours CAM
- [ ] Highlight failure case của Grad-CAM để justify contribution

### 6.2 Quantitative Tables
- [ ] **Table 1 — BUSI Classification** (mean ± std qua 5 folds):

| Method | Acc | F1-macro | AUC | IoU↑ | Dice↑ | Inside↑ | Outside↓ |
|---|---|---|---|---|---|---|---|
| EfficientNet-B3 + Grad-CAM | | | | | | | |
| EfficientNet-B3 + CAM | | | | | | | |
| Ours + L_align | | | | | | | |
| **Ours (Full)** | | | | | | | |

- [ ] **Table 2 — BUS-BRA Generalizability** (mean ± std):

| Method | Acc | F1 | AUC | IoU↑ | Dice↑ |
|---|---|---|---|---|---|
| EfficientNet-B3 + Grad-CAM | | | | | |
| **Ours (Full)** | | | | | |

- [ ] **Table 3 — Ablation** (BUSI, Fold 1):

| λ1 | λ2 | Acc | IoU | Inside | Outside |
|---|---|---|---|---|---|
| 0 | 0 | | | | |
| 1.0 | 0 | | | | |
| 0 | 0.3 | | | | |
| **1.0** | **0.3** | | | | |

### 6.3 Additional Analysis
- [ ] **Per-class XAI metrics:** Benign vs Malignant riêng — malignant thường khó hơn
- [ ] **Training curve:** Plot L_total, L_cls, L_align, L_outside theo epoch
- [ ] **CAM resolution discussion:** Note trong paper rằng CAM từ 7×7 → upsample, blurry là inherent limitation của CAM (không phải bug)

---

## 🗂️ Phase 7: Writing Paper (Ngày 8-14)

### 7.1 Structure
```
Abstract (250 words)
1. Introduction (~600 words)
   1.1 Background & Motivation
   1.2 Problem Statement: Post-hoc XAI không đảm bảo alignment
   1.3 Contributions (4 bullets)
   1.4 Paper Organization

2. Related Work (~600 words)
   2.1 Breast Ultrasound Classification (cite DNBCD, EfficientNet, ViT)
   2.2 XAI in Medical Imaging (cite Grad-CAM, "More than just a heatmap")
   2.3 CAM-based Localization (cite Zhou 2016, CAM-guided SAM)
   2.4 Gap Statement (transition sang method)

3. Method (~800 words)
   3.1 Architecture Overview (Figure 1 — block diagram)
   3.2 Intrinsic CAM từ GAP (công thức, tại sao differentiable)
   3.3 Loss Function (L_cls, L_align, L_outside với công thức)
   3.4 Normal Class Handling (tại sao quan trọng)
   3.5 Training Strategy (2-phase warmup)

4. Experiments (~400 words)
   4.1 Datasets (BUSI + BUS-BRA, statistics table)
   4.2 Implementation Details (hyperparameters, hardware)
   4.3 Evaluation Metrics (cả cls lẫn XAI)
   4.4 Baselines

5. Results (~600 words)
   5.1 BUSI Results (Table 1 + analysis)
   5.2 BUS-BRA Generalizability (Table 2)
   5.3 Ablation Study (Table 3)
   5.4 Qualitative Analysis (Figure 2 — heatmap comparison)

6. Discussion (~400 words)
   6.1 Accuracy ≠ Trustworthy Explanation (key message)
   6.2 Normal Class Behavior
   6.3 CAM Resolution Limitation
   6.4 Clinical Implications

7. Conclusion (~150 words)

References
```

### 7.2 Writing Checklist
- [ ] Abstract: mention method + 2 datasets + key metrics (XAI + cls)
- [ ] Introduction: cite DNBCD Fig.28 như motivation rõ ràng
- [ ] Method Figure 1: block diagram rõ ràng, có mũi tên gradient flow
- [ ] Equation numbering: đánh số tất cả công thức
- [ ] Figure 2: qualitative comparison — đây là figure compelling nhất
- [ ] Kết quả phải report mean ± std qua 5 folds (không chỉ best fold)
- [ ] Statistical test: paired t-test giữa proposed và best baseline
- [ ] Limitations section: mention CAM resolution, no decoder
- [ ] Code availability statement: "Code available at [GitHub URL]"

---

## 🗂️ Phase 8: Submission Prep (Ngày 14-15)

### 8.1 Journal: Computers in Biology and Medicine
- [ ] Check author guidelines: https://www.sciencedirect.com/journal/computers-in-biology-and-medicine
- [ ] Word limit: ~8000 words (không tính references)
- [ ] Figure format: 300 DPI, TIFF hoặc EPS
- [ ] Reference style: numbered [1], [2]...
- [ ] Cover letter: 1 trang, highlight gap + contribution + fit với journal

### 8.2 Code Release
- [ ] Upload code lên GitHub với README
- [ ] Include: training script, evaluation script, pretrained weights (best fold)
- [ ] Requirements.txt với version cụ thể
- [ ] Instructions để reproduce results

### 8.3 Final Checks
- [ ] Proofread toàn bộ paper (đặc biệt method section)
- [ ] Check tất cả figure labels và table captions
- [ ] Verify tất cả citations có trong reference list
- [ ] Check supplementary (nếu có): full results per fold

---

## ⚠️ Risk Register & Mitigation

| Rủi ro | Khả năng | Impact | Cách xử lý |
|---|---|---|---|
| Proposed accuracy thấp hơn baseline | Trung bình | Cao | Frame: trade-off nhỏ về accuracy, gain lớn về XAI quality |
| CAM blurry, IoU thấp | Cao | Trung bình | Dùng soft IoU; note là inherent CAM limitation; vẫn tốt hơn Grad-CAM |
| Reviewer yêu cầu decoder/SAM | Trung bình | Thấp | "Mục tiêu không phải best segmentation mà là trustworthy classification" |
| BUS-BRA kết quả kém | Thấp | Cao | Nếu kém hơn BUSI, frame là "domain gap challenge, future work" |
| L_align không converge | Thấp | Cao | Giảm λ1; thêm warmup epochs |

---

## 📊 Expected Results Range

Dựa trên literature, kết quả hợp lý cho proposed method:

| Metric | BUSI Expected | BUS-BRA Expected |
|---|---|---|
| Accuracy | 88–92% | 86–90% |
| F1-macro | 85–90% | 84–89% |
| AUC | 0.95–0.98 | 0.93–0.96 |
| IoU (vs Grad-CAM) | +5–15% | +4–12% |
| Inside Ratio | 0.65–0.80 | 0.60–0.75 |
| Outside Ratio | 0.20–0.35 | 0.25–0.40 |

Grad-CAM baseline expected IoU: ~0.30–0.45 trên BUSI (theo literature).

---

## 📅 Master Timeline

```
Week 1 (Ngày 1-7):
  Ngày 0:   Phase 0 — Setup, download datasets
  Ngày 1:   Phase 1 — Data pipeline hoàn chỉnh
  Ngày 2:   Phase 2+3 — Model + Loss + Metrics
  Ngày 3-5: Phase 5.3 — Train BUSI 5-fold (4 baselines × 5 folds = 20 runs)
  Ngày 5-6: Phase 5.4 — Train BUS-BRA 5-fold (2 baselines × 5 folds = 10 runs)
  Ngày 6-7: Phase 5.5 + Phase 6 — Ablation + Analysis + Figures

Week 2 (Ngày 8-14):
  Ngày 8-12: Phase 7 — Writing (2 sections/day)
  Ngày 13:   Phase 7 — Revision + polish
  Ngày 14:   Phase 7 — Final proofread

Week 3 (Ngày 15):
  Ngày 15:   Phase 8 — Submission prep + upload
```

---

## 🔑 Key Technical Decisions Summary

| Decision | Choice | Lý do |
|---|---|---|
| Backbone | EfficientNet-B3 | Balance performance/size cho ~780-1875 ảnh |
| CAM type | Intrinsic GAP-based | Differentiable, feedback vào training |
| L_align | Dice Loss | Robust hơn BCE với imbalanced mask |
| L_outside | Mean activation outside | Simple, interpretable, effective |
| λ1 | 1.0 | Bằng với L_cls, không dominate |
| λ2 | 0.3 | Nhỏ hơn để không penalize quá mạnh |
| Training | 2-phase warmup | Tránh destroy pretrained features |
| BUSI splits | 5-fold stratified | Không có sẵn → tự tạo |
| BUS-BRA splits | Standardized 5-fold | Reproducibility |
| Normal class | Skip mask loss | Không có lesion để align |
| Target venue | Computers in Bio & Med | Scope khớp, IF ~7, Q1 Scopus |

---

*Last updated: Plan version 1.0*  
*Status: Ready to implement*
