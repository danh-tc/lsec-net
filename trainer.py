import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import BUSIDataset, get_transforms, XAI_ACC_THRESHOLD
from models.lsec_net import LSECNet
from losses.losses import LSECLoss
from metrics.metrics import evaluate_model, aggregate_results
from sklearn.metrics import f1_score

BACKBONE_LR = 2e-5   # ConvNeXt-Tiny fine-tune LR after warmup
HEAD_LR     = 5e-5   # head LR after warmup


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    return value


def _collect_logits(model, loader, device):
    model.eval()
    labels, logits = [], []
    with torch.no_grad():
        for batch in loader:
            out, _ = model(batch['image'].to(device))
            logits.append(out.cpu().numpy())
            labels.extend(batch['label'].numpy().tolist())
    return np.array(labels), np.concatenate(logits, axis=0)


def calibrate_logit_bias(model, loader, device, search_min=-1.5, search_max=1.5, search_step=0.1):
    labels, logits = _collect_logits(model, loader, device)
    values = np.arange(search_min, search_max + 1e-9, search_step)
    best = {'score': -1.0, 'accuracy': 0.0, 'bias': [0.0, 0.0, 0.0]}

    for b0 in values:
        for b1 in values:
            for b2 in values:
                bias = np.array([b0, b1, b2])
                preds = (logits + bias).argmax(axis=1)
                score = f1_score(labels, preds, average='macro', zero_division=0)
                accuracy = float((preds == labels).mean())
                if score > best['score']:
                    best = {
                        'score': float(score),
                        'accuracy': accuracy,
                        'bias': [float(x) for x in bias],
                    }

    print(
        "  Calibration: "
        f"val_f1={best['score']:.4f} | val_acc={best['accuracy']:.4f} | "
        f"bias={best['bias']}"
    )
    return best


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_balanced_sampler(labels, generator=None):
    counts = np.bincount(labels, minlength=3).astype(float)
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


def run_fold(model, criterion, train_loader, val_loader,
             epochs, device, fold_idx, use_mask_loss=True,
             mixup_prob=0.5, mixup_alpha=0.2,
             warmup_epochs=3, backbone_lr=BACKBONE_LR,
             head_lr=HEAD_LR, warmup_lr=2e-4,
             debug=False):
    """
    2-phase training for a single fold:
      Phase 1 (warmup): backbone frozen, head only at lr=1e-3
      Phase 2:          backbone unfrozen via add_param_group (preserves head
                        momentum), backbone lr=5e-5, head lr=1e-4, cosine decay

    Mixup (alpha=0.2) is applied only for the baseline variant (no mask loss)
    because mixed images have no valid corresponding GT mask for alignment.

    Returns (best_state_dict, best_val_f1_macro).
    """
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.gap.parameters()) +
                       list(model.dropout.parameters()) +
                       list(model.classifier.parameters()))

    WARMUP = 1 if debug else warmup_epochs
    use_mixup = (not use_mask_loss) and mixup_prob > 0 and mixup_alpha > 0

    # Phase 1: freeze backbone, train head only
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=warmup_lr, weight_decay=1e-3)
    scheduler = None

    best_f1, best_state, patience = 0.0, None, 0
    history = []

    for epoch in range(epochs):

        # ── Switch to phase 2 ────────────────────────────────────
        if epoch == WARMUP:
            for p in backbone_params:
                p.requires_grad = True
            # add_param_group preserves head Adam momentum accumulated so far
            optimizer.add_param_group(
                {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 1e-3}
            )
            optimizer.param_groups[0]['lr'] = head_lr
            if epochs > WARMUP:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs - WARMUP)
            patience = 0  # give phase 2 a fresh patience budget

        # ── Train epoch ──────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img    = batch['image'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            if use_mixup and np.random.random() < mixup_prob:
                lam   = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx   = torch.randperm(img.size(0), device=device)
                mixed = lam * img + (1 - lam) * img[idx]
                lab_b = labels[idx]

                logits, feat   = model(mixed)
                loss_a, parts_a = criterion(logits, feat, model, labels,  mask)
                loss_b, parts_b = criterion(logits, feat, model, lab_b,   mask)
                loss       = lam * loss_a + (1 - lam) * loss_b
                loss_parts = {k: lam * parts_a[k] + (1 - lam) * parts_b[k]
                              for k in parts_a}
            else:
                logits, feat     = model(img)
                loss, loss_parts = criterion(logits, feat, model, labels, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if debug:
                print(f"    [debug] batch {batch_idx:3d} | "
                      f"loss={loss.item():.4f}  "
                      f"cls={loss_parts['cls']:.4f}  "
                      f"align={loss_parts['align']:.4f}  "
                      f"out={loss_parts['out']:.4f}")

        if scheduler and epoch >= WARMUP:
            scheduler.step()

        # ── Validation ───────────────────────────────────────────
        val_res = evaluate_model(
            f'fold{fold_idx}_ep{epoch}', model, val_loader,
            device, save_cm=False)

        pg_str = f"{val_res['pointing_game']:.4f}" if val_res['pointing_game'] is not None else 'N/A'
        print(f"  Epoch {epoch:3d} | "
              f"loss={epoch_loss / len(train_loader):.4f} | "
              f"val_acc={val_res['accuracy']:.4f} | "
              f"val_f1={val_res['f1_macro']:.4f} | "
              f"PG={pg_str}")
        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_loader),
            **val_res,
        })

        # ── Early stopping ───────────────────────────────────────
        if val_res['f1_macro'] > best_f1:
            best_f1    = val_res['f1_macro']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if patience >= 10:
            print(f"  Early stop at epoch {epoch}")
            break

    return best_state, best_f1, history


def train_and_evaluate(train_val, test_set, folds,
                       use_mask_loss, n_folds_to_run,
                       epochs, batch_size, device,
                       variant_name, tta=False, output_dir=None,
                       aug='default', label_smoothing=0.1,
                       mixup_prob=0.5, mixup_alpha=0.2,
                       class_weight_mode='inverse', dropout=0.3,
                       warmup_epochs=3, backbone_lr=BACKBONE_LR,
                       head_lr=HEAD_LR, warmup_lr=2e-4,
                       calibrate_logits=False,
                       sampler='shuffle',
                       mask_lambda1=1.0, mask_lambda2=0.3,
                       seed=42,
                       backbone_weights=None,
                       debug=False):
    """
    Runs n_folds_to_run folds for one variant (baseline or proposed).
    Prints a warning after fold 0 if test accuracy < XAI_ACC_THRESHOLD.
    Returns aggregated results dict (mean ± std across folds).
    """
    fold_test_results = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(folds[:n_folds_to_run]):
        print(f"\n{'='*50}")
        print(f"  {variant_name.upper()} — Fold {fold_idx} / {n_folds_to_run - 1}")
        print(f"{'='*50}")

        train_data = [train_val[i] for i in train_idx]
        val_data   = [train_val[i] for i in val_idx]

        train_labels  = [x[2] for x in train_data]
        class_weights = None
        if class_weight_mode == 'inverse':
            class_weights = compute_class_weights(train_labels, 3).to(device)

        model     = LSECNet(num_classes=3, dropout=dropout, weights_path=backbone_weights).to(device)
        criterion = LSECLoss(
            lambda1=mask_lambda1 if use_mask_loss else 0.0,
            lambda2=mask_lambda2 if use_mask_loss else 0.0,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
        )

        generator = torch.Generator()
        generator.manual_seed(seed + fold_idx)
        train_sampler = None
        shuffle_train = True
        if sampler == 'balanced':
            train_sampler = make_balanced_sampler(train_labels, generator=generator)
            shuffle_train = False

        train_loader = DataLoader(
            BUSIDataset(train_data, get_transforms('train', aug=aug)),
            batch_size=batch_size, shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=4, pin_memory=True,
            worker_init_fn=_seed_worker, generator=generator)
        val_loader = DataLoader(
            BUSIDataset(val_data, get_transforms('val')),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            worker_init_fn=_seed_worker)
        test_loader = DataLoader(
            BUSIDataset(test_set, get_transforms('val')),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
            worker_init_fn=_seed_worker)

        best_state, _, history = run_fold(
            model, criterion, train_loader, val_loader,
            epochs, device, fold_idx, use_mask_loss=use_mask_loss,
            mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
            warmup_epochs=warmup_epochs, backbone_lr=backbone_lr,
            head_lr=head_lr, warmup_lr=warmup_lr, debug=debug)

        if output_dir:
            history_path = os.path.join(output_dir, f'{variant_name}_fold{fold_idx}_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(_json_safe(history), f, indent=2)

        # Evaluate on locked test set
        model.load_state_dict(best_state)
        calibration = None
        logit_bias = None
        if calibrate_logits:
            calibration = calibrate_logit_bias(model, val_loader, device)
            logit_bias = calibration['bias']
            if output_dir:
                calibration_path = os.path.join(output_dir, f'{variant_name}_fold{fold_idx}_calibration.json')
                with open(calibration_path, 'w', encoding='utf-8') as f:
                    json.dump(_json_safe(calibration), f, indent=2)

        print(f"\n  [Test set — Fold {fold_idx}]{'  (TTA)' if tta else ''}")
        test_res = evaluate_model(
            f'{variant_name}_fold{fold_idx}', model, test_loader,
            device, save_cm=True, tta=tta, output_dir=output_dir,
            logit_bias=logit_bias)
        if calibration is not None:
            test_res['logit_bias'] = calibration['bias']
            test_res['calibration_val_f1'] = calibration['score']
            test_res['calibration_val_acc'] = calibration['accuracy']

        fold_test_results.append(test_res)

        ckpt_path = f'{variant_name}_fold{fold_idx}.pth'
        if output_dir:
            ckpt_path = os.path.join(output_dir, ckpt_path)
        torch.save(best_state, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

        if output_dir:
            results_path = os.path.join(output_dir, f'{variant_name}_fold_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(_json_safe(fold_test_results), f, indent=2)

        # ── Early-signal gate after fold 0 ───────────────────────
        if fold_idx == 0:
            acc = test_res['accuracy']
            pg  = test_res['pointing_game']
            if acc < XAI_ACC_THRESHOLD:
                print(f"\n  {'!'*56}")
                print(f"  [WARNING] Fold 0 accuracy = {acc:.4f} < {XAI_ACC_THRESHOLD}")
                print("  XAI metrics skipped. Consider debugging before running more folds.")
                print(f"  {'!'*56}\n")
            else:
                pg_str = f'{pg:.4f}' if pg is not None else 'N/A'
                print(f"\n  [Fold 0 OK] acc={acc:.4f} | Pointing Game={pg_str}")
                if n_folds_to_run > 1:
                    print(f"  Proceeding with folds 1–{n_folds_to_run - 1} …\n")

    print(f"\n{'='*50}")
    print(f"  {variant_name.upper()} — Aggregated ({n_folds_to_run} folds)")
    print(f"{'='*50}")
    aggregate = aggregate_results(fold_test_results)
    if output_dir:
        aggregate_path = os.path.join(output_dir, f'{variant_name}_aggregate_results.json')
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(aggregate), f, indent=2)
    return aggregate
