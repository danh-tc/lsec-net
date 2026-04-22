import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import BUSIDataset, get_transforms, XAI_ACC_THRESHOLD
from models.lsec_net import LSECNet
from losses.losses import LSECLoss
from metrics.metrics import evaluate_model, aggregate_results


def compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    return torch.tensor(len(labels) / (num_classes * counts), dtype=torch.float32)


def run_fold(model, criterion, train_loader, val_loader,
             epochs, device, fold_idx, debug=False):
    """
    2-phase training for a single fold:
      Phase 1 (warmup epochs): backbone frozen, head only
      Phase 2 (remaining):     full fine-tune with cosine LR decay

    Returns (best_state_dict, best_val_f1_macro).
    """
    backbone_params = list(model.backbone.parameters())
    head_params     = (list(model.gap.parameters()) +
                       list(model.dropout.parameters()) +
                       list(model.classifier.parameters()))

    WARMUP = 5 if not debug else 1

    # Phase 1: freeze backbone
    for p in backbone_params:
        p.requires_grad = False
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scheduler = None

    best_f1, best_state, patience = 0.0, None, 0

    for epoch in range(epochs):

        # Switch to phase 2
        if epoch == WARMUP:
            for p in backbone_params:
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5},
                {'params': head_params,     'lr': 1e-4},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - WARMUP)

        # ── Train epoch ──────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            img    = batch['image'].to(device)
            mask   = batch['mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, feat      = model(img)
            loss, loss_parts  = criterion(logits, feat, model, labels, mask)
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

    return best_state, best_f1


def train_and_evaluate(train_val, test_set, folds,
                       use_mask_loss, n_folds_to_run,
                       epochs, batch_size, device,
                       variant_name, debug=False):
    """
    Runs n_folds_to_run folds for one variant (baseline or proposed).
    Prints a warning after fold 0 if test accuracy < XAI_ACC_THRESHOLD.
    Returns aggregated results dict (mean ± std across folds).
    """
    fold_test_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds[:n_folds_to_run]):
        print(f"\n{'='*50}")
        print(f"  {variant_name.upper()} — Fold {fold_idx} / {n_folds_to_run - 1}")
        print(f"{'='*50}")

        train_data = [train_val[i] for i in train_idx]
        val_data   = [train_val[i] for i in val_idx]

        train_labels  = [x[2] for x in train_data]
        class_weights = compute_class_weights(train_labels, 3).to(device)

        model     = LSECNet(num_classes=3).to(device)
        criterion = LSECLoss(
            lambda1=1.0 if use_mask_loss else 0.0,
            lambda2=0.3 if use_mask_loss else 0.0,
            class_weights=class_weights,
        )

        train_loader = DataLoader(
            BUSIDataset(train_data, get_transforms('train')),
            batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
        val_loader = DataLoader(
            BUSIDataset(val_data, get_transforms('val')),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
        test_loader = DataLoader(
            BUSIDataset(test_set, get_transforms('val')),
            batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

        best_state, _ = run_fold(
            model, criterion, train_loader, val_loader,
            epochs, device, fold_idx, debug=debug)

        # Evaluate on locked test set
        model.load_state_dict(best_state)
        print(f"\n  [Test set — Fold {fold_idx}]")
        test_res = evaluate_model(
            f'{variant_name}_fold{fold_idx}', model, test_loader,
            device, save_cm=True)

        fold_test_results.append(test_res)

        ckpt_path = f'{variant_name}_fold{fold_idx}.pth'
        torch.save(best_state, ckpt_path)
        print(f"  Checkpoint: {ckpt_path}")

        # ── Early-signal gate after fold 0 ───────────────────────
        if fold_idx == 0:
            acc = test_res['accuracy']
            pg  = test_res['pointing_game']
            if acc < XAI_ACC_THRESHOLD:
                print(f"\n  {'!'*56}")
                print(f"  [WARNING] Fold 0 accuracy = {acc:.4f} < {XAI_ACC_THRESHOLD}")
                print(f"  XAI metrics skipped. Consider debugging before running more folds.")
                print(f"  {'!'*56}\n")
            else:
                pg_str = f'{pg:.4f}' if pg is not None else 'N/A'
                print(f"\n  [Fold 0 OK] acc={acc:.4f} | Pointing Game={pg_str}")
                if n_folds_to_run > 1:
                    print(f"  Proceeding with folds 1–{n_folds_to_run - 1} …\n")

    print(f"\n{'='*50}")
    print(f"  {variant_name.upper()} — Aggregated ({n_folds_to_run} folds)")
    print(f"{'='*50}")
    return aggregate_results(fold_test_results)
