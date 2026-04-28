"""
Post-processing tools for a training run.

Commands
--------
  # Training curves (loss, F1, accuracy, pointing-game per epoch)
  python plot_history.py curves --run_dir runs/<run_name>
  python plot_history.py curves --run_dir runs/<run_name> --no_folds

  # Paired t-test: baseline vs proposed on XAI + classification metrics
  python plot_history.py stats  --run_dir runs/<run_name>
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


METRICS = [
    ('train_loss',    'Train Loss',         False),
    ('f1_macro',      'Val F1 macro',       True),
    ('accuracy',      'Val Accuracy',       True),
    ('pointing_game', 'Val Pointing Game',  True),
]

COLORS = {
    'baseline': '#2196F3',
    'proposed': '#F44336',
}


def load_histories(run_dir, variant):
    """Return list of per-epoch dicts, one list per fold."""
    histories = []
    fold = 0
    while True:
        path = os.path.join(run_dir, f'{variant}_fold{fold}_history.json')
        if not os.path.exists(path):
            break
        with open(path, 'r', encoding='utf-8') as f:
            histories.append(json.load(f))
        fold += 1
    return histories


def _to_array(histories, key):
    """
    Extract metric across folds, padding shorter folds with NaN.
    Returns (n_folds, max_epochs) array.
    """
    rows = []
    for h in histories:
        vals = [ep.get(key) for ep in h]
        rows.append(vals)
    max_len = max(len(r) for r in rows)
    arr = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if v is not None:
                arr[i, j] = v
    return arr


def plot_metric(ax, histories, key, label, color, show_folds=True):
    arr = _to_array(histories, key)
    if np.all(np.isnan(arr)):
        ax.text(0.5, 0.5, f'{key}\n(no data)', transform=ax.transAxes,
                ha='center', va='center', color='grey')
        return

    epochs = np.arange(1, arr.shape[1] + 1)
    mean   = np.nanmean(arr, axis=0)
    std    = np.nanstd(arr, axis=0)

    if show_folds:
        for i, row in enumerate(arr):
            ax.plot(epochs, row, alpha=0.25, color=color, linewidth=0.8)

    ax.plot(epochs, mean, color=color, linewidth=2.0, label=label)
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.15, color=color)


def make_figure(run_dir, variants, show_folds):
    n_rows = len(METRICS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]

    loaded = {}
    for v in variants:
        h = load_histories(run_dir, v)
        if not h:
            print(f"  [WARN] No history files found for variant '{v}' in {run_dir}")
        else:
            print(f"  Loaded {len(h)} fold(s) for variant '{v}'")
        loaded[v] = h

    for ax, (key, title, _) in zip(axes, METRICS):
        for v in variants:
            if not loaded[v]:
                continue
            color = COLORS.get(v, '#333333')
            plot_metric(ax, loaded[v], key, label=v, color=color,
                        show_folds=show_folds)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Paired t-test
# ─────────────────────────────────────────────

XAI_METRICS = ['pointing_game', 'soft_iou', 'inside_ratio', 'auprc']
CLS_METRICS  = ['accuracy', 'f1_macro', 'auc']


def _load_fold_results(run_dir, variant):
    path = os.path.join(run_dir, f'{variant}_fold_results.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_stats(run_dir):
    print(f"\nRun dir : {run_dir}")

    base  = _load_fold_results(run_dir, 'baseline')
    prop  = _load_fold_results(run_dir, 'proposed')
    n     = min(len(base), len(prop))

    if n < 2:
        print("  Need at least 2 folds for a paired t-test.")
        return

    print(f"  Folds   : {n}\n")
    print(f"  {'Metric':<20} {'Base mean':>10} {'Prop mean':>10} {'Diff':>8} {'p-value':>10}  Sig?")
    print("  " + "-" * 68)

    for key in XAI_METRICS + CLS_METRICS:
        b_vals = [base[i][key] for i in range(n) if base[i].get(key) is not None]
        p_vals = [prop[i][key] for i in range(n) if prop[i].get(key) is not None]

        paired_n = min(len(b_vals), len(p_vals))
        if paired_n < 2:
            print(f"  {key:<20} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>10}")
            continue

        b_arr = np.array(b_vals[:paired_n])
        p_arr = np.array(p_vals[:paired_n])

        _, pval = stats.ttest_rel(b_arr, p_arr)
        diff = p_arr.mean() - b_arr.mean()
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = ''
        section = '[XAI] ' if key in XAI_METRICS else '[CLS] '

        print(f"  {section+key:<20} {b_arr.mean():>10.4f} {p_arr.mean():>10.4f}"
              f" {diff:>+8.4f} {pval:>10.4f}  {sig}")

    print("\n  Significance: * p<0.05   ** p<0.01   *** p<0.001")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    if args.command == 'curves':
        out_dir = os.path.join(args.run_dir, 'plots')
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nRun dir : {args.run_dir}")
        fig = make_figure(args.run_dir, args.variants, show_folds=not args.no_folds)
        out_path = os.path.join(out_dir, 'training_curves.png')
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved   : {out_path}")

    elif args.command == 'stats':
        run_stats(args.run_dir)


def parse_args():
    p = argparse.ArgumentParser(description='Post-processing tools for a training run')
    sub = p.add_subparsers(dest='command', required=True)

    # curves sub-command
    c = sub.add_parser('curves', help='Plot training curves')
    c.add_argument('--run_dir', required=True)
    c.add_argument('--variants', nargs='+', default=['baseline', 'proposed'])
    c.add_argument('--no_folds', action='store_true',
                   help='Show only mean ± std band, hide individual folds')

    # stats sub-command
    s = sub.add_parser('stats', help='Paired t-test: baseline vs proposed')
    s.add_argument('--run_dir', required=True)

    return p.parse_args()


if __name__ == '__main__':
    main()
