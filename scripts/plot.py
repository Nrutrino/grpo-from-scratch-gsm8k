"""Plot training curves from logs/train_log.csv and logs/eval_log.csv.

    python scripts/plot.py --log_dir logs --out logs/curves.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv


def read(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def smooth(y, w=20):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tr = read(os.path.join(a.log_dir, "train_log.csv"))
    ev = read(os.path.join(a.log_dir, "eval_log.csv"))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("acc", "Rollout accuracy (train, sampled T=1)"), ("kl", "KL to reference"),
              ("len", "Completion length (tokens)")]
    for axis, (key, title) in zip(ax.flat[:3], panels):
        axis.plot(tr["step"], tr[key], alpha=0.25, color="C0")
        s = smooth(tr[key])
        axis.plot(tr["step"][len(tr["step"]) - len(s):], s, color="C0")
        axis.set_title(title); axis.set_xlabel("step")
    if ev:
        # Two different sample sizes share this log: the cheap mid-run evals (small n, noisy)
        # and the full-test-split evals at step 0 / final.  Plot them differently so the
        # noisy points are not read as a trend of the same precision.
        full = ev["n"] >= ev["n"].max()
        a_ = ax[1, 1]
        a_.plot(ev["step"][~full], ev["test_acc"][~full] * 100, marker="o", mfc="none", color="C0",
                alpha=0.7, label=f"periodic eval (n={int(ev['n'][~full].min()) if (~full).any() else 0}, noisy)")
        a_.scatter(ev["step"][full], ev["test_acc"][full] * 100, marker="*", s=160, color="C3", zorder=3,
                   label=f"full test split (n={int(ev['n'].max())})")
        for x, y in zip(ev["step"][full], ev["test_acc"][full] * 100):
            a_.annotate(f"{y:.1f}%", (x, y), fontsize=8, textcoords="offset points", xytext=(6, -12))
        a_.set_title("GSM8K test accuracy (greedy, %)"); a_.set_xlabel("step"); a_.legend(fontsize=7)
    fig.tight_layout()
    out = a.out or os.path.join(a.log_dir, "curves.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
