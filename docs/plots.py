"""
docs/plots.py — BLACKWALL benchmark figures (roofline + precision ladder).

Data are the MEASURED numbers from the cuBLAS/cuBLASLt GEMM sweep (N=8192) on the
RTX 5060 Ti (sm_120) — the same values tabulated in the README. This script only
*visualizes* them (no re-run needed; the GEMM timings are in the README table, the
~381 GB/s bandwidth ceiling comes from the sister project ICEPICK).

The roofline ridge points are derived here as peak/bandwidth and match the README
to the digit (FP32 45, FP16 231, FP8 485, FP4 897) — a self-consistency check.

Run:  python docs/plots.py        Output: docs/*.png (committed deliverables).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
BG, PANEL = "#0a0c12", "#0e121c"
CYAN, MAGENTA, AMBER, LIME = "#00e6c8", "#ff46a0", "#ffb446", "#9dff5a"
TEXT, MUTED, GRID = "#c8d6e0", "#7a8796", "#1b2434"
FOOTER = "measured on RTX 5060 Ti (sm_120), CUDA 13 · cuBLAS/cuBLASLt GEMM N=8192 · BLACKWALL"

# ---- measured (README table) ----
BW = 0.381  # TB/s sustained (ICEPICK F2b)
CEIL = [("FP32", 17.1, CYAN), ("FP16", 87.9, AMBER), ("FP8", 184.6, MAGENTA), ("FP4", 341.9, LIME)]
LADDER = [("FP32", 17.1, "1.0×"), ("TF32", 23.8, "1.4×"), ("BF16/FP16\nFP32-acc", 47.0, "2.75×"),
          ("FP16\nFP16-acc", 87.9, "5.1×"), ("FP8\ne4m3", 184.6, "10.8×"), ("FP4\nnvfp4", 341.9, "20.0×")]
LCOL = [CYAN, "#3fd0e0", "#7ec8a0", AMBER, MAGENTA, LIME]


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
        "axes.grid": True, "grid.alpha": 0.5, "grid.linewidth": 0.7,
        "axes.titlecolor": CYAN, "axes.titlesize": 13, "axes.titleweight": "bold",
        "font.size": 11, "font.family": "DejaVu Sans Mono", "figure.dpi": 140,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _footer(fig):
    fig.text(0.99, 0.01, FOOTER, ha="right", va="bottom", color=MUTED, fontsize=7.5, style="italic")


def roofline():
    ai = np.logspace(0, 3.4, 400)               # arithmetic intensity, FLOP/byte
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.plot(ai, BW * ai, color=MUTED, lw=1.6, ls="--", label=f"memory roof ({BW*1000:.0f} GB/s)")
    for name, peak, col in CEIL:
        ridge = peak / BW
        ax.hlines(peak, ridge, ai[-1], color=col, lw=2.4)
        ax.plot(ai[ai <= ridge], BW * ai[ai <= ridge], color=col, lw=2.4)
        ax.scatter([ridge], [peak], s=55, color=col, zorder=5, edgecolor="white", linewidth=0.6)
        ax.annotate(f"{name}  {peak:.0f} TFLOP/s\nridge {ridge:.0f} FLOP/byte",
                    (ridge, peak), textcoords="offset points", xytext=(8, -22 if name != "FP4" else 6),
                    color=col, fontsize=8.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity  (FLOP / byte)")
    ax.set_ylabel("attainable throughput  (TFLOP/s)")
    ax.set_title("BLACKWALL roofline — consumer Blackwell across the precision spectrum")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5, loc="lower right")
    ax.text(0.02, 0.97, "lower precision → ridge moves right →\nFP4 needs ~900 FLOP/byte: cache reuse is mandatory",
            transform=ax.transAxes, va="top", color=TEXT, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=GRID, alpha=0.9))
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _footer(fig)
    fig.savefig(OUT / "roofline.png", bbox_inches="tight"); plt.close(fig)
    print("  wrote docs/roofline.png")


def precision_ladder():
    names = [l[0] for l in LADDER]; vals = [l[1] for l in LADDER]; spd = [l[2] for l in LADDER]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(range(len(names)), vals, color=LCOL, edgecolor=BG, linewidth=1.5, zorder=3)
    for i, (b, s) in enumerate(zip(bars, spd)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{vals[i]:.1f}\n{s}",
                ha="center", va="bottom", color=TEXT, fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("GEMM throughput  (TFLOP/s)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Each precision halving ≈ doubles throughput — a clean 20× ladder to FP4")
    ax.grid(axis="x", alpha=0)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _footer(fig)
    fig.savefig(OUT / "precision_ladder.png", bbox_inches="tight"); plt.close(fig)
    print("  wrote docs/precision_ladder.png")


if __name__ == "__main__":
    _style()
    print("Generating BLACKWALL figures (measured data)...")
    roofline()
    precision_ladder()
    print("Done ->", OUT)
