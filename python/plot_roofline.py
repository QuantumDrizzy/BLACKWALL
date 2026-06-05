"""
BLACKWALL — roofline figure (THE TRACE).

Plots the measured precision spectrum on Blackwell sm_120 (RTX 5060 Ti), N=8192,
from the cuBLAS/cuBLASLt GEMM bench (src/gemm_bench.cu). Numbers are the measured
run; the bench is the source of truth.

    python python/plot_roofline.py   ->   docs/roofline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Measured TFLOP/s @ N=8192 (RTX 5060 Ti, sm_120, ~2.57 GHz; FP32 CUDA-core peak 23.7)
LABELS  = ["FP32", "TF32", "BF16/FP16\nf32-acc", "FP16\nf16-acc", "FP8\ne4m3", "FP4\nnvfp4"]
TFLOPS  = [17.1, 23.8, 47.0, 87.9, 184.6, 341.9]
SPEEDUP = [1.0, 1.39, 2.75, 5.15, 10.82, 20.04]
FP32_PEAK = 23.7

# netrunner palette: shallow (cool) -> deep (hot)
COLORS = ["#3a3f5a", "#2f6f8f", "#22b3a6", "#b8c24a", "#ff7a18", "#ff2d55"]

plt.rcParams.update({
    "figure.facecolor": "#0a0a0f", "axes.facecolor": "#0a0a0f",
    "axes.edgecolor": "#39ff9c", "axes.labelcolor": "#d7faff",
    "xtick.color": "#d7faff", "ytick.color": "#d7faff", "text.color": "#d7faff",
    "font.family": "DejaVu Sans Mono",
})

fig, ax = plt.subplots(figsize=(12, 6.5))
x = range(len(LABELS))
bars = ax.bar(x, TFLOPS, color=COLORS, edgecolor="#0a0a0f", width=0.72, zorder=3)

# FP32 CUDA-core peak reference
ax.axhline(FP32_PEAK, ls="--", lw=1.0, color="#39ff9c", alpha=0.55, zorder=2)
ax.text(len(LABELS) - 0.5, FP32_PEAK + 6, "FP32 CUDA-core peak 23.7", color="#39ff9c",
        fontsize=8, ha="right", alpha=0.8)

for i, (b, tf, sp) in enumerate(zip(bars, TFLOPS, SPEEDUP)):
    ax.text(b.get_x() + b.get_width() / 2, tf + 6, f"{tf:.0f}", ha="center",
            fontsize=11, fontweight="bold", color="#ffffff", zorder=4)
    ax.text(b.get_x() + b.get_width() / 2, tf / 2,
            ("baseline" if sp == 1.0 else f"{sp:.1f}x"), ha="center", va="center",
            fontsize=10, fontweight="bold", color="#0a0a0f", zorder=4)

ax.set_xticks(list(x)); ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylabel("dense GEMM throughput  (TFLOP/s)", fontsize=11)
ax.set_ylim(0, 380)
ax.set_title("BLACKWALL  —  Blackwell sm_120 precision spectrum  (honest GEMM roofline)",
             fontsize=13, fontweight="bold", color="#39ff9c", pad=14)
ax.text(0.0, -0.16, "RTX 5060 Ti · CUDA 13 · cuBLAS/cuBLASLt · N=8192 · "
        "CUDA-event timed, warmup, medians · the dive: FP32 -> FP4 = 20x · throughput-only",
        transform=ax.transAxes, fontsize=8, color="#7fa6b5")
ax.grid(axis="y", ls=":", alpha=0.18, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

out = Path(__file__).resolve().parents[1] / "docs" / "roofline.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
print("saved", out)
