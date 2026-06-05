# BLACKWALL

> *The netrunners run the Blackwall. I run **with** the Blackwell.*

Honest roofline characterization of the **NVIDIA Blackwell** tensor cores
(consumer, **sm_120**, RTX 5060 Ti) across the **precision spectrum** —
FP32 → TF32 → BF16 → FP16 → FP8 → FP4. Measured on the metal. No inflated numbers.

The dive: lower precision = deeper into the Net = faster, more dangerous (lossier).
This repo measures exactly *how much* faster, at *what* depth — and tells the truth
about the trade.

---

## The trace so far — `:: BREACH` (cuBLAS GEMM, N = 8192)

GPU: **RTX 5060 Ti** · sm_120 · 36 SMs · ~2.57 GHz · FP32 CUDA-core peak **23.7 TFLOP/s**

| precision | TFLOP/s | vs FP32 |
|-----------|--------:|--------:|
| FP32 (CUDA cores)            |  17.0 | 1.00× — **72% of peak** |
| TF32                         |  23.2 | 1.36× |
| BF16 / FP16 (FP32 accumulate)|  46.4 | 2.73× |
| FP16 (FP16 accumulate)       |  85.7 | 5.03× |
| **FP8 e4m3 → bf16**          | **182.1** | **10.64×** |

**What the trace says — honestly:**
- FP32 SGEMM sits at **72% of the computed CUDA-core peak** — a healthy anchor. The
  measurement is real, not cherry-picked.
- **FP32 accumulation costs ~2× on the tensor cores.** The "safe-for-training" path
  (BF16/FP16 + FP32 accumulate) is ~46 TFLOP/s; dropping to FP16 accumulate nearly
  doubles it to ~86. Speed vs numerical headroom — that trade *is* mixed precision.
- **FP8 reaches 182 TFLOP/s (10.6× FP32)**, ≈ 96% of the estimated dense-FP8 ceiling.
  This is where consumer Blackwell earns its keep.

---

## Honesty contract (non-negotiable)

- TFLOP/s from **CUDA-event timing, warmup, mean over many iters** — never a max.
- The FP32 peak is **computed** (cores × 2 × boost clock) and reported as **% of peak**.
  Tensor rows report absolute + speedup vs FP32 — **no vendor "AI TOPS"** quoted
  (those are sparse / FP4 and not comparable to dense GEMM here).
- `[KNOWN_LIMIT]` consumer Blackwell ≠ datacenter (no NVLink/HBM; FP64 is crippled
  on GeForce and deliberately not characterized).
- This pass is **throughput-only** (zero data). A value-correctness check vs an FP32
  reference lands with the FP4 work — *a fast GEMM that is wrong is worthless.* Stated
  plainly, not hidden.

---

## Phases (netrunner ops)

| op | what | status |
|----|------|--------|
| **BREACH**    | cuBLAS GEMM roofline, FP32 → FP8 | ✅ done |
| **RAM**       | GDDR7 memory bandwidth | ⏳ |
| **DEEP DIVE** | FP4 (nvfp4 / mxfp4), beyond the wall | ⏳ |
| **THE TRACE** | correctness check + roofline figure + the honest report | ⏳ |

---

## Build & run

```bash
# Load the MSVC env first. NOTE: call vcvars64.bat WITHOUT `>nul` —
# redirecting its output leaves cl.exe off PATH and nvcc can't find the host compiler.
call "...\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -arch=sm_120 src/gemm_bench.cu -lcublas -lcublasLt -o gemm_bench
./gemm_bench
```

## Hardware / toolchain
RTX 5060 Ti (Blackwell, sm_120, 16 GB) · CUDA 13.0 · cuBLAS / cuBLASLt · MSVC 14.44.

---
*Honest roofline. No 354×. The trace doesn't lie.* 👽⚡
