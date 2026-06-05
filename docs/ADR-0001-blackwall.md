# ADR-0001: BLACKWALL — honest precision-spectrum roofline on Blackwell (sm_120)

**Status:** Accepted
**Date:** 2026-06-05
**Decider:** Antonio (QuantumDrizzy)

## Context

The RTX 5060 Ti is consumer **Blackwell (sm_120)**: 5th-gen tensor cores with
FP8/FP6/FP4 + GDDR7. The genuine wins over the prior baseline are **low-precision
tensor-core throughput** and **memory bandwidth** — *not* FP64 (crippled on GeForce)
and *not* the datacenter feature set (no NVLink/HBM). BLACKWALL characterizes exactly
where Blackwell's throughput lives across the precision spectrum, honestly, as a
portfolio artifact for systems/performance/CUDA roles (e.g. NVIDIA DevTech, Zürich).

The name is the pun: netrunners run the **Blackwall** (the Cyberpunk 2077 AI firewall);
this runs *with* the **Blackwell** GPU. The precision dive FP32→FP4 maps to going
deeper into the Net — faster, lossier, more dangerous.

## Decision

A small, honest CUDA benchmark. **C++/CUDA** core (cuBLAS / cuBLASLt; CUTLASS for FP4),
**Python** for plots/report, **nvcc/CMake** build. Right tool per domain — no Rust here.

### The honesty contract (the whole point)
- CUDA-event timing, warmup, **mean over many iters** — never a cherry-picked max.
- Declare the baseline: FP32 CUDA-core peak is **computed** (cores × 2 × boost clock);
  report **% of peak**. Tensor rows report absolute TFLOP/s + speedup vs FP32. **Never
  quote vendor "AI TOPS"** as a peak (sparse / FP4, not comparable).
- **Verify correctness** of low-precision GEMMs vs an FP32 reference (bounded rel-error)
  — a fast GEMM that is *wrong* is worthless. (Lands in THE TRACE.)
- `[KNOWN_LIMIT]` consumer ≠ datacenter; FP64 not characterized; throughput-only until
  the correctness gate.
- The discipline is the "distrust a too-good (or too-low) number" rule applied to our
  own results — already paid off (see Consequences).

## Build sequence (honesty-gated)

| Op | Build | Gate |
|----|-------|------|
| **BREACH** | cuBLAS GEMM sweep, FP32/TF32/BF16/FP16(2 accum)/FP8 | TFLOP/s + % of peak, event-timed, medians — **✅ done** |
| **DEEP DIVE** | FP4 (nvfp4) via **cuBLASLt block-scaling** (no CUTLASS needed) | 342 TFLOP/s · 20× FP32 · ≈2× FP8 — **✅ done** (throughput) |
| **THE TRACE** | roofline figure + numerical correctness vs an FP16 reference | figure **✅** · FP8/FP4 correctness (real block-scaling) = **open gate**, not faked |
| **RAM** | GDDR7 bandwidth microbench | GB/s vs spec, % of peak — ⏳ |

## Consequences

**Easier:** a differentiated, current, job-relevant artifact (Blackwell FP8/FP4, honest
roofline) in the converting archetype (systems/perf/CUDA).
**Harder:** must resist quoting marketing peaks; must verify correctness, not just speed;
FP4/CUTLASS is the real toolchain risk (gated last, de-risked via cuBLASLt first).

**De-risk mines already caught** (the reason de-risk goes first):
1. `vcvars64.bat >nul` leaves `cl.exe` off PATH → nvcc fails (drop the redirect).
2. Macro-internal `e` collided with a `cudaEvent_t e` (rename event vars).
3. `cudaDeviceProp::clockRate` removed in CUDA 13 → use `cudaDeviceGetAttribute`.
4. cuBLASLt FP8 rejected the config until the **D scale was removed** for a BF16 output
   (D scale only applies to an FP8 output).

## Result — BREACH + DEEP DIVE (N=8192, RTX 5060 Ti)

FP32 17.1 (72% of 23.7 peak) · TF32 23.8 (1.39×) · BF16/FP16 f32-acc 47.0 (2.75×) ·
FP16 f16-acc 87.9 (5.15×) · FP8 e4m3→bf16 184.6 (10.82×) · **FP4 nvfp4→bf16 341.9 (20.04×)**.
The ladder ≈ doubles each precision halving — the consistency *is* the honesty check.
