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
| **BREACH** | cuBLAS GEMM sweep, FP32/TF32/BF16/FP16(2 accum)/FP8 | TFLOP/s + % of peak, event-timed, medians — **done** |
| **RAM** | GDDR7 bandwidth microbench | GB/s vs spec, % of peak |
| **DEEP DIVE** | FP4 (nvfp4/mxfp4) via cuBLASLt → CUTLASS fallback | correctness vs FP16 + TFLOP/s + honest FP4-over-FP8 delta. `[KNOWN_LIMIT]` if toolchain stalls |
| **THE TRACE** | correctness check + roofline figure + README report | error bounded, figure reproducible |

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

## BREACH result (N=8192, RTX 5060 Ti)

FP32 17.0 (72% of 23.7 peak) · TF32 23.2 (1.36×) · BF16/FP16 f32-acc 46.4 (2.73×) ·
FP16 f16-acc 85.7 (5.03×) · **FP8 e4m3→bf16 182.1 (10.64×)**.
