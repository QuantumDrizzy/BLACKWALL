// BLACKWALL — gemm_bench.cu
// ============================================================================
// Phase 1 "BREACH" — GEMM roofline across the precision spectrum on Blackwell
// (sm_120), via cuBLAS. The tensor-core uplift, MEASURED on the metal.
//
// Covered here: FP32 (CUDA cores) · TF32 · BF16 · FP16 (FP32-accum & FP16-accum).
// FP8 (cuBLASLt) and FP4 (CUTLASS, "DEEP DIVE") come next, once these are pinned.
//
// Honesty discipline (non-negotiable):
//   - CUDA-event timing, warmup, mean over many iters (not a cherry-picked max).
//   - FP32 CUDA-core peak is COMPUTED from deviceQuery (cores x 2 x boost clock)
//     and reported as % of peak. For tensor precisions we report the absolute
//     TFLOP/s + the speedup vs FP32 — we do NOT quote a fabricated tensor peak
//     (vendor "AI TOPS" are sparse / FP8-FP4 and not comparable here).
//   - Throughput-only pass (zero matrices; GEMM does not short-circuit on zeros).
//     A value-correctness check vs an FP32 reference lands with the FP8 work,
//     where precision actually bites. Stated plainly, not hidden.
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#define CUDA_CHECK(x)   do{ cudaError_t err_=(x); if(err_!=cudaSuccess){ \
    fprintf(stderr,"CUDA  %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(err_)); exit(1);} }while(0)
#define CUBLAS_CHECK(x) do{ cublasStatus_t st_=(x); if(st_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d  status %d\n",__FILE__,__LINE__,(int)st_); exit(1);} }while(0)

// One timed square GEMM (M=N=K=n, no transpose). All operands share `typ`.
// alpha/beta type follows the compute type (half for COMPUTE_16F, else float).
static double time_gemm(cublasHandle_t h, int n, cudaDataType_t typ,
                        cublasComputeType_t comp, int iters) {
    const size_t elems = (size_t)n * n;
    const size_t bytes = (typ == CUDA_R_16F || typ == CUDA_R_16BF) ? 2 : 4;
    void *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, elems * bytes)); CUDA_CHECK(cudaMemset(A, 0, elems * bytes));
    CUDA_CHECK(cudaMalloc(&B, elems * bytes)); CUDA_CHECK(cudaMemset(B, 0, elems * bytes));
    CUDA_CHECK(cudaMalloc(&C, elems * bytes)); CUDA_CHECK(cudaMemset(C, 0, elems * bytes));

    const float  alpha_f = 1.0f, beta_f = 0.0f;
    const __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    const bool   half_scalars = (comp == CUBLAS_COMPUTE_16F);
    const void*  alpha = half_scalars ? (const void*)&alpha_h : (const void*)&alpha_f;
    const void*  beta  = half_scalars ? (const void*)&beta_h  : (const void*)&beta_f;

    auto run = [&]() {
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                  alpha, A, typ, n, B, typ, n,
                                  beta,  C, typ, n, comp, CUBLAS_GEMM_DEFAULT));
    };

    for (int i = 0; i < 5; ++i) run();                 // warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventRecord(evStart));
    for (int i = 0; i < iters; ++i) run();
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));

    CUDA_CHECK(cudaEventDestroy(evStart)); CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaFree(A)); CUDA_CHECK(cudaFree(B)); CUDA_CHECK(cudaFree(C));
    return (double)ms / iters;                          // mean ms / iter
}

// FP8 (e4m3 in, bf16 out) GEMM via cuBLASLt — the consumer-Blackwell headline.
// FP8 matmul on cuBLASLt requires the TN layout (op(A)=A^T) plus A/B/D scale
// pointers; we use unit scales (throughput-only, zero data). Returns ms/iter,
// or -1 if no FP8 algo is available for this config (reported honestly).
static double time_fp8_gemm(cublasLtHandle_t lt, int n, int iters) {
    const size_t elems = (size_t)n * n;
    const size_t wsBytes = 32ull * 1024 * 1024;
    void *A = nullptr, *B = nullptr, *D = nullptr, *ws = nullptr;
    CUDA_CHECK(cudaMalloc(&A, elems));            // e4m3: 1 byte/elem
    CUDA_CHECK(cudaMalloc(&B, elems));
    CUDA_CHECK(cudaMalloc(&D, elems * 2));        // bf16 out: 2 bytes/elem
    CUDA_CHECK(cudaMalloc(&ws, wsBytes));
    CUDA_CHECK(cudaMemset(A, 0, elems)); CUDA_CHECK(cudaMemset(B, 0, elems));
    CUDA_CHECK(cudaMemset(D, 0, elems * 2));

    float one = 1.0f; float *sA = nullptr, *sB = nullptr, *sD = nullptr;
    CUDA_CHECK(cudaMalloc(&sA, 4)); CUDA_CHECK(cudaMalloc(&sB, 4)); CUDA_CHECK(cudaMalloc(&sD, 4));
    CUDA_CHECK(cudaMemcpy(sA, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sB, &one, 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sD, &one, 4, cudaMemcpyHostToDevice));

    cublasLtMatmulDesc_t op;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sA, sizeof(sA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sB, sizeof(sB)));
    // No D scale: output is BF16, not FP8 (D scale only applies to an FP8 output).

    cublasLtMatrixLayout_t lA, lB, lD;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lA, CUDA_R_8F_E4M3, n, n, n));  // K x M (TN)
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lB, CUDA_R_8F_E4M3, n, n, n));  // K x N
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lD, CUDA_R_16BF,    n, n, n));  // M x N

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes)));
    cublasLtMatmulHeuristicResult_t heur{}; int got = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt, op, lA, lB, lD, lD, pref, 1, &heur, &got);
    if (got == 0) {
        fprintf(stderr, "# FP8: no algo for n=%d (heuristic status %d, got %d)\n", n, (int)hs, got);
        cudaFree(A); cudaFree(B); cudaFree(D); cudaFree(ws);
        cudaFree(sA); cudaFree(sB); cudaFree(sD);
        return -1.0;
    }

    const float alpha = 1.0f, beta = 0.0f;
    auto run = [&]() {
        CUBLAS_CHECK(cublasLtMatmul(lt, op, &alpha, A, lA, B, lB, &beta,
                                    D, lD, D, lD, &heur.algo, ws, wsBytes, 0));
    };
    for (int i = 0; i < 5; ++i) run();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) run();
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lA); cublasLtMatrixLayoutDestroy(lB); cublasLtMatrixLayoutDestroy(lD);
    cublasLtMatmulDescDestroy(op);
    cudaFree(A); cudaFree(B); cudaFree(D); cudaFree(ws);
    cudaFree(sA); cudaFree(sB); cudaFree(sD);
    return (double)ms / iters;
}

int main() {
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    // FP32 CUDA-core peak: sm_120 has 128 FP32 lanes/SM, 2 FLOP/FMA.
    int clk_khz = 0;  // prop.clockRate was removed in CUDA 13 -> use the attribute
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    const double boost_ghz = clk_khz / 1.0e6;           // attribute is in kHz
    const long   cuda_cores = (long)p.multiProcessorCount * 128;
    const double fp32_peak  = cuda_cores * 2.0 * boost_ghz / 1.0e3;  // TFLOP/s

    printf("# BLACKWALL :: BREACH — GEMM roofline\n");
    printf("# GPU: %s  (sm_%d%d, %.0f GB, %d SMs, ~%.2f GHz boost)\n",
           p.name, p.major, p.minor, p.totalGlobalMem / 1e9, p.multiProcessorCount, boost_ghz);
    printf("# FP32 CUDA-core peak (%ld cores x 2 x %.2f GHz) = %.1f TFLOP/s\n",
           cuda_cores, boost_ghz, fp32_peak);
    printf("# tensor rows: absolute TFLOP/s + speedup vs FP32 (no vendor peak quoted).\n\n");

    cublasHandle_t h; CUBLAS_CHECK(cublasCreate(&h));
    cublasLtHandle_t lt; CUBLAS_CHECK(cublasLtCreate(&lt));

    struct Prec { const char* name; cudaDataType_t typ; cublasComputeType_t comp; };
    const std::vector<Prec> precs = {
        { "FP32",        CUDA_R_32F,  CUBLAS_COMPUTE_32F },             // CUDA cores
        { "TF32",        CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_TF32 },   // TC
        { "BF16/f32acc", CUDA_R_16BF, CUBLAS_COMPUTE_32F },            // TC
        { "FP16/f32acc", CUDA_R_16F,  CUBLAS_COMPUTE_32F },            // TC
        { "FP16/f16acc", CUDA_R_16F,  CUBLAS_COMPUTE_16F },            // TC, half accum
    };
    const std::vector<int> sizes = { 2048, 4096, 8192 };

    printf("%-12s %6s %10s %10s %12s\n", "precision", "N", "ms/iter", "TFLOP/s", "vs FP32 /%pk");
    printf("------------------------------------------------------------\n");
    for (int n : sizes) {
        int iters = (n <= 2048) ? 100 : (n <= 4096 ? 50 : 20);
        double fp32_tflops = 0.0;
        for (const auto& pr : precs) {
            double ms = time_gemm(h, n, pr.typ, pr.comp, iters);
            double tflops = 2.0 * (double)n * n * n / (ms / 1e3) / 1e12;
            if (pr.typ == CUDA_R_32F && pr.comp == CUBLAS_COMPUTE_32F) {
                fp32_tflops = tflops;
                printf("%-12s %6d %10.3f %10.1f %11.0f%%\n",
                       pr.name, n, ms, tflops, 100.0 * tflops / fp32_peak);
            } else {
                printf("%-12s %6d %10.3f %10.1f %10.2fx\n",
                       pr.name, n, ms, tflops, tflops / fp32_tflops);
            }
        }
        // FP8 (e4m3 in, bf16 out) via cuBLASLt — the consumer-Blackwell headline.
        double fp8_ms = time_fp8_gemm(lt, n, iters);
        if (fp8_ms > 0.0) {
            double tf = 2.0 * (double)n * n * n / (fp8_ms / 1e3) / 1e12;
            printf("%-12s %6d %10.3f %10.1f %10.2fx\n",
                   "FP8e4m3>bf16", n, fp8_ms, tf, tf / fp32_tflops);
        }
        printf("------------------------------------------------------------\n");
    }

    CUBLAS_CHECK(cublasDestroy(h));
    cublasLtDestroy(lt);
    return 0;
}
