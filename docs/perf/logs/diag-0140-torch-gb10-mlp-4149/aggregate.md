# 区間内訳（正式） #4149

- RULE: fixed-at-utc=2026-11-10T04:37:14Z tree_sha=faf0f6bce1ab592e7b6f82e67164e47b48452867 tree_fingerprint=f18e2a5c088e6e9d6b331024da34d51d46a951f5be20bdcfa786fe47609c0d17
- round 状態: r1=PASS r2=PASS r3=PASS r4=PASS r5=PASS（PASS のみ集計）
- 時間は ms。中央値・最小〜最大は run 間。比は run ごとの 量÷窓 の中央値（窓のない section は空欄）。空欄は unavailable（推定値で埋めない）。
- 推定: 窓から実測区間を引いた残差。CPU は autograd・Python 解釈・メモリ確保・dispatch が混ざる。CUDA は enqueue と並行して進む GPU 実行待ちが混ざる。
- CUDA の窓は X が device 上にあり H2D を含まない（fandhe 側は窓に H2D を含む。並べて読むときの非対称）。
- profiler 腕の値は時間として使っていない（op 数・kernel／API 回数のみ）。

## cpu_train

| 量 | タグ | 中央値 ms | 最小〜最大 ms | 窓に対する比（中央値） | 比の最小〜最大 |
|---|---|---|---|---|---|
| window | 実測 | 0.36748 | 0.32915〜0.38693 | 1.000 | 1.000〜1.000 |
| gemm_kernel_batch10 | 実測 | 0.24635 | 0.20495〜0.26225 | 0.679 | 0.530〜0.785 |
| gemm_mm_pure_sum | 実測 | 0.25152 | 0.23664〜0.76798 | 0.752 | 0.632〜1.985 |
| ew_fwd | 実測 | 0.00941 | 0.00927〜0.00942 | 0.025 | 0.024〜0.028 |
| update | 実測 | 0.02091 | 0.02023〜0.02289 | 0.061 | 0.054〜0.070 |
| loss_item | 実測 | 0.00021 | 0.00021〜0.00022 | 0.001 | 0.001〜0.001 |
| residual | 推定 | 0.08824 | 0.04154〜0.15144 | 0.240 | 0.124〜0.391 |
| fwd_nongemm | 推定 | 0.00777 | -0.02157〜0.03845 | 0.021 | -0.065〜0.103 |
| bwd_nongemm | 推定 | 0.00998 | -0.00754〜0.05194 | 0.027 | -0.023〜0.134 |
| parts_minus_window | 実測 | -0.07976 | -0.08710〜-0.05114 | -0.217 | -0.240〜-0.155 |
| dispatch_floor_est | 推定 | 0.01871 | 0.01831〜0.01909 | 0.051 | 0.048〜0.056 |
| torch_vs_direct_gap | 実測 | 0.06632 | 0.02453〜0.07463 | 0.177 | 0.063〜0.223 |

## cpu_infer

| 量 | タグ | 中央値 ms | 最小〜最大 ms | 窓に対する比（中央値） | 比の最小〜最大 |
|---|---|---|---|---|---|
| window | 実測 | 0.13226 | 0.12379〜0.14942 | 1.000 | 1.000〜1.000 |
| gemm_kernel_batch10 | 実測 | 0.11168 | 0.10001〜0.11696 | 0.783 | 0.756〜0.902 |
| gemm_mm_pure_sum | 実測 | 0.11455 | 0.10352〜0.13334 | 0.875 | 0.772〜0.980 |
| ew_fwd | 実測 | 0.00494 | 0.00486〜0.00506 | 0.038 | 0.034〜0.040 |
| readout | 実測 | 0.00077 | 0.00077〜0.00078 | 0.006 | 0.005〜0.006 |
| checksum | 実測 | 0.00299 | 0.00293〜0.00300 | 0.023 | 0.020〜0.024 |
| residual | 推定 | 0.02348 | 0.00341〜0.02422 | 0.158 | 0.028〜0.178 |
| dispatch_floor_est | 推定 | 0.00542 | 0.00538〜0.00552 | 0.041 | 0.036〜0.043 |
| torch_vs_direct_gap | 実測 | 0.00213 | -0.01170〜0.01601 | 0.017 | -0.084〜0.122 |

## cuda_infer

| 量 | タグ | 中央値 ms | 最小〜最大 ms | 窓に対する比（中央値） | 比の最小〜最大 |
|---|---|---|---|---|---|
| window | 実測 | 0.04067 | 0.03973〜0.04118 | 1.000 | 1.000〜1.000 |
| issue | 実測 | 0.02794 | 0.02746〜0.02809 | 0.685 | 0.682〜0.697 |
| readout_sync | 実測 | 0.00736 | 0.00730〜0.00765 | 0.183 | 0.181〜0.188 |
| checksum | 実測 | 0.00304 | 0.00301〜0.00325 | 0.076 | 0.073〜0.081 |
| gpu_span_event | 実測 | 0.03318 | 0.03293〜0.03336 | 0.814 | 0.810〜0.838 |
| gemm_event_sum | 実測 | 0.02709 | 0.02618〜0.02813 | 0.669 | 0.644〜0.691 |
| residual | 推定 | 0.00208 | 0.00174〜0.00254 | 0.052 | 0.044〜0.062 |
| issue_minus_gpu_span | 実測 | -0.00527 | -0.00562〜-0.00511 | -0.129 | -0.142〜-0.126 |
| readout_sync_minus_tiny_roundtrip | 実測 | -0.00284 | -0.00290〜-0.00223 | -0.070 | -0.073〜-0.055 |
| graph_replay_ref | 実測 | 0.02346 | 0.02328〜0.02349 | 0.577 | 0.565〜0.591 |

## cuda_fixed

| 量 | タグ | 中央値 ms | 最小〜最大 ms | 窓に対する比（中央値） | 比の最小〜最大 |
|---|---|---|---|---|---|
| sync_idle | 実測 | 0.00249 | 0.00245〜0.00251 |  |  |
| tiny_roundtrip | 実測 | 0.01020 | 0.00988〜0.01034 |  |  |
| launch_floor_add | 実測 | 0.00324 | 0.00321〜0.00331 |  |  |
| event_record_pair | 実測 | 0.00851 | 0.00841〜0.00870 |  |  |

## cuda_shapes

| 量 | タグ | 中央値 ms | 最小〜最大 ms | 窓に対する比（中央値） | 比の最小〜最大 |
|---|---|---|---|---|---|
| mm_event_l1_fwd | 実測 | 0.01470 | 0.01358〜0.01568 |  |  |
| mm_event_l1_dw | 実測 | 0.01216 | 0.01216〜0.01227 |  |  |
| mm_event_l1_dx | 実測 | 0.01637 | 0.01627〜0.01706 |  |  |
| mm_event_l2_fwd | 実測 | 0.01245 | 0.01234〜0.01259 |  |  |
| mm_event_l2_dw | 実測 | 0.01238 | 0.01219〜0.01256 |  |  |
| mm_event_l2_dx | 実測 | 0.01243 | 0.01213〜0.01259 |  |  |

## profiler 腕（op 数・kernel／API 回数のみ。時間は使わない）

| section | step あたり aten op（全体） | 同（最上位） | aten::mm が子 op を持つ |
|---|---|---|---|
| cpu_train | 99 | 34 | はい |
| cpu_infer | 19 | 10 | はい |
| cuda_infer | 16 | 10 | はい |

### cuda_infer の CUDA 活動

- cudaGraphLaunch の有無（いずれかの round で観測）: なし（eager）
- CUDA API 回数 / step（run 間の中央値）: cuLaunchKernel=2.00, cudaDeviceSynchronize=0.04, cudaLaunchKernel=3.00, cudaMemcpyAsync=1.00, cudaStreamSynchronize=1.00

| kernel 名 | 回数 / step（run 間の中央値） |
|---|---|
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<float> const&)::{lambda(int)#1})` | 2.00 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, float const*, float*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` | 2.00 |
| `Memcpy DtoH (Device -> Pageable)` | 1.00 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>, false>(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#7}::operator()() const::{lambda(float)#1}, std::array<char*, 2ul>)` | 1.00 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_align1>(cutlass_80_simt_sgemm_128x32_8x5_nn_align1::Params)` | 1.00 |
| `void cutlass::Kernel2<cutlass_80_simt_sgemm_32x128_8x5_nn_align1>(cutlass_80_simt_sgemm_32x128_8x5_nn_align1::Params)` | 1.00 |
