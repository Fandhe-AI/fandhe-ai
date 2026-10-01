# sm_121 ISA プローブ集計（イシュー #2122）

## G0（前提ゲート）

- 成立

mode=official　git_head=7b6019aeb7856eb9cb2e28929a735b9938bc1503

- compile プロセス: 完走
- device_attributes_dump（記録のみ）: ok

## プローブ別の判定

| プローブ | AC | 条項 | home | compute_121 | compute_121a | compute_121f |
|---|---|---|---|---|---|---|
| ctl.copy | BASE | R-CTL | sm_80 | 成立 | 成立 | 成立 |
| macro.arch | AC1 | R-TC5 | sm_80 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tc5.alloc | AC1 | R-TC5 | sm_100a | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） |
| tc5.ld | AC1 | R-TC5 | sm_100a | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） |
| tc5.cross | AC1 | R-TC5 | sm_100a | ロード失敗 | ロード失敗 | ロード失敗 |
| mma.tf32.m16n8k8 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.tf32.m16n8k4 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.f16.m16n8k16.f32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.f16.m16n8k8.f32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.f16.m16n8k16.f16 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.bf16.m16n8k16.f32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.bf16.m16n8k8.f32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.f64.m8n8k4 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.f16.m8n8k4 | AC4 | R-MMA | sm_80 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.f64.m16n8k4 | AC4 | R-MMA | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.f64.m16n8k8 | AC4 | R-MMA | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.f64.m16n8k16 | AC4 | R-MMA | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.s8.m16n8k32 | AC4 | R-MMA | sm_80 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.e4m3.m16n8k32 | AC4 | R-MMA | sm_89 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.e5m2.m16n8k32 | AC4 | R-MMA | sm_89 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.f8f6f4.m16n8k32 | AC4 | R-MMA | sm_120a | ptxas 拒否（オフライン） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.block_scale.m16n8k64 | AC4 | R-MMA | sm_120a | ptxas 拒否（オフライン） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| mma.ldmatrix.x1 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.ldmatrix.x2 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.ldmatrix.x4 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.ldmatrix.x4_trans | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| mma.stmatrix.x4 | AC4 | R-MMA | sm_90 | 成立 | 成立 | 成立 |
| simt.fma_f32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| simt.fma_f64 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| simt.fma_f16x2 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| simt.fma_bf16x2 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| simt.f32x2_add | AC4 | R-MMA | sm_100 | 成立 | 成立 | 成立 |
| simt.f32x2_mul | AC4 | R-MMA | sm_100 | 成立 | 成立 | 成立 |
| simt.f32x2_fma | AC4 | R-MMA | sm_100 | 成立 | 成立 | 成立 |
| simt.cvt_tf32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| simt.elect_sync | AC4 | R-MMA | sm_90 | 成立 | 成立 | 成立 |
| simt.redux_u32 | AC4 | R-MMA | sm_80 | 成立 | 成立 | 成立 |
| simt.redux_f32 | AC4 | R-MMA | sm_100a | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） |
| wgmma.m64n8k16 | AC5 | R-HOPPER | sm_90a | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） |
| hop.griddepcontrol | AC5 | R-HOPPER | sm_90 | 成立 | 成立 | 成立 |
| hop.fence_proxy_async | AC5 | R-HOPPER | sm_90 | 成立 | 成立 | 成立 |
| snr.dec | AC5 | R-SNR | sm_90a | ptxas 拒否（オフライン） | 成立 | 成立 |
| snr.incdec | AC5 | R-SNR | sm_90a | ptxas 拒否（オフライン） | 成立 | 成立 |
| clu.dims1 | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| clu.dims2 | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| clu.dims4 | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| clu.dims8 | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| clu.dims16 | AC5 | R-CLU | sm_90 | 実行時エラー | 実行時エラー | 実行時エラー |
| clu.dsmem | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| ctl.raw | BASE | R-CTL | sm_80 | 成立 | 成立 | 成立 |
| ctl.rawmap | BASE | R-CTL | sm_90 | 成立 | 成立 | 成立 |
| clu.rt2 | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| clu.rt4 | AC5 | R-CLU | sm_90 | 成立 | 成立 | 成立 |
| tma.base_cta | AC3 | R-TMA-BASE | sm_90 | 成立 | 成立 | 成立 |
| tma.base_cluster | AC3 | R-TMA-BASE | sm_90 | 成立 | 成立 | 成立 |
| tma.coord | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.oob_none | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.oob_nan | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.oob_neg | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.swz32 | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.swz64 | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.swz128 | AC3 | R-TMA-SEM | sm_90 | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） |
| tma.store | AC3 | R-TMA-XFER | sm_90 | 成立 | 成立 | 成立 |
| tma.prefetch | AC3 | R-TMA-XFER | sm_90 | 成立 | 成立 | 成立 |
| tma.multicast | AC3 | R-TMA-XFER | sm_90 | 成立 | 成立 | 成立 |
| tma.bulk_cta | AC3 | R-TMA-XFER | sm_90 | 成立 | 成立 | 成立 |
| tma.bulk_cluster | AC3 | R-TMA-XFER | sm_90 | 成立 | 成立 | 成立 |
| attr.limits | AC2 | R-GUIDE | sm_80 | 成立 | 成立 | 成立 |
| attr.cluster | AC2 | R-GUIDE | sm_80 | 成立 | 成立 | 成立 |
| attr.misc | AC2 | R-GUIDE | sm_80 | 成立 | 成立 | 成立 |

## R-GUIDE（ガイドとの突き合わせ）

| claim | 出典 | 判定 | 理由コード | 測定 |
|---|---|---|---|---|
| C01-warps-per-sm | .claude/skills/nvidia-cuda/references/blackwell-tuning/streaming-multiprocessor.md:23 | 一致（12.0 の記述を外挿） | - | 測定値=1536,1536,1536 |
| C02-blocks-per-sm | .claude/skills/nvidia-cuda/references/blackwell-tuning/streaming-multiprocessor.md:26 | 不一致 | - | 測定値=24,24,24 |
| C03-smem-per-sm | .claude/skills/nvidia-cuda/references/blackwell-tuning/streaming-multiprocessor.md:27 | 不一致 | - | 測定値=102400,102400,102400 |
| C04-smem-per-block | .claude/skills/nvidia-cuda/references/blackwell-tuning/streaming-multiprocessor.md:28 | 一致（12.0 の記述を外挿） | - | 測定値=101376,101376,101376 |
| C05-portable-cluster-8 | .claude/skills/nvidia-cuda/references/blackwell-tuning/streaming-multiprocessor.md:33 | 一致（12.0 の記述を外挿） | - | 判定=成立,成立,成立 |
| C06-tcgen05-sm100plus | .claude/skills/nvidia-cuda/references/ptx-isa/instructions-matrix-multiply.md:26 | 不一致 | - | 判定=ptxas 拒否（オフライン）,ptxas 拒否（オフライン）,ptxas 拒否（オフライン） |
| C07-wgmma-unavailable | docs/cuda-tensor-core-design.md:138 | 一致（12.0 の記述を外挿） | - | 判定=ptxas 拒否（オフライン）,ptxas 拒否（オフライン）,ptxas 拒否（オフライン） |
| C08-tcgen05-unavailable | docs/cuda-tensor-core-design.md:139 | 一致（12.0 の記述を外挿） | - | 判定=ptxas 拒否（オフライン）,ptxas 拒否（オフライン）,ptxas 拒否（オフライン） |
| C09-cluster-1x1x1 | docs/cuda-tensor-core-design.md:140 | 不一致 | - | 判定=成立,成立,成立 |
| C10-static-smem-48kb | .claude/skills/nvidia-cuda/references/blackwell-tuning/memory-system.md:26 | 一致（12.0 の記述を外挿） | - | 測定値=49152,49152,49152 |

## R-HOPPER（Hopper との差分）

Hopper 列は GB10 上の NVRTC（ptxas）による sm_90a 受理の実測であり、Hopper 実機での実行は未検証。PTX ISA 9.0 の節番号は要確認。

| プローブ | home | Hopper(sm_90a) S2 | sm_121 系 S2（121/121a/121f） | 判定 | 方向 | PTX ISA 9.0 節 |
|---|---|---|---|---|---|---|
| ctl.copy | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| macro.arch | sm_80 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tc5.alloc | sm_100a | rejected | rejected/rejected/rejected | ptxas 拒否（オフライン） | 両方で拒否 | 要確認 |
| tc5.ld | sm_100a | rejected | rejected/rejected/rejected | ptxas 拒否（オフライン） | 両方で拒否 | 要確認 |
| tc5.cross | sm_100a | rejected | ok/ok/ok | ロード失敗 | sm_121 のみ受理（逆方向） | 要確認 |
| mma.tf32.m16n8k8 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.tf32.m16n8k4 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.f16.m16n8k16.f32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.f16.m16n8k8.f32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.f16.m16n8k16.f16 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.bf16.m16n8k16.f32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.bf16.m16n8k8.f32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.f64.m8n8k4 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.f16.m8n8k4 | sm_80 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.f64.m16n8k4 | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.f64.m16n8k8 | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.f64.m16n8k16 | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.s8.m16n8k32 | sm_80 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.e4m3.m16n8k32 | sm_89 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.e5m2.m16n8k32 | sm_89 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| mma.f8f6f4.m16n8k32 | sm_120a | rejected | rejected/ok/ok | ptxas 拒否（オフライン）/受理のみ（実行意味論は未検証） | 判定不能（target 間で不一致） | 要確認 |
| mma.block_scale.m16n8k64 | sm_120a | rejected | rejected/ok/ok | ptxas 拒否（オフライン）/受理のみ（実行意味論は未検証） | 判定不能（target 間で不一致） | 要確認 |
| mma.ldmatrix.x1 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.ldmatrix.x2 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.ldmatrix.x4 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.ldmatrix.x4_trans | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| mma.stmatrix.x4 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.fma_f32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.fma_f64 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.fma_f16x2 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.fma_bf16x2 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.f32x2_add | sm_100 | rejected | ok/ok/ok | 成立 | sm_121 のみ受理（逆方向） | 要確認 |
| simt.f32x2_mul | sm_100 | rejected | ok/ok/ok | 成立 | sm_121 のみ受理（逆方向） | 要確認 |
| simt.f32x2_fma | sm_100 | rejected | ok/ok/ok | 成立 | sm_121 のみ受理（逆方向） | 要確認 |
| simt.cvt_tf32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.elect_sync | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.redux_u32 | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| simt.redux_f32 | sm_100a | rejected | rejected/rejected/rejected | ptxas 拒否（オフライン） | 両方で拒否 | 要確認 |
| wgmma.m64n8k16 | sm_90a | ok | rejected/rejected/rejected | ptxas 拒否（オフライン） | Hopper のみ受理 | 要確認 |
| hop.griddepcontrol | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| hop.fence_proxy_async | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| snr.dec | sm_90a | ok | rejected/ok/ok | ptxas 拒否（オフライン）/成立 | 判定不能（target 間で不一致） | 要確認 |
| snr.incdec | sm_90a | ok | rejected/ok/ok | ptxas 拒否（オフライン）/成立 | 判定不能（target 間で不一致） | 要確認 |
| clu.dims1 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| clu.dims2 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| clu.dims4 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| clu.dims8 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| clu.dims16 | sm_90 | ok | ok/ok/ok | 実行時エラー | 両方で受理 | 要確認 |
| clu.dsmem | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| ctl.raw | sm_80 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| ctl.rawmap | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| clu.rt2 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| clu.rt4 | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.base_cta | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.base_cluster | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.coord | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.oob_none | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.oob_nan | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.oob_neg | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.swz32 | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.swz64 | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.swz128 | sm_90 | ok | ok/ok/ok | 受理のみ（実行意味論は未検証） | 両方で受理 | 要確認 |
| tma.store | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.prefetch | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.multicast | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.bulk_cta | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |
| tma.bulk_cluster | sm_90 | ok | ok/ok/ok | 成立 | 両方で受理 | 要確認 |

## R-TMA 観測（候補モデルとの一致。値は S5 の detail に記録した観測）

| プローブ | target | 判定 | 観測 | 注意 |
|---|---|---|---|---|
| tma.coord | compute_121 | 受理のみ（実行意味論は未検証） | polls=14 class=ELEM_INNER_FIRST | - |
| tma.coord | compute_121a | 受理のみ（実行意味論は未検証） | polls=14 class=ELEM_INNER_FIRST | - |
| tma.coord | compute_121f | 受理のみ（実行意味論は未検証） | polls=14 class=ELEM_INNER_FIRST | - |
| tma.oob_none | compute_121 | 受理のみ（実行意味論は未検証） | polls=13 inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000 | - |
| tma.oob_none | compute_121a | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000 | - |
| tma.oob_none | compute_121f | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000 | - |
| tma.oob_nan | compute_121 | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=NAN oob_distinct=0x7ff77ff7 | - |
| tma.oob_nan | compute_121a | 受理のみ（実行意味論は未検証） | polls=13 inrange=MATCH oob_elems=96 oob_fill=NAN oob_distinct=0x7ff77ff7 | - |
| tma.oob_nan | compute_121f | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=NAN oob_distinct=0x7ff77ff7 | - |
| tma.oob_neg | compute_121 | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000 | - |
| tma.oob_neg | compute_121a | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000 | - |
| tma.oob_neg | compute_121f | 受理のみ（実行意味論は未検証） | polls=14 inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000 | - |
| tma.swz32 | compute_121 | 受理のみ（実行意味論は未検証） | polls=14 class=XOR_ADDR_BITS | - |
| tma.swz32 | compute_121a | 受理のみ（実行意味論は未検証） | polls=14 class=XOR_ADDR_BITS | - |
| tma.swz32 | compute_121f | 受理のみ（実行意味論は未検証） | polls=13 class=XOR_ADDR_BITS | - |
| tma.swz64 | compute_121 | 受理のみ（実行意味論は未検証） | polls=13 class=XOR_ADDR_BITS+SRC_B64_MODEL | - |
| tma.swz64 | compute_121a | 受理のみ（実行意味論は未検証） | polls=14 class=XOR_ADDR_BITS+SRC_B64_MODEL | - |
| tma.swz64 | compute_121f | 受理のみ（実行意味論は未検証） | polls=13 class=XOR_ADDR_BITS+SRC_B64_MODEL | - |
| tma.swz128 | compute_121 | 受理のみ（実行意味論は未検証） | polls=14 class=XOR_ADDR_BITS | - |
| tma.swz128 | compute_121a | 受理のみ（実行意味論は未検証） | polls=14 class=XOR_ADDR_BITS | - |
| tma.swz128 | compute_121f | 受理のみ（実行意味論は未検証） | polls=14 class=XOR_ADDR_BITS | - |

## R-LEGACY

| legacy | 対応 | 結論 | exit |
|---|---|---|---|
| setmaxnreg_probe_dec_accel_real_device | snr.dec@compute_121a | run_ok | 0 |
| setmaxnreg_probe_dec_base_real_device | snr.dec@compute_121 | load_failed | 0 |
| setmaxnreg_probe_incdec_accel_real_device | snr.incdec@compute_121a | run_ok | 0 |
| setmaxnreg_probe_incdec_base_real_device | snr.incdec@compute_121 | load_failed | 0 |
| tma_probe_real_device@tma_execution_probe | tma.base_cluster@compute_121 | run_ok | 0 |
| tma_probe_real_device@tma_execution_probe_cta | tma.base_cta@compute_121 | run_ok | 0 |
| tma_probe_real_device@tma_nvrtc_compile_probe | -@- | recorded | 0 |
