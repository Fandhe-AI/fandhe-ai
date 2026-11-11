# 判定 #4163（label: 4163）

**verdict: UNDETERMINED**

INFER_GRAPH_DEFAULT_ENABLED は本イシューでは変えない（反転は #4167・所有者承認）。

## undetermined 要因
- r3: 阻害 3 件

## reject 要因
- ratio > 1.00: infer-fresh-1024

## セル

| セル | off 中央値 [s] | on 中央値 [s] | ratio | ratio_r (round 順) | 5/5 一貫 | checksum 一致 | 帰属 |
|---|---|---|---|---|---|---|---|
| infer-fresh-64 | 0.000152115 | 0.000143042 | 0.9404 | 0.9342, 0.9600, 0.9231, 0.9545, 0.9428 | はい | はい | stream 変更のみ（capture に届かない） |
| infer-fresh-1024 | 0.000305443 | 0.000327850 | 1.0734 | 0.9438, 0.9747, 1.0734, 1.2750, 0.4058 | いいえ | はい | stream 変更のみ（capture に届かない） |
| infer-fresh-4096 | 0.000809073 | 0.000801705 | 0.9909 | 0.9909, 0.9870, 0.9669, 1.0087, 0.8925 | いいえ | はい | stream 変更のみ（capture に届かない） |
| infer-reuse-64 | 0.000077588 | 0.000077533 | 0.9993 | 0.9899, 0.9901, 0.9983, 1.0001, 0.9997 | いいえ | はい | stream 変更のみ（常駐入力の本番経路は capture に届かない） |
| infer-reuse-1024 | 0.000112507 | 0.000112057 | 0.9960 | 0.9992, 0.9976, 0.9909, 0.9918, 0.9939 | はい | はい | stream 変更のみ（常駐入力の本番経路は capture に届かない） |
| infer-reuse-4096 | 0.000277240 | 0.000274424 | 0.9898 | 0.9898, 0.9758, 0.9972, 0.9827, 1.0177 | いいえ | はい | stream 変更のみ（常駐入力の本番経路は capture に届かない） |
| infer-host-64 | 0.000090692 | 0.000085547 | 0.9433 | 0.9401, 0.9433, 0.9434, 0.9543, 0.9428 | はい | はい | stream 変更 + capture の合算 |
| infer-host-1024 | 0.000174266 | 0.000168690 | 0.9680 | 0.9743, 0.9683, 0.9681, 0.9650, 0.9618 | はい | はい | stream 変更 + capture の合算 |
| infer-host-4096 | 0.000498648 | 0.000494066 | 0.9908 | 0.9902, 0.9897, 0.9915, 0.9869, 0.9937 | はい | はい | stream 変更 + capture の合算 |
| train-fresh | 0.000509401 | 0.000505303 | 0.9920 | 0.9630, 0.9920, 1.0056, 0.9556, 0.9455 | いいえ | はい | stream 変更のみ（学習への波及） |
| train-reuse | 0.000135922 | 0.000135114 | 0.9941 | 0.9929, 0.9926, 0.9954, 0.9933, 0.9947 | はい | はい | stream 変更のみ（学習への波及） |

## ノイズ帯の記録（判定外。NB-RANGE5 を名前で参照）

| セル | ratio_r の最大-最小 |
|---|---|
| infer-fresh-64 | 0.0369 |
| infer-fresh-1024 | 0.8693 |
| infer-fresh-4096 | 0.1162 |
| infer-reuse-64 | 0.0102 |
| infer-reuse-1024 | 0.0083 |
| infer-reuse-4096 | 0.0419 |
| infer-host-64 | 0.0142 |
| infer-host-1024 | 0.0125 |
| infer-host-4096 | 0.0068 |
| train-fresh | 0.0601 |
| train-reuse | 0.0028 |

## 前提ゲート

| ゲート | 結果 |
|---|---|
| r0-linear | 合格 |
| r0-graph | 合格 |
| r1a | 合格（PASS=5 SKIP=0） |
| r1b | 5/5 合格 |
| r2-64 | 全行一致 |
| r2-1024 | 全行一致 |
| r2-4096 | 全行一致 |
| r3 | N=3・阻害 3 件・既定OFF前提 1 件・不安定 3 件 |
| r3-B（補助・判定外） | rc=101 |

## R1(b) の失敗詳細
- 失敗なし

## R3 の分類（各腕 N 回。阻害 = 腕 A で毎回失敗・腕 C で毎回成功。不安定は記録のみで判定に使わない）

| crate | テスト | 腕C（回ごとの結果） | 腕A（回ごとの結果） | 分類 |
|---|---|---|---|---|
| fandhe-ai-backend-cuda | gemm_wmma_tf32_opt::wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096 | ok×3 | ok×2/FAILED×1 | 不安定（判定外） |
| fandhe-ai-backend-cuda | gemm_wmma_tf32_staged::wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096 | ok×3 | FAILED×1/ok×2 | 不安定（判定外） |
| fandhe-ai-backend-cuda | graph_capture_real_device_optin_off::opt_in_off_keeps_legacy_stream_and_returns_none | ok×3 | FAILED×3 | 既定OFF前提 |
| fandhe-ai-backend-cuda | lib::context_cache::tests::cached_allocator_keeps_distinct_contexts_on_same_ordinal_separate | ok×3 | FAILED×3 | 阻害 |
| fandhe-ai-backend-cuda | lib::gemm::tests::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096 | ok×3 | ok×2/FAILED×1 | 不安定（判定外） |
| fandhe-ai-backend-cuda | lib::module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache | ok×3 | FAILED×3 | 阻害 |
| fandhe-ai-backend-cuda | lib::scalar_op_cache_wiring_tests::cached_scalar_unary_kernel_distinct_devices_do_not_share_cache | ok×3 | FAILED×3 | 阻害 |
