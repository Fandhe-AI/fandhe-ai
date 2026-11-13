# 判定 #4163（label: 4163v3）

**verdict: ADOPT**

INFER_GRAPH_DEFAULT_ENABLED は本イシューでは変えない（反転は #4167・所有者承認）。

## undetermined 要因
- なし

## reject 要因
- なし

## セル

| セル | off 中央値 [s] | on 中央値 [s] | ratio | ratio_r (round 順) | 5/5 一貫 | checksum 一致 | 帰属 |
|---|---|---|---|---|---|---|---|
| infer-fresh-64 | 0.000152139 | 0.000142792 | 0.9386 | 0.9458, 0.9478, 0.9119, 0.9393, 0.9066 | はい | はい | stream 変更のみ（capture に届かない） |
| infer-fresh-1024 | 0.000417748 | 0.000386684 | 0.9256 | 0.5584, 1.0209, 1.0039, 1.4190, 0.9504 | いいえ | はい | stream 変更のみ（capture に届かない） |
| infer-fresh-4096 | 0.000817648 | 0.000803570 | 0.9828 | 0.7318, 0.9758, 1.0763, 0.9785, 0.9605 | いいえ | はい | stream 変更のみ（capture に届かない） |
| infer-reuse-64 | 0.000078470 | 0.000077580 | 0.9887 | 0.9882, 0.9885, 0.9903, 0.9870, 0.9961 | はい | はい | stream 変更のみ（常駐入力の本番経路は capture に届かない） |
| infer-reuse-1024 | 0.000112434 | 0.000112314 | 0.9989 | 0.9989, 0.9914, 0.9816, 1.0063, 0.9915 | いいえ | はい | stream 変更のみ（常駐入力の本番経路は capture に届かない） |
| infer-reuse-4096 | 0.000274464 | 0.000274038 | 0.9984 | 1.0254, 0.9978, 1.0215, 0.9751, 0.9999 | いいえ | はい | stream 変更のみ（常駐入力の本番経路は capture に届かない） |
| infer-host-64 | 0.000090741 | 0.000085317 | 0.9402 | 0.9525, 0.9345, 0.9363, 0.9448, 0.9434 | はい | はい | stream 変更 + capture の合算 |
| infer-host-1024 | 0.000173898 | 0.000168706 | 0.9701 | 0.9653, 0.9733, 0.9704, 0.9701, 0.9652 | はい | はい | stream 変更 + capture の合算 |
| infer-host-4096 | 0.000498472 | 0.000495096 | 0.9932 | 0.9976, 0.9944, 0.9904, 0.9904, 0.9932 | はい | はい | stream 変更 + capture の合算 |
| train-fresh | 0.000530056 | 0.000491844 | 0.9279 | 0.9771, 1.0004, 0.9253, 0.9154, 0.9623 | いいえ | はい | stream 変更のみ（学習への波及） |
| train-reuse | 0.000136305 | 0.000135417 | 0.9935 | 0.9950, 0.9990, 0.9970, 0.9903, 0.9968 | はい | はい | stream 変更のみ（学習への波及） |

## ノイズ帯の記録（判定外。NB-RANGE5 を名前で参照）

| セル | ratio_r の最大-最小 |
|---|---|
| infer-fresh-64 | 0.0412 |
| infer-fresh-1024 | 0.8606 |
| infer-fresh-4096 | 0.3445 |
| infer-reuse-64 | 0.0092 |
| infer-reuse-1024 | 0.0247 |
| infer-reuse-4096 | 0.0503 |
| infer-host-64 | 0.0180 |
| infer-host-1024 | 0.0081 |
| infer-host-4096 | 0.0072 |
| train-fresh | 0.0850 |
| train-reuse | 0.0087 |

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
| r3 | N=3・阻害 0 件・既定OFF前提 1 件・不安定 2 件 |
| r3-B（補助・判定外） | rc=101 |

## R1(b) の失敗詳細
- 失敗なし

## R3 の分類（各腕 N 回。阻害 = 腕 A で毎回失敗・腕 C で毎回成功。不安定は記録のみで判定に使わない）

| crate | テスト | 腕C（回ごとの結果） | 腕A（回ごとの結果） | 分類 |
|---|---|---|---|---|
| fandhe-ai-backend-cuda | gemm_wmma_tf32_opt::wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096 | FAILED×1/ok×2 | FAILED×2/ok×1 | 不安定（判定外） |
| fandhe-ai-backend-cuda | gemm_wmma_tf32_staged::wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096 | FAILED×1/ok×2 | FAILED×1/ok×2 | 不安定（判定外） |
| fandhe-ai-backend-cuda | graph_capture_real_device_optin_off::opt_in_off_keeps_legacy_stream_and_returns_none | ok×3 | FAILED×3 | 既定OFF前提 |
