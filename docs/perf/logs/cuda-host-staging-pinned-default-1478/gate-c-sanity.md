# ゲート C（framework-compare gemm cuda reuse）非後退ガード — 縮小スコープ実測記録

イシュー #1478 計画の F2（構造的到達性の事前調査）: `autodiff`／
`facade`／`bench-fandhe` のいずれからも `tensor_core::buffer::
MemoryOps::with_host_view` の呼び出しは 0 件（`grep` 確認済み）。
`Var::matmul` の出力は `gemm.rs::run_f32_kernel` 内部の `readback`
（`memory.rs::readback`。イシュー #1437 で `ReadbackDest::
PretouchedFresh` へ切替済み・`host_staging` とは別経路）で既に
ホスト常駐化されるため、`gemm --device cuda --mode reuse` の計測
コード経路は `HOST_STAGING_KIND` の値に依存しない。よってゲート C は
「本変更が本番経路を壊していないこと」を確認する非後退ガードであり、
性能改善を探しに行くものではない（想定結果は「差なし」）。

## 実施範囲（時間制約による縮小）

計画（`docs/perf/cuda-host-staging-pinned-default-1478/README.md` 骨子）
は `run_gemm_gate_cuda.sh` による N=1024/2048/4096 の 5 回計測中央値
（bench-fandhe／bench-candle 交互実行）を想定していたが、本イシュー
セッションの時間制約により、以下の縮小スコープで代替した:

- `run_gemm_gate_cuda.sh`（5 回計測・candle 併走・fail-closed manifest
  検証）は実行せず、`bench-fandhe` バイナリを before／after 両ツリーで
  直接ビルドし、`--task gemm --device cuda --size <N> --mode reuse` を
  **各 1 回**実行して比較した（5 回計測中央値ではない）。
- before: `~/work/rust-ai-library-run-1478-base/`（コミット `e8cd3a2`。
  `perf/1478-host-staging-pinned-default` の分岐元 = origin/main HEAD。
  `crates/facade` へ path patch）
- after: `~/work/rust-ai-library-run-1478/`（コミット
  `0fda7bcbd1c59f8eb960872de6c8255e398eec53`。本イシューの test コミット
  まで反映。同じく path patch）
- 両ビルドとも `patch.crates-io.fandhe-ai.path=<各ツリー>/crates/facade`
  を `--config` で指定（`bench_fandhe_pin_guard.sh` が要求する借用ビュー
  readout 対応のため。イシュー #1438 参照）。

## 実測結果（1 回計測。GB10・2026-09-09・GPU utilization 0%・常駐サービス
以外の GPU プロセスなし）

| N | before median_s | after median_s | ratio (after/before) | checksum 一致 | parity_fail_count |
|---|---|---|---|---|---|
| 1024 | 0.002195842 | 0.002176947 | 0.9914 | 一致（-1855.597736） | 0（両方） |
| 2048 | 0.008702434 | 0.008346656 | 0.9591 | 一致（-6016.774008） | 0（両方） |
| 4096 | 0.038974362 | 0.038511928 | 0.9881 | 一致（-25768.747284） | 0（両方） |

全 N で `checksum` が完全一致（F2 の構造的非到達を実測で裏付け）・
`parity_fail_count` 0（両方）・timing は誤差範囲内で非後退（比 0.96〜0.99。
1 回計測のためノイズを含むが、悪化方向の兆候はない）。

## 結論

F2（構造的非到達）と本実測（checksum 完全一致・非後退）の両方から、
`HOST_STAGING_KIND` の既定切替（イシュー #1478）は `gemm cuda reuse`
の本番経路に影響しないことを確認した。5 回計測中央値による正式な
`run_gemm_gate_cuda.sh` ゲート再計測は本イシューでは実施しておらず、
必要であれば別イシューで追加できる（対象外事項。§ `docs/perf/
cuda-host-view-staging-readout.md` §8 参照）。
