# イシュー #1585 実測記録（CUDA H2D pinned staging・ゲート A／Layer A／Layer B）

## 位置づけ

`docs/perf/cuda-h2d-pinned-staging.md` §3 の事前登録判定規則のうち、
ゲート A（`#[ignore]` 実機テスト全件 pass）・Layer A（framework-compare 非後退）・
Layer B（H2D 単体マイクロ A/B）を DGX Spark GB10 実機で 2026-09-16 に実行した記録。
ゲート A は pass、Layer A は判定 8 セル全て後退、Layer B は全 N 非改善で、verdict は
**REJECT**（`docs/perf/cuda-h2d-pinned-staging.md` §4.2）・既定 OFF
（`PINNED_H2D_ENABLED=false`）は不変・opt-in 実装は維持する。

## 実行したもの

- `gateA.log`: `cargo test -p fandhe-ai-backend-cuda --release --all-features
  --test pinned_h2d_real_device -- --ignored --nocapture --test-threads=1`
  の出力（3 テスト pass・0 fail・rc=0）。
  - `pinned_h2d_sequential_uploads_match_plain_upload_bit_exact`
  - `pinned_h2d_upload_into_partial_update_matches_plain_bit_exact`
  - `pinned_h2d_upload_after_release_staging_is_bit_exact`
- `env_info.txt`: 実測環境（GB10・driver 580.173.02・CUDA 13.0・rustc 1.97.0・
  ツリー 3e43bbd0〈crates/・scripts/ は origin/main 565300e4 と同一〉・負荷は
  record_only 相当）。内部ホスト名は masked。

## 経緯

ゲート A 実測時点（2026-09-16 01:41Z）では Layer A／B のハーネスが本リポジトリに
未実装だったため verdict は undetermined としていた。同日中に両ハーネスを本ブランチで
実装し（下記）、03:15Z〜03:20Z に GB10 で実測して REJECT を確定した。

## Layer B（H2D 単体マイクロ A/B。イシュー #1585 本セッションで追加）

`docs/perf/cuda-h2d-pinned-staging.md` §3 の事前登録判定規則
「H2D 単体（発行＋`synchronize`）で N ごとに 5 プロセス起動中央値の
`pinned_staged/pageable`。`< 1.00` を改善、`>= 1.00` を非改善として
記録」を実行するハーネスを実装し、同日 GB10 で実測した（結果は
`aggregate.md`。全 N 非改善）。

### 実装物

- `crates/backend-cuda/tests/pinned_h2d_upload_ab_1585.rs`（`#[ignore]`・
  `internal-diagnostics` feature 必須）: N∈{1024,2048,4096} の f32 N×N
  テンソルについて、`pageable`（`set_pinned_h2d_enabled(false)`）と
  `pinned_staged`（`true`）の 2 腕で「`MemoryOps::upload` 発行 →
  `CudaDevice::stream().synchronize()`」を 1 反復として計測する
  （`bench_harness::run` の 20/20 warmup+計測・cold 1 回を個別記録）。
  読み戻し結果の bit 同一性（`fold_bits` 畳み込み）・
  `H2dStagingCache` の hit（`h2d_staging_stats()`）も検証する。
- `docs/perf/logs/cuda-h2d-pinned-staging-1585/aggregate.py`
  （python3 標準ライブラリのみ・`--self-test` 付き）: `layer_b_run{1..5}.log`
  から CSV 行を抽出し、N ごとに 5 run の `median_ms` 中央値を取り、
  `pinned_staged/pageable` 比と改善／非改善判定を `aggregate.md` として
  出力する（5 run 揃わない・セル欠落・重複行は fail-closed で失敗する）。
- `docs/perf/logs/cuda-h2d-pinned-staging-1585/run_layer_b.sh`:
  上記テストを 5 プロセス起動し `layer_b_run{1..5}.log` へ記録した後
  `aggregate.py` を呼ぶ。`uptime_before.txt`／`uptime_after.txt` に
  実行前後の負荷を記録する。

### 実行手順（DGX Spark GB10 実機）

```sh
./docs/perf/logs/cuda-h2d-pinned-staging-1585/run_layer_b.sh
```

保存されるファイル: `layer_b_run{1..5}.log`・`aggregate.md`・
`uptime_before.txt`・`uptime_after.txt`（2026-09-16 GB10 実測済み。全 N 非改善）。

### Layer A（framework-compare 非後退。2026-09-16 GB10 実測済み）

実行本体は `scripts/bench/framework-compare/run_ab_pinned_h2d_cuda.sh`
（`AB_PATCH_FACADE_PATH` 必須・専有ゲート既定 ON・`AB_LOAD_GATE_MODE=record_only` で
opt-out）。保存先 `layer_a/`:

- `compare-pinned-h2d-pinned-h2d-1585-{gemm,train,infer}.md`（`.err` は空）
- `results-dgx-pinned-h2d-ab-pinned-h2d-1585-{gemm,train,infer,phases}.jsonl`
- `gate-pinned-h2d-ab-pinned-h2d-1585.log`（専有ゲート 3 サンプル通過）・`skipped-*.log`（空）
- `runner/`（DGX 側の一括ランナー `dgx_run_1585.sh`・`progress.log`・各ログ。内部ホスト名・絶対パスはマスク済み）

結果: 判定 8 セル全て後退（checksum 完全一致）・副次 infer 64 reuse のみ改善方向。
verdict は `docs/perf/cuda-h2d-pinned-staging.md` §4.2 のとおり **REJECT**。
