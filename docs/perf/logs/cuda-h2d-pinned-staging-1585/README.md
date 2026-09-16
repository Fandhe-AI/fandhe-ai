# イシュー #1585 実測記録（CUDA H2D pinned staging・ゲート A）

## 位置づけ

`docs/perf/cuda-h2d-pinned-staging.md` §3 の事前登録判定規則のうち、
**ゲート A（`#[ignore]` 実機テスト全件 pass）のみ**を DGX Spark GB10 実機で
2026-09-16 に実行した記録。Layer A／Layer B は未実施であり、verdict は
**undetermined のまま**・既定 OFF（`PINNED_H2D_ENABLED=false`）を維持する。

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

## 未実施（本セッションでは実行不能だったもの）

- **Layer A（framework-compare 非後退）**: `scripts/bench/framework-compare/`
  に pinned-h2d 版 `run_ab_*.sh` がなく、`--pinned-h2d` フラグを持つベンチも
  存在しないため実行不能。
- **Layer B（H2D 単体マイクロ A/B）**: `pinned_staged/pageable` を N ごとに
  5 プロセス起動で計測するハーネスが未実装のため実行不能。

いずれも本セッションでは新規実装しない（実装コード・ハーネスは触らない方針）。

## 再開条件

Layer A／B のハーネス実装（framework-compare `--pinned-h2d`・H2D 単体
マイクロベンチ）は別イシューとして起票し、実装後に §3 の事前登録規則
（off→on 交互 5 run・中央値比・checksum 完全一致）に従って GB10 で再計測する。
ゲート A は本記録で pass 済みのため、再計測時は Layer A／B のみで判定を確定
できる。

## Layer B（H2D 単体マイクロ A/B。イシュー #1585 本セッションで追加）

`docs/perf/cuda-h2d-pinned-staging.md` §3 の事前登録判定規則
「H2D 単体（発行＋`synchronize`）で N ごとに 5 プロセス起動中央値の
`pinned_staged/pageable`。`< 1.00` を改善、`>= 1.00` を非改善として
記録」を実行するハーネスを実装した（本追記時点ではハーネスの実装のみ
で、GB10 実機実測は未実施のまま次セッションへ引き継ぐ）。

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
`uptime_before.txt`・`uptime_after.txt`（本追記時点ではいずれも未生成。
実機実測を実施したセッションで追加する）。

### 未実施（本セッションでは実機なしのため実行不能）

- Layer B の GB10 実機実測自体（`run_layer_b.sh` の実行）。
- Layer A（framework-compare 非後退）は README 冒頭のとおり別途
  ハーネス実装が必要（本セッションのスコープ外）。
