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
