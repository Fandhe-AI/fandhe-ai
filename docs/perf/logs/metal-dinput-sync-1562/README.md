# `d_input` 経路の同期境界・回収余地の定量化（イシュー #1562）実測ログ

`Op::LinearResident` の VJP が `d_input` を求める際に呼ぶ
`MetalBackendOps::gemm_resident_lhs`（`docs/backend-metal-command-batching-
design.md` §7.3）が backward フェーズ単独にどれだけの同期境界・壁時間を
占めるかを定量化するログ置き場。

**現状（本 PR 時点）**: このディレクトリはまだ空。本 PR を書いた実行環境に
Apple Silicon 実機へのアクセス経路がないため、下記の実測は未実施のまま
記入欄として残す（§7.3.4 の記入欄と対応）。Mac セッションで
`orchestrate.sh` を実行し、生成物をこのディレクトリへ収める。

## 実行手順

```sh
cd docs/perf/logs/metal-dinput-sync-1562
sh orchestrate.sh
# ドライラン（コマンド列の確認のみ・実行しない）:
sh orchestrate.sh --dry-run
```

## 生成物

- `backward_phase.log`: 方針 A（`mnist_scale_train_reuse_metal_backward_
  dinput_phase`）の `cargo test -p fandhe-ai --release --test
  mnist_scale_train_reuse_bench -- --ignored --nocapture
  mnist_scale_train_reuse_metal_backward_dinput_phase` 全出力
- `resident_lhs_phase.log`: 方針 B（`resident_lhs_dinput_phase_bench`）の
  `cargo test -p fandhe-ai-backend-metal --release --test
  resident_lhs_dinput_phase_bench -- --ignored --nocapture` 全出力
- `resident_lhs_phase_gpu_timestamps.log`: 同上を `--features
  internal-diagnostics` 付きで再実行した出力（`kernel_gpu` 内訳込み）
- `env_info.txt`: `uname -srm`・`sw_vers`・`rustc -V`・
  `sysctl machdep.cpu.brand_string`・実行コミットの `git rev-parse HEAD`
  （内部ホスト名は含めない）
- `uptime_before.txt`／`uptime_after.txt`: 実行前後の 1 回計測（record_only
  運用。専有ゲートは課さない——本イシューは相対比較〈Variant A vs B〉が
  主目的で、共有負荷は両 Variant に対称に乗るため個別の絶対値ほど負荷に
  敏感ではないと想定するが、参考のため記録する）

## 事前登録判定規則（record only・non-gating。計測後に緩和しない）

本イシューは受け入れ条件が「コード変更は無しでもよい」測定タスクであり、
以下は合否判定ではなく `docs/backend-metal-command-batching-design.md`
§7.3.4 への転記手順を確定するための事前宣言:

1. 実行コマンドは `orchestrate.sh` のもの（release ビルド必須。debug
   ビルドは GEMM 自体の絶対値が意味を持たないほど遅くなる）
2. `backward_phase.log` から `encode_deltas`／`command_buffer_deltas`／
   `wait_deltas`（5 trial 分）と `backward-only median/q1/q3` を抽出し、
   §7.3.1 の事前登録仮説（`encode_delta=4`・`command_buffer_delta=
   wait_delta=2`）と一致するか確認する。不一致の場合は乖離をそのまま
   記録し（record only のため隠さない）、原因調査を新規イシューへ引き継ぐ
3. `resident_lhs_phase.log`／`resident_lhs_phase_gpu_timestamps.log` から
   L1・L2 それぞれの `variant_a` 4 区間・`variant_b` encode_only・
   `recoverable_upper_bound` を抽出し §7.3.4 の表へ転記する
4. 共有負荷下で可（専有ゲート不要。相対比較〈A vs B〉が主目的のため）。
   `uptime_before.txt`／`uptime_after.txt` の load average を
   §7.3.4 へ転記する

## 関連

- `docs/backend-metal-command-batching-design.md` §7.2（#1555 引き継ぎ元）・
  §7.3（本イシューの調査結果・比較表・実測記入欄）
- `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（回収案 (c) の設計）
- `docs/perf/linear-forward-device-gpu.md`（回収案 (b) が参照する
  `linear_forward_device` の先例）
- 親イシュー #1557 → #1561 → 本イシュー #1562（測定）→ #1563（実装・
  前後比較。次の子イシュー）
