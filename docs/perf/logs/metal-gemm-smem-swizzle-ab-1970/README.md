# 協調ロード XOR swizzle 軸（イシュー #1970）実機実測 README

機構実装は `crates/backend-metal/src/{tile.rs, gemm.rs, pipeline.rs,
spec_source.rs, shaders/gemm.metal}`。Linux 実行可能な検証（型検査・
Rust 側置換モデル・シェーダソース証跡テスト）は本実装セッション
（Linux 実装環境）で完了済み。**実機〈Apple Silicon〉での bit 一致実行・
kernel_gpu 5 run A/B は本 PR 時点で未実施**（本ディレクトリは実測ログの
置き場のみで、成果物自体はこの README・スキャフォールドのみ）。

本 README・`orchestrate.sh`・`aggregate.py` は Mac セッションが実測を
行う際の手順書を兼ねる。

## 実行手順

1. worktree を最新化し、`cargo build -p fandhe-ai-backend-metal --release`
   が通ることを確認する（Apple Silicon・macOS）。
2. 前提ゲート（AC-1）:
   ```sh
   sh docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh gate
   ```
   `crates/backend-metal/src/gemm.rs` の `smem_swizzle_bit_match_*` 6 本
   （新規）+ `coop_load_bit_match_*` 6 本（既存。非後退確認）を実行する。
   1 件でも FAIL なら打ち切り、正しさ不成立として当該 head は REJECT を
   確定し性能 A/B は実施しない（是正せず記録のみ。事前登録判定規則 1）。
3. 性能 A/B（AC-2。record_only・専有ゲートなし）を 5 回、run 番号
   1〜5 で順に実行する:
   ```sh
   sh docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh 1
   sh docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh 2
   sh docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh 3
   sh docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh 4
   sh docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/orchestrate.sh 5
   ```
   各 run は `crates/backend-metal/src/gemm_smem_swizzle_diag_tests.rs::
   xor_swizzle_kernel_gpu_ab_production_sizes` を 1 回実行し、
   `kernel_gpu_run<N>.log`（本体出力）・`uptime_before_run<N>.txt`・
   `pmset_therm_{before,after}_run<N>.txt`・`run<N>_monitor.log`（10 秒
   間隔の負荷推移サンプラー）・`run<N>_procs.txt`（固定 watchlist の
   件数のみ。内部情報は記録しない）を保存する。
4. 集計:
   ```sh
   python3 docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/aggregate.py
   ```
   `aggregate.md` を生成する（判定規則は `aggregate.py` 冒頭・`docs/perf/
   metal-gemm-coop-load-candidates.md` §7・`docs/perf/
   metal-gemm-n4096-kernel-gap.md` の該当節・issue #1970 コメントを正とし
   本 README では書き写さない）。
5. `docs/perf/metal-gemm-coop-load-candidates.md` §7・`docs/perf/
   metal-gemm-n4096-kernel-gap.md` の実機実測記入欄を `aggregate.md` の
   結果で埋める（コード変更は伴わない事実転記のみ）。

## 保存すべきファイル一覧

- `gate_run.log`
- `kernel_gpu_run{1,2,3,4,5}.log`
- `uptime_before_run{1,2,3,4,5}.txt`
- `pmset_therm_{before,after}_run{1,2,3,4,5}.txt`
- `run{1,2,3,4,5}_monitor.log`
- `run{1,2,3,4,5}_procs.txt`
- `env_info.txt`（記入欄。内部ホスト名は含めない）
- `aggregate.md`（`aggregate.py` の生成物）

## 対象 arm（7 arm。事前登録・issue コメント固定）

`L0-P4-S0`（本番既定・base）・`L0-P0-S0`（対照）・`L0-P4-S1`・
`L0-P0-S1`・`L0-P8-S1`・`L0-P4-S2`・`L0-P0-S2`。`L1-*`
（`CoopLoadLayout::RowStrided`）は #1300 で REJECT 済みのため対象外
（bit 一致は `smem_swizzle_bit_match_*` 側が全 6 協調ロード候補 ×
`{ATile, BothTiles}` の 12 head を被覆する）。

## 注意

- 内部ホスト名・ユーザー名・絶対パス（`env_info.txt` 以外）は記録しない。
- いずれの判定でも本番既定（`tile::COOP_LOAD_CONFIG`・`tile::
  SMEM_SWIZZLE`・`tile::select*`・`MetalGemm::new`）は変更しない
  （事前登録判定規則 6）。本番結線は別イシュー・ユーザー承認事項。
