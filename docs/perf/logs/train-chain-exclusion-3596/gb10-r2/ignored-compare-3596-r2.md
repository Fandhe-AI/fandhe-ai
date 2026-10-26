# #[ignore] 全件の突き合わせ（イシュー #3596・RULE-r2・受入基準 2）

RULE-r2.txt「#[ignore] 全件の突き合わせ」の 4 コマンドを before（`f7fe6b4df`）→ after（`616be4ff7`。`crates/` は RULE-r2 の基準 `63f85eff2` と同一）の順に、GB10（aarch64・driver 580.173.02・rustc 1.97.1）で既定環境（`FANDHE_AI_CUDA_GRAPH_*` 未設定。コマンド 3 のみ `FANDHE_AI_CUDA_GRAPH_TRAIN=1`）で実行した。計測（ランナー）とは並走させていない。ホスト名・ホームディレクトリはマスク済み（`<home>`）。
コマンド 4 の生ログは 2.7 億バイト級（`--nocapture`）のため、`ignored/ignored-*-4.tail.log`（末尾 3000 バイト）と `--no-fail-fast` の抜粋 `ignored/nff-*-4.log`（`test ` 行・`test result` 行・panic 行・`rc=`）だけを残した。

## 結果（RULE 逐語のコマンド。ログ `ignored/ignored-{before,after}-N.log`）

| コマンド | before | after | 備考 |
|---|---|---|---|
| 1 backend-cuda `--all-features`（`--skip train_graph_shared_workspace_tests_3596` を含む） | lib: 140 passed / 3 failed（rc=101） | lib: 142 passed / 1 failed（rc=101） | lib の失敗で cargo が打ち切るため統合テストは未実行（下記の補足で補う） |
| 2 `graph_capture_real_device` | 2 passed（rc=0） | 2 passed（rc=0） | 同一 |
| 3 `train_step_graph_diag_tests_3260`（`TRAIN=1`） | 8 passed（rc=0） | **9 passed（rc=0）** | after は G1 を含む 9 件が全件 pass（初回 RULE の GB10 実測では G1 が不合格だった） |
| 4 facade `--tests` | 失敗で打ち切り（rc=101） | 同左（rc=101） | 失敗は両ツリーで同じ（`cuda_graph_step_bit_identity::graph_capture` は環境変数 `FANDHE_AI_CUDA_GRAPH_STEP=1` の別プロセス前提）。環境変数前提のテストを一括実行したことによる |

## 補足実行（`--no-fail-fast`。RULE にないコマンドの追加。打ち切りで未実行の対象を補うため。`ignored/nff-*.log`）

| コマンド | before | after |
|---|---|---|
| 1 に `--no-fail-fast` | 473 ok / 14 FAILED（10 targets failed） | 473 ok / 15 FAILED（12 targets failed） |
| 4 に `--no-fail-fast` | 233 ok / 3 targets failed | 233 ok / 3 targets failed（同じ 3 件: `cuda_graph_step_bit_identity`・`cuda_graph_train_step_bit_identity`・`cuda_train_infer_sync_census_3189`） |

## after にだけ出た FAIL（補足実行の差分）

| テスト | 分類 | 根拠 |
|---|---|---|
| `small_shape_matrix_unit_has_no_floor_tflops_record` | 並走による一過性（フレーク） | 単独実行は after・before とも 3/3 pass（`ignored/single.log`）。RULE 逐語のコマンド 1 には出ていない |
| `wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096` | 既存の不安定（性能判定の揺れ） | 単独実行は after 2 fail / 1 pass、before 2 fail / 1 pass（`ignored/single.log`）で両ツリー同率。同種の `wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096` は両ツリーの一括実行で失敗 |

排他が原因と分類した新規 fail は **0 件**。before にだけ出た FAIL は `resident_input_diag_tests_3250::gemm_device_matches_host_gemm_bitwise`（RULE 逐語のコマンド 1 でも before のみ）で、共有状態の揺れによる既存フレークと扱う。ほかに `tensor_core_parity_record`・`int8_mma_probe_exec_selected`・`sm121_isa_probe_exec_selected` など TF32／f16 の parity・stress・probe 系 13 件は両ツリー共通の既存の失敗で、排他と無関係。補足実行のみで after 側の失敗数が 1 多いのは上表の 2 件（並走や性能判定の揺れ）と before 側の 1 件の差による。
