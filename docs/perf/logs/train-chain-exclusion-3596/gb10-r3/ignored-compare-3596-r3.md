# #[ignore] 全件の突き合わせ（イシュー #3596・RULE-r3・受入基準 2）

RULE-r3.txt「#[ignore] 全件の突き合わせ」の 4 コマンドを before（`f7fe6b4df`）→ after（`46a2a43d9`。`crates/` は RULE-r3 の基準 `2794cfba6` と同一）の順に、GB10（aarch64・driver 580.173.02・rustc 1.97.1）で既定環境（`FANDHE_AI_CUDA_GRAPH_*` 未設定。コマンド 3 のみ `FANDHE_AI_CUDA_GRAPH_TRAIN=1`）で実行した。計測（ランナー）とは並走させていない。ホスト名・ホームディレクトリはマスク済み（`<home>`・`<node>`）。ログは `test ` 行・`test result` 行・`Running` 行・panic 行・`rc=` のみを残した抜粋（`--nocapture` の全出力は保存していない）。

## 結果（ログ `ignored/ignored-{before,after}-N.log`）

| コマンド | before | after | 備考 |
|---|---|---|---|
| 1 backend-cuda `--all-features`（`--no-fail-fast`） | 472 ok / 15 failed 行（rc=101） | 473 ok / 15 failed 行（rc=101） | 失敗名は 14 件が共通。差分は下表の各 1 件 |
| 2 `graph_capture_real_device` | 2 passed（rc=0） | 2 passed（rc=0） | 同一 |
| 3 `train_step_graph_diag_tests_3260`（`TRAIN=1`） | 8 passed（rc=0） | **9 passed（rc=0）** | after は G1 を含む 9 件が全件 pass |
| 4 facade `--tests`（`--no-fail-fast`・`--test-threads=1`） | 233 ok（rc=101） | 233 ok（rc=101） | 失敗した `test result` 行は両ツリーとも 3 本で同一（個別の失敗テスト行は無し） |

## 新規 fail（after にだけある FAIL）の分類

| テスト | 単独再実行（after 2 回 → before 2 回） | 分類 |
|---|---|---|
| `resident_input_diag_tests_3250::gemm_device_matches_host_gemm_bitwise` | after 2/2 pass・before 2/2 pass（`ignored/single.log`） | 並走（同一 binary 内の他テスト）による一過性。排他が原因ではない |

before にだけある FAIL は `module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache` 1 件で、両ツリーとも単独 2/2 pass（一過性）。

両ツリーに共通する 14 件（`gemm_tf32_optin_on_matches_cpu_across_shapes`・`mma_f16_k4096_stress`・`resident_input_diag_tests_3250::gemm_device_window_has_no_input_h2d` ほか）は排他導入前から存在する既存の fail（`gemm_device_window_has_no_input_h2d` は両ツリーの単独実行でも 2/2 fail）。

**排他が原因の新規 fail は 0 件**。G1 を含む `train_step_graph_diag_tests_3260` は after で 9 件全件 pass。
