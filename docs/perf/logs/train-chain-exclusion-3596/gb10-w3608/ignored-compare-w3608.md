# #[ignore] 全件の突き合わせ（イシュー #3608・RULE-w3608・受入基準 2）

RULE-w3608.txt「#[ignore] 全件の突き合わせ」の 4 コマンドを before（`e2660eba0`）→ after（`08902dbf3`。`crates/` は RULE の基準 `7ea459d9d` と同一）の順に、GB10（aarch64・driver 580.173.02・rustc 1.97.0）で既定環境（`FANDHE_AI_CUDA_GRAPH_*` 未設定。コマンド 3 のみ `FANDHE_AI_CUDA_GRAPH_TRAIN=1`）で実行した。計測（ランナー）とは並走させていない（計測前に実施）。ホスト名・ホームディレクトリはマスク済み（`<home>`・`<cuda-node>`）。

ログは `ignored/`。コマンド 4 のログは、ビット列ダンプ（`out[...]`・`....bits =` の行。計 2.6 GB 超）を除いた抜粋で、`test` 行・`test result` 行・panic 行・`rc=` は残してある。

## 実行した内容

- 規則どおりの逐語実行: `ignored/{before,after}-cmd{1,2,3,4}.log`。コマンド 1・4 は最初の失敗する test binary で cargo が打ち切るため、後続の binary が走らない。
- 補助（規則の逐語実行ではない）: コマンド 1・4 を `--no-fail-fast` 付きで再実行した `ignored/{before,after}-cmd{1,4}-nff.log`。全 binary を最後まで走らせて突き合わせるためで、突き合わせの本体はこちらで行った。

## 結果

| コマンド | before | after | 備考 |
|---|---|---|---|
| 1 backend-cuda `--all-features`（逐語・打ち切りあり） | lib: 141 passed / 2 failed（rc=101） | lib: 140 passed / 3 failed（rc=101） | 失敗名は下表 |
| 1 補助（`--no-fail-fast`） | 473 ok（rc=101） | 473 ok（rc=101） | 失敗名は 12 件が共通（before 13 件・after 15 件）。差分は下表（before のみ 1 件・after のみ 3 件） |
| 2 `graph_capture_real_device` | 2 passed（rc=0） | 2 passed（rc=0） | 同一 |
| 3 `train_step_graph_diag_tests_3260`（`TRAIN=1`） | **9 passed**（rc=0） | **9 passed**（rc=0） | RULE の注記「before 8 件・after 9 件」は RULE-r3 から写した古い記述。観測では before・after とも 9 件（`#[test]` も両ツリーで 9 件。judge の `g2-3260` も 9 固定）。RULE は改変せず、観測値を記録する |
| 4 facade `--tests`（逐語・打ち切りあり／補助 `--no-fail-fast`） | 233 ok（補助。rc=101） | 233 ok（補助。rc=101） | 失敗は両ツリーで同じ 3 件（下記）。差分なし |

コマンド 4 の失敗 3 件（両ツリー共通）。いずれも本件と無関係: `cuda_graph_step_bit_identity::graph_capture`・`cuda_graph_train_step_bit_identity::in_process_capture_matches_plain_chain_bit_exact`（`FANDHE_AI_CUDA_GRAPH_TRAIN=1` なしで起動すると自ら拒否する。既定環境での実行は規則どおりで、前提ゲート g4-facade が `TRAIN=1` で pass）・`cuda_train_infer_sync_census_3189::cuda_train_infer_sync_census`。

## 新規 fail（after にだけある FAIL）の分類

| テスト | 単独再実行（after 3 回 → before 3 回） | 分類 |
|---|---|---|
| `module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache` | after 3/3 pass・before 3/3 pass（`ignored/iso-{after,before}-solo-N.log`） | 並走（同一 binary 内の他テストがプロセス全体のモジュールキャッシュのヒット数を動かす。assert は `hits_after_second >= hits_before_second + …`）による一過性。待ち型の入口が原因ではない。suite 内の実行では after の 2 回（逐語・補助）とも落ち、before の 2 回は落ちなかった。前回の RULE-r3 の記録では before 側で同じテストが落ちている（同一 binary の並走起因の揺れ。after は同一 binary 内のテスト数が多い〈filtered out 1426 件対 1416 件〉ため競合が増えた可能性があるが、未確認）。単独では両ツリーとも 3/3 pass で、待ち型の入口が原因とする根拠は無い |
| `wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`（`gemm_wmma_tf32_opt`） | after 2/3 fail・before 1/3 fail（`ignored/iso-{after,before}-perf-solo-N.log`） | 既存の不安定。TFLOPS の大小を比べる性能閾値テストで、before の単独でも落ちる（opt 4.271 対 tiled 5.431 TFLOPS）。単独でも pass と fail が混在する。原因は未調査（環境要因と推定。確定していない） |
| `wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096`（`gemm_wmma_tf32_staged`） | after 1/3 fail・before 1/3 fail | 同上（既存の不安定・環境要因） |

before にだけある FAIL は `resident_input_diag_tests_3250::gemm_device_matches_host_gemm_bitwise` 1 件（RULE-r3 の記録では after 側で出て、単独 pass の一過性と分類した同じテスト）。今回の逐語実行では after 側で落ちたのは `…_window_has_no_input_h2d`（共通の既存 fail）と上記の module_cache。

両ツリーに共通する失敗（`gemm_tf32_optin_on_matches_cpu_across_shapes`・`mma_f16_k4096_stress`・`resident_input_diag_tests_3250::gemm_device_window_has_no_input_h2d` ほか計 12 件）は #3608 以前から存在する既存の fail（RULE-r3 の記録の共通 14 件と同じ種類）。

## 分類の件数

| 分類 | 件数 | テスト |
|---|---|---|
| 本件（待ち型の入口）が原因 | **0** | |
| 既存の不具合 | 2 | `wmma_tf32_opt_exceeds…`・`wmma_tf32_staged_exceeds…`（性能閾値の不安定。before でも再現） |
| 並走による一過性 | 1 | `cuda_gemm_new_second_construction_reuses_module_cache` |
| 環境要因 | 0 | （性能閾値の 2 件は環境要因の疑いを併記。確定していない） |

**本件が原因の新規 fail は 0 件**。G1（`unfenced_…`）を含む `train_step_graph_diag_tests_3260` は after で 9 件全件 pass。

## 限界

- 突き合わせは 1 回ずつの実行。分類は単独 3 回の再実行に基づく。
- `OOC3608` 本体（`out_of_chain_ops_wait_for_foreign_capture_…`）は既定環境で必ず拒否されるため突き合わせから除外した（RULE の注記どおり）。検証は前提ゲート g5（`TRAIN=1`・5 回）で行った。
