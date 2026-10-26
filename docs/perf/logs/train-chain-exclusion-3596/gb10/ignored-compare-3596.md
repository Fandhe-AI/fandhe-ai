# #[ignore] 全件の突き合わせ（イシュー #3596・受入基準 2）

RULE.txt「#[ignore] 全件の突き合わせ」の 4 コマンドを before（`f7fe6b4df`）→ after（`087e7f6d4`。`crates/` は `d3f38f71b` と同一）の順に、GB10 で既定環境（`FANDHE_AI_CUDA_GRAPH_*` 未設定。コマンド 3 のみ `FANDHE_AI_CUDA_GRAPH_TRAIN=1`）で実行した。計測と並走させていない。
生ログは 2.7 億バイト級（コマンド 4 の `--nocapture` 出力）のため、本ディレクトリの `ignored/*.log` は `test ` 行・`test result` 行・panic 行・`rc=` 行だけを残した抜粋である（元の行数・バイト数は `ignored/_raw-sizes.txt`）。

## 結果（RULE 逐語のコマンド）

| コマンド | before | after | 備考 |
|---|---|---|---|
| 1 backend-cuda `--all-features` | lib: 140 passed / 3 failed（rc=101） | lib: 141 passed / 2 failed（rc=101） | lib の失敗で cargo が打ち切るため統合テストは未実行（下記の補足で補う） |
| 2 `graph_capture_real_device` | 2 passed（rc=0） | 2 passed（rc=0） | 同一 |
| 3 `train_step_graph_diag_tests_3260` | 8 passed（rc=0） | **8 passed / 1 failed（rc=101）** | after は 9 件。失敗は G1（下記） |
| 4 facade `--tests` | 3 件失敗で打ち切り（rc=101） | 同左（rc=101） | 失敗は両ツリーで同じ 3 件（`graph_capture`・`in_process_capture_matches_plain_chain_bit_exact`・`cuda_train_infer_sync_census`）。環境変数前提のテストを一括実行したことによる |

## 補足実行（`--no-fail-fast`。RULE にないコマンドの追加。打ち切りで未実行の対象を補うため）

| コマンド | before | after |
|---|---|---|
| 1 に `--no-fail-fast` | 474 passed / 14 failed | 475 passed / 13 failed |
| 4 に `--no-fail-fast` | 291 passed / 3 targets failed | 291 passed / 3 targets failed（同じ 3 件） |

## after にだけ出た FAIL

| テスト | 分類 | 根拠 |
|---|---|---|
| `train_step_graph_diag_tests_3260::unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（G1） | **未分類・排他の目的に反する失敗（未解決）**。before には存在しない新規テスト | 下記「G1」 |
| `small_shape_matrix_unit_has_no_floor_tflops_record`（補足 1 のみ） | 並走による一過性（フレーク） | after・before とも単独実行 3/3 pass（`single-small-shape-*.log`）。RULE 逐語のコマンド 1 には出ていない |

before にだけ出た FAIL（`init_cost_diag_gemm_new_lru_cold_vs_warm`・`cuda_gemm_new_second_construction_reuses_module_cache`）は共有 LRU の汚染による既存のフレーク（#1107 のメッセージ付き assert）で、after でも片方の実行で出ている。両ツリー共通の失敗（`gemm_device_window_has_no_input_h2d`・TF32/TF32x3/f16 の parity・stress・probe 系）は排他と無関係な既存事項として扱う。

## G1 の詳細（受入基準 1・不合格）

- 失敗内容: 非 capture のスレッド（thread 1）の loss 列が単独実行の参照と異なる（`loss idx N: <値> vs <値>` で、別 step の loss らしい値が読めている）。poison ではなく **loss／結果の取り違え**。
- 再現: 8 回中 7 回失敗（コマンド 3 の 1 回目・単独再実行 5/5・ランナー前提ゲート g2 の 1 回）。ランナーの g1（`--exact` 単独）は 1 回だけ pass。つまり確率的で、30 繰り返しのうちどこかで必ず落ちる頻度。
- 位置づけ: 設計記録 §8.11 の「排他により並行しても bit 一致」が GB10 で成立していない。排他の実装（`d3f38f71b`）で足りない経路が残っている可能性が高いが、原因の特定・修正は本 Issue の実測範囲外（`crates/` は変更しない）。所有者判断事項として報告する。
