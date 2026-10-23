# 学習ステップ常駐化の部品の実機 parity 記録（イシュー #3224）

## 状態

| 機体 | 状態 |
|---|---|
| Mac mini（Apple M4） | **実測済み**（`e62cce21b`・2026-10-23・M1〜M5 x 5 run 全 pass・記録は `macmini/`） |
| DGX Spark GB10（sm_121） | **実測済み・C5 が fail**（`45347ae0a`・2026-10-23・C1〜C4 x 5 run は全 pass、C5 は 5 run とも 2/4 fail。記録は `gb10/`。下記「DGX Spark GB10」「不一致」） |
| 開発機（Apple M4 Max）での試走（記録ではない・1 回のみ） | 下記「試走で分かったこと」 |

事前登録規則は `RULE.txt`（2026-10-23 固定。実測値を 1 件も含まない）。実測を取るときは、規則を変えず、
`stage-tree.sh`（`../metal-infer-bimodal-macmini-3183/stage-tree.sh`）で `git archive` 展開して
`.rev-stamp`・`.tree-manifest` つきで機体へ転送し、ノード側で `run_parity.sh` を実行する。

## 目的

学習 1 step を同期 1 回にする常駐チェーン（#3248・#3251・#3252）の部品 6 個（Metal 3・CUDA 3）について、
(1) 既存の比較先と bit 一致（NaN はクラス一致）すること、(2) 新版が wait／同期を出さないこと
（実機カウンタ）を、機体・版つきで記録する。比較先は設計 `docs/train-step-resident-parts-design.md` §4。
tolerance・parity baseline・判定式は変えない。

## 部品と対象テスト

MLP は 784→256→10・batch 64（形状の定義は `crates/backend-cpu/src/gemm_blis/jobgrid_probe.rs` の `SHAPES`）。
l1_fwd／l2_fwd（NN）は forward GEMM で本イシューの 6 部品に入らない。

| 部品 | MLP 形状 | 比較先 | Metal | CUDA |
|---|---|---|---|---|
| ReLU マスク | L1 出力 `[64, 256]`（RowMajor／Transposed） | `elementwise_mul_mask` 同一式 | M1・M4 | C1・C5 |
| MSE forward／backward | 出力 `[64, 10]`（640 要素） | 同バックエンドのホスト版 `mse_loss`／`mse_loss_backward` | M2・M4 | C2・C5 |
| dW（TN・bias あり） | `(B,in,out)` = (64,784,256)・(64,256,10) | `gemm_fp32_strict_into_with_bias_reduce_tracked` | M3・M4 | C3・C4・C5 |
| d_input（`Transposed`） | 同上 | `gemm_resident_lhs` | M3・M4 | C3・C4・C5 |

テスト識別子（M1〜M5・C1〜C5）は `RULE.txt` を正とする。M4（`train_step_parts_no_wait_3224`）と
C5（`train_step_parts_sync_diag_tests_3224`）は backward 部分列を `_tracked` 版だけで連結した区間で
wait／同期が 0 であることと、各部品の出力がホスト版の連鎖結果と bit 一致することを確かめる。

カウンタの範囲（CUDA）: `diag_count_h2d` は `gemm.rs` の H2D だけを数え、ReLU・MSE のホスト版の
`clone_htod` は数えない。新版は入力が `DeviceBuffer` なので構造上 H2D がなく、確認は
`stream_syncs`／`d2h_calls` = 0 とソース上 `clone_htod`・`readback` を呼ばないこと（設計 §3）による。

## 手順

1. `stage-tree.sh <commit> <dir>` で展開 → 機体へ rsync（`.git`・`.env*`・`real-hardware-verification-env.local.md` を除外）→
   ノード側で `.rev-stamp` を読み `.tree-manifest` を `shasum -a 256 -c` で照合（不一致なら中止）。
2. ビルドを 1 回だけ実施（`build.log`。判定外）: `cargo build --release --tests -p <crate>`。
3. `./run_parity.sh <metal|cuda> <outdir>`（独立プロセス x 5 run。bash 3.2 互換。macOS は `nohup` サブシェルで切り離す）。
4. ログ・env・`.rev-stamp` を回収し、ホスト名・ユーザー名・絶対パスを `<metal-node>`・`<cuda-node>`・`<home>` へ置換。
5. `python3 -I summarize.py <metal|cuda> <dir>` で `summary.md` を生成（欠損・不一致は非 0 終了）。
6. 本 README の下記記入欄を埋める。

## 試走で分かったこと（開発機 M4 Max・1 回・記録ではない）

事前登録の対象外（正式記録は Mac mini）だが、規則を決める前に判明した事実として残す。

- `train_step_parts_no_wait_3224`（2 テスト）・`mse_device_parity`（5）・`gemm_device_input_parity`（6。
  `(64,784,256)` 追加後）は pass。tracked 連結区間の `wait_until_completed` 差は 0。
- `relu_mask_device_parity` の `denormal_mask_selects_like_cpu_reference` は **fail**。
  マスクが正の非正規化数（`1e-40`）のとき、CPU 参照は `m > 0.0` が真（g を通す）だが、Metal は 0 を返した
  （`idx 0: 0 vs -11`）。GPU 側で非正規化数が 0 へ flush されている（FTZ）とみられる。設計 §3 の
  「正の非正規化数の mask は CPU と同じく真になる」は CUDA（#3221）の記述で、Metal では成立しない。
  - 分類: 非正規化数の扱い（ftz）。戻し先: **#3218**（Metal ReLU マスク）。#3218 でマスクのビット判定へ修正済み（下記「#3218 修正後の確認」）。
  - 他の ReLU テスト（`(64,256)` 追加後の RowMajor／Transposed・tracked wait 0）は pass。
  - 本テストは共有 `pattern` に混ぜず専用テストに分けた（他形状の判定を巻き込まないため）。
    tolerance・判定式は変えていない。子 issue の reopen・コメントは未実施（ユーザー承認待ち）。

## 実測記入欄

### Mac mini（Apple M4）

記録: `macmini/`（`.rev-stamp`・`tree-manifest-check.txt`・`env_info.txt`・`build.log`・`<name>.run{1..5}.log`・`DONE`・`summary.md`）。
ホスト名・ユーザー名・絶対パスは `<metal-node>`・`<user>`・`<home>` へ置換済み。

- 日付: 2026-10-23。計測コード: `e62cce21b7001327f0e5807e6539b863f80fdbd1`（main。#3218 の FTZ 修正〈PR #3548〉を含む）。
  `dirty=0`（`tree-manifest-check.txt` 11894 件が全件 OK）。crate 版 0.11.0（workspace）。
- 環境: Apple M4・macOS 27.0.1・rustc 1.99.0（b940084d7 2026-09-28）。`cargo build --release --tests -p fandhe-ai-backend-metal` は `build_rc=0`（判定外）。
- 実行: `run_parity.sh metal`（独立プロセス x 5 run）。foreign の cargo／rustc 検知なし。invalid 0 件・再実行 0 件。

| テスト | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|
| M1 relu_mask_device_parity | pass 6/6 | pass 6/6 | pass 6/6 | pass 6/6 | pass 6/6 |
| M2 mse_device_parity | pass 5/5 | pass 5/5 | pass 5/5 | pass 5/5 | pass 5/5 |
| M3 gemm_device_input_parity | pass 6/6 | pass 6/6 | pass 6/6 | pass 6/6 | pass 6/6 |
| M4 train_step_parts_no_wait_3224 | pass 2/2 | pass 2/2 | pass 2/2 | pass 2/2 | pass 2/2 |
| M5 gemm_device_tracked_token | pass 1/1 | pass 1/1 | pass 1/1 | pass 1/1 | pass 1/1 |

総合: pass（`summary.md`。`summarize.py` は期待件数 6/5/6/2/1 を厳密検査して 0 終了）。

カウンタ: カウンタ値はログへ出力されず、テスト内の assert が検査する。pass はその assert の成立を意味する。
tracked 連結区間の `wait_until_completed` 差 = 0 は `crates/backend-metal/tests/train_step_parts_no_wait_3224.rs:206`、
各部品の出力のビット一致（NaN はクラス一致）は同ファイル `assert_bits_or_nan_class`（:52）。
tracked 版の失敗トークン登録と非 wait は `crates/backend-metal/src/ops.rs` の
`gemm_device_tracked_registers_token_and_does_not_wait`（M5）。

6 部品の判定（Metal）: 6 部品すべて pass（bit 一致・wait 0）。

| 部品 | Metal | CUDA |
|---|---|---|
| ReLU マスク | pass（M1・M4） | 未実測 |
| MSE forward／backward | pass（M2・M4） | 未実測 |
| dW（TN・bias あり） | pass（M3・M4） | 未実測 |
| d_input（Transposed） | pass（M3・M4） | 未実測 |

`macmini-3218-fix/` との関係: 同ディレクトリは PR ブランチ上・マージ前の `af2c2ada` での #3218 修正の確認記録である。
`macmini/` はマージ後の main（`e62cce21b`）での別の計測であり、RULE の「取り直し」ではない。両方とも pass で、どちらの結果も選別・除外していない。
正式記録は `macmini/`。

### DGX Spark GB10

記録: `gb10/`（`.rev-stamp`・`tree-manifest-check.txt`・`env_info.txt`・`build.log`・`<name>.run{1..5}.log`・`DONE`・`summary.md`）。
ホスト名・ユーザー名・絶対パスは `<cuda-node>`・`<user>`・`<home>` へ置換済み。

- 経緯: 先に使った GB10（dbd9）は常駐 vLLM が許可リストに無く中止した（結果の選別ではない）。所有者が 2026-10-23 に指定した別の GB10 ノード
  （dbd9 とは別。FQDN は記載しない）で取り直した。計測前の `nvidia-smi --query-compute-apps` は 0 件・`utilization.gpu` 0 %。
  `PARITY_RESIDENT_APP_REGEX` は未設定。実機ロック（`~/work/.fandhe-hw-lock`）を取得して実施し、回収後に解放した。
- 日付: 2026-10-23。計測コード: `45347ae0a5f3ca436b8136424a136d660c5a5280`（main。#3227 の readback 先読み既定 ON〈PR #3552〉を含む）。
  `dirty=0`（`tree-manifest-check.txt` 11998 件が全件 OK・`.env*`／`settings.local.json`／`*.local.md` の混入 0 件）。crate 版 0.11.0（workspace）。
- Metal 記録との関係: `e62cce21b..45347ae0a` の `crates/backend-metal`・`crates/backend-cpu`・`crates/tensor-core` の差分は空
  （`git diff --stat` の出力が空。Metal の部品は `macmini/` の計測時から無変更）。
- 環境: GB10（sm_121）・driver 580.173.02・CUDA 13.0 V13.0.88・rustc 1.97.1（8bab26f4f 2026-07-14）・aarch64 Linux。
  `cargo build --release --tests -p fandhe-ai-backend-cuda` は `build_rc=0`（判定外）。
- 実行: `run_parity.sh cuda`（独立プロセス x 5 run・`run_parity rc=0`）。invalid 0 件・再実行 0 件。

| テスト | run1 | run2 | run3 | run4 | run5 |
|---|---|---|---|---|---|
| C1 relu_mask_device_parity | pass 5/5 | pass 5/5 | pass 5/5 | pass 5/5 | pass 5/5 |
| C2 mse_device_parity | pass 5/5 | pass 5/5 | pass 5/5 | pass 5/5 | pass 5/5 |
| C3 gemm_device_input_parity | pass 7/7 | pass 7/7 | pass 7/7 | pass 7/7 | pass 7/7 |
| C4 gemm_device_input_diag_tests_3223 | pass 1/1 | pass 1/1 | pass 1/1 | pass 1/1 | pass 1/1 |
| C5 train_step_parts_sync_diag_tests_3224 | **fail 2/4** | **fail 2/4** | **fail 2/4** | **fail 2/4** | **fail 2/4** |

総合: **fail**（`summary.md`。`summarize.py` は期待件数 5/5/7/1/4 を厳密検査し、C5 の 2 件 fail で非 0 終了）。

カウンタ: カウンタ値はテスト内の assert が検査する（pass はその成立を意味する）。
`crates/backend-cuda/src/gemm_device_input_diag_tests_3223.rs:97-99`（`h2d_calls`／`stream_syncs`／`d2h_calls` = 0・C4 は pass）、
（以下の行番号は計測対象 `45347ae0a` 時点のもの。是正コミットでの assert 削除・`last_out` 追加により現行コードとはずれている）
`crates/backend-cuda/src/train_step_parts_sync_diag_tests_3224.rs:278-280`・`:377-378`（GEMM・backward 連結。C5 のうち pass 2 件）、
`:140-141`（ReLU）・`:188-189`・`:206-207`（MSE）の「新版は同期 0・D2H 0」assert。対照の assert は `:97-98`・`:221-222`（`>= 1`）。
正の非正規化数マスク（NVRTC の ftz 無効）は C1 の pass（`relu_mask_device_parity.rs:32` の `1e-40`／`-1e-40`）で確認した。

C5 の fail 2 件は**対照 assert**（`download`／ホスト版が D2H を数えること）で、新版の判定 assert ではない。ただし ReLU は対照が layout ループの
最初の周回の末尾にあるため、後続の周回は未実行（下記「不一致」）。

6 部品の判定（CUDA）: GEMM（dW・d_input）は pass（C3・C4・C5 の GEMM テスト）。ReLU・MSE は bit 一致（C1・C2）が pass。
同期ゼロの実機カウンタ（C5）は対照 assert の fail により ReLU・MSE で未完了。

| 部品 | Metal | CUDA |
|---|---|---|
| ReLU マスク | pass（M1・M4） | bit 一致 pass（C1）・C5 fail（対照 assert） |
| MSE forward／backward | pass（M2・M4） | bit 一致 pass（C2）・C5 fail（対照 assert） |
| dW（TN・bias あり） | pass（M3・M4） | pass（C3・C4・C5） |
| d_input（Transposed） | pass（M3・M4） | pass（C3・C4・C5） |

## 不一致

Metal（Mac mini・`macmini/`）の正式記録に不一致はない（M1〜M5 x 5 run 全 pass）。試走（開発機）で出た Metal の正の非正規化数マスクの不一致（部品: ReLU マスク・形状 `[64,256]`・入力: マスク `1e-40`・最初の食い違い: `idx 0: 0 vs -11`・分類: FTZ・戻し先 #3218）は #3218 の修正で解消した。

**CUDA（GB10・`gb10/`・`45347ae0a`）: C5 `train_step_parts_sync_diag_tests_3224` の 2 テストが 5 run とも同一に fail**（bit 不一致ではない）。

| テスト | 食い違い（ログ） | 場所 |
|---|---|---|
| `relu_mask_device_issues_no_sync_no_d2h` | `download must D2H: ... d2h_calls: 0, ... stream_syncs: 1` | `train_step_parts_sync_diag_tests_3224.rs:98`（`assert_download_is_counted`。`45347ae0a` 時点の行番号） |
| `mse_device_issues_no_sync_no_d2h` | `host mse must D2H: ... d2h_calls: 0, ... stream_syncs: 1` | 同 `:222` |

- 部品・形状: ReLU マスク `[64,256]`（RowMajor。対照は最初の周回）・MSE `[64,10]` のホスト版。入力は固定パターン。
- 新版の判定 assert（同期 0・D2H 0・bit 一致）は対照より前にあり通過している。落ちたのは「`download`／ホスト版が `d2h_calls >= 1` を数える」という対照だけ。`stream_syncs` は 1 と数えられている。
- 原因分類（推定。実機で未切り分け）: **カウンタの数え方**。計測コードは #3227（PR #3552・readback 先読み既定 ON）を含む。先読み経路の readback が
  `d2h_calls` を増やさなくなり、対照が前提とした「ホストの readback = D2H を数える」が既定経路で成立しなくなったとみられる。
  新版（device 入力版）の挙動の不一致ではない。`e62cce21b`（先読み導入前）で同テストが pass するかは未確認（本記録は `45347ae0a` の事実のみ）。
- 戻し先: 対照 assert（`45347ae0a` 時点の `:92-99`・`:221-222`。現行コードでは行がずれている）はテスト側の前提で、先読み既定 ON（#3227）との整合を取る修正が要る。
  RULE.txt に従い tolerance・baseline・判定式は変更しない。子 issue の reopen・起票は未実施（ユーザー承認待ち）。
- **是正（レビュー指摘対応・未検証）**: D2H の対照 assert（`assert_download_is_counted`・ホスト版 MSE の対照。`d2h_calls >= 1` と `stream_syncs >= 1`）は**維持**している（検証条件は弱めていない）。
  計数漏れは readback 先読み経路が `d2h_calls` を増やさなかったことが原因のため、計数を `memory::readback_with`（`crates/backend-cuda/src/memory.rs`）の共通出口へ集約して修正した。
  ReLU の対照は layout ループの外へ出し、Transposed 周回の「同期 0・D2H 0」と bit 一致が対照より前に必ず実行される形にした。新版の判定 assert・tolerance・baseline は不変。
  **GB10 での再計測は未実施（未検証）**。C5 を pass とする記録はこの修正後の再計測を待つ。上記の `gb10/` の fail 記録は `45347ae0a` 時点の事実としてそのまま残す。

## #3218 修正後の確認（Mac mini・2026-10-23）

イシュー #3218 の reopen 分（Metal ReLU マスク選択を、マスクの正負をビットで判定する方式へ修正）の確認として、
Mac mini（Apple M4・macOS 27.0.1・rustc 1.99.0）で RULE.txt の M1〜M5 を独立プロセス x 5 run 取得した。
**PR ブランチ上・マージ前のコミット `af2c2adabfa6bdd25d3b64dcc68e3024152f3ac9`**（`dirty=0`・`.tree-manifest` を
`shasum -a 256 -c` で全件照合済み・crate 版 0.11.0）。ログ一式は `macmini-3218-fix/`（`.rev-stamp`・`env_info.txt`・
`build.log`・`summary.md`。ホスト名・ユーザー名・絶対パスは `<metal-node>`・`<user>`・`<home>` へ置換）。

| テスト | 結果（5 run） |
|---|---|
| M1 relu_mask_device_parity | 5/5 pass（各 6 件。`denormal_mask_selects_like_cpu_reference` を含む） |
| M2 mse_device_parity | 5/5 pass（各 5 件） |
| M3 gemm_device_input_parity | 5/5 pass（各 6 件） |
| M4 train_step_parts_no_wait_3224 | 5/5 pass（各 2 件） |
| M5 gemm_device_tracked_token | 5/5 pass（1 件） |

- invalid（退避済み試行）: 0 件。
- M1 の denormal ケース（`1e-40`・`1e-45`・`f32::from_bits(1)`・`0x007F_FFFF`・`f32::MIN_POSITIVE`・`±0`・`+inf`・正負の NaN、
  RowMajor／Transposed）は CPU 参照と bit 一致（NaN はクラス一致）。試走で出た `idx 0: 0 vs -11` は解消した。
- 正式記録としては、マージ後の main で取り直した `macmini/` を採る（本取得分は #3218 修正の確認記録として残す。上記「実測記入欄」参照）。
