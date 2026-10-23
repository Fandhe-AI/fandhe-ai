# 学習ステップ常駐化の部品の実機 parity 記録（イシュー #3224）

## 状態

| 機体 | 状態 |
|---|---|
| Mac mini（Apple M4） | **未実測**（正式記録は未取得） |
| DGX Spark GB10（sm_121） | **未実測**（正式記録は未取得） |
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

## 実測記入欄（未記入）

### Mac mini（Apple M4）

未実測。記入項目: 日付・コミット SHA と `dirty=0`・crate 版・rustc・macOS 版・M1〜M5 x 5 run の表・
カウンタ（tracked の wait 差 0／非 tracked 対照 +1）・invalid の有無と理由・6 部品の判定。

### DGX Spark GB10

未実測。記入項目: 日付・コミット SHA と `dirty=0`・crate 版・rustc・ドライバ／CUDA 版・C1〜C5 x 5 run の表・
カウンタ（`stream_syncs`／`d2h_calls`／`h2d_calls` = 0・対照 >= 1）・invalid の有無と理由・6 部品の判定。

## 不一致

上記「試走で分かったこと」の Metal 非正規化数を除き、正式記録としての不一致は未取得。

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
- 上記「状態」表の Mac mini の正式記録の採否は #3224 側の判断に委ねる（本取得分は版・手順が揃っており、正式記録として使える）。状態表は書き換えない。
