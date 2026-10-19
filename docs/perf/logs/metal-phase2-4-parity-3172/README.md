# Phase 2〜4 追加 8 機能の Metal 実機 parity（イシュー #3172）

状態: 実測済み。**12 本中 pass 12・fail 0**（paged attention は backend・facade とも 5 回すべて pass）。
実行日 2026-10-19（UTC 18:11〜18:12）。実行コミット `665370196b6d2477770d843e7d976f2f54f9973b`（`rev-stamp.txt` は `dirty=0`）。

対象は 3D プーリング（#3029）・ConvTranspose3d（#3033）・MaxUnpool（#3037）・Fold／Unfold（#3041）・
LocalResponseNorm（#3045）・複数入力 jvp（#3064）・vmap(grad)（#3073）・paged attention（#3113）の
Metal `#[ignore]` parity テスト 12 本を Mac mini（Apple M4）で測ること。CUDA 側は #3171 で GB10 実測済み。

## 前回測定の不採用と再測定

前回測定（PR #3474）は結果が全 pass だったが、ホスト名のマスクが境界なしの部分文字列置換で LRN のテスト名の一部を壊し、
回収後のログを手で直していたため、所有者判断で不採用になった（未マージで close）。今回は手順欠陥（記録の完全性）が
理由の再測定であり、結果による選別ではない。判定・入力・コマンドの判定部分は前回と同じで、今回の結果は
pass／fail にかかわらず採用する。前回のログは参照のみで本提出物には流用しない。

## 是正（run_metal_parity.sh）

- ホスト名（長・短）にもユーザー名と同じ前後境界（英数字・`_`・`.`・`-` 以外）を課す。
- 本実行の冒頭で実値（hostname・hostname -s・USER・HOME）によるマスク自己検査を行い、失敗なら cargo を起動しない。
- 事後に、マスク後の各ログに期待するテスト名の `test <名前> ... ` 行が逐語で残ることを検査する。無ければ exit 2。
- 回収ログは手で編集しない。漏れ・化けはスクリプトを直し、新コミットで事前登録し直して測り直す。
- `--self-test` に、短縮ホスト名がテスト名の部分文字列になりうる一般語（local・response・norm）のケースを含める。

## 事前登録

`RULE.txt`（実測前にコミットして固定）。判定は REQ-2 統一複合判定で tolerance・baseline・テストコードは変えない。
paged attention（9〜12）の判定は既存の `../paged-attention-parity-3113/RULE.txt` に従い、5 回すべて pass で pass、
`PAGED_PARITY` 行は 5 回の中央値を記録する。

宣言日時の訂正: `RULE.txt` の宣言日時 `2026-10-19T18:20Z` は作成時の見積り誤りで、実際の事前登録コミット
`665370196`（RULE.txt・本スクリプト・本 README を含む）の作成は 2026-10-19T18:10:41Z（`git log` の committer 日時。JST 2026-10-20 03:10:41）で、
実機実行（`env_info.txt` の `date_utc` 18:11Z）より前である。RULE.txt は事前登録の原本として書き換えていない。

## 結果

| # | 機能（ディレクトリ） | テスト | 回数 | 結果 | 備考 |
|---|---|---|---|---|---|
| 1 | `compat-sequential-pool3d-3029` | `metal_pool3d_forward_backward_matches_cpu` | 1 | pass | `running 1 test`・`1 passed` |
| 2 | `compat-sequential-conv-transpose3d-3033` | `metal_conv_transpose3d_forward_backward_matches_cpu` | 1 | pass | 同上 |
| 3 | `compat-sequential-max-unpool-3037` | `metal_functional_max_unpool_2d_forward_backward_matches_cpu` | 1 | pass | 同上 |
| 4 | `compat-sequential-max-unpool-3037` | `metal_functional_max_unpool_3d_forward_backward_matches_cpu` | 1 | pass | 同上 |
| 5 | `compat-sequential-fold-unfold-3041` | `metal_fold_unfold_forward_backward_matches_cpu` | 1 | pass | 同上 |
| 6 | `compat-sequential-local-response-norm-3045` | `metal_local_response_norm_forward_backward_matches_cpu` | 1 | pass | 同上。テスト名はマスクで壊れていない |
| 7 | `functional-transforms-jvp-multi-3064` | `metal_jvp_multi_matches_cpu_reference` | 1 | pass | 同上 |
| 8 | `functional-transforms-vmap-grad-3073` | `metal_vmap_grad_matches_cpu_reference` | 1 | pass | `--exact`。同居テストは `16 filtered out` |
| 9〜11 | `paged-attention-parity-3113`（backend） | `metal_paged_attention_matrix_matches_cpu_reference`・`..._tracked_matrix_matches_cpu_reference`・`..._tolerance_constants_pinned` | 5 | 5/5 回 pass | 毎回 `running 3 tests`・`3 passed` |
| 12 | `paged-attention-parity-3113`（facade） | `metal_paged_attention_forward_matches_cpu_tape` | 5 | 5/5 回 pass | 毎回 `running 1 test`・`1 passed` |

- `summary.tsv` は 18 行（単発 8 + paged 10 run）+ ヘッダで、全行の終了コードが 0、`verdict=pass`。スクリプトの終了コードは 0
  （`any_fail=0 any_undetermined=0 leak=0 ident_fail=0 summary_bad=0 median_rc=0`）。
- paged の `PAGED_PARITY` 行は 5 回すべてで 136 行（`metal` 68・`metal-tracked` 68）が完全一致し、fail は全行 0・`judgement=strict`。
  5 回中央値は `../paged-attention-parity-3113/metal/paged_parity_median.tsv`。`BASELINES` は 0 件のまま。
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance 定数・baseline・テストコード・入力は変更していない。
  対象 9 ファイルに bit 一致を求めるアサートは無い。
- 各テストの実行時間は 0.03〜0.10 秒と短く、LRN・Pool3d などはホストフォールバック経路の可能性がある。GPU カーネル経由かは本測定からは断定しない（追従課題）。
- 実行前の load average（1 分値）は 6.62、実行後は 11.49（`env_info.txt`。本実行自身のビルド・テストを含む）。実行前に他の cargo／rustc が無いことを `pgrep` で確認した（並列度 1）。時間計測ではなく pass／fail の判定のため結果には影響しない。

## 成果物と検証

| 置き場所 | 内容 |
|---|---|
| 本ディレクトリ | `RULE.txt`・`run_metal_parity.sh`・`README.md`・`rev-stamp.txt`・`env_info.txt`（実行前後・`script_sha256`）・`build.log`（判定外）・`summary.tsv`・`SHA256SUMS` |
| `../<機能ディレクトリ>/metal/` | テストごとのログ（末尾に `rc=0`）と `rev-stamp.txt` の写し。paged は `*.run{1..5}.log` と `paged_parity_median.tsv` |

`SHA256SUMS` は実行出力ディレクトリ基準のパス（`logs/<機能>/<ファイル>`）で書かれている。配置後は次で無改変を検証できる
（本ディレクトリで実行。全 23 ファイルが `OK`）。

```
sed -E 's|  logs/([^/]+)/|  ../\1/metal/|' SHA256SUMS | shasum -a 256 -c
```

`env_info.txt` の `script_sha256` は実行時の版（コミット `66537019`）の `run_metal_parity.sh` のハッシュである。
その後 `711430ce`（leak_check のホスト名検査の修正）でスクリプトが変更されたため、現在のファイルのハッシュとは一致しない
（`env_info.txt` の記録は変更しない）。実行時の版は次で照合する。

```
git show 66537019:docs/perf/logs/metal-phase2-4-parity-3172/run_metal_parity.sh | shasum -a 256
```

## 申し送り（本件の対象外）

- GPU カーネル経路かホストフォールバックかの判別（追従課題）。
- 各機能の決定記録・`site/` の「Metal 実機未実測」表記の更新。
- 兄弟の `../metal-facade-parity-3166/`・`../metal-autodiff-parity-3169/`（別 PR）のスクリプトにも、ホスト名を境界なしの部分文字列で置換する同種の実装が残りうる。本件では触らない。
