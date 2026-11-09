# 診断 (3) の RULE 雛形・集計スクリプト・記録先（#4144）

本ディレクトリは**雛形であり、実測を含まない**。1 位でないセルの診断 (3)（相手の構成を再現した場合に届く比）を
Phase 2 の各 issue（#4146・#4147・#4154 ほか）が同じ規則で測るための、測る前に固定する部品を置く。
式の定義は `docs/perf/loss-attribution-matrix.md` §10.1（F1 = カーネル置換、F2 = fresh 窓の再現）を正とする。

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する RULE の雛形。末尾 `## PARAMS` が機械可読で、数値はそこが唯一の正 |
| `aggregate_diag.py` | 標準ライブラリのみの集計。厳格パース、`--self-test`、`--check-template`、`--fingerprint` |

## 記録先の命名

`docs/perf/logs/diag-0140-<対象>-<issue番号>/`（末尾に issue 番号を置く既存規約に合わせる。`diag-0140-*` で一括に引ける）。

- `<対象>`: 単一セルならセル ID の小文字（例 `m-cpu-trn`）、複数セルなら題材 slug（例 `mlp-macmini`・`mlp-gb10`）。
- 機体を分けるときは配下に `macmini/`・`gb10/` を置く。
- 中身: `RULE.txt`・`README.md`・`aggregate_diag.py`（雛形のコピー）・`round1..N/{gate.txt, values.tsv, raw/}`・`env/`・生成物 `aggregate.md`・`aggregate.tsv`。
- RULE の改訂は `RULE.v2.txt` を別ファイルで置き、旧版を残す。集計は `--rule RULE.v2.txt` で指定する。

## Phase 2 の手順

以下はすべて `python3 -I`（カレントのモジュールを読まない）で実行する。

1. 記録先を作り、`RULE.template.txt` を `RULE.txt` に、`aggregate_diag.py` をそのままコピーする。
2. `python3 -I aggregate_diag.py --fingerprint <tree-dir>` の値と `sha256sum aggregate_diag.py` を PARAMS に埋め、
   `【記入…】` と `<…>` をすべて置換する（`cell:` 行はセルごとに 1 行。`scoreboard_ratio`・`prev_class` は `unreached_0140.tsv` から転記し、`kernel_owner` は測る前に宣言する）。
3. 雛形側で `--check-template`、スクリプト側で `--self-test` を通す。
4. `RULE.txt` を**測る前に単独コミット**して push する。
5. 計測し、`round{i}/` に `gate.txt`・`values.tsv`・`raw/` を置く。
6. `python3 -I aggregate_diag.py --dir <記録先>` で集計する。
7. 判定を `unreached_0140.tsv` に反映するかは、実測した issue 側で別途決める（本 issue では変えない）。

### 入力形式

- `round{i}/gate.txt`: 1 語目が `PASS`／`TIMEOUT`／`INVALID`。2 語目以降は自由記述（load1 の値など）。PASS 以外のラウンドは正式系列から外れ、status=partial（確度は常に低）になる。
- `round{i}/values.tsv`: ヘッダ `cell<TAB>quantity<TAB>value<TAB>clock`。`quantity` は `fandhe_window`・`fandhe_fresh_window`・`opp_window`・`fandhe_pure_kernel`・`opp_pure_kernel`（ms、有限で正）。
  `clock` は純カーネル行で必須（`gpu_timestamp`・`cuda_event`・`wall` など。F1 は両側が同じタグのときだけ計算）、他の行は `-`。
  `formulas` が要る量は必須で、要らない量が混ざっていれば止まる。
- ハーネス固有の出力から `values.tsv` への変換は各 issue の責務。生出力は `raw/` に残す。

### 停止する条件（AggError、終了コード 1）

欠損行・重複行・未知のセル／量・非有限／0／負の値・純カーネル > 窓・F1 の分母 0 以下・round ディレクトリ欠落・PASS が 0 本・
PARAMS のプレースホルダ残存・各桁数の不正・`aggregate_script_sha256` の不一致。

## 公開面の注意

README・RULE・集計出力・コミットに実機ホスト名・絶対ホームパスを書かない。Mac mini は `<metal-node>`、GB10 は `<cuda-node>` に統一する。

## Phase 3（A/B）への流用メモ

ヘッダ・ゲート・tree 指紋・スクリプトハッシュ固定は A/B でも共通に使える。A/B の確定条件 NB-SEP（`docs/perf/noise-band-decision.md`）は別雛形になり、本 issue では実装しない。
