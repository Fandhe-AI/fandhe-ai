# metal-gemm-candidate-ab-2111

candle／MLX steel 解析差分由来の Metal GEMM 候補（イシュー #2110）の kernel_gpu 5 run A/B 用スキャフォールド。
候補の定義・根拠・除外表は [`docs/perf/metal-gemm-steel-candidates.md`](../../metal-gemm-steel-candidates.md)。

## 分担

| イシュー | 担当 |
|---|---|
| #2110 | 機構（`UNROLL_LOAD_ENABLED`・candle 相当タイル arm）・bit 一致／parity の自己テスト・計測ハーネス・本スキャフォールド（`RULE.txt` は実測前に固定済み） |
| #2111 | M4 Max 実機での gate・5 run 計測・`aggregate.py` の判定・結線判断（本番既定の変更はユーザー承認事項） |

`RULE.txt` は実測前に固定した規則で、実測結果を見た後に緩和・変更しない。

## 実行手順（Apple Silicon 実機・#2111）

1. `env_info.txt` を記入する（ホスト名・ユーザー名・絶対パスは書かない）。専有ゲート（RULE.txt 7.）を満たして計測する通常運用では **run 1 の前に** `load_policy: exclusive_gate` を明記する（`aggregate.py` は `exclusive_gate` 以外・未記入・行欠落を常に REFERENCE_ONLY にするため、この行が無いと採用判定が出ない）。専有できない場合のみ `load_policy: record_only` と理由を宣言する（下記「判定後の手順」）。
2. `./orchestrate.sh gate` — bit 一致 3 本と CPU 参照 parity 1 本。FAIL なら A/B は実施しない（RULE.txt 1.）。
3. `./orchestrate.sh 1` … `./orchestrate.sh 5` — プロセス独立の 5 run。各 run の前に load1 < 8.0 を最大 30 分待つ。
4. `python3 aggregate.py` — arm 別に最終判定（ADOPT_CANDIDATE／REJECT／UNDETERMINED／NOT_ADOPTABLE／INCOMPLETE／REFERENCE_ONLY）を出力する。採否の正は `^arm=<名> verdict=` の行のみ。各 arm には上書き前のデータ由来の判定と全理由を `underlying_verdict`／`underlying_reason`、最終判定が下位判定を上書きした全理由（入力不完全・参考扱い）を `override_reason` として別行で併記する（診断情報であり採否ではない。参考扱い系列でも bit 不一致・run 間ハッシュ不一致・データ欠落を区別して読める。最終判定の優先度は従来どおり INCOMPLETE > REFERENCE_ONLY > データ由来の判定）。前提ゲート不成立時（`verdict=REJECT`）は arm 別判定を出さず、`reference_reason`／`problem` の行のみを併記する。
5. 出力を `aggregate.md` として保存し、`docs/perf/metal-gemm-steel-candidates.md` の実測記入欄を埋める。

Linux 等では `./orchestrate.sh gate --dry-run`／`./orchestrate.sh 1 --dry-run` で分岐だけ検証できる。
`python3 aggregate.py --self-test` は判定ロジックの固定 fixture 検証（実測不要）。

`aggregate.py` は fail-closed: `load_gate.log` の run 1〜5 各記録・各 `kernel_gpu_run{i}.log` の対象テスト成功（`test result: ok`・`0 failed`）・`env_info.txt` の `run{i} completed` 記録のいずれかが欠ける／失敗の場合、全 arm を INCOMPLETE にし採用判定を出さない（RULE.txt 2.・7.）。

## 保存するファイル

`gate_run.log`・`kernel_gpu_run{1..5}.log`・`load_gate.log`・`run{1..5}_monitor.log`・`run{1..5}_procs.txt`・
`uptime_before_run{1..5}.txt`・`pmset_therm_{before,after}_run{1..5}.txt`・`env_info.txt`・`aggregate.md`。

コミット前に、ログ中のホスト名・ユーザー名・絶対パスを `<home>` 等へマスクする
（`.claude/rules/security.md`。並走プロセスは `run*_procs.txt` に件数のみ記録される）。

## 判定後の手順（#2111）

- `record_only` の宣言（専有できない場合）は **run 1 の前に** `env_info.txt` へ理由付きで書く（`env_info.txt` の運用。RULE.txt 7. は専有ゲートと timeout 時の REFERENCE_ONLY のみを定めており、事前宣言は事前登録規則に無い運用上の取り決めである）。事後宣言は認めない。**`load_policy: record_only`（および `exclusive_gate` 以外・未記入・行欠落）の系列は参考扱い（REFERENCE_ONLY）であり、採用判定・結線判断の根拠にしない**。`aggregate.py` は `env_info.txt` の `load_policy` を読み、`exclusive_gate` 以外では最終判定（`verdict=`）に ADOPT_CANDIDATE 等を出さず `REFERENCE_ONLY` のみを出力する（fail-closed。RULE.txt 7.）。上書き前の判定・理由は `underlying_*` 行に残るが診断情報であり、採用根拠にしない。採用判断には専有ゲートを満たした再計測が必要。
- ADOPT_CANDIDATE が出ても**結線前にユーザー承認を取る**（RULE.txt 10.）。結線手順は `docs/perf/metal-gemm-steel-candidates.md` §8。
- 全 arm が ADOPT_CANDIDATE でない場合は結線せず、判定・中央値の記入と本ディレクトリの収録のみで完了とする。
- 保存するログは、コミット前にホスト名・ユーザー名・絶対パスをマスクする（上記マスク規則）。
