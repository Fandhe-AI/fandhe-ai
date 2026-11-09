# cpu-gemm-reuse-release-sizegate-macmini-4159

イシュー #4159（eager release のサイズゲートで Mac mini が後退しないことを確かめる）の記録置き場。
設計・到達性・限界の本文は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §19。GB10 側は #4158（`../cpu-gemm-reuse-release-sizegate-4158/`）。

状態: **部品のみ・未実測**。`RULE.txt` は部品 PR のマージ後、計測より前に単独コミットで固定する
（`run_ab_eager_release_cpu.sh` は `RULE.txt` が無いと fail-closed で止まるため、事前登録前は正式系列を取れない）。
Mac mini の実測と判定は未実施（`macmini/`・`judge-4159.md` はまだ無い）。

## 収録物

| パス | 内容 |
|---|---|
| `session-scripts/session-4159-macmini.sh` | 実機ロックの取得と解放・run スクリプトの呼び出し・`steps.log` を 1 本で行うセッション |
| `RULE.txt` | 事前登録規則（計測前に別コミットで固定。後から追加） |
| `macmini/`・`judge-4159.md`・`rev-stamp-verification-4159.md` | 実測後に追加 |

after 腕の 1 行差分は #4158 の `../cpu-gemm-reuse-release-sizegate-4158/on-arm.patch`（sha256 `efb25b5fb70947dcf268518232debfc83f8e7e951b3d8a5f88bfa0e650f33978`）をそのまま使う。
「GB10 と同じ腕」を 1 つのソースで保証するためコピーしない。

## RULE の骨子（案。固定は計測前）

- 腕: before = main（`EAGER_MATMUL_RELEASE_DEFAULT_ENABLED=false`）／after = 同じコミットでその 1 行だけ `true`。サイズゲート閾値（32 MiB）は両腕共通。precheck が `size_gate_max_bytes=33554432` を両腕で assert する
- セル（17）: 既定経路 12（gemm N=256／512／1024／2048 × fresh／reuse、train／infer × fresh／reuse）＋診断経路 5（`--diag-legacy-tape-reuse`: gemm reuse N=256／512／1024／2048・train reuse）
- N=4096 は入れない。出力 64 MiB は after でもゲートで解放されず構造上 before と同じ。本 A/B は「ゲートの下で解放が残るサイズの非後退」を確かめるもので、Mac mini でゲートそのものを試すものではない
- 区分: 判定セル = 診断経路 5（機構が実際に働く）。A/A 参照セル = 既定経路 12（`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §18.2 のとおり eager release が届かず構造上 A/A。後退の判定には数えない）
- 判定（#3226 の形を緩めない。ratio = after／before の 5 round 中央値）:
  - REGRESS（exit 1）: 判定セルのどれかが 5/5 round で ratio > 1.00、または 17 セルのどれかで checksum 不一致
  - NONREGRESS（exit 0）: 判定セル 5 つすべての中央値 ratio <= 1.00、A/A 参照セルに 5/5 後退なし、checksum 全一致、負荷ゲートの証跡あり
  - undetermined（exit 2）: 上記以外。A/A 参照セルの 5/5 後退（計測系の系統差か到達性の読解誤り）・データ不正・欠損・record-only を含む
  - ADOPT は出さない。既定の反転は両機体の判定を見たうえで #3227 が判断する
- 事前の仮説: #3226 の Mac mini（当時の既定経路。いまの診断経路に相当）で gemm reuse N=256 は 1.067（round 別 0.996〜1.128）、N=1024／2048 は 0.958／0.949（§8.4）。N=256 は undetermined か REGRESS に倒れうる
- 専有: 5 round とも load1 < 3.0（30 秒間隔・最大 30 分待つ）。実機ロックは `session-4159-macmini.sh` が mkdir 方式で取る。GPU の compute プロセス検査は適用しない（Mac mini に無い）
- ノイズ帯 NB-RANGE5（`docs/perf/noise-band-decision.md`・案・所有者承認待ち）は判定に使わず、round 比の最小〜最大を参考列に記録するだけ

## 実行手順

ホスト名は公開面で `<metal-node>`（実名は Git 管理外の `docs/real-hardware-verification-env.local.md`）。
手順 1〜2 は #3226 の `../cpu-gemm-reuse-release-3226/README.md` と同じ（bundle 配布・`.rev-stamp`）。

1. 部品を入れたコミットの Git クローンを `$HOME/work/r4159/runner` に作り、`.rev-stamp` を置く（アーカイブ展開木は不可。run スクリプトが `git rev-parse --show-toplevel` と HEAD 一致を要求し、ハードウェアロック取得直後に失敗する）
2. `before` と `after` を runner と同一コミットのクローンとして置く（各 `.rev-stamp` 付き）。`after` だけ `git apply <sizegate-4158>/on-arm.patch`
3. `zsh -lc` 経由の nohup で `bash session-scripts/session-4159-macmini.sh <label>` を起動する（計測中に別のビルドを並走させない）
4. 回収: `out-<label>/`（`steps.log`・`run.log` に加え、終了時に JSONL・負荷ゲートログ・compare・judge をラベル一致分だけ `artifacts/` へ集約済み）を `macmini/` へ。マスク規則はホスト名 → `masked`、`$HOME` → `<home>`、ユーザー名 → `<user>`。マスク後に `spark`・`mac-mini`・`/home/`・`/Users/` が 0 件であることを grep で確認する
5. `judge-4159.md`（`python3 -I judge_eager_release_ab.py --rule-set 4159 ...` の出力）と `rev-stamp-verification-4159.md` を置き、`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §19.4 を更新する

- 開発機の疎通確認は `AB_PRECHECK=report-only`（正式系列にならない）
