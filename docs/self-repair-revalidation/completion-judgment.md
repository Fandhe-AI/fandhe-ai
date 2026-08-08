# TASK-3.3e 結果評価・完了判定（イシュー #144）

TASK-3.3（`docs/spec/05-tasks.md:144`）「自作コア上での自己修復ループ完走の
再実証」の最終サブタスク。バグ修正種別（TASK-3.3b・#141）・機能追加種別
（TASK-3.3c・#142）の実証記録と統合記録（TASK-3.3d・#143）を評価材料とし、
REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`「自作コアに
対する自己修復ループの人間介在なし完走を新実装リポで再実証すること」）の
充足を判定する。本文書は AI が判定案を起草するところまでを担い、最終判定は
人間が行う（8 節）。

> **注記（#142 再調査による更新）**: 機能追加種別（TASK-3.3c・#142）は本
> 文書作成後に `self-repair run` CLI 経由へ移行し、`crates/self-repair/src/
> diff_signals.rs::api_signature_touched` の誤検出（新規追加のみの
> `pub fn` 追加を一律「API 破壊」と判定していた再実装バグ。ガードレール
> 本体の判定ロジック・閾値の問題ではない）を修正した。現在は完走判定基準
> 1・4・5・6 をすべて充足し `LoopOutcome::Adopted`・exit 0 に到達している
> （実測は [`feature-addition/README.md`](./feature-addition/README.md) を
> 正とする）。バグ修正種別（TASK-3.3b・#141）も同様に PR #372 で
> `self-repair run` CLI 経由・候補 diff 直接実測への再実証を完了し、
> 判定基準 6 項目すべて「充足」となった（実測は
> [`bug-fix/README.md`](./bug-fix/README.md) §4 を正とする。統合記録は
> TASK-3.3d・#143 で [`README.md`](./README.md) §1〜§6 へ反映済み）。
> 以下 §3〜§8（特に「部分充足」「未充足」評価）はこれらの更新前の
> lib 直接呼び出し・合成ワークロード時点のものであり、all-or-nothing
> 規則の再適用を含む本文書全体の再評価は本イシュー（#144）側で改めて行う。

## 1. 評価対象と評価材料

| 種別 | 出典 |
|------|------|
| 統合サマリ・充足マトリクス | [`README.md`](./README.md) §1〜§6 |
| バグ修正種別の詳細記録 | [`bug-fix/README.md`](./bug-fix/README.md) |
| バグ修正種別の生データ | [`bug-fix/loop-report.json`](./bug-fix/loop-report.json)・[`bug-fix/loop-log.jsonl`](./bug-fix/loop-log.jsonl) |
| 機能追加種別の詳細記録 | [`feature-addition/README.md`](./feature-addition/README.md) |
| 機能追加種別の生データ | [`feature-addition/loop-report.json`](./feature-addition/loop-report.json)・[`feature-addition/loop-log.jsonl`](./feature-addition/loop-log.jsonl) |
| 実証計画・完走判定基準 6 項目 | `docs/self-repair-revalidation-plan.md` 5 節 |
| REQ-3 の受け入れ基準 | `docs/spec/04-requirements.md:94-101` |

## 2. ログ整合性の事前確認

判定の根拠が改竄・破損されていないことを、判定作業に先立ち確認した
（`.claude/rules/security.md` A08「自己修復ループが取り込む AI 生成変更は
ガードレール 3 分岐判定を必ず経由する」の前提となる記録そのものの完全性
検査。README §4 監査手順 1 に対応）。

- **検証方法**: `self_repair::verify_chain(path)`（`crates/self-repair/src/logging.rs:472`）
  を、コミット済みの `bug-fix/loop-log.jsonl`・`feature-addition/loop-log.jsonl`
  の 2 ファイルに対し、`crates/self-repair` の一時統合テスト経由で実行した
  （恒久コードは追加していない。検証専用の一時テストファイルを実行後に削除）。
- **検証日**: 2026-08-07
- **結果**:

  | 対象ファイル | `verify_chain` の結果 |
  |-------------|----------------------|
  | `docs/self-repair-revalidation/bug-fix/loop-log.jsonl` | `Ok(())` |
  | `docs/self-repair-revalidation/feature-addition/loop-log.jsonl` | `Ok(())` |

- **限界（README §4-4 と同じ）**: `verify_chain` は SHA-256 ハッシュチェーンの
  整合性検査であり、フィールド改変・レコード削除・順序入れ替え・未知
  フィールド注入は検知できるが、ログ末尾の切り詰め（正規の最終レコード削除）
  は単体では検知できない（`docs/self-repair-log-format.md` 6 節 3）。外部
  アンカー運用（同仕様書 7 節）は文書化のみで自動化実装はなく、本判定でも
  それを補う追加検証は行っていない。この限界を踏まえてもなお、以降の評価は
  「チェーン整合性が確認された記録」に基づくものである。

## 3. 完走判定基準 6 項目の評価

実証計画（`docs/self-repair-revalidation-plan.md` 5 節）が定める判定基準
6 項目について、README §2 の充足マトリクスをもとに「完了判定を妨げるか
否か」を個別に評価する。過大表現をしない方針（README §2〜§3 の記述方針）を
踏襲し、「部分充足」区分を「充足」へ格上げしない。

### 基準 1: `self-repair run` の 1 回起動・追加の人間入力なしで `AutoApply` へ到達

- **評価**: バグ修正は部分充足（lib 直接呼び出し `SelfRepairLoop::run` での
  代替）。機能追加は**充足**（#142 再調査分の更新: `self-repair run` CLI を
  1 回起動し `Adopted`・exit 0 へ到達。冒頭注記・
  [`feature-addition/README.md`](./feature-addition/README.md) 参照）。
- **完了判定への影響**: 妨げない。バグ修正種別の `self-repair run` CLI
  バイナリ本体経由での再実証は `docs/guardrail-self-repair-cli.md` §3 が
  仕様のみ確定させた後続タスクの実装対象であり（README §5）、ガードレール
  閾値・テスト許容誤差の緩和には一切起因しない。人間の介在なく検出→修正
  生成→検証→取り込み判断の 1 ループが完走したという事実（`loop-report.json`
  の `outcome`）自体は両種別とも実測されている。

### 基準 2: 検証 4 ゲート全通過・`guardrail` 3 分岐判定を迂回なく経由

- **評価**: バグ修正は部分充足（bench ゲートは機構完走確認であり `bench`
  フィールドは `NotRun` のまま `guardrail::decide` へ渡る）。機能追加は充足
  （#142 再調査分の更新: `RepairCompositeGate` が build/test/clippy＋候補
  diff 直接ベンチを全ゲート通過後に実測し `bench` フィールドが `Measured`
  として渡る。`feature-addition/loop-report.json` の `gate_report:
  "build=pass test=pass clippy=pass bench=measured-direct"`）。
- **完了判定への影響**: 妨げない。機能追加種別は本基準を完全充足しており、
  「2 種別のうち少なくとも 1 種別で 4 ゲート全実測」を満たす。バグ修正種別の
  部分充足は、4 ゲート合成の `src/` 本体への昇格が #136 系のスコープ
  （README §5）であることに起因し、ガードレール閾値・許容誤差の緩和では
  ない。`guardrail::decide` を唯一の取り込み判断経路として使用し迂回経路が
  ないこと自体は両種別とも実測されている（README §1「迂回経路なし」）。

### 基準 3: `--max-attempts` 上限内で完走

- **評価**: 充足（バグ修正: `max_attempts=2` で attempt 2 採用〈設定値の出典:
  `crates/self-repair/tests/revalidation_bug_fix.rs:702`〉。機能追加:
  `max_attempts=5` で attempt 2 採用〈設定値の出典:
  `crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs:580`〉）。
  上限値そのものは両ファイルの `NonZeroU32::new(..)` 呼び出しに明記されており、
  `loop-report.json` の `attempt_count`（実際の試行回数）とは別の値であることに
  注意する。`loop-report.json` の `thresholds` フィールドには
  `bench_median_max_pct`・`bench_runs_min`・`lines_max` のみが記録され
  `max_attempts` は含まれないため、上限値の根拠は `loop-report.json` 単体では
  確認できず、上記ハーネスソースの参照が必須である。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

### 基準 4: JSON Lines ログのハッシュチェーン検証（`self-repair verify-log`）を通過

- **評価**: バグ修正は部分充足（本イシューで新規充足。`loop-log.jsonl` を
  出力し、lib `verify_chain` 呼び出しによる検証で `Ok` を得た〈本文書 2 節で
  再検証済み〉。`self-repair verify-log` 外部コマンド CLI は未実装のため
  未使用）。機能追加は**充足**（#142 再調査分の更新: `self-repair verify-log`
  を外部コマンドとして別プロセス起動し exit 0 を確認）。
- **完了判定への影響**: 妨げない。バグ修正種別の CLI バイナリ未使用は基準 1
  と同一のスコープ外事項（`docs/guardrail-self-repair-cli.md` §3 後続タスク）
  に起因し、検証ロジック自体（`verify_chain` が実装するハッシュチェーン
  アルゴリズム）はハーネスが呼び出す関数と CLI が呼び出す関数が同一である。
  TASK-3.3d（#143）で「未充足」から「部分充足」へ格上げされた実測
  （README §3）を本文書 2 節で独立に再確認したことで、判定材料としての
  信頼性を追加で担保した。

### 基準 5: ベンチ劣化中央値が承認済み閾値内（5 回計測中央値）

- **評価**: バグ修正は部分充足（機構完走確認は合成ワークロード。中央値
  0.234% < 5.0% は経路実測）。機能追加は**充足**（#142 再調査分の更新:
  `RepairCompositeGate`／`DirectBenchRunner` による候補 diff 直接実測
  〈baseline commit と候補適用済み sandbox 双方の release ビルドを比較〉。
  中央値 0.49% < 5.0%）。
- **完了判定への影響**: 妨げない。承認済み閾値（`guardrail.toml`
  `bench_median_max_pct=5.0`・`bench_runs_min=5`）自体は両種別とも一切
  変更せず、5 回計測・中央値採用という判定ロジックも変更なく適用されている
  （README §1）。バグ修正種別の「合成ワークロードでは候補 diff 固有の劣化を
  検出できない」という限界は、ベンチ計測系を実際の候補コードに対して実行
  する経路（4 ゲート合成の `src/` 本体昇格、#136 系）が未完了であることに
  起因する設計上の制約であり、ガードレール閾値・許容誤差の緩和ではない
  （README §5・§6）。

### 基準 6: 判定レポート JSON の `signal_source` が `"measured"`

- **評価**: バグ修正は未充足（スコープ外。`bug-fix/loop-report.json` に
  `signal_source` フィールドなし）。機能追加は充足
  （`feature-addition/loop-report.json` の `signal_source == "measured"`）。
- **完了判定への影響**: 妨げない。「バグ修正・機能追加の少なくとも 2 種別で
  完走」という REQ-3 受け入れ基準（4 節参照）に対し、本基準は 2 種別中
  1 種別（機能追加）で完全充足しており、実証計画 §1.2・§7 節が定める
  `signal_source == "measured"` のレポートのみを実証結果として採用する
  という制約自体は満たされている。バグ修正種別のハーネスに同フィールドが
  存在しないのは、TASK-3.3b（#141）実装当時のハーネス設計判断であり、
  ガードレール閾値・許容誤差の緩和には起因しない。

## 4. REQ-3 受け入れ基準との突合

| REQ-3 受け入れ基準（`docs/spec/04-requirements.md:95-101`） | 実測根拠 |
|---|---|
| バグ修正・機能追加の少なくとも 2 種別で中断なく 1 ループ完走 | 両種別とも `LoopOutcome::Adopted`（`bug-fix/loop-report.json` `outcome.kind`・`feature-addition/loop-report.json` `outcome`） |
| **（v2 追加）** 自作コアに対しても人間の介在なく完走を新実装リポで再実証 | 両題材とも `crates/autodiff`・`crates/self-repair` 自体（自作対象 7 項目、REQ-1）を対象に、`sandbox_bug_injection_commit`（バグ修正）・`leaky_relu` 新規実装（機能追加）で実施し、人間入力なしで完走した（README §1） |
| 危険な変更を 100% 却下・見逃しなし | 両種別とも attempt 1 が既知正解値テスト不合格で却下（バグ修正: `mlp_grad_*_matches_numeric` analytic/numeric 不一致。機能追加: `leaky_relu_matches_known_values` `got=0.05, want=0.5`）。attempt 2 まで見逃しなく到達 |
| 数値精度回帰を既知正解値誤差検証テストで検出 | バグ修正 attempt 1: `diff=0.87382805`・`diff=0.17694956`・`diff=0.0500679` の 3 件を検出・却下 |
| 試行回数・所要時間・判断根拠のログ出力 | 両種別とも `loop-report.json`（`attempt_count`・`total_duration_ms`・`reason`/`outcome`）＋ `loop-log.jsonl`（ハッシュチェーン付き段階記録） |
| 改竄検知可能な形式でのログ記録 | `verify_chain` によるハッシュチェーン検証を両種別とも `Ok` で通過（本文書 2 節） |
| ベンチゲートの計測系を新実装の計測基盤へ付け替え | 両種別とも `crates/bench-harness` 相当の計測経路を使用（README §1・`gate_report`） |

## 5. 判定案（AI 起草・人間の判定対象）

`docs/self-repair-revalidation-plan.md` 5 節が定める完走判定基準の decision
rule は「**すべて満たした場合のみ**『人間介在なし完走』（REQ-3 受け入れ基準）
と認める」という all-or-nothing 規則である（TASK-3.3a・#140 で人間承認済み）。
3 節の評価のとおり、基準 1・4・5 はバグ修正種別が部分充足（機能追加種別は
#142 再調査分の更新で充足へ移行済み。冒頭注記参照）、基準 6 はバグ修正種別で
未充足であり、6 項目全部を**両種別とも**満たしてはいない。**この
all-or-nothing 規則を厳密に適用すると、本評価の結論は「未充足」である。**

- 未充足となった項目（基準 1・4・5 のバグ修正側部分充足、基準 6 のバグ修正側
  未充足）はいずれも文書化済みスコープ外事項（バグ修正側の CLI 未実装・
  4 ゲート合成の `src/` 本体未昇格・合成ワークロード限界・ハーネス設計判断）
  に起因し、ガードレール閾値・テスト許容誤差の緩和には起因しない（3 節）。
- CLI 経由の完全再現（基準 1・4 の完全充足）・4 ゲート合成の `src/` 本体
  昇格によるベンチ劣化率の直接実測（基準 2・5 の完全充足）は、後続タスク
  （`docs/guardrail-self-repair-cli.md` §3・#136 系）の残課題として追跡する。
- 上記を踏まえ、AI 起草判定としては「plan 5 節の all-or-nothing 規則に
  厳密に従えば結論は『未充足』である」ことを一次判定として明記したうえで、
  未充足の内訳がすべてスコープ外事項に起因し閾値・許容誤差の緩和を伴わない
  という限定条件のもとでのみ、plan の規則からの **逸脱** として
  「条件付き充足」への読み替えを人間に検討してもらう、という二段構えの
  判定案とする。「条件付き充足」は plan 5 節に定義されたカテゴリではなく、
  本文書が新設を提案する逸脱先であることを明記する。plan 記載の
  all-or-nothing 規則そのものを緩めるか否かは `.claude/rules/security.md`
  が定める「判定基準の変更は人間承認必須」の精神に照らし、8 節で人間が
  明示的に選択する。
- 自動運転（人間介在なし判断が求められる実行環境）のため安全側に倒し、
  本 AI 起草判定は「充足」はもとより「条件付き充足」も確定させるものでは
  なく、8 節の人間判定に委ねる判定案の位置づけに留める。

## 6. 副次的な確認事項

- 両種別とも `guardrail.toml`（TASK-4.3c・#117 確定値）・`policy-exclusion.toml`
  を一切変更していない（README §1「一切変更せず」の記述をリポジトリの
  現行 `guardrail.toml` と突き合わせて確認済み）。
- 判定基準 6 項目（実証計画 §5）自体も本イシューでは変更していない
  （TASK-3.3a・#140 で人間承認済みの基準をそのまま適用）。

## 7. 後続事項

- spec 側 REQ-3 受け入れ基準チェックボックス（`docs/spec/04-requirements.md:95-101`）
  の更新は正本リポジトリ（Fandhe-AI/rust-ai-library-spec）側の対応であり、
  本リポでは行わない（`docs/spec/` 編集禁止。`.claude/rules/out-of-scope-tracking.md`）。
  本文書の判定結果を踏まえたチェックボックス更新を、正本リポ側で対応する
  ことをユーザーへ提案する。
- 未充足・スコープ外事項の追跡先は README §5 の表をそのまま参照する
  （`self-repair run`/`verify-log` CLI 実装・4 ゲート合成の `src/` 本体
  昇格・外部アンカー運用の自動化）。本イシューで新たに追跡が必要となった
  事項はない。

## 8. 人間判定の記録

TASK-3.3a（`docs/self-repair-revalidation-plan.md` 8 節）の先例に準拠し、
本文書を含む PR の人間によるレビュー・マージをもって、TASK-3.3e の
完了判定（イシュー #144 受け入れ条件「人間による完了判定が記録されている」）
の記録とする。

| 項目 | 内容 |
|------|------|
| 判定者 | （PR マージ時に記入） |
| 判定日 | （PR マージ時に記入） |
| 判定結果 | plan 5 節の all-or-nothing 規則を厳密適用した場合の一次結果は
**未充足**（5 節）。人間は次のいずれかを選択して記入する: (a) **未充足**
（規則どおり。CLI 実装・4 ゲート合成の `src/` 本体昇格を待って再判定）、
(b) **条件付き充足**（5 節で提案する、plan 5 節の規則からの逸脱を明示的に
承認し、未充足の内訳がスコープ外事項のみに起因することを確認した上での
判定。plan 5 節の規則そのものを緩和する判断であることを認識して選択する）、
(c) **差し戻し**（評価そのもののやり直しが必要と判断した場合）。 |
| 判定対象 | TASK-3.3（自作コア上での自己修復ループ完走の再実証）・REQ-3 v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`） |

本文書を含む PR のレビュー・マージが、TASK-3.3e の受け入れ条件「人間による
完了判定が記録されている」の人間承認に当たる。ただし本パイプラインでは
PR のマージはオーケストレーションスクリプトが実行するため、上表の各欄が
実際に人手で記入される運用になっていない可能性がある（TASK-3.3a §8 の
前例踏襲という位置づけ自体は妥当だが、この運用ギャップは認識しておく必要が
ある）。空欄のまま残存する場合は、判定記録の未完了として別途フォロー
アップが必要である。
