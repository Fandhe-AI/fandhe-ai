# TASK-3.3e 結果評価・完了判定（イシュー #144）

TASK-3.3（`docs/spec/05-tasks.md:144`）「自作コア上での自己修復ループ完走の
再実証」の最終サブタスク。バグ修正種別（TASK-3.3b・#141）・機能追加種別
（TASK-3.3c・#142）の実証記録と統合記録（TASK-3.3d・#143）を評価材料とし、
REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`「自作コアに
対する自己修復ループの人間介在なし完走を新実装リポで再実証すること」）の
充足を判定する。本文書は AI が判定案を起草するところまでを担い、最終判定は
人間が行う（8 節）。

> **注記（本イシュー #144 での全面再評価）**: 本文書の初版（一次結果
> 「未充足」・PR #343）に対し、ユーザーは 2026-08-08 に **(c) 差し戻し
> （完全充足まで実装継続）** を選択した
> （[#139 コメント](https://github.com/Fandhe-AI/rust-ai-library/issues/139#issuecomment-5223546073)。
> AI 自動運転セッションの AskUserQuestion による確認）。この判断自体が
> 「人間による記録済みの判断」であり、その指示内容は「全基準の完全充足を
> 確認後に #139 のクローズ検証を再実施する」というものだった。これを受けて
> #131（基準 1 の CLI 実装）・#137（基準 2/5 のベンチ直接実測昇格）・
> #141（バグ修正種別の再実証）・#142（機能追加種別の再実証）・
> #143（統合記録更新）・#145（基準 4 の `verify-log` CLI 実装）の差し戻し
> 対応がすべて完了・クローズした（PR #355・#356・#361・#372・#374 等）。
> 本イシュー（#144）はその差し戻し対応を受けた §3〜§8 の全面再評価
> （all-or-nothing 規則の再適用を含む）を担う。旧版の「部分充足」
> 「未充足」評価・「条件付き充足」への逸脱提案は、以下の再評価に伴い
> すべて置き換える。

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

判定の根拠が改竄・破損されていないことを、判定作業に先立ち独立に確認した
（`.claude/rules/security.md` A08「自己修復ループが取り込む AI 生成変更は
ガードレール 3 分岐判定を必ず経由する」の前提となる記録そのものの完全性
検査。README §4 監査手順 1 に対応）。

- **検証方法**: `self-repair verify-log`（`docs/guardrail-self-repair-cli.md`
  §3.2 が仕様を定める外部コマンド CLI。`crates/self-repair/src/main.rs`）を
  独立プロセスとして 2 回起動し、`bug-fix/loop-log.jsonl`・
  `feature-addition/loop-log.jsonl` それぞれを検証した（旧版の lib
  `verify_chain` 直接呼び出しから、CLI バイナリ経由の第三者再検証手順へ
  更新。TASK-3.4・#145 の差し戻し対応〈PR #356〉で `verify-log` CLI
  バイナリが実装済みとなったことによる）。
- **検証日**: 2026-08-08
- **結果**:

  | 対象ファイル | 終了コード | `records` | `last_seq` | `last_hash` |
  |-------------|-----------|-----------|-----------|-------------|
  | `docs/self-repair-revalidation/bug-fix/loop-log.jsonl` | 0 | 5 | 4 | `d75680e54da8581f97a98953ca92bc2c6cb53bd30c69ae0a0af70a20bca3fb2e` |
  | `docs/self-repair-revalidation/feature-addition/loop-log.jsonl` | 0 | 5 | 4 | `db9926887a3c124aebdfe94ed86eee55ae139fb07152b3e1c79dc3df5b00760e` |

  （実行コマンド:
  `cargo run -p self-repair --bin self-repair -- verify-log --log docs/self-repair-revalidation/<種別>/loop-log.jsonl`。
  両ファイルとも `OK: ログチェーンの整合性を確認しました` を stdout に出力し
  exit 0。`records`/`last_seq`/`last_hash` は README §4 手順 3 の記録値と
  一致する）
- **限界（README §3 と同じ）**: `verify_chain`（`self-repair verify-log` が
  内部で呼ぶハッシュチェーン検証）は、フィールド改変・レコード削除・順序
  入れ替え・未知フィールド注入は検知できるが、ログ末尾の切り詰め（正規の
  最終レコード削除）は単体では検知できない（`docs/self-repair-log-format.md`
  6 節 3）。外部アンカー運用（同仕様書 7 節）は文書化のみで自動化実装は
  なく、本判定でもそれを補う追加検証は行っていない。この限界を踏まえても
  なお、以降の評価は「チェーン整合性が確認された記録」に基づくものである。

## 3. 完走判定基準 6 項目の評価

実証計画（`docs/self-repair-revalidation-plan.md` 5 節）が定める判定基準
6 項目について、README §2 の充足マトリクス（各個別 README から転記済み）を
一次資料（`loop-report.json`・`loop-log.jsonl`・ハーネスソース）と独立に
突き合わせて再評価する。過大表現をしない方針（README §2〜§3 の記述方針）を
踏襲する。

### 基準 1: `self-repair run` の 1 回起動・追加の人間入力なしで `AutoApply` へ到達

- **評価**: バグ修正・機能追加ともに**充足**。両種別とも `loop-report.json`
  の `invocation` フィールドに `self-repair run --kind <種別> ... --log ...
  --output ...` の 1 回起動が記録され、`outcome: "Adopted"` に到達している
  （`bug-fix/loop-report.json`・`feature-addition/loop-report.json`）。
  `self-repair run` CLI バイナリ本体は #131 の差し戻し対応（PR #361 で
  end-to-end 結線）で実装済みであり、旧版が指摘していたバグ修正側の
  「lib 直接呼び出しでの代替」は解消済み。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

### 基準 2: 検証 4 ゲート全通過・`guardrail` 3 分岐判定を迂回なく経由

- **評価**: バグ修正・機能追加ともに**充足**。`RepairCompositeGate`
  （TASK-3.2a・#137 の差し戻し対応で実装）が候補 diff に対し build/test/
  clippy＋直接ベンチを実測し、両 `loop-report.json` の
  `adopted_evidence.gate_report` が `"build=pass test=pass clippy=pass
  bench=measured-direct"` と一致する。`guardrail::decide` を唯一の判定経路
  として使用しており迂回経路はない（README §1）。旧版が指摘していたバグ
  修正側の「合成ワークロード機構完走確認・`bench` フィールド `NotRun`」は
  解消済み。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

### 基準 3: `--max-attempts` 上限内で完走

- **評価**: バグ修正・機能追加ともに**充足**。両 `loop-report.json` の
  `invocation` フィールドに `--max-attempts 5` が記録され（バグ修正:
  `bug-fix/loop-report.json` `invocation`。機能追加:
  `feature-addition/loop-report.json` `invocation`）、上限値 5 のうち
  `attempt_count=2`（attempt 2 で `Adopted`）で完走している。上限値の
  ハーネス側の設定根拠は `crates/self-repair/tests/revalidation_bug_fix.rs:660`・
  `crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs:642`
  （CLI 引数構築時のフォーマット文字列。`--max-attempts` の実引数指定箇所は
  それぞれ同ファイル 524 行目・494 行目）で確認できる。`loop-report.json`
  の `thresholds` フィールドには `bench_median_max_pct`・`bench_runs_min`・
  `lines_max` のみが記録され `max_attempts` は含まれないため、上限値の
  根拠として `invocation` フィールドおよび上記ハーネスソースの参照が必須
  である点は旧版から変わらない。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

### 基準 4: JSON Lines ログのハッシュチェーン検証（`self-repair verify-log`）を通過

- **評価**: バグ修正・機能追加ともに**充足**。`self-repair verify-log` を
  外部コマンドとして別プロセス起動し、両ファイルとも exit 0・
  `records=5, last_seq=4` を確認した（本文書 2 節で再検証済み。TASK-3.3d・
  #143 の統合記録と一致）。旧版が指摘していたバグ修正側の「CLI 未実装・
  lib 呼び出しのみ」は #145 の差し戻し対応（PR #356）で解消済み。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

### 基準 5: ベンチ劣化中央値が承認済み閾値内（5 回計測中央値）

- **評価**: バグ修正・機能追加ともに**充足**。`RepairCompositeGate`／
  `DirectBenchRunner`（#137 の差し戻し対応で実装）が候補 diff そのものを
  直接 5 回計測している（baseline commit と候補適用済み sandbox 双方の
  release ビルドを比較）。バグ修正: 中央値 -74.86%
  （`bench_median_pct=-74.85729651466627`。負値は改善方向〈誤って混入
  させた sigmoid すり替えバグ比での高速化であり、性能改善そのものが目的
  ではない〉）。機能追加: 中央値 0.49%
  （`bench_median_pct=0.4876067129737649`）。いずれも
  `guardrail.toml` `bench_median_max_pct=5.0` を下回る。閾値
  （`bench_median_max_pct=5.0`・`bench_runs_min=5`）自体は一切変更して
  いない。旧版が指摘していたバグ修正側の「合成ワークロードでは候補 diff
  固有の劣化を検出できない」という限界は、4 ゲート合成の `src/` 本体昇格
  （#137）により解消済み。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

### 基準 6: 判定レポート JSON の `signal_source` が `"measured"`

- **評価**: バグ修正・機能追加ともに**充足**。`bug-fix/loop-report.json`・
  `feature-addition/loop-report.json` いずれも `signal_source: "measured"`
  を持つ。旧版が指摘していたバグ修正側の「同フィールドなし」は #141 の
  再実証（PR #372）で解消済み。
- **完了判定への影響**: 妨げない。両種別とも完全充足。

## 4. REQ-3 受け入れ基準との突合

| REQ-3 受け入れ基準（`docs/spec/04-requirements.md:95-101`） | 実測根拠 |
|---|---|
| バグ修正・機能追加の少なくとも 2 種別で中断なく 1 ループ完走 | 両種別とも `outcome: "Adopted"`（`bug-fix/loop-report.json`・`feature-addition/loop-report.json`） |
| **（v2 追加）** 自作コアに対しても人間の介在なく完走を新実装リポで再実証 | 両題材とも `crates/autodiff`・`crates/self-repair` 自体（自作対象 7 項目、REQ-1）を対象に、`sandbox_bug_injection_commit`（バグ修正）・`leaky_relu` 新規実装（機能追加）で実施し、CLI バイナリ 1 回起動・人間入力なしで完走した（README §1・本文書 3 節 基準 1） |
| 危険な変更を 100% 却下・見逃しなし | 両種別とも attempt 1 が既知正解値テスト不合格で却下（バグ修正: `mlp_grad_*_matches_numeric` analytic/numeric 不一致。機能追加: `leaky_relu_matches_known_values` `got=0.05, want=0.5`）。attempt 2 まで見逃しなく到達 |
| 数値精度回帰を既知正解値誤差検証テストで検出 | バグ修正 attempt 1: `diff=0.87382805`・`diff=0.17694956`・`diff=0.0500679` の 3 件を検出・却下 |
| 試行回数・所要時間・判断根拠のログ出力 | 両種別とも `loop-report.json`（`attempt_count`・`total_duration_ms`・`outcome`）＋ `loop-log.jsonl`（ハッシュチェーン付き段階記録） |
| 改竄検知可能な形式でのログ記録 | `self-repair verify-log`（外部コマンド）によるハッシュチェーン検証を両種別とも exit 0 で通過（本文書 2 節） |
| ベンチゲートの計測系を新実装の計測基盤へ付け替え | 両種別とも `RepairCompositeGate`／`DirectBenchRunner`（`crates/bench-harness` 相当の計測経路）を候補 diff 直接実測へ使用（README §1・`gate_report`） |

## 5. 判定案（AI 起草・人間の判定対象）

`docs/self-repair-revalidation-plan.md` 5 節が定める完走判定基準の decision
rule は「**すべて満たした場合のみ**『人間介在なし完走』（REQ-3 受け入れ基準）
と認める」という all-or-nothing 規則である（TASK-3.3a・#140 で人間承認済み。
本文書はこの規則を一切変更していない）。3 節の再評価のとおり、判定基準
6 項目はいずれも**両種別とも**「充足」に到達している（README §2 の充足
マトリクスと一致）。

- **この all-or-nothing 規則を厳密に適用すると、本評価の結論は「充足」で
  ある。** 旧版（PR #343）が提案していた「条件付き充足」への逸脱は、
  全項目が規則どおり完全充足したことにより不要となったため本改訂で削除
  した。plan 5 節の規則そのものを緩めたり読み替えたりする判断は一切
  行っていない（規則からの逸脱を求めていない）。
- 残存する限界（検証スコープは `crates/autodiff` 単体クレート・
  `verify_chain` の末尾切り詰め検知限界・外部アンカー運用の未自動化。
  README §3・§5）はいずれもガードレール閾値・テスト許容誤差の緩和には
  起因しない。`guardrail.toml`（TASK-4.3c・#117 確定値）・
  `policy-exclusion.toml` は本イシュー・差し戻し対応（#131・#137・#141・
  #142・#143・#145）を通じて一切変更していない（6 節）。
- 自動運転（人間介在なし判断が求められる実行環境）のため安全側に倒し、
  本 AI 起草判定は「充足」を確定させるものではなく、8 節の人間判定に
  委ねる判定案の位置づけに留める。

## 6. 副次的な確認事項

- 両種別とも `guardrail.toml`（TASK-4.3c・#117 確定値）・`policy-exclusion.toml`
  を一切変更していない（README §1「一切変更せず」の記述をリポジトリの
  現行 `guardrail.toml` と突き合わせて確認済み。2 節の検証と同日に再確認）。
- 判定基準 6 項目（実証計画 §5）自体も本イシューでは変更していない
  （TASK-3.3a・#140 で人間承認済みの基準をそのまま適用）。

## 7. 後続事項

- spec 側 REQ-3 受け入れ基準チェックボックス（`docs/spec/04-requirements.md:95-101`）
  の更新は正本リポジトリ（Fandhe-AI/rust-ai-library-spec）側の対応であり、
  本リポでは行わない（`docs/spec/` 編集禁止。`.claude/rules/out-of-scope-tracking.md`）。
  本文書の判定結果（5 節）を踏まえたチェックボックス更新を、正本リポ側で
  対応することをユーザーへ提案する。
- 未充足・スコープ外事項の追跡先は README §5 の表をそのまま参照する
  （外部アンカー運用の自動化・実 workspace 全体を対象にした検証）。本
  イシューで新たに追跡が必要となった事項はない。

## 8. 人間判定の記録

### 8.1 初回判定（一次結果「未充足」）に対する人間判断の記録（履歴）

本文書の初版（PR #343。all-or-nothing 規則を厳密適用した一次結果
「未充足」）に対し、ユーザーは以下のとおり判断した。

| 項目 | 内容 |
|------|------|
| 判断日 | 2026-08-08 |
| 記録場所 | [#139 コメント](https://github.com/Fandhe-AI/rust-ai-library/issues/139#issuecomment-5223546073) |
| 判断手段 | AI 自動運転セッションの AskUserQuestion による確認 |
| 判断内容 | **(c) 差し戻し**（完全充足まで実装継続）。判定基準（all-or-nothing 6 項目）・guardrail 閾値・除外リストは変更せず、#131・#137・#141・#142・#143・#144・#145 を reopen して差し戻し対応を行い、全基準の完全充足を確認後に #139 のクローズ検証を再実施する、という指示 |
| 指示の充足状況 | 3 節のとおり、差し戻し対応（#131・#137・#141・#142・#143・#145 のクローズ）により判定基準 6 項目が両種別とも完全充足に到達した。よって (c) の指示条件（「完全充足まで実装継続」）は満たされている |

### 8.2 再判定（最終判定）の記録

**本 PR のレビュー・マージという事実そのものは、(a) 充足の人間判定を
構成しない**。マージ操作の実行者（`mergedBy.login`）はオーケストレーション・
bot・squash-merge を代行する自動化経路でありうるうえ、マージという行為
自体は「(a) 充足を判定した」ことを何ら保証しない（レビュー・承認なしの
直接マージ、レビュー承認はしたが (a)/(c) いずれの判定文言も残さない承認
など、マージ主体・判定内容を検証しないまま「マージ＝人間の完了判定」と
みなすと、bot によるマージでもイシュー #144 の受け入れ条件「人間による
完了判定が記録されている」を形式的に充足したことになってしまう。旧版の
運用ギャップの注意はここに引き継ぐ）。

したがって TASK-3.3e の完了判定は、以下の 3 条件をすべて満たすことを
もって記録済みとする。

1. **人間の判定者**が、GitHub 上のレビューまたはコメントとして
   **(a) 充足／(b) 部分充足／(c) 差し戻し のいずれかを明示した文言**を
   投稿していること（マージ操作それ自体は判定文言の代替にならない）
2. その投稿の **URL・投稿者（GitHub ログイン名）・投稿日時**を本節の表に
   記録していること
3. 投稿者が bot・オーケストレーション用アカウントでないこと（`[bot]`
   サフィックスや自動運転セッションの識別子でないことを目視確認する）

8.1 の (c) 差し戻し判断自体はこの要件を満たす記録済みの人間判断であり、
その指示が求めていた「完全充足」（3 節・5 節）はすでに満たされている
（3 節）。5 節の AI 起草判定案は **充足**であるが、これは AI による判定
「案」に留まり、下表の人間判定が別途記録されるまで TASK-3.3e は未完了
とする。

| 項目 | 内容 |
|------|------|
| 判定者 | aLiz-Nancy（人間アカウント。`[bot]` サフィックスなしを目視確認済み。判断手段は 8.1 節と同一の AI 自動運転セッションの AskUserQuestion による確認） |
| 判定日 | 2026-08-08（コメント投稿日時 2026-08-08T08:46:29Z） |
| 判定記録 URL | <https://github.com/Fandhe-AI/rust-ai-library/issues/144#issuecomment-5225382637> |
| 判定結果 | **(a) 充足**（判定基準 6 項目がバグ修正・機能追加の両種別とも完全充足に到達したことを確認。guardrail 閾値・ポリシー除外リスト・テスト許容誤差の変更なし） |
| 判定対象 | TASK-3.3（自作コア上での自己修復ループ完走の再実証）・REQ-3 v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`） |

### 8.3 判定者・判定日・判定結果の反映手順（機械的）

上表は、人間の判定者が本 PR（またはイシュー #144）に対して (a)/(b)/(c)
いずれかを明示したレビューまたはコメントを投稿した後、以下の手順で
本文書へ反映する。この反映は #139 クローズ検証に対してブロッキングで
あり、**人間判定コメント／レビューが存在しない状態でマージ確定情報
（`mergedBy`／`mergedAt`）のみを根拠に反映してはならない**。未反映のまま
#139 クローズ検証へ進んではならない。

1. 人間の判定者が本 PR にレビュー（`gh pr review`）またはコメント
   （`gh pr comment` / `gh issue comment 144`）として (a)/(b)/(c) の
   判定文言を投稿する
2. `gh pr view 375 --json reviews,comments` または
   `gh issue view 144 --json comments` で当該投稿を取得し、投稿者が
   bot・オーケストレーション用アカウントでないことを確認する
3. 取得した投稿者ログイン名を判定者欄、投稿日時を判定日欄、投稿の
   `html_url`（コメントの permalink）を判定記録 URL 欄、投稿本文の判定
   文言を判定結果欄へ、本文書を更新するフォローアップコミットで反映する
   （8.2 表の該当セルを実値に置換する）
4. 反映コミットは本イシュー（#144）のフォローアップとして記録する
