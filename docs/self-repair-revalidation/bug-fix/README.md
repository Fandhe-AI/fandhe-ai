# 自己修復ループ再実証: バグ修正種別（TASK-3.3b・イシュー #141）

本ディレクトリは REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）
「自作コアに対する自己修復ループの人間介在なし完走を新実装リポで再実証すること」の
うち、バグ修正種別の実証実行（TASK-3.3b）の記録である。題材・完走判定基準は
`docs/self-repair-revalidation-plan.md`（TASK-3.3a・#140・人間承認済み）4.1 節・5 節を
そのまま用いる（all-or-nothing 6 項目。#140 ユーザー判断で確定・変更していない）。

## 0. 差し戻し履歴（#139 (c)・2 度目）

本イシューは一度 PR #341 で「lib 直接呼び出しハーネス」により完走実証済みだったが、
#139 のユーザー判断 (c)「差し戻し」（2026-08-08）により reopen された。差し戻しで
割り当てられた残作業は「基準 2/5/6: バグ修正種別の再実証を CLI 経由・bench 実測・
`signal_source: "measured"` 付きで再実行」である。本ディレクトリの記録は
この差し戻し対応（本 PR）の実測結果に更新済みである。旧版（lib 直接呼び出し・
合成ワークロード完走確認のみ）の内容は git 履歴（PR #341）を参照。

先例: 機能追加種別（#142）は PR #361 で同種の CLI 経由再実証へ移行完了している
（`crates/self-repair/tests/feature_addition_loop_completion_task_3_3c.rs`・
`docs/self-repair-revalidation/feature-addition/`）。本ドキュメント・ハーネスは
その構成をバグ修正種別へ写像したものである。

## 1. 実施内容

- 題材: `crates/autodiff/src/var.rs` の `Var::relu` 実装本体を sigmoid 相当の演算
  グラフにすり替えるバグ注入（実証計画 4.1 節の推奨案。PoC-2 題材 (a) の v2 移植。
  #140 承認済みのまま変更していない）。
- 実証ハーネス: `crates/self-repair/tests/revalidation_bug_fix.rs`
  （`#[ignore]` 分離の統合テスト。理由はテストのコンパイル前コメント参照）。
  `self_repair` を lib として直接構築するのではなく、`env!("CARGO_BIN_EXE_
  self-repair")` で実バイナリを **1 回だけ起動**し、続けて `self-repair
  verify-log` を別プロセスとして起動する。
- 実行コマンド:

  ```sh
  SELF_REPAIR_TASK_3_3B_WRITE_DOCS=1 \
    cargo test -p self-repair --test revalidation_bug_fix -- --ignored --nocapture
  ```

  環境変数を省略した場合は本ディレクトリ（tracked）を書き換えず、`target/
  self-repair-revalidation/bug-fix`（`.gitignore` 済み）へのみ出力する
  （非破壊に試し実行したい場合はこちらを使う。`self-repair run` を docs
  反映のためだけに再実行しない設計。7 節参照）。
- 実行環境: CPU バックエンドのみ（CUDA・Metal 実機依存なし）。
- 改竄検知ログ: 本ディレクトリの `loop-log.jsonl`（JSON Lines・SHA-256
  ハッシュチェーン。CLI バイナリ内部の `self_repair::LogWriter` が `--log`
  へ書き出したものをそのまま複製）。

## 2. 実施手順（ハーネス内部）

1. **準備リポジトリの構築**: `crates/autodiff` を一意な一時ディレクトリへ
   再帰コピーし、`Cargo.toml` の workspace 継承（`version`/`edition`/`license`/
   `publish`・`tensor-core`/`bench-harness` への相対 path 依存・`serde`/
   `serde_json`）をすべて実体値へ展開して独立 crate 化する（末尾に空の
   `[workspace]` テーブルを追記し親 workspace から切り離す）。候補 diff 直接
   実測（4 節）向けの決定的ベンチワークロード bin（`src/bin/bench_workload.rs`。
   `autodiff::Var::relu` の forward+backward を反復）を追加し、`Var::relu` の
   forward 呼び出しを `eval::relu` → `eval::sigmoid` へ書き換える（バグ注入。
   `Op::Relu` の登録自体は変更しないため backward は勾配式のまま計算され、
   forward・backward の不整合が既知正解値テストで検出される）。`cargo
   generate-lockfile` で `Cargo.lock` を確定させたうえで単独の git リポジトリ
   として `git init`・1 コミット（この HEAD が `baseline_commit` になる）。
   以降の git 操作・cargo 実行はすべてこの準備リポジトリ・CLI 内部の隔離
   sandbox に閉じ、メイン working copy・共有 git 状態には一切触れない
   （並列イシュー実行時のグローバル状態保護）。
2. **候補列の書き出し**: attempt 1（誤り: `eval::sigmoid` を別の誤り
   `eval::tanh` に置換するのみ）・attempt 2（正解: `relu` 実装〈`eval::relu`〉
   の復元）を JSON（`--candidates`）へ書き出す。
3. **`self-repair run` を 1 回起動**（完走判定基準 1）:

   ```sh
   self-repair run --kind bug-fix --repo <準備リポジトリ> --max-attempts 5 \
     --log loop-log.jsonl --output loop-report.json --candidates <candidates.json> \
     --bench-bin bench_workload --workload-source src/bin/bench_workload.rs \
     --config guardrail.toml --policy-exclusion policy-exclusion.toml \
     --allow-candidate-exec
   ```

   CLI 内部では `--repo`（準備リポジトリ）を `RunSandbox::create` が更に
   `git clone --local` した隔離 sandbox で、検出（`BugFixDetector`。
   `cargo test --release`）→ 修正生成（`BugFixFixGenerator`）→ 検証
   （`RepairCompositeGate`。build/test/clippy -D warnings の 3 ゲート＋
   候補 diff 直接ベンチ実測）→ 取り込み判断（`GuardrailAdoptionJudge` →
   `guardrail::decide`。sandbox 直下の `guardrail.toml`・`policy-exclusion.toml`
   〈確定値・本実証では一切変更しない〉を使用）の 1 ループを実行する。判定の
   迂回経路はない。
4. **`self-repair verify-log` を外部コマンド経由で検証**（完走判定基準 4）:
   `self-repair verify-log --log loop-log.jsonl` を別プロセスとして起動し、
   exit 0・`OK:` メッセージを確認する。
5. **証跡の記録**: `--output` JSON（`loop-report.json`）へ実行コマンドライン
   （`invocation`）・所要時間・issue/task 番号・充足内訳の注記（`notes`）を
   追記し、`SELF_REPAIR_TASK_3_3B_WRITE_DOCS=1` 明示時のみ本ディレクトリへ
   複製する。

## 3. `self-repair run` の検証スコープ

準備リポジトリ（`crates/autodiff` 単体クレート）を対象に build/test/clippy/
bench の 4 ゲートを実行する（実 workspace 全体は対象外。実行時間の理由。
`verify_gates_integration.rs` と同じスコーピング判断）。実機（CUDA・Metal）
依存はなく CPU バックエンドのみで完走する。

## 4. 実証計画 5 節「完走判定基準」との対応（CLI 経由再実証後）

| # | 判定基準 | 充足状況 |
|---|---------|---------|
| 1 | `self-repair run --kind bug-fix` の 1 回起動・追加の人間入力なしで終了コード 0（`Verdict::AutoApply`）に到達 | **充足**。CLI バイナリ（PR #361）を 1 回起動し、`outcome=Adopted`・`exit=0` に到達した（`loop-report.json` 実測） |
| 2 | 検証 4 ゲート（build／test --release／clippy -D warnings／bench）全通過。`guardrail` の 3 分岐判定を経由し、迂回経路がないこと | **充足**。`RepairCompositeGate`（TASK-3.2a・#137）が候補 diff（`var.rs`）に対し 4 ゲートを実測し、`gate_report="build=pass test=pass clippy=pass bench=measured-direct"` を記録。`guardrail::decide` を唯一の判定経路として使用 |
| 3 | `--max-attempts` 上限内で完走すること | **充足**。`--max-attempts 5` のうち attempt 2（正解）で `Adopted` に到達（`attempt_count=2`） |
| 4 | JSON Lines ログのハッシュチェーン検証（`self-repair verify-log`）を通過すること | **充足**。`self-repair verify-log` CLI（#145）を外部コマンドとして起動し exit 0・`OK: ... records=5, last_seq=4` を確認した |
| 5 | ベンチ劣化中央値が承認済み閾値内（5 回計測の中央値採用・単発計測禁止。閾値は変更しない） | **充足**。候補 diff（`var.rs` の relu 実装復元）そのものを `DirectBenchRunner` が直接計測。`bench_measurements_pct` は 5 件（`MIN_BENCH_ITERATIONS`）、中央値は `bench_median_pct` に記録され `guardrail.toml` の `bench_median_max_pct` 閾値内 |
| 6 | 判定レポート JSON の `signal_source` フィールドが `"measured"` であること | **充足**。`loop-report.json` の `signal_source` は `"measured"`（`--signals` 契約検証パスを経由しない実シグナル計測） |

## 5. 実行結果サマリ

`loop-report.json` より:

- 最終結論: `outcome="Adopted"`（終了コード 0）
- 試行回数: `attempt_count=2`（attempt 1: 検証不合格で却下 → attempt 2: 検証通過・
  取り込み採用）
- attempt 1 の却下理由: `cargo test --release`（`BugFixDetector`・
  `RepairCompositeGate` の test ゲート）が既知正解値テスト
  （`crates/autodiff/tests/backward.rs` の `mlp_grad_*_matches_numeric`）で
  analytic 勾配と numeric 勾配の不一致により失敗（`eval::tanh` へのすり替えでも
  forward・backward の不整合は解消しない）
- attempt 2 の採用理由: `adopted_evidence` が示す 4 ゲート全通過
  （`gate_report="build=pass test=pass clippy=pass bench=measured-direct"`）＋
  診断由来シグナル（`lines_changed`・`api_broken=false`・`gaming_suspect=false`・
  `exclusion_rule_ids=[]`）＋候補 diff 直接実測ベンチが `guardrail::decide` の
  自動適用条件をすべて満たした

## 6. スコープ外事項（`.claude/rules/out-of-scope-tracking.md` 準拠）

以下は本イシュー（#141）のスコープ外として記録し、後続イシューで追跡する。

- **統合記録の棚卸し**（`docs/self-repair-revalidation/README.md` 充足マトリクス・
  `completion-judgment.md`）: #143／#144 のスコープとして本 PR では触れない。
- **完走可否の人間評価**: #144（本 README・`loop-report.json` を判定基準に
  照らして評価する）。
- **実 workspace 全体を対象にした検証**: TASK-3.3 系の別スコープ（3 節参照）。

## 7. 再現方法

```sh
SELF_REPAIR_TASK_3_3B_WRITE_DOCS=1 \
  cargo test -p self-repair --test revalidation_bug_fix -- --ignored --nocapture
```

候補列（attempt 1・attempt 2 の内容）・検証ゲート・取り込み判断はいずれも決定的
であり、`guardrail.toml`／`policy-exclusion.toml`（本実証では一切変更しない）を
変えない限り、再実行しても同一の最終結論（`Adopted`）に到達する。ただしベンチ
計測値（`bench_measurements_pct`・`bench_median_pct`）自体は実行環境の負荷に
応じて多少変動しうる（5 回計測中央値の採用により単発ノイズは吸収される。
実装計画 §7 リスク対策）。`--repo`（準備リポジトリ）・`--candidates` の絶対パスは
使い捨て一時ディレクトリのため実行のたび変わる（`invocation` フィールド参照）。

## 8. 改竄検知ログ（`loop-log.jsonl`）の監査手順

`loop-log.jsonl` は `docs/self-repair-log-format.md`（TASK-3.4・#145 の正式
仕様書）が定める JSON Lines・SHA-256 ハッシュチェーン形式である。監査手順・
フォーマット詳細は同仕様書 6 節を参照。本実証固有の要点のみ以下に示す。

- **段階列**: `loop_start`（`kind: "bug_fix"`）→ `detection` →
  `attempt`（attempt 1: `verification_failed`）→ `attempt`（attempt 2:
  `adopted`）→ `loop_outcome`（`outcome.kind: "adopted"`）の 5 レコード
  （`loop-report.json` の `attempt_count: 2` と対応）。
- **検証**: `self-repair verify-log --log loop-log.jsonl` を独立した外部
  コマンドとして起動して検証する（完走判定基準 4）。監査担当者は
  `cargo run -p self-repair -- verify-log --log docs/self-repair-revalidation/
  bug-fix/loop-log.jsonl` で第三者として再検証できる。
- **改竄検知の実効性**: `verify_chain` がフィールド改変・レコード削除・
  順序入れ替え・未知フィールド注入をいずれも検知することは
  `crates/self-repair/src/logging.rs` の単体テスト
  （`verify_chain_detects_stage_field_tampering`・
  `verify_chain_detects_recorded_at_field_tampering`・
  `verify_chain_detects_field_tampering`・
  `verify_chain_detects_record_deletion`・
  `verify_chain_detects_record_reordering`・
  `verify_chain_rejects_unknown_top_level_field_injection`）で個別に
  実証済みであり、本実証で改めて同種の負検査を作り直すことはしない。
- **末尾切り詰めの限界**: `verify_chain` 単体では検知できない
  （`docs/self-repair-log-format.md` 6 節 3）。外部アンカー運用（同仕様書
  7 節）は運用指針の文書化のみで自動化実装は行っていない
  （out-of-scope-tracking.md 準拠）。
