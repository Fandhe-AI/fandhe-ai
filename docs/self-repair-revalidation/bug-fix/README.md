# 自己修復ループ再実証: バグ修正種別（TASK-3.3b・イシュー #141）

本ディレクトリは REQ-3 の v2 追加受け入れ基準（`docs/spec/04-requirements.md:96`）
「自作コアに対する自己修復ループの人間介在なし完走を新実装リポで再実証すること」の
うち、バグ修正種別の実証実行（TASK-3.3b）の記録である。題材・完走判定基準は
`docs/self-repair-revalidation-plan.md`（TASK-3.3a・#140・人間承認済み）4.1 節・5 節を
そのまま用いる。

## 1. 実施内容

- 題材: `crates/autodiff/src/var.rs` の `Var::relu` 実装本体を sigmoid 相当の演算
  グラフにすり替えるバグ注入（実証計画 4.1 節の推奨案。PoC-2 題材 (a) の v2 移植）。
- 実証ハーネス: `crates/self-repair/tests/revalidation_bug_fix.rs`
  （`#[ignore]` 分離の統合テスト。理由はテストのコンパイル前コメント参照）。
- 実行コマンド:

  ```sh
  SELF_REPAIR_REVALIDATION_OUT=docs/self-repair-revalidation/bug-fix \
    cargo test -p self-repair --test revalidation_bug_fix -- --ignored --nocapture
  ```

- 実施日時: `loop-report.json` の `started_at_unix_ms`（UNIX ミリ秒）。
- 実行環境: CPU バックエンドのみ（CUDA・Metal 実機依存なし）。

## 2. 実施手順（ハーネス内部）

1. **サンドボックス構築**: 実行中リポジトリを `git clone --local --no-hardlinks`
   でリポジトリ外の一時ディレクトリへ複製する。以降の git 操作・cargo 実行は
   すべてこのサンドボックス内に閉じ、メイン working copy・共有 git 状態には
   一切触れない（並列イシュー実行時のグローバル状態保護）。
2. **バグ注入**: サンドボックスの `crates/autodiff/src/var.rs` の `relu` メソッド
   本体（forward 呼び出し行）を `eval::relu` → `eval::sigmoid` へ書き換え、
   サンドボックス内でコミットする（`Op::Relu` の登録自体は変更しない。backward は
   `Op::Relu` の勾配式のまま計算されるため、forward・backward の不整合が既知正解値
   テストで検出される）。このコミットが diff の起点（baseline）になる。
3. **検出**: `self_repair::BugFixDetector`（`cargo test --release`。検証対象は
   `crates/autodiff` 1 クレート）が既知正解値テストの失敗を `Finding` として検出。
4. **修正生成**: `self_repair::BugFixFixGenerator` に決定的な候補列を注入する
   （PoC-2「修正試行 1: 誤り・却下 → 修正試行 2: 正解・採用」の写像）。
   - attempt 1（誤り）: `eval::sigmoid` を別の誤り `eval::tanh` に置換するのみで、
     依然として `Op::Relu` の勾配式と forward 値が一致しない。
   - attempt 2（正解）: `relu` 実装（`eval::relu`）を復元する。
5. **検証**: `crates/self-repair/tests/revalidation_bug_fix.rs` 内の
   `RevalidationVerificationGate`（`self_repair::stages::VerificationGate` 実装）が
   `self_repair::CargoVerificationGate`（build → test --release → clippy -D
   warnings の 3 ゲート）と `self_repair::verify_bench::SelfRepairBenchGate`
   （ベンチゲート機構）を合成して実行する。3 ゲートすべて通過した試行のみ、
   追加でベンチゲート機構を実行する（3 節「4 ゲート合成についての設計上の制約」
   参照）。
6. **取り込み判断**: `self_repair::GuardrailAdoptionJudge` → `guardrail::decide`
   （sandbox 直下の `guardrail.toml`〈TASK-4.3c 確定値〉をそのまま使用）。判定の
   迂回経路はない。
7. **完走ログ書き出し**: `self_repair::LoopReport`（試行回数・所要時間・判断根拠）
   を JSON 化し、本ディレクトリの `loop-report.json` へ書き出す。

## 3. 4 ゲート合成についての設計上の制約（重要）

`self_repair::verify_gates::CargoVerificationGate`（build/test/clippy の 3 ゲート）
と `self_repair::verify_bench::SelfRepairBenchGate`（ベンチゲート）の合成
（4 ゲート化）は `crates/self-repair/src/` 本体へまだ結線されていない
（#136 系のスコープ。`crates/self-repair/src/lib.rs` モジュールコメント参照）。

本ハーネスは `tests/` 配下の統合テストクレート（`self_repair` を外部クレートとして
利用する）であり、`self_repair::outcome::VerifiedEvidence::new` は `pub(crate)`
のため、ここから新しい `VerifiedEvidence`（bench シグナルを実測値へ差し替えたもの）
を構築できない（A08 の型レベル境界。`crates/self-repair/src/outcome.rs` の
ドキュメント参照）。

このため、`loop-report.json` の `bench_gate_mechanism` フィールドが記録するベンチ
計測値は、**ベンチゲート機構自体の完走確認**（合成ワークロード。relu forward
相当の要素毎演算を baseline・candidate で同一に実行し、`bench_runs_min` 回以上・
中央値判定で機構が正しく完走し閾値内に収まることを確認するもの）であり、
**候補 diff（bug fix のワークツリー差分）そのものの性能劣化率実測ではない**。
候補ファイルはサンドボックス（別プロセス空間）にあり、本テストプロセスへ動的
リンクして直接呼び出す経路がないため、候補コードそのものをベンチ計測することは
本ハーネスの構成上できない。

実際に `guardrail::decide` へ渡される `VerifiedEvidence` の `bench` フィールドは、
`CargoVerificationGate::verify` が返す値（3 ゲート未計測のため常に
`guardrail::BenchSignal::NotRun`）をそのまま使う。`guardrail::DecisionInput::new`
は「全ゲート通過 + `NotRun`」を矛盾とはしない（`crates/self-repair/src/
verify_gates.rs` のドキュメント参照）ため、取り込み判断自体はこの制約の影響を
受けない（判定の迂回経路にはならない）。

候補 diff に対する劣化率実測（真の 4 ゲート合成の `src/` 本体への昇格）は
#136 系の残作業として別途追跡する（7 節参照）。

## 4. 実証計画 5 節「完走判定基準」との対応

| # | 判定基準 | 本実証での充足状況 |
|---|---------|---------------------|
| 1 | `self-repair run --kind bug-fix` の 1 回起動・追加の人間入力なしで終了コード 0（`Verdict::AutoApply`）に到達 | **部分充足**。`self-repair run` CLI バイナリが本イシュー時点で未実装のため、lib 直接呼び出し（`SelfRepairLoop::run`）経由で「1 回起動・追加の人間入力なし」を満たし、最終結論 `LoopOutcome::Adopted`（`guardrail::Verdict::AutoApply` 相当）に到達したことを確認した。CLI 形態での再実施は 7 節参照 |
| 2 | 検証 4 ゲート（build／test --release／clippy -D warnings／bench）全通過。`guardrail` の 3 分岐判定を lib 直接呼び出しで経由し、迂回経路がないこと | **充足**（3 ゲートは実測・ベンチゲートは機構完走確認。3 節参照）。`guardrail::decide` を唯一の判定経路として使用し、迂回経路は存在しない |
| 3 | `--max-attempts` 上限内で完走すること | **充足**。`max_attempts = 2` で attempt 2（正解）にて `Adopted` に到達（`loop-report.json` の `attempt_count`） |
| 4 | JSON Lines ログのハッシュチェーン検証（`self-repair verify-log`）を通過すること | **未充足（スコープ外）**。JSON Lines ハッシュチェーンログ・`verify-log` は TASK-3.4（#145）のスコープ。本実証では `LoopReport` 由来の JSON（`loop-report.json`）で試行回数・所要時間・判断根拠を記録した |
| 5 | ベンチ劣化中央値が承認済み閾値内（5 回計測の中央値採用・単発計測禁止。閾値は変更しない） | **部分充足**。ベンチゲート機構自体は `bench_runs_min`（sandbox の `guardrail.toml` 確定値。5 回）以上・中央値判定で完走し閾値内（`bench_gate_mechanism.median_pct` 参照）。ただし 3 節のとおり合成ワークロードであり候補 diff の実測ではない |
| 6 | 判定レポート JSON の `signal_source` フィールドが `"measured"` であること | **未充足（スコープ外）**。`signal_source` フィールドは `self-repair run` CLI（`docs/guardrail-self-repair-cli.md` §2.1）の出力仕様であり、CLI 未実装の本イシュー時点では該当フィールド自体が存在しない |

## 5. 実行結果サマリ

`loop-report.json` より:

- 最終結論: `LoopOutcome::Adopted`（`guardrail::Verdict::AutoApply` 相当）
- 試行回数: 2（attempt 1: 検証不合格で却下 → attempt 2: 検証通過・取り込み採用）
- 合計所要時間: `total_duration_ms`（サンドボックス構築・バグ注入は計測対象外。
  検出開始から最終結論確定までを計測）
- attempt 1 の却下理由: `cargo test --release` の既知正解値テスト
  （`crates/autodiff/tests/backward.rs` の `mlp_grad_*_matches_numeric`）が
  analytic 勾配と numeric 勾配の不一致で失敗（`eval::tanh` へのすり替えでも
  forward・backward の不整合は解消しない）
- attempt 2 の採用理由: 全ゲート通過（build/test/clippy）＋ベンチゲート機構
  完走（合成ワークロード・劣化率中央値が閾値内）＋ diff 由来シグナル
  （`lines_changed`・`api_broken`・`gaming_suspect`・`exclusion_rule_ids`）が
  すべて `guardrail::decide` の自動適用条件を満たした

## 6. スコープ外事項（`.claude/rules/out-of-scope-tracking.md` 準拠）

以下は本イシュー（#141）のスコープ外として記録し、後続イシューで追跡する。

- **`self-repair run`/`verify-log` CLI バイナリの実装**
  （`docs/guardrail-self-repair-cli.md` §3）: 未実装のため lib 直接呼び出しで
  代替した。CLI 実装後の再実施は CLI 実装タスクの後続イシューのスコープ。
- **JSON Lines ハッシュチェーンログ・`verify-log` 検証**（判定基準 4）:
  TASK-3.4（イシュー #145）のスコープ。
- **候補 diff に対するベンチ劣化率実測（真の 4 ゲート合成の `src/` 本体への
  昇格）**（3 節）: `verify_bench::SelfRepairBenchGate` の
  `stages::VerificationGate` への正式結線は #136 系の残作業として親イシューで
  追跡済み。
- **機能追加種別の完走実証**: #142（本ハーネスのサンドボックス構築・診断
  ロジック〈`git`／diff／ポリシー除外評価のヘルパー関数〉は #142 で再利用可能な
  形にできる）。
- **記録ディレクトリ構成の最終確定・記録整備**: #143。
- **完走可否の人間評価**: #144（本 README・`loop-report.json` を判定基準に
  照らして評価する）。

## 7. 再現方法

```sh
SELF_REPAIR_REVALIDATION_OUT=docs/self-repair-revalidation/bug-fix \
  cargo test -p self-repair --test revalidation_bug_fix -- --ignored --nocapture
```

候補列（attempt 1・attempt 2 の内容）・検証ゲート・取り込み判断はいずれも決定的
であり、`guardrail.toml`／`policy-exclusion.toml`（sandbox にクローンされた
確定値。本実証では一切変更しない）を変えない限り、再実行しても同一の最終結論
（`Adopted`）に到達する（実証計画 3 節 (e)「決定的シードで再現可能」の要求に
対する対応。本題材は乱数を使わない純粋関数のため、決定的シードユーティリティ
（`guardrail::determinism`）自体は不要である）。
