# 自己修復ループ試行ログのフォーマット仕様（TASK-3.4）

TASK-3.4（`docs/spec/05-tasks.md`）の成果物である「ログフォーマット仕様」を
定める正式仕様書。REQ-3（`docs/spec/04-requirements.md`）受け入れ基準の
「試行ログから取り込み判断の根拠を追跡できること」、および
[`.claude/rules/security.md`](../.claude/rules/security.md) の
「ループ試行ログは改竄検知可能な形式で記録し、取り込み判断の根拠を
追跡可能にする」に対応する。

想定読者は、自己修復ループの試行ログを事後監査する担当者、および
CLI バイナリ（`self-repair run`/`verify-log`。
[`docs/guardrail-self-repair-cli.md`](./guardrail-self-repair-cli.md) 3 節が
CLI 境界を確定済み）でログ書き込み・検証を配線する実装者である。

実装本体は
[`crates/self-repair/src/logging.rs`](../crates/self-repair/src/logging.rs)
（TASK-3.4・イシュー #145）であり、本仕様書はその実装からの転記である
（推測・希望的な将来仕様を書かない。以下、行番号参照は本イシュー実装時点の
同ファイルを指す）。

v1（`Fandhe-AI/rust-ai-library-v1` `tools/self-repair/src/logging.rs`。
TASK-3.2-S1/S2・イシュー #47/#48）からの移植。JSON Lines フォーマット・
ハッシュチェーン計算式・`DOMAIN_PREFIX`・genesis 値は v1 と完全に同一であり
（`DOMAIN_PREFIX` の `v1` はログフォーマットの版数を指し、リポジトリ世代
（v1/v2）とは無関係のため変更していない）、v1 が出力・検証したログファイルを
そのまま v2 の `verify_chain` で検証できる。

## 0. v1 からの実装差分（依存構成のみ。フォーマット自体は不変）

v1 は `sha2`（`Sha256`）・`thiserror` に依存するが、いずれも v2 の許容依存
8 区分（[`.claude/rules/deps-policy.md`](../.claude/rules/deps-policy.md)）に
含まれず、依存追加はユーザー承認事項のため本移植では次のように代替した。

- `sha2::Sha256` → [`crates/self-repair/src/sha256.rs`](../crates/self-repair/src/sha256.rs)
  （FIPS 180-4 準拠の自作実装。NIST 既知テストベクタで検証済み。
  `crates/guardrail/src/toml_lite.rs` が `toml` クレートを自作パーサで
  代替したのと同じ方針）
- `thiserror::Error` derive → `crate::error::SelfRepairError` と同じ
  手書き `Display`/`Error` 実装

`sha2` クレート採用への切り替え可否は依存追加のためユーザー判断事項であり、
現時点では採用していない（イシュー #145 の PR 本文に記録）。

## 1. 目的・根拠

- **REQ-3**: 自己修復ループが取り込んだ変更の判断根拠を、後から検証可能な
  形で残す。
- **TASK-3.4**: 「ログフォーマットの移植」の成果物としてのログフォーマット
  仕様書。
- **security.md A08（ソフトウェア・データ整合性）**: AI が生成した変更の
  取り込みはガードレール 3 分岐判定を経由する必要があり、判定の迂回を
  事後に検知できることが監査の前提となる。ログはその一次記録である。

## 2. ファイル形式

- **JSON Lines**（1 行 1 レコード、UTF-8、末尾改行付き）。追記専用
  （[`LogWriter::append_stages`](../crates/self-repair/src/logging.rs)
  が `OpenOptions::new().append(true).create(true)` でオープンする。
  既存内容の上書き・削除は行わない）。
- `LogWriter::open` は既存ファイルがあれば末尾まで読み、次に書くべき
  `seq` と直前レコードの `hash` を復元してからチェーンを継続する
  （`read_tail_state`）。整合性チェックは `open` 時には行わない
  （書き込み継続に必要な状態復元のみ。整合性を確認したい場合は利用側が
  明示的に 6 節の `verify_chain` を呼ぶ）。
- 各行は単独でパース可能な JSON オブジェクトであり、`reason` 等の値に
  改行・`"` ・バックスラッシュを含んでいても `serde_json` のエスケープに
  よって 1 行 1 レコードの構造が壊れない（8 節参照）。

## 3. レコード共通スキーマ

各行（レコード）は次の 6 フィールドを持つ（`LogRecord` 構造体。
`crates/self-repair/src/logging.rs` 内、非公開型）。

| フィールド | 型 | 説明 |
|-----------|----|------|
| `seq` | `u64` | 0 始まりの連番。ハッシュチェーンの本質的な改竄検知に対する**人間が目視で確認するための冗長な補助情報**（連番自体もハッシュに混入されるため欠番・重複は 6 節の検証で検知される）。 |
| `recorded_at_unix_ms` | `u128`（JSON 上は数値） | レコード書き込み時刻（UNIX ミリ秒）。`SystemTime::now()` 取得失敗時は `0` にフォールバックする（本番経路で `unwrap`/`expect` を使わない方針。`now_unix_ms` 参照）。 |
| `stage` | `string` | この段階の識別子。4 節の 5 種のいずれか。 |
| `payload` | `object` | 段階別の内容。4 節参照。 |
| `prev_hash` | `string`（16 進小文字 64 文字） | 直前レコードの `hash`。先頭レコードは 5 節の genesis 値。 |
| `hash` | `string`（16 進小文字 64 文字） | 5 節の計算式によるこのレコードのハッシュ。 |

トップレベルは `#[serde(deny_unknown_fields)]` を付けている。ハッシュは
既知フィールドのみを対象に再計算するため、これがないと未知フィールドの
注入が 6 節の検証をすり抜けてしまう（security.md A03/A08・fail-closed 方針）。

## 4. 段階（stage）別 payload スキーマ

段階列は 1 回のループ実行につき
`loop_start → detection → attempt ×n → loop_outcome`
（正常終了。`LoopReport` 由来）、または
`attempt ×n → loop_failure`
（段階の実行自体がエラーで終了。`LoopFailure` 由来。`loop_start`/
`detection` を出力しない理由は後述）の順で並ぶ
（`stages_for_report`/`stages_for_failure`）。

### 4.1 `loop_start`

対象種別を記録する（`repair_kind_payload`）。

```json
{ "kind": "bug_fix" }
```

`kind` は `RepairKind::as_machine_id`（`crates/self-repair/src/kind.rs`）
が返す機械可読文字列で、`bug_fix` / `perf_regression` / `feature_addition`
のいずれか。

### 4.2 `detection`

```json
{ "action_needed": true }
```

`action_needed` は `report.outcome` が `LoopOutcome::NoActionNeeded` で
**ない**ことの真偽値（`!matches!(report.outcome, LoopOutcome::NoActionNeeded)`）。
検出段階で「修正不要」と判断されループを開始しなかった場合は `false` になる。

### 4.3 `attempt`

1 試行分の記録（`attempt_record_payload`）。

```json
{
  "attempt": 1,
  "duration_ms": 10,
  "outcome": { "kind": "verification_failed", "reason": "..." }
}
```

- `attempt`: 試行番号（1 始まり）。
- `duration_ms`: この試行に要した時間（ミリ秒）。
- `outcome`: `AttemptOutcome`（`crates/self-repair/src/report.rs`）を
  `attempt_outcome_payload` が写像した値。全 5 variant を次の表に示す
  （`_ =>` を使わず全 variant を明示する fail-closed 方針。判断根拠を
  取りこぼさないための設計）。

| `AttemptOutcome` variant | `outcome.kind` | 追加フィールド |
|---|---|---|
| `VerificationFailed { reason }` | `"verification_failed"` | `reason`: 検証ゲート不合格の理由 |
| `AdoptionRejectedRetryable { reason }` | `"adoption_rejected_retryable"` | `reason`: 再試行可能な却下の理由 |
| `Adopted` | `"adopted"` | なし |
| `Escalated { reason }` | `"escalated"` | `reason`: 人間レビューへ回した理由（`guardrail::decide` 由来） |
| `RejectedFinal { reason }` | `"rejected_final"` | `reason`: 再試行不能な却下の理由 |

### 4.4 `loop_outcome`

`LoopReport`（正常終了）の最終結論（`stages_for_report` 末尾）。

```json
{
  "outcome": { "kind": "adopted" },
  "total_duration_ms": 123,
  "attempt_count": 1
}
```

- `outcome`: `LoopOutcome`（`crates/self-repair/src/outcome.rs`）を
  `loop_outcome_payload` が写像した値。全 5 variant を次の表に示す。
- `total_duration_ms`: 検出開始〜最終結論確定までの合計所要時間（ミリ秒）。
- `attempt_count`: 実施した試行回数（`LoopReport::attempt_count()`）。

| `LoopOutcome` variant | `outcome.kind` | 追加フィールド |
|---|---|---|
| `NoActionNeeded` | `"no_action_needed"` | なし |
| `Adopted` | `"adopted"` | なし |
| `Escalated { reason }` | `"escalated"` | `reason`: エスカレーション理由 |
| `Rejected { stage, reason }` | `"rejected"` | `stage`: 却下が確定した段階名、`reason`: 却下理由 |
| `Exhausted` | `"exhausted"` | なし |

### 4.5 `loop_failure`

`LoopFailure`（段階の実行自体がエラーで終了）の記録（`stages_for_failure`
末尾）。

```json
{ "error": "...", "attempt_count": 1 }
```

- `error`: `SelfRepairError` の `Display` 文字列（`failure.error.to_string()`）。
- `attempt_count`: 失敗するまでに実施された試行回数
  （`failure.attempts.len()`）。

`LoopFailure` 由来のログには `loop_start`/`detection` レコードを**出力しない**。
`LoopFailure` は `kind`（対象種別）を保持しない構造体であり、存在しない情報を
偽装して出力しないという fail-closed 方針（security.md A05 と同じ考え方）に
よる。したがってエラー終了したループのログは `attempt ×n → loop_failure`
のみで構成される（`n` は 0 の場合もある）。

## 5. ハッシュチェーン仕様

各レコードの `hash` は次の式で計算する（`compute_hash`。書き込み時
（`LogWriter::append_stages`）と検証時（`verify_chain`）で同一関数を呼び、
計算式の乖離を防ぐ）。

```text
hash = SHA-256(
    DOMAIN_PREFIX
    || seq (u64, リトルエンディアン 8 バイト)
    || 0x00
    || recorded_at_unix_ms (u128, リトルエンディアン 16 バイト)
    || 0x00
    || stage (UTF-8 バイト列)
    || 0x00
    || prev_hash (UTF-8 バイト列)
    || canonical_payload_json
)
```

16 進小文字でエンコードして `hash` フィールドへ格納する。SHA-256 自体は
[`crates/self-repair/src/sha256.rs`](../crates/self-repair/src/sha256.rs)
の自作実装（0 節参照）を使う。

- **`DOMAIN_PREFIX`**: 固定文字列 `"rust-ai-library/self-repair/log/v1"`。
  本ログ以外の用途で計算された SHA-256 値を「有効な直前ハッシュ」として
  誤って受理する衝突・流用を防ぐドメイン分離（security.md A08）。
- **ハッシュ対象フィールドの範囲**: `payload` だけでなく `seq` /
  `recorded_at_unix_ms` / `stage` / `prev_hash` の全フィールドを対象に
  含める。`payload` を変えずに `stage` や `recorded_at_unix_ms` だけを
  書き換える改竄（例:「取り込み確定」の記録を「単なる検出メモ」に見せかける、
  判断時刻を偽装する）を見逃さないための設計判断であり、v1 実装時からの
  意図的な選択である。
- **フィールド境界のセパレータ**: `seq`/`recorded_at_unix_ms`/`stage` の
  各境界に `\0` バイトを挟んで結合する。セパレータなしで単純連結すると
  `"a"+"bc"` と `"ab"+"c"` のような異なる値の組が同一バイト列に潰れて
  衝突する余地があるため。
- **`canonical_payload_json`**: `serde_json::to_vec(payload)` の出力。
  本クレートは `serde_json` の `preserve_order` feature を有効化して
  いないため、`serde_json::Value` の内部表現に従いオブジェクトの
  キーはアルファベット順で安定する。同一内容の `payload` からは常に
  同一バイト列が得られることを前提にする。
- **genesis 値**: 先頭レコード（`seq = 0`）の `prev_hash` に使う値。
  固定文字列 `"self-repair-log-v1"` の SHA-256（16 進小文字）で、
  実際の値は次の通り（v1・v2 双方の `genesis_hash()` 実装で一致することを
  確認済み。`crates/self-repair/src/sha256.rs`・`logging.rs` のテスト参照）。

  ```text
  be26b2311e026d01ceabc4dc7b360f8583ee203c4e4a858aef631c698c4b1a28
  ```

  `None` や空文字列を使わずハッシュ値で固定するのは、改竄側が同じ値を
  用意しやすく検知力が弱まることを避けるため。

## 6. 監査・追跡手順

取り込み判断の根拠をログのみから復元する手順は次の通り。

1. **チェーン整合性の検証**: `verify_chain(path)` を呼ぶ。全レコードを
   先頭から再走査し、次の 3 点をいずれも fail-closed に検証する
   （1 箇所でも不一致・パース不能な行があれば直ちに `Err`。部分的に
   正しい範囲だけを認める緩い検証はしない）。
   - `seq` が 0 から連続していること（欠番・重複の検知）
   - `prev_hash` が直前レコードの `hash` と一致すること（削除・並べ替えの検知）
   - レコードから 5 節の式で再計算した値が記録済み `hash` と一致すること
     （`seq`/`recorded_at_unix_ms`/`stage`/`payload` いずれかの改変の検知）

   `verify_chain` が `Err(LogError::ChainViolation { seq, reason })` を
   返した場合、そのログは信頼できないため以降の手順（2）に進まず、
   改竄・破損の疑いとして扱う（security.md「ログを信頼できない入力として
   扱い、`verify_chain` 通過後に利用する」）。

2. **時系列に沿った判断根拠の復元**: `verify_chain` を通過したログについて、
   `stage` ごとに次の情報を辿る。
   - `loop_start` の `kind` で対象種別を確認する。
   - `detection` の `action_needed` で、そもそも修正が必要と判断されたか
     確認する（`false` ならこの後 `attempt` は存在しない）。
   - 各 `attempt` の `outcome.kind`/`outcome.reason` を順に読み、
     「何回目の試行が・どの段階で・なぜ却下/採用されなかったか」を
     時系列に復元する（4.3 節の対応表）。
   - 最後の `loop_outcome`（または `loop_failure`）で最終結論・所要時間・
     試行回数を確認する（4.4/4.5 節の対応表）。

   以上により、`verify_chain` 通過済みログと本仕様書の対応表だけを頼りに、
   実装（`self_repair` クレートの Rust 型）を参照せずとも判断根拠を
   追跡できる。

3. **検知できる改竄・検知できない改竄**:
   - **検知できる**: レコードのフィールド改変（`stage`/`recorded_at_unix_ms`
     を含む）・レコード削除・順序入れ替え（いずれも `verify_chain` が
     `ChainViolation` として検知する）。
   - **検知できない**: **末尾切り詰め**（ファイル末尾の正当なレコードを
     まるごと削除する改竄）。削除後に残る最終レコードまでのチェーンは
     自己無矛盾になるため、`verify_chain` 単体では検知できない
     （`crates/self-repair/src/logging.rs` の `verify_chain` ドキュメンテー
     ションコメント参照）。この弱点は 7 節の外部アンカー運用で補う。

## 7. 外部アンカー運用（末尾切り詰め対策）

6 節 3 の「検知できない改竄（末尾切り詰め）」を補うため、ログファイル外の
改竄困難な媒体へ最終レコードの `hash` を記録する運用指針を以下に定める。
**本節は運用指針の文書化のみを行い、自動化実装は行わない**
（自動運転による安全側の判断。out-of-scope-tracking.md 準拠。実装が
必要と判断された場合は、別途 Issue 化をユーザーに提案する）。

- **記録タイミング**: `LogWriter::append_report`/`append_failure` 呼び出し
  直後（ループ 1 回分の書き込み完了時点）。
- **記録対象**: その時点の最終レコードの `hash` フィールド値と、対応する
  ログファイルパス・`seq`。
- **記録先の候補**（いずれも本ログファイル自体より改竄が困難、または
  改竄すれば別途検知可能な媒体を選ぶ）:
  - 取り込み判断に対応する PR のコメント（`guardrail` CLI・自己修復ループの
    CI 実行から `gh pr comment` 等で投稿する）
  - CI 実行ログ・アーティファクト（self-hosted runner のジョブログとして
    別途保存される。ci.md の CI 基盤を利用する）
- **検証手順**: 監査時、記録済みの外部アンカー値と、ログファイル末尾
  レコードの実際の `hash` を突き合わせる。一致すれば末尾切り詰めが
  発生していないことを保証できる。

## 8. セキュリティ上の注意

- **秘密情報の非混入**: ログ（特に `attempt.outcome.reason`・
  `loop_outcome.outcome.reason`・`loop_failure.error`）に API キー・
  トークン・パスワードを書かない。これは `reason` 文字列を生成する側
  （検証ゲート・取り込み判断・`SelfRepairError` の各実装）の責務であり、
  本仕様書・`logging.rs` 自体は文字列の内容を検閲しない
  （security.md「秘密情報の混入防止」）。
- **信頼できない入力としての扱い**: ログファイルはディスク上の平文
  JSON Lines であり、`self_repair` クレート外からも書き換え得る。
  利用側（監査者・CLI バイナリ）は 6 節の `verify_chain` を通過した
  後にのみ内容を判断根拠として利用し、通過前のログを信頼しない
  （A08「判定の迂回経路を作らない」と同じ思想）。
- **エスケープの前提**: `reason` 等の値に改行・`"` ・バックスラッシュ等の
  JSON 特殊文字が含まれても、`serde_json` の標準エスケープにより
  1 行 1 レコードの構造は保たれる（文字列連結による手組み JSON 生成は
  行わない。security.md A03「シェル呼び出しでユーザー入力を直接展開しない」
  と同種のインジェクション対策方針を JSON 出力にも適用したもの）。
- **fail-closed の一貫性**: `verify_chain` は 1 箇所の不一致でも即座に
  `Err` を返し、部分的な信頼を認めない。本仕様書に記載した検証手順から
  この挙動を緩和する変更（許容誤差の導入等）は、ガードレール閾値・
  テスト許容誤差の変更に準じユーザー承認が必要（security.md「自己修復
  ループ固有のガードレール」）。

## 9. 未着手事項（スコープ外）

- CLI バイナリ（`self-repair run`/`verify-log`）の実装
  （[`docs/guardrail-self-repair-cli.md`](./guardrail-self-repair-cli.md)
  3.1〜3.2 節が CLI 境界を確定済みだが、実装自体は後続タスク）
- 外部アンカー（7 節）の自動化実装（運用指針の文書化のみで、自動化は行わない）
- `self-repair` の呼び出し元（`SelfRepairLoop::run` の実行主体）から
  `LogWriter`/`verify_chain` への結線（本イシューはログ機構自体の提供
  までがスコープ。呼び出し元との結線は別イシュー）
