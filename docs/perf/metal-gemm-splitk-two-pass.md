# Metal GEMM split-K 2 パスカーネル 実装記録（イシュー #1474）

イシュー #1474「split-K 2 パス GEMM カーネルと選択純関数 `should_split_k` を opt-in で実装する」
の実装記録。`docs/backend-metal-splitk-decision.md`（#810・#1308 で「採用検討推奨」確定）を受け、
機構の実装と正しさの自己検証を行う。**本イシューのスコープは機構実装と正しさ検証のみ**（性能 A/B
は #1475、`select_for_device`／`dispatch_auto` への本番結線可否は #1476）。本 PR では
`dispatch_auto` は変更しない（opt-in・未結線）。

## 1. 実装概要

- `crates/backend-metal/src/shaders/gemm.metal`: function constant `SPLIT_K_ENABLED`（index 16）・
  `SplitKParams` 構造体（`partitions`/`k_per_partition`/`reserved0`/`reserved1`）・
  `gemm_simdgroup_tiled` への K 区間分割（パス 1。3 次元 dispatch・`tgid.z` がパーティション番号）・
  縮約カーネル `gemm_splitk_reduce`（パス 2。パーティション昇順の固定順序 Neumaier 補償和。
  `atomic` 不使用）を追加。`SPLIT_K_ENABLED=false` では `k_begin=0`／`k_end=dims.k`／`c_out=c` に
  畳み込まれ既存経路と演算列が完全同一（bit 同一）。
- `crates/backend-metal/src/tile.rs`: MLX Case 1 型の選択純関数 `should_split_k`／
  `should_split_k_with`（`SplitKParams`〈パラメータ化済み〉・`SplitKPlan`）を objc2 非依存で追加。
  MLX Case 1 単体では `(256,256,*)`・正方 512 以上も true になってしまうため、#1308 の並列度条件
  （`groups = ceil(m/bm)*ceil(n/bn) < max_groups`）を明示パラメータとして追加している。
- `crates/backend-metal/src/pipeline.rs`／`spec_source.rs`: `GemmGateConstants.split_k_enabled`・
  `SpecParams.split_k_enabled` を追加（既存呼び出し元はすべて `false`）。
- `crates/backend-metal/src/gemm.rs`: `SplitKParams`（repr(C)・レイアウト一致テスト付き）・
  `encode_dispatch_tiled` の split-K 対応拡張・split-K 専用パイプラインキャッシュ・
  `pipeline_splitk_reduce`・opt-in 公開入口（`dispatch_split_k_strided_prepared`／
  `dispatch_split_k_strided_prepared_with_plan`）・`SplitKRoute`／`SplitKFallbackReason`・診断
  カウンタを追加。スクラッチ確保失敗・検証不成立時は `unwrap`/`expect` を使わず
  `SplitKRoute::Classic { reason }` で既存の classic 経路（`dispatch_strided_tiled_prepared` +
  `tile::select_for_device`）へ fail-closed フォールバックする。
- `crates/backend-metal/tests/shader_source_evidence.rs`: function constant 宣言列挙・
  `gemm_splitk_reduce` の `atomic` 不使用／境界検査／固定順序ループ／Neumaier 補償和を静的検証する
  Linux 実行可能テストを追加。
- `crates/backend-metal/tests/common/splitk_parity_baseline.rs`（新規）: §5 の非後退契約（後述）。

## 2. REQ-8 境界検査

- パス 1: `part >= sk.partitions` で grid depth 超過をガード（早期 `return`）。`k_begin >= k_end`
  で空区間をガード。既存の K 端数境界述語（`bk_eff = min(BK, k_end - p0)`）は区間端で切れるため
  K 端数・区間境界の手動検査を維持。
- パス 2（`gemm_splitk_reduce`）: `row >= dims.m || col >= dims.n` で縮約 grid の M×N 境界を手動
  チェック。

## 3. AC 対応表

| AC | 内容 | 結果 |
|----|------|------|
| AC-1 | run-to-run bit 同一（split 数 2/4/8/16/32） | 達成（`tests/gemm_splitk_bit_match.rs`。§4） |
| AC-2 | classic 経路・CPU 参照実装との REQ-2 判定（対象 9 形状 + K 端数境界 2 形状 × NN/NT/TN/TT） | **厳密ゼロ fail 判定**（§5.5。実測ベースライン非後退方式は未承認のため差し戻し・8/11 形状で既知 FAIL） |
| AC-3 | `should_split_k` の Linux 単体テスト | 達成（`tile.rs` 内 `#[cfg(test)]`） |
| AC-4 | REQ-8 手動境界検査維持 | 達成（§2） |
| AC-5 | 本番 `dispatch_auto`／`select_for_device` 不変 | 達成（§6） |
| AC-6 | `cargo fmt`／`clippy -D warnings` green・内部ホスト名なし | 達成（§7） |

## 4. AC-1: run-to-run bit 同一（実機実測。Apple M4 Max）

`cargo test -p fandhe-ai-backend-metal --release --features internal-diagnostics --test gemm_splitk_bit_match -- --ignored --nocapture`

対象 9 形状 × partitions ∈ {2,4,8,16,32} の split-K 経路が 2 回実行で bit 完全一致、かつ
`MetalGemm::new()` の `dispatch_auto`／`dispatch_tiled_prepared`（classic 経路）が split-K 追加後も
2 回実行で bit 完全一致することを確認（`docs/perf/logs/metal-gemm-splitk-two-pass-1474/bit_match.log`）。

```
test classic_dispatch_auto_remains_run_to_run_bit_exact_after_split_k_addition ... ok
test split_k_run_to_run_bit_match_for_target_shapes_and_partition_counts ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.11s
```

## 5. AC-2: split-K 経路の正しさ判定方式（実機実測で判明した構造的特性）

### 5.1 当初計画との差分

実装計画では AC-2 を「classic 経路・CPU 参照実装のいずれとも REQ-2 統一複合判定（相対誤差
1e-3 未満または絶対誤差 1e-5 未満）で厳密ゼロ fail」としていた。実機実測（Apple M4 Max）の結果、
この厳密ゼロ fail は split-K の構造的特性により達成できないことが判明したため、判定方式を
**実測ベースライン非後退方式**（`crates/backend-cuda/tests/common/parity_baseline.rs` と同型。
`docs/spec/04-requirements.md` REQ-2 2026-09-02 追記の適用）へ変更した。tolerance 定数
（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）自体は一切変更していない。

### 5.2 原因の特定

`gemm_splitk_reduce` を単純逐次加算から Neumaier 改良版 Kahan 補償和へ変更しても fail は解消
しなかった（(32,32,8192) NN: 是正前後とも `fail_count=1/1024`・`max_rel_err≈1.07e-3`）。これは
誤差の発生源が縮約側の結合方法ではなく、**パーティション分割そのもの**（各パーティションが
独立した FMA 連鎖で部分和を求め、単一の連続 K ループとは異なる結合順序で最終的に加算される）に
あることを示す。

追加診断（`partitions` 別感度）: `(64,64,2048)` を `partitions ∈ {2,4,8,16,32}` で計測したところ、
**最小分割の `partitions=2` の時点で既に一部要素が閾値を超過**した（`max_rel_err=0.002030`。
`fail_count` 自体は 0 だが同一要素で `abs`・`rel` 双方が閾値超過する組合せが `partitions=4` 以降
発生）。分割数を増やすほど `fail_count` は非減少に推移する（2/4/8/16/32 partitions でそれぞれ
0/1/2/2/2 件、4096 要素中）。

古典（classic）経路（`dispatch_strided_tiled_prepared` + `tile::select_for_device`）は同一乱数
シードの `(32,32,8192)` で CPU 参照実装（`matmul_reference_fma`）と **bit 完全一致**
（`fail_count=0/1024`・`max_abs_diff=0.0`）することを別途確認しており、この誤差が split-K 固有の
現象であることを裏付ける。

### 5.3 対象 11 形状 × 4 転置パターンの全数実測結果

`docs/perf/logs/metal-gemm-splitk-two-pass-1474/parity_survey_all_shapes.log`
（`fandhe_ai_backend_cpu::parity::compare` の非 panic 版で全 44 組合せを走査）。4 種の転置
パターン（NN/NT/TN/TT）は論理的に同一の行列積を異なる物理レイアウトで計算するだけであり、
実測でも常に同一の集計値になることを確認した（そのため `(m, n, k)` 単位で 1 行のみ記録する）。

| (m, n, k) | total | fail_count | mean_abs_diff | max_abs_diff | max_rel_err |
|---|---|---|---|---|---|
| (32, 32, 2048) | 1024 | 0 | 9.04e-6 | 8.39e-5 | 2.02e-4 |
| (32, 32, 4096) | 1024 | 0 | 1.68e-5 | 2.02e-4 | 1.70e-4 |
| (32, 32, 8192) | 1024 | 1 | 3.53e-5 | 3.36e-4 | 1.07e-3 |
| (64, 64, 2048) | 4096 | 2 | 8.76e-6 | 9.54e-5 | 3.88e-3 |
| (64, 64, 4096) | 4096 | 0 | 1.75e-5 | 2.06e-4 | 7.52e-4 |
| (64, 64, 8192) | 4096 | 2 | 3.53e-5 | 3.05e-4 | 2.23e-3 |
| (128, 128, 2048) | 16384 | 2 | 8.94e-6 | 1.18e-4 | 3.35e-3 |
| (128, 128, 4096) | 16384 | 8 | 1.78e-5 | 1.64e-4 | 8.53e-3 |
| (128, 128, 8192) | 16384 | 2 | 3.53e-5 | 3.81e-4 | 1.61e-3 |
| (64, 64, 2056)（K 端数） | 4096 | 1 | 9.06e-6 | 1.26e-4 | 1.27e-3 |
| (128, 128, 2064)（K 端数） | 16384 | 2 | 9.13e-6 | 1.22e-4 | 1.31e-1 |

`(128, 128, 2064)` の `max_rel_err` が突出しているのは、真値が 0 近傍の要素で桁落ちが起きた
（`abs_diff` 自体は他行と同水準）ためであり、K 端数固有の異常ではなく split-K 全般が持つ
「近ゼロ要素での相対誤差外れ値」という性質の一例と判断した。

11 形状中 3 形状（(32,32,2048)・(32,32,4096)・(64,64,4096)）は厳密ゼロ fail を満たすが、残り 8
形状は満たさない。よって「実機実測で成立が確認された形状に限り厳密ゼロ fail 判定とする」という
CUDA 側方式をそのまま適用すると大半の形状が対象から漏れるため、本イシューでは**全形状に対し
実測ベースライン非後退方式を一律適用**する設計とした（`tests/common/splitk_parity_baseline.rs`）。

### 5.4 非後退契約の実装

`crates/backend-metal/tests/common/splitk_parity_baseline.rs::BASELINES`
に上記表の実測値をそのまま記録し（表記丸め対応で天井値は最終桁 +1）、
`assert_no_split_k_parity_regression` が `total`・`fail_count`・`mean_abs_diff`・`max_abs_diff`・
`max_rel_err` の 5 点非後退を検査する。`crates/backend-metal/tests/gemm_splitk_parity.rs` の
受け入れテストはこの非後退契約で判定する（`docs/perf/logs/metal-gemm-splitk-two-pass-1474/parity.log`）。

```
test split_k_matches_classic_and_cpu_reference_for_target_shapes_and_transpose_patterns ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.15s
```

いずれのケースも `dispatch_split_k_strided_prepared` の戻り値が `SplitKRoute::Split` であることを
assert しており、フォールバック（classic 経路）による自明合格ではない。

### 5.5 判定方式の差し戻し（PR #1496 codex-review 指摘。2026-09-09）

上記 §5.1〜§5.4 の実測ベースライン非後退方式（§5.4 実装）は、PR #1496 の codex-review により
P1 指摘を受けた: spec REQ-2 2026-09-02 追記（TF32/f16 Tensor Core 経路の受け入れ判定方式）は
CUDA 側 TF32/f16 経路限定であり、Metal f32 split-K への適用拡張・具体的な baseline 値の追加には
別途ユーザー承認が必要（`.claude/rules/coding-rust.md` の「バックエンド間数値一致テストの許容
誤差を単独で緩和しない」原則）。当該 PR には承認記録がなかったため、承認を得るまでの間は
`crates/backend-metal/tests/gemm_splitk_parity.rs` の判定方式を `fandhe_ai_backend_cpu::parity::
assert_parity` による厳密ゼロ fail 判定へ差し戻した。

この差し戻しにより、§5.3 の表で `fail_count > 0` の 8 形状（(32,32,8192)・(64,64,2048)・
(64,64,4096) 以外の全形状）は実機（Apple Silicon）で本テストを実行すると `#[ignore]` テストが
FAIL する既知の状態になる（実測値・原因は §5.2／§5.3 のまま変わらない）。`tests/common/
splitk_parity_baseline.rs::BASELINES` の実測データ・`assert_no_split_k_parity_regression` 関数
自体はコードとして保持しているが、`gemm_splitk_parity.rs` からは現在未使用。適用拡張の是非・
具体的な baseline 値の承認を得た場合は、判定方式を再度実測ベースライン非後退方式へ切り替える
（tolerance 定数自体は §5.1 のとおり変更しない）。

適用拡張の是非は `docs/backend-metal-splitk-parity-judgment-decision.md`（イシュー #1511）で
決定記録 draft 化した（候補比較・推奨〈baseline 非後退方式・全 11 形状一律適用〉・spec 起票用
本文案を整理。承認記録は draft 作成時点では未承認のまま。承認され次第、判定方式の切替は
別イシュー #1512、`SPLIT_K_NUMERIC_CONTRACT_APPROVED` ゲート解除は別イシュー #1513 で扱う）。

### 5.6 公開入口の数値契約ゲート（PR #1496 codex-review P1 指摘・再対応。2026-09-09）

§5.5 の差し戻しは受け入れテスト（`#[ignore]`・CI 非実行）の判定方式のみを対象としており、
自動判定入口 `MetalGemm::dispatch_split_k_strided_prepared`（`should_split_k` が `Some` を返す
形状であれば split-K を実行して `Ok(SplitKRoute::Split)` を返す）自体は無条件で成功を返して
いた。この入口はテストコード以外からも呼び出し可能な `pub fn` であり、`dispatch_auto` へ
未結線であること・opt-in であることは、§5.3 で判明した「対象 11 形状中 8 形状が REQ-2 統一
複合判定を満たさない」という事実と、この公開入口が無条件に成功を返す実装との不整合を解消し
ない（PR #1496 codex-review P1 再指摘。`crates/backend-metal/src/gemm.rs:2401` 付近）。

対応として `crate::gemm::SPLIT_K_NUMERIC_CONTRACT_APPROVED`（既定 `false`）を追加し、
`dispatch_split_k_strided_prepared` は本フラグが `false` の間 `should_split_k` の判定結果に
関わらず常に classic 経路へフォールバックする（新設 `SplitKFallbackReason::
NumericContractPendingApproval`）よう変更した。§5.2 の実測が示すとおりこの誤差は入力データにも
依存する（近ゼロ要素での相対誤差外れ値）ため、形状単位の allowlist（例えば §5.3 で厳密ゼロ
fail を満たした 3 形状のみ許可する）では数値契約を機構的に保証できないと判断し、適用拡張の
承認（§5.5 と同じユーザー承認事項）を得るまでは自動判定入口を一律無効化する設計とした。

`_with_plan` 版（`dispatch_split_k_strided_prepared_with_plan`）はこのゲートの対象外のまま
維持する（AC-1 bit-match テスト・AC-2 parity テストが split-K 経路自体の構造的妥当性を明示的に
検証するための診断専用入口という位置づけは §5.1〜§5.5 と変わらない）。`gemm_splitk_parity.rs`
は `should_split_k` が算出した計画を `_with_plan` へ明示的に渡す形へ更新し、判定方式（§5.5 の
厳密ゼロ fail）自体は変更していない。

Linux 相当チェック（`cargo test -p fandhe-ai-backend-metal`・`cargo clippy --workspace
--all-targets --all-features -- -D warnings`）は green（実機 `#[ignore]` テストの再実行は
Apple Silicon 実機依存のため本対応では未実施。§5.5 の既知 FAIL 状態は変更なし）。

### 5.7 `_with_plan` 版の可視性ゲート（PR #1496 codex-review P1 再指摘対応。2026-09-09）

§5.6 は自動判定入口（`dispatch_split_k_strided_prepared`）のみを数値契約ゲートで塞いだが、
`_with_plan` 版（`dispatch_split_k_strided_prepared_with_plan`）は「AC-1／AC-2 の診断専用入口」
という位置づけのまま無条件 `pub fn` だった。この関数は `should_split_k` の自動判定・
`SPLIT_K_NUMERIC_CONTRACT_APPROVED` ゲートいずれも経由しないため、crates.io 公開クレートの
利用者が `should_split_k(32, 32, 8192)` 等の計画を自分で構築して直接渡せば、§5.3 で厳密ゼロ
fail が不成立と判明済みの形状であっても split-K を無条件に実行して `Ok(SplitKRoute::Split)` を
返してしまう（PR #1496 codex-review P1 再指摘。「診断専用」というコメントだけでは本番利用を
制限できず AGENTS.md「数値契約の統一」に違反する公開経路が残っていた）。

対応として `MetalGemm::encode_tiled_prepared`（イシュー #1259。`Cargo.toml` の
`internal-diagnostics` feature コメント参照）と同型の 2 分岐可視性ゲートを適用した:
`internal-diagnostics` feature（既定 OFF）を有効化したビルドでのみ `pub`、既定ビルドでは
`pub(crate)` に絞りクレート外部から到達不能にする。共通実装は `pub(crate)` の `_impl` 関数へ
切り出し、2 分岐は薄い委譲ラッパーとした（`crates/backend-metal/src/gemm.rs::MetalGemm::
dispatch_split_k_strided_prepared_with_plan`／`_impl`）。`Self::dispatch_split_k_strided_
prepared`（自動判定入口。承認後に `SPLIT_K_NUMERIC_CONTRACT_APPROVED=true` へ切り替えた際に
`_with_plan` を内部呼び出しする）はクレート内部呼び出しのためどちらの分岐でも到達できる。

AC-1（`tests/gemm_splitk_bit_match.rs`）・AC-2（`tests/gemm_splitk_parity.rs`）は本 feature を
要求する `required-features = ["internal-diagnostics"]` を `Cargo.toml` の `[[test]]` へ追加した
（`gemm_transpose_route_ab_bench` の `required-features` 先例と同型）。両テストとも macOS 実機
限定（`#![cfg(target_os = "macos")]`）・`#[ignore]` のため、CI（GitHub ホステッド・ubuntu-latest）
での実行対象範囲は変わらない。実機再実行コマンドは `--features internal-diagnostics` を追加した
（§4 のコマンド例を更新）。

Linux 相当チェック（`cargo test -p fandhe-ai-backend-metal --all-features`・`cargo clippy
--workspace --all-targets --all-features -- -D warnings`）は green。実機 `#[ignore]` テストの
再実行は Apple Silicon 実機依存のため本対応では未実施（§5.3〜§5.5 の既知 FAIL 状態・§5.2 の
bit-match／parity 実測値自体はコード変更なしのため不変）。

### 5.8 判定方式の再切替（イシュー #1512。2026-09-10 承認後）

§5.5 の差し戻し後、イシュー #1511 で適用拡張・baseline 値がユーザー承認され
（`docs/backend-metal-splitk-parity-judgment-decision.md` §7・2026-09-10 ユーザー承認）、
本イシュー（#1512）で `crates/backend-metal/tests/gemm_splitk_parity.rs` の判定を
`fandhe_ai_backend_cpu::parity::assert_parity`（厳密ゼロ fail 判定）から
`crates/backend-metal/tests/common/splitk_parity_baseline.rs::assert_no_split_k_parity_
regression`（実測ベースライン非後退方式。§5.4 実装をそのまま再利用）へ再切替した。
対象 11 形状すべてへ一律適用する（§3 の決定どおり、CUDA 先例の形状二分方式は不採用）。
`crates/backend-metal/tests/common/splitk_parity_baseline.rs::BASELINES` の数値自体は §5.4
実装時点から一切変更していない（承認記録の §7 表と一致）。tolerance 定数
（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）も不変。`SPLIT_K_NUMERIC_CONTRACT_APPROVED`
（自動判定入口 `dispatch_split_k_strided_prepared` のゲート）は本イシューのスコープ外で
`false` のまま（解除は別イシュー #1513）。

**Linux 実行可能な契約テストの新設**: `gemm_splitk_parity.rs` は macOS 実機限定
（`#![cfg(target_os = "macos")]`）・`required-features = ["internal-diagnostics"]` のため、
`common::splitk_parity_baseline` モジュール自体は Linux CI（GitHub ホステッド・
ubuntu-latest）で型検査カバレッジがゼロだった。本イシューで `crates/backend-metal/tests/
splitk_parity_baseline_contract.rs`（macOS cfg・`required-features` いずれも持たない通常の
`[[test]]`）を新設し、`BASELINES` が承認記録からの独立転記と全一致すること（改竄・緩和の
黙った混入の検出）・`find_baseline` の契約・`assert_no_split_k_parity_regression` の 5 種
falsification（`crates/backend-cuda/tests/parity_nonregression.rs` と同方針）を通常
`cargo test -p fandhe-ai-backend-metal` の対象へ含めた（10 tests、全 pass）。

Linux 相当チェック（`cargo fmt --all --check`・`cargo clippy -p fandhe-ai-backend-metal
--all-targets --all-features -- -D warnings`〈本クレート単体では green。`--workspace` は
`backend-cuda` 側の無関係な pre-existing dead-code エラーにより本 PR 実行環境では失敗する。
`git stash` で本変更を除いても同一エラーが再現することを確認済みで、本イシューの変更による
ものではない〉・`cargo test -p fandhe-ai-backend-metal --all-features`・`cargo check -p
fandhe-ai-backend-metal --tests --all-features --target aarch64-apple-darwin`・
`make check-cross-metal-tests`・`make doc-warnings`・`make test`）は green。

**M4 Max 実機実測（イシュー #1512 記入欄）**: 未実測。`docs/perf/logs/metal-gemm-splitk-
parity-baseline-1512/`（`run_parity.sh`・事前登録判定規則・README）を用意し、Mac セッション
での実行を申し送る。実測完了後、本節へ以下を追記する: (a) 11 形状 × 4 パターン全 44 組合せが
記録済みベースライン ceiling 以下であることの確認結果、(b) §5.3 の表との実測値差分の有無、
(c) `uptime_during.log` から見た load average 推移。

### 5.9 公開入口の数値契約ゲート解除（イシュー #1513。2026-09-10）

§5.8 で判定方式を承認済み baseline 非後退方式へ再切替した後、本イシューで残る作業
（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` の `true` への切り替え）を実施した。

**切替内容**: `crates/backend-metal/src/gemm.rs:135`（現在値。切替前は `false`）を
`false` → `true` へ変更した。これに伴い `dispatch_split_k_strided_prepared`（自動判定
入口）は `crate::tile::should_split_k` が `Some` を返す形状（承認済み 11 形状を含む）で
実際に split-K 2 パス経路（`SplitKRoute::Split`）を実行するようになり、`None` を返す
形状のみ classic 経路（`SplitKFallbackReason::NotEligible`）へフォールバックする。
承認出典は `docs/backend-metal-splitk-parity-judgment-decision.md` §7（2026-09-10
ユーザー承認: 適用拡張可・全 11 形状一律 baseline・`BASELINES` 11 行承認）。

**保持した公開 API（後方互換）**: `SplitKFallbackReason::NumericContractPendingApproval`
variant は crates.io 公開済みクレート（`fandhe-ai-backend-metal`）の public enum である
ため削除しない（破壊的変更を避ける）。doc comment を「承認済みのため現在は到達しない・
後方互換のため保持」へ書き換えた。ゲート解除後、本 variant を実際に返す経路はコード上
存在しない（`gemm.rs::dispatch_split_k_strided_prepared` の分岐は
`SPLIT_K_NUMERIC_CONTRACT_APPROVED` が `true` の場合 `NotEligible` のみを使う）。

**本番経路は不変**: `dispatch_auto`／`crate::tile::select_for_device` への結線は本
イシューのスコープ外（#1516・Phase 2 へ引き継ぎ）。`crate::ops::MetalBackendOps::gemm`
（`dispatch_auto` が担う本番経路）は `dispatch_split_k_strided_prepared` を呼ばない
ことを実装時点で確認済み:

```
$ grep -rn "dispatch_split_k_strided_prepared" crates/backend-metal/src/ops.rs
（出力なし）
```

したがって本切替は `MetalBackendOps::gemm` の挙動・性能に一切影響しない。

**追記（#1516）**: 上記「本番経路は不変」は #1513 時点（本節記述時点）の事実であり、
#1516 で `dispatch_auto` へ split-K 2 パス経路の**定数ゲート付き結線**を実装した
（既定 `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = false`）。ゲートが `false` の間は
本段落が記述する状態（`dispatch_auto` は `dispatch_split_k_strided_prepared` 系を呼ばない）
と実質的に同一の挙動（bit 同一の classic 経路）が維持される。詳細・切替条件は
`docs/backend-metal-splitk-decision.md` §5「本番結線（#1516）」を参照。

**framework-compare A/B**: 「計測対象なし」と判断した。本番経路（`dispatch_auto`・
`MetalBackendOps::gemm`）へのコード変更がゼロであるため（上記出力参照）、
#1272 §5.11・#1476 の先例（本番結線なしの変更は framework-compare 計測対象なし）と
同型の判断。

**新規テスト**:
- `crates/backend-metal/tests/splitk_parity_baseline_contract.rs::
  approved_baseline_shapes_are_split_k_eligible`（Linux 実行可能・CI 対象）:
  承認済み `BASELINES` 11 行すべてについて `should_split_k(m, n, k)` が `Some` を
  返し `partitions >= 2` であることを assert する（ゲート解除後の公開入口が split-K
  経路を実行する前提となる純関数部分の集合レベル検査）。
- `crates/backend-metal/tests/gemm_splitk_auto_entry_parity.rs`（`#![cfg(target_os =
  "macos")]`・`#[ignore]`・`required-features` なし）: 公開入口
  `dispatch_split_k_strided_prepared` 自体を直接呼ぶ受け入れテスト 2 件。
  - `auto_entry_dispatches_split_k_for_eligible_shapes_and_matches_baseline`:
    承認済み 11 形状 × NN/NT/TN/TT で戻り値が `SplitKRoute::Split` であること・
    `partitions` が `should_split_k` の算出値と一致すること・出力が
    `assert_no_split_k_parity_regression` を満たすことを確認する。
  - `auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes`:
    `should_split_k` が `None` を返す形状（`(512,512,512)`・`(64,64,63)`）で
    公開入口が `SplitKRoute::Classic { reason: NotEligible }` を返し
    （`NumericContractPendingApproval` では**ない**ことを直接 assert）、CPU 参照
    実装と bit 完全一致することを確認する。

`required-features` を指定しない設計により、両テストは
`cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`
の型検査対象に含まれる（`required-features` 付きテストが同ターゲットの型検査
対象に含まれないという #1526 の観察〈`docs/backend-metal-splitk-parity-judgment-
decision.md` 系の教訓〉を本テストでは回避する）。

**Linux 相当チェック結果**（実装時点）:

```
cargo fmt --all --check                      … green
cargo clippy -p fandhe-ai-backend-metal \
  --all-targets --all-features -- -D warnings  … green
cargo test -p fandhe-ai-backend-metal --all-features … green（新規 1 件含む Linux 実行対象）
cargo check -p fandhe-ai-backend-metal --tests \
  --target aarch64-apple-darwin (+ --all-features) … green（新規 #[ignore] 2 件の型検査含む）
make check-cross-metal-tests / check-cross-cpu-tests / doc-warnings / test / deps-forbidden … green
```

（`cargo clippy --workspace --all-targets --all-features` は本実行環境で `backend-cuda`
側の無関係な pre-existing エラーにより失敗するため、§5.8 と同様変更を除いても同一エラーが
再現することを確認したうえで `-p fandhe-ai-backend-metal` 限定の結果を正とする。）

**M4 Max 実機実測（本イシュー #1513 記入欄）**: 未実測。`docs/perf/logs/metal-gemm-
splitk-auto-entry-1513/`（`run_auto_entry.sh`・事前登録判定規則・README）を用意し、
Mac セッションでの実行を申し送る。実測完了後、本節へ以下を追記する:

(a) 新規 `#[ignore]` テスト `auto_entry_dispatches_split_k_for_eligible_shapes_and_
    matches_baseline` が 11 形状 × 4 パターン = 44 組合せすべてで `SplitKRoute::Split`
    へ到達し、記録済みベースライン ceiling 以下であることの確認結果
(b) `auto_entry_falls_back_to_classic_not_eligible_for_non_split_k_shapes` が
    非対象形状で `NotEligible`（`NumericContractPendingApproval` ではない）を返す
    ことの確認結果
(c) 既存 `gemm_splitk_bit_match.rs`（`classic_dispatch_auto_remains_run_to_run_bit_
    exact_after_split_k_addition` を含む）・`gemm_splitk_parity.rs` の非後退確認結果
(d) `uptime_during.log` から見た load average 推移

## 6. AC-5: 本番経路の非後退確認

`tile::select`／`select_for_device`／`select_with_occupancy_for_device`・`MetalGemm::new`／
`dispatch_auto`・`MetalBackendOps::gemm` は無変更。以下の既存 `#[ignore]` テスト群を実機
（Apple M4 Max）で再実行し非後退を確認した
（`docs/perf/logs/metal-gemm-splitk-two-pass-1474/existing_regression_check.log`）:

`gemm_fine_barrier_bit_match`・`gemm_swizzle_bit_match`・`gemm_simdgroup_parity`・
`gemm_auto_parity`・`gemm_dynamic_tile_parity`・`gemm_strided_parity`・`gemm_transposed_parity`・
`gemm_resident_parity`・`backend_ops_real_device` — 全て pass（既知の pre-existing FAIL なし）。

## 7. Linux 相当チェック（AC-6）

`cargo fmt --all -- --check` green。`cargo clippy --workspace --all-targets --all-features -- -D
warnings` green。`cargo test -p fandhe-ai-backend-metal`（Linux 相当。`tile::` 単体テスト・
`gemm.rs` レイアウトテスト・`shader_source_evidence`・example 群）275 passed / 0 failed（64
ignored は実機依存）。`make check-cross-metal-tests`（`aarch64-apple-darwin` 型検査）green。

## 8. スコープ外（Issue 追跡）

- 性能 A/B（対象 9 形状・5 回計測中央値・専有ゲート）: #1475。
- `select_for_device`／`dispatch_auto`／`MetalBackendOps::gemm` への結線可否: #1476。
  **結線せずと確定**（性能判定 undetermined・数値契約未承認の 2 ブロッカー。
  `docs/backend-metal-splitk-decision.md` §4）。
- f16／hfrag カーネルの split-K・TT 以外の一般 stride・`gemm_bias_act` 融合経路への適用・
  `TileClassMode::Split` との併用は対象外。
- 縮約側アルゴリズムの追加改善（例: ツリー型縮約・より高精度な補償和の組合せ）は、§5.2 の原因
  特定（分割そのものに起因）を踏まえると効果が限定的と見込まれ、本イシューでは追加調査しない。
