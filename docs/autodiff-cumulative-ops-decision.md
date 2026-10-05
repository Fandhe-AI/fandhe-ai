# 累積演算（`cummax`・`cummin`・`logcumsumexp`）の CPU 実装記録（イシュー #2636）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-fft-ops-decision.md`（#2631）の「共有カーネルを
`tensor-core` に置く」方式を再適用した実装記録であり、**承認記録ではない**（facade 公開形の承認は
#2677 で依頼中。公開自体は承認後の #2678）。

## 0. 結論

- 累積最大／最小（値と索引）と累積 `logsumexp` を CPU 参照実装として内部クレートへ追加した。
  - 共有カーネル: `fandhe_ai_tensor_core::cumulative`（`cummax_host`／`cummin_host`／`logcumsumexp_host`／
    `logcumsumexp_vjp_host`・`cumulative_layout`）。走査規則・数値契約の単一情報源。
  - `BackendOps` に既定 `Unsupported` のメソッド 3 件（`scan_cummax`／`scan_cummin`／`scan_logcumsumexp`）を
    追加（非破壊拡張）。CPU は共有カーネルを呼ぶだけの override。CUDA／Metal は変更なし。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::cumulative_ops`（`cummax`／`cummin`／`logcumsumexp`）。
    専用 `Op` 3 variant（`Op::Cummax`／`Op::Cummin`／`Op::Logcumsumexp`）と VJP は `grad.rs`。
- facade 公開は行わない。`CumulativeOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。既定 `Unsupported` → 共有ホストカーネルへのフォールバックで動作する。
  実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定（事実のみ）

- 3 演算は REQ-9 Tier 2 の列挙（topk／sort／cumsum）に名前がない。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、
  #2636 の受入条件（内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について
  承認済みとは記録しない（承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| 共有カーネル | `tensor-core/src/cumulative.rs` | レイアウト検査（`CumulativeLayout`）・forward 3 種・`logcumsumexp` の VJP |
| バックエンド抽象 | `tensor-core/src/backend_ops.rs` | `scan_*` 3 件（既定 `Unsupported`） |
| CPU | `backend-cpu/src/ops.rs` | 共有カーネルを呼ぶだけの override（`contiguous()` 済みスライスを渡す） |
| autodiff | `autodiff/src/cumulative_ops.rs`・`tape.rs`・`grad.rs` | 自由関数 3 件・`Op` 3 variant・VJP |

- 共有カーネル方式（FFT と同じ）を選び、`cumsum` の「`eval` と `backend-cpu::scan` の複製＋bit 一致テスト」方式は
  採らなかった。複製による乖離とその検証コストを避けるため。
- 命名規律: 素の `fn cummax`／`fn cummin`／`fn logcumsumexp` は `autodiff/src/cumulative_ops.rs` の各 1 件のみ
  （workspace インベントリが固定）。trait メソッドは `scan_*`、共有カーネルは `*_host`。
- `Var`／`Tape` に inherent メソッドは足していない（足すと facade 公開面が広がる）。
- `Op::Cummax`／`Op::Cummin` は `Op::Sort`／`Op::Topk` と payload が同じ（`index: Tensor<i32>`）だが、VJP の腕は
  **独立**にした。cummax の索引は重複する（例 `[3,1,2]` → `[0,0,0]`）ため `ScatterReduce::Add` は必須で、
  「重複添字は存在しない」ことを前提にした Sort／Topk の腕のコメントと矛盾させないため。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応。f64 自動微分経路は対象外。

## 3. 数値契約

`cummax`／`cummin`:

- 算術を含まない比較・選択のみ。値は入力の要素と bit 一致（`±0` の符号も保存）。lane（`dim` 以外の軸の組）ごとに
  `dim` 昇順の逐次走査。
- 更新規則: 現在値 `cur` が NaN でなければ、`x >= cur`（`cummin` は `x <= cur`）または `x` が NaN のとき
  `cur = x`・索引更新。`cur` が NaN になった後は NaN 入力でのみ索引が更新される。**タイは後勝ち・NaN は伝播**。
  この規則は PyTorch 2.14.0 の実測（fixture のタイ・NaN・`±0` ケース）で確定した（実装前は ATen からの推定）。
- VJP: `scatter_with_fallback`（`ScatterReduce::Add`。出力位置ごと `f64` アキュムレータ・1 回 downcast。
  coding-rust.md の長軸縮約契約）。

`logcumsumexp`:

- lane ごとに `f64` アキュムレータ。`acc = log_add_exp(acc, x)` の各時点を 1 回だけ `f32` へ downcast して
  書き出し、downcast 値は読み戻さない。先頭要素は `acc = x[0] as f64` で種を置き `out[0]` は bit 一致で往復する。
- `log_add_exp(a, b)` の分岐順: どちらかが NaN なら NaN → `a == b` かつ非有限なら `a`（`inf - inf = NaN` を
  避ける）→ それ以外は `max + ln_1p(exp(min - max))`。
- `exp`／`ln_1p` は libm 依存のためクレート間 bit 同一は受入条件にしない（REQ-2 判定で比較）。同一入力の
  run-to-run は bit 一致。`mul_add` は使わず、matmul 系 FMA 契約には触れない。
- VJP: `d_x[j] = Σ_{i>=j} g[i]·exp(x[j] - out[i])` を O(n) の逆向き再帰で計算する
  （`T[n-1] = g[n-1]`・`T[j] = g[j] + exp(out[j] - out[j+1])·T[j+1]`・`d_x[j] = exp(x[j] - out[j])·T[j]`）。
  `out` は forward と同じ `log_add_exp` で `f64` のまま lane 内で再計算する（`f32` へ丸めた記録値は使わない）。
  `out` は単調非減少で `x[j] <= out[j]` のため指数の引数は常に 0 以下で overflow しない。O(n²) の直接形は
  テスト用オラクル限定。作業バッファは lane 長の `Vec<f64>` 1 本のみ。

非有限入力は 3 演算とも事前に拒否せず伝播する（`cumsum`・FFT と同じ）。

## 4. 境界検査

- `cumulative_layout` が確保・実体化より前に `dim` 範囲（rank 0 を含む）と要素数／バイト数の `checked_mul`・
  `isize::MAX` 超過を型付きエラー（`ShapeError::AxisOutOfRange`／`ElementCountOverflow`）で拒否する。
  要素数 0 は部分積を計算する前に空出力へ倒す（`[usize::MAX, usize::MAX, 0]` でも panic しない）。
- `cummax`／`cummin` の索引上限（軸長 − 1 が `i32::MAX` 以下）は 2 層で検査する: autodiff 入口が実体化より前に
  `check_i32_indices`、カーネルが `i32::try_from`（無検査 `as i32` を使わない）。`ShapeError::IndexRangeOverflow`
  （`sort`／`topk` と同じ契約）。
- カーネルは入力スライス長を再検査する（`ElementCountMismatch`）。`unsafe`／`get_unchecked`／本番経路の
  `unwrap`／`expect` は使わない。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape（値・索引とも）が入力と異なる場合は `BackendError::ShapeMismatch` で拒否する。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/cumulative-pytorch-reference/`。f32 は
u32 ビットパターンで保存）。有限ケース 49 件で `cummax`／`cummin` の値 bit 一致・索引完全一致、
`logcumsumexp` と全勾配が REQ-2 統一複合判定で一致した。非有限ケース 39 件では forward（値・索引）が
クラス一致し、`cummax`／`cummin` の勾配も一致した。

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 索引の型 | `int64` | `Tensor<i32>`（`Var::sort`／`topk` と同じ慣例） | 軸長 − 1 が `i32::MAX` 超は型付きエラー |
| `dim` | 負の添字可（実測: 受理） | `usize`（`Var::cumsum` と同じ。負は表現不能） | 差分。受け入れ範囲外 |
| 0 次元入力 | 受理（`dim=0`／`-1`。実測） | `ShapeError::AxisOutOfRange` で拒否 | 差分 |
| 軸長 0 | 受理（空出力。実測） | 受理（空出力） | 一致 |
| `dim` 範囲外 | `IndexError` | `ShapeError::AxisOutOfRange` | 一致（拒否） |
| タイ | 後勝ち（実測） | 後勝ち | 一致。既存 `Var::max`（先勝ち）とは規則が異なる点に注意 |
| NaN | 伝播。NaN 到達後は NaN 入力でのみ索引更新（実測） | 同じ | 一致 |
| `+inf` を含む lane の `logcumsumexp` 勾配 | `+inf` 要素位置が 0 になる等（fixture 実測） | `x - out = inf - inf = NaN` で NaN が混じる | 実測差分 4 ケース（`posinf_mid`／`posinf_first`／`posinf_neginf`／`batched_nan_dim1`）。受入条件にしない（非有限入力の勾配一致は対象外）。不一致が `+inf` を含む lane に限ることをテストで固定 |
| `-inf`・NaN を含む lane の `logcumsumexp` 勾配 | — | 同じ（NaN を含む lane は全要素 NaN） | 一致（9 ケース） |
| 整数 dtype・complex・`out=` 引数 | あり | 非対応 | `Var` は f32 のみ |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/cumulative.rs`（単体）: タイ後勝ち・NaN 規則・`±0`・非末尾 `dim`・空軸・軸長 1・
  境界エラー・`i32` 上限・スライス長不一致・`logcumsumexp` の大振幅・`log_add_exp` の非有限分岐・VJP の
  O(n²) オラクル一致と run-to-run bit 決定性。`backend_ops.rs`: 既定 `Unsupported`。
- `crates/autodiff/tests/cumulative_parity.rs`（18 件）: fixture 突合（有限 49・非有限 39）・エラーケース突合・
  `logcumsumexp` 勾配の O(n²) オラクルと中心差分・大振幅勾配の有限性・cummax の重複索引での加算・タイなし
  中心差分・モック `BackendOps` による `Unsupported` フォールバックと他エラーの伝播と誤 shape（値・索引）・
  `dim` 範囲外／rank 0／空軸・巨大 broadcast view の確保前拒否・テープ記録数（各 1 ノード）・`create_graph` が
  型付きエラーであること・run-to-run 決定性。
- `crates/backend-cpu/tests/cumulative_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps::scan_*` の直接
  呼び出し（解析値・strided 入力・型付きエラー・決定性）、CUDA（macOS では Metal も）の `scan_*` が
  `Unsupported` を返し panic しないこと。
- `crates/facade/tests/cumulative_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。
  CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `cumulative_ops` への 1 行委譲で公開する。

- `Var::cummax(&self, dim: usize)`・`Var::cummin(&self, dim: usize)`（`-> Result<(Var<'t>, Tensor<i32>), AutodiffError>`。
  戻り値形は `Var::topk`／`Var::sort` と同形）
- `Var::logcumsumexp(&self, dim: usize)`（`-> Result<Var<'t>, AutodiffError>`。`Var::cumsum` と同形）
- `cumulative_ops` モジュール・`tensor_core::cumulative` は再エクスポートしない。新規公開型なし。
- inherent メソッドの追加のみで非破壊。承認依頼は #2677、公開は承認後の #2678。
- 承認後は `CumulativeOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

承認事項（**すべて未承認**）: 上記 3 メソッドの公開、およびメソッド名・引数形（`dim: usize`・索引 `Tensor<i32>`）。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679）。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定。
- CUDA／Metal の GPU 専用カーネルと実機計測（§10）。
- `create_graph`（高階微分）・activation checkpoint 対象化・f64／f16／bf16 自動微分経路での 3 演算。
- 負の `dim`・0 次元入力の受理・索引の `int64` 化。
- `+inf` を含む lane の `logcumsumexp` 勾配の PyTorch 完全一致（§5）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `CumulativeOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>` に公開されるとコンパイルが失敗する正のプローブ |
| `cumulative_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `cumulative_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_cumulative_ops`（＋自己テスト） | facade src の再エクスポート・`pub mod cumulative_ops`／`pub mod cumulative`・3 名の `fn` 宣言の否定検査 |
| `workspace_declares_cumulative_ops_fn_names_only_in_allowed_locations` | workspace 全体で 3 名の `fn` 宣言が `autodiff/src/cumulative_ops.rs` の各 1 件のみ |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_cumulative_ops_match_cpu_reference`・
`metal_cumulative_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順は
`docs/perf/logs/cumulative-ops-2636/README.md`。

## 11. 出典

- `docs/autodiff-fft-ops-decision.md`（#2631）・`docs/autodiff-nonfinite-ops-decision.md`（#2635）・
  `docs/autodiff-topk-unique-ops-decision.md`（sort／topk の索引慣例）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/cumulative-pytorch-reference/README.md`
