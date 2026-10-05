# ヒストグラム・二分探索系（`histc`・`bincount`・`searchsorted`・`bucketize`）の CPU 実装記録（イシュー #2638）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-stat-reduce-ops-decision.md`（#2637）・
`docs/autodiff-cumulative-ops-decision.md`（#2636）の「共有カーネルを `tensor-core` に置く」方式を再適用した実装記録であり、
**承認記録ではない**（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678・#2679）。

## 0. 結論

- ヒストグラム・二分探索系 4 種（`histc`・`bincount`〈重みなし／重みあり〉・`searchsorted`・`bucketize`）を CPU 参照実装として
  内部クレートへ追加した。**4 演算はすべて非微分**で、VJP・`Op`・tape ノードを持たない（§5）。
  - 共有カーネル: `fandhe_ai_tensor_core::binning`（`histc_host`／`bincount_host`／`bincount_weighted_host`／
    `searchsorted_host`・`histc_check`／`bincount_check`／`searchsorted_layout`／`bucketize_layout`・`BinningError`）。
    範囲決定・ビン添字の算術・探索手順・アキュムレータ契約の単一情報源。
  - `BackendOps` に既定 `Unsupported` のメソッド 4 件（`binning_histc`／`binning_bincount`／`binning_bincount_weighted`／
    `binning_searchsorted`）を追加（非破壊拡張）。CPU は共有カーネルを呼ぶだけの override。CUDA／Metal は変更なし。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::binning_ops`（`histc`／`bincount`／`bincount_weighted`／
    `searchsorted`／`bucketize`）。
- facade 公開は行わない。`BinningOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。既定 `Unsupported` → 共有ホストカーネルへのフォールバックで動作する。
  実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定（事実のみ）

- 4 演算は REQ-9 Tier の列挙に名前がない（`docs/compat-api-scope.md` 1 節・`docs/compat-feature-gap.md`・`docs/spec/` に
  `histc`／`bincount`／`searchsorted`／`bucketize` の出現なし。`git grep` 実測 0 件）。同名 `fn` の既存宣言もない。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、#2638 の受入条件
  （内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について承認済みとは記録しない
  （承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| 共有カーネル | `tensor-core/src/binning.rs` | 検査関数（確保前）・forward 4 種・`SearchSortedLayout`（フィールド非公開）・`BinningError` |
| バックエンド抽象 | `tensor-core/src/backend_ops.rs` | `binning_*` 4 件（既定 `Unsupported`） |
| CPU | `backend-cpu/src/ops.rs` | 共有カーネルを呼ぶだけの override（`contiguous()` 済みスライスを渡し、検査を再実行） |
| autodiff | `autodiff/src/binning_ops.rs`・`error.rs` | 自由関数 5 件・`From<BinningError>` |

- 4 演算は出力が整数・勾配なしのため `tape.rs`／`grad.rs`／`var.rs` は編集していない（`topk_unique_ops::unique_with_options`・
  `nonfinite_ops::isnan` と同型の detached 出力）。tape へノードを積まない（`push_eager`／`push_lazy` を呼ばない）。
- `bincount` の入力は整数で `Var`（f32 のみ）では表せないため `Tensor<i32>` を取り、`BackendOps` へ到達するために
  `tape: &Tape` を明示引数にする。`bincount_weighted` も対称に `tape` を取り、`weights` の tape との不一致は
  `AutodiffError::TapeMismatch`。`searchsorted`／`bucketize` は 2 オペランドのクロステープ検査を shape 検査より前に行う。
- `bucketize` は `boundaries` が 1 次元であることを検査したうえで、`searchsorted(boundaries, input, right)` と同じ
  カーネル（`binning_searchsorted`）を呼ぶ 1 次元特例。
- 命名規律: 素の `fn histc`／`bincount`／`bincount_weighted`／`searchsorted`／`bucketize` は
  `autodiff/src/binning_ops.rs` の各 1 件のみ（workspace インベントリが固定）。trait メソッドは `binning_*`、共有カーネルは
  `*_host`。`Var`／`Tape`／`Tensor` に inherent メソッドは足していない（足すと facade 公開面が広がる）。
- 索引・カウントは既存慣例（`Var::sort`／`topk`／`UniqueOutput::counts`）に合わせ `Tensor<i32>`。

## 3. 数値契約と意味論

PyTorch 2.14.0 の実行値 fixture で実測して確定した（実装前の仮説と異なった点は §5 に明記）。

| 演算 | 契約 |
|---|---|
| `histc` | 入力を平坦化。範囲決定は ATen と同じく `f64` で、`min == max` かつ入力が非空なら入力の最小・最大（NaN を含めば NaN）、なお等しければ `[min - 1, max + 1]`。`min > max`・`bins == 0`・範囲が非有限は型付きエラー。範囲確定後は **`f32` のまま同じ演算順**で、左右端 `lo = start + step * 0`・`hi = end - step * 0`（`step = (end - start) / bins`。`torch.linspace` の端点算術。`max - min` が `f32` で溢れると `step = inf` で両端が NaN になり、PyTorch と同じく全要素が無視されて全ビン 0）、要素ごとに `!(x >= lo && x <= hi)` なら無視（NaN・範囲外）、それ以外は `trunc((x - lo) * bins / (hi - lo))`、結果が `bins` なら最終ビン。カウントは `u64` で数え最後に 1 回だけ `f32` へ変換する。`f64` で添字を計算すると境界要素のビンがずれる。`mul_add` は使わず matmul 系 FMA 契約には触れない |
| `bincount`（重みなし） | 1 次元・非負のみ。出力長 `max(max(input) + 1, minlength)`、空入力は長さ `minlength` の零。カウントは `i32`（入力長が `i32::MAX` を超える場合は `ShapeError::IndexRangeOverflow`） |
| `bincount`（重みあり） | `weights` は入力と同長（空入力は PyTorch と同じく重みを見ず長さ `minlength` の零）。ビンごとに **`f64` アキュムレータ**へ入力順に加算し最後に 1 回 `f32` へ downcast（長軸縮約の `f64` 契約。`.claude/rules/coding-rust.md`）。非有限重みは伝播 |
| `searchsorted` | `sorted_sequence` は rank ≥ 1。rank 1 なら `values` は任意 shape（0 次元可）、rank N ≥ 2 なら `values` は同 rank で先頭 N−1 軸が一致。`right = false` は下限（`!(mid >= v)` で右へ）、`true` は上限（`!(mid > v)` で右へ）、中点 `start + (end - start) / 2`。PyTorch の探索手順をそのまま再現するため重複・NaN・未ソート列でも結果が一致し、結果は常に `[0, 列長]` 内。未ソート列は検査しない（PyTorch と同じ）。列長が `i32::MAX` 超は `ShapeError::IndexRangeOverflow` |
| `bucketize` | `boundaries` は 1 次元必須。`searchsorted(boundaries, input, right)` と同値（`right` の向きも同じ。実測確認済み） |

同一入力の run-to-run は bit 一致。算術を伴うのは `histc` の添字計算と重み付き `bincount` の和のみ。

## 4. 境界検査

- 形状・要素数・バイト数は `checked_mul`、`isize::MAX` 超過は `ShapeError::ElementCountOverflow`。確保・実体化より前に
  autodiff 入口が検査する（巨大 broadcast view〈要素数 1 の `Var` を `[1 << 61]` へ `broadcast_to`〉は実体化前に拒否）。
  `SearchSortedLayout` はフィールド非公開で検査関数だけが生成する。カーネルは入力スライス長を再検査する。
- 出力長が入力「値」や引数に依存する箇所（`bincount` の最大値・`minlength`、`histc` の `bins`）は、全要素の検証 →
  `checked_add`／`usize::try_from` → バイト数検査 → `Vec::try_reserve_exact` の順で、確保失敗を abort ではなく型付き
  エラー（`ElementCountOverflow`）にする。恣意的な上限値は新設していない。
- 添字変換は `usize::try_from`／`i32::try_from`。`histc` の添字は有限かつ範囲内であることを確認してから変換し、
  `bins - 1` 以下へ収める（範囲外は無視。fail-closed）。
- `unsafe`／`get_unchecked`／本番経路の `unwrap`／`expect` は使わない。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape が期待と異なる場合は `BackendError::ShapeMismatch` で拒否する。

## 5. PyTorch 2.14.0 との差分・実測で確定した点

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/binning-pytorch-reference/`。f32 は u32
ビットパターンで保存）。`histc` 37 件のカウントは bit 一致、`bincount` 16 件・`searchsorted` 34 件・`bucketize` 14 件の
索引・カウントは完全一致（重み付き `bincount` は REQ-2 判定。ただし下表の意図的な差 1 件を除く）。例外 16 件
（`histc` 6・`bincount` 4・`searchsorted` 4・`bucketize` 2）は本実装も型付きエラー。

実装前の仮説から**実測で修正した点**:

| 項目 | 計画時の仮説 | PyTorch 2.14.0 の実測 | 本実装 |
|---|---|---|---|
| `histc` のビン決定 | 隣接エッジの局所探索（`torch.linspace` のエッジ列と突合）を伴う見込み | **局所探索なし**。`trunc((x - lo) * bins / (hi - lo))` のみ（局所探索を入れると `[0, 1]`・`bins=3` のエッジ入力でビンがずれた） | 実測に合わせた（エッジ列は作らず左右端だけ `torch.linspace` 算術で再現） |
| `bincount(weights)` の微分性 | 非微分（`requires_grad=False`） | `requires_grad=True`・`grad_fn=NotImplemented` で、backward は `derivative for aten::bincount is not implemented` で失敗 | 実質非微分で、detached を返す。重みへの勾配は対象外 |
| 空入力 ＋ 長さ不一致の重み | 例外 | **例外にならない**（重みを見ずに長さ `minlength` の零） | 同じ（fixture `weights_ignored_when_input_empty`） |

差分表:

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 非微分性 | `histc`・`searchsorted`・`bucketize` は `requires_grad=False`。`bincount(weights)` は上記 | tape ノードなし・detached | 受入条件の「微分可能なものは VJP」に該当する演算なし |
| 索引・カウントの型 | `int64` | `Tensor<i32>`（`Var::sort`／`topk` と同じ慣例） | 列長・入力長が `i32::MAX` 超は型付きエラー |
| `histc` の出力型 | `float32` | `Tensor<f32>` | 一致。カウントは `u64` で数え `f32` へ 1 回変換（PyTorch は `f32` の `+1` 累積のため 2^24 超で飽和。本実装は丸め。実用上の差は 2^24 個超の同一ビンのみ） |
| 重み付き `bincount` の和 | `f32` 逐次加算（`[1e8, 1.0, -1e8]` → 0） | `f64` アキュムレータ後に 1 回 downcast（→ 1.0） | **意図した差**（長軸縮約の `f64` 契約。coding-rust.md）。fixture の `w_cancel` 1 件はこの差を固定する（PyTorch 値 0・本実装値 1 を各々 assert）。他の重み付き 15 件は REQ-2 判定で一致 |
| `histc` の NaN | 既定範囲では `range of [nan, nan] is not finite` 例外。明示範囲では NaN 要素を無視 | 同じ | 一致 |
| `histc` の範囲オーバーフロー（`max - min` が `f32` で `inf`） | 例外にならず全ビン 0 | 同じ | 一致 |
| `histc` の `bins` | `int64`（負値は例外） | `usize`（負値は表現不能） | 差分。`bins = 0` は両者とも例外 |
| 未ソート列・NaN を含む列 | 規定なし。実測は上記探索手順の結果 | 同じ手順のため一致 | 一致を fixture で確認（結果は常に `[0, 列長]` 内） |
| `searchsorted` の `side`・`sorter`・`out_int32`、`bucketize` の `out_int32`、`out=` | あり | 非対応 | 対象外（§8） |
| 整数 dtype 入力 | あり | `f32` のみ（`bincount` の入力のみ `Tensor<i32>`） | 対象外 |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/binning.rs`（単体）: 格子・最終ビン・既定範囲・空入力・不正引数・NaN／範囲外の無視・総和と決定性・
  `bincount` の `minlength`・負値／rank／巨大確保の拒否・`f64` アキュムレータ（相殺列）・`searchsorted` の素朴線形走査
  オラクル一致・レイアウト規則・巨大 shape の確保前拒否・NaN／未ソート列・バッチ・スライス長再検査・空列。
  `backend_ops.rs`: 既定 `Unsupported`。
- `crates/autodiff/tests/binning_parity.rs`: fixture 突合（`histc` bit 一致・索引／カウント完全一致・重み付きは REQ-2 判定）、
  例外ケース、非微分性（fixture）と tape 非記録、モック `BackendOps` による `Unsupported` フォールバックと他エラーの伝播と
  誤 shape、`TapeMismatch`（shape 検査より前）、巨大 broadcast view・巨大 `bins`／`minlength` の確保前拒否、0 次元／空入力、
  run-to-run 決定性。
- `crates/backend-cpu/tests/binning_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps::binning_*` の直接呼び出し
  （解析値・strided 入力・型付きエラー・空入力・決定性）、CUDA（macOS では Metal も）の `binning_*` が `Unsupported` を返し
  panic しないこと。
- `crates/facade/tests/binning_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。CUDA／Metal 実機は
  `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。**inherent メソッドの 1 行委譲**で公開する（モジュール再エクスポート・`Sequential::add_*` は使わない）。

- `Var::histc(&self, bins: usize, min: f32, max: f32)`（`-> Result<Tensor<f32>, AutodiffError>`）
- `Var::searchsorted(&self /* sorted_sequence */, values: &Var<'t>, right: bool)`・
  `Var::bucketize(&self /* input */, boundaries: &Var<'t>, right: bool)`（`-> Result<Tensor<i32>, AutodiffError>`）
- `Tape::bincount(&self, input: &Tensor<i32>, minlength: usize)`・
  `Tape::bincount_weighted(&self, input: &Tensor<i32>, weights: &Var<'_>, minlength: usize)`
  （整数入力に `Var` の受け手がないため `Tape` に置く。`&self.0` を渡すだけの委譲）
- `binning_ops`／`binning` モジュールと `BinningError` は再エクスポートしない。新規公開型なし・メソッド追加のみで非破壊。
- 承認後は `BinningOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

承認事項（**すべて未承認**）: 上記 5 メソッドの公開、およびメソッド名・引数形（`right: bool`・索引／カウントの
`Tensor<i32>`・`bincount` を `Tape` 側に置くこと）。承認依頼は #2677、公開は承認後の #2678・#2679。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679。保留ガードの正ガード反転を含む）。
- CUDA／Metal の GPU 専用カーネルと GB10／M4 Max 実機計測（§10）。
- `searchsorted` の `side`／`sorter`／`out_int32`、`torch.histogram`／`histogramdd`、整数 dtype 入力、索引・カウントの `int64` 化。
- `bincount_weighted` の重みに対する勾配（PyTorch も backward が未実装であることを実測確認済み）、`create_graph`・
  activation checkpoint・f64／f16／bf16 経路。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定、
  `MIN_KNOWN_PROBE_BLOCKS` の更新。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `BinningOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッド・型・モジュールが `Var`／`Tape`／`Tensor<f32>`／`Tensor<i32>`／facade ルートに公開されるとコンパイルが失敗する正のプローブ |
| `binning_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `binning_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_binning_ops`（＋自己テスト） | facade src の再エクスポート・`BinningError` の独自宣言・`pub mod binning_ops`／`pub mod binning`・5 名の `fn` 宣言の否定検査 |
| `workspace_declares_binning_ops_fn_names_only_in_allowed_locations` | workspace 全体で 5 名の `fn` 宣言が `autodiff/src/binning_ops.rs` の各 1 件だけであること |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_binning_ops_match_cpu_reference`・
`metal_binning_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順は
`docs/perf/logs/binning-ops-2638/README.md`。

## 11. 出典

- `docs/autodiff-stat-reduce-ops-decision.md`（#2637）・`docs/autodiff-cumulative-ops-decision.md`（#2636）・
  `docs/autodiff-topk-unique-ops-decision.md`（索引慣例）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/binning-pytorch-reference/README.md`
