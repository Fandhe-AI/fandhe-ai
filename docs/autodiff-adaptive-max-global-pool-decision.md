# AdaptiveMaxPool2d・AdaptiveMaxPool1d・GlobalPool の設計判断記録

イシュー #2160（親 #2131）。`docs/autodiff-spatial-layers-decision.md`
（#2159）・`docs/conv3d-design.md` 系（#2158）と同型の記録。

## §0 結論

PyTorch の `nn.AdaptiveMaxPool2d`／`nn.AdaptiveMaxPool1d` 相当・ONNX
`GlobalAveragePool`／`GlobalMaxPool`（Keras `GlobalAveragePooling*`／
`GlobalMaxPooling*` 相当）を、`fandhe_ai_autodiff` の `nn` モジュール
（`crates/autodiff/src/nn/pooling.rs`）へ 3 型（`AdaptiveMaxPool2d`・
`AdaptiveMaxPool1d`・`GlobalPool`〈`GlobalPoolMode::{Avg,Max}` を
`#[non_exhaustive]` enum で持つ〉）追加した。**MaxUnpool はスコープ外**
（`docs/pooling-ops-design.md` §11 で既に取り消し線済み。本 issue でも
実装しない）。

forward の VJP は新規 `Op` を追加せず既存 `crate::tape::Op::MaxPool2d`
を再利用する（VJP が `params` に非依存で `input` shape と `index` のみ
から scatter_add が成立するため。`Op::MaxPool2d` の doc comment 参照）。
`GlobalPool(Avg)` は既存 `Var::adaptive_avg_pool2d`／`adaptive_avg_pool1d`
にそのまま委譲するため、既存カーネル・bit 一致契約を変更しない。

`Var` に inherent の `pub fn` は追加していない（`Var` は facade から
再エクスポートされるため。#2143・#2144・#2146・#2158・#2159 の先例）。
`conv3d_ops` よりもさらに一段保守的に、forward 本体
（`crates/autodiff/src/adaptive_max_pool_ops.rs`）を **非 `pub mod`**・
`pub(crate) fn` の自由関数として実装し、`nn::pooling` の層のみから
呼ばれる経路に限定した（`conv3d_ops` は `pub mod` かつ `pub fn` のため、
`Var` への衝突プローブを doctest で機械的に検証できるが、本 issue の
承認未確定な範囲では facade からの到達経路をそもそも作らない方を選ぶ）。

facade 公開（`compat::Sequential::add_adaptive_max_pool2d`／
`add_adaptive_max_pool1d`／`add_global_pool`・`Var::adaptive_max_pool2d`／
`adaptive_max_pool1d` の委譲メソッド）は承認待ちのまま対象外とし、
`crates/facade/src/lib.rs::AdaptiveMaxGlobalPoolHoldDoctestGuard`
（正のプローブ doctest）と `crates/facade/tests/api_surface.rs` の
ソース走査（2 テスト＋ `compat::Sequential` 非公開検査＋自己テスト）
で多層固定している。

## §1 背景

イシュー #2160・親 #2131 はいずれもコメント 0 件（2026-09-25 時点。
着手前に `gh issue view` で確認済み）。親 #2131 はこのツリーでの
facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の 2 段階と定めて
いるため、本実装は内部クレート限定に倒す（同じツリーの先例 #2158・
#2159 と同じ判断）。

## §2 設計判断

### 2.1 層の構成（GlobalPool の意味論）

`docs/compat-api-scope.md` §1.2・`docs/compat-feature-gap.md` に
GlobalPool の行・命名の指定がないため、次のとおり定めた。

| 型 | 入力 | 出力 | 委譲先 |
|---|---|---|---|
| `AdaptiveMaxPool2d { output_size: [usize; 2] }` | `[N,C,H,W]` | `([N,C,Ho,Wo], Tensor<i32>)` | `adaptive_max_pool_ops::adaptive_max_pool2d` |
| `AdaptiveMaxPool1d { output_size: usize }` | `[N,C,L]` | `([N,C,Lo], Tensor<i32>)` | `[N,C,1,L]` へ reshape して 2d を再利用 |
| `GlobalPool { mode: GlobalPoolMode, keepdims: bool }` | rank 3 `[N,C,L]` または rank 4 `[N,C,H,W]` | `keepdims=true`: `[N,C,1]`／`[N,C,1,1]`（ONNX Global*Pool 互換）。`false`: `[N,C]`（Keras 既定互換） | `Avg`: 既存 `adaptive_avg_pool2d([1,1])`／`adaptive_avg_pool1d(1)`。`Max`: `adaptive_max_pool_ops` の `output_size=1` 版 |

`GlobalPoolMode` は `#[non_exhaustive] pub enum { Avg, Max }`。variant
の追加で拡張できる形にした。

### 2.2 forward の配線

- **CPU 参照実装**: `crates/backend-cpu/src/pooling.rs::
  adaptive_max_pool2d`（`max_pool2d` のタイ規則・NaN 規則と
  `adaptive_avg_pool2d` の窓決定規則〈`adaptive_window`〉を組み合わせる）
- **`BackendOps::adaptive_max_pool2d`**（`crates/tensor-core/src/
  backend_ops.rs`）: 既定 `Unsupported`（非破壊拡張。`Var` への
  非公開のため facade 公開面には現れない）
- **ホスト参照実装**: `crates/autodiff/src/eval.rs::
  adaptive_max_pool2d`（CPU 側と意図的に同一アルゴリズムを複製する
  既存慣習に従う）
- **フォールバック**: `crates/autodiff/src/grad.rs::
  adaptive_max_pool2d_with_fallback`（`max_pool2d_with_fallback` と
  同型）
- **forward 本体**: `crates/autodiff/src/adaptive_max_pool_ops.rs`
  （`pub(crate)` 自由関数。`Op::MaxPool2d` を再利用して push）

### 2.3 facade 公開の保留ガード

`crates/facade/src/lib.rs::AdaptiveMaxGlobalPoolHoldDoctestGuard`
（`#[cfg(doctest)]`）で `Var`／`Tensor<f32>`／`Tape` への
`adaptive_max_pool2d`／`adaptive_max_pool1d` メソッド衝突プローブと
`compat::Sequential::add_adaptive_max_pool2d`／`add_adaptive_max_pool1d`／
`add_global_pool` の UFCS 衝突プローブを併用する。`adaptive_max_pool_ops`
自体が非 `pub mod` のため、`conv3d_ops`（#2158）が持つ自由関数の衝突
プローブ・`workspace_declares_*_fn_names_only_in_allowed_locations` 型の
ソース走査インベントリは不要（そもそも facade へ到達するモジュール
パスが存在しない）。facade の `nn` は `pub mod rnn;` にのみ固定済み
のため、型自体の glob 衝突プローブも不要。

### 2.4 索引表現可能範囲検査（`H·W <= i32::MAX`）の全入口統一（codex-review 是正）

`AdaptiveMaxPool2d`／`AdaptiveMaxPool1d`／`GlobalPool(Max)` の索引は
`i32` のため、`H·W`（1d は `l`）が `i32::MAX` を超えると索引が表現
不能になる。この検査は**出力が空かどうか（`out_numel == 0`。例:
空バッチ `N=0`）の判定より前に、値・索引を返す全入口で行う**という
一般原則を確定した（codex-review 指摘・PR #2280。3 回目の指摘のため
本 PR で差分内の全入口を洗い出して統一した）。

`out_shape`（`adaptive_pool2d_out_shape` が返す `[N, C, Hout, Wout]`）
の積は `N=0` で `0` になり、`checked_numel_for` はこの積しか見ない
ため `H·W` 自体の overflow を検出できない。したがって索引範囲検査は
`N`／`out_numel` に一切依存させず、`input` の `H`／`W`（1d は `l`）を
直接見て行う。

- 検査ロジックの単一情報源は `adaptive_max_pool_ops::
  check_max_index_range_shape`（`ShapeError` を直接返す）と、それを
  `AutodiffError` へラップする `check_max_index_range`。`crate::eval`
  （戻り値 `Result<_, ShapeError>`）と `crate::grad`／`crate::
  nn::module`（戻り値 `Result<_, AutodiffError>`）の両方の文脈から
  同一ロジックを共有する
- 検査を行う入口（本 PR で確認・是正済み。すべて早期 return／
  `input` の前処理〈reshape・contiguous 化〉より前）:
  - `adaptive_max_pool_ops::adaptive_max_pool2d`／`adaptive_max_pool1d`
    （tape 経路。既存）
  - `nn::module::Module::forward_host`（`AdaptiveMaxPool2d`／
    `AdaptiveMaxPool1d`／`GlobalPool` rank 4・rank 3 分岐。rank 3
    分岐は本 PR で `x4` 構築より前へ移動）
  - `crate::grad::adaptive_max_pool2d_with_fallback`（バックエンド
    dispatch とホストフォールバックの合流点。本 PR で新規追加。
    多層防御）
  - `crate::eval::adaptive_max_pool2d`（ホスト参照実装本体。本 PR で
    `out_numel == 0` の早期 return より前へ検査を追加）
  - `backend-cpu::pooling::adaptive_max_pool2d`（CPU カーネル本体。
    本 PR で新規追加。`fandhe_ai_autodiff` に依存できないため
    `pooling::check_max_index_range` として同一ロジックを複製）
  - `backend-cpu::ops.rs::CpuBackendOps::adaptive_max_pool2d`
    （`BackendOps` trait 実装。本 PR で新規追加。`pooling::
    adaptive_max_pool2d` への委譲より前に検査する多層防御）

## §3 契約（変更禁止・維持）

- tolerance・baseline・`Cargo.toml` の依存・ガードレール閾値・
  `docs/spec/` は変更していない
- 新規 `unsafe` なし・依存追加なし
- crates.io 公開済みの `fandhe-ai =0.9.0`・`fandhe-ai-autodiff` の
  公開 API は非破壊（新しい pub 型の追加・既定実装付き `BackendOps`
  メソッドの追加・`Module` trait への既定実装付きメソッドの追加のみ）
- 既存 `Op::MaxPool2d`・`max_pool2d`・`adaptive_avg_pool2d` 系の VJP・
  数値契約は変更していない（新規 `Op` variant match arm を追加せず
  既存 arm を共有）

## §4 テスト

- `crates/tensor-core/src/backend_ops.rs::
  adaptive_max_pool2d_default_is_unsupported`
- `crates/backend-cpu/src/pooling.rs` の unit test（割り切れる縮小・
  重なり窓のタイ先勝ち・拡大・タイ・NaN 伝播・batch 0・非 contiguous
  一致・`in % out == 0` のとき `max_pool2d` と bit 一致・空バッチでも
  `H·W > i32::MAX` を `IndexRangeOverflow` で拒否する§2.4 回帰）
- `crates/backend-cpu/tests/pooling_parity.rs::
  backend_ops_adaptive_max_pool2d_rejects_index_range_overflow_on_empty_batch`
  （`CpuBackendOps::adaptive_max_pool2d` 経由の同回帰）
- `crates/autodiff/src/eval.rs::
  adaptive_max_pool2d_host_fallback_tests::
  rejects_index_range_overflow_on_empty_batch_before_early_return`
  （ホスト参照実装本体の同回帰。`pub(crate)` のため crate 内 unit
  test）
- `crates/autodiff/tests/nn_adaptive_max_global_pool.rs`: 割り切れる
  形状での `MaxPool2d` との bit 一致・恒等写像・重なり窓の
  `scatter_add` 勾配・数値微分突合・1d≡2d reshape・
  `GlobalPool(Avg)`≡`AdaptiveAvgPool2d`／`GlobalPool(Max)` values≡
  `AdaptiveMaxPool2d` values・rank 3／4・`keepdims` の shape・無効
  引数拒否・rank 不一致拒否・`forward_host`≡`forward`・
  `nn::Sequential` 統合・`named_parameters` 空・`is_pooling`
- `crates/facade/tests/adaptive_max_global_pool_backend_parity.rs`:
  CPU（`CpuBackendOps`）対 `NaiveOps` の forward／backward bit 完全
  一致（純粋な選択演算・`GlobalPool(Avg)` は既存 f64 契約のため）と、
  `cuda_*`／`metal_*` の `#[ignore]` parity（実機未実測。`docs/perf/
  logs/adaptive-max-global-pool-2160/README.md` 参照）
- `crates/facade/src/lib.rs::AdaptiveMaxGlobalPoolHoldDoctestGuard`
  （正のプローブ doctest）・`crates/facade/tests/api_surface.rs` の
  `adaptive_max_global_pool_hold_doctest_globs_all_pub_modules`・
  `adaptive_max_global_pool_hold_doctest_probe_body_matches_fixed_
  contract`・`compat_sequential_does_not_expose_adaptive_max_global_
  pool_add_methods`（＋自己テスト）
- 既存 pooling 系テスト（`crates/autodiff/tests/pooling.rs`・
  `nn_pooling.rs`・`crates/facade/tests/pooling_backend_parity.rs`・
  `compat_sequential_pooling.rs`・`crates/backend-cpu/tests/
  pooling_parity.rs`）は無変更で green（REQ-2 非後退）

## §5 スコープ外（`out-of-scope-tracking.md` に従う。Issue 起票は
ユーザー承認後）

- **MaxUnpool**（`docs/pooling-ops-design.md` §11）
- `compat::Sequential::add_adaptive_max_pool2d`／
  `add_adaptive_max_pool1d`／`add_global_pool`・`Var::
  adaptive_max_pool2d`／`adaptive_max_pool1d` の facade 公開（承認待ち）
- `adaptive_max_pool2d` の公開関数 API（`Var` メソッドまたは `pub mod`
  自由関数）
- CUDA／Metal の adaptive max 専用カーネル（現状はホストフォール
  バック。`in % out == 0` のときに既存 `max_pool2d` カーネルへ振り替
  える最適化も含む）
- 3d 版・channels_last・ONNX `GlobalMaxPool`／`GlobalAveragePool` の
  import／export（#2200 系）・CUDA／Metal 実機での parity 実測
  （perf log へ申し送り）

## §6 承認事項（未承認として列挙）

1. facade `compat::Sequential` への `add_adaptive_max_pool2d`／
   `add_global_pool`（必要なら `add_adaptive_max_pool1d` も）の追加
   （経路 2）と、`AdaptiveMaxGlobalPoolHoldDoctestGuard`・
   `api_surface.rs` の対応する否定ガードの撤去
2. adaptive max pooling の関数形 API を facade へ公開すること
   （経路 1。`Var::adaptive_max_pool2d`／`adaptive_max_pool1d` の
   委譲メソッド追加）

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
`#[ignore]` テストを未実行のまま出荷する。実行コマンド・記入欄は
`docs/perf/logs/adaptive-max-global-pool-2160/README.md` を参照。
