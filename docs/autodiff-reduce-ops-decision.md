# prod・logsumexp・any・all・norm_p（p-ノルム）の設計判断記録

イシュー #2147（親 #2131）。`docs/autodiff-matrix-ops-decision.md` と
同型の記録。

## §0 結論

PyTorch 互換の縮約演算 5 種（`prod`／`logsumexp`／`any`／`all`／
`norm_p`）を、**`fandhe_ai_autodiff` のうち facade が再エクスポート
しない自由関数モジュール `reduce_ops`**（`crates/autodiff/src/
reduce_ops.rs`）として実装した（`matrix_ops`〈#2144〉・
`rearrange_ops`〈#2143〉・`bool_ops`〈#2141〉と同じ判断枠組み）。
`Var` に inherent の `pub fn` は追加していない。

Issue #2147 は「facade への 5 メソッドの `pub use` 再エクスポート
（経路 2）」を承認事項として明示しているが、`facade` は `Var` を
そのまま再エクスポートしている（`crates/facade/src/lib.rs`）ため、
`Var` に inherent メソッドを 1 つ足すだけで facade の公開面が広がる。
このツリー（親 #2131）の先例に倣い、承認が取れるまで **autodiff に
新設したモジュール `reduce_ops` の自由関数 5 個**で「5 メソッド」を
満たす（承認後に追加する作業は `Var::prod` 等の薄い委譲メソッドと
facade ガードの撤去のみ）。

`prod`／`any`／`all` は既存 `Op` の合成のみ（新規 `Op` なし）。
`logsumexp`／`norm_p` は数値安定化のため専用 `Op`（`Op::LogSumExp`／
`Op::PNorm`）を追加した——`sum`／`exp`／`log` や `max`／`pow`／`sum`
の素朴な合成では、全要素が `-inf`（または `+inf`）の lane・overflow
を起こす `p` で `NaN` が出るため（`Var::std` の先例と同じ理由）。

## §1 背景

イシュー #2147・親 #2131 のどちらにも所有者の承認コメントはない
（着手前に `gh issue view --json comments` で確認済み）。親 #2131 は
このツリーでの facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の
2 段階と定めているため、本実装は内部クレート限定に倒す。

## §2 各演算の設計

### §2.1 共通方針

シグネチャは既存の `Var::sum`／`max` にそろえ、`dim: Option<usize>`
を取る（`None` は全軸縮約でスカラー `[]` を返す）。keepdim 版・複数軸
版（`*_dims`）はスコープ外。

### §2.2 `prod`: 既存 Op の合成（新規 Op なし）

`cumprod(dim) → narrow(dim, n-1, 1) → contiguous() → squeeze(dim)`。
`dim=None` のときは先に `reshape([numel])` してから `cumprod(0)` を
取る。

- `Op::Cumprod`（#1731）の forward は `f64` アキュムレータで計算し
  1 回だけ `f32` へ落とす。VJP は除算を使わない厳密形（排他 prefix
  積）のため零要素の個数に依らず正確。
- **`contiguous()` を `squeeze` の直前に挟む理由**: `narrow` は
  `dim` が末尾軸でない限り非 contiguous な stride view を返しうる。
  `squeeze`（`Var::reshape` への委譲）は contiguous な入力を要求する
  （`ShapeError::NonContiguousReshape`）。`matrix_ops::diag_2d_to_1d`
  の `narrow → gather → squeeze` は `gather` が暗黙に実体化するため
  同じ問題を踏まないが、`prod` は `gather` を経由しないため明示的に
  `contiguous()` を挟む必要がある（実装時に `dim=Some(axis)`（`axis`
  が末尾軸でない 2-D 入力）の統合テストで `NonContiguousReshape` の
  回帰を検出し是正した）。
- 空縮約（`n == 0`）は PyTorch と同じく単位元 **1.0** を返す。
  `narrow(n-1)` の underflow を避けるため合成より前に分岐する。

### §2.3 `any`／`all`: 既存 Op の合成（新規 Op なし）

`any`: `x.ne(&zero) → max(dim)`。`all`: `x.ne(&zero) → min(dim)`。
`zero` は同じ tape 上のスカラー葉（`Tensor::scalar(0.0)` →
`Tape::push_leaf`）。

- 非ゼロを真とする。`NaN != 0` は真なので `NaN` は真として扱い、
  PyTorch と一致する。`-0.0` は偽。
- 出力値は厳密に `0.0` か `1.0` だけで縮約順序に依存しないため、
  3 バックエンドで **bit 完全一致**する。
- 勾配は `ScalarBinaryOp::Ne` の VJP がゼロを返すため、合成しただけで
  自動的に勾配ゼロの tape ノードになる。
- 空縮約: `max`／`min` は単位元を持たずエラーになるため、合成より前に
  分岐し `any(∅) = 0.0`・`all(∅) = 1.0` を定数葉で返す（PyTorch と
  同じ）。
- bool 出力版は #2141（`bool_ops`）の対象で本モジュールの対象外。

### §2.4 `logsumexp`: 専用 Op

新規: `Op::LogSumExp { input, dim }`・defaulted
`BackendOps::logsumexp`（既定 `Unsupported`）・
`eval::logsumexp_along`（ホスト参照実装）・
`grad.rs::logsumexp_vjp`。

forward（lane ごと）:
1. `m = max(x)` を `f64` で求める（`NaN` 伝播 max。`f64::max` は
   非 `NaN` 側を返してしまうため専用の `nan_propagating_max_f64` を
   使う）。`m` が非有限（±inf）なら安定化シフトを `0` に置き換える。
2. `acc = Σ exp((x_i as f64) - shift)` を `f64` で蓄積する。
3. `y = ln(acc) + shift` を計算し、`f32` へは 1 回だけ落とす。

VJP: `dx_i = g · exp(x_i - y)` を `f64` で計算して `f32` へ落とす。
`y == -inf`（全要素が `-inf`）の lane は勾配 `0` とする——PyTorch は
この lane で `NaN` を返すが、`NaN` 勾配で学習を汚染しない安全側の
判断（§4 に記録）。

空縮約は `AutodiffError::InvalidArgument`（`Var::max`・`Var::norm` と
同じく `-inf` を黙って返さない）。

backend-cpu の `reduction::logsumexp` は eval と**同じ lane ごとの
逐次 `f64` アルゴリズム**で実装し（lane 間だけ rayon で並列化）、CPU
カーネルとホストフォールバックの結果が bit 一致する構造にした。

### §2.5 `norm_p`: 専用 Op

`VectorNormOrd` は拡張しない（crates.io 公開クレートで `Eq` を
derive しているため `Lp(f32)` を足すと壊れる。既存の
`eval::vector_norm_along`／`reduction::NormKind` には `_ => 0.0` の
安全側フォールバックがあり、未知 variant を静かに 0 とみなす経路が
既にあるため、新規 variant の追加ではなく別 `Op` にする方が
fail-closed）。

新規: `Op::PNorm { input, p: f32, dim }`（`Op` は
`#[derive(Debug, Clone)]` のみのため `f32` を持たせてよい）・
defaulted `BackendOps::vector_norm_p`（既定 `Unsupported`）・
`eval::vector_norm_p_along`・`grad.rs::pnorm_vjp`。

**`p` の検証（fail-closed）**: 有限かつ `p > 0` のみ受け付ける。
`NaN`・±inf・`0`・負の値は `InvalidArgument` で拒否する。PyTorch は
inf・`0`・負の `p` も受け付けるが、inf ノルム勾配の分配方式（本リポの
`max_vjp` は先勝ちのみで均等分配は別イシュー #2154 の対象）が未定の
ため見送る（§4・§5 に記録）。

`p == 1.0`／`p == 2.0` は既存の `Var::norm_l1`／`norm_l2`（`Var::norm`
`pub(crate)` 経由）へ委譲する。これにより `norm_p(x, 2.0, d)` と
`norm_l2(d)` が **bit 同一**になる（`crates/autodiff/tests/
reduction_parity.rs` で固定）。

forward は overflow を避けるスケール形で計算する:
- `mx = max|x_i|` を `f64` で求める（`NaN` 伝播）。
- `mx == 0` なら `0`、`mx` が `inf` なら `inf`、`NaN` は伝播。
- それ以外は `norm = mx · (Σ (|x_i|/mx)^p)^(1/p)` を `f64` で計算し、
  `f32` へは 1 回だけ落とす（`p` が大きい場合〈例 `p=50,
  |x|≈1e38`〉でも `f64` で overflow しない）。

VJP: `dx_i = g · sign(x_i) · (|x_i|/norm)^(p−1)` を比の形（`norm^
(p−1)` を直接求めず overflow を避ける）で `f64` 計算する。
- `norm == 0` の lane は全要素 `0`。
- `x_i == 0` の要素は `0`（`p < 1` で `0^(負)` が `inf` になるのを
  防ぐ劣勾配の選択）。

空縮約は既存の `Var::norm` と同じく `InvalidArgument`。

## §3 数値契約（まとめ）

| 演算 | forward | backward |
|---|---|---|
| `prod` | `Op::Cumprod` の `f64` アキュムレータに委譲（REQ-2 統一複合判定。縮約順序がバックエンド依存） | 除算を使わない厳密形の合成 VJP（REQ-2 統一複合判定） |
| `logsumexp` | `f64` 二段計算・専用 `Op`（REQ-2 統一複合判定） | `f64` 二段計算（REQ-2 統一複合判定） |
| `any`／`all` | 厳密 `0.0`／`1.0`（**bit 完全一致**） | `ne` の VJP がゼロ（**bit 完全一致**） |
| `norm_p`（`p≠1,2`） | `f64` スケール形・専用 `Op`（REQ-2 統一複合判定） | `f64` 比の形（REQ-2 統一複合判定） |
| `norm_p`（`p=1,2`） | 既存 `norm_l1`／`norm_l2` へ委譲（**bit 同一**） | 同上へ委譲 |

## §4 PyTorch との差分

- `logsumexp` の `y == -inf`（縮約対象が全て `-inf`）lane の勾配は
  `0`（PyTorch は `NaN`）。`NaN` 勾配で学習を汚染しないための安全側
  の判断。
- `norm_p` の `p` は有限かつ正のみ許容（`0`・負・±inf・`NaN` は
  `AutodiffError::InvalidArgument`）。PyTorch は inf・`0`・負の `p`
  も受け付けるが、inf ノルムの勾配分配方式が本リポでは未定（#2154）
  のため見送る。
- keepdim 版・複数軸版（`*_dims`）は非対応。

## §5 スコープ外（out-of-scope-tracking の候補）

- facade 公開（`Var::prod` 等の委譲メソッド追加と保留ガードの撤去）:
  経路 2 の承認待ち。窓口は #2147・#2131。
- bool 出力版の `any`／`all`: #2141 に従属。
- GPU 専用カーネル（`logsumexp`・`norm_p`）: 別イシュー。
- `p ∈ {0, ±inf, 負}` のノルム: inf ノルム勾配の分配方式が #2154 の
  決定に依存するため。
- keepdim 版・複数軸版（`*_dims`）: 対象外。
- CUDA・Metal の実機 parity 実測: 申し送り（`docs/perf/logs/
  reduce-ops-2147/README.md`）。

## §6 承認事項（未承認として列挙）

1. facade 公開（経路 2。上記スコープ外 1 と同じ）
2. bool 出力版の `any`／`all`（#2141 側の承認事項）
3. GPU 専用カーネル
4. `p ∈ {0, ±inf, 負}` のノルム（#2154 の決定待ち）

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/reduce-ops-2147/README.md` へ測定コマンド案・
期待結果を申し送る。

## §8 実装記録（イシュー #2147）

- `crates/tensor-core/src/backend_ops.rs`: `BackendOps::logsumexp`・
  `vector_norm_p`（defaulted・既定 `Unsupported`）＋回帰テスト
  `logsumexp_vector_norm_p_defaults_are_unsupported`
- `crates/backend-cpu/src/reduction.rs`: `ReduceError::InvalidOrder
  (f32)` variant・`logsumexp`／`vector_norm_p`（`nan_propagating_
  max_f64`・`logsumexp_slice`／`axis_reduce_logsumexp`・
  `vector_norm_p_slice`／`axis_reduce_vector_norm_p`）
- `crates/backend-cpu/src/ops.rs`: `CpuBackendOps::logsumexp`／
  `vector_norm_p` の結線・`reduce_error_to_backend_error` の
  `EmptyReduction` 写像へ `"logsumexp"`／`"norm_p"` を追加・
  `InvalidOrder` の写像を追加
- `crates/autodiff/src/tape.rs`: `Op::LogSumExp`・`Op::PNorm` の
  variant と 3 箇所の網羅 match（`is_checkpoint_eligible`・
  `for_each_input`・`supports_create_graph`。いずれも `Op::Var`／
  `Op::VectorNorm` と同じ扱い）
- `crates/autodiff/src/eval.rs`: `nan_propagating_max_f64`（`pub
  (crate)`）・`logsumexp_along`・`vector_norm_p_along`
- `crates/autodiff/src/grad.rs`: `vjp` への 2 arm 追加・
  `logsumexp_vjp`・`pnorm_vjp`
- `crates/autodiff/src/reduce_ops.rs`（新規）: `prod`・`logsumexp`・
  `any`・`all`・`norm_p`（5 自由関数）・モジュール doc・単体テスト
  34 件（forward・エッジケース・エラー系・勾配の有限差分検算・
  `NaN`／`inf`／overflow 系）
- `crates/autodiff/src/lib.rs`: `pub mod reduce_ops;`・クレート doc
  へイシュー #2147 の要約を追記
- `crates/autodiff/tests/reduction_parity.rs`（新規）: NaiveOps 上の
  閉形式・`f64` 参照値との突合・bit 同一契約（`norm_p(1)` ≡
  `norm_l1`・`norm_p(2)` ≡ `norm_l2`）・中心差分による勾配検査・
  エラー系（13 件）
- `crates/facade/src/lib.rs`: `VarReduceOpsHoldDoctestGuard`（正の
  プローブ doctest。`VarMatrixOpsHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`:
  `reduce_ops_hold_doctest_globs_all_pub_modules`・
  `reduce_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_reduce_ops`・
  `workspace_declares_reduce_ops_fn_names_only_in_allowed_locations`
  （4 テスト。**期待集合は `matrix_ops` と異なり単一ファイルに閉じ
  ない**——着手前の再 grep で `logsumexp` という関数名が
  `tensor-core::backend_ops`（trait デフォルトメソッド）・
  `backend-cpu::{reduction, ops}`（CPU 実装）にも正規に存在すること
  が判明したため、期待集合を「`autodiff/src/reduce_ops.rs` に
  5 件（各演算 1 件） + `logsumexp` が上記 3 箇所に追加で 1 件ずつ」
  という allowlist へ調整した。`prod`・`any`・`all`・`norm_p`〈完全
  一致の識別子。`vector_norm_p` とは別名〉は `reduce_ops.rs` にのみ
  存在する）
- `crates/facade/tests/reduce_ops_backend_parity.rs`（新規）: CPU と
  NaiveOps の `any`／`all` forward・勾配ゼロ性の bit 完全一致、
  `prod`／`logsumexp`／`norm_p` forward／backward の REQ-2 統一複合
  判定（属性なし 4 件）＋CUDA／Metal の `#[ignore]`（未実測。8 件。
  `docs/perf/logs/reduce-ops-2147/README.md` 参照）
- `docs/compat-api-scope.md`: §1.2 へ追補（facade 経路 2 未適用のため
  保留と明記）
- `docs/README.md`: 本 doc・perf log README の索引行を追加

承認取得後の追随（本イシューでは未実施）: `Var::prod` 等の薄い委譲
メソッド追加、facade 保留ガード（`VarReduceOpsHoldDoctestGuard`・
対応する否定ガード 4 件）の撤去。
