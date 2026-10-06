# 索引付き更新（`scatter_reduce`・`index_add`・`index_copy`・`masked_scatter`）の CPU 実装記録（イシュー #2641）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-cumulative-ops-decision.md`（#2636）の「共有カーネルを
`tensor-core` に置く」方式（`scatter_reduce`）と、`docs/autodiff-shape-view-ops-decision.md`（#2639）の
「既存 `Op` の合成」方式（残り 3 演算）を使い分けた実装記録であり、**承認記録ではない**（facade 公開形の承認は
#2677 で依頼中。公開自体は承認後の #2678・#2679）。

## 0. 結論

- 索引付き更新 4 演算を内部クレートへ追加した。
  - `scatter_reduce`: 共有ホストカーネル `fandhe_ai_tensor_core::indexed_update`（`scatter_reduce_layout`／
    `scatter_reduce_host`／`scatter_reduce_vjp_host`）＋専用 `Op::IndexedScatterReduce`＋`BackendOps::
    indexed_scatter_reduce`（既定 `Unsupported`。CPU のみ override）。
  - `index_add`／`index_copy`／`masked_scatter`: 既存の `Var::scatter_add`／`Var::scatter`（`Op::Scatter`）と
    view 系の合成のみ（新規 `Op`・`BackendOps` メソッドなし）。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::indexed_update_ops`（4 件）。
- facade 公開は行わない。`IndexedUpdateOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは追加していない。実機テストは `#[ignore]` のまま未実測（§10）。

## 1. 着手時の判定（事実のみ）

- 4 演算は REQ-9 Tier 2 の列挙に名前がなく、workspace に同名の `fn` は存在しなかった。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、
  #2641 の受入条件（内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について
  承認済みとは記録しない（承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 演算 | 方式 | 新規 `Op` | `BackendOps` | `tensor-core` |
|---|---|---|---|---|
| `index_add` | 既存 `Var::scatter_add` の合成 | なし | 変更なし | 変更なし |
| `index_copy` | 既存 `Var::scatter` の合成 | なし | 変更なし | 変更なし |
| `masked_scatter` | view 系＋`narrow`＋`Var::scatter` の合成 | なし | 変更なし | 変更なし |
| `scatter_reduce` | 共有ホストカーネル＋専用 `Op`＋VJP | `Op::IndexedScatterReduce` | `indexed_scatter_reduce`（既定 `Unsupported`） | 新規モジュール `indexed_update` |

- **既存の `ScatterReduce`（`Overwrite`／`Add`）は拡張しない**。`#[non_exhaustive]` のため variant 追加は契約上
  許されるが、既存の全消費者（`autodiff::eval::scatter`・`backend-cpu::gather_scatter::scatter`・CUDA／Metal の
  `run_scatter_f32`）は未知 variant を `debug_assert!` のうえ release では黙って `Overwrite` として処理する。
  `Prod`／`Mean`／`Amax`／`Amin` を足すと、CUDA／Metal 利用者がエラーなしで誤った数値を受け取りうる（実機検証も
  本環境ではできない）。そのため別 enum `ScatterReduceMode` と別 trait メソッド `indexed_scatter_reduce` を新設した。
- 命名規律: 素の `fn scatter_reduce`／`fn index_add`／`fn index_copy`／`fn masked_scatter` は
  `autodiff/src/indexed_update_ops.rs` の各 1 件のみ（workspace インベントリが固定）。trait メソッドは
  `indexed_scatter_reduce`、共有カーネルは `scatter_reduce_layout`／`scatter_reduce_host`／
  `scatter_reduce_vjp_host`、`Op` の variant は `IndexedScatterReduce`（型 `ScatterReduce` との混同を避ける）。
- `Var`／`Tape`／`Tensor` に inherent メソッドは足していない（足すと facade 公開面が広がる）。
- **`index_add` の `alpha` は持たない**。`Var` に f32 スカラー乗算の入口がなく、実現には隠し定数ノード
  （`Tape::var_no_grad`）が要る。最初の演算より前に作られた定数は葉プレフィックスに入り、`Tape::leaf_count()`／
  `leaf(i)`／`reset()` 後の保持対象として利用者から見える副作用になる。呼び出し側が `source` を事前にスケール
  すれば等価。公開形に `alpha` を含めるかは承認事項（§7、未承認）。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応。f64 自動微分経路は対象外。
- VJP の上流 shape 検査には `check_fft_upstream_shape` を流用している（名前が FFT 固有である点は既知のレビュー
  指摘候補。本 PR では改名しない）。

## 3. 数値契約

「触れられた位置」= 1 個以上の `src` が書かれた出力位置。**触れられなかった位置は入力のビット列をそのまま写す**
（アキュムレータを通さない）。`include_self == false` のとき、触れられた位置は入力値を寄与から外す。

| mode | 触れられた位置の計算 | 数値契約 |
|---|---|---|
| `Sum` | `(include_self ? input : +0.0) + Σ src` | 位置ごとの `f64` アキュムレータへ走査順（`index`／`src` の行優先）に加算し 1 回だけ `f32` へ downcast。`include_self == true` は `ScatterReduce::Add` と bit 一致（テストで固定） |
| `Mean` | 上の和 ÷ 寄与数（`src` の個数 ＋ `include_self` なら 1） | 和・除算とも `f64`、1 回 downcast |
| `Prod` | `(include_self ? input : 1) × Π src` | `f64` で走査順に乗算、1 回 downcast。`mul_add` は使わない（matmul 系 FMA 契約には触れない） |
| `Amax`／`Amin` | 寄与の最大／最小 | 比較と選択のみ（値は寄与のいずれかと bit 一致）。種は `include_self` なら入力、そうでなければ最初の `src`。**厳密に大きい（小さい）ときのみ更新（タイは先勝ち）**・NaN は伝播。この規則は PyTorch 2.14.0 の実測（`±0`・NaN ケース）で確定した |

`index_add` は `Add` の `f64` 決定的集約。`index_copy` は行優先で最後の書き手が勝つ（PyTorch は重複添字で未定義。
`index_put` と同じ「より強い契約」。VJP も最後の書き手だけに流れる）。`masked_scatter` はコピーのみ（forward は
bit 一致）。

VJP（`scatter_reduce`。`f64` で計算し各要素 1 回 downcast。入力値・`src`・`index` を実体化して再計算する）:

- `Sum`: `d_src[p] = g[pos(p)]`、`d_input = g`（`include_self == false` では触れられた位置を 0）。
- `Mean`: `d_src[p] = g[pos(p)] / count`、`d_input[q] = g[q] / count`（`include_self == false` では触れられた位置を 0）。
- `Prod`: 割り戻し（`result / 値`・`P / v`）は使わず、位置ごとの寄与列（`src` は走査順・`include_self` の自己寄与は末尾）の
  前置積×後置積で各寄与を除いた積を直接求める（`f64`）。0・inf を含む lane でも inf/inf の NaN を生まない。
  NaN 寄与を含む lane は全寄与の勾配を NaN とする（PyTorch と同じ）。
- `Amax`／`Amin`: 結果と等しい寄与の個数 `N` で `g` を均等に分ける（`include_self == true` なら入力も数える）。
  結果が NaN の位置は `N == 0` で勾配 0。
- 触れられていない位置の `d_input` は全 mode で `g`。

## 4. 境界検査

- `scatter_reduce`: `check_same_tape` → `scatter_reduce_layout`（`scatter_out_shape` による `dim`・rank・
  `index`／`src` の shape 一致・`dim` 以外の軸で `index <= input`、要素数と `f64`／`usize` 幅のバイト数の
  `checked_mul`・`isize::MAX` 超過）→ `index` 全値の範囲検査 → 実体化 → `BackendOps` → 戻り shape 検証
  → `push_eager`。共有カーネルも入力スライス長と添字範囲（負値を含む）を再検査する
  （`ShapeError::ElementCountMismatch`／`IndexOutOfRange`）。VJP 冒頭で上流勾配の shape を検査する。
- `index_add`／`index_copy`（`plan_axis_index`）: `dim < rank` → `index` は rank 1 で長さ `source.shape[dim]` →
  `source` の rank 一致と **`dim` 以外の全軸の完全一致**（`scatter_out_shape` は `<=` しか見ないためここで等号を
  要求して fail-closed にする）→ 添字の値域（負値を含む）→ `checked_bytes_for` → `index` の broadcast view。
- `masked_scatter`: `broadcast_shape`（`x`・`mask` の双方向。`bool_ops::masked_select` と同じ）→
  `checked_bytes_for`・`checked_axis_len_as_i32`・`checked_index_alloc_len` → 真の個数 `M` と `source` 要素数 `S` の
  比較（`M > S` は `InvalidArgument`）→ flat 添字（`Vec<i32>`、長さ `M`）。`M == 0` も同じ合成経路を通る
  （空 index の `scatter` が成立する。テストで固定）。
- **すべての検査を tape にノードを積む前に終える**（エラー時に孤児ノードを残さない。テストで固定）。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape が異なる場合は `BackendError::ShapeMismatch` で拒否する。
- `unsafe`／`get_unchecked`／本番経路の `unwrap`／`expect` は使わない。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/indexed-update-pytorch-reference/`。
f32 は u32 ビットパターンで保存）。

- 有限群 172 件（`scatter_reduce` 156・`index_add` 4・`index_copy` 4・`masked_scatter` 8）: forward と入力・src
  勾配がすべて REQ-2 統一複合判定で一致（`Amax`／`Amin`・`index_copy`・`masked_scatter` の forward は bit 一致）。
  `Prod` の 0 が 0／1／2 個以上のケースも勾配が一致した。
- `selfeq` 群（8 件）と `nonfinite` 群（70 件）: **forward は全件一致**。勾配のみ次の 26 件で PyTorch と異なり、
  差が `Amax`／`Amin` の勾配分配と `Prod` の inf 寄与に限られることをテストで固定している（`KNOWN_GRAD_DIFFS`）。

| 項目 | PyTorch 2.14.0（実測） | 本実装 | 扱い |
|---|---|---|---|
| `include_self == false` の `Amax`／`Amin` で、触れられた位置の入力値が結果と偶然一致する場合の勾配分配 | 入力も分配数に数える（`src` への勾配が `g / (一致数 + 1)` になる） | 入力は寄与に含まれないため数えない | 実測差分 10 件（`selfeq` 群 6 件、`nonfinite` 群の `nan_self_amin_noself`・`posinf_src_amin_noself`・`neginf_src_amax_noself`・`neginf_src_amin_noself` の 4 件）。本実装の方が「勾配の総和が上流勾配と一致する」。生成スクリプトは一致が起きない入力を主系列（有限群）に残し、一致する行を独立の `selfeq` 群へ自動で分ける |
| `Prod` の VJP で inf 寄与を含む lane（例 `include_self=false`・`src=[inf, 2]`・`g=1`） | `結果 / 値` の割り戻しで inf/inf = NaN（`src[0]` の勾配も NaN） | 他寄与の積を直接求め `src[0]` の勾配は 2.0（PR #2785 レビュー P1） | 実測差分 10 件（`posinf_src_prod_*`・`neginf_src_prod_*`・`inf_minus_inf_prod_*`・`all_neginf_nonself_prod_*`・`all_posinf_nonself_prod_*` 各 self／noself）。NaN 寄与の lane は一致 |
| 結果が NaN の位置の `Amax`／`Amin` 勾配（`include_self` 真偽とも） | NaN（`g / 0`） | 0（`N == 0` は勾配なし） | 実測差分 6 件（`nan_src_amax_self`／`nan_src_amax_noself`／`nan_src_amin_self`／`nan_src_amin_noself`／`nan_self_amax_self`／`nan_self_amin_self`）。非有限入力の勾配一致は受入条件にしない |
| 索引の型 | `int64` | `Tensor<i32>`（`Var::gather`／`scatter` と同じ慣例） | 範囲外は型付きエラー |
| `index` が `src` より小さい形 | 受理（実測） | `index.shape == src.shape` を要求し拒否 | 差分（スコープ外。`Var::scatter` と同じ契約） |
| 0 次元入力の `scatter_reduce` | 受理（実測） | `ShapeError::AxisOutOfRange` で拒否 | 差分 |
| 範囲外・負の添字 | `RuntimeError`／`IndexError` | `AutodiffError::InvalidArgument`（負値を含む） | 一致（拒否） |
| `dim` 範囲外 | `IndexError` | `ShapeError::AxisOutOfRange` | 一致（拒否） |
| `index_add`／`index_copy` の長さ・他軸 shape 不一致 | `RuntimeError`／`IndexError` | `AutodiffError::Shape` | 一致（拒否） |
| `masked_scatter` の source 不足 | `RuntimeError` | `AutodiffError::InvalidArgument` | 一致（拒否） |
| `masked_scatter` の mask が broadcast 不能 | `RuntimeError` | `AutodiffError::Shape`（`broadcast_shape`） | 一致（拒否） |
| `masked_scatter` の mask が `x` より大きい broadcast | 受理（実測。出力は broadcast 後の shape） | 受理（同じ） | 一致 |
| `index_copy` の重複添字 | 未定義 | 行優先で最後の書き手が勝つ（勾配も同様） | 本実装の方が強い契約 |
| `index_add` の `alpha` | あり | なし（§2） | 差分（事前スケールで等価） |
| 負の `dim`・整数 dtype・`out=` 引数 | あり | 非対応 | `Var` は f32 のみ |

PyTorch 側で backward が例外になる組合せは無かったため、勾配を独自に作った組合せは無い。tolerance・baseline は
変更していない。

## 6. テスト構成

- `crates/tensor-core/src/indexed_update.rs`（単体）: 5 mode × `include_self` の手計算値・未書き込み位置の
  ビット保存（NaN・`-0`・`inf`）・`Sum` の `f64` 逐次和・非末尾 `dim`・空 index・境界エラー（巨大 shape・
  範囲外／負の添字・スライス長不一致）・VJP の中心差分一致・`Prod` の 0 を含む排他的積・タイ均等分配と NaN の
  勾配 0・run-to-run bit 決定性。`backend_ops.rs`: 既定 `Unsupported`。
- `crates/autodiff/tests/indexed_update_parity.rs`（17 件）: fixture 突合（有限群は完全一致、`selfeq`／
  `nonfinite` 群は差分名を列挙して固定）・エラーケース突合・`Sum`＋`include_self` が `scatter_add` と bit 一致・
  `index_add` が手で展開した `scatter_add` と一致・`index_copy` の最後の書き手と勾配・`masked_scatter` の余剰
  source の勾配 0・中心差分・モック `BackendOps`（`Unsupported` フォールバック／他エラーの伝播／誤 shape／
  `scatter` が `Unsupported` のときの合成 3 演算）・ノード数（各 1、エラー時は増えない）・別 tape は
  `TapeMismatch`・巨大 broadcast view の確保前拒否・`create_graph` が型付きエラー・run-to-run 決定性。
- `crates/backend-cpu/tests/indexed_update_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps::
  indexed_scatter_reduce` の直接呼び出し（解析値・strided 入力・型付きエラー・決定性）、CUDA（macOS では
  Metal も）が `Unsupported` を返し panic しないこと。
- `crates/facade/tests/indexed_update_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし。
  `scatter_reduce` 全 mode×`include_self`、`index_add`、`index_copy`、`masked_scatter`）。CUDA／Metal 実機は
  `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `indexed_update_ops` への 1 行委譲で公開する。

- `Var::scatter_reduce(&self, dim, index: &Tensor<i32>, src: &Var<'t>, reduce: ScatterReduceMode, include_self: bool)`
- `Var::index_add(&self, dim, index, source)`・`Var::index_copy(&self, dim, index, source)`
- `Var::masked_scatter(&self, mask: &Tensor<bool>, source: &Var<'t>)`
- 引数型 `ScatterReduceMode` を facade ルートから再エクスポート（`QuantileInterpolation` の推奨形と同じ）。
- `indexed_update_ops` モジュール・`tensor_core::indexed_update` は再エクスポートしない。

理由は既存の `Var::scatter`／`scatter_add`／`index_put` と同じ呼び出し形になること。inherent メソッドと型の
再エクスポートの追加のみで非破壊。承認依頼は #2677、公開は承認後の #2678・#2679。承認後は
`IndexedUpdateOpsHoldDoctestGuard` と否定ガードを承認形の正ガードへ反転する。

承認事項（**すべて未承認**）: 上記 4 メソッドと `ScatterReduceMode` の公開、メソッド名・引数形、`index_add` の
`alpha` を公開形に含めるか。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679）。
- CUDA／Metal の GPU 専用カーネルと実機計測（§10）。
- `index_add` の `alpha`、負の `dim`・負の添字、int64 索引、0 次元入力、`index` が `src` より小さい形、
  0 次元の `index`。
- `create_graph`（高階微分）・activation checkpoint・f64／f16／bf16 自動微分経路。
- 既存 `ScatterReduce`（`Overwrite`／`Add`）と `ScatterReduceMode` の統合、`scatter` の並列化。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）改定。
- 非有限入力・`include_self == false` の一致ケースでの PyTorch との勾配完全一致（§5 に実測差分を記録）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `IndexedUpdateOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>` に公開されるとコンパイルが失敗する正のプローブ |
| `indexed_update_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `indexed_update_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_indexed_update_ops`（＋自己テスト） | facade src の再エクスポート・`pub mod indexed_update_ops`／`pub mod indexed_update`・`ScatterReduceMode` の独自宣言・4 名の `fn` 宣言の否定検査 |
| `workspace_declares_indexed_update_ops_fn_names_only_in_allowed_locations` | workspace 全体で 4 名の `fn` 宣言が `autodiff/src/indexed_update_ops.rs` の各 1 件のみ |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_indexed_update_ops_match_cpu_reference`・
`metal_indexed_update_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順と、演算の系統ごとに異なる
期待結果（`scatter_reduce` はホストフォールバックの確認、合成 3 演算は既存 GPU scatter カーネル経由）は
`docs/perf/logs/indexed-update-ops-2641/README.md`。

## 11. 出典

- `docs/autodiff-cumulative-ops-decision.md`（#2636）・`docs/autodiff-shape-view-ops-decision.md`（#2639）・
  `docs/autodiff-indexing-inplace-design.md`（`index_put` の合成方式）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/indexed-update-pytorch-reference/README.md`
