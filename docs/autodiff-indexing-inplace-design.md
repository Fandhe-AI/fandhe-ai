# advanced indexing・index_put・index_put_ の設計判断記録

イシュー #2148（親 #2131）。`docs/autodiff-reduce-ops-decision.md` と
同型の記録。

## §0 結論

PyTorch 互換の複数軸整数配列索引 3 種（`advanced_indexing`／
`index_put`／`index_put_`）を、**`fandhe_ai_autodiff` のうち facade が
再エクスポートしない自由関数モジュール `indexing_ops`**（`crates/
autodiff/src/indexing_ops.rs`）として実装した（`matrix_ops`〈#2144〉・
`reduce_ops`〈#2147〉・`rearrange_ops`〈#2143〉と同じ判断枠組み）。
`Var` に inherent の `pub fn` は追加していない。

新規 `Op` はゼロ。既存の `Op::Gather`（`Var::index_select` 経由）・
`Op::Scatter`（`Var::scatter`／`scatter_add` 経由）と view 系
（`reshape`／`broadcast_to`／`contiguous`）の合成のみで forward・VJP
双方の意味論を過不足なく表現できる。`crates/backend-*`・
`crates/tensor-core` は変更していない。

`index_put_`（in-place 代入の糖衣）は、`Var<'t>` が `Copy` なハンドル・
`Tensor<f32>` が immutable 値・tape が append-only という不変値 API の
制約上、「`index_put` を呼んでローカルハンドルを再束縛するだけ」の
関数として実装した（tape ノードも `Tensor` も書き換えない。§2.1）。

Issue #2148 は「facade への 3 関数の `pub use` 再エクスポート（経路
2）」を承認事項として明示しているが、`facade` は `Var` をそのまま
再エクスポートしているため（`crates/facade/src/lib.rs`）、`Var` に
inherent メソッドを 1 つ足すだけで facade の公開面が広がる。このツリー
（親 #2131）の先例に倣い、承認が取れるまで **autodiff に新設した
モジュール `indexing_ops` の自由関数 3 個**で満たす（承認後に追加する
作業は `Var::advanced_indexing` 等の薄い委譲メソッドと facade ガードの
撤去のみ）。

## §1 背景

イシュー #2148・親 #2131 のどちらにも所有者の承認コメントはない（着手
前に `gh issue view --json comments` で確認済み）。親 #2131 はこの
ツリーでの facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の 2
段階と定めているため、本実装は内部クレート限定に倒す。

課題: `Var::gather`／`index_select`／`scatter`／`scatter_add`（#1776）
は単一軸の索引しか扱えない。PyTorch の複数軸整数配列索引（`x[i0,
i1]` の読み出し）と索引位置への代入（`x[i0, i1] = v`・
`torch.index_put(accumulate=…)`）にはまだ対応がなかった
（`docs/compat-api-scope.md` §1.2 Tier 1「index 系」行の残り）。

## §2 設計判断

### §2.1「in-place」の解釈: 関数的な再束縛

`Var<'t>` は `Copy` なハンドル、`Tensor<f32>` は `Arc` を共有する
immutable 値、tape は append-only であるため、バッファを書き換える
本当の in-place はデータモデル上表現できない（`&mut self` で tape
ノードの値を書き換える API は作らない）。

代入 `tensor[idx] = value` の正は、非破壊版 `index_put(x, indices,
values, accumulate) -> Var`（PyTorch `torch.index_put`〈アンダース
コアなし〉相当）とする。3 つ目のメソッド `index_put_`（in-place 代入
の糖衣）は、自由関数 `index_put_(x: &mut Var<'t>, …) -> Result<(),
AutodiffError>` とし、中身は `*x = index_put(x, …)?` の**ローカル
ハンドルの再束縛だけ**である。

含意（doctest・単体テストで固定済み）:

- 同じノードを指す他の `Var` コピーは古い値のまま（エイリアスが無い。
  PyTorch の view／storage 共有とは異なる）。
- 旧ノード（葉を含む）は tape に残り、勾配もそのまま受け取る。
  `Gradients::get(&x_new)` は新ノードの勾配を返す。
- `nn` パラメータ（`Tensor<f32>`）は変わらない。パラメータ更新は
  `Module::set_parameter` の経路を使う。
- PyTorch は requires_grad な葉への in-place を拒否するが、本 API は
  エイリアスが無いため勾配の正しさが崩れない。拒否せず許容する
  （§4）。
- Rust の `IndexMut` は `&mut Output` を返す契約のため、新ノードを
  作る `x[idx] = v` 構文は作れない（演算子構文は対象外。§5）。

関数名 `index_put_`（末尾アンダースコア）は `non_snake_case`・clippy
のいずれにも抵触しないことを実装時に確認済み（`cargo clippy
--workspace --all-targets --all-features -- -D warnings` 通過）。

### §2.2 API 配置

`crates/autodiff/src/indexing_ops.rs` に自由関数 3 つを置く:

| 関数 | シグネチャ | PyTorch 相当 |
|---|---|---|
| `advanced_indexing` | `fn advanced_indexing<'t>(x: &Var<'t>, indices: &[Tensor<i32>]) -> Result<Var<'t>, AutodiffError>` | `x[i0, i1, …]`（先頭 k 軸の整数配列索引） |
| `index_put` | `fn index_put<'t>(x: &Var<'t>, indices: &[Tensor<i32>], values: &Var<'t>, accumulate: bool) -> Result<Var<'t>, AutodiffError>` | `torch.index_put` |
| `index_put_` | `fn index_put_<'t>(x: &mut Var<'t>, indices: &[Tensor<i32>], values: &Var<'t>, accumulate: bool) -> Result<(), AutodiffError>` | `Tensor.index_put_`／`x[idx] = v`（再束縛による糖衣） |

`Var` の inherent メソッドにはしない（§0 参照）。`crates/autodiff/
src/lib.rs` に `pub mod indexing_ops;` を追加した。

### §2.3 `Op`・`BackendOps`・カーネルは新設しない

Issue の対象範囲にある「3 カーネル（`backend-cpu/src/ops.rs`）」は
見積もり上の想定だった。既存の `Op::Gather`／`Op::Scatter`
（`Overwrite`／`Add`）と view 系（`reshape`／`broadcast_to`／
`contiguous`）を組み合わせれば、forward・VJP ともに意味論を過不足
なく表現できる。

`gather`／`scatter` は CPU（`backend-cpu::gather_scatter`）・CUDA
（#1777）・Metal（#1778）に実装済みで、既定では `Unsupported` の
とき「ホストフォールバック」する。「CUDA／Metal へ到達できる」という
Issue の契約は追加作業なしで満たせる。

`BackendOps` へのメソッド追加は crates.io 公開済みの `tensor-core`
の公開 trait を広げることになるので避ける（`matrix_ops` §2.2 と同じ
判断）。`crates/backend-*/**`・`crates/tensor-core/**` は変更しない。

### §2.4 合成方式

用語: `x.shape = [d0, …, d_{k-1}] ++ R`（`k = indices.len()`、`R` は
残り軸）、`B = broadcast_shape(indices[*].shape())`、`M = numel(B)`、
`P = d0 × … × d_{k-1}`。

共通の前処理（`fn plan_flat_index`。非公開ヘルパー）:

1. `1 <= k <= rank` を検査する（`k=0` は `InvalidArgument`、`k>rank`
   は `InvalidArgument`）。
2. `fandhe_ai_tensor_core::broadcast_shape` を畳み込んで `B` を求める
   （失敗は `AutodiffError::Shape`）。
3. `P` を `checked_mul` で求め、`checked_axis_len_as_i32` を通す
   （平坦添字を `Tensor<i32>` で持つため）。
4. `M` を `checked_mul` で求め、`checked_index_alloc_len` を通す。
5. 各 index テンソルを `B` へ broadcast して行優先に走査し、
   `0 <= v < d_j` を検査する（違反は `InvalidArgument`。負値は拒否）。
   検査を通ったものから `flat = Σ v_j × stride_j` を `checked_mul`／
   `checked_add` で計算し、`Tensor<i32>` `[M]` を作る。

`advanced_indexing`: `x.contiguous()` → `reshape([P] ++ R)` →
`index_select(0, flat[M])` → `reshape(B ++ R)`。VJP は `Op::Gather`
の VJP（決定的な `scatter_add`）で自動的に成り立つ。重複添字の勾配は
加算される（PyTorch と同じ）。

`index_put`:

1. `values.broadcast_to(B ++ R)` → `contiguous()` →
   `reshape([M] ++ R)`。broadcast できない場合は `ShapeError`。
2. 添字: `flat[M]` → `reshape([M, 1, …, 1])` → `broadcast_to([M] ++
   R)`（`scatter` は `index.shape() == src.shape()` を要求する。
   `Var::scatter` 内部の `index.contiguous()` にそのまま渡せるため
   事前の `contiguous()` は不要）。
3. `x.contiguous()` → `reshape([P] ++ R)` を作り、`accumulate=false`
   なら `scatter(0, idx, src)`（`Overwrite`）、`accumulate=true` なら
   `scatter_add(0, idx, src)` → `reshape(x.shape)`。
4. VJP: `Op::Scatter` の既存 VJP をそのまま使う。`Overwrite` では
   書き込まれた位置の `d_x` が 0 になり、`d_values` は最後の書き手
   だけに流れる。`Add` では `d_x = g`、`d_values = gather(g)`。その後
   `broadcast_to` の VJP（`reduce_to_shape`）で values の元の shape
   へ縮約される。

`index_put_`: `index_put` を呼んで `*x` を再束縛するだけ。

## §3 数値契約

- `advanced_indexing` の forward、`index_put(accumulate=false)` の
  forward: コピーだけ（算術なし）なので、3 バックエンドで構造的に
  **bit 完全一致**する（`NaN` の payload も保たれる。`cpu_advanced_
  indexing_preserves_nan_payload` で固定済み）。
- `index_put(accumulate=true)` の forward、および重複添字を含む
  backward（`scatter_add`）: `ScatterReduce::Add` の `f64` 決定的
  集約契約に従う。CPU と NaiveOps は
  `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合
  判定）で比較する。Metal の `scatter_add` は既存の `crates/
  backend-metal/tests/gather_scatter_parity.rs` で CPU と bit 一致を
  確認済み（本 PR の差分外）。CUDA・Metal の実機実測は #7 へ申し
  送る。
- 重複の無い backward: scatter の寄与は各位置 1 つだけで、残りは
  厳密な `+0.0` なので bit 一致する。

| 演算 | forward | backward |
|---|---|---|
| `advanced_indexing` | コピーのみ（**bit 完全一致**） | `Op::Gather` の決定的 `scatter_add` VJP（重複なし: bit 一致／重複あり: REQ-2 統一複合判定） |
| `index_put`（`accumulate=false`） | コピーのみ（**bit 完全一致**。重複は最後の書き手） | `Op::Scatter(Overwrite)` の VJP（重複なし: bit 一致／重複あり: REQ-2 統一複合判定） |
| `index_put`（`accumulate=true`） | `ScatterReduce::Add` の `f64` 決定的集約（REQ-2 統一複合判定） | `Op::Scatter(Add)` の VJP（REQ-2 統一複合判定） |
| `index_put_` | `index_put` に同じ | `index_put` に同じ |

## §4 PyTorch との差異

- 重複添字で `accumulate: false` の場合、PyTorch は未定義だが本実装は
  `ScatterReduce::Overwrite` の契約により B の行優先走査で**最後の
  書き手が勝つ決定的な結果**になる（PyTorch より強い契約）。VJP も
  同じ契約に揃っている。
- 重複添字で `accumulate: true` の場合: `ScatterReduce::Add` の `f64`
  アキュムレータによる行優先の逐次加算（決定的）。
- 負の添字（wrap-around）は拒否する（`gather`／`scatter` の既存契約
  と揃えて fail-closed にする。#1776 の追補でも対象外）。
- 索引できるのは先頭の連続 `k` 軸だけ。スライスや `None` を挟む指定、
  先頭以外の軸への索引は対象外（`permute` を先に使って代替できる）。
- bool マスク索引（`x[mask]`）は出力 shape がデータに依存する
  （`masked_select` と同じ種類）ため対象外。
- 添字の dtype は `Tensor<i32>` だけ（既存の `gather` と同じ）。
  PyTorch の int64 には対応しない。
- `M == 0`: `advanced_indexing` は shape `B ++ R` の空テンソルを返す。
  `index_put` は恒等（`reshape` を通して x と同じ値）。
  `numel(x) == 0` かつ `M > 0` は範囲検査で `InvalidArgument`
  になる（`dim_size == 0` に対しどんな `v >= 0` も範囲外のため、
  §2.4 の範囲検査が特別扱いなしに自然に拒否する）。どのケースでも
  panic しない（`advanced_indexing_empty_index_returns_empty_output`・
  `index_put_identity_when_index_empty` で固定済み）。
- `index_put_` はエイリアスが無いため、PyTorch が requires_grad な
  葉への in-place を拒否するのに対し本 API は許容する（§2.1）。

## §5 スコープ外（out-of-scope-tracking の候補）

- facade 公開（`Var::advanced_indexing`／`index_put`／`index_put_` の
  委譲メソッド追加と保留ガードの撤去）: 経路 2 の承認待ち。窓口は
  #2148・#2131。
- 負の添字の wrap-around・int64 添字・スライスと混在する指定・先頭
  以外の軸への索引・bool マスク索引（`x[mask]`）: §4 参照。
- 演算子構文 `x[idx] = v`（Rust の `IndexMut` の契約上作れない）。
- GPU 専用カーネル（既存の `gather`／`scatter` の経路で到達するため
  不要。性能最適化が要るなら別イシュー）。
- CUDA（GB10）・Metal（M4 Max）の実機 parity の実測: 申し送り
  （`docs/perf/logs/indexing-inplace-2148/README.md`）。

## §6 承認事項（未承認として列挙）

1. facade 公開面の拡張（Issue 記載の承認事項）
2. 負の添字の wrap-around の受け入れ（`gather`／`scatter` の既存契約
   の変更を伴う）
3. `backend-cpu`／GPU の専用カーネル・`BackendOps` の拡張
   （`tensor-core` の公開 trait を広げる）

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/indexing-inplace-2148/README.md` へ測定
コマンド案・期待結果を申し送る。

## §8 実装記録（イシュー #2148）

- `crates/autodiff/src/indexing_ops.rs`（新規）: `advanced_indexing`・
  `index_put`・`index_put_`（3 自由関数）・非公開ヘルパー
  `plan_flat_index`・モジュール doc・単体テスト 23 件（forward・
  broadcast・重複添字・エラー系・勾配の有限差分検算・`index_put_` の
  再束縛意味論）
- `crates/autodiff/src/lib.rs`: `pub mod indexing_ops;`
- `crates/facade/src/lib.rs`: `VarIndexingOpsHoldDoctestGuard`（正の
  プローブ 1 ブロック方式。`matrix_ops`／`reduce_ops` と同型）
- `crates/facade/tests/api_surface.rs`: 4 テスト追加
  （`indexing_ops_hold_doctest_globs_all_pub_modules`・
  `indexing_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_indexing_ops`・
  `workspace_declares_indexing_ops_fn_names_only_in_autodiff_
  indexing_ops`）
- `crates/facade/tests/indexing_ops_backend_parity.rs`（新規）: CPU
  vs NaiveOps の parity テスト 5 件 + CUDA／Metal 実機 `#[ignore]`
  テスト 4 件（`docs/perf/logs/indexing-inplace-2148/README.md` へ
  申し送り）
- `docs/compat-api-scope.md`・`docs/compat-feature-gap.md`：本 PR の
  実装状況を追記
