# view 系ノード（reshape / transpose）の再計算方式化（イシュー #1047）

## 1. 背景

親イシュー #1043（Phase 3「カーネル融合・autodiff 実行モデルの強化」）
の一項目。burn-autodiff の checkpointing（`MemoryBound { retro_forward
}` 相当: view 系ノードは forward 出力を保持せず、backward で親ノードから
再導出する）を参考に、`autodiff` テープ上の view 系ノード（`reshape`・
`transpose`）が中間バッファを一切確保しないことを機械的に保証する。

## 2. 現状分析（実装前調査で確認した事実）

- 実装前の `autodiff` には view 系 `Var` 演算が存在しなかった
  （演算セット: `matmul`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・
  `max`・`mse_loss`・`cross_entropy_loss`。`crates/autodiff/src/var.rs`・
  `docs/public-api-design.md` §3.2）
- `tensor-core::Tensor` の `reshape`（contiguous 時）・`transpose`・
  `permute`・`narrow`・`broadcast_to` はいずれも `Arc<Storage>` 共有の
  zero-copy view（`crates/tensor-core/src/tensor.rs`）。非 contiguous な
  `reshape` は `ShapeError::NonContiguousReshape`（案 A・エラー方式。
  `docs/public-api-design.md` §2.2.1 の未決事項のうち安全側を採用）
- `TapeNode { op, shape, value: OnceCell<Tensor<f32>>, lazy_chain_size
  }`。`push_eager`（常時実体化）／`push_lazy`（elementwise 5 演算限定・
  `value` 空）／`push_resident_leaf`（ホスト値なし）の 3 経路が既にあり、
  「`value` を空のまま登録する」先例（`Op::ResidentLeaf`）が存在した
- 実体化は層 1 `materialize_fallible`（`Result`）・層 2
  `materialize_non_fallible`（infallible）の 2 段。いずれも
  `build_lazy_plan`／`fallback_per_op`／`eval_fallback` の 3 走査器が
  elementwise 5 演算の連結成分のみを `interior` として収集し、それ以外
  （非 elementwise・未実体化）は `_` 分岐で「葉」として扱う
- 融合設計（`docs/kernel-fusion.md` 表 4）は「transpose を挟む連鎖は
  融合しない（融合境界）」を既に確定済み

## 3. 設計（採用: retro_forward 相当の再導出モデル）

### 3.1 新規 `Op` variant と登録経路

- `Op::Reshape { input: NodeId }`（出力 shape は `TapeNode.shape` に
  保持済みのため payload に持たない）
- `Op::Transpose { input: NodeId, dim0: usize, dim1: usize }`
- 新規登録経路 `Tape::push_view(op, shape) -> NodeId`
  （`crates/autodiff/src/tape.rs`）: `value: OnceCell::new()`（**ホスト
  値を持たない**）・`lazy_chain_size: 0`。`push_eager`（値渡し）とも
  `push_lazy`（elementwise 5 演算限定）とも異なる第 3 の登録経路

**呼び出し契約**: `push_view` の `op` が指す `input` は、呼び出し前に
層 1（`materialize_fallible`）で実体化済みであること（`Var::reshape`／
`Var::transpose` がこの順序を守る）。view は既存バッファへの別解釈
でしかなく、参照先の実体が存在することを前提に `resolve_view`
（下記）が infallible に動作できる設計としたためである。

### 3.2 再導出ヘルパー `resolve_view`

`tape.rs::resolve_view(nodes, id) -> Tensor<f32>`（infallible）は
`Op::Reshape`/`Op::Transpose` を入力側へ再帰的に辿り、最初に実体化済み
（`value.get().is_some()`）なノードの値へ `reshape`/`transpose` を順に
適用して返す。**バッファは `Arc` 共有のみで確保しない**（`tensor-core`
側の `reshape`/`transpose` 自体が zero-copy）。`Err` 分岐（forward 側の
shape 検査が正しければ到達しない）は `debug_assert!` + 安全側フォール
バック（全要素 `0.0`）に吸収し、本番経路 panic 禁止方針を守る。

### 3.3 読み出し側の結線

- `materialize_fallible`／`materialize_non_fallible`
  （`crate::tape`）の冒頭に `op.is_view()` 分岐を追加し、
  `resolve_view` の結果を対象ノード自身の `OnceCell` にのみキャッシュ
  する（連鎖の途中ノードの `OnceCell` は触れない設計は既存の融合実装と
  同型）
- `lazy_leaf_value(nodes, n)`（旧 `lazy_leaf_value(node)` から署名変更）
  は 3 走査器（`build_lazy_plan`／`fallback_per_op`／`eval_fallback`）が
  view ノードを「未実体化の葉」として読む際に `resolve_view` へ委譲する
  よう変更した。この変更がないと、elementwise 演算の入力に view ノード
  が渡された場合（例: `relu(transpose(x))`）に旧来の契約違反フォール
  バック（`debug_assert!(false)` + ゼロ埋め）へ誤って落ちてしまう
  （view は「ホスト値を持たないのが正常」であり契約違反ではないため）

### 3.4 `Var` 公開 API

- `Var::reshape(&self, shape: &[usize]) -> Result<Var<'t>,
  AutodiffError>`: ①要素数一致検査（`checked_mul` によるオーバーフロー
  検査を含む）→ ②層 1 で入力を実体化 → ③実体化値の `is_contiguous()`
  を検査（非 contiguous なら `ShapeError::NonContiguousReshape`。
  `tensor-core::Tensor::reshape` の案 A に合わせ、暗黙コピーで
  「バッファ確保 0」の契約を破らない）→ ④`push_view` で記録
- `Var::transpose(&self, dim0, dim1) -> Result<Var<'t>, AutodiffError>`:
  ①軸範囲検査 → ②層 1 で入力を実体化 → ③`push_view` で記録

### 3.5 VJP（`grad.rs::vjp`）

- `Op::Reshape { input }`: `upstream.reshape(&input_shape)` を試み、
  非 contiguous（上流に `transpose` が挟まる場合等）で失敗したときのみ
  `upstream.contiguous().reshape(..)`（勾配バッファ側の明示コピーで
  あり、view ノード自身が確保を持つわけではない）にフォールバックする
  ——zero-copy を優先する順序を明記した
- `Op::Transpose { input, dim0, dim1 }`: 対合性（同じ軸で 2 回適用する
  と恒等）を利用し `upstream.transpose(dim0, dim1)` のみで閉じる
  （zero-copy）
- いずれも `Err`（forward 側の契約違反時のみ到達）は `debug_assert!` +
  安全側フォールバックで吸収し、ゼロ勾配で黙って続行しない

## 4. 採らなかった案

- **view 出力を `push_eager` で即実体化（`Tensor` view を保存）**:
  `Arc` 共有のためバッファ確保自体は 0 だが、「値を保持せず backward
  時に再導出する」という受け入れ条件の意図と乖離し、将来の tape 再利用
  （#1048）でノードクリア時にも view ヘッダが残ってしまう。不採用
- **`build_lazy_plan` 等へ `ops` を渡し view を融合連鎖の内部で解決**:
  3 走査器と `FusionPlan` 構築の横断変更・層 1/層 2 のエラー契約の
  再設計が必要。`docs/kernel-fusion.md` が transpose 混在連鎖を非融合と
  既に確定済みのため本イシューでは不要と判断し、スコープ外として
  §6 に記録する

## 5. スコープ外（別イシュー候補。起票はユーザー承認事項）

- `permute`／`narrow`／`broadcast_to` の `Var` 化（同じ `push_view`／
  `resolve_view` 骨格で追加可能）
- elementwise 融合連鎖の**内部**で view を解決する（`build_lazy_plan`
  への `ops` 配線・`FusionPlan` へのレイアウト情報導入）
- `tensor-core` 非 contiguous `reshape` の案 B（暗黙コピー）採否
  （ユーザー承認事項のまま未決）
- #1048（tape 再利用・ノードクリア API）での view ノード扱い
- GPU 常駐テンソル（`ResidentLeaf`/`LinearResident`）に対する view
  （デバイス側 stride 対応が必要）

## 6. メモリ実測記録（CPU ホスト側・CI 相当の一般環境）

`crates/autodiff/tests/view_zero_alloc.rs`（`bench_harness::
alloc_tracker::TrackingAllocator` による実測。`[[test]] harness =
false` の専用単一プロセス・単一スレッドバイナリ。実行:
`cargo test -p fandhe-ai-autodiff --test view_zero_alloc --
--nocapture`）。

入力バッファ: `2048 × 2048` f32（16,777,216 バイト = 16 MiB）。

| 検査 | 純増分ピーク | 閾値 | 判定 |
|---|---:|---:|---|
| `transpose` forward（ノード記録のみ） | 16 バイト | 16,777 バイト（バッファの 1/1000） | 合格（バッファの約 100 万分の 1） |
| `reshape` forward（ノード記録のみ） | 32 バイト | 16,777 バイト（バッファの 1/1000） | 合格（バッファの約 50 万分の 1） |
| `transpose` を 5 回連鎖した `Tape::backward` | 16,778,076 バイト | 33,554,432 バイト（バッファの 2 倍） | 合格（実測はバッファ 1 個分 + 約 860 バイトのヘッダオーバーヘッドのみ。`Op::Sum` の VJP が必要とする唯一の実データ確保） |

forward の 2 検査（16 バイト・32 バイト）は `TapeNode` の `shape: Vec<usize>`
ヒープ確保そのもの（`usize` 1〜2 個分）であり、テンソルデータの複製は
一切発生していないことを裏付ける。backward の検査は 5 段の view 連鎖を
経ても追加確保が連鎖長に比例せず「損失側で本来必要な勾配バッファ 1 個
分」に収まることを示し、`resolve_view` が各ノードで `Arc` 共有のみを
行い実データコピーを重ねないという設計を実測で裏付けている。

Linux x86_64・CUDA/Metal 実機なしの環境での計測であり、GPU 実機実測は
本イシューの対象外（本イシューは CPU ホスト側 autodiff の view API に
閉じるため。§5「スコープ外」参照）。
