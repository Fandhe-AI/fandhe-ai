# Metal GEMM の転置パターン別 strided 入口・先頭次元 collapse（イシュー #1040）

親: #1029（GEMM カーネルの candle 超え）。兄弟 #1037（タイル構成のテーブル
駆動選択 × 形状 × 転置）は、本イシューが確定させる `crate::layout::
TransposePattern`／`MatrixLayout` をタイル表のキーとして利用する役割分担
とする。

## 1. 背景

学習ループの VJP（`crates/autodiff/src/grad.rs` の `Op::LinearResident`
分岐）は `transpose2d(upstream)` で作った転置 view を
`ops.rs::MetalBackendOps::gemm_resident_lhs`／`gemm_resident_rhs` へ渡す。
従来はここで無条件に `Tensor::contiguous()` を呼んでおり、毎ステップ
ホスト側の転置コピー（repack）が発生していた。

また、Metal カーネル（`shaders/gemm.metal`）は A を `[m,k]` 行優先・
B を `[k,n]` 行優先・`Dims{m,n,k}` のみで添字計算しており、lda/ldb や
転置フラグを受け取る手段がなかった。バッチ次元（`[B,M,K]@[K,N]` を
`[B*M,K]@[K,N]` へ畳む経路）も存在しなかった（`tile.rs::actual_groups`
doc 参照）。

## 2. カーネル選択（転置パターン別入口）

| パターン | 選択カーネル | 理由 |
|---|---|---|
| NN（両オペランドが行優先 contiguous） | 既存 `gemm_simdgroup_tiled`（`dispatch_auto`／`tile::select`） | 既存の高速経路を変えない（#1037 の対象） |
| NT / TN / TT および collapse 経路・常駐 VJP 経路 | `gemm_tiled_bias_act` を stride 対応に拡張した strided 経路 | VJP 常駐経路が現に使うカーネル。実次元のまま padding 不要で、添字だけ変えれば FMA 累算順が不変（NN 指定時は既存とビット同一） |

`gemm_simdgroup_tiled` への転置ロード導入（`simdgroup_load` の transpose
引数・staged ロードの転置 scatter）は MSL コンパイル確認が Mac 実機必須の
ため本イシューでは行わず、#1037 へ引き継ぐ。

## 3. 設計

### 3.1 `crate::layout`（`cfg(target_os = "macos")` を付けない純粋モジュール）

- `TransposePattern`（NN/NT/TN/TT。`Copy + Eq + Hash`）
- `MatrixLayout { rows, cols, ld, transposed }`
- `classify_2d(shape, strides) -> Option<MatrixLayout>`: 行優先／列優先
  （転置 view）を分類。stride 0 のブロードキャスト・負 stride・rank≠2 は
  `None`（呼び出し側は従来の `contiguous()` へフォールバック）
- `collapse_leading_dims(shape, strides) -> Option<MatrixLayout>`:
  `[B0,…,M,K]` の先頭次元を `[B0*…*M, K]` へ畳む（candle の collapse 条件
  と同種）
- `required_span(&MatrixLayout) -> Option<usize>`: バッファ長検証用

### 3.2 `shaders/gemm.metal::GemmStrides`

```c
struct GemmStrides {
    uint lda;
    uint ldb;
    uint trans_a;
    uint trans_b;
};
```

`gemm_tiled_bias_act` に `constant GemmStrides& st [[buffer(7)]]` を追加し、
A/B の添字を

- `A(row, kk) = trans_a ? a[kk*lda + row] : a[row*lda + kk]`
- `B(kk, col) = trans_b ? b[col*ldb + kk] : b[kk*ldb + col]`

へ一般化した。手動境界チェック（`row < m && a_col < k` 等。REQ-8）は
変更していない。NN（`lda==k`・`ldb==n`・両フラグ 0）では式が従来と完全に
一致するため、既存 `gemm_tiled` / 非融合合成とのビット同一契約は保たれる。

### 3.3 `crate::gemm`

- `GemmStrides`（repr(C)。MSL 側とレイアウト一致。`size_of == 16` を
  Linux で機械検証）
- `validate_strided_dims`: 論理形状整合・`offset + required_span <=
  buf.len()`・`u32` 上限を fail-closed に検証
- `MetalGemm::dispatch_strided_bias_act_prepared`: 転置パターン・stride
  対応の新規公開入口
- `MetalGemm::dispatch_bias_act_prepared`（既存後方互換入口）は NN
  （`lda=k`・`ldb=n`）で新関数へ委譲するよう変更（数値結果は非後退）

### 3.4 `crate::ops`

- `gemm_resident_lhs`／`gemm_resident_rhs`: `layout::classify_2d` が
  `Some` を返す入力（転置 view を含む）は `Tensor::as_view_slice`
  （`tensor-core` 側の新規追加。非 contiguous でも非負 stride なら
  storage の借用スライスを返す）をそのまま Metal バッファへアップロード
  する（`MetalMemory::upload_view`）。`None`（stride 0 のブロードキャスト
  等）のみ `Tensor::contiguous()` へフォールダックし、
  `RESIDENT_HOST_REPACK_COUNT`（`pub(crate)` 診断カウンタ）を増やす
- `MetalBackendOps::gemm_collapsed_lhs`（inherent メソッド。`BackendOps`
  trait は変更しない）: `layout::collapse_leading_dims` が `Some` を返す
  `[B0,…,M,K]` は zero-copy、`None` の場合は `contiguous()` 後に
  `[B0*…*M, K]` へ reshape してから同じ strided 入口へ渡す

### 3.5 `tensor-core::Tensor::as_view_slice`

非 contiguous な view でも、全 strides が非負である限り
`[offset, offset + span)`（`span = 1 + Σ (shape_i − 1)·stride_i`）を
storage から借用で返す。`as_slice`（contiguous 限定）と異なり転置 view
もカバーする。`facade` 公開面には影響しない内部クレートの追加 API。

## 4. スコープ外（別イシューへ引き継ぎ）

- `gemm_simdgroup_tiled` 系への転置ロード・タイル表（#1037）
- `BackendOps` trait へのバッチ matmul／転置指定メソッド追加、
  `grad.rs` の `d_input = g @ Wᵀ` 直接計算（`transpose2d` 排除）は公開 API
  拡張のため別イシュー提案
- CUDA／CPU の同型 zero-repack 化（`backend-cuda`／`backend-cpu` の
  `ops.rs` も `contiguous()` を呼んでいる）
- Metal 実機実測（M4 Max）と性能目標判定

**#1046 での消化状況（追補）**: 上記のうち「`grad.rs` のホスト参照
実装経路（`eval::matmul` 経由）の転置コピー除去」は #1046 で消化した
（`layout` の分類ロジック〈`classify_2d`／`MatrixLayout`〉は `autodiff`
側にもクレート内非公開モジュールとして複製し、`backend-metal::layout`
は変更していない〈PR #1077 codex-review 対応の経緯を含め
`docs/matmul-vjp-zero-copy-decision.md` 参照〉）。
「`BackendOps` trait 拡張（`gemm_resident_rhs_nt` 等）による
`d_input` のデバイス側直接計算化」「CUDA／CPU の同型 zero-repack 化」
「Metal 実機実測」は #1046 でも未消化のまま引き続き別イシュー行きで
ある（`docs/matmul-vjp-zero-copy-decision.md` §3.2 が現時点の一覧）。

## 5. 実機実測（未実施。Mac セッションで実施予定）

```sh
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

- `tests/gemm_strided_parity.rs`: NN/NT/TN/TT parity・NN ビット同一・
  collapse parity
- `src/ops.rs` の `gemm_resident_lhs_transposed_b_does_not_increment_repack_counter`:
  転置 view 入力時に `RESIDENT_HOST_REPACK_COUNT` が増えないことの確認
- 学習 1 ステップ（Linear + SGD 常駐）の before/after 性能比較

（実測値は未記入。Mac 実機セッションでの実行後にこの節を更新する）
