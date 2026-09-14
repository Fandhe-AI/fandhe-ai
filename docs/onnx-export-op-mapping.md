# ONNX export op マッピング（イシュー #1773）

`onnx-interop` 内部（`crate::onnx::export_ops`）の export 側 op マッピング
（`ExportOp` -> `NodeProto`）の対応表・契約を記録する。facade へは一切公開しない
（`docs/compat-api-scope.md` 対象外・`onnx-interop` は crates.io 非公開クレート）。

## 0. スコープ境界

- 本 issue の対象は「`interp.rs` が読む 22 op の逆方向（内部 op -> `NodeProto`）」
  のみ。autodiff `Op`／`Tape` -> `ExportOp` の橋渡し・facade 公開は #1653／#1775
  のスコープ（`docs/facade-onnx-import-exposure-decision.md`）。
- import -> export -> import の総合 roundtrip・未対応 op を含むモデルの
  fail-closed 確認という総合テストは #1774 のスコープ。本ドキュメント・実装の
  単体テストは op 単位の対称性検証に限定する。

## 1. 契約

- **属性は既定値であっても常に全て書き出す**（省略すると「属性欠落＝既定値」
  という対称性テストが空虚に pass してしまうため）。唯一の例外は
  `ExportOp::Transpose::perm`: `None`（省略時の意味論＝rank 依存の軸反転）を
  静的な既定値で埋めず、属性自体を省略する。
- `NodeProto.domain` は常に空文字列（既定 opset）。
- 出力は全 op 単一（`interp.rs::require_single_output` と対称）。
- opset 形態は `ExportOptions::default().opset_version=17` に合わせ、
  `Reshape.shape`・`Squeeze`/`Unsqueeze.axes`・`Slice.starts/ends/axes/steps`
  は入力テンソル名として表す（attr 形〈opset<13〉は export しない）。
- `check_exportable`（層 B）は `build_model_proto` の先頭で必ず呼ばれ、
  `op_type` が allowlist（`SUPPORTED_OP_TYPES`）に含まれ、かつ `domain` が
  空文字列であることを検査する。違反は `ExportError::UnsupportedOp` で
  fail-closed に拒否する（無言 skip はしない）。

## 2. 対応表

| op_type | Tier | 入力（`[]` は省略可） | 属性 name: 型 = 既定値 | `interp.rs` 読み取り箇所 |
|---------|------|------|------|------|
| Gemm | 1 | A, B, [C] | alpha: FLOAT=1.0 / beta: FLOAT=1.0 / transA: INT=0 / transB: INT=0 | `compute_gemm` |
| MatMul | 1 | A, B | なし | `compute_matmul` |
| Add | 1 | A, B | なし | `compute_add` |
| Mul | 1 | A, B | なし | `compute_mul` |
| Div | 1 | A, B | なし | `compute_div` |
| Relu | 1 | X | なし | `compute_relu` |
| Sigmoid | 1 | X | なし | `compute_sigmoid` |
| Softmax | 1 | X | axis: INT=-1 | `compute_softmax` |
| Reshape | 1 | data, shape（入力） | allowzero: INT=0 | `compute_reshape` |
| Mod | 3 | A, B | fmod: INT=0 | `compute_mod` |
| Sqrt | 3 | X | なし | `compute_sqrt` |
| Erf | 3 | X | なし | `compute_erf` |
| Shape | 2 | data | なし | `compute_shape` |
| Gather | 2 | data, indices | axis: INT=0 | `compute_gather` |
| Unsqueeze | 2 | data, axes（入力・opset>=13 形） | なし | `compute_unsqueeze` |
| Squeeze | 3 | data, [axes]（入力・opset>=13 形） | なし | `compute_squeeze` |
| Concat | 2 | inputs...（1 個以上・可変長） | axis: INT（必須） | `compute_concat` |
| Slice | 2 | data, starts, ends, [axes], [steps]（全入力・opset>=13 形） | なし | `compute_slice` |
| Transpose | 3 | data | perm: INTS（省略可。省略時は rank 依存の軸反転） | `compute_transpose` |
| Cast | 3 | input | to: INT（1/7/9/10 のみ。export 側は値検査しない） | `compute_cast` |
| Constant | 3 | なし | value〈TENSOR〉／value_float〈FLOAT〉／value_floats〈FLOATS〉／value_int〈INT〉／value_ints〈INTS〉のいずれか 1 つ | `compute_constant` |
| LayerNormalization | 3 | X, Scale, [B] | axis: INT=-1 / epsilon: FLOAT=1e-5 | `compute_layer_normalization` |

Tier 1（必須・受入条件）・Tier 2（`slice_repro.onnx` fixture roundtrip に必要）・
Tier 3（推奨・全実装済み）。22 op すべて実装済み（削減なし）。

## 3. `Constant` の `value`（TENSOR）契約

`ConstantAttr::Tensor(RawTensor)` を書き出す際、埋め込む `TensorProto.name` は
ノードの出力名（`node.outputs[0]`）を決定的に用いる。`TensorProto.name` は
import 側（`decode_tensor`）がエラーメッセージにしか使わないため、この選択に
実害はない。

## 4. arity 検査（層 A）

必須入力の個数・非空文字列、省略可入力は末尾のみ・空文字列可、出力は全 op
厳密 1（`interp.rs::require_single_output` と対称）。違反は
`ExportError::InputArityMismatch`／`OutputArityMismatch`／
`EmptyRequiredInput` で拒否する。

## 5. 対象外事項

- autodiff `Op`／`Tape`／`Sequential` -> `ExportOp` の橋渡し・facade 公開
  （#1653／#1775）
- import -> export -> import の fixture roundtrip テスト・`interp::run` bit
  同一確認の総合テスト（#1774）
- `value_info`／`TypeProto` 非出力による外部ツール（`onnx.checker`）妥当性
  （#1772 既知事項）
- opset<13 の attr 形（`Squeeze`/`Unsqueeze` の `axes` 属性）での export・
  `Shape` の `start`/`end`・`LayerNormalization` の `stash_type`・`Constant`
  の `value_string(s)`／sparse（import 側も不参照）
