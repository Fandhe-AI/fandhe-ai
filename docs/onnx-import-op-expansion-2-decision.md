# ONNX import op 拡大 2（Gemm 変種・Clip・活性化・演算）設計判断記録

イシュー #2186（親: ONNX import op 拡大シリーズ）。

## 1. 背景・目的

`onnx-interop::onnx::interp`（TASK-7.2b・#78）が対応する op を、PyTorch など
から export した代表的なモデル部品（ReLU6＝`Clip(0,6)`、Linear head＝
`Gemm(transB=1)`、GELU、attention mask の `Where`＋`Expand`、LayerNorm 分解の
`ReduceMean`、`Pad`→`Conv`、アップサンプリング `Resize`）を実行できるところ
まで広げる。追加する op は `Clip`・`Tanh`・`Gelu`・`Where`・`Expand`・
`ReduceMean`・`Pad`・`Resize` の 8 個。加えて既存 `Gemm` の属性検証を固める
（旧実装は `alpha`／`beta`／`transA`／`transB` を無検証で読んでいた）。

## 2. 委譲の形

新規 8 op は `crate::ops`（`Tensor<f32>` 専用のホスト関数）へは実装せず、
`fandhe_ai_autodiff::Var` の同名演算（`tanh`／`gelu`／`gelu_tanh`／`clamp`／
`where_cond`／`expand`／`mean_dims`（+ 全軸縮約は `mean(None)`）／`pad`／
`interpolate`）へ委譲する（`crates/onnx-interop/src/onnx/interp_ext.rs`）。
理由: `crate::ops` に等価演算が存在せず新規実装が必要になるが、
`fandhe_ai_autodiff::Var` が既に数値契約込み（正規化統計・勾配の長軸縮約の
`f64` アキュムレータ規律等）で提供する実装を二重実装しない判断。

各 `compute_*` は `Tape::new()`（naive CPU 参照実装）を毎回新規に構築し、
入力を `tape.var_no_grad(&t)` で無勾配 `Var` として載せてから演算を呼び、
`.to_tensor()` で結果を取り出す。`interp::run_impl` が受け取る
`dev_ops: Option<&dyn BackendOps>`（CUDA／Metal 実行 opt-in。イシュー
#2077）はここでは一切参照しない——`Tape::new_with_ops` は
`Box<dyn BackendOps + Send>` の所有値を要求するため、借用
`Option<&dyn BackendOps>` からは構築できない。このため新規 8 op は
**常にホスト（CPU）実行**であり、`run_impl` のディスパッチ表では常に
`used_device = false` を返す。CUDA／Metal への到達は既定の未対応
フォールバック（ホスト計算）で担保する（契約「CUDA／Metal は常にホスト
計算へフォールバック」を満たす）。

非 F32 の dtype は `Var`（f32 専用）では扱えない。`Expand` のみ shape 計算
のサブグラフで `i64` が現れるため、tensor-core の既存演算
`Tensor::broadcast_to(..).contiguous()`（算術を含まない純コピー）で
`I64`／`Bool`／`F16` にも対応する。それ以外の新規 op は非 F32 を
`InterpError::TypeMismatch` で拒否する。

## 3. autograd（`onnx/autograd.rs`）の扱い

`autograd.rs` の forward は `crate::ops` の同一関数を使うため interp と
bit 一致するという不変条件を持つが、今回の委譲形（`fandhe_ai_autodiff::Var`
への直接委譲・毎回新規 `Tape`）はこの不変条件の外にある。このため新規 8 op
の勾配対応（autograd 経路への接続）は実装しない。`dispatch_node` の明示的な
fail-closed 腕（`"Gather" | "Unsqueeze" | ... | "Transpose"`）に新規 8 op を
追加し、`AutogradError::UnsupportedInAutograd` を返す（`other =>` の一般
`UnsupportedOp` に落とすと「未対応 op_type」と誤診断され、interp 側の対応
状況と矛盾するため）。

`autograd.rs` の `"Gemm"` 腕は属性を独自に読んでいたため、`interp::
read_gemm_attrs`（下記）へ切り替えた（forward は引き続き `ops::gemm` を
使う）。

## 4. op ごとの受理・拒否（要約）

Graph は opset を保持しないため、opset によって入出力の形が変わる op
（Clip・ReduceMean・Pad・Resize）は「入力数・属性の有無」という構造から
判定する。両方の形式が同時に検出された場合は fail-closed（
`InterpError::InvalidAttribute`）で拒否し、無言でどちらかを優先しない。

| op | 受理する形 | 主な拒否ケース |
|---|---|---|
| `Tanh` | 入力 1・F32 | 非 F32・入力数不一致 |
| `Gelu` | 入力 1・F32。`approximate`（STRING）は `"none"`（既定）／`"tanh"` | 上記以外の `approximate`（空文字列含む） |
| `Clip` | attr 形（opset 6: `min`／`max` FLOAT 属性）／入力形（opset 11+: 第 2・第 3 入力は省略可・rank 0 の F32）。省略側は `-inf`／`+inf` | 形式混在・非 rank0 の min/max・非 F32 |
| `Where` | 入力 3（cond: Bool、X／Y: F32）。双方向 broadcast | cond が非 Bool・X/Y が非 F32 |
| `Expand` | 入力 2（data、shape は 1 次元 I64）。出力形状は `broadcast_shape(input.shape, shape)` | 負の dim・shape が非 1 次元 |
| `ReduceMean` | `axes` を attr（opset<18）／入力（opset 18）で受ける。`keepdims`（既定 1）・`noop_with_empty_axes`（既定 0） | attr と入力の混在・軸範囲外・重複軸（`Var::mean_dims` 委譲） |
| `Pad` | attr 形（`pads`／`value`／`mode`）／入力形（`pads` 必須・`constant_value`／`axes` 省略可）。`mode="constant"` のみ | 形式混在・`reflect`/`edge`/`wrap`・負の pads（クロップ） |
| `Resize` | rank 4（NCHW）限定。N/C 軸倍率 1 固定。受理する `mode`/`coordinate_transformation_mode`/`nearest_mode` の組合せは本 doc §4.1 | ONNX 既定 nearest・`cubic`・`antialias`／`exclude_outside`≠0・`keep_aspect_ratio_policy`≠stretch・非整数倍率 |

### 4.1 Resize の受理する組合せ

- `nearest` + `asymmetric` + `floor` → `InterpolateMode::Nearest`
- `nearest` + `half_pixel` + `round_prefer_ceil` → `InterpolateMode::NearestExact`
- `linear` + `half_pixel` → `Bilinear{align_corners: false}`
- `linear` + `pytorch_half_pixel`（出力サイズ 1 の特例を除く） → `Bilinear{align_corners: false}`
- `linear` + `align_corners` → `Bilinear{align_corners: true}`

`scales` は「1 以上の整数倍率」または「入力サイズを割り切る整数分の 1」の
みを受理する（ONNX は与えられた倍率をそのまま座標変換式に使うため、
`in/out` の対応が一意に決まらない倍率は写像できない）。

## 5. Gemm の固め

- `read_gemm_attrs`（`interp.rs`。interp と autograd の両方が使う単一
  情報源）で `alpha`／`beta`（FLOAT 型検証）・`transA`／`transB`（INT 型
  検証）を読む。旧実装は無検証の `attr_f32`／`attr_i64` を使っており、
  型を偽装した属性（例: `alpha` を INT 型で送り `f` をゼロ値のまま残す）
  が無言で既定値へ fallback していた。
- 入力数は 2〜3 個（`InputArityMismatch`）に限定する。
- 旧 opset（Gemm-6 以前）の `broadcast` 属性: 省略または非 0 は現行仕様
  相当のユニ方向ブロードキャストを許容、`0` は `C` の shape が出力
  `[M, N]` と厳密に一致する場合のみ受理する。

## 6. 参照値の出自

外部ツール（ONNX Runtime／PyTorch）はローカル環境に存在しないため、
テスト内の f64/f32 独立手計算（ONNX 仕様の算術式をコードから独立に
導出）を参照値とする。Nearest 系・Expand・Where・Pad の値コピー部分は
純コピーのため bit 完全一致（`assert_eq!`）で検証し、Gelu／Bilinear 等の
浮動小数点演算を伴う部分は REQ-7 事前固定判定式
（`abs_err / (|ref| + 1e-6) <= 1e-3`）で検証する（REQ-2 の複合判定とは
別指標。既存 `tests/onnx_interp.rs` と同じ方針）。

## 7. スコープ外・追跡候補

- 新規 8 op の autograd 対応（`Var` 経路での勾配・interp との bit 一致
  契約の再設計）
- `Where`／`Pad`／`ReduceMean` の非 F32（I64 等）対応
- Resize の既定 nearest（`half_pixel` + `round_prefer_floor`）・`cubic`・
  `antialias`・`tf_crop_and_resize`・`tf_half_pixel_for_nn`・rank 4 以外・
  非整数 scales
- Pad の `reflect`／`edge`／`wrap`・負の pads（クロップ）
- export allowlist（`SUPPORTED_OP_TYPES`・`ExportOp`）への新規 8 op の追加
  （import → export の roundtrip は現状非対称）
- 新規 op の GPU device 結線（`interp_device`）
- `Expand`／`Resize` の出力要素数の上限（DoS 緩和。巨大な broadcast／
  resize によるメモリ確保に既存 op と同様の上限を設けていない残余
  リスク）
- ONNX Model Zoo tier B モデルの再プローブ（`Conv` の `auto_pad`／group
  conv。#2199／#2200 マージ後）

## 8. 承認事項

なし。依存・`unsafe`・facade 公開面・tolerance はいずれも不変で、
`InterpError` への variant 追加（`Autodiff { node, source }`）は
`#[non_exhaustive]` の契約範囲内。
