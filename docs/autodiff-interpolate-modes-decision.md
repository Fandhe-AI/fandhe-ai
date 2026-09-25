# `Var::interpolate` の残りモード実装 設計判断記録（イシュー #2152）

## 0. 背景

`Var::interpolate`（`torch.nn.functional.interpolate`／`tf.image.resize`
相当）は `InterpolateMode::Nearest`（#1757）と `InterpolateMode::Bilinear
{ align_corners }`（#1762）のみを持っていた。`docs/compat-feature-gap.md`
の #1762 追補は `scale_factor`／`recompute_scale_factor`・`antialias`・
`linear`／`trilinear`／`bicubic`・`nearest-exact` を対象外として記録して
いた。本イシューは親 #2131（PyTorch／TF 置き換え水準の API 網羅）の一部
として、この残りのうち `antialias` と `recompute_scale_factor=False` 相当
の座標系を除く全項目を実装した。

## 1. 「6 モード」の解釈

- PyTorch の `mode` は `nearest | linear | bilinear | bicubic | trilinear
  | area | nearest-exact` の 7 種。実装済みだったのは 2 種（`nearest`・
  `bilinear`）で、**残りは 5 種**である。
- イシュータイトルの「6（scale_factor・linear・bicubic 等）」は、#1762
  追補の対象外リスト（`scale_factor`・5 モード）と一致する。したがって
  本実装は **新規 variant 5 個（`Linear`／`Trilinear`／`Bicubic`／`Area`／
  `NearestExact`）＋ `scale_factor`（variant ではなく引数）で 6 項目**と
  解釈した。
- 受入条件文言の「6 variant 追加」とは数が合わない。実在しない 6 番目の
  モードを作って数を合わせることはしていない（安全側の判断）。
- `antialias`（新しいカーネル設計が要る）と `recompute_scale_factor=False`
  相当の座標系（§6）は対象外のまま。

## 2. `InterpolateMode` の variant 設計

`crates/tensor-core/src/backend_ops.rs`。`#[non_exhaustive]` enum への
variant 追加のため公開 API 非破壊（`ScatterReduce`／`Activation` と同じ
拡張性方針）。

| variant | 空間軸数 | フィールド |
|---|---|---|
| `NearestExact` | 任意（`1..=rank`） | なし |
| `Area` | 任意（`1..=rank`） | なし |
| `Linear` | ちょうど 1 | `align_corners: bool` |
| `Trilinear` | ちょうど 3 | `align_corners: bool` |
| `Bicubic` | ちょうど 2 | `align_corners: bool` |

`f32` 等のフィールドは持たせない（`InterpolateMode` は `Eq`/`Hash` を
derive しており、浮動小数点フィールドがあると崩れるため。既存
`Bilinear`/`Linear`/`Trilinear`/`Bicubic` はいずれも `bool` のみ）。

rank 検査は `ops_shape::interpolate_out_shape_for_mode` に一元化した
（`match mode { ... }` で期待 rank を求め、`size.len()` と不一致なら
`ShapeError::RankMismatch` を返す。`Bilinear` の既存検査と同じ構成）。

## 3. モード別の数値仕様

### 3.1 NearestExact（bit 完全一致契約）

PyTorch `nearest-exact` は `min(floor((dst+0.5)*in/out), in-1)`（`f32`
計算）。本実装は float を使わない整数専用の等価式 `src = ((2*dst+1)*in)
/ (2*out)` を `u128` で計算してから `min(src, in-1)` を取る
（`tensor-core::interpolate::nearest_exact_src_coord`。`nearest_src_coord`
と同じ overflow 対策）。`Nearest`（`src=(dst*in)/out`）とは half-pixel
オフセットの有無が異なるため一般に異なる添字を返す。

forward・VJP とも `Nearest` と同型（算術を含まない純コピー演算 →
scatter_add 1 tap）で、CPU ネイティブ・ホスト参照とも同じ関数を呼ぶため
bit 完全一致する。3 バックエンド間の bit 完全一致契約は `Nearest` と
同一（CUDA／Metal は本 issue でカーネル実装せず `Unsupported` による
ホストフォールバックのまま——GPU 側の丸めが Rust ホスト参照実装と一致
するかは未実測）。

### 3.2 Area（= adaptive average pooling）

軸ごとの窓は既存の `ops_shape::adaptive_window(o, in, out)` を単一情報源
として再利用する（`Op::AdaptiveAvgPool2d` と共有）。

forward（`autodiff::eval::interpolate_area`・`backend-cpu::interpolate::
interpolate_area`）: 窓内を空間軸の先頭を最も遅く・末尾を最も速く走査
する固定順序（row-major ネストループと同型。`adaptive_avg_pool2d` の
`for oh { for ow { .. } }` と同じ順序）で `f64` アキュムレータへ加算し、
最後に 1 回だけ `f32` へ downcast する（`.claude/rules/coding-rust.md`
「勾配の長軸縮約」節と同じ精度規律）。2 軸の場合は `backend-cpu::
pooling::adaptive_avg_pool2d` と **bit 完全一致**することをテストで
固定した（`crates/backend-cpu/src/interpolate.rs::tests::
interpolate_area_matches_adaptive_avg_pool2d_bit_exact`）。

空間軸数は任意（1..=rank）——`Nearest`／`NearestExact` と同じ rank 制約。

VJP（`grad::interpolate_area_vjp`）: 出力位置あたりの tap 数（窓要素数）
が形状依存で可変なため、`Bilinear`／`Linear`／`Trilinear`／`Bicubic` の
ような固定 K 個 scatter_add 平坦化とは異なる構成にした。`Op::
AdaptiveAvgPool2d` の VJP（`adaptive_avg_pool2d_vjp`）と同じ「`f64`
アキュムレータを持つ稠密配列への直接加算」方式を空間軸任意次元へ一般化
した（scatter_with_fallback を経由しないため backend 側 GPU scatter
実装は使わないが、`adaptive_avg_pool2d_vjp` も同じ設計であり本クレート
内で既に受け入れられている先例——計画からの妥当な簡略化と判断した）。

### 3.3 Linear（1 軸）／Trilinear（3 軸）

各軸の座標は既存の `bilinear_src_coord(dst, in, scale, align_corners)`
をそのまま使う（軸ごとの関数のため再利用できる）。

新しいブレンド関数（`tensor-core::interpolate`）:

- `linear_blend(v0, v1, l1) = l1.mul_add(v1, (1-l1)*v0)`（`bilinear_blend`
  の行方向補間と同じ式順）
- `trilinear_blend(v000..v111, l1x, l1y, l1z)`: `z0` 面・`z1` 面をそれ
  ぞれ既存の `bilinear_blend` で合成し、`linear_blend` と同じ式で `z`
  軸方向に結ぶ固定順序

VJP は `Linear`=2 tap・`Trilinear`=8 tap の重み付き scatter_add
（`grad::linear_src_index_and_weight_map`／`trilinear_src_index_and_
weight_map`。`bilinear_src_index_and_weight_map` と同型）。

受入契約は REQ-2 統一複合判定（GPU は将来の専用カーネルでも FMA 契約の
影響を受けるため）。ただし CPU ネイティブとホスト参照は同じ Rust 関数を
呼ぶため、テストでは bit 完全一致を検証した（`Bilinear` の前例と同じ）。

### 3.4 Bicubic（2 軸・16 tap）

PyTorch 準拠: cubic convolution で `A = -0.75`。

- `cc1(x) = ((A+2)*x - (A+3))*x*x + 1`（`|x|<=1`）
- `cc2(x) = ((A*x - 5A)*x + 8A)*x - 4A`（`1<|x|<2`）
- 係数は `[cc2(t+1), cc1(t), cc1(1-t), cc2(2-t)]`

**座標のクランプ有無（実装前に PyTorch ソースで確認済み）**: PyTorch の
`area_pixel_compute_source_index(scale, dst, align_corners, cubic)` は

```cpp
if (align_corners) {
  return scale * dst_index;
} else {
  scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
  return (!cubic && src_idx < 0) ? 0 : src_idx;
}
```

（`aten/src/ATen/native/UpSample.h`。2026-09-25 に
`raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/
UpSample.h` を取得して確認）。`cubic=true` のとき `align_corners=false`
でも `src<0` を**クランプしない**（cubic 補間は `[-1,0,1,2]` の近傍参照
が必要なため）。`bilinear_src_coord` の `max(0.0)` クランプとは異なる
契約であるため、`bicubic_src_taps` は `bilinear_src_coord` を呼ばず独立
に座標式を実装した。

`i = floor(src)`（`isize`。負値になりうる）、`t = src - i`。4 tap
`i-1..=i+2` をそれぞれ `[0, in_size-1]` へ個別にクランプする（REQ-8 の
境界検査。クランプで重複した tap もそのまま加算する——PyTorch の端点
飽和と同じ）。出力値はクランプしない（overshoot を許す。PyTorch と同じ）。

ブレンド（`tensor-core::interpolate::bicubic_blend`）: 行方向（x）の
4-tap 補間を 4 行分行ってから、列方向（y）の 4-tap 補間を行う固定順序
（PyTorch `cubic_interp1d` の入れ子順と同じ）。`fma` 連鎖で構成する。

VJP は 16 tap の重み付き scatter_add（`grad::bicubic_src_index_and_
weight_map`。重み = `taps_y.w[j] * taps_x.w[k]`）。

受入契約は REQ-2 統一複合判定（Rust 実装同士は bit 完全一致を検証済み）。

### 3.5 既存モード

`Nearest`／`Bilinear` の forward・VJP・CPU／CUDA／Metal カーネルは本
イシューで一切変更していない（既存テストと実機ベースラインを守るため）。

### 3.6 scale_factor（variant ではなく引数）

**配置判断**: `Var` に新しいメソッドは追加しなかった（`Var` は facade が
再エクスポートしているため、新しい `pub fn` を足すと facade 公開面の
拡張になり #2131 の承認ゲートにかかる）。autodiff に自由関数を置くと、
facade の保留ガード一式（doctest ガードと `api_surface.rs` のテスト）が
必要になる。そこで **tensor-core の純関数**
`interpolate_size_from_scale_factor(spatial_in: &[usize], scale_factor:
&[f64]) -> Result<Vec<usize>, ScaleFactorError>` とした。tensor-core は
内部クレートで、facade は個別の項目しか再エクスポートしないため、この
追加は承認事項にならない（`docs/compat-api-scope.md` §0）。利用者は
戻り値を既存の `Var::interpolate(&size, mode)` にそのまま渡す。

意味: PyTorch と同じく `out = floor(in as f64 * s)`。検査は fail-closed:
長さの不一致・`s` が非有限／`<=0`・`out==0`・`usize` への変換 overflow
（`out_f >= usize::MAX as f64` を `as usize` 変換の**前に**検査。`as` は
範囲外を無言で飽和させるため）をそれぞれ拒否する。

座標系: 導出した `size` を使って通常の interpolate を行う。これは
PyTorch の `recompute_scale_factor=True` と同じ意味になる。既定
（`None`／`False`）では座標計算に `1/scale_factor` をそのまま使うので、
`in*s` が整数でない場合は結果がわずかに異なる（整数倍のときは一致する。
この差分は明記のうえ対象外とした）。

`ScaleFactorError`: `tensor-core::interpolate` モジュール内に新規定義
（`Debug`/`Display`/`std::error::Error`、`#[non_exhaustive]`）。facade が
再エクスポートしている `ShapeError` に variant を足すと公開面が広がる
ため、それは避けた。

## 4. VJP 実装形の整理（計画からの妥当な簡略化）

計画では `Area` の VJP も scatter_add ベース（可変長を連結した 1 次元
index／src）で構成する案だったが、実装時に `Op::AdaptiveAvgPool2d` の
VJP（`adaptive_avg_pool2d_vjp`）がまったく同じ設計課題（可変 tap 数）を
「稠密配列への直接 `f64` 加算」で解決している既存先例を発見したため、
`interpolate_area_vjp` も同じ方式を採用した。理由:

- 可変長 index／src の scatter_add 平坦化は、行内の各出力位置の
  オフセットを別途管理する必要があり実装・レビューコストが高い
- 稠密配列直接加算は本クレート内で既に受け入れられている設計
  （`adaptive_avg_pool2d_vjp`）であり、正しさの検証（テスト・レビュー）
  が容易
- backend 側 GPU scatter オフロードを失う点は `adaptive_avg_pool2d_vjp`
  も同じトレードオフを持ち、Area の使用頻度（ダウンサンプリング用途が
  中心）を踏まえ許容できると判断した

`NearestExact`／`Linear`／`Trilinear`／`Bicubic` は固定 K タップのため
`Bilinear` と同型の scatter_add 平坦化を維持し、共通ヘルパー
`grad::interpolate_fixed_tap_scatter_vjp`（`k` を引数化）へ切り出して
重複を除いた。

## 5. 数値契約のまとめ

| モード | forward 3 バックエンド契約 | CPU ネイティブ ⟷ ホスト参照 |
|---|---|---|
| `Nearest`／`NearestExact` | bit 完全一致（算術なし） | bit 完全一致 |
| `Bilinear`／`Linear`／`Trilinear`／`Bicubic` | REQ-2 統一複合判定 | bit 完全一致（同一 Rust 関数） |
| `Area` | REQ-2 統一複合判定 | bit 完全一致（`adaptive_avg_pool2d` とも一致） |

いずれも既存の tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_
THRESHOLD`）・baseline・ガードレール閾値は変更していない。

## 6. 対象外（decision doc・PR に明記）

- GPU 専用カーネル（CUDA／Metal の新モード。`Unsupported` によるホスト
  フォールバックのまま。実機 parity は Mac／GB10 セッションへ申し送り。
  `docs/perf/logs/interpolate-modes-2152/README.md`）
- `antialias`
- `recompute_scale_factor=False`（既定）相当の「与えられた scale を
  座標に使う」座標系
- `Var` への `interpolate_scale_factor` のようなメソッド追加と facade の
  新公開面
- TF 固有の `lanczos3/5`・`gaussian`・`mitchellcubic`

## 7. 承認事項

なし。`#[non_exhaustive]` の `InterpolateMode` への variant 追加は
非破壊で、tensor-core への純関数の追加は facade 公開面に影響しない。
依存・tolerance・baseline・ガードレール閾値・spec はいずれも変更して
いない。

## 8. 変更ファイル一覧

- `crates/tensor-core/src/interpolate.rs`: 座標・重み・ブレンド関数
  （`nearest_exact_src_coord`／`linear_blend`／`trilinear_blend`／
  `bicubic_src_taps`／`bicubic_blend`）・`ScaleFactorError`／
  `interpolate_size_from_scale_factor`
- `crates/tensor-core/src/backend_ops.rs`: `InterpolateMode` の 5
  variant 追加
- `crates/tensor-core/src/ops_shape.rs`: `interpolate_out_shape_for_mode`
  の rank 検査を mode 別に拡張
- `crates/tensor-core/src/lib.rs`: 新規 pub 項目の再エクスポート
- `crates/autodiff/src/eval.rs`: ホスト参照実装 5 関数
- `crates/autodiff/src/grad.rs`: `interpolate_with_fallback` 分岐追加・
  VJP 5 モード追加・共通ヘルパー
  （`interpolate_fixed_tap_scatter_vjp`／`interpolate_area_vjp`）
- `crates/autodiff/src/var.rs`・`tape.rs`: doc 更新
- `crates/backend-cpu/src/interpolate.rs`・`ops.rs`: CPU ネイティブ実装
  5 関数・許可リスト拡張
- `crates/autodiff/tests/interpolate_parity.rs`（新規）
- `crates/facade/tests/interpolate_backend_parity.rs`（追記）
- `crates/backend-cpu/src/interpolate.rs`（`mod tests` 追記）
- `crates/tensor-core/src/interpolate.rs`・`ops_shape.rs`（`mod tests`
  追記）
