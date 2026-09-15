# Pooling（Max／Avg／AdaptiveAvg）実装方式・VJP 設計

> 注記: 本 doc のファイル名は意図的に `pooling-ops` とし `pool` 単独を避けた。
> リポジトリ内には既に `docs/memory-pool-design.md`・
> `docs/device-memory-pool-design.md`・
> `docs/backend-cuda-pool-allocator-decision.md`・
> `docs/backend-metal-buffer-pool-decision.md`・各クレートの `pool.rs`
> （メモリのサイズクラス別プールアロケータ）が存在し、これらと本 doc
> （Pooling 演算＝空間縮約 NN レイヤー）は無関係な別概念である。混同を
> 避けるため冒頭で明示する。

## 0. 前提・スコープ

本 issue（#1727）は **設計 doc の作成のみ**で、コード変更を含まない。実装は
後続 3 issue（親 #1607 配下）へ引き継ぐ:

- #1728: `backend-cpu` 実装
- #1729: `backend-cuda` 実装
- #1730: `backend-metal` 実装

対象は `docs/compat-api-scope.md` §1.2「Pooling | #1607」・
`docs/compat-feature-gap.md` §2.7「Pooling（Max/AvgPool）」（難度 L・
「なし」）が指す欠落機能で、CNN 到達に必要な Tier 1 部品である。

### 0.1 依存関係の扱い

- 親 #1606（Conv）・#1641（Conv 実装方式設計）は本 doc 作成時点で OPEN。
  Pooling は im2col／直接畳み込みのどちらを採るかという判断には依存しない
  （縮約窓の走査のみで畳み込み積和を行わない）ため、#1641 の結論を本 doc の
  着手ブロッカーとはしない。ただし NCHW レイアウト・`stride`／`padding`／
  `dilation` の引数命名規約は #1641 が確定する共有規約と整合させるべきであり、
  #1728 着手前に #1641 の結論との齟齬がないか確認することを §12 に承認事項
  として記録する。
- facade 公開面の拡張手続き（`docs/compat-api-scope.md` §5）はユーザー承認が
  必要だが、本 issue はコード変更を伴わないため対象外。承認が必要な事項は
  §12 に分離して記録する。

## 1. コード事実（根拠。HEAD `1a1bcd5a57d19e226b22c33d47bb8b3df51779da` 時点）

後続実装が再利用・踏襲するパターンを、実コードの参照箇所とともに列挙する。

- **先勝ち決定的 VJP**: `crates/autodiff/src/grad.rs:3423`
  `extremum_first_match_vjp`（旧 `max_vjp`。イシュー #1720 で `Op::Max`／
  `Op::Min` 共有ヘルパへ改称）。`out_value` と `==` 一致する縮約軸上の最初の
  要素へ上流勾配を置く。`docs/compat-api-scope.md` の記録（#1720 追補）にも
  ある通り、`fandhe_ai::compat` に `amax` 相当の公開 API が無い現状では均等
  分配（PyTorch の複数最大値分配）との互換要求が生じないため先勝ちを採用して
  いる。MaxPool の VJP もこのタイ規則を踏襲する（§6）。
- **`ScatterReduce::Add` の決定的集約契約**: `crates/tensor-core/src/
  backend_ops.rs:200`（`enum ScatterReduce`）。`Add` バリアントは
  「row-major 走査順・`f64` アキュムレータでの逐次加算・最後に 1 回だけ
  `f32` へ downcast」という決定的契約を持つ（同ファイル `Overwrite`／`Add`
  の doc comment）。3 バックエンド実装:
  - CPU: `crates/backend-cpu/src/gather_scatter.rs:166`（`pub fn scatter`）。
    単一スレッド逐次ループでこの契約を実装（同ファイル doc comment。
    並列化は将来の性能最適化 issue のスコープ）。
  - CUDA: `crates/backend-cuda/src/kernels_gather_scatter.rs`。
  - Metal: `crates/backend-metal/src/gather_scatter.rs`・
    `crates/backend-metal/src/shaders/gather_scatter.metal`。
    `scatter_add` は binary64 逐次加算の 64bit 整数ソフトウェア
    エミュレーションで CPU と bit 完全一致（`docs/compat-feature-gap.md`
    #1778 追補）。
- **非追跡 index payload を `Op` に埋め込み VJP は scatter へ委譲するパターン**:
  `crates/autodiff/src/var.rs:2747`（`pub fn topk`）・`Op::Topk`（`tape.rs`）。
  MaxPool の索引（`Tensor<i32>`）も同型で `Op` payload に保持し、`Var` の
  公開戻り値としても返す設計とする（§5）。
- **出力 shape 関数群**: `crates/tensor-core/src/ops_shape.rs`。
  `scatter_out_shape`（458 行目。出力 shape は常に `input_shape` で
  `dim` 軸長を無制約とする）・`topk_out_shape`（523 行目）。この無制約性に
  より `k = Hout·Wout ≠ H·W` でも scatter が成立する（§6 の根拠）。
- **poison 伝播の網羅 match**: `crates/autodiff/src/tape.rs:1063`
  （`fn for_each_input`）。新設 `Op::MaxPool2d`／`Op::AvgPool2d`／
  `Op::AdaptiveAvgPool2d` はこの match へ入力ノードを登録する
  （`docs/autodiff-checkpoint-design.md` §3.5.1 の poison 伝播規約）。
- **`f64` 相当の長軸縮約契約**: `.claude/rules/coding-rust.md`「正規化統計・
  勾配の長軸縮約」節。Metal 実装形の到達点は
  `crates/backend-metal/src/soft_f64.rs`（binary64 逐次加算の 64bit 整数
  ソフトウェアエミュレーション。bias 勾配 #1659／#1666 で確立）。AvgPool の
  縮約もこの方針へ揃える（§7）。
- **`Module` trait**: `crates/autodiff/src/nn/module.rs:38`
  （`pub trait Module`）。`forward`／`forward_host` の 2 経路を持つ既存規約
  （`crates/autodiff/src/nn/` 配下: `activation.rs`・`linear.rs`・
  `loss.rs`・`norm.rs`・`rnn.rs` 等と同型）。
- **`BackendOps` trait**: `crates/tensor-core/src/backend_ops.rs:371`
  （`pub trait BackendOps`）。既定 `Unsupported` を返す非破壊拡張パターン
  （`rmsnorm`／`layer_norm`／`gather`／`scatter`／`scalar_unary` 等と同型）。

## 2. 対象演算・レイアウト

- 対象: `MaxPool1d`／`MaxPool2d`／`AvgPool1d`／`AvgPool2d`／
  `AdaptiveAvgPool1d`／`AdaptiveAvgPool2d`（PyTorch `nn.MaxPool2d` 等相当）。
- レイアウト: **NCHW（2d: rank 4 `[N,C,H,W]`）／NCL（1d: rank 3 `[N,C,L]`）
  固定**。channels_last は対象外（§11）。
- **1d は 2d へ併合する**: `[N,C,L]` を `Var::reshape`（view 系 Op・中間
  バッファ非確保・bit 同一）で `[N,C,1,L]` へ持ち上げ、2d カーネル 1 系統で
  処理してから戻す。カーネル・`BackendOps` メソッドは 2d のみ新設する
  （3 バックエンド × 3 演算 = 9 カーネルに抑える）。1d の `kernel_size`／
  `stride`／`padding`／`dilation` は H 軸を `(1, 1, 0, 1)` として展開する。

## 3. パラメータと検査（A03 入力検証）

パラメータ: `kernel_size`・`stride`（既定 `= kernel_size`）・`padding`・
`dilation`（Max のみ。Avg は 1 固定）・`ceil_mode`・`count_include_pad`
（Avg のみ・既定 `true`）。`divisor_override` は対象外（§11）。

検査規則（`Var` 入口で `AutodiffError::InvalidArgument`／`Shape`。
`BackendOps` 実装側でも fail-closed に再検査し判定迂回経路を作らない）:

- rank 一致（2d は 4、1d 併合後も 4）。
- `kernel ≥ 1`・`stride ≥ 1`・`dilation ≥ 1`。
- `padding ≤ floor(kernel / 2)`（**dilation に依存しない**）。PyTorch は
  2 段の独立した検査を通過して初めて構成を受理する（`aten/src/ATen/
  native/PoolingChecks.h::pool2d_shape_check` の
  `TORCH_CHECK(kW/2 >= padW && kH/2 >= padH, "pad should be smaller than
  or equal to half of kernel size", …)`〈dilation 非依存〉と、`aten/src/
  ATen/native/Pool.h::pooling_output_shape` の `TORCH_CHECK(pad <=
  effective_kernel_size(kernelSize, dilation) / 2, …)`〈`
  effective_kernel_size(k, d) = (k − 1) · d + 1`。dilation 依存〉の両方。
  いずれも curl で直接取得した `pytorch/pytorch` main ブランチのソース
  〈2026-09-14 時点〉で確認済み）。`effective_kernel_size(k, d) ≥ k`
  （`d ≥ 1` のとき等号は `d=1`）が常に成り立つため
  `floor(effective_kernel_size(k,d)/2) ≥ floor(k/2)` となり、**`kW/2 ≥
  padW` 側が常により厳しい（binding な）制約になる**。結果として両検査
  の積集合は dilation に依存しない `padding ≤ floor(kernel/2)` に帰着
  する。`((kernel − 1) · dilation + 1) / 2`（`effective_kernel_size`
  のみを見た式）は `d > 1` のとき本来の上限より緩く、誤って許可してしまう
  （下記の境界例を参照）。`(kernel − 1) · dilation / 2`（dilation 依存の
  単純な floor 式）も `d=1` のとき `floor((k−1)/2) ≠ floor(k/2)` となる
  ため誤って拒否しうる（例: `kernel=2, dilation=1` で `(2−1)·1/2=0` だが
  真の上限は `floor(2/2)=1`）。Max／Avg 双方の検査に `floor(kernel/2)`
  を適用する（Avg は `dilation=1` 固定のため、この式は Avg の実質的な
  上限とも一致する）。境界値の受入例:
  - `kernel=2, dilation=1, padding=1` → 上限 `floor(2/2)=1` で許可
    （`padding=2` は上限超過のため拒否）。
  - `kernel=3, dilation=2, padding=1` → 上限 `floor(3/2)=1` で許可。
    `padding=2` は `effective_kernel_size` 側の検査（上限
    `floor(5/2)=2`）だけを見ると許可されるように見えるが、`kW/2 ≥ padW`
    側の検査（上限 `floor(3/2)=1`）で拒否される。dilation を増やしても
    `floor(kernel/2)` 自体は変わらないため、この構成は拒否が正しい。
- 出力長 `≥ 1`。
- 要素数積は `checked_mul` で計算する（`reduction::max` の
  `ElementCountOverflow` パターンを踏襲）。
- MaxPool の索引値域は `H · W ≤ i32::MAX`（`Tensor<i32>` 索引契約。
  `Op::Topk` と同じ制約）。
- adaptive の `output_size ≥ 1`。`output_size ≤ 入力長` は要求しない
  （PyTorch 準拠。窓は §4 の式で自然に定義され拡大側も成立する）。
- **空間軸（`H`／`W`。1d 併合後は `H=1` 固定側を除く `W` 軸）は
  `≥ 1` を要求し、`0` を `ShapeError` で拒否する。Max／Avg／Adaptive の
  全 pooling 種別へ一律適用する**（`out_h`／`out_w` を計算する前段で
  rank 検査の直後に置く）。PyTorch
  `aten/src/ATen/native/AdaptiveAveragePooling.cpp` の
  `for (const auto i : {-2, -1}) { TORCH_CHECK(input.size(i) > 0, "…
  Expected input to have non-zero size for non-batch dimensions …"); }`
  を出典として確認済み（`H`／`W`〈末尾 2 軸〉のみを検査し `C` は対象外。
  2026-09-14 時点の `pytorch/pytorch` main ブランチのソースを参照）。
  **バッチ軸 `N=0`・チャンネル軸 `C=0` は本検査の対象外**（空バッチ・空
  チャンネルは出力も空になるだけで 0 除算を起こさないため拒否しない。
  区別する理由は、`N`／`C` は縮約対象ではなく縮約窓の外側の走査軸である
  のに対し、`H`／`W` は adaptive の `divisor = end − start` の直接の
  入力である点にある）。
  - adaptive 側の動機: `start = floor(o * in / out)`・
    `end = ceil((o + 1) * in / out)`（§4）は `in = 0` のとき常に
    `start = end = 0` となり `divisor = 0` の 0 除算（`f64` 昇格後の
    `0.0 / 0.0` で `NaN`。forward・VJP 双方で発生しうる）を招く。
  - **非 adaptive 側は `padding > 0` のとき §4 の負分子ゲートだけでは
    不十分**であり本検査が必須: 例えば `in=0, k=2, s=2, p=1, d=1`
    （`padding=1` は §3 の上限 `floor(2/2)=1` 以下のため許可される
    構成）では分子 `0 + 2·1 − 1·(2−1) − 1 = 0` となり負分子
    ゲートを素通りして `pool_out_len = 0+1 = 1` を返してしまう
    （padding のみで構成された窓を「有効な出力」として通過させる誤り。
    `H=0`／`W=0` かつ `padding=0` の場合は従来どおり負分子ゲートでも
    拒否できるが、本検査を一律適用することで `padding` の値に依存しない
    単一の判定にする）。
- **`dilation` による空窓（有効入力を含まない窓）を `ShapeError` で拒否
  する**: `H ≥ 1` かつ `padding ≤ floor(kernel/2)`（上記 2 検査を通過
  済み）でも、`kernel=2, dilation ≥ 2` かつ `H = dilation − 1` の構成
  （必然的に `out_len = 1` の単一窓）では、窓のタップ位置が
  `−1` と `H`（= `dilation − 1`）となり、両方とも走査範囲外
  （左右の padding／範囲外）で有効入力を一切含まない。例:
  `in=1, kernel=2, stride=1, padding=1, dilation=2`（タップ `−1, 1`）・
  `in=1, kernel=2, stride=2, padding=1, dilation=2`（同じくタップ
  `−1, 1`）。導出（証明のスケッチ）: 最初のタップ `o·s − p` が
  `≥ 0` なら `padding ≤ floor(kernel/2)` の下で必ず `< H` になり空窓は
  起きない（`o·s − p ≥ H` は `(out_len−1)·s ≥ H + p` を要求し §4 の
  出力長式と矛盾する）。最初のタップが `< 0` なら、非負になる最小の
  タップは `padding ≤ (kernel−1)·dilation` の下で `[0, dilation−1]`
  に収まるため、`dilation ≤ H` であれば常に `< H` で有効。したがって
  空窓が起こりうるのは `dilation > H` の場合に限られ、上記 2 検査の下で
  唯一到達可能な構成は `kernel=2`（`padding` の上限が `1` に固定される
  ため）かつ `H = dilation − 1`（`H < dilation` かつ `H ≥ dilation − 1`
  を要求する境界）のみである。よって本検査は
  **`kernel = 2` かつ `dilation > H` の構成のみを対象**とし
  （`kernel = 1` は `dilation` が実質無効なため対象外・`kernel ≥ 3` は
  上記導出により空窓が起こり得ない）、これを `ShapeError` で拒否する。
  Avg は `dilation = 1` 固定のため本検査は実質的に発火しない。**PyTorch
  との意図的な非互換**: PyTorch の実カーネル（`hstart` を `dilation`
  刻みで非負へ整列してから走査する実装）はこの構成でも空窓のまま
  `hstart ≥ hend` の縮退した走査区間を許容し、AvgPool
  `count_include_pad=false` では 0 除算・MaxPool では未定義の勝者
  索引を生じさせる（2026-09-14 時点の `pytorch/pytorch` main ソース
  読解による確認）。本設計は §5 の「索引は常に有効」契約・§6 の
  `scatter_add` VJP 契約を維持するため、この縮退構成を入口で拒否する
  （sentinel 索引・無勾配 VJP 等の特殊扱いは §5／§6 の契約を複雑化する
  ため採らない）。
- `ceil_mode`: **v1 は `false` のみサポート**。`true` を渡した場合は
  `InvalidArgument` で拒否する。PyTorch の「最後の窓は入力または左
  padding 内で始まらなければならない」規則は将来対応の参考として §11 へ
  記録し、後続 issue（`.claude/rules/out-of-scope-tracking.md`）へ引き継ぐ。

## 4. 出力 shape 関数（`tensor-core::ops_shape` へ新設予定。設計のみ）

```text
pool_out_len(in, k, s, p, d) = floor((in + 2p − d(k−1) − 1) / s) + 1
```

除算は明示的に **floor**（数学的な意味での floor）と契約する。Rust の
符号付き整数 `/` は **ゼロ方向丸め**であり、分子が負のとき floor と
一致しない。例えば `in=1, k=2, s=2, p=0, d=1`（`padding=0` は §3 の
上限 `floor(2/2)=1` 以下のため許可される構成）では分子
`1 + 0 − 1·(2−1) − 1 = −1` となり、floor 除算では `floor(−1/2) = −1` で
出力長 `−1+1 = 0` となり `ShapeError` で正しく拒否されるべきところ、Rust
の `/` をそのまま使うと `−1 / 2 = 0`（ゼロ方向丸め）となり出力長
`0+1 = 1` が誤って通過してしまう。**実装契約**: 分子（`in + 2p −
d(k−1) − 1`）を `checked_sub` 等で計算したうえで **負の場合は floor
除算を行わず直ちに `ShapeError`** とする（非負の分子同士では floor と
ゼロ方向丸めが一致するため、分子が非負であることを確認した後は通常の
整数除算でよい）。この負分子拒否ゲートを floor 契約の実装手段として
必須とする。結果（`+1` 後）が 0 以下になる場合も同様に `ShapeError`。

- `pool2d_out_shape(shape, kernel, stride, padding, dilation)`
- `adaptive_pool2d_out_shape(shape, output_size)`

adaptive の窓（PyTorch と同式・整数演算のみで決定的）:

```text
start = floor(o * in / out)
end   = ceil((o + 1) * in / out)
```

padding は暗黙（値として保持されず、MaxPool の勝者索引にもなり得ない）。
padding 領域を除外する境界検査はカーネル側で必ず手動実施する（REQ-8。
性能最適化を理由に省略しない）。

## 5. MaxPool forward と索引契約

戻り値は `(values, indices: Tensor<i32>)`。索引は **(n, c) 平面内の
flat 添字 `h · W + w`**（PyTorch `return_indices=True` と同じ意味論）。
`Var::max_pool2d` は `Var::topk`（`var.rs:2747`）と同様に索引を公開戻り値
として返す（決定。索引公開は後続の MaxUnpool 実装の前提になるため）。

**タイ規則（決定）**: 窓内を row-major（`kh` 外側・`kw` 内側）で走査し
`v > best` のときのみ更新する。同値は最初の位置が勝つ（先勝ち決定的）。
`extremum_first_match_vjp`（`grad.rs:3423`）・`eval::argmax` と同一規約であり、
#1718（amax／amin 勾配分配方式）が均等分配へ確定しても、MaxPool は索引経路
（PyTorch `max_pool2d` も索引経路）であるため影響を受けない。

**NaN 規則（決定）**: 既存 `BackendOps::max`（`f32::max`／`fmaxf`。NaN 非
伝播）と `eval::max`（NaN 伝播）の不一致が既存コードに存在する事実を踏まえ、
MaxPool は **NaN 伝播（PyTorch `max_pool2d` 準拠）・索引は走査順で最初に
現れた NaN の位置**とする。更新条件:

```text
v > best || (v.is_nan() && !best.is_nan())
```

PyTorch は最後の NaN の索引を返すため索引のみ差異が生じる（値は一致・
NaN 入力は parity 判定対象外とする）。これは tolerance／baseline の変更
ではなく意味論の決定であり、3 バックエンドで同一規則を実装すれば bit 完全
一致が成立する。

**数値契約**: 純粋な選択演算（丸めなし）のため **3 バックエンド bit 完全
一致（値・索引とも）** を受入基準とする。REQ-2 統一複合判定は使わない。

## 6. MaxPool VJP（argmax 経路。新規 backward カーネル不要）

`Op::MaxPool2d { input: NodeId, index: Tensor<i32> }`（`Op::Topk` と同型:
非追跡 payload・`params` は VJP に不要のため保持しない）。

VJP 手順:

1. `input` を `[N·C, H·W]`、`upstream`／`index` を `[N·C, Hout·Wout]` へ
   reshape する。
2. `d_input = scatter_add(zeros_like(input_flat), dim=1, index, upstream)`。
   `scatter_out_shape`（`ops_shape.rs:458`）は `dim` 軸長を無制約とするため
   `Hout·Wout ≠ H·W` でも成立する（`Op::Topk` VJP と同じ根拠）。
3. `[N, C, H, W]` へ reshape して返す。

**重なり窓（`stride < kernel`）** では同一入力位置が複数の窓の勝者になり
うる。これは scatter_add の重複添字ケースそのものであり、
`ScatterReduce::Add` の決定的集約契約（row-major 走査順・`f64` アキュムレータ・
1 回 downcast）に従うことで 3 バックエンド bit 一致が成立する（GPU 側は
`BackendOps::scatter` 実装済み・`Unsupported` 時はホスト `eval::scatter` へ
フォールバック）。

padding 位置は §5 の索引契約上勝者になり得ないため索引は常に `[0, H·W)`
の範囲に収まる（forward 側で保証）。VJP 側でも `debug_assert!` と安全側
スキップ（境界外索引を無視する）を設ける。この保証は §3 の
`dilation` による空窓拒否検査（`kernel=2` かつ `dilation > H` の構成を
`ShapeError` で拒否）が「すべての窓が少なくとも 1 つの有効入力を含む」
ことを前提として成立している。

## 7. AvgPool／AdaptiveAvgPool forward

```text
out = (Σ_{窓内（padding 除く）} x) / divisor
```

`divisor` は `count_include_pad = true` なら `kh · kw`（padding 込み）、
`false` なら有効要素数。adaptive は可変窓で
`divisor = (end_h − start_h) · (end_w − start_w)`（§4 の窓式）。

**縮約精度契約（決定）**: `AdaptiveAvgPool(output_size=1)` は global average
pooling＝長軸縮約に該当するため、`.claude/rules/coding-rust.md` の `f64`
アキュムレータ方針を **Avg 系全体へ一律適用**する: 窓内を row-major 固定順で
`f64` に昇格して逐次加算 → `f64` で除算 → 最後に 1 回だけ `f32` へ downcast
する。CUDA は `double`、Metal は `soft_f64`（`crates/backend-metal/src/
soft_f64.rs`。bias 勾配 #1659 で確立済みの既存手段）で実装し、ホスト参照
実装と **bit 完全一致**を契約とする（REQ-2 複合判定は使わない。tolerance
は不変）。

代替案（f32 のみの逐次和・Neumaier 補償和）は棄却する: 相殺列でホストと
一致せず、`docs/metal-grad-reduction-parity-judgment-decision.md` で記録
された Tier A/B 契約案（事前判定可能な誤差上界方式）が撤回された経緯と
同じ理由による。

## 8. AvgPool／AdaptiveAvgPool VJP

```text
d_input[n,c,h,w] = Σ_{(oh,ow) ∋ (h,w)} upstream[n,c,oh,ow] / divisor(oh,ow)
```

実装形: **出力定常**（入力位置ごとに、それを含む出力窓を `(oh, ow)`
row-major 順に走査）で `f64` 逐次加算・1 回 downcast する（Metal の scatter
実装が採る順序等価性の考え方と同じ）。重なり窓の加算順を固定することで
3 バックエンド bit 一致が成立する。

`Op::AvgPool2d { input, kernel, stride, padding, count_include_pad }`・
`Op::AdaptiveAvgPool2d { input }`（`output_size` は出力ノード shape から
導出し非保持とする）。

v1 の VJP は **ホスト側のみ**（`crates/autodiff/src/grad.rs`。`cumsum`
（#1731）と同方針）とする。GPU backward カーネルはスコープ外として §11 へ
記録する。

## 9. `BackendOps`／`Op`／`Var`／`nn` 配線案（設計のみ。実装は #1728〜）

- `BackendOps`（`crates/tensor-core/src/backend_ops.rs`）へ既定 `Unsupported`
  の非破壊拡張として 3 メソッドを追加する:
  - `max_pool2d(x, &Pool2dParams) -> Result<(Tensor<f32>, Tensor<i32>), _>`
  - `avg_pool2d(x, &Pool2dParams, count_include_pad) -> Result<Tensor<f32>, _>`
  - `adaptive_avg_pool2d(x, [usize; 2]) -> Result<Tensor<f32>, _>`

  `Pool2dParams`（`#[non_exhaustive]`・コンストラクタ経由。`ScatterReduce`
  と同方針で公開 API 非破壊を保つ）は tensor-core 内の型とし facade へは
  再エクスポートしない。
- `Var` メソッドはプリミティブ引数（`[usize; 2]`・`Option<[usize; 2]>`・
  `bool`）で受け、facade へ新規型を露出しない（facade 到達経路は既存
  `Var` 再エクスポート経由・新規 `pub use`／`pub fn` を facade クレートへ
  追加しない）。`Unsupported` のときのみ `eval::*` ホスト参照実装へ
  フォールバックし、他エラーは伝播する（判定迂回経路を作らない。A08）。
- `Op` は非融合・`push_eager` で常時実体化し、`is_checkpoint_eligible =
  false` とする。`Op::for_each_input`（`tape.rs:1063`。網羅 match）へ入力を
  登録し poison 伝播（`docs/autodiff-checkpoint-design.md` §3.5.1）を維持
  する。
- `nn::{MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool1d,
  AdaptiveAvgPool2d}`（`crates/autodiff/src/nn/pooling.rs` 新設予定）。
  `Module::forward` に加え `forward_host` をオーバーライドし、
  `BackendOps` 直呼び＋`Unsupported` フォールバックで `forward` と bit
  一致させる（既存 `nn::linear`／`nn::norm` 等と同型）。
- 後続 issue で新設するバックエンド側モジュール名は既存の「メモリプール」
  命名（`pool.rs` 系）と衝突しないよう次を指定する:
  - `crates/backend-cpu/src/pooling.rs`
  - `crates/backend-cuda/src/{pooling.rs, kernels_pooling.rs}`
  - `crates/backend-metal/src/{pooling.rs, shaders/pooling.metal}`

## 10. バックエンド別実装形（#1728〜#1730 への指針）

- **CPU**: 参照実装（単一スレッド逐次。`gather_scatter.rs` の規律を踏襲）。
  並列化は out-of-scope として追跡する。
- **CUDA**: 1 スレッド＝1 出力位置。NVRTC 静的文字列・座標演算は
  `long long`・padding 境界の手動検査（REQ-8）。Avg は `double` 累積。
- **Metal**: 1 スレッド＝1 出力位置。Avg は `soft_f64`。REQ-8 境界検査を
  維持する。
- 3 バックエンドとも forward 出力（Max: 値＋索引／Avg: 値）は CPU 参照実装
  と **bit 完全一致**が受入基準。実機（DGX Spark GB10／Apple M4 Max）
  依存テストは `#[ignore]` 分離とし、未実測の場合は各実装 issue の doc に
  記入欄を残す。

## 11. スコープ外（`.claude/rules/out-of-scope-tracking.md` 対象）

- MaxUnpool・LPPool・FractionalMaxPool・AdaptiveMaxPool
- 3d 版（`MaxPool3d` 等）
- channels_last レイアウト
- `ceil_mode = true`
- `divisor_override`
- GPU backward カーネル（v1 はホスト VJP のみ）
- CPU 並列化
- ONNX export マッピング（`docs/onnx-export-op-mapping.md`）
- facade `compat::Sequential::add_max_pool2d` 等の統合ヘルパー

## 12. 承認事項（#1728 着手前の前提）

- **facade 公開面拡張**: `compat::Sequential::add_max_pool2d` 等
  （`docs/compat-api-scope.md` §5 経路 2）はユーザー承認が未取得のため、
  本 doc の対象外（§11）のまま承認事項として記録する。`#1714` の親 issue
  コメントでの承認が先例となる。
- **`nn::*Pool*` の追加自体**: 内部クレート（`fandhe_ai_autodiff::nn`）への
  追加であり承認は不要。ただし facade からの到達は既存 `Var` 再エクスポート
  経由に限り、facade クレートへの新規 `pub` 追加は行わない。
- **tolerance／baseline は不変**。変更が必要になった場合は別途承認を要する。
- **#1641 との整合確認**: #1641（`docs/conv-ops-design.md` §2／§0.2）で
  NCHW／NCL・1d→2d 併合・引数命名（`kernel_size`／`stride`／`padding`／
  `dilation`）が本 doc と同一規約で確定した（整合確認の保留を解消）。
  `ceil_mode` は Conv に存在せず齟齬なし。`padding` の許容範囲は Conv が
  意図的に上限なし（Pooling の `padding ≤ floor(kernel/2)` とは異なる）
  であることも確認済み。

## 13. #1728〜#1730 への受入テスト一覧

- shape／引数検査: 境界値（`kernel=1`・`stride=1`・`padding` 上限）・
  overflow（`checked_mul`）・`ceil_mode=true` 拒否・`padding` 上限超過の
  拒否。
- `padding` 上限式（§3）の境界例: `kernel=2, dilation=1, padding=1`
  （許可）／`padding=2`（拒否）・`kernel=3, dilation=2, padding=1`
  （許可）／`padding=2`（拒否。`effective_kernel_size` 側の検査だけでは
  許可されてしまうため `kW/2 ≥ padW` 側の検査が効いていることを検証する
  回帰）を固定する。
- 出力長 floor 除算（§4）の負分子拒否: `in=1, k=2, s=2, p=0, d=1` が
  `ShapeError`（出力長 0）として拒否され、ゼロ方向丸めによる誤った出力長
  1 を返さないことを確認する。
- 空間軸ゼロ長拒否（§3。Max／Avg／Adaptive 一律）: `[N,C,0,W]`／
  `[N,C,H,0]` が `ShapeError` で拒否されること・`[0,C,H,W]`（空バッチ）
  は拒否されず出力形状 `[0,C,out_h,out_w]` が得られることを区別して
  確認する。非 adaptive 側は `padding=0`（負分子ゲート経由）と
  `padding>0`（`in=0, k=2, s=2, p=1, d=1` のように負分子ゲートを素通り
  しうる構成。§3）の両方を回帰テストとして固定する。
- `dilation` による空窓拒否（§3）: `in=1, kernel=2, stride=1, padding=1,
  dilation=2` および `in=1, kernel=2, stride=2, padding=1, dilation=2`
  が `ShapeError` で拒否されること・境界の反例として
  `in=2, kernel=2, stride=1, padding=1, dilation=2`（`H = dilation` で
  検査対象外）は許可され、索引が `[0, H·W)` の範囲に収まることを確認
  する。
- タイ先勝ち: 全要素同値の窓で索引が窓先頭になること。
- NaN 伝播と最初の NaN 索引・padding 非勝者（索引が常に `[0, H·W)`）。
- 重なり窓（`stride < kernel`）の重複添字 VJP が `scatter_add` 契約と一致
  すること（手計算値と bit 一致）。
- 数値微分突合: Avg は全形状、Max はタイのない入力で実施。
- 1d と `[N, C, 1, L]` 2d の bit 一致。
- `forward` と `forward_host` の bit 一致。
- `BackendOps` 実装と `eval` フォールバックの bit 一致。
- GPU: CPU との bit 一致（値・索引）を `#[ignore]` 実機テストで確認。
  REQ-8 境界検査の静的テスト（既存 `kernels_*.rs` の書式に倣う）。

## 14. 出典

- 親 #1607・依存 #1606／#1641・関連 #1718／#1720。
- `docs/compat-api-scope.md` §1.2「Pooling」・
  `docs/compat-feature-gap.md` §2.7「Pooling（Max/AvgPool）」。
- `docs/norm-ops-design.md`（#1596）・
  `docs/autodiff-rnn-cell-tape-design.md`（#1646）・
  `docs/backend-dtype-dispatch-design.md`（#1648）: 設計 doc のみで実装を
  伴わない先行例。
- `docs/autodiff-checkpoint-design.md` §3.5.1: poison 伝播規約。
- `docs/metal-grad-reduction-parity-judgment-decision.md`・
  `.claude/rules/coding-rust.md`「正規化統計・勾配の長軸縮約」節: `f64`
  相当の縮約契約・Metal 実装形の到達点。
- `crates/autodiff/src/grad.rs:3423`（`extremum_first_match_vjp`）・
  `crates/tensor-core/src/backend_ops.rs:200`（`ScatterReduce`）・
  `crates/autodiff/src/var.rs:2747`（`topk`）・
  `crates/tensor-core/src/ops_shape.rs:458,523`
  （`scatter_out_shape`／`topk_out_shape`）・
  `crates/autodiff/src/tape.rs:1063`（`for_each_input`）。

## 15. 実装記録（記入欄）

本 issue（#1727）はコード変更を含まない設計 doc のみ。実装記録は
#1728（CPU）・#1729（CUDA）・#1730（Metal）の各 issue で本節へ追記する
（または各 issue 側の doc へ記録し本節から forward pointer を張る）。

### Metal 実装（イシュー #1730）

**数値方式**: Max は選択演算のみ（算術を含まない）のため CPU 参照実装
と bit 完全一致。Avg／Adaptive は窓内の有効タップを row-major に
soft-f64（binary64 の 64bit 整数ソフトウェアエミュレーション）
アキュムレータへ逐次加算し `divisor`（`u32` から厳密変換した binary64
値。`pool_f64_from_uint`）で soft-f64 除算してから 1 回だけ `f32` へ
narrow する。加算・除算とも正しく丸められた binary64 のためハード
ウェア `f64` と一致し、CPU `f64` 参照実装と bit 完全一致する設計
（BatchNorm・LayerNorm のような mul／rsqrt を伴う二重丸め問題が
Pooling には存在しないため、REQ-2 統一複合判定ではなく bit 完全一致
を採用）。

**カーネル構成**: `crates/backend-metal/src/shaders/pooling.metal`
（`max_pool2d_f32`／`avg_pool2d_f32`／`adaptive_avg_pool2d_f32` の 3
カーネル。1 スレッド = 1 出力位置・`PoolDims` 構造体を `setBytes` で
渡す）。soft-f64 プリミティブ（`pool_f64_{clz64, widen, add, narrow,
mul64_wide, normalize_mantissa, div64_wide, div}`）は
`batch_norm.metal::bn_f64_*` の接頭辞置換による逐語複製（本演算に
必要な最小集合のみ複製し、`sub`／`mul`〈elementwise〉／`rsqrt`／
`add_ro`／`scale_pow2` 等は複製しない）。`pool_f64_from_uint`
（`u32 -> binary64` の厳密変換）は本ファイル固有の追加ヘルパー。

**判定契約**: 値・索引とも CPU 側ホスト逐語モデル
（`crates/backend-metal/src/pooling_model.rs`。`cfg` なし・Linux でも
単体テストが回る）と bit 完全一致。MaxPool のタイは先勝ち（`v > best`
の厳密比較）・NaN は最初に出現した索引で固定される契約
（`docs/pooling-ops-design.md` §5 のタイ規則・NaN 伝播索引契約の
Metal 側実装）。

**実装ファイル**: `crates/backend-metal/src/{pooling.rs（macOS 限定・
起動 API・`MetalPooling`）, pooling_model.rs（cfg なし・derive_pool_dims
／derive_adaptive_dims／ホスト逐語モデル）, shaders/pooling.metal,
error.rs（`InvalidPoolingShape`／`PoolingSizeLimitExceeded` 追加）,
lib.rs（モジュール配線・`pub use pooling::MetalPooling`）}`。

**エラー写像**: `PoolingPrepareError::SizeLimitExceeded`（形状パラ
メータ導出値が `u32::MAX` 超過、または MaxPool 索引契約
`plane_in <= i32::MAX` 超過）→ `MetalError::PoolingSizeLimitExceeded`
（Layer B 実装時にホストフォールバックへ写像される想定）・
`InvalidShape` → `MetalError::InvalidPoolingShape`。

**テスト**: `tests/pooling_source_evidence.rs`（Linux 実行可能。REQ-8
手動境界検査・soft-f64 契約文字列・`PoolDims` フィールド順・
`pool_f64_*` ↔ `bn_f64_*` 逐語一致ドリフトガード）・
`tests/pooling_parity.rs`（`#![cfg(target_os = "macos")]`・全
`#[ignore]`。ホストモデルとの bit 完全一致・特殊値契約・`N=0` 早期
return・`PoolingSizeLimitExceeded` 型付きエラー）・
`pooling_model.rs` 内の単体テスト（`derive_pool_dims`／
`derive_adaptive_dims` の全ゲート境界例・独立 `f64` naive 参照との
bit 完全一致）。

**Layer B（`MetalBackendOps::{max_pool2d, avg_pool2d,
adaptive_avg_pool2d}` への trait 結線）は本 PR 時点では未実施**:
着手時点で `crates/tensor-core::backend_ops.rs`／`ops_shape.rs` に
`Pool2dParams`／該当 `BackendOps` メソッドが存在しなかった（兄弟
issue #1728〈CPU〉が並列実行中で未マージ）ため、`ops.rs` への推測
実装は行わず `crate::pooling::MetalPooling` の起動 API を trait 非
依存の引数（`&[f32]` + shape タプル）で完結させた。#1728 マージ後の
結線手順: `ops.rs` に `map_pooling_error`（`PoolingSizeLimitExceeded`
→ `BackendError::Unsupported`〈ホストフォールバック〉・
`InvalidPoolingShape` → `BackendError::ShapeMismatch`）と
`MetalBackendOps::{max_pool2d, avg_pool2d, adaptive_avg_pool2d}`
（`pool2d_out_shape` で再検査 → `cached_pooling`〈`context_cache.rs`
へ追加〉→ `run_*`）を追加する（親 #1607 を受け皿として記録。
`.claude/rules/out-of-scope-tracking.md`）。

**対象外・引き継ぎ**: GPU backward カーネル（MaxPool VJP は
`scatter_add` 経由・Avg VJP はホスト側のみ。設計 doc §11）・
`ceil_mode=true`・`divisor_override`・channels_last・3d・
MaxUnpool／AdaptiveMaxPool・facade 公開面拡張（設計 doc §12）・
M4 Max 実機実測（`docs/perf/logs/metal-pooling-1730/README.md` へ
申し送り）。
### #1729（CUDA 実装）

**前提の事実（2026-09-15 `main` 調査）**: 兄弟イシュー #1728（`backend-cpu`。
§9 が指す共有基盤 `Pool2dParams`・`BackendOps::max_pool2d`／`avg_pool2d`／
`adaptive_avg_pool2d`・出力 shape 関数）は本イシュー実装時点で `main` に
未マージだった。そのため本イシューは共有基盤（`tensor-core`／`autodiff`）
へ一切触れず、`crates/backend-cuda` クレート内に閉じた forward カーネル
実装のみを行った（**`ops.rs::CudaBackendOps` への override 配線は含まない**。
#1728 マージ後の小さな追従 PR へ引き継ぐ）。

**実装ファイル**:

- `crates/backend-cuda/src/kernels_pooling.rs`（新規）: `MAX_POOL2D_F32`・
  `AVG_POOL2D_F32`・`ADAPTIVE_AVG_POOL2D_F32` の 3 NVRTC 静的文字列。
  `POOLING_BLOCK_DIM = 256`。REQ-8 境界検査（`if (idx < numel)`）・
  `long long` 添字演算の静的テストを含む。
- `crates/backend-cuda/src/pooling.rs`（新規）: `CudaPooling`（`new`・
  `run_max_pool2d_f32`・`run_avg_pool2d_f32`・`run_adaptive_avg_pool2d_f32`）。
  `Pool2dParams` へ依存せずプリミティブ引数（`[usize; 2]`・`bool`）を受け、
  §3／§4 のパラメータ検査・出力 shape 導出をクレート内で自己完結して行う
  （`validate_and_shape`／`validate_and_shape_adaptive`／`pool_out_len`
  ほか）。`ops.rs` 側 override が無いためモジュール全体を
  `#![allow(dead_code)]`（配線後に撤去予定。理由はファイル doc 参照）。
- `crates/backend-cuda/src/pooling_model.rs`（新規。`#![cfg(test)]`）:
  `kernels_pooling.rs` 3 カーネルの逐語ホスト Rust モデル
  （`sort_model.rs`／`unique_model.rs` と同型の意図的複製）。実機
  `#[ignore]` テストの bit 一致オラクル。
- `crates/backend-cuda/src/pooling_real_device_tests.rs`（新規。`pooling.rs`
  末尾から `#[cfg(test)] #[path]` で登録。`CudaPooling` が非公開のため
  `tests/` からは到達不可という事情は `context_cache.rs::
  poison_recovery_real_device_tests` と同型）: 属性なし環境適応スモーク
  （`CudaDevice::new`／`CudaPooling::new` いずれの失敗でも panic しない
  ことを確認。libcuda はあるが libnvrtc が無い環境が実際に観測された）
  ＋ `#[ignore]` 形状網羅（基本形・重なり窓・dilation 境界・1d 併合・256
  ブロック境界またぎ・NaN／±inf／−0.0・run-to-run bit 同一・
  `count_include_pad` 両値・adaptive global average／upsampling）。
- `crates/backend-cuda/src/error.rs`：`CudaError::InvalidPoolingShape`・
  `PoolingSizeLimitExceeded` を追加。
- `crates/backend-cuda/src/lib.rs`：`mod kernels_pooling; mod pooling; mod
  pooling_model;` を登録。

**数値契約の実装**: MaxPool は §5 の先勝ち決定的タイ規則・NaN 伝播
（`v > best || (isnan(v) && !isnan(best))`。`-INFINITY` 番兵ではなく最初の
有効タップで初期化）を算術なしで実装し `pooling_model.rs` と bit 完全一致
契約（値・索引とも）。Avg／AdaptiveAvg は §7 の `double` 逐次加算＋1 回
`float` downcast（CUDA は `double` をハードウェアでネイティブサポートする
ため Metal のような soft-f64 エミュレーションは不要）。

**`context_cache::cached_pooling` を追加しなかった理由**: `ops.rs` に呼び
出し元が無いため、追加しても `pub(crate)` 未使用として dead_code lint に
より `cargo clippy --all-features -- -D warnings` が失敗する（`#[cfg(test)]`
以外の到達経路が現時点で存在しない）。#1728 マージ後の override 配線と
同時に追加する。

**検証**: `cargo test -p fandhe-ai-backend-cuda --lib pooling` で単体テスト
27 件（属性なし）が pass・実機 `#[ignore]` テスト 11 件は CUDA 実機必須の
まま記録・`cargo clippy --workspace --all-targets --all-features --
-D warnings` は本 PR の変更を含めて green（既存の未使用コード起因の
warning はベースライン不変）。**GB10 実機実測は本エージェント実行環境に
到達手段が無いため未実施のまま `docs/perf/logs/cuda-pooling-1729/` へ
申し送る**。
