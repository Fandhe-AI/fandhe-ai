# dtype 変換基盤（cast）の設計・実装記録（イシュー #1750・親 #1613）

## 0. 前提・承認状況

親 #1613 のユーザー承認コメント（2026-09-12）で、本 issue の「承認事項」が要求する 3 点
（公開クレート `fandhe-ai-tensor-core` の `BackendOps` trait 拡張・facade 公開面の拡張・
前提 spec 改定 #1591〈PR #1661 マージ済み・CLOSED〉）は解決済み。`docs/compat-api-scope.md`
§1.2 Tier 1 に `cast`（行 231）が列挙済みのため §5 の再適用は不要。

## 1. 背景・目的

`Var`（`autodiff`）は `Tensor<f32>` 専用であり、`Tensor<f64>`／`i32`／`i64`／`bool` は
`Element` として生成できるものの相互変換 API が存在しなかった
（`docs/compat-feature-gap.md` §2.12「`.to(dtype)`: なし」）。

本 issue は f32 をハブとする 8 方向の cast（f32→{f64,i32,i64,bool}・{f64,i32,i64,bool}→f32）
を、以下の層として実装する:

- (a) `tensor-core` の型基盤（sealed trait `CastElement`・dtype タグ `CastDType`・
  ホスト参照実装 `cast_from_f32`／`cast_to_f32`）
- (b) バックエンド dispatch 面（`BackendOps::cast_ops` capability accessor・
  `CastOps` trait）
- (c) CPU 実装（`backend-cpu::cast`）
- (d) `Var`／`Tape` の公開経路（`Var::cast`／`Var::to_f32`／`Tape::var_from`）
- (e) facade 到達経路（`CastDType`／`CastElement` の再エクスポート・
  `facade::Tape::var_from`）

兄弟イシュー #1751（CUDA／Metal cast カーネル）は本 issue が確定させる trait 面と
数値契約に従属する。

## 2. 勾配契約

出力が f32 以外の cast は勾配を打ち切る（tape ノードを記録せず detached な
`Tensor<T>` を返す。`Var::argmax`／`Var::unique` と同型）。f32 系（`Var::to_f32`＝
恒等）のみ勾配が伝播する。非 f32 → f32（`Tape::var_from`）は新しい葉
（`Op::Leaf`）として登録され、変換元テンソルへ勾配は流れない。

`Var::cast::<f32>()` も detached なコピーを返すだけであり勾配は伝播しない
（doc comment で明記。勾配を保つ f32 系の恒等射が必要な場合は `Var::to_f32` を
使う）。

## 3. 型基盤（`crates/tensor-core/src/cast.rs`）

### 3.1 `CastDType`

```rust
#[non_exhaustive]
pub enum CastDType { F32, F64, I32, I64, Bool }
```

`crate::element::ScalarDType`（演算対象 dtype 専用・`TypedOps<T>`／GEMM 経路選択
規則用）とは**別 enum**。`Scalar` は `f32`／`f64`／`half::f16`／`half::bf16` の
4 型に封印されており、`I32`／`I64`／`Bool` を追加すると `Scalar` の意味論
（演算対象・GEMM 経路選択規則の対象）を汚染する
（`docs/backend-dtype-dispatch-design.md` §4.4 の整理を維持）。

`#[non_exhaustive]`: 将来 dtype（`half::f16`／`half::bf16` 等）を追加する際に
公開 API を破壊しないため。

### 3.2 `CastElement`（sealed trait）

```rust
pub trait CastElement: Element + private::Sealed {
    const CAST_DTYPE: CastDType;
    fn from_f32(v: f32) -> Self;
    fn into_f32(self) -> f32;
    fn backend_cast_from_f32(ops: &dyn CastOps, x: &Tensor<f32>) -> Result<Tensor<Self>, BackendError>;
    fn backend_cast_to_f32(ops: &dyn CastOps, x: &Tensor<Self>) -> Result<Tensor<f32>, BackendError>;
}
```

実装対象は `f32`／`f64`／`i32`／`i64`／`bool` の 5 型。`half::f16`／`half::bf16`
は対象外（sealed＋`#[non_exhaustive]` のため後続で非破壊追加可能）。

`backend_cast_from_f32`／`backend_cast_to_f32` は「型ごとに固定の 1 メソッドを
呼ぶだけ」の橋渡しであり、`CastOps`（動的ディスパッチ面）の対応メソッドへ
委譲する。`f32` 自身の実装は恒等コピー（`ops` を使わない）。

### 3.3 数値契約（方向別）

| 方向 | 規則（Rust `as` 意味論） | GPU 実装への注記（#1751） |
|---|---|---|
| f32→f64 | 完全表現（exact） | Metal は `double` 非対応のため恒久 `Unsupported`（ホストフォールバック。`docs/backend-dtype-dispatch-design.md` §14 と整合） |
| f64→f32 | 最近接偶数丸め・範囲超過は ±inf | 同上 |
| f32→i32／i64 | ゼロ方向切り捨て・範囲外は飽和・NaN→0 | CUDA は `cvt.rzi.sat`（`__float2int_rz`／`__float2ll_rz`）が同意味論。Metal の `int(float)` は範囲外が未定義のため NaN→0・明示 clamp を必須とする |
| i32／i64→f32 | 最近接偶数丸め（`\|v\| > 2^24` は非可逆） | CUDA `__int2float_rn`／`__ll2float_rn`・Metal `float(int)` |
| f32→bool | `v != 0.0`（NaN→true・−0.0→false） | 比較のみ |
| bool→f32 | `true→1.0`・`false→0.0` | ホスト側で `bool → u8` に変換してから転送する |

Rust 上の注意: `bool as f32`／`f32 as bool` はコンパイル不能のため `if v { 1.0 }
else { 0.0 }`／`v != 0.0` を明示する（`CastElement` 実装内で対応済み）。

**GPU 実装（#1751）への `bool` 実体化契約**: GPU 側は `u8`（0／1）で転送し、
ホストで `Tensor<bool>` を構築する際は必ず `u8 → (v != 0)`（または `== 1` の
fail-closed 検査）で実体化する。生バイトの transmute／再解釈で `bool` を作る
ことは Rust の 0／1 妥当性不変条件に反し UB のため禁止する。

### 3.4 バックエンド間 parity

算術を含まない変換のため**bit 完全一致契約**（pad／one_hot／unique と同型）。
NaN のみ payload がプラットフォーム依存（Rust `as` の float→float 変換は
NaN payload を保証しない）のため「非 NaN は bit 一致・NaN はクラス一致」と
表記する。REQ-2 複合判定・tolerance／baseline は不変。

### 3.5 ホスト参照実装

```rust
pub fn cast_from_f32<T: CastElement>(x: &Tensor<f32>) -> Result<Tensor<T>, ShapeError>;
pub fn cast_to_f32<T: CastElement>(x: &Tensor<T>) -> Result<Tensor<f32>, ShapeError>;
```

手順: `checked_numel_for::<T>(x.shape())` で要素数積のオーバーフロー・
アロケーション上限を事前検査（`T` が f32 より大きい型〈`f64`／`i64`〉の場合、
入力 `x` 自体は妥当な shape でも `T` へキャストした際のバイト量が `Vec` の
アロケーション上限を超えうるため、`T` 基準で改めて検査する。`unique.rs` の
transpose 済み view 対策・PR #1828 の教訓と同種）→ `host_slice()`（非
contiguous view も稠密化）→ 要素ごと `from_f32`／`into_f32` → `Tensor::new`
（出力は常に contiguous・shape 保存）。

`Tensor<f32>::cast<T>()`／`Tensor<T>::to_f32()` は上記関数への薄い委譲
（inherent メソッド）。

## 4. バックエンド dispatch 面（`CastOps`）

```rust
pub trait CastOps {
    fn cast_f32_to_f64(&self, x: &Tensor<f32>) -> Result<Tensor<f64>, BackendError> { .. }
    fn cast_f32_to_i32(&self, x: &Tensor<f32>) -> Result<Tensor<i32>, BackendError> { .. }
    fn cast_f32_to_i64(&self, x: &Tensor<f32>) -> Result<Tensor<i64>, BackendError> { .. }
    fn cast_f32_to_bool(&self, x: &Tensor<f32>) -> Result<Tensor<bool>, BackendError> { .. }
    fn cast_f64_to_f32(&self, x: &Tensor<f64>) -> Result<Tensor<f32>, BackendError> { .. }
    fn cast_i32_to_f32(&self, x: &Tensor<i32>) -> Result<Tensor<f32>, BackendError> { .. }
    fn cast_i64_to_f32(&self, x: &Tensor<i64>) -> Result<Tensor<f32>, BackendError> { .. }
    fn cast_bool_to_f32(&self, x: &Tensor<bool>) -> Result<Tensor<f32>, BackendError> { .. }
}
```

全メソッドに既定実装（`Unsupported` fail-safe）を持たせるため、バックエンドは
対応する方向のみを部分的にオーバーライドできる（例: Metal は f64 の 2 方向
のみ既定のまま残せる）。

`BackendOps::cast_ops(&self) -> Option<&dyn CastOps> { None }` を
`typed_ops_f64` 等と同型の capability accessor として 1 メソッドだけ追加した
（`BackendOps` 自体のメソッド数増加を 1 に抑える）。

### API 配置案の比較

| 案 | 内容 | 採否 |
|---|---|---|
| **採用: accessor + 型パラメータ trait** | `BackendOps::cast_ops() -> Option<&dyn CastOps>`・`CastOps` は 8 固定メソッド | 採用。`TypedOps<T>`（#1687）と同じ設計方針を踏襲し、`BackendOps` 自体を汚染しない |
| 8 既定メソッド直付け案 | `BackendOps` へ `cast_f32_to_f64` 等 8 メソッドを直接追加（既定 `Unsupported`） | 不採用。`BackendOps` のメソッド数が肥大化し続ける（cast 以外の dtype 拡張でも同じ問題が再発する） |
| any-tensor enum 案 | `enum AnyTensor { F32(..), F64(..), .. }` を介した動的型消去 | 不採用。静的型検査の恩恵を失い、`Var::cast::<T>()` のような型パラメータ API と整合しない |

### フォールバック規則（2 段）

1. `ops.cast_ops()` が `None` → ホスト参照実装（`tensor_core::cast::*`）
2. `ops.cast_ops()` が `Some` だが個別方向が `Err(Unsupported)` → ホスト参照実装
3. それ以外の `Err` は伝播（判定迂回経路を作らない。`.claude/rules/security.md` A08）

呼び出し元（`autodiff::grad::cast_from_f32_with_fallback`／
`cast_to_f32_with_fallback`）は戻り shape が入力 shape と一致することを事後
検査し、不一致は `ShapeMismatch` で fail-closed に拒否する。

## 5. CPU 実装（`crates/backend-cpu/src/cast.rs`）

`impl CastOps for CpuBackendOps` は 8 方向すべてを
`fandhe_ai_tensor_core::cast::{cast_from_f32, cast_to_f32}` へ委譲する
（単一情報源の再利用。`gather_scatter.rs`／`unique.rs` のように独自アルゴリズムを
複製する必要がなく、算術を含まない純粋な選択・変換のため乖離のリスクがない）。
並列化しない（性能最適化は `.claude/rules/out-of-scope-tracking.md` 対象）。

`ops.rs::CpuBackendOps::cast_ops` は `typed_ops_bf16` 等と同じ `Some(self)`
パターンでオーバーライドする。

## 6. autodiff（`crates/autodiff/src/{grad.rs, var.rs, tape.rs}`）

- `grad::cast_from_f32_with_fallback<T: CastElement>`／
  `cast_to_f32_with_fallback<T: CastElement>`: §4 のフォールバック規則・
  shape 事後検査を実装する。要素数積のオーバーフロー検査は
  `tensor_core::cast::cast_from_f32`／`cast_to_f32` 内部（`checked_numel_for`）
  で完結しており、`unique_with_fallback` と異なり呼び出し元での重複検査は
  不要。
- `Var::cast<T: CastElement>(&self) -> Result<Tensor<T>, AutodiffError>`:
  `materialize_fallible` で実体化 → `cast_from_f32_with_fallback`。**非微分・
  detached**（`push_eager` を呼ばず tape ノードを追加しない）。
- `Var::to_f32(&self) -> Var<'t>`: `*self` を返す恒等（`Var: Copy`）。
- `Tape::var_from<T: CastElement>(&self, tensor: &Tensor<T>) -> Result<Var<'_>, AutodiffError>`:
  `cast_to_f32_with_fallback` → `self.var(&f32_val)`（葉 1 ノード追加）。

VJP 追加は行わない: 出力が非 f32 の cast には勾配定義が存在しないため、
「明示的な打ち切り（ノード非記録）」が契約であり、新規 `Op` variant・
`grad.rs::vjp` の分岐は追加しない（`docs/unique-facade-exposure-decision.md`
§3 案 A と同じ整理）。

## 7. facade（`crates/facade/src/lib.rs`）

`pub use fandhe_ai_tensor_core::{CastDType, CastElement};` の 1 行を追加。
`CastOps`（動的ディスパッチ面）は再エクスポートしない——利用者は
`Var::cast::<T>()`／`Tape::var_from(&Tensor<T>)` で到達する
（`crates/facade/tests/api_surface.rs::facade_does_not_reexport_cast_ops`
が機械的に固定する）。

`facade::Tape` へ `pub fn var_from<T: CastElement>(&self, tensor: &Tensor<T>) ->
Result<Var<'_>, AutodiffError>` を委譲追加（`var`／`backward` と同型）。

## 8. テスト構成

- `crates/tensor-core/src/cast.rs`: 手書きゴールデン値（NaN→0・
  `i32::MIN`／`i32::MAX`・`i64::MIN`／`i64::MAX` での飽和・`-0.0`→false・
  NaN→true・`2^24+1` の i64→f32 丸め非可逆・`f64::MAX`→`inf`・`f32→f64` の
  exact 往復）・`CastOps` 既定実装の object-safety・`backend_cast_*` の
  橋渡し検証。
- `crates/tensor-core/src/backend_ops.rs`: `cast_ops` accessor の既定 `None`・
  `&dyn BackendOps` 経由での具象 `CastOps` 到達検証。
- `crates/backend-cpu/src/cast.rs`: accessor が `Some`・8 方向がホスト参照
  実装と bit 一致（非 NaN）／クラス一致（NaN）・非 contiguous view。CPU
  `CastOps` はホスト参照実装へ委譲するだけのため、これは「委譲の配線」
  検証であり独立実装同士の検証ではない（doc comment に明記）。
- `crates/autodiff/tests/tape_recording.rs`: `cast` がノードを追加しない・
  `to_f32` は恒等でノード追加なし・`var_from` は葉 1 ノード追加。
- `crates/autodiff/tests/cast.rs`: 勾配契約（cast 呼び出しが無関係な
  backward に影響しない・`to_f32` は勾配が届く・`var_from` の葉に解析的
  勾配）・フォールバック規則（accessor `None`／`Some` だが `Unsupported`）・
  エラー伝播（`Unsupported` 以外は伝播）・出力 shape 不正の fail-closed 拒否。
- `crates/facade/tests/api_surface.rs`: `CastDType`／`CastElement` の facade
  到達性・`CastOps` 非再エクスポートの固定。
- `crates/facade/tests/cast_backend_parity.rs`: CPU vs NaiveOps の 8 方向
  bit 一致（属性なし）・CUDA／Metal は `#[ignore]`（#1750 時点では両
  バックエンドとも `CastOps` accessor が `None` のためホストフォールバック
  経路の確認に留まる。#1751 がカーネルを実装した後も同じテストで契約検証
  できる設計）。

## 9. スコープ外（`.claude/rules/out-of-scope-tracking.md` に従い記録）

- CUDA／Metal の cast カーネルの実機実測（#1751 で実装は完了したが、
  DGX Spark GB10／Apple Silicon 実機は本エージェント実行環境に到達
  手段がないため未実施のまま Mac／GB10 セッションへ申し送り。§11
  参照）。
- `half::f16`／`bf16` を cast 対象へ追加すること（sealed＋`#[non_exhaustive]`
  のため後続で非破壊追加可能）。
- 非 f32 ↔ 非 f32 の直接変換（f64→i64 等。現状は f32 経由の合成となり
  非可逆になりうるため API として提供しない）。
- 整数／bool 値を保持する `Var`（tape ノード）・GPU 常駐チェーン上の
  cast・実行時 dtype 指定 API（`to_dtype(CastDType)` と any-tensor 戻り値）。
- CPU cast の rayon 並列化等の性能最適化。

## 10. 実装記録

- 2026-09-14: tensor-core 型基盤（`cast.rs`・`backend_ops.rs::cast_ops`
  accessor）・CPU 実装（`backend-cpu/src/cast.rs`）・autodiff 公開経路
  （`Var::cast`／`Var::to_f32`／`Tape::var_from`）・facade 到達経路
  （`CastDType`／`CastElement` 再エクスポート・`facade::Tape::var_from`）を
  実装。`cargo test --workspace --all-features`・
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`・
  `cargo fmt --all --check` すべて green。CUDA／Metal 実機は未実測のまま
  #1751 へ申し送り。

## 11. 実装記録（#1751）

CUDA（8 方向すべて）・Metal（f64 2 方向を除く 6 方向）のネイティブ
カーネルを実装し `cast_ops` accessor を `Some(self)` へオーバーライド
した（イシュー #1751・親 #1613）。

**ファイル一覧**:

- CUDA: `crates/backend-cuda/src/kernels_cast.rs`（8 カーネルソース。
  `#[cfg(test)]` 文字列テスト 6 件）・`src/cast.rs`（`CudaCast`・
  `impl CastOps for CudaBackendOps`）・`src/context_cache.rs::cached_cast`・
  `src/memory.rs`（`ReadbackSentinel for f64／i64／u8`）・`src/ops.rs`
  （`cast_ops` accessor・`checked_shape_numel` を `pub(crate)` 化）・
  `tests/cast_ops_contract.rs`（driver 非接触の常時 CI 実行テスト）・
  `tests/cast_parity.rs`（環境適応スモーク＋`#[ignore]` 実機テスト）
- Metal: `src/shaders/cast.metal`（6 カーネル）・`src/cast_buffer.rs`
  （`MetalCastBuffer<T>`。`i32`／`i64`／`u8` を扱う要素型 generic な
  バッファ。`crate::buffer::MetalBuffer`〈f32 専用〉・`crate::
  half_buffer::MetalHalfBuffer`〈f16 専用〉と同じ設計判断で既存
  シグネチャへ触れない独立型）・`src/cast.rs`（`MetalCast`・`impl
  CastOps for MetalBackendOps`）・`src/context_cache.rs::cached_cast`・
  `src/ops.rs`（同上）・`tests/cast_source_evidence.rs`（Linux 実行
  可能な文字列証跡 9 件）・`tests/cast_parity.rs`（`#[ignore]` 実機
  テスト）
- facade: `crates/facade/tests/cast_backend_parity.rs`（`#[ignore]`
  テストを 8 方向〈CUDA〉／6 方向〈Metal〉へ拡張）

**カーネル記述規則**（bit 完全一致契約を満たすための共通規則。
`kernels_cast.rs`／`shaders/cast.metal` 冒頭コメントが正）:

1. NaN／非ゼロ判定は bit パターン（CUDA `__float_as_uint`／Metal
   `as_type<uint>`）で行い、`isnan()`・通常の浮動小数点比較には
   依存しない。
2. f32→i32／i64 の飽和境界は単一の明示式・ヘッダ非依存のリテラル
   定数（`INT_MAX`／`LLONG_MAX` 等のマクロは使わない）で書く。
3. 整数→f32 は最近接偶数丸め（CUDA `__int2float_rn`／`__ll2float_rn`
   intrinsic を明示指定・Metal は `float(int)`／`float(long)` の既定
   丸めに委ねる）。
4. f32↔f64（CUDA のみ）は単純なキャスト（`(double)v`／`(float)d`）。
5. `bool` は `unsigned char`（CUDA）／`uchar`（Metal）の 0／1 として
   転送し、ホスト側で必ず `v != 0` により `bool` を実体化する（生
   バイトの transmute／再解釈は行わない）。

**§3.3 GPU 実装への注記からの実際の逸脱**: 当初案（`cvt.rzi.sat` 系
intrinsic への依存）ではなく、CUDA も明示 clamp 式（規則 2）で統一した
（Metal と実装形を揃え、両バックエンドとも同一の飽和境界リテラルを
共有できるようにするため）。

**accessor 切替に伴う意図した挙動変更**: `cast_ops` を `None` →
`Some(self)` へ切り替えたため、CUDA／Metal デバイス不在環境で
`Device::Cuda`／`Device::Metal` の tape から `Var::cast` を呼ぶと、
従来の暗黙ホストフォールバックではなく `CudaUnavailable`／
`KernelLaunchFailed` 系のエラーが表面化する（`CastOps` の
フォールバック規則は「`Unsupported` のみフォールバック」であり内部
契約違反・driver 不在をホストフォールバックで覆い隠さないため。
`Var::unique`／`Var::matmul` と同じ既存の挙動であり退行ではない）。

**事前登録した代替案（§11 実測待ち）**: Metal `float(long)`（i64→f32）
の最近接偶数丸めは実機未確認。M4 Max 実機実測で `cast_i64_to_f32`
のみ bit 不一致が判明した場合は、その方向のみ Metal 側で既定
`Unsupported`（ホストフォールバック）へ戻す（tolerance は変更しない）。
`tests/cast_parity.rs`（Metal）の i64→f32 fixture（`2^24+1`・`2^25+1`・
`2^25+3` 等のタイケースを含む）がこの判定材料になる。

**検証**: `cargo test --workspace --all-features`・
`cargo clippy --workspace --all-targets --all-features -- -D warnings`・
`cargo fmt --all --check`・`cargo check -p fandhe-ai-backend-metal
--tests --target aarch64-apple-darwin` すべて green（Linux 実行環境）。
CUDA／Metal 実機（DGX Spark GB10／Apple Silicon）での `#[ignore]`
テスト実行は本エージェント実行環境に到達手段がないため未実施のまま
申し送る。
