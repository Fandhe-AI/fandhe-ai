# dtype 別 dispatch（`BackendOps` の dtype 多重化）方式の設計記録（#1648）

イシュー #1648「dtype 別 dispatch（`BackendOps` の dtype 多重化）方式を設計する」に対応する。親: #1626（低レイヤー診断・機能網羅ツリー #1570 の sub (a)）。

本ドキュメントは**設計のみの記録であり、コード変更（`crates/**`）を一切伴わない**。実装は後続イシュー（#1649 CPU → #1650 CUDA → #1651 Metal）が本設計に従って行う。tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline（`ParityBaseline::BASELINES`）の変更・新規追加は本設計のスコープ外であり、実際に本文書はそれらを一切変更しない。

**本文書自体は承認記録ではない**。§7 に列挙する事項は実装着手前にユーザー承認が必要（`docs/spec/` 正本の変更を伴わないが、公開クレート `fandhe-ai-tensor-core` の trait 拡張であるため）。

棚卸し時点の HEAD（origin/main）: `f91cafa3`（2026-09-13）。`file:line` は同時点のもの。

## 0. 判断サマリ

- 推奨方式は **capability accessor ＋ 型パラメータ trait**（§4 の案 D）: `BackendOps` に既定 `None` を返すアクセサ `typed_ops_f64`／`typed_ops_f16`／`typed_ops_bf16(&self) -> Option<&dyn TypedOps<T>>` を非破壊追加し、dtype ごとの演算本体は新設 `pub trait TypedOps<T: Scalar>`（trait 型パラメータのため object-safe）に集約する。`fn memory_ops(&self) -> Option<&dyn MemoryOps>`（`crates/tensor-core/src/backend_ops.rs:354`）と同型の既存パターンをそのまま踏襲する
- 不変条件（機械検査可能な形で列挙。実装側の受け入れ条件とする）:
  1. `dyn BackendOps` の object safety が崩れない（`assert_object_safe(_ops: &dyn BackendOps) {}`。`crates/tensor-core/src/backend_ops.rs:1868` と同型のテストで担保する）
  2. 既存 `BackendOps` メソッドのシグネチャ・挙動が不変（公開 API 非破壊）
  3. f32 経路は before/after で bit 同一（新 trait を f32 にも実装する場合、既存メソッドへの委譲のみで構成する）
  4. 新規追加は既定 `None`／`Unsupported` から開始する（fail-closed。#1649〜#1651 の受け入れ条件と一致させる）
- 本設計は `dispatch::DType`（`crates/tensor-core/src/dispatch.rs:30`）を拡張しない。REQ-11 の「行列演算ユニットの明示切替 API を利用者に提供しない」制約とは独立の理由（後述 §4.4）による

## 1. 背景・目的

現状、演算可能な dtype は `f32` のみである。`Tensor<T: Element>`（`T` は `f32`／`f64`／`i32`／`i64`／`bool`／`half::f16`。`crates/tensor-core/src/element.rs:24-83`）自体はジェネリックだが、`BackendOps`（`crates/tensor-core/src/backend_ops.rs:328`）・`MemoryOps`（`crates/tensor-core/src/buffer.rs:374` 付近）・`DeviceBufferView`・autodiff の `Var`／`Tape`（`Tape.ops: Box<dyn BackendOps + Send>`。`crates/autodiff/src/tape.rs:775`）はすべて `f32` に固定されている。

`docs/public-api-design.md` は 2 箇所でこの未決事項を明記している。

- §4.2（`crates/tensor-core/src/backend_ops.rs` 相当の設計節、`docs/public-api-design.md:637`）: 「`BackendOps` を `T: Element` でジェネリック化するか、`f16` 専用の並行トレイトを追加するかは TASK-1.9 実装時に決定する」
- §6-8（`docs/public-api-design.md:721`）: 同内容の要約

本イシューはこの未決事項を閉じ、実装側（#1649〜#1651）が従う具体案を確定する。

## 2. 現状のコード事実（棚卸し）

| 事実 | 出典 |
|---|---|
| `Element` は `Copy + Send + Sync + Debug + PartialEq + 'static` に `zero()`／`one()` を要求する unsealed trait。実装対象は `f32`／`f64`／`i32`／`half::f16`／`i64`／`bool` | `crates/tensor-core/src/element.rs:24-83` |
| `BackendOps` は `f32` 固定。`memory_ops(&self) -> Option<&dyn MemoryOps> { None }` という「既定 `None` を返すアクセサ trait」の非破壊拡張パターンが既に存在する | `crates/tensor-core/src/backend_ops.rs:328,354` |
| `dyn BackendOps` の object safety はテスト `assert_object_safe(_ops: &dyn BackendOps) {}` で機械検査されている | `crates/tensor-core/src/backend_ops.rs:1868` |
| `Tape.ops: Box<dyn BackendOps + Send>`。dtype ジェネリックにする場合は `Tape<T>` 化が必要（本段階ではスコープ外。§8） | `crates/autodiff/src/tape.rs:775` |
| `dispatch::DType { F32, F16 }` は `#[non_exhaustive]` **なし**。variant 追加は下流の全網羅 match を壊す破壊的変更 | `crates/tensor-core/src/dispatch.rs:30-40` |
| `select_gemm_kernel` は利用者向け明示切替 API ではなく、CUDA／Metal の GEMM 自動経路入口が内部で呼ぶ規則エンジン（REQ-11 の受け入れ基準） | `crates/tensor-core/src/dispatch.rs:9-19`、`docs/dispatch-rules-design.md` §5.1 |
| CUDA f16 Tensor Core 経路 `CudaGemmAuto::run_f16`（mma.sync 優先→WMMA フォールバック）は `BackendOps` から到達不能（`gemm_auto.rs` は `backend_ops.rs` の外） | `crates/backend-cuda/src/gemm_auto.rs:1623,1768` |
| CUDA f32 精度切替 `CudaGemmPrecision { Fp32Strict, Tf32, Tf32x3 }`（既定 `Fp32Strict`）は `CudaBackendOps::gemm` のみに適用。dtype 切替ではなく同一 f32 dtype 内の演算精度切替 | `crates/backend-cuda/src/precision.rs:12-77` |
| CUDA 側 dtype 一般化の前例 `pub(crate) trait PoolDtype: DeviceRepr + Sized`（`f32`／`f16` 実装済み）。プールアロケータ限定で公開 API ではない | `crates/backend-cuda/src/pool.rs:202,229,254` |
| Metal f16 タイル GEMM 入口 `gemm::MetalGemm::dispatch_f16_auto_unverified` は `_unverified` suffix・`#[doc(hidden)]`（PR #346 codex-review 指摘により意図的に未検証扱いのまま維持） | `crates/backend-metal/src/lib.rs:91-183` |
| Metal（MSL）は `double` 型非対応。既存の回避策は「64bit 整数による `f64` 加算のソフトウェアエミュレーション」（`soft_f64.rs`。GEMM の bias 勾配縮約限定・bit 完全一致契約） | `crates/backend-metal/src/soft_f64.rs:1-30`、`.claude/rules/coding-rust.md`「正規化統計・勾配の長軸縮約」節 |
| onnx-interop の dtype タグ付き enum `pub enum Value { F32, I64, Bool, F16 }` が「enum で dtype を運ぶ」方式の前例 | `crates/onnx-interop/src/onnx/interp.rs:62` |
| `half = "=2.7.1"`（`bf16` を同梱）。`cudarc = "=0.19.8"` は `f16` feature 済み依存 | `Cargo.toml:112,146` |
| `docs/compat-feature-gap.md` §2.12 は `float64`／`float16`/`bfloat16` を「部分実装（`BackendOps`/`Var` の算術対象外）」・必要工数 XL と記録済み | `docs/compat-feature-gap.md:319-320` |

## 3. 関連 issue との境界

- **#1613（cast `.to(dtype)`）**: f32 ⇔ 他 dtype の変換 API は #1613 側の責務。本設計は変換後の「dtype ごとの演算実行」経路のみを扱う（#1750 で `CastDType`／`CastElement`／`BackendOps::cast_ops`／`CastOps`・`Var::cast`／`to_f32`／`Tape::var_from` を実装済み。設計・数値契約は `docs/tensor-core-cast-design.md`）
- **#1625（AMP: automatic mixed precision）**: 本 dtype 多重化を前提として損失スケーリングを実装する。本設計には AMP のスケーリング契約を含めない
- **#1627（量子化）**: spec 除外事項に従属し着手不可。本設計の対象外

## 4. 候補比較

| 案 | 概要 | object safety | 公開 API 非破壊 | 演算×dtype の増え方 | 判定 |
|---|---|---|---|---|---|
| A | `BackendOps` のメソッドを `fn gemm<T: Element>(...)` へジェネリック化 | **不成立**（メソッドの generics は `dyn` 非互換。Rust の基本制約） | 破壊的変更 | 1 定義 | 不採用 |
| B | dtype 別メソッド族を直接追加（`gemm_f64`／`gemm_f16`／`gemm_bf16` 等を既定 `Unsupported` で追加） | 成立 | 非破壊 | 演算数 × dtype 数で線形爆発。命名規約の維持コストが増大し続ける | 最小集合なら可能だが非推奨 |
| C | dtype タグ付き enum（`onnx-interop::Value` 前例）を受ける `gemm_dyn(&self, a: &TensorDyn, ...)` | 成立 | 非破壊 | 1 定義／演算・実行時 match で分岐 | 代替案。型安全性を実行時チェックへ落とし f32 経路にも分岐コストが乗る |
| D | `trait TypedOps<T: Scalar>` ＋ `BackendOps::typed_ops_<dtype>() -> Option<&dyn TypedOps<T>>` | 成立（trait は型パラメータであり `dyn` 自体を要求しない。個々の `TypedOps<f32>` 等は具象型でアクセスするため dyn 互換性の問題が生じない） | 非破壊（既定 `None`。`memory_ops` と同型） | 1 定義／演算・backend × dtype ごとに `impl TypedOps<T> for XxxBackendOps` | **推奨** |

### 4.1 案 D の詳細

新設型（いずれも公開クレート `fandhe-ai-tensor-core`。§7 の承認対象）:

```rust
/// 演算対象になりうる dtype の capability 境界。
/// `Element`（`crates/tensor-core/src/element.rs:24`）は unsealed で
/// 外部実装が存在しうるため変更しない。演算に必要な追加境界は
/// sub-trait として表現する。
pub trait Scalar: Element {
    const DTYPE: ScalarDType;
    // 四則演算・比較・f32/f64 相互変換に必要な最小境界（実装時に確定）
}

/// 演算対象 dtype のタグ。`dispatch::DType`（GEMM 経路選択専用・
/// `#[non_exhaustive]` なし）とは別の列挙とし、拡張してよいものと
/// してよくないものを区別する。
#[non_exhaustive]
pub enum ScalarDType { F32, F64, F16, Bf16 }

/// dtype 別の演算本体。`BackendOps` の `typed_ops_<dtype>()` accessor
/// 経由でのみ取得する。
pub trait TypedOps<T: Scalar> {
    fn gemm(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn add(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn mul(&self, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn relu(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn exp(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn tanh(&self, a: &Tensor<T>) -> Result<Tensor<T>, BackendError>;
    fn sum(&self, a: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>, BackendError>;
    fn max(&self, a: &Tensor<T>, dim: Option<usize>) -> Result<Tensor<T>, BackendError>;
}
```

`sum`／`max` は既存 `BackendOps::sum`／`max`（`crates/tensor-core/src/backend_ops.rs:805-806`）と同じ `dim: Option<usize>`（`None` は全要素縮約・`Some(d)` は軸 `d` に沿った縮約）を保持する。dtype 多重化はこの引数を変更する理由にならないため、`TypedOps<T>` でも軸指定を落とさない。

`BackendOps` への非破壊追加:

```rust
fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>> { None }
fn typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>> { None }
fn typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>> { None }
```

`TypedOps<f32>` を追加するかは実装側判断とするが、追加する場合は既存 `BackendOps` メソッドへの委譲のみで構成し bit 同一を保つ（§0 不変条件 3）。これにより `fn run<T: Scalar>(ops: &dyn TypedOps<T>)` のような dtype ジェネリックな利用側コードが書ける。

入出力はホスト常駐 `Tensor<T>`（`BackendOps` v1 と同じ契約）に限定する。`DeviceBuffer<T>` 常駐経路（`memory_ops`／`linear_forward_device` 系）の dtype 多重化は「段階 B」として本設計のスコープ外に置く（§8）。

### 4.2 最小演算集合（第 1 段）

`gemm`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max` の 8 演算（既存 `BackendOps` v1 の演算集合と同一）。`gemm_bias_act`・`mse_loss`・softmax 系・resident 系・fusion 対応は第 2 段以降（別 issue）とし、本表で対象を明示的に固定することで #1649〜#1651 が対象を無制限に広げないようにする。

### 4.3 利用側の dtype 選択と REQ-11 の関係

dtype の選択は「`Tensor<f16>` を渡す」という**型で決まる入力**であり、REQ-11 が禁じる「行列演算ユニットの明示的な設定項目としての切替 API」には該当しない。GEMM 内部のカーネル経路選択（TF32 か SIMT か、mma.sync か WMMA か）は既存 `select_gemm_kernel(caps, shape, DType::F16)`（`crates/tensor-core/src/dispatch.rs`）をそのまま再利用し、`TypedOps<f16>::gemm` の実装内部から呼ぶ。dtype 多重化のための並行規則エンジンは作らない。

### 4.4 `dispatch::DType` を拡張しない理由

`dispatch::DType`（`crates/tensor-core/src/dispatch.rs:30`）は `#[non_exhaustive]` を付けていないため、variant 追加（`F64`・`Bf16` 等）は下流の全網羅 `match` を壊す破壊的変更になる。加えて `select_gemm_kernel` は GEMM カーネル経路選択専用の決定表（`docs/dispatch-rules-design.md` §5.3）に紐づいており、f64／bf16 の GEMM 経路が実機実測・承認を経ていない現時点でこの決定表へ組み込むのは時期尚早である。本設計は代わりに新設 `ScalarDType`（`#[non_exhaustive]` 付き）で dtype を表現し、GEMM 実行時にのみ `ScalarDType → Option<dispatch::DType>` の明示マッピング（`F32 → Some(F32)`、`F16 → Some(F16)`、`F64`／`Bf16 → None`＝規則エンジン非対象。呼び出し側でカーネル経路を直接選ぶ）を経由して接続する。`dispatch::DType` の `#[non_exhaustive]` 化自体は破壊的変更を伴うため別途承認判断に委ねる（§8）。

## 5. 実現可否表（dtype × backend）

| dtype | CPU | CUDA | Metal |
|---|---|---|---|
| f64 | 実装可（`f64::mul_add` 参照実装。並列 BLIS 化は任意） | SIMT カーネル実装可能だが性能目的なし。既定 `Unsupported` から開始 | **構造的に不可**（MSL に `double` 型が存在しない。`soft_f64.rs` の 64bit 整数エミュレーションは bias 勾配縮約というスカラー累算専用に作られたものであり、GEMM 全体を `f64` 精度で動かす手段ではない）→ 恒久 `Unsupported`（fail-closed） |
| f16 | `half` によるソフトウェア変換で f32 累算（aarch64 fp16 intrinsics は `unsafe` を伴うため実装時に別途 security-auditor 承認が必要。既定はソフトウェア変換） | 既存 `CudaGemmAuto::run_f16`（mma.sync 優先→WMMA）を結線。parity は REQ-2 形状別判定方式の既存 baseline 範囲内（新規 baseline 追加は行わない） | 既存 `dispatch_f16_auto_unverified` を結線。`_unverified`／`#[doc(hidden)]` の解除可否は #1651 の承認事項（§7-4） |
| bf16 | `half::bf16` で f32 累算（依存追加なし。`half =2.7.1` に同梱） | **可**（イシュー #1704 で確定。`cudarc-0.19.8/src/driver/safe/core.rs:990,1038` に `unsafe impl ValidAsZeroBits`／`DeviceRepr for half::bf16` が `#[cfg(feature = "f16")]` 配下で存在し、workspace `Cargo.toml:112-118` の `cudarc` 依存は `f16` feature を既に有効化済み。依存・feature 変更なしで `TypedOps<bf16>` 実装済み。§12 参照） | **可**（イシュー #1706 で実装済み。ホスト側 bf16⇔f32 変換＋既存 f32 `BackendOps` カーネルへの委譲方式は MSL `bfloat` 型の可用性に依存しないため、(b) ネイティブ bf16 経路の可否とは独立に (a) `TypedOps<bf16>` を実装できることが判明した。§14 参照。MSL `bfloat`／`simdgroup_bfloat8x8` の実機可用性は `crate::typed_bf16_probe_diag_tests` のコンパイルプローブへ切り出し・M4 Max 実機未実測のまま Mac セッションへ申し送り） |

## 6. 数値契約（tolerance／baseline 不変）

- **累算契約**: f16／bf16 は入力を f32 へ昇格し `f32::mul_add` で累算、最後に 1 回だけ元の dtype へ丸める（GPU の「f16 入力・f32 累算」Tensor Core と同型）。f64 は `f64::mul_add` を使う。matmul 系 FMA 契約（`.claude/rules/coding-rust.md`「バックエンド構成」節）と整合し、これを変更しない。正規化統計・長軸縮約の `f64`／soft-f64 アキュムレータ契約（同ルール文書の別節）も本段階の最小演算集合（§4.2）には含まれないため不変のまま
- **判定方法**: f16／bf16 出力は、参照値側も出力 dtype と同じ丸め（f32 参照計算 → f16／bf16 へ最近接丸め → f32 へ再昇格）を経てから f32 昇格後の実測出力と既存 `compare`／`assert_parity`（`crates/backend-cpu/src/parity.rs:148,239`）で判定する。参照値を丸め前の f32 のまま比較する方式は採らない: 例えば `1 + 2^-8` の bf16 最近接偶数丸め結果は丸め前 f32 参照値に対し相対誤差 約 0.003891・絶対誤差 約 0.003906（複合判定の両閾値 `RELATIVE_TOLERANCE`＝1e-3・`ABSOLUTE_RESCUE_THRESHOLD`＝1e-5 をいずれも超過）となり、丸め自体が正しい bf16 実装ですら不合格になりうる（tolerance 定数の緩和ではなく、比較対象を出力 dtype の表現可能値に揃える判定契約の整備で解消する）。dtype ごとの丸め関数（`half::f16::from_f32`／`half::bf16::from_f32`）は `half =2.7.1` に既存でありこの目的のためだけの新規実装は不要。f64 出力は f64 のまま同一の定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）で判定するヘルパー追加は実装側（#1649）の作業とし、定数自体は共有し変更しない。CUDA f16 GEMM の既知不合格形状は spec REQ-2（2026-09-02／2026-09-12 追記）の実測 baseline 非後退方式の既存範囲で扱い、**本設計は新規 baseline を追加しない**（追加には実機実測値と人間承認が必要）
- **bit 同一契約**: f32 経路（既存メソッド・`TypedOps<f32>` を追加する場合はその委譲実装）は before/after で bit 同一であることを実装側の受け入れ条件とする

## 7. 承認事項（実装着手の前提。本文書は承認記録ではない）

1. 公開クレート `fandhe-ai-tensor-core` の `BackendOps` への既定メソッド追加（`typed_ops_f64`／`typed_ops_f16`／`typed_ops_bf16`）
2. 新規公開型 `Scalar`・`ScalarDType`（`#[non_exhaustive]`）・`TypedOps<T>` の追加、および `impl Element for half::bf16` の追加
3. `dispatch::DType` を拡張しない方針（§4.4）。拡張が必要になった場合は破壊的変更として別途承認を要する
4. Metal f16 入口 `dispatch_f16_auto_unverified` の `_unverified`／`#[doc(hidden)]` 解除可否（#1651 の実装時判断）
5. CPU f16／bf16 経路で `unsafe` intrinsics（aarch64 fp16 等）を使う場合の承認（既定はソフトウェア変換で `unsafe` 非導入。#1649）
6. 最小演算集合（§4.2）と、#1649〜#1651 の受け入れ条件の再スコープ（`Var`／`Tape`／VJP は本段階の対象外。§8 参照）
7. facade（`crates/facade`）公開面への昇格は本設計の対象外。`docs/compat-api-scope.md` §5 に定める昇格手続きを別途要する

## 7.5 実装記録（#1698・CPU `TypedOps<f16>`）

親 #1649 コメント（2026-09-12 ユーザー承認。§7-1〜7-2 の承認事項に対応）を受け、
`crates/backend-cpu` に `impl TypedOps<half::f16> for CpuBackendOps`（§4.2 最小
集合 8 演算：`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`）を実装した。

- **設計**: §5 の実現可否表どおり `half` によるソフトウェア変換（aarch64 fp16
  intrinsics 等の `unsafe` は使わない。承認事項 5 の既定側）。各演算は
  「f16 → f32 昇格（`crate::typed_f16::upcast_f16`）→ 既存 f32
  `BackendOps` カーネルへ委譲 → f32 → f16 へ 1 回丸め
  （`crate::typed_f16::downcast_f32`）」の 3 段構成で実装し、`elementwise.rs`／
  `gemm.rs`／`gemm_blis/**`／`reduction.rs`／`parity.rs`（f32 経路本体）は
  一切変更しない
- **ファイル**: `crates/backend-cpu/src/typed_f16.rs`（新規）・`ops.rs`
  （`typed_ops_f16(&self) -> Option<&dyn TypedOps<half::f16>>` accessor を
  `Some(self)` へ結線）・`lib.rs`（`mod typed_f16;`）
- **数値契約**: `gemm` は `f16::from_f32(matmul_reference_fma(...))` と bit
  完全一致（既存 GEMM 契約テストが f32 経路の bit 一致を別途保証済み）。
  `add`／`mul`／`relu`／`exp`／`tanh` はスカラー参照実装の f16 丸め値と bit
  完全一致。`sum`／`max` は f64 逐次和／`f32::max` 参照との複合判定
  （`assert_parity`。tolerance 定数は不変）。`exp` 等で f16 表現範囲
  （`|x| <= 65504`）を超える場合は IEEE 754 の ±inf 丸めへ落ちる（PyTorch
  `half` と同じ挙動。既知事項として明文化）
- **テスト**: `crates/backend-cpu/src/typed_f16.rs` 内単体テスト（既知値・
  空/端点・エラー経路）・`crates/backend-cpu/tests/typed_ops_f16_parity.rs`
  （形状グリッド・非 contiguous view・CHUNK 境界を跨ぐ reduction）
- **f32 経路無変更の根拠**: `git diff --stat main -- crates/backend-cpu/src/elementwise.rs crates/backend-cpu/src/gemm.rs crates/backend-cpu/src/gemm_blis crates/backend-cpu/src/reduction.rs crates/backend-cpu/src/parity.rs crates/tensor-core` が空
- **スコープ外**: bf16（#1699）・CUDA（#1650）・Metal（#1651）・`Var`／`Tape`／
  VJP・facade 公開面昇格・aarch64 fp16 intrinsics 高速化・非 contiguous
  view の stride 読み高速経路（性能最適化は対象外）

## 8. スコープ外・引き継ぎ

- **`Var`／`Tape` の dtype 一般化**: `Tape.ops: Box<dyn BackendOps + Send>`（`crates/autodiff/src/tape.rs:775`）は本段階では `f32` のまま不変。dtype ジェネリックな `Var<T>`・VJP・`FusionPlan` の対応は別イシュー。設計判断は `docs/autodiff-var-dtype-multiplexing-design.md`（#2061）で記録済み（`Var<T>` フル一般化は見送り・narrow opt-in パターンを推奨案として記録。標準化は同 doc §10 承認事項 1 の承認待ち・未確定。段階 0）
- **AMP（損失スケーリング）連携**: #1625 側の責務
- **cast（`.to(dtype)`）**: #1613 側の責務（#1750 で実装済み。`docs/tensor-core-cast-design.md`）
- **`MemoryOps`／`DeviceBuffer<T>` 常駐経路の dtype 多重化**（段階 B）: `linear_forward_device` 系のデバイス常駐チェーンへの dtype 拡張は本設計に含めない
- **fusion（カーネル融合機構）の dtype 対応**: `kernel-fusion.md` の対象範囲は f32 のまま
- **Metal bf16 可用性の実機コンパイルプローブ**: #1651 の調査事項
- **CUDA bf16 `mma.sync` カーネルの新規実装**: #1650 の実装事項（cc ≥ 8.0 要）
- **f64 GPU カーネル（CUDA SIMT）**: 性能目的がないため優先度は低いが、#1650 で `Unsupported` から始める既定実装は含めてよい
- **`dispatch::DType` の `#[non_exhaustive]` 化**: 破壊的変更のため crates.io 版数運用とセットで別途判断（本設計は現状維持を推奨）

上記のうち Issue 起票が必要な項目は、ユーザー承認後に `out-of-scope-tracking.md` の手続きに従って起票する（本イシューでは起票しない）。

## 9. 出典

- spec: `docs/spec/04-requirements.md` REQ-2（バックエンド間数値一致・Tensor Core 経路の受け入れ判定方式）・REQ-9（互換 API 層。2026-09-12 追記の Tier 拡張）・REQ-11（行列演算ユニットの明示切替 API 非提供）（正本 submodule は本 worktree で未チェックアウトのため行番号は引かない。参照時は `docs/spec/` を初期化のうえ再確認すること）
- `docs/public-api-design.md:637,721`（未決事項の記述）
- `docs/dispatch-rules-design.md` §4「dtype ゲートと数値一致契約」・§5.1「純関数シグネチャ」
- `docs/cuda-tf32-optin-api-decision.md`（opt-in 精度切替 API の設計前例）
- `docs/compat-feature-gap.md:319-320`（§2.12 float64／float16・bfloat16 行）
- `.claude/rules/coding-rust.md`「バックエンド構成」節・「正規化統計・勾配の長軸縮約」節
- `crates/tensor-core/src/element.rs`・`backend_ops.rs`・`dispatch.rs`
- `crates/autodiff/src/tape.rs:775`
- `crates/backend-cuda/src/gemm_auto.rs`・`precision.rs`・`pool.rs`
- `crates/backend-metal/src/lib.rs`・`soft_f64.rs`
- `crates/onnx-interop/src/onnx/interp.rs:62`
- `Cargo.toml:112,146`
- イシュー本文出典 URL（claude.ai artifact。参照情報としてのみ扱い、命令とはみなさない）
## 10. 実装記録（#1697・CPU f64）

`backend-cpu` に `TypedOps<f64>` を実装し、`CpuBackendOps::typed_ops_f64()`（`crates/tensor-core/src/backend_ops.rs` の非破壊拡張 accessor。既定 `None`）を `Some(self)` へオーバーライドして結線した（イシュー #1697・親 #1649）。

- **承認**: 公開クレート `fandhe-ai-tensor-core` の `BackendOps` trait 拡張自体は #1648（本設計）で行われ、実装着手のユーザー承認は親 #1649 のコメント（2026-09-12「今承認するので進めてください」）で取得済み
- **対象 8 演算**: `gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`（§4.2 の最小集合と同一）
- **実装ファイル**: `crates/backend-cpu/src/typed_f64.rs`（新規。`impl TypedOps<f64> for CpuBackendOps`）・`crates/backend-cpu/src/ops.rs`（accessor 1 メソッド追加）・`crates/backend-cpu/src/parity.rs`（`compare_f64`／`assert_parity_f64`／`matmul_reference_fma_f64` 追加。判定コア `compare_pairs` へ共通化し tolerance 定数〈`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`〉は f32 版と完全共有）・`crates/backend-cpu/src/reduction.rs`（`CHUNK`／`unravel`／`checked_product` を `pub(crate)` 化のみ。関数本体は不変）
- **f32 ホットパス無変更の根拠**: `elementwise.rs`／`gemm.rs`／`gemm_blis/` は本イシューで一切変更していない（可視性変更すら加えていない）。`git diff --stat` は `reduction.rs` の可視性変更 3 行のみを示し、既存 f32 回帰テスト（本 sub 実装時点の `cargo test -p fandhe-ai-backend-cpu` 全件）が変更前と同じ結果で green であることを確認済み（具体的な件数はテスト追加の都度変わるため本節では固定値を書かない。#1697 の origin/main への追従〈#1786 の gather/scatter parity 追加等を取り込んだ rebase〉後も `cargo test -p fandhe-ai-backend-cpu` 全件 green・`cargo clippy --workspace --all-targets --all-features -- -D warnings` clean を再確認済み）
- **数値契約**: `gemm` は `f64::mul_add`・C の行（i）単位のみを rayon 並列化（BLIS packing・NT/TN fast path は対象外）。`sum`（全縮約）は `CHUNK`（4096）単位の `par_chunks` によるチャンク内逐次・チャンク間固定順序結合（アキュムレータは出力 dtype と同一の f64 のため downcast なし）。軸指定 reduction は出力要素側のみ並列・縮約軸は昇順逐次。`max` の単位元は `f64::NEG_INFINITY`・`f64::max`（NaN 非伝播は f32 版と同じ既知事項）。elementwise は f32 版と同型の 2 層構成（contiguous fast path → 非負 stride 読み `ReadOperandF64` → `Tensor::get` フォールバック）
- **parity ヘルパー**: `compare_f64`／`assert_parity_f64` は f32 版と同一の判定ロジック（`compare_pairs` 共通コア）・同一定数。`matmul_reference_fma_f64` は `gemm_f64` の bit 一致参照点（形状検証は `gemm::GemmError` の既存 variant を再利用）
- **テスト**: クレート内単体テスト 13 件（`typed_f64.rs` 手計算値中心）・統合テスト 8 件（`tests/typed_ops_f64_parity.rs`。決定的乱数入力・`PARALLEL_THRESHOLD` 超サイズでの並列経路・非 contiguous view・空縮約・shape エラー・f32 版との REQ-2 複合判定〈`cross_dtype_f64_vs_f32_composite_parity`〉）。`RAYON_NUM_THREADS=1` と既定並列度の両条件で bit 一致を確認済み
- **スコープ外**: f64 gemm の NT/TN 転置 fast path・BLIS 型 packing／2D 動的分配の f64 化（性能目的がないため参照実装のまま）・`elementwise.rs`／`reduction.rs` の `T: Element` 汎用化（#1698／#1699 完了後に共通化候補として検討）・`max` の NaN 非伝播セマンティクス（f32 版と同じ既知事項）・`Var`／`Tape`／VJP・facade 公開面への昇格（§7-7・§8。`docs/compat-api-scope.md` §5 手続き要）・f16／bf16（#1698／#1699）・CUDA（#1650）・Metal（#1651）
## 11. 実装記録（#1699・CPU `TypedOps<bf16>`）

- `impl Element for half::bf16`（受け入れ条件の一部）は #1687（PR #1784）で既に追加済みであり、本イシューでは追加不要（`crates/tensor-core/src/element.rs:90`・`Scalar` 実装は同ファイル 186〜189 行目）と確認したうえで着手した。
- 対象 8 演算（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`。§4.2 の最小集合）を `crates/backend-cpu/src/typed_bf16.rs`（新規）に実装し、`CpuBackendOps::typed_ops_bf16()` を `Some(self)` へオーバーライドした（`crates/backend-cpu/src/ops.rs`。`memory_ops` と同型の capability accessor パターン）。
- **実装方式**: bf16 専用カーネルは書かず、`promote`（`Tensor<bf16>` → `Tensor<f32>`。bf16→f32 は損失なし）→ 既存 f32 カーネル（`ops::CpuBackendOps` 経由の本番 GEMM・`elementwise::{add,mul,relu,exp,tanh}`・`reduction::{sum,max}`）→ `round_to_bf16`（`bf16::from_f32` による最近接偶数丸め。演算全体を通じて 1 回のみ）の 3 段ラッパーとした。これにより `crates/backend-cpu/src/{elementwise,gemm,reduction,parity}.rs`・`gemm_blis/`・`crates/tensor-core/**` は無変更（`git diff --stat main` で空を確認済み）。
- `sum` は f32 カーネル（`reduction::sum`）内部の f64 アキュムレータ契約（`.claude/rules/coding-rust.md`）にそのまま乗る「方式 (a)」を採用した。`reduction::CHUNK` 等を可視化して f64 中間値へ直接アクセスする「方式 (b)」は不要な可視性拡張を伴うため採らなかった。
- **判定契約**: bf16 は仮数 8 bit（1 ulp ≈ 2^-8）で `parity::RELATIVE_TOLERANCE`（1e-3）より粗いため、「参照値も出力 dtype（bf16）へ丸めてから比較する」契約を単体テストで固定した（`add_one_plus_two_pow_neg_eight_rounds_to_one_via_bf16_tie_to_even`）。異なる累積順序の f32 経路を乱数入力で両側 bf16 丸めして比較する検証は行っていない（丸め境界を跨いで 1 bf16 ulp ずれうるため）。`gemm` は本番 BLIS 経路と `parity::matmul_reference_fma` が bit 完全一致する既存契約（`tests/gemm_blis_parity.rs`）があるため、この問題を踏まずに済む。
- `reduce_error_to_backend_error`（`crates/backend-cpu/src/ops.rs`）を `pub(crate)` 化し、`typed_bf16` から再利用した（重複実装を避ける）。tolerance 定数・baseline は不変。
- テストは `typed_bf16.rs` 内 `#[cfg(test)]`（19 件。手計算値・空縮約・shape エラー・非 contiguous・NaN／±inf のクラス一致・accessor 経由 dispatch・小整数入力での f32 `BackendOps` との bit 一致を含む）で完結し、専用の統合テストファイルは追加していない（8 演算とも `CpuBackendOps` 単体で完結し、クレート境界を跨ぐ検証が不要だったため）。
- `cargo fmt --all --check`・`cargo clippy --workspace --all-targets --all-features -- -D warnings`・`cargo test -p fandhe-ai-backend-cpu --all-features`（440 passed）・`cargo test --workspace --all-features` は green（ワークスペース全体実行時に観測した `fandhe-ai --doc` の 1 件の失敗は、並行実行中の他セッションと共有 target ディレクトリ間のビルドキャッシュ競合〈E0460〉によるもので、本変更とは無関係であることを単独再実行で確認済み）。
- `Var`／`Tape`／VJP・facade 公開面昇格・`MemoryOps`／resident 系・カーネル融合・bf16 専用 SIMD は対象外のまま（§8）。f16（#1698）・CUDA bf16（#1704）・Metal bf16（#1706）は本イシューでは触れない。
## 12. 実装記録（#1703・CUDA `TypedOps<f64>`〈`Unsupported` 起点〉／`TypedOps<f16>`〈`run_f16` 結線〉）

`backend-cuda` に `TypedOps<f64>` と `TypedOps<half::f16>` を実装し、`CudaBackendOps::typed_ops_f64()`／`typed_ops_f16()`（既定 `None`）を `Some(self)` へオーバーライドして結線した（イシュー #1703・親 #1650）。承認根拠は #1697 と同じ親 #1649 コメント（2026-09-12 ユーザー承認）であり、CUDA 固有の追加承認事項はない（`gh issue view 1649/1650 --comments` で確認済み）。

- **f64（8 演算すべて `Unsupported`）**: `crates/backend-cuda/src/typed_f64.rs`（新規）。CUDA の f64 GEMM／elementwise／reduction カーネルには性能上の目的がない（§8「f64 GPU カーネル（CUDA SIMT）」）ため、8 演算すべてを driver 呼び出し前（`context_cache::cached_device` にすら触れない）に `BackendError::Unsupported` で fail-closed に拒否する。`crate::ops::CudaBackendOps` の `linalg_*` 未実装オーバーライドと同じ設計判断。
- **f16 `gemm`: `CudaGemmAuto::run_f16` への結線**: `crates/backend-cuda/src/typed_f16.rs`（新規）の `gemm` は、`context_cache::cached_gemm_auto`（新設。`cached_mma_tf32x3` と同型のプロセス内キャッシュ）経由で `CudaGemmAuto`（naive／tiled／WMMA／`mma.sync` 全 f16/f32 GEMM カーネルを保持する自動経路選択スイート。#1703 以前は `facade`／`backend-cuda::ops`／`bench-harness` のいずれからも到達不能だった）を取得し `run_f16` へそのまま委譲する薄いパススルー。カーネル選択ロジック自体（mma 優先→WMMA→tiled→naive のフォールバック連鎖・形状ゲート）は一切複製しない。`f32` 側 `BackendOps::gemm`（`context_cache::cached_gemm` 経由の tiled 固定経路）には一切影響しない（別スイート・別キャッシュキー）。
- **`CudaGemmAuto::new` のエラー契約変更（PR #1797 codex-review・Cursor Bugbot 指摘の是正）**: 当初実装は `CudaWmmaGemm::new`／`CudaMmaGemm::new` の構築失敗を無条件に `.ok()`／文字列化で握り潰していたため、`load_module`/`load_function` 経由の sticky な `CudaError::Driver` も対応外 capability と同じ fail-soft 扱いになり、外側の `with_driver_call`（`TypedOps<f16>::gemm` 経由）が観測できず ordinal が poison されないまま後続呼び出しへ進んでいた（codex-review P0 指摘）。是正後は `context_cache::is_sticky_driver_error`（`classify_cuda_result` の分類テーブルを再利用する薄いラッパー。新規 `pub(crate)` 関数）で sticky／operation-local を区別し、**sticky な `CudaError::Driver` のみ** `Self::new` 自体の `Err` として呼び出し元へ伝播する（`.claude/rules/security.md` A08 fail-closed）。`TensorCoreUnsupported`／`NvrtcUnavailable`／`Compile` に加え、operation-local な `CudaError::Driver`（`CUDA_ERROR_OUT_OF_MEMORY`／`CUDA_ERROR_NO_BINARY_FOR_GPU`／`CUDA_ERROR_INVALID_PTX` 等）は従来どおり `wmma`/`mma` を `None` としてキャッシュし `run_f16` を tiled へフォールバックさせる（全ての `CudaError::Driver` を無条件に早期 return すると、本来 `None` キャッシュされるべき operation-local な構築失敗まで `context_cache::cached_gemm_auto` へキャッシュされず、`TypedOps<f16>::gemm` の呼び出しのたびに NVRTC フルコンパイルを再試行する性能劣化になる。Cursor Bugbot 指摘）。この契約変更は `CudaGemmAuto::new`（本イシュー #1703 で新設した経路。§12 冒頭のとおり #1703 以前は本番経路から未参照）に閉じており、**`f32` 側 `BackendOps::gemm`（`context_cache::cached_gemm` 経由の tiled 固定経路）・`elementwise.rs`／`reduce.rs`／`gemm.rs`（f32 実装本体）はこの変更の影響を受けない**（別スイート・別キャッシュキーであり、`CudaGemmAuto::new` 自体を呼び出さないため）。この分類は `CudaWmmaGemm::new`（`gemm_wmma.rs`）・`CudaMmaGemm::new`（`gemm_mma.rs`）が内部で追加コンパイルする**opt／swizzle 変種の構築失敗**にも同じ規則で適用する必要がある（codex-review P0 再指摘・PR #1797）。両ファイルとも当初は opt／swizzle カーネル（`compile_wmma_f16_opt`〈`gemm_wmma.rs:157` 相当〉・swizzle 版 `compile_mma_f16`〈`gemm_mma.rs:373` 相当〉）の構築失敗を無条件に `.ok()`／文字列化で吸収し `wmma_f16_opt_error`／`swizzle_compile_error` へ退避するのみだったため、`load_module`/`load_function` 経由の sticky な `CudaError::Driver` がこの内側の吸収層で握り潰され、外側 `CudaGemmAuto::new`（および `with_driver_call`）が一切観測できないまま `Ok` を返す抜け道が残っていた。是正後は両ファイルとも `context_cache::is_sticky_driver_error` で opt／swizzle 構築失敗を分類し、sticky な場合のみ `Self::new`（`CudaWmmaGemm::new`／`CudaMmaGemm::new`）自体の `Err` として伝播する（`?` で伝播する base カーネル構築失敗と同じ扱いへ揃えた）。operation-local な失敗（対応外 capability・NVRTC コンパイル失敗・operation-local な `CudaError::Driver`）は従来どおり `wmma_f16_opt`／`mma_f16_swizzle` を `None` としてフォールバックさせる契約は不変。
- **f16 残り 7 演算（f32 昇格→既存 CUDA カーネル→1 回丸め）**: `add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` は `crates/backend-cpu/src/typed_f16.rs`（PR #1793・#1698）と同じ合成方式（`Tensor::host_slice` で昇格→既存 `<CudaBackendOps as BackendOps>` の f32 カーネル〈`elementwise::CudaElementwise`／`reduce::CudaReduce`〉へ委譲→`f16::from_f32` で 1 回丸め）を採る。各メソッドは構造的に `f16::from_f32(BackendOps::op(upcast(x)))` という不変条件を満たす。
- **可視性変更**: `crates/backend-cuda/src/ops.rs` の `CudaBackendOps::device_handle_raw`／`with_driver_call`（従来 private）を `pub(crate)` へ緩和し、`typed_f16::gemm` が同じ driver 呼び出し境界（`begin_driver_call`／`observe_cuda_result` による poison 検査）を再利用できるようにした。可視性以外の実装・契約は不変。
- **f32 ホットパス無変更の根拠**: `elementwise.rs`／`reduce.rs`／`gemm.rs`（f32 実装本体）は本イシューで一切変更していない。`gemm_auto.rs` は `CudaGemmAuto::new` のエラー契約変更（上記）とそれに伴う doc comment 是正のみで、`f32` 側 `BackendOps::gemm` が経由する `gemm.rs::CudaGemm`（`CudaGemmAuto` とは別の型）には触れていない。`gemm_wmma.rs`／`gemm_mma.rs` の変更点も opt／swizzle 変種構築時の sticky driver エラー伝播（上記）と doc comment 是正のみであり、両ファイルの基本版カーネル構築（`wmma_f16`／`mma_f16`。`f32` ホットパスからは元々未参照）・`f32` 側 `gemm.rs::CudaGemm`（TF32 opt を含む）には触れていない。`ops.rs` は accessor 2 メソッドの追加とコメント 1 箇所の是正・上記可視性変更のみ。
- **テスト**: クレート内単体テスト（`typed_f64.rs` 8 演算 `Unsupported` 確認・`typed_f16.rs` accessor／shape エラー／upcast-downcast 往復・`ops.rs` の poison-ordinal 拒否）・統合テスト（`tests/typed_ops_f64_contract.rs`。GPU 不要）・（`tests/typed_ops_f16_parity.rs`。GPU 不要な accessor／shape 検証テストに加え、GB10 実機 `#[ignore]` テスト 3 件: `gemm` が `CudaGemmAuto::run_f16` 直接呼び出しと bit 完全一致・CPU 参照実装〈f16 丸め後〉との REQ-2 複合判定・7 演算が CPU `BackendOps`〈f32〉を f16 へ丸めた値と REQ-2 複合判定で一致）。GB10 実機実測は本エージェント実行環境に CUDA 実機への到達手段がないため未実施のまま記入欄を残す。
- **数値契約**: f64 は driver 非接触（数値契約自体が存在しない）。f16 の `gemm` は `CudaGemmAuto::run_f16` の既存数値契約（GB10 実機実測済み。`docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` 等）をそのまま継承。残り 7 演算は「f32 へ昇格・f32 累算・最後に 1 回丸め」（CPU f16 と同一。`.claude/rules/coding-rust.md` の FMA 契約統一とは独立の軸）。
- **スコープ外**: CUDA `__half` 専用 elementwise／reduction カーネル（H2D 転送量削減の性能最適化）・f64 SIMT GEMM カーネル・bf16（#1704）・Metal（#1705）・`Var`／`Tape`／VJP・facade 公開面への昇格（§7-7・§8。`docs/compat-api-scope.md` §5 手続き要）・`MemoryOps`／`DeviceBuffer<T>` 常駐経路の dtype 多重化（段階 B）・`dispatch::DType` の拡張

**GB10 実機実測（2026-09-16 追記）**: `crates/backend-cuda/tests/typed_ops_f16_parity.rs`
（`--ignored`）を転送元コミット `3e43bbd0` で実行し **2 pass / 1 fail**。pass は gemm
（`run_f16` 結線）と型付きエラー契約、FAIL は
`typed_f16_elementwise_and_reduction_match_cpu_backend_ops_rounded` で、f32 昇格後に委譲する
`CudaBackendOps::sum` が reduction カーネル（`kernels_reduce.rs`）の NVRTC コンパイルエラー
（`identifier "INFINITY" is undefined`・compute_121）で失敗するため（f16 経路自体ではなく
委譲先 f32 reduction の不具合。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`
§3.1）。`typed_ops_f64_contract.rs` は `#[ignore]` なし（GPU 非依存の契約テスト）のため
実機では対象外。

## 13. 調査・実装記録（#1704・CUDA `TypedOps<bf16>`）

`backend-cuda` に `TypedOps<half::bf16>` を実装し、`CudaBackendOps::typed_ops_bf16()`（`crates/tensor-core/src/backend_ops.rs` の非破壊拡張 accessor。既定 `None`）を `Some(self)` へオーバーライドして結線した（イシュー #1704・親 #1650）。

### 12.1 調査結果（R1）: cudarc `DeviceRepr` 可用性は「可」

§5 の「bf16 × CUDA」セルは当初「cudarc 0.19.8 に bf16 向け `DeviceRepr` 実装があるか未検証」だったが、本イシューで次のとおり確定した（依存・feature の追加変更は不要でユーザー承認不要）。

| 事実 | 出典 |
|---|---|
| `unsafe impl DeviceRepr for half::bf16 {}` | `~/.cargo/registry/src/index.crates.io-*/cudarc-0.19.8/src/driver/safe/core.rs:1038`（`#[cfg(feature = "f16")]` 配下） |
| `unsafe impl ValidAsZeroBits for half::bf16 {}` | 同 `core.rs:990`（`#[cfg(feature = "f16")]` 配下） |
| workspace の `cudarc` は `features = ["driver", "nvrtc", "dynamic-loading", "cuda-13000", "f16"]` で `f16` feature 有効 | ルート `Cargo.toml:112-118`・`.claude/rules/deps-policy.md` CUDA 区分 |
| `clone_htod`／`clone_dtoh`／`alloc_zeros` は bf16 でそのまま使える | 同 `core.rs:1559,1592,1630` の `T: DeviceRepr`（＋ `ValidAsZeroBits`）境界 |

### 12.2 実装方式: ホスト側 bf16⇔f32 変換＋既存 f32 `BackendOps` カーネルへの委譲

`crates/backend-cpu/src/typed_f16.rs`（#1698）と同型の 3 段構成（昇格 → 既存 f32 経路への委譲 → 丸め）を採る。bf16 デバイス常駐で widen/narrow する専用カーネルは追加せず（性能最適化はスコープ外）、`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` の 8 演算とも既存 CUDA f32 経路（H2D・NVRTC カーネル・D2H を含む）へホスト側変換の後に委譲する。

- **`gemm` は `gemm_fp32_strict` へ委譲する（`gemm` ではない）**: CUDA の `BackendOps::gemm` は `crate::precision::gemm_precision()`（TF32／TF32x3 opt-in モード）で暗黙に精度が変わりうる。§6 の f32 厳密累算契約を保証するため、opt-in フラグを一切参照しない `gemm_fp32_strict`（CUDA オーバーライドは `ops.rs:1700` 付近）を使う。CPU 版が `gemm`（無印）を使っているのは CPU に TF32 の概念がなく `gemm`＝常に f32 厳密経路のためであり、CUDA ではそのまま踏襲しない
- **accessor `typed_ops_bf16` は無条件に `Some(self)` を返す**: `memory_ops` は `CudaMemory` 構築のため `device_handle()`（driver 初期化）を経由する必要があり driver 不在時に `None` へ縮退するが、`TypedOps<bf16>` の実体は `self` 自身で driver に一切触れる必要がない。実行時の CUDA 不在は各演算メソッド内部が `BackendError::CudaUnavailable` を返す形で伝える

### 12.3 実装ファイル

- `crates/backend-cuda/src/typed_bf16.rs`（新規。`upcast_bf16`／`downcast_f32`・`impl TypedOps<bf16> for CudaBackendOps`）
- `crates/backend-cuda/src/ops.rs`（`typed_ops_bf16` accessor 1 メソッド追加。**`TypedOps` を top-level `use` しない**——`self.add`／`self.relu` 等の f32 専用内部呼び出しが `impl TypedOps<bf16>` の同名メソッドと衝突し「multiple applicable items in scope」で解決不能になるため、戻り値型でのみ `fandhe_ai_tensor_core::TypedOps<bf16>` を完全修飾参照する。CPU 側 `ops.rs` が `TypedOps<f64>`／`TypedOps<f16>` を top-level `use` せず戻り値型でのみ完全修飾しているのと同じ回避策）
- `crates/backend-cuda/src/lib.rs`（`mod typed_bf16;` の追加）
- `crates/backend-cuda/tests/typed_ops_bf16_parity.rs`（新規。環境適応スモーク・層 1〈同一バックエンド内の構造的不変条件・run-to-run 決定性〉・層 2〈CPU f32 参照実装を bf16 丸めした値とのクロスバックエンド判定〉）

### 12.4 f32 経路無変更の根拠

`git diff --stat` は `crates/backend-cuda/src/{typed_bf16.rs,lib.rs,ops.rs}` と `tests/typed_ops_bf16_parity.rs` のみを示し、`kernels_*.rs`／`elementwise.rs`／`gemm*.rs`／`reduce.rs`／`memory.rs`／`precision.rs` は無変更（`ops.rs` の差分も `typed_ops_bf16` メソッド 1 つと import 1 行のみ）。既存 f32 回帰テスト（`cargo test -p fandhe-ai-backend-cuda --all-features`。lib 787 件・統合テスト群）が全 green であることを確認済み。

### 12.5 parity テストの設計（tolerance 不変で成立させるための構成）

参照値を bf16 へ丸めてから比較する方式は、丸め前の f32 値が両者で bit 一致している場合にのみ健全である（bf16 の 1 ulp は相対 2^-8 ≈ 0.0039 で `RELATIVE_TOLERANCE`＝1e-3 を超え、丸め境界をまたぐと不合格になりうる。§6 参照）。tolerance 変更は禁止のため、テストを二層で構成する。

- **層 1（主判定）**: 同一バックエンド内の構造的不変条件 `TypedOps::<bf16>::op(x) == bf16::from_f32(BackendOps::op_f32(f32(x)))` を bit 完全一致で検証し、run-to-run 決定性も確認する
- **層 2（クロスバックエンド）**: CPU の既存 f32 `BackendOps` をホスト側で bf16 丸めした値を参照とし、小整数（`[-8, 8]`）・小 K（16 以下）の入力に限定することで丸め境界またぎが構造的に起きない条件下で `fandhe_ai_backend_cpu::assert_parity`（複合判定 1e-3/1e-5・変更なし）を適用する。`exp`／`tanh` はこの技法（層 2）が使えないため層 2 の対象外とし、層 1（構造的不変条件の bit 完全一致）で検証する
- **注記**: CPU 側 `TypedOps<bf16>` はイシュー #1699・PR #1794 で実装済み・origin/main マージ済み（§11）。層 2 の参照値は本ファイル作成時点の実装同様 CPU の既存 f32 `BackendOps` をホスト側で丸めた値を使っており、`TypedOps::<bf16>::op` 同士の比較への差し替えは本イシューのスコープ外のまま残す（後続イシューへ引き継ぐ）
- **是正（PR レビュー指摘）**: 当初の `run_layer1_structural_checks`（`tests/typed_ops_bf16_parity.rs`）は `gemm`／`add`／`mul`／`relu`／`sum`／`max` のみを検証し `exp`／`tanh` を含んでいなかった（本節「層 1 のみで検証する」という記述と実装が不一致だった）。層 1 は同一バックエンド内比較のため丸め境界またぎの問題がなく `exp`／`tanh` の追加は技術的に容易であり、`run_layer1_structural_checks` へ両演算のチェックを追加して是正済み

### 12.6 GB10 実機実測

本エージェント実行環境に DGX Spark GB10 実機への到達手段がないため、`#[ignore]` テスト（`typed_ops_bf16_matches_across_shapes`・`typed_bf16.rs::tests` の `#[cfg(test)]` コンパイル時検査は Linux で実行済み）は未実測のまま記入欄を残す。実行コマンド:

```sh
cargo test -p fandhe-ai-backend-cuda --release --test typed_ops_bf16_parity -- --ignored --nocapture
```

**GB10 実機実測（2026-09-16 追記）**: `crates/backend-cuda/tests/typed_ops_bf16_parity.rs`
（`--ignored`）を転送元コミット `3e43bbd0` で実行し **0 pass / 1 fail**
（`typed_ops_bf16_matches_across_shapes`）。ホスト側 bf16⇔f32 変換後に委譲する
`CudaBackendOps::sum` が reduction カーネルの NVRTC コンパイルエラー（`INFINITY` 未定義）
で失敗するため、gemm／elementwise を含む同一テスト内の後続比較に到達しない（bf16 変換
経路自体の不一致は未観測。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.1）。
判定は FAIL のまま記録し、tolerance／baseline は変更しない。

### 12.7 スコープ外

bf16 デバイス常駐経路（bf16 のまま H2D し device 側で widen/narrow する 2 カーネル方式）・bf16 `mma.sync` GEMM カーネル（cc ≥ 8.0。§8「#1650 の実装事項」）・CUDA `TypedOps<f64>`／`TypedOps<f16>`（#1703）・Metal（#1651）・`Var`／`Tape`／VJP の dtype 一般化・facade 公開面への昇格・GB10 実機実測（§12.6）。CPU bf16 は #1699・PR #1794 で実装済み・origin/main マージ済み（§11）

## 14. 実装記録（#1705・Metal `TypedOps<f64>`〈恒久 `Unsupported`〉／`TypedOps<f16>`〈`dispatch_f16_auto_unverified` 結線〉）

`backend-metal` に `TypedOps<half::f16>` を実装し、`MetalBackendOps::typed_ops_f16()`（`crates/tensor-core/src/backend_ops.rs` の非破壊拡張 accessor。既定 `None`）を `Some(self)` へオーバーライドして結線した（イシュー #1705・親 #1651）。承認根拠は #1697／#1703 と同じ親 #1649／#1651 コメント（2026-09-12 ユーザー承認）であり、Metal 固有の追加承認事項はない。

### 14.1 `TypedOps<f64>` は恒久 `Unsupported`・accessor `None` のまま（実装なし）

§5 の実現可否表のとおり、MSL には `double` 型が存在せず GEMM 全体を `f64` 精度で動かす手段が構造的にない（`soft_f64.rs` の 64bit 整数エミュレーションは bias 勾配縮約というスカラー累算専用であり GEMM の代替にならない）。CUDA `typed_f64.rs`（`Some(self)` を返しつつ 8 演算すべて `Unsupported` で明示応答する方式）とは異なり、**Metal は `impl TypedOps<f64> for MetalBackendOps` 自体を追加せず、`typed_ops_f64()` accessor もオーバーライドしない**（trait 既定の `None` のまま）。理由は、CUDA の f64 が「driver 非接触の設計上の選択」（性能上の目的がないため実装しないだけで実装自体は可能）であるのに対し、Metal の f64 は「MSL の型システム上そもそも構築不能」という異なる性質のブロッカーであり、accessor `None`（capability 不在の明示）の方が「実装したが何もできない」より実情に忠実なため。呼び出し側は `typed_ops_f64().is_none()` を見て fail-closed に扱える（`memory_ops` 等の既存 capability accessor と同型の契約）。この判断は本イシュー着手前に設計 §5 で既に固定されていたものを実装で追認したものであり、実装フェーズでの新規判断ではない。

### 14.2 `TypedOps<f16>::gemm`: `dispatch_f16_auto_unverified` への内部結線

`crate::gemm::MetalGemm::dispatch_f16_auto_unverified`（動的タイル選択の f16 自動経路。イシュー #798・`_unverified` suffix・`#[doc(hidden)]` を維持）は、REQ-2 統一複合判定の検証を担う `tests/gemm_f16_auto_parity.rs`（`#[ignore]`。#799 実機セッションで M4 Max 実機 8/8 PASS 記録済み。`docs/perf/metal-f16-vs-mps-f16.md:378`）を持ちながら、本イシュー以前は `facade`／`ops::MetalBackendOps`（f32 `BackendOps::gemm`）／`dispatch_backend_auto`（真の production 自動経路）のいずれからも到達不能だった。

`crates/backend-metal/src/typed_f16.rs::gemm` は `context_cache::cached_context`／`cached_gemm`（プロセス内キャッシュ。イシュー #930）経由で `MetalGemm` を取得し `dispatch_f16_auto_unverified` へそのまま委譲する薄いパススルーであり、`tile::select_for_device` のタイル選択ロジック自体は一切複製しない。到達経路は `Tensor<f16>` という型でのみ決まる（`TypedOps<f16>::gemm` を呼ぶには `Tensor<f16>` を構築する必要がある。REQ-11 整合）ため、`f32` 側 `BackendOps::gemm`（`dispatch_auto`／`dispatch_strided_bias_act_prepared` 経由）から `dispatch_f16_auto_unverified` へ暗黙に迂回する経路は存在しない（fail-closed）。`_unverified` suffix・`#[doc(hidden)]` の解除は #1651 の承認事項 §7-4 が本イシューでは未承認のため維持し、`gemm.rs` の doc comment にその旨を追記した（本体・シグネチャ・属性は不変）。

### 14.3 残り 7 演算: f32 昇格 → 既存 Metal `BackendOps` カーネル → 1 回丸め

`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` は CPU（#1698）・CUDA（#1703）と同じ 3 段構成（`Tensor::host_slice` で昇格 → 既存 `<MetalBackendOps as BackendOps>` の f32 カーネルへ委譲 → `f16::from_f32` で 1 回丸め）を採る。各メソッドは構造的に `f16::from_f32(BackendOps::op(upcast(x)))`（要素ごと bit 一致）という不変条件を満たす。

**`max` のみ `Unsupported` を継承する（`sum` はイシュー #1896 で結線済み）**: `ops::MetalBackendOps::max`（f32）は reduction カーネル未実装のため常に `BackendError::Unsupported` を返す。`typed_f16.rs::max` は上記 3 段構成をそのまま適用するだけで、委譲先が常に `Unsupported` を返すため構造的に同じ結果になる——ホスト側で reduction を計算して `Unsupported` を偽装しない（バックエンド内部で CPU 計算を隠す silent fallback を避ける方針。レイヤリング上、ホストフォールバックは `autodiff` 側の責務）。`ops::MetalBackendOps::sum` はイシュー #1896 で `crate::reduce::MetalReduce` へ結線されたため、`typed_f16.rs::sum` は本ファイルの変更なしにそのまま f32 経路の成功結果（CPU 参照実装と bit 完全一致）を `f16::from_f32` で丸めた値を返すようになった（`docs/backend-metal-reduce-sum-design.md` §9）。`max`（`min` も含む）が将来実装されれば同様に `typed_f16.rs` は変更なしでそのまま有効になる。

### 14.4 実装ファイル

- `crates/backend-metal/src/typed_f16.rs`（新規。`upcast_f16`／`downcast_f32`・`impl TypedOps<f16> for MetalBackendOps`・`#[cfg(test)]` 5 件）
- `crates/backend-metal/src/lib.rs`（`#[cfg(target_os = "macos")] pub(crate) mod typed_f16;` の追加・モジュール doc 2 箇所への追補）
- `crates/backend-metal/src/ops.rs`（`typed_ops_f16` accessor 1 メソッド追加。**`TypedOps` を top-level `use` しない**——`self.add`／`self.relu` 等の f32 内部呼び出しが `impl TypedOps<f16>` の同名メソッドと衝突するため、戻り値型でのみ `fandhe_ai_tensor_core::TypedOps<half::f16>` を完全修飾参照する。CPU／CUDA 側 `ops.rs` と同じ回避策）
- `crates/backend-metal/src/gemm.rs`（`dispatch_f16_auto_unverified` の doc comment のみ追記。関数本体・シグネチャ・属性は不変）
- `crates/backend-metal/tests/typed_ops_f16_parity.rs`（新規。`#![cfg(target_os = "macos")]`。実機非依存の accessor／shape 検証・`#[ignore]` 実機依存の L1 gemm bit 一致・L2 gemm／elementwise vs CPU 参照・零次元形状のデバイス到達確認）
- `crates/backend-metal/tests/typed_ops_source_evidence.rs`（新規。cfg なし・Linux CI 常時実行。`typed_ops_f64` 非オーバーライド・`TypedOps<f64>` 非実装・`typed_ops_f16` accessor 結線・`dispatch_f16_auto_unverified` 呼び出し証跡・暗黙 f32 フォールバック不在を文字列検査で固定。doc comment 中の言及による誤検出を避けるため `//` コメント行を除去してから検査する）

### 14.5 f32 経路無変更の根拠

`git diff --stat` は `crates/backend-metal/src/{typed_f16.rs（新規）,lib.rs,ops.rs}` と `gemm.rs`（doc comment のみ）・`tests/typed_ops_f16_parity.rs`（新規）・`tests/typed_ops_source_evidence.rs`（新規）のみを示し、`shaders/`・`elementwise.rs`・`context_cache.rs`・`crates/tensor-core/**` は無変更（`ops.rs` の差分も `typed_ops_f16` accessor 1 メソッド＋ doc 1 文、`gemm.rs` の差分も doc comment 追記のみ）。既存 f32 回帰テスト（`cargo test -p fandhe-ai-backend-metal --all-features`。Linux 実行可能な単体・統合テストがすべて green）・`cargo fmt --all --check`／`cargo clippy -p fandhe-ai-backend-metal --all-targets`（backend-metal 由来の警告・エラーは 0 件。`fandhe-ai-backend-cuda` 側の pre-existing dead-code エラー〈本 PR 変更対象外・stash 比較で main 上でも再現することを確認済み〉が `--all-features` 経由でビルドグラフに混入し `cargo clippy` 全体を止めるが、backend-metal 由来の指摘は 0 件）・`make check-cross-metal-tests` 相当（`cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`。green）・`RUSTDOCFLAGS="-D warnings" cargo doc`（workspace 全体・backend-metal／backend-cpu の aarch64-apple-darwin クロスとも green）で非後退を確認済み。

### 14.6 テスト構成・実機実測の記入欄

- `typed_f16.rs::tests`（accessor 契約・shape 検証・`max` の `Unsupported` 継承・範囲外 `dim` の `sum` `ShapeMismatch`・upcast/downcast 往復。イシュー #1896 で `sum` 関連テストを再構成済み）・`tests/typed_ops_f16_parity.rs` の非 `#[ignore]` 部（accessor・shape 検証・零次元形状は `#[ignore]` 側）はいずれも `cfg(target_os = "macos")`／`#![cfg(target_os = "macos")]` 限定のため、本エージェント実行環境（ネイティブ Linux）ではコンパイル対象に入らずテスト実行・pass 確認はできない。本エージェント環境で実施できたのは `cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`（クロス型検査。green）に留まり、実行・pass 確認は Mac セッションへ申し送る。`tests/typed_ops_source_evidence.rs` 5 件は cfg なしのため Linux CI で実際に実行され green（`cargo test -p fandhe-ai-backend-metal --test typed_ops_source_evidence` で確認済み）
- `#[ignore]`（Apple Silicon 実機依存）: `tests/typed_ops_f16_parity.rs` の L1 gemm bit 一致・L2 gemm／elementwise vs CPU 参照 rounded・零次元形状のデバイス到達確認。本エージェント実行環境に Apple Silicon 実機への到達手段がなく未実施のまま記入欄を残す（Mac セッションへ申し送り）。実行コマンド:

```sh
cargo test -p fandhe-ai-backend-metal --release --test typed_ops_f16_parity -- --ignored --nocapture
```

**M4 Max 実機実測（2026-09-16 追記）**: `crates/backend-metal/tests/typed_ops_f16_parity.rs`
（`--ignored`）を origin/main `3e43bbd0` で実行し **4 pass / 0 fail**（REQ-2 統一複合判定。
ログ `docs/perf/logs/metal-realdevice-phase2-2026-09-16/backend-metal_typed_ops_f16_parity.log`）。
tolerance 変更なし。

## 15. 調査・実装記録（#1706・Metal `TypedOps<bf16>`）

`backend-metal` に `TypedOps<half::bf16>` を実装し、`MetalBackendOps::typed_ops_bf16()`（既定 `None`）を `Some(self)` へオーバーライドして結線した（イシュー #1706・親 #1651）。

### 15.1 (a) と (b) の分離（本イシューの核心）

§5「bf16 × Metal」は当初「未検証（MSL `bfloat` 型・`simdgroup_matrix` 対応をコンパイルプローブで確認する必要がある）」としていたが、これは 2 つの独立した事実を混同していた。

- **(a) `TypedOps<bf16>` の実装可否**: 兄弟イシュー #1704（CUDA bf16）・#1699（CPU bf16）はいずれもホスト側 bf16⇔f32 変換＋既存 f32 `BackendOps` カーネルへの委譲で実装しており、これは MSL `bfloat` 型の可用性に**依存しない**。§6 の数値契約（f16／bf16 は f32 昇格・f32 累算・最後に 1 回丸め）自体が最適化ではなく契約であるため、Metal も同じ 3 段ラッパー方式で **実装済み**（本イシューで解決）
- **(b) MSL `bfloat`／`simdgroup_bfloat8x8` の実機可用性**: これが決めるのは将来のデバイス常駐ネイティブ bf16 経路（bf16 のまま H2D・`simdgroup_bfloat8x8` GEMM 等）の実現可能性であり、`crate::typed_bf16_probe_diag_tests` のコンパイルプローブへ切り出す（**未実測**。本段階のスコープ外）

`TypedOps<bf16>` の accessor が `None` から `Some(self)` へ変わる分岐は (b) の結果に依存しない。

### 15.2 実装方式: ホスト側 bf16⇔f32 変換＋既存 f32 `BackendOps` カーネルへの委譲

`crates/backend-cuda/src/typed_bf16.rs`（#1704）・`crates/backend-cpu/src/typed_bf16.rs`（#1699）と同型の 3 段構成（昇格 → 既存 f32 経路への委譲 → 丸め）を採る。bf16 デバイス常駐で widen/narrow する専用 MSL カーネルは追加しない（性能最適化はスコープ外）。`gemm`／`add`／`mul`／`relu`／`exp`／`tanh` の 6 演算は既存 Metal f32 経路（`gemm.rs`／`elementwise.rs`）へ委譲し、`sum`／`max` は Metal f32 `BackendOps` が GPU カーネル未実装で常に `Unsupported` を返すためそのまま伝播する（f32 版より高機能にしない）。

- **`gemm` は `gemm_fp32_strict` へ委譲する（`gemm` ではない）**: Metal の `BackendOps::gemm` は `gemm::MetalGemm::dispatch_auto`（split-K opt-in 経路。既定 ON）へ委譲する。`gemm_fp32_strict` の Metal オーバーライドは既定実装（`self.gemm(a, b)` へ委譲）のままのため、現時点では両者の呼び出しは実質同一経路だが、CUDA 版（TF32 opt-in を明示的に避ける）との対称性・「opt-in 精度モードを参照しない」意図を明示するため `gemm_fp32_strict` を選ぶ
- **accessor `typed_ops_bf16` は無条件に `Some(self)` を返す**: `TypedOps<bf16>` の実体は `self`（ZST）自身でありデバイス初期化に一切触れない。実行時の Metal 実機不在・演算失敗は各演算メソッド内部（`BackendOps::add` 等・`gemm_fp32_strict`）が型付きエラーで返す

### 15.3 実装ファイル

- `crates/backend-metal/src/typed_bf16_convert.rs`（新規。cfg なし。`upcast_bf16`／`downcast_f32`。Linux で単体テスト実行可能）
- `crates/backend-metal/src/typed_bf16.rs`（新規。macOS 限定。`impl TypedOps<bf16> for MetalBackendOps` 8 演算）
- `crates/backend-metal/src/typed_bf16_probe_diag_tests.rs`（新規。cfg(all(test, macos))・全 `#[ignore]`・非 gating。(b) の調査専用。P0 デバイス属性／P1 bfloat スカラー／P2 `simdgroup_bfloat8x8` MMA／P3 `simdgroup_store` bfloat のコンパイルプローブ＋P4 roundtrip 数値スモーク）
- `crates/backend-metal/src/ops.rs`（`typed_ops_bf16` accessor 1 メソッド追加）
- `crates/backend-metal/src/lib.rs`（3 モジュールの登録）
- `crates/backend-metal/tests/typed_ops_bf16_parity.rs`（新規。層 1〈同一バックエンド内の構造的不変条件・run-to-run 決定性〉・層 2〈CPU f32 参照実装を bf16 丸めした値とのクロスバックエンド判定〉。`#![cfg(target_os = "macos")]` + `#[ignore]`）

### 15.4 f32 経路無変更の根拠

`git diff --stat` は上記新規ファイルと `ops.rs`／`lib.rs` の追記行のみを示し、`gemm.rs`／`elementwise.rs`／`shaders/**`／`pipeline.rs`／`tensor-core/**` は無変更。

### 15.5 parity テストの設計

CUDA §13.5 と同じ理由（bf16 の 1 ulp は相対誤差 `RELATIVE_TOLERANCE`＝1e-3 を超えうる）で、tolerance を変更せず二層構成とする。

- **層 1（主判定）**: 同一バックエンド内の構造的不変条件 `TypedOps::<bf16>::op(x) == bf16::from_f32(BackendOps::op_f32(f32(x)))` を bit 完全一致で検証（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`。`sum` はイシュー #1896 で `crate::reduce::MetalReduce` へ結線され層 1 の対象化済み。`max` は `Unsupported` のため引き続き対象外）。run-to-run 決定性も確認する
- **層 2（クロスバックエンド）**: CPU の既存 f32 `BackendOps` をホスト側で bf16 丸めした値を参照とし、小整数（`[-8, 8]`）・小 K（16 以下）の入力に限定して `fandhe_ai_backend_cpu::assert_parity`（複合判定 1e-3/1e-5・変更なし）を適用する（`gemm`／`add`／`mul`／`relu`）。`exp`／`tanh` はこの技法が使えないため層 2 の対象外とし層 1 で検証する

### 15.6 M4 Max 実機実測

本エージェント実行環境に Apple Silicon 実機への到達手段がないため、`#[ignore]` テスト（`typed_ops_bf16_matches_across_shapes`・`typed_ops_bf16_gemm_shape_mismatch_returns_typed_error`。`typed_bf16_probe_diag_tests` の P0〜P4）は未実測のまま記入欄を残す。実行コマンド・保存先は `docs/perf/logs/metal-typed-bf16-probe-1706/README.md` を参照。

```sh
cargo test -p fandhe-ai-backend-metal --release --test typed_ops_bf16_parity -- --ignored --nocapture
```

**M4 Max 実機実測（2026-09-16 追記）**（origin/main `3e43bbd0`・共有負荷下・
`docs/perf/logs/metal-typed-bf16-probe-1706/`〈`typed_bf16_probe.log`・
`typed_ops_bf16_parity.log`〉）:

- `typed_ops_bf16_parity.rs`（`--ignored`）: **2 pass / 0 fail**
  （`typed_ops_bf16_matches_across_shapes`〈REQ-2〉・
  `typed_ops_bf16_gemm_shape_mismatch_returns_typed_error`）。(a) の実装は実機でも成立
- `typed_bf16_probe_diag_tests`（非 gating。コンパイル可否の事実記録のみ）:
  - P0 `p0_device_attributes`: `device_architecture=applegpu_g16s`・
    `supports_apple7/8/9=true`・`supports_metal3=true`
  - P1 `p1_bfloat_scalar_compile_probe`: `bfloat` スカラー／`bfloat4`／変換の
    コンパイル **ok**（既定言語版・`Version3_1` の両条件）
  - P2 `p2_simdgroup_bfloat_mma_compile_probe`: `simdgroup_bfloat8x8` の
    `simdgroup_load`／`simdgroup_multiply_accumulate` のコンパイル **ok**
    （既定・3.1 とも）。**コンパイル可否のみ**で、実行時の数値・性能は未検証
  - P3 `p3_simdgroup_store_bfloat_compile_probe`: `simdgroup_store(float8x8, device bfloat*)`
    は **error**（`deduced conflicting types for parameter T (float vs. bfloat)`。
    f16 版〈#380〉と同様の見込みどおり。型を揃えた store が必要）
  - P4 `p4_bfloat_roundtrip_numeric_smoke`: 256 要素中 **256 一致・0 不一致**
    （ホスト `half::bf16::from_f32` と bit 一致）
- (b) の結論: MSL `bfloat`／`simdgroup_bfloat8x8` は本機（M4 Max・macOS 26.6.2）で
  コンパイル可能であり、デバイス常駐ネイティブ bf16 経路の実現可能性は
  「コンパイルレベルでは可」。採否判定・性能実測は本記録の対象外（§15.7）

### 14.7 スコープ外

Metal `half` 専用 elementwise／reduction カーネル（H2D 転送量削減の性能最適化）・f16 NT/TN strided 入口（非 contiguous view の性能最適化）・`_unverified`／`#[doc(hidden)]` の解除（§7-4 の別途承認事項）・Metal f32 `sum`／`max` reduction カーネル自体の実装（実装されれば f16 版は自動的に有効化される。out-of-scope-tracking.md 対象。後続イシュー起票の要否はユーザー承認後に判断）・bf16（#1706）・`Var`／`Tape`／VJP・facade 公開面への昇格・`MemoryOps`／`DeviceBuffer<f16>` 常駐経路（段階 B）・M4 Max 実機実測（§14.6）。

**#1895 で追記**: Metal f32 `sum`（全要素・単一軸）reduction カーネルは `crate::reduce`（`MetalReduce`）として実装済み（`docs/backend-metal-reduce-sum-design.md`）。**#1896 で追記**: `MetalBackendOps::sum` を `context_cache::cached_reduce` 経由で結線済み・`typed_f16`／`typed_bf16` の `sum` はコード変更なしで自動有効化された（同 doc §9）。

### 15.7 スコープ外

MSL `bfloat`／`simdgroup_bfloat8x8` を用いるデバイス常駐ネイティブ bf16 経路（(b) の実測が可の場合の後続候補）・Metal f32 `max` reduction カーネル自体（未実装。`sum` はイシュー #1896 で結線済み）・Metal `TypedOps<f64>`（恒久 `Unsupported`）／`TypedOps<f16>`（#1705）・`Var`／`Tape`／VJP・facade 公開面への昇格・M4 Max 実機実測（§15.6）。CPU bf16 は #1699・CUDA bf16 は #1704 で実装済み・origin/main マージ済み。

## 16. facade 公開面への昇格（イシュー #1939）

### 16.1 承認記録

2026-09-17・issue #1939 コメントの承認依頼「選択肢 A」をユーザーが承認。
公開する facade 新規公開面は以下の 4 件に限定する（`docs/compat-api-scope.md`
§5 経路 2 の手続き）。

1. `pub use fandhe_ai_tensor_core::{Scalar, ScalarDType, TypedOps};`
2. `fandhe_ai::Tape::typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>>`
3. `fandhe_ai::Tape::typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>>`
4. `fandhe_ai::Tape::typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>>`

付随制約（承認事項）: 勾配なし（`Var`／autograd 非経由）・8 演算限定
（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`）・
`half::f16`／`half::bf16` は利用者が `fandhe_ai_tensor_core` 経由で
直接名指しする前提とし、facade は `half` 自体を再エクスポートしない。

### 16.2 到達経路

```
facade::Tape::typed_ops_f64/_f16/_bf16
  → autodiff::Tape::typed_ops_f64/_f16/_bf16（新設。§4.1 で確定した
    `BackendOps::typed_ops_*` capability accessor への狭い委譲。
    `Tape::device`〈イシュー #1614〉と同型のパターン）
  → BackendOps::typed_ops_f64/_f16/_bf16（§7 の既存 capability accessor。
    既定 `None`）
```

autodiff 側 3 アクセサ（`crates/autodiff/src/tape.rs`）は **内部クレート
の到達経路**であり facade 公開面の追加ではない。`Tape::ops()` 自体は
`pub(crate)` のまま（REQ-12。任意 `BackendOps` 実装を注入できる公開 API
を設けない）で、返すのは `&dyn TypedOps<T>` の不変借用のみ。

### 16.3 バックエンド × dtype の対応状況（HEAD 時点）

| バックエンド | `f64` | `f16` | `bf16` |
|---|---|---|---|
| CPU | `Some`（#1697。8 演算とも動作） | `Some`（#1698。ソフトウェア変換） | `Some`（#1699。f32 カーネル再利用） |
| CUDA | `Some`（#1703。8 演算とも `Err(Unsupported)`） | `Some`（#1703。`gemm` はネイティブ f16 経路・残り 7 演算は f32 昇格） | `Some`（#1704。ホスト側変換＋既存 f32 経路委譲） |
| Metal | `None`（恒久。#1705） | `Some`（#1705。`dispatch_f16_auto_unverified` 内部結線） | `Some`（#1706 (a)。ホスト側変換＋既存 f32 経路委譲。(b) MSL ネイティブ経路は未採否） |

### 16.4 検証

facade のみを import する統合テスト（`crates/facade/tests/api_surface.rs::
typed_ops_types_are_reachable_via_facade`）で、CPU バックエンドにおける
`f64::add`・`f16::relu`・`bf16::relu` の実行結果がホスト参照実装と
一致することを確認済み（f64 は `to_bits` 比較で bit 完全一致・f16／bf16
は厳密に表現可能な値の一致）。ソース走査による否定ガード
（`facade_does_not_reexport_half`・`facade_cargo_toml_does_not_depend_on_half`）
で承認範囲逸脱（`half` の再エクスポート・facade 直接依存化）を機械固定。

### 16.5 スコープ外

`Var`／`Tape` autograd 経由の dtype 一般化（§8）・facade での `half`
再エクスポート・8 演算以外の dtype 別演算の公開・`CastOps` の公開・
`AmpDType` の `ScalarDType` への統合リファクタ（`AmpDType` 自体は
本 issue の対象外のまま不変）。CUDA／Metal 実機での facade 経由
`typed_ops_*` 実測は本実装エージェント実行環境に到達手段がなく未実施
（各バックエンドの `TypedOps<T>` 実装自体は §12〜§14 で個別に検証・
申し送り済み）。
