# デバイス常駐更新の設計（イシュー #934・#933 ツリー第 1 段）

## 0. 位置づけ・スコープ

本ドキュメントは #933（親イシュー「学習ループのパラメータ更新をデバイス常駐化し
ホスト往復を削減する」）の 3 分解のうち **設計のみ**を担当する #934 の成果物である。
実装は #935、数値一致（parity）テスト・ベンチ非後退確認は #936 が担う。本文書は
両イシューがそのまま着手できる具体度（trait シグネチャ・契約・検証基準）で
更新経路（§2）・所有権（§3）・数値一致契約（§4）の 3 点を定める。

コード（`crates/`）・spec（`docs/spec/`）・CI・依存は本イシューでは一切変更しない。

## 1. 背景・実測根拠

フレームワーク横並びベンチ（PR #915・`scripts/bench/framework-compare/results/summary.md`・
計測 2026-08-28）で、MLP 学習 1 ステップ（784→256→10・ReLU・バッチ 64・MSE・SGD lr=0.01）が
以下のとおり fandhe-ai と candle/Burn とで 1 桁以上の差が実測されている。

| デバイス | fandhe-ai | candle | burn | 出典行 |
|---|---|---|---|---|
| CUDA（環境 2・DGX Spark GB10） | 2.418 s | 268.1 µs | — | summary.md:165-166 |
| CPU（環境 1・Apple M4 Max） | 18.185 ms | 797.5 µs | 626.5 µs | summary.md:72-74 |
| Metal（環境 1） | 48.845 ms | 751.8 µs | 1.606 ms | summary.md:75-77 |

summary.md 自身の注記（191 行目）が指摘するとおり、この差の支配要因は 2 つに
分解できる:

- **(a) 毎ステップの tape（CUDA コンテキスト等）初期化コスト**: fandhe-ai の
  計測プロトコルは「計測ごとに新しい `tape_for(Device::Cuda(0))` を作る」ため、
  N 非依存の約 440〜460 ms（CUDA。summary.md:191）の固定オーバーヘッドが毎回計上
  される。この要因は phase-2 #922 系（#929/#930 完了・#931 設計中）が担当する。
- **(b) パラメータ更新のホスト往復**: 本ドキュメント（#933 ツリー・#934/#935/#936）
  が担当する要因。(a) を除去しても本要因は独立に残る（tape を使い回しても、
  学習ステップごとの param/grad のホスト⇔デバイス転送は別途発生する）。

本ドキュメントは (b) のみを扱う。(a) との合算値（2.418 s 等）を (b) 単独の
定量的な削減目標として引用しない（両要因が未分離のため）。(b) の定量的な
影響度は #935 実装後、#936 のベンチで tape 再利用モード（`--mode reuse`。
`gemm` タスクのみ現状対応）に相当する形の計測を検討する。

## 2. 現状整理

| 項目 | 現状 | 根拠 |
|---|---|---|
| パラメータ保持 | `nn::Linear` がホスト `Tensor<f32>` を所有。毎ステップ `Linear::from_parameters` で新規構築（イミュータブルな丸ごと差し替え運用） | `crates/autodiff/src/nn/linear.rs`・`crates/autodiff/src/optim/sgd.rs:20` コメント |
| 更新計算 | `optim::Sgd::step` がホスト側 `Tensor<f32>` ループで `torch.optim.SGD` の更新則（weight_decay → momentum → nesterov → 減算の式順序）を計算し、更新後 `Vec<Tensor<f32>>` を返す関数型 API | `crates/autodiff/src/optim/sgd.rs:1-23`（アルゴリズム引用）・`:183-200`（`step` シグネチャ・位置対応契約） |
| 書き戻し | `Sequential::apply_parameters`（two-pass: 1 パス目で全層の shape 完全一致検証込み `Linear` 再構築、2 パス目で代入。#426 の置換前 shape 完全一致契約） | `crates/facade/src/compat/sequential.rs:229-` |
| デバイス転送抽象 | `MemoryOps`（`alloc_zeroed`/`upload`/`download`）と `DeviceBuffer<f32>`（`Box<dyn BufferHandle>` 経由の不透明ハンドル・RAII 解放・空テンソル契約・download 同期契約）は `tensor-core::buffer` に実装済みだが、`BackendOps` の supertrait としては未結線（TASK-1.9c 時点で意図的に未結線。`backend_ops.rs` 冒頭コメント。本設計〈§3.1〉は supertrait 化ではなく `mem: &dyn MemoryOps` 明示引数で結線する） | `crates/tensor-core/src/buffer.rs:1-13`・`crates/tensor-core/src/backend_ops.rs:10-27` |
| カーネル入口 | `BackendOps` はホスト `Tensor<f32>` 入出力契約。CPU 実装は追加転送コストなし、CUDA/Metal 実装は各メソッド内で H2D（`upload` 相当）→ カーネル → D2H（`download` 相当）を完結させる。GEMM 以外の一部カーネル（GPU 側 elementwise/reduction の一部）は `BackendError::Unsupported` fail-safe | `crates/tensor-core/src/backend_ops.rs:28-40`（trait 定義冒頭コメント）・`:83-103`（メソッド一覧） |
| 数値一致 | バックエンド間統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」・FMA 契約（CPU: `f32::mul_add`、GPU: 既定 FMA 契約） | `.claude/rules/coding-rust.md` |
| 隣接イシュー | #931（デバイスハンドル再利用の公開 API 設計。phase-2）・#932（optimizer facade 公開の設計判断）・#935（本設計の実装）・#936（parity・ベンチ非後退確認） | gh issue 参照 |

現行の学習 1 ステップの転送回数（概算・per-step）:

1. forward: 各層で `BackendOps::gemm`/`add`/`relu` 呼び出しのたびに引数テンソルが
   H2D、結果が D2H（CUDA/Metal 実装内部で毎回発生。`Tensor<f32>` 契約のため）
2. backward（`Tape::backward`）: 同様に演算ごとの転送 + 最終的にホスト `Tensor` の
   勾配列を返す
3. `Sgd::step`: ホスト `Tensor<f32>` 上で更新計算（デバイス転送なし。ただし
   入力の `params`/`grads` が既にホストにある前提＝上記 1・2 で download 済み）
4. `Sequential::apply_parameters` → `Linear::from_parameters`: 更新後パラメータを
   新規 `Linear` として保持（次ステップの forward で再度 upload される）

つまり **パラメータそのものは「ステップ開始時に前ステップの結果としてホストへ
download 済み → 今回ステップの forward で再度 upload」という往復を毎ステップ
繰り返す**。これが本ドキュメントが解消対象とする (b) の具体的な構造である。

## 3. 更新経路（design decision 1）

### 3.1 採用方針: `MemoryOps` は明示引数として渡す（supertrait 化はしない）+ デフォルトメソッド + 常駐 forward 入口

`tensor-core::buffer.rs` は当初から「`MemoryOps` は `BackendOps` の
supertrait となる想定（TASK-1.9c では結線しない）」と明記していた
（`buffer.rs:11-13`）。**本設計はこの保留を supertrait 化では解消しない**
（旧稿はここで `BackendOps: MemoryOps` の supertrait 化を採用と記載して
いたが、レビュー指摘 #934〈line 157〉のとおり、`fandhe-ai-tensor-core`
は 2026-08-23 に crates.io 公開済みの crate であり `BackendOps` は
Rust の可視性上真に公開の trait である。supertrait を追加すると、
`impl BackendOps for X` を持つ既存の外部実装型はすべて追加で
`impl MemoryOps for X` を要求されコンパイル不能になる——これは
「既存メソッドのシグネチャ変更なし」で担保できる非破壊性の範囲を
超え、trait 定義そのものへの破壊的変更である。`.claude/rules/security.md`
A08・AGENTS.md の公開 API 非破壊契約に抵触するため不採用とする）。

代わりに、`sgd_step_device`／`linear_forward_resident` の両メソッドへ
**`mem: &dyn MemoryOps` を明示引数として追加する**。デフォルト実装は
`self.download`/`self.upload` ではなく `mem.download`/`mem.upload` を
呼ぶため、`BackendOps` 自身が `MemoryOps` を実装している必要が一切なく、
trait 定義（`pub trait BackendOps { ... }`）は変更しない。呼び出し側
（`autodiff::optim::DeviceParamStore`。§3.3a）は、各 `XBackendOps` が
すでに `impl MemoryOps for XBackendOps`（#935 で追加。§3.2）を持つため、
同じ具象値を `&dyn BackendOps` と `&dyn MemoryOps` の 2 通りの trait
オブジェクトとして扱い、後者を `mem` 引数へ渡せばよい。

`tensor-core::BackendOps` に対し、次の 2 メソッドを追加する（既存メソッドの
シグネチャ変更なし・trait 定義への supertrait 追加もなし。TASK-1.9c の
非破壊拡張方針〈`backend_ops.rs:19-27`〉を踏襲する真の非破壊拡張）:

```rust
/// デバイス常駐パラメータの in-place SGD 更新入口（#934 設計・#935 実装）。
///
/// `param` を `grad`（および `velocity`。momentum 有効時のみ `Some`）から
/// 直接更新し、ホストへの往復を発生させない契約とする。既定実装は
/// fail-safe（§3.2）。`mem` は呼び出し側が渡す同一バックエンドの
/// `MemoryOps` 実装（`BackendOps` の supertrait ではない。§3.1 の
/// 採否理由参照）。CPU/CUDA/Metal 各バックエンドは自身のカーネルで
/// オーバーライドできる（デフォルト実装のままでも正しく動作する）。
///
/// **デフォルトメソッドとして trait 定義に本体を持たせる**（`;` で終わる
/// 必須メソッド宣言ではない。レビュー指摘 #934 のとおり、本体なしの
/// 必須メソッドとして追加すると、既存の外部 `impl BackendOps for X` を
/// 破壊する。§3.2 が確定する「fail-safe 合成をデフォルト実装とする」
/// 方針は、この本体をここに書くことで初めて成立する）。本体は次を
/// 呼ぶ（velocity は `Some` のときのみ同様に往復させる）:
fn sgd_step_device(
    &self,
    mem: &dyn MemoryOps,
    param: &mut DeviceBuffer<f32>,
    grad: &DeviceBuffer<f32>,
    velocity: Option<&mut DeviceBuffer<f32>>,
    config: &SgdStepConfig,
) -> Result<(), BackendError> {
    // 既定実装（§3.2）: download → ホスト参照実装（`f32::mul_add`。
    // §5.2 の FMA 契約）で更新 → upload。§3.1 の採否理由（object safety・
    // 非破壊性）により `self.download`/`self.upload` ではなく
    // `mem.download`/`mem.upload` を呼ぶ。実装詳細（ホスト側計算関数へ
    // の委譲）は #935 が確定する。
    default_sgd_step_via_host(mem, param, grad, velocity, config)
}

/// `DeviceParamStore`（§3.3）常駐時の `nn::Linear` forward 入口
/// （#934 設計・#935 実装）。既存の `gemm`/`add` 系メソッド（ホスト
/// `Tensor<f32>` 入出力契約。§2）と異なり、`weight`/`bias` はデバイス
/// 常駐のまま渡す。`input`/戻り値は本段階（§3.3）のスコープに従い
/// ホスト `Tensor<f32>` のまま（forward/backward の完全デバイス常駐化は
/// §6(b) のとおりスコープ外）。
///
/// **呼び出し元は 2 系統ある**（§3.3a で確定。旧稿はここを単一経路と
/// 誤って記述していた）:
/// 1. **Tape 非依存の resident 推論**（`predict` 相当。勾配不要）は、
///    このメソッドを直接呼ぶ。これが本メソッド本来の主眼であり、下記の
///    既定実装がそのまま使われる。**ただし `weight`/`bias` が
///    `DeviceParamStore`（§3.3）由来の `DeviceBuffer` である場合、
///    呼び出し元は `Sequential::predict_resident`（§3.3c。P0 レビュー
///    指摘の解消）を経由する契約とする**。理由: 本メソッドは
///    `tensor-core`（`DeviceParamStore`・poisoned 状態を一切知らない層）
///    に閉じた「渡された `DeviceBuffer` をそのまま downloaded 値として
///    計算するだけ」の入口であり、`DeviceParamStore` 由来のバッファを
///    そのまま渡す直接呼び出しを許すと、§4.3 が定める poisoned 検査
///    （`step`／`sync_to_host`／`forward_resident`／`predict_resident`）
///    をこの経路だけが素通りしてしまう（`DeviceParamStore` に一切
///    紐付かない、呼び出し元が独立に保持する `DeviceBuffer` に対する
///    直接呼び出しは、poisoned という概念自体が存在しないため引き続き
///    許可される）。**この契約は §3.3c が確定するとおりコードレビュー
///    規律ではなく型・可視性で構造的に強制する**（`DeviceParamStore` は
///    `weight`/`bias`/`velocity_*` の `&DeviceBuffer` を返す公開
///    アクセサを一切持たず、`DeviceParamStore::with_resident_buffers`
///    〈poisoned 検査済みスコープ付きクロージャ。§3.3c〉の中でしか
///    参照を得られないため、この関数を `DeviceParamStore` 由来のバッファ
///    で直接呼び出すこと自体が構造的に不可能である）。
/// 2. **`Tape` 追跡下の学習 forward**（`Sequential::forward_resident`。
///    §3.3b）は、**このメソッドを呼ばない**。VJP を成立させるには
///    `Op::MatMul`/`Op::Add` 登録経路（`Tape` を持つ `autodiff` 層でしか
///    組み立てられない）を通す必要があり、`tensor-core`（`autodiff` に
///    非依存）に閉じたこのメソッドの内部では `Tape` へのノード登録が
///    行えないため（§3.3b で結線を確定する）。
///
/// この 2 系統の重複を避けるため、`Sequential::forward_resident` は
/// 「`mem.download` で weight/bias スナップショットを得る」処理のみ
/// このメソッドの既定実装と共有し（下記本体を呼ぶ形にはしない。
/// 独立に `mem.download` を呼ぶ。§3.3b）、以降の計算は既存
/// `Op::MatMul`/`Op::Add` 経路に委ねる。既定実装は次のとおり:
fn linear_forward_resident(
    &self,
    mem: &dyn MemoryOps,
    input: &Tensor<f32>,
    weight: &DeviceBuffer<f32>,
    bias: Option<&DeviceBuffer<f32>>,
) -> Result<Tensor<f32>, BackendError> {
    // 既定実装（§3.2）: download(weight)・download(bias) → 既存
    // gemm+add（ホスト Tensor<f32> 契約）へ委譲。CPU は §2 のとおり
    // 追加転送コストなし。CUDA/Metal 最適化時は weight/bias の再
    // アップロードを省略するオーバーライドに置き換えられる（§3.2）。
    default_linear_forward_via_host(mem, input, weight, bias)
}
```

`SgdStepConfig`（`tensor-core` 側に新設。`autodiff::optim::SgdConfig` の
lr・momentum・dampening・weight_decay・nesterov をそのまま保持する変換先の
設定構造体）は §1 のアルゴリズム引用（`sgd.rs:1-23`）と同一の意味を持つ
フィールド名にし、`autodiff` 側の変換関数（`impl From<&SgdConfig> for
SgdStepConfig` 等）は #935 が実装する。`tensor-core` は `autodiff` に依存しない
（既存の依存方向: `autodiff` → `tensor-core`）ため、`SgdStepConfig` は
`tensor-core` 側に独立定義し、`autodiff` 側が変換して渡す構成とする。

**`is_first_step: bool` フィールド（Cursor Bugbot 指摘の解消）**:
`sgd.rs:1-23` のアルゴリズムは momentum 有効時、`t = 1`（その
パラメータに対する最初の `step` 呼び出し）でのみ `b ← g`（dampening
`τ` を適用しない）、`t ≥ 2` では `b ← μ·b + (1−τ)·g` と分岐する。
`DeviceParamStore` 常駐時の `velocity: DeviceBuffer<f32>` はセッション
開始時にゼロ初期化される（§3.3）ため、この `t = 1` 分岐を
`sgd_step_device` 側が知らないまま「常にゼロ初期化された `velocity` に
対して `b ← μ·b + (1−τ)·g` を適用する」実装にすると、`τ ≠ 0` の場合に
初回ステップの結果が `(1−τ)·g` となり、正しい `g` と食い違う（旧稿は
この分岐をどこにも持たせておらず、ゼロ初期化 `velocity` から `t = 1`
分岐を復元できない実装不備があった）。これを解消するため、
`SgdStepConfig` に `is_first_step: bool` を追加する: `DeviceParamStore`
は各パラメータ位置ごとに「これまで `step` が呼ばれた回数」を保持し
（§4.2 の位置対応契約と同じキー空間。全パラメータは同一のステップ数で
足並みを揃えて進む設計のため、実装上は `DeviceParamStore` 全体で単一の
ステップカウンタ 1 つを持てば足りる）、1 回目の呼び出しでのみ
`is_first_step = true` を渡す。`sgd_step_device` の既定実装（本節冒頭の
コード）は `is_first_step` が `true` のとき `b ← g` に分岐し、`false` の
ときのみ `b ← μ·b + (1−τ)·g` を適用する（`sgd.rs:1-23` の分岐をそのまま
移植する。式順序自体は §5.2 の FMA 契約に従う）。

### 3.2 デフォルト実装の方針: fail-safe 合成 + object safety の確定

既存の未実装カーネル（GPU 側 elementwise/reduction の一部）は `BackendError::
Unsupported` を返す設計だが（`backend_ops.rs:28-33`）、`sgd_step_device`／
`linear_forward_resident` は**それとは異なり「必ず成立するが遅い」合成を
デフォルト実装とする**:

```text
sgd_step_device の既定実装        = download(param) → ホスト参照実装
                                    （CPU と同一式順序・f32::mul_add）で
                                    更新 → upload(param) で書き戻し
                                    （velocity も同様に往復）
linear_forward_resident の既定実装 = download(weight)・download(bias)
                                    → 既存 gemm+add（ホスト Tensor 契約）
                                    へ委譲
```

採否理由: `Unsupported` + 呼び出し側ホストフォールバックにすると、#935 の
実装順序（CPU → CUDA → Metal）の途中で「まだデバイスカーネルを実装していない
バックエンド」を使うコードパスがこれらのメソッドを呼べず、`autodiff::optim`
側に「両方の経路（デバイス常駐 API と旧ホスト API）を呼び分ける分岐」を
恒久的に持たせることになる。デフォルト実装が fail-safe に成立していれば、
`autodiff::optim::DeviceParamStore`（§3.3）は常に `sgd_step_device`／
`linear_forward_resident` だけを呼べばよく、各バックエンドが最適化された
カーネルを持つかどうかは実装詳細に留められる。

**object safety・非破壊性の確定**（レビュー指摘 #934: line 157。旧稿は
`BackendOps` が `MemoryOps` の supertrait ではないため、デフォルト実装が
`Self: MemoryOps` を要求すると `Box<dyn BackendOps + Send>`（既存の
呼び出し形態。`facade`/`autodiff` 側が保持する型）から呼び出せない、と
して supertrait 化を提案していたが、これは crates.io 公開済み trait への
破壊的変更になるため不採用と確定した。§3.1 のとおり、`mem: &dyn
MemoryOps` を明示引数として渡す方式を採用する。`MemoryOps` は
`buffer.rs:203` のドキュメンテーションコメントが明記するとおり
「object-safe に設計している（`&dyn MemoryOps` として扱える）」ため、
`&dyn MemoryOps` を関数引数として受け渡すこと自体に制約はない。
`sgd_step_device`／`linear_forward_resident` のデフォルト実装は
`mem.download(..)`／`mem.upload(..)` を呼ぶだけでよく、`BackendOps`
trait 定義（`Box<dyn BackendOps + Send>` を含む既存の呼び出し形態）は
一切変更されないため object safety・既存実装型の互換性いずれも
保たれる。

現状 `CpuBackendOps`／`CudaBackendOps`／`MetalBackendOps`（`BackendOps`
実装型）と `CpuMemory`／`CudaMemory`／`MetalMemory`（既存の `MemoryOps`
実装型。`backend-{cpu,cuda,metal}/src/memory.rs`）は別構造体である
（`ops.rs`／`memory.rs` 実装箇所を実測確認済み）。#935 は各
`XBackendOps` へ `impl MemoryOps for XBackendOps` を追加する（既存
`XMemory` の実装本体へ委譲する形でよく、`XBackendOps` が `XMemory` を
フィールドとして保持するか、都度その場で構築するかは実装時に選べる
実装詳細とする）。この `impl MemoryOps for XBackendOps` は §3.1 の
trait 定義（supertrait 不採用）が要求するものではなく、**呼び出し側の
利便性のため**に追加する: `DeviceParamStore` が同一の `XBackendOps`
値を `&dyn BackendOps` と `&dyn MemoryOps`（= `mem` 引数）の 2 通りの
trait オブジェクトとして扱えるようにする（同じ具象値を 2 度保持・
生成する必要がない）。外部の `BackendOps` 実装型がこの `impl` を
持たない場合でも、`BackendOps` trait 自体のコンパイルには一切影響
しない（§3.1 の非破壊性）。**この「同じ具象値の 2 通りの trait
オブジェクト」という構成そのものが、§3.2a で確定する `self`/`mem`
同一性検査（ポインタ ID 比較）の前提となる**——`DeviceParamStore` が
`mem` を得る唯一の経路をこの構成に限定するからこそ、`self` と `mem`
が指す実体が常に同一であることを検査で機械的に確認できる（もし
`mem` を無関係な `MemoryOps` 実装値から都度構築してよいことにすると、
§3.2a の識別子ベース検査は意味を持たなくなる）。

### 3.2a `self`（`BackendOps`）と `mem`（`MemoryOps`）の同一バックエンド・
同一インスタンス検査（P1 レビュー指摘の解消）

**指摘の核心 (1)（同一バックエンドの検証手段がない）**: §3.1／§3.3a は
`sgd_step_device`／`linear_forward_resident`／`forward_resident` いずれも
「呼び出し元が保持する同一バックエンドの `MemoryOps` 実装値をそのまま
渡す」ことを前提としているが、これはドキュメント上の慣習（呼び出し元の
責任）に過ぎず、`self: &dyn BackendOps` と `mem: &dyn MemoryOps` が実際に
同一バックエンド・同一インスタンスであることを検証する仕組みが存在
しなかった。`BackendOps::device()`（`backend_ops.rs:86`）による
`param`/`grad` 間のデバイス一致検査（§4.2）は `self` と `mem` の組み
合わせ自体を対象にしていないため、外部 `BackendOps` 実装（第三者が
実装する `impl BackendOps for CustomBackend` 等。crates.io 公開 trait の
ため想定内）を含む公開契約では、誤って別バックエンドの `mem`（例:
`CudaBackendOps` に対し `MetalMemory` を渡す）を渡せてしまい、誤った
handle の扱い・別 context での操作・静かな誤計算につながりうる。

**指摘の核心 (2)（デバイス種別一致だけでは同一インスタンスを保証
できない。P1 レビュー指摘の追加分の解消）**: 旧稿の採用方針は
`mem.device() == self.device()` という `Device` 列挙値どうしの値比較
のみを検査手段としていた。しかし `Device`（`device.rs:47`）は
`Cuda(usize)`（ordinal）のように**デバイス種別＋序数**の値であって
インスタンス識別子ではないため、同一 ordinal を指す 2 つの**別々の**
バックエンドインスタンス（例: 同一 CUDA ordinal 0 に対して構築された、
互いに無関係な 2 つの `CudaContext` に基づく `CudaBackendOps` と
`CudaMemory`。あるいは #931 のデバイスハンドル再利用が確定する前の
過渡期に、呼び出し元が誤って別セッションの `MemoryOps` 値を渡した
場合）を、この値比較は区別できない。`self` の指す実行 context と
`mem` の指す実行 context が実際には異なるにもかかわらず「一致」と
判定してしまう検査は、§3.2a (1) が解消しようとした問題（誤った
handle の組み合わせによる静かな誤計算）を部分的にしか塞げていない。

**採用方針（識別子は「同一デバイス種別」ではなく「同一具象値」を
検証する）**: 上記 (2) を解消するため、検査の主軸を `Device` の値
比較から**ポインタ ID による同一具象値検査**へ置き換える。§3.2 で
確定したとおり `mem` は「`self` と同じ具象値を `&dyn MemoryOps` として
再解釈したもの」という構成（`impl MemoryOps for XBackendOps`）に限定
されるため、`self` と `mem` が指すデータの先頭アドレスが一致するか
どうかは「両者が文字どおり同一のバックエンドインスタンスに由来する
か」を直接検証できる、値比較よりも厳密な識別子である。`BackendOps`・
`MemoryOps` の両方に次のデフォルトメソッドを追加する:

```rust
pub trait BackendOps {
    // 既存メソッドは変更しない。

    /// この trait オブジェクトが指す実体（データ部分）の識別子を返す
    /// （P1 レビュー指摘 (2) の解消: `Device` の値比較〈種別＋序数〉
    /// だけでは同一デバイス種別の別インスタンスを区別できないため、
    /// より厳密な「同一具象値かどうか」の識別に用いる）。
    /// `self as *const Self as *const ()` は `&dyn BackendOps` の
    /// fat pointer からデータポインタ（vtable を含まない部分）のみを
    /// 取り出す標準的な手法（`Rc::ptr_eq`/`Arc::ptr_eq` が内部で使う
    /// ものと同種）であり、参照を外れる（dereference する）操作を
    /// 一切含まないため `unsafe` を要さない。**デフォルトメソッドとして
    /// 追加する**ため、既存の外部 `impl BackendOps for X` は
    /// オーバーライド不要でそのままコンパイルでき、かつ「オーバー
    /// ライドし忘れて既定値のまま検査が機能しない」という失敗モード
    /// （後述 §7 の `MemoryOps::device()` 旧稿が抱えていたのと同種の
    /// リスク）が構造的に発生しない。
    fn instance_id(&self) -> *const () {
        self as *const Self as *const ()
    }
}

pub trait MemoryOps {
    // 既存メソッド（alloc_zeroed／upload／download）は変更しない。

    /// [`BackendOps::instance_id`] と対になる識別子。実装・根拠は同一。
    fn instance_id(&self) -> *const () {
        self as *const Self as *const ()
    }

    /// この `MemoryOps` 実装が属するデバイスを返す（P1 レビュー指摘 (1)
    /// の解消の一部として残す。`BackendOps::device()`〈backend_ops.rs:86〉
    /// と対になる識別情報）。**デフォルトメソッドとして追加する**（`;`
    /// で終わる必須メソッドにすると既存の外部 `impl MemoryOps for X` を
    /// 破壊するため、§3.1／§3.2 と同じ非破壊拡張方針を踏襲する）。
    /// **`self`/`mem` 同一性検査の合否は下記のとおり `instance_id()` が
    /// 決定するため、本メソッドの役割は §4.2 の既存 `param`/`grad`
    /// デバイス一致検査（`instance_id()` を持たない `Device` 値どうしの
    /// 比較で足りる既存用途）との後方互換・診断用エラーメッセージの
    /// 充実に限定される（`instance_id()` が一致していれば `device()` は
    /// 論理的に必ず一致するため、`self`/`mem` 同一性検査そのものには
    /// 不要——後述）。既定は `None`。
    fn device(&self) -> Option<Device> {
        None
    }
}
```

`sgd_step_device`／`linear_forward_resident` の既定実装（§3.1・§3.2）は、
`mem.download`／`mem.upload` を呼ぶ**前に**次の検査を行う:

```text
if self.instance_id() != mem.instance_id() {
    return Err(BackendMismatch);
    // ポインタ ID が不一致 = self と mem が同一の具象値に由来しない
    // （§3.2 の構成〈mem は self と同じ値の再解釈でなければならない〉
    // に違反している）ことを意味し、デバイス種別が偶然一致していても
    // fail-closed に拒否する（P1 レビュー指摘 (2) の解消）。エラー
    // メッセージ生成時は `self.device()`／`mem.device()`（`Some`
    // であれば）を診断情報として付記してよいが、判定そのものには
    // 使わない。
}
```

`instance_id()` はデフォルトメソッドであり、`impl MemoryOps for
XBackendOps`（§3.2。#935 が追加）は明示的なオーバーライドを一切
必要とせず常に正しい識別子を返す（`self as *const Self as *const ()`
は具象型 `XBackendOps` へ委譲した時点で自動的にその値のアドレスを
指す）ため、**§3.2a (1) の旧稿が抱えていた「`MemoryOps::device()` の
オーバーライドを実装側が忘れると検査が常に `BackendMismatch` を返す
（あるいは常に一致してしまう）」というフェイルモードは、`self`/`mem`
同一性検査については構造的に存在しない**。`Device` ベースの
`device()` はそれとは独立に §4.2 の既存用途（`param`/`grad` 間の
デバイス一致検査）で使われ続けるため、各 `XMemory`
（`CpuMemory`／`CudaMemory`／`MetalMemory`）が本メソッドをオーバー
ライドして `Some(Device::Cpu)`／`Some(Device::Cuda(ordinal))`／
`Some(Device::Metal)` を返すことは引き続き #935 の実装事項として
引き渡す（§7）が、これは `self`/`mem` 同一性検査の正しさの前提
条件ではない。

新設する `BackendError::BackendMismatch`（variant 名は #935 で確定）は
「`self`（`BackendOps`）と `mem`（`MemoryOps`）が同一の具象値に由来
しない」ことを呼び出し元が判別できるための型付きエラーとする。
`Sequential::forward_resident`（§3.3a）・`Sequential::predict_resident`
（§3.3c。`DeviceParamStore::with_resident_buffers` 経由）についても、
内部で `BackendOps` の各メソッドへ委譲する前に同じ検査を適用する
（`forward_resident` は `ops`／`mem` の両方を明示引数として受け取る
ため、同一箇所で検査できる）。

**この検査が前提とする呼び出し契約**: `self`/`mem` 同一性検査は
「`mem` が `self` と文字どおり同一の具象値の別 trait オブジェクト
表現である」ことを要求する（§3.2 の構成そのもの）。将来的に「同一
デバイス上の異なるインスタンス間で `mem` を使い回したい」という
正当な要求が生じた場合（例: #931 のデバイスハンドル再利用設計が
複数の論理ハンドルを許容する構成を採るケース）、本節の検査はその
組み合わせを意図的に拒否する。これは本設計のスコープ外とし、
必要になった時点で #931 側の結論を踏まえて再検討する（§7「#931／#932
との整合前提」参照）。

### 3.3 段階設計: 「param + optimizer 状態のみ常駐、勾配は毎ステップ 1 回 upload」

現行 `Tape::backward` はホスト `Tensor` を返す（forward/backward の完全
デバイス常駐化は `DeviceBuffer` 版 `BackendOps`〈`upload`/`download` を
含む全面移行〉が前提であり、変更範囲が 4h 粒度を大きく超えるため #931・
phase-4（#924）へ接続するスコープ外事項とする。§5(b) 参照）。

そのため第 1 段（#935 スコープ）は次の分担とする:

- **常駐化する（param の再アップロードを排除する）**: パラメータ本体
  （`weight`/`bias`）・momentum バッファ（`velocity_weight`／
  `velocity_bias`。§3.3a のとおり `weight`/`bias` 別個に保持する）。学習セッション
  開始時に 1 回 `upload` し、以降 `param` への **再アップロードは発生
  させない**（`sgd_step_device` の in-place 更新のみで完結させる）。
- **常駐化しない（毎ステップ／毎 forward 1 回転送。§3.3b で確定）**:
  forward/backward の入出力（入力バッチ・中間活性）に加え、
  **weight/bias の VJP 用ホストスナップショット**（`forward_resident`
  呼び出しごとに `mem.download` で 1 回。§3.3b）と**勾配**
  （`Tape::backward` が返すホスト勾配を `sgd_step_device` 呼び出し
  直前に 1 回だけ `upload`）の 2 種を含む。旧稿はこの download を
  ここに列挙し損ねており、codex-review 指摘（毎 forward の download が
  転送縮退契約と矛盾する）はこの記述漏れを指した正当な指摘である。
  本節はこの記述を修正することで整合させる（§3.3b「転送モデルとの
  整合性の明記」参照）。

これにより、現行「forward での param upload → backward 後の grad
download → ホスト更新 → 次ステップの param 再 upload」という毎ステップの
param 往復（§2 末尾）のうち、**param の再アップロード**は「学習開始時の
1 回の upload + 終了時（または明示同期時）の 1 回の `sync_to_host`」に
縮退する。一方、**weight/bias の download**（VJP 用スナップショット。
forward 呼び出しごとに 1 回）と **grad の upload**（1 ステップごとに
1 回）は本段階では残る（前者は forward/backward が既存 `Op::MatMul`/
`Op::Add`〈`Tensor<f32>` 契約〉の VJP をそのまま再利用するために必要
〈§3.3b〉、後者は forward/backward 自体が `Tensor<f32>` 契約のカーネルを
呼ぶ限り勾配は一度ホストへ出てくるため）。本設計が実測（§1）に対して
主張する削減効果は「param 往復のうち再アップロード分の排除」であり、
「param に関わる全転送の排除」ではない点を、性能目標・ベンチ受け入れ
基準（#936 引き渡し。§7）の解釈上ここで明確化する。

### 3.3a forward への結線（レビュー指摘 #934: line 134 の解消）

§3.3 の常駐化が成立するには、`nn::Linear`/`Sequential` の forward が
**`DeviceParamStore` の `DeviceBuffer` を直接読む**必要がある。既存
`Sequential::forward`（`sequential.rs:102`）／`SequentialVars::forward`
（`sequential.rs:315`）は `self.layers` の各 `Linear` が保持する**ホスト**
`weight`/`bias`（§2 の現状整理表「パラメータ保持」行）を読む。`Linear` は
`apply_parameters`（two-pass 再構築。§3.5）を通じてのみ更新されるが、
`DeviceParamStore` 常駐時は毎ステップの `apply_parameters` 呼び出しを
行わない（§3.3 の目的そのもの）ため、`Linear` 側のホスト値は「セッション
開始時のスナップショット」のまま stale になる。これが指摘の核心であり、
本節で結線を確定する:

- **所有者は `Sequential` レベル**（`Linear` 単体ではない）。`Linear` は
  毎ステップ `apply_parameters` で丸ごと再構築される設計（§2）のため、
  `Linear` 自身に常駐ストアへの参照を持たせると再構築のたびに配線し直す
  必要が生じ、寿命管理が複雑になる。`Sequential` は既に `apply_parameters`
  の two-pass 検証（§3.5・#426）の主体であり、層の並び順（インデックス）
  を単一に把握している唯一のクレート内オブジェクトのため、常駐ストアの
  結線もここに集約する。
- **層インデックス対応・型の所在（Cursor Bugbot 指摘の解消）**:
  `DeviceParamStore` は `autodiff::optim` に属する（§3.4）が、`Sequential`
  の実体は `facade::compat::sequential::Sequential` であり（`autodiff`
  にも同名の非推奨型 `autodiff::compat::sequential::Sequential` が残存する。
  CLAUDE.md の compat 面移設注記参照）、依存方向は `facade` →
  `autodiff` のみで逆方向（`autodiff` → `facade`）は取れない。したがって
  `DeviceParamStore::new` は **`&Sequential` を直接受け取らない**。
  代わりに `Sequential::trainable_parameters()`（既存 API。`apply_parameters`
  と同じ並び順契約〈§4.2「位置対応契約」・`sgd.rs:183-200`〉）が返す
  `Vec<&Tensor<f32>>` 相当の「学習対象パラメータの並び」をそのまま
  受け取る形にする: `DeviceParamStore::new(mem: &dyn MemoryOps, params:
  &[TrainableParam])`（host `Tensor<f32>` から `DeviceBuffer<f32>` への
  upload を行うため、他のデバイス操作 API〈§3.1／§3.2／§3.2a／§3.3a〉と
  同様に `mem: &dyn MemoryOps` を明示引数として受け取る。§3.3a で一度検出・
  修正した `forward_resident` の mem 引数欠落と同種の欠落を未然に防ぐ）。
  （`TrainableParam` は `autodiff` 側に新設する `{ weight: &Tensor<f32>,
  bias: Option<&Tensor<f32>> }` 相当の平坦なスライス型。確定シグネチャ
  は #935）。呼び出し元（`facade::compat::Sequential::forward_resident`
  もしくはその初期化ヘルパー）が `trainable_parameters()` 相当の抽出を
  行ってから `DeviceParamStore::new` へ渡すことで、`autodiff` は
  `facade`/`Sequential` 型を一切知らずに済む（依存方向を壊さない）。
  **対応表のキー空間**: この「学習対象パラメータの並び」（`weight`→
  `bias` の順で活性化層をスキップした並び）を単一の真実源とし、内部の
  層インデックスは `Sequential.layers`（活性化層を挟む生の層リスト）の
  添字ではなく、この並びに対する 0 始まりの連番（trainable 層インデックス）
  とする。各キーに対応する値は `weight: DeviceBuffer<f32>`・
  `bias: Option<DeviceBuffer<f32>>`・`velocity_weight:
  Option<DeviceBuffer<f32>>`・`velocity_bias: Option<DeviceBuffer<f32>>`
  （いずれも momentum 有効時のみ）の組であり、**1 キーに複数バッファを
  保持できる**構造とする（「層インデックス→単一 `DeviceBuffer`」という
  誤読を避けるため本節で明示する）。**momentum バッファは `weight`／
  `bias` それぞれ専用に分ける（codex-review 指摘の解消）**: `weight` と
  `bias` は一般に shape が異なり（例: `Linear` の `weight` は
  `[out, in]`、`bias` は `[out]`）、かつ §3.1 の `sgd_step_device` は
  `weight`・`bias` それぞれに対して個別に 1 回ずつ呼ばれる契約
  （`velocity: Option<&mut DeviceBuffer<f32>>` を 1 パラメータにつき
  1 個受け取るシグネチャ。§3.1）である。旧稿の「1 キーにつき単一の
  `velocity`」という記述では、この単一バッファを `weight` 用・`bias` 用
  のどちらの `sgd_step_device` 呼び出しにも使い回すことになり、
  shape 不一致（`weight` の velocity を `bias` の shape で読む、または
  その逆）で §4.2 のデバイス一致検査／shape 検査に失敗するか、検査を
  すり抜けた場合は誤った momentum 更新を静かに適用しうる。これを避ける
  ため、`velocity_weight`／`velocity_bias` を独立した `DeviceBuffer`
  として保持し、`sgd_step_device(weight, grad_weight, velocity_weight,
  …)`／`sgd_step_device(bias, grad_bias, velocity_bias, …)` のように
  対応する片方のみを渡す（`bias` が存在しない層では
  `velocity_bias` も常に `None`）。`Sequential::forward_resident`（後述）は
  `Sequential.layers` を通常どおりイテレーションしつつ、`Linear` 層に
  出会うたびにこの trainable 層インデックスを別途カウントアップして
  対応するキーを引く（活性化層はカウントしない）。既存の位置対応契約
  そのものは変更しないため、新たな対応規則を追加で決定する必要はない。
- **新規 forward 入口**: `Sequential` に既存 `forward` を置き換えない
  追加メソッド `forward_resident<'t>(&self, tape: &'t Tape, input: &Var<'t>,
  store: &mut DeviceParamStore, mem: &dyn MemoryOps) -> Result<Var<'t>,
  AutodiffError>`（仮称。確定シグネチャは #935）を追加する。**`mem: &dyn
  MemoryOps` を明示引数として受け取る（codex-review 指摘の解消）**:
  §3.3b のとおり `forward_resident` は各層で `mem.download` を呼ぶ必要が
  あるが、`Sequential`（`facade::compat::sequential::Sequential`）が
  保持するのは `&dyn BackendOps`（または `Box<dyn BackendOps + Send>`）の
  みであり、`&dyn BackendOps` から `&dyn MemoryOps` へ実行時に再変換する
  安全な手段はない（trait オブジェクトの downcast は対象の具象型を
  知らない限り成立せず、§3.2 で確定した「supertrait 化はしない」方針とも
  整合しない）。したがって `mem` は `sgd_step_device`／
  `linear_forward_resident`（§3.1）と同じ設計軸で、**呼び出し元が保持する
  同一バックエンドの `MemoryOps` 実装値をそのまま明示引数として渡す**
  （旧稿はこの引数を欠落させており、§3.3b の学習経路が実装不能だった。
  呼び出し元は §3.4 のとおり `facade` 側であり、`Sequential` 初期化時に
  `BackendOps` と対になる `MemoryOps` 実装値〈同一 `XBackendOps` に対する
  `impl MemoryOps for XBackendOps`。§3.2〉を既に把握しているため、
  `forward_resident` 呼び出し時にそれをそのまま渡せばよく、`Store` 側に
  `MemoryOps` の所有権を持たせる代替案〈`DeviceParamStore` が自身の
  `mem` を保持する案〉は採らない: `DeviceParamStore` は tape とも
  backend インスタンスとも独立な寿命を持つ設計〈§4.1〉であり、`mem` の
  所有権まで持たせると特定バックエンドインスタンスへの暗黙の結び付きが
  生じ、§4.1 の「バックエンド／tape のいずれの寿命にも依存しない」設計
  意図と矛盾する）。**`store` を `&mut` で受け取る**
  （`&` ではない）: §3.3b のとおり、この呼び出しが生成する weight/bias
  leaf の `NodeId` を `store` 自身の内部状態として書き込むため（呼び出し
  元がこの対応表を別途持ち回る必要をなくす。Cursor Bugbot 指摘の解消。
  下記参照）。内部実装は既存 `forward` と同じ層イテレーションだが、
  各 `Linear` 層について自身のホスト `weight`/`bias` の代わりに `store`
  から対応するインデックスの `DeviceBuffer` を読む。**`BackendOps::
  linear_forward_resident`（§3.1）は呼ばない**——同メソッドは
  `tensor-core`（`Tape`/`Var` を知らない）に閉じた「downloaded 値を
  ホスト経由で計算するだけ」の入口であり、weight/bias の VJP を
  成立させる `Op::MatMul`/`Op::Add` 登録は `autodiff` 層（`Tape` を
  保持する層）でしか組み立てられないため（§3.1 の「呼び出し元は 2 系統
  ある」注記・レビュー指摘 #934 の重複 2 件〈resident forward の出力が
  どの Op/NodeId として Tape に登録されるか未定義〉の解消）。入力
  バッチ・中間活性・活性化関数（ReLU 等）は既存 `forward` と同一。
  weight/bias の勾配（VJP）が成立するための結線は §3.3b で確定する
  （レビュー指摘 #934: line 233 の解消。旧稿の「登録ロジックは変更
  しない」という記述は誤りであり、本節で訂正する）。
- **既存 `forward` との関係**: 既存 `forward`（ホスト `Tensor` のみで
  完結）は変更・削除しない（§3.5 の「既存 API 非破壊」方針）。
  `DeviceParamStore` を使う学習ループは `forward_resident` を呼び、
  `predict`／保存等は既存 `forward` + 明示同期後の `apply_parameters`
  を使う（§3.5）。

### 3.3b 常駐パラメータの勾配経路・VJP（レビュー指摘 #934: line 233 の解消）

**指摘の核心**: `linear_forward_resident`（§3.1）は `weight`/`bias` を
`DeviceBuffer` として直接受け取るのみで、既存の `Tape::backward` が
勾配を計算できる対象（`Op::MatMul` が保持する `NodeId` に対応する
`Var` leaf）としては一切登録されない。旧稿はこの点を「forward の
配線を変えるだけ」と過小評価しており、実際には weight/bias に対する
VJP が成立しない（`Sgd::step`／`sgd_step_device` が必要とする `grad`
の出所が設計上欠落する）。本節はこの結線を確定する。

**採用方針**: `Sequential::forward_resident` 自身が weight/bias の
「その時点のスナップショット」を毎 forward 呼び出しで `mem.download`
し、**既存 `Op::MatMul`/`Op::Add` 登録経路（`Linear` 単体ではなく
`Sequential` 内の forward 実装が既に持つ経路）にそのまま乗せて** `Tape`
の leaf `Var` として登録する方式を採る（`BackendOps::
linear_forward_resident` は §3.1 のとおりこの経路では呼ばれず、
Tape 非依存の resident 推論専用に残す）:

- **アクセス経路は `store.with_resident_buffers`（§3.3c）に限定する
  （codex-review P1〈line 543〉指摘の解消: `DeviceParamStore` は
  `weight`/`bias` の生の `&DeviceBuffer` を返す public アクセサを
  §3.3c のとおり一切持たないため、旧稿の `store.weight(layer_idx)`／
  `store.bias(layer_idx)` という直接呼び出しはそもそも実装不能
  だった。`facade` クレート〈`forward_resident` の実装場所〉は
  `autodiff::optim::DeviceParamStore` の非公開フィールドへ触れられず、
  §3.3c が確定した「個別バッファへの唯一のアクセス経路」という制約は
  `predict_resident` だけでなく `forward_resident` にも等しく適用
  される）**。`Sequential::forward_resident` は各層のホスト
  `Tensor<f32>` スナップショットを次の手順で取得する:
  1. `store.with_resident_buffers(|buffers| { ... })`（§3.3c。`&self`
     で足りる読み取り専用貸し出し）を 1 回呼び、クロージャ内で
     `Sequential.layers` を辿って `Linear` 層に出会うたびに対応する
     `buffers[trainable_idx]`（`ResidentLayerBuffers`）を引き、
     `mem.download(buffers[trainable_idx].weight)`／
     `mem.download(buffers[trainable_idx].bias)`（`MemoryOps`。§3.1 の
     `mem` 引数と同じ実装値）でホスト `Tensor<f32>` スナップショットを
     取得して `Vec<(usize, Tensor<f32>, Option<Tensor<f32>>)>`
     （層インデックス・weight スナップショット・bias スナップショット）
     として**所有権ごと**クロージャの戻り値 `R` に積んで返す。
  2. §3.1 で確定した `with_resident_buffers` の `for<'a> FnOnce(&'a
     [ResidentLayerBuffers<'a>]) -> Result<R, BackendError>` という
     HRTB 境界により、`R`（このスナップショット列）は
     `ResidentLayerBuffers<'a>` が持つ `&DeviceBuffer` 参照を一切
     含められない（含めれば型検査に失敗する）ため、ここで得られる
     `Vec<Tensor<f32>>` は `DeviceBuffer` から独立した所有権付きの
     ホスト値であることが型システムにより保証される。したがって
     `with_resident_buffers` の呼び出しが返った時点（クロージャの
     借用スコープが終わった時点）で `store` への `&mut` アクセスが
     再び可能になり、これは同一メソッド内で `store` を後述の
     `pending_grad_sources` 書き込み（`&mut DeviceParamStore`。§3.3a）
     に使うために必要（同時に `&self` の読み取り借用と `&mut self` の
     書き込み借用を両立させる必要がないため、借用チェッカーと矛盾
     しない）。
  3. 得られたスナップショット列を順に `tape.var(&snapshot)`（既存
     API。`tape.rs:288`。leaf ノードとして登録する既存の入口）で
     **毎 forward 呼び出しで新規に**登録する（前ステップの `NodeId`
     を再利用しない。§4.1 の「stale なホスト値を現在のパラメータ
     として使わせない」契約と整合させるため、スナップショットは常に
     その回の `download` 結果のみを表す）。以降の計算（`gemm`+`add`・
     活性化関数）は既存 `forward` と同じ `Op::MatMul`／`Op::Add`
     登録経路にそのまま乗せる。既存の VJP 実装（`Op::MatMul`/`Op::Add`
     の backward）は一切変更しない。
  - poisoned であれば `with_resident_buffers` がクロージャを一切
    呼ばず `BackendError`（poisoned variant）を返すため、
    `forward_resident` はこの 1 回の呼び出しだけで §4.3 の poison
    チェック（下記参照）を自動的に満たす。
  - weight/bias の再アップロードを省略する最適化オーバーライド
    （§3.1 の resident 推論専用経路側の最適化）は、この
    `forward_resident` の `mem.download` 自体には影響しない
    （両者は §3.1 のとおり別経路のため）。「勾配を計算するために
    weight/bias が host 値として少なくとも一度観測可能であること」は
    `forward_resident` を使う限り常に必要であり、省略しない。
- **NodeId の保持先は `DeviceParamStore` 自身**（Cursor Bugbot 指摘の
  解消: 呼び出し元が `Gradients::get` 用のハンドルを得られない問題）。
  `forward_resident` は `&mut DeviceParamStore`（§3.3a）で受け取った
  `store` へ、各層で生成した weight/bias leaf の `NodeId`（`Var`
  自体ではなく `NodeId`。`Var<'t>` は `Tape` への借用を持つため
  `DeviceParamStore`〈`Tape` とは独立の寿命。§4.1〉のフィールドとして
  保持できない）を、層インデックスと対応付けて書き込む
  （`pending_grad_sources: Option<{ tape_id: TapeId, entries: Vec<{
  weight: NodeId, bias: Option<NodeId> }> }>` 相当。前回分は forward
  呼び出しのたびに上書きする「1 forward 分のみ保持する」一時状態であり、
  §4.1 の恒久的な単一真実源〈`DeviceBuffer` 群〉とは別区分の状態である
  ことを明示する）。
  - **`TapeId` によるタイプ一致検査（codex-review 指摘の解消:
    `NodeId` のみの保存では Tape 同一性を検証できない）**: `NodeId` は
    単に「その `Tape` インスタンス内での連番」に過ぎず、`Tape` インスタンス
    をまたいで一意である保証はない（毎ステップ `Tape` を再生成する現行
    設計〈§4.1〉では、新しい `Tape` でも同じ添字の `NodeId` が別の
    無関係なノードを指しうる）。旧稿は `pending_grad_sources` に
    `NodeId` のみを保存しており、`step` に渡された `tape` が
    `forward_resident` 呼び出し時に使われた `Tape` と実際に同一である
    ことを検証していなかった。これは呼び出し元が誤って別セッションの
    `Tape`（例: 前ステップの使い回し・別モデルの `Tape`）を `step` に
    渡した場合、`Var::from_raw(tape, node_id)` が「その `Tape` の中で
    たまたま同じ添字を持つ無関係なノード」を指してしまい、無関係な
    勾配を静かに weight/bias の更新へ適用しうる欠陥である。これを塞ぐ
    ため、`forward_resident` は `tape` から取得できる一意な識別子
    `TapeId`（`Tape` 生成時に採番される値。既存 `tape.rs` に未実装なら
    #935 で新設する軽量フィールドとし、`Tape` の公開契約・既存 API は
    変更しない）を `pending_grad_sources` へ `entries` と併せて保存する。
    `step` はまず `tape.id() == pending_grad_sources.tape_id` を検査し、
    不一致（または `pending_grad_sources` が `None`）であれば
    `Var::from_raw` を一切呼ばずに型付きエラー（`BackendError` の新設
    variant。名称は #935 で確定。「渡された Tape が forward_resident
    呼び出し時の Tape と一致しない」ことを呼び出し元が判別できることを
    本設計で確定する）を返す（fail-closed）。
- **2 回目以降の `forward_resident` 呼び出しに対する検証（codex-review
  指摘の解消: 複数 forward による無検証上書き）**: 旧稿は
  `pending_grad_sources` を「forward 呼び出しのたびに無条件で上書きする」
  としていたが、`step()`（下記）が呼ばれる前に同一 `store` に対して
  `forward_resident` が 2 回以上呼ばれた場合（例: 勾配蓄積・
  マイクロバッチ学習で複数 forward → 1 backward → 1 step という
  呼び出し順序を取るコード）、1 回目の forward が生成した leaf の
  `NodeId` が無検証に破棄され、1 回目の forward 分の重みに対する勾配が
  `step` に一切反映されないまま `Result::Ok` を返してしまう（誤った
  学習結果を正常終了として返す、静かなデータ欠落）。本設計はこれを
  「`DeviceParamStore` は 1 回の `forward_resident` 呼び出しにつき
  高々 1 回の `step` 呼び出しを前提とする単一 forward／単一 step 契約」
  として確定し、次の fail-closed 検査を追加する: `forward_resident` は
  呼び出し開始時に `pending_grad_sources` が既に `Some`（前回 forward
  分が `step` によってまだ消費されていない）であれば、新規の
  `mem.download`／`tape.var` 登録を一切行わず型付きエラー（`BackendError`
  の新設 variant。「未消費の forward_resident 結果が残っている状態での
  再呼び出し」であることを呼び出し元が判別できることを本設計で確定する。
  名称は #935 が確定する）を返す。`step()` は正常終了時（§4.3 の poisoned
  へ遷移しない場合）に `pending_grad_sources` を `None` へ戻し、次の
  `forward_resident` 呼び出しを許可する。複数 forward にまたがる勾配
  蓄積（同一パラメータに対する複数回の forward 分の勾配を合算してから
  1 回 `step` する運用）は本設計のスコープ外とし、必要になった時点で
  別途 #935 以降で検討する（§7「スコープ外事項」へ追記する）。
- **`pending_grad_sources` の解放経路（Medium/Cursor Bugbot 指摘の
  解消: 未消費ロックからの復旧手段がない）**: 上記の「1 回の
  `forward_resident` につき高々 1 回の `step`」契約は、`forward_resident`
  成功後に `tape.backward()` が失敗する（`autodiff` 側のエラー・panic
  からの回復等）、または呼び出し元がそもそも `step` を呼ばずに学習
  ループを中断する場合、`pending_grad_sources` が `Some` のまま残り、
  以後すべての `forward_resident` 呼び出しが「未消費の forward_resident
  結果が残っている」エラーで拒否され続ける、という回復不能な
  デッドロックを生む。旧稿はこの状態からの復旧手段を「`DeviceParamStore`
  全体の再構築・パラメータ再アップロード」以外に定めておらず、コストが
  高い。これを解消するため、`DeviceParamStore` に次のメソッドを追加
  する:

  ```rust
  impl DeviceParamStore {
      /// 未消費の `pending_grad_sources`（前回 `forward_resident` が
      /// 記録した `NodeId`／`TapeId` の一時状態）を、対応する `step`
      /// を実行せずに破棄する（Medium/Cursor Bugbot 指摘の解消）。
      ///
      /// **安全性の根拠**: `pending_grad_sources` は `NodeId`／
      /// `TapeId` の一時的な帳簿にすぎず（§3.3b 冒頭）、
      /// `forward_resident` 自体はどの `DeviceBuffer`（`weight`/
      /// `bias`/`velocity_*`）へも書き込みを行わない（書き込みは
      /// `step` 内の `sgd_step_device` 呼び出し時のみ発生する。§4.3）。
      /// したがって本メソッドは §4.1 の単一真実源・§4.3 の poisoned
      /// 状態のいずれにも影響せず、`DeviceParamStore` が保持する
      /// `DeviceBuffer` 群を一切変更しない、副作用のない状態リセット
      /// である（poisoned 状態のリセットとは別物であり、本メソッドは
      /// poisoned 状態を解除しない。poisoned からの回復手段は
      /// §4.3 のとおり変更しない）。
      ///
      /// 呼び出し後、次の `forward_resident` 呼び出しが再び許可される。
      /// 破棄すべき保留状態がない場合（`pending_grad_sources` が
      /// 既に `None`）は何もせず `false` を返す（エラーにはしない。
      /// 冪等な「念のための呼び出し」を許容する）。
      pub fn abandon_pending_forward(&mut self) -> bool {
          // 実装詳細（フィールドクリアのみ）は #935 が確定する。
          unimplemented!()
      }
  }
  ```

  本メソッドは `tape`／`TapeId` の一致検査を要求しない（`step` とは
  異なり、呼び出し元が「このまま前回 forward 分を破棄したい」という
  明示的な意思表示そのものであり、誤った `Tape` を渡すリスクが
  `step` のケースと同型ではないため）。呼び出し元が誤って有効な
  勾配計算の途中でこれを呼んだ場合の結果は「その forward 分の更新が
  単に適用されない」であり、既存の「`step` を呼び忘れた」場合と
  同じ挙動（無害だが学習が進まない）に留まる。
- `tape.backward()` 実行後、呼び出し元は `DeviceParamStore::step(&mut
  self, tape: &Tape, grads: &Gradients, mem: &dyn MemoryOps, config:
  &SgdStepConfig) -> Result<(), BackendError>`（仮称。確定シグネチャは
  #935）を呼ぶ。`step` はまず上記の `TapeId` 一致検査を行い、通過した
  場合のみ内部で `pending_grad_sources` の各 `NodeId` に
  ついて `Var::from_raw(tape, node_id)`（`autodiff` クレート内部限定の
  `pub(crate)` コンストラクタ。`autodiff::optim` は同一クレートのため
  呼べる。`var.rs:63`）で `Var` を再構築し、`grads.get(&var)`
  （`backward.rs:49`。既存 `Gradients` API をそのまま使う）で
  `grad_weight`／`grad_bias`（ホスト `Tensor<f32>`）を取り出したうえで、
  `Sequential::trainable_parameters()` と同じ並び順（§3.3a）で
  `sgd_step_device` へ 1 パラメータずつ upload・呼び出しする。これは
  §3.3 が定める「`Tape::backward` が返すホスト勾配を `sgd_step_device`
  呼び出し直前に 1 回だけ `upload` する」という転送モデルとそのまま
  整合する。呼び出し元は `NodeId`／`Var` を一切自分で保持する必要が
  なく、`forward_resident` の戻り値（出力 `Var`。損失計算に使う）と
  `tape.backward()` の戻り値（`Gradients`）だけを受け渡せばよい。
  正常終了時は上記のとおり `pending_grad_sources` を `None` に戻す。
- **新規 Op／新規 Tape API は不要**: 既存 `Op::MatMul`/`Op::Add` の
  VJP 実装・既存 `Gradients::get`／`Var::from_raw`（いずれも実装済み
  API）をそのまま再利用するため、`Tape`／`Gradients` 側に「resident
  専用の勾配側チャネル」等の拡張は要らない。これが、この結線が 4h
  粒度の #935 実装に収まる根拠である。
- **既存 `forward`（非常駐）との差異はここに限定される**: 通常の
  `forward` は `Linear` が保持するホスト `Tensor` をそのまま
  leaf 登録するのに対し、`forward_resident` は `DeviceParamStore`
  からの `download` 結果を leaf 登録し、その `NodeId` を `store` へ
  書き戻す。leaf 登録・VJP 計算の仕組み自体は共通である。

**転送モデルとの整合性の明記（レビュー指摘 #934 の重複 2 件・codex-review
P1〈毎 forward の download が §3.3 の転送縮退契約と矛盾する〉の解消）**:
上記のとおり `forward_resident` は毎 forward 呼び出しで weight/bias の
`mem.download` を行う。これは §3.3 が「常駐化しない（毎ステップ 1 回
転送）」に列挙した対象（forward/backward の入出力）には含まれておらず、
旧稿はこの download を転送モデルの記述に反映し損ねていた。実態を正しく
記すと、本設計が §3.3 で削減するのはあくまで **param の再アップロード
（毎ステップの `upload`）のみ**であり、param の **download**（VJP 用の
host スナップショット取得）は forward 呼び出しごとに 1 回残る
（velocity・grad の扱いは §3.3 の記述のまま変更しない）。この download を
さらに削減するには、`Op::MatMul`/`Op::Add` に依存しない
resident-aware な VJP（`DeviceBuffer` 上で直接 backward を実行する専用
Op）が必要であり、これは forward/backward 全体を `DeviceBuffer` 契約へ
移行する変更（§6(b)）と同水準の変更範囲になるため、本設計（#933 ツリー
第 1 段）のスコープ外とする（§7「スコープ外事項」へ追記する）。§3.3 の
「常駐化する」「常駐化しない」の 2 分類は、この download を「常駐化しない
（forward ごとに 1 回転送）」の対象へ追加する形で更新する
（forward/backward の入出力〈入力バッチ・中間活性〉・weight/bias の
VJP 用スナップショット・勾配、の 3 種が「毎ステップ／毎 forward 1 回
転送」の対象となる）。

### 3.3c `DeviceParamStore` 経由の resident 推論エントリ（P0 レビュー
指摘 2 件の解消: (1) resident 推論経路が poisoned 検査を経由しない、
(2) poisoned 検査の迂回防止をコードレビュー規律のみに依存している。
Cursor Bugbot 指摘の解消: resident 推論が活性化関数を実行しない）

**指摘の核心 (1)（poisoned 検査の迂回）**: §3.1 で確定した `BackendOps::
linear_forward_resident` の「呼び出し元は 2 系統」記述は、系統 1
（Tape 非依存の resident 推論）を「このメソッドを直接呼ぶ」とだけ定めて
おり、`weight`/`bias` の `DeviceBuffer` を `DeviceParamStore` から
読み出して渡す具体的な経路には一切触れていなかった。`DeviceParamStore`
は `sgd_step_device` の実行時エラー（§4.3）により poisoned 状態へ
遷移しうるが、poisoned 検査は `step`／`sync_to_host`／`forward_resident`
の 3 箇所にしか課されておらず（旧稿の §4.3 最終箇条）、
`linear_forward_resident` を直接呼ぶ resident 推論はこの 3 箇所の
どれにも該当しない。結果として、更新途中の実行時エラーで一部パラメータ
のみ更新された poisoned な `DeviceParamStore` から `weight`/`bias` の
`DeviceBuffer` 参照を取り出し、それをそのまま `linear_forward_resident`
へ渡せば、poisoned フラグを一切検査せずに破損混在状態のバッファで推論が
実行できてしまう（fail-closed 契約・AGENTS.md「fail-closed の維持（P0）」
への抵触）。

**指摘の核心 (2)（迂回防止の実効性）**: 旧稿（本節の前稿）は上記 (1) の
解消策として「`DeviceParamStore::predict_resident` を新設し、直接呼び出し
禁止を docstring で定める」対応のみを採ったが、この禁止はコードレビュー
規律にのみ依拠しており型・可視性で強制されていなかった。しかも
`facade` 側の呼び出し元（`Sequential`）が `store` 内のバッファを読む
ためには何らかのアクセス経路が必要であり、旧稿はそのアクセス経路
自体を未定義のまま残していた。将来 `DeviceParamStore` へ `weight`/
`bias` の生の `&DeviceBuffer` を返す public アクセサが（`facade` 側の
実装上の都合等で）追加されれば、そのアクセサから得た参照を
`linear_forward_resident` へ直接渡すことで poisoned 検査を迂回できて
しまい、docstring 上の禁止は実効性を持たない。

**指摘の核心 (3)（Cursor Bugbot・活性化関数の欠落）**: 旧稿の
`predict_resident` は `DeviceParamStore`（`autodiff::optim`。`Linear`
層の `weight`/`bias` のみを保持し、`Sequential` の層構成〈活性化層を
含む〉を一切知らない型。§3.3a）のメソッドとして定義されていたため、
`ops.linear_forward_resident` を各層の `weight`/`bias` へ順に適用する
だけで、層間の `ReLU` 等の活性化関数を一切実行しない実装にしかなり
えなかった。既存 `Sequential::forward`（活性化層を含む全層を順に
適用する。§2）・`forward_resident`（§3.3a・同様に活性化層を含む）との
非対称であり、resident 推論の出力が非 resident の `predict` と一致しない
（数値一致契約 §5 にも抵触しうる）。

**採用方針（迂回不能な構造的強制 + 活性化関数の結線）**: 上記 3 点を
同時に解消するため、責務を次のとおり分離する:

- **`DeviceParamStore` は `weight`/`bias`/`velocity_*` の生の
  `&DeviceBuffer` を返す public アクセサを一切持たない**（既存設計の
  追認であり本節が新たに課す制約ではない。§3.3a はこの型を「常駐
  ストア」とのみ定義しており、個別バッファの getter は元より定義して
  いない）。個別バッファへ触れる唯一の手段として、次の poisoned
  検査済みスコープ付きクロージャ API を新設する:

  ```rust
  impl DeviceParamStore {
      /// poisoned 検査を通過した場合に限り、内部の trainable 層バッファ
      /// 群（`weight`/`bias`/`velocity_*`。§3.3a の対応表と同じ並び順）
      /// への読み取り専用アクセスをクロージャへ一時的に貸し出す
      /// （P0 レビュー指摘 (2) の解消: 型・可視性による構造的強制）。
      ///
      /// **迂回不能性の根拠（codex-review P1〈line 795〉指摘の解消:
      /// `'_` を明示 `for<'a>` へ書き換える）**: 引数 `f` の境界は
      /// `impl FnOnce(&[ResidentLayerBuffers<'_>]) -> Result<R, BackendError>`
      /// ではなく、下記のとおり **`for<'a> FnOnce(&'a
      /// [ResidentLayerBuffers<'a>]) -> Result<R, BackendError>` と明示
      /// する**。理由: `Fn` 系トレイトの糖衣構文では引数位置の省略
      /// ライフタイム（`'_` を含む）は暗黙に高階（HRTB）へ束縛される
      /// 場合があるが、この省略規則は「関数境界の構文糖衣として現れる
      /// 場合」に限られた言語仕様上の特例であり、`'_` という表記自体が
      /// 常に HRTB を保証する汎用の記法ではない（省略規則の適用条件が
      /// 崩れる書き換え——例えば `ResidentLayerBuffers<'_>` を型エイリアス
      /// 経由に置き換える、`Box<dyn FnOnce(...)>` 等の非糖衣形へ変更する
      /// ——を将来加えた際に、`'_` の書き方のままでは意図しない具体
      /// ライフタイムへ静かに縮退しうる）。`for<'a>` を明示すること
      /// で、`f` に渡る `&'a [ResidentLayerBuffers<'a>]`（`'a` は
      /// `with_resident_buffers` の呼び出しごとに新規に選ばれる高階
      /// ライフタイムであり、`std::thread::scope` と同型の
      /// scoped-closure パターン）が構文の変化に依存せず常に高階で
      /// あることをシグネチャ自体に固定する。戻り値の型 `R` はこの
      /// 高階ライフタイム `'a` を含められない（含めばコンパイルエラーに
      /// なる——`R: 'static` 相当の制約を明示的に課すまでもなく、`'a`
      /// が呼び出しごとに新規の匿名リージョンであるため `R` の中で
      /// 名指しできないという型検査上の帰結）ため、生の `DeviceBuffer`
      /// 参照をクロージャのスコープ外へ持ち出す経路がコンパイラの
      /// ボローチェッカーにより存在しない。したがって「`DeviceParamStore`
      /// 由来のバッファを `linear_forward_resident` へ直接渡す」という
      /// 迂回は、docstring 上の禁止ではなく型システムにより不可能になる
      /// （このため、後述 §3.3b の `forward_resident` はクロージャ内で
      /// `mem.download` した「所有権を持つホスト `Tensor<f32>` の
      /// スナップショット」だけを `R` として持ち出し、クロージャの
      /// スコープを抜けたあとに `store` への `&mut` 書き込みを行う
      /// 構成を取れる）。poisoned であれば `f` を一切呼ばず（クロージャ
      /// の中身が実行されないため、破損混在状態のバッファに触れる経路
      /// 自体が生じない）、`f` が要求する戻り値型と同じ `Result` 型の
      /// `BackendError`（poisoned variant）を返す。
      pub fn with_resident_buffers<R>(
          &self,
          f: impl for<'a> FnOnce(&'a [ResidentLayerBuffers<'a>]) -> Result<R, BackendError>,
      ) -> Result<R, BackendError> {
          // 実装詳細（poisoned フラグ検査 → 内部バッファ配列の構築 →
          // f 呼び出し）は #935 が確定する。`ResidentLayerBuffers` の
          // フィールド（`weight`/`bias`/`velocity_weight`/
          // `velocity_bias` の `&DeviceBuffer<f32>`／
          // `Option<&DeviceBuffer<f32>>`）も #935 が確定する。
          unimplemented!()
      }
  }
  ```

- **Tape 非依存の resident 推論エントリは `Sequential` 側に置く**
  （`facade::compat::Sequential::predict_resident`。`forward_resident`
  〈§3.3a〉と対になる読み取り専用版）。`Sequential` は既に層構成
  （`Linear` と活性化層を含む並び。§2・§3.3a）を唯一把握するオブジェクト
  であるため、活性化関数の適用箇所を正しく判断できるのは `Sequential`
  のみであり、`DeviceParamStore`（`autodiff::optim`）側にこの責務を
  置くこと自体が指摘 (3) の構造的な原因だった。

  **活性化種別の判別手段（codex-review P1〈line 833〉・Bugbot 指摘の
  解消: 「既存 forward と同じ活性化メソッドを適用する」だけでは
  未定義だった具体的な結線）**: 既存 `Sequential::forward`（ホスト
  `Tensor`／`Var` 経路）は `Vec<Box<dyn Module>>` の多態 `forward`
  （`Module::forward(&self, tape, input) -> Var`。`nn/module.rs`）に
  委譲することで、`Module` 実装型（`Linear`／`Relu`／`Sigmoid`／
  `Tanh`）を個別に判別せずに活性化を適用している。しかし
  `predict_resident` は `Tape`/`Var` を経由しない `BackendOps`
  （ホスト `Tensor<f32>` 契約）経路であり、`Module::forward` を
  呼べない。現行の `Module` trait（`nn/module.rs`）が持つ唯一の型
  判別フックは `as_linear`/`as_linear_mut`（`Linear` 層の識別専用）
  であり、活性化層 3 種のうちどれかを判別する手段が存在しない。
  加えて `BackendOps`（`backend_ops.rs:91-98`）は `relu`／`tanh` は
  持つが `sigmoid` を持たず、`Sigmoid` 層に出会った場合に呼べる
  `BackendOps` メソッドがそもそも欠けている。本設計はこの 2 点を
  解消するため、次を新設する:

  ```rust
  // `autodiff::nn::module`（`Module` trait と同じモジュール。
  // 活性化種別は `docs/compat-api-scope.md` §1 が定める閉集合
  // 〈ReLU/Sigmoid/Tanh の 3 種〉のみであるため、`as_linear` と
  // 同じ「閉集合を明示フックとして列挙する」設計軸〈module.rs
  // 冒頭コメント〉を踏襲し、`std::any::Any` によるダウンキャスト
  // は使わない）。
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum ActivationKind {
      Relu,
      Sigmoid,
      Tanh,
  }

  pub trait Module {
      // 既存メソッド（forward／as_linear／as_linear_mut）は変更しない。

      /// この層が活性化関数（`Relu`/`Sigmoid`/`Tanh`）であれば対応する
      /// [`ActivationKind`] を返す（P1 レビュー指摘の解消: `Tape` を
      /// 経由しない resident 推論〈predict_resident〉が、`Module::forward`
      /// の多態 dispatch に頼らず活性化種別を判別するためのフック）。
      /// 既定実装は `None`（`Linear` を含む非活性化層はオーバーライド
      /// しない。デフォルトメソッドのため既存の外部 `impl Module for X`
      /// を破壊しない非破壊拡張）。
      fn as_activation_kind(&self) -> Option<ActivationKind> {
          None
      }
  }

  impl Module for Relu {
      fn as_activation_kind(&self) -> Option<ActivationKind> {
          Some(ActivationKind::Relu)
      }
  }
  impl Module for Sigmoid {
      fn as_activation_kind(&self) -> Option<ActivationKind> {
          Some(ActivationKind::Sigmoid)
      }
  }
  impl Module for Tanh {
      fn as_activation_kind(&self) -> Option<ActivationKind> {
          Some(ActivationKind::Tanh)
      }
  }
  ```

  ```rust
  pub trait BackendOps {
      // 既存メソッド（gemm／add／relu／tanh 等）は変更しない。

      /// シグモイド活性化（`1 / (1 + exp(-x))`）。P1 レビュー指摘の
      /// 解消: `predict_resident` が `ActivationKind::Sigmoid` を
      /// 呼べるようにするための新設メソッド。既存の未実装カーネル
      /// 群（`backend_ops.rs:28-33`）と同じ fail-safe 方針を踏襲し、
      /// **デフォルト実装は `BackendError::Unsupported` を返す**
      /// （§3.2 冒頭が区別するとおり、`sgd_step_device`／
      /// `linear_forward_resident` の「必ず成立する」デフォルトとは
      /// 異なる区分——`sigmoid` は既存の `relu`/`tanh` と同格の
      /// 通常カーネルであり、ホスト経由の汎用フォールバックを
      /// 新設しない）。CPU 実装は #935 が `f32::exp` ベースで即時
      /// 追加し（`f32::mul_add` 契約とは無関係な単項関数のため
      /// §5.2 の FMA 契約は適用外）、CUDA/Metal は実装順序
      /// （§3.4「実装順序は CPU → CUDA → Metal」）に従って追って
      /// 追加する。**デフォルトメソッドとして追加する**ため、`relu`/
      /// `tanh` と異なり既存の外部 `impl BackendOps for X` を破壊
      /// しない（trait 定義への非破壊拡張。§3.1 と同じ方針）。
      fn sigmoid(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
          Err(BackendError::Unsupported)
      }
  }
  ```

  ```rust
  impl Sequential {
      /// `store` 常駐のパラメータを使う Tape 非依存の resident 推論
      /// （`predict` 相当）。`forward_resident`（§3.3a）と対になる
      /// 読み取り専用 API（`&self`。`store` は poisoned チェック用に
      /// `&DeviceParamStore` で十分——書き込みは発生しない）。
      pub fn predict_resident(
          &self,
          store: &DeviceParamStore,
          ops: &dyn BackendOps,
          mem: &dyn MemoryOps,
          input: &Tensor<f32>,
      ) -> Result<Tensor<f32>, BackendError> {
          // `store.with_resident_buffers` の呼び出し 1 回のみでクロージャ
          // 内へ入り、以後は `self.layers`（既存 `forward` と同じ層
          // イテレーション）を順に辿る: `Linear` 層（`layer.as_linear()`
          // が `Some` を返す層）は対応する `ResidentLayerBuffers` を
          // trainable 層インデックス（§3.3a）で引いて
          // `ops.linear_forward_resident(mem, current, weight, bias)` を
          // 呼ぶ。活性化層（`layer.as_activation_kind()` が `Some` を
          // 返す層）は `match` で `ActivationKind::Relu => ops.relu(&current)`／
          // `ActivationKind::Sigmoid => ops.sigmoid(&current)`／
          // `ActivationKind::Tanh => ops.tanh(&current)` を呼び、結果で
          // `current` を差し替える（`ops.sigmoid` が `Unsupported` を
          // 返した場合はそのまま呼び出し元へ伝播する——CPU 実装が揃う
          // までは Sigmoid を含む `Sequential` の resident 推論は
          // fail-closed に失敗する、既存 `relu`/`tanh` と同じ規約）。
          // `as_linear()`/`as_activation_kind()` のいずれも `None` を
          // 返す層は `docs/compat-api-scope.md` §1 の閉集合契約上
          // 到達しないが、到達した場合は型付きエラー（新設 variant。
          // 名称は #935 が確定）で fail-closed に扱う。既存
          // `forward`／`forward_resident` と同一の層適用順序をここでも
          // 維持するため、活性化関数は resident 推論でも通常どおり
          // 実行され、CPU 実装が揃った状態では既存 `predict` と同一の
          // 演算列（同じ `relu`/`sigmoid`/`tanh` 実装への委譲）になる
          // ため #936 の数値一致契約（§5.1）を満たせる（Cursor Bugbot
          // 指摘の解消）。確定シグネチャ・層イテレーション詳細は #935
          // が実装する。
          unimplemented!()
      }
  }
  ```

  poisoned であれば `store.with_resident_buffers` がクロージャを
  一切呼ばずに `BackendError`（poisoned variant）を返すため、
  `predict_resident` は poisoned 検査を必ず forward 計算より先に
  行う契約を自動的に満たす（`with_resident_buffers` 自体が唯一の
  バッファアクセス経路であるため、この検査を素通りする経路は
  存在しない）。

**§4.3 との整合**: これにより poisoned 検査の対象は `step`／
`sync_to_host`／`forward_resident`／`predict_resident` の 4 箇所となる
（§4.3 の最終箇条を本節の追加に合わせて更新する）。`linear_forward_resident`
自体・`with_resident_buffers` が返すクロージャの中身自体は `tensor-core`
（`predict_resident` は `facade` 層。`with_resident_buffers` は
`autodiff::optim` 層）に留まり、poisoned 検査そのものは常に
`DeviceParamStore::with_resident_buffers` の呼び出し時点で行う、
という責務分担が本節の確定事項である（`tensor-core` は引き続き
`autodiff::optim::DeviceParamStore` に依存しない設計〈既存の依存方向〉
を維持する）。

### 3.4 クレート責務分担

| クレート | 責務 |
|---|---|
| `tensor-core` | `sgd_step_device`／`linear_forward_resident`／`sigmoid`（§3.3c。activation 判別と対になる新設デフォルトメソッド）trait メソッド（`mem: &dyn MemoryOps` 明示引数。supertrait 化はしない非破壊拡張）・`BackendOps::instance_id`／`MemoryOps::instance_id`（§3.2a。`self`/`mem` 同一性検査用デフォルトメソッド）・`SgdStepConfig` 型定義（§3.1・§3.2） |
| `backend-cpu`／`backend-cuda`／`backend-metal` | 各バックエンドのカーネル実装（CPU: 逐次または `rayon` 並列 in-place・CUDA: NVRTC 1 カーネル・Metal: compute shader 1 個）＋ `impl MemoryOps for XBackendOps`（§3.2。呼び出し側の利便性のための追加であり trait 定義上の要求ではない。`instance_id` はデフォルトメソッドのためオーバーライド不要）＋ `sigmoid` の CPU 実装（§3.3c。CUDA/Metal は実装順序どおり後続）。実装順序は CPU → CUDA → Metal（#935 引き渡し事項。§6） |
| `autodiff::nn` | `ActivationKind` enum・`Module::as_activation_kind`（§3.3c。`Relu`/`Sigmoid`/`Tanh` の 3 型がオーバーライド。既存 `Module::forward`／`as_linear`／`as_linear_mut` は変更しない非破壊拡張） |
| `autodiff::optim` | `DeviceParamStore`（常駐ストア保持型。§3.3・§4 の所有者）・`SgdConfig → SgdStepConfig` 変換・`Tape` の勾配出力を `DeviceParamStore` へ渡す統合ロジック（§3.3b の leaf 登録・勾配取り出しを含む）・`DeviceParamStore::with_resident_buffers`（poisoned 検査済みスコープ付きクロージャ。§3.3c。個別バッファへの唯一のアクセス経路） |
| `facade`（`compat::Sequential` 経由で `Sequential::forward_resident`／`Sequential::predict_resident` を実装） | `Sequential` への `forward_resident`・`predict_resident`（§3.3b・§3.3c。いずれも `store.with_resident_buffers` 経由で個別バッファへアクセスし、`as_linear`／`as_activation_kind` で層種別を判別しつつ既存の層イテレーション順序を再利用する）・層インデックス対応表の保持（§3.3a）。公開面は #932 の optimizer facade 公開設計の結論に従属させる（本ドキュメントは compat 面の意匠を確定しない） |

### 3.5 既存 API との関係

`Sequential::apply_parameters`／`trainable_parameters`（ホスト `Tensor<f32>`
契約）は**公開契約として変更しない**（公開 API 非破壊はガードレール条件。
`.claude/rules/security.md` A08）。デバイス常駐経路（`DeviceParamStore` 経由の
学習ループ）は既存経路に追加する新経路であり、既存経路を置き換えない。
`DeviceParamStore` からホスト `Tensor<f32>` への同期（学習終了時・`predict`/
保存前）は明示 API（例: `DeviceParamStore::sync_to_host(&self, mem: &dyn
MemoryOps) -> Vec<Tensor<f32>>`。`DeviceBuffer<f32>` から host `Tensor<f32>`
への download を行うため `new` と同様に `mem: &dyn MemoryOps` を明示引数
として受け取る。確定シグネチャは #935 で行う）を介して行い、
`apply_parameters` へ渡す形で
既存の書き戻し経路と接続する。

## 4. 所有権（design decision 2）

### 4.1 所有者と単一真実源

学習セッション中は `autodiff::optim::DeviceParamStore` が保持する
`DeviceBuffer<f32>` 群を**正**とする。`DeviceParamStore` の寿命は `Tape` とは
独立とする（毎ステップの tape 再生成・再初期化〈#922 系〉をまたいで生存する。
tape 初期化コスト削減〈(a)〉とパラメータ常駐化〈(b)〉が互いに独立した最適化
であることの表れでもある）。

`nn::Linear` が保持するホスト `Tensor<f32>` は、学習セッション中は
「セッション開始時にアップロードした初期値のスナップショット」であり、
`DeviceParamStore` 側の更新を都度ミラーしない（stale になり得る）。誤って
stale なホスト値を「現在のパラメータ」として使う事故を防ぐため、次の契約と
する:

- 学習ループ内で `Linear`（または `Sequential`）のホスト `Tensor` を直接
  読み取る API（`weight()`/`bias()`/`trainable_parameters()`）は、
  `DeviceParamStore` 使用中は「最後に明示同期した時点のスナップショット」
  であることをドキュメンテーションコメントで明記する（#935 で追記）。
- 明示同期 API（§3.5 の `sync_to_host`）を呼んだ直後のみ最新値であることが
  保証される。同期タイミングは「学習終了時」「`predict`/保存直前」の 2 箇所
  を最低限の契約とし、#935 の実装がこれ以外の暗黙同期（例: 勝手なタイミングで
  自動 download する等）を行わないことを fail-closed の前提とする（暗黙
  コピー禁止方針。`docs/public-api-design.md` §2.2.1 と同じ思想）。

### 4.2 fail-closed 検査

`sgd_step_device` 呼び出し前に `DeviceParamStore` が担う検証（`Sgd::step`
既存契約〈`sgd.rs:183-200`〉と同水準を装置常駐版へ引き継ぐ):

- **デバイス一致検査**: `param.device() == grad.device()`（対応する
  `velocity_weight`／`velocity_bias` がある場合は同様に検査。§3.3a の
  とおり `weight` 呼び出しには `velocity_weight`、`bias` 呼び出しには
  `velocity_bias` のみを渡すため、この検査も呼び出しごとに独立に行う）
  でなければ `BackendError::DeviceMismatch` を返す
  （`buffer.rs` の `DeviceBuffer::device()` を用いる）。
- **位置対応契約**: `params.len() != grads.len()` は `InvalidArgument`。
  momentum 有効時、2 回目以降の呼び出しで件数・各 `DeviceBuffer` の
  `shape()` が前ステップと変化した場合も `InvalidArgument`（`sgd.rs:196-215`
  の既存検証と同じ判定を `DeviceBuffer` 版に移植する）。
- **shape 保存契約**: `param` の `shape()` は呼び出し前後で不変（in-place
  更新であり形状変化は想定しない）。`Sequential::apply_parameters` の
  #426（置換前 shape 完全一致検証）と同水準の検査を、常駐経由の更新でも
  維持する。

### 4.3 アトミック性

`Sequential::apply_parameters` の two-pass 方式（1 パス目で全層の検証込み
再構築、2 パス目で代入。`sequential.rs:229-` 以降のコメント）を in-place
更新にも適用する契約とする:

```text
検証フェーズ: 全パラメータ・全 velocity_weight／velocity_bias の shape・
              device 一致を検査
              （1 件でも不一致なら早期 return、どのバッファも未更新）
更新フェーズ: 検証済みの全件について sgd_step_device を呼ぶ
```

**検証フェーズが排除するのは事前検証可能な失敗のみ**（レビュー指摘
#934: line 223。`sgd_step_device` は `Result` を返すため、検証フェーズを
通過した後の更新フェーズでも、2 件目以降の呼び出しがデバイス側の実行時
エラー〈CUDA カーネル起動失敗・ドライバエラー・OOM 等〉で失敗しうる。
これは事前の shape/device 検証では検出できない失敗であり、two-pass
方式だけでは「一部パラメータのみ更新済み」の混在状態を排除できない。
これが指摘の正当な核心であり、本節は保証範囲を次のとおり確定する（
チェックポイント/ロールバック方式は、更新のたびに全パラメータ分の
一時バッファを追加確保する必要があり、本設計が §1 で削減対象とする
ホスト往復コストとは別種だが同水準の追加コストを生むため不採用とする）:

- **検証フェーズの保証（変更なし）**: shape/device 不一致・件数不一致は
  検証フェーズで検出され、1 件でも不一致なら更新フェーズは一切開始
  されない（どのバッファも未更新のまま `Result::Err` を返す）。
- **更新フェーズの保証（新設・非保証の明示）**: 検証フェーズ通過後の
  更新フェーズで `sgd_step_device` が実行時エラーを返した場合、
  `DeviceParamStore` はその時点で更新済みのバッファと未更新のバッファが
  混在する状態になり得ることを**明示的に非保証（no atomicity）の契約と
  する**。この場合 `DeviceParamStore::step` は最初に発生したエラーを
  `Result::Err` として返すと同時に、ストア自身を **poisoned 状態**へ
  遷移させる（`std::sync::Mutex` の poisoning と同じ考え方）。poisoned
  状態の `DeviceParamStore` は以後 `step`／`sync_to_host`（§3.5）の
  いずれも `BackendError`（新設 variant。名称は #935 で確定するが、
  「poisoned のため再初期化が必要」であることを呼び出し元が判別できる
  こと自体は本設計で確定する）を返し、暗黙に「部分更新済みの値」を
  正常値として使わせない（暗黙コピー禁止方針〈§4.1〉と同じ「stale／
  不正な値を誤って使わせない」設計軸の延長）。回復手段は「セッション
  開始時に保持していたホスト `Tensor<f32>`（§4.1 のスナップショット。
  最後の明示同期時点の値）から `DeviceParamStore` を再構築する」ことに
  限定する（デバイス側の実行時エラーはドライバ・ハードウェア起因が
  多く、同一 `DeviceParamStore` インスタンス内での自動リトライは対象と
  しない）。
- **§4.1／§4.4 との整合**: poisoned 状態は「新しい状態」を追加するのみで、
  §4.1 の単一真実源（`DeviceParamStore` が正）・stale スナップショット
  契約（ホスト `Tensor` は最後の明示同期時点の値）とも、§4.4 の RAII
  一本化（poisoned 状態でも `Drop` 経由の解放は通常どおり働く。
  poisoned は「使用不可」を表すフラグであり、解放不能状態ではない）とも
  矛盾しない。
- **`forward_resident`／`predict_resident` も poison チェック対象とする
  （Cursor Bugbot 指摘・P0 レビュー指摘の解消）**: 旧稿は `step`／
  `sync_to_host` のみ poisoned 時に `BackendError` を返すと定めており、
  `Sequential::forward_resident`（§3.3a・§3.3b）が破損混在状態の
  バッファをチェックなしに読める抜け穴が残っていた。§3.3b の改訂
  （codex-review P1〈line 543〉指摘の解消）により、`forward_resident`
  は個別の poisoned フラグ検査コードを別途持つのではなく、
  `Sequential::predict_resident`（§3.3c）と全く同じ経路——
  `store.with_resident_buffers`（§3.3c。個別バッファへの唯一の
  アクセス経路）——を経由してのみ `weight`/`bias` の `DeviceBuffer`
  へアクセスする。poisoned であれば `with_resident_buffers` がクロー
  ジャを一切呼ばず `BackendError`（poisoned variant）を返し、
  `forward_resident` 側では `mem.download` にも `ops.linear_forward_resident`
  （§3.1）にも到達しない（`predict_resident` 側も同様に
  `ops.linear_forward_resident` へ到達しない）。`DeviceParamStore` が
  個別バッファの public アクセサを一切持たない設計（§3.3c）により、
  `with_resident_buffers` を経由せず `weight`/`bias` の `DeviceBuffer`
  を得る手段自体が存在しないため、この poison チェックはコードレビュー
  規律ではなく型・可視性で構造的に強制される（`forward_resident`／
  `predict_resident` のいずれについても同一の構造的強制であり、
  片方だけが手動チェックに依存するという非対称は残らない）。これに
  より「破損した混在状態のパラメータで forward／推論が実行される」
  経路を閉じる（`step`／`sync_to_host`／`forward_resident`／
  `predict_resident` の 4 箇所すべてが poison チェックを通過して初めて
  `DeviceParamStore` の内部状態へアクセスできることを、本設計が確定
  する fail-closed 契約とする）。

### 4.4 解放・スレッド境界

- **RAII 一本化**: `DeviceBuffer`/`BufferHandle` の既存方針（`buffer.rs`
  「解放方針」節）をそのまま踏襲する。`DeviceParamStore` に明示 `free()`
  は設けない。
- **Send/Sync**: 現行の `DeviceBuffer`/`BufferHandle` が `Send`/`Sync` を
  過剰に約束しない方針を維持し、`DeviceParamStore` も同様とする（#935 で
  必要になった時点で最小限のみ追加する）。
- **空テンソル契約**: `numel() == 0` のパラメータ（想定は稀だが、テストの
  境界ケース等）は `buffer.rs` の空テンソル契約（FFI を呼ばない空ハンドル）
  をそのまま適用する。

## 5. 数値一致契約（design decision 3）

### 5.1 参照実装と判定基準

CPU ホスト参照実装（既存 `optim::Sgd::step` の式順序・`f32::mul_add`。
`sgd.rs:1-23` のアルゴリズム）を正とし、デバイス常駐更新後のパラメータとの
比較は既存の統一複合判定を**そのまま適用する**:

> 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満（`.claude/rules/coding-rust.md`）

**新しい tolerance を新設・緩和しない**。変更が必要と判明した場合は
ユーザー承認必須（`.claude/rules/coding-rust.md`「テスト・ベンチ」節・
`.claude/rules/security.md`「自己修復ループ固有のガードレール」節と同じ
判断軸を適用する）。

### 5.2 FMA 契約

更新カーネル（CPU/CUDA/Metal のいずれも）は既存の丸め方針統一
（`.claude/rules/coding-rust.md`「バックエンド構成」節）に従う: CPU 参照
実装は `f32::mul_add`、GPU 側は各バックエンドの既定 FMA 契約
（CUDA NVRTC・Metal `simdgroup_multiply_accumulate` 相当）と揃える。

更新式の演算順序は §1 のアルゴリズム引用（weight decay 加算 → momentum
更新 → nesterov 分岐 → 減算）を全バックエンドで同一に固定する。順序が
バックエンド間で異なると、複合判定の閾値内であっても丸め誤差の蓄積パターンが
変わり得るため、#935 実装レビューで式順序の一致を明示確認する。

`linear_forward_resident`（§3.1・§3.3a）の出力は、同一入力・同一
weight/bias 値に対する既存 `gemm`+`add`（ホスト `Tensor<f32>` 契約）経路の
出力と本節と同じ統一複合判定を満たすことを #936 の parity テストへ
引き渡す（forward 経路を切り替えても数値契約は変わらないことの確認。
既定実装〈§3.2〉はホスト経路への委譲そのものであるため定義上一致するが、
CUDA/Metal が weight/bias 再アップロードを省略する最適化カーネルへ
オーバーライドした場合はカーネル実装差による丸め誤差が生じ得るため、
オーバーライド後も本判定での確認を要する）。

### 5.3 累積 parity の判定方式

単一ステップの判定（§5.1）に加え、#936 の parity テスト設計への引き渡し
事項として次を定める:

- **多ステップ累積判定**: 決定的シード（既存の決定的シード設定ユーティリティ
  を使用。`.claude/rules/coding-rust.md`「テスト・ベンチ」節）で 100 step
  程度学習を回し、CPU ホスト参照実装で計算した最終パラメータと、デバイス
  常駐経路（CUDA/Metal）で計算した最終パラメータを比較する。判定基準は
  単一ステップと同じ複合判定を最終値に適用する（中間ステップごとの判定は
  必須としない。誤差蓄積を最終値で捕捉する設計）。
- **ベンチ様式**: 5 回計測の中央値を採用する既存方針（`.claude/rules/
  coding-rust.md`「テスト・ベンチ」節）を #936 のベンチ（常駐化前後の
  per-step 時間比較）にも適用する。

### 5.4 境界検査

更新カーネルでも手動境界チェックを省略しない（REQ-8 境界検査規約。
`.claude/rules/coding-rust.md`「カーネル実装の境界検査」節）。性能目標
達成を理由にベクトル化ロード等で境界検査を省略する場合は、シェーダ側で
手動境界チェックを維持したうえで行う（既存 GEMM カーネルと同じ制約）。

## 6. 代替案と採否

| 案 | 内容 | 採否 |
|---|---|---|
| (a) 暗黙常駐キャッシュ方式 | `Tensor` 内部にデバイスキャッシュを持たせ、書き込み時に暗黙で同期する方式（candle の `Var::set` 型） | **不採用**。暗黙コピー禁止方針（`docs/public-api-design.md` §2.2.1）・型分離（ホスト `Tensor` とデバイス `DeviceBuffer` を明確に分ける現行設計）との不整合。stale 値の誤用検知も難しくなる |
| (b) `BackendOps` 全面 `DeviceBuffer` 化を先行 | forward/backward を含む全カーネル入口（`gemm`/`add`/`relu` 等すべて）を `DeviceBuffer` 契約へ一気に移行してから更新経路を設計する | **不採用（本イシュースコープでは）**。変更範囲が 4h 粒度を大きく超え、#931（デバイスハンドル再利用の公開 API 設計）・phase-4（#924）と重複する。§3.1・§3.3a で追加する `linear_forward_resident` はこの不採用案とは異なる: 対象は `Linear` forward の weight/bias 入力のみ（1 メソッドの追加）であり、入力バッチ・中間活性・backward・他の全カーネル（`add`/`relu`/畳み込み等）は既存のホスト `Tensor<f32>` 契約のまま変更しない。「全カーネル入口の一括移行」という本 (b) 案の対象範囲（forward/backward 全体）とは規模・影響範囲が異なるため、本設計のスコープに含めても 4h 粒度・#931/phase-4 との重複を引き起こさない |
| (c) 採用案 | §2〜§4 の非破壊拡張（`sgd_step_device`／`linear_forward_resident` デフォルトメソッド・`mem: &dyn MemoryOps` 明示引数）+ 常駐ストア（`DeviceParamStore`）+ 段階的常駐化（param/velocity のみ常駐、grad は毎ステップ 1 回 upload）+ `Sequential::forward_resident` による forward 結線・weight/bias の Tape leaf 登録による VJP 成立（§3.3a・§3.3b）+ 更新フェーズの poisoned 状態契約（§4.3） | **採用** |

## 7. スコープ境界・受け渡し

### #935（実装）への引き渡し事項

- §3.1 の確定シグネチャ（`sgd_step_device`／`linear_forward_resident`／
  `SgdStepConfig`。いずれも `mem: &dyn MemoryOps` 明示引数を持ち、
  `BackendOps` trait 定義への supertrait 追加はしない）をそのまま
  `tensor-core::backend_ops` へ追加する（`buffer.rs:11-13` の保留
  コメント「`MemoryOps` は `BackendOps` の supertrait となる想定」は
  本設計で**不採用と確定**したため、#935 で該当コメントを「supertrait
  化は crates.io 公開 trait への破壊的変更となるため不採用（§3.1）」
  という趣旨に更新する）。
- 各 `XBackendOps`（`CpuBackendOps`／`CudaBackendOps`／`MetalBackendOps`）
  へ `impl MemoryOps for XBackendOps` を追加する（既存 `XMemory` 実装への
  委譲か、フィールド保持かは §3.2 のとおり実装詳細として #935 が選ぶ。
  trait 構造・呼び出し形態への影響はどちらでもない。この `impl` は
  呼び出し側〈`DeviceParamStore`〉が `mem` 引数を得るための利便性であり
  `BackendOps` trait 自体は要求しない）。
- 実装順序: CPU → CUDA → Metal（既存 GEMM カーネルの実装順序・PoC 実証の
  蓄積と揃える）。
- フォールバック契約: §3.2 のデフォルト実装（`sgd_step_device`:
  `mem.download` → ホスト更新 → `mem.upload`。`linear_forward_resident`:
  `mem.download` → 既存 gemm+add 委譲）を先に用意し、各バックエンドが
  最適化カーネルでオーバーライドする形で段階的に置き換える（デフォルト
  実装のままでも正しく動作することを前提に、性能改善は独立した最適化
  として進められる）。
- `Sequential::forward_resident`（§3.3a・§3.3b）を
  `facade::compat::Sequential` へ追加し、`DeviceParamStore::new` が
  `trainable_parameters()` と同一の位置対応契約で層インデックス→
  `DeviceBuffer` 対応表を構築する。`forward_resident` は `&mut
  DeviceParamStore` に加え `mem: &dyn MemoryOps`（§3.3a。呼び出し元が
  `Sequential` 初期化時に把握している `BackendOps` と対になる
  `MemoryOps` 実装値をそのまま渡す）を受け取り、§3.3b のとおり forward
  呼び出しごとに weight/bias の downloaded スナップショットを `Tape`
  の leaf として新規登録したうえで、生成した `NodeId` を `TapeId`
  （§3.3b の `Tape` 同一性検査用）と併せて `store` 自身の一時状態
  （`pending_grad_sources`）へ書き込む。`pending_grad_sources` が既に
  `Some`（前回 forward 分が未消費）の状態での再呼び出しは §3.3b の
  とおり型付きエラーで拒否する。`tape.backward()` 後、呼び出し
  元は `DeviceParamStore::step(&mut self, tape: &Tape, grads:
  &Gradients, mem: &dyn MemoryOps, config: &SgdStepConfig)` を呼び、
  `step` はまず `tape` の `TapeId` が `pending_grad_sources` に保存された
  ものと一致するかを検査したうえで（不一致は型付きエラー。§3.3b）、
  内部で `Var::from_raw`（`var.rs:63`。`autodiff` クレート内部
  限定）+ `Gradients::get`（`backward.rs:49`）を使って
  `pending_grad_sources` から `grad_weight`／`grad_bias` を取り出す
  （新規 `Op`／`Tape` API 拡張は不要。既存 `Op::MatMul`/`Op::Add` の
  VJP・既存 `Gradients` API をそのまま再利用する。`Tape` に新設する
  `TapeId` 採番のみが #935 の新規実装事項となる）。
- `DeviceParamStore` の API（`new`/`step`/`sync_to_host` 等の確定
  シグネチャ）・`autodiff::optim` 内の配置は #935 が決める。決定済みの
  設計軸（`Sgd` を置き換えず `Sgd::step` 相当のロジックを §5.1 の参照
  実装として再利用する構成・§4.3 の poisoned 状態を表す `BackendError`
  variant を持つこと・§3.1 の `is_first_step` を用いた momentum 初回
  ステップの dampening 分岐）はそのまま踏襲する。
- **`DeviceParamStore::with_resident_buffers`（§3.3c）を実装する**:
  poisoned フラグ検査 → `ResidentLayerBuffers` 配列の構築 → 引数
  クロージャ呼び出し、の順で実装する。`DeviceParamStore` には
  `weight`/`bias`/`velocity_*` の生の `&DeviceBuffer` を返す public
  アクセサを一切追加しない（本メソッドが個別バッファへの唯一の
  アクセス経路であることの実装上の担保）。
- **`ActivationKind`／`Module::as_activation_kind`（§3.3c・`autodiff::nn`）
  を実装する**: `Relu`／`Sigmoid`／`Tanh` の 3 型が対応する variant を
  返すようオーバーライドする（既定 `None` は `Linear` を含む他の
  `Module` 実装がそのまま踏襲する非破壊拡張）。
- **`BackendOps::sigmoid`（§3.3c・`tensor-core`）を実装する**: CPU は
  #935 で即時追加（既存 `relu`/`tanh` と同じ扱い）、CUDA/Metal は
  §3.4 の実装順序（CPU → CUDA → Metal）に従って追って追加する。
  デフォルト実装（`BackendError::Unsupported`）はカーネル未実装の
  バックエンドでも trait のコンパイルを壊さない。
- **`Sequential::predict_resident`（§3.3c・`facade` クレート）を実装
  する**: `store.with_resident_buffers` のクロージャ内でのみ
  `ops.linear_forward_resident`（§3.1）を呼び、`self.layers`
  （既存 `forward`／`forward_resident` と同じ層イテレーション）を
  辿って `layer.as_linear()`／`layer.as_activation_kind()`（§3.3c）で
  層種別を判別し、`ActivationKind` に対応する `ops.relu`／
  `ops.sigmoid`／`ops.tanh` を適用する（Cursor Bugbot 指摘の解消。
  resident 推論の出力が非 resident の `predict` と一致することを
  #936 の parity テストで確認する。`Sigmoid` を含む `Sequential` で
  `ops.sigmoid` が `Unsupported` を返す間は resident 推論も
  fail-closed に失敗する契約であることをテストで確認する）。poisoned
  検査の迂回不能性は `with_resident_buffers` の型・可視性設計
  （§3.3c）そのものにより担保されるため、`facade`/`autodiff` 側の
  コードレビュー規律には依存しない。
- **`BackendOps::instance_id`／`MemoryOps::instance_id`（§3.2a）を
  実装する**: いずれもデフォルトメソッド（`self as *const Self as
  *const ()`）のため各バックエンド側でのオーバーライドは不要。
  `impl MemoryOps for XBackendOps`（§3.2）を追加した時点で自動的に
  正しい識別子を返す。
- **`MemoryOps::device()`（§3.2a）を `CpuMemory`／`CudaMemory`／
  `MetalMemory` へ実装する**: 各々 `Some(Device::Cpu)`／
  `Some(Device::Cuda(ordinal))`／`Some(Device::Metal)` を返す。§4.2 の
  既存 `param`/`grad` デバイス一致検査・診断用エラーメッセージで
  引き続き使用する（`self`/`mem` 同一性検査自体の合否は
  `instance_id()` が決定するため、本オーバーライドの有無は
  `sgd_step_device`／`linear_forward_resident`／`forward_resident`／
  `predict_resident` の `BackendMismatch` 判定には影響しない）。
- **`DeviceParamStore::abandon_pending_forward`（§3.3b 末尾）を実装
  する**: `pending_grad_sources` を副作用なくクリアする冪等メソッド。
  `step` を呼ばずに学習ループを中断・再試行するテストケース（回復
  シナリオの受け入れテスト）を #935 のテストに追加する。
- 更新カーネル（CUDA NVRTC・Metal compute shader）の `unsafe` 使用は
  FFI 境界に限定し、理由コメント + security-auditor レビュー必須
  （`.claude/rules/security.md`「unsafe」節）。

### #936（parity・ベンチ非後退確認）への引き渡し事項

- §5.3 の累積 parity 判定方式・ベンチ様式をそのまま使用する。
- 常駐化前後（旧経路 `Sgd::step` + `apply_parameters` 経由 vs 新経路
  `DeviceParamStore`）で per-step 時間を比較し、(b) の要因分離が有効に
  機能しているかを確認する（(a) の tape 初期化コストとは独立に、この
  比較単体で改善が見えることが望ましいが、#931 の tape 再利用が未着手の
  間は改善量が (a) に隠れる可能性がある点に留意する）。

### #931／#932 との整合前提

- #931（デバイスハンドル再利用の公開 API 設計）が確定するデバイス/tape
  再利用 API と、本設計の `DeviceParamStore`（tape 非依存の寿命）は独立
  だが、#935 の実装時に「tape 再利用時に `DeviceParamStore` をどう束ねるか」
  の結線点が生じる。本ドキュメントはこの結線の具体設計を確定しない
  （#931 側の結論を待つ）。
- #932（optimizer facade 公開の設計判断）が確定する公開面の意匠に
  `facade` 側の統合を従属させる（§3.4）。

### spec 更新の要否

本設計は REQ-2（数値一致複合判定・FMA 契約）・REQ-8（境界検査）の枠内で
完結し、`docs/spec/` の要件・受け入れ基準の変更を要しない見込みである。
実装中に spec 変更が必要と判明した場合は、`docs/spec/` を直接編集せず
正本リポジトリ（`Fandhe-AI/rust-ai-library-spec`）側での対応をユーザーへ
提案する（`.claude/rules/out-of-scope-tracking.md`「仕様変更が必要な場合」節）。

### スコープ外事項（本ツリー #933 の対象外）

- forward/backward の完全デバイス常駐化（`DeviceBuffer` 版 `BackendOps` への
  全面移行）— §6(b) 参照。接続先: #931・phase-4（#924）。
- **resident-aware な VJP**（`Op::MatMul`/`Op::Add` の host `Tensor<f32>`
  経路を経由せず、`DeviceBuffer` 上で weight/bias の勾配を直接計算する
  専用 Op）— §3.3b「転送モデルとの整合性の明記」参照。これが実現すれば
  forward ごとの weight/bias `download`（現設計で残る転送）をさらに
  削減できるが、変更範囲が forward/backward 全体の `DeviceBuffer` 契約
  移行（§6(b)）と同水準になるため本ツリーのスコープ外とする。接続先:
  §6(b) と同じく #931・phase-4（#924）。
- **複数 forward にまたがる勾配蓄積（マイクロバッチ学習）のネイティブ
  サポート**（§3.3b「2 回目以降の `forward_resident` 呼び出しに対する
  検証」参照）: 本設計は「1 forward・1 backward・1 step」を単位契約とし、
  `step` に未消費のまま 2 回目の `forward_resident` を呼ぶことを型付き
  エラーで拒否する（データを静かに欠落させる誤りを防ぐ fail-closed 措置）。
  複数 forward 分の勾配を合算してから 1 回 `step` する運用（マイクロ
  バッチ）を正式にサポートするための「位置ごとの勾配集約」機構は
  本ツリーのスコープ外とし、必要になった時点で別途 Issue 化する。
- f16 経路への拡張。
- optimizer 公開面（`facade` 側の意匠）の変更 — 接続先: #932。
