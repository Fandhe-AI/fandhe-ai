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
| CPU（環境 1・Apple M4 Max） | 18.185 ms | 797.5 µs | 626.5 µs | summary.md:74-76 |
| Metal（環境 1） | 48.845 ms | 751.8 µs | 1.606 ms | summary.md:77 |

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
fn sgd_step_device(
    &self,
    mem: &dyn MemoryOps,
    param: &mut DeviceBuffer<f32>,
    grad: &DeviceBuffer<f32>,
    velocity: Option<&mut DeviceBuffer<f32>>,
    config: &SgdStepConfig,
) -> Result<(), BackendError>;

/// `DeviceParamStore`（§3.3）常駐時の `nn::Linear` forward 入口
/// （#934 設計・#935 実装）。既存の `gemm`/`add` 系メソッド（ホスト
/// `Tensor<f32>` 入出力契約。§2）と異なり、`weight`/`bias` はデバイス
/// 常駐のまま渡す。`input`/戻り値は本段階（§3.3）のスコープに従い
/// ホスト `Tensor<f32>` のまま（forward/backward の完全デバイス常駐化は
/// §6(b) のとおりスコープ外）。オーバーライドしない場合の既定実装は
/// `mem.download(weight)`/`mem.download(bias)` でホスト値を得たうえで
/// 既存の `gemm`+`add` 相当の計算へ委譲する（CPU は §2 のとおり追加転送
/// コストなし。CUDA/Metal は最適化時、weight/bias の再アップロードを
/// 省略するオーバーライドに置き換える）。§3.3b のとおり、この
/// downloaded スナップショットは同時に Tape の leaf として登録される。
fn linear_forward_resident(
    &self,
    mem: &dyn MemoryOps,
    input: &Tensor<f32>,
    weight: &DeviceBuffer<f32>,
    bias: Option<&DeviceBuffer<f32>>,
) -> Result<Tensor<f32>, BackendError>;
```

`SgdStepConfig`（`tensor-core` 側に新設。`autodiff::optim::SgdConfig` の
lr・momentum・dampening・weight_decay・nesterov をそのまま保持する変換先の
設定構造体）は §1 のアルゴリズム引用（`sgd.rs:1-23`）と同一の意味を持つ
フィールド名にし、`autodiff` 側の変換関数（`impl From<&SgdConfig> for
SgdStepConfig` 等）は #935 が実装する。`tensor-core` は `autodiff` に依存しない
（既存の依存方向: `autodiff` → `tensor-core`）ため、`SgdStepConfig` は
`tensor-core` 側に独立定義し、`autodiff` 側が変換して渡す構成とする。

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
しない（§3.1 の非破壊性）。

### 3.3 段階設計: 「param + optimizer 状態のみ常駐、勾配は毎ステップ 1 回 upload」

現行 `Tape::backward` はホスト `Tensor` を返す（forward/backward の完全
デバイス常駐化は `DeviceBuffer` 版 `BackendOps`〈`upload`/`download` を
含む全面移行〉が前提であり、変更範囲が 4h 粒度を大きく超えるため #931・
phase-4（#924）へ接続するスコープ外事項とする。§5(b) 参照）。

そのため第 1 段（#935 スコープ）は次の分担とする:

- **常駐化する**: パラメータ本体（`weight`/`bias`）・momentum バッファ
  （`velocity`）。学習セッション開始時に 1 回 `upload` し、以降は
  `sgd_step_device` の in-place 更新のみで完結させる。
- **常駐化しない（毎ステップ 1 回転送）**: forward/backward の入出力
  （入力バッチ・中間活性・勾配）。`Tape::backward` が返すホスト勾配を
  `sgd_step_device` 呼び出し直前に 1 回だけ `upload` する。

これにより、現行「forward での param upload → backward 後の grad
download → ホスト更新 → 次ステップの param 再 upload」という毎ステップの
param 往復（§2 末尾）が「学習開始時の 1 回の upload + 終了時（または明示
同期時）の 1 回の download」に縮退する。grad の upload は本段階では残る
（forward/backward 自体が `Tensor<f32>` 契約のカーネルを呼ぶ限り、勾配は
一度ホストへ出てくるため）。

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
  受け取る形にする: `DeviceParamStore::new(params: &[TrainableParam])`
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
  `bias: Option<DeviceBuffer<f32>>`・`velocity: Option<DeviceBuffer<f32>>`
  （momentum 有効時のみ）の組であり、**1 キーに複数バッファを保持できる**
  構造とする（「層インデックス→単一 `DeviceBuffer`」という誤読を避ける
  ため本節で明示する）。`Sequential::forward_resident`（後述）は
  `Sequential.layers` を通常どおりイテレーションしつつ、`Linear` 層に
  出会うたびにこの trainable 層インデックスを別途カウントアップして
  対応するキーを引く（活性化層はカウントしない）。既存の位置対応契約
  そのものは変更しないため、新たな対応規則を追加で決定する必要はない。
- **新規 forward 入口**: `Sequential` に既存 `forward` を置き換えない
  追加メソッド `forward_resident(&self, tape: &Tape, input: &Var,
  store: &DeviceParamStore) -> Result<Var, AutodiffError>`（仮称。確定
  シグネチャは #935）を追加する。内部実装は既存 `forward` と同じ層
  イテレーションだが、各 `Linear` 層について自身のホスト `weight`/
  `bias` の代わりに `store` から対応するインデックスの `DeviceBuffer` を
  取得し、`BackendOps::linear_forward_resident`（§3.1）を呼ぶ。入力
  バッチ・中間活性・活性化関数（ReLU 等）は既存 `forward` と同一だが、
  `Tape`/`Var` への登録ロジックは既存 `forward` の登録（`Op::MatMul`
  ベース）をそのまま流用できない。weight/bias の勾配（VJP）が成立する
  ための結線は §3.3b で別途確定する（レビュー指摘 #934: line 233 の
  解消。旧稿の「登録ロジックは変更しない」という記述は誤りであり、
  本節で訂正する）。
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

**採用方針**: `linear_forward_resident` の既定実装（§3.2: `mem.download`
→ 既存 `gemm`+`add` へ委譲）と、weight/bias 再アップロードを省略する
最適化オーバーライドの両方に共通して使える形として、**forward 呼び出し
ごとに weight/bias の「その時点のスナップショット」を `Tape` の leaf
`Var` として登録する**方式を採る:

- `Sequential::forward_resident` は各層について、まず
  `mem.download(store.weight(layer_idx))`／
  `mem.download(store.bias(layer_idx))`（`MemoryOps`。§3.1 の `mem`
  引数と同じ実装値）でホスト `Tensor<f32>` スナップショットを取得し、
  これを `tape.leaf(snapshot)` として **毎 forward 呼び出しで新規に**
  登録する（前ステップの `NodeId` を再利用しない。§4.1 の「stale な
  ホスト値を現在のパラメータとして使わせない」契約と整合させるため、
  スナップショットは常にその回の `download` 結果のみを表す）。
  以降の計算（`gemm`+`add`・活性化関数）は既存 `forward` と同じ
  `Op::MatMul`／`Op::Add` 登録経路にそのまま乗せる。これにより
  `linear_forward_resident` は「downloaded スナップショットを作って
  既存 forward 経路へ渡す薄い橋渡し」として実装でき、既存の VJP
  実装（`Op::MatMul`/`Op::Add` の backward）を一切変更せずに
  weight/bias の勾配が計算できる。
  - 最適化オーバーライド（weight/bias の再アップロードを省略し
    device 上で完結させる実装）を採用するバックエンドであっても、
    **backward のために必要な最小限のスナップショット取得
    （`mem.download`）自体は省略しない**契約とする。省略できるのは
    「forward の計算そのものを host Tensor 経由の既存 `gemm` へ
    委譲する部分」（＝計算コストの重複）であり、「勾配を計算する
    ために weight/bias が host 値として少なくとも一度観測可能である
    こと」は省略しない。これにより §3.1 の最適化余地（再アップロード
    省略）と本節の VJP 成立要件は両立する。
- `Sequential::forward_resident` は各層で生成した weight/bias leaf の
  `NodeId` を（`Linear` 単体ではなく `Sequential` が §3.3a のとおり
  層インデックスを一元管理するため）層インデックスと対応付けて
  一時的に保持する（呼び出し中のみ有効なローカル対応表でよく、
  `DeviceParamStore` の恒久的な状態には含めない）。
- `tape.backward()` 実行後、この対応表を使って各層の `grad_weight`／
  `grad_bias`（ホスト `Tensor<f32>`。既存 `Tape::backward` が
  `NodeId` ごとに返す勾配と同一の仕組みで得られる）を
  `Sequential::trainable_parameters()` と同じ並び順（§3.3a）で
  取り出し、`DeviceParamStore::step`（§3.4・#935 が命名）へ渡す。
  これは §3.3 が既に定めた「`Tape::backward` が返すホスト勾配を
  `sgd_step_device` 呼び出し直前に 1 回だけ `upload` する」という
  転送モデルとそのまま整合する（新しい転送経路を追加しない）。
- **新規 Op／新規 Tape API は不要**: 既存 `Op::MatMul`/`Op::Add` の
  VJP 実装をそのまま再利用するため、`Tape` 側に「resident 専用の
  勾配側チャネル」等の拡張は要らない。`linear_forward_resident`
  自身は「スナップショットを取得して既存 forward 経路へ渡す」薄い
  合成として実装できることが、この結線が 4h 粒度の #935 実装に
  収まる根拠である。
- **既存 `forward`（非常駐）との差異はここに限定される**: 通常の
  `forward` は `Linear` が保持するホスト `Tensor` をそのまま
  leaf 登録するのに対し、`forward_resident` は `DeviceParamStore`
  からの `download` 結果を leaf 登録する。leaf 登録・VJP 計算の
  仕組み自体は共通である。

### 3.4 クレート責務分担

| クレート | 責務 |
|---|---|
| `tensor-core` | `sgd_step_device`／`linear_forward_resident` trait メソッド（`mem: &dyn MemoryOps` 明示引数。supertrait 化はしない非破壊拡張）・`SgdStepConfig` 型定義（§3.1・§3.2） |
| `backend-cpu`／`backend-cuda`／`backend-metal` | 各バックエンドのカーネル実装（CPU: 逐次または `rayon` 並列 in-place・CUDA: NVRTC 1 カーネル・Metal: compute shader 1 個）＋ `impl MemoryOps for XBackendOps`（§3.2。呼び出し側の利便性のための追加であり trait 定義上の要求ではない）。実装順序は CPU → CUDA → Metal（#935 引き渡し事項。§6） |
| `autodiff::optim` | `DeviceParamStore`（常駐ストア保持型。§3.3・§4 の所有者）・`SgdConfig → SgdStepConfig` 変換・`Tape` の勾配出力を `DeviceParamStore` へ渡す統合ロジック（§3.3b の leaf 登録・勾配取り出しを含む） |
| `facade`（`compat::Sequential` 経由で `Sequential::forward_resident` を実装） | `Sequential` への `forward_resident`・層インデックス対応表の保持（§3.3a）。公開面は #932 の optimizer facade 公開設計の結論に従属させる（本ドキュメントは compat 面の意匠を確定しない） |

### 3.5 既存 API との関係

`Sequential::apply_parameters`／`trainable_parameters`（ホスト `Tensor<f32>`
契約）は**公開契約として変更しない**（公開 API 非破壊はガードレール条件。
`.claude/rules/security.md` A08）。デバイス常駐経路（`DeviceParamStore` 経由の
学習ループ）は既存経路に追加する新経路であり、既存経路を置き換えない。
`DeviceParamStore` からホスト `Tensor<f32>` への同期（学習終了時・`predict`/
保存前）は明示 API（例: `DeviceParamStore::sync_to_host() -> Vec<Tensor<f32>>`。
確定シグネチャは #935 で行う）を介して行い、`apply_parameters` へ渡す形で
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

- **デバイス一致検査**: `param.device() == grad.device()`（`velocity` がある
  場合は同様に検査）でなければ `BackendError::DeviceMismatch` を返す
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
検証フェーズ: 全パラメータ・全 velocity の shape・device 一致を検査
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
- **`forward_resident` も poison チェック対象とする（Cursor Bugbot 指摘の
  解消）**: 旧稿は `step`／`sync_to_host` のみ poisoned 時に
  `BackendError` を返すと定めており、`Sequential::forward_resident`
  （§3.3a・§3.3b）が破損混在状態のバッファをチェックなしに読める抜け穴が
  残っていた。`forward_resident` は各層の `DeviceBuffer` を読む
  （`mem.download` する）**前に** `DeviceParamStore` の poisoned
  フラグを検査し、poisoned であれば計算を一切行わず同じ
  `BackendError`（poisoned variant）を返す契約とする。これにより
  「破損した混在状態のパラメータで forward が実行される」経路を閉じる
  （`step`／`sync_to_host`／`forward_resident` の 3 箇所すべてが
  poison チェックを通過して初めて `DeviceParamStore` の内部状態へ
  アクセスできることを、本設計が確定する fail-closed 契約とする）。

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
  `DeviceBuffer` 対応表を構築する。`forward_resident` は §3.3b の
  とおり、forward 呼び出しごとに weight/bias の downloaded スナップ
  ショットを `Tape` の leaf として新規登録し、`tape.backward()` 後に
  同じ層インデックス対応で `grad_weight`／`grad_bias` を取り出せる
  ようにする（新規 `Op`／`Tape` API 拡張は不要。既存 `Op::MatMul`/
  `Op::Add` の VJP をそのまま再利用する）。
- `DeviceParamStore` の API（`new`/`step`/`sync_to_host` 等の確定
  シグネチャ）・`autodiff::optim` 内の配置は #935 が決める。決定済みの
  設計軸（`Sgd` を置き換えず `Sgd::step` 相当のロジックを §5.1 の参照
  実装として再利用する構成・§4.3 の poisoned 状態を表す `BackendError`
  variant を持つこと）はそのまま踏襲する。
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
- f16 経路への拡張。
- optimizer 公開面（`facade` 側の意匠）の変更 — 接続先: #932。
