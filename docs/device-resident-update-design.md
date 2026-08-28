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
| デバイス転送抽象 | `MemoryOps`（`alloc_zeroed`/`upload`/`download`）と `DeviceBuffer<f32>`（`Box<dyn BufferHandle>` 経由の不透明ハンドル・RAII 解放・空テンソル契約・download 同期契約）は `tensor-core::buffer` に実装済みだが、`BackendOps` の supertrait としては未結線（TASK-1.9c 時点で意図的に未結線。`backend_ops.rs` 冒頭コメント。本設計〈§3.1〉は supertrait 化ではなく `BackendOps::memory_ops()` アクセサ〈`self` から `mem` を導出するデフォルトメソッド〉で結線する） | `crates/tensor-core/src/buffer.rs:1-13`・`crates/tensor-core/src/backend_ops.rs:10-27` |
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

### 3.1 採用方針: `BackendOps::memory_ops()` アクセサ経由で `mem` を導出する（supertrait 化はしない）+ デフォルトメソッド + 常駐 forward 入口

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

**採用方針（改訂: PR #951 レビューで下記の「明示引数」方式が構造的に
破綻することが判明したため、`BackendOps` 自身に `mem` を返させる方式へ
置き換える）**: 旧稿は `sgd_step_device`／`linear_forward_resident`
（および `DeviceParamStore::new`／`step`／`forward_resident`／
`predict_resident`）のすべてに `mem: &dyn MemoryOps` を**呼び出し元が
明示的に渡す引数**として追加していた。この方式は次の 2 点で破綻する
（Cursor Bugbot 指摘・codex-review P1 の解消。詳細は旧 §3.2a を参照して
いた自己/`mem` 同一性検査ごと本節で置き換える）:

1. **呼び出し元が「同じ具象値」を 2 通りの trait オブジェクトとして
   本当に用意できるとは限らない**。`DeviceParamStore::step` の唯一の
   実引数は `tape: &Tape`（`ops` を所有する）であり、呼び出し元が
   別途保持する `mem: &dyn MemoryOps` は**独立に構築・借用された値**に
   なりうる。両者が偶然にも同一具象値に由来していても、「別々の
   `Box`／別々の借用として渡される」構造自体は変わらない。
2. **`facade` は元より `&dyn BackendOps` を一切保持できない**
   （`REQ-12`。`facade::resolve_ops` は非公開関数で `BackendOps` を
   `facade` の外へ一切出さない）。`Sequential::forward_resident`／
   `predict_resident` は `facade` に実装されるため、旧稿が要求する
   `mem: &dyn MemoryOps` を呼び出し元（アプリケーションコード）から
   受け取る経路自体が REQ-12 の禁止する「任意 `BackendOps`／
   `MemoryOps` 実装の facade への注入」を再導入してしまう。

**解決策**: `mem` を独立引数として受け渡すのをやめ、`BackendOps` 自身に
「自分に対応する `MemoryOps` ビューを返す」デフォルトメソッドを持たせる。
これにより `mem` は常に `self` から導出され、`self` と `mem` が別々の
値になる可能性が構造的に消える（両者を独立に比較・検証する必要が
そもそもなくなる）:

```rust
pub trait BackendOps {
    // 既存メソッドは変更しない。

    /// この `BackendOps` 実装に対応する `MemoryOps` ビューを返す
    /// （codex-review P1・Cursor Bugbot 指摘の解消: `mem` を呼び出し元が
    /// 独立に用意する「明示引数」方式は、self と mem が別々の trait
    /// オブジェクトになりうる構造そのものが破綻の原因だった。本メソッド
    /// は `self` から `mem` を導出する唯一の経路とすることで、
    /// 「self と mem が同一具象値に由来するか」を検証する必要自体を
    /// 消す）。**デフォルトメソッドとして追加する**（`;` で終わる必須
    /// メソッドにすると crates.io 公開済み trait への破壊的変更になる
    /// ため、§3.1 冒頭の非破壊拡張方針を踏襲する）。既定は `None`
    /// （`MemoryOps` 未実装のバックエンドは `sgd_step_device`／
    /// `linear_forward_resident` の既定実装から `BackendError::
    /// Unsupported` を受け取る——fail-open ではなく fail-closed）。
    /// `impl BackendOps for XBackendOps`（#935）は `XBackendOps` 自身が
    /// `impl MemoryOps for XBackendOps`（§3.2）を持つ場合、`Some(self)`
    /// を返すようオーバーライドする。`&self`・ジェネリクスなし・
    /// `Self: Sized` 境界なしのため object safety を壊さず（`Box<dyn
    /// BackendOps + Send>` を含む既存呼び出し形態は変更されない）、
    /// `Some(self)` は `XBackendOps` が両方の trait を実装している
    /// 場合にのみ型検査を通る（`impl BackendOps for XBackendOps` の
    /// 実装ブロック内で `self: &XBackendOps` が `&dyn MemoryOps` へ
    /// unsize coercion されるだけであり、`unsafe` を要さない）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        None
    }
}
```

`sgd_step_device`／`linear_forward_resident` の両メソッドは、旧稿が
外部引数として要求していた `mem: &dyn MemoryOps` を**削除**し、既定
実装の内部で `self.memory_ops()` を呼んで導出する（`Ok`/`Some` で
なければ `BackendError::Unsupported` を返す）。`BackendOps` 自身が
`MemoryOps` を実装している必要は引き続きなく、trait 定義
（`pub trait BackendOps { ... }`）へ supertrait を追加しない点は
旧稿から変更しない。呼び出し側（`autodiff::optim::DeviceParamStore`。
§3.3a）は `mem` を一切保持・受け渡ししなくなり、`tape.ops()`
（`pub(crate)`。同一クレート `autodiff` 内でのみ到達可能）経由で得た
`&dyn BackendOps` に対して `.memory_ops()` を呼ぶだけでよい。

`tensor-core::BackendOps` に対し、次のメソッドを追加する（既存メソッドの
シグネチャ変更なし・trait 定義への supertrait 追加もなし。TASK-1.9c の
非破壊拡張方針〈`backend_ops.rs:19-27`〉を踏襲する真の非破壊拡張。
`memory_ops()` は本節冒頭ですでに追加済みのため、ここでは
`sgd_step_device` のみを追加する）:

```rust
/// デバイス常駐パラメータの in-place SGD 更新入口（#934 設計・#935 実装）。
///
/// `param` を `grad`（および `velocity`。momentum 有効時のみ `Some`）から
/// 直接更新し、ホストへの往復を発生させない契約とする。既定実装は
/// fail-safe（§3.2）。`mem` は外部引数として受け取らず、既定実装の内部で
/// `self.memory_ops()`（本節冒頭で新設）を呼んで導出する（`None` なら
/// `BackendError::Unsupported`）。CPU/CUDA/Metal 各バックエンドは自身の
/// カーネルでオーバーライドできる（デフォルト実装のままでも正しく
/// 動作する）。
///
/// **デフォルトメソッドとして trait 定義に本体を持たせる**（`;` で終わる
/// 必須メソッド宣言ではない。レビュー指摘 #934 のとおり、本体なしの
/// 必須メソッドとして追加すると、既存の外部 `impl BackendOps for X` を
/// 破壊する。§3.2 が確定する「fail-safe 合成をデフォルト実装とする」
/// 方針は、この本体をここに書くことで初めて成立する）。本体は次を
/// 呼ぶ（velocity は `Some` のときのみ同様に往復させる）:
fn sgd_step_device(
    &self,
    param: &mut DeviceBuffer<f32>,
    grad: &DeviceBuffer<f32>,
    velocity: Option<&mut DeviceBuffer<f32>>,
    config: &SgdStepConfig,
) -> Result<(), BackendError> {
    // 既定実装（§3.2）: self.memory_ops() で mem を導出（None なら
    // Unsupported）→ download → ホスト参照実装（`f32::mul_add`。
    // §5.2 の FMA 契約）で更新 → upload。実装詳細（ホスト側計算関数へ
    // の委譲）は #935 が確定する。
    let mem = self.memory_ops().ok_or(BackendError::Unsupported)?;
    default_sgd_step_via_host(mem, param, grad, velocity, config)
}
```

**旧稿が定義していた `linear_forward_resident`（Tape 非依存の resident
推論専用メソッド）は本改訂で削除する**（codex-review P1・Cursor Bugbot
指摘の解消。詳細は §3.3c を参照）。旧稿は「呼び出し元は 2 系統ある」と
述べ、系統 1（Tape 非依存の resident 推論）の主要呼び出し元として
`Sequential::predict_resident` を位置付けていたが、§3.3c の改訂により
`predict_resident` は `forward_resident`（系統 2）と同じ `Tape`/`Var`
経路を再利用する設計に変わったため、系統 1 を単独で担う呼び出し元が
本設計内に存在しなくなった。`tensor-core` に呼び出し元のない default
メソッドを追加すると、実装（#935）・数値一致テスト（#936）双方の対象が
実際には駆動されない「死んだ拡張点」になり、`.claude/rules/coding-rust.md`
の品質基準（未使用コードを残さない）とも整合しないため、本設計の
スコープからは除外する。`DeviceParamStore` に一切紐付かない、呼び出し元が
独立に保持する `DeviceBuffer` に対する resident 推論が将来必要になった
場合は、その時点で改めて別 Issue として設計する（`sgd_step_device` と
同じ `memory_ops()` 導出パターンをそのまま踏襲できる）。

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
Unsupported` を返す設計だが（`backend_ops.rs:28-33`）、`sgd_step_device`
は**それとは異なり「必ず成立するが遅い」合成をデフォルト実装とする**:

```text
sgd_step_device の既定実装 = self.memory_ops() で mem を導出（None なら
                             Unsupported）→ download(param) → ホスト
                             参照実装（CPU と同一式順序・f32::mul_add）
                             で更新 → upload(param) で書き戻し
                             （velocity も同様に往復）
```

採否理由: `Unsupported` 一辺倒にすると、#935 の実装順序（CPU → CUDA →
Metal）の途中で「まだデバイスカーネルを実装していないバックエンド」を
使うコードパスがこのメソッドを呼べず、`autodiff::optim` 側に「両方の
経路（デバイス常駐 API と旧ホスト API）を呼び分ける分岐」を恒久的に
持たせることになる。デフォルト実装が fail-safe に成立していれば、
`autodiff::optim::DeviceParamStore`（§3.3）は常に `sgd_step_device`
だけを呼べばよく、各バックエンドが最適化されたカーネルを持つかどうかは
実装詳細に留められる（`self.memory_ops()` が `None` を返す＝`MemoryOps`
未実装のバックエンドについては `Unsupported` を返す。これは「必ず成立
する」対象を「`MemoryOps` を実装済みのバックエンド」に限定するもので
あり、fail-safe 方針と矛盾しない——fail-open ではなく fail-closed の
`Unsupported` である点が §3.3c で新設する `sigmoid` 等の通常カーネルと
同格の扱いに揃う）。

**object safety・非破壊性の確定**（レビュー指摘 #934: line 157。旧稿は
`BackendOps` が `MemoryOps` の supertrait ではないため、デフォルト実装が
`Self: MemoryOps` を要求すると `Box<dyn BackendOps + Send>`（既存の
呼び出し形態。`facade`/`autodiff` 側が保持する型）から呼び出せない、と
して supertrait 化を提案していたが、これは crates.io 公開済み trait への
破壊的変更になるため不採用と確定した。§3.1 のとおり、`BackendOps::
memory_ops()`（`&self` を取るだけの object-safe なデフォルトメソッド）
経由で `mem` を導出する方式を採用する。`MemoryOps` は `buffer.rs:203`
のドキュメンテーションコメントが明記するとおり「object-safe に設計
している（`&dyn MemoryOps` として扱える）」ため、`Option<&dyn
MemoryOps>` を返り値として持ち回ること自体に制約はない。
`sgd_step_device` のデフォルト実装は `self.memory_ops()` の戻り値へ
`mem.download(..)`／`mem.upload(..)` を呼ぶだけでよく、`BackendOps`
trait 定義（`Box<dyn BackendOps + Send>` を含む既存の呼び出し形態）は
一切変更されないため object safety・既存実装型の互換性いずれも
保たれる。

現状 `CpuBackendOps`／`CudaBackendOps`／`MetalBackendOps`（`BackendOps`
実装型）と `CpuMemory`／`CudaMemory`／`MetalMemory`（既存の `MemoryOps`
実装型。`backend-{cpu,cuda,metal}/src/memory.rs`）は別構造体である
（`ops.rs`／`memory.rs` 実装箇所を実測確認済み）。#935 は各
`XBackendOps` へ `impl MemoryOps for XBackendOps` を追加し（既存
`XMemory` の実装本体へ委譲する形でよく、`XBackendOps` が `XMemory` を
フィールドとして保持するか、都度その場で構築するかは実装時に選べる
実装詳細とする）、`impl BackendOps for XBackendOps` の `memory_ops()`
オーバーライドを `Some(self)` にする（`XBackendOps` 自身が両方の
trait を実装しているため、`self: &XBackendOps` を `&dyn MemoryOps`
として unsize coercion で返すだけであり `unsafe` を要さない）。外部の
`BackendOps` 実装型がこの `impl` を持たない場合でも、`BackendOps`
trait 自体のコンパイルには一切影響しない（§3.1 の非破壊性。
`memory_ops()` の既定 `None` がそのまま使われ、`sgd_step_device` は
`Unsupported` を返すのみ）。

**旧稿にあった `self`（`BackendOps`）と `mem`（`MemoryOps`）の同一性
検査（§3.2a・`instance_id()`／`BackendMismatch`）は本改訂で削除する**
（Cursor Bugbot 指摘「Pointer identity rejects valid backends」「ZST
instance IDs are not unique」の解消）。理由: 旧稿は `mem` を「呼び出し元が
独立に用意して渡す引数」として扱っていたため、`self` と `mem` が本当に
同一具象値に由来するかを実行時に検証する必要があった。しかしこの検証は
2 つの理由で構造的に機能しなかった——(a) `self as *const Self as *const
()` によるポインタ比較は、`DeviceParamStore::step` が `tape` 所有の
`Box<dyn BackendOps>` と呼び出し元が別途保持する `mem` を渡す構成である
限り、両者が同一の `Box` 割り当てになることはなく、実運用の呼び出しが
常に不一致と判定される（正当な呼び出しを fail-closed に拒否する）。
(b) `CpuBackendOps`／`MetalBackendOps` のような ZST（ゼロサイズ型）は
異なるインスタンスへの参照がダングリングポインタとして同一アドレスを
指しうるため、`instance_id()` は異なるインスタンスを区別できない場合が
ある（fail-open）。§3.1 の改訂（`mem` を `self.memory_ops()` から導出する
方式へ変更）はこの検査が対象としていた「`self` と `mem` が別々の値に
なりうる」という前提そのものを消す。`mem` は常に `self` 自身が返す値で
あるため、両者が異なる具象値に由来する余地が構造的に存在せず、実行時の
同一性検査は不要になる。新設した `BackendError::BackendMismatch`
variant（旧稿）もこの用途では不要となるため追加しない（§4.2 の
`param`/`grad` 間デバイス一致検査は `DeviceBuffer::device()` の値比較を
用いる別の既存検査であり、本節の削除とは無関係にそのまま維持する）。
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
  受け取る形にする: `DeviceParamStore::new(tape: &Tape, params:
  &[TrainableParam])`（host `Tensor<f32>` から `DeviceBuffer<f32>` への
  upload を行うため `mem` を要するが、`mem: &dyn MemoryOps` を外部引数
  として受け取ることはしない。**`DeviceParamStore` は `autodiff::optim`
  に属し、`Tape`（`autodiff::tape`）と同一クレートであるため、`Tape::
  ops()`〈`pub(crate)`。`tape.rs:308`〉へ直接アクセスできる**——この
  crate 内可視性は `DeviceParamStore::new` の呼び出し元が `facade` で
  あっても崩れない（`pub(crate)` は「そのアイテムを定義したコード」の
  可視性であり、「そのコードを呼び出す側」の可視性ではないため、
  `DeviceParamStore::new` の実装本体〈`autodiff` 側で書かれる〉は
  `tape.ops()` を呼べる。§3.3a 末尾の `forward_resident` シグネチャで
  この非対称——`autodiff::optim` は `tape.ops()` に届くが `facade` は
  届かない——を利用する）。したがって `new` は内部で `tape.ops().
  memory_ops()`〈§3.1〉から `mem` を導出し、`self.device = tape.ops().
  device()`〈`BackendOps::device()`。既存の必須メソッドで常に正しい値を
  返す〉を保持する（後述 `Sequential::predict_resident` が
  `facade::tape_for(store.device())` を呼ぶために必要。§3.3c）。
  旧稿の `mem: &dyn MemoryOps` 明示引数は削除する（§3.1 の改訂と同じ
  理由: 呼び出し元〈facade〉が独立に `mem` を用意する経路は REQ-12 が
  禁じる `BackendOps`/`MemoryOps` 実装の facade への注入を再導入する
  ため。codex-review P1「'forward_resident' では要求されたバックエンド
  同一性検査を実装できない」の解消——この経路を `tape: &Tape` 単独に
  揃えることで、`forward_resident`〈後述〉も同じパターンを踏襲できる）。
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
- **`DeviceParamStore::device(&self) -> Device`**（新設。`new` が
  `tape.ops().device()` から取得し内部フィールドとして保持した値を
  そのまま返す読み取り専用アクセサ）。`BackendOps::device()`
  （`backend_ops.rs:86`）は既存の**必須**メソッド（デフォルトを持たない
  ため実装漏れがありえない）であり、旧稿が §3.2a で持ち出した
  `MemoryOps::device()`（デフォルト `None`。実装漏れで fail-open しうる）
  とは異なる、常に信頼できる値である。この `device()` アクセサは
  `Sequential::predict_resident`（§3.3c）が「`store` に対応する
  `Tape` を自前で構築する」ために使う。
- **新規 forward 入口**: `Sequential` に既存 `forward` を置き換えない
  追加メソッド `forward_resident<'t>(&self, tape: &'t Tape, input: &Var<'t>,
  store: &mut DeviceParamStore) -> Result<Var<'t>, AutodiffError>`
  （仮称。確定シグネチャは #935）を追加する。**`mem`／`ops` のいずれも
  引数に持たない（codex-review P1「'forward_resident' では要求された
  バックエンド同一性検査を実装できない」の解消）**: 旧稿は本メソッドが
  `mem: &dyn MemoryOps` を明示引数として受け取ると定めていたが、
  `Sequential`（`facade::compat::sequential::Sequential`）は REQ-12 に
  より `&dyn BackendOps`/`&dyn MemoryOps` のいずれも一切保持・受け取り
  できない（`facade::resolve_ops` は非公開関数。§3.1 冒頭で確定した
  破綻理由 2 点のうち (2) がここで顕在化していた）。**採用方針**:
  `mem` を要する「weight/bias スナップショットの download + `Tape` leaf
  登録」処理そのものを `facade` から `autodiff::optim::DeviceParamStore`
  側の新設メソッドへ移す（§3.3b で確定する
  `DeviceParamStore::register_resident_leaves`）。`facade` の
  `forward_resident` はこのメソッドを `tape`（自身がすでに引数として
  持つ）だけで呼び出せばよく、`mem`/`ops` に一切触れない。**`store`
  を `&mut` で受け取る**（`&` ではない）: §3.3b のとおり、この呼び出しが
  生成する weight/bias leaf の `NodeId` を `store` 自身の内部状態として
  書き込むため（呼び出し元がこの対応表を別途持ち回る必要をなくす。
  Cursor Bugbot 指摘の解消。下記参照）。内部実装は既存 `forward` と
  同じ層イテレーションだが、各 `Linear` 層について自身のホスト
  `weight`/`bias` の代わりに `register_resident_leaves` が返す
  `Var`（`store` から download・leaf 登録済み）を使う。weight/bias の
  勾配（VJP）が成立するための結線は §3.3b で確定する（レビュー指摘
  #934: line 233 の解消。旧稿の「登録ロジックは変更しない」という
  記述は誤りであり、本節で訂正する）。
- **既存 `forward` との関係**: 既存 `forward`（ホスト `Tensor` のみで
  完結）は変更・削除しない（§3.5 の「既存 API 非破壊」方針）。
  `DeviceParamStore` を使う学習ループは `forward_resident` を呼び、
  `predict`／保存等は既存 `forward` + 明示同期後の `apply_parameters`
  を使う（§3.5）。

### 3.3b 常駐パラメータの勾配経路・VJP（レビュー指摘 #934: line 233 の解消）

**指摘の核心**: 旧稿が §3.1 に置いていた `linear_forward_resident` は
`weight`/`bias` を `DeviceBuffer` として直接受け取るのみで、既存の
`Tape::backward` が勾配を計算できる対象（`Op::MatMul` が保持する
`NodeId` に対応する `Var` leaf）としては一切登録されない構造だった。
旧稿はこの点を「forward の配線を変えるだけ」と過小評価しており、実際
には weight/bias に対する VJP が成立しない（`Sgd::step`／
`sgd_step_device` が必要とする `grad` の出所が設計上欠落する）。本節は
この結線を確定する（`linear_forward_resident` 自体は §3.1 の改訂で
本設計から削除済みであり、以下は §3.1 とは独立に完結する）。

**採用方針（改訂: `mem` への到達不能な `facade` の代わりに
`autodiff::optim` 側へ処理を寄せる）**: 旧稿は `Sequential::
forward_resident`（`facade` に実装）自身が `store.with_resident_
buffers` のクロージャ内で `mem.download` を呼ぶ構成だったが、§3.3a の
改訂により `forward_resident` は `mem` を一切受け取らない。そこで
weight/bias の「その時点のスナップショットを download し `Tape` の
leaf として登録する」処理そのものを `autodiff::optim::
DeviceParamStore` 側の新設メソッドへ移す:

```rust
impl DeviceParamStore {
    /// weight/bias の現在値を download し、`tape` の leaf `Var` として
    /// 登録したうえで、生成した `NodeId` を `TapeId` と併せて
    /// `pending_grad_sources`（下記）へ書き込む（`forward_resident`
    /// 専用。§3.3a）。`&mut self` で受け取る理由: 本メソッド自身が
    /// `pending_grad_sources` を書き込む（呼び出し元の `facade` 側に
    /// 書き込みロジックを持たせない。Cursor Bugbot 指摘の解消。下記
    /// 参照）。
    ///
    /// **`mem`/`ops` を一切受け取らない（codex-review P1 の解消）**:
    /// `DeviceParamStore` は `autodiff::optim` に属し、`Tape`
    /// （`autodiff::tape`）と同一クレートであるため、本メソッドの
    /// 実装本体は `tape.ops()`〈`pub(crate)`。§3.3a〉→
    /// `.memory_ops()`〈§3.1〉という経路で `mem` を自ら導出できる。
    /// 呼び出し元（`facade`）はこの経路に一切触れない。
    ///
    /// 内部実装: `self.with_resident_buffers(|buffers| { ... })`
    /// （poisoned 検査済みスコープ付きクロージャ。§3.3c）を 1 回呼び、
    /// クロージャ内で各 trainable 層の `buffers[i].weight`／
    /// `buffers[i].bias` を `mem.download(..)` してホスト
    /// `Tensor<f32>` スナップショットを取得し、`Vec<(usize,
    /// Tensor<f32>, Option<Tensor<f32>>)>` として**所有権ごと**
    /// クロージャの戻り値へ積んで返す（§3.3c の `for<'a>` HRTB 境界
    /// により、`&DeviceBuffer` 参照自体はクロージャのスコープ外へ
    /// 一切持ち出せない。持ち出せるのは download 済みの独立した
    /// ホスト値のみ）。`with_resident_buffers` の呼び出しが返った
    /// 時点（クロージャの借用スコープが終わった時点）で `self` への
    /// `&mut` アクセスが再び可能になるため、続けて各スナップショットを
    /// `tape.var(&snapshot)`（既存 API。`tape.rs:288`）で**毎回新規に**
    /// leaf 登録し（前ステップの `NodeId` を再利用しない。§4.1 の
    /// 「stale なホスト値を現在のパラメータとして使わせない」契約と
    /// 整合）、生成された `NodeId` 列を `tape.id()`（`TapeId`。下記）と
    /// 併せて `self.pending_grad_sources` へ書き込んでから、呼び出し元へ
    /// 返す `Var<'t>` 列を組み立てる。poisoned であれば
    /// `with_resident_buffers` がクロージャを一切呼ばず `BackendError`
    /// （poisoned variant）を返すため、本メソッドはこの 1 回の呼び出し
    /// だけで §4.3 の poison チェックを自動的に満たす。
    ///
    /// 戻り値の `ResidentLeafVars<'t> { trainable_idx: usize, weight:
    /// Var<'t>, bias: Option<Var<'t>> }`（新設。`autodiff::optim` に
    /// 定義し `pub` とする——`facade` が層イテレーション時にこの型を
    /// 直接扱うため）は trainable 層インデックス順（§3.3a の対応表と
    /// 同じ並び）の `Vec` として返す。
    pub fn register_resident_leaves<'t>(
        &mut self,
        tape: &'t Tape,
    ) -> Result<Vec<ResidentLeafVars<'t>>, BackendError> {
        // 実装詳細は #935 が確定する。
        unimplemented!()
    }
}
```

- `Sequential::forward_resident`（facade）は `store.register_resident_
  leaves(tape)?` を 1 回呼び、返る `Vec<ResidentLeafVars<'t>>` を
  trainable 層インデックスで引きながら `self.layers` を辿る。`Linear`
  層では対応する `ResidentLeafVars` から `fandhe_ai_autodiff::nn::
  linear::LinearVars { weight, bias }`（既存の公開フィールド型。
  `Linear::bind`〈`linear.rs:131`〉が返すものと同型）を構築して
  `.forward(&current)` を呼ぶ（既存 `SequentialVars::forward`
  〈`sequential.rs:315`〉が `self.linears`〈`Linear::bind` 由来の
  `LinearVars`〉に対して行っているのと全く同じ呼び出しパターンを、
  `DeviceParamStore` 由来の `LinearVars` に対して行うだけである）。
  活性化層では既存 `forward`／`SequentialVars::forward` と同じく
  `layer.forward(&tape.0, &current)`（`Module::forward` への多態
  dispatch）を呼ぶ。**この設計により、`forward_resident` の計算列は
  `SequentialVars::forward` と同一の `Var` API・同一の `Module::
  forward` 経路をたどるため、`Op::MatMul`/`Op::Add`/活性化関数いずれの
  VJP・数値計算も既存実装をそのまま再利用でき、新規の弱いレイヤ
  （旧稿の `linear_forward_resident`／`ops.sigmoid` 等）を一切必要と
  しない**。`facade` は `mem`／`ops`／`BackendOps`／`MemoryOps` の
  いずれにも一切触れない。
- **NodeId の保持先は `DeviceParamStore` 自身**（Cursor Bugbot 指摘の
  解消: 呼び出し元が `Gradients::get` 用のハンドルを得られない問題）。
  上記のとおり `register_resident_leaves`（`&mut self` で `store` 自身が
  実行する。`forward_resident` から見れば `store.register_resident_
  leaves(tape)?` という 1 回の呼び出しの内部で完結する）が、生成した
  weight/bias leaf の `NodeId`（`Var` 自体ではなく `NodeId`。`Var<'t>`
  は `Tape` への借用を持つため `DeviceParamStore`〈`Tape` とは独立の
  寿命。§4.1〉のフィールドとして保持できない）を、層インデックスと
  対応付けて自身へ書き込む
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
  self, tape: &Tape, grads: &Gradients, config: &SgdStepConfig) ->
  Result<(), BackendError>`（仮称。確定シグネチャは #935）を呼ぶ。
  **`mem` を外部引数として受け取らない**（`new`／`register_resident_
  leaves` と同じ理由: `step` は `autodiff::optim` に属し `tape.ops()`
  〈`pub(crate)`〉→ `.memory_ops()`〈§3.1〉で内部から導出できるため。
  §3.1 の `sgd_step_device`〈`self.memory_ops()` で導出〉と同じ経路を、
  `step` 自身も `tape.ops()` を `self` として辿ることでそのまま使える）。
  `step` はまず上記の `TapeId` 一致検査を行い、通過した
  場合のみ内部で `pending_grad_sources` の各 `NodeId` に
  ついて `Var::from_raw(tape, node_id)`（`autodiff` クレート内部限定の
  `pub(crate)` コンストラクタ。`autodiff::optim` は同一クレートのため
  呼べる。`var.rs:63`）で `Var` を再構築し、`grads.get(&var)`
  （`backward.rs:49`。既存 `Gradients` API をそのまま使う）で
  `grad_weight`／`grad_bias`（ホスト `Tensor<f32>`）を取り出したうえで、
  `Sequential::trainable_parameters()` と同じ並び順（§3.3a）で
  `tape.ops().sgd_step_device(..)` を 1 パラメータずつ upload・
  呼び出しする。これは
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

**指摘の核心 (1)（poisoned 検査の迂回）**: 旧稿が §3.1 に置いていた
`BackendOps::linear_forward_resident`（本改訂で削除済み。§3.1）の
「呼び出し元は 2 系統」記述は、系統 1（Tape 非依存の resident 推論）を
「このメソッドを直接呼ぶ」とだけ定めており、`weight`/`bias` の
`DeviceBuffer` を `DeviceParamStore` から読み出して渡す具体的な経路に
は一切触れていなかった。`DeviceParamStore` は `sgd_step_device` の
実行時エラー（§4.3）により poisoned 状態へ遷移しうるが、poisoned 検査は
`step`／`sync_to_host`／`forward_resident` の 3 箇所にしか課されて
おらず（旧稿の §4.3 最終箇条）、`linear_forward_resident` を直接呼ぶ
resident 推論はこの 3 箇所のどれにも該当しなかった。結果として、
更新途中の実行時エラーで一部パラメータのみ更新された poisoned な
`DeviceParamStore` から `weight`/`bias` の `DeviceBuffer` 参照を
取り出し、それをそのまま渡せば poisoned フラグを一切検査せずに破損
混在状態のバッファで推論が実行できてしまう（fail-closed 契約・
AGENTS.md「fail-closed の維持（P0）」への抵触）という問題があった。
本改訂で `linear_forward_resident` 自体を削除したため、この特定の
迂回経路は消えたが、下記の `with_resident_buffers`（個別バッファへの
唯一のアクセス経路）は引き続き必要である——理由は指摘 (2) のとおり、
迂回不能性を型・可視性で構造的に強制するためであり、`linear_forward_
resident` の有無に依存しない設計上の要請である。

**指摘の核心 (2)（迂回防止の実効性）**: `DeviceParamStore` は
`weight`/`bias`/`velocity_*` の生の `&DeviceBuffer` を返す public
アクセサを持たない設計を維持する。`facade` 側の呼び出し元
（`Sequential`）が `store` 内のバッファ由来の値（download 済みホスト
スナップショット）を得るための唯一の経路を、poisoned 検査済みスコープ
付きクロージャ API（`with_resident_buffers`。下記）に限定し続ける。

**指摘の核心 (3)（Cursor Bugbot・活性化関数の欠落・sigmoid parity）**:
旧稿の `predict_resident` は `DeviceParamStore`（`autodiff::optim`。
`Linear` 層の `weight`/`bias` のみを保持し、`Sequential` の層構成
〈活性化層を含む〉を一切知らない型。§3.3a）の外側で、生の `BackendOps`
（`ops.linear_forward_resident`／新設 `ops.sigmoid` 等）を直接呼ぶ
実装を想定していた。これは 2 つの問題を生む: (a) `facade` が `ops`／
`mem` を保持できない（REQ-12・codex-review P1「'predict_resident' が
facade に任意の 'BackendOps' 注入経路を再導入する」）。(b) 活性化関数
の適用が既存 `predict`（`Var::sigmoid` → `eval::sigmoid` という
確立された数値経路）とは別の新設 `BackendOps::sigmoid`（素朴な
`1 / (1 + exp(-x))`）を経由するため、両者の数値表現が食い違う保証が
なく §5.1 の parity 契約を満たすとは限らない（Cursor Bugbot 指摘
「Sigmoid path breaks predict parity」）。

**採用方針（迂回不能な構造的強制 + `forward_resident` と同一経路の
再利用による parity の構造的保証）**: 上記を同時に解消するため、
責務を次のとおり分離する:

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
      /// 由来のバッファを個別バッファアクセサ経由で外部へ持ち出す」
      /// という迂回は、docstring 上の禁止ではなく型システムにより
      /// 不可能になる（このため、§3.3b の `DeviceParamStore::
      /// register_resident_leaves` はクロージャ内で `mem.download`
      /// した「所有権を持つホスト `Tensor<f32>` のスナップショット」
      /// だけを `R` として持ち出し、クロージャのスコープを抜けたあとに
      /// `self` への `&mut` 書き込みを行う構成を取れる）。poisoned
      /// であれば `f` を一切呼ばず（クロージャの中身が実行されない
      /// ため、破損混在状態のバッファに触れる経路自体が生じない）、
      /// `f` が要求する戻り値型と同じ `Result` 型の
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

- **`Tape` 依存の resident 推論エントリを `Sequential` 側に置く**
  （`facade::compat::Sequential::predict_resident`。`forward_resident`
  〈§3.3a〉と対になる読み取り専用版。改訂: 旧稿の「Tape 非依存」という
  設計軸は撤回する——理由は指摘 (3) のとおり、`Tape` を経由しない独自
  経路は既存 `predict`／`forward_resident` の活性化関数の数値経路と
  分岐し、parity を構造的に保証できないため）。`Sequential` は既に
  層構成（`Linear` と活性化層を含む並び。§2・§3.3a）を唯一把握する
  オブジェクトであるため、活性化関数の適用箇所を正しく判断できるのは
  `Sequential` のみである点は旧稿から変更しない。

  **採用方針（`forward_resident` と同一の `Var`/`Module::forward` 経路を
  再利用する。ActivationKind／`BackendOps::sigmoid` は新設しない）**:
  `predict_resident` は次の 3 手順のみで構成する。

  1. **`facade::tape_for(store.device())?`（既存の公開 composition
     root。§3.1 末尾で新設した `DeviceParamStore::device()`）で、
     `store` が常駐する先と同じデバイス種別の `Tape` をこの呼び出し
     専用に構築する**。これは既存 `predict`（`crate::tape()` で
     CPU 既定の `Tape` を毎回構築する。`sequential.rs:126`）と同じ
     設計軸——**呼び出しごとに `Tape` を生成・破棄する**——を、
     デバイスを `store.device()` に合わせて踏襲したものである。
     `facade::tape_for` は既に公開 API（REQ-12 が許す唯一の
     `BackendOps` 解決経路）であるため、この手順は `ops`／`mem` の
     いずれにも触れない。
  2. **`store.snapshot_resident_leaves(&tape)`（`autodiff::optim`。
     新設。下記）を 1 回呼び、`Vec<ResidentLeafVars<'t>>` を得る**。
     `register_resident_leaves`（§3.3b。`forward_resident` 専用）との
     違いは `&self` で受け取り `pending_grad_sources` を一切書き込まない
     点のみ（`predict_resident` はこの呼び出し後に `step` を期待
     しないため、学習ループ側の `pending_grad_sources` 状態と干渉
     しない——学習中の `store` に対して `predict_resident` を呼んでも
     進行中の forward/step サイクルを壊さない）。内部実装は
     `with_resident_buffers` を経由して poisoned 検査を必ず先に行う
     （§3.3c 冒頭の指摘 (1)(2) の解消はここでも同様に成立する）。

     ```rust
     impl DeviceParamStore {
         /// [`DeviceParamStore::register_resident_leaves`] の読み取り専用版
         /// （`predict_resident` 専用）。`pending_grad_sources` を
         /// 書き込まない・`TapeId` 一致検査の対象にもしない点のみが
         /// 異なり、download・leaf 登録の内部実装（`with_resident_
         /// buffers` 経由の poisoned 検査を含む）は共有する。
         pub fn snapshot_resident_leaves<'t>(
             &self,
             tape: &'t Tape,
         ) -> Result<Vec<ResidentLeafVars<'t>>, BackendError> {
             // 実装詳細は #935 が確定する。
             unimplemented!()
         }
     }
     ```

  3. **`self.layers` を `forward_resident`（§3.3b）と全く同じパターンで
     辿り、`Linear` 層では `LinearVars { weight, bias }.forward(&current)`
     （`fandhe_ai_autodiff::nn::linear::LinearVars`。手順 2 の
     `ResidentLeafVars` から構築）を、活性化層では `layer.forward(&tape.0,
     &current)`（`Module::forward` への多態 dispatch）を呼ぶ。最終的な
     `Var` を `.to_tensor()`（既存 `predict` と同じ。`sequential.rs:128`）
     でホスト `Tensor<f32>` へ変換して返す。**この手順は `forward_resident`
     の層イテレーションと完全に同一のコードパスであり**（違いは
     `tape.backward()` を呼ばない点のみ）、活性化層は `Module::forward`
     経由で既存 `Var::sigmoid`／`Var::relu`／`Var::tanh` → 既存
     `eval::sigmoid` 等をそのまま通る。これにより predict_resident の
     出力は既存 `predict`（同じ `Module::forward` 経路）と**構造的に
     同一の数値計算列**になり、§5.1 の parity 契約を「テストで確認する」
     のではなく「同じコードパスを通る設計そのもの」で満たす
     （Cursor Bugbot 指摘「Sigmoid path breaks predict parity」の解消）。
     新設した `ActivationKind`／`Module::as_activation_kind`／
     `BackendOps::sigmoid`（いずれも旧稿）はこの設計では不要になる
     ため追加しない——`Module::forward` の既存多態 dispatch がそのまま
     活性化種別を判別済みであり、`predict_resident` 側で改めて種別を
     判別する必要がないため（旧稿が `Tape` 非依存の生 `BackendOps` 経路
     を取っていたために、この既存の判別機構を使えず新設が必要になって
     いた。本改訂はその前提自体を解消する）。

  ```rust
  impl Sequential {
      /// `store` 常駐のパラメータを使う resident 推論（`predict` 相当）。
      /// `forward_resident`（§3.3a）と対になる読み取り専用 API
      /// （`&self`。`store` も `&DeviceParamStore` で十分——書き込みは
      /// 発生しない）。内部で `facade::tape_for(store.device())` により
      /// 専用の `Tape` を構築する（`ops`／`mem` を一切外部から受け取ら
      /// ない。codex-review P1「'predict_resident' が facade に任意の
      /// 'BackendOps' 注入経路を再導入する」の解消）。
      pub fn predict_resident(
          &self,
          store: &DeviceParamStore,
          input: &Tensor<f32>,
      ) -> Result<Tensor<f32>, AutodiffError> {
          // 1. let tape = crate::tape_for(store.device())?;
          // 2. let leaves = store.snapshot_resident_leaves(&tape)?;
          // 3. self.layers を forward_resident と同じパターンで辿り、
          //    Linear 層は LinearVars::forward、活性化層は
          //    Module::forward（layer.forward(&tape.0, &current)）を
          //    呼ぶ。最終 Var を .to_tensor() して返す。
          // 確定シグネチャ・エラー型変換（BackendError → AutodiffError。
          // 既存 predict/forward と同じ変換規約）は #935 が実装する。
          unimplemented!()
      }
  }
  ```

  poisoned であれば `store.snapshot_resident_leaves`（内部で
  `with_resident_buffers` を経由する）がクロージャを一切呼ばずに
  `BackendError`（poisoned variant）を返すため、`predict_resident` は
  poisoned 検査を必ず forward 計算より先に行う契約を自動的に満たす
  （`with_resident_buffers` 自体が唯一のバッファアクセス経路であるため、
  この検査を素通りする経路は存在しない）。

**§4.3 との整合**: これにより poisoned 検査の対象は `step`／
`sync_to_host`／`forward_resident`（`register_resident_leaves` 経由）／
`predict_resident`（`snapshot_resident_leaves` 経由）の 4 箇所となる
（§4.3 の最終箇条を本節の追加に合わせて更新する）。`with_resident_
buffers` が返すクロージャの中身自体は `tensor-core` に依存せず
`autodiff::optim` 層に留まり、poisoned 検査そのものは常に
`DeviceParamStore::with_resident_buffers` の呼び出し時点で行う、
という責務分担が本節の確定事項である（`tensor-core` は引き続き
`autodiff::optim::DeviceParamStore` に依存しない設計〈既存の依存方向〉
を維持する）。

### 3.4 クレート責務分担

| クレート | 責務 |
|---|---|
| `tensor-core` | `BackendOps::memory_ops()`（§3.1。`mem` を `self` から導出する唯一の経路。デフォルトメソッド・非破壊拡張）・`sgd_step_device`（§3.1・§3.2。既定実装が `self.memory_ops()` を使う）・`SgdStepConfig` 型定義（§3.1・§3.2） |
| `backend-cpu`／`backend-cuda`／`backend-metal` | 各バックエンドのカーネル実装（CPU: 逐次または `rayon` 並列 in-place・CUDA: NVRTC 1 カーネル・Metal: compute shader 1 個）＋ `impl MemoryOps for XBackendOps`（§3.2）＋ `impl BackendOps for XBackendOps` 側の `memory_ops()` オーバーライド（`Some(self)`。§3.1・§3.2）。実装順序は CPU → CUDA → Metal（#935 引き渡し事項。§6） |
| `autodiff::optim` | `DeviceParamStore`（常駐ストア保持型。§3.3・§4 の所有者。`device: Device` フィールド・`device()` アクセサを含む。§3.3a）・`SgdConfig → SgdStepConfig` 変換・`Tape` の勾配出力を `DeviceParamStore` へ渡す統合ロジック・`DeviceParamStore::with_resident_buffers`（poisoned 検査済みスコープ付きクロージャ。§3.3c。個別バッファへの唯一のアクセス経路）・`DeviceParamStore::register_resident_leaves`（`forward_resident` 専用。`&mut self`・`pending_grad_sources` を書く。§3.3b）・`DeviceParamStore::snapshot_resident_leaves`（`predict_resident` 専用。`&self`・読み取り専用版。§3.3c）。いずれも `tape.ops()`〈`pub(crate)`〉→ `.memory_ops()` の経路で `mem` を内部導出し、`facade` へ `mem`/`ops` を一切渡さない |
| `facade`（`compat::Sequential` 経由で `Sequential::forward_resident`／`Sequential::predict_resident` を実装） | `Sequential` への `forward_resident`・`predict_resident`（§3.3b・§3.3c。いずれも `tape: &Tape` のみを受け取り `register_resident_leaves`／`snapshot_resident_leaves` が返す `Var` を使って `LinearVars::forward`／`Module::forward` を既存 `SequentialVars::forward` と同じパターンで呼ぶ。`mem`／`ops`／`BackendOps`／`MemoryOps` にはいずれも一切触れない）・層インデックス対応表の保持（§3.3a） |

### 3.5 既存 API との関係

`Sequential::apply_parameters`／`trainable_parameters`（ホスト `Tensor<f32>`
契約）は**公開契約として変更しない**（公開 API 非破壊はガードレール条件。
`.claude/rules/security.md` A08）。デバイス常駐経路（`DeviceParamStore` 経由の
学習ループ）は既存経路に追加する新経路であり、既存経路を置き換えない。
`DeviceParamStore` からホスト `Tensor<f32>` への同期（学習終了時・`predict`/
保存前）は明示 API（例: `DeviceParamStore::sync_to_host(&self, tape: &Tape)
-> Result<Vec<Tensor<f32>>, BackendError>`。`DeviceBuffer<f32>` から host
`Tensor<f32>` への download を行うため `mem` を要するが、`new`／
`register_resident_leaves`／`snapshot_resident_leaves`〈§3.3a・§3.3b・
§3.3c〉と同じ理由・同じパターンで `mem: &dyn MemoryOps` を外部引数として
受け取らず、`tape.ops().memory_ops()` から内部導出する。呼び出し元
〈`facade`〉は `predict_resident` と同様 `facade::tape_for(store.device())`
で構築した `Tape` をそのまま渡せばよい。確定シグネチャは #935 で行う）を
介して行い、`apply_parameters` へ渡す形で
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
  バッファをチェックなしに読める抜け穴が残っていた。`forward_resident`
  は個別の poisoned フラグ検査コードを別途持つのではなく、
  `register_resident_leaves`（§3.3b）・`predict_resident` が呼ぶ
  `snapshot_resident_leaves`（§3.3c）のいずれも内部で
  `store.with_resident_buffers`（§3.3c。個別バッファへの唯一の
  アクセス経路）を経由してのみ `weight`/`bias` の `DeviceBuffer` へ
  アクセスする。poisoned であれば `with_resident_buffers` がクロー
  ジャを一切呼ばず `BackendError`（poisoned variant）を返し、
  `register_resident_leaves`／`snapshot_resident_leaves` のいずれも
  `mem.download` に到達しない（`forward_resident`／`predict_resident`
  はこの戻り値のエラーをそのまま伝播するだけでよい）。`DeviceParamStore`
  が個別バッファの public アクセサを一切持たない設計（§3.3c）により、
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

**`forward_resident`／`predict_resident` の forward 計算列は、既存
`SequentialVars::forward`／`Sequential::forward`（`Module::forward` への
多態 dispatch・`Var::matmul`/`Var::add`/`Var::relu`/`Var::sigmoid`/
`Var::tanh`）と同一のコードパスを再利用する設計（§3.3b・§3.3c）である
ため、通常の GEMM/活性化カーネルの数値契約（本節・既存 `gemm`/`relu`
等の判定基準）がそのまま適用され、resident 経路専用の追加 parity
判定は不要である**（旧稿が想定していた `linear_forward_resident` 専用の
parity 確認は、同メソッドの削除〈§3.1〉に伴い対象がなくなったため
削除する）。#936 が確認すべき対象は §5.3 の累積 parity（`sgd_step_
device` の更新式・§5.2 の FMA 契約）に集約される。

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
| (b) `BackendOps` 全面 `DeviceBuffer` 化を先行 | forward/backward を含む全カーネル入口（`gemm`/`add`/`relu` 等すべて）を `DeviceBuffer` 契約へ一気に移行してから更新経路を設計する | **不採用（本イシュースコープでは）**。変更範囲が 4h 粒度を大きく超え、#931（デバイスハンドル再利用の公開 API 設計）・phase-4（#924）と重複する。§3.1 で追加する `sgd_step_device` はこの不採用案とは異なる: 対象は in-place SGD 更新のみ（1 メソッドの追加）であり、forward/backward・入力バッチ・中間活性・他の全カーネル（`gemm`/`add`/`relu`/畳み込み等）は既存のホスト `Tensor<f32>` 契約のまま変更しない（`forward_resident`／`predict_resident` は既存 `Var`/`Module::forward` 経路を再利用するのみで新規カーネル入口を追加しない。§3.3b・§3.3c）。「全カーネル入口の一括移行」という本 (b) 案の対象範囲（forward/backward 全体）とは規模・影響範囲が異なるため、本設計のスコープに含めても 4h 粒度・#931/phase-4 との重複を引き起こさない |
| (c) 採用案 | §2〜§4 の非破壊拡張（`sgd_step_device` デフォルトメソッド・`BackendOps::memory_ops()` による `mem` の自己導出）+ 常駐ストア（`DeviceParamStore`）+ 段階的常駐化（param/velocity のみ常駐、grad は毎ステップ 1 回 upload）+ `Sequential::forward_resident`／`predict_resident` による既存 `Var`/`Module::forward` 経路の再利用（forward 結線・weight/bias の Tape leaf 登録による VJP 成立・resident 推論の parity を構造的に保証。§3.3a・§3.3b・§3.3c）+ 更新フェーズの poisoned 状態契約（§4.3） | **採用** |

## 7. スコープ境界・受け渡し

### #935（実装）への引き渡し事項

- §3.1 の確定シグネチャ（`BackendOps::memory_ops()`〈`self` から `mem`
  を導出するデフォルトメソッド〉・`sgd_step_device`〈`mem` 引数なし。
  内部で `self.memory_ops()` を呼ぶ〉・`SgdStepConfig`。`BackendOps`
  trait 定義への supertrait 追加はしない）をそのまま `tensor-core::
  backend_ops` へ追加する（`buffer.rs:11-13` の保留コメント
  「`MemoryOps` は `BackendOps` の supertrait となる想定」は本設計で
  **不採用と確定**したため、#935 で該当コメントを「supertrait 化は
  crates.io 公開 trait への破壊的変更となるため不採用（§3.1）」という
  趣旨に更新する）。旧稿にあった `linear_forward_resident` は本改訂
  （§3.1）で削除済みのため実装対象に含めない。
- 各 `XBackendOps`（`CpuBackendOps`／`CudaBackendOps`／`MetalBackendOps`）
  へ `impl MemoryOps for XBackendOps` を追加し（既存 `XMemory` 実装への
  委譲か、フィールド保持かは §3.2 のとおり実装詳細として #935 が選ぶ）、
  `impl BackendOps for XBackendOps` の `memory_ops()` オーバーライドを
  `Some(self)` にする（§3.1・§3.2。`DeviceParamStore` が `mem` を得る
  唯一の経路がこのオーバーライドである）。
- 実装順序: CPU → CUDA → Metal（既存 GEMM カーネルの実装順序・PoC 実証の
  蓄積と揃える）。
- フォールバック契約: §3.2 のデフォルト実装（`sgd_step_device`:
  `self.memory_ops()` で `mem` を導出〈`None` なら `Unsupported`〉→
  `mem.download` → ホスト更新 → `mem.upload`）を先に用意し、各バック
  エンドが最適化カーネルでオーバーライドする形で段階的に置き換える
  （デフォルト実装のままでも正しく動作することを前提に、性能改善は
  独立した最適化として進められる）。
- **`DeviceParamStore::device(&self) -> Device`（§3.3a）を実装する**:
  `new` が `tape.ops().device()`（既存の必須メソッド）から取得した値を
  フィールドとして保持し、そのまま返す読み取り専用アクセサ。
- `DeviceParamStore::new(tape: &Tape, params: &[TrainableParam])`（§3.3a。
  `mem` を外部引数として受け取らず `tape.ops().memory_ops()` から内部
  導出する）を `autodiff::optim` へ実装し、`trainable_parameters()` と
  同一の位置対応契約で層インデックス→`DeviceBuffer` 対応表を構築する。
- `Sequential::forward_resident<'t>(&self, tape: &'t Tape, input:
  &Var<'t>, store: &mut DeviceParamStore) -> Result<Var<'t>,
  AutodiffError>`（§3.3a・§3.3b）を `facade::compat::Sequential` へ
  追加する。**`mem`／`ops` のいずれも受け取らない**（§3.3a。codex-review
  P1「'forward_resident' では要求されたバックエンド同一性検査を実装
  できない」の解消）。内部は `store.register_resident_leaves(tape)?`
  （`autodiff::optim::DeviceParamStore` に実装する。§3.3b。内部で
  `tape.ops().memory_ops()` から `mem` を導出し、weight/bias を
  download・`tape.var(..)` で leaf 登録・`NodeId`/`TapeId` を
  `pending_grad_sources` へ書き込む）を 1 回呼び、返る
  `Vec<ResidentLeafVars<'t>>` を使って `self.layers` を辿りながら
  `Linear` 層では `fandhe_ai_autodiff::nn::linear::LinearVars {
  weight, bias }.forward(&current)` を、活性化層では
  `layer.forward(&tape.0, &current)`（`Module::forward`）を呼ぶ
  （既存 `SequentialVars::forward`〈`sequential.rs:315`〉と同一の
  イテレーションパターン。§3.3b）。`pending_grad_sources` が既に
  `Some`（前回 forward 分が未消費）の状態での再呼び出しは
  `register_resident_leaves` 内で型付きエラーとして拒否する
  （§3.3b）。`tape.backward()` 後、呼び出し元は `DeviceParamStore::
  step(&mut self, tape: &Tape, grads: &Gradients, config:
  &SgdStepConfig) -> Result<(), BackendError>`（§3.3b。同じく `mem`
  を受け取らず `tape.ops().memory_ops()` から導出する）を呼び、`step`
  はまず `tape` の `TapeId` が `pending_grad_sources` に保存された
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
- **`DeviceParamStore::register_resident_leaves`／
  `snapshot_resident_leaves`（§3.3b・§3.3c）を実装する**: 前者は
  `&mut self`（`pending_grad_sources` を書く。`forward_resident`
  専用）、後者は `&self`（書き込まない。`predict_resident` 専用）。
  いずれも `with_resident_buffers` を経由して poisoned 検査を行い、
  `tape.ops().memory_ops()` から導出した `mem` で download する内部
  実装を共有する。戻り値 `ResidentLeafVars<'t> { trainable_idx: usize,
  weight: Var<'t>, bias: Option<Var<'t>> }` を新設し `pub` とする。
- **`Sequential::predict_resident(&self, store: &DeviceParamStore,
  input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError>`（§3.3c・
  `facade` クレート）を実装する**: **`ops`／`mem` のいずれも受け取らない**
  （codex-review P1「'predict_resident' が facade に任意の 'BackendOps'
  注入経路を再導入する」の解消）。内部で `facade::tape_for(store.
  device())?` により専用の `Tape` を構築し、`store.snapshot_resident_
  leaves(&tape)?` → `forward_resident` と同じ層イテレーション
  （`LinearVars::forward`／`Module::forward`）→ `.to_tensor()` の順で
  計算する。この設計により resident 推論の出力は既存 `predict` と
  構造的に同一の数値計算列になり（`Module::forward` の既存多態
  dispatch を経由するため）、旧稿が新設していた `ActivationKind`／
  `Module::as_activation_kind`／`BackendOps::sigmoid` はいずれも
  実装しない（§3.3c。Cursor Bugbot 指摘「Sigmoid path breaks predict
  parity」の解消: 別経路を新設せず既存経路を再利用することで parity
  を構造的に保証する）。poisoned 検査の迂回不能性は
  `with_resident_buffers` の型・可視性設計（§3.3c）そのものにより
  担保されるため、`facade`/`autodiff` 側のコードレビュー規律には
  依存しない。
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
