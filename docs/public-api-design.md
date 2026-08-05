# 公開 API シグネチャ設計書（tensor-core・autodiff・backend 入口）

- 対応イシュー: #182（親 #181）
- 対象クレート: `tensor-core`・`autodiff`・バックエンド入口（`backend-cpu`/`backend-cuda`/`backend-metal` が実装する抽象層）
- 位置づけ: 本文書は**設計文書のみ**であり、実行可能なコードは含まない。現状リポは M0 未着手で `crates/` も `Cargo.toml` も存在しないため、Rust シグネチャのコンパイル検証は本イシューでは実施しない。TASK-1.4（自作テンソル型 productize）・TASK-1.5（autodiff productize）・TASK-1.9（バックエンド抽象層実装）の各実装イシューで本文書との突合を行うこと（`docs/spec/05-tasks.md:47,54,82`）。
- 対象外: `compat::array`/`compat::Sequential` 等の互換 API 層（REQ-9）の詳細シグネチャ、guardrail/self-repair の CLI 仕様（兄弟イシュー #183）、演算グラフ・カーネル融合機構の API（イシュー #161・TASK-12.1a）。詳細は本文書末尾「スコープ外」を参照。

## 1. 設計原則

- **完全自作コア**（REQ-1 v2、`docs/spec/04-requirements.md:42`）。Burn 等の既存 ML フレームワークへの統合は行わない。許容依存は 8 区分のみ（`.claude/rules/deps-policy.md`）、禁止リスト（`burn` 系一式・`cubecl`・`candle`・`tch`・`ndarray`）は CI で機械検査する。
- **バックエンド切替は feature フラグなしの cfg ベース**（PoC-v2-5 実証構成、`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）。`cudarc` は無条件依存＋動的ロード、`objc2`/`objc2-foundation`/`objc2-metal` は `cfg(target_os = "macos")` で分離する。
- **型付きエラー**。本番経路で `unwrap()`/`expect()` を使わない（`.claude/rules/coding-rust.md`）。すべての失敗しうる公開 API は `Result<T, E>` を返す。
- **命名は PyTorch/NumPy 慣習に寄せる**（相互運用性の観点）。ただし PyTorch/NumPy 互換の薄いラッパー層（`compat::array`/`compat::Sequential` 相当、REQ-9）自体は本設計のスコープ外であり、本文書が定めるのは自作コアの素の公開 API である。

## 2. tensor-core 公開 API

### 2.1 ストレージ分離設計（PoC-v2-1 からの差分）

PoC-v2-1 は `Tensor<T>` を「行優先連続バッファ + `Vec<T>` 所有」で確定した（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「テンソル型の設計判断」表）。同表は所有権の論点で「専用 View 型（`TensorView<'a, T>`）は後続 PoC で必要になった時点で追加する」と明示的に留保しており、**本イシューはその留保されていた追加判断点**を扱う（PoC-v2-1 の確定事項の書き換えではない）。

zero-copy view（reshape/transpose/narrow）を成立させるため、ストレージを共有バッファへ切り出す:

```rust
/// テンソル本体。`storage` を複数の `Tensor` が共有することで
/// view 系操作（transpose/narrow/reshape の contiguous ケース）を
/// データコピーなしで表現する。
///
/// メモリレイアウト（行優先 + strides）・NumPy 互換ブロードキャスト
/// （stride 0 による同一要素の繰り返し読み）は PoC-v2-1 の確定事項を
/// 維持する（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`）。
pub struct Tensor<T: Element> {
    storage: Arc<Storage<T>>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

/// 実データを保持する共有バッファ。`Tensor` 間で `Arc` 共有される。
/// 単一所有かつ非共有であることが分かっている経路（学習ループの
/// 勾配バッファ等）では、将来的に `Arc::get_mut` による
/// in-place 更新の最適化余地を残す。
struct Storage<T: Element> {
    data: Vec<T>,
}
```

### 2.2 zero-copy view API

```rust
impl<T: Element> Tensor<T> {
    /// 2 軸の strides を入れ替えるのみ。常に zero-copy。
    /// 転置後は非 contiguous になりうる（`is_contiguous()` で判定）。
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor<T>, ShapeError>;

    /// 指定軸の [start, start+len) をスライスする。offset/shape の
    /// 調整のみで常に zero-copy。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor<T>, ShapeError>;

    /// 新しい shape へ再解釈する。contiguous な場合のみ zero-copy。
    /// 非 contiguous な場合の扱いは未決事項（2.2.1 参照）。
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<T>, ShapeError>;

    /// 現在のテンソルが行優先で連続配置されているか判定する。
    pub fn is_contiguous(&self) -> bool;

    /// 非 contiguous な場合に、行優先連続バッファへ実体化した
    /// 新しい `Tensor` を返す（常にコピーを伴う明示 API）。
    /// contiguous な場合は自身の複製（`Arc` 共有のまま）を返す。
    pub fn contiguous(&self) -> Tensor<T>;
}
```

#### 2.2.1 未決事項: 非 contiguous な `reshape` の方針

`reshape` は contiguous 時のみ zero-copy が成立する。非 contiguous なテンソルに対する呼び出しをどう扱うかは 2 案あり、本イシューでは決定しない（ユーザー承認が必要な採否論点）:

- **案 A（エラー）**: `ShapeError` を返し、呼び出し側に明示的な `contiguous().reshape(..)` を要求する。NumPy の `reshape` は暗黙コピーを許容するため慣習からは外れるが、暗黙のパフォーマンス劣化（意図しないフルコピー）を防げる。
- **案 B（暗黙コピー）**: 内部で `contiguous()` を呼んでからコピーを伴う reshape を行う。NumPy 慣習（`ndarray.reshape` は非 contiguous でも動く）に近いが、AI エージェントが意図せず高コストなコピーを埋め込みうる。

安全側の初期実装方針としては案 A（エラー）を推奨する。理由: 本リポの差別化価値は AI 自律メンテナンス（REQ-3）であり、暗黙のコピー発生は性能回帰の原因を不透明にする。最終決定はユーザー承認を経ること。

### 2.3 要素型

```rust
/// テンソルが扱える要素型の最小抽象化（PoC-v2-1 の型パラメータ方針）。
/// `f32`/`f64`/`i32` に加え、GPU バックエンド（CUDA/Metal）で使用する
/// `half::f16` を実装対象とする。
pub trait Element: Copy + Send + Sync + 'static { /* ... */ }
```

### 2.4 生成系

```rust
impl<T: Element> Tensor<T> {
    /// 実行時 shape 検査を行うコンストラクタ（PoC-v2-1 で確定した
    /// 方式）。`data.len()` が `shape` の要素数積と一致しない場合
    /// `ShapeError` を返す。
    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>, ShapeError>;

    pub fn zeros(shape: &[usize]) -> Tensor<T>;
    pub fn ones(shape: &[usize]) -> Tensor<T>;
    pub fn full(shape: &[usize], value: T) -> Tensor<T>;
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Tensor<T>, ShapeError>;
}
```

### 2.5 REQ-10 との関係: rank を型に載せるか

REQ-10 の受け入れ基準は「rank を型に載せるか（`Tensor<T, const R: usize>` 等）は自作 API 設計時に決定し、ドキュメントに記録すること」と要求する（`docs/spec/04-requirements.md:211` 付近、TASK-10.2 対応）。本文書でこれを決定する。

**決定**: 基盤層の `Tensor<T>` は rank を型パラメータに含めない（実行時 rank）。理由は PoC-v2-1 の確定事項（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`「テンソル型の設計判断」表）と同一で、safetensors/ONNX からロードする重みの shape・rank は実行時にしか決まらないため。const generics による型レベル shape 検証は、コンパイル時に shape が既知な層（固定サイズの Linear 層等）に限定した**後続レイヤー**（TASK-10.x）として、基盤 `Tensor<T>` の上に別途構築する。

この方針に伴う限界（REQ-10 が明記を要求する事項）:
- バッチ次元・シーケンス長など実行時に変動する次元は型レベル shape の対象に含めない（可変バッチ推論との衝突を避ける、v1 PoC-7 の教訓を踏襲）。
- 「値が偶然一致するケース」（例: バッチ数と特徴量数がたまたま同じ）は型でも実行時でも検出できない構造的な限界として残る。

## 3. autodiff 公開 API（勾配追跡の型分離）

### 3.1 型分離方式

`tensor-core::Tensor<T>` は**常に非追跡**とする。勾配追跡は `autodiff` クレートの別型 `Var` で表し、`Tape` 上の `NodeId` を保持する。`Tensor<T>` に対する演算はテープを一切構築せず、`Var` に対する演算のみがテープへ記録される。これにより「ON/OFF の型分離」がコンパイル時に保証される。

採用方式は PoC-v2-2 が確定した**動的テープ式**（Wengert list）である（`docs/spec/03-poc/poc-v2-2-autodiff/README.md:170`「採用方式の確定」）。

```rust
/// 演算を記録する Wengert list。`Var` 上の演算のみがここに記録される。
/// ノード列は `RefCell` で包み、`Var` からの内部可変性による追記を
/// 可能にする（借用モデルの詳細は `Var` のドキュメント参照）。
pub struct Tape { nodes: RefCell<Vec<TapeNode>> }

impl Tape {
    pub fn new() -> Tape;

    /// 非追跡の `Tensor<f32>` を、テープ上の葉ノードとして登録する。
    pub fn var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

/// テープ上の 1 ノードを指す追跡対象値。値そのものではなく
/// `NodeId` + テープへの参照を保持し、演算のたびにテープへ新しい
/// ノードを追加する。
///
/// **借用モデル**: PoC-v2-2 確定 API（`Tape::matmul(&mut self, ...)`、
/// `docs/spec/03-poc/poc-v2-2-autodiff/README.md:28-31`）は `&mut Tape`
/// を要求するが、本設計では `Var` 単体を式中で連鎖させたい
/// （`a.matmul(&b).relu()` 等）ため、`Tape` 内部を `RefCell` で
/// 包み、`Var` は共有参照 `&'t Tape` を保持する形へ変更する。
/// ノード追加は `RefCell::borrow_mut()` 経由の内部可変性で行う。
/// PoC-v2-2 が確定したのは「動的テープ式（Wengert list）」という
/// 記録方式そのものであり、呼び出し規約（`&mut` か内部可変性か）は
/// 本文書側で productize 時の API 使い勝手を優先し決定する
/// （TASK-1.5 実装時に借用チェッカ上の問題が出た場合は PoC-v2-2 の
/// `&mut Tape` 方式へ戻す判断もありうる。決定は実装時に確定する）。
pub struct Var<'t> { tape: &'t Tape, id: NodeId }

impl<'t> Var<'t> {
    /// 追跡を外し、現在の値を非追跡の `Tensor<f32>` として取り出す。
    /// テープ内部が `RefCell` のため `Ref<'_, Tensor<f32>>` を返す。
    pub fn value(&self) -> Ref<'_, Tensor<f32>>;

    // 演算セット（3.2 参照）。各メソッドは新しい `Var` を返す。
    pub fn matmul(&self, other: &Var<'t>) -> Var<'t>;
    pub fn add(&self, other: &Var<'t>) -> Var<'t>;        // bias broadcast 含む
    pub fn mul(&self, other: &Var<'t>) -> Var<'t>;
    pub fn relu(&self) -> Var<'t>;
    pub fn exp(&self) -> Var<'t>;
    pub fn tanh(&self) -> Var<'t>;
    pub fn sum(&self, dim: Option<usize>) -> Var<'t>;
    pub fn max(&self, dim: Option<usize>) -> Var<'t>;
    pub fn mse_loss(&self, target: &Var<'t>) -> Var<'t>;
}

impl Tape {
    /// `loss` から逆伝播し、テープ上の全 `Var` に対する勾配を計算する。
    pub fn backward(&self, loss: &Var<'_>) -> Result<Gradients, AutodiffError>;
}

/// `backward()` の結果。`NodeId` をキーに勾配テンソルを保持する。
pub struct Gradients { /* ... */ }

impl Gradients {
    pub fn get(&self, var: &Var) -> Option<&Tensor<f32>>;
}
```

`no_grad` 相当（勾配追跡を一時的に止める）は、専用フラグ API を設けず「`Tensor<T>` のまま演算する」ことで表現する。`Tensor<T>` と `Var` は別型であるため、追跡なしの経路を選ぶことはコンパイル時に強制される。

### 3.2 演算セット（初期）

PoC-v2-2 と PoC-v2-5 で実測済みの演算に合わせる（`docs/spec/03-poc/poc-v2-2-autodiff/README.md:28`、`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md:49` 付近の `MetalOps` 公開 API 一覧）:

`matmul`・`add`（bias broadcast 含む。逆伝播は reduction）・`mul`・`relu`・`exp`・`tanh`・`sum`・`max`・`mse_loss`。

演算セットの拡張（Conv 系・Softmax 等）は後続タスク（TASK-1.5 以降）へ委譲する。

## 4. backend 入口公開 API

### 4.1 デバイス選択

```rust
/// 実行先デバイス。`Metal` は `cfg(target_os = "macos")` でのみ
/// 存在する（PoC-v2-5 実証構成、feature フラグは設けない）。
/// `cudarc` は無条件依存 + 動的ロードのため `Cuda` variant は
/// 全 OS で存在するが、実行時に CUDA ドライバ不在なら
/// `BackendError` を返す（4.4 参照）。
pub enum Device {
    Cpu,
    Cuda(usize),
    #[cfg(target_os = "macos")]
    Metal,
}

impl Device {
    /// 実行時に利用可能なデバイスを検出して返す。
    pub fn available() -> Vec<Device>;
}
```

**既定選択の方針（未決事項）**: CUDA を既定で有効化するかどうかの具体的な構成決定は REQ-2 でも未検証のまま残っている（`docs/spec/04-requirements.md` REQ-2 受け入れ基準「バックエンド有効化構成」の項）。本文書では既定デバイス選択ロジックを確定しない。TASK-1.9 実装時にユーザー承認を得て決定すること。

### 4.2 カーネル入口トレイト案

PoC-v2-5 の `MetalOps` 公開 API（`gemm`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/`max`、`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）と対称な safe API を、バックエンド抽象層のトレイトとして定義する。`unsafe`/FFI は各バックエンド実装内部（`cudarc`・`objc2` 系呼び出し境界）に閉じ込める。

**デバイス常駐バッファ**: `Tensor<T>` の `Storage` はホスト（`Vec<T>`）に固定されている（2.1）。`BackendOps` の各メソッドが毎回 `&Tensor<f32>`（ホスト常駐）を受け取り `Tensor<f32>`（ホスト常駐）を返す素朴な形にすると、CUDA/Metal 経路では演算ごとにホスト⇔デバイス転送が発生し、複数演算を連鎖する GPU ワークロード（REQ-2・REQ-11 が前提とする行列演算ユニット活用）で転送コストが支配的になる。これを避けるため、デバイス常駐バッファを表す型を導入し、転送は明示 API（`upload`/`download`）に閉じ込める:

```rust
/// デバイス上に確保されたバッファへの不透明ハンドル。中身は各
/// バックエンド実装（`backend-cuda`/`backend-metal`）が保持し、
/// `tensor-core`/backend 入口からは shape・dtype・所属 `Device` の
/// メタデータのみが見える。ホスト `Tensor<T>` とは明確に別型とする
/// ことで、「今どちらに実データがあるか」を型で追跡できるようにする。
pub struct DeviceBuffer<T: Element> {
    device: Device,
    shape: Vec<usize>,
    _marker: PhantomData<T>,
    // 実体（CUDA デバイスポインタ／Metal MTLBuffer 等）は
    // 各バックエンド実装内部が保持し、ここには公開しない。
}

/// 各バックエンド（CPU/CUDA/Metal）が実装するカーネル入口。
/// 公開 API はすべて safe。`unsafe` は実装側の FFI 境界に限定する
/// （PoC-v2-4/v2-5 の設計方針を踏襲）。演算は `DeviceBuffer` 同士で
/// 完結し、ホスト往復は `upload`/`download` を呼んだ箇所にのみ
/// 発生する。
pub trait BackendOps {
    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError>;

    fn gemm(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;

    // elementwise
    fn add(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn mul(&self, a: &DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn relu(&self, a: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn exp(&self, a: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn tanh(&self, a: &DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>, BackendError>;

    // reduction
    fn sum(&self, a: &DeviceBuffer<f32>, dim: Option<usize>) -> Result<DeviceBuffer<f32>, BackendError>;
    fn max(&self, a: &DeviceBuffer<f32>, dim: Option<usize>) -> Result<DeviceBuffer<f32>, BackendError>;
}
```

`BackendError::DeviceMismatch` は、`gemm`/`add` 等に異なる `Device` に属する `DeviceBuffer` を渡した場合に返す（`DeviceBuffer::device` フィールドで検査可能）。この設計は PoC-v2-5 の `MetalOps` 実測 API（`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）をそのまま転記したものではなく、本イシューで追加した拡張である旨を明記する。`DeviceBuffer` 内部表現（CUDA デバイスポインタのラップ方法・Metal `MTLBuffer` の寿命管理等）は TASK-1.9 実装時に確定する。

### 4.3 API 契約として明記する数値仕様

- **バックエンド間数値一致**: 全ペア共通で「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の複合判定（`docs/spec/04-requirements.md` REQ-2、PoC-v2-5 の事前固定判定基準と同一）。
- **FMA 契約統一**: CPU 参照実装は `f32::mul_add` を用い、GPU 側（CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）の既定 FMA 契約と揃える。PoC-v2-5 は K=4096 の GEMM ストレスケースで、CPU 側が `acc += a * b`（乗算・加算を別々に丸め）のままだと 262,144 セル中 7 セルが複合判定を外れ、`acc = a.mul_add(b, acc)` に差し替えると完全一致（fail_cells=0/262144）することを実測確認済み（`docs/spec/04-requirements.md` REQ-2、`.claude/rules/coding-rust.md`）。
- **Metal precise math の明示**: `MTLCompileOptions.mathFloatingPointFunctions` を `Precise` に設定し、シェーダ側でも `metal::precise::exp`/`metal::precise::tanh` 等を明示使用する（`mathMode=Safe` のみでは transcendental 関数が `metal::fast` 経路にディスパッチされ複合判定の余裕が薄くなることが PoC-v2-5 で判明。`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md:49` 付近）。
- **カーネル境界検査の維持（REQ-8）**: 性能下限・最適化の達成を理由にシェーダ・カーネル側の手動境界チェックを省略しない。境界検査を無効化する最適化を適用する場合は手動境界チェックを維持したまま行う（`.claude/rules/coding-rust.md`）。本契約は `BackendOps` の全実装（CPU intrinsics・CUDA NVRTC/mma・Metal simdgroup）に適用する。

### 4.4 実行時エラー

```rust
/// バックエンド抽象層のエラー型。CUDA ドライバ不在（`cudarc` の
/// 動的ロード失敗）は「CUDA toolkit 非搭載環境でもビルド成立する」
/// という REQ-1 の契約の実行時側の受け皿として、型付きエラーで返す。
#[derive(Debug)]
pub enum BackendError {
    /// `cudarc` の動的ロードに失敗した（CUDA ドライバ・toolkit 不在等）。
    CudaUnavailable(String),
    /// 入力テンソルの shape が演算の要求と合わない。
    ShapeMismatch(ShapeError),
    /// デバイス間でテンソルが混在している等の不整合。
    DeviceMismatch,
}
```

## 5. エラー型一覧・横断事項

| 型 | 役割 | 発生源 |
|----|------|--------|
| `ShapeError` | テンソル生成・view 操作・reshape の shape 不整合 | `tensor-core` |
| `AutodiffError` | `Tape::backward` の失敗（グラフ不整合等） | `autodiff` |
| `BackendError` | バックエンド実行時エラー（CUDA 不在・shape 不一致・デバイス不整合） | backend 入口 |

外部フォーマット（safetensors/ONNX）読み込み時は、長さ・形状の検証を先行させる契約とする（`.claude/rules/security.md` A03 インジェクション対策）。`onnx-interop` クレートの `ShapeError` 送出経路は本設計のスコープ外（REQ-7 系、別イシューで扱う）だが、`tensor-core::Tensor::new`/`from_slice` の実行時 shape 検査がこの検証の受け皿となる設計である点を接続点として明記する。

## 6. 未決事項・採否論点の一覧（ユーザー判断用）

1. **非 contiguous な `reshape` の方針**（2.2.1）: エラー（案 A、本文書の推奨）か暗黙コピー（案 B）か。
2. **CUDA 既定有効化の構成決定**（4.1）: REQ-2 でも未検証のまま残っている。`Device::available()` が返す既定デバイスの選択ロジックは本文書では確定しない。
3. **rank 型載せの最終確定**（2.5）: 本文書は基盤層を実行時 rank、型レベル shape を後続レイヤー限定とする方針を決定として記録した。TASK-10.x 実装時にこの方針で問題がないか再確認すること。
4. **演算グラフ／融合機構（イシュー #161）との接続点**: `Var`/`Tape` の演算記録が将来の融合機構（TASK-12.1a）とどう接続するかは本文書では設計しない。`Tape` の `Op` 列挙が融合対象の中間表現候補になりうる点のみ接続点として記録する。
5. **`DeviceBuffer` の内部表現**（4.2）: CUDA デバイスポインタ・Metal `MTLBuffer` の具体的なラップ方法・寿命管理（`Drop` での解放タイミング等）は本文書では確定しない。TASK-1.9 実装時に決定する。
6. **`Var`/`Tape` の借用モデル**（3.1）: 本文書は productize 時の API 使い勝手を優先し `RefCell` + 共有参照方式を採用したが、PoC-v2-2 確定 API は `&mut Tape` 方式だった。TASK-1.5 実装時に借用チェッカ上の問題が生じた場合は `&mut Tape` 方式へ戻す判断もありうる。

## スコープ外（out-of-scope-tracking 対象）

- **compat API 層**（`compat::array`/`compat::Sequential` の詳細シグネチャ）: REQ-9 系の後続タスク。本文書は境界（自作コアの素の API とは別レイヤーであること）のみを 1 章で明記した。
- **guardrail/self-repair の CLI 仕様**: 兄弟イシュー #183 が担当。
- **演算グラフ・カーネル融合機構の API**: イシュー #161（TASK-12.1a）。接続点のみ 6.4 に記載。
- **シグネチャのコンパイル検証・実装**: TASK-1.4（自作テンソル型 productize）以降の実装イシューへ引き継ぐ。M0 未着手のため `crates/`・`Cargo.toml` が存在せず、本イシューでは検証しない。
