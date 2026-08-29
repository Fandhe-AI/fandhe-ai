//! カーネルディスパッチ機構（TASK-1.9c・#46）。
//!
//! 単一の計算記述（[`BackendOps`] を受け取る関数）から CPU／CUDA／Metal
//! いずれのバックエンドのカーネルへも呼び分けられるようにする入口。
//! `device`（TASK-1.9a・#44）と同じ依存逆転構成を踏襲する: trait 定義を
//! 3 バックエンドクレートが依存できる本クレートに置き、各バックエンド
//! クレート（`backend-cpu`／`backend-cuda`／`backend-metal`）側で実装する
//! （`tensor-core` → `backend-*` の逆依存は作らない）。
//!
//! シグネチャは `docs/public-api-design.md` §4.2 の `BackendOps` trait案を
//! 正本としつつ、以下の点で拡張・簡略化している（同文書「TASK-1.9 実装
//! イシューで本文書との突合を行うこと」に対応。突合結果は同文書にも
//! 注記する）:
//!
//! - **`DeviceBuffer`／`upload`／`download` を含めない**。§4.2 が示す
//!   デバイス常駐バッファ型・転送 API は TASK-1.9b（#45）の担当であり、
//!   本イシュー時点で `tensor-core`・3 バックエンドクレートいずれにも
//!   存在しない（実装開始時に `git fetch origin main` で確認済み）。
//!   本イシューの受け入れ条件は「同一コードで 3 バックエンドのカーネルが
//!   呼び分けられる」（機構的な呼び分け）であり、既存カーネル入口
//!   （CPU `gemm_blis_parallel`・CUDA `CudaGemm::run_tiled_f32`・Metal
//!   `MetalGemm::dispatch_auto`）がいずれもホスト常駐 `&[f32]` を受け取り
//!   内部で H2D／D2H 転送を完結させる契約であるため、`DeviceBuffer` なしで
//!   本受け入れ条件を満たせる。§4.2 の `DeviceBuffer` 版シグネチャへの
//!   移行（`upload`／`download` の追加）は #45 のマージ後、`BackendOps` の
//!   非破壊拡張（デフォルトメソッド追加等）として TASK-1.9d（#47）以降で
//!   検討する
//! - 各メソッドはホスト常駐 [`Tensor<f32>`](crate::Tensor) を受け取り
//!   [`Tensor<f32>`](crate::Tensor) を返す（§4.2 の `DeviceBuffer<f32>` を
//!   `Tensor<f32>` に読み替えた形）。CPU 実装は転送コストが発生しないため
//!   このままで問題なく、CUDA／Metal 実装は各メソッド内で
//!   `Tensor::as_slice` → カーネル呼び出し（内部で H2D／D2H）→
//!   `Tensor::new` で完結させる
//! - 未実装カーネル（CUDA／Metal の elementwise・reduction。TASK-1.9c 時点
//!   では両バックエンドとも GEMM カーネルのみ実装済み）は
//!   [`crate::device::BackendError::Unsupported`]（本イシューで追加した
//!   非破壊拡張 variant）を返す fail-safe 実装とする。GPU 側
//!   elementwise・reduction カーネルの実装自体は本イシューのスコープ外
//!   （out-of-scope-tracking.md 対象。引き継ぎ先はユーザー承認を得て別
//!   Issue で追跡する）
//!
//! ディスパッチ規則（形状・HW 判定による経路選択）は TASK-11.2b（#68）の
//! 担当でありスコープ外（`docs/dispatch-rules-design.md`。TASK-11.2a・
//! #67）。既定デバイス選択ロジック（CUDA 既定有効化の構成決定含む）も
//! ユーザー承認必須のためスコープ外（`device` モジュールと同方針）。
//! 3 バックエンド横断の統合テストは TASK-1.9d（#47）が本格的に担当し、
//! 本イシューは受け入れ条件検証に必要な最小限のテストに留める。

use crate::Tensor;
use crate::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use crate::device::{BackendError, Device};
use crate::dispatch_failure::DispatchFailureCell;
use crate::fusion::FusionPlan;

/// [`BackendOps::sgd_step_device`] の 1 ステップ分のハイパーパラメータ
/// （イシュー #935・`docs/device-resident-update-design.md` §3.1）。
///
/// `fandhe_ai_autodiff::optim::sgd::SgdConfig`（ホスト参照実装。`lr`／
/// `momentum`／`dampening`／`weight_decay`／`nesterov` の 5 フィールド）と
/// 同じ意味論のフィールドに `is_first_step` を加えたもの。`autodiff`
/// クレートは `tensor-core` へ依存する側（`tensor-core` → `autodiff` の
/// 逆依存は作らない）であるため、`SgdConfig` をここへ再エクスポートせず
/// 独立した型として定義する（`fandhe_ai_autodiff::optim::device_store::
/// DeviceParamStore::step` が `SgdConfig` から本型へ変換して渡す）。
///
/// `is_first_step`: PyTorch `torch.optim.SGD` の momentum 初期化規則
/// （`docs/spec` 由来。`fandhe_ai_autodiff::optim::sgd` モジュールコメント
/// 「Algorithm」節）は「初回 step は `b ← g`、2 回目以降は
/// `b ← μ·b + (1−τ)·g`」であり、この分岐はパラメータの値そのものではなく
/// 呼び出し元（`DeviceParamStore`）が保持するステップカウンタに依存する。
/// `SgdConfig` 自体は構築後不変（`fandhe_ai_autodiff::optim::sgd::SgdConfig`
///参照）だが、`is_first_step` はステップごとに変化するため `SgdConfig` の
/// フィールドではなく本型（呼び出しごとに構築する値）のフィールドとする。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdStepConfig {
    /// 学習率。
    pub lr: f32,
    /// momentum 係数 `μ`。`0.0` は momentum 無効（`velocity` 引数は
    /// 無視してよい）。
    pub momentum: f32,
    /// dampening `τ`。
    pub dampening: f32,
    /// weight decay `λ`（L2 正則化。`torch.optim.SGD` と同じく `p` に
    /// 係数を乗じて勾配へ加算する）。
    pub weight_decay: f32,
    /// nesterov momentum を使うか。
    pub nesterov: bool,
    /// このパラメータ列にとって最初の `step()` 呼び出しか（momentum
    /// バッファの初期化分岐。上記フィールドドキュメント参照）。
    pub is_first_step: bool,
}

/// GEMM epilogue で適用する activation 種別（TASK-12.1f・#203）。
///
/// [`BackendOps::gemm_bias_act`] の第 4 引数として渡す。CUTLASS 系実測
/// （epilogue 融合で平均 1.38〜1.45 倍。イシュー #203）が動機の
/// Linear+bias+ReLU 相当パターンを表現できれば TASK-12.1f の受け入れ
/// 条件を満たせるため、まず `Relu` のみを持つ。`#[non_exhaustive]` は
/// 公開 API 非破壊（ガードレール条件・`.claude/rules/security.md`）を
/// 保ちながら将来 `Gelu`／`Sigmoid` 等を追加できるようにするため
/// （呼び出し側の網羅的 match を破壊しない。`GemmError`・`ParityError`
/// と同方針）。
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// activation なし（bias 加算のみ、または恒等関数）。
    None,
    /// `max(x, 0)`。`BackendOps::relu` と同一の定義を epilogue 内で適用する。
    Relu,
}

/// 各バックエンド（CPU／CUDA／Metal）が実装するカーネル入口
/// （`docs/public-api-design.md` §4.2。差分はモジュール冒頭コメント参照）。
///
/// object-safe に設計している（`&dyn BackendOps` として扱える。
/// [`ops_for`] が複数バックエンドを横断して選択する際に使用する）。
/// v1 は PoC-v2-5 実測 API（`MetalOps`）のスコープに合わせて `f32` 固定
/// とする（f16 経路のジェネリック化は §4.2 6-8 のとおり保留）。
///
/// 公開 API はすべて safe。`unsafe` は各バックエンド実装内部の FFI 境界
/// （`cudarc`・`objc2` 系呼び出し）に閉じ込める
/// （`.claude/rules/coding-rust.md`）。
pub trait BackendOps {
    /// このインスタンスが対応する [`Device`]（呼び出し元がログ・
    /// エラーメッセージで識別するために使う）。
    fn device(&self) -> Device;

    /// このバックエンドの [`MemoryOps`]（確保・アップロード・ダウンロード）
    /// 実装への参照（イシュー #935・`docs/device-resident-update-design.md`
    /// §3.1）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は `None`（`MemoryOps` を持たない）。`BackendOps` を
    /// `MemoryOps` の supertrait にする案（`buffer.rs` モジュール冒頭
    /// コメント旧稿）は crates.io 公開済み trait への破壊的変更となる
    /// ため不採用と確定した（設計文書 §3.1）。本デフォルトメソッド追加は
    /// `gemm_bias_act`／`run_fused` と同じ非破壊拡張パターン（`BackendOps`
    /// を実装する外部クレートは何もしなくても既存実装のままコンパイル
    /// が通る）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::new` が
    /// `tape.ops().memory_ops()` を呼び、`None` の場合は
    /// [`BackendError::Unsupported`] としてデバイス常駐パラメータ更新を
    /// 拒否する（fail-closed。「`memory_ops()` を呼ぶフォールバック合成は
    /// 設けない」という設計文書 §3.2 改訂の確定事項に従い、`Some` を返す
    /// バックエンドのみがこの経路をサポートする）。CPU／CUDA／Metal の 3
    /// バックエンドはいずれも本デフォルトを `Some(self)` へオーバーライド
    /// する（各バックエンドクレートの `ops.rs` 参照）。
    fn memory_ops(&self) -> Option<&dyn MemoryOps> {
        None
    }

    /// SGD の 1 パラメータ分の更新をデバイス上で in-place に実行する
    /// （イシュー #935・`docs/device-resident-update-design.md` §3.2）。
    ///
    /// `param`／`grad`／`velocity`（momentum 有効時のみ）はいずれも
    /// このバックエンド自身が確保した [`DeviceBuffer<f32>`]
    /// （[`MemoryOps::alloc_zeroed`]／[`MemoryOps::upload`] の戻り値）を
    /// 要求する契約。呼び出し元（`fandhe_ai_autodiff::optim::device_store::
    /// DeviceParamStore::step`）は毎ステップ `grad` のみをアップロードし
    /// `param`／`velocity` は前ステップから使い回すことで、param の
    /// ホスト再アップロードを排除する（本イシューの受け入れ条件）。
    ///
    /// **呼び出し元は全パラメータを連結した単一バッファで 1 回だけ呼ぶ**
    /// （イシュー #1023「パラメータ横断の単一連結バッファ化」）。
    /// `param`／`grad`／`velocity` はいずれもパラメータ数だけ個別に渡す
    /// のではなく、`DeviceParamStore` が全パラメータを 1 本の shape
    /// `[total_numel]` バッファへ連結して常駐させ、`step()` ごとに本
    /// メソッド（`sgd_step_device_tracked` 経由）を 1 回だけ起動する。
    /// 本メソッド自体は要素単位で shape 非依存に定義されているため、
    /// この呼び出し規約変更はシグネチャ・カーネル実装（CPU／CUDA／
    /// Metal のいずれも）に一切変更を要求しない。
    ///
    /// 更新式は `fandhe_ai_autodiff::optim::sgd`（`Sgd::step` ホスト参照
    /// 実装）と同一の項順序（weight_decay → momentum〈`is_first_step` で
    /// `b ← g` 分岐〉→ nesterov → 減算）を 3 バックエンドで揃える契約
    /// （設計文書 §5.2）。カーネル境界検査は省略しない（REQ-8・
    /// `.claude/rules/coding-rust.md`）。
    ///
    /// # デフォルト実装（非破壊拡張）
    /// 既定は常に [`BackendError::Unsupported`] を返す fail-closed
    /// （`memory_ops()` を呼ぶフォールバック合成は設けない。設計文書
    /// §3.2 改訂）。CPU／CUDA／Metal はこのデフォルトを実カーネルで
    /// オーバーライドする。
    ///
    /// # エラー
    /// - `param`／`grad`／`velocity` のいずれかがこのバックエンドの
    ///   ハンドル型へダウンキャストできない・デバイスが一致しない →
    ///   [`BackendError::DeviceMismatch`]
    /// - shape が一致しない → [`BackendError::ShapeMismatch`]
    /// - `config.momentum != 0.0` なのに `velocity` が `None` →
    ///   [`BackendError::Unsupported`]
    fn sgd_step_device(
        &self,
        _param: &mut DeviceBuffer<f32>,
        _grad: &DeviceBuffer<f32>,
        _velocity: Option<&mut DeviceBuffer<f32>>,
        _config: &SgdStepConfig,
    ) -> Result<(), BackendError> {
        Err(BackendError::Unsupported(
            "sgd_step_device: default fail-safe (no in-place SGD kernel available)".into(),
        ))
    }

    /// [`BackendOps::sgd_step_device`] と同型だが、Metal のコマンド
    /// バッファ共有（イシュー #1017・`docs/backend-metal-command-
    /// batching-design.md`）向けに共有失敗トークン
    /// [`DispatchFailureCell`] を追加引数として受け取る非破壊拡張
    /// （`gemm_bias_act`／`run_fused` と同じ「デフォルトメソッド追加」
    /// パターン。`BackendOps` の SemVer 非破壊拡張）。
    ///
    /// # デフォルト実装
    /// 既定は `token` を無視して [`BackendOps::sgd_step_device`] へ
    /// そのまま委譲する。CPU は dispatch ごとに同期実行するため実行時
    /// エラーが呼び出し元に即座に返り、遅延失敗トークンを必要としない
    /// （このデフォルトのままでよい）。CUDA はイシュー #1013
    /// （`docs/backend-cuda-async-execution-design.md` §5）でカーネル
    /// 起動直後の都度 `synchronize()` を除去し非同期実行契約へ移行した
    /// が、本 `token`（`DispatchFailureCell`）は使わずオーバーライドも
    /// しない（このデフォルトのまま）。`backend-cuda::context_cache` は
    /// ordinal 単位の poison 状態機械（`begin_driver_call`／
    /// `observe_driver_result`／`observe_cuda_result`／`is_poisoned`。
    /// 単一ストリームの FIFO 順序保証を前提に sticky エラー観測時点で
    /// ordinal を poison する設計）を備え、PR #1064（イシュー #1013 の
    /// codex-review P0 指摘への対応）で `backend-cuda::ops`／
    /// `backend-cuda::memory` の `BackendOps`／`MemoryOps` 実装境界
    /// （`with_driver_call` ヘルパー）へ結線済みである
    /// （`docs/backend-cuda-async-execution-design.md` §12）。
    /// これにより、`sgd_step_device` 自身のカーネル起動が sticky な
    /// 実行時エラーを引き起こした場合、その ordinal 上で最初に
    /// `observe_cuda_result` が観測した時点（同一ステップの起動自体・
    /// 別の同一ステップ内 driver 呼び出し・または別テンソル演算の
    /// いずれか）で poison 化され、以降の `sgd_step_device` 呼び出しは
    /// `begin_driver_call` の拒否により `Err` を返す。`DeviceParamStore::
    /// step` は `sgd_step_device_tracked` が返す `Err` を常に
    /// `poisoned.store(true, ..)` へ変換する（`device_store.rs`
    /// `step` 実装参照）ため、この `Err` は必ず `StorePoisoned` への
    /// 自己遷移につながる。ただし検出は「次に同一 ordinal 上で
    /// driver 呼び出しが起きた時点」に限られ、poison からの**回復**
    /// （`context_cache::invalidate_with` の呼び出し）は #1062 へ
    /// 引き継いだままである。Metal のみ
    /// `backend-metal::ops::MetalBackendOps` がオーバーライドし、
    /// `MetalContext::encode` と**同一ロック区間で** `token` をバッチへ
    /// 登録する（encode と登録の間に別スレッドの `synchronize` が
    /// 割り込む競合を防ぐ。設計文書 §3.7 (2)）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::step`
    /// が呼び出し元となり、自身が保持する `failure_token` を渡す
    /// （4 つの状態機械エントリ全てが `token.is_set()` を検査して
    /// 自己 poison する。`device_store.rs` モジュール冒頭コメント参照）。
    fn sgd_step_device_tracked(
        &self,
        param: &mut DeviceBuffer<f32>,
        grad: &DeviceBuffer<f32>,
        velocity: Option<&mut DeviceBuffer<f32>>,
        config: &SgdStepConfig,
        _token: &DispatchFailureCell,
    ) -> Result<(), BackendError> {
        self.sgd_step_device(param, grad, velocity, config)
    }

    /// 行列積 `C = A @ B` を計算する（`A: [m, k]`・`B: [k, n]` の 2 次元
    /// テンソルのみ受け付ける。shape 不整合は
    /// [`BackendError::ShapeMismatch`]）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    // elementwise（`docs/public-api-design.md` §4.2 と同じ 5 演算）
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    // reduction（`docs/public-api-design.md` §4.2 と同じ 2 演算）
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;

    /// GEMM の epilogue（bias 加算・activation）を融合した
    /// `act(A @ B + bias)` を計算する（TASK-12.1f・#203）。
    ///
    /// `bias` は `[n]`（`B` の列数）の 1 次元テンソルで、`A @ B: [m, n]` の
    /// 各行へブロードキャスト加算される（`None` の場合は bias 加算を
    /// 省略する）。`act` は bias 加算後に適用する
    /// （[`Activation::None`] なら恒等関数）。
    ///
    /// # デフォルト実装（非融合合成）
    ///
    /// 本メソッドは **デフォルトメソッド**として追加している（`BackendOps`
    /// の非破壊拡張。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）。デフォルト実装は `gemm` →
    /// （`bias` があれば）`add`（行方向ブロードキャスト。
    /// `docs/public-api-design.md` §4.2 のブロードキャスト規約に従い
    /// `[n]` を `[1, n]` として `[m, n]` へ揃える）→ `act` に応じた
    /// activation メソッド呼び出しの 3 段合成である。CPU バックエンドは
    /// [`crate`] を利用する `backend-cpu::ops::CpuBackendOps` がこの
    /// デフォルトを **カーネル内融合実装でオーバーライド**し、中間
    /// `Tensor` 2 個の割当・GEMM 結果の再読み出しパスを削減する
    /// （CUTLASS 系実測で epilogue 融合が平均 1.38〜1.45 倍。動機は
    /// イシュー #203）。CUDA はイシュー #599 で
    /// `backend-cuda::ops::CudaBackendOps::gemm_bias_act` が本デフォルトを
    /// **カーネル内融合実装でオーバーライド**した（CPU と同じ「bias が
    /// `None` または `[n]` 厳密一致なら融合、それ以外は非融合合成へ
    /// フォールバック」という分岐条件。`backend-cuda::ops::
    /// gemm_bias_act_route` 参照）。Metal はイシュー #605 で
    /// `backend-metal::ops::MetalBackendOps::gemm_bias_act` が本デフォルトを
    /// **カーネル内融合実装でオーバーライド**した（CPU／CUDA と同じ「bias
    /// が `None` または `[n]` 厳密一致なら融合、それ以外は非融合合成へ
    /// フォールバック」という分岐条件。`backend-metal::ops::
    /// gemm_bias_act_route` 参照）。CPU／CUDA／Metal の 3 バックエンドが
    /// すべて融合カーネルでオーバーライド済みとなった。
    ///
    /// `bias` の shape が `[n]` の場合（CPU バックエンドでは融合カーネルの
    /// 対応範囲）はそのまま計算する。`[n]` でない場合は `add` の NumPy
    /// 互換ブロードキャスト判定へ委譲し、`out: [m, n]` へブロードキャスト
    /// **不能**な場合にのみ [`BackendError::ShapeMismatch`] を返す
    /// （`[1]`・`[1, n]`・`[m, n]` 等ブロードキャスト可能な shape は
    /// 成功する。CPU／CUDA／Metal で同一の意味論。#203 Review 指摘）。
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        let mut out = self.gemm(a, b)?;
        if let Some(bias) = bias {
            out = self.add(&out, bias)?;
        }
        out = match act {
            Activation::None => out,
            Activation::Relu => self.relu(&out)?,
        };
        Ok(out)
    }

    /// デバイス常駐 `w`（・`bias`）のまま `y = a @ w (+ bias)` を計算する
    /// （イシュー #1022・#1023「R3」・`docs/device-resident-update-design.md`
    /// §3.3e）。
    ///
    /// `fandhe_ai_autodiff::optim::device_store::DeviceParamStore::linear_forward`
    /// が学習ループの forward で使う。`a`（ホスト常駐）は毎ステップ変化
    /// する活性化値、`w`（デバイス常駐）は学習対象パラメータであり、
    /// `sgd_step_device` と同じく **本メソッドが `w`／`bias` を
    /// ホストへ download しない**ことが受け入れ条件の中核（本イシューが
    /// 排除する対象は「forward のたびにパラメータをホストへ落とす」
    /// D2H であり、`a`・戻り値の D2H は含まない。`docs/device-resident-
    /// update-design.md` §1.2 の解釈）。
    ///
    /// `w`／`bias` は [`DeviceBufferView`]（イシュー #1023「パラメータ
    /// 横断の単一連結バッファ化」後、`DeviceParamStore` が全パラメータを
    /// 1 本の連結 `DeviceBuffer<f32>` として保持するため、個々の
    /// パラメータは連結バッファ内の要素オフセット範囲としてしか
    /// 表現できない。「R3: 要素オフセット付き常駐ビュー」設計。
    /// `docs/device-resident-update-design.md` 追補参照）で渡す。実装は
    /// `view.offset()..view.offset() + view.numel()` の範囲のみを
    /// `view.shape()` の重みとして扱う契約（この範囲チェック自体は
    /// [`DeviceBufferView::new`] が構築時に行うため、本メソッドの実装は
    /// 追加のオフセット境界検査を要しないが、カーネル側の手動境界検査
    /// 〈REQ-8〉は従来どおり省略しない）。
    ///
    /// `bias` は `Some` の場合 `[n]`（`w` の列数）への行方向複製のみ
    /// 対応する（[`BackendOps::gemm_bias_act`] の融合カーネルと同じ厳密
    /// 一致契約。ブロードキャスト全般は非対応）。`k`（`a` の列数 = `w`
    /// の行数）が 0 の呼び出しは `sgd_step_device` と同様に呼び出し元
    /// （`fandhe_ai_autodiff::nn::linear::Linear::new` が `in_features == 0`
    /// を構築時に拒否する）の契約により実運用では到達しない。
    ///
    /// # デフォルト実装
    ///
    /// 本メソッドは `sgd_step_device`／`gemm_bias_act` と同じ非破壊拡張
    /// （デフォルトメソッド追加。公開 API 非破壊はガードレール条件・
    /// `.claude/rules/security.md`）であり、既定は
    /// [`BackendError::Unsupported`] を返す fail-safe とする（デバイス
    /// 常駐オペランドを扱えないバックエンドが誤って黙示のホスト
    /// フォールバック〈`w` を download してから `gemm_bias_act` へ委譲する
    /// 等〉を行い、D2H 排除という受け入れ条件を静かに破ることを防ぐため。
    /// `download` してよいなら本メソッドを呼ぶ意味がない）。CPU／CUDA／
    /// Metal の各実装はこのデフォルトをカーネル呼び出しでオーバーライド
    /// する（`backend-cpu::ops::CpuBackendOps`・`backend-cuda::ops::
    /// CudaBackendOps`・`backend-metal::ops::MetalBackendOps` 参照）。
    fn gemm_resident_rhs(
        &self,
        _a: &Tensor<f32>,
        _w: DeviceBufferView<'_>,
        _bias: Option<DeviceBufferView<'_>>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_resident_rhs: default fail-safe (no resident-operand GEMM kernel available)"
                .into(),
        ))
    }

    /// デバイス常駐 `w` のまま `c = w @ b` を計算する（イシュー #1022・
    /// #1023「R3」）。
    ///
    /// `DeviceParamStore` の resident backward（`Op::LinearResident` の
    /// VJP。`fandhe_ai_autodiff::grad`）が `d_input^T = w @ g^T` を計算する
    /// ために使う（`w: [k, n]`・`b: [n, m]` → `c: [k, m]`。呼び出し元が
    /// `c` を転置して `d_input: [m, k]` を得る）。[`Self::gemm_resident_rhs`]
    /// と対になる「常駐オペランドが左辺」の形（`w` が左、`b` がホスト
    /// 常駐の右辺）。`w` が [`DeviceBufferView`] を取る理由は
    /// [`Self::gemm_resident_rhs`] と同じ。
    ///
    /// # デフォルト実装
    ///
    /// [`Self::gemm_resident_rhs`] と同じ理由・同じ fail-safe 方針
    /// （[`BackendError::Unsupported`]）のデフォルトメソッド。
    fn gemm_resident_lhs(
        &self,
        _w: DeviceBufferView<'_>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "gemm_resident_lhs: default fail-safe (no resident-operand GEMM kernel available)"
                .into(),
        ))
    }

    /// 融合グラフ（#162 が検出した elementwise 連鎖・#163 が生成する
    /// カーネル）を 1 回のカーネル呼び出しで実行する（TASK-12.1d・#164）。
    ///
    /// `gemm_bias_act` と同型の非破壊拡張（デフォルトメソッド追加）。
    /// デフォルト実装は `BackendError::Unsupported` を返す fail-safe
    /// （既存 elementwise・reduction 未実装カーネルと同じ設計）であり、
    /// `fandhe_ai_autodiff::Tape` の実体化経路（`materialize_fallible`／
    /// `materialize_non_fallible`。`crates/autodiff/src/tape.rs`）は
    /// `Unsupported` を検出した場合に `leaves` を使わず `self`（同じ
    /// `ops`）の per-op メソッド（`add`／`mul`／`relu`／`exp`／`tanh`）へ
    /// 逐次フォールバックする契約（`docs/fusion-graph-design.md` §3.4・
    /// §3.5.2・§3.5.3）。CPU 融合実行の提供元は `backend-cpu` 側の
    /// `run_fused` オーバーライド（#163 のスコープ。本イシュー〈#164〉
    /// 時点では #163 が未マージのため、CPU 側も本デフォルト実装のまま
    /// フォールバックする）。CUDA／Metal は融合カーネル生成が未実装の間
    /// このデフォルトへフォールバックする。
    fn run_fused(
        &self,
        _plan: &FusionPlan,
        _leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "run_fused: default fail-safe (no fusion kernel available)".into(),
        ))
    }

    /// デバイスメモリプール（`backend-cuda::pool::CudaAllocator` 等。
    /// イシュー #1020・REQ-14）がアイドル保持している分を即座に実解放する。
    ///
    /// `crate::pool::PooledMemory::release_all_pooled`（`MemoryOps` デコレータ
    /// 側の解放 API）とは別経路であり、本メソッドはホットパス確保
    /// （`backend_ops::BackendOps` 経由の GEMM／elementwise／softmax カーネル）
    /// が使う `SizeClassPool` を対象とする。既定実装は no-op（`Ok(())`）で
    /// あり、プール未接続のバックエンド（`backend-cpu`・`backend-metal`。
    /// Metal 実装は後続 #1021）を破壊しない非破壊拡張（デフォルトメソッド
    /// 追加）である。`backend-cuda::CudaBackendOps` はプール実体へ委譲する
    /// オーバーライドを持つ（`crates/backend-cuda/src/ops.rs`）。
    fn release_cached_device_memory(&self) -> Result<(), BackendError> {
        Ok(())
    }

    /// デバイスメモリプールの現在の利用統計（[`crate::PoolStats`]）を返す。
    ///
    /// プールを持たないバックエンド（既定実装。`backend-cpu`・
    /// 本イシュー時点の `backend-metal`）は `None` を返す。`backend-cuda`
    /// のみ `Some(stats)` を返すオーバーライドを持つ。
    fn device_memory_pool_stats(&self) -> Option<crate::PoolStats> {
        None
    }
}

/// 複数の `&dyn BackendOps` を横断して `device` に一致する実装を選択する。
///
/// `device::select_from`（TASK-1.9a）と同型の注入式ディスパッチ:
/// `tensor-core` は `backend-cpu`／`backend-cuda`／`backend-metal` を直接
/// 参照できないため、呼び出し側（結線を担う上位クレート・テスト）が
/// `ops` を注入する。本関数こそが受け入れ条件「同一コードで 3 バック
/// エンドのカーネルが呼び分けられる」の直接の実装であり、`device` の
/// variant にのみ基づいて対応実装を返す（形状・HW ヒューリスティクスは
/// 一切持ち込まない。TASK-11.2b・#68 のスコープ）。
///
/// 対応する実装が `ops` に含まれない場合は
/// [`BackendError::DeviceUnavailable`] を返す（`device::select_from` と
/// 同じエラー variant・同じ意味論。「対応 provider／ops 未登録」を表す）。
pub fn ops_for<'a>(
    ops: &[&'a dyn BackendOps],
    device: Device,
) -> Result<&'a dyn BackendOps, BackendError> {
    ops.iter()
        .find(|candidate| candidate.device() == device)
        .copied()
        .ok_or_else(|| {
            BackendError::DeviceUnavailable(format!(
                "no BackendOps registered for device {device:?}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::BufferHandle;
    use crate::error::ShapeError;
    use std::any::Any;

    /// テスト専用のモック `BackendOps`。実バックエンドに依存せず
    /// `ops_for` の選択ロジックを検証するために `tensor-core` 内で定義
    /// する（実バックエンドの検証は各バックエンドクレートの結合テスト
    /// で行う。`device` モジュールの `MockProvider` と同じ位置付け）。
    struct MockOps(Device);

    impl BackendOps for MockOps {
        fn device(&self) -> Device {
            self.0
        }

        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: gemm".into()))
        }

        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: add".into()))
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    /// `gemm_bias_act` のデフォルト実装（非融合合成）を数値検証するための
    /// naive 計算モック。`MockOps`（常に `Unsupported`）と異なり `gemm`／
    /// `add`／`relu` を実際に計算する（行方向ブロードキャストのみ対応する
    /// 簡易 `add`。テスト用途のため `Tensor::get`／strided view には
    /// 対応しない）。
    struct ComputingMockOps;

    impl BackendOps for ComputingMockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }

        fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let (m, k) = (a.shape()[0], a.shape()[1]);
            let n = b.shape()[1];
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let mut out = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for p in 0..k {
                        acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                    }
                    out[i * n + j] = acc;
                }
            }
            Tensor::new(out, &[m, n]).map_err(BackendError::ShapeMismatch)
        }

        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            // テストで使う形状のみ対応: `a: [m, n]`・`b: [n]`（行方向
            // ブロードキャスト）または同一 shape。
            let a_shape = a.shape().to_vec();
            let a_data = a.as_slice().expect("test: a must be contiguous");
            let b_data = b.as_slice().expect("test: b must be contiguous");
            let out = if b.shape() == a.shape() {
                a_data
                    .iter()
                    .zip(b_data)
                    .map(|(x, y)| x + y)
                    .collect::<Vec<_>>()
            } else if b.shape().len() == 1 && a_shape.len() == 2 && b.shape()[0] == a_shape[1] {
                let n = a_shape[1];
                a_data
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| x + b_data[idx % n])
                    .collect::<Vec<_>>()
            } else {
                return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                    expected: a_shape.len(),
                    actual: b.shape().len(),
                }));
            };
            Tensor::new(out, &a_shape).map_err(BackendError::ShapeMismatch)
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: mul".into()))
        }

        fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            let data = a.as_slice().expect("test: a must be contiguous");
            let out = data.iter().map(|x| x.max(0.0)).collect::<Vec<_>>();
            Tensor::new(out, a.shape()).map_err(BackendError::ShapeMismatch)
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("computing mock: max".into()))
        }
    }

    /// object-safe であることの型検査を兼ねる（`Box<dyn BackendOps>` が
    /// 構築できることをコンパイル時に確認する）。
    fn assert_object_safe(_ops: &dyn BackendOps) {}

    #[test]
    fn gemm_bias_act_default_matches_manual_composition() {
        let ops = ComputingMockOps;
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let bias = Tensor::new(vec![-100.0, 1.0], &[2]).unwrap();

        // A@B = [[19, 22], [43, 50]] → + bias [-100, 1] → [[-81, 23], [-57, 51]]
        // → relu → [[0, 23], [0, 51]]
        let out = ops
            .gemm_bias_act(&a, &b, Some(&bias), Activation::Relu)
            .expect("gemm_bias_act should succeed");
        assert_eq!(out.as_slice().unwrap(), &[0.0, 23.0, 0.0, 51.0]);
    }

    #[test]
    fn gemm_bias_act_default_no_bias_no_act_matches_gemm() {
        let ops = ComputingMockOps;
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let plain_gemm = ops.gemm(&a, &b).unwrap();
        let fused = ops
            .gemm_bias_act(&a, &b, None, Activation::None)
            .expect("gemm_bias_act should succeed");
        assert_eq!(
            plain_gemm.as_slice().unwrap(),
            fused.as_slice().unwrap(),
            "bias=None・act=None は gemm と同一結果のはず"
        );
    }

    #[test]
    fn gemm_bias_act_default_propagates_unsupported_from_composed_ops() {
        // `MockOps` は `gemm` 自体が `Unsupported` を返すため、
        // デフォルト実装が最初のステップのエラーをそのまま伝播することを
        // 検証する（GPU バックエンドが GEMM 自体未実装の場合の fail-safe。
        // elementwise 未実装〈`add`/`relu` が `Unsupported`〉の伝播は
        // `backend-cuda`/`backend-metal` の結合テスト側で検証する）。
        let ops = MockOps(Device::Cpu);
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

        let result = ops.gemm_bias_act(&a, &b, None, Activation::Relu);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    #[test]
    fn ops_for_dispatches_to_matching_device() {
        let cpu = MockOps(Device::Cpu);
        let cuda = MockOps(Device::Cuda(0));
        let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

        let selected = ops_for(&ops, Device::Cuda(0)).expect("cuda ops registered");
        assert_eq!(selected.device(), Device::Cuda(0));
        assert_object_safe(selected);

        let selected = ops_for(&ops, Device::Cpu).expect("cpu ops registered");
        assert_eq!(selected.device(), Device::Cpu);
    }

    #[test]
    fn ops_for_missing_device_returns_device_unavailable() {
        let cpu = MockOps(Device::Cpu);
        let ops: Vec<&dyn BackendOps> = vec![&cpu];

        // `ops_for` の `Ok` 側は `&dyn BackendOps` を含み `Debug` を実装
        // しないため `expect_err` は使わず、`is_err`／`matches!` で
        // `Err` 経路のみ検査する。
        let result = ops_for(&ops, Device::Cuda(0));
        assert!(result.is_err());
        assert!(matches!(result, Err(BackendError::DeviceUnavailable(_))));
    }

    #[test]
    fn unsupported_error_carries_shape_error_independently() {
        // `BackendError::Unsupported` が既存 variant（`ShapeMismatch` 等）と
        // 独立して構築・表示できることを確認する（非破壊追加の検証）。
        let err = BackendError::Unsupported("elementwise add on cuda".into());
        assert!(err.to_string().contains("elementwise add on cuda"));

        let shape_err = BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: 1,
        });
        assert!(!shape_err.to_string().is_empty());
    }

    #[test]
    fn run_fused_default_returns_unsupported() {
        // `run_fused`（TASK-12.1d・#164）のデフォルト実装は `Unsupported`
        // を返す fail-safe（`gemm_bias_act` 等の既存 elementwise・
        // reduction 未実装カーネルと同型の設計。backend_ops.rs 冒頭コメ
        // ント参照）。`MockOps` はこのデフォルトを override しない。
        let ops = MockOps(Device::Cpu);
        // `from_ops`（`fusion::plan`。TASK-12.1c・#163）は「`Input` エント
        // リのみで elementwise ノードが 1 個も無い」プランを
        // `FusionPlanError::NoElementwiseNode` として拒否する契約
        // （融合する意味が無いため。`plan.rs` ドキュメント参照）ため、本
        // テストは最小の elementwise ノード（`Relu`）を 1 個含む有効な
        // プランを使う。
        let plan = crate::fusion::FusionPlan::from_ops(
            vec![
                crate::fusion::FusedOpKind::Input { leaf_index: 0 },
                crate::fusion::FusedOpKind::Relu { input: 0 },
            ],
            vec![4],
            crate::dispatch::DType::F32,
            1,
        )
        .expect("from_ops should succeed for a minimal single-op plan");
        let leaf = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let leaves: Vec<&Tensor<f32>> = vec![&leaf];
        let result = ops.run_fused(&plan, &leaves);
        assert!(matches!(result, Err(BackendError::Unsupported(_))));
    }

    /// テスト専用の最小 `BufferHandle`（イシュー #1017・
    /// `sgd_step_device_tracked_default_delegates_to_sgd_step_device`
    /// が `DeviceBuffer<f32>` を構築するためだけに使う。データの実体は
    /// 持たず downcast のためだけの空ハンドル）。
    #[derive(Debug)]
    struct EmptyHandle;

    impl BufferHandle for EmptyHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    fn empty_device_buffer(device: Device) -> DeviceBuffer<f32> {
        DeviceBuffer::new(device, vec![1], Box::new(EmptyHandle))
    }

    /// [`BackendOps::sgd_step_device_tracked`] のデフォルト実装が
    /// `token` を無視して [`BackendOps::sgd_step_device`] へそのまま
    /// 委譲することを確認する（イシュー #1017 の非破壊拡張ガード。
    /// `MockOps` はいずれのメソッドもオーバーライドしていないため、
    /// 両者が同一の `Unsupported` メッセージを返すことで委譲を検証する）。
    #[test]
    fn sgd_step_device_tracked_default_delegates_to_sgd_step_device() {
        let ops = MockOps(Device::Cpu);
        let mut param = empty_device_buffer(Device::Cpu);
        let grad = empty_device_buffer(Device::Cpu);
        let config = SgdStepConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            is_first_step: true,
        };
        let token = DispatchFailureCell::new();

        let direct = ops.sgd_step_device(&mut param, &grad, None, &config);
        let tracked = ops.sgd_step_device_tracked(&mut param, &grad, None, &config, &token);

        match (direct, tracked) {
            (Err(BackendError::Unsupported(a)), Err(BackendError::Unsupported(b))) => {
                assert_eq!(a, b);
            }
            other => panic!("expected both to return the same Unsupported error: {other:?}"),
        }
        // デフォルト委譲は token に一切触れない。
        assert!(!token.is_set());
    }
}
