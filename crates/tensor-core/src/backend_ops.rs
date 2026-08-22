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
use crate::device::{BackendError, Device};
use crate::fusion::FusionPlan;

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
    use crate::error::ShapeError;

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
}
