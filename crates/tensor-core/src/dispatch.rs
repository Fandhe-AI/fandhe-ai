//! GEMM 経路選択（ディスパッチ規則）の抽象層実装（TASK-11.2b・#68）。
//!
//! REQ-11 v2（`docs/spec/04-requirements.md`）は、CubeCL autotune 相当の
//! 実行時ベンチマーク探索を廃し、**決定的な自作ディスパッチ規則**で
//! 行列演算ユニット（NVIDIA Tensor Core／Apple `simdgroup_matrix`）経路の
//! 選択を行う方針へ書き直し済みである。本モジュールはその規則の純関数
//! 実装であり、設計は先行イシュー #67（TASK-11.2a）が整備した
//! `docs/dispatch-rules-design.md`（決定表: 同文書 §5.3）を忠実に実装する。
//!
//! [`select_gemm_kernel`] は **利用者向けの明示切替 API ではない**。
//! REQ-11 受け入れ基準「行列演算ユニットの活用は、ライブラリの明示的な
//! 設定項目として利用者に提供しないこと」を満たすため、本関数は
//! `backend-cuda`／`backend-metal` の GEMM 自動経路入口（`gemm_auto.rs`・
//! `MetalGemm::dispatch_backend_auto` 等）が**内部で**呼ぶ規則エンジンで
//! あり、環境変数・feature flag・API 引数による上書き機構は設けない
//! （`crates/backend-cpu/src/gemm_blis/microkernel.rs` の ISA dispatch
//! 方針と同じ判断。外部入力が `unsafe` の駆動経路に影響しないため
//! OWASP A03 の観点で安全側）。
//!
//! 副作用なし・決定的な純関数として実装するため、実機（CUDA ドライバ・
//! Metal デバイス）なしで [`DeviceCaps`] を直接構築するだけでユニット
//! テスト可能である（本モジュール末尾の `#[cfg(test)]` 参照）。

/// GEMM の入力・出力 dtype（`docs/dispatch-rules-design.md` §4「dtype
/// ゲート」に対応）。
///
/// `BackendOps` v1 は f32 専用（`docs/public-api-design.md:469`）だが、
/// CUDA 側は f16 WMMA 経路（[`crate::device`] とは独立の `backend-cuda`
/// クレートが持つ）が既に実装済みのため、本 enum は両方を扱う。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32bit 浮動小数点。CUDA では既定で Tiled 経路（TF32 既定採用は
    /// #186 の実測・承認まで保留。§4 参照）。Metal では形状閾値付きで
    /// `simdgroup_matrix` 経路の対象になる。
    F32,
    /// 16bit 浮動小数点（`half::f16`）。CUDA WMMA 経路の対象
    /// （`cc >= CUDA_WMMA_MIN_CC`、形状下限なし）。
    F16,
}

/// GEMM の形状（`M x K` @ `K x N` = `M x N`）。
///
/// 経路選択の形状判定は `min(M, N, K)`（[`GemmShape::min_dim`]）を軸に
/// 行う（`docs/dispatch-rules-design.md` §3「形状判定」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemmShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

impl GemmShape {
    /// 新規 `GemmShape` を構築する。
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }

    /// `min(M, N, K)`。Metal の形状閾値判定（[`METAL_SIMDGROUP_MIN_DIM`]）
    /// に使う（`docs/dispatch-rules-design.md` §3.2）。
    pub fn min_dim(&self) -> u32 {
        self.m.min(self.n).min(self.k)
    }
}

/// フォールバック連鎖の到達先（`docs/dispatch-rules-design.md` §5.1・
/// §5.2）。CUDA の Tensor Core（WMMA/mma）と Metal の `simdgroup_matrix`
/// はいずれも行列演算ユニット経路として本 enum の `MatrixUnit` に統合する
/// （呼び出し側の `backend-cuda`／`backend-metal` が自クレート固有の型
/// （`CudaWmmaGemm`／`GemmVariant::Simdgroup` 系）へ写像する。§5.1
/// 「KernelKind: フォールバック連鎖の到達先を表す列挙」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    /// 行列演算ユニット経路（CUDA: WMMA/mma Tensor Core、Metal:
    /// `simdgroup_matrix`）。
    MatrixUnit,
    /// 共有メモリ／threadgroup メモリによるタイル化経路。
    Tiled,
    /// タイル化なしの素朴な経路（最終フォールバック）。
    Naive,
}

/// CUDA WMMA f16 経路が要求する compute capability 下限（major, minor）。
///
/// v1 PoC-8（CubeCL autotune 前提）の参考値・一般的な NVIDIA アーキ
/// テクチャ世代対応（WMMA は Volta＝cc 7.0 以降）による**暫定値**であり、
/// 世代境界の実機再確認は #69（TASK-11.2c）が担当する
/// （`docs/dispatch-rules-design.md` §2 表・確定度「暫定」欄）。
/// `backend-cuda::gemm_wmma::MIN_COMPUTE_CAPABILITY_MAJOR`（`= 7`）と
/// 値を一致させる契約（本定数が正本。閾値定数は 1 箇所集約する設計
/// 方針、§3.2「閾値定数は 1 箇所に集約する」）。
pub const CUDA_WMMA_MIN_CC: (i32, i32) = (7, 0);

/// CUDA TF32 経路が要求する compute capability 下限（major, minor）。
///
/// TF32 対応 Tensor Core は Ampere（cc 8.0）以降という一般的な世代対応
/// による**暫定値**（`docs/dispatch-rules-design.md` §2 表）。TF32 経路
/// 自体の実装・既定採用可否は #62・#186（TASK-11.1g）の実測・ユーザー
/// 承認まで保留のため、現時点の [`select_gemm_kernel`] 決定表では本定数
/// は未使用（`cc` ゲートを満たしても f32 は常に [`KernelKind::Tiled`] を
/// 返す。§4「f32 CUDA TF32: 既定採用を保留」）。将来 TF32 経路が有効化
/// される際に使う分岐点として先んじて定義のみ行う。
#[allow(dead_code)]
pub const CUDA_TF32_MIN_CC: (i32, i32) = (8, 0);

/// Metal `simdgroup_matrix` 経路を選択する形状下限（`min(M, N, K)`）。
///
/// v1 PoC-8 実測（M=N=K=256 で accelerated が unit の約 20.5 倍高速・
/// 256/512 の 2 点のみ計測）を参考値とした**暫定値**。境界形状（384・640
/// 等）の実測再検証は #69（TASK-11.2c）が担当する
/// （`docs/dispatch-rules-design.md` §3.1・§3.2）。CUDA 側には対応する
/// 形状閾値を設けない非対称設計（§3.2「CUDA との非対称設計の実体」）。
pub const METAL_SIMDGROUP_MIN_DIM: u32 = 512;

/// GEMM 実行時に参照するデバイスケイパビリティ（`docs/dispatch-rules-design.md`
/// §5.1「`DeviceCaps`: HW 判定結果をキャッシュした構造体」）。
///
/// デバイス初期化時に 1 回構築し、GEMM 呼び出しごとの FFI（`cudarc`・
/// `objc2-metal`）照会を繰り返さない契約（§2.1「判定タイミング」。
/// `backend-cuda`／`backend-metal` 側の初期化コードがこの契約を守る）。
/// フィールドはすべて `pub` とし、実機なしのテスト・`backend-*` クレート
/// の初期化コードから直接構築できるようにする（FFI なしで構築可能な
/// pub コンストラクタ、計画書 §3.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceCaps {
    /// CUDA compute capability（major, minor）。CUDA ドライバ不在・
    /// `backend-cuda` 以外の呼び出し元では `None`（fail-safe。§2.2）。
    pub cuda_compute_capability: Option<(i32, i32)>,
    /// Metal `MTLGPUFamily::Apple7` 以上に対応しているか
    /// （`MTLDevice::supportsFamily(MTLGPUFamilyApple7)`）。判定不能・
    /// 非 macOS 環境では `false`（fail-safe。§2.2「非対応ファミリは
    /// tiled → naive へフォールバック」）。
    pub metal_apple7_supported: bool,
}

impl DeviceCaps {
    /// CUDA デバイス向けの `DeviceCaps` を構築する。`backend-cuda` の
    /// デバイス初期化コード（`CudaDevice::compute_capability()`）から
    /// 呼ばれる想定（§2.1）。
    pub fn cuda(compute_capability: (i32, i32)) -> Self {
        Self {
            cuda_compute_capability: Some(compute_capability),
            metal_apple7_supported: false,
        }
    }

    /// CUDA ドライバ不在・非対応時の `DeviceCaps`（fail-safe。§2.2）。
    pub fn cuda_unavailable() -> Self {
        Self {
            cuda_compute_capability: None,
            metal_apple7_supported: false,
        }
    }

    /// Metal デバイス向けの `DeviceCaps` を構築する。`backend-metal` の
    /// コンテキスト初期化コード（`MTLDevice::supportsFamily`）から呼ばれる
    /// 想定（§2.1）。
    pub fn metal(apple7_supported: bool) -> Self {
        Self {
            cuda_compute_capability: None,
            metal_apple7_supported: apple7_supported,
        }
    }
}

/// 形状・HW ケイパビリティ・dtype から使用カーネル経路を決定する
/// （`docs/dispatch-rules-design.md` §5.1 のシグネチャをそのまま実装）。
///
/// 副作用なし・決定的（同一入力に対し常に同一出力）。**利用者向け切替
/// API ではなく、バックエンド実装が内部で呼ぶ規則エンジンである**
/// （REQ-11 受け入れ基準「明示切替 API を提供しない」。モジュール冒頭
/// コメント参照）。
///
/// 決定表（§5.3。`caps.metal_apple7_supported` が真の場合は CUDA
/// ケイパビリティを無視し Metal 判定を優先する。両者が同時に true に
/// なる呼び出しは想定しないが、`DeviceCaps` はバックエンド別に構築される
/// ため型レベルでは排他性を強制しない設計とした。呼び出し元
/// （`backend-cuda`／`backend-metal`）はそれぞれ自バックエンド分のみを
/// 設定した `DeviceCaps` を渡す契約とする）:
///
/// | HW ゲート | dtype | 形状 | 選択経路 |
/// |---|---|---|---|
/// | CUDA `cc >= 7.0` | F16 | 任意（形状下限なし） | `MatrixUnit`（WMMA f16） |
/// | CUDA `cc >= 7.0`（f32 は TF32 経路保留） | F32 | 任意 | `Tiled` |
/// | CUDA `cc` 非対応 or ドライバ不在 | 任意 | 任意 | `Tiled`（呼び出し元がさらに `Naive` へフォールバックしうる） |
/// | Metal Apple7 以上 | F32 | `min(M,N,K) >= 512` | `MatrixUnit`（simdgroup） |
/// | Metal Apple7 以上 | F32 | `min(M,N,K) < 512` | `Tiled` |
/// | Metal Apple7 未満 | F32 | 任意 | `Tiled`（呼び出し元がさらに `Naive` へフォールバックしうる） |
///
/// 端数形状（fragment 整数倍でない形状）は経路除外の条件にしない
/// （§3.3。境界検査はカーネル内部の責務であり、ディスパッチ側は形状の
/// 大小のみを判定する）。フォールバック連鎖の最終段（`Tiled` が使えない
/// 場合の `Naive`）は本関数の責務外とし、`backend-cuda`／`backend-metal`
/// の呼び出し元がカーネルコンパイル失敗等の実行時エラーを検知して
/// フォールバックする（§5.2。fail-safe の実装は各バックエンドクレート
/// 側、`gemm_auto.rs` 等を参照）。
pub fn select_gemm_kernel(caps: &DeviceCaps, shape: GemmShape, dtype: DType) -> KernelKind {
    // Metal Apple7 以上: 形状閾値付きで MatrixUnit（simdgroup）。
    // CUDA compute_capability が Some の呼び出しでは metal_apple7_supported
    // は false（DeviceCaps::cuda コンストラクタ）であるためこの分岐には
    // 入らない。
    if caps.metal_apple7_supported {
        return if shape.min_dim() >= METAL_SIMDGROUP_MIN_DIM {
            KernelKind::MatrixUnit
        } else {
            KernelKind::Tiled
        };
    }

    // CUDA: cc ゲートを満たす場合のみ dtype 別に分岐する。cc はタプルの
    // 辞書式比較（major 優先・同一 major なら minor 比較）で判定する。
    if let Some(cc) = caps.cuda_compute_capability {
        if cc >= CUDA_WMMA_MIN_CC {
            return match dtype {
                // f16: 形状下限なしで常に MatrixUnit（WMMA）。§3.2。
                DType::F16 => KernelKind::MatrixUnit,
                // f32: TF32 経路は #186 待ちのため常に Tiled（§4）。
                DType::F32 => KernelKind::Tiled,
            };
        }
        // cc 非対応（cc < 7.0）: Tiled へ倒す（§5.3 決定表 3 行目）。
        return KernelKind::Tiled;
    }

    // CUDA ドライバ不在・Metal Apple7 未満・いずれの HW 判定にも
    // 該当しない（判定不能）場合は fail-safe で Tiled を返す（§2.2・§5.3）。
    KernelKind::Tiled
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CUDA f16: cc >= 7.0 なら形状によらず MatrixUnit ---

    #[test]
    fn cuda_f16_cc_at_gate_selects_matrix_unit_regardless_of_shape() {
        let caps = DeviceCaps::cuda((7, 0));
        // 小形状（256）でも MatrixUnit（§3.2「CUDA は形状下限を設けない」）。
        let small = GemmShape::new(256, 256, 256);
        assert_eq!(
            select_gemm_kernel(&caps, small, DType::F16),
            KernelKind::MatrixUnit
        );
        let large = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, large, DType::F16),
            KernelKind::MatrixUnit
        );
    }

    #[test]
    fn cuda_f16_gb10_cc_12_1_selects_matrix_unit() {
        // GB10（Blackwell、sm_121）。cc major が gate の major を上回る
        // ケースをタプル比較の落とし穴なく扱えることを検証する。
        let caps = DeviceCaps::cuda((12, 1));
        let shape = GemmShape::new(512, 512, 512);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F16),
            KernelKind::MatrixUnit
        );
    }

    #[test]
    fn cuda_f16_cc_below_gate_falls_back_to_tiled() {
        // cc (6, x) は WMMA 非対応（Pascal 以前）。
        let caps = DeviceCaps::cuda((6, 9));
        let shape = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F16),
            KernelKind::Tiled
        );
    }

    #[test]
    fn cuda_f16_cc_just_below_minor_boundary_falls_back_to_tiled() {
        // (7, 0) がゲートちょうど。同一 major で minor が届かない
        // 組み合わせは実在しない（cc は major.minor の辞書式比較で
        // major が同じなら常に minor >= 0 のため 7.0 未満は major < 7
        // のみ）が、タプル比較の境界を明示的に検証する。
        let caps = DeviceCaps::cuda((6, 99));
        let shape = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F16),
            KernelKind::Tiled
        );
    }

    // --- CUDA f32: TF32 既定採用保留のため常に Tiled ---

    #[test]
    fn cuda_f32_with_cc_8_0_selects_tiled_pending_tf32_approval() {
        let caps = DeviceCaps::cuda((8, 0));
        let shape = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F32),
            KernelKind::Tiled
        );
    }

    #[test]
    fn cuda_f32_with_cc_7_0_selects_tiled() {
        let caps = DeviceCaps::cuda((7, 0));
        let shape = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F32),
            KernelKind::Tiled
        );
    }

    // --- CUDA: ドライバ不在・cc 非対応の fail-safe ---

    #[test]
    fn cuda_unavailable_selects_tiled() {
        let caps = DeviceCaps::cuda_unavailable();
        let shape = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F16),
            KernelKind::Tiled
        );
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F32),
            KernelKind::Tiled
        );
    }

    // --- Metal: 形状閾値の境界値（511/512） ---

    #[test]
    fn metal_apple7_min_dim_at_threshold_selects_matrix_unit() {
        let caps = DeviceCaps::metal(true);
        let shape = GemmShape::new(512, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F32),
            KernelKind::MatrixUnit
        );
    }

    #[test]
    fn metal_apple7_min_dim_just_below_threshold_selects_tiled() {
        let caps = DeviceCaps::metal(true);
        let shape = GemmShape::new(511, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F32),
            KernelKind::Tiled
        );
    }

    #[test]
    fn metal_apple7_min_dim_uses_minimum_of_all_three_axes() {
        // M・N は大形状だが K のみ小さい場合、min(M,N,K) 判定により
        // Tiled へ倒れることを検証する（軸ごとの独立判定ではないこと）。
        let caps = DeviceCaps::metal(true);
        let shape = GemmShape::new(4096, 4096, 256);
        assert_eq!(
            select_gemm_kernel(&caps, shape, DType::F32),
            KernelKind::Tiled
        );
    }

    // --- Metal: Apple7 未満の fail-safe ---

    #[test]
    fn metal_apple7_unsupported_selects_tiled_regardless_of_shape() {
        let caps = DeviceCaps::metal(false);
        let large = GemmShape::new(4096, 4096, 4096);
        assert_eq!(
            select_gemm_kernel(&caps, large, DType::F32),
            KernelKind::Tiled
        );
    }

    // --- 端数形状: 経路除外の条件にしない（§3.3） ---

    #[test]
    fn fragment_unaligned_shape_does_not_exclude_matrix_unit_path() {
        // WMMA fragment（16 の倍数）にも simdgroup_matrix（8 の倍数）にも
        // 揃わない 513 という端数形状でも、HW・形状ゲートさえ満たせば
        // MatrixUnit を選択する（境界検査はカーネル内部の責務。§3.3）。
        let cuda_caps = DeviceCaps::cuda((7, 0));
        let shape = GemmShape::new(513, 513, 513);
        assert_eq!(
            select_gemm_kernel(&cuda_caps, shape, DType::F16),
            KernelKind::MatrixUnit
        );

        let metal_caps = DeviceCaps::metal(true);
        assert_eq!(
            select_gemm_kernel(&metal_caps, shape, DType::F32),
            KernelKind::MatrixUnit
        );
    }

    // --- GemmShape::min_dim ---

    #[test]
    fn gemm_shape_min_dim_returns_smallest_axis() {
        assert_eq!(GemmShape::new(10, 20, 30).min_dim(), 10);
        assert_eq!(GemmShape::new(30, 10, 20).min_dim(), 10);
        assert_eq!(GemmShape::new(30, 20, 10).min_dim(), 10);
    }
}
