//! NVRTC コンパイルヘルパ（カーネルソース非依存の基盤部分）。
//!
//! PoC-v2-3 の `compile`（`docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs:84-117`）
//! を productize したもの。カーネルソース文字列・起動 API は #33/#34 の
//! スコープであり、本モジュールは「ソース文字列と arch を渡すと `Ptx` を
//! 返す」境界のみを担う。
//!
//! # A03（インジェクション）対応
//!
//! `CUDA_INCLUDE_PATH` 環境変数はコンパイルオプションの include パス
//! 文字列としてのみ `CompileOptions::include_paths` へ渡し、シェル展開・
//! コマンド実行には一切使わない（`.claude/rules/security.md`）。
//! カーネルソース（`src` 引数）はコンパイル時定数（`&'static str`）
//! であることを呼び出し元（#33/#34）に要求し、外部入力をソース文字列へ
//! 連結しない方針とする。

use std::num::NonZeroU32;

use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use crate::error::CudaError;

/// カーネル特化パラメータ記述子（Phase C-1・イシュー #504）。
///
/// 親イシュー #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・
/// 静的タイル選定）の先頭タスクであり、後続タスクが共通に使う
/// 「キャッシュキーの単位」を定義する。C-2（自作ハッシュ・ディレクトリ
/// 命名・#506）・C-4（プロセス内 LRU・#511）・C-6（テンプレート展開・
/// #516）から利用される想定であり、本モジュールではキー型の定義のみを
/// 担う（コンパイル・キャッシュ本体は持たない）。
///
/// 設計は 2 つの参照実装を踏まえる:
/// - DeepGEMM: キャッシュキーに「カーネル名・コンパイラ種別＋バージョン
///   （例 NVRTC12.9）・コンパイルフラグ・ソース」を含め、GPU アーキ違い
///   （`--gpu-architecture=sm_XX`）が自動的に別エントリになる設計。
/// - metal-flash-attention: descriptor 全パラメータを `Optional` で持ち
///   未確定時は `fatalError` する設計 + `Equatable + Hashable` なキー型。
///
/// 本型は後者の「Optional + 実行時 panic」を Rust の型システムで排除する
/// （受け入れ基準）。全フィールドを [`CudaKernelDescriptor::new`] の必須
/// 引数として要求し、ブロックタイル寸法・パイプライン段数は
/// [`NonZeroU32`] でゼロ値を型レベルで排除するため、「未確定パラメータの
/// まま生成できる」状態が構築できない。
///
/// フィールドは private + getter とし、構築後にゼロ値へ書き換えられない
/// （不変条件を型で維持する）。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaKernelDescriptor {
    /// カーネル名。C-2（#506）のキャッシュディレクトリ命名
    /// （`kernel.<name>.<hash>`）で使われる想定であり、`&'static str`
    /// （コンパイル時定数）に限定して外部入力の混入を型で遮断する
    /// （A03 対策。`compile_ptx` の `src` 引数契約〈本ファイル冒頭〉と
    /// 同型の判断）。
    kernel_name: &'static str,
    /// GEMM 形状（M/N/K）。`tensor_core::dispatch` の既存 `Hash + Eq` 型を
    /// 再利用し、ディスパッチ規則（#67/#68）が扱う形状表現と揃える。
    shape: tensor_core::dispatch::GemmShape,
    /// ブロックタイル M 次元。
    block_m: NonZeroU32,
    /// ブロックタイル N 次元。
    block_n: NonZeroU32,
    /// ブロックタイル K 次元。
    block_k: NonZeroU32,
    /// パイプライン段数。SMEM 予算からの逆算ロジックは
    /// [`derive_pipeline_stages`]（C-8・#521 で実装）が担い、本型は
    /// 導出済みの値をデータとして保持するのみ。
    stages: NonZeroU32,
    /// 入出力 dtype。`tensor_core::dispatch::DType` を再利用する。
    dtype: tensor_core::dispatch::DType,
}

impl CudaKernelDescriptor {
    /// `CudaKernelDescriptor` を構築する。
    ///
    /// `block_m`／`block_n`／`block_k`／`stages` がゼロの場合は
    /// `CudaError::InvalidKernelDescriptor` を返す（panic 経路なし。
    /// `unwrap`/`expect` は使わない。`.claude/rules/coding-rust.md`）。
    ///
    /// `kernel_name` はパス走査文字（`/`・`\`・`..`）と空文字列を拒否する
    /// （codex-review 指摘・PR #641・P2）。`kernel_name` は `&'static str`
    /// でも `"../escape"` 等のリテラルは書けてしまい、C-2（#506）の
    /// キャッシュディレクトリ命名（`kernel.<name>.<hash>`）で
    /// そのままパスセグメントへ使われる想定のため、型（`&'static str`）
    /// だけでは意図しないパス脱出を防げない。値の妥当性は構築時に
    /// ここで検証し、C-2 側は「検証済み文字列である」ことを前提にできる。
    pub fn new(
        kernel_name: &'static str,
        shape: tensor_core::dispatch::GemmShape,
        block_m: u32,
        block_n: u32,
        block_k: u32,
        stages: u32,
        dtype: tensor_core::dispatch::DType,
    ) -> Result<Self, CudaError> {
        if kernel_name.is_empty()
            || kernel_name.contains('/')
            || kernel_name.contains('\\')
            || kernel_name.contains("..")
        {
            return Err(CudaError::InvalidKernelDescriptor {
                detail: format!(
                    "kernel_name must be a non-empty path-safe segment \
                     (no '/', '\\\\', or \"..\"), got {kernel_name:?}"
                ),
            });
        }
        let non_zero = |value: u32, field: &str| {
            NonZeroU32::new(value).ok_or_else(|| CudaError::InvalidKernelDescriptor {
                detail: format!("{field} must be non-zero (got 0)"),
            })
        };
        Ok(Self {
            kernel_name,
            shape,
            block_m: non_zero(block_m, "block_m")?,
            block_n: non_zero(block_n, "block_n")?,
            block_k: non_zero(block_k, "block_k")?,
            stages: non_zero(stages, "stages")?,
            dtype,
        })
    }

    /// カーネル名。
    pub fn kernel_name(&self) -> &'static str {
        self.kernel_name
    }

    /// GEMM 形状。
    pub fn shape(&self) -> tensor_core::dispatch::GemmShape {
        self.shape
    }

    /// ブロックタイル M 次元。
    pub fn block_m(&self) -> NonZeroU32 {
        self.block_m
    }

    /// ブロックタイル N 次元。
    pub fn block_n(&self) -> NonZeroU32 {
        self.block_n
    }

    /// ブロックタイル K 次元。
    pub fn block_k(&self) -> NonZeroU32 {
        self.block_k
    }

    /// パイプライン段数。
    pub fn stages(&self) -> NonZeroU32 {
        self.stages
    }

    /// 入出力 dtype。
    pub fn dtype(&self) -> tensor_core::dispatch::DType {
        self.dtype
    }
}

/// コンパイルキャッシュのキー（Phase C-1・イシュー #504）。
///
/// [`CudaKernelDescriptor`] に加え、環境依存パラメータ（compute
/// capability・NVRTC バージョン・コンパイルフラグ）を含める。DeepGEMM の
/// 設計（キャッシュキーに compiler signature を含め GPU アーキ違いが
/// 自動的に別エントリになる）を踏襲し、環境が変わった際に古いキャッシュ
/// エントリが誤って再利用されない性質を担保する（OWASP A08 整合性。
/// `.claude/rules/security.md`）。
///
/// ソースコード断片のハッシュ（DeepGEMM の `code` 相当）は本キーには
/// 含めない（C-5・#514 のスコープ。ソース断片によるキー拡張・キャッシュ
/// 無効化はそちらで扱う）。ディレクトリ命名（`kernel.<name>.<hash>`）と
/// ハッシュ関数自体は C-2（#506）のスコープであり、本型は「ハッシュ化
/// される前のキーの単位」を定義するのみ。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CudaKernelCacheKey {
    descriptor: CudaKernelDescriptor,
    /// `CudaDevice::compute_capability()` と同型（`device.rs`）。
    compute_capability: (i32, i32),
    /// NVRTC バージョン（[`nvrtc_version`] が返す値。DeepGEMM の
    /// compiler signature「NVRTC12.9」相当）。
    nvrtc_version: (i32, i32),
    /// `--gpu-architecture` 等のコンパイルフラグ。アーキ違いが自動的に
    /// 別キャッシュエントリになる性質はこのフィールドが担う。
    ///
    /// 順序契約: `Vec<String>` は要素順序を `Hash`/`Eq` に含めるため、
    /// 呼び出し元は意味的に同一なフラグ集合を常に決定的な順序で
    /// 構築すること（`HashSet` 等の非決定順コレクションから直接
    /// 構築しない）。契約違反時の影響は誤ヒット（安全性違反）方向では
    /// なく無駄なキャッシュミス方向のみだが、C-2（#506）でのキー消費側
    /// 実装時にも本契約を維持する。
    compile_flags: Vec<String>,
}

impl CudaKernelCacheKey {
    /// `descriptor`（検証済み）と環境パラメータからキーを構築する。
    /// `descriptor` は [`CudaKernelDescriptor::new`] を経由済みのため
    /// infallible（`Result` を返さない）。
    pub fn new(
        descriptor: CudaKernelDescriptor,
        compute_capability: (i32, i32),
        nvrtc_version: (i32, i32),
        compile_flags: Vec<String>,
    ) -> Self {
        Self {
            descriptor,
            compute_capability,
            nvrtc_version,
            compile_flags,
        }
    }

    /// `device`（`CudaDevice`）から compute capability を取得し、
    /// `nvrtc_version()` で NVRTC バージョンを取得してキーを構築する
    /// 実機導線。CI では GPU が無いため `new`（literal 構築）でテストする
    /// （ユニットテスト参照）。
    pub fn from_device(
        descriptor: CudaKernelDescriptor,
        device: &crate::device::CudaDevice,
        compile_flags: Vec<String>,
    ) -> Result<Self, CudaError> {
        Ok(Self::new(
            descriptor,
            device.compute_capability(),
            nvrtc_version()?,
            compile_flags,
        ))
    }

    /// カーネル特化パラメータ記述子。
    pub fn descriptor(&self) -> &CudaKernelDescriptor {
        &self.descriptor
    }

    /// compute capability（`major`, `minor`）。
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// NVRTC バージョン（`major`, `minor`）。
    pub fn nvrtc_version(&self) -> (i32, i32) {
        self.nvrtc_version
    }

    /// コンパイルフラグ。
    pub fn compile_flags(&self) -> &[String] {
        &self.compile_flags
    }
}

/// リンクされている NVRTC のバージョンを `(major, minor)` で返す。
///
/// [`CudaKernelCacheKey::from_device`] から呼ばれ、DeepGEMM の
/// compiler signature（例 NVRTC12.9）相当をキャッシュキーへ含めるために
/// 使う。`compile_ptx` と同じく `is_culib_present()` で先にプローブし、
/// 不在なら `CudaError::NvrtcUnavailable` を返す（dynamic-loading の
/// panic 経路回避。CUDA toolkit 非搭載環境でも panic しない契約は本関数
/// でも維持する。`build-no-cuda-toolkit` CI ジョブと整合）。
///
/// # Safety（`unsafe` 使用箇所）
///
/// `cudarc::nvrtc::sys::nvrtcVersion` は cudarc 0.19.8 に safe wrapper が
/// ないため FFI 境界の `unsafe` を必要最小限で使う（`.claude/rules/
/// security.md`）。呼び出し前に `is_culib_present()` で `libnvrtc` の
/// 存在を確認済みであり、`major`/`minor` はスタック上のローカル変数への
/// 有効なポインタを渡す（NVRTC API 契約上、両ポインタは非 null かつ
/// 呼び出し中のみ書き込まれる）。
pub fn nvrtc_version() -> Result<(i32, i32), CudaError> {
    // SAFETY: `cudarc::nvrtc::sys::is_culib_present()` は `libnvrtc` の
    // 存在確認のみを行う `dlopen` ベースのプローブで、引数を取らず
    // ポインタ・共有可変状態も扱わない（cudarc 0.19.8 に safe wrapper が
    // ないため FFI 境界の `unsafe` を必要最小限で使う。`.claude/rules/
    // security.md`）。本呼び出し自体が「`libnvrtc` が存在するか」を
    // 判定する目的であり、後続の `nvrtcVersion` 呼び出し（252 行目付近）
    // の前提確認としてここで先に行う。
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(CudaError::NvrtcUnavailable {
            detail: "libnvrtc dynamic library not found (dlopen failed); \
                     CUDA toolkit is not installed or not on the library search path"
                .to_string(),
        });
    }

    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    // SAFETY: `is_culib_present()`（上記ブロック）で `libnvrtc` の存在を
    // 確認済みのうえで呼ぶ。`major`/`minor` はこの直前で初期化した
    // スタックローカル変数への `&mut` であり、非 null かつ呼び出し中の
    // 書き込みに対して有効な唯一の可変参照を渡す（NVRTC API 契約上、
    // 両ポインタは `nvrtcVersion` 呼び出し中のみ書き込まれる）。
    // 1 番目のブロック（`is_culib_present()`）とは前提条件（引数なし・
    // 副作用なしの存在確認 vs 出力ポインタへの書き込みを伴うバージョン
    // 取得）が異なるため、ここで独立して根拠を明示する。
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(CudaError::NvrtcUnavailable {
            detail: format!("nvrtcVersion() failed: {result:?}"),
        });
    }
    Ok((major, minor))
}

/// `derive_pipeline_stages` が受け付ける段数の絶対上限（DeepGEMM
/// `kNumMaxStages`（`csrc/jit_kernels/heuristics/sm90.hpp` 付近）を踏襲）。
/// SMEM 予算がどれだけ大きくても、これ以上の段数はレジスタ圧・命令数
/// 増加に見合わないため候補から除外する。
pub const MAX_PIPELINE_STAGES: u32 = 16;

// カーネル契約上の絶対下限は `STAGES >= 2`（`kernels_mma.rs` の
// `cp.async.wait_group "n"(STAGES - 2)` が前提とする。同ファイル 194 行目
// 付近のコメント参照）。`derive_pipeline_stages` の `min_required`
// （DeepGEMM 由来の 3／4 段要求）はこの絶対下限より常に大きい値を返すため
// 通常この下限が直接効くことはないが、タイル構成が変わった場合の安全側の
// 前提として本コメントに明示する（値そのものを保持するだけの未使用定数は
// 置かない）。

/// SMEM 予算から `cp.async` パイプライン段数を逆算する（C-8・#521）。
///
/// # 導出式（DeepGEMM `sm90.hpp` の `get_smem_config` 系ヒューリスティクス
/// を踏襲）
///
/// ```text
/// smem_per_stage = (block_m * block_k + block_k * block_n) * bytes_per_element
/// derived        = min(smem_budget / smem_per_stage, MAX_PIPELINE_STAGES)
/// min_required   = if block_m * block_n < 128 * 192 { 4 } else { 3 }
/// ```
///
/// `derived < min_required` の場合は候補不成立として `Err` を返す
/// （DeepGEMM が対応する構成を棄却するのと同義。カーネル契約自体の絶対
/// 下限は `STAGES >= 2` だが、DeepGEMM 由来の `min_required` は常にそれ
/// 以上のため、本関数が返す `Err` はより保守的な「性能上望ましくない
/// 段数」を弾く判定になる）。
///
/// # 呼び出し文脈
///
/// SMEM 予算（`smem_budget_bytes`）は呼び出し元（`gemm_auto.rs` の
/// 結線ヘルパー想定。C-8 実装計画 §4）がデバイスの
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK` を実行時に取得し、
/// 本カーネル群が動的 SMEM opt-in（`cudaFuncSetAttribute`）を行わない
/// 静的 `__shared__` 構成であることを踏まえ 49,152 バイト（48KiB）で
/// クランプしたうえで渡す想定である（ハードコード定数を持たない設計。
/// `docs/perf/sm121-device-attributes.md` の SMEM 容量行が「未実測」の
/// ままであるため、実測値を推定で定数化しない安全側判断）。本関数自体は
/// 予算値の出所を問わない純関数として、その判断とは独立にテスト可能に
/// する。
///
/// # オーバーフロー安全性
///
/// `block_m`／`block_n`／`block_k` は `u32`（`CudaKernelDescriptor` の
/// フィールド型と同じ）で受け取り、乗算は `u64` へ拡張した
/// `checked_mul`／`checked_add` で行う。巨大な block 値を渡されても
/// panic せず `Err` を返す（本番経路で `unwrap`/`expect` を使わない
/// 契約。`.claude/rules/coding-rust.md`）。
pub fn derive_pipeline_stages(
    block_m: NonZeroU32,
    block_n: NonZeroU32,
    block_k: NonZeroU32,
    bytes_per_element: NonZeroU32,
    smem_budget_bytes: u64,
) -> Result<NonZeroU32, CudaError> {
    let (bm, bn, bk, bpe) = (
        u64::from(block_m.get()),
        u64::from(block_n.get()),
        u64::from(block_k.get()),
        u64::from(bytes_per_element.get()),
    );

    let a_tile_elems = bm
        .checked_mul(bk)
        .ok_or_else(|| overflow_err("block_m * block_k"))?;
    let b_tile_elems = bk
        .checked_mul(bn)
        .ok_or_else(|| overflow_err("block_k * block_n"))?;
    let tile_elems = a_tile_elems
        .checked_add(b_tile_elems)
        .ok_or_else(|| overflow_err("block_m*block_k + block_k*block_n"))?;
    let smem_per_stage = tile_elems
        .checked_mul(bpe)
        .ok_or_else(|| overflow_err("tile_elems * bytes_per_element"))?;

    if smem_per_stage == 0 {
        return Err(CudaError::InvalidKernelDescriptor {
            detail: "derive_pipeline_stages: smem_per_stage must be non-zero \
                     (block_m/block_n/block_k/bytes_per_element are all \
                     NonZeroU32, so this indicates an internal invariant \
                     violation)"
                .to_string(),
        });
    }

    let derived_raw = smem_budget_bytes / smem_per_stage;
    let derived = derived_raw.min(u64::from(MAX_PIPELINE_STAGES));

    // DeepGEMM 由来の最小段数要求: 小タイル（BM*BN < 128*192）はレジスタ
    // 圧が相対的に軽いぶんレイテンシ隠蔽により多くの段数を要求し（4 段）、
    // それ以外は 3 段を要求する（`docs/spec` には記載のない実装判断で
    // あり、根拠は本関数冒頭ドキュメンテーションコメントの参照実装）。
    let block_area = bm
        .checked_mul(bn)
        .ok_or_else(|| overflow_err("block_m * block_n"))?;
    let min_required: u64 = if block_area < 128 * 192 { 4 } else { 3 };

    if derived < min_required {
        return Err(CudaError::InvalidKernelDescriptor {
            detail: format!(
                "derive_pipeline_stages: derived stage count {derived} \
                 (smem_budget_bytes={smem_budget_bytes}, \
                 smem_per_stage={smem_per_stage}) is below the DeepGEMM-derived \
                 minimum requirement {min_required} (block_m={block_m}, \
                 block_n={block_n}, tile area={block_area})",
                block_m = block_m.get(),
                block_n = block_n.get(),
            ),
        });
    }

    // `derived >= min_required >= 3 > 0` かつ `u32` 範囲内（`MAX_PIPELINE_STAGES`
    // による上限クランプ済み）であるため `NonZeroU32::new` は必ず `Some` を
    // 返すが、`unwrap`/`expect` を使わず型付きエラーへ倒す
    // （`.claude/rules/coding-rust.md`）。
    let derived_u32 = u32::try_from(derived).map_err(|_| overflow_err("derived stage count"))?;
    NonZeroU32::new(derived_u32).ok_or_else(|| CudaError::InvalidKernelDescriptor {
        detail: format!(
            "derive_pipeline_stages: internal invariant violation, derived \
             stage count {derived_u32} was zero despite min_required={min_required}"
        ),
    })
}

/// [`derive_pipeline_stages`] のオーバーフロー検出箇所を集約するヘルパ。
/// 巨大な block 寸法が渡された場合に `u64` 中間値でも表現できない旨を
/// `CudaError::InvalidKernelDescriptor` として返す（panic 経路を作らない）。
fn overflow_err(step: &str) -> CudaError {
    CudaError::InvalidKernelDescriptor {
        detail: format!("derive_pipeline_stages: overflow computing {step}"),
    }
}

/// `src` を `arch`（`compute_XY` 形式。`CudaDevice::arch()` が返す値）
/// 向けに NVRTC でコンパイルし `Ptx` を返す。
///
/// 手順:
/// 1. `cudarc::nvrtc::sys::is_culib_present()` で `libnvrtc` の存在を
///    確認する（driver 側と同様、NVRTC 側にも `dlopen` 失敗時の panic
///    経路があるため。cudarc-0.19.8/src/nvrtc/sys/mod.rs:529）。
///    不在なら `CudaError::NvrtcUnavailable` を返す。
/// 2. include_paths なしでコンパイルを試みる。
/// 3. 失敗した場合のみ、`CUDA_INCLUDE_PATH` 環境変数または既知の候補
///    パス（CUDA 13.0 標準インストール先）で順に再試行する
///    （`cuda_fp16.h` 等が NVRTC 組み込みで解決できない環境向け。
///    PoC-v2-3 の 2 段構えを踏襲）。
///
/// `fmad`（NVRTC の FMA 契約フラグ）は明示せず NVRTC 既定のまま扱う。
/// CPU 参照実装（`f32::mul_add`）との丸め方針統一（PoC-v2-5 の K=4096
/// ストレスケースで実測確認済み。`.claude/rules/coding-rust.md`）は
/// この既定 FMA 契約が前提であり、呼び出し元が独自に上書きしないこと。
pub fn compile_ptx(src: &str, arch: &str) -> Result<Ptx, CudaError> {
    // SAFETY: `cudarc::nvrtc::sys::is_culib_present()` は `libnvrtc` の
    // 存在確認のみを行う `dlopen` ベースのプローブで、引数を取らず
    // ポインタ・共有可変状態も扱わない（cudarc 0.19.8 に safe wrapper が
    // ないため FFI 境界の `unsafe` を必要最小限で使う。`.claude/rules/
    // security.md`）。`nvrtc_version()` と同じく、`libnvrtc` 不在時に
    // dynamic-loading 側の panic 経路を踏まないための事前プローブとして
    // 呼び出し本体（NVRTC コンパイル）の前に行う。
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(CudaError::NvrtcUnavailable {
            detail: "libnvrtc dynamic library not found (dlopen failed); \
                     CUDA toolkit is not installed or not on the library search path"
                .to_string(),
        });
    }

    // `CompileOptions::arch` は `Option<&'static str>`（cudarc 0.19.8
    // src/nvrtc/safe.rs:240）のため呼び出しごとの `String` をそのまま
    // 渡せず `Box::leak` が必要になる。デバイス初期化・カーネル
    // コンパイルは #33/#34 でも通常デバイスあたり定数回（カーネル種別数）
    // に限られるため、リーク量は限定的であることをここに契約として
    // 明記する（無制限ループでの呼び出しは避けること）。
    let arch_static: &'static str = Box::leak(arch.to_string().into_boxed_str());
    let base = CompileOptions {
        arch: Some(arch_static),
        ..Default::default()
    };

    if let Ok(ptx) = compile_ptx_with_opts(src, base.clone()) {
        return Ok(ptx);
    }

    let candidates = [
        std::env::var("CUDA_INCLUDE_PATH").ok(),
        Some("/usr/local/cuda/include".to_string()),
        Some("/usr/local/cuda-13.0/targets/sbsa-linux/include".to_string()),
        Some("/usr/local/cuda-13.0/targets/x86_64-linux/include".to_string()),
    ];
    for path in candidates.into_iter().flatten() {
        let opts = CompileOptions {
            include_paths: vec![path],
            ..base.clone()
        };
        if let Ok(ptx) = compile_ptx_with_opts(src, opts) {
            return Ok(ptx);
        }
    }

    // 最終的にすべて失敗した場合は、include_paths なしの試行のエラーを
    // 呼び出し元に返す（フォールバック群のどれが失敗したかより、
    // 素の失敗理由の方が診断に有用。PoC-v2-3 と同じ判断）。
    compile_ptx_with_opts(src, base).map_err(CudaError::from)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use tensor_core::dispatch::{DType, GemmShape};

    use super::*;

    fn sample_descriptor() -> CudaKernelDescriptor {
        CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            2,
            DType::F32,
        )
        .expect("valid descriptor parameters must not fail")
    }

    fn sample_key() -> CudaKernelCacheKey {
        CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        )
    }

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    // 受け入れ基準 1: 未確定パラメータ（ゼロ値）のまま構築できず、
    // panic ではなく型付きエラーを返すこと。
    #[test]
    fn new_rejects_zero_block_m() {
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            0,
            64,
            32,
            2,
            DType::F32,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_stages() {
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            32,
            0,
            DType::F32,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    // codex-review 指摘（PR #641・P2）: `new_rejects_zero_block_m` /
    // `new_rejects_zero_stages` のみでは `block_n`・`block_k` の
    // ゼロ値拒否が未検証だった。`non_zero` ヘルパはフィールドごとに
    // 独立して呼ばれるため、各フィールドを個別に検証する。
    #[test]
    fn new_rejects_zero_block_n() {
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            0,
            32,
            2,
            DType::F32,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    #[test]
    fn new_rejects_zero_block_k() {
        let result = CudaKernelDescriptor::new(
            "wmma_tf32_f32",
            GemmShape::new(4096, 4096, 4096),
            64,
            64,
            0,
            2,
            DType::F32,
        );
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    // codex-review 指摘（PR #641・P2）: `kernel_name: &'static str` は
    // 型だけでは `"../escape"` 等のパス走査文字列を排除できない。C-2
    // （#506）のディレクトリ命名（`kernel.<name>.<hash>`）で使われる前に
    // ここで拒否し、後続タスクが「検証済み文字列」を前提にできるように
    // する。
    #[test]
    fn new_rejects_path_traversal_kernel_name() {
        for bad_name in ["../escape", "a/b", "a\\b", "..", ""] {
            let result = CudaKernelDescriptor::new(
                bad_name,
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
            );
            assert!(
                matches!(result, Err(CudaError::InvalidKernelDescriptor { .. })),
                "kernel_name {bad_name:?} must be rejected"
            );
        }
    }

    // 受け入れ基準: 同一パラメータから構築した 2 キーは `==` かつ
    // 同一ハッシュになること（`HashMap` キーとして使うための前提）。
    #[test]
    fn identical_parameters_produce_equal_keys_and_hashes() {
        let a = sample_key();
        let b = sample_key();
        assert_eq!(a, b);
        assert_eq!(hash_of(&a), hash_of(&b));
    }

    // 受け入れ基準: shape・BM/BN/BK・stages・dtype・compute capability・
    // NVRTC バージョン・compile_flags のいずれか 1 つを変えると不一致に
    // なること（DeepGEMM のアーキ違い＝別エントリの性質を各フィールドへ
    // 一般化して検証する）。
    #[test]
    fn changing_any_field_produces_distinct_key() {
        let base = sample_key();

        let different_shape = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32",
                GemmShape::new(2048, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_shape);

        let different_block_m = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                32,
                64,
                32,
                2,
                DType::F32,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_block_m);

        // codex-review 指摘（PR #641・P2）: `block_n`・`block_k`・
        // `stages`・`kernel_name` は `different_block_m` までのケースでは
        // 未検証だった。各フィールドを単独で変更し、`Hash`/`Eq` の
        // derive がフィールド漏れなく全メンバを含んでいることを検証する。
        let different_block_n = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                32,
                32,
                2,
                DType::F32,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_block_n);

        let different_block_k = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                16,
                2,
                DType::F32,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_block_k);

        let different_stages = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                3,
                DType::F32,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_stages);

        let different_kernel_name = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32_v2",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F32,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_kernel_name);

        // `kernel_name` は固定したまま `dtype` のみ変更する。
        // `kernel_name` を同時に変えると、万一 `dtype` が `Hash`/`Eq`
        // から欠落しても `kernel_name` 側の差分だけで
        // `assert_ne!` が通ってしまい検出できない
        // （codex-review 指摘・PR #641・P2）。
        let different_dtype = CudaKernelCacheKey::new(
            CudaKernelDescriptor::new(
                "wmma_tf32_f32",
                GemmShape::new(4096, 4096, 4096),
                64,
                64,
                32,
                2,
                DType::F16,
            )
            .unwrap(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_dtype);

        // compute capability 違い（DeepGEMM のアーキ違い＝別エントリの
        // 性質そのもの）。
        let different_cc = CudaKernelCacheKey::new(
            sample_descriptor(),
            (9, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_cc);

        let different_nvrtc_version = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (13, 0),
            vec!["--gpu-architecture=compute_80".to_string()],
        );
        assert_ne!(base, different_nvrtc_version);

        let different_flags = CudaKernelCacheKey::new(
            sample_descriptor(),
            (8, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_90".to_string()],
        );
        assert_ne!(base, different_flags);
    }

    // 受け入れ基準: `HashMap` でのヒット/ミス動作のスモークテスト。
    #[test]
    fn cache_key_works_as_hashmap_key() {
        let mut cache: HashMap<CudaKernelCacheKey, &str> = HashMap::new();
        cache.insert(sample_key(), "compiled-ptx-placeholder");

        assert_eq!(cache.get(&sample_key()), Some(&"compiled-ptx-placeholder"));

        let miss_key = CudaKernelCacheKey::new(
            sample_descriptor(),
            (9, 0),
            (12, 9),
            vec!["--gpu-architecture=compute_90".to_string()],
        );
        assert_eq!(cache.get(&miss_key), None);
    }

    // 実機（DGX Spark GB10 等）依存: `libnvrtc` の実バージョンを問い合わせる。
    // CUDA toolkit 非搭載環境（通常 CI）では `NvrtcUnavailable` を返す想定
    // であり panic しないため `#[ignore]` で分離する（`make
    // test-ignored-cuda` 導線。`.claude/rules/coding-rust.md` 実機分離規約）。
    #[test]
    #[ignore]
    fn nvrtc_version_returns_ok_on_real_device() {
        let version = nvrtc_version().expect("NVRTC must be present on the real-device runner");
        assert!(version.0 > 0, "NVRTC major version must be positive");
    }

    // `derive_pipeline_stages`（C-8・#521）のユニットテスト群。GPU 不要
    // （純関数）のため通常 CI で実行される（`#[ignore]` なし）。

    fn nz(value: u32) -> NonZeroU32 {
        NonZeroU32::new(value).expect("test helper: value must be non-zero")
    }

    // 現行タイル構成（#494。`kernels_mma.rs::MMA_BM`=64/`MMA_BN`=128/
    // `MMA_BK`=32・f16=2 バイト）で、静的 SMEM 上限 49,152 バイト
    // （48KiB）を予算として渡すと 4 段が導出されること（実装計画 §3.2
    // の検算: 49152 / 12288 = 4、タイル面積 64*128=8192 < 128*192=24576
    // のため min_required=4 も満たす）。
    #[test]
    fn derive_pipeline_stages_matches_current_tile_configuration() {
        let stages = derive_pipeline_stages(nz(64), nz(128), nz(32), nz(2), 49_152)
            .expect("current tile configuration must derive a valid stage count");
        assert_eq!(stages.get(), 4);
    }

    // 予算が 1 段分にも満たない場合は `min_required` 未達で `Err`。
    #[test]
    fn derive_pipeline_stages_rejects_insufficient_budget() {
        let result = derive_pipeline_stages(nz(64), nz(128), nz(32), nz(2), 1_024);
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }

    // 予算が極端に大きい場合でも `MAX_PIPELINE_STAGES`（16）でクランプ
    // されること。
    #[test]
    fn derive_pipeline_stages_clamps_at_max_pipeline_stages() {
        let stages = derive_pipeline_stages(nz(64), nz(128), nz(32), nz(2), u64::MAX)
            .expect("huge budget must still derive a valid (clamped) stage count");
        assert_eq!(stages.get(), MAX_PIPELINE_STAGES);
    }

    // 小タイル（block_m * block_n < 128*192）は最小 4 段要求。3 段相当の
    // 予算では `Err` になり、4 段分の予算でちょうど成立することを確認する。
    #[test]
    fn derive_pipeline_stages_small_tile_requires_four_stages() {
        // block_m=32, block_n=64 → 面積 2048 < 24576（小タイル）。
        // 1 段あたり (32*32 + 32*64) * 2 = 6144 バイト。3 段分の予算
        // （18432）では 3 段しか導出できず min_required=4 未達で Err。
        let insufficient = derive_pipeline_stages(nz(32), nz(64), nz(32), nz(2), 18_432);
        assert!(matches!(
            insufficient,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));

        // 4 段分ちょうど（24576）なら成立する。
        let stages = derive_pipeline_stages(nz(32), nz(64), nz(32), nz(2), 24_576)
            .expect("4-stage budget must satisfy the small-tile minimum requirement");
        assert_eq!(stages.get(), 4);
    }

    // 大タイル（block_m * block_n >= 128*192）は最小 3 段要求。
    #[test]
    fn derive_pipeline_stages_large_tile_requires_three_stages() {
        // block_m=128, block_n=192 → 面積 24576（境界値。`< 128*192` は
        // false なので大タイル扱い）。1 段あたり (128*32 + 32*192) * 2 =
        // 20480 バイト。3 段分ちょうど（61440）で成立する。
        let stages = derive_pipeline_stages(nz(128), nz(192), nz(32), nz(2), 61_440)
            .expect("3-stage budget must satisfy the large-tile minimum requirement");
        assert_eq!(stages.get(), 3);
    }

    // オーバーフロー安全性: 巨大な block 値を渡しても panic せず `Err`
    // を返す（`u64` 中間値でも `checked_mul` が `None` を返すケース）。
    #[test]
    fn derive_pipeline_stages_rejects_overflowing_block_dimensions() {
        let result =
            derive_pipeline_stages(nz(u32::MAX), nz(u32::MAX), nz(u32::MAX), nz(4), 49_152);
        assert!(matches!(
            result,
            Err(CudaError::InvalidKernelDescriptor { .. })
        ));
    }
}
