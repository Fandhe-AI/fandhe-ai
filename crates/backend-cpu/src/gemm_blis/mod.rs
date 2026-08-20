//! BLIS/GotoBLAS2 5-loop model（jc→pc→ic→jr→ir）による GEMM（TASK-1.6f・#184）。
//!
//! [`crate::gemm`] の `gemm_naive`／`gemm_blocked`／`gemm_parallel`（TASK-1.6a）
//! は自動ベクトル化頼みのスカラーループで、PoC-v2-1 実測では対 PyTorch CPU 比
//! 5.3%（M=N=K=2048/4096 最小値）に留まり REQ-8 の CPU 最適化後下限（20%）に
//! 届かない。本モジュールは `std::arch` intrinsics（NEON／AVX2+FMA／AVX-512F）
//! による マイクロカーネル・A/B packing（[`pack`]）・キャッシュ階層ブロッキング
//! （MC/KC/NC）を実装し、性能向上を狙う。
//!
//! ## 実行時 ISA ディスパッチ（#185・TASK-1.6g）
//!
//! TASK-1.6f まではマイクロカーネルの ISA 選択がコンパイル時 cfg のみ
//! だったため、x86_64 の既定ビルド（`RUSTFLAGS` なし）では実行 CPU が
//! AVX2/AVX-512 を持っていてもスカラーへ落ちていた。本モジュールの
//! 公開入口（[`gemm_blis`]／[`gemm_blis_parallel`]）は `dispatch_region`
//! で 1 回だけ ISA トークンの検出・選択を行い、[`gemm_blis_region`] を
//! モノモーフィック化されたジェネリック関数として呼ぶ（トークン型による
//! 健全な dispatch の設計は [`microkernel`] モジュールドキュメント参照）。
//!
//! ## 公開 API 非破壊（既存 3 関数は変更しない）
//!
//! [`crate::gemm::gemm_naive`]／`gemm_blocked`／`gemm_parallel` は #24
//! （TASK-1.6d・PoC-v2-1 比 3 段階性能確認）の参照点として変更しない
//! （公開 API 非破壊はガードレール条件・`.claude/rules/security.md`）。
//! [`gemm_blis`]／[`gemm_blis_parallel`] のシグネチャも #185 で変更しない
//! （dispatch はこれら関数の内部実装としてのみ追加）。
//!
//! ## bit 完全一致契約（REQ-2）
//!
//! [`microkernel`] の各カーネルは C 要素ごとの累積を p 昇順の FMA 連鎖で
//! 行い、レーン間縮約（split-k 等）を一切行わない設計とすることで、
//! `gemm_naive` と bit 完全一致が成立する（`tests/gemm_blis_parity.rs`）。
//! 累積順序を変える最適化（split-k・ゼロ初期化してからの後加算方式）は
//! 本契約を壊すため、将来追加する場合は数値一致テストの契約変更として
//! ユーザー承認事項である。実行時 ISA ディスパッチ導入後もどの ISA が
//! 選ばれても結果は bit 完全一致するため、既存 parity テストはそのまま
//! 実行時 dispatch 経路の検証を兼ねる。
//!
//! ## 境界検査（REQ-8）
//!
//! 公開入口は [`crate::gemm::validate_dims`]（`checked_mul` によるオーバー
//! フロー検査・スライス長検査）を再利用する。packing・端タイルの C
//! 書き戻しは安全な slice 操作で行い、intrinsics のロード／ストアは
//! マイクロカーネル関数入口の `assert!` で長さを検査した直後の最小
//! `unsafe` ブロックに限定する（[`microkernel::neon`]／[`microkernel::avx2`]／
//! [`microkernel::avx512`] 参照）。dispatch 導入を理由とした境界検査の
//! 省略は行わない。

// `cache_params`／`partition` は #753 の本番未結線スコープ（下記
// [`gemm_blis_parallel_2d_with_blocks`] ドキュメント参照）のため、
// 呼び出し元がテスト専用パラメータ化入口・実機 A/B ハーネス
// （いずれも `#[cfg(test)]`）に限られる。`gemm_blis_with_kernel_and_blocks`・
// `gemm_blis_parallel_with_blocks`（#564）・`gemm_blis_shared_b_region`・
// `dispatch_shared_b`（#750）と同じ「本番未結線の間はモジュール自体を
// `#[cfg(test)]` にする」既存パターンを踏襲し、`cargo build`（`cfg(test)`
// 無効時）での dead_code 検出を構造的に避ける（`#[allow(dead_code)]` に
// よる黙らせは行わない。`.claude/rules/coding-rust.md`）。sysctl FFI
// （`cache_params::sysctl_ffi`。macOS 限定）の型・借用検査は `cargo test`
// （`rust-ci` の test ジョブ）が `cfg(test)` を有効化した状態でコンパイル
// する際に行われる（同ジョブは Linux ホストのため `cfg(target_os =
// "macos")` 自体は無効化されコンパイル対象に含まれない。macOS 実機での
// 検証手順は `docs/perf/cpu-gemm-runtime-cache-detect.md` 参照）。
#[cfg(test)]
mod cache_params;
pub mod microkernel;
mod pack;
#[cfg(test)]
mod partition;

use std::ops::Range;

use tensor_core::Activation;

use crate::gemm::{BlockSizes, GemmError, validate_dims};
#[cfg(not(target_arch = "x86_64"))]
use microkernel::Isa;
use microkernel::Microkernel;
// `mod tests`（`use super::*;` で全取り込み）内のテストが ISA 分岐を問わず
// `ScalarKernel` を直接参照するため、cfg 条件に `test` を加える
// （aarch64 の非テストビルドではスカラーカーネル型を経路上使わないが、
// テストバイナリのコンパイルには必要。#185 レビュー指摘）。
#[cfg(any(not(target_arch = "aarch64"), test))]
use microkernel::ScalarKernel;
use pack::{APackTile, BPackTile, pack_a, pack_b};
use rayon::prelude::*;

/// 端タイル専用の C スタックバッファ最大要素数（`MR * NR` の全 ISA 中の
/// 最大値。AVX-512 の 8×32=256 が最大。ジェネリック const 式は stable
/// Rust では使えないため固定長で確保し、各カーネルモジュールの
/// `const _: () = assert!(MR * NR <= 256);` でこの上限を守ることを
/// コンパイル時に検査する）。
///
/// #557 により、完全タイル（`mr_eff == MR && nr_eff == NR`）は C の実
/// バッファへ [`Microkernel::run_with_ldc`] の `ldc` 契約経由で直接ロード/ストア
/// するため、本バッファは境界検査（REQ-8）が必要な**端タイル専用**に
/// 用途が絞られた（参照実装 matrixmultiply の「完全タイルは直接、端
/// タイルのみマスク付きバッファ」という設計に倣う。以前は全タイルが
/// 本バッファ経由でタイルあたり 2×MR×NR 要素のコピー往復を余分に
/// 払っていた）。
const MAX_TILE: usize = 256;

/// キャッシュブロッキングの行方向ブロックサイズ（A のパネル高さ）。
///
/// [`crate::gemm`] の既存定数（PoC-v2-1 実測環境で選定）を起点値として
/// 踏襲する。マイクロカーネルの MR/NR は ISA ごとに異なる（[`microkernel`]
/// 参照）ため、本モジュール独自の定数として持つ（`gemm` モジュールの
/// MC/KC/NC とは独立にチューニング可能にする）。
///
/// **対象実機・再選定状況（#564・#749・#753 へ引き継ぎ）**: 本 3 定数の
/// チューニング対象実機は **Apple M4 Max**（firestorm 系。
/// `docs/perf/gemm-optimization-baseline.md` §3・イシュー #481 で確定）。
/// BLIS/OpenBLAS の aarch64 向け参照実装値（firestorm: MC=480/KC=4096/
/// NC=9600 等）近傍を含む実機スイープは 2026-08-19 に M4 Max で実施済みで、
/// 詳細実測値・採否根拠は `docs/perf/cpu-gemm-blocking-sweep.md` §7 に
/// 記録する（#749）。MC・KC は単独拡大が全サイズで劣化したため PoC-v2-1
/// 値のまま維持する。
///
/// **NC の n 依存拡大（NC=9600・n>=4096 で約 9.9% 改善）は本 PR 時点で
/// 未有効化（PR #766・codex-review 再指摘）**: 実測を行った M4 Max
/// 個体の正確な `hw.model` 識別子が実測セッション終了時点で記録されて
/// おらず本 PR 時点では復元不能（`docs/perf/cpu-gemm-blocking-sweep.md`
/// の「実測値の捏造・placeholder 値での完了扱いは行わない」方針に従い、
/// `Mac16,` prefix 等の広い一致条件へ後退させて有効化することもしない）。
/// 実機固有値を検証していないターゲット・機種への適用は
/// `.claude/rules/coding-rust.md` の方針に反するため、識別子が判明する
/// までは常時 `default_blocks()`（固定 NC=512）を返す（#753〈sysctl
/// ベース MC/KC/NC 動的算出〉で機種識別を含めて再検討する）。
const MC: usize = 128;
/// 縮約次元（K）のブロックサイズ。B パネル（KC×NC×4B）が L1/L2 に収まる値。
const KC: usize = 256;
/// 列方向ブロックサイズ（B のパネル幅）。本 PR 時点の唯一の適用値
/// （上記 NC 拡大の未有効化理由を参照）。
const NC: usize = 512;

/// [`gemm_blis`]／`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`
/// （本番 3 公開関数）が使う既定ブロックサイズ（上記 `MC`/`KC`/`NC`
/// 定数と同一値）。[`crate::gemm::BlockSizes`] 型を再利用してパラメータ化
/// した理由は [`dispatch_region`] のドキュメント参照（#564・§3.1
/// `crate::gemm::gemm_blocked` 向け `BlockSizes` 導入〈#24〉と同じ前例
/// 踏襲）。値自体は `gemm` モジュールの既定値と独立にチューニング可能な
/// ままにするため、`BlockSizes::poc_v2_1_default()` を直接使わずここで
/// 本モジュール専用の定数から構築する。
const fn default_blocks() -> BlockSizes {
    BlockSizes {
        mc: MC,
        kc: KC,
        nc: NC,
    }
}

/// panel packing バッファ（A/B 各 1 本）を gemm 呼び出し単位で 1 回だけ
/// 確保し、5-loop 内の全 jc×pc×ic ブロックで使い回すための保持構造体
/// （#556。matrixmultiply 等参照実装の「gemm 呼び出しあたり 1 回確保＋
/// オフセット分割」方針に倣う）。
///
/// [`dispatch_region`] がカーネル型（`K::MR`／`K::NR`）確定直後に 1 回
/// 構築し、[`gemm_blis_region`] へ可変参照で渡す。直列経路（`gemm_blis`）
/// は gemm 呼び出しあたり 1 組、並列経路（`gemm_blis_parallel`／
/// `gemm_blis_bias_act_parallel`）は rayon の行パネルタスクごとに
/// `dispatch_region` が呼ばれるため**タスクごとに 1 組を所有**する
/// （`Vec` の所有権がタスクローカルに閉じるため、事前一括確保＋
/// `par_chunks_mut` とのオフセット分割を要さずコンパイル時にデータ競合
/// が排除される。B packing のスレッド間重複計算〈同じ B 列ブロックを
/// 複数タスクが個別に pack し直す〉は本変更のスコープ外＝既存挙動のまま。
/// 将来の並列分割再構成候補として PR 本文に記載する）。
struct PanelBuffers {
    /// B パネル用バッファ（全 jc×pc ブロック中で最大の nc_len×kc_len 組が
    /// 必要とする要素数で確保。ループ内は `nr_blocks*kc_len*nr` 要素の
    /// 先頭サブスライスのみ使う）。
    b_panel: Vec<f32>,
    /// A パネル用バッファ（全 jc×pc×ic ブロック中で最大の mc_len×kc_len
    /// 組が必要とする要素数で確保。ループ内は `mr_blocks*kc_len*mr` 要素の
    /// 先頭サブスライスのみ使う）。
    a_panel: Vec<f32>,
}

impl PanelBuffers {
    /// `n`（C の列数）・`k_dim`（縮約次元）・`mc_total`（この呼び出しが
    /// 担当する行数。並列時はパネル 1 つぶん）から、[`gemm_blis_region`]
    /// の全反復を通じて必要になる最大バッファ長を 1 回で計算し確保する。
    ///
    /// 各反復の必要量は `min(blocks.nc, n).div_ceil(nr)*min(blocks.kc,k_dim)*nr`
    /// （B）／`min(blocks.mc, mc_total).div_ceil(mr)*min(blocks.kc,k_dim)*mr`
    /// （A）が常に上界になる（先頭ブロックが nc_len/mc_len/kc_len 最大で、
    /// 末尾ブロックはこれらが縮むのみ。§4.1 計画）。`blocks`（[`BlockSizes`]）
    /// はパラメータ化（#564）により実行時値になったが、乗算オーバーフローの
    /// 懸念はない: 呼び出し元（[`gemm_blis_with_kernel_and_blocks`]／
    /// [`gemm_blis_parallel_with_blocks`] 等のパラメータ化入口）が
    /// [`validate_dims`] を先に通しており、`m*k`／`k*n`／`m*n` が `usize`
    /// で非オーバーフローと確定済みの `n`／`k_dim`／`mc_total` に対して
    /// `blocks.{nc,kc,mc}` は常に `.min()` でクランプしてから乗算される
    /// （`blocks` 側にどれだけ大きな値〈firestorm 参照値 KC=4096/NC=9600
    /// 等〉を渡しても、実際の乗算対象は非オーバーフロー確定済みの dim
    /// 由来値に収まる）。本番 3 公開関数は [`default_blocks`]（コンパイル
    /// 時定数）のみを渡すためこの経路も従来通り安全。`n == 0`／
    /// `k_dim == 0`／`mc_total == 0` では長さ 0 のバッファになり、5-loop
    /// 自体が回らないためサブスライス取得も発生せず問題ない。
    fn new<K: Microkernel>(n: usize, k_dim: usize, mc_total: usize, blocks: BlockSizes) -> Self {
        let (b_len, a_len) = panel_capacity(n, k_dim, mc_total, K::MR, K::NR, blocks);
        PanelBuffers {
            b_panel: vec![0.0f32; b_len],
            a_panel: vec![0.0f32; a_len],
        }
    }
}

/// [`PanelBuffers::new`] の容量計算本体（B 長・A 長の順で返す）。`mr`／`nr`
/// を型パラメータでなく引数に取ることで、単体テストが
/// [`Microkernel`] 実装を経由せず MR/NR の任意の組（全 ISA カーネル定数
/// 相当）に対して総当たり検証できるようにしている（#556 テスト計画
/// §5-3）。`blocks`（[`BlockSizes`]）は #564 でパラメータ化した MC/KC/NC。
fn panel_capacity(
    n: usize,
    k_dim: usize,
    mc_total: usize,
    mr: usize,
    nr: usize,
    blocks: BlockSizes,
) -> (usize, usize) {
    let kc_len_max = blocks.kc.min(k_dim);
    let nc_len_max = blocks.nc.min(n);
    let mc_len_max = blocks.mc.min(mc_total);
    let nr_blocks_max = nc_len_max.div_ceil(nr);
    let mr_blocks_max = mc_len_max.div_ceil(mr);
    (
        nr_blocks_max * kc_len_max * nr,
        mr_blocks_max * kc_len_max * mr,
    )
}

/// 単一スレッドの BLIS 5-loop GEMM（jc→pc→ic→jr→ir）。
///
/// `gemm_blis_parallel` はこの関数の内部ロジック（[`gemm_blis_region`]）を
/// 行パネルごとに並列呼び出しすることで並列化する（`gemm_blocked`／
/// `gemm_parallel` と同じ構成。`crate::gemm` 参照）。
pub fn gemm_blis(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    dispatch_region(a, b, c, n, k, 0..m, default_blocks())
}

/// `gemm_blis` を `rayon` で行パネル並列化した版。
///
/// C を行方向にパネル分割し、各パネルを独立スレッドで [`gemm_blis_region`]
/// に渡す（`crate::gemm::gemm_parallel` と同じ並列化戦略。C の書き込み
/// 範囲がパネルごとに排他的なためデータ競合なし）。
pub fn gemm_blis_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;

    // n == 0 は shape として合法だが par_chunks_mut(panel_rows * n) は
    // チャンクサイズ 0 でパニックする（`crate::gemm::gemm_parallel` と
    // 同じ理由・同じ対処。3 実装間の parity 契約を保つため no-op で返す）。
    if n == 0 {
        return Ok(());
    }

    let num_threads = rayon::current_num_threads().max(1);
    let panel_rows = m.div_ceil(num_threads).max(1);
    // 呼び出しあたり 1 回だけ確定し、全 rayon 行パネルタスクへ同一値を
    // キャプチャして渡す（既定ブロックサイズは n に依存しないため
    // ループ外で確定させても意味は変わらないが、`dispatch_region`
    // 呼び出しごとの再計算を避ける従来方針を踏襲する）。
    let blocks = default_blocks();

    // B パネル共有経路（`dispatch_shared_b`。案 B・イシュー #750）は
    // 本番既定へは**採用しない**（`docs/cpu-gemm-b-packing-sharing.md`・
    // `docs/cpu-gemm-b-packing-sharing-decision.md` 参照）。受け入れ条件 2
    // （Apple M4 Max 実機実測での M=N=K=2048/4096 非劣化確認）を満たす
    // 前提条件が未充足のため、実装・bit 完全一致検証・単体/統合テスト
    // （`dispatch_shared_b`／`gemm_blis_shared_b_region`／
    // `gemm_blis_ic_loop`・`#[cfg(test)]` の
    // [`gemm_blis_parallel_with_blocks`] 経由）は残しつつ、本番公開入口
    // からの呼び出しは行わない（PR #758〈#740 mma_f16 swizzle〉の
    // 「実装は入れるが実機ゲート未通過のうちは本番結線しない」判断と
    // 同型。実機実測とユーザー承認を経た別 PR でのみ既定切替を検討する）。
    // 従来どおり行パネルごとに [`dispatch_region`] を独立呼び出しする
    // （B packing はタスクごとに個別実行＝共有化前の挙動）。
    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .try_for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            dispatch_region(a, b, c_chunk, n, k, row_start..row_end, blocks)
        })
}

/// `gemm_blis_parallel` に GEMM epilogue（bias 加算・activation）を融合した版
/// （TASK-12.1f・#203）。
///
/// 非融合実行（`gemm_blis_parallel` → `add`〈bias〉→ `relu` の 3 パス・
/// 中間 `Vec<f32>` 2 個割当）に対し、C を行パネル並列で計算した直後（各
/// パネルがまだキャッシュ熱いうち）に同じ `rayon` タスク内で epilogue を
/// 適用することで、C の再読み出しパス・中間バッファ割当を削減する
/// （CUTLASS 系実測で epilogue 融合が平均 1.38〜1.45 倍。動機はイシュー
/// #203。`docs/perf/cpu-gemm-epilogue-fusion.md` に本環境での実測を記録）。
///
/// `bias` は `Some(&[f32])` の場合 `n`（`B` の列数）と同じ長さが必須で、
/// 各行へ加算される（`docs/public-api-design.md` §4.2 のブロードキャスト
/// 規約と同じ「`[n]` を行方向へ複製」の意味論。`tensor_core::BackendOps::
/// gemm_bias_act` のデフォルト実装〈`add` の broadcast〉と等価）。長さ
/// 不一致は [`GemmError::BiasLenMismatch`]（カーネル本体アクセス前に
/// 検証。REQ-8・OWASP A03）。
///
/// # bit 完全一致契約
///
/// epilogue（bias 加算・activation）は要素ごとに独立な演算で、パネル間の
/// 演算順序（並列実行のタスク分割）に依存しない。したがって本関数の
/// 結果は「`gemm_blis_parallel` で C を計算した後、全体へ 1 回だけ
/// bias 加算・activation を適用した結果」と **bit 完全一致**する
/// （`tests/gemm_epilogue_parity.rs` で検証）。GEMM 本体の FMA 契約
/// （`f32::mul_add`）・累積順序は `gemm_blis_parallel` から変更しない。
///
/// `#[allow(clippy::too_many_arguments)]`: `gemm_blis_parallel`（`a`／`b`／
/// `c`／`m`／`n`／`k` の 6 引数）に epilogue パラメータ（`bias`／`act`）を
/// 追加した結果 8 引数になる。GEMM カーネル公開入口の既存慣例
/// （`crates/backend-cpu/src/gemm.rs` の同 attribute 使用箇所と同方針）に
/// 従い、構造体化はせず素朴な引数列のまま許容する。
#[allow(clippy::too_many_arguments)]
pub fn gemm_blis_bias_act_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    bias: Option<&[f32]>,
    act: Activation,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    if let Some(bias) = bias
        && bias.len() != n
    {
        return Err(GemmError::BiasLenMismatch {
            expected: n,
            actual: bias.len(),
        });
    }
    // `Activation` は未知 variant（`#[non_exhaustive]`）を早期に拒否する。
    // `apply_epilogue` 内で検証すると、`try_for_each` が行パネル並列の
    // 途中パネルまで epilogue を適用した後にエラー終了しうる（`c` が
    // 部分適用の不定状態で返る）。呼び出し前にここで検証し、GEMM 本体・
    // epilogue のいずれにも触れない状態でのみエラーを返す。
    if !matches!(act, Activation::None | Activation::Relu) {
        return Err(GemmError::UnsupportedActivation);
    }

    // n == 0 は shape として合法（`gemm_blis_parallel` と同じ理由で
    // no-op）。epilogue も対象要素が無いため何もすることがない。
    if n == 0 {
        return Ok(());
    }

    let num_threads = rayon::current_num_threads().max(1);
    let panel_rows = m.div_ceil(num_threads).max(1);
    // 呼び出しあたり 1 回だけ確定し、全 rayon 行パネルタスクへ同一値を
    // キャプチャして渡す（`gemm_blis_parallel` と同じ理由）。
    let blocks = default_blocks();

    // GEMM 本体は `gemm_blis_parallel` と同じ理由で B パネル共有経路
    // （`dispatch_shared_b`）を本番既定へ採用しない（上記
    // `gemm_blis_parallel` 実装コメント参照。イシュー #750・実機ゲート
    // 未通過）。従来どおり行パネルごとに `dispatch_region` を独立呼び出し
    // する。epilogue（bias 加算・activation）は要素ごとに独立な演算で
    // パネル分割順序に依存しないため（本関数冒頭のドキュメンテーション
    // コメント「bit 完全一致契約」参照）、GEMM 本体の完了後に `c` 全体へ
    // 1 回だけ適用する（設計 doc §C 案 B の選択肢 (a)。
    // `par_chunks_mut(panel_rows * n)` で行パネル並列に適用することで、
    // T=1（`panel_rows == m` で単一チャンク）では従来と同一の 1 パスに
    // なる）。
    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .try_for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            dispatch_region(a, b, c_chunk, n, k, row_start..row_end, blocks)?;
            apply_epilogue(c_chunk, n, bias, act)
        })
}

/// [`gemm_blis_bias_act_parallel`] の epilogue 適用部（bias 行ブロード
/// キャスト加算 → activation）。`c_panel` は行パネル 1 つ分（`rows * n`
/// 要素、`n` 単位の行区切り）を対象とし、境界検査は行・列とも `n`
/// 由来のスライス長で構造的に保証する（明示 `assert`／`unsafe` を要しない。
/// REQ-8）。
///
/// `Activation` は `#[non_exhaustive]`（`tensor_core::backend_ops`）のため
/// `_ =>` で未知 variant を静かに無視せず、[`GemmError::UnsupportedActivation`]
/// を返す（未対応 activation を無視して不正な結果を返す fail-open を避ける。
/// `tensor-core` と同一ワークスペースで管理されるため通常到達しないが、
/// variant 追加時に本関数の更新漏れがあれば早期に検出できる）。
fn apply_epilogue(
    c_panel: &mut [f32],
    n: usize,
    bias: Option<&[f32]>,
    act: Activation,
) -> Result<(), GemmError> {
    if let Some(bias) = bias {
        for row in c_panel.chunks_mut(n) {
            for (x, b) in row.iter_mut().zip(bias.iter()) {
                *x += *b;
            }
        }
    }
    match act {
        Activation::None => {}
        Activation::Relu => {
            for x in c_panel.iter_mut() {
                *x = x.max(0.0);
            }
        }
        _ => return Err(GemmError::UnsupportedActivation),
    }
    Ok(())
}

/// 検出済みトークンを優先順位（Avx512 > Avx2 > Scalar）で直接 `try_new`
/// し、最初に構築できたトークンで [`gemm_blis_region`] を呼ぶ（実行時
/// dispatch の唯一の入口。`gemm_blis`／`gemm_blis_parallel` の両方から
/// 呼ばれる）。[`microkernel::Isa::detect`]（[`microkernel::select_isa`]
/// と同じ優先順位ロジック）を経由せず `try_new` の成否だけで分岐する
/// ことで、本番経路に `unwrap`／`expect`／`unreachable!` を一切置かない
/// （`.claude/rules/coding-rust.md` の「本番経路で unwrap()/expect() を
/// 使わない」規約。`Isa::detect` 自体は単体テスト・将来のテレメトリ用の
/// introspection API として残す）。環境変数等による dispatch 上書きは
/// 設けない（OWASP A03・`.claude/rules/security.md`）。
#[cfg(target_arch = "x86_64")]
fn dispatch_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    // AVX-512 経路は `avx512_stable` cfg（`backend-cpu` クレートルートの
    // `build.rs` が AVX-512F intrinsics のコンパイル可否を実測して発行。
    // [`microkernel::avx512`] モジュールドキュメント参照）が立っている
    // 場合のみ試す。立っていない rustc では
    // `Avx512Kernel` 自体がコンパイル対象外のため、AVX2 から直接試行する
    // （実行 CPU が AVX-512F を持っていてもコンパイラの stable 化状況に
    // 応じて AVX2 へフォールバックする。数値一致は ISA 間 bit 完全一致
    // 契約のため結果には影響しない）。
    let mc_total = rows.end - rows.start;
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        let mut bufs = PanelBuffers::new::<microkernel::Avx512Kernel>(n, k, mc_total, blocks);
        return gemm_blis_region(kernel, a, b, c, n, k, rows, &mut bufs, blocks);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        let mut bufs = PanelBuffers::new::<microkernel::Avx2Kernel>(n, k, mc_total, blocks);
        gemm_blis_region(kernel, a, b, c, n, k, rows, &mut bufs, blocks)
    } else {
        let mut bufs = PanelBuffers::new::<ScalarKernel>(n, k, mc_total, blocks);
        gemm_blis_region(ScalarKernel, a, b, c, n, k, rows, &mut bufs, blocks)
    }
}

/// aarch64 版 [`dispatch_region`]。NEON は baseline ISA のため
/// [`microkernel::Isa::detect`] は常に `Isa::Neon` を返す（実行時検出不要。
/// [`microkernel`] モジュールドキュメント参照）。
#[cfg(target_arch = "aarch64")]
fn dispatch_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    let mc_total = rows.end - rows.start;
    let mut bufs = PanelBuffers::new::<microkernel::NeonKernel>(n, k, mc_total, blocks);
    gemm_blis_region(
        microkernel::NeonKernel,
        a,
        b,
        c,
        n,
        k,
        rows,
        &mut bufs,
        blocks,
    )
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_region`]。実行時検出対象の
/// ISA を持たないため常に [`ScalarKernel`] を使う。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dispatch_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    let mc_total = rows.end - rows.start;
    let mut bufs = PanelBuffers::new::<ScalarKernel>(n, k, mc_total, blocks);
    gemm_blis_region(ScalarKernel, a, b, c, n, k, rows, &mut bufs, blocks)
}

/// `gemm_blis`／`gemm_blis_parallel` 共通の 5-loop 本体。`c` はパネル
/// 先頭が `row_start` 行目に対応するスライス（`crate::gemm::gemm_blocked_region`
/// と同じ相対オフセット規約）。引数検証は呼び出し元の公開入口で完了済み
/// の前提。
///
/// ループ順は jc（列ブロック）→ pc（K ブロック）→ ic（行ブロック）→
/// jr（NR 単位の列パネル）→ ir（MR 単位の行パネル）。B パネルは pc/jc
/// ブロックごとに 1 回だけ packing して ic ループ全体で再利用し、A パネルは
/// ic ブロックごとに 1 回だけ packing して jr ループ全体で再利用する
/// （BLIS/GotoBLAS2 の packing 再利用による bandwidth 削減。PoC-v2-1
/// README「設計判断」節と同じ狙い）。
///
/// `K::MR`／`K::NR`（[`Microkernel`] トレイトの定数）でタイル形状を決め、
/// `kernel.run_with_ldc(...)` で累積計算を呼ぶ（#185 でジェネリック化。ISA ごとに
/// 呼び出し元でモノモーフィックに特殊化されるため、この関数自体に
/// `unsafe` は現れない）。C タイルは [`MAX_TILE`]（全 ISA 中の MR*NR 最大値）
/// 固定長スタックバッファを確保し、`K::MR * K::NR` ぶんだけスライスして
/// 使う（ジェネリック const 式は stable Rust で使えないための対処。各
/// カーネルモジュールの `const _: () = assert!(MR * NR <= MAX_TILE);` が
/// この前提をコンパイル時に検査する）。
///
/// `bufs`（[`PanelBuffers`]）は呼び出し元（[`dispatch_region`]）が
/// カーネル型確定直後に 1 回確保した A/B panel バッファで、jc×pc×ic の
/// 全反復を通じて使い回す（#556。以前は各反復で `vec![...]` を都度確保
/// していた。M=N=K=4096・MC=128/KC=256/NC=512 換算で B は 128 回・A は
/// 4,096 回のヒープ確保がこの 1 回確保へ削減される。実測は
/// `docs/perf/cpu-gemm-packing-buffer-reuse.md`）。各反復は `bufs` の
/// 先頭から必要長ぶんのサブスライスを再借用するのみで、`pack_a`／
/// `pack_b` が呼び出しのたびに有効レーンを全上書き（端タイルは
/// `dst.fill(0.0)` してから書く。`pack.rs` 参照）するため前反復の残留値
/// には依存せず、bit 完全一致契約（REQ-2）・FMA 契約・累積順序は一切
/// 変更しない。
///
/// `#[allow(clippy::too_many_arguments)]`: `bufs`／`blocks` 追加により
/// 9 引数になる。本ファイル内の既存慣例（[`gemm_blis_bias_act_parallel`] の
/// 同 attribute 使用箇所と同方針）に従い、構造体化はせず素朴な引数列の
/// まま許容する。
///
/// `blocks`（[`BlockSizes`]）は #564 で MC/KC/NC をパラメータ化したもの。
/// K 方向の加算順序・C タイルのロード／書き戻し構造は `blocks` の値に
/// 依らず不変のため、任意の `blocks` で [`crate::gemm::gemm_naive`] との
/// bit 完全一致契約（REQ-2）が成立する（`gemm_blis_with_kernel_and_blocks`
/// のパリティテストで直接検証）。
/// ## エラー伝播（#691 レビュー P1 再指摘への対応）
///
/// 内部の `kernel.run_with_ldc(...)` 呼び出しは構築上（組み込みカーネルの
/// `MR`／`NR` はコンパイル時定数で 1 以上、完全タイル・端タイルいずれも
/// 必要長ちょうどのスライス／バッファを渡す）常に境界検査を満たすため
/// `TileBoundsError` を返さないはずだが、以前は `unwrap_or_else` +
/// `unreachable!` で `Result` を panic へ変換していた（AGENTS.md「本番
/// 経路の panic 禁止」・`.claude/rules/coding-rust.md`「本番経路で
/// unwrap()/expect() を使わない」への抵触）。本関数は `Result<(), GemmError>`
/// を返し、`?`（[`GemmError::MicrokernelTileBounds`] への `From` 変換）で
/// 呼び出し元まで型付きエラーとして伝播させる（[`GemmError`] の `#[non_exhaustive]`
/// により呼び出し元の網羅的 match は破壊しない）。
#[allow(clippy::too_many_arguments)]
fn gemm_blis_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    bufs: &mut PanelBuffers,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    let a = &a[row_start * k_dim..];

    for jc in (0..n).step_by(blocks.nc) {
        let nc_len = blocks.nc.min(n - jc);
        for pc in (0..k_dim).step_by(blocks.kc) {
            let kc_len = blocks.kc.min(k_dim - pc);

            // B パネル packing: nc_len を NR 単位のブロックに分割し、各
            // ブロックを kc_len*NR 要素の連続領域として 1 本のバッファに
            // 詰める（ic ループ全体で使い回すため pc/jc ブロックごとに
            // 1 回のみ実行）。pack_b が panel サブスライスへ直接書き込む
            // ため中間 Vec 確保・copy_from_slice は発生しない（#554:
            // BLIS/matrixmultiply の呼び出し側確保バッファへ直接書き込む
            // packing 方式に合わせ二段コピーを廃止）。バッファ自体は
            // `bufs.b_panel`（呼び出し元 `dispatch_region` が 1 回確保
            // 済み）の先頭サブスライスを再借用する（#556。ループ内での
            // `vec![...]` 確保をゼロにする）。直列経路（本関数）は単一
            // タスクのみが呼ぶため、B packing はここでは並列化しない
            // （並列化版は [`gemm_blis_shared_b_region`] 側。#750）。
            let nr_blocks = nc_len.div_ceil(nr);
            let b_panel = &mut bufs.b_panel[..nr_blocks * kc_len * nr];
            for jr_block in 0..nr_blocks {
                let jr = jr_block * nr;
                let nr_eff = nr.min(nc_len - jr);
                pack_b(
                    &mut b_panel[jr_block * kc_len * nr..(jr_block + 1) * kc_len * nr],
                    b,
                    BPackTile {
                        n_total: n,
                        kc_start: pc,
                        kc_len,
                        col_start: jc + jr,
                        nr,
                        nr_eff,
                    },
                );
            }

            // ic（行パネル）以下のループ本体は [`gemm_blis_ic_loop`] へ
            // 切り出し済み（#750）。並列経路（[`gemm_blis_shared_b_region`]）
            // が「共有 B パネルをタスクごとの行範囲へ適用する」ために同じ
            // 関数を再利用するための共通化であり、本関数（直列経路）では
            // `bufs.a_panel` 全体・`mc_total`（このパネル全域）をそのまま
            // 渡すだけで従来どおりの計算になる。
            let ctx = IcLoopContext {
                b_panel,
                n,
                k_dim,
                pc,
                kc_len,
                jc,
                nr_blocks,
                nc_len,
                blocks,
            };
            gemm_blis_ic_loop(kernel, a, c, mc_total, &mut bufs.a_panel, &ctx)?;
        }
    }
    Ok(())
}

/// [`gemm_blis_ic_loop`] へ渡す (jc,pc) ブロック文脈パラメータ（引数数を
/// 抑えるための束ね。[`pack::APackTile`]／[`pack::BPackTile`] と同じ設計
/// 判断。#750）。全フィールド `Copy`（`b_panel` は共有参照）のため
/// `#[derive(Clone, Copy)]` で値渡しできる。
#[derive(Clone, Copy)]
struct IcLoopContext<'b> {
    /// 呼び出し元が pc/jc ブロックごとに 1 回だけ pack 済みの B パネル
    /// （`nr_blocks * kc_len * nr` 要素）。[`gemm_blis_region`]（直列）
    /// では `bufs.b_panel` のサブスライス、[`gemm_blis_shared_b_region`]
    /// （並列・#750）ではタスク間で共有する 1 本のバッファのサブスライス。
    b_panel: &'b [f32],
    n: usize,
    k_dim: usize,
    pc: usize,
    kc_len: usize,
    jc: usize,
    nr_blocks: usize,
    nc_len: usize,
    blocks: BlockSizes,
}

/// `gemm_blis_region`（直列）・`gemm_blis_shared_b_region`（並列・共有 B。
/// #750）共通の ic（行パネル）以下のループ本体（ic→jr→ir）。
///
/// 呼び出し元が pc/jc ブロックごとに 1 回だけ pack 済みの `ctx.b_panel`
/// を読み取り専用で受け取り、`a`（このタスクが担当する行範囲の先頭が
/// 行 0 に対応するよう既にオフセット済みのスライス）・`c`（同じく行 0
/// 対応でオフセット済み）に対して `mc_total` 行ぶんの A packing・カーネル
/// 呼び出しを行う。A packing 用バッファ `a_panel` は呼び出し元が所有する
/// もの（直列経路は `bufs.a_panel`、並列経路はタスクローカルな
/// `Vec<f32>`）を可変参照で受け取り、`gemm_blis_region` から移設した
/// ロジック自体は一切変更していない（ic/jr/ir の反復順・`pack_a` の
/// 書き込み内容・C タイルのロード/書き戻し構造がそのままのため、
/// bit 完全一致契約〈REQ-2〉・FMA 契約・累積順序を保つ。#750 実装計画
/// §4.1「誰がいつ pack したか」だけが変わるという設計根拠の直接反映）。
#[allow(clippy::too_many_arguments)]
fn gemm_blis_ic_loop<K: Microkernel>(
    kernel: K,
    a: &[f32],
    c: &mut [f32],
    mc_total: usize,
    a_panel: &mut [f32],
    ctx: &IcLoopContext,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let IcLoopContext {
        b_panel,
        n,
        k_dim,
        pc,
        kc_len,
        jc,
        nr_blocks,
        nc_len,
        blocks,
    } = *ctx;

    let mut ic = 0;
    while ic < mc_total {
        let mc_len = blocks.mc.min(mc_total - ic);

        // A パネル packing: mc_len を MR 単位のブロックに分割（jr ループ
        // 全体で使い回すため ic ブロックごとに 1 回のみ）。pack_a が
        // panel サブスライスへ直接書き込むため中間 Vec 確保・
        // copy_from_slice は発生しない（#554。B packing と同じ理由）。
        let mr_blocks = mc_len.div_ceil(mr);
        let a_panel = &mut a_panel[..mr_blocks * kc_len * mr];
        for ir_block in 0..mr_blocks {
            let ir = ir_block * mr;
            let mr_eff = mr.min(mc_len - ir);
            pack_a(
                &mut a_panel[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr],
                a,
                APackTile {
                    k_total: k_dim,
                    row_start: ic + ir,
                    mr,
                    mr_eff,
                    kc_start: pc,
                    kc_len,
                },
            );
        }

        for jr_block in 0..nr_blocks {
            let jr = jr_block * nr;
            let nr_eff = nr.min(nc_len - jr);
            let bp_slice = &b_panel[jr_block * kc_len * nr..(jr_block + 1) * kc_len * nr];

            for ir_block in 0..mr_blocks {
                let ir = ir_block * mr;
                let mr_eff = mr.min(mc_len - ir);
                let ap_slice = &a_panel[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr];
                let col_base = jc + jr;

                if mr_eff == mr && nr_eff == nr {
                    // 完全タイル（#557）: C の実バッファへ
                    // `Microkernel::run_with_ldc` の `ldc` 契約経由で直接
                    // ロード/ストアし、コピーイン/コピーアウトの往復を
                    // 省く。`row0` はこのタイル原点（行 ic+ir・列
                    // col_base）の C 上のオフセットで、サブスライス長
                    // `(mr-1)*n + nr` は行 mr-1・列 nr-1 までを覆う
                    // （`ldc = n`）。完全タイルゆえ `col_base + nr <= n`
                    // が成立し `ldc(=n) >= nr` も自動的に満たされる
                    // （[`microkernel::Microkernel::run_with_ldc`] の
                    // `ldc` 契約参照）。スライス取得自体が範囲外なら
                    // panic する安全操作であり、カーネル入口の `ldc`／
                    // 長さ検査と合わせ REQ-8 の境界検査を二重に満たす。
                    // `run_with_ldc` は外部の `Microkernel` 実装からも
                    // 到達しうる公開入口のため `Result` を返す契約
                    // （#691 レビュー P1 対応）で、本呼び出しは private
                    // な本関数内部から組み込みカーネル（`ScalarKernel`／
                    // `NeonKernel`／`Avx2Kernel`／`Avx512Kernel`。いずれも
                    // `MR`／`NR` はモジュール定数でコンパイル時に 1
                    // 以上）へ、完全タイルゆえ自動的に満たされる
                    // `ldc(=n) >= nr` と、上記スライス長
                    // `(mr-1)*n+nr`（= 必要長そのもの）で呼ぶため、
                    // 境界検査は構築上常に成功するはずだが、
                    // `unreachable!` による panic 変換（#691 レビュー
                    // P1 再指摘）を避け、`?` で
                    // `GemmError::MicrokernelTileBounds` として呼び出し
                    // 元まで型付きエラーで伝播させる（実際に `Err` に
                    // なることは想定していない fail-safe だが、本番経路
                    // の panic 禁止規約を優先する）。
                    let row0 = (ic + ir) * n + col_base;
                    let c_direct = &mut c[row0..row0 + (mr - 1) * n + nr];
                    kernel.run_with_ldc(ap_slice, bp_slice, c_direct, n, kc_len)?;
                } else {
                    // 端タイル: 従来どおり `MAX_TILE` スタックバッファへ
                    // コピーインし、有効部（mr_eff×nr_eff）のみコピー
                    // バックする（padding レーン mr_eff..mr, nr_eff..nr
                    // はゼロのままでよい。書き戻し時に不使用）。
                    // ir_block×jr_block のたびのヒープ確保を避けるため
                    // 固定長スタック配列を使う（Review 指摘: M=N=K=2048
                    // では ir/jr ループの反復数が数十万に達し `Vec`
                    // 確保が無視できないオーバーヘッドになるため）。
                    let mut c_tile_buf = [0.0f32; MAX_TILE];
                    let c_tile = &mut c_tile_buf[..mr * nr];
                    for i in 0..mr_eff {
                        let src =
                            &c[(ic + ir + i) * n + col_base..(ic + ir + i) * n + col_base + nr_eff];
                        c_tile[i * nr..i * nr + nr_eff].copy_from_slice(src);
                    }

                    // `ldc = nr` は組み込みカーネルの `NR` 定数そのもの
                    // であり `c_tile` は `mr*nr` ちょうどの長さで確保
                    // しているため、境界検査は構築上常に成功するはず
                    // （上記完全タイル分岐と同じ根拠）。同様に `?` で
                    // 型付きエラーとして伝播させ、`unreachable!` による
                    // panic 変換を避ける（#691 レビュー P1 再指摘）。
                    kernel.run_with_ldc(ap_slice, bp_slice, c_tile, nr, kc_len)?;

                    for i in 0..mr_eff {
                        let dst = &mut c
                            [(ic + ir + i) * n + col_base..(ic + ir + i) * n + col_base + nr_eff];
                        dst.copy_from_slice(&c_tile[i * nr..i * nr + nr_eff]);
                    }
                }
            }
        }

        ic += blocks.mc;
    }
    Ok(())
}

/// `#[cfg(test)]` の [`gemm_blis_parallel_with_blocks`] の複数タスク経路
/// （実タスク数 >= 2）が呼ぶ、B パネルをタスク間で 1 本だけ共有する
/// 5-loop 本体（イシュー #750・設計 doc
/// `docs/cpu-gemm-b-packing-sharing-decision.md` 案 B）。本番公開入口
/// （[`gemm_blis_parallel`]／[`gemm_blis_bias_act_parallel`]）からは
/// 呼ばれない（下記「本番未結線」節参照）。
///
/// jc/pc ループは直列（[`gemm_blis_region`] と同じ昇順）のまま、各
/// (jc,pc) ブロックで B を 1 本だけ pack して `&[f32]` として全タスクへ
/// 共有し、ic（行パネル）だけをタスク間で並列化する（[`gemm_blis_ic_loop`]
/// を「可変 pack → 不変 `&[f32]` 共有読み」の借用分割で呼ぶ。データ競合は
/// コンパイル時に排除される。`unsafe` は使わない）。C 各要素の FMA 連鎖
/// （p 昇順）・`pack_a`／`pack_b` の書き込み内容は [`gemm_blis_region`]
/// と一切変わらないため、`gemm_naive` との bit 完全一致契約（REQ-2）を
/// 保つ（「誰がいつ pack したか」だけが変わる）。
///
/// B packing 自体も `nr` ブロック単位で `par_chunks_mut` により並列化する
/// （各チャンクは書き込み先が排他かつ内容が他チャンクに依存しないため
/// 実行順序に関わらず結果は直列版と同一。#750）。
///
/// A パネル用バッファはタスクごとに 1 本ずつ（`Vec` の所有権がタスク
/// ローカルに閉じるため事前一括確保＋オフセット分割を要さずコンパイル時
/// にデータ競合が排除される。[`PanelBuffers`] ドキュメントコメントが
/// 「タスクごとに 1 組所有する」と述べていた従来設計をそのまま踏襲する）
/// gemm 呼び出しの (jc,pc) ループ全体で使い回すため、ループ外で 1 回だけ
/// 確保する（#556 の確保削減方針をタスク数ぶんに拡張）。
///
/// **本番未結線（#750・codex-review P1 是正）**: 本関数・
/// [`dispatch_shared_b`] は本番公開入口（[`gemm_blis_parallel`]／
/// [`gemm_blis_bias_act_parallel`]）からは呼ばれない
/// （`docs/perf/cpu-gemm-b-packing-sharing.md` 参照。受け入れ条件 2
/// ＝ Apple M4 Max 実測非劣化確認を満たすまでの採用ゲート）。
/// `#[cfg(test)]`（`#[cfg(test)]` の [`gemm_blis_parallel_with_blocks`]
/// 経由）で bit 完全一致検証のみ行う。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gemm_blis_shared_b_region<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k_dim: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    let mr = K::MR;
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    let a = &a[row_start * k_dim..];

    let num_threads = rayon::current_num_threads().max(1);
    let panel_rows = mc_total.div_ceil(num_threads).max(1);
    let num_tasks = mc_total.div_ceil(panel_rows);

    // 共有 B バッファ: 全 jc ブロック中の最大 nc_len に対応する 1 本
    // （`panel_capacity` の B 長算出は `mc_total` に依存しない）。
    let (b_cap, _) = panel_capacity(n, k_dim, mc_total, mr, nr, blocks);
    let mut b_panel_buf = vec![0.0f32; b_cap];

    // タスクローカル A バッファ: 各タスクが担当しうる最大行数
    // （`panel_rows`。最終タスクのみこれより少ない行数を担当しうるが、
    // `panel_capacity` は `mc_total` に対して単調非減少なため
    // `panel_rows` を渡した容量が全タスクの上界になる）に対応する容量で
    // 1 回だけ確保し、(jc,pc) 反復間で使い回す。
    let (_, a_cap) = panel_capacity(n, k_dim, panel_rows, mr, nr, blocks);
    let mut a_bufs: Vec<Vec<f32>> = (0..num_tasks).map(|_| vec![0.0f32; a_cap]).collect();

    for jc in (0..n).step_by(blocks.nc) {
        let nc_len = blocks.nc.min(n - jc);
        for pc in (0..k_dim).step_by(blocks.kc) {
            let kc_len = blocks.kc.min(k_dim - pc);
            let nr_blocks = nc_len.div_ceil(nr);
            let b_panel = &mut b_panel_buf[..nr_blocks * kc_len * nr];

            // B packing の nr ブロック単位並列化（#750）。各チャンクは
            // 排他的な書き込み先で内容も他チャンクに依存しないため、
            // 実行順序に関わらず結果は直列版（[`gemm_blis_region`]）と
            // 同一（bit 完全一致契約に影響しない）。
            b_panel
                .par_chunks_mut(kc_len * nr)
                .enumerate()
                .for_each(|(jr_block, dst)| {
                    let jr = jr_block * nr;
                    let nr_eff = nr.min(nc_len - jr);
                    pack_b(
                        dst,
                        b,
                        BPackTile {
                            n_total: n,
                            kc_start: pc,
                            kc_len,
                            col_start: jc + jr,
                            nr,
                            nr_eff,
                        },
                    );
                });

            let b_panel_ref: &[f32] = b_panel;
            let ctx = IcLoopContext {
                b_panel: b_panel_ref,
                n,
                k_dim,
                pc,
                kc_len,
                jc,
                nr_blocks,
                nc_len,
                blocks,
            };

            // ic（行パネル）のタスク間並列化。`c`（この呼び出しが担当する
            // 行範囲全体）を `panel_rows` 行ずつに分割し、各タスクが
            // `b_panel_ref`（この (jc,pc) ブロックで 1 回だけ pack 済み・
            // 全タスク共有の読み取り専用スライス）を参照しつつ、自分の
            // タスクローカル A バッファへ packing して計算する
            // （データ競合はコンパイル時の借用分割で排除。`unsafe` 不要）。
            c.par_chunks_mut(panel_rows * n)
                .enumerate()
                .zip(a_bufs.par_iter_mut())
                .try_for_each(|((task_idx, c_chunk), a_buf)| {
                    let task_mc = c_chunk.len() / n;
                    if task_mc == 0 {
                        return Ok(());
                    }
                    let task_row_start = task_idx * panel_rows;
                    let a_task = &a[task_row_start * k_dim..];
                    gemm_blis_ic_loop(kernel, a_task, c_chunk, task_mc, a_buf, &ctx)
                })?;
        }
    }
    Ok(())
}

/// 検出済みトークンを優先順位（Avx512 > Avx2 > Scalar）で直接 `try_new`
/// し、最初に構築できたトークンで [`gemm_blis_shared_b_region`] を呼ぶ
/// （[`dispatch_region`] の共有 B 版・実タスク数 >= 2 の並列経路専用の
/// dispatch 入口。イシュー #750）。ロジックは `dispatch_region` と同一で、
/// 呼ぶ先の関数のみが異なる。本番未結線（[`gemm_blis_shared_b_region`]
/// ドキュメンテーションコメント参照）のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
fn dispatch_shared_b(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    #[cfg(avx512_stable)]
    if let Some(kernel) = microkernel::Avx512Kernel::try_new() {
        return gemm_blis_shared_b_region(kernel, a, b, c, n, k, rows, blocks);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        gemm_blis_shared_b_region(kernel, a, b, c, n, k, rows, blocks)
    } else {
        gemm_blis_shared_b_region(ScalarKernel, a, b, c, n, k, rows, blocks)
    }
}

/// aarch64 版 [`dispatch_shared_b`]（#750）。[`dispatch_region`] の
/// aarch64 版と同じ理由で NEON 固定（実行時検出不要）。本番未結線
/// （[`gemm_blis_shared_b_region`] ドキュメンテーションコメント参照）
/// のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(target_arch = "aarch64")]
fn dispatch_shared_b(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    gemm_blis_shared_b_region(microkernel::NeonKernel, a, b, c, n, k, rows, blocks)
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_shared_b`]（#750）。
/// [`dispatch_region`] の同 arch 版と同じ理由で [`ScalarKernel`] 固定。
/// 本番未結線（[`gemm_blis_shared_b_region`] ドキュメンテーションコメント
/// 参照）のため `#[cfg(test)]`。
#[cfg(test)]
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dispatch_shared_b(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    rows: Range<usize>,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    gemm_blis_shared_b_region(ScalarKernel, a, b, c, n, k, rows, blocks)
}

/// テスト専用: 実行環境の実際の ISA 検出結果に依らず、指定したカーネル
/// トークンを強制して [`gemm_blis`] 相当の計算を行う（受け入れ条件
/// 「非対応環境でスカラーフォールバックが動作する」を環境非依存で検証
/// するためのヘルパー。`#[cfg(test)]` 到達可能な `pub(crate)` として公開し、
/// 統合テスト側からは使わない〈lib 単体テストの `mod tests` から使う〉）。
#[cfg(test)]
pub(crate) fn gemm_blis_with_kernel<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    let mut bufs = PanelBuffers::new::<K>(n, k, m, default_blocks());
    gemm_blis_region(kernel, a, b, c, n, k, 0..m, &mut bufs, default_blocks())
}

/// [`gemm_blis_with_kernel_and_blocks`]／[`gemm_blis_parallel_with_blocks`]
/// （実行時に任意の [`BlockSizes`] を受け付けるパラメータ化入口。#564
/// スイープ基盤）が [`PanelBuffers::new`] へ到達する前に検証する:
/// `mc`／`kc`／`nc` のいずれかが 0 だと `gemm_blis_region` 内の
/// `step_by(0)`（`crate::gemm::gemm_blocked_region` の同種バグ〈Cursor
/// Bugbot #231〉と同じ既知の危険）でパニックするため、`GemmError::
/// ZeroBlockSize`（`crate::gemm` 側で既に定義済みのエラー variant。
/// gemm/gemm_blis 間で共有する `GemmError` 型のため新規 variant 追加は
/// 不要）を再利用して早期拒否する（OWASP A03・`.claude/rules/security.md`）。
///
/// panel 容量計算（[`panel_capacity`]）の乗算オーバーフローについては、
/// 本関数の引数として渡る `n`／`k_dim`／`mc_total` が呼び出し元
/// `validate_dims` を先に通過済み（`m*k`／`k*n`／`m*n` が `usize` に収まると
/// 検証済み）であることと、`panel_capacity` が常に `blocks.{mc,kc,nc}
/// .min(dim)` でクランプしてから乗算する構造（[`panel_capacity`] 参照）
/// により、`blocks` 側にどれだけ大きな値（firestorm 参照値
/// KC=4096/NC=9600 等）を渡しても実際の乗算対象は非オーバーフロー確定
/// 済みの dim 由来値に収まるため、追加のオーバーフロー検査は実装上
/// 到達不能と判断し設けない（0 値検査のみで fail-closed 契約を満たす）。
#[cfg(test)]
fn validate_block_sizes(blocks: BlockSizes) -> Result<(), GemmError> {
    if blocks.mc == 0 || blocks.kc == 0 || blocks.nc == 0 {
        return Err(GemmError::ZeroBlockSize {
            mc: blocks.mc,
            kc: blocks.kc,
            nc: blocks.nc,
        });
    }
    Ok(())
}

/// テスト・スイープ専用: [`gemm_blis_with_kernel`] の任意 `BlockSizes` 版
/// （#564）。実運用経路（[`gemm_blis`] 等）は [`default_blocks`]
/// （コンパイル時定数）のみを渡すためこの入口は通過しない。
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_blis_with_kernel_and_blocks<K: Microkernel>(
    kernel: K,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;
    let mut bufs = PanelBuffers::new::<K>(n, k, m, blocks);
    gemm_blis_region(kernel, a, b, c, n, k, 0..m, &mut bufs, blocks)
}

/// テスト専用: [`gemm_blis_shared_b_region`]（B パネル共有経路。#750）の
/// bit 完全一致を任意 `BlockSizes`（#564）で検証するための入口。
/// `gemm_blis_parallel`（本番公開入口）は共有経路を採用しない
/// （[`gemm_blis_shared_b_region`] ドキュメンテーションコメント参照）ため、
/// 本関数は `gemm_blis_parallel` 本体とは分岐が異なる（実タスク数 1 なら
/// `dispatch_region`・2 以上なら `dispatch_shared_b` を常に経由し、
/// 共有経路のテストカバレッジを維持する）。`blocks` のみ呼び出し元から
/// 注入できる。
#[cfg(test)]
pub(crate) fn gemm_blis_parallel_with_blocks(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    // n == 0 の no-op 対処は `gemm_blis_parallel` と同じ理由（本ファイル
    // 冒頭のコメント参照）。
    if n == 0 {
        return Ok(());
    }

    let num_threads = rayon::current_num_threads().max(1);
    let panel_rows = m.div_ceil(num_threads).max(1);

    if m <= panel_rows {
        return dispatch_region(a, b, c, n, k, 0..m, blocks);
    }
    dispatch_shared_b(a, b, c, n, k, 0..m, blocks)
}

/// #753: MC タイル境界に整列した行範囲分配（[`partition::row_ranges_for_workers`]）
/// を使う `gemm_blis_parallel` の 2 次元タイルジョブ分配版。
///
/// 従来の `panel_rows = m.div_ceil(num_threads)` による静的パネル分割
/// （[`gemm_blis_parallel`]）は、MC タイル**数**が `num_threads` で
/// 割り切れない形状では端数タイルを特定 worker へ偏らせる（#753 実装
/// 計画 §3.2）。本関数は行バンド数を [`partition::split_evenly`]
/// （gemm crate `gemm.rs` の n_jobs 分配方式を参照した均等割り）で
/// 均等化してから行範囲へ変換することで、その偏りを ±1 タイルへ抑える。
///
/// 各 worker が受け取る行範囲は依然として `[0, m)` を隙間なく分割した
/// disjoint な連続区間であるため、`c.split_at_mut` の連鎖のみで
/// `unsafe` なしに実現できる。タイル単位で非連続に分配する完全な 2 次元
/// ジョブ分配（gemm crate 本来の方式）は生ポインタによる `unsafe`
/// ラッパーを要するため、PR #766（「常に不活性な sysctl FFI」が P0/P1
/// 指摘で撤去された経緯）を踏まえ #753 では採用しない（
/// [`partition`] モジュールドキュメント「unsafe を使わない設計判断」・
/// `docs/perf/cpu-gemm-runtime-cache-detect.md` 参照）。
///
/// 本番未結線（[`gemm_blis_parallel`]／[`gemm_blis_bias_act_parallel`]
/// からは呼ばれない。受け入れ条件 2＝実機 5 回中央値での非劣化確認が
/// 本 PR のスコープ外のため。#750・#758 と同型の判断）。テスト専用
/// パラメータ化入口として `blocks` を直接受け取る（[`gemm_blis_parallel_with_blocks`]
/// と同じ設計）。
#[cfg(test)]
pub(crate) fn gemm_blis_parallel_2d_with_blocks(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    validate_block_sizes(blocks)?;

    // n == 0 の no-op 対処は `gemm_blis_parallel` と同じ理由（本ファイル
    // 冒頭のコメント参照）。
    if n == 0 {
        return Ok(());
    }

    let num_threads = rayon::current_num_threads().max(1);
    let row_ranges = partition::row_ranges_for_workers(m, blocks.mc, num_threads);

    // 安全な disjoint 分割: `row_ranges` は `[0, m)` を隙間なく連続分割
    // したものなので（[`partition::row_ranges_for_workers`] の被覆完全性
    // 契約。`partition::tests::row_ranges_for_workers_covers_m_contiguously_and_disjointly`
    // で検証済み）、`c` を先頭から順に `split_at_mut` で切り出せば各
    // worker の担当範囲が構築上重複しない（コンパイル時の借用検査で保証。
    // `unsafe` 不要）。
    let mut remaining: &mut [f32] = c;
    let mut chunks: Vec<(usize, &mut [f32])> = Vec::with_capacity(row_ranges.len());
    for r in &row_ranges {
        let len = (r.end - r.start) * n;
        let (head, tail) = remaining.split_at_mut(len);
        chunks.push((r.start, head));
        remaining = tail;
    }

    chunks.into_par_iter().try_for_each(|(row_start, c_chunk)| {
        let row_end = row_start + c_chunk.len() / n;
        dispatch_region(a, b, c_chunk, n, k, row_start..row_end, blocks)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_blis_matches_hand_computed_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_blis_rejects_a_len_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let err = gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            GemmError::ALenMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    // MC/KC/NC 境界を跨ぐ多ブロック形状・並列版との bit 完全一致は
    // `bench_harness`（乱数生成）を要するため、lib 単体テスト
    // （本 `mod tests`）ではなく統合テスト `tests/gemm_blis_parity.rs`
    // 側に集約する。理由: `bench_harness` は `serde_json` を推移依存に
    // 持ち、lib 単体テストバイナリへ持ち込むと `reduction.rs` 側の
    // 無関係な `assert_eq!(&[usize], &[])` が `usize: PartialEq<_>` の
    // 複数実装（`core` と `serde_json::Value` 向け）で型推論あいまいに
    // なり `E0282/E0283` を起こす（同一バイナリにリンクされる依存の
    // trait 実装がクレート全体で可視になるため。実装時に実測確認済み）。
    // 統合テスト（`tests/`）は個別バイナリのためこの問題が生じない。

    #[test]
    fn gemm_blis_parallel_handles_zero_n_as_noop() {
        let a = vec![1.0f32; 4];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        assert!(gemm_blis_parallel(&a, &b, &mut c, 2, 0, 2).is_ok());
    }

    #[test]
    fn gemm_blis_handles_zero_dims() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        assert!(gemm_blis(&a, &b, &mut c, 0, 0, 0).is_ok());
    }

    /// xorshift32 による疑似乱数ベクトル生成（テスト専用。`bench_harness`
    /// を持ち込まない理由は [`microkernel::avx2`] の同名関数のドキュメント
    /// コメント参照。lib 単体テストバイナリへ `serde_json` 推移依存を
    /// 持ち込むと型推論あいまいで E0282/E0283 を起こすため）。
    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f64 / u32::MAX as f64) as f32
            })
            .collect()
    }

    /// 受け入れ条件「非対応環境でスカラーフォールバックが動作する」を
    /// 実行環境の実際の ISA 検出結果に依らず検証する（[`gemm_blis_with_kernel`]
    /// で [`ScalarKernel`] を強制し、[`crate::gemm::gemm_naive`] と bit
    /// 完全一致することを確認）。MC/KC/NC 境界を跨ぐ形状を選ぶ。
    #[test]
    fn gemm_blis_scalar_kernel_forced_matches_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x1111_1111, m * k);
        let b = xorshift32_vec(0x2222_2222, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_scalar = vec![0.0f32; m * n];
        gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_scalar, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_scalar,
            "ScalarKernel 強制経路は gemm_naive と bit 完全一致するはず"
        );
    }

    /// 実行環境で検出された ISA を使う公開入口 [`gemm_blis`] の結果と、
    /// [`ScalarKernel`] 強制経路の結果が bit 完全一致することを確認する
    /// （ISA 間 bit 一致契約〈REQ-2〉の実行時 dispatch 版検証。どの ISA が
    /// 選ばれても [`gemm_blis`] は `ScalarKernel` 強制経路と同じ結果を返す
    /// はず）。
    #[test]
    fn gemm_blis_detected_isa_matches_scalar_forced_bit_exact() {
        let (m, n, k) = (129, 130, 131);
        let a = xorshift32_vec(0x3333_3333, m * k);
        let b = xorshift32_vec(0x4444_4444, k * n);

        let mut c_detected = vec![0.0f32; m * n];
        gemm_blis(&a, &b, &mut c_detected, m, n, k).unwrap();

        let mut c_scalar = vec![0.0f32; m * n];
        gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_scalar, m, n, k).unwrap();

        assert_eq!(
            c_detected, c_scalar,
            "実行時検出された ISA 経路と ScalarKernel 強制経路は bit 完全一致するはず"
        );
    }

    /// #557: 全タイルが完全タイル（直接経路のみを通る）形状で
    /// `gemm_naive` と bit 完全一致することを確認する（[`ScalarKernel`]
    /// 強制。MR=NR=4 の scalar タイル形状に対し m・n がともに倍数の形状
    /// を選ぶことで、端タイル分岐（コピー経路）を一切通さずに直接経路
    /// のみを検証する）。
    #[test]
    fn gemm_blis_scalar_kernel_all_full_tiles_matches_naive_bit_exact() {
        // ScalarKernel は MR=4・NR=4（scalar.rs 参照）。m・n をともに 4 の
        // 倍数にし、かつ MC/KC/NC 境界（128/256/512）を跨ぐ形状を選び、
        // 全 ic/jc/jr/ir 反復で mr_eff == MR && nr_eff == NR が成立する
        // ようにする。
        let (m, n, k) = (256, 512, 300);
        assert_eq!(m % ScalarKernel::MR, 0);
        assert_eq!(n % ScalarKernel::NR, 0);

        let a = xorshift32_vec(0x5555_5555, m * k);
        let b = xorshift32_vec(0x6666_6666, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let mut c_direct = vec![0.0f32; m * n];
        gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_direct, m, n, k).unwrap();

        assert_eq!(
            c_naive, c_direct,
            "全タイル完全（C 直接経路のみ）でも gemm_naive と bit 完全一致するはず"
        );
    }

    /// [`validate_block_sizes`] が `mc`／`kc`／`nc` の 0 値を早期拒否する
    /// ことを検証する（`step_by(0)` パニック防止。`crate::gemm::
    /// GemmError::ZeroBlockSize` 同種バグ〈Cursor Bugbot #231〉の
    /// gemm_blis 版再発防止。#564）。3 フィールドそれぞれを 0 にしたケースを
    /// 個別に検査する。
    #[test]
    fn gemm_blis_with_kernel_and_blocks_rejects_zero_block_size() {
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 4];
        let mut c = vec![0.0f32; 4];

        for blocks in [
            BlockSizes {
                mc: 0,
                kc: 4,
                nc: 4,
            },
            BlockSizes {
                mc: 4,
                kc: 0,
                nc: 4,
            },
            BlockSizes {
                mc: 4,
                kc: 4,
                nc: 0,
            },
        ] {
            let err =
                gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c, 2, 2, 2, blocks)
                    .unwrap_err();
            assert!(
                matches!(err, GemmError::ZeroBlockSize { .. }),
                "blocks={blocks:?} は ZeroBlockSize で拒否されるはず: {err:?}"
            );
        }
    }

    /// 参照実装値近傍を含む非既定 `BlockSizes`（#564 §3.4 候補グリッドの
    /// 縮小版・境界を跨ぐ／下回る／firestorm 近傍の組）でも
    /// [`crate::gemm::gemm_naive`] と bit 完全一致することを検証する
    /// （§3.2「bit 完全一致契約が維持される根拠」の直接検証。x86_64 でも
    /// 実行可能。ScalarKernel 強制で ISA 差を排除する）。
    #[test]
    fn gemm_blis_non_default_block_sizes_match_naive_bit_exact() {
        let (m, n, k) = (37, 53, 71);
        let a = xorshift32_vec(0x9999_9999, m * k);
        let b = xorshift32_vec(0xaaaa_aaaa, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        // 現行値・境界跨ぎの小さい奇数系・firestorm 参照値近傍（MC=480/
        // KC=4096/NC=9600 は m,n,k=37/53/71 に対して常に 1 ブロックに
        // クランプされるが、境界跨ぎと同じ「clamp が正しく効く」経路を
        // 検証する意味を持つ）を横断する。
        for blocks in [
            default_blocks(),
            BlockSizes {
                mc: 8,
                kc: 4,
                nc: 12,
            },
            BlockSizes {
                mc: 16,
                kc: 17,
                nc: 19,
            },
            BlockSizes {
                mc: 480,
                kc: 4096,
                nc: 9600,
            },
            BlockSizes {
                mc: 256,
                kc: 1024,
                nc: 4096,
            },
        ] {
            let mut c_blocked = vec![0.0f32; m * n];
            gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c_blocked, m, n, k, blocks)
                .unwrap();

            assert_eq!(
                c_naive, c_blocked,
                "blocks={blocks:?} は gemm_naive と bit 完全一致するはず"
            );
        }
    }

    /// [`gemm_blis_parallel_with_blocks`]（実運用の行パネル並列経路
    /// `gemm_blis_parallel` の任意 `BlockSizes` 版。#564 スイープ基盤）が、
    /// 非既定 `blocks`（境界跨ぎ・firestorm 参照値近傍）でも
    /// [`crate::gemm::gemm_naive`] と bit 完全一致することを検証する
    /// （epilogue 融合と同じ理由〈要素ごとの演算はタスク分割順序に依存
    /// しない〉で、並列パネル分割数に依らず結果が一致するはずという
    /// §3.2 の主張を並列経路でも直接確認する。x86_64 でも実行可能）。
    #[test]
    fn gemm_blis_parallel_non_default_block_sizes_match_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0xdddd_dddd, m * k);
        let b = xorshift32_vec(0xeeee_eeee, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        for blocks in [
            default_blocks(),
            BlockSizes {
                mc: 16,
                kc: 17,
                nc: 19,
            },
            BlockSizes {
                mc: 480,
                kc: 4096,
                nc: 9600,
            },
        ] {
            let mut c_parallel = vec![0.0f32; m * n];
            gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, blocks).unwrap();

            assert_eq!(
                c_naive, c_parallel,
                "blocks={blocks:?} の並列経路は gemm_naive と bit 完全一致するはず"
            );
        }
    }

    /// #753: 2 次元タイルジョブ分配版
    /// [`gemm_blis_parallel_2d_with_blocks`] が `gemm_naive` と bit 完全
    /// 一致することを、MC タイル数がスレッド数で割り切れない形状
    /// （`blocks.mc=64` に対し `m=523` → タイル数 9・スレッド数
    /// [1,2,3,5,16] のいずれとも非整除）× 複数スレッド数で検証する。
    /// `gemm_blis_parallel_with_blocks` と異なる行範囲計算
    /// （[`partition::row_ranges_for_workers`]）を経由しても、C 各要素の
    /// FMA 連鎖・累積順序は `dispatch_region`／`gemm_blis_region` を
    /// そのまま再利用するため変化しない（bit 完全一致契約・REQ-2）。
    #[test]
    fn gemm_blis_parallel_2d_matches_naive_bit_exact_across_thread_pools() {
        let (m, n, k) = (523, 600, 700);
        let blocks = BlockSizes {
            mc: 64,
            kc: 256,
            nc: 512,
        };
        let a = xorshift32_vec(0x5e5e_5e5e, m * k);
        let b = xorshift32_vec(0x6f6f_6f6f, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        for num_threads in [1usize, 2, 3, 5, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap_or_else(|e| panic!("{num_threads} スレッドの rayon プール構築に失敗: {e}"));

            let mut c_2d = vec![0.0f32; m * n];
            pool.install(|| {
                gemm_blis_parallel_2d_with_blocks(&a, &b, &mut c_2d, m, n, k, blocks).unwrap()
            });

            assert_eq!(
                c_naive, c_2d,
                "gemm_blis_parallel_2d_with_blocks（num_threads={num_threads}）が \
                 gemm_naive と bit 一致しない"
            );
        }
    }

    /// #753: MC/KC/NC の実行時キャッシュ検出（[`cache_params::detected_blocks`]）
    /// で算出したブロックサイズを [`gemm_blis_parallel_with_blocks`] へ
    /// 渡しても `gemm_naive` と bit 完全一致することを検証する（実行環境
    /// 依存の値であっても、GEMM 本体の FMA 契約・累積順序は `blocks` の
    /// 値に依らず不変という [`gemm_blis_region`] の契約〈本ファイル冒頭
    /// ドキュメント「bit 完全一致契約」〉が MC/KC/NC の動的算出後も
    /// 成立することの回帰テスト）。`ScalarKernel` の `MR`／`NR` を渡し、
    /// 実行 ISA に依らず全環境で同じ形状の `blocks` になるようにする。
    #[test]
    fn gemm_blis_parallel_detected_blocks_match_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x7a7a_7a7a, m * k);
        let b = xorshift32_vec(0x8b8b_8b8b, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let blocks = cache_params::detected_blocks(ScalarKernel::MR, ScalarKernel::NR);

        let mut c_parallel = vec![0.0f32; m * n];
        gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, blocks).unwrap();

        assert_eq!(
            c_naive, c_parallel,
            "detected_blocks() 由来の blocks={blocks:?} は gemm_naive と bit 完全一致するはず"
        );
    }

    /// イシュー #750 の受け入れ条件 3（スレッド数 1 では従来と同一経路・
    /// 同一性能）を直接検証する: `num_threads(1)` の rayon プール内で
    /// `gemm_blis_parallel` を呼ぶと `m <= panel_rows`（`panel_rows == m`）
    /// が常に成立し `dispatch_shared_b`（B パネル共有経路）を一切経由し
    /// ない設計になっている（`gemm_blis_parallel` 実装コメント参照）。
    /// 本テストはその結果として `gemm_naive` と bit 完全一致することを
    /// MC/KC/NC 境界を跨ぐ形状で確認する。
    #[test]
    fn gemm_blis_parallel_single_thread_pool_matches_naive_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let a = xorshift32_vec(0x1a1a_1a1a, m * k);
        let b = xorshift32_vec(0x2b2b_2b2b, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap_or_else(|e| panic!("1 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_parallel = vec![0.0f32; m * n];
        pool.install(|| gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap());

        assert_eq!(
            c_naive, c_parallel,
            "num_threads=1 の gemm_blis_parallel は gemm_naive と bit 完全一致するはず（#750 受け入れ条件 3）"
        );
    }

    /// イシュー #750: B パネル共有経路（[`gemm_blis_shared_b_region`]）が
    /// 複数 (jc,pc) ブロック（＝複数回の同期点）を跨いでも
    /// [`gemm_blis_region`]（直列経路）と bit 完全一致することを、小さい
    /// `BlockSizes`（mc=16・kc=17・nc=19。既定値より大幅に小さく多数の
    /// (jc,pc) 反復を強制する）で検証する。固定 4 スレッドプールで
    /// `m > panel_rows`（実タスク数 >= 2）を確定させ、共有 B 経路を確実に
    /// 通す。
    #[test]
    fn gemm_blis_shared_b_region_multi_sync_point_matches_serial_bit_exact() {
        let (m, n, k) = (200, 600, 700);
        let blocks = BlockSizes {
            mc: 16,
            kc: 17,
            nc: 19,
        };
        let a = xorshift32_vec(0x3c3c_3c3c, m * k);
        let b = xorshift32_vec(0x4d4d_4d4d, k * n);

        let mut c_serial = vec![0.0f32; m * n];
        gemm_blis_with_kernel_and_blocks(ScalarKernel, &a, &b, &mut c_serial, m, n, k, blocks)
            .unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap_or_else(|e| panic!("4 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_shared_b = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_with_blocks(&a, &b, &mut c_shared_b, m, n, k, blocks).unwrap()
        });

        assert_eq!(
            c_serial, c_shared_b,
            "多 (jc,pc) 同期点を跨ぐ B パネル共有経路は直列経路と bit 完全一致するはず（#750）"
        );
    }

    /// イシュー #750: 実タスク数 Q が rayon の稼働スレッド数 T を下回る
    /// 形状（m が小さく `m.div_ceil(num_threads) < num_threads` となる
    /// ケース）でも `gemm_naive` と bit 完全一致することを確認する
    /// （`gemm_blis_shared_b_region` の `num_tasks = mc_total.div_ceil(
    /// panel_rows)` が実際のタスク数を正しく導出し、`a_bufs`／
    /// `c.par_chunks_mut` の長さがずれないことの回帰）。
    ///
    /// 本番公開入口 `gemm_blis_parallel` は B パネル共有経路
    /// （`dispatch_shared_b`）を採用しない（本ファイル冒頭
    /// `gemm_blis_parallel` 実装コメント参照。#750・codex-review P1 是正）
    /// ため、`gemm_blis_shared_b_region` の Q<T 回帰を実際に検証するには
    /// `#[cfg(test)]` 限定のテスト専用入口 [`gemm_blis_parallel_with_blocks`]
    /// を経由する必要がある（Cursor Bugbot 指摘・commit f27f233 是正: 本
    /// テストが `gemm_blis_parallel` を直接呼ぶと共有経路を一切経由せず
    /// 回帰検証が失効する）。
    #[test]
    fn gemm_blis_parallel_matches_naive_bit_exact_when_tasks_fewer_than_threads() {
        let (m, n, k) = (10, 130, 40);
        let a = xorshift32_vec(0x5e5e_5e5e, m * k);
        let b = xorshift32_vec(0x6f6f_6f6f, k * n);

        let mut c_naive = vec![0.0f32; m * n];
        crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap_or_else(|e| panic!("16 スレッドの rayon プール構築に失敗: {e}"));

        let mut c_parallel = vec![0.0f32; m * n];
        pool.install(|| {
            gemm_blis_parallel_with_blocks(&a, &b, &mut c_parallel, m, n, k, default_blocks())
                .unwrap()
        });

        assert_eq!(
            c_naive, c_parallel,
            "m={m} 行を num_threads=16 で分割する実タスク数 Q < T のケースは \
             gemm_naive と bit 完全一致するはず（#750）"
        );
    }

    /// [`panel_capacity`]（[`PanelBuffers::new`] の容量計算本体）が、
    /// `gemm_blis_region` の全 jc×pc×ic ブロック反復における実際の必要量
    /// （`nr_blocks*kc_len*nr`／`mr_blocks*kc_len*mr`）の上界になっている
    /// ことを、複数形状 × 全 ISA カーネル定数相当の (mr, nr) 総当たりで
    /// 検証する（#556。§4.1 の「先頭ブロックが最大」という設計根拠の
    /// 直接検証。ブロック境界を跨ぐ形状〈MC/KC/NC 未満・ちょうど・超過〉
    /// を含める）。
    #[test]
    fn panel_capacity_upper_bounds_all_block_iterations() {
        // scalar 4x4・neon 8x12（既定）・neon 12x8（A/B 対抗変種。#559）・
        // avx2 6x16・avx512 8x32（`microkernel/*.rs`）。
        const KERNEL_DIMS: [(usize, usize); 5] = [(4, 4), (8, 12), (12, 8), (6, 16), (8, 32)];
        // MC/KC/NC 境界（128/256/512）を跨ぐ・下回る・ちょうどの形状。
        const SHAPES: [(usize, usize, usize); 6] = [
            (1, 1, 1),
            (7, 7, 7),
            (128, 256, 128),
            (129, 257, 129),
            (600, 700, 300),
            (512, 256, 128),
        ];
        // 検証対象の BlockSizes 候補（レビュー指摘 #564: 既定値のみでは
        // `gemm_blis_with_kernel_and_blocks`／`gemm_blis_parallel_with_blocks`
        // が新規に開放した非既定 blocks 経路の upper bound 不変条件を
        // 検証できない）。既定値・MC/KC/NC 境界を跨ぐ小さめの値・firestorm
        // 参照値近傍（実機スイープ候補として想定される大きめの値）を含める。
        // この不変条件が破れた場合の失敗モードは `gemm_blis_region` 内での
        // スライスインデックスパニックであり、境界を跨ぐ値でこそ検知できる。
        let blocks_candidates: [BlockSizes; 3] = [
            default_blocks(),
            BlockSizes {
                mc: 16,
                kc: 17,
                nc: 19,
            },
            BlockSizes {
                mc: 480,
                kc: 4096,
                nc: 9600,
            },
        ];

        for &blocks in &blocks_candidates {
            for &(mr, nr) in &KERNEL_DIMS {
                for &(n, k_dim, mc_total) in &SHAPES {
                    let (b_cap, a_cap) = panel_capacity(n, k_dim, mc_total, mr, nr, blocks);

                    for jc in (0..n).step_by(blocks.nc) {
                        let nc_len = blocks.nc.min(n - jc);
                        for pc in (0..k_dim).step_by(blocks.kc) {
                            let kc_len = blocks.kc.min(k_dim - pc);
                            let nr_blocks = nc_len.div_ceil(nr);
                            let b_needed = nr_blocks * kc_len * nr;
                            assert!(
                                b_needed <= b_cap,
                                "B 容量不足: blocks={blocks:?},mr={mr},nr={nr},n={n},k={k_dim},\
                                 mc_total={mc_total},jc={jc},pc={pc}: needed={b_needed} > cap={b_cap}"
                            );

                            let mut ic = 0;
                            while ic < mc_total {
                                let mc_len = blocks.mc.min(mc_total - ic);
                                let mr_blocks = mc_len.div_ceil(mr);
                                let a_needed = mr_blocks * kc_len * mr;
                                assert!(
                                    a_needed <= a_cap,
                                    "A 容量不足: blocks={blocks:?},mr={mr},nr={nr},n={n},k={k_dim},\
                                     mc_total={mc_total},jc={jc},pc={pc},ic={ic}: \
                                     needed={a_needed} > cap={a_cap}"
                                );
                                ic += blocks.mc;
                            }
                        }
                    }
                }
            }
        }
    }

    /// `default_blocks()` の値がすべて 0 でないこと（0 だと
    /// `panel_capacity` の `div_ceil`／`step_by` がパニックしうる）を
    /// 確認する（実装計画 §6 OWASP A03 の静的保証をテストで裏付ける）。
    #[test]
    fn default_blocks_never_returns_zero_sized_block() {
        let blocks = default_blocks();
        assert!(blocks.mc > 0 && blocks.kc > 0 && blocks.nc > 0);
    }

    /// n が大きい場合（4096 前後。#749 で NC 拡大分岐の対象だった範囲）
    /// でも本番経路（[`gemm_blis`]／[`gemm_blis_parallel`]）が
    /// `gemm_naive` と bit 完全一致することを、閾値の直前・直後・NR=12
    /// （NEON 8×12 既定カーネルの NR）非整数倍の端タイルが生じる n で
    /// 検証する（m／k は小さく保ち実行時間を抑える）。
    #[test]
    fn gemm_blis_and_parallel_large_n_match_naive_bit_exact() {
        for &(m, k) in &[(5usize, 7usize), (131, 259)] {
            for n in [4095usize, 4096, 4097, 4100] {
                let a = xorshift32_vec(0x1234_5678, m * k);
                let b = xorshift32_vec(0x9abc_def0, k * n);

                let mut c_naive = vec![0.0f32; m * n];
                crate::gemm::gemm_naive(&a, &b, &mut c_naive, m, n, k).unwrap();

                let mut c_blis = vec![0.0f32; m * n];
                gemm_blis(&a, &b, &mut c_blis, m, n, k).unwrap();
                assert_eq!(
                    c_naive, c_blis,
                    "gemm_blis（m={m},n={n},k={k}）は gemm_naive と bit 完全一致するはず"
                );

                let mut c_parallel = vec![0.0f32; m * n];
                gemm_blis_parallel(&a, &b, &mut c_parallel, m, n, k).unwrap();
                assert_eq!(
                    c_naive, c_parallel,
                    "gemm_blis_parallel（m={m},n={n},k={k}）は gemm_naive と bit 完全一致するはず"
                );
            }
        }
    }

    // --- NEON MR=8×NR=12 拡張の aarch64 限定検証（イシュー #559）---
    //
    // x86_64 開発環境では実行不能なため `cfg(target_arch = "aarch64")` で
    // ゲートする（`cargo check --target aarch64-unknown-linux-gnu` の
    // クロス型検査対象にはなるが、x86_64 通常 CI では実行されない。
    // 実機での bit 一致・A/B 実測は `docs/perf/cpu-gemm-neon-mr8-nr12.md`
    // 参照）。

    /// 既定 8×12 カーネル・12×8 A/B 対抗変種いずれも [`ScalarKernel`]
    /// 強制経路と bit 完全一致することを確認する（受け入れ条件 3: parity
    /// テストが green であること）。MC/KC/NC 境界を跨ぐ形状を選ぶ。
    ///
    /// k の一覧はイシュー #561（NEON k=4 アンロール）の主ループ／端数
    /// 分離（k_main = k - k%4）の剰余網羅用に拡張した: 元の k=700 は
    /// KC=256 ブロック分割で各領域の kc_len が 256/256/188（いずれも
    /// 4 の倍数）となり k%4 の剰余が常に 0 で端数ループを一切通らない
    /// （#561 で新設した端数ループの検証漏れになる）。k=701〜703 を
    /// 追加し、各々 KC 分割後の最終領域 kc_len が 189/190/191（k%4 が
    /// 1/2/3）になることで剰余 1/2/3 を通す。MC/NC 境界跨ぎは元の
    /// k=700・(m,n)=(200,600) ケースで既に検証済みのため、追加した
    /// k=701〜703 は端数分岐の検証のみが目的（重複検証を避け aarch64
    /// 実機セッションでの実行コストを抑えるため）小さい (m,n) を使う。
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_8x12_and_12x8_match_scalar_forced_bit_exact() {
        use microkernel::{Neon12x8Kernel, NeonBLaneqKernel, NeonKernel};

        for (i, &(m, n, k)) in [
            (200usize, 600usize, 700usize),
            (16, 20, 701),
            (16, 20, 702),
            (16, 20, 703),
        ]
        .iter()
        .enumerate()
        {
            let seed_a = 0x5555_5555u32 + i as u32;
            let seed_b = 0x6666_6666u32 + i as u32;
            let a = xorshift32_vec(seed_a, m * k);
            let b = xorshift32_vec(seed_b, k * n);

            let mut c_scalar = vec![0.0f32; m * n];
            gemm_blis_with_kernel(ScalarKernel, &a, &b, &mut c_scalar, m, n, k).unwrap();

            let mut c_neon_8x12 = vec![0.0f32; m * n];
            gemm_blis_with_kernel(NeonKernel, &a, &b, &mut c_neon_8x12, m, n, k).unwrap();
            assert_eq!(
                c_scalar, c_neon_8x12,
                "NeonKernel（既定 8×12・k={k}）は ScalarKernel 強制経路と bit 完全一致するはず"
            );

            let mut c_neon_12x8 = vec![0.0f32; m * n];
            gemm_blis_with_kernel(Neon12x8Kernel, &a, &b, &mut c_neon_12x8, m, n, k).unwrap();
            assert_eq!(
                c_scalar, c_neon_12x8,
                "Neon12x8Kernel（A/B 対抗変種・k={k}）は ScalarKernel 強制経路と bit 完全一致するはず"
            );

            // イシュー #748: B 側レーン参照 FMA 変種（列優先 acc）も
            // ScalarKernel 強制経路と bit 完全一致するはず（k%4 の剰余
            // 網羅は上記グリッドを共用）。
            let mut c_neon_b_laneq = vec![0.0f32; m * n];
            gemm_blis_with_kernel(NeonBLaneqKernel, &a, &b, &mut c_neon_b_laneq, m, n, k).unwrap();
            assert_eq!(
                c_scalar, c_neon_b_laneq,
                "NeonBLaneqKernel（B レーン参照変種・k={k}）は ScalarKernel 強制経路と bit 完全一致するはず"
            );
        }
    }

    /// [`NeonKernel`]（既定 8×12）と [`Neon12x8Kernel`]（firestorm 型 A/B
    /// 対抗変種）のスループットを同一形状で計測し中央値を報告する
    /// （`.claude/rules/coding-rust.md` の 5 回計測中央値規約）。
    /// 採用可否の判定は本テストの出力を見た人間／後続セッションが行う
    /// ため、本テスト自体は勝敗を assert しない（イシュー #559 §2.3）。
    /// `#[ignore]` 分離（実機実行専用。`.claude/rules/coding-rust.md`
    /// 実機分離方針）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "aarch64 実機での A/B 性能計測専用（--release 実行推奨）"]
    fn neon_8x12_vs_12x8_ab_median_throughput() {
        use microkernel::{Neon12x8Kernel, NeonKernel};
        use std::time::Instant;

        fn run_once<K: Microkernel>(
            kernel: K,
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k_dim: usize,
        ) -> f64 {
            let mut c = vec![0.0f32; m * n];
            let start = Instant::now();
            gemm_blis_with_kernel(kernel, a, b, &mut c, m, n, k_dim).unwrap();
            start.elapsed().as_secs_f64()
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        for dim in [512usize, 1024, 2048] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0x7777_7777, m * k);
            let b = xorshift32_vec(0x8888_8888, k * n);

            // キャッシュ・TLB を双方カーネルで温めてから計測に入る
            // （ウォームアップなしだと先に計測する側が cold cache を
            // 引き、後段が温まった状態を引き継ぐため系統的に偏る）。
            run_once(NeonKernel, &a, &b, m, n, k);
            run_once(Neon12x8Kernel, &a, &b, m, n, k);

            // 各反復で計測順序を交互化し、cursor[bot] 指摘（PR #693・
            // レビュースレッド PRRT_kwDOTuUCJc6ZnFza）が挙げた
            // 「NeonKernel を常に先に計測するため後段の
            // Neon12x8Kernel がキャッシュ/TLB の温まった状態を
            // 引き継ぎ 12x8 有利に系統的偏りうる」問題を解消する。
            let mut samples_8x12 = Vec::with_capacity(5);
            let mut samples_12x8 = Vec::with_capacity(5);
            for i in 0..5 {
                if i % 2 == 0 {
                    samples_8x12.push(run_once(NeonKernel, &a, &b, m, n, k));
                    samples_12x8.push(run_once(Neon12x8Kernel, &a, &b, m, n, k));
                } else {
                    samples_12x8.push(run_once(Neon12x8Kernel, &a, &b, m, n, k));
                    samples_8x12.push(run_once(NeonKernel, &a, &b, m, n, k));
                }
            }

            let median_8x12 = median(samples_8x12);
            let median_12x8 = median(samples_12x8);

            println!(
                "dim={dim}: NeonKernel(8x12) median={median_8x12:.6}s, \
                 Neon12x8Kernel(12x8) median={median_12x8:.6}s"
            );
        }
    }

    /// [`microkernel::NeonKernel`]（既定・A レーン参照・行優先 acc）と
    /// [`microkernel::NeonBLaneqKernel`]（B レーン参照・列優先 acc。
    /// イシュー #748）のスループットを同一形状で計測し中央値を報告する
    /// （`.claude/rules/coding-rust.md` の 5 回計測中央値規約・上記
    /// `neon_8x12_vs_12x8_ab_median_throughput` と同型のウォームアップ・
    /// 交互実行パターン）。C タイル転置コストを新規に抱えるため、
    /// 採用可否（既定ディスパッチへの接続）の判定は本テスト出力を見た
    /// 人間／後続セッションが行う（本テスト自体は勝敗を assert しない。
    /// #748 実装計画 §2 の fail-closed 方針）。`#[ignore]` 分離（実機実行
    /// 専用。`.claude/rules/coding-rust.md` 実機分離方針）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "aarch64 実機での A/B 性能計測専用（--release 実行推奨）"]
    fn neon_8x12_vs_b_laneq_ab_median_throughput() {
        use microkernel::{NeonBLaneqKernel, NeonKernel};
        use std::time::Instant;

        fn run_once<K: Microkernel>(
            kernel: K,
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k_dim: usize,
        ) -> f64 {
            let mut c = vec![0.0f32; m * n];
            let start = Instant::now();
            gemm_blis_with_kernel(kernel, a, b, &mut c, m, n, k_dim).unwrap();
            start.elapsed().as_secs_f64()
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        for dim in [512usize, 1024, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0x9999_9999, m * k);
            let b = xorshift32_vec(0xAAAA_AAAA, k * n);

            // ウォームアップ（キャッシュ・TLB を双方カーネルで温める。
            // `neon_8x12_vs_12x8_ab_median_throughput` と同じ理由）。
            run_once(NeonKernel, &a, &b, m, n, k);
            run_once(NeonBLaneqKernel, &a, &b, m, n, k);

            // 計測順序を交互化し系統的偏りを避ける（同上）。
            let mut samples_a_lane = Vec::with_capacity(5);
            let mut samples_b_lane = Vec::with_capacity(5);
            for i in 0..5 {
                if i % 2 == 0 {
                    samples_a_lane.push(run_once(NeonKernel, &a, &b, m, n, k));
                    samples_b_lane.push(run_once(NeonBLaneqKernel, &a, &b, m, n, k));
                } else {
                    samples_b_lane.push(run_once(NeonBLaneqKernel, &a, &b, m, n, k));
                    samples_a_lane.push(run_once(NeonKernel, &a, &b, m, n, k));
                }
            }

            let median_a_lane = median(samples_a_lane);
            let median_b_lane = median(samples_b_lane);

            println!(
                "dim={dim}: NeonKernel(A-lane) median={median_a_lane:.6}s, \
                 NeonBLaneqKernel(B-lane) median={median_b_lane:.6}s"
            );
        }
    }

    /// MC/KC/NC 実機スイープ（イシュー #564）: 参照実装値（BLIS
    /// firestorm: MC=480/KC=4096/NC=9600・OpenBLAS: NC 相当 4096）近傍を
    /// 含む候補グリッド × REQ-8 判定形状で [`gemm_blis_parallel_with_blocks`]
    /// （実運用経路 `gemm_blis_parallel` 相当）の中央値スループットを
    /// 計測・報告する（`.claude/rules/coding-rust.md` の 5 回計測中央値
    /// 規約。計測順は [`neon_8x12_vs_12x8_ab_median_throughput`] と同じ
    /// 理由でインターリーブし cache/TLB の系統的偏りを避ける）。
    ///
    /// 対象実機は **Apple M4 Max**（firestorm 系。#481 §3 確定・
    /// `docs/perf/gemm-optimization-baseline.md`）。候補グリッド・選定
    /// 判断基準・実測結果の記録先は `docs/perf/cpu-gemm-blocking-sweep.md`
    /// を参照。採用可否の判定は本テストの出力を見た人間／後続セッションが
    /// 行うため、本テスト自体は勝敗を assert しない（#559 §2.3 と同方針）。
    ///
    /// `#[ignore]` 分離（実機実行専用。`cargo test -p backend-cpu --release
    /// -- --ignored mc_kc_nc_blocking_sweep_median_throughput` で M4 Max
    /// 上から実行する）。
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore = "aarch64 実機（Apple M4 Max）での MC/KC/NC スイープ計測専用（--release 実行推奨。#564）"]
    fn mc_kc_nc_blocking_sweep_median_throughput() {
        use std::time::Instant;

        // §3.4 候補グリッド（現行値・軸別分離・firestorm 参照値そのまま・
        // 中間点）。MC は NEON 既定マイクロカーネル MR=8（#559）の倍数。
        const CANDIDATES: [(&str, BlockSizes); 8] = [
            (
                "現行値",
                BlockSizes {
                    mc: 128,
                    kc: 256,
                    nc: 512,
                },
            ),
            (
                "NC拡大(OpenBLAS相当)",
                BlockSizes {
                    mc: 128,
                    kc: 256,
                    nc: 4096,
                },
            ),
            (
                "NC拡大(firestorm)",
                BlockSizes {
                    mc: 128,
                    kc: 256,
                    nc: 9600,
                },
            ),
            (
                "KC拡大(firestorm)",
                BlockSizes {
                    mc: 128,
                    kc: 4096,
                    nc: 512,
                },
            ),
            (
                "MC拡大(firestorm)",
                BlockSizes {
                    mc: 480,
                    kc: 256,
                    nc: 512,
                },
            ),
            (
                "firestorm全軸",
                BlockSizes {
                    mc: 480,
                    kc: 4096,
                    nc: 9600,
                },
            ),
            (
                "中間点",
                BlockSizes {
                    mc: 256,
                    kc: 1024,
                    nc: 4096,
                },
            ),
            (
                "firestormMC/KC+OpenBLAS-NC",
                BlockSizes {
                    mc: 480,
                    kc: 4096,
                    nc: 4096,
                },
            ),
        ];

        fn run_once(
            a: &[f32],
            b: &[f32],
            m: usize,
            n: usize,
            k_dim: usize,
            blocks: BlockSizes,
        ) -> f64 {
            let mut c = vec![0.0f32; m * n];
            let start = Instant::now();
            gemm_blis_parallel_with_blocks(a, b, &mut c, m, n, k_dim, blocks).unwrap();
            start.elapsed().as_secs_f64()
        }

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        // REQ-8 判定形状（M=N=K=2048/4096）+ 参考 1024（計画 §3.4）。
        for dim in [1024usize, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0xbbbb_bbbb, m * k);
            let b = xorshift32_vec(0xcccc_cccc, k * n);

            // 全候補を 1 巡ウォームアップしてから計測に入る（cache/TLB を
            // 均等に温める。単純な「先頭候補が cold cache を引く」偏りを
            // 避ける）。
            for &(_, blocks) in &CANDIDATES {
                run_once(&a, &b, m, n, k, blocks);
            }

            // 各反復で候補の計測順序をローテーションし、特定候補が常に
            // 先頭/末尾になることによる系統的偏りを避ける
            // （[`neon_8x12_vs_12x8_ab_median_throughput`] の A/B
            // インターリーブと同じ狙いを候補数 N へ一般化）。
            let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(5); CANDIDATES.len()];
            for rep in 0..5 {
                for offset in 0..CANDIDATES.len() {
                    let idx = (offset + rep) % CANDIDATES.len();
                    let (_, blocks) = CANDIDATES[idx];
                    samples[idx].push(run_once(&a, &b, m, n, k, blocks));
                }
            }

            println!("=== dim={dim} ===");
            for (i, &(label, blocks)) in CANDIDATES.iter().enumerate() {
                let med = median(samples[i].clone());
                println!(
                    "  {label} (mc={},kc={},nc={}): median={med:.6}s",
                    blocks.mc, blocks.kc, blocks.nc
                );
            }
        }
    }

    /// #753 実装計画 §3.3: `default_blocks()`（固定値）・
    /// `cache_params::detected_blocks()`（実行時キャッシュ検出）・
    /// `gemm_blis_parallel_2d_with_blocks`（2 次元タイルジョブ分配）の
    /// 3 経路を dim ∈ {512, 1024, 2048, 4096} で 5 回計測の中央値により
    /// A/B 計測する（`.claude/rules/coding-rust.md` の 5 回計測中央値
    /// 規約・[`mc_kc_nc_blocking_sweep_median_throughput`] と同じ計測順
    /// インターリーブ方針）。
    ///
    /// 受け入れ条件 2（実機 5 回中央値での非劣化・gemm crate との差の
    /// 縮小または逆転）の判定自体は本テストの範囲外（本テストは計測結果を
    /// 標準出力へ記録するのみで勝敗を assert しない。`.claude/rules/coding-rust.md`
    /// の「実機依存テストは `#[ignore]` で分離」・#564 の
    /// `mc_kc_nc_blocking_sweep_median_throughput` と同方針）。
    ///
    /// `#[ignore]`（実機実行専用。`cargo test -p backend-cpu --release --
    /// --ignored runtime_cache_detect_and_2d_partition_ab_median_throughput`
    /// で個別実行する想定。非 macOS・非実機環境でも `--release` なしで
    /// フォールバック経路のスモークとして完走することは
    /// `cache_params::tests::detected_blocks_returns_valid_block_sizes_on_any_platform`
    /// が別途保証する）。
    #[test]
    #[ignore = "実機（対象は Apple M4 Max。#753）での A/B 計測専用。--release 推奨"]
    fn runtime_cache_detect_and_2d_partition_ab_median_throughput() {
        use std::time::Instant;

        fn median(mut samples: Vec<f64>) -> f64 {
            samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
            samples[samples.len() / 2]
        }

        // 実行時に一度だけ決定する: 対象は本テスト実行環境で実際に
        // dispatch_region が選ぶマイクロカーネルではなく、
        // `default_blocks()` と同じく ISA 非依存の代表値（ScalarKernel の
        // MR/NR）で `detected_blocks()` を評価する（本番 3 公開関数の
        // `blocks` が ISA によらず単一の値である設計〈本ファイル冒頭
        // `default_blocks` ドキュメント参照〉と揃える）。
        let detected = cache_params::detected_blocks(ScalarKernel::MR, ScalarKernel::NR);
        println!(
            "detected_blocks: mc={} kc={} nc={} (default: mc={} kc={} nc={})",
            detected.mc, detected.kc, detected.nc, MC, KC, NC
        );

        for dim in [512usize, 1024, 2048, 4096] {
            let (m, n, k) = (dim, dim, dim);
            let a = xorshift32_vec(0x9c9c_9c9c, m * k);
            let b = xorshift32_vec(0xadad_adad, k * n);

            fn run_default(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> f64 {
                let mut c = vec![0.0f32; m * n];
                let start = Instant::now();
                gemm_blis_parallel(a, b, &mut c, m, n, k).unwrap();
                start.elapsed().as_secs_f64()
            }
            fn run_detected(
                a: &[f32],
                b: &[f32],
                m: usize,
                n: usize,
                k: usize,
                blocks: BlockSizes,
            ) -> f64 {
                let mut c = vec![0.0f32; m * n];
                let start = Instant::now();
                gemm_blis_parallel_with_blocks(a, b, &mut c, m, n, k, blocks).unwrap();
                start.elapsed().as_secs_f64()
            }
            fn run_2d(
                a: &[f32],
                b: &[f32],
                m: usize,
                n: usize,
                k: usize,
                blocks: BlockSizes,
            ) -> f64 {
                let mut c = vec![0.0f32; m * n];
                let start = Instant::now();
                gemm_blis_parallel_2d_with_blocks(a, b, &mut c, m, n, k, blocks).unwrap();
                start.elapsed().as_secs_f64()
            }

            // 全経路を 1 巡ウォームアップ（cache/TLB を均等に温める。
            // `mc_kc_nc_blocking_sweep_median_throughput` と同じ狙い）。
            run_default(&a, &b, m, n, k);
            run_detected(&a, &b, m, n, k, detected);
            run_2d(&a, &b, m, n, k, default_blocks());

            let mut samples_default = Vec::with_capacity(5);
            let mut samples_detected = Vec::with_capacity(5);
            let mut samples_2d = Vec::with_capacity(5);
            for i in 0..5 {
                // 計測順序を交互化し系統的偏りを避ける（同上）。
                match i % 3 {
                    0 => {
                        samples_default.push(run_default(&a, &b, m, n, k));
                        samples_detected.push(run_detected(&a, &b, m, n, k, detected));
                        samples_2d.push(run_2d(&a, &b, m, n, k, default_blocks()));
                    }
                    1 => {
                        samples_detected.push(run_detected(&a, &b, m, n, k, detected));
                        samples_2d.push(run_2d(&a, &b, m, n, k, default_blocks()));
                        samples_default.push(run_default(&a, &b, m, n, k));
                    }
                    _ => {
                        samples_2d.push(run_2d(&a, &b, m, n, k, default_blocks()));
                        samples_default.push(run_default(&a, &b, m, n, k));
                        samples_detected.push(run_detected(&a, &b, m, n, k, detected));
                    }
                }
            }

            println!(
                "dim={dim}: default median={:.6}s / detected median={:.6}s / \
                 2d-partition median={:.6}s",
                median(samples_default),
                median(samples_detected),
                median(samples_2d),
            );
        }
    }
}
