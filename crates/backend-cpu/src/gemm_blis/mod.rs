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

pub mod microkernel;
mod pack;

use std::ops::Range;

use tensor_core::Activation;

use crate::gemm::{GemmError, validate_dims};
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

/// C タイルのスタックバッファ最大要素数（`MR * NR` の全 ISA 中の最大値。
/// AVX-512 の 8×32=256 が最大。ジェネリック const 式は stable Rust では
/// 使えないため固定長で確保し、各カーネルモジュールの
/// `const _: () = assert!(MR * NR <= 256);` でこの上限を守ることを
/// コンパイル時に検査する）。
const MAX_TILE: usize = 256;

/// キャッシュブロッキングの行方向ブロックサイズ（A のパネル高さ）。
///
/// [`crate::gemm`] の既存定数（PoC-v2-1 実測環境で選定）を起点値として
/// 踏襲する。マイクロカーネルの MR/NR は ISA ごとに異なる（[`microkernel`]
/// 参照）ため、本モジュール独自の定数として持つ（`gemm` モジュールの
/// MC/KC/NC とは独立にチューニング可能にする）。再チューニングは #24 の
/// スコープ。
const MC: usize = 128;
/// 縮約次元（K）のブロックサイズ。
const KC: usize = 256;
/// 列方向ブロックサイズ（B のパネル幅）。
const NC: usize = 512;

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
    /// 各反復の必要量は `min(NC, n).div_ceil(nr)*min(KC,k_dim)*nr`
    /// （B）／`min(MC, mc_total).div_ceil(mr)*min(KC,k_dim)*mr`（A）が
    /// 常に上界になる（先頭ブロックが nc_len/mc_len/kc_len 最大で、末尾
    /// ブロックはこれらが縮むのみ。§4.1 計画）。MC/KC/NC はコンパイル時
    /// 定数（B ≦ 512KiB・A ≦ 約 132KiB 相当）で上界が押さえられるため
    /// 乗算オーバーフローの懸念はない。`n == 0`／`k_dim == 0`／
    /// `mc_total == 0` では長さ 0 のバッファになり、5-loop 自体が回らない
    /// ためサブスライス取得も発生せず問題ない。
    fn new<K: Microkernel>(n: usize, k_dim: usize, mc_total: usize) -> Self {
        let (b_len, a_len) = panel_capacity(n, k_dim, mc_total, K::MR, K::NR);
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
/// §5-3）。
fn panel_capacity(n: usize, k_dim: usize, mc_total: usize, mr: usize, nr: usize) -> (usize, usize) {
    let kc_len_max = KC.min(k_dim);
    let nc_len_max = NC.min(n);
    let mc_len_max = MC.min(mc_total);
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
    dispatch_region(a, b, c, n, k, 0..m);
    Ok(())
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

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            dispatch_region(a, b, c_chunk, n, k, row_start..row_end);
        });
    Ok(())
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

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .try_for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            dispatch_region(a, b, c_chunk, n, k, row_start..row_end);
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
fn dispatch_region(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize, rows: Range<usize>) {
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
        let mut bufs = PanelBuffers::new::<microkernel::Avx512Kernel>(n, k, mc_total);
        return gemm_blis_region(kernel, a, b, c, n, k, rows, &mut bufs);
    }
    if let Some(kernel) = microkernel::Avx2Kernel::try_new() {
        let mut bufs = PanelBuffers::new::<microkernel::Avx2Kernel>(n, k, mc_total);
        gemm_blis_region(kernel, a, b, c, n, k, rows, &mut bufs);
    } else {
        let mut bufs = PanelBuffers::new::<ScalarKernel>(n, k, mc_total);
        gemm_blis_region(ScalarKernel, a, b, c, n, k, rows, &mut bufs);
    }
}

/// aarch64 版 [`dispatch_region`]。NEON は baseline ISA のため
/// [`microkernel::Isa::detect`] は常に `Isa::Neon` を返す（実行時検出不要。
/// [`microkernel`] モジュールドキュメント参照）。
#[cfg(target_arch = "aarch64")]
fn dispatch_region(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize, rows: Range<usize>) {
    debug_assert_eq!(Isa::detect(), Isa::Neon);
    let mc_total = rows.end - rows.start;
    let mut bufs = PanelBuffers::new::<microkernel::NeonKernel>(n, k, mc_total);
    gemm_blis_region(microkernel::NeonKernel, a, b, c, n, k, rows, &mut bufs);
}

/// aarch64／x86_64 以外の arch 版 [`dispatch_region`]。実行時検出対象の
/// ISA を持たないため常に [`ScalarKernel`] を使う。
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn dispatch_region(a: &[f32], b: &[f32], c: &mut [f32], n: usize, k: usize, rows: Range<usize>) {
    debug_assert_eq!(Isa::detect(), Isa::Scalar);
    let mc_total = rows.end - rows.start;
    let mut bufs = PanelBuffers::new::<ScalarKernel>(n, k, mc_total);
    gemm_blis_region(ScalarKernel, a, b, c, n, k, rows, &mut bufs);
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
/// `kernel.run(...)` で累積計算を呼ぶ（#185 でジェネリック化。ISA ごとに
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
/// `#[allow(clippy::too_many_arguments)]`: `bufs` 追加により 8 引数に
/// なる。本ファイル内の既存慣例（[`gemm_blis_bias_act_parallel`] の
/// 同 attribute 使用箇所と同方針）に従い、構造体化はせず素朴な引数列の
/// まま許容する。
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
) {
    let mr = K::MR;
    let nr = K::NR;
    let row_start = rows.start;
    let mc_total = rows.end - rows.start;
    let a = &a[row_start * k_dim..];

    for jc in (0..n).step_by(NC) {
        let nc_len = NC.min(n - jc);
        for pc in (0..k_dim).step_by(KC) {
            let kc_len = KC.min(k_dim - pc);

            // B パネル packing: nc_len を NR 単位のブロックに分割し、各
            // ブロックを kc_len*NR 要素の連続領域として 1 本のバッファに
            // 詰める（ic ループ全体で使い回すため pc/jc ブロックごとに
            // 1 回のみ実行）。pack_b が panel サブスライスへ直接書き込む
            // ため中間 Vec 確保・copy_from_slice は発生しない（#554:
            // BLIS/matrixmultiply の呼び出し側確保バッファへ直接書き込む
            // packing 方式に合わせ二段コピーを廃止）。バッファ自体は
            // `bufs.b_panel`（呼び出し元 `dispatch_region` が 1 回確保
            // 済み）の先頭サブスライスを再借用する（#556。ループ内での
            // `vec![...]` 確保をゼロにする）。
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

            let mut ic = 0;
            while ic < mc_total {
                let mc_len = MC.min(mc_total - ic);

                // A パネル packing: mc_len を MR 単位のブロックに分割
                // （jr ループ全体で使い回すため ic ブロックごとに 1 回のみ）。
                // pack_a が panel サブスライスへ直接書き込むため中間 Vec
                // 確保・copy_from_slice は発生しない（#554。B packing と
                // 同じ理由）。バッファ自体は `bufs.a_panel`（呼び出し元
                // `dispatch_region` が 1 回確保済み）の先頭サブスライスを
                // 再借用する（#556）。
                let mr_blocks = mc_len.div_ceil(mr);
                let a_panel = &mut bufs.a_panel[..mr_blocks * kc_len * mr];
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
                        let ap_slice =
                            &a_panel[ir_block * kc_len * mr..(ir_block + 1) * kc_len * mr];

                        // C タイルの現在値をロード（複数 pc ブロックに
                        // またがる累積を成立させるため、ゼロ初期化せず
                        // 実際の現在値を読み込む）。padding レーン
                        // （mr_eff..mr, nr_eff..nr）はゼロのままでよい
                        // （書き戻し時に有効部のみコピーするため不使用）。
                        // `MAX_TILE` 固定長スタックバッファを確保し
                        // `mr*nr` ぶんだけ使う（ir_block×jr_block のたびに
                        // ヒープ確保しない。Review 指摘: M=N=K=2048 では
                        // ir/jr ループの反復数が数十万に達し `Vec` 確保が
                        // 無視できないオーバーヘッドになるため）。
                        let mut c_tile_buf = [0.0f32; MAX_TILE];
                        let c_tile = &mut c_tile_buf[..mr * nr];
                        let col_base = jc + jr;
                        for i in 0..mr_eff {
                            let src = &c[(ic + ir + i) * n + col_base
                                ..(ic + ir + i) * n + col_base + nr_eff];
                            c_tile[i * nr..i * nr + nr_eff].copy_from_slice(src);
                        }

                        kernel.run(ap_slice, bp_slice, c_tile, kc_len);

                        for i in 0..mr_eff {
                            let dst = &mut c[(ic + ir + i) * n + col_base
                                ..(ic + ir + i) * n + col_base + nr_eff];
                            dst.copy_from_slice(&c_tile[i * nr..i * nr + nr_eff]);
                        }
                    }
                }

                ic += MC;
            }
        }
    }
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
    let mut bufs = PanelBuffers::new::<K>(n, k, m);
    gemm_blis_region(kernel, a, b, c, n, k, 0..m, &mut bufs);
    Ok(())
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

    /// [`panel_capacity`]（[`PanelBuffers::new`] の容量計算本体）が、
    /// `gemm_blis_region` の全 jc×pc×ic ブロック反復における実際の必要量
    /// （`nr_blocks*kc_len*nr`／`mr_blocks*kc_len*mr`）の上界になっている
    /// ことを、複数形状 × 全 ISA カーネル定数相当の (mr, nr) 総当たりで
    /// 検証する（#556。§4.1 の「先頭ブロックが最大」という設計根拠の
    /// 直接検証。ブロック境界を跨ぐ形状〈MC/KC/NC 未満・ちょうど・超過〉
    /// を含める）。
    #[test]
    fn panel_capacity_upper_bounds_all_block_iterations() {
        // scalar 4x4・neon 8x8・avx2 6x16・avx512 8x32（`microkernel/*.rs`）。
        const KERNEL_DIMS: [(usize, usize); 4] = [(4, 4), (8, 8), (6, 16), (8, 32)];
        // MC/KC/NC 境界（128/256/512）を跨ぐ・下回る・ちょうどの形状。
        const SHAPES: [(usize, usize, usize); 6] = [
            (1, 1, 1),
            (7, 7, 7),
            (128, 256, 128),
            (129, 257, 129),
            (600, 700, 300),
            (512, 256, 128),
        ];

        for &(mr, nr) in &KERNEL_DIMS {
            for &(n, k_dim, mc_total) in &SHAPES {
                let (b_cap, a_cap) = panel_capacity(n, k_dim, mc_total, mr, nr);

                for jc in (0..n).step_by(NC) {
                    let nc_len = NC.min(n - jc);
                    for pc in (0..k_dim).step_by(KC) {
                        let kc_len = KC.min(k_dim - pc);
                        let nr_blocks = nc_len.div_ceil(nr);
                        let b_needed = nr_blocks * kc_len * nr;
                        assert!(
                            b_needed <= b_cap,
                            "B 容量不足: mr={mr},nr={nr},n={n},k={k_dim},mc_total={mc_total},\
                             jc={jc},pc={pc}: needed={b_needed} > cap={b_cap}"
                        );

                        let mut ic = 0;
                        while ic < mc_total {
                            let mc_len = MC.min(mc_total - ic);
                            let mr_blocks = mc_len.div_ceil(mr);
                            let a_needed = mr_blocks * kc_len * mr;
                            assert!(
                                a_needed <= a_cap,
                                "A 容量不足: mr={mr},nr={nr},n={n},k={k_dim},mc_total={mc_total},\
                                 jc={jc},pc={pc},ic={ic}: needed={a_needed} > cap={a_cap}"
                            );
                            ic += MC;
                        }
                    }
                }
            }
        }
    }
}
