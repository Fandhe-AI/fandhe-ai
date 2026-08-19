//! aarch64 NEON マイクロカーネル（既定 MR=8×NR=12・24 accumulator、
//! `vfmaq_laneq_f32`）。
//!
//! `cfg(target_arch = "aarch64")` 限定でコンパイルされる（NEON は aarch64
//! のベースライン ISA であり、Metal 実機（Apple M4 Max）・DGX Spark GB10
//! の Grace CPU 側いずれでも常時利用可能なため、[`super::avx2`] と異なり
//! `target_feature` によるコンパイル時分岐は不要）。x86_64 開発環境では
//! 実行検証できないため、`cargo check --target aarch64-unknown-linux-gnu`
//! によるコンパイル検証に留める（実機実行確認は `#[ignore]` テストへ委ねる。
//! `.claude/rules/coding-rust.md` 実機分離方針）。
//!
//! ## MR=8×NR=12 拡張（イシュー #559）
//!
//! 旧実装（#552）は MR=8×NR=8（アキュムレータ `float32x4_t` × 16 本）
//! だったが、aarch64 は v0〜v31 の 32 本のベクタレジスタを持ち、BLIS
//! armv8a sgemm カーネルの参照実装（8×12・アキュムレータ 24 本）はこれを
//! より活用する。本モジュールは既定カーネルを MR=8×NR=12（acc 24 本 + B
//! 3 本 + A 2 本 = 29 本、v0〜v31 に収まる）へ拡張し、A/B 実測用に
//! [`kernel_12x8`]（firestorm 型変種。§ 下部）を併設する。
//!
//! A オペランドのロードはレーン選択 FMA（`vfmaq_laneq_f32`。単一命令
//! `FMLA v.4s, v.4s, v.s[lane]`）を用いる（イシュー #552）。旧実装は p
//! ごとに A の各行値をスカラー読み出し → `vdupq_n_f32` で明示 broadcast
//! → `vfmaq_f32` する 2 命令方式だったが、[`super::super::pack::pack_a`]
//! が A パネルを p-major・mr 方向連続（`dst[p * mr + i]`）で packing する
//! ため、p ごとに `vld1q_f32` 2 回（8 行ぶん一括）で A 値をロードし、
//! レーンを直接 FMA へ渡せる。broadcast 命令（DUP）とスカラーロードを
//! 排し k-step あたりの命令数を削減する（BLIS armv8a sgemm カーネル・
//! matrixmultiply の sgemm_kernel と同技法）。
//!
//! FMA 契約（REQ-2）: `vfmaq_laneq_f32(acc, b, a, LANE)` は
//! `acc + b * a[LANE]` の単一 fused multiply-add であり、`DUP` は
//! ブロードキャストのみで演算順序・丸めに影響しないため、旧
//! `vdupq_n_f32` + `vfmaq_f32` 方式と数学的に同一（`p` 昇順の FMA 連鎖・
//! 乗算順序は不変）。MR=8×NR=12 化は C 各要素の演算順序（p 昇順の FMA
//! 連鎖・レーン間縮約なし）自体を変えずタイル形状のみを変えるため、
//! この bit 完全一致契約は理論上維持される（実機での実測は下記
//! `docs/perf/cpu-gemm-neon-mr8-nr12.md` の環境ゲート判定を参照）。
//! [`super::scalar::kernel`]・[`super::avx2`] と丸めが同一になる契約
//! （PoC-v2-5 の K=4096 ストレスケースで GPU 側含め実測確認済み）も
//! 維持され、bit 完全一致は `gemm_blis_parity` テストで検証する
//! （aarch64 実行環境では NEON 経路が既定選択されるため実機実行時に
//! この経路の bit 一致が検査される）。
//!
//! ## k=4 アンロール＋ソフトウェアパイプライン（イシュー #561）
//!
//! [`kernel`]・[`kernel_12x8`] いずれも k ループを BLIS armv8a 参照実装の
//! `k_iter = k/4`（主ループ）・`k_left = k%4`（端数ループ）分離技法へ
//! 書き換えた。主ループは 4 ステップ（p, p+1, p+2, p+3）を 1 チャンクと
//! して展開し、各ステップの A/B オペランドロード（`vld1q_f32`）を
//! 直前ステップの FMA 列の合間へ先出し発行する（2 段ソフトウェア
//! パイプライン）ことでロードレイテンシの隠蔽を狙う。
//!
//! **bit 完全一致契約は変更しない**: アキュムレータ（`acc[i][j]`）は
//! 分割せず、FMA の演算順序（p 昇順・行 i 昇順・`[0]`→`[1]`→`[2]`）は
//! 主ループ・端数ループを通じて元実装（#559）と完全に同一のまま
//! 保つ。変更するのはロードの発行位置（ソースコード上の並び）のみで
//! あり、`vld1q_f32` 自体は丸め・演算順序に影響しないため、この
//! 並べ替えは冒頭の bit 完全一致契約に抵触しない。
//!
//! **先読み境界**: 主ループはチャンク内（p〜p+3、p はチャンク先頭で
//! 4 の倍数）でのみ先読みし、チャンク境界を越えて次チャンク・端数
//! 領域を読み出さない。`k_main = kc_len - (kc_len % 4)` としたとき
//! チャンク内の最大オフセットは `p + 3 <= k_main - 1 < kc_len` が
//! 常に成り立つため、チャンク内で先読みする p〜p+3 のいずれのロードも
//! 入口 `assert_eq!` が保証する `kc_len` 範囲を超えない（REQ-8: 境界
//! チェックを省略しない）。端数ループ（`k_main..kc_len`）は #559 と
//! 同一の 1 ステップ構造をそのまま維持する。
//!
//! レジスタ収支: 既定カーネルは acc 24 本 + 現行ステップのオペランド
//! 5 本で 29 本（v0〜v31 の 32 本以内）だが、先読み対象ステップの
//! オペランド（最大 5 本）が一時的に重複して生存するため、チャンク内
//! 短時間ではあるが 32 本を超えスピルしうる（BLIS 参照実装でも
//! 同種のトレードオフが存在する）。実効性能・スピル有無は aarch64
//! 実機でのみ計測可能であり、本イシューでは実装・クロス型検査・
//! x86_64 側リグレッションまでを実施し、実測は
//! `docs/perf/cpu-gemm-neon-k4-unroll.md` の環境ゲート判定に従い
//! fail-closed で後続セッションへ引き継ぐ（悪化が実機で確認された
//! 場合はロード配置のみ簡素化する余地を残す。アンロール自体〈k_main/
//! k_left 分離〉は維持する）。

//! ## B 側レーン参照 FMA 変種（イシュー #748）
//!
//! [`kernel`]（既定）・[`kernel_12x8`] はいずれも `vfmaq_laneq_f32` の
//! レーン参照オペランドを **A 側**に割り当てている（`acc + b * a[LANE]`、
//! acc は行優先＝C の行 i・列 4g..4g+4）。gemm crate（faer 実体）・
//! matrixmultiply が採る技法は逆で、**B 側**をレーン参照にする
//! （`acc + a * b[LANE]`）。この場合アキュムレータは列優先（acc[j][h] =
//! C の列 j・行 4h..4h+4）へ転置される。
//!
//! [`compute_b_laneq`] はこの変種を実装する。**この bit 完全一致契約は
//! 有限値（非 NaN）の乗数に限る**: IEEE-754-2008 は乗算の可換性
//! （`a*b` と `b*a` が同一結果）を要求するが、両オペランドが NaN の
//! 場合にどちらの NaN を（どの符号・payload で）返すかは実装依存と
//! されており（§6.2）、オペランド順序を入れ替える本変種
//! （`acc + a*b[LANE]`。[`compute`] の `acc + b*a[LANE]` から乗数順を
//! 交換）が同じ NaN 選択規則に従う保証はない。したがって
//! `acc + b*a` と `acc + a*b` の bit 完全一致は**有限値の入力に対して
//! のみ**主張する（各 C 要素は引き続き p ごとに 1 回だけ FMA を受け
//! p 昇順の連鎖も不変であるため、有限値では [`compute`] と bit 完全
//! 一致する。ローカル検証は `compute_b_laneq_matches_compute_bit_exact`
//! テスト・有限値のみ）。NaN を含む入力に対する両変種間の一致は
//! 主張しない（未検証。`compute_b_laneq_nan_input_does_not_panic`
//! テストで NaN 入力自体がパニックしないことのみ確認する）。呼び出し元
//! （現在は [`super::NeonBLaneqKernel`] 経由の A/B 計測専用で既定
//! ディスパッチには未接続）が NaN を含むタイルに対し両変種の bit 一致を
//! 前提にしてはならない。
//!
//! C タイル入口ロード・出口ストアは、C メモリが row-major
//! （`c[i*ldc+j]`）である一方 acc が列優先であるため転置を要する。
//! 新規 unsafe 面を広げないため、スカラーによる gather/scatter
//! （スタック上 `[f32; 4]` 経由）で実装し、`vzip`/`vtrn` 系ベクトル化
//! 転置は導入しない（実測で入口/出口コストが支配的と判明した場合の
//! 追加最適化候補として計画時点で out-of-scope とした）。
//!
//! k ループの A/B ロードは `vld1q_f32_x2`（A: 8 行を 1 命令）・
//! `vld1q_f32_x3`（B: 12 列を 1 命令）による複数レジスタ同時ロードを
//! 用いる（stable rustc での aarch64 向けコンパイル可否は計画セッションで
//! 事前検証済み）。k=4 アンロール＋ソフトウェアパイプライン構造は
//! [`compute`]（#561）と同型。
//!
//! [`kernel`]・[`kernel_with_ldc`]・[`kernel_12x8`] は本変種の追加により
//! 一切変更しない（A/B 比較の基準・公開 API 非破壊。#748 実装計画）。
//! 既定ディスパッチ（[`super::NeonKernel`]）への接続は実機での bit
//! 一致・非劣化確認後に判断する（fail-closed。未接続の間は
//! [`super::NeonBLaneqKernel`] 経由の A/B 計測専用）。

use std::arch::aarch64::{
    float32x4_t, vfmaq_laneq_f32, vld1q_f32, vld1q_f32_x2, vld1q_f32_x3, vst1q_f32,
};

/// マイクロカーネルタイルの行数（既定カーネル。BLIS armv8a 型）。
pub const MR: usize = 8;
/// マイクロカーネルタイルの列数（f32x4 レジスタ 3 本ぶん。#559）。
pub const NR: usize = 12;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素）固定長で確保するため、コンパイル時に検査する
// （#185。8*12=96 <= 256）。
const _: () = assert!(MR * NR <= 256);

// k ループのレーン展開（`fma_row!` 8 回展開）は MR=8（a0/a1 の 2 レジスタ・
// レーン 0..3 固定）・NR=12（b0/b1/b2 の 3 レジスタ固定）を前提にハードコード
// されている。旧実装（`for i in 0..MR`）と異なりこの前提は暗黙のため、
// MR/NR を変更した場合にコンパイルエラーで検知できるようにする（#552・
// #559 で NR=8→12 へ更新。変更を怠ると `assert_eq!(ap.len(), MR * kc_len)`
// は通過したまま行の一部が計算から欠落し、実行時パニックなしに誤った
// 結果を返しうる）。
const _: () = assert!(MR == 8 && NR == 12);

/// [`super::scalar::kernel`] と同一の累積契約（p 昇順・mul_add 連鎖）を
/// NEON `vfmaq_laneq_f32`（レーン選択 FMA）で実装する。C タイルをレーンごとに独立したレジスタへ
/// ロードし、レーン間縮約を一切行わないため、`p` ごとの `a[p][i]・b[p][j]`
/// への乗算順序はスカラー版と bit 完全一致する。
///
/// # `ldc` 契約（#557）
///
/// `c` は要素 `c[i*ldc+j]`（`i in 0..MR`・`j in 0..NR`）のみを読み書きする。
/// 完全タイル呼び出しでは `ldc = n`（C の実列数）で C バッファへ直接、
/// 端タイル呼び出しでは `ldc = NR` で密パッキングされたスタックバッファへ
/// アクセスする（[`super::Microkernel::run`] 契約と同一）。
///
/// # エラー（#691 レビュー P0 再指摘への対応）
///
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR`（その積の
/// オーバーフローを含む。#691 レビュー P0 再指摘
/// `PRRT_kwDOTuUCJc6ZrXKs`）・`ldc < NR`／`c.len() < (MR - 1) * ldc + NR`
/// のいずれも [`super::TileBoundsError`] として `Result::Err` を返す
/// （panic しない。本関数は外部の `Microkernel` 実装からも到達しうる
/// 公開入口のため）。以降の `unsafe` ロード／ストアはこの検査済み長さの
/// 範囲内でのみ行う。
///
/// # 公開 API 非破壊（#691 レビュー指摘への対応）
///
/// [`super::scalar::kernel_with_ldc`] のドキュメント参照。本モジュールも
/// 同じ理由で従来シグネチャを [`kernel`] として残す。
pub fn kernel_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    super::check_panel_lengths(MR, NR, kc_len, ap.len(), bp.len())?;
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    compute(ap, bp, c, ldc, kc_len);
    Ok(())
}

/// [`kernel_with_ldc`]／[`kernel`] 共通の演算本体（境界検査は呼び出し元の
/// 責務。#691 レビュー P1 再指摘 `PRRT_kwDOTuUCJc6ZrQZG` 対応: `Result` を
/// `panic!` へ変換する経路をなくすため、検査ロジックと演算を分離する）。
fn compute(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    // 要素、c は最大アクセスオフセット `(MR-1)*ldc+NR-1` を含む長さである
    // ことが保証されている（#557: `ldc` 一般化。完全タイル呼び出しでは
    // `ldc = n` で C バッファへ直接、端タイル呼び出しでは `ldc = NR` で
    // スタックバッファへアクセスする）。以下のロード／ストアはいずれも
    // この範囲内のオフセットに限定される:
    // - bp: p*NR, p*NR+4, p*NR+8..+12 の最大は p=kc_len-1 でも
    //   (kc_len-1)*NR+12 = bp.len() を超えない。
    // - ap: p*MR, p*MR+4..+8 の最大は p=kc_len-1 でも (kc_len-1)*MR+8 =
    //   ap.len() を超えない。
    // - c: i*ldc, i*ldc+4, i*ldc+8..+12 の最大は i=MR-1 でも
    //   (MR-1)*ldc+12 <= (MR-1)*ldc+NR = c.len() 上界（`ldc >= NR` を
    //   入口 assert で保証済み）を超えない。
    // k=4 アンロール（#561）: 主ループは p を 4 刻みのチャンクで処理し
    // チャンク内（p..p+3）のみ先読みするため、チャンク先頭 p は 4 の
    // 倍数で p+3 <= k_main-1 < kc_len が常に成り立つ（冒頭モジュール
    // コメント参照）。よって主ループの先読みロードも上記 p 範囲の
    // オフセット上界証明の内側に収まる。端数ループは p in
    // k_main..kc_len で上記と同一の 1 ステップ構造。
    // NEON は aarch64 のベースライン機能であり実行時検出は不要（本モジュール
    // が `cfg(target_arch = "aarch64")` 限定コンパイルであることが前提）。
    unsafe {
        let mut acc: [[float32x4_t; 3]; MR] = std::array::from_fn(|i| {
            [
                vld1q_f32(c[i * ldc..].as_ptr()),
                vld1q_f32(c[i * ldc + 4..].as_ptr()),
                vld1q_f32(c[i * ldc + 8..].as_ptr()),
            ]
        });

        // レーン対応表: pack_a（`dst[p * mr + i]`。p-major・mr 方向連続）
        // により、a0 は行 0..3（レーン k = 行 k）・a1 は行 4..7（レーン
        // k = 行 4+k）を保持する。`vfmaq_laneq_f32::<LANE>(acc, b, a)` は
        // `acc + b * a[LANE]` のため、行 i (<4) は `(a0, i)`、行 i (>=4)
        // は `(a1, i-4)` を参照する。行 i 昇順・`[0]` → `[1]` → `[2]` の
        // 順・p 昇順の FMA 連鎖はいずれも旧 8×8 版・`vdupq_n_f32` 版と
        // 同一に保つ（bit 完全一致契約の前提。冒頭モジュールコメント参照）。
        macro_rules! fma_row {
            ($acc_i:expr, $a:expr, $lane:literal, $b0:expr, $b1:expr, $b2:expr) => {{
                $acc_i[0] = vfmaq_laneq_f32::<$lane>($acc_i[0], $b0, $a);
                $acc_i[1] = vfmaq_laneq_f32::<$lane>($acc_i[1], $b1, $a);
                $acc_i[2] = vfmaq_laneq_f32::<$lane>($acc_i[2], $b2, $a);
            }};
        }

        // ステップ p の A/B オペランドをまとめてロードする（k=4 アンロール
        // の各段で共通利用。#561）。A の 8 行ぶんは #552 と同じく 2 回の
        // vld1q_f32 で一括ロードする（行 0..4 は a0、行 4..8 は a1）。
        macro_rules! load_step {
            ($p:expr) => {
                (
                    vld1q_f32(bp[$p * NR..].as_ptr()),
                    vld1q_f32(bp[$p * NR + 4..].as_ptr()),
                    vld1q_f32(bp[$p * NR + 8..].as_ptr()),
                    vld1q_f32(ap[$p * MR..].as_ptr()),
                    vld1q_f32(ap[$p * MR + 4..].as_ptr()),
                )
            };
        }

        // 主ループ（k_main = kc_len - kc_len%4）: 4 ステップ（p, p+1, p+2,
        // p+3）を 1 チャンクとして展開する。各ステップの FMA 呼び出し順
        // （p 昇順・行 0..8 昇順・[0]→[1]→[2]）は端数ループ・旧実装と
        // 完全に同一（bit 完全一致契約）。次ステップのロードは直前
        // ステップの FMA 列の合間へ先出しし、ロードレイテンシを隠蔽する
        // （2 段ソフトウェアパイプライン。冒頭モジュールコメント参照）。
        let k_main = kc_len - (kc_len % 4);
        let mut p = 0;
        while p < k_main {
            let (b0_0, b1_0, b2_0, a0_0, a1_0) = load_step!(p);
            fma_row!(acc[0], a0_0, 0, b0_0, b1_0, b2_0);
            fma_row!(acc[1], a0_0, 1, b0_0, b1_0, b2_0);
            fma_row!(acc[2], a0_0, 2, b0_0, b1_0, b2_0);
            fma_row!(acc[3], a0_0, 3, b0_0, b1_0, b2_0);

            let (b0_1, b1_1, b2_1, a0_1, a1_1) = load_step!(p + 1);
            fma_row!(acc[4], a1_0, 0, b0_0, b1_0, b2_0);
            fma_row!(acc[5], a1_0, 1, b0_0, b1_0, b2_0);
            fma_row!(acc[6], a1_0, 2, b0_0, b1_0, b2_0);
            fma_row!(acc[7], a1_0, 3, b0_0, b1_0, b2_0);

            let (b0_2, b1_2, b2_2, a0_2, a1_2) = load_step!(p + 2);
            fma_row!(acc[0], a0_1, 0, b0_1, b1_1, b2_1);
            fma_row!(acc[1], a0_1, 1, b0_1, b1_1, b2_1);
            fma_row!(acc[2], a0_1, 2, b0_1, b1_1, b2_1);
            fma_row!(acc[3], a0_1, 3, b0_1, b1_1, b2_1);

            let (b0_3, b1_3, b2_3, a0_3, a1_3) = load_step!(p + 3);
            fma_row!(acc[4], a1_1, 0, b0_1, b1_1, b2_1);
            fma_row!(acc[5], a1_1, 1, b0_1, b1_1, b2_1);
            fma_row!(acc[6], a1_1, 2, b0_1, b1_1, b2_1);
            fma_row!(acc[7], a1_1, 3, b0_1, b1_1, b2_1);

            fma_row!(acc[0], a0_2, 0, b0_2, b1_2, b2_2);
            fma_row!(acc[1], a0_2, 1, b0_2, b1_2, b2_2);
            fma_row!(acc[2], a0_2, 2, b0_2, b1_2, b2_2);
            fma_row!(acc[3], a0_2, 3, b0_2, b1_2, b2_2);
            fma_row!(acc[4], a1_2, 0, b0_2, b1_2, b2_2);
            fma_row!(acc[5], a1_2, 1, b0_2, b1_2, b2_2);
            fma_row!(acc[6], a1_2, 2, b0_2, b1_2, b2_2);
            fma_row!(acc[7], a1_2, 3, b0_2, b1_2, b2_2);

            fma_row!(acc[0], a0_3, 0, b0_3, b1_3, b2_3);
            fma_row!(acc[1], a0_3, 1, b0_3, b1_3, b2_3);
            fma_row!(acc[2], a0_3, 2, b0_3, b1_3, b2_3);
            fma_row!(acc[3], a0_3, 3, b0_3, b1_3, b2_3);
            fma_row!(acc[4], a1_3, 0, b0_3, b1_3, b2_3);
            fma_row!(acc[5], a1_3, 1, b0_3, b1_3, b2_3);
            fma_row!(acc[6], a1_3, 2, b0_3, b1_3, b2_3);
            fma_row!(acc[7], a1_3, 3, b0_3, b1_3, b2_3);

            p += 4;
        }

        // 端数ループ（k_left = kc_len - k_main < 4）: #559 と同一の
        // 1 ステップ構造をそのまま維持する（REQ-8: 手動境界検査の省略
        // 禁止。p < kc_len の範囲内であることは主ループ終了条件から
        // 自明）。
        while p < kc_len {
            let b0 = vld1q_f32(bp[p * NR..].as_ptr());
            let b1 = vld1q_f32(bp[p * NR + 4..].as_ptr());
            let b2 = vld1q_f32(bp[p * NR + 8..].as_ptr());
            let a0 = vld1q_f32(ap[p * MR..].as_ptr());
            let a1 = vld1q_f32(ap[p * MR + 4..].as_ptr());

            fma_row!(acc[0], a0, 0, b0, b1, b2);
            fma_row!(acc[1], a0, 1, b0, b1, b2);
            fma_row!(acc[2], a0, 2, b0, b1, b2);
            fma_row!(acc[3], a0, 3, b0, b1, b2);
            fma_row!(acc[4], a1, 0, b0, b1, b2);
            fma_row!(acc[5], a1, 1, b0, b1, b2);
            fma_row!(acc[6], a1, 2, b0, b1, b2);
            fma_row!(acc[7], a1, 3, b0, b1, b2);

            p += 1;
        }

        for (i, acc_i) in acc.iter().enumerate() {
            vst1q_f32(c[i * ldc..].as_mut_ptr(), acc_i[0]);
            vst1q_f32(c[i * ldc + 4..].as_mut_ptr(), acc_i[1]);
            vst1q_f32(c[i * ldc + 8..].as_mut_ptr(), acc_i[2]);
        }
    }
}

/// [`compute`] の B 側レーン参照変種（イシュー #748。モジュール冒頭
/// コメント § 参照）。境界検査は呼び出し元（[`kernel_b_laneq_with_ldc`]）
/// の責務であり、本関数は検査済み長さの範囲内でのみ `unsafe` ロード／
/// ストアを行う（[`compute`] と同じ分離方針。#691 レビュー P1 再指摘
/// `PRRT_kwDOTuUCJc6ZrQZG` 対応の踏襲）。
fn compute_b_laneq(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    // SAFETY: オフセット上界の証明は [`compute`] と同一の入口契約（呼び出し元が
    // `check_panel_lengths`／`check_c_tile_bounds` を通す）に基づく:
    // - bp: p*NR..p*NR+12 の最大は p=kc_len-1 でも (kc_len-1)*NR+12 =
    //   bp.len() を超えない（panel 長 = kc_len*NR 以上であることを
    //   `check_panel_lengths` が保証）。
    // - ap: p*MR..p*MR+8 の最大は p=kc_len-1 でも (kc_len-1)*MR+8 =
    //   ap.len() を超えない（panel 長 = kc_len*MR 以上であることを同様に
    //   保証）。
    // - c: [`compute`] と同じく `(MR-1)*ldc+NR <= c.len()`（C タイル境界）が
    //   入口 assert で保証済み。列優先アクセス（`(4*h+r)*ldc+j`、r in
    //   0..4・h in 0..2・j in 0..NR）の最大オフセットは r=3,h=1,j=NR-1 の
    //   とき `(MR-1)*ldc+NR-1 < c.len()` に一致し、行優先の [`compute`] と
    //   同じ要素集合を走査するのみで新たな範囲外アクセスを生まない。
    // - k=4 アンロール（#561 と同型）の先読み上界: 主ループはチャンク内
    //   （p..p+3、p は 4 の倍数）でのみ先読みし、`k_main = kc_len -
    //   kc_len%4` のとき p+3 <= k_main-1 < kc_len が常に成り立つため
    //   上記オフセット上界の内側に収まる。
    // - 各 intrinsic のポインタ有効範囲: 入口の C ロード・主／端数ループの
    //   `vld1q_f32`／`vld1q_f32_x2`／`vld1q_f32_x3` はいずれも
    //   `.as_ptr()` から連続 4／8／12 要素分を読むのみで、読み取り元は
    //   上記で範囲内が証明済みのスライス（`ap`／`bp`）またはスタック上の
    //   固定長配列 `lane: [f32; 4]`（常に 4 要素分の有効領域を持つ）の
    //   内部ポインタである。出口の `vst1q_f32` も同じスタック上
    //   `lane: [f32; 4]` への書き込みのみで `c` へは直接ストアせず、
    //   スカラー代入で `c[(4*h+r)*ldc+j]` へ書き戻す（この添字は上記
    //   `(MR-1)*ldc+NR-1 < c.len()` の証明と同一集合）。`vfmaq_laneq_f32`
    //   はポインタを取らずレジスタのみを扱うため対象外。
    unsafe {
        // C タイルロード＋転置: acc[j][h] = C の列 j・行 4h..4h+4
        // （[`compute`] の行優先 acc[i][g] = C の行 i・列 4g..4g+4 とは
        // 転置の関係。冒頭 #748 節参照）。c は row-major のため列方向は
        // 連続しておらず、スカラー gather で一時 `[f32; 4]` へ集めてから
        // `vld1q_f32` する（新規 unsafe 面を広げないための安全策）。
        let mut acc: [[float32x4_t; 2]; NR] = std::array::from_fn(|j| {
            std::array::from_fn(|h| {
                let mut lane = [0.0f32; 4];
                for (r, slot) in lane.iter_mut().enumerate() {
                    *slot = c[(4 * h + r) * ldc + j];
                }
                vld1q_f32(lane.as_ptr())
            })
        });

        // レーン対応表: b0 は列 0..3（レーン k = 列 k）・b1 は列 4..7
        // （レーン k = 列 4+k）・b2 は列 8..11（レーン k = 列 8+k）。
        // `vfmaq_laneq_f32::<LANE>(acc, a, b)` は `acc + a * b[LANE]` の
        // ため、列 j (<4) は `(b0, j)`、列 j (4..8) は `(b1, j-4)`、
        // 列 j (8..12) は `(b2, j-8)` を参照する。列 j 昇順・行グループ
        // h 昇順（`[0]`→`[1]`）・p 昇順の FMA 連鎖を主・端数ループ通じて
        // 一貫させ、bit 完全一致契約（冒頭モジュールコメント）を保つ
        // （この契約は有限値の入力に限る。NaN 入力での [`compute`] との
        // bit 一致は主張しない。冒頭モジュールコメント §748 節参照）。
        macro_rules! fma_col {
            ($acc_j:expr, $b:expr, $lane:literal, $a0:expr, $a1:expr) => {{
                $acc_j[0] = vfmaq_laneq_f32::<$lane>($acc_j[0], $a0, $b);
                $acc_j[1] = vfmaq_laneq_f32::<$lane>($acc_j[1], $a1, $b);
            }};
        }

        // ステップ p の A/B オペランドをまとめてロードする。A の 8 行
        // ぶんは `vld1q_f32_x2` の 1 命令で一括ロード（`.0` = 行 0..4、
        // `.1` = 行 4..8）、B の 12 列ぶんは `vld1q_f32_x3` の 1 命令で
        // 一括ロード（`.0`/`.1`/`.2` = 列 0..4/4..8/8..12）。#748 実装
        // 計画の事前検証（stable rustc・aarch64-unknown-linux-gnu 向け
        // コンパイル可否）により導入した複数レジスタ同時ロード命令。
        macro_rules! load_step {
            ($p:expr) => {{
                let b = vld1q_f32_x3(bp[$p * NR..].as_ptr());
                let a = vld1q_f32_x2(ap[$p * MR..].as_ptr());
                (b.0, b.1, b.2, a.0, a.1)
            }};
        }

        // 主ループ（k_main = kc_len - kc_len%4）: [`compute`] と同型の
        // 4 ステップチャンク展開・先読みインターリーブ（2 段ソフトウェア
        // パイプライン。モジュール冒頭コメント参照）。FMA 呼び出し順
        // （p 昇順・列 0..12 昇順・[0]→[1]）は端数ループ・[`compute`] の
        // 行優先版と要素ごとの p 昇順連鎖という点で同一。
        let k_main = kc_len - (kc_len % 4);
        let mut p = 0;
        while p < k_main {
            let (b0_0, b1_0, b2_0, a0_0, a1_0) = load_step!(p);
            fma_col!(acc[0], b0_0, 0, a0_0, a1_0);
            fma_col!(acc[1], b0_0, 1, a0_0, a1_0);
            fma_col!(acc[2], b0_0, 2, a0_0, a1_0);
            fma_col!(acc[3], b0_0, 3, a0_0, a1_0);

            let (b0_1, b1_1, b2_1, a0_1, a1_1) = load_step!(p + 1);
            fma_col!(acc[4], b1_0, 0, a0_0, a1_0);
            fma_col!(acc[5], b1_0, 1, a0_0, a1_0);
            fma_col!(acc[6], b1_0, 2, a0_0, a1_0);
            fma_col!(acc[7], b1_0, 3, a0_0, a1_0);

            let (b0_2, b1_2, b2_2, a0_2, a1_2) = load_step!(p + 2);
            fma_col!(acc[8], b2_0, 0, a0_0, a1_0);
            fma_col!(acc[9], b2_0, 1, a0_0, a1_0);
            fma_col!(acc[10], b2_0, 2, a0_0, a1_0);
            fma_col!(acc[11], b2_0, 3, a0_0, a1_0);

            fma_col!(acc[0], b0_1, 0, a0_1, a1_1);
            fma_col!(acc[1], b0_1, 1, a0_1, a1_1);
            fma_col!(acc[2], b0_1, 2, a0_1, a1_1);
            fma_col!(acc[3], b0_1, 3, a0_1, a1_1);

            let (b0_3, b1_3, b2_3, a0_3, a1_3) = load_step!(p + 3);
            fma_col!(acc[4], b1_1, 0, a0_1, a1_1);
            fma_col!(acc[5], b1_1, 1, a0_1, a1_1);
            fma_col!(acc[6], b1_1, 2, a0_1, a1_1);
            fma_col!(acc[7], b1_1, 3, a0_1, a1_1);
            fma_col!(acc[8], b2_1, 0, a0_1, a1_1);
            fma_col!(acc[9], b2_1, 1, a0_1, a1_1);
            fma_col!(acc[10], b2_1, 2, a0_1, a1_1);
            fma_col!(acc[11], b2_1, 3, a0_1, a1_1);

            fma_col!(acc[0], b0_2, 0, a0_2, a1_2);
            fma_col!(acc[1], b0_2, 1, a0_2, a1_2);
            fma_col!(acc[2], b0_2, 2, a0_2, a1_2);
            fma_col!(acc[3], b0_2, 3, a0_2, a1_2);
            fma_col!(acc[4], b1_2, 0, a0_2, a1_2);
            fma_col!(acc[5], b1_2, 1, a0_2, a1_2);
            fma_col!(acc[6], b1_2, 2, a0_2, a1_2);
            fma_col!(acc[7], b1_2, 3, a0_2, a1_2);
            fma_col!(acc[8], b2_2, 0, a0_2, a1_2);
            fma_col!(acc[9], b2_2, 1, a0_2, a1_2);
            fma_col!(acc[10], b2_2, 2, a0_2, a1_2);
            fma_col!(acc[11], b2_2, 3, a0_2, a1_2);

            fma_col!(acc[0], b0_3, 0, a0_3, a1_3);
            fma_col!(acc[1], b0_3, 1, a0_3, a1_3);
            fma_col!(acc[2], b0_3, 2, a0_3, a1_3);
            fma_col!(acc[3], b0_3, 3, a0_3, a1_3);
            fma_col!(acc[4], b1_3, 0, a0_3, a1_3);
            fma_col!(acc[5], b1_3, 1, a0_3, a1_3);
            fma_col!(acc[6], b1_3, 2, a0_3, a1_3);
            fma_col!(acc[7], b1_3, 3, a0_3, a1_3);
            fma_col!(acc[8], b2_3, 0, a0_3, a1_3);
            fma_col!(acc[9], b2_3, 1, a0_3, a1_3);
            fma_col!(acc[10], b2_3, 2, a0_3, a1_3);
            fma_col!(acc[11], b2_3, 3, a0_3, a1_3);

            p += 4;
        }

        // 端数ループ（k_left = kc_len - k_main < 4）: [`compute`] と同型
        // の 1 ステップ構造（REQ-8: 手動境界検査の省略禁止。p < kc_len の
        // 範囲内であることは主ループ終了条件から自明）。
        while p < kc_len {
            let b = vld1q_f32_x3(bp[p * NR..].as_ptr());
            let a = vld1q_f32_x2(ap[p * MR..].as_ptr());
            let (b0, b1, b2, a0, a1) = (b.0, b.1, b.2, a.0, a.1);

            fma_col!(acc[0], b0, 0, a0, a1);
            fma_col!(acc[1], b0, 1, a0, a1);
            fma_col!(acc[2], b0, 2, a0, a1);
            fma_col!(acc[3], b0, 3, a0, a1);
            fma_col!(acc[4], b1, 0, a0, a1);
            fma_col!(acc[5], b1, 1, a0, a1);
            fma_col!(acc[6], b1, 2, a0, a1);
            fma_col!(acc[7], b1, 3, a0, a1);
            fma_col!(acc[8], b2, 0, a0, a1);
            fma_col!(acc[9], b2, 1, a0, a1);
            fma_col!(acc[10], b2, 2, a0, a1);
            fma_col!(acc[11], b2, 3, a0, a1);

            p += 1;
        }

        // 出口ストア＋転置: acc[j][h]（列 j・行 4h..4h+4）を row-major の
        // `c[i*ldc+j]` へスカッタで書き戻す（入口ロードと対の転置）。
        for j in 0..NR {
            for h in 0..2 {
                let mut lane = [0.0f32; 4];
                vst1q_f32(lane.as_mut_ptr(), acc[j][h]);
                for (r, &v) in lane.iter().enumerate() {
                    c[(4 * h + r) * ldc + j] = v;
                }
            }
        }
    }
}

/// [`compute_b_laneq`] の検査つき公開入口（[`kernel_with_ldc`] と同型・
/// 同じ `ldc` 契約。イシュー #748）。[`super::super::dispatch_region`] の
/// 既定駆動経路には接続しない（実機での bit 一致・非劣化確認後に判断。
/// モジュール冒頭 #748 節参照）。[`super::NeonBLaneqKernel`] からのみ
/// 呼ばれる。
pub fn kernel_b_laneq_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    super::check_panel_lengths(MR, NR, kc_len, ap.len(), bp.len())?;
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    compute_b_laneq(ap, bp, c, ldc, kc_len);
    Ok(())
}

/// [`kernel_b_laneq_with_ldc`] の `ldc = NR` 密パッキング契約固定版
/// （[`kernel`] と同型の `assert!` 検査。`Result` を `panic!` へ変換する
/// 経路を作らないための独立関数。[`super::NeonBLaneqKernel::run`] から
/// 直接委譲される）。
///
/// ## `#[cfg(test)]` 限定（PR #765 codex-review P1 対応）
///
/// [`super::NeonBLaneqKernel`] は `super::super::dispatch_region` の既定
/// 駆動経路に接続されない `gemm_blis::mod` の `#[cfg(test)]` A/B 計測
/// テスト専用トークンであり、本番ビルドの到達可能経路を持たない
/// （呼び出し元は本関数のドキュメント参照）。にもかかわらず本関数
/// 自体が本番ビルドへコンパイルされていると、契約違反時に `panic!` へ
/// 変換する `assert!`/`assert_eq!` を含む関数が「本番経路の panic 禁止」
/// 規約（`.claude/rules/coding-rust.md` テスト・ベンチ節・AGENTS.md）の
/// 対象として誤認・誤用されうる。唯一の呼び出し元
/// [`super::NeonBLaneqKernel::run`] 側と対にして `#[cfg(test)]` を付け、
/// 本番ビルドから完全に除外する（テスト専用の性質をコンパイル単位でも
/// 保証する）。
#[cfg(test)]
pub fn kernel_b_laneq(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    assert!(
        super::panel_len_matches(ap.len(), MR, kc_len),
        "packed A panel length mismatch (or MR*kc_len overflow): ap.len()={}, MR={MR}, kc_len={kc_len}",
        ap.len()
    );
    assert!(
        super::panel_len_matches(bp.len(), kc_len, NR),
        "packed B panel length mismatch (or kc_len*NR overflow): bp.len()={}, kc_len={kc_len}, NR={NR}",
        bp.len()
    );
    assert_eq!(c_tile.len(), MR * NR, "C tile length mismatch");
    compute_b_laneq(ap, bp, c_tile, NR, kc_len);
}

/// マイクロカーネルタイルの行数（12×8 変種。firestorm 型 A/B 対抗。#559）。
pub const MR_12X8: usize = 12;
/// マイクロカーネルタイルの列数（12×8 変種。f32x4 レジスタ 2 本ぶん）。
pub const NR_12X8: usize = 8;

const _: () = assert!(MR_12X8 * NR_12X8 <= 256);
const _: () = assert!(MR_12X8 == 12 && NR_12X8 == 8);

/// Apple M1 系 firestorm コア向け A/B 対抗変種（12 行 × 8 列、acc 24 本
/// = 12 行 × 2 レジスタ、A 3 本・B 2 本 = 計 29 本で v0〜v31 に収まる）。
///
/// [`kernel`]（既定・8×12）との実機 A/B 計測専用であり、
/// [`super::super::dispatch_region`] の駆動経路には接続しない
/// （イシュー #559 §2.3・`crates/backend-cpu/src/gemm_blis/mod.rs` の
/// `#[cfg(test)]` `mod tests` 経由の A/B 計測テストからのみ呼ばれる）。
/// 累積契約（p 昇順・レーン間縮約なし）は [`kernel`] と同一であり、
/// `gemm_naive` との bit 完全一致が理論上成立する（実機未実測。
/// `docs/perf/cpu-gemm-neon-mr8-nr12.md` 参照）。
///
/// # Panics
///
/// `ap.len() != MR_12X8 * kc_len`／`bp.len() != kc_len * NR_12X8`／
/// `c_tile.len() != MR_12X8 * NR_12X8` のいずれかであればパニックする
/// （REQ-8 境界検査規約。[`kernel`] と同じ契約）。
pub fn kernel_12x8(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    // `ap`／`bp` の長さ検査は `checked_mul` 判定の [`super::panel_len_matches`]
    // 経由に統一する（#691 レビュー P0 再指摘 `PRRT_kwDOTuUCJc6ZrXKs` と
    // 同型の未検査乗算オーバーフローを防ぐ）。
    assert!(
        super::panel_len_matches(ap.len(), MR_12X8, kc_len),
        "packed A panel length mismatch (or MR_12X8*kc_len overflow): ap.len()={}, MR_12X8={MR_12X8}, kc_len={kc_len}",
        ap.len()
    );
    assert!(
        super::panel_len_matches(bp.len(), kc_len, NR_12X8),
        "packed B panel length mismatch (or kc_len*NR_12X8 overflow): bp.len()={}, kc_len={kc_len}, NR_12X8={NR_12X8}",
        bp.len()
    );
    assert_eq!(c_tile.len(), MR_12X8 * NR_12X8, "C tile length mismatch");

    // SAFETY: 直前の assert により ap は MR_12X8*kc_len 要素、bp は
    // kc_len*NR_12X8 要素、c_tile は MR_12X8*NR_12X8(=96) 要素ちょうど
    // であることが保証されている。以下のロード／ストアはいずれもこの
    // 範囲内のオフセットに限定される:
    // - bp: p*NR_12X8, p*NR_12X8+4 の最大は p=kc_len-1 でも
    //   (kc_len-1)*NR_12X8+8 = bp.len() を超えない。
    // - ap: p*MR_12X8, +4, +8..+12 の最大は p=kc_len-1 でも
    //   (kc_len-1)*MR_12X8+12 = ap.len() を超えない。
    // - c_tile: i*NR_12X8, i*NR_12X8+4 の最大は i=MR_12X8-1 でも
    //   (MR_12X8-1)*NR_12X8+8 = c_tile.len() を超えない。
    // k=4 アンロール（#561）: [`kernel`] と同一の理由で、主ループは
    // チャンク内（p..p+3、p は 4 の倍数）でのみ先読みし
    // p+3 <= k_main-1 < kc_len が常に成り立つため上記オフセット上界
    // 証明の内側に収まる（冒頭モジュールコメント参照）。
    // NEON は aarch64 のベースライン機能であり実行時検出は不要（本モジュール
    // が `cfg(target_arch = "aarch64")` 限定コンパイルであることが前提）。
    unsafe {
        let mut acc: [[float32x4_t; 2]; MR_12X8] = std::array::from_fn(|i| {
            [
                vld1q_f32(c_tile[i * NR_12X8..].as_ptr()),
                vld1q_f32(c_tile[i * NR_12X8 + 4..].as_ptr()),
            ]
        });

        // レーン対応表: a0 は行 0..3（レーン k = 行 k）・a1 は行 4..7
        // （レーン k = 行 4+k）・a2 は行 8..11（レーン k = 行 8+k）。
        // 行 i 昇順・`[0]` → `[1]` の順・p 昇順の FMA 連鎖は [`kernel`]
        // と同じ契約を保つ。
        macro_rules! fma_row {
            ($acc_i:expr, $a:expr, $lane:literal, $b0:expr, $b1:expr) => {{
                $acc_i[0] = vfmaq_laneq_f32::<$lane>($acc_i[0], $b0, $a);
                $acc_i[1] = vfmaq_laneq_f32::<$lane>($acc_i[1], $b1, $a);
            }};
        }

        // ステップ p の A/B オペランドをまとめてロードする（[`kernel`]
        // の `load_step!` と同型。A の 12 行ぶんは 3 回の vld1q_f32 で
        // 一括ロード（行 0..4 は a0、行 4..8 は a1、行 8..12 は a2）。
        macro_rules! load_step {
            ($p:expr) => {
                (
                    vld1q_f32(bp[$p * NR_12X8..].as_ptr()),
                    vld1q_f32(bp[$p * NR_12X8 + 4..].as_ptr()),
                    vld1q_f32(ap[$p * MR_12X8..].as_ptr()),
                    vld1q_f32(ap[$p * MR_12X8 + 4..].as_ptr()),
                    vld1q_f32(ap[$p * MR_12X8 + 8..].as_ptr()),
                )
            };
        }

        // 主ループ: [`kernel`] と同型の 4 ステップチャンク展開・先読み
        // インターリーブ（冒頭モジュールコメント参照）。FMA 呼び出し順
        // （p 昇順・行 0..12 昇順・[0]→[1]）は端数ループ・旧実装と完全
        // に同一。
        let k_main = kc_len - (kc_len % 4);
        let mut p = 0;
        while p < k_main {
            let (b0_0, b1_0, a0_0, a1_0, a2_0) = load_step!(p);
            fma_row!(acc[0], a0_0, 0, b0_0, b1_0);
            fma_row!(acc[1], a0_0, 1, b0_0, b1_0);
            fma_row!(acc[2], a0_0, 2, b0_0, b1_0);
            fma_row!(acc[3], a0_0, 3, b0_0, b1_0);

            let (b0_1, b1_1, a0_1, a1_1, a2_1) = load_step!(p + 1);
            fma_row!(acc[4], a1_0, 0, b0_0, b1_0);
            fma_row!(acc[5], a1_0, 1, b0_0, b1_0);
            fma_row!(acc[6], a1_0, 2, b0_0, b1_0);
            fma_row!(acc[7], a1_0, 3, b0_0, b1_0);

            let (b0_2, b1_2, a0_2, a1_2, a2_2) = load_step!(p + 2);
            fma_row!(acc[8], a2_0, 0, b0_0, b1_0);
            fma_row!(acc[9], a2_0, 1, b0_0, b1_0);
            fma_row!(acc[10], a2_0, 2, b0_0, b1_0);
            fma_row!(acc[11], a2_0, 3, b0_0, b1_0);

            fma_row!(acc[0], a0_1, 0, b0_1, b1_1);
            fma_row!(acc[1], a0_1, 1, b0_1, b1_1);
            fma_row!(acc[2], a0_1, 2, b0_1, b1_1);
            fma_row!(acc[3], a0_1, 3, b0_1, b1_1);

            let (b0_3, b1_3, a0_3, a1_3, a2_3) = load_step!(p + 3);
            fma_row!(acc[4], a1_1, 0, b0_1, b1_1);
            fma_row!(acc[5], a1_1, 1, b0_1, b1_1);
            fma_row!(acc[6], a1_1, 2, b0_1, b1_1);
            fma_row!(acc[7], a1_1, 3, b0_1, b1_1);
            fma_row!(acc[8], a2_1, 0, b0_1, b1_1);
            fma_row!(acc[9], a2_1, 1, b0_1, b1_1);
            fma_row!(acc[10], a2_1, 2, b0_1, b1_1);
            fma_row!(acc[11], a2_1, 3, b0_1, b1_1);

            fma_row!(acc[0], a0_2, 0, b0_2, b1_2);
            fma_row!(acc[1], a0_2, 1, b0_2, b1_2);
            fma_row!(acc[2], a0_2, 2, b0_2, b1_2);
            fma_row!(acc[3], a0_2, 3, b0_2, b1_2);
            fma_row!(acc[4], a1_2, 0, b0_2, b1_2);
            fma_row!(acc[5], a1_2, 1, b0_2, b1_2);
            fma_row!(acc[6], a1_2, 2, b0_2, b1_2);
            fma_row!(acc[7], a1_2, 3, b0_2, b1_2);
            fma_row!(acc[8], a2_2, 0, b0_2, b1_2);
            fma_row!(acc[9], a2_2, 1, b0_2, b1_2);
            fma_row!(acc[10], a2_2, 2, b0_2, b1_2);
            fma_row!(acc[11], a2_2, 3, b0_2, b1_2);

            fma_row!(acc[0], a0_3, 0, b0_3, b1_3);
            fma_row!(acc[1], a0_3, 1, b0_3, b1_3);
            fma_row!(acc[2], a0_3, 2, b0_3, b1_3);
            fma_row!(acc[3], a0_3, 3, b0_3, b1_3);
            fma_row!(acc[4], a1_3, 0, b0_3, b1_3);
            fma_row!(acc[5], a1_3, 1, b0_3, b1_3);
            fma_row!(acc[6], a1_3, 2, b0_3, b1_3);
            fma_row!(acc[7], a1_3, 3, b0_3, b1_3);
            fma_row!(acc[8], a2_3, 0, b0_3, b1_3);
            fma_row!(acc[9], a2_3, 1, b0_3, b1_3);
            fma_row!(acc[10], a2_3, 2, b0_3, b1_3);
            fma_row!(acc[11], a2_3, 3, b0_3, b1_3);

            p += 4;
        }

        // 端数ループ: [`kernel`] と同型・#559 と同一の 1 ステップ構造。
        while p < kc_len {
            let b0 = vld1q_f32(bp[p * NR_12X8..].as_ptr());
            let b1 = vld1q_f32(bp[p * NR_12X8 + 4..].as_ptr());
            let a0 = vld1q_f32(ap[p * MR_12X8..].as_ptr());
            let a1 = vld1q_f32(ap[p * MR_12X8 + 4..].as_ptr());
            let a2 = vld1q_f32(ap[p * MR_12X8 + 8..].as_ptr());

            fma_row!(acc[0], a0, 0, b0, b1);
            fma_row!(acc[1], a0, 1, b0, b1);
            fma_row!(acc[2], a0, 2, b0, b1);
            fma_row!(acc[3], a0, 3, b0, b1);
            fma_row!(acc[4], a1, 0, b0, b1);
            fma_row!(acc[5], a1, 1, b0, b1);
            fma_row!(acc[6], a1, 2, b0, b1);
            fma_row!(acc[7], a1, 3, b0, b1);
            fma_row!(acc[8], a2, 0, b0, b1);
            fma_row!(acc[9], a2, 1, b0, b1);
            fma_row!(acc[10], a2, 2, b0, b1);
            fma_row!(acc[11], a2, 3, b0, b1);

            p += 1;
        }

        for (i, acc_i) in acc.iter().enumerate() {
            vst1q_f32(c_tile[i * NR_12X8..].as_mut_ptr(), acc_i[0]);
            vst1q_f32(c_tile[i * NR_12X8 + 4..].as_mut_ptr(), acc_i[1]);
        }
    }
}

/// [`kernel_with_ldc`] の従来シグネチャ後方互換ラッパー（`ldc = NR` 固定・
/// 密パッキング契約）。新規呼び出し元は `ldc` を明示できる
/// [`kernel_with_ldc`] を使うこと。
///
/// ## 戻り値非破壊（#691 レビュー P1 再指摘への対応）
///
/// [`super::scalar::kernel`] のドキュメント参照。本関数も同じ理由で
/// 従来どおり `()` を返す必須シグネチャへ戻し、`check_c_tile_bounds` の
/// `Result` を `panic!` へ変換する経路は持たない（`compute` へ直接
/// 委譲する。契約違反時の挙動は #557 以前と同一）。`ap`／`bp` の長さ検査は
/// [`super::panel_len_matches`] で `checked_mul` によりオーバーフローも
/// 確実に不一致として扱う（#691 レビュー P0 再指摘
/// `PRRT_kwDOTuUCJc6ZrXKs`）。`c_tile.len() == MR * NR` の検査は `ldc`
/// 一般化のリファクタで一時的に失われていたが、`compute` への委譲前に
/// 明示的に復元した（#691 レビュー再指摘 cursor
/// `PRRT_kwDOTuUCJc6ZrXO1`: `compute` は生ポインタの NEON ロード/ストア
/// であり範囲チェックを経ないため、検査なしでは未定義動作に到達しうる）。
pub fn kernel(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    assert!(
        super::panel_len_matches(ap.len(), MR, kc_len),
        "packed A panel length mismatch (or MR*kc_len overflow): ap.len()={}, MR={MR}, kc_len={kc_len}",
        ap.len()
    );
    assert!(
        super::panel_len_matches(bp.len(), kc_len, NR),
        "packed B panel length mismatch (or kc_len*NR overflow): bp.len()={}, kc_len={kc_len}, NR={NR}",
        bp.len()
    );
    assert_eq!(c_tile.len(), MR * NR, "C tile length mismatch");
    compute(ap, bp, c_tile, NR, kc_len);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift32 による疑似乱数ベクトル生成（テスト専用・本体非依存。
    /// [`super::avx2`] の同名関数のドキュメントコメント参照）。
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

    /// 手計算 2x2（MR/NR=8 タイルの左上 2x2 のみ使用・残りはゼロ）で
    /// FMA 累積が正しいことを確認する（scalar.rs の同種テストと同じ
    /// ケース。aarch64 実機／エミュレーションでのみ実行される）。
    #[test]
    fn kernel_matches_hand_computed_subset() {
        let kc_len = 2;
        let mut ap = vec![0.0f32; MR * kc_len];
        let mut bp = vec![0.0f32; kc_len * NR];
        ap[0] = 1.0;
        ap[1] = 3.0;
        ap[MR] = 2.0;
        ap[MR + 1] = 4.0;
        bp[0] = 5.0;
        bp[1] = 6.0;
        bp[NR] = 7.0;
        bp[NR + 1] = 8.0;

        let mut c_tile = vec![0.0f32; MR * NR];
        kernel(&ap, &bp, &mut c_tile, kc_len);

        assert_eq!(c_tile[0], 19.0);
        assert_eq!(c_tile[1], 22.0);
        assert_eq!(c_tile[NR], 43.0);
        assert_eq!(c_tile[NR + 1], 50.0);
    }

    /// #557: `ldc > NR`（完全タイル C 直接経路の想定）でも `ldc = NR`
    /// と bit 完全一致し、ギャップ列を破壊しないことを検証する（scalar.rs／
    /// avx2.rs の同種テストと同一パターン）。
    #[test]
    fn kernel_with_larger_ldc_matches_tight_packing_and_preserves_gap() {
        let kc_len = 5;
        let ap = xorshift32_vec(0xE0FF_EE01, MR * kc_len);
        let bp = xorshift32_vec(0xE0FF_EE02, kc_len * NR);
        let c_init = xorshift32_vec(0xE0FF_EE03, MR * NR);

        let mut c_tight = c_init.clone();
        kernel_with_ldc(&ap, &bp, &mut c_tight, NR, kc_len).unwrap();

        let ldc = NR + 5;
        let sentinel = -777.0f32;
        let mut c_gapped = vec![sentinel; (MR - 1) * ldc + ldc];
        for i in 0..MR {
            c_gapped[i * ldc..i * ldc + NR].copy_from_slice(&c_init[i * NR..i * NR + NR]);
        }
        kernel_with_ldc(&ap, &bp, &mut c_gapped, ldc, kc_len).unwrap();

        for i in 0..MR {
            for j in 0..NR {
                assert_eq!(
                    c_gapped[i * ldc + j],
                    c_tight[i * NR + j],
                    "ldc={ldc} 経路と ldc=NR 経路は bit 完全一致するはず（i={i}, j={j}）"
                );
            }
            for j in NR..ldc {
                assert_eq!(
                    c_gapped[i * ldc + j],
                    sentinel,
                    "ギャップ列（i={i}, j={j}）は直接ストアで破壊されてはならない"
                );
            }
        }
    }

    /// `ldc < NR` は panic ではなく `Result::Err` として返る（#691
    /// レビュー P1 再指摘への対応。従来は `should_panic` テストだった）。
    #[test]
    fn kernel_rejects_ldc_smaller_than_nr() {
        let ap = vec![0.0f32; MR * 2];
        let bp = vec![0.0f32; 2 * NR];
        let mut c_tile = vec![0.0f32; MR * NR];
        let err = kernel_with_ldc(&ap, &bp, &mut c_tile, NR - 1, 2).unwrap_err();
        assert_eq!(
            err,
            super::super::TileBoundsError::LdcTooSmall {
                ldc: NR - 1,
                nr: NR
            }
        );
    }

    // --- B 側レーン参照 FMA 変種（イシュー #748）のローカル検証 ---
    //
    // 本イシューの最重要ローカル検証は [`kernel_with_ldc`]（A レーン参照・
    // 既定）との bit 完全一致（`compute_b_laneq_matches_compute_bit_exact`）
    // である。既定カーネルは一切変更していないため、既存 3 テスト
    // （手計算・ldc 拡張・ldc 過小エラー）は無変更のまま上に残す。

    /// 手計算 2x2（[`kernel_matches_hand_computed_subset`] と同一ケース）で
    /// [`kernel_b_laneq_with_ldc`] の FMA 累積が正しいことを確認する。
    #[test]
    fn kernel_b_laneq_matches_hand_computed_subset() {
        let kc_len = 2;
        let mut ap = vec![0.0f32; MR * kc_len];
        let mut bp = vec![0.0f32; kc_len * NR];
        ap[0] = 1.0;
        ap[1] = 3.0;
        ap[MR] = 2.0;
        ap[MR + 1] = 4.0;
        bp[0] = 5.0;
        bp[1] = 6.0;
        bp[NR] = 7.0;
        bp[NR + 1] = 8.0;

        let mut c_tile = vec![0.0f32; MR * NR];
        kernel_b_laneq_with_ldc(&ap, &bp, &mut c_tile, NR, kc_len).unwrap();

        assert_eq!(c_tile[0], 19.0);
        assert_eq!(c_tile[1], 22.0);
        assert_eq!(c_tile[NR], 43.0);
        assert_eq!(c_tile[NR + 1], 50.0);
    }

    /// [`kernel_b_laneq_with_ldc`]（B 側レーン参照・列優先 acc）は
    /// [`kernel_with_ldc`]（既定・A 側レーン参照・行優先 acc）と
    /// bit 完全一致する（冒頭 #748 節・[`compute_b_laneq`] のドキュメント
    /// コメントで述べた FMA 乗数可換性による契約の実機ローカル検証）。
    /// k%4 の剰余（0/1/2/3）を網羅する kc_len を用いる。
    #[test]
    fn compute_b_laneq_matches_compute_bit_exact() {
        for (i, &kc_len) in [4usize, 5, 6, 7, 32, 33, 34, 35].iter().enumerate() {
            let ap = xorshift32_vec(0xB1A2_0001 + i as u32, MR * kc_len);
            let bp = xorshift32_vec(0xB1A2_0002 + i as u32, kc_len * NR);
            let c_init = xorshift32_vec(0xB1A2_0003 + i as u32, MR * NR);

            let mut c_default = c_init.clone();
            kernel_with_ldc(&ap, &bp, &mut c_default, NR, kc_len).unwrap();

            let mut c_b_laneq = c_init.clone();
            kernel_b_laneq_with_ldc(&ap, &bp, &mut c_b_laneq, NR, kc_len).unwrap();

            assert_eq!(
                c_default, c_b_laneq,
                "kernel_b_laneq_with_ldc（B レーン参照・kc_len={kc_len}）は \
                 kernel_with_ldc（既定・A レーン参照）と bit 完全一致するはず"
            );
        }
    }

    /// NaN を含む入力に対し [`kernel_b_laneq_with_ldc`] がパニックせず完了する
    /// ことのみを確認する（#765 codex-review P1 再指摘対応）。冒頭モジュール
    /// コメント §748 節で明記した通り、NaN 入力に対する
    /// [`kernel_with_ldc`]（A レーン参照）との bit 完全一致は主張しない
    /// （IEEE-754-2008 §6.2: 両オペランドが NaN の場合にどちらの NaN
    /// payload/符号を返すかは実装依存であり、乗数順序を交換する本変種が
    /// 同一の選択規則に従う保証がないため）。よって本テストは
    /// `assert_eq!` による bit 一致比較を行わず、NaN 伝播により C タイルの
    /// 該当要素が NaN になること（非 NaN へ「消える」誤り方はしないこと）
    /// のみを検査する。
    #[test]
    fn compute_b_laneq_nan_input_does_not_panic() {
        let kc_len = 4;
        let mut ap = vec![1.0f32; MR * kc_len];
        let bp = vec![1.0f32; kc_len * NR];
        ap[0] = f32::NAN;
        let mut c_tile = vec![0.0f32; MR * NR];

        // パニックしないことが本テストの主目的（ldc=NR の密パッキング契約）。
        kernel_b_laneq_with_ldc(&ap, &bp, &mut c_tile, NR, kc_len).unwrap();

        // ap[0] は A の行 0・p=0 に対応し、B レーン参照変種では列優先 acc へ
        // 転置されるため C の行 0・全列（j=0..NR）が p=0 の FMA で NaN に
        // 汚染される。NaN が汚染前提の要素へ伝播していること（消失しない
        // こと）のみを確認し、payload/符号の bit 一致は検査しない。
        for (j, &v) in c_tile.iter().enumerate().take(NR) {
            assert!(
                v.is_nan(),
                "NaN 入力（ap[0]）は C の行 0・列 {j} へ伝播するはず"
            );
        }
    }

    /// [`kernel_b_laneq_with_ldc`] も `ldc > NR`（完全タイル C 直接経路）で
    /// `ldc = NR` と bit 完全一致し、ギャップ列を破壊しないことを検証する
    /// （[`kernel_with_larger_ldc_matches_tight_packing_and_preserves_gap`]
    /// と同一パターン）。
    #[test]
    fn kernel_b_laneq_with_larger_ldc_matches_tight_packing_and_preserves_gap() {
        let kc_len = 5;
        let ap = xorshift32_vec(0xB1A2_1001, MR * kc_len);
        let bp = xorshift32_vec(0xB1A2_1002, kc_len * NR);
        let c_init = xorshift32_vec(0xB1A2_1003, MR * NR);

        let mut c_tight = c_init.clone();
        kernel_b_laneq_with_ldc(&ap, &bp, &mut c_tight, NR, kc_len).unwrap();

        let ldc = NR + 5;
        let sentinel = -777.0f32;
        let mut c_gapped = vec![sentinel; (MR - 1) * ldc + ldc];
        for i in 0..MR {
            c_gapped[i * ldc..i * ldc + NR].copy_from_slice(&c_init[i * NR..i * NR + NR]);
        }
        kernel_b_laneq_with_ldc(&ap, &bp, &mut c_gapped, ldc, kc_len).unwrap();

        for i in 0..MR {
            for j in 0..NR {
                assert_eq!(
                    c_gapped[i * ldc + j],
                    c_tight[i * NR + j],
                    "ldc={ldc} 経路と ldc=NR 経路は bit 完全一致するはず（i={i}, j={j}）"
                );
            }
            for j in NR..ldc {
                assert_eq!(
                    c_gapped[i * ldc + j],
                    sentinel,
                    "ギャップ列（i={i}, j={j}）は直接ストアで破壊されてはならない"
                );
            }
        }
    }

    /// [`kernel_b_laneq_with_ldc`] も `ldc < NR` を panic ではなく
    /// `Result::Err` として返す（[`kernel_rejects_ldc_smaller_than_nr`]
    /// と同一パターン）。
    #[test]
    fn kernel_b_laneq_rejects_ldc_smaller_than_nr() {
        let ap = vec![0.0f32; MR * 2];
        let bp = vec![0.0f32; 2 * NR];
        let mut c_tile = vec![0.0f32; MR * NR];
        let err = kernel_b_laneq_with_ldc(&ap, &bp, &mut c_tile, NR - 1, 2).unwrap_err();
        assert_eq!(
            err,
            super::super::TileBoundsError::LdcTooSmall {
                ldc: NR - 1,
                nr: NR
            }
        );
    }
}
