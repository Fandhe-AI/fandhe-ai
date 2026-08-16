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

use std::arch::aarch64::{float32x4_t, vfmaq_laneq_f32, vld1q_f32, vst1q_f32};

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
/// # Panics
///
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR`／
/// `c_tile.len() != MR * NR` のいずれかであればパニックする（REQ-8
/// 境界検査規約: 呼び出し元契約を関数入口で明示検査し、以降の
/// `unsafe` ロード／ストアはこの検査済み長さの範囲内でのみ行う）。
pub fn kernel(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    assert_eq!(c_tile.len(), MR * NR, "C tile length mismatch");

    // SAFETY: 直前の assert により ap は MR*kc_len 要素、bp は kc_len*NR
    // 要素、c_tile は MR*NR(=96) 要素ちょうどであることが保証されている。
    // 以下のロード／ストアはいずれもこの範囲内のオフセットに限定される:
    // - bp: p*NR, p*NR+4, p*NR+8..+12 の最大は p=kc_len-1 でも
    //   (kc_len-1)*NR+12 = bp.len() を超えない。
    // - ap: p*MR, p*MR+4..+8 の最大は p=kc_len-1 でも (kc_len-1)*MR+8 =
    //   ap.len() を超えない。
    // - c_tile: i*NR, i*NR+4, i*NR+8..+12 の最大は i=MR-1 でも
    //   (MR-1)*NR+12 = c_tile.len() を超えない。
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
                vld1q_f32(c_tile[i * NR..].as_ptr()),
                vld1q_f32(c_tile[i * NR + 4..].as_ptr()),
                vld1q_f32(c_tile[i * NR + 8..].as_ptr()),
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
            vst1q_f32(c_tile[i * NR..].as_mut_ptr(), acc_i[0]);
            vst1q_f32(c_tile[i * NR + 4..].as_mut_ptr(), acc_i[1]);
            vst1q_f32(c_tile[i * NR + 8..].as_mut_ptr(), acc_i[2]);
        }
    }
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
    assert_eq!(ap.len(), MR_12X8 * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR_12X8, "packed B panel length mismatch");
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
