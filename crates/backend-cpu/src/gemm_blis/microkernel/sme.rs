//! aarch64 SME（Scalable Matrix Extension）マイクロカーネル
//! （MR=16×NR=16、`fmopa` 非拡張 FP32 外積・ZA0 単一タイル。イシュー #1587）。
//!
//! **モジュールは `cfg(target_arch = "aarch64")` のみでコンパイルする**
//! （`super::avx2`〈x86_64 限定のためコードスパン表記〉と同じ理由: モジュール単位で追加条件を課すと、
//! テスト限定の実行時検出ガード付き直接検証が行えなくなるため）。実際に
//! SME 命令を発行する `compute`（非公開関数のためコードスパン表記）は
//! 「呼び出し元が実行 CPU の SME 対応（`super::SmeKernel::try_new`。
//! `pub(crate)` のためコードスパン表記）経由の実行時検出）を保証する」契約の
//! `unsafe fn` とし、コンパイル時 `target_feature` によるゲートは行わない
//! （SME はコンパイラ intrinsics ではなく生アセンブリで発行するため、
//! `#[target_feature(enable = "sme")]` は不要かつ rustc stable では
//! 認識されない）。
//!
//! ## 設計判断: MR=16×NR=16・ZA0 単一タイル
//!
//! SVL=512 bit（f32 ベクトルレジスタ 1 本 = 16 要素）環境では ZA0〜ZA3 の
//! 4 タイルを使う MR=32×NR=32 案（2×2 外積ブロック）も可能だが、本実装は
//! `super::super::MAX_TILE`（非公開定数のためコードスパン表記。256 要素。
//! 全 ISA 共通の端タイル用スタック
//! バッファ長）を変更せず既存 ISA（GB10 の NEON 経路等）へ副作用を
//! 与えないことを優先し、ZA0 単一タイル（MR*NR=16*16=256=MAX_TILE で
//! ちょうど収まる）の MR=16×NR=16 を採用する（計画リスク §10「フォール
//! バック案」）。将来 32×32（4 タイル）を追加する場合は `MAX_TILE` 拡大の
//! 副作用（GB10 NEON 端タイル計測）を別途実施すること。
//!
//! ## bit 完全一致契約（REQ-2・`.claude/rules/coding-rust.md` FMA 契約統一）
//!
//! `fmopa za0.s, p0/m, p0/m, zn.s, zm.s` は Arm ARM（DDI0616）により
//! 「非拡張 FP32 外積: 各要素 `za[i][j] = fma(zn[i], zm[j], za[i][j])`
//! （単一丸めの fused multiply-add）」と定義される。`compute`（非公開
//! 関数のためコードスパン表記）は
//! ループ前に **C の現在値を ZA0 へプリロード**し、`p` を昇順に走査して
//! `kc_len` 回 `fmopa` を発行した後に ZA0 を C へストアし直す（zero-init
//! して最後に加算する方式は丸めが 1 回増え不一致になるため採らない）。
//! これにより各 `c[i][j]` は「初期値 = 呼び出し時点の `c[i][j]`、p 昇順に
//! 1 回ずつ `fma(a[p][i], b[p][j], acc)`」という演算列になり、
//! [`super::neon`] の `vfmaq_laneq_f32(acc, b, a, lane)` = `acc + b*a[lane]`
//! （単一 FMA・p 昇順・レーン間縮約なし）と**乗算の可換性（IEEE-754-2008
//! §5.4.1。有限値に限る）を除いて演算列が完全に同一**になる。有限値
//! 入力での bit 完全一致は `tests/gemm_blis_parity.rs`（本番入口経由）・
//! 本モジュール下部の単体テスト（scalar 参照・NEON 参照との直接比較）で
//! 検証する。
//!
//! NaN を含む入力については、両オペランドが異なる payload の NaN のとき
//! IEEE-754-2008 §6.2 の NaN 選択規則が `fma(a,b,c)` と
//! `mul_add`／`vfmaq_laneq_f32` の実装間で一致する保証がない（`neon`
//! モジュール `compute_b_laneq` §748 節と同じ理由）ため、NaN 混入入力に
//! 対する bit 一致は主張しない（panic なし・NaN 位置一致のみを診断的に
//! 確認する）。
//!
//! ## 検出との関係
//!
//! 本モジュールの関数はいずれも「実行 CPU が SME・非拡張 FP32 外積
//! （`SME_F32F32`）に対応し、かつ SVL=64 バイト（512 bit）である」ことを
//! 呼び出し元契約とする（`super::SmeKernel::try_new`（`pub(crate)`
//! のためコードスパン表記）。
//! `crate::sme_detect::sme_report()` が fail-closed に判定する）。

use std::arch::asm;

/// マイクロカーネルタイルの行数（ZA0 の行数。SVL=512 bit 前提）。
pub const MR: usize = 16;
/// マイクロカーネルタイルの列数（ZA0 の列数）。
pub const NR: usize = 16;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素）固定長で確保するため、コンパイル時に検査する
// （MR*NR=256 でちょうど一致。他 ISA と同型の契約）。
const _: () = assert!(MR * NR <= 256);
const _: () = assert!(MR == 16 && NR == 16);

/// AAPCS64／ACLE の private-ZA 関数契約（Arm ACLE
/// <https://arm-software.github.io/acle/main/acle.html#inline-assembly>
/// 「TPIDR2 block」節）に基づき、呼び出し元スレッドに**保留中の lazy ZA
/// save が無い**（`TPIDR2_EL0 == 0`）ことを確認する（codex-review P0
/// 指摘 `PRRT_kwDOTuUCJc6h0ZMD` 対応）。
///
/// ## 背景（なぜ必要か）
///
/// `TPIDR2_EL0` が非ゼロの場合、それは「呼び出し元（さらに上位の
/// private-ZA 関数）が ZA を dormant 状態（PSTATE.ZA=1 のまま
/// ハードウェアへ実体を残し、OS へは未保存）にして本関数を呼んだ」こと
/// を意味する。この状態を無視して [`compute`] が `smstart`／`fmopa`／
/// `mova` で ZA0 を上書きすると、呼び出し元が後で参照するはずだった
/// ZA の内容を silently に破壊する（TPIDR2 ブロックに登録された
/// `za_save_buffer` へは誰も保存しないまま失われる）。
///
/// ## 採用方式（fail-closed。ACLE 規定の lazy save 完了処理は未実装）
///
/// ACLE は「ZA を変更する前に TPIDR2 ブロック（`za_save_buffer`
/// ポインタ・`num_za_save_slices` 等）に従い保留中の save を完了し
/// `TPIDR2_EL0` をゼロクリアする」ことを private-ZA 関数の責務として
/// 規定するが、この完了処理（SVL 依存の行単位ストアループ・ブロック
/// レイアウトの精密な解釈）を検証手段なしに実装すると誤りが実機検証
/// なしには判別できない。本関数は代わりに `TPIDR2_EL0 != 0` を検出
/// する**判定のみ**を提供し、呼び出し元（[`super::current_thread_capable`]
/// 等）はこの判定が真の場合、`compute` を一切呼ばず（＝ZA に一切触れず）
/// 安全な [`scalar_fallback`] へ切り替える（本モジュール冒頭「bit 完全
/// 一致契約」節の演算列を維持したまま安全側に倒す設計。
/// `.claude/rules/security.md` の fail-closed 方針）。この設計は
/// 「lazy save を壊さない」という AAPCS64 契約そのものは満たすが、
/// ZA を伴う私的呼び出し規約の完全実装（save 完了）ではない点に注意する。
///
/// # Safety
///
/// 呼び出し元は実行 CPU が SME 対応（`TPIDR2_EL0` は `FEAT_SME` の一部で
/// あり、`crate::sme_detect::sme_report().kernel_enabled` が真の場合に
/// 限り読み取り可能。非対応 CPU でこのシステムレジスタへアクセスすると
/// UNDEFINED 命令例外〈実態は SIGILL〉になりうる）ことを保証しなければ
/// ならない。
pub(crate) unsafe fn has_pending_lazy_za_save() -> bool {
    let tpidr2: u64;
    // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により実行 CPU は
    // SME 対応。`mrs` はメモリアクセス・スタック使用を伴わない読み取り
    // 専用命令で、フラグレジスタも変更しない
    // （`options(nomem, nostack, preserves_flags)` の宣言どおり）。
    //
    // レジスタ名は記号名 `tpidr2_el0` ではなく符号化名
    // `S3_3_C13_C0_5`（op0=3, op1=3, CRn=13, CRm=0, op2=5。Arm ARM の
    // システムレジスタエンコーディング表参照）を使う。記号名の
    // 認識可否はアセンブラ（LLVM integrated assembler）のバージョンに
    // 依存しうるのに対し、符号化名は常に解釈可能なためツールチェーン
    // 間の移植性が高い（実機〈Apple M4 Max〉で両形式とも動作確認済みだが
    // 符号化名を採用する）。
    unsafe {
        asm!(
            ".arch_extension sme",
            "mrs {0}, S3_3_C13_C0_5",
            out(reg) tpidr2,
            options(nomem, nostack, preserves_flags),
        );
    }
    tpidr2 != 0
}

/// [`kernel_unchecked_with_ldc`]／[`kernel_unchecked`] 共通の演算本体。
///
/// # Safety
///
/// 呼び出し元は次を保証しなければならない:
/// - 実行 CPU が SME・非拡張 FP32 外積（`SME_F32F32`）に対応し、SVL が
///   64 バイト（512 bit。`crate::sme_detect` の `REQUIRED_SVL_BYTES` と
///   同じ値）であること（`smstart` 後の
///   `fmopa`／`mova`／`ld1w`／`st1w` がこの前提でのみ健全）。
/// - [`has_pending_lazy_za_save`] が `false`（`TPIDR2_EL0 == 0`）で
///   あることを**この呼び出しの直前**に確認済みであること（本関数
///   ドキュメント「背景」節。呼び出し元が ZA を dormant のまま本関数を
///   呼ぶと、`smstart`／`mova` による ZA0 上書きが呼び出し元の保留中
///   lazy save を破壊する）。
/// - `ap.len() == MR * kc_len`・`bp.len() == kc_len * NR`（p-major packing。
///   `pack_a`／`pack_b` の `dst[p*mr+i]`／`dst[p*nr+j]` 契約）。
/// - `c.len() >= (MR - 1) * ldc + NR`（`ldc >= NR`）。
unsafe fn compute(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    // 呼び出し元契約（本関数の `# Safety` 節）により、以下のロード／
    // ストアはいずれもこの範囲内のオフセットに限定される:
    // - bp: p*NR の最大は p=kc_len-1 でも (kc_len-1)*NR+NR = bp.len() を
    //   超えない。
    // - ap: p*MR の最大は p=kc_len-1 でも (kc_len-1)*MR+MR = ap.len() を
    //   超えない。
    // - c（プリロード・ストア両ループとも i in 0..MR）: 最大オフセットは
    //   (MR-1)*ldc+NR <= c.len()（`ldc >= NR` は呼び出し元契約）。
    let c_ptr = c.as_mut_ptr();
    let a_ptr = ap.as_ptr();
    let b_ptr = bp.as_ptr();
    let ldc_bytes = ldc * size_of::<f32>();
    let kc = kc_len;

    // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により実行 CPU は
    // SME・非拡張 FP32 外径（`SME_F32F32`）に対応し SVL=64 バイト。
    //
    // レジスタ・状態契約（`docs/cpu-gemm-sme-fmopa-microkernel.md` §3 の
    // asm 契約節を参照。1 箇所に局所化）:
    // - `.arch_extension sme` はアセンブラへ SME 命令の使用を許可する
    //   ディレクティブ（コンパイル時のみに影響）。
    // - `smstart`/`smstop` を 1 つの asm! ブロック内で対にして閉じ、
    //   ブロック内で SME 以外の SIMD 命令を混在させない。
    // - `smstart` は Z0-Z31/P0-P15（および FFR）の内容を不定化するため
    //   `out("v0") _ ... out("v31") _`・`out("p0") _ ... out("p15") _` を
    //   全列挙する（FFR は本ルーチンが `ldff1`/`ldnf1` 系の投機ロードを
    //   使わないため未使用・未依存。値の破棄自体は smstart/smstop の
    //   ハードウェア契約でありコンパイラへ追加の宣言余地はない）。
    // - `mova` のスライスインデックスレジスタは w12-w15 限定のため
    //   `out("w12") _` を明示する。
    // - `subs`/`cmp`/`b.lt`/`b.ne`/`cbz` でフラグを書き換えるため
    //   `preserves_flags` は付けない。
    // - `ld1w`/`st1w`（通常のメモリアクセス）を使うため `nomem`/`pure`
    //   は付けない。スタックを使わないため `options(nostack)`。
    // - ポインタ演算（`ldc_bytes`）は asm 外で `usize` 乗算により確定し、
    //   asm へは検査済みバイトストライドのみ渡す（呼び出し元契約が
    //   保証する範囲内でのみ加算するため、桁あふれの検査は本関数の
    //   `# Safety` 契約〈呼び出し元が渡す `ap`/`bp`/`c` の長さ検査〉に
    //   委ねる。呼び出し元 `kernel_with_ldc`/`kernel_unchecked_with_ldc`
    //   はいずれも `super::check_panel_lengths`/`check_c_tile_bounds`
    //   〈`checked_mul`/`checked_add` 使用〉を先に通す。REQ-8）。
    // - C プリロード（`cpre`）・ストア（`cpost`）は同一の初期ポインタ値
    //   （`c_ptr`）から独立に 2 つの汎用レジスタへ展開し、各ループ内で
    //   `ldc_bytes` ずつ進める（プリロードループがポインタを消費しても
    //   ストアループ用の元ポインタが失われないようにするため）。
    unsafe {
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            // C プリロード: 16 行を za0h.s[0..16] へロードする
            // （行 i は c_ptr + i*ldc_bytes から NR=16 要素）。
            "mov w12, #0",
            "10:",
            "ld1w {{z0.s}}, p0/z, [{cpre}]",
            "mova za0h.s[w12, #0], p0/m, z0.s",
            "add {cpre}, {cpre}, {ldc_bytes}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lt 10b",
            // k ループ: kc_len == 0 ならスキップ（za0 は C の値のまま）。
            // `kc` は `usize`（64 bit）のフルレジスタ（`{kc}`＝X レジスタ）で
            // 扱う。`cbz`/`subs` の 32 bit（`{kc:w}`＝W レジスタ）版は
            // `kc_len` を暗黙的に下位 32 bit へ切り詰めるため、
            // `kc_len > u32::MAX` の呼び出し（`BlockSizes::kc` が極端に
            // 大きい場合。理論上は `usize` の契約上あり得る）で無音に
            // 誤った反復回数になりうる（advisor レビュー指摘。到達可能な
            // `kc_len` は `blocks.kc` で事実上有界だが、SAFETY 契約を
            // レジスタ幅の暗黙の仮定に依存させない）。
            "cbz {kc}, 12f",
            "11:",
            "ld1w {{z1.s}}, p0/z, [{a}]",
            "ld1w {{z2.s}}, p0/z, [{b}]",
            "fmopa za0.s, p0/m, p0/m, z1.s, z2.s",
            "add {a}, {a}, #64",
            "add {b}, {b}, #64",
            "subs {kc}, {kc}, #1",
            "b.ne 11b",
            "12:",
            // ストア: za0h.s[0..16] を C の 16 行へ書き戻す。
            "mov w12, #0",
            "13:",
            "mova z3.s, p0/m, za0h.s[w12, #0]",
            "st1w {{z3.s}}, p0, [{cpost}]",
            "add {cpost}, {cpost}, {ldc_bytes}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lt 13b",
            "smstop",
            cpre = inout(reg) c_ptr => _,
            cpost = inout(reg) c_ptr => _,
            a = inout(reg) a_ptr => _,
            b = inout(reg) b_ptr => _,
            kc = inout(reg) kc => _,
            ldc_bytes = in(reg) ldc_bytes,
            out("w12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
            out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _,
            out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _,
            out("p15") _,
            out("ffr") _,
            options(nostack),
        );
    }
}

/// [`super::TileBoundsError`] 検査つきの `ldc` 契約版（`compute`〈非公開
/// 関数のためコードスパン表記〉へ委譲）。[`super::neon::kernel_with_ldc`]
/// と同型の入口だが、SME は実行時検出済みトークン（[`super::SmeKernel`]）
/// 経由でのみ安全に呼べるため `unsafe fn` とする（`super::avx2::kernel_unchecked_with_ldc`。
/// x86_64 限定のためコードスパン表記）と同型）。
///
/// ## 保留中 lazy ZA save の自己防御（codex-review P0 再指摘対応）
///
/// 本関数は `super::SmeKernel::run_with_ldc`〈`pub(crate)` のため
/// コードスパン表記〉経由に限らず外部から直接呼び出しうる `pub unsafe fn`
/// のため、`# Safety` 契約が SME 対応・SVL のみを要求し
/// `has_pending_lazy_za_save`〈非公開関数のためコードスパン表記〉の事前
/// 確認を呼び出し元の努力目標にとどめる設計では、契約を字面どおり満たした
/// 呼び出しでも保留中の lazy ZA save を破壊しうる（本モジュール「背景」
/// 節）。そこで本関数自身が `has_pending_lazy_za_save` を検査し、非ゼロ
/// （保留中）であれば `compute`〈非公開関数のためコードスパン表記〉を
/// 一切呼ばず（＝ZA に一切触れず）`scalar_fallback`〈非公開関数のため
/// コードスパン表記〉（`compute` と同一の演算列で bit 完全一致）へ
/// 切り替える。
///
/// # Safety
///
/// 呼び出し元は実行 CPU が SME・非拡張 FP32 外積（`SME_F32F32`）に対応し
/// SVL=`crate::sme_detect::REQUIRED_SVL_BYTES`（非公開定数のため
/// コードスパン表記。64 バイト）であることを保証しなければならない
/// （`super::SmeKernel::try_new`〈`pub(crate)` のためコードスパン表記〉
/// 経由の実行時検出済み呼び出しがこれを
/// 満たす）。**`TPIDR2_EL0` が非ゼロ（保留中の lazy ZA save あり）の場合は
/// 本関数が自動的にフォールバックへ切り替えるため、呼び出し元が事前に
/// `has_pending_lazy_za_save`〈非公開関数のためコードスパン表記〉を確認
/// する必要はない**（上記「保留中 lazy ZA save の自己防御」節）。
pub unsafe fn kernel_unchecked_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    super::check_panel_lengths(MR, NR, kc_len, ap.len(), bp.len())?;
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により実行 CPU は
    // SME 対応（`TPIDR2_EL0` へのアクセスが安全。`has_pending_lazy_za_save`
    // の `# Safety` 契約を満たす）。
    if unsafe { has_pending_lazy_za_save() } {
        // 保留中の lazy ZA save を検出。`compute` を呼ばず ZA に一切
        // 触れないフォールバックへ切り替える（上記「保留中 lazy ZA save
        // の自己防御」節）。長さ・境界は直前の検査で確認済み。
        scalar_fallback(ap, bp, c, ldc, kc_len);
        return Ok(());
    }
    // SAFETY: [`compute`] のドキュメント参照（直前の検査により長さ前提を
    // 満たし、SME 対応は本関数の呼び出し元契約として引き継ぐ。上記検査で
    // 保留中の lazy ZA save がないことも確認済み）。
    unsafe { compute(ap, bp, c, ldc, kc_len) };
    Ok(())
}

/// 従来シグネチャ後方互換ラッパー（`ldc = NR` 固定・密パッキング契約。
/// `super::avx2::kernel_unchecked`（x86_64 限定のためコードスパン表記）
/// と同型）。
///
/// ## 保留中 lazy ZA save の自己防御
///
/// [`kernel_unchecked_with_ldc`] と同じ理由・同じ方式で
/// `has_pending_lazy_za_save`〈非公開関数のためコードスパン表記〉を自己
/// 検査し、保留中であれば `scalar_fallback`〈非公開関数のためコード
/// スパン表記〉へ切り替える（同関数「保留中 lazy ZA save の自己防御」
/// 節参照）。
///
/// ## 長さ・境界検査を型付きエラー化（codex-review P1 再指摘対応）
///
/// 以前は本関数自身が `assert!`／`assert_eq!` で `ap`／`bp`／`c` の長さを
/// 検査していたため、`SmeKernel::run` を経由せずに本関数（`pub unsafe fn`）
/// を直接呼び出す外部コードへ panic が漏れ、「本番経路の panic 禁止」
/// （`.claude/rules/security.md`／AGENTS.md）に抵触していた。境界検査
/// 自体は維持しつつ（REQ-8 境界検査規約）、[`kernel_unchecked_with_ldc`]
/// （`super::check_panel_lengths`／`super::check_c_tile_bounds` を経由し
/// [`super::TileBoundsError`] を返す）へ `ldc = NR` で委譲する形へ変更
/// した（`assert!` の重複実装を持たない）。
///
/// # Safety
///
/// [`kernel_unchecked_with_ldc`] と同一（`TPIDR2_EL0` 非ゼロ時の
/// 自動フォールバックを含む）。
pub unsafe fn kernel_unchecked(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）を [`kernel_unchecked_with_ldc`]
    // へそのまま引き継ぐ。
    unsafe { kernel_unchecked_with_ldc(ap, bp, c, NR, kc_len) }
}

/// `unsafe { compute(...) }`（`fmopa` アセンブリ）を発行できない場合の
/// 安全な Rust フォールバック（イシュー #1587 codex-review P1 再指摘
/// `PRRT_kwDOTuUCJc6h0ZMD` への対応）。
///
/// `SmeKernel` は「構築したスレッド」の SME 対応・SVL を保証するのみで、
/// `Copy` により別スレッド（Rayon worker）へ渡された場合はそのスレッド
/// 自身の SVL が異なる（Linux では `prctl(PR_SME_SET_VL)` によりスレッド
/// ごとに変更可能）ことがある（[`super::SmeKernel`] doc 参照）。以前は
/// 実行スレッド自身の再確認に失敗した場合 `panic!` していたが、これは
/// `cfg(test)` 外の本番ライブラリコードであり
/// `.claude/rules/security.md`／AGENTS.md「本番経路の panic 禁止」に
/// 抵触する。本関数は `compute` の代わりに実行される安全なフォール
/// バックとして、`compute` と**同一の演算列**（p 昇順・要素ごとに 1 回
/// の `f32::mul_add` を適用し、レーン間の並べ替え・再結合を行わない）を
/// 再現する。これにより `fmopa`（Arm ARM DDI0616 の非拡張 FP32 外積:
/// `za[i][j] = fma(zn[i], zm[j], za[i][j])`）と本関数の結果は有限値
/// 入力で bit 完全一致する（モジュール冒頭「bit 完全一致契約」節参照。
/// `compute` が実行される経路と全く同一の呼び出し元契約〈`ap`／`bp`
/// の長さ・`ldc` 境界〉を要求するため、境界検査は呼び出し元
/// （[`kernel_unchecked_with_ldc`] 相当）にまかせず本関数自身でも
/// 行う）。
fn scalar_fallback(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    for p in 0..kc_len {
        for i in 0..MR {
            let a_val = ap[p * MR + i];
            for j in 0..NR {
                let idx = i * ldc + j;
                c[idx] = a_val.mul_add(bp[p * NR + j], c[idx]);
            }
        }
    }
}

/// [`scalar_fallback`] の `ldc` 契約版・境界検査つき入口（
/// [`kernel_unchecked_with_ldc`] と同型の検査を行い、`compute` の代わりに
/// [`scalar_fallback`] を呼ぶ。`unsafe` を含まないため `unsafe fn` では
/// ない）。
pub(crate) fn scalar_fallback_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    super::check_panel_lengths(MR, NR, kc_len, ap.len(), bp.len())?;
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    scalar_fallback(ap, bp, c, ldc, kc_len);
    Ok(())
}

/// [`scalar_fallback_with_ldc`] の従来シグネチャ版（`ldc = NR` 固定・
/// 密パッキング契約。[`kernel_unchecked`] と同型の長さ検査を行う）。
///
/// ## 長さ・境界検査を型付きエラー化（codex-review P1 再指摘対応）
///
/// [`kernel_unchecked`] と同じ理由により、`assert!`／`assert_eq!` の
/// 重複実装をやめ [`scalar_fallback_with_ldc`] へ `ldc = NR` で委譲する
/// （`unsafe` を含まないため呼び出し規約は変更せず、戻り値のみ
/// [`super::TileBoundsError`] を返す `Result` へ変更する）。
pub(crate) fn scalar_fallback_kernel(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    scalar_fallback_with_ldc(ap, bp, c, NR, kc_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm_blis::microkernel::{Microkernel, SmeKernel, TileBoundsError};

    /// [`kernel_unchecked`]（`pub unsafe fn`。`SmeKernel::run` を経由せず
    /// 外部から直接呼びうる）が、長さ不一致（`ap` 長不足）を `panic!` で
    /// はなく `Result::Err(TileBoundsError::PanelLengthMismatch)` として
    /// 返すことを確認する（codex-review P1 再指摘対応。本テストの主張は
    /// [`kernel_unchecked`] doc「長さ・境界検査を型付きエラー化」節）。
    ///
    /// 実機（SME 対応 aarch64）非依存で実行できる: `ap`／`bp`／`c` の長さ
    /// 検査（`super::check_panel_lengths`／`super::check_c_tile_bounds`）
    /// は [`has_pending_lazy_za_save`]（`TPIDR2_EL0` 読み取り。SME 対応
    /// CPU を要求）や `compute`（`fmopa` 発行）より前に必ず実行され、
    /// 不一致時はそれらの SME 命令発行前に early return するため、SME
    /// 非対応の aarch64 環境（本ファイルは `cfg(target_arch = "aarch64")`
    /// 限定でコンパイルされるため実行環境は常に aarch64）でも安全に
    /// 呼び出せる（本モジュール上部「その他の実機依存テスト」で使う
    /// `#[ignore]` は不要）。
    #[test]
    fn kernel_unchecked_returns_err_instead_of_panicking_on_ap_length_mismatch() {
        let kc_len = 3usize;
        // 本来必要な `MR * kc_len` より 1 要素少ない `ap`。
        let ap = vec![0.0f32; MR * kc_len - 1];
        let bp = vec![0.0f32; kc_len * NR];
        let mut c = vec![0.0f32; MR * NR];

        // SAFETY: 上記コメントのとおり、長さ検査は SME 命令発行より前に
        // 完了し `Err` で early return するため、実行 CPU の SME 対応
        // 有無に関わらず本呼び出しは安全に完了する。
        let result = unsafe { kernel_unchecked(&ap, &bp, &mut c, kc_len) };

        assert_eq!(
            result,
            Err(TileBoundsError::PanelLengthMismatch {
                panel: "ap",
                actual: ap.len()
            }),
            "panic ではなく Result::Err を返すはず"
        );
    }

    /// [`kernel_unchecked`] が C タイル長不一致（`c.len() != MR * NR`）も
    /// 同様に `Result::Err(TileBoundsError)` で返すことを確認する
    /// （[`kernel_unchecked_returns_err_instead_of_panicking_on_ap_length_mismatch`]
    /// と同型・実機非依存の理由も同一）。
    #[test]
    fn kernel_unchecked_returns_err_instead_of_panicking_on_c_length_mismatch() {
        let kc_len = 2usize;
        let ap = vec![0.0f32; MR * kc_len];
        let bp = vec![0.0f32; kc_len * NR];
        // 本来必要な `MR * NR` より短い `c`。
        let mut c = vec![0.0f32; MR * NR - 1];

        // SAFETY: `kernel_unchecked_returns_err_instead_of_panicking_on_ap_length_mismatch`
        // と同一の理由。
        let result = unsafe { kernel_unchecked(&ap, &bp, &mut c, kc_len) };

        assert!(
            result.is_err(),
            "panic ではなく Result::Err を返すはず: {result:?}"
        );
    }

    /// [`scalar_fallback_kernel`]（`unsafe` を含まない安全な公開
    /// フォールバック入口）も同じ長さ検査を経由し、`panic!` ではなく
    /// `Result::Err(TileBoundsError)` を返すことを確認する
    /// （codex-review P1 再指摘「scalar_fallback_kernel の 418・423・
    /// 428 行」対応）。
    #[test]
    fn scalar_fallback_kernel_returns_err_instead_of_panicking_on_length_mismatch() {
        let kc_len = 4usize;
        let ap = vec![0.0f32; MR * kc_len];
        // 本来必要な `kc_len * NR` より短い `bp`。
        let bp = vec![0.0f32; kc_len * NR - 1];
        let mut c = vec![0.0f32; MR * NR];

        let result = scalar_fallback_kernel(&ap, &bp, &mut c, kc_len);

        assert_eq!(
            result,
            Err(TileBoundsError::PanelLengthMismatch {
                panel: "bp",
                actual: bp.len()
            }),
            "panic ではなく Result::Err を返すはず"
        );
    }

    /// 有限値・非正規化数入力を含むスカラー参照（p 昇順 `f32::mul_add`
    /// 連鎖）との bit 完全一致を検証する下請け関数。以下の各テストは
    /// SME 実機（例: Apple M4）依存のため `#[ignore]` で分離する
    /// （`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]`
    /// で分離」。codex-review P1 指摘 `PRRT_kwDOTuUCJc6h0P7c` 対応）。
    /// `#[ignore]` 実行時点でも `SmeKernel::try_new()` が `None`（実行
    /// 環境が SME 非対応）を返す場合は実行時スキップする（非対応実機
    /// での `--ignored` 実行を green に保つための二重の安全策）。
    fn scalar_reference(
        ap: &[f32],
        bp: &[f32],
        c_init: &[f32],
        ldc: usize,
        kc_len: usize,
    ) -> Vec<f32> {
        let mut c = c_init.to_vec();
        for i in 0..MR {
            for j in 0..NR {
                let mut acc = c[i * ldc + j];
                for p in 0..kc_len {
                    acc = ap[p * MR + i].mul_add(bp[p * NR + j], acc);
                }
                c[i * ldc + j] = acc;
            }
        }
        c
    }

    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_matches_scalar_reference_finite_values() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        for &kc_len in &[0usize, 1, 3, 4, 17, 255, 256] {
            let ap = xorshift32_vec(0x1234_5678 ^ kc_len as u32, MR * kc_len);
            let bp = xorshift32_vec(0x9abc_def0 ^ kc_len as u32, kc_len * NR);
            let c_init = xorshift32_vec(0x1111_2222 ^ kc_len as u32, MR * NR);

            let mut c_sme = c_init.clone();
            kernel.run(&ap, &bp, &mut c_sme, kc_len);

            let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
            assert_eq!(
                c_sme, expected,
                "kc_len={kc_len} で scalar 参照と bit 完全一致するはず"
            );
        }
    }

    /// 端タイル相当（`ldc > NR`）での bit 完全一致（[`kernel_unchecked_with_ldc`]
    /// 経由）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_with_ldc_matches_scalar_reference_strided() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 37;
        let ldc = NR + 5;
        let ap = xorshift32_vec(0xaaaa_bbbb, MR * kc_len);
        let bp = xorshift32_vec(0xcccc_dddd, kc_len * NR);
        // `ldc` ストライドの C バッファ（行間ギャップあり）。
        let mut c = xorshift32_vec(0xeeee_ffff, (MR - 1) * ldc + ldc);
        let c_init = c.clone();

        kernel.run_with_ldc(&ap, &bp, &mut c, ldc, kc_len).unwrap();

        // `scalar_reference` は任意の `ldc` を尊重し `c[i*ldc+j]`
        // （`j in 0..NR`）のみを書き換える（ギャップ列 `j>=NR` は
        // 触れない）ため、ストライド付きバッファ全体を 1 回で計算できる。
        let expected = scalar_reference(&ap, &bp, &c_init, ldc, kc_len);
        assert_eq!(
            c, expected,
            "ldc={ldc} のストライド付きバッファ全体が scalar 参照と \
             bit 完全一致するはず（ギャップ領域が変更されないことを含む）"
        );
    }

    /// run-to-run bit 同一性（同一入力を 2 回実行して比較。R3(a)）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_is_deterministic_across_runs() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 129;
        let ap = xorshift32_vec(0x2468_1357, MR * kc_len);
        let bp = xorshift32_vec(0x1357_2468, kc_len * NR);
        let c_init = xorshift32_vec(0x0f0f_0f0f, MR * NR);

        let mut c1 = c_init.clone();
        kernel.run(&ap, &bp, &mut c1, kc_len);
        let mut c2 = c_init.clone();
        kernel.run(&ap, &bp, &mut c2, kc_len);

        assert_eq!(c1, c2, "同一入力の 2 回実行は bit 同一であるはず");
    }

    /// 非正規化数（約 1e-40）入力での scalar 参照との bit 完全一致
    /// （R3(b)。ストリーミングモードでの FPCR/FZ 差の検出を狙う）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_matches_scalar_reference_denormal_values() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 8;
        let denormal = 1e-40f32;
        assert!(
            denormal != 0.0 && denormal.abs() < f32::MIN_POSITIVE,
            "テスト前提: 1e-40 は非正規化数であるはず"
        );
        let ap = vec![denormal; MR * kc_len];
        let bp = vec![denormal; kc_len * NR];
        let c_init = vec![denormal; MR * NR];

        let mut c_sme = c_init.clone();
        kernel.run(&ap, &bp, &mut c_sme, kc_len);

        let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
        assert_eq!(
            c_sme, expected,
            "非正規化数入力でも scalar 参照と bit 完全一致するはず"
        );
    }

    /// 非正規化数「結果」の FTZ（flush-to-zero）差異を検出する
    /// （advisor レビュー指摘: 上記
    /// `sme_kernel_matches_scalar_reference_denormal_values` は入力が
    /// 非正規化数のケースのみを検証しており、`a=b=c=1e-40` では
    /// `fma(1e-40,1e-40,1e-40)` の積項が 0 へアンダーフローするため
    /// 「非正規化数の乗算結果がストリーミングモードでフラッシュされない
    /// こと」自体は検証できていなかった）。本テストは正規化数どうしの
    /// 積が非正規化数（`1e-20 * 1e-20 = 1e-40`）になる入力を使い、SME
    /// 側が FTZ でこの結果を 0 へ潰さず scalar 参照と bit 完全一致する
    /// ことを確認する（R3(b) が本来意図した FZ 判別）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_matches_scalar_reference_denormal_result_from_normal_operands() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 1;
        let normal_small = 1e-20f32;
        assert!(
            normal_small.is_normal(),
            "テスト前提: 1e-20 は正規化数であるはず"
        );
        let denormal_product = normal_small * normal_small;
        assert!(
            denormal_product != 0.0 && denormal_product.abs() < f32::MIN_POSITIVE,
            "テスト前提: 1e-20 * 1e-20 は非正規化数（アンダーフロー結果）であるはず"
        );

        let ap = vec![normal_small; MR * kc_len];
        let bp = vec![normal_small; kc_len * NR];
        let c_init = vec![0.0f32; MR * NR];

        let mut c_sme = c_init.clone();
        kernel.run(&ap, &bp, &mut c_sme, kc_len);

        let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
        // scalar_reference（`f32::mul_add`）自身が非正規化数の結果を
        // 生成することを前提の一部として確認する（テスト自体の健全性）。
        assert!(
            expected
                .iter()
                .all(|&v| v != 0.0 && v.abs() < f32::MIN_POSITIVE),
            "scalar 参照側も非正規化数を生成するはず（テスト前提）"
        );
        assert_eq!(
            c_sme, expected,
            "正規化数どうしの積がアンダーフローする非正規化数の結果を、\
             SME 側が FTZ で 0 へ潰さず scalar 参照と bit 完全一致するはず"
        );
    }

    /// NaN 混入入力で panic しないことのみを確認する（bit 一致は
    /// 主張しない。モジュール冒頭ドキュメント参照）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_nan_input_does_not_panic() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };
        let kc_len = 4;
        let mut ap = xorshift32_vec(0x1a1a_1a1a, MR * kc_len);
        ap[0] = f32::NAN;
        let bp = xorshift32_vec(0x2b2b_2b2b, kc_len * NR);
        let mut c = xorshift32_vec(0x3c3c_3c3c, MR * NR);
        kernel.run(&ap, &bp, &mut c, kc_len);
        // panic しないことのみを確認する。
    }

    /// AAPCS64／Arm ACLE が規定する TPIDR2 ブロックの最小レイアウト
    /// （<https://arm-software.github.io/acle/main/acle.html#tpidr2_el0>
    /// 「TPIDR2 block」節）を模した構造体。オフセット 0 に保存バッファ
    /// 先頭アドレス（`za_save_buffer`）・オフセット 8 に
    /// `num_za_save_slices`（16 bit）を置き、残りは予約領域とする。
    /// ACLE はブロック自体を 16 バイト境界へ整列することを要求するため
    /// `#[repr(C, align(16))]` を付ける。
    ///
    /// codex-review 追加指摘 1（本ファイル冒頭の呼び出し元コメント参照）
    /// への対応: 従来はダミー定数（有効なメモリを指さない値）を
    /// `TPIDR2_EL0` へ直接設定していたが、これは AAPCS64 の
    /// 「`TPIDR2_EL0 != 0` は有効な TPIDR2 ブロックを指す」という不変
    /// 条件に違反する。フォールバック方式（[`has_pending_lazy_za_save`]
    /// doc「採用方式」節）自体は `compute` を呼ばないため本実装の下で
    /// 実際に lazy save トラップが発生することはないが、`unsafe` の
    /// 不変条件はハードウェア側の契約として独立に満たす必要がある
    /// （検証対象コード自身が守ることだけを安全性の根拠にしない）。
    #[repr(C, align(16))]
    struct Tpidr2Block {
        za_save_buffer: u64,
        num_za_save_slices: u16,
        _reserved: [u8; 6],
    }

    /// [`Tpidr2Block::za_save_buffer`] が指す保存バッファ本体を 16 バイト
    /// 整列で確保する RAII ラッパー。`Vec<u8>` は align(1) しか保証しない
    /// ため、`std::alloc` を直接使い明示的な整列を要求する。
    struct AlignedSaveBuffer {
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
    }

    impl AlignedSaveBuffer {
        /// `len` バイトをゼロ初期化して 16 バイト整列で確保する。
        /// `len` は呼び出し元（[`enter_dormant_za_with_pending_lazy_save`]）
        /// が実行時 SVL から算出した正の値を渡す契約。
        fn new_zeroed(len: usize) -> Self {
            let layout = std::alloc::Layout::from_size_align(len, 16)
                .expect("len は MR × 実行時 SVL バイトで常に正・有効な整列のはず");
            // SAFETY: `layout` はサイズ 0 でない（呼び出し元契約）。
            // `alloc_zeroed` はゼロ初期化済みメモリを返すため、以降の
            // `msr`/`mrs` 経由でこのバッファが万一参照されても未初期化
            // 読み取りにはならない。
            let raw = unsafe { std::alloc::alloc_zeroed(layout) };
            let ptr = std::ptr::NonNull::new(raw)
                .unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
            Self { ptr, layout }
        }

        fn as_mut_ptr(&mut self) -> *mut u8 {
            self.ptr.as_ptr()
        }
    }

    impl Drop for AlignedSaveBuffer {
        fn drop(&mut self) {
            // SAFETY: `self.ptr`／`self.layout` は `new_zeroed` が
            // `std::alloc::alloc_zeroed` で確保したものと一致し、
            // `AlignedSaveBuffer` は `Clone`/`Copy` を実装しないため
            // 二重解放は起こらない。
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) };
        }
    }

    /// [`enter_dormant_za_with_pending_lazy_save`] が確保した TPIDR2
    /// ブロック・保存バッファの所有権を呼び出し元へ返す RAII ガード。
    /// `TPIDR2_EL0` に設定したアドレスが指すメモリを、
    /// [`read_za0_and_clear_pending_lazy_save`] が `TPIDR2_EL0` を
    /// クリアするまで生存させ続けるために必要（クリア前にこの構造体を
    /// drop するとダングリングポインタを指したままになる。追加指摘 1
    /// 対応）。フィールドは値を読み書きしないため `_` 接頭辞を付けるが、
    /// 生存期間の保持自体が本構造体の責務。
    ///
    /// codex-review 追加指摘（P0）対応: 正常経路（手順 4.
    /// [`read_za0_and_clear_pending_lazy_save`]）に到達する前に
    /// 呼び出し側の `assert_eq!`／`expect` が panic すると、以前の実装
    /// では `TPIDR2_EL0` が解放済みメモリ（`_block`／`_save_buffer`）を
    /// 指したまま unwind が続いた（`Drop` 未実装のため）。これは
    /// [`Tpidr2Block`] doc が要求する「クリアされるまでポインタが有効な
    /// メモリを指し続ける」不変条件、および `unsafe` の統制不変条件に
    /// 違反する。[`Drop`] を実装し、フィールド（`_block`／
    /// `_save_buffer`）の解放より **前**に必ず `TPIDR2_EL0` クリア＋
    /// `smstop` の後始末を完了させる（Rust の drop 順序はガード自身の
    /// `Drop::drop` 本体を実行してからフィールドを宣言順に drop するため、
    /// この順序は言語仕様として保証される）。
    ///
    /// **codex-review P0 再指摘対応（ストリーミングモードを同じ asm
    /// ブロック内で解除する。詳細は [`Self::read_za0_and_finish_streaming`]
    /// doc「背景」節）**: 正常経路（[`read_za0_and_clear_pending_lazy_save`]）
    /// は [`Self::read_za0_and_finish_streaming`] が ZA0 読み出し・
    /// `TPIDR2_EL0` クリア・`smstop` を**単一の asm ブロック内**で完結
    /// させる。panic 時の [`Drop::drop`] は [`Self::finish_pending_save`]
    /// （dormant のまま panic した場合専用。`smstart` を発行しないため
    /// PSTATE.SM の遷移自体を伴わない、別の独立した asm ブロック）を
    /// 呼ぶ。両者は asm を共有しないが、`cleaned_up` フラグにより
    /// 二重後始末（`msr`／`smstop` の重複発行）を防ぐ。
    struct PendingLazySaveGuard {
        _block: Box<Tpidr2Block>,
        _save_buffer: AlignedSaveBuffer,
        /// [`Self::read_za0_and_finish_streaming`]（正常経路）または
        /// [`Self::finish_pending_save`]（panic 時の `Drop` 経路）に
        /// よる後始末が完了済みかどうか。いずれか一方が既に完了して
        /// いれば、もう一方（実際には `Drop::drop` 経由の
        /// `finish_pending_save`）は何もしない（二重の `msr`／`smstop`
        /// 発行を避けるため）。
        cleaned_up: bool,
    }

    impl PendingLazySaveGuard {
        /// ZA0 の読み出し・保留中 save の解消（`TPIDR2_EL0` クリア）・
        /// ストリーミングモード終了（`smstop`）を**単一の asm! ブロック
        /// 内**で完了する（codex-review P0 指摘対応。`crates/backend-cpu/src/gemm_blis/microkernel/sme.rs:1091`
        /// 「ストリーミングモードを同じ asm ブロック内で解除する」）。
        ///
        /// ## 背景（以前の実装の問題点）
        ///
        /// 以前は「`smstart sm` → ZA0 読み出し」（asm ブロック A）の
        /// 直後に、通常の Rust メソッド呼び出し [`Self::finish_pending_save`]
        /// （asm ブロック B: `msr TPIDR2_EL0, xzr` → `smstop`）を呼ぶ
        /// 2 ブロック構成だった。ブロック A を抜けた時点で
        /// PSTATE.SM=1 のまま Rust へ戻り、ブロック A・B の間に挟まる
        /// 関数呼び出し・`cleaned_up` の条件分岐はコンパイラ生成コード
        /// であるにもかかわらず、ストリーミングモード中の実行を前提と
        /// しない一般の命令列でありうる。Arm ACLE の asm 制約
        /// (<https://arm-software.github.io/acle/main/acle.html#asm-restrictions-related-to-streaming-mode>)
        /// は「各 asm が呼び出し時点の PSTATE.SM を保存し、モード切替で
        /// 変更されるレジスタを宣言する」ことを private-ZA 関数の
        /// インライン asm へ要求しており、2 ブロック構成はこの制約に
        /// 違反していた。
        ///
        /// ## 採用方式
        ///
        /// 本メソッドは `smstart sm`・ZA0 読み出し・
        /// `msr TPIDR2_EL0, xzr`・`smstop` を単一の asm! ブロックに
        /// まとめ、**asm を抜けた時点で必ず PSTATE.SM=0（かつ
        /// PSTATE.ZA=0）に戻す**ことでこの制約を満たす。AAPCS64 が
        /// 要求する順序（ZA 無効化より前に保留中 save を解消する）も、
        /// 同一ブロック内で `msr` を `smstop` より先に発行することで
        /// 満たす。asm を抜けた**後**（もはや SME 状態遷移を伴わない
        /// 通常の Rust コードとして安全）に `self.cleaned_up = true` を
        /// 更新し、以降 [`Drop::drop`] 経由で呼ばれる
        /// [`Self::finish_pending_save`]（dormant 専用の後始末。本
        /// メソッドとは asm を共有しない）を no-op にする。
        ///
        /// # Safety
        ///
        /// 呼び出し元は実行 CPU が SME 対応であり、本スレッドが
        /// dormant（PSTATE.SM=0・PSTATE.ZA=1。
        /// [`enter_dormant_za_with_pending_lazy_save`] が返した直後の
        /// 状態）にあることを保証しなければならない。
        /// `out.len() == MR * NR` を満たさない場合の挙動は未定義
        /// （`st1w` が範囲外を書く）。
        unsafe fn read_za0_and_finish_streaming(&mut self, out: &mut [f32]) {
            // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により本
            // スレッドは dormant（PSTATE.SM=0・PSTATE.ZA=1）。
            // `smstart sm` で SM のみ再度有効化するため、既に有効な
            // PSTATE.ZA=1 の内容は変化しない（PSTATE.ZA が 0→1 へ遷移
            // するときにのみゼロ初期化されるという Arm SME の仕様を
            // 本実装セッションで実機事前確認済み）。
            //
            // レジスタ・状態契約（`compute`〈本モジュール上部〉と同型）:
            // `smstart`/`smstop` を本ブロック内で対にして閉じ、モード
            // 切替で不定化される Z0-Z31/P0-P15（および FFR）を
            // `out("v0") _ ... out("ffr") _` で全列挙する。`mova` の
            // インデックスレジスタは w12 限定のため `out("w12") _` を
            // 明示する。`cmp`/`b.lt` でフラグを書き換えるため
            // `preserves_flags` は付けない。`st1w`（メモリアクセス）を
            // 使うため `nomem`/`pure` は付けない。スタック未使用のため
            // `options(nostack)`。
            unsafe {
                let out_p = out.as_mut_ptr();
                asm!(
                    ".arch_extension sme",
                    "smstart sm",
                    "ptrue p0.s",
                    "mov w12, #0",
                    "21:",
                    "mova z1.s, p0/m, za0h.s[w12, #0]",
                    "st1w {{z1.s}}, p0, [{out_p}]",
                    "add {out_p}, {out_p}, #64",
                    "add w12, w12, #1",
                    "cmp w12, #16",
                    "b.lt 21b",
                    // AAPCS64 の要求順序: ZA 無効化（smstop）より前に
                    // 保留中 save の解消（TPIDR2_EL0 クリア）を完了する。
                    // この順序を破らないよう、同一ブロック内で `msr` を
                    // `smstop` より必ず先に発行する。
                    "msr S3_3_C13_C0_5, xzr",
                    "smstop",
                    out_p = inout(reg) out_p => _,
                    out("w12") _,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                    out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                    out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                    out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
                    out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                    out("v30") _, out("v31") _,
                    out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
                    out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _,
                    out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _,
                    out("p15") _,
                    out("ffr") _,
                    options(nostack),
                );
            }
            // asm を抜けた時点で PSTATE.SM=0・PSTATE.ZA=0（`smstop` 完了
            // 済み）であり、以降は通常の Rust コードとして安全に実行
            // できる（本フィールド書き込みに SME 状態遷移は関与しない）。
            self.cleaned_up = true;
        }

        /// `TPIDR2_EL0` のクリア（保留中 save の解消）と `smstop`（ZA を
        /// 含む SME 状態の無効化）を行う後始末。AAPCS64 が要求する順序
        /// （ZA 無効化より前に保留中 save を解消する）を満たすため
        /// `msr` を `smstop` より必ず先に発行する。
        ///
        /// [`Drop::drop`]（panic 等により正常経路
        /// [`Self::read_za0_and_finish_streaming`] に到達しなかった
        /// 場合の異常経路）**専用**の後始末（codex-review P0 指摘対応で
        /// 正常経路とは asm を共有しない構成へ変更した。以前は
        /// [`read_za0_and_clear_pending_lazy_save`] の正常経路も本
        /// メソッドを呼んでいたが、`smstart sm` 直後の asm ブロックを
        /// 抜けてから本メソッドの asm へ入るまでの間に、関数呼び出し・
        /// `cleaned_up` の条件分岐というコンパイラ生成コードが
        /// PSTATE.SM=1 のまま実行される問題があった〈上記
        /// [`Self::read_za0_and_finish_streaming`] doc「背景」節〉。
        /// 正常経路は同メソッドが単一 asm ブロックで完結させるため、
        /// 本メソッドは「dormant のまま panic した場合」にのみ呼ばれる
        /// ── その場合 `smstart` を一切発行しないため PSTATE.SM の
        /// モード切替自体を伴わない）。`cleaned_up` により冪等（正常
        /// 経路が既に完了させていれば no-op）。
        ///
        /// # Safety
        ///
        /// 呼び出し元は実行 CPU が SME 対応であり、本スレッドが
        /// dormant（PSTATE.SM=0・PSTATE.ZA=1。
        /// [`enter_dormant_za_with_pending_lazy_save`] が返した直後の
        /// 状態）にあることを保証しなければならない
        /// （[`Self::read_za0_and_finish_streaming`] が正常終了して
        /// いれば `cleaned_up=true` により本メソッドの asm は実行され
        /// ないため、この契約は「正常経路に到達しなかった panic 時」に
        /// のみ実際に要求される）。
        unsafe fn finish_pending_save(&mut self) {
            if self.cleaned_up {
                // 正常経路（`read_za0_and_finish_streaming`）が既に
                // 後始末を終えている場合、後続の `Drop::drop` はここで
                // no-op になる（`msr`／`smstop` の二重発行を防ぐ）。
                return;
            }
            // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により本
            // スレッドは dormant（PSTATE.SM=0）で `smstart` を発行
            // しないため PSTATE.SM のモード切替自体は発生しない。
            // それでも `smstop`（PSTATE.ZA を 1→0 へ遷移）がハード
            // ウェア契約上 Z/P/FFR の内容を不定化しうる可能性を排除
            // しきれないため、防御的に `compute`（本モジュール上部）と
            // 同型のレジスタクロバー宣言
            // （`out("v0") _ ... out("ffr") _`）を付ける（codex-review
            // 指摘: asm ブロックのクロバー宣言の網羅性確認）。`msr`
            // （システムレジスタ書き込み）・`smstop`（PSTATE 遷移）とも
            // メモリへアクセスしないため `nomem` を付ける。Arm ACLE は
            // SMSTART／SMSTOP が NZCV フラグを変更しないと規定するため
            // `preserves_flags` を付ける。スタック未使用のため
            // `nostack`。
            unsafe {
                asm!(
                    ".arch_extension sme",
                    // AAPCS64 の要求順序: ZA 無効化（smstop）より前に
                    // 保留中 save の解消（TPIDR2_EL0 クリア）を完了する。
                    "msr S3_3_C13_C0_5, xzr",
                    "smstop",
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                    out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                    out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                    out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                    out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
                    out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                    out("v30") _, out("v31") _,
                    out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
                    out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _,
                    out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _,
                    out("p15") _,
                    out("ffr") _,
                    options(nomem, nostack, preserves_flags),
                );
            }
            self.cleaned_up = true;
        }
    }

    impl Drop for PendingLazySaveGuard {
        fn drop(&mut self) {
            // SAFETY: `PendingLazySaveGuard` は
            // `enter_dormant_za_with_pending_lazy_save` の戻り値としてのみ
            // 構築され、同関数の doc が要求するとおり dormant
            // （PSTATE.SM=0・PSTATE.ZA=1）のまま生存する（正常経路
            // [`Self::read_za0_and_finish_streaming`] は単一 asm 内で
            // PSTATE.SM=0 まで完全に戻してから `cleaned_up=true` を
            // 立てるため、そちらを経由済みなら本関数は
            // `finish_pending_save` の `cleaned_up` チェックで no-op に
            // なる）。呼び出し元は本ガードを `SmeKernel::try_new()` を
            // 確認済みのスレッドから spawn したクロージャ内でのみ生成
            // する契約（`enter_dormant_za_with_pending_lazy_save`
            // `# Safety` 節）。
            unsafe { self.finish_pending_save() };
        }
    }

    /// ZA0 へ既知パターン（`za_pattern.len() == MR * NR`）を書き込み、
    /// `smstop sm`（PSTATE.SM のみ解除・PSTATE.ZA は 1 のまま =
    /// **dormant**）してから、実在する TPIDR2 ブロック（[`Tpidr2Block`]）
    /// を指す非ゼロ値を `TPIDR2_EL0`（符号化名 `S3_3_C13_C0_5`）へ設定
    /// する（AAPCS64 の「保留中 lazy ZA save」を模す）。ブロックの
    /// `num_za_save_slices` は本関数が ZA0 へ書き込んだ行数（`MR`）に
    /// 合わせ、保存バッファは `MR * svl_bytes`（実行時 `rdsvl` 相当。
    /// `crate::sme_detect::sme_report` 経由で取得）バイトをゼロ初期化
    /// して確保する。返す [`PendingLazySaveGuard`] は
    /// [`read_za0_and_clear_pending_lazy_save`] へ渡すまで呼び出し元が
    /// 保持し、`TPIDR2_EL0` クリアより前に drop してはならない。
    ///
    /// [`sme_kernel_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`]・
    /// [`sme_kernel_unchecked_with_ldc_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`]・
    /// [`sme_kernel_unchecked_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`]
    /// で共有する検証手順の手順 1 を切り出した共通ヘルパ（codex-review
    /// P0 指摘 `PRRT_kwDOTuUCJc6h0ZMD` およびその追加指摘への対応）。
    ///
    /// # Safety
    ///
    /// 呼び出し元は実行 CPU が SME 対応であることを保証しなければ
    /// ならない（`SmeKernel::try_new()` を確認済みのスレッドから spawn
    /// したクロージャ内で呼ぶ想定）。`za_pattern.len() == MR * NR` を
    /// 満たさない場合の挙動は未定義（`ld1w` が範囲外を読む）。
    ///
    /// `.arch_extension sme` はコンパイル時のみに影響。`smstart`
    /// （Z/P レジスタ不定化）に対応する `out("v0") _ ... out("p15") _`
    /// 全列挙・`mova` の index レジスタ `w12` の宣言は `compute`
    /// （本モジュール上部）と同型の契約。フラグ変更命令（`cmp`/`b.lt`）を
    /// 使うため `preserves_flags` は付けない。`ld1w`（メモリアクセス）を
    /// 使うため `nomem`/`pure` は付けない。スタック未使用のため
    /// `options(nostack)`。本関数はテスト専用の ZA 状態構築であり
    /// `compute` の呼び出し規約とは独立。
    unsafe fn enter_dormant_za_with_pending_lazy_save(za_pattern: &[f32]) -> PendingLazySaveGuard {
        // 実行時 SVL。`SmeKernel::try_new()` を確認済みのスレッド前提
        // （呼び出し元契約）のため、`crate::sme_detect::sme_report()` は
        // 常に `kernel_enabled == true` かつ `svl_bytes == Some(_)` を
        // 返すはず（本番マイクロカーネルと同じ SVL を都度取得する方針。
        // モジュール冒頭「検出との関係」節）。
        let svl_bytes = crate::sme_detect::sme_report()
            .svl_bytes
            .expect("呼び出し元契約により SME 対応・SVL 確認済みのはず");

        // 保存バッファは ZA0 へ実際に書き込んだ行数（MR）× SVL バイト
        // 分だけ確保する（本関数のフォールバック検証では実トラップは
        // 発生しないため full ZA 分は不要。呼び出し元 doc 参照）。
        let mut save_buffer = AlignedSaveBuffer::new_zeroed(MR * svl_bytes);
        let save_buffer_addr = save_buffer.as_mut_ptr() as u64;

        let block = Box::new(Tpidr2Block {
            za_save_buffer: save_buffer_addr,
            num_za_save_slices: MR as u16,
            _reserved: [0; 6],
        });
        let block_addr = &*block as *const Tpidr2Block as u64;

        // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）を引き継ぐ。
        unsafe {
            let p = za_pattern.as_ptr();
            asm!(
                ".arch_extension sme",
                "smstart",
                "ptrue p0.s",
                "mov w12, #0",
                "20:",
                "ld1w {{z0.s}}, p0/z, [{p}]",
                "mova za0h.s[w12, #0], p0/m, z0.s",
                "add {p}, {p}, #64",
                "add w12, w12, #1",
                "cmp w12, #16",
                "b.lt 20b",
                // SM のみ解除（ZA は有効のまま）: dormant 状態。
                "smstop sm",
                // AAPCS64 の「保留中 lazy ZA save」を模す。`{block}` は
                // 上で確保した実在の TPIDR2 ブロックのアドレス
                // （ダミー値ではない。追加指摘 1 対応）。
                "msr S3_3_C13_C0_5, {block}",
                p = inout(reg) p => _,
                block = in(reg) block_addr,
                out("w12") _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
                out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
                out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
                out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
                out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
                out("v30") _, out("v31") _,
                out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
                out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _,
                out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _,
                out("p15") _,
                out("ffr") _,
                options(nostack),
            );
        }

        PendingLazySaveGuard {
            _block: block,
            _save_buffer: save_buffer,
            cleaned_up: false,
        }
    }

    /// dormant のあいだ（PSTATE.SM=0, PSTATE.ZA=1）に検証対象の呼び出し
    /// （`kernel.run`／`kernel_unchecked_with_ldc`／`kernel_unchecked`）を
    /// 済ませたあと、ZA0 を `out.len() == MR * NR` へ読み出し・
    /// `TPIDR2_EL0` クリア・`smstop` までを
    /// [`PendingLazySaveGuard::read_za0_and_finish_streaming`]（単一の
    /// asm! ブロック内で完結する。同メソッド doc「背景」節参照）へ
    /// 委譲する（[`enter_dormant_za_with_pending_lazy_save`] と対になる
    /// 検証手順の手順 4〜5 を切り出した共通ヘルパ）。
    ///
    /// codex-review 追加指摘 2 対応: AAPCS64 は「ZA を無効化する前に
    /// 保留中の save 状態を解消する（`TPIDR2_EL0` をゼロクリアする）」
    /// 順序を要求する。旧実装は `smstop`（PSTATE.ZA を 0 へ）した後に
    /// 別の asm ブロックで `TPIDR2_EL0` をクリアしており、その間
    /// 「ZA は無効なのに保留中 save を示す TPIDR2_EL0 が非ゼロのまま」
    /// という AAPCS64 上不正な中間状態を作っていた。
    /// [`PendingLazySaveGuard::read_za0_and_finish_streaming`] は
    /// 単一 asm ブロック内で `msr` を `smstop` より先に発行することで
    /// この順序を満たす。
    ///
    /// codex-review 追加指摘（P0）対応: 以前は「ZA0 読み出し→
    /// `TPIDR2_EL0` クリア→`smstop`」を単一の asm ブロックで行っていた
    /// ため、呼び出し元の `assert_eq!` 等が読み出し結果を検査する前に
    /// panic しても後始末自体は既に完了していたが、逆に
    /// [`enter_dormant_za_with_pending_lazy_save`] 呼び出し後・本関数
    /// 呼び出し前に呼び出し元が panic するケース（3 実機テストとも
    /// `kernel.run` 直後の `assert_eq!` が本関数呼び出しより前にある）
    /// では後始末が一切走らなかった。本関数は
    /// `guard.read_za0_and_finish_streaming(out)` へ ZA0 読み出し・
    /// 後始末の両方を委譲し（[`Drop`] とは別経路だが `cleaned_up` で
    /// 冪等）、panic 経路は [`Drop::drop`]（
    /// [`PendingLazySaveGuard::finish_pending_save`]。本関数の呼び出しに
    /// 到達しなかった場合の後始末）が担うことで、正常経路・panic 経路
    /// のどちらでも後始末が保証される。
    ///
    /// **codex-review P0 再指摘対応（ストリーミングモードを同じ asm
    /// ブロック内で解除する）**: 以前の本関数は「ZA0 読み出し（asm
    /// ブロック）→ Rust の関数呼び出し・`cleaned_up` の条件分岐
    /// （コンパイラ生成コード）→ 後始末（別の asm ブロック）」という
    /// 構成で、ZA0 読み出しの asm ブロックを抜けた時点で
    /// PSTATE.SM=1 のまま Rust コードへ戻っていた。これは Arm ACLE の
    /// asm 制約（各 asm が呼び出し時点の PSTATE.SM を保存することを
    /// 要求する）に違反しうる。現在は ZA0 読み出し・`TPIDR2_EL0`
    /// クリア・`smstop` を単一の asm ブロックにまとめた
    /// [`PendingLazySaveGuard::read_za0_and_finish_streaming`] へ委譲
    /// することで、asm を抜けた時点で必ず PSTATE.SM=0 に戻す（詳細は
    /// 同メソッド doc「背景」「採用方式」節）。
    ///
    /// `guard`（[`enter_dormant_za_with_pending_lazy_save`] が返した
    /// TPIDR2 ブロック・保存バッファの所有権）は `TPIDR2_EL0` を
    /// クリアした後に drop する（クリア前に参照先メモリを解放しない
    /// ため。追加指摘 1 対応。`read_za0_and_finish_streaming` 完了後に
    /// `guard` を関数末尾で自然に drop させることでこれを満たす）。
    ///
    /// # Safety
    ///
    /// [`enter_dormant_za_with_pending_lazy_save`] と同一（実行 CPU が
    /// SME 対応であること。`out.len() == MR * NR` を満たさない場合の
    /// 挙動は未定義）。
    unsafe fn read_za0_and_clear_pending_lazy_save(
        out: &mut [f32],
        mut guard: PendingLazySaveGuard,
    ) {
        // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）を
        // `PendingLazySaveGuard::read_za0_and_finish_streaming` へ
        // 引き継ぐ（dormant〈PSTATE.SM=0・PSTATE.ZA=1〉直後の状態で
        // あること）。ZA0 読み出し・`TPIDR2_EL0` クリア・`smstop` は
        // 同メソッド内の単一 asm ブロックで完結する（codex-review P0
        // 指摘対応。同メソッド doc「背景」節参照）。
        unsafe { guard.read_za0_and_finish_streaming(out) };

        // `TPIDR2_EL0` は上記で既にクリア済み（`smstop` も完了済みで
        // PSTATE.SM=0 に戻っている）。ここで `guard` を明示的に drop
        // し、保存バッファ・ブロックの解放が「クリア後」であることを
        // コード上明確にする（追加指摘 1 対応。実際には関数末尾で
        // 自然に drop されるが、契約の可視化のため明示する）。
        drop(guard);
    }

    /// [`PendingLazySaveGuard`] を [`read_za0_and_clear_pending_lazy_save`]
    /// へ渡さずに（＝panic 時と同じく手順 4 に到達しない経路で）drop
    /// した場合でも、`Drop::drop` が `TPIDR2_EL0` クリア・`smstop` の
    /// 後始末を行うことを検証する（codex-review 追加指摘（P0）対応。
    /// [`PendingLazySaveGuard`] doc 参照）。
    ///
    /// 実際の panic を発生させる代わりに、意図的に本関数呼び出しを
    /// 省略して即座に `drop(guard)` する（panic-unwind 経路と `drop`
    /// が呼ばれる点は同一であり、`Drop::drop` の実装自体は panic
    /// 発生有無を区別しないため、本テストは非破壊なまま
    /// `Drop::drop` が単独で正しく後始末することを確認できる）。
    ///
    /// 確認する 2 点:
    /// 1. `TPIDR2_EL0 == 0`（[`has_pending_lazy_za_save`] が偽を返す。
    ///    ダングリングポインタを指したまま残らないことの直接証拠）。
    /// 2. `smstart`（フル: SM・ZA 両方）で PSTATE.ZA=0 から 1 へ正常に
    ///    遷移できる（`PendingLazySaveGuard::Drop` が `smstop` で
    ///    PSTATE.SM／PSTATE.ZA を無効化済みであることの間接証拠。
    ///    無効化されていなければ `SmeKernel::try_new()` 経由の以降の
    ///    `kernel.run` 呼び出しが同スレッド上で不整合な状態から
    ///    始まってしまう）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_pending_lazy_save_guard_drop_without_read_clears_tpidr2_and_za() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };

        let handle = std::thread::spawn(move || {
            let za_pattern = xorshift32_vec(0xeeee_5555, MR * NR);

            // SAFETY: 手順は
            // `sme_kernel_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`
            // と同型（`SmeKernel::try_new()` を確認済みのスレッドから
            // spawn したクロージャ内で直接実行）。
            let guard = unsafe { enter_dormant_za_with_pending_lazy_save(&za_pattern) };

            // `read_za0_and_clear_pending_lazy_save` を呼ばず、
            // panic 時と同じ「後始末未完了のまま drop される」経路を
            // 意図的に再現する。
            drop(guard);

            // 確認 1: `TPIDR2_EL0` がクリアされている
            // （ダングリングポインタを指したまま残っていない）。
            //
            // SAFETY: 本スレッドは `SmeKernel::try_new()` を確認済み
            // （`has_pending_lazy_za_save` `# Safety` 節と同一契約）。
            let still_pending = unsafe { has_pending_lazy_za_save() };
            assert!(
                !still_pending,
                "PendingLazySaveGuard を Drop しただけで TPIDR2_EL0 が \
                 クリアされているはず（panic 時の後始末保証。\
                 codex-review 追加指摘対応）"
            );

            // 確認 2: PSTATE.SM／PSTATE.ZA が無効化済みで、以降
            // 通常どおり `kernel.run` を実行できる（`smstart` 由来の
            // ZA=0→1 ゼロ初期化を経由する新規呼び出しが問題なく通る）。
            let kc_len = 2usize;
            let ap = xorshift32_vec(0x1234_5678, MR * kc_len);
            let bp = xorshift32_vec(0x8765_4321, kc_len * NR);
            let c_init = xorshift32_vec(0x0f0f_0f0f, MR * NR);
            let mut c = c_init.clone();
            kernel.run(&ap, &bp, &mut c, kc_len);
            let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
            assert_eq!(
                c, expected,
                "Drop 後の後始末が完了していれば、以降の kernel.run は \
                 通常経路（fmopa）で scalar 参照と bit 完全一致するはず"
            );
        });
        handle
            .join()
            .expect("PendingLazySaveGuard Drop 後始末検証用スレッドが panic した");
    }

    /// 保留中 lazy ZA save（`TPIDR2_EL0 != 0`）がある状態で `kernel.run`
    /// を呼んだとき、`compute`（`fmopa` 経路。ZA0 を上書きする）を一切
    /// 実行せず、呼び出し元が dormant のまま残した ZA0 の内容を破壊
    /// しないことを検証する（codex-review P0 指摘
    /// `PRRT_kwDOTuUCJc6h0ZMD` 対応。[`has_pending_lazy_za_save`] doc
    /// 「採用方式」節参照）。
    ///
    /// ## 検証手順
    ///
    /// 1. 専用スレッド上で [`enter_dormant_za_with_pending_lazy_save`]
    ///    により ZA0 に既知パターンを書き込み dormant 状態を作る（実機で
    ///    「dormant のあいだ〈PSTATE.SM=0〉は通常の Rust コードを安全に
    ///    実行できる」ことを事前確認済み — PSTATE.ZA=1 は ZA タイル用
    ///    命令以外へ影響しない）。
    /// 2. dormant のあいだに `kernel.run(...)` を呼ぶ。
    ///    `current_thread_capable()` が `TPIDR2_EL0 != 0` を検出し
    ///    `compute` を呼ばずスカラーフォールバックへ切り替わるはず
    ///    （本テストの主張）。
    /// 3. 結果が scalar 参照と bit 完全一致すること（フォールバック
    ///    自体の正しさ）を確認する。
    /// 4. [`read_za0_and_clear_pending_lazy_save`] で ZA0 を読み出し、
    ///    手順 1 で書き込んだパターンと bit 完全一致する（＝`compute`
    ///    が一度も ZA0 を上書きしていない）ことを確認する。
    ///
    /// `TPIDR2_EL0`／ストリーミングモード／ZA はスレッドごとの状態
    /// （モジュール冒頭「SVL はスレッドごとに異なりうる」節と同じ理由）
    /// のため、本テストは専用スレッドで実行し他のテストへ影響させない。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_falls_back_without_corrupting_dormant_za_when_lazy_save_pending() {
        let Some(kernel) = SmeKernel::try_new() else {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        };

        let handle = std::thread::spawn(move || {
            let za_pattern = xorshift32_vec(0xdddd_4444, MR * NR);
            let mut za_readback = vec![0.0f32; MR * NR];

            // 手順 1。
            //
            // SAFETY: `SmeKernel::try_new()` が実行 CPU の SME 対応を
            // 確認済み（本スレッドはこれを確認したスレッドから spawn
            // されたクロージャ内で直接実行するため、`SmeKernel` 構造体
            // doc の「構築したスレッド」に関する注意は本 unsafe 呼び出し
            // 自体〈`try_new` を呼んだのと同じスレッド上でのテスト専用
            // ZA 操作〉には影響しない）。
            let guard = unsafe { enter_dormant_za_with_pending_lazy_save(&za_pattern) };

            // 手順 2〜3: dormant のあいだ（PSTATE.SM=0）に kernel.run を
            // 呼ぶ。`current_thread_capable()` が TPIDR2_EL0 != 0 を検出し
            // フォールバックへ切り替わるはず。
            let kc_len = 3usize;
            let ap = xorshift32_vec(0xaaaa_1111, MR * kc_len);
            let bp = xorshift32_vec(0xbbbb_2222, kc_len * NR);
            let c_init = xorshift32_vec(0xcccc_3333, MR * NR);
            let mut c = c_init.clone();
            kernel.run(&ap, &bp, &mut c, kc_len);

            let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
            assert_eq!(
                c, expected,
                "TPIDR2_EL0 非ゼロ時はフォールバックの結果が scalar 参照と \
                 bit 完全一致するはず"
            );

            // 手順 4。
            //
            // SAFETY: 手順 1 と同型の契約。
            unsafe { read_za0_and_clear_pending_lazy_save(&mut za_readback, guard) };

            assert_eq!(
                za_readback, za_pattern,
                "TPIDR2_EL0 非ゼロ時は compute（fmopa 経路）が一度も \
                 ZA0 を上書きしないはず（呼び出し元の dormant ZA を \
                 破壊しない）"
            );
        });
        handle
            .join()
            .expect("SME フォールバック検証用スレッドが panic した");
    }

    /// [`sme_kernel_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`]
    /// の [`kernel_unchecked_with_ldc`] 直接呼び出し版。
    ///
    /// codex-review P0 追加指摘（`kernel_unchecked_with_ldc`／
    /// `kernel_unchecked` は公開 `unsafe fn` のため `SmeKernel::run*`
    /// を経由せず外部から直接呼びうるが、以前はこれらの入口自身が
    /// [`has_pending_lazy_za_save`] を検査していなかった）への対応。
    /// 本テストは同じ dormant ZA + `TPIDR2_EL0` 設定の構成で
    /// `kernel_unchecked_with_ldc` を直接呼び、結果の bit 一致・ZA0 の
    /// 非破壊・`TPIDR2_EL0` が非ゼロのまま維持されることを検証する
    /// （[`kernel_unchecked_with_ldc`] の doc「保留中 lazy ZA save の
    /// 自己防御」節が主張する契約そのもの）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_unchecked_with_ldc_falls_back_without_corrupting_dormant_za_when_lazy_save_pending()
     {
        // `pub unsafe fn kernel_unchecked_with_ldc` を直接呼ぶには実行
        // CPU の SME 対応確認が必要（`SmeKernel::try_new()` を代わりに
        // 使い、`Some` の場合のみ実行することでこれを満たす）。
        if SmeKernel::try_new().is_none() {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        }

        let handle = std::thread::spawn(move || {
            let za_pattern = xorshift32_vec(0xeeee_5555, MR * NR);
            let mut za_readback = vec![0.0f32; MR * NR];

            // SAFETY: 呼び出し元スレッド自身の SME 対応は本スレッドを
            // spawn した外側で `SmeKernel::try_new()` により確認済み
            // （`enter_dormant_za_with_pending_lazy_save` の `# Safety`
            // 契約を満たす）。
            let guard = unsafe { enter_dormant_za_with_pending_lazy_save(&za_pattern) };

            // 端タイル相当（`ldc > NR`）で `kernel_unchecked_with_ldc` を
            // 直接呼ぶ。`TPIDR2_EL0` が非ゼロのため、本関数自身が
            // `has_pending_lazy_za_save` を検査し `compute` を呼ばず
            // `scalar_fallback` へ切り替わるはず。
            let kc_len = 5usize;
            let ldc = NR + 3;
            let ap = xorshift32_vec(0x1357_9bdf, MR * kc_len);
            let bp = xorshift32_vec(0x2468_ace0, kc_len * NR);
            let mut c = xorshift32_vec(0x0f0f_1e1e, (MR - 1) * ldc + ldc);
            let c_init = c.clone();

            // SAFETY: 実行 CPU は SME 対応（呼び出し元契約）。
            // `TPIDR2_EL0` が非ゼロの場合は本関数自身がフォールバックへ
            // 切り替えるため（`kernel_unchecked_with_ldc` doc「保留中
            // lazy ZA save の自己防御」節）、事前の
            // `has_pending_lazy_za_save` 確認は不要（本テストの主張）。
            unsafe { kernel_unchecked_with_ldc(&ap, &bp, &mut c, ldc, kc_len) }
                .expect("長さ・境界は事前に検査済みのため成功するはず");

            let expected = scalar_reference(&ap, &bp, &c_init, ldc, kc_len);
            assert_eq!(
                c, expected,
                "TPIDR2_EL0 非ゼロ時は kernel_unchecked_with_ldc 自身が \
                 フォールバックへ切り替わり scalar 参照と bit 完全一致 \
                 するはず"
            );

            // SAFETY: 手順 1 と同型の契約。
            unsafe { read_za0_and_clear_pending_lazy_save(&mut za_readback, guard) };

            assert_eq!(
                za_readback, za_pattern,
                "kernel_unchecked_with_ldc を TPIDR2_EL0 非ゼロで直接 \
                 呼んでも compute（fmopa 経路）が一度も ZA0 を上書き \
                 しないはず（呼び出し元の dormant ZA を破壊しない）"
            );
        });
        handle
            .join()
            .expect("kernel_unchecked_with_ldc フォールバック検証用スレッドが panic した");
    }

    /// [`sme_kernel_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`]
    /// の [`kernel_unchecked`] 直接呼び出し版。
    ///
    /// [`sme_kernel_unchecked_with_ldc_falls_back_without_corrupting_dormant_za_when_lazy_save_pending`]
    /// と同じ codex-review P0 追加指摘への対応（従来シグネチャ後方互換
    /// ラッパー側の入口も同様に自己防御することを検証する）。
    #[test]
    #[ignore = "実機（SME 対応 aarch64。例: Apple M4）限定の検証専用（イシュー #1587。\
                cargo test -p fandhe-ai-backend-cpu --lib -- --ignored sme_kernel --nocapture）"]
    fn sme_kernel_unchecked_falls_back_without_corrupting_dormant_za_when_lazy_save_pending() {
        if SmeKernel::try_new().is_none() {
            eprintln!("SME 非対応環境のためスキップ");
            return;
        }

        let handle = std::thread::spawn(move || {
            let za_pattern = xorshift32_vec(0xffff_6666, MR * NR);
            let mut za_readback = vec![0.0f32; MR * NR];

            // SAFETY: `kernel_unchecked_with_ldc_falls_back_...` と同型
            // の契約。
            let guard = unsafe { enter_dormant_za_with_pending_lazy_save(&za_pattern) };

            let kc_len = 7usize;
            let ap = xorshift32_vec(0x1122_3344, MR * kc_len);
            let bp = xorshift32_vec(0x5566_7788, kc_len * NR);
            let c_init = xorshift32_vec(0x99aa_bbcc, MR * NR);
            let mut c = c_init.clone();

            // SAFETY: 実行 CPU は SME 対応（呼び出し元契約）。
            // `TPIDR2_EL0` が非ゼロの場合は `kernel_unchecked` 自身が
            // フォールバックへ切り替える（`kernel_unchecked` doc「保留中
            // lazy ZA save の自己防御」節）。
            unsafe { kernel_unchecked(&ap, &bp, &mut c, kc_len) }
                .expect("長さは事前に用意済みのため成功するはず");

            let expected = scalar_reference(&ap, &bp, &c_init, NR, kc_len);
            assert_eq!(
                c, expected,
                "TPIDR2_EL0 非ゼロ時は kernel_unchecked 自身がフォール \
                 バックへ切り替わり scalar 参照と bit 完全一致するはず"
            );

            // SAFETY: 手順 1 と同型の契約。
            unsafe { read_za0_and_clear_pending_lazy_save(&mut za_readback, guard) };

            assert_eq!(
                za_readback, za_pattern,
                "kernel_unchecked を TPIDR2_EL0 非ゼロで直接呼んでも \
                 compute（fmopa 経路）が一度も ZA0 を上書きしないはず \
                 （呼び出し元の dormant ZA を破壊しない）"
            );
        });
        handle
            .join()
            .expect("kernel_unchecked フォールバック検証用スレッドが panic した");
    }
}
