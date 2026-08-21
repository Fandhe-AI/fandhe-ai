//! MC/KC/NC の実行時キャッシュ検出（イシュー #753・§3.1）。
//!
//! [`super::default_blocks`] は Apple M4 Max 実測（#749・#481 §3）に基づく
//! コンパイル時定数だが、実行環境が変われば L1D／L2 実容量も変わるため、
//! 固定値は実機ごとの最適点から外れうる。本モジュールは `sysctl`
//! （macOS 限定）で実測した L1D／L2 サイズから BLIS 解析モデル系の技法
//! （L1 に A/B マイクロパネルが追い出し合わずに共存する条件で KC を、
//! L2 に A パネルが収まる条件で MC を、というキャッシュ階層ブロッキングの
//! 一般的な導出方針。gemm crate `cache.rs` の技法を**参照**するのみで
//! コードは転記しない）で MC/KC/NC を算出する。
//!
//! ## デッドコード化を避ける設計（PR #766 の教訓）
//!
//! PR #766 では「常に不活性な sysctl FFI」（機種識別子ベースの判定分岐が
//! 実測未確定のため常に旧経路へ落ちる構成）が codex-review の P0/P1 指摘で
//! 撤去された。本モジュールは同じ轍を踏まないよう、以下の 3 点で構成する:
//!
//! 1. **純関数化**（[`compute_blocks`]）: 機種判定ではなくキャッシュ
//!    サイズからの算出式のため、macOS 以外を含む全プラットフォームで
//!    単体テスト可能（`mod tests` 参照。CI（GitHub ホステッド
//!    `ubuntu-latest`）上でも実行され続ける）。
//! 2. **`cargo test` 経由でのコンパイル検証**: 本モジュール自体
//!    （[`super`] の `mod cache_params;` 宣言）は本番未結線の間
//!    `#[cfg(test)]` のため、CI の通常ビルドジョブ
//!    （`cargo build (linux / aarch64-apple-darwin)`。`.claude/rules/ci.md`）
//!    には**含まれない**（`cfg(test)` 無効時は構造的にコンパイル対象外）。
//!    純関数（[`compute_blocks`] 等・`cfg(target_os = "macos")` 非依存）の
//!    型・借用検査は `rust-ci` の test ジョブ（`cargo test`。`cfg(test)` 有効）
//!    が Linux ホスト上で担うが、同ジョブは `target_os != "macos"` のため
//!    sysctl FFI 部（[`sysctl_ffi`]）自体はコンパイル対象に含まれない
//!    （**レビュー指摘・#753**: 通常の macOS クロスビルドジョブも
//!    `cfg(test)` 無効のため到達せず、結果として `sysctl_ffi` を継続的に
//!    型・借用検査するジョブが存在しない状態だった）。この空白を埋めるため
//!    `ci.yml` build ジョブに `cargo check -p backend-cpu --tests --target
//!    aarch64-apple-darwin`（`cfg(test)` 有効かつ `target_os = "macos"`
//!    クロスターゲット。backend-metal 向け同型ステップと同じ手法。
//!    `Makefile` の `check-cross-cpu-tests` と同一コマンド）を追加し、
//!    `sysctl_ffi` を継続的コンパイル検証の対象に含めている（詳細は
//!    `docs/perf/cpu-gemm-runtime-cache-detect.md` §3）。
//! 3. **常に到達可能な公開入口**（[`detected_blocks`]）: `#[cfg(test)]`
//!    ではなく通常ビルドから到達可能な `pub(crate)` 関数とし、
//!    非 macOS・sysctl 失敗時は [`super::default_blocks`] へ
//!    フォールバックする（本モジュール `mod tests` の
//!    `detected_blocks_returns_valid_block_sizes_on_any_platform` が
//!    Linux CI 上でこのフォールバック経路を実行し続ける）。
//!
//! ## 本番未結線
//!
//! [`detected_blocks`] は本番 3 公開関数（[`super::gemm_blis`]／
//! [`super::gemm_blis_parallel`]／[`super::gemm_blis_bias_act_parallel`]）
//! からは呼ばれない（受け入れ条件 2＝実機 5 回中央値での非劣化確認が
//! 本 PR（イシュー #794）のスコープ外のため。Apple M4 Max 実機への到達
//! 手段が本セッション環境（Linux x86_64）に存在せず（`docs/
//! real-hardware-verification-env.local.md` 不在）実機計測を実行できない
//! ことを確認し、#750・#758・#753 と同型の fail-closed 判断として本番
//! 既定を切り替えていない。`docs/perf/cpu-gemm-runtime-cache-detect.md`
//! 参照）。テスト専用のパラメータ化入口（[`super::gemm_blis_parallel_with_blocks`]
//! 等）・実機 A/B ハーネス（`mod tests` の `#[ignore]` テスト）から到達する。
//!
//! ## MC/KC のキャップ・NC の動的算出（イシュー #794）
//!
//! [`compute_blocks`] の KC／MC は「L1D／L2 実容量から導出した理論値」と
//! 「#749 実測（M4 Max・`docs/perf/cpu-gemm-blocking-sweep.md` §7）で
//! 非劣化を確認済みの現行コンパイル時既定（[`super::KC`]＝256・
//! [`super::MC`]＝128）」の小さい方をとる。理論値を無条件に採用しない
//! 理由: #749 は KC／MC の単独拡大が全サイズで劣化することを実測しており、
//! L1D／L2 実容量からの素朴な理論値（M4 Max 代表値では KC≈1228・
//! MC≈1024 相当）はこの劣化域に張り付く。逆に、M4 Max の実測最適点
//! そのものへ一致するよう予算比率等の定数を調整すると `hw.model` 分岐を
//! 「容量から逆算した係数」の形へ置き換えただけの機種固定化になり、
//! PR #766 の撤去理由（本ファイル冒頭「デッドコード化を避ける設計」節）
//! に反する。実測で確認済みの上限をキャップとして課すだけに留めるため、
//! 小容量 L1D／L2 環境（組込み・仮想化等）では理論値がキャップを下回り
//! 引き続き縮む（fail-closed 方向は維持。実測未確認の「拡大」方向にのみ
//! キャップをかける）。
//!
//! NC（本イシューの主眼）は逆に #749 で拡大（NC=9600）が n>=4096 で
//! 改善しているため、キャップを課さず L2 残余からの理論値をそのまま
//! 採用する。KC が上記キャップにより `super::KC`（256）へ収束する結果、
//! B パネル 1 要素あたりの K 方向バイト数が固定され、NC は L2 実容量へ
//! 比例して動的に決まる（M4 Max 代表値では nc_raw ≈ 8192。#749 実測の
//! 改善域〈NC=9600〉に近い値だが、機種名や実測値そのものへの逆算では
//! なく L2 容量からの算出のため PR #766 の撤去理由に抵触しない）。
//! n=2048 で NC 拡大が劣化した実測（#749 §7 (iv)）と非両立の可能性が
//! 残るが、[`detected_blocks`] は上記のとおり本番未結線のため、
//! 形状（n）依存の扱いは実機計測イテレーションで別途判断する
//! （イシュー #794 §8・`docs/perf/cpu-gemm-runtime-cache-detect.md`）。

use std::sync::OnceLock;

use crate::gemm::BlockSizes;

/// f32 要素 1 つのバイト数（GEMM の A/B/C は全て f32。`.claude/rules/coding-rust.md`
/// は本クレートの数値型を f32 に統一しており、本モジュールの容量計算も
/// これに合わせる）。
const F32_BYTES: usize = 4;

/// KC のクランプ下限・上限。下限は極小 L1D（組込み・仮想化環境等）でも
/// マイクロカーネル 1 反復あたりの演算密度を落としすぎないための保守的な
/// 最小値、上限は #749 実測（`docs/perf/cpu-gemm-blocking-sweep.md` §7）の
/// 候補グリッドで検証済みの最大値（KC=4096・firestorm 参照値）に合わせる。
const KC_MIN: usize = 128;
const KC_MAX: usize = 4096;
/// MC のクランプ下限・上限。上限は同じく #749 実測グリッドの firestorm
/// 参照値（MC=480）を上回る余裕を持たせつつ、L2 実効容量の非現実的な
/// 過大評価を防ぐ。
const MC_MIN: usize = 64;
const MC_MAX: usize = 1024;
/// NC のクランプ下限・上限。上限は #749 実測（NC=9600 は n>=4096 でのみ
/// 改善・n=2048 では劣化）を踏まえ、firestorm 参照値（NC=9600）を包含
/// しつつ無制限の拡大を防ぐ値とする。
const NC_MIN: usize = 256;
const NC_MAX: usize = 16384;

/// L1D サイズの正当性検査範囲（バイト）。sysctl の戻り値は外部入力として
/// 扱い、明らかに非現実的な値（0・極小・極大）は算出前に拒否する
/// （OWASP A03・`.claude/rules/security.md`）。下限 4KiB は実在する最小級
/// L1D の下限を大きく下回らない値、上限 8MiB は実在する L1D 実装を
/// 大きく超える安全側の値。
const L1D_SANE_MIN: usize = 4 * 1024;
const L1D_SANE_MAX: usize = 8 * 1024 * 1024;
/// L2 サイズの正当性検査範囲（バイト）。理由は L1D と同じ。
const L2_SANE_MIN: usize = 128 * 1024;
const L2_SANE_MAX: usize = 256 * 1024 * 1024;

/// `x` を `multiple` の倍数へ切り上げる（`multiple == 0` は `x` を素通し）。
///
/// レビュー指摘（#753）: `mr`／`nr`（[`compute_blocks`] 冒頭で 0 のみ検査
/// 済みで上限は検査していない）を `multiple` に渡すと `div_ceil(...) *
/// multiple` が大きな値で `usize` オーバーフローしうる（debug では
/// panic、release ではラップして誤ったブロックサイズを返す）。`checked_mul`
/// で防御し、オーバーフロー時は `None` を返す（呼び出し元は
/// [`compute_blocks`] を経由し最終的に [`super::default_blocks`] への
/// フォールバックへ落ちる。受け入れ条件 3 と同じ fail-closed 方針）。
fn round_up_to_multiple(x: usize, multiple: usize) -> Option<usize> {
    if multiple == 0 {
        return Some(x);
    }
    x.div_ceil(multiple).checked_mul(multiple)
}

/// `raw` を `[min, max]` へクランプしたうえで `multiple` の倍数へ丸める
/// （レビュー指摘 #753: 従来は「切り上げしてからクランプ」の順だったため、
/// クランプが効いた場合に結果が `multiple` の倍数であるという契約が
/// 破れていた。`MC_MIN=64`／`NC_MIN=256` は NEON `NR=12` 等では倍数に
/// ならない値のため、先にクランプしてから倍数へ丸める必要がある）。
///
/// 切り上げた結果が `max` を超える場合（`max` 自体が `multiple` の倍数で
/// ない場合に起こりうる）は、`max` 以下で `multiple` の倍数になる最大値
/// へ切り下げる。これにより返り値は常に `multiple` の倍数であり、かつ
/// `max` を超えない（`min` をわずかに下回りうるが、`multiple` が `min`
/// 以下である通常のマイクロカーネル構成では起こらない）。
///
/// `multiple > max`（切り下げ先が `0` になる）場合は `None` を返す
/// （レビュー指摘 #753: 従来は `(max / multiple) * multiple` が `0` に
/// なる場合をそのまま `Some(0)` として返しており、[`compute_blocks`] の
/// 「MR/NR が非 0 かつ容量が正当なら有効なブロックサイズを返し、算出
/// 不能時は `None` で fail-closed にフォールバックする」契約に反していた。
/// 現行のマイクロカーネル定数〈`ScalarKernel` MR=4/NR=4・`NeonKernel`
/// MR=8/NR=12 等〉は `MC_MAX`／`NC_MAX` を大きく下回るため到達しないが、
/// 将来より大きな MR/NR のマイクロカーネルを追加した際の回帰を防ぐ）。
///
/// [`round_up_to_multiple`] のオーバーフロー検出（`multiple` に極端に
/// 大きな `mr`／`nr` が渡された場合）も `None` として伝播する。
fn clamp_to_multiple(raw: usize, min: usize, max: usize, multiple: usize) -> Option<usize> {
    let clamped = raw.clamp(min, max);
    let rounded = round_up_to_multiple(clamped, multiple)?;
    if rounded <= max {
        return Some(rounded);
    }
    let floor = (max / multiple) * multiple;
    if floor == 0 {
        return None;
    }
    Some(floor)
}

/// L1D／L2 実測値（バイト）と、対象マイクロカーネルの `MR`／`NR`
/// （[`super::microkernel::Microkernel`] の型定数）から MC/KC/NC を算出する
/// 純関数（全プラットフォームで単体テスト可能。#753 §3.1）。
///
/// - **KC**: A マイクロパネル（`MR × KC` 要素）と B マイクロパネル
///   （`KC × NR` 要素）が L1D に共存し追い出し合わない条件から算出する。
///   L1D の連想度は `sysctl` から取得できないため、保守的に「L1D 実容量の
///   半分」を予算とする（残り半分は C アキュムレータタイル・他の常駐
///   データ・連想度に由来する実効容量低下の余裕分。gemm crate `cache.rs`
///   の「L1 連想度と A パネルの追い出し関係を整合させる」技法を、
///   連想度非取得という制約下での保守的な固定仮定として反映したもの）。
/// - **MC**: A パネル（`MC × KC × 4B`）が L2 実容量の一定割合に収まる
///   条件から算出する（予算は同じく半分。同時に B パネルもコアクラスタ
///   共有の L2 に常駐するため）。`MR` の倍数へ切り上げる。
/// - **NC**: L2 の残余容量から B パネル（`KC × NC × 4B`）が収まる上限を
///   算出する。`NR` の倍数へ切り上げる。#749 実測（NC=9600 は n>=4096
///   でのみ改善・n=2048 では劣化）と矛盾しないよう [`NC_MAX`] でクランプ
///   する。
///
/// 各値は [`KC_MIN`]〜[`NC_MAX`] の範囲へクランプする。`l1d_bytes`／
/// `l2_bytes` が正当性検査範囲外、`mr`／`nr` が 0 の場合、または
/// `mr`／`nr` に起因する中間計算（`mr + nr`・`F32_BYTES * (mr + nr)`・
/// [`clamp_to_multiple`] 内の丸め）が `usize` オーバーフローする場合は
/// `None`（呼び出し元は [`super::default_blocks`] へフォールバックする。
/// 受け入れ条件 3。オーバーフロー検出はレビュー指摘 #753: `mr`／`nr` は
/// 0 のみ検査しており上限を検査していなかったため、大きな値で
/// debug では panic・release では誤った丸め値が生じうる状態だった）。
/// オーバーフローしない範囲でも `mr`／`nr` が [`MC_MAX`]／[`NC_MAX`]
/// を超える場合（[`clamp_to_multiple`] が倍数への切り下げ先を `0` としか
/// 表現できない）も同じく `None` とする（レビュー指摘 #753: 従来は
/// `Some(BlockSizes { mc: 0, .. })` 等の無効値をそのまま返しており、
/// 「0 以外の MR/NR と正当な容量なら有効なブロックサイズを返す」契約に
/// 反していた）。
pub(crate) fn compute_blocks(
    l1d_bytes: usize,
    l2_bytes: usize,
    mr: usize,
    nr: usize,
) -> Option<BlockSizes> {
    if mr == 0 || nr == 0 {
        return None;
    }
    if !(L1D_SANE_MIN..=L1D_SANE_MAX).contains(&l1d_bytes) {
        return None;
    }
    if !(L2_SANE_MIN..=L2_SANE_MAX).contains(&l2_bytes) {
        return None;
    }

    let l1_budget = l1d_bytes / 2;
    // `checked_add`／`checked_mul` で防御する（`mr`／`nr` の 0 排除だけでは
    // 上限未検証のため大きな値でオーバーフローしうる。オーバーフロー時は
    // `None` を返し呼び出し元のフォールバックへ委ねる）。
    let per_k_bytes = F32_BYTES.checked_mul(mr.checked_add(nr)?)?;
    if per_k_bytes == 0 {
        // `mr`／`nr` を 0 排除済みのため理論上到達しないが、`compute_blocks`
        // 全体の fail-closed 方針（0 除算防止）に合わせ明示的に拒否する。
        return None;
    }
    // KC は「L1D 実容量から算出できる理論上限」と「#749 実測（M4 Max・
    // `docs/perf/cpu-gemm-blocking-sweep.md` §7）で確認済みの非劣化点
    // `super::KC`（現行コンパイル時既定 256）」の小さい方をとる。理論値を
    // 直接使わない理由（イシュー #794 レビュー指摘）: #749 は KC の
    // 単独拡大が全サイズで劣化することを実測しており、L1D 実容量から
    // 素朴に導出した理論値（M4 Max 代表値では KC≈1228）はこの劣化域に
    // 張り付く。かといって M4 Max 実測値そのものに一致するよう予算比率
    // 等の定数を逆算すると、`hw.model` 分岐を「係数」の形に置き換えた
    // だけの機種固定化になり PR #766 の撤去理由（`cache_params.rs:12-16`
    // モジュールドキュメント）に反する。実測で確認済みの上限を
    // キャップとして課すだけに留め、小容量 L1D 環境（組込み・仮想化等）
    // では理論値がキャップを下回り引き続き縮む（fail-closed 方向は
    // 維持。実測未確認の「拡大」方向にのみキャップをかける）。
    let kc_theoretical = l1_budget / per_k_bytes;
    let kc = kc_theoretical.min(super::KC).clamp(KC_MIN, KC_MAX);

    let l2_budget = l2_bytes / 2;
    let kc_bytes = F32_BYTES.checked_mul(kc)?;
    if kc_bytes == 0 {
        return None;
    }
    // MC も KC と同じ理由（#749: 単独拡大が全サイズで劣化）で
    // `super::MC`（現行既定 128）をキャップとする。
    //
    // レビュー指摘（イシュー #794・codex-review P2・Cursor Bugbot Medium。
    // PR #815）: 従来は「`mc_raw` を `super::MC` へキャップ→
    // `clamp_to_multiple(.., MC_MIN, MC_MAX, mr)` で `mr` の倍数へ丸め」
    // という順序だった。`clamp_to_multiple` 自身は渡された `max`
    // （`MC_MAX=1024`）に対してのみ「丸め上げが `max` を超えたら `max`
    // 以下の倍数へ切り下げる」安全策を持つため、`mr` が `super::MC`
    // （128）の約数でない場合（例: `Avx2Kernel` の `MR=6`）は丸め上げが
    // `super::MC` は超えつつ `MC_MAX` は超えない値（132）に着地し、
    // 非劣化キャップ契約が破られていた。
    //
    // 修正: `clamp_to_multiple` の `max` 引数へ直接 `super::MC` を渡す
    // （`MC_MAX` との `min` で `super::MC` が将来 `MC_MAX` を上回る
    // 設定ミスをしても安全側に倒す）。これにより「丸め上げが `max` を
    // 超えたら `max` 以下の倍数へ切り下げ、それも不可能なら `None`」
    // という既存の（テスト済みの）安全策がそのまま非劣化キャップにも
    // 適用され、倍数契約〈`mc % mr == 0`〉と非劣化キャップ
    //〈`mc <= super::MC`〉の両立を常に保つ。
    let mc_cap = super::MC.min(MC_MAX);
    let mc_raw = l2_budget / kc_bytes;
    let mc = clamp_to_multiple(mc_raw, MC_MIN, mc_cap, mr)?;

    // NC は #794 の主眼（本イシュー: NC 動的算出）。MC/KC と異なり
    // #749 実測では NC 拡大（NC=9600）が n>=4096 で改善しており、
    // キャップを課さず L2 残余（A パネル分を除いた残り。`l2_budget` は
    // A 側に割り当てた予算のため、B 側は残りの半分＝`l2_budget` 自体を
    // 使う）から算出した理論値をそのまま採用する。上記で `kc` が
    // `super::KC`（256）へキャップされた結果、`kc_bytes` は `KC` 由来の
    // 小さい値に固定され、L2 実容量が大きいほど `nc_raw` も比例して
    // 大きくなる（M4 Max 代表値では nc_raw ≈ 8192。#749 実測の
    // 改善域〈NC=9600〉に近い一方、機種名や実測値そのものへの逆算では
    // なく L2 容量からの算出のため PR #766 の撤去理由に抵触しない）。
    // n=2048 で NC 拡大が劣化した実測（#749 §7 (iv)）とは非両立の
    // 可能性が残るが、本関数は実機ゲート未通過のため本番未結線
    // （`detected_blocks` の呼び出し元がテスト専用入口・A/B ハーネスに
    // 限られる。`mod.rs` 冒頭コメント）であり、形状（n）依存の扱いは
    // 実機計測イテレーションで別途判断する（イシュー #794 §8）。
    let nc_raw = l2_budget / kc_bytes;
    let nc = clamp_to_multiple(nc_raw, NC_MIN, NC_MAX, nr)?;

    Some(BlockSizes { mc, kc, nc })
}

/// `sysctlbyname` の書き戻し長 `len` と 8 バイトのゼロ初期化バッファ `buf`
/// から `usize` を組み立てる純関数（FFI 呼び出し（[`sysctl_ffi::read_usize`]）
/// から切り離し、`cfg(target_os = "macos")` に依存せず全プラットフォームで
/// 単体テスト可能にしたもの。`compute_blocks` と同じ設計方針。`mod tests`
/// 参照）。
///
/// レビュー指摘（Cursor Bugbot Medium・PR #773）: Darwin の
/// `hw.perflevel0.*cachesize` 系 sysctl ノードは `CTLTYPE_INT`（`man 3
/// sysctl` の型一覧。4 バイト）でありうるが、旧実装は書き戻し長が
/// `size_of::<usize>()`（64-bit で 8 バイト、`CTLTYPE_QUAD` 相当）と完全
/// 一致しないと失敗扱いにしていた。該当ノードが `CTLTYPE_INT` の実機
/// （M4 実測。macOS 実機不可のため本 PR では型検査のみで実測は未確定）
/// では検出が常に [`super::default_blocks`] へフォールバックし、実機 A/B
/// ハーネス（`docs/perf/cpu-gemm-runtime-cache-detect.md` §3）が「既定 vs
/// 既定」の自己比較になってしまう。
///
/// 本関数は書き戻し長が `4`（`CTLTYPE_INT` 相当。`u32` として解釈し
/// `usize` へ拡張）または `8`（`CTLTYPE_QUAD` 相当。`u64` としてそのまま
/// 解釈）のときのみ受理し、それ以外の長さは `compute_blocks` と同じ
/// fail-closed 方針で `None` を返す。Darwin/aarch64（本モジュールが対象と
/// する唯一の環境。`cfg(target_os = "macos")` はこのファイル内では
/// aarch64-apple-darwin にのみ及ぶ）はリトルエンディアンのため
/// `from_le_bytes` で組み立てて良い。`buf` は呼び出し元
/// （[`sysctl_ffi::read_usize`]）でゼロ初期化してから渡す前提のため、
/// 4 バイト書き戻し時に未使用の上位 4 バイトは `0` であることを利用しない
/// （`4` の分岐で明示的に先頭 4 バイトのみを読む）。
fn assemble_cache_value_le(buf: [u8; 8], len: usize) -> Option<usize> {
    match len {
        4 => {
            let mut b4 = [0u8; 4];
            b4.copy_from_slice(&buf[..4]);
            Some(u32::from_le_bytes(b4) as usize)
        }
        8 => Some(u64::from_le_bytes(buf) as usize),
        _ => None,
    }
}

/// `sysctlbyname`（macOS/BSD 系 libSystem が提供する標準 API）による
/// L1D／L2 実測値の読み取り（`cfg(target_os = "macos")` 限定）。
///
/// 追加クレート依存を使わない理由: `libc` は許容 9 区分
/// （`.claude/rules/deps-policy.md`）外であり、`sysctlbyname` は macOS
/// 実行環境に常にリンクされている libSystem の C ABI 関数のため、
/// `objc2` 系（`cfg(target_os = "macos")` 限定の許容依存）と同様に
/// `extern "C"` の自前宣言で足りる（#749 時点の実装〈PR #766・撤去済み〉
/// と同方式）。
#[cfg(target_os = "macos")]
mod sysctl_ffi {
    use std::ffi::{CString, c_char, c_int, c_void};

    // SAFETY: この `extern "C"` 宣言は macOS/BSD 系 libSystem が公開する
    // 標準 API `sysctlbyname`（`<sys/sysctl.h>`、`man 3 sysctlbyname`）の
    // シグネチャと一致させている: 戻り値は `c_int`（0 は成功、非 0 はエラー。
    // errno 相当）、引数は `name: *const c_char`（NUL 終端文字列）・
    // `oldp: *mut c_void`／`oldlenp: *mut usize`（読み取り先バッファと
    // その長さ、in/out）・`newp: *mut c_void`／`newlen: usize`（書き込み
    // 値、本モジュールでは常に null／0 で読み取り専用呼び出しに限定）で、
    // C ABI 上の型幅・呼び出し規約（`extern "C"`）は libSystem のヘッダ
    // 定義と 1:1 対応する。シンボルは全 macOS 実行環境に常にリンクされる
    // libSystem が提供するため動的ロード不要で解決可能（`cfg(target_os =
    // "macos")` 限定でのみコンパイルされ、他 OS では宣言自体が存在しない）。
    // 個々の呼び出し引数の安全性（ポインタ有効性・長さ整合）は呼び出し側
    // `read_usize` の SAFETY コメントを参照。
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    /// 指定した sysctl 名の `usize` 値を読む。取得失敗（戻り値 != 0）・
    /// 書き戻し長が 4／8 バイトいずれでもない・値 0 は `None`（外部入力の
    /// 検証。OWASP A03・`.claude/rules/security.md`）。
    ///
    /// レビュー指摘（Cursor Bugbot Medium・PR #773）: `hw.perflevel0.*
    /// cachesize` 系 sysctl ノードは `CTLTYPE_INT`（4 バイト）でありうるため、
    /// 従来の「書き戻し長が `size_of::<usize>()`（8 バイト）と完全一致」
    /// 検査は該当ノードで常に失敗し検出が恒久的にフォールバックしていた。
    /// 8 バイトのゼロ初期化バッファへ読み、書き戻し長 4／8 双方を
    /// [`super::assemble_cache_value_le`]（FFI から独立した純関数。単体
    /// テスト対象）で解釈する。
    pub(super) fn read_usize(name: &str) -> Option<usize> {
        // `sysctlbyname` は NUL 終端 C 文字列を要求する契約
        // （`man 3 sysctlbyname`）。`name` に埋め込み NUL が含まれる
        // 呼び出しは本モジュール内の固定文字列のみのため到達しないが、
        // `CString::new` は防御的に検査してから変換する。
        let cname = CString::new(name).ok()?;
        let mut buf = [0u8; 8];
        let mut len = buf.len();
        // SAFETY: `cname` はこの呼び出しの生存期間中有効な NUL 終端 C
        // 文字列。`oldp` は `buf.len()`（8 バイト）ちょうどの有効な書き込み
        // 先（ローカル配列 `buf` の先頭アドレス）で、`oldlenp` にその長さを
        // 渡す（`sysctlbyname` の「呼び出し前に `*oldlenp` へバッファ長を
        // 設定する」契約）。ノードが `CTLTYPE_INT`（4 バイト）でも
        // `CTLTYPE_QUAD`（8 バイト）でも `oldp` の書き込み先は 8 バイト
        // 分確保済みのため書き込み超過は起こらない。`newp` は null・
        // `newlen` は 0 とし「値を変更しない読み取り専用呼び出し」の契約を
        // 満たす。戻り値・書き戻された `len` は呼び出し直後に検査する。
        // FFI 境界の `unsafe` はこの 1 箇所に限定する
        // （`.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要
        // 最小限に留め、理由をコメントで明記」）。
        let ret = unsafe {
            sysctlbyname(
                cname.as_ptr(),
                buf.as_mut_ptr() as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret != 0 {
            return None;
        }
        let value = super::assemble_cache_value_le(buf, len)?;
        if value == 0 {
            return None;
        }
        Some(value)
    }
}

/// Apple Silicon の P コア（`hw.perflevel0`。高性能クラスタ）の L1D／L2
/// 実測値（バイト）を読む。E コア（`hw.perflevel1`）は対象外
/// （`gemm_blis` の実行時 ISA ディスパッチ〈#185〉・マイクロカーネル選定は
/// P コア想定でチューニングされている。#481 §3 と同じ前提）。
#[cfg(target_os = "macos")]
fn read_cache_sizes() -> Option<(usize, usize)> {
    let l1d = sysctl_ffi::read_usize("hw.perflevel0.l1dcachesize")?;
    let l2 = sysctl_ffi::read_usize("hw.perflevel0.l2cachesize")?;
    Some((l1d, l2))
}

/// macOS 以外は sysctl 経路自体を持たないため常に `None`
/// （[`detected_blocks`] が [`super::default_blocks`] へフォールバックする）。
#[cfg(not(target_os = "macos"))]
fn read_cache_sizes() -> Option<(usize, usize)> {
    None
}

/// プロセス内で 1 回だけ [`read_cache_sizes`] を評価しキャッシュする
/// （`sysctl` はプロセス生存期間中不変のため呼び出しごとの再取得は不要）。
static CACHE_SIZES: OnceLock<Option<(usize, usize)>> = OnceLock::new();

fn detected_cache_sizes() -> Option<(usize, usize)> {
    *CACHE_SIZES.get_or_init(read_cache_sizes)
}

/// 実行時検出した L1D／L2 実測値から [`compute_blocks`] で MC/KC/NC を
/// 算出する（非 macOS・sysctl 失敗・算出不能時は [`super::default_blocks`]
/// へフォールバック。受け入れ条件 3）。
///
/// 本番未結線（モジュールドキュメント参照）。テスト専用パラメータ化入口・
/// 実機 A/B ハーネスから `mr`／`nr` に対象マイクロカーネルの
/// [`super::microkernel::Microkernel::MR`]／`NR` を渡して呼ぶ想定。
pub(crate) fn detected_blocks(mr: usize, nr: usize) -> BlockSizes {
    detected_cache_sizes()
        .and_then(|(l1d, l2)| compute_blocks(l1d, l2, mr, nr))
        .unwrap_or_else(super::default_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_blocks_apple_m4_max_like_values_stays_within_clamp_bounds() {
        // Apple M4 Max P コア相当の代表値（L1D=192KiB・L2=16MiB。#481 §3）。
        // NEON 既定カーネル（MR=8・NR=12。#559）で算出する。
        let blocks = compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 12)
            .expect("正当な範囲の値は Some を返すはず");
        assert!((KC_MIN..=KC_MAX).contains(&blocks.kc));
        assert!((MC_MIN..=MC_MAX).contains(&blocks.mc));
        assert!((NC_MIN..=NC_MAX).contains(&blocks.nc));
        // MC は MR（8）の倍数へ丸めている。
        assert_eq!(blocks.mc % 8, 0);
        assert_eq!(blocks.nc % 12, 0);
    }

    #[test]
    fn compute_blocks_m4_max_like_values_kc_mc_cap_to_current_defaults() {
        // レビュー指摘（イシュー #794）: 既存の
        // `compute_blocks_apple_m4_max_like_values_stays_within_clamp_bounds`
        // は range 内であることしか検証しておらず、
        // `.min(super::KC)`／`.min(super::MC)`（本 PR の主要な挙動変更＝
        // キャップ）を削除しても pass し続けるため新挙動を保証しない。
        // M4 Max P コア相当（L1D=192KiB・L2=16MiB・NEON MR=8/NR=12）では
        // 理論値（KC≈1228・MC≈1024 相当）がキャップを上回るため、
        // キャップが効いて `kc == super::KC`（256）・`mc == super::MC`
        // （128）へ一致することを明示的に固定する。
        let blocks = compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 12)
            .expect("正当な範囲の値は Some を返すはず");
        // `mod tests` は `cache_params` の子モジュールのため、`super` は
        // `cache_params` を指す（`KC`／`MC` の定義元 `gemm_blis` ではない）。
        // `cache_params` 本体（`mod tests` の外）の `super::KC` と異なり、
        // ここでは `super::super::KC` で `gemm_blis::KC` を参照する。
        assert_eq!(
            blocks.kc,
            super::super::KC,
            "kc は super::KC へキャップされるはず"
        );
        assert_eq!(
            blocks.mc,
            super::super::MC,
            "mc は super::MC へキャップされるはず"
        );
    }

    #[test]
    fn compute_blocks_nc_grows_monotonically_with_l2_capacity() {
        // レビュー指摘（イシュー #794）: NC はキャップなしで L2 実容量に
        // 比例して動的に算出される（本 PR の主眼）。この挙動を直接検証する
        // テストがなかったため、L2 サイズが異なる 2 ケース（4 MiB と
        // 16 MiB。KC/MC は上記キャップにより両ケースとも
        // `super::KC`／`super::MC` へ収束し固定されるため、NC の差分のみが
        // L2 容量差を反映する）で NC が単調増加することを固定する。
        let small_l2 = compute_blocks(192 * 1024, 4 * 1024 * 1024, 8, 12)
            .expect("正当な範囲の値は Some を返すはず");
        let large_l2 = compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 12)
            .expect("正当な範囲の値は Some を返すはず");
        assert!(
            large_l2.nc > small_l2.nc,
            "L2 実容量が大きいほど nc も大きくなるはず（small={}, large={}）",
            small_l2.nc,
            large_l2.nc
        );
    }

    #[test]
    fn compute_blocks_rejects_zero_mr_or_nr() {
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, 0, 12).is_none());
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 0).is_none());
    }

    #[test]
    fn compute_blocks_rejects_zero_cache_sizes() {
        assert!(compute_blocks(0, 16 * 1024 * 1024, 8, 12).is_none());
        assert!(compute_blocks(192 * 1024, 0, 8, 12).is_none());
    }

    #[test]
    fn compute_blocks_rejects_implausibly_small_cache_sizes() {
        // 正当性検査範囲（[`L1D_SANE_MIN`]／[`L2_SANE_MIN`]）を下回る値は
        // sysctl の異常値（破損・不正取得）とみなし拒否する。
        assert!(compute_blocks(64, 16 * 1024 * 1024, 8, 12).is_none());
        assert!(compute_blocks(192 * 1024, 4096, 8, 12).is_none());
    }

    #[test]
    fn compute_blocks_rejects_implausibly_large_cache_sizes() {
        // 正当性検査範囲（[`L1D_SANE_MAX`]／[`L2_SANE_MAX`]）を超える値は
        // 非現実的（sysctl の異常値・将来の桁違いなハードウェア変化）と
        // みなし拒否する（0 値検査だけでは捉えられない fail-closed 契約）。
        assert!(compute_blocks(64 * 1024 * 1024, 16 * 1024 * 1024, 8, 12).is_none());
        assert!(compute_blocks(192 * 1024, 1024 * 1024 * 1024, 8, 12).is_none());
    }

    #[test]
    fn compute_blocks_clamped_mc_nc_stay_multiples_for_non_divisor_mr_nr() {
        // レビュー指摘（#753）: `MC_MIN`／`NC_MIN`（64／256）は `mr`／`nr`
        // が 7 のような非自明な値のとき倍数にならない。クランプが効く
        // 状況（L2 実容量を極端に大きくし `mc_raw`／`nc_raw` を
        // `MC_MAX`／`NC_MAX` へ張り付かせる）でも `mc`／`nc` が `mr`／`nr`
        // の倍数であるというドキュメント契約（本ファイル冒頭・
        // `docs/perf/cpu-gemm-runtime-cache-detect.md` §2）が成立し続ける
        // ことを検証する（M4 Max 相当値〈MR=8・NR=12〉は `MC_MAX=1024`・
        // `NC_MAX=16384` がたまたま倍数のため、この回帰は非自明な
        // `mr`／`nr` でなければ検出できない）。
        let blocks = compute_blocks(192 * 1024, 1024 * 1024 * 1024 / 8, 7, 13)
            .or_else(|| compute_blocks(192 * 1024, L2_SANE_MAX, 7, 13))
            .expect("正当な範囲の値は Some を返すはず");
        assert_eq!(blocks.mc % 7, 0, "mc={} は mr=7 の倍数ではない", blocks.mc);
        assert_eq!(
            blocks.nc % 13,
            0,
            "nc={} は nr=13 の倍数ではない",
            blocks.nc
        );
        assert!((MC_MIN..=MC_MAX).contains(&blocks.mc));
        assert!((NC_MIN..=NC_MAX).contains(&blocks.nc));
    }

    /// レビュー指摘（イシュー #794・codex-review P2・Cursor Bugbot Medium。
    /// PR #815）の回帰テスト: `mr` が `super::MC`（128）の約数でない場合
    /// （`Avx2Kernel` の `MR=6`。128 は 6 の倍数でないため `clamp_to_multiple`
    /// が 128 へ丸め上げると 132 になり非劣化キャップを超えていた）でも
    /// `mc <= super::MC` が常に成立することを、L2 実容量を極端に大きく
    /// して `mc_raw` をキャップへ張り付かせた状態で検証する。
    #[test]
    fn compute_blocks_mc_never_exceeds_non_degradation_cap_for_non_divisor_mr() {
        // AVX2 マイクロカーネル実値（MR=6・NR=16）を使う。L2 を
        // `L2_SANE_MAX` まで大きくし `mc_raw` が確実に `super::MC` へ
        // 張り付く（キャップされる）状況を作る。
        let blocks = compute_blocks(192 * 1024, L2_SANE_MAX, 6, 16)
            .expect("正当な範囲の値は Some を返すはず");
        assert!(
            blocks.mc <= super::super::MC,
            "mc={} は非劣化キャップ super::MC={} を超えている（#794 レビュー指摘の回帰）",
            blocks.mc,
            super::super::MC
        );
        assert_eq!(blocks.mc % 6, 0, "mc={} は mr=6 の倍数ではない", blocks.mc);
        assert!((MC_MIN..=MC_MAX).contains(&blocks.mc));
    }

    #[test]
    fn compute_blocks_small_l1d_clamps_kc_to_minimum() {
        // L1D 実容量が小さいほど算出 KC は縮むが、KC_MIN を下回らない
        // （マイクロカーネル 1 反復あたりの演算密度を落としすぎない下限）。
        let blocks = compute_blocks(L1D_SANE_MIN, 16 * 1024 * 1024, 8, 12).unwrap();
        assert_eq!(blocks.kc, KC_MIN);
    }

    #[test]
    fn compute_blocks_rejects_mr_nr_overflowing_intermediate_arithmetic() {
        // レビュー指摘（#753）: `mr`／`nr` は 0 のみ検査しており上限を
        // 検査していなかったため、`mr + nr`（`checked_add`）・
        // `F32_BYTES * (mr + nr)`（`checked_mul`）が `usize` オーバーフロー
        // する組み合わせで debug では panic・release では誤った丸め値が
        // 生じうる状態だった。`usize::MAX` に近い `mr`／`nr` で `None`
        // （fail-closed。呼び出し元は `default_blocks()` へフォールバック）
        // を返すことを検証する。
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, usize::MAX, 1).is_none());
        assert!(
            compute_blocks(
                192 * 1024,
                16 * 1024 * 1024,
                usize::MAX / 2,
                usize::MAX / 2 + 2
            )
            .is_none()
        );
        // `clamp_to_multiple` 側（`round_up_to_multiple` の `checked_mul`）
        // も同様に防御されていることを、`mc_raw`／`nc_raw` が確実にクランプ
        // される極端に大きな L2 実容量と、それ自体は `mr + nr` を
        // オーバーフローさせない範囲の巨大な `mr` の組み合わせで検証する。
        assert!(compute_blocks(192 * 1024, L2_SANE_MAX, usize::MAX / 4, 12).is_none());
    }

    #[test]
    fn compute_blocks_rejects_mr_exceeding_mc_max_without_overflow() {
        // レビュー指摘（#753）: `mr = 2048` は `usize` オーバーフローを
        // 起こさないが `MC_MAX = 1024` を超えるため、従来の
        // `clamp_to_multiple` は `(MC_MAX / mr) * mr == 0` をそのまま
        // `Some(mc = 0)` として返していた（「0 以外の MR/NR と正当な容量
        // なら有効なブロックサイズを返す」契約違反）。現行のマイクロ
        // カーネル定数（最大でも `NeonKernel` MR=8/NR=12）はこの値に到達
        // しないが、将来より大きな MR/NR を追加した際の回帰を防ぐため
        // `None`（fail-closed）を返すことを固定する。
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, 2048, 12).is_none());
        // 同じ理由で NC 側（`nr` が `NC_MAX = 16384` を超える場合）も
        // `None` を返すことを検証する。
        assert!(compute_blocks(192 * 1024, 16 * 1024 * 1024, 8, 32768).is_none());
    }

    /// [`detected_blocks`] は非 macOS・sysctl 失敗時に必ず
    /// [`super::default_blocks`] 相当の非ゼロ値を返す（受け入れ条件 3。
    /// Linux CI 上でもこのフォールバック経路を実行し続けることで、
    /// 本モジュールが「常に不活性なデッドコード」（PR #766 で撤去された
    /// 構成）に陥っていないことを検証する）。
    #[test]
    fn detected_blocks_returns_valid_block_sizes_on_any_platform() {
        let blocks = detected_blocks(8, 12);
        assert!(blocks.mc > 0 && blocks.kc > 0 && blocks.nc > 0);
    }

    /// レビュー指摘（Cursor Bugbot Medium・PR #773）の回帰テスト:
    /// `hw.perflevel0.*cachesize` 系ノードが `CTLTYPE_INT`（4 バイト）で
    /// 書き戻された場合でも正しく解釈できることを FFI を経由せず検証する。
    #[test]
    fn assemble_cache_value_le_accepts_4_byte_ctltype_int() {
        // 192 * 1024 = 196608 (0x00030000) を 4 バイト・リトルエンディアンで格納。
        let mut buf = [0u8; 8];
        buf[..4].copy_from_slice(&192u32.wrapping_mul(1024).to_le_bytes());
        assert_eq!(assemble_cache_value_le(buf, 4), Some(192 * 1024));
    }

    /// 8 バイト（`CTLTYPE_QUAD` 相当）の書き戻しは従来どおり受理する。
    #[test]
    fn assemble_cache_value_le_accepts_8_byte_ctltype_quad() {
        let value: usize = 16 * 1024 * 1024;
        let buf = (value as u64).to_le_bytes();
        assert_eq!(assemble_cache_value_le(buf, 8), Some(value));
    }

    /// 4／8 以外の書き戻し長は fail-closed で `None`（`compute_blocks` と
    /// 同じ方針。異常値をそのまま数値として解釈しない）。
    #[test]
    fn assemble_cache_value_le_rejects_other_lengths() {
        let buf = [0u8; 8];
        assert!(assemble_cache_value_le(buf, 0).is_none());
        assert!(assemble_cache_value_le(buf, 1).is_none());
        assert!(assemble_cache_value_le(buf, 3).is_none());
        assert!(assemble_cache_value_le(buf, 5).is_none());
        assert!(assemble_cache_value_le(buf, 7).is_none());
    }
}
