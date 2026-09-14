//! イシュー #1706 調査: MSL `bfloat` 型・`simdgroup_bfloat8x8` の
//! コンパイル可用性を実機でコンパイルプローブし、将来のデバイス常駐
//! ネイティブ bf16 経路（bf16 のまま H2D・`simdgroup_bfloat8x8` GEMM 等）
//! の実現可能性を判断する材料を整備する診断テスト群。
//!
//! # 位置づけ（調査限定。本番結線なし）
//!
//! `crate::typed_bf16`（`impl TypedOps<half::bf16> for MetalBackendOps`。
//! ホスト側変換＋既存 f32 カーネル委譲）の実装可否は本ファイルの結果に
//! **依存しない**（`typed_bf16.rs` モジュール doc 「(a) と (b) の分離」
//! 参照）。本ファイルが決めるのは (b)（ネイティブ bf16 経路の実現可能性）
//! のみであり、`gemm.rs`／`shaders/gemm.metal`／`tile.rs`／`Cargo.toml`
//! への変更を一切含まない（新規診断テストファイルの追加のみ）。
//!
//! # MSL ソースの配置（`src/shaders/` に置かない理由）
//!
//! `crate::gemm_mpp_diag_tests` と同じ判断: `src/shaders/*.metal` は
//! 本番 `include_str!` 対象・`shader_source_evidence` テストの走査対象
//! のため、本調査専用のソースをそこへ置くと本番シェーダ集合に誤って
//! 混入したように見える。本モジュール内の `const &str` として保持する。
//!
//! # 配置理由
//!
//! `crate::context::MetalContext`・`crate::pipeline::compile_options`
//! （`pub(crate)`）へ到達するため、integration test ではなく `lib.rs`
//! の兄弟モジュールとして配置する（`crate::gemm_mpp_diag_tests` と
//! 同じ理由）。`objc2` 系 FFI 型に触れるため
//! `cfg(all(test, target_os = "macos"))` を付ける。
//!
//! # 非 gating（成否を `println!` で記録し panic しない）
//!
//! 可否自体が調査対象のデータであり、失敗を異常系として扱わない
//! （`crate::gemm_mpp_diag_tests::mpp_metal4_compile_probe` と同方針）。
//! すべて `#[ignore]`（Metal 実機依存）。
//!
//! # unsafe の範囲
//!
//! 新規 `unsafe` は本ファイル内（`#[cfg(test)]` 限定）に閉じ、
//! `crate::gemm_mpp_diag_tests::encode_mpp_nn` と同一形の FFI 呼び出し
//! （`setBuffer_offset_atIndex`／`newBufferWithBytes_length_options`）
//! のみを追加する。本番コードへの unsafe 追加はない。
//!
//! # 実測記入欄
//!
//! 本エージェント実行環境（Linux）には Apple Silicon 実機への到達手段が
//! ないため、本ファイルの `#[ignore]` テストは未実測のまま Mac
//! セッションへ申し送る。実行コマンド・保存先は
//! `docs/perf/logs/metal-typed-bf16-probe-1706/README.md`・実測結果の
//! 転記先は `docs/backend-dtype-dispatch-design.md` §13.6 を参照。

use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLGPUFamily, MTLLanguageVersion,
    MTLSize,
};

use crate::context::MetalContext;
use crate::error::MetalError;
use crate::pipeline::{MtlLibrary, make_pipeline};

/// P1: `bfloat` スカラー／`bfloat4`・`float(bfloat)`／`bfloat(float)`
/// 変換を含む最小カーネル。MSL 3.1+ で可能という期待
/// （`.claude/skills/apple-silicon/references/msl/data-types.md` の
/// 仕様参照。実測で確定する）。
const P1_BFLOAT_SCALAR_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void p1_bfloat_scalar_probe(
    device const float* in_ptr [[buffer(0)]],
    device float* out_ptr [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    bfloat b = static_cast<bfloat>(in_ptr[gid]);
    bfloat4 b4 = bfloat4(b, b, b, b);
    out_ptr[gid] = static_cast<float>(b4.x);
}
"#;

/// P2: `simdgroup_bfloat8x8` の `simdgroup_load` ＋
/// `simdgroup_multiply_accumulate(simdgroup_float8x8&, simdgroup_bfloat8x8,
/// simdgroup_bfloat8x8, simdgroup_float8x8)`。既存 f16 カーネル
/// （`gemm_simdgroup_f16`・`simdgroup_half8x8` A/B＋`simdgroup_float8x8`
/// 累算。イシュー #380）と同型の精度契約（bf16 フラグメント・f32
/// 累算）を bf16 で試す。
const P2_SIMDGROUP_BFLOAT_MMA_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void p2_simdgroup_bfloat_mma_probe(
    device const bfloat* a_ptr [[buffer(0)]],
    device const bfloat* b_ptr [[buffer(1)]],
    device float* c_ptr [[buffer(2)]])
{
    simdgroup_bfloat8x8 sa;
    simdgroup_bfloat8x8 sb;
    simdgroup_float8x8 sc = simdgroup_float8x8(0.0f);

    simdgroup_load(sa, a_ptr, 8);
    simdgroup_load(sb, b_ptr, 8);
    simdgroup_multiply_accumulate(sc, sa, sb, sc);
    simdgroup_store(sc, c_ptr, 8);
}
"#;

/// P3: `simdgroup_store(simdgroup_float8x8, device bfloat*)`。f16 版
/// （イシュー #380 実機確定）では `simdgroup_store(float8x8, device
/// half*)` が型不一致でコンパイル不可だったため、bf16 でも同様の
/// 制約が見込まれる（型不一致で不可の見込み。実測で確定する）。
const P3_SIMDGROUP_STORE_BFLOAT_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void p3_simdgroup_store_bfloat_probe(
    device const bfloat* a_ptr [[buffer(0)]],
    device const bfloat* b_ptr [[buffer(1)]],
    device bfloat* c_ptr [[buffer(2)]])
{
    simdgroup_bfloat8x8 sa;
    simdgroup_bfloat8x8 sb;
    simdgroup_float8x8 sc = simdgroup_float8x8(0.0f);

    simdgroup_load(sa, a_ptr, 8);
    simdgroup_load(sb, b_ptr, 8);
    simdgroup_multiply_accumulate(sc, sa, sb, sc);
    // 型不一致が見込まれる箇所（P2 は device float* へ store・本箇所は
    // device bfloat* へ store）。
    simdgroup_store(sc, c_ptr, 8);
}
"#;

/// P4 数値スモーク（P1 可の場合のみ意味を持つ）。`float(bfloat(in[i]))`
/// を実行し、ホスト `half::bf16::from_f32(in[i]).to_f32()` と比較する
/// 決定的入力（tie ケース・±inf・NaN クラス・通常値）。
const P4_ROUNDTRIP_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void p4_bfloat_roundtrip_probe(
    device const float* in_ptr [[buffer(0)]],
    device float* out_ptr [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // 手動境界チェック（REQ-8。coding-rust.md「カーネル実装の境界検査」）:
    // dispatch grid はスレッドグループ単位でしか割り当てられないため、
    // n がスレッドグループ幅で割り切れない場合に末尾のスレッドが
    // 配列外を読み書きしうる。本カーネルは調査専用の非本番コードだが、
    // 規約を一貫して適用する。
    if (gid >= n) {
        return;
    }
    bfloat b = static_cast<bfloat>(in_ptr[gid]);
    out_ptr[gid] = static_cast<float>(b);
}
"#;

/// [`crate::gemm_mpp_diag_tests::compile_mpp_source`] と同型のコンパイル
/// ヘルパー。`crate::pipeline::compile_options()`（本番既定と同一の
/// 丸め方針）に対し、`lang` が `Some` の場合のみ言語版を明示
/// オーバーライドする（本番既定言語版のままコンパイルできるかどうか
/// 自体も運用上有用な事実のため、`None` と `Some(Version3_1)` の
/// 2 条件を用意する。実装計画 §3.3）。
fn compile_probe_source(
    device: &crate::context::MtlDevice,
    src: &str,
    lang: Option<MTLLanguageVersion>,
) -> Result<Retained<MtlLibrary>, MetalError> {
    let options = crate::pipeline::compile_options();
    if let Some(v) = lang {
        options.setLanguageVersion(v);
    }
    let ns_src = NSString::from_str(src);
    device
        .newLibraryWithSource_options_error(&ns_src, Some(&options))
        .map_err(|err| MetalError::LibraryCompilation {
            message: err.localizedDescription().to_string(),
        })
}

/// 1 つのソースを (i) 本番既定言語版・(ii) `Version3_1` 明示 の 2 条件で
/// コンパイルし、結果を `println!` する（非 gating。呼び出し元テスト
/// 名を `label` として記録し、後で `docs/perf/logs/
/// metal-typed-bf16-probe-1706/compile_probe.log` から grep しやすく
/// する）。
fn probe_compile_both_lang_conditions(device: &crate::context::MtlDevice, label: &str, src: &str) {
    match compile_probe_source(device, src, None) {
        Ok(_) => println!("bf16_probe label={label} lang=default compile_result=ok"),
        Err(e) => {
            println!("bf16_probe label={label} lang=default compile_result=error message={e:?}")
        }
    }
    match compile_probe_source(device, src, Some(MTLLanguageVersion::Version3_1)) {
        Ok(_) => println!("bf16_probe label={label} lang=3.1 compile_result=ok"),
        Err(e) => println!("bf16_probe label={label} lang=3.1 compile_result=error message={e:?}"),
    }
}

/// P0: デバイス属性の記録（`crate::gemm_mpp_diag_tests::
/// mpp_metal4_compile_probe` の Apple7/8/9・アーキテクチャ名版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn p0_device_attributes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let device = ctx.device();

    let arch_name = device.architecture().name().to_string();
    let supports_apple7 = device.supportsFamily(MTLGPUFamily::Apple7);
    let supports_apple8 = device.supportsFamily(MTLGPUFamily::Apple8);
    let supports_apple9 = device.supportsFamily(MTLGPUFamily::Apple9);
    let supports_metal3 = device.supportsFamily(MTLGPUFamily::Metal3);

    println!(
        "bf16_probe label=p0_device_attributes device_architecture={arch_name} \
         supports_apple7={supports_apple7} supports_apple8={supports_apple8} \
         supports_apple9={supports_apple9} supports_metal3={supports_metal3}"
    );
}

/// P1: `bfloat` スカラー／`bfloat4`・変換のコンパイル可否（2 言語版条件）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn p1_bfloat_scalar_compile_probe() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    probe_compile_both_lang_conditions(ctx.device(), "p1_bfloat_scalar", P1_BFLOAT_SCALAR_SRC);
}

/// P2: `simdgroup_bfloat8x8` の load／mma のコンパイル可否（2 言語版
/// 条件）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn p2_simdgroup_bfloat_mma_compile_probe() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    probe_compile_both_lang_conditions(
        ctx.device(),
        "p2_simdgroup_bfloat_mma",
        P2_SIMDGROUP_BFLOAT_MMA_SRC,
    );
}

/// P3: `simdgroup_store(simdgroup_float8x8, device bfloat*)` のコンパイル
/// 可否（型不一致で不可の見込み。実測で確定する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn p3_simdgroup_store_bfloat_compile_probe() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    probe_compile_both_lang_conditions(
        ctx.device(),
        "p3_simdgroup_store_bfloat",
        P3_SIMDGROUP_STORE_BFLOAT_SRC,
    );
}

/// [`p4_bfloat_roundtrip_numeric_smoke`] が使う決定的乱数入力
/// （tie ケース・±inf・NaN・通常値を含む 256 要素）。
fn p4_probe_inputs() -> Vec<f32> {
    let mut inputs = vec![
        1.0f32 + f32::from_bits(0x3b80_0000), // bf16 tie ケース（2^-8）
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
    ];
    // 決定的疑似乱数（`bench_harness` に依存させず、xorshift を
    // その場で複製する。P4 は数値スモークのみが目的で決定性の乱数生成
    // 器の選定自体に意味はないため、テスト内に閉じた最小実装で足りる）。
    let mut state: u64 = 0x1706_bf16;
    for _ in 0..250 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // 上位 32bit を符号なし整数として取り出し `u32::MAX` で正規化する
        // （`f32::from_bits` へ直接投げると指数部が偏り極小値に潰れる。
        // レビュー指摘: 旧実装は `state >> 40` で 24bit しか残らず
        // 全反復で同一の極小値近傍〈= -500.0〉に収束していた）。
        let raw = (state >> 32) as u32;
        let unit = raw as f32 / u32::MAX as f32; // [0.0, 1.0]
        let v = unit * 1000.0 - 500.0; // [-500.0, 500.0]
        inputs.push(v);
    }
    inputs
}

/// P4: `float(bfloat(in[i]))` の数値スモーク（P1 可の場合のみ意味を
/// 持つ。P1 が不可ならこのテストもコンパイル失敗として同じ理由で
/// 失敗する）。ホスト `half::bf16::from_f32` と bit 単位で突き合わせ、
/// 一致件数・不一致件数を `println!` する（非 gating）。
///
/// P1〜P3 は本番既定言語版・`Version3_1` 明示のいずれかが成功すれば
/// bfloat 利用可能と判定する（`probe_compile_both_lang_conditions`）ため、
/// 本テストも同じ 2 条件フォールバック（既定 → 失敗時のみ `Version3_1`）
/// でコンパイルする。既定が失敗し `Version3_1` のみ成功する環境で
/// 早期 return すると (b) roundtrip 数値データが収集されず Mac
/// セッションへ引き継がれないため（Cursor Bugbot 指摘・イシュー #1706
/// PR #1803 レビュー）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn p4_bfloat_roundtrip_numeric_smoke() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let device = ctx.device();

    let library = match compile_probe_source(device, P4_ROUNDTRIP_SRC, None) {
        Ok(lib) => lib,
        Err(default_err) => {
            match compile_probe_source(
                device,
                P4_ROUNDTRIP_SRC,
                Some(MTLLanguageVersion::Version3_1),
            ) {
                Ok(lib) => {
                    println!(
                        "bf16_probe label=p4_bfloat_roundtrip lang=default compile_result=error \
                         message={default_err:?} lang=3.1 compile_result=ok"
                    );
                    lib
                }
                Err(v31_err) => {
                    println!(
                        "bf16_probe label=p4_bfloat_roundtrip lang=default compile_result=error \
                         message={default_err:?} lang=3.1 compile_result=error message={v31_err:?} \
                         (P1 が不可の場合はこの失敗も想定内。p1_bfloat_scalar_compile_probe の \
                         結果を先に確認すること)"
                    );
                    return;
                }
            }
        }
    };
    let pipeline = make_pipeline(device, &library, "p4_bfloat_roundtrip_probe")
        .expect("p4_bfloat_roundtrip_probe パイプラインの構築に失敗した");

    let inputs = p4_probe_inputs();
    let n = inputs.len();

    let in_buf = crate::buffer::MetalBuffer::new_with_data(&ctx, &inputs)
        .expect("input upload must succeed");
    let out_buf =
        crate::buffer::MetalBuffer::new_zeroed(&ctx, n).expect("output allocation must succeed");

    let thread_execution_width = pipeline.threadExecutionWidth();
    let tg_width = thread_execution_width.min(n).max(1);
    let n_u32 = n as u32;
    ctx.encode(
        "p4_bfloat_roundtrip_probe",
        &[in_buf.raw(), out_buf.raw()],
        None,
        |encoder| {
            encoder.setComputePipelineState(&pipeline);
            // SAFETY: `crate::gemm_mpp_diag_tests::encode_mpp_nn` と同一
            // の契約。`in_buf`／`out_buf` は `synchronize` 完了まで
            // 呼び出し元スタックフレームで生存する（`ctx.encode` が
            // `resources` を通じて `in_flight` へ retain する）。
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(in_buf.raw()), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(out_buf.raw()), 0, 1);
            }
            // SAFETY: `n_u32` はローカル変数で `setBytes` 呼び出しの間
            // 生存し、渡す長さ（`size_of::<u32>()`）はポインタ先の型と
            // 一致する（`crate::gemm_mpp_diag_tests::encode_mpp_nn` と
            // 同型の契約）。
            unsafe {
                encoder.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&n_u32).cast(),
                    std::mem::size_of::<u32>(),
                    2,
                );
            }
            // `dispatchThreads_threadsPerThreadgroup`（非一様
            // threadgroup）は本 objc2-metal バージョンにバインディング
            // が存在しないため、`dispatchThreadgroups_threadsPerThreadgroup`
            // （タイル境界を超えうる）＋カーネル側の手動境界チェック
            // （`n` 引数。REQ-8）で構成する。
            let threadgroups = MTLSize {
                width: n.div_ceil(tg_width),
                height: 1,
                depth: 1,
            };
            let threads_per_tg = MTLSize {
                width: tg_width,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
        },
    )
    .expect("p4_bfloat_roundtrip_probe dispatch must succeed");

    ctx.synchronize().expect("synchronize must succeed");
    let out = out_buf.read_to_vec();

    let mut mismatches = 0usize;
    for (i, &v) in inputs.iter().enumerate() {
        let host_expected = half::bf16::from_f32(v).to_f32();
        let device_actual = out[i];
        let matches = if host_expected.is_nan() {
            device_actual.is_nan()
        } else {
            host_expected.to_bits() == device_actual.to_bits()
        };
        if !matches {
            mismatches += 1;
            println!(
                "bf16_probe label=p4_bfloat_roundtrip mismatch idx={i} input={v} \
                 host_expected={host_expected} device_actual={device_actual}"
            );
        }
    }
    println!(
        "bf16_probe label=p4_bfloat_roundtrip total={n} mismatches={mismatches} \
         matches={}",
        n - mismatches
    );
}
