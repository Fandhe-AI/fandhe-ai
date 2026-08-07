//! `backend-cpu` のビルドスクリプト。
//!
//! AVX-512 intrinsics（`std::arch::x86_64::_mm512_*`）と
//! `#[target_feature(enable = "avx512f")]` は、Rust の stable チャネルで
//! 段階的に安定化された（`stdarch_x86_avx512` ライブラリ機能・
//! `avx512_target_feature` の 2 系統。安定化バージョンをここで断定的に
//! 記述すると将来の rustc リリースでずれるおそれがあるため、バージョン
//! 比較ではなく実際にコンパイルを試す probe 方式を採る）。
//!
//! self-hosted CI runner（`.claude/rules/ci.md`）は「未導入の場合のみ
//! 導入する」冪等セルフヒール方針（`rustup show` のみで `rustup update`
//! は行わない）のため、rustc の実バージョンは runner ごとに異なりうる
//! （PR #337 の CI 実測: `runs-on: self-hosted` プール中の 1 台が rustc
//! 1.88.0 でビルド失敗〈E0658 計 14 件〉。同一 PR の他ジョブは 1.89 以上の
//! runner に着地して成功していたと推測される）。
//!
//! [`crate::gemm_blis::microkernel::avx512`] モジュールおよび
//! [`crate::gemm_blis::microkernel::Avx512Kernel`] は本スクリプトが
//! probe 成功時に発行する `avx512_stable` cfg が立っている場合のみ
//! コンパイル対象となる（未安定化の rustc では AVX2 へ自動的に
//! フォールバックする。[`crate::gemm_blis::mod::dispatch_region`] 参照）。
//! rustc バージョンではなく実コンパイルで判定する理由:
//! `is_x86_feature_detected!` は実行時 CPU 機能検出でありコンパイラの
//! stable 化状況とは無関係のため使えず、バージョン番号の決め打ちは
//! 安定化タイミングの記憶違い・将来の rustc 変更に対して脆いため。
//!
//! probe は x86_64 ターゲット時のみ実行する（[`microkernel`] モジュール
//! 側もモジュール宣言を `target_arch = "x86_64"` で絞っており、他 arch
//! では avx512 経路自体が存在しないため probe 自体が無意味）。

use std::env;
use std::path::Path;
use std::process::Command;

/// AVX-512F ロード intrinsics と `#[target_feature(enable = "avx512f")]`
/// の双方を用いる最小スニペット。PR #337 の CI で実際に E0658 を出した
/// 2 系統（`stdarch_x86_avx512` ライブラリ機能・`avx512_target_feature`）
/// をそのまま踏むことで、「この rustc でこの機能が使えるか」を
/// バージョン番号を介さず直接検査する。
const PROBE_SRC: &str = r#"
#[target_feature(enable = "avx512f")]
unsafe fn probe(p: *const f32) -> std::arch::x86_64::__m512 {
    unsafe { std::arch::x86_64::_mm512_loadu_ps(p) }
}
"#;

fn main() {
    // 未知の cfg として `-D warnings` に倒されないよう、本スクリプトが
    // 発行しうる cfg 名を明示登録する（Rust 1.80+ の check-cfg lint 対応。
    // 単一コロン記法は cargo 1.77 未満でも黙って無視されるだけで壊れない
    // ため、プール内の古い cargo に対する保険として単一コロンを使う）。
    println!("cargo:rustc-check-cfg=cfg(avx512_stable)");

    // avx512 モジュールは x86_64 のみが対象（[`crate::gemm_blis::microkernel`]
    // 参照）。他 arch では probe 自体が不要（cfg を立てても本体コードは
    // どのみち `target_arch = "x86_64"` でゲートされ効果がない）。
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch != "x86_64" {
        return;
    }

    if probe_avx512_compiles() {
        println!("cargo:rustc-cfg=avx512_stable");
    }
}

/// [`PROBE_SRC`] を実際に `rustc --emit=metadata` へ通し、AVX-512F
/// intrinsics と `target_feature` が stable rustc でコンパイル可能かを
/// 実測する。失敗（プロセス起動不可・非 0 終了）は安全側（AVX-512 無効）
/// に倒す。
fn probe_avx512_compiles() -> bool {
    // Cargo は `RUSTC` 環境変数経由で呼び出し対象の rustc バイナリを渡す
    // （クロスコンパイル・ツールチェーン上書き時も正しいバイナリを指す）。
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out_dir = match env::var_os("OUT_DIR") {
        Some(dir) => dir,
        None => return false,
    };
    let target = env::var("TARGET").unwrap_or_default();

    let probe_path = Path::new(&out_dir).join("avx512_probe.rs");
    if std::fs::write(&probe_path, PROBE_SRC).is_err() {
        return false;
    }
    let metadata_out = Path::new(&out_dir).join("avx512_probe.rmeta");

    let mut cmd = Command::new(&rustc);
    cmd.arg("--edition")
        .arg("2021")
        .arg("--crate-type")
        .arg("lib")
        .arg("--emit=metadata")
        .arg("-o")
        .arg(&metadata_out)
        .arg(&probe_path);
    if !target.is_empty() {
        cmd.arg("--target").arg(&target);
    }

    matches!(cmd.output(), Ok(output) if output.status.success())
}
