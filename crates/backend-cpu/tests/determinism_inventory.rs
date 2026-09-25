//! 決定性モード（イシュー #2157・親 #2131）の棚卸し結果
//! （`docs/autodiff-determinism-mode-design.md` §2）を固定する fail-closed
//! インベントリテスト。
//!
//! (a) ソース走査: `crates/backend-cpu/src/**/*.rs` の中で rayon 並列
//!     イテレータ（`par_iter`／`par_chunks`／`into_par_iter`／
//!     `rayon::`）を使うファイル集合が allowlist と完全一致すること
//!     （過不足いずれも検出。新しいファイルが並列化を導入した場合、
//!     決定性の根拠を棚卸しし本ファイルの allowlist を更新するまで
//!     CI を落とす）。rayon 並列イテレータと `.sum()`／`.reduce(`／
//!     `reduce_with` が同一文中に共起する箇所（分割依存の縮約——結果が
//!     スレッド割り当てに依存しうる典型パターン）が 0 件であること。
//!     `fetch_add`（等の atomic read-modify-write 蓄積）が本番未結線の
//!     診断コード 1 箇所（`gemm_blis/mod.rs`。`#[cfg(test)]` ゲート済み）
//!     以外に存在しないこと。
//!
//! (b) end-to-end スレッド数不変性: `Tape::new_with_ops(Box::new(
//!     CpuBackendOps::new()))` 上で並列経路に入る規模の MLP を実行し、
//!     1 スレッド・4 スレッドで forward・backward が bit 完全一致する
//!     こと。決定性モード ON でも同様。

use std::path::{Path, PathBuf};

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::determinism::set_deterministic;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::Tensor;

// =====================================================================
// (a) ソース走査インベントリ
// =====================================================================

/// [`rayon_marker_files`] などが辿るディレクトリ探索の対象拡張子。
const RS_EXT: &str = "rs";

/// `crates/backend-cpu/src` を再帰走査し `(相対パス, 内容)` を集める。
fn visit_rs_files(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            visit_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(RS_EXT) {
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            out.push((path, content));
        }
    }
}

fn backend_cpu_src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// rayon 並列イテレータ・`rayon::` 直接参照のいずれかを含むファイル名
/// （`src/` からの相対パス、`/` 区切り）の固定 allowlist。
/// `docs/autodiff-determinism-mode-design.md` §2.1 の実測結果（24 件）。
/// `gemm_prefetch_bandwidth_diag_tests.rs` は `#[cfg(all(test,
/// target_arch = "aarch64"))]` ゲート済みのテスト専用ファイル（`src/
/// lib.rs:191`）で、本番ビルドに一切含まれないが「rayon マーカーを
/// 含むファイル」としては引き続き allowlist に載せる（走査自体は
/// テキスト走査でありビルド設定を見ないため）。
const RAYON_MARKER_FILE_ALLOWLIST: &[&str] = &[
    "batch_norm.rs",
    "bce.rs",
    "elementwise.rs",
    "fused_elementwise.rs",
    "gb10_affinity.rs",
    "gemm.rs",
    "gemm_blis/mod.rs",
    "gemm_blis/partition.rs",
    "gemm_prefetch_bandwidth_diag_tests.rs",
    "huber.rs",
    "kl_div.rs",
    "layer_norm.rs",
    "lib.rs",
    "mse.rs",
    "nll.rs",
    "ops.rs",
    "reduction.rs",
    "rmsnorm.rs",
    "rnn_cell.rs",
    "scalar_elementwise.rs",
    "small_shape_thread_cap.rs",
    "softmax.rs",
    "thread_limit.rs",
    "typed_f64.rs",
];

fn contains_rayon_marker(content: &str) -> bool {
    content.contains("par_iter")
        || content.contains("par_chunks")
        || content.contains("into_par_iter")
        || content.contains("rayon::")
}

#[test]
fn rayon_marker_files_match_fixed_allowlist() {
    let src_dir = backend_cpu_src_dir();
    let mut files = Vec::new();
    visit_rs_files(&src_dir, &mut files);

    let mut found: Vec<String> = files
        .iter()
        .filter(|(_, content)| contains_rayon_marker(content))
        .map(|(path, _)| {
            path.strip_prefix(&src_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    found.sort();

    let mut expected: Vec<String> = RAYON_MARKER_FILE_ALLOWLIST
        .iter()
        .map(|s| s.to_string())
        .collect();
    expected.sort();

    assert_eq!(
        found, expected,
        "crates/backend-cpu/src の rayon 並列イテレータ使用ファイル集合が\
         固定 allowlist（docs/autodiff-determinism-mode-design.md §2.1）\
         からドリフトしている（過不足いずれも fail-closed に検出する）。\
         新しいファイルが並列化を導入した場合は決定性の根拠を棚卸しし\
         allowlist を更新すること: found={found:?}"
    );
}

/// コメント・文字列リテラルを大雑把に除去する（`//` 行コメント・`/* */`
/// ブロックコメント・`"..."` 文字列リテラルを空白へ置換）。厳密な
/// トークナイザではないが、本テストの目的（コード上の rayon 並列
/// イテレータと `.sum()`／`.reduce(` の共起検出）には十分な近似。
fn strip_comments_and_strings(content: &str) -> String {
    let mut out = String::with_capacity(content.len());
    let mut chars = content.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' && chars.peek() == Some(&'/') {
            for c2 in chars.by_ref() {
                if c2 == '\n' {
                    out.push('\n');
                    break;
                }
            }
            continue;
        }
        if c == '/' && chars.peek() == Some(&'*') {
            chars.next();
            let mut prev = ' ';
            for c2 in chars.by_ref() {
                if prev == '*' && c2 == '/' {
                    break;
                }
                prev = c2;
            }
            continue;
        }
        if c == '"' {
            let mut escaped = false;
            for c2 in chars.by_ref() {
                if escaped {
                    escaped = false;
                    continue;
                }
                if c2 == '\\' {
                    escaped = true;
                    continue;
                }
                if c2 == '"' {
                    break;
                }
            }
            out.push(' ');
            continue;
        }
        out.push(c);
    }
    out
}

/// rayon 並列イテレータのマーカー（`.par_iter(`／`.par_chunks(`／
/// `.into_par_iter(`／`.par_iter_mut(`）と `.sum()`／`.reduce(`／
/// `reduce_with(` が同一文（`;` 区切り）中に共起する箇所を数える
/// （分割依存の縮約——rayon の並列イテレータをそのまま `.sum()`／
/// `.reduce(` へ流す典型パターンの検出。`docs/autodiff-determinism-
/// mode-design.md` §2.3）。
fn count_par_reduce_cooccurrences(content: &str) -> usize {
    let cleaned = strip_comments_and_strings(content);
    let par_markers = [
        ".par_iter(",
        ".par_chunks(",
        ".into_par_iter(",
        ".par_iter_mut(",
    ];
    let reduce_markers = [".sum()", ".reduce(", "reduce_with("];
    cleaned
        .split(';')
        .filter(|stmt| {
            par_markers.iter().any(|m| stmt.contains(m))
                && reduce_markers.iter().any(|m| stmt.contains(m))
        })
        .count()
}

/// `#[cfg(all(test, target_arch = "aarch64"))]` ゲート済み（`src/
/// lib.rs:191`）で本番ビルドに一切含まれない、帯域計測専用の診断
/// テストファイル。ベンチマーク用の read-sum 計測（`par_chunks` →
/// `.sum()`）は数値結果ではなくスループット計測が目的であり、
/// 決定性契約（本番経路の縮約順序）の対象外として明示的に除外する
/// （`docs/autodiff-determinism-mode-design.md` §2.1 の allowlist
/// コメントと同じ理由）。
const DIAG_TEST_ONLY_FILE_EXCLUSIONS: &[&str] = &["gemm_prefetch_bandwidth_diag_tests.rs"];

#[test]
fn no_rayon_parallel_reduce_cooccurrence_in_backend_cpu_src() {
    let src_dir = backend_cpu_src_dir();
    let mut files = Vec::new();
    visit_rs_files(&src_dir, &mut files);

    let mut offending: Vec<String> = Vec::new();
    for (path, content) in &files {
        let rel = path
            .strip_prefix(&src_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if DIAG_TEST_ONLY_FILE_EXCLUSIONS.contains(&rel.as_str()) {
            continue;
        }
        let count = count_par_reduce_cooccurrences(content);
        if count > 0 {
            offending.push(format!(
                "{}: {count} 件",
                path.strip_prefix(&src_dir).unwrap_or(path).display()
            ));
        }
    }
    assert!(
        offending.is_empty(),
        "rayon 並列イテレータと .sum()/.reduce(/reduce_with が同一文中に\
         共起する箇所が見つかった（分割依存の縮約はスレッド割り当てに\
         結果が依存しうるため決定性契約に違反する疑いがある。\
         docs/autodiff-determinism-mode-design.md §2.3 の棚卸し結果は\
         0 件だったため、この検出は棚卸し後の回帰を意味する）: {offending:?}"
    );
}

/// atomic read-modify-write（`fetch_add`／`fetch_sub`／
/// `compare_exchange`／`fetch_or`／`fetch_and`／`fetch_max`／
/// `fetch_min`）の出現行数を数える。
fn count_atomic_rmw_occurrences(content: &str) -> usize {
    let cleaned = strip_comments_and_strings(content);
    let markers = [
        "fetch_add",
        "fetch_sub",
        "compare_exchange",
        "fetch_or",
        "fetch_and",
        "fetch_max",
        "fetch_min",
    ];
    cleaned
        .lines()
        .filter(|line| markers.iter().any(|m| line.contains(m)))
        .count()
}

#[test]
fn atomic_rmw_occurrences_match_expected_test_only_count() {
    let src_dir = backend_cpu_src_dir();
    let mut files = Vec::new();
    visit_rs_files(&src_dir, &mut files);

    let mut total = 0usize;
    let mut per_file: Vec<(String, usize)> = Vec::new();
    for (path, content) in &files {
        let count = count_atomic_rmw_occurrences(content);
        if count > 0 {
            let rel = path
                .strip_prefix(&src_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            per_file.push((rel, count));
            total += count;
        }
    }

    // `docs/autodiff-determinism-mode-design.md` §2.2・§2.4 実測:
    // 実コード上の atomic read-modify-write 出現は
    // `gemm_blis/mod.rs::gemm_blis_ic_dynamic_region`（`#[cfg(test)]`
    // ゲート済み・本番未結線の診断コード）内の 1 箇所のみ。本番経路に
    // atomic 蓄積は存在しない。
    assert_eq!(
        total, 1,
        "atomic read-modify-write の出現数が期待（1 件・gemm_blis/mod.rs\
         の #[cfg(test)] 限定診断コードのみ）と一致しない（新規の atomic\
         蓄積が本番経路へ混入した可能性がある）: per_file={per_file:?}"
    );
    assert_eq!(
        per_file,
        vec![("gemm_blis/mod.rs".to_string(), 1usize)],
        "atomic read-modify-write の出現元ファイルが期待\
         （gemm_blis/mod.rs のみ）と一致しない: {per_file:?}"
    );
}

// =====================================================================
// (b) end-to-end スレッド数不変性
// =====================================================================

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `Tape::new_with_ops(Box::new(CpuBackendOps::new()))` 上で並列経路に
/// 入る規模の MLP（batch 64・in 256・hidden 512・out 10。全縮約の要素数
/// は `reduction.rs::CHUNK`〈4096〉を超える）の forward・backward を
/// 実行し、`(loss の to_bits, dW1 の to_bits 列, db1 の to_bits 列)` を
/// 返す。
fn run_mlp(seed: u64) -> (u32, Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 40) % 2000) as f32 - 1000.0) * 0.001
    };

    let batch = 64usize;
    let in_dim = 256usize;
    let hidden = 512usize;
    let out_dim = 10usize;

    let x_data: Vec<f32> = (0..batch * in_dim).map(|_| next()).collect();
    let w1_data: Vec<f32> = (0..in_dim * hidden).map(|_| next()).collect();
    let b1_data: Vec<f32> = (0..hidden).map(|_| next()).collect();
    let w2_data: Vec<f32> = (0..hidden * out_dim).map(|_| next()).collect();
    let target_data: Vec<f32> = (0..batch * out_dim).map(|_| next()).collect();

    let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let x = tape.var(&t(x_data, &[batch, in_dim]));
    let w1 = tape.var(&t(w1_data, &[in_dim, hidden]));
    let b1 = tape.var(&t(b1_data, &[hidden]));
    let w2 = tape.var(&t(w2_data, &[hidden, out_dim]));
    let target = tape.var(&t(target_data, &[batch, out_dim]));

    let h = x.matmul(&w1).unwrap();
    let h = h.add(&b1).unwrap();
    let h = h.relu();
    let out = h.matmul(&w2).unwrap();
    let loss = out.mse_loss(&target).unwrap();

    let loss_bits = loss
        .to_tensor()
        .get(&[])
        .expect("mse_loss はスカラー")
        .to_bits();

    let grads = tape.backward(&loss).unwrap();
    let dw1_bits: Vec<u32> = grads
        .get(&w1)
        .unwrap()
        .expect("w1 は loss に到達する")
        .as_slice()
        .expect("w1 は contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let db1_bits: Vec<u32> = grads
        .get(&b1)
        .unwrap()
        .expect("b1 は loss に到達する")
        .as_slice()
        .expect("b1 は contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect();

    (loss_bits, dw1_bits, db1_bits)
}

/// 1 スレッド・4 スレッドの rayon スコープで `run_mlp` の結果が bit
/// 完全一致することを、決定性モード OFF／ON の両方で確認する。
#[test]
fn cpu_backend_mlp_forward_backward_is_thread_count_invariant() {
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build single-thread rayon pool");
    let multi = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("failed to build 4-thread rayon pool");

    const SEED: u64 = 0x243F_6A88_85A3_08D3;

    // 決定性モード OFF（既定）。
    let a = single.install(|| run_mlp(SEED));
    let b = multi.install(|| run_mlp(SEED));
    assert_eq!(a, b, "決定性モード OFF でスレッド数により結果が変わった");

    // 決定性モード ON でも同一結果（no-op 契約。
    // docs/autodiff-determinism-mode-design.md §0・§3）。
    set_deterministic(true);
    let c = single.install(|| run_mlp(SEED));
    let d = multi.install(|| run_mlp(SEED));
    set_deterministic(false);
    assert_eq!(c, d, "決定性モード ON でスレッド数により結果が変わった");
    assert_eq!(
        a, c,
        "決定性モード ON/OFF で同一スレッド数でも結果が変わった\
         （no-op 契約違反）"
    );
}
