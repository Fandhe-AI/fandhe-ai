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
///
/// fail-closed 契約（`docs/autodiff-determinism-mode-design.md` §3.3・
/// セキュリティ基準「検査スクリプト fail-open 禁止」）: ディレクトリ
/// 列挙・エントリ取得・ファイル読み取りのいずれかが失敗した場合、
/// その失敗を空扱いへ読み替えて検査を通してしまうと、実際には
/// allowlist 突合の対象から漏れたファイルが「rayon マーカーなし」と
/// 誤判定されうる。従って各失敗はテスト panic として fail-closed に
/// 伝播させる（検査失敗をそのままテスト失敗として可視化する）。
fn visit_rs_files(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| {
        panic!(
            "determinism_inventory: read_dir({}) に失敗した（fail-closed:\
             走査失敗を「検査対象なし」へ読み替えない）: {e}",
            dir.display()
        )
    });
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| {
            panic!(
                "determinism_inventory: {} 配下のエントリ列挙に失敗した\
                 （fail-closed）: {e}",
                dir.display()
            )
        });
        paths.push(entry.path());
    }
    paths.sort();
    for path in paths {
        if path.is_dir() {
            visit_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(RS_EXT) {
            let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "determinism_inventory: {} の読み取りに失敗した\
                     （fail-closed）: {e}",
                    path.display()
                )
            });
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
/// `.par_chunks_mut(`／`.into_par_iter(`／`.par_iter_mut(`）と
/// 縮約マーカー（`.sum()`／`.sum::<T>()` 等の型指定付き turbofish 呼び
/// 出し／`.reduce(`／`reduce_with(`）が共起する箇所を数える（分割依存の
/// 縮約——rayon の並列イテレータをそのまま `.sum()`／`.reduce(` へ流す
/// 典型パターンの検出。`docs/autodiff-determinism-mode-design.md`
/// §2.3）。`.par_chunks_mut(` は GEMM 等の書き込み先チャンク分割で実際に
/// 使われるマーカーであり、これを欠くと backend-cpu の並列イテレータ
/// 使用箇所の一部が検出対象から漏れる（codex-review 指摘・PR #2274）。
///
/// 検出は 2 段構成: (1) 同一文（`;` 区切り）中の共起（従来どおり）。
/// (2) `let (mut )?IDENT = <par marker を含む式>;` で並列イテレータの
/// 結果を変数へ代入し、後続の文で `IDENT.` に対して縮約マーカーを
/// 呼び出す「変数代入を挟んで文をまたぐ並列縮約」（同一文分割のみでは
/// 検出漏れになる。codex-review 指摘・PR #2274）。縮約マーカーの側も
/// `.sum(` 前方一致に加え `.sum::<` を別マーカーとして扱う。`.sum(` は
/// `.sum::<f32>()` のような turbofish では `sum` の直後に `(` が来ず
/// `::<` が挟まるため単独では検出できない。
fn count_par_reduce_cooccurrences(content: &str) -> usize {
    let cleaned = strip_comments_and_strings(content);
    let par_markers = [
        ".par_iter(",
        ".par_chunks(",
        ".par_chunks_mut(",
        ".into_par_iter(",
        ".par_iter_mut(",
    ];
    let reduce_markers = [".sum(", ".sum::<", ".reduce(", "reduce_with("];

    let statements: Vec<&str> = cleaned.split(';').collect();

    // 並列イテレータの結果が代入された識別子集合（文をまたいだ検出用）。
    // 一度汚染された識別子は関数末尾まで汚染済みとみなす保守的近似
    // （再代入・シャドーイングの追跡はしない。fail-closed 側に倒す）。
    let mut tainted: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut count = 0usize;

    for stmt in &statements {
        let has_par = par_markers.iter().any(|m| stmt.contains(m));
        let has_reduce = reduce_markers.iter().any(|m| stmt.contains(m));

        if has_par && has_reduce {
            // (1) 同一文中の共起。
            count += 1;
        } else if has_reduce {
            // (2) 過去の文で汚染された識別子に対する縮約呼び出し。
            if tainted
                .iter()
                .any(|ident| stmt.contains(&format!("{ident}.")))
            {
                count += 1;
            }
        }

        if has_par && let Some(ident) = assigned_identifier(stmt) {
            tainted.insert(ident);
        }
    }

    count
}

/// `let (mut )?IDENT = ...` 形の文から代入先識別子を抽出する
/// （[`count_par_reduce_cooccurrences`] の文をまたぐ汚染追跡が使う）。
/// 該当しない場合は `None`。
///
/// [`count_par_reduce_cooccurrences`] は `;` のみで文を分割するため、
/// ブロック境界（`{`／`}`）をまたいだ直後の文（関数本体の最初の文・
/// if/for ブロック本体の最初の文等）は、直前のブロック開始トークン
/// （関数宣言・条件式・`{` 自体）が同じ `;` 区切りチャンクに含まれて
/// しまう。例えば関数本体の最初の文 `fn f() {\n    let it =
/// data.par_iter();` は `;` 分割後も 1 チャンク
/// `"fn f() {\n    let it = data.par_iter()"` のままで、先頭が
/// `"let "` ではないため代入として検出できず、後続の `it.sum()` を
/// 見逃していた（codex-review 指摘・PR #2274 review r4105558135）。
/// 対処として、チャンク内最後の `{`／`}` より後ろだけを実効的な文と
/// みなす（ブロック境界をまたいだ前段のテキストを読み飛ばす）。
fn assigned_identifier(stmt: &str) -> Option<&str> {
    let block_start = stmt.rfind(['{', '}']).map(|i| i + 1).unwrap_or(0);
    let stmt = &stmt[block_start..];

    let rest = stmt.trim_start().strip_prefix("let ")?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix("mut ").unwrap_or(rest).trim_start();
    let ident_end = rest.find(|c: char| !(c.is_alphanumeric() || c == '_'))?;
    let ident = &rest[..ident_end];
    if ident.is_empty() {
        return None;
    }
    let after = rest[ident_end..].trim_start();
    // 型注釈 `: T` の有無に関わらず代入先を検出する（`let ident = ...`
    // だけでなく `let ident: T = data.par_iter();` のような型注釈付き
    // let も並列イテレータの汚染源として追跡する必要がある。codex-review
    // 指摘・Bugbot 指摘 PR #2274: `assigned_identifier` が識別子直後の
    // `=` のみを代入とみなしていたため、型注釈を挟む代入が汚染集合から
    // 漏れ、後続の `.sum()`／`.reduce(` が並列縮約カウントから漏れて
    // いた）。`after` を走査し、最初に現れる単独の `=`（`==` の一部で
    // はなく、直前が `!`／`<`／`>`／`=` でもないもの）を代入演算子とみ
    // なす。型注釈本体（`: T` 部分）に単独 `=` が現れるケース（const
    // generics のデフォルト値等）は本テストが対象とするコードパターン
    // には現れないため割り切る。
    let bytes = after.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'=' {
            continue;
        }
        let next_is_eq = bytes.get(i + 1) == Some(&b'=');
        let prev_is_cmp = i > 0 && matches!(bytes[i - 1], b'!' | b'<' | b'>' | b'=');
        if !next_is_eq && !prev_is_cmp {
            return Some(ident);
        }
    }
    None
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
/// 入る規模の MLP（batch 512・in 256・hidden 512・out 10）の
/// forward・backward を実行し、`(loss の to_bits, dW1 の to_bits 列,
/// db1 の to_bits 列)` を返す。`mse_loss` の全縮約要素数は
/// `batch * out_dim = 5120` で `reduction.rs::CHUNK`〈4096〉を超え
/// （4096 要素の第 1 チャンク＋ 1024 要素の端数チャンクの 2 チャンク
/// 構成）、チャンク内逐次 → チャンク間結合という複数チャンク縮約の
/// 経路とスレッド数不変性を実際に検証できる形状にしている
/// （batch 64・out 10＝640 要素は CHUNK 未満で単一チャンクに収まり
/// 複数チャンク縮約を検証できていなかった。codex-review 指摘・
/// PR #2274）。
fn run_mlp(seed: u64) -> (u32, Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((s >> 40) % 2000) as f32 - 1000.0) * 0.001
    };

    let batch = 512usize;
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

// =====================================================================
// count_par_reduce_cooccurrences の検出回帰テスト
// （codex-review 指摘・PR #2274: 文をまたぐ並列縮約・turbofish 縮約の
// 検出漏れの再発防止）
// =====================================================================

#[test]
fn count_par_reduce_detects_same_statement_cooccurrence() {
    let src = "let s: f32 = data.par_iter().sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_variable_assignment_across_statements() {
    // 変数代入を挟んで文をまたぐ並列縮約（allowlist 対象ファイル内で
    // 検査をすり抜けていたパターン。codex-review 指摘・PR #2274）。
    let src = "let it = data.par_iter(); let s: f32 = it.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_typed_let_assignment_across_statements() {
    // 型注釈付き let（`let it: T = data.par_iter();`）を挟んで文をまた
    // ぐ並列縮約。`assigned_identifier` が識別子直後の `=` のみを代入と
    // みなしていたため、型注釈を挟むケースは汚染集合から漏れ、後続の
    // `.sum()` が検出をすり抜けていた（codex-review 指摘・Bugbot 指摘・
    // PR #2274）。
    let src = "let it: Vec<f32> = data.par_iter().collect(); let s = it.iter().sum::<f32>();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_turbofish_sum() {
    // `.sum::<f32>()` は `.sum(` 前方一致では検出できない
    // （codex-review 指摘・PR #2274）。
    let src = "let s = data.par_iter().sum::<f32>();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_unrelated_statements() {
    let src = "let x = 1 + 2; let y = data.iter().sum::<f32>(); let z = data.par_iter().count();";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_does_not_double_count_same_statement_hit() {
    // 同一文で共起した場合は (1) の分岐だけがカウントし、(2) の分岐と
    // 二重計上しない。
    let src = "let s: f32 = data.par_iter().sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn assigned_identifier_extracts_let_binding() {
    assert_eq!(assigned_identifier("let x = 1"), Some("x"));
    assert_eq!(
        assigned_identifier("let mut y = data.par_iter()"),
        Some("y")
    );
    assert_eq!(assigned_identifier("x == 1"), None);
    assert_eq!(assigned_identifier("data.iter()"), None);
}

#[test]
fn count_par_reduce_detects_first_statement_in_function_body() {
    // 関数本体の最初の文（ブロック開始 `{` 直後の `let`）に対する
    // 検出漏れの再発防止（codex-review 指摘・PR #2274 review
    // r4105558135）。`;` 分割のみでは `fn f() {\n let it =
    // data.par_iter();` が 1 チャンクとして残り、先頭が `let ` では
    // ないため代入を検出できず、後続の `it.sum()` を見逃していた。
    let src = "fn f(data: &[f32]) -> f32 {\n    let it = data.par_iter();\n    it.sum()\n}";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn assigned_identifier_strips_block_boundary_prefix_before_let() {
    // ブロック開始直後の `let`（`{` がチャンク内に混入するケース）で
    // も代入先識別子を抽出できる（codex-review 指摘・PR #2274 review
    // r4105558135）。
    assert_eq!(
        assigned_identifier("fn f() {\n    let it = data.par_iter()"),
        Some("it")
    );
    // 複数ブロックをまたぐ場合も最後のブロック境界より後ろだけを見る。
    assert_eq!(
        assigned_identifier("if x { let it = data.par_iter()"),
        Some("it")
    );
}

#[test]
fn assigned_identifier_extracts_typed_let_binding() {
    // 型注釈付き let（`let ident: T = ...`）も代入として追跡する
    // （codex-review 指摘・Bugbot 指摘・PR #2274）。
    assert_eq!(
        assigned_identifier("let it: Vec<f32> = data.par_iter().collect()"),
        Some("it")
    );
    assert_eq!(
        assigned_identifier("let mut it: f32 = data.par_iter().sum()"),
        Some("it")
    );
    // 型注釈のみで代入が無い場合（`;` 区切りの空文等）は None を保つ。
    assert_eq!(assigned_identifier("let it: Vec<f32>"), None);
}
