//! 決定性モード（イシュー #2157・親 #2131）の棚卸し結果
//! （`docs/autodiff-determinism-mode-design.md` §2）を固定する fail-closed
//! インベントリテスト。
//!
//! (a) ソース走査: `crates/backend-cpu/src/**/*.rs` の中で rayon 並列
//!     イテレータ（`par_iter`／`par_chunks`／`into_par_iter`／
//!     `rayon::`）を使うファイル集合が allowlist と完全一致すること
//!     （過不足いずれも検出。新しいファイルが並列化を導入した場合、
//!     決定性の根拠を棚卸しし本ファイルの allowlist を更新するまで
//!     CI を落とす）。rayon 並列イテレータ（メソッド呼び出し形・
//!     メソッド値／パス参照形・UFCS 形いずれも）と `.sum()`／
//!     `.reduce(`／`reduce_with`／`.product(`／`.fold(` 等の縮約が、
//!     レキシカルスコープ・ブロック深さ・括弧深さを考慮した文境界の
//!     上で共起する箇所（分割依存の縮約——結果がスレッド割り当てに
//!     依存しうる典型パターン）が 0 件であること（走査方式は下記
//!     §「文境界・レキシカルスコープ・汚染追跡の方式」を参照。
//!     PR #2274 で 6 ラウンド、その後の敵対的レビューでさらに 4 件の
//!     検出漏れ・過検出が指摘され、その都度走査方式を是正した）。
//!     `fetch_add`（等の atomic read-modify-write 蓄積）が本番未結線
//!     の診断コード 1 箇所（`gemm_blis/mod.rs`。`#[cfg(test)]` ゲート
//!     済み）以外に存在しないこと。
//!
//! (b) end-to-end スレッド数不変性: `Tape::new_with_ops(Box::new(
//!     CpuBackendOps::new()))` 上で並列経路に入る規模の MLP を実行し、
//!     1 スレッド・4 スレッドで forward・backward が bit 完全一致する
//!     こと。決定性モード ON でも同様。
//!
//! ## 文境界・レキシカルスコープ・汚染追跡の方式
//!
//! 旧実装は `cleaned.split(';')` によるフラットな文分割ヒューリス
//! ティックで、関数本体最初の `let`・型注釈 `let`・turbofish
//! `.sum::<T>()`・`.par_bridge(`・balanced ブロック初期化式などを順に
//! 継ぎ足しで対応してきたが、6 ラウンド連続で同類型の検出漏れが
//! 指摘された（未閉じ `{` の剥がしが残余の `}` を無視する等）。その後、
//! 文分割をブロック深さ考慮の再帰下降へ置き換えたが、汚染集合の
//! 管理を「`fn` という語が文の先頭付近に現れたら丸ごと `clear()`
//! する」ヒューリスティックで行っていたため、ローカル `fn`・
//! `fn(i32) -> i32` 型注釈・ローカル `impl` を挟むだけで汚染集合が
//! 全消去される、UFCS 呼び出しを検出できない、`Vec` への `collect`
//! で連鎖が切れた初期化式を誤って汚染するといった過検出・検出漏れが
//! 敵対的レビューで指摘された。現行方式は次の 5 点で構成する:
//!
//! 1. **字句前処理**（`strip_comments_and_strings`）: `//`／`/* */`
//!    （ネスト対応）・`"..."`・raw string（`r"..."`／`r#"..."#`）・
//!    byte string（`b"..."`／`br#"..."#`）・char/byte literal（`'{'`／
//!    `';'`／`'\''`／`'\u{7b}'` 等）を空白へ置換する。ライフタイム
//!    （`'a`／`'static`）は「エスケープ列または 1 文字の直後に閉じ `'`
//!    が続く」場合のみ char literal とみなす判定により誤って消費しない。
//! 2. **ブロック深さを考慮した文境界**（`statement_spans`）:
//!    `()`／`[]`／`{}` の深さを追跡し、深さ 0 での `;` と、深さ 0 に
//!    戻る `}` のうち直後（空白を挟んで）が式の継続（`.`／`?`／二項
//!    演算子／`[`／`(`／`else`／`catch`）でないものを文の終端とする。
//! 3. **縮約マーカーの括弧深さ条件**（`has_reduce_after`）: 並列マーカー
//!    （メソッド呼び出し形 `.par_iter(` 等・パス参照形 `T::par_iter`）
//!    より後方かつ同じか浅い括弧深さに縮約マーカーが現れる場合のみ
//!    共起とみなす。加えて `ParallelIterator::sum(...)`・
//!    `Trait::reduce(...)` のような UFCS（fully-qualified）呼び出し
//!    構文も `has_ufcs_reduction`（`UFCS_REDUCE_MARKERS`）が別途走査し、
//!    呼び出し引数リストの内側に並列マーカーまたは汚染識別子があれば
//!    1 件とカウントする（同一文内で他規則と二重計上しない）。
//! 4. **`.collect(` による並列→逐次の連鎖の遮断**: 並列マーカー・
//!    汚染識別子と縮約マーカーの間に、同じか浅い深さの `.collect(`
//!    （順序保持が型で明示される `Vec` への collect。`COLLECT_MARKERS`）
//!    が挟まる場合は接続しないとみなす（`has_reduce_after`）。同じ規則
//!    は文をまたぐ汚染の**発生源**判定（`initializer_taints`）にも
//!    適用する: `let parts = data.par_iter().map(f)
//!    .collect::<Vec<f32>>();` のように初期化式が Vec 確定で終わる
//!    場合は束縛先を汚染しない（`.collect().into_iter().fold(..)` と
//!    同じ「順序保持 collect 後は決定的」という理由）。`.collect(` の
//!    後に再び並列マーカーが現れる場合（`x.par_iter()
//!    .collect::<Vec<_>>().par_iter()`）はその後方のマーカー自身が
//!    汚染源になる。
//! 5. **レキシカルスコープを持つ関数単位の汚染追跡**
//!    （`count_in_scope`。再帰処理）: 文中の任意位置の `let`
//!    （型注釈・タプルパターンを含む。`parse_let_binding_range`）
//!    または `let` を伴わない単純代入 `IDENT = <式>;`
//!    （`parse_plain_assignment_range`）を検出し、初期化式が汚染源
//!    （4 の規則）を含めば束縛名を汚染する（`classify_taint`。
//!    クロージャ本体やブロック式初期化子も初期化式のテキストをその
//!    まま見る規則で自然に捕捉する）。汚染集合はレキシカルスコープ
//!    単位で管理する: **fn アイテムの本体は空集合から始め**（fn は
//!    ローカル変数をキャプチャしない）、それ以外のネストしたブロック
//!    （if／for／loop／match アーム／ブロック式／クロージャ本体／
//!    impl・mod 内の非 fn アイテム）は現在の汚染集合を引き継ぐ。
//!    ネストしたブロック内の `let` による汚染はそのブロックに閉じ
//!    親へ漏らさないが、`let` を伴わない単純代入による汚染だけは
//!    親スコープへ伝播する（fn アイテムの境界をまたぐ場合は伝播しない）。
//!
//! 同一文中の共起判定（`has_depth_aware_par_reduce`・
//! `has_ufcs_reduction`）は、文内にネストした `{...}` ブロック本体を
//! 空白で塗りつぶしたテキスト（`statement_own_view`）に対して行う
//! （ネスト内は `count_in_scope` の再帰が別のレキシカルスコープとして
//! 検査するため二重計上しない）。ただし `let` 初期化式の汚染判定
//! （`classify_taint`）は塗りつぶし前の初期化式全体を見る（`let it =
//! { let x = 1; data.par_iter() };` のような場合に `.par_iter(` を
//! 見失わないため）。
//!
//! ## 既知の限界
//!
//! 関数引数・戻り値経由（並列イテレータを別関数へ渡して内部で縮約
//! する）の値の流れは追跡しない。この経路の導入自体は
//! `no_parallel_iterator_valued_function_signature_in_backend_cpu_src`
//! が、戻り値型に `ParallelIterator`／`IndexedParallelIterator` を含む
//! 関数定義や、それらを引数型に取る関数定義（`impl ParallelIterator`
//! 等）が `crates/backend-cpu/src` に 0 件であることを fail-closed に
//! 検査することで、経路そのものの導入を検知する。

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::determinism::set_deterministic;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::Tensor;

// =====================================================================
// (a) ソース走査インベントリ
// =====================================================================

/// `rayon_marker_files` などが辿るディレクトリ探索の対象拡張子。
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

/// `text` 中に識別子境界で `par_` から始まる語（`par_iter`／
/// `par_chunks_mut`／将来の `par_windows`／`par_split` 等）が現れるかを
/// 判定する（`contains_rayon_marker` 専用。`find_par_marker_positions`
/// と異なり `.` 前置・`(` 後続を要求しない**生テキスト**走査であり、
/// コメント中の `` `par_chunks_mut(...)` `` のような記述も拾う。旧実装
/// の `content.contains("par_iter") || content.contains("par_chunks")`
/// の単純部分文字列判定を破壊的に狭めないため、境界条件を緩めた形で
/// 汎用化した）。
fn contains_generic_par_marker(text: &str) -> bool {
    text.match_indices("par_").any(|(i, _)| {
        i == 0 || {
            let prev = text.as_bytes()[i - 1];
            !is_ident_char(prev as char)
        }
    })
}

/// rayon 並列イテレータ・`rayon::` 直接参照の有無を判定する（ファイル
/// allowlist 突合専用。走査は生ソース `content` に対して行う）。
/// `par_` で始まる識別子全般（`contains_generic_par_marker`）を拾う
/// ことで、個別列挙では追従が漏れる新規の `par_windows`／`par_split`
/// 等の rayon 並列イテレータ命名にも fail-closed に対応する
/// （`docs/autodiff-determinism-mode-design.md` §3.3。codex-review
/// 指摘・PR #2274）。旧実装が拾っていた `into_par_iter`（`par_` 接頭
/// 辞パターンに当たらない）・`par_bridge`・`rayon::` は個別に維持し、
/// allowlist（24 件。§2.1）を後退させない。
fn contains_rayon_marker(content: &str) -> bool {
    content.contains("into_par_iter")
        || content.contains("par_bridge")
        || content.contains("rayon::")
        || contains_generic_par_marker(content)
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

// =====================================================================
// 字句前処理: コメント・文字列・char/byte literal の除去
// =====================================================================

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `s` が `word` から始まり、かつ `word` の直後が識別子文字でない
/// （キーワード境界が保たれている）ことを判定する。
fn starts_with_word(s: &str, word: &str) -> bool {
    if !s.starts_with(word) {
        return false;
    }
    match s[word.len()..].chars().next() {
        Some(c) => !is_ident_char(c),
        None => true,
    }
}

/// `chars[start]` が raw string／raw byte string（`r"..."`／
/// `r#"..."#`／`br"..."`／`br#"..."#`。`#` の個数は開始と終了で一致
/// させる）の先頭である場合に消費文字数を返す。`start` の直前が識別子
/// 文字の場合は呼び出し側で呼ばないこと（`for`／`bar` 等の識別子途中の
/// `r` を接頭辞と誤認しないため）。
fn try_consume_raw_string(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    let mut i = start;
    if chars.get(i) == Some(&'b') {
        i += 1;
    }
    if chars.get(i) != Some(&'r') {
        return None;
    }
    i += 1;
    let mut hashes = 0usize;
    while chars.get(i) == Some(&'#') {
        hashes += 1;
        i += 1;
    }
    if chars.get(i) != Some(&'"') {
        return None;
    }
    i += 1;
    loop {
        if i >= n {
            // 未終端: 走査破綻を防ぐため残り全体を消費する（fail-closed
            // というより走査継続のための保守的措置。整形済みソースでは
            // 発生しない）。
            return Some(n - start);
        }
        if chars[i] == '"' {
            let mut j = i + 1;
            let mut matched = 0usize;
            while matched < hashes && chars.get(j) == Some(&'#') {
                j += 1;
                matched += 1;
            }
            if matched == hashes {
                return Some(j - start);
            }
        }
        i += 1;
    }
}

/// `chars[start]`（`"`）から始まる非 raw 文字列リテラルを消費し、
/// 消費文字数を返す。エスケープ（`\"` 等）を考慮する。
fn consume_quoted_string(chars: &[char], start: usize) -> usize {
    let n = chars.len();
    let mut i = start + 1;
    let mut escaped = false;
    while i < n {
        let c = chars[i];
        if escaped {
            escaped = false;
            i += 1;
            continue;
        }
        if c == '\\' {
            escaped = true;
            i += 1;
            continue;
        }
        if c == '"' {
            i += 1;
            break;
        }
        i += 1;
    }
    i - start
}

/// `chars[start]`（`'`）が char literal／byte literal の開始であれば
/// 消費文字数を返す。ライフタイム（`'a`／`'static`）との区別は
/// 「エスケープ列または 1 文字の直後に閉じ `'` が続くか」で判定する
/// （直後に閉じ `'` が来なければライフタイムとみなし `None` を返す。
/// 呼び出し側は `'` 自体を通常の文字として出力する）。
fn try_consume_char_literal(chars: &[char], start: usize) -> Option<usize> {
    let n = chars.len();
    if chars.get(start) != Some(&'\'') {
        return None;
    }
    let mut i = start + 1;
    if i >= n {
        return None;
    }
    if chars[i] == '\\' {
        i += 1;
        if i >= n {
            return None;
        }
        match chars[i] {
            'u' => {
                i += 1;
                if chars.get(i) != Some(&'{') {
                    return None;
                }
                i += 1;
                while i < n && chars[i] != '}' {
                    i += 1;
                }
                if i >= n {
                    return None;
                }
                i += 1;
            }
            'x' => {
                i += 1;
                let mut consumed = 0;
                while consumed < 2 && i < n && chars[i].is_ascii_hexdigit() {
                    i += 1;
                    consumed += 1;
                }
            }
            _ => {
                i += 1;
            }
        }
    } else {
        i += 1;
    }
    if chars.get(i) == Some(&'\'') {
        Some(i + 1 - start)
    } else {
        None
    }
}

/// コメント（`//`・ネスト対応 `/* */`）・文字列リテラル（`"..."`・raw
/// string `r"..."`／`r#"..."#`・byte string `b"..."`／`br#"..."#`）・
/// char/byte literal（`'{'`／`';'`／`'\''`／`'\u{7b}'` 等）を空白へ
/// 置換する。ライフタイム（`'a`／`'static`）は char literal と誤認せず
/// そのまま残す（詳細はモジュール冒頭の「文境界・汚染追跡の方式」・
/// `try_consume_char_literal`）。厳密なトークナイザではないが、本
/// テストの目的（コード上の rayon 並列イテレータと縮約マーカーの
/// 共起検出）には十分な近似であり、`count_atomic_rmw_occurrences`
/// とも共用する。
fn strip_comments_and_strings(content: &str) -> String {
    let chars: Vec<char> = content.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(content.len());
    let mut i = 0usize;

    while i < n {
        let c = chars[i];

        if c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            if i < n {
                out.push('\n');
                i += 1;
            }
            continue;
        }

        if c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            let mut depth = 1i32;
            while i < n && depth > 0 {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                    continue;
                }
                if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    continue;
                }
                i += 1;
            }
            continue;
        }

        let prev_is_ident = i > 0 && is_ident_char(chars[i - 1]);

        if !prev_is_ident && let Some(consumed) = try_consume_raw_string(&chars, i) {
            i += consumed;
            out.push(' ');
            continue;
        }

        if !prev_is_ident && c == 'b' && chars.get(i + 1) == Some(&'"') {
            let consumed = consume_quoted_string(&chars, i + 1);
            i += 1 + consumed;
            out.push(' ');
            continue;
        }

        if c == '"' {
            let consumed = consume_quoted_string(&chars, i);
            i += consumed;
            out.push(' ');
            continue;
        }

        if !prev_is_ident
            && c == 'b'
            && chars.get(i + 1) == Some(&'\'')
            && let Some(consumed) = try_consume_char_literal(&chars, i + 1)
        {
            i += 1 + consumed;
            out.push(' ');
            continue;
        }

        // 閉じ `'` が続かない場合（ライフタイム）は `try_consume_char_literal`
        // が `None` を返し、通常の文字としてそのまま出力する（下の
        // `out.push(c)` へフォールスルー）。
        if c == '\''
            && let Some(consumed) = try_consume_char_literal(&chars, i)
        {
            i += consumed;
            out.push(' ');
            continue;
        }

        out.push(c);
        i += 1;
    }
    out
}

// =====================================================================
// ブロック深さを考慮した文境界・括弧深さ
// =====================================================================

/// `text` 中の各バイト位置（char 境界）における「その文字を処理する
/// 直前」の `()`／`[]`／`{}` 合算深さを返す（`text.len()` の位置には
/// 走査終了時点の深さを入れる）。並列マーカー・縮約マーカーの出現
/// 位置の括弧深さ比較（`has_depth_aware_par_reduce`・
/// `has_tainted_reduction_usage`）に使う。
fn compute_depths(text: &str) -> Vec<i32> {
    let mut depths = vec![0i32; text.len() + 1];
    let mut depth = 0i32;
    for (i, c) in text.char_indices() {
        depths[i] = depth;
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = (depth - 1).max(0),
            _ => {}
        }
    }
    depths[text.len()] = depth;
    depths
}

/// `text` 中の深さ 0（`()`／`[]`／`{}` 合算）で開く `{...}` ブロックの
/// `(開き `{` の byte 位置, 閉じ `}` の byte 位置)` を列挙する。
/// `count_in_scope`（再帰下降）と `statement_own_view`
/// （同一文中の共起判定用の塗りつぶし）の双方が使う。
fn find_depth0_braces(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut depth = 0i32;
    let mut open_stack: Vec<usize> = Vec::new();
    for (i, c) in text.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = (depth - 1).max(0),
            '{' => {
                if depth == 0 {
                    open_stack.push(i);
                }
                depth += 1;
            }
            '}' => {
                depth = (depth - 1).max(0);
                if depth == 0
                    && let Some(open_idx) = open_stack.pop()
                {
                    spans.push((open_idx, i));
                }
            }
            _ => {}
        }
    }
    spans
}

/// `text` を深さ 0 の文境界（`;` および式の継続でない深さ 0 の `}`）
/// で分割し、各文の `(start, end)` バイト範囲を返す（空白のみの文は
/// 除外）。ブロック内部の再帰的な文分割・レキシカルスコープ管理は
/// `count_in_scope` が担う（本関数は 1 段のみ）。
fn statement_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut depth = 0i32;
    let mut stmt_start = 0usize;

    for (i, c) in text.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' => depth = (depth - 1).max(0),
            '}' => {
                depth = (depth - 1).max(0);
                if depth == 0 {
                    let end = i + c.len_utf8();
                    let rest = text[end..].trim_start();
                    let continues = rest.chars().next().is_some_and(|c2| {
                        matches!(
                            c2,
                            '.' | '?'
                                | '+'
                                | '-'
                                | '*'
                                | '/'
                                | '%'
                                | '&'
                                | '|'
                                | '^'
                                | '<'
                                | '>'
                                | '='
                                | '!'
                                | '['
                                | '('
                        )
                    }) || starts_with_word(rest, "else")
                        || starts_with_word(rest, "catch");
                    if !continues {
                        spans.push((stmt_start, end));
                        stmt_start = end;
                    }
                }
            }
            ';' if depth == 0 => {
                spans.push((stmt_start, i));
                stmt_start = i + 1;
            }
            _ => {}
        }
    }
    if stmt_start < text.len() {
        spans.push((stmt_start, text.len()));
    }

    spans
        .into_iter()
        .filter(|(s, e)| !text[*s..*e].trim().is_empty())
        .collect()
}

/// `raw`（1 文）が内包する深さ 0 の `{...}` ブロック本体を空白へ塗り
/// つぶしたテキストを返す。同一文中の並列マーカー／縮約マーカー共起
/// 判定（`has_depth_aware_par_reduce`・`has_tainted_reduction_usage`）
/// はこのテキストに対して行い、ネストしたブロックの中身は
/// `count_in_scope` が別途、独立したレキシカルスコープとして再帰検査
/// するため二重計上しない。`let` 初期化式の汚染判定（`classify_taint`）
/// はこれとは別に塗りつぶし前の `raw` をそのまま使う。
fn statement_own_view(raw: &str) -> String {
    let spans = find_depth0_braces(raw);
    let mut blanked = vec![false; raw.len()];
    for (open_idx, close_idx) in &spans {
        for flag in blanked.iter_mut().take(*close_idx).skip(*open_idx + 1) {
            *flag = true;
        }
    }
    raw.char_indices()
        .map(|(i, c)| if blanked[i] { ' ' } else { c })
        .collect()
}

// =====================================================================
// 並列マーカー・縮約マーカーの検出
// =====================================================================

/// `.par_` で始まり `(` へ続くメソッド呼び出し全般（`.par_iter(`／
/// `.par_chunks(`／`.par_chunks_mut(`／`.par_iter_mut(`／`.par_bridge(`／
/// `.par_windows(`／`.par_split(` 等。新規命名を個別列挙せずに拾う
/// ための汎用規則）・`.into_par_iter(`（`par_` 接頭辞パターンに当た
/// らない rayon 変換経路）に加え、`::par_` で始まるパス参照
/// （`T::par_iter`。メソッド値として変数へ束縛して後から呼び出す形
/// 〈`let f = T::par_iter; let it = f(&data); it.sum()`〉のため、末尾
/// `(` の有無を問わず識別子境界のみで拾う。敵対的レビュー指摘）の
/// 出現バイト位置を返す。
fn find_par_marker_positions(text: &str) -> Vec<usize> {
    let mut positions = Vec::new();
    for (i, _) in text.match_indices(".par_") {
        let rest = &text[i + 5..];
        let ident_len = rest.find(|c: char| !is_ident_char(c)).unwrap_or(rest.len());
        if ident_len > 0 && rest[ident_len..].starts_with('(') {
            positions.push(i);
        }
    }
    for (i, _) in text.match_indices("::par_") {
        let rest = &text[i + 6..];
        let ident_len = rest.find(|c: char| !is_ident_char(c)).unwrap_or(rest.len());
        if ident_len > 0 {
            positions.push(i);
        }
    }
    for (i, _) in text.match_indices(".into_par_iter(") {
        positions.push(i);
    }
    positions.sort_unstable();
    positions.dedup();
    positions
}

/// rayon の分割依存縮約マーカー。`.sum(`／`.sum::<`（turbofish）に加え
/// `.product(`／`.fold(`／`.fold_with(`／`.fold_chunks(`／`.try_fold(`／
/// `.try_reduce(`／`.try_reduce_with(`・`.reduce(`／`reduce_with(` を
/// 含む（rayon の `fold` 系はスレッド分割依存の部分結果を返すため
/// `sum`／`reduce` と同類型）。
const REDUCE_MARKERS: &[&str] = &[
    ".sum(",
    ".sum::<",
    ".product(",
    ".product::<",
    ".reduce(",
    "reduce_with(",
    ".fold(",
    ".fold_with(",
    ".fold_chunks(",
    ".try_fold(",
    ".try_reduce(",
    ".try_reduce_with(",
];

/// `ParallelIterator::collect` 相当のマーカー。`data.par_chunks(..)
/// .map(..).collect::<Vec<_>>().into_iter().fold(..)`
/// （`crates/backend-cpu/src/mse.rs::mse_sum_sq_f32` 等、本クレートの
/// 縮約実装で広く使われる実測イディオム。§E 実測）は `.collect(` で
/// 一旦インデックス順の `Vec` へ確定し、`.into_iter()` 以降は通常の
/// 逐次 `Iterator` になるため、`.collect(` より後方の `.fold(` 等は
/// 直前の並列マーカー・汚染識別子とは**接続されない**。
/// `has_reduce_after` がこの遮断を判定する。遮断は順序保持が型で
/// 明示される `Vec` への collect（`.collect::<Vec<`・
/// `.collect_into_vec(`）に限る。型推論に任せた `.collect()` や
/// `HashMap`／`HashSet` 等への collect は反復順序が決まらない場合が
/// あるため遮断せず、後続の縮約を並列縮約として数える（fail-closed 側）。
const COLLECT_MARKERS: &[&str] = &[".collect::<Vec<", ".collect_into_vec("];

/// `text`（`compute_depths` の対象と同一）中の `markers` 各出現に
/// ついて `(バイト位置, その位置の括弧深さ)` を返す。
fn find_marker_depths(text: &str, depths: &[i32], markers: &[&str]) -> Vec<(usize, i32)> {
    let mut hits = Vec::new();
    for marker in markers {
        for (i, _) in text.match_indices(marker) {
            hits.push((i, depths.get(i).copied().unwrap_or(0)));
        }
    }
    hits
}

/// `origin_idx`（深さ `origin_depth`。並列マーカーまたは汚染識別子の
/// 出現位置）より後方かつ**同じか浅い**括弧深さに縮約マーカーが現れる
/// かを判定する。ただし `origin_idx` と縮約マーカーの間に、同じか浅い
/// 深さの `.collect(` が挟まる場合はイテレータ連鎖が断ち切られている
/// とみなし接続しない（`COLLECT_MARKERS` doc 参照）。
fn has_reduce_after(
    origin_idx: usize,
    origin_depth: i32,
    reduce_hits: &[(usize, i32)],
    collect_hits: &[(usize, i32)],
) -> bool {
    reduce_hits.iter().any(|&(r_idx, r_depth)| {
        r_idx > origin_idx
            && r_depth <= origin_depth
            && !collect_hits.iter().any(|&(c_idx, c_depth)| {
                c_idx > origin_idx && c_idx < r_idx && c_depth <= origin_depth
            })
    })
}

/// 1 文の中で、並列マーカーより後方かつ**同じか浅い**括弧深さに
/// 縮約マーカーが現れるかを判定する（§C: `data.par_iter().map(..)
/// .sum::<f32>()` は検出、`data.par_chunks(n).map(|c| c.iter()
/// .sum::<f32>()).collect(..)`〈チャンク内逐次和〉・`data.par_chunks(..)
/// .map(..).collect::<Vec<_>>().into_iter().fold(..)`〈collect で
/// 連鎖が切れる〉は非検出）。`view` は `statement_own_view` で塗り
/// つぶし済みのテキストを渡す。
fn has_depth_aware_par_reduce(view: &str) -> bool {
    let par_positions = find_par_marker_positions(view);
    if par_positions.is_empty() {
        return false;
    }
    let depths = compute_depths(view);
    let reduce_hits = find_marker_depths(view, &depths, REDUCE_MARKERS);
    if reduce_hits.is_empty() {
        return false;
    }
    let collect_hits = find_marker_depths(view, &depths, COLLECT_MARKERS);
    par_positions.iter().any(|&p_idx| {
        let p_depth = depths.get(p_idx).copied().unwrap_or(0);
        has_reduce_after(p_idx, p_depth, &reduce_hits, &collect_hits)
    })
}

/// `text` 中で識別子 `ident` が識別子境界（前後が識別子文字でない）で
/// 一致する出現バイト位置を返す。
fn find_ident_occurrences(text: &str, ident: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    for (i, _) in text.match_indices(ident) {
        let before_ok = i == 0 || !is_ident_char(bytes[i - 1] as char);
        let after = i + ident.len();
        let after_ok = after >= bytes.len() || !is_ident_char(bytes[after] as char);
        if before_ok && after_ok {
            out.push(i);
        }
    }
    out
}

/// 過去の文で汚染された識別子（`tainted`）が、この文中で識別子境界
/// 一致し、かつ後方の同じか浅い括弧深さに縮約マーカーが現れるかを
/// 判定する（文をまたぐ並列縮約。`view` は `statement_own_view`）。
/// `.collect(` による連鎖の遮断は `has_reduce_after` と同じ規則。
fn has_tainted_reduction_usage(view: &str, tainted: &HashSet<String>) -> bool {
    if tainted.is_empty() {
        return false;
    }
    let depths = compute_depths(view);
    let reduce_hits = find_marker_depths(view, &depths, REDUCE_MARKERS);
    if reduce_hits.is_empty() {
        return false;
    }
    let collect_hits = find_marker_depths(view, &depths, COLLECT_MARKERS);
    tainted.iter().any(|ident| {
        find_ident_occurrences(view, ident).iter().any(|&i_idx| {
            let i_depth = depths.get(i_idx).copied().unwrap_or(0);
            has_reduce_after(i_idx, i_depth, &reduce_hits, &collect_hits)
        })
    })
}

// =====================================================================
// UFCS（fully-qualified）縮約呼び出しの検出（敵対的レビュー指摘 REQ 2）
// =====================================================================

/// `REDUCE_MARKERS` と同じメソッド名を `::` 前置の関数呼び出し構文
/// （`ParallelIterator::sum(...)`・`Trait::reduce(...)` 等。UFCS）で
/// 検出するマーカー。`.sum(` ではなく `::sum(` のようにレシーバを
/// メソッド構文の外（第 1 引数）に取る呼び出しは `.method(` 走査では
/// 拾えないため、別途走査する。
const UFCS_REDUCE_MARKERS: &[&str] = &[
    "::sum(",
    "::sum::<",
    "::product(",
    "::product::<",
    "::reduce(",
    "::reduce_with(",
    "::fold(",
    "::fold_with(",
    "::fold_chunks(",
    "::try_fold(",
    "::try_reduce(",
    "::try_reduce_with(",
];

/// `text[open_idx]` が `(` であることを前提に、対応する `)` のバイト
/// 位置を深さ追跡（ネスト対応）で返す。
fn find_matching_paren(text: &str, open_idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open_idx) != Some(&b'(') {
        return None;
    }
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(open_idx) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// `after_marker` が turbofish マーカー（`::method::<`）の直後（型引数
/// リストの内側）を指すとき、型引数を深さ追跡で読み飛ばした直後に
/// 続く呼び出しの開き `(` のバイト位置を返す（`Trait::method::<T>(..)`
/// 形）。
fn find_turbofish_call_paren(text: &str, after_marker: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 1i32;
    let mut i = after_marker;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    i += 1;
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    while i < bytes.len() && (bytes[i] as char).is_whitespace() {
        i += 1;
    }
    if bytes.get(i) == Some(&b'(') {
        Some(i)
    } else {
        None
    }
}

/// `text` 中の UFCS 縮約呼び出し（`UFCS_REDUCE_MARKERS`）を検出し、
/// それぞれの呼び出し引数リストの `(内容の開始バイト位置, 終了バイト
/// 位置)` を列挙する。
fn find_ufcs_reduce_call_bodies(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for marker in UFCS_REDUCE_MARKERS {
        for (i, _) in text.match_indices(marker) {
            let marker_end = i + marker.len();
            let open_idx = if marker.ends_with('(') {
                marker_end - 1
            } else {
                match find_turbofish_call_paren(text, marker_end) {
                    Some(idx) => idx,
                    None => continue,
                }
            };
            if let Some(close_idx) = find_matching_paren(text, open_idx) {
                out.push((open_idx + 1, close_idx));
            }
        }
    }
    out
}

/// UFCS 形の縮約呼び出し（`ParallelIterator::sum(data.par_iter()...)`
/// 等）を検出し、その引数リスト内に並列マーカー（Vec への collect で
/// 連鎖が切れていないもの）または汚染識別子（識別子境界一致）があれば
/// 真を返す（REQ 2。`view` は `statement_own_view` 済みテキスト）。
fn has_ufcs_reduction(view: &str, tainted: &HashSet<String>) -> bool {
    for (arg_start, arg_end) in find_ufcs_reduce_call_bodies(view) {
        let args = &view[arg_start..arg_end];
        let par_positions = find_par_marker_positions(args);
        if !par_positions.is_empty() {
            let depths = compute_depths(args);
            let collect_hits = find_marker_depths(args, &depths, COLLECT_MARKERS);
            let has_uncollected_par = par_positions.iter().any(|&p_idx| {
                let p_depth = depths.get(p_idx).copied().unwrap_or(0);
                !collect_hits
                    .iter()
                    .any(|&(c_idx, c_depth)| c_idx > p_idx && c_depth <= p_depth)
            });
            if has_uncollected_par {
                return true;
            }
        }
        if tainted
            .iter()
            .any(|t| !find_ident_occurrences(args, t).is_empty())
        {
            return true;
        }
    }
    false
}

// =====================================================================
// let／単純代入からの汚染識別子抽出
// =====================================================================

/// パターン・型注釈テキスト中の深さ 0 の区切り文字（`:`（`::` を除く）
/// または代入の `=`（`==`／`!=`／`<=`／`>=`／`=>` を除く））を探す。
/// 見つかった場合 `(バイト位置, 区切りが `:` か)` を返す。
fn find_pattern_delimiter(s: &str) -> Option<(usize, bool)> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = (depth - 1).max(0),
            b':' if depth == 0 => {
                if bytes.get(i + 1) == Some(&b':') {
                    i += 2;
                    continue;
                }
                return Some((i, true));
            }
            b'=' if depth == 0 => {
                let next_eq = bytes.get(i + 1) == Some(&b'=');
                let prev_cmp = i > 0 && matches!(bytes[i - 1], b'!' | b'<' | b'>' | b'=');
                if !next_eq && !prev_cmp {
                    return Some((i, false));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// 型注釈テキスト中で、代入の `=`（比較・アロー演算子を除く）を深さ 0
/// （`()`／`[]`／`{}`／`<>` 合算。型パス・ジェネリクスの `<...>` 内の
/// `=`〈associated type binding 等〉を除外するための簡易近似）で探す。
fn find_top_level_eq(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b')' | b']' | b'}' | b'>' => depth = (depth - 1).max(0),
            b'=' if depth == 0 => {
                let next_eq = bytes.get(i + 1) == Some(&b'=');
                let prev_cmp = i > 0 && matches!(bytes[i - 1], b'!' | b'<' | b'>' | b'=');
                if !next_eq && !prev_cmp {
                    return Some(i);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// パターンテキストから束縛識別子をすべて抽出する（`mut`／`ref`／
/// `box`／`_` は除く）。タプルパターン `(a, mut b)` にも対応する。
fn extract_pattern_idents(pattern: &str) -> Vec<String> {
    let mut idents = Vec::new();
    let mut cur = String::new();
    for c in pattern.chars().chain(std::iter::once(' ')) {
        if is_ident_char(c) {
            cur.push(c);
        } else if !cur.is_empty() {
            if !matches!(cur.as_str(), "mut" | "ref" | "box" | "_") {
                idents.push(cur.clone());
            }
            cur.clear();
        }
    }
    idents
}

/// 文中の任意位置の `let` キーワードを探し、`(パターンのバイト範囲,
/// 初期化式の開始バイト位置)` を返す（型注釈の有無いずれにも対応。
/// 初期化式が存在しない `let ident: T;` 相当は `None`）。旧実装の
/// 「ブロック境界文字列を先頭から剥がす」処理は、`count_in_scope`
/// が既にブロック単位で文を再帰分割しているため不要になった。
///
/// **`view`（`statement_own_view` で深さ 0 のネストしたブロック本体を
/// 塗りつぶし済みのテキスト）に対して探索する**ことが重要な契約:
/// 生の `raw` で探索すると、`fn f() { let it = data.par_iter(); ... }`
/// のような「関数本体全体が 1 文」になるケースで、関数本体の奥深くに
/// ある無関係な `let`（実際は再帰分割された別の子文として個別に検査
/// 済み）まで拾ってしまい、その初期化式が「関数本体の残り全部」に
/// 広がってしまう（`crates/backend-cpu/src/fused_elementwise.rs`
/// `run_fused_elementwise` で実測した誤検出。§E 実測）。`view` では
/// ネストしたブロック本体が空白化されているため、そのブロック内部に
/// ある `let` は見つからず、この文自身が持つ最上位の `let`（もしあれ
/// ば）のみが見つかる。戻り値のバイト位置は `raw` と `view` の間で
/// 1 対 1 対応する（塗りつぶしは文字を空白へ置換するのみで長さを
/// 変えない）ため、呼び出し側はこの位置で `raw`（塗りつぶし前）を
/// 直接スライスして初期化式全体（ネストしたブロック内の並列マーカー
/// を含む）を取得できる。
fn parse_let_binding_range(view: &str) -> Option<(std::ops::Range<usize>, usize)> {
    let mut search_from = 0usize;
    while let Some(rel) = view[search_from..].find("let") {
        let idx = search_from + rel;
        let before_ok = idx == 0 || !is_ident_char(view.as_bytes()[idx - 1] as char);
        let after_idx = idx + 3;
        let after_ok = view[after_idx..]
            .chars()
            .next()
            .is_some_and(|c| c.is_whitespace());
        if before_ok && after_ok {
            let pattern_start = after_idx;
            let (delim_idx, is_colon) = find_pattern_delimiter(&view[pattern_start..])?;
            let abs_delim = pattern_start + delim_idx;
            if is_colon {
                let eq_rel = find_top_level_eq(&view[abs_delim + 1..])?;
                let eq_idx = abs_delim + 1 + eq_rel;
                return Some((pattern_start..abs_delim, eq_idx + 1));
            }
            return Some((pattern_start..abs_delim, abs_delim + 1));
        }
        search_from = idx + 3;
    }
    None
}

/// `let` を伴わない単純代入 `IDENT = <式>;`（複合代入・比較演算子は
/// 除く）から `(識別子のバイト範囲, 初期化式の開始バイト位置)` を
/// 抽出する。`parse_let_binding_range` と同じ理由で `view` に対して
/// 探索する（先頭が `{`／`}` の文はここで自然に除外される）。
fn parse_plain_assignment_range(view: &str) -> Option<(std::ops::Range<usize>, usize)> {
    let leading_ws = view.len() - view.trim_start().len();
    let s = &view[leading_ws..];
    let ident_len = s.find(|c: char| !is_ident_char(c))?;
    if ident_len == 0 {
        return None;
    }
    let ident = &s[..ident_len];
    if matches!(
        ident,
        "let" | "if" | "for" | "while" | "return" | "match" | "fn" | "else" | "loop"
    ) {
        return None;
    }
    let after_ident = leading_ws + ident_len;
    let rest = &view[after_ident..];
    let ws_after_ident = rest.len() - rest.trim_start().len();
    let eq_idx = after_ident + ws_after_ident;
    let bytes = view.as_bytes();
    if bytes.get(eq_idx) != Some(&b'=') || bytes.get(eq_idx + 1) == Some(&b'=') {
        return None;
    }
    Some((leading_ws..after_ident, eq_idx + 1))
}

/// 初期化式 `init`（塗りつぶし前のテキスト）が汚染源を含むかを判定
/// する（敵対的レビュー指摘 REQ 3: `let parts = data.par_iter()
/// .map(f).collect::<Vec<f32>>();` のような Vec への collect で終わる
/// 初期化式を誤って汚染していた過検出の是正）。並列マーカー
/// （`find_par_marker_positions`。`::par_` パス参照を含む）または
/// 既存の汚染識別子が、その後ろに同じか浅い深さの `COLLECT_MARKERS`
/// （順序保持が型で明示される Vec への collect）を伴わずに現れる
/// 場合のみ汚染源とみなす。`let v = x.par_iter().collect::<Vec<_>>()
/// .par_iter();` のように collect の後に再び並列マーカーが現れる
/// 場合は、その後方のマーカー自身が「後続に collect を伴わない」ため
/// 汚染する。
fn initializer_taints(init: &str, tainted: &HashSet<String>) -> bool {
    let depths = compute_depths(init);
    let collect_hits = find_marker_depths(init, &depths, COLLECT_MARKERS);
    let not_broken_by_collect = |idx: usize, depth: i32| {
        !collect_hits
            .iter()
            .any(|&(c_idx, c_depth)| c_idx > idx && c_depth <= depth)
    };

    let par_taints = find_par_marker_positions(init).into_iter().any(|p_idx| {
        let p_depth = depths.get(p_idx).copied().unwrap_or(0);
        not_broken_by_collect(p_idx, p_depth)
    });
    if par_taints {
        return true;
    }

    tainted.iter().any(|ident| {
        find_ident_occurrences(init, ident)
            .into_iter()
            .any(|i_idx| {
                let i_depth = depths.get(i_idx).copied().unwrap_or(0);
                not_broken_by_collect(i_idx, i_depth)
            })
    })
}

/// [`classify_taint`] の結果。`let` による束縛（このレキシカルスコープ
/// に閉じ、親スコープへは漏らさない）と、`let` を伴わない単純代入
/// （親スコープへ伝播しうる）を区別する（敵対的レビュー指摘 REQ 1）。
enum TaintKind {
    None,
    Let(Vec<String>),
    Assignment(Vec<String>),
}

/// 1 文（`raw`）を解析し、新たに汚染される識別子を [`TaintKind`] として
/// 返す。パターン探索は `statement_own_view`（深さ 0 のネストした
/// ブロック本体を塗りつぶし済み）に対して行うが、初期化式が汚染源を
/// 含むかの判定（`initializer_taints`）は塗りつぶし前の `raw` を使う
/// （`let it = { let x = 1; data.par_iter() };` のような場合に
/// `.par_iter(` を見失わないため）。
fn classify_taint(raw: &str, tainted: &HashSet<String>) -> TaintKind {
    let view = statement_own_view(raw);
    if let Some((pattern_range, init_start)) = parse_let_binding_range(&view) {
        let init = &raw[init_start..];
        return if initializer_taints(init, tainted) {
            TaintKind::Let(extract_pattern_idents(&view[pattern_range]))
        } else {
            TaintKind::None
        };
    }
    if let Some((pattern_range, init_start)) = parse_plain_assignment_range(&view) {
        let init = &raw[init_start..];
        return if initializer_taints(init, tainted) {
            TaintKind::Assignment(vec![view[pattern_range].to_string()])
        } else {
            TaintKind::None
        };
    }
    TaintKind::None
}

/// [`classify_taint`] の `Let`／`Assignment` 種別を区別しない簡易版
/// （単体テスト・後方互換用）。スコープ境界を持つ本走査本体
/// （`count_in_scope`）は種別を区別する `classify_taint` を直接使う。
fn extract_taint_targets(raw: &str, tainted: &HashSet<String>) -> Vec<String> {
    match classify_taint(raw, tainted) {
        TaintKind::Let(idents) | TaintKind::Assignment(idents) => idents,
        TaintKind::None => Vec::new(),
    }
}

/// `raw` の先頭文が関数アイテム（`fn`／`pub fn`／属性・`async`／
/// `unsafe`／`extern "C"` 修飾付きを含む）であるかを判定する
/// （`count_in_scope` のレキシカルスコープ境界検出専用。敵対的レビュー
/// 指摘 REQ 1）。判定条件: 最初の深さ 0 `{` より前のテキストに、
/// 識別子境界で `fn` の直後に**空白を 1 文字以上挟んで**識別子が
/// 続く箇所があること。関数ポインタ型注釈（`fn(i32) -> i32`。`fn`
/// の直後が `(` で空白を挟まない）は対象外とする。旧実装は「`fn` と
/// いう語が出現するか」のみを見ており、ローカル `fn`・`fn(...)` 型
/// 注釈・ローカル `impl` を挟むだけで `tainted.clear()` が誤って走り、
/// 汚染集合が全消去される不具合があった（この不具合自体は
/// `count_in_scope` がレキシカルスコープで汚染集合を管理する設計へ
/// 置き換えたことで構造的に解消しているが、`fn(` 型注釈と実際の
/// 関数アイテムを区別する判定精度自体は、`let x: fn() -> i32 = ||
/// { data.par_iter() };` のようにクロージャ本体を関数アイテムの本体
/// と誤認しない（クロージャは外側変数をキャプチャするため独立スコープ
/// にしてはならない）ために必要）。
fn looks_like_fn_item(raw: &str) -> bool {
    let prefix = match find_depth0_braces(raw).first() {
        Some((open_idx, _)) => &raw[..*open_idx],
        None => raw,
    };
    find_ident_occurrences(prefix, "fn").into_iter().any(|idx| {
        let after = &prefix[idx + 2..];
        let trimmed = after.trim_start();
        after.len() != trimmed.len()
            && trimmed
                .chars()
                .next()
                .is_some_and(|c| c.is_alphabetic() || c == '_')
    })
}

// =====================================================================
// 並列縮約の共起件数（レキシカルスコープごとの再帰処理）
// =====================================================================

/// `text`（あるレキシカルスコープの内容。ファイル全体・fn 本体・
/// if／for／match アーム・ブロック式・クロージャ本体など）を document
/// 順に処理し、`(このスコープ内で検出した並列縮約の共起件数, 呼び出し
/// 元へ伝播すべき汚染識別子集合)` を返す（敵対的レビュー指摘 REQ 1）。
///
/// 旧実装は文をフラットに集めて順に処理し、`looks_like_fn_item` で
/// 「関数アイテムらしき文」を検出するたびファイル全体の汚染集合を
/// まるごと `clear()` していた。このため関数の途中にローカル `fn`・
/// `fn(i32) -> i32` 型注釈・ローカル `impl` を 1 つ置くだけで、それ
/// 以降の文をまたぐ検出がすべて無効になる不具合があった。本関数は
/// 汚染集合の管理を**レキシカルスコープ単位の再帰**へ置き換える:
///
/// - fn アイテムの本体は `tainted_in` を無視し空集合から始める
///   （fn はローカル変数をキャプチャしない）。
/// - それ以外のネストしたブロック（if／for／loop／match アーム／
///   ブロック式／クロージャ本体／impl・mod 内の非 fn アイテム）は
///   現在の `tainted` を clone して引き継ぐ（双方向: 親の汚染状態を
///   引き継ぎ、`let` なし代入による新規汚染は親へ書き戻す）。
/// - ネストしたブロック内の `let` で新たに汚染された名前は、そのブ
///   ロックのスコープに閉じ、親へは漏らさない。`let` なし代入
///   `IDENT = …` による汚染だけを親スコープへ伝播する
///   （`TaintKind::Assignment`）。ただし fn アイテムの境界をまたぐ
///   場合は、代入による汚染であっても一切伝播しない（fn は独立した
///   呼び出しスコープであり、呼び出し元のローカル変数と偶然同名で
///   あっても無関係なため）。
fn count_in_scope(text: &str, tainted_in: &HashSet<String>) -> (usize, HashSet<String>) {
    let mut tainted = tainted_in.clone();
    let mut count = 0usize;
    let mut leaked_here: HashSet<String> = HashSet::new();

    for (s, e) in statement_spans(text) {
        let raw = &text[s..e];
        let view = statement_own_view(raw);

        // 同一文中の共起（メソッド呼び出し形・UFCS 形）と、過去の文
        // からの汚染識別子の使用は、いずれも「1 件」として数える
        // （相互排他的な `||` 集約のため二重計上しない）。
        if has_depth_aware_par_reduce(&view)
            || has_tainted_reduction_usage(&view, &tainted)
            || has_ufcs_reduction(&view, &tainted)
        {
            count += 1;
        }

        match classify_taint(raw, &tainted) {
            TaintKind::Let(idents) => {
                for ident in idents {
                    tainted.insert(ident);
                }
            }
            TaintKind::Assignment(idents) => {
                for ident in idents {
                    tainted.insert(ident.clone());
                    leaked_here.insert(ident);
                }
            }
            TaintKind::None => {}
        }

        let is_fn_item = looks_like_fn_item(raw);
        for (open_idx, close_idx) in find_depth0_braces(raw) {
            if open_idx >= close_idx {
                continue;
            }
            let inner = &raw[open_idx + 1..close_idx];
            let child_tainted = if is_fn_item {
                HashSet::new()
            } else {
                tainted.clone()
            };
            let (inner_count, inner_leaked) = count_in_scope(inner, &child_tainted);
            count += inner_count;
            if !is_fn_item {
                for ident in inner_leaked {
                    tainted.insert(ident.clone());
                    leaked_here.insert(ident);
                }
            }
        }
    }

    (count, leaked_here)
}

/// rayon 並列イテレータと縮約マーカー（`.sum()`／`.reduce(`／
/// `reduce_with`／`.product(`／`.fold(` 等・UFCS 形の `::sum(` 等）
/// が、ブロック深さ・括弧深さを考慮した文境界の上で共起する箇所を
/// 数える（分割依存の縮約の検出。詳細はモジュール冒頭コメント
/// 「文境界・汚染追跡の方式」）。ファイル全体を最外殻のレキシカル
/// スコープとして `count_in_scope` へ委譲する（汚染集合の初期値は
/// 空集合）。
fn count_par_reduce_cooccurrences(content: &str) -> usize {
    let cleaned = strip_comments_and_strings(content);
    count_in_scope(&cleaned, &HashSet::new()).0
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
        "rayon 並列イテレータと縮約マーカー（.sum()/.reduce(/reduce_with/\
         .product(/.fold( 等）がブロック深さを考慮した文境界の上で共起\
         する箇所が見つかった（分割依存の縮約はスレッド割り当てに結果が\
         依存しうるため決定性契約に違反する疑いがある。\
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
// 既知の限界（関数引数・戻り値経由の並列イテレータ）の fail-closed 検出
// =====================================================================

fn line_is_use_declaration(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("use ") || t.starts_with("pub use ") || t.starts_with("pub(crate) use ")
}

/// 戻り値型に `ParallelIterator`／`IndexedParallelIterator` を含む
/// 関数定義や、それらを引数型に取る関数定義（`impl ParallelIterator`
/// 等）が `crates/backend-cpu/src` に存在しないことを検査する。本
/// テストの走査（`count_par_reduce_cooccurrences` のトークン走査）は
/// 関数引数・戻り値経由で並列イテレータの値が流れる経路を追跡しない
/// （モジュール冒頭「既知の限界」）ため、その経路の導入自体をここで
/// fail-closed に検出する。`use` 文・コメント（既に
/// `strip_comments_and_strings` で除去済み）は対象外とする。
#[test]
fn no_parallel_iterator_valued_function_signature_in_backend_cpu_src() {
    let src_dir = backend_cpu_src_dir();
    let mut files = Vec::new();
    visit_rs_files(&src_dir, &mut files);

    let mut offending: Vec<String> = Vec::new();
    for (path, content) in &files {
        let cleaned = strip_comments_and_strings(content);
        for (line_no, line) in cleaned.lines().enumerate() {
            if line_is_use_declaration(line) {
                continue;
            }
            if line.contains("ParallelIterator") {
                let rel = path.strip_prefix(&src_dir).unwrap_or(path).display();
                offending.push(format!("{rel}:{}: {}", line_no + 1, line.trim()));
            }
        }
    }
    assert!(
        offending.is_empty(),
        "戻り値型・引数型に ParallelIterator／IndexedParallelIterator を\
         含む関数定義が見つかった（use 文・コメントを除く）。この経路は\
         count_par_reduce_cooccurrences のトークン走査が追跡しない既知の\
         限界（モジュール冒頭コメント）に該当するため、導入する場合は\
         決定性の根拠を棚卸しし、本テストの exemption を追加するまで\
         CI を落とす: {offending:?}"
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
// （codex-review 指摘・Cursor Bugbot 指摘・PR #2274:
// 6 ラウンドにわたる検出漏れの再発防止。旧 `assigned_identifier` 個別
// テストは新しいプリミティブ〈parse_let_binding_range・
// extract_taint_targets 等〉向けに書き換えている）
// =====================================================================

#[test]
fn count_par_reduce_detects_same_statement_cooccurrence() {
    let src = "let s: f32 = data.par_iter().sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_variable_assignment_across_statements() {
    let src = "let it = data.par_iter(); let s: f32 = it.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_typed_let_assignment_across_statements() {
    let src = "let it: Vec<f32> = data.par_iter().collect(); let s = it.iter().sum::<f32>();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_turbofish_sum() {
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
    let src = "let s: f32 = data.par_iter().sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_first_statement_in_function_body() {
    // 関数本体の最初の文（ブロック開始 `{` 直後の `let`）に対する検出
    // 漏れの再発防止（codex-review 指摘・PR #2274 review r4105558135）。
    let src = "fn f(data: &[f32]) -> f32 {\n    let it = data.par_iter();\n    it.sum()\n}";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_reduction_after_if_block_then_let() {
    // 未閉じ `{` の剥がしが残余の `}` を無視するため、先行する if
    // ブロックの後の文で `let` が検出されない不具合の再発防止
    // （Cursor Bugbot 指摘・PR #2274）。
    let src = "fn f() { if c { foo(); } let it = data.par_iter(); let s = it.sum(); }";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_reduction_after_for_block_then_let() {
    let src = "for i in 0..n { g(i); } let it = data.par_iter(); it.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_reduction_after_block_expression_initializer() {
    // 初期化ブロック内の `;` で分割され `let it` を見失う不具合の
    // 再発防止（codex P1 指摘・PR #2274）。
    let src = "let it = { let x = 1; data.par_iter() }; it.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_par_bridge_same_statement() {
    let src = "let s: f32 = data.iter().par_bridge().sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_par_bridge_across_statements() {
    let src = "let it = data.iter().par_bridge(); let s: f32 = it.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn contains_rayon_marker_detects_par_bridge() {
    assert!(contains_rayon_marker("data.iter().par_bridge().sum()"));
}

#[test]
fn count_par_reduce_detects_reduction_after_balanced_block_initializer() {
    let src = "let it = if x { a } else { b }.par_iter(); let s: f32 = it.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_does_not_double_count_same_statement_and_taint() {
    let src = "fn f(){ let s: f32 = data.par_iter().sum(); }";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_closure_taint() {
    let src = "let f = |d: &[f32]| d.par_iter(); let s: f32 = f(x).sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_taint_propagation() {
    let src = "let it = data.par_iter(); let it2 = it; it2.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_plain_assignment_taint() {
    let src = "let it; it = data.par_iter(); it.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_tuple_pattern_taint() {
    let src = "let (a, b) = (data.par_iter(), 0); a.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_additional_reduce_markers() {
    assert_eq!(
        count_par_reduce_cooccurrences("data.par_iter().product::<f32>()"),
        1
    );
    assert_eq!(
        count_par_reduce_cooccurrences(
            "data.par_iter().fold(|| 0.0, |a, b| a + b).collect::<Vec<_>>()"
        ),
        1
    );
    assert_eq!(
        count_par_reduce_cooccurrences("data.par_windows(2).map(|w| w[0]).sum::<f32>()"),
        1
    );
}

#[test]
fn count_par_reduce_ignores_lexical_semicolons_and_braces() {
    assert_eq!(
        count_par_reduce_cooccurrences("let c = '{'; let it = data.par_iter(); it.sum()"),
        1
    );
    assert_eq!(
        count_par_reduce_cooccurrences("let c = ';'; let it = data.par_iter(); it.sum()"),
        1
    );
    assert_eq!(
        count_par_reduce_cooccurrences(
            "let s = r#\"}; let x\"#; let it = data.par_iter(); it.sum()"
        ),
        1
    );
}

#[test]
fn count_par_reduce_ignores_lifetime_annotations() {
    let src = "fn g<'a>(d: &'a [f32]) -> f32 { let it = d.par_iter(); it.sum() }";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_chunked_sequential_sum_collect() {
    let src = "data.par_chunks(n).map(|c| c.iter().sum::<f32>()).collect::<Vec<_>>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_ignores_taint_without_reduction_usage() {
    let src = "let it = data.par_iter(); let s = unit.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

/// `parse_let_binding_range` は `view`（`statement_own_view` 済み
/// テキスト）に対して呼ぶ契約のため、テストでもネストしたブロックを
/// 持たない入力（`statement_own_view` が恒等写像になる入力）で検証する。
#[test]
fn parse_let_binding_range_extracts_pattern_and_initializer() {
    let src = "let x = 1";
    let (pattern, init_start) = parse_let_binding_range(src).unwrap();
    assert_eq!(&src[pattern], " x ");
    assert_eq!(&src[init_start..], " 1");

    let src = "let mut y = data.par_iter()";
    let (pattern, init_start) = parse_let_binding_range(src).unwrap();
    assert_eq!(&src[pattern], " mut y ");
    assert_eq!(&src[init_start..], " data.par_iter()");

    assert!(parse_let_binding_range("x == 1").is_none());
    assert!(parse_let_binding_range("data.iter()").is_none());
}

#[test]
fn parse_let_binding_range_handles_typed_and_tuple_patterns() {
    let src = "let it: Vec<f32> = data.par_iter().collect()";
    let (pattern, _) = parse_let_binding_range(src).unwrap();
    assert_eq!(
        extract_pattern_idents(&src[pattern]),
        vec!["it".to_string()]
    );

    let src = "let (a, mut b) = (data.par_iter(), 0)";
    let (pattern, _) = parse_let_binding_range(src).unwrap();
    assert_eq!(&src[pattern], " (a, mut b) ");
}

#[test]
fn extract_taint_targets_detects_plain_assignment() {
    let tainted = HashSet::new();
    assert_eq!(
        extract_taint_targets("it = data.par_iter()", &tainted),
        vec!["it".to_string()]
    );
    assert_eq!(
        extract_taint_targets("it.sum()", &tainted),
        Vec::<String>::new()
    );
}

#[test]
fn count_par_reduce_does_not_break_chain_on_unordered_collect() {
    // 順序保持が型で明示されない collect（HashSet・型推論任せ）の後の
    // 逐次縮約は連鎖を遮断しない（fail-closed 側に倒す）。
    let unordered = "fn f(data: &[u32]) -> u32 { data.par_iter().copied().collect::<HashSet<_>>().into_iter().fold(0, |a, b| a ^ b) }";
    assert_eq!(count_par_reduce_cooccurrences(unordered), 1);
    let inferred = "fn f(data: &[f32]) -> f32 { data.par_iter().copied().collect().into_iter().fold(0.0, |a, b| a + b) }";
    assert_eq!(count_par_reduce_cooccurrences(inferred), 1);
    let ordered = "fn f(data: &[f32]) -> f32 { data.par_chunks(4).map(|c| c[0]).collect::<Vec<_>>().into_iter().fold(0.0, |a, b| a + b) }";
    assert_eq!(count_par_reduce_cooccurrences(ordered), 0);
}

// =====================================================================
// レキシカルスコープ管理・UFCS・`::par_` の回帰テスト
// （敵対的レビュー指摘: `looks_like_fn_item` による `tainted.clear()`
// が、文の先頭 `{` より前に `fn` が現れるだけの偽陽性〈ローカル
// fn・`fn(i32) -> i32` 型注釈・ローカル impl〉で汚染集合を全消去して
// いた不具合の再発防止。`count_in_scope` のレキシカルスコープ方式・
// UFCS 縮約呼び出し・`::par_` パス参照検出・Vec collect による
// `extract_taint_targets` 側の連鎖遮断を検証する）
// =====================================================================

#[test]
fn count_par_reduce_survives_local_fn_item_between_taint_and_usage() {
    // ローカル fn を挟んでも、それ以前に汚染された識別子の追跡が
    // 継続すること（旧実装は `looks_like_fn_item` が「fn」という語の
    // 出現だけで判定していたため、ここで汚染集合が全消去されていた）。
    let src =
        "fn f(data: &[f32]) -> f32 { let it = data.par_iter(); fn helper() -> i32 { 7 } it.sum() }";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_survives_fn_pointer_type_annotation() {
    // `fn(i32) -> i32` 型注釈（関数アイテムではない）を挟んでも汚染
    // 集合を全消去しないこと。
    let src = "fn f(data: &[f32], h: fn(i32) -> i32) -> f32 { let it = data.par_iter(); let cb: fn(i32) -> i32 = h; it.sum() }";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_survives_local_impl_block() {
    // ローカル impl（`fn` を含む）を挟んでも汚染集合を全消去しない
    // こと。impl 内の `fn m` 自身は独立スコープ（本体は空集合開始）
    // で処理されるが、外側関数の汚染集合には影響しない。
    let src = "fn f(data: &[f32]) -> f32 { let it = data.par_iter(); impl S { fn m(&self) {} } it.sum() }";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_ufcs_sum() {
    // UFCS（fully-qualified）形の縮約呼び出し（`.sum(` ではなく
    // `::sum(`）も検出する（敵対的レビュー指摘 REQ 2）。
    let src = "ParallelIterator::sum(data.par_iter().map(|x| x))";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_assignment_taint_leaking_out_of_nested_block() {
    // ネストしたブロック内の `let` なし代入による汚染は、外側スコープ
    // へ伝播する（`let` による束縛は伝播しない点との対比。REQ 1）。
    let src = "let mut it; if c { it = data.par_iter(); } it.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_par_iter_as_method_value() {
    // メソッド値／パス参照経由の並列マーカー（`T::par_iter` を変数へ
    // 束縛して後から呼び出す形。REQ 4）。
    let src = "let f = T::par_iter; let it = f(&data); it.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_vec_collect_before_let_bound_sum() {
    // Vec への collect で連鎖が切れた後の `let` 経由の縮約は汚染しない
    // （敵対的レビュー指摘 REQ 3。`extract_taint_targets`／
    // `classify_taint` 側にも `.collect(` 連鎖遮断を適用する）。
    let src = "let parts = data.par_iter().map(f).collect::<Vec<f32>>(); \
               let s = parts.iter().sum::<f32>();";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_ignores_taint_across_separate_fn_items() {
    // 別々の fn 間でパラメータ名が衝突しても汚染が漏れないこと
    // （`fn a` の `it`〈汚染〉と `fn b` の `it`〈引数。無関係〉は
    // 独立したスコープを持つ）。
    let src = "fn a(d: &[f32]) { let it = d.par_iter(); } \
               fn b(it: &[f32]) -> f32 { it.iter().sum() }";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_ignores_let_taint_leaking_out_of_nested_block() {
    // ネストしたブロック内の `let` による汚染は外側スコープへ漏れない
    // こと（`let なし代入だけが伝播する`〈REQ 1〉との対比）。ブロック
    // 外で同名 `it` を非汚染の初期化式で再束縛しても検出されない。
    let src = "{ let it = d.par_iter(); } let it = v; it.iter().sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}
