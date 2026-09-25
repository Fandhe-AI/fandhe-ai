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
//!     PR #2274 で 6 ラウンド、その後の敵対的レビューでさらに 11 件の
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
//! 敵対的レビューで指摘された。さらにその是正後、次の 4 件が追加で
//! 指摘された: (i) 「並列マーカーと同じか浅い深さの collect なら
//! 一律に遮断する」という単純な深さ比較は、`(data.par_iter(),
//! other.collect::<Vec<_>>()).0` のようにタプル要素として無関係に
//! 同居するだけの collect まで遮断してしまう（codex-review 指摘 P1）。
//! (ii) `statement_own_view` がネストした `{...}` ブロック本体を単純
//! 空白化していたため、`{ .. }.sum()` のようにブロック式の直後に
//! 連鎖する縮約が、同一文判定・UFCS 判定のどちらからも見えなかった
//! （Cursor Bugbot 指摘）。(iii) (ii) の初回是正はブロック本体を合成
//! マーカー文字列（`.par_iter(` 等）へ実際に書き換える方式だった
//! ため、ブロック本体が合成マーカーの最小長（7 バイト）未満だと検出
//! できない Critical な抜けがあった（`{it}`・`{ acc }` のような短い
//! 汚染識別子 1 つだけのブロック式。敵対的レビュー指摘）。
//! `statement_own_view` がテキストへ何も書き込まず「仮想マーカー
//! 位置」を別チャネル（バイト位置のリスト）として返す方式へ置き換え、
//! ブロック本体の長さに一切依存しない検出とした。(iv) さらにその後、
//! 4 件が同時に指摘された: 固定文字列の部分一致（`.par_iter(`・
//! `.sum(` 等）では空白・改行を挟む呼び出し（`data.par_iter ()
//! .sum ()`・複数行の連鎖・`sum :: < f32 > ()` のような turbofish の
//! 空白入り形）を見逃す（codex-review 指摘 P1）・固定文字列
//! `.into_par_iter(` では `rayon::iter::IntoParallelIterator::
//! into_par_iter(data)` のような UFCS 形の任意の修飾パスを拾えない
//! （codex-review 指摘 P1）・(i) で導入した「同一メソッド連鎖」判定が
//! `{` を連鎖の構成要素として扱っておらず、ブロック式の仮想マーカーの
//! 直後に `.collect::<Vec<_>>()` が続いても遮断が素通りする過検出
//! （Cursor Bugbot 指摘）・仮想マーカーの生存判定が並列マーカーには
//! collect 遮断を適用する一方、汚染識別子にはしていなかったため
//! `Vec` collect された連鎖でしか使われていない汚染識別子でもブロック
//! を「生きている」と誤判定する過検出（Cursor Bugbot 指摘）。この
//! 4 件は同類型（固定文字列一致・字句レベルの近似の限界）としてまとめ
//! て塞ぎ、マーカー検出のすべて（並列マーカー・縮約マーカー・collect
//! マーカー・連鎖の連続性判定）を `tokenize` による簡易字句解析
//! （トークン列の照合）へ置き換えた。現行方式は次の 5 点で構成する:
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
//! 3. **トークン照合によるマーカー検出**（`tokenize`・`find_par_marker_tokens`・
//!    `find_reduce_marker_tokens`・`find_ufcs_reduce_calls`・
//!    `find_collect_marker_tokens`）: 空白・改行を無視してトークン列
//!    へ分割し、並列マーカーは「識別子が `par_` 接頭辞または
//!    `into_par_iter` で、直前のトークンが `.` または `::`」、縮約
//!    マーカーは「識別子が縮約名の集合（`REDUCE_MARKER_NAMES`）に
//!    含まれ、直前が `.`（メソッド形）または `::`（UFCS 形）で、直後
//!    が呼び出し括弧または turbofish 経由の呼び出し括弧」、collect
//!    マーカーは「`.collect::<Vec<`・`.collect_into_vec(`」で判定する。
//!    直前・直後のトークンのみを見るため、`T::par_iter`（パス参照
//!    形）・`rayon::iter::IntoParallelIterator::into_par_iter(data)`
//!    （UFCS 形。修飾パスの深さ・区切りの数によらない）・
//!    `ParallelIterator::sum(...)`（UFCS 形の縮約。`has_ufcs_reduction`
//!    が呼び出し引数リストの内側を別途走査し、並列マーカーまたは汚染
//!    識別子があれば 1 件とカウントする。同一文内で他規則と二重計上
//!    しない）を一律に扱える。
//! 4. **`.collect(` による並列→逐次の連鎖の遮断（同一メソッド連鎖上
//!    に限る）**: 並列マーカー・汚染識別子と縮約マーカーの間に collect
//!    マーカーが挟まる場合は接続しないとみなすが、「同じか浅い深さ」
//!    という単純な比較だけでは遮断範囲が広すぎる（codex-review 指摘
//!    P1: `(data.par_iter(), other.collect::<Vec<_>>()).0` のように
//!    無関係なタプル要素として同居するだけの collect まで遮断して
//!    しまう）。`is_same_method_chain` が「到達点（collect）自身の
//!    深さを連鎖の合流点とみなし、そこまでトークンの深さが一度も
//!    下回らず、かつ合流点の深さちょうどにあるトークンがすべて
//!    メソッド連鎖の構成要素（識別子・`.`／`::`／`:`・turbofish の
//!    `<`・`>`・`?`・括弧〈`()`／`{}`／`[]` すべて〉）であること」を
//!    判定し、これを満たす collect のみを遮断とみなす
//!    （`has_reduce_after`）。括弧に `{`／`}` を含めるのは Cursor
//!    Bugbot 指摘の是正: 仮想マーカーの起点はブロックの `{` トークン
//!    そのものであり、これを連鎖の構成要素として認めないと
//!    `{ data.par_iter() }.collect::<Vec<_>>()` のようなブロック式
//!    直後の collect による遮断が素通りしてしまう（過検出）。同じ
//!    判定は文をまたぐ汚染の**発生源**判定（`initializer_taints`）
//!    にも適用する: `let parts = data.par_iter().map(f)
//!    .collect::<Vec<f32>>();` のように初期化式が Vec 確定で終わる
//!    場合は束縛先を汚染しない。`.collect(` の後に再び並列マーカーが
//!    現れる場合（`x.par_iter().collect::<Vec<_>>().par_iter()`）は
//!    その後方のマーカー自身が汚染源になる。
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
//! 塗りつぶしたテキストと「仮想マーカー位置」の一覧（`statement_own_view`
//! の戻り値）に対して行う（ネスト内は `count_in_scope` の再帰が別の
//! レキシカルスコープとして検査するため二重計上しない）。ブロック本体
//! が「生きている（4 の規則で collect に遮断されていない）並列
//! マーカー」または「生きている（同じく collect 遮断を適用した）
//! 汚染識別子」を含む場合は、ブロックの開き `{` のバイト位置を仮想
//! マーカーとして記録し（テキストへは何も書き込まない。長さに依存
//! しない副チャネル。`statement_own_view` doc 参照）、
//! `has_depth_aware_par_reduce`・`has_ufcs_reduction`・連鎖の遮断判定
//! （`is_same_method_chain` 経由）のいずれも実マーカーと仮想マーカーの
//! 両方を起点として扱う。`let` 初期化式の汚染判定（`classify_taint`）
//! は塗りつぶし前の初期化式全体を見る（`let it = { let x = 1;
//! data.par_iter() };` のような場合に `.par_iter(` を見失わないため）。
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
//!
//! `is_same_method_chain` は「合流点（collect）の深さちょうどにある
//! トークン」のみで連鎖の連続性を判定するトークンレベルの近似であり、
//! 完全な構文解析ではない。`.zip(other.par_X(..))` のように、起点
//! より深い位置（`.zip(` の引数内）にある並列マーカーが、無関係な
//! 縮約マーカーと**単一式クロージャ本体**の中でたまたま同じ括弧深さに
//! なる場合、理論上は誤って共起と判定されうる（本クレートの実際の
//! コードは複数行の縮約を伴うクロージャに一貫してブロック本体
//! `{ .. }` を使うため、この深さの偶然の一致は 1 段回避される。
//! `count_par_reduce_does_not_flag_zip_argument_marker_blocked_by_outer_collect`
//! の doc コメント参照）。

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
/// 判定する（`contains_rayon_marker` 専用。`find_par_marker_tokens`
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

/// `raw`（1 文）が内包する深さ 0 の `{...}` ブロック本体を空白で
/// 塗りつぶしたテキストと、「仮想の並列マーカー位置」の一覧を返す
/// （`(塗りつぶし済みテキスト, 仮想マーカーのバイト位置一覧)`）。
/// 同一文中の並列マーカー／縮約マーカー共起判定
/// （`has_depth_aware_par_reduce`・`has_tainted_reduction_usage`・
/// `has_ufcs_reduction`）はこのテキスト・仮想マーカー一覧に対して
/// 行い、ネストしたブロックの中身は `count_in_scope` が別途、独立
/// したレキシカルスコープとして再帰検査するため二重計上しない。
/// `let` 初期化式の汚染判定（`classify_taint`）はこれとは別に塗り
/// つぶし前の `raw` をそのまま使う。
///
/// ブロック本体が「生きている（`is_same_method_chain` の判定で Vec
/// collect に遮断されていない）並列マーカー」または「生きている汚染
/// 識別子」を含む場合、ブロックの開き `{` のバイト位置を仮想マーカー
/// 位置として記録する（トークン化した際の深さは `{` 自身のトークン
/// 深さ——すなわちブロック式そのものの、ブロックの外側から見た深さ
/// ——と一致する）。これにより `{ let x = 1; data.par_iter() }.sum()`・
/// `if c { a.par_iter() } else { b.par_iter() }.sum()`・
/// `unsafe { data.par_iter() }.sum::<f32>()`・`match k { _ =>
/// data.par_iter() }.sum::<f32>()`・`if flag { it } else { acc }.sum()`
/// （`it`／`acc` が汚染済み）のようにブロック式の直後に連鎖した縮約を、
/// 同一文中の共起判定・UFCS 判定が検出できる（Cursor Bugbot 指摘: 旧
/// 実装は塗りつぶし後に完全に空白化していたため、ブロックの外側からも
/// ブロックを再帰検査する `count_in_scope` の子スコープからも、この
/// 連鎖が見えなかった）。
///
/// 汚染識別子の「生存」判定（`has_live_tainted_ref`）は並列マーカーと
/// **同じ** collect 遮断規則を適用する。旧実装は汚染識別子の存在
/// だけで仮想マーカーを立てており、`{ it.map(f).collect::<Vec<_>>() }
/// .into_iter().sum();`（`it` は汚染済み）のように、ブロック内で
/// `it` が Vec collect された連鎖でしか使われていない場合まで「生きて
/// いる」と誤判定していた（Cursor Bugbot 指摘）。
fn statement_own_view(raw: &str, tainted: &HashSet<String>) -> (String, Vec<usize>) {
    let spans = find_depth0_braces(raw);
    let mut out: Vec<u8> = raw.as_bytes().to_vec();
    let mut virtual_markers = Vec::new();

    for (open_idx, close_idx) in &spans {
        let interior_start = open_idx + 1;
        let interior_end = *close_idx;
        if interior_start >= interior_end {
            continue;
        }
        let interior_tokens = tokenize(&raw[interior_start..interior_end]);
        let is_live = has_live_chain_marker(&interior_tokens, &[])
            || has_live_tainted_ref(&interior_tokens, tainted);

        if is_live {
            virtual_markers.push(*open_idx);
        }
        for b in &mut out[interior_start..interior_end] {
            *b = b' ';
        }
    }

    let view = String::from_utf8(out).expect(
        "空白への置換は find_depth0_braces（char_indices 由来で文字境界に\
         整列済み）が返す範囲のみを対象とし、範囲の開始・終了は常に有効な\
         文字境界であるため、範囲内を ASCII 空白で埋め尽くしても全体として \
         UTF-8 として有効であり続ける",
    );
    (view, virtual_markers)
}

// =====================================================================
// トークン化・並列マーカー・縮約マーカーの検出（敵対的レビュー指摘
// codex P1 × 2・Cursor Bugbot Medium × 2 の是正）
// =====================================================================
//
// 旧実装は固定文字列の部分一致（`text.contains(".par_iter(")` 等）で
// マーカーを検出していたため、次の 2 件を見逃していた:
//   - 空白・改行を挟むメソッド呼び出し（`data.par_iter ().sum ()`・
//     複数行に折り返した連鎖・`sum :: < f32 > ()` のような turbofish
//     の空白入り形）。原因は `.par_iter(`／`.sum(` 等の固定文字列一致
//     そのものが空白を許容しないこと。
//   - UFCS 形の任意の修飾パス（`rayon::iter::IntoParallelIterator::
//     into_par_iter(data)`）。原因は `.into_par_iter(` という固定
//     文字列が「直前が `.`」の形しか拾えないこと。
// マーカー検出のすべて（並列マーカー・縮約マーカー・collect マーカー・
// 連鎖の連続性判定）を、簡易字句解析によるトークン列の照合へ置き換える。

/// トークンの種別。識別子・区切り記号（`::` は 1 トークンとして結合
/// する）・括弧・その他の記号に分類する。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenKind {
    Ident,
    Dot,
    ColonColon,
    Colon,
    Lt,
    Gt,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Question,
    /// `,`・`;`・`=`・二項演算子・`|` 等、連鎖の構成要素とみなさない
    /// その他の記号。
    Other,
}

/// トークン 1 個。`start` は `tokenize` の入力テキスト中のバイト位置
/// （文字境界）、`depth` は `find_depth0_braces` 等と同じ規約
/// （`()`／`[]`／`{}` 合算。開き括弧はそれ自身の増分前、閉じ括弧は
/// それ自身の減分前の深さ）でのトークン自身の括弧深さ。
#[derive(Clone, Copy, Debug)]
struct Token<'a> {
    kind: TokenKind,
    text: &'a str,
    start: usize,
    depth: i32,
}

/// `text`（既に `strip_comments_and_strings` 済み。コメント・文字列・
/// char/byte literal は既に空白へ置換済みのため、本関数はそれ以外の
/// 記号のみを扱えばよい）をトークン列へ分割する。空白・改行はトークン
/// 間で無視する（これにより `data.par_iter ()`・複数行の連鎖・
/// `sum :: < f32 > ()` のような空白入りの書き方も、詰めて書いた場合と
/// 同一のトークン列になる）。
fn tokenize(text: &str) -> Vec<Token<'_>> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut tokens = Vec::new();
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < n {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if is_ident_char(c) {
            let start = i;
            // raw identifier（`r#sum`・`r#par_iter` 等）は `r#` を除いた
            // 識別子 1 トークンとして扱う（`r`・`#`・`sum` の 3 トークンへ
            // 分かれると直前トークン判定〈`.`／`::`〉をすり抜けるため）。
            let name_start = if c == 'r'
                && bytes.get(i + 1) == Some(&b'#')
                && bytes.get(i + 2).is_some_and(|&b| is_ident_char(b as char))
            {
                i += 2;
                i
            } else {
                start
            };
            while i < n && is_ident_char(bytes[i] as char) {
                i += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Ident,
                text: &text[name_start..i],
                start,
                depth,
            });
            continue;
        }
        if c == ':' && bytes.get(i + 1) == Some(&b':') {
            tokens.push(Token {
                kind: TokenKind::ColonColon,
                text: &text[i..i + 2],
                start: i,
                depth,
            });
            i += 2;
            continue;
        }
        let start = i;
        let kind = match c {
            '.' => TokenKind::Dot,
            ':' => TokenKind::Colon,
            '<' => TokenKind::Lt,
            '>' => TokenKind::Gt,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '?' => TokenKind::Question,
            _ => TokenKind::Other,
        };
        tokens.push(Token {
            kind,
            text: &text[start..start + 1],
            start,
            depth,
        });
        match kind {
            TokenKind::LParen | TokenKind::LBrace | TokenKind::LBracket => depth += 1,
            TokenKind::RParen | TokenKind::RBrace | TokenKind::RBracket => {
                depth = (depth - 1).max(0);
            }
            _ => {}
        }
        i += 1;
    }
    tokens
}

/// rayon の分割依存縮約を表す識別子名の集合。`sum`／`product`／
/// `reduce`／`reduce_with`・`fold` 系（`fold`／`fold_with`／
/// `fold_chunks`）・`try_fold`／`try_reduce`／`try_reduce_with`
/// （rayon の `fold` 系はスレッド分割依存の部分結果を返すため
/// `sum`／`reduce` と同類型）。メソッド形（直前が `.`）・UFCS 形
/// （直前が `::`）のいずれで現れても同じ名前集合で判定する。
const REDUCE_MARKER_NAMES: &[&str] = &[
    "sum",
    "product",
    "reduce",
    "reduce_with",
    "fold",
    "fold_with",
    "fold_chunks",
    "try_fold",
    "try_reduce",
    "try_reduce_with",
];

/// `tokens[ident_idx]`（識別子トークン）の直後が呼び出し括弧 `(` か、
/// turbofish（`::` `<` ... 対応する `>` ... `(`）かを判定し、その場合
/// 呼び出しの開き `(` のトークンインデックスを返す。
fn call_open_paren_after(tokens: &[Token], ident_idx: usize) -> Option<usize> {
    let mut i = ident_idx + 1;
    if tokens.get(i).map(|t| t.kind) == Some(TokenKind::LParen) {
        return Some(i);
    }
    if tokens.get(i).map(|t| t.kind) == Some(TokenKind::ColonColon)
        && tokens.get(i + 1).map(|t| t.kind) == Some(TokenKind::Lt)
    {
        i += 2;
        let mut depth = 1i32;
        while i < tokens.len() && depth > 0 {
            match tokens[i].kind {
                TokenKind::Lt => depth += 1,
                TokenKind::Gt => depth -= 1,
                _ => {}
            }
            i += 1;
        }
        if tokens.get(i).map(|t| t.kind) == Some(TokenKind::LParen) {
            return Some(i);
        }
    }
    None
}

/// `tokens[open_idx]` が `(` であることを前提に、対応する `)` の
/// トークンインデックスを深さ追跡（ネスト対応）で返す。
fn matching_close_paren(tokens: &[Token], open_idx: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (j, tok) in tokens.iter().enumerate().skip(open_idx) {
        match tok.kind {
            TokenKind::LParen => depth += 1,
            TokenKind::RParen => {
                depth -= 1;
                if depth == 0 {
                    return Some(j);
                }
            }
            _ => {}
        }
    }
    None
}

/// `tokens` 中で並列マーカー（識別子が `par_` 接頭辞または
/// `into_par_iter` であり、直前の非空白トークンが `.` または `::`）
/// であるトークンのインデックス一覧を返す。メソッド呼び出し形
/// （`.par_iter(`）・パス参照形（`T::par_iter`。メソッド値として
/// 変数へ束縛して後から呼び出す形〈`let f = T::par_iter; let it =
/// f(&data); it.sum()`〉のため末尾 `(` の有無を問わない）・UFCS 形
/// （`rayon::iter::IntoParallelIterator::into_par_iter(data)`。直前
/// トークンのみを見るため、修飾パスの深さ・区切りの数によらず検出
/// する）を一律に扱う（敵対的レビュー指摘 codex P1）。
fn find_par_marker_tokens(tokens: &[Token]) -> Vec<usize> {
    (1..tokens.len())
        .filter(|&i| {
            let tok = &tokens[i];
            tok.kind == TokenKind::Ident
                && (tok.text.starts_with("par_") || tok.text == "into_par_iter")
                && matches!(tokens[i - 1].kind, TokenKind::Dot | TokenKind::ColonColon)
        })
        .collect()
}

/// `tokens` 中でメソッド呼び出し形の縮約マーカー（識別子が
/// `REDUCE_MARKER_NAMES` に含まれ、直前が `.`、直後が呼び出し括弧
/// または turbofish 経由の呼び出し括弧）であるトークンのインデックス
/// 一覧を返す。
fn find_reduce_marker_tokens(tokens: &[Token]) -> Vec<usize> {
    (1..tokens.len())
        .filter(|&i| {
            let tok = &tokens[i];
            tok.kind == TokenKind::Ident
                && REDUCE_MARKER_NAMES.contains(&tok.text)
                && tokens[i - 1].kind == TokenKind::Dot
                && call_open_paren_after(tokens, i).is_some()
        })
        .collect()
}

/// `tokens` 中の UFCS 形の縮約呼び出し（`ParallelIterator::sum(...)`・
/// `Trait::reduce(...)` 等。直前が `::` の `REDUCE_MARKER_NAMES`）を
/// 検出し、`(識別子トークン index, 開き ( トークン index, 閉じ )
/// トークン index)` を列挙する（敵対的レビュー指摘 REQ 2）。
fn find_ufcs_reduce_calls(tokens: &[Token]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for i in 1..tokens.len() {
        let tok = &tokens[i];
        if tok.kind != TokenKind::Ident || !REDUCE_MARKER_NAMES.contains(&tok.text) {
            continue;
        }
        if tokens[i - 1].kind != TokenKind::ColonColon {
            continue;
        }
        if let Some(open) = call_open_paren_after(tokens, i)
            && let Some(close) = matching_close_paren(tokens, open)
        {
            out.push((i, open, close));
        }
    }
    out
}

/// `tokens` 中で `Vec` への collect マーカー（`.collect::<Vec<`・
/// `.collect_into_vec(`）であるトークン（`collect`／`collect_into_vec`
/// 識別子トークン）のインデックス一覧を返す。順序保持が型で明示される
/// `Vec` への collect に限る。型推論に任せた `.collect()` や
/// `HashMap`／`HashSet` 等への collect は反復順序が決まらない場合が
/// あるため対象外とし、後続の縮約を並列縮約として数える
/// （fail-closed 側。`.collect(` でインデックス順の `Vec` へ一旦確定
/// したあとの `.into_iter()` 以降は通常の逐次 `Iterator` になるため、
/// `Vec` への collect より後方の縮約マーカーは直前の並列マーカー・
/// 汚染識別子とは**接続されない**——`is_same_method_chain` がこの
/// 遮断を判定する）。
fn find_collect_marker_tokens(tokens: &[Token]) -> Vec<usize> {
    (1..tokens.len())
        .filter(|&i| {
            let tok = &tokens[i];
            if tok.kind != TokenKind::Ident || tokens[i - 1].kind != TokenKind::Dot {
                return false;
            }
            match tok.text {
                "collect" => {
                    tokens.get(i + 1).map(|t| t.kind) == Some(TokenKind::ColonColon)
                        && tokens.get(i + 2).map(|t| t.kind) == Some(TokenKind::Lt)
                        && tokens.get(i + 3).map(|t| (t.kind, t.text))
                            == Some((TokenKind::Ident, "Vec"))
                }
                "collect_into_vec" => tokens.get(i + 1).map(|t| t.kind) == Some(TokenKind::LParen),
                _ => false,
            }
        })
        .collect()
}

/// トークン種別が「メソッド連鎖の構成要素」とみなせるかを判定する
/// （識別子・`.`／`::`／`:`・turbofish の `<`・`>`・`?`・括弧
/// 〈`()`／`{}`／`[]` すべて〉）。`,`・`;`・`=`・二項演算子・`|` 等
/// （`TokenKind::Other`）は含まない。
///
/// 括弧に `{`／`}` を含めるのは Cursor Bugbot 指摘の是正: 仮想マーカー
/// （`statement_own_view` がブロック式の開き `{` に立てるマーカー）の
/// 起点はブロックの `{` トークンそのものであり、`{ data.par_iter() }
/// .collect::<Vec<_>>()` のように直後に collect が続く場合、`{`
/// トークン自身がちょうど collect の合流点の深さに位置する。`{` を
/// 連鎖の構成要素として認めないと、この `{` の時点で判定が打ち切られ
/// 同一連鎖とみなされず、ブロック式の直後の collect による遮断が
/// 素通りしてしまう（過検出）。
fn is_chain_connector(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Ident
            | TokenKind::Dot
            | TokenKind::ColonColon
            | TokenKind::Colon
            | TokenKind::Lt
            | TokenKind::Gt
            | TokenKind::Question
            | TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::LBracket
            | TokenKind::RBracket
    )
}

/// `tokens` において、`from_idx` から `to_idx` までが**同一のメソッド
/// 連鎖上**にあるかを判定する（codex-review 指摘 P1: 無関係な `Vec`
/// collect による誤遮断の是正。`let it = (data.par_iter(), other
/// .collect::<Vec<_>>()).0;` のようにタプル要素として同居するだけの
/// collect は `data.par_iter()` の連鎖を遮断しない）。
///
/// `tokens[to_idx]`（`.collect(` 等の到達点）自身の深さ
/// `target_depth` を「連鎖の合流点の深さ」とみなし、次の 2 条件を
/// 満たす場合のみ真を返す:
///
/// 1. `from_idx..to_idx` の範囲でトークンの深さが `target_depth` を
///    一度も下回らない（`from_idx` 自身の深さが `target_depth` より
///    深い場合——たとえば `data.par_chunks(..).zip(other.par_chunks(..))
///    .map(..).collect(..)` の `other.par_chunks(` のように `.zip(`
///    の引数として起点より深い位置にある場合——は、起点からいったん
///    `.zip(`／`.map(` の呼び出し括弧を閉じて `target_depth` まで
///    戻ってくる正当な連鎖として許容する）。
/// 2. `target_depth` ちょうどの深さにあるトークンが、すべて
///    `is_chain_connector` を満たすこと。
fn is_same_method_chain(tokens: &[Token], from_idx: usize, to_idx: usize) -> bool {
    if from_idx >= to_idx {
        return false;
    }
    let target_depth = tokens[to_idx].depth;
    for tok in &tokens[from_idx..to_idx] {
        if tok.depth < target_depth {
            return false;
        }
        if tok.depth == target_depth && !is_chain_connector(tok.kind) {
            return false;
        }
    }
    true
}

/// `origin_idx` より後方（トークンインデックス順）かつ**同じか浅い**
/// 括弧深さに縮約マーカーが現れるかを判定する。ただし `origin_idx` と
/// 縮約マーカーの間に、`origin_idx` と**同一のメソッド連鎖上**
/// （`is_same_method_chain`）にある collect マーカーが挟まる場合は
/// イテレータ連鎖が断ち切られているとみなし接続しない。縮約マーカー
/// 側の「同じか浅い深さ」判定自体は緩めない（fail-closed 側を維持
/// する）。
fn has_reduce_after(
    tokens: &[Token],
    origin_idx: usize,
    reduce_tokens: &[usize],
    collect_tokens: &[usize],
) -> bool {
    let origin_depth = tokens[origin_idx].depth;
    reduce_tokens.iter().any(|&r_idx| {
        r_idx > origin_idx
            && tokens[r_idx].depth <= origin_depth
            && !collect_tokens.iter().any(|&c_idx| {
                c_idx > origin_idx
                    && c_idx < r_idx
                    && is_same_method_chain(tokens, origin_idx, c_idx)
            })
    })
}

/// `tokens` 中に「生きている」（`is_same_method_chain` の意味で同一
/// 連鎖上の `Vec` collect に遮断されていない）並列マーカーが 1 つでも
/// あるかを判定する。`extra_par_tokens`（`tokens` 中のトークン
/// インデックス一覧）が与えられた場合、実マーカーと同様に起点として
/// 扱う（`statement_own_view` のブロック本体を「生きているか」判定
/// する際は `&[]`。`has_ufcs_reduction` が UFCS 呼び出し引数内の
/// 仮想マーカーを渡す際に使う）。
fn has_live_chain_marker(tokens: &[Token], extra_par_tokens: &[usize]) -> bool {
    let collect_tokens = find_collect_marker_tokens(tokens);
    let mut par_tokens = find_par_marker_tokens(tokens);
    par_tokens.extend_from_slice(extra_par_tokens);
    par_tokens.into_iter().any(|p_idx| {
        !collect_tokens
            .iter()
            .any(|&c_idx| c_idx > p_idx && is_same_method_chain(tokens, p_idx, c_idx))
    })
}

/// `tokens` 中に、`tainted` のいずれかの識別子と同名のトークンで
/// 「生きている」（同一連鎖上の `Vec` collect に遮断されていない）
/// 出現が 1 つでもあるかを判定する。並列マーカーと同じ collect 遮断
/// 規則を適用する（Cursor Bugbot 指摘: 旧実装は汚染識別子の出現有無
/// のみで判定しており、`{ it.map(f).collect::<Vec<_>>() }
/// .into_iter().sum();`〈`it` は汚染済み〉のように Vec collect された
/// 連鎖でしか使われていない汚染識別子まで「生きている」と誤判定して
/// いた）。トークン照合のため識別子境界は自然に保たれる。
fn has_live_tainted_ref(tokens: &[Token], tainted: &HashSet<String>) -> bool {
    let collect_tokens = find_collect_marker_tokens(tokens);
    tainted.iter().any(|name| {
        find_ident_name_tokens(tokens, name).into_iter().any(|idx| {
            !collect_tokens
                .iter()
                .any(|&c_idx| c_idx > idx && is_same_method_chain(tokens, idx, c_idx))
        })
    })
}

/// `tokens` 中で識別子トークンのテキストが `name` と一致するものの
/// インデックス一覧を返す（トークン照合のため識別子境界は自然に
/// 保たれる）。
fn find_ident_name_tokens(tokens: &[Token], name: &str) -> Vec<usize> {
    tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.kind == TokenKind::Ident && t.text == name)
        .map(|(i, _)| i)
        .collect()
}

/// `tokens`（`view` をトークン化したもの）中で、仮想マーカーの
/// バイト位置一覧 `byte_positions`（`statement_own_view` の戻り値）に
/// 対応する `{` トークンのインデックス一覧を返す。
fn virtual_marker_token_indices(tokens: &[Token], byte_positions: &[usize]) -> Vec<usize> {
    byte_positions
        .iter()
        .filter_map(|&p| {
            tokens
                .iter()
                .position(|t| t.kind == TokenKind::LBrace && t.start == p)
        })
        .collect()
}

/// 1 文の中で、並列マーカー（実マーカー・`virtual_markers` の仮想
/// マーカーの両方）より後方かつ同じか浅い括弧深さに縮約マーカーが
/// 現れるかを判定する（`data.par_iter().map(..).sum::<f32>()` は
/// 検出、`data.par_chunks(n).map(|c| c.iter().sum::<f32>())
/// .collect(..)`〈チャンク内逐次和〉・`data.par_chunks(..).map(..)
/// .collect::<Vec<_>>().into_iter().fold(..)`〈collect で連鎖が切れる〉
/// は非検出）。`view` は `statement_own_view` で塗りつぶし済みの
/// テキスト、`virtual_markers` は同関数が返す仮想マーカー位置一覧。
fn has_depth_aware_par_reduce(view: &str, virtual_markers: &[usize]) -> bool {
    let tokens = tokenize(view);
    let mut par_tokens = find_par_marker_tokens(&tokens);
    par_tokens.extend(virtual_marker_token_indices(&tokens, virtual_markers));
    if par_tokens.is_empty() {
        return false;
    }
    let reduce_tokens = find_reduce_marker_tokens(&tokens);
    if reduce_tokens.is_empty() {
        return false;
    }
    let collect_tokens = find_collect_marker_tokens(&tokens);
    par_tokens
        .iter()
        .any(|&p_idx| has_reduce_after(&tokens, p_idx, &reduce_tokens, &collect_tokens))
}

/// 過去の文で汚染された識別子（`tainted`）が、この文中でトークンと
/// して一致し、かつ後方の同じか浅い括弧深さに縮約マーカーが現れるか
/// を判定する（文をまたぐ並列縮約。`view` は `statement_own_view`）。
/// `.collect(` による連鎖の遮断は `has_reduce_after` と同じ規則。
fn has_tainted_reduction_usage(view: &str, tainted: &HashSet<String>) -> bool {
    if tainted.is_empty() {
        return false;
    }
    let tokens = tokenize(view);
    let reduce_tokens = find_reduce_marker_tokens(&tokens);
    if reduce_tokens.is_empty() {
        return false;
    }
    let collect_tokens = find_collect_marker_tokens(&tokens);
    tainted.iter().any(|name| {
        find_ident_name_tokens(&tokens, name)
            .into_iter()
            .any(|idx| has_reduce_after(&tokens, idx, &reduce_tokens, &collect_tokens))
    })
}

/// UFCS 形の縮約呼び出し（`ParallelIterator::sum(data.par_iter()...)`
/// 等）を検出し、その引数リスト内に生きている（`has_live_chain_marker`。
/// 同一連鎖上の Vec collect で遮断されていない）並列マーカー（実マーカー・
/// `virtual_markers` の仮想マーカーの両方）または生きている汚染識別子
/// （`has_live_tainted_ref`）があれば真を返す（敵対的レビュー指摘
/// REQ 2。`view`・`virtual_markers` は `statement_own_view` の戻り値）。
/// 引数リストは呼び出しごとに独立して再トークン化する（深さ 0 起点の
/// 局所的な連鎖判定にするため。元の `has_depth_aware_par_reduce` と
/// 同じ理由）。
fn has_ufcs_reduction(view: &str, tainted: &HashSet<String>, virtual_markers: &[usize]) -> bool {
    let tokens = tokenize(view);
    for (_, open, close) in find_ufcs_reduce_calls(&tokens) {
        let arg_start = tokens[open].start + 1;
        let arg_end = tokens[close].start;
        if arg_start >= arg_end {
            continue;
        }
        let args = &view[arg_start..arg_end];
        let arg_tokens = tokenize(args);
        let local_virtual_markers: Vec<usize> = virtual_markers
            .iter()
            .filter(|&&p| p >= arg_start && p < arg_end)
            .filter_map(|&p| {
                arg_tokens
                    .iter()
                    .position(|t| t.kind == TokenKind::LBrace && t.start == p - arg_start)
            })
            .collect();
        if has_live_chain_marker(&arg_tokens, &local_virtual_markers) {
            return true;
        }
        if has_live_tainted_ref(&arg_tokens, tainted) {
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
/// 初期化式を誤って汚染していた過検出の是正）。並列マーカー（`::par_`
/// パス参照・UFCS 形を含む）または既存の汚染識別子が、その後ろに
/// **同一のメソッド連鎖上**（`is_same_method_chain`）の Vec への
/// collect を伴わずに現れる場合のみ汚染源とみなす（`has_live_chain_marker`・
/// `has_live_tainted_ref` と同一の判定。`let v = x.par_iter()
/// .collect::<Vec<_>>().par_iter();` のように collect の後に再び
/// 並列マーカーが現れる場合は、その後方のマーカー自身が「後続に同一
/// 連鎖上の collect を伴わない」ため汚染する。無関係な `Vec` collect
/// （`let it = (data.par_iter(), other.collect::<Vec<_>>()).0;` の
/// ようにタプル要素として同居するだけの collect）は同一連鎖上にない
/// ため遮断しない〈codex-review 指摘 P1〉）。
fn initializer_taints(init: &str, tainted: &HashSet<String>) -> bool {
    let tokens = tokenize(init);
    has_live_chain_marker(&tokens, &[]) || has_live_tainted_ref(&tokens, tainted)
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
    let (view, _virtual_markers) = statement_own_view(raw, tainted);
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

/// `text` 中で識別子 `ident` が識別子境界（前後が識別子文字でない）で
/// 一致する出現バイト位置を返す（`looks_like_fn_item` が「`fn` という
/// 語」の出現位置を探す専用。マーカー検出は `tokenize` ベースのトークン
/// 照合へ置き換え済みだが、こちらは単一キーワードの境界一致で足りる
/// 単純な用途のため、既存の text ベースの実装を維持する）。
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
        let (view, virtual_markers) = statement_own_view(raw, &tainted);

        // 同一文中の共起（メソッド呼び出し形・UFCS 形）と、過去の文
        // からの汚染識別子の使用は、いずれも「1 件」として数える
        // （相互排他的な `||` 集約のため二重計上しない）。
        if has_depth_aware_par_reduce(&view, &virtual_markers)
            || has_tainted_reduction_usage(&view, &tainted)
            || has_ufcs_reduction(&view, &tainted, &virtual_markers)
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

// =====================================================================
// `is_same_method_chain`（メソッド連鎖連続性）の回帰テスト
// （codex-review 指摘 P1: 無関係な Vec collect による誤遮断の是正）
// =====================================================================

#[test]
fn count_par_reduce_detects_tuple_sibling_unrelated_collect() {
    // タプル要素として同居するだけの無関係な collect は、別のタプル
    // 要素の par_iter の連鎖を遮断しない（codex P1 の逐語回帰例）。
    let src = "let it = (data.par_iter(), other.collect::<Vec<_>>()).0; \
               let sum = it.sum::<f32>();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_same_chain_collect_inside_tuple_element() {
    // 同一タプル要素内で par_iter → collect が同一連鎖上にある場合は
    // 従来どおり遮断する（codex P1 の逐語回帰例。無関係な collect の
    // 誤遮断是正が「連鎖上の collect」まで遮断しなくする regression
    // にならないことの確認）。
    let src = "let v = (a.par_iter().map(f).collect::<Vec<_>>(), 0).0; \
               v.iter().sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_does_not_flag_zip_argument_marker_blocked_by_outer_collect() {
    // 実ソース実測で判明した過検出パターンそのもの（`mse.rs::
    // mse_sum_sq_f32` の逐語形。`bce.rs`／`huber.rs`／`kl_div.rs` も
    // 同型）の固定回帰。`.zip(` の引数として現れる 2 つ目の
    // par_chunks（起点より深い位置にある）も、外側の collect によって
    // 同一連鎖として正しく遮断される必要がある（`is_same_method_chain`
    // の「到達点の深さを合流点とする」設計メモ参照）。クロージャ本体を
    // ブロック（`{ .. }`）にする（本クレートの実際のスタイル）ことが
    // 重要: 単一式クロージャ本体だと内側の `.fold(` がたまたま `.zip(`
    // 引数の par_chunks と同じ括弧深さになり、無関係な組と誤って
    // ペアリングされうる（ブロック本体は 1 段深くなるため、この
    // 偶然の深さ一致を避ける。既知の限界として §「文境界・
    // レキシカルスコープ・汚染追跡の方式」に注記）。
    let src = "fn f(pred: &[f32], target: &[f32]) -> f32 { \
               pred.par_chunks(4).zip(target.par_chunks(4)) \
               .map(|(p, t)| { p.iter().zip(t.iter()).fold(0.0, |a, (x, y)| a + (x - y)) }) \
               .collect::<Vec<f32>>().into_iter().fold(0.0, |a, v| a + v) }";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

// =====================================================================
// ブロック式に連鎖した縮約の回帰テスト
// （Cursor Bugbot 指摘: statement_own_view の塗りつぶしがブロック式
// 直後の連鎖を隠してしまい、同一文判定・UFCS 判定のどちらからも
// 検出できなかった不具合の再発防止）
// =====================================================================

#[test]
fn count_par_reduce_detects_reduction_chained_after_block_expression() {
    let src = "let s: f32 = { let x = 1; data.par_iter() }.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_reduction_chained_after_if_else_block_expression() {
    let src = "let s: f32 = if c { a.par_iter() } else { b.par_iter() }.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_reduction_chained_after_unsafe_block_expression() {
    let src = "unsafe { data.par_iter() }.sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_reduction_chained_after_match_block_expression() {
    let src = "match k { _ => data.par_iter() }.sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_reduction_chained_after_collected_block_expression() {
    // ブロック本体の中で par_chunks → collect が同一連鎖上で完結して
    // いる場合は、ブロックを「生きている並列マーカー」とみなさない
    // （仮想マーカーを記録しない）。
    let src = "let v = if c { data.par_chunks(4).map(f).collect::<Vec<_>>() } \
               else { vec![] }.into_iter().sum::<f32>();";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

// =====================================================================
// 仮想マーカー（長さに依存しないブロック式連鎖検出）の回帰テスト
// （敵対的レビュー指摘 Critical: 旧実装はテキストへ合成マーカー
// 文字列を実際に書き込む方式だったため、ブロック本体が合成マーカーの
// 最小長〈7 バイト〉未満だと検出できなかった。`{it}`・`{ acc }` の
// ような短い汚染識別子 1 つだけのブロック式が典型例。`statement_own_view`
// が返す仮想マーカー位置一覧〈テキストに何も書き込まない副チャネル〉
// へ置き換えたことで、ブロック本体の長さに一切依存しなくなったことを
// 固定する）
// =====================================================================

#[test]
fn count_par_reduce_detects_short_tainted_identifier_block_packed() {
    // 詰めた形 `{it}`／`{acc}`（4〜6 文字の識別子。旧実装の合成マーカー
    // 最小長 7 バイトに満たず検出漏れしていた）。
    let src = "let it = data.par_iter(); let acc = other.par_iter(); \
               let s: f32 = if flag { it } else { acc }.sum();";
    let packed = src.replace("{ it }", "{it}").replace("{ acc }", "{acc}");
    assert_eq!(count_par_reduce_cooccurrences(&packed), 1);
}

#[test]
fn count_par_reduce_detects_short_tainted_identifier_block_spaced() {
    // rustfmt が生成する `{ it }`（前後に空白を 1 つずつ挟む）形。
    let src = "let it = data.par_iter(); let acc = other.par_iter(); \
               let s: f32 = if flag { it } else { acc }.sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_short_tainted_identifier_in_match_arms() {
    let src = "let a = data.par_iter(); let b = other.par_iter(); \
               match k { 0 => {a}, _ => {b} }.sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_short_untainted_identifier_block() {
    let src = "{x}.sum()";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

// =====================================================================
// トークン照合方式の回帰テスト（敵対的レビュー指摘 codex P1 × 2・
// Cursor Bugbot Medium × 2。固定文字列の部分一致をやめ、簡易 lexer に
// よるトークン列の照合へ置き換えたことの固定）
// =====================================================================

#[test]
fn count_par_reduce_detects_spaced_method_calls() {
    // codex P1: 空白を挟むメソッド呼び出し（`.par_iter(`／`.sum(` の
    // 固定文字列一致では検出できなかった）。
    let src = "data.par_iter ().sum ()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_multiline_chain() {
    // codex P1: 改行を挟むメソッド連鎖。
    let src = "data\n    .par_iter()\n    .map(|x| x)\n    .sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_spaced_turbofish() {
    // codex P1: turbofish の空白入り形（`sum :: < f32 > ()`）。
    let src = "data.par_iter().sum :: < f32 > ()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_ufcs_into_par_iter() {
    // codex P1: UFCS 形の `into_par_iter`（固定文字列
    // `.into_par_iter(` では「直前が `.`」の形しか拾えず、任意の修飾
    // パスを経由する UFCS 呼び出しを見逃していた）。
    let src = "rayon::iter::IntoParallelIterator::into_par_iter(data).sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_detects_ufcs_sum_with_spaces() {
    let src = "ParallelIterator :: sum (data.par_iter())";
    assert_eq!(count_par_reduce_cooccurrences(src), 1);
}

#[test]
fn count_par_reduce_ignores_block_expression_collected_before_let_bound_usage() {
    // Bugbot Medium（過検出）: 仮想マーカーがブロック式直後の collect
    // による遮断を通り抜けていた（`is_same_method_chain` が `{` を
    // 連鎖の構成要素として扱っていなかったため）。
    let src = "let v = { data.par_iter() }.collect::<Vec<_>>(); v.iter().sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_ignores_block_expression_collected_in_same_statement() {
    let src = "{ data.par_iter() }.collect::<Vec<_>>().into_iter().sum::<f32>()";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_ignores_tainted_ref_only_used_in_collected_chain() {
    // Bugbot Medium（過検出）: 仮想マーカーの生存判定が、並列マーカー
    // には collect 遮断を適用する一方、汚染識別子にはしていなかった。
    // `it` が Vec collect された連鎖でしか使われていないブロックは
    // 「生きている」と判定してはならない。
    let src = "let it = data.par_iter(); \
               let s: f32 = { it.map(f).collect::<Vec<_>>() }.into_iter().sum();";
    assert_eq!(count_par_reduce_cooccurrences(src), 0);
}

#[test]
fn count_par_reduce_detects_raw_identifier_method_names() {
    // raw identifier（`r#sum`）で書いた縮約名・並列マーカーも通常の
    // 識別子と同じトークンとして照合する。
    assert_eq!(
        count_par_reduce_cooccurrences("fn f(data: &[f32]) -> f32 { data.par_iter().r#sum() }"),
        1
    );
    assert_eq!(
        count_par_reduce_cooccurrences(
            "fn f(data: &[f32]) -> f32 { data.r#par_iter().sum::<f32>() }"
        ),
        1
    );
}
