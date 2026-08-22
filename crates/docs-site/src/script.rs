//! docs サイトが出力する唯一の JS（イシュー #871）。
//!
//! # 役割・呼び出し文脈
//!
//! `crate::layout` は本モジュール追加以前は JS を 1 バイトも出力していなかった
//! （`layout.rs` モジュール冒頭「スコープ境界」コメント参照）。本モジュールは
//! テーマトグル（ダーク/ライト切替）と全文検索 UI の初期化を担う、
//! docs-site が出力する唯一の JS 資産。参照実装 `fandhe-backend`
//! `crates/docs-site/src/script.rs`（イシュー #390/#396 相当）からの移植。
//!
//! - [`INLINE_THEME_BOOTSTRAP`]: `crate::layout::docs_page` が `<head>` の
//!   先頭（stylesheet `<link>` より前）へ同期実行の `<script>` として埋め込む
//!   FOUC 抑止スニペット。`localStorage` に保存済みのテーマがあれば CSS 適用
//!   前に `<html data-theme="...">` を確定させる。
//! - [`SITE_JS`]: `crate::build::build_site` が [`SCRIPT_REL_PATH`]
//!   （`out` 起点）へ書き出す本体。`.docs-theme-toggle` ボタンのラベル・
//!   `aria-pressed` 更新、クリック時の切替・保存、`.docs-search` 全文検索
//!   （索引の遅延 `fetch`・部分一致検索・結果描画）、いずれも配線完了後に
//!   のみ `hidden` 属性を解除する（fail-closed）。
//!
//! # セキュリティ不変条件（`.claude/rules/security.md`・`.claude/rules/coding-rust.md`）
//!
//! `crate::html::Node::Text` は `<script>` の中身であっても必ず
//! `crate::html::render`（内部で `escape_text`）を経由する。`<script>` の
//! 中身は HTML パーサが実体参照を復号しない raw text であるため、エスケープ
//! 対象文字（`< > &`）を含む JS ソースを `Node::text` 経由で埋め込むと構文が
//! 壊れる。[`INLINE_THEME_BOOTSTRAP`]・[`SITE_JS`] は文字列リテラルに
//! バッククォート（テンプレートリテラル）のみを使い、`&&` の代わりに `||` を
//! 使うことでこれらの文字を一切含まない。[`is_escape_safe`] がこの性質を
//! 機械検証し、[`inline_theme_bootstrap`] は検証に落ちた場合 `None` を返す
//! fail-closed のアクセサとする（`html::Node` に生 HTML 注入用バリアントは
//! 存在しないため、検証落ち時は `<script>` 自体を出力しない選択のみ取れる。
//! `html.rs` モジュール冒頭の安全性契約参照）。
//!
//! `${`（テンプレートリテラル補間）も [`is_escape_safe`] の対象外文字列として
//! 禁止する。本モジュールの定数はすべて `&'static str` で外部入力・
//! `site/nav.toml` 由来の値を一切含まないが、将来の変数補間の混入を
//! テストで機械的にブロックする構造的な防御である。
//!
//! `localStorage` はスクリプトの実行主体（同一オリジンの他スクリプト・
//! 利用者自身）が改変できる非信頼データのため、[`INLINE_THEME_BOOTSTRAP`]・
//! [`SITE_JS`] のいずれも読み出した値を `dark`/`light` の allowlist と
//! 一致した場合のみ `data-theme` へ反映する。

/// [`SITE_JS`] の出力先（`out` 起点の相対パス）。
///
/// `crate::build::build_site` が本パスへ書き出し、`crate::layout::docs_page`
/// が `<script defer src>` で参照する単一実装点。
pub(crate) const SCRIPT_REL_PATH: &str = "assets/site.js";

/// テーマ選択を保存する `localStorage` キー。
///
/// GitHub Pages では本サイトが fandhe 系 docs サイト（`fandhe-ai.github.io`）
/// と同一オリジンになり得るため、`fandhe-docs-theme`・
/// `fandhe-backend-docs-theme`（他リポの同種実装が使うキー）と衝突しない
/// 専用キーにする。[`INLINE_THEME_BOOTSTRAP`] と [`SITE_JS`] の双方が同じ
/// キーを参照する契約であることを本モジュールの
/// `script_js_and_inline_bootstrap_share_the_same_storage_key`（`tests`）が
/// 固定する（キー名の二重管理ドリフト検知）。
// `#[allow(dead_code)]`: 本番経路（`INLINE_THEME_BOOTSTRAP`・`SITE_JS`）は
// キー文字列をそれぞれのテンプレートリテラル内に直書きしており、この定数を
// 実行時には参照しない（JS 定数はコンパイル時に確定した `&'static str`
// であり、`concat!` 等で定数を文字列リテラルへ埋め込む手段が Rust に無い
// ため）。本定数は `#[cfg(test)]` の
// `script_js_and_inline_bootstrap_share_the_same_storage_key` が両定数へ同じ
// キー名が書かれていることを機械検証するための「単一の真実源」として存在する
// （キー名の変更を片方だけ書き換える事故＝ドリフトを防ぐ）。通常ビルドでは
// テストが無効化されるため dead_code 警告が出るが、上記の理由により
// `#[allow]` する。
#[allow(dead_code)]
pub(crate) const THEME_STORAGE_KEY: &str = "rust-ai-library-docs-theme";

/// `<head>` の先頭付近（スタイルシートより前）に同期実行で埋め込む FOUC 抑止
/// スニペット。
///
/// `localStorage` から保存済みテーマを読み、`dark`/`light` のいずれかであれば
/// `<html>` の `data-theme` 属性を CSS 適用前に確定させる。`localStorage`
/// アクセス例外（Safari プライベートブラウズ等）は握りつぶし、失敗時は
/// `data-theme` 未設定のまま（`assets/site.css` の
/// `@media (prefers-color-scheme: dark)` 経路）へ退避する。
///
/// 責務はここまで（属性設定のみ）。ボタンのイベント配線・ラベル更新はすべて
/// [`SITE_JS`] 側が担う（`site.js` の読み込み失敗時にもこのスニペットだけは
/// 動作し、保存済みテーマの反映は維持される）。
pub(crate) const INLINE_THEME_BOOTSTRAP: &str = "try{var t=localStorage.getItem(`rust-ai-library-docs-theme`);if(t===`dark`||t===`light`){document.documentElement.setAttribute(`data-theme`,t);}}catch(e){}";

/// [`SCRIPT_REL_PATH`] へ書き出す `assets/site.js` の全量。
///
/// 責務（テーマトグル）:
///
/// 1. `document.readyState === "loading"` なら `DOMContentLoaded` を待ち、
///    そうでなければ即座に配線を実行する。
/// 2. `init()` 内で `.docs-theme-toggle` ボタンを取得する（無ければ即
///    return。docs-site 以外のページ・将来の骨格変更で要素が消えても例外を
///    投げない防御的実装）。
/// 3. 実効テーマを解決する: `<html data-theme>` 属性値（`dark`/`light` のみ
///    採用） → 無ければ `matchMedia("(prefers-color-scheme: dark)")`。
/// 4. ボタンのラベル・`aria-pressed` を実効テーマに合わせて初期化する
///    （この時点では `data-theme` を書き込まない。利用者が未選択なら OS 設定
///    追従のままにする）。
/// 5. `click` で実効テーマの反対側へ切替 → `localStorage` へ保存（例外は
///    握りつぶす） → `data-theme` 属性を更新 → ラベル更新。
/// 6. **すべての配線が完了した後にのみ** `hidden` 属性を解除する（`site.js`
///    の読み込み失敗時に「押しても何も起きないボタン」を残さないため）。
///
/// 責務（全文検索）:
///
/// 1. `.docs-search` / `.docs-search-input` / `#docs-search-results` を
///    取得する（無ければ即 return）。
/// 2. 索引 URL は `data-search-index` 属性から読む（空なら return）。
/// 3. 索引は初回 `focus` または初回 `input` のいずれか早い方で 1 度だけ
///    `fetch` する（`loading` フラグで多重取得を抑止）。取得失敗は `loadFailed`
///    フラグで終端失敗状態を保持し、以降の `input` イベントでは無条件の
///    再 `fetch` を行わない（404・ネットワークエラー後のキー入力のたびの
///    retry storm 防止）。`res.json()` の結果は `isValidSearchIndex` で
///    `version`・`base_path`・`pages` 配列・各要素の `href`/`title`/`text`
///    型を検証してから採用する（不正データを無検証で `indexData` へ入れると
///    `pages.forEach` 等が TypeError で検索 UI を壊すため。検証失敗も
///    `loadFailed` の終端失敗状態へ fail-closed で合流する）。
/// 4. クエリは小文字化して部分一致判定する。スコアはタイトル一致 + 3 /
///    本文一致 + 1 の決定的加算とし、0 点のページは除外した上でスコア降順に
///    並べ替え、上位 `SEARCH_MAX_RESULTS`（8 件）のみを描画する。
/// 5. 結果の描画は `document.createElement` + `textContent` +
///    `setAttribute` のみで行う（`innerHTML` 等は使わない）。href は
///    `isSafeHref` で検証したもののみ描画する: `/` から始まり、バック
///    スラッシュを含まず、`new URL(href, location.origin)` で
///    解決した結果の `origin` が `location.origin` と一致するもの（同一
///    オリジンの相対パス）のみを同一オリジン制約として受理する多層防御を
///    行う。バックスラッシュを明示拒否するのは、WHATWG の special URL 仕様が
///    パース時にバックスラッシュをスラッシュ相当として扱うため、`/` 始まり
///    チェックだけでは `/\evil.example/` のような値が `//evil.example/`
///    （外部オリジン）へ正規化され得るため（origin 一致検証はこの正規化ゆれ
///    を個別に塞ぐのではなく解決結果そのものを検証する最終防御）。
/// 6. `Escape` キーで入力・結果をクリアする。
/// 7. **すべての配線が完了した後にのみ** `.docs-search` の `hidden` を
///    解除する（テーマトグルと同じ fail-closed パターン）。
///
/// 文字列リテラルはすべてバッククォート（テンプレートリテラル。補間は使わ
/// ない）を使い、`&&` の代わりに `||` またはネストした `if` を、比較演算子
/// `<`/`>` の代わりに `!==`/`===`/`indexOf(...) !== -1`/sort コンパレータを
/// 使うことでエスケープ対象文字（`< > &`）を含まない（[`is_escape_safe`]
/// 参照）。`innerHTML` / `insertAdjacentHTML` / `document.write` / `eval` /
/// `new Function` は使わない（DOM 操作は
/// `setAttribute`/`removeAttribute`/`textContent`/`createElement`/
/// `appendChild`/`addEventListener` に限定する）。
pub(crate) const SITE_JS: &str = "\
(function () {
  var STORAGE_KEY = `rust-ai-library-docs-theme`;
  var SEARCH_MAX_RESULTS = 8;
  var toggle;

  function effectiveTheme() {
    var attr = document.documentElement.getAttribute(`data-theme`);
    if (attr === `dark` || attr === `light`) {
      return attr;
    }
    var prefersDark = false;
    if (window.matchMedia) {
      prefersDark = window.matchMedia(`(prefers-color-scheme: dark)`).matches;
    }
    return prefersDark ? `dark` : `light`;
  }

  function applyLabel(theme) {
    toggle.setAttribute(`aria-pressed`, theme === `dark` ? `true` : `false`);
    toggle.textContent = theme === `dark` ? `Light` : `Dark`;
  }

  function storeTheme(theme) {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch (err) {
      // localStorage が使えない環境（Safari プライベートブラウズ等）では
      // 保存をあきらめ、今回の切替自体は続行する。
    }
  }

  function init() {
    toggle = document.querySelector(`.docs-theme-toggle`);
    if (!toggle) {
      return;
    }
    applyLabel(effectiveTheme());
    toggle.addEventListener(`click`, function () {
      var next = effectiveTheme() === `dark` ? `light` : `dark`;
      storeTheme(next);
      document.documentElement.setAttribute(`data-theme`, next);
      applyLabel(next);
    });
    // 配線がすべて完了した後にのみ可視化する（上記 doc コメント手順 6）。
    toggle.removeAttribute(`hidden`);
  }

  function initSearch() {
    var container = document.querySelector(`.docs-search`);
    if (!container) {
      return;
    }
    var input = document.querySelector(`.docs-search-input`);
    if (!input) {
      return;
    }
    var results = document.querySelector(`#docs-search-results`);
    if (!results) {
      return;
    }
    var indexUrl = input.getAttribute(`data-search-index`);
    if (!indexUrl) {
      return;
    }

    var indexData = null;
    var loading = false;
    var loadFailed = false;

    function loadIndex() {
      if (indexData) {
        return;
      }
      if (loading) {
        return;
      }
      if (loadFailed) {
        // 直前の fetch が失敗して終端状態に入っている。404 やネットワーク
        // エラー後の再入力のたびに再試行し続けるのを避ける
        // （キー入力ごとの無条件リトライ防止）。
        return;
      }
      loading = true;
      fetch(indexUrl)
        .then(function (res) {
          if (!res.ok) {
            throw new Error(`search index fetch failed`);
          }
          return res.json();
        })
        .then(function (data) {
          if (!isValidSearchIndex(data)) {
            // スキーマ不一致（404 が JSON を返す・キャッシュ破損・
            // version 不一致等）を無検証で pages.forEach へ渡すと
            // TypeError で検索 UI が停止する。loadFailed の終端失敗状態へ
            // fail-closed で誘導し、既存の fetch 失敗経路と同じ扱いにする。
            loading = false;
            loadFailed = true;
            return;
          }
          indexData = data;
          loading = false;
          renderResults(input.value);
        })
        .catch(function () {
          // 索引取得に失敗しても検索 UI 自体は壊さず、結果 0 件のまま
          // フォールバックする。loadFailed を立てて以降の input イベントでの
          // 無条件再試行を止める終端失敗状態とする。
          loading = false;
          loadFailed = true;
        });
    }

    function clearResults() {
      while (results.firstChild) {
        results.removeChild(results.firstChild);
      }
    }

    function isSafeHref(href) {
      if (typeof href !== `string`) {
        return false;
      }
      if (href.indexOf(`/`) !== 0) {
        return false;
      }
      // バックスラッシュを含む href を明示的に拒否する。WHATWG の special
      // URL 仕様ではパース時にバックスラッシュがスラッシュ相当として扱わ
      // れるため、`/` 始まりチェックだけでは（`//` 非開始チェックを併用
      // しても）バックスラッシュを使った値が `//evil.example/` 相当へ
      // 正規化され、意図した同一オリジン制約を迂回できてしまう。
      if (href.indexOf(`\\\\`) !== -1) {
        return false;
      }
      var resolved;
      try {
        resolved = new URL(href, location.origin);
      } catch (e) {
        return false;
      }
      // 上記の事前拒否に加え、実際に URL を解決したうえで origin が
      // 一致することまで検証する（同一オリジン制約の最終的な担保。
      // 正規化の抜け穴を個別に塞ぎ続けるのではなく、解決結果そのものを
      // 検証することで fail-closed にする）。
      if (resolved.origin !== location.origin) {
        return false;
      }
      return true;
    }

    function isValidSearchIndex(data) {
      if (typeof data !== `object` || data === null) {
        return false;
      }
      if (data.version !== 1) {
        return false;
      }
      if (typeof data.base_path !== `string`) {
        return false;
      }
      if (!Array.isArray(data.pages)) {
        return false;
      }
      for (var i = 0; i !== data.pages.length; i++) {
        var page = data.pages[i];
        if (typeof page !== `object` || page === null) {
          return false;
        }
        if (typeof page.href !== `string`) {
          return false;
        }
        if (typeof page.title !== `string`) {
          return false;
        }
        if (typeof page.text !== `string`) {
          return false;
        }
      }
      return true;
    }

    function scorePage(page, query) {
      var score = 0;
      if (page.title.toLowerCase().indexOf(query) !== -1) {
        score = score + 3;
      }
      if (page.text.toLowerCase().indexOf(query) !== -1) {
        score = score + 1;
      }
      return score;
    }

    function renderResults(rawQuery) {
      var query = rawQuery.toLowerCase();
      clearResults();
      if (query.length === 0) {
        results.setAttribute(`hidden`, ``);
        return;
      }
      if (!indexData) {
        results.setAttribute(`hidden`, ``);
        return;
      }
      var matches = [];
      indexData.pages.forEach(function (page) {
        var score = scorePage(page, query);
        if (score !== 0) {
          matches.push({ page: page, score: score });
        }
      });
      matches.sort(function (a, b) {
        return b.score - a.score;
      });
      var top = matches.slice(0, SEARCH_MAX_RESULTS);
      if (top.length === 0) {
        results.setAttribute(`hidden`, ``);
        return;
      }
      var list = document.createElement(`ul`);
      top.forEach(function (entry) {
        var page = entry.page;
        if (!isSafeHref(page.href)) {
          return;
        }
        var item = document.createElement(`li`);
        var link = document.createElement(`a`);
        link.setAttribute(`href`, page.href);
        link.textContent = page.title;
        item.appendChild(link);
        list.appendChild(item);
      });
      results.appendChild(list);
      results.removeAttribute(`hidden`);
    }

    input.addEventListener(`focus`, loadIndex);
    input.addEventListener(`input`, function () {
      loadIndex();
      renderResults(input.value);
    });
    input.addEventListener(`keydown`, function (event) {
      if (event.key === `Escape`) {
        input.value = ``;
        clearResults();
        results.setAttribute(`hidden`, ``);
      }
    });

    // 配線がすべて完了した後にのみ可視化する（上記 doc コメント手順 7）。
    container.removeAttribute(`hidden`);
  }

  function ready() {
    init();
    initSearch();
  }

  if (document.readyState === `loading`) {
    document.addEventListener(`DOMContentLoaded`, ready);
  } else {
    ready();
  }
})();
";

/// `source` が HTML エスケープ対象文字（`< > &`）を 1 文字も含まず、かつ
/// テンプレートリテラル補間（`${`）を含まないかを判定する純関数。
///
/// `crate::html::render` の内部 `escape_text`（`&` `<` `>` の 3 種類）と
/// 対象文字を完全一致させることで、`<script>` の中身（HTML パーサが実体
/// 参照を復号しない raw text）に埋め込んでも構文が壊れないことを保証する。
/// `${` の禁止は、将来変数補間を追加しようとした際にこのテストが機械的に
/// 検知するための構造的な防御である（変数補間は非信頼データを script
/// コンテキストへ注入する経路になり得るため、docs-site では導入しない方針）。
pub(crate) fn is_escape_safe(source: &str) -> bool {
    !source.chars().any(|c| matches!(c, '<' | '>' | '&')) && !source.contains("${")
}

/// [`INLINE_THEME_BOOTSTRAP`] が [`is_escape_safe`] を満たす場合のみ `Some`
/// を返す fail-closed のアクセサ。
///
/// `crate::layout::docs_page` はこの関数が `None` を返した場合 `<script>`
/// 自体を出力しない（壊れた JS を配信するくらいなら `prefers-color-scheme`
/// 追従へ退避する）。
pub(crate) fn inline_theme_bootstrap() -> Option<&'static str> {
    if is_escape_safe(INLINE_THEME_BOOTSTRAP) {
        Some(INLINE_THEME_BOOTSTRAP)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_theme_bootstrap_is_escape_safe() {
        assert!(is_escape_safe(INLINE_THEME_BOOTSTRAP));
    }

    #[test]
    fn site_js_is_escape_safe() {
        assert!(is_escape_safe(SITE_JS));
    }

    #[test]
    fn inline_theme_bootstrap_accessor_returns_some_for_the_safe_constant() {
        assert_eq!(inline_theme_bootstrap(), Some(INLINE_THEME_BOOTSTRAP));
    }

    #[test]
    fn is_escape_safe_rejects_html_escape_target_characters() {
        assert!(!is_escape_safe("a<b"));
        assert!(!is_escape_safe("a>b"));
        assert!(!is_escape_safe("a&b"));
    }

    #[test]
    fn is_escape_safe_rejects_template_literal_interpolation() {
        assert!(!is_escape_safe("var x = `${y}`;"));
    }

    #[test]
    fn is_escape_safe_accepts_plain_js_without_interpolation() {
        assert!(is_escape_safe(
            "(function () { var x = `plain`; return x; })();"
        ));
    }

    /// キー名の二重管理ドリフト検知: [`INLINE_THEME_BOOTSTRAP`] と
    /// [`SITE_JS`] の双方が [`THEME_STORAGE_KEY`] と同じ文字列を参照する
    /// ことを固定する（片方だけキー名を変更してリロード後の復元が壊れる
    /// 事故を防ぐ）。
    #[test]
    fn script_js_and_inline_bootstrap_share_the_same_storage_key() {
        assert!(INLINE_THEME_BOOTSTRAP.contains(THEME_STORAGE_KEY));
        assert!(SITE_JS.contains(THEME_STORAGE_KEY));
    }

    /// `localStorage` アクセスの例外握りつぶし（try/catch）が消えていないこと
    /// を固定する。Safari プライベートブラウズ等での例外時にスクリプト全体が
    /// 停止し、既存機能まで壊れる回帰を防ぐ回帰テスト。
    #[test]
    fn inline_theme_bootstrap_swallows_localstorage_exceptions() {
        assert!(INLINE_THEME_BOOTSTRAP.contains("try{"));
        assert!(INLINE_THEME_BOOTSTRAP.contains("catch"));
    }

    #[test]
    fn site_js_swallows_localstorage_exceptions() {
        assert!(SITE_JS.contains("try {"));
        assert!(SITE_JS.contains("catch"));
    }

    /// [`SITE_JS`] は `hidden` の解除をイベント配線完了後にのみ行う（上記
    /// doc コメント手順 6）。`removeAttribute` 呼び出しが `init` 関数の最後
    /// （`addEventListener` の後）に位置することを、文字列上の出現順で固定する。
    #[test]
    fn site_js_reveals_toggle_only_after_click_handler_is_wired() {
        let listener_pos = SITE_JS
            .find("addEventListener")
            .expect("SITE_JS should wire a click handler");
        let reveal_pos = SITE_JS
            .find("removeAttribute(`hidden`)")
            .expect("SITE_JS should reveal the toggle by removing the hidden attribute");
        assert!(
            listener_pos < reveal_pos,
            "hidden の解除はイベント配線より後である必要がある"
        );
    }

    /// `.docs-theme-toggle` の `querySelector` 呼び出しが `init` 関数の中
    /// （`readyState` 分岐より後）に位置することを固定する。トップレベルで
    /// 即時実行してしまうと、`DOMContentLoaded` 待ちフォールバックが要素取得前
    /// に済んだ null 判定によって意味を持たなくなる。
    #[test]
    fn site_js_queries_toggle_element_inside_init_not_at_top_level() {
        let ready_state_check_pos = SITE_JS
            .find("document.readyState")
            .expect("SITE_JS should branch on document.readyState");
        let query_selector_pos = SITE_JS
            .find("document.querySelector(`.docs-theme-toggle`)")
            .expect("SITE_JS should query the toggle element");
        assert!(
            query_selector_pos < ready_state_check_pos,
            "querySelector の呼び出しは init 関数定義内（readyState 分岐より前のソース位置）にある必要がある"
        );

        let init_fn_pos = SITE_JS
            .find("function init()")
            .expect("SITE_JS should define an init function");
        assert!(
            init_fn_pos < query_selector_pos,
            "querySelector の呼び出しは init 関数の中に位置する必要がある"
        );
    }

    /// [`SITE_JS`] は危険な DOM 操作 API（`innerHTML`/`insertAdjacentHTML`/
    /// `document.write`/`eval`/`new Function`）を使わない（OWASP A03）。
    #[test]
    fn site_js_does_not_use_dangerous_dom_apis() {
        for needle in [
            "innerHTML",
            "insertAdjacentHTML",
            "document.write",
            "eval(",
            "new Function",
        ] {
            assert!(!SITE_JS.contains(needle), "SITE_JS should not use {needle}");
        }
    }

    /// [`SITE_JS`] が検索入力欄（`.docs-search-input`）と索引 URL 属性
    /// （`data-search-index`）を参照することを固定する。
    #[test]
    fn site_js_references_search_input_and_index_attribute() {
        assert!(SITE_JS.contains(".docs-search-input"));
        assert!(SITE_JS.contains("data-search-index"));
        assert!(SITE_JS.contains("#docs-search-results"));
    }

    /// 検索索引の `fetch` 失敗をサイレントにフォールバックする `catch` が
    /// 存在することを固定する（索引取得失敗時も UI を壊さない契約の回帰テスト）。
    #[test]
    fn site_js_search_fetch_has_a_silent_catch_fallback() {
        let fetch_pos = SITE_JS
            .find("fetch(indexUrl)")
            .expect("SITE_JS should fetch the search index");
        let catch_pos = SITE_JS
            .find(".catch(function ()")
            .expect("SITE_JS should swallow search index fetch failures");
        assert!(
            fetch_pos < catch_pos,
            "catch は fetch(indexUrl) より後に位置する必要がある"
        );
    }

    /// [`SITE_JS`] は検索 UI（`.docs-search`）の `hidden` 解除を、入力欄への
    /// イベント配線がすべて完了した後にのみ行う（テーマトグルと同じ
    /// fail-closed パターン、上記 doc コメント手順 7）。
    #[test]
    fn site_js_reveals_search_ui_only_after_wiring_is_complete() {
        let keydown_listener_pos = SITE_JS
            .find("input.addEventListener(`keydown`")
            .expect("SITE_JS should wire a keydown handler on the search input");
        let reveal_pos = SITE_JS
            .find("container.removeAttribute(`hidden`)")
            .expect("SITE_JS should reveal the search UI by removing the hidden attribute");
        assert!(
            keydown_listener_pos < reveal_pos,
            "検索 UI の hidden 解除はイベント配線より後である必要がある"
        );
    }

    /// 検索結果の href 検証（`isSafeHref`）が `/` 始まりの事前チェックに加え、
    /// `new URL` で解決した結果の `origin` が `location.origin` と一致する
    /// ことまで検証することを固定する（同一オリジン制約の最終担保）。
    #[test]
    fn site_js_search_validates_result_hrefs_before_rendering() {
        assert!(SITE_JS.contains("function isSafeHref(href)"));
        assert!(SITE_JS.contains("href.indexOf(`/`) !== 0"));
        assert!(SITE_JS.contains("new URL(href, location.origin)"));
        assert!(SITE_JS.contains("resolved.origin !== location.origin"));
    }

    /// `isSafeHref` がバックスラッシュを含む href を明示的に拒否することを
    /// 固定する（WHATWG special URL のバックスラッシュ＝スラッシュ扱い
    /// 正規化を悪用した `//evil.example/` 相当への迂回の回帰防止。
    /// OWASP A03/A01 系の外部入力 URL 検証欠陥）。
    #[test]
    fn site_js_is_safe_href_rejects_backslash() {
        assert!(SITE_JS.contains(r"href.indexOf(`\\`) !== -1"));
    }

    /// 取得した検索索引（`res.json()`）を無検証で `indexData` へ保存せず、
    /// `version`・`base_path`・`pages` 配列・各要素の `href`/`title`/`text`
    /// 型を検証してから採用することを固定する。不正データ（404 が JSON を
    /// 返す・キャッシュ破損等）で `pages.forEach` が TypeError を起こし
    /// 検索 UI が停止する回帰を防ぐ。
    #[test]
    fn site_js_validates_search_index_schema_before_use() {
        assert!(SITE_JS.contains("function isValidSearchIndex(data)"));
        assert!(SITE_JS.contains("if (!isValidSearchIndex(data))"));
        assert!(SITE_JS.contains("data.version !== 1"));
        assert!(SITE_JS.contains("Array.isArray(data.pages)"));
    }

    /// 検索索引の `fetch` が失敗した場合、`loadFailed` という終端失敗状態が
    /// `catch` 内で立てられ、`loadIndex` の先頭（`fetch` 呼び出しより前）で
    /// その状態を見て早期リターンすることを固定する。この構造がないと、404 や
    /// ネットワークエラー後に検索ボックスへの `input` イベントのたびに無条件で
    /// `fetch` が再試行されてしまう。
    #[test]
    fn site_js_search_fetch_failure_sets_terminal_state_to_avoid_retry_storm() {
        let load_index_pos = SITE_JS
            .find("function loadIndex()")
            .expect("SITE_JS should define loadIndex");
        let load_failed_check_pos = SITE_JS
            .find("if (loadFailed)")
            .expect("SITE_JS should short-circuit loadIndex once a fetch failure is recorded");
        let fetch_pos = SITE_JS
            .find("fetch(indexUrl)")
            .expect("SITE_JS should fetch the search index");
        let catch_pos = SITE_JS
            .find(".catch(function ()")
            .expect("SITE_JS should swallow search index fetch failures");
        let load_failed_set_pos = SITE_JS
            .rfind("loadFailed = true;")
            .expect("SITE_JS should record a terminal failure state on fetch error");

        assert!(
            load_index_pos < load_failed_check_pos,
            "loadFailed の判定は loadIndex 関数の中に位置する必要がある"
        );
        assert!(
            load_failed_check_pos < fetch_pos,
            "loadFailed の早期リターンは fetch 呼び出しより前に位置する必要がある"
        );
        assert!(
            catch_pos < load_failed_set_pos,
            "loadFailed = true の代入は catch ハンドラの中に位置する必要がある"
        );
    }
}
