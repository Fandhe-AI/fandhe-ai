//! `docs-site`: GitHub Pages 公開ツリー（イシュー #865 Phase 1）向けの
//! 開発者・CI 専用 SSG（静的サイトジェネレータ）クレート。
//!
//! # 位置づけ
//!
//! 本クレートは `tensor-core`・`autodiff`・`backend-*`・`facade` 等の本体ライブラリの
//! 公開 API とは無関係のドキュメントビルドツールであり、workspace member ではあるが
//! `publish = false`（`Cargo.toml` の workspace 継承）。開発者のローカル環境・CI から
//! `cargo run -p docs-site -- --root <repo_root> --out <dir>` の形で起動される
//! （`main.rs` の CLI 骨格を参照）。
//!
//! # クレート構成（イシュー #870 時点）
//!
//! - [`nav`][]: `site/nav.toml`（サイトタイトル・セクション・ページ構成マニフェスト）の
//!   fail-closed TOML サブセットパーサーとデータモデル。ファイルシステムに依存しない
//!   純関数 [`nav::parse_nav`] と、`page.source` の実ファイル存在検証を分離した
//!   [`nav::validate_sources`] からなる（`section.index_path` は #870 で追加）
//! - [`html`][]: 最小 HTML ノード層（`Node` enum + 既定エスケープ `render`）。
//!   `markdown`・`layout` の両方が最終的な HTML 文字列化をここへ集約する
//! - [`markdown`][]: 自作 Markdown → `html::Node` 変換（外部クレート非依存）
//! - [`layout`][]: ページ骨格（ヘッダ・サイドバー・本文）の組み立て。
//!   イシュー #871 でテーマトグルボタン・検索 UI・`<head>` への FOUC 抑止
//!   スクリプト埋め込みを追加した
//! - [`theme`][]: ビルド時に埋め込むテーマ CSS 定数（`assets/site.css`）
//! - `search`: 全文検索インデックス生成（sans-I/O な純関数群。イシュー
//!   #871）。ページ本文の `html::Node` 木から抽出したプレーンテキストを
//!   `assets/search-index.json` へ決定的直列化する
//! - `script`: テーマ切替・全文検索のクライアント側 JS 定数（イシュー
//!   #871）。`build` が `assets/site.js` へ書き出し、`layout` が `<script>`
//!   経由で埋め込み・参照する
//! - [`build`][]: 上記を結線したビルドパイプライン（`nav.toml` 読み込み →
//!   パース → 検証 → 各ページの Markdown→HTML 変換・検索索引収集 → `<out>`
//!   への書き出し）
//!
//! # 参照実装との関係
//!
//! 参照実装 `fandhe-backend`（`crates/docs-site`。外部依存ゼロの TOML
//! サブセットパーサー・行番号付きエラー・入力サイズ上限・`unwrap()` 不使用の
//! 設計方針）を参照するが、本リポジトリは `fandhe-frontend` 系クレートに依存できない
//! （deps-policy.md の許容 9 区分外・ユーザー承認必須のため追加しない）ため、
//! `html::Node` を自作の最小 HTML ノード層として実装する。参照実装が持つ
//! 生 HTML 注入用バリアント（`RawHtml` 相当）はあえて設けず、エスケープを
//! 構造的に迂回できない設計にしている（`html.rs` モジュールコメント参照）。
//!
//! # unsafe の使用範囲
//!
//! 本クレートは既定でクレート全体の `unsafe` を禁止する
//! （`.claude/rules/coding-rust.md` の unsafe 最小化方針・`.claude/rules/security.md`）。
//! 唯一の例外が `build` モジュール内の非公開 `fd_walk` モジュール（unix 版）:
//! `page.source`／出力先パスへの
//! アクセスを fd 相対（`openat`/`mkdirat`/`renameat`/`unlinkat`、いずれも
//! `O_NOFOLLOW`）で行う FFI 境界であり、`canonicalize` 等のパス文字列ベースの
//! 検証では防げない中間ディレクトリのシンボリックリンク差し替え TOCTOU
//! （codex-review 指摘・PR #899 追加ラウンド、P0 x2）を閉じるために新設した。
//! `std` は `openat` 相当の fd 相対 API を公開していないため、この境界だけは
//! `unsafe` な `extern "C"` 宣言（追加 Cargo 依存なし。`std` が既にリンクする
//! system libc の安定 C ABI 関数を直接呼ぶ）が必要になる。よって
//! `forbid(unsafe_code)` から `deny(unsafe_code)` へ緩め、`fd_walk` モジュール
//! （unix 版）にのみ `#[allow(unsafe_code)]` を局所付与する。それ以外の
//! モジュール・関数では `unsafe` は引き続き `deny` により拒否される。
#![deny(unsafe_code)]

pub mod build;
pub mod html;
pub mod layout;
pub mod markdown;
pub mod nav;
pub(crate) mod script;
pub(crate) mod search;
pub mod theme;
