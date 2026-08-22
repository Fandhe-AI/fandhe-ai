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
//! # クレート構成（イシュー #869 時点）
//!
//! - [`nav`][]: `site/nav.toml`（サイトタイトル・セクション・ページ構成マニフェスト）の
//!   fail-closed TOML サブセットパーサーとデータモデル。ファイルシステムに依存しない
//!   純関数 [`nav::parse_nav`] と、`page.source` の実ファイル存在検証を分離した
//!   [`nav::validate_sources`] からなる
//! - [`build`][]: [`nav`] を用いたビルドパイプラインの枠組み（`nav.toml` 読み込み →
//!   パース → 検証 → 出力ディレクトリ作成）。Markdown→HTML 変換・layout・テーマ CSS は
//!   兄弟イシュー #870 で `build::build_site` の拡張点として追加される
//!
//! # 参照実装との関係
//!
//! 参照実装 `fandhe-backend`（`crates/docs-site`。外部依存ゼロの TOML
//! サブセットパーサー・行番号付きエラー・入力サイズ上限・`unwrap()` 不使用の
//! 設計方針）を参照するが、本リポジトリは `fandhe-frontend` 系クレートに依存できない
//! （deps-policy.md の許容 9 区分外・ユーザー承認必須のため追加しない）ため、
//! HTML ノード生成を含まない Node レンダラ非依存の純データモデルとして実装する
//! （sidebar 等の HTML 生成は #870 で自作 layout モジュールが担う）。
//!
//! # unsafe 不使用
//!
//! 本クレートは FFI 境界を持たないため、クレート全体で `unsafe` を禁止する
//! （`.claude/rules/coding-rust.md` の unsafe 最小化方針・`.claude/rules/security.md`）。
#![forbid(unsafe_code)]

pub mod build;
pub mod nav;
