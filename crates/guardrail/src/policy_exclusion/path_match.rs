//! ポリシー除外リスト（`policy_exclusion`）向けの自作パス glob マッチャ。
//!
//! v1 資産 `policy-exclusion.toml` の `any_diff_in_paths` 方式は `glob` クレート
//! （`.claude/rules/deps-policy.md` 許容依存 8 区分に非該当）を用いていたが、本
//! クレートは新規依存を自己判断で追加しない方針（`delegation-impl.md`）のため、
//! 対応構文をリテラル・`?`・`*`（`/` を跨がない）・単独セグメント `**`
//! （0 個以上のディレクトリ）に限定した最小マッチャを自作する（イシュー #122・
//! `docs/spec/05-tasks.md` TASK-5.2a 計画 4 節）。
//!
//! 未対応構文（文字クラス `[...]`・ブレース展開 `{...}` 等）は構築時
//! （[`PathPattern::compile`]）に型付きエラーで拒否し、match 時に黙って
//! `false` へ倒す fail-closed 経路を作らない（`.claude/rules/security.md` A08。
//! 「発火すべき除外ルールが発火せず自動適用へ倒れる」迂回を防ぐ）。
//!
//! マッチングは動的計画法（2 段の DP テーブル）で行い、指数的バックトラック
//! を構造的に排除する（`*a*a*a...` 等の病的パターンでも入力長の多項式時間で
//! 終了する。計画 4 節「マッチャは反復で実装」）。

use std::fmt;

/// パターン検証・照合で共通に使う型付きエラー。
///
/// クレート横断のエラー統合（`crate::error::GuardrailError`）には合流させず
/// `policy_exclusion` モジュール内で完結させる（計画 4 節 5 番・#124 で
/// `PolicyExclusionError` → `GuardrailError` の変換経路を整備する想定）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternError {
    pub pattern: String,
    pub reason: &'static str,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid path pattern '{}': {}",
            self.pattern, self.reason
        )
    }
}

impl std::error::Error for PatternError {}

/// 1 セグメント（`/` で区切った 1 階層分）のパターン表現。
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    /// `**`（0 個以上のディレクトリ階層に一致する単独セグメント）。
    DoubleStar,
    /// リテラル・`?`・`*` からなる通常セグメント（`/` は含まない）。
    Literal(String),
}

/// 構築時に構文検証を通過したパスパターン。
///
/// 非公開フィールド＋アクセサ経由の構築のみを許し、検証を経ない `PathPattern`
/// を型システム上作れないようにする（v1 の不変条件保持パターン踏襲。計画 4 節 3 番）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathPattern {
    raw: String,
    segments: Vec<Segment>,
}

impl PathPattern {
    /// パターン文字列を検証つきでコンパイルする。
    ///
    /// 拒否する入力（fail-closed。空文字列で「常に match しない」曖昧な
    /// パターンを許さない）:
    /// - 空文字列
    /// - `/` 始まり（絶対パス）
    /// - `..` セグメント（リポジトリルート外参照）
    /// - 制御文字を含む
    /// - 未対応 glob 構文（`[`・`]`・`{`・`}`・`!`・`\`）
    /// - `**` がセグメント全体でなく他の文字と同居している（例: `a**b`）
    pub fn compile(raw: &str) -> Result<Self, PatternError> {
        if raw.is_empty() {
            return Err(PatternError {
                pattern: raw.to_string(),
                reason: "pattern must not be empty",
            });
        }
        if raw.starts_with('/') {
            return Err(PatternError {
                pattern: raw.to_string(),
                reason: "absolute paths are not allowed (must be repo-relative)",
            });
        }
        if raw.chars().any(|c| c.is_control()) {
            return Err(PatternError {
                pattern: raw.to_string(),
                reason: "control characters are not allowed",
            });
        }
        const UNSUPPORTED: &[char] = &['[', ']', '{', '}', '!', '\\'];
        if raw.chars().any(|c| UNSUPPORTED.contains(&c)) {
            return Err(PatternError {
                pattern: raw.to_string(),
                reason: "unsupported glob syntax (only literal, '?', '*', and standalone '**' are supported)",
            });
        }

        let mut segments = Vec::new();
        for raw_seg in raw.split('/') {
            if raw_seg.is_empty() {
                return Err(PatternError {
                    pattern: raw.to_string(),
                    reason: "empty path segment (e.g. consecutive '/' or trailing '/')",
                });
            }
            if raw_seg == ".." {
                return Err(PatternError {
                    pattern: raw.to_string(),
                    reason: "'..' segments are not allowed",
                });
            }
            if raw_seg == "**" {
                segments.push(Segment::DoubleStar);
                continue;
            }
            if raw_seg.contains("**") {
                return Err(PatternError {
                    pattern: raw.to_string(),
                    reason: "'**' must be a standalone path segment",
                });
            }
            segments.push(Segment::Literal(raw_seg.to_string()));
        }

        Ok(PathPattern {
            raw: raw.to_string(),
            segments,
        })
    }

    /// 検証済みパターンの原文（レポート・ログ出力用）。
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// リポジトリルート相対パス `path` がこのパターンに一致するか判定する。
    ///
    /// `path` はマッチャ内部では検証しない（呼び出し側 = `any_diff_in_paths`
    /// が変更ファイル一覧をそのまま渡す設計。パターン側のみ構築時検証で
    /// fail-closed を担保する）。呼び出し元は `path` がリポジトリルート相対の
    /// 正規化済み（先頭 `/` なし・`./` プレフィックスなし・`\` を含まない）
    /// パスであることを保証する契約を負う（git diff の quoted/octal-escaped
    /// パス等、非正規化パスは黙って不一致になりうる。正規化は変更ファイル
    /// 一覧の取得元である #103／#124 側の責務。イシュー #122 レビュー指摘）。
    ///
    /// **注意（fail-open 方向のリスク）**: 組み込み既定の 3 パターン
    /// （[`crate::policy_exclusion::builtin_defaults`]）はいずれも先頭が `**`
    /// であり、`**` は空セグメントも吸収するため非正規化パス（先頭 `/` 等）
    /// でも現状は正しく match する（DP 挙動で確認済み）。しかし将来 `**` で
    /// 始まらないカスタムパターンが TOML ロード等（#124）で導入された場合、
    /// 非正規化パスに対して沈黙の不一致が起こりうる。これは本モジュール冒頭
    /// （`mod.rs`）の fail-closed／A08 基準（「発火すべき除外ルールが発火せず
    /// 自動適用へ倒れる経路を作らない」）に照らすと **fail-open 方向**
    /// （ルールが発火せず無条件人間承認を経ずに自動適用が通ってしまいうる）
    /// である。安全側の「不一致」ではない点に注意すること。
    /// 契約違反をテスト・開発ビルドで早期検知するため、明らかな非正規化パス
    /// （先頭 `/`・`\` を含む）は `debug_assert!` で検出する（本番経路では
    /// `.claude/rules/coding-rust.md` の `unwrap()`/`expect()` 禁止方針に
    /// 揃え、release ビルドではパニックしない。ただし上記の通りこれは
    /// 安全側ではなく契約違反の早期検知を目的とした assert に過ぎない）。
    pub fn matches(&self, path: &str) -> bool {
        debug_assert!(
            !path.starts_with('/') && !path.contains('\\'),
            "PathPattern::matches expects a normalized, repo-root-relative path \
             (no leading '/', no '\\'); got {path:?}. Normalize the changed-file \
             list before calling matches() (see #103 / #124)."
        );
        let path_segments: Vec<&str> = path.split('/').collect();
        segments_match(&self.segments, &path_segments)
    }
}

/// セグメント列同士の DP 照合（`**` の 0 個以上ディレクトリ一致を含む）。
///
/// `dp[i][j]` は「パターン先頭 `i` セグメントが、パス先頭 `j` セグメントに
/// 一致するか」を表す。`np * nt` に比例した反復回数で完了し、バックトラック
/// を伴わない（計画 4 節 4 番）。
fn segments_match(pattern: &[Segment], path: &[&str]) -> bool {
    let np = pattern.len();
    let nt = path.len();
    let mut dp = vec![vec![false; nt + 1]; np + 1];
    dp[0][0] = true;

    for i in 1..=np {
        if pattern[i - 1] == Segment::DoubleStar {
            dp[i][0] = dp[i - 1][0];
        }
    }

    for i in 1..=np {
        for j in 1..=nt {
            dp[i][j] = match &pattern[i - 1] {
                Segment::DoubleStar => dp[i - 1][j] || dp[i][j - 1],
                Segment::Literal(pat_seg) => {
                    dp[i - 1][j - 1] && segment_literal_match(pat_seg, path[j - 1])
                }
            };
        }
    }

    dp[np][nt]
}

/// 1 セグメント内の `?`（任意 1 文字）・`*`（`/` を跨がない任意 0 文字以上）を
/// DP で照合する（指数的バックトラック排除。計画 4 節「病的パターン」対策）。
fn segment_literal_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (np, nt) = (p.len(), t.len());

    let mut dp = vec![vec![false; nt + 1]; np + 1];
    dp[0][0] = true;
    for i in 1..=np {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }
    for i in 1..=np {
        for j in 1..=nt {
            dp[i][j] = match p[i - 1] {
                '*' => dp[i - 1][j] || dp[i][j - 1],
                '?' => dp[i - 1][j - 1],
                c => dp[i - 1][j - 1] && c == t[j - 1],
            };
        }
    }
    dp[np][nt]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_and_matches_globstar_prefix() {
        // v1 PR #181 で確立された意味論: `**/` は 0 ディレクトリにも一致する。
        let p = PathPattern::compile("**/src/model*.rs").unwrap();
        assert!(p.matches("src/model_v2.rs"));
        assert!(p.matches("crates/tensor-core/src/model_mlp.rs"));
    }

    #[test]
    fn star_does_not_cross_slash_boundary() {
        // v1 PR #181 指摘の再現テスト: `model*.rs` の `*` は `/` を跨がない。
        let p = PathPattern::compile("**/src/model*.rs").unwrap();
        assert!(!p.matches("src/model/layer.rs"));
    }

    #[test]
    fn matches_nn_directory_subtree() {
        let p = PathPattern::compile("**/src/nn/**").unwrap();
        assert!(p.matches("crates/x/src/nn/layer.rs"));
        assert!(p.matches("src/nn/deep/nested/file.rs"));
        assert!(!p.matches("src/nn_helpers.rs"));
    }

    #[test]
    fn matches_star_model_star_directory_pattern() {
        let p = PathPattern::compile("**/src/*model*/**").unwrap();
        assert!(p.matches("crates/foo/src/mymodel/x.rs"));
        assert!(!p.matches("crates/foo/src/other/x.rs"));
    }

    #[test]
    fn rejects_empty_pattern() {
        assert!(PathPattern::compile("").is_err());
    }

    #[test]
    fn rejects_absolute_path() {
        assert!(PathPattern::compile("/etc/passwd").is_err());
    }

    #[test]
    fn rejects_dotdot_segment() {
        assert!(PathPattern::compile("../secret/**").is_err());
    }

    #[test]
    fn rejects_unsupported_glob_syntax() {
        assert!(PathPattern::compile("src/[abc].rs").is_err());
        assert!(PathPattern::compile("src/{a,b}.rs").is_err());
    }

    #[test]
    fn rejects_double_star_mixed_with_other_chars() {
        assert!(PathPattern::compile("a**b/x.rs").is_err());
    }

    #[test]
    fn rejects_control_characters() {
        assert!(PathPattern::compile("src/\u{0007}model.rs").is_err());
    }

    #[test]
    fn question_mark_matches_exactly_one_character() {
        // `?` は任意の 1 文字にのみ一致する（0 文字・2 文字以上には一致しない）。
        let p = PathPattern::compile("src/model?.rs").unwrap();
        assert!(p.matches("src/model1.rs"));
        assert!(p.matches("src/modelx.rs"));
        assert!(!p.matches("src/model.rs"));
        assert!(!p.matches("src/model12.rs"));
    }

    #[test]
    fn pattern_not_starting_with_double_star_requires_exact_prefix() {
        // 組み込み既定 3 パターンはいずれも `**` 始まりだが、`**` で始まらない
        // カスタムパターン（#124 で TOML ロードにより理論上導入されうる）は
        // 正規化済みパスに対しては先頭セグメントの完全一致を要求する
        // （`matches` のドキュメンテーションコメント「fail-open 方向のリスク」
        // 参照。将来の回帰検知用）。
        let p = PathPattern::compile("src/model*.rs").unwrap();
        assert!(p.matches("src/model_v2.rs"));
        assert!(!p.matches("crates/x/src/model_v2.rs"));
    }

    #[test]
    fn pathological_pattern_does_not_blow_up() {
        // 指数的バックトラック排除の確認: 長い `*a*a*...` パターンでも
        // 多項式時間で判定が終わること（タイムアウトしないことをテストとする）。
        let pattern = format!("{}b", "*a".repeat(30));
        let p = PathPattern::compile(&pattern).unwrap();
        let text = "a".repeat(40);
        assert!(!p.matches(&text));
    }
}
