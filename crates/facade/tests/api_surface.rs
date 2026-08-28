//! 公開面の機械検査（受入基準 2。REQ-12「任意 `BackendOps` 実装を注入
//! できる公開 API を設けない」）。
//!
//! `crates/autodiff/tests/architecture_boundaries.rs` と同型のソース
//! 走査による回帰ガード: `crates/facade/src/` の全 `.rs` を対象に、
//! (a) `pub use` で `Tape`／`BackendOps`／`new_with_ops` を再エクスポート
//! していないこと、(b) `pub fn` のシグネチャが `BackendOps` を引数として
//! 直接受け取っていないことを固定する。利用者向け公開面が [`Device`]
//! 識別子のみに限定される（`fandhe_ai::tape()`／`fandhe_ai::tape_for(Device)`）
//! ことの構造的裏付け。**`visit_rs_files` は `src/` 配下を再帰走査する
//! ため、TASK-9.4（#411）で追加した `src/compat/`（`compat::array`／
//! `compat::Sequential`。旧 `fandhe_ai_autodiff::compat` からの移設）も自動的に
//! 走査対象へ含まれる**（旧 `compat::Sequential::predict_with_ops` は
//! `BackendOps` を直接引数に取っていたため、移設後の `fandhe_ai::compat`
//! がこれを公開していないことも本テストが機械的に固定する）。
//!
//! (c) `src/optim.rs`（イシュー #961）は昇格元公開面
//! （`fandhe_ai_autodiff::optim`／`fandhe_ai_autodiff::nn::optim`）と 1 対 1 で
//! 対応し、facade 独自の型・関数を持ち込まない純再エクスポートである
//! ことを固定する（`optim_module_reexports_exactly_expected_surface`／
//! `optim_module_is_pure_reexport`）。
//!
//! **A03 インジェクション対策の一環**でもある: `crates/facade/`
//! （`Cargo.toml`・`src/`）以外は走査しない固定パスのみを対象とし、
//! 外部入力を受け取らない（`.claude/rules/security.md`）。

use std::path::Path;

fn facade_crate_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_to_string_or_panic(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("test fixture: {} が読めない: {e}", path.display()))
}

fn visit_rs_files(dir: &Path, f: &mut impl FnMut(&Path, &str)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files(&path, f);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let content = read_to_string_or_panic(&path);
            f(&path, &content);
        }
    }
}

/// `crates/facade/src/` の `pub use` が `Tape`／`BackendOps`／
/// `new_with_ops` を再エクスポートしていないことを固定する
/// （モジュール冒頭コメント (a)）。
#[test]
fn facade_does_not_reexport_tape_or_backend_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            for forbidden in ["Tape", "BackendOps", "new_with_ops"] {
                if trimmed.contains(forbidden) {
                    offending.push(format!(
                        "{}: `{trimmed}` が {forbidden} を含む",
                        path.display()
                    ));
                }
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が Tape/BackendOps/new_with_ops を再エクスポートしている\
         （REQ-12「任意 BackendOps 実装を注入できる公開 API を設けない」違反）: {offending:?}"
    );
}

/// `crates/facade/src/` の `pub fn` シグネチャが `BackendOps` を引数
/// として直接受け取っていないことを固定する（モジュール冒頭コメント
/// (b)）。公開関数の入力は [`Device`] 識別子のみであるべき（受入基準 2）。
#[test]
fn facade_public_functions_do_not_accept_backend_ops_argument() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub fn") && trimmed.contains("BackendOps") {
                offending.push(format!("{}: `{trimmed}`", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の pub fn が BackendOps を直接受け取っている\
         （REQ-12「Device 識別子のみを公開面とする」違反）: {offending:?}"
    );
}

/// 公開関数 `tape_for` の入力が [`fandhe_ai::Device`] 識別子のみであることの
/// コンパイル時検証（受入基準 2）。型シグネチャが変わればこのテスト自体が
/// コンパイルエラーになるため、宣言的な固定として機能する。
///
/// 戻り値の型注釈は `fandhe_ai::Tape`（newtype。`fandhe_ai_autodiff::Tape` ではない）
/// である点も併せて固定する（codex-review PR #424 P1 是正: `fandhe_ai_autodiff::Tape`
/// を facade の公開シグネチャへ直接露出させない。`src/lib.rs` モジュール
/// doc「`Tape`（composition root が構築する値）の扱い」参照）。
#[test]
fn tape_for_accepts_device_identifier_only() {
    // `Device::Cpu` は常に構築可能（デバイス列挙・検証不要）。
    let device: fandhe_ai::Device = fandhe_ai::Device::Cpu;
    let result: Result<fandhe_ai::Tape, fandhe_ai::BackendError> = fandhe_ai::tape_for(device);
    assert!(result.is_ok(), "Device::Cpu の tape_for は常に成功するはず");
}

/// `fandhe_ai::tape()`（既定 CPU）が `CpuBackendOps` を構築していることを
/// ソース走査で固定する（`Tape::ops()` は `pub(crate)` のため統合テスト
/// から実行時に観測できない。`fusion_default_parity.rs` が数値一致で
/// 検証する「融合有効」という結論と、この「CPU バックエンドを結線して
/// いる」という前提を混同しないよう、前提のほうを本テストで明示的に
/// 固定する）。
#[test]
fn tape_reexport_wires_cpu_backend_ops() {
    let lib_rs = facade_crate_root().join("src/lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    let tape_fn = content
        .split("pub fn tape()")
        .nth(1)
        .expect("fandhe_ai::tape() の定義が見つからない");
    // 次の `pub fn` 定義（`tape_for`）が始まる手前までを `tape()` の本体とみなす。
    let tape_fn_body = tape_fn.split("pub fn tape_for").next().unwrap_or(tape_fn);
    assert!(
        tape_fn_body.contains("CpuBackendOps"),
        "fandhe_ai::tape() の本体が CpuBackendOps を構築していない\
         （既定バックエンド＝CPU の構造的裏付けが崩れている）"
    );
}

/// `crates/facade/src/compat/` の `pub fn` シグネチャが `fandhe_ai_autodiff::Tape`
/// （生の内部クレート型）を直接引数に取っていないことを固定する
/// （codex-review PR #424 P1 是正: 内部クレートの型を facade の公開
/// シグネチャへ直接露出させない。`fandhe_ai::Tape`〈newtype〉のみを取る
/// べきこと・`src/lib.rs` モジュール doc「`Tape`（composition root が
/// 構築する値）の扱い」参照）。
#[test]
fn compat_public_functions_do_not_accept_raw_autodiff_tape_argument() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut offending = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub fn") && trimmed.contains("fandhe_ai_autodiff::Tape") {
                offending.push(format!("{}: `{trimmed}`", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "fandhe_ai::compat の pub fn が fandhe_ai_autodiff::Tape（生の内部クレート型）を\
         直接引数に取っている（内部クレートの型が公開シグネチャへ露出。\
         fandhe_ai::Tape〈newtype〉を使うべき）: {offending:?}"
    );
}

/// `src/optim.rs`（イシュー #961）専用の固定パス。ソース走査対象は
/// `crates/facade/` 配下の固定パスのみに限定する（A03 対策。モジュール
/// 冒頭コメント参照）。
fn optim_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/optim.rs")
}

/// `src/optim.rs` の `pub use` 行から `{...}` 内の識別子を抽出し、
/// 昇格元公開面（`fandhe_ai_autodiff::optim`／`fandhe_ai_autodiff::nn::optim`）
/// と完全一致（過不足とも fail）することを固定する（モジュール冒頭
/// コメント (c)）。各行の path 接頭辞が上記 2 経路のいずれかであることも
/// 検査し、`tensor_core` 等の無関係なクレートからの混入を遮断する。
#[test]
fn optim_module_reexports_exactly_expected_surface() {
    let path = optim_rs_path();
    let content = read_to_string_or_panic(&path);

    let allowed_prefixes = [
        "pub use fandhe_ai_autodiff::optim::",
        "pub use fandhe_ai_autodiff::nn::optim::",
    ];

    let mut found: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut offending_lines = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("pub use") {
            continue;
        }
        let Some(prefix) = allowed_prefixes
            .iter()
            .find(|prefix| trimmed.starts_with(**prefix))
        else {
            offending_lines.push(trimmed.to_string());
            continue;
        };
        // `pub use <prefix>{A, B, C};` の `{...}` 部分を抽出する。単一
        // 識別子の再エクスポート（`{}` なし）は本ファイルでは使わない
        // 契約のため、`{`/`}` が見つからない行は不正として扱う。
        let rest = &trimmed[prefix.len()..];
        let Some(open) = rest.find('{') else {
            offending_lines.push(trimmed.to_string());
            continue;
        };
        let Some(close) = rest.find('}') else {
            offending_lines.push(trimmed.to_string());
            continue;
        };
        for ident in rest[open + 1..close].split(',') {
            let ident = ident.trim();
            if !ident.is_empty() {
                found.insert(ident.to_string());
            }
        }
    }

    assert!(
        offending_lines.is_empty(),
        "src/optim.rs の pub use が昇格元公開面\
         （fandhe_ai_autodiff::optim / fandhe_ai_autodiff::nn::optim）以外の\
         接頭辞を持つか、`{{...}}` 形式でない行を含む: {offending_lines:?}"
    );

    let expected: std::collections::BTreeSet<String> = [
        "AdamW",
        "AdamWConfig",
        "ClipGradResult",
        "clip_grad_norm",
        "global_grad_norm",
        "ConstantLr",
        "LrScheduler",
        "StepLr",
        "Sgd",
        "SgdConfig",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    assert_eq!(
        found, expected,
        "src/optim.rs が再エクスポートする識別子が期待集合と一致しない\
         （過不足いずれも不可。昇格元公開面と 1 対 1 対応であることの固定）"
    );
}

/// `src/optim.rs` が facade 独自の型・関数を定義しない純再エクスポート
/// モジュールであることを固定する（モジュール冒頭コメント (c)）。
///
/// codex-review PR #972 P2 是正: 旧実装は行頭文字列接頭辞
/// （`pub fn`／`pub struct`／`pub enum`／`pub trait`／`impl `）の列挙
/// だけを見ていたため、`pub type`／`pub const`／`pub static`／`pub mod`／
/// `pub union`／`pub async fn`／`pub(crate) fn` 等、契約上あってはならない
/// 公開宣言の追加を見逃していた。本実装はコメント・文字列リテラル・
/// char リテラル（ライフタイム注記は区別して保持する）を除去したうえで
/// トークン境界に基づき `pub` キーワードを走査し、直後のアイテム種別が
/// `use`（再エクスポート）でなければ fail-closed に拒否する。`impl` は
/// 可視性修飾子の有無に関わらず単独で拒否する（既存 forbidden_prefixes
/// の "impl " 相当をトークン境界検査に置き換えたもの）。
#[test]
fn optim_module_is_pure_reexport() {
    let path = optim_rs_path();
    let content = read_to_string_or_panic(&path);
    let offending = scan_forbidden_pub_items(&content);
    assert!(
        offending.is_empty(),
        "src/optim.rs が facade 独自の型・関数・impl・pub type/const/static/mod/union 等の\
         公開宣言を定義している（純再エクスポートモジュールの契約違反。Rust トークン境界\
         走査による検出。文字列接頭辞列挙では見逃していた `pub type`/`pub const`/`pub static`/\
         `pub mod`/`pub union`/`pub async fn` 等を含む）: {offending:?}"
    );
}

/// [`optim_module_is_pure_reexport`] が使う前処理: 行コメント・ブロック
/// コメント（ネスト対応）・文字列リテラル（raw string 含む）・char
/// リテラルの中身を、境界を保ったままスペースへ置換した `Vec<char>` を
/// 返す。文字列・コメント中に `pub fn` 等の語が現れても誤検出しない
/// ための前処理であり、出力は入力と文字数が一致する（走査後の char
/// index がそのまま元テキストの char index として使える）。ライフタイム
/// 注記（`'a` 等）は char リテラルと区別し、そのまま残す。
fn strip_comments_and_literals(src: &str) -> Vec<char> {
    let chars: Vec<char> = src.chars().collect();
    let len = chars.len();
    let mut out = Vec::with_capacity(len);
    let mut i = 0usize;
    while i < len {
        let c = chars[i];
        // 行コメント。
        if c == '/' && i + 1 < len && chars[i + 1] == '/' {
            while i < len && chars[i] != '\n' {
                out.push(' ');
                i += 1;
            }
            continue;
        }
        // ブロックコメント（Rust 仕様どおりネスト対応）。
        if c == '/' && i + 1 < len && chars[i + 1] == '*' {
            let mut depth = 1i32;
            out.push(' ');
            out.push(' ');
            i += 2;
            while i < len && depth > 0 {
                if i + 1 < len && chars[i] == '/' && chars[i + 1] == '*' {
                    depth += 1;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else if i + 1 < len && chars[i] == '*' && chars[i + 1] == '/' {
                    depth -= 1;
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                } else {
                    out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                    i += 1;
                }
            }
            continue;
        }
        // raw string リテラル（r"..."／r#"..."#／r##"..."## 等）。
        if c == 'r' {
            let mut j = i + 1;
            let mut hashes = 0usize;
            while j < len && chars[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < len && chars[j] == '"' {
                out.push(' ');
                out.extend(std::iter::repeat_n(' ', hashes));
                out.push(' ');
                let mut k = j + 1;
                loop {
                    if k >= len {
                        i = k;
                        break;
                    }
                    if chars[k] == '"' {
                        let mut h = 0usize;
                        let mut m = k + 1;
                        while m < len && chars[m] == '#' && h < hashes {
                            h += 1;
                            m += 1;
                        }
                        if h == hashes {
                            out.push(' ');
                            out.extend(std::iter::repeat_n(' ', hashes));
                            k = m;
                            i = k;
                            break;
                        }
                        out.push(' ');
                        k += 1;
                    } else {
                        out.push(if chars[k] == '\n' { '\n' } else { ' ' });
                        k += 1;
                    }
                }
                continue;
            }
            // 通常の識別子 `r` として下の通常処理へフォールスルーする。
        }
        // 通常の文字列リテラル。
        if c == '"' {
            out.push(' ');
            i += 1;
            while i < len {
                if chars[i] == '\\' && i + 1 < len {
                    out.push(' ');
                    out.push(' ');
                    i += 2;
                    continue;
                }
                if chars[i] == '"' {
                    out.push(' ');
                    i += 1;
                    break;
                }
                out.push(if chars[i] == '\n' { '\n' } else { ' ' });
                i += 1;
            }
            continue;
        }
        // char リテラル（'x'／'\n' 等）とライフタイム注記（'a 等）の判別。
        if c == '\'' {
            if i + 1 < len && chars[i + 1] == '\\' {
                let mut k = i + 2;
                while k < len && chars[k] != '\'' {
                    k += 1;
                }
                if k < len {
                    out.extend(std::iter::repeat_n(' ', k - i + 1));
                    i = k + 1;
                    continue;
                }
            } else if i + 2 < len && chars[i + 2] == '\'' {
                out.push(' ');
                out.push(' ');
                out.push(' ');
                i += 3;
                continue;
            }
            // ライフタイム注記はコード構造の一部として保持する。
            out.push(c);
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// 識別子先頭文字（ASCII のみ。本リポの Rust 識別子は ASCII 前提）。
fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

/// 識別子構成文字。
fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `chars[..idx]` 中の改行数から 1 始まりの行番号を求める（オフェンス
/// 報告用）。
fn line_at(chars: &[char], idx: usize) -> usize {
    chars[..idx].iter().filter(|&&c| c == '\n').count() + 1
}

/// コメント・文字列リテラルを除去したうえで、`pub` キーワードのうち
/// 直後のアイテム種別が `use` でないもの（`pub type`／`pub const`／
/// `pub static`／`pub mod`／`pub union`／`pub fn`／`pub struct`／
/// `pub enum`／`pub trait`／`pub(crate) fn` 等。可視性修飾子
/// `pub(...)` の有無を問わない）と、可視性修飾子の有無を問わない
/// `impl` ブロックをトークン境界（識別子の前後が識別子構成文字でない
/// こと）で検出する。文字列接頭辞の列挙ではなくキーワード単位の走査の
/// ため、契約上禁止される公開宣言の種別を将来追加しても取りこぼさない。
fn scan_forbidden_pub_items(original: &str) -> Vec<String> {
    let cleaned = strip_comments_and_literals(original);
    let len = cleaned.len();
    let mut offenses = Vec::new();
    let mut i = 0usize;
    while i < len {
        if is_ident_start(cleaned[i]) {
            let start = i;
            let mut j = i + 1;
            while j < len && is_ident_char(cleaned[j]) {
                j += 1;
            }
            let word: String = cleaned[start..j].iter().collect();
            if word == "pub" {
                let mut k = j;
                while k < len && cleaned[k].is_whitespace() {
                    k += 1;
                }
                // 可視性修飾子 `pub(crate)`／`pub(super)` 等を読み飛ばす。
                if k < len && cleaned[k] == '(' {
                    let mut depth = 1i32;
                    k += 1;
                    while k < len && depth > 0 {
                        match cleaned[k] {
                            '(' => depth += 1,
                            ')' => depth -= 1,
                            _ => {}
                        }
                        k += 1;
                    }
                    while k < len && cleaned[k].is_whitespace() {
                        k += 1;
                    }
                }
                if k < len && is_ident_start(cleaned[k]) {
                    let ks = k;
                    let mut ke = k + 1;
                    while ke < len && is_ident_char(cleaned[ke]) {
                        ke += 1;
                    }
                    let next_word: String = cleaned[ks..ke].iter().collect();
                    if next_word != "use" {
                        offenses.push(format!(
                            "line {}: `pub {next_word}` は再エクスポート（`pub use`）以外の公開宣言",
                            line_at(&cleaned, start)
                        ));
                    }
                }
            } else if word == "impl" {
                offenses.push(format!(
                    "line {}: `impl` ブロックの定義",
                    line_at(&cleaned, start)
                ));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    offenses
}

/// `fandhe_ai::optim` の全再エクスポート型・関数が facade のみを通じて
/// 到達可能であることのコンパイル時固定（モジュール冒頭コメント (c)、
/// 受入基準 1）。`fandhe_ai_autodiff` は import しない。
///
/// `let _: fandhe_ai::SgdConfig = fandhe_ai::optim::SgdConfig::new(...)`
/// でクレート root 再エクスポートと `optim::SgdConfig` が同一型である
/// ことも併せて固定する（`src/lib.rs` root `SgdConfig` コメント参照）。
#[test]
fn optim_types_are_reachable_via_facade_only() {
    let sgd_config = fandhe_ai::optim::SgdConfig::new(0.1);
    // root 再エクスポートと `optim::SgdConfig` が同一型であることの固定。
    let _same_type: fandhe_ai::SgdConfig = sgd_config;
    let mut sgd = fandhe_ai::optim::Sgd::new(sgd_config)
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
    let _ = &mut sgd;

    let mut adamw = fandhe_ai::optim::AdamW::new(fandhe_ai::optim::AdamWConfig::default())
        .unwrap_or_else(|e| panic!("test fixture: AdamW::new が失敗した: {e}"));
    let _ = &mut adamw;

    let constant_lr = fandhe_ai::optim::ConstantLr::new(0.1)
        .unwrap_or_else(|e| panic!("test fixture: ConstantLr::new が失敗した: {e}"));
    let step_lr = fandhe_ai::optim::StepLr::new(0.1, 2, 0.5)
        .unwrap_or_else(|e| panic!("test fixture: StepLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &constant_lr;
    let _: &dyn fandhe_ai::optim::LrScheduler = &step_lr;

    let result: fandhe_ai::optim::ClipGradResult = fandhe_ai::optim::clip_grad_norm(&[], 1.0)
        .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm が失敗した: {e}"));
    assert_eq!(
        result.total_norm, 0.0,
        "test fixture: 空スライスの norm は 0"
    );
    assert!(!result.scaled, "test fixture: 空スライスは scaled しない");

    let global_norm = fandhe_ai::optim::global_grad_norm(&[])
        .unwrap_or_else(|e| panic!("test fixture: global_grad_norm が失敗した: {e}"));
    assert_eq!(global_norm, 0.0, "test fixture: 空スライスの norm は 0");
}
