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
//! `optim_module_is_pure_reexport`）。イシュー #1722 で AMP（`GradScaler`／
//! `GradScalerConfig`／`UnscaleResult`／`scale_loss`／`scale_grads`／
//! `unscale_grads`／`has_non_finite`。実体は `fandhe_ai_autodiff::nn::optim::amp`
//! モジュールだが再エクスポートは `nn::optim` 経由）を期待集合へ追加した。
//! イシュー #1742 で Adam（coupled L2 weight decay。`Adam`／`AdamConfig`。
//! 実体は `fandhe_ai_autodiff::nn::optim::adam` モジュール）を期待集合へ
//! 追加した。イシュー #1743（親 #1610）で RMSprop（`RmsProp`／
//! `RmsPropConfig`）・Adagrad（`Adagrad`／`AdagradConfig`）を期待集合へ
//! 追加した。イシュー #1744 で LAMB（`Lamb`／`LambConfig`。実体は
//! `fandhe_ai_autodiff::nn::optim::lamb` モジュール）を期待集合へ追加した。
//! イシュー #1746（親 #1611）で ReduceLrOnPlateau
//! （`ReduceLrOnPlateau`／`ReduceLrOnPlateauConfig`／`PlateauMode`／
//! `ThresholdMode`。実体は `fandhe_ai_autodiff::nn::optim::
//! reduce_lr_on_plateau` モジュール）を期待集合へ追加した。
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

/// `crates/facade/src/` が `DeviceAllocator`／`BufferHandle`／
/// `SizeClassPool`（内部クレートのプール実装型。イシュー #1020・REQ-14）
/// を一切公開していないことを固定する（`docs/device-memory-pool-design.md`
/// §「facade 到達経路」: facade はプール実装型を露出させず、
/// [`fandhe_ai::PoolStats`]（POD）と `release_cached_memory`/
/// `memory_pool_stats`（unit/Option 返却関数）のみを確定入口とする）。
#[test]
fn facade_does_not_expose_pool_implementation_types() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for forbidden in ["DeviceAllocator", "BufferHandle", "SizeClassPool"] {
            if content.contains(forbidden) {
                offending.push(format!("{}: `{forbidden}` を含む", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade がプール実装型（DeviceAllocator/BufferHandle/SizeClassPool）を\
         公開面に露出させている: {offending:?}"
    );
}

/// `fandhe_ai::release_cached_memory`／`memory_pool_stats` が `pub fn`
/// として、`PoolStats` が `pub use` として存在することを固定する
/// （イシュー #1020 の facade 確定入口。`docs/compat-api-scope.md` §0）。
#[test]
fn facade_exposes_pool_release_api_and_pool_stats_reexport() {
    let lib_rs = facade_crate_root().join("src/lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    assert!(
        content.contains("pub fn release_cached_memory"),
        "fandhe_ai::release_cached_memory が pub fn として見つからない"
    );
    assert!(
        content.contains("pub fn memory_pool_stats"),
        "fandhe_ai::memory_pool_stats が pub fn として見つからない"
    );
    let has_pool_stats_reexport = content
        .lines()
        .any(|line| line.trim_start().starts_with("pub use") && line.contains("PoolStats"));
    assert!(
        has_pool_stats_reexport,
        "fandhe_ai::PoolStats の pub use 再エクスポートが見つからない"
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
        "Adagrad",
        "AdagradConfig",
        "Adam",
        "AdamConfig",
        "AdamW",
        "AdamWConfig",
        "Lamb",
        "LambConfig",
        "ClipGradResult",
        "clip_grad_norm",
        "clip_grad_value",
        "global_grad_norm",
        "ConstantLr",
        "CosineAnnealingLr",
        "ExponentialLr",
        "LinearWarmupLr",
        "LrScheduler",
        "OneCycleAnneal",
        "OneCycleLr",
        "OneCycleLrConfig",
        "StepLr",
        "RmsProp",
        "RmsPropConfig",
        "PlateauMode",
        "ThresholdMode",
        "ReduceLrOnPlateau",
        "ReduceLrOnPlateauConfig",
        "Sgd",
        "SgdConfig",
        "GradScaler",
        "GradScalerConfig",
        "UnscaleResult",
        "has_non_finite",
        "scale_grads",
        "scale_loss",
        "unscale_grads",
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

    // Adam（coupled L2 weight decay。イシュー #1742）の facade 到達性固定。
    // `AdamConfig::default().weight_decay == 0.0`（PyTorch `torch.optim.Adam`
    // の既定値）が `AdamWConfig::default()` の `0.01` とドリフトしないこと
    // も併せて固定する（`nn::optim::adam` モジュール doc 参照）。
    let adam_config = fandhe_ai::optim::AdamConfig::default();
    assert_eq!(
        adam_config.weight_decay, 0.0,
        "test fixture: AdamConfig の既定 weight_decay は PyTorch Adam と同じ 0.0"
    );
    let mut adam = fandhe_ai::optim::Adam::new(adam_config)
        .unwrap_or_else(|e| panic!("test fixture: Adam::new が失敗した: {e}"));
    let _ = &mut adam;

    // RMSprop／Adagrad（イシュー #1743・親 #1610）が facade のみを
    // 通じて到達可能であることの固定＋既定値ドリフトガード
    // （`torch.optim.RMSprop`／`torch.optim.Adagrad` の既定値と一致する
    // ことを `nn::optim::rmsprop`／`adagrad` doc と合わせて固定する）。
    let rmsprop_config = fandhe_ai::optim::RmsPropConfig::default();
    assert_eq!(
        rmsprop_config.alpha, 0.99,
        "test fixture: RmsPropConfig の既定 alpha は torch.optim.RMSprop と同じ 0.99"
    );
    let mut rmsprop = fandhe_ai::optim::RmsProp::new(rmsprop_config)
        .unwrap_or_else(|e| panic!("test fixture: RmsProp::new が失敗した: {e}"));
    let _ = &mut rmsprop;

    let adagrad_config = fandhe_ai::optim::AdagradConfig::default();
    assert_eq!(
        adagrad_config.eps, 1e-10,
        "test fixture: AdagradConfig の既定 eps は torch.optim.Adagrad と同じ 1e-10"
    );
    let mut adagrad = fandhe_ai::optim::Adagrad::new(adagrad_config)
        .unwrap_or_else(|e| panic!("test fixture: Adagrad::new が失敗した: {e}"));
    let _ = &mut adagrad;

    // LAMB（イシュー #1744）の facade 到達性固定。既定値ドリフトガード
    // （`eps=1e-6` は AdamW／Adam の `1e-8` と異なる・`weight_decay=0.0`。
    // `nn::optim::lamb` モジュール doc 参照）。
    let lamb_config = fandhe_ai::optim::LambConfig::default();
    assert_eq!(
        lamb_config.eps, 1e-6,
        "test fixture: LambConfig の既定 eps は paper／apex／torch_optimizer 共通の 1e-6"
    );
    assert_eq!(
        lamb_config.weight_decay, 0.0,
        "test fixture: LambConfig の既定 weight_decay は 0.0"
    );
    let mut lamb = fandhe_ai::optim::Lamb::new(lamb_config)
        .unwrap_or_else(|e| panic!("test fixture: Lamb::new が失敗した: {e}"));
    let _ = &mut lamb;

    let constant_lr = fandhe_ai::optim::ConstantLr::new(0.1)
        .unwrap_or_else(|e| panic!("test fixture: ConstantLr::new が失敗した: {e}"));
    let step_lr = fandhe_ai::optim::StepLr::new(0.1, 2, 0.5)
        .unwrap_or_else(|e| panic!("test fixture: StepLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &constant_lr;
    let _: &dyn fandhe_ai::optim::LrScheduler = &step_lr;

    // イシュー #1745: CosineAnnealingLr／ExponentialLr／LinearWarmupLr
    // が facade のみ import で構築でき、既存 2 型と同じ `&dyn
    // LrScheduler` へ coercion できることを固定する。
    let cosine_lr = fandhe_ai::optim::CosineAnnealingLr::new(0.1, 4, 0.0)
        .unwrap_or_else(|e| panic!("test fixture: CosineAnnealingLr::new が失敗した: {e}"));
    let exponential_lr = fandhe_ai::optim::ExponentialLr::new(0.1, 0.5)
        .unwrap_or_else(|e| panic!("test fixture: ExponentialLr::new が失敗した: {e}"));
    let linear_warmup_lr = fandhe_ai::optim::LinearWarmupLr::new(0.1, 4, 0.25)
        .unwrap_or_else(|e| panic!("test fixture: LinearWarmupLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &cosine_lr;
    let _: &dyn fandhe_ai::optim::LrScheduler = &exponential_lr;
    let _: &dyn fandhe_ai::optim::LrScheduler = &linear_warmup_lr;

    // イシュー #1747: OneCycleLr が facade のみ import で構築でき、
    // 既存スケジューラと同じ `&dyn LrScheduler` へ coercion できる
    // ことを固定する。`OneCycleLrConfig::new` の既定値が PyTorch
    // `OneCycleLR` の既定値と一致することも併せて固定する
    // （`RmsPropConfig::default().alpha` 等と同型のドリフトガード）。
    let one_cycle_config = fandhe_ai::optim::OneCycleLrConfig::new(0.1, 10);
    assert_eq!(
        one_cycle_config.pct_start, 0.3,
        "test fixture: OneCycleLrConfig::new の既定 pct_start は PyTorch と同じ 0.3"
    );
    assert_eq!(
        one_cycle_config.anneal_strategy,
        fandhe_ai::optim::OneCycleAnneal::Cos,
        "test fixture: OneCycleLrConfig::new の既定 anneal_strategy は PyTorch と同じ 'cos'"
    );
    assert_eq!(
        one_cycle_config.div_factor, 25.0,
        "test fixture: OneCycleLrConfig::new の既定 div_factor は PyTorch と同じ 25.0"
    );
    assert_eq!(
        one_cycle_config.final_div_factor, 1e4,
        "test fixture: OneCycleLrConfig::new の既定 final_div_factor は PyTorch と同じ 1e4"
    );
    assert!(
        !one_cycle_config.three_phase,
        "test fixture: OneCycleLrConfig::new の既定 three_phase は PyTorch と同じ false"
    );
    let one_cycle_lr = fandhe_ai::optim::OneCycleLr::new(one_cycle_config)
        .unwrap_or_else(|e| panic!("test fixture: OneCycleLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &one_cycle_lr;

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

    // clip_grad_value（イシュー #1753・親 #1631。value 方式 gradient
    // clipping）が facade のみを通じて到達可能であることの固定。
    let clipped_values = fandhe_ai::optim::clip_grad_value(&[], 1.0)
        .unwrap_or_else(|e| panic!("test fixture: clip_grad_value が失敗した: {e}"));
    assert!(
        clipped_values.is_empty(),
        "test fixture: 空スライスの clip_grad_value は空 Vec"
    );

    // AMP（イシュー #1722）: `GradScalerConfig::default()` が PyTorch
    // `torch.cuda.amp.GradScaler` の既定 `init_scale=2**16` と一致することの
    // ドリフトガード（`nn::optim::amp::GradScalerConfig` doc 参照）。
    let config = fandhe_ai::optim::GradScalerConfig::default();
    assert_eq!(
        config.init_scale, 65536.0,
        "test fixture: GradScalerConfig の既定 init_scale は PyTorch と同じ 2**16"
    );

    let scaler = fandhe_ai::optim::GradScaler::new(config)
        .unwrap_or_else(|e| panic!("test fixture: GradScaler::new が失敗した: {e}"));
    assert_eq!(
        scaler.scale(),
        65536.0,
        "test fixture: 構築直後の scale は init_scale と一致するはず"
    );

    let scaled = fandhe_ai::optim::scale_grads(&[], 1.0)
        .unwrap_or_else(|e| panic!("test fixture: scale_grads が失敗した: {e}"));
    assert!(scaled.is_empty(), "test fixture: 空スライスは空 Vec を返す");

    let unscale_result: fandhe_ai::optim::UnscaleResult = fandhe_ai::optim::unscale_grads(&[], 1.0)
        .unwrap_or_else(|e| panic!("test fixture: unscale_grads が失敗した: {e}"));
    assert!(
        !unscale_result.found_non_finite,
        "test fixture: 空スライスに非有限値は含まれない"
    );
    assert!(
        !unscale_result.should_skip_step(),
        "test fixture: 空スライスの unscale 結果は step をスキップしない"
    );

    assert!(
        !fandhe_ai::optim::has_non_finite(&[]),
        "test fixture: 空スライスに非有限値は含まれない"
    );

    // `scale_loss` は `&Var` を受け取る唯一の AMP 関数（`crate::Var::mul` の
    // 合成のみで実装。`optim.rs` モジュール doc「REQ-12 との整合」節参照）。
    let tape = fandhe_ai::tape();
    let loss = tape.var(&fandhe_ai::Tensor::scalar(1.0_f32));
    let scaled_loss = fandhe_ai::optim::scale_loss(&loss, 2.0)
        .unwrap_or_else(|e| panic!("test fixture: scale_loss が失敗した: {e}"));
    let scaled_value = scaled_loss
        .to_tensor()
        .get(&[])
        .unwrap_or_else(|| panic!("test fixture: スカラー shape [] のはず"));
    assert_eq!(
        scaled_value, 2.0,
        "test fixture: scale_loss(1.0, 2.0) は 2.0 のはず"
    );

    // ReduceLrOnPlateau（イシュー #1746・親 #1611）が facade のみを
    // 通じて到達可能であることの固定＋既定値ドリフトガード
    // （PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau` の既定値と
    // 一致することを `nn::optim::reduce_lr_on_plateau` doc と合わせて
    // 固定する）。
    let plateau_config = fandhe_ai::optim::ReduceLrOnPlateauConfig::default();
    assert_eq!(
        plateau_config.mode,
        fandhe_ai::optim::PlateauMode::Min,
        "test fixture: ReduceLrOnPlateauConfig の既定 mode は PyTorch と同じ Min"
    );
    assert_eq!(
        plateau_config.threshold_mode,
        fandhe_ai::optim::ThresholdMode::Rel,
        "test fixture: ReduceLrOnPlateauConfig の既定 threshold_mode は PyTorch と同じ Rel"
    );
    assert_eq!(
        plateau_config.patience, 10,
        "test fixture: ReduceLrOnPlateauConfig の既定 patience は PyTorch と同じ 10"
    );
    let mut plateau = fandhe_ai::optim::ReduceLrOnPlateau::new(0.1, plateau_config)
        .unwrap_or_else(|e| panic!("test fixture: ReduceLrOnPlateau::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &plateau;
    let updated_lr = plateau
        .step(1.0)
        .unwrap_or_else(|e| panic!("test fixture: ReduceLrOnPlateau::step が失敗した: {e}"));
    assert_eq!(
        updated_lr, 0.1,
        "test fixture: 初回観測（改善扱い）では減衰しないはず"
    );
}

/// デバイスメモリプール（イシュー #1021）の公開面固定（受入基準
/// `docs/device-memory-pool-design.md` §3.1「`tensor-core` にはハンドル
/// の内部表現を一切含まない POD 型のみを置く」）。
///
/// (a) `src/` の `pub use`／`pub fn` シグネチャに低水準アロケータ
/// 型（`DeviceAllocator`／`BufferHandle`）が一切現れないことを固定する
/// （`facade_does_not_reexport_tape_or_backend_ops` と同型の走査）。
#[test]
fn facade_does_not_expose_low_level_allocator_types() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            let is_pub_surface = trimmed.starts_with("pub use") || trimmed.starts_with("pub fn");
            if !is_pub_surface {
                continue;
            }
            for forbidden in ["DeviceAllocator", "BufferHandle"] {
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
        "facade の公開面が低水準アロケータ型（DeviceAllocator/BufferHandle）を\
         含んでいる（設計文書 §3.1 のカプセル化契約違反）: {offending:?}"
    );
}

/// (b) `release_cached_memory`／`memory_pool_stats`（`pub fn`）・
/// `PoolStats`（`pub use`）が facade の公開面に存在することを固定する
/// （REQ-14 の明示解放 API・診断統計が利用者から到達できることの
/// コンパイル時裏付け）。
#[test]
fn release_cached_memory_and_pool_stats_are_reachable_via_facade() {
    // CPU はプールを持たないため常に `Ok(())`／`Ok(None)`（`BackendOps`
    // の既定実装。`docs/device-memory-pool-design.md` §3.1）。
    fandhe_ai::release_cached_memory(fandhe_ai::Device::Cpu)
        .expect("test fixture: CPU の release_cached_memory は常に成功するはず");
    let stats = fandhe_ai::memory_pool_stats(fandhe_ai::Device::Cpu)
        .expect("test fixture: CPU の memory_pool_stats は常に成功するはず");
    assert!(
        stats.is_none(),
        "test fixture: CPU はプールを持たないため None のはず"
    );

    // `PoolStats` 自体がクレート root から到達可能で POD であることの
    // 型検査（`Copy`/`Clone`/`Debug`/`Default` を要求しない最小限の
    // 構築・比較のみ行う）。
    let a = fandhe_ai::PoolStats::default();
    let b = a;
    assert_eq!(a, b, "test fixture: PoolStats は値として比較できるはず");
}

/// `fandhe_ai::manual_seed`（イシュー #1724）が facade から呼び出し可能な
/// `pub fn` として型検査できることを固定する（コンパイル時裏付け）。
/// グローバル RNG 状態を実際に変更するため、他テストとの競合を避ける
/// 目的で値の検証は行わず、呼べることだけを確認する（決定性・独立性の
/// 単体テストは `crates/tensor-core/src/rng.rs`・
/// `crates/autodiff/src/nn/init.rs` 側に別途整備済み）。
#[test]
fn manual_seed_is_reachable_via_facade() {
    fandhe_ai::manual_seed(42);
}

/// `fandhe_ai::{randn, rand, randint}`（イシュー #1725）が facade から
/// 呼び出し可能な `pub fn` として型検査できることを固定する（コンパイル
/// 時裏付け）。グローバル RNG 状態を変更するため値の検証は行わず、呼べ
/// て `Result` を受け取れることだけを確認する（決定性・アルゴリズムの
/// 単体テストは `crates/tensor-core/src/rng.rs` 側に整備済み）。
#[test]
fn rng_tensor_generators_are_reachable_via_facade() {
    let _n: Result<fandhe_ai::Tensor<f32>, _> = fandhe_ai::randn(&[2, 3]);
    let _u: Result<fandhe_ai::Tensor<f32>, _> = fandhe_ai::rand(&[2, 3]);
    let _i: Result<fandhe_ai::Tensor<i32>, fandhe_ai::RngError> = fandhe_ai::randint(0, 10, &[4]);
}

/// `fandhe_ai::{arange, linspace, eye, zeros_like, ones_like}`（イシュー
/// #1726）が facade から呼び出し可能な `pub fn` として型検査できることを
/// 固定する（コンパイル時裏付け。[`rng_tensor_generators_are_reachable_via_facade`]
/// と同型）。
#[test]
fn creation_tensor_generators_are_reachable_via_facade() {
    let _a: Result<fandhe_ai::Tensor<f32>, fandhe_ai::CreationError> =
        fandhe_ai::arange(0.0, 5.0, 1.0);
    let _l: Result<fandhe_ai::Tensor<f32>, fandhe_ai::CreationError> =
        fandhe_ai::linspace(0.0, 1.0, 3);
    let _e: Result<fandhe_ai::Tensor<f32>, fandhe_ai::ShapeError> = fandhe_ai::eye(3);
    let like = fandhe_ai::Tensor::<f32>::zeros(&[2, 3]).unwrap();
    let _z: Result<fandhe_ai::Tensor<f32>, fandhe_ai::ShapeError> = fandhe_ai::zeros_like(&like);
    let _o: Result<fandhe_ai::Tensor<f32>, fandhe_ai::ShapeError> = fandhe_ai::ones_like(&like);
}

/// `fandhe_ai::src/` の公開面に、プロセスグローバル RNG の内部実装型
/// （`Xorshift64Star`・内部アクセサ `with_global_rng`）が一切露出して
/// いないことを固定する（`manual_seed`／`randn`／`rand`／`randint`〈#1725〉
/// のみを公開面とし、それらが内部で使う抽選アクセサはサポート対象外に
/// 留める設計。`docs/rng-global-contract-design.md`）。
#[test]
fn facade_does_not_expose_rng_internal_types() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for forbidden in ["Xorshift64Star", "with_global_rng"] {
            if content.contains(forbidden) {
                offending.push(format!("{}: {forbidden} を含む", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が RNG 内部実装型・内部アクセサを含んでいる: {offending:?}"
    );
}

/// facade（crates.io 公開クレート `fandhe-ai`）が非公開クレート
/// `onnx-interop`（`publish = false`。#1775・
/// `docs/facade-onnx-export-exposure-decision.md`）へ通常依存を持たない
/// ことを固定する。
///
/// `docs/crates-io-publishing-order.md` §6 は「公開 6 クレートの
/// `[dependencies]` に非公開クレートが現れない」ことを実測確認済みの
/// 前提としているが、CI は `cargo publish --dry-run` を実行しないため、
/// 誰かが `[dependencies]` へ `onnx-interop = { path = "../onnx-interop" }`
/// を追加しても通常の `cargo build`／`cargo test` は成功してしまい、
/// 壊れるのは次回リリース（`release-all.yml`）実行時になる。本テストは
/// この盲点を CI 時点の失敗へ前倒しする（ホストクレート `Cargo.toml`
/// のみを対象とする固定パス走査で、外部入力を受け取らない。
/// `.claude/rules/security.md` A08）。
///
/// **段階 0 の間の負のガードである**: `onnx-interop` の crates.io 公開
/// 承認（`docs/facade-onnx-import-exposure-decision.md` §6.1）とラッパー
/// API 実装 issue（`docs/facade-onnx-export-exposure-decision.md` §6）
/// が完了したら、本テストは削除ではなく「承認済み依存形状
/// （`version = "=x.y.z"` 併記等）の検査」へ差し替える。
///
/// `[dev-dependencies]` は対象外（公開クレートの `Cargo.toml` に残っても
/// `cargo publish` を壊さない。`bench-harness` 等の既存 dev-dependency と
/// 同じ扱い）。`[dependencies]`・`[build-dependencies]`・
/// `[target.'cfg(...)'.dependencies]`・`[dependencies.<name>]` 形の
/// ヘッダ・`package = "..."` によるリネーム迂回はいずれも検出する。
const FORBIDDEN_ONNX_NAMES: [&str; 3] = [
    "onnx-interop",
    "fandhe-ai-onnx-interop",
    "fandhe-ai-interop",
];

fn is_dev_dependencies_section(section: &str) -> bool {
    section.contains("dev-dependencies")
}

fn is_relevant_dependency_section(section: &str) -> bool {
    section.contains("dependencies") && !is_dev_dependencies_section(section)
}

/// `facade_does_not_depend_on_unpublished_onnx_interop` の走査ロジック本体。
/// 実 `Cargo.toml` からの呼び出しと、パース境界（コメント付きヘッダ行
/// 等）を固定する合成入力からの呼び出しの両方から使えるよう、ファイル
/// 読み込みと判定ロジックを分離した（#1775 PR #1821 codex-review 指摘の
/// 回帰テスト `dependency_section_header_with_trailing_comment_is_recognized`
/// 用）。戻り値は `(offending, saw_known_dependency_line)`。
fn scan_onnx_dependency_offenses(content: &str) -> (Vec<String>, bool) {
    let mut current_section = String::new();
    let mut offending = Vec::new();
    let mut saw_known_dependency_line = false;

    for line in content.lines() {
        let trimmed = line.trim();
        // コメント除去（`#` 以降）。TOML の文字列値に `#` を含む既存行は
        // 本ファイルには存在しないため、この単純化で安全に判定できる。
        // ヘッダ判定（`[section]` の角カッコ）もこのコメント除去後の
        // `code_part` に対して行う: `[build-dependencies] # export support`
        // のようなコメント付きヘッダ行は元の `trimmed` では `]` で終わら
        // ないため、コメント除去前に判定すると `current_section` の切替
        // を見落とし、以降の依存走査が丸ごと読み飛ばされてしまう
        // （codex-review 指摘。#1775 PR #1821）。
        let code_part = trimmed
            .split_once('#')
            .map(|(a, _)| a)
            .unwrap_or(trimmed)
            .trim();
        if code_part.starts_with('[') && code_part.ends_with(']') {
            current_section = code_part.to_string();
            if is_relevant_dependency_section(&current_section) {
                for name in FORBIDDEN_ONNX_NAMES {
                    if current_section.contains(name) {
                        offending.push(format!(
                            "section header `{current_section}` が `{name}` を含む"
                        ));
                    }
                }
            }
            continue;
        }
        if !is_relevant_dependency_section(&current_section) {
            continue;
        }
        if code_part.is_empty() {
            continue;
        }
        let Some((key, value)) = code_part.split_once('=') else {
            continue;
        };
        let key = key.trim().trim_matches('"');
        if FORBIDDEN_ONNX_NAMES.contains(&key) {
            offending.push(format!(
                "`{current_section}` に依存 `{key}` を検出: `{trimmed}`"
            ));
        }
        // `package = "onnx-interop"` によるクレート名リネーム迂回も検出する。
        for name in FORBIDDEN_ONNX_NAMES {
            if value.contains(name) {
                offending.push(format!(
                    "`{current_section}` の行 `{trimmed}` の値に `{name}` を検出\
                     （package リネーム等での迂回を含む）"
                ));
            }
        }
        if key == "fandhe-ai-tensor-core" || key == "fandhe-ai-autodiff" {
            saw_known_dependency_line = true;
        }
    }

    (offending, saw_known_dependency_line)
}

#[test]
fn facade_does_not_depend_on_unpublished_onnx_interop() {
    let cargo_toml_path = facade_crate_root().join("Cargo.toml");
    let content = read_to_string_or_panic(&cargo_toml_path);

    let (offending, saw_known_dependency_line) = scan_onnx_dependency_offenses(&content);

    // 空虚 pass 防止の自己検証: 少なくとも 1 行、既知の依存
    // （`fandhe-ai-tensor-core` 等）を対象セクション内で実際に走査した
    // ことを確認する。
    assert!(
        saw_known_dependency_line,
        "自己検証: 既知の依存行（fandhe-ai-tensor-core／fandhe-ai-autodiff）が\
         依存セクション内で検出されなかった。走査ロジックが空虚に pass して\
         いる可能性がある（{cargo_toml_path:?} を確認）"
    );
    assert!(
        offending.is_empty(),
        "facade（crates.io 公開クレート）が非公開クレート onnx-interop へ\
         通常依存している（docs/crates-io-publishing-order.md §6 違反。\
         cargo publish が次回リリースで壊れる）: {offending:?}"
    );
}

/// codex-review 指摘（#1775 PR #1821）の回帰固定: 依存セクションヘッダ行
/// に行末コメントが付く実在パターン（`[build-dependencies] # export
/// support` 等）でも `current_section` の切替が見落とされず、ヘッダ直後
/// の禁止依存行がすり抜けずに検出されることを確認する。コメント除去前に
/// `]` で終わるかどうかだけを見て判定する実装だと、この形のヘッダ行は
/// 「セクションヘッダではない普通の行」として無視され、以降の依存走査
/// （`onnx-interop` 検出）が丸ごと読み飛ばされてしまう。
#[test]
fn dependency_section_header_with_trailing_comment_is_recognized() {
    let synthetic_cargo_toml = r#"
[package]
name = "fandhe-ai"

[dependencies]
fandhe-ai-tensor-core = { version = "=0.9.0", path = "../tensor-core" }
fandhe-ai-autodiff = { version = "=0.9.0", path = "../autodiff" }

[build-dependencies] # export support
onnx-interop = { path = "../onnx-interop" }
"#;

    let (offending, saw_known_dependency_line) =
        scan_onnx_dependency_offenses(synthetic_cargo_toml);

    assert!(
        saw_known_dependency_line,
        "自己検証: 合成入力の [dependencies] セクションが走査されなかった"
    );
    assert!(
        !offending.is_empty(),
        "コメント付きヘッダ `[build-dependencies] # export support` の直後にある\
         禁止依存 `onnx-interop` が検出されなかった（ヘッダ切替の見落としが\
         再発している）"
    );
    assert!(
        offending
            .iter()
            .any(|entry| entry.contains("onnx-interop") && entry.contains("build-dependencies")),
        "検出結果に build-dependencies セクションでの onnx-interop 依存が\
         含まれているはず: {offending:?}"
    );
}

/// facade の `src/` が `onnx-interop`（クレート名を Rust 識別子化した
/// `onnx_interop`）を `use`／型パス等で一切参照していないことを固定する
/// （上記 Cargo.toml 側ガードの対を成す src 側チェック。#1775）。
#[test]
fn facade_sources_do_not_reference_onnx_interop() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for forbidden in [
                "onnx_interop",
                "fandhe_ai_onnx_interop",
                "fandhe_ai_interop",
            ] {
                if line.contains(forbidden) {
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
        "facade の src/ が onnx-interop を参照している\
         （facade は非公開クレートへ依存しない設計。#1775）: {offending:?}"
    );
}

/// `fandhe_ai::{CastDType, CastElement}`（イシュー #1750）が facade から
/// 到達可能であること・`Var::cast`／`Tape::var_from` が facade 経由でも
/// 型検査できることを固定する（コンパイル時裏付け。
/// `rng_tensor_generators_are_reachable_via_facade` と同型）。`CastDType`
/// は `#[non_exhaustive]` のためワイルドカード腕で網羅する。
#[test]
fn cast_types_are_reachable_via_facade() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&fandhe_ai::Tensor::<f32>::zeros(&[3]).unwrap());
    let casted: fandhe_ai::Tensor<i32> = x.cast().unwrap();
    assert_eq!(casted.shape(), &[3]);

    let bool_in = fandhe_ai::Tensor::<bool>::new(vec![true, false], &[2]).unwrap();
    let y = tape.var_from(&bool_in).unwrap();
    assert_eq!(y.to_tensor().shape(), &[2]);

    let dtype: fandhe_ai::CastDType = <i32 as fandhe_ai::CastElement>::CAST_DTYPE;
    let _label = match dtype {
        fandhe_ai::CastDType::F32 => "f32",
        fandhe_ai::CastDType::F64 => "f64",
        fandhe_ai::CastDType::I32 => "i32",
        fandhe_ai::CastDType::I64 => "i64",
        fandhe_ai::CastDType::Bool => "bool",
        _ => "unknown",
    };
}

/// `crates/facade/src/` の `pub use` が `CastOps`（dtype 変換の動的
/// ディスパッチ面）を再エクスポートしていないことを固定する
/// （`docs/tensor-core-cast-design.md`「facade は `CastOps` を
/// 再エクスポートしない」設計判断。`facade_does_not_reexport_tape_
/// or_backend_ops` と同型の走査）。
#[test]
fn facade_does_not_reexport_cast_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            if trimmed.contains("CastOps") {
                offending.push(format!("{}: `{trimmed}` が CastOps を含む", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が CastOps を再エクスポートしている\
         （動的ディスパッチ面は非公開の設計判断に違反）: {offending:?}"
    );
}

/// `fandhe_ai::available_devices`（イシュー #1614）が `pub fn` として、
/// `fandhe_ai::Tape::device`／`Tape::transfer` が facade の公開面に
/// 存在することを固定する（`facade_exposes_pool_release_api_and_pool_
/// stats_reexport` と同型のソース走査）。
#[test]
fn facade_exposes_available_devices_and_tape_transfer() {
    let lib_rs = facade_crate_root().join("src/lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    assert!(
        content.contains("pub fn available_devices"),
        "fandhe_ai::available_devices が pub fn として見つからない"
    );
    assert!(
        content.contains("pub fn device(&self) -> Device"),
        "fandhe_ai::Tape::device が pub fn として見つからない"
    );
    assert!(
        content.contains("pub fn transfer"),
        "fandhe_ai::Tape::transfer が pub fn として見つからない"
    );
}

/// `crates/facade/src/` の `pub use` が `DeviceProvider`／`DeviceInfo`／
/// `enumerate_all`／`select_from`（`tensor-core::device` の下位 API。
/// 利用者向け公開面は `Device` 識別子のみに限定する方針）を再エクス
/// ポートしていないことを固定する（`facade_does_not_reexport_cast_ops`
/// と同型の走査。イシュー #1614）。
#[test]
fn facade_does_not_reexport_device_provider_internals() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            for forbidden in [
                "DeviceProvider",
                "DeviceInfo",
                "enumerate_all",
                "select_from",
            ] {
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
        "facade の公開面が DeviceProvider 系の下位 API を再エクスポートしている\
         （利用者向け公開面は Device 識別子のみとする方針に違反）: {offending:?}"
    );
}

/// `fandhe_ai::available_devices`（イシュー #1614）がコンパイル時に
/// `Vec<fandhe_ai::Device>` を返す `pub fn` として到達可能であることの
/// コンパイル時検証（`manual_seed_is_reachable_via_facade` と同型）。
#[test]
fn available_devices_is_reachable_via_facade() {
    let devices: Vec<fandhe_ai::Device> = fandhe_ai::available_devices();
    assert!(
        devices.contains(&fandhe_ai::Device::Cpu),
        "available_devices() は常に Device::Cpu を含むはず"
    );
}

/// `src/data.rs`（イシュー #1615）専用の固定パス（`optim_rs_path` と
/// 同型）。
fn data_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/data.rs")
}

/// `src/data.rs` の `pub use` 行から `{...}` 内の識別子を抽出し、
/// 昇格元公開面（`fandhe_ai_tensor_core::data`）と完全一致（過不足とも
/// fail）することを固定する（`optim_module_reexports_exactly_expected_
/// surface` と同型の検査）。
#[test]
fn data_module_reexports_exactly_expected_surface() {
    let path = data_rs_path();
    let content = read_to_string_or_panic(&path);

    let allowed_prefix = "pub use fandhe_ai_tensor_core::data::";

    let mut found: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut offending_lines = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("pub use") {
            continue;
        }
        if !trimmed.starts_with(allowed_prefix) {
            offending_lines.push(trimmed.to_string());
            continue;
        }
        let rest = &trimmed[allowed_prefix.len()..];
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
        "src/data.rs の pub use が昇格元公開面（fandhe_ai_tensor_core::data）以外の\
         接頭辞を持つか、`{{...}}` 形式でない行を含む: {offending_lines:?}"
    );

    let expected: std::collections::BTreeSet<String> = [
        "Batches",
        "DataError",
        "DataLoader",
        "DataLoaderConfig",
        "Dataset",
        "TensorDataset",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    assert_eq!(
        found, expected,
        "src/data.rs が再エクスポートする識別子が期待集合と一致しない\
         （過不足いずれも不可。昇格元公開面と 1 対 1 対応であることの固定）"
    );
}

/// `src/data.rs` が facade 独自の型・関数を定義しない純再エクスポート
/// モジュールであることを固定する（`optim_module_is_pure_reexport` と
/// 同じ走査ロジック〈`scan_forbidden_pub_items`〉を再利用する）。
#[test]
fn data_module_is_pure_reexport() {
    let path = data_rs_path();
    let content = read_to_string_or_panic(&path);
    let offending = scan_forbidden_pub_items(&content);
    assert!(
        offending.is_empty(),
        "src/data.rs が facade 独自の型・関数・impl・pub type/const/static/mod/union 等の\
         公開宣言を定義している（純再エクスポートモジュールの契約違反）: {offending:?}"
    );
}

/// `fandhe_ai::data` の全再エクスポート型が facade のみを通じて到達
/// 可能であることのコンパイル時固定（`optim_types_are_reachable_via_
/// facade_only` と同型。`fandhe_ai_tensor_core` は import しない）。
#[test]
fn data_types_are_reachable_via_facade_only() {
    let features = fandhe_ai::Tensor::<f32>::new(vec![0.0, 1.0, 2.0, 3.0], &[4, 1])
        .unwrap_or_else(|e| panic!("test fixture: features tensor の構築に失敗: {e}"));
    let labels = fandhe_ai::Tensor::<i32>::new(vec![0, 1, 0, 1], &[4])
        .unwrap_or_else(|e| panic!("test fixture: labels tensor の構築に失敗: {e}"));

    let dataset = (
        fandhe_ai::data::TensorDataset::new(features)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}")),
        fandhe_ai::data::TensorDataset::new(labels)
            .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗した: {e}")),
    );
    let config = fandhe_ai::data::DataLoaderConfig::new(2)
        .shuffle(false)
        .drop_last(false);
    let loader = fandhe_ai::data::DataLoader::new(dataset, config)
        .unwrap_or_else(|e| panic!("test fixture: DataLoader::new が失敗した: {e}"));

    assert_eq!(loader.len(), 2);
    let mut total = 0usize;
    let batches: fandhe_ai::data::Batches<'_, _> = loader.iter();
    for batch in batches {
        let (x, y): (fandhe_ai::Tensor<f32>, fandhe_ai::Tensor<i32>) =
            batch.unwrap_or_else(|e: fandhe_ai::data::DataError| {
                panic!("test fixture: batch が失敗した: {e}")
            });
        assert_eq!(x.shape()[0], y.shape()[0]);
        total += x.shape()[0];
    }
    assert_eq!(total, 4);

    // `Dataset` trait 自体も facade 経由で到達可能であることの固定
    // （`use fandhe_ai::data::Dataset;` なしでは `.len()`／`.batch()`
    // が呼べない）。
    fn assert_is_dataset<D: fandhe_ai::data::Dataset>(_d: &D) {}
    let probe = fandhe_ai::data::TensorDataset::new(fandhe_ai::Tensor::<f32>::zeros(&[2]).unwrap())
        .unwrap();
    assert_is_dataset(&probe);
}
