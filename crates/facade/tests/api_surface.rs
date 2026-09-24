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
//!
//! `var_does_not_implement_arithmetic_operator_traits_while_2136_on_hold`
//! は上記のソース走査型ガードとは別方式（型レベルの正のプローブ）で、
//! `Var`（`crate::var.rs`。facade からは `fandhe_ai::Var` として `lib.rs:184`
//! で再エクスポート済み）が `Add`／`Sub`／`Mul`／`Div`／`Neg` 等の演算子
//! トレイトを実装していないことを固定する（イシュー #2136 は承認待ちで
//! 保留。詳細は `docs/autodiff-var-operator-overload-design.md` §14）。

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
                // バックスラッシュ直後の 1 文字（エスケープ本体の先頭。
                // `\'`／`\\`／`\n`／`\u` 等）は、それが `'\''` の `\'` の
                // ように閉じクォートと同じ文字であっても判定せず無条件に
                // 1 文字消費する。これをしないと `'\''` の 2 文字目の `'`
                // を閉じクォートと誤認識し、直後の実際の閉じクォートから
                // 再同期してしまう（例: `('\'','"')` の `"` を文字列
                // リテラル開始と誤認識し後続コードを丸ごと呑み込む）。
                let mut k = i + 2;
                if k < len {
                    k += 1;
                }
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

/// `fandhe_ai::set_cuda_onnx_gpu_execution_enabled`／
/// `fandhe_ai::cuda_onnx_gpu_execution_enabled`（イシュー #2077）が
/// facade クレート root から到達可能であることのコンパイル時固定
/// （数値検証・実行時分岐は `tests/interop_onnx_gpu_execution_optin.rs`
/// が担う。本テストはプロセスグローバルフラグを変更しないよう、往復後
/// 必ず既定 `false` へ戻す）。
#[test]
fn cuda_onnx_gpu_execution_optin_is_reachable_via_facade() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let original = fandhe_ai::cuda_onnx_gpu_execution_enabled();
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(true);
    assert!(fandhe_ai::cuda_onnx_gpu_execution_enabled());
    fandhe_ai::set_cuda_onnx_gpu_execution_enabled(original);
}

/// Metal 版（macOS 限定）の同型固定。
#[cfg(target_os = "macos")]
#[test]
fn metal_onnx_gpu_execution_optin_is_reachable_via_facade() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let original = fandhe_ai::metal_onnx_gpu_execution_enabled();
    fandhe_ai::set_metal_onnx_gpu_execution_enabled(true);
    assert!(fandhe_ai::metal_onnx_gpu_execution_enabled());
    fandhe_ai::set_metal_onnx_gpu_execution_enabled(original);
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

/// facade（crates.io 公開クレート `fandhe-ai`）が
/// `onnx-interop`（公開名 `fandhe-ai-onnx-interop`）へ**承認済み依存形状
/// でのみ**依存していることを固定する（イシュー #2017）。
///
/// ONNX import（`OnnxModel`／`OnnxValue`／`OnnxError`。`src/interop/
/// onnx.rs`）は #2017 で公開済みのため、facade は onnx-interop への通常
/// 依存を正式に持つ。**本テストは「依存の有無」ではなく「承認された
/// 依存の形状（path・version 併記・取得元）から逸脱していないか」を
/// 検査する正のガード**（旧テスト `facade_does_not_depend_on_
/// unpublished_onnx_interop`——依存そのものを禁止する負のガードだった
/// ——を #2017 で本テストへ差し替えた。`docs/facade-onnx-import-exposure-
/// decision.md` §6.1 で予告済みの差し替え。safetensors save／load は
/// #2019 で facade 公開済み（`docs/facade-safetensors-exposure-
/// decision.md` §11）。ONNX export は依然として別途ユーザー承認が必要な
/// 段階 0 のまま（`docs/facade-onnx-export-exposure-decision.md`）。
///
/// `docs/crates-io-publishing-order.md` §6 は「公開クレートの
/// `[dependencies]` に facade ラッパー未承認のクレートが現れない」ことを
/// 実測確認済みの前提としているが、CI は `cargo publish --dry-run` を
/// 実行しないため、誰かが依存の取得元を `git = ...` へ差し替えたり
/// version 併記を外したりしても通常の `cargo build`／`cargo test` は
/// 成功してしまい、意図しない公開面の拡張・取得元差し替えが気付かれ
/// にくい。本テストはこの盲点を CI 時点の検出へ前倒しする（ホスト
/// クレート `Cargo.toml` のみを対象とする固定パス走査で、外部入力を
/// 受け取らない。`.claude/rules/security.md` A08）。
///
/// 承認済み依存形状の要件（すべて fail-closed）:
/// 1. `fandhe-ai-onnx-interop` は**素の `[dependencies]`**（コメント除去後
///    `[dependencies]` に完全一致するセクション）に**丁度 1 回**だけ現れる。
///    `[build-dependencies]`・`[target.'cfg(...)'.dependencies]`・
///    `[dependencies.<name>]` テーブル形式・別名（`onnx-interop`・
///    `fandhe-ai-interop`）・`package = "..."` によるリネーム迂回は
///    すべて違反。
/// 2. 依存行の値に `path = "../onnx-interop"` と、他の公開 path 依存
///    （`fandhe-ai-tensor-core`。基準値として先に走査する）と**同一の**
///    `version = "=x.y.z"` を含む（バンプ時のドリフト検出）。
/// 3. 依存行の値に `git`／`registry`／`branch`／`rev`／`tag`／`optional`／
///    `features` キーを含まない（取得元差し替え・feature フラグなし cfg
///    方針〈coding-rust.md〉との整合）。
///
/// `[dev-dependencies]` は対象外（`bench-harness` 等の既存
/// dev-dependency と同じ扱い。`cargo publish` を壊さない）。
const ONNX_INTEROP_DEPENDENCY_KEY: &str = "fandhe-ai-onnx-interop";
const ONNX_INTEROP_ALT_NAMES: [&str; 2] = ["onnx-interop", "fandhe-ai-interop"];
const APPROVED_PLAIN_DEPENDENCIES_SECTION: &str = "[dependencies]";
/// 依存行の値に現れてはならないキー（取得元差し替え・feature フラグ
/// 追加を検出する）。`" <key> "`／`"<key>="` の形で走査するため、他の
/// 単語の部分文字列に誤爆しない（例: "average" は "rev" を含まない
/// 連続部分列を持たない上、本チェックは区切り文字を伴う形でのみ一致
/// する）。
const FORBIDDEN_DEPENDENCY_VALUE_KEYS: [&str; 6] =
    ["git", "registry", "branch", "rev", "tag", "optional"];

fn is_dev_dependencies_section(section: &str) -> bool {
    section.contains("dev-dependencies")
}

fn is_relevant_dependency_section(section: &str) -> bool {
    section.contains("dependencies") && !is_dev_dependencies_section(section)
}

/// `value`（`{ path = "...", version = "=x.y.z" }` 形の依存値）から
/// `version` キーのクォート内リテラルを抽出する。見つからなければ
/// `None`（version 指定なしとして扱う）。
fn extract_version_literal(value: &str) -> Option<String> {
    let after_key = value.split_once("version")?.1;
    let after_eq = after_key.split_once('=')?.1;
    let after_first_quote = after_eq.split_once('"')?.1;
    let (literal, _) = after_first_quote.split_once('"')?;
    Some(literal.to_string())
}

/// 値の中に禁止キー（`FORBIDDEN_DEPENDENCY_VALUE_KEYS`）・`features` が
/// TOML インライン table のキーとして現れるかを検査する。
///
/// `{ optional = true }` のような各エントリを `,` で分割し、`=` より
/// 前のキー部分だけを取り出して比較する字句解析方式を取る。単純な
/// 部分文字列検査（`" key "` / `"key ="` 形の固定パターン照合）では、
/// 空白を伴わない `optional=true`・タブ区切り・`"optional" = true` の
/// ような引用符付きキーで一致せず検査を回避できてしまうため（codex-review
/// 指摘 P1・#2024）、キー部分をトリム＋引用符除去したうえで完全一致で
/// 比較する。
fn value_contains_forbidden_key(value: &str, key: &str) -> bool {
    value.split(',').any(|segment| {
        let segment = segment.trim().trim_start_matches('{').trim_end_matches('}');
        match segment.split_once('=') {
            Some((k, _)) => k.trim().trim_matches('"').trim_matches('\'') == key,
            None => false,
        }
    })
}

/// `value`（インライン table 形の依存値）中に `package` キーが現れ、その
/// 値（クォートを除去した後の文字列）が `names` のいずれかと完全一致する
/// かを検査する（`package` によるクレート名リネーム迂回の検出）。
///
/// TOML はダブルクォート文字列（`"..."`）とシングルクォートのリテラル
/// 文字列（`'...'`）の両方を許容する。旧実装（`value.contains(&format!("\"{name}\""))`）
/// はダブルクォートのみを対象としていたため、`package = 'fandhe-ai-onnx-interop'`
/// のようなシングルクォート表記で検査を回避できてしまっていた
/// （codex-review 指摘 P1・#2024 2 回目レビュー）。`value_contains_forbidden_key`
/// と同じキー／値の字句分割方式を用い、クォート種別に依存せず判定する。
fn value_has_package_rename_to(value: &str, names: &[&str]) -> bool {
    value.split(',').any(|segment| {
        let segment = segment.trim().trim_start_matches('{').trim_end_matches('}');
        match segment.split_once('=') {
            Some((k, v)) => {
                let k = k.trim().trim_matches('"').trim_matches('\'');
                if k != "package" {
                    return false;
                }
                let v = v.trim().trim_matches('"').trim_matches('\'');
                names.contains(&v)
            }
            None => false,
        }
    })
}

/// `facade_depends_on_onnx_interop_only_in_approved_shape` の走査ロジック
/// 本体。実 `Cargo.toml` からの呼び出しと、パース境界（コメント付き
/// ヘッダ行等）を固定する合成入力からの呼び出しの両方から使えるよう、
/// ファイル読み込みと判定ロジックを分離した（#1775 PR #1821
/// codex-review 指摘の回帰テスト
/// `dependency_section_header_with_trailing_comment_is_recognized` 用の
/// 構造を #2017 でも踏襲する）。戻り値は
/// `(offending, saw_known_dependency_line)`。
fn scan_onnx_dependency_shape(content: &str) -> (Vec<String>, bool) {
    // 1 パス目: バンプ時ドリフト検出の基準となる `fandhe-ai-tensor-core`
    // の version リテラルを、素の `[dependencies]` セクション内から
    // 先に集める（出現順序に依存しないようにする）。
    let mut approved_version: Option<String> = None;
    {
        let mut section = String::new();
        for line in content.lines() {
            let trimmed = line.trim();
            let code_part = trimmed
                .split_once('#')
                .map(|(a, _)| a)
                .unwrap_or(trimmed)
                .trim();
            if code_part.starts_with('[') && code_part.ends_with(']') {
                section = code_part.to_string();
                continue;
            }
            if section != APPROVED_PLAIN_DEPENDENCIES_SECTION {
                continue;
            }
            if let Some((key, value)) = code_part.split_once('=')
                && key.trim().trim_matches('"') == "fandhe-ai-tensor-core"
            {
                approved_version = extract_version_literal(value);
            }
        }
    }

    let mut current_section = String::new();
    let mut offending = Vec::new();
    let mut saw_known_dependency_line = false;
    let mut plain_dependency_occurrences = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        // コメント除去（`#` 以降）。ヘッダ判定もコメント除去後の
        // `code_part` に対して行う理由は #1775 PR #1821 の codex-review
        // 指摘（コメント付きヘッダ行の見落とし）と同じ。
        let code_part = trimmed
            .split_once('#')
            .map(|(a, _)| a)
            .unwrap_or(trimmed)
            .trim();
        if code_part.starts_with('[') && code_part.ends_with(']') {
            current_section = code_part.to_string();
            if is_relevant_dependency_section(&current_section)
                && current_section != APPROVED_PLAIN_DEPENDENCIES_SECTION
            {
                let mentions_target = current_section.contains(ONNX_INTEROP_DEPENDENCY_KEY)
                    || ONNX_INTEROP_ALT_NAMES
                        .iter()
                        .any(|name| current_section.contains(name));
                if mentions_target {
                    offending.push(format!(
                        "section header `{current_section}` が onnx-interop への言及を含む\
                         （承認形状は素の `{APPROVED_PLAIN_DEPENDENCIES_SECTION}` のみ）"
                    ));
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

        if key == "fandhe-ai-tensor-core" || key == "fandhe-ai-autodiff" {
            saw_known_dependency_line = true;
        }

        let is_target_key = key == ONNX_INTEROP_DEPENDENCY_KEY;
        // 別名キー（`onnx-interop = { .. }` のように prefix なしでキー
        // そのものが別名）の検出。`path = "../onnx-interop"` のように
        // 値の一部として正当に "onnx-interop" 文字列が現れる（依存先
        // ディレクトリ名）ケースと区別するため、値側の別名検出は
        // `package = "..."` キーによるリネーム迂回（クォート内完全一致）
        // に限定する（単純な部分文字列一致は path 値を誤検出するため
        // 不採用）。
        let is_alt_key = ONNX_INTEROP_ALT_NAMES.contains(&key);
        let package_rename_names: Vec<&str> = ONNX_INTEROP_ALT_NAMES
            .iter()
            .copied()
            .chain(std::iter::once(ONNX_INTEROP_DEPENDENCY_KEY))
            .collect();
        let value_has_package_rename = value_has_package_rename_to(value, &package_rename_names);
        // テーブル形式（`[dependencies.<alias>]`／`[build-dependencies.alias]`
        // 配下の独立行としての `package = "..."`）による迂回の検出。
        // インライン table（`{ package = "..." }`）内の package キーは
        // `value_has_package_rename_to` が検出するが、テーブル形式では
        // `package = "..."` 自体が独立した `key = value` 行になり、
        // value 側に更なる `=` が現れないため `value_has_package_rename_to`
        // の分割ロジック（value を `,` 分割し各セグメントを `=` で再分割
        // してキーを取り出す方式）では検出できずすり抜ける（codex-review
        // 指摘 P1・#2024）。ここでは行の key 自体が `package` であり、
        // value（クォート除去後）が onnx-interop 名のいずれかと一致する
        // ケースを直接判定する。
        let top_level_package_rename = key == "package" && {
            let v = value.trim().trim_matches('"').trim_matches('\'');
            package_rename_names.contains(&v)
        };
        let has_package_rename = value_has_package_rename || top_level_package_rename;
        if !is_target_key && !is_alt_key && !has_package_rename {
            continue;
        }

        if current_section != APPROVED_PLAIN_DEPENDENCIES_SECTION {
            offending.push(format!(
                "`{current_section}` の行 `{trimmed}` は承認形状外のセクションにある\
                 （素の `{APPROVED_PLAIN_DEPENDENCIES_SECTION}` 以外は禁止）"
            ));
            continue;
        }
        if is_alt_key {
            offending.push(format!(
                "`{trimmed}` は別名キー `{key}` による onnx-interop 依存（承認名は \
                 `{ONNX_INTEROP_DEPENDENCY_KEY}` のみ）"
            ));
        }
        if has_package_rename {
            offending.push(format!(
                "`{trimmed}` の値に `package` によるクレート名リネームを検出\
                 （迂回の疑い）"
            ));
        }
        if !is_target_key {
            continue;
        }

        plain_dependency_occurrences += 1;
        if !value.contains("path = \"../onnx-interop\"") {
            offending.push(format!(
                "`{trimmed}` に承認済み path `path = \"../onnx-interop\"` がない"
            ));
        }
        match (extract_version_literal(value), &approved_version) {
            (Some(v), Some(approved)) if &v == approved => {}
            (Some(v), Some(approved)) => offending.push(format!(
                "`{trimmed}` の version `{v}` が fandhe-ai-tensor-core の version \
                 `{approved}` と不一致（バンプ漏れの疑い）"
            )),
            (None, _) => offending.push(format!("`{trimmed}` に version 指定がない")),
            (_, None) => offending.push(
                "基準となる fandhe-ai-tensor-core の version 行が見つからなかった\
                 （自己検証の失敗）"
                    .to_string(),
            ),
        }
        for forbidden_key in FORBIDDEN_DEPENDENCY_VALUE_KEYS {
            if value_contains_forbidden_key(value, forbidden_key) {
                offending.push(format!("`{trimmed}` に禁止キー `{forbidden_key}` を検出"));
            }
        }
        if value_contains_forbidden_key(value, "features") {
            offending.push(format!(
                "`{trimmed}` に `features` キーを検出（feature フラグなし cfg 方針との不整合）"
            ));
        }
    }

    if plain_dependency_occurrences == 0 {
        offending.push(format!(
            "`{APPROVED_PLAIN_DEPENDENCIES_SECTION}` に `{ONNX_INTEROP_DEPENDENCY_KEY}` が見つからない"
        ));
    } else if plain_dependency_occurrences > 1 {
        offending.push(format!(
            "`{APPROVED_PLAIN_DEPENDENCIES_SECTION}` に `{ONNX_INTEROP_DEPENDENCY_KEY}` が\
             {plain_dependency_occurrences} 回出現（丁度 1 回のはず）"
        ));
    }

    (offending, saw_known_dependency_line)
}

#[test]
fn facade_depends_on_onnx_interop_only_in_approved_shape() {
    let cargo_toml_path = facade_crate_root().join("Cargo.toml");
    let content = read_to_string_or_panic(&cargo_toml_path);

    let (offending, saw_known_dependency_line) = scan_onnx_dependency_shape(&content);

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
        "facade（crates.io 公開クレート）の onnx-interop 依存が承認済み形状から\
         逸脱している（意図しない公開面拡張・取得元差し替えの検出。\
         `docs/facade-onnx-import-exposure-decision.md` 参照）: {offending:?}"
    );
}

/// 承認形状（`path = "../onnx-interop"`・version 併記・素の
/// `[dependencies]`）の合成入力は違反として検出されないことを確認する
/// （`facade_depends_on_onnx_interop_only_in_approved_shape` の pass 経路
/// 自体の回帰固定）。
#[test]
fn approved_shape_synthetic_input_is_not_flagged() {
    let synthetic_cargo_toml = r#"
[package]
name = "fandhe-ai"

[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }
fandhe-ai-autodiff = { path = "../autodiff", version = "=0.9.0" }
fandhe-ai-onnx-interop = { path = "../onnx-interop", version = "=0.9.0" }
"#;

    let (offending, saw_known_dependency_line) = scan_onnx_dependency_shape(synthetic_cargo_toml);

    assert!(
        saw_known_dependency_line,
        "自己検証: 合成入力の [dependencies] セクションが走査されなかった"
    );
    assert!(
        offending.is_empty(),
        "承認形状の合成入力が誤って違反として検出された: {offending:?}"
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

    let (offending, saw_known_dependency_line) = scan_onnx_dependency_shape(synthetic_cargo_toml);

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

/// 承認済み version（`fandhe-ai-tensor-core` と同一）を外した・ドリフト
/// させた合成入力が違反として検出されることを確認する（要件 2 の固定）。
#[test]
fn version_drift_or_missing_version_is_flagged() {
    let missing_version = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }
fandhe-ai-onnx-interop = { path = "../onnx-interop" }
"#;
    let (offending, _) = scan_onnx_dependency_shape(missing_version);
    assert!(
        offending.iter().any(|e| e.contains("version 指定がない")),
        "version 指定なしが検出されなかった: {offending:?}"
    );

    let drifted_version = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }
fandhe-ai-onnx-interop = { path = "../onnx-interop", version = "=0.8.0" }
"#;
    let (offending, _) = scan_onnx_dependency_shape(drifted_version);
    assert!(
        offending.iter().any(|e| e.contains("不一致")),
        "version ドリフトが検出されなかった: {offending:?}"
    );
}

/// 取得元差し替え（`git = ...`）が違反として検出されることを確認する
/// （要件 3 の固定）。
#[test]
fn git_source_override_is_flagged() {
    let git_override = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }
fandhe-ai-onnx-interop = { git = "https://example.invalid/onnx-interop", version = "=0.9.0" }
"#;
    let (offending, _) = scan_onnx_dependency_shape(git_override);
    assert!(
        offending.iter().any(|e| e.contains("禁止キー `git`")),
        "git 取得元差し替えが検出されなかった: {offending:?}"
    );
}

/// `package = "..."` によるクレート名リネーム迂回がダブルクォート・
/// シングルクォート（TOML リテラル文字列）のいずれでも検出されることを
/// 確認する（codex-review 指摘 P1・#2024 2 回目レビューの回帰固定。旧
/// 実装はダブルクォートのみを対象としシングルクォート表記ですり抜け
/// 可能だった）。
#[test]
fn package_rename_bypass_is_flagged_regardless_of_quote_style() {
    let double_quoted = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }

[build-dependencies]
alias = { package = "fandhe-ai-onnx-interop", path = "../onnx-interop", version = "=0.9.0" }
"#;
    let (offending, _) = scan_onnx_dependency_shape(double_quoted);
    assert!(
        offending.iter().any(|e| e.contains("package")),
        "ダブルクォートの package リネームが検出されなかった: {offending:?}"
    );

    let single_quoted = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }

[build-dependencies]
alias = { package = 'fandhe-ai-onnx-interop', path = '../onnx-interop', version = '=0.9.0' }
"#;
    let (offending, _) = scan_onnx_dependency_shape(single_quoted);
    assert!(
        offending.iter().any(|e| e.contains("package")),
        "シングルクォート（TOML リテラル文字列）の package リネームが\
         検出されなかった: {offending:?}"
    );
}

/// テーブル形式の依存宣言下に独立行として現れる `package = "..."`
/// （インライン table の外側。例: `[build-dependencies.alias]` セクション
/// 配下の `package = "fandhe-ai-onnx-interop"` 行）による package
/// リネーム迂回が検出されることを確認する（codex-review 指摘 P1・#2024
/// の回帰固定。旧実装は `value_has_package_rename_to` がインライン
/// table〈`{ package = "..." }`〉のみを対象としており、テーブル形式
/// 配下の独立 `package = "..."` 行は value 側に `=` を含まないため
/// キー抽出ロジックが素通りしてしまい、承認済み通常依存〈本テストの
/// `fandhe-ai-tensor-core`〉を残したまま検出をすり抜けられた）。
#[test]
fn table_form_package_rename_bypass_is_flagged() {
    let table_form_rename = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }

[build-dependencies.alias]
package = "fandhe-ai-onnx-interop"
path = "../onnx-interop"
version = "=0.9.0"
"#;
    let (offending, saw_known_dependency_line) = scan_onnx_dependency_shape(table_form_rename);
    assert!(
        saw_known_dependency_line,
        "自己検証: 合成入力の [dependencies] セクションが走査されなかった"
    );
    // `offending` の非空性だけを見ると、この合成入力が承認済み
    // `fandhe-ai-onnx-interop` の素の `[dependencies]` エントリを含まない
    // ため「依存が見つからない」という無関係なオフェンスだけでも常に
    // 非空になり、テーブル形式 package リネーム検出自体が後退しても
    // 本テストが green のまま残る空虚検査になる（Cursor Bugbot Medium
    // 指摘・#2024。兄弟テスト
    // `package_rename_bypass_is_flagged_regardless_of_quote_style` と
    // 同様にメッセージへ `package` を含むオフェンスの存在を直接検査する）。
    assert!(
        offending.iter().any(|e| e.contains("package")),
        "テーブル形式配下の独立 `package = \"fandhe-ai-onnx-interop\"` 行による \
         リネーム迂回が検出されなかった（すり抜け再発）: {offending:?}"
    );

    // シングルクォート（TOML リテラル文字列）でも同様に検出されることを
    // 確認する（インライン table 側の既存回帰
    // `package_rename_bypass_is_flagged_regardless_of_quote_style` と
    // 同じ観点をテーブル形式へ拡張）。
    let table_form_rename_single_quoted = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }

[build-dependencies.alias]
package = 'fandhe-ai-onnx-interop'
path = '../onnx-interop'
version = '=0.9.0'
"#;
    let (offending, _) = scan_onnx_dependency_shape(table_form_rename_single_quoted);
    assert!(
        offending.iter().any(|e| e.contains("package")),
        "テーブル形式配下の独立 `package = 'fandhe-ai-onnx-interop'`（シングル\
         クォート）行によるリネーム迂回が検出されなかった: {offending:?}"
    );
}

/// `[dependencies.<name>]` テーブル形式による迂回が違反として検出される
/// ことを確認する（要件 1 の固定）。
#[test]
fn dependencies_table_form_is_flagged() {
    let table_form = r#"
[dependencies]
fandhe-ai-tensor-core = { path = "../tensor-core", version = "=0.9.0" }

[dependencies.fandhe-ai-onnx-interop]
path = "../onnx-interop"
version = "=0.9.0"
"#;
    let (offending, _) = scan_onnx_dependency_shape(table_form);
    assert!(
        offending
            .iter()
            .any(|e| e.contains("dependencies.fandhe-ai-onnx-interop")),
        "[dependencies.<name>] テーブル形式が検出されなかった: {offending:?}"
    );
}

/// facade の `src/` のうち `src/interop/` 配下以外が `onnx-interop`
/// （クレート名を Rust 識別子化した `onnx_interop`）を `use`／型パス等で
/// 参照していないことを固定する（上記 Cargo.toml 側ガードの対を成す
/// src 側チェック。#1775 の負ガードを #2017 で承認モジュール限定の正
/// ガードへ差し替え）。`src/interop/` 自体は onnx-interop 型を扱う唯一の
/// 承認済みモジュールのため対象外とする。
#[test]
fn facade_sources_reference_onnx_interop_only_in_interop_module() {
    let src_dir = facade_crate_root().join("src");
    let interop_dir = src_dir.join("interop");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        if path.starts_with(&interop_dir) {
            return;
        }
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
        "facade の src/interop/ 以外が onnx-interop を参照している\
         （onnx-interop 型は承認済みモジュール src/interop/ 配下に閉じ込める設計。\
         #2017）: {offending:?}"
    );
}

/// `src/interop/onnx.rs` に承認範囲外の公開アイテム（`pub struct`／
/// `pub enum`／`pub fn`／`pub trait`／`pub type`／`pub const`／
/// `pub static`／`pub mod`）が存在しないかを走査する。承認範囲は
/// `interop::onnx::{OnnxModel, OnnxValue, OnnxError, OnnxExportOptions}`
/// と `OnnxModel::{from_bytes, from_path, run, to_bytes, to_path,
/// from_sequential}` の 10 件のみ（イシュー #2017・#2018・#2037）。
///
/// `interop_module_exposes_only_approved_onnx_surface` の従来実装は
/// 期待する 6 文字列が「存在すること」の contains 検査のみで、
/// `OnnxModel::to_bytes` のような承認外の追加公開メソッドが増えても
/// 検出できなかった（codex-review 指摘 P2・#2024）。本関数は逆方向
/// （ファイル中の全 `pub` 宣言を列挙し、承認リストの外側にあるものを
/// 検出する）で拒否側のガードを担う。`pub(crate)`／`pub(super)` 等の
/// 限定可視性は crate 外へ公開されないため対象外。
///
/// `pub use`（再エクスポート）も走査対象に含める（codex-review 指摘
/// P2・#2024 2 回目レビュー。`onnx.rs` の承認範囲は 6 件の型定義のみで
/// 再エクスポートは 1 件も承認していないため、`kind == "use"` を
/// `ALLOWED_PUB_ITEMS` に一致し得ない種別として扱うだけで
/// `pub use fandhe_ai_onnx_interop::...` のような追加を機構的に拒否
/// できる）。`async`／`unsafe`／`extern` 修飾子付き宣言
/// （`pub async fn`／`pub unsafe fn` 等）も `pub` 直後の識別子を種別と
/// 誤認せず読み飛ばしたうえで種別を判定する（Cursor Bugbot Low 指摘・
/// #2024。`scan_forbidden_pub_items` の同種修飾子スキップと同じ方針）。
///
/// `pub fn` については、パラメータ列・戻り値型を含む完全なシグネチャ
/// （`fn` から本体開始 `{` または宣言終端 `;` まで）を抽出し、
/// `FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` のいずれかを含んでいないかも
/// 検査する。旧実装（`interop_module_exposes_only_approved_onnx_surface`
/// 内の行単位 `pub` 行検査）は複数行にまたがる戻り値型宣言で内部型
/// （`ModelProto` 等）の露出を見逃していた（codex-review 指摘 P2・
/// #2024。本関数へ統合し単一の走査でシグネチャ全体を検査する）。
fn scan_unapproved_onnx_pub_items(original: &str) -> Vec<String> {
    const ALLOWED_PUB_ITEMS: [(&str, &str); 10] = [
        ("struct", "OnnxModel"),
        ("enum", "OnnxValue"),
        ("enum", "OnnxError"),
        ("struct", "OnnxExportOptions"),
        ("fn", "from_bytes"),
        ("fn", "from_path"),
        ("fn", "run"),
        ("fn", "to_bytes"),
        ("fn", "to_path"),
        ("fn", "from_sequential"),
    ];
    const SCANNED_KINDS: [&str; 9] = [
        "struct", "enum", "fn", "trait", "type", "const", "static", "mod", "use",
    ];
    const QUALIFIER_KEYWORDS: [&str; 3] = ["async", "unsafe", "extern"];
    const FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS: [&str; 10] = [
        "ModelProto",
        "NodeProto",
        "prost::",
        "onnx::graph::Graph",
        "ExportError",
        "onnx::export::",
        "export_nn",
        "ExportNode",
        "NnExportParts",
        "nn::Module",
    ];

    let cleaned = strip_comments_and_literals(original);
    let len = cleaned.len();
    let mut offenses = Vec::new();
    let mut i = 0usize;
    while i < len {
        if !is_ident_start(cleaned[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1;
        while j < len && is_ident_char(cleaned[j]) {
            j += 1;
        }
        let word: String = cleaned[start..j].iter().collect();
        if word != "pub" {
            i = j;
            continue;
        }
        let mut k = j;
        while k < len && cleaned[k].is_whitespace() {
            k += 1;
        }
        if k < len && cleaned[k] == '(' {
            // `pub(crate)`／`pub(super)` 等。crate 外非公開のため
            // スキップする（括弧の対応を数えて閉じ括弧まで読み飛ばす）。
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
            i = k;
            continue;
        }
        // `async`／`unsafe`／`extern` 修飾子を種別と誤認しないよう
        // 読み飛ばす（`pub async fn` 等）。
        loop {
            if !(k < len && is_ident_start(cleaned[k])) {
                break;
            }
            let qs = k;
            let mut qe = k + 1;
            while qe < len && is_ident_char(cleaned[qe]) {
                qe += 1;
            }
            let candidate: String = cleaned[qs..qe].iter().collect();
            if !QUALIFIER_KEYWORDS.contains(&candidate.as_str()) {
                break;
            }
            k = qe;
            while k < len && cleaned[k].is_whitespace() {
                k += 1;
            }
            // `extern "C"` のような文字列リテラルはコメント除去段階で
            // 空白へ置換済みのため追加処理は不要。
        }
        if !(k < len && is_ident_start(cleaned[k])) {
            i = j;
            continue;
        }
        let ks = k;
        let mut ke = k + 1;
        while ke < len && is_ident_char(cleaned[ke]) {
            ke += 1;
        }
        let kind: String = cleaned[ks..ke].iter().collect();
        if !SCANNED_KINDS.contains(&kind.as_str()) {
            i = j;
            continue;
        }
        if kind == "use" {
            // `onnx.rs` の承認範囲は型定義 6 件のみで再エクスポートは
            // 0 件のため、`pub use` は再エクスポート先のパスが
            // 識別子で始まらない形（`pub use {a, b};` のグループ化・
            // `pub use ::krate::...;` の先頭 `::` 等）でも取りこぼさず
            // 無条件にオフェンスとして記録する（advisor 指摘。下の
            // 識別子抽出ブロックは `use` 以降が識別子で始まる場合しか
            // 拾えず、その前提が崩れる形を素通りさせてしまうため）。
            offenses.push(format!(
                "line {}: `pub use` は onnx.rs で承認されていない再エクスポート",
                line_at(&cleaned, start)
            ));
            i = j;
            continue;
        }
        let mut m = ke;
        while m < len && cleaned[m].is_whitespace() {
            m += 1;
        }
        if m < len && is_ident_start(cleaned[m]) {
            let ns = m;
            let mut ne = m + 1;
            while ne < len && is_ident_char(cleaned[ne]) {
                ne += 1;
            }
            let name: String = cleaned[ns..ne].iter().collect();
            let approved = ALLOWED_PUB_ITEMS
                .iter()
                .any(|(k2, n2)| *k2 == kind.as_str() && *n2 == name.as_str());
            if !approved {
                offenses.push(format!(
                    "line {}: `pub {kind} {name}` は承認範囲外の公開アイテム",
                    line_at(&cleaned, start)
                ));
            }
            // `pub fn` は承認済みのものも含め、パラメータ列・戻り値型を
            // 含む完全なシグネチャを走査し、複数行にまたがる戻り値型
            // 宣言中の内部型露出（`ModelProto` 等）を検出する。
            if kind == "fn" {
                let mut p = ne;
                while p < len && cleaned[p] != '(' && cleaned[p] != '{' && cleaned[p] != ';' {
                    p += 1;
                }
                if p < len && cleaned[p] == '(' {
                    let mut depth = 1i32;
                    p += 1;
                    while p < len && depth > 0 {
                        match cleaned[p] {
                            '(' => depth += 1,
                            ')' => depth -= 1,
                            _ => {}
                        }
                        p += 1;
                    }
                }
                while p < len && cleaned[p] != '{' && cleaned[p] != ';' {
                    p += 1;
                }
                let signature: String = cleaned[start..p.min(len)].iter().collect();
                for forbidden in FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS {
                    if signature.contains(forbidden) {
                        offenses.push(format!(
                            "line {}: `pub fn {name}` のシグネチャが内部クレート型 \
                             `{forbidden}` を含む（複数行の戻り値型宣言を含む）",
                            line_at(&cleaned, start)
                        ));
                    }
                }
            }
        }
        i = j;
    }
    offenses
}

/// `scan_unapproved_onnx_pub_items` が承認範囲外の `pub fn`（例:
/// `OnnxModel::input_names`）を検出することを確認する（codex-review
/// 指摘 P2・#2024 の回帰固定。`to_bytes` は #2018 で承認範囲へ追加された
/// ため合成例からは差し替えている）。
#[test]
fn unapproved_onnx_pub_fn_is_flagged() {
    let synthetic = r#"
pub struct OnnxModel {
    graph: (),
}

impl OnnxModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        unimplemented!()
    }

    pub fn input_names(&self) -> Vec<String> {
        unimplemented!()
    }
}
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses.iter().any(|e| e.contains("input_names")),
        "承認範囲外の `pub fn input_names` が検出されなかった: {offenses:?}"
    );
}

/// `scan_unapproved_onnx_pub_items` が `pub use` 再エクスポート（通常の
/// パス形・グループ化形・先頭 `::` 形のいずれも）を承認範囲外として
/// 検出することを確認する（codex-review 指摘 P2・#2024 の回帰固定。
/// `onnx.rs` の承認範囲は型定義 6 件のみで再エクスポートは 0 件のため、
/// `pub use` はいかなる形でも許容されない）。グループ化形
/// （`pub use {a, b};`）・先頭 `::` 形（`pub use ::krate::...;`）は
/// `use` 直後が識別子で始まらないため、名前抽出（`is_ident_start`
/// 前提）に依存する検出だと素通りしうる（advisor 指摘）。`kind == "use"`
/// を検出した時点で無条件にオフェンスを記録する fail-closed 実装に
/// よってこれらも検出できることを併せて固定する。
#[test]
fn unapproved_onnx_pub_use_reexport_is_flagged() {
    let normal_path = r#"
pub struct OnnxModel {
    graph: (),
}

pub use fandhe_ai_onnx_interop::onnx::proto::encode_model;
"#;
    let offenses = scan_unapproved_onnx_pub_items(normal_path);
    assert!(
        offenses.iter().any(|e| e.contains("pub use")),
        "承認範囲外の `pub use` 再エクスポート（通常のパス形）が検出されなかった: \
         {offenses:?}"
    );

    let grouped = r#"
pub struct OnnxModel {
    graph: (),
}

pub use {fandhe_ai_onnx_interop::onnx::proto::encode_model};
"#;
    let offenses = scan_unapproved_onnx_pub_items(grouped);
    assert!(
        offenses.iter().any(|e| e.contains("pub use")),
        "承認範囲外の `pub use` 再エクスポート（グループ化形）が検出されなかった: \
         {offenses:?}"
    );

    let leading_colon = r#"
pub struct OnnxModel {
    graph: (),
}

pub use ::fandhe_ai_onnx_interop::onnx::proto::encode_model;
"#;
    let offenses = scan_unapproved_onnx_pub_items(leading_colon);
    assert!(
        offenses.iter().any(|e| e.contains("pub use")),
        "承認範囲外の `pub use` 再エクスポート（先頭 `::` 形）が検出されなかった: \
         {offenses:?}"
    );
}

/// `scan_unapproved_onnx_pub_items` が `async`／`unsafe` 修飾子付きの
/// `pub fn`（`pub async fn`／`pub unsafe fn`）も種別を正しく `fn` と
/// 判定して承認リストと照合することを確認する（Cursor Bugbot Low
/// 指摘・#2024。修飾子を種別と誤認してスキップすると、これらの宣言が
/// 検査を素通りしてしまう）。
#[test]
fn unapproved_onnx_pub_fn_with_qualifiers_is_flagged() {
    let synthetic = r#"
pub struct OnnxModel {
    graph: (),
}

impl OnnxModel {
    pub async fn to_bytes_async(&self) -> Vec<u8> {
        unimplemented!()
    }

    pub unsafe fn to_bytes_unsafe(&self) -> Vec<u8> {
        unimplemented!()
    }
}
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses.iter().any(|e| e.contains("to_bytes_async")),
        "承認範囲外の `pub async fn` が検出されなかった: {offenses:?}"
    );
    assert!(
        offenses.iter().any(|e| e.contains("to_bytes_unsafe")),
        "承認範囲外の `pub unsafe fn` が検出されなかった: {offenses:?}"
    );
}

/// `scan_unapproved_onnx_pub_items` が複数行にまたがる `pub fn` の
/// 戻り値型宣言中の内部クレート型露出（`ModelProto` 等）を検出する
/// ことを確認する（codex-review 指摘 P2・#2024 3 回目レビュー。旧
/// 実装の行単位 `pub` 行検査は戻り値型が改行を挟むと見逃していた）。
#[test]
fn multiline_return_type_internal_leak_is_flagged() {
    let synthetic = r#"
pub struct OnnxModel {
    graph: (),
}

impl OnnxModel {
    pub fn from_bytes(
        bytes: &[u8],
    ) -> Result<
        ModelProto,
        OnnxError,
    > {
        unimplemented!()
    }
}
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses
            .iter()
            .any(|e| e.contains("from_bytes") && e.contains("ModelProto")),
        "複数行の戻り値型に含まれる内部型 `ModelProto` の露出が検出されなかった: \
         {offenses:?}"
    );
}

/// `scan_unapproved_onnx_pub_items` が承認範囲 10 件（`OnnxModel`／
/// `OnnxValue`／`OnnxError`／`OnnxExportOptions` の型定義 4 件と
/// `OnnxModel::{from_bytes, from_path, run, to_bytes, to_path,
/// from_sequential}` のメソッド 6 件）をすべて含む合成ソースに対して
/// オフェンス 0 件を返すことを確認する（空虚 pass 防止。承認範囲の
/// 拡張〈#2018・#2037〉自体が正しく反映されていることの正例テスト）。
#[test]
fn approved_onnx_surface_yields_no_offenses() {
    let synthetic = r#"
pub struct OnnxModel {
    graph: (),
}

impl OnnxModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, OnnxError> {
        unimplemented!()
    }

    pub fn from_path(path: &str) -> Result<Self, OnnxError> {
        unimplemented!()
    }

    pub fn run(&self) -> Result<(), OnnxError> {
        unimplemented!()
    }

    pub fn to_bytes(&self, options: &OnnxExportOptions) -> Result<Vec<u8>, OnnxError> {
        unimplemented!()
    }

    pub fn to_path(&self, path: &str, options: &OnnxExportOptions) -> Result<(), OnnxError> {
        unimplemented!()
    }

    pub fn from_sequential(model: &Sequential) -> Result<Self, OnnxError> {
        unimplemented!()
    }
}

pub enum OnnxValue {
    F32(()),
}

pub enum OnnxError {
    Io(()),
}

pub struct OnnxExportOptions {
    pub ir_version: i64,
    pub opset_version: i64,
}
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses.is_empty(),
        "承認範囲 10 件のみの合成ソースでオフェンスが検出された（空虚 pass 防止\
         テストの前提が崩れている）: {offenses:?}"
    );
}

/// `scan_unapproved_onnx_pub_items` が承認範囲外の追加 `pub fn`
/// （`from_layers`）と、`from_sequential` の引数型・戻り値型に内部型
/// （`&[Box<dyn fandhe_ai_autodiff::nn::Module>]`／`ExportError`）が
/// 紛れ込んだ場合の両方を検出することを確認する（イシュー #2037・
/// 承認外追加の機械拒否）。
#[test]
fn unapproved_onnx_export_pub_items_remain_flagged() {
    let synthetic = r#"
pub struct OnnxModel {
    graph: (),
}

impl OnnxModel {
    pub fn from_layers(layers: &[Box<dyn fandhe_ai_autodiff::nn::Module>]) -> Result<Self, ExportError> {
        unimplemented!()
    }

    pub fn from_sequential(
        layers: &[Box<dyn fandhe_ai_autodiff::nn::Module>],
    ) -> Result<Self, ExportError> {
        unimplemented!()
    }
}
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses.iter().any(|o| o.contains("from_layers")),
        "承認範囲外の `pub fn from_layers` が検出されなかった: {offenses:?}"
    );
    assert!(
        offenses
            .iter()
            .any(|o| o.contains("from_sequential") && o.contains("nn::Module")),
        "`from_sequential` の引数型に内部型 `nn::Module` が含まれる違反が\
         検出されなかった: {offenses:?}"
    );
    assert!(
        offenses
            .iter()
            .any(|o| o.contains("from_sequential") && o.contains("ExportError")),
        "`from_sequential` の戻り値型に内部型 `ExportError` が含まれる違反が\
         検出されなかった: {offenses:?}"
    );
}

/// `src/interop/` の公開面が承認範囲（`interop::onnx::{OnnxModel,
/// OnnxValue, OnnxError}` と `OnnxModel::{from_bytes, from_path, run}`）
/// のみであることを固定する。`prost`・`onnx-interop` の内部型
/// （`ModelProto`／`NodeProto`／`Graph` 等）が `pub` シグネチャへ現れない
/// ことも併せて検査する（薄いラッパー原則の機械的裏付け。#2017）。
/// `safetensors` サブモジュール（#2019）固有の検査は
/// `interop_safetensors_module_is_pure_reexport`／
/// `interop_safetensors_reexports_exactly_expected_surface` を参照。
#[test]
fn interop_module_exposes_only_approved_onnx_surface() {
    let interop_dir = facade_crate_root().join("src").join("interop");

    let mod_rs_content = read_to_string_or_panic(&interop_dir.join("mod.rs"));
    let mod_offending = scan_forbidden_pub_items(&mod_rs_content);
    // `mod.rs` は `pub mod onnx;`／`pub mod safetensors;`（#2019）の
    // 2 件のみを許容する（`scan_forbidden_pub_items` は `pub use` 以外の
    // `pub` アイテムを検出するため、`pub mod` 宣言も検出対象になる。
    // したがってここでは「丁度 2 件・いずれも `pub mod`」であることを
    // 検査する）。
    assert_eq!(
        mod_offending.len(),
        2,
        "src/interop/mod.rs の公開アイテムが想定外（`pub mod onnx;`／\
         `pub mod safetensors;` の丁度 2 件のはず）: {mod_offending:?}"
    );
    assert!(
        mod_offending.iter().all(|item| item.contains("mod")),
        "src/interop/mod.rs の公開アイテムに `pub mod` 以外が含まれる: {mod_offending:?}"
    );
    // `scan_forbidden_pub_items` は `pub use` を意図的に対象外とする
    // （他ガード箇所での「再エクスポートは許容する」前提のため）が、
    // `src/interop/mod.rs` は `pub mod onnx;` のみを承認しており
    // 再エクスポートは 1 件も承認していない。`pub use` が別途紛れ込んで
    // いないかをここで直接検査する（codex-review 指摘 P2・#2024
    // 「mod.rs 側も同様に pub use が検査対象外」を解消）。
    let mod_rs_pub_use_offending: Vec<&str> = mod_rs_content
        .lines()
        .map(str::trim_start)
        .filter(|line| line.starts_with("pub use"))
        .collect();
    assert!(
        mod_rs_pub_use_offending.is_empty(),
        "src/interop/mod.rs に承認範囲外の `pub use` 再エクスポートがある: \
         {mod_rs_pub_use_offending:?}"
    );

    let onnx_rs_path = interop_dir.join("onnx.rs");
    let onnx_rs_content = read_to_string_or_panic(&onnx_rs_path);

    // 内部クレートの型（`prost`・`ModelProto`・`NodeProto`・`Graph` 等）が
    // `pub` シグネチャへ露出していないことを、`pub` を含む行に限定して
    // 検査する（doc comment・private ヘルパ内の同名参照は対象外）。
    let mut leaked_internal_types = Vec::new();
    for line in onnx_rs_content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") || trimmed.starts_with("//!") || trimmed.starts_with("///") {
            continue;
        }
        if !trimmed.starts_with("pub ") && !trimmed.contains(" pub ") {
            continue;
        }
        for forbidden in [
            "ModelProto",
            "NodeProto",
            "prost::",
            "onnx::graph::Graph",
            "ExportError",
            "onnx::export::",
            "export_nn",
            "ExportNode",
            "NnExportParts",
            "nn::Module",
        ] {
            if line.contains(forbidden) {
                leaked_internal_types.push(format!("`{trimmed}` が {forbidden} を含む"));
            }
        }
    }
    assert!(
        leaked_internal_types.is_empty(),
        "src/interop/onnx.rs の pub シグネチャに内部クレート型が露出している: \
         {leaked_internal_types:?}"
    );

    // 公開型・メソッド名が期待どおり存在することの空虚 pass 防止。
    for expected in [
        "pub struct OnnxModel",
        "pub enum OnnxValue",
        "pub enum OnnxError",
        "pub struct OnnxExportOptions",
        "pub fn from_bytes",
        "pub fn from_path",
        "pub fn run",
        "pub fn to_bytes",
        "pub fn to_path",
        "pub fn from_sequential",
    ] {
        assert!(
            onnx_rs_content.contains(expected),
            "src/interop/onnx.rs に期待する公開シグネチャ `{expected}` が見つからない"
        );
    }

    // 承認範囲外の追加公開アイテム（例: `pub fn input_names` 等）が
    // 紛れ込んでいないことを網羅的に固定する（codex-review 指摘 P2・
    // #2024。上記の contains 検査は期待シグネチャの存在確認のみで、
    // 承認外の追加 pub アイテムを拒否できていなかった）。
    let unapproved_pub_items = scan_unapproved_onnx_pub_items(&onnx_rs_content);
    assert!(
        unapproved_pub_items.is_empty(),
        "src/interop/onnx.rs に承認範囲外の公開アイテムがある（薄いラッパー原則\
         違反の疑い。docs/facade-onnx-import-exposure-decision.md §12 参照）: \
         {unapproved_pub_items:?}"
    );
}

/// `fandhe_ai::interop::onnx::{OnnxModel, OnnxValue, OnnxError}` が
/// facade から到達可能であること・`OnnxError` が `#[non_exhaustive]` の
/// ためワイルドカード腕併用で `match` できることをコンパイル時に固定
/// する（`cast_types_are_reachable_via_facade` と同型。#2017）。
#[test]
fn onnx_import_types_are_reachable_via_facade() {
    use fandhe_ai::interop::onnx::{OnnxError, OnnxModel, OnnxValue};

    let err = OnnxModel::from_bytes(&[0x08u8, 0xffu8]).unwrap_err();
    let _label = match &err {
        OnnxError::Io(_) => "io",
        OnnxError::Decode { .. } => "decode",
        OnnxError::UnsupportedDataType { .. } => "unsupported_data_type",
        OnnxError::UnsupportedOp { .. } => "unsupported_op",
        OnnxError::MissingFeed { .. } => "missing_feed",
        OnnxError::UnknownFeed { .. } => "unknown_feed",
        OnnxError::UnsupportedLayer { .. } => "unsupported_layer",
        OnnxError::InvalidModel { .. } => "invalid_model",
        OnnxError::Execution { .. } => "execution",
        _ => "unknown",
    };

    let _value: Option<OnnxValue> = None;
}

/// `fandhe_ai::interop::onnx::{OnnxModel::to_bytes, OnnxModel::to_path,
/// OnnxExportOptions}`（イシュー #2018）が facade から到達可能であること
/// をコンパイル時に固定する（`onnx_import_types_are_reachable_via_facade`
/// と同型）。既定値ドリフトガード（`ir_version=8`・`opset_version=17`。
/// `docs/facade-onnx-export-exposure-decision.md` §4 承認事項 1）も併せて
/// 固定する。
#[test]
fn onnx_export_types_are_reachable_via_facade() {
    use fandhe_ai::interop::onnx::{OnnxError, OnnxExportOptions, OnnxModel};

    let options = OnnxExportOptions::default();
    assert_eq!(
        options.ir_version, 8,
        "test fixture: OnnxExportOptions の既定 ir_version は 8 のはず"
    );
    assert_eq!(
        options.opset_version, 17,
        "test fixture: OnnxExportOptions の既定 opset_version は 17 のはず"
    );

    // 未構築のモデルは扱えないため、`from_bytes` の失敗パスから
    // `to_bytes`／`to_path` の型シグネチャのみをコンパイル時に固定する
    // （`OnnxModel` インスタンスを要求しない静的な型検査）。
    fn _to_bytes_signature(m: &OnnxModel, o: &OnnxExportOptions) -> Result<Vec<u8>, OnnxError> {
        m.to_bytes(o)
    }
    fn _to_path_signature(
        m: &OnnxModel,
        path: &std::path::Path,
        o: &OnnxExportOptions,
    ) -> Result<(), OnnxError> {
        m.to_path(path, o)
    }
}

/// `fandhe_ai::interop::onnx::OnnxModel::from_sequential` が facade から
/// 到達可能であること（型シグネチャの静的固定）・空の `Sequential`
/// （層 0 個）が `OnnxError::InvalidModel` で拒否されることを固定する
/// （イシュー #2037。`onnx_export_types_are_reachable_via_facade` と
/// 同型）。
#[test]
fn onnx_export_from_sequential_is_reachable_via_facade() {
    use fandhe_ai::compat::Sequential;
    use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};

    fn _sig(m: &Sequential) -> Result<OnnxModel, OnnxError> {
        OnnxModel::from_sequential(m)
    }

    let empty = Sequential::new();
    let err = OnnxModel::from_sequential(&empty).unwrap_err();
    assert!(
        matches!(err, OnnxError::InvalidModel { .. }),
        "空の Sequential は OnnxError::InvalidModel で拒否されるはず: {err:?}"
    );
}

/// `src/interop/safetensors.rs`（イシュー #2019）専用の固定パス。
fn interop_safetensors_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/interop/safetensors.rs")
}

/// `src/interop/safetensors.rs` が facade 独自の型・関数を定義しない
/// 純再エクスポートモジュールであることを固定する（`optim_module_is_
/// pure_reexport` と同じ走査ロジック〈`scan_forbidden_pub_items`〉を
/// 再利用する。#2019）。
#[test]
fn interop_safetensors_module_is_pure_reexport() {
    let path = interop_safetensors_rs_path();
    let content = read_to_string_or_panic(&path);
    let offending = scan_forbidden_pub_items(&content);
    assert!(
        offending.is_empty(),
        "src/interop/safetensors.rs が facade 独自の型・関数・impl・pub type/const/static/\
         mod/union 等の公開宣言を定義している（純再エクスポートモジュールの契約違反）: \
         {offending:?}"
    );
}

/// `interop_safetensors_reexports_exactly_expected_surface` と合成入力
/// テスト `interop_safetensors_reexport_scanner_rejects_root_path_and_glob`
/// が共有する実際の走査・許可判定本体（#2019 の codex-review P2 指摘を
/// 受け #2025 で抽出）。`content` の各行を `allowed_prefixes` に対して
/// 判定し、承認接頭辞以外の行・解釈できない行を `offending_lines` へ、
/// 承認接頭辞行から抽出した識別子を `found` へ積む。`pub use <prefix>A;`
/// の単一識別子形と `pub use <prefix>{A, B};` の複数識別子形の両方を
/// 受理する一方、`*`（glob 再エクスポート）を含む識別子は明示的に
/// offending として拒否する（元実装は glob をブレースなし単一識別子と
/// 誤って受理し、後続の `assert_eq!` 頼みでしか検出できなかったため、
/// この関数自体を合成入力テストで直接検証できるよう是正した）。
fn scan_safetensors_reexport_lines(
    content: &str,
    allowed_prefixes: &[&str],
) -> (Vec<String>, std::collections::BTreeSet<String>) {
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
        let rest = &trimmed[prefix.len()..];
        if let (Some(open), Some(close)) = (rest.find('{'), rest.find('}')) {
            let mut braced_offending = false;
            for ident in rest[open + 1..close].split(',') {
                let ident = ident.trim();
                if ident.is_empty() {
                    continue;
                }
                if ident.contains('*') {
                    braced_offending = true;
                    continue;
                }
                found.insert(ident.to_string());
            }
            if braced_offending {
                offending_lines.push(trimmed.to_string());
            }
        } else {
            let ident = rest.trim_end_matches(';').trim();
            if ident.is_empty() || ident.contains(['{', '}', ':', '*']) {
                offending_lines.push(trimmed.to_string());
                continue;
            }
            found.insert(ident.to_string());
        }
    }

    (offending_lines, found)
}

/// `src/interop/safetensors.rs` の `pub use` 行から `{...}` 内の識別子を
/// 抽出し、昇格元公開面（`fandhe_ai_onnx_interop::st_load`／`st_save`）
/// と完全一致（過不足とも fail）することを固定する
/// （`optim_module_reexports_exactly_expected_surface` と同型）。各行の
/// path 接頭辞が `st_load::`／`st_save::` のいずれかであることも検査し、
/// クレートルート直下の別実装 `LoadError`／`require_keys`
/// （`onnx::interp` 用。本モジュールが再エクスポートしてはならない型）
/// からの混入・モジュール丸ごと／glob 再エクスポートを遮断する（#2019）。
#[test]
fn interop_safetensors_reexports_exactly_expected_surface() {
    let path = interop_safetensors_rs_path();
    let content = read_to_string_or_panic(&path);

    let allowed_prefixes = [
        "pub use fandhe_ai_onnx_interop::st_load::",
        "pub use fandhe_ai_onnx_interop::st_save::",
    ];

    let (offending_lines, found) = scan_safetensors_reexport_lines(&content, &allowed_prefixes);

    assert!(
        offending_lines.is_empty(),
        "src/interop/safetensors.rs の pub use が昇格元公開面\
         （fandhe_ai_onnx_interop::st_load / st_save）以外の接頭辞を持つか、\
         解釈できない形式の行を含む: {offending_lines:?}"
    );

    let expected: std::collections::BTreeSet<String> = [
        "LoadError",
        "SaveError",
        "load_safetensors_f32",
        "load_safetensors_f32_from_bytes",
        "require_keys",
        "save_safetensors_f32",
        "save_safetensors_f32_to_bytes",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    assert_eq!(
        found, expected,
        "src/interop/safetensors.rs が再エクスポートする識別子が期待集合と一致しない\
         （過不足いずれも不可。昇格元公開面〈st_load／st_save〉と 1 対 1 対応であることの\
         固定。クレートルート直下の別実装 LoadError／require_keys の誤混入を含む）"
    );
}

/// `interop_safetensors_reexports_exactly_expected_surface` が使う実際の
/// 走査・許可判定本体（`scan_safetensors_reexport_lines`）へ不正入力を
/// 直接渡し、両方とも offending として拒否されることを確認する
/// （codex-review 指摘: 旧実装は接頭辞判定のみを検証する合成テストで
/// glob 入力が `bad_line.contains('*')` により機械的に true 判定される
/// ため検出可否に関係なく assert が成立してしまっていた。#2025）。
#[test]
fn interop_safetensors_reexport_scanner_rejects_root_path_and_glob() {
    let allowed_prefixes = [
        "pub use fandhe_ai_onnx_interop::st_load::",
        "pub use fandhe_ai_onnx_interop::st_save::",
    ];

    // クレートルート直下の別実装（`LoadError`／`require_keys`）を
    // 承認接頭辞なしで再エクスポートしようとする行。
    let (offending, found) = scan_safetensors_reexport_lines(
        "pub use fandhe_ai_onnx_interop::{LoadError, require_keys};",
        &allowed_prefixes,
    );
    assert!(
        !offending.is_empty(),
        "承認接頭辞を持たないクレートルート直下パスの再エクスポートが offending として\
         検出されなかった（実処理の回帰）"
    );
    assert!(
        found.is_empty(),
        "offending として拒否されるべき行から識別子が found へ混入した: {found:?}"
    );

    // 承認接頭辞は持つが glob（`st_load::*`）で丸ごと再エクスポートしよ
    // うとする行。
    let (offending, found) = scan_safetensors_reexport_lines(
        "pub use fandhe_ai_onnx_interop::st_load::*;",
        &allowed_prefixes,
    );
    assert!(
        !offending.is_empty(),
        "承認接頭辞配下の glob 再エクスポートが offending として検出されなかった\
         （実処理の回帰。`*` を通常の単一識別子として誤って受理していないか確認）"
    );
    assert!(
        found.is_empty(),
        "offending として拒否されるべき glob 行から識別子が found へ混入した: {found:?}"
    );
}

/// `fandhe_ai::interop::safetensors::{LoadError, SaveError, ...}` が
/// facade から到達可能であること・両エラー型が `#[non_exhaustive]` の
/// ためワイルドカード腕併用で `match` できることをコンパイル時に固定
/// する（`onnx_import_types_are_reachable_via_facade` と同型。#2019）。
#[test]
fn interop_safetensors_types_are_reachable_via_facade() {
    use fandhe_ai::interop::safetensors::{LoadError, SaveError, load_safetensors_f32_from_bytes};

    let err = load_safetensors_f32_from_bytes(&[]).unwrap_err();
    let _label = match &err {
        LoadError::Io(_) => "io",
        LoadError::SafetensorsFormat(_) => "safetensors_format",
        LoadError::MissingKeys(_) => "missing_keys",
        LoadError::UnsupportedDtype { .. } => "unsupported_dtype",
        LoadError::DataLengthMismatch { .. } => "data_length_mismatch",
        LoadError::Shape(_) => "shape",
        _ => "unknown",
    };

    let _save_label = |e: &SaveError| -> &'static str {
        match e {
            SaveError::Io(_) => "io",
            SaveError::SafetensorsFormat(_) => "safetensors_format",
            SaveError::DataUnavailable { .. } => "data_unavailable",
            _ => "unknown",
        }
    };
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

/// `fandhe_ai::{Scalar, ScalarDType, TypedOps}`（イシュー #1939・
/// `docs/compat-api-scope.md` §5 経路 2 承認）が facade から到達可能で
/// あること・`Tape::typed_ops_f64`／`_f16`／`_bf16` が実際に CPU
/// バックエンドで `Some` を返し、返った `&dyn TypedOps<T>` を通じて
/// 演算が実行できることを固定する（コンパイル時裏付け＋実行時検証。
/// `cast_types_are_reachable_via_facade` と同型）。`half::f16`／
/// `half::bf16` は facade が再エクスポートしないため、利用者は
/// `fandhe_ai_tensor_core`（本テストクレートの通常依存）経由で直接
/// 名指しする。`ScalarDType` は `#[non_exhaustive]` のためワイルドカード
/// 腕で網羅する。
#[test]
fn typed_ops_types_are_reachable_via_facade() {
    // `Scalar` を型パラメータ境界として名指しできることの固定
    // （object safety は問わない・`TypedOps<T: Scalar>` の `T` 側）。
    fn _assert_scalar<T: fandhe_ai::Scalar>() {}
    let _ = _assert_scalar::<f32>;

    let tape = fandhe_ai::tape();

    // f64: TypedOps<f64> が Some を返し、add の結果がホスト f64 計算値と
    // bit 完全一致すること（tolerance を導入しない）。
    let ops_f64: Option<&dyn fandhe_ai::TypedOps<f64>> = tape.typed_ops_f64();
    let ops_f64 = ops_f64.expect("CPU backend は TypedOps<f64> に対応するはず（#1697）");
    let a64 = fandhe_ai::Tensor::<f64>::new(vec![1.0, 2.0, 3.0], &[3])
        .expect("test fixture: shape とデータ長は一致させている");
    let b64 = fandhe_ai::Tensor::<f64>::new(vec![10.0, 20.0, 30.0], &[3])
        .expect("test fixture: shape とデータ長は一致させている");
    let sum64 = ops_f64
        .add(&a64, &b64)
        .expect("CPU TypedOps<f64>::add は Ok のはず");
    assert_eq!(
        sum64.host_slice().into_owned(),
        vec![11.0, 22.0, 33.0],
        "f64 add はホスト f64 計算値と bit 完全一致するはず"
    );

    // f16／bf16: fandhe_ai_tensor_core 経由で直接型を名指しし、relu を
    // 1 演算実行して Ok と厳密に表現可能な値の一致を確認する。
    let ops_f16: Option<&dyn fandhe_ai::TypedOps<fandhe_ai_tensor_core::f16>> =
        tape.typed_ops_f16();
    let ops_f16 = ops_f16.expect("CPU backend は TypedOps<f16> に対応するはず（#1698）");
    let neg_and_pos = fandhe_ai::Tensor::<fandhe_ai_tensor_core::f16>::new(
        vec![
            fandhe_ai_tensor_core::f16::from_f32(-1.0),
            fandhe_ai_tensor_core::f16::from_f32(2.0),
        ],
        &[2],
    )
    .expect("test fixture: shape とデータ長は一致させている");
    let relu16 = ops_f16
        .relu(&neg_and_pos)
        .expect("CPU TypedOps<f16>::relu は Ok のはず");
    assert_eq!(
        relu16.host_slice().into_owned(),
        vec![
            fandhe_ai_tensor_core::f16::from_f32(0.0),
            fandhe_ai_tensor_core::f16::from_f32(2.0)
        ],
        "f16 relu はホスト参照実装と一致するはず"
    );

    let ops_bf16: Option<&dyn fandhe_ai::TypedOps<fandhe_ai_tensor_core::bf16>> =
        tape.typed_ops_bf16();
    let ops_bf16 = ops_bf16.expect("CPU backend は TypedOps<bf16> に対応するはず（#1699）");
    let neg_and_pos_bf16 = fandhe_ai::Tensor::<fandhe_ai_tensor_core::bf16>::new(
        vec![
            fandhe_ai_tensor_core::bf16::from_f32(-1.0),
            fandhe_ai_tensor_core::bf16::from_f32(2.0),
        ],
        &[2],
    )
    .expect("test fixture: shape とデータ長は一致させている");
    let relu_bf16 = ops_bf16
        .relu(&neg_and_pos_bf16)
        .expect("CPU TypedOps<bf16>::relu は Ok のはず");
    assert_eq!(
        relu_bf16.host_slice().into_owned(),
        vec![
            fandhe_ai_tensor_core::bf16::from_f32(0.0),
            fandhe_ai_tensor_core::bf16::from_f32(2.0)
        ],
        "bf16 relu はホスト参照実装と一致するはず"
    );

    // ScalarDType（#[non_exhaustive]）をワイルドカード腕付きで網羅
    // できることの固定（`cast_types_are_reachable_via_facade` の
    // `CastDType` 網羅と同型）。
    let dtype: fandhe_ai::ScalarDType = fandhe_ai::ScalarDType::F64;
    let _label = match dtype {
        fandhe_ai::ScalarDType::F32 => "f32",
        fandhe_ai::ScalarDType::F64 => "f64",
        fandhe_ai::ScalarDType::F16 => "f16",
        fandhe_ai::ScalarDType::Bf16 => "bf16",
        _ => "unknown",
    };
}

/// `fandhe_ai::Tape::typed_ops_f64`／`_f16`／`_bf16`（イシュー #1939）が
/// facade の公開面（`pub fn`）としてソース上に存在することを固定する
/// （`facade_exposes_available_devices_and_tape_transfer` と同型の
/// ソース走査。`typed_ops_types_are_reachable_via_facade` のコンパイル
/// 時裏付けを補完する）。
#[test]
fn facade_exposes_typed_ops_accessors_on_tape() {
    let lib_rs = facade_crate_root().join("src/lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    for needle in [
        "pub use fandhe_ai_tensor_core::{Scalar, ScalarDType, TypedOps};",
        "pub fn typed_ops_f64(&self) -> Option<&dyn TypedOps<f64>>",
        "pub fn typed_ops_f16(&self) -> Option<&dyn TypedOps<f16>>",
        "pub fn typed_ops_bf16(&self) -> Option<&dyn TypedOps<bf16>>",
    ] {
        assert!(
            content.contains(needle),
            "crates/facade/src/lib.rs に `{needle}` が見つからない（#1939）"
        );
    }
}

/// `crates/facade/src/` の `pub use` が `half`（`half::f16`／
/// `half::bf16` を含む）を再エクスポートしていないことを固定する
/// （承認事項「facade で `half` を再エクスポートしない」。イシュー
/// #1939・`facade_does_not_reexport_cast_ops` と同型の走査）。識別子
/// 境界で判定し、`ScalarDType::F16`／`Bf16` 等の大文字始まりの
/// variant 名は誤検知しない（`half::` プレフィックス・裸の `f16`／
/// `bf16` トークンのみを対象とする）。
#[test]
fn facade_does_not_reexport_half() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            if trimmed.contains("half::") {
                offending.push(format!("{}: `{trimmed}` が half:: を含む", path.display()));
                continue;
            }
            // 裸の `f16`／`bf16` トークン（識別子境界判定。`ScalarDType`
            // 等の大文字始まり variant とは別トークンのため誤検知しない）。
            for token in ["f16", "bf16"] {
                let mut search_from = 0usize;
                while let Some(pos) = trimmed[search_from..].find(token) {
                    let abs = search_from + pos;
                    let before_ok = trimmed[..abs]
                        .chars()
                        .next_back()
                        .is_none_or(|c| !c.is_alphanumeric() && c != '_');
                    let after_idx = abs + token.len();
                    let after_ok = trimmed[after_idx..]
                        .chars()
                        .next()
                        .is_none_or(|c| !c.is_alphanumeric() && c != '_');
                    if before_ok && after_ok {
                        offending.push(format!(
                            "{}: `{trimmed}` が裸の `{token}` トークンを含む",
                            path.display()
                        ));
                    }
                    search_from = abs + token.len();
                }
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が half（f16／bf16）を再エクスポートしている\
         （承認事項違反。イシュー #1939）: {offending:?}"
    );
}

/// `crates/facade/Cargo.toml` の依存セクションに `half` への直接依存が
/// 追加されていないことを固定する（承認事項「facade で `half` を
/// 再エクスポートしない」の裏付け。`half` は `fandhe-ai-tensor-core`
/// 経由の推移的依存のままで足りる。イシュー #1939）。
#[test]
fn facade_cargo_toml_does_not_depend_on_half() {
    let cargo_toml_path = facade_crate_root().join("Cargo.toml");
    let content = read_to_string_or_panic(&cargo_toml_path);
    let mut current_section = String::new();
    let mut offending = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        let code_part = trimmed
            .split_once('#')
            .map(|(a, _)| a)
            .unwrap_or(trimmed)
            .trim();
        if code_part.starts_with('[') && code_part.ends_with(']') {
            current_section = code_part.to_string();
            continue;
        }
        if !is_relevant_dependency_section(&current_section) || code_part.is_empty() {
            continue;
        }
        let Some((key, _value)) = code_part.split_once('=') else {
            continue;
        };
        let key = key.trim().trim_matches('"');
        if key == "half" {
            offending.push(format!(
                "`{current_section}` に依存 `half` を検出: `{trimmed}`"
            ));
        }
    }
    assert!(
        offending.is_empty(),
        "crates/facade/Cargo.toml が half へ直接依存している\
         （承認事項違反。イシュー #1939）: {offending:?}"
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

/// Keras 風 `compile()`／`fit()`／`evaluate()`（イシュー #1761）の
/// 新規公開型（`fandhe_ai::compat::{Loss, Optimizer, FitConfig,
/// History}`）が `fandhe_ai` のみの import で構築でき、`FitTarget`
/// が `f32`／`i32` の型境界として機能することを固定する。
///
/// callbacks（イシュー #1763）の新規公開型
/// （`fandhe_ai::compat::{Callback, EarlyStopping, ModelCheckpoint,
/// LrSchedule, Monitor, MonitorMode}`）が `fandhe_ai` のみの import で
/// 構築できること・`fandhe_ai::optim::Sgd::set_lr`（LR scheduler 結線
/// 用に追加した学習率更新 API）が facade 経由で到達可能であることも
/// 同テストで固定する。
#[test]
fn fit_types_are_reachable_via_facade_only() {
    use fandhe_ai::compat::{
        Callback, EarlyStopping, FitConfig, FitTarget, History, Loss, LrSchedule, ModelCheckpoint,
        Monitor, MonitorMode, Optimizer,
    };

    let _loss_mse = Loss::Mse;
    let _loss_ce = Loss::CrossEntropy;

    let _optimizer = Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.1));

    let config = FitConfig::new(3, 2).shuffle(true).drop_last(true);
    assert_eq!(config, FitConfig::new(3, 2).shuffle(true).drop_last(true));

    // `History` は `#[non_exhaustive]`（フィールド `loss` は `pub`）
    // のため struct literal では構築できない（クレート外からの想定
    // 構築経路は `Sequential::fit` の戻り値のみ）。型自体が facade
    // のみ import で参照可能であることだけを固定する。
    fn assert_is_history_type(_h: &History) {}
    let _ = assert_is_history_type;

    // `FitTarget` が facade のみ import で型境界として使えることの固定
    // （sealed trait のため呼び出し元は実装を追加できない）。
    fn assert_is_fit_target<T: FitTarget>() {}
    assert_is_fit_target::<f32>();
    assert_is_fit_target::<i32>();

    // callbacks（イシュー #1763）: 各型が facade のみ import で構築・
    // 到達可能であることを固定する。
    let _monitor = Monitor::ValLoss;
    let _mode = MonitorMode::Min;

    let early_stopping = EarlyStopping::new(3)
        .monitor(Monitor::Loss)
        .mode(MonitorMode::Min)
        .min_delta(0.0)
        .expect("test fixture: min_delta(0.0) は有効値のはず")
        .restore_best_weights(false);
    assert_eq!(early_stopping.stopped_epoch(), None);
    let _cb_early_stopping = Callback::EarlyStopping(early_stopping);

    let checkpoint = ModelCheckpoint::new()
        .monitor(Monitor::Loss)
        .mode(MonitorMode::Min)
        .save_best_only(true)
        // `to_file`（イシュー #2073）が facade のみ import で到達
        // 可能であることの固定点。ビルダーは FS に触れないため
        // 一時ディレクトリは不要（`callbacks.rs::ModelCheckpoint::
        // to_file` doc 参照）。
        .to_file("fandhe-ai-2073-unused.safetensors");
    assert_eq!(checkpoint.best_value(), None);
    let _cb_checkpoint = Callback::ModelCheckpoint(checkpoint);

    let lr_schedule = LrSchedule::per_epoch(
        fandhe_ai::optim::StepLr::new(0.1, 1, 0.5)
            .expect("test fixture: StepLr::new(0.1, 1, 0.5) は有効値のはず"),
    );
    assert_eq!(lr_schedule.epoch(), 0);
    let _cb_lr_schedule = Callback::LrSchedule(lr_schedule);

    // `Sgd::set_lr`（イシュー #1763。LR scheduler 結線用の学習率更新
    // API）が facade 経由でも到達可能であることを固定する。
    let mut sgd = fandhe_ai::optim::Sgd::new(fandhe_ai::optim::SgdConfig::new(0.1))
        .expect("test fixture: SgdConfig::new(0.1) は有効値のはず");
    sgd.set_lr(0.05)
        .expect("test fixture: set_lr(0.05) は有効値のはず");
}

/// metrics（accuracy・precision・recall・F1・confusion matrix。イシュー
/// #2072・親 #2059）: 公開面（`Metrics`・`MetricsResult`・
/// `Sequential::fit_with_metrics`・`Monitor::ValMetric`）が facade のみ
/// import で構築・到達可能であることを固定する（`fit_types_are_
/// reachable_via_facade_only` と同型）。
#[test]
fn metrics_types_are_reachable_via_facade_only() {
    use fandhe_ai::compat::{
        FitConfig, Loss, Metrics, MetricsResult, Monitor, Optimizer, Sequential,
    };
    use fandhe_ai::{Tensor, optim::SgdConfig};

    let _accuracy = Metrics::Accuracy;
    let _precision = Metrics::Precision;
    let _recall = Metrics::Recall;
    let _f1 = Metrics::F1;
    let _confusion = Metrics::ConfusionMatrix;

    let _monitor_val_metric = Monitor::ValMetric(Metrics::Accuracy);

    // `MetricsResult` は `#[non_exhaustive]` のため struct literal では
    // 構築できない（想定構築経路は `MetricsResult::compute` のみ）。
    // 型自体が facade のみ import で参照可能であることを固定する。
    fn assert_is_metrics_result_type(_m: &MetricsResult) {}
    let _ = assert_is_metrics_result_type;

    let logits = Tensor::<f32>::new(vec![0.1, 0.9, 0.8, 0.2], &[2, 2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let target = Tensor::<i32>::new(vec![1, 0], &[2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let result = MetricsResult::compute(&[Metrics::Accuracy], &logits, &target)
        .expect("test fixture: compute は成功するはず");
    assert_eq!(result.accuracy, Some(1.0));

    // `Sequential::fit_with_metrics` が facade のみ import で到達可能
    // であることを固定する（`metrics = &[]` は既存 `fit_with_callbacks`
    // と同一の演算列になる契約——`compat_sequential_metrics.rs::
    // fit_with_metrics_empty_matches_fit_with_callbacks_bit_exact` で
    // 検証済み）。
    let mut model = Sequential::new()
        .add_linear(2, 2, 0x1234)
        .expect("test fixture: add_linear は成功するはず");
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.1)), Loss::CrossEntropy)
        .expect("test fixture: compile は成功するはず");
    let x = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4], &[2, 2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let y = Tensor::<i32>::new(vec![0, 1], &[2])
        .expect("test fixture: shape とデータ長は事前に一致させている");
    let history = model
        .fit_with_metrics(
            &x,
            &y,
            FitConfig::new(1, 2),
            Some((&x, &y)),
            &mut [],
            &[Metrics::Accuracy],
        )
        .expect("test fixture: fit_with_metrics は成功するはず");
    assert_eq!(history.val_metrics.len(), 1);
}

/// `crates/facade/src/` の `pub use` が `CustomFunction`（ユーザー定義
/// forward／backward プラグイン機構。イシュー #1946・案 B）を
/// 再エクスポートしていないことを固定する（`docs/autodiff-custom-
/// function-decision.md` §12.5 (b)「facade 公開面」は未承認のまま対象外。
/// `facade_does_not_reexport_cast_ops` と同型の走査）。
/// **イシュー #2064（2026-09-22）時点でも §12.5 (b) の承認コメントが
/// 確認できなかったため未承認のまま維持する**（同 doc §15 保留記録）。
#[test]
fn facade_does_not_reexport_custom_function() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            if trimmed.contains("CustomFunction") {
                offending.push(format!(
                    "{}: `{trimmed}` が CustomFunction を含む",
                    path.display()
                ));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が CustomFunction を再エクスポートしている\
         （§12.5 (b) は未承認のまま対象外という設計判断に違反）: {offending:?}"
    );
}

/// `crates/facade/src/` の `pub use` が `nn::init`（PyTorch `torch.nn.
/// init.*` 相当の初期化関数群・イシュー #2140）を再エクスポートして
/// いないことを固定する（`docs/facade-nn-init-exposure-decision.md`
/// §0・§3「facade 公開（承認事項・経路 2）は未承認のまま保留」。
/// `facade_does_not_reexport_custom_function` と同型の走査）。
/// `nn_mod_declares_only_rnn_submodule` が `pub mod init;` 追加自体を
/// 別途固定する一方、本テストは `pub use fandhe_ai_autodiff::nn::
/// init::...` のような迂回経路（`nn/mod.rs` 以外のファイルからの
/// 再エクスポート）も走査対象に含める。
#[test]
fn facade_does_not_reexport_nn_init() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            if trimmed.contains("nn::init") {
                offending.push(format!(
                    "{}: `{trimmed}` が nn::init を含む",
                    path.display()
                ));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が nn::init を再エクスポートしている\
         （facade-nn-init-exposure-decision.md §3 は未承認のまま対象外\
         という設計判断に違反）: {offending:?}"
    );
}

/// facade 独自の `struct Tape`（`crates/facade/src/lib.rs`）が
/// `Tape::custom` への転送メソッドを持たないことを固定する（`Tape::
/// var_no_grad` の前例〈`docs/autodiff-custom-function-decision.md`
/// §12.4「入口」〉と同じ「転送メソッドを追加しない限り facade から
/// 到達不能」という設計を、転送メソッド自体が生えていないことで直接
/// 検査する）。承認 (b) を得て転送メソッドを追加する際は本テストを
/// 更新する。**イシュー #2064（2026-09-22）時点でも §12.5 (b) の承認
/// コメントが確認できなかったため未承認のまま維持する**（同 doc §15
/// 保留記録）。
#[test]
fn facade_tape_does_not_expose_custom_forwarding_method() {
    let lib_rs = facade_crate_root().join("src/lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    // コメント・文字列・char リテラルを除去してから `declares_pub_fn`
    // （トークン列の連続一致で `pub`・`fn`・関数名の間の空白量に影響
    // されず、`unsafe`／`const`／`async` 修飾子の挿入にも対応する）で
    // 判定する（codex-review 指摘・PR #2212: 旧実装
    // `contains_pub_fn_custom_declaration` は固定文字列 `"pub fn custom"`
    // の部分一致のため `pub async fn custom`／`pub unsafe fn custom` の
    // ような修飾子挿入で否定ガードを迂回できた。`compat_sequential_does_
    // not_expose_custom_add_method` と同じ前処理＋判定の組み合わせに
    // 揃える）。
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    assert!(
        !declares_pub_fn(&cleaned, "custom"),
        "facade 独自の Tape に `pub fn custom(...)` 宣言（ジェネリクス・\
         lifetime 付き `pub fn custom<'t>(`・`unsafe`／`const`／`async` \
         修飾子付きを含む）が見つかった\
         （§12.5 (b) 未承認のまま到達可能にしてしまっている）"
    );
}

/// `crates/facade/src/` に `CustomFunction` 識別子が一切現れないことを
/// 固定する（イシュー #2064 AC-4。`facade_does_not_reexport_custom_
/// function` は `pub use` 行のみを走査するため、CUDA Graph step との
/// 結線・gradcheck 相当の補助関数等、`pub use` を経由しない別の合成
/// 入口〈`pub fn foo(f: CustomFunction)` のような新規シグネチャ〉が
/// 生えても検出できない。本テストは `facade_does_not_expose_pool_
/// implementation_types` と同型のファイル全文走査で、承認 (b) 前に
/// そうした合成入口が facade へ混入することを構造的に遮断する）。
#[test]
fn facade_public_functions_do_not_take_custom_function() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        if content.contains("CustomFunction") {
            offending.push(path.display().to_string());
        }
    });
    assert!(
        offending.is_empty(),
        "facade の src/ に CustomFunction 識別子が現れている\
         （§12.5 (b) 未承認のまま合成入口を設けてしまっている）: {offending:?}"
    );
}

/// `crates/facade/src/compat/` に `Sequential::add_custom` 相当の合成
/// メソッド（`pub fn add_custom`）が生えていないことを固定する
/// （イシュー #2064 AC-4。`compat_sequential_does_not_expose_rnn_add_
/// methods` と同型。`Sequential` 平坦鎖への `CustomFunction` 合成入口を
/// 承認 (b) 前に設けないことの機械固定）。
#[test]
fn compat_sequential_does_not_expose_custom_add_method() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut offending = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        if declares_pub_fn(&cleaned, "add_custom") {
            offending.push(path.display().to_string());
        }
    });
    assert!(
        offending.is_empty(),
        "src/compat 配下に add_custom が見つかった\
         （§12.5 (b) 未承認のまま Sequential への合成入口を設けてしまっている）: {offending:?}"
    );
}

/// `content`（生の Rust ソース文字列。内部で `strip_comments_and_
/// literals` を適用する）に、可視性キーワード・修飾子・宣言文脈
/// （inherent impl／trait impl／trait 定義／自由関数のいずれか）を問わず
/// `fn <fn_name>(` または `fn <fn_name><`（ジェネリクス付き）の宣言が
/// 存在するかを判定する（[`facade_source_declares_no_custom_fn_in_any_
/// context`] 専用）。`declares_pub_fn` は `pub` トークンを起点に走査する
/// ため、`trait CustomExt { fn custom(&self); }` のような可視性キー
/// ワードを伴わない trait メソッド宣言（trait 自体が `pub` であれば
/// 実装型を通じて外部から呼び出せる）を見逃す。本関数は `fn` トークン
/// そのものを起点にするためこの迂回を検出できる（codex-review 指摘・
/// PR #2212: `crates/facade/src/compat/` に `extensions` のような新設
/// モジュールを追加し、そこに `CustomExt` trait と `impl CustomExt for
/// Var { fn custom(&self) {} }` を生やす迂回に対する多層防御）。
fn declares_fn_named(content: &str, fn_name: &str) -> bool {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (i, token) in tokens.iter().enumerate() {
        if token == "fn"
            && tokens.get(i + 1).map(String::as_str) == Some(fn_name)
            && matches!(tokens.get(i + 2).map(String::as_str), Some("(") | Some("<"))
        {
            return true;
        }
    }
    false
}

#[test]
fn declares_fn_named_detects_trait_method_and_ignores_comments_and_strings() {
    assert!(declares_fn_named(
        "pub trait CustomExt { fn custom(&self); }",
        "custom"
    ));
    assert!(declares_fn_named(
        "impl CustomExt for Var { fn custom(&self) {} }",
        "custom"
    ));
    assert!(declares_fn_named(
        "trait AddCustomExt { fn add_custom<T>(&self, v: T); }",
        "add_custom"
    ));
    assert!(!declares_fn_named("// fn custom(&self) {}", "custom"));
    assert!(!declares_fn_named("let s = \"fn custom(\";", "custom"));
    assert!(!declares_fn_named("fn custom_extra(&self) {}", "custom"));
}

/// facade 全ソース（`crates/facade/src/` 配下の全 `.rs`）に `fn custom`／
/// `fn add_custom` の宣言が可視性・宣言文脈（inherent impl・trait impl・
/// trait 定義・自由関数のいずれか）を問わず一切現れないことを固定する
/// （codex-review 指摘・PR #2212）。`facade_tape_does_not_expose_custom_
/// forwarding_method`（`lib.rs` の `pub fn custom` のみ）・
/// `compat_sequential_does_not_expose_custom_add_method`（`src/compat/`
/// 配下の `pub fn add_custom` のみ）はいずれも `pub fn` 形の宣言しか
/// 検査しないため、可視性キーワードを伴わない trait メソッド宣言経由の
/// 合成入口（例: `compat::extensions::CustomExt` のような新設モジュール
/// に生える `.custom()`）を見逃す。本テストは `declares_fn_named` により
/// `fn` トークンそのものを起点に facade 全体を走査し、上記 2 テストを
/// 補完する多層防御の 1 層とする。
#[test]
fn facade_source_declares_no_custom_fn_in_any_context() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for fn_name in ["custom", "add_custom"] {
            if declares_fn_named(content, fn_name) {
                offending.push(format!("{}: `fn {fn_name}` 宣言", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の src/ に `fn custom`／`fn add_custom` 宣言が可視性・文脈を問わず\
         見つかった（§12.5 (b) 未承認のまま合成入口を設けてしまっている）: {offending:?}"
    );
}

/// `text` を「識別子トークン」と「区切り文字（1 文字）」に分解した
/// トークン列へ変換する（空白は読み飛ばす）。`declares_pub_fn` 専用の
/// ユーティリティ。`(`／`<` などの区切り文字もトークンとして残すことで、
/// `pub`・`fn`・関数名の間に改行を挟んだ有効な Rust 記法を、固定文字列
/// 一致ではなくトークン列の連続一致で検出できるようにする（codex-review
/// 指摘・PR #2212 その 5: `cleaned.contains("pub fn add_custom")` の
/// 固定文字列一致は `pub\nfn add_custom(` のような改行を挟んだ宣言を
/// 見逃す）。`crates/autodiff/tests/architecture_boundaries.rs` の同名
/// ユーティリティと同型（クレートをまたぐ integration test 間でヘルパー
/// を共有できないため個別実装）。
fn tokenize_including_punctuation(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '\'' || c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            if c == '\'' {
                i += 1;
            }
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        } else {
            tokens.push(c.to_string());
            i += 1;
        }
    }
    tokens
}

/// `tokens[j..]` の先頭から連続する `fn` 宣言の修飾子トークン
/// （`unsafe`／`const`／`async`／`extern` およびその ABI 文字列リテラル。
/// 任意順・0 個以上の繰り返し）を読み飛ばし、修飾子列の直後の index を
/// 返す（codex-review 指摘・PR #2212 その 11）: 旧実装は `unsafe`／
/// `const`／`async` の 3 種のみを読み飛ばしており `pub extern "C" fn
/// custom` のような ABI 指定付き宣言を見逃していた（`extern` トークンが
/// `fn` 直前の位置に来るため `tokens.get(j) == Some("fn")` の一致に
/// 失敗し `declares_pub_fn` が false を返す）。`extern` は ABI 文字列
/// リテラル（`"C"` 等）を伴う場合と伴わない場合の両方が有効な Rust
/// 記法であるため、`extern` の直後にトークン化された文字列リテラル
/// （`"` 開始トークンから対応する `"` 終了トークンまで）が続けばそれも
/// まとめて読み飛ばす。`crates/autodiff/tests/architecture_boundaries.
/// rs` の同名ユーティリティと同型（クレートをまたぐ integration test
/// 間でヘルパーを共有できないため個別実装）。
fn skip_fn_declaration_qualifiers(tokens: &[String], mut j: usize) -> usize {
    loop {
        match tokens.get(j).map(String::as_str) {
            Some("unsafe" | "const" | "async") => {
                j += 1;
            }
            Some("extern") => {
                j += 1;
                if tokens.get(j).map(String::as_str) == Some("\"") {
                    j += 1;
                    while let Some(tok) = tokens.get(j) {
                        j += 1;
                        if tok == "\"" {
                            break;
                        }
                    }
                }
            }
            _ => break,
        }
    }
    j
}

/// `content`（コメント除去済み想定）に `pub fn <fn_name>(` または
/// `pub fn <fn_name><`（ジェネリクス付き）宣言が存在するかをトークン列
/// の連続一致で判定する。`pub`・`fn`・`fn_name` の間の空白量（改行を
/// 含む）に影響されない。`pub`・`fn` の間に `unsafe`／`const`／`async`／
/// `extern`（ABI 文字列リテラル付き含む）修飾子（0 個以上・任意順の
/// 繰り返し）が挟まる宣言も検出する（`skip_fn_declaration_qualifiers`。
/// `unapproved_onnx_pub_fn_with_qualifiers_is_flagged` が既に固定
/// している `pub async fn`／`pub unsafe fn` の扱いに合わせる）。
/// `pub(crate) fn ...` のようなスコープ付き可視性は「独立した `pub`
/// トークンの直後に修飾子または `fn` トークンが続かない」ため一致しない。
/// `crates/autodiff/tests/architecture_boundaries.rs` の同名ユーティリ
/// ティと同型（クレートをまたぐ integration test 間でヘルパーを共有
/// できないため個別実装）。
fn declares_pub_fn(content: &str, fn_name: &str) -> bool {
    let tokens = tokenize_including_punctuation(content);
    for (i, token) in tokens.iter().enumerate() {
        if token != "pub" {
            continue;
        }
        let j = skip_fn_declaration_qualifiers(&tokens, i + 1);
        if tokens.get(j).map(String::as_str) == Some("fn")
            && tokens.get(j + 1).map(String::as_str) == Some(fn_name)
            && matches!(tokens.get(j + 2).map(String::as_str), Some("(") | Some("<"))
        {
            return true;
        }
    }
    false
}

#[test]
fn declares_pub_fn_detects_newline_separated_declaration() {
    assert!(declares_pub_fn(
        "pub\nfn add_custom(&self) {}",
        "add_custom"
    ));
    assert!(declares_pub_fn(
        "pub\n    fn\nadd_custom<T>(&self) {}",
        "add_custom"
    ));
    assert!(!declares_pub_fn(
        "pub fn add_custom_foo(&self) {}",
        "add_custom"
    ));
    assert!(!declares_pub_fn(
        "pub(crate) fn add_custom(&self) {}",
        "add_custom"
    ));
    assert!(!declares_pub_fn("let add_custom = 1;", "add_custom"));
    assert!(declares_pub_fn(
        "pub unsafe fn add_custom(&self) {}",
        "add_custom"
    ));
    assert!(declares_pub_fn(
        "pub const fn add_custom() {}",
        "add_custom"
    ));
}

/// [`declares_pub_fn`] が `extern`（ABI 文字列リテラル付き・なし双方）を
/// 挟んだ宣言も検出することを固定する回帰テスト（codex-review 指摘・
/// PR #2212 その 11）。
#[test]
fn declares_pub_fn_detects_extern_qualified_declaration() {
    assert!(declares_pub_fn(
        "pub extern \"C\" fn add_custom() {}",
        "add_custom"
    ));
    assert!(declares_pub_fn(
        "pub unsafe extern \"C\" fn add_custom() {}",
        "add_custom"
    ));
    assert!(declares_pub_fn(
        "pub extern fn add_custom() {}",
        "add_custom"
    ));
}

/// AMP 統合（イシュー #1961。`compat::Sequential::compile_with_amp`）の
/// 新規公開型（`fandhe_ai::compat::{AmpConfig, AmpDType}`）が `fandhe_ai`
/// のみの import で構築でき、`Sequential::compile_with_amp`／
/// `amp_loss_scale` が到達可能であることを固定する。
#[test]
fn amp_fit_types_are_reachable_via_facade() {
    use fandhe_ai::compat::{AmpConfig, AmpDType, Loss, Optimizer, Sequential};

    let config_f16 = AmpConfig::new(AmpDType::F16);
    let config_bf16 =
        AmpConfig::new(AmpDType::Bf16).grad_scaler(fandhe_ai::optim::GradScalerConfig {
            init_scale: 128.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 100,
        });
    assert_eq!(config_f16, AmpConfig::new(AmpDType::F16));
    assert_ne!(config_f16, config_bf16);

    let mut model = Sequential::new()
        .add_linear(2, 2, 1)
        .expect("test fixture: add_linear は有効値のはず");
    assert_eq!(model.amp_loss_scale(), None);
    model
        .compile_with_amp(
            Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.1)),
            Loss::Mse,
            config_f16,
        )
        .expect("test fixture: compile_with_amp は有効値のはず");
    assert!(model.is_compiled());
    // `AmpConfig::new` は `GradScalerConfig::default()`（`init_scale =
    // 65536.0`）を使う（`training.rs::AmpConfig::new` doc 参照）。
    assert_eq!(model.amp_loss_scale(), Some(65536.0));
}

// =====================================================================
// nn::rnn 公開面（イシュー #1955）のガード。
// `data_*` 3 点セット（`data_module_reexports_exactly_expected_
// surface`／`data_module_is_pure_reexport`／`data_types_are_reachable_
// via_facade_only`）を鏡写しにする。
// =====================================================================

fn nn_rnn_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/nn/rnn.rs")
}

fn nn_mod_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/nn/mod.rs")
}

/// `src/nn/rnn.rs` の `pub use` 行から `{...}` 内の識別子を抽出し、
/// 昇格元公開面（`fandhe_ai_autodiff::nn`）と完全一致（過不足とも
/// fail）することを固定する（`data_module_reexports_exactly_expected_
/// surface` と同型の検査）。**#2133 の保留（`docs/facade-nn-module-
/// exposure-decision.md` §12）も本テストが担う**: 期待集合に `Module`
/// を含めていないため、`src/nn/rnn.rs` の `pub use` 行へ `Module` を
/// 追加すると本テストが fail する。承認後に案 B（facade 独自 trait）を
/// 実装する際は期待集合を正ガードへ更新する。
#[test]
fn nn_rnn_module_reexports_exactly_expected_surface() {
    let path = nn_rnn_rs_path();
    let content = read_to_string_or_panic(&path);

    let allowed_prefix = "pub use fandhe_ai_autodiff::nn::";

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
        "src/nn/rnn.rs の pub use が昇格元公開面（fandhe_ai_autodiff::nn）以外の\
         接頭辞を持つか、`{{...}}` 形式でない行を含む: {offending_lines:?}"
    );

    let expected: std::collections::BTreeSet<String> = [
        "Gru",
        "GruCellVars",
        "Lstm",
        "LstmCellVars",
        "LstmSeqOutput",
        "Rnn",
        "RnnCellVars",
        "RnnSeqOutput",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    assert_eq!(
        found, expected,
        "src/nn/rnn.rs が再エクスポートする識別子が期待集合と一致しない\
         （過不足いずれも不可。`forward_seq` の入出力に必要な型に限定する\
         承認スコープの固定。`RnnCell`／`LstmCell`／`GruCell`・`Module` は\
         意図して対象外）"
    );
}

/// `src/nn/rnn.rs` が facade 独自の型・関数を定義しない純再エクスポート
/// モジュールであることを固定する（`data_module_is_pure_reexport` と
/// 同じ走査ロジック〈`scan_forbidden_pub_items`〉を再利用する）。
/// `src/nn/mod.rs`（`pub mod rnn;` を含む）は対象外
/// （`nn_mod_declares_only_rnn_submodule` が別途固定する）。
#[test]
fn nn_rnn_module_is_pure_reexport() {
    let path = nn_rnn_rs_path();
    let content = read_to_string_or_panic(&path);
    let offending = scan_forbidden_pub_items(&content);
    assert!(
        offending.is_empty(),
        "src/nn/rnn.rs が facade 独自の型・関数・impl・pub type/const/static/mod/union 等の\
         公開宣言を定義している（純再エクスポートモジュールの契約違反）: {offending:?}"
    );
}

/// `src/nn/mod.rs` の公開宣言が `pub mod rnn;` の 1 件のみであること
/// を固定する（将来の無断拡大を fail-closed に検出する）。**#2133 の
/// 保留（`docs/facade-nn-module-exposure-decision.md` §12）も本テストが
/// 担う**: 案 B 採用時に想定する `nn::module`／`nn::container` 新設
/// （`pub mod module;`／`pub mod container;`）はこの完全一致検査に
/// より現時点では fail する。承認後に案 B を実装する際は期待集合
/// （`["rnn", "module", "container"]` 等）へ更新する。**#2140 の保留
/// （`docs/facade-nn-init-exposure-decision.md` §3・§4）も本テストが
/// 担う**: `nn::init` の facade 公開（条件付き手順 §4）で想定する
/// `pub mod init;` 追加はこの完全一致検査により現時点では fail する。
/// 承認後に実装する際は期待集合（`["init", "rnn"]` 等）へ更新する。
#[test]
fn nn_mod_declares_only_rnn_submodule() {
    let path = nn_mod_rs_path();
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();

    let declared: Vec<&str> = cleaned
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("pub mod"))
        .collect();

    assert_eq!(
        declared,
        vec!["pub mod rnn;"],
        "src/nn/mod.rs が宣言する pub mod が `pub mod rnn;` の 1 件と一致しない\
         （nn 公開面の無断拡大を検知）: {declared:?}"
    );
}

/// `fandhe_ai::nn::rnn` の全再エクスポート型が facade のみを通じて
/// 到達可能であることのコンパイル時＋実行時固定（`data_types_are_
/// reachable_via_facade_only` と同型。`fandhe_ai_autodiff` は
/// import しない）。`Tape::rnn_forward_seq`／`lstm_forward_seq`／
/// `gru_forward_seq`（本テストが呼ぶ橋渡し入口）も同時に固定する。
#[test]
fn nn_rnn_types_are_reachable_via_facade_only() {
    use fandhe_ai::nn::rnn::{
        Gru, GruCellVars, Lstm, LstmSeqOutput, Rnn, RnnCellVars, RnnSeqOutput,
    };

    let tape = fandhe_ai::tape();

    // Rnn: forward_seq → backward → params 経由の勾配取得。
    let rnn = Rnn::new(2, 3, true, 7).expect("test fixture: Rnn::new は有効値のはず");
    // T=3, B=1, D=2（input_size=2 と一致させる）。
    let x = fandhe_ai::Tensor::<f32>::new(vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[3usize, 1, 2])
        .expect("test fixture: x tensor の構築に失敗");
    let out: RnnSeqOutput<'_, RnnCellVars<'_>> = tape
        .rnn_forward_seq(&rnn, &x, None)
        .expect("test fixture: rnn_forward_seq は有効値のはず");
    let loss = out
        .outputs
        .last()
        .expect("test fixture: outputs は空でないはず")
        .sum(None)
        .expect("test fixture: sum は有効値のはず");
    let grads = tape
        .backward(&loss)
        .expect("test fixture: backward は有効値のはず");
    assert!(
        grads
            .get(&out.params.weight_ih)
            .expect("test fixture: get は有効値のはず")
            .is_some()
    );

    // Lstm: h0/c0 を明示して forward_seq → backward。
    let lstm = Lstm::new(2, 3, true, 11).expect("test fixture: Lstm::new は有効値のはず");
    let h0 = tape.var(
        &fandhe_ai::Tensor::<f32>::zeros(&[1, 3]).expect("test fixture: zeros は有効値のはず"),
    );
    let c0 = tape.var(
        &fandhe_ai::Tensor::<f32>::zeros(&[1, 3]).expect("test fixture: zeros は有効値のはず"),
    );
    let lstm_out: LstmSeqOutput<'_> = tape
        .lstm_forward_seq(&lstm, &x, Some(&h0), Some(&c0))
        .expect("test fixture: lstm_forward_seq は有効値のはず");
    let lstm_loss = lstm_out
        .c_n
        .sum(None)
        .expect("test fixture: sum は有効値のはず");
    let lstm_grads = tape
        .backward(&lstm_loss)
        .expect("test fixture: backward は有効値のはず");
    assert!(
        lstm_grads
            .get(&lstm_out.params.weight_hh)
            .expect("test fixture: get は有効値のはず")
            .is_some()
    );
    let _: &Option<fandhe_ai::Var<'_>> = &lstm_out.params.bias_ih;

    // Gru: forward_seq → backward。
    let gru = Gru::new(2, 3, true, 13).expect("test fixture: Gru::new は有効値のはず");
    let gru_out: RnnSeqOutput<'_, GruCellVars<'_>> = tape
        .gru_forward_seq(&gru, &x, None)
        .expect("test fixture: gru_forward_seq は有効値のはず");
    let gru_loss = gru_out
        .h_n
        .sum(None)
        .expect("test fixture: sum は有効値のはず");
    let gru_grads = tape
        .backward(&gru_loss)
        .expect("test fixture: backward は有効値のはず");
    assert!(
        gru_grads
            .get(&gru_out.params.weight_ih)
            .expect("test fixture: get は有効値のはず")
            .is_some()
    );
    let _: &Option<fandhe_ai::Var<'_>> = &out.params.bias_hh;
}

/// `compat::Sequential` に `add_rnn`／`add_lstm`／`add_gru` が存在
/// しないことを固定する（承認スコープ「`Sequential::add_*` は追加
/// しない」の機械固定。`nn_rnn_module_reexports_exactly_expected_
/// surface` が Cell 型・`Module` の非再エクスポートを別途固定する）。
#[test]
fn compat_sequential_does_not_expose_rnn_add_methods() {
    let compat_dir = facade_crate_root().join("src/compat");
    let forbidden = ["pub fn add_rnn", "pub fn add_lstm", "pub fn add_gru"];
    let mut offenses = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        for needle in forbidden {
            if cleaned.contains(needle) {
                offenses.push(format!("{}: {needle}", path.display()));
            }
        }
    });
    assert!(
        offenses.is_empty(),
        "src/compat 配下に add_rnn／add_lstm／add_gru が見つかった\
         （承認スコープ〈#1955〉は Sequential への追加を認めていない）: {offenses:?}"
    );
}

/// `crates/facade/src/` の `pub use` が `CreateGraphResult`（子テープ
/// 方式の高階微分結果型。イシュー #1942／#1943 で内部クレート
/// `fandhe_ai_autodiff` に実装済み）を再エクスポートしていないことを
/// 固定する（`docs/autodiff-higher-order-grad-decision.md` §10 承認
/// 事項 5「facade 公開面への高階 API 追加」はイシュー #2063 時点で
/// リポジトリ所有者の明示的な承認コメントが確認できず未承認のまま
/// 対象外。`facade_does_not_reexport_custom_function` と同型の走査）。
#[test]
fn facade_does_not_reexport_create_graph_result() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            if trimmed.contains("CreateGraphResult") {
                offending.push(format!(
                    "{}: `{trimmed}` が CreateGraphResult を含む",
                    path.display()
                ));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が CreateGraphResult を再エクスポートしている\
         （承認事項 5 は未承認のまま対象外という設計判断に違反）: {offending:?}"
    );
}

/// facade 独自の `struct Tape`（`crates/facade/src/lib.rs`）が
/// `Tape::backward_create_graph` への委譲メソッドを持たないことを
/// 固定する（`Tape::var_no_grad` の前例と同じ「委譲メソッドを追加
/// しない限り facade から到達不能」という設計を、委譲メソッド自体が
/// 生えていないことで直接検査する）。承認事項 5 の承認を得て委譲
/// メソッドを追加する際は本テストを正ガードへ更新する。
#[test]
fn facade_tape_does_not_expose_backward_create_graph_method() {
    let lib_rs = facade_crate_root().join("src/lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    assert!(
        !contains_pub_fn_declaration(&content, "backward_create_graph"),
        "facade 独自の Tape に `pub fn backward_create_graph(...)` 宣言\
         （ジェネリクス・lifetime 付き `pub fn backward_create_graph<'c>(`\
         を含む）が見つかった（承認事項 5 未承認のまま到達可能に\
         してしまっている）"
    );
}

/// `pub fn <name>` 宣言（`pub fn <name>(` に加え、ジェネリクス・
/// lifetime 付き `pub fn <name><'a>(` のような宣言も含む）の検出。
/// `<name>` の直後に任意個の空白、続けて任意で `<...>`（ジェネリクス・
/// lifetime パラメータ節。ネストする `<>` を素朴にカウントして対応
/// する）、さらに任意個の空白を挟んで `(` が現れる形を宣言とみなす
/// （`pub fn <name>_foo(` のような無関係な識別子への誤検出は、
/// `<name>` 直後が英数字／`_` の場合を除外することで避ける）。
fn contains_pub_fn_declaration(content: &str, name: &str) -> bool {
    let needle = format!("pub fn {name}");
    let bytes = content.as_bytes();
    let mut search_start = 0usize;
    while let Some(rel_idx) = content[search_start..].find(needle.as_str()) {
        let idx = search_start + rel_idx;
        let after = idx + needle.len();
        search_start = after;
        // `<name>` の直後が識別子構成文字（英数字／`_`）なら
        // `<name>_foo` 等の無関係な関数名なので除外する。
        if bytes
            .get(after)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
        {
            continue;
        }
        let mut pos = after;
        // 任意個の空白（改行含む）をスキップする。
        while bytes.get(pos).is_some_and(|b| b.is_ascii_whitespace()) {
            pos += 1;
        }
        // 任意で `<...>`（ジェネリクス／lifetime 節）をスキップする。
        // ネストする `<>`（例: `<T: Foo<Bar>>`）にも対応するため
        // 深さカウンタで対応する `>` まで読み飛ばす。
        if bytes.get(pos) == Some(&b'<') {
            let mut depth = 0i32;
            while let Some(b) = bytes.get(pos) {
                match b {
                    b'<' => depth += 1,
                    b'>' => {
                        depth -= 1;
                        if depth == 0 {
                            pos += 1;
                            break;
                        }
                    }
                    _ => {}
                }
                pos += 1;
            }
            if depth != 0 {
                // 対応する `>` が見つからないまま終端した場合は
                // 宣言として確定できないので次の occurrence を探す。
                continue;
            }
        }
        // 任意個の空白をスキップし、`(` が続けば宣言とみなす。
        while bytes.get(pos).is_some_and(|b| b.is_ascii_whitespace()) {
            pos += 1;
        }
        if bytes.get(pos) == Some(&b'(') {
            return true;
        }
    }
    false
}

#[test]
fn contains_pub_fn_declaration_detects_variants() {
    assert!(contains_pub_fn_declaration("pub fn foo(", "foo"));
    assert!(contains_pub_fn_declaration("pub fn foo<'t>(", "foo"));
    assert!(contains_pub_fn_declaration(
        "pub fn foo<'t, T: Bar<Baz>>(",
        "foo"
    ));
    assert!(contains_pub_fn_declaration("pub fn foo  (\n", "foo"));
    assert!(!contains_pub_fn_declaration("pub fn foo_bar(", "foo"));
    assert!(!contains_pub_fn_declaration(
        "// pub fn foo_baz(\nfn other() {}",
        "foo"
    ));
    assert!(!contains_pub_fn_declaration("let foo = 1;", "foo"));
}

// ============================================================================
// model 公開面の機械検査（イシュー #2087・親 #2082）
// ============================================================================

/// `src/model.rs` に承認範囲外の公開アイテムが存在しないかを走査する。
/// 承認範囲は `model::{ModelRegistry, ModelError}` と
/// `ModelRegistry::{new, with_cache_dir, cache_dir, load,
/// available_models}` の 7 件のみ（PR 本文「承認事項」節。
/// `scan_unapproved_onnx_pub_items` と同型の独自許可リストを持つ
/// 兄弟関数として実装する。既存 onnx 用関数は並列 PR の競合面拡大を
/// 避けるため byte 単位で不変のまま触らない）。
fn scan_unapproved_model_pub_items(original: &str) -> Vec<String> {
    const ALLOWED_PUB_ITEMS: [(&str, &str); 7] = [
        ("struct", "ModelRegistry"),
        ("enum", "ModelError"),
        ("fn", "new"),
        ("fn", "with_cache_dir"),
        ("fn", "cache_dir"),
        ("fn", "load"),
        ("fn", "available_models"),
    ];
    const SCANNED_KINDS: [&str; 9] = [
        "struct", "enum", "fn", "trait", "type", "const", "static", "mod", "use",
    ];
    const QUALIFIER_KEYWORDS: [&str; 3] = ["async", "unsafe", "extern"];
    const FORBIDDEN_SIGNATURE_SUBSTRINGS: [&str; 3] = ["BackendOps", "Tape", "onnx_interop"];

    let cleaned = strip_comments_and_literals(original);
    let len = cleaned.len();
    let mut offenses = Vec::new();
    let mut i = 0usize;
    while i < len {
        if !is_ident_start(cleaned[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut j = i + 1;
        while j < len && is_ident_char(cleaned[j]) {
            j += 1;
        }
        let word: String = cleaned[start..j].iter().collect();
        if word != "pub" {
            i = j;
            continue;
        }
        let mut k = j;
        while k < len && cleaned[k].is_whitespace() {
            k += 1;
        }
        if k < len && cleaned[k] == '(' {
            // `pub(crate)`／`pub(super)` 等。crate 外非公開のためスキップ。
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
            i = k;
            continue;
        }
        loop {
            if !(k < len && is_ident_start(cleaned[k])) {
                break;
            }
            let qs = k;
            let mut qe = k + 1;
            while qe < len && is_ident_char(cleaned[qe]) {
                qe += 1;
            }
            let candidate: String = cleaned[qs..qe].iter().collect();
            if !QUALIFIER_KEYWORDS.contains(&candidate.as_str()) {
                break;
            }
            k = qe;
            while k < len && cleaned[k].is_whitespace() {
                k += 1;
            }
        }
        if !(k < len && is_ident_start(cleaned[k])) {
            i = j;
            continue;
        }
        let ks = k;
        let mut ke = k + 1;
        while ke < len && is_ident_char(cleaned[ke]) {
            ke += 1;
        }
        let kind: String = cleaned[ks..ke].iter().collect();
        if !SCANNED_KINDS.contains(&kind.as_str()) {
            i = j;
            continue;
        }
        if kind == "use" {
            offenses.push(format!(
                "line {}: `pub use` は model.rs で承認されていない再エクスポート",
                line_at(&cleaned, start)
            ));
            i = j;
            continue;
        }
        let mut m = ke;
        while m < len && cleaned[m].is_whitespace() {
            m += 1;
        }
        if m < len && is_ident_start(cleaned[m]) {
            let ns = m;
            let mut ne = m + 1;
            while ne < len && is_ident_char(cleaned[ne]) {
                ne += 1;
            }
            let name: String = cleaned[ns..ne].iter().collect();
            let approved = ALLOWED_PUB_ITEMS
                .iter()
                .any(|(k2, n2)| *k2 == kind.as_str() && *n2 == name.as_str());
            if !approved {
                offenses.push(format!(
                    "line {}: `pub {kind} {name}` は model.rs で承認範囲外の公開アイテム",
                    line_at(&cleaned, start)
                ));
            }
            if kind == "fn" {
                let mut p = ne;
                while p < len && cleaned[p] != '(' && cleaned[p] != '{' && cleaned[p] != ';' {
                    p += 1;
                }
                if p < len && cleaned[p] == '(' {
                    let mut depth = 1i32;
                    p += 1;
                    while p < len && depth > 0 {
                        match cleaned[p] {
                            '(' => depth += 1,
                            ')' => depth -= 1,
                            _ => {}
                        }
                        p += 1;
                    }
                }
                while p < len && cleaned[p] != '{' && cleaned[p] != ';' {
                    p += 1;
                }
                let signature: String = cleaned[start..p.min(len)].iter().collect();
                for forbidden in FORBIDDEN_SIGNATURE_SUBSTRINGS {
                    if signature.contains(forbidden) {
                        offenses.push(format!(
                            "line {}: `pub fn {name}` のシグネチャが禁止文字列 `{forbidden}` を含む",
                            line_at(&cleaned, start)
                        ));
                    }
                }
            }
        }
        i = j;
    }
    offenses
}

fn model_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/model.rs")
}

#[test]
fn model_module_exposes_only_approved_surface() {
    let content = read_to_string_or_panic(&model_rs_path());
    let offenses = scan_unapproved_model_pub_items(&content);
    assert!(
        offenses.is_empty(),
        "src/model.rs に承認範囲外の公開アイテムが見つかった: {offenses:?}"
    );
}

/// `scan_unapproved_model_pub_items` が合成入力で承認範囲外の
/// `pub fn` を検出できることの自己テスト。
#[test]
fn scan_unapproved_model_pub_items_detects_offense() {
    let synthetic = "pub struct ModelRegistry;\npub fn rogue_method() {}\n";
    let offenses = scan_unapproved_model_pub_items(synthetic);
    assert_eq!(offenses.len(), 1, "offenses={offenses:?}");
    assert!(offenses[0].contains("rogue_method"));
}

/// [`fandhe_ai::model::{ModelRegistry, ModelError}`] が facade から
/// 到達可能であることをコンパイル時に固定する。`ModelError` は
/// `#[non_exhaustive]` のためワイルドカード腕を持つ `match` で
/// variant を網羅できることも併せて確認する。
#[test]
fn model_types_are_reachable_via_facade() {
    fn _assert_reachable(_registry: fandhe_ai::model::ModelRegistry) {}

    fn _assert_error_matchable(e: &fandhe_ai::model::ModelError) -> &'static str {
        match e {
            fandhe_ai::model::ModelError::CacheDirUnavailable => "cache_dir_unavailable",
            fandhe_ai::model::ModelError::InvalidComponent { .. } => "invalid_component",
            fandhe_ai::model::ModelError::NotFound { .. } => "not_found",
            fandhe_ai::model::ModelError::TooLarge { .. } => "too_large",
            fandhe_ai::model::ModelError::Load(_) => "load",
            fandhe_ai::model::ModelError::Io(_) => "io",
            _ => "unknown",
        }
    }
}

fn lib_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/lib.rs")
}

/// `tokens[after_name_idx..]`（mod 名の直後の位置）から `;`（外部
/// ファイル参照の終端）または対応する `{ ... }`（インライン本体）を
/// 読み飛ばし、読み飛ばした直後の index を返す（[`scan_top_level_pub_
/// mods`] の private mod・`#[path]` 除外分岐の共通処理）。ブレース対応は
/// 単純な深さカウント（コメント・文字列は呼び出し元で除去済み前提）。
fn skip_mod_declaration_body(tokens: &[String], after_name_idx: usize) -> usize {
    match tokens.get(after_name_idx).map(String::as_str) {
        Some(";") => after_name_idx + 1,
        Some("{") => {
            let mut depth = 1i32;
            let mut k = after_name_idx + 1;
            while k < tokens.len() && depth > 0 {
                match tokens[k].as_str() {
                    "{" => depth += 1,
                    "}" => depth -= 1,
                    _ => {}
                }
                k += 1;
            }
            k
        }
        // 文法上あり得ない形（構文エラー）。呼び出し元のループを無限に
        // 停止させないよう 1 トークンだけ進める。
        _ => after_name_idx,
    }
}

/// 渡されたトークン列（コメント・文字列除去済み・単一モジュールスコープ
/// 相当）の**直下**にある `pub mod <name>;`／`pub mod <name> { ... }` を
/// 収集する（[`collect_public_module_paths_recursive`] の下請け）。
///
/// - `pub(crate) mod`／`pub(super) mod`／`pub(self) mod`／`pub(in ...) mod`
///   は `pub` トークンの直後が `mod` ではなく `(` になるため一致しない
///   （`declares_pub_fn` と同型の判別）。
/// - 非 `pub` な `mod name { ... }`（private。`#[cfg(...)]` が付いていても
///   同様）はブレース対応だけ取って中身を丸ごと読み飛ばし、内部の宣言は
///   一切収集しない（外部から到達不能なため。合成入力テスト
///   `collect_public_module_paths_recursive_ignores_private_mod_subtree`
///   参照）。
/// - **モデル化できない `pub mod` 形は黙って除外せず fail-closed に
///   panic する**（codex P2 指摘・PR #2212 その 2: `#[path]` 属性付き
///   `pub mod` を「非公開扱いで除外」する旧実装は、そこから公開された
///   拡張が glob import 走査から静かに落ちる盲点だった）。対象は次の
///   3 種:
///   1. 属性ブロック（複数スタックも走査。`cfg_attr(.., path = ..)` の
///      ような間接形も含む）に `path` 識別子が現れ、属性列の直後が
///      `pub mod` であるもの（属性値の文字列リテラルは呼び出し元の
///      `strip_comments_and_literals` で既に空白化されており実際の
///      解決先ファイルを本関数だけでは特定できないため）。
///   2. 属性ブロックに `cfg`／`cfg_attr` 識別子が現れ、属性列の直後が
///      `pub mod` であるもの（cfg 条件次第で公開面がビルド構成ごとに
///      変わり、単一のビルド構成しか見ない本走査では機械的に判定
///      できないため）。**私有な `mod`（`pub` を伴わない）に付いた
///      `cfg` は対象外**（内部が最初から到達不能であることは cfg の
///      有無に関わらず変わらない）。
///   3. `pub mod <name>` の直後が `;`／`{` のいずれでもないもの
///      （raw identifier `pub mod r#ext;`・ジェネリクス構文の混入等。
///      識別子として認識できない場合も同様）。
/// - `fn`／`impl`／`struct` 等、mod 以外のブレース構造は対応する `}` まで
///   丸ごと読み飛ばす（内部の `mod` 宣言は外部から到達不能）。
fn scan_top_level_pub_mods(tokens: &[String]) -> Vec<(String, Option<Vec<String>>)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        // `#[ ... ]` 属性: 連続してスタックしうる（例: `#[cfg(test)]`
        // ＋ `#[allow(dead_code)]`）ため、対応する属性ブロックをすべて
        // 読み進めたうえで、その属性列のいずれかが `path`／`cfg`／
        // `cfg_attr` 識別子を含み、かつ属性列の直後が `pub mod` である
        // 場合に fail-closed panic する（上記ドキュメンテーションコメント
        // 参照）。
        if tokens[i] == "#" && tokens.get(i + 1).map(String::as_str) == Some("[") {
            let attrs_start = i;
            let mut j = i;
            let mut has_path = false;
            let mut has_cfg = false;
            while tokens.get(j).map(String::as_str) == Some("#")
                && tokens.get(j + 1).map(String::as_str) == Some("[")
            {
                let mut depth = 1i32;
                let mut k = j + 2;
                while k < tokens.len() && depth > 0 {
                    match tokens[k].as_str() {
                        "[" => depth += 1,
                        "]" => depth -= 1,
                        _ => {}
                    }
                    k += 1;
                }
                if tokens[j..k].iter().any(|t| t == "path") {
                    has_path = true;
                }
                if tokens[j..k].iter().any(|t| t == "cfg" || t == "cfg_attr") {
                    has_cfg = true;
                }
                j = k;
            }
            let next_is_pub_mod = tokens.get(j).map(String::as_str) == Some("pub")
                && tokens.get(j + 1).map(String::as_str) == Some("mod");
            if next_is_pub_mod {
                assert!(
                    !has_path,
                    "scan_top_level_pub_mods: `#[path = ...]`（`cfg_attr` 内の \
                     path を含む間接形も含む）が付いた `pub mod` はモデル化\
                     できない（属性値の文字列リテラルは前処理で空白化済みの\
                     ため実際の解決先ファイルを特定できない。facade は \
                     #[path] を使わない契約であり、これが現れること自体が\
                     想定外の構造）: tokens[{attrs_start}..{j}]={:?}",
                    &tokens[attrs_start..j]
                );
                assert!(
                    !has_cfg,
                    "scan_top_level_pub_mods: `#[cfg(...)]`／`#[cfg_attr(...)]` \
                     が付いた `pub mod` はモデル化できない（cfg 条件次第で \
                     公開面がビルド構成ごとに変わり、本走査は単一のビルド \
                     構成しか見ていないため機械的に判定できない）: \
                     tokens[{attrs_start}..{j}]={:?}",
                    &tokens[attrs_start..j]
                );
            }
            i = j;
            continue;
        }
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("mod") {
            let name_idx = i + 2;
            let name = tokens.get(name_idx).unwrap_or_else(|| {
                panic!(
                    "scan_top_level_pub_mods: `pub mod` の直後にモジュール名が\
                     見つからない（ファイル末尾で構文が打ち切られている等、\
                     モデル化できない構造）"
                )
            });
            assert!(
                name.chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_alphabetic() || c == '_'),
                "scan_top_level_pub_mods: `pub mod` の直後が識別子ではない\
                 （モデル化できない構造）: {name:?}"
            );
            match tokens.get(name_idx + 1).map(String::as_str) {
                Some(";") => {
                    out.push((name.clone(), None));
                    i = name_idx + 2;
                    continue;
                }
                Some("{") => {
                    let body_start = name_idx + 2;
                    let end_after_brace = skip_mod_declaration_body(tokens, name_idx + 1);
                    let body_end = end_after_brace - 1; // 対応する `}` の index
                    out.push((name.clone(), Some(tokens[body_start..body_end].to_vec())));
                    i = end_after_brace;
                    continue;
                }
                other => {
                    panic!(
                        "scan_top_level_pub_mods: `pub mod {name}` の直後が `;`／\
                         `{{` のいずれでもない（raw identifier 形\
                         〈`pub mod r#{name}`〉・ジェネリクス構文の混入等、\
                         モデル化できない形。直後のトークン: {other:?}）"
                    );
                }
            }
        }
        if tokens[i] == "mod" {
            // 到達するのは private mod のみ（`pub mod` は上の分岐で
            // 既に消費済み）。中身は収集せず読み飛ばす。
            let name_idx = i + 1;
            i = skip_mod_declaration_body(tokens, name_idx + 1);
            continue;
        }
        if tokens[i] == "{" {
            // mod 以外のブレース構造（fn 本体・impl・trait・struct 等）は
            // 対応する `}` まで丸ごと読み飛ばす。
            let mut depth = 1i32;
            let mut k = i + 1;
            while k < tokens.len() && depth > 0 {
                match tokens[k].as_str() {
                    "{" => depth += 1,
                    "}" => depth -= 1,
                    _ => {}
                }
                k += 1;
            }
            i = k;
            continue;
        }
        i += 1;
    }
    out
}

/// `dir` から `pub mod <name>;` の解決先ファイル（`<dir>/<name>.rs` また
/// は `<dir>/<name>/mod.rs`。Rust 2018 モジュール解決規約）を読み込み、
/// 内容と子モジュール探索用ディレクトリ（`<dir>/<name>/`）を返す。
/// 存在しない場合は fail-closed に panic する（`pub mod` 宣言と実ファイル
/// のドリフトを黙って見逃さない）。
fn resolve_external_pub_mod_file(dir: &Path, name: &str) -> (String, std::path::PathBuf) {
    let as_file = dir.join(format!("{name}.rs"));
    if as_file.is_file() {
        return (read_to_string_or_panic(&as_file), dir.join(name));
    }
    let as_mod_dir_file = dir.join(name).join("mod.rs");
    if as_mod_dir_file.is_file() {
        return (read_to_string_or_panic(&as_mod_dir_file), dir.join(name));
    }
    panic!(
        "resolve_external_pub_mod_file: `pub mod {name};` の解決先ファイルが\
         見つからない（{} も {} も存在しない）",
        as_file.display(),
        as_mod_dir_file.display()
    );
}

/// [`scan_top_level_pub_mods`] を再帰的に適用し、`tokens`（`dir` 直下の
/// あるファイル、または `prefix` が指すインライン `pub mod` 本体）から
/// 到達可能な全 `pub mod` パス（ドット区切りではなく `::` 区切り。例:
/// `nn::rnn`）を `out` へ収集する（[`collect_public_module_paths`] の
/// 下請け）。
fn collect_public_module_paths_recursive(
    tokens: &[String],
    dir: &Path,
    prefix: &str,
    out: &mut std::collections::BTreeSet<String>,
) {
    for (name, body) in scan_top_level_pub_mods(tokens) {
        let path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}::{name}")
        };
        out.insert(path.clone());
        match body {
            Some(inner_tokens) => {
                // インライン `pub mod <name> { ... }` 本体内の子 `pub mod
                // child;` は Rust 2018 の解決規約で `<dir>/<name>/child.rs`
                // （または `<dir>/<name>/child/mod.rs`）に置かれるため、
                // 子ディレクトリを `<dir>/<name>` へ進めてから再帰する
                // （親の `dir` のまま再帰すると `<dir>/child.rs` を誤って
                // 解決する。Cursor Bugbot 指摘・PR #2212）。
                collect_public_module_paths_recursive(&inner_tokens, &dir.join(&name), &path, out);
            }
            None => {
                let (content, child_dir) = resolve_external_pub_mod_file(dir, &name);
                let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
                let child_tokens = tokenize_including_punctuation(&cleaned);
                collect_public_module_paths_recursive(&child_tokens, &child_dir, &path, out);
            }
        }
    }
}

/// facade の `src/lib.rs` から到達可能な全 `pub mod` パス（ネスト含む。
/// 例: `nn::rnn`・`interop::onnx`・`interop::safetensors`）を再帰的に
/// 収集する（[`custom_function_hold_doctest_globs_all_pub_modules`]
/// 専用。codex-review 指摘・PR #2212: 旧実装 `declared_pub_mod_names` は
/// `lib.rs` 直下の `pub mod` 宣言しか見ておらず、`nn::rnn`／
/// `interop::onnx`／`interop::safetensors` のようなネストした公開面に
/// 生えた合成入口〈例: `compat::extensions::CustomExt` のような trait
/// 経由の `.custom()`〉を doctest のドリフト検査が見逃していた）。
fn collect_public_module_paths(src_dir: &Path) -> std::collections::BTreeSet<String> {
    let lib_rs = src_dir.join("lib.rs");
    let content = read_to_string_or_panic(&lib_rs);
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut out = std::collections::BTreeSet::new();
    collect_public_module_paths_recursive(&tokens, src_dir, "", &mut out);
    out
}

/// [`collect_public_module_paths_recursive`] が、インライン `pub mod`
/// 本体を再帰収集しつつ、private mod（`mod`／`pub(crate) mod`／
/// `pub(super) mod`／`pub(self) mod`／`pub(in crate::a) mod`）配下の
/// `pub mod`・fn 本体内の `pub mod` を到達不能として除外することを
/// 固定する合成入力テスト（codex-review 指摘・PR #2212 対応の中核）。
#[test]
fn collect_public_module_paths_recursive_ignores_private_mod_subtree() {
    let src = r#"
        pub mod a {
            pub mod extensions {
                pub trait CustomExt {}
            }
            mod hidden {
                pub mod x {}
            }
        }
        pub(crate) mod c {}
        pub(super) mod sup {}
        pub(self) mod slf {}
        pub(in crate::a) mod scoped {}
        mod d;
        fn f() {
            pub mod not_reachable {}
        }
    "#;
    let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut out = std::collections::BTreeSet::new();
    collect_public_module_paths_recursive(&tokens, Path::new("/nonexistent"), "", &mut out);
    let expected: std::collections::BTreeSet<String> = ["a", "a::extensions"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        out, expected,
        "private mod（scoped pub〈pub(crate)／pub(super)／pub(self)／\
         pub(in ...)〉を含む）・fn 本体内の pub mod が誤って公開パスとして\
         収集された、またはインライン pub mod の再帰収集に脱落がある"
    );
}

/// インライン `pub mod outer { pub mod child; }` の `child` が
/// `<dir>/outer/child.rs`（Rust 2018 規約）から解決され、その中の
/// `pub mod` まで再帰収集されることを固定する（Cursor Bugbot 指摘・
/// PR #2212: 親の `dir` のまま再帰すると `<dir>/child.rs` を誤って
/// 解決していた）。一時ディレクトリに実ファイルを置いて検証する。
#[test]
fn collect_public_module_paths_recursive_resolves_file_child_of_inline_mod() {
    let root = std::env::temp_dir().join(format!(
        "fandhe-ai-api-surface-inline-mod-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("outer")).expect("一時ディレクトリを作成できない");
    std::fs::write(
        root.join("outer").join("child.rs"),
        "pub mod grandchild {}\n",
    )
    .expect("child.rs を書き込めない");
    // 誤った解決先（`<dir>/child.rs`）には別内容を置き、誤解決なら
    // 収集結果が変わる（`wrong` が混入する）ようにする。
    std::fs::write(root.join("child.rs"), "pub mod wrong {}\n").expect("child.rs を書き込めない");

    let tokens = tokenize_including_punctuation("pub mod outer { pub mod child; }");
    let mut out = std::collections::BTreeSet::new();
    collect_public_module_paths_recursive(&tokens, &root, "", &mut out);
    let _ = std::fs::remove_dir_all(&root);

    let expected: std::collections::BTreeSet<String> =
        ["outer", "outer::child", "outer::child::grandchild"]
            .into_iter()
            .map(String::from)
            .collect();
    assert_eq!(
        out, expected,
        "インライン pub mod 配下のファイル子モジュールを <dir>/<outer>/child.rs から解決できていない"
    );
}

/// `src`（コメント・リテラル除去は呼び出し元が事前に済ませた生の
/// トークン列）から `scan_top_level_pub_mods` を呼び、panic したかどうか
/// を [`std::panic::catch_unwind`] で観測する（[`scan_top_level_pub_mods_
/// rejects_unmodelable_mod_forms`] 専用）。標準の panic hook が標準エラー
/// へ出力するメッセージは各テストケースの意図した panic であり異常では
/// ないため、走査中は一時的に hook を無効化する。
fn scan_top_level_pub_mods_panics(src: &str) -> bool {
    let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| {
        scan_top_level_pub_mods(&tokens);
    });
    std::panic::set_hook(previous_hook);
    result.is_err()
}

/// [`scan_top_level_pub_mods`] が、モデル化できない `pub mod` 形
/// （`#[path]`・`cfg_attr(.., path = ..)` 経由の間接 path・`#[cfg(...)]`／
/// `#[cfg_attr(...)]` が付いた `pub mod`・raw identifier 形）を黙って
/// 除外せず fail-closed に panic することを固定する（codex P2 指摘・
/// PR #2212 その 2 への対応。旧テスト `scan_top_level_pub_mods_excludes_
/// path_attribute_mod`——`#[path]` 付き `pub mod` を「非公開扱いで
/// 除外」される前提を固定していた——を置き換える）。負例
/// （`#[cfg(test)] mod tests {}`〈private mod への cfg は対象外〉・
/// `#[allow(dead_code)] pub mod ok {}`〈cfg 系でも path でもない属性〉）
/// は panic しないことも併せて固定する。
#[test]
fn scan_top_level_pub_mods_rejects_unmodelable_mod_forms() {
    let panicking_cases: &[&str] = &[
        // 1) 直接の `#[path = "..."]`。
        r#"#[path = "custom_location.rs"] pub mod weird;"#,
        // 2) `cfg_attr(.., path = ..)` 経由の間接 path。
        r#"#[cfg_attr(target_os = "macos", path = "mac.rs")] pub mod weird;"#,
        // 3) `#[cfg(...)]` が付いた `pub mod`。
        r#"#[cfg(feature = "x")] pub mod gated;"#,
        // 4) `#[cfg_attr(...)]`（path キーなし）が付いた `pub mod`。
        r#"#[cfg_attr(test, allow(dead_code))] pub mod gated2;"#,
        // 5) raw identifier 形。
        "pub mod r#ext;",
        // 6) 属性が複数スタックし、そのうち 1 つに cfg が含まれる場合。
        r#"#[allow(dead_code)] #[cfg(test)] pub mod stacked;"#,
    ];
    for src in panicking_cases {
        assert!(
            scan_top_level_pub_mods_panics(src),
            "モデル化できない pub mod 形が panic せず黙って通過した: {src:?}"
        );
    }

    let non_panicking_cases: &[&str] = &[
        // private mod への cfg は対象外（中身は最初から到達不能）。
        "#[cfg(test)] mod tests {}",
        // path でも cfg 系でもない属性は無関係。
        "#[allow(dead_code)] pub mod ok {}",
        // 属性なしの通常形。
        "pub mod plain;",
    ];
    for src in non_panicking_cases {
        assert!(
            !scan_top_level_pub_mods_panics(src),
            "モデル化可能な pub mod 形が誤って panic した: {src:?}"
        );
    }
}

/// `src/lib.rs` の `struct <struct_name>;` 宣言に**直接**付いた
/// ドキュメンテーションコメント（`///` の連続。空行・非 `///` 行で途切れた
/// 時点で走査を止める）を、直前の `#[allow(dead_code)]`・
/// `#[cfg(doctest)]` の並びを検証したうえで抽出する（[`custom_function_
/// hold_doctest_probe_body_matches_fixed_contract`]・[`nn_module_hold_
/// doctest_probe_body_matches_fixed_contract`] 共用。元は
/// `VarCustomHoldDoctestGuard` 専用の固定名関数だったが、#2133 で
/// `NnModuleHoldDoctestGuard` にも同じ抽出が必要になったため `struct_name`
/// 引数版へ最小限リファクタした。挙動は `struct_name` に `"VarCustomHold
/// DoctestGuard"` を渡した場合と不変）。`#[path]` 付き `pub mod` の非公開
/// 扱い除外（`scan_top_level_pub_mods`）と同種の「モデル化できない構造は
/// 解決せず fail-closed に拒否する」方針で、`#[cfg(doctest)]` が直前に
/// 見つからない・`struct <struct_name>;` 宣言自体が見つからない場合は
/// panic する（doc コメントの取り違えによる drift 検査の無力化を防ぐ）。
/// 返り値は `///` 接頭辞（と直後の 1 個のスペース。rustdoc の正規化と同じ
/// 規約）を除去した行の列。
fn extract_hold_doctest_guard_doc(content: &str, struct_name: &str) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let target = format!("struct {struct_name};");
    let struct_idx = lines
        .iter()
        .position(|l| l.trim() == target)
        .unwrap_or_else(|| {
            panic!(
                "src/lib.rs に `{target}` 宣言が見つからない\
             （否定ガードの本命足場自体が削除・改名された可能性がある）"
            )
        });
    assert!(struct_idx >= 2, "{target} の直前に属性 2 行分の余地がない");
    assert_eq!(
        lines[struct_idx - 1].trim(),
        "#[allow(dead_code)]",
        "{target} の直前が `#[allow(dead_code)]` ではない"
    );
    assert_eq!(
        lines[struct_idx - 2].trim(),
        "#[cfg(doctest)]",
        "{target} の直前が `#[cfg(doctest)]` ではない\
         （本足場が `cfg(doctest)` 外で有効化され、通常ビルドを壊しうる）"
    );

    let mut doc_lines: Vec<String> = Vec::new();
    let mut i = struct_idx - 2; // `#[cfg(doctest)]` 行の index
    while i > 0 && lines[i - 1].trim_start().starts_with("///") {
        i -= 1;
        let raw = lines[i].trim_start();
        let rest = raw.strip_prefix("///").unwrap_or(raw);
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        doc_lines.push(rest.to_string());
    }
    // 上のループは `#[cfg(doctest)]` の直前から**上へ**遡って収集する
    // ため、収集順は文書の末尾行から先頭行への逆順になっている。
    // 元の文書順（先頭 → 末尾）へ戻す。
    doc_lines.reverse();
    assert!(
        !doc_lines.is_empty(),
        "{target} に直接付いた `///` doc コメントが見つからない"
    );
    doc_lines
}

/// `doc_lines`（[`extract_hold_doctest_guard_doc`] の戻り値）
/// から、厳密に裸の ```` ``` ```` フェンス（`ignore`／`no_run`／
/// `compile_fail` 等の修飾を一切伴わない）で区切られた doctest ブロックを
/// **ちょうど 1 つ**抽出し、その本文行（フェンス自体を含まない）を返す。
/// フェンス開始行が裸の ```` ``` ```` でない場合・ブロック数が 1 で
/// ない場合・フェンスが閉じられていない場合は fail-closed に panic する
/// （codex-review 指摘・PR #2212: stable rustdoc は `compile_fail,EXXXX`
/// のエラーコードを照合しないため、`compile_fail` への回帰・`ignore`
/// 等での doctest 無効化を機械的に拒否する）。
fn extract_single_bare_fenced_doctest_block(doc_lines: &[String]) -> Vec<String> {
    let mut blocks: Vec<Vec<String>> = Vec::new();
    let mut current: Option<Vec<String>> = None;
    for line in doc_lines {
        let trimmed = line.trim_end();
        if let Some(body) = current.as_mut() {
            if trimmed == "```" {
                blocks.push(std::mem::take(body));
                current = None;
            } else {
                body.push(line.clone());
            }
            continue;
        }
        // フェンス候補行の判定は「先頭の連続バッククォート数がちょうど
        // 3」の場合に限る。本 doc コメントの地の文（このコメント自体
        // 含む）は旧実装の説明で ```` ```compile_fail,E0599 ```` の
        // ような 4 連続バッククォート（本規約の quad-fence 引用記法。
        // 3 連続を含む文字列をインライン引用する際に使う）を使うため、
        // 先頭 3 連続だけで判定すると地の文を誤ってフェンス開始と
        // 誤検出する（自己レビューで判明）。
        let leading_backticks = trimmed
            .trim_start()
            .chars()
            .take_while(|&c| c == '`')
            .count();
        if leading_backticks == 3 {
            assert_eq!(
                trimmed.trim_start(),
                "```",
                "VarCustomHoldDoctestGuard のフェンス開始行が裸の ``` ではない\
                 （`ignore`／`no_run`／`compile_fail` 等の修飾が付いている。\
                 stable rustdoc はエラーコードを照合しないため compile_fail への\
                 回帰は空合格を招く）: {trimmed:?}"
            );
            current = Some(Vec::new());
        }
    }
    assert!(
        current.is_none(),
        "VarCustomHoldDoctestGuard の doctest フェンスが閉じられていない"
    );
    assert_eq!(
        blocks.len(),
        1,
        "VarCustomHoldDoctestGuard の doctest ブロック数が 1 ではない\
         （正のプローブ 1 ブロック方式からの逸脱）: {}",
        blocks.len()
    );
    blocks.into_iter().next().unwrap_or_default()
}

/// `block_lines`（doctest ブロック本文）を、`use fandhe_ai::<mod>::*;`
/// 形（ネストした `pub mod` の glob import。クレートルート自体の
/// `use fandhe_ai::*;` は対象外）の集合と、それ以外の本文行に分離する
/// （[`custom_function_hold_doctest_globs_all_pub_modules`]・
/// [`custom_function_hold_doctest_probe_body_matches_fixed_contract`]
/// 共用）。
fn split_glob_imports_and_probe_body(
    block_lines: &[String],
) -> (std::collections::BTreeSet<String>, Vec<String>) {
    let mut globs = std::collections::BTreeSet::new();
    let mut body = Vec::new();
    for line in block_lines {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("use fandhe_ai::")
            && let Some(path) = rest.strip_suffix("::*;")
            && !path.is_empty()
        {
            globs.insert(path.to_string());
            continue;
        }
        body.push(line.clone());
    }
    (globs, body)
}

/// `VarCustomHoldDoctestGuard` の唯一の doctest ブロックが glob import
/// するネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod` 宣言
/// 集合が一致することを固定する（doctest 本文と `pub mod` 宣言の
/// ドリフト防止。新しい `pub mod` を facade へ追加した際、doctest 側の
/// `use` 一覧の更新を機械的に強制する）。
#[test]
fn custom_function_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarCustomHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "VarCustomHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`custom_function_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**（`__FandheHoldProbe` トレイト定義・`Var`／`Tape`／
/// `compat::Sequential` への実装・`__probe_*` 関数群）が固定文言
/// [`HOLD_PROBE_BODY`] と 1 行たりとも違わず一致することを固定する
/// （codex-review 指摘・PR #2212: rustdoc の `# ` 隠し行・プローブの
/// 削除・別名へのシャドーイング等で正のプローブを骨抜きにする改変を
/// 機械的に拒否する。glob 集合のドリフト検査だけでは本文の改変・
/// 削除を検出できないため、両テストは互いに独立した防御層を成す）。
#[test]
fn custom_function_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarCustomHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, HOLD_PROBE_BODY,
        "VarCustomHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 HOLD_PROBE_BODY からドリフトしている。正のプローブ\
         （__FandheHoldProbe トレイト・各型への実装・__probe_* 関数）の\
         削除・弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`custom_function_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の `VarCustomHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致する
/// 必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に含む）。
const HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
struct __FandheHoldMarker;\n\
\n\
trait __FandheHoldProbe {\n\
\x20\x20\x20\x20fn custom(&self) -> __FandheHoldMarker;\n\
\x20\x20\x20\x20fn add_custom(&self) -> __FandheHoldMarker;\n\
}\n\
\n\
impl<'t> __FandheHoldProbe for fandhe_ai::Var<'t> {\n\
\x20\x20\x20\x20fn custom(&self) -> __FandheHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn add_custom(&self) -> __FandheHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn custom(&self) -> __FandheHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn add_custom(&self) -> __FandheHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn custom(&self) -> __FandheHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn add_custom(&self) -> __FandheHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_var(x: &fandhe_ai::Var<'_>) {\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = fandhe_ai::Var::custom(x);\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = x.custom();\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = fandhe_ai::Var::add_custom(x);\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = x.add_custom();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = fandhe_ai::Tape::custom(x);\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = x.custom();\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = fandhe_ai::Tape::add_custom(x);\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = x.add_custom();\n\
}\n\
\n\
fn __probe_sequential(x: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = fandhe_ai::compat::Sequential::custom(x);\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = x.custom();\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = fandhe_ai::compat::Sequential::add_custom(x);\n\
\x20\x20\x20\x20let _: __FandheHoldMarker = x.add_custom();\n\
}";

// =====================================================================
// #2133（親 #2132・#2131）の facade 公開保留固定（`NnModuleHoldDoctestGuard`）。
// `VarCustomHoldDoctestGuard` 系（上記 2 テスト・HOLD_PROBE_BODY）と同型の
// 正のプローブ 1 ブロック方式のドリフト検査。承認未取得の経緯・多層防御の
// 位置づけは `docs/facade-nn-module-exposure-decision.md` §12 参照。
// =====================================================================

/// [`custom_function_hold_doctest_globs_all_pub_modules`] の
/// `NnModuleHoldDoctestGuard` 版。`crates/facade/src/lib.rs` の
/// `NnModuleHoldDoctestGuard` doc 内の唯一の doctest ブロックが glob
/// import するネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod`
/// 宣言集合が一致することを固定する。
#[test]
fn nn_module_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "NnModuleHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "NnModuleHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`custom_function_hold_doctest_probe_body_matches_fixed_contract`] の
/// `NnModuleHoldDoctestGuard` 版。doctest ブロックの glob 以外の本文
/// （ローカル `__fandhe_nn_hold_probe` モジュール定義・`use` ・`__probe`
/// 関数）が固定文言 [`NN_MODULE_HOLD_PROBE_BODY`] と 1 行たりとも違わず
/// 一致することを固定する（`# ` 隠し行・プローブの削除・別名への
/// シャドーイング等で正のプローブを骨抜きにする改変を機械的に拒否する）。
#[test]
fn nn_module_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "NnModuleHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, NN_MODULE_HOLD_PROBE_BODY,
        "NnModuleHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 NN_MODULE_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_nn_hold_probe モジュール・__probe 関数）の\
         削除・弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`nn_module_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の `NnModuleHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致する
/// 必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に含む）。
const NN_MODULE_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_nn_hold_probe {\n\
\x20\x20\x20\x20pub trait Module {}\n\
\x20\x20\x20\x20pub struct ModuleList;\n\
}\n\
use __fandhe_nn_hold_probe::*;\n\
\n\
fn __probe(_: &dyn Module, _: ModuleList, _: &Sequential) {}";

/// facade src の全 `pub use` 文（`pub(..) use` は対象外）から
/// [`collect_pub_use_leaves`] で葉（ソース側・rename 前）を集め、葉が
/// `Module`／`ModuleList` である行、または葉が `Sequential` かつパスに
/// `fandhe_ai_autodiff`／`nn` を含む行を違反とする（#2133 Step 2-1）。
/// `compat/mod.rs` の `pub use sequential::{Sequential, SequentialVars};`
/// （パスに `fandhe_ai_autodiff`／`nn` を含まない）は正当な既存形のため
/// 許容する。別名（`as Layer` 等）の前のソース側の葉で判定するため、
/// `pub use fandhe_ai_autodiff::nn::Module as Layer;` のような別名
/// 再エクスポートも検出する。
#[test]
fn facade_does_not_reexport_nn_module_or_containers() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let mut i = 0usize;
        while i < tokens.len() {
            if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
                let mut end = i + 2;
                while end < tokens.len() && tokens[end] != ";" {
                    end += 1;
                }
                let path_tokens = &tokens[i + 2..end.min(tokens.len())];
                let leaves = collect_pub_use_leaves(path_tokens);
                let path_contains_nn_autodiff = path_tokens
                    .iter()
                    .any(|t| t == "fandhe_ai_autodiff" || t == "nn");
                for leaf in leaves {
                    let offense = match leaf.as_str() {
                        "Module" | "ModuleList" => true,
                        "Sequential" => path_contains_nn_autodiff,
                        _ => false,
                    };
                    if offense {
                        offending.push(format!("{}: leaf={leaf}", path.display()));
                    }
                }
                i = (end + 1).min(tokens.len());
                continue;
            }
            i += 1;
        }
    });
    assert!(
        offending.is_empty(),
        "facade の pub use が nn::Module／ModuleList／nn 系 Sequential を\
         再エクスポートしている（#2133 未承認のまま対象外という設計判断に\
         違反）: {offending:?}"
    );
}

/// [`facade_does_not_reexport_nn_module_or_containers`] の自己テスト
/// （正例・負例の合成入力）。
#[test]
fn facade_does_not_reexport_nn_module_or_containers_detects_each_category() {
    fn offenses(content: &str) -> Vec<String> {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let mut offending = Vec::new();
        let mut i = 0usize;
        while i < tokens.len() {
            if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
                let mut end = i + 2;
                while end < tokens.len() && tokens[end] != ";" {
                    end += 1;
                }
                let path_tokens = &tokens[i + 2..end.min(tokens.len())];
                let leaves = collect_pub_use_leaves(path_tokens);
                let path_contains_nn_autodiff = path_tokens
                    .iter()
                    .any(|t| t == "fandhe_ai_autodiff" || t == "nn");
                for leaf in leaves {
                    let offense = match leaf.as_str() {
                        "Module" | "ModuleList" => true,
                        "Sequential" => path_contains_nn_autodiff,
                        _ => false,
                    };
                    if offense {
                        offending.push(format!("leaf={leaf}"));
                    }
                }
                i = (end + 1).min(tokens.len());
                continue;
            }
            i += 1;
        }
        offending
    }

    // 正例。
    assert!(!offenses("pub use fandhe_ai_autodiff::nn::Module;").is_empty());
    assert!(!offenses("pub use fandhe_ai_autodiff::nn::{Module as Layer};").is_empty());
    assert!(!offenses("pub use fandhe_ai_autodiff::nn::{self as n, ModuleList};").is_empty());
    assert!(!offenses("pub use fandhe_ai_autodiff::nn::Sequential;").is_empty());

    // 負例: `compat::Sequential`（パスに fandhe_ai_autodiff／nn を含まない）。
    assert!(offenses("pub use sequential::{Sequential, SequentialVars};").is_empty());
    // 負例: 非 pub。
    assert!(offenses("use fandhe_ai_autodiff::nn::Module;").is_empty());
}

/// facade src に facade 独自の `trait Module`／`struct ModuleList`／
/// `enum`／`type` 版・`struct Sequential`（`src/compat/sequential.rs` 以外）
/// の宣言が可視性を問わず存在しないことを固定する（#2133 Step 2-2。
/// `declares_fn_named` と同型のトークン走査で `trait`／`struct`／`enum`／
/// `type` トークンの直後の識別子を見る）。`use fandhe_ai_autodiff::nn::
/// {..., Module, ...}`（非公開 import）・`Box<dyn Module>`（型参照）・
/// `compat/sequential.rs` 自身の `pub struct Sequential` は正当な既存形
/// のため許容する。
#[test]
fn facade_declares_no_nn_module_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_nn_module_item_declarations(content, path) {
            offending.push(offense);
        }
    });
    assert!(
        offending.is_empty(),
        "facade src に nn::Module／ModuleList 相当の独自宣言が見つかった\
         （#2133 未承認のまま対象外という設計判断に違反）: {offending:?}"
    );
}

/// `content`（`path` 由来）を走査し、`trait`／`struct`／`enum`／`type`
/// トークンの直後に `Module`／`ModuleList` が続く宣言、または `Sequential`
/// が続く宣言（`path` のファイル名が `compat/sequential.rs` 以外）を
/// 検出して違反文字列の列を返す（[`facade_declares_no_nn_module_items`]・
/// その自己テスト共用）。
fn scan_nn_module_item_declarations(content: &str, path: &Path) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let is_compat_sequential = path
        .to_string_lossy()
        .replace('\\', "/")
        .ends_with("compat/sequential.rs");
    let mut offending = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        if !matches!(token.as_str(), "trait" | "struct" | "enum" | "type") {
            continue;
        }
        let Some(name) = tokens.get(i + 1).map(String::as_str) else {
            continue;
        };
        let offense = match name {
            "Module" | "ModuleList" => true,
            "Sequential" => !is_compat_sequential,
            _ => false,
        };
        if offense {
            offending.push(format!("{}: `{token} {name}`", path.display()));
        }
    }
    offending
}

/// [`facade_declares_no_nn_module_items`]（[`scan_nn_module_item_
/// declarations`]）の自己テスト（正例・負例の合成入力）。
#[test]
fn facade_declares_no_nn_module_items_detects_each_category() {
    let other = Path::new("src/nn/module.rs");
    let compat_seq = Path::new("src/compat/sequential.rs");

    // 正例。
    assert!(!scan_nn_module_item_declarations("pub trait Module {}", other).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct ModuleList;", other).is_empty());
    assert!(
        !scan_nn_module_item_declarations("pub type Layer = u8; pub struct ModuleList;", other)
            .is_empty()
    );
    assert!(!scan_nn_module_item_declarations("pub struct Sequential;", other).is_empty());

    // 負例: 非公開 import・型参照。
    assert!(
        scan_nn_module_item_declarations(
            "use fandhe_ai_autodiff::nn::{Linear, Module, ReLU};",
            other
        )
        .is_empty()
    );
    assert!(
        scan_nn_module_item_declarations(
            "pub(crate) fn layers(&self) -> &[Box<dyn Module>] {}",
            other
        )
        .is_empty()
    );
    // 負例: compat/sequential.rs 自身の Sequential 宣言。
    assert!(scan_nn_module_item_declarations("pub struct Sequential {}", compat_seq).is_empty());
    // 負例: コメント・文字列リテラル中。
    assert!(scan_nn_module_item_declarations("// pub trait Module {}", other).is_empty());
}

/// `src/compat` 配下に `add_module`／`add_boxed`／`push_module` の
/// `pub fn` 宣言が存在しないことを固定する（#2133 Step 2-3。
/// `compat_sequential_does_not_expose_rnn_add_methods` と同型。
/// `docs/facade-nn-module-exposure-decision.md` §9 で `add_module` は
/// スコープ外と明記済み）。
#[test]
fn compat_sequential_does_not_expose_module_add_methods() {
    let compat_dir = facade_crate_root().join("src/compat");
    let forbidden = ["add_module", "add_boxed", "push_module"];
    let mut offenses = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        for name in forbidden {
            if contains_pub_fn_declaration(content, name) {
                offenses.push(format!("{}: pub fn {name}", path.display()));
            }
        }
    });
    assert!(
        offenses.is_empty(),
        "src/compat 配下に add_module／add_boxed／push_module が見つかった\
         （承認スコープ〈#2133〉は Sequential への追加を認めていない）: {offenses:?}"
    );
}

/// [`compat_sequential_does_not_expose_module_add_methods`] の自己テスト。
#[test]
fn compat_sequential_does_not_expose_module_add_methods_detects_offense() {
    assert!(contains_pub_fn_declaration(
        "pub fn add_module(&mut self, m: impl Module + 'static) {}",
        "add_module"
    ));
    assert!(!contains_pub_fn_declaration(
        "pub fn add_linear(&mut self, l: Linear) {}",
        "add_module"
    ));
}

/// facade（crates.io 公開クレート `fandhe-ai`）の `Cargo.toml` が
/// `doctest = false` を持たないことを固定する（イシュー #2064 PR #2212
/// codex-review 指摘への対応: `[lib] doctest = false` を設定されると
/// `VarCustomHoldDoctestGuard` の正のプローブ doctest が `cargo test`
/// で一切実行されなくなり、本命ガードが静かに無力化される）。
#[test]
fn facade_cargo_toml_keeps_doctests_enabled() {
    let cargo_toml = read_to_string_or_panic(&facade_crate_root().join("Cargo.toml"));
    for line in cargo_toml.lines() {
        let trimmed = line.trim();
        let code_part = trimmed.split_once('#').map(|(a, _)| a).unwrap_or(trimmed);
        let normalized: String = code_part.chars().filter(|c| !c.is_whitespace()).collect();
        assert_ne!(
            normalized, "doctest=false",
            "crates/facade/Cargo.toml が `doctest = false` を設定しており、\
             VarCustomHoldDoctestGuard の正のプローブ doctest が実行されなく\
             なっている（本命ガードの無力化）"
        );
    }
}

/// `tokens[idx]`（`"fn"` トークンであることは呼び出し元が保証する）の
/// 直後に、通常形（`fn <target>(`／`fn <target><`）または raw
/// identifier 形（`fn r # <target>(`／`fn r # <target><`。トークナイザ
/// が `r#custom` を `"r"`・`"#"`・`"custom"` の 3 トークンへ分解する
/// ため個別に判定する）で `target` という名前の宣言が続くかを判定する
/// （[`count_fn_declarations_by_name`] 専用）。可視性・宣言文脈
/// （inherent impl・trait impl・trait 定義〈デフォルトメソッド含む〉・
/// blanket impl・自由関数・マクロ本体内のいずれか）は問わない——`fn`
/// トークンの直後の位置だけで判定するため、`fn` より前に付く修飾子
/// （`pub`／`pub(crate)`／`unsafe`／`const`／`async`／`extern "C"` 等）
/// は本判定に一切影響しない（`fn(i32)` のような関数ポインタ型も、直後が
/// `target` という識別子ではなく `(` のため自然に除外される）。
fn fn_declaration_target_name_matches(tokens: &[String], idx: usize, target: &str) -> bool {
    debug_assert_eq!(tokens.get(idx).map(String::as_str), Some("fn"));
    // raw identifier 形（`fn r#custom(`）。
    if tokens.get(idx + 1).map(String::as_str) == Some("r")
        && tokens.get(idx + 2).map(String::as_str) == Some("#")
        && tokens.get(idx + 3).map(String::as_str) == Some(target)
        && matches!(
            tokens.get(idx + 4).map(String::as_str),
            Some("(") | Some("<")
        )
    {
        return true;
    }
    // 通常形（`fn custom(`）。
    tokens.get(idx + 1).map(String::as_str) == Some(target)
        && matches!(
            tokens.get(idx + 2).map(String::as_str),
            Some("(") | Some("<")
        )
}

/// `tokens`（コメント・リテラル除去済みソースのトークン列）中に現れる
/// `fn <target_name>` 宣言の総数を数える（[`workspace_declares_custom_
/// fn_only_on_tape`] 専用。可視性・宣言文脈を問わず数え上げる契約は
/// [`fn_declaration_target_name_matches`] のドキュメンテーションコメント
/// 参照）。
fn count_fn_declarations_by_name(tokens: &[String], target_name: &str) -> usize {
    tokens
        .iter()
        .enumerate()
        .filter(|(i, token)| {
            token.as_str() == "fn" && fn_declaration_target_name_matches(tokens, *i, target_name)
        })
        .count()
}

/// [`fn_declaration_target_name_matches`]／[`count_fn_declarations_by_name`]
/// が、可視性・宣言文脈（trait のデフォルトメソッド・blanket impl・
/// 参照型への impl・raw identifier・extern ABI 修飾子付き）を問わず
/// `fn custom`／`fn add_custom` 宣言を検出し、コメント・文字列・raw
/// 文字列中の同型テキストは検出しないことを固定する合成入力テスト
/// （[`workspace_declares_custom_fn_only_on_tape`] の検出ロジック自体の
/// 自己テスト）。
#[test]
fn count_fn_declarations_by_name_detects_all_declaration_contexts() {
    let positive_cases_custom: &[&str] = &[
        // trait のデフォルトメソッド＋空 impl。
        "pub trait Ext { fn custom(&self) -> u8 { 0 } } impl Ext for Var {}",
        // blanket impl。
        "impl<T> Ext for T { fn custom(&self) {} }",
        // 参照型・ライフタイム省略への impl。
        "impl Ext for &Var<'_> { fn custom(&self) {} }",
        // raw identifier。
        "impl Var { fn r#custom() {} }",
        // extern ABI 修飾子付き（`fn` トークン直後の判定には無関係だが、
        // `fn` より前の修飾子が判定を妨げないことも併せて確認する）。
        "impl Var { pub extern \"C\" fn custom() {} }",
        // マクロ本体内。
        "macro_rules! m { () => { fn custom(&self) {} } }",
    ];
    for src in positive_cases_custom {
        let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        assert_eq!(
            count_fn_declarations_by_name(&tokens, "custom"),
            1,
            "src={src:?} tokens={tokens:?}"
        );
    }

    let positive_cases_add_custom: &[&str] = &[
        "trait Ext { fn add_custom<T>(&self, v: T); }",
        "impl Var { fn r#add_custom() {} }",
    ];
    for src in positive_cases_add_custom {
        let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        assert_eq!(
            count_fn_declarations_by_name(&tokens, "add_custom"),
            1,
            "src={src:?} tokens={tokens:?}"
        );
    }

    // 負例: コメント・文字列・raw 文字列中は検出しない。関数ポインタ型
    // （`fn(...)`）・別名関数（`custom_extra`）も検出しない。
    let negative_cases: &[&str] = &[
        "// fn custom(&self) {}",
        "let s = \"fn custom(\";",
        "let r = r#\"fn custom(\"#;",
        "fn custom_extra(&self) {}",
        "type F = fn(u8) -> u8;",
        "#[cfg(test)] mod tests { fn helper() {} }",
    ];
    for src in negative_cases {
        let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        assert_eq!(
            count_fn_declarations_by_name(&tokens, "custom"),
            0,
            "src={src:?} tokens={tokens:?}"
        );
    }
}

/// `crates/` 直下の各クレート（非公開クレート `docs-site`・
/// `bench-harness`・`guardrail`・`self-repair` を含む全メンバー）を
/// 走査し、その `crate_dir/src/` 配下の相対パスを返す（`crates/<name>`
/// 自体は含まない）。
fn workspace_crates_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate>/ の親ディレクトリ（crates/）が取得できない")
        .to_path_buf()
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、コメント・リテラルを
/// 除去したトークン列上で `fn`／（`custom`｜`add_custom`｜raw identifier
/// 形）の宣言が、可視性・宣言文脈（inherent impl・trait impl・trait
/// 定義〈デフォルトメソッド含む〉・blanket impl・自由関数・マクロ本体
/// 内のいずれか）を問わず現れる箇所を全て数え上げ、その集合が
/// `crates/autodiff/src/tape.rs`（`Tape::custom`。イシュー #1946 案 B）
/// の 1 件のみであることを固定する（workspace 全体の定義元インベント
/// リ。イシュー #2064 PR #2212 codex-review 指摘〈P2〉への対応: facade
/// のソース走査・`VarCustomHoldDoctestGuard` の正のプローブはいずれも
/// 「facade から到達可能か」しか見ないため、facade の外
/// （`onnx-interop`・`backend-*`・`tensor-core` 等）に `custom`／
/// `add_custom` を持つ trait impl が新設され、facade がそれを glob
/// できる形で将来公開してしまった場合に備え、そもそもの定義元を先に
/// 塞ぐ多層防御の最内層とする）。
#[test]
fn workspace_declares_custom_fn_only_on_tape() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    let Ok(entries) = std::fs::read_dir(&crates_dir) else {
        panic!(
            "workspace crates ディレクトリが読めない: {}",
            crates_dir.display()
        );
    };
    let mut crate_dirs: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(
        !crate_dirs.is_empty(),
        "workspace crates ディレクトリ配下にクレートが 1 件も見つからない\
         （テスト自体が検査対象を見失っている可能性がある）"
    );

    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            for fn_name in ["custom", "add_custom"] {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    let rel = path
                        .strip_prefix(&crates_dir)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }

    let expected: std::collections::BTreeMap<String, usize> =
        [("autodiff/src/tape.rs::custom".to_string(), 1usize)]
            .into_iter()
            .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の `fn custom`／`fn add_custom` 宣言\
         集合が `crates/autodiff/src/tape.rs::custom`（1 件）のみという\
         期待と一致しない（過不足いずれも fail-closed に検出する。新たな\
         定義元が見つかった場合、それが承認済みの §12.5 (b) 実装なのか\
         迂回経路の混入なのかを確認すること）: {found:?}"
    );
}

/// `content`（facade src の 1 ファイル）を走査し、本ファイルのソース
/// 走査ガード群がモデル化できない構造（[`facade_source_uses_only_
/// modelable_structures`] 専用）を全て列挙する。違反時は個別のオフェンス
/// 文字列を返す（空なら違反なし）。検出対象:
/// - `#[..]`／`#![..]` 内の `path` 識別子（`#[path]`。走査ロジックが
///   実ファイルを特定できない構造）
/// - `#[cfg(...)]`／`#[cfg_attr(...)]` が付いた `pub use`／`pub mod`
///   （複数属性スタックも許容。cfg 条件次第で公開面がビルド構成ごとに
///   変わり、単一のビルド構成しか見ない本走査では機械的に判定できない）
/// - `include!`（`include_str!`／`include_bytes!` は対象外。ファイル
///   内容を静的にインライン展開し、ソース走査が見ているテキストと
///   実際にコンパイルされる内容が乖離しうる）
/// - `macro_rules!` 定義（マクロ展開後の実際のアイテムをソース走査が
///   静的に把握できない）
/// - `extern crate`（2018 edition 以降では通常不要な明示的クレート
///   参照で、想定外の別名 import 経路になりうる）
/// - raw identifier（`r#ident`。`declares_fn_named` 等は raw identifier
///   を正規化して検出するが、本走査対象の構造検査自体は raw identifier
///   の使用そのものを許さない契約とする——facade src には raw
///   identifier を要する識別子〈予約語衝突〉が存在しないため）
/// - `pub use ...::*`（glob 再エクスポート。再エクスポートされる識別子
///   集合が静的に列挙できなくなる）
/// - `pub use` の葉が `self`（`pub use foo::{self, bar};` 等。モジュール
///   自体を別名で再エクスポートする形で、[`collect_pub_use_leaves`] の
///   葉 allowlist 検査の対象外になってしまう）
fn scan_facade_unmodelable_structures(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offenses = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        // 属性ブロック（`#[...]`／`#![...]`）。
        if tokens[i] == "#" {
            let mut open_idx = i + 1;
            if tokens.get(open_idx).map(String::as_str) == Some("!") {
                open_idx += 1;
            }
            if tokens.get(open_idx).map(String::as_str) == Some("[") {
                let attr_start = i;
                let mut depth = 1i32;
                let mut m = open_idx + 1;
                while m < tokens.len() && depth > 0 {
                    match tokens[m].as_str() {
                        "[" => depth += 1,
                        "]" => depth -= 1,
                        _ => {}
                    }
                    m += 1;
                }
                if tokens[attr_start..m].iter().any(|t| t == "path") {
                    offenses.push(format!(
                        "attribute contains `path` identifier: {:?}",
                        &tokens[attr_start..m]
                    ));
                }
                if tokens[attr_start..m]
                    .iter()
                    .any(|t| t == "cfg" || t == "cfg_attr")
                {
                    // 属性列直後（複数スタックも許容）が `pub use`／
                    // `pub mod` であれば違反。
                    let mut after = m;
                    loop {
                        if tokens.get(after).map(String::as_str) != Some("#") {
                            break;
                        }
                        let mut next_open = after + 1;
                        if tokens.get(next_open).map(String::as_str) == Some("!") {
                            next_open += 1;
                        }
                        if tokens.get(next_open).map(String::as_str) != Some("[") {
                            break;
                        }
                        let mut d2 = 1i32;
                        let mut kk = next_open + 1;
                        while kk < tokens.len() && d2 > 0 {
                            match tokens[kk].as_str() {
                                "[" => d2 += 1,
                                "]" => d2 -= 1,
                                _ => {}
                            }
                            kk += 1;
                        }
                        after = kk;
                    }
                    let gated_pub_use_or_mod = tokens.get(after).map(String::as_str) == Some("pub")
                        && matches!(
                            tokens.get(after + 1).map(String::as_str),
                            Some("use") | Some("mod")
                        );
                    if gated_pub_use_or_mod {
                        offenses.push(format!(
                            "cfg/cfg_attr gates pub use/pub mod: {:?}",
                            &tokens[attr_start..m]
                        ));
                    }
                }
                i = m;
                continue;
            }
        }
        if tokens[i] == "include" && tokens.get(i + 1).map(String::as_str) == Some("!") {
            offenses.push("include! macro invocation".to_string());
        }
        if tokens[i] == "macro_rules" && tokens.get(i + 1).map(String::as_str) == Some("!") {
            offenses.push("macro_rules! definition".to_string());
        }
        if tokens[i] == "extern" && tokens.get(i + 1).map(String::as_str) == Some("crate") {
            offenses.push("extern crate declaration".to_string());
        }
        if tokens[i] == "r"
            && tokens.get(i + 1).map(String::as_str) == Some("#")
            && tokens
                .get(i + 2)
                .and_then(|t| t.chars().next())
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        {
            offenses.push(format!("raw identifier r#{}", tokens[i + 2]));
        }
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            // use tree の内部に `;` は現れないため、次の `;` までを
            // この文のスパンとみなす。
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let span_end = (end + 1).min(tokens.len());
            let span = &tokens[i..span_end];
            if span.iter().any(|t| t == "*") {
                offenses.push(format!("pub use に glob（`*`）を含む: {span:?}"));
            }
            if span.iter().any(|t| t == "self") {
                offenses.push(format!("pub use に `self` リーフを含む: {span:?}"));
            }
            i = span_end;
            continue;
        }
        i += 1;
    }
    offenses
}

/// facade src 全体が [`scan_facade_unmodelable_structures`] の違反を
/// 一切含まないことを固定する（イシュー #2064 codex P2 指摘への対応。
/// `facade_source_declares_no_custom_fn_in_any_context`・
/// `workspace_declares_custom_fn_only_on_tape` 等の否定ガードは、いずれも
/// 「コメント・文字列リテラルを除去したソーステキストのトークン走査」
/// という前提の上に成り立つ。この前提を崩す構造（`#[path]` による
/// ファイル分割の隠蔽・`cfg` によるビルド構成依存の公開面・
/// `include!`／`macro_rules!` によるテキスト非静的な展開・raw
/// identifier・glob 再エクスポート・`self` リーフ再エクスポート）が
/// facade src に混入すると、上記ガード群がソースを正しく読めなくなる
/// （旧実装が `#[path]` 付き `pub mod` を「黙って除外」していたのと
/// 同型の盲点。本テストはそれらの構造そのものの混入を未然に禁止する）。
#[test]
fn facade_source_uses_only_modelable_structures() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_facade_unmodelable_structures(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の src/ にソース走査ガードがモデル化できない構造が\
         見つかった: {offending:?}"
    );
}

/// [`scan_facade_unmodelable_structures`] の自己テスト（正例・負例の
/// 合成入力）。
#[test]
fn scan_facade_unmodelable_structures_detects_each_category() {
    let positive_cases: &[&str] = &[
        r#"#[path = "custom_location.rs"] pub mod weird;"#,
        r#"#[cfg(test)] pub use foo::Bar;"#,
        r#"#[cfg_attr(test, allow(dead_code))] pub mod gated;"#,
        r#"include!("generated.rs");"#,
        r#"macro_rules! m { () => {}; }"#,
        r#"extern crate serde;"#,
        r#"fn f() { let x = r#move; }"#,
        r#"pub use foo::bar::*;"#,
        r#"pub use foo::{self, bar};"#,
    ];
    for src in positive_cases {
        let offenses = scan_facade_unmodelable_structures(src);
        assert!(!offenses.is_empty(), "正例が検出されなかった: {src:?}");
    }

    let negative_cases: &[&str] = &[
        // include_str!／include_bytes! は対象外。
        r#"const X: &str = include_str!("x.txt");"#,
        r#"const Y: &[u8] = include_bytes!("y.bin");"#,
        // cfg でも path でもない属性。
        r#"#[allow(dead_code)] pub mod ok {}"#,
        // 非公開 mod への cfg（`pub` を伴わない）は対象外。
        r#"#[cfg(test)] mod tests {}"#,
        // 通常の pub use（glob・self リーフなし）。
        r#"pub use foo::{Bar, Baz};"#,
        // fn 内の通常マクロ呼び出し（`println!`）は対象外。
        r#"fn f() { println!("x"); }"#,
        // raw 文字列・日本語コメント中の疑似トークンは無視される。
        "// r#custom や include!(\"x\") は日本語コメント中の言及\n\
         let s = r#\"raw include!(\"y\") text\"#;",
    ];
    for src in negative_cases {
        let offenses = scan_facade_unmodelable_structures(src);
        assert!(
            offenses.is_empty(),
            "負例が誤って検出された: {src:?} -> {offenses:?}"
        );
    }
}

/// `path_tokens`（`pub use` の `use` の直後から終端 `;` の手前までの
/// トークン列。例: `foo::{bar, baz as Qux}`）を use tree として展開し、
/// 各終端エントリの**ソース側**の最終パスセグメント（`as` による
/// ローカル別名は無視する。`baz as Qux` の葉は `Qux` ではなく `baz`）を
/// 集めて返す（[`facade_pub_use_leaves_are_not_modules`] 専用）。`{}`
/// ネスト・`as`・`self`・先頭 `::`・末尾カンマを扱う。`*`（glob）は
/// 個別の識別子を持たないため葉として数えない（glob 自体の混入は
/// [`facade_source_uses_only_modelable_structures`] が別途禁止する）。
fn collect_pub_use_leaves(path_tokens: &[String]) -> Vec<String> {
    /// 1 つの use tree ノード（単一パス、または `{ ... }` グループ）を
    /// `tokens[i..]` から解析し、葉を `out` へ積みながら消費後の index
    /// を返す（[`collect_pub_use_leaves`] の下請け）。
    fn parse_tree(tokens: &[String], mut i: usize, out: &mut Vec<String>) -> usize {
        // 先頭の `::`（絶対パス）を読み飛ばす。
        if tokens.get(i).map(String::as_str) == Some(":")
            && tokens.get(i + 1).map(String::as_str) == Some(":")
        {
            i += 2;
        }
        let mut last_segment: Option<String> = None;
        loop {
            match tokens.get(i).map(String::as_str) {
                Some("{") => {
                    // グループ: 直前のパス接頭辞（あれば）は葉ではなく
                    // 単なる修飾子のため捨て、グループ内の各要素を
                    // 再帰的に展開する。
                    i += 1;
                    loop {
                        match tokens.get(i).map(String::as_str) {
                            Some("}") => {
                                i += 1;
                                break;
                            }
                            Some(",") => {
                                i += 1;
                            }
                            None => break,
                            _ => {
                                i = parse_tree(tokens, i, out);
                            }
                        }
                    }
                    return i;
                }
                Some("*") => {
                    // glob: 個別の葉を持たない。
                    return i + 1;
                }
                Some(seg) if seg != "as" && seg != "," && seg != "}" => {
                    last_segment = Some(seg.to_string());
                    i += 1;
                    if tokens.get(i).map(String::as_str) == Some(":")
                        && tokens.get(i + 1).map(String::as_str) == Some(":")
                    {
                        i += 2;
                        continue;
                    }
                }
                _ => {}
            }
            break;
        }
        if tokens.get(i).map(String::as_str) == Some("as") {
            // ローカル別名: ソース側の最終セグメント（`last_segment`）を
            // 葉として採用し、別名自体（`tokens[i + 1]`）は無視する。
            i += 2;
        }
        if let Some(seg) = last_segment {
            out.push(seg);
        }
        i
    }

    let mut i = 0usize;
    let mut out = Vec::new();
    while i < path_tokens.len() {
        match path_tokens.get(i).map(String::as_str) {
            Some(",") => {
                i += 1;
            }
            None => break,
            _ => {
                i = parse_tree(path_tokens, i, &mut out);
            }
        }
    }
    out
}

/// [`collect_pub_use_leaves`] の合成入力テスト。
#[test]
fn collect_pub_use_leaves_expands_nested_groups_and_source_side_renames() {
    let cases: &[(&str, &[&str])] = &[
        ("a::{b::{c, D}, e as F, g::*}", &["c", "D", "e"]),
        ("::fandhe_ai_autodiff as ad", &["fandhe_ai_autodiff"]),
        ("crate::hidden::ext as Ext", &["ext"]),
        ("foo::{Bar, Baz}", &["Bar", "Baz"]),
    ];
    for (src, expected) in cases {
        let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let leaves = collect_pub_use_leaves(&tokens);
        assert_eq!(leaves, expected.to_vec(), "src={src:?} leaves={leaves:?}");
    }
}

/// [`facade_pub_use_leaves_are_not_modules`] が要求する、facade src で
/// 承認済みの小文字始まりの `pub use` 葉（関数の再エクスポート）の
/// allowlist。facade は型を UpperCamelCase・関数を snake_case で公開する
/// 既存の命名規約に従うため、小文字始まりの葉は「関数の再エクスポート」
/// である契約とする。列挙は 2026-09 時点の実際の facade src を走査した
/// 結果を正とする（`crates/facade/src/data.rs`・`interop/safetensors.rs`・
/// `optim.rs`・`compat/mod.rs` の各 `pub use`）。
const LOWERCASE_PUB_USE_LEAF_ALLOWLIST: &[&str] = &[
    // `compat/mod.rs`（`compat::array`。イシュー #411）。
    "array",
    // `interop/safetensors.rs`（イシュー #2019）。
    "load_safetensors_f32",
    "load_safetensors_f32_from_bytes",
    "require_keys",
    "save_safetensors_f32",
    "save_safetensors_f32_to_bytes",
    // `optim.rs`（イシュー #961 ほか。grad clipping／AMP 関数群）。
    "clip_grad_norm",
    "clip_grad_value",
    "global_grad_norm",
    "has_non_finite",
    "scale_grads",
    "scale_loss",
    "unscale_grads",
];

/// facade src の全 `pub use` 文（`pub(..) use` はスコープ付き可視性の
/// ため対象外。`pub` トークン直後が `use` のもののみ対象）を走査し、
/// 各文の葉（[`collect_pub_use_leaves`]。ソース側・rename 前）のうち
/// 小文字始まりのものが [`LOWERCASE_PUB_USE_LEAF_ALLOWLIST`] と完全に
/// 一致することを固定する（イシュー #2064 codex P2 指摘への対応の一環。
/// 小文字始まりの葉で allowlist にないものは、`pub use fandhe_ai_
/// autodiff::nn as ad_nn;` のような「モジュールを別名で再エクスポート
/// する」迂回経路の兆候として fail-closed に拒否する。`pub(crate) use`
/// は `tokens[i]=="pub" && tokens[i+1]=="use"` の完全一致でしか反応
/// しないため対象外——`pub(crate) use hidden::ext;` は `tokens[i+1]`
/// が `(` になり葉として集計されない）。**#2133 の保留（`docs/facade-nn-
/// module-exposure-decision.md` §12）の補完層も担う**: `Module`・
/// `ModuleList` は大文字始まりのため本テストの検出対象外（`facade_does_
/// not_reexport_nn_module_or_containers` が #2133 本来の検出を担う）。
/// 本テストは小文字葉 `nn` の別名モジュール再エクスポート（`pub use
/// fandhe_ai_autodiff::nn as ad_nn;`。`Module` 系に限らない一般形）を
/// 引き続き捕捉する。
#[test]
fn facade_pub_use_leaves_are_not_modules() {
    let src_dir = facade_crate_root().join("src");
    let mut unexpected: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    visit_rs_files(&src_dir, &mut |_path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let mut i = 0usize;
        while i < tokens.len() {
            if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
                let mut end = i + 2;
                while end < tokens.len() && tokens[end] != ";" {
                    end += 1;
                }
                let leaves = collect_pub_use_leaves(&tokens[i + 2..end.min(tokens.len())]);
                for leaf in leaves {
                    if leaf.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                        && !LOWERCASE_PUB_USE_LEAF_ALLOWLIST.contains(&leaf.as_str())
                    {
                        unexpected.insert(leaf);
                    }
                }
                i = (end + 1).min(tokens.len());
                continue;
            }
            i += 1;
        }
    });
    assert!(
        unexpected.is_empty(),
        "facade の pub use に allowlist 外の小文字葉が見つかった\
         （モジュールの誤再エクスポートの可能性がある）: {unexpected:?}"
    );
}

/// [`facade_pub_use_leaves_are_not_modules`] の検出ロジック自体の
/// 自己テスト（合成入力）。
#[test]
fn facade_pub_use_leaves_are_not_modules_detects_unapproved_lowercase_leaf() {
    fn unexpected_lowercase_leaves(content: &str) -> std::collections::BTreeSet<String> {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let mut unexpected = std::collections::BTreeSet::new();
        let mut i = 0usize;
        while i < tokens.len() {
            if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
                let mut end = i + 2;
                while end < tokens.len() && tokens[end] != ";" {
                    end += 1;
                }
                let leaves = collect_pub_use_leaves(&tokens[i + 2..end.min(tokens.len())]);
                for leaf in leaves {
                    if leaf.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                        && !LOWERCASE_PUB_USE_LEAF_ALLOWLIST.contains(&leaf.as_str())
                    {
                        unexpected.insert(leaf);
                    }
                }
                i = (end + 1).min(tokens.len());
                continue;
            }
            i += 1;
        }
        unexpected
    }

    // 正例: モジュールを別名で再エクスポートする迂回経路。
    let leaked = unexpected_lowercase_leaves("pub use fandhe_ai_autodiff::nn as ad_nn;");
    assert!(
        leaked.contains("nn"),
        "モジュールの別名再エクスポートが検出されなかった: {leaked:?}"
    );

    // 負例: `pub(crate) use` は葉として集計されない。
    let scoped = unexpected_lowercase_leaves("pub(crate) use hidden::ext;");
    assert!(
        scoped.is_empty(),
        "pub(crate) use が誤って葉として集計された: {scoped:?}"
    );

    // 負例: allowlist 内の既知の関数再エクスポート。
    let approved = unexpected_lowercase_leaves("pub use array::{ArrayData, array};");
    assert!(
        approved.is_empty(),
        "allowlist 内の葉が誤って違反として検出された: {approved:?}"
    );
}

/// `line` 中に `ident` が識別子単位（前後が `is_ident_char` でない
/// 位置）で現れるかを検査する（イシュー #2134「否定ガード」節。
/// `contains` の部分一致だと `summary` のような一般語が別識別子の
/// 部分文字列として誤検出しうるため、トークン境界で判定する）。
fn line_contains_identifier(line: &str, ident: &str) -> bool {
    let chars: Vec<char> = line.chars().collect();
    let ident_chars: Vec<char> = ident.chars().collect();
    let ident_len = ident_chars.len();
    if ident_len == 0 || chars.len() < ident_len {
        return false;
    }
    for start in 0..=(chars.len() - ident_len) {
        if chars[start..start + ident_len] != ident_chars[..] {
            continue;
        }
        let before_ok = start == 0 || !is_ident_char(chars[start - 1]);
        let after_idx = start + ident_len;
        let after_ok = after_idx >= chars.len() || !is_ident_char(chars[after_idx]);
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// `crates/facade/src/**` の `pub use` 行に `ModuleDict`／`summary`
/// （`fandhe_ai_autodiff::nn::container` に イシュー #2134 で追加した
/// 内部クレート限定の新規公開面）が識別子単位で現れないことを固定
/// する（`facade_does_not_reexport_create_graph_result` と同型の否定
/// ガード）。
///
/// # 背景（実装計画 §2.1）
///
/// イシュー #2134・親 #2131 とも承認コメントが確認できないうえ、
/// facade は `Module` trait 自体を公開していないため
/// `Box<dyn Module>` を受ける `ModuleDict`・`&dyn Module` を受ける
/// `summary` は #2133（`Module` trait の facade 公開）完了まで facade
/// からは意味を成さない。本 PR では `crates/facade/src/**` を変更
/// しないため、本テストは「未公開」という現状を fail-closed に固定
/// するもの。承認取得後の実施形（#2133 完了後の再エクスポート等）を
/// 追加する際は本テストを更新すること。
#[test]
fn facade_does_not_reexport_module_dict_or_summary() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            for ident in ["ModuleDict", "summary"] {
                if line_contains_identifier(trimmed, ident) {
                    offending.push(format!(
                        "{}: `{trimmed}` が `{ident}` を識別子単位で含む",
                        path.display()
                    ));
                }
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が ModuleDict／summary（イシュー #2134 の内部クレート限定\
         新規公開面）を再エクスポートしている（承認未取得のまま対象外という\
         設計判断に違反）: {offending:?}"
    );
}

/// `crates/facade/src/compat/sequential.rs` に
/// `parameter_count`／`summary`／`named_modules` の `pub fn` が存在
/// しないことを固定する（イシュー #2134 実装計画 §2.1「代替として
/// facade 側には『未公開』状態を固定する否定ガードを追加する」）。
/// 承認取得後にこれらの薄い委譲 `pub fn` を追加する際は、本テストを
/// 正ガード（存在することを検査するテスト）へ更新すること。
#[test]
fn compat_sequential_has_no_introspection_methods() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    for name in ["parameter_count", "summary", "named_modules"] {
        assert!(
            !contains_pub_fn_declaration(&content, name),
            "crates/facade/src/compat/sequential.rs に `pub fn {name}(...)` \
             が見つかった（イシュー #2134 は facade 公開面拡張を未承認のまま\
             対象外としている設計判断に違反）"
        );
    }
}

/// KV キャッシュ付き attention（イシュー #2084・親 #2059。設計正本
/// `docs/kv-cache-design.md` §6 承認事項 2）の facade 公開（K-2。
/// `add_stateful_attention`・`StatefulAttention` 相当の 2 `pub fn`）は
/// 未承認のため保留する。`facade_does_not_reexport_module_dict_or_summary`
/// と同型の否定ガード: facade の src/ に①`fn add_stateful_attention`
/// 宣言（可視性・宣言文脈を問わず。[`declares_fn_named`] 参照）、②
/// `KvCache`／`StatefulAttention` を識別子単位で含む `pub use` 行、の
/// いずれも存在しないことを固定する。承認取得後に薄い委譲 `pub fn`／
/// 再エクスポートを追加する際は本テストを正ガードへ更新すること。
#[test]
fn facade_does_not_expose_kv_cache_stateful_attention() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        if declares_fn_named(content, "add_stateful_attention") {
            offending.push(format!(
                "{}: `fn add_stateful_attention` 宣言",
                path.display()
            ));
        }
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            for ident in ["KvCache", "StatefulAttention"] {
                if line_contains_identifier(trimmed, ident) {
                    offending.push(format!(
                        "{}: `{trimmed}` が `{ident}` を識別子単位で含む",
                        path.display()
                    ));
                }
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が KV キャッシュ（#2084 の K-2。`add_stateful_attention`／\
         `KvCache`／`StatefulAttention`）を公開している（`docs/kv-cache-design.md` \
         §6 承認事項 2 が未取得のまま対象外としている設計判断に違反）: {offending:?}"
    );
}

/// `fandhe_ai::Var`（`Var<'t>`・借用 `&Var<'t>`）が算術演算子トレイト
/// （`Add`／`Sub`／`Mul`／`Div`／各 `*Assign`／`Neg`）を実装していない
/// ことを固定する（イシュー #2136。実装計画 §3.2）。**`Var op Var`／
/// `Var op &Var`／`Var op f32`／`Var op f64` に加え、逆向き（スカラー
/// 左辺）の `f32 op Var`／`f64 op Var`（PR #2247 codex-review 指摘 1）
/// と、スカラー右辺・借用レシーバの複合代入 `Var: *Assign<f32/f64>`・
/// `&Var: *Assign<...>`（同指摘 2）も同じ否定ガード方式で固定する**。
///
/// # 背景
///
/// イシュー #2136（`Var` 演算子オーバーロード実装）・設計元の #2135・
/// 親 #2131・設計 PR #2237 のいずれにも、リポジトリ所有者による明示
/// 承認コメントが確認できない（bot（`github-actions`／codex）の自動
/// レビューのみ）。`docs/compat-api-scope.md` §5 は「§5 経路 2 の承認が
/// 得られるまで #2136（実装）は着手不可」と定めており、`Var` に
/// トレイト impl を追加すると `crates/facade/src/lib.rs:184` の
/// `pub use fandhe_ai_autodiff::{..., Var, ...}` を通じて facade の
/// 公開面が自動的に拡大するため（「autodiff にだけ入れて facade には
/// 出さない」は構造上できない）、本 PR では `crates/autodiff/src/**`・
/// `crates/facade/src/**` を変更せず、本テストで現状（未実装）を
/// fail-closed に固定する。
///
/// # ガード方式（#2133 の `NnModuleHoldDoctestGuard` とは別方式）
///
/// #2133 のガードはソースの `pub use` 行に現れる「名前」の衝突を
/// 検出する方式だが、トレイト実装は再エクスポートのような新しい
/// 識別子を導入しないため同じ方式では検出できない。代わりに
/// `static_assertions::assert_not_impl_any!` と同じ原理の曖昧性
/// トリック（依存追加を避け手書き）を用いる: 対象の型パラメータ
/// （`()`）へブランケット実装した `AmbiguousIfImpl<()>` に加え、
/// 「対象トレイトを実装している場合に限り」別の型パラメータ
/// （`Invalid`）へも実装されるブランケット実装を用意すると、対象の
/// トレイトが実際に実装されている場合にのみ `<$ty as
/// AmbiguousIfImpl<_>>::probe` の型パラメータ推論が曖昧になり
/// （E0283 系）、このテストバイナリのコンパイル自体が失敗する
/// （fail-closed）。文字列走査の heuristics（`.claude/rules/
/// out-of-scope-tracking.md` 系の過去指摘往復を招いた方式）は使わない。
///
/// # 既知の限界
///
/// 判定は rustc のコンパイル時トレイト解決に依るため、無効な `cfg`
/// （例 `target_os = "macos"` 限定）の下に置かれた演算子 impl は
/// Linux CI では検出できない。コア型の算術演算子 impl を OS 限定に
/// する正当な理由はないため許容する。
///
/// # 承認取得後の撤去方針
///
/// #2136 の承認取得後、演算子オーバーロードを実装する際は、実装対象
/// の型・トレイトの組に対応する `assert_not_impl!` 行を本テストから
/// 削除すること（`crates/autodiff/tests/operator_overload.rs`
/// （新規）の bit 一致テストへ置き換える）。
#[test]
fn var_does_not_implement_arithmetic_operator_traits_while_2136_on_hold() {
    // このテスト関数のスコープに閉じたヘルパー。他テストの名前空間を
    // 汚さないための private な trait・macro（モジュールを分けない
    // ことで `use` の追加が不要になる）。
    trait AmbiguousIfImpl<A> {
        fn probe() {}
    }
    impl<T: ?Sized> AmbiguousIfImpl<()> for T {}

    macro_rules! assert_not_impl {
        ($ty:ty: $($tr:path),+ $(,)?) => {{
            $({
                struct Invalid;
                impl<T: ?Sized + $tr> AmbiguousIfImpl<Invalid> for T {}
            })+
            // `_` の推論が一意に定まらない（ブランケット実装が複数
            // 候補になる）場合、対象トレイトが実装されていることを
            // 意味し、コンパイルエラーで検出する。
            let _ = <$ty as AmbiguousIfImpl<_>>::probe;
        }};
    }

    use fandhe_ai::Var;
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    assert_not_impl!(
        Var<'static>:
        Add<Var<'static>>, Add<&'static Var<'static>>, Add<f32>, Add<f64>,
    );
    assert_not_impl!(
        &'static Var<'static>:
        Add<Var<'static>>, Add<&'static Var<'static>>, Add<f32>, Add<f64>,
    );

    assert_not_impl!(
        Var<'static>:
        Sub<Var<'static>>, Sub<&'static Var<'static>>, Sub<f32>, Sub<f64>,
    );
    assert_not_impl!(
        &'static Var<'static>:
        Sub<Var<'static>>, Sub<&'static Var<'static>>, Sub<f32>, Sub<f64>,
    );

    assert_not_impl!(
        Var<'static>:
        Mul<Var<'static>>, Mul<&'static Var<'static>>, Mul<f32>, Mul<f64>,
    );
    assert_not_impl!(
        &'static Var<'static>:
        Mul<Var<'static>>, Mul<&'static Var<'static>>, Mul<f32>, Mul<f64>,
    );

    assert_not_impl!(
        Var<'static>:
        Div<Var<'static>>, Div<&'static Var<'static>>, Div<f32>, Div<f64>,
    );
    assert_not_impl!(
        &'static Var<'static>:
        Div<Var<'static>>, Div<&'static Var<'static>>, Div<f32>, Div<f64>,
    );

    assert_not_impl!(Var<'static>: Neg);
    assert_not_impl!(&'static Var<'static>: Neg);

    assert_not_impl!(
        Var<'static>:
        AddAssign<Var<'static>>, AddAssign<&'static Var<'static>>,
    );
    assert_not_impl!(
        Var<'static>:
        SubAssign<Var<'static>>, SubAssign<&'static Var<'static>>,
    );
    assert_not_impl!(
        Var<'static>:
        MulAssign<Var<'static>>, MulAssign<&'static Var<'static>>,
    );
    assert_not_impl!(
        Var<'static>:
        DivAssign<Var<'static>>, DivAssign<&'static Var<'static>>,
    );

    // 逆向き（スカラー左辺）の演算子実装（`f32 + Var` 等）。codex-review
    // 指摘（PR #2247 スレッド 1 件目）: `Var op f32` のみでは `f32 op Var`
    // の追加を検出できないため、`f32`／`f64` を対象型とした
    // `Add`/`Sub`/`Mul`/`Div<Var<'static>>`／`<&'static Var<'static>>`
    // も同じ否定ガード方式で固定する。
    //
    // # `AmbiguousIfImpl` を使い回さない理由
    //
    // 上記の `Var<'static>: Add<f32>, Add<f64>, ...` の検査は
    // `impl<T: ?Sized + Add<f32>> AmbiguousIfImpl<Invalid> for T {}` の
    // ような無条件ブランケット実装を生成する。この `impl` はブロック内
    // 宣言でもコンパイル単位全体（このクレート全体）でトレイト解決に
    // 参加するため、`f32: Add<f32>`（プリミティブの自明な反射的実装）
    // にも該当してしまい、以後 `<f32 as AmbiguousIfImpl<_>>::probe` を
    // 呼ぶと無関係な既存ブロックの `Invalid` と `()` の 2 候補が生じて
    // 常に E0283（曖昧）になる（実測確認済み: 本節をこのまま
    // `AmbiguousIfImpl` へ追加すると `f32` プローブ時点で fail-closed
    // ではなく偽陽性のコンパイルエラーになる）。プリミティブ型を `$ty`
    // に取る本節専用に、別トレイト `AmbiguousIfImplRev`／別マクロ
    // `assert_not_impl_rev!` を用意して汚染を避ける。
    trait AmbiguousIfImplRev<A> {
        fn probe() {}
    }
    impl<T: ?Sized> AmbiguousIfImplRev<()> for T {}

    macro_rules! assert_not_impl_rev {
        ($ty:ty: $($tr:path),+ $(,)?) => {{
            $({
                struct Invalid;
                impl<T: ?Sized + $tr> AmbiguousIfImplRev<Invalid> for T {}
            })+
            let _ = <$ty as AmbiguousIfImplRev<_>>::probe;
        }};
    }

    assert_not_impl_rev!(
        f32:
        Add<Var<'static>>, Add<&'static Var<'static>>,
        Sub<Var<'static>>, Sub<&'static Var<'static>>,
        Mul<Var<'static>>, Mul<&'static Var<'static>>,
        Div<Var<'static>>, Div<&'static Var<'static>>,
    );
    assert_not_impl_rev!(
        f64:
        Add<Var<'static>>, Add<&'static Var<'static>>,
        Sub<Var<'static>>, Sub<&'static Var<'static>>,
        Mul<Var<'static>>, Mul<&'static Var<'static>>,
        Div<Var<'static>>, Div<&'static Var<'static>>,
    );

    // スカラー右辺の複合代入（`Var: *Assign<f32/f64>`）・借用レシーバ
    // （`&Var: *Assign<...>`）。codex-review 指摘（PR #2247 スレッド 2
    // 件目）: 上記は `Var op= Var/&Var` のみを検査しており、スカラー
    // 右辺（`v += 1.0f32` 等）と借用レシーバ経由の複合代入実装を検査
    // していなかった。`&'static Var<'static>` 側は `*Assign` が通常
    // `&mut self` を要求するため実装され得ないが、将来の変則的な実装
    // （例: 内部可変性を用いた impl）も多層防御として同じ方式で固定する。
    assert_not_impl!(
        Var<'static>:
        AddAssign<f32>, AddAssign<f64>,
        SubAssign<f32>, SubAssign<f64>,
        MulAssign<f32>, MulAssign<f64>,
        DivAssign<f32>, DivAssign<f64>,
    );
    assert_not_impl!(
        &'static Var<'static>:
        AddAssign<Var<'static>>, AddAssign<&'static Var<'static>>,
        AddAssign<f32>, AddAssign<f64>,
        SubAssign<Var<'static>>, SubAssign<&'static Var<'static>>,
        SubAssign<f32>, SubAssign<f64>,
        MulAssign<Var<'static>>, MulAssign<&'static Var<'static>>,
        MulAssign<f32>, MulAssign<f64>,
        DivAssign<Var<'static>>, DivAssign<&'static Var<'static>>,
        DivAssign<f32>, DivAssign<f64>,
    );
}

// =====================================================================
// #2141（親 #2131）の facade 公開保留固定（`VarBoolOpsHoldDoctestGuard`）。
// `VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard` 系と同型の
// 正のプローブ 1 ブロック方式のドリフト検査。承認事項・多層防御の
// 位置づけは `docs/autodiff-bool-ops-exposure-decision.md` §6 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `VarBoolOpsHoldDoctestGuard` doc 内の
/// 唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （[`custom_function_hold_doctest_globs_all_pub_modules`] の
/// `VarBoolOpsHoldDoctestGuard` 版）。
#[test]
fn bool_ops_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarBoolOpsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "VarBoolOpsHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`bool_ops_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの **glob
/// 以外の本文**（ローカル自由関数群・`__FandheBoolHoldProbe` トレイト
/// 定義・`Var`／`Tensor<bool>`／`Tensor<f32>`／`Tape` への実装・
/// `__probe_*` 関数群）が固定文言 [`BOOL_OPS_HOLD_PROBE_BODY`] と 1 行
/// たりとも違わず一致することを固定する（`custom_function_hold_
/// doctest_probe_body_matches_fixed_contract` と同じ理由: rustdoc の
/// `# ` 隠し行・プローブの削除・別名へのシャドーイング等で正のプローブ
/// を骨抜きにする改変を機械的に拒否する）。
#[test]
fn bool_ops_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarBoolOpsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, BOOL_OPS_HOLD_PROBE_BODY,
        "VarBoolOpsHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 BOOL_OPS_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_bool_hold_probe モジュール・__FandheBoolHoldProbe\
         トレイト・__probe_* 関数）の削除・弱体化・隠し行の混入がないか\
         確認すること。"
    );
}

/// [`bool_ops_hold_doctest_probe_body_matches_fixed_contract`] が要求
/// する固定文言。`crates/facade/src/lib.rs` の `VarBoolOpsHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致
/// する必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に
/// 含む）。
const BOOL_OPS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_bool_hold_probe {\n\
\x20\x20\x20\x20pub mod bool_ops {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn gt_bool() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn ge_bool() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn lt_bool() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn le_bool() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn eq_bool() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn ne_bool() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn logical_and() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn logical_or() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn logical_not() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn masked_select() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_bool_hold_probe::*;\n\
\n\
struct __FandheBoolMarker;\n\
\n\
trait __FandheBoolHoldProbe {\n\
\x20\x20\x20\x20fn gt_bool(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn ge_bool(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn lt_bool(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn le_bool(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn eq_bool(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn ne_bool(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn logical_and(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn logical_or(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn logical_not(&self) -> __FandheBoolMarker;\n\
\x20\x20\x20\x20fn masked_select(&self) -> __FandheBoolMarker;\n\
}\n\
\n\
impl<'t> __FandheBoolHoldProbe for fandhe_ai::Var<'t> {\n\
\x20\x20\x20\x20fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
}\n\
\n\
impl __FandheBoolHoldProbe for fandhe_ai::Tensor<bool> {\n\
\x20\x20\x20\x20fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
}\n\
\n\
impl __FandheBoolHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
}\n\
\n\
impl __FandheBoolHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn gt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ge_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn lt_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn le_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn eq_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn ne_bool(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_and(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_or(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn logical_not(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
\x20\x20\x20\x20fn masked_select(&self) -> __FandheBoolMarker { __FandheBoolMarker }\n\
}\n\
\n\
fn __probe_free_fns() {\n\
\x20\x20\x20\x20// `bool_ops::` を経由した経路解決（`use fandhe_ai::*;` が\n\
\x20\x20\x20\x20// 同名モジュールを glob 公開していれば、名前解決自体が\n\
\x20\x20\x20\x20// 曖昧になり E0659 でコンパイル失敗する。バレ識別子の\n\
\x20\x20\x20\x20// 未使用 glob 衝突は rustc が検出しないため、経路として\n\
\x20\x20\x20\x20// 実際に `bool_ops` を解決させる必要がある）。\n\
\x20\x20\x20\x20bool_ops::gt_bool();\n\
\x20\x20\x20\x20bool_ops::ge_bool();\n\
\x20\x20\x20\x20bool_ops::lt_bool();\n\
\x20\x20\x20\x20bool_ops::le_bool();\n\
\x20\x20\x20\x20bool_ops::eq_bool();\n\
\x20\x20\x20\x20bool_ops::ne_bool();\n\
\x20\x20\x20\x20bool_ops::logical_and();\n\
\x20\x20\x20\x20bool_ops::logical_or();\n\
\x20\x20\x20\x20bool_ops::logical_not();\n\
\x20\x20\x20\x20bool_ops::masked_select();\n\
}\n\
\n\
fn __probe_var(x: &fandhe_ai::Var<'_>) {\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Var::gt_bool(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.gt_bool();\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Var::masked_select(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.masked_select();\n\
}\n\
\n\
fn __probe_tensor_bool(x: &fandhe_ai::Tensor<bool>) {\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Tensor::logical_and(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.logical_and();\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Tensor::logical_not(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.logical_not();\n\
}\n\
\n\
fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Tensor::gt_bool(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.gt_bool();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Tape::gt_bool(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.gt_bool();\n\
}";

/// bool 出力比較 6 種・logical 3 種・`masked_select`（10 個の関数名。
/// イシュー #2141）。[`facade_does_not_reexport_or_declare_bool_ops`]・
/// [`workspace_declares_bool_ops_fn_names_only_in_autodiff_bool_ops`]
/// が共用する。
const BOOL_OPS_FN_NAMES: [&str; 10] = [
    "gt_bool",
    "ge_bool",
    "lt_bool",
    "le_bool",
    "eq_bool",
    "ne_bool",
    "logical_and",
    "logical_or",
    "logical_not",
    "masked_select",
];

/// facade src 全体（`crates/facade/src/**`）に、`bool_ops` を参照する
/// `pub use`（`pub use fandhe_ai_autodiff::bool_ops;` 等のモジュール
/// 再エクスポート・別名含む）も、[`BOOL_OPS_FN_NAMES`]（10 個）の `fn`
/// 宣言（可視性・宣言文脈を問わない。[`count_fn_declarations_by_name`]
/// と同じ検出契約）も存在しないことを固定する（`VarBoolOpsHoldDoctestGuard`
/// の正のプローブと多層防御を成す最内層のソース走査ガード。
/// `facade_does_not_reexport_nn_module_or_containers` 系と同型）。
#[test]
fn facade_does_not_reexport_or_declare_bool_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "bool_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `bool_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in BOOL_OPS_FN_NAMES {
            let count = count_fn_declarations_by_name(&tokens, fn_name);
            if count > 0 {
                offending.push(format!(
                    "{}: `fn {fn_name}` 宣言が {count} 件見つかった",
                    path.display()
                ));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が bool_ops（イシュー #2141 の内部クレート限定\
         新規公開面。facade 公開は承認待ちのため対象外という設計判断に\
         違反）を再エクスポート、または同名の fn を宣言している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`BOOL_OPS_FN_NAMES`]
/// （10 個）の `fn` 宣言が `crates/autodiff/src/bool_ops.rs` の 1 ファイル
/// のみ（各 1 件）に定義されていることを固定する（`workspace_declares_
/// custom_fn_only_on_tape` と同型の workspace 全体インベントリ。facade
/// のソース走査・`VarBoolOpsHoldDoctestGuard` の正のプローブはいずれも
/// 「facade から到達可能か」しか見ないため、facade の外に同名の trait
/// impl が新設され将来 facade が glob できる形で公開してしまう場合に
/// 備え、そもそもの定義元を先に塞ぐ多層防御の最内層とする）。
#[test]
fn workspace_declares_bool_ops_fn_names_only_in_autodiff_bool_ops() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    let Ok(entries) = std::fs::read_dir(&crates_dir) else {
        panic!(
            "workspace crates ディレクトリが読めない: {}",
            crates_dir.display()
        );
    };
    let mut crate_dirs: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(
        !crate_dirs.is_empty(),
        "workspace crates ディレクトリ配下にクレートが 1 件も見つからない\
         （テスト自体が検査対象を見失っている可能性がある）"
    );

    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            for fn_name in BOOL_OPS_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    let rel = path
                        .strip_prefix(&crates_dir)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }

    let expected: std::collections::BTreeMap<String, usize> = BOOL_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/bool_ops.rs::{name}"), 1usize))
        .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の bool_ops 系 `fn` 宣言集合が\
         `crates/autodiff/src/bool_ops.rs`（各 1 件）のみという期待と\
         一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}
