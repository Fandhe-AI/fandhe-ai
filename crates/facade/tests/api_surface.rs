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
//!
//! `hooks_hold_doctest_globs_all_pub_modules`・`hooks_hold_doctest_probe_
//! body_matches_fixed_contract`・`workspace_declares_no_hook_registration_
//! fns`・`autodiff_declares_no_register_hook_fn` の 4 テストは
//! `VarHooksHoldDoctestGuard`（`VarBoolOpsHoldDoctestGuard` 系と同型の
//! 正のプローブ 1 ブロック方式＋workspace 全体のソース走査＋
//! `crates/autodiff/src/` 限定の `register_hook` allowlist 化ガード）で、
//! forward・backward hooks（イシュー #2139。親 #2138・#2131）の facade
//! 公開保留を固定する。#2139 は設計 doc §11 の承認事項 5 項目がそろう
//! まで着手不可という設計判断（`docs/autodiff-forward-backward-hooks-
//! design.md` §13）のため、本体実装（`crates/autodiff/**`）自体を
//! 含まない保留固定 PR である（4 層構成の内訳は同 doc §13.3。
//! `docs/compat-api-scope.md` §5 に同期する）。

use std::path::Path;

mod common;
use common::temp_dir::TempDirGuard;

fn facade_crate_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read_to_string_or_panic(path: &Path) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("test fixture: {} が読めない: {e}", path.display()))
}

/// `dir` 配下の `.rs` を再帰走査するテストユーティリティ。走査失敗
/// （`read_dir` の Err・エントリ列挙中の Err）を黙って握り潰すと、
/// 本テストファイルの各ガード（「本体未実装を fail-closed に固定する」
/// 契約。モジュール冒頭コメント参照）が「該当ファイルが 0 件見つかった」
/// と「走査自体が失敗した」を区別できず、後者を前者と誤認して
/// `found.is_empty()`／`offending.is_empty()` の判定が意図せず成立し
/// てしまう（fail-open 化。codex-review 指摘・PR #2254）。そのため
/// `read_dir` の Err・エントリ列挙中の Err はいずれも `panic!` で
/// 即座に伝播し、判定対象の走査が不完全なまま成立させない。
fn visit_rs_files(dir: &Path, f: &mut impl FnMut(&Path, &str)) {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("test fixture: {} の read_dir に失敗: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| {
            panic!(
                "test fixture: {} 配下のエントリ列挙に失敗: {e}",
                dir.display()
            )
        });
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
        // `pub use <prefix>{A, B, C};` の `{...}` 部分を抽出する。
        //
        // 単一識別子の `pub use <prefix>Name;` 形（`{...}` なし）も許容する。
        // rustfmt は `{X}`（1 要素の group）を波括弧なしへ整形するため、
        // 1 件でも単一識別子行が増えると往復で赤くなるのを避ける（#2172 で
        // 導入。#2502 で L-BFGS は 3 型の波括弧形になったが許容は維持）。
        let rest = &trimmed[prefix.len()..];
        match (rest.find('{'), rest.find('}')) {
            (Some(open), Some(close)) => {
                for ident in rest[open + 1..close].split(',') {
                    let ident = ident.trim();
                    if !ident.is_empty() {
                        found.insert(ident.to_string());
                    }
                }
            }
            (None, None) => {
                let Some(name) = rest.strip_suffix(';').map(str::trim) else {
                    offending_lines.push(trimmed.to_string());
                    continue;
                };
                let is_single_identifier = !name.is_empty()
                    && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    && name.chars().next().is_some_and(|c| !c.is_ascii_digit());
                if !is_single_identifier {
                    offending_lines.push(trimmed.to_string());
                    continue;
                }
                found.insert(name.to_string());
            }
            _ => {
                // `{` のみ／`}` のみ（閉じ忘れ等）は不正形式として拒否する。
                offending_lines.push(trimmed.to_string());
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
        "Adadelta",
        "AdadeltaConfig",
        "Adamax",
        "AdamaxConfig",
        "NAdam",
        "NAdamConfig",
        "RAdam",
        "RAdamConfig",
        "Adagrad",
        "AdagradConfig",
        "Adam",
        "AdamConfig",
        "AdamW",
        "AdamWConfig",
        "Lamb",
        "LambConfig",
        "Lbfgs",
        "LbfgsConfig",
        "LbfgsLineSearch",
        "ClipGradResult",
        "clip_grad_norm",
        "clip_grad_value",
        "global_grad_norm",
        "ConstantLr",
        "CosineAnnealingLr",
        "ExponentialLr",
        "LinearWarmupLr",
        "LrScheduler",
        "CosineAnnealingWarmRestarts",
        "CyclicLr",
        "LambdaLr",
        "MultiStepLr",
        "SequentialLr",
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

    // Adadelta／Adamax／NAdam／RAdam（イシュー #2501。#2171 実装の facade 公開）
    // の facade のみ経由の到達性＋既定値ドリフトガード（値は
    // `nn::optim::{adadelta,adamax,nadam,radam}` の `Default` 実装と一致）。
    let adadelta_config = fandhe_ai::optim::AdadeltaConfig::default();
    assert_eq!(adadelta_config.lr, 1.0, "AdadeltaConfig の既定 lr は 1.0");
    assert_eq!(adadelta_config.rho, 0.9, "AdadeltaConfig の既定 rho は 0.9");
    let mut adadelta = fandhe_ai::optim::Adadelta::new(adadelta_config)
        .unwrap_or_else(|e| panic!("test fixture: Adadelta::new が失敗した: {e}"));
    let _ = &mut adadelta;

    let adamax_config = fandhe_ai::optim::AdamaxConfig::default();
    assert_eq!(adamax_config.lr, 2e-3, "AdamaxConfig の既定 lr は 2e-3");
    let mut adamax = fandhe_ai::optim::Adamax::new(adamax_config)
        .unwrap_or_else(|e| panic!("test fixture: Adamax::new が失敗した: {e}"));
    let _ = &mut adamax;

    let nadam_config = fandhe_ai::optim::NAdamConfig::default();
    assert_eq!(
        nadam_config.momentum_decay, 4e-3,
        "NAdamConfig の既定 momentum_decay は 4e-3"
    );
    let mut nadam = fandhe_ai::optim::NAdam::new(nadam_config)
        .unwrap_or_else(|e| panic!("test fixture: NAdam::new が失敗した: {e}"));
    let _ = &mut nadam;

    let radam_config = fandhe_ai::optim::RAdamConfig::default();
    assert_eq!(radam_config.lr, 1e-3, "RAdamConfig の既定 lr は 1e-3");
    let mut radam = fandhe_ai::optim::RAdam::new(radam_config)
        .unwrap_or_else(|e| panic!("test fixture: RAdam::new が失敗した: {e}"));
    let _ = &mut radam;

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

    // LR スケジューラ拡張 5 種（イシュー #2176 実装・#2503 公開）が facade
    // のみを通じて構築でき、`&dyn LrScheduler` へ coerce できることの固定。
    let multi_step = fandhe_ai::optim::MultiStepLr::new(0.1, &[2, 4], 0.5)
        .unwrap_or_else(|e| panic!("test fixture: MultiStepLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &multi_step;
    assert_eq!(fandhe_ai::optim::LrScheduler::lr_at(&multi_step, 2), 0.05);

    let warm_restarts = fandhe_ai::optim::CosineAnnealingWarmRestarts::new(0.1, 4, 2, 0.0)
        .unwrap_or_else(|e| {
            panic!("test fixture: CosineAnnealingWarmRestarts::new が失敗した: {e}")
        });
    let _: &dyn fandhe_ai::optim::LrScheduler = &warm_restarts;
    assert_eq!(fandhe_ai::optim::LrScheduler::lr_at(&warm_restarts, 0), 0.1);

    let cyclic = fandhe_ai::optim::CyclicLr::new(0.01, 0.1, 2, None)
        .unwrap_or_else(|e| panic!("test fixture: CyclicLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &cyclic;
    assert_eq!(fandhe_ai::optim::LrScheduler::lr_at(&cyclic, 0), 0.01);

    let lambda = fandhe_ai::optim::LambdaLr::new(0.1, |step| 0.5_f64.powi(step as i32))
        .unwrap_or_else(|e| panic!("test fixture: LambdaLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &lambda;
    assert_eq!(fandhe_ai::optim::LrScheduler::lr_at(&lambda, 1), 0.05);

    let first = fandhe_ai::optim::StepLr::new(0.1, 1, 0.5)
        .unwrap_or_else(|e| panic!("test fixture: StepLr::new が失敗した: {e}"));
    let second = fandhe_ai::optim::ConstantLr::new(0.01)
        .unwrap_or_else(|e| panic!("test fixture: ConstantLr::new が失敗した: {e}"));
    let sequential = fandhe_ai::optim::SequentialLr::new(
        vec![
            Box::new(first) as Box<dyn fandhe_ai::optim::LrScheduler>,
            Box::new(second) as Box<dyn fandhe_ai::optim::LrScheduler>,
        ],
        vec![2],
    )
    .unwrap_or_else(|e| panic!("test fixture: SequentialLr::new が失敗した: {e}"));
    let _: &dyn fandhe_ai::optim::LrScheduler = &sequential;
    assert_eq!(fandhe_ai::optim::LrScheduler::lr_at(&sequential, 2), 0.01);
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
/// from_sequential, from_path_with_limits}` と `OnnxExternalDataLimits` の
/// 12 件のみ（イシュー #2017・#2018・#2037・#2360）。
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
    const ALLOWED_PUB_ITEMS: [(&str, &str); 12] = [
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
        ("struct", "OnnxExternalDataLimits"),
        ("fn", "from_path_with_limits"),
    ];
    const SCANNED_KINDS: [&str; 9] = [
        "struct", "enum", "fn", "trait", "type", "const", "static", "mod", "use",
    ];
    const QUALIFIER_KEYWORDS: [&str; 3] = ["async", "unsafe", "extern"];
    // `nn::Module` の本来の目的は内部型 `fandhe_ai_autodiff::nn::Module` の漏洩検出だが、
    // 部分文字列一致のため facade 公開名 `fandhe_ai::nn::Module`（#2395）にも一致する。
    // 走査対象は `src/interop/onnx.rs` 限定で、facade `nn::Module` を受ける ONNX API は
    // 承認範囲外のため、この一致は意図した fail-closed（衝突ではない）。将来承認する場合は
    // 本エントリを同時に見直す（`onnx_forbidden_nn_module_substring_does_not_collide_with_facade_nn_module`）。
    const FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS: [&str; 12] = [
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
        "ExternalDataOptions",
        "external_data::",
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

/// `scan_unapproved_onnx_pub_items` が承認範囲 12 件（`OnnxModel`／
/// `OnnxValue`／`OnnxError`／`OnnxExportOptions`／`OnnxExternalDataLimits` の
/// 型定義 5 件と `OnnxModel::{from_bytes, from_path, from_path_with_limits,
/// run, to_bytes, to_path, from_sequential}` のメソッド 7 件）をすべて含む合成ソースに対して
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

pub struct OnnxExternalDataLimits {
    pub max_total_bytes: u64,
    pub max_external_files: usize,
}

impl OnnxModel {
    pub fn from_path_with_limits(
        path: &str,
        limits: &OnnxExternalDataLimits,
    ) -> Result<Self, OnnxError> {
        unimplemented!()
    }
}
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses.is_empty(),
        "承認範囲 12 件のみの合成ソースでオフェンスが検出された（空虚 pass 防止\
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
        // `nn::Module` は内部型の漏洩検出用だが facade 公開名にも部分一致する。
        // 意図した fail-closed（onnx.rs は facade `nn::Module` を受けない）。
        // 詳細は `scan_unapproved_onnx_pub_items` の同名リスト直上のコメントを参照。
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
            "ExternalDataOptions",
            "external_data::",
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
        "pub struct OnnxExternalDataLimits",
        "pub fn from_path_with_limits",
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

/// external data 予算まわり（イシュー #2360）の承認外追加を検出する負例。
#[test]
fn unapproved_onnx_external_data_items_are_flagged() {
    let synthetic = r#"
impl OnnxModel {
    pub fn from_path_with_limits(path: &str, options: &fandhe_ai_onnx_interop::onnx::external_data::ExternalDataOptions) -> Result<Self, OnnxError> {
        unimplemented!()
    }
}

pub type OnnxExternalDataLimits = ExternalDataOptions;
pub use fandhe_ai_onnx_interop::onnx::external_data::ExternalDataOptions;
"#;
    let offenses = scan_unapproved_onnx_pub_items(synthetic);
    assert!(
        offenses
            .iter()
            .any(|o| o.contains("from_path_with_limits") && o.contains("ExternalDataOptions")),
        "シグネチャ内の内部型が検出されなかった: {offenses:?}"
    );
    assert!(
        offenses.iter().any(|o| o.contains("type")),
        "承認外の `pub type` が検出されなかった: {offenses:?}"
    );
    assert!(
        offenses.iter().any(|o| o.contains("use")),
        "`pub use` 再エクスポートが検出されなかった: {offenses:?}"
    );
}

/// `OnnxExternalDataLimits`／`from_path_with_limits` が facade から到達可能で
/// 既定値・フィールド変更・シグネチャが固定されていること（#2360）。
#[test]
fn onnx_external_data_limits_are_reachable_via_facade() {
    use fandhe_ai::interop::onnx::{OnnxError, OnnxExternalDataLimits, OnnxModel};
    use std::path::Path;

    let mut l = OnnxExternalDataLimits::default();
    assert_eq!(l.max_total_bytes, 64 * 1024 * 1024 * 1024);
    assert_eq!(l.max_external_files, 4096);
    l.max_total_bytes = 1 << 20;
    l.max_external_files = 1;
    assert_eq!(l.max_total_bytes, 1 << 20);
    assert_eq!(l.max_external_files, 1);

    fn _sig(p: &Path, l: &OnnxExternalDataLimits) -> Result<OnnxModel, OnnxError> {
        OnnxModel::from_path_with_limits(p, l)
    }
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
        "CollateFn",
        "DataError",
        "DataLoader",
        "DataLoaderConfig",
        "Dataset",
        "HookedBatches",
        "HookedDataLoader",
        "PrefetchBatches",
        "PrefetchConfig",
        "PrefetchDataLoader",
        "RandomSampler",
        "Sampler",
        "SamplerBatches",
        "SamplerDataLoader",
        "SequentialSampler",
        "TensorDataset",
        "TransformFn",
        "WeightedRandomSampler",
        "default_collate",
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

    // Sampler 系・フック系（#2505。決定記録 §5）も facade のみで到達可能。
    fn assert_is_sampler<S: fandhe_ai::data::Sampler>(_s: &S) {}
    let seq = fandhe_ai::data::SequentialSampler::new(4, 2, false)
        .unwrap_or_else(|e| panic!("test fixture: SequentialSampler::new が失敗: {e}"));
    assert_is_sampler(&seq);
    let rand = fandhe_ai::data::RandomSampler::new(4, 2, false)
        .unwrap_or_else(|e| panic!("test fixture: RandomSampler::new が失敗: {e}"));
    assert_is_sampler(&rand);
    let weighted =
        fandhe_ai::data::WeightedRandomSampler::new(vec![1.0, 2.0, 3.0, 4.0], 4, true, 2, false)
            .unwrap_or_else(|e| panic!("test fixture: WeightedRandomSampler::new が失敗: {e}"));
    assert_is_sampler(&weighted);

    let feats = fandhe_ai::Tensor::<f32>::new(vec![0.0, 1.0, 2.0, 3.0], &[4, 1])
        .unwrap_or_else(|e| panic!("test fixture: features の構築に失敗: {e}"));
    let ds = fandhe_ai::data::TensorDataset::new(feats.clone())
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
    let mut sl = fandhe_ai::data::SamplerDataLoader::new(ds, seq)
        .unwrap_or_else(|e| panic!("test fixture: SamplerDataLoader::new が失敗: {e}"));
    let sb: fandhe_ai::data::SamplerBatches<'_, _> = sl.iter();
    assert_eq!(sb.count(), 2);

    let ds = fandhe_ai::data::TensorDataset::new(feats)
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
    let sampler = fandhe_ai::data::SequentialSampler::new(4, 2, false)
        .unwrap_or_else(|e| panic!("test fixture: SequentialSampler::new が失敗: {e}"));
    let mut hooked = fandhe_ai::data::HookedDataLoader::new(ds, sampler)
        .unwrap_or_else(|e| panic!("test fixture: HookedDataLoader::new が失敗: {e}"))
        .with_try_transform(Ok)
        .with_try_collate(fandhe_ai::data::default_collate);
    let hb: fandhe_ai::data::HookedBatches<'_, f32> = hooked.iter();
    assert_eq!(hb.count(), 2);

    let _t: fandhe_ai::data::TransformFn<f32> = Box::new(Ok);
    let _c: fandhe_ai::data::CollateFn<f32> = Box::new(fandhe_ai::data::default_collate);
    let samples = [
        fandhe_ai::Tensor::<f32>::zeros(&[3]).unwrap_or_else(|e| panic!("fixture: {e}")),
        fandhe_ai::Tensor::<f32>::zeros(&[3]).unwrap_or_else(|e| panic!("fixture: {e}")),
    ];
    let stacked = fandhe_ai::data::default_collate(&samples)
        .unwrap_or_else(|e| panic!("test fixture: default_collate が失敗: {e}"));
    assert_eq!(stacked.shape(), &[2, 3]);

    // マルチワーカー prefetch（#2506。決定記録 §4・§8）も facade のみで到達可能。
    // 逐次経路（num_workers=0）と並列経路（num_workers>=1）の双方を固定する。
    for workers in [0usize, 2] {
        let cfg = fandhe_ai::data::PrefetchConfig::new(workers, 2)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchConfig::new が失敗: {e}"));
        assert_eq!(cfg.num_workers(), workers);
        assert_eq!(cfg.prefetch_depth(), 2);
        let ds = fandhe_ai::data::TensorDataset::new(
            fandhe_ai::Tensor::<f32>::new(vec![0.0, 1.0, 2.0, 3.0], &[4, 1])
                .unwrap_or_else(|e| panic!("test fixture: features の構築に失敗: {e}")),
        )
        .unwrap_or_else(|e| panic!("test fixture: TensorDataset::new が失敗: {e}"));
        let sampler = fandhe_ai::data::SequentialSampler::new(4, 2, false)
            .unwrap_or_else(|e| panic!("test fixture: SequentialSampler::new が失敗: {e}"));
        let mut pl = fandhe_ai::data::PrefetchDataLoader::new(ds, sampler, cfg)
            .unwrap_or_else(|e| panic!("test fixture: PrefetchDataLoader::new が失敗: {e}"));
        assert_eq!(pl.num_batches(), Some(2));
        assert_eq!(pl.config().num_workers(), workers);
        let pb: fandhe_ai::data::PrefetchBatches<'_, _> = pl.iter();
        assert_eq!(pb.count(), 2);
    }
}

// =====================================================================
// データフック（イシュー #2182 → #2505）: `Sampler` 系・`HookedDataLoader`
// 系・`default_collate` 等 11 名は、#2182 では facade 公開保留として
// `DataHooksHoldDoctestGuard` と否定ガードで固定していたが、#2505 で
// `fandhe_ai::data` からの素の再エクスポートを承認形
// （`docs/tensor-core-data-sampler-hooks-decision.md` §5。ルート #2499 の
// ユーザー承認〈依頼文記載の承認日 2026-10-04〉）として公開したため、doctest ガードとその
// ドリフト検査 2 件を削除し、ソース走査ガードを「承認した形だけを
// 許す」正ガードへ反転した。`DataLoader` への統合（fn 宣言）は引き続き
// 禁止する。
// =====================================================================

/// 型名 10 個（イシュー #2182）。[`scan_data_hooks_reexports_and_
/// declarations`]・[`workspace_declares_data_hooks_names_only_in_
/// tensor_core_data`] が共用する。
const DATA_HOOKS_TYPE_NAMES: [&str; 10] = [
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "WeightedRandomSampler",
    "SamplerDataLoader",
    "SamplerBatches",
    "HookedDataLoader",
    "HookedBatches",
    "TransformFn",
    "CollateFn",
];

/// fn 名 5 個（イシュー #2182。`with_sampler` はプローブのみが使う
/// 予防的な識別子で、現行実装には存在しない——将来 facade が
/// `DataLoader::with_sampler` のような名前で迂回するのを事前に塞ぐ）。
/// [`scan_data_hooks_reexports_and_declarations`] が共用する。
const DATA_HOOKS_FN_NAMES: [&str; 5] = [
    "default_collate",
    "with_transform",
    "with_try_transform",
    "with_collate",
    "with_try_collate",
];

/// 承認形で公開する 11 識別子（[`DATA_HOOKS_TYPE_NAMES`] の 10 型と
/// `default_collate`。#2505）。
const DATA_HOOKS_APPROVED_REEXPORT_NAMES: [&str; 11] = [
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "WeightedRandomSampler",
    "SamplerDataLoader",
    "SamplerBatches",
    "HookedDataLoader",
    "HookedBatches",
    "TransformFn",
    "CollateFn",
    "default_collate",
];

/// `pub use` の path トークン列が `fandhe_ai_tensor_core :: data :: …`
/// で始まるか（承認形の接頭辞）を判定する。
fn data_hooks_approved_prefix(path_tokens: &[String]) -> bool {
    let want = ["fandhe_ai_tensor_core", ":", ":", "data", ":", ":"];
    path_tokens.len() >= want.len() && path_tokens.iter().zip(want).all(|(a, b)| a == b)
}

/// [`facade_reexports_data_hooks_items_only_in_approved_shape`]・その
/// 自己テストが共用する検出本体。`content`（1 ファイル分のソース）の
/// `pub use` から [`collect_pub_use_leaves`] で 11 識別子の葉を集め、
/// 承認形（接頭辞が `fandhe_ai_tensor_core::data::` で `as` 別名を伴わない）
/// の出現葉を第 1 要素、承認形から外れる出現（別 path 接頭辞・別名）、
/// 10 型名の `trait`／`struct`／`enum`／`type` 独自宣言、
/// [`DATA_HOOKS_FN_NAMES`] の `fn` 宣言（`DataLoader` への統合や facade
/// 独自 fn の禁止。[`count_fn_declarations_by_name`] と同じ検出契約）を
/// 第 2 要素（違反）として返す。コメント・文字列リテラル中の出現と
/// 非公開 `use` は無視する。
fn scan_data_hooks_reexports_and_declarations(content: &str) -> (Vec<String>, Vec<String>) {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut approved: Vec<String> = Vec::new();
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let hits: Vec<String> = collect_pub_use_leaves(path_tokens)
                .into_iter()
                .filter(|leaf| DATA_HOOKS_APPROVED_REEXPORT_NAMES.contains(&leaf.as_str()))
                .collect();
            let shape_ok =
                data_hooks_approved_prefix(path_tokens) && !path_tokens.iter().any(|t| t == "as");
            for leaf in hits {
                if shape_ok {
                    approved.push(leaf);
                } else {
                    offending.push(format!("承認形外の pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && let Some(name) = tokens.get(i + 1)
            && DATA_HOOKS_TYPE_NAMES.contains(&name.as_str())
        {
            offending.push(format!("{} {name} 宣言", tokens[i]));
        }
        i += 1;
    }

    for fn_name in DATA_HOOKS_FN_NAMES {
        let count = count_fn_declarations_by_name(&tokens, fn_name);
        if count > 0 {
            offending.push(format!("`fn {fn_name}` 宣言が {count} 件"));
        }
    }
    (approved, offending)
}

/// 承認形の正ガード（#2505 で `facade_does_not_reexport_or_declare_
/// data_hooks` から反転）。facade src 全体で、11 識別子がそれぞれ
/// `src/data.rs` の `pub use fandhe_ai_tensor_core::data::…`（別名なし）
/// としてちょうど 1 回だけ出現し、承認形外の再エクスポート・同名の独自
/// 宣言・[`DATA_HOOKS_FN_NAMES`] の `fn` 宣言が存在しないことを
/// fail-closed に固定する。
#[test]
fn facade_reexports_data_hooks_items_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_data_rs: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let (approved, bad) = scan_data_hooks_reexports_and_declarations(content);
        for offense in bad {
            offending.push(format!("{}: {offense}", path.display()));
        }
        if path.ends_with("src/data.rs") {
            approved_in_data_rs.extend(approved);
        } else {
            for leaf in approved {
                offending.push(format!(
                    "{}: data.rs 以外での pub use leaf={leaf}",
                    path.display()
                ));
            }
        }
    });
    approved_in_data_rs.sort();
    let mut expected: Vec<String> = DATA_HOOKS_APPROVED_REEXPORT_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    expected.sort();
    assert!(
        offending.is_empty(),
        "facade の公開面がデータフック（#2505。Sampler 系・HookedDataLoader 系・\
         default_collate）を承認形（src/data.rs の `pub use fandhe_ai_tensor_core::\
         data::…`・別名なし）以外で再エクスポート、独自宣言、または DataLoader への\
         統合用 fn を宣言している（`docs/tensor-core-data-sampler-hooks-decision.md` \
         §5）: {offending:?}"
    );
    assert_eq!(
        approved_in_data_rs, expected,
        "src/data.rs に承認形の 11 識別子がちょうど 1 回ずつ存在しない\
         （過不足・重複いずれも fail。検査対象を見失った場合を含む）"
    );
}

/// [`scan_data_hooks_reexports_and_declarations`] の自己テスト
/// （正例・負例の合成入力）。
#[test]
fn facade_reexports_data_hooks_items_only_in_approved_shape_detects_each_category() {
    let scan = scan_data_hooks_reexports_and_declarations;
    let (ok, bad) = scan("pub use fandhe_ai_tensor_core::data::{Sampler, SequentialSampler};");
    assert!(bad.is_empty());
    assert_eq!(ok, vec!["Sampler", "SequentialSampler"]);
    let (ok, bad) = scan("pub use fandhe_ai_tensor_core::data::default_collate;");
    assert!(bad.is_empty());
    assert_eq!(ok, vec!["default_collate"]);
    // 違反: 別名・別 path 接頭辞・独自宣言・fn 宣言。
    for src in [
        "pub use fandhe_ai_tensor_core::data::HookedBatches as Foo;",
        "pub use fandhe_ai_tensor_core::Sampler;",
        "pub struct HookedBatches;",
        "pub fn with_transform() {}",
        "fn default_collate() {}",
    ] {
        assert!(!scan(src).1.is_empty(), "src={src:?}");
    }
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use ...::Sampler;",
        "let s = \"CollateFn\";",
        "use fandhe_ai_tensor_core::data::RandomSampler;",
        "pub use fandhe_ai_tensor_core::data::Dataset;",
    ] {
        let (ok, bad) = scan(src);
        assert!(ok.is_empty() && bad.is_empty(), "src={src:?}");
    }
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`DATA_HOOKS_TYPE_NAMES`]（10 個。トレイト／struct／type 宣言）と
/// [`DATA_HOOKS_FN_NAMES`]（5 個。`fn` 宣言）の定義元集合を固定する
/// （`workspace_declares_rng_distribution_names_only_in_allowed_
/// locations` と同型のインベントリ）。
///
/// **期待集合**（着手前確認のグレップで型名・fn 名いずれも既存の衝突は
/// 0 件だった。実装後の実測ですべて `crates/tensor-core/src/data.rs`
/// 1 箇所ずつに定義された）。
#[test]
fn workspace_declares_data_hooks_names_only_in_tensor_core_data() {
    let crates_dir = workspace_crates_dir();
    let mut found_types: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut found_fns: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

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
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");

            let mut i = 0usize;
            while i < tokens.len() {
                if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
                    && let Some(name) = tokens.get(i + 1)
                    && DATA_HOOKS_TYPE_NAMES.contains(&name.as_str())
                {
                    *found_types.entry(format!("{rel}::{name}")).or_insert(0) += 1;
                }
                i += 1;
            }

            for fn_name in DATA_HOOKS_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found_fns.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }

    let expected_types: std::collections::BTreeMap<String, usize> = [
        ("tensor-core/src/data.rs::Sampler".to_string(), 1usize),
        (
            "tensor-core/src/data.rs::SequentialSampler".to_string(),
            1usize,
        ),
        ("tensor-core/src/data.rs::RandomSampler".to_string(), 1usize),
        (
            "tensor-core/src/data.rs::WeightedRandomSampler".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::SamplerDataLoader".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::SamplerBatches".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::HookedDataLoader".to_string(),
            1usize,
        ),
        ("tensor-core/src/data.rs::HookedBatches".to_string(), 1usize),
        ("tensor-core/src/data.rs::TransformFn".to_string(), 1usize),
        ("tensor-core/src/data.rs::CollateFn".to_string(), 1usize),
    ]
    .into_iter()
    .collect();

    let expected_fns: std::collections::BTreeMap<String, usize> = [
        (
            "tensor-core/src/data.rs::default_collate".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::with_transform".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::with_try_transform".to_string(),
            1usize,
        ),
        ("tensor-core/src/data.rs::with_collate".to_string(), 1usize),
        (
            "tensor-core/src/data.rs::with_try_collate".to_string(),
            1usize,
        ),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        found_types, expected_types,
        "workspace 全体（crates/*/src/）の Sampler 系型宣言集合が期待\
         （tensor-core/src/data.rs に各 1 件）と一致しない（過不足いずれも\
         fail-closed に検出する）: {found_types:?}"
    );
    assert_eq!(
        found_fns, expected_fns,
        "workspace 全体（crates/*/src/）の with_transform 等 fn 宣言集合が\
         期待（tensor-core/src/data.rs に各 1 件。with_sampler は現行\
         実装に存在しないため期待集合に含めない）と一致しない\
         （過不足いずれも fail-closed に検出する。新たな定義元が見つかった\
         場合、それが承認済みの実装なのか迂回経路の混入なのかを確認する\
         こと）: {found_fns:?}"
    );
}

/// マルチワーカー prefetch（イシュー #2183）の公開型 3 個。#2183 では
/// facade 公開を保留して否定ガードで固定していたが、#2506（ルート #2499
/// の承認。Issue 記載の承認日 2026-10-04。公開形は
/// `docs/tensor-core-data-prefetch-decision.md` §4・§8）で承認形
/// （`src/data.rs` の `pub use fandhe_ai_tensor_core::data::…`・別名なし）
/// の正ガードへ反転した。fn 名インベントリは持たない（`new`／`iter`／
/// `config` 等の汎用名は検出力がないため。決定記録 §4・§5）。
const PREFETCH_TYPE_NAMES: [&str; 3] = ["PrefetchConfig", "PrefetchDataLoader", "PrefetchBatches"];

/// [`facade_reexports_prefetch_items_only_in_approved_shape`]・その
/// 自己テストが共用する検出本体（[`scan_data_hooks_reexports_and_
/// declarations`] と同型。fn 名インベントリを持たない点のみ異なる）。
/// 承認形の出現葉を第 1 要素、承認形外の再エクスポート（別 path 接頭辞・
/// 別名）と独自宣言を第 2 要素（違反）として返す。
fn scan_prefetch_reexports_and_declarations(content: &str) -> (Vec<String>, Vec<String>) {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut approved: Vec<String> = Vec::new();
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let hits: Vec<String> = collect_pub_use_leaves(path_tokens)
                .into_iter()
                .filter(|leaf| PREFETCH_TYPE_NAMES.contains(&leaf.as_str()))
                .collect();
            let shape_ok =
                data_hooks_approved_prefix(path_tokens) && !path_tokens.iter().any(|t| t == "as");
            for leaf in hits {
                if shape_ok {
                    approved.push(leaf);
                } else {
                    offending.push(format!("承認形外の pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && let Some(name) = tokens.get(i + 1)
            && PREFETCH_TYPE_NAMES.contains(&name.as_str())
        {
            offending.push(format!("{} {name} 宣言", tokens[i]));
        }
        i += 1;
    }
    (approved, offending)
}

/// 承認形の正ガード（#2506 で `facade_does_not_reexport_or_declare_
/// prefetch` から反転）。facade src 全体で、3 識別子がそれぞれ
/// `src/data.rs` の `pub use fandhe_ai_tensor_core::data::…`（別名なし）
/// としてちょうど 1 回だけ出現し、承認形外の再エクスポート・同名の独自
/// 宣言が存在しないことを fail-closed に固定する。
#[test]
fn facade_reexports_prefetch_items_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_data_rs: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let (approved, bad) = scan_prefetch_reexports_and_declarations(content);
        for offense in bad {
            offending.push(format!("{}: {offense}", path.display()));
        }
        if path.ends_with("src/data.rs") {
            approved_in_data_rs.extend(approved);
        } else {
            for leaf in approved {
                offending.push(format!(
                    "{}: data.rs 以外での pub use leaf={leaf}",
                    path.display()
                ));
            }
        }
    });
    approved_in_data_rs.sort();
    let mut expected: Vec<String> = PREFETCH_TYPE_NAMES.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert!(
        offending.is_empty(),
        "facade の公開面が DataLoader マルチワーカー prefetch（#2506。\
         PrefetchConfig・PrefetchDataLoader・PrefetchBatches）を承認形（src/data.rs の \
         `pub use fandhe_ai_tensor_core::data::…`・別名なし）以外で再エクスポートまたは\
         独自宣言している（`docs/tensor-core-data-prefetch-decision.md` §4・§8）: \
         {offending:?}"
    );
    assert_eq!(
        approved_in_data_rs, expected,
        "src/data.rs に承認形の 3 識別子がちょうど 1 回ずつ存在しない\
         （過不足・重複いずれも fail。検査対象を見失った場合を含む）"
    );
}

/// [`scan_prefetch_reexports_and_declarations`] の自己テスト
/// （正例・負例の合成入力）。
#[test]
fn facade_reexports_prefetch_items_only_in_approved_shape_detects_each_category() {
    let scan = scan_prefetch_reexports_and_declarations;
    let (ok, bad) = scan(
        "pub use fandhe_ai_tensor_core::data::{PrefetchBatches, PrefetchConfig, PrefetchDataLoader};",
    );
    assert!(bad.is_empty());
    assert_eq!(
        ok,
        vec!["PrefetchBatches", "PrefetchConfig", "PrefetchDataLoader"]
    );
    let (ok, bad) = scan("pub use fandhe_ai_tensor_core::data::PrefetchConfig;");
    assert!(bad.is_empty());
    assert_eq!(ok, vec!["PrefetchConfig"]);
    // 違反: 別名・別 path 接頭辞・独自宣言。
    for src in [
        "pub use fandhe_ai_tensor_core::data::PrefetchConfig as Foo;",
        "pub use fandhe_ai_tensor_core::PrefetchConfig;",
        "pub struct PrefetchBatches;",
        "type PrefetchConfig = u8;",
    ] {
        assert!(!scan(src).1.is_empty(), "src={src:?}");
    }
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use ...::PrefetchConfig;",
        "let s = \"PrefetchDataLoader\";",
        "use fandhe_ai_tensor_core::data::PrefetchBatches;",
        "pub use fandhe_ai_tensor_core::data::Dataset;",
    ] {
        let (ok, bad) = scan(src);
        assert!(ok.is_empty() && bad.is_empty(), "src={src:?}");
    }
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`PREFETCH_TYPE_
/// NAMES`]（3 個。トレイト／struct／type 宣言）の定義元集合を固定する
/// （`workspace_declares_data_hooks_names_only_in_tensor_core_data` と
/// 同型のインベントリ）。
///
/// **期待集合**（再エクスポートは宣言ではないため #2506 の反転後も不変。着手前確認のグレップで型名の衝突は 0 件だった。
/// 実装後の実測ですべて `crates/tensor-core/src/data.rs` 1 箇所ずつに
/// 定義された）。
#[test]
fn workspace_declares_prefetch_names_only_in_tensor_core_data() {
    let crates_dir = workspace_crates_dir();
    let mut found_types: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

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
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");

            let mut i = 0usize;
            while i < tokens.len() {
                if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
                    && let Some(name) = tokens.get(i + 1)
                    && PREFETCH_TYPE_NAMES.contains(&name.as_str())
                {
                    *found_types.entry(format!("{rel}::{name}")).or_insert(0) += 1;
                }
                i += 1;
            }
        });
    }

    let expected_types: std::collections::BTreeMap<String, usize> = [
        (
            "tensor-core/src/data.rs::PrefetchConfig".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::PrefetchDataLoader".to_string(),
            1usize,
        ),
        (
            "tensor-core/src/data.rs::PrefetchBatches".to_string(),
            1usize,
        ),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        found_types, expected_types,
        "workspace 全体（crates/*/src/）の prefetch 型宣言集合が期待\
         （tensor-core/src/data.rs に各 1 件）と一致しない（過不足いずれも\
         fail-closed に検出する）: {found_types:?}"
    );
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
    let _loss_l1 = Loss::L1;
    let _loss_bce = Loss::Bce;
    let _loss_bcewithlogits = Loss::BceWithLogits;
    let _loss_nll = Loss::Nll;
    let _loss_kldiv = Loss::KlDiv;
    let _loss_huber = Loss::Huber;
    let _loss_smoothl1 = Loss::SmoothL1;

    let _optimizer = Optimizer::Sgd(fandhe_ai::optim::SgdConfig::new(0.1));
    // イシュー #2170: RmsProp／Adagrad／Lamb variant の facade 経由
    // 到達性（facade のみ import での構築可能性）を固定する。
    let _optimizer_rmsprop = Optimizer::RmsProp(fandhe_ai::optim::RmsPropConfig::default());
    let _optimizer_adagrad = Optimizer::Adagrad(fandhe_ai::optim::AdagradConfig::default());
    let _optimizer_lamb = Optimizer::Lamb(fandhe_ai::optim::LambConfig::default());
    // イシュー #2172（2026-09-27 承認）: Lbfgs variant・`LbfgsConfig` の
    // facade 経由到達性を固定する（`Lbfgs`／`LbfgsLineSearch` 自体は
    // 承認範囲外のため facade 未再エクスポートのまま。`optim.rs` 冒頭
    // コメント「L-BFGS」節参照）。
    let _optimizer_lbfgs = Optimizer::Lbfgs(fandhe_ai::optim::LbfgsConfig::default());

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

/// `fandhe_ai::nn::init`（PyTorch `torch.nn.init.*` 相当。イシュー #2504。
/// ルート #2499 の一括承認・`docs/compat-api-scope.md` §5 経路 2）が公開する
/// 13 名（初期化関数 9 個＋補助 4 個。`docs/facade-nn-init-exposure-decision.md`
/// §2.1）。承認形の正ガード
/// [`facade_reexports_nn_init_items_only_in_approved_shape`] と
/// `nn_init_module_reexports_exactly_expected_surface` が共用する。
const NN_INIT_NAMES: [&str; 13] = [
    "FanMode",
    "Nonlinearity",
    "calculate_fan_in_and_fan_out",
    "calculate_gain",
    "constant",
    "kaiming_normal",
    "kaiming_uniform",
    "normal",
    "orthogonal",
    "trunc_normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
];

/// [`facade_reexports_nn_init_items_only_in_approved_shape`]・その自己
/// テストが共用する検出本体。`content`（1 ファイル分）の `pub use` から
/// [`NN_INIT_NAMES`] の葉を集め、承認形（接頭辞 `fandhe_ai_autodiff::nn::init::`・
/// `as` 別名なし。[`nn_init_approved_prefix`]）の出現葉を第 1 要素、承認形
/// から外れる出現（別 path・別名・glob）と同名の独自宣言（`trait`／`struct`／
/// `enum`／`type`／`fn`）を第 2 要素（違反）として返す。加えて path に
/// `nn::init` を含む `pub use` は承認形でなければ（`NN_INIT_NAMES` 以外の葉を
/// 含めて）違反とする。
fn scan_nn_init_reexports_and_declarations(content: &str) -> (Vec<String>, Vec<String>) {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut approved: Vec<String> = Vec::new();
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let shape_ok =
                nn_init_approved_prefix(path_tokens) && !path_tokens.iter().any(|t| t == "as");
            let mentions_init_path = path_tokens
                .windows(4)
                .any(|w| w[0] == "nn" && w[1] == ":" && w[2] == ":" && w[3] == "init");
            let hits: Vec<String> = collect_pub_use_leaves(path_tokens)
                .into_iter()
                .filter(|leaf| NN_INIT_NAMES.contains(&leaf.as_str()))
                .collect();
            if shape_ok {
                // glob は承認形の接頭辞でも禁止（13 名の明示列挙のみ許可）。
                if path_tokens.iter().any(|t| t == "*") {
                    offending.push("承認形接頭辞の glob 再エクスポート".to_string());
                }
                approved.extend(hits);
            } else {
                for leaf in hits {
                    offending.push(format!("承認形外の pub use leaf={leaf}"));
                }
                if mentions_init_path {
                    offending.push("承認形外の nn::init 経由 pub use".to_string());
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(
            tokens[i].as_str(),
            "trait" | "struct" | "enum" | "type" | "fn"
        ) && tokens
            .get(i + 1)
            .map(|t| NN_INIT_NAMES.contains(&t.as_str()))
            .unwrap_or(false)
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    (approved, offending)
}

/// 承認形の正ガード（#2504 で `facade_does_not_reexport_nn_init` から反転。
/// 先例: #2501 の `facade_reexports_optimizer_ext_items_only_in_approved_shape`）。
/// facade src 全体で、[`NN_INIT_NAMES`] の 13 名が `src/nn/init.rs` の
/// `pub use fandhe_ai_autodiff::nn::init::…`（別名・glob なし）としてちょうど
/// 1 回ずつ出現し、承認形外の再エクスポート・迂回経路・同名の独自宣言が
/// 存在しないことを fail-closed に固定する。
#[test]
fn facade_reexports_nn_init_items_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_init_rs: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let (approved, bad) = scan_nn_init_reexports_and_declarations(content);
        for offense in bad {
            offending.push(format!("{}: {offense}", path.display()));
        }
        if path.ends_with("src/nn/init.rs") {
            approved_in_init_rs.extend(approved);
        } else {
            for leaf in approved {
                offending.push(format!(
                    "{}: nn/init.rs 以外での pub use leaf={leaf}",
                    path.display()
                ));
            }
        }
    });
    approved_in_init_rs.sort();
    let mut expected: Vec<String> = NN_INIT_NAMES.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert!(
        offending.is_empty(),
        "facade の公開面が nn::init（#2504）を承認形（src/nn/init.rs の \
         `pub use fandhe_ai_autodiff::nn::init::…`・別名・glob なし）以外で再エクスポート、\
         または独自宣言している（`docs/facade-nn-init-exposure-decision.md` §4）: {offending:?}"
    );
    assert_eq!(
        approved_in_init_rs, expected,
        "src/nn/init.rs に承認形の 13 識別子がちょうど 1 回ずつ存在しない\
         （過不足・重複いずれも fail。検査対象を見失った場合を含む）"
    );
}

/// [`scan_nn_init_reexports_and_declarations`] の自己テスト（正例・負例の合成入力）。
#[test]
fn facade_reexports_nn_init_items_only_in_approved_shape_detects_each_category() {
    let scan = scan_nn_init_reexports_and_declarations;
    let (ok, bad) = scan("pub use fandhe_ai_autodiff::nn::init::{constant, normal};");
    assert!(bad.is_empty());
    assert_eq!(ok, vec!["constant", "normal"]);
    // 違反: 別名。
    assert!(
        !scan("pub use fandhe_ai_autodiff::nn::init::uniform as u;")
            .1
            .is_empty()
    );
    // 違反: 別 path 接頭辞（迂回経路）。
    assert!(
        !scan("pub use fandhe_ai_autodiff::nn::init as i;")
            .1
            .is_empty()
    );
    assert!(!scan("pub use crate::nn::init::orthogonal;").1.is_empty());
    // 違反: glob。
    assert!(
        !scan("pub use fandhe_ai_autodiff::nn::init::*;")
            .1
            .is_empty()
    );
    // 違反: 独自宣言。
    assert!(!scan("pub fn xavier_uniform() {}").1.is_empty());
    assert!(!scan("pub enum FanMode {}").1.is_empty());
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use fandhe_ai_autodiff::nn::init::normal;",
        "let s = \"uniform\";",
        "use fandhe_ai_autodiff::nn::init::normal;",
        "pub use fandhe_ai_autodiff::nn::optim::AdamW;",
    ] {
        let (ok, bad) = scan(src);
        assert!(ok.is_empty() && bad.is_empty(), "src={src:?}");
    }
}

fn nn_init_rs_path() -> std::path::PathBuf {
    facade_crate_root().join("src/nn/init.rs")
}

/// `src/nn/init.rs` の `pub use` 行から `{...}` 内の識別子を抽出し、
/// [`NN_INIT_NAMES`] の 13 名と完全一致（過不足とも fail）することを固定する
/// （`nn_rnn_module_reexports_exactly_expected_surface` の鏡写し。rustfmt の
/// 折り返しに依存しないよう、コメント除去後に `;` 区切りの文として解析する）。
#[test]
fn nn_init_module_reexports_exactly_expected_surface() {
    let content = read_to_string_or_panic(&nn_init_rs_path());
    // rustfmt が複数行へ折り返す場合があるため、コメントを除去したうえで
    // `;` 区切りの文として解析する。
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let allowed_prefix = "pub use fandhe_ai_autodiff::nn::init::";
    let mut found: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut offending_lines = Vec::new();
    for stmt in cleaned.split(';') {
        let flat = stmt.split_whitespace().collect::<Vec<_>>().join(" ");
        if !flat.starts_with("pub use") {
            continue;
        }
        let compact = flat.replace(' ', "");
        let Some(rest) = compact.strip_prefix(&allowed_prefix.replace(' ', "")) else {
            offending_lines.push(flat);
            continue;
        };
        let (Some(open), Some(close)) = (rest.find('{'), rest.find('}')) else {
            offending_lines.push(flat);
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
        "src/nn/init.rs の pub use が fandhe_ai_autodiff::nn::init:: 以外の接頭辞を持つか、\
         `{{...}}` 形式でない行を含む: {offending_lines:?}"
    );
    let expected: std::collections::BTreeSet<String> =
        NN_INIT_NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        found, expected,
        "src/nn/init.rs が再エクスポートする識別子が 13 名の期待集合と一致しない"
    );
}

/// `src/nn/init.rs` が facade 独自の型・関数を定義しない純再エクスポート
/// モジュールであることを固定する（[`scan_forbidden_pub_items`] を再利用）。
#[test]
fn nn_init_module_is_pure_reexport() {
    let content = read_to_string_or_panic(&nn_init_rs_path());
    let offending = scan_forbidden_pub_items(&content);
    assert!(
        offending.is_empty(),
        "src/nn/init.rs が facade 独自の公開宣言を定義している\
         （純再エクスポートモジュールの契約違反）: {offending:?}"
    );
}

/// `fandhe_ai::nn::init` の 13 名が facade のみを通じて到達可能であること
/// のコンパイル時＋実行時固定（`fandhe_ai_autodiff` は import しない）。
#[test]
fn nn_init_items_are_reachable_via_facade_only() {
    use fandhe_ai::nn::init::{
        FanMode, Nonlinearity, calculate_fan_in_and_fan_out, calculate_gain, constant,
        kaiming_normal, kaiming_uniform, normal, orthogonal, trunc_normal, uniform, xavier_normal,
        xavier_uniform,
    };
    fandhe_ai::manual_seed(7);
    let t: fandhe_ai::Tensor<f32> = uniform(&[2, 3], -1.0, 1.0).expect("test fixture: uniform");
    assert_eq!(t.shape(), &[2usize, 3]);
    normal(&[2, 3], 0.0, 1.0).expect("test fixture: normal");
    constant(&[2], 1.5).expect("test fixture: constant");
    xavier_uniform(&[4, 5], 1.0).expect("test fixture: xavier_uniform");
    xavier_normal(&[4, 5], 1.0).expect("test fixture: xavier_normal");
    kaiming_uniform(&[4, 5], 0.0, FanMode::FanIn, Nonlinearity::Relu)
        .expect("test fixture: kaiming_uniform");
    kaiming_normal(&[4, 5], 0.0, FanMode::FanOut, Nonlinearity::Relu)
        .expect("test fixture: kaiming_normal");
    orthogonal(&[3, 3], 1.0).expect("test fixture: orthogonal");
    trunc_normal(&[2, 3], 0.0, 1.0, -2.0, 2.0).expect("test fixture: trunc_normal");
    assert!(calculate_gain(Nonlinearity::Relu) > 1.0);
    assert_eq!(
        calculate_fan_in_and_fan_out(&[4, 5]).expect("test fixture: fan"),
        (5, 4)
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
/// 追加すると本テストが fail する。案 B（facade 独自 trait）は #2395・#2396 で
/// `src/nn/mod.rs` の `pub use` として実装済みで、本テストの期待集合は不変
/// （`Module` は rnn 経由では再エクスポートしない）。nn 全体の公開 item 集合は
/// `nn_mod_public_items_match_expected_set` が固定する（#2399）。
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
/// （`nn_mod_declares_only_init_and_rnn_submodules` が別途固定する）。
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
/// 担う**: `nn::container` 等の `pub mod` 新設はこの完全一致検査に
/// より fail する。#2395 の `nn::Module` は非公開 `mod module;` と
/// `pub use module::Module;` で公開したため期待集合は不変
/// （`pub use` 経路は `facade_does_not_reexport_nn_module_or_containers`
/// が固定する）。**#2140 の保留
/// （`docs/facade-nn-init-exposure-decision.md` §3・§4）も本テストが
/// 担う**: `nn::init` の facade 公開（条件付き手順 §4）で想定する
/// `pub mod init;` 追加はこの完全一致検査により現時点では fail する。
/// 承認後に実装する際は期待集合（`["init", "rnn"]` 等）へ更新する。
#[test]
fn nn_mod_declares_only_init_and_rnn_submodules() {
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
        vec!["pub mod init;", "pub mod rnn;"],
        "src/nn/mod.rs が宣言する pub mod が `pub mod init;`・`pub mod rnn;` の 2 件と一致しない\
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

// ============================================================================
// compat::model_io 公開面の機械検査（イシュー #2369・親 #2362）
// ============================================================================

/// `src/compat/model_io.rs` の公開アイテム（`pub` 直後が `(` でないもの。`pub(crate)`／
/// `pub(super)` は対象外）が承認範囲の `fn save_model`・`fn load_model`・`enum ModelIoError`
/// の 3 件だけであることを走査する（`scan_unapproved_model_pub_items` と同型の独自許可リスト）。
fn scan_unapproved_model_io_pub_items(original: &str) -> Vec<String> {
    const ALLOWED_PUB_ITEMS: [(&str, &str); 3] = [
        ("fn", "save_model"),
        ("fn", "load_model"),
        ("enum", "ModelIoError"),
    ];
    let cleaned: String = strip_comments_and_literals(original).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offenses = Vec::new();
    for i in 0..tokens.len() {
        if tokens[i] != "pub" || tokens.get(i + 1).map(String::as_str) == Some("(") {
            continue;
        }
        let kind = tokens.get(i + 1).map(String::as_str).unwrap_or_default();
        let name = tokens.get(i + 2).map(String::as_str).unwrap_or_default();
        if !ALLOWED_PUB_ITEMS.contains(&(kind, name)) {
            offenses.push(format!("pub {kind} {name}"));
        }
    }
    offenses
}

#[test]
fn model_io_module_exposes_only_approved_surface() {
    let content = read_to_string_or_panic(&facade_crate_root().join("src/compat/model_io.rs"));
    let offenses = scan_unapproved_model_io_pub_items(&content);
    assert!(
        offenses.is_empty(),
        "src/compat/model_io.rs に承認範囲外の公開アイテムが見つかった: {offenses:?}"
    );
    // 承認済みの 3 件が実在すること（走査の空振りで通過しない）。
    assert!(content.contains("pub fn save_model("));
    assert!(content.contains("pub fn load_model("));
    assert!(content.contains("pub enum ModelIoError"));
}

/// [`scan_unapproved_model_io_pub_items`] の自己テスト（承認済みは違反 0、
/// 承認外の `pub` 項目は検出し、`pub(crate)`／`pub(super)` は対象外）。
#[test]
fn scan_unapproved_model_io_pub_items_detects_offense() {
    let approved = "pub enum ModelIoError {}\npub fn save_model() {}\npub fn load_model() {}\n\
                    pub(crate) fn helper() {}\npub(super) struct Hidden;\n";
    assert!(scan_unapproved_model_io_pub_items(approved).is_empty());
    let offenses = scan_unapproved_model_io_pub_items("pub fn rogue() {}\npub struct Extra;\n");
    assert_eq!(offenses.len(), 2, "offenses={offenses:?}");
    assert!(offenses[0].contains("rogue"));
    assert!(!scan_unapproved_model_io_pub_items("pub fn save_model_v2() {}").is_empty());
}

/// `fandhe_ai::compat::{save_model, load_model, ModelIoError}` が facade から到達可能で、
/// 承認済みのシグネチャ（`Result<_, ModelIoError>`・`impl AsRef<Path>`）であることを
/// コンパイル時に固定する。`ModelIoError` は `#[non_exhaustive]` のためワイルドカード腕を持つ
/// `match` で 7 variant を網羅できることも併せて確認する。
#[test]
fn model_io_items_are_reachable_via_facade() {
    use fandhe_ai::compat::{ModelIoError, Sequential};

    fn _assert_error_matchable(e: &ModelIoError) -> &'static str {
        match e {
            ModelIoError::Io(_) => "io",
            ModelIoError::Manifest { .. } => "manifest",
            ModelIoError::Safetensors(_) => "safetensors",
            ModelIoError::UnsupportedModel { .. } => "unsupported_model",
            ModelIoError::Mismatch { .. } => "mismatch",
            ModelIoError::Autodiff(_) => "autodiff",
            ModelIoError::TooLarge { .. } => "too_large",
            _ => "unknown",
        }
    }
    fn _assert_signatures(model: &Sequential, dir: &Path) {
        let _: Result<(), ModelIoError> = fandhe_ai::compat::save_model(model, dir);
        let _: Result<(), ModelIoError> =
            fandhe_ai::compat::save_model(model, std::path::PathBuf::new());
        let _: Result<Sequential, ModelIoError> = fandhe_ai::compat::load_model(dir);
        let _: Result<Sequential, ModelIoError> =
            fandhe_ai::compat::load_model(std::path::PathBuf::new());
    }
    // `std::error::Error` を実装していること（`?` で `Box<dyn Error>` へ流せる）。
    fn _assert_error_trait<E: std::error::Error + Send + Sync + 'static>() {}
    _assert_error_trait::<ModelIoError>();
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
    let guard = TempDirGuard::new("api-surface-inline-mod");
    let root = guard.path().to_path_buf();
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
/// hold_doctest_probe_body_matches_fixed_contract`]・
/// `nn_module_hold_doctest_probe_body_matches_fixed_contract`
/// （#2396 で削除済み）共用。元は
/// `VarCustomHoldDoctestGuard` 専用の固定名関数だったが、#2133 で
/// `NnModuleHoldDoctestGuard`（#2396 で削除済み）にも同じ抽出が必要になったため `struct_name`
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

/// facade src の全 `pub use` 文（`pub(..) use` は対象外）から
/// [`collect_pub_use_leaves`] で葉（ソース側・rename 前）を集め、承認済みの配置
/// 以外での `Module`／`ModuleList`／`Sequential` の再エクスポートを違反とする
/// （#2133 Step 2-1 の否定ガードを #2396 で正ガードへ切り替え）。
///
/// 承認済みの形は `src/nn/mod.rs` の次の 2 つだけ（#2395・#2396。非公開 `mod` からの
/// 再エクスポート）:
/// - `pub use module::Module;`
/// - `pub use container::{ModuleList, Sequential};`
///
/// `src/nn/mod.rs` 内では、葉が 3 名のいずれかで上記完全一致形でない `pub use`
/// （別名・分割形を含む）をすべて違反とする。`src/nn/mod.rs` の外では
/// `Module`／`ModuleList` の葉は常に違反、`Sequential` の葉はパスに
/// `fandhe_ai_autodiff`／`nn`／`container` を含むとき違反とする。
/// `compat/mod.rs` の `pub use sequential::{Sequential, SequentialVars};` は許容する。
/// 別名（`as Layer` 等）の前のソース側の葉で判定するため別名再エクスポートも検出する。
#[test]
fn facade_does_not_reexport_nn_module_or_containers() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut allowed_total = 0usize;
    visit_rs_files(&src_dir, &mut |path, content| {
        let (offenses, allowed) = scan_nn_module_reexports(content, path);
        offending.extend(offenses);
        allowed_total += allowed;
    });
    assert!(
        offending.is_empty(),
        "facade の pub use が承認済みの 2 経路（`src/nn/mod.rs` の `pub use \
         module::Module;`・`pub use container::{{ModuleList, Sequential}};`）以外で \
         nn::Module／ModuleList／nn 系 Sequential を再エクスポートしている: {offending:?}"
    );
    // インベントリ: 承認済みの葉がちょうど 3 件あること（走査の空振り検出）。
    assert_eq!(
        allowed_total, 3,
        "src/nn/mod.rs の承認済み `pub use`（Module・ModuleList・Sequential の葉）が\
         ちょうど 3 件であること"
    );
}

/// `content`（`path` 由来）の `pub use` を走査し、違反文字列の列と
/// 「承認済み経路」の葉の出現数を返す（[`facade_does_not_reexport_nn_module_or_containers`]
/// ・自己テスト共用）。承認済み経路は、ファイルパス末尾が `src/nn/mod.rs` で、
/// パストークンがちょうど `module :: Module`（葉 1 件）または
/// `container :: { ModuleList , Sequential }`（葉 2 件）の `pub use` だけである。
fn scan_nn_module_reexports(content: &str, path: &Path) -> (Vec<String>, usize) {
    let is_nn_mod = path
        .to_string_lossy()
        .replace('\\', "/")
        .ends_with("src/nn/mod.rs");
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending = Vec::new();
    let mut allowed = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            let path_strs = path_tokens.iter().map(String::as_str).collect::<Vec<_>>();
            let path_contains_nn_autodiff = path_tokens
                .iter()
                .any(|t| t == "fandhe_ai_autodiff" || t == "nn" || t == "container");
            let is_approved_form = is_nn_mod
                && (path_strs == ["module", ":", ":", "Module"]
                    || path_strs
                        == [
                            "container",
                            ":",
                            ":",
                            "{",
                            "ModuleList",
                            ",",
                            "Sequential",
                            "}",
                        ]);
            for leaf in leaves {
                let is_target = matches!(leaf.as_str(), "Module" | "ModuleList" | "Sequential");
                let offense = if !is_target {
                    false
                } else if is_approved_form {
                    allowed += 1;
                    false
                } else if is_nn_mod {
                    true
                } else {
                    match leaf.as_str() {
                        "Sequential" => path_contains_nn_autodiff,
                        _ => true,
                    }
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
    (offending, allowed)
}

/// [`facade_does_not_reexport_nn_module_or_containers`] の自己テスト
/// （正例・負例の合成入力）。
#[test]
fn facade_does_not_reexport_nn_module_or_containers_detects_each_category() {
    let nn_mod = Path::new("crates/facade/src/nn/mod.rs");
    let lib = Path::new("crates/facade/src/lib.rs");
    let off = |c: &str, p: &Path| scan_nn_module_reexports(c, p).0;

    // 正例（違反として検出される）。
    assert!(!off("pub use module::Module;", lib).is_empty());
    assert!(!off("pub use fandhe_ai_autodiff::nn::Module;", nn_mod).is_empty());
    assert!(!off("pub use module::{Module, ModuleList};", nn_mod).is_empty());
    assert_eq!(
        off("pub use module::{Module, ModuleList};", nn_mod).len(),
        2
    );
    assert!(!off("pub use fandhe_ai_autodiff::nn::{Module as Layer};", nn_mod).is_empty());
    assert!(
        !off(
            "pub use fandhe_ai_autodiff::nn::{self as n, ModuleList};",
            lib
        )
        .is_empty()
    );
    assert!(!off("pub use fandhe_ai_autodiff::nn::Sequential;", nn_mod).is_empty());
    assert!(
        !off(
            "pub use fandhe_ai_autodiff::nn::{ModuleList, Sequential};",
            nn_mod
        )
        .is_empty()
    );
    // #2396: 承認済み配置（`src/nn/mod.rs` の container 形）の別名・分割形・ファイル違い。
    assert!(!off("pub use container::Sequential as Seq;", nn_mod).is_empty());
    assert!(!off("pub use container::{ModuleList as L, Sequential};", nn_mod).is_empty());
    assert!(!off("pub use container::ModuleList;", nn_mod).is_empty());
    assert!(!off("pub use container::Sequential;", nn_mod).is_empty());
    assert!(!off("pub use nn::Sequential;", lib).is_empty());
    assert!(!off("pub use container::{ModuleList, Sequential};", lib).is_empty());
    assert!(!off("pub use module::ModuleList;", nn_mod).is_empty());

    // 負例: 承認済みの 2 形（違反 0・葉 1 件／2 件）。
    assert_eq!(
        scan_nn_module_reexports("pub use module::Module;", nn_mod),
        (Vec::new(), 1)
    );
    assert_eq!(
        scan_nn_module_reexports("pub use container::{ModuleList, Sequential};", nn_mod),
        (Vec::new(), 2)
    );
    // 負例: `compat::Sequential`（パスに fandhe_ai_autodiff／nn／container を含まない）。
    assert!(off("pub use sequential::{Sequential, SequentialVars};", lib).is_empty());
    assert!(
        off(
            "pub use sequential::{Sequential, SequentialVars};",
            Path::new("crates/facade/src/compat/mod.rs")
        )
        .is_empty()
    );
    // 負例: 非 pub。
    assert!(off("use fandhe_ai_autodiff::nn::Module;", nn_mod).is_empty());
}

/// `pub trait Module` の本体（brace 深さ 1）から `fn` を「required（`;` 終わり）」
/// 「defaulted（本体あり）」に分類し、シグネチャのトークン列とともに返す。
/// `Module` trait が 0 件・複数件の場合は `None`（走査の空振り検出用）。
fn scan_module_trait_methods(content: &str) -> Option<Vec<(String, bool, Vec<String>)>> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let starts: Vec<usize> = (0..tokens.len().saturating_sub(2))
        .filter(|&i| tokens[i] == "pub" && tokens[i + 1] == "trait" && tokens[i + 2] == "Module")
        .collect();
    if starts.len() != 1 {
        return None;
    }
    let mut i = starts[0] + 3;
    while i < tokens.len() && tokens[i] != "{" {
        i += 1;
    }
    let mut depth = 0i32;
    let mut out = Vec::new();
    while i < tokens.len() {
        match tokens[i].as_str() {
            "{" => depth += 1,
            "}" => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            "fn" if depth == 1 => {
                let name = tokens.get(i + 1).cloned().unwrap_or_default();
                let mut j = i;
                let mut paren = 0i32;
                while j < tokens.len() {
                    match tokens[j].as_str() {
                        "(" => paren += 1,
                        ")" => paren -= 1,
                        "{" | ";" if paren == 0 => break,
                        _ => {}
                    }
                    j += 1;
                }
                let required = tokens.get(j).map(String::as_str) == Some(";");
                out.push((name, required, tokens[i..j.min(tokens.len())].to_vec()));
            }
            _ => {}
        }
        i += 1;
    }
    Some(out)
}

/// #2395 の正ガード: facade `nn::Module` の面が承認済み集合
/// （required = `forward` のみ・defaulted = 14 件。#2400 で凍結 API 3 件と `children_mut`・#2401 で introspection 4 件を追加）と完全一致すること
/// （#2338 承認事項 2）。`forward_host`・`as_*` 等の内部フックの混入を拒否する。
/// #2400 で凍結 API 3 件（`set_requires_grad`・`freeze`・`requires_grad`）を追加し 9 件。
/// #2401 で鏡写しのメソッドを足すときは本集合を更新する。
#[test]
fn facade_nn_module_trait_methods_match_approved_set() {
    let src = read_to_string_or_panic(&facade_crate_root().join("src/nn/module.rs"));
    let methods = scan_module_trait_methods(&src).expect("`pub trait Module` がちょうど 1 件");
    assert_module_surface(&methods).expect("承認済み集合と一致");
}

fn assert_module_surface(methods: &[(String, bool, Vec<String>)]) -> Result<(), String> {
    use std::collections::BTreeSet;
    let required: BTreeSet<&str> = methods
        .iter()
        .filter(|m| m.1)
        .map(|m| m.0.as_str())
        .collect();
    let defaulted: BTreeSet<&str> = methods
        .iter()
        .filter(|m| !m.1)
        .map(|m| m.0.as_str())
        .collect();
    let want_req: BTreeSet<&str> = ["forward"].into();
    let want_def: BTreeSet<&str> = [
        "named_parameters",
        "set_parameter",
        "state_dict",
        "load_state_dict",
        "set_training",
        "training",
        "set_requires_grad",
        "freeze",
        "requires_grad",
        "children",
        "named_modules",
        "parameter_count",
        "type_name",
        // #2400 レビュー是正（PR #2426。2026-09-29 ユーザー承認）: `children` と対の公開
        // defaulted メソッド。凍結ロールバックが利用者定義の複合層も葉単位で復元するために使う。
        "children_mut",
    ]
    .into();
    if required != want_req || defaulted != want_def {
        return Err(format!("required={required:?} defaulted={defaulted:?}"));
    }
    Ok(())
}

/// シグネチャに内部型（`fandhe_ai_autodiff`・`BackendOps`・裸の `Tape`）が現れず、
/// `forward` の第 1 引数型が `TapeRef` であること（REQ-12）。
fn check_module_signatures(methods: &[(String, bool, Vec<String>)]) -> Result<(), String> {
    for (name, _, toks) in methods {
        if let Some(bad) = toks
            .iter()
            .find(|t| matches!(t.as_str(), "fandhe_ai_autodiff" | "BackendOps" | "Tape"))
        {
            return Err(format!("{name}: 内部型 `{bad}` が公開シグネチャに現れる"));
        }
        if name == "forward" {
            // `self` `,` `tape` `:` <型> の <型> 先頭が TapeRef。
            let pos = toks.iter().position(|t| t == "tape");
            let ty = pos.and_then(|p| toks.get(p + 2));
            if ty.map(String::as_str) != Some("TapeRef") {
                return Err("forward の第 1 引数型が TapeRef でない".into());
            }
        }
    }
    Ok(())
}

#[test]
fn facade_nn_module_trait_signatures_hide_internal_types() {
    let src = read_to_string_or_panic(&facade_crate_root().join("src/nn/module.rs"));
    let methods = scan_module_trait_methods(&src).expect("`pub trait Module` がちょうど 1 件");
    check_module_signatures(&methods).expect("シグネチャ検査");
}

/// 正ガード 2 種の自己テスト（合成入力で各逸脱が検出されること）。
#[test]
fn facade_nn_module_trait_guards_detect_each_category() {
    let ok = "pub trait Module { fn forward<'t>(&self, tape: TapeRef<'t>, input: &Var<'t>) -> R; \
        fn named_parameters(&self) -> V { V } fn set_parameter(&mut self, n: &str) -> R { R } \
        fn state_dict(&self) -> H { H } fn load_state_dict(&mut self, s: H) -> R { R } \
        fn set_training(&mut self, t: bool) {} fn training(&self) -> bool { true } \
        fn set_requires_grad(&mut self, r: bool) -> R { R } fn freeze(&mut self) -> R { R } \
        fn requires_grad(&self) -> bool { true } \
        fn children(&self) -> V { V } fn named_modules(&self) -> V { V } \
        fn parameter_count(&self) -> usize { 0 } fn type_name(&self) -> &'static str { S } \
        fn children_mut(&mut self) -> V { V } }";
    let m = scan_module_trait_methods(ok).expect("ok");
    assert!(assert_module_surface(&m).is_ok());
    assert!(check_module_signatures(&m).is_ok());

    let extra_def = ok.replace("fn training", "fn forward_host(&self) {} fn training");
    let m = scan_module_trait_methods(&extra_def).expect("extra");
    assert!(assert_module_surface(&m).is_err());
    let missing = ok.replace("fn training(&self) -> bool { true }", "");
    let m = scan_module_trait_methods(&missing).expect("missing");
    assert!(assert_module_surface(&m).is_err());
    let missing_intro = ok.replace("fn type_name(&self) -> &'static str { S }", "");
    let m = scan_module_trait_methods(&missing_intro).expect("missing_intro");
    assert!(assert_module_surface(&m).is_err());
    let extra_req = ok.replace("fn training", "fn extra(&self); fn training");
    let m = scan_module_trait_methods(&extra_req).expect("req");
    assert!(assert_module_surface(&m).is_err());
    let as_hook = ok.replace(
        "fn training",
        "fn as_linear(&self) -> Option<u8> { None } fn training",
    );
    let m = scan_module_trait_methods(&as_hook).expect("as");
    assert!(assert_module_surface(&m).is_err());

    let host = ok.replace(
        "fn training",
        "fn h(&self, ops: &dyn BackendOps) {} fn training",
    );
    let m = scan_module_trait_methods(&host).expect("host");
    assert!(check_module_signatures(&m).is_err());
    let raw = ok.replace("tape: TapeRef<'t>", "tape: &'t Tape");
    let m = scan_module_trait_methods(&raw).expect("raw");
    assert!(check_module_signatures(&m).is_err());
    let autodiff = ok.replace("input: &Var<'t>", "input: &fandhe_ai_autodiff::Var<'t>");
    let m = scan_module_trait_methods(&autodiff).expect("ad");
    assert!(check_module_signatures(&m).is_err());

    assert!(scan_module_trait_methods("pub struct X;").is_none());
}

/// facade src に facade 独自の `trait Module`／`struct ModuleList`／
/// `enum`／`type` 版・`struct Sequential`（`src/compat/sequential.rs` 以外）
/// の宣言が可視性を問わず存在しないことを固定する（#2133 Step 2-2。
/// `declares_fn_named` と同型のトークン走査で `trait`／`struct`／`enum`／
/// `type` トークンの直後の識別子を見る）。`use fandhe_ai_autodiff::nn::
/// {..., Module, ...}`（非公開 import）・`Box<dyn Module>`（型参照）・
/// `compat/sequential.rs` 自身の `pub struct Sequential` は正当な既存形
/// のため許容する。
///
/// #2396 で承認済み配置を正ガード化した: `struct ModuleList`／`struct Sequential` は
/// `src/nn/container.rs` のときだけ許容する（`trait`／`enum`／`type` 版は違反のまま）。
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
        "facade src に承認済み（`src/nn/module.rs` の `trait Module`・#2395、\
         `src/nn/container.rs` の `struct ModuleList`／`struct Sequential`・#2396／`struct ModuleDict`・#2402）以外の \
         nn::Module／ModuleList 相当の独自宣言が見つかった: {offending:?}"
    );
    // インベントリ: 承認済みの `trait Module` 宣言がちょうど 1 件。
    let module_rs = src_dir.join("nn/module.rs");
    let approved = scan_trait_module_decl_count(&read_to_string_or_panic(&module_rs));
    assert_eq!(
        approved, 1,
        "src/nn/module.rs の `trait Module` はちょうど 1 件"
    );
    // インベントリ（#2396）: 承認済みの `struct ModuleList`／`struct Sequential` が各 1 件。
    let container_rs = read_to_string_or_panic(&src_dir.join("nn/container.rs"));
    for name in ["ModuleList", "Sequential", "ModuleDict"] {
        assert_eq!(
            scan_decl_count(&container_rs, "struct", name),
            1,
            "src/nn/container.rs の `struct {name}` はちょうど 1 件"
        );
    }
}

/// `<kind> <name>` トークン列の出現数（コメント・リテラル除外）。
fn scan_decl_count(content: &str, kind: &str, name: &str) -> usize {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    tokens
        .windows(2)
        .filter(|w| w[0] == kind && w[1] == name)
        .count()
}

/// `trait Module` トークン列の出現数（コメント・リテラル除外）。
fn scan_trait_module_decl_count(content: &str) -> usize {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    tokens
        .windows(2)
        .filter(|w| w[0] == "trait" && w[1] == "Module")
        .count()
}

/// `content`（`path` 由来）を走査し、`trait`／`struct`／`enum`／`type`
/// トークンの直後に `Module`／`ModuleList` が続く宣言、または `Sequential`
/// が続く宣言（`path` のファイル名が `compat/sequential.rs` 以外）を
/// 検出して違反文字列の列を返す（[`facade_declares_no_nn_module_items`]・
/// その自己テスト共用）。
fn scan_nn_module_item_declarations(content: &str, path: &Path) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let norm_path = path.to_string_lossy().replace('\\', "/");
    let is_compat_sequential = norm_path.ends_with("compat/sequential.rs");
    let is_nn_module_rs = norm_path.ends_with("src/nn/module.rs");
    let is_nn_container_rs = norm_path.ends_with("src/nn/container.rs");
    let mut offending = Vec::new();
    for (i, token) in tokens.iter().enumerate() {
        if !matches!(token.as_str(), "trait" | "struct" | "enum" | "type") {
            continue;
        }
        let Some(name) = tokens.get(i + 1).map(String::as_str) else {
            continue;
        };
        let offense = match name {
            // #2395: `src/nn/module.rs` の `trait Module` のみ承認済み。
            "Module" if token == "trait" && is_nn_module_rs => false,
            // #2396: `src/nn/container.rs` の `struct ModuleList`／`struct Sequential` のみ承認済み。
            "ModuleList" | "Sequential" if token == "struct" && is_nn_container_rs => false,
            // #2402: `src/nn/container.rs` の `struct ModuleDict` のみ承認済み。
            "ModuleDict" if token == "struct" && is_nn_container_rs => false,
            "ModuleDict" => true,
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
    let other = Path::new("src/nn/rnn.rs");
    let module_rs = Path::new("src/nn/module.rs");
    let compat_seq = Path::new("src/compat/sequential.rs");
    let container_rs = Path::new("src/nn/container.rs");

    // 正例。
    assert!(!scan_nn_module_item_declarations("pub trait Module {}", other).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct ModuleList;", other).is_empty());
    assert!(
        !scan_nn_module_item_declarations("pub type Layer = u8; pub struct ModuleList;", other)
            .is_empty()
    );
    assert!(!scan_nn_module_item_declarations("pub struct Sequential;", other).is_empty());

    // 承認済み: `src/nn/module.rs` の `trait Module` のみ許容。他の宣言は違反のまま。
    assert!(scan_nn_module_item_declarations("pub trait Module {}", module_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct Module;", module_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct ModuleList;", module_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct Sequential;", module_rs).is_empty());

    // 承認済み（#2396）: `src/nn/container.rs` の `struct ModuleList`／`struct Sequential` のみ許容。
    assert!(scan_nn_module_item_declarations("pub struct ModuleList;", container_rs).is_empty());
    assert!(scan_nn_module_item_declarations("pub struct Sequential;", container_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub trait ModuleList {}", container_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct Module;", container_rs).is_empty());
    assert!(
        !scan_nn_module_item_declarations("pub type Sequential = u8;", container_rs).is_empty()
    );
    assert!(!scan_nn_module_item_declarations("pub enum Sequential {}", container_rs).is_empty());
    assert_eq!(
        scan_decl_count(
            "pub struct Sequential; struct Sequential;",
            "struct",
            "Sequential"
        ),
        2
    );

    // 承認済み（#2402）: `src/nn/container.rs` の `struct ModuleDict` のみ許容。
    assert!(scan_nn_module_item_declarations("pub struct ModuleDict;", container_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct ModuleDict;", other).is_empty());
    assert!(!scan_nn_module_item_declarations("pub struct ModuleDict;", module_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub trait ModuleDict {}", container_rs).is_empty());
    assert!(!scan_nn_module_item_declarations("pub enum ModuleDict {}", container_rs).is_empty());
    assert!(
        !scan_nn_module_item_declarations("pub type ModuleDict = u8;", container_rs).is_empty()
    );

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

/// `src/compat` 配下の `add_module`／`add_boxed`／`push_module` の `fn` 宣言の扱いを判定する
/// （#2133 Step 2-3・#2398）。`rel_path` は `src/compat` からの相対パス。
/// `add_boxed`／`push_module` は常に違反。`add_module` は `sequential.rs` 内にちょうど 1 件の場合のみ
/// 許容する（#2398 でユーザー承認済みの唯一の公開入口）。違反の説明文を返す。
fn module_add_method_offenses(rel_path: &str, content: &str) -> Vec<String> {
    let tokens = tokenize_including_punctuation(
        &strip_comments_and_literals(content)
            .iter()
            .collect::<String>(),
    );
    let mut out = Vec::new();
    for name in ["add_boxed", "push_module"] {
        if count_fn_declarations_by_name(&tokens, name) > 0 {
            out.push(format!("{rel_path}: fn {name}"));
        }
    }
    let n = count_fn_declarations_by_name(&tokens, "add_module");
    let allowed = rel_path == "sequential.rs" && n == 1;
    if n > 0 && !allowed {
        out.push(format!("{rel_path}: fn add_module x{n}"));
    }
    out
}

/// `src/compat` 配下で `add_module` は `sequential.rs` の 1 件だけを許容し（#2398。
/// `docs/facade-nn-module-exposure-decision.md` §9 の旧「スコープ外」をユーザー承認 2026-09-29 で上書き）、
/// `add_boxed`／`push_module` は引き続き禁止する。許容件数のインベントリ assert で走査の空振りを検出する。
#[test]
fn compat_sequential_does_not_expose_module_add_methods() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut offenses = Vec::new();
    let mut approved = 0usize;
    visit_rs_files(&compat_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&compat_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        offenses.extend(module_add_method_offenses(&rel, content));
        let tokens = tokenize_including_punctuation(
            &strip_comments_and_literals(content)
                .iter()
                .collect::<String>(),
        );
        approved += count_fn_declarations_by_name(&tokens, "add_module");
    });
    assert!(
        offenses.is_empty(),
        "src/compat 配下に承認外の add_module／add_boxed／push_module が見つかった\
         （承認は sequential.rs の add_module 1 件のみ〈#2398〉）: {offenses:?}"
    );
    assert_eq!(
        approved, 1,
        "承認済み add_module がちょうど 1 件であること（走査の空振り検出）"
    );
}

/// [`compat_sequential_does_not_expose_module_add_methods`] の自己テスト。
#[test]
fn compat_sequential_does_not_expose_module_add_methods_detects_offense() {
    let one = "pub fn add_module<M: Module + 'static>(mut self, m: M) -> Self { self }";
    // 正例: sequential.rs 内の 1 件は許容。コメント中の宣言風テキストは無視される。
    assert!(module_add_method_offenses("sequential.rs", one).is_empty());
    assert!(
        module_add_method_offenses("training.rs", "// pub fn add_module(self) {}\nfn x() {}")
            .is_empty()
    );
    // 負例: sequential.rs 以外の add_module、add_boxed／push_module、2 宣言。
    assert!(!module_add_method_offenses("training.rs", one).is_empty());
    assert!(!module_add_method_offenses("sequential.rs", "pub fn add_boxed(self) {}").is_empty());
    assert!(
        !module_add_method_offenses("sequential.rs", "pub fn push_module(&mut self) {}").is_empty()
    );
    let two = format!("{one}\n{one}");
    assert!(!module_add_method_offenses("sequential.rs", &two).is_empty());
}

/// `Sequential::add_module` が承認済みの唯一の入口として、承認済みシグネチャ
/// （`crate::nn::Module` bound・`'static`・`-> Self`。内部型を含まない）で存在することを固定する
/// 正ガード（#2398。REQ-12 公開面の最小化）。
#[test]
fn compat_sequential_add_module_is_sole_approved_entry() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(count_fn_declarations_by_name(&tokens, "add_module"), 1);
    assert!(sequential_add_module_signature_ok(&cleaned));
}

/// `pub fn add_module` から本体開始 `{` までの宣言部が承認済みシグネチャか判定する。
fn sequential_add_module_signature_ok(cleaned: &str) -> bool {
    let Some(start) = cleaned.find("pub fn add_module") else {
        return false;
    };
    let Some(len) = cleaned[start..].find('{') else {
        return false;
    };
    let sig: String = cleaned[start..start + len]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    ["crate::nn::Module", "'static", "->Self"]
        .iter()
        .all(|w| sig.contains(w))
        && ["fandhe_ai_autodiff", "Box<dyn", "BackendOps", "Tape"]
            .iter()
            .all(|w| !sig.contains(w))
}

/// [`compat_sequential_add_module_is_sole_approved_entry`] の自己テスト。
#[test]
fn compat_sequential_add_module_is_sole_approved_entry_detects_offense() {
    let ok = "pub fn add_module<M: crate::nn::Module + 'static>(mut self, m: M) -> Self {";
    assert!(sequential_add_module_signature_ok(ok));
    for bad in [
        "pub fn add_module<M: crate::nn::Module + 'static>(mut self, m: M) -> Result<Self, E> {",
        "pub fn add_module<M: Module>(mut self, m: M) -> Self {",
        "pub fn add_module(mut self, m: Box<dyn crate::nn::Module + 'static>) -> Self {",
        "pub fn add_module<M: crate::nn::Module + 'static>(mut self, t: &Tape, m: M) -> Self {",
        "pub fn add_module<M: fandhe_ai_autodiff::nn::Module + crate::nn::Module + 'static>(self, m: M) -> Self {",
        "pub fn add_linear(self) -> Self {",
    ] {
        assert!(!sequential_add_module_signature_ok(bad), "{bad}");
    }
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
    // `compat/mod.rs`（`compat::save_model`／`compat::load_model`。イシュー #2369・親 #2362）。
    "save_model",
    "load_model",
    // `nn/mod.rs`（`nn::summary`。イシュー #2402。`ModuleDict` と同じ承認済み 1 行）。
    "summary",
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
    // `data.rs`（イシュー #2505。Sampler／フック系の自由関数 `default_collate`）。
    "default_collate",
    // `nn/init.rs`（イシュー #2504。`torch.nn.init.*` 相当の初期化関数 9 個＋補助関数 2 個）。
    "uniform",
    "normal",
    "constant",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "trunc_normal",
    "calculate_gain",
    "calculate_fan_in_and_fan_out",
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
    let approved = unexpected_lowercase_leaves(
        "pub use array::{ArrayData, array}; pub use container::{ModuleDict, summary};",
    );
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

/// facade の `pub use` が `ModuleDict`／`summary` を承認済みの 1 形だけで公開することを固定する
/// （イシュー #2402 で承認済み配置を正ガード化。旧 #2134 では facade に `Module` trait が無く
/// 「未公開」を否定ガードで固定していた）。
///
/// 承認済みの形は `src/nn/mod.rs` の `pub use container::{ModuleDict, summary};`（トークン列
/// 完全一致・葉 2 件）だけである。パスのトークン列のどこかに `ModuleDict`／`summary` が現れる
/// `pub use`（別ファイル・`fandhe_ai_autodiff::nn::*` 経由・`as` 別名・分割形・順序違い・
/// 複数行）はすべて違反とする。承認済み配置の追加・変更時は本ガードを更新すること。
#[test]
fn facade_does_not_reexport_module_dict_or_summary() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    let mut allowed_total = 0usize;
    visit_rs_files(&src_dir, &mut |path, content| {
        let (offenses, allowed) = scan_module_dict_summary_reexports(content, path);
        offending.extend(offenses);
        allowed_total += allowed;
    });
    assert!(
        offending.is_empty(),
        "facade の `pub use` が承認済みの 1 形（`src/nn/mod.rs` の \
         `pub use container::{{ModuleDict, summary}};`）以外で ModuleDict／summary を公開している: \
         {offending:?}"
    );
    // インベントリ: 承認済みの葉がちょうど 2 件（走査の空振り検出）。
    assert_eq!(
        allowed_total, 2,
        "src/nn/mod.rs の承認済み `pub use`（ModuleDict・summary の葉）がちょうど 2 件であること"
    );
}

/// `content`（`path` 由来）の `pub use` を走査し、`ModuleDict`／`summary` を含むものについて
/// 違反文字列の列と承認済みの葉の件数を返す（[`facade_does_not_reexport_module_dict_or_summary`]
/// ・自己テスト共用）。
fn scan_module_dict_summary_reexports(content: &str, path: &Path) -> (Vec<String>, usize) {
    let is_nn_mod = path
        .to_string_lossy()
        .replace('\\', "/")
        .ends_with("src/nn/mod.rs");
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending = Vec::new();
    let mut allowed = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let hits = path_tokens
                .iter()
                .filter(|t| matches!(t.as_str(), "ModuleDict" | "summary"))
                .count();
            if hits > 0 {
                let path_strs = path_tokens.iter().map(String::as_str).collect::<Vec<_>>();
                let approved = is_nn_mod
                    && path_strs
                        == [
                            "container",
                            ":",
                            ":",
                            "{",
                            "ModuleDict",
                            ",",
                            "summary",
                            "}",
                        ];
                if approved {
                    allowed += hits;
                } else {
                    offending.push(format!(
                        "{}: `pub use {}`",
                        path.display(),
                        path_strs.join(" ")
                    ));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        i += 1;
    }
    (offending, allowed)
}

/// [`scan_module_dict_summary_reexports`] の自己テスト（正例・負例の合成入力）。
#[test]
fn facade_does_not_reexport_module_dict_or_summary_detects_each_category() {
    let nn_mod = Path::new("crates/facade/src/nn/mod.rs");
    let lib = Path::new("crates/facade/src/lib.rs");
    let off = |c: &str, p: &Path| scan_module_dict_summary_reexports(c, p).0;

    // 承認済みの 1 形（違反 0・葉 2 件）。
    assert_eq!(
        scan_module_dict_summary_reexports("pub use container::{ModuleDict, summary};", nn_mod),
        (Vec::new(), 2)
    );
    // 複数行でも同じ形として扱う。
    assert_eq!(
        scan_module_dict_summary_reexports(
            "pub use container::{\n    ModuleDict,\n    summary,\n};",
            nn_mod
        )
        .1,
        0,
        "末尾カンマ付きはトークン列が異なるため承認形ではない（違反）"
    );

    // 違反。
    assert!(!off("pub use container::{ModuleDict, summary};", lib).is_empty());
    assert!(!off("pub use container::{summary, ModuleDict};", nn_mod).is_empty());
    assert!(!off("pub use container::ModuleDict;", nn_mod).is_empty());
    assert!(!off("pub use container::summary;", nn_mod).is_empty());
    assert!(!off("pub use container::ModuleDict as D;", nn_mod).is_empty());
    assert!(!off("pub use container::{ModuleDict, summary as s};", nn_mod).is_empty());
    assert!(!off("pub use fandhe_ai_autodiff::nn::ModuleDict;", nn_mod).is_empty());
    assert!(!off("pub use fandhe_ai_autodiff::nn::summary;", lib).is_empty());
    assert!(!off("pub use foo::Bar as ModuleDict;", nn_mod).is_empty());
    assert!(!off("pub use foo::bar as summary;", nn_mod).is_empty());

    // 負例: 非 pub・`pub(crate)`・無関係な識別子・コメント中。
    assert!(off("use container::{ModuleDict, summary};", nn_mod).is_empty());
    assert!(off("pub(crate) use container::{ModuleDict, summary};", nn_mod).is_empty());
    assert!(off("pub use container::{ModuleList, Sequential};", nn_mod).is_empty());
    assert!(off("// pub use container::ModuleDict;", nn_mod).is_empty());
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
///
/// 本テストは多層防御の最内層（1 行単位の直列走査）であり、`src/lib.rs`
/// の `KvCacheHoldDoctestGuard`（正のプローブ doctest）・
/// `facade_does_not_reexport_or_declare_kv_cache_items`（トークン方式。
/// 複数行・別名・独自宣言を検出）と多層で保留を固定する
/// （`docs/kv-cache-design.md` §10）。workspace 全体（facade 以外の
/// クレート内部の private 宣言も含む）の名前インベントリは、facade
/// 到達可能性の保証と無関係な内部宣言まで固定してしまうため採用しない
/// （codex-review 指摘。PR #2252）。
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

/// `KvCacheHoldDoctestGuard` の唯一の doctest ブロックが glob import する
/// ネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod` 宣言集合が
/// 一致することを固定する（`nn_module_hold_doctest_globs_all_pub_modules`〈#2396 で削除済み〉・
/// `bool_ops_hold_doctest_globs_all_pub_modules` と同型。新しい `pub mod`
/// を facade へ追加した際、doctest 側の `use` 一覧の更新を機械的に強制
/// する）。
#[test]
fn kv_cache_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "KvCacheHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "KvCacheHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`kv_cache_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの **glob
/// 以外の本文**（`__FandheKvHoldMarker`・`__fandhe_kv_hold_probe` モジュール・
/// `__FandheKvHoldProbe` トレイト定義・`compat::Sequential` への実装・
/// `__probe` 関数）が固定文言 [`KV_CACHE_HOLD_PROBE_BODY`] と 1 行たりとも
/// 違わず一致することを固定する（rustdoc の `# ` 隠し行・プローブの削除・
/// 別名へのシャドーイング等で正のプローブを骨抜きにする改変を機械的に
/// 拒否する）。
#[test]
fn kv_cache_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "KvCacheHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, KV_CACHE_HOLD_PROBE_BODY,
        "KvCacheHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 KV_CACHE_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_kv_hold_probe モジュール・__FandheKvHoldProbe\
         トレイト・__probe 関数）の削除・弱体化・隠し行の混入がないか\
         確認すること。"
    );
}

/// [`kv_cache_hold_doctest_probe_body_matches_fixed_contract`] が要求
/// する固定文言。`crates/facade/src/lib.rs` の `KvCacheHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致
/// する必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に
/// 含む）。
const KV_CACHE_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_kv_hold_probe {\n\
\x20\x20\x20\x20pub struct KvCache;\n\
\x20\x20\x20\x20pub struct StatefulAttention;\n\
\x20\x20\x20\x20pub struct __FandheKvHoldMarker;\n\
\x20\x20\x20\x20pub fn add_stateful_attention() -> __FandheKvHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheKvHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_kv_hold_probe::*;\n\
\n\
trait __FandheKvHoldProbe {\n\
\x20\x20\x20\x20fn add_stateful_attention(&self) -> __FandheKvHoldMarker;\n\
}\n\
\n\
impl __FandheKvHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn add_stateful_attention(&self) -> __FandheKvHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheKvHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe(_: KvCache, _: StatefulAttention, x: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheKvHoldMarker = add_stateful_attention();\n\
\x20\x20\x20\x20let _: __FandheKvHoldMarker = fandhe_ai::compat::Sequential::add_stateful_attention(x);\n\
\x20\x20\x20\x20let _: __FandheKvHoldMarker = x.add_stateful_attention();\n\
}";

/// [`facade_does_not_reexport_or_declare_kv_cache_items`]・その自己テスト
/// が共用する検出本体。facade src 全体（`crates/facade/src/**`）の `pub use` から
/// [`collect_pub_use_leaves`] で別名にする前の葉を集め `KvCache`／
/// `StatefulAttention` を検出し（単一行・複数行・ネストした group・別名も
/// 検出）、`trait`／`struct`／`enum`／`type` 直後の `KvCache`／
/// `StatefulAttention` 独自宣言、`add_stateful_attention` の `fn` 宣言
/// （可視性・宣言文脈を問わない。[`count_fn_declarations_by_name`] と同じ
/// 検出契約）を違反として返す。
fn scan_kv_cache_reexports_and_declarations(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if matches!(leaf.as_str(), "KvCache" | "StatefulAttention") {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && matches!(
                tokens.get(i + 1).map(String::as_str),
                Some("KvCache") | Some("StatefulAttention")
            )
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    let count = count_fn_declarations_by_name(&tokens, "add_stateful_attention");
    if count > 0 {
        offending.push(format!("`fn add_stateful_attention` 宣言が {count} 件"));
    }
    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`KvCache`／
/// `StatefulAttention` を識別子単位で含む `pub use`（複数行・ネストした
/// group・別名含む）も、facade 独自の `trait`／`struct`／`enum`／`type`
/// 宣言も、`add_stateful_attention` の `fn` 宣言も存在しないことを固定
/// する（`KvCacheHoldDoctestGuard` の正のプローブと多層防御を成す最内層
/// のソース走査ガード。`facade_does_not_reexport_or_declare_bool_ops` と
/// 同型。既存の 1 行単位走査
/// `facade_does_not_expose_kv_cache_stateful_attention` の穴〈複数行
/// `pub use`・facade 独自宣言〉を塞ぐ）。
#[test]
fn facade_does_not_reexport_or_declare_kv_cache_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_kv_cache_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が KV キャッシュ（#2084 の K-2。`KvCache`／\
         `StatefulAttention`／`add_stateful_attention`）を再エクスポート、\
         独自宣言、または同名の fn を宣言している（`docs/kv-cache-design.md` \
         §6 承認事項 2 が未取得のまま対象外としている設計判断に違反）: {offending:?}"
    );
}

/// [`scan_kv_cache_reexports_and_declarations`]（[`facade_does_not_
/// reexport_or_declare_kv_cache_items`]）の自己テスト（正例・負例の合成
/// 入力）。単一行・複数行・別名・独自宣言・fn 宣言の各正例と、コメントや
/// 文字列リテラル中の出現・非公開 `use` の負例を検証する。
#[test]
fn facade_does_not_reexport_or_declare_kv_cache_items_detects_each_category() {
    // 正例: 単一行 pub use。
    assert!(
        !scan_kv_cache_reexports_and_declarations("pub use fandhe_ai_autodiff::nn::KvCache;")
            .is_empty()
    );
    // 正例: 複数行 pub use（複数行グループ再エクスポートの穴）。
    assert!(
        !scan_kv_cache_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::nn::{\n    KvCache,\n};"
        )
        .is_empty()
    );
    // 正例: ネストした group・別名。
    assert!(
        !scan_kv_cache_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::nn::{attention::{StatefulAttention as SA}};"
        )
        .is_empty()
    );
    // 正例: facade 独自宣言。
    assert!(!scan_kv_cache_reexports_and_declarations("pub struct KvCache;").is_empty());
    assert!(
        !scan_kv_cache_reexports_and_declarations("pub type StatefulAttention = u8;").is_empty()
    );
    // 正例: fn 宣言（可視性を問わない）。
    assert!(
        !scan_kv_cache_reexports_and_declarations(
            "impl Sequential { fn add_stateful_attention(&mut self) {} }"
        )
        .is_empty()
    );

    // 負例: 非公開 import。
    assert!(
        scan_kv_cache_reexports_and_declarations("use fandhe_ai_autodiff::nn::KvCache;").is_empty()
    );
    // 負例: コメント・文字列リテラル中の出現。
    assert!(
        scan_kv_cache_reexports_and_declarations(
            "// pub use fandhe_ai_autodiff::nn::KvCache;\nlet s = \"KvCache\";"
        )
        .is_empty()
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
/// # ガード方式（#2133 の `NnModuleHoldDoctestGuard`〈#2396 で削除済み〉とは別方式）
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
// #2510 で比較 6 種・`masked_select` は `Var` 委譲として承認形へ部分反転済み
// （logical 3 件・Tensor/Tape 上配置・モジュール再エクスポートは保留を維持）。
// `VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard`（#2396 で削除済み）系と同型の
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
/// doctest_probe_body_matches_fixed_contract`と同じ理由: rustdoc の
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
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Var::logical_and(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.logical_and();\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Var::logical_or(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.logical_or();\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = fandhe_ai::Var::logical_not(x);\n\
\x20\x20\x20\x20let _: __FandheBoolMarker = x.logical_not();\n\
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
/// [`workspace_declares_bool_ops_fn_names_in_approved_places_only`]
/// が共用する。#2510 で 7 件は `Var` 委譲として承認済みのため、承認分
/// （[`BOOL_OPS_APPROVED_VAR_METHOD_NAMES`]）と保留分
/// （[`BOOL_OPS_HELD_FN_NAMES`]）の連結として定義する。
const BOOL_OPS_FN_NAMES: [&str; 10] = [
    "gt_bool",
    "ge_bool",
    "lt_bool",
    "le_bool",
    "eq_bool",
    "ne_bool",
    "masked_select",
    "logical_and",
    "logical_or",
    "logical_not",
];

/// `Var` の委譲メソッドとして facade 公開が承認された 7 件（イシュー
/// #2510・ルート #2499 一括承認・`docs/autodiff-bool-ops-exposure-
/// decision.md` §6）。
const BOOL_OPS_APPROVED_VAR_METHOD_NAMES: [&str; 7] = [
    "gt_bool",
    "ge_bool",
    "lt_bool",
    "le_bool",
    "eq_bool",
    "ne_bool",
    "masked_select",
];

/// 公開形が未決で保留のままの logical 3 件（#2594）。
const BOOL_OPS_HELD_FN_NAMES: [&str; 3] = ["logical_and", "logical_or", "logical_not"];

/// `crates/autodiff/src/var.rs` の委譲メソッド本体の固定（トークンを
/// 空白連結した形。`determinism_fn_body` の出力と照合する）。独自実装化
/// や別関数への迂回（`checked_bytes_for` 等の事前検査の迂回を含む）を
/// 拒否するため、本体は `crate::bool_ops::<name>(self, <arg>)` 1 式のみ。
const BOOL_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 7] = [
    ("gt_bool", "crate : : bool_ops : : gt_bool ( self , other )"),
    ("ge_bool", "crate : : bool_ops : : ge_bool ( self , other )"),
    ("lt_bool", "crate : : bool_ops : : lt_bool ( self , other )"),
    ("le_bool", "crate : : bool_ops : : le_bool ( self , other )"),
    ("eq_bool", "crate : : bool_ops : : eq_bool ( self , other )"),
    ("ne_bool", "crate : : bool_ops : : ne_bool ( self , other )"),
    (
        "masked_select",
        "crate : : bool_ops : : masked_select ( self , mask )",
    ),
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
        "facade の公開面が承認形（#2510。`Var` 委譲 7 件のみ。実体は autodiff\
         側）の外で bool_ops を再エクスポート、または同名の fn を facade 側で\
         宣言している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`BOOL_OPS_FN_NAMES`]
/// （10 個）の `fn` 宣言の定義元を固定する（#2510 で正ガードへ反転）。
/// 承認済み 7 件は `crates/autodiff/src/bool_ops.rs`（自由関数）と
/// `crates/autodiff/src/var.rs`（`Var` 委譲メソッド）に各 1 件、保留の
/// logical 3 件は `bool_ops.rs` のみ 1 件。過不足はいずれも fail-closed。
/// さらに `var.rs` の 7 委譲メソッドの本体が
/// [`BOOL_OPS_VAR_EXPECTED_BODIES`] と一致すること（`bool_ops` 自由関数
/// への 1 式委譲のまま）を固定する（`workspace_declares_custom_fn_only_
/// on_tape` と同型の workspace 全体インベントリ）。
#[test]
fn workspace_declares_bool_ops_fn_names_in_approved_places_only() {
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

    let mut var_rs_tokens: Option<Vec<String>> = None;
    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in BOOL_OPS_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
            if rel == "autodiff/src/var.rs" {
                var_rs_tokens = Some(tokens);
            }
        });
    }

    let mut expected: std::collections::BTreeMap<String, usize> = BOOL_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/bool_ops.rs::{name}"), 1usize))
        .collect();
    for name in BOOL_OPS_APPROVED_VAR_METHOD_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の bool_ops 系 `fn` 宣言集合が\
         期待（bool_ops.rs に 10 件・承認済み 7 件は var.rs にも各 1 件。\
         logical 3 件は bool_ops.rs のみ）と一致しない（過不足いずれも\
         fail-closed に検出する。新たな定義元が見つかった場合、それが\
         承認済みの実装なのか迂回経路の混入なのかを確認すること）: {found:?}"
    );

    let var_tokens = var_rs_tokens.expect("crates/autodiff/src/var.rs が走査されなかった");
    for (name, expected_body) in BOOL_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&var_tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（bool_ops 自由関数への 1 式\
             委譲）と一致しない"
        );
    }
    assert_eq!(
        BOOL_OPS_VAR_EXPECTED_BODIES.len(),
        BOOL_OPS_APPROVED_VAR_METHOD_NAMES.len()
    );
    let _ = BOOL_OPS_HELD_FN_NAMES;
}

// =====================================================================
// #2143 実装・#2511 公開（親 #2500・ルート #2499）の正ガード。旧否定ガード
// （`VarRearrangeOpsHoldDoctestGuard`・対応するソース走査 4 件）を、承認形
// （`Var::flip`／`roll`／`repeat`／`tile` の 1 行委譲）だけを許す形へ反転した
// （先例 #2198・#2338・#2507・#2510）。承認事項は
// `docs/autodiff-rearrange-ops-decision.md` §6 参照。
// =====================================================================

/// `repeat`・`tile`・`flip`・`roll`（4 個の関数名。イシュー #2143）。
/// [`facade_does_not_reexport_or_declare_rearrange_ops`]・
/// [`workspace_declares_rearrange_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const REARRANGE_OPS_FN_NAMES: [&str; 4] = ["repeat", "tile", "flip", "roll"];

/// facade src 全体（`crates/facade/src/**`）に、`rearrange_ops` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::rearrange_ops;` 等の
/// モジュール再エクスポート・別名含む）も、[`REARRANGE_OPS_FN_NAMES`]
/// （4 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する。承認形は `Var` の inherent 委譲メソッド（`autodiff` 側）のみで、
/// facade でのモジュール再エクスポート・別名・独自 `fn` 宣言は承認形に
/// 含まれないため引き続き拒否する（#2511 で保留ガードから正ガードの一部へ
/// 位置づけを変更。検査ロジックは不変）。
#[test]
fn facade_does_not_reexport_or_declare_rearrange_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "rearrange_ops")
            {
                offending.push(format!(
                    "{}: `{trimmed}` が `rearrange_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in REARRANGE_OPS_FN_NAMES {
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
        "facade が rearrange_ops を再エクスポート、または同名の fn を宣言している\
         （承認形は autodiff の `Var` 委譲メソッドのみ。イシュー #2511）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`REARRANGE_OPS_FN_NAMES`]
/// （4 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// bool_ops_fn_names_in_approved_places_only` と同型のインベントリ）。
///
/// **期待集合は `crates/autodiff/src/rearrange_ops.rs`（各 1 件）・
/// `crates/autodiff/src/var.rs`（`Var` 委譲メソッド。各 1 件。#2511）に加え、
/// `crates/backend-cuda/src/gemm.rs::tile`（GEMM タイル設定の無関係な
/// inherent メソッド。2 件）を明示的に含める**。`tile` という名前が
/// GEMM 側で既に使われているため、bool_ops 系と異なり衝突が実在する
/// （`backend-cuda` の `tile` は facade から到達できない private／
/// crate 内 API のため facade 公開面には影響しない。実装計画「設計
/// 判断」§2.2・着手前確認の再 grep 結果で確定させた）。
#[test]
fn workspace_declares_rearrange_ops_fn_names_only_in_approved_locations() {
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
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in REARRANGE_OPS_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }

    let mut expected: std::collections::BTreeMap<String, usize> = REARRANGE_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/rearrange_ops.rs::{name}"), 1usize))
        .collect();
    for name in REARRANGE_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }
    expected.insert("backend-cuda/src/gemm.rs::tile".to_string(), 2);

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の rearrange_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/rearrange_ops.rs` 各 1 件 +\
         `autodiff/src/var.rs` 各 1 件 + `backend-cuda/src/gemm.rs::tile` 2 件）と一致しない（過不足\
         いずれも fail-closed に検出する。新たな定義元が見つかった場合、\
         それが承認済みの実装なのか迂回経路の混入なのかを確認すること）: \
         {found:?}"
    );
}

/// `var.rs` の 4 委譲メソッド本体の承認形（`rearrange_ops` 自由関数への
/// 1 行委譲。引数名も固定）。独自実装・スタブへのすり替えと境界検査の
/// 迂回を拒否する（#2511。[`BOOL_OPS_VAR_EXPECTED_BODIES`] と同型）。
const REARRANGE_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 4] = [
    ("flip", "crate : : rearrange_ops : : flip ( self , dims )"),
    (
        "roll",
        "crate : : rearrange_ops : : roll ( self , shifts , dims )",
    ),
    (
        "repeat",
        "crate : : rearrange_ops : : repeat ( self , repeats )",
    ),
    ("tile", "crate : : rearrange_ops : : tile ( self , reps )"),
];

/// `var.rs` の 4 委譲メソッドの本体が [`REARRANGE_OPS_VAR_EXPECTED_BODIES`]
/// と一致することを固定する（本体抽出は名前非依存の
/// [`determinism_fn_body`] を再利用。`pub fn` がちょうど 1 件のとき本体を返す）。
#[test]
fn var_rearrange_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in REARRANGE_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（rearrange_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけ（`fandhe_ai_autodiff` を import
/// しない）で 4 メソッドへ到達でき、シグネチャが承認形と一致し、実際に
/// 適用して期待値が得られることを固定する（スタブでは通らない。#2511）。
#[test]
fn var_rearrange_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    fn sig_flip<'t>() -> fn(&Var<'t>, &[usize]) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::flip
    }
    type RollSig<'t> = fn(&Var<'t>, &[isize], &[usize]) -> Result<Var<'t>, AutodiffError>;
    fn sig_roll<'t>() -> RollSig<'t> {
        Var::<'t>::roll
    }
    fn sig_repeat<'t>() -> fn(&Var<'t>, &[usize]) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::repeat
    }
    fn sig_tile<'t>() -> fn(&Var<'t>, &[usize]) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::tile
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0], &[3]).expect("tensor"));
    assert_eq!(vals(&sig_flip()(&x, &[0]).expect("flip")), [3.0, 2.0, 1.0]);
    assert_eq!(
        vals(&sig_roll()(&x, &[1], &[0]).expect("roll")),
        [3.0, 1.0, 2.0]
    );
    assert_eq!(
        vals(&sig_repeat()(&x, &[2]).expect("repeat")),
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        vals(&sig_tile()(&x, &[2]).expect("tile")),
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );
}

// =====================================================================
// #2145 実装・#2512 公開の正ガード。旧否定ガード（`VarScalarUnaryOpsHoldDoctestGuard`・
// ソース走査 4 件）を、承認形（`Var` の 1 行委譲メソッドのみ）だけを許す形へ反転した
// （先例 #2198・#2338・#2510・#2511）。承認事項は
// `docs/autodiff-scalar-unary-ops-decision.md` §9 参照。
// =====================================================================

/// `floor`・`ceil`・`round`・`sign`・`reciprocal`・`rsqrt`・`erf`・
/// `pow_scalar`（8 個の関数名。イシュー #2145）。
/// [`facade_does_not_reexport_or_declare_scalar_unary_ops`]・
/// [`workspace_declares_scalar_unary_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const SCALAR_UNARY_OPS_FN_NAMES: [&str; 8] = [
    "floor",
    "ceil",
    "round",
    "sign",
    "reciprocal",
    "rsqrt",
    "erf",
    "pow_scalar",
];

/// facade src 全体（`crates/facade/src/**`）に、`scalar_unary_ops` を
/// 参照する `pub use`（`pub use fandhe_ai_autodiff::scalar_unary_ops;`
/// 等のモジュール再エクスポート・別名含む）も、[`SCALAR_UNARY_OPS_FN_NAMES`]
/// （8 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する（承認形は autodiff の `Var` 委譲メソッドのみ。モジュール再
/// エクスポートと facade 側の独自宣言を拒否する。
/// `facade_does_not_reexport_or_declare_rearrange_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_scalar_unary_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use")
                && line_contains_identifier(trimmed, "scalar_unary_ops")
            {
                offending.push(format!(
                    "{}: `{trimmed}` が `scalar_unary_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in SCALAR_UNARY_OPS_FN_NAMES {
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
        "facade の公開面が scalar_unary_ops を再エクスポート、または同名の fn を\
         宣言している（承認形は autodiff の `Var` 委譲メソッドのみ。\
         イシュー #2512）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`SCALAR_UNARY_OPS_FN_NAMES`]（8 個）の `fn` 宣言の定義元集合を固定
/// する（`workspace_declares_rearrange_ops_fn_names_only_in_approved_
/// locations`（#2511 で改名。旧名 `..._in_autodiff_rearrange_ops`）と同型のインベントリ）。
///
/// **期待集合は `crates/autodiff/src/scalar_unary_ops.rs`（各 1 件）に
/// 加え、`crates/onnx-interop/src/ops/activation.rs::erf`（ONNX `Erf`
/// 演算子の無関係な実装。1 件）を明示的に含める**（実装計画「着手前
/// 確認」の再 grep 結果で確定させた。他 7 名の `fn` 宣言は workspace 内
/// に存在しない）。
#[test]
fn workspace_declares_scalar_unary_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in SCALAR_UNARY_OPS_FN_NAMES {
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

    let mut expected: std::collections::BTreeMap<String, usize> = SCALAR_UNARY_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/scalar_unary_ops.rs::{name}"), 1usize))
        .collect();
    for name in SCALAR_UNARY_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }
    expected.insert("onnx-interop/src/ops/activation.rs::erf".to_string(), 1);

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の scalar_unary_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/scalar_unary_ops.rs` 各 1 件 +\
         `autodiff/src/var.rs` 各 1 件 + `onnx-interop/src/ops/activation.rs::erf` 1 件）と一致しない（過不足\
         いずれも fail-closed に検出する。新たな定義元が見つかった場合、\
         それが承認済みの実装なのか迂回経路の混入なのかを確認すること）: \
         {found:?}"
    );
}

/// `var.rs` の 8 委譲メソッド本体の承認形（`scalar_unary_ops` 自由関数への
/// 1 行委譲。引数名も固定）。独自実装・スタブへのすり替えを拒否する
/// （#2512。[`REARRANGE_OPS_VAR_EXPECTED_BODIES`] と同型）。
const SCALAR_UNARY_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 8] = [
    ("floor", "crate : : scalar_unary_ops : : floor ( self )"),
    ("ceil", "crate : : scalar_unary_ops : : ceil ( self )"),
    ("round", "crate : : scalar_unary_ops : : round ( self )"),
    ("sign", "crate : : scalar_unary_ops : : sign ( self )"),
    (
        "reciprocal",
        "crate : : scalar_unary_ops : : reciprocal ( self )",
    ),
    ("rsqrt", "crate : : scalar_unary_ops : : rsqrt ( self )"),
    ("erf", "crate : : scalar_unary_ops : : erf ( self )"),
    (
        "pow_scalar",
        "crate : : scalar_unary_ops : : pow_scalar ( self , exponent )",
    ),
];

/// `var.rs` の 8 委譲メソッドの本体が
/// [`SCALAR_UNARY_OPS_VAR_EXPECTED_BODIES`] と一致することを固定する。
#[test]
fn var_scalar_unary_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in SCALAR_UNARY_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（scalar_unary_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで 8 メソッドへ到達でき、シグネチャが
/// 承認形と一致し、実際に適用して期待値が得られることを固定する
/// （スタブでは通らない。#2512）。
#[test]
fn var_scalar_unary_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type UnarySig<'t> = fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>;
    type PowSig<'t> = fn(&Var<'t>, f32) -> Result<Var<'t>, AutodiffError>;
    fn sigs<'t>() -> [UnarySig<'t>; 7] {
        [
            Var::<'t>::floor,
            Var::<'t>::ceil,
            Var::<'t>::round,
            Var::<'t>::sign,
            Var::<'t>::reciprocal,
            Var::<'t>::rsqrt,
            Var::<'t>::erf,
        ]
    }
    fn sig_pow<'t>() -> PowSig<'t> {
        Var::<'t>::pow_scalar
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![2.5_f32, -1.5, 4.0], &[3]).expect("tensor"));
    let [floor, ceil, round, sign, reciprocal, rsqrt, _erf] = sigs();
    assert_eq!(vals(&floor(&x).expect("floor")), [2.0, -2.0, 4.0]);
    assert_eq!(vals(&ceil(&x).expect("ceil")), [3.0, -1.0, 4.0]);
    assert_eq!(vals(&round(&x).expect("round")), [2.0, -2.0, 4.0]);
    assert_eq!(vals(&sign(&x).expect("sign")), [1.0, -1.0, 1.0]);
    assert_eq!(vals(&reciprocal(&x).expect("reciprocal"))[2], 0.25);
    assert_eq!(vals(&rsqrt(&x).expect("rsqrt"))[2], 0.5);
    assert_eq!(
        vals(&sig_pow()(&x, 2.0).expect("pow_scalar")),
        [6.25, 2.25, 16.0]
    );
}

/// `var.rs` の reduce_ops 5 委譲メソッド本体の承認形（`reduce_ops` 自由関数への
/// 1 行委譲。引数名も固定。#2514）。
const REDUCE_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 5] = [
    ("prod", "crate : : reduce_ops : : prod ( self , dim )"),
    (
        "logsumexp",
        "crate : : reduce_ops : : logsumexp ( self , dim )",
    ),
    ("any", "crate : : reduce_ops : : any ( self , dim )"),
    ("all", "crate : : reduce_ops : : all ( self , dim )"),
    (
        "norm_p",
        "crate : : reduce_ops : : norm_p ( self , p , dim )",
    ),
];

/// `var.rs` の reduce_ops 5 委譲メソッドの本体が
/// [`REDUCE_OPS_VAR_EXPECTED_BODIES`] と一致することを固定する。
#[test]
fn var_reduce_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in REDUCE_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（reduce_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで reduce_ops 5 メソッドへ到達でき、
/// シグネチャが承認形と一致し、実際に適用して期待値が得られることを固定する
/// （スタブでは通らない。#2514）。
#[test]
fn var_reduce_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type DimSig<'t> = fn(&Var<'t>, Option<usize>) -> Result<Var<'t>, AutodiffError>;
    type NormSig<'t> = fn(&Var<'t>, f32, Option<usize>) -> Result<Var<'t>, AutodiffError>;
    fn sigs<'t>() -> [DimSig<'t>; 4] {
        [
            Var::<'t>::prod,
            Var::<'t>::logsumexp,
            Var::<'t>::any,
            Var::<'t>::all,
        ]
    }
    fn sig_norm<'t>() -> NormSig<'t> {
        Var::<'t>::norm_p
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], &[4]).expect("tensor"));
    let [prod, logsumexp, any, all] = sigs();
    assert_eq!(vals(&prod(&x, None).expect("prod")), [24.0]);
    assert_eq!(vals(&any(&x, None).expect("any")), [1.0]);
    assert_eq!(vals(&all(&x, None).expect("all")), [1.0]);
    let lse = vals(&logsumexp(&x, None).expect("logsumexp"))[0];
    assert!((lse - 4.440_19).abs() < 1e-4, "lse={lse}");
    let y = tape.var(&Tensor::new(vec![3.0_f32, 4.0], &[2]).expect("tensor"));
    assert_eq!(vals(&sig_norm()(&y, 2.0, None).expect("norm_p")), [5.0]);
}

/// `var.rs` の extremum_ops 2 委譲メソッド本体の承認形（#2514）。
const EXTREMUM_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 2] = [
    ("amax", "crate : : extremum_ops : : amax ( self , dim )"),
    ("amin", "crate : : extremum_ops : : amin ( self , dim )"),
];

/// `var.rs` の extremum_ops 2 委譲メソッドの本体が
/// [`EXTREMUM_OPS_VAR_EXPECTED_BODIES`] と一致することを固定する。
#[test]
fn var_extremum_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in EXTREMUM_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（extremum_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで `amax`／`amin` へ到達でき、シグネチャが
/// 承認形と一致し、実際に適用して期待値が得られることを固定する（#2514）。
#[test]
fn var_extremum_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type DimSig<'t> = fn(&Var<'t>, Option<usize>) -> Result<Var<'t>, AutodiffError>;
    fn sigs<'t>() -> [DimSig<'t>; 2] {
        [Var::<'t>::amax, Var::<'t>::amin]
    }
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![3.0_f32, 1.0, 3.0, -2.0], &[4]).expect("tensor"));
    let [amax, amin] = sigs();
    assert_eq!(
        amax(&x, None)
            .expect("amax")
            .to_tensor()
            .host_slice()
            .into_owned(),
        [3.0]
    );
    assert_eq!(
        amin(&x, None)
            .expect("amin")
            .to_tensor()
            .host_slice()
            .into_owned(),
        [-2.0]
    );
}

// =====================================================================
// #2139（親 #2138・#2131）の facade 公開保留固定（`VarHooksHoldDoctestGuard`）。
// `VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard`（#2396 で削除済み）／
// `VarBoolOpsHoldDoctestGuard` 系と同型の正のプローブ 1 ブロック方式の
// ドリフト検査に加え、workspace 全体のソース走査による定義元インベント
// リを持つ。承認事項・多層防御の位置づけは
// `docs/autodiff-forward-backward-hooks-design.md` §13 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `VarHooksHoldDoctestGuard` doc 内の
/// 唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （[`custom_function_hold_doctest_globs_all_pub_modules`] の
/// `VarHooksHoldDoctestGuard` 版）。
#[test]
fn hooks_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarHooksHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "VarHooksHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`hooks_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの **glob
/// 以外の本文**（型・モジュール名の衝突プローブ `__fandhe_hooks_hold_probe`・
/// メソッド名の衝突プローブ `__FandheHooksHoldProbe` トレイト定義・
/// `Var`／`Tape`／`compat::Sequential` への実装・`__probe_*` 関数群）が
/// 固定文言 [`HOOKS_HOLD_PROBE_BODY`] と 1 行たりとも違わず一致すること
/// を固定する（`bool_ops_hold_doctest_probe_body_matches_fixed_contract`
/// と同じ理由: rustdoc の `# ` 隠し行・プローブの削除・別名への
/// シャドーイング等で正のプローブを骨抜きにする改変を機械的に拒否する）。
#[test]
fn hooks_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarHooksHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, HOOKS_HOLD_PROBE_BODY,
        "VarHooksHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 HOOKS_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_hooks_hold_probe モジュール・\
         __FandheHooksHoldProbe トレイト・__probe_* 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`hooks_hold_doctest_probe_body_matches_fixed_contract`] が要求
/// する固定文言。`crates/facade/src/lib.rs` の `VarHooksHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致
/// する必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に
/// 含む）。
const HOOKS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_hooks_hold_probe {\n\
\x20\x20\x20\x20pub struct HookHandle;\n\
\x20\x20\x20\x20pub struct ForwardHooked;\n\
\x20\x20\x20\x20pub struct ForwardHookCtx;\n\
\x20\x20\x20\x20pub mod hooks {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn __probe() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_hooks_hold_probe::*;\n\
\n\
fn __probe_types(_: HookHandle, _: ForwardHooked, _: ForwardHookCtx) {\n\
\x20\x20\x20\x20hooks::__probe();\n\
}\n\
\n\
struct __FandheHooksMarker;\n\
\n\
trait __FandheHooksHoldProbe {\n\
\x20\x20\x20\x20fn register_forward_hook(&self) -> __FandheHooksMarker;\n\
\x20\x20\x20\x20fn register_backward_hook(&self) -> __FandheHooksMarker;\n\
\x20\x20\x20\x20fn register_hook(&self) -> __FandheHooksMarker;\n\
\x20\x20\x20\x20fn remove_hook(&self) -> __FandheHooksMarker;\n\
\x20\x20\x20\x20fn remove_backward_hook(&self) -> __FandheHooksMarker;\n\
}\n\
\n\
impl<'t> __FandheHooksHoldProbe for fandhe_ai::Var<'t> {\n\
\x20\x20\x20\x20fn register_forward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn register_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn register_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn remove_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn remove_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
}\n\
\n\
impl __FandheHooksHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn register_forward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn register_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn register_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn remove_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn remove_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
}\n\
\n\
impl __FandheHooksHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn register_forward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn register_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn register_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn remove_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
\x20\x20\x20\x20fn remove_backward_hook(&self) -> __FandheHooksMarker { __FandheHooksMarker }\n\
}\n\
\n\
fn __probe_var(x: &fandhe_ai::Var<'_>) {\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Var::register_forward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_forward_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Var::register_backward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_backward_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Var::register_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Var::remove_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.remove_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Var::remove_backward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.remove_backward_hook();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Tape::register_forward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_forward_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Tape::register_backward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_backward_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Tape::register_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Tape::remove_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.remove_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::Tape::remove_backward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.remove_backward_hook();\n\
}\n\
\n\
fn __probe_sequential(x: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::register_forward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_forward_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::register_backward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_backward_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::register_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.register_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::remove_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.remove_hook();\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = fandhe_ai::compat::Sequential::remove_backward_hook(x);\n\
\x20\x20\x20\x20let _: __FandheHooksMarker = x.remove_backward_hook();\n\
}";

/// [`workspace_declares_no_hook_registration_fns`] が検査する 4 つの
/// 関数名。`register_hook` はあえて含めない（並行する #2182 の DataLoader
/// transform フック等、正当な用途で使われうる汎用名のため。facade から
/// 到達できないことは `VarHooksHoldDoctestGuard` の doctest が固定する）。
///
/// **`register_hook` の除外は無条件ではない**（codex-review 指摘・PR #2254・
/// discussion_r4096728116）。facade の `Tape`（`pub struct Tape(pub(crate)
/// fandhe_ai_autodiff::Tape)`）は `Deref` を持たない newtype のため、doctest
/// 正のプローブ（[`hooks_hold_doctest_probe_body_matches_fixed_contract`]）は
/// facade に再エクスポートされたメソッドしか検出できない。`register_hook` を
/// この配列から一律除外したままだと、`fandhe_ai_autodiff::Tape::
/// register_hook`（facade を経由しない autodiff 側の本体実装）が doctest・
/// 本走査のいずれからも検出されず、§11 承認前の本体実装を fail-closed に
/// 止めるという受入ガードの前提が崩れる。この穴は
/// [`autodiff_declares_no_register_hook_fn`] が
/// `crates/autodiff/src/` 限定で `register_hook` の定義元を明示的に
/// allowlist 化（= 0 件固定）することで塞ぐ（`crates/autodiff` 以外の
/// クレート、たとえば #2182 の DataLoader transform フックでの
/// `register_hook` という名称の使用は本配列・本 workspace 全体走査の
/// 対象外のまま許容する）。
const HOOK_REGISTRATION_FN_NAMES: [&str; 4] = [
    "register_forward_hook",
    "register_backward_hook",
    "remove_hook",
    "remove_backward_hook",
];

/// [`HOOK_REGISTRATION_FN_NAMES`] が `register_hook` を意図的に除外している
/// 穴（doctest 正のプローブは facade 経由の到達可能性しか見ず、facade
/// `Tape` newtype は autodiff `Tape` を `Deref` しないため autodiff 側の
/// 本体実装を検出できない）を塞ぐ、`crates/autodiff/src/` 限定の否定ガード
/// （codex-review 指摘・PR #2254・discussion_r4096728116。`docs/
/// autodiff-forward-backward-hooks-design.md` §13.3 の「正当な定義元だけを
/// 明示的に allowlist 化する」対応）。`Var`／`Tape`／`nn::Sequential`（PyTorch
/// 互換 API が実装候補として想定する型。§11 承認事項 4 の受入基準改訂で
/// 変わりうる）はいずれも `crates/autodiff/src/` 配下に定義されているため、
/// このクレート限定で `register_hook` の `fn` 宣言が 0 件であることを固定
/// すれば、正当な定義元（本 PR 時点で存在しない）が承認前に紛れ込むことを
/// 宣言文脈・可視性を問わず検出できる。`crates/autodiff` の外（#2182 の
/// DataLoader transform フック等）での同名関数の使用は引き続き許容する
/// （[`HOOK_REGISTRATION_FN_NAMES`] のコメント参照）。
#[test]
fn autodiff_declares_no_register_hook_fn() {
    let crates_dir = workspace_crates_dir();
    let autodiff_src_dir = crates_dir.join("autodiff").join("src");
    assert!(
        autodiff_src_dir.is_dir(),
        "crates/autodiff/src ディレクトリが見つからない（テスト自体が検査対象を\
         見失っている可能性がある）: {}",
        autodiff_src_dir.display()
    );

    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    visit_rs_files(&autodiff_src_dir, &mut |path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let count = count_fn_declarations_by_name(&tokens, "register_hook");
        if count > 0 {
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            *found.entry(rel.to_string()).or_insert(0) += count;
        }
    });

    assert!(
        found.is_empty(),
        "crates/autodiff/src/ に register_hook の `fn` 宣言が見つかった\
         （イシュー #2139 は §11 の承認事項 5 項目がそろうまで着手不可という\
         設計判断〈docs/autodiff-forward-backward-hooks-design.md §13〉に\
         違反する可能性がある。承認済みの実装であれば本ガード自体を撤去\
         すること）: {found:?}"
    );
}

/// [`autodiff_declares_no_register_hook_fn`] が使う
/// [`count_fn_declarations_by_name`] が `register_hook` を実際に検出できる
/// ことを固定する合成入力の自己テスト（
/// [`count_fn_declarations_by_name_detects_hook_registration_fn_names`] は
/// [`HOOK_REGISTRATION_FN_NAMES`] の 4 関数名のみを対象とし `register_hook`
/// を含まないため、検出器自体が `register_hook` を検出できることは別途
/// 固定する必要がある）。
#[test]
fn count_fn_declarations_by_name_detects_register_hook() {
    let src = "impl Var { pub fn register_hook(&self) {} }";
    let cleaned: String = strip_comments_and_literals(src).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(
        count_fn_declarations_by_name(&tokens, "register_hook"),
        1,
        "src={src:?} tokens={tokens:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`HOOK_REGISTRATION_FN_NAMES`]（4 個）の `fn` 宣言が可視性・宣言文脈
/// を問わず 1 件も存在しないことを固定する（`workspace_declares_custom_
/// fn_only_on_tape`・`workspace_declares_bool_ops_fn_names_in_
/// approved_places_only` と同型の workspace 全体インベントリだが、本イシュー
/// #2139 は本体実装自体が承認待ちのため「唯一の定義元」ではなく「0 件」
/// を期待値とする点が異なる）。facade のソース走査・
/// `VarHooksHoldDoctestGuard` の正のプローブはいずれも「facade から到達
/// 可能か」しか見ないため、facade の外（`autodiff`・`backend-*`・
/// `onnx-interop` 等）に同名の関数が新設された場合に備え、そもそもの
/// 定義自体の不在を固定する多層防御の最内層とする。
#[test]
fn workspace_declares_no_hook_registration_fns() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    let entries = std::fs::read_dir(&crates_dir).unwrap_or_else(|e| {
        panic!(
            "workspace crates ディレクトリが読めない: {}: {e}",
            crates_dir.display()
        )
    });
    // 個別エントリの列挙エラーも `visit_rs_files`（本ファイル関数 doc
    // 参照）と同じ理由で `flatten()` により握り潰さず、fail-closed に
    // `panic!` で伝播する（codex-review 指摘・PR #2254）。
    let mut crate_dirs: Vec<std::path::PathBuf> = entries
        .map(|e| {
            e.unwrap_or_else(|err| {
                panic!("{} 配下のエントリ列挙に失敗: {err}", crates_dir.display())
            })
        })
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
            for fn_name in HOOK_REGISTRATION_FN_NAMES {
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

    assert!(
        found.is_empty(),
        "workspace 全体（crates/*/src/）に register_forward_hook／\
         register_backward_hook／remove_hook／remove_backward_hook の \
         `fn` 宣言が見つかった\
         （イシュー #2139 は §11 の承認事項 5 項目がそろうまで着手不可\
         という設計判断〈docs/autodiff-forward-backward-hooks-design.md\
         §13〉に違反する可能性がある。承認済みの実装であれば本ガード\
         自体を撤去すること）: {found:?}"
    );
}

/// [`workspace_declares_no_hook_registration_fns`] が使う
/// [`count_fn_declarations_by_name`] が、対象 4 関数名を実際に検出
/// できることを固定する合成入力の自己テスト（検出器自体が機能して
/// いなければ、前者の「0 件」判定が空合格になり得るため）。
#[test]
fn count_fn_declarations_by_name_detects_hook_registration_fn_names() {
    for fn_name in HOOK_REGISTRATION_FN_NAMES {
        let src = format!("impl Var {{ pub fn {fn_name}(&self) {{}} }}");
        let cleaned: String = strip_comments_and_literals(&src).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        assert_eq!(
            count_fn_declarations_by_name(&tokens, fn_name),
            1,
            "src={src:?} tokens={tokens:?}"
        );
    }
}

// =====================================================================
// #2144 実装・#2513 公開（親 #2500・ルート #2499）の正ガード。旧否定ガード
// （`VarMatrixOpsHoldDoctestGuard`・対応するソース走査 4 件）を、承認形
// （`Var::tril`／`triu`／`diag`／`trace`／`outer`／`dot` の 1 行委譲）だけを
// 許す形へ反転した（先例 #2198・#2338・#2511）。承認事項は
// `docs/autodiff-matrix-ops-decision.md` §6 参照。
// =====================================================================

/// `tril`・`triu`・`diag`・`trace`・`outer`・`dot`（6 個の関数名。
/// イシュー #2144）。[`facade_does_not_reexport_or_declare_matrix_ops`]・
/// [`workspace_declares_matrix_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const MATRIX_OPS_FN_NAMES: [&str; 6] = ["tril", "triu", "diag", "trace", "outer", "dot"];

/// facade src 全体（`crates/facade/src/**`）に、`matrix_ops` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::matrix_ops;` 等の
/// モジュール再エクスポート・別名含む）も、[`MATRIX_OPS_FN_NAMES`]
/// （6 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する。承認形は `Var` の inherent 委譲メソッド（`autodiff` 側）のみで、
/// facade でのモジュール再エクスポート・別名・独自 `fn` 宣言は承認形に
/// 含まれないため引き続き拒否する（#2513 で保留ガードから正ガードの一部へ
/// 位置づけを変更。検査ロジックは不変）。
#[test]
fn facade_does_not_reexport_or_declare_matrix_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "matrix_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `matrix_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in MATRIX_OPS_FN_NAMES {
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
        "facade が matrix_ops を再エクスポート、または同名の fn を宣言している\
         （承認形は autodiff の `Var` 委譲メソッドのみ。イシュー #2513）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`MATRIX_OPS_FN_NAMES`]
/// （6 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// rearrange_ops_fn_names_only_in_approved_locations`（#2511 で改名）と同型の
/// インベントリ）。
///
/// **期待集合は `crates/autodiff/src/matrix_ops.rs`（各 1 件）・
/// `crates/autodiff/src/var.rs`（`Var` 委譲メソッド。各 1 件。#2513）**
/// （他クレートとの衝突は見つからなかった）。
#[test]
fn workspace_declares_matrix_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in MATRIX_OPS_FN_NAMES {
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

    let mut expected: std::collections::BTreeMap<String, usize> = MATRIX_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/matrix_ops.rs::{name}"), 1usize))
        .collect();
    for name in MATRIX_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の matrix_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/matrix_ops.rs` 各 1 件 +\
         `crates/autodiff/src/var.rs` 各 1 件）と\
         一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 6 委譲メソッド本体の承認形（`matrix_ops` 自由関数への
/// 1 行委譲。引数名も固定）。独自実装・スタブへのすり替えと境界検査の
/// 迂回を拒否する（#2513。[`REARRANGE_OPS_VAR_EXPECTED_BODIES`] と同型）。
const MATRIX_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 6] = [
    ("tril", "crate : : matrix_ops : : tril ( self , diagonal )"),
    ("triu", "crate : : matrix_ops : : triu ( self , diagonal )"),
    ("diag", "crate : : matrix_ops : : diag ( self , diagonal )"),
    ("trace", "crate : : matrix_ops : : trace ( self )"),
    ("outer", "crate : : matrix_ops : : outer ( self , other )"),
    ("dot", "crate : : matrix_ops : : dot ( self , other )"),
];

/// `var.rs` の 6 委譲メソッドの本体が [`MATRIX_OPS_VAR_EXPECTED_BODIES`]
/// と一致することを固定する（本体抽出は [`determinism_fn_body`] を再利用）。
#[test]
fn var_matrix_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in MATRIX_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（matrix_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけ（`fandhe_ai_autodiff` を import
/// しない）で 6 メソッドへ到達でき、シグネチャが承認形と一致し、実際に
/// 適用して期待値が得られることを固定する（スタブでは通らない。#2513）。
#[test]
fn var_matrix_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type DiagSig<'t> = fn(&Var<'t>, isize) -> Result<Var<'t>, AutodiffError>;
    type TraceSig<'t> = fn(&Var<'t>) -> Result<Var<'t>, AutodiffError>;
    type PairSig<'t> = fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, AutodiffError>;
    fn sig_tril<'t>() -> DiagSig<'t> {
        Var::<'t>::tril
    }
    fn sig_triu<'t>() -> DiagSig<'t> {
        Var::<'t>::triu
    }
    fn sig_diag<'t>() -> DiagSig<'t> {
        Var::<'t>::diag
    }
    fn sig_trace<'t>() -> TraceSig<'t> {
        Var::<'t>::trace
    }
    fn sig_outer<'t>() -> PairSig<'t> {
        Var::<'t>::outer
    }
    fn sig_dot<'t>() -> PairSig<'t> {
        Var::<'t>::dot
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let data: Vec<f32> = (1..=9).map(|v| v as f32).collect();
    let m = tape.var(&Tensor::new(data, &[3, 3]).expect("tensor"));
    assert_eq!(
        vals(&sig_tril()(&m, 0).expect("tril")),
        [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
    );
    assert_eq!(
        vals(&sig_triu()(&m, 0).expect("triu")),
        [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
    );
    assert_eq!(vals(&sig_diag()(&m, 0).expect("diag")), [1.0, 5.0, 9.0]);
    assert_eq!(vals(&sig_trace()(&m).expect("trace")), [15.0]);

    let a = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0], &[3]).expect("tensor"));
    let b = tape.var(&Tensor::new(vec![4.0_f32, 5.0, 6.0], &[3]).expect("tensor"));
    assert_eq!(vals(&sig_dot()(&a, &b).expect("dot")), [32.0]);
    assert_eq!(
        vals(&sig_outer()(&a, &b).expect("outer")),
        [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]
    );
}

// =====================================================================
// #2146 実装・#2516 公開（親 #2500・ルート #2499）の正ガード。
// `Var` 委譲メソッド 5 件（承認形）だけを許し、`Tensor<f32>`／`Tape` 上への
// 配置・モジュール再エクスポートは引き続き拒否する（#2510 型の部分反転。先例
// #2198・#2338・#2511）。`compat::Sequential::add_*` 5 種は #2529 で公開済みで、
// 承認形だけを許す正ガードを本節末尾の #2529 節へ新設した（#2528 の
// `DropoutEmbeddingBagHoldDoctestGuard` 縮小と同じ部分反転）。承認事項・多層防御の位置づけは
// `docs/autodiff-activation-ops-decision.md` §6 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `VarActivationOpsHoldDoctestGuard` doc
/// 内の唯一の doctest ブロックが glob import するネスト `pub mod` 集合
/// と、`src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （旧 `matrix_ops_hold_doctest_globs_all_pub_modules`〈#2513 で削除済み〉 の
/// `VarActivationOpsHoldDoctestGuard` 版）。
#[test]
fn activation_ops_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarActivationOpsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "VarActivationOpsHoldDoctestGuard の doctest ブロックが glob import\
         するモジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`activation_ops_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`ACTIVATION_OPS_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（旧 `matrix_ops_hold_
/// doctest_probe_body_matches_fixed_contract`〈#2513 で削除済み〉と同じ理由: rustdoc の
/// `# ` 隠し行・プローブの削除・別名へのシャドーイング等で正のプローブ
/// を骨抜きにする改変を機械的に拒否する）。
#[test]
fn activation_ops_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarActivationOpsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, ACTIVATION_OPS_HOLD_PROBE_BODY,
        "VarActivationOpsHoldDoctestGuard の doctest ブロック本文（glob\
         以外）が固定文言 ACTIVATION_OPS_HOLD_PROBE_BODY からドリフト\
         している。正のプローブ（__fandhe_activation_hold_probe\
         モジュール・__FandheActivationHoldProbe トレイト・__probe_* 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`activation_ops_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `VarActivationOpsHoldDoctestGuard` doc 内の唯一の doctest ブロック
/// から、ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）
/// を除いた本文と 1 行単位で完全一致する必要がある（クレートルート
/// 自体の `use fandhe_ai::*;` は本文に含む）。
const ACTIVATION_OPS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_activation_hold_probe {\n\
\x20\x20\x20\x20pub mod activation_ops {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn mish() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn hardtanh() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn relu6() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn prelu() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn glu() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_activation_hold_probe::*;\n\
\n\
struct __FandheActivationMarker;\n\
\n\
trait __FandheActivationHoldProbe {\n\
\x20\x20\x20\x20fn mish(&self) -> __FandheActivationMarker;\n\
\x20\x20\x20\x20fn hardtanh(&self) -> __FandheActivationMarker;\n\
\x20\x20\x20\x20fn relu6(&self) -> __FandheActivationMarker;\n\
\x20\x20\x20\x20fn prelu(&self) -> __FandheActivationMarker;\n\
\x20\x20\x20\x20fn glu(&self) -> __FandheActivationMarker;\n\
}\n\
\n\
impl __FandheActivationHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn mish(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn hardtanh(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn relu6(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn prelu(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn glu(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
}\n\
\n\
impl __FandheActivationHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn mish(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn hardtanh(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn relu6(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn prelu(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
\x20\x20\x20\x20fn glu(&self) -> __FandheActivationMarker { __FandheActivationMarker }\n\
}\n\
\n\
fn __probe_free_fns() {\n\
\x20\x20\x20\x20// `activation_ops::` を経由した経路解決（`use fandhe_ai::*;` が\n\
\x20\x20\x20\x20// 同名モジュールを glob 公開していれば、名前解決自体が曖昧に\n\
\x20\x20\x20\x20// なり E0659 でコンパイル失敗する）。\n\
\x20\x20\x20\x20activation_ops::mish();\n\
\x20\x20\x20\x20activation_ops::hardtanh();\n\
\x20\x20\x20\x20activation_ops::relu6();\n\
\x20\x20\x20\x20activation_ops::prelu();\n\
\x20\x20\x20\x20activation_ops::glu();\n\
}\n\
\n\
fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheActivationMarker = fandhe_ai::Tensor::hardtanh(x);\n\
\x20\x20\x20\x20let _: __FandheActivationMarker = x.hardtanh();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheActivationMarker = fandhe_ai::Tape::prelu(x);\n\
\x20\x20\x20\x20let _: __FandheActivationMarker = x.prelu();\n\
}";

/// `mish`・`hardtanh`・`relu6`・`prelu`・`glu`（5 個の関数名。イシュー
/// #2146）。[`facade_does_not_reexport_or_declare_activation_ops`]・
/// [`workspace_declares_activation_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const ACTIVATION_OPS_FN_NAMES: [&str; 5] = ["mish", "hardtanh", "relu6", "prelu", "glu"];

/// facade src 全体（`crates/facade/src/**`）に、`activation_ops` を
/// 参照する `pub use`（`pub use fandhe_ai_autodiff::activation_ops;` 等
/// のモジュール再エクスポート・別名含む）も、[`ACTIVATION_OPS_FN_NAMES`]
/// （5 個）の `fn` 宣言
/// （可視性・宣言文脈を問わない。[`count_fn_declarations_by_name`] と
/// 同じ検出契約）も存在しないことを固定する
/// （`VarActivationOpsHoldDoctestGuard` の正のプローブと多層防御を成す
/// 最内層のソース走査ガード。`facade_does_not_reexport_or_declare_
/// matrix_ops` と同型。`compat::Sequential::add_*`〈`add_mish` 等〉は #2529 で
/// 承認形として公開済みのため本テストの検査対象から外し、正ガード
/// `compat_sequential_activation_layers_add_methods_have_approved_signatures`
/// ほかが担う）。
#[test]
fn facade_does_not_reexport_or_declare_activation_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "activation_ops")
            {
                offending.push(format!(
                    "{}: `{trimmed}` が `activation_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in ACTIVATION_OPS_FN_NAMES {
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
        "facade の公開面が activation_ops（承認形は autodiff の `Var` 委譲\
         メソッドと #2529 の `compat::Sequential::add_*` のみ。facade 側の\
         モジュール再エクスポート・同名 fn の宣言は承認形外）を\
         再エクスポート、または同名の fn を宣言している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`ACTIVATION_OPS_FN_NAMES`]（5 個）の `fn` 宣言の定義元集合を固定
/// する（`workspace_declares_matrix_ops_fn_names_only_in_autodiff_
/// matrix_ops` と同型のインベントリ）。
///
/// **期待集合は `crates/autodiff/src/activation_ops.rs`（各 1 件）に
/// 加え、`crates/tensor-core/src/scalar_op.rs::relu6`（private な既存
/// 定義。着手前確認の実測 grep で判明。実装計画「インベントリを実測
/// する」手順）**。
#[test]
fn workspace_declares_activation_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in ACTIVATION_OPS_FN_NAMES {
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

    let mut expected: std::collections::BTreeMap<String, usize> = ACTIVATION_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/activation_ops.rs::{name}"), 1usize))
        .collect();
    for name in ACTIVATION_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }
    expected.insert("tensor-core/src/scalar_op.rs::relu6".to_string(), 1usize);

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の activation_ops 系 `fn` 宣言\
         集合が期待（`crates/autodiff/src/activation_ops.rs` 各 1 件・\
         `autodiff/src/var.rs` 各 1 件・\
         `crates/tensor-core/src/scalar_op.rs::relu6` 1 件）と一致しない\
         （過不足いずれも fail-closed に検出する。新たな定義元が見つかった\
         場合、それが承認済みの実装なのか迂回経路の混入なのかを確認\
         すること）: {found:?}"
    );
}

/// `var.rs` の 5 委譲メソッド本体の承認形（`activation_ops` 自由関数への
/// 1 行委譲。引数名も固定）。独自実装・スタブへのすり替えと境界検査の
/// 迂回を拒否する（#2516。[`REARRANGE_OPS_VAR_EXPECTED_BODIES`] と同型）。
const ACTIVATION_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 5] = [
    ("mish", "crate : : activation_ops : : mish ( self )"),
    (
        "hardtanh",
        "crate : : activation_ops : : hardtanh ( self , min , max )",
    ),
    ("relu6", "crate : : activation_ops : : relu6 ( self )"),
    (
        "prelu",
        "crate : : activation_ops : : prelu ( self , weight )",
    ),
    ("glu", "crate : : activation_ops : : glu ( self , dim )"),
];

/// `var.rs` の 5 委譲メソッドの本体が [`ACTIVATION_OPS_VAR_EXPECTED_BODIES`]
/// と一致することを固定する（本体抽出は [`determinism_fn_body`] を再利用）。
#[test]
fn var_activation_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in ACTIVATION_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（activation_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで 5 メソッドへ到達でき、シグネチャが
/// 承認形と一致し、実際に適用して厳密に決まる期待値が得られることを固定する
/// （スタブでは通らない。#2516）。
#[test]
fn var_activation_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    fn sig_mish<'t>() -> fn(&Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::mish
    }
    fn sig_hardtanh<'t>() -> fn(&Var<'t>, f32, f32) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::hardtanh
    }
    fn sig_relu6<'t>() -> fn(&Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::relu6
    }
    fn sig_prelu<'t>() -> fn(&Var<'t>, &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::prelu
    }
    fn sig_glu<'t>() -> fn(&Var<'t>, usize) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::glu
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let z = tape.var(&Tensor::new(vec![0.0_f32], &[1]).expect("tensor"));
    assert_eq!(vals(&sig_mish()(&z).expect("mish")), [0.0]);

    let x = tape.var(&Tensor::new(vec![-2.0_f32, 0.5, 2.0], &[3]).expect("tensor"));
    assert_eq!(
        vals(&sig_hardtanh()(&x, -1.0, 1.0).expect("hardtanh")),
        [-1.0, 0.5, 1.0]
    );

    let r = tape.var(&Tensor::new(vec![-1.0_f32, 3.0, 7.0], &[3]).expect("tensor"));
    assert_eq!(vals(&sig_relu6()(&r).expect("relu6")), [0.0, 3.0, 6.0]);

    let p = tape.var(&Tensor::new(vec![-2.0_f32, 3.0], &[2]).expect("tensor"));
    let w = tape.var(&Tensor::new(vec![0.25_f32], &[1]).expect("tensor"));
    assert_eq!(vals(&sig_prelu()(&p, &w).expect("prelu")), [-0.5, 3.0]);

    let g = tape.var(&Tensor::new(vec![2.0_f32, 4.0, 0.0, 0.0], &[4]).expect("tensor"));
    assert_eq!(vals(&sig_glu()(&g, 0).expect("glu")), [1.0, 2.0]);
}

// =====================================================================
// #2147 実装・#2514 公開（親 #2500・ルート #2499）の reduce_ops 正ガード。
// 旧否定ガード（`VarReduceOpsHoldDoctestGuard`）を、承認形（`Var` の 1 行
// 委譲メソッド 5 件のみ）だけを許す形へ反転した（先例 #2198・#2338・
// #2511・#2512）。モジュール再エクスポート・別名・独自実装へのすり替えは
// 拒否する。公開形の記録は `docs/autodiff-reduce-ops-decision.md` §6・§9 参照。
// =====================================================================

/// `prod`・`logsumexp`・`any`・`all`・`norm_p`（5 個の関数名。イシュー
/// #2147）。[`facade_does_not_reexport_or_declare_reduce_ops`]・
/// [`workspace_declares_reduce_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const REDUCE_OPS_FN_NAMES: [&str; 5] = ["prod", "logsumexp", "any", "all", "norm_p"];

/// facade src 全体（`crates/facade/src/**`）に、`reduce_ops` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::reduce_ops;` 等の
/// モジュール再エクスポート・別名含む）も、[`REDUCE_OPS_FN_NAMES`]
/// （5 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する（旧 `VarReduceOpsHoldDoctestGuard`〈#2514 で削除〉に代わる
/// 最内層のソース走査ガード。`facade_does_not_reexport_or_declare_
/// matrix_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_reduce_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "reduce_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `reduce_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in REDUCE_OPS_FN_NAMES {
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
        "facade の公開面が reduce_ops を再エクスポート、または同名の fn を\
         宣言している（承認形は autodiff の `Var` 委譲メソッドのみ。\
         イシュー #2514）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`REDUCE_OPS_FN_NAMES`]
/// （5 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// matrix_ops_fn_names_only_in_approved_locations`（#2513 で改名）と同型のインベン
/// トリ）。
///
/// **期待集合**（着手前確認の再 grep で判明。実装計画「インベントリを
/// 実測する」手順）: `prod`・`any`・`all`・`norm_p` は
/// `crates/autodiff/src/reduce_ops.rs` にのみ 1 件ずつ存在する。
/// `logsumexp` はそれに加えて `BackendOps` trait のデフォルトメソッド
/// （`crates/tensor-core/src/backend_ops.rs`）・CPU 実装
/// （`crates/backend-cpu/src/reduction.rs`・`crates/backend-cpu/src/
/// ops.rs`）にも 1 件ずつ存在する正規の宣言元。
#[test]
fn workspace_declares_reduce_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in REDUCE_OPS_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("autodiff/src/reduce_ops.rs::prod", 1usize),
        ("autodiff/src/reduce_ops.rs::logsumexp", 1usize),
        ("autodiff/src/reduce_ops.rs::any", 1usize),
        ("autodiff/src/reduce_ops.rs::all", 1usize),
        ("autodiff/src/reduce_ops.rs::norm_p", 1usize),
        ("autodiff/src/var.rs::prod", 1usize),
        ("autodiff/src/var.rs::logsumexp", 1usize),
        ("autodiff/src/var.rs::any", 1usize),
        ("autodiff/src/var.rs::all", 1usize),
        ("autodiff/src/var.rs::norm_p", 1usize),
        ("tensor-core/src/backend_ops.rs::logsumexp", 1usize),
        ("backend-cpu/src/reduction.rs::logsumexp", 1usize),
        ("backend-cpu/src/ops.rs::logsumexp", 1usize),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の reduce_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/reduce_ops.rs` 5 件 +\
         `autodiff/src/var.rs` 5 件 + `logsumexp` の BackendOps trait・CPU 実装 3 件）と一致しない\
         （過不足いずれも fail-closed に検出する。新たな定義元が\
         見つかった場合、それが承認済みの実装なのか迂回経路の混入なのか\
         を確認すること）: {found:?}"
    );
}

// =====================================================================
// #2150 実装・#2515 公開（親 #2500・ルート #2499）の正ガード。旧否定ガード
// （`VarLinalgOpsHoldDoctestGuard`・doctest 固定文言 2 件）を、承認形（`Var` の
// 1 行委譲メソッドのみ）だけを許す形へ反転した（先例 #2198・#2338・#2510・#2511）。
// 承認事項・残る承認事項は `docs/autodiff-linalg-ops-decision.md` §6 参照。
// =====================================================================

/// `eigh`・`slogdet`・`pinv`・`matrix_rank`・`lstsq`（5 個の関数名。
/// イシュー #2150・#2515）。[`facade_does_not_reexport_or_declare_linalg_ops`]・
/// [`workspace_declares_linalg_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const LINALG_OPS_FN_NAMES: [&str; 5] = ["eigh", "slogdet", "pinv", "matrix_rank", "lstsq"];

/// facade src 全体（`crates/facade/src/**`）に、`linalg_ops` を参照
/// する `pub use`（モジュール再エクスポート・別名含む）も、
/// [`LINALG_OPS_FN_NAMES`]（5 個）の `fn` 宣言も存在しないことを固定
/// する（承認形は autodiff の `Var` 委譲メソッドのみ。イシュー #2515。
/// `facade_does_not_reexport_or_declare_rearrange_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_linalg_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "linalg_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `linalg_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in LINALG_OPS_FN_NAMES {
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
        "facade の公開面が linalg_ops を再エクスポート、または同名の fn を\
         宣言している（承認形は autodiff の `Var` 委譲メソッドのみ。\
         イシュー #2515）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`LINALG_OPS_FN_NAMES`]
/// （5 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// rearrange_ops_fn_names_only_in_approved_locations` と同型のインベン
/// トリ。#2515 で `autodiff/src/var.rs` の委譲メソッド各 1 件を承認形として追加）。
///
/// **期待集合**（実測: `eigh`・`slogdet`・`pinv`・`matrix_rank`・
/// `lstsq` は `crates/autodiff/src/linalg_ops.rs`（公開入口）・
/// `crates/autodiff/src/eval/linalg.rs`（ホスト参照実装。`slogdet` は
/// `slogdet_logabsdet_vjp` 等の別名のため 1 件のまま）・
/// `crates/backend-cpu/src/linalg.rs`（CPU 本番実装）にそれぞれ 1 件
/// ずつ存在する。加えて #2515 で `crates/autodiff/src/var.rs` の `Var`
/// 委譲メソッドが各 1 件）。
#[test]
fn workspace_declares_linalg_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in LINALG_OPS_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("autodiff/src/linalg_ops.rs::eigh", 1usize),
        ("autodiff/src/linalg_ops.rs::slogdet", 1usize),
        ("autodiff/src/linalg_ops.rs::pinv", 1usize),
        ("autodiff/src/linalg_ops.rs::matrix_rank", 1usize),
        ("autodiff/src/linalg_ops.rs::lstsq", 1usize),
        ("autodiff/src/eval/linalg.rs::eigh", 1usize),
        ("autodiff/src/eval/linalg.rs::slogdet", 1usize),
        ("autodiff/src/eval/linalg.rs::pinv", 1usize),
        ("autodiff/src/eval/linalg.rs::matrix_rank", 1usize),
        ("autodiff/src/eval/linalg.rs::lstsq", 1usize),
        ("backend-cpu/src/linalg.rs::eigh", 1usize),
        ("backend-cpu/src/linalg.rs::slogdet", 1usize),
        ("backend-cpu/src/linalg.rs::pinv", 1usize),
        ("backend-cpu/src/linalg.rs::matrix_rank", 1usize),
        ("backend-cpu/src/linalg.rs::lstsq", 1usize),
        ("autodiff/src/var.rs::eigh", 1usize),
        ("autodiff/src/var.rs::slogdet", 1usize),
        ("autodiff/src/var.rs::pinv", 1usize),
        ("autodiff/src/var.rs::matrix_rank", 1usize),
        ("autodiff/src/var.rs::lstsq", 1usize),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の linalg_ops 系 `fn` 宣言集合が\
         期待（`autodiff/src/linalg_ops.rs`・`autodiff/src/eval/linalg.rs`・\
         `backend-cpu/src/linalg.rs`・`autodiff/src/var.rs` に各 5 件）と一致しない（過不足いずれも\
         fail-closed に検出する。新たな定義元が見つかった場合、それが\
         承認済みの実装なのか迂回経路の混入なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 5 委譲メソッド本体の承認形（`linalg_ops` 自由関数への
/// 1 行委譲。引数名も固定）。独自実装・スタブへのすり替えと入口検証
/// （`require_finite`・`rcond` 検査・確保前上限検査）の迂回を拒否する
/// （#2515。[`REARRANGE_OPS_VAR_EXPECTED_BODIES`] と同型）。
const LINALG_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 5] = [
    ("eigh", "crate : : linalg_ops : : eigh ( self )"),
    ("slogdet", "crate : : linalg_ops : : slogdet ( self )"),
    ("pinv", "crate : : linalg_ops : : pinv ( self , rcond )"),
    (
        "matrix_rank",
        "crate : : linalg_ops : : matrix_rank ( self , rcond )",
    ),
    (
        "lstsq",
        "crate : : linalg_ops : : lstsq ( self , b , rcond )",
    ),
];

/// `var.rs` の 5 委譲メソッドの本体が [`LINALG_OPS_VAR_EXPECTED_BODIES`]
/// と一致することを固定する（本体抽出は [`determinism_fn_body`] を再利用）。
#[test]
fn var_linalg_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in LINALG_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（linalg_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけ（`fandhe_ai_autodiff` を import
/// しない）で 5 メソッドへ到達でき、シグネチャ・pub フィールドが承認形と
/// 一致し、実際に適用して期待値が得られることを固定する（#2515）。
/// `EighVars`／`SlogdetVars` は facade から名前で書けないため fn ポインタでは
/// 固定せず、`Var::eigh(&x)`／`x.eigh()` の両呼び出しとフィールド型で固定する。
#[test]
fn var_linalg_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type RcondSig<'t> = fn(&Var<'t>, Option<f32>) -> Result<Var<'t>, AutodiffError>;
    fn sig_pinv<'t>() -> RcondSig<'t> {
        Var::<'t>::pinv
    }
    fn sig_matrix_rank<'t>() -> RcondSig<'t> {
        Var::<'t>::matrix_rank
    }
    type LstsqSig<'t> = fn(&Var<'t>, &Var<'t>, Option<f32>) -> Result<Var<'t>, AutodiffError>;
    fn sig_lstsq<'t>() -> LstsqSig<'t> {
        Var::<'t>::lstsq
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let d = tape.var(&Tensor::new(vec![3.0_f32, 0.0, 0.0, 1.0], &[2, 2]).expect("tensor"));

    let e1 = Var::eigh(&d).expect("eigh");
    let e2 = d.eigh().expect("eigh method");
    let ev: &Var<'_> = &e1.eigenvalues;
    let evec: &Var<'_> = &e2.eigenvectors;
    assert_eq!(vals(ev), [1.0, 3.0]);
    assert_eq!(evec.to_tensor().shape(), &[2, 2]);

    let neg = tape.var(&Tensor::new(vec![1.0_f32, 0.0, 0.0, -2.0], &[2, 2]).expect("tensor"));
    let s1 = Var::slogdet(&neg).expect("slogdet");
    let s2 = neg.slogdet().expect("slogdet method");
    let sign: &Var<'_> = &s1.sign;
    let logabs: &Var<'_> = &s2.logabsdet;
    assert_eq!(vals(sign), [-1.0]);
    assert_eq!(vals(logabs).len(), 1);

    let p = tape.var(&Tensor::new(vec![2.0_f32, 0.0, 0.0, 4.0], &[2, 2]).expect("tensor"));
    assert_eq!(
        vals(&sig_pinv()(&p, None).expect("pinv")),
        [0.5, 0.0, 0.0, 0.25]
    );

    let r = tape.var(&Tensor::new(vec![1.0_f32, 0.0, 0.0, 0.0], &[2, 2]).expect("tensor"));
    assert_eq!(
        vals(&sig_matrix_rank()(&r, None).expect("matrix_rank")),
        [1.0]
    );

    let i2 = tape.var(&Tensor::new(vec![1.0_f32, 0.0, 0.0, 1.0], &[2, 2]).expect("tensor"));
    let b = tape.var(&Tensor::new(vec![5.0_f32, 7.0], &[2, 1]).expect("tensor"));
    assert_eq!(
        vals(&sig_lstsq()(&i2, &b, None).expect("lstsq")),
        [5.0, 7.0]
    );
}

// =====================================================================
// #2149（親 #2131）の einsum batch 添字縮約の正ガード（#2517 で反転）。
// 旧 `VarEinsumBatchHoldDoctestGuard`（#2149）と対応する否定テスト 2 件
// （`einsum_batch_hold_doctest_*`）は #2517 で削除済み。承認形は
// 「`Var::einsum` 自体が batch 添字付き縮約を受理する」ことだけで、
// 新しい公開名は追加しない（`einsum_batch` の再エクスポート・別名・
// facade 側の `fn einsum_batched` 宣言は引き続き拒否する）。承認の記録は
// `docs/autodiff-einsum-batch-decision.md` §11 参照。
// =====================================================================

/// `einsum_batched`（1 個の関数名。イシュー #2149）。
/// [`facade_does_not_reexport_or_declare_einsum_batch`]・
/// [`workspace_declares_einsum_batched_fn_only_in_autodiff_einsum_batch`]
/// が共用する。
const EINSUM_BATCH_FN_NAMES: [&str; 1] = ["einsum_batched"];

/// facade src 全体（`crates/facade/src/**`）に、`einsum_batch` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::einsum_batch;` 等の
/// モジュール再エクスポート・別名含む）も、[`EINSUM_BATCH_FN_NAMES`]
/// （1 個）の `fn` 宣言（可視性・`self` の有無を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約——`Var::einsum` と
/// 同じ「`self` を取らない関連関数」として追加される経路も本走査が
/// 捕捉する）も存在しないことを固定する（#2517 で承認形〈`Var::einsum` の
/// 受理拡張のみ〉の正ガードへ位置づけを変更。検査ロジックは不変。
/// 旧 `VarEinsumBatchHoldDoctestGuard` は削除済み）。
#[test]
fn facade_does_not_reexport_or_declare_einsum_batch() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "einsum_batch") {
                offending.push(format!(
                    "{}: `{trimmed}` が `einsum_batch` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in EINSUM_BATCH_FN_NAMES {
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
        "facade の公開面が einsum_batch（互換用の内部クレート限定入口。\
         承認形〈#2517。Var::einsum の batch 添字受理拡張のみ〉は新しい\
         公開名を追加しない）を再エクスポート、または同名の fn を宣言\
         している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`EINSUM_BATCH_FN_NAMES`]
/// （1 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// rearrange_ops_fn_names_only_in_approved_locations`（#2511 で改名）と同型の
/// インベントリ）。
///
/// **期待集合は `crates/autodiff/src/einsum_batch.rs`（1 件）のみ**
/// （着手前確認の再 grep で他クレートとの衝突は見つからなかった。実装
/// 計画「インベントリを実測する」手順）。
#[test]
fn workspace_declares_einsum_batched_fn_only_in_autodiff_einsum_batch() {
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
            for fn_name in EINSUM_BATCH_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = EINSUM_BATCH_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/einsum_batch.rs::{name}"), 1usize))
        .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の einsum_batch 系 `fn` 宣言集合が\
         `crates/autodiff/src/einsum_batch.rs`（1 件）のみという期待と\
         一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}

/// 承認形（#2517）の固定: `Var::einsum` と `einsum_batched` の本体が
/// いずれも `crate::einsum::einsum(spec, operands)` への 1 行委譲である
/// こと（[`determinism_fn_body`] を再利用。`pub fn` がちょうど 1 件の
/// とき本体を返す。`einsum.rs::einsum` は `pub(crate)` のため対象外）。
#[test]
fn var_einsum_and_einsum_batched_are_thin_delegations() {
    const EXPECTED: &str = "crate : : einsum : : einsum ( spec , operands )";
    for (rel, name) in [
        ("autodiff/src/var.rs", "einsum"),
        ("autodiff/src/einsum_batch.rs", "einsum_batched"),
    ] {
        let content = read_to_string_or_panic(&workspace_crates_dir().join(rel));
        let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(EXPECTED),
            "{rel} の `{name}` の本体が承認形（crate::einsum::einsum への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけ（`fandhe_ai_autodiff` を import
/// しない）で `Var::einsum` が batch 添字付き縮約を受理し、シグネチャが
/// 承認形と一致し、形状・値が `Var::matmul` と bit 一致することを固定する
/// （スタブでは通らない。#2517）。
#[test]
fn var_einsum_batch_contraction_is_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type EinsumSig<'t> = fn(&str, &[&Var<'t>]) -> Result<Var<'t>, AutodiffError>;
    fn sig_einsum<'t>() -> EinsumSig<'t> {
        Var::<'t>::einsum
    }

    let tape = fandhe_ai::tape();
    let a_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0..40).map(|i| i as f32 * 0.125 - 1.0).collect();
    let a = tape.var(&Tensor::new(a_data, &[2, 3, 4]).expect("tensor a"));
    let b = tape.var(&Tensor::new(b_data, &[2, 4, 5]).expect("tensor b"));
    let out = sig_einsum()("bij,bjk->bik", &[&a, &b]).expect("batch 縮約は受理される");
    assert_eq!(out.to_tensor().shape(), &[2, 3, 5]);
    let direct = a.matmul(&b).expect("matmul");
    let got = out.to_tensor().host_slice().into_owned();
    let want = direct.to_tensor().host_slice().into_owned();
    assert_eq!(got.len(), 30);
    assert!(
        got.iter()
            .zip(want.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "Var::einsum の batch 縮約が Var::matmul と bit 一致しない"
    );
}

// =====================================================================
// #2148 実装・#2518 公開（親 #2500・ルート #2499）の正ガード。旧否定ガード
// （`VarIndexingOpsHoldDoctestGuard`・対応するテスト 2 件・固定文言）を、承認形
// （`Var::advanced_indexing`／`index_put`／`index_put_` の 1 行委譲）だけを許す形へ
// 反転した（先例 #2198・#2338・#2511）。承認事項は
// `docs/autodiff-indexing-inplace-design.md` §6 参照。
// =====================================================================

/// `advanced_indexing`・`index_put`・`index_put_`（3 個の関数名。イシュー
/// #2148）。[`facade_does_not_reexport_or_declare_indexing_ops`]・
/// [`workspace_declares_indexing_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const INDEXING_OPS_FN_NAMES: [&str; 3] = ["advanced_indexing", "index_put", "index_put_"];

/// facade src 全体（`crates/facade/src/**`）に、`indexing_ops` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::indexing_ops;` 等の
/// モジュール再エクスポート・別名含む）も、[`INDEXING_OPS_FN_NAMES`]
/// （3 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する。承認形は autodiff 側の `Var` の inherent 委譲メソッドのみで、
/// facade でのモジュール再エクスポート・別名・独自の `fn` 宣言は引き続き
/// 拒否する（#2518 で保留ガードから正ガードの一部へ位置づけを変更。検査
/// ロジックは不変）。
#[test]
fn facade_does_not_reexport_or_declare_indexing_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "indexing_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `indexing_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in INDEXING_OPS_FN_NAMES {
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
        "facade が indexing_ops を再エクスポート、または同名の fn を宣言している\
         （承認形は autodiff の `Var` 委譲メソッドのみ。イシュー #2518）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`INDEXING_OPS_FN_NAMES`]
/// （3 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// reduce_ops_fn_names_only_in_allowed_locations` と同型のインベン
/// トリ）。
///
/// **期待集合**（着手前確認の再 grep で判明。実装計画「インベントリを
/// 実測する」手順）: `advanced_indexing`・`index_put`・`index_put_` は
/// いずれも `crates/autodiff/src/indexing_ops.rs`（各 1 件）と
/// `crates/autodiff/src/var.rs`（`Var` 委譲メソッド。各 1 件。#2518）に
/// のみ存在する。
#[test]
fn workspace_declares_indexing_ops_fn_names_only_in_approved_locations() {
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
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in INDEXING_OPS_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }

    let mut expected: std::collections::BTreeMap<String, usize> = INDEXING_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/indexing_ops.rs::{name}"), 1usize))
        .collect();
    for name in INDEXING_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の indexing_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/indexing_ops.rs` 各 1 件 +\
         `autodiff/src/var.rs` 各 1 件）と一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 3 委譲メソッド本体の承認形（`indexing_ops` 自由関数への
/// 1 行委譲。引数名も固定）。独自実装・スタブへのすり替えと添字検査の
/// 迂回を拒否する（#2518。[`REARRANGE_OPS_VAR_EXPECTED_BODIES`] と同型）。
const INDEXING_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 3] = [
    (
        "advanced_indexing",
        "crate : : indexing_ops : : advanced_indexing ( self , indices )",
    ),
    (
        "index_put",
        "crate : : indexing_ops : : index_put ( self , indices , values , accumulate )",
    ),
    (
        "index_put_",
        "crate : : indexing_ops : : index_put_ ( self , indices , values , accumulate )",
    ),
];

/// `var.rs` の 3 委譲メソッドの本体が [`INDEXING_OPS_VAR_EXPECTED_BODIES`]
/// と一致することを固定する（本体抽出は [`determinism_fn_body`] を再利用）。
#[test]
fn var_indexing_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in INDEXING_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（indexing_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけ（`fandhe_ai_autodiff` を import
/// しない）で 3 メソッドへ到達でき、シグネチャが承認形と一致し、実際に
/// 適用して期待値が得られることを固定する（スタブでは通らない。#2518）。
#[test]
fn var_indexing_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type ReadSig<'t> = fn(&Var<'t>, &[Tensor<i32>]) -> Result<Var<'t>, AutodiffError>;
    type PutSig<'t> =
        fn(&Var<'t>, &[Tensor<i32>], &Var<'t>, bool) -> Result<Var<'t>, AutodiffError>;
    type PutInPlaceSig<'t> =
        fn(&mut Var<'t>, &[Tensor<i32>], &Var<'t>, bool) -> Result<(), AutodiffError>;
    fn sig_read<'t>() -> ReadSig<'t> {
        Var::<'t>::advanced_indexing
    }
    fn sig_put<'t>() -> PutSig<'t> {
        Var::<'t>::index_put
    }
    fn sig_put_in_place<'t>() -> PutInPlaceSig<'t> {
        Var::<'t>::index_put_
    }
    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    let tape = fandhe_ai::tape();
    let mut x = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0], &[3]).expect("tensor"));
    let idx = [Tensor::new(vec![2_i32, 0], &[2]).expect("index")];
    let v = tape.var(&Tensor::new(vec![10.0_f32, 20.0], &[2]).expect("values"));
    assert_eq!(
        vals(&sig_read()(&x, &idx).expect("advanced_indexing")),
        [3.0, 1.0]
    );
    assert_eq!(
        vals(&sig_put()(&x, &idx, &v, false).expect("index_put")),
        [20.0, 2.0, 10.0]
    );
    sig_put_in_place()(&mut x, &idx, &v, true).expect("index_put_");
    assert_eq!(vals(&x), [21.0, 2.0, 13.0]);
}

// =====================================================================
// #2153 実装・#2519 公開（親 #2500・ルート #2499）の topk_unique_ops 正ガード。
// 旧否定ガード（`VarTopkUniqueOpsHoldDoctestGuard`）を、承認形（`Var` の 1 行
// 委譲メソッド 3 件と、入出力型 3 個の autodiff ルート経由 `pub use` のみ）だけを
// 許す形へ反転した（先例 #2198・#2338・#2511・#2518）。公開形の記録は
// `docs/autodiff-topk-unique-ops-decision.md` §6 参照。
// =====================================================================

/// `topk_with_options`・`unique_with_options`・`unique_consecutive`
/// （3 個の関数名。イシュー #2153）。[`facade_does_not_reexport_or_
/// declare_topk_unique_ops`]・[`workspace_declares_topk_unique_ops_fn_
/// names_only_in_allowed_locations`] が共用する。
const TOPK_UNIQUE_OPS_FN_NAMES: [&str; 3] = [
    "topk_with_options",
    "unique_with_options",
    "unique_consecutive",
];

/// facade src 全体（`crates/facade/src/**`）に、`topk_unique_ops` を
/// 参照する `pub use`（`pub use fandhe_ai_autodiff::topk_unique_ops;`
/// 等のモジュール再エクスポート・別名含む）も、
/// [`TOPK_UNIQUE_OPS_FN_NAMES`]（3 個）の `fn` 宣言（可視性・宣言
/// 文脈を問わない。[`count_fn_declarations_by_name`] と同じ検出契約）
/// も存在しないことを固定する。承認形は autodiff の `Var` 委譲メソッド
/// （とクレートルート経由の型 3 個）のみで、facade でのモジュール再エクスポート・
/// 別名・独自 `fn` 宣言は引き続き拒否する（#2519 で正ガードの一部へ位置づけ
/// を変更。`facade_does_not_reexport_or_declare_indexing_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_topk_unique_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use")
                && line_contains_identifier(trimmed, "topk_unique_ops")
            {
                offending.push(format!(
                    "{}: `{trimmed}` が `topk_unique_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in TOPK_UNIQUE_OPS_FN_NAMES {
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
        "facade が topk_unique_ops を再エクスポート、または同名の fn を宣言\
         している（承認形は autodiff の `Var` 委譲メソッドのみ。#2519）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`TOPK_UNIQUE_OPS_FN_NAMES`]（3 個）の `fn` 宣言の定義元集合を
/// 固定する（`workspace_declares_indexing_ops_fn_names_only_in_
/// approved_locations`〈旧 only_in_autodiff_indexing_ops。#2518 で改名〉と同型のインベントリ）。
///
/// **期待集合**: `topk_with_options`・`unique_with_options`・
/// `unique_consecutive` はいずれも `crates/autodiff/src/topk_unique_ops.rs`
/// （各 1 件）と `crates/autodiff/src/var.rs`（`Var` 委譲メソッド。各 1 件。
/// #2519）にのみ存在する。
#[test]
fn workspace_declares_topk_unique_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in TOPK_UNIQUE_OPS_FN_NAMES {
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

    let mut expected: std::collections::BTreeMap<String, usize> = TOPK_UNIQUE_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/topk_unique_ops.rs::{name}"), 1usize))
        .collect();
    for name in TOPK_UNIQUE_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の topk_unique_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/topk_unique_ops.rs` 各 1 件 +\
         `autodiff/src/var.rs` 各 1 件）と一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 3 委譲メソッド本体の承認形（`topk_unique_ops` 自由関数への
/// 1 行委譲。引数名も固定）。確保前検査・0-d 拒否・`i32` 上限検査の迂回や
/// 独自実装・スタブへのすり替えを拒否する（#2519）。
const TOPK_UNIQUE_OPS_VAR_EXPECTED_BODIES: [(&str, &str); 3] = [
    (
        "topk_with_options",
        "crate : : topk_unique_ops : : topk_with_options ( self , k , opts )",
    ),
    (
        "unique_with_options",
        "crate : : topk_unique_ops : : unique_with_options ( self , opts )",
    ),
    (
        "unique_consecutive",
        "crate : : topk_unique_ops : : unique_consecutive ( self , opts )",
    ),
];

/// `var.rs` の 3 委譲メソッドの本体が [`TOPK_UNIQUE_OPS_VAR_EXPECTED_BODIES`]
/// と一致することを固定する（本体抽出は [`determinism_fn_body`] を再利用）。
#[test]
fn var_topk_unique_ops_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in TOPK_UNIQUE_OPS_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（topk_unique_ops 自由関数への \
             1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai` 経由だけ（`fandhe_ai_autodiff` を import しない）で
/// 3 メソッドと入出力型 3 個へ到達でき、シグネチャが承認形と一致し、実際に
/// 適用して期待値が得られることを固定する（スタブでは通らない。#2519）。
#[test]
fn var_topk_unique_ops_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, TopkOptions, UniqueOptions, UniqueOutput, Var};

    type TopkSig<'t> =
        fn(&Var<'t>, usize, TopkOptions) -> Result<(Var<'t>, Tensor<i32>), AutodiffError>;
    type UniqueSig<'t> = fn(&Var<'t>, UniqueOptions) -> Result<UniqueOutput, AutodiffError>;
    fn sig_topk<'t>() -> TopkSig<'t> {
        Var::<'t>::topk_with_options
    }
    fn sig_unique<'t>() -> UniqueSig<'t> {
        Var::<'t>::unique_with_options
    }
    fn sig_consecutive<'t>() -> UniqueSig<'t> {
        Var::<'t>::unique_consecutive
    }

    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![3.0_f32, 1.0, 2.0], &[3]).expect("tensor"));
    let opts = TopkOptions::default().with_dim(-1).with_sorted(false);
    let (values, index) = sig_topk()(&x, 2, opts).expect("topk_with_options");
    assert_eq!(values.to_tensor().host_slice().into_owned(), [3.0, 2.0]);
    assert_eq!(index.host_slice().into_owned(), [0, 2]);

    let y = tape.var(&Tensor::new(vec![2.0_f32, 2.0, 1.0, 2.0], &[4]).expect("tensor"));
    let uo = UniqueOptions::default()
        .with_return_inverse(true)
        .with_return_counts(true);
    let out = sig_unique()(&y, uo).expect("unique_with_options");
    assert_eq!(out.values.host_slice().into_owned(), [1.0, 2.0]);
    assert_eq!(
        out.inverse.expect("inverse").host_slice().into_owned(),
        [1, 1, 0, 1]
    );
    assert_eq!(
        out.counts.expect("counts").host_slice().into_owned(),
        [1, 3]
    );
    let out = sig_consecutive()(&y, uo).expect("unique_consecutive");
    assert_eq!(out.values.host_slice().into_owned(), [2.0, 1.0, 2.0]);
    assert_eq!(
        out.counts.expect("counts").host_slice().into_owned(),
        [2, 1, 1]
    );
}

/// `topk_unique_ops` の入出力型 3 個（#2519）。
const TOPK_UNIQUE_TYPE_NAMES: [&str; 3] = ["TopkOptions", "UniqueOptions", "UniqueOutput"];

/// 承認形の唯一の `pub use` 行（`src/lib.rs`。autodiff ルート経由・別名なし）。
const TOPK_UNIQUE_TYPES_APPROVED_LINE: &str =
    "pub use fandhe_ai_autodiff::{TopkOptions, UniqueOptions, UniqueOutput};";

/// facade src の 1 ファイル内容から、3 型名を識別子として含む `pub use` 行
/// （空白正規化済み）を集める検出本体。コメント・文字列リテラルは無視する。
fn scan_topk_unique_type_pub_use_lines(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    cleaned
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("pub use") || l.starts_with("pub(crate) use"))
        .filter(|l| {
            TOPK_UNIQUE_TYPE_NAMES
                .iter()
                .any(|n| line_contains_identifier(l, n))
        })
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

/// 3 型の再エクスポートが `src/lib.rs` の承認形 1 行だけであることを固定する
/// （別名・`topk_unique_ops::` 経由・別モジュールからの再エクスポートは fail。
/// `facade_reexports_prefetch_items_only_in_approved_shape` と同型）。
#[test]
fn facade_reexports_topk_unique_types_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_lib_rs = 0usize;
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in scan_topk_unique_type_pub_use_lines(content) {
            if path.ends_with("src/lib.rs") && line == TOPK_UNIQUE_TYPES_APPROVED_LINE {
                approved_in_lib_rs += 1;
            } else {
                offending.push(format!("{}: `{line}`", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade が TopkOptions・UniqueOptions・UniqueOutput を承認形（src/lib.rs の \
         `{TOPK_UNIQUE_TYPES_APPROVED_LINE}` 1 行）以外で再エクスポートしている \
         （#2519）: {offending:?}"
    );
    assert_eq!(
        approved_in_lib_rs, 1,
        "src/lib.rs に承認形の再エクスポート行がちょうど 1 行存在しない\
         （検査対象を見失った場合を含む）"
    );
}

/// [`scan_topk_unique_type_pub_use_lines`] の自己テスト（合成入力）。
#[test]
fn facade_reexports_topk_unique_types_only_in_approved_shape_detects_each_category() {
    let scan = scan_topk_unique_type_pub_use_lines;
    assert_eq!(
        scan(TOPK_UNIQUE_TYPES_APPROVED_LINE),
        vec![TOPK_UNIQUE_TYPES_APPROVED_LINE.to_string()]
    );
    // 違反: 別名・別経路・部分再エクスポート（承認形と文字列一致しない行として検出される）。
    for src in [
        "pub use fandhe_ai_autodiff::TopkOptions as Foo;",
        "pub use fandhe_ai_autodiff::topk_unique_ops::UniqueOutput;",
        "pub use fandhe_ai_autodiff::{UniqueOptions};",
    ] {
        let hits = scan(src);
        assert_eq!(hits.len(), 1, "src={src:?}");
        assert_ne!(hits[0], TOPK_UNIQUE_TYPES_APPROVED_LINE, "src={src:?}");
    }
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use fandhe_ai_autodiff::TopkOptions;",
        "let s = \"pub use x::UniqueOutput;\";",
        "use fandhe_ai_autodiff::TopkOptions;",
        "pub use fandhe_ai_autodiff::Var;",
    ] {
        assert!(scan(src).is_empty(), "src={src:?}");
    }
}
// =====================================================================
// #2154 実装・#2514 公開（親 #2500・ルート #2499）の extremum_ops 正ガード。
// 旧否定ガード（`VarExtremumOpsHoldDoctestGuard`）を、承認形（`Var` の 1 行
// 委譲メソッド 2 件のみ）だけを許す形へ反転した（先例 #2512）。公開形の
// 記録は `docs/autodiff-amax-grad-distribution-decision.md` §9 参照。
// =====================================================================

/// `amax`・`amin`（2 個の関数名。イシュー #2154）。
/// [`facade_does_not_reexport_or_declare_extremum_ops`]・
/// [`workspace_declares_extremum_ops_fn_names_only_in_approved_locations`]
/// が共用する。
const EXTREMUM_OPS_FN_NAMES: [&str; 2] = ["amax", "amin"];

/// facade src 全体（`crates/facade/src/**`）に、`extremum_ops` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::extremum_ops;` 等の
/// モジュール再エクスポート・別名含む）も、[`EXTREMUM_OPS_FN_NAMES`]
/// （2 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する（旧 `VarExtremumOpsHoldDoctestGuard`〈#2514 で削除〉に代わる
/// 最内層のソース走査ガード。`facade_does_not_reexport_or_declare_
/// reduce_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_extremum_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "extremum_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `extremum_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in EXTREMUM_OPS_FN_NAMES {
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
        "facade の公開面が extremum_ops を再エクスポート、または同名の fn を\
         宣言している（承認形は autodiff の `Var` 委譲メソッドのみ。\
         イシュー #2514）: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`EXTREMUM_OPS_FN_NAMES`]
/// （2 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// reduce_ops_fn_names_only_in_allowed_locations` と同型のインベン
/// トリ）。
///
/// **期待集合**（着手前確認の再 grep で判明。実装計画「インベントリを
/// 実測する」手順）: `amax`・`amin` はいずれも `crates/autodiff/src/
/// extremum_ops.rs` にのみ 1 件ずつ存在する。`crates/backend-cpu/src/
/// reduction.rs` のテストコードには `let amax = ...`／`let amin = ...`
/// というローカル変数束縛があるが、`fn` 宣言ではないため
/// [`count_fn_declarations_by_name`] は数えない（着手前確認の実行で
/// 確かめた。誤検出しないことのドキュメント）。
#[test]
fn workspace_declares_extremum_ops_fn_names_only_in_approved_locations() {
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
            for fn_name in EXTREMUM_OPS_FN_NAMES {
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

    let mut expected: std::collections::BTreeMap<String, usize> = EXTREMUM_OPS_FN_NAMES
        .iter()
        .map(|name| (format!("autodiff/src/extremum_ops.rs::{name}"), 1usize))
        .collect();
    // 承認形の `Var` 委譲メソッド（#2514）。
    for name in EXTREMUM_OPS_FN_NAMES {
        expected.insert(format!("autodiff/src/var.rs::{name}"), 1usize);
    }

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の extremum_ops 系 `fn` 宣言集合が\
         `crates/autodiff/src/extremum_ops.rs`（2 件）＋ `autodiff/src/var.rs`（2 件）という期待と\
         一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}

// =====================================================================
// #2157 実装・#2507 公開（親 #2499）の決定論モード正ガード。旧否定ガード
// （`DeterminismHoldDoctestGuard`）を、承認形（crate ルートの委譲 `pub fn`
// 2 件）だけを許す形へ反転した（先例 #2198・#2690）。モジュール再エクス
// ポート・別名・独自実装へのすり替えは拒否する。公開形の記録は
// `docs/autodiff-determinism-mode-design.md` §6 参照。
// =====================================================================

/// `set_deterministic`・`is_deterministic`（2 個の関数名。イシュー
/// #2157・#2507）。[`facade_declares_determinism_fns_only_as_approved_root_delegations`]・
/// [`workspace_declares_determinism_fn_names_only_in_allowed_locations`]
/// が共用する。
const DETERMINISM_FN_NAMES: [&str; 2] = ["set_deterministic", "is_deterministic"];

/// 承認形の関数本体（トークンを空白連結した形）。引数名 `enabled` も固定する。
const DETERMINISM_FN_EXPECTED_BODIES: [(&str, &str); 2] = [
    (
        "set_deterministic",
        "fandhe_ai_autodiff : : determinism : : set_deterministic ( enabled ) ;",
    ),
    (
        "is_deterministic",
        "fandhe_ai_autodiff : : determinism : : is_deterministic ( )",
    ),
];

/// トークン列から `pub fn <fn_name>` 宣言の本体（最外 `{ }` の内側）を
/// 空白連結で返す。宣言が `pub` 付きでちょうど 1 件でなければ `None`。
fn determinism_fn_body(tokens: &[String], fn_name: &str) -> Option<String> {
    let mut found: Option<String> = None;
    for (i, token) in tokens.iter().enumerate() {
        if token != "fn" || !fn_declaration_target_name_matches(tokens, i, fn_name) {
            continue;
        }
        if found.is_some() || i == 0 || tokens[i - 1] != "pub" {
            return None;
        }
        let open = (i..tokens.len()).find(|&j| tokens[j] == "{")?;
        let mut depth = 0usize;
        let mut close = None;
        for (j, t) in tokens.iter().enumerate().skip(open) {
            match t.as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(j);
                        break;
                    }
                }
                _ => {}
            }
        }
        found = Some(tokens[open + 1..close?].join(" "));
    }
    found
}

/// 承認形（イシュー #2507・2026-10-04 承認）の固定: [`DETERMINISM_FN_NAMES`]
/// の `fn` 宣言が `crates/facade/src/lib.rs` にちょうど 1 件ずつあり、
/// 他の facade src には無いこと、`fandhe_ai_autodiff::determinism::*` へ
/// 委譲していること、`determinism` を識別子に含む `pub use`（モジュール
/// 再エクスポート・別名）が無いことを固定する（旧否定ガードの反転）。
#[test]
fn facade_declares_determinism_fns_only_as_approved_root_delegations() {
    let src_dir = facade_crate_root().join("src");
    let lib_rs = lib_rs_path();
    let mut offending: Vec<String> = Vec::new();
    let mut root_counts: std::collections::BTreeMap<&str, usize> =
        DETERMINISM_FN_NAMES.iter().map(|n| (*n, 0usize)).collect();
    let mut root_seen = false;
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "determinism") {
                offending.push(format!(
                    "{}: `{trimmed}` が `determinism` を識別子単位で含む（承認形外）",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let is_root = path == lib_rs;
        if is_root {
            root_seen = true;
        }
        for fn_name in DETERMINISM_FN_NAMES {
            let count = count_fn_declarations_by_name(&tokens, fn_name);
            if is_root {
                *root_counts.entry(fn_name).or_insert(0) += count;
            } else if count > 0 {
                offending.push(format!(
                    "{}: `fn {fn_name}` 宣言が lib.rs 以外に {count} 件",
                    path.display()
                ));
            }
        }
        if is_root {
            // 各 `pub fn` の本体全体が、対応する内部関数への委譲 1 文だけで
            // あることを固定する（呼び出しが lib.rs のどこかにあるだけでは
            // 別関数へ移したり独自実装へ差し替えても通るため。codex-review
            // 指摘・PR #2699）。
            for (fn_name, expected_body) in DETERMINISM_FN_EXPECTED_BODIES {
                match determinism_fn_body(&tokens, fn_name) {
                    Some(body) if body == expected_body => {}
                    other => offending.push(format!(
                        "lib.rs: `{fn_name}` の本体が `fandhe_ai_autodiff::determinism::{fn_name}` への委譲のみではない（期待 `{expected_body}`・実際 {other:?}）"
                    )),
                }
            }
        }
    });
    assert!(
        root_seen,
        "facade src から lib.rs を見失った（fail-closed）: {}",
        lib_rs.display()
    );
    for (name, count) in &root_counts {
        if *count != 1 {
            offending.push(format!(
                "lib.rs: `fn {name}` 宣言が {count} 件（ちょうど 1 件であること）"
            ));
        }
    }
    assert!(
        offending.is_empty(),
        "facade の determinism 公開が承認形（イシュー #2507: crate ルートの\
         `pub fn` 2 件が `fandhe_ai_autodiff::determinism::*` へ委譲）から\
         逸脱している（モジュール再エクスポート・別名・独自実装は承認形外）: \
         {offending:?}"
    );
}

/// `fandhe_ai::{set_deterministic, is_deterministic}`（イシュー #2507）が
/// facade 経由で到達でき、委譲先の内部クレートと状態を共有することを
/// 固定する。同一バイナリ内の並列テストと競合しないよう LOCK で直列化し、
/// 元の値へ戻す。
#[test]
fn determinism_mode_is_reachable_via_facade() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let _: fn(bool) = fandhe_ai::set_deterministic;
    let _: fn() -> bool = fandhe_ai::is_deterministic;
    let original = fandhe_ai::is_deterministic();
    fandhe_ai::set_deterministic(true);
    assert!(fandhe_ai::is_deterministic());
    assert!(fandhe_ai_autodiff::determinism::is_deterministic());
    fandhe_ai::set_deterministic(original);
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`DETERMINISM_FN_NAMES`]
/// （2 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// reduce_ops_fn_names_only_in_allowed_locations` と同型のインベン
/// トリ）。
///
/// **期待集合**: `set_deterministic`・`is_deterministic` は
/// `crates/autodiff/src/determinism.rs` と `crates/facade/src/lib.rs`
/// （#2507 の委譲 `pub fn`）に 1 件ずつ存在する（計 4 エントリ）。
#[test]
fn workspace_declares_determinism_fn_names_only_in_allowed_locations() {
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
            for fn_name in DETERMINISM_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = DETERMINISM_FN_NAMES
        .iter()
        .flat_map(|name| {
            [
                (format!("autodiff/src/determinism.rs::{name}"), 1usize),
                (format!("facade/src/lib.rs::{name}"), 1usize),
            ]
        })
        .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の determinism 系 `fn` 宣言集合が\
         `autodiff/src/determinism.rs`・`facade/src/lib.rs`（各 2 件）のみという期待と\
         一致しない（過不足いずれも fail-closed に検出する。新たな定義元\
         が見つかった場合、それが承認済みの実装なのか迂回経路の混入\
         なのかを確認すること）: {found:?}"
    );
}

// =====================================================================
// #2156（親 #2131）の facade 公開保留固定（`RngDistributionsHoldDoctestGuard`）。
// `VarMatrixOpsHoldDoctestGuard`（#2144。#2513 で削除済み）と同型の正のプローブ 1
// ブロック方式のドリフト検査に加え、workspace 全体のソース走査による
// 定義元インベントリを持つ。承認事項・多層防御の位置づけは
// `docs/rng-distributions-generator-decision.md` 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `RngDistributionsHoldDoctestGuard` doc
/// 内の唯一の doctest ブロックが glob import するネスト `pub mod` 集合
/// と、`src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （旧 `matrix_ops_hold_doctest_globs_all_pub_modules`〈#2513 で削除済み〉 の
/// `RngDistributionsHoldDoctestGuard` 版）。
#[test]
fn rng_distributions_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "RngDistributionsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "RngDistributionsHoldDoctestGuard の doctest ブロックが glob\
         import するモジュール集合が src/lib.rs の pub mod 宣言集合と\
         ドリフトしている（declared={declared:?}, doctest={globbed:?}）。\
         新しい pub mod を追加した場合は doctest 側の use 一覧にも\
         追加すること。"
    );
}

/// [`rng_distributions_hold_doctest_globs_all_pub_modules`] が glob
/// import 集合の一致のみを固定するのに対し、本テストは doctest ブロック
/// の**glob 以外の本文**が固定文言 [`RNG_DISTRIBUTIONS_HOLD_PROBE_BODY`]
/// と 1 行たりとも違わず一致することを固定する（旧 `matrix_ops_hold_
/// doctest_probe_body_matches_fixed_contract`〈#2513 で削除済み〉と同じ理由: rustdoc の
/// `# ` 隠し行・プローブの削除・別名へのシャドーイング等で正のプローブ
/// を骨抜きにする改変を機械的に拒否する）。
#[test]
fn rng_distributions_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "RngDistributionsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, RNG_DISTRIBUTIONS_HOLD_PROBE_BODY,
        "RngDistributionsHoldDoctestGuard の doctest ブロック本文（glob\
         以外）が固定文言 RNG_DISTRIBUTIONS_HOLD_PROBE_BODY からドリフト\
         している。正のプローブ（__fandhe_rng_dist_hold_probe モジュール・\
         __FandheRngDistHoldProbe トレイト・__probe_* 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`rng_distributions_hold_doctest_probe_body_matches_fixed_contract`]
/// が要求する固定文言。`crates/facade/src/lib.rs` の
/// `RngDistributionsHoldDoctestGuard` doc 内の唯一の doctest ブロック
/// から、ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）
/// を除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体
/// の `use fandhe_ai::*;` は本文に含む）。
const RNG_DISTRIBUTIONS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_rng_dist_hold_probe {\n\
\x20\x20\x20\x20pub struct Generator;\n\
\x20\x20\x20\x20pub fn bernoulli() {}\n\
\x20\x20\x20\x20pub fn multinomial() {}\n\
}\n\
use __fandhe_rng_dist_hold_probe::*;\n\
\n\
struct __FandheRngDistMarker;\n\
\n\
trait __FandheRngDistHoldProbe {\n\
\x20\x20\x20\x20fn bernoulli(&self) -> __FandheRngDistMarker;\n\
\x20\x20\x20\x20fn multinomial(&self) -> __FandheRngDistMarker;\n\
\x20\x20\x20\x20fn normal(&self) -> __FandheRngDistMarker;\n\
}\n\
\n\
impl<'t> __FandheRngDistHoldProbe for fandhe_ai::Var<'t> {\n\
\x20\x20\x20\x20fn bernoulli(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }\n\
\x20\x20\x20\x20fn multinomial(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }\n\
\x20\x20\x20\x20fn normal(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }\n\
}\n\
\n\
impl __FandheRngDistHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn bernoulli(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }\n\
\x20\x20\x20\x20fn multinomial(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }\n\
\x20\x20\x20\x20fn normal(&self) -> __FandheRngDistMarker { __FandheRngDistMarker }\n\
}\n\
\n\
fn __probe_free_fns(_: Generator) {\n\
\x20\x20\x20\x20// 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して\n\
\x20\x20\x20\x20// いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。\n\
\x20\x20\x20\x20bernoulli();\n\
\x20\x20\x20\x20multinomial();\n\
}\n\
\n\
// `normal` だけは `nn::init::normal`（#2504 で公開済み）が同名の\n\
// 別機能として facade に存在するため、`nn::init` を除く全 `pub mod`\n\
// だけを glob したスコープで衝突検査する（`nn::init` を含めると\n\
// 常に曖昧になる）。\n\
mod __fandhe_rng_dist_normal_scope {\n\
\x20\x20\x20\x20use fandhe_ai::*;\n\
\n\
\x20\x20\x20\x20mod __fandhe_rng_dist_normal_probe {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn normal() {}\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20use __fandhe_rng_dist_normal_probe::*;\n\
\n\
\x20\x20\x20\x20pub fn __probe_normal() {\n\
\x20\x20\x20\x20\x20\x20\x20\x20normal();\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_var(x: &fandhe_ai::Var<'_>) {\n\
\x20\x20\x20\x20let _: __FandheRngDistMarker = fandhe_ai::Var::bernoulli(x);\n\
\x20\x20\x20\x20let _: __FandheRngDistMarker = x.bernoulli();\n\
\x20\x20\x20\x20let _: __FandheRngDistMarker = fandhe_ai::Var::normal(x);\n\
\x20\x20\x20\x20let _: __FandheRngDistMarker = x.normal();\n\
}\n\
\n\
fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheRngDistMarker = fandhe_ai::Tensor::multinomial(x);\n\
\x20\x20\x20\x20let _: __FandheRngDistMarker = x.multinomial();\n\
}";

/// `bernoulli`・`multinomial`・`normal`（3 個の関数名。イシュー #2156）。
/// `Generator`（型名）と合わせて
/// [`facade_does_not_reexport_or_declare_rng_distributions`]・
/// [`workspace_declares_rng_distribution_names_only_in_allowed_locations`]
/// が共用する。
const RNG_DISTRIBUTIONS_FN_NAMES: [&str; 3] = ["bernoulli", "multinomial", "normal"];

/// [`facade_does_not_reexport_or_declare_rng_distributions`]・その自己
/// テストが共用する検出本体。facade src 全体（`crates/facade/src/**`）
/// の `pub use` から [`collect_pub_use_leaves`] で別名にする前の葉を
/// 集め `Generator` を検出し（単一行・複数行・ネストした group・別名も
/// 検出）、`trait`／`struct`／`enum`／`type` 直後の `Generator` 独自
/// 宣言、[`RNG_DISTRIBUTIONS_FN_NAMES`]（3 個）の `fn` 宣言（可視性・
/// 宣言文脈を問わない。[`count_fn_declarations_by_name`] と同じ検出
/// 契約）を違反として返す（`scan_kv_cache_reexports_and_declarations`
/// と同型）。
fn scan_rng_distributions_reexports_and_declarations(
    content: &str,
    is_nn_init_rs: bool,
) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            // #2504: `nn::init::normal`（`torch.nn.init.normal_` 相当。RNG 分布 `normal`
            // とは別機能）の承認済み再エクスポートだけを経路限定で許可する
            // （`src/nn/init.rs`・接頭辞 `fandhe_ai_autodiff::nn::init::`・別名なし）。
            let nn_init_shape_ok = is_nn_init_rs
                && nn_init_approved_prefix(path_tokens)
                && !path_tokens.iter().any(|t| t == "as");
            for leaf in leaves {
                if leaf == "normal" && nn_init_shape_ok {
                    continue;
                }
                if leaf == "Generator" || RNG_DISTRIBUTIONS_FN_NAMES.contains(&leaf.as_str()) {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens.get(i + 1).map(String::as_str) == Some("Generator")
        {
            offending.push(format!("{} Generator 宣言", tokens[i]));
        }
        i += 1;
    }

    for fn_name in RNG_DISTRIBUTIONS_FN_NAMES {
        let count = count_fn_declarations_by_name(&tokens, fn_name);
        if count > 0 {
            offending.push(format!("`fn {fn_name}` 宣言が {count} 件"));
        }
    }
    offending
}

/// `pub use` の path トークン列が `fandhe_ai_autodiff :: nn :: init :: …`
/// で始まるか（`nn::init` の承認形の接頭辞）を判定する
/// （[`optimizer_ext_approved_prefix`] と同型）。
fn nn_init_approved_prefix(path_tokens: &[String]) -> bool {
    let want = [
        "fandhe_ai_autodiff",
        ":",
        ":",
        "nn",
        ":",
        ":",
        "init",
        ":",
        ":",
    ];
    path_tokens.len() >= want.len() && path_tokens.iter().zip(want).all(|(a, b)| a == b)
}

/// [`scan_rng_distributions_reexports_and_declarations`] の自己テスト。
/// `normal` の承認済み `nn::init` 再エクスポートだけが経路限定で許可され、
/// それ以外（別ファイル・別 path・別名・RNG 分布の `normal`）は違反になる。
#[test]
fn scan_rng_distributions_allows_only_approved_nn_init_normal() {
    let scan = scan_rng_distributions_reexports_and_declarations;
    // 正例: src/nn/init.rs の承認形（group・単一とも）。
    assert!(
        scan(
            "pub use fandhe_ai_autodiff::nn::init::{constant, normal, uniform};",
            true
        )
        .is_empty()
    );
    assert!(scan("pub use fandhe_ai_autodiff::nn::init::normal;", true).is_empty());
    // 負例: RNG 分布の `normal`（tensor_core::rng）は src/nn/init.rs でも違反。
    assert!(!scan("pub use fandhe_ai_tensor_core::rng::normal;", true).is_empty());
    // 負例: 承認形でも nn/init.rs 以外のファイルからは違反。
    assert!(!scan("pub use fandhe_ai_autodiff::nn::init::normal;", false).is_empty());
    // 負例: 別名。
    assert!(!scan("pub use fandhe_ai_autodiff::nn::init::normal as n;", true).is_empty());
    // 負例: 他の RNG 分布名・Generator は nn/init.rs でも違反のまま。
    assert!(!scan("pub use fandhe_ai_autodiff::nn::init::bernoulli;", true).is_empty());
    assert!(!scan("pub use fandhe_ai_tensor_core::rng::Generator;", true).is_empty());
    // 負例: fn 宣言は従来どおり違反。
    assert!(!scan("pub fn normal() {}", true).is_empty());
}

/// `RngDistributionsHoldDoctestGuard` の入れ子スコープ
/// `__fandhe_rng_dist_normal_scope`（`nn::init` 衝突の回避用。#2504）が
/// glob する集合が「`pub mod` 全集合から `nn::init` だけを除いたもの」と
/// 完全一致し、ローカルの `normal` プローブと `normal();` 呼び出しを
/// 保つことを固定する（`split_glob_imports_and_probe_body` は入れ子内の
/// glob 行も外側と同じ集合へ吸収するため、別途検査が必要）。
#[test]
fn rng_distributions_normal_scope_globs_all_pub_modules_except_nn_init() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "RngDistributionsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let start = block
        .iter()
        .position(|l| l.trim() == "mod __fandhe_rng_dist_normal_scope {")
        .expect("入れ子スコープ __fandhe_rng_dist_normal_scope が見つからない");
    let mut globs = std::collections::BTreeSet::new();
    let mut inner: Vec<String> = Vec::new();
    for line in &block[start + 1..] {
        if line.trim() == "}" && !line.starts_with(' ') {
            break;
        }
        let t = line.trim();
        if let Some(path) = t
            .strip_prefix("use fandhe_ai::")
            .and_then(|r| r.strip_suffix("::*;"))
            && !path.is_empty()
        {
            globs.insert(path.to_string());
        }
        inner.push(t.to_string());
    }
    let mut expected = declared;
    assert!(
        expected.remove("nn::init"),
        "nn::init が pub mod 集合にない"
    );
    assert_eq!(
        globs, expected,
        "入れ子スコープの glob 集合が `pub mod` 全集合から nn::init を除いたものと不一致"
    );
    for needle in [
        "use fandhe_ai::*;",
        "pub fn normal() {}",
        "use __fandhe_rng_dist_normal_probe::*;",
        "normal();",
    ] {
        assert!(
            inner.iter().any(|l| l == needle),
            "入れ子スコープに `{needle}` がない（正のプローブの骨抜き）: {inner:?}"
        );
    }
    assert!(
        !inner.iter().any(|l| l.contains("nn::init")),
        "入れ子スコープが nn::init を glob している（常に曖昧になりプローブが無意味化する）"
    );
}

/// facade src 全体（`crates/facade/src/**`）に、`Generator` を識別子
/// 単位で含む `pub use`（複数行・ネストした group・別名含む）も、
/// facade 独自の `trait`／`struct`／`enum`／`type` 宣言も、
/// [`RNG_DISTRIBUTIONS_FN_NAMES`]（`bernoulli`／`multinomial`／
/// `normal`）の `fn` 宣言も存在しないことを固定する
/// （`RngDistributionsHoldDoctestGuard` の正のプローブと多層防御を成す
/// 最内層のソース走査ガード。`facade_does_not_reexport_or_declare_kv_
/// cache_items` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_rng_distributions() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let is_nn_init_rs = path.ends_with("src/nn/init.rs");
        for offense in scan_rng_distributions_reexports_and_declarations(content, is_nn_init_rs) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が RNG 確率分布サンプラー（#2156 の\
         `bernoulli`／`multinomial`／`normal`／`Generator`。内部クレート\
         限定の新規公開面。facade 公開は承認待ちのため対象外という設計\
         判断に違反）を再エクスポート、独自宣言、または同名の fn を\
         宣言している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`RNG_DISTRIBUTIONS_FN_NAMES`]（3 個）の `fn` 宣言の定義元集合を
/// 固定する（`workspace_declares_matrix_ops_fn_names_only_in_autodiff_
/// matrix_ops` と同型のインベントリ）。
///
/// **期待集合**（着手前確認の再 grep で `crates/autodiff/src/nn/
/// init.rs:615` の既存 `pub fn normal`〈1 件〉のみが見つかった。実装後
/// の実測で `crates/tensor-core/src/rng.rs` に各名 2 件〈自由関数 1 件 +
/// `Generator` の同名メソッド 1 件〉が追加された）:
/// - `autodiff/src/nn/init.rs::normal` = 1（既存。無変更）
/// - `tensor-core/src/rng.rs::bernoulli` = 2
/// - `tensor-core/src/rng.rs::multinomial` = 2
/// - `tensor-core/src/rng.rs::normal` = 2
#[test]
fn workspace_declares_rng_distribution_names_only_in_allowed_locations() {
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
            for fn_name in RNG_DISTRIBUTIONS_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("autodiff/src/nn/init.rs::normal".to_string(), 1usize),
        ("tensor-core/src/rng.rs::bernoulli".to_string(), 2usize),
        ("tensor-core/src/rng.rs::multinomial".to_string(), 2usize),
        ("tensor-core/src/rng.rs::normal".to_string(), 2usize),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の bernoulli／multinomial／normal\
         `fn` 宣言集合が期待（autodiff/src/nn/init.rs::normal 1 件・\
         tensor-core/src/rng.rs の各名 2 件〈自由関数 + Generator\
         メソッド〉）と一致しない（過不足いずれも fail-closed に検出\
         する。新たな定義元が見つかった場合、それが承認済みの実装なのか\
         迂回経路の混入なのかを確認すること）: {found:?}"
    );
}

/// `SpatialLayersHoldDoctestGuard` の唯一の doctest ブロックが glob
/// import するネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod`
/// 宣言集合が一致することを固定する（`kv_cache_hold_doctest_globs_all_
/// pub_modules` と同型。新しい `pub mod` を facade へ追加した際、
/// doctest 側の `use` 一覧の更新を機械的に強制する。イシュー #2159）。
#[test]
fn spatial_layers_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "SpatialLayersHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "SpatialLayersHoldDoctestGuard の doctest ブロックが glob import\
         するモジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加する\
         こと。"
    );
}

/// [`spatial_layers_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**（`__fandhe_spatial_hold_probe` モジュール・
/// `__probe` 関数。#2521・#2522 で trait 経由プローブは除去済み）が固定文言
/// [`SPATIAL_LAYERS_HOLD_PROBE_BODY`] と 1 行たりとも違わず一致する
/// ことを固定する（rustdoc の `# ` 隠し行・プローブの削除・別名への
/// シャドーイング等で正のプローブを骨抜きにする改変を機械的に拒否
/// する。イシュー #2159）。
#[test]
fn spatial_layers_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "SpatialLayersHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, SPATIAL_LAYERS_HOLD_PROBE_BODY,
        "SpatialLayersHoldDoctestGuard の doctest ブロック本文（glob\
         以外）が固定文言 SPATIAL_LAYERS_HOLD_PROBE_BODY からドリフト\
         している。正のプローブ（__fandhe_spatial_hold_probe モジュール・\
         __probe 関数）の削除・弱体化・隠し行の混入がないか確認する\
         こと。"
    );
}

/// [`spatial_layers_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `SpatialLayersHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const SPATIAL_LAYERS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_spatial_hold_probe {\n\
\x20\x20\x20\x20pub struct ConvTranspose1d;\n\
\x20\x20\x20\x20pub struct Upsample;\n\
\x20\x20\x20\x20pub struct ZeroPad2d;\n\
\x20\x20\x20\x20pub struct Identity;\n\
\x20\x20\x20\x20pub struct Unflatten;\n\
\x20\x20\x20\x20pub struct __FandheSpatialHoldMarker;\n\
\x20\x20\x20\x20pub fn add_conv_transpose1d() -> __FandheSpatialHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheSpatialHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20pub fn add_unflatten() -> __FandheSpatialHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheSpatialHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_spatial_hold_probe::*;\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20_: ConvTranspose1d,\n\
\x20\x20\x20\x20_: Upsample,\n\
\x20\x20\x20\x20_: ZeroPad2d,\n\
\x20\x20\x20\x20_: Identity,\n\
\x20\x20\x20\x20_: Unflatten,\n\
) {\n\
\x20\x20\x20\x20let _: __FandheSpatialHoldMarker = add_conv_transpose1d();\n\
\x20\x20\x20\x20let _: __FandheSpatialHoldMarker = add_unflatten();\n\
}";

// =====================================================================
// イシュー #2521（親 #2520・ルート #2499 の一括承認）: `ConvTranspose1d`／
// `Unflatten` の facade 公開の正ガード。保留ガード
// （`SpatialLayersHoldDoctestGuard`）から該当 2 層の add_*／Var メソッドの
// プローブを外し、承認形（`Var` の 1 行委譲メソッド 2 個・`compat::Sequential`
// の add_* 2 個）だけを許す形へ反転した。型の再エクスポートと自由関数での
// 公開は保留ガードの衝突プローブで引き続き禁止する。承認事項は
// `docs/autodiff-spatial-layers-decision.md` §6 参照。
// =====================================================================

/// #2521 で公開する 4 名の `fn` 名（`Var` 2 個 + `Sequential` 2 個）。
const SPATIAL_FACADE_FN_NAMES: [&str; 4] = [
    "conv_transpose1d",
    "unflatten",
    "add_conv_transpose1d",
    "add_unflatten",
];

/// workspace 全体（`crates/*/src/`）で [`SPATIAL_FACADE_FN_NAMES`] の `fn`
/// 宣言の定義元集合を固定する（過不足とも fail-closed。迂回実装の混入検出）。
#[test]
fn workspace_declares_spatial_facade_fn_names_only_in_approved_locations() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut crate_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("workspace crates ディレクトリが読めない")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(!crate_dirs.is_empty());
    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in SPATIAL_FACADE_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }
    let expected: std::collections::BTreeMap<String, usize> = [
        "autodiff/src/var.rs::conv_transpose1d",
        "autodiff/src/var.rs::unflatten",
        "facade/src/compat/sequential.rs::add_conv_transpose1d",
        "facade/src/compat/sequential.rs::add_unflatten",
    ]
    .iter()
    .map(|k| (k.to_string(), 1usize))
    .collect();
    assert_eq!(
        found, expected,
        "#2521 の 4 名の fn 宣言の定義元集合が承認形とずれている（新たな定義元が\
         承認済みの実装なのか迂回経路なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 2 メソッド本体の承認形（`nn` 側共有 forward への 1 行委譲）。
const SPATIAL_VAR_EXPECTED_BODIES: [(&str, &str); 2] = [
    (
        "conv_transpose1d",
        "crate : : nn : : conv_transpose1d_forward ( self , weight , bias , stride , padding , output_padding , dilation , groups , )",
    ),
    (
        "unflatten",
        "crate : : nn : : unflatten_forward ( self , dim , sizes )",
    ),
];

/// `Var::conv_transpose1d`／`Var::unflatten` が共有 forward への薄い委譲で
/// あることを固定する（スタブ・独自実装へのすり替えを拒否）。
#[test]
fn var_spatial_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in SPATIAL_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（共有 forward への 1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで 2 メソッドへ到達でき、シグネチャが
/// 承認形と一致し、実際に適用して期待 shape・値が得られることを固定する。
#[test]
fn var_spatial_methods_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type ConvTSig<'t> = fn(
        &Var<'t>,
        &Var<'t>,
        Option<&Var<'t>>,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) -> Result<Var<'t>, AutodiffError>;
    fn sig_conv_t<'t>() -> ConvTSig<'t> {
        Var::<'t>::conv_transpose1d
    }
    fn sig_unflatten<'t>() -> fn(&Var<'t>, usize, &[usize]) -> Result<Var<'t>, AutodiffError> {
        Var::<'t>::unflatten
    }

    let tape = fandhe_ai::tape();
    // x: [1, 1, 3] = [1, 2, 3]、w: [1, 1, 2] = [1, 1]、stride 1 → [1, 1, 4]
    let x = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0], &[1, 1, 3]).expect("tensor"));
    let w = tape.var(&Tensor::new(vec![1.0_f32, 1.0], &[1, 1, 2]).expect("tensor"));
    let y = sig_conv_t()(&x, &w, None, 1, 0, 0, 1, 1).expect("conv_transpose1d");
    assert_eq!(y.to_tensor().shape(), [1, 1, 4]);
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [1.0, 3.0, 5.0, 3.0]
    );

    let flat = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]).expect("t"));
    let u = sig_unflatten()(&flat, 1, &[2, 3]).expect("unflatten");
    assert_eq!(u.to_tensor().shape(), [1, 2, 3]);
    assert!(sig_unflatten()(&flat, 1, &[]).is_err());
    assert!(sig_unflatten()(&flat, 1, &[4, 2]).is_err());
}

/// `Sequential::add_conv_transpose1d`／`add_unflatten` の宣言部（`pub fn` から
/// 本体開始 `{` まで）が承認形か判定する。
fn sequential_spatial_add_signature_ok(cleaned: &str, name: &str, expected_params: &str) -> bool {
    let needle = format!("pub fn {name}");
    let Some(start) = cleaned.find(&needle) else {
        return false;
    };
    let Some(len) = cleaned[start..].find('{') else {
        return false;
    };
    let sig: String = cleaned[start..start + len]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let want: String = format!("pub fn {name}({expected_params}")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    // rustfmt の末尾カンマ有無に依存しないよう `,)` を `)` へ正規化して比較する。
    sig.replace(",)", ")") == want.replace(",)", ")")
}

const ADD_CONV_TRANSPOSE1D_PARAMS: &str = "mut self, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, output_padding: usize, dilation: usize, groups: usize, seed: u64, ) -> Result<Self, AutodiffError>";
const ADD_UNFLATTEN_PARAMS: &str =
    "mut self, dim: usize, unflattened_size: Vec<usize>, ) -> Result<Self, AutodiffError>";

/// `compat::Sequential` の 2 add_* が承認シグネチャで 1 件ずつ存在する。
#[test]
fn compat_sequential_spatial_add_methods_have_approved_signatures() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(
        count_fn_declarations_by_name(&tokens, "add_conv_transpose1d"),
        1
    );
    assert_eq!(count_fn_declarations_by_name(&tokens, "add_unflatten"), 1);
    assert!(sequential_spatial_add_signature_ok(
        &cleaned,
        "add_conv_transpose1d",
        ADD_CONV_TRANSPOSE1D_PARAMS
    ));
    assert!(sequential_spatial_add_signature_ok(
        &cleaned,
        "add_unflatten",
        ADD_UNFLATTEN_PARAMS
    ));
}

/// [`compat_sequential_spatial_add_methods_have_approved_signatures`] の自己テスト。
#[test]
fn compat_sequential_spatial_add_methods_have_approved_signatures_detects_offense() {
    let ok = "pub fn add_unflatten(mut self, dim: usize, unflattened_size: Vec<usize>,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok,
        "add_unflatten",
        ADD_UNFLATTEN_PARAMS
    ));
    for bad in [
        "pub fn add_unflatten(mut self, dim: usize, unflattened_size: Vec<usize>) -> Self {",
        "pub fn add_unflatten(mut self, dim: i64, unflattened_size: Vec<usize>,) -> Result<Self, AutodiffError> {",
        "pub fn add_unflatten(mut self, unflattened_size: Vec<usize>, dim: usize,) -> Result<Self, AutodiffError> {",
        "pub fn add_linear(mut self) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(bad, "add_unflatten", ADD_UNFLATTEN_PARAMS),
            "{bad}"
        );
    }
}

/// 承認済みの 3 メソッド名（イシュー #2522・ルート #2499 の 2026-10-04 一括承認。
/// `docs/autodiff-spatial-layers-decision.md` §6）。
const SPATIAL_LAYER_APPROVED_ADD_METHODS: [&str; 3] =
    ["add_upsample", "add_zero_pad2d", "add_identity"];

/// `content` 内の `pub fn {name}` 宣言の件数（先頭空白・`pub fn name(` 形の行のみ数える）。
fn count_pub_fn_declarations(content: &str, name: &str) -> usize {
    let needle = format!("pub fn {name}(");
    content
        .lines()
        .filter(|l| l.trim_start().starts_with(&needle))
        .count()
}

/// `src/compat` 配下で承認済み 3 メソッドがそれぞれちょうど 1 件の `pub fn`
/// として宣言されていることを fail-closed で固定する正ガード（イシュー #2522。
/// 0 件＝公開の脱落、2 件以上＝重複宣言の混入を拒否する）。
#[test]
fn compat_sequential_exposes_spatial_layer_add_methods_issue_2522() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut counts = [0usize; 3];
    visit_rs_files(&compat_dir, &mut |_path, content| {
        for (i, name) in SPATIAL_LAYER_APPROVED_ADD_METHODS.iter().enumerate() {
            counts[i] += count_pub_fn_declarations(content, name);
        }
    });
    assert_eq!(
        counts,
        [1, 1, 1],
        "src/compat 配下の add_upsample／add_zero_pad2d／add_identity の pub fn 宣言数が\
         各 1 件でない（counts={counts:?}）"
    );
}

/// [`compat_sequential_exposes_spatial_layer_add_methods_issue_2522`] の
/// 自己テスト（合成入力で 1 件・0 件・重複を判別できることを確認する）。
#[test]
fn compat_sequential_exposes_spatial_layer_add_methods_issue_2522_counts_declarations() {
    let one = "    pub fn add_identity(mut self) -> Self {\n        self\n    }\n";
    assert_eq!(count_pub_fn_declarations(one, "add_identity"), 1);
    assert_eq!(count_pub_fn_declarations(one, "add_upsample"), 0);
    let dup = format!("{one}{one}");
    assert_eq!(count_pub_fn_declarations(&dup, "add_identity"), 2);
    assert_eq!(
        count_pub_fn_declarations("// pub fn add_identity() {}", "add_identity"),
        0
    );
}

// =====================================================================
// イシュー #2529（親 #2520・ルート #2499 の 2026-10-04 一括承認）:
// `compat::Sequential::add_mish`／`add_hardtanh`／`add_relu6`／`add_glu`／`add_prelu` の
// 正ガード。旧 `VarActivationOpsHoldDoctestGuard` の (b) `add_*` 衝突プローブを撤去し、
// 承認形だけを許す正ガード（シグネチャ一致・`pub fn` ちょうど 1 件）へ反転した。
// `activation_ops` のモジュール再エクスポート・`Tensor<f32>`／`Tape` 上への配置を拒む
// 保留ガードは未承認のまま残す。`docs/autodiff-activation-ops-decision.md` §9 参照。
// =====================================================================

const ADD_MISH_PARAMS: &str = "mut self) -> Self";
const ADD_HARDTANH_PARAMS: &str =
    "mut self, min_val: f32, max_val: f32, ) -> Result<Self, AutodiffError>";
const ADD_RELU6_PARAMS: &str = "mut self) -> Self";
const ADD_GLU_PARAMS: &str = "mut self, dim: usize) -> Self";
const ADD_PRELU_PARAMS: &str =
    "mut self, num_parameters: usize, init: f32, ) -> Result<Self, AutodiffError>";

const ACTIVATION_LAYER_ADD_METHODS: [(&str, &str); 5] = [
    ("add_mish", ADD_MISH_PARAMS),
    ("add_hardtanh", ADD_HARDTANH_PARAMS),
    ("add_relu6", ADD_RELU6_PARAMS),
    ("add_glu", ADD_GLU_PARAMS),
    ("add_prelu", ADD_PRELU_PARAMS),
];

/// `compat::Sequential` の 5 add_* が承認シグネチャで 1 件ずつ存在する。
#[test]
fn compat_sequential_activation_layers_add_methods_have_approved_signatures() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, params) in ACTIVATION_LAYER_ADD_METHODS {
        assert_eq!(count_fn_declarations_by_name(&tokens, name), 1, "{name}");
        assert!(
            sequential_spatial_add_signature_ok(&cleaned, name, params),
            "{name} のシグネチャが承認形と一致しない"
        );
    }
}

/// [`compat_sequential_activation_layers_add_methods_have_approved_signatures`] の自己テスト。
#[test]
fn compat_sequential_activation_layers_add_methods_have_approved_signatures_detects_offense() {
    for (ok, name, params) in [
        (
            "pub fn add_mish(mut self) -> Self {",
            "add_mish",
            ADD_MISH_PARAMS,
        ),
        (
            "pub fn add_hardtanh(mut self, min_val: f32, max_val: f32,) -> Result<Self, AutodiffError> {",
            "add_hardtanh",
            ADD_HARDTANH_PARAMS,
        ),
        (
            "pub fn add_relu6(mut self) -> Self {",
            "add_relu6",
            ADD_RELU6_PARAMS,
        ),
        (
            "pub fn add_glu(mut self, dim: usize) -> Self {",
            "add_glu",
            ADD_GLU_PARAMS,
        ),
        (
            "pub fn add_prelu(mut self, num_parameters: usize, init: f32,) -> Result<Self, AutodiffError> {",
            "add_prelu",
            ADD_PRELU_PARAMS,
        ),
    ] {
        assert!(
            sequential_spatial_add_signature_ok(ok, name, params),
            "{ok}"
        );
    }
    for (bad, name, params) in [
        // 引数の追加
        (
            "pub fn add_mish(mut self, beta: f32) -> Self {",
            "add_mish",
            ADD_MISH_PARAMS,
        ),
        // 戻り値が Result
        (
            "pub fn add_relu6(mut self) -> Result<Self, AutodiffError> {",
            "add_relu6",
            ADD_RELU6_PARAMS,
        ),
        // 引数の欠落
        (
            "pub fn add_hardtanh(mut self, min_val: f32,) -> Result<Self, AutodiffError> {",
            "add_hardtanh",
            ADD_HARDTANH_PARAMS,
        ),
        // 引数順の入替
        (
            "pub fn add_hardtanh(mut self, max_val: f32, min_val: f32,) -> Result<Self, AutodiffError> {",
            "add_hardtanh",
            ADD_HARDTANH_PARAMS,
        ),
        // 型違い（dim: isize）
        (
            "pub fn add_glu(mut self, dim: isize) -> Self {",
            "add_glu",
            ADD_GLU_PARAMS,
        ),
        // 戻り値が Self
        (
            "pub fn add_prelu(mut self, num_parameters: usize, init: f32,) -> Self {",
            "add_prelu",
            ADD_PRELU_PARAMS,
        ),
        // init の欠落
        (
            "pub fn add_prelu(mut self, num_parameters: usize,) -> Result<Self, AutodiffError> {",
            "add_prelu",
            ADD_PRELU_PARAMS,
        ),
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(bad, name, params),
            "{bad}"
        );
    }
}

/// `src/compat` 配下で 5 メソッドがそれぞれちょうど 1 件の `pub fn` として宣言されている
/// （0 件＝公開の脱落、2 件以上＝重複宣言の混入を拒否する正ガード）。
#[test]
fn compat_sequential_exposes_activation_layers_add_methods_issue_2529() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut counts = [0usize; 5];
    visit_rs_files(&compat_dir, &mut |_path, content| {
        for (i, (name, _)) in ACTIVATION_LAYER_ADD_METHODS.iter().enumerate() {
            counts[i] += count_pub_fn_declarations(content, name);
        }
    });
    assert_eq!(
        counts,
        [1, 1, 1, 1, 1],
        "src/compat 配下の add_mish／add_hardtanh／add_relu6／add_glu／add_prelu の pub fn 宣言数が\
         各 1 件でない（counts={counts:?}）"
    );
}

/// [`compat_sequential_exposes_activation_layers_add_methods_issue_2529`] の自己テスト。
#[test]
fn compat_sequential_exposes_activation_layers_add_methods_issue_2529_counts_declarations() {
    let one = "    pub fn add_mish(mut self) -> Self {\n        self\n    }\n";
    assert_eq!(count_pub_fn_declarations(one, "add_mish"), 1);
    assert_eq!(count_pub_fn_declarations(one, "add_relu6"), 0);
    let dup = format!("{one}{one}");
    assert_eq!(count_pub_fn_declarations(&dup, "add_mish"), 2);
    assert_eq!(
        count_pub_fn_declarations("// pub fn add_mish() {}", "add_mish"),
        0
    );
}

// =====================================================================
// イシュー #2523（親 #2520・ルート #2499 の 2026-10-04 一括承認）:
// `compat::Sequential::add_conv_transpose2d` の正ガード。保留ガードは存在しなかった
// （`ConvTranspose2d` を禁じるプローブ・否定テストなし）ため反転対象はなく、承認形
// だけを許す正ガード（シグネチャ一致・`pub fn` ちょうど 1 件）を新設する。
// `docs/conv-ops-design.md` §15「#2523 実装記録」参照。
// =====================================================================

const ADD_CONV_TRANSPOSE2D_PARAMS: &str = "mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], output_padding: [usize; 2], dilation: [usize; 2], groups: usize, seed: u64, ) -> Result<Self, AutodiffError>";

/// `compat::Sequential::add_conv_transpose2d` が承認シグネチャで 1 件だけ存在する。
#[test]
fn compat_sequential_conv_transpose2d_add_method_has_approved_signature() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(
        count_fn_declarations_by_name(&tokens, "add_conv_transpose2d"),
        1
    );
    assert!(sequential_spatial_add_signature_ok(
        &cleaned,
        "add_conv_transpose2d",
        ADD_CONV_TRANSPOSE2D_PARAMS
    ));
}

/// [`compat_sequential_conv_transpose2d_add_method_has_approved_signature`] の自己テスト。
#[test]
fn compat_sequential_conv_transpose2d_add_method_has_approved_signature_detects_offense() {
    let ok = "pub fn add_conv_transpose2d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], output_padding: [usize; 2], dilation: [usize; 2], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok,
        "add_conv_transpose2d",
        ADD_CONV_TRANSPOSE2D_PARAMS
    ));
    for bad in [
        // output_padding 欠落
        "pub fn add_conv_transpose2d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {",
        // 引数順の入替（output_padding と dilation）
        "pub fn add_conv_transpose2d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2], output_padding: [usize; 2], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {",
        // 型違い（スカラー）
        "pub fn add_conv_transpose2d(mut self, in_channels: usize, out_channels: usize, kernel_size: usize, stride: [usize; 2], padding: [usize; 2], output_padding: [usize; 2], dilation: [usize; 2], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {",
        // 戻り値が Self
        "pub fn add_conv_transpose2d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], output_padding: [usize; 2], dilation: [usize; 2], groups: usize, seed: u64,) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(
                bad,
                "add_conv_transpose2d",
                ADD_CONV_TRANSPOSE2D_PARAMS
            ),
            "{bad}"
        );
    }
}

/// `src/compat` 配下で `pub fn add_conv_transpose2d(` がちょうど 1 件（0 件＝公開の脱落、
/// 2 件以上＝重複宣言の混入を拒否する正ガード）。
#[test]
fn compat_sequential_exposes_conv_transpose2d_add_method_issue_2523() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut count = 0usize;
    visit_rs_files(&compat_dir, &mut |_path, content| {
        count += count_pub_fn_declarations(content, "add_conv_transpose2d");
    });
    assert_eq!(
        count, 1,
        "src/compat 配下の add_conv_transpose2d の pub fn 宣言数が 1 件でない（count={count}）"
    );
}

/// [`compat_sequential_exposes_conv_transpose2d_add_method_issue_2523`] の自己テスト。
#[test]
fn compat_sequential_exposes_conv_transpose2d_add_method_issue_2523_counts_declarations() {
    let one = "    pub fn add_conv_transpose2d(mut self) -> Self {\n        self\n    }\n";
    assert_eq!(count_pub_fn_declarations(one, "add_conv_transpose2d"), 1);
    assert_eq!(count_pub_fn_declarations(one, "add_conv_transpose1d"), 0);
    let dup = format!("{one}{one}");
    assert_eq!(count_pub_fn_declarations(&dup, "add_conv_transpose2d"), 2);
}

// =====================================================================
// イシュー #2525（親 #2520・ルート #2499 の 2026-10-04 一括承認）:
// `compat::Sequential::add_group_norm`／`add_instance_norm` の正ガード。保留ガードは
// 存在しなかった（`GroupNorm`／`InstanceNorm` を禁じるプローブ・否定テストなし）ため
// 反転対象はなく、承認形だけを許す正ガード（シグネチャ一致・`pub fn` ちょうど 1 件）を
// 新設する。`num_channels` 引数・affine 引数は承認形に含まれない。
// `docs/norm-ops-design.md` §11「#2525 実装記録」参照。
// =====================================================================

const ADD_GROUP_NORM_PARAMS: &str =
    "mut self, groups: usize, eps: f32, ) -> Result<Self, AutodiffError>";
const ADD_INSTANCE_NORM_PARAMS: &str = "mut self, eps: f32, ) -> Result<Self, AutodiffError>";

/// `compat::Sequential::add_group_norm`／`add_instance_norm` が承認シグネチャで 1 件ずつ存在する。
#[test]
fn compat_sequential_group_instance_norm_add_methods_have_approved_signatures() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, params) in [
        ("add_group_norm", ADD_GROUP_NORM_PARAMS),
        ("add_instance_norm", ADD_INSTANCE_NORM_PARAMS),
    ] {
        assert_eq!(count_fn_declarations_by_name(&tokens, name), 1, "{name}");
        assert!(
            sequential_spatial_add_signature_ok(&cleaned, name, params),
            "{name} のシグネチャが承認形と一致しない"
        );
    }
}

/// [`compat_sequential_group_instance_norm_add_methods_have_approved_signatures`] の自己テスト。
#[test]
fn compat_sequential_group_instance_norm_add_methods_have_approved_signatures_detects_offense() {
    let ok_g = "pub fn add_group_norm(mut self, groups: usize, eps: f32,) -> Result<Self, AutodiffError> {";
    let ok_i = "pub fn add_instance_norm(mut self, eps: f32,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok_g,
        "add_group_norm",
        ADD_GROUP_NORM_PARAMS
    ));
    assert!(sequential_spatial_add_signature_ok(
        ok_i,
        "add_instance_norm",
        ADD_INSTANCE_NORM_PARAMS
    ));
    for bad in [
        // groups 欠落
        "pub fn add_group_norm(mut self, eps: f32,) -> Result<Self, AutodiffError> {",
        // num_channels の追加
        "pub fn add_group_norm(mut self, groups: usize, num_channels: usize, eps: f32,) -> Result<Self, AutodiffError> {",
        // 型違い
        "pub fn add_group_norm(mut self, groups: usize, eps: f64,) -> Result<Self, AutodiffError> {",
        // 戻り値が Self
        "pub fn add_group_norm(mut self, groups: usize, eps: f32,) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(bad, "add_group_norm", ADD_GROUP_NORM_PARAMS),
            "{bad}"
        );
    }
    for bad in [
        // 引数欠落
        "pub fn add_instance_norm(mut self) -> Result<Self, AutodiffError> {",
        // affine 引数の追加
        "pub fn add_instance_norm(mut self, eps: f32, affine: bool,) -> Result<Self, AutodiffError> {",
        // 戻り値が Self
        "pub fn add_instance_norm(mut self, eps: f32,) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(
                bad,
                "add_instance_norm",
                ADD_INSTANCE_NORM_PARAMS
            ),
            "{bad}"
        );
    }
}

/// `src/compat` 配下で `pub fn add_group_norm(`／`add_instance_norm(` が各ちょうど 1 件
/// （0 件＝公開の脱落、2 件以上＝重複宣言の混入を拒否する正ガード）。
#[test]
fn compat_sequential_exposes_group_instance_norm_add_methods_issue_2525() {
    let compat_dir = facade_crate_root().join("src/compat");
    let (mut group, mut instance) = (0usize, 0usize);
    visit_rs_files(&compat_dir, &mut |_path, content| {
        group += count_pub_fn_declarations(content, "add_group_norm");
        instance += count_pub_fn_declarations(content, "add_instance_norm");
    });
    assert_eq!(group, 1, "add_group_norm の pub fn 宣言数が 1 件でない");
    assert_eq!(
        instance, 1,
        "add_instance_norm の pub fn 宣言数が 1 件でない"
    );
}

/// [`compat_sequential_exposes_group_instance_norm_add_methods_issue_2525`] の自己テスト。
#[test]
fn compat_sequential_exposes_group_instance_norm_add_methods_issue_2525_counts_declarations() {
    let one = "    pub fn add_group_norm(mut self) -> Self {\n        self\n    }\n";
    assert_eq!(count_pub_fn_declarations(one, "add_group_norm"), 1);
    assert_eq!(count_pub_fn_declarations(one, "add_instance_norm"), 0);
    let dup = format!("{one}{one}");
    assert_eq!(count_pub_fn_declarations(&dup, "add_group_norm"), 2);
}

// =====================================================================
// イシュー #2162（親 #2131）: PixelShuffle・PixelUnshuffle の facade
// 公開保留（#2526 で add_*／Var メソッドは公開済みに反転、型・自由関数のみ保留継続）を
// 検査するテスト群。`PixelShuffleHoldDoctestGuard`（`src/
// lib.rs`）の正のプローブ 1 ブロック方式のドリフト検査と、
// `compat::Sequential::add_pixel_shuffle`／`add_pixel_unshuffle` の
// 非宣言を持つ。`spatial_layers_hold_doctest_*`（#2159）と同型。
// 承認事項・多層防御の位置づけは
// `docs/autodiff-pixel-shuffle-decision.md` §6 参照。
// =====================================================================

/// `PixelShuffleHoldDoctestGuard` の唯一の doctest ブロックが glob
/// import するネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod`
/// 宣言集合が一致することを固定する（`spatial_layers_hold_doctest_
/// globs_all_pub_modules` と同型。新しい `pub mod` を facade へ追加した
/// 際、doctest 側の `use` 一覧の更新を機械的に強制する。イシュー
/// #2162）。
#[test]
fn pixel_shuffle_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "PixelShuffleHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "PixelShuffleHoldDoctestGuard の doctest ブロックが glob import\
         するモジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加する\
         こと。"
    );
}

/// [`pixel_shuffle_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**（`__fandhe_pixel_shuffle_hold_probe` モジュール・
/// `__probe` 関数）が固定文言 [`PIXEL_SHUFFLE_HOLD_PROBE_BODY`] と 1 行たりとも
/// 違わず一致することを固定する（rustdoc の `# ` 隠し行・プローブの
/// 削除・別名へのシャドーイング等で正のプローブを骨抜きにする改変を
/// 機械的に拒否する。イシュー #2162）。
#[test]
fn pixel_shuffle_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "PixelShuffleHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, PIXEL_SHUFFLE_HOLD_PROBE_BODY,
        "PixelShuffleHoldDoctestGuard の doctest ブロック本文（glob\
         以外）が固定文言 PIXEL_SHUFFLE_HOLD_PROBE_BODY からドリフト\
         している。正のプローブ（__fandhe_pixel_shuffle_hold_probe\
         モジュール・__probe 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`pixel_shuffle_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `PixelShuffleHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const PIXEL_SHUFFLE_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_pixel_shuffle_hold_probe {\n\
\x20\x20\x20\x20pub struct PixelShuffle;\n\
\x20\x20\x20\x20pub struct PixelUnshuffle;\n\
\x20\x20\x20\x20pub struct __FandhePixelShuffleHoldMarker;\n\
\x20\x20\x20\x20pub fn add_pixel_shuffle() -> __FandhePixelShuffleHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandhePixelShuffleHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20pub fn add_pixel_unshuffle() -> __FandhePixelShuffleHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandhePixelShuffleHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_pixel_shuffle_hold_probe::*;\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20_: PixelShuffle,\n\
\x20\x20\x20\x20_: PixelUnshuffle,\n\
) {\n\
\x20\x20\x20\x20let _: __FandhePixelShuffleHoldMarker = add_pixel_shuffle();\n\
\x20\x20\x20\x20let _: __FandhePixelShuffleHoldMarker = add_pixel_unshuffle();\n\
}";

// =====================================================================
// イシュー #2526（親 #2520・ルート #2499 の一括承認）: `PixelShuffle`／
// `PixelUnshuffle` の facade 公開の正ガード。保留ガード
// （`PixelShuffleHoldDoctestGuard`）から add_*／Var メソッドのプローブを外し、
// 承認形（`Var` の 1 行委譲メソッド 2 個・`compat::Sequential` の add_* 2 個）
// だけを許す形へ反転した。型の再エクスポートと自由関数での公開は保留ガードの
// 衝突プローブで引き続き禁止する。承認事項は
// `docs/autodiff-pixel-shuffle-decision.md` §6 参照。
// =====================================================================

/// #2526 で公開する 4 名の `fn` 名（`Var` 2 個 + `Sequential` 2 個）。
const PIXEL_SHUFFLE_FACADE_FN_NAMES: [&str; 4] = [
    "pixel_shuffle",
    "pixel_unshuffle",
    "add_pixel_shuffle",
    "add_pixel_unshuffle",
];

/// workspace 全体（`crates/*/src/`）で [`PIXEL_SHUFFLE_FACADE_FN_NAMES`] の
/// `fn` 宣言の定義元集合を固定する（過不足とも fail-closed。迂回実装の混入検出）。
#[test]
fn workspace_declares_pixel_shuffle_facade_fn_names_only_in_approved_locations() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut crate_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("workspace crates ディレクトリが読めない")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(!crate_dirs.is_empty());
    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in PIXEL_SHUFFLE_FACADE_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }
    let expected: std::collections::BTreeMap<String, usize> = [
        "autodiff/src/var.rs::pixel_shuffle",
        "autodiff/src/var.rs::pixel_unshuffle",
        "facade/src/compat/sequential.rs::add_pixel_shuffle",
        "facade/src/compat/sequential.rs::add_pixel_unshuffle",
    ]
    .iter()
    .map(|k| (k.to_string(), 1usize))
    .collect();
    assert_eq!(
        found, expected,
        "#2526 の 4 名の fn 宣言の定義元集合が承認形とずれている（新たな定義元が\
         承認済みの実装なのか迂回経路なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 2 メソッド本体の承認形（`nn` 側共有 forward への 1 行委譲）。
const PIXEL_SHUFFLE_VAR_EXPECTED_BODIES: [(&str, &str); 2] = [
    (
        "pixel_shuffle",
        "crate : : nn : : pixel_shuffle_forward ( self , upscale_factor )",
    ),
    (
        "pixel_unshuffle",
        "crate : : nn : : pixel_unshuffle_forward ( self , downscale_factor )",
    ),
];

/// `Var::pixel_shuffle`／`Var::pixel_unshuffle` が共有 forward への薄い委譲で
/// あることを固定する（スタブ・独自実装へのすり替えを拒否）。
#[test]
fn var_pixel_shuffle_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in PIXEL_SHUFFLE_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（共有 forward への 1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで 2 メソッドへ到達でき、シグネチャが
/// 承認形と一致し、実際に適用して期待 shape・値が得られることを固定する。
#[test]
fn var_pixel_shuffle_methods_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type Sig<'t> = fn(&Var<'t>, usize) -> Result<Var<'t>, AutodiffError>;
    fn sig_shuffle<'t>() -> Sig<'t> {
        Var::<'t>::pixel_shuffle
    }
    fn sig_unshuffle<'t>() -> Sig<'t> {
        Var::<'t>::pixel_unshuffle
    }

    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![0.0_f32, 1.0, 2.0, 3.0], &[1, 4, 1, 1]).expect("tensor"));
    let y = sig_shuffle()(&x, 2).expect("pixel_shuffle");
    assert_eq!(y.to_tensor().shape(), [1, 1, 2, 2]);
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [0.0, 1.0, 2.0, 3.0]
    );
    let z = sig_unshuffle()(&y, 2).expect("pixel_unshuffle");
    assert_eq!(z.to_tensor().shape(), [1, 4, 1, 1]);
    assert!(sig_shuffle()(&x, 0).is_err());
    assert!(sig_shuffle()(&x, 3).is_err());
    assert!(sig_unshuffle()(&x, 0).is_err());
}

const ADD_PIXEL_SHUFFLE_PARAMS: &str =
    "mut self, upscale_factor: usize) -> Result<Self, AutodiffError>";
const ADD_PIXEL_UNSHUFFLE_PARAMS: &str =
    "mut self, downscale_factor: usize) -> Result<Self, AutodiffError>";

/// `compat::Sequential` の 2 add_* が承認シグネチャで 1 件ずつ存在する。
#[test]
fn compat_sequential_pixel_shuffle_add_methods_have_approved_signatures() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(
        count_fn_declarations_by_name(&tokens, "add_pixel_shuffle"),
        1
    );
    assert_eq!(
        count_fn_declarations_by_name(&tokens, "add_pixel_unshuffle"),
        1
    );
    assert!(sequential_spatial_add_signature_ok(
        &cleaned,
        "add_pixel_shuffle",
        ADD_PIXEL_SHUFFLE_PARAMS
    ));
    assert!(sequential_spatial_add_signature_ok(
        &cleaned,
        "add_pixel_unshuffle",
        ADD_PIXEL_UNSHUFFLE_PARAMS
    ));
}

/// [`compat_sequential_pixel_shuffle_add_methods_have_approved_signatures`] の自己テスト。
#[test]
fn compat_sequential_pixel_shuffle_add_methods_have_approved_signatures_detects_offense() {
    let ok = "pub fn add_pixel_shuffle(mut self, upscale_factor: usize,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok,
        "add_pixel_shuffle",
        ADD_PIXEL_SHUFFLE_PARAMS
    ));
    for bad in [
        "pub fn add_pixel_shuffle(mut self, upscale_factor: usize) -> Self {",
        "pub fn add_pixel_shuffle(mut self, upscale_factor: i64) -> Result<Self, AutodiffError> {",
        "pub fn add_pixel_shuffle(&mut self, upscale_factor: usize) -> Result<Self, AutodiffError> {",
        "pub fn add_linear(mut self) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(
                bad,
                "add_pixel_shuffle",
                ADD_PIXEL_SHUFFLE_PARAMS
            ),
            "{bad}"
        );
    }
}

// =====================================================================
// イシュー #2158（親 #2131）: Conv3d の facade 公開保留を検査する
// テスト群。`VarConv3dHoldDoctestGuard`（`src/lib.rs`）の正のプローブ
// 1 ブロック方式のドリフト検査に加え、workspace 全体のソース走査による
// 定義元インベントリを持つ。イシュー #2524（ルート #2499 の一括承認）で
// `Var::conv3d`／`compat::Sequential::add_conv3d` の 2 形が公開されたため、
// 保留ガードは承認外の形（`conv3d_ops` 再エクスポート・`Tensor`／`Tape` の
// `conv3d`）だけを禁じる縮小形になり、承認形は正ガード（シグネチャ・宣言数・
// 委譲本体の固定）で守る。多層防御の位置づけは `docs/conv-ops-design.md` §16.7 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `VarConv3dHoldDoctestGuard` doc 内の
/// 唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （`activation_ops_hold_doctest_globs_all_pub_modules` の
/// `VarConv3dHoldDoctestGuard` 版）。
#[test]
fn conv3d_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarConv3dHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "VarConv3dHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトして\
         いる（declared={declared:?}, doctest={globbed:?}）。新しい pub\
         mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`conv3d_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの **glob
/// 以外の本文**が固定文言 [`CONV3D_HOLD_PROBE_BODY`] と 1 行たりとも
/// 違わず一致することを固定する（rustdoc の `# ` 隠し行・プローブの
/// 削除・別名へのシャドーイング等で正のプローブを骨抜きにする改変を
/// 機械的に拒否する）。
#[test]
fn conv3d_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "VarConv3dHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, CONV3D_HOLD_PROBE_BODY,
        "VarConv3dHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 CONV3D_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_conv3d_hold_probe モジュール・\
         __FandheConv3dHoldProbe／__FandheConv3dAddProbe トレイト・\
         __probe_* 関数）の削除・弱体化・隠し行の混入がないか確認する\
         こと。"
    );
}

/// [`conv3d_hold_doctest_probe_body_matches_fixed_contract`] が要求する
/// 固定文言。`crates/facade/src/lib.rs` の `VarConv3dHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致
/// する必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に
/// 含む）。
const CONV3D_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_conv3d_hold_probe {\n\
\x20\x20\x20\x20pub mod conv3d_ops {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn conv3d() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_conv3d_hold_probe::*;\n\
\n\
struct __FandheConv3dMarker;\n\
\n\
trait __FandheConv3dHoldProbe {\n\
\x20\x20\x20\x20fn conv3d(&self) -> __FandheConv3dMarker;\n\
}\n\
\n\
impl __FandheConv3dHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn conv3d(&self) -> __FandheConv3dMarker { __FandheConv3dMarker }\n\
}\n\
\n\
impl __FandheConv3dHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn conv3d(&self) -> __FandheConv3dMarker { __FandheConv3dMarker }\n\
}\n\
\n\
fn __probe_free_fns() {\n\
\x20\x20\x20\x20// `conv3d_ops::` を経由した経路解決（`use fandhe_ai::*;` が\n\
\x20\x20\x20\x20// 同名モジュールを glob 公開していれば、名前解決自体が曖昧に\n\
\x20\x20\x20\x20// なり E0659 でコンパイル失敗する）。\n\
\x20\x20\x20\x20conv3d_ops::conv3d();\n\
}\n\
\n\
fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheConv3dMarker = fandhe_ai::Tensor::conv3d(x);\n\
\x20\x20\x20\x20let _: __FandheConv3dMarker = x.conv3d();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheConv3dMarker = fandhe_ai::Tape::conv3d(x);\n\
\x20\x20\x20\x20let _: __FandheConv3dMarker = x.conv3d();\n\
}";

/// facade src 全体（`crates/facade/src/**`）に、`conv3d_ops` を参照する
/// `pub use`（モジュール再エクスポート・別名含む）も、`fn conv3d` の宣言も
/// 存在せず、`fn add_conv3d` は承認形として `src/compat/sequential.rs` にだけ
/// ちょうど 1 件あることを固定する（`VarConv3dHoldDoctestGuard` の正の
/// プローブと多層防御を成すソース走査ガード。イシュー #2524 で `add_conv3d`
/// を「0 件」から「承認ファイルに 1 件」へ反転した）。
#[test]
fn facade_declares_conv3d_names_only_in_approved_form() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "conv3d_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `conv3d_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let is_sequential = path.ends_with("compat/sequential.rs");
        for (fn_name, allowed) in [
            ("conv3d", 0usize),
            ("add_conv3d", usize::from(is_sequential)),
        ] {
            let count = count_fn_declarations_by_name(&tokens, fn_name);
            if count != allowed {
                offending.push(format!(
                    "{}: `fn {fn_name}` 宣言が {count} 件（許容 {allowed} 件）",
                    path.display()
                ));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が conv3d の承認形（Var::conv3d 委譲・compat::Sequential::\
         add_conv3d。イシュー #2524）から逸脱している: {offending:?}"
    );
}

const ADD_CONV3D_PARAMS: &str = "mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], groups: usize, seed: u64, ) -> Result<Self, AutodiffError>";

/// `compat::Sequential::add_conv3d` が承認シグネチャで 1 件だけ存在する（イシュー #2524）。
#[test]
fn add_conv3d_signature_matches_approved_contract() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(count_fn_declarations_by_name(&tokens, "add_conv3d"), 1);
    assert!(sequential_spatial_add_signature_ok(
        &cleaned,
        "add_conv3d",
        ADD_CONV3D_PARAMS
    ));
}

/// [`add_conv3d_signature_matches_approved_contract`] の自己テスト。
#[test]
fn add_conv3d_signature_matches_approved_contract_detects_offense() {
    let ok = "pub fn add_conv3d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok,
        "add_conv3d",
        ADD_CONV3D_PARAMS
    ));
    for bad in [
        // groups 欠落
        "pub fn add_conv3d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], seed: u64,) -> Result<Self, AutodiffError> {",
        // 引数順の入替（stride と padding）
        "pub fn add_conv3d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 3], padding: [usize; 3], stride: [usize; 3], dilation: [usize; 3], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {",
        // 型違い（2 軸）
        "pub fn add_conv3d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 2], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], groups: usize, seed: u64,) -> Result<Self, AutodiffError> {",
        // 戻り値が Self
        "pub fn add_conv3d(mut self, in_channels: usize, out_channels: usize, kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3], dilation: [usize; 3], groups: usize, seed: u64,) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(bad, "add_conv3d", ADD_CONV3D_PARAMS),
            "{bad}"
        );
    }
}

/// `src/compat` 配下で `pub fn add_conv3d(` がちょうど 1 件（0 件＝公開の脱落、
/// 2 件以上＝重複宣言の混入を拒否する正ガード）。
#[test]
fn compat_sequential_exposes_conv3d_add_method_issue_2524() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut count = 0usize;
    visit_rs_files(&compat_dir, &mut |_path, content| {
        count += count_pub_fn_declarations(content, "add_conv3d");
    });
    assert_eq!(
        count, 1,
        "src/compat 配下の add_conv3d の pub fn 宣言数が 1 件でない（count={count}）"
    );
}

/// [`compat_sequential_exposes_conv3d_add_method_issue_2524`] の自己テスト。
#[test]
fn compat_sequential_exposes_conv3d_add_method_issue_2524_counts_declarations() {
    let one = "    pub fn add_conv3d(mut self) -> Self {\n        self\n    }\n";
    assert_eq!(count_pub_fn_declarations(one, "add_conv3d"), 1);
    assert_eq!(count_pub_fn_declarations(one, "add_conv2d"), 0);
    let dup = format!("{one}{one}");
    assert_eq!(count_pub_fn_declarations(&dup, "add_conv3d"), 2);
}

/// `Var::conv3d`（`autodiff/src/var.rs`）の本体が `conv3d_ops::conv3d` への 1 式委譲で
/// あることを固定し、facade の `fandhe_ai::Var` だけで承認シグネチャの関数ポインタへ
/// 束縛でき、実際に適用して期待 shape・値が得られることを確認する（イシュー #2524）。
#[test]
fn var_conv3d_is_thin_delegation_with_approved_signature() {
    let path = workspace_crates_dir().join("autodiff/src/var.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    assert_eq!(
        determinism_fn_body(&tokens, "conv3d").as_deref(),
        Some(
            "crate : : conv3d_ops : : conv3d ( self , weight , bias , stride , padding , dilation , groups )"
        ),
        "Var::conv3d の本体が承認形（conv3d_ops::conv3d への 1 式委譲）と一致しない"
    );

    let tape = fandhe_ai::tape();
    let x = tape.var(&fandhe_ai::Tensor::new(vec![1.0_f32; 8], &[1, 1, 2, 2, 2]).unwrap());
    let w = tape.var(&fandhe_ai::Tensor::new(vec![1.0_f32; 8], &[1, 1, 2, 2, 2]).unwrap());
    // UFCS で呼ぶことで引数順・型（`[usize; 3]` ×3・`usize`・`Option<&Var>`）を承認形に固定する。
    let y: Result<fandhe_ai::Var<'_>, fandhe_ai::AutodiffError> =
        fandhe_ai::Var::conv3d(&x, &w, None, [1; 3], [0; 3], [1; 3], 1);
    let out = y.unwrap().to_tensor();
    assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(out.contiguous().as_slice().unwrap(), &[8.0_f32]);
}
/// `conv3d`・`im2col3d`・`col2im3d`（3 個の関数名。イシュー #2158）の
/// workspace 全体（`crates/*/src/`）における `fn` 宣言の定義元集合が、
/// 実装計画で列挙した許容集合とちょうど一致することを固定する
/// （`workspace_declares_activation_ops_fn_names_only_in_approved_locations` と同型のインベントリ）。
///
/// **期待集合（着手時に `grep -rn "fn conv3d\b\|fn im2col3d\b\|fn
/// col2im3d\b" crates/*/src` で実測確認済み）**:
/// - `conv3d`: `tensor-core/src/backend_ops.rs`（`BackendOps` trait の
///   既定実装）・`autodiff/src/conv3d_ops.rs`（自由関数。CPU は
///   `conv3d` を override しないため他に無い）・`autodiff/src/var.rs`
///   （`Var::conv3d` 委譲メソッド。イシュー #2524）
/// - `add_conv3d`: `facade/src/compat/sequential.rs`（イシュー #2524）
/// - `im2col3d`／`col2im3d`: `tensor-core/src/backend_ops.rs`（trait 既定
///   実装）・`backend-cpu/src/im2col.rs`（本体）・`backend-cpu/src/
///   ops.rs`（override）・`autodiff/src/eval.rs`（ホスト参照実装）
#[test]
fn workspace_declares_conv3d_fn_names_only_in_allowed_locations() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

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
            for fn_name in ["conv3d", "add_conv3d", "im2col3d", "col2im3d"] {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    let rel = path
                        .strip_prefix(&crates_dir)
                        .unwrap_or(path)
                        .to_string_lossy()
                        .replace('\\', "/");
                    found
                        .entry(fn_name.to_string())
                        .or_default()
                        .push(format!("{rel} ({count})"));
                }
            }
        });
    }

    let expected: std::collections::BTreeMap<String, Vec<String>> = [
        (
            "conv3d".to_string(),
            vec![
                "tensor-core/src/backend_ops.rs (1)".to_string(),
                "autodiff/src/conv3d_ops.rs (1)".to_string(),
                "autodiff/src/var.rs (1)".to_string(),
            ],
        ),
        (
            "add_conv3d".to_string(),
            vec!["facade/src/compat/sequential.rs (1)".to_string()],
        ),
        (
            "im2col3d".to_string(),
            vec![
                "tensor-core/src/backend_ops.rs (1)".to_string(),
                "backend-cpu/src/im2col.rs (1)".to_string(),
                "backend-cpu/src/ops.rs (1)".to_string(),
                "autodiff/src/eval.rs (1)".to_string(),
            ],
        ),
        (
            "col2im3d".to_string(),
            vec![
                "tensor-core/src/backend_ops.rs (1)".to_string(),
                "backend-cpu/src/im2col.rs (1)".to_string(),
                "backend-cpu/src/ops.rs (1)".to_string(),
                "autodiff/src/eval.rs (1)".to_string(),
            ],
        ),
    ]
    .into_iter()
    .map(|(k, mut v)| {
        v.sort();
        (k, v)
    })
    .collect();

    let mut found_sorted = found.clone();
    for v in found_sorted.values_mut() {
        v.sort();
    }

    assert_eq!(
        found_sorted, expected,
        "conv3d／im2col3d／col2im3d の fn 宣言の定義元集合が実装計画の\
         期待集合とドリフトしている（found={found_sorted:?},\
         expected={expected:?}）。新たな定義元が見つかった場合、それが\
         承認済みの実装なのか迂回経路の混入なのかを確認すること。"
    );
}

// =====================================================================
// イシュー #2161（親 #2131）: Dropout2d・AlphaDropout・EmbeddingBag の
// facade 公開保留（#2528 で `Var` メソッド・`Sequential::add_*`・
// `EmbeddingBagMode` は公開済みに反転。層型・自由関数のみ保留継続）を
// 検査するテスト群。`DropoutEmbeddingBagHoldDoctestGuard`（`src/lib.rs`）の
// 正のプローブ 1 ブロック方式のドリフト検査を持つ。承認事項・多層防御の
// 位置づけは `docs/autodiff-dropout-embedding-bag-decision.md` §6・§8 参照。
// =====================================================================

/// `DropoutEmbeddingBagHoldDoctestGuard` の唯一の doctest ブロックが
/// glob import するネスト `pub mod` 集合と、`src/lib.rs` の実際の
/// `pub mod` 宣言集合が一致することを固定する（`spatial_layers_hold_
/// doctest_globs_all_pub_modules` と同型。イシュー #2161）。
#[test]
fn dropout_embedding_bag_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "DropoutEmbeddingBagHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "DropoutEmbeddingBagHoldDoctestGuard の doctest ブロックが glob\
         import するモジュール集合が src/lib.rs の pub mod 宣言集合と\
         ドリフトしている（declared={declared:?}, doctest={globbed:?}）。\
         新しい pub mod を追加した場合は doctest 側の use 一覧にも追加\
         すること。"
    );
}

/// [`dropout_embedding_bag_hold_doctest_globs_all_pub_modules`] が glob
/// import 集合の一致のみを固定するのに対し、本テストは doctest
/// ブロックの**glob 以外の本文**（`__fandhe_dropout_embedding_bag_hold_
/// probe` モジュール・`__probe` 関数）が固定文言
/// [`DROPOUT_EMBEDDING_BAG_HOLD_PROBE_BODY`] と 1 行たりとも違わず
/// 一致することを固定する（rustdoc の `# ` 隠し行・プローブの削除・
/// 別名へのシャドーイング等で正のプローブを骨抜きにする改変を機械的に
/// 拒否する。イシュー #2161）。
#[test]
fn dropout_embedding_bag_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "DropoutEmbeddingBagHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, DROPOUT_EMBEDDING_BAG_HOLD_PROBE_BODY,
        "DropoutEmbeddingBagHoldDoctestGuard の doctest ブロック本文\
         （glob 以外）が固定文言 DROPOUT_EMBEDDING_BAG_HOLD_PROBE_BODY\
         からドリフトしている。正のプローブ（__fandhe_dropout_\
         embedding_bag_hold_probe モジュール・__probe 関数）の\
         削除・弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`dropout_embedding_bag_hold_doctest_probe_body_matches_fixed_contract`]
/// が要求する固定文言。`crates/facade/src/lib.rs` の
/// `DropoutEmbeddingBagHoldDoctestGuard` doc 内の唯一の doctest
/// ブロックから、ネスト `pub mod` の glob import 行
/// （`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致
/// する必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に
/// 含む）。
const DROPOUT_EMBEDDING_BAG_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_dropout_embedding_bag_hold_probe {\n\
\x20\x20\x20\x20pub struct Dropout2d;\n\
\x20\x20\x20\x20pub struct AlphaDropout;\n\
\x20\x20\x20\x20pub struct EmbeddingBag;\n\
\x20\x20\x20\x20pub struct EmbeddingBagVars;\n\
\x20\x20\x20\x20pub struct __FandheDropoutEmbeddingBagHoldMarker;\n\
\x20\x20\x20\x20pub fn add_dropout2d() -> __FandheDropoutEmbeddingBagHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheDropoutEmbeddingBagHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20pub fn add_alpha_dropout() -> __FandheDropoutEmbeddingBagHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheDropoutEmbeddingBagHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20pub fn add_embedding_bag() -> __FandheDropoutEmbeddingBagHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheDropoutEmbeddingBagHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_dropout_embedding_bag_hold_probe::*;\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20_: Dropout2d,\n\
\x20\x20\x20\x20_: AlphaDropout,\n\
\x20\x20\x20\x20_: EmbeddingBag,\n\
\x20\x20\x20\x20_: EmbeddingBagVars,\n\
) {\n\
\x20\x20\x20\x20let _: __FandheDropoutEmbeddingBagHoldMarker = add_dropout2d();\n\
\x20\x20\x20\x20let _: __FandheDropoutEmbeddingBagHoldMarker = add_alpha_dropout();\n\
\x20\x20\x20\x20let _: __FandheDropoutEmbeddingBagHoldMarker = add_embedding_bag();\n\
}";

// =====================================================================
// イシュー #2528（親 #2520・ルート #2499 の一括承認）: Dropout2d・
// AlphaDropout・EmbeddingBag の facade 公開の正ガード。保留ガード
// （`DropoutEmbeddingBagHoldDoctestGuard`）から `Sequential`／`Var` の
// プローブと `EmbeddingBagMode` の型プローブを外し、承認形（`Var` の 1 行
// 委譲メソッド 3 個・`compat::Sequential` の add_* 3 個・ルートの
// `EmbeddingBagMode` 再エクスポート 1 行）だけを許す形へ反転した。層型
// （`Dropout2d`／`AlphaDropout`／`EmbeddingBag`／`EmbeddingBagVars`）の
// 再エクスポートと自由関数での公開は未承認のまま禁止する。承認事項は
// `docs/autodiff-dropout-embedding-bag-decision.md` §6・§8 参照。
// =====================================================================

/// #2528 で公開する 6 名の `fn` 名（`Var` 3 個 + `Sequential` 3 個）。
const DROPOUT_EMBEDDING_BAG_FACADE_FN_NAMES: [&str; 6] = [
    "dropout2d",
    "alpha_dropout",
    "embedding_bag",
    "add_dropout2d",
    "add_alpha_dropout",
    "add_embedding_bag",
];

/// workspace 全体（`crates/*/src/`）で [`DROPOUT_EMBEDDING_BAG_FACADE_FN_NAMES`] の
/// `fn` 宣言の定義元集合を固定する（過不足とも fail-closed。迂回実装・自由関数公開の
/// 混入検出）。
#[test]
fn workspace_declares_dropout_embedding_bag_facade_fn_names_only_in_approved_locations() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut crate_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("workspace crates ディレクトリが読めない")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(!crate_dirs.is_empty());
    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in DROPOUT_EMBEDDING_BAG_FACADE_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }
    let expected: std::collections::BTreeMap<String, usize> = [
        "autodiff/src/var.rs::dropout2d",
        "autodiff/src/var.rs::alpha_dropout",
        "autodiff/src/var.rs::embedding_bag",
        "facade/src/compat/sequential.rs::add_dropout2d",
        "facade/src/compat/sequential.rs::add_alpha_dropout",
        "facade/src/compat/sequential.rs::add_embedding_bag",
    ]
    .iter()
    .map(|k| (k.to_string(), 1usize))
    .collect();
    assert_eq!(
        found, expected,
        "#2528 の fn 宣言の定義元集合が承認形とずれている（新たな定義元が承認済みの\
         実装なのか迂回経路なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 3 メソッド本体の承認形（`nn` の共有 forward への 1 行委譲）。
const DROPOUT_EMBEDDING_BAG_VAR_EXPECTED_BODIES: [(&str, &str); 3] = [
    (
        "dropout2d",
        "crate : : nn : : dropout2d_forward ( self , p , training )",
    ),
    (
        "alpha_dropout",
        "crate : : nn : : alpha_dropout_forward ( self , p , training )",
    ),
    (
        "embedding_bag",
        "crate : : nn : : embedding_bag_forward ( self , ids , mode , padding_idx )",
    ),
];

/// `Var::dropout2d`／`alpha_dropout`／`embedding_bag` が共有 forward への薄い委譲で
/// あることを固定する（スタブ・独自実装へのすり替えを拒否）。
#[test]
fn var_dropout_embedding_bag_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in DROPOUT_EMBEDDING_BAG_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（共有 forward への 1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで 3 メソッドへ到達でき、シグネチャが承認形と
/// 一致し、実際に適用して期待 shape・値が得られ、エラー経路が `Err` になることを固定する。
#[test]
fn var_dropout_embedding_bag_methods_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, EmbeddingBagMode, Tensor, Var};

    type DropSig<'t> = fn(&Var<'t>, f32, bool) -> Result<Var<'t>, AutodiffError>;
    type BagSig<'t> = fn(
        &Var<'t>,
        &Tensor<i32>,
        EmbeddingBagMode,
        Option<usize>,
    ) -> Result<Var<'t>, AutodiffError>;
    fn drop2d<'t>() -> DropSig<'t> {
        Var::<'t>::dropout2d
    }
    fn alpha<'t>() -> DropSig<'t> {
        Var::<'t>::alpha_dropout
    }
    fn bag<'t>() -> BagSig<'t> {
        Var::<'t>::embedding_bag
    }

    let tape = fandhe_ai::tape();
    let data: Vec<f32> = (0..8).map(|v| v as f32).collect();
    let x = tape.var(&Tensor::new(data.clone(), &[1, 2, 2, 2]).expect("tensor"));
    // eval（training=false）・p=0 は恒等。
    for f in [drop2d(), alpha()] {
        let y = f(&x, 0.5, false).expect("eval は恒等");
        assert_eq!(y.to_tensor().host_slice().into_owned(), data);
        let y = f(&x, 0.0, true).expect("p=0 は恒等");
        assert_eq!(y.to_tensor().host_slice().into_owned(), data);
        // p の範囲外・NaN は Err。
        assert!(f(&x, 1.5, true).is_err());
        assert!(f(&x, -0.1, true).is_err());
        assert!(f(&x, f32::NAN, true).is_err());
    }
    // p=1.0 は全要素 0（Dropout2d は全チャネル drop・AlphaDropout は特例）。
    for f in [drop2d(), alpha()] {
        let y = f(&x, 1.0, true).expect("p=1");
        assert_eq!(y.to_tensor().shape(), [1, 2, 2, 2]);
        assert!(y.to_tensor().host_slice().iter().all(|v| *v == 0.0));
    }
    // Dropout2d は rank 4 限定。
    let x3 = tape.var(&Tensor::new(vec![1.0_f32; 8], &[2, 2, 2]).expect("tensor"));
    assert!(drop2d()(&x3, 0.5, true).is_err());
    assert!(drop2d()(&x3, 0.5, false).is_err());
    // AlphaDropout は任意 rank。
    assert!(alpha()(&x3, 1.0, true).is_ok());

    // EmbeddingBag: weight [4, 2]・ids [2, 2]。
    let w = tape.var(
        &Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).expect("weight"),
    );
    let ids = Tensor::<i32>::new(vec![0, 1, 2, 3], &[2, 2]).expect("ids");
    let sum = bag()(&w, &ids, EmbeddingBagMode::Sum, None).expect("sum");
    assert_eq!(sum.to_tensor().shape(), [2, 2]);
    assert_eq!(
        sum.to_tensor().host_slice().into_owned(),
        [4.0, 6.0, 12.0, 14.0]
    );
    let mean = bag()(&w, &ids, EmbeddingBagMode::Mean, None).expect("mean");
    assert_eq!(
        mean.to_tensor().host_slice().into_owned(),
        [2.0, 3.0, 6.0, 7.0]
    );
    let max = bag()(&w, &ids, EmbeddingBagMode::Max, None).expect("max");
    assert_eq!(
        max.to_tensor().host_slice().into_owned(),
        [3.0, 4.0, 7.0, 8.0]
    );
    // padding_idx に一致する id は縮約から除外される。
    let padded = bag()(&w, &ids, EmbeddingBagMode::Sum, Some(0)).expect("padding");
    assert_eq!(
        padded.to_tensor().host_slice().into_owned(),
        [3.0, 4.0, 12.0, 14.0]
    );
    // エラー経路: padding_idx 範囲外・範囲外 id・ids の rank 違い・weight の rank 違い。
    assert!(bag()(&w, &ids, EmbeddingBagMode::Sum, Some(4)).is_err());
    let bad_ids = Tensor::<i32>::new(vec![0, 4, 2, 3], &[2, 2]).expect("ids");
    assert!(bag()(&w, &bad_ids, EmbeddingBagMode::Sum, None).is_err());
    let neg_ids = Tensor::<i32>::new(vec![0, -1, 2, 3], &[2, 2]).expect("ids");
    assert!(bag()(&w, &neg_ids, EmbeddingBagMode::Sum, None).is_err());
    let flat_ids = Tensor::<i32>::new(vec![0, 1], &[2]).expect("ids");
    assert!(bag()(&w, &flat_ids, EmbeddingBagMode::Sum, None).is_err());
    let w1 = tape.var(&Tensor::new(vec![1.0_f32, 2.0], &[2]).expect("weight"));
    assert!(bag()(&w1, &ids, EmbeddingBagMode::Sum, None).is_err());
}

const ADD_DROPOUT2D_PARAMS: &str = "mut self, p: f32) -> Result<Self, AutodiffError>";
const ADD_ALPHA_DROPOUT_PARAMS: &str = "mut self, p: f32) -> Result<Self, AutodiffError>";
const ADD_EMBEDDING_BAG_PARAMS: &str = "mut self, num_embeddings: usize, embedding_dim: usize, mode: EmbeddingBagMode, padding_idx: Option<usize>, seed: u64, ) -> Result<Self, AutodiffError>";

/// `compat::Sequential` の 3 add_* が承認シグネチャで 1 件ずつ存在する。
#[test]
fn compat_sequential_dropout_embedding_bag_add_methods_have_approved_signatures() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, params) in [
        ("add_dropout2d", ADD_DROPOUT2D_PARAMS),
        ("add_alpha_dropout", ADD_ALPHA_DROPOUT_PARAMS),
        ("add_embedding_bag", ADD_EMBEDDING_BAG_PARAMS),
    ] {
        assert_eq!(count_fn_declarations_by_name(&tokens, name), 1, "{name}");
        assert!(
            sequential_spatial_add_signature_ok(&cleaned, name, params),
            "{name} のシグネチャが承認形と一致しない"
        );
    }
}

/// [`compat_sequential_dropout_embedding_bag_add_methods_have_approved_signatures`]
/// の自己テスト。
#[test]
fn compat_sequential_dropout_embedding_bag_add_methods_have_approved_signatures_detects_offense() {
    let ok = "pub fn add_dropout2d(mut self, p: f32,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok,
        "add_dropout2d",
        ADD_DROPOUT2D_PARAMS
    ));
    for bad in [
        "pub fn add_dropout2d(mut self, p: f64) -> Result<Self, AutodiffError> {",
        "pub fn add_dropout2d(mut self, p: f32) -> Self {",
        "pub fn add_dropout2d(&mut self, p: f32) -> Result<Self, AutodiffError> {",
        "pub fn add_linear(mut self) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(bad, "add_dropout2d", ADD_DROPOUT2D_PARAMS),
            "{bad}"
        );
    }
    let ok_bag = "pub fn add_embedding_bag(mut self, num_embeddings: usize, embedding_dim: usize, mode: EmbeddingBagMode, padding_idx: Option<usize>, seed: u64,) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok_bag,
        "add_embedding_bag",
        ADD_EMBEDDING_BAG_PARAMS
    ));
    // 引数順の入れ替え・mode の欠落は不一致。
    for bad in [
        "pub fn add_embedding_bag(mut self, embedding_dim: usize, num_embeddings: usize, mode: EmbeddingBagMode, padding_idx: Option<usize>, seed: u64,) -> Result<Self, AutodiffError> {",
        "pub fn add_embedding_bag(mut self, num_embeddings: usize, embedding_dim: usize, padding_idx: Option<usize>, seed: u64,) -> Result<Self, AutodiffError> {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(
                bad,
                "add_embedding_bag",
                ADD_EMBEDDING_BAG_PARAMS
            ),
            "{bad}"
        );
    }
}

/// ルートで再エクスポートを許す型の唯一の承認形（`src/lib.rs`）。
const EMBEDDING_BAG_MODE_APPROVED_LINE: &str = "pub use fandhe_ai_autodiff::nn::EmbeddingBagMode;";

/// facade が再エクスポートしてはならない層型（未承認）。
const DROPOUT_EMBEDDING_BAG_LAYER_TYPE_NAMES: [&str; 4] = [
    "Dropout2d",
    "AlphaDropout",
    "EmbeddingBag",
    "EmbeddingBagVars",
];

/// facade src の 1 ファイル内容から、`EmbeddingBagMode` または未承認の層型名を識別子として
/// 含む `pub use` 行（空白正規化済み）を集める検出本体。コメント・文字列リテラルは無視する。
fn scan_embedding_bag_pub_use_lines(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    cleaned
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("pub use") || l.starts_with("pub(crate) use"))
        .filter(|l| {
            line_contains_identifier(l, "EmbeddingBagMode")
                || DROPOUT_EMBEDDING_BAG_LAYER_TYPE_NAMES
                    .iter()
                    .any(|n| line_contains_identifier(l, n))
        })
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

/// `EmbeddingBagMode` の再エクスポートが `src/lib.rs` の承認形 1 行だけで、層型
/// （`Dropout2d`／`AlphaDropout`／`EmbeddingBag`／`EmbeddingBagVars`）が facade の
/// どこからも再エクスポートされないことを固定する。
#[test]
fn facade_reexports_embedding_bag_mode_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_lib_rs = 0usize;
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in scan_embedding_bag_pub_use_lines(content) {
            if path.ends_with("src/lib.rs") && line == EMBEDDING_BAG_MODE_APPROVED_LINE {
                approved_in_lib_rs += 1;
            } else {
                offending.push(format!("{}: `{line}`", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade が EmbeddingBagMode を承認形（src/lib.rs の `{EMBEDDING_BAG_MODE_APPROVED_LINE}` \
         1 行）以外で、または層型（Dropout2d／AlphaDropout／EmbeddingBag／EmbeddingBagVars）を\
         再エクスポートしている（#2528）: {offending:?}"
    );
    assert_eq!(
        approved_in_lib_rs, 1,
        "src/lib.rs に承認形の再エクスポート行がちょうど 1 行存在しない\
         （検査対象を見失った場合を含む）"
    );
}

/// [`scan_embedding_bag_pub_use_lines`] の自己テスト（合成入力）。
#[test]
fn facade_reexports_embedding_bag_mode_only_in_approved_shape_detects_each_category() {
    let scan = scan_embedding_bag_pub_use_lines;
    assert_eq!(
        scan(EMBEDDING_BAG_MODE_APPROVED_LINE),
        vec![EMBEDDING_BAG_MODE_APPROVED_LINE.to_string()]
    );
    // 違反: 層型・別名・別経路・グループ形（承認形と文字列一致しない行として検出される）。
    for src in [
        "pub use fandhe_ai_autodiff::nn::Dropout2d;",
        "pub use fandhe_ai_autodiff::nn::AlphaDropout;",
        "pub use fandhe_ai_autodiff::nn::EmbeddingBag;",
        "pub use fandhe_ai_autodiff::nn::{EmbeddingBagVars, EmbeddingBagMode};",
        "pub use fandhe_ai_autodiff::nn::EmbeddingBagMode as Mode;",
        "pub(crate) use fandhe_ai_autodiff::nn::EmbeddingBag;",
    ] {
        let hits = scan(src);
        assert_eq!(hits.len(), 1, "src={src:?}");
        assert_ne!(hits[0], EMBEDDING_BAG_MODE_APPROVED_LINE, "src={src:?}");
    }
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use fandhe_ai_autodiff::nn::EmbeddingBag;",
        "let s = \"pub use x::EmbeddingBagMode;\";",
        "use fandhe_ai_autodiff::nn::EmbeddingBag;",
        "pub use fandhe_ai_autodiff::Var;",
    ] {
        assert!(scan(src).is_empty(), "src={src:?}");
    }
}

// =====================================================================
// イシュー #2530（ルート #2499 の一括承認）: MultiheadAttention の
// オプション（`MultiheadAttentionConfig`）の facade 公開を固定する正ガード群。
// 旧保留ガード（#2163。`MhaOptionsHoldDoctestGuard`・否定ガード）を反転した
// もので、公開面は承認形の 2 点（`compat::MultiheadAttentionConfig` の
// 再エクスポート 1 件・`compat::Sequential::add_multihead_attention_with_config`
// 1 件）に限り完全一致で固定する。根拠は `docs/autodiff-mha-options-decision.md`。
// =====================================================================

/// `src/compat` 配下の `fn add_multihead_attention_with_config` 宣言の
/// 件数と、宣言ファイルの相対パス一覧を返す（トークン単位。コメント・
/// 文字列中の同名テキストは数えない）。
fn mha_options_add_method_declarations(files: &[(String, String)]) -> Vec<String> {
    let mut found = Vec::new();
    for (rel, content) in files {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let n = count_fn_declarations_by_name(&tokens, "add_multihead_attention_with_config");
        for _ in 0..n {
            found.push(rel.clone());
        }
    }
    found
}

/// `add_multihead_attention_with_config` がちょうど 1 件・
/// `compat/sequential.rs` で宣言されていること（0 件・2 件以上・別ファイルは
/// fail-closed で失敗）。
#[test]
fn compat_sequential_declares_mha_options_add_method_exactly_once() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&compat_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    assert_eq!(
        mha_options_add_method_declarations(&files),
        vec!["sequential.rs".to_string()],
        "add_multihead_attention_with_config は compat/sequential.rs の 1 件のみ（イシュー #2530）"
    );
}

/// [`compat_sequential_declares_mha_options_add_method_exactly_once`] の
/// 自己テスト（0 件・2 重宣言・別ファイル宣言・コメント内を区別できること）。
#[test]
fn compat_sequential_declares_mha_options_add_method_detects_offense() {
    let decl = "impl S { pub fn add_multihead_attention_with_config(self) {} }";
    let files = vec![
        ("sequential.rs".to_string(), decl.to_string()),
        ("other.rs".to_string(), decl.to_string()),
        (
            "doc.rs".to_string(),
            "// fn add_multihead_attention_with_config(self) {}".to_string(),
        ),
        ("dup.rs".to_string(), format!("{decl} {decl}")),
    ];
    assert_eq!(
        mha_options_add_method_declarations(&files),
        vec!["sequential.rs", "other.rs", "dup.rs", "dup.rs"]
    );
    assert!(mha_options_add_method_declarations(&[]).is_empty());
}

/// `MultiheadAttentionConfig` を識別子として含む `pub use` 文（`;` 終端まで）
/// を `(ファイル相対パス, 空白なしトークン連結)` で全件返す。facade 独自の
/// `struct`／`type`／`enum` 宣言は `(ファイル, "<decl>")` として返す。
fn mha_config_exposures(files: &[(String, String)]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (rel, content) in files {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for (i, t) in tokens.iter().enumerate() {
            if matches!(t.as_str(), "struct" | "enum" | "type" | "union")
                && tokens.get(i + 1).map(String::as_str) == Some("MultiheadAttentionConfig")
            {
                out.push((rel.clone(), "<decl>".to_string()));
            }
            if t == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
                let end = tokens[i..]
                    .iter()
                    .position(|x| x == ";")
                    .map_or(tokens.len(), |p| i + p);
                let stmt = &tokens[i + 2..end];
                if stmt.iter().any(|x| x == "MultiheadAttentionConfig") {
                    out.push((rel.clone(), stmt.concat()));
                }
            }
        }
    }
    out
}

/// facade src 全体で `MultiheadAttentionConfig` の公開は
/// `compat/mod.rs` の `pub use fandhe_ai_autodiff::nn::MultiheadAttentionConfig;`
/// ちょうど 1 件（別名・グループ化・別ファイル・独自型宣言はすべて拒否）。
#[test]
fn facade_reexports_multihead_attention_config_only_from_compat() {
    let src_dir = facade_crate_root().join("src");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&src_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    assert_eq!(
        mha_config_exposures(&files),
        vec![(
            "compat/mod.rs".to_string(),
            "fandhe_ai_autodiff::nn::MultiheadAttentionConfig".to_string()
        )],
        "MultiheadAttentionConfig の facade 公開は compat/mod.rs の承認形 1 件のみ（イシュー #2530）"
    );
}

/// [`facade_reexports_multihead_attention_config_only_from_compat`] の自己テスト。
#[test]
fn facade_reexports_multihead_attention_config_detects_offense() {
    let f = |rel: &str, src: &str| vec![(rel.to_string(), src.to_string())];
    let ok = "pub use fandhe_ai_autodiff::nn::MultiheadAttentionConfig;";
    assert_eq!(
        mha_config_exposures(&f("compat/mod.rs", ok)),
        vec![(
            "compat/mod.rs".to_string(),
            "fandhe_ai_autodiff::nn::MultiheadAttentionConfig".to_string()
        )]
    );
    for bad in [
        "pub use fandhe_ai_autodiff::nn::MultiheadAttentionConfig as Cfg;",
        "pub use fandhe_ai_autodiff::nn::{Linear, MultiheadAttentionConfig};",
        "pub struct MultiheadAttentionConfig;",
    ] {
        let got = mha_config_exposures(&f("compat/mod.rs", bad));
        assert_eq!(got.len(), 1, "{bad}");
        assert_ne!(
            got[0].1, "fandhe_ai_autodiff::nn::MultiheadAttentionConfig",
            "{bad}"
        );
    }
    assert!(mha_config_exposures(&f("a.rs", "// pub use x::MultiheadAttentionConfig;")).is_empty());
}

/// 公開面だけで（`fandhe_ai::compat` のみ import）config を作り、
/// `add_multihead_attention_with_config` → `predict` まで通る（実行時の正プローブ）。
#[test]
fn multihead_attention_config_is_reachable_via_facade_only() {
    use fandhe_ai::compat::{MultiheadAttentionConfig, Sequential};
    let cfg = MultiheadAttentionConfig::new(4, 2).with_bias(false);
    let model = Sequential::new()
        .add_multihead_attention_with_config(cfg, 1)
        .expect("承認形の公開面で構築できる");
    assert_eq!(model.trainable_parameters().len(), 4);
}
// =====================================================================
// イシュー #2532・#2533（親 #2531・ルート #2499 の一括承認）: Transformer・
// TransformerDecoderLayer・TransformerConfig の facade 再エクスポートと
// `compat::Sequential::add_transformer_decoder_layer`／`add_transformer` の公開を固定する
// 正ガード群。旧保留ガード（#2165。#2533 で `TransformerDecoderHoldDoctestGuard` ごと撤去）
// を反転したもので、公開面は承認形（`nn/mod.rs` の 1 文の再エクスポート・
// `add_transformer_decoder_layer`／`add_transformer` 各 1 件）に完全一致で限定する。
// 根拠は `docs/autodiff-transformer-decoder-decision.md`。
// =====================================================================

/// `nn/mod.rs` の承認形（`pub use` 文のトークンを空白なしで連結したもの。#2533 で
/// `Transformer` を加えた 3 名形）。
const TRANSFORMER_DECODER_APPROVED_REEXPORT: &str =
    "fandhe_ai_autodiff::nn::{Transformer,TransformerConfig,TransformerDecoderLayer}";

/// `Transformer`／`TransformerDecoderLayer`／`TransformerConfig` を識別子として含む `pub use` 文を
/// `(ファイル相対パス, 空白なしトークン連結)` で、facade 独自の `struct`／`enum`／`type`／
/// `union` 宣言を `(ファイル, "<decl>")` で全件返す。コメント・文字列リテラルは無視する。
/// `pub(crate) use` は `pub` の次が `(` のため走査対象外だが、crate 内専用で公開面に現れない。
fn transformer_decoder_exposures(files: &[(String, String)]) -> Vec<(String, String)> {
    const NAMES: [&str; 3] = [
        "Transformer",
        "TransformerDecoderLayer",
        "TransformerConfig",
    ];
    let mut out = Vec::new();
    for (rel, content) in files {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for (i, t) in tokens.iter().enumerate() {
            if matches!(t.as_str(), "struct" | "enum" | "type" | "union")
                && tokens
                    .get(i + 1)
                    .is_some_and(|n| NAMES.contains(&n.as_str()))
            {
                out.push((rel.clone(), "<decl>".to_string()));
            }
            if t == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
                let end = tokens[i..]
                    .iter()
                    .position(|x| x == ";")
                    .map_or(tokens.len(), |p| i + p);
                let stmt = &tokens[i + 2..end];
                if stmt.iter().any(|x| NAMES.contains(&x.as_str())) {
                    out.push((rel.clone(), stmt.concat()));
                }
            }
        }
    }
    out
}

/// 公開は `nn/mod.rs` の承認形 1 文ちょうど 1 件（別名・分割・順序違い・別ファイル・独自型宣言は
/// すべて拒否。件数 0 の空振りも拒否）。
#[test]
fn facade_reexports_transformer_decoder_items_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&src_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    assert_eq!(
        transformer_decoder_exposures(&files),
        vec![(
            "nn/mod.rs".to_string(),
            TRANSFORMER_DECODER_APPROVED_REEXPORT.to_string()
        )],
        "Transformer／TransformerDecoderLayer／TransformerConfig の facade 公開は nn/mod.rs の承認形 1 文のみ（イシュー #2532・#2533）"
    );
}

/// [`facade_reexports_transformer_decoder_items_only_in_approved_shape`] の自己テスト
/// （承認形・各違反類型・無視されるべき入力を区別できること）。
#[test]
fn facade_reexports_transformer_decoder_items_only_in_approved_shape_detects_each_category() {
    let f = |rel: &str, src: &str| vec![(rel.to_string(), src.to_string())];
    let ok = "pub use fandhe_ai_autodiff::nn::{Transformer, TransformerConfig, TransformerDecoderLayer};";
    assert_eq!(
        transformer_decoder_exposures(&f("nn/mod.rs", ok)),
        vec![(
            "nn/mod.rs".to_string(),
            TRANSFORMER_DECODER_APPROVED_REEXPORT.to_string()
        )]
    );
    // 別ファイルは同一文でも (ファイル, 文) が承認形（nn/mod.rs）と一致しない。
    assert_eq!(
        transformer_decoder_exposures(&f("lib.rs", ok))[0].0,
        "lib.rs"
    );
    for bad in [
        "pub use fandhe_ai_autodiff::nn::TransformerDecoderLayer as Layer;",
        "pub use fandhe_ai_autodiff::nn::TransformerDecoderLayer;",
        "pub use fandhe_ai_autodiff::nn::{TransformerDecoderLayer, TransformerConfig};",
        "pub use fandhe_ai_autodiff::nn::{TransformerConfig, TransformerDecoderLayer};",
        "pub use fandhe_ai_autodiff::nn::{TransformerConfig, Transformer, TransformerDecoderLayer};",
        "pub use fandhe_ai_autodiff::nn::Transformer as T;",
        "pub use fandhe_ai_autodiff::nn::Transformer;",
        "pub struct Transformer;",
        "pub struct TransformerConfig;",
        "pub enum TransformerDecoderLayer {}",
    ] {
        let got = transformer_decoder_exposures(&f("nn/mod.rs", bad));
        assert!(!got.is_empty(), "{bad}");
        assert!(
            got.iter()
                .all(|(_, s)| s != TRANSFORMER_DECODER_APPROVED_REEXPORT),
            "{bad}"
        );
    }
    // 分割して 2 文にした場合は 2 件検出される（承認形 1 件の要求を満たさない）。
    assert_eq!(
        transformer_decoder_exposures(&f(
            "nn/mod.rs",
            "pub use a::TransformerConfig; pub use a::TransformerDecoderLayer;"
        ))
        .len(),
        2
    );
    for src in [
        "// pub use x::TransformerConfig;",
        "let s = \"pub use x::TransformerDecoderLayer;\";",
        "use fandhe_ai_autodiff::nn::TransformerConfig;",
        "use fandhe_ai_autodiff::nn::Transformer;",
    ] {
        assert!(
            transformer_decoder_exposures(&f("nn/mod.rs", src)).is_empty(),
            "{src}"
        );
    }
}

/// `add_transformer_decoder_layer` の `fn` 宣言を持つファイルの相対パスを宣言件数分返す
/// （トークン単位。コメント・文字列中の同名テキストは数えない）。
fn transformer_decoder_add_declarations(files: &[(String, String)]) -> Vec<String> {
    let mut found = Vec::new();
    for (rel, content) in files {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let n = count_fn_declarations_by_name(&tokens, "add_transformer_decoder_layer");
        for _ in 0..n {
            found.push(rel.clone());
        }
    }
    found
}

/// `add_transformer_decoder_layer` がちょうど 1 件・`compat/sequential.rs` で宣言されて
/// いること（0 件・2 件以上・別ファイルは fail-closed で失敗）。
#[test]
fn compat_sequential_declares_add_transformer_decoder_layer_exactly_once() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&compat_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    assert_eq!(
        transformer_decoder_add_declarations(&files),
        vec!["sequential.rs".to_string()],
        "add_transformer_decoder_layer は compat/sequential.rs の 1 件のみ（イシュー #2532）"
    );
}

/// [`compat_sequential_declares_add_transformer_decoder_layer_exactly_once`] の自己テスト。
#[test]
fn compat_sequential_declares_add_transformer_decoder_layer_detects_offense() {
    let decl = "impl S { pub fn add_transformer_decoder_layer(self) {} }";
    let files = vec![
        ("sequential.rs".to_string(), decl.to_string()),
        ("other.rs".to_string(), decl.to_string()),
        (
            "doc.rs".to_string(),
            "// fn add_transformer_decoder_layer(self) {}".to_string(),
        ),
        ("dup.rs".to_string(), format!("{decl} {decl}")),
        (
            "enc.rs".to_string(),
            "impl S { pub fn add_transformer_encoder(self) {} }".to_string(),
        ),
    ];
    assert_eq!(
        transformer_decoder_add_declarations(&files),
        vec!["sequential.rs", "other.rs", "dup.rs", "dup.rs"]
    );
    assert!(transformer_decoder_add_declarations(&[]).is_empty());
}

/// `add_transformer` の `fn` 宣言を持つファイルの相対パスを宣言件数分返す（イシュー #2533）。
/// トークン単位のため `add_transformer_encoder`／`add_transformer_decoder_layer` は数えない。
fn transformer_add_declarations(files: &[(String, String)]) -> Vec<String> {
    let mut found = Vec::new();
    for (rel, content) in files {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let n = count_fn_declarations_by_name(&tokens, "add_transformer");
        for _ in 0..n {
            found.push(rel.clone());
        }
    }
    found
}

/// `add_transformer` がちょうど 1 件・`compat/sequential.rs` で宣言されていること
/// （0 件・2 件以上・別ファイルは fail-closed で失敗）。
#[test]
fn compat_sequential_declares_add_transformer_exactly_once() {
    let compat_dir = facade_crate_root().join("src/compat");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&compat_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&compat_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    assert_eq!(
        transformer_add_declarations(&files),
        vec!["sequential.rs".to_string()],
        "add_transformer は compat/sequential.rs の 1 件のみ（イシュー #2533）"
    );
}

/// [`compat_sequential_declares_add_transformer_exactly_once`] の自己テスト。
#[test]
fn compat_sequential_declares_add_transformer_detects_offense() {
    let decl = "impl S { pub fn add_transformer(self) {} }";
    let files = vec![
        ("sequential.rs".to_string(), decl.to_string()),
        ("other.rs".to_string(), decl.to_string()),
        (
            "doc.rs".to_string(),
            "// fn add_transformer(self) {}".to_string(),
        ),
        ("dup.rs".to_string(), format!("{decl} {decl}")),
        (
            "enc.rs".to_string(),
            "impl S { pub fn add_transformer_encoder(self) {} pub fn add_transformer_decoder_layer(self) {} }"
                .to_string(),
        ),
    ];
    assert_eq!(
        transformer_add_declarations(&files),
        vec!["sequential.rs", "other.rs", "dup.rs", "dup.rs"]
    );
    assert!(transformer_add_declarations(&[]).is_empty());
}

/// 公開面だけで（`fandhe_ai` のみ import。`fandhe_ai_autodiff` は使わない）型名が解決でき、
/// `add_transformer_decoder_layer`／`add_transformer` まで通る（実行時の正プローブ）。
#[test]
fn transformer_decoder_types_are_reachable_via_facade_only() {
    use fandhe_ai::compat::Sequential;
    use fandhe_ai::nn::{Transformer, TransformerConfig, TransformerDecoderLayer};
    let cfg = TransformerConfig::new(4, 2)
        .with_num_encoder_layers(1)
        .with_num_decoder_layers(1)
        .with_dim_feedforward(8);
    assert_eq!(cfg.d_model(), 4);
    let _ = std::any::type_name::<Transformer>();
    // 型名がパスとして解決できること（構築は `FeedForwardActivation` 非公開のため行わない）。
    let _ = std::any::type_name::<TransformerDecoderLayer>();
    let model = Sequential::new()
        .add_transformer_decoder_layer(4, 2, 8, 1)
        .expect("承認形の公開面で構築できる");
    assert_eq!(model.trainable_parameters().len(), 26);
    let model = Sequential::new()
        .add_transformer(cfg, 1)
        .expect("承認形の公開面で構築できる");
    assert_eq!(model.trainable_parameters().len(), 16 + 2 + 26 + 2);
}
// =====================================================================
// イシュー #2160（親 #2131）: AdaptiveMaxPool2d／AdaptiveMaxPool1d／
// GlobalPool の facade 公開保留（#2527 で `Var` メソッド・`Sequential::add_*`・
// `GlobalPoolMode` は公開済みに反転。層型・`Tensor`／`Tape` メソッドのみ保留継続）
// を検査するテスト群。`AdaptiveMaxGlobalPoolHoldDoctestGuard`（`src/lib.rs`）の
// 正のプローブ 1 ブロック方式のドリフト検査を持つ。承認事項・多層防御の位置づけは
// `docs/autodiff-adaptive-max-global-pool-decision.md` §6・§8 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `AdaptiveMaxGlobalPoolHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックが glob import するネスト `pub mod`
/// 集合と、`src/lib.rs` の実際の `pub mod` 宣言集合が一致することを
/// 固定する（`conv3d_hold_doctest_globs_all_pub_modules` の
/// `AdaptiveMaxGlobalPoolHoldDoctestGuard` 版）。
#[test]
fn adaptive_max_global_pool_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines =
        extract_hold_doctest_guard_doc(&content, "AdaptiveMaxGlobalPoolHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "AdaptiveMaxGlobalPoolHoldDoctestGuard の doctest ブロックが glob\
         import するモジュール集合が src/lib.rs の pub mod 宣言集合と\
         ドリフトしている（declared={declared:?}, doctest={globbed:?}）。\
         新しい pub mod を追加した場合は doctest 側の use 一覧にも追加\
         すること。"
    );
}

/// [`adaptive_max_global_pool_hold_doctest_globs_all_pub_modules`] が
/// glob import 集合の一致のみを固定するのに対し、本テストは doctest
/// ブロックの **glob 以外の本文**が固定文言
/// [`ADAPTIVE_MAX_GLOBAL_POOL_HOLD_PROBE_BODY`] と 1 行たりとも違わず
/// 一致することを固定する（rustdoc の `# ` 隠し行・プローブの削除・
/// 別名へのシャドーイング等で正のプローブを骨抜きにする改変を機械的に
/// 拒否する）。
#[test]
fn adaptive_max_global_pool_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines =
        extract_hold_doctest_guard_doc(&content, "AdaptiveMaxGlobalPoolHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, ADAPTIVE_MAX_GLOBAL_POOL_HOLD_PROBE_BODY,
        "AdaptiveMaxGlobalPoolHoldDoctestGuard の doctest ブロック本文\
         （glob 以外）が固定文言 ADAPTIVE_MAX_GLOBAL_POOL_HOLD_PROBE_BODY\
         からドリフトしている。正のプローブ\
         （__FandheAdaptiveMaxPoolHoldProbe トレイト・__probe_* 関数）の削除・弱体化・隠し行の混入がないか\
         確認すること。"
    );
}

/// [`adaptive_max_global_pool_hold_doctest_probe_body_matches_fixed_
/// contract`] が要求する固定文言。`crates/facade/src/lib.rs` の
/// `AdaptiveMaxGlobalPoolHoldDoctestGuard` doc 内の唯一の doctest
/// ブロックから、ネスト `pub mod` の glob import 行（`use fandhe_ai::
/// <mod>::*;`）を除いた本文と 1 行単位で完全一致する必要がある
/// （クレートルート自体の `use fandhe_ai::*;` は本文に含む）。
const ADAPTIVE_MAX_GLOBAL_POOL_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
struct __FandheAdaptiveMaxPoolMarker;\n\
\n\
trait __FandheAdaptiveMaxPoolHoldProbe {\n\
\x20\x20\x20\x20fn adaptive_max_pool2d(&self) -> __FandheAdaptiveMaxPoolMarker;\n\
\x20\x20\x20\x20fn adaptive_max_pool1d(&self) -> __FandheAdaptiveMaxPoolMarker;\n\
}\n\
\n\
impl __FandheAdaptiveMaxPoolHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn adaptive_max_pool2d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }\n\
\x20\x20\x20\x20fn adaptive_max_pool1d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }\n\
}\n\
\n\
impl __FandheAdaptiveMaxPoolHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn adaptive_max_pool2d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }\n\
\x20\x20\x20\x20fn adaptive_max_pool1d(&self) -> __FandheAdaptiveMaxPoolMarker { __FandheAdaptiveMaxPoolMarker }\n\
}\n\
\n\
fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tensor::adaptive_max_pool2d(x);\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool2d();\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tensor::adaptive_max_pool1d(x);\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool1d();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tape::adaptive_max_pool2d(x);\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool2d();\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = fandhe_ai::Tape::adaptive_max_pool1d(x);\n\
\x20\x20\x20\x20let _: __FandheAdaptiveMaxPoolMarker = x.adaptive_max_pool1d();\n\
}";

// =====================================================================
// イシュー #2527（親 #2520・ルート #2499 の一括承認）: AdaptiveMaxPool2d／
// AdaptiveMaxPool1d／GlobalPool の facade 公開の正ガード。保留ガード
// （`AdaptiveMaxGlobalPoolHoldDoctestGuard`）から `Var`・`Sequential` の
// プローブを外し、承認形（`Var` の 1 行委譲メソッド 2 個・`compat::Sequential`
// の add_* 3 個・ルートの `GlobalPoolMode` 再エクスポート 1 行）だけを許す
// 形へ反転した。層型の再エクスポートと `Tensor`／`Tape` への同名メソッドは
// 未承認のまま禁止する。承認事項は
// `docs/autodiff-adaptive-max-global-pool-decision.md` §6・§8 参照。
// =====================================================================

/// #2527 で公開する 5 名の `fn` 名（`Var` 2 個 + `Sequential` 3 個）に、
/// 既存の定義元（内部クレート）を持つ同名 `fn` を含めた検査対象。
const ADAPTIVE_MAX_GLOBAL_POOL_FACADE_FN_NAMES: [&str; 5] = [
    "adaptive_max_pool2d",
    "adaptive_max_pool1d",
    "add_adaptive_max_pool2d",
    "add_adaptive_max_pool1d",
    "add_global_pool",
];

/// workspace 全体（`crates/*/src/`）で [`ADAPTIVE_MAX_GLOBAL_POOL_FACADE_FN_NAMES`]
/// の `fn` 宣言の定義元集合を固定する（過不足とも fail-closed。迂回実装の混入検出）。
/// `adaptive_max_pool2d` は #2160 の内部実装（`adaptive_max_pool_ops`・`eval`・
/// `backend-cpu`・`tensor-core` の BackendOps）にも同名定義があるため、それらを
/// 含む全集合で固定する。
#[test]
fn workspace_declares_adaptive_max_global_pool_facade_fn_names_only_in_approved_locations() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut crate_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&crates_dir)
        .expect("workspace crates ディレクトリが読めない")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(!crate_dirs.is_empty());
    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            let rel = path
                .strip_prefix(&crates_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/");
            for fn_name in ADAPTIVE_MAX_GLOBAL_POOL_FACADE_FN_NAMES {
                let count = count_fn_declarations_by_name(&tokens, fn_name);
                if count > 0 {
                    *found.entry(format!("{rel}::{fn_name}")).or_insert(0) += count;
                }
            }
        });
    }
    let expected: std::collections::BTreeMap<String, usize> = [
        // #2527 で公開した承認形。
        "autodiff/src/var.rs::adaptive_max_pool2d",
        "autodiff/src/var.rs::adaptive_max_pool1d",
        "facade/src/compat/sequential.rs::add_adaptive_max_pool2d",
        "facade/src/compat/sequential.rs::add_adaptive_max_pool1d",
        "facade/src/compat/sequential.rs::add_global_pool",
        // #2160 の内部実装（共有 forward・ホスト参照実装・BackendOps）。
        "autodiff/src/adaptive_max_pool_ops.rs::adaptive_max_pool2d",
        "autodiff/src/adaptive_max_pool_ops.rs::adaptive_max_pool1d",
        "autodiff/src/eval.rs::adaptive_max_pool2d",
        "backend-cpu/src/pooling.rs::adaptive_max_pool2d",
        "backend-cpu/src/ops.rs::adaptive_max_pool2d",
        "tensor-core/src/backend_ops.rs::adaptive_max_pool2d",
    ]
    .iter()
    .map(|k| (k.to_string(), 1usize))
    .collect();
    assert_eq!(
        found, expected,
        "#2527 の fn 宣言の定義元集合が承認形とずれている（新たな定義元が承認済みの\
         実装なのか迂回経路なのかを確認すること）: {found:?}"
    );
}

/// `var.rs` の 2 メソッド本体の承認形（非公開モジュール `adaptive_max_pool_ops` の
/// 共有 forward への 1 行委譲）。
const ADAPTIVE_MAX_POOL_VAR_EXPECTED_BODIES: [(&str, &str); 2] = [
    (
        "adaptive_max_pool2d",
        "crate : : adaptive_max_pool_ops : : adaptive_max_pool2d ( self , output_size )",
    ),
    (
        "adaptive_max_pool1d",
        "crate : : adaptive_max_pool_ops : : adaptive_max_pool1d ( self , output_size )",
    ),
];

/// `Var::adaptive_max_pool2d`／`adaptive_max_pool1d` が共有 forward への薄い委譲で
/// あることを固定する（スタブ・独自実装へのすり替えを拒否）。
#[test]
fn var_adaptive_max_pool_methods_are_thin_delegations() {
    let content = read_to_string_or_panic(&workspace_crates_dir().join("autodiff/src/var.rs"));
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, expected_body) in ADAPTIVE_MAX_POOL_VAR_EXPECTED_BODIES {
        let actual = determinism_fn_body(&tokens, name);
        assert_eq!(
            actual.as_deref(),
            Some(expected_body),
            "var.rs の `Var::{name}` の本体が承認形（共有 forward への 1 行委譲）と一致しない"
        );
    }
}

/// facade の `fandhe_ai::Var` 経由だけで 2 メソッドへ到達でき、シグネチャが
/// 承認形と一致し、実際に適用して期待 shape・値・索引が得られることを固定する。
#[test]
fn var_adaptive_max_pool_methods_are_reachable_via_facade_only() {
    use fandhe_ai::{AutodiffError, Tensor, Var};

    type Sig2<'t> = fn(&Var<'t>, [usize; 2]) -> Result<(Var<'t>, Tensor<i32>), AutodiffError>;
    type Sig1<'t> = fn(&Var<'t>, usize) -> Result<(Var<'t>, Tensor<i32>), AutodiffError>;
    fn sig2<'t>() -> Sig2<'t> {
        Var::<'t>::adaptive_max_pool2d
    }
    fn sig1<'t>() -> Sig1<'t> {
        Var::<'t>::adaptive_max_pool1d
    }

    let tape = fandhe_ai::tape();
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let x = tape.var(&Tensor::new(data, &[1, 1, 4, 4]).expect("tensor"));
    let (y, idx) = sig2()(&x, [2, 2]).expect("adaptive_max_pool2d");
    assert_eq!(y.to_tensor().shape(), [1, 1, 2, 2]);
    assert_eq!(
        y.to_tensor().host_slice().into_owned(),
        [5.0, 7.0, 13.0, 15.0]
    );
    assert_eq!(idx.host_slice().into_owned(), [5, 7, 13, 15]);
    assert!(sig2()(&x, [0, 2]).is_err());
    assert!(sig2()(&x, [2, 0]).is_err());

    let x1 = tape.var(&Tensor::new(vec![1.0_f32, 3.0, 2.0, 5.0], &[1, 1, 4]).expect("tensor"));
    let (y1, idx1) = sig1()(&x1, 2).expect("adaptive_max_pool1d");
    assert_eq!(y1.to_tensor().shape(), [1, 1, 2]);
    assert_eq!(y1.to_tensor().host_slice().into_owned(), [3.0, 5.0]);
    assert_eq!(idx1.host_slice().into_owned(), [1, 3]);
    assert!(sig1()(&x1, 0).is_err());
    // rank 不一致は Err（1d 層へ rank 4）。
    assert!(sig1()(&x, 2).is_err());
}

const ADD_ADAPTIVE_MAX_POOL2D_PARAMS: &str =
    "mut self, output_size: [usize; 2]) -> Result<Self, AutodiffError>";
const ADD_ADAPTIVE_MAX_POOL1D_PARAMS: &str =
    "mut self, output_size: usize) -> Result<Self, AutodiffError>";
const ADD_GLOBAL_POOL_PARAMS: &str = "mut self, mode: GlobalPoolMode, keepdims: bool) -> Self";

/// `compat::Sequential` の 3 add_* が承認シグネチャで 1 件ずつ存在する。
#[test]
fn compat_sequential_adaptive_max_global_pool_add_methods_have_approved_signatures() {
    let path = facade_crate_root().join("src/compat/sequential.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    for (name, params) in [
        ("add_adaptive_max_pool2d", ADD_ADAPTIVE_MAX_POOL2D_PARAMS),
        ("add_adaptive_max_pool1d", ADD_ADAPTIVE_MAX_POOL1D_PARAMS),
        ("add_global_pool", ADD_GLOBAL_POOL_PARAMS),
    ] {
        assert_eq!(count_fn_declarations_by_name(&tokens, name), 1, "{name}");
        assert!(
            sequential_spatial_add_signature_ok(&cleaned, name, params),
            "{name} のシグネチャが承認形と一致しない"
        );
    }
}

/// [`compat_sequential_adaptive_max_global_pool_add_methods_have_approved_signatures`]
/// の自己テスト。
#[test]
fn compat_sequential_adaptive_max_global_pool_add_methods_have_approved_signatures_detects_offense()
{
    let ok = "pub fn add_global_pool(mut self, mode: GlobalPoolMode, keepdims: bool,) -> Self {";
    assert!(sequential_spatial_add_signature_ok(
        ok,
        "add_global_pool",
        ADD_GLOBAL_POOL_PARAMS
    ));
    for bad in [
        "pub fn add_global_pool(mut self, mode: GlobalPoolMode, keepdims: bool) -> Result<Self, AutodiffError> {",
        "pub fn add_global_pool(mut self, mode: GlobalPoolMode) -> Self {",
        "pub fn add_global_pool(&mut self, mode: GlobalPoolMode, keepdims: bool) -> Self {",
        "pub fn add_linear(mut self) -> Self {",
    ] {
        assert!(
            !sequential_spatial_add_signature_ok(bad, "add_global_pool", ADD_GLOBAL_POOL_PARAMS),
            "{bad}"
        );
    }
    let ok2 = "pub fn add_adaptive_max_pool2d(mut self, output_size: [usize; 2]) -> Result<Self, AutodiffError> {";
    assert!(sequential_spatial_add_signature_ok(
        ok2,
        "add_adaptive_max_pool2d",
        ADD_ADAPTIVE_MAX_POOL2D_PARAMS
    ));
    assert!(!sequential_spatial_add_signature_ok(
        "pub fn add_adaptive_max_pool2d(mut self, output_size: usize) -> Result<Self, AutodiffError> {",
        "add_adaptive_max_pool2d",
        ADD_ADAPTIVE_MAX_POOL2D_PARAMS
    ));
}

/// ルートで再エクスポートを許す型の唯一の承認形（`src/lib.rs`）。
const GLOBAL_POOL_MODE_APPROVED_LINE: &str = "pub use fandhe_ai_autodiff::nn::GlobalPoolMode;";

/// facade が再エクスポートしてはならない層型（未承認）。
const ADAPTIVE_MAX_GLOBAL_POOL_LAYER_TYPE_NAMES: [&str; 3] =
    ["AdaptiveMaxPool2d", "AdaptiveMaxPool1d", "GlobalPool"];

/// facade src の 1 ファイル内容から、`GlobalPoolMode` または未承認の層型名を識別子として
/// 含む `pub use` 行（空白正規化済み）を集める検出本体。コメント・文字列リテラルは無視する。
fn scan_global_pool_pub_use_lines(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    cleaned
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("pub use") || l.starts_with("pub(crate) use"))
        .filter(|l| {
            line_contains_identifier(l, "GlobalPoolMode")
                || ADAPTIVE_MAX_GLOBAL_POOL_LAYER_TYPE_NAMES
                    .iter()
                    .any(|n| line_contains_identifier(l, n))
        })
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

/// `GlobalPoolMode` の再エクスポートが `src/lib.rs` の承認形 1 行だけで、
/// `AdaptiveMaxPool2d`／`AdaptiveMaxPool1d`／`GlobalPool` の層型が facade のどこからも
/// 再エクスポートされないことを固定する（`facade_reexports_topk_unique_types_only_in_
/// approved_shape` と同型）。
#[test]
fn facade_reexports_global_pool_mode_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_lib_rs = 0usize;
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in scan_global_pool_pub_use_lines(content) {
            if path.ends_with("src/lib.rs") && line == GLOBAL_POOL_MODE_APPROVED_LINE {
                approved_in_lib_rs += 1;
            } else {
                offending.push(format!("{}: `{line}`", path.display()));
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade が GlobalPoolMode を承認形（src/lib.rs の `{GLOBAL_POOL_MODE_APPROVED_LINE}` \
         1 行）以外で、または層型（AdaptiveMaxPool2d／AdaptiveMaxPool1d／GlobalPool）を再エクスポート\
         している（#2527）: {offending:?}"
    );
    assert_eq!(
        approved_in_lib_rs, 1,
        "src/lib.rs に承認形の再エクスポート行がちょうど 1 行存在しない\
         （検査対象を見失った場合を含む）"
    );
}

/// [`scan_global_pool_pub_use_lines`] の自己テスト（合成入力）。
#[test]
fn facade_reexports_global_pool_mode_only_in_approved_shape_detects_each_category() {
    let scan = scan_global_pool_pub_use_lines;
    assert_eq!(
        scan(GLOBAL_POOL_MODE_APPROVED_LINE),
        vec![GLOBAL_POOL_MODE_APPROVED_LINE.to_string()]
    );
    // 違反: 層型・別名・別経路・グループ形（承認形と文字列一致しない行として検出される）。
    for src in [
        "pub use fandhe_ai_autodiff::nn::GlobalPool;",
        "pub use fandhe_ai_autodiff::nn::AdaptiveMaxPool2d;",
        "pub use fandhe_ai_autodiff::nn::{AdaptiveMaxPool1d, GlobalPoolMode};",
        "pub use fandhe_ai_autodiff::nn::GlobalPoolMode as Mode;",
        "pub(crate) use fandhe_ai_autodiff::nn::GlobalPool;",
    ] {
        let hits = scan(src);
        assert_eq!(hits.len(), 1, "src={src:?}");
        assert_ne!(hits[0], GLOBAL_POOL_MODE_APPROVED_LINE, "src={src:?}");
    }
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use fandhe_ai_autodiff::nn::GlobalPool;",
        "let s = \"pub use x::GlobalPoolMode;\";",
        "use fandhe_ai_autodiff::nn::GlobalPool;",
        "pub use fandhe_ai_autodiff::Var;",
    ] {
        assert!(scan(src).is_empty(), "src={src:?}");
    }
}
// =====================================================================
// イシュー #2164（親 #2131）: RNN／LSTM／GRU の多層・双方向・dropout
// （`RnnConfig`）の facade 公開保留を検査するテスト群。
// `RnnConfigHoldDoctestGuard`（`src/lib.rs`）の正のプローブ 1 ブロック
// 方式のドリフト検査を `spatial_layers_hold_doctest_*` と同型で持つ。
// 承認事項・多層防御の位置づけは
// `docs/autodiff-rnn-stacked-config-decision.md` §8 参照。
// =====================================================================

/// `RnnConfigHoldDoctestGuard` の唯一の doctest ブロックが glob import
/// するネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod` 宣言
/// 集合が一致することを固定する（`spatial_layers_hold_doctest_globs_
/// all_pub_modules` と同型。イシュー #2164）。
#[test]
fn rnn_config_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "RnnConfigHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "RnnConfigHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトして\
         いる（declared={declared:?}, doctest={globbed:?}）。新しい pub\
         mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`rnn_config_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`RNN_CONFIG_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（rustdoc の `# ` 隠し行・
/// プローブの削除・別名へのシャドーイング等で正のプローブを骨抜きに
/// する改変を機械的に拒否する。イシュー #2164）。
#[test]
fn rnn_config_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "RnnConfigHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, RNN_CONFIG_HOLD_PROBE_BODY,
        "RnnConfigHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 RNN_CONFIG_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_rnn_config_hold_probe モジュール・\
         __FandheRnnConfigTapeProbe／__FandheRnnConfigWithConfigProbe\
         トレイト・__probe 関数）の削除・弱体化・隠し行の混入がないか\
         確認すること。"
    );
}

/// [`rnn_config_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `RnnConfigHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const RNN_CONFIG_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_rnn_config_hold_probe {\n\
\x20\x20\x20\x20pub struct RnnConfig;\n\
\x20\x20\x20\x20pub struct StackedRnn;\n\
\x20\x20\x20\x20pub struct StackedLstm;\n\
\x20\x20\x20\x20pub struct StackedGru;\n\
\x20\x20\x20\x20pub struct StackedRnnSeqOutput;\n\
\x20\x20\x20\x20pub struct StackedLstmSeqOutput;\n\
}\n\
use __fandhe_rnn_config_hold_probe::*;\n\
\n\
struct __FandheRnnConfigMarker;\n\
\n\
trait __FandheRnnConfigTapeProbe {\n\
\x20\x20\x20\x20fn stacked_rnn_forward_seq(&self) -> __FandheRnnConfigMarker;\n\
\x20\x20\x20\x20fn stacked_lstm_forward_seq(&self) -> __FandheRnnConfigMarker;\n\
\x20\x20\x20\x20fn stacked_gru_forward_seq(&self) -> __FandheRnnConfigMarker;\n\
}\n\
\n\
impl __FandheRnnConfigTapeProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn stacked_rnn_forward_seq(&self) -> __FandheRnnConfigMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheRnnConfigMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn stacked_lstm_forward_seq(&self) -> __FandheRnnConfigMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheRnnConfigMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn stacked_gru_forward_seq(&self) -> __FandheRnnConfigMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheRnnConfigMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
trait __FandheRnnConfigWithConfigProbe {\n\
\x20\x20\x20\x20fn with_config(&self) -> __FandheRnnConfigMarker;\n\
}\n\
\n\
impl __FandheRnnConfigWithConfigProbe for fandhe_ai::nn::rnn::Rnn {\n\
\x20\x20\x20\x20fn with_config(&self) -> __FandheRnnConfigMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheRnnConfigMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheRnnConfigWithConfigProbe for fandhe_ai::nn::rnn::Lstm {\n\
\x20\x20\x20\x20fn with_config(&self) -> __FandheRnnConfigMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheRnnConfigMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheRnnConfigWithConfigProbe for fandhe_ai::nn::rnn::Gru {\n\
\x20\x20\x20\x20fn with_config(&self) -> __FandheRnnConfigMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheRnnConfigMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20_: RnnConfig,\n\
\x20\x20\x20\x20_: StackedRnn,\n\
\x20\x20\x20\x20_: StackedLstm,\n\
\x20\x20\x20\x20_: StackedGru,\n\
\x20\x20\x20\x20_: StackedRnnSeqOutput,\n\
\x20\x20\x20\x20_: StackedLstmSeqOutput,\n\
\x20\x20\x20\x20tape: &fandhe_ai::Tape,\n\
\x20\x20\x20\x20rnn: &fandhe_ai::nn::rnn::Rnn,\n\
\x20\x20\x20\x20lstm: &fandhe_ai::nn::rnn::Lstm,\n\
\x20\x20\x20\x20gru: &fandhe_ai::nn::rnn::Gru,\n\
) {\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = fandhe_ai::Tape::stacked_rnn_forward_seq(tape);\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = tape.stacked_rnn_forward_seq();\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = fandhe_ai::Tape::stacked_lstm_forward_seq(tape);\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = tape.stacked_lstm_forward_seq();\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = fandhe_ai::Tape::stacked_gru_forward_seq(tape);\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = tape.stacked_gru_forward_seq();\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = fandhe_ai::nn::rnn::Rnn::with_config(rnn);\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = rnn.with_config();\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = fandhe_ai::nn::rnn::Lstm::with_config(lstm);\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = lstm.with_config();\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = fandhe_ai::nn::rnn::Gru::with_config(gru);\n\
\x20\x20\x20\x20let _: __FandheRnnConfigMarker = gru.with_config();\n\
}";

// =====================================================================
// #2166（親 #2131）の facade 公開保留固定（`LossOpsHoldDoctestGuard`）。
// `VarReduceOpsHoldDoctestGuard`（イシュー #2147。#2514 で削除済み）と同型の正のプローブ
// 1 ブロック方式のドリフト検査に加え、workspace 全体のソース走査による
// 定義元インベントリを持つ。承認事項・多層防御の位置づけは
// `docs/autodiff-loss-ops-decision.md` §5 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `LossOpsHoldDoctestGuard` doc 内の
/// 唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （`reduce_ops_hold_doctest_globs_all_pub_modules`〈#2514 で削除済み〉の
/// `LossOpsHoldDoctestGuard` 版）。
#[test]
fn loss_ops_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "LossOpsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "LossOpsHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`loss_ops_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`LOSS_OPS_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（`reduce_ops_hold_
/// doctest_probe_body_matches_fixed_contract`と同じ理由: rustdoc の
/// `# ` 隠し行・プローブの削除・別名へのシャドーイング等で正のプローブ
/// を骨抜きにする改変を機械的に拒否する）。
#[test]
fn loss_ops_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "LossOpsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, LOSS_OPS_HOLD_PROBE_BODY,
        "LossOpsHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 LOSS_OPS_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_loss_hold_probe モジュール・\
         __FandheLossHoldProbe トレイト・__probe_* 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`loss_ops_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `LossOpsHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const LOSS_OPS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_loss_hold_probe {\n\
\x20\x20\x20\x20pub mod loss_ops {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn l1_loss() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn cross_entropy_loss_with() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn cosine_embedding_loss() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn margin_ranking_loss() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn triplet_margin_loss() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn poisson_nll_loss() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn ctc_loss() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_loss_hold_probe::*;\n\
\n\
struct __FandheLossMarker;\n\
\n\
trait __FandheLossHoldProbe {\n\
\x20\x20\x20\x20fn l1_loss(&self) -> __FandheLossMarker;\n\
\x20\x20\x20\x20fn cross_entropy_loss_with(&self) -> __FandheLossMarker;\n\
\x20\x20\x20\x20fn cosine_embedding_loss(&self) -> __FandheLossMarker;\n\
\x20\x20\x20\x20fn margin_ranking_loss(&self) -> __FandheLossMarker;\n\
\x20\x20\x20\x20fn triplet_margin_loss(&self) -> __FandheLossMarker;\n\
\x20\x20\x20\x20fn poisson_nll_loss(&self) -> __FandheLossMarker;\n\
\x20\x20\x20\x20fn ctc_loss(&self) -> __FandheLossMarker;\n\
}\n\
\n\
impl<'t> __FandheLossHoldProbe for fandhe_ai::Var<'t> {\n\
\x20\x20\x20\x20fn l1_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn cross_entropy_loss_with(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn cosine_embedding_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn margin_ranking_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn triplet_margin_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn poisson_nll_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn ctc_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
}\n\
\n\
impl __FandheLossHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn l1_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn cross_entropy_loss_with(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn cosine_embedding_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn margin_ranking_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn triplet_margin_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn poisson_nll_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn ctc_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
}\n\
\n\
impl __FandheLossHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn l1_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn cross_entropy_loss_with(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn cosine_embedding_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn margin_ranking_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn triplet_margin_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn poisson_nll_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
\x20\x20\x20\x20fn ctc_loss(&self) -> __FandheLossMarker { __FandheLossMarker }\n\
}\n\
\n\
fn __probe_free_fns() {\n\
\x20\x20\x20\x20// `loss_ops::` を経由した経路解決（`use fandhe_ai::*;` が\n\
\x20\x20\x20\x20// 同名モジュールを glob 公開していれば、名前解決自体が曖昧に\n\
\x20\x20\x20\x20// なり E0659 でコンパイル失敗する）。\n\
\x20\x20\x20\x20loss_ops::l1_loss();\n\
\x20\x20\x20\x20loss_ops::cross_entropy_loss_with();\n\
\x20\x20\x20\x20loss_ops::cosine_embedding_loss();\n\
\x20\x20\x20\x20loss_ops::margin_ranking_loss();\n\
\x20\x20\x20\x20loss_ops::triplet_margin_loss();\n\
\x20\x20\x20\x20loss_ops::poisson_nll_loss();\n\
\x20\x20\x20\x20loss_ops::ctc_loss();\n\
}\n\
\n\
fn __probe_var(x: &fandhe_ai::Var<'_>) {\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::l1_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.l1_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::cross_entropy_loss_with(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.cross_entropy_loss_with();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::cosine_embedding_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.cosine_embedding_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::margin_ranking_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.margin_ranking_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::triplet_margin_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.triplet_margin_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::poisson_nll_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.poisson_nll_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Var::ctc_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.ctc_loss();\n\
}\n\
\n\
fn __probe_tensor_f32(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tensor::l1_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.l1_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tensor::cosine_embedding_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.cosine_embedding_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tensor::margin_ranking_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.margin_ranking_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tensor::ctc_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.ctc_loss();\n\
}\n\
\n\
fn __probe_tape(x: &fandhe_ai::Tape) {\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tape::cross_entropy_loss_with(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.cross_entropy_loss_with();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tape::triplet_margin_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.triplet_margin_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tape::poisson_nll_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.poisson_nll_loss();\n\
\x20\x20\x20\x20let _: __FandheLossMarker = fandhe_ai::Tape::ctc_loss(x);\n\
\x20\x20\x20\x20let _: __FandheLossMarker = x.ctc_loss();\n\
}";

/// `l1_loss`・`cross_entropy_loss_with`（イシュー #2166）・
/// `cosine_embedding_loss`・`margin_ranking_loss`・
/// `triplet_margin_loss`・`poisson_nll_loss`（イシュー #2167）・
/// `ctc_loss`（イシュー #2168。7 個の関数名）。
/// [`facade_does_not_reexport_or_declare_loss_ops`]・
/// [`workspace_declares_loss_ops_fn_names_only_in_allowed_locations`]
/// が共用する。
const LOSS_OPS_FN_NAMES: [&str; 7] = [
    "l1_loss",
    "cross_entropy_loss_with",
    "cosine_embedding_loss",
    "margin_ranking_loss",
    "triplet_margin_loss",
    "poisson_nll_loss",
    "ctc_loss",
];

/// facade src 全体（`crates/facade/src/**`）に、`loss_ops` を参照
/// する `pub use`（`pub use fandhe_ai_autodiff::loss_ops;` 等の
/// モジュール再エクスポート・別名含む）も、[`LOSS_OPS_FN_NAMES`]
/// （2 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する（`LossOpsHoldDoctestGuard` の正のプローブと多層防御を
/// 成す最内層のソース走査ガード。`facade_does_not_reexport_or_declare_
/// reduce_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_loss_ops() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") && line_contains_identifier(trimmed, "loss_ops") {
                offending.push(format!(
                    "{}: `{trimmed}` が `loss_ops` を識別子単位で含む",
                    path.display()
                ));
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in LOSS_OPS_FN_NAMES {
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
        "facade の公開面が loss_ops（イシュー #2166 の内部クレート限定\
         新規公開面。facade 公開は承認待ちのため対象外という設計判断に\
         違反）を再エクスポート、または同名の fn を宣言している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`LOSS_OPS_FN_NAMES`]
/// （7 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_
/// reduce_ops_fn_names_only_in_allowed_locations` と同型のインベン
/// トリ）。
///
/// **期待集合**（着手前確認の再 grep で判明。実装計画「インベントリを
/// 実測する」手順）: `l1_loss`・`cross_entropy_loss_with`（イシュー
/// #2166）・`cosine_embedding_loss`・`margin_ranking_loss`・
/// `triplet_margin_loss`・`poisson_nll_loss`（イシュー #2167）・
/// `ctc_loss`（イシュー #2168）は
/// いずれも `crates/autodiff/src/loss_ops.rs` にのみ 1 件ずつ存在する。
#[test]
fn workspace_declares_loss_ops_fn_names_only_in_allowed_locations() {
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
            for fn_name in LOSS_OPS_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("autodiff/src/loss_ops.rs::l1_loss", 1usize),
        ("autodiff/src/loss_ops.rs::cross_entropy_loss_with", 1usize),
        ("autodiff/src/loss_ops.rs::cosine_embedding_loss", 1usize),
        ("autodiff/src/loss_ops.rs::margin_ranking_loss", 1usize),
        ("autodiff/src/loss_ops.rs::triplet_margin_loss", 1usize),
        ("autodiff/src/loss_ops.rs::poisson_nll_loss", 1usize),
        ("autodiff/src/loss_ops.rs::ctc_loss", 1usize),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の loss_ops 系 `fn` 宣言集合が\
         期待（`crates/autodiff/src/loss_ops.rs` 7 件）と一致しない\
         （過不足いずれも fail-closed に検出する。新たな定義元が\
         見つかった場合、それが承認済みの実装なのか迂回経路の混入なのか\
         を確認すること）: {found:?}"
    );
}

// =====================================================================
// #2173（親 #2131）の facade 公開保留固定（`ParamGroupsHoldDoctestGuard`）。
// `RnnConfigHoldDoctestGuard`（イシュー #2164）と同型の正のプローブ
// 1 ブロック方式のドリフト検査に加え、workspace 全体のソース走査による
// 定義元インベントリを持つ。承認事項の位置づけは
// `docs/autodiff-param-groups-decision.md` §5 参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `ParamGroupsHoldDoctestGuard` doc 内の
/// 唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （`rnn_config_hold_doctest_globs_all_pub_modules` の
/// `ParamGroupsHoldDoctestGuard` 版）。
#[test]
fn param_groups_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "ParamGroupsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "ParamGroupsHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`param_groups_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`PARAM_GROUPS_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（rustdoc の `# ` 隠し行・
/// プローブの削除・別名へのシャドーイング等で正のプローブを骨抜きに
/// する改変を機械的に拒否する。イシュー #2173）。
#[test]
fn param_groups_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "ParamGroupsHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, PARAM_GROUPS_HOLD_PROBE_BODY,
        "ParamGroupsHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 PARAM_GROUPS_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_param_groups_hold_probe モジュール・\
         __FandheParamGroupsCompileProbe／__FandheParamGroupsStepProbe\
         トレイト・__probe 関数）の削除・弱体化・隠し行の混入がないか\
         確認すること。"
    );
}

/// [`param_groups_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `ParamGroupsHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const PARAM_GROUPS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_param_groups_hold_probe {\n\
\x20\x20\x20\x20pub struct ParamGroup;\n\
\x20\x20\x20\x20pub trait ParamGroupStep {}\n\
}\n\
use __fandhe_param_groups_hold_probe::*;\n\
\n\
fn __probe_trait<T: ?Sized + ParamGroupStep>() {}\n\
\n\
struct __FandheParamGroupsMarker;\n\
\n\
trait __FandheParamGroupsCompileProbe {\n\
\x20\x20\x20\x20fn compile_with_param_groups(&self) -> __FandheParamGroupsMarker;\n\
}\n\
\n\
impl __FandheParamGroupsCompileProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn compile_with_param_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
trait __FandheParamGroupsStepProbe {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker;\n\
}\n\
\n\
impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Sgd {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheParamGroupsStepProbe for fandhe_ai::optim::AdamW {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Adam {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheParamGroupsStepProbe for fandhe_ai::optim::RmsProp {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Adagrad {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheParamGroupsStepProbe for fandhe_ai::optim::Lamb {\n\
\x20\x20\x20\x20fn step_with_groups(&self) -> __FandheParamGroupsMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheParamGroupsMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20_: ParamGroup,\n\
\x20\x20\x20\x20seq: &fandhe_ai::compat::Sequential,\n\
\x20\x20\x20\x20sgd: &fandhe_ai::optim::Sgd,\n\
\x20\x20\x20\x20adamw: &fandhe_ai::optim::AdamW,\n\
\x20\x20\x20\x20adam: &fandhe_ai::optim::Adam,\n\
\x20\x20\x20\x20rmsprop: &fandhe_ai::optim::RmsProp,\n\
\x20\x20\x20\x20adagrad: &fandhe_ai::optim::Adagrad,\n\
\x20\x20\x20\x20lamb: &fandhe_ai::optim::Lamb,\n\
) {\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::compile_with_param_groups(seq);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = seq.compile_with_param_groups();\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = fandhe_ai::optim::Sgd::step_with_groups(sgd);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = sgd.step_with_groups();\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = fandhe_ai::optim::AdamW::step_with_groups(adamw);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = adamw.step_with_groups();\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = fandhe_ai::optim::Adam::step_with_groups(adam);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = adam.step_with_groups();\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = fandhe_ai::optim::RmsProp::step_with_groups(rmsprop);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = rmsprop.step_with_groups();\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = fandhe_ai::optim::Adagrad::step_with_groups(adagrad);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = adagrad.step_with_groups();\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = fandhe_ai::optim::Lamb::step_with_groups(lamb);\n\
\x20\x20\x20\x20let _: __FandheParamGroupsMarker = lamb.step_with_groups();\n\
}";

/// `ParamGroup`・`ParamGroupStep`・`step_with_groups`・
/// `compile_with_param_groups`（イシュー #2173。4 個の識別子・fn 名）。
/// [`facade_does_not_reexport_or_declare_param_groups`]・
/// [`workspace_declares_param_group_fn_names_only_in_allowed_locations`]
/// が共用する。
const PARAM_GROUPS_FN_NAMES: [&str; 2] = ["step_with_groups", "compile_with_param_groups"];

/// facade src 全体（`crates/facade/src/**`）に、`ParamGroup`／
/// `ParamGroupStep`／`param_group` を識別子単位で含む `pub use`
/// （別名・ネストした group 経由含む）も、[`PARAM_GROUPS_FN_NAMES`]
/// （2 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）も存在しないことを
/// 固定する（`ParamGroupsHoldDoctestGuard` の正のプローブと多層防御を
/// 成す最内層のソース走査ガード。`facade_does_not_reexport_or_declare_
/// loss_ops` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_param_groups() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") {
                for ident in ["ParamGroup", "ParamGroupStep", "param_group"] {
                    if line_contains_identifier(trimmed, ident) {
                        offending.push(format!(
                            "{}: `{trimmed}` が `{ident}` を識別子単位で含む",
                            path.display()
                        ));
                    }
                }
            }
        }
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        for fn_name in PARAM_GROUPS_FN_NAMES {
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
        "facade の公開面が param groups（イシュー #2173 の内部クレート限定\
         新規公開面。facade 公開は承認待ちのため対象外という設計判断に\
         違反）を再エクスポート、または同名の fn を宣言している: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、`step_with_groups`・
/// `step_with_slot_hparams`・`resolve_slot_hparams`・
/// `compile_with_param_groups` の `fn` 宣言の定義元集合を固定する
/// （`workspace_declares_loss_ops_fn_names_only_in_allowed_locations`
/// と同型のインベントリ）。
///
/// **期待集合**（イシュー #2298 で `Adadelta`／`Adamax`／`NAdam`／
/// `RAdam` の 4 impl・4 `step_with_slot_hparams` を追加。着手前確認の
/// 再 grep で判明）: `step_with_groups`（trait 宣言 1 件 + impl 10 件、
/// いずれも `crates/autodiff/src/nn/optim/param_group.rs`）・
/// `resolve_slot_hparams`（同ファイルに 1 件）・`step_with_slot_hparams`
/// （`adamw.rs`・`adam.rs`・`rmsprop.rs`・`adagrad.rs`・`lamb.rs`・
/// `optim/sgd.rs`・`adadelta.rs`・`adamax.rs`・`nadam.rs`・`radam.rs` に
/// 各 1 件）・`compile_with_param_groups`（0 件。承認待ちのため未実装）。
#[test]
fn workspace_declares_param_group_fn_names_only_in_allowed_locations() {
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

    const NAMES: [&str; 4] = [
        "step_with_groups",
        "resolve_slot_hparams",
        "step_with_slot_hparams",
        "compile_with_param_groups",
    ];

    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            for fn_name in NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        (
            "autodiff/src/nn/optim/param_group.rs::step_with_groups",
            11usize,
        ),
        (
            "autodiff/src/nn/optim/param_group.rs::resolve_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/adamw.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/adam.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/rmsprop.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/adagrad.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/lamb.rs::step_with_slot_hparams",
            1usize,
        ),
        ("autodiff/src/optim/sgd.rs::step_with_slot_hparams", 1usize),
        (
            "autodiff/src/nn/optim/adadelta.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/adamax.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/nadam.rs::step_with_slot_hparams",
            1usize,
        ),
        (
            "autodiff/src/nn/optim/radam.rs::step_with_slot_hparams",
            1usize,
        ),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の param groups 系 `fn` 宣言集合が\
         期待と一致しない（過不足いずれも fail-closed に検出する。\
         `compile_with_param_groups` は 0 件のはず——承認前に実装が\
         紛れ込んでいないかを含めて確認すること）: {found:?}"
    );
}
// =====================================================================
// イシュー #2171（親 #2131）→ #2501（親 #2499）: Adadelta／Adamax／
// NAdam／RAdam の facade 公開（承認形の正ガード）。#2171 では公開保留を
// `OptimizerExtHoldDoctestGuard`（`src/lib.rs`）と否定テストで固定して
// いたが、#2501 で `fandhe_ai::optim` からの素の再エクスポートを承認形
// （`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md`
// §8）として公開したため、doctest ガードとそのドリフト検査 2 件を削除し、
// ソース走査ガードを「承認した形だけを許す」正ガードへ反転した。
// =====================================================================

/// 承認形で公開する 8 識別子（Adadelta／Adamax／NAdam／RAdam と各 `*Config`）。
const OPTIMIZER_EXT_NAMES: [&str; 8] = [
    "Adadelta",
    "AdadeltaConfig",
    "Adamax",
    "AdamaxConfig",
    "NAdam",
    "NAdamConfig",
    "RAdam",
    "RAdamConfig",
];

/// `pub use` の path トークン列が `fandhe_ai_autodiff :: nn :: optim :: …`
/// で始まるか（承認形の接頭辞）を判定する。
fn optimizer_ext_approved_prefix(path_tokens: &[String]) -> bool {
    let want = [
        "fandhe_ai_autodiff",
        ":",
        ":",
        "nn",
        ":",
        ":",
        "optim",
        ":",
        ":",
    ];
    path_tokens.len() >= want.len() && path_tokens.iter().zip(want).all(|(a, b)| a == b)
}

/// [`facade_reexports_optimizer_ext_items_only_in_approved_shape`]・その
/// 自己テストが共用する検出本体。`content`（1 ファイル分のソース）の
/// `pub use` から [`collect_pub_use_leaves`] で 8 識別子の葉を集め、
/// 承認形（接頭辞が `fandhe_ai_autodiff::nn::optim::` で `as` 別名を
/// 伴わない）の出現葉を第 1 要素、承認形から外れる出現（別 path 接頭辞・
/// 別名）と同名の `trait`／`struct`／`enum`／`type` 独自宣言を第 2 要素
/// （違反）として返す。コメント・文字列リテラル中の出現と非公開 `use` は
/// 無視する。
fn scan_optimizer_ext_reexports_and_declarations(content: &str) -> (Vec<String>, Vec<String>) {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut approved: Vec<String> = Vec::new();
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let hits: Vec<String> = collect_pub_use_leaves(path_tokens)
                .into_iter()
                .filter(|leaf| OPTIMIZER_EXT_NAMES.contains(&leaf.as_str()))
                .collect();
            let shape_ok = optimizer_ext_approved_prefix(path_tokens)
                && !path_tokens.iter().any(|t| t == "as");
            for leaf in hits {
                if shape_ok {
                    approved.push(leaf);
                } else {
                    offending.push(format!("承認形外の pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens
                .get(i + 1)
                .map(|t| OPTIMIZER_EXT_NAMES.contains(&t.as_str()))
                .unwrap_or(false)
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    (approved, offending)
}

/// 承認形の正ガード（#2501 で `facade_does_not_reexport_or_declare_
/// optimizer_ext_items` から反転）。facade src 全体で、8 識別子がそれぞれ
/// `src/optim.rs` の `pub use fandhe_ai_autodiff::nn::optim::…`（別名なし）
/// としてちょうど 1 回だけ出現し、承認形外の再エクスポート・同名の独自
/// 宣言が存在しないことを fail-closed に固定する。
#[test]
fn facade_reexports_optimizer_ext_items_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    let mut approved_in_optim_rs: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let (approved, bad) = scan_optimizer_ext_reexports_and_declarations(content);
        for offense in bad {
            offending.push(format!("{}: {offense}", path.display()));
        }
        if path.ends_with("src/optim.rs") {
            approved_in_optim_rs.extend(approved);
        } else {
            for leaf in approved {
                offending.push(format!(
                    "{}: optim.rs 以外での pub use leaf={leaf}",
                    path.display()
                ));
            }
        }
    });
    approved_in_optim_rs.sort();
    let mut expected: Vec<String> = OPTIMIZER_EXT_NAMES.iter().map(|s| s.to_string()).collect();
    expected.sort();
    assert!(
        offending.is_empty(),
        "facade の公開面が optimizer 拡張（#2501。Adadelta／Adamax／NAdam／RAdam）を\
         承認形（src/optim.rs の `pub use fandhe_ai_autodiff::nn::optim::…`・別名なし）\
         以外で再エクスポート、または独自宣言している（`docs/autodiff-optimizer-\
         adadelta-adamax-nadam-radam-decision.md` §8）: {offending:?}"
    );
    assert_eq!(
        approved_in_optim_rs, expected,
        "src/optim.rs に承認形の 8 識別子がちょうど 1 回ずつ存在しない\
         （過不足・重複いずれも fail。検査対象を見失った場合を含む）"
    );
}

/// [`scan_optimizer_ext_reexports_and_declarations`] の自己テスト
/// （正例・負例の合成入力）。
#[test]
fn facade_reexports_optimizer_ext_items_only_in_approved_shape_detects_each_category() {
    let scan = scan_optimizer_ext_reexports_and_declarations;
    // 承認形: group・単一識別子。違反なし・葉が承認側に載る。
    let (ok, bad) = scan("pub use fandhe_ai_autodiff::nn::optim::{Adadelta, AdadeltaConfig};");
    assert!(bad.is_empty());
    assert_eq!(ok, vec!["Adadelta", "AdadeltaConfig"]);
    let (ok, bad) = scan("pub use fandhe_ai_autodiff::nn::optim::NAdam;");
    assert!(bad.is_empty());
    assert_eq!(ok, vec!["NAdam"]);
    // 違反: 別名。
    assert!(
        !scan("pub use fandhe_ai_autodiff::nn::optim::NAdam as Foo;")
            .1
            .is_empty()
    );
    // 違反: 別 path 接頭辞。
    assert!(
        !scan("pub use fandhe_ai_autodiff::optim::Adamax;")
            .1
            .is_empty()
    );
    // 違反: 独自宣言。
    assert!(!scan("pub struct RAdamConfig;").1.is_empty());
    // 無視される: コメント・文字列リテラル・非公開 use・無関係な pub use。
    for src in [
        "// pub use ...::Adadelta;",
        "let s = \"Adamax\";",
        "use fandhe_ai_autodiff::nn::optim::RAdam;",
        "pub use fandhe_ai_autodiff::nn::optim::AdamW;",
    ] {
        let (ok, bad) = scan(src);
        assert!(ok.is_empty() && bad.is_empty(), "src={src:?}");
    }
}
// =====================================================================
// イシュー #2174（親 #2131）: optimizer state_dict（save/load・
// safetensors 経由）の facade 公開保留を検査するテスト群。
// `OptimizerStateDictHoldDoctestGuard`（`src/lib.rs`）の正のプローブ
// 1 ブロック方式のドリフト検査に加え、facade src 全体への非再エクス
// ポート・非独自宣言・workspace 全体の `fn state_dict`／
// `fn load_state_dict` 宣言元インベントリを固定する。承認事項の位置
// づけは `docs/autodiff-optimizer-state-dict-decision.md` §5 を参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `OptimizerStateDictHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックが glob import するネスト `pub mod`
/// 集合と、`src/lib.rs` の実際の `pub mod` 宣言集合が一致することを
/// 固定する（`param_groups_hold_doctest_globs_all_pub_modules` の
/// `OptimizerStateDictHoldDoctestGuard` 版）。
#[test]
fn optimizer_state_dict_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "OptimizerStateDictHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "OptimizerStateDictHoldDoctestGuard の doctest ブロックが glob\
         import するモジュール集合が src/lib.rs の pub mod 宣言集合と\
         ドリフトしている（declared={declared:?}, doctest={globbed:?}）。\
         新しい pub mod を追加した場合は doctest 側の use 一覧にも追加\
         すること。"
    );
}

/// [`optimizer_state_dict_hold_doctest_globs_all_pub_modules`] が glob
/// import 集合の一致のみを固定するのに対し、本テストは doctest ブロック
/// の**glob 以外の本文**が固定文言
/// [`OPTIMIZER_STATE_DICT_HOLD_PROBE_BODY`] と 1 行たりとも違わず一致
/// することを固定する（rustdoc の `# ` 隠し行・プローブの削除・別名へ
/// のシャドーイング等で正のプローブを骨抜きにする改変を機械的に拒否
/// する。イシュー #2174）。
#[test]
fn optimizer_state_dict_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "OptimizerStateDictHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, OPTIMIZER_STATE_DICT_HOLD_PROBE_BODY,
        "OptimizerStateDictHoldDoctestGuard の doctest ブロック本文（glob\
         以外）が固定文言 OPTIMIZER_STATE_DICT_HOLD_PROBE_BODY から\
         ドリフトしている。正のプローブ（__fandhe_optim_state_dict_hold_\
         probe モジュール・__FandheOptimizerStateDictProbe トレイト・\
         __probe 関数）の削除・弱体化・隠し行の混入がないか確認する\
         こと。"
    );
}

/// [`optimizer_state_dict_hold_doctest_probe_body_matches_fixed_contract`]
/// が要求する固定文言。`crates/facade/src/lib.rs` の
/// `OptimizerStateDictHoldDoctestGuard` doc 内の唯一の doctest ブロック
/// から、ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）
/// を除いた本文と 1 行単位で完全一致する必要がある（クレートルート
/// 自体の `use fandhe_ai::*;` は本文に含む）。
const OPTIMIZER_STATE_DICT_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_optim_state_dict_hold_probe {\n\
\x20\x20\x20\x20pub trait OptimizerStateDict {}\n\
}\n\
use __fandhe_optim_state_dict_hold_probe::*;\n\
\n\
fn __probe_trait<T: ?Sized + OptimizerStateDict>() {}\n\
\n\
struct __FandheOptimizerStateDictMarker;\n\
\n\
trait __FandheOptimizerStateDictProbe {\n\
\x20\x20\x20\x20fn state_dict(&self) -> __FandheOptimizerStateDictMarker;\n\
\x20\x20\x20\x20fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker;\n\
}\n\
\n\
impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::AdamW {\n\
\x20\x20\x20\x20fn state_dict(&self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::Adam {\n\
\x20\x20\x20\x20fn state_dict(&self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::RmsProp {\n\
\x20\x20\x20\x20fn state_dict(&self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::Adagrad {\n\
\x20\x20\x20\x20fn state_dict(&self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheOptimizerStateDictProbe for fandhe_ai::optim::Lamb {\n\
\x20\x20\x20\x20fn state_dict(&self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load_state_dict(&mut self) -> __FandheOptimizerStateDictMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheOptimizerStateDictMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20adamw: &mut fandhe_ai::optim::AdamW,\n\
\x20\x20\x20\x20adam: &mut fandhe_ai::optim::Adam,\n\
\x20\x20\x20\x20rmsprop: &mut fandhe_ai::optim::RmsProp,\n\
\x20\x20\x20\x20adagrad: &mut fandhe_ai::optim::Adagrad,\n\
\x20\x20\x20\x20lamb: &mut fandhe_ai::optim::Lamb,\n\
) {\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::AdamW::state_dict(adamw);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = adamw.state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::AdamW::load_state_dict(adamw);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = adamw.load_state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adam::state_dict(adam);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = adam.state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adam::load_state_dict(adam);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = adam.load_state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::RmsProp::state_dict(rmsprop);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = rmsprop.state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::RmsProp::load_state_dict(rmsprop);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = rmsprop.load_state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adagrad::state_dict(adagrad);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = adagrad.state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Adagrad::load_state_dict(adagrad);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = adagrad.load_state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Lamb::state_dict(lamb);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = lamb.state_dict();\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = fandhe_ai::optim::Lamb::load_state_dict(lamb);\n\
\x20\x20\x20\x20let _: __FandheOptimizerStateDictMarker = lamb.load_state_dict();\n\
}";

/// facade src 全体（`crates/facade/src/**`）に、`OptimizerStateDict`／
/// `state_dict`（モジュール名としての識別子）を識別子単位で含む
/// `pub use`（別名・ネストした group 経由含む）が存在しないことを
/// 固定する（`OptimizerStateDictHoldDoctestGuard` の正のプローブと
/// 多層防御を成す最内層のソース走査ガード。`facade_does_not_reexport_
/// or_declare_param_groups` と同型）。`fn state_dict`／
/// `fn load_state_dict` の宣言元は
/// [`workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations`]
/// が別途固定する（facade 側は許可集合に含まれないため、そちらが
/// facade への追加も検出する）。
#[test]
fn facade_does_not_reexport_or_declare_optimizer_state_dict() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for line in content.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("pub use") {
                for ident in ["OptimizerStateDict", "state_dict"] {
                    if line_contains_identifier(trimmed, ident) {
                        offending.push(format!(
                            "{}: `{trimmed}` が `{ident}` を識別子単位で含む",
                            path.display()
                        ));
                    }
                }
            }
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が optimizer state_dict（イシュー #2174 の内部\
         クレート限定新規公開面。facade 公開は承認待ちのため対象外と\
         いう設計判断に違反）を再エクスポートしている: {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、`fn state_dict`／
/// `fn load_state_dict` の定義元集合を固定する
/// （`workspace_declares_param_group_fn_names_only_in_allowed_locations`
/// と同型のインベントリ）。
///
/// **期待集合**（着手前確認の再 grep で判明）: 既存 2 か所
/// （`autodiff/src/nn/module.rs`〈`Module` trait 既定実装〉・
/// `facade/src/compat/sequential.rs`〈`Sequential` inherent〉）に加え、
/// 本イシュー（#2174）が追加した [`OptimizerStateDict`] trait 宣言
/// （`autodiff/src/nn/optim/state_dict.rs`）と、9 optimizer ファイル
/// （`adamw`・`adam`・`rmsprop`・`adagrad`・`lamb`・`adadelta`・
/// `adamax`・`nadam`・`radam`）各 1 件ずつの impl。加えて #2366 が追加した
/// `Lbfgs` 専用 inherent API（`autodiff/src/nn/optim/lbfgs.rs`）各 1 件。
/// #2367 が追加した `Sgd` の impl（`autodiff/src/optim/sgd.rs`）各 1 件。
#[test]
fn workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations() {
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

    const NAMES: [&str; 2] = ["state_dict", "load_state_dict"];

    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            for fn_name in NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("autodiff/src/nn/module.rs::state_dict", 1usize),
        ("autodiff/src/nn/module.rs::load_state_dict", 1usize),
        ("facade/src/compat/sequential.rs::state_dict", 1usize),
        ("facade/src/compat/sequential.rs::load_state_dict", 1usize),
        // #2395: facade 独自 `nn::Module` の defaulted メソッド（承認済み）。
        ("facade/src/nn/module.rs::state_dict", 1usize),
        ("facade/src/nn/module.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/state_dict.rs::state_dict", 1usize),
        (
            "autodiff/src/nn/optim/state_dict.rs::load_state_dict",
            1usize,
        ),
        ("autodiff/src/nn/optim/adamw.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/adamw.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/adam.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/adam.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/rmsprop.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/rmsprop.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/adagrad.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/adagrad.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/lamb.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/lamb.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/adadelta.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/adadelta.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/adamax.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/adamax.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/nadam.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/nadam.rs::load_state_dict", 1usize),
        ("autodiff/src/nn/optim/radam.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/radam.rs::load_state_dict", 1usize),
        // イシュー #2366: `Lbfgs` 専用 inherent API（トレイト非実装）。
        // `Lbfgs` は #2502 で facade 公開済みのため `fandhe_ai::optim::Lbfgs`
        // から到達可能。`OptimizerStateDict` トレイトの facade 公開は #2555
        // の範囲（定義元は増えないため期待件数は不変）。
        ("autodiff/src/nn/optim/lbfgs.rs::state_dict", 1usize),
        ("autodiff/src/nn/optim/lbfgs.rs::load_state_dict", 1usize),
        // イシュー #2367: `Sgd`（`crate::optim`。momentum の velocity のみ）。
        ("autodiff/src/optim/sgd.rs::state_dict", 1usize),
        ("autodiff/src/optim/sgd.rs::load_state_dict", 1usize),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の state_dict／load_state_dict\
         系 `fn` 宣言集合が期待と一致しない（過不足いずれも fail-closed\
         に検出する。facade への新規宣言が紛れ込んでいないか、9\
         optimizer 以外への実装漏れ・過剰実装がないかを含めて確認\
         すること）: {found:?}"
    );
}

// =====================================================================
// イシュー #2198（親 #2172・ルート #2131）・#2502（親 #2500・ルート #2499）:
// LBFGS の facade 公開・`compile()` 統合を検査するテスト群。#2198（2026-09-27
// 所有者承認）で `compat::Optimizer::Lbfgs(LbfgsConfig)` variant・
// `LbfgsConfig` の再エクスポート・`compile()`/`fit()` 統合を、#2502
// （ルート #2499 本文「承認範囲」節の一括承認。`docs/autodiff-lbfgs-decision.md` §8）で
// `Lbfgs`・`LbfgsLineSearch` の再エクスポートを実装済み。旧否定ガード
// （`LbfgsHoldDoctestGuard` と facade src 走査 6 項目）は #2502 で撤去し、
// 承認した形だけを許す正ガード（`optim_module_reexports_exactly_expected_surface`
// の期待集合・`lbfgs_types_are_reachable_via_facade_only`）へ反転した。
// =====================================================================

/// `Lbfgs`／`LbfgsConfig`／`LbfgsLineSearch` が facade だけの import で到達でき、
/// strong Wolfe 指定の手動 closure ループが動くことの固定（`fandhe_ai_autodiff`
/// は import しない。`nn_module_types_are_reachable_via_facade_only` と同型）。
#[test]
fn lbfgs_types_are_reachable_via_facade_only() {
    use fandhe_ai::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
    use fandhe_ai::{AutodiffError, Tensor};

    // 2 variant とも名指しできること（`#[non_exhaustive]` のため `_` 腕が必要）。
    for ls in [LbfgsLineSearch::None, LbfgsLineSearch::StrongWolfe] {
        match ls {
            LbfgsLineSearch::None | LbfgsLineSearch::StrongWolfe => {}
            _ => {}
        }
    }

    let target = [1.0_f32, -2.0, 3.0];
    let quad = |p: &[Tensor<f32>]| -> (f32, Vec<Tensor<f32>>) {
        let x = p[0].contiguous();
        let x = x.as_slice().expect("test fixture: contiguous");
        let loss: f32 = x.iter().zip(target).map(|(a, c)| (a - c) * (a - c)).sum();
        let g: Vec<f32> = x.iter().zip(target).map(|(a, c)| 2.0 * (a - c)).collect();
        (
            loss,
            vec![Tensor::new(g, &[3]).expect("test fixture: shape")],
        )
    };

    let cfg = LbfgsConfig {
        line_search: LbfgsLineSearch::StrongWolfe,
        ..LbfgsConfig::default()
    };
    let mut opt = Lbfgs::new(cfg).expect("Lbfgs::new");
    let params = vec![Tensor::new(vec![0.0_f32; 3], &[3]).expect("test fixture: shape")];
    let (initial, _) = quad(&params);
    let updated = opt.step_closure(&params, quad).expect("step_closure");
    let (after, _) = quad(&updated);
    assert!(
        after.is_finite() && after < initial,
        "L-BFGS(strong Wolfe) の損失が減少していない: {initial} -> {after}"
    );

    // 可失敗 closure 版の戻り値エラー型が facade の `AutodiffError` に固定されること。
    let r: Result<Vec<Tensor<f32>>, AutodiffError> =
        opt.try_step_closure(&updated, |p| Ok(quad(p)));
    assert!(r.is_ok());
}

/// facade src 全体（`crates/facade/src/**`）を走査し、`enum Optimizer`
/// 定義（`compat::Optimizer`。`OptimizerState` 等の同名接頭辞を持つ別
/// enum とは区別する）の直接の variant 名の中に `Lbfgs`（大文字小文字を
/// 無視した表記揺れ含む。`LBFGS`／`LBfgs` 等）が存在するかを検出する。
/// `enum` `Optimizer` `{` の完全一致でトークン列を探索し、対応する `}`
/// までの間で中括弧の深さを追跡する。variant 名は深さ 1 に入った直後
/// （開き `{` の直後）と、深さ 1 での `,` の直後にのみ現れる識別子と
/// してのみ収集するため、variant の payload 型（`Lbfgs(LbfgsConfig)` の
/// `LbfgsConfig` 等）は対象に含まれない。doc コメント（`strip_comments_
/// and_literals` で事前に除去済み）に加え、variant 直前の属性
/// （`#[deprecated]` 等。`#` トークンから対応する `]` までを読み飛ばす。
/// 属性の読み飛ばし中は variant 開始位置の判定を維持したままにするため、
/// 属性付き variant（`#[deprecated]\nLbfgs(LbfgsConfig)` 等）も variant
/// 名として正しく判定できる。イシュー #2198 の Codex レビュー指摘: 旧実装は
/// `,` 直後の最初のトークンを無条件で variant 名扱いしていたため、属性の
/// 先頭 `#` を variant 名候補として消費してしまい、続く実際の variant 名
/// （`Lbfgs`）を検出できなかった）も読み飛ばして対象に含めない。
/// `enum Optimizer` が 1 件も見つからない場合は検査対象を見失ったことと
/// して扱い、呼び出し元が fail-closed で panic する（戻り値 `None`）。
/// 2026-09-27 承認後は本走査を [`compat_optimizer_enum_has_lbfgs_variant`]
/// が**正**のガード（variant がちょうど 1 個存在すること）へ転用する。
fn scan_optimizer_enum_variants_for_lbfgs(content: &str) -> Option<Vec<String>> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut found_enum = false;
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "enum"
            && tokens.get(i + 1).map(String::as_str) == Some("Optimizer")
            && tokens.get(i + 2).map(String::as_str) == Some("{")
        {
            found_enum = true;
            let mut depth: i32 = 1;
            let mut j = i + 3;
            // 開き `{` の直後は新しい variant の開始位置（`,` 直後と同じ
            // 扱い）。
            let mut at_variant_start = true;
            while j < tokens.len() && depth > 0 {
                match tokens[j].as_str() {
                    "{" => {
                        depth += 1;
                        at_variant_start = false;
                    }
                    "}" => {
                        depth -= 1;
                        at_variant_start = false;
                    }
                    "," if depth == 1 => {
                        at_variant_start = true;
                    }
                    "#" if depth == 1 && at_variant_start => {
                        // variant 直前の属性（`#[...]`）を読み飛ばす。
                        // variant 名判定はまだ始まっていないため
                        // at_variant_start は true のまま維持し、属性の
                        // 次に続く実際の variant 名を取りこぼさない。
                        j += 1;
                        if tokens.get(j).map(String::as_str) == Some("[") {
                            let mut bracket_depth: i32 = 1;
                            j += 1;
                            while j < tokens.len() && bracket_depth > 0 {
                                match tokens[j].as_str() {
                                    "[" => bracket_depth += 1,
                                    "]" => bracket_depth -= 1,
                                    _ => {}
                                }
                                j += 1;
                            }
                        }
                        continue;
                    }
                    tok if depth == 1 && at_variant_start => {
                        if tok.eq_ignore_ascii_case("lbfgs") {
                            offending.push(tok.to_string());
                        }
                        at_variant_start = false;
                    }
                    _ => {}
                }
                j += 1;
            }
            i = j;
            continue;
        }
        i += 1;
    }

    if found_enum { Some(offending) } else { None }
}

/// `crates/facade/src/**` 全体を走査し、`compat::Optimizer` enum の
/// variant に `Lbfgs`（表記揺れなし・ちょうど 1 個）が存在することを
/// 固定する（2026-09-27 承認〈#2172 コメント〉により
/// `compat_optimizer_enum_has_no_lbfgs_variant` から反転した正のガード。
/// 旧 `LbfgsHoldDoctestGuard`〈#2502 で撤去〉の variant プローブ削除と対を
/// 成していた）。`enum Optimizer` 定義がワークスペース全体で 1 件も見つからない
/// 場合は、検査対象自体を見失ったものとして fail-closed に失敗する。
#[test]
fn compat_optimizer_enum_has_lbfgs_variant() {
    let src_dir = facade_crate_root().join("src");
    let mut found_enum = false;
    let mut hits: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        if let Some(h) = scan_optimizer_enum_variants_for_lbfgs(content) {
            found_enum = true;
            for hit in h {
                hits.push(format!("{}: variant {hit}", path.display()));
            }
        }
    });
    assert!(
        found_enum,
        "crates/facade/src/** から `enum Optimizer` 定義を 1 件も抽出\
         できなかった（テスト自体が検査対象を見失っている可能性がある。\
         `compat::Optimizer` のファイル移動・改名を確認すること）"
    );
    assert_eq!(
        hits.len(),
        1,
        "compat::Optimizer enum に Lbfgs variant がちょうど 1 個含まれる\
         はず（2026-09-27 承認〈#2172 コメント〉。過不足いずれも\
         fail-closed に検出する）: {hits:?}"
    );
    assert!(
        hits[0].ends_with("variant Lbfgs"),
        "compat::Optimizer enum の Lbfgs variant の表記が正規形（大文字・\
         小文字含め `Lbfgs`）と一致しない: {hits:?}"
    );
}

/// [`scan_optimizer_enum_variants_for_lbfgs`]（[`compat_optimizer_enum_
/// has_lbfgs_variant`]）の自己テスト（正例・負例の合成入力）。
#[test]
fn scan_optimizer_enum_variants_for_lbfgs_detects_each_category() {
    // 正例: 単純な tuple variant 追加。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "pub enum Optimizer { Sgd(SgdConfig), Lbfgs(LbfgsConfig) }"
        ),
        Some(vec!["Lbfgs".to_string()])
    );
    // 正例: doc コメント・属性付きの variant（コメントは
    // strip_comments_and_literals で除去済み想定の入力）。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "#[non_exhaustive]\npub enum Optimizer {\n    Sgd(SgdConfig),\n    /// doc\n    Lbfgs(LbfgsConfig),\n}"
        ),
        Some(vec!["Lbfgs".to_string()])
    );
    // 正例: 表記揺れ（大文字）。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs("pub enum Optimizer { LBFGS(LbfgsConfig) }"),
        Some(vec!["LBFGS".to_string()])
    );
    // 負例: 現行の variant 集合のみ（Lbfgs なし）。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "pub enum Optimizer { Sgd(SgdConfig), AdamW(AdamWConfig) }"
        ),
        Some(vec![])
    );
    // 負例: 別 enum（OptimizerState）への同名 variant は対象外。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs("enum OptimizerState { Lbfgs(Lbfgs) }"),
        None
    );
    // 負例: payload 型名としての出現は variant 名ではないため対象外。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "pub enum Optimizer { Sgd(LbfgsPayloadNotAVariant) }"
        ),
        Some(vec![])
    );
    // 負例: `enum Optimizer` 自体が存在しない。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs("pub struct Unrelated;"),
        None
    );
    // 正例: variant 自体に属性が付いている場合（イシュー #2198 Codex
    // レビュー指摘の回帰防止）。属性の先頭 `#` を variant 名として誤検出
    // せず、属性を読み飛ばした先の `Lbfgs` を正しく検出する。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "pub enum Optimizer { Sgd(SgdConfig), #[deprecated] Lbfgs(LbfgsConfig) }"
        ),
        Some(vec!["Lbfgs".to_string()])
    );
    // 正例: 属性付き variant が先頭（開き `{` の直後）にある場合。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "pub enum Optimizer { #[deprecated] Lbfgs(LbfgsConfig), Sgd(SgdConfig) }"
        ),
        Some(vec!["Lbfgs".to_string()])
    );
    // 負例: 属性付き variant だが Lbfgs ではない場合は誤検出しない。
    assert_eq!(
        scan_optimizer_enum_variants_for_lbfgs(
            "pub enum Optimizer { #[deprecated] Sgd(SgdConfig) }"
        ),
        Some(vec![])
    );
}

// =====================================================================
// イシュー #2179（親 #2131「PyTorch／TF 置き換えの API 網羅」）: EMA
// （`ExponentialMovingAverage`）の facade 公開・`FitConfig`／
// `Sequential` 接続の保留を検査するテスト群。`EmaHoldDoctestGuard`
// （`src/lib.rs`）の正のプローブ 1 ブロック方式のドリフト検査に加え、
// facade src 全体への非再エクスポート・非独自宣言（型名）・非
// inherent メソッド追加（`use_ema`／`ema_decay`）を固定する。承認事項
// の位置づけは `docs/autodiff-ema-decision.md` §4 を参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `EmaHoldDoctestGuard` doc 内の唯一の
/// doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （`optimizer_ext_hold_doctest_globs_all_pub_modules` の
/// `EmaHoldDoctestGuard` 版）。
#[test]
fn ema_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "EmaHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "EmaHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトして\
         いる（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`ema_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの**glob
/// 以外の本文**（`__fandhe_ema_hold_probe` モジュール・`__probe_type`
/// 関数・`__FandheEmaHoldProbe` トレイト・`FitConfig`／`Sequential`
/// への実装・`__probe_fit_config`／`__probe_sequential` 関数）が固定
/// 文言 [`EMA_HOLD_PROBE_BODY`] と 1 行たりとも違わず一致することを
/// 固定する（rustdoc の `# ` 隠し行・プローブの削除・別名への
/// シャドーイング等で正のプローブを骨抜きにする改変を機械的に拒否
/// する）。
#[test]
fn ema_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "EmaHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, EMA_HOLD_PROBE_BODY,
        "EmaHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 EMA_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（型名 glob 衝突・inherent メソッド衝突の両方）の\
         削除・弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`ema_hold_doctest_probe_body_matches_fixed_contract`] が要求する
/// 固定文言。`crates/facade/src/lib.rs` の `EmaHoldDoctestGuard` doc
/// 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import 行
/// （`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致する
/// 必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に含む）。
const EMA_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_ema_hold_probe {\n\
\x20\x20\x20\x20pub struct ExponentialMovingAverage;\n\
}\n\
use __fandhe_ema_hold_probe::*;\n\
\n\
fn __probe_type(_: ExponentialMovingAverage) {}\n\
\n\
struct __FandheEmaHoldMarker;\n\
\n\
trait __FandheEmaHoldProbe {\n\
\x20\x20\x20\x20fn use_ema(&self) -> __FandheEmaHoldMarker;\n\
\x20\x20\x20\x20fn ema_decay(&self) -> __FandheEmaHoldMarker;\n\
}\n\
\n\
impl __FandheEmaHoldProbe for fandhe_ai::compat::FitConfig {\n\
\x20\x20\x20\x20fn use_ema(&self) -> __FandheEmaHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheEmaHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn ema_decay(&self) -> __FandheEmaHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheEmaHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheEmaHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn use_ema(&self) -> __FandheEmaHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheEmaHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn ema_decay(&self) -> __FandheEmaHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheEmaHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_fit_config(x: &fandhe_ai::compat::FitConfig) {\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = fandhe_ai::compat::FitConfig::use_ema(x);\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = x.use_ema();\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = fandhe_ai::compat::FitConfig::ema_decay(x);\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = x.ema_decay();\n\
}\n\
\n\
fn __probe_sequential(x: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = fandhe_ai::compat::Sequential::use_ema(x);\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = x.use_ema();\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = fandhe_ai::compat::Sequential::ema_decay(x);\n\
\x20\x20\x20\x20let _: __FandheEmaHoldMarker = x.ema_decay();\n\
}";

/// [`facade_does_not_reexport_or_declare_ema_items`]・その自己テストが
/// 共用する検出本体。facade src 全体（`crates/facade/src/**`）の
/// `pub use` から [`collect_pub_use_leaves`] で別名にする前の葉を集め
/// `ExponentialMovingAverage` を検出し（単一行・複数行・ネストした
/// group・別名も検出）、`trait`／`struct`／`enum`／`type` 直後の同名
/// 独自宣言を違反として返す（`scan_optimizer_ext_reexports_and_declarations`
/// と同型）。加えて `EmaHoldDoctestGuard` のプローブ 2「inherent
/// メソッド追加」に対応するソース走査として、`fn use_ema`／
/// `fn ema_decay` 宣言（可視性・宣言文脈を問わない）も違反として返す
/// （`facade_source_declares_no_custom_fn_in_any_context` と同型の
/// 「定義元そのものを許さない」多層防御）。
fn scan_ema_reexports_and_declarations(content: &str) -> Vec<String> {
    const TYPE_NAMES: [&str; 1] = ["ExponentialMovingAverage"];
    const METHOD_NAMES: [&str; 2] = ["use_ema", "ema_decay"];
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if TYPE_NAMES.contains(&leaf.as_str()) {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens
                .get(i + 1)
                .map(|t| TYPE_NAMES.contains(&t.as_str()))
                .unwrap_or(false)
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        if tokens[i] == "fn"
            && tokens
                .get(i + 1)
                .map(|t| METHOD_NAMES.contains(&t.as_str()))
                .unwrap_or(false)
        {
            offending.push(format!(
                "fn {} 宣言",
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`ExponentialMovingAverage`
/// を識別子単位で含む `pub use`（複数行・ネストした group・別名含む）
/// も、facade 独自の `trait`／`struct`／`enum`／`type` 宣言も、
/// `use_ema`／`ema_decay` という名前の `fn` 宣言（可視性・宣言文脈を
/// 問わない）も存在しないことを固定する（`EmaHoldDoctestGuard` の
/// 正のプローブと多層防御を成す最内層のソース走査ガード）。
#[test]
fn facade_does_not_reexport_or_declare_ema_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_ema_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が EMA（イシュー #2179。ExponentialMovingAverage／\
         use_ema／ema_decay）を再エクスポート、または独自宣言している\
         （`docs/autodiff-ema-decision.md` §4 承認事項が未取得のまま\
         対象外としている設計判断に違反）: {offending:?}"
    );
}

/// [`scan_ema_reexports_and_declarations`]（[`facade_does_not_reexport_
/// or_declare_ema_items`]）の自己テスト（正例・負例の合成入力）。
#[test]
fn facade_does_not_reexport_or_declare_ema_items_detects_each_category() {
    // 正例: 単一行 pub use。
    assert!(
        !scan_ema_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::nn::ExponentialMovingAverage;"
        )
        .is_empty()
    );
    // 正例: 複数行 pub use（group）。
    assert!(
        !scan_ema_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::nn::{\n    ExponentialMovingAverage,\n    Linear,\n};"
        )
        .is_empty()
    );
    // 正例: 別名 pub use。
    assert!(
        !scan_ema_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::nn::ExponentialMovingAverage as Ema;"
        )
        .is_empty()
    );
    // 正例: 独自 struct 宣言。
    assert!(
        !scan_ema_reexports_and_declarations("pub struct ExponentialMovingAverage;").is_empty()
    );
    // 正例: inherent メソッド追加（fn use_ema）。
    assert!(
        !scan_ema_reexports_and_declarations(
            "impl FitConfig {\n    pub fn use_ema(mut self, on: bool) -> Self {\n        self\n    }\n}"
        )
        .is_empty()
    );
    // 正例: inherent メソッド追加（fn ema_decay）。
    assert!(
        !scan_ema_reexports_and_declarations(
            "impl FitConfig {\n    pub fn ema_decay(mut self, decay: f32) -> Self {\n        self\n    }\n}"
        )
        .is_empty()
    );
    // 負例: コメント中の出現。
    assert!(
        scan_ema_reexports_and_declarations("// pub use ...::ExponentialMovingAverage;").is_empty()
    );
    // 負例: 文字列リテラル中の出現。
    assert!(
        scan_ema_reexports_and_declarations("let s = \"ExponentialMovingAverage\";").is_empty()
    );
    // 負例: 非公開 use。
    assert!(
        scan_ema_reexports_and_declarations(
            "use fandhe_ai_autodiff::nn::ExponentialMovingAverage;"
        )
        .is_empty()
    );
    // 負例: 無関係な pub use。
    assert!(
        scan_ema_reexports_and_declarations("pub use fandhe_ai_autodiff::nn::AdamW;").is_empty()
    );
    // 負例: 無関係な fn 宣言。
    assert!(scan_ema_reexports_and_declarations("pub fn use_dropout(&self) {}").is_empty());
}

// =====================================================================
// LR スケジューラ拡張 5 種（イシュー #2176 実装・#2503 公開。親 #2499）の
// 正ガード。旧 `LrSchedulerExtHoldDoctestGuard` 系の否定ガード 4 テストを
// 「承認した形だけを許す」形へ反転した（決定記録
// `docs/autodiff-lr-scheduler-ext-decision.md` §8 実装記録。先例: #2198
// `compat_optimizer_enum_has_lbfgs_variant`）。
// =====================================================================

/// 承認形で公開する 5 名。
const LR_SCHEDULER_EXT_NAMES: [&str; 5] = [
    "MultiStepLr",
    "CosineAnnealingWarmRestarts",
    "CyclicLr",
    "LambdaLr",
    "SequentialLr",
];

/// `pub use` の use tree を宣言単位（完全なパス + `as` 別名）の
/// エントリへ展開する（[`scan_lr_scheduler_ext_items`] の下請け）。
/// `tokens[i..]` の 1 ノード（単一パス or `{ ... }` グループ）を `prefix`
/// 付きで解析し、`(完全パスのセグメント列, 別名)` を `out` へ積んで
/// 消費後の index を返す。`*`（glob）は個別の識別子を持たないため積まない。
fn lr_ext_parse_use_tree(
    tokens: &[String],
    mut i: usize,
    prefix: &[String],
    out: &mut Vec<(Vec<String>, Option<String>)>,
) -> usize {
    let mut path: Vec<String> = prefix.to_vec();
    if tokens.get(i).map(String::as_str) == Some(":")
        && tokens.get(i + 1).map(String::as_str) == Some(":")
    {
        i += 2;
    }
    loop {
        match tokens.get(i).map(String::as_str) {
            Some("{") => {
                i += 1;
                loop {
                    match tokens.get(i).map(String::as_str) {
                        Some("}") => {
                            i += 1;
                            break;
                        }
                        Some(",") => i += 1,
                        None => break,
                        _ => i = lr_ext_parse_use_tree(tokens, i, &path, out),
                    }
                }
                return i;
            }
            Some("*") => return i + 1,
            Some(seg) if seg != "as" && seg != "," && seg != "}" => {
                path.push(seg.to_string());
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
    let mut alias = None;
    if tokens.get(i).map(String::as_str) == Some("as") {
        alias = tokens.get(i + 1).cloned();
        i += 2;
    }
    // `a::b::{self}` のように `self` で終わる場合は直前セグメントを葉とみなす。
    if path.last().map(String::as_str) == Some("self") && path.len() > 1 {
        path.pop();
    }
    if path.len() > prefix.len() {
        out.push((path, alias));
    }
    i
}

/// 正ガード・自己テストが共用する検出本体。`content` の `pub use` を
/// **宣言単位**（完全パス + 別名）に展開し、葉または公開される別名が
/// 5 名のいずれかであるエントリと、`trait`／`struct`／`enum`／`type` の
/// 同名独自宣言を `(種別, 名前)` で返す。種別 `"pub use"` は完全パスが
/// ちょうど `fandhe_ai_autodiff::nn::optim::<名>` で別名なしの承認形のみ。
/// それ以外（別経路・別名公開・承認名への別名付け替え・他型を承認名で
/// 公開）は `"pub use(非承認経路)"`／`"alias"` として返す。コメント・
/// 文字列リテラルは除去済みの走査対象のみを見る。
fn scan_lr_scheduler_ext_items(content: &str) -> Vec<(String, String)> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut found: Vec<(String, String)> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let mut entries: Vec<(Vec<String>, Option<String>)> = Vec::new();
            let mut j = 0usize;
            while j < path_tokens.len() {
                if path_tokens[j] == "," {
                    j += 1;
                } else {
                    j = lr_ext_parse_use_tree(path_tokens, j, &[], &mut entries);
                }
            }
            for (segs, alias) in entries {
                let leaf = segs.last().cloned().unwrap_or_default();
                let leaf_hit = LR_SCHEDULER_EXT_NAMES.contains(&leaf.as_str());
                let alias_hit = alias
                    .as_deref()
                    .is_some_and(|a| LR_SCHEDULER_EXT_NAMES.contains(&a));
                if !leaf_hit && !alias_hit {
                    continue;
                }
                let approved_path = segs.len() == 4
                    && segs[0] == "fandhe_ai_autodiff"
                    && segs[1] == "nn"
                    && segs[2] == "optim";
                if leaf_hit && alias.is_none() && approved_path {
                    found.push(("pub use".to_string(), leaf));
                } else if let Some(a) = alias.filter(|_| alias_hit) {
                    found.push(("alias".to_string(), a));
                } else {
                    found.push((format!("pub use(非承認経路 {})", segs.join("::")), leaf));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens
                .get(i + 1)
                .is_some_and(|t| LR_SCHEDULER_EXT_NAMES.contains(&t.as_str()))
        {
            found.push((tokens[i].clone(), tokens[i + 1].clone()));
        }
        i += 1;
    }
    found
}

/// 承認形（`optim.rs` の `pub use fandhe_ai_autodiff::nn::optim::…` の
/// 葉としてちょうど 1 回ずつ。宣言単位で完全パスを検証済み）以外の
/// 出現を違反として返す。`optim.rs` 以外での再エクスポート・別名・
/// 独自宣言、承認形の欠落・重複は fail-closed。
fn lr_scheduler_ext_violations(files: &[(String, String)]) -> Vec<String> {
    let mut violations = Vec::new();
    let mut approved: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (rel, content) in files {
        let is_optim = rel == "optim.rs";
        for (kind, name) in scan_lr_scheduler_ext_items(content) {
            if is_optim
                && kind == "pub use"
                && let Some(n) = LR_SCHEDULER_EXT_NAMES.iter().find(|n| **n == name)
            {
                *approved.entry(n).or_insert(0) += 1;
                continue;
            }
            violations.push(format!("{rel}: {kind} {name}"));
        }
    }
    for n in LR_SCHEDULER_EXT_NAMES {
        let c = approved.get(n).copied().unwrap_or(0);
        if c != 1 {
            violations.push(format!(
                "optim.rs: {n} の承認形再エクスポートが {c} 回（期待 1 回）"
            ));
        }
    }
    violations
}

/// facade src 全体で、LR スケジューラ拡張 5 名が `src/optim.rs` の
/// `pub use fandhe_ai_autodiff::nn::optim::…` としてちょうど 1 回ずつ
/// 公開され、他ファイルでの再エクスポート・別名・独自宣言が無いことを
/// 固定する（検出 0 件なら検査対象を見失ったとして失敗する）。
#[test]
fn facade_reexports_lr_scheduler_ext_items_only_in_approved_shape() {
    let src_dir = facade_crate_root().join("src");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&src_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    let violations = lr_scheduler_ext_violations(&files);
    assert!(
        violations.is_empty(),
        "LR スケジューラ拡張 5 種の facade 公開が承認形\
         （`docs/autodiff-lr-scheduler-ext-decision.md` §8）から外れている: {violations:?}"
    );
}

/// [`lr_scheduler_ext_violations`] の自己テスト（正例・負例の合成入力）。
/// `SequentialLr` と既存の `compat::Sequential` は識別子単位で衝突しない。
#[test]
fn facade_reexports_lr_scheduler_ext_items_only_in_approved_shape_detects_each_category() {
    let approved_optim = "pub use fandhe_ai_autodiff::nn::optim::{CosineAnnealingWarmRestarts, CyclicLr};\n\
pub use fandhe_ai_autodiff::nn::optim::{LambdaLr, MultiStepLr, SequentialLr};\n";
    let with_lib = |extra: &str| {
        vec![
            ("optim.rs".to_string(), approved_optim.to_string()),
            ("lib.rs".to_string(), extra.to_string()),
        ]
    };
    // 正例（承認形のみ・無関係な出現は誤検出しない）。
    assert!(lr_scheduler_ext_violations(&with_lib("")).is_empty());
    assert!(
        lr_scheduler_ext_violations(&with_lib("pub use crate::compat::Sequential;")).is_empty()
    );
    assert!(lr_scheduler_ext_violations(&with_lib("// pub use x::MultiStepLr;")).is_empty());
    assert!(lr_scheduler_ext_violations(&with_lib("let s = \"CyclicLr\";")).is_empty());
    // 負例: 別ファイルでの再エクスポート。
    assert!(
        !lr_scheduler_ext_violations(&with_lib(
            "pub use fandhe_ai_autodiff::nn::optim::MultiStepLr;"
        ))
        .is_empty()
    );
    // 負例: 別名。
    let aliased = vec![(
        "optim.rs".to_string(),
        format!("{approved_optim}pub use fandhe_ai_autodiff::nn::optim::CyclicLr as Foo;\n"),
    )];
    assert!(!lr_scheduler_ext_violations(&aliased).is_empty());
    // 負例: 別ファイルでの別名公開（元の葉が承認名でなくても検出する）。
    assert!(
        !lr_scheduler_ext_violations(&with_lib("pub use crate::other::Other as MultiStepLr;"))
            .is_empty()
    );
    // 負例: optim.rs 内でも承認名を別経路・他型から公開する（正しい行が
    // 他に残っていても承認形として数えない）。
    let wrong_path = vec![(
        "optim.rs".to_string(),
        format!("{approved_optim}pub use crate::other::CyclicLr;\n"),
    )];
    assert!(!lr_scheduler_ext_violations(&wrong_path).is_empty());
    let wrong_alias = vec![(
        "optim.rs".to_string(),
        format!("{approved_optim}pub use crate::other::Other as LambdaLr;\n"),
    )];
    assert!(!lr_scheduler_ext_violations(&wrong_alias).is_empty());
    // 負例: 承認形の 1 つを別経路に差し替えると承認形の欠落でも失敗する。
    let swapped = vec![(
        "optim.rs".to_string(),
        "pub use fandhe_ai_autodiff::nn::optim::{CosineAnnealingWarmRestarts, CyclicLr};\n\
pub use fandhe_ai_autodiff::nn::optim::{LambdaLr, SequentialLr};\n\
pub use crate::other::MultiStepLr;\n"
            .to_string(),
    )];
    assert!(!lr_scheduler_ext_violations(&swapped).is_empty());
    // 負例: 独自 struct 宣言。
    assert!(!lr_scheduler_ext_violations(&with_lib("pub struct LambdaLr;")).is_empty());
    // 負例: 承認形の欠落（検査対象を見失った場合も fail-closed）。
    assert!(!lr_scheduler_ext_violations(&[("optim.rs".to_string(), String::new())]).is_empty());
    // 負例: 重複公開。
    let dup = vec![(
        "optim.rs".to_string(),
        format!("{approved_optim}pub use fandhe_ai_autodiff::nn::optim::SequentialLr;\n"),
    )];
    assert!(!lr_scheduler_ext_violations(&dup).is_empty());
}

// =====================================================================
// イシュー #2169（親 #2131）→ #2509: `compat::Loss` enum の variant 集合を
// 承認形 9 種へ固定する正ガード（#2509 で否定ガードから反転。保留 doctest
// `CompileLossVariantsHoldDoctestGuard` は撤去済み）。決定記録は `docs/
// facade-compile-loss-variants-decision.md` §4・§5 を参照。
// =====================================================================

/// `crates/facade/src/compat/training.rs` の `Loss` enum（`compat::Loss`）の
/// variant 集合が、承認形（`Mse`・`CrossEntropy`・`L1`・`Bce`・`BceWithLogits`・
/// `Nll`・`KlDiv`・`Huber`・`SmoothL1` の 9 種を本順で）と完全一致することを
/// 固定する正ガード。2026-10-04 承認（ルート #2499）により #2509 で、従来の
/// 「2 variant のみ」否定ガード（イシュー #2169）から反転した。承認外の
/// variant 追加・削除・並べ替えを拒否する（`docs/
/// facade-compile-loss-variants-decision.md` §2・§4・§5）。
#[test]
fn compat_loss_enum_variants_are_exactly_approved_set() {
    let path = facade_crate_root().join("src/compat/training.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);

    // "pub" "enum" "Loss" "{" の完全一致列を数える（0 件・2 件以上は
    // fail-closed で失敗させる: enum の削除・複数定義・別名への変更を
    // 見逃さない）。
    let mut match_starts: Vec<usize> = Vec::new();
    for i in 0..tokens.len() {
        if tokens.get(i).map(String::as_str) == Some("pub")
            && tokens.get(i + 1).map(String::as_str) == Some("enum")
            && tokens.get(i + 2).map(String::as_str) == Some("Loss")
            && tokens.get(i + 3).map(String::as_str) == Some("{")
        {
            match_starts.push(i + 4);
        }
    }
    assert_eq!(
        match_starts.len(),
        1,
        "training.rs 内の `pub enum Loss {{` 宣言がちょうど 1 件ではない\
         （0 件: enum が削除・改名された。2 件以上: 重複定義。いずれも\
         本テストが検査対象を見失っている）: {} 件",
        match_starts.len()
    );

    let variants = collect_top_level_enum_variant_idents(&tokens, match_starts[0]);
    assert_eq!(
        variants,
        [
            "Mse",
            "CrossEntropy",
            "L1",
            "Bce",
            "BceWithLogits",
            "Nll",
            "KlDiv",
            "Huber",
            "SmoothL1",
        ]
        .map(String::from)
        .to_vec(),
        "compat::Loss の variant 集合が承認形 9 種からドリフトしている\
         （承認外の追加・削除・並べ替えの可能性。\
         `docs/facade-compile-loss-variants-decision.md` §2・§4 参照）: {variants:?}"
    );
}

/// [`compat_loss_enum_variants_are_exactly_approved_set`] が
/// 使う共通トークン走査。`tokens[start..]`（`pub enum <Name> {` の `{`
/// 直後）から、対応する閉じ `}` までを brace 深さで追跡し、深さ 1 の
/// 識別子トークンを variant 名として収集する（`#[...]` 属性・カンマは
/// 読み飛ばす）。`Loss` は unit variant のみのため、tuple／struct
/// variant の内部（`(...)`／`{...}`）は本関数の対象外——ネストした
/// `{` は brace 深さのみを増減させ、内部の識別子は収集しない。
fn collect_top_level_enum_variant_idents(tokens: &[String], start: usize) -> Vec<String> {
    let mut depth = 1usize;
    let mut i = start;
    let mut variants = Vec::new();
    let mut expect_variant_start = true;
    while i < tokens.len() && depth > 0 {
        match tokens[i].as_str() {
            "{" => {
                depth += 1;
                i += 1;
                continue;
            }
            "}" => {
                depth -= 1;
                i += 1;
                continue;
            }
            "#" => {
                i += 1;
                if tokens.get(i).map(String::as_str) == Some("[") {
                    let mut bracket_depth = 1usize;
                    i += 1;
                    while i < tokens.len() && bracket_depth > 0 {
                        match tokens[i].as_str() {
                            "[" => bracket_depth += 1,
                            "]" => bracket_depth -= 1,
                            _ => {}
                        }
                        i += 1;
                    }
                }
                continue;
            }
            "," => {
                expect_variant_start = true;
                i += 1;
                continue;
            }
            token => {
                if depth == 1 && expect_variant_start {
                    let is_ident = token
                        .chars()
                        .next()
                        .map(|c| c.is_ascii_alphabetic() || c == '_')
                        .unwrap_or(false);
                    if is_ident {
                        variants.push(token.to_string());
                        expect_variant_start = false;
                    }
                }
                i += 1;
            }
        }
    }
    variants
}

/// [`collect_top_level_enum_variant_idents`] の自己テスト（正例・負例
/// の合成入力）。
#[test]
fn collect_top_level_enum_variant_idents_detects_each_category() {
    let tokens = tokenize_including_punctuation("pub enum Loss { Mse , CrossEntropy , }");
    // "pub" "enum" "Loss" "{" の直後（index 4）から走査する。
    assert_eq!(
        collect_top_level_enum_variant_idents(&tokens, 4),
        vec!["Mse".to_string(), "CrossEntropy".to_string()]
    );
    // 属性付き variant（`#[non_exhaustive]` 相当のダミー属性）を読み
    // 飛ばすこと。
    let tokens_with_attr =
        tokenize_including_punctuation("pub enum Loss { # [ foo ] Mse , CrossEntropy , }");
    assert_eq!(
        collect_top_level_enum_variant_idents(&tokens_with_attr, 4),
        vec!["Mse".to_string(), "CrossEntropy".to_string()]
    );
    // 末尾カンマなしでも最後の variant を取りこぼさないこと。
    let tokens_no_trailing = tokenize_including_punctuation("pub enum Loss { Mse , CrossEntropy }");
    assert_eq!(
        collect_top_level_enum_variant_idents(&tokens_no_trailing, 4),
        vec!["Mse".to_string(), "CrossEntropy".to_string()]
    );
}

// =====================================================================
// イシュー #2178（親 #2131）: callbacks（CsvLogger・JsonLogger・
// LambdaCallback）の facade 公開保留を検査するテスト群。
// `CallbacksLoggersHoldDoctestGuard`（`src/lib.rs`）の正のプローブの
// ドリフト検査に加え、facade の再エクスポート・独自宣言の不在と
// `compat::Callback` enum の variant 集合を固定する。承認事項の
// 位置づけは `docs/compat-callbacks-loggers-decision.md` §2・§6 を参照。
// =====================================================================

/// `crates/facade/src/lib.rs` の `CallbacksLoggersHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックが glob import するネスト `pub mod`
/// 集合と、`src/lib.rs` の実際の `pub mod` 宣言集合が一致することを
/// 固定する（`optimizer_ext_hold_doctest_globs_all_pub_modules`（#2501 で削除済み） の
/// `CallbacksLoggersHoldDoctestGuard` 版）。
#[test]
fn callbacks_loggers_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "CallbacksLoggersHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "CallbacksLoggersHoldDoctestGuard の doctest ブロックが glob\
         import するモジュール集合が src/lib.rs の pub mod 宣言集合と\
         ドリフトしている（declared={declared:?}, doctest={globbed:?}）。\
         新しい pub mod を追加した場合は doctest 側の use 一覧にも追加\
         すること。"
    );
}

/// [`callbacks_loggers_hold_doctest_globs_all_pub_modules`] が glob
/// import 集合の一致のみを固定するのに対し、本テストは doctest
/// ブロックの**glob 以外の本文**（`__fandhe_callbacks_loggers_hold_probe`
/// モジュール・`__probe` 関数）が固定文言
/// [`CALLBACKS_LOGGERS_HOLD_PROBE_BODY`] と 1 行たりとも違わず一致する
/// ことを固定する（rustdoc の `# ` 隠し行・プローブの削除・別名への
/// シャドーイング等で正のプローブを骨抜きにする改変を機械的に拒否する）。
#[test]
fn callbacks_loggers_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "CallbacksLoggersHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, CALLBACKS_LOGGERS_HOLD_PROBE_BODY,
        "CallbacksLoggersHoldDoctestGuard の doctest ブロック本文（glob\
         以外）が固定文言 CALLBACKS_LOGGERS_HOLD_PROBE_BODY からドリフト\
         している。正のプローブ（__fandhe_callbacks_loggers_hold_probe\
         モジュール・__probe 関数）の削除・弱体化・隠し行の混入がないか\
         確認すること。\n--- actual ---\n{actual}"
    );
}

/// [`callbacks_loggers_hold_doctest_probe_body_matches_fixed_contract`]
/// が要求する固定文言。`crates/facade/src/lib.rs` の
/// `CallbacksLoggersHoldDoctestGuard` doc 内の唯一の doctest ブロック
/// から、ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）
/// を除いた本文と 1 行単位で完全一致する必要がある（クレートルート
/// 自体の `use fandhe_ai::*;` は本文に含む）。
const CALLBACKS_LOGGERS_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_callbacks_loggers_hold_probe {\n\
\x20\x20\x20\x20pub struct CsvLogger;\n\
\x20\x20\x20\x20pub struct JsonLogger;\n\
\x20\x20\x20\x20pub struct CSVLogger;\n\
\x20\x20\x20\x20pub struct JSONLogger;\n\
\x20\x20\x20\x20pub struct LambdaCallback;\n\
}\n\
use __fandhe_callbacks_loggers_hold_probe::*;\n\
\n\
fn __probe(\n\
\x20\x20\x20\x20_: CsvLogger,\n\
\x20\x20\x20\x20_: JsonLogger,\n\
\x20\x20\x20\x20_: CSVLogger,\n\
\x20\x20\x20\x20_: JSONLogger,\n\
\x20\x20\x20\x20_: LambdaCallback,\n\
\x20\x20\x20\x20_: &Callback,\n\
) {\n\
}";

/// [`facade_does_not_reexport_or_declare_callback_loggers`]・その
/// 自己テストが共用する検出本体。facade src 全体（`crates/facade/
/// src/**`）の `pub use` から [`collect_pub_use_leaves`] で別名にする
/// 前の葉を集め `CsvLogger`／`JsonLogger`／`CSVLogger`／`JSONLogger`／
/// `LambdaCallback` を検出し（単一行・複数行・ネストした group・別名も
/// 検出）、`trait`／`struct`／`enum`／`type` 直後の同名独自宣言、
/// および `on_epoch_end` の `fn` 宣言を違反として返す
/// （`scan_optimizer_ext_reexports_and_declarations` と同型。
/// `LambdaCallback::on_epoch_end` はコンストラクタ用の関連関数のため
/// `fn` 宣言の検出を追加する点が `scan_optimizer_ext_*` との差分）。
fn scan_callback_loggers_reexports_and_declarations(content: &str) -> Vec<String> {
    const TYPE_NAMES: [&str; 5] = [
        "CsvLogger",
        "JsonLogger",
        "CSVLogger",
        "JSONLogger",
        "LambdaCallback",
    ];
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if TYPE_NAMES.contains(&leaf.as_str()) {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens
                .get(i + 1)
                .map(|t| TYPE_NAMES.contains(&t.as_str()))
                .unwrap_or(false)
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    offending.extend(
        (0..count_fn_declarations_by_name(&tokens, "on_epoch_end"))
            .map(|_| "fn on_epoch_end 宣言".to_string()),
    );

    offending
}

/// facade src 全体（`crates/facade/src/**`）に、CsvLogger／JsonLogger／
/// CSVLogger／JSONLogger／LambdaCallback（5 個の型名）を識別子単位で
/// 含む `pub use`（複数行・ネストした group・別名含む）も、facade 独自の
/// `trait`／`struct`／`enum`／`type` 宣言も、`on_epoch_end` の `fn`
/// 宣言も存在しないことを固定する（`CallbacksLoggersHoldDoctestGuard`
/// の正のプローブと多層防御を成す最内層のソース走査ガード。
/// `facade_does_not_reexport_or_declare_optimizer_ext_items`（#2501 で `facade_reexports_optimizer_ext_items_only_in_approved_shape` へ反転）と同型）。
#[test]
fn facade_does_not_reexport_or_declare_callback_loggers() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_callback_loggers_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が callbacks ロガー拡張（#2178。CsvLogger／\
         JsonLogger／CSVLogger／JSONLogger／LambdaCallback）を再エクス\
         ポート、または独自宣言している（`docs/compat-callbacks-loggers-\
         decision.md` §2 承認事項が未取得のまま対象外としている設計\
         判断に違反）: {offending:?}"
    );
}

/// [`scan_callback_loggers_reexports_and_declarations`]（[`facade_does_
/// not_reexport_or_declare_callback_loggers`]）の自己テスト（正例・負例
/// の合成入力）。
#[test]
fn facade_does_not_reexport_or_declare_callback_loggers_detects_each_category() {
    // 正例: 単一行 pub use。
    assert!(
        !scan_callback_loggers_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::CsvLogger;"
        )
        .is_empty()
    );
    // 正例: 複数行 pub use（group）。
    assert!(
        !scan_callback_loggers_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::{\n    JsonLogger,\n    LambdaCallback,\n};"
        )
        .is_empty()
    );
    // 正例: 別名 pub use。
    assert!(
        !scan_callback_loggers_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::CSVLogger as Foo;"
        )
        .is_empty()
    );
    // 正例: 独自 struct 宣言。
    assert!(!scan_callback_loggers_reexports_and_declarations("pub struct JSONLogger;").is_empty());
    // 正例: on_epoch_end の fn 宣言。
    assert!(
        !scan_callback_loggers_reexports_and_declarations(
            "impl LambdaCallback { pub fn on_epoch_end() {} }"
        )
        .is_empty()
    );
    // 負例: コメント中の出現。
    assert!(
        scan_callback_loggers_reexports_and_declarations("// pub use ...::CsvLogger;").is_empty()
    );
    // 負例: 無関係な型・関数名。
    assert!(
        scan_callback_loggers_reexports_and_declarations(
            "pub struct EarlyStopping; impl EarlyStopping { pub fn new() {} }"
        )
        .is_empty()
    );
}

/// glob 衝突では enum variant の追加を検出できないため、`compat::
/// Callback` enum（`crates/facade/src/compat/callbacks.rs`。現行は
/// `EarlyStopping`／`ModelCheckpoint`／`LrSchedule` の 3 variant のみ）
/// が、承認なしに `CsvLogger`／`JsonLogger`／`Lambda` 等の variant を
/// 増やされていないことを固定する（`compat_loss_enum_variants_are_
/// exactly_mse_and_cross_entropy` の `Callback` 版。イシュー #2178・
/// `docs/compat-callbacks-loggers-decision.md` §6）。
#[test]
fn compat_callback_enum_variants_are_exactly_expected_while_2178_on_hold() {
    let path = facade_crate_root().join("src/compat/callbacks.rs");
    let content = read_to_string_or_panic(&path);
    let cleaned: String = strip_comments_and_literals(&content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);

    // "pub" "enum" "Callback" "{" の完全一致列を数える（0 件・2 件以上は
    // fail-closed で失敗させる: enum の削除・複数定義・別名への変更を
    // 見逃さない）。
    let mut match_starts: Vec<usize> = Vec::new();
    for i in 0..tokens.len() {
        if tokens.get(i).map(String::as_str) == Some("pub")
            && tokens.get(i + 1).map(String::as_str) == Some("enum")
            && tokens.get(i + 2).map(String::as_str) == Some("Callback")
            && tokens.get(i + 3).map(String::as_str) == Some("{")
        {
            match_starts.push(i + 4);
        }
    }
    assert_eq!(
        match_starts.len(),
        1,
        "callbacks.rs 内の `pub enum Callback {{` 宣言がちょうど 1 件では\
         ない（0 件: enum が削除・改名された。2 件以上: 重複定義。いずれも\
         本テストが検査対象を見失っている）: {} 件",
        match_starts.len()
    );

    let variants = collect_top_level_enum_variant_idents(&tokens, match_starts[0]);
    assert_eq!(
        variants,
        vec![
            "EarlyStopping".to_string(),
            "ModelCheckpoint".to_string(),
            "LrSchedule".to_string(),
        ],
        "compat::Callback の variant 集合が [\"EarlyStopping\",\
         \"ModelCheckpoint\", \"LrSchedule\"] からドリフトしている\
         （未承認のまま variant が追加された可能性。`docs/compat-\
         callbacks-loggers-decision.md` §2 の承認事項参照）: {variants:?}"
    );
}

// #2177（親 #2131）の facade 公開保留固定（`FitWeightingHoldDoctestGuard`）。
// `CallbacksLoggersHoldDoctestGuard`（#2178）と同型の 4 テスト構成。

/// `crates/facade/src/lib.rs` の `FitWeightingHoldDoctestGuard` doc 内の
/// doctest が glob import する `pub mod` 集合が、`src/lib.rs` の実際の
/// `pub mod` 宣言集合と一致することを固定する（`callbacks_loggers_hold_
/// doctest_globs_all_pub_modules` と同型）。
#[test]
fn fit_weighting_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "FitWeightingHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "FitWeightingHoldDoctestGuard の doctest ブロックが glob import\
         するモジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`fit_weighting_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`FIT_WEIGHTING_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（rustdoc の `# ` 隠し行・
/// プローブの削除・別名へのシャドーイング等で正のプローブを骨抜きに
/// する改変を機械的に拒否する）。
#[test]
fn fit_weighting_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "FitWeightingHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, FIT_WEIGHTING_HOLD_PROBE_BODY,
        "FitWeightingHoldDoctestGuard の doctest ブロック本文（glob 以外）\
         が固定文言 FIT_WEIGHTING_HOLD_PROBE_BODY からドリフトしている。\
         正のプローブ（__fandhe_fit_weighting_hold_probe モジュール・\
         __FandheFitWeightHoldProbe トレイト・__probe 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。\n--- actual ---\n{actual}"
    );
}

/// [`fit_weighting_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `FitWeightingHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体
/// の `use fandhe_ai::*;` は本文に含む）。
const FIT_WEIGHTING_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_fit_weighting_hold_probe {\n\
\x20\x20\x20\x20pub struct FitWeights;\n\
}\n\
use __fandhe_fit_weighting_hold_probe::*;\n\
\n\
struct __FandheFitWeightHoldMarker;\n\
\n\
trait __FandheFitWeightHoldProbe {\n\
\x20\x20\x20\x20fn validation_split(&self) -> __FandheFitWeightHoldMarker;\n\
\x20\x20\x20\x20fn class_weight(&self) -> __FandheFitWeightHoldMarker;\n\
\x20\x20\x20\x20fn sample_weight(&self) -> __FandheFitWeightHoldMarker;\n\
\x20\x20\x20\x20fn fit_with_weights(&self) -> __FandheFitWeightHoldMarker;\n\
\x20\x20\x20\x20fn fit_weighted(&self) -> __FandheFitWeightHoldMarker;\n\
}\n\
\n\
impl __FandheFitWeightHoldProbe for fandhe_ai::compat::FitConfig {\n\
\x20\x20\x20\x20fn validation_split(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn class_weight(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn sample_weight(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn fit_with_weights(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn fit_weighted(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheFitWeightHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn validation_split(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn class_weight(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn sample_weight(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn fit_with_weights(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn fit_weighted(&self) -> __FandheFitWeightHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheFitWeightHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe(_: FitWeights, cfg: &fandhe_ai::compat::FitConfig, seq: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheFitWeightHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::FitConfig::validation_split(cfg);\n\
\x20\x20\x20\x20let _: __FandheFitWeightHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::FitConfig::class_weight(cfg);\n\
\x20\x20\x20\x20let _: __FandheFitWeightHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::FitConfig::sample_weight(cfg);\n\
\x20\x20\x20\x20let _: __FandheFitWeightHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::fit_with_weights(seq);\n\
\x20\x20\x20\x20let _: __FandheFitWeightHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::fit_weighted(seq);\n\
}";

/// [`facade_does_not_reexport_or_declare_fit_weighting_items`]・その
/// 自己テストが共用する検出本体。facade src 全体（`crates/facade/
/// src/**`）の `pub use` から [`collect_pub_use_leaves`] で別名にする
/// 前の葉を集め `FitWeights` を検出し（単一行・複数行・ネストした
/// group・別名も検出）、`trait`／`struct`／`enum`／`type` 直後の同名
/// 独自宣言、および `validation_split`／`class_weight`／
/// `sample_weight`／`fit_with_weights`／`fit_weighted` の `fn` 宣言を
/// 違反として返す（`scan_callback_loggers_reexports_and_declarations`
/// と同型。`class_weight`／`sample_weight` を候補名に含めるのは、
/// 受入基準の字面〈`FitConfig` へ直接フィールド追加〉どおりの禁止された
/// 実装経路自体も検出対象に含めるため。`crates/autodiff/src/loss_ops.rs`
/// 等の内部クレート側の既存 `class_weight` 言及は走査対象外〈facade の
/// `src` のみを走査するため〉）。
fn scan_fit_weighting_reexports_and_declarations(content: &str) -> Vec<String> {
    const TYPE_NAMES: [&str; 1] = ["FitWeights"];
    const FN_NAMES: [&str; 5] = [
        "validation_split",
        "class_weight",
        "sample_weight",
        "fit_with_weights",
        "fit_weighted",
    ];
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if TYPE_NAMES.contains(&leaf.as_str()) {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens
                .get(i + 1)
                .map(|t| TYPE_NAMES.contains(&t.as_str()))
                .unwrap_or(false)
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    for name in FN_NAMES {
        offending.extend(
            (0..count_fn_declarations_by_name(&tokens, name)).map(|_| format!("fn {name} 宣言")),
        );
    }

    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`FitWeights`（型名）を
/// 識別子単位で含む `pub use`（複数行・ネストした group・別名含む）も、
/// facade 独自の `trait`／`struct`／`enum`／`type` 宣言も、
/// `validation_split`／`class_weight`／`sample_weight`／
/// `fit_with_weights`／`fit_weighted` の `fn` 宣言も存在しないことを
/// 固定する（`FitWeightingHoldDoctestGuard` の正のプローブと多層防御を
/// 成す最内層のソース走査ガード。
/// `facade_does_not_reexport_or_declare_callback_loggers` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_fit_weighting_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_fit_weighting_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が fit() の重み付け拡張（#2177。FitWeights／\
         validation_split／class_weight／sample_weight／\
         fit_with_weights／fit_weighted）を再エクスポート、または\
         独自宣言している（`docs/compat-fit-sample-weighting-decision.md`\
         §2 承認事項が未取得のまま対象外としている設計判断に違反）: \
         {offending:?}"
    );
}

/// [`scan_fit_weighting_reexports_and_declarations`]（[`facade_does_not_
/// reexport_or_declare_fit_weighting_items`]）の自己テスト（正例・負例
/// の合成入力）。
#[test]
fn facade_does_not_reexport_or_declare_fit_weighting_items_detects_each_category() {
    // 正例: 単一行 pub use（新規型）。
    assert!(
        !scan_fit_weighting_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::FitWeights;"
        )
        .is_empty()
    );
    // 正例: 複数行 pub use（group）。
    assert!(
        !scan_fit_weighting_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::{\n    FitConfig,\n    FitWeights,\n};"
        )
        .is_empty()
    );
    // 正例: 別名 pub use。
    assert!(
        !scan_fit_weighting_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::FitWeights as Foo;"
        )
        .is_empty()
    );
    // 正例: 独自 struct 宣言。
    assert!(!scan_fit_weighting_reexports_and_declarations("pub struct FitWeights;").is_empty());
    // 正例: validation_split の fn 宣言（ビルダー）。
    assert!(
        !scan_fit_weighting_reexports_and_declarations(
            "impl FitConfig { pub fn validation_split(self, f: f32) -> Self { self } }"
        )
        .is_empty()
    );
    // 正例: 受入基準の字面どおりの禁止経路（FitConfig への class_weight
    // フィールド追加想定の fn 宣言）。
    assert!(
        !scan_fit_weighting_reexports_and_declarations(
            "impl FitConfig { pub fn class_weight(self, w: HashMap<u32, f32>) -> Self { self } }"
        )
        .is_empty()
    );
    // 正例: fit_with_weights の fn 宣言。
    assert!(
        !scan_fit_weighting_reexports_and_declarations(
            "impl Sequential { pub fn fit_with_weights(&mut self) {} }"
        )
        .is_empty()
    );
    // 負例: コメント中の出現。
    assert!(
        scan_fit_weighting_reexports_and_declarations("// pub use ...::FitWeights;").is_empty()
    );
    // 負例: 無関係な型・関数名。
    assert!(
        scan_fit_weighting_reexports_and_declarations(
            "pub struct FitConfig; impl FitConfig { pub fn new() {} }"
        )
        .is_empty()
    );
}

/// `FitConfig` が `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` を
/// 維持していることを固定する（`crates/facade/src/compat/training.rs:
/// 175`）。受入基準の字面（`HashMap<u32, f32>` の class_weight・
/// `&[f32]` の sample_weight を直接フィールド追加）どおりに実装すると
/// `Copy`（`HashMap` 保持）・`Eq`（`f32` 保持）のいずれかが外れ、
/// crates.io 出荷済み `fandhe-ai =0.9.0` の公開 API 非破壊契約に反する
/// （`docs/compat-fit-sample-weighting-decision.md` §2）。本テストは
/// その非破壊契約（derive 維持）を正のガードとして直接固定する。
#[test]
fn fit_config_keeps_copy_eq_for_0_9_0_compat() {
    fn assert_copy_eq<T: Copy + Eq>() {}
    assert_copy_eq::<fandhe_ai::compat::FitConfig>();
}

// =====================================================================
// #2508（親 #2500・ルート #2499。2026-10-04 承認）`FitConfig::
// accumulate_steps` 公開面の正ガード。#2180 の保留（`GradAccumulation
// HoldDoctestGuard` と否定テスト 3 件）を、承認した形（公開ビルダー
// 1 件のみ）だけを許す正ガードへ反転した。先例は #2198 の
// `compat_optimizer_enum_has_lbfgs_variant`・#2338 の `nn_module_*`。
// =====================================================================

/// `fn accumulate_steps` の宣言が facade src 全体でちょうど 1 件
/// （`src/compat/training.rs`）であることを固定する。0 件（公開の
/// 取り下げ・改名）も 2 件以上（別経路の二重定義）も fail-closed。
#[test]
fn facade_declares_fit_config_accumulate_steps_exactly_once() {
    let src_dir = facade_crate_root().join("src");
    let mut found: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        let count = count_fn_declarations_by_name(&tokens, "accumulate_steps");
        for _ in 0..count {
            found.push(path.to_string_lossy().replace('\\', "/"));
        }
    });
    assert_eq!(
        found.len(),
        1,
        "`fn accumulate_steps` の宣言は facade src 全体で 1 件（FitConfig の\
         公開ビルダー。イシュー #2508）のはず: {found:?}"
    );
    assert!(
        found[0].ends_with("src/compat/training.rs"),
        "`fn accumulate_steps` は src/compat/training.rs の FitConfig に\
         宣言されるはず: {found:?}"
    );
}

/// #2180 のテスト専用セッター `with_accumulate_steps_for_test` が残って
/// いないことを固定する（公開経路と並ぶ迂回入口を残さない）。
#[test]
fn facade_does_not_declare_with_accumulate_steps_for_test() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
        let tokens = tokenize_including_punctuation(&cleaned);
        if count_fn_declarations_by_name(&tokens, "with_accumulate_steps_for_test") > 0 {
            offending.push(path.display().to_string());
        }
    });
    assert!(
        offending.is_empty(),
        "テスト専用セッター with_accumulate_steps_for_test が残っている\
         （#2508 で公開ビルダーへ置き換え済みのはず）: {offending:?}"
    );
}

/// `FitConfig::accumulate_steps` が facade の import のみ
/// （`fandhe_ai_autodiff` 不要）で到達でき、シグネチャ
/// `fn(FitConfig, u32) -> FitConfig` と「値を実際に書き換える」挙動を
/// 持つことを固定する（`self` をそのまま返すスタブでは通らない）。
#[test]
fn fit_config_accumulate_steps_is_reachable_via_facade_only() {
    use fandhe_ai::compat::FitConfig;
    let f: fn(FitConfig, u32) -> FitConfig = FitConfig::accumulate_steps;
    let base = FitConfig::new(1, 1);
    assert_eq!(f(base, 1), base, "既定値 1 は FitConfig::new と同値のはず");
    assert_ne!(f(base, 2), base, "n=2 はフィールドを書き換えるはず");
    assert_eq!(f(base, 2), f(base, 2));
    assert_ne!(f(base, 2), f(base, 3), "n の違いが区別されるはず");
}

// =====================================================================
// #2184（親 #2131）の facade 公開保留固定（`TrainStepHoldDoctestGuard`）。
// カスタム学習 step フック本体（`CustomStepHook`・`Sequential::run_fit`
// への配線）は実装済みで、保留対象は facade 公開面 3 件（`TrainStepFn`／
// `TrainStepOptimizer`／`TrainStepOutput`・`Sequential::
// fit_with_train_step`）のみ。`FitWeightingHoldDoctestGuard`（#2177）と
// 同型の 4 テスト構成。
// =====================================================================

/// `crates/facade/src/lib.rs` の `TrainStepHoldDoctestGuard` doc 内の
/// 唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する。
#[test]
fn train_step_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "TrainStepHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "TrainStepHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`train_step_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`TRAIN_STEP_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（rustdoc の `# ` 隠し行・
/// プローブの削除・別名へのシャドーイング等で正のプローブを骨抜きに
/// する改変を機械的に拒否する）。
#[test]
fn train_step_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "TrainStepHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, TRAIN_STEP_HOLD_PROBE_BODY,
        "TrainStepHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 TRAIN_STEP_HOLD_PROBE_BODY からドリフトしている。\
         正のプローブ（__FandheTrainStepHoldProbe トレイト・__probe_type／\
         __probe 関数）の削除・弱体化・隠し行の混入がないか確認すること。\n\
         --- actual ---\n{actual}"
    );
}

/// [`train_step_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `TrainStepHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const TRAIN_STEP_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_train_step_hold_probe {\n\
\x20\x20\x20\x20pub struct TrainStepFn;\n\
\x20\x20\x20\x20pub struct TrainStepOptimizer;\n\
\x20\x20\x20\x20pub struct TrainStepOutput;\n\
}\n\
use __fandhe_train_step_hold_probe::*;\n\
\n\
struct __FandheTrainStepHoldMarker;\n\
\n\
trait __FandheTrainStepHoldProbe {\n\
\x20\x20\x20\x20fn train_step_fn(&self) -> __FandheTrainStepHoldMarker;\n\
\x20\x20\x20\x20fn fit_with_train_step(&self) -> __FandheTrainStepHoldMarker;\n\
}\n\
\n\
impl __FandheTrainStepHoldProbe for fandhe_ai::compat::FitConfig {\n\
\x20\x20\x20\x20fn train_step_fn(&self) -> __FandheTrainStepHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheTrainStepHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn fit_with_train_step(&self) -> __FandheTrainStepHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheTrainStepHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheTrainStepHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn train_step_fn(&self) -> __FandheTrainStepHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheTrainStepHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn fit_with_train_step(&self) -> __FandheTrainStepHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheTrainStepHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_type(_: TrainStepFn, _: TrainStepOptimizer, _: TrainStepOutput) {}\n\
\n\
fn __probe(cfg: &fandhe_ai::compat::FitConfig, seq: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheTrainStepHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::FitConfig::train_step_fn(cfg);\n\
\x20\x20\x20\x20let _: __FandheTrainStepHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::fit_with_train_step(seq);\n\
}";

/// [`facade_does_not_reexport_or_declare_train_step_items`]・その
/// 自己テストが共用する検出本体。facade src 全体（`crates/facade/
/// src/**`）の `pub use` から [`collect_pub_use_leaves`] で別名にする
/// 前の葉を集め `TrainStepFn`／`TrainStepOptimizer`／`TrainStepOutput`
/// を検出し（単一行・複数行・ネストした group・別名も検出）、
/// `trait`／`struct`／`enum`／`type` 直後の同名独自宣言、および
/// `train_step_fn`／`fit_with_train_step` の `fn` 宣言を違反として返す
/// （`scan_fit_weighting_reexports_and_declarations` と同型）。
fn scan_train_step_reexports_and_declarations(content: &str) -> Vec<String> {
    const TYPE_NAMES: [&str; 3] = ["TrainStepFn", "TrainStepOptimizer", "TrainStepOutput"];
    const FN_NAMES: [&str; 2] = ["train_step_fn", "fit_with_train_step"];
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if TYPE_NAMES.contains(&leaf.as_str()) {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens
                .get(i + 1)
                .map(|t| TYPE_NAMES.contains(&t.as_str()))
                .unwrap_or(false)
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    for name in FN_NAMES {
        offending.extend(
            (0..count_fn_declarations_by_name(&tokens, name)).map(|_| format!("fn {name} 宣言")),
        );
    }

    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`TrainStepFn`／
/// `TrainStepOptimizer`／`TrainStepOutput`（型名）を識別子単位で含む
/// `pub use`（複数行・ネストした group・別名含む）も、facade 独自の
/// `trait`／`struct`／`enum`／`type` 宣言も、`train_step_fn`／
/// `fit_with_train_step` の `fn` 宣言も存在しないことを固定する
/// （`TrainStepHoldDoctestGuard` の正のプローブと多層防御を成す最内層
/// のソース走査ガード。`facade_does_not_reexport_or_declare_fit_
/// weighting_items` と同型）。テスト専用入口
/// `Sequential::fit_custom_step_for_test`（`crates/facade/src/compat/
/// training.rs`）はこれらの名前とは異なる別名のため検出対象に含まれ
/// ない（意図的な命名回避。同ファイルの doc 参照）。
#[test]
fn facade_does_not_reexport_or_declare_train_step_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_train_step_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面がカスタム学習 step フック（#2184。TrainStepFn／\
         TrainStepOptimizer／TrainStepOutput／train_step_fn／\
         fit_with_train_step）を再エクスポート、または独自宣言している\
         （`docs/compat-train-step-hook-decision.md` §5 承認事項が未取得\
         のまま対象外としている設計判断に違反）: {offending:?}"
    );
}

/// [`scan_train_step_reexports_and_declarations`]（[`facade_does_not_
/// reexport_or_declare_train_step_items`]）の自己テスト（正例・負例の
/// 合成入力）。
#[test]
fn facade_does_not_reexport_or_declare_train_step_items_detects_each_category() {
    // 正例: 単一行 pub use（新規型）。
    assert!(
        !scan_train_step_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::TrainStepFn;"
        )
        .is_empty()
    );
    // 正例: 複数行 pub use（group）。
    assert!(
        !scan_train_step_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::{\n    TrainStepOptimizer,\n    TrainStepOutput,\n};"
        )
        .is_empty()
    );
    // 正例: 別名 pub use。
    assert!(
        !scan_train_step_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::TrainStepFn as Foo;"
        )
        .is_empty()
    );
    // 正例: 独自 struct 宣言。
    assert!(!scan_train_step_reexports_and_declarations("pub struct TrainStepOutput;").is_empty());
    // 正例: train_step_fn の fn 宣言（型エイリアス相当のフリー関数想定）。
    assert!(
        !scan_train_step_reexports_and_declarations(
            "pub fn train_step_fn() -> TrainStepFn { TrainStepFn }"
        )
        .is_empty()
    );
    // 正例: fit_with_train_step の fn 宣言。
    assert!(
        !scan_train_step_reexports_and_declarations(
            "impl Sequential { pub fn fit_with_train_step(&mut self) {} }"
        )
        .is_empty()
    );
    // 負例: コメント中の出現。
    assert!(scan_train_step_reexports_and_declarations("// pub use ...::TrainStepFn;").is_empty());
    // 負例: 内部テスト専用 helper 名（意図的な命名回避との非衝突確認）。
    assert!(
        scan_train_step_reexports_and_declarations(
            "impl Sequential { fn fit_custom_step_for_test(&mut self) {} }"
        )
        .is_empty()
    );
    // 負例: 無関係な型・関数名。
    assert!(
        scan_train_step_reexports_and_declarations(
            "pub struct FitConfig; impl FitConfig { pub fn new() {} }"
        )
        .is_empty()
    );
}

// =====================================================================
// #2188（親 #2131）で導入し、#2369（親 #2362）で正ガードへ反転した
// `compat::save_model`／`load_model`／`ModelIoError` の facade 公開面の固定。
// 自由関数とエラー型は承認・公開済みのため、ソース走査は「承認済みの形で
// ちょうど 1 件だけ存在すること」を固定する正ガード
// （`facade_model_io_public_surface_matches_approved_contract` ほか）になった。
// 代替案の inherent メソッド（`Sequential::save`／`load`）は承認範囲外のため、
// `ModelIoHoldDoctestGuard`（代替案専用へ縮小）と `impl Sequential` 内
// `fn save`／`fn load` の 0 件固定（workspace インベントリ）を維持している。
// =====================================================================

/// `crates/facade/src/lib.rs` の `ModelIoHoldDoctestGuard` doc 内の
/// doctest が glob import する `pub mod` 集合が、`src/lib.rs` の実際の
/// `pub mod` 宣言集合と一致することを固定する（`train_step_hold_doctest_
/// globs_all_pub_modules` と同型）。
#[test]
fn model_io_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "ModelIoHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "ModelIoHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`model_io_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`MODEL_IO_HOLD_PROBE_BODY`] と 1 行
/// たりとも違わず一致することを固定する（rustdoc の `# ` 隠し行・
/// プローブの削除・別名へのシャドーイング等で正のプローブを骨抜きに
/// する改変を機械的に拒否する）。
#[test]
fn model_io_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "ModelIoHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, MODEL_IO_HOLD_PROBE_BODY,
        "ModelIoHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 MODEL_IO_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（#2369 で縮小済み。\
         __FandheModelIoHoldProbe トレイト・__probe_* 関数群）の削除・\
         弱体化・隠し行の混入がないか確認すること。\n--- actual ---\n{actual}"
    );
}

/// [`model_io_hold_doctest_probe_body_matches_fixed_contract`] が要求
/// する固定文言。`crates/facade/src/lib.rs` の `ModelIoHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致する
/// 必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に含む）。
const MODEL_IO_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
struct __FandheModelIoHoldMarker;\n\
\n\
trait __FandheModelIoHoldProbe {\n\
\x20\x20\x20\x20fn save_model(&self) -> __FandheModelIoHoldMarker;\n\
\x20\x20\x20\x20fn load_model(&self) -> __FandheModelIoHoldMarker;\n\
\x20\x20\x20\x20fn save(&self) -> __FandheModelIoHoldMarker;\n\
\x20\x20\x20\x20fn load(&self) -> __FandheModelIoHoldMarker;\n\
}\n\
\n\
impl __FandheModelIoHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn save_model(&self) -> __FandheModelIoHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheModelIoHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load_model(&self) -> __FandheModelIoHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheModelIoHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn save(&self) -> __FandheModelIoHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheModelIoHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn load(&self) -> __FandheModelIoHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheModelIoHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_inherent_method(seq: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheModelIoHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::save_model(seq);\n\
\x20\x20\x20\x20let _: __FandheModelIoHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::load_model(seq);\n\
\x20\x20\x20\x20let _: __FandheModelIoHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::save(seq);\n\
\x20\x20\x20\x20let _: __FandheModelIoHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::load(seq);\n\
}";

/// `model_io` 正ガード（[`facade_model_io_public_surface_matches_approved_contract`]・
/// その自己テストが共用）の 1 ファイル分の検出結果。
///
/// #2369（親 #2362）で `compat::save_model`／`compat::load_model`／
/// `compat::ModelIoError` が公開されたため、#2188 の否定ガード（「存在しないこと」）を
/// 正ガード（「承認済みの形で、承認済みの場所に、ちょうど 1 件だけ存在すること」）へ反転した。
#[derive(Default)]
struct ModelIoFileScan {
    /// 修飾子なし（private）の `mod model_io` 宣言数。
    private_mod_decls: usize,
    /// `pub mod model_io`・`pub(crate) mod model_io` 等、修飾子付きの `mod model_io` 宣言数。
    qualified_mod_decls: usize,
    /// 葉に `model_io`／`save_model`／`load_model`／`ModelIoError` を持つ
    /// `pub use` 文の正規化文字列（トークンを空白なしで連結したもの）。
    pub_use_statements: Vec<String>,
    /// `enum ModelIoError` 宣言数。
    enum_decls: usize,
    /// `trait`／`struct`／`type` による `ModelIoError` 宣言数。
    other_type_decls: usize,
    /// `fn save_model` 宣言数。
    save_model_fns: usize,
    /// `fn load_model` 宣言数。
    load_model_fns: usize,
    /// `impl Sequential { .. }` 内の `fn save`／`fn load` 宣言数（代替案。承認外）。
    alt_impl_offenses: usize,
}

/// `crates/facade/src/` 内で承認済みの `pub use`（compat 公開面。`compat/mod.rs` の 1 文 1 行）。
const MODEL_IO_APPROVED_PUB_USE: &str = "model_io::{ModelIoError,load_model,save_model}";
/// `mod model_io;` と承認済み `pub use` を置く唯一のファイル（`src/` からの相対パス）。
const MODEL_IO_MOD_FILE: &str = "compat/mod.rs";
/// `save_model`／`load_model`／`ModelIoError` の唯一の定義ファイル（同上）。
const MODEL_IO_IMPL_FILE: &str = "compat/model_io.rs";

/// 1 ファイルから [`ModelIoFileScan`] を作る（コメント・リテラルを除いたトークン列を走査。
/// `scan_train_step_reexports_and_declarations` と同型で、単一行・複数行・ネストした
/// group・別名の `pub use` も葉の単位で検出する）。
fn scan_model_io_file(content: &str) -> ModelIoFileScan {
    const REEXPORT_LEAF_NAMES: [&str; 4] = ["model_io", "save_model", "load_model", "ModelIoError"];
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut scan = ModelIoFileScan::default();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            if leaves
                .iter()
                .any(|leaf| REEXPORT_LEAF_NAMES.contains(&leaf.as_str()))
            {
                // 複数行 group の末尾カンマ（rustfmt の整形差）は同一文として扱う。
                scan.pub_use_statements
                    .push(path_tokens.concat().replace(",}", "}"));
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if tokens[i] == "ModelIoError" && i > 0 {
            match tokens[i - 1].as_str() {
                "enum" => scan.enum_decls += 1,
                "trait" | "struct" | "type" => scan.other_type_decls += 1,
                _ => {}
            }
        }
        if tokens[i] == "mod" && tokens.get(i + 1).map(String::as_str) == Some("model_io") {
            // 直前が `pub` または `)`（`pub(crate)` 等の修飾子）なら修飾子付き。
            let qualified = i > 0 && matches!(tokens[i - 1].as_str(), "pub" | ")");
            if qualified {
                scan.qualified_mod_decls += 1;
            } else {
                scan.private_mod_decls += 1;
            }
        }
        i += 1;
    }

    scan.save_model_fns = count_fn_declarations_by_name(&tokens, "save_model");
    scan.load_model_fns = count_fn_declarations_by_name(&tokens, "load_model");
    scan.alt_impl_offenses = scan_sequential_alt_save_load_impls(content).len();
    scan
}

/// `(src からの相対パス, 内容)` の集合に対し、`model_io` 公開面の正契約を検査して違反を返す
/// （空なら適合）。契約:
///
/// - `mod model_io;` は `compat/mod.rs` にちょうど 1 件で、`pub` 等の修飾子なし
/// - `model_io`／`save_model`／`load_model`／`ModelIoError` を葉に持つ `pub use` は
///   `compat/mod.rs` の固定形 `pub use model_io::{ModelIoError, load_model, save_model};`
///   ちょうど 1 文だけ（別名・分割・他ファイルからの再エクスポートは違反）
/// - `ModelIoError` の型宣言は `compat/model_io.rs` の `enum` 1 件だけ
/// - `fn save_model`／`fn load_model` は `compat/model_io.rs` に各 1 件だけ
/// - 代替案 `Sequential::save`／`Sequential::load`（`impl Sequential` 内の `fn save`／`fn load`）は 0 件
///   （承認外。#2362 契約）
fn model_io_contract_violations(files: &[(String, String)]) -> Vec<String> {
    let mut violations: Vec<String> = Vec::new();
    let mut private_mods = 0usize;
    let mut approved_uses = 0usize;
    let mut impl_enums = 0usize;
    let mut impl_save_fns = 0usize;
    let mut impl_load_fns = 0usize;

    for (rel, content) in files {
        let scan = scan_model_io_file(content);
        let is_mod_file = rel == MODEL_IO_MOD_FILE;
        let is_impl_file = rel == MODEL_IO_IMPL_FILE;

        for _ in 0..scan.qualified_mod_decls {
            violations.push(format!("{rel}: 修飾子付きの mod model_io 宣言"));
        }
        for _ in 0..scan.private_mod_decls {
            if is_mod_file {
                private_mods += 1;
            } else {
                violations.push(format!(
                    "{rel}: {MODEL_IO_MOD_FILE} 以外の mod model_io 宣言"
                ));
            }
        }
        for stmt in &scan.pub_use_statements {
            if is_mod_file && stmt == MODEL_IO_APPROVED_PUB_USE {
                approved_uses += 1;
            } else {
                violations.push(format!("{rel}: 承認外の pub use（{stmt}）"));
            }
        }
        for _ in 0..scan.enum_decls {
            if is_impl_file {
                impl_enums += 1;
            } else {
                violations.push(format!(
                    "{rel}: {MODEL_IO_IMPL_FILE} 以外の enum ModelIoError 宣言"
                ));
            }
        }
        for _ in 0..scan.other_type_decls {
            violations.push(format!(
                "{rel}: trait／struct／type による ModelIoError 宣言"
            ));
        }
        for (name, count, slot) in [
            ("save_model", scan.save_model_fns, &mut impl_save_fns),
            ("load_model", scan.load_model_fns, &mut impl_load_fns),
        ] {
            if is_impl_file {
                *slot += count;
            } else {
                for _ in 0..count {
                    violations.push(format!("{rel}: {MODEL_IO_IMPL_FILE} 以外の fn {name} 宣言"));
                }
            }
        }
        for _ in 0..scan.alt_impl_offenses {
            violations.push(format!(
                "{rel}: impl Sequential 内の fn save／fn load 宣言（代替案は承認外）"
            ));
        }
    }

    for (what, actual) in [
        ("mod model_io（private）", private_mods),
        ("承認済みの pub use", approved_uses),
        ("enum ModelIoError", impl_enums),
        ("fn save_model", impl_save_fns),
        ("fn load_model", impl_load_fns),
    ] {
        if actual != 1 {
            violations.push(format!("{what} がちょうど 1 件ではない（{actual} 件）"));
        }
    }
    violations
}

/// `impl Sequential { .. }`／`impl <Trait> for Sequential { .. }`
/// ブロック内に限定して `fn save`／`fn load` 宣言を検出する
/// （`docs/compat-model-io-decision.md` §2 の代替公開 API 案
/// `Sequential::save(&self, dir)`／`Sequential::load(dir)` を facade が
/// 追加したことを検出する。PR #2317 review 指摘）。`save`／`load` は
/// ワークスペース内に無関係な既存宣言（例:
/// `crates/facade/src/model.rs::ModelRegistry::load`）があるため、
/// グローバルな `fn` 名走査（[`count_fn_declarations_by_name`] を素朴に
/// 適用する形）では誤検出する。`impl` ヘッダ（`impl` から本体開始 `{`
/// の直前まで）のトークン列に識別子 `Sequential` を含む場合に限り、
/// その impl 本体（対応する閉じ `}` まで深さカウントで走査）だけを
/// 対象に `fn save`／`fn load` を数える。
///
/// 本関数自体は呼び出し元が渡した走査対象（文字列 `content`）にのみ
/// 依存し `Sequential` 型の一意性を前提にしない。呼び出し元
/// （[`facade_model_io_public_surface_matches_approved_contract`]）が
/// `crates/facade/src/**` に限定して呼ぶのは型の一意性ゆえではなく
/// **依存方向**が理由: 本イシューが対象とする公開型
/// `fandhe_ai::compat::Sequential`（facade クレート `fandhe-ai` で定義）
/// を名指しできるのは facade に依存するクレートだけだが、facade を
/// 依存する workspace クレートは存在しない（`crates/*/Cargo.toml` に
/// package 名 `fandhe-ai` への依存宣言なし）ため、facade 外のクレートは
/// そもそも `impl ... for Sequential` の `Sequential` としてこの公開型を
/// 参照できない。ワークスペース内には識別子 `Sequential` を持つ別の
/// 内部専用型（`crates/autodiff/src/compat/sequential.rs`・
/// `crates/autodiff/src/nn/container.rs`。facade から非再エクスポート）
/// も存在するが、これらは本イシューの対象型ではないため、facade 限定の
/// 走査で公開面の保留固定としては十分。ワークスペース全体（他クレートの
/// 同名内部型を含む）を横断する定義元インベントリは別テスト
/// [`workspace_declares_sequential_alt_save_load_fn_names_only_in_allowed_locations`]
/// が本関数を再利用して担う。
fn scan_sequential_alt_save_load_impls(content: &str) -> Vec<String> {
    const ALT_FN_NAMES: [&str; 2] = ["save", "load"];
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] != "impl" {
            i += 1;
            continue;
        }
        let mut header_end = i + 1;
        while header_end < tokens.len() && tokens[header_end] != "{" {
            header_end += 1;
        }
        let is_sequential_impl = tokens[i + 1..header_end].iter().any(|t| t == "Sequential");
        if !is_sequential_impl || header_end >= tokens.len() {
            i = header_end + 1;
            continue;
        }
        // 対応する閉じ `}` まで深さカウントで走査する（ネストした
        // ブロック・関数本体の `{`／`}` も跨いで対応を取る）。
        let mut depth = 0usize;
        let mut j = header_end;
        let mut body_end = tokens.len();
        while j < tokens.len() {
            match tokens[j].as_str() {
                "{" => depth += 1,
                "}" => {
                    depth -= 1;
                    if depth == 0 {
                        body_end = j;
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        let body_tokens = &tokens[header_end..body_end.min(tokens.len())];
        for name in ALT_FN_NAMES {
            offending.extend(
                (0..count_fn_declarations_by_name(body_tokens, name))
                    .map(|_| format!("impl Sequential 内の fn {name} 宣言")),
            );
        }
        i = body_end + 1;
    }

    offending
}

/// facade src 全体（`crates/facade/src/**`）の `model_io` 公開面が、承認済みの形で
/// ちょうど 1 件だけ存在することを固定する正ガード（イシュー #2369・親 #2362。
/// #2188 の否定ガード `facade_does_not_reexport_or_declare_model_io` を反転したもの）。
/// 契約の詳細は [`model_io_contract_violations`] を参照。`save_model`／`load_model` の
/// 定義元インベントリは [`workspace_declares_model_io_fn_names_only_in_allowed_locations`]
/// が workspace 全体で別途固定する。
#[test]
fn facade_model_io_public_surface_matches_approved_contract() {
    let src_dir = facade_crate_root().join("src");
    let mut files: Vec<(String, String)> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let rel = path
            .strip_prefix(&src_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        files.push((rel, content.to_string()));
    });
    files.sort();
    assert!(
        files.len() > 1,
        "facade src から .rs ファイルを十分に抽出できなかった（テスト自体が検査対象を見失っている可能性がある）"
    );
    let violations = model_io_contract_violations(&files);
    assert!(
        violations.is_empty(),
        "facade の model_io 公開面（compat::save_model／load_model／ModelIoError。#2369）が\
         承認済みの形（`docs/compat-model-io-decision.md` §2 item 1）から外れている: {violations:?}"
    );
}

/// [`model_io_contract_violations`]（[`facade_model_io_public_surface_matches_approved_contract`]）の
/// 自己テスト。承認済みの形は違反 0、各違反カテゴリの合成入力は違反を検出することを固定する
/// （正のプローブ。否定ガードだけでは検出器の空振りに気づけないため）。
#[test]
fn facade_model_io_public_surface_detects_each_category() {
    fn baseline() -> Vec<(String, String)> {
        vec![
            (
                "compat/mod.rs".to_string(),
                "mod model_io;\npub use model_io::{ModelIoError, load_model, save_model};\n"
                    .to_string(),
            ),
            (
                "compat/model_io.rs".to_string(),
                "pub enum ModelIoError {}\npub fn save_model() {}\npub fn load_model() {}\n"
                    .to_string(),
            ),
        ]
    }
    fn with(mutate: impl FnOnce(&mut Vec<(String, String)>)) -> Vec<String> {
        let mut files = baseline();
        mutate(&mut files);
        model_io_contract_violations(&files)
    }

    // 正例: 承認済みの形は違反 0（複数行の group でも同一文とみなす）。
    assert!(model_io_contract_violations(&baseline()).is_empty());
    assert!(
        with(|f| {
            f[0].1 = "mod model_io;\npub use model_io::{\n    ModelIoError,\n    load_model,\n    save_model,\n};\n"
                .to_string();
        })
        .is_empty()
    );
    // 負例: コメント中の出現・無関係な宣言は違反にならない。
    assert!(
        with(|f| f.push((
            "lib.rs".to_string(),
            "// pub use crate::compat::save_model;\npub struct FitConfig;\nimpl FitConfig { pub fn new() {} }\n"
                .to_string()
        )))
        .is_empty()
    );
    // 負例: `Sequential` を含まない impl の `fn load` は代替案ではない。
    assert!(
        with(|f| f.push((
            "model.rs".to_string(),
            "impl ModelRegistry { pub fn load(&self) {} }\n".to_string()
        )))
        .is_empty()
    );

    // 違反: 何も存在しない（承認済みの公開面が消えた）。
    assert!(!model_io_contract_violations(&[]).is_empty());
    // 違反: `mod model_io` の公開・修飾子付き・他ファイルでの宣言・重複。
    assert!(!with(|f| f[0].1 = f[0].1.replace("mod model_io;", "pub mod model_io;")).is_empty());
    assert!(
        !with(|f| f[0].1 = f[0].1.replace("mod model_io;", "pub(crate) mod model_io;")).is_empty()
    );
    assert!(!with(|f| f.push(("lib.rs".to_string(), "mod model_io;".to_string()))).is_empty());
    // 違反: 別名・分割・他ファイルからの再エクスポート。
    assert!(
        !with(|f| {
            f[0].1 = "mod model_io;\npub use model_io::ModelIoError as Foo;\npub use model_io::{load_model, save_model};\n"
                .to_string();
        })
        .is_empty()
    );
    assert!(
        !with(|f| {
            f[0].1 = "mod model_io;\npub use model_io::{ModelIoError, load_model};\npub use model_io::save_model;\n"
                .to_string();
        })
        .is_empty()
    );
    assert!(
        !with(|f| f.push((
            "lib.rs".to_string(),
            "pub use crate::compat::save_model;".to_string()
        )))
        .is_empty()
    );
    assert!(
        !with(|f| f.push((
            "lib.rs".to_string(),
            "pub use crate::compat::{ModelIoError as E};".to_string()
        )))
        .is_empty()
    );
    // 違反: 他ファイルでの型・関数の宣言・model_io.rs 内での重複。
    assert!(
        !with(|f| f.push(("lib.rs".to_string(), "pub struct ModelIoError;".to_string())))
            .is_empty()
    );
    assert!(
        !with(|f| f.push(("lib.rs".to_string(), "pub enum ModelIoError {}".to_string())))
            .is_empty()
    );
    assert!(
        !with(|f| f.push(("lib.rs".to_string(), "pub fn load_model() {}".to_string()))).is_empty()
    );
    assert!(!with(|f| f[1].1.push_str("pub fn save_model() {}\n")).is_empty());
    // 違反: 代替案の inherent メソッド（承認外）。
    assert!(
        !with(|f| f[1]
            .1
            .push_str("impl Sequential { pub fn save(&self, dir: &Path) {} }\n"))
        .is_empty()
    );
    assert!(
        !with(|f| f.push((
            "compat/sequential.rs".to_string(),
            "impl Sequential { pub fn load(dir: &Path) {} }".to_string()
        )))
        .is_empty()
    );
    assert!(
        !with(|f| f.push((
            "compat/sequential.rs".to_string(),
            "impl ModelIo for Sequential { fn save(&self) {} }".to_string()
        )))
        .is_empty()
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、`fn save_model`／
/// `fn load_model` の定義元集合を固定する
/// （`workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_
/// locations` と同型のインベントリ）。
///
/// **期待集合**（#2369・親 #2362 で差し替え済み）: `save_model`／
/// `load_model` という名前の `fn` 宣言は workspace 全体（`crates/*/src/`）で
/// `facade/src/compat/model_io.rs` の各 1 件だけ
/// （`docs/compat-model-io-decision.md` §2 item 1）。過不足いずれも
/// fail-closed に検出する。
#[test]
fn workspace_declares_model_io_fn_names_only_in_allowed_locations() {
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

    const NAMES: [&str; 2] = ["save_model", "load_model"];

    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            for fn_name in NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("facade/src/compat/model_io.rs::save_model".to_string(), 1),
        ("facade/src/compat/model_io.rs::load_model".to_string(), 1),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の save_model／load_model 系\
         `fn` 宣言集合が期待（facade/src/compat/model_io.rs の各 1 件）と\
         一致しない（承認外の場所への実装が紛れ込んだ、または定義が\
         消えた可能性がある。過不足いずれも fail-closed に検出する）: {found:?}"
    );
}

/// [`workspace_declares_model_io_fn_names_only_in_allowed_locations`]
/// （`save_model`／`load_model` の定義元インベントリ）を、
/// `docs/compat-model-io-decision.md` §2 の代替公開 API 案
/// `Sequential::save(&self, dir)`／`Sequential::load(dir)` にも横展開した
/// 第 3 層。[`facade_model_io_public_surface_matches_approved_contract`]（第 2 層）は
/// `crates/facade/src/**` に走査対象を限定しているため（`Sequential` は
/// facade でのみ公開され、facade を依存する workspace クレートが存在しない
/// ため facade 外から `impl ... for Sequential` を書けない。PR #2317
/// review 再確認時の指摘: 「`Sequential` 型はこのワークスペースで facade
/// にのみ定義される」という以前の理由づけは誤りで、実際には
/// `crates/autodiff/src/compat/sequential.rs`・
/// `crates/autodiff/src/nn/container.rs` にも同名の別型 `Sequential` が
/// 存在する。ただしこれらは facade から再エクスポートされない内部専用型
/// であり本イシューの対象外の型のため、第 2 層の走査範囲限定自体は妥当。
/// 正しい制約は「型の一意性」ではなく「依存方向」: workspace 内のどの
/// クレートも `fandhe-ai`（facade）package に依存していないため、facade
/// 外のクレートは facade の `compat::Sequential` を名指しできず、
/// `impl ... for` 節にも書けない）、facade の走査だけでは「承認前に
/// 本番実装がどこか別クレートへ迂回的に紛れ込んでいないか」という
/// ワークスペース全体の定義元インベントリという第 3 層の役割を満たさない。
/// 本テストは [`scan_sequential_alt_save_load_impls`]
/// （`Sequential` を含む `impl` ヘッダ配下に限定した `fn save`／`fn load`
/// 宣言の検出。ワークスペース内の無関係な `save`／`load`〈例:
/// `ModelRegistry::load`〉を誤検出しない）を `crates/*/src/` 全体へ適用し、
/// 期待集合（空集合）との完全一致を固定する。
///
/// **期待集合**（着手前確認の再 grep で判明。2026-09-27）: ワークスペース内
/// に識別子 `Sequential` を含む `impl` ブロックは facade（本イシュー対象の
/// `compat::Sequential`）に加え、autodiff 内部専用の 2 型
/// （`crates/autodiff/src/compat/sequential.rs::Sequential`・
/// `crates/autodiff/src/nn/container.rs::Sequential`）にも存在するが、
/// いずれにも `fn save`／`fn load` 宣言はなく空集合。#2369（親 #2362）では
/// 承認されたのが自由関数 `save_model`／`load_model` だけで、代替案の inherent メソッドは
/// 承認範囲外のため、空集合の固定を維持する（差し替えない）。
#[test]
fn workspace_declares_sequential_alt_save_load_fn_names_only_in_allowed_locations() {
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
            let offenses = scan_sequential_alt_save_load_impls(content);
            if !offenses.is_empty() {
                let rel = path
                    .strip_prefix(&crates_dir)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .replace('\\', "/");
                *found.entry(rel).or_insert(0) += offenses.len();
            }
        });
    }

    let expected: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の `impl Sequential {{ .. }}`／\
         `impl <Trait> for Sequential {{ .. }}` 内 fn save／fn load 宣言\
         集合が期待（空集合）と一致しない（承認前に代替 API\
         〈`Sequential::save`／`Sequential::load`〉の実装が facade 外へ\
         紛れ込んだ、または facade 側で追加された可能性がある。過不足\
         いずれも fail-closed に検出する）: {found:?}"
    );
}

/// `crates/facade/src/lib.rs` 内の全 hold ガード doctest（`mod
/// __fandhe_*_hold_probe { ... }` を `use <mod>::*;` で glob import する
/// 正のプローブ方式のブロック）を横断走査し、各プローブモジュール内で
/// `pub` 定義された名前（trait・struct・fn・enum・type・mod・const・
/// static・use）が、その glob import 行より後ろの doctest 本文で
/// 少なくとも 1 回**参照**されていることを検査する（イシュー #2304 の
/// codex レビュー指摘: `OptimizerStateDictHoldDoctestGuard` の
/// `OptimizerStateDict` トレイトが glob import されるだけで一度も
/// 名前解決されず、facade がこの名前を再エクスポートしても glob 衝突
/// 〈E0659〉は「その名前を実際に使ったときにだけ」発生するため検出
/// できなかった欠陥。`ParamGroupsHoldDoctestGuard` の `ParamGroupStep`
/// トレイトにも同型の欠陥が既存で存在した。本テストは同型の欠陥の
/// 再発を機械的に防ぐ再発防止ガードであり、個々のガードの固定文言
/// 契約〈`*_HOLD_PROBE_BODY` 系〉とは独立の横断監査を担う）。
///
/// 走査ロジックは `pub` 宣言行の単純な字句マッチと、doctest 本文の
/// 識別子トークン化による部分文字列検査で行う（本テストは workspace
/// 許容依存 8 区分に含まれない正規表現クレートを使わず、手書きの
/// 字句走査で完結させる。deps-policy.md）。
#[test]
fn hold_doctest_probe_blocks_reference_every_glob_imported_item() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let audits = scan_hold_probe_blocks(&content);

    // 正のプローブ: 走査対象が空振りで通過するのを防ぐため、検出した
    // プローブブロック数が既知の下限（2026-09-26 時点の実測値 32 から、#2505・#2508・#2511・#2512・#2513・#2514・#2515・#2517・#2518・#2519・#2530・#2533 の保留ガード削除後は 20）以上
    // であることを固定する。将来ブロックが追加された場合はこの下限を
    // 上方修正する（削減時は本テストが個別に指摘する）。
    const MIN_KNOWN_PROBE_BLOCKS: usize = 20;
    assert!(
        audits.len() >= MIN_KNOWN_PROBE_BLOCKS,
        "hold ガード doctest のプローブモジュール検出数が既知の下限を\
         下回っている（走査ロジック自体が壊れ空振りで通過している疑いが\
         ある）: 検出数={}, 下限={MIN_KNOWN_PROBE_BLOCKS}",
        audits.len()
    );

    // 全数照合: フェンス解析を介さない独立集計（`///` 行の単純な
    // 行走査のみで `mod __fandhe_..._hold_probe {` を数える）と
    // `audits.len()` が完全一致することを検査する。フェンス解析側
    // （`scan_hold_probe_blocks_in_doc_run`／`scan_hold_probe_blocks_in_body`）
    // に将来 quad-fence 誤検出のような飲み込みバグが再発しても、
    // 件数の不一致として機械的に検出できるようにする（Cursor Bugbot
    // 指摘・PR #2304。下限定数 `MIN_KNOWN_PROBE_BLOCKS` だけでは
    // 「一部が飲み込まれても下限を上回る」ケースを見逃すため）。
    let independent_count = count_hold_probe_mod_declarations_in_doc_comments(&content);
    assert_eq!(
        audits.len(),
        independent_count,
        "フェンス解析による走査件数（{}）と、フェンス解析を介さない\
         独立集計（{independent_count}）が一致しない（フェンス走査が\
         一部の `mod __fandhe_..._hold_probe {{ ... }}` を飲み込んで\
         見落としている疑いがある）",
        audits.len()
    );

    // 正のプローブ: 本テストが検出対象に含めるべき既知の 2 例
    // （イシュー #2304 で修正した欠陥そのもの）が走査集合に含まれる
    // ことを固定する。
    let mod_names: std::collections::BTreeSet<&str> =
        audits.iter().map(|a| a.mod_name.as_str()).collect();
    for expected in [
        "__fandhe_param_groups_hold_probe",
        "__fandhe_optim_state_dict_hold_probe",
    ] {
        assert!(
            mod_names.contains(expected),
            "既知のプローブモジュール `{expected}` が走査対象に含まれて\
             いない（走査ロジックのフェンス検出・mod 境界検出が壊れて\
             いる疑いがある）"
        );
    }

    let mut offenses: Vec<String> = Vec::new();
    for audit in &audits {
        for item in &audit.unreferenced_items {
            offenses.push(format!(
                "{}::{item}（glob import 後の doctest 本文で一度も\
                 参照されていない。この名前は facade が公開しても glob\
                 衝突を起こさず、保留固定として機能しない）",
                audit.mod_name
            ));
        }
    }
    assert!(
        offenses.is_empty(),
        "hold ガード doctest のプローブモジュールに、glob import 後の\
         本文で一度も参照されない `pub` 定義が存在する（明示的に参照する\
         プローブ〈関数境界での型使用・`fn __probe_trait<T: ?Sized +\
         Trait>() {{}}` 等〉を追加すること）: {offenses:?}"
    );
}

/// [`hold_doctest_probe_blocks_reference_every_glob_imported_item`] の
/// 走査結果 1 件（1 プローブモジュール分）。
struct HoldProbeBlockAudit {
    mod_name: String,
    unreferenced_items: Vec<String>,
}

/// [`scan_hold_probe_blocks`] の全数照合用の独立集計。`lib.rs` の
/// 全文（`content`）から `///` 行だけを単純に走査し、フェンス解析を
/// 一切介さずに `mod __fandhe_..._hold_probe { ... }` 定義行の個数を
/// 数える。フェンス解析（`scan_hold_probe_blocks_in_doc_run`）が
/// quad-fence 誤検出等で一部の doctest を飲み込んでも、本関数は
/// フェンス構造に依存しないためその影響を受けず、両者の件数比較で
/// 飲み込みバグを検出できる（Cursor Bugbot 指摘・PR #2304）。
fn count_hold_probe_mod_declarations_in_doc_comments(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let Some(raw) = line.trim_start().strip_prefix("///") else {
                return false;
            };
            let raw = raw.strip_prefix(' ').unwrap_or(raw);
            let trimmed = raw.trim();
            trimmed.starts_with("mod __fandhe")
                && trimmed.ends_with("_hold_probe {")
                && !trimmed.starts_with("pub mod")
        })
        .count()
}

/// `lib.rs` の全文（`content`）から、`///` doc コメントの連続領域に
/// 現れる裸／タグ付きフェンスの doctest ブロックを走査し、各ブロック
/// 内の `mod __fandhe_..._hold_probe { ... }` 定義 1 つにつき
/// [`HoldProbeBlockAudit`] を 1 件生成する。
fn scan_hold_probe_blocks(content: &str) -> Vec<HoldProbeBlockAudit> {
    let mut audits = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0usize;
    while i < lines.len() {
        if !lines[i].trim_start().starts_with("///") {
            i += 1;
            continue;
        }
        // 連続する `///` 行 1 ラン分を doc テキストへ変換する。
        let mut doc_lines: Vec<String> = Vec::new();
        while i < lines.len() && lines[i].trim_start().starts_with("///") {
            let raw = lines[i].trim_start();
            let rest = raw.strip_prefix("///").unwrap_or(raw);
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            doc_lines.push(rest.to_string());
            i += 1;
        }
        audits.extend(scan_hold_probe_blocks_in_doc_run(&doc_lines));
    }
    audits
}

/// [`scan_hold_probe_blocks`] が抽出した doc テキスト 1 ラン分から、
/// フェンス区切りの doctest ブロックを抜き出し、各ブロック内の
/// `mod __fandhe_..._hold_probe { ... }` を監査する。
fn scan_hold_probe_blocks_in_doc_run(doc_lines: &[String]) -> Vec<HoldProbeBlockAudit> {
    let mut audits = Vec::new();
    let mut i = 0usize;
    while i < doc_lines.len() {
        let trimmed = doc_lines[i].trim_end();
        // フェンス開始行の判定は「先頭の連続バッククォート数がちょうど
        // 3」の場合に限る（extract_single_bare_fenced_doctest_block と
        // 同じ判定基準に統一。Cursor Bugbot 指摘・PR #2304）。lib.rs の
        // 地の文には quad-fence 引用記法（4 連続バッククォートで
        // フェンス表記そのものをインライン引用する書き方。例:
        // ```` ```compile_fail,E0599 ```` が地の文の 1 行に現れる形）が
        // あり、「3 以上」で判定すると地の文を誤ってフェンス開始と
        // 誤検出し、以降の doctest を丸ごと読み飛ばして監査から
        // 漏らしてしまう（Cursor Bugbot 指摘・PR #2304。修正前はこの誤検出
        // により後続プローブモジュールが監査対象から静かに脱落し
        // うる構造上の欠陥だった）。
        let leading_backticks = trimmed
            .trim_start()
            .chars()
            .take_while(|&c| c == '`')
            .count();
        if leading_backticks != 3 {
            i += 1;
            continue;
        }
        // フェンス開始行を見つけた。閉じフェンス（前後トリム後
        // "```"）まで本文を収集する。
        let mut body: Vec<String> = Vec::new();
        i += 1;
        while i < doc_lines.len() && doc_lines[i].trim() != "```" {
            body.push(doc_lines[i].clone());
            i += 1;
        }
        // fail-closed: 閉じフェンスが見つからないまま doc ラン終端に
        // 達した場合は、黙って終端扱いにせず panic する。閉じ忘れの
        // まま本文欠落（body が途中で打ち切られる）を通過させると、
        // フェンス内のプローブモジュールが不完全な形で監査され、
        // 検出漏れを見逃す可能性があるため。
        assert!(
            i < doc_lines.len(),
            "hold プローブ走査: doctest フェンスが閉じられていない\
             （doc ラン終端に達した）。開始行付近の本文: {body:?}"
        );
        // 閉じフェンス自体を読み飛ばす。
        i += 1;
        audits.extend(scan_hold_probe_blocks_in_body(&body));
    }
    audits
}

/// doctest ブロック本文（フェンスを含まない）から
/// `mod __fandhe_..._hold_probe { ... }` 定義を探し、`pub` 定義された
/// 名前が対応する `use <mod>::*;` 行より後ろで参照されているかを判定
/// する。1 ブロックに複数のプローブモジュール定義があっても全て拾う。
fn scan_hold_probe_blocks_in_body(body: &[String]) -> Vec<HoldProbeBlockAudit> {
    let mut audits = Vec::new();
    let mut i = 0usize;
    while i < body.len() {
        let trimmed = body[i].trim();
        let is_probe_mod_start = trimmed.starts_with("mod __fandhe")
            && trimmed.ends_with("_hold_probe {")
            && !trimmed.starts_with("pub mod");
        if !is_probe_mod_start {
            i += 1;
            continue;
        }
        let mod_name = trimmed
            .strip_prefix("mod ")
            .and_then(|rest| rest.strip_suffix(" {"))
            .unwrap_or_default()
            .to_string();

        // ブレース深さカウントで対応する閉じ行を探す（ネストした
        // `pub mod` を含んでも壊れないよう、単純な文字列一致ではなく
        // 深さで判定する）。
        let mut depth: i32 = 1;
        let mod_body_start = i + 1;
        let mut mod_end = body.len();
        let mut j = mod_body_start;
        while j < body.len() {
            let opens = body[j].matches('{').count() as i32;
            let closes = body[j].matches('}').count() as i32;
            depth += opens - closes;
            if depth <= 0 {
                mod_end = j;
                break;
            }
            j += 1;
        }

        let mut items: Vec<String> = Vec::new();
        for line in &body[mod_body_start..mod_end] {
            let t = line.trim();
            let Some(rest) = t.strip_prefix("pub ") else {
                continue;
            };
            let mut tokens = rest.split_whitespace();
            let Some(keyword) = tokens.next() else {
                continue;
            };
            if !matches!(
                keyword,
                "trait" | "struct" | "fn" | "enum" | "type" | "mod" | "const" | "static" | "use"
            ) {
                continue;
            }
            let Some(name_raw) = tokens.next() else {
                continue;
            };
            let name: String = name_raw
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                items.push(name);
            }
        }

        // `use <mod_name>::*;` 行（glob import）をブロック全体から探す。
        let use_line = format!("use {mod_name}::*;");
        let use_idx = body.iter().position(|l| l.trim() == use_line);
        let rest_tokens: std::collections::HashSet<String> = match use_idx {
            Some(idx) => tokenize_identifiers(&body[(idx + 1)..].join("\n")),
            None => std::collections::HashSet::new(),
        };

        let unreferenced_items: Vec<String> = items
            .into_iter()
            .filter(|name| !rest_tokens.contains(name))
            .collect();

        audits.push(HoldProbeBlockAudit {
            mod_name,
            unreferenced_items,
        });

        i = mod_end + 1;
    }
    audits
}

/// `text` を識別子トークン（英数字・アンダースコアの連続runs）へ分割
/// した集合を返す（[`scan_hold_probe_blocks_in_body`] の参照検査用）。
fn tokenize_identifiers(text: &str) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            current.push(c);
        } else if !current.is_empty() {
            out.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.insert(current);
    }
    out
}

/// 回帰テスト（PR #2304 Bugbot 指摘）: `scan_hold_probe_blocks_in_doc_run`
/// が、quad-fence 引用記法（4 連続バッククォートでフェンス表記自体を
/// インライン引用する地の文行）をフェンス開始と誤検出せず、その直後に
/// 続く裸フェンスの hold プローブ doctest を正しく監査対象に含めること
/// を確認する。修正前（「先頭バッククォート数が 3 以上」で判定する旧
/// ロジック）では、quad-fence 引用行が誤ってフェンス開始とみなされ、
/// 直後に現れる本物の裸フェンス開始行がその誤検出フェンスの「閉じ」と
/// して消費されてしまい、後続の裸フェンス doctest ブロック
/// （本テストの `__fandhe_regression_hold_probe`）が丸ごと走査から
/// 脱落する（`scan_hold_probe_blocks_in_doc_run` が 0 件しか返さない）。
#[test]
fn scan_hold_probe_blocks_in_doc_run_survives_quad_fence_quotation_in_prose() {
    // `doc_lines` は `scan_hold_probe_blocks` が `///` プレフィックスを
    // 剥がした後の doc テキスト行相当（本テストはフェンス解析単体を
    // 検査するため、`///` 剥がし処理を経由せず直接構築する）。
    let doc_lines: Vec<String> = [
        "地の文の説明。旧実装は 3 本の",
        "```` ```compile_fail,E0599 ```` doctest ブロックだった。しかし",
        "quad-fence 引用がここに現れる（本物のフェンス開始ではない）。",
        "```",
        "mod __fandhe_regression_hold_probe {",
        "    pub trait RegressionProbeTrait {}",
        "}",
        "use __fandhe_regression_hold_probe::*;",
        "fn __probe_unrelated() {}",
        "```",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    let audits = scan_hold_probe_blocks_in_doc_run(&doc_lines);

    assert_eq!(
        audits.len(),
        1,
        "quad-fence 引用行の直後にある裸フェンスの hold プローブが\
         走査から脱落している（quad-fence 誤検出の再発）: {}",
        audits.len()
    );
    assert_eq!(audits[0].mod_name, "__fandhe_regression_hold_probe");
    // `RegressionProbeTrait` は glob import 後の本文で一度も参照されて
    // いないため、未参照として検出されるはずである（プローブとして
    // 機能していることの確認）。
    assert_eq!(
        audits[0].unreferenced_items,
        vec!["RegressionProbeTrait".to_string()],
        "quad-fence 誤検出により本文の取り込み範囲がずれ、未参照判定が\
         想定と異なる結果になっている"
    );
}

/// 回帰テスト（PR #2304）: 裸フェンスが閉じられないまま doc ラン終端に
/// 達した場合、`scan_hold_probe_blocks_in_doc_run` は黙って打ち切らず
/// fail-closed に panic することを確認する。
#[test]
#[should_panic(expected = "doctest フェンスが閉じられていない")]
fn scan_hold_probe_blocks_in_doc_run_panics_on_unclosed_fence() {
    let doc_lines: Vec<String> = [
        "```",
        "mod __fandhe_unclosed_hold_probe {",
        "    pub trait UnclosedProbeTrait {}",
        "}",
        "use __fandhe_unclosed_hold_probe::*;",
        // 閉じフェンス "```" を意図的に省略する。
    ]
    .into_iter()
    .map(str::to_string)
    .collect();

    let _ = scan_hold_probe_blocks_in_doc_run(&doc_lines);
}
// =====================================================================
// NpyIoHoldDoctestGuard（イシュー #2189・親 #2131）: `RngDistributionsHold
// DoctestGuard`（#2156）系のテスト（`rng_distributions_hold_doctest_
// globs_all_pub_modules`／`rng_distributions_hold_doctest_probe_body_
// matches_fixed_contract`／`facade_does_not_reexport_or_declare_rng_
// distributions`／`workspace_declares_rng_distribution_names_only_in_
// allowed_locations`）を鏡写しにする。
// =====================================================================

/// `crates/facade/src/lib.rs` の `NpyIoHoldDoctestGuard` doc 内の唯一の
/// doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （`rng_distributions_hold_doctest_globs_all_pub_modules` と同型）。
#[test]
fn npy_io_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "NpyIoHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "NpyIoHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`npy_io_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの**glob 以外
/// の本文**が固定文言 [`NPY_IO_HOLD_PROBE_BODY`] と 1 行たりとも違わず
/// 一致することを固定する（`rng_distributions_hold_doctest_probe_body_
/// matches_fixed_contract` と同じ理由: rustdoc の `# ` 隠し行・プローブの
/// 削除・別名へのシャドーイング等で正のプローブを骨抜きにする改変を
/// 機械的に拒否する）。
#[test]
fn npy_io_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "NpyIoHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, NPY_IO_HOLD_PROBE_BODY,
        "NpyIoHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 NPY_IO_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_npy_io_hold_probe モジュール・\
         __FandheNpyIoHoldProbe トレイト・__probe_* 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`npy_io_hold_doctest_probe_body_matches_fixed_contract`] が要求する
/// 固定文言。`crates/facade/src/lib.rs` の `NpyIoHoldDoctestGuard` doc 内
/// の唯一の doctest ブロックから、ネスト `pub mod` の glob import 行
/// （`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致する
/// 必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に含む）。
const NPY_IO_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_npy_io_hold_probe {\n\
\x20\x20\x20\x20pub struct NpyError;\n\
\x20\x20\x20\x20pub fn load_npy() {}\n\
\x20\x20\x20\x20pub fn save_npy() {}\n\
\x20\x20\x20\x20pub fn load_npz() {}\n\
\x20\x20\x20\x20pub fn save_npz() {}\n\
\x20\x20\x20\x20pub mod npy {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn __mark() {}\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20pub mod npz {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn __mark() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_npy_io_hold_probe::*;\n\
\n\
struct __FandheNpyIoHoldMarker;\n\
\n\
trait __FandheNpyIoHoldProbe {\n\
\x20\x20\x20\x20fn load_npy(&self) -> __FandheNpyIoHoldMarker;\n\
\x20\x20\x20\x20fn save_npy(&self) -> __FandheNpyIoHoldMarker;\n\
\x20\x20\x20\x20fn load_npz(&self) -> __FandheNpyIoHoldMarker;\n\
\x20\x20\x20\x20fn save_npz(&self) -> __FandheNpyIoHoldMarker;\n\
}\n\
\n\
impl __FandheNpyIoHoldProbe for fandhe_ai::Tensor<f32> {\n\
\x20\x20\x20\x20fn load_npy(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }\n\
\x20\x20\x20\x20fn save_npy(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }\n\
\x20\x20\x20\x20fn load_npz(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }\n\
\x20\x20\x20\x20fn save_npz(&self) -> __FandheNpyIoHoldMarker { __FandheNpyIoHoldMarker }\n\
}\n\
\n\
fn __probe_free_fns(_: NpyError) {\n\
\x20\x20\x20\x20// 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して\n\
\x20\x20\x20\x20// いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。\n\
\x20\x20\x20\x20load_npy();\n\
\x20\x20\x20\x20save_npy();\n\
\x20\x20\x20\x20load_npz();\n\
\x20\x20\x20\x20save_npz();\n\
\x20\x20\x20\x20npy::__mark();\n\
\x20\x20\x20\x20npz::__mark();\n\
}\n\
\n\
fn __probe_tensor(x: &fandhe_ai::Tensor<f32>) {\n\
\x20\x20\x20\x20let _: __FandheNpyIoHoldMarker = fandhe_ai::Tensor::load_npy(x);\n\
\x20\x20\x20\x20let _: __FandheNpyIoHoldMarker = x.save_npy();\n\
\x20\x20\x20\x20let _: __FandheNpyIoHoldMarker = fandhe_ai::Tensor::load_npz(x);\n\
\x20\x20\x20\x20let _: __FandheNpyIoHoldMarker = x.save_npz();\n\
}";

/// fn 名 4 個（イシュー #2189）。[`scan_npy_io_reexports_and_declarations`]・
/// [`workspace_declares_npy_io_names_only_in_allowed_locations`] が共用
/// する。
const NPY_IO_FN_NAMES: [&str; 4] = ["load_npy", "save_npy", "load_npz", "save_npz"];

/// [`facade_does_not_reexport_or_declare_npy_io`]・その自己テストが共用
/// する検出本体。facade src 全体（`crates/facade/src/**`）の `pub use`
/// から [`collect_pub_use_leaves`] で別名にする前の葉を集め `NpyError`
/// を検出し（単一行・複数行・ネストした group・別名も検出）、
/// `trait`／`struct`／`enum`／`type` 直後の `NpyError` 独自宣言、
/// [`NPY_IO_FN_NAMES`]（4 個）の `fn` 宣言（可視性・宣言文脈を問わない。
/// [`count_fn_declarations_by_name`] と同じ検出契約）を違反として返す
/// （`scan_rng_distributions_reexports_and_declarations` と同型）。
fn scan_npy_io_reexports_and_declarations(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if leaf == "NpyError" {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens.get(i + 1).map(String::as_str) == Some("NpyError")
        {
            offending.push(format!("{} NpyError 宣言", tokens[i]));
        }
        i += 1;
    }

    for fn_name in NPY_IO_FN_NAMES {
        let count = count_fn_declarations_by_name(&tokens, fn_name);
        if count > 0 {
            offending.push(format!("`fn {fn_name}` 宣言が {count} 件"));
        }
    }
    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`NpyError` を識別子単位
/// で含む `pub use`（複数行・ネストした group・別名含む）も、facade
/// 独自の `trait`／`struct`／`enum`／`type` 宣言も、[`NPY_IO_FN_NAMES`]
/// （`load_npy`／`save_npy`／`load_npz`／`save_npz`）の `fn` 宣言も
/// 存在しないことを固定する（`NpyIoHoldDoctestGuard` の正のプローブと
/// 多層防御を成す最内層のソース走査ガード。
/// `facade_does_not_reexport_or_declare_rng_distributions` と同型）。
#[test]
fn facade_does_not_reexport_or_declare_npy_io() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_npy_io_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が npy/npz 読み書き（#2189 の `load_npy`／\
         `save_npy`／`load_npz`／`save_npz`／`NpyError`。内部クレート限定の\
         新規公開面。facade 公開は承認待ちのため対象外という設計判断に\
         違反）を再エクスポート、独自宣言、または同名の fn を宣言している: \
         {offending:?}"
    );
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、[`NPY_IO_FN_NAMES`]
/// （4 個）の `fn` 宣言の定義元集合を固定する（`workspace_declares_rng_
/// distribution_names_only_in_allowed_locations` と同型のインベントリ）。
///
/// **期待集合**（実装後の実測で `crates/tensor-core/src/io/npy.rs` に
/// `load_npy`／`save_npy`、`crates/tensor-core/src/io/npz.rs` に
/// `load_npz`／`save_npz` が各 1 件ずつ追加された）:
/// - `tensor-core/src/io/npy.rs::load_npy` = 1
/// - `tensor-core/src/io/npy.rs::save_npy` = 1
/// - `tensor-core/src/io/npz.rs::load_npz` = 1
/// - `tensor-core/src/io/npz.rs::save_npz` = 1
#[test]
fn workspace_declares_npy_io_names_only_in_allowed_locations() {
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
            for fn_name in NPY_IO_FN_NAMES {
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

    let expected: std::collections::BTreeMap<String, usize> = [
        ("tensor-core/src/io/npy.rs::load_npy".to_string(), 1usize),
        ("tensor-core/src/io/npy.rs::save_npy".to_string(), 1usize),
        ("tensor-core/src/io/npz.rs::load_npz".to_string(), 1usize),
        ("tensor-core/src/io/npz.rs::save_npz".to_string(), 1usize),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        found, expected,
        "workspace 全体（crates/*/src/）の load_npy／save_npy／load_npz／\
         save_npz `fn` 宣言集合が期待（tensor-core/src/io/npy.rs・\
         tensor-core/src/io/npz.rs に各 1 件ずつ）と一致しない（過不足\
         いずれも fail-closed に検出する。新たな定義元が見つかった場合、\
         それが承認済みの実装なのか迂回経路の混入なのかを確認すること）: \
         {found:?}"
    );
}

// --- generate()（イシュー #2191）の facade 公開保留固定 ---------------

/// `generate()` 自己回帰ループ（イシュー #2191。設計正本
/// `docs/facade-generate-decision.md`）の facade 公開（`inference::
/// generate`／`GenerateConfig`／`SamplingStrategy`／`AutoregressiveModel`
/// 相当）は未承認のため保留する。`facade_does_not_expose_kv_cache_
/// stateful_attention` と同型の否定ガード: facade の src/ に①`fn
/// generate` 宣言（可視性・宣言文脈を問わず。[`declares_fn_named`]
/// 参照）、②`GenerateConfig`／`SamplingStrategy`／
/// `AutoregressiveModel` を識別子単位で含む `pub use` 行、のいずれも
/// 存在しないことを固定する。承認取得後に薄い委譲 `pub fn`／
/// 再エクスポートを追加する際は本テストを正ガードへ更新すること。
///
/// 本テストは多層防御の最内層（1 行単位の直列走査）であり、`src/lib.rs`
/// の `GenerateHoldDoctestGuard`（正のプローブ doctest）・
/// `facade_does_not_reexport_or_declare_generate_items`（トークン方式。
/// 複数行・別名・独自宣言を検出）と多層で保留を固定する
/// （`docs/facade-generate-decision.md` §8）。workspace 全体（facade
/// 以外のクレート内部の private 宣言も含む）の名前インベントリは、
/// facade 到達可能性の保証と無関係な内部宣言まで固定してしまうため
/// 採用しない（KvCache §10.1 の codex-review 指摘・PR #2252 と同じ理由。
/// `crates/self-repair` に `fn generate` トレイトメソッドが複数ある）。
#[test]
fn facade_does_not_expose_generate_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        if declares_fn_named(content, "generate") {
            offending.push(format!("{}: `fn generate` 宣言", path.display()));
        }
        for line in content.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("pub use") {
                continue;
            }
            for ident in ["GenerateConfig", "SamplingStrategy", "AutoregressiveModel"] {
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
        "facade の公開面が generate()（#2191。`generate`／`GenerateConfig`／\
         `SamplingStrategy`／`AutoregressiveModel`）を公開している\
         （`docs/facade-generate-decision.md` §8 が未取得のまま対象外と\
         している設計判断に違反）: {offending:?}"
    );
}

/// `GenerateHoldDoctestGuard` の唯一の doctest ブロックが glob import する
/// ネスト `pub mod` 集合と、`src/lib.rs` の実際の `pub mod` 宣言集合が
/// 一致することを固定する（`kv_cache_hold_doctest_globs_all_pub_modules`
/// と同型）。
#[test]
fn generate_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "GenerateHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "GenerateHoldDoctestGuard の doctest ブロックが glob import する\
         モジュール集合が src/lib.rs の pub mod 宣言集合とドリフトしている\
         （declared={declared:?}, doctest={globbed:?}）。新しい pub mod を\
         追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`generate_hold_doctest_globs_all_pub_modules`] が glob import 集合の
/// 一致のみを固定するのに対し、本テストは doctest ブロックの **glob
/// 以外の本文**が固定文言 [`GENERATE_HOLD_PROBE_BODY`] と 1 行たりとも
/// 違わず一致することを固定する（rustdoc の `# ` 隠し行・プローブの削除・
/// 別名へのシャドーイング等で正のプローブを骨抜きにする改変を機械的に
/// 拒否する）。
#[test]
fn generate_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "GenerateHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, GENERATE_HOLD_PROBE_BODY,
        "GenerateHoldDoctestGuard の doctest ブロック本文（glob 以外）が\
         固定文言 GENERATE_HOLD_PROBE_BODY からドリフトしている。正の\
         プローブ（__fandhe_generate_hold_probe モジュール・\
         __FandheGenerateHoldProbe トレイト・各 __probe_* 関数）の削除・\
         弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`generate_hold_doctest_probe_body_matches_fixed_contract`] が要求
/// する固定文言。`crates/facade/src/lib.rs` の `GenerateHoldDoctestGuard`
/// doc 内の唯一の doctest ブロックから、ネスト `pub mod` の glob import
/// 行（`use fandhe_ai::<mod>::*;`）を除いた本文と 1 行単位で完全一致
/// する必要がある（クレートルート自体の `use fandhe_ai::*;` は本文に
/// 含む）。
const GENERATE_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_generate_hold_probe {\n\
\x20\x20\x20\x20pub mod inference {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn generate() {}\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub struct GenerateConfig;\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub struct SamplingStrategy;\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub struct AutoregressiveModel;\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20pub fn generate() {}\n\
\x20\x20\x20\x20pub struct GenerateConfig;\n\
\x20\x20\x20\x20pub struct SamplingStrategy;\n\
\x20\x20\x20\x20pub struct AutoregressiveModel;\n\
}\n\
use __fandhe_generate_hold_probe::*;\n\
\n\
struct __FandheGenerateHoldMarker;\n\
\n\
trait __FandheGenerateHoldProbe {\n\
\x20\x20\x20\x20fn generate(&self) -> __FandheGenerateHoldMarker;\n\
}\n\
\n\
impl __FandheGenerateHoldProbe for fandhe_ai::Tape {\n\
\x20\x20\x20\x20fn generate(&self) -> __FandheGenerateHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheGenerateHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
impl __FandheGenerateHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn generate(&self) -> __FandheGenerateHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandheGenerateHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_module_path() {\n\
\x20\x20\x20\x20inference::generate();\n\
\x20\x20\x20\x20let _ = inference::GenerateConfig;\n\
\x20\x20\x20\x20let _ = inference::SamplingStrategy;\n\
\x20\x20\x20\x20let _ = inference::AutoregressiveModel;\n\
}\n\
\n\
fn __probe_free_fn() {\n\
\x20\x20\x20\x20generate();\n\
\x20\x20\x20\x20let _ = GenerateConfig;\n\
\x20\x20\x20\x20let _ = SamplingStrategy;\n\
\x20\x20\x20\x20let _ = AutoregressiveModel;\n\
}\n\
\n\
fn __probe_inherent_method(tape: &fandhe_ai::Tape, seq: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandheGenerateHoldMarker = fandhe_ai::Tape::generate(tape);\n\
\x20\x20\x20\x20let _: __FandheGenerateHoldMarker = fandhe_ai::compat::Sequential::generate(seq);\n\
\x20\x20\x20\x20let _: __FandheGenerateHoldMarker = tape.generate();\n\
\x20\x20\x20\x20let _: __FandheGenerateHoldMarker = seq.generate();\n\
}";

/// [`facade_does_not_reexport_or_declare_generate_items`]・その自己
/// テストが共用する検出本体。facade src 全体（`crates/facade/src/**`）
/// の `pub use` から [`collect_pub_use_leaves`] で別名にする前の葉を
/// 集め `GenerateConfig`／`SamplingStrategy`／`AutoregressiveModel` を
/// 検出し（単一行・複数行・ネストした group・別名も検出）、`trait`／
/// `struct`／`enum`／`type` 直後の同名独自宣言、`generate` の `fn` 宣言
/// （可視性・宣言文脈を問わない）を違反として返す。
fn scan_generate_reexports_and_declarations(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if matches!(
                    leaf.as_str(),
                    "GenerateConfig" | "SamplingStrategy" | "AutoregressiveModel"
                ) {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && matches!(
                tokens.get(i + 1).map(String::as_str),
                Some("GenerateConfig") | Some("SamplingStrategy") | Some("AutoregressiveModel")
            )
        {
            offending.push(format!(
                "{} {} 宣言",
                tokens[i],
                tokens.get(i + 1).map(String::as_str).unwrap_or_default()
            ));
        }
        i += 1;
    }

    let count = count_fn_declarations_by_name(&tokens, "generate");
    if count > 0 {
        offending.push(format!("`fn generate` 宣言が {count} 件"));
    }
    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`GenerateConfig`／
/// `SamplingStrategy`／`AutoregressiveModel` を識別子単位で含む
/// `pub use`（複数行・ネストした group・別名含む）も、facade 独自の
/// `trait`／`struct`／`enum`／`type` 宣言も、`generate` の `fn` 宣言も
/// 存在しないことを固定する（`GenerateHoldDoctestGuard` の正のプローブと
/// 多層防御を成す最内層のソース走査ガード。
/// `facade_does_not_reexport_or_declare_kv_cache_items` と同型で、既存の
/// 1 行単位走査 `facade_does_not_expose_generate_items` の穴〈複数行
/// `pub use`・facade 独自宣言〉を塞ぐ）。
#[test]
fn facade_does_not_reexport_or_declare_generate_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_generate_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が generate()（#2191。`GenerateConfig`／\
         `SamplingStrategy`／`AutoregressiveModel`／`generate`）を\
         再エクスポート、独自宣言、または同名の fn を宣言している\
         （`docs/facade-generate-decision.md` §8 が未取得のまま対象外と\
         している設計判断に違反）: {offending:?}"
    );
}

/// [`scan_generate_reexports_and_declarations`]（[`facade_does_not_
/// reexport_or_declare_generate_items`]）の自己テスト（正例・負例の
/// 合成入力）。単一行・複数行・別名・独自宣言・fn 宣言の各正例と、
/// コメントや文字列リテラル中の出現・非公開 `use` の負例を検証する。
#[test]
fn facade_does_not_reexport_or_declare_generate_items_detects_each_category() {
    // 正例: 単一行 pub use。
    assert!(
        !scan_generate_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::generate::GenerateConfig;"
        )
        .is_empty()
    );
    // 正例: 複数行 pub use（複数行グループ再エクスポートの穴）。
    assert!(
        !scan_generate_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::generate::{\n    SamplingStrategy,\n};"
        )
        .is_empty()
    );
    // 正例: ネストした group・別名。
    assert!(
        !scan_generate_reexports_and_declarations(
            "pub use fandhe_ai_autodiff::{generate::{AutoregressiveModel as ARM}};"
        )
        .is_empty()
    );
    // 正例: facade 独自宣言。
    assert!(!scan_generate_reexports_and_declarations("pub struct GenerateConfig;").is_empty());
    assert!(
        !scan_generate_reexports_and_declarations("pub type SamplingStrategy = u8;").is_empty()
    );
    // 正例: fn 宣言（可視性を問わない）。
    assert!(!scan_generate_reexports_and_declarations("pub fn generate() -> u8 { 0 }").is_empty());

    // 負例: 非公開 import。
    assert!(
        scan_generate_reexports_and_declarations(
            "use fandhe_ai_autodiff::generate::GenerateConfig;"
        )
        .is_empty()
    );
    // 負例: コメント・文字列リテラル中の出現。
    assert!(
        scan_generate_reexports_and_declarations(
            "// pub use fandhe_ai_autodiff::generate::GenerateConfig;\n\
             let s = \"GenerateConfig\";"
        )
        .is_empty()
    );
}

// =====================================================================
// PredictBatchesHoldDoctestGuard（イシュー #2192・親 #2131）:
// `NpyIoHoldDoctestGuard`（#2189）系のテスト（`npy_io_hold_doctest_
// globs_all_pub_modules`／`npy_io_hold_doctest_probe_body_matches_fixed_
// contract`／`facade_does_not_reexport_or_declare_npy_io`／
// `workspace_declares_npy_io_names_only_in_allowed_locations`）を鏡写し
// にする。
// =====================================================================

/// `crates/facade/src/lib.rs` の `PredictBatchesHoldDoctestGuard` doc 内
/// の唯一の doctest ブロックが glob import するネスト `pub mod` 集合と、
/// `src/lib.rs` の実際の `pub mod` 宣言集合が一致することを固定する
/// （`npy_io_hold_doctest_globs_all_pub_modules` と同型）。
#[test]
fn predict_batches_hold_doctest_globs_all_pub_modules() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let doc_lines = extract_hold_doctest_guard_doc(&content, "PredictBatchesHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (globbed, _body) = split_glob_imports_and_probe_body(&block);
    assert!(
        !declared.is_empty(),
        "src/lib.rs から pub mod 宣言を 1 件も抽出できなかった\
         （テスト自体が検査対象を見失っている可能性がある）"
    );
    assert_eq!(
        declared, globbed,
        "PredictBatchesHoldDoctestGuard の doctest ブロックが glob import\
         するモジュール集合が src/lib.rs の pub mod 宣言集合とドリフト\
         している（declared={declared:?}, doctest={globbed:?}）。新しい\
         pub mod を追加した場合は doctest 側の use 一覧にも追加すること。"
    );
}

/// [`predict_batches_hold_doctest_globs_all_pub_modules`] が glob import
/// 集合の一致のみを固定するのに対し、本テストは doctest ブロックの
/// **glob 以外の本文**が固定文言 [`PREDICT_BATCHES_HOLD_PROBE_BODY`] と
/// 1 行たりとも違わず一致することを固定する（`npy_io_hold_doctest_probe_
/// body_matches_fixed_contract` と同じ理由: rustdoc の `# ` 隠し行・
/// プローブの削除・別名へのシャドーイング等で正のプローブを骨抜きにする
/// 改変を機械的に拒否する）。
#[test]
fn predict_batches_hold_doctest_probe_body_matches_fixed_contract() {
    let content = read_to_string_or_panic(&lib_rs_path());
    let doc_lines = extract_hold_doctest_guard_doc(&content, "PredictBatchesHoldDoctestGuard");
    let block = extract_single_bare_fenced_doctest_block(&doc_lines);
    let (_globbed, body) = split_glob_imports_and_probe_body(&block);
    let actual = body.join("\n");
    assert_eq!(
        actual, PREDICT_BATCHES_HOLD_PROBE_BODY,
        "PredictBatchesHoldDoctestGuard の doctest ブロック本文（glob 以外）\
         が固定文言 PREDICT_BATCHES_HOLD_PROBE_BODY からドリフトしている。\
         正のプローブ（__fandhe_predict_batches_hold_probe モジュール・\
         __FandhePredictBatchesHoldProbe トレイト・__probe_* 関数）の\
         削除・弱体化・隠し行の混入がないか確認すること。"
    );
}

/// [`predict_batches_hold_doctest_probe_body_matches_fixed_contract`] が
/// 要求する固定文言。`crates/facade/src/lib.rs` の
/// `PredictBatchesHoldDoctestGuard` doc 内の唯一の doctest ブロックから、
/// ネスト `pub mod` の glob import 行（`use fandhe_ai::<mod>::*;`）を
/// 除いた本文と 1 行単位で完全一致する必要がある（クレートルート自体の
/// `use fandhe_ai::*;` は本文に含む）。
const PREDICT_BATCHES_HOLD_PROBE_BODY: &str = "use fandhe_ai::*;\n\
\n\
mod __fandhe_predict_batches_hold_probe {\n\
\x20\x20\x20\x20pub struct PhaseMetrics;\n\
\x20\x20\x20\x20pub fn get_phase_metrics() {}\n\
\x20\x20\x20\x20pub fn current_phase_metrics() {}\n\
\x20\x20\x20\x20pub fn reset_phase_metrics() {}\n\
\x20\x20\x20\x20pub mod inference {\n\
\x20\x20\x20\x20\x20\x20\x20\x20pub fn __mark() {}\n\
\x20\x20\x20\x20}\n\
}\n\
use __fandhe_predict_batches_hold_probe::*;\n\
\n\
struct __FandhePredictBatchesHoldMarker;\n\
\n\
trait __FandhePredictBatchesHoldProbe {\n\
\x20\x20\x20\x20fn predict_batches(&self) -> __FandhePredictBatchesHoldMarker;\n\
\x20\x20\x20\x20fn get_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker;\n\
\x20\x20\x20\x20fn current_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker;\n\
}\n\
\n\
impl __FandhePredictBatchesHoldProbe for fandhe_ai::compat::Sequential {\n\
\x20\x20\x20\x20fn predict_batches(&self) -> __FandhePredictBatchesHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandhePredictBatchesHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn get_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandhePredictBatchesHoldMarker\n\
\x20\x20\x20\x20}\n\
\x20\x20\x20\x20fn current_phase_metrics(&self) -> __FandhePredictBatchesHoldMarker {\n\
\x20\x20\x20\x20\x20\x20\x20\x20__FandhePredictBatchesHoldMarker\n\
\x20\x20\x20\x20}\n\
}\n\
\n\
fn __probe_free_fns(_: PhaseMetrics) {\n\
\x20\x20\x20\x20// 修飾なし呼び出し（`use fandhe_ai::*;` が同名を glob 公開して\n\
\x20\x20\x20\x20// いれば、名前解決自体が曖昧になり E0659 でコンパイル失敗する）。\n\
\x20\x20\x20\x20get_phase_metrics();\n\
\x20\x20\x20\x20current_phase_metrics();\n\
\x20\x20\x20\x20reset_phase_metrics();\n\
\x20\x20\x20\x20inference::__mark();\n\
}\n\
\n\
fn __probe_sequential(seq: &fandhe_ai::compat::Sequential) {\n\
\x20\x20\x20\x20let _: __FandhePredictBatchesHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::predict_batches(seq);\n\
\x20\x20\x20\x20let _: __FandhePredictBatchesHoldMarker = seq.get_phase_metrics();\n\
\x20\x20\x20\x20let _: __FandhePredictBatchesHoldMarker =\n\
\x20\x20\x20\x20\x20\x20\x20\x20fandhe_ai::compat::Sequential::current_phase_metrics(seq);\n\
}";

/// fn 名 4 個（イシュー #2192）。[`scan_predict_batches_reexports_and_declarations`]・
/// [`workspace_declares_predict_batches_fn_names_nowhere`] が共用する。
const PREDICT_BATCHES_FN_NAMES: [&str; 4] = [
    "predict_batches",
    "get_phase_metrics",
    "current_phase_metrics",
    "reset_phase_metrics",
];

/// [`facade_does_not_reexport_or_declare_predict_batches_items`]・その
/// 自己テストが共用する検出本体。facade src 全体（`crates/facade/src/**`）
/// の `pub use` から [`collect_pub_use_leaves`] で別名にする前の葉を集め
/// `PhaseMetrics` を検出し（単一行・複数行・ネストした group・別名も
/// 検出）、`trait`／`struct`／`enum`／`type` 直後の `PhaseMetrics` 独自
/// 宣言、[`PREDICT_BATCHES_FN_NAMES`]（4 個）の `fn` 宣言（可視性・宣言
/// 文脈を問わない。[`count_fn_declarations_by_name`] と同じ検出契約）、
/// `pub mod inference` 宣言（facade 独自の `inference` 公開モジュール
/// 新設）を違反として返す（`scan_npy_io_reexports_and_declarations` と
/// 同型。`pub mod inference` 検出のみ追加）。
fn scan_predict_batches_reexports_and_declarations(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            let leaves = collect_pub_use_leaves(path_tokens);
            for leaf in leaves {
                if leaf == "PhaseMetrics" {
                    offending.push(format!("pub use leaf={leaf}"));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        if matches!(tokens[i].as_str(), "trait" | "struct" | "enum" | "type")
            && tokens.get(i + 1).map(String::as_str) == Some("PhaseMetrics")
        {
            offending.push(format!("{} PhaseMetrics 宣言", tokens[i]));
        }
        if tokens[i] == "pub"
            && tokens.get(i + 1).map(String::as_str) == Some("mod")
            && tokens.get(i + 2).map(String::as_str) == Some("inference")
        {
            offending.push("pub mod inference 宣言".to_string());
        }
        i += 1;
    }

    for fn_name in PREDICT_BATCHES_FN_NAMES {
        let count = count_fn_declarations_by_name(&tokens, fn_name);
        if count > 0 {
            offending.push(format!("`fn {fn_name}` 宣言が {count} 件"));
        }
    }
    offending
}

/// facade src 全体（`crates/facade/src/**`）に、`PhaseMetrics` を識別子
/// 単位で含む `pub use`（複数行・ネストした group・別名含む）も、facade
/// 独自の `trait`／`struct`／`enum`／`type` 宣言も、
/// [`PREDICT_BATCHES_FN_NAMES`]（`predict_batches`／`get_phase_metrics`／
/// `current_phase_metrics`／`reset_phase_metrics`）の `fn` 宣言も、
/// `pub mod inference` 宣言も存在しないことを固定する
/// （`PredictBatchesHoldDoctestGuard` の正のプローブと多層防御を成す
/// 最内層のソース走査ガード。`facade_does_not_reexport_or_declare_npy_io`
/// と同型）。
#[test]
fn facade_does_not_reexport_or_declare_predict_batches_items() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_predict_batches_reexports_and_declarations(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面がバッチ推論・phase 計測（#2192 の\
         `predict_batches`／`PhaseMetrics`／`get_phase_metrics`／\
         `current_phase_metrics`／`reset_phase_metrics`／`pub mod\
         inference`。内部実装限定の新規公開面。facade 公開は承認待ちの\
         ため対象外という設計判断に違反）を再エクスポート、独自宣言、\
         または同名の fn／pub mod を宣言している: {offending:?}"
    );
}

/// [`facade_does_not_reexport_or_declare_predict_batches_items`] の自己
/// テスト（各違反カテゴリの合成ソースを検出できることを固定する。
/// `facade_model_io_public_surface_detects_each_category`
/// と同型）。
#[test]
fn facade_does_not_reexport_or_declare_predict_batches_items_detects_each_category() {
    // 正例: 単一行 pub use。
    assert!(
        !scan_predict_batches_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::PhaseMetrics;"
        )
        .is_empty()
    );
    // 正例: 複数行 pub use（group）。
    assert!(
        !scan_predict_batches_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::{\n    Sequential,\n    PhaseMetrics,\n};"
        )
        .is_empty()
    );
    // 正例: 別名 pub use。
    assert!(
        !scan_predict_batches_reexports_and_declarations(
            "pub use fandhe_ai_facade::compat::PhaseMetrics as Foo;"
        )
        .is_empty()
    );
    // 正例: 独自 struct 宣言。
    assert!(
        !scan_predict_batches_reexports_and_declarations("pub struct PhaseMetrics;").is_empty()
    );
    // 正例: predict_batches の fn 宣言。
    assert!(
        !scan_predict_batches_reexports_and_declarations(
            "impl Sequential { pub fn predict_batches(&self) {} }"
        )
        .is_empty()
    );
    // 正例: get_phase_metrics の fn 宣言。
    assert!(
        !scan_predict_batches_reexports_and_declarations(
            "pub fn get_phase_metrics() -> PhaseMetrics { todo!() }"
        )
        .is_empty()
    );
    // 正例: pub mod inference 宣言。
    assert!(!scan_predict_batches_reexports_and_declarations("pub mod inference {}").is_empty());
    // 負例: コメント中の出現。
    assert!(
        scan_predict_batches_reexports_and_declarations("// pub use ...::predict_batches;")
            .is_empty()
    );
    // 負例: 無関係な型・関数名。
    assert!(
        scan_predict_batches_reexports_and_declarations(
            "pub struct FitConfig; impl FitConfig { pub fn new() {} }"
        )
        .is_empty()
    );
    // 負例: 非公開 `mod inference`（facade 内部実装。本イシューの内部
    // 実装が使う `mod inference;` そのものは違反ではない）。
    assert!(scan_predict_batches_reexports_and_declarations("mod inference;").is_empty());
}

/// workspace 全体（`crates/*/src/`）を再帰走査し、
/// [`PREDICT_BATCHES_FN_NAMES`]（4 個）の `fn` 宣言が**どこにも存在しない
/// こと**を固定する（`workspace_declares_npy_io_names_only_in_allowed_
/// locations` と異なり、本イシューは facade 内部実装が `#[cfg(test)]`
/// 限定のメソッド〈`Sequential::run_loader_inference` 等〉のみで、上記
/// 4 名は一切使わない設計のため、期待集合は空）。
#[test]
fn workspace_declares_predict_batches_fn_names_nowhere() {
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
            for fn_name in PREDICT_BATCHES_FN_NAMES {
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

    assert!(
        found.is_empty(),
        "workspace 全体（crates/*/src/）に predict_batches／\
         get_phase_metrics／current_phase_metrics／reset_phase_metrics\
         のいずれかの `fn` 宣言が見つかった（期待集合は空。#2192 の内部\
         実装はこれらの名前を一切使わない設計のため、見つかった場合は\
         承認済みの実装か迂回経路の混入かを確認すること）: {found:?}"
    );
}

// ==== #2394 TapeRef（借用ハンドル型。var 系メソッドのみ）のガード ====
//
// `TapeRef` は `src/lib.rs` に `pub struct` として直接定義する（`pub use` を
// 通さないため `facade_does_not_reexport_tape_or_backend_ops` は不変）。
// 公開面は `var`／`var_from`／`var_no_grad` の 3 メソッドと `From<&Tape>` のみで、
// 生の `fandhe_ai_autodiff::Tape` へ抜ける経路（`Deref`／`AsRef`／`Tape` を返す
// メソッド等）を持たないことを、トークン走査で fail-closed に固定する
// （REQ-12。#2338 承認事項 4）。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FnVis {
    /// 制限なしの `pub`。
    Public,
    /// `pub(crate)` 等の制限付き可視性。
    Restricted,
    /// 可視性指定なし（trait impl 内の fn を含む）。
    Private,
}

#[derive(Debug)]
struct FnInfo {
    vis: FnVis,
    name: String,
    /// `->` 以降（`{`／`where`／`;` の手前まで）のトークン列。戻り値なしは空。
    ret: Vec<String>,
}

#[derive(Debug)]
struct ImplInfo {
    /// trait impl の trait パス（`for` の手前）のトークン列。固有 impl は `None`。
    trait_tokens: Option<Vec<String>>,
    fns: Vec<FnInfo>,
}

fn tokens_of(content: &str) -> Vec<String> {
    let stripped: String = strip_comments_and_literals(content).into_iter().collect();
    tokenize_including_punctuation(&stripped)
}

/// `tokens[i]` が開き括弧（`open`）のとき、対応する閉じ括弧の index を返す。
fn matching_close(tokens: &[String], i: usize, open: &str, close: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (j, t) in tokens.iter().enumerate().skip(i) {
        if t == open {
            depth += 1;
        } else if t == close {
            depth -= 1;
            if depth == 0 {
                return Some(j);
            }
        }
    }
    None
}

/// `tokens[fn_idx]`（`fn`）の可視性・名前・戻り値型を読む。
fn parse_fn_at(tokens: &[String], fn_idx: usize) -> Option<FnInfo> {
    let name = tokens.get(fn_idx + 1)?.clone();
    // 可視性: 修飾子を遡って読み飛ばし、直前が `pub` か `pub ( ... )` かを見る。
    let mut k = fn_idx;
    while k > 0
        && matches!(
            tokens[k - 1].as_str(),
            "unsafe" | "const" | "async" | "extern"
        )
    {
        k -= 1;
    }
    let vis = if k > 0 && tokens[k - 1] == "pub" {
        FnVis::Public
    } else if k > 0 && tokens[k - 1] == ")" {
        // `pub ( crate )` 形: 対応する `(` の直前が `pub` なら制限付き。
        let mut depth = 0usize;
        let mut open = None;
        for j in (0..k).rev() {
            if tokens[j] == ")" {
                depth += 1;
            } else if tokens[j] == "(" {
                depth -= 1;
                if depth == 0 {
                    open = Some(j);
                    break;
                }
            }
        }
        match open {
            Some(o) if o > 0 && tokens[o - 1] == "pub" => FnVis::Restricted,
            _ => FnVis::Private,
        }
    } else {
        FnVis::Private
    };
    // ジェネリクス・引数リストを読み飛ばす。
    let mut j = fn_idx + 2;
    if tokens.get(j).map(String::as_str) == Some("<") {
        j = matching_close(tokens, j, "<", ">")? + 1;
    }
    if tokens.get(j).map(String::as_str) != Some("(") {
        return None;
    }
    j = matching_close(tokens, j, "(", ")")? + 1;
    let mut ret = Vec::new();
    if tokens.get(j).map(String::as_str) == Some("-")
        && tokens.get(j + 1).map(String::as_str) == Some(">")
    {
        j += 2;
        while let Some(t) = tokens.get(j) {
            if matches!(t.as_str(), "{" | ";" | "where") {
                break;
            }
            ret.push(t.clone());
            j += 1;
        }
    }
    Some(FnInfo { vis, name, ret })
}

/// `content` 内の `impl ... <type_name> ... { ... }`（固有 impl・trait impl の両方。
/// 対象型はパスの最終セグメントの完全一致）を集める。コメント・文字列リテラルは無視する。
fn collect_type_impls(content: &str, type_name: &str) -> Vec<ImplInfo> {
    let tokens = tokens_of(content);
    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if tokens[i] != "impl" {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        if tokens.get(j).map(String::as_str) == Some("<") {
            match matching_close(&tokens, j, "<", ">") {
                Some(c) => j = c + 1,
                None => break,
            }
        }
        let header_start = j;
        while j < tokens.len() && !matches!(tokens[j].as_str(), "{" | ";" | "where") {
            j += 1;
        }
        let header = &tokens[header_start..j];
        // `where` 節を読み飛ばして本体の `{` へ。
        while j < tokens.len() && tokens[j] != "{" && tokens[j] != ";" {
            j += 1;
        }
        if tokens.get(j).map(String::as_str) != Some("{") {
            i += 1;
            continue;
        }
        let for_pos = header.iter().position(|t| t == "for");
        let (trait_tokens, target) = match for_pos {
            Some(p) => (Some(header[..p].to_vec()), &header[p + 1..]),
            None => (None, header),
        };
        let target_path: Vec<&String> = target
            .iter()
            .take_while(|t| t.as_str() != "<")
            .filter(|t| t.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
            .collect();
        let matches_type = target_path.last().is_some_and(|t| t.as_str() == type_name);
        let Some(close) = matching_close(&tokens, j, "{", "}") else {
            break;
        };
        if matches_type {
            let mut fns = Vec::new();
            let mut depth = 0usize;
            for k in j..=close {
                match tokens[k].as_str() {
                    "{" => depth += 1,
                    "}" => depth -= 1,
                    "fn" if depth == 1 => {
                        if let Some(f) = parse_fn_at(&tokens, k) {
                            fns.push(f);
                        }
                    }
                    _ => {}
                }
            }
            out.push(ImplInfo { trait_tokens, fns });
        }
        i = close + 1;
    }
    out
}

/// trait パスのトークン列から trait 名（最初の `<` の手前の最終識別子）を得る。
fn trait_name(tokens: &[String]) -> String {
    tokens
        .iter()
        .take_while(|t| t.as_str() != "<")
        .filter(|t| t.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
        .last()
        .cloned()
        .unwrap_or_default()
}

/// 戻り値型トークン列が生の `Tape`／`BackendOps` へ到達しうるか（`TapeRef` の impl 用。
/// facade `Tape` も含め識別子 `Tape` の完全一致を禁止する。`TapeRef` は別トークン）。
fn ret_reaches_raw_tape(ret: &[String]) -> bool {
    ret.iter()
        .any(|t| matches!(t.as_str(), "fandhe_ai_autodiff" | "Tape" | "BackendOps"))
}

/// `TapeRef` の impl 群を `src/` 全体から集め、宣言元ファイルとともに返す。
fn collect_tape_ref_impls_in_src() -> Vec<ImplInfo> {
    let src_dir = facade_crate_root().join("src");
    let mut all = Vec::new();
    visit_rs_files(&src_dir, &mut |_path, content| {
        all.extend(collect_type_impls(content, "TapeRef"));
    });
    all
}

/// 正ガード: `TapeRef` の公開メソッドは `var`／`var_from`／`var_no_grad` のみ、
/// 制限付き可視性は `from_autodiff` のみ、手書き trait impl は `Debug`／`From` のみ
/// （`Deref`／`AsRef` 等で生の `Tape` へ抜ける経路の追加を検出）。
#[test]
fn tape_ref_public_surface_is_exactly_var_family() {
    let impls = collect_tape_ref_impls_in_src();
    let mut public = std::collections::BTreeSet::new();
    let mut restricted = std::collections::BTreeSet::new();
    let mut traits = std::collections::BTreeSet::new();
    let mut from_headers = 0usize;
    for imp in &impls {
        match &imp.trait_tokens {
            None => {
                for f in &imp.fns {
                    match f.vis {
                        FnVis::Public => {
                            public.insert(f.name.clone());
                        }
                        FnVis::Restricted => {
                            restricted.insert(f.name.clone());
                        }
                        FnVis::Private => {
                            panic!("TapeRef の固有 impl に可視性なしの fn `{}` がある", f.name)
                        }
                    }
                }
            }
            Some(tr) => {
                let name = trait_name(tr);
                if name == "From" {
                    from_headers += 1;
                    assert!(
                        tr.iter().any(|t| t == "Tape")
                            && !tr.iter().any(|t| t == "fandhe_ai_autodiff"),
                        "From の入力は facade の `&Tape` でなければならない: {tr:?}"
                    );
                }
                traits.insert(name);
            }
        }
    }
    let to_set = |xs: &[&str]| -> std::collections::BTreeSet<String> {
        xs.iter().map(|s| (*s).to_string()).collect()
    };
    assert_eq!(
        public,
        to_set(&["var", "var_from", "var_no_grad"]),
        "TapeRef の公開メソッド集合"
    );
    assert_eq!(
        restricted,
        to_set(&["from_autodiff"]),
        "TapeRef の制限付き可視性 fn 集合"
    );
    assert_eq!(
        traits,
        to_set(&["Debug", "From"]),
        "TapeRef の手書き trait impl 集合"
    );
    assert_eq!(from_headers, 1, "From<&Tape> の impl はちょうど 1 件");
}

/// インベントリ: `struct TapeRef` の宣言が `src/lib.rs` にちょうど 1 件で、
/// タプルフィールドが `pub(crate)`、derive が `Clone`・`Copy` のみ。0 件（走査空振り）も fail。
#[test]
fn tape_ref_declared_once_with_crate_private_field() {
    let src_dir = facade_crate_root().join("src");
    let mut decls: Vec<(String, Vec<String>)> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        let tokens = tokens_of(content);
        for (i, t) in tokens.iter().enumerate() {
            if t == "struct" && tokens.get(i + 1).map(String::as_str) == Some("TapeRef") {
                let mut j = i + 2;
                if tokens.get(j).map(String::as_str) == Some("<") {
                    j = matching_close(&tokens, j, "<", ">").expect("ジェネリクスが閉じている") + 1;
                }
                assert_eq!(
                    tokens.get(j).map(String::as_str),
                    Some("("),
                    "タプル構造体であること"
                );
                let field_vis: Vec<String> = tokens[j + 1..j + 5].to_vec();
                // derive 属性を遡って集める。
                let mut derives = Vec::new();
                let mut k = i;
                if k > 0 && tokens[k - 1] == "pub" {
                    k -= 1;
                }
                while k > 0 && tokens[k - 1] == "]" {
                    let mut depth = 0usize;
                    let mut open = k - 1;
                    for m in (0..k).rev() {
                        if tokens[m] == "]" {
                            depth += 1;
                        } else if tokens[m] == "[" {
                            depth -= 1;
                            if depth == 0 {
                                open = m;
                                break;
                            }
                        }
                    }
                    if tokens.get(open + 1).map(String::as_str) == Some("derive") {
                        derives.extend(
                            tokens[open + 2..k - 1]
                                .iter()
                                .filter(|t| {
                                    t.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
                                })
                                .cloned(),
                        );
                    }
                    k = open.saturating_sub(1); // `#` を飛ばす
                }
                derives.sort();
                let mut rec = field_vis;
                rec.push("|".to_string());
                rec.extend(derives);
                decls.push((path.to_string_lossy().replace('\\', "/"), rec));
            }
        }
    });
    assert_eq!(
        decls.len(),
        1,
        "struct TapeRef の宣言はちょうど 1 件: {decls:?}"
    );
    let (path, rec) = &decls[0];
    assert!(path.ends_with("src/lib.rs"), "宣言は src/lib.rs: {path}");
    assert_eq!(
        rec.as_slice(),
        ["pub", "(", "crate", ")", "|", "Clone", "Copy"],
        "フィールド可視性は pub(crate)・derive は Clone/Copy のみ"
    );
}

/// 否定ガード: `TapeRef` の全 fn（固有・trait impl 両方）の戻り値型が生の `Tape`／
/// `BackendOps` を含まない。加えて crate 全体の `pub fn` の戻り値型に
/// `fandhe_ai_autodiff::Tape` が現れないことも固定する。
#[test]
fn tape_ref_pub_fns_do_not_return_raw_tape() {
    for imp in collect_tape_ref_impls_in_src() {
        for f in &imp.fns {
            assert!(
                !ret_reaches_raw_tape(&f.ret),
                "TapeRef::{} の戻り値型が生の Tape へ到達: {:?}",
                f.name,
                f.ret
            );
        }
    }
    let src_dir = facade_crate_root().join("src");
    let mut offending = Vec::new();
    let mut scanned_pub_fns = 0usize;
    visit_rs_files(&src_dir, &mut |path, content| {
        let tokens = tokens_of(content);
        for (i, t) in tokens.iter().enumerate() {
            if t != "fn" {
                continue;
            }
            let Some(f) = parse_fn_at(&tokens, i) else {
                continue;
            };
            if f.vis != FnVis::Public {
                continue;
            }
            scanned_pub_fns += 1;
            let raw = f.ret.windows(4).any(|w| {
                w[0] == "fandhe_ai_autodiff" && w[1] == ":" && w[2] == ":" && w[3] == "Tape"
            });
            if raw {
                offending.push(format!("{}: pub fn {}", path.display(), f.name));
            }
        }
    });
    assert!(
        scanned_pub_fns > 0,
        "pub fn を 1 件も走査できていない（走査空振り）"
    );
    assert!(
        offending.is_empty(),
        "pub fn の戻り値に fandhe_ai_autodiff::Tape が現れる: {offending:?}"
    );
}

/// 自己テスト: 合成入力で各違反類型が検出され、無関係な入力は誤検知しない。
#[test]
fn collect_type_impls_detects_each_category() {
    let src = "impl<'t> TapeRef<'t> {\n pub fn backward(&self) {}\n pub(crate) fn from_autodiff(t: &'t X) -> Self { Self(t) }\n pub\n unsafe fn raw(&self)\n -> &'t fandhe_ai_autodiff::Tape { todo!() }\n pub const fn c() {}\n}\n\
        impl Deref for TapeRef<'_> { type Target = X; fn deref(&self) -> &X { todo!() } }\n\
        impl<'t> From<&'t Tape> for crate::TapeRef<'t> { fn from(t: &'t Tape) -> Self { todo!() } }\n\
        impl Other { pub fn ignored(&self) {} }\n\
        // impl TapeRef { pub fn in_comment() {} }\n\
        const S: &str = \"impl TapeRef { pub fn in_string() {} }\";";
    let impls = collect_type_impls(src, "TapeRef");
    assert_eq!(impls.len(), 3, "固有 1 + Deref + From: {impls:?}");
    let inherent = &impls[0];
    assert!(inherent.trait_tokens.is_none());
    let vis = |n: &str| inherent.fns.iter().find(|f| f.name == n).map(|f| f.vis);
    assert_eq!(vis("backward"), Some(FnVis::Public));
    assert_eq!(vis("from_autodiff"), Some(FnVis::Restricted));
    assert_eq!(
        vis("raw"),
        Some(FnVis::Public),
        "改行・unsafe 修飾子付きも検出"
    );
    assert_eq!(vis("c"), Some(FnVis::Public));
    assert_eq!(vis("in_comment"), None);
    assert_eq!(vis("in_string"), None);
    let raw = inherent.fns.iter().find(|f| f.name == "raw").expect("raw");
    assert!(ret_reaches_raw_tape(&raw.ret), "生の Tape を返す fn を検出");
    assert_eq!(
        trait_name(impls[1].trait_tokens.as_deref().expect("trait")),
        "Deref"
    );
    assert_eq!(
        trait_name(impls[2].trait_tokens.as_deref().expect("trait")),
        "From"
    );
    let ret_ok = vec!["Var".to_string(), "<".into(), "'t".into(), ">".into()];
    assert!(!ret_reaches_raw_tape(&ret_ok));
    let ret_ref = vec!["TapeRef".to_string()];
    assert!(
        !ret_reaches_raw_tape(&ret_ref),
        "TapeRef は別トークンで誤検知しない"
    );
    let ret_facade_tape = vec!["&".to_string(), "Tape".into()];
    assert!(ret_reaches_raw_tape(&ret_facade_tape));
}
// ---------------------------------------------------------------------------
// facade `nn::{Module, ModuleList, Sequential}`・`TapeRef` の公開面の正ガード
// （イシュー #2399・親 #2338 受け入れ条件 4）
//
// 個別に固定済みの観点（`pub mod` の集合・`pub use` の承認形・trait メソッド集合・
// 宣言位置）に加え、ここでは (1) facade だけの import での到達性、(2) `src/nn/*.rs` の
// 「全種別の公開 item 集合」の完全一致、(3) ONNX 走査の禁止部分文字列 `nn::Module` との
// 区別を固定する。期待集合は #2400〜#2402 等で公開メソッドが増える場合は承認のうえ
// 更新する（`facade_nn_module_trait_methods_match_approved_set` と同じ運用）。
// ---------------------------------------------------------------------------

/// `nn::{Module, ModuleList, Sequential}` と `TapeRef` が facade だけの import で到達でき、
/// シグネチャ・基本動作が固定されていることのコンパイル時＋実行時固定
/// （`nn_rnn_types_are_reachable_via_facade_only` と同型。`fandhe_ai_autodiff` は
/// import しない。#2394・#2395・#2396 の公開面）。
#[test]
fn nn_module_types_are_reachable_via_facade_only() {
    use fandhe_ai::TapeRef;
    use fandhe_ai::nn::{Module, ModuleList, Sequential};

    /// パラメータ 1 個のローカル層。
    struct Scale {
        w: fandhe_ai::Tensor<f32>,
        training: bool,
    }
    impl Module for Scale {
        fn forward<'t>(
            &self,
            tape: TapeRef<'t>,
            input: &fandhe_ai::Var<'t>,
        ) -> Result<fandhe_ai::Var<'t>, fandhe_ai::AutodiffError> {
            let w = tape.var(&self.w);
            input.mul(&w)
        }
        fn named_parameters(&self) -> Vec<(String, &fandhe_ai::Tensor<f32>)> {
            vec![("w".to_string(), &self.w)]
        }
        fn set_parameter(
            &mut self,
            name: &str,
            value: fandhe_ai::Tensor<f32>,
        ) -> Result<(), fandhe_ai::AutodiffError> {
            if name != "w" || value.shape() != self.w.shape() {
                return Err(fandhe_ai::AutodiffError::InvalidArgument(
                    "unknown or shape mismatch".to_string(),
                ));
            }
            self.w = value;
            Ok(())
        }
        fn set_training(&mut self, training: bool) {
            self.training = training;
        }
        fn training(&self) -> bool {
            self.training
        }
    }

    // コンパイル時プローブ: シグネチャの固定（`dyn Module` が成り立つこと・`TapeRef` の取得）。
    fn _fwd<'t>(
        m: &dyn Module,
        t: TapeRef<'t>,
        x: &fandhe_ai::Var<'t>,
    ) -> Result<fandhe_ai::Var<'t>, fandhe_ai::AutodiffError> {
        m.forward(t, x)
    }
    fn _tr(t: &fandhe_ai::Tape) -> TapeRef<'_> {
        TapeRef::from(t)
    }

    let scale = || Scale {
        w: fandhe_ai::Tensor::<f32>::from_slice(&[2.0, 3.0], &[2])
            .expect("test fixture: w の構築に失敗"),
        training: true,
    };
    let x_data = fandhe_ai::Tensor::<f32>::from_slice(&[1.0, 1.0], &[2])
        .expect("test fixture: x の構築に失敗");

    // ModuleList: 保持器。forward は Err。
    let mut list = ModuleList::new();
    assert!(list.is_empty());
    list.push(Box::new(scale()));
    assert_eq!(list.len(), 1);
    assert!(list.get(0).is_some() && list.get(1).is_none());
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    assert!(list.forward(TapeRef::from(&tape), &x).is_err());

    // Sequential: add／push／layers／From<ModuleList>。
    let mut seq = Sequential::new().add(scale());
    seq.push(Box::new(scale()));
    assert_eq!((seq.len(), seq.layers().len()), (2, 2));
    let from_list = Sequential::from(list);
    assert_eq!(from_list.len(), 1);

    // forward → backward → 入力勾配。
    let out = seq
        .forward(TapeRef::from(&tape), &x)
        .expect("test fixture: forward");
    let loss = out.sum(None).expect("test fixture: sum");
    let grads = tape.backward(&loss).expect("test fixture: backward");
    assert!(grads.get(&x).expect("test fixture: get").is_some());

    // state_dict／load_state_dict の往復と set_training／training。
    let sd = seq.state_dict();
    assert_eq!(sd.len(), 2);
    seq.load_state_dict(sd)
        .expect("test fixture: load_state_dict");
    assert!(seq.training());
    seq.set_training(false);
    assert!(!seq.training());
}

/// `(種別, 名前)` の集合。
type PubItemSet = std::collections::BTreeSet<(String, String)>;
/// 名前の集合。
type NameSet = std::collections::BTreeSet<String>;

/// 単一ファイルの「brace／paren／bracket 深さ 0」にある `pub` item を
/// `(種別, 名前)` の集合として返す（制限なし `pub` と `pub(...)` 制限付きを別集合に分ける）。
/// `pub use` は use tree を展開した葉ごとに `("use", 葉)` とする。
/// 関数本体・impl・trait 本体・`#[cfg(test)] mod tests { .. }` の中身は深さ 1 以上のため
/// 走査対象外（impl 側は [`scan_type_impl_surface`] が別途固定する）。
/// コメント・文字列リテラルは無視する。
fn scan_top_level_pub_items(content: &str) -> (PubItemSet, PubItemSet) {
    const KINDS: [&str; 10] = [
        "mod", "use", "fn", "struct", "enum", "trait", "type", "const", "static", "union",
    ];
    let tokens = tokens_of(content);
    let mut public = std::collections::BTreeSet::new();
    let mut restricted = std::collections::BTreeSet::new();
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < tokens.len() {
        match tokens[i].as_str() {
            "{" | "(" | "[" => depth += 1,
            "}" | ")" | "]" => depth -= 1,
            "pub" if depth == 0 => {
                let mut j = i + 1;
                let mut is_restricted = false;
                if tokens.get(j).map(String::as_str) == Some("(") {
                    is_restricted = true;
                    j = matching_close(&tokens, j, "(", ")").map_or(tokens.len(), |c| c + 1);
                }
                // `const fn` の `const` は修飾子。`pub const X` の `const` は種別。
                while let Some(t) = tokens.get(j) {
                    let is_qualifier = matches!(t.as_str(), "async" | "unsafe" | "extern")
                        || (t == "const" && tokens.get(j + 1).map(String::as_str) == Some("fn"));
                    if !is_qualifier {
                        break;
                    }
                    j += 1;
                }
                let kind = tokens.get(j).cloned().unwrap_or_default();
                let entry_names: Vec<String> = if kind == "use" {
                    let mut end = j + 1;
                    while end < tokens.len() && tokens[end] != ";" {
                        end += 1;
                    }
                    collect_pub_use_leaves(&tokens[j + 1..end.min(tokens.len())])
                } else if KINDS.contains(&kind.as_str()) {
                    tokens.get(j + 1).cloned().into_iter().collect()
                } else {
                    // 未知の種別（`macro` 等）も見逃さず、種別名そのものを記録する。
                    vec![String::new()]
                };
                for name in entry_names {
                    let set = if is_restricted {
                        &mut restricted
                    } else {
                        &mut public
                    };
                    set.insert((kind.clone(), name));
                }
            }
            _ => {}
        }
        i += 1;
    }
    (public, restricted)
}

fn pair_set(items: &[(&str, &str)]) -> std::collections::BTreeSet<(String, String)> {
    items
        .iter()
        .map(|(k, n)| (k.to_string(), n.to_string()))
        .collect()
}

/// `type_name` の impl 群から (固有 impl の制限なし pub fn 名, 固有 impl の非 pub fn 名,
/// 手書き trait impl の trait 名) を返す。
fn scan_type_impl_surface(content: &str, type_name: &str) -> (NameSet, Vec<String>, NameSet) {
    let mut public = std::collections::BTreeSet::new();
    let mut nonpublic = Vec::new();
    let mut traits = std::collections::BTreeSet::new();
    for imp in collect_type_impls(content, type_name) {
        match &imp.trait_tokens {
            None => {
                for f in imp.fns {
                    if f.vis == FnVis::Public {
                        public.insert(f.name);
                    } else {
                        nonpublic.push(f.name);
                    }
                }
            }
            Some(t) => {
                traits.insert(trait_name(t));
            }
        }
    }
    (public, nonpublic, traits)
}

fn nn_src(file: &str) -> String {
    read_to_string_or_panic(&facade_crate_root().join("src/nn").join(file))
}

/// 正ガード: `src/nn/mod.rs` の全種別の公開 item が
/// `pub mod rnn` と `pub use` の 8 件（`Module`・`ModuleDict`・`ModuleList`・`Sequential`・`summary`。
/// `ModuleDict`／`summary` は #2402 で承認済み。`Transformer`／`TransformerConfig`／
/// `TransformerDecoderLayer` は #2532・#2533 で承認済み）に完全一致する。
/// `pub fn`／`pub struct` 等の追加や `pub mod container;` 等の新設は fail する。
#[test]
fn nn_mod_public_items_match_expected_set() {
    let (public, restricted) = scan_top_level_pub_items(&nn_src("mod.rs"));
    assert_eq!(
        public,
        pair_set(&[
            ("mod", "init"),
            ("mod", "init"),
            ("mod", "rnn"),
            ("use", "Module"),
            ("use", "ModuleDict"),
            ("use", "ModuleList"),
            ("use", "Sequential"),
            ("use", "Transformer"),
            ("use", "TransformerConfig"),
            ("use", "TransformerDecoderLayer"),
            ("use", "summary"),
        ]),
        "src/nn/mod.rs の公開 item 集合が期待と一致しない（nn 公開面の無断拡大を検知）"
    );
    assert_eq!(
        restricted,
        pair_set(&[("use", "FacadeModuleAdapter")]),
        "制限付き pub item は crate 内アダプタの再エクスポート 1 件のみ（#2401）"
    );
}

/// 正ガード: `src/nn/module.rs` の公開 item は `trait Module` のみ。
/// 制限付き可視性のまま残すのは crate 内アダプタ `FacadeModuleAdapter`（REQ-12）と
/// 走査用の内部型別名 `NodeKey`（`pub(super) type`。#2401）の 2 件。
#[test]
fn nn_module_rs_public_items_match_expected_set() {
    let (public, restricted) = scan_top_level_pub_items(&nn_src("module.rs"));
    assert_eq!(public, pair_set(&[("trait", "Module")]));
    assert_eq!(
        restricted,
        pair_set(&[("struct", "FacadeModuleAdapter"), ("type", "NodeKey")]),
        "FacadeModuleAdapter・NodeKey は制限付き可視性のままであること（公開面へ出さない）"
    );
}

/// 正ガード: `src/nn/container.rs` の公開 item は `ModuleDict`・`ModuleList`・`Sequential` の
/// 3 構造体と `fn summary`（#2402 で承認済み）のみ。制限付き可視性は #2400（PR #2426）の
/// `set_requires_grad` ロールバック用 crate 内ヘルパー（葉単位スナップショット型と
/// snapshot／restore 関数。`FacadeModuleAdapter` からも使う）の 3 件のみ。
#[test]
fn nn_container_rs_public_items_match_expected_set() {
    let (public, restricted) = scan_top_level_pub_items(&nn_src("container.rs"));
    assert_eq!(
        public,
        pair_set(&[
            ("fn", "summary"),
            ("struct", "ModuleDict"),
            ("struct", "ModuleList"),
            ("struct", "Sequential"),
        ])
    );
    assert_eq!(
        restricted,
        pair_set(&[
            ("enum", "RequiresGradSnapshot"),
            ("fn", "restore_requires_grad"),
            ("fn", "snapshot_requires_grad"),
        ]),
        "制限付き pub item は凍結ロールバック用の crate 内ヘルパー 3 件のみ"
    );
}

/// 正ガード: `ModuleList`／`Sequential` の固有 pub メソッド集合と手書き trait impl 集合。
/// 固有 impl に可視性なし・制限付きの fn を置かない（`Deref` 等で内部へ抜ける経路の追加も
/// trait impl 集合で検出する）。
#[test]
fn nn_containers_inherent_and_trait_impls_match_expected_set() {
    let content = nn_src("container.rs");
    let to_set = |xs: &[&str]| -> std::collections::BTreeSet<String> {
        xs.iter().map(|s| s.to_string()).collect()
    };
    let (list_pub, list_np, list_traits) = scan_type_impl_surface(&content, "ModuleList");
    assert_eq!(
        list_pub,
        to_set(&[
            "new", "push", "len", "is_empty", "get", "get_mut", "iter", "iter_mut"
        ])
    );
    assert!(
        list_np.is_empty(),
        "ModuleList の非 pub 固有 fn: {list_np:?}"
    );
    assert_eq!(list_traits, to_set(&["Default", "FromIterator", "Module"]));

    let (seq_pub, seq_np, seq_traits) = scan_type_impl_surface(&content, "Sequential");
    assert_eq!(
        seq_pub,
        to_set(&[
            "new",
            "push",
            "add",
            "len",
            "is_empty",
            "layers",
            "layers_mut"
        ])
    );
    assert!(seq_np.is_empty(), "Sequential の非 pub 固有 fn: {seq_np:?}");
    assert_eq!(seq_traits, to_set(&["Default", "From", "Module"]));

    let (dict_pub, dict_np, dict_traits) = scan_type_impl_surface(&content, "ModuleDict");
    assert_eq!(
        dict_pub,
        to_set(&[
            "new",
            "from_pairs",
            "insert",
            "remove",
            "get",
            "get_mut",
            "contains_key",
            "keys",
            "iter",
            "iter_mut",
            "len",
            "is_empty"
        ])
    );
    assert!(
        dict_np.is_empty(),
        "ModuleDict の非 pub 固有 fn: {dict_np:?}"
    );
    assert_eq!(dict_traits, to_set(&["Default", "Module"]));
}

/// [`scan_top_level_pub_items`]・[`scan_type_impl_surface`] の自己テスト
/// （合成入力で各カテゴリの逸脱が検出され、現行の形が正例として通ることを確認する）。
#[test]
fn nn_public_item_set_scanners_detect_each_category() {
    let base = "mod container; mod module; pub mod init; pub mod rnn;\n\
                pub use container::{ModuleDict, ModuleList, Sequential, summary};\n\
                pub use module::Module;\n";
    let expected_mod = pair_set(&[
        ("mod", "init"),
        ("mod", "rnn"),
        ("use", "Module"),
        ("use", "ModuleDict"),
        ("use", "ModuleList"),
        ("use", "Sequential"),
        ("use", "summary"),
    ]);
    // 正例。
    assert_eq!(scan_top_level_pub_items(base).0, expected_mod);
    // 負例: 追加された公開 item はすべて期待集合から外れる。
    for extra in [
        "pub fn x() {}",
        "pub mod container;",
        "pub use module::FacadeModuleAdapter;",
        "pub struct S;",
        "pub const fn c() {}",
        "pub unsafe fn u() {}",
        "pub macro m() {}",
    ] {
        let src = format!("{base}{extra}\n");
        assert_ne!(
            scan_top_level_pub_items(&src).0,
            expected_mod,
            "追加が検出されなかった: {extra}"
        );
    }
    // 関数本体・テストモジュール内の `pub` は走査対象外（深さ 1 以上）。
    let nested = format!("{base}#[cfg(test)] mod tests {{ pub fn t() {{}} }}\n");
    assert_eq!(scan_top_level_pub_items(&nested).0, expected_mod);
    // 制限付き可視性は別集合。フィールドの `pub(crate)` は拾わない。
    let (p, r) = scan_top_level_pub_items("pub(crate) struct A<P>(pub(crate) P); pub trait T {}");
    assert_eq!(p, pair_set(&[("trait", "T")]));
    assert_eq!(r, pair_set(&[("struct", "A")]));

    // impl 側: 固有 pub fn の追加・非 pub 固有 fn・追加 trait impl の検出。
    let imp = "impl Sequential { pub fn new() -> Self { todo!() } }\n\
               impl Default for Sequential { fn default() -> Self { todo!() } }";
    let (pubs, np, traits) = scan_type_impl_surface(imp, "Sequential");
    assert_eq!((pubs.len(), np.len(), traits.len()), (1, 0, 1));
    let bad = format!(
        "{imp}\nimpl Sequential {{ pub fn extra(&self) {{}} fn hidden(&self) {{}} \
         pub(crate) fn r(&self) {{}} }}\nimpl Deref for Sequential {{ }}"
    );
    let (pubs, np, traits) = scan_type_impl_surface(&bad, "Sequential");
    assert!(pubs.contains("extra"));
    assert!(np.contains(&"hidden".to_string()) && np.contains(&"r".to_string()));
    assert!(traits.contains("Deref"));
}

/// ONNX 走査の禁止部分文字列 `nn::Module`（`src/interop/onnx.rs` 限定）は、本来は内部型
/// `fandhe_ai_autodiff::nn::Module` の漏洩検出用だが、部分文字列一致のため facade の公開名
/// `fandhe_ai::nn::Module`（#2395）にも一致する。走査対象は `src/interop/onnx.rs` に限られ、
/// facade `nn::Module` を受け取る ONNX API は承認範囲外
/// （`docs/facade-onnx-export-exposure-decision.md`）のため、この一致は意図した fail-closed で
/// 誤検知（衝突）ではない。現状 `onnx.rs` は `nn::Module` を一切参照せず衝突していないことと、
/// 走査が両方の名前を検出することを固定する。将来 facade `nn::Module` から ONNX へ export する
/// API を承認する場合は、このエントリと許容リストを同時に見直す。
#[test]
fn onnx_forbidden_nn_module_substring_does_not_collide_with_facade_nn_module() {
    let onnx_path = facade_crate_root().join("src/interop/onnx.rs");
    let content = read_to_string_or_panic(&onnx_path);
    let compact: String = strip_comments_and_literals(&content)
        .into_iter()
        .filter(|c| !c.is_whitespace())
        .collect();
    assert!(
        !compact.contains("nn::Module"),
        "src/interop/onnx.rs が nn::Module を参照している（facade／内部のいずれも承認範囲外）"
    );

    for ty in ["crate::nn::Module", "fandhe_ai::nn::Module"] {
        let src =
            format!("pub fn from_module(m: &dyn {ty}) -> Result<Self, OnnxError> {{ todo!() }}");
        let offenses = scan_unapproved_onnx_pub_items(&src);
        assert!(
            offenses.iter().any(|o| o.contains("from_module")),
            "承認範囲外の fn 名が検出されなかった ({ty}): {offenses:?}"
        );
        assert!(
            offenses.iter().any(|o| o.contains("nn::Module")),
            "シグネチャ中の nn::Module が検出されなかった ({ty}): {offenses:?}"
        );
    }

    // nn の各ファイルは ONNX 走査の対象パス（src/interop/onnx.rs）と別である。
    for file in ["mod.rs", "module.rs", "container.rs"] {
        assert_ne!(facade_crate_root().join("src/nn").join(file), onnx_path);
    }
}
// =====================================================================
// イシュー #2372（親 #2362）: `fandhe_ai_autodiff::nn::optim::amp::
// grad_scaler_from_state` が facade の公開面に現れないことの固定（AC4）。
//
// `save_model`／`load_model`（`compat::model_io`）は GradScaler の状態を復元するため内部クレートの
// 自由関数 `grad_scaler_from_state` を呼ぶが、これを facade へ再エクスポートすると公開面が広がる
// （inherent メソッドにしなかった理由は同関数の doc 参照）。3 層で固定する:
//   1. facade src の `pub use`・宣言のソース走査（否定ガード。自己テスト付き）
//   2. workspace 全体の `fn grad_scaler_from_state` 宣言インベントリ
//   3. コンパイル時の正のプローブ（inherent 追加・glob 漏出をコンパイルエラーで検出）
// stable rustdoc は `compile_fail` のエラーコードを照合しないため doctest には頼らない。
// =====================================================================

/// facade src が `grad_scaler_from_state`／`amp` モジュールを `pub use`（別名・group 経由含む）、
/// または `nn::optim`／`amp` への glob `pub use` で再エクスポートするか、
/// `fn grad_scaler_from_state` を宣言していれば違反を返す。
fn scan_grad_scaler_from_state_leaks(content: &str) -> Vec<String> {
    let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
    let tokens = tokenize_including_punctuation(&cleaned);
    let mut offending: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i] == "pub" && tokens.get(i + 1).map(String::as_str) == Some("use") {
            let mut end = i + 2;
            while end < tokens.len() && tokens[end] != ";" {
                end += 1;
            }
            let path_tokens = &tokens[i + 2..end.min(tokens.len())];
            for token in path_tokens {
                if token == "grad_scaler_from_state" || token == "amp" {
                    offending.push(format!("pub use path segment={token}"));
                }
            }
            // glob（`optim::*`・`amp::*`）は `amp` の項目を漏らしうる。
            for (idx, token) in path_tokens.iter().enumerate() {
                if token == "*"
                    && idx >= 3
                    && path_tokens[idx - 1] == ":"
                    && path_tokens[idx - 2] == ":"
                    && matches!(path_tokens[idx - 3].as_str(), "optim" | "amp")
                {
                    offending.push(format!("{} への glob pub use", path_tokens[idx - 3]));
                }
            }
            i = (end + 1).min(tokens.len());
            continue;
        }
        i += 1;
    }
    if count_fn_declarations_by_name(&tokens, "grad_scaler_from_state") > 0 {
        offending.push("fn grad_scaler_from_state 宣言".to_string());
    }
    offending
}

#[test]
fn facade_does_not_reexport_or_declare_grad_scaler_from_state() {
    let src_dir = facade_crate_root().join("src");
    let mut offending: Vec<String> = Vec::new();
    visit_rs_files(&src_dir, &mut |path, content| {
        for offense in scan_grad_scaler_from_state_leaks(content) {
            offending.push(format!("{}: {offense}", path.display()));
        }
    });
    assert!(
        offending.is_empty(),
        "facade の公開面が grad_scaler_from_state（イシュー #2372・内部クレート限定）を再エクスポート、\
         または宣言している: {offending:?}"
    );
}

#[test]
fn facade_does_not_reexport_or_declare_grad_scaler_from_state_detects_each_category() {
    let leaks = scan_grad_scaler_from_state_leaks;
    // 正例: 単行・group・別名・モジュール・glob・宣言。
    for src in [
        "pub use fandhe_ai_autodiff::nn::optim::amp::grad_scaler_from_state;",
        "pub use fandhe_ai_autodiff::nn::optim::amp::{scale_loss, grad_scaler_from_state};",
        "pub use fandhe_ai_autodiff::nn::optim::amp::grad_scaler_from_state as f;",
        "pub use fandhe_ai_autodiff::nn::optim::amp;",
        "pub use fandhe_ai_autodiff::nn::optim::amp::*;",
        "pub use fandhe_ai_autodiff::nn::optim::*;",
        "pub fn grad_scaler_from_state() {}",
    ] {
        assert!(!leaks(src).is_empty(), "検出されなかった: {src}");
    }
    // 負例: コメント・文字列リテラル・非公開 use・無関係な pub use・承認済みの GradScaler 系。
    for src in [
        "// pub use x::grad_scaler_from_state;",
        "let s = \"grad_scaler_from_state\";",
        "use fandhe_ai_autodiff::nn::optim::amp::grad_scaler_from_state;",
        "pub use fandhe_ai_autodiff::nn::optim::AdamW;",
        "pub use fandhe_ai_autodiff::nn::optim::{GradScaler, GradScalerConfig, UnscaleResult};",
        "fn call() { grad_scaler_from_state(c, 1.0, 0); }",
    ] {
        assert!(leaks(src).is_empty(), "誤検出: {src}");
    }
}

/// workspace 全体で `fn grad_scaler_from_state` の宣言が内部クレートの 1 か所だけであることを固定する
/// （facade からの呼び出しは宣言ではないため対象外）。
#[test]
fn workspace_declares_grad_scaler_from_state_only_in_autodiff_amp() {
    let crates_dir = workspace_crates_dir();
    let mut found: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut crate_dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&crates_dir)
        .unwrap_or_else(|e| panic!("crates ディレクトリが読めない: {e}"))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    crate_dirs.sort();
    assert!(!crate_dirs.is_empty(), "検査対象を見失っている");
    for crate_dir in &crate_dirs {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }
        visit_rs_files(&src_dir, &mut |path, content| {
            let cleaned: String = strip_comments_and_literals(content).into_iter().collect();
            let tokens = tokenize_including_punctuation(&cleaned);
            let count = count_fn_declarations_by_name(&tokens, "grad_scaler_from_state");
            if count > 0 {
                let rel = path
                    .strip_prefix(&crates_dir)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .replace('\\', "/");
                *found.entry(rel).or_insert(0) += count;
            }
        });
    }
    let expected: std::collections::BTreeMap<String, usize> =
        [("autodiff/src/nn/optim/amp.rs".to_string(), 1usize)]
            .into_iter()
            .collect();
    assert_eq!(
        found, expected,
        "grad_scaler_from_state の宣言集合が期待と一致しない"
    );
}

/// 正のプローブ 1（型パス）: `GradScaler` に同名のローカル trait 関数を生やし、型パス呼び出しの
/// 戻り値を `Marker` に束縛する。将来 inherent の `from_state`／`grad_scaler_from_state` が facade
/// 経由の `GradScaler` に加わると、inherent が trait より優先されて戻り値型が変わり、
/// このモジュールがコンパイルできなくなる（fail-closed）。
mod grad_scaler_from_state_type_path_probe {
    pub struct Marker;
    pub trait FromStateProbe {
        fn from_state() -> Marker;
        fn grad_scaler_from_state() -> Marker;
    }
    impl FromStateProbe for fandhe_ai::optim::GradScaler {
        fn from_state() -> Marker {
            Marker
        }
        fn grad_scaler_from_state() -> Marker {
            Marker
        }
    }
    pub fn probe() -> (Marker, Marker) {
        (
            <fandhe_ai::optim::GradScaler as FromStateProbe>::from_state(),
            fandhe_ai::optim::GradScaler::grad_scaler_from_state(),
        )
    }
}

/// facade の全公開モジュールパス（`src/lib.rs` から到達可能な `pub mod`）。下の glob probe が網羅する。
const GRAD_SCALER_PROBE_MODULES: [&str; 10] = [
    "compat",
    "data",
    "interop",
    "interop::onnx",
    "interop::safetensors",
    "model",
    "nn",
    "nn::init",
    "nn::rnn",
    "optim",
];

/// 正のプローブ 2（glob 漏出）: ローカルの同名自由関数と facade の各公開モジュールを `pub use` の
/// glob で並べ、呼び出しの曖昧性（E0659）で漏出を検出する。漏出すると `api_surface` のテスト
/// バイナリ全体がビルドに失敗するが、fail-closed として受け入れる。
/// 漏出がない正常時はどの glob も名前を供給しないため、`unused_imports` はこの probe の存在意義そのもの
/// （未使用であることが期待状態）であり、このモジュール限定で許可する。
#[allow(unused_imports)]
mod grad_scaler_from_state_glob_probe {
    pub struct Marker;
    mod local {
        pub fn grad_scaler_from_state() -> super::Marker {
            super::Marker
        }
    }
    pub use self::local::*;
    pub use fandhe_ai::compat::*;
    pub use fandhe_ai::data::*;
    pub use fandhe_ai::interop::onnx::*;
    pub use fandhe_ai::interop::safetensors::*;
    pub use fandhe_ai::interop::*;
    pub use fandhe_ai::model::*;
    pub use fandhe_ai::nn::init::*;
    pub use fandhe_ai::nn::rnn::*;
    pub use fandhe_ai::nn::*;
    pub use fandhe_ai::optim::*;
    pub use fandhe_ai::*;

    pub fn probe() -> Marker {
        grad_scaler_from_state()
    }
}

#[test]
fn grad_scaler_from_state_positive_probes_compile_and_glob_list_matches_lib_rs() {
    let _ = grad_scaler_from_state_type_path_probe::probe();
    let _ = grad_scaler_from_state_glob_probe::probe();
    let declared = collect_public_module_paths(&facade_crate_root().join("src"));
    let probed: std::collections::BTreeSet<String> = GRAD_SCALER_PROBE_MODULES
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        declared, probed,
        "facade の pub mod 集合と glob probe の一覧がドリフトしている。\
         新しい pub mod を足したら GRAD_SCALER_PROBE_MODULES と \
         grad_scaler_from_state_glob_probe の `pub use` を更新すること"
    );
}
