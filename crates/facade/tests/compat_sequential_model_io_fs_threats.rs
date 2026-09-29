//! `fandhe_ai::compat::{save_model, load_model}` のファイルシステム脅威（symlink・非通常
//! ファイル・差し替え・削除しない契約・中断）の統合テスト（イシュー #2376・親 #2362）。
//!
//! 出典: `docs/compat-model-io-decision.md` §6 後半・§12.3・§13.3（脅威棚卸し表）・§13.6。
//! 公開 API と `std` のみで検証する（名前生成の衝突注入は `src/compat/model_io/fs_threat_tests.rs`、
//! 検査〜open 間の差し替えは `src/fs_guard.rs` の単体テストが担う）。基本ケースは
//! `compat_sequential_model_io.rs` にあり、本ファイルはその補完で重複させない。
//! symlink の参照先は常に各テストの一時ディレクトリ配下に置く。

use std::path::Path;

use fandhe_ai::compat::{ModelIoError, Sequential, load_model, save_model};

mod common;
use common::temp_dir::TempDirGuard;

fn model(seed: u64) -> Sequential {
    Sequential::new()
        .add_linear(3, 4, seed)
        .and_then(|m| m.add_relu().add_linear(4, 2, seed + 1))
        .expect("構築できるはず")
}

fn state_bits(m: &Sequential) -> Vec<(String, Vec<u32>)> {
    let mut v: Vec<_> = m
        .state_dict()
        .iter()
        .map(|(k, t)| {
            let c = t.contiguous();
            (
                k.clone(),
                c.as_slice()
                    .expect("連続のはず")
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
            )
        })
        .collect();
    v.sort();
    v
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("読めるはず")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn safetensors_entries(dir: &Path) -> Vec<String> {
    entries(dir)
        .into_iter()
        .filter(|n| n.starts_with("model.") && n.ends_with(".safetensors"))
        .collect()
}

fn try_load(dir: &Path) -> Result<(), ModelIoError> {
    load_model(dir).map(|_| ())
}

fn assert_io_rejects(dir: &Path, what: &str) {
    assert!(
        matches!(try_load(dir), Err(ModelIoError::Io(_))),
        "{what}: Io で拒否されるはず"
    );
}

/// 保存済みディレクトリを作り、(guard, dir, manifest 参照先の safetensors 名) を返す。
fn saved(label: &str) -> (TempDirGuard, std::path::PathBuf, String) {
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("model");
    save_model(&model(1), &dir).expect("保存できるはず");
    let st = safetensors_entries(&dir);
    assert_eq!(st.len(), 1);
    (guard, dir.clone(), st[0].clone())
}

/// FIFO を `mkfifo` で作る（固定引数のみ。存在しない環境は黙って skip せず失敗させる）。
fn mkfifo(path: &Path) {
    let status = std::process::Command::new("mkfifo")
        .arg(path)
        .status()
        .expect("mkfifo を起動できるはず");
    assert!(status.success(), "mkfifo が成功するはず");
}

/// `f` を別スレッドで走らせ、ハングしたらテストを失敗させる（`O_NONBLOCK` 退行の検出）。
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("ハングせず結果を返すはず")
}

#[cfg(unix)]
#[test]
fn load_rejects_dangling_symlink_manifest_and_safetensors() {
    for target_is_manifest in [true, false] {
        let (_g, dir, st) = saved("dangling");
        let leaf = if target_is_manifest {
            dir.join("manifest.json")
        } else {
            dir.join(&st)
        };
        std::fs::remove_file(&leaf).expect("消せるはず");
        std::os::unix::fs::symlink(dir.join("nowhere"), &leaf).expect("symlink");
        assert_io_rejects(&dir, "dangling symlink");
    }
}

#[cfg(unix)]
#[test]
fn load_rejects_leaf_replaced_with_symlink_after_initial_write() {
    for target_is_manifest in [true, false] {
        let (g, dir, st) = saved("swap-symlink");
        let name = if target_is_manifest {
            "manifest.json".to_string()
        } else {
            st
        };
        let leaf = dir.join(&name);
        // 内容が同一の外部コピーへ差し替える（内容不一致ではなく symlink 自体が拒否理由になる）。
        let copy = g.path().join("outside-copy");
        std::fs::copy(&leaf, &copy).expect("コピーできるはず");
        std::fs::remove_file(&leaf).expect("消せるはず");
        std::os::unix::fs::symlink(&copy, &leaf).expect("symlink");
        assert_io_rejects(&dir, &name);
    }
}

#[cfg(unix)]
#[test]
fn load_rejects_fifo_manifest_and_safetensors_without_hanging() {
    for target_is_manifest in [true, false] {
        let (_g, dir, st) = saved("fifo");
        let leaf = if target_is_manifest {
            dir.join("manifest.json")
        } else {
            dir.join(&st)
        };
        std::fs::remove_file(&leaf).expect("消せるはず");
        mkfifo(&leaf);
        let d = dir.clone();
        let result = bounded(move || try_load(&d));
        assert!(matches!(result, Err(ModelIoError::Io(_))), "FIFO は拒否");
    }
}

/// 既存の `manifest.json` が symlink でも、`rename` はリンクエントリ自体を置換し
/// 参照先へ書き込まない（§13.3 「`manifest.json` の rename 置換先」）。
#[cfg(all(unix, any(target_os = "linux", target_os = "macos")))]
#[test]
fn save_replaces_symlinked_manifest_entry_without_writing_through() {
    for dangling in [false, true] {
        let g = TempDirGuard::new("manifest-symlink");
        let dir = g.path().join("model");
        std::fs::create_dir(&dir).expect("mkdir");
        let outside = g.path().join("outside.txt");
        if !dangling {
            std::fs::write(&outside, b"outside-bytes").expect("書き込めるはず");
        }
        std::os::unix::fs::symlink(&outside, dir.join("manifest.json")).expect("symlink");

        let m = model(5);
        save_model(&m, &dir).expect("保存できるはず");

        let meta = std::fs::symlink_metadata(dir.join("manifest.json")).expect("メタデータ");
        assert!(meta.file_type().is_file(), "symlink は通常ファイルへ置換");
        if dangling {
            assert!(!outside.exists(), "参照先は作られない");
        } else {
            assert_eq!(
                std::fs::read(&outside).expect("読めるはず"),
                b"outside-bytes"
            );
        }
        let loaded = load_model(&dir).expect("新モデルを復元できるはず");
        assert_eq!(state_bits(&m), state_bits(&loaded));
    }
}

/// 再保存は旧世代・よそ者の命名規則一致ファイル・古い一時ファイルを一切削除しない。
#[cfg(unix)]
#[test]
fn resave_never_deletes_old_generation_strangers_or_stale_tmp() {
    let (_g, dir, old_st) = saved("no-delete");
    let old_bytes = std::fs::read(dir.join(&old_st)).expect("読めるはず");
    let stranger = format!("model.{}.safetensors", "deadbeef".repeat(4));
    let stale_tmp = ".manifest.json.tmp-1-2-3";
    std::fs::write(dir.join(&stranger), b"stranger").expect("書き込めるはず");
    std::fs::write(dir.join(stale_tmp), b"stale").expect("書き込めるはず");

    let b = model(9);
    save_model(&b, &dir).expect("再保存できるはず");

    assert_eq!(std::fs::read(dir.join(&old_st)).expect("残存"), old_bytes);
    assert_eq!(
        std::fs::read(dir.join(&stranger)).expect("残存"),
        b"stranger"
    );
    assert_eq!(std::fs::read(dir.join(stale_tmp)).expect("残存"), b"stale");
    assert_eq!(safetensors_entries(&dir).len(), 3);
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_eq!(state_bits(&b), state_bits(&loaded));
}

/// 中断した保存の残骸（孤立 safetensors・一時 manifest）があっても旧コミットを読め、
/// 再保存も成功して残骸に触れない。
#[cfg(unix)]
#[test]
fn interrupted_save_leaves_previous_commit_loadable() {
    let g = TempDirGuard::new("interrupted");
    let dir = g.path().join("model");
    let a = model(1);
    save_model(&a, &dir).expect("保存できるはず");
    let orphan = format!("model.{}.safetensors", "0123456789abcdef".repeat(2));
    let tmp = ".manifest.json.tmp-9-9-9";
    std::fs::write(dir.join(&orphan), b"garbage").expect("書き込めるはず");
    std::fs::write(dir.join(tmp), b"{ partial").expect("書き込めるはず");

    let loaded = load_model(&dir).expect("旧コミットを復元できるはず");
    assert_eq!(state_bits(&a), state_bits(&loaded));

    let b = model(20);
    save_model(&b, &dir).expect("再保存できるはず");
    assert_eq!(std::fs::read(dir.join(&orphan)).expect("残存"), b"garbage");
    assert_eq!(std::fs::read(dir.join(tmp)).expect("残存"), b"{ partial");
    let loaded = load_model(&dir).expect("新コミットを復元できるはず");
    assert_eq!(state_bits(&b), state_bits(&loaded));
}

/// `rename` が失敗（`manifest.json` が中身のあるディレクトリ）したら自己所有の一時 manifest を
/// 削除し、既存ディレクトリには触れない（§13.6）。孤立 safetensors は残る（§12.3 手順 1）。
#[cfg(unix)]
#[test]
fn save_removes_own_tmp_when_rename_fails() {
    let g = TempDirGuard::new("rename-fail");
    let dir = g.path().join("model");
    std::fs::create_dir_all(dir.join("manifest.json")).expect("mkdir");
    std::fs::write(dir.join("manifest.json").join("inner"), b"inner").expect("書き込めるはず");

    let err = save_model(&model(1), &dir).expect_err("rename 失敗で Err のはず");
    assert!(matches!(err, ModelIoError::Io(_)));

    let names = entries(&dir);
    assert!(
        !names.iter().any(|n| n.starts_with(".manifest.json.tmp-")),
        "一時 manifest は残らない: {names:?}"
    );
    assert_eq!(
        std::fs::read(dir.join("manifest.json").join("inner")).expect("残存"),
        b"inner"
    );
    assert_eq!(
        safetensors_entries(&dir).len(),
        1,
        "孤立 safetensors は残る"
    );
}

/// `dir` 自身が symlink でも保存・復元できる（§13.3 第 2 行。許容）。
#[cfg(all(unix, any(target_os = "linux", target_os = "macos")))]
#[test]
fn save_and_load_through_symlinked_dir() {
    let g = TempDirGuard::new("dir-symlink");
    let real = g.path().join("real");
    std::fs::create_dir(&real).expect("mkdir");
    let link = g.path().join("link");
    std::os::unix::fs::symlink(&real, &link).expect("symlink");

    let m = model(3);
    save_model(&m, &link).expect("symlink 経由で保存できるはず");
    assert!(real.join("manifest.json").is_file());
    let loaded = load_model(&link).expect("symlink 経由で復元できるはず");
    assert_eq!(state_bits(&m), state_bits(&loaded));
}
