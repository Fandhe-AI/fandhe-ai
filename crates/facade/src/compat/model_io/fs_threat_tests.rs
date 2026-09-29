//! `save_model` の書き込み経路（`write_prepared_with`）に対する衝突注入・中断ウィンドウの
//! 単体テスト（決定記録 `docs/compat-model-io-decision.md` §6 後半・§12.3 手順 1〜2・§13.3。
//! イシュー #2376）。`model_io.rs` の `mod fs_threat_tests;` で取り込まれる unix 限定の子モジュール。
//!
//! 世代 ID・一時 manifest 名の生成器を差し替え、事前配置した衝突エントリ（有効／dangling
//! symlink・通常ファイル・ディレクトリ）が「触れられずに」別名で成功する、または全候補衝突で
//! `Err` になり既存エントリが完全に不変であることを固定する。公開面（`api_surface.rs` の
//! 承認済み項目）は増やさず、ここのヘルパーはすべて非公開。

use super::*;

use std::collections::BTreeMap;
use std::path::PathBuf;

/// 排他作成した一時ディレクトリの RAII ガード（drop で削除。`remove_dir_all` は symlink を辿らない）。
struct Tmp(PathBuf);

impl Tmp {
    fn new(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!(
            "fandhe-ai-model-io-threat-{}-{nanos}-{label}",
            std::process::id()
        ));
        std::fs::create_dir(&p).expect("一時ディレクトリを作成できるはず");
        Self(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// エントリ 1 件の観測結果（種別と、通常ファイルなら内容・symlink なら参照先）。
#[derive(Debug, PartialEq, Eq)]
enum Entry {
    File(Vec<u8>),
    Symlink(PathBuf),
    Dir,
}

/// `dir` 直下の全エントリを名前順に写す（symlink は追跡しない）。
fn snapshot(dir: &Path) -> BTreeMap<String, Entry> {
    let mut map = BTreeMap::new();
    for e in std::fs::read_dir(dir).expect("読めるはず").flatten() {
        let path = e.path();
        let meta = std::fs::symlink_metadata(&path).expect("メタデータを読めるはず");
        let entry = if meta.file_type().is_symlink() {
            Entry::Symlink(std::fs::read_link(&path).expect("read_link できるはず"))
        } else if meta.is_dir() {
            Entry::Dir
        } else {
            Entry::File(std::fs::read(&path).expect("読めるはず"))
        };
        map.insert(e.file_name().to_string_lossy().into_owned(), entry);
    }
    map
}

fn gen_name(n: u128) -> String {
    format!("{n:032x}")
}

fn st_name(n: u128) -> String {
    format!("model.{}.safetensors", gen_name(n))
}

fn sample_prepared() -> PreparedSave {
    prepare_save(&tests::sample_model_for_threats()).expect("検証を通るはず")
}

fn manifest_target(dir: &Path) -> String {
    let bytes = std::fs::read(dir.join(MANIFEST_FILE_NAME)).expect("manifest を読めるはず");
    parse_manifest(&bytes)
        .expect("manifest を解釈できるはず")
        .safetensors_file
}

/// 一部候補が衝突しても、衝突エントリに触れずに別世代名で成功する（受入基準 R3）。
#[test]
fn save_retries_past_colliding_generation_candidates() {
    let tmp = Tmp::new("gen-partial");
    let dir = &tmp.0;
    let outside = dir.join("outside.bin");
    std::fs::write(&outside, b"outside").expect("書き込めるはず");
    let dangling_target = dir.join("never-created");
    std::os::unix::fs::symlink(&outside, dir.join(st_name(1))).expect("symlink");
    std::os::unix::fs::symlink(&dangling_target, dir.join(st_name(2))).expect("symlink");
    std::fs::write(dir.join(st_name(3)), b"keep").expect("書き込めるはず");
    let before = snapshot(dir);

    let mut n = 0u128;
    write_prepared_with(
        dir,
        &sample_prepared(),
        || {
            n += 1;
            gen_name(n)
        },
        tmp_manifest_name,
    )
    .expect("4 番目の候補で成功するはず");

    assert_eq!(manifest_target(dir), st_name(4));
    let after = snapshot(dir);
    for k in [1u128, 2, 3] {
        assert_eq!(
            before[&st_name(k)],
            after[&st_name(k)],
            "衝突エントリ {k} は不変"
        );
    }
    assert_eq!(std::fs::read(&outside).expect("読めるはず"), b"outside");
    assert!(!dangling_target.exists(), "dangling の参照先は作られない");
}

/// 一時 manifest 名の一部候補が衝突しても、既存エントリに触れずに成功する（R3）。
#[test]
fn save_retries_past_colliding_tmp_manifest_candidates() {
    let tmp = Tmp::new("tmp-partial");
    let dir = &tmp.0;
    let outside = dir.join("outside.bin");
    std::fs::write(&outside, b"outside").expect("書き込めるはず");
    let dangling_target = dir.join("never-created");
    let tmp_name = |n: u32| format!(".manifest.json.tmp-t{n}");
    std::os::unix::fs::symlink(&outside, dir.join(tmp_name(1))).expect("symlink");
    std::os::unix::fs::symlink(&dangling_target, dir.join(tmp_name(2))).expect("symlink");
    std::fs::create_dir(dir.join(tmp_name(3))).expect("mkdir");
    let before = snapshot(dir);

    let mut n = 0u32;
    write_prepared_with(
        dir,
        &sample_prepared(),
        || gen_name(9),
        || {
            n += 1;
            tmp_name(n)
        },
    )
    .expect("4 番目の候補で成功するはず");

    let after = snapshot(dir);
    for k in [1u32, 2, 3] {
        assert_eq!(
            before[&tmp_name(k)],
            after[&tmp_name(k)],
            "衝突エントリ {k} は不変"
        );
    }
    assert!(
        !after.contains_key(&tmp_name(4)),
        "rename 済みで一時名は残らない"
    );
    assert!(matches!(after[MANIFEST_FILE_NAME], Entry::File(_)));
    assert_eq!(std::fs::read(&outside).expect("読めるはず"), b"outside");
    assert!(!dangling_target.exists());
}

/// 世代 ID の全候補が衝突したら `Err(AlreadyExists)` で、ディレクトリは完全に不変（R4）。
#[test]
fn save_gives_up_when_every_generation_candidate_collides() {
    let tmp = Tmp::new("gen-all");
    let dir = &tmp.0;
    write_prepared(dir, &sample_prepared()).expect("事前保存できるはず");
    let outside = dir.join("outside.bin");
    std::fs::write(&outside, b"outside").expect("書き込めるはず");
    std::os::unix::fs::symlink(&outside, dir.join(st_name(7))).expect("symlink");
    let before = snapshot(dir);

    let mut calls = 0usize;
    let err = write_prepared_with(
        dir,
        &sample_prepared(),
        || {
            calls += 1;
            gen_name(7)
        },
        tmp_manifest_name,
    )
    .expect_err("全候補が衝突すれば失敗するはず");
    assert!(matches!(err, ModelIoError::Io(e) if e.kind() == io::ErrorKind::AlreadyExists));
    assert_eq!(calls, MAX_TMP_NAME_ATTEMPTS);
    assert_eq!(before, snapshot(dir), "新規エントリ 0 件・既存は不変");
}

/// 一時 manifest 名の全候補が衝突したら失敗し、旧コミットは無傷で読める。
/// 残るのは §12.3 手順 1 で受容済みの孤立 safetensors ちょうど 1 件（R4・R7 中断）。
#[test]
fn save_gives_up_when_every_tmp_candidate_collides_and_old_manifest_stays_committed() {
    let tmp = Tmp::new("tmp-all");
    let dir = &tmp.0;
    let old = tests::sample_model_for_threats();
    let old_prepared = prepare_save(&old).expect("検証を通るはず");
    write_prepared(dir, &old_prepared).expect("事前保存できるはず");
    let colliding = ".manifest.json.tmp-fixed";
    let outside = dir.join("outside.bin");
    std::fs::write(&outside, b"outside").expect("書き込めるはず");
    std::os::unix::fs::symlink(&outside, dir.join(colliding)).expect("symlink");
    let before = snapshot(dir);

    let mut calls = 0usize;
    let err = write_prepared_with(dir, &sample_prepared(), generation_id, || {
        calls += 1;
        colliding.to_string()
    })
    .expect_err("全候補が衝突すれば失敗するはず");
    assert!(matches!(err, ModelIoError::Io(e) if e.kind() == io::ErrorKind::AlreadyExists));
    assert_eq!(calls, MAX_TMP_NAME_ATTEMPTS);

    let after = snapshot(dir);
    for (k, v) in &before {
        assert_eq!(Some(v), after.get(k), "既存エントリ {k} は不変");
    }
    let added: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
    assert_eq!(added.len(), 1, "追加は孤立 safetensors のみ: {added:?}");
    assert!(is_valid_safetensors_file_name(added[0]));

    #[cfg(any(
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        target_os = "macos"
    ))]
    {
        let loaded = load_from_dir_with_limits(dir, MAX_MANIFEST_BYTES, MAX_MODEL_FILE_BYTES)
            .expect("コミット点に未到達なら旧モデルを読めるはず");
        let (a, b) = (old.state_dict(), loaded.state_dict());
        assert_eq!(a.len(), b.len());
        for (k, t) in &a {
            let (x, y) = (t.contiguous(), b[k].contiguous());
            assert!(
                x.as_slice()
                    .expect("連続のはず")
                    .iter()
                    .zip(y.as_slice().expect("連続のはず"))
                    .all(|(p, q)| p.to_bits() == q.to_bits()),
                "{k}: 旧モデルと bit 一致するはず"
            );
        }
    }
}
