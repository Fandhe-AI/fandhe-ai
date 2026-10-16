//! `fandhe_ai::model::ModelRegistry::save`（イシュー #3119 で導入・本ファイルは #3120）の
//! 結合テスト。公開 API（`fandhe_ai`）と `std` だけを使う。
//!
//! 出典: `docs/facade-model-registry-decision.md` §16.4・§16.9・§17、`src/model.rs` の
//! `save` doc。往復（R1）・パス検証（R2）・障害時に既存が無傷で一時ファイルが残らないこと
//! （R3）を固定する。
//!
//! 単体テスト（`src/model.rs` の `mod tests`）との分担: 確定直前フックでの失敗注入・
//! 一時名衝突・`<name>`／`<version>` の途中差し替えは単体テストが担当する。本ファイルは
//! 公開 API だけで再現できる経路の補完であり、重複させない。
//!
//! 非 F32 入力は `save` の引数型 `HashMap<String, Tensor<f32>>` によりコンパイル時に拒否
//! される（`api_surface.rs::model_registry_signatures_are_pinned` が型を固定する）。
//! 実行時に再現できる検出範囲は「F16 ファイルの `load` 拒否と `save` による置き換え」に限る。
//!
//! シンボリックリンクの参照先は必ず各テストの一時ディレクトリ配下に置き、実ユーザーの
//! ファイルには触れない。chmod による失敗注入は root 実行で素通りして偽の green になる
//! ため使わない。

mod common;

// `save` の対応述語は `src/model.rs::ModelRegistry::save` の cfg と同一。変更時は同期する。
// `cfg(not)` 側の対のテストが、対応範囲のずれを該当プラットフォームで検知する。
#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
mod supported {
    use super::common::temp_dir::TempDirGuard;
    use fandhe_ai::Tensor;
    use fandhe_ai::compat::Sequential;
    use fandhe_ai::model::{ModelError, ModelRegistry};
    use std::collections::{BTreeMap, HashMap};
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::Path;

    type Bits = BTreeMap<String, (Vec<usize>, Vec<u32>)>;

    /// 形状の異なる複数テンソル。`seed` で値を変える。`view` は非連続（stride 0）。
    fn state(seed: u32) -> HashMap<String, Tensor<f32>> {
        let s = seed as f32;
        let mut m = HashMap::new();
        let a: Vec<f32> = (0..24).map(|i| s + i as f32 * 0.37 - 3.0).collect();
        m.insert("a".to_string(), Tensor::new(a, &[2, 3, 4]).unwrap());
        m.insert(
            "b".to_string(),
            Tensor::new(vec![s, -s, 0.5, f32::MIN_POSITIVE, 1e-30], &[5]).unwrap(),
        );
        m.insert("c".to_string(), Tensor::new(vec![s + 0.125], &[1]).unwrap());
        let base = Tensor::new(vec![s, s + 1.0, s + 2.0, s + 3.0], &[1, 4]).unwrap();
        m.insert("view".to_string(), base.broadcast_to(&[3, 4]).unwrap());
        m
    }

    fn bits(m: &HashMap<String, Tensor<f32>>) -> Bits {
        m.iter()
            .map(|(k, t)| {
                let c = t.contiguous();
                let v = c.as_slice().unwrap().iter().map(|x| x.to_bits()).collect();
                (k.clone(), (t.shape().to_vec(), v))
            })
            .collect()
    }

    fn tmp_leftovers(dir: &Path) -> Vec<String> {
        fn walk(dir: &Path, out: &mut Vec<String>) {
            let Ok(rd) = fs::read_dir(dir) else { return };
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().into_owned();
                if n.starts_with(".model.safetensors.tmp-") {
                    out.push(n);
                }
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    walk(&e.path(), out);
                }
            }
        }
        let mut out = Vec::new();
        walk(dir, &mut out);
        out
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    fn pairs(v: &[(&str, &[&str])]) -> Vec<(String, Vec<String>)> {
        v.iter()
            .map(|(n, vs)| (n.to_string(), vs.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    // ---- R1 往復 ----

    #[test]
    fn save_load_roundtrips_multiple_tensors_bit_exact() {
        let g = TempDirGuard::new("save-roundtrip");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let st = state(1);
        reg.save("mlp", "v1", &st).unwrap();
        let loaded = reg.load("mlp", "v1").unwrap();
        assert_eq!(bits(&st), bits(&loaded));
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    #[test]
    fn save_multiple_names_versions_and_list_sorted() {
        let g = TempDirGuard::new("save-multi");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let inputs = [
            ("mlp", "v2", state(2)),
            ("mlp", "v1", state(3)),
            ("cnn", "v1", state(4)),
        ];
        for (n, v, s) in &inputs {
            reg.save(n, v, s).unwrap();
        }
        assert_eq!(
            reg.available_models(),
            pairs(&[("cnn", &["v1"]), ("mlp", &["v1", "v2"])])
        );
        for (n, v, s) in &inputs {
            assert_eq!(bits(s), bits(&reg.load(n, v).unwrap()), "{n}/{v}");
        }
    }

    #[test]
    fn overwrite_replaces_only_target_version() {
        let g = TempDirGuard::new("save-overwrite");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let (v1, v2, cnn) = (state(10), state(11), state(12));
        reg.save("mlp", "v1", &v1).unwrap();
        reg.save("mlp", "v2", &v2).unwrap();
        reg.save("cnn", "v1", &cnn).unwrap();
        let before = reg.available_models();
        let newer = state(99);
        reg.save("mlp", "v1", &newer).unwrap();
        assert_eq!(bits(&newer), bits(&reg.load("mlp", "v1").unwrap()));
        assert_eq!(bits(&v2), bits(&reg.load("mlp", "v2").unwrap()));
        assert_eq!(bits(&cnn), bits(&reg.load("cnn", "v1").unwrap()));
        assert_eq!(before, reg.available_models());
        let md = fs::symlink_metadata(g.path().join("mlp/v1/model.safetensors")).unwrap();
        assert!(md.file_type().is_file());
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    #[test]
    fn sequential_state_dict_roundtrips_via_registry() {
        let build = |s1, s2| {
            Sequential::new()
                .add_linear(4, 8, s1)
                .unwrap()
                .add_relu()
                .add_linear(8, 2, s2)
                .unwrap()
        };
        let model = build(7, 8);
        let g = TempDirGuard::new("save-sequential");
        let reg = ModelRegistry::with_cache_dir(g.path());
        reg.save("seq", "v1", &model.state_dict()).unwrap();
        let mut other = build(90, 91);
        other
            .load_state_dict(reg.load("seq", "v1").unwrap())
            .unwrap();
        let x = Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, 4]).unwrap();
        let to_bits = |t: Tensor<f32>| -> Vec<u32> {
            t.contiguous()
                .as_slice()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(
            to_bits(model.predict(&x).unwrap()),
            to_bits(other.predict(&x).unwrap())
        );
    }

    // ---- R2 パス検証 ----

    #[test]
    fn save_rejects_invalid_components_before_fs_access() {
        let bad = [
            "",
            ".",
            "..",
            "...",
            "../x",
            "/tmp/x",
            "C:\\x",
            "a/b",
            "a\\b",
            "a:b",
            "a\0b",
            "a\tb",
            "a\rb",
            "a\nb",
            "a\x1bb",
            "a\x7fb",
            ".hidden",
            "モデル",
            "a b",
        ];
        let st = state(1);
        for v in bad {
            for (kind, name, version) in [("name", v, "v1"), ("version", "mlp", v)] {
                let g = TempDirGuard::new("save-invalid");
                let root = g.path().join("root");
                let reg = ModelRegistry::with_cache_dir(&root);
                match reg.save(name, version, &st) {
                    Err(ModelError::InvalidComponent { kind: k, .. }) => {
                        assert_eq!(k, kind, "{v:?}")
                    }
                    other => panic!("{kind}={v:?}: InvalidComponent を期待したが {other:?}"),
                }
                assert!(!root.exists(), "{kind}={v:?}: ルートが作られた");
            }
        }
    }

    /// 許可文字だけで NAME_MAX（255）を超える名前。`validate_component` は長さを見ないため
    /// 拒否の種類（InvalidComponent か OS 由来の Io）は固定せず、`Err` と無残骸だけを固定する。
    #[test]
    fn save_rejects_extremely_long_name_without_residue() {
        let g = TempDirGuard::new("save-long");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let long = "a".repeat(300);
        assert!(reg.save(&long, "v1", &state(1)).is_err());
        assert!(!g.path().join(&long).exists());
        assert!(reg.available_models().is_empty());
        assert!(tmp_leftovers(g.path()).is_empty());
        assert!(reg.save("mlp", &long, &state(1)).is_err());
        assert!(reg.available_models().is_empty());
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    fn symlink_rejected(link_at_name: bool, dangling: bool) {
        let g = TempDirGuard::new("save-symlink-dir");
        let root = g.path().join("root");
        let outside = g.path().join("outside");
        if !dangling {
            fs::create_dir(&outside).unwrap();
        }
        if link_at_name {
            fs::create_dir(&root).unwrap();
            symlink(&outside, root.join("mlp")).unwrap();
        } else {
            fs::create_dir_all(root.join("mlp")).unwrap();
            symlink(&outside, root.join("mlp/v1")).unwrap();
        }
        let reg = ModelRegistry::with_cache_dir(&root);
        match reg.save("mlp", "v1", &state(1)) {
            Err(ModelError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput),
            other => panic!("Io(InvalidInput) を期待したが {other:?}"),
        }
        if dangling {
            assert!(!outside.exists(), "dangling 参照先が作られた");
        } else {
            assert!(names(&outside).is_empty(), "リンク先へ書き込まれた");
        }
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    #[test]
    fn save_rejects_symlinked_name_dir_and_writes_nothing_outside() {
        symlink_rejected(true, false);
        symlink_rejected(true, true);
    }

    #[test]
    fn save_rejects_symlinked_version_dir_and_writes_nothing_outside() {
        symlink_rejected(false, false);
        symlink_rejected(false, true);
    }

    #[test]
    fn save_replaces_symlinked_leaf_without_writing_through() {
        let g = TempDirGuard::new("save-symlink-leaf");
        let root = g.path().join("root");
        let secret = g.path().join("secret.txt");
        fs::write(&secret, b"secret").unwrap();
        fs::create_dir_all(root.join("mlp/v1")).unwrap();
        let leaf = root.join("mlp/v1/model.safetensors");
        symlink(&secret, &leaf).unwrap();
        let reg = ModelRegistry::with_cache_dir(&root);
        let st = state(5);
        reg.save("mlp", "v1", &st).unwrap();
        assert_eq!(fs::read(&secret).unwrap(), b"secret");
        assert!(fs::symlink_metadata(&leaf).unwrap().file_type().is_file());
        assert_eq!(bits(&st), bits(&reg.load("mlp", "v1").unwrap()));
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    #[test]
    fn save_and_load_through_symlinked_root() {
        let g = TempDirGuard::new("save-symlink-root");
        let real = g.path().join("real");
        fs::create_dir(&real).unwrap();
        let link = g.path().join("link");
        symlink(&real, &link).unwrap();
        let reg = ModelRegistry::with_cache_dir(&link);
        let st = state(6);
        reg.save("mlp", "v1", &st).unwrap();
        assert!(real.join("mlp/v1/model.safetensors").is_file());
        assert_eq!(bits(&st), bits(&reg.load("mlp", "v1").unwrap()));
    }

    #[test]
    fn save_accepts_inner_dots_within_root() {
        let g = TempDirGuard::new("save-inner-dots");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let st = state(7);
        reg.save("a..b", "v1.0.0-rc_1", &st).unwrap();
        assert!(
            g.path()
                .join("a..b/v1.0.0-rc_1/model.safetensors")
                .is_file()
        );
        assert_eq!(names(g.path()), vec!["a..b".to_string()]);
        assert_eq!(bits(&st), bits(&reg.load("a..b", "v1.0.0-rc_1").unwrap()));
    }

    // ---- R3 障害 ----

    #[test]
    fn rename_failure_removes_own_tmp_and_keeps_other_models() {
        let g = TempDirGuard::new("save-rename-fail");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let v1 = state(1);
        reg.save("mlp", "v1", &v1).unwrap();
        // `model.safetensors` が中身入りディレクトリだと rename が失敗する（root でも成立）。
        let blocker = g.path().join("mlp/v2/model.safetensors");
        fs::create_dir_all(&blocker).unwrap();
        fs::write(blocker.join("inner"), b"keep").unwrap();
        assert!(matches!(
            reg.save("mlp", "v2", &state(2)),
            Err(ModelError::Io(_))
        ));
        assert!(tmp_leftovers(g.path()).is_empty());
        assert_eq!(fs::read(blocker.join("inner")).unwrap(), b"keep");
        assert_eq!(bits(&v1), bits(&reg.load("mlp", "v1").unwrap()));
        assert_eq!(reg.available_models(), pairs(&[("mlp", &["v1"])]));
    }

    #[test]
    fn oversize_is_rejected_before_fs_and_keeps_existing() {
        let g = TempDirGuard::new("save-oversize");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let v1 = state(1);
        reg.save("mlp", "v1", &v1).unwrap();
        // stride 0 view なので実メモリは確保されない。4 * numel > 1 GiB。
        let base = Tensor::new(vec![0.0_f32], &[1]).unwrap();
        let huge = base.broadcast_to(&[(1usize << 28) + 1]).unwrap();
        let big = HashMap::from([("w".to_string(), huge)]);
        match reg.save("mlp", "v1", &big) {
            Err(ModelError::TooLarge { max, len, .. }) => {
                assert_eq!(max, 1 << 30);
                assert!(len > max);
            }
            other => panic!("TooLarge を期待したが {other:?}"),
        }
        assert_eq!(bits(&v1), bits(&reg.load("mlp", "v1").unwrap()));
        assert!(tmp_leftovers(g.path()).is_empty());
        // 新規 name はディレクトリも作られない（概算はファイルシステム操作より前）。
        assert!(matches!(
            reg.save("fresh", "v1", &big),
            Err(ModelError::TooLarge { .. })
        ));
        assert!(!g.path().join("fresh").exists());
        // バイト数の u64 overflow は len = u64::MAX で上限超過扱い。
        let overflow = base.broadcast_to(&[1usize << 62]).unwrap();
        let ov = HashMap::from([("w".to_string(), overflow)]);
        match reg.save("fresh", "v1", &ov) {
            Err(ModelError::TooLarge { len, .. }) => assert_eq!(len, u64::MAX),
            other => panic!("TooLarge(u64::MAX) を期待したが {other:?}"),
        }
        assert!(!g.path().join("fresh").exists());
    }

    #[test]
    fn empty_model_is_rejected_and_keeps_existing() {
        let g = TempDirGuard::new("save-empty");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let v1 = state(1);
        reg.save("mlp", "v1", &v1).unwrap();
        assert!(matches!(
            reg.save("mlp", "v1", &HashMap::new()),
            Err(ModelError::EmptyModel { .. })
        ));
        assert_eq!(bits(&v1), bits(&reg.load("mlp", "v1").unwrap()));
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    /// 非 F32 は `save` の引数型で拒否される（コンパイル時）。実行時に再現できる経路として、
    /// F16 の safetensors を直接置いた場合の `load` 拒否と、`save` による原子的置き換えを確認する。
    #[test]
    fn non_f32_file_is_rejected_by_load_and_replaced_by_save() {
        let g = TempDirGuard::new("save-non-f32");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let v1 = state(1);
        reg.save("mlp", "v1", &v1).unwrap();
        let header = br#"{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header);
        file.extend_from_slice(&[0, 0]);
        let dir = g.path().join("mlp/v2");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), &file).unwrap();
        assert!(matches!(reg.load("mlp", "v2"), Err(ModelError::Load(_))));
        assert_eq!(bits(&v1), bits(&reg.load("mlp", "v1").unwrap()));
        let v2 = state(2);
        reg.save("mlp", "v2", &v2).unwrap();
        assert_eq!(bits(&v2), bits(&reg.load("mlp", "v2").unwrap()));
        assert!(tmp_leftovers(g.path()).is_empty());
    }

    /// エラー表示に一時ディレクトリの絶対パスが含まれない（OWASP A09）。
    #[test]
    fn save_error_display_has_no_path() {
        let g = TempDirGuard::new("save-display");
        let reg = ModelRegistry::with_cache_dir(g.path());
        let root = g.path().to_string_lossy().into_owned();
        let base = Tensor::new(vec![0.0_f32], &[1]).unwrap();
        let big = HashMap::from([(
            "w".to_string(),
            base.broadcast_to(&[(1usize << 28) + 1]).unwrap(),
        )]);
        let errs = [
            reg.save("..", "v1", &state(1)).unwrap_err(),
            reg.save("mlp", "v1", &HashMap::new()).unwrap_err(),
            reg.save("mlp", "v1", &big).unwrap_err(),
        ];
        for e in errs {
            let s = format!("{e} {e:?}");
            assert!(!s.contains(&root), "{s}");
        }
    }
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
#[test]
fn save_is_unsupported_and_creates_nothing_on_non_supported_platform() {
    use common::temp_dir::TempDirGuard;
    use fandhe_ai::Tensor;
    use fandhe_ai::model::{ModelError, ModelRegistry};
    use std::collections::HashMap;

    let g = TempDirGuard::new("save-unsupported");
    let root = g.path().join("root");
    let reg = ModelRegistry::with_cache_dir(&root);
    let st = HashMap::from([("w".to_string(), Tensor::new(vec![1.0_f32], &[1]).unwrap())]);
    match reg.save("mlp", "v1", &st) {
        Err(ModelError::Io(e)) => assert_eq!(e.kind(), std::io::ErrorKind::Unsupported),
        other => panic!("Io(Unsupported) を期待したが {other:?}"),
    }
    assert!(!root.exists());
}
