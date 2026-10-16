//! HF Hub 実通信テスト（イシュー #3132。すべて `#[ignore]`・CI では実行しない）。
//!
//! 役割: `HfHub::download` の端から端まで（HTTPS 取得・sha256 pin 検証・キャッシュ配置）を
//! 実際の huggingface.co に対して確認する。facade 側の取得経路はモック化できない
//! （転送差し替え口が facade 内部限定のため）ので、本クレートでの結合確認はここだけが担う。
//! `resolve_revision`／`list_files` は本番転送層 `UreqJsonTransport`（#3393・`cfg(unix)`）経由の
//! 実通信テストを末尾に置く（`live_resolve_main_and_list_files`）。
//!
//! 入力（環境変数。公開 repo のみ。値はログへ出さない）:
//! - `FANDHE_AI_HF_TEST_REPO`: 直下に `model.safetensors` を持つ公開 repo_id
//! - `FANDHE_AI_HF_TEST_COMMIT`: その commit sha（40 桁小文字 16 進）
//! - `FANDHE_AI_HF_TEST_SHA256`: 当該ファイルの sha256（64 桁 16 進）
//!
//! 本クレートは認証トークンを読まない・送らない。`--ignored` で明示実行したのに入力が無い場合は
//! 黙って成功させず panic する（false-green 防止）。

use std::path::PathBuf;

use fandhe_ai::model::{DownloadOptions, ModelError, ModelRegistry, Sha256Pin};

use crate::{CommitSha, HfError, HfHub, HfRepoId};

struct Inputs {
    repo: HfRepoId,
    commit: CommitSha,
    sha256: String,
}

fn inputs() -> Inputs {
    let get = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let (Some(r), Some(c), Some(h)) = (
        get("FANDHE_AI_HF_TEST_REPO"),
        get("FANDHE_AI_HF_TEST_COMMIT"),
        get("FANDHE_AI_HF_TEST_SHA256"),
    ) else {
        panic!(
            "実通信テストには FANDHE_AI_HF_TEST_REPO / FANDHE_AI_HF_TEST_COMMIT / \
             FANDHE_AI_HF_TEST_SHA256 の設定が必要（crates/hf-hub/README.md 参照）"
        );
    };
    Inputs {
        repo: HfRepoId::parse(&r).expect("FANDHE_AI_HF_TEST_REPO が不正"),
        commit: CommitSha::parse(&c).expect("FANDHE_AI_HF_TEST_COMMIT が不正"),
        sha256: h,
    }
}

/// プロセス ID 付きの一時キャッシュ。Drop で成否に関わらず削除する。
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        Self(std::env::temp_dir().join(format!("fandhe-ai-hf-live-{tag}-{}", std::process::id())))
    }

    fn hub(&self) -> HfHub {
        HfHub::new(ModelRegistry::with_cache_dir(self.0.clone()))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn flip_first_nibble(hex: &str) -> String {
    let mut c: Vec<char> = hex.chars().collect();
    c[0] = if c[0] == '0' { '1' } else { '0' };
    c.into_iter().collect()
}

#[test]
#[ignore = "HF Hub 実通信（huggingface.co）。CI では実行しない"]
fn live_download_with_correct_pin_succeeds() {
    let i = inputs();
    let s = Scratch::new("ok");
    let pin = Sha256Pin::from_hex(&i.sha256).unwrap();
    let m = s
        .hub()
        .download(
            &i.repo,
            &i.commit,
            "model.safetensors",
            pin,
            DownloadOptions::default(),
        )
        .expect("正しい pin で取得できる");
    assert_eq!(m.version(), i.commit.as_str());
    assert_eq!(
        m.name(),
        format!("hf--{}--{}", i.repo.namespace(), i.repo.name())
    );
    assert!(
        s.0.join(m.name())
            .join(m.version())
            .join(m.filename())
            .is_file()
    );
}

#[test]
#[ignore = "HF Hub 実通信（huggingface.co）。CI では実行しない"]
fn live_download_with_wrong_pin_fails_and_publishes_nothing() {
    let i = inputs();
    let s = Scratch::new("bad");
    let pin = Sha256Pin::from_hex(&flip_first_nibble(&i.sha256)).unwrap();
    let r = s.hub().download(
        &i.repo,
        &i.commit,
        "model.safetensors",
        pin,
        DownloadOptions::default(),
    );
    assert!(matches!(
        r,
        Err(HfError::Model(ModelError::HashMismatch { .. }))
    ));
    let name = format!("hf--{}--{}", i.repo.namespace(), i.repo.name());
    assert!(
        !s.0.join(name)
            .join(i.commit.as_str())
            .join("model.safetensors")
            .exists()
    );
}

#[test]
#[ignore = "HF Hub 実通信（huggingface.co）。CI では実行しない"]
fn live_argument_pin_overrides_options_pin() {
    let i = inputs();
    let s = Scratch::new("prio");
    let good = Sha256Pin::from_hex(&i.sha256).unwrap();
    let wrong = Sha256Pin::from_hex(&flip_first_nibble(&i.sha256)).unwrap();
    let opts = DownloadOptions::default().expected_sha256(wrong);
    s.hub()
        .download(&i.repo, &i.commit, "model.safetensors", good, opts)
        .expect("引数の pin が優先される");
}

#[test]
#[ignore = "HF Hub 実通信（huggingface.co）。CI では実行しない"]
fn live_nonexistent_repo_fails_closed_without_token() {
    // 環境変数不要（ネットワークのみ）。トークンは送らないので 401 か 404 になる。
    let s = Scratch::new("none");
    let repo = HfRepoId::parse("fandhe-ai-nonexistent-ns-0/nonexistent-model-0").unwrap();
    let commit = CommitSha::parse(&"0".repeat(40)).unwrap();
    let pin = Sha256Pin::from_hex(&"0".repeat(64)).unwrap();
    let r = s.hub().download(
        &repo,
        &commit,
        "model.safetensors",
        pin,
        DownloadOptions::default(),
    );
    match r {
        Err(HfError::Model(ModelError::Http { status })) => assert!(matches!(status, 401 | 404)),
        other => panic!(
            "fail-closed のはずが {:?}",
            other.map(|m| m.version().to_owned())
        ),
    }
}

#[cfg(unix)]
#[test]
#[ignore = "HF Hub 実通信（huggingface.co）。CI では実行しない"]
fn live_resolve_main_and_list_files() {
    // 環境変数の repo を使う（commit・sha256 は不要）。`main` の sha 解決とファイル一覧を確認する。
    let repo = HfRepoId::parse(
        &std::env::var("FANDHE_AI_HF_TEST_REPO")
            .ok()
            .filter(|v| !v.is_empty())
            .expect("FANDHE_AI_HF_TEST_REPO の設定が必要（README 参照）"),
    )
    .expect("FANDHE_AI_HF_TEST_REPO が不正");
    let mut t = crate::transport::UreqJsonTransport::new().expect("transport");
    let sha = crate::api::resolve_revision_via(&mut t, &repo, &crate::Revision::main())
        .expect("main を解決できる");
    assert_eq!(sha.as_str().len(), 40);
    let files = crate::api::list_files_via(&mut t, &repo, &sha).expect("一覧を取得できる");
    assert!(!files.is_empty());
}
