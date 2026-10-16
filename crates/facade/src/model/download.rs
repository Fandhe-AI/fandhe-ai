//! `ModelRegistry::download` の取得手順本体（イシュー #3125。`cfg(unix)`）。
//!
//! 呼び出し元は `model.rs` の `download_with` で、本モジュールは「検証済み URL → 手動
//! リダイレクト追従 → ヘッダ検査 → ストリーミング受信（サイズ上限・SHA-256・進捗）→
//! 検証 → dirfd 相対の atomic 公開」を 1 関数（[`run`]）に束ねる。転送は [`Transport`]
//! 越し、保存先ディレクトリ fd は遅延取得クロージャ越しにするため、単体テストはソケットも
//! 実キャッシュも使わずに全分岐を再現できる。公開点は `renameat` 1 回だけで、検証前の
//! ファイルを `model.safetensors` の名前で置く経路を作らない（A08。設計 §5・§6）。
//! 失敗時は自分が作った一時ファイルだけを `Drop` ガードで削除する（早期 return と
//! progress コールバック内 panic の unwind の両方を覆う）。

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::OwnedFd;
use std::time::Instant;

use super::dirfd;
use super::sha256::Sha256;
use super::transport::Transport;
use super::url::{ValidatedUrl, resolve_location};
use super::{DownloadProgress, MODEL_FILE_NAME, ModelError, Sha256Pin};

/// 受信バッファ長。
const CHUNK: usize = 64 * 1024;

/// 1 回の取得に固定で渡す入力（URL は検証済み）。
pub(super) struct Job<'a> {
    pub(super) url: ValidatedUrl,
    pub(super) name: &'a str,
    pub(super) version: &'a str,
    pub(super) max_bytes: u64,
    pub(super) max_redirects: u32,
    pub(super) deadline: Instant,
    pub(super) expected: Option<Sha256Pin>,
}

/// 自分が作った private staging ディレクトリと一時ファイルの `Drop` ガード。`commit` 済みで
/// なければ staging 内の一時ファイルを削除し、staging ディレクトリは常に除去を試みる。staging は
/// 0700 で作成者しか入れないため、名前による削除が他者の成果物を消すことはない。
struct TmpGuard<'a> {
    parent: &'a OwnedFd,
    staging: OwnedFd,
    staging_name: String,
    tmp_name: String,
    file: File,
    armed: bool,
}

impl Drop for TmpGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            dirfd::unlink_in(&self.staging, &self.tmp_name);
        }
        dirfd::remove_staging(self.parent, &self.staging_name);
    }
}

fn invalid_url(reason: &'static str) -> ModelError {
    ModelError::InvalidUrl { reason }
}

/// 取得を実行して `model.safetensors` を公開する。`open_dir` は応答ヘッダの検査を通った後に
/// 初めて呼ぶ（失敗する取得で `<name>/<version>` を作らないため）。`verify_before_commit` は
/// 公開直前の検証フック（#3126 が safetensors ヘッダ検証を差し込む位置。本 PR では空）。
pub(super) fn run(
    transport: &mut dyn Transport,
    job: Job<'_>,
    mut progress: Option<&mut dyn FnMut(DownloadProgress)>,
    open_dir: &mut dyn FnMut() -> std::io::Result<OwnedFd>,
    next_staging: &mut dyn FnMut() -> String,
    next_tmp: &mut dyn FnMut() -> String,
    verify_before_commit: &mut dyn FnMut(&File) -> Result<(), ModelError>,
) -> Result<(), ModelError> {
    let mut current = job.url.clone();
    let mut hops: u32 = 0;
    let resp = loop {
        let r = transport.get(&current, job.deadline)?;
        match r.status {
            200 => break r,
            301 | 302 | 303 | 307 | 308 => {
                if hops >= job.max_redirects {
                    return Err(ModelError::TooManyRedirects {
                        max: job.max_redirects,
                    });
                }
                let Some(loc) = r.location.as_deref() else {
                    return Err(invalid_url("リダイレクトに Location がありません"));
                };
                current = resolve_location(&current, loc)?;
                hops += 1;
            }
            status => return Err(ModelError::Http { status }),
        }
    };

    // 伸長を行わないため、圧縮されたボディではハッシュ・長さ検証が意味を失う。
    if let Some(enc) = resp.content_encoding.as_deref() {
        // 複数ヘッダ行は転送層が `,` 連結済み。全トークンが空または identity でなければ拒否する。
        if enc
            .split(',')
            .map(str::trim)
            .any(|t| !t.is_empty() && !t.eq_ignore_ascii_case("identity"))
        {
            return Err(ModelError::Network(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "圧縮されたボディは受け付けません",
            )));
        }
    }
    if let Some(len) = resp.content_length
        && len > job.max_bytes
    {
        return Err(ModelError::TooLarge {
            name: job.name.to_string(),
            version: job.version.to_string(),
            len,
            max: job.max_bytes,
        });
    }

    let dir = open_dir().map_err(ModelError::Io)?;
    let (staging, staging_name) =
        dirfd::create_staging(&dir, next_staging).map_err(ModelError::Io)?;
    let (file, tmp_name) = match dirfd::create_tmp(&staging, next_tmp) {
        Ok(v) => v,
        Err(e) => {
            dirfd::remove_staging(&dir, &staging_name);
            return Err(ModelError::Io(e));
        }
    };
    let mut guard = TmpGuard {
        parent: &dir,
        staging,
        staging_name,
        tmp_name,
        file,
        armed: true,
    };

    let mut body = resp.body;
    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = body.read(&mut buf).map_err(ModelError::Network)?;
        if n == 0 {
            break;
        }
        received = received.saturating_add(n as u64);
        if received > job.max_bytes {
            return Err(ModelError::TooLarge {
                name: job.name.to_string(),
                version: job.version.to_string(),
                len: received,
                max: job.max_bytes,
            });
        }
        hasher.update(&buf[..n]);
        guard.file.write_all(&buf[..n]).map_err(ModelError::Io)?;
        if let Some(cb) = progress.as_mut() {
            cb(DownloadProgress {
                received,
                total: resp.content_length,
            });
        }
    }
    drop(body);

    if let Some(expected) = resp.content_length
        && expected != received
    {
        return Err(ModelError::ContentLengthMismatch { expected, received });
    }
    let actual = Sha256Pin(hasher.finalize());
    if let Some(expected) = job.expected
        && expected != actual
    {
        return Err(ModelError::HashMismatch { expected, actual });
    }
    guard.file.sync_all().map_err(ModelError::Io)?;
    verify_before_commit(&guard.file)?;
    // 一時ファイルは 0700 の private staging 内にあり、作成者以外は名前を差し替えられない。
    // 検証した実体と公開する実体の一致は構造的に保証されるため、名前の再照合や公開後の
    // 撤去（別処理の成果物を消しうる）は行わない（A08）。公開は `renameat` 1 回のみ。
    dirfd::rename_across(&guard.staging, &guard.tmp_name, &dir, MODEL_FILE_NAME)
        .map_err(ModelError::Io)?;
    guard.armed = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::transport::RawResponse;
    use super::super::url::validate_https_url;
    use super::*;
    use std::collections::VecDeque;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    enum Step {
        Resp {
            status: u16,
            location: Option<&'static str>,
            content_length: Option<u64>,
            encoding: Option<&'static str>,
            body: Vec<u8>,
        },
        Err,
    }

    fn ok_body(body: &[u8]) -> Step {
        Step::Resp {
            status: 200,
            location: None,
            content_length: Some(body.len() as u64),
            encoding: None,
            body: body.to_vec(),
        }
    }

    fn redirect(status: u16, loc: Option<&'static str>) -> Step {
        Step::Resp {
            status,
            location: loc,
            content_length: Some(0),
            encoding: None,
            body: Vec::new(),
        }
    }

    struct Mock {
        steps: VecDeque<Step>,
        requested: Vec<String>,
    }

    impl Mock {
        fn new(steps: Vec<Step>) -> Self {
            Self {
                steps: steps.into(),
                requested: Vec::new(),
            }
        }
    }

    impl Transport for Mock {
        fn get(&mut self, url: &ValidatedUrl, _d: Instant) -> Result<RawResponse, ModelError> {
            self.requested.push(url.as_str().to_string());
            match self.steps.pop_front().expect("scripted step") {
                Step::Resp {
                    status,
                    location,
                    content_length,
                    encoding,
                    body,
                } => Ok(RawResponse {
                    status,
                    location: location.map(String::from),
                    content_length,
                    content_encoding: encoding.map(String::from),
                    body: Box::new(Cursor::new(body)),
                }),
                Step::Err => Err(ModelError::Network(std::io::Error::other("mock"))),
            }
        }
    }

    fn sha(data: &[u8]) -> Sha256Pin {
        let mut h = Sha256::new();
        h.update(data);
        Sha256Pin(h.finalize())
    }

    struct Env {
        root: PathBuf,
    }

    impl Env {
        fn new(tag: &str) -> Self {
            let (pid, nanos, counter) = crate::fs_write::name_entropy();
            let root =
                std::env::temp_dir().join(format!("fandhe-dl-{tag}-{pid}-{counter}-{nanos}"));
            std::fs::create_dir_all(&root).expect("mkdir");
            Self { root }
        }
        fn leaf(&self) -> PathBuf {
            self.root.join("m/v").join(MODEL_FILE_NAME)
        }
        fn leftovers(&self) -> Vec<String> {
            let d = self.root.join("m/v");
            match std::fs::read_dir(d) {
                Ok(rd) => rd
                    .flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| n != MODEL_FILE_NAME)
                    .collect(),
                Err(_) => Vec::new(),
            }
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn job(max_bytes: u64, max_redirects: u32, expected: Option<Sha256Pin>) -> Job<'static> {
        Job {
            url: validate_https_url("https://a.example/m").unwrap(),
            name: "m",
            version: "v",
            max_bytes,
            max_redirects,
            deadline: Instant::now() + Duration::from_secs(60),
            expected,
        }
    }

    fn exec(
        env: &Env,
        mock: &mut Mock,
        job: Job<'_>,
        progress: Option<&mut dyn FnMut(DownloadProgress)>,
    ) -> Result<(), ModelError> {
        let root = env.root.clone();
        let mut open = move || -> std::io::Result<OwnedFd> {
            let r = dirfd::open_root(&root)?;
            let n = dirfd::open_or_create_dir(&r, "m")?;
            dirfd::open_or_create_dir(&n, "v")
        };
        let mut stg = || ".s".to_string();
        let mut n = 0u32;
        let mut next = move || {
            n += 1;
            format!(".t{n}")
        };
        run(
            mock,
            job,
            progress,
            &mut open,
            &mut stg,
            &mut next,
            &mut |_: &File| Ok(()),
        )
    }

    #[test]
    fn success_publishes_file_and_verifies_pin() {
        let env = Env::new("ok");
        let data = b"hello model".to_vec();
        let mut mock = Mock::new(vec![ok_body(&data)]);
        exec(&env, &mut mock, job(1 << 20, 5, Some(sha(&data))), None).unwrap();
        assert_eq!(std::fs::read(env.leaf()).unwrap(), data);
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn success_without_pin() {
        let env = Env::new("nopin");
        let mut mock = Mock::new(vec![ok_body(b"abc")]);
        exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap();
        assert_eq!(std::fs::read(env.leaf()).unwrap(), b"abc");
    }

    #[test]
    fn hash_mismatch_publishes_nothing_and_keeps_existing() {
        let env = Env::new("hash");
        std::fs::create_dir_all(env.root.join("m/v")).unwrap();
        std::fs::write(env.leaf(), b"old").unwrap();
        let mut mock = Mock::new(vec![ok_body(b"new data")]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, Some(sha(b"other"))), None).unwrap_err();
        assert!(matches!(e, ModelError::HashMismatch { .. }));
        assert_eq!(std::fs::read(env.leaf()).unwrap(), b"old");
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn hash_mismatch_creates_no_model_file() {
        let env = Env::new("hash2");
        let mut mock = Mock::new(vec![ok_body(b"new data")]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, Some(sha(b"other"))), None).unwrap_err();
        assert!(matches!(e, ModelError::HashMismatch { .. }));
        assert!(!env.leaf().exists());
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn content_length_mismatch_is_detected_and_cleaned() {
        for declared in [3u64, 100] {
            let env = Env::new("cl");
            let mut mock = Mock::new(vec![Step::Resp {
                status: 200,
                location: None,
                content_length: Some(declared),
                encoding: None,
                body: b"abcdef".to_vec(),
            }]);
            let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
            // 宣言が小さい場合も受信は 6 バイトまで進み、不一致として検出される。
            assert!(
                matches!(e, ModelError::ContentLengthMismatch { .. }),
                "declared={declared}"
            );
            assert!(!env.leaf().exists());
            assert!(env.leftovers().is_empty());
        }
    }

    #[test]
    fn declared_length_over_limit_rejected_before_reading_body() {
        let env = Env::new("big");
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: Some(1000),
            encoding: None,
            body: vec![0; 10],
        }]);
        let e = exec(&env, &mut mock, job(100, 5, None), None).unwrap_err();
        assert!(matches!(
            e,
            ModelError::TooLarge {
                len: 1000,
                max: 100,
                ..
            }
        ));
        // ボディ前に拒否するため `<name>/<version>` も作らない。
        assert!(!env.root.join("m").exists());
    }

    #[test]
    fn chunked_body_over_limit_is_cut_off() {
        let env = Env::new("chunked");
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: None,
            encoding: None,
            body: vec![1; 200],
        }]);
        let e = exec(&env, &mut mock, job(100, 5, None), None).unwrap_err();
        assert!(matches!(e, ModelError::TooLarge { max: 100, .. }));
        assert!(!env.leaf().exists());
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn follows_https_redirect_and_revalidates_location() {
        let env = Env::new("redir");
        let mut mock = Mock::new(vec![
            redirect(302, Some("https://cdn.example/x#f")),
            redirect(307, Some("/y")),
            ok_body(b"data"),
        ]);
        exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap();
        assert_eq!(
            mock.requested,
            vec![
                "https://a.example/m",
                "https://cdn.example/x",
                "https://cdn.example/y"
            ]
        );
    }

    #[test]
    fn bad_redirects_are_invalid_url() {
        for loc in [
            Some("http://b.example/x"),
            Some("https://u:p@b.example/x"),
            Some("../x"),
            None,
        ] {
            let env = Env::new("badredir");
            let mut mock = Mock::new(vec![redirect(302, loc)]);
            let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
            assert!(matches!(e, ModelError::InvalidUrl { .. }), "loc={loc:?}");
            assert_eq!(mock.requested.len(), 1);
        }
    }

    #[test]
    fn too_many_redirects() {
        let env = Env::new("many");
        let mut mock = Mock::new(vec![
            redirect(302, Some("/a")),
            redirect(302, Some("/b")),
            redirect(302, Some("/c")),
        ]);
        let e = exec(&env, &mut mock, job(1 << 20, 2, None), None).unwrap_err();
        assert!(matches!(e, ModelError::TooManyRedirects { max: 2 }));
        assert_eq!(mock.requested.len(), 3);
    }

    #[test]
    fn non_200_statuses_are_http_errors() {
        for status in [404u16, 500, 304, 206, 204, 401] {
            let env = Env::new("status");
            let mut mock = Mock::new(vec![redirect(status, None)]);
            let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
            assert!(
                matches!(e, ModelError::Http { status: s } if s == status),
                "status={status}"
            );
        }
    }

    #[test]
    fn content_encoding_is_rejected_but_identity_allowed() {
        let env = Env::new("enc");
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: Some(3),
            encoding: Some("gzip"),
            body: b"abc".to_vec(),
        }]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
        assert!(matches!(e, ModelError::Network(_)));
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: Some(3),
            encoding: Some("identity"),
            body: b"abc".to_vec(),
        }]);
        exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap();
        // 別行で届いた gzip（転送層が連結した値）も拒否する。
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: Some(3),
            encoding: Some("identity,gzip"),
            body: b"abc".to_vec(),
        }]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
        assert!(matches!(e, ModelError::Network(_)));
    }

    #[test]
    fn transport_error_propagates() {
        let env = Env::new("terr");
        let mut mock = Mock::new(vec![Step::Err]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
        assert!(matches!(e, ModelError::Network(_)));
    }

    #[test]
    fn progress_is_monotonic_and_reports_total() {
        let env = Env::new("prog");
        let data = vec![7u8; CHUNK * 2 + 5];
        let mut mock = Mock::new(vec![ok_body(&data)]);
        let mut seen: Vec<(u64, Option<u64>)> = Vec::new();
        let mut cb = |p: DownloadProgress| seen.push((p.received, p.total));
        exec(&env, &mut mock, job(1 << 30, 5, None), Some(&mut cb)).unwrap();
        assert!(seen.len() >= 3);
        assert!(seen.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(seen.last().unwrap().0, data.len() as u64);
        assert!(seen.iter().all(|(_, t)| *t == Some(data.len() as u64)));
    }

    #[test]
    fn panic_in_progress_callback_removes_tmp() {
        let env = Env::new("panic");
        let mut mock = Mock::new(vec![ok_body(b"abc")]);
        let mut cb = |_: DownloadProgress| panic!("boom");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = exec(&env, &mut mock, job(1 << 20, 5, None), Some(&mut cb));
        }));
        assert!(r.is_err());
        assert!(!env.leaf().exists());
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn verify_hook_failure_removes_tmp_and_publishes_nothing() {
        let env = Env::new("verify");
        let mut mock = Mock::new(vec![ok_body(b"abc")]);
        let root = env.root.clone();
        let mut open = move || -> std::io::Result<OwnedFd> {
            let r = dirfd::open_root(&root)?;
            let n = dirfd::open_or_create_dir(&r, "m")?;
            dirfd::open_or_create_dir(&n, "v")
        };
        let mut next = || ".t".to_string();
        let mut stg = || ".s".to_string();
        let e = run(
            &mut mock,
            job(1 << 20, 5, None),
            None,
            &mut open,
            &mut stg,
            &mut next,
            &mut |_: &File| Err(ModelError::Io(std::io::Error::other("verify"))),
        )
        .unwrap_err();
        assert!(matches!(e, ModelError::Io(_)));
        assert!(!env.leaf().exists());
        assert!(env.leftovers().is_empty());
        let _ = Path::new("");
    }

    #[test]
    fn staged_tmp_is_in_private_dir_and_failure_keeps_published_model() {
        use std::os::unix::fs::PermissionsExt;
        let env = Env::new("stage");
        std::fs::create_dir_all(env.root.join("m/v")).unwrap();
        std::fs::write(env.leaf(), b"old").unwrap();
        let mut mock = Mock::new(vec![ok_body(b"abc")]);
        let root = env.root.clone();
        let mut open = {
            let root = root.clone();
            move || -> std::io::Result<OwnedFd> {
                let r = dirfd::open_root(&root)?;
                let n = dirfd::open_or_create_dir(&r, "m")?;
                dirfd::open_or_create_dir(&n, "v")
            }
        };
        let mut stg = || ".s".to_string();
        let mut next = || ".t".to_string();
        let e = run(
            &mut mock,
            job(1 << 20, 5, None),
            None,
            &mut open,
            &mut stg,
            &mut next,
            &mut |_: &File| {
                // 一時ファイルは共有ディレクトリ直下ではなく 0700 の staging 内にだけ存在する。
                let v = root.join("m/v");
                assert!(!v.join(".t").exists());
                assert!(v.join(".s/.t").is_file());
                let mode = std::fs::metadata(v.join(".s"))
                    .unwrap()
                    .permissions()
                    .mode();
                assert_eq!(mode & 0o077, 0);
                Err(ModelError::Io(std::io::Error::other("verify")))
            },
        )
        .unwrap_err();
        assert!(matches!(e, ModelError::Io(_)));
        // 失敗時は既存の公開済みモデルに触れず、staging も残さない。
        assert_eq!(std::fs::read(env.leaf()).unwrap(), b"old");
        assert!(env.leftovers().is_empty());
    }
}
