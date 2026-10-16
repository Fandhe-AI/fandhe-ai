//! `ModelRegistry::download` の取得手順本体（イシュー #3125・#3126。`cfg(unix)`）。
//!
//! 呼び出し元は `model.rs` の `download_with` で、本モジュールは「検証済み URL → 条件付き GET
//! （キャッシュあり時のみ）→ 手動リダイレクト追従 → ヘッダ検査 → ストリーミング受信
//! （サイズ上限・SHA-256・進捗）→ 検証 → dirfd 相対の atomic 公開 → `download.json` 公開」を
//! 1 関数（[`run`]）に束ねる。転送は [`Transport`] 越し、保存先ディレクトリ fd は遅延取得
//! クロージャ越しにするため、単体テストはソケットも実キャッシュも使わずに全分岐を再現できる。
//! モデル本体の公開点は `renameat` 1 回だけで、検証前のファイルを `model.safetensors` の名前で
//! 置く経路を作らない（A08。設計 §5・§6）。失敗時は自分が作った一時ファイルだけを `Drop`
//! ガードで削除する（早期 return と progress コールバック内 panic の unwind の両方を覆う）。
//!
//! `304` は無条件の再利用シグナルにしない（#3126。§5）: 条件付きで送ったときだけ受理し、既存
//! `model.safetensors` の SHA-256 を計算し直して pin（なければ保存値）と照合する。一致しなければ
//! 条件なしの GET で全量を取り直す。公開順序は「モデル本体 → `download.json`」で、間で止まっても
//! 古い JSON の SHA が合わず次回は取り直しになるため自然に回復する。

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::OwnedFd;
use std::time::Instant;

use super::cache::{self, CacheMeta};
use super::dirfd;
use super::sha256::Sha256;
use super::transport::{RawResponse, Transport};
use super::url::{ValidatedUrl, resolve_location};
use super::{DownloadProgress, MODEL_FILE_NAME, ModelError, Sha256Pin, map_fs_error};

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
    /// 条件付き GET の根拠になる既存キャッシュ（`url` 一致・etag あり・pin 整合を呼び出し側が
    /// 確認済みのものだけ。`None` なら常に無条件 GET）。
    pub(super) cache: Option<CacheMeta>,
}

/// `run` が必要とする副作用の注入口（保存先 fd・一時名・検証・時刻）。単体テストが実キャッシュ
/// なしで全分岐を再現できるようにまとめている。
pub(super) struct Hooks<'a> {
    /// 既存の `<name>/<version>` を**作らずに**開く（`304` 再検証用。不在は `None`）。
    pub(super) lookup_dir: &'a mut dyn FnMut() -> std::io::Result<Option<OwnedFd>>,
    /// 保存先を（無ければ作って）開く。応答ヘッダの検査後にだけ呼ぶ。
    pub(super) open_dir: &'a mut dyn FnMut() -> std::io::Result<OwnedFd>,
    pub(super) next_staging: &'a mut dyn FnMut() -> String,
    pub(super) next_tmp: &'a mut dyn FnMut() -> String,
    /// 公開直前の検証フック（safetensors レイアウト検証。方式 (ii)）。
    pub(super) verify_before_commit: &'a mut dyn FnMut(&File) -> Result<(), ModelError>,
    /// `fetched_at` 用の UNIX エポック秒。
    pub(super) now: &'a mut dyn FnMut() -> u64,
}

/// 自分が作った private staging ディレクトリと一時ファイルの `Drop` ガード。`commit` 済みで
/// なければ staging 内の一時ファイルを削除し、staging ディレクトリは常に除去を試みる。staging は
/// 0700 で作成者しか入れないため、名前による削除が他者の成果物を消すことはない。`cache.rs` の
/// `download.json` 公開も同じ形で共用する。
pub(super) struct TmpGuard<'a> {
    parent: &'a OwnedFd,
    pub(super) staging: OwnedFd,
    staging_name: String,
    pub(super) tmp_name: String,
    pub(super) file: File,
    pub(super) armed: bool,
}

impl<'a> TmpGuard<'a> {
    /// `parent` 直下に private staging と一時ファイルを作る。失敗時は作った staging を除去する。
    pub(super) fn create(
        parent: &'a OwnedFd,
        next_staging: &mut dyn FnMut() -> String,
        next_tmp: &mut dyn FnMut() -> String,
    ) -> std::io::Result<Self> {
        let (staging, staging_name) = dirfd::create_staging(parent, next_staging)?;
        let (file, tmp_name) = match dirfd::create_tmp(&staging, next_tmp) {
            Ok(v) => v,
            Err(e) => {
                dirfd::remove_staging(parent, &staging_name);
                return Err(e);
            }
        };
        Ok(Self {
            parent,
            staging,
            staging_name,
            tmp_name,
            file,
            armed: true,
        })
    }
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

/// 1 回のリダイレクト追従の結果。
enum Fetched {
    /// `200`。ボディを受信する。
    Full(RawResponse),
    /// 条件付き GET への `304`。
    NotModified,
}

/// リダイレクトを追従して最終応答を得る。`etag` が `Some` のときは全ホップに `If-None-Match` を
/// 送り、`304` を受理する（条件付きで送っていない `304` は異常として `Http{304}`）。
fn fetch(
    transport: &mut dyn Transport,
    job: &Job<'_>,
    etag: Option<&str>,
) -> Result<Fetched, ModelError> {
    let mut current = job.url.clone();
    let mut hops: u32 = 0;
    loop {
        let r = transport.get(&current, job.deadline, etag)?;
        match r.status {
            200 => return Ok(Fetched::Full(r)),
            304 if etag.is_some() => return Ok(Fetched::NotModified),
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
    }
}

/// `304` の再検証。既存 `model.safetensors` の SHA-256 を計算し直し、pin（なければ保存値）と
/// 一致したときだけ `fetched_at` を更新した `download.json` を公開して `Ok(true)`。一致しない・
/// 読めない場合は `Ok(false)`（呼び出し側が無条件 GET で取り直す）。モデル本体には触れない。
fn revalidate(
    job: &Job<'_>,
    cached: &CacheMeta,
    hooks: &mut Hooks<'_>,
) -> Result<bool, ModelError> {
    let want = job.expected.unwrap_or(cached.sha256);
    let Some(dir) = (hooks.lookup_dir)().ok().flatten() else {
        return Ok(false);
    };
    if cache::rehash(&dir, job.max_bytes) != Some(want) {
        return Ok(false);
    }
    let mut meta = cached.clone();
    meta.fetched_at = (hooks.now)();
    cache::publish(&dir, &meta, &mut *hooks.next_staging, &mut *hooks.next_tmp)?;
    Ok(true)
}

/// 取得を実行して `model.safetensors` を公開する。`open_dir` は応答ヘッダの検査を通った後に
/// 初めて呼ぶ（失敗する取得で `<name>/<version>` を作らないため）。
pub(super) fn run(
    transport: &mut dyn Transport,
    job: Job<'_>,
    mut progress: Option<&mut dyn FnMut(DownloadProgress)>,
    mut hooks: Hooks<'_>,
) -> Result<(), ModelError> {
    // 条件付き GET は、検証済みの etag を持つキャッシュがあるときだけ。再検証に失敗したら
    // 1 回だけ無条件でやり直す（やり直しの `304` は `Http{304}`）。
    let cached = job
        .cache
        .as_ref()
        .filter(|c| c.etag.as_deref().is_some_and(cache::valid_etag));
    let resp = match fetch(transport, &job, cached.and_then(|c| c.etag.as_deref()))? {
        Fetched::Full(r) => r,
        Fetched::NotModified => {
            // `NotModified` は `cached` が `Some` のときだけ返る。
            if let Some(c) = cached
                && revalidate(&job, c, &mut hooks)?
            {
                return Ok(());
            }
            match fetch(transport, &job, None)? {
                Fetched::Full(r) => r,
                Fetched::NotModified => return Err(ModelError::Http { status: 304 }),
            }
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

    let dir = (hooks.open_dir)().map_err(map_fs_error)?;
    let mut guard = TmpGuard::create(&dir, &mut *hooks.next_staging, &mut *hooks.next_tmp)
        .map_err(map_fs_error)?;

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
        guard.file.write_all(&buf[..n]).map_err(map_fs_error)?;
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
    guard.file.sync_all().map_err(map_fs_error)?;
    (hooks.verify_before_commit)(&guard.file)?;
    // 一時ファイルは 0700 の private staging 内にあり、作成者以外は名前を差し替えられない。
    // 検証した実体と公開する実体の一致は構造的に保証されるため、名前の再照合や公開後の
    // 撤去（別処理の成果物を消しうる）は行わない（A08）。モデル本体の公開は `renameat` 1 回のみ。
    dirfd::rename_across(&guard.staging, &guard.tmp_name, &dir, MODEL_FILE_NAME)
        .map_err(map_fs_error)?;
    guard.armed = false;
    // 次の `download.json` 公開が同名の staging を再利用できるよう、先に staging を片付ける。
    drop(guard);
    // モデル本体の公開後に `download.json` を公開する。ここで失敗してもモデルは検証済みで
    // 公開されており、古い JSON は次回の `304` 再検証で SHA 不一致となり全量取り直しになる。
    let meta = CacheMeta {
        url: job.url.as_str().to_string(),
        etag: resp.etag.filter(|e| cache::valid_etag(e)),
        content_length: received,
        sha256: actual,
        fetched_at: (hooks.now)(),
    };
    cache::publish(&dir, &meta, &mut *hooks.next_staging, &mut *hooks.next_tmp)
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
            etag: Option<&'static str>,
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
            etag: None,
            body: body.to_vec(),
        }
    }

    fn redirect(status: u16, loc: Option<&'static str>) -> Step {
        Step::Resp {
            status,
            location: loc,
            content_length: Some(0),
            encoding: None,
            etag: None,
            body: Vec::new(),
        }
    }

    struct Mock {
        steps: VecDeque<Step>,
        requested: Vec<String>,
        /// 各リクエストに付いた `If-None-Match`（`requested` と同じ添字）。
        conditional: Vec<Option<String>>,
    }

    impl Mock {
        fn new(steps: Vec<Step>) -> Self {
            Self {
                steps: steps.into(),
                requested: Vec::new(),
                conditional: Vec::new(),
            }
        }
    }

    impl Transport for Mock {
        fn get(
            &mut self,
            url: &ValidatedUrl,
            _d: Instant,
            if_none_match: Option<&str>,
        ) -> Result<RawResponse, ModelError> {
            self.requested.push(url.as_str().to_string());
            self.conditional.push(if_none_match.map(String::from));
            match self.steps.pop_front().expect("scripted step") {
                Step::Resp {
                    status,
                    location,
                    content_length,
                    encoding,
                    etag,
                    body,
                } => Ok(RawResponse {
                    status,
                    location: location.map(String::from),
                    content_length,
                    content_encoding: encoding.map(String::from),
                    etag: etag.map(String::from),
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
        fn meta_path(&self) -> PathBuf {
            self.root.join("m/v").join(cache::DOWNLOAD_META_FILE)
        }
        fn meta(&self) -> Option<CacheMeta> {
            cache::parse(&std::fs::read(self.meta_path()).ok()?).ok()
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
                    .filter(|n| n != MODEL_FILE_NAME && n != cache::DOWNLOAD_META_FILE)
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
            cache: None,
        }
    }

    /// 条件付き GET の根拠になる保存済みメタデータ付きの Job。
    fn job_cached(expected: Option<Sha256Pin>, meta: CacheMeta) -> Job<'static> {
        let mut j = job(1 << 20, 5, expected);
        j.cache = Some(meta);
        j
    }

    fn meta_for(data: &[u8], etag: &str, fetched_at: u64) -> CacheMeta {
        CacheMeta {
            url: "https://a.example/m".into(),
            etag: Some(etag.into()),
            content_length: data.len() as u64,
            sha256: sha(data),
            fetched_at,
        }
    }

    fn etag_ok(body: &[u8], etag: &'static str) -> Step {
        Step::Resp {
            status: 200,
            location: None,
            content_length: Some(body.len() as u64),
            encoding: None,
            etag: Some(etag),
            body: body.to_vec(),
        }
    }

    fn not_modified() -> Step {
        redirect(304, None)
    }

    fn exec(
        env: &Env,
        mock: &mut Mock,
        job: Job<'_>,
        progress: Option<&mut dyn FnMut(DownloadProgress)>,
    ) -> Result<(), ModelError> {
        exec_at(env, mock, job, progress, 1000)
    }

    fn exec_at(
        env: &Env,
        mock: &mut Mock,
        job: Job<'_>,
        progress: Option<&mut dyn FnMut(DownloadProgress)>,
        now_secs: u64,
    ) -> Result<(), ModelError> {
        exec_full(env, mock, job, progress, now_secs, &mut |_: &File| Ok(()))
    }

    fn exec_full(
        env: &Env,
        mock: &mut Mock,
        job: Job<'_>,
        progress: Option<&mut dyn FnMut(DownloadProgress)>,
        now_secs: u64,
        verify: &mut dyn FnMut(&File) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let root = env.root.clone();
        let mut open = {
            let root = root.clone();
            move || -> std::io::Result<OwnedFd> {
                let r = dirfd::open_root(&root)?;
                let n = dirfd::open_or_create_dir(&r, "m")?;
                dirfd::open_or_create_dir(&n, "v")
            }
        };
        let mut lookup = move || -> std::io::Result<Option<OwnedFd>> {
            let Some(r) = dirfd::open_root_existing(&root)? else {
                return Ok(None);
            };
            let Some(n) = dirfd::open_existing_dir(&r, "m")? else {
                return Ok(None);
            };
            dirfd::open_existing_dir(&n, "v")
        };
        let mut stg = || ".s".to_string();
        let mut n = 0u32;
        let mut next = move || {
            n += 1;
            format!(".t{n}")
        };
        let mut now = move || now_secs;
        run(
            mock,
            job,
            progress,
            Hooks {
                lookup_dir: &mut lookup,
                open_dir: &mut open,
                next_staging: &mut stg,
                next_tmp: &mut next,
                verify_before_commit: verify,
                now: &mut now,
            },
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
                etag: None,
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
            etag: None,
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
            etag: None,
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
            etag: None,
            body: b"abc".to_vec(),
        }]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
        assert!(matches!(e, ModelError::Network(_)));
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: Some(3),
            encoding: Some("identity"),
            etag: None,
            body: b"abc".to_vec(),
        }]);
        exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap();
        // 別行で届いた gzip（転送層が連結した値）も拒否する。
        let mut mock = Mock::new(vec![Step::Resp {
            status: 200,
            location: None,
            content_length: Some(3),
            encoding: Some("identity,gzip"),
            etag: None,
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
        let e = exec_full(
            &env,
            &mut mock,
            job(1 << 20, 5, None),
            None,
            1000,
            &mut |_: &File| Err(ModelError::Io(std::io::Error::other("verify"))),
        )
        .unwrap_err();
        assert!(matches!(e, ModelError::Io(_)));
        assert!(!env.leaf().exists());
        // 検証失敗では `download.json` も公開しない。
        assert!(env.meta().is_none());
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
        let e = exec_full(
            &env,
            &mut mock,
            job(1 << 20, 5, None),
            None,
            1000,
            &mut |_: &File| {
                // 一時ファイルは共有ディレクトリ直下ではなく 0700 の staging 内にだけ存在する。
                let v = root.join("m/v");
                assert!(!v.join(".t1").exists());
                assert!(v.join(".s/.t1").is_file());
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

    // ---- #3126: キャッシュ・条件付き GET・304 再検証 ----

    /// 1 回目の 200（ETag 付き）で `download.json` が作られ、`Job.cache` に使える状態にする。
    fn first_fetch(env: &Env, data: &[u8], etag: &'static str, at: u64) -> CacheMeta {
        let mut mock = Mock::new(vec![etag_ok(data, etag)]);
        exec_at(env, &mut mock, job(1 << 20, 5, None), None, at).unwrap();
        assert_eq!(mock.conditional, vec![None], "初回は条件なし");
        env.meta().expect("download.json")
    }

    #[test]
    fn success_writes_download_json_with_all_fields() {
        let env = Env::new("meta");
        let data = b"hello model";
        let m = first_fetch(&env, data, "\"e1\"", 100);
        assert_eq!(m.url, "https://a.example/m");
        assert_eq!(m.etag.as_deref(), Some("\"e1\""));
        assert_eq!(m.content_length, data.len() as u64);
        assert_eq!(m.sha256, sha(data));
        assert_eq!(m.fetched_at, 100);
        // 0600 のまま公開される（クエリに署名トークンを含みうる URL を保存するため）。
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(env.meta_path())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0);
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn not_modified_reuses_cache_and_updates_only_fetched_at() {
        let env = Env::new("304");
        let data = b"model bytes";
        let m1 = first_fetch(&env, data, "\"e1\"", 100);
        let mut mock = Mock::new(vec![not_modified()]);
        exec_at(&env, &mut mock, job_cached(None, m1.clone()), None, 200).unwrap();
        assert_eq!(mock.conditional, vec![Some("\"e1\"".to_string())]);
        assert_eq!(std::fs::read(env.leaf()).unwrap(), data);
        let m2 = env.meta().unwrap();
        assert_eq!(m2.fetched_at, 200);
        assert_eq!(
            CacheMeta {
                fetched_at: 100,
                ..m2
            },
            m1,
            "fetched_at 以外は不変"
        );
        assert!(env.leftovers().is_empty());
    }

    #[test]
    fn tampered_cache_with_304_triggers_unconditional_refetch() {
        let env = Env::new("tamper");
        let data = b"model bytes";
        let m1 = first_fetch(&env, data, "\"e1\"", 100);
        std::fs::write(env.leaf(), b"tampered!!!").unwrap();
        let mut mock = Mock::new(vec![not_modified(), etag_ok(data, "\"e2\"")]);
        exec_at(&env, &mut mock, job_cached(None, m1), None, 300).unwrap();
        assert_eq!(mock.conditional, vec![Some("\"e1\"".to_string()), None]);
        assert_eq!(std::fs::read(env.leaf()).unwrap(), data, "取り直して復旧");
        let m = env.meta().unwrap();
        assert_eq!((m.etag.as_deref(), m.fetched_at), (Some("\"e2\""), 300));
    }

    #[test]
    fn missing_model_file_with_304_refetches() {
        let env = Env::new("missing");
        let data = b"model bytes";
        let m1 = first_fetch(&env, data, "\"e1\"", 100);
        std::fs::remove_file(env.leaf()).unwrap();
        let mut mock = Mock::new(vec![not_modified(), etag_ok(data, "\"e1\"")]);
        exec(&env, &mut mock, job_cached(None, m1), None).unwrap();
        assert_eq!(std::fs::read(env.leaf()).unwrap(), data);
    }

    #[test]
    fn symlinked_model_file_with_304_is_not_reused() {
        let env = Env::new("symlink304");
        let data = b"model bytes";
        let m1 = first_fetch(&env, data, "\"e1\"", 100);
        let outside = env.root.join("outside.bin");
        std::fs::write(&outside, data).unwrap();
        std::fs::remove_file(env.leaf()).unwrap();
        std::os::unix::fs::symlink(&outside, env.leaf()).unwrap();
        let mut mock = Mock::new(vec![not_modified(), etag_ok(data, "\"e1\"")]);
        exec(&env, &mut mock, job_cached(None, m1), None).unwrap();
        assert_eq!(mock.conditional.len(), 2, "リンクは辿らず取り直す");
        assert!(!std::fs::symlink_metadata(env.leaf()).unwrap().is_symlink());
    }

    #[test]
    fn pin_is_used_for_revalidation_instead_of_stored_sha() {
        let env = Env::new("pin304");
        let data = b"model bytes";
        let m1 = first_fetch(&env, data, "\"e1\"", 100);
        // pin が保存値と一致していれば再利用できる。
        let mut mock = Mock::new(vec![not_modified()]);
        exec_at(
            &env,
            &mut mock,
            job_cached(Some(sha(data)), m1.clone()),
            None,
            200,
        )
        .unwrap();
        // 保存値は一致するが pin が食い違う（呼び出し側が通常は弾く不整合）場合、
        // 再検証は pin で行い不一致のため取り直す。取り直した内容も pin と合わなければ
        // HashMismatch となり、既存ファイルには触れない。
        let mut mock = Mock::new(vec![not_modified(), etag_ok(b"other bytes", "\"e9\"")]);
        let e = exec(&env, &mut mock, job_cached(Some(sha(b"zzz")), m1), None).unwrap_err();
        assert!(matches!(e, ModelError::HashMismatch { .. }));
        assert_eq!(std::fs::read(env.leaf()).unwrap(), data);
        assert_eq!(env.meta().unwrap().fetched_at, 200);
    }

    #[test]
    fn cache_without_etag_never_sends_conditional_get() {
        let env = Env::new("noetag");
        let data = b"abc";
        let mut m = first_fetch(&env, data, "\"e1\"", 100);
        m.etag = None;
        let mut mock = Mock::new(vec![ok_body(data)]);
        exec(&env, &mut mock, job_cached(None, m), None).unwrap();
        assert_eq!(mock.conditional, vec![None]);
        // etag を返さない応答で JSON の etag も null に更新される（古い etag を残さない）。
        assert_eq!(env.meta().unwrap().etag, None);
    }

    #[test]
    fn invalid_etag_in_response_is_not_stored() {
        let env = Env::new("badetag");
        let mut mock = Mock::new(vec![etag_ok(b"abc", "bad\r\netag")]);
        exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap();
        assert_eq!(env.meta().unwrap().etag, None);
    }

    #[test]
    fn unsolicited_and_second_304_are_http_errors() {
        // 条件付きで送っていない 304。
        let env = Env::new("unsol");
        let mut mock = Mock::new(vec![not_modified()]);
        let e = exec(&env, &mut mock, job(1 << 20, 5, None), None).unwrap_err();
        assert!(matches!(e, ModelError::Http { status: 304 }));
        // 再検証失敗後の無条件 GET への 304。
        let env = Env::new("second304");
        let m = first_fetch(&env, b"abc", "\"e1\"", 1);
        std::fs::write(env.leaf(), b"xyz").unwrap();
        let mut mock = Mock::new(vec![not_modified(), not_modified()]);
        let e = exec(&env, &mut mock, job_cached(None, m), None).unwrap_err();
        assert!(matches!(e, ModelError::Http { status: 304 }));
        assert_eq!(std::fs::read(env.leaf()).unwrap(), b"xyz", "既存に触れない");
    }

    #[test]
    fn redirect_hops_carry_if_none_match() {
        let env = Env::new("redir304");
        let data = b"abc";
        let m = first_fetch(&env, data, "\"e1\"", 1);
        let mut mock = Mock::new(vec![
            redirect(302, Some("https://cdn.example/x")),
            not_modified(),
        ]);
        exec_at(&env, &mut mock, job_cached(None, m), None, 9).unwrap();
        assert_eq!(
            mock.conditional,
            vec![Some("\"e1\"".to_string()), Some("\"e1\"".to_string())]
        );
        assert_eq!(env.meta().unwrap().fetched_at, 9);
    }

    #[test]
    fn not_modified_without_existing_dir_refetches_without_creating_before_headers() {
        let env = Env::new("nodir");
        let data = b"abc";
        let meta = meta_for(data, "\"e1\"", 1);
        let mut mock = Mock::new(vec![not_modified(), etag_ok(data, "\"e1\"")]);
        exec(&env, &mut mock, job_cached(None, meta), None).unwrap();
        assert_eq!(std::fs::read(env.leaf()).unwrap(), data);
        // 参照段は作成しない: ヘッダ段階で失敗するなら `<name>` も作られない。
        let env2 = Env::new("nodir2");
        let mut mock = Mock::new(vec![
            not_modified(),
            Step::Resp {
                status: 200,
                location: None,
                content_length: Some(1000),
                encoding: None,
                etag: None,
                body: vec![0; 10],
            },
        ]);
        let mut j = job_cached(None, meta_for(data, "\"e1\"", 1));
        j.max_bytes = 100;
        let e = exec(&env2, &mut mock, j, None).unwrap_err();
        assert!(matches!(e, ModelError::TooLarge { .. }));
        assert!(!env2.root.join("m").exists());
    }

    #[test]
    fn stale_json_with_newer_model_self_heals_on_next_304() {
        let env = Env::new("heal");
        let data = b"model bytes";
        let mut m_old = first_fetch(&env, data, "\"e1\"", 1);
        // JSON は古いまま（モデルだけ新しい）状態を再現する: 本体を書き換えて 304。
        let new_data = b"newer model";
        std::fs::write(env.leaf(), new_data).unwrap();
        m_old.fetched_at = 1;
        let mut mock = Mock::new(vec![not_modified(), etag_ok(new_data, "\"e2\"")]);
        exec(&env, &mut mock, job_cached(None, m_old), None).unwrap();
        assert_eq!(std::fs::read(env.leaf()).unwrap(), new_data);
        assert_eq!(env.meta().unwrap().sha256, sha(new_data));
    }

    #[test]
    fn storage_full_and_permission_errors_are_typed() {
        for (kind, full) in [
            (std::io::ErrorKind::StorageFull, true),
            (std::io::ErrorKind::PermissionDenied, false),
        ] {
            let env = Env::new("typed");
            let mut mock = Mock::new(vec![ok_body(b"abc")]);
            let mut open = || -> std::io::Result<OwnedFd> { Err(std::io::Error::from(kind)) };
            let mut lookup = || -> std::io::Result<Option<OwnedFd>> { Ok(None) };
            let mut stg = || ".s".to_string();
            let mut next = || ".t".to_string();
            let mut verify = |_: &File| Ok(());
            let mut now = || 0u64;
            let e = run(
                &mut mock,
                job(1 << 20, 5, None),
                None,
                Hooks {
                    lookup_dir: &mut lookup,
                    open_dir: &mut open,
                    next_staging: &mut stg,
                    next_tmp: &mut next,
                    verify_before_commit: &mut verify,
                    now: &mut now,
                },
            )
            .unwrap_err();
            if full {
                assert!(matches!(e, ModelError::StorageFull(_)), "{e:?}");
            } else {
                assert!(matches!(e, ModelError::PermissionDenied(_)), "{e:?}");
            }
            let _ = &env;
        }
    }

    #[test]
    fn cache_read_falls_back_on_corrupt_oversized_or_linked_json() {
        let env = Env::new("corrupt");
        let dir_path = env.root.join("m/v");
        std::fs::create_dir_all(&dir_path).unwrap();
        let r = dirfd::open_root(&env.root).unwrap();
        let n = dirfd::open_or_create_dir(&r, "m").unwrap();
        let v = dirfd::open_or_create_dir(&n, "v").unwrap();
        let path = env.meta_path();
        let good = cache::serialize(&meta_for(b"abc", "\"e1\"", 1));
        std::fs::write(&path, &good).unwrap();
        assert!(cache::read(&v).is_some());
        std::fs::write(&path, b"{not json").unwrap();
        assert!(cache::read(&v).is_none());
        std::fs::write(&path, vec![b' '; 70 * 1024]).unwrap();
        assert!(cache::read(&v).is_none(), "64 KiB 超は無視");
        std::fs::remove_file(&path).unwrap();
        let outside = env.root.join("outside.json");
        std::fs::write(&outside, &good).unwrap();
        std::os::unix::fs::symlink(&outside, &path).unwrap();
        assert!(cache::read(&v).is_none(), "シンボリックリンクは辿らない");
    }
}
