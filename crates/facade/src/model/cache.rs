//! ダウンロードのローカルキャッシュ `download.json`（イシュー #3126。`cfg(unix)`）。
//!
//! 呼び出し元は `model.rs` の `download_via`（キャッシュ参照段）と `download.rs` の `run`
//! （`304` 再検証・取得後のメタデータ公開）。`<cache_dir>/<name>/<version>/download.json` に
//! 取得メタデータ（`url`・`final_url`・`etag`・`content_length`・`sha256`・`fetched_at`）を置き、次回の
//! 条件付き GET（`If-None-Match`）と `304` 受信時の再検証に使う（設計 §5）。
//!
//! 保存値は**破損・部分改変の検出用**にすぎず真正性の根拠ではない（真正性は利用者が指定する
//! pin による。§6 A08）。`download.json` は信頼できない入力として扱い、サイズ上限・厳格な
//! スキーマ（未知キー拒否）・印字可能 ASCII のみの etag（ヘッダインジェクション対策）で検証し、
//! 違反は「キャッシュなし」として再取得へフォールバックする（A03）。読み書きはすべて検証済み
//! dirfd 基準・`O_NOFOLLOW` で行い、パス文字列を解決し直さない（A01）。

use std::io::{Read, Write};
use std::os::fd::OwnedFd;

use serde_json::{Map, Value, json};

use super::dirfd;
use super::download::TmpGuard;
use super::sha256::Sha256;
use super::url::validate_https_url;
use super::{MODEL_FILE_NAME, ModelError, Sha256Pin, map_fs_error};

/// キャッシュメタデータのファイル名（`<name>/<version>/` 直下）。
pub(super) const DOWNLOAD_META_FILE: &str = "download.json";
/// `download.json` の読み取り上限（5 項目の小さな JSON のため十分に小さく固定する）。
const MAX_META_BYTES: u64 = 64 * 1024;
/// `url` の長さ上限（`url::validate_https_url` の上限と整合する安全側の値）。
const MAX_URL_LEN: usize = 8192;
/// `etag` の長さ上限。
const MAX_ETAG_LEN: usize = 1024;
/// 再ハッシュの読み取りチャンク長。
const CHUNK: usize = 64 * 1024;

/// `download.json` の内容。`etag` が `None` のサーバ（検証子なし）では条件付き GET を送らない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CacheMeta {
    pub(super) url: String,
    /// リダイレクト追従後に `200` を返した最終取得 URL。`etag` はこの URL のリソースの検証子で
    /// あり、条件付き GET はこの URL へのリクエストにだけ付ける（リダイレクト先が変わったとき
    /// 別リソースの同一 ETag による `304` で古いモデルを再利用しないため）。
    pub(super) final_url: String,
    pub(super) etag: Option<String>,
    pub(super) content_length: u64,
    pub(super) sha256: Sha256Pin,
    /// 取得（または `304` 再検証）時刻。UNIX エポック秒。
    pub(super) fetched_at: u64,
}

fn bad(reason: &'static str) -> ModelError {
    ModelError::CacheMetadata { reason }
}

/// `If-None-Match` に載せてよい etag か。印字可能 ASCII（0x20〜0x7E）のみ・長さ 1〜1024。
/// CR／LF／NUL・制御文字・非 ASCII は拒否する（ヘッダインジェクション対策）。保存時・読み出し時・
/// 送信直前の 3 か所で共通に使う。
pub(super) fn valid_etag(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_ETAG_LEN && s.bytes().all(|b| (0x20..=0x7e).contains(&b))
}

fn check(meta: &CacheMeta) -> Result<(), ModelError> {
    if meta.url.len() > MAX_URL_LEN || validate_https_url(&meta.url).is_err() {
        return Err(bad("url が不正です"));
    }
    if meta.final_url.len() > MAX_URL_LEN || validate_https_url(&meta.final_url).is_err() {
        return Err(bad("final_url が不正です"));
    }
    if let Some(e) = &meta.etag
        && !valid_etag(e)
    {
        return Err(bad("etag が不正です"));
    }
    Ok(())
}

/// `download.json` のバイト列をスキーマ検証してパースする。最上位はオブジェクトで、キーは
/// ちょうど 6 つ（未知キー・不足は拒否）。値の型・桁・`url`・`final_url`・`etag` も検証する。
pub(super) fn parse(bytes: &[u8]) -> Result<CacheMeta, ModelError> {
    let v: Value = serde_json::from_slice(bytes).map_err(|_| bad("JSON として解釈できません"))?;
    let Value::Object(obj) = v else {
        return Err(bad("最上位がオブジェクトではありません"));
    };
    let known = [
        "url",
        "final_url",
        "etag",
        "content_length",
        "sha256",
        "fetched_at",
    ];
    if obj.len() != known.len() || obj.keys().any(|k| !known.contains(&k.as_str())) {
        return Err(bad("キー集合が不正です"));
    }
    let get = |k: &str| obj.get(k).ok_or_else(|| bad("キーが不足しています"));
    let url = get("url")?
        .as_str()
        .ok_or_else(|| bad("url の型が不正です"))?
        .to_string();
    let final_url = get("final_url")?
        .as_str()
        .ok_or_else(|| bad("final_url の型が不正です"))?
        .to_string();
    let etag = match get("etag")? {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        _ => return Err(bad("etag の型が不正です")),
    };
    let content_length = get("content_length")?
        .as_u64()
        .ok_or_else(|| bad("content_length の型が不正です"))?;
    let sha256 = Sha256Pin::from_hex(
        get("sha256")?
            .as_str()
            .ok_or_else(|| bad("sha256 の型が不正です"))?,
    )
    .map_err(|_| bad("sha256 が 64 桁の 16 進ではありません"))?;
    let fetched_at = get("fetched_at")?
        .as_u64()
        .ok_or_else(|| bad("fetched_at の型が不正です"))?;
    let meta = CacheMeta {
        url,
        final_url,
        etag,
        content_length,
        sha256,
        fetched_at,
    };
    check(&meta)?;
    Ok(meta)
}

/// `download.json` のバイト列を作る（6 キー固定順・`sha256` は小文字 16 進）。
pub(super) fn serialize(meta: &CacheMeta) -> Vec<u8> {
    let mut m = Map::new();
    m.insert("url".into(), json!(meta.url));
    m.insert("final_url".into(), json!(meta.final_url));
    m.insert("etag".into(), json!(meta.etag));
    m.insert("content_length".into(), json!(meta.content_length));
    m.insert("sha256".into(), json!(meta.sha256.to_string()));
    m.insert("fetched_at".into(), json!(meta.fetched_at));
    // `Value` のシリアライズは失敗しない（文字列キー・有限値のみ）。
    serde_json::to_vec(&Value::Object(m)).unwrap_or_default()
}

/// `version_dir` 直下の `download.json` を読む。不在・I/O エラー・不正な内容はすべて `None`
/// （キャッシュなし扱いで再取得へフォールバックする）。そのため [`ModelError::CacheMetadata`] が
/// この経路で呼び出し側へ届くことはなく、届くのは [`publish`] が不正なメタデータを拒否した場合だけ。
pub(super) fn read(version_dir: &OwnedFd) -> Option<CacheMeta> {
    let mut f = dirfd::open_file_no_follow(version_dir, DOWNLOAD_META_FILE, MAX_META_BYTES)
        .ok()
        .flatten()?;
    let mut buf = Vec::new();
    (&mut f)
        .take(MAX_META_BYTES + 1)
        .read_to_end(&mut buf)
        .ok()?;
    if buf.len() as u64 > MAX_META_BYTES {
        return None;
    }
    parse(&buf).ok()
}

/// `version_dir` 直下の既存 `model.safetensors` の SHA-256 をストリーミングで計算し直す
/// （`304` 再検証用。全量をメモリへ載せない）。不在・非通常ファイル・シンボリックリンク・
/// 読み取り失敗・上限（`max`）超過は `None`（不一致と同じ扱いで取り直す）。
pub(super) fn rehash(version_dir: &OwnedFd, max: u64) -> Option<Sha256Pin> {
    let mut f = dirfd::open_file_no_follow(version_dir, MODEL_FILE_NAME, max)
        .ok()
        .flatten()?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    let mut total: u64 = 0;
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                total = total.saturating_add(n as u64);
                if total > max {
                    return None;
                }
                hasher.update(&buf[..n]);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    Some(Sha256Pin(hasher.finalize()))
}

/// `download.json` を private staging 経由で atomic に公開する（一時ファイル → `write_all` →
/// `sync_all` → `renameat`）。`RENAME_NOREPLACE` は付けず既存を atomic replace する。不正な
/// メタデータは書かずに [`ModelError::CacheMetadata`]。FS エラーは [`map_fs_error`] を通す。
pub(super) fn publish(
    version_dir: &OwnedFd,
    meta: &CacheMeta,
    next_staging: &mut dyn FnMut() -> String,
    next_tmp: &mut dyn FnMut() -> String,
) -> Result<(), ModelError> {
    check(meta)?;
    let bytes = serialize(meta);
    let mut guard = TmpGuard::create(version_dir, next_staging, next_tmp).map_err(map_fs_error)?;
    guard.file.write_all(&bytes).map_err(map_fs_error)?;
    guard.file.sync_all().map_err(map_fs_error)?;
    dirfd::rename_across(
        &guard.staging,
        &guard.tmp_name,
        version_dir,
        DOWNLOAD_META_FILE,
    )
    .map_err(map_fs_error)?;
    guard.armed = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> CacheMeta {
        CacheMeta {
            url: "https://a.example/m?x=1".into(),
            final_url: "https://cdn.example/m?x=1".into(),
            etag: Some("\"abc\"".into()),
            content_length: 12,
            sha256: Sha256Pin([7; 32]),
            fetched_at: 1_700_000_000,
        }
    }

    #[test]
    fn serialize_parse_round_trip() {
        let m = meta();
        assert_eq!(parse(&serialize(&m)).unwrap(), m);
        let mut n = meta();
        n.etag = None;
        assert_eq!(parse(&serialize(&n)).unwrap(), n);
    }

    #[test]
    fn parse_rejects_malformed_documents() {
        let good = String::from_utf8(serialize(&meta())).unwrap();
        let sha = Sha256Pin([7; 32]).to_string();
        let cases: Vec<String> = vec![
            "not json".into(),
            "[]".into(),
            "{}".into(),
            good.replace("}", ",\"extra\":1}"),
            good.replace(&sha, &sha[..62]),
            good.replace(&sha, &"g".repeat(64)),
            good.replace("\"abc\\\"", "\"a\\r\\nb\\\""),
            good.replace("\"content_length\":12", "\"content_length\":\"12\""),
            good.replace("\"content_length\":12", "\"content_length\":-1"),
            good.replace("https://a.example", "http://a.example"),
            good.replace("https://cdn.example", "http://cdn.example"),
            good.replace("\"final_url\"", "\"final_urk\""),
            good.replace("https://a.example", "https://u:p@a.example"),
            good.replace("\"fetched_at\":1700000000", "\"fetched_at\":1.5"),
        ];
        for c in cases {
            assert!(
                matches!(parse(c.as_bytes()), Err(ModelError::CacheMetadata { .. })),
                "accepted: {c}"
            );
        }
    }

    #[test]
    fn etag_validation() {
        assert!(valid_etag("\"abc\""));
        assert!(valid_etag("W/\"x y\""));
        for bad in ["", "a\r\nb", "a\0b", "a\tb", "caf\u{e9}", &"x".repeat(1025)] {
            assert!(!valid_etag(bad), "{bad:?}");
        }
    }
}
