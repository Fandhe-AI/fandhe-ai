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

/// 手書きパーサーが扱う JSON 値（`download.json` のスキーマで現れる型だけ）。
enum Val {
    Str(String),
    Null,
    Num(u64),
}

/// 固定スキーマ専用の最小 JSON パーサー（`serde_json` は facade の通常依存にしないため手書き。
/// deps-policy.md）。最上位は 1 段のオブジェクトで、値は文字列・`null`・非負整数（符号・小数・
/// 指数・先頭ゼロ不可）のみ。重複キー・末尾の余剰バイトは拒否する。
struct Scanner<'a> {
    b: &'a [u8],
    i: usize,
}

impl Scanner<'_> {
    fn ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn lit(&mut self, word: &[u8]) -> bool {
        if self.b[self.i..].starts_with(word) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let h = self.b.get(self.i..self.i + 4)?;
        self.i += 4;
        let mut v = 0u32;
        for &c in h {
            v = v * 16 + (c as char).to_digit(16)?;
        }
        Some(v)
    }

    fn string(&mut self) -> Option<String> {
        if !self.eat(b'"') {
            return None;
        }
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => break,
                0x00..=0x1f => return None,
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.lit(b"\\u") {
                                    return None;
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return None;
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(cp)?
                        }
                        _ => return None,
                    };
                    let mut tmp = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                }
                _ => out.push(c),
            }
        }
        String::from_utf8(out).ok()
    }

    fn number(&mut self) -> Option<u64> {
        let start = self.i;
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        let d = &self.b[start..self.i];
        if d.is_empty() || (d.len() > 1 && d[0] == b'0') {
            return None;
        }
        // 小数・指数は直後の文字が `,`／`}` でないため後続の検査で拒否される。
        std::str::from_utf8(d).ok()?.parse().ok()
    }

    fn value(&mut self) -> Option<Val> {
        self.ws();
        match self.b.get(self.i)? {
            b'"' => self.string().map(Val::Str),
            b'n' => self.lit(b"null").then_some(Val::Null),
            b'0'..=b'9' => self.number().map(Val::Num),
            _ => None,
        }
    }

    fn object(&mut self) -> Option<Vec<(String, Val)>> {
        if !self.eat(b'{') {
            return None;
        }
        let mut out: Vec<(String, Val)> = Vec::new();
        if !self.eat(b'}') {
            loop {
                let k = self.string()?;
                if out.iter().any(|(e, _)| *e == k) || !self.eat(b':') {
                    return None;
                }
                let v = self.value()?;
                out.push((k, v));
                if self.eat(b',') {
                    continue;
                }
                if self.eat(b'}') {
                    break;
                }
                return None;
            }
        }
        self.ws();
        (self.i == self.b.len()).then_some(out)
    }
}

/// `download.json` のバイト列をスキーマ検証してパースする。最上位はオブジェクトで、キーは
/// ちょうど 6 つ（未知キー・不足は拒否）。値の型・桁・`url`・`final_url`・`etag` も検証する。
pub(super) fn parse(bytes: &[u8]) -> Result<CacheMeta, ModelError> {
    let obj = Scanner { b: bytes, i: 0 }
        .object()
        .ok_or_else(|| bad("JSON として解釈できません"))?;
    let known = [
        "url",
        "final_url",
        "etag",
        "content_length",
        "sha256",
        "fetched_at",
    ];
    if obj.len() != known.len() || obj.iter().any(|(k, _)| !known.contains(&k.as_str())) {
        return Err(bad("キー集合が不正です"));
    }
    let get = |k: &str| {
        obj.iter()
            .find(|(e, _)| e == k)
            .map(|(_, v)| v)
            .ok_or_else(|| bad("キーが不足しています"))
    };
    let text = |k: &str, msg: &'static str| match get(k)? {
        Val::Str(s) => Ok(s.clone()),
        _ => Err(bad(msg)),
    };
    let num = |k: &str, msg: &'static str| match get(k)? {
        Val::Num(n) => Ok(*n),
        _ => Err(bad(msg)),
    };
    let url = text("url", "url の型が不正です")?;
    let final_url = text("final_url", "final_url の型が不正です")?;
    let etag = match get("etag")? {
        Val::Null => None,
        Val::Str(s) => Some(s.clone()),
        Val::Num(_) => return Err(bad("etag の型が不正です")),
    };
    let content_length = num("content_length", "content_length の型が不正です")?;
    let sha256 = Sha256Pin::from_hex(&text("sha256", "sha256 の型が不正です")?)
        .map_err(|_| bad("sha256 が 64 桁の 16 進ではありません"))?;
    let fetched_at = num("fetched_at", "fetched_at の型が不正です")?;
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

/// JSON 文字列リテラル（引用符込み）へエスケープして追記する。
fn json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `download.json` のバイト列を作る（6 キー固定順・`sha256` は小文字 16 進）。
pub(super) fn serialize(meta: &CacheMeta) -> Vec<u8> {
    let mut s = String::from("{\"url\":");
    json_str(&meta.url, &mut s);
    s.push_str(",\"final_url\":");
    json_str(&meta.final_url, &mut s);
    s.push_str(",\"etag\":");
    match &meta.etag {
        Some(e) => json_str(e, &mut s),
        None => s.push_str("null"),
    }
    s.push_str(&format!(
        ",\"content_length\":{},\"sha256\":\"{}\",\"fetched_at\":{}}}",
        meta.content_length, meta.sha256, meta.fetched_at
    ));
    s.into_bytes()
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
            good.replace("\"fetched_at\":1700000000", "\"fetched_at\":01700000000"),
            good.replace("\"fetched_at\":1700000000", "\"fetched_at\":1e9"),
            good.replace("\"fetched_at\"", "\"content_length\""),
            format!("{good} x"),
            good.replace("\"abc\\\"", "\"\\ud800\\\""),
        ];
        for c in cases {
            assert!(
                matches!(parse(c.as_bytes()), Err(ModelError::CacheMetadata { .. })),
                "accepted: {c}"
            );
        }
    }

    #[test]
    fn parse_decodes_unicode_escapes() {
        let good = String::from_utf8(serialize(&meta())).unwrap();
        let escaped = good.replace("abc", "\\u0061bc");
        assert_eq!(parse(escaped.as_bytes()).unwrap(), meta());
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
