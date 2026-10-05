//! `CsvLogger`／`JsonLogger`（`callbacks.rs`）のファイル I/O・整形・
//! 既存ログ検証の private helper（イシュー #2571・親 #2570・
//! `docs/compat-callbacks-loggers-decision.md` §4・§7）。
//!
//! 役割は「行の組み立て」「既存ファイルの上限付き読み込みと構文検証」
//! 「一時ファイル＋`rename` による原子的書き出し」に限る。`callbacks.rs`
//! の 2 型だけから呼ばれ、公開面には何も出さない（全て `pub(super)`）。
//! 依存は std のみ（`serde_json` は facade の依存区分外のため手書き。
//! `.claude/rules/deps-policy.md`）。
//!
//! # セキュリティ上の前提（決定記録 §7）
//!
//! - 列名（キー）は固定 ASCII で、ユーザー由来の文字列は出力しない。
//! - 非有限値は固定リテラル（CSV: `NaN`／`inf`／`-inf`、JSON: クォート付き
//!   文字列 `"NaN"`／`"Infinity"`／`"-Infinity"`）で書く。
//! - append 時に読む既存ファイルは入力として信頼しない: サイズ上限を
//!   課し、パーサは明示スタック＋深さ上限で再帰せず、不正入力は panic
//!   せず `Err` を返す（fail-closed）。`unwrap`／`expect` は使わない。

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use super::metrics::Metrics;
use super::training::History;

/// CSV 既存ファイルの先頭行（ヘッダ）として読む最大バイト数。
/// 列は高々 8 個・各十数バイトで実ヘッダは 100 B 程度のため 4 KiB は十分
/// に大きい。改行が現れないまま超過したら異常ファイルとして拒否する。
pub(super) const MAX_HEADER_BYTES: u64 = 4096;

/// JSON 既存ファイルとして読み込む最大バイト数。1 epoch あたり約 200 B
/// のため 64 MiB は数十万 epoch 分に相当し、正常な学習ログは必ず収まる。
/// 無制限の `read_to_end` は巨大・悪性ファイルでメモリを使い尽くすため
/// 避ける（`fs_guard.rs` 冒頭・PR #2226 の教訓）。
pub(super) const MAX_JSON_LOG_BYTES: u64 = 64 * 1024 * 1024;

/// JSON パーサの入れ子深さ上限。明示スタック方式で再帰はしないが、
/// 悪性の深い入れ子でスタック Vec が際限なく伸びるのを防ぐ。
pub(super) const MAX_JSON_DEPTH: usize = 128;

/// fit 開始時に確定する列集合（`epoch` 列は常に先頭にあり含まない）。
/// 順序は固定: `loss, lr, [val_loss, [val_accuracy, val_precision,
/// val_recall, val_f1]]`。`metrics` 引数の並びや重複には依存しない。
/// `ConfusionMatrix` は非スカラーのため列にしない。
pub(super) fn columns(has_validation: bool, metrics: &[Metrics]) -> Vec<&'static str> {
    let mut cols = vec!["loss", "lr"];
    if has_validation {
        cols.push("val_loss");
        for (m, name) in [
            (Metrics::Accuracy, "val_accuracy"),
            (Metrics::Precision, "val_precision"),
            (Metrics::Recall, "val_recall"),
            (Metrics::F1, "val_f1"),
        ] {
            if metrics.contains(&m) {
                cols.push(name);
            }
        }
    }
    cols
}

/// `History` の epoch `e` から列 `name` の値を取り出す。欠損は `None`。
fn value_of(name: &str, h: &History, e: usize) -> Option<f32> {
    match name {
        "loss" => h.loss.get(e).copied(),
        "lr" => h.lr.get(e).copied(),
        "val_loss" => h.val_loss.get(e).copied(),
        "val_accuracy" => h.val_metrics.get(e).and_then(|m| m.accuracy),
        "val_precision" => h.val_metrics.get(e).and_then(|m| m.precision),
        "val_recall" => h.val_metrics.get(e).and_then(|m| m.recall),
        "val_f1" => h.val_metrics.get(e).and_then(|m| m.f1),
        _ => None,
    }
}

/// 列ごとの値を集める。欠損は内部契約違反（`training.rs` が列を揃えて
/// から呼ぶ）として `Err`（panic させない）。
pub(super) fn collect_values(
    cols: &[&'static str],
    h: &History,
    e: usize,
) -> Result<Vec<f32>, String> {
    cols.iter()
        .map(|c| {
            value_of(c, h, e)
                .ok_or_else(|| format!("History の列 {c} に epoch {e} の値が無い（内部契約違反）"))
        })
        .collect()
}

/// CSV ヘッダ行（末尾の改行を含まない）。
pub(super) fn csv_header(cols: &[&str]) -> String {
    let mut s = String::from("epoch");
    for c in cols {
        s.push(',');
        s.push_str(c);
    }
    s
}

/// CSV データ行（末尾の改行を含まない）。f32 は `Display`（最短往復表現。
/// 非有限値は `NaN`／`inf`／`-inf`）。
pub(super) fn csv_row(epoch: usize, values: &[f32]) -> String {
    let mut s = epoch.to_string();
    for v in values {
        s.push(',');
        s.push_str(&v.to_string());
    }
    s
}

/// JSON の number／非有限値トークン。
fn json_number(v: f32) -> String {
    if v.is_nan() {
        "\"NaN\"".to_string()
    } else if v == f32::INFINITY {
        "\"Infinity\"".to_string()
    } else if v == f32::NEG_INFINITY {
        "\"-Infinity\"".to_string()
    } else {
        v.to_string()
    }
}

/// JSON の epoch オブジェクト 1 件（単一行）。
pub(super) fn json_row(cols: &[&str], epoch: usize, values: &[f32]) -> String {
    let mut s = format!("{{\"epoch\":{epoch}");
    for (c, v) in cols.iter().zip(values) {
        s.push_str(&format!(",\"{c}\":{}", json_number(*v)));
    }
    s.push('}');
    s
}

/// 要素文字列（各々 1 個の JSON オブジェクト）から配列全体を組み立てる。
pub(super) fn json_array(elems: &[String]) -> String {
    if elems.is_empty() {
        return "[]\n".to_string();
    }
    let mut s = String::from("[\n");
    s.push_str(&elems.join(",\n"));
    s.push_str("\n]\n");
    s
}

/// 先頭行を上限付きで読む（改行は含めない）。ファイルが空なら `None`。
/// 改行が見つからないまま `MAX_HEADER_BYTES` に達したら `Err`。
pub(super) fn read_first_line_limited(path: &Path) -> Result<Option<String>, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut r = BufReader::new(f.take(MAX_HEADER_BYTES));
    let mut buf = Vec::new();
    let n = r.read_until(b'\n', &mut buf).map_err(|e| e.to_string())?;
    if n == 0 {
        return Ok(None);
    }
    let ended = buf.last() == Some(&b'\n');
    if !ended && n as u64 >= MAX_HEADER_BYTES {
        return Err(format!(
            "先頭行が {MAX_HEADER_BYTES} バイトを超えても改行に達しない（CSV ではない可能性）"
        ));
    }
    if ended {
        buf.pop();
    }
    String::from_utf8(buf)
        .map(Some)
        .map_err(|_| "先頭行が UTF-8 ではない".to_string())
}

/// 既存ファイルが存在し非空かを返す。`NotFound` のみ「無い」（`Ok(false)`）として
/// 扱い、それ以外の取得失敗（権限・ELOOP 等）は `Err` にして書き込みへ進ませない
/// （空ファイル扱いでヘッダ・JSON 検証を省略して追記するのを防ぐ）。
pub(super) fn existing_nonempty(path: &Path) -> Result<bool, String> {
    match fs::metadata(path) {
        Ok(m) => Ok(m.len() > 0),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("既存ファイルのメタデータを取得できない: {e}")),
    }
}

/// ファイル全体を上限付きで UTF-8 文字列として読む。
pub(super) fn read_limited(path: &Path) -> Result<String, String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    if len > MAX_JSON_LOG_BYTES {
        return Err(format!(
            "既存ファイルが上限 {MAX_JSON_LOG_BYTES} バイトを超えている（{len} バイト）"
        ));
    }
    let mut buf = Vec::new();
    f.take(MAX_JSON_LOG_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| e.to_string())?;
    if buf.len() as u64 > MAX_JSON_LOG_BYTES {
        return Err(format!(
            "既存ファイルが上限 {MAX_JSON_LOG_BYTES} バイトを超えている"
        ));
    }
    String::from_utf8(buf).map_err(|_| "既存ファイルが UTF-8 ではない".to_string())
}

/// 親ディレクトリが無ければ作る（`ModelCheckpoint::persist` と同型。
/// 親が空文字列〈相対のファイル名のみ〉ならスキップ）。
pub(super) fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 一時ファイル名の一意化カウンタ（同一プロセス内の並行 fit 用）。
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 一時ファイル（同一ディレクトリ・`create_new`）→ `sync_all` →
/// `rename` で `bytes` を原子的に配置する。途中クラッシュ時も正規パスには
/// 旧ファイルか新ファイルのどちらか完全なものだけが残る。失敗時は自分が
/// 作った一時ファイルだけを best-effort で削除する。
pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let file_name = path
        .file_name()
        .ok_or_else(|| "パスにファイル名が無い".to_string())?
        .to_string_lossy()
        .into_owned();
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty());
    let pid = std::process::id();
    let mut created: Option<(std::path::PathBuf, File)> = None;
    for _ in 0..16 {
        let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp_name = format!(".{file_name}.tmp-{pid}-{n}-{nanos}");
        let tmp = match dir {
            Some(d) => d.join(tmp_name),
            None => std::path::PathBuf::from(tmp_name),
        };
        match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(f) => {
                created = Some((tmp, f));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    let (tmp, mut f) =
        created.ok_or_else(|| "一時ファイルの作成を再試行上限まで失敗".to_string())?;
    let res = f.write_all(bytes).and_then(|_| f.sync_all()).and_then(|_| {
        drop(f);
        fs::rename(&tmp, path)
    });
    if let Err(e) = res {
        let _ = fs::remove_file(&tmp);
        return Err(e.to_string());
    }
    Ok(())
}

/// 既存 JSON ログの 1 要素（トップレベル配列の 1 オブジェクト）。
#[derive(Debug)]
pub(super) struct ParsedElement {
    /// 原文の部分文字列（再シリアライズせずそのまま保持する）。
    pub(super) raw: String,
    /// そのオブジェクトのトップレベルキー（エスケープ復号後）。
    pub(super) keys: Vec<String>,
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while matches!(b.get(i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
        i += 1;
    }
    i
}

/// `b[i] == b'"'` から文字列を読み、復号した内容と次位置を返す。
fn scan_string(b: &[u8], start: usize) -> Result<(String, usize), String> {
    let mut i = start + 1;
    let mut out: Vec<u8> = Vec::new();
    loop {
        let c = *b.get(i).ok_or("文字列が閉じていない")?;
        match c {
            b'"' => {
                return Ok((String::from_utf8_lossy(&out).into_owned(), i + 1));
            }
            0..=0x1f => return Err("文字列内に制御文字がある".to_string()),
            b'\\' => {
                let e = *b.get(i + 1).ok_or("エスケープが途中で終わっている")?;
                match e {
                    b'"' | b'\\' | b'/' => out.push(e),
                    b'b' => out.push(0x08),
                    b'f' => out.push(0x0c),
                    b'n' => out.push(b'\n'),
                    b'r' => out.push(b'\r'),
                    b't' => out.push(b'\t'),
                    b'u' => {
                        let hex = b.get(i + 2..i + 6).ok_or("\\u エスケープが短い")?;
                        let mut v: u32 = 0;
                        for h in hex {
                            let d = (*h as char)
                                .to_digit(16)
                                .ok_or("\\u エスケープが 16 進でない")?;
                            v = v * 16 + d;
                        }
                        let mut i_adv = 4;
                        let ch = if (0xD800..0xDC00).contains(&v) {
                            // 上位サロゲートは直後の \uDC00..=\uDFFF と対でのみ有効。
                            let lo = if b.get(i + 6) == Some(&b'\\') && b.get(i + 7) == Some(&b'u')
                            {
                                let h2 = b.get(i + 8..i + 12).ok_or("\\u エスケープが短い")?;
                                let mut w: u32 = 0;
                                for h in h2 {
                                    let d = (*h as char)
                                        .to_digit(16)
                                        .ok_or("\\u エスケープが 16 進でない")?;
                                    w = w * 16 + d;
                                }
                                Some(w)
                            } else {
                                None
                            };
                            match lo {
                                Some(w) if (0xDC00..0xE000).contains(&w) => {
                                    i_adv = 10;
                                    char::from_u32(0x10000 + ((v - 0xD800) << 10) + (w - 0xDC00))
                                        .ok_or("不正なサロゲートペア")?
                                }
                                _ => return Err("対を持たない上位サロゲート".to_string()),
                            }
                        } else {
                            // 単独の下位サロゲートはここで None になり Err。
                            char::from_u32(v).ok_or("対を持たない下位サロゲート")?
                        };
                        let mut tmp = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                        i += i_adv;
                    }
                    _ => return Err("不正なエスケープ".to_string()),
                }
                i += 2;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
}

/// JSON number 文法（RFC 8259）に従って読み、次位置を返す。
fn scan_number(b: &[u8], mut i: usize) -> Result<usize, String> {
    let digits = |mut j: usize| {
        let s = j;
        while matches!(b.get(j), Some(b'0'..=b'9')) {
            j += 1;
        }
        (j, j > s)
    };
    if b.get(i) == Some(&b'-') {
        i += 1;
    }
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits(i).0,
        _ => return Err("不正な数値".to_string()),
    }
    if b.get(i) == Some(&b'.') {
        let (j, ok) = digits(i + 1);
        if !ok {
            return Err("不正な数値（小数部）".to_string());
        }
        i = j;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(b.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let (k, ok) = digits(j);
        if !ok {
            return Err("不正な数値（指数部）".to_string());
        }
        i = k;
    }
    Ok(i)
}

fn scan_literal(b: &[u8], i: usize, lit: &str) -> Result<usize, String> {
    if b.get(i..i + lit.len()) == Some(lit.as_bytes()) {
        Ok(i + lit.len())
    } else {
        Err("不正なリテラル".to_string())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Value,
    ValueOrArrEnd,
    KeyOrObjEnd,
    Key,
    Colon,
    AfterValue,
}

/// `b[i] == b'{'` から 1 個の JSON オブジェクトを明示スタックで読み、
/// 終端位置とトップレベルキーを返す（再帰しない）。
fn scan_object(b: &[u8], mut i: usize) -> Result<(usize, Vec<String>), String> {
    let mut keys: Vec<String> = Vec::new();
    let mut stack: Vec<u8> = Vec::new();
    let mut mode = Mode::Value;
    loop {
        i = skip_ws(b, i);
        let c = *b.get(i).ok_or("JSON が途中で終わっている")?;
        match mode {
            Mode::Value | Mode::ValueOrArrEnd => {
                if mode == Mode::ValueOrArrEnd && c == b']' {
                    stack.pop();
                    i += 1;
                    if stack.is_empty() {
                        return Ok((i, keys));
                    }
                    mode = Mode::AfterValue;
                    continue;
                }
                match c {
                    b'{' | b'[' => {
                        if stack.len() >= MAX_JSON_DEPTH {
                            return Err(format!("入れ子が深さ上限 {MAX_JSON_DEPTH} を超えた"));
                        }
                        stack.push(c);
                        i += 1;
                        mode = if c == b'{' {
                            Mode::KeyOrObjEnd
                        } else {
                            Mode::ValueOrArrEnd
                        };
                        continue;
                    }
                    b'"' => i = scan_string(b, i)?.1,
                    b'-' | b'0'..=b'9' => i = scan_number(b, i)?,
                    b't' => i = scan_literal(b, i, "true")?,
                    b'f' => i = scan_literal(b, i, "false")?,
                    b'n' => i = scan_literal(b, i, "null")?,
                    _ => return Err("値として不正な文字".to_string()),
                }
                if stack.is_empty() {
                    return Ok((i, keys));
                }
                mode = Mode::AfterValue;
            }
            Mode::KeyOrObjEnd | Mode::Key => {
                if mode == Mode::KeyOrObjEnd && c == b'}' {
                    stack.pop();
                    i += 1;
                    if stack.is_empty() {
                        return Ok((i, keys));
                    }
                    mode = Mode::AfterValue;
                    continue;
                }
                if c != b'"' {
                    return Err("オブジェクトのキーが文字列でない".to_string());
                }
                let (k, ni) = scan_string(b, i)?;
                if stack.len() == 1 {
                    keys.push(k);
                }
                i = ni;
                mode = Mode::Colon;
            }
            Mode::Colon => {
                if c != b':' {
                    return Err("キーの後に : が無い".to_string());
                }
                i += 1;
                mode = Mode::Value;
            }
            Mode::AfterValue => match (stack.last().copied(), c) {
                (Some(b'{'), b',') => {
                    i += 1;
                    mode = Mode::Key;
                }
                (Some(b'['), b',') => {
                    i += 1;
                    mode = Mode::Value;
                }
                (Some(b'{'), b'}') | (Some(b'['), b']') => {
                    stack.pop();
                    i += 1;
                    if stack.is_empty() {
                        return Ok((i, keys));
                    }
                }
                _ => return Err("区切りまたは閉じ括弧が不正".to_string()),
            },
        }
    }
}

/// 既存 JSON ログを「オブジェクトの配列」として検証し、各要素の原文と
/// トップレベルキーを返す。配列でない・要素がオブジェクトでない・構文
/// 不正・末尾ゴミはすべて `Err`。
pub(super) fn parse_log_array(src: &str) -> Result<Vec<ParsedElement>, String> {
    let b = src.as_bytes();
    let mut i = skip_ws(b, 0);
    if b.get(i) != Some(&b'[') {
        return Err("トップレベルが配列ではない".to_string());
    }
    i = skip_ws(b, i + 1);
    let mut out = Vec::new();
    if b.get(i) == Some(&b']') {
        i += 1;
    } else {
        loop {
            i = skip_ws(b, i);
            if b.get(i) != Some(&b'{') {
                return Err("配列の要素がオブジェクトではない".to_string());
            }
            let (end, keys) = scan_object(b, i)?;
            out.push(ParsedElement {
                raw: src[i..end].to_string(),
                keys,
            });
            i = skip_ws(b, end);
            match b.get(i) {
                Some(b',') => i += 1,
                Some(b']') => {
                    i += 1;
                    break;
                }
                _ => return Err("配列の区切りが不正".to_string()),
            }
        }
    }
    if skip_ws(b, i) != b.len() {
        return Err("配列の後に余分なデータがある".to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniq_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "fandhe-logger-io-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn columns_are_fixed_order_independent_of_metrics_order() {
        assert_eq!(columns(false, &[]), vec!["loss", "lr"]);
        assert_eq!(columns(true, &[]), vec!["loss", "lr", "val_loss"]);
        assert_eq!(
            columns(
                true,
                &[
                    Metrics::F1,
                    Metrics::ConfusionMatrix,
                    Metrics::Accuracy,
                    Metrics::F1
                ]
            ),
            vec!["loss", "lr", "val_loss", "val_accuracy", "val_f1"]
        );
    }

    #[test]
    fn csv_and_json_rows_use_fixed_non_finite_tokens() {
        let cols = ["loss", "lr"];
        assert_eq!(csv_header(&cols), "epoch,loss,lr");
        assert_eq!(csv_row(3, &[f32::NAN, f32::NEG_INFINITY]), "3,NaN,-inf");
        assert_eq!(csv_row(0, &[0.5, f32::INFINITY]), "0,0.5,inf");
        assert_eq!(
            json_row(&cols, 1, &[f32::NAN, f32::INFINITY]),
            "{\"epoch\":1,\"loss\":\"NaN\",\"lr\":\"Infinity\"}"
        );
        assert_eq!(
            json_row(&cols, 2, &[f32::NEG_INFINITY, 0.25]),
            "{\"epoch\":2,\"loss\":\"-Infinity\",\"lr\":0.25}"
        );
    }

    #[test]
    fn json_array_roundtrips_through_parser_and_keeps_raw_spans() {
        let cols = ["loss"];
        let elems = vec![json_row(&cols, 0, &[1.5]), json_row(&cols, 1, &[f32::NAN])];
        let text = json_array(&elems);
        let parsed = parse_log_array(&text).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].raw, elems[0]);
        assert_eq!(parsed[1].raw, elems[1]);
        assert_eq!(parsed[0].keys, vec!["epoch", "loss"]);
        assert_eq!(json_array(&[]), "[]\n");
        assert!(parse_log_array("[]\n").unwrap().is_empty());
    }

    #[test]
    fn parser_accepts_nested_values_and_escapes() {
        let src =
            r#" [ {"epoch": 0, "a\"b": [1, {"x": null}, "s\u00e9\n"], "t": true, "n": -1.5e+3} ] "#;
        let p = parse_log_array(src).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].keys, vec!["epoch", "a\"b", "t", "n"]);
    }

    #[test]
    fn parser_rejects_malformed_inputs_without_panicking() {
        for bad in [
            "",
            "{}",
            "[1]",
            "[{]",
            "[{\"a\":1}",
            "[{\"a\":1}] x",
            "[{\"a\":1},]",
            "[{\"a\" 1}]",
            "[{\"a\":01}]",
            "[{\"a\":1.}]",
            "[{\"a\":\"\\q\"}]",
            "[{\"a\":\"\\u12\"}]",
            "[{\"a\":\"\\uD800\"}]",
            "[{\"a\":\"\\uDC00\"}]",
            "[{\"a\":\"\\uD800x\"}]",
            "[{\"a\":\"\\uD800\\u0041\"}]",
            "[{\"a\":\"unterminated}]",
            "[{\"a\":tru}]",
        ] {
            assert!(parse_log_array(bad).is_err(), "should reject: {bad:?}");
        }
    }

    #[test]
    fn parser_accepts_surrogate_pair() {
        let v = parse_log_array("[{\"a\":\"\\uD83D\\uDE00\"}]").unwrap();
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn existing_nonempty_distinguishes_not_found_from_other_errors() {
        let d = std::env::temp_dir().join(format!("fandhe-ne-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        assert_eq!(existing_nonempty(&d.join("nope")), Ok(false));
        let f = d.join("f");
        std::fs::write(&f, b"x").unwrap();
        assert_eq!(existing_nonempty(&f), Ok(true));
        assert!(existing_nonempty(&f.join("child")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn parser_rejects_excessive_depth_without_stack_overflow() {
        let deep = format!("[{{\"a\":{}{}}}]", "[".repeat(100_000), "]".repeat(100_000));
        assert!(parse_log_array(&deep).is_err());
        let ok_depth = format!(
            "[{{\"a\":{}{}}}]",
            "[".repeat(MAX_JSON_DEPTH - 1),
            "]".repeat(MAX_JSON_DEPTH - 1)
        );
        assert!(parse_log_array(&ok_depth).is_ok());
    }

    #[test]
    fn write_atomic_replaces_and_leaves_no_temp_files() {
        let dir = uniq_dir("atomic");
        let path = dir.join("sub").join("log.json");
        ensure_parent_dir(&path).unwrap();
        write_atomic(&path, b"first").unwrap();
        write_atomic(&path, b"second").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
        let names: Vec<_> = fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["log.json".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn limited_readers_enforce_caps() {
        let dir = uniq_dir("limits");
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("h.csv");
        fs::write(&p, "epoch,loss,lr\n1,2,3\n").unwrap();
        assert_eq!(
            read_first_line_limited(&p).unwrap().as_deref(),
            Some("epoch,loss,lr")
        );
        fs::write(&p, "epoch,loss,lr").unwrap();
        assert_eq!(
            read_first_line_limited(&p).unwrap().as_deref(),
            Some("epoch,loss,lr")
        );
        fs::write(&p, "").unwrap();
        assert_eq!(read_first_line_limited(&p).unwrap(), None);
        fs::write(&p, vec![b'a'; MAX_HEADER_BYTES as usize + 10]).unwrap();
        assert!(read_first_line_limited(&p).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
