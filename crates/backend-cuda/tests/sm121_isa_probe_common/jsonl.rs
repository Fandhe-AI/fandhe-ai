//! `SM121_PROBE_JSON {...}` 行の組み立て（手書きのエスケープ。依存追加なし。
//! ユーザー承認 2026-10-01・イシュー #2122 計画 8.4）。
//!
//! 出力は stdout の 1 行 1 レコード。`aggregate.py` が `json.loads` で厳密に
//! 読む。値は文字列・整数・真偽のみ（浮動小数点は NaN／inf の扱いが曖昧に
//! なるため持たない）。`detail` は切り詰めない（ptxas ログの全文を残す）。

use std::fmt::Write as _;

/// ログ行の識別子（`aggregate.py` の `PREFIX` と一致させる）。
pub const PREFIX: &str = "SM121_PROBE_JSON ";

/// JSON 文字列リテラルの本体（前後の引用符を含まない）へエスケープする。
///
/// 引用符・バックスラッシュ・制御文字（U+0000〜U+001F）・DEL・非 ASCII を
/// すべてエスケープし、出力を ASCII のみにする（非 BMP は surrogate pair）。
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (c as u32) == 0x7f => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c if c.is_ascii() => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out
}

/// 1 レコード分のフィールド列（挿入順を保つ）。
#[derive(Debug, Default)]
pub struct Record {
    fields: Vec<(&'static str, Value)>,
}

#[derive(Debug)]
enum Value {
    Str(String),
    Int(u64),
    Bool(bool),
}

impl Record {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn str(mut self, key: &'static str, value: &str) -> Self {
        self.fields.push((key, Value::Str(value.to_string())));
        self
    }

    pub fn int(mut self, key: &'static str, value: u64) -> Self {
        self.fields.push((key, Value::Int(value)));
        self
    }

    pub fn boolean(mut self, key: &'static str, value: bool) -> Self {
        self.fields.push((key, Value::Bool(value)));
        self
    }

    /// `{"k":"v",...}` 形式へ直列化する（キーは呼び出し側が固定する
    /// ASCII 識別子のみ。実行時文字列をキーにしない）。
    pub fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (i, (key, value)) in self.fields.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "\"{key}\":");
            match value {
                Value::Str(s) => {
                    let _ = write!(out, "\"{}\"", escape(s));
                }
                Value::Int(n) => {
                    let _ = write!(out, "{n}");
                }
                Value::Bool(b) => {
                    let _ = write!(out, "{b}");
                }
            }
        }
        out.push('}');
        out
    }

    /// stdout へ 1 行出力する。Rust の stdout は行バッファ（LineWriter）の
    /// ため、パイプ越しでも改行ごとに flush され、後続段でハングしても
    /// 既出の段の記録が失われない。
    pub fn emit(&self) {
        println!("{PREFIX}{}", self.to_json());
    }
}
