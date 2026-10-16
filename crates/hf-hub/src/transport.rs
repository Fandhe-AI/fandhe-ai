//! JSON API 転送層の抽象（イシュー #3131）。
//!
//! 本番実装 `UreqJsonTransport`（`ureq`・第 11 区分・`cfg(unix)`）は #3393 で結線した
//! （`transport/ureq_json.rs`。承認根拠 #3128 コメント 6220036919）。`api.rs` のロジックは
//! この trait 越しにテストされる。
//! 実装側の契約: `https` 限定・証明書検証有効・リダイレクト非追従（3xx はそのまま返す）・
//! 全タイムアウト明示・環境 proxy 無効・エラー文字列に URL を含めない。

use crate::HfError;
use crate::url::ApiUrl;

#[cfg(unix)]
mod ureq_json;

/// 公開化（#3394）まで本番コードからは呼ばれないため非 test では dead_code を許す。
#[cfg(unix)]
#[allow(dead_code, unused_imports)]
pub(crate) use ureq_json::UreqJsonTransport;

/// 転送層の応答（ステータスと上限付きで読んだボディ）。
#[derive(Debug)]
pub(crate) struct JsonResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

pub(crate) trait JsonTransport {
    /// GET する。`max_bytes` を超える受信は実装側でも打ち切ること（呼び出し側も再検査する）。
    fn get_json(&mut self, url: &ApiUrl, max_bytes: usize) -> Result<JsonResponse, HfError>;
}
