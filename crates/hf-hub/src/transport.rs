//! JSON API 転送層の抽象（イシュー #3131）。
//!
//! 本番実装（`ureq`・第 11 区分）は HF クレートでの利用承認が未取得のため結線していない
//! （設計 §9.12）。`api.rs` のロジックはこの trait 越しにテストされる。
//! 実装側の契約: `https` 限定・証明書検証有効・リダイレクト非追従（3xx はそのまま返す）・
//! 全タイムアウト明示・環境 proxy 無効・エラー文字列に URL を含めない。

use crate::HfError;
use crate::url::ApiUrl;

/// 転送層の応答（ステータスと上限付きで読んだボディ）。
pub(crate) struct JsonResponse {
    pub(crate) status: u16,
    pub(crate) body: Vec<u8>,
}

pub(crate) trait JsonTransport {
    /// GET する。`max_bytes` を超える受信は実装側でも打ち切ること（呼び出し側も再検査する）。
    fn get_json(&mut self, url: &ApiUrl, max_bytes: usize) -> Result<JsonResponse, HfError>;
}
