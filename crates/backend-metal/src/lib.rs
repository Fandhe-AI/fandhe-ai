//! Metal バックエンド。
//!
//! Metal バインディング経路は `wgpu` ではなく `objc2-metal` 直接呼び出しを採用する
//! （TASK-1.8d、#41）。PoC-v2-4 実測（Apple M4 Max）で、同一アルゴリズム（tiled GEMM）
//! 比較において `objc2-metal` 直接実装が `wgpu`（境界検査無効化後）より約 2.3 倍高速
//! （size=4096: 2.123 TFLOPS 対 0.920 TFLOPS）であり、かつ `simdgroup_matrix`（8×8
//! ハードウェア行列演算命令）は WGSL に相当命令が存在せず `wgpu` 経由では原理的に
//! 到達不可能なことを確認済み（Metal 直接はさらに約 1.5 倍、3.134 TFLOPS）。
//! `wgpu` は 124 パッケージロックの大規模依存 + `pollster` を要するのに対し、
//! `objc2` 系は許容依存 3 crate（`.claude/rules/deps-policy.md`）で `unsafe` を
//! FFI 境界に局所化できる。判断根拠・実測値の全体は
//! `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「経路選定の比較判断」節（正本）と
//! `docs/backend-metal-wgpu-decision.md`（実装リポ側の要約。REQ-8 境界検査規約との
//! 関係も記載）を参照。「約 2.3 倍」は naga の自動境界検査を無効化した後の wgpu 値との
//! 比較であり、WGSL 側の手動境界チェック自体は維持された状態での計測である
//! （境界検査省略の正当化に用いない。REQ-8）。
//!
//! `tensor-core` の演算グラフノードを MSL カーネル（simdgroup 系命令を含む）へ変換して実行する。
//! バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証構成。REQ-2）とし、
//! `objc2` / `objc2-foundation` / `objc2-metal` は `cfg(target_os = "macos")` で分離する
//! （非 macOS 環境のビルドに影響を与えない。`.claude/rules/deps-policy.md`）。
//!
//! `backend-cpu` との数値一致は統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で
//! 検証する。丸め方針（FMA 契約）は Metal `simdgroup_multiply_accumulate` の既定 FMA 契約を
//! CPU 参照実装（`f32::mul_add`）と揃える（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。
//! `.claude/rules/coding-rust.md`）。カーネルの手動境界検査は最適化を理由に省略しない（REQ-8）。
//! FFI 境界の `unsafe`（objc2 系）は必要最小限に留め理由コメントを付す
//! （`.claude/rules/security.md`）。
//!
//! TASK-1.8a（#38）でデバイス・コマンドキュー・バッファ管理の基盤（[`context::MetalContext`]・
//! [`buffer::MetalBuffer`]・[`error::MetalError`]）を実装済み。MSL ライブラリのコンパイル・
//! パイプライン構築・ディスパッチ経路（naive GEMM）は TASK-1.8b（#39）、simdgroup カーネルは
//! TASK-1.8c（#40）で追加する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.1・TASK-1.8）。
//!
//! 非 macOS 環境ではモジュールごとコンパイル対象外になる（feature フラグなしの cfg ベース。
//! PoC-v2-5 実証構成・REQ-2）。

#[cfg(target_os = "macos")]
pub mod buffer;
#[cfg(target_os = "macos")]
pub mod context;
#[cfg(target_os = "macos")]
pub mod error;

#[cfg(target_os = "macos")]
pub use buffer::MetalBuffer;
#[cfg(target_os = "macos")]
pub use context::MetalContext;
#[cfg(target_os = "macos")]
pub use error::MetalError;
