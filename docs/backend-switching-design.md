# バックエンド切替構成: feature フラグなし cfg ベースの設計根拠（#51・TASK-2.1c）

イシュー #51「docs(backend): TASK-2.1c 切替構成の文書化・PoC-v2-5 突合」（親: TASK-2.1 = #48）に対応する。
TASK-2.1（親タスク、`docs/spec/05-tasks.md`）の受け入れ条件「構成根拠が PoC 番号つきで文書化されていること」を満たすためのドキュメント。設計そのものの正本は PoC-v2-5（`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`）と `04-requirements.md` REQ-2 であり、本ドキュメントは実装リポ側にその要点と出典を明記する（PR #233 の `docs/backend-metal-wgpu-decision.md` と同形式）。

## 判断サマリ

**バックエンド切替（CPU／CUDA／Metal）は feature フラグなしの cfg ベースを正式構成とする。**

- `cudarc`（CUDA）は **無条件依存＋動的ロード**（`dynamic-loading` feature）
- `objc2`／`objc2-foundation`／`objc2-metal`（Metal）は **`[target.'cfg(target_os = "macos")'.dependencies]` 分離**＋コード側 `#[cfg(target_os = "macos")]` 囲み
- `rayon`（CPU）は無条件依存

根拠は REQ-2 の受け入れ基準「単一の計算記述から、自作バックエンド抽象層の切替（`cudarc` 無条件依存＋動的ロード、`objc2` 系は `cfg(target_os = "macos")` 分離）で CPU・CUDA・Metal の 3 バックエンドが動作すること（PoC-v2-5 で実証済みの feature フラグなし構成を正式化する）」（`docs/spec/04-requirements.md` REQ-2 受け入れ基準 1 項目目）に基づく。

## 構成の 3 要素と PoC 突合

| 要素 | 構成 | PoC 出典 |
|------|------|---------|
| CUDA（`cudarc`） | 無条件依存＋動的ロード（`dynamic-loading` feature）。CUDA toolkit 非搭載環境でも `cargo build` が成立し、toolkit の要求は実行時のみ | PoC-v2-3（`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`）で実証。PoC-v2-5「実施内容 1」で 2 マシン構成（Apple M4 Max・NVIDIA DGX Spark GB10）として再確認 |
| Metal（`objc2`／`objc2-foundation`／`objc2-metal`） | `[target.'cfg(target_os = "macos")'.dependencies]` 分離＋モジュール／バイナリ本体を `#[cfg(target_os = "macos")]` で囲む。非 macOS（DGX Spark = aarch64 Linux）では該当コード・依存ごとビルド対象から外れる | PoC-v2-5「実施内容 1」（`docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`「1 クレートを 2 マシンでビルドする構成」節） |
| CPU（`rayon`） | 無条件依存 | PoC-v2-1（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`）採用実測（naive/blocked 比 約 6〜8.5 倍改善） |

PoC-v2-5 は上記構成で「同一クレートを Apple M4 Max（CPU・Metal 実行）と NVIDIA DGX Spark GB10（CPU・CUDA 実行）の両方でビルドする」ことを実測し、「事前にスケルトン（`Cargo.toml` + `tensor`/`rng`/`cpu_ref` のみ）を DGX Spark へ `rsync` してビルドを確認してから、カーネル本体の実装に進んだ」（同 README「実施内容 1」）というプロセスで、feature フラグなしのまま `cargo build --release` が両マシンで成立することを確認している。

## なぜ feature フラグを使わないか

1. **PoC-v2-5 の直接実証**: 「これにより非 macOS（DGX Spark = aarch64 Linux）では該当コードごとビルド対象から外れ、feature フラグなしで `cargo build --release` が両マシンで成立する」（同 README「実施内容 1」）。切替に feature が不要であることの直接根拠である。
2. **v1 の feature 切替は前提が消滅**: v1（Burn 基盤）は「型エイリアス + Cargo feature」でバックエンド切替を実現していたが（REQ-2 詳細節、`docs/spec/04-requirements.md:68`）、v2 は REQ-1 全面改定（イシュー #22）により `burn`・`cubecl`・`ndarray` 等への依存が禁止されているため、この前提自体が成立しない（同 REQ-2「2026-08-05 v2 前提差し替え」節）。
3. **検証マトリクス増大・経路欠落リスクの構造的排除**: feature 組合せが増えるほど CI での検証マトリクスが組合せ的に増加し、feature 指定漏れによる経路欠落（本来ビルドされるべきコードがビルドされない）が起きうる。cfg ベースはターゲット（`target_os` 等）から自動的に決定されるため、feature の指定ミスという失敗モード自体が存在しない。

## 実装リポでの正式化の対応表

PoC-v2-5 実証構成が、本リポのどこに正式反映されているかを示す（TASK-2.1a・イシュー #49・PR #234）。

| PoC-v2-5 実証構成 | 本リポの反映先 |
|------|------|
| `code/rust/Cargo.toml`（CUDA 無条件依存＋動的ロード・Metal cfg 分離） | workspace `Cargo.toml` `[workspace.dependencies]`（`cudarc =0.19.8` driver/nvrtc/dynamic-loading/cuda-13000/f16 feature。TASK-1.1b・イシュー #5） |
| 同上（CUDA 側の crate 構成） | `crates/backend-cuda/Cargo.toml`（`cudarc.workspace = true`。cfg 分離しない無条件依存。TASK-2.1a・イシュー #49） |
| 同上（Metal 側の crate 構成） | `crates/backend-metal/Cargo.toml`（`[target.'cfg(target_os = "macos")'.dependencies]` に `objc2`／`objc2-foundation`／`objc2-metal` を配置。TASK-2.1a・イシュー #49） |
| 「Mac・DGX Spark 2 マシンでのビルド確認」 | CI `build` ジョブ「cargo build (linux / aarch64-apple-darwin)」（TASK-2.1b・イシュー #50・PR #238。`.github/workflows/ci.yml`）。macOS runner 未登録のため `aarch64-apple-darwin` クロスターゲット・lib-only ビルドで Metal 有効経路（`cfg(target_os = "macos")`）をコンパイル検証する |

## スコープ外・関連事項

- **数値一致の前提条件**（FMA 契約統一・Metal precise math 明示）は REQ-2 の数値一致基準側の事項であり、本ドキュメントの対象（切替構成の設計根拠）には含めない。TASK-2.2（イシュー #52。TASK-2.2a は PR #239 で着手済み）が担保する。
- **CUDA を既定で有効化するか等の具体的な構成決定**は、REQ-2 でも「本要件でも未検証のまま残る」と明記された残存課題である（`docs/spec/04-requirements.md` REQ-2 受け入れ基準「バックエンド有効化構成（feature 追加の要否を含む）の決定」節）。本ドキュメントでは決定しない。TASK-2.5（`05-tasks.md`）が対応する。
- **ビルド・実行可否マトリクス**（各バックエンドの詰まりポイントの文書化）は TASK-2.4（`docs/backend-matrix.md`）の担当範囲であり、本ドキュメントには含めない。

## 出典

- 正本: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/README.md`「実施内容 1」節
- `docs/spec/04-requirements.md` REQ-2「マルチ GPU バックエンド対応（脱 CUDA）（2026-08-05 v2 前提差し替え）」
- `docs/spec/05-tasks.md` TASK-2.1〜2.5
- `docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`（`cudarc` 動的ロードによる CUDA 非搭載ビルド成立）
- `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`（CPU `rayon` 採用実測）
- `.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」
- `.claude/rules/deps-policy.md`（許容依存 8 区分: CUDA・Metal 区分の条件）
