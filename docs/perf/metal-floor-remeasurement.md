# Metal 最適化後 f32/f16 対 PyTorch MPS 比 確定計測 記録（#572・Phase F-2）

イシュー #572「bench(backend-metal): 最適化後の Metal f32/f16 対 PyTorch MPS 比を確定計測」の実測記録。
GEMM 最適化ツリー（ルート #479）の Phase F（親 #569「再計測・parity 非後退確認・REQ-8 下限再確定」）の
F-2 に対応する。

## 目的・受け入れ条件対応

Phase D（Metal マルチ simdgroup 化・ロード最適化。親 #530、D-1〜D-5 が本番経路〈`MetalGemm::dispatch_auto`〉
に適用済み）完了後の Metal f32/f16 GEMM について、`docs/performance-targets.md` §4 の計測プロトコル
（warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3、ホスト転送を伴わない完了待ち、決定的シード、判定
対象形状 2048/4096）で対 PyTorch MPS 比を**確定計測**し、既存 `docs/perf/` と同形式で記録する。

本イシューの核心は f32 側の計測境界問題の解消である。`docs/perf/gemm-optimization-baseline.md` §2 が
確定したとおり:

- Metal f32 の現行ベンチ入口 `MetalGemm::dispatch_auto`（`crates/backend-metal/src/gemm.rs:297`）は
  1 ディスパッチごとに A/B アップロード＋C readback を含む「転送込み」境界であり、単独では §4 の
  同期方式契約（ホスト転送を伴わない完了待ち）を満たさない
- f16 側には §4 準拠の prepared 入口 `dispatch_f16_prepared_unverified`（同 `gemm.rs:475`。エンコード＋
  コマンドバッファ完了待ちのみ計測）が既に存在するが、**f32 側には同型の prepared 入口が存在しなかった**
- 同ドキュメント §2 は「(i) f16 と同型の §4 準拠 prepared ディスパッチ入口を f32 側にも用意したうえでの
  f32 再計測は Phase F の #572 のスコープ」と明記していた

本イシューでこの (i) を解消する [`MetalGemm::dispatch_tiled_prepared`]（`crates/backend-metal/src/gemm.rs`）
を追加した（下記「実測バイナリ」参照）。

依存 #547（D-10。`docs/perf/metal-gemm-dynamic-tile.md`「Phase D 完了時点再計測」節）は close 済み
（PR #696。計測手順・記録テンプレート整備済み）。

## 実行環境の制約（本ドキュメント作成セッション）

本ドキュメントは Linux worktree で作成された。`docs/real-hardware-verification-env.md` §1 のとおり
Metal 実機（Apple M4 Max）は「ローカル直接実行」であり、本 Linux 環境からは到達できない（SSH リモートは
CUDA ノードのみ）。よって本イシューは #547（D-10）の先例と同方式を採る:

1. f32 prepared 入口のコード整備（Linux でビルド・clippy・非実機テストまで検証可能）
2. ベンチ入口・parity テストの整備
3. 計測手順＋記録テンプレートの完全整備

**実測値の記入は Mac 実機セッションへ申し送る**（下記「状態」節参照）。

## 実測バイナリ

### f32: `crates/backend-metal/examples/gemm_f32_prepared_bench.rs`（本イシューで新規追加）

- `MetalGemm::dispatch_tiled_prepared`（本イシューで新規追加。`crates/backend-metal/src/gemm.rs`）を
  使う。`tile::select(m, n, k)` で選んだ [`TileConfig`] 候補と、事前確保・アップロード済みの
  [`MetalBuffer`]（A/B/C。実効次元 8 の倍数へ [`pad8`] 済み）を渡し、エンコード＋コマンドバッファ
  完了待ちのみを計測する（f16 側 `dispatch_f16_prepared_unverified` と同型の計測境界）
- `pipeline_for_tile` がデバイス上限超過等でサイレントに `TileConfig::SINGLE_SIMDGROUP_8X8` へ
  フォールバックしうるため、`dispatch_tiled_prepared` は実際に採用された構成（resolved）を返す。
  ベンチ出力の `resolved_tile_config=` にこれを含め、フォールバック透明性を確保する
- 既存 `gemm_bench.rs`（`dispatch_auto` 経由・転送込み境界。#381 比較系列）は改変しない。両者は
  独立した計測系列として維持する
- 形状: M=N=K = 512／1024／2048／4096（`gemm_f16_bench.rs` と同一形状帯。512 は起動オーバーヘッド
  支配のため参考値）
- 計測プロトコル: `bench_harness::protocol::run`（warmup 20 回以上・計測 20 回以上・中央値/Q1/Q3。
  TASK-8.1）・決定的シード `0xC0FFEE`

### f16: `crates/backend-metal/examples/gemm_f16_bench.rs`（既存。イシュー #156・#380 で確立済み）

`MetalGemm::dispatch_f16_prepared_unverified` を使う既存バイナリをそのまま再利用する（変更なし）。

## 入力検証（OWASP A03。`.claude/rules/security.md`）

`dispatch_tiled_prepared` の呼び出し元は任意の実効次元・バッファ長を渡せるため、エンコード（FFI）前に
[`validate_prepared_inputs_f32`]（`crates/backend-metal/src/gemm.rs`）が以下を fail-closed で検証する
（f16 版 `validate_prepared_inputs`・PR #346 codex-review P1-1 指摘と同水準）:

1. `m_eff`/`n_eff`/`k_eff` がいずれも 8 の倍数であること
2. `a_buf.len() == m_eff*k_eff`・`b_buf.len() == k_eff*n_eff`・`c_buf.len() == m_eff*n_eff`

回帰確認は `tests/gemm_dynamic_tile_parity.rs::dispatch_tiled_prepared_rejects_undersized_and_misaligned_inputs`
（`#[ignore]`・Metal 実機依存。`MetalBuffer` の確保に Metal デバイスが必要なため Linux 上の pure 単体
テストは書けない。`crates/backend-metal/src/gemm.rs` 内コメント参照）で行う。

## 数値一致（parity）確認

`tests/gemm_dynamic_tile_parity.rs::dispatch_tiled_prepared_matches_dispatch_variant`（`#[ignore]`・
Metal 実機依存）が、`dispatch_tiled_prepared`（prepared 入口）と `dispatch_variant`（一括入口）の
出力が完全一致することを確認する（計測境界のみが異なる同一カーネル呼び出しのため）。既存 tolerance
定数・REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は変更しない。

## 計測手順（Apple Silicon 実機）

```sh
git fetch origin
git checkout bench/572-metal-floor-remeasurement   # 本イシューの実装ブランチ

# 1. 数値一致確認を先に行う（既存 parity テスト群。閾値は緩和しない）
cargo test -p backend-metal --release -- --ignored --nocapture

# 2. Rust 側ベンチを各 5 回独立実行し、size ごとに中央値を採用する
#    （MeasurementConfig::default() 自体が warmup 20・計測 20・中央値を内包するため、
#    5 プロセス独立実行との組み合わせで「5 回計測の中央値」下限
#    〈.claude/rules/coding-rust.md〉を二重に満たす。#547 先例と同方式）
cargo run -p backend-metal --example gemm_f32_prepared_bench --release
cargo run -p backend-metal --example gemm_f16_bench --release
```

PyTorch 側は一時 venv（リポジトリ管理外。`.venv-mps-bench` 先例）で実行する:

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch
python3 scripts/bench/gemm_bench_torch_mps_f32.py
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

Rust 側と同様に各 5 回独立実行し、size ごとの中央値を採用する。

計測衛生（#381・#383・#547 先例と同方式）: AC 電源接続、外部ディスプレイのコンポジタ負荷を許容するが
他 GPU 負荷アプリ（ブラウザ動画・Xcode ビルド・ローカル LLM 等）は終了する。Rust 側・PyTorch 側の
同時実行を避け、各ラン前後に
`pgrep -fl "gemm_f32_prepared_bench|gemm_f16_bench|gemm_bench_torch_mps"` で他プロセスとの競合が
ないことを確認する（競合検出時は破棄・取り直す）。

## 計測環境（実測時に記入）

| 項目 | 値 |
|------|-----|
| チップ | （未計測） |
| OS | （未計測） |
| rustc | （未計測） |
| torch | （未計測） |
| 計測コミット SHA | （未計測） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20・計測 20・中央値/Q1/Q3。TASK-8.1）を 5 回独立実行し size ごとに中央値採用（Rust・PyTorch 双方） |
| 決定的シード | `0xC0FFEE` |
| 同期境界 | Rust: コマンドバッファ完了待ち（f32: `dispatch_tiled_prepared`／f16: `dispatch_f16_prepared_unverified`）／PyTorch: `torch.mps.synchronize()` |

## f32 結果（`dispatch_tiled_prepared`。§4 準拠 prepared 入口）

| size | Metal f32 TFLOPS（5 回中央値） | 採用 TileConfig（resolved） | PyTorch MPS f32 TFLOPS（5 回中央値） | Metal/PyTorch 比 |
|------|------|------|------|------|
| 512  | （未計測） | （未計測） | （未計測） | （未計測） |
| 1024 | （未計測） | （未計測） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） | （未計測） | （未計測） |

判定対象形状（REQ-8「判定対象形状」節）は **M=N=K=2048・4096 の実測比率の最小値**。512/1024 は参考値。

候補下限値（参考算出。`bench_harness::floor_lower_bound` を用いる）: （未計測）

## f16 結果（`dispatch_f16_prepared_unverified`。既存入口）

| size | Metal f16 TFLOPS（5 回中央値） | PyTorch MPS f16 TFLOPS（5 回中央値） | Metal/PyTorch 比 | 対 #383 分母改善率 |
|------|------|------|------|------|
| 512  | （未計測） | （未計測） | （未計測） | （未計測） |
| 1024 | （未計測） | （未計測） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） | （未計測） | （未計測） |

判定対象形状は f32 と同じく M=N=K=2048・4096 の最小値。

候補下限値（参考算出。`bench_harness::floor_lower_bound` を用いる）: （未計測）

## REQ-8 下限値の扱い

**REQ-8 下限値（初期リリース 20%／最適化後 30%、f16 15%／未設定）は本ドキュメントでは変更しない。**
変更は F-5（#577・人間承認タスク）のみが行う。本ドキュメントは候補下限値の参考算出（上記
`bench_harness::floor_lower_bound` 欄）を提供するに留め、下限の最終確定・
`docs/spec/04-requirements.md` への反映判断は行わない（`docs/spec/` は本リポでは編集しない）。

## 状態: 未計測。実機セッションで消化

本ドキュメントは Linux worktree で作成され、Metal 実機（Apple Silicon）が同一セッションで使用できない
ため計測手順・記録テンプレートのみを整備した（#547 節・`metal-gemm-float4-staged-load.md` 先例と同
方式）。実機到達可能なセッションが「計測手順」節の手順で計測し、上記「f32 結果」「f16 結果」「計測
環境」の各表を実測値で埋めること。

**#547 節（`docs/perf/metal-gemm-dynamic-tile.md`「Phase D 完了時点再計測」）の未計測テンプレートの
記入は本イシューのスコープ外**（close 済みイシューの記録）。同一 Mac セッションで併せて埋める判断は
実機セッション側に委ねる。

内部ホスト名等の実値は書かない（#461 のプレースホルダ方針。実測時の原文は
`docs/real-hardware-verification-env.local.md` へ記録する）。

## 動作確認（Linux セッションで実施済み）

- `cargo build --workspace --all-targets`
- `cargo build -p backend-metal --examples --release`（`gemm_f32_prepared_bench` を含む stub ビルド
  成立を確認）
- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`（Linux 実行分。実機依存 `#[ignore]` テストは除外）
- `git diff --stat` で `crates/bench-harness/src/threshold.rs`・数値一致 tolerance 定数・
  `docs/spec/`・`guardrail.toml` に差分がないことを確認

## 未実施・後続作業

- **実機実測**: 「状態」節のとおり本イシューでは未実施。実機セッションへ申し送る
- **候補下限値の最終確定・REQ-8 反映判断**: F-5（#577・人間承認）が実測完了後に対応する
