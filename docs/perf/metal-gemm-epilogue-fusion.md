# Metal GEMM epilogue 融合（bias・activation）計測記録（#605）

イシュー #605「feat(backend-metal): elementwise カーネル追加・
`gemm_bias_act` 実融合化と Metal 実機での複合 WL 計測」の実測記録。
CUDA 側 `docs/perf/cuda-gemm-epilogue-fusion.md`（#599）と同一方針・同一
形式で記録する。

## 現状: 実機検証未完・ブロック中

本イシューの実装（`crates/backend-metal/src/shaders/elementwise.metal`・
`elementwise.rs`・`shaders/gemm.metal::gemm_tiled_bias_act`・
`gemm.rs::MetalGemm::run_tiled_bias_act_f32`・`ops.rs::MetalBackendOps` の
elementwise 5 演算実装・`gemm_bias_act` オーバーライド）は本セッションの
実行環境（Linux サンドボックス）で完結させた。Linux 上で実行可能な範囲
（`cargo fmt`・`cargo clippy --workspace`・`cargo test --workspace`・
`cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin`・
`cargo clippy -p fandhe-ai-backend-metal --all-targets --target aarch64-apple-darwin`）
はすべて green を確認済み（下記「Linux 側で確認済みの検証」参照）。

`docs/real-hardware-verification-env.md` §1 が示す Metal 実機
（Apple Silicon。Mac ローカル直接実行のみで SSH 経路が存在しない）へは
本セッションから到達できないため、以下は**未実施**である（実測値を
捏造しない。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
採用」・security.md の実測原則に従う。#599 の承認済み先例と同じ扱い）:

- `cargo test -p fandhe-ai-backend-metal --release --test gemm_bias_act_parity --
  --ignored --nocapture`（elementwise・`gemm_bias_act` の CPU-Metal 数値
  一致・融合 vs 非融合合成の複合判定・bias 形状グリッド・`k=0` 縮退）
- `crates/backend-metal/src/gemm.rs` 内クレート内テスト 2 件
  （`run_tiled_bias_act_f32_increments_fused_launch_counter`・
  `run_tiled_bias_act_f32_k_zero_does_not_increment_fused_launch_counter`。
  `cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture` に含む）
- 既存 `#[ignore]` 実機テスト全体（回帰確認。既存 GEMM カーネル
  〈`gemm_naive`／`gemm_tiled`／`gemm_simdgroup`／`gemm_simdgroup_tiled`／
  `gemm_simdgroup_f16`〉は本イシューで一切変更していないため理論上は
  非後退のはずだが、実機実測による確認は未実施）
- 融合 vs 非融合合成（`gemm`→`add`→`relu`）の 5 回計測中央値ベンチ
- Transformer 複合ワークロード（G-14）の適用前後計測（
  `docs/perf/transformer-workload-baseline.md` §7 Metal 記入枠。Metal
  実行経路自体〈`crates/backend-metal/tests/transformer_workload_metal.rs`〉
  は本イシューでは未着手のまま別イシューへ切り出す。下記「スコープ外へ
  切り出した事項」参照）
- GEMM 単体（REQ-8 Metal f32/f16 行）の非劣化再計測（
  `docs/perf/metal-floor-remeasurement.md` の手順）

## 実機検証時の再現コマンド（未実施・手順のみ記録）

```bash
# docs/real-hardware-verification-env.md §1 の手順で Mac 実機へ
# ブランチを取得したうえで、実機上で実行する。

# 1. 新規テスト（elementwise・gemm_bias_act の CPU-Metal 数値一致・
#    融合 vs 非融合合成の複合判定・bias 形状グリッド・k=0 縮退）
cargo test -p fandhe-ai-backend-metal --release --test gemm_bias_act_parity \
  -- --ignored --nocapture

# 2. 既存実機テスト全体（回帰確認。融合カーネル起動カウンタの
#    in-crate テスト 2 件を含む）
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture

# 3. GEMM 単体（REQ-8 Metal f32/f16 行）の非劣化再計測
#    docs/perf/metal-floor-remeasurement.md の手順に従う
```

## Linux 側で確認済みの検証

| 項目 | 結果 |
|------|------|
| `cargo fmt --all -- --check` | green |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | green |
| `cargo test --workspace` | green |
| `cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin` | green |
| `cargo clippy -p fandhe-ai-backend-metal --all-targets --target aarch64-apple-darwin -- -D warnings` | 新規コードに起因する warning なし（既存 4 件は本イシュー無関係の pre-existing。`git stash` 比較で確認済み） |
| `elementwise_gemm_bias_act_source_evidence.rs`（Linux 実行） | green（6 件） |

## 実測すべき項目（実機到達後に本節を実測値で置き換える）

| 項目 | 状態 |
|------|------|
| CPU-Metal `gemm_bias_act` 数値一致（REQ-2 複合判定） | 未実施 |
| Metal 上の融合 vs 非融合合成の複合判定一致 | 未実施 |
| elementwise 5 演算の CPU-Metal 数値一致 | 未実施 |
| bias 形状グリッド（`[n]`・`[1]`・`[1,n]`・不整合拒否・`k=0` 縮退） | 未実施 |
| `BIAS_ACT_FUSED_LAUNCH_COUNT` によるフォールバック非経由の確認 | 未実施 |
| 融合 vs 非融合の 5 回計測中央値ベンチ | 未実施 |
| Transformer 複合 WL（G-14）適用前後の計測（実行経路自体が別イシュー） | 未実施（実行経路未実装。下記参照） |
| GEMM 単体（REQ-8 Metal f32/f16 行）の非劣化再計測 | 未実施 |

## スコープ外へ切り出した事項

- **Metal 複合ワークロード実行経路（G-4 §3.3 相当）**: `crates/bench-harness`
  の forward 合成を線形層の実行経路注入可能な形へ一般化し、
  `crates/backend-metal/tests/transformer_workload_metal.rs`（適用前後の
  計測）を追加する作業は、CPU 側テスト（`crates/bench-harness/tests/
  transformer_workload.rs`）の挙動不変性を崩さずに行うための設計検討・
  実装量が大きく、かつ CUDA 側（G-12・#602）でも同型の作業が本ファイル
  作成時点で未着手であることを確認した。実機検証できない本環境で拙速な
  リファクタリングを行うリスクを避けるため、本イシューでは実装せず
  別イシューへ切り出す（ユーザー承認を得たうえで out-of-scope-tracking.md
  に従い追跡する）。
- **simdgroup 系カーネルへの epilogue 適用**: 実装計画どおりスコープ外
  （`gemm_tiled` ベースの `gemm_tiled_bias_act` のみ追加）。

## マージ可否についての注記

実機検証未完のため、本 PR のマージ可否はレビュー・ユーザー判断に委ねる
（#599 の承認済み先例に従う安全側判断）。CI で実行可能な範囲（fmt／
clippy／単体テスト／Linux 側クロス型検査）はすべて green であることを
上記「Linux 側で確認済みの検証」で確認済み。
