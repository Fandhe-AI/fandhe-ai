# Metal f32 `max` reduction の Mac mini 実機 parity（イシュー #3475）

`MetalBackendOps::max`（全要素 2 段・単一軸 1 段の GPU カーネル。NaN 読み飛ばし・全 NaN は `-inf`・±0 混在は `+0`・CPU 参照と bit 完全一致の契約。設計は [`backend-metal-reduce-sum-design.md`](../../../backend-metal-reduce-sum-design.md) §14）の `#[ignore]` 実機テスト 9 本を Mac mini（Apple M4）で実測した記録。カーネル実装と CPU 参照との bit 一致テストは PR #3479 で導入済みで、本記録は実測のみ（コード・tolerance 定数・parity baseline・ガードレール閾値・依存は不変）。

- 判定規則: 事前登録の [`RULE.txt`](./RULE.txt)（実機実行より前の独立コミットで固定。事後に変更していない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（`../metal-facade-parity-3166/run_metal_parity.sh` の骨格を流用）
- 実行日: 2026-10-19（UTC）。機体: Mac mini（Apple M4・macOS 27.0.1・rustc 1.99.0）。ホスト名は記録しない
- 対象コミット: `922c285e087c620ef95e41426f832d9819867486`（RULE.txt を追加したコミット。`rev-stamp.txt` は 40 桁 1 行。送信前のローカル `git status --porcelain` は 0 件）
- 展開ツリーの確認: `git archive` 由来のツリーの `crates/` 指紋（`find crates -type f | sort | xargs shasum | shasum`）はローカルとノードで一致（`931d909e52d880c618ee9db114de6e73755fdf10`）
- 有効条件: `b6c0ae40ad63ee35260b9ae57f6c6066b2e4efe4`（PR #3479 のマージコミット）との間で、`crates/*/src`・`crates/*/Cargo.toml`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` と対象テストファイル 6 本に差分がないことを確認した（`crates/` 配下の差分はテストのみの別 2 ファイル）。**実行したコードは #3479 のマージ内容と同じ**
- 実行環境: debug プロファイル・各テスト 1 回のみ・再実行なし・perl alarm 1800 秒。展開先は専用の `~/work/rust-ai-library-run-3475`、`CARGO_TARGET_DIR` は共有 target（§3 既定からの意図的な逸脱は並列セッションとの衝突回避。RULE.txt 参照）
- 並走: 実行前に他の cargo／rustc プロセスなし（待機 0 回。`env_info.txt`）。判定外の `--no-run` ビルドを先に実施（`build.log`）
- 生ログ・環境・整合性: `logs/<群>/<テスト関数名>.log`（末尾に `rc=`）・`env_info.txt`・`summary.tsv`・`SHA256SUMS`。ホスト名・ユーザー名・ホームパスのみマスク済みで、手で編集していない。マスク後に各ログで `test <対象テスト名> ... ` の行がちょうど 1 行残ることを確認済み（スクリプトが検査）

## 結果

| # | パッケージ・ターゲット | テスト | 結果 |
|---|---|---|---|
| 1 | backend-metal `reduce_parity` | `metal_max_all_matches_cpu_bit_exact` | pass |
| 2 | 同上 | `metal_max_axis_matches_cpu_bit_exact` | pass |
| 3 | 同上 | `metal_max_nan_zero_inf_edge_cases_match_cpu_bit_exact` | pass |
| 4 | 同上 | `metal_max_signaling_nan_matches_cpu_bit_exact` | pass |
| 5 | backend-metal `backend_ops_real_device` | `backend_ops_max_matches_cpu_bit_exact` | pass |
| 6 | backend-metal `typed_ops_f16_parity` | `max_matches_f32_backend_ops_rounded_bit_exact` | pass |
| 7 | backend-metal `--lib` | `typed_bf16::tests::max_matches_f32_backend_ops_rounded_bit_exact` | pass |
| 8 | fandhe-ai `reduce_backend_parity` | `metal_max_all_and_axis_forward_and_backward_match_cpu_bit_exact` | pass |
| 9 | fandhe-ai `compat_sequential_dropout_embedding_bag_backend_parity` | `metal_dropout_embedding_bag_bit_exact` | pass |

合計: 9 件中 pass 9／fail 0／undetermined 0。全件 rc=0・`running 1 test`・`1 passed; 0 failed`。

- `Unsupported` は返らず、CPU 参照と bit 一致した（NaN〈quiet・signaling〉読み飛ばし・全 NaN は `-inf`・±0 混在は `+0` を含む）。Mac mini は aarch64 のため `cfg(target_arch = "aarch64")` 限定の ±0 厳密 assert も有効。
- #3164 で `Unsupported` により fail した `metal_dropout_embedding_bag_bit_exact`（9 番）は本測定で pass。#3164 の fail 記録（コメント・クローズ）は履歴として残し、本 PR では触れない。
- GPU 経路について: `max` は GPU カーネルへ結線済みでホストフォールバックを持たない（設計 doc §14）。ただし本ログ自体から経路を直接観測してはいないため、経路は設計 doc とコードに依拠する（ログ上の観測は未確認）。
- 対象外: `reduce_parity.rs` の sum・argext・min、他イシューの Metal 実測の測り直し、`max` に関する古い「Unsupported」記述の更新。
