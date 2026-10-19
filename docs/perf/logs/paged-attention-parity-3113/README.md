# paged attention（K-3）の CPU 参照一致と CUDA／Metal 実機 parity（#3113）未実測の申し送り

> **Metal は #3172 で Mac mini（Apple M4）実測済み（2026-10-19・対象 main `b640875f`・`dirty=0`・5 回すべて pass）。** 判定規則は同ディレクトリの `RULE.txt`（#3113 で固定済み）を正とする。

`docs/device-resident-kv-paged-attention-design.md` §5・§14 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、下記の `#[ignore]` テストは**未実測**である。実測は #3171（GB10）・#3172（Mac mini）が担当する。
事前宣言は同ディレクトリの `RULE.txt`（実測前に固定し、事後に緩めない）。

## 対象テスト

| 層 | テスト（ファイル） | 意図 |
|---|---|---|
| backend | `cuda_paged_attention_matrix_matches_cpu_reference`（`crates/backend-cuda/tests/paged_kv_real_device.rs`） | `CASES` 全件（ブロック境界 ±1・複数ブロック・s_max>128・Dh>128・Dh=1・複数トークン追記・巻き戻し後の再追記・大振幅）を CPU 参照と比較。fail は集めて最後に 1 回 panic |
| backend | `cuda_paged_attention_tolerance_constants_pinned`（同上） | 実機ログにも tolerance 定数の固定の証跡を残す |
| backend | `metal_paged_attention_matrix_matches_cpu_reference`・`metal_paged_attention_tracked_matrix_matches_cpu_reference`・`metal_paged_attention_tolerance_constants_pinned`（`crates/backend-metal/tests/paged_kv_real_device.rs`） | 同上。tracked 版は `DispatchFailureCell` が立たないことも確認 |
| facade | `cuda_paged_attention_forward_matches_cpu_tape`・`metal_paged_attention_forward_matches_cpu_tape`（`crates/facade/tests/paged_attention_facade.rs`。後者は `cfg(target_os = "macos")`） | `tape_for(device)` と CPU tape へ同じ重み・入力を流す端から端まで。射影 GEMM を含むため判定は厳密のみ |

CPU 側（CI で実行済み）: `paged_attention_facade.rs` の `block_boundary_matrix_matches_k1`・`long_multi_block_sequence_matches_k1`・
`empty_cache_zero_length_step_then_prefill_matches_k1`・`clear_then_reuse_matches_fresh_k1`・
`rewind_then_reappend_matches_k1_on_truncated_sequence`・`out_of_range_inputs_are_err_and_keep_state`。
判定構成の契約は `crates/backend-{cuda,metal}/tests/paged_attention_judgement_contract.rs`、
マトリクス駆動部の CPU 自己検査は `matrix_harness_runs_on_cpu_against_cpu`。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で。ノード上での実行形式は docs/real-hardware-verification-env.md §4／§7 に従う）
cargo test -p fandhe-ai-backend-cuda --release --test paged_kv_real_device -- --ignored --nocapture cuda_paged_attention
cargo test -p fandhe-ai --release --test paged_attention_facade -- --ignored --nocapture cuda_paged_attention

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai-backend-metal --release --test paged_kv_real_device -- --ignored --nocapture metal_paged_attention
cargo test -p fandhe-ai --release --test paged_attention_facade -- --ignored --nocapture metal_paged_attention
```

## ログ書式

backend 層は 1 ステップ 1 行を標準出力へ出す（`--nocapture` 必須）。書式は契約テスト `metric_line_format_is_pinned` で固定している。

```
PAGED_PARITY backend=<cuda|metal|metal-tracked> case=<id> step=<n> total=<要素数> fail=<件数> max_abs=<値> max_rel=<値> mean_abs=<値> judgement=<strict|baseline>
```

## 手順

1. 実測前に `RULE.txt` をコミットして固定する（本 PR で固定済み）。
2. 上記コマンドを独立プロセスで 5 回実行し、全出力をログとして保存する。計測中はビルドを並走させない。
3. 合否は「5 回すべて pass で pass」。`PAGED_PARITY` 行の数値は 5 回の中央値を記録し、不一致があれば事実として記録する。
4. ログは本ディレクトリ配下へ置く。ホスト名・個人パスは `<cuda-node>`・`<metal-node>`・`<home>` へ置換し、`.rev-stamp`（実測したコミット）を同梱する。
5. `docs/README.md` の `logs/paged-attention-parity-3113/` の行へ実測の要約を追記する（索引の更新は README 側で行う。CLAUDE.md には書かない）。

## fail したとき

- tolerance 定数・判定式・カーネルは変えない（`.claude/rules/coding-rust.md`）。
- `PAGED_PARITY` の `case`・`step`・`total`・`fail`・`max_abs`・`max_rel`・`mean_abs` を転記して事実を記録し、申し送る。
- baseline 非後退判定へ切り替える場合は、`crates/backend-*/tests/common/paged_attention_judgement.rs` の `BASELINES` へ
  実機実測値だけで行を足す（`case_id`・`step`・`total`・各 ceiling・`measured_on`・`log_path`・`approval`）。
  人間承認が必須で、別 PR とし、契約テスト `baselines_are_empty_until_measured_and_approved` の assert を同 PR で変える。
- facade 層（射影を含む合成経路）の fail は、backend 層マトリクスの結果と合わせて事実を記録する。baseline 切り替えは backend 層の表で扱う。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測（#3171） | DGX Spark GB10 | 上記 cuda コマンド（backend・facade） | 未実測 | |
| 2026-10-19 | Mac mini（Apple M4） | 上記 metal コマンド（backend・facade。実測は `--test-threads=1` 無し・`--release`） | pass（backend 3 件 × 5 回、facade 1 件 × 5 回、全 pass） | main `b640875f`・dirty=0。`PAGED_PARITY` は 5 回完全一致 |

## Metal 実測記入欄（#3172）

- 判定規則: `RULE.txt`（実測前に固定済み。本実測コミットでは変更していない）。baseline 切り替えは 0 件（`BASELINES` は空のまま）で、全ステップ `judgement=strict`
- 実行日: 2026-10-19（UTC）
- 対象コミット: main `b640875f86399515b2dab4dfc38089caaa5e5c08`（`metal/rev-stamp.txt` の 2 行目は `dirty=0`。`metal/env_info.txt` の実行前後の stamp と一致）
- 機体: Mac mini（Apple M4（10 コア）・メモリ 16GiB・macOS 27.0.1（26A434）・rustc 1.99.0・cargo 1.99.0・arm64（`metal/env_info.txt`））
- 実行コマンド（独立プロセスで各 5 回。「測定コマンド案」の Metal 2 行のとおり）:
  - `cargo test -p fandhe-ai-backend-metal --release --test paged_kv_real_device -- --ignored --nocapture metal_paged_attention`
  - `cargo test -p fandhe-ai --release --test paged_attention_facade -- --ignored --nocapture metal_paged_attention`
- ログ: `metal/paged_kv_real_device.run{1..5}.log`・`metal/paged_attention_facade.run{1..5}.log`（末尾 `rc=0`）・`metal/env_info.txt`・`metal/build.log`（判定外）・`metal/rev-stamp.txt`
- 数値: `metal/paged_parity_median.tsv`（backend ∈ {`metal`, `metal-tracked`} の case/step 136 行。5 回の中央値。5 回で全行が完全一致のため中央値は各回の値と同じ）

| 層 | テスト | 結果（5 回） |
|---|---|---|
| backend | `metal_paged_attention_matrix_matches_cpu_reference` | 5/5 pass |
| backend | `metal_paged_attention_tracked_matrix_matches_cpu_reference` | 5/5 pass |
| backend | `metal_paged_attention_tolerance_constants_pinned` | 5/5 pass |
| facade | `metal_paged_attention_forward_matches_cpu_tape` | 5/5 pass |

- 件数: backend は毎回 `running 3 tests`・`3 passed; 0 failed`、facade は毎回 `running 1 test`・`1 passed; 0 failed`
- `PAGED_PARITY` の `fail` 合計は 0（136 行すべて `fail=0`）
- tolerance 定数・判定式・baseline・カーネル・テストコードは変更していない
- 古い `*_on_metal` の 3 件は #3113 の対象外のため測っていない
