# paged attention（K-3）の CPU 参照一致と CUDA／Metal 実機 parity（#3113）未実測の申し送り

> CUDA は #3171 で GB10 実測済み（2026-10-19・コミット b640875f・backend 2 件×5 回・facade 1 件×5 回すべて pass）。結果は末尾「CUDA 実測記入欄（#3171）」節。

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
| 2026-10-19 | DGX Spark GB10 | 上記 cuda コマンド（backend・facade。`--test-threads=1` 付き） | 5 回すべて pass（backend 2/2・facade 1/1 が各回。fail 0） | `.rev-stamp` dirty=0・ログ `cuda/*.run{1..5}.log`・中央値表 `cuda/paged_parity_median.tsv` |
| 未実測（#3172） | Apple Silicon（Mac mini M4） | 上記 metal コマンド（backend・facade） | 未実測 | |

## CUDA 実測記入欄（#3171）

- 事前登録規則: 既存の `RULE.txt`（2026-10-16 固定。本 PR では変更していない）。実行は同規則どおり独立プロセスで **5 回**（イシュー #3171 本文の既定の 1 回より厳しい側）。
- 実行日 2026-10-19。対象コミット `b640875f86399515b2dab4dfc38089caaa5e5c08`（`cuda/rev-stamp.txt` は `dirty=0`。main の祖先）。
- 環境: GPU は NVIDIA GB10（aarch64・sm_121）、ドライバ 580.173.02、CUDA／nvcc 13.0.88、rustc 1.97.0。詳細は `cuda/env_info.txt`（各実行の前後を収録）。
- GPU 占有の前後比較: 各実行の前後の GPU 使用率は 20 スナップショット中 16 が 0%、4 が 4%（`paged_attention_facade.run3` after・`run5` after、`paged_kv_real_device.run4` before・after。常駐サービス由来の揺らぎで、テストの判定には使わない。計算プロセス一覧は全スナップショットで同一）。計算プロセスは常駐サービス 3 件（`<resident-process>`。メモリ使用量は前後で同一）のみで、実行中に新たな GPU プロセスは現れなかった。常駐サービスは止めていない。
- 実行コマンド: `-p fandhe-ai-backend-cuda --release --test paged_kv_real_device -- --ignored --nocapture --test-threads=1 cuda_paged_attention`、`-p fandhe-ai --release --test paged_attention_facade -- --ignored --nocapture --test-threads=1 cuda_paged_attention`。各ログ末尾に `rc=0`。
- 対象外: `paged_kv_real_device.rs` の古い `#[ignore]` 2 件（`paged_kv_matches_cpu_reference_on_real_device`・`single_token_attention_returns_v_row_on_real_device`）は名前が `cuda_` で始まらず #3113 の範囲外のため、フィルタ `cuda_paged_attention` で走らない（`3 filtered out`）。

| 層 | テスト | 5 回の結果 | ログ |
|---|---|---|---|
| backend | `cuda_paged_attention_matrix_matches_cpu_reference`・`cuda_paged_attention_tolerance_constants_pinned` | 毎回 `running 2 tests` → `2 passed; 0 failed`（5/5 回 pass） | `cuda/paged_kv_real_device.run{1..5}.log` |
| facade | `cuda_paged_attention_forward_matches_cpu_tape` | 毎回 `running 1 test` → `1 passed; 0 failed`（5/5 回 pass） | `cuda/paged_attention_facade.run{1..5}.log` |

### PAGED_PARITY（backend 層）の 5 回中央値

- `PAGED_PARITY` 行は 1 回あたり 68 行（case×step）。5 回で全 68 行の `total`・`fail`・`max_abs`・`max_rel`・`mean_abs` が完全一致した（不一致 0）ため、中央値は各回の値と同じ。fail は全行 0、`judgement=strict`（`BASELINES` は 0 件のまま。追加していない）。
- 全 68 行の中央値表は `cuda/paged_parity_median.tsv`（列: case, step, total, fail, max_abs, max_rel, mean_abs, judgement, runs_identical）。
- facade 層は `PAGED_PARITY` 行を出さない（厳密判定のみ）。
- ウォームアップ（`--release` ビルド。`cuda/build.log`）は判定外。測定中にビルドは並走させていない（並列度 1）。
- tolerance 定数・判定式・`BASELINES`・テストコードは変更していない。再実行・選別はしていない。
