# Phase 6 の NN 層の CUDA 実機 parity（#3860）

対象は Phase 6 の NN 層（`add_upsample` の `scale_factor`・Hardshrink／Softshrink・Bilinear・BatchNorm3d・CosineSimilarity／
PairwiseDistance・ChannelShuffle・3 次元適応プーリング／LPPool／FractionalMaxPool。実装 #3814・#3817〜#3827）の forward・backward。
機体は `<cuda-node>`（DGX Spark GB10。`docs/real-hardware-verification-env.md`）。雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T12:55Z。実機実行より前にコミット 77c2cbd58 で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3856 版 `../create-graph-c-cuda-parity-3856/run_cuda_parity.sh` の複製に、
  `train_gpu_parity` の undetermined を判定に反映する分岐を足した。GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 19 本中 19 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。

**要対応（所有者へ報告）: 雛形 `crates/facade/tests/train_gpu_parity.rs` の nn 族は登録 0 件。** `train_parity_nn_cuda_matches_cpu` は
`undetermined: 族 nn の登録が 0 件`（`crates/facade/tests/common/train_gpu_parity.rs:193`）で panic し、exit 101・判定 undetermined だった。
構造上 pass にならないため、RULE.txt で事前に宣言したとおり、各実装 issue が足した `#[ignore]` の CUDA parity テスト（各 PR が #3860 へ
申し送ったもの）を測定対象にした。雛形の nn 族は完了判定に数えていない。所有者の判断（2026-11-04）により、雛形への nn 族の追加は #4026 で行う。雛形（学習ステップ）は #4026 の登録待ちで未実測であり、#3860 は #4026 のマージ後に雛形テストを再計測してから閉じる（この記録の PR は `Refs`）。

- 対象テストの実在は対象コミットで `--ignored --list` により確かめた（Linux でも cfg で除かれない。`metal_*` だけが `cfg(target_os = "macos")`）。
  ノード上でも各テストの前に `--list` の選択件数が 1 であることを確かめた（`summary.tsv` に undetermined が無い）。
- 実行コミット（`rev-stamp.txt`）: `77c2cbd58a9daca71731c2133553d898bb9f10af`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読んで送信側と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`2238e7b7…a4da`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3860 2026-11-04T12:56:53Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無し。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 各テストは `Device::Cuda(0)` を `tape_for(...).expect(...)` または層経路へ直接渡す。CUDA が使えなければ panic するため、
  CPU への無言のフォールバックで pass することは無い（早期 return・skip の分岐が無いことを対象ファイルで確かめた）。各テストの route
  （デバイスカーネルかホスト実行か）は記録していない。pass はテストの経路で CPU 参照と判定内で一致したことだけを示す。
- 各テストは 1 秒未満で終わった（`test result` 行の `finished in`）。

## 結果表

judge は RULE.txt の参考の写しで、判定契約の正は各テストのコード。Req2 は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）、
Bit は `to_bits` の一致。

| # | 実装 issue | `--test` | テスト（`--exact`） | judge | 結果 |
|---|---|---|---|---|---|
| 1 | #3814 | `compat_sequential_spatial_layers_backend_parity` | `cuda_upsample_scale_factor_nearest_bit_exact` | Bit | pass |
| 2 | #3814 | 同上 | `cuda_upsample_scale_factor_bilinear_matches_cpu` | Req2 | pass |
| 3 | #3817 | `shrink_ops_backend_parity` | `cuda_bit_exact_forward_matches_cpu_reference` | Bit | pass |
| 4 | #3817 | 同上 | `cuda_bit_exact_backward_matches_cpu_reference` | Bit | pass |
| 5 | #3818 | `bilinear_ops_backend_parity` | `cuda_bilinear_forward_matches_cpu_reference` | Req2 | pass |
| 6 | #3818 | 同上 | `cuda_bilinear_backward_matches_cpu_reference` | Req2 | pass |
| 7 | #3819 | `batch_norm_backend_parity` | `cuda_batch_norm_train_forward_rank5_matches_cpu` | Req2 | pass |
| 8 | #3820 | `compat_sequential_layers_backend_parity` | `cuda_batch_norm3d_matches_cpu` | Req2 | pass |
| 9 | #3821 | `similarity_ops_backend_parity` | `cuda_bit_exact_forward_matches_cpu_reference` | Bit | pass |
| 10 | #3821 | 同上 | `cuda_bit_exact_backward_matches_cpu_reference` | Bit | pass |
| 11 | #3822 | `channel_shuffle_backend_parity` | `cuda_channel_shuffle_forward_matches_cpu` | Bit | pass |
| 12 | #3822 | 同上 | `cuda_channel_shuffle_backward_matches_cpu` | Bit | pass |
| 13 | #3822 | `compat_sequential_channel_shuffle_backend_parity` | `cuda_channel_shuffle_layers_bit_exact` | Bit | pass |
| 14 | #3823 | `adaptive_pool3d_ops_backend_parity` | `cuda_adaptive_pool3d_ops_match_cpu_reference` | Bit（索引一致を含む） | pass |
| 15 | #3824 | `compat_sequential_adaptive_pool3d_backend_parity` | `cuda_compat_sequential_adaptive_pool3d_matches_cpu` | Req2 | pass |
| 16 | #3825 | `lp_pool_ops_backend_parity` | `cuda_lp_pool_ops_match_cpu_reference` | Req2 | pass |
| 17 | #3825 | `compat_sequential_lp_pool_backend_parity` | `cuda_lp_pool_layers_match_cpu` | Req2 | pass |
| 18 | #3826 | `fractional_max_pool_ops_backend_parity` | `cuda_fractional_max_pool_ops_match_cpu_reference` | Bit（索引一致を含む） | pass |
| 19 | #3827 | `compat_sequential_adaptive_pool3d` | `fractional_max_pool::backend_parity::cuda_compat_sequential_fractional_max_pool_matches_cpu` | Req2 | pass |
| 20 | #3778（雛形） | `train_gpu_parity` | `train_parity_nn_cuda_matches_cpu` | — | undetermined（登録 0 件。完了判定に数えない） |

各行の根拠は `summary.tsv`（exit code・`running 1 test` の件数・`test result` 行）と `logs/<群>/<テスト名>.log`（末尾に `rc=<終了コード>`）。
追跡表の行との対応: 1〜2 は「`add_upsample` の `scale_factor` 指定」、3〜4 は C1、5〜6 は C2、7〜8 は C3、9〜10 は C5・C6、11〜13 は C7、
14〜19 は C9。対象の実装 issue は既に CLOSED。Metal（Mac mini）は #3861 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`。15 ターゲット）のログ |
| `summary.tsv` | 20 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
