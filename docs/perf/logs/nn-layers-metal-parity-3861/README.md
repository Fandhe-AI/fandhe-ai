# Phase 6 の NN 層の Metal 実機 parity（#3861）

対象は Phase 6 の NN 層（upsample の倍率・Hardshrink／Softshrink・Bilinear・BatchNorm3d・CosineSimilarity／PairwiseDistance・
ChannelShuffle・3 次元適応プーリング・LPPool・FractionalMaxPool。実装 #3814・#3817〜#3827）の forward・backward で、
各実装 issue が足した `#[ignore]` の Metal テスト 19 本を CPU 参照と突き合わせた。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。
`docs/real-hardware-verification-env.md` §7）。GB10 版は #3860。雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T14:11Z。実機実行より前にコミット 6c29359cc で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../create-graph-c-metal-parity-3858/run_metal_parity.sh`（#3858）の複製に、GB10 版 #3860 と同じ `TRAIN_PARITY` の判定分岐を足した
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 19 本中 19 本 pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。

**要対応（所有者へ報告。GB10 の #3860 と同じ）: 雛形 `crates/facade/tests/train_gpu_parity.rs` の nn 族は登録 0 件。**
`train_parity_nn_metal_matches_cpu` は `undetermined: 族 nn の登録が 0 件`（`crates/facade/tests/common/train_gpu_parity.rs:193`）で panic し、
exit 101・判定 undetermined だった。構造上 pass にならないため、RULE.txt で事前に宣言したとおり各実装 issue の `#[ignore]` Metal テストを
測定対象にし、雛形の nn 族は完了判定に数えていない。

- 実行コミット（`rev-stamp.txt`）: `6c29359cc6d798a7f1b73ee396ff3e4e30c72f2c`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`1d906cac…2159`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3861 2026-11-04T14:11:33Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・各テスト前とも無し。メモリの空き率は実行前後とも 72%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（44 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- デバイス経路: 各テストは CUDA 版と同じヘルパーに `Device::Metal` を渡し、`tape_for(...).expect(...)` または層経路で受ける。Metal が使えなければ
  panic するため、CPU への無言のフォールバックで pass することは無い（早期 return・skip の分岐が無いことを対象ファイルで確かめた）。
  各テストの route（デバイスカーネルかホスト実行か）は記録していない。pass はテストの経路で CPU 参照と判定内で一致したことだけを示す。
- Bit 判定のテスト（upsample nearest・shrink・類似度・ChannelShuffle・適応プーリング 3D の ops・FractionalMaxPool の ops）は Metal でも
  CPU 参照と bit 一致した（判定は各テストのコードのまま）。
- 事前ビルド（15 ターゲットの `--no-run`）は 1 本目が 9.85 秒、残りは各 0.29〜0.55 秒（`build.log`）。各テストは 0.12〜0.37 秒で終わった。

## BatchNorm3d の統計の契約（#3819・#3820）

イシュー本文の「scale/ssq 方式の契約で確かめる」は、`.claude/rules/coding-rust.md` の正規化統計の契約（統計を f64 相当で蓄積し、最後に
1 回だけ f32 へ落とす。GPU・CPU は bit 一致を主張せず REQ-2 統一複合判定）として扱い、RULE.txt の `bn_contract` に事前登録した。

- 実装形は scale/ssq ではなく **soft-f64**（IEEE 754 binary64 の 64bit 整数ソフトウェアエミュレーション）。Metal の `batch_norm_train_f32`
  （`crates/backend-metal/src/shaders/batch_norm.metal:18-41`）は、LayerNorm の PR #1671 で scale/ssq が REQ-2 を満たせない反例が出たため、
  実装時点から scale/ssq を採らず soft-f64 にしている。f64 相当という契約はより強く満たす。
- rank 5 入力は `batch_norm_layout` で `(n, c, D*H*W)` へ畳まれ、`BackendOps::batch_norm_train` の Metal 実装
  （`crates/backend-metal/src/ops.rs:5803-5840`。サイズによるホスト経路への切替なし）から同じカーネルを通る。
- 結果: `metal_batch_norm_train_forward_rank5_matches_cpu`・`metal_batch_norm3d_matches_cpu` ともに REQ-2（`assert_parity`）で pass。
- 範囲外: テスト入力は小さな有限値（rank 5 テストは `[2,3,2,2,2]`・チャネルあたり M=16）で、overflow 耐性（`2e20` 級の入力）は確かめていない。
  確かめるなら新規テストで、所有者の承認が要る。

## 結果表

| # | 実装 issue | テスト（`--test` ターゲット／`--exact`） | 判定 | 結果 |
|---|---|---|---|---|
| 1 | #3814 | `compat_sequential_spatial_layers_backend_parity`／`metal_upsample_scale_factor_nearest_bit_exact` | Bit | pass |
| 2 | #3814 | `compat_sequential_spatial_layers_backend_parity`／`metal_upsample_scale_factor_bilinear_matches_cpu` | Req2 | pass |
| 3 | #3817 | `shrink_ops_backend_parity`／`metal_bit_exact_forward_matches_cpu_reference` | Bit | pass |
| 4 | #3817 | `shrink_ops_backend_parity`／`metal_bit_exact_backward_matches_cpu_reference` | Bit | pass |
| 5 | #3818 | `bilinear_ops_backend_parity`／`metal_bilinear_forward_matches_cpu_reference` | Req2 | pass |
| 6 | #3818 | `bilinear_ops_backend_parity`／`metal_bilinear_backward_matches_cpu_reference` | Req2 | pass |
| 7 | #3819 | `batch_norm_backend_parity`／`metal_batch_norm_train_forward_rank5_matches_cpu` | Req2 | pass |
| 8 | #3820 | `compat_sequential_layers_backend_parity`／`metal_batch_norm3d_matches_cpu` | Req2 | pass |
| 9 | #3821 | `similarity_ops_backend_parity`／`metal_bit_exact_forward_matches_cpu_reference` | Bit | pass |
| 10 | #3821 | `similarity_ops_backend_parity`／`metal_bit_exact_backward_matches_cpu_reference` | Bit | pass |
| 11 | #3822 | `channel_shuffle_backend_parity`／`metal_channel_shuffle_forward_matches_cpu` | Bit | pass |
| 12 | #3822 | `channel_shuffle_backend_parity`／`metal_channel_shuffle_backward_matches_cpu` | Bit | pass |
| 13 | #3822 | `compat_sequential_channel_shuffle_backend_parity`／`metal_channel_shuffle_layers_bit_exact` | Bit | pass |
| 14 | #3823 | `adaptive_pool3d_ops_backend_parity`／`metal_adaptive_pool3d_ops_match_cpu_reference` | Bit（索引一致を含む） | pass |
| 15 | #3824 | `compat_sequential_adaptive_pool3d_backend_parity`／`metal_compat_sequential_adaptive_pool3d_matches_cpu` | Req2 | pass |
| 16 | #3825 | `lp_pool_ops_backend_parity`／`metal_lp_pool_ops_match_cpu_reference` | Req2 | pass |
| 17 | #3825 | `compat_sequential_lp_pool_backend_parity`／`metal_lp_pool_layers_match_cpu` | Req2 | pass |
| 18 | #3826 | `fractional_max_pool_ops_backend_parity`／`metal_fractional_max_pool_ops_match_cpu_reference` | Bit（索引一致を含む） | pass |
| 19 | #3827 | `compat_sequential_adaptive_pool3d`／`fractional_max_pool::backend_parity::metal_compat_sequential_fractional_max_pool_matches_cpu` | Req2 | pass |
| 20 | #3778（雛形） | `train_gpu_parity`／`train_parity_nn_metal_matches_cpu` | - | undetermined（登録 0 件。完了判定外） |

判定列は RULE.txt の参考の写し（判定契約は各テストのコードが正）。各行の根拠は `summary.tsv`（exit code・`running 1 test` の件数・
`test result` 行）と `logs/<群>/<テスト名>.log`（末尾に `rc=`）。対象の実装 issue #3814・#3817〜#3827 は既に CLOSED。CUDA（GB10）は #3860 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（15 ターゲットの `--no-run`）のログ |
| `summary.tsv` | 20 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
