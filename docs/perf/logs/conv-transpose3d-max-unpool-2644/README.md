# ConvTranspose3d・MaxUnpool（#2644）CUDA／Metal 実機未実測の申し送り

> CUDA は #3161 で GB10 実測済み（2026-10-19・コミット `b640875f`・pass 1/1）。結果は末尾「CUDA 実測記入欄（#3161）」節。
> Metal は Mac mini（Apple M4）で実測済み（#3162・2026-10-19・コミット `d79f4895`・pass 1/1）。結果は末尾の「Metal 実測記入欄（#3162）」節、成果物は `metal/`。

`docs/autodiff-conv-transpose3d-max-unpool-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::conv_transpose3d_ops`・`max_unpool_ops`（#2850 以降は同テストが facade 公開の `Var::conv_transpose3d`／`Var::max_unpool1d/2d/3d` 経由で同じ経路を測る）の
`crates/facade/tests/conv_transpose3d_max_unpool_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_conv_transpose3d_max_unpool_match_cpu_reference`・`metal_conv_transpose3d_max_unpool_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

本実装は新規 GPU カーネルを持たない。したがって本テストは **GPU カーネル新設の parity ではなく、既存フックを
実機が通る経路が CPU tape と同じ結果になることの確認**である。

- ConvTranspose3d: `gemm_batched`（forward）・`gemm_batched_fp32_strict`（VJP）は実機の GEMM が走る。
  `im2col3d`／`col2im3d` は CUDA／Metal とも既定 `Unsupported` を返す（`crates/backend-cpu/tests/
  backend_ops_dispatch.rs` の `*_im2col3d_col2im3d_are_unsupported_not_panic` で固定）ため、autodiff 側の
  ホストフォールバック（`eval::im2col3d`／`col2im3d`）へ到達する。forward・`d_input`・`d_weight`・`d_bias` は
  REQ-2 統一複合判定を満たす見込み（CUDA の TF32 opt-in を有効にした場合は `docs/spec/04-requirements.md` REQ-2 の
  Tensor Core 経路の判定方式に従う）。
- MaxUnpool: 既存の GPU `scatter`（forward。`Overwrite`）・`gather`（VJP）が実機で走る。重複索引を含む最後の書き手の
  契約が GPU カーネルでも保たれるかが確認点で、forward・勾配とも bit 一致する見込み。**GPU の `scatter` が重複索引で
  CPU と異なる勝者を返す場合は REQ-2 違反ではなく `Overwrite` の決定性契約の違反**であり、別イシューで扱う
  （tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。
- 将来 GPU 専用カーネル（`im2col3d`／`col2im3d`・unpool 専用）を実装する場合は、`col2im3d` の `f64` アキュムレータ契約
  （Metal は `soft_f64`）と `Overwrite` の最後の書き手契約の適用要否を別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-19 | DGX Spark GB10 | `cargo test -p fandhe-ai --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture --test-threads=1 cuda`（`cuda/RULE.txt`） | pass（1/1） | `.rev-stamp` `b640875f`・dirty=0。ログは `cuda/` |
| 2026-10-19 | Mac mini（Apple M4） | `cargo test -p fandhe-ai --release --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（`metal/RULE.txt`） | pass 1/1（fail 0） | `.rev-stamp` `d79f4895`（40 桁は `metal/rev-stamp.txt`）。ログは `metal/conv_transpose3d_max_unpool_backend_parity.log` |

## Metal 実測記入欄（#3162）

- 判定規則は `metal/RULE.txt`（実機実行より前の独立コミットで main HEAD `d79f4895` を対象に再登録済み。事後に変更していない）。
  所有者判断（2026-10-19）により、前回の c0deb302 での実測は採用していない。M4 Max（手元機）の値は使っていない。
- 実行日（UTC）: 2026-10-19。機体: Mac mini（Apple M4・メモリ 16GiB・macOS 27.0.1・rustc 1.99.0）。ホスト名・ユーザー名・個人パスは記録しない。
- 対象コミット: `d79f4895d1337ab5035f76f79eea7619196de874`（`metal/rev-stamp.txt`。`git rev-parse HEAD` の 40 桁 1 行）。展開ツリーは `git archive` で作成し、
  RULE 更新コミットが `crates/`・`Cargo.toml`・`Cargo.lock` を変えていないこと（`git diff --quiet d79f4895 HEAD -- crates Cargo.toml Cargo.lock`）を確認済み。
  ノード側の `.rev-stamp` は実行前後とも一致した。
- 実行コマンド: `cargo test -p fandhe-ai --release --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（`perl -e 'alarm shift; exec @ARGV' 1800` で包む。各ファイル 1 回のみで選別・再実行はしていない）。
  判定外の `--no-run` ビルドは 1 回だけ行った（6 機能共通。`../pool3d-ops-2643/metal/build.log`）。
- 成果物: `metal/env_info.txt`（実行前後）・`metal/conv_transpose3d_max_unpool_backend_parity.log`（末尾に `rc=`）・`metal/rev-stamp.txt`・`metal/RULE.txt`。

| ファイル | 関数 | 結果（rc） |
|---|---|---|
| `conv_transpose3d_max_unpool_backend_parity.rs` | `metal_conv_transpose3d_max_unpool_match_cpu_reference` | pass（rc=0） |

- 合計: pass 1 / fail 0 / undetermined 0（「running 1 test」が期待件数 1 と一致）。
- tolerance 定数・baseline・判定式・テストコードは不変。
- 注記: 実機ノードは他用途と共有で、実行前後の load average が 6〜9 だった（判定テストのため時間計測には使わない）。
  GPU カーネル経路とホストフォールバック経路のどちらを通ったかはログから断定できないため未確認である。

## CUDA 実測記入欄（#3161）

- 判定規則・fail の扱い・実行コマンド・期待件数（1 件）は事前登録規則 `cuda/RULE.txt` を正とする（実測前に固定済み。事後に変更していない）。
- 実行コマンド: 上記 `RULE.txt` のとおり独立プロセスで 1 回（並列度 1・`--release` なし・`timeout 1800`）。再実行・選別はしていない。
- 対象コミット: `b640875f`（`cuda/rev-stamp.txt`: 1 行目 sha・2 行目 `dirty=0`。実行前・各ファイル実行後・実行後の 8 回の読み出しがすべて一致）。
- 環境: NVIDIA GB10（aarch64・ドライバ 580.173.02）・CUDA 13.0（nvcc 13.0.88）・rustc 1.97.0。詳細は `cuda/env_info.txt`（実行前後）。
- 備考: 常駐サービスの GPU プロセス 3 件（実行前後で同一 pid・`utilization.gpu` 0%）以外の GPU 負荷なし。作業ツリーは初回転送でビルドキャッシュが冷えていたため、判定外のウォームアップ（`--no-run`、`cuda/build.log`）で 6 テストファイル分をビルドしてからテストを実行した（テスト自体は 1 回ずつ）。

| ファイル | 関数 | 結果 | 備考 |
|---|---|---|---|
| `crates/facade/tests/conv_transpose3d_max_unpool_backend_parity.rs` | `cuda_conv_transpose3d_max_unpool_match_cpu_reference` | pass | |

- 合計 1/1 pass（fail 0）。実行時間 0.60s と短く、経路確認（GPU フック経由かフォールバックか）は追従課題。フォールバック経路を通ったとは断定しない。
- tolerance 定数・baseline・判定規則・テストコードは不変。
