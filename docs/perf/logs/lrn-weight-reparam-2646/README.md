# LocalResponseNorm・weight_norm・spectral_norm（#2646）CUDA／Metal 実機未実測の申し送り

> Metal は Mac mini（Apple M4）で実測済み（#3162・2026-10-19・コミット `d79f4895`・pass 1/1）。結果は末尾の「Metal 実測記入欄（#3162）」節、成果物は `metal/`。

`docs/autodiff-lrn-weight-reparam-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::{lrn_ops, weight_reparam_ops}` の `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` のうち
CUDA（`Device::Cuda(0)`）・Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_lrn_weight_reparam_match_cpu_reference`・`metal_lrn_weight_reparam_match_cpu_reference`）は `#[ignore]` の
まま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test lrn_weight_reparam_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test lrn_weight_reparam_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green。
NaiveOps tape のホストフォールバックと bit 一致）。

## 期待結果

本実装は新規 GPU カーネルを持たず、CUDA／Metal は新規 3 フック（`lrn_forward`・`weight_norm_forward`・
`spectral_norm_forward`）を override しない（既定 `Unsupported`）。したがって本テストは **新規 GPU カーネルの parity では
なく、既定 `Unsupported` → ホストフォールバック経路が CPU tape と bit 一致することの確認**である。

- forward は共有ホストカーネルを通り、CPU tape（`CpuBackendOps` の override も同じ共有カーネル）と bit 一致する見込み。
  VJP・`SpectralNormState` の power iteration は両 tape ともホスト側のみで bit 一致する見込み。
- 周辺の演算（`mul`・`sum`・backward の勾配集約）は各バックエンドの既存カーネルを通るため、この部分が外れた場合は本機能
  ではなく当該バックエンドの既存契約違反として扱う。
- **外れた場合は REQ-2 違反として tolerance を緩めず**、原因を特定して別イシューで扱う（`.claude/rules/coding-rust.md`）。
- 将来 GPU 専用の LRN／weight_norm／spectral_norm カーネルを実装する場合は、正規化統計の `f64` 相当契約（Metal は
  `double` 非対応のため補償和または `soft_f64` 系）の適用要否を別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 2026-10-19 | Mac mini（Apple M4） | `cargo test -p fandhe-ai --release --test lrn_weight_reparam_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（`metal/RULE.txt`） | pass 1/1（fail 0） | `.rev-stamp` `d79f4895`（40 桁は `metal/rev-stamp.txt`）。ログは `metal/lrn_weight_reparam_backend_parity.log` |

#2851 以降、本テストは `Var` のメソッド経由（`Var::local_response_norm`／`Var::weight_norm`／`Var::spectral_norm`）で同じ経路を測る（テスト名・`#[ignore]`・判定は不変。上表の記入欄は未実測のまま）。

## Metal 実測記入欄（#3162）

- 判定規則は `metal/RULE.txt`（実機実行より前の独立コミットで main HEAD `d79f4895` を対象に再登録済み。事後に変更していない）。
  所有者判断（2026-10-19）により、前回の c0deb302 での実測は採用していない。M4 Max（手元機）の値は使っていない。
- 実行日（UTC）: 2026-10-19。機体: Mac mini（Apple M4・メモリ 16GiB・macOS 27.0.1・rustc 1.99.0）。ホスト名・ユーザー名・個人パスは記録しない。
- 対象コミット: `d79f4895d1337ab5035f76f79eea7619196de874`（`metal/rev-stamp.txt`。`git rev-parse HEAD` の 40 桁 1 行）。展開ツリーは `git archive` で作成し、
  RULE 更新コミットが `crates/`・`Cargo.toml`・`Cargo.lock` を変えていないこと（`git diff --quiet d79f4895 HEAD -- crates Cargo.toml Cargo.lock`）を確認済み。
  ノード側の `.rev-stamp` は実行前後とも一致した。
- 実行コマンド: `cargo test -p fandhe-ai --release --test lrn_weight_reparam_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（`perl -e 'alarm shift; exec @ARGV' 1800` で包む。各ファイル 1 回のみで選別・再実行はしていない）。
  判定外の `--no-run` ビルドは 1 回だけ行った（6 機能共通。`../pool3d-ops-2643/metal/build.log`）。
- 成果物: `metal/env_info.txt`（実行前後）・`metal/lrn_weight_reparam_backend_parity.log`（末尾に `rc=`）・`metal/rev-stamp.txt`・`metal/RULE.txt`。

| ファイル | 関数 | 結果（rc） |
|---|---|---|
| `lrn_weight_reparam_backend_parity.rs` | `metal_lrn_weight_reparam_match_cpu_reference` | pass（rc=0） |

- 合計: pass 1 / fail 0 / undetermined 0（「running 1 test」が期待件数 1 と一致）。
- tolerance 定数・baseline・判定式・テストコードは不変。
- 注記: 実機ノードは他用途と共有で、実行前後の load average が 6〜9 だった（判定テストのため時間計測には使わない）。
  GPU カーネル経路とホストフォールバック経路のどちらを通ったかはログから断定できないため未確認である。
