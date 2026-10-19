# LocalResponseNorm・weight_norm・spectral_norm（#2646）CUDA／Metal 実機未実測の申し送り

> Metal（Mac mini・Apple M4）は 2026-10-19 に実機実測済み（#3162・pass 1/1・対象コミット `c0deb302`）。下の「Metal 実測記入欄」を参照。


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
| 2026-10-19 | Mac mini（Apple M4） | `lrn_weight_reparam_backend_parity` の `metal_` を `RULE.txt` のコマンドで実行 | pass 1/1 | `c0deb302`・`dirty=0`・ログは `metal/`（事前登録規則 `metal/RULE.txt`） |

#2851 以降、本テストは `Var` のメソッド経由（`Var::local_response_norm`／`Var::weight_norm`／`Var::spectral_norm`）で同じ経路を測る（テスト名・`#[ignore]`・判定は不変。上表の記入欄は未実測のまま）。

## Metal 実測記入欄（#3162）

- 判定規則・実行コマンド・期待件数（1 件）は事前登録規則 `metal/RULE.txt` を正とする。実機実行より前に固定済みで、事後に変更していない（`RULE.txt` は無変更）。
- 実行日（UTC）: 2026-10-19。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0。詳細は `metal/env_info.txt` の実行前後 2 節。他の cargo／rustc 不在を実行前後に確認）。
- 実行コマンド（`RULE.txt` どおり 1 回だけ。再実行による選別なし）:

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test lrn_weight_reparam_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

- 対象コミット: `c0deb302dfbdfb8361075eea9f4270a5b45dd5b1`（`.rev-stamp` は 2 行目 `dirty=0`。`metal/rev-stamp.txt` に写し、ノード側で各実行の前後に読み出して一致を確認した）。
  main HEAD ではなく `c0deb302` を測ったのは、`RULE.txt` が「`crates/` 配下が対象コミット（`RULE.txt` 追加コミットの親 `75afa05`）と同一内容」を有効条件にしているため。
  `git diff 75afa05 c0deb302 -- crates/` は空で条件を満たす。その後の main では `crates/` が変わっているので、現行 main での再確認は後続候補（未起票）。
  展開ツリーは `git archive c0deb302` で作り、`crates/` のファイル指紋（`shasum -a 256` の集約値）をローカルとノードで突合して一致を確認した。
- 判定外のウォームアップとして 6 テストバイナリを `--release` で `--no-run` ビルドした（ログ: `metal/build.log`。`exit=0`）。実行時間は参考値で、判定には使わない。

| ファイル | 関数 | 結果 |
|---|---|---|
| `crates/facade/tests/lrn_weight_reparam_backend_parity.rs` | `metal_lrn_weight_reparam_match_cpu_reference` | pass |

- 合計: `running 1 tests` → `1 passed; 0 failed`（exit 0。期待件数 1 と一致）。ログ: `metal/lrn_weight_reparam_backend_parity.log`。
- tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・判定・入力・テストコードは変更していない。
- 本実測は pass／fail の判定のみ。ホストフォールバック経路（`Unsupported`）を実際に通ったかの確認はログから断定できないため、経路確認は追従課題とする。
