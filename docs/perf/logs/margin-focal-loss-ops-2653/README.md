# MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss（#2653）CUDA／Metal 実機未実測の申し送り

> Metal（Mac mini・Apple M4）は 2026-10-19 に実機実測済み（#3162・pass 2/2・対象コミット `c0deb302`）。下の「Metal 実測記入欄」を参照。


`docs/autodiff-margin-focal-loss-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする計 4 テストは `#[ignore]` のまま未実測である。

- CUDA: `cuda_bit_exact_forward_matches_cpu_reference`・`cuda_bit_exact_backward_matches_cpu_reference`
- Metal: 上記と対称の `metal_*` 2 件

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test margin_focal_loss_ops_backend_parity -- --ignored --nocapture cuda_

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test margin_focal_loss_ops_backend_parity -- --ignored --nocapture metal_
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_*`）で既に検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない。新規 Op 4 種は常にホスト参照実装（`eval::margin_focal_loss`）で計算されるため、
  CUDA／Metal の tape でも CPU tape と **bit 一致**する見込み（REQ-2 複合判定を使う経路はない）。
- 形状は小さく（`[3, 4]`）、Metal の split-K 経路が発動する形状は含まない。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 2026-10-19 | Mac mini（Apple M4） | `margin_focal_loss_ops_backend_parity` の `metal_` を `RULE.txt` のコマンドで実行 | pass 2/2 | `c0deb302`・`dirty=0`・ログは `metal/`（事前登録規則 `metal/RULE.txt`） |

## Metal 実測記入欄（#3162）

- 判定規則・実行コマンド・期待件数（2 件）は事前登録規則 `metal/RULE.txt` を正とする。実機実行より前に固定済みで、事後に変更していない（`RULE.txt` は無変更）。
- 実行日（UTC）: 2026-10-19。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0。詳細は `metal/env_info.txt` の実行前後 2 節。他の cargo／rustc 不在を実行前後に確認）。
- 実行コマンド（`RULE.txt` どおり 1 回だけ。再実行による選別なし）:

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test margin_focal_loss_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

- 対象コミット: `c0deb302dfbdfb8361075eea9f4270a5b45dd5b1`（`.rev-stamp` は 2 行目 `dirty=0`。`metal/rev-stamp.txt` に写し、ノード側で各実行の前後に読み出して一致を確認した）。
  main HEAD ではなく `c0deb302` を測ったのは、`RULE.txt` が「`crates/` 配下が対象コミット（`RULE.txt` 追加コミットの親 `75afa05`）と同一内容」を有効条件にしているため。
  `git diff 75afa05 c0deb302 -- crates/` は空で条件を満たす。その後の main では `crates/` が変わっているので、現行 main での再確認は後続候補（未起票）。
  展開ツリーは `git archive c0deb302` で作り、`crates/` のファイル指紋（`shasum -a 256` の集約値）をローカルとノードで突合して一致を確認した。
- 判定外のウォームアップとして 6 テストバイナリを `--release` で `--no-run` ビルドした（ログ: `metal/build.log`。`exit=0`）。実行時間は参考値で、判定には使わない。

| ファイル | 関数 | 結果 |
|---|---|---|
| `crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` | `metal_bit_exact_backward_matches_cpu_reference` | pass |
| `crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` | `metal_bit_exact_forward_matches_cpu_reference` | pass |

- 合計: `running 2 tests` → `2 passed; 0 failed`（exit 0。期待件数 2 と一致）。ログ: `metal/margin_focal_loss_ops_backend_parity.log`。
- tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・判定・入力・テストコードは変更していない。
- 本実測は pass／fail の判定のみ。ホストフォールバック経路（`Unsupported`）を実際に通ったかの確認はログから断定できないため、経路確認は追従課題とする。
