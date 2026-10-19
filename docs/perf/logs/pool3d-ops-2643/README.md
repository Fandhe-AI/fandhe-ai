# 3D プーリング 2 種（#2643）CUDA／Metal 実機未実測の申し送り

> Metal（Mac mini・Apple M4）は 2026-10-19 に実機実測済み（#3162・pass 1/1・対象コミット `c0deb302`）。下の「Metal 実測記入欄」を参照。


`docs/autodiff-pool3d-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`fandhe_ai_autodiff::pool3d_ops`（`max_pool3d`／`avg_pool3d`。#2850 以降は同テストが facade 公開の `Var::max_pool3d`／`avg_pool3d` 経由で同じ経路を測る）の
`crates/facade/tests/pool3d_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする 2 テスト
（`cuda_pool3d_ops_match_cpu_reference`・`metal_pool3d_ops_match_cpu_reference`）は
`#[ignore]` のまま未実測である。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test pool3d_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test pool3d_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_matches_naive_reference`）で既に検証済み（green）。

## 期待結果

- CUDA／Metal は `BackendOps::pool3d_max`／`pool3d_avg` を持たず既定の `Unsupported` を返すため、実機テストは
  常に共有ホストカーネル（`fandhe_ai_tensor_core::pool3d`）へフォールバックする経路になる想定。よって本テストは
  **GPU カーネルの parity ではなく、フォールバック経路が CPU tape と同じ結果になることの確認**であり、
  Max は値 bit 一致・索引完全一致、Avg と各 backward は REQ-2 統一複合判定を満たす見込み。
- 将来 GPU 専用カーネルを実装する場合、Max はタイ先勝ち・NaN 伝播規則（最初の NaN の索引）を維持するか、
  Avg は `f64` アキュムレータ契約（Metal は `soft_f64`）の適用要否を、別イシューの決定記録で判断する。

この前提が崩れる場合は本 README の「期待結果」を更新し、REQ-2 判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 2026-10-19 | Mac mini（Apple M4） | `pool3d_ops_backend_parity` の `metal_` を `RULE.txt` のコマンドで実行 | pass 1/1 | `c0deb302`・`dirty=0`・ログは `metal/`（事前登録規則 `metal/RULE.txt`） |

## Metal 実測記入欄（#3162）

- 判定規則・実行コマンド・期待件数（1 件）は事前登録規則 `metal/RULE.txt` を正とする。実機実行より前に固定済みで、事後に変更していない（`RULE.txt` は無変更）。
- 実行日（UTC）: 2026-10-19。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0。詳細は `metal/env_info.txt` の実行前後 2 節。他の cargo／rustc 不在を実行前後に確認）。
- 実行コマンド（`RULE.txt` どおり 1 回だけ。再実行による選別なし）:

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test pool3d_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

- 対象コミット: `c0deb302dfbdfb8361075eea9f4270a5b45dd5b1`（`.rev-stamp` は 2 行目 `dirty=0`。`metal/rev-stamp.txt` に写し、ノード側で各実行の前後に読み出して一致を確認した）。
  main HEAD ではなく `c0deb302` を測ったのは、`RULE.txt` が「`crates/` 配下が対象コミット（`RULE.txt` 追加コミットの親 `75afa05`）と同一内容」を有効条件にしているため。
  `git diff 75afa05 c0deb302 -- crates/` は空で条件を満たす。その後の main では `crates/` が変わっているので、現行 main での再確認は後続候補（未起票）。
  展開ツリーは `git archive c0deb302` で作り、`crates/` のファイル指紋（`shasum -a 256` の集約値）をローカルとノードで突合して一致を確認した。
- 判定外のウォームアップとして 6 テストバイナリを `--release` で `--no-run` ビルドした（ログ: `metal/build.log`。`exit=0`）。実行時間は参考値で、判定には使わない。

| ファイル | 関数 | 結果 |
|---|---|---|
| `crates/facade/tests/pool3d_ops_backend_parity.rs` | `metal_pool3d_ops_match_cpu_reference` | pass |

- 合計: `running 1 tests` → `1 passed; 0 failed`（exit 0。期待件数 1 と一致）。ログ: `metal/pool3d_ops_backend_parity.log`。
- tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・判定・入力・テストコードは変更していない。
- 本実測は pass／fail の判定のみ。ホストフォールバック経路（`Unsupported`）を実際に通ったかの確認はログから断定できないため、経路確認は追従課題とする。
