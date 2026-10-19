# ConvTranspose3d・MaxUnpool（#2644）CUDA／Metal 実機未実測の申し送り

> Metal（Mac mini・Apple M4）は 2026-10-19 に実機実測済み（#3162・pass 1/1・対象コミット `c0deb302`）。下の「Metal 実測記入欄」を参照。


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
| 未実測 | DGX Spark GB10 | 上記 CUDA | 未実測 | |
| 2026-10-19 | Mac mini（Apple M4） | `conv_transpose3d_max_unpool_backend_parity` の `metal_` を `RULE.txt` のコマンドで実行 | pass 1/1 | `c0deb302`・`dirty=0`・ログは `metal/`（事前登録規則 `metal/RULE.txt`） |

## Metal 実測記入欄（#3162）

- 判定規則・実行コマンド・期待件数（1 件）は事前登録規則 `metal/RULE.txt` を正とする。実機実行より前に固定済みで、事後に変更していない（`RULE.txt` は無変更）。
- 実行日（UTC）: 2026-10-19。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0。詳細は `metal/env_info.txt` の実行前後 2 節。他の cargo／rustc 不在を実行前後に確認）。
- 実行コマンド（`RULE.txt` どおり 1 回だけ。再実行による選別なし）:

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test conv_transpose3d_max_unpool_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

- 対象コミット: `c0deb302dfbdfb8361075eea9f4270a5b45dd5b1`（`.rev-stamp` は 2 行目 `dirty=0`。`metal/rev-stamp.txt` に写し、ノード側で各実行の前後に読み出して一致を確認した）。
  main HEAD ではなく `c0deb302` を測ったのは、`RULE.txt` が「`crates/` 配下が対象コミット（`RULE.txt` 追加コミットの親 `75afa05`）と同一内容」を有効条件にしているため。
  `git diff 75afa05 c0deb302 -- crates/` は空で条件を満たす。その後の main では `crates/` が変わっているので、現行 main での再確認は後続候補（未起票）。
  展開ツリーは `git archive c0deb302` で作り、`crates/` のファイル指紋（`shasum -a 256` の集約値）をローカルとノードで突合して一致を確認した。
- 判定外のウォームアップとして 6 テストバイナリを `--release` で `--no-run` ビルドした（ログ: `metal/build.log`。`exit=0`）。実行時間は参考値で、判定には使わない。

| ファイル | 関数 | 結果 |
|---|---|---|
| `crates/facade/tests/conv_transpose3d_max_unpool_backend_parity.rs` | `metal_conv_transpose3d_max_unpool_match_cpu_reference` | pass |

- 合計: `running 1 tests` → `1 passed; 0 failed`（exit 0。期待件数 1 と一致）。ログ: `metal/conv_transpose3d_max_unpool_backend_parity.log`。
- tolerance（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・判定・入力・テストコードは変更していない。
- 本実測は pass／fail の判定のみ。ホストフォールバック経路（`Unsupported`）を実際に通ったかの確認はログから断定できないため、経路確認は追従課題とする。
