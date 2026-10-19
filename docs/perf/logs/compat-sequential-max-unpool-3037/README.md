# FunctionalBuilder の MaxPool 索引・MaxUnpool 層経路 実機 parity 申し送り（イシュー #3037）

> CUDA は #3171 で GB10 実測済み（2026-10-19・コミット b640875f・pass 2/2）。結果は末尾「CUDA 実測記入欄（#3171）」節。

本イシューは MaxPool 索引・MaxUnpool ノードの保存対応（Functional manifest の op 追加）と、層（Functional
グラフ）経路の `#[ignore]` parity テストの追加まで。実機（CUDA／Metal）の実測と結果記入は Phase 8 の
#3171・#3172 が担う。CPU 側の正しさは `crates/facade/tests/compat_functional_max_unpool.rs`・
`compat_functional_max_unpool_model_io.rs` で検証済み。tolerance・baseline は変更しない。
演算単体の parity は `conv_transpose3d_max_unpool_backend_parity.rs` が担い、本テストは層（Functional グラフ）経由。

テストファイル名: Issue 本文は `compat_sequential_<名前>_backend_parity.rs` としているが、MaxUnpool は
`Sequential` に載せない確定形（決定記録 `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §6 案 A・§12）の
ため実体は `crates/facade/tests/compat_functional_max_unpool_backend_parity.rs` とした。ログ置き場の名前だけ
Issue 指定どおり維持している。

## 判定契約

- SegNet 型 `FunctionalModel`（conv ブロック → `max_pool*_with_indices` → conv ブロック → `max_unpool*`）の
  2D（入力 NCHW `[2, 2, 6, 6]`）・3D（入力 NCDHW `[1, 2, 4, 4, 4]`）について、forward 出力と `sum` 損失に対する
  入力勾配を CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）で比較する。
- パラメータ勾配は `FunctionalModel::bind` が非公開で公開面から取れないため比較対象に含めない。
- 期待結果の注記: GPU カーネルの有無は演算単体の記録に従う。カーネルが無い場合は既定 `Unsupported` からの
  ホストフォールバック経路の整合の確認になる。fail が出ても tolerance は緩めない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_functional_max_unpool_backend_parity -- --ignored cuda_
# Metal（Mac mini〔Apple M4〕）
cargo test -p fandhe-ai --test compat_functional_max_unpool_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_functional_max_unpool_2d_forward_backward_matches_cpu | pass 1/1（`cuda/compat_functional_max_unpool_backend_parity.log`・`cuda/RULE.txt`・`.rev-stamp` dirty=0） | 2026-10-19 |
| GB10 | cuda_functional_max_unpool_3d_forward_backward_matches_cpu | pass 1/1（`cuda/compat_functional_max_unpool_backend_parity.log`・`cuda/RULE.txt`・`.rev-stamp` dirty=0） | 2026-10-19 |
| Mac mini（Apple M4） | metal_functional_max_unpool_2d_forward_backward_matches_cpu | 未実測（#3172） | |
| Mac mini（Apple M4） | metal_functional_max_unpool_3d_forward_backward_matches_cpu | 未実測（#3172） | |

## CUDA 実測記入欄（#3171）

- 事前登録規則: `cuda/RULE.txt`（コミット `5fd2294e` で実測前に固定。実測は `b640875f` 時点の origin/main を `git archive` で書き出して実行）。
- 実行回数: 独立プロセスで 1 回（再実行・選別なし）。実行日 2026-10-19（UTC 05:3x。`cuda/env_info.txt` の `date_utc`）。
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`cuda/rev-stamp.txt` は `dirty=0`。main の祖先）。
- 環境: GPU は NVIDIA GB10（aarch64・sm_121）、ドライバ 580.173.02、CUDA／nvcc 13.0.88、rustc 1.97.0。詳細は `cuda/env_info.txt`（各実行の前後を収録）。
- GPU 占有の前後比較: 実行前後とも GPU 使用率 0%、計算プロセスは常駐サービス 3 件（`<resident-process>`。メモリ使用量は前後で同一）のみで、実行中に新たな GPU プロセスは現れなかった。常駐サービスは止めていない。
- 実行コマンド: `cuda/RULE.txt` のとおり（`-p fandhe-ai --test compat_functional_max_unpool_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`）。ログ末尾に `rc=0`。

| ファイル | テスト | 結果 | ログ |
|---|---|---|---|
| `compat_functional_max_unpool_backend_parity.rs` | `cuda_functional_max_unpool_2d_forward_backward_matches_cpu` | pass | `cuda/compat_functional_max_unpool_backend_parity.log` |
| `compat_functional_max_unpool_backend_parity.rs` | `cuda_functional_max_unpool_3d_forward_backward_matches_cpu` | pass | `cuda/compat_functional_max_unpool_backend_parity.log` |

- ログの `running 2 tests` と `test result: ok. 2 passed; 0 failed` の件数は事前登録の期待件数 2 と一致。合計 pass 2/2。
- ウォームアップ（`cargo test --no-run` のビルド。`cuda/build.log`）は判定外で、測定中にビルドは並走させていない（並列度 1）。
- tolerance 定数・判定式・baseline・テストコード・入力集合は変更していない（事前登録規則 `cuda/RULE.txt` のとおり）。再実行・選別はしていない。
- 実行時間は 1 秒前後と短く、ホストフォールバックの可能性がある。経路（GPU フック経由かフォールバックか）は本測定からは断定しない（追従課題）。
