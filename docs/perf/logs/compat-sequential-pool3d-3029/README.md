# compat::Sequential の Pool3d 層経路 実機 parity 申し送り（イシュー #3029）

> CUDA は #3171 で GB10 実測済み（2026-10-19・コミット b640875f・pass 1/1）。結果は末尾「CUDA 実測記入欄（#3171）」節。
> Metal は #3172 で Mac mini（Apple M4）実測済み（2026-10-19・実行コミット 66537019・pass 1/1）。結果は末尾「Metal 実測記入欄（#3172）」節。

本イシューは `max_pool3d`／`avg_pool3d` の保存対応と、層経路の `#[ignore]` parity テストの追加まで。
実機（CUDA／Metal）の実測と結果記入は Phase 8 の #3171・#3172 が担う。CPU 側の正しさは
`crates/facade/tests/compat_sequential_pool3d.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_max_pool3d`→`add_avg_pool3d`）の forward 出力と、`sum` 損失に対する入力勾配
  （pool 層は無パラメータ）を CPU と REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `assert_parity`）で比較する。入力は全要素が異なる ramp（max の索引タイブレークに依存させない）。
- 期待結果の注記: pool3d の GPU カーネルは存在せず、既定 `Unsupported` からホストフォールバックする。
  したがって本テストが確認するのは GPU カーネルの parity ではなくホストフォールバック経路の整合
  （`pool3d-ops-2643/README.md` と同じ位置づけ）。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored cuda_
# Metal（Mac mini〔Apple M4〕）
cargo test -p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored metal_
```

## 結果記入欄

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_pool3d_forward_backward_matches_cpu | pass 1/1（`cuda/compat_sequential_pool3d_backend_parity.log`・`cuda/RULE.txt`・`.rev-stamp` dirty=0） | 2026-10-19 |
| Mac mini（Apple M4） | metal_pool3d_forward_backward_matches_cpu | pass 1/1（`metal/metal_pool3d_forward_backward_matches_cpu.log`・`../metal-phase2-4-parity-3172/RULE.txt`・`.rev-stamp` dirty=0） | 2026-10-19 |

## CUDA 実測記入欄（#3171）

- 事前登録規則: `cuda/RULE.txt`（コミット `5fd2294e` で実測前に固定。実測は `b640875f` 時点の origin/main を `git archive` で書き出して実行）。
- 実行回数: 独立プロセスで 1 回（再実行・選別なし）。実行日 2026-10-19（UTC 05:3x。`cuda/env_info.txt` の `date_utc`）。
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`cuda/rev-stamp.txt` は `dirty=0`。main の祖先）。
- 環境: GPU は NVIDIA GB10（aarch64・sm_121）、ドライバ 580.173.02、CUDA／nvcc 13.0.88、rustc 1.97.0。詳細は `cuda/env_info.txt`（各実行の前後を収録）。
- GPU 占有の前後比較: 実行前後とも GPU 使用率 0%、計算プロセスは常駐サービス 3 件（`<resident-process>`。メモリ使用量は前後で同一）のみで、実行中に新たな GPU プロセスは現れなかった。常駐サービスは止めていない。
- 実行コマンド: `cuda/RULE.txt` のとおり（`-p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`）。ログ末尾に `rc=0`。

| ファイル | テスト | 結果 | ログ |
|---|---|---|---|
| `compat_sequential_pool3d_backend_parity.rs` | `cuda_pool3d_forward_backward_matches_cpu` | pass | `cuda/compat_sequential_pool3d_backend_parity.log` |

- ログの `running 1 test` と `test result: ok. 1 passed; 0 failed` の件数は事前登録の期待件数 1 と一致。合計 pass 1/1。
- ウォームアップ（`cargo test --no-run` のビルド。`cuda/build.log`）は判定外で、測定中にビルドは並走させていない（並列度 1）。
- tolerance 定数・判定式・baseline・テストコード・入力集合は変更していない（事前登録規則 `cuda/RULE.txt` のとおり）。再実行・選別はしていない。
- 実行時間は 1 秒前後と短く、ホストフォールバックの可能性がある。経路（GPU フック経由かフォールバックか）は本測定からは断定しない（追従課題）。

## Metal 実測記入欄（#3172）

- 事前登録規則: `../metal-phase2-4-parity-3172/RULE.txt`（実測前にコミット `66537019` で固定。前回測定〈PR #3474〉は記録手順の欠陥で不採用となり、本測定はその再測定。判定・入力・コマンドの判定部分は同一で、結果は pass／fail にかかわらず採用）。
- 実行回数: 独立プロセスで 1 回（再実行・選別なし）。実行日 2026-10-19（UTC 18:1x。`../metal-phase2-4-parity-3172/env_info.txt` の `date_utc`）。
- 対象コミット: `5df7986446fd6e19bbfa2d73539fef95ba2e9d0e` 以降で `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` に差分が無い `66537019`（`metal/rev-stamp.txt` は `dirty=0`）。展開ツリーを rsync で Mac mini に転送して実行した。
- 環境: Mac mini（Apple M4・macOS 27.0.1・rustc 1.99.0）。詳細は `../metal-phase2-4-parity-3172/env_info.txt`（実行の前後を収録。ホスト名・ユーザー名・絶対パスはマスク済みで、ログは手で編集していない）。
- 実行コマンド: `-p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored --exact --nocapture --test-threads=1 metal_pool3d_forward_backward_matches_cpu`（`--release` なし）。各ログ末尾に `rc=0`。
- 実行ロジックとマスクの自己検査・識別子保存検査: `../metal-phase2-4-parity-3172/run_metal_parity.sh`。提出物の改変検査は `../metal-phase2-4-parity-3172/SHA256SUMS`（README の手順で検証）。

| ファイル | テスト | 結果 | ログ |
|---|---|---|---|
| `compat_sequential_pool3d_backend_parity.rs` | `metal_pool3d_forward_backward_matches_cpu` | pass | `metal/metal_pool3d_forward_backward_matches_cpu.log` |

- ログの `running 1 test` と `test result: ok. 1 passed; 0 failed` の件数は事前登録の期待件数 1 と一致。合計 pass 1/1
- ビルド（`../metal-phase2-4-parity-3172/build.log`）は判定外で、測定中にビルドは並走させていない（並列度 1。実行前に `pgrep` で他の cargo／rustc が無いことを確認）。
- tolerance 定数・判定式・baseline・テストコード・入力集合は変更していない。再実行・選別はしていない。
- 実行時間は 0.1 秒前後と短く、ホストフォールバックの可能性がある。経路（GPU フック経由かフォールバックか）は本測定からは断定しない（追従課題）。
