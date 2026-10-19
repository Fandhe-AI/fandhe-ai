# compat::Sequential の活性化 9 層の実機 parity 申し送り（イシュー #2679）

> Metal は Mac mini（Apple M4）で実測済み（#3164・2026-10-19）。結果は結果記入欄と末尾の「Metal 実測記入欄（#3164）」節、成果物は `metal/` を参照。

対象は `add_selu`・`add_celu`・`add_softsign`・`add_hardsigmoid`・`add_log_sigmoid`・`add_softmin`・
`add_tanhshrink`・`add_threshold`・`add_rrelu`。本エージェントの実行環境には CUDA／Metal 実機への到達手段が
ないため、実機テストは未実測のまま GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_activation_scalar_layers.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

facade `Sequential` 経由の `forward` を同じ入力の CPU `predict` と比較し、さらに `Linear → 層 → Linear` の
学習 1 ステップの全パラメータ勾配を CPU と比較する。

- REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`。相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
  forward・勾配の両方に使う。新しい tolerance 定数は作らない。
- RReLU は推論モード（`eval()`。固定傾き `(lower + upper) / 2`）で比較する。学習時の傾きは乱数のため比較しない。
- 9 層は専用の GPU カーネルを持たない（既存演算の合成、または `scalar_unary`／ホスト参照へのフォールバック）。
  この比較は「フォールバックを含む経路が CPU と同じ結果になること」の確認であり、GPU カーネル自体の parity ではない。
- 常駐経路（`predict_resident`）上での無状態層の GPU 通過は CPU でのみ検証済みで、GPU では未確認。
- Softmin 系 4 層（`add_softmin`・`add_tanhshrink`・`add_threshold`・`add_rrelu`）は `forward_host` を持たず、
  `predict` は tape 経路へフォールバックする（`docs/autodiff-softmin-threshold-ops-decision.md` §7）。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity -- --ignored cuda_
# Metal（Mac mini〈Apple M4〉）
cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_activation_scalar_layers_parity | 未実測 | |
| Mac mini（Apple M4） | metal_activation_scalar_layers_parity | pass | 2026-10-19 |

## CUDA 実測手順（#3163・GB10 未到達のため未実測）

- 判定規則は事前登録の `cuda/RULE.txt` を正とする（実測前に固定）。GB10 行は未実測のまま。
- 実行: `cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`
- 回収: ログ・`env_info.txt`・`.rev-stamp` をマスク（RULE.txt 記載）して `cuda/` に置き、上の結果記入欄の GB10 行と本節を実測値で更新する。

## Metal 実測記入欄（#3164）

- 判定規則: 事前登録の `metal/RULE.txt`（実機実行より前の独立コミットで固定。事後に変更していない）
- 実行日: 2026-10-19（UTC）。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0）。ホスト名は記録しない
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`dirty=0`。`metal/rev-stamp.txt`。`crates/` の指紋はローカルとノードで一致を確認）
- 実行コマンド: `cargo test -p fandhe-ai --test compat_sequential_activation_scalar_layers_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（debug・ノードに `timeout` が無いため perl alarm 1800 秒。各 1 回のみ実行。判定外の `--no-run` ビルドを先に実施: 9 ファイル共通で `spatial-2522/metal/build.log`）
- 実行前後の環境と `.rev-stamp`: `metal/env_info.txt`（実行前後とも他の cargo／rustc プロセスなし）
- 生ログ: `metal/compat_sequential_activation_scalar_layers_backend_parity.log`（末尾に `rc=`。ホスト名・ユーザー名・ホームパスはマスク済み）

| ファイル | 関数 | 結果 |
|---|---|---|
| `compat_sequential_activation_scalar_layers_backend_parity.rs` | `metal_activation_scalar_layers_parity` | pass（rc=0） |

- 合計: 1 件中 pass 1／fail 0／undetermined 0（`running 1 test` が期待件数と一致）。
- tolerance 定数・baseline・判定式・テストコードは不変。
- 注記: Metal の GPU カーネル経路を通ったか、Unsupported 経由のホストフォールバック経路を通ったかは本ログから断定できず未確認。
