# compat::Sequential の PixelShuffle／PixelUnshuffle 実機 parity 申し送り（イシュー #2526）

> Metal は Mac mini（Apple M4）で実測済み（#3164・2026-10-19）。結果は結果記入欄と末尾の「Metal 実測記入欄（#3164）」節、成果物は `metal/` を参照。

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_pixel_shuffle.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_pixel_shuffle` → `add_pixel_unshuffle` → `add_pixel_shuffle`）の `forward` 出力を、
  CPU `predict` と **bit 完全一致**で比較する（算術を含まない contiguous／reshape／permute の合成のため）。
- 新規 `Op`／`BackendOps`／カーネルはない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity -- --ignored cuda_
# Metal（Mac mini〈Apple M4〉）
cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_pixel_shuffle_layers_bit_exact | 未実測 | |
| Mac mini（Apple M4） | metal_pixel_shuffle_layers_bit_exact | pass | 2026-10-19 |

## CUDA 実測手順（#3163・GB10 未到達のため未実測）

- 判定規則は事前登録の `cuda/RULE.txt` を正とする（実測前に固定）。GB10 行は未実測のまま。
- 実行: `cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`
- 回収: ログ・`env_info.txt`・`.rev-stamp` をマスク（RULE.txt 記載）して `cuda/` に置き、上の結果記入欄の GB10 行と本節を実測値で更新する。

## Metal 実測記入欄（#3164）

- 判定規則: 事前登録の `metal/RULE.txt`（実機実行より前の独立コミットで固定。事後に変更していない）
- 実行日: 2026-10-19（UTC）。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0）。ホスト名は記録しない
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`dirty=0`。`metal/rev-stamp.txt`。`crates/` の指紋はローカルとノードで一致を確認）
- 実行コマンド: `cargo test -p fandhe-ai --test compat_sequential_pixel_shuffle_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（debug・ノードに `timeout` が無いため perl alarm 1800 秒。各 1 回のみ実行。判定外の `--no-run` ビルドを先に実施: 9 ファイル共通で `spatial-2522/metal/build.log`）
- 実行前後の環境と `.rev-stamp`: `metal/env_info.txt`（実行前後とも他の cargo／rustc プロセスなし）
- 生ログ: `metal/compat_sequential_pixel_shuffle_backend_parity.log`（末尾に `rc=`。ホスト名・ユーザー名・ホームパスはマスク済み）

| ファイル | 関数 | 結果 |
|---|---|---|
| `compat_sequential_pixel_shuffle_backend_parity.rs` | `metal_pixel_shuffle_layers_bit_exact` | pass（rc=0） |

- 合計: 1 件中 pass 1／fail 0／undetermined 0（`running 1 test` が期待件数と一致）。
- tolerance 定数・baseline・判定式・テストコードは不変。
- 注記: Metal の GPU カーネル経路を通ったか、Unsupported 経由のホストフォールバック経路を通ったかは本ログから断定できず未確認。
