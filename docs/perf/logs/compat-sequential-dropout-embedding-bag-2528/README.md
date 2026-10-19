# compat::Sequential の Dropout2d／AlphaDropout／EmbeddingBag 実機 parity 申し送り（イシュー #2528）

> Metal は Mac mini（Apple M4）で実測済み（#3164・2026-10-19）。結果は結果記入欄と末尾の「Metal 実測記入欄（#3164）」節、成果物は `metal/` を参照。

本エージェントの実行環境には CUDA／Metal 実機への到達手段がないため、実機テストは未実測のまま
GB10・M4 Max セッションへ申し送る。CPU 側の正しさは Linux 実行可能な
`crates/facade/tests/compat_sequential_dropout_embedding_bag.rs` で検証済み。tolerance・baseline は変更しない。

## 判定契約

- facade `Sequential`（`add_dropout2d(0.5)` → `add_alpha_dropout(0.25)`）の `forward` 出力を、同じ `manual_seed` から実行した CPU
  `predict` と **bit 完全一致**で比較する（`Op::Dropout` の再利用。GPU 側はホストフォールバック経由。マスクはホストで生成）。
- `add_embedding_bag(6, 4, mode, Some(0), 21)`（`mode` は Sum／Mean／Max の 3 種）の `forward` 出力を CPU `predict` と **bit 完全一致**で比較する
  （既存の `embedding_bag_backend_parity.rs` は Sum／Mean を REQ-2 統一複合判定・Max を bit 完全一致としているため、実機で不一致が出た場合は
  同テストの判定方式との差を確認する。本テストで tolerance を緩めない）。
- 新規 `Op`／`BackendOps`／カーネルはない。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity -- --ignored cuda_
# Metal（Mac mini〈Apple M4〉）
cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity -- --ignored metal_
```

## 結果記入欄（未実測）

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_dropout_embedding_bag_bit_exact | 未実測 | |
| Mac mini（Apple M4） | metal_dropout_embedding_bag_bit_exact | **fail**（Unsupported・要対応。#3164） | 2026-10-19 |

## CUDA 実測手順（#3163・GB10 未到達のため未実測）

- 判定規則は事前登録の `cuda/RULE.txt` を正とする（実測前に固定）。GB10 行は未実測のまま。
- 実行: `cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`
- 回収: ログ・`env_info.txt`・`.rev-stamp` をマスク（RULE.txt 記載）して `cuda/` に置き、上の結果記入欄の GB10 行と本節を実測値で更新する。

## Metal 実測記入欄（#3164）

- 判定規則: 事前登録の `metal/RULE.txt`（実機実行より前の独立コミットで固定。事後に変更していない）
- 実行日: 2026-10-19（UTC）。機体: Mac mini（Apple M4・16GiB・macOS 27.0.1・rustc 1.99.0）。ホスト名は記録しない
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`dirty=0`。`metal/rev-stamp.txt`。`crates/` の指紋はローカルとノードで一致を確認）
- 実行コマンド: `cargo test -p fandhe-ai --test compat_sequential_dropout_embedding_bag_backend_parity -- --ignored --nocapture --test-threads=1 metal_`（debug・ノードに `timeout` が無いため perl alarm 1800 秒。各 1 回のみ実行。判定外の `--no-run` ビルドを先に実施: 9 ファイル共通で `spatial-2522/metal/build.log`）
- 実行前後の環境と `.rev-stamp`: `metal/env_info.txt`（実行前後とも他の cargo／rustc プロセスなし）
- 生ログ: `metal/compat_sequential_dropout_embedding_bag_backend_parity.log`（末尾に `rc=`。ホスト名・ユーザー名・ホームパスはマスク済み）

| ファイル | 関数 | 結果 |
|---|---|---|
| `compat_sequential_dropout_embedding_bag_backend_parity.rs` | `metal_dropout_embedding_bag_bit_exact` | **fail**（rc=101、0 passed; 1 failed） |

- 合計: 1 件中 pass 0／fail 1／undetermined 0。
- fail の内容: Metal 経路の `Sequential::forward`（どの段〈Dropout／Sum／Mean／Max〉かはログから断定しない。メッセージは `max` reduction 由来のため EmbeddingBag の Max モードの可能性が高い）が `Backend(Unsupported("MetalBackendOps::max: reduction カーネル未実装（TASK-1.9c スコープ外）"))`（`compat_sequential_dropout_embedding_bag_backend_parity.rs:67`） を返して `unwrap()` で panic した。数値の不一致ではなく Metal バックエンドの `max` reduction が未実装であることによる。ログの最大誤差の出力はない。**要対応**（原因の切り分けと修正は別イシュー候補。tolerance・baseline・判定・テストは変えていない）。
- tolerance 定数・baseline・判定式・テストコードは不変。
- 注記: Metal の GPU カーネル経路を通ったか、Unsupported 経由のホストフォールバック経路を通ったかは本ログから断定できず未確認。
