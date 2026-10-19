# compat::Sequential の Pool3d 層経路 実機 parity 申し送り（イシュー #3029）

> **Metal は #3172 で Mac mini（Apple M4）実測済み（2026-10-19・対象 main `b640875f`・`dirty=0`・pass 1/1）。** 事前登録規則は `metal/RULE.txt`。

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
# Metal（Mac mini・Apple M4）
cargo test -p fandhe-ai --test compat_sequential_pool3d_backend_parity -- --ignored metal_
```

## 結果記入欄

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_pool3d_forward_backward_matches_cpu | 未実測（#3171） | |
| Mac mini（Apple M4） | metal_pool3d_forward_backward_matches_cpu | pass（b640875f・dirty=0） | 2026-10-19 |

## Metal 実測記入欄（#3172）

- 事前登録規則: `metal/RULE.txt`（実測前のコミットで固定。実測コミットでは変更していない）
- 実行日: 2026-10-19（UTC）
- 対象コミット: main `b640875f86399515b2dab4dfc38089caaa5e5c08`（`metal/rev-stamp.txt` の 2 行目は `dirty=0`。`metal/env_info.txt` の実行前後の stamp と一致）。事前登録コミットとの間で `crates/` の差分は無い
- 機体: Mac mini（Apple M4（10 コア）・メモリ 16GiB・macOS 27.0.1（26A434）・rustc 1.99.0・cargo 1.99.0・arm64（`metal/env_info.txt`））
- 実行コマンド: `cargo test -p fandhe-ai --release --test compat_sequential_pool3d_backend_parity -- --ignored --nocapture --test-threads=1 metal_`
  （`--release --nocapture --test-threads=1` は判定に影響しない実行オプション。上の素のコマンドとの差はこれだけ）
- ログ: `metal/compat_sequential_pool3d_backend_parity.log`（末尾 `rc=0`）・`metal/env_info.txt`・`metal/build.log`（判定外のウォームアップ）・`metal/rev-stamp.txt`

| テスト | 結果 |
|---|---|
| `metal_pool3d_forward_backward_matches_cpu` | pass（1 回） |

- 件数: `running 1 test`・`1 passed; 0 failed` で期待件数 1 と一致
- tolerance 定数・判定式・baseline・テストコード・入力は変更していない（REQ-2 統一複合判定。厳密ゼロ fail）
- 経路: pool3d は GPU カーネルが無く `Unsupported` からのホストフォールバックを想定する経路のため、本 pass は GPU カーネルの parity とは書かない。実際の経路（GPU フックかホストフォールバックか）は本測定では確認していない。
