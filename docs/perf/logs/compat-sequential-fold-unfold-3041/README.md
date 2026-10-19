# compat::Sequential の Unfold／Fold 層経路 実機 parity 申し送り（イシュー #3041）

> **Metal は #3172 で Mac mini（Apple M4）実測済み（2026-10-19・対象 main `b640875f`・`dirty=0`・pass 1/1）。** 事前登録規則は `metal/RULE.txt`。

本イシューは `unfold`／`fold` の保存対応と、層経路の `#[ignore]` parity テストの追加まで。
実機（CUDA／Metal）の実測と結果記入は Phase 8 の #3171・#3172 が担う。CPU 側の正しさは
`crates/facade/tests/compat_sequential_fold_unfold.rs` で検証済み。tolerance・baseline は変更しない
（fail が出ても tolerance を緩めない）。演算単体の parity は `fold_unfold_backend_parity.rs` が担い、
本テストは層（`Sequential`）経由。

## 判定契約

- facade `Sequential`（`add_unfold([2,3],[2,1],[1,1],[1,2])` → `add_fold([6,7],[2,3],[2,1],[1,1],[1,2])`。
  入力 NCHW `[2, 3, 6, 7]` の ramp）の forward 出力と、`sum` 損失に対する入力勾配を CPU と
  REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`assert_parity`）で比較する。
- 期待結果の注記: `fold`／`unfold` は既存の `im2col`／`col2im` 経路を通るため、新規カーネルの
  parity ではなく既存カーネル経路の層経由での確認になる。

## 実行コマンド

```
# CUDA（DGX Spark GB10）
cargo test -p fandhe-ai --test compat_sequential_fold_unfold_backend_parity -- --ignored cuda_
# Metal（Mac mini・Apple M4）
cargo test -p fandhe-ai --test compat_sequential_fold_unfold_backend_parity -- --ignored metal_
```

## 結果記入欄

| 環境 | テスト | 結果 | 日付 |
|---|---|---|---|
| GB10 | cuda_fold_unfold_forward_backward_matches_cpu | 未実測（#3171） | |
| Mac mini（Apple M4） | metal_fold_unfold_forward_backward_matches_cpu | pass（b640875f・dirty=0） | 2026-10-19 |

## Metal 実測記入欄（#3172）

- 事前登録規則: `metal/RULE.txt`（実測前のコミットで固定。実測コミットでは変更していない）
- 実行日: 2026-10-19（UTC）
- 対象コミット: main `b640875f86399515b2dab4dfc38089caaa5e5c08`（`metal/rev-stamp.txt` の 2 行目は `dirty=0`。`metal/env_info.txt` の実行前後の stamp と一致）。事前登録コミットとの間で `crates/` の差分は無い
- 機体: Mac mini（Apple M4（10 コア）・メモリ 16GiB・macOS 27.0.1（26A434）・rustc 1.99.0・cargo 1.99.0・arm64（`metal/env_info.txt`））
- 実行コマンド: `cargo test -p fandhe-ai --release --test compat_sequential_fold_unfold_backend_parity -- --ignored --nocapture --test-threads=1 metal_`
  （`--release --nocapture --test-threads=1` は判定に影響しない実行オプション。上の素のコマンドとの差はこれだけ）
- ログ: `metal/compat_sequential_fold_unfold_backend_parity.log`（末尾 `rc=0`）・`metal/env_info.txt`・`metal/build.log`（判定外のウォームアップ）・`metal/rev-stamp.txt`

| テスト | 結果 |
|---|---|
| `metal_fold_unfold_forward_backward_matches_cpu` | pass（1 回） |

- 件数: `running 1 test`・`1 passed; 0 failed` で期待件数 1 と一致
- tolerance 定数・判定式・baseline・テストコード・入力は変更していない（REQ-2 統一複合判定。厳密ゼロ fail）
- 経路: 既存 im2col／col2im 経路の層経由での確認。実行経路は本測定では確認していない。
