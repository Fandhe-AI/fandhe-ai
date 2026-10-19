# サンプルごとの勾配（`Tape::vmap_grad`。#3073）CUDA／Metal 実機未実測の申し送り

> **Metal は #3172 で Mac mini（Apple M4）実測済み（2026-10-19・対象 main `b640875f`・`dirty=0`・pass 1/1）。** 事前登録規則は `metal/RULE.txt`。

`docs/autodiff-functional-transforms-design.md` §32 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/tests/functional_transforms_facade.rs` の 2 テスト
（`cuda_vmap_grad_matches_cpu_reference`・`metal_vmap_grad_matches_cpu_reference`。後者は `cfg(target_os = "macos")` 限定）
は `#[ignore]` のまま**未実測**である。実測は #3171（GB10）・#3172（Mac mini）が担当する。

同テストは facade `Tape::vmap_grad` を `tape_for(device)` の実機子テープで呼び、CPU の結果および
`vmap`／`vmap_grad`／`vjp` を経由しない明示ループ参照（スライスごとに新しいテープで `backward`）と比べる。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test functional_transforms_facade -- --ignored --nocapture cuda_vmap_grad

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test functional_transforms_facade -- --ignored --nocapture metal_vmap_grad
```

注記: `cuda` だけで絞ると同じバイナリの #2931 のテストも走るため `vmap_grad` で絞る。
`scripts/test-ignored.sh metal-facade` は本テストバイナリを含まない。上記の直接コマンドを使う。
CPU 版は同ファイルの属性なしテスト（`vmap_grad_*`）で検証済み（green）。

## 期待結果

- 新規カーネルは存在しない。既存 Op（narrow／reshape／contiguous・mul・tanh・matmul・sum）の合成と、
  サンプルごとの子テープでの `backward` の繰り返しという新しい呼び出し形が CPU と同じ結果になることを確かめる。
- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
  `fandhe_ai_backend_cpu::parity::assert_parity`）。bit 一致は契約にしない。形状は小さく Metal split-K は発動しない。
- 実機では定数（matmul の重み）をサンプルごとに転送する点を除き同じ経路を通る。
- 判定を外れた場合や `Unsupported` が出た場合は事実を記録して申し送る（tolerance の単独緩和は行わない。
  `.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 未実測（#3171） | DGX Spark GB10 | 上記 cuda コマンド | 未実測 | |
| 2026-10-19 | Mac mini（Apple M4） | 上記 metal コマンド（実測は `--release --test-threads=1` 付き） | pass（1/1） | main `b640875f`・dirty=0 |

## Metal 実測記入欄（#3172）

- 事前登録規則: `metal/RULE.txt`（実測前のコミットで固定。実測コミットでは変更していない）
- 実行日: 2026-10-19（UTC）
- 対象コミット: main `b640875f86399515b2dab4dfc38089caaa5e5c08`（`metal/rev-stamp.txt` の 2 行目は `dirty=0`。`metal/env_info.txt` の実行前後の stamp と一致）。事前登録コミットとの間で `crates/` の差分は無い
- 機体: Mac mini（Apple M4（10 コア）・メモリ 16GiB・macOS 27.0.1（26A434）・rustc 1.99.0・cargo 1.99.0・arm64（`metal/env_info.txt`））
- 実行コマンド: `cargo test -p fandhe-ai --release --test functional_transforms_facade -- --ignored --nocapture --test-threads=1 metal_vmap_grad`
  （`--release --nocapture --test-threads=1` は判定に影響しない実行オプション。上の素のコマンドとの差はこれだけ）
- ログ: `metal/functional_transforms_facade.log`（末尾 `rc=0`）・`metal/env_info.txt`・`metal/build.log`（判定外のウォームアップ）・`metal/rev-stamp.txt`

| テスト | 結果 |
|---|---|
| `metal_vmap_grad_matches_cpu_reference` | pass（1 回） |

- 件数: `running 1 test`・`1 passed; 0 failed` で期待件数 1 と一致
- tolerance 定数・判定式・baseline・テストコード・入力は変更していない（REQ-2 統一複合判定。厳密ゼロ fail）
- 経路: フィルタは `metal_vmap_grad`（`metal_` では同バイナリの #2931 テストも走るため）。新規カーネルは無く、実行経路は本測定では確認していない。
