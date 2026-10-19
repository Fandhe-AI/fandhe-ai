# サンプルごとの勾配（`Tape::vmap_grad`。#3073）CUDA／Metal 実機未実測の申し送り

> CUDA は #3171 で GB10 実測済み（2026-10-19・コミット b640875f・pass 1/1）。結果は末尾「CUDA 実測記入欄（#3171）」節。

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
| 2026-10-19 | DGX Spark GB10 | `cuda/RULE.txt` のコマンド（`--test-threads=1` 付き。フィルタ `cuda_vmap_grad`） | pass 1/1 | `.rev-stamp` dirty=0・ログ `cuda/functional_transforms_facade.log` |
| 未実測（#3172） | Apple Silicon（Mac mini） | 上記 metal コマンド | 未実測 | |

## CUDA 実測記入欄（#3171）

- 事前登録規則: `cuda/RULE.txt`（コミット `5fd2294e` で実測前に固定。実測は `b640875f` 時点の origin/main を `git archive` で書き出して実行）。
- 実行回数: 独立プロセスで 1 回（再実行・選別なし）。実行日 2026-10-19（UTC 05:3x。`cuda/env_info.txt` の `date_utc`）。
- 対象コミット: `b640875f86399515b2dab4dfc38089caaa5e5c08`（`cuda/rev-stamp.txt` は `dirty=0`。main の祖先）。
- 環境: GPU は NVIDIA GB10（aarch64・sm_121）、ドライバ 580.173.02、CUDA／nvcc 13.0.88、rustc 1.97.0。詳細は `cuda/env_info.txt`（各実行の前後を収録）。
- GPU 占有の前後比較: 実行前後とも GPU 使用率 0%、計算プロセスは常駐サービス 3 件（`<resident-process>`。メモリ使用量は前後で同一）のみで、実行中に新たな GPU プロセスは現れなかった。常駐サービスは止めていない。
- 実行コマンド: `cuda/RULE.txt` のとおり（`-p fandhe-ai --test functional_transforms_facade -- --ignored --nocapture --test-threads=1 cuda_vmap_grad`）。ログ末尾に `rc=0`。

| ファイル | テスト | 結果 | ログ |
|---|---|---|---|
| `functional_transforms_facade.rs` | `cuda_vmap_grad_matches_cpu_reference` | pass | `cuda/functional_transforms_facade.log` |

- ログの `running 1 test` と `test result: ok. 1 passed; 0 failed` の件数は事前登録の期待件数 1 と一致。合計 pass 1/1。
- ウォームアップ（`cargo test --no-run` のビルド。`cuda/build.log`）は判定外で、測定中にビルドは並走させていない（並列度 1）。
- tolerance 定数・判定式・baseline・テストコード・入力集合は変更していない（事前登録規則 `cuda/RULE.txt` のとおり）。再実行・選別はしていない。
- 実行フィルタは `cuda_vmap_grad`（`cuda` では同じバイナリの #2931 `cuda_functional_transforms_match_cpu_reference` まで走るため）。実行は 14 件が filtered out。
