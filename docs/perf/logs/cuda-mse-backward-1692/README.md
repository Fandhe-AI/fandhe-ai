# イシュー #1692 実測スキャフォールド（CUDA `mse_loss_backward` ストリーム順序契約）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10 実機
への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ディレクトリはスキャフォールド
（実行スクリプト・事前登録判定規則・記入欄）のみを提供し、実測は GB10
実機を持つセッションへ申し送る（`docs/perf/logs/train-resident-grad-
cuda-1560/` と同型の運用）。

## 目的

イシュー #1692「CUDA 側 `mse_loss_backward` のストリーム順序契約を確認し
GB10 で A/B する」の実測記録先。コード読解の結果、本番コード
（`crates/backend-cuda/src/{ops,mse}.rs`）への機能変更は不要と確認済み
（`docs/perf/cuda-mse-backward-stream-contract.md` §1）。本ディレクトリ
は新設マイクロベンチ（`crates/facade/tests/mse_backward_bench.rs`）を
before/after 2 ツリーで実行し、ノイズ床・再現性を記録する。

## 比較対象 2 腕（事前登録・固定）

- **before 腕**: 本イシューのブランチのマージ直前の `main`（新設ベンチ
  ファイル・`mse.rs` の doc comment 追記なし）
- **after 腕**: 本イシューのブランチ（`crates/*/src` の機能差分は
  `mse.rs` の doc comment 追記のみ。`git diff <before_sha> -- 'crates/*/src'`
  が意味のある機能差分を含まないことは本 PR の diff で確認できる）

両腕とも workspace `version = "0.8.0"`。`crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch（CLI `--config` 引数のみ。
deps-policy.md 第 9 区分）は不要（本ベンチは framework-compare 経由では
なく `cargo test -p fandhe-ai --test mse_backward_bench` を直接実行する）。

## 事前登録判定規則（計測前に固定・結果を見て変更しない）

`docs/perf/cuda-mse-backward-stream-contract.md` §2 のとおり:

- **対象**: `mse_backward_cases` の `train_shape`（主）・`general_shape`
  （副。`SIZES = [16384, 65536, 1048576]`）
- **手順**: 5 round・run 単位で起動順を反転・別プロセス
- **判定**: `median_s` の ratio(after/before) <= 1.00 を非後退の目安と
  するが、**`crates/*/src` の機能差分がなければ ADOPT／REJECT の判定
  対象ではなく、ノイズ床・再現性の記録として扱う**
- **checksum**: `grad[...].fold_bits` が before/after で完全一致するこ
  と
- **正しさ**: `crates/backend-cuda/tests/mse_parity.rs::
  mse_matches_cpu_across_shapes`（`#[ignore]`）を実行し REQ-2 複合判定
  pass を確認する

## 使い方（GB10 実機。ユーザー承認・別セッション）

```bash
BEFORE_TREE=/home/<user>/work/rust-ai-library-run-1692-before \
AFTER_TREE=/home/<user>/work/rust-ai-library-run-1692-after \
  ./orchestrate.sh 1692
```

`--dry-run` で経路解決のみ検証できる（実機不要。Linux で自己検証可能）。

## 出力

- `run-<label>.log`: 各 round の生出力
- `bench_lines.txt`: `bench[...].median_s=` 行の抽出（before/after 別）
- `env_info.txt`: 実行環境記入欄（内部ホスト名は書かない）
