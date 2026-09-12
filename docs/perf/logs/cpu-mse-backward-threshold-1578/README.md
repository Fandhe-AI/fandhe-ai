# mse_loss_backward 逐次フォールバックしきい値（イシュー #1578）実測ログ

事前登録した規則・全体の判定・原因分析は
`docs/perf/cpu-mse-backward-sequential-threshold.md` を正とする。
本ディレクトリはその実測生ログを保存する。

## 構成

- `m4max/`・`gb10/`: 機体別（内部ホスト名は含めない）
  - `run{1..5}.log`: Phase 0 マイクロベンチ（`mse_backward_threshold_sweep`。
    forced-seq／forced-par の中央値 ns を機体ごと・サイズごとに記録）
    のプロセス起動 5 回分
  - `compare-train-1578-cpu.md`: Phase 1（framework-compare train A/B・
    `--device cpu` の reuse セル。事前登録した必須判定）の結果表
  - `compare-train-1578-cpu-fresh-reference.md`: 同 fresh セル（対照・
    参考。判定には用いない）
  - `uptime-1578-cpu-1578.log`: Phase 1 各 run 前後の `uptime`
  - `env_info.txt`: 機体属性・実行時の共有負荷（record_only。専有ゲート
    なし）

## 再現

Phase 0:
```
cargo test -p fandhe-ai-backend-cpu --release --lib \
  mse::tests::mse_backward_threshold_sweep -- --ignored --nocapture
```

Phase 1（`scripts/bench/framework-compare/run_ab_1578.sh`。
`AB_BEFORE_FACADE_PATH`／`AB_AFTER_FACADE_PATH` に before（`origin/main`
の `crates/facade`）／after（本ブランチの `crates/facade`）の絶対パスを
指定）:
```
AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
AB_DEVICE=cpu AB_ROUNDS=5 \
  bash scripts/bench/framework-compare/run_ab_1578.sh 1578
```

### 候補しきい値 `T = 1 << 18`（262144）を復元する手順（重要）

§7 の判定（Phase 1 REJECT）により、本ブランチが出荷する
`MSE_BACKWARD_PARALLEL_MIN_ELEMS` は **`0`**（常に並列・変更前と
bit 同一の挙動）で確定している（`crates/backend-cpu/src/mse.rs`）。
したがって `AB_AFTER_FACADE_PATH` に本ブランチをそのまま指定すると、
before／after 双方が「常に並列」経路になり、上表（§6）に記録した
Phase 1 の A/B（after 腕が候補 `T = 1 << 18` を適用した状態）を
再現できない。

candidate `T = 1 << 18` を適用した after 腕を再現するには、本ブランチ
の隔離 checkout（例: `git worktree add` や別 clone）を用意したうえで、
コミットせずに定数のみを一時的に書き換える:

```
# 1) 本ブランチの隔離 checkout を用意する（例: git worktree）
git worktree add /path/to/after-t262144 <このブランチの commit>

# 2) 候補しきい値へ一時的に書き換える（コミットしない）
cd /path/to/after-t262144
sed -i.bak \
  's/pub(crate) const MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = 0;/pub(crate) const MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = 1 << 18;/' \
  crates/backend-cpu/src/mse.rs

# 3) この checkout の crates/facade を AB_AFTER_FACADE_PATH に指定する
AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
AB_AFTER_FACADE_PATH=/path/to/after-t262144/crates/facade \
AB_DEVICE=cpu AB_ROUNDS=5 \
  bash scripts/bench/framework-compare/run_ab_1578.sh 1578
```

before 腕は本イシューの機構（`MSE_BACKWARD_PARALLEL_MIN_ELEMS`・
`_with_threshold`）自体を持たないコミット（例:
`crates/backend-cpu/src/mse.rs` 変更前の親コミット、または本機構が
未マージの `origin/main`）を指定する。当時の実測がどの正確な
`origin/main` commit sha に対して行われたかは記録されていないため、
`ratio` の絶対値を厳密に再現する場合は
機構追加コミットの親（`git log --oneline -- crates/backend-cpu/src/mse.rs`
で確認できる直前のコミット）を before 腕として使う方が、日々動く
`origin/main` の HEAD よりも再現性が高い。

出荷済みの既定挙動（`T = 0`・常に並列）を検証したい場合は、上記の
書き換えをせず本ブランチをそのまま `AB_AFTER_FACADE_PATH` に指定
すればよい（ただしこれは §6 の A/B とは異なる比較になる点に注意）。
