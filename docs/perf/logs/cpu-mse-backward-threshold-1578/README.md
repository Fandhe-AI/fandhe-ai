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
指定。§6 の A/B を厳密に再現する場合は、本ブランチをそのまま使わず
下記「候補しきい値 `T = 1 << 18` を復元する手順」に従うこと）:
```
AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
AB_DEVICE=cpu AB_ROUNDS=5 \
  bash scripts/bench/framework-compare/run_ab_1578.sh 1578
```

### 候補しきい値 `T = 1 << 18`（262144）を復元する手順（重要）

`docs/perf/cpu-mse-backward-sequential-threshold.md` §7 の判定
（Phase 1 REJECT）により、本ブランチが出荷する
`MSE_BACKWARD_PARALLEL_MIN_ELEMS` は **`0`**（常に並列・変更前と
bit 同一の挙動）で確定している（`crates/backend-cpu/src/mse.rs`）。
したがって `AB_AFTER_FACADE_PATH` に本ブランチをそのまま指定すると、
before／after 双方が「常に並列」経路になり、同 doc §6 に記録した
Phase 1 の A/B（after 腕が候補 `T = 1 << 18` を適用した状態）を
再現できない。

機構を追加したコミットは `446b9912`
（`perf(backend-cpu): mse_loss_backward に要素数しきい値の逐次
フォールバック機構を追加する（既定 0＝常に並列）`）で、その直接の親
`f07d4822`（`perf(autodiff): reuse backward の ReLU マスクを非連続
transpose view に stride 対応させる (#1667)`）は本機構を持たない
`origin/main` 上のコミットである。当時の実測がどの正確な
`origin/main` commit sha に対して行われたかは記録として残っていない
ため、`ratio` の絶対値を厳密に再現する場合は、日々動く
`origin/main` の HEAD ではなく、この `f07d4822`（machine-readable な
最も近い再構成可能地点）を before 腕として固定する。

candidate `T = 1 << 18` を適用した after 腕を再現するには、`446b9912`
（以降の 2 コミットは docs のみで機構への変更はない）の隔離 checkout
（例: `git worktree add` や別 clone）を用意したうえで、コミットせずに
定数のみを一時的に書き換える:

```
# 0) before 腕: 機構を持たない直近の origin/main コミットを隔離 checkout する
git worktree add /path/to/before f07d4822

# 1) after 腕: 機構追加コミットを隔離 checkout する
git worktree add /path/to/after-t262144 446b9912

# 2) 候補しきい値へ一時的に書き換える（コミットしない）。
#    facade（AB_AFTER_FACADE_PATH に指定する crate）は同一 checkout 内の
#    backend-cpu を path 依存で取り込むため、この書き換えは after 腕の
#    ビルドへそのまま反映される（run_ab_1578.sh の cargo tree ガードは
#    fandhe-ai（facade）の解決先のみを検査し、その依存先である
#    backend-cpu の中身までは検査しない点に注意）。
cd /path/to/after-t262144
sed -i.bak \
  's/pub(crate) const MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = 0;/pub(crate) const MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = 1 << 18;/' \
  crates/backend-cpu/src/mse.rs
# 書き換えが実際に反映されたことを確認する（一致しなければ手順失敗）
grep -n 'MSE_BACKWARD_PARALLEL_MIN_ELEMS: usize = 1 << 18;' crates/backend-cpu/src/mse.rs

# 3) 各 checkout の crates/facade を AB_BEFORE_FACADE_PATH／
#    AB_AFTER_FACADE_PATH に指定する
AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
AB_AFTER_FACADE_PATH=/path/to/after-t262144/crates/facade \
AB_DEVICE=cpu AB_ROUNDS=5 \
  bash scripts/bench/framework-compare/run_ab_1578.sh 1578
```

出荷済みの既定挙動（`T = 0`・常に並列）を検証したい場合は、上記の
書き換えをせず本ブランチをそのまま `AB_AFTER_FACADE_PATH` に指定
すればよい（ただしこれは同 doc §6 の A/B とは異なる比較になる点に
注意）。
