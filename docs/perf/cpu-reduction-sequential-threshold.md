# CPU 縮約の rayon fork-join 要素数しきい値化（bit 同一・opt-in・既定 OFF）

イシュー #2101（親 #2099）。前段の低レイヤー診断（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4）で、
CPU backward の非 GEMM 残差の一因が小さい要素数での rayon fork-join 固定費
（640 要素の MSE backward で約 70 µs）と分かった。本件はその対策機構の導入記録である。

## 状態

- 機構: **導入済み・既定 OFF**（`REDUCTION_SEQUENTIAL_FALLBACK_ENABLED = false`）。ゲート OFF の間は変更前と完全に同一の経路。
- しきい値 `REDUCTION_PARALLEL_MIN_ELEMS = 1 << 18` は**未実測の暫定候補**（ゲート OFF の間は効かない）。
- 実機（M4 Max・GB10）の実測: **未実施**（`docs/perf/logs/elemental-reduction-threshold-2101/{m4max,gb10}/README.md` に手順）。
- #2102: 両機体 A/B の基盤（`scripts/bench/framework-compare/run_ab_reduction_threshold_cpu.sh`・`compare_gemm_ab.py --sizes large`・集計 `aggregate_sweep.py`）と**事前登録規則**（`docs/perf/logs/elemental-reduction-ab-2102/RULE.txt`）を整備済み。A/B・結線判断は**実測未実施のため未結線で、既定 OFF を維持**（ADOPT の場合のみ後続 PR で `reduction.rs` の 2 定数を切り替える）。
- `crates/autodiff/src/grad.rs`・`crates/backend-cpu/src/ops.rs` は #2102 でも 0 行変更（次節の理由による）。

## 対象サイトの食い違い（重要）

イシューは `crates/autodiff/src/grad.rs` の fork-join を挙げているが、`autodiff` は rayon に依存しない
（`Cargo.toml` は `fandhe-ai-tensor-core` のみ）ため、`accumulate`・`elementwise_mul_mask`・`reduce_to_shape` は
もともと逐次のホストループである。`ELEMENTWISE_VJP_VIA_BACKEND_OPS`（#1583 で REJECT 確定）の先にある
`backend-cpu::elementwise` は既に `PARALLEL_THRESHOLD = 1 << 15` を持つ。autodiff へ rayon を足すには
`Cargo.toml` の変更が要り、変更禁止項目に当たる。よって `grad.rs` は **0 行変更**とした。

しきい値のない rayon サイトが実在するのは `crates/backend-cpu/src/reduction.rs`
（`BackendOps::sum/max/min/mean/var/vector_norm/argmax/argmin` の CPU 本体）であり、機構はここに置いた。

## 設計

- 2 ヘルパー `chunk_partials`（全縮約）・`map_outputs`（軸指定）が全 rayon サイトの入口。判定は `SeqPolicy::run_sequential(numel)`（`enabled && numel < min_elems`）の 1 か所。
- しきい値の尺度は入力要素数（`dim=None` も `dim=Some` の `outer*axis_len*inner` も同じ）。
- `logsumexp`／`vector_norm_p` の全縮約は元から逐次（eval と bit 一致契約）なので対象外。
- テスト用の強制ポリシーは `#[cfg(test)]` の thread_local のみ。公開面（`pub` API）は増やさない。

### bit 同一の論証

- 全縮約: 逐次腕も「チャンク内 fold → チャンク番号順 fold」の 2 段構造を保つ（全体 1 本 fold にはしない）。`par_chunks().collect()` は入力順を保持するため両腕は同一の結合順序。
- 軸指定: 各出力要素は縮約軸を昇順に逐次累積する。出力要素間は独立で、`Range` の `collect()` は順序保持。
- 回帰テスト（`reduction::tests::forced_seq_and_par_are_bit_identical_*`）: CHUNK 境界・候補しきい値境界を含む n、NaN・-0.0・inf・subnormal・相殺列、軸指定、transpose／broadcast view、スレッド数 1／4 で seq・par・default の `to_bits()` 完全一致を確認する。
- CPU の出力がゲート状態によらず不変なので、CPU を参照点とする CUDA・Metal の parity テストの判定は変わらない（CPU と GPU の bit 一致は主張しない。REQ-2 は複合判定）。CUDA・Metal のコードには触れていない。

## 既存記録との違い

| 記録 | 内容 | 本件との違い |
|------|------|-------------|
| `elementwise-vjp-backend-ops.md`（#1583） | VJP の計算経路を BackendOps へ振り替えるルーティング変更。per-call 転送・並列経路のコストで REJECT | 本件は同一カーネル内で要素数により逐次・並列を選ぶだけ。計算経路・結合順序は不変 |
| `cpu-mse-backward-sequential-threshold.md`（#1578） | `mse_loss_backward` のしきい値化。GB10 reuse 1.0167× で REJECT | 対象サイトが異なる（縮約）。同型の再実行ではない。`mse_loss_backward` は本件でも対象外 |

## 期待効果の限界（#2102 への申し送り）

framework-compare の train（MLP・batch 64）の CPU 経路で通る rayon サイトは MSE（除外）と GEMM（別方針）だけで、
bias 勾配は `eval::reduce_bias_grad_rows`、SGD は逐次ループである。本機構を結線しても train・infer に差が出ない
可能性が高い。#2102 の主な証拠は `#[ignore]` のスイープ（`reduction_threshold_sweep`）になり、train・infer は非後退の
確認に使う。train の高速化は約束しない。

## 事前登録規則

実測前に固定した規則は `docs/perf/logs/elemental-reduction-threshold-2101/RULE.txt` を正とする（本 doc へ二重管理しない）。

## スコープ外

- `mse_loss_backward`（#1578 REJECT）
- `mse.rs::mse_sum_sq_f32`・`bce.rs` backward のしきい値化（候補。別途判断）
- `elementwise::PARALLEL_THRESHOLD` の見直し（共有値。承認事項）
- `grad.rs` の逐次ホストループの高速化
- 実機計測・Metal のパリティ実行（Mac・GB10 セッションへ申し送り）
