# CPU GEMM B パネル packing のスレッド間共有化＋packing 並列化 計測記録（#750）

イシュー #750「B パネル packing のスレッド間共有化（設計済み案 B）+ packing 並列化」の実測記録。

## 本 PR の結論: 実装のみ・本番不採用（採用ゲート未通過）

`docs/cpu-gemm-b-packing-sharing-decision.md`（#565）が設計検討済みの**案 B**（BLIS 方式:
jc/pc 直列 ＋ (jc,pc) ブロックごとに B を 1 本だけ共有 pack ＋ ic 行パネル並列）を実装した
（`crates/backend-cpu/src/gemm_blis/mod.rs` の `gemm_blis_shared_b_region`／`gemm_blis_ic_loop`／
`dispatch_shared_b`）。あわせて B packing 自体を `nr` ブロック単位で `par_chunks_mut` により
並列化した。

**本番公開入口（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`。`crate::ops` の
`BackendOps` 実装から呼ばれる実運用経路）は、B パネル共有経路（`dispatch_shared_b`）を
採用しない**。従来どおり行パネルごとに独立して `dispatch_region`（B をタスクごとに
個別 pack する従来方式＝ PerTaskPrivateB 相当）を呼ぶ。共有経路の実装
（`gemm_blis_shared_b_region`／`dispatch_shared_b`）は `#[cfg(test)]` 限定で残し、bit 完全
一致・回帰テスト（`#[cfg(test)]` の `gemm_blis_parallel_with_blocks` 経由）のみに使う。

これは PR #758（イシュー #740・mma_f16 threadblock swizzle）の最終マージ状態（同 PR 内の
commit `8269801`「mma_f16 threadblock swizzle の本番結線を差し戻す」により、一時的に本番既定
コンストラクタ `CudaMmaGemm::new` へ結線した swizzle 変種を差し戻し、`internal-diagnostics`
feature 限定の `new_with_swizzle`／`new_without_swizzle` 経由でのみ到達可能な構成へ戻した状態）
で採られた「実装・実機 A/B 計測基盤は入れるが、実機性能ゲートを通過するまでは本番既定へ結線
しない」判断と同型であり、ユーザー承認済みの前例に倣う（codex-review P1 指摘・
thread `PRRT_kwDOTuUCJc6arIUt` を受けた是正）。

## 採用ゲート（実機ゲート。マージ後の残作業ではなく将来の採用判断の前提条件）

以下の受け入れ条件は、**マージ後に「いつか実施する残作業」ではなく、本変更を本番既定へ
昇格させる（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel` から `dispatch_shared_b` を
呼ぶよう再度結線する）ために必ず満たすべき前提条件**として扱う。満たされない限り、
`gemm_blis_shared_b_region`／`dispatch_shared_b` はテスト専用コードのまま維持し、本番経路を
変更しない。

| # | 受け入れ条件 | 本 PR 時点の状況 |
|---|---|---|
| 1 | `tests/gemm_blis_parity.rs` を含む全 parity テストが bit 完全一致で全 pass | 満たす（後述「数値一致」節） |
| 2 | 実機（Apple M4 Max）5 回計測中央値で M=N=K=2048/4096 の非劣化＋改善を確認し記録 | **未実施（環境ゲート未達。下記参照。本番採用の前提条件として未充足）** |
| 3 | スレッド数 1 では従来と同一経路・同一性能（並列化による退行なし） | 満たす（構造的に保証。後述） |

**採用判断のフロー**: 受け入れ条件 2（Apple M4 Max 実機実測での非劣化・改善確認）を満たし、
かつユーザー承認を得た**別 PR** でのみ、`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`
を `dispatch_shared_b` へ結線する変更を行う。本 PR ではこの結線を行わない。

## 実機性能実測（受け入れ条件 2）: 環境ゲート未達・実測未実施

本タスクを実行した実装エージェントの環境は Linux x86_64 の隔離コンテナ（`.claude/worktrees/`
配下の git worktree）であり、`docs/real-hardware-verification-env.md` が要求する Apple M4 Max
実機への SSH アクセス（`docs/real-hardware-verification-env.local.md`）が存在しない（ローカル
実値ファイルは `.gitignore` 対象で本セッションには供給されていない）。

`docs/cpu-gemm-blocking-sweep.md`・`docs/cpu-gemm-prefetch-decision.md` 等の既存 fail-closed
前例（**実測値の捏造・placeholder 値での完了扱いは行わない**）に従い、本 PR 時点では
M=N=K=2048/4096 の実機実測（5 回計測中央値・`gemm_blis_baseline_pytorch_square_512_to_4096`
または新規ハーネスによる before/after 比較）を**実施しない**。実機実測は以下のコマンドで
Apple M4 Max 実機から再現可能な状態にある（**本番採用の可否を判断するための前提条件**であり、
マージ後に無条件で実施すべき残作業ではない）:

```bash
# before（main・本番既定の PerTaskPrivateB 相当経路）・after（`dispatch_shared_b` へ本番結線
# した検証用ブランチ）それぞれで実行し、1024/2048/4096 の median_secs / TFLOPS を比較する。
cargo test -p backend-cpu --release -- --ignored gemm_blis_baseline_pytorch_square_512_to_4096 --nocapture
```

2048/4096 で非劣化が確認できない場合は、ロードバランス改善（ic チャンク粒度の細分化）を
A/B し、それでも非劣化を満たせなければ本変更の採用を見送り記録のみ残す方針（実装計画 §8）。
非劣化・改善が確認できた場合に限り、上記「採用判断のフロー」に従って別 PR で本番結線する。

### 追記（イシュー #793・環境ゲート再判定）

イシュー #793「共有 B packing の実機非劣化ゲートと本番結線」で本番結線の着手を試みたが、
実装セッションの環境は本 PR（#750）時点と同じく `uname -sm` が `Linux x86_64`（隔離 worktree）
であり、`docs/real-hardware-verification-env.local.md` も未供給（存在しない）ため、Apple M4 Max
実機到達手段が確立できなかった。上記の fail-closed 前例（実測値の捏造・placeholder 値での完了
扱いを行わない）に従い、#793 でも受け入れ条件 2 の実機実測・本番結線は**実施していない**。
再現手順（本節冒頭のコマンド）は変更なくそのまま有効。**受け入れ条件 2 は依然未充足であり、
イシュー #793 は本項目の充足待ちとしてオープンのまま維持する**。

## 受け入れ条件 3（スレッド数 1 の同一経路）: 構造的保証＋テストで確認

本番公開入口（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）は共有経路を採用しないため、
スレッド数に関わらず（1 の場合も複数の場合も）常に本変更前と同一の `dispatch_region` 単発／
行パネル並列呼び出し経路を通る。テスト専用入口 `gemm_blis_parallel_with_blocks`
（`#[cfg(test)]`）のみが `panel_rows = m.div_ceil(num_threads).max(1)` を計算した直後に
`if m <= panel_rows` で早期分岐し、`num_threads == 1`（`panel_rows == m` が常に成立）では
`dispatch_region` を、実タスク数 2 以上では `dispatch_shared_b` を呼び分けて共有経路の
bit 完全一致を検証する。

この構造的保証を単体テスト `gemm_blis_parallel_single_thread_pool_matches_naive_bit_exact`
（`crates/backend-cpu/src/gemm_blis/mod.rs`）で直接検証済み（`rayon::ThreadPoolBuilder::
num_threads(1)` で強制し `gemm_naive` と bit 完全一致することを確認）。

## 数値一致（受け入れ条件 1）

- 既存 `tests/gemm_blis_parity.rs`（`gemm_blis_parallel_matches_naive_bit_exact_across_thread_pools`
  ほか。num_threads = 1/3/16 を横断し bit 完全一致を検証）・`tests/gemm_epilogue_parity.rs` は
  無変更で全 pass（`cargo test -p backend-cpu` 実測。141 lib 単体テスト・17 gemm_blis_parity・
  14 gemm_epilogue_parity すべて green）
- 新設した単体テスト（`crates/backend-cpu/src/gemm_blis/mod.rs`）:
  - `gemm_blis_parallel_single_thread_pool_matches_naive_bit_exact`（受け入れ条件 3 の直接検証。
    本番公開入口 `gemm_blis_parallel` を直接呼ぶ。T=1 では共有経路を経由しないため本番経路の
    検証で十分）
  - `gemm_blis_shared_b_region_multi_sync_point_matches_serial_bit_exact`（`#[cfg(test)]` 限定の
    テスト専用入口 `gemm_blis_parallel_with_blocks` 経由。小さい `BlockSizes`〈mc=16/kc=17/
    nc=19〉で多数の (jc,pc) 同期点を強制し、B パネル共有経路が直列経路と bit 完全一致すること
    を検証）
  - `gemm_blis_parallel_matches_naive_bit_exact_when_tasks_fewer_than_threads`（`#[cfg(test)]`
    限定の `gemm_blis_parallel_with_blocks` 経由。実タスク数 Q が rayon 稼働スレッド数 T を
    下回る形状〈m=10・num_threads=16〉での `gemm_blis_shared_b_region` の `num_tasks` 導出回帰。
    本番公開入口 `gemm_blis_parallel` は共有経路を採用しないため、共有経路自体の回帰検証には
    テスト専用入口を経由する必要がある〈Cursor Bugbot 指摘・commit f27f233 是正〉）
- 新設した統合テスト（`crates/backend-cpu/tests/gemm_epilogue_parity.rs`）:
  - `gemm_bias_act_matches_composed_reference_bit_exact_across_thread_pools`（num_threads =
    1/2/3/16 を横断し、epilogue が GEMM 本体完了後にちょうど 1 回だけ適用されることを
    非融合合成参照との bit 完全一致で検証）

## 参考（非公式・受け入れ条件 2 を満たすものではない・本番未結線時点の診断値）

Apple M4 Max 実機の代替にはならず、かつ本番へ結線されていない `dispatch_shared_b` を
テスト専用入口経由で直接動作させた開発環境（Linux x86_64・AVX2+FMA・12 論理コア）での
診断値であり、**本番既定経路の性能を表すものではない**。共有経路の実装が実機非依存に
正常動作することの sanity check として記録するのみで、受け入れ条件 2 の充足を主張するもの
ではない（受け入れ条件 2 充足には Apple M4 Max 実機での本番結線前後比較が必須）:

```
M=N=K=512:  gemm_blis_parallel median=0.002002s（対 gemm_parallel 1.776x）
M=N=K=1024: gemm_blis_parallel median=0.005129s（対 gemm_parallel 2.444x）
M=N=K=2048: gemm_blis_parallel median=0.040295s（対 gemm_parallel 1.801x）
```

このハーネスは本変更前後の直接比較（before/after）ではなく `gemm_blis_parallel` 対
`crate::gemm::gemm_parallel`（別実装）の比較のため、案 B 導入による性能変化の定量評価には
使えない。また計測時点では `dispatch_shared_b` が `gemm_blis_parallel` 本体に結線されていた
実装段階の値であり、本 PR で本番結線を撤回した後の `gemm_blis_parallel` の挙動（従来経路）
とは異なる経路の実測値である点に注意する。

## 計測環境（本記録作成時点。実機実測ではなく開発環境の記録）

| 項目 | 値 |
|------|-----|
| OS | Linux 7.0.0-29-generic（x86_64） |
| 論理コア数 | 12（`nproc`） |
| 用途 | ビルド・lint・単体/統合テストの実行環境、および上記「参考」節の非公式 sanity check |

## 関連

- 設計 doc: `docs/cpu-gemm-b-packing-sharing-decision.md`（#565）
- 実装: `crates/backend-cpu/src/gemm_blis/mod.rs`（`gemm_blis_shared_b_region`／
  `gemm_blis_ic_loop`／`IcLoopContext`／`dispatch_shared_b`。いずれも `#[cfg(test)]` 限定で
  本番未結線）
