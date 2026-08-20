# CPU GEMM B パネル packing のスレッド間共有化＋packing 並列化 計測記録（#750）

イシュー #750「B パネル packing のスレッド間共有化（設計済み案 B）+ packing 並列化」の実測記録。

## 背景

`docs/cpu-gemm-b-packing-sharing-decision.md`（#565）が設計検討済みの**案 B**（BLIS 方式:
jc/pc 直列 ＋ (jc,pc) ブロックごとに B を 1 本だけ共有 pack ＋ ic 行パネル並列）を実装した
（`crates/backend-cpu/src/gemm_blis/mod.rs` の `gemm_blis_shared_b_region`／`gemm_blis_ic_loop`／
`dispatch_shared_b`）。あわせて B packing 自体を `nr` ブロック単位で `par_chunks_mut` により
並列化した。

実タスク数が 1（`m <= panel_rows`。`num_threads == 1` を含む）の場合は従来の
`dispatch_region` 単発呼び出しへ早期分岐し、B パネル共有経路を一切経由しない
（受け入れ条件 3。`gemm_blis_parallel` 実装コメント参照）。

## 受け入れ条件と本記録の対応

| # | 受け入れ条件 | 本 PR 時点の状況 |
|---|---|---|
| 1 | `tests/gemm_blis_parity.rs` を含む全 parity テストが bit 完全一致で全 pass | 満たす（後述「数値一致」節） |
| 2 | 実機（Apple M4 Max）5 回計測中央値で M=N=K=2048/4096 の非劣化＋改善を確認し記録 | **未実施（環境ゲート未達。下記参照）** |
| 3 | スレッド数 1 では従来と同一経路・同一性能（並列化による退行なし） | 満たす（構造的に保証。後述） |

## 実機性能実測（受け入れ条件 2）: 環境ゲート未達・実測未実施

本タスクを実行した実装エージェントの環境は Linux x86_64 の隔離コンテナ（`.claude/worktrees/`
配下の git worktree）であり、`docs/real-hardware-verification-env.md` が要求する Apple M4 Max
実機への SSH アクセス（`docs/real-hardware-verification-env.local.md`）が存在しない（ローカル
実値ファイルは `.gitignore` 対象で本セッションには供給されていない）。

`docs/cpu-gemm-blocking-sweep.md`・`docs/cpu-gemm-prefetch-decision.md` 等の既存 fail-closed
前例（**実測値の捏造・placeholder 値での完了扱いは行わない**）に従い、本 PR 時点では
M=N=K=2048/4096 の実機実測（5 回計測中央値・`gemm_blis_baseline_pytorch_square_512_to_4096`
または新規ハーネスによる before/after 比較）を**実施しない**。実機実測は以下のコマンドで
Apple M4 Max 実機から再現可能な状態にある（本 PR のマージ後の残作業）:

```bash
# before（main）・after（本ブランチ）それぞれで実行し、1024/2048/4096 の
# median_secs / TFLOPS を比較する。
cargo test -p backend-cpu --release -- --ignored gemm_blis_baseline_pytorch_square_512_to_4096 --nocapture
```

2048/4096 で非劣化が確認できない場合は、ロードバランス改善（ic チャンク粒度の細分化）を
A/B し、それでも非劣化を満たせなければ本変更の採用を見送り記録のみ残す方針（実装計画 §8）。

## 受け入れ条件 3（スレッド数 1 の同一経路）: 構造的保証＋テストで確認

`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`／`gemm_blis_parallel_with_blocks` は
`panel_rows = m.div_ceil(num_threads).max(1)` を計算した直後に `if m <= panel_rows` で
早期分岐し、`dispatch_region`（本変更前と同一のロジック）を単発呼び出しする。
`num_threads == 1` では `panel_rows == m` が常に成立するため、この分岐へ必ず含まれる
（B パネル共有経路 `dispatch_shared_b` を一切経由しない）。

この構造的保証を単体テスト `gemm_blis_parallel_single_thread_pool_matches_naive_bit_exact`
（`crates/backend-cpu/src/gemm_blis/mod.rs`）で直接検証済み（`rayon::ThreadPoolBuilder::
num_threads(1)` で強制し `gemm_naive` と bit 完全一致することを確認）。

## 数値一致（受け入れ条件 1）

- 既存 `tests/gemm_blis_parity.rs`（`gemm_blis_parallel_matches_naive_bit_exact_across_thread_pools`
  ほか。num_threads = 1/3/16 を横断し bit 完全一致を検証）・`tests/gemm_epilogue_parity.rs` は
  無変更で全 pass（`cargo test -p backend-cpu` 実測。141 lib 単体テスト・17 gemm_blis_parity・
  14 gemm_epilogue_parity すべて green）
- 新設した単体テスト（`crates/backend-cpu/src/gemm_blis/mod.rs`）:
  - `gemm_blis_parallel_single_thread_pool_matches_naive_bit_exact`（受け入れ条件 3 の直接検証）
  - `gemm_blis_shared_b_region_multi_sync_point_matches_serial_bit_exact`（小さい
    `BlockSizes`〈mc=16/kc=17/nc=19〉で多数の (jc,pc) 同期点を強制し、B パネル共有経路が
    直列経路と bit 完全一致することを検証）
  - `gemm_blis_parallel_matches_naive_bit_exact_when_tasks_fewer_than_threads`（実タスク数
    Q が rayon 稼働スレッド数 T を下回る形状〈m=10・num_threads=16〉の回帰）
- 新設した統合テスト（`crates/backend-cpu/tests/gemm_epilogue_parity.rs`）:
  - `gemm_bias_act_matches_composed_reference_bit_exact_across_thread_pools`（num_threads =
    1/2/3/16 を横断し、epilogue が GEMM 本体完了後にちょうど 1 回だけ適用されることを
    非融合合成参照との bit 完全一致で検証）

## 参考（非公式・受け入れ条件 2 を満たすものではない）: 開発環境 x86_64 での動作確認

Apple M4 Max 実機の代替にはならないが、本変更が実際に動作し明確な退行がないことの参考として、
開発環境（Linux x86_64・AVX2+FMA）で既存の `gemm_blis_perf_square_512_1024_2048`（12 論理
コア環境。実タスク数 >= 2 で B パネル共有経路 `dispatch_shared_b` を経由する）を実行した:

```
M=N=K=512:  gemm_blis_parallel median=0.002002s（対 gemm_parallel 1.776x）
M=N=K=1024: gemm_blis_parallel median=0.005129s（対 gemm_parallel 2.444x）
M=N=K=2048: gemm_blis_parallel median=0.040295s（対 gemm_parallel 1.801x）
```

このハーネスは本変更前後の直接比較（before/after）ではなく `gemm_blis_parallel` 対
`crate::gemm::gemm_parallel`（別実装）の比較のため、案 B 導入による性能変化の定量評価には
使えない。あくまで「共有 B 経路が実機非依存に正常動作する」ことの sanity check であり、
受け入れ条件 2 の充足を主張するものではない。

## 計測環境（本記録作成時点。実機実測ではなく開発環境の記録）

| 項目 | 値 |
|------|-----|
| OS | Linux 7.0.0-29-generic（x86_64） |
| 論理コア数 | 12（`nproc`） |
| 用途 | ビルド・lint・単体/統合テストの実行環境、および上記「参考」節の非公式 sanity check |

## 関連

- 設計 doc: `docs/cpu-gemm-b-packing-sharing-decision.md`（#565）
- 実装: `crates/backend-cpu/src/gemm_blis/mod.rs`（`gemm_blis_shared_b_region`／
  `gemm_blis_ic_loop`／`IcLoopContext`／`dispatch_shared_b`）
