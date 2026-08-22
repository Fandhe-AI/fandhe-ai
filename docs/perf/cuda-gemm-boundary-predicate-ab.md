# CUDA GEMM 境界検査ビットマスク事前計算 A/B 計測記録（#537）

イシュー #537「perf(backend-cuda): 境界検査のビットマスク事前計算化の要否を実機で評価」の A/B 計測手順・
head 案設計スケッチ・判断基準・記録テンプレート。`crates/backend-cuda/src/kernels_mma.rs` の `MMA_F16`
カーネル（f16 `mma.sync`/`ldmatrix`/`cp.async` 経路）が A/B タイルロード時・エピローグ書き戻し時に毎回
評価している境界比較を、CUTLASS `predicated_tile_access_iterator.h` の `compute_predicates_` と同型の
「スレッドごとに 1 回だけビットマスクへ事前計算し、以後はビット参照のみで判定する」方式へ置き換える価値
があるかを評価する。

## 状態: 未計測・実機計測待ち（本番カーネルは一切変更していない）

本ドキュメントは #497（蛇行走査。`docs/perf/cuda-gemm-serpentine-ab.md`）・PR #657 の codex-review 指摘
（P1: 性能改善を実測せずに `MMA_F16` 本番カーネルへ変更を導入している）を受けた前例（doc-only 整備 →
実機セッションで計測 → 判断基準を満たした場合のみ別 PR で導入）と同一方針に従う。Linux worktree
（NVRTC 非搭載環境）では実機計測ができないため、本 PR の時点では計測手順・head 案設計・記録テンプレート
のみを整備し、`kernels_mma.rs` 自体は未変更のまま据え置く。実機ツリー #408 側のセッションで下記手順に
従い計測し、判断基準を満たした場合にのみ別 PR で導入する。

## 参照実装の根拠（CUTLASS `predicated_tile_access_iterator.h`）

CUTLASS の `PredicatedTileAccessIterator` は、タイルアクセスごとの有効/無効判定を毎回の座標比較として
行わず、以下の 3 段階に分離している。

- `compute_predicates_()`: イテレータ構築時（K ループに入る前）に 1 回だけ、担当する全アクセスの
  有効/無効を `uint32_t predicates_` のビット配列へ書き込む
- `valid()`: K ループ内では `predicates_ & mask_` のビット参照のみで有効/無効を判定する（座標比較の
  再計算をしない）
- `clear_mask()`: 残余 K タイル（K が block tile で割り切れない末尾）に入る際、対応ビットを一括で 0
  埋めする

本実装（`kernels_mma.rs`）は上記のいずれも行わず、A/B ロードのたびに `gr < m && gc < k`（A 側）・
`gr < k && gc < n`（B 側）の比較演算を実行し、エピローグでも要素ごとに `r0 < m && c0 < n` 等の比較を
評価する。

## 本実装の現状（現行 main 時点の行番号・証跡）

| 箇所 | ファイル・行 | 内容 |
|------|------------|------|
| A タイルロード | `crates/backend-cuda/src/kernels_mma.rs:415-425`（`LOAD_A_STAGE` マクロ） | 8 要素チャンクごとに `int valid = (gr < m && gc < k) ? 16 : 0;`（423 行）を毎回評価し `mma_cp_async16` の src-size に渡す |
| B タイルロード | `crates/backend-cuda/src/kernels_mma.rs:427-437`（`LOAD_B_STAGE` マクロ） | 同様に `int valid = (gr < k && gc < n) ? 16 : 0;`（435 行） |
| エピローグ guarded store | `crates/backend-cuda/src/kernels_mma.rs:690-704` | `WARP_TILES_M x WARP_TILES_N`（2x2）の mi/nj ループ内で、フラグメント 4 要素それぞれに `r0 < m && c0 < n`・`r0 < m && c1 < n`・`r1 < m && c0 < n`・`r1 < m && c1 < n`（699-702 行）を評価してから `c[...] = __float2half(...)` する |
| ソース証跡テスト | `crates/backend-cuda/src/kernels_mma.rs:763-778`（`mma_f16_source_retains_req8_boundary_guards`） | 上記 6 つの式文字列（`"gr < m && gc < k"`・`"gr < k && gc < n"`・`"r0 < m && c0 < n"`・`"r0 < m && c1 < n"`・`"r1 < m && c0 < n"`・`"r1 < m && c1 < n"`）を needle として `MMA_F16.contains(needle)` をロックしている |

**採用する場合、このソース証跡テストは事前計算後のコード（ビットマスク変数名・`valid()` 相当の参照式）
に合わせて更新が必須になる**（needle 文字列がそのまま残らないため）。REQ-8「境界検査を省略しない」は
検査の削除ではなく前倒しなので、更新後も何らかの形で境界検査の実在を機械検証するテストを維持すること。

## head 案の設計スケッチ

実機セッションが短時間で再実装できる粒度で、変更の要点のみを記す（コンパイル可能な完全なパッチではない）。

### A/B ロード側

各スレッドが担当する `(row, col0)` チャンクは K ループの外で決まる固定値であるため、行方向の述語
（A 側: `gr < m`。B 側: `gc < n`）は K ループへ入る前に 1 回だけビット化できる。列方向（K 方向）は
`validate_mma_alignment`（ホスト側 `run_f16` が起動前に必ず検証。`k % 8 == 0 && n % 8 == 0` を要求）
により 8 要素チャンク単位の二値判定になり、部分無効が生じ得るのは最終 K タイル（`k0 + BK > k` となる
最後の反復）のみである。したがって:

```c
// K ループ前（プロローグ）: 行方向の有効/無効を 1 回だけビット化する。
// idx でループする範囲は BM*BK/8（A）・BK*BN/8（B）個の固定チャンクなので
// uint32_t 1 語（32bit）では収まらない可能性がある点に注意
// （BM/BK/BN の実寸に応じて複数語 or per-thread レジスタでの保持を検討）。
unsigned row_valid_a = /* 行方向 gr < m をチャンクごとに 1 ビットへ詰めたもの */;
unsigned row_valid_b = /* 行方向 gc < n をチャンクごとに 1 ビットへ詰めたもの */;

// K ループ内（各 stage）: 最終タイル以外は行方向ビットのみで判定できる。
// 最終タイルに限り列方向マスクを別途 1 回だけ計算して AND する。
#define LOAD_A_STAGE(stage, k0) \
    for (...) { \
        ...
        int valid = (row_valid_a_bit(idx) && (is_last_k_tile ? col_mask_a_bit(idx) : 1)) ? 16 : 0; \
        ...
    }
```

### エピローグ側

`(r0 < m && c0 < n)` 等の 4 条件は warp 全体で見るとループ不変（`WARP_TILES_M x WARP_TILES_N x 4` =
`2 x 2 x 4 = 16` 要素分）である。エピローグに入った時点で 1 回だけ `uint32_t` 1 語（16bit で収まる）へ
事前計算し、mi/nj ループ内ではビット参照のみで分岐する。

```c
// エピローグ入口で 1 回だけ計算（mi/nj/成分インデックスから一意なビット位置へ）
unsigned epilogue_valid = 0;
#pragma unroll
for (int mi = 0; mi < WARP_TILES_M; ++mi) {
    for (int nj = 0; nj < WARP_TILES_N; ++nj) {
        int r0 = row0_warp + mi * MMA_M + group_id;
        int r1 = r0 + 8;
        int c0 = col0_warp + nj * MMA_N + tid_in_group * 2;
        int c1 = c0 + 1;
        int bit = (mi * WARP_TILES_N + nj) * 4;
        epilogue_valid |= (r0 < m && c0 < n) ? (1u << (bit + 0)) : 0;
        epilogue_valid |= (r0 < m && c1 < n) ? (1u << (bit + 1)) : 0;
        epilogue_valid |= (r1 < m && c0 < n) ? (1u << (bit + 2)) : 0;
        epilogue_valid |= (r1 < m && c1 < n) ? (1u << (bit + 3)) : 0;
    }
}
// 書き戻しループはビット参照のみ（座標比較を再実行しない）
```

### REQ-8 適合の論拠

`coding-rust.md`「カーネル実装の境界検査（REQ-8）」は「性能下限・最適化の達成を理由に手動境界チェックを
省略しない」ことを求めるが、本方式は検査の**省略ではなく前倒し**（K ループ・書き戻しループへ入る前に 1
回だけ評価し、以後はそのビットを参照する）であり、境界外要素へアクセスしない契約自体は維持される。
クランプ済みアドレス計算（`gr_c`/`gc_c` によって境界外ポインタを作らない現行契約。415-425/427-437 行の
コメント参照）も変更しない。

### 懸念事項（切り分けが必須）

- 述語語（`row_valid_a`・`row_valid_b`・`epilogue_valid` 相当）がレジスタに常駐すると、`#493` の
  `WARP_TILES_M x WARP_TILES_N`（2x2）レジスタブロッキングと合わせてレジスタ圧迫要因になりうる
- スピルが起きると local memory アクセスが発生し、「改善なし」ではなく「性能後退」として TFLOPS に
  現れる。したがって **TFLOPS 比較の前に、必ずレジスタ使用量・local memory 使用量の base/head 差分を
  確認する**（`docs/perf/cuda-gemm-serpentine-ab.md` で確立した手順と同じ理由）

## 計測手順（DGX Spark GB10・sm_121 実機）

base（変更前）と head（変更後）それぞれについて計測し、5 回計測の中央値 TFLOPS を比較する
（`bench-harness::protocol::run` が中央値計測を実装済み。`coding-rust.md` 準拠。接続・転送手順は
`docs/real-hardware-verification-env.md` に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（変更前 = 本 PR の状態。境界検査は毎回比較のまま。origin/main 相当）
git checkout <base-sha>
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release > /tmp/gemm_mma_bench_base.txt

# head（本ドキュメントの head 案設計スケッチに従い実装した実験ブランチ）
git checkout <predicate-experiment-branch>
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release > /tmp/gemm_mma_bench_head.txt
```

出力形式（`crates/backend-cuda/examples/gemm_mma_bench.rs` 参照）の `MMA_F16` 経路（f16
`mma.sync`/`ldmatrix`/`cp.async`）の TFLOPS を base/head で突き合わせる。

数値一致確認（採否判断より前に必須。境界検査の前倒しのみで値そのものは変わらないはずという前提を
検証する）:

```sh
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture
```

`cpu_cuda_mma_parity`・`parity_nonregression`（tolerance pin テスト含む）等が green であること
（tolerance 定数〈`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`〉・`parity_baseline.rs` は変更しない）。

レジスタスピル確認（TFLOPS 比較の前に必須。上記「懸念事項」参照）:

```sh
# NVRTC の -Xptxas -v 相当（レジスタ使用量ログ）で base/head 間の register 数・
# local memory 使用量に差がないことを確認してから TFLOPS を比較する
```

## 判断基準

- base に対し head の中央値 TFLOPS が改善していれば「採用」とし、`kernels_mma.rs` へビットマスク事前
  計算コードを再実装する PR を起票する。採用 PR の受け入れ条件は以下を全て満たすこと:
  - parity 非後退契約（tolerance 定数〈`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`〉不変の機械
    確認・fail 比率/mean_abs_diff の非後退・FMA 契約維持）
  - `docs/perf/` への実測結果追記（本ドキュメントへの追記または新規ファイル）
  - 検査テスト（レジスタ・parity）を実装と同一 PR に含める
  - REQ-8 の境界検査そのものを維持する（前倒しであって省略ではないことをコードとコメントで示す）
  - ソース証跡テスト `mma_f16_source_retains_req8_boundary_guards`（763-778 行）を、事前計算後の
    コードに合わせて整合更新する（needle 文字列・境界検査の実在を機械検証する形を維持する）
- 改善が確認できなければ**採用しない**と判断し、その判断と実測値を本ドキュメントへ記録して本イシュー
  （#537）をクローズする
- **未計測の間は「採用済み」として扱わない**。本番カーネルへの変更導入は、上記いずれかの判断が実機
  計測をもって確定してから行う（暫定導入は行わない）

## 実測結果

（未計測。実機セッションで本節へ追記する）
