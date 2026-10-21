# CUDA GEMM 境界検査ビットマスク事前計算 A/B 計測記録（#537）

イシュー #537「perf(backend-cuda): 境界検査のビットマスク事前計算化の要否を実機で評価」の A/B 計測手順・
head 案設計スケッチ・判断基準・記録テンプレート。`crates/backend-cuda/src/kernels_mma.rs` の `MMA_F16`
カーネル（f16 `mma.sync`/`ldmatrix`/`cp.async` 経路）が A/B タイルロード時・エピローグ書き戻し時に毎回
評価している境界比較を、CUTLASS `predicated_tile_access_iterator.h` の `compute_predicates_` と同型の
「スレッドごとに 1 回だけビットマスクへ事前計算し、以後はビット参照のみで判定する」方式へ置き換える価値
があるかを評価する。

## 状態: 実測済み（イシュー #3190 再開分・2026-10-21）。判定は「不採用」（本番カーネルは一切変更していない）

`logs/optin-ab-gb10-3190/on-boundary-predicate.patch`（head 腕。計測専用で main へは入れない）を実装し、GB10 で前提ゲート (a)(b) と
5 プロセスの TFLOPS A/B を事前登録規則（`RULE.txt` 8・`RULE-reopen.txt`・`-2`・`-3`）どおりに実施した。結果は末尾「実測結果」節。
本ドキュメントの設計スケッチ・手順は、実装時に実際の `kernels_mma.rs`（`LOAD_*_STAGE_GROUP` 方式）に合わせて具体化した。
前例（#497 の蛇行走査・PR #657 の codex-review 指摘）と同じ方針で、改善が確認できない案は本番へ導入しない。

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

| 箇所 | ファイル・行（base = `230937f5f`。#3190 再開分で再確認） | 内容 |
|------|------------|------|
| A タイルロード | `crates/backend-cuda/src/kernels_mma.rs:1330-1342`（`LOAD_A_STAGE_GROUP` マクロ） | 8 要素チャンクごとに `int valid = (gr < DIM_M && gc < DIM_K) ? 16 : 0;` を毎回評価し `mma_cp_async16` の src-size に渡す。`gr_c`/`gc_c` のクランプ計算が先行 |
| B タイルロード | 同 `1344-1356`（`LOAD_B_STAGE_GROUP` マクロ） | 同様に `int valid = (gr < DIM_K && gc < DIM_N) ? 16 : 0;` |
| エピローグ guarded store | 同 `1680-1705` | `WARP_TILES_M x WARP_TILES_N`（2x2）の mi/nj ループ内で `r0/r1 < DIM_M && c0/c1 < DIM_N` の 4 条件を評価し、`__half2` ペア store（#805）または c0 単独のスカラー store へ分岐 |
| ソース証跡テスト | 同 `2874`（`mma_f16_source_retains_req8_boundary_guards`）ほか、特化 render の 2 テスト | 6 つの式文字列を needle として `contains` をロック。ペア store は別テスト（#805）がロック |

（旧版の行番号 `415-425` 等は `LOAD_A_STAGE`/`LOAD_B_STAGE` 時代の値で、現在は上表。`m`/`n`/`k` 直書きは `DIM_M`/`DIM_N`/`DIM_K` マクロ経由へ変わっている。）

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

イシュー #3190（再開分）。GB10（sm_121・CUDA 13.0・20 コア）。生ログと集計は `logs/optin-ab-gb10-3190/b5/`（`aggregate.md` が `aggregate_b5.py` の出力）。
ツリーは `b5/rev-stamp.txt`（base = `230937f5f`、head = base + `on-boundary-predicate.patch` を未コミットで適用。
`230937f5f` から `origin/main`（`ef3b2d715`）の間の `backend-cuda` の `src`・`examples` の差分は 0 行）。

### head 案（`on-boundary-predicate.patch`）の要点

- A の行述語 `a_gr < DIM_M` と B の列述語 `b_gc < DIM_N` は k0 に依存しないので、グループ g ごとに入口で 1 回評価し、ビット（`a_row_ok`／`b_col_ok` の第 g ビット）に詰める。
  K 方向の述語は各 stage で評価して AND する（`row_ok && gc < DIM_K`／`col_ok && gr < DIM_K`）。`gr_c`/`gc_c` のクランプ計算と範囲外ゼロ充填は不変。
  ループが 2 周目以降に入る構成（`blockDim.x` が小さい非既定 config）では、ビットを使わず従来の比較（`gr < DIM_M`／`gc < DIM_N`）へ落とす（`idx == g * *_GROUP_CHUNKS + tid` の判定）。
- エピローグは入口で 16 ビットの `epilogue_valid` を 1 回計算し（ビット `(mi*WARP_TILES_N+nj)*4 + {0..3}` = `[r0&c1, r0&c0, r1&c1, r1&c0]`）、書き戻しはビット参照のみ。`__half2` ペア store と c0 単独フォールバックの構造は不変。
- ソース証跡テストの needle: 述語を作る式（`a_gr < DIM_M`・`b_gc < DIM_N`・`row_ok && gc < DIM_K`・`col_ok && gr < DIM_K`・`r0 < DIM_M && c0 < DIM_N` 等のエピローグ 4 式）と、
  ビット参照式（`((a_row_ok >> (g)) & 1u)`・`((b_col_ok >> (g)) & 1u)`・`epilogue_valid & (1u << (ep_shift + {0..3}u))`）の両方を `contains` でロックする（3 テストの needle を更新）。
  検査は省略でなく前倒しであることを機械的に検証する形を保った。REQ-8 の境界検査を省く変種は作っていない。
- Mac での確認: `cargo test -p fandhe-ai-backend-cuda --lib kernels_mma`（NVRTC 不要の範囲）が通る。NVRTC のコンパイルは GB10 で確認した（下記 (a) の lib テストと `mma_ptx_dump`）。

### 前提ゲート (a) parity 非後退（`parity-{base,head}.log`）

`cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics --no-fail-fast --test cpu_cuda_mma_parity --test gemm_mma --test parity_nonregression --test specialized_mma_parity --test gemm_auto --test dispatch_boundary -- --ignored --test-threads=1`
と `--lib kernels_mma` を base・head の両方で実行した。**成立**: テスト 111 件（head ok 109・FAILED 1・ignored 1）。base ok → head FAILED の後退は 0 件。

| テスト | base | head | 備考 |
|---|---|---|---|
| `cpu_cuda_mma_parity::mma_f16_k4096_stress` | FAILED | FAILED | **base でも FAILED**（head の後退ではない。別の `..._non_regression` は ok） |
| `kernels_mma::...::mma_f16_sources_compile_with_nvrtc_when_available` | ignored | ignored | 理由「手動実行」。非対象 |

作業の開示: (a) のコマンドは 2 回訂正した（初版は `internal-diagnostics` 指定漏れで integration が実行されず、次は最初の失敗バイナリで cargo が停止した）。
旧ログは `parity-attempt1-*.log`・`parity-attempt2-*.log` として残してあり、訂正は再実行前にコミットした（`RULE-reopen-2.txt`・`RULE-reopen-3.txt`）。
判定式は変えていない。

### 前提ゲート (b) レジスタ・local memory（`ptxas-{base,head}.log`。`ptxas -arch=sm_121 -v`、本番構成 `mma_f16_base.ptx`）

| 腕 | registers | stack frame | spill stores／loads | lmem |
|---|---|---|---|---|
| base | 58 | 0 B | 0／0 B | 0 B |
| head | 66 | 0 B | 0／0 B | 0 B |

**成立**（lmem・stack は増えず、spill は 0）。registers は +8（記録のみで判定に使わない）。

### TFLOPS（`gemm_mma_bench` の `mma_f16_tflops`。5 プロセス独立・round 内の順序反転・専有ゲートは全 10 run PASS）

| size | run 別 base（TFLOPS） | run 別 head（TFLOPS） | base 中央値 | head 中央値 | head/base | 5/5 改善 | 5/5 後退 |
|---:|---|---|---:|---:|---:|:-:|:-:|
| 512 | 16.6606 / 16.7772 / 16.9638 / 16.9982 / 17.0500 | 16.0548 / 16.1475 / 16.3044 / 16.1010 / 16.2099 | 16.9638 | 16.1475 | 0.9519 | - | ○ |
| 1024 | 38.3041 / 38.5240 / 38.4028 / 38.5794 / 38.6127 | 31.8278 / 31.9490 / 32.0558 / 32.0176 / 31.9186 | 38.5240 | 31.9490 | 0.8293 | - | ○ |
| 2048 | 52.3647 / 52.3673 / 52.4056 / 52.3366 / 52.3928 | 40.3964 / 40.2859 / 38.1205 / 40.1652 / 40.1142 | 52.3673 | 40.1652 | 0.7670 | - | ○ |
| 4096 | 55.9432 / 56.1344 / 56.0882 / 55.8402 / 55.7558 | 40.8075 / 40.9511 / 40.6860 / 40.7247 / 41.0863 | 55.9432 | 40.8075 | 0.7294 | - | ○ |

round 番号と起動順は `b5/gate.log`（奇数 round は base→head、偶数 round は head→base）と `run{1..5}-{base,head}.log`。

### 判定: **不採用**（RULE.txt 8: いずれかの形状で 5/5 run 一貫して後退）

4 形状すべてで 5/5 後退（head/base は 0.73〜0.95、大きい形状ほど悪化）。採用候補の条件（全形状で head の中央値 ≥ base）も満たさない。
本番の `MMA_F16` は変更せず、#537 は「境界述語の事前計算（この形）は採らない」という実測つきの判断で閉じられる。
考えられる要因（**推定。本計測では切り分けていない**）: registers が 58 → 66 に増え、`__launch_bounds__`／occupancy 側の制約を受けた可能性、
および `cp.async` 発行直前の述語の依存関係が長くなった可能性。切り分けるなら、述語ビットを一部だけ（例: エピローグのみ）前倒しする別案を新規に事前登録して測る。
本計測では head 案の再調整（事後の変更）はしない。

