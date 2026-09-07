# CPU GEMM `IcDynamic`（ic 限定動的配布）variant の実装記録

イシュー #1366（親: #1365。兄弟: #1367「両実機実測・採否判定」）。

**採否（2026-09-07 追補・#1367）**: 両実機実測の結果 **REJECT（不採用）**。DGX Spark GB10 で N=1024/2048 の対 RowPanel 比が 0.6263／0.8492 と大きく後退し、Apple M4 Max も N=1024 が僅かに未達（0.9878）のため採用ゲートを満たさない。本番結線は行わない。詳細は `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §14 を参照。

## 1. 目的・位置づけ

`crates/backend-cpu/src/gemm_blis/mod.rs` の並列 GEMM 本番既定 `RowPanel`
（`gemm_blis_parallel_with_transpose`）は C を `m.div_ceil(num_threads)` 行の
静的パネルへ等分割し、各 rayon タスクが独立に `dispatch_region`（jc→pc→ic→jr→ir）
を回す。既存候補 `SharedB`（#750）・`SharedBPcOuter`（#1041）は B（および A）の
重複 pack を解消したが、両実機実測（#1140／#1141／#1144）で `RowPanel` を大きく
下回り非採用が確定した。`SharedBPcOuter` はさらに行パネルを
`mc_total.div_ceil(num_workers)` で **静的に** 等分割するため、MC タイル数が
ワーカー数で割り切れない形状や異種コア環境（`docs/perf/cpu-gemm-default-thread-limit.md`・
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §13 の DGX Spark GB10 非一様
コア構成の実測）では負荷不均衡が生じうる。

本 issue は行パネルの配布を `AtomicUsize` カウンタによる動的配布へ変更する
だけでなく、B の列ブロッキングも変更した新 variant
`GemmDriverVariant::IcDynamic`（`gemm_blis_ic_dynamic_region`）を `#[cfg(test)]`
限定で追加し、`RowPanel` との bit 完全一致を回帰テストで確認する。
`SharedBPcOuter` は B を (pc,jc) ごとに `blocks.nc` 幅で pack し jc ループで
列を順に処理するのに対し、`IcDynamic` は jc ループを持たず `blocks.nc` を
使わずに pc ごとへ列全幅 `n` を 1 回で pack する。この違いにより B バッファの
メモリ使用量（`nc` 幅 → 列全幅 `n`。形状によっては増大する）・キャッシュ
局所性・pc ごとの同期点の数（jc ブロック数ぶん → 1 回）も変わる（§2.1 参照）。
**本番結線・採否判定は行わない**（両実機実測は #1367 へ引き継ぐ）。

## 2. 設計

### 2.1 データフロー

```
gemm_blis_parallel_variant(IcDynamic, a, b, c, m, n, k, blocks)
  └ dispatch_ic_dynamic(a, b, c, n, k, 0..m, blocks)   // arch 別 ISA トークン確定
      └ gemm_blis_ic_dynamic_region<K>(kernel, a, b, c, n, k_dim, rows, blocks)
          num_workers = effective_num_threads(rayon::current_num_threads())
          panel_rows  = ic_dynamic_panel_rows(mc_total, blocks.mc, K::MR, num_workers)
          num_panels  = mc_total.div_ceil(panel_rows)
          b_panel_buf: Vec<f32>（ic_dynamic_b_capacity(n, kc_max, K::NR) 容量）
          a_bufs: Vec<Vec<f32>>（num_workers 本 × task_a_capacity(panel_rows, kc_max, K::MR) 容量）
          for pc in (0..k_dim).step_by(blocks.kc):        // 最外・逐次
              kc_len = blocks.kc.min(k_dim - pc)
              b_panel = pack_b を列全幅 n ぶん 1 回（par_chunks_mut で nr ブロック並列）
              slots: Vec<Mutex<Option<&mut [f32]>>> = c.chunks_mut(panel_rows * n).map(Mutex::new).collect()
              counter = AtomicUsize::new(0)
              a_bufs.par_iter_mut().try_for_each(|a_buf| loop {
                  idx = counter.fetch_add(1, Relaxed)
                  if idx >= num_panels { return Ok(()) }
                  c_panel = slots[idx].lock().take()   // 高々 1 回・無競合
                  pack_a(...) を c_panel の行範囲ぶん 1 回
                  gemm_blis_jr_ir_loop(kernel, c_panel, ..., &ctx)?
              })?
```

### 2.2 動的配布の機構（`unsafe` 非導入）

C を `panel_rows` 行ずつのパネルへ `chunks_mut` で分割し、各パネルの
`&mut [f32]` を `Mutex<Option<&mut [f32]>>` スロットへ 1 個ずつ格納する。
worker は `AtomicUsize::fetch_add(1, Ordering::Relaxed)` で担当パネル index を
確定してから、対応するスロットを `lock().take()` して `&mut` を取り出す。
index はパネル数を上限に単調増加し、各 index は高々 1 worker しか claim
しないため、同じスロットへ 2 worker が競合することは構築上発生しない
（1 パネル 1 回のロックで常に無競合）。`&mut` パネル間の排他性はコンパイル時の
借用検査で保証されており、`unsafe` は不要（issue #1366 のスコープ「`unsafe` を
新規導入しない」を満たす）。

### 2.3 bit 完全一致契約（REQ-2）を保つ根拠

`gemm_blis_shared_b_pc_outer_region` と同じ論法がそのまま成り立つ: pc は
本 variant でも外側で昇順に回り、C の各要素は (pc, 行パネル) の組で見て一意な
1 つの行パネルにのみ属する。行パネルをどの worker が・どの順序で claim するかは
互いに素な C 要素集合の処理順序を並び替えるだけで、同一要素の pc 昇順・カーネル内
p 昇順の蓄積順序には影響しない。

### 2.4 行パネル粒度・B footprint

`ic_dynamic_panel_rows(mc_total, mc, mr, num_workers)` は
`mc_total.div_ceil(num_workers)` を `mr` の倍数へ切り上げたうえで `mc` を
超えないようクランプし、`SharedBPcOuter` と異なりパネル数がワーカー数以上に
なるよう `blocks.mc` の上限も同時に満たす（中形状でパネル数 < ワーカー数となり
動的配布の意味がなくなるのを防ぐ）。

B バッファは pc ごとに列全幅 `n` を 1 回だけ pack するため、footprint は
`n.div_ceil(NR) * KC * NR` で N=4096・KC=256（既定値）のとき約 4 MiB。
N が極端に大きい場合（例: 65536）は約 64 MiB になりうる点を留意する
（jc ループを持たないため `blocks.nc` による分割はできない設計上の制約）。

## 3. 回帰テスト（`crates/backend-cpu/src/gemm_blis/mod.rs`・`mod tests`）

| テスト | 内容 |
|---|---|
| `gemm_blis_parallel_variant_all_candidates_match_naive_bit_exact` | 全候補（`RowPanel`・`SharedB`・`SharedBPcOuter`・`IcDynamic`）が 5 形状 × スレッド数 1/2/3/16 で `gemm_naive` と bit 完全一致 |
| `gemm_blis_ic_dynamic_multi_pc_sync_point_matches_serial_bit_exact` | `mc=16/kc=17/nc=19`・(200,600,700)・4 スレッドで直列実装と bit 完全一致（小 kc で pc 同期点を複数強制） |
| `gemm_blis_ic_dynamic_matches_naive_bit_exact_when_tasks_fewer_than_threads` | (10,130,40)・16 スレッドでパネル数 < ワーカー数のケース |
| `gemm_blis_ic_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads` | 8 形状（端タイル・非整列 n/k・k=0 no-op・512³ を含む）× スレッド数 1/2/3/16 で `RowPanel` と bit 完全一致（issue 表題の直接検証。C 初期値を非ゼロ乱数にして累積の意味論も確認） |
| `gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large`（`#[ignore]`） | 1024/2048/4096 正方・release ビルドでの `RowPanel` との bit 完全一致（プール既定スレッド数を使う唯一のテストのため `effective_num_threads`＋`ic_dynamic_panel_rows` の現実的なワーカー数での組み合わせを検証する。本エージェント実行環境〈macOS aarch64〉で 1 回 pass 確認済み。#1367 の実機実測前の事前確認用） |
| `ic_dynamic_panel_rows_bounds_and_alignment` | 純関数契約（下限 1・mc 以下・mr 倍数整列・クランプ挙動） |
| `ic_dynamic_b_capacity_matches_formula_and_detects_overflow` | 容量式の代表値一致・`usize::MAX` 近傍でのオーバーフロー検出 |

`gemm_blis_variant_ab_1024_2048`／`gemm_blis_variant_ab_4096`（既存 A/B 一括計測
ハーネス。`#[ignore]`）の `variants` 配列にも `IcDynamic` を追加済みのため、
#1367 はコード変更なしで両実機実測を実行できる。

## 4. 検証コマンド

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p fandhe-ai-backend-cpu --all-features
cargo check -p fandhe-ai-backend-cpu --tests --all-features --target x86_64-unknown-linux-gnu
make check-cross-cpu-tests   # aarch64-apple-darwin 型検査
```

いずれも green（`-p fandhe-ai-backend-cpu` 単体スコープでの `cargo clippy` は
dev-dependency `fandhe-ai-backend-cuda` 由来の dead-code 誤検出〈`-p` スコープ限定の
既知の環境依存事象。HEAD 変更前から再現し本変更と無関係〉が起きるが、
`cargo clippy --workspace --all-targets --all-features -- -D warnings`〈CI が
実際に実行する形〉では発生しない）。

## 5. スモーク実行（採否根拠にしない）

```
cargo test -p fandhe-ai-backend-cpu --release -- --ignored gemm_blis_variant_ab_1024_2048 --nocapture
```

上記を本エージェント実行環境（macOS・aarch64・他セッション並走の共有負荷下）で 1 回
実行し、`IcDynamic` の行が出力されることのみを確認した（動作確認目的。値は本 PR の
採用根拠にしない・#1367 の実機実測を待つ）。

## 6. 実機実測（実施済み・#1367。2026-09-07）

5 回独立プロセス中央値（GFLOP/s）。

| 実機 | N=1024 | N=2048 | N=4096 | IcDynamic/RowPanel(1024/2048/4096) | 実施日 |
|---|---|---|---|---|---|
| Apple M4 Max | RowPanel 635.868 / IcDynamic 628.141 | RowPanel 736.337 / IcDynamic 742.778 | RowPanel 812.951 / IcDynamic 902.014 | 0.9878 / 1.0087 / 1.1096 | 2026-09-07 |
| DGX Spark GB10 | RowPanel 530.338 / IcDynamic 332.159 | RowPanel 699.913 / IcDynamic 594.350 | RowPanel 1136.666 / IcDynamic 1111.147 | 0.6263 / 0.8492 / 0.9775 | 2026-09-07 |

**採否: REJECT（不採用）**。両実機で採用ゲート（N=1024/2048 で ratio≥1.00・
N=4096 で ratio≥0.95・両実機一致）を満たさない。DGX の N=1024/2048 が
特に大きく後退している。判定根拠・原因推定・生ログは
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §14・
`docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/` を参照。

## 出典

- イシュー #1366（本ドキュメントの起票元）・#1365（親）・#1367（兄弟・両実機実測）
- #565／#750（B 共有 案 B）・#1041／#1144（`SharedBPcOuter`。実機非採用確定）
- #1307／#1310／#1311（2D 動的分配。本 issue は ic 限定に絞ったスコープ）
- `crates/backend-cpu/src/gemm_blis/mod.rs`
- `docs/cpu-gemm-b-packing-sharing-decision.md`（§F 追記）
- `docs/perf/cpu-gemm-candle-cpu-retune.md`（既存候補の実機実測記録・運用方針）
- `.claude/rules/coding-rust.md`（unsafe 最小化方針・bit 完全一致契約）
- `.claude/rules/security.md`（unsafe レビュー体制・OWASP 観点）
