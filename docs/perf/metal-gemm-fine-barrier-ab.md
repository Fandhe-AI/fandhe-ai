# Metal GEMM simdgroup 細粒度同期（`simdgroup_barrier(mem_none)`）A/B 計測記録（#809）

イシュー #809「perf(backend-metal): simdgroup_barrier 細粒度同期とバッファプール要否の調査」の A/B 計測手順・記録テンプレート。
MLX steel `mma.h`（`docs/perf/metal-gemm-bottleneck-diagnosis.md`〈#487〉・`docs/backend-metal-mlx-classic-nax-decision.md`
〈#549〉で構成対比済み）が simdgroup フラグメントロード間に用いる `simdgroup_barrier(mem_flags::mem_none)`
（`threadgroup_barrier` より軽量な simdgroup スコープのフェンス）を、`gemm_simdgroup_tiled` の staged 経路 kk ループへ
適用した効果を計測する。

## 状態: M4 Max 実機実測完了・判定不可（イシュー #1278）

イシュー #1278 で M4 Max 実機（Apple Silicon）セッションでの実測を消化した。結論は**判定不可（undetermined）**:
フェーズ 1（安定性セルフチェック）が全 4 試行・全サイズで安定性ゲート（`STABILITY_SPREAD_GATE=0.05`）を満たさず、
フェーズ 2（A/B 判定）へ一度も到達しなかった（詳細は下記「実測記録」節・`docs/perf/logs/
metal-gemm-fine-barrier-ab-1278/`）。`crate::tile::FINE_BARRIER_ENABLED` は引き続き `false` のままであり、本番挙動
（`MetalGemm::new` の既定経路）に変更はない。

`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_metal_source_declares_fine_barrier_enabled_function_constant`・
`gemm_simdgroup_tiled_source_gates_fine_barrier_between_fragment_load_and_mma`（Linux CI・ubuntu-latest 上で実行）
により、`FINE_BARRIER_ENABLED` function constant（index 8）の宣言と、B フラグメントロード直後・MMA 発行直前という
挿入位置の契約は機械検査済み。数値契約（AC-1: base/head 出力のビット単位一致）は
`crates/backend-metal/tests/gemm_fine_barrier_bit_match.rs`（`dispatch_auto`・`dispatch_tiled_prepared` 両経路・
size 512/1024/2048/4096）で M4 Max 実機実測により確認済み（PASS。`docs/perf/logs/
metal-gemm-fine-barrier-ab-1278/bit_match_test.log`）。

## 適用箇所と数値契約

`shaders/gemm.metal` の `gemm_simdgroup_tiled`（本番既定経路。`dispatch_variant(GemmVariant::SimdgroupTiled(..))`）の
staged 経路（`USE_TGP_STAGING=true`）kk ループ内、A/B フラグメント一括ロード（`a_frag`/`b_frag`。イシュー #745 で
レジスタ常駐化済み）と MMA 発行（`simdgroup_multiply_accumulate`）の間に、`FINE_BARRIER_ENABLED` でゲートされた
`simdgroup_barrier(mem_flags::mem_none)` を挿入した。

- **direct-load（else 節）経路は適用対象外**: フラグメント再ロード構造が staged 経路と異なるため（実装計画 §2.1）。
- `tile_a`/`tile_b`（threadgroup メモリ）の内容自体は前段の `threadgroup_barrier(mem_flags::mem_threadgroup)`（協調
  ロード直後）で既に確定済みのため、この挿入は演算オペランド列（値・順序）を一切変えない。よって base（barrier
  なし）/head（barrier あり）の出力は理論上ビット単位で一致するはずである（#536・#538 と同じ論法）。
- `examples/gemm_fine_barrier_ab_bench.rs` のフェーズ 0 でこの数値契約を計測前に自己検証する（`assert_eq!` で
  `dispatch_auto` の出力を直接比較。size 256/512/1024/2048/4096。イシュー #1278 で 2048/4096 まで拡張済み）。
  独立の受け入れテスト（`tests/gemm_fine_barrier_bit_match.rs`）でも `dispatch_auto`・`dispatch_tiled_prepared`
  両経路を size 512/1024/2048/4096 で検証する（イシュー #1278 AC-1）。

## 計測手順（Apple Silicon 実機）

`bench_harness::ab::run_ab`（`docs/perf/metal-bench-noise-protocol.md` 参照）で base（`fine_barrier_enabled=false`）/
head（`fine_barrier_enabled=true`）を同一プロセス内・interleaved に計測する。

```sh
git fetch origin
git checkout main   # 本イシューの実装は main へマージ済み

# 実行前のサーマル状態を記録する（非特権コマンド。sudo 必須の powermetrics は不使用）。
pmset -g therm

cargo build --release -p fandhe-ai-backend-metal --example gemm_fine_barrier_ab_bench
./target/release/examples/gemm_fine_barrier_ab_bench > /tmp/gemm_fine_barrier_ab_bench.txt

# 実行後のサーマル状態も記録する。
pmset -g therm
```

**5 run 運用（イシュー #1278 で確立）**: 上記を 5 回（目標）繰り返し、size ごとの `head_over_base` の run 間中央値で
最終判断する（単発 run のばらつきに引きずられないため。`docs/perf/metal-bench-noise-protocol.md`・#1188 の 5 回計測
先例と同じ理由）。フェーズ 1 不成立（`verdict=undetermined`）の run は `ROUNDS`/`COOLDOWN`/`MIN_WARMUP`
（`examples/gemm_fine_barrier_ab_bench.rs` 冒頭の定数。増やす方向のみ調整し、一時変更はコミットしない）を調整して
再試行してよいが、総試行回数は 8 回を上限とし、それでも有効 run が 5 未満なら「判定不可」で確定する（#1278 の
実測を参照）。

`gemm_fine_barrier_ab_bench` は次の順で実行する:

1. **フェーズ 0（bit 一致自己検証）**: base/head の `dispatch_auto` 出力を size 256/512/1024/2048/4096 で直接比較する
   （イシュー #1278 で 2048/4096 まで拡張）。不一致の場合は `panic` し、フェーズ 1・2 へは進まない（安全側判断）。
2. **フェーズ 1（安定性セルフチェック）**: 本番既定（fine barrier off）の `MetalGemm::new` を使い、対照カーネルの
   spread が全サイズ（256〜4096）で `bench_harness::ab::STABILITY_SPREAD_GATE` 以下か確認する。いずれかのサイズが
   gate を超過した場合はフェーズ 2 へ進まず「判定不可」を出力して終了する（`docs/perf/metal-bench-noise-protocol.md`
   §「安定性ゲートと不成立時の中断規定」）。gate を満たさない場合は同ドキュメントの調整手順（`ROUNDS`/`COOLDOWN`/
   `MIN_WARMUP` を増やす方向のみ）に従い再実行する。
3. **フェーズ 2（prepared 境界 A/B）**: `MetalGemm::new_with_fine_barrier` で構築した base/head を
   `dispatch_tiled_prepared`（アップロード済みバッファ共有・確定 `TileConfig` 直接指定）で計測する。出力は
   size ∈ {256, 512, 1024, 2048, 4096} ごとに `base_median_tflops`・`head_median_tflops`・`head_over_base`
   （head/base 比）・`spread_base`・`spread_head`・base/head 双方の `resolved` タイル構成を含む。

数値一致確認（採否判断より前に必須。フェーズ 0 に加え、既存の parity テストでも再確認する）:

```sh
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`gemm_auto_parity` 等が green であること（tolerance は変更しない。
coding-rust.md）。

## 判断基準（イシュー #809 実装計画 §3.1）

- **採用**: size 2048/4096 の `head_over_base`（head/base の中央値 TFLOPS 比）に改善があり、かつ他サイズ
  （256〜1024）で劣化中央値 5% 超がない場合、`crates/backend-metal/src/tile.rs` の `FINE_BARRIER_ENABLED` を `true`
  へ変更してコミットし、`MetalGemm::new`（本番経路）の既定挙動で細粒度同期を有効化する（#784 の swizzle 結線と
  同型。このドキュメント・`tile.rs`/`gemm.rs`/`pipeline.rs`/`shaders/gemm.metal` の doc comment を実測結果込みで
  更新する。結線後は既定経路 parity テスト〈`gemm_dynamic_tile_parity` 等〉を再実行して非後退を確認する）
- **不採用**: 改善が確認できない場合、`FINE_BARRIER_ENABLED=false` のまま維持し、本ドキュメントへ実測結果と
  不採用の判断理由を記録する（機構自体は revert せず残す。実機未検証構成を安全に切り替え可能な状態に保つ設計
  判断は `SWIZZLE_ENABLED`〈#540〉と同型）。
- **判定不可**: フェーズ 1（安定性セルフチェック）が繰り返し不成立で、5 run 中央値の算出に足る有効 run が
  得られない場合。`FINE_BARRIER_ENABLED=false` のまま維持し、判定不可に至った経緯（試行回数・観測した spread・
  負荷状況）を本ドキュメントへ記録する。

判定統計・許容誤差・閾値自体は変更しない（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・
`docs/spec/04-requirements.md` REQ-8 の対象事項のためユーザー承認が必要。イシュー #809 の実装スコープには含まない）。

## 実測記録（イシュー #1278・2026-09-06・M4 Max 実機）

**結論: 判定不可（undetermined）**。フェーズ 1（安定性セルフチェック）が 4 試行・全サイズ（256〜4096）で
`STABILITY_SPREAD_GATE=0.05` を満たさず、フェーズ 2（A/B 本計測）へ一度も到達しなかった。5 run 中央値の算出に
必要な有効 run が 0 のため、`head_over_base` の実測値は存在しない。

| size | run1 spread（既定 ROUNDS=6） | run2 spread（ROUNDS=10） | run3 spread | run4 spread | gate |
|------|------|------|------|------|------|
| 256  | 0.2424 | 5.1231 | 0.0884 | 0.2350 | 0.05 |
| 512  | 0.3936 | 1.0048 | 0.5086 | 0.4714 | 0.05 |
| 1024 | 0.0618 | 0.7344 | 1.1225 | 0.7989 | 0.05 |
| 2048 | 0.3087 | 0.8708 | 0.9465 | 0.5286 | 0.05 |
| 4096 | 0.5697 | 0.0791 | 0.3011 | 0.2296 | 0.05 |

全 4 試行とも gate（0.05）を 2〜100 倍超過し、`verdict=undetermined` で終了した（`docs/perf/logs/
metal-gemm-fine-barrier-ab-1278/fine_barrier_ab_run{1..4}.log`）。size=256 の round_tflops はこの GPU の通常想定値
より 1 桁前後低い水準（0.02〜0.19 TFLOPS）で推移しており、計測自体が外部要因で圧迫されていたと判断した。

**原因分析**: 計測実行中の `uptime` load average は 3.47〜13.04 の範囲で推移し、30 分の待機上限を経ても
3.0 以下へ安定しなかった（`docs/perf/logs/metal-gemm-fine-barrier-ab-1278/env_info.txt`）。`ps aux` 確認により、
本イシューの worktree とは無関係な**別リポジトリ（fandhe-frontend 系）の `cargo build`／`cargo test` プロセスが
同一マシン上で複数並走**していることを確認した。これは #1186／#1187（`docs/perf/logs/
metal-gemm-transpose-route-ab-{1186,1187}/`）が観測した「兄弟 Metal ベンチ issue との GPU 負荷競合」よりも広く、
このマシンを共有する他リポジトリのセッション全般からの CPU 負荷が常態化している状況と判断した。
`ROUNDS`/`COOLDOWN`/`MIN_WARMUP` を増やす方向へ調整（#1187 と同じ値: 10/8s/3s）しても spread は改善せず、
むしろ run2 の size=256 では 1 round が 0.0231 TFLOPS まで落ち込む極端な外れ値が生じた。試行回数 4（実装計画の
上限 8 に対し、#1187 と同水準の試行数で一貫して同じ傾向が再現したため、これ以上の調整では収束しないと判断し
打ち切った）。

サーマル記録（`pmset -g therm` 実行前後）: 全試行で「No thermal warning level has been recorded」（サーマル
スロットリングは観測されず、原因はサーマルではなく外部プロセスの CPU 負荷競合と判断した）。

数値契約（AC-1）は別途 PASS 済み: `tests/gemm_fine_barrier_bit_match.rs` の 2 テスト
（`fine_barrier_on_off_bit_match_dispatch_auto`・`fine_barrier_on_off_bit_match_dispatch_tiled_prepared`）が
size 512/1024/2048/4096 で base/head 出力のビット単位一致を確認した（`docs/perf/logs/
metal-gemm-fine-barrier-ab-1278/bit_match_test.log`）。既存 parity テスト（`gemm_dynamic_tile_parity`・
`cpu_metal_parity`・`gemm_auto_parity` 等。`docs/perf/logs/metal-gemm-fine-barrier-ab-1278/
parity_ignored_tests.log`）も非後退で green（tolerance 不変）。

環境情報・実効パラメータ・load average 推移の全詳細は `docs/perf/logs/metal-gemm-fine-barrier-ab-1278/env_info.txt`
を参照。

## 引き継ぎ（#1280 へ）

- **採否結果**: 判定不可（上記「実測記録」節）。`FINE_BARRIER_ENABLED=false` を維持し、本番結線は行わない。
- **`tile.rs` doc comment との不整合**: `crates/backend-metal/src/tile.rs` の `FINE_BARRIER_ENABLED` doc comment は
  「不採用時: 本機構一式を revert」と書いているが、本ドキュメントの判断基準は「不採用（および判定不可）でも
  機構は残す（`SWIZZLE_ENABLED`〈#540〉と同型）」としている。本イシュー（#1278）は計測・記録のみのスコープの
  ため `tile.rs` は変更しておらず、この不整合は未解消のまま残る。#1280 で結線判断と併せて整合させること
  （`tile.rs` の doc comment 更新が必要）。
- **`ROUNDS`/`COOLDOWN`/`MIN_WARMUP` の恒久値化**: 本イシューでは一時ローカル変更（10/8s/3s）で足り、コードの
  既定値（6/2s/1s）は変更していない。恒久的に増やすべきかは、このマシンの他リポジトリからの負荷競合が本イシュー
  固有の一過性要因か否かの見極めが必要なため、#1280 の判断に委ねる。
- **再計測の要否**: 判定不可のまま `FINE_BARRIER_ENABLED=false` を維持する運用は #795・#1037 等の先例と同型で
  問題ないが、負荷の少ない時間帯での再計測により結論が変わる可能性は残る（本イシューのスコープ外）。

## 参照

- `crates/backend-metal/src/shaders/gemm.metal`（`FINE_BARRIER_ENABLED` function constant 宣言・staged 経路挿入箇所）
- `crates/backend-metal/src/pipeline.rs::make_pipeline_with_constants`（index 8 特殊化）
- `crates/backend-metal/src/tile.rs::FINE_BARRIER_ENABLED`（本番既定値。コミット状態既定 `false` ロックテスト
  `fine_barrier_enabled_is_false_by_default`）
- `crates/backend-metal/src/gemm.rs::MetalGemm::new_with_fine_barrier`（A/B 計測専用入口）
- `crates/backend-metal/examples/gemm_fine_barrier_ab_bench.rs`（A/B ベンチ本体。イシュー #1278 でフェーズ 0 の
  size 拡張・`verdict=` 出力・実効パラメータ println を追加）
- `crates/backend-metal/tests/gemm_fine_barrier_bit_match.rs`（AC-1 の独立受け入れテスト。イシュー #1278 で新規追加）
- `crates/backend-metal/tests/shader_source_evidence.rs`（Linux CI 上のシェーダ証跡機械検査）
- `docs/backend-metal-buffer-pool-decision.md`（同イシュー #809 のバッファプール要否調査。本ドキュメントとは
  独立の判断対象）
- `docs/perf/logs/metal-gemm-fine-barrier-ab-1278/`（イシュー #1278 の実測ログ・env_info）
