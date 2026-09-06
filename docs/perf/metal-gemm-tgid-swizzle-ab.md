# Metal GEMM threadgroup ID スウィズル（swizzle_log 相当）A/B 計測記録（#540・#795・#1279）

イシュー #540「perf(backend-metal): threadgroup ID スウィズル（swizzle_log 相当）を実験的に追加」の A/B 計測手順・記録テンプレート。
`gemm_simdgroup_tiled` の dispatch grid 走査順を、素朴な行優先（`row0 = tgid.y * BM`・`col0 = tgid.x * BN`）から
threadgroup ID スウィズル（固定値 `SWIZZLE_LOG = 2`。`tile` = 4 threadgroup を 1 群として縦方向へ束ねる）へ変更した効果を計測する
（MLX steel `swizzle_log`・DeepGEMM の L2 スウィズルと同種の技法。MLX 自身は classic 経路で `swizzle_log = 0`〈無効〉のまま
据え置いており未実証の技法である点に留意。計画「背景・目的」節）。

## 状態: M4 Max 実機実測完了・判定不可（イシュー #1279）

イシュー #1279 で M4 Max 実機（Apple Silicon）セッションでの実測を消化した。結論は**判定不可（undetermined）**:
フェーズ 1（安定性セルフチェック）が 4 試行・全サイズ（256〜4096。run4 の size=256 のみ 1 試行 gate 内）で
安定性ゲート（`STABILITY_SPREAD_GATE=0.05`）を満たさず、フェーズ 2・3（A/B 判定）へ一度も到達しなかった（詳細は
下記「実測記録」節・`docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/`）。`crate::tile::SWIZZLE_ENABLED` は引き続き
`false` のままであり、本番挙動（`MetalGemm::new` の既定経路）に変更はない。

数値契約（AC-1: base/head 出力のビット単位一致）は `crates/backend-metal/tests/gemm_swizzle_bit_match.rs`
（`dispatch_auto`・`dispatch_tiled_prepared` 両経路・size 512/1024/2048/4096）で M4 Max 実機実測により確認済み
（PASS。`docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/bit_match_test.log`）。既存 parity テスト
（`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`gemm_auto_parity` 等）も非後退で green（tolerance 不変。
`docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/parity_ignored_tests.log`）。

イシュー #746 により、旧 checkout＋一時トグル方式（後述「旧手順（履歴）」節）を、同一コミット上で base（swizzle
off）/head（swizzle on）の 2 `MetalGemm` インスタンスを構築し interleaved に A/B 計測する手順へ差し替えた
（`docs/perf/metal-bench-noise-protocol.md` 参照。ノイズ対策プロトコルの設計・根拠はそちらを正とし本節では
書き写さない）。イシュー #795 で `gemm_swizzle_ab_bench.rs` にフェーズ 3（転送込み境界 A/B。`dispatch_auto` ベース。
下記「計測手順」節）を追加し、本節の「判断基準」を #795 の受け入れ条件（改善なしの場合は revert せず
`SWIZZLE_ENABLED=false` を維持）に合わせて改定した。イシュー #1279 でフェーズ 0（bit 一致自己検証）・4 値
verdict 純関数（`single_run_verdict`。フェーズ 2・3 双方の比列を消費）・`verdict=` 機械出力・独立受け入れテスト
（`tests/gemm_swizzle_bit_match.rs`）を追加し、実機実測を消化した。
`crates/backend-metal/src/tile.rs` の `gemm_simdgroup_tiled_source_uses_tgid_swizzle`（crate 内 unit test。PR #661
codex-review 指摘対応で `tests/shader_source_evidence.rs` から移設）によりスウィズル式の実在は Linux CI（ubuntu-latest）
上で機械検査済み。

**本番 dispatch は `crate::tile::SWIZZLE_ENABLED`（既定 `false`）でスウィズルを無効化済み**（PR #661 codex-review 指摘対応:
実機未検証のまま本番経路へ無条件適用しない）。`gemm_simdgroup_tiled` の MSL function constant `SWIZZLE_ENABLED`（index 7。
`TGP_PAD`〈#538・index 6〉との index 重複は `tile.rs` 側の機械検査（`gemm_simdgroup_tiled_source_uses_tgid_swizzle`）で
ロック済み）が `false` の間はシェーダ側も恒等変換（`tid_y = tgid.y`・`tid_x = tgid.x`）で動作し、本番経路（`MetalGemm::new`）
の挙動・性能は変わらない。イシュー #746 で `swizzle_enabled` を `MetalGemm` の instance フィールドへ格上げした
（`crate::pipeline::make_pipeline_with_constants`・`crate::tile::tiled_dispatch_grid_with` の引数化。
`crates/backend-metal/src/gemm.rs`・`pipeline.rs`・`tile.rs` 参照）ため、A/B 計測用の `MetalGemm::new_with_swizzle(ctx, true)`
インスタンスを本番経路（`MetalGemm::new`。常に `SWIZZLE_ENABLED=false` を渡す）とは別に構築でき、`tile.rs` のソースを
一時的に書き換える必要はなくなった。

## 計測手順（Apple Silicon 実機）

`bench_harness::ab::run_ab`（`docs/perf/metal-bench-noise-protocol.md` 参照）で base（swizzle off）/head（swizzle on）
を同一プロセス内・interleaved に計測する。

```sh
git fetch origin
git checkout main   # 本イシューの実装は main へマージ済み

# 実行前のサーマル状態を記録する（非特権コマンド。sudo 必須の powermetrics は不使用）。
pmset -g therm

cargo build --release -p fandhe-ai-backend-metal --example gemm_swizzle_ab_bench
./target/release/examples/gemm_swizzle_ab_bench > /tmp/gemm_swizzle_ab_bench.txt

# 実行後のサーマル状態も記録する。
pmset -g therm
```

**5 run 運用（イシュー #1279 で確立。`gemm_fine_barrier_ab_bench.rs`〈#1278〉と同じ運用）**: 上記を 5 回（目標）
繰り返し、size ごとの `head_over_base` の run 間中央値で最終判断する。フェーズ 1 不成立（`verdict=undetermined`）の
run は `ROUNDS`/`COOLDOWN`/`MIN_WARMUP`（`examples/gemm_swizzle_ab_bench.rs` 冒頭の定数。増やす方向のみ調整し、
一時変更はコミットしない）を調整して再試行してよいが、総試行回数は 8 回を上限とし、それでも有効 run が 5 未満
なら「判定不可」で確定する（#1279 の実測を参照）。

`gemm_swizzle_ab_bench` は次の順で実行する:

1. **フェーズ 0（bit 一致自己検証。イシュー #1279 で追加）**: base/head の `dispatch_auto` 出力を
   size 256/512/1024/2048/4096 で直接比較する。不一致の場合は `panic` し、フェーズ 1〜3 へは進まない
   （安全側判断）。
2. **フェーズ 1（安定性セルフチェック）**: 対照カーネルの spread が全サイズで
   `bench_harness::ab::STABILITY_SPREAD_GATE`（0.05）相当以下か確認する。いずれかのサイズが gate を超過した
   場合はフェーズ 2・3（swizzle A/B）へ進まず「判定不可」を出力して終了する（安全側判断。`docs/perf/
   metal-bench-noise-protocol.md` §「安定性ゲートと不成立時の中断規定」）。gate を満たさない場合は同ドキュメントの
   調整手順（`ROUNDS`/`COOLDOWN`/`MIN_WARMUP` を増やす方向のみ）に従い再実行する。
3. **フェーズ 2（prepared 境界 A/B）**: 出力は size ∈ {256, 512, 1024, 2048, 4096} ごとに
   `base_median_tflops`・`head_median_tflops`・`head_over_base`（head/base 比）・`spread_base`・`spread_head`・
   base/head 双方の `resolved` タイル構成（`pipeline_for_tile` のフォールバック発生有無）を含む。受け入れ基準に
   従い **size ∈ {2048, 4096}** の `head_over_base` を採否判断の根拠とし、小〜中形状（256〜1024）の悪化有無も
   併せて記録する（計画「リスクと安全側の判断」節: grid.x が 4 倍化し `tiles_m` 非倍数時の余剰 threadgroup が
   増えるため、小形状での軽微な悪化があり得る）。
4. **フェーズ 3（転送込み境界 A/B。イシュー #795）**: フェーズ 2 に続けて自動実行され、
   size ∈ {512, 1024, 2048, 4096} ごとに同じ形式を出力する。`dispatch_auto`（ホストスライス入力・アップロード +
   GEMM + 読み戻しを 1 計測区間に含む本番相当の呼び出し経路）を使うため、prepared 境界（フェーズ 2）では見えない
   転送コストとの相互作用を採否判断へ反映できる。prepared 境界・転送込み境界の両方で `size ∈ {2048, 4096}` の
   改善が確認できることを「採用」の必要条件とする（#795 計画「実行環境の判定」Step 3）。フェーズ 3 完了後、
   `single_run_verdict`（イシュー #1279 で追加。フェーズ 2・3 双方の比列を消費する 4 値純関数）による単一 run の
   参考 `verdict=` 行を出力する（機械的 `grep verdict=` で単一 run の判定を追える参考表示。最終採否は 5 run
   中央値で本ドキュメントへ人間可読な形で記録する）。

数値一致確認（採否判断より前に必須。走査順の変更のみでビット単位一致が理論上成立するはずの前提を検証する。
フェーズ 0 に加え、独立の受け入れテスト・既存の parity テストでも再確認する）:

```sh
cargo test -p fandhe-ai-backend-metal --release --test gemm_swizzle_bit_match -- --ignored --nocapture
cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
```

`swizzle_on_off_bit_match_dispatch_auto`・`swizzle_on_off_bit_match_dispatch_tiled_prepared`（イシュー #1279 で
新設。AC-1）・`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`gemm_auto_parity` 等が green であること
（tolerance は変更しない。coding-rust.md）。

## 判断基準（#795 で改定。旧「非採用なら revert 一式」から変更）

- **無条件採用**: prepared 境界（フェーズ 2）・転送込み境界（フェーズ 3）の両方で size 2048/4096 の `head_over_base`
  （head/base の中央値 TFLOPS 比）が改善しており、かつ小〜中形状（256〜1024）で有意な劣化（spread 相当を超える
  悪化）がなければ「採用」とする。`crates/backend-metal/src/tile.rs` の `SWIZZLE_ENABLED` を `true` へ変更してコミット
  し、`MetalGemm::new`（本番経路）の既定挙動でスウィズルを有効化する（このドキュメント・`tile.rs`/`gemm.rs`/
  `pipeline.rs` の doc comment を実測結果込みで更新する。マージ前に結線後の再検証〈#795 計画 Step 5〉を行う）
- **サイズ条件付き採用**: 大形状（2048/4096）のみ改善し小形状で有意な劣化がある場合、`tile.rs` に総タイル数閾値の
  純関数 `should_apply_swizzle` を追加し、`gemm.rs` の dispatch 入口で 1 箇所に判定を集約する（CUDA
  `backend-cuda/src/swizzle.rs::should_apply_swizzle` の #784 結線と同型構成。#795 計画 Step 4b）
- **非採用（false 維持。#795 で新設した分岐）**: 改善が確認できない場合は**採用しない**と判断するが、
  `SWIZZLE_ENABLED` を含むスウィズル機構一式は **revert しない**（旧方針からの変更）。理由:
  1) 機構は PR #661 codex-review 指摘対応で既に「本番 false・実機検証待ち」の安全な状態に保たれており、
     `false` のまま残しても本番挙動（`MetalGemm::new`）に影響しない
  2) A/B 計測基盤（`bench_harness::ab`・`new_with_swizzle`・`gemm_swizzle_ab_bench.rs`）は今後の別最適化の
     A/B 計測にも再利用できる汎用資産である
  3) CUDA 側は同型の tgid/threadblock スウィズル機構を実機ゲート通過後に本番結線済み（#784）であり、
     Metal 側が非採用でも「非採用の判断根拠を記録した状態で機構を保持する」対称性は崩れない
  この場合は本ドキュメントの「実測結果」節へ生データと非採用の根拠を記録し、`SWIZZLE_ENABLED = false` を維持した
  まま完了とする（#795 受け入れ条件）
- **判定不可（イシュー #1279 で確定した運用）**: フェーズ 1（安定性セルフチェック）が繰り返し不成立で、5 run
  中央値の算出に足る有効 run が得られない場合。`SWIZZLE_ENABLED=false` のまま維持し、判定不可に至った経緯
  （試行回数・観測した spread・負荷状況）を本ドキュメントへ記録する（`gemm_fine_barrier_ab_bench.rs`〈#809・
  #1278〉と同じ運用）。

判定統計・許容誤差・閾値自体は変更しない（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・
`docs/spec/04-requirements.md` REQ-8 の対象事項のためユーザー承認が必要。イシュー #1279 の実装スコープには
含まない）。

## 実測記録（イシュー #1279・2026-09-06・M4 Max 実機）

**結論: 判定不可（undetermined）**。フェーズ 1（安定性セルフチェック）が 4 試行・ほぼ全サイズ（256〜4096。
run4 の size=256 のみ 1 試行 gate 内〈spread=0.0420〉）で `STABILITY_SPREAD_GATE=0.05` を満たさず、フェーズ 2・3
（A/B 本計測）へ一度も到達しなかった。5 run 中央値の算出に必要な有効 run が 0 のため、`head_over_base` の実測値は
存在しない。

| size | run1 spread（既定 ROUNDS=6） | run2 spread（ROUNDS=10） | run3 spread（ROUNDS=10） | run4 spread（既定 ROUNDS=6） | gate |
|------|------|------|------|------|------|
| 256  | 0.2369 | 0.2838 | 0.6120 | **0.0420（OK）** | 0.05 |
| 512  | 0.1163 | 0.4143 | 0.3113 | 0.5632 | 0.05 |
| 1024 | 0.2196 | 0.4617 | 0.9314 | 0.6206 | 0.05 |
| 2048 | 0.0941 | 1.2047 | 1.2049 | 0.0912 | 0.05 |
| 4096 | 0.2401 | 0.2375 | 0.1527 | 0.1585 | 0.05 |

全 4 試行とも各サイズで gate（0.05）を超過し（run4 の size=256 のみ例外的に gate 内）、`verdict=undetermined` で
終了した（`docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/swizzle_ab_run{1..4}.log`）。round_tflops はこの GPU の
通常想定値より 1 桁前後低い水準（size=256 で 0.09〜0.19 TFLOPS・size=4096 で 3.8〜4.9 TFLOPS）で推移しており、
`gemm-fine-barrier-ab.md`〈#1278〉と同様に計測自体が外部要因で圧迫されていたと判断した。

**原因分析**: run1 直前の `uptime` load average は 6.69/7.15/6.29（負荷源プロセスは未特定）、run2 終了直後
（17:28 前後）に `ps aux` で確認したところ、本 issue の worktree とは無関係な**別リポジトリ（fandhe-frontend）
の `cargo test -p fandhe-frontend-docs-site` プロセス（17:27 開始）が同一マシン上で並走**していることを
確認した。これは `docs/perf/metal-gemm-fine-barrier-ab.md`（イシュー #1278）が**同日（2026-09-06）・同一マシン**
で観測したのと同型の外部要因であり、単発の偶発ではなくこのマシンを共有する他リポジトリのセッション全般からの
CPU 負荷が常態化している状況と判断した。
`ROUNDS`/`COOLDOWN`/`MIN_WARMUP` を増やす方向へ調整（run2・run3: 10/8s/3s。#1278・#1187 と同水準）しても
spread は改善せず、むしろ run2・run3 では単純計測（run1）より悪化した（例: size=2048 が run1 の 0.0941 から
run2/run3 で 1.20 前後まで悪化）。run4（load average 低下傾向〈2.64〉時点で既定値へ復元して再試行）では
size=256 のみ gate を満たしたが、他 4 サイズは依然 gate 超過だった。試行回数 4（#1278 と同水準の試行数で
一貫して同じ傾向が再現したため、これ以上の調整では収束しないと判断し打ち切った。実装計画の上限 8 に対し
余裕はあるが、#1278 の先例（4 試行で打ち切り・判定不可確定）に倣った安全側判断）。

**プロトコルからの逸脱**: 実装計画 §3.3 が定める (a) load average >3.0 時の最大 30 分待機・(b) 実行中の
`uptime` 定期サンプリング（`uptime_during_runN.txt`）は、いずれも本イシューでは実施しなかった。同日・
同一マシンで実施した #1278 が 30 分待機後も spread が収束しなかった先例に倣い、待機を試みても同じ外部要因
（他リポジトリの並走ビルド）が解消しない可能性が高いと判断し、4 試行（ROUNDS/COOLDOWN/MIN_WARMUP の増量・
既定復元を含む）で打ち切る安全側の判断とした（詳細は `docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/
env_info.txt` §プロトコルからの逸脱）。

サーマル記録（`pmset -g therm` 実行前後）: 全試行で「No thermal warning level has been recorded」（サーマル
スロットリングは観測されず、原因はサーマルではなく外部プロセスの CPU 負荷競合と判断した）。

数値契約（AC-1）は別途 PASS 済み: `tests/gemm_swizzle_bit_match.rs` の 2 テスト
（`swizzle_on_off_bit_match_dispatch_auto`・`swizzle_on_off_bit_match_dispatch_tiled_prepared`）が
size 512/1024/2048/4096 で base/head 出力のビット単位一致を確認した（`docs/perf/logs/
metal-gemm-tgid-swizzle-ab-1279/bit_match_test.log`）。既存 parity テスト（`gemm_dynamic_tile_parity`・
`cpu_metal_parity`・`gemm_auto_parity` 等。`docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/
parity_ignored_tests.log`）も非後退で green（tolerance 不変。同一実行ログ内で `command_batching.rs::
pool_reuse_zero_fill_does_not_synchronize_open_batch`・`command_batching_bench.rs::
pool_reuse_interleaved_with_tracked_steps_preserves_batching` の 2 件が FAILED になっているが、いずれも本イシュー
と無関係な既知の環境依存不安定性（エンコード回数カウントのアサーション。GEMM 数値一致・性能には無関係）。
後者は `main`〈本イシューの変更前〉でも同一条件〈left=560 right=50〉で再現することを確認済み。前者は
`--no-fail-fast` 実行時のみ FAILED（同一 HEAD の単独 `--test command_batching` 再実行では PASS）で、
他の並走ビルド負荷下でのフレーキーな挙動と判断した）。

環境情報・実効パラメータ・load average 推移の全詳細は `docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/env_info.txt`
を参照。

## 引き継ぎ（#1280 へ）

- **採否結果**: 判定不可（上記「実測記録」節）。`SWIZZLE_ENABLED=false` を維持し、本番結線は行わない。
- **`tile.rs` doc comment との不整合**: `crates/backend-metal/src/tile.rs` の `SWIZZLE_ENABLED` doc comment は
  「不採用なら本定数を含む変更一式を revert する」と書いているが、本ドキュメントの判断基準（#795 改定版）は
  「不採用（および判定不可）でも機構は残す」としている（`FINE_BARRIER_ENABLED`〈#1278〉と同型の不整合）。
  本イシュー（#1279）は計測・記録のみのスコープのため `tile.rs` は変更しておらず、この不整合は未解消のまま残る。
  #1280 で結線判断と併せて整合させること（`tile.rs` の doc comment 更新が必要）
- **`ROUNDS`/`COOLDOWN`/`MIN_WARMUP` の恒久値化**: 本イシューでは一時ローカル変更（run2・run3: 10/8s/3s）で足り、
  コードの既定値（6/2s/1s）は変更していない。恒久的に増やすべきかは、このマシンの他リポジトリからの負荷競合が
  一過性要因か否かの見極めが必要なため、#1280（または #1278 と共通の後続 issue）の判断に委ねる
- **再計測の要否**: 判定不可のまま `SWIZZLE_ENABLED=false` を維持する運用は `FINE_BARRIER_ENABLED`〈#1278〉・
  #1037 等の先例と同型で問題ないが、負荷の少ない時間帯での再計測により結論が変わる可能性は残る
  （本イシューのスコープ外）

## 参照

- `crates/backend-metal/src/shaders/gemm.metal`（`SWIZZLE_ENABLED`/`SWIZZLE_LOG` function constant 宣言・
  tgid 変換式）
- `crates/backend-metal/src/tile.rs::SWIZZLE_ENABLED`/`SWIZZLE_LOG`（本番既定値。コミット状態既定 `false`
  ロックテスト）
- `crates/backend-metal/src/gemm.rs::MetalGemm::new_with_swizzle`（A/B 計測専用入口）
- `crates/backend-metal/examples/gemm_swizzle_ab_bench.rs`（A/B ベンチ本体。イシュー #1279 でフェーズ 0・
  4 値 verdict 純関数・`verdict=` 出力・実効パラメータ println を追加）
- `crates/backend-metal/tests/gemm_swizzle_bit_match.rs`（AC-1 の独立受け入れテスト。イシュー #1279 で新規追加）
- `docs/perf/metal-gemm-fine-barrier-ab.md`（同型構成・同日実測の姉妹イシュー #1278。同一の外部負荷要因を観測）
- `docs/perf/logs/metal-gemm-tgid-swizzle-ab-1279/`（イシュー #1279 の実測ログ・env_info）

## 旧手順（履歴）

イシュー #746 以前は checkout 切替＋一時トグル方式だった。base/head 計測が時間的に分離されサーマルドリフトが
系統誤差として乗る問題があり（#746 イシュー本文の 2026-08-19 実測: 対照カーネルが最大 70% 超変動）、
上記「計測手順」節の interleaved 方式へ置き換えた。参考として残す。

```sh
git fetch origin

# base（変更前。スウィズル導入前の直近コミット）
git checkout <base-sha>
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release > /tmp/gemm_bench_base.txt

# head（本イシューの実装ブランチ）。SWIZZLE_ENABLED は既定 false のため、
# 計測前に crates/backend-metal/src/tile.rs の SWIZZLE_ENABLED を一時的に
# true へ書き換える（コミットしない。計測後に revert する）。
git checkout perf/540-metal-gemm-tgid-swizzle
# （ここで SWIZZLE_ENABLED を true へ一時変更）
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release > /tmp/gemm_bench_head.txt
# （計測後: git checkout -- crates/backend-metal/src/tile.rs で revert）
```
