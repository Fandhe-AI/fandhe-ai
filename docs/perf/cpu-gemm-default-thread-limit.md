# CPU GEMM 既定スレッド数の大コア限定（イシュー #1363）

親 #1362・祖 #1361（Phase 6 CPU スケジューリング補完）・ルート #1269。

## 0. 位置づけ

`docs/perf/cpu-gemm-candle-cpu-retune.md` §8.2 の `RAYON_NUM_THREADS`
スイープで、Apple M4 Max（P12+E4）・DGX Spark GB10（X925×10+A725×10）
とも「大コア数付近でスループットが落ち込み全コアで部分回復する」という
非単調性が観測された。`gemm_blis_parallel`（`crates/backend-cpu/src/
gemm_blis/mod.rs`）の静的等分割行パネル（`c.par_chunks_mut(panel_rows *
n)`。`panel_rows = m.div_ceil(rayon::current_num_threads())`）が異種
コア（big.LITTLE 系）構成で little コア律速になる仮説の検証前提として、
本イシューは `RAYON_NUM_THREADS` 未指定時の既定並列度を「物理大コア数」
へ限定する制御をプラットフォーム判定付きで実装した。

**本 PR（#1363）のスコープは実装とフォールバックのテストのみであり、
性能上の採否判断（勝敗）は含まない**。§8.2 のスイープ自体は
「スレッド数＝大コア数」が谷に見えるデータでもあるため、本実装は
「勝ちが確定した最適化」ではなく「#1364 が検証する仮説の実装」と
位置づける。両実機 5 回中央値・framework-compare 前後比較による採否は
#1364 に引き継ぐ。

## 1. 判定モジュール

`crates/backend-cpu/src/thread_limit.rs`（非 `cfg(test)`・依存追加なし・
`unsafe` 新規導入なし）。

- **macOS**: `/usr/sbin/sysctl -n hw.perflevel0.logicalcpu`／
  `hw.logicalcpu` を `std::process::Command`（絶対パス・固定引数・
  `env_clear()`）で実行し、前者が後者より真に小さい場合のみ P コア数を
  大コア数とする。既存 `gemm_blis::cache_params::sysctl_ffi`
  （`unsafe extern "C" sysctlbyname`）は `#[cfg(test)]` 限定・本番非到達の
  ままとし、本番到達化（unsafe 面の実質的な新設）は行わない
  （PR #766「常に不活性な sysctl 経路」撤去の教訓・自動運転ではユーザー
  承認を得られないため安全側の `Command` 経路を採用）
- **Linux**: `/sys/devices/system/cpu/cpu<N>/cpu_capacity` を走査し、
  値が非一様（big.LITTLE 系）なら最大値と一致するコア数を大コア数と
  する。全 CPU が一様（同種コア。x86_64 CI 環境等）・1 つでも欠損／
  parse 不能なら `None`（no-op）
- **上記以外のプラットフォーム**: 常に `None`

判定結果は `OnceLock` で 1 回だけキャッシュする（初回の並列 GEMM 呼び
出しで評価）。`current <= 1`（シングルスレッドプール）ではプラット
フォーム I/O 自体を省略し、起動プローブ（`bench-harness::startup_probe`）
等への影響を避ける。

## 2. 適用範囲・結線

`crates/backend-cpu/src/gemm_blis/mod.rs` の `rayon::current_num_threads()
.max(1)` 8 箇所すべて（実装計画時点では 5 箇所と見積もっていたが、実装
時の grep で本番 2 箇所（`gemm_blis_parallel_with_transpose`・
`gemm_blis_bias_act_parallel`）に加え `#[cfg(test)]` の A/B 計測ハーネス
6 箇所（`gemm_blis_shared_b_region`・`gemm_blis_parallel_with_blocks`・
`gemm_blis_parallel_row_panel_with_blocks`・`gemm_blis_parallel_2d_with_blocks`・
`gemm_blis_shared_b_pc_outer_region`・`mod tests` 内 `run_detected`）と
判明。とくに `gemm_blis_parallel_row_panel_with_blocks` は「本番公開入口
とロジックを完全一致させる」ことが前提の A/B 基準線〈PR #1075
codex-review 指摘〉のため、本番と同じ実効スレッド数算出を適用しないと
基準線が崩れる。よって全 8 箇所へ適用した）を
`crate::thread_limit::effective_num_threads(rayon::current_num_threads())`
へ差し替えた。`crates/gemm.rs::gemm_parallel`（#24 の参照点・公開 API
非破壊）・elementwise／rmsnorm／softmax／reduction は対象外
（親イシューのスコープは `gemm_blis_parallel`）。

単一 const ゲート `thread_limit::BIG_CORE_LIMIT_ENABLED = true` で本番
結線済み（本番結線は事前承認済み。#1364 が性能後退と判断した場合は
このゲートを `false` へ 1 行差し戻すだけで済む）。

## 3. `RAYON_NUM_THREADS` との関係

`RAYON_NUM_THREADS` が有効な正整数として設定されている場合、その値を
rayon が採用した既定として尊重し、大コア数による上限は適用しない
（新しい環境変数は追加していない。`RAYON_NUM_THREADS=<全論理コア数>`
が「限定なし（従来挙動）」の再現手段になり、#1364 は同一バイナリで
on/off 両腕を計測できる）。

## 4. フォールバック契約

判定失敗（sandbox でのプロセス spawn 不可・sysfs 不在・同種コア構成・
上記以外のプラットフォーム等）は fail-safe に「現行既定
（`rayon::current_num_threads()`）をそのまま使う」へ倒れる。panic 経路
なし（`unwrap`／`expect` 不使用）。`resolve` 純関数の契約テスト
（`crates/backend-cpu/src/thread_limit.rs` の `mod tests`）・sysfs
fixture テスト（非一様／一様／欠損／不正値／デコイディレクトリ）・
sysctl stdout parse テストで網羅する。

## 5. 起動コストへの影響

検出は初回並列 GEMM 呼び出しで 1 回のみ（`OnceLock`）。Linux は sysfs
読み取り（μs オーダー）。macOS は `sysctl` 子プロセス 1 回（ms オーダー）
で、`bench-harness::startup_probe` の CPU 経路（`ops.gemm` を 1 回呼ぶ）
に 1 回分の検出コストが乗りうる。`make startup-bench` への影響の実測は
本 PR では行っていない（時間制約。#1364 の実測時に併せて確認する）。

## 6. 実機実測（イシュー #1364 で実施・採否確定）

両実機で `thread_limit_report()` の検出生値・`RAYON_NUM_THREADS` on/off の
framework-compare 前後比較（`run_gemm_gate_cpu.sh` 同一バイナリ 5 回計測・
`compare_gemm_ab.py --device cpu`）を実施した。詳細な結果表・env_info は
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §13・
`docs/perf/logs/cpu-gemm-thread-limit-1364/` を参照し、本節では要点のみ記す。

### 6.1 検出生値

- **Apple M4 Max**: `hw.perflevel0.logicalcpu`=12・`hw.logicalcpu`=16 →
  `detected_big_cores=Some(12)`（P コア数を正しく検出）
- **DGX Spark GB10（Grace CPU）**: `lscpu` は Cortex-X925 ×10（big）＋
  Cortex-A725 ×10（little）の 2 群構成を報告するが、実際の
  `/sys/devices/system/cpu/cpu<N>/cpu_capacity` は 5 段階の非一様な分布
  （718×5・997×5・731×5・1017×4・1024×1）であり 2 群と一致しない。
  `big_cores_from_capacities` は「最大値と一致するコア数」を大コア数と
  するため、この分布では最大値 1024 を持つコア 1 個のみを検出してしまい
  `detected_big_cores=Some(1)` となった。§1 の判定方式（②）が前提とする
  「非一様なら 2 群」という単純化が、この実機の capacity 値の粒度
  （おそらく DVFS ブースト状態を反映した多段階値）とは整合しない

### 6.2 on/off 比較結果と採否

- **Apple M4 Max**: 全 6 セル（N=512/1024/2048 × fresh/reuse）で
  on（限定あり。effective=12）が off（限定なし。effective=16）に対し
  0.80〜0.91 倍（非後退・改善）。checksum 完全一致
- **DGX Spark GB10**: `detected_big_cores=Some(1)` により on 腕が実質
  シングルスレッド（effective=1）へ縮退し、全 6 セルで 1.19〜4.34 倍の
  重大な性能後退（N=2048 reuse: 35.0 ms→152.0 ms）。checksum 完全一致
  （並列度は結果に影響しない設計どおり）
- **総合判定: REJECT（不採用）**。決定規則（両実機・全判定可能セルで
  reuse の ratio<=1.05 を要求）に対し DGX が明確に不合格。§0 で位置づけ
  たとおり本イシュー（#1363）は実装のみで勝敗判断を含まないとしていたが、
  #1364 の実測により「大コア数＝谷」であった §8.2 のスイープ結果と整合する
  形で不採用が確定した
- **反映**: `BIG_CORE_LIMIT_ENABLED` を `false` へ差し戻し済み
  （`crates/backend-cpu/src/thread_limit.rs`。#1364 のコミット）。
  `effective_num_threads` は常に `current`（rayon 既定値）をそのまま返す
- **スコープ外**: DGX（Linux 非対称コア構成）の検出手段の見直し
  （`cpu_capacity` 以外の指標。例: `topology/capacity_dmips_mhz`・モデル名
  グルーピング）は本イシューでは対応しない

## 7. セキュリティ考慮（OWASP Top 10）

- **A03 インジェクション**: 外部入力は環境変数（`RAYON_NUM_THREADS`）・
  sysfs 内容・`sysctl` stdout の 3 系統。`Command` は絶対パス
  `/usr/sbin/sysctl`・固定引数のみ（ユーザー入力を一切連結しない）・
  `env_clear()` で PATH 依存を排除。stdout／sysfs 値は正の `usize` かつ
  上限（4096）付きで厳密 parse し、失敗は全て `None`（フォールバック）
- **A05 設定ミス**: 判定不能時は現行既定へ倒す fail-safe。panic 経路
  なし。単一 const ゲートで無効化可能
- **A06 脆弱コンポーネント**: 依存追加なし（`=x.y.z` 固定・`Cargo.lock`
  不変）
- **A08 整合性**: bit 完全一致契約（REQ-2）・FMA 契約は不変。tolerance／
  baseline／ガードレール閾値には触れない。`unsafe` の新規導入なし
