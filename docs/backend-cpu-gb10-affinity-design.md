# GB10 小形状 GEMM 大コア affinity 自機判定の設計（イシュー #1576）

親: #1571（低レイヤー診断の性能候補）。依存 #1574（マージ済み・PR #1654）。関連: #1364（`BIG_CORE_LIMIT_ENABLED`。REJECT 確定・`cpu_capacity` 誤検出）・#1305/#1319（taskset pin 実測の一次データ）・#1575（M4 Max 側の別軸対応。本イシューとは独立）。

## 1. 背景・目的

`docs/perf/lowlayer-diagnosis-2026-09-12.md` §3（実機実測。出典 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/rayon-sweep.jsonl`／`rayon-sweep-pinned.jsonl`）で、GB10（DGX Spark GB10。Cortex-X925 ×10 + Cortex-A725 ×10）は次の非自明な特性を持つことが確定している。

| 条件 | train reuse（size=64・3 起動中央値） |
|---|---|
| 無 pin T20（全コア） | 1.029 s |
| 無 pin T10（スレッド数のみ大コア数に一致） | 2.359 s（**T20 より遅い**） |
| 大コア pin T10（`taskset -c 5-9,15-19`） | 0.857 s（**T20 より速い**） |

`crate::thread_limit`（#1363・`BIG_CORE_LIMIT_ENABLED`）はスレッド**数**を大コア数へ制限するだけで OS レベルの affinity は設定せず、GB10 実機実測（#1364）で REJECT 確定済み（`cpu_capacity` sysfs 誤検出に加え、上表のとおりスレッド数を絞るだけではむしろ悪化する）。効果があるのは実際に OS レベルで大コアへ thread affinity を設定することであり、これは #1364 とは異なる新規メカニズムである。

## 2. 大コア判定（検出）— 依存追加なし・unsafe 不要

`cpu_capacity` sysfs（#1364 で誤検出確定: 718/731/1017/1024/997 の非一様 5 段階分布、`big_cores_from_capacities` が最大値 1024 を持つコアを 1 個だけ検出）は使えないため、`crate::gb10_affinity` は以下 2 つの独立指標をクロスバリデーションする（`docs/perf/logs/cpu-gemm-rayon-sweep-1305/env_info_raw-dgx.txt` の実機実測で両者が完全一致することを確認済み）。

1. **`cpufreq/cpuinfo_max_freq`**（`/sys/devices/system/cpu/cpu<N>/cpufreq/cpuinfo_max_freq`）: 2 群に分かれる（3900MHz 群＝X925 ×10 / 2808MHz 群＝A725 ×10。`lscpu -e` の MAXMHZ 列と一致）
2. **`regs/identification/midr_el1`**（ARM MIDR_EL1 の partnum ビット `[15:4]`）: `cpu0`（little 群）= `0xd87`（Cortex-A725）、`cpu5`（big 群）= `0xd85`（Cortex-X925）

両指標がそれぞれ厳密に 2 群へ分かれ、かつ両者の CPU-id 分割が（群の順序に依らず）完全一致する場合のみ、周波数の高い方の群を大コア群として確定する（partnum の大小や ARM コアコード表のハードコードは行わない＝将来の SoC 世代でも破綻しない）。いずれかのファイルが欠損・parse 不能・2 群にきれいに分かれない・両指標の分割が食い違う場合は `None`（判定不能・no-op）とする。

`midr_el1` は x86_64 には存在しないため、この判定は構造的に ARM 系の非対称構成にのみ発火する（Intel P/E コア機・CI の x86_64 ランナーは自然に `None` へフォールバックする）。

### cgroup／cpuset 制約への fail-closed 対応

`/proc/self/status` の `Cpus_allowed_list:` 行を読み、検出した大コア CPU-id 集合が allowed 集合の**部分集合**であることを確認する（`sched_setaffinity` が禁止 CPU に対し `EINVAL` を返すのを待たず、事前に安全側へ倒す）。allowed 一覧自体が読めない・parse できない場合も安全側（判定不能）へ倒す。

### `RAYON_NUM_THREADS` との関係

`crate::thread_limit` と同じ設計判断として、`RAYON_NUM_THREADS` が有効な正整数として明示設定されている場合は本機構自体を適用しない（ユーザーの明示指定を上書きしない。`thread_limit::parse_env_num_threads` を `pub(crate)` 化して共有する）。

この検出ロジックはすべて `std::fs::read_to_string` ベースの sysfs 読み取りであり、**新規依存・unsafe は不要**（既存 `thread_limit.rs::big_cores_from_sysfs` と同型の安全な I/O）。

## 3. Affinity 設定（本体機構）— unsafe FFI（個別承認済み）

診断が示す通り、性能改善に必要なのは「スレッド**数**の制限」ではなく「OS レベルでの thread-to-CPU pinning」である。Rust std にはこの機能が無く、Linux では `sched_setaffinity(2)` システムコールが必要。許容依存 9 区分（`.claude/rules/deps-policy.md`）に `libc`／`core_affinity` は含まれず、追加はユーザー承認が要る。**新規クレートを追加せず**、`crates/backend-cpu/src/gemm_blis/cache_params.rs::sysctl_ffi`（macOS `sysctlbyname`）と同型の生 FFI 宣言（Rust std は glibc に動的リンクしているため、追加クレートなしで呼び出せる）で実装する。

```rust
#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const [u64; 16]) -> i32;
}
```

- 本 unsafe FFI の導入は `.claude/rules/security.md`「unsafe」節（FFI 境界に限定・理由コメント必須・レビュー必須）の対象。実装 Agent はイシュー #1576 の計画承認フローの一部として本節の内容を提示し承認を得た（着手前承認済み）
- `#[cfg(target_os = "linux")]` に限定（macOS には波及しない。#1575 は既に別ファイル・別軸）
- マスク構築はビット演算のみの safe コード。`unsafe` ブロックは syscall 呼び出し 1 箇所のみ（`crates/backend-cpu/src/gb10_affinity.rs::affinity_ffi::pin_current_thread_to_cpu`）、`// SAFETY:` コメントで「有効な固定長配列への参照を渡すのみ・戻り値のエラーは無視して fail-safe（affinity 未設定のまま続行）」を明記
- 戻り値が非 0（失敗）でも **panic・unwrap しない**（本番経路での `unwrap`/`expect` 禁止規約。設定失敗時は該当スレッドが affinity 無しで動作し続けるだけで正しさに影響しない）

## 4. 専用スレッドプール（グローバルプールを汚染しない）

`rayon::ThreadPoolBuilder::build_global()` はプロセス全体のデフォルトプールを差し替えるため、`backend-cpu` 以外（`mse_loss_backward` 等の他の rayon 利用箇所）にも影響し、ホストアプリケーション側の rayon 利用とも衝突しうる。**GEMM 専用の `OnceLock<Option<rayon::ThreadPool>>`**（`crate::gb10_affinity::affinity_pool`）を新設し、影響範囲を GEMM の 2 エントリ関数（`gemm_blis_parallel_with_transpose`・`gemm_blis_bias_act_parallel`）に限定する。

専用プール（大コア数スレッド）とグローバルプール（全コア数スレッド）が同時に活性化しうる（例: train の backward 中に非 GEMM rayon 処理が並走する場合）ためオーバーサブスクリプションの余地は残るが、GEMM 専用プールは `.install()` される短命スコープに限られる。A/B 実測（§6）で train／infer のエンドツーエンド（GEMM だけでなく非 GEMM 区間込み）の非後退を確認することで検出する。

## 5. 適用範囲（ルーティング）— 小形状限定

大形状（N=1024/2048/4096 の正方 GEMM）を専用プール（大コア数のみ）へ回すと、既に本番採用済みの 2D 動的分配（`gemm_blis::TWO_D_DYNAMIC_PRODUCTION_ENABLED`。全コア前提・#1313 で ADOPT）から性能を奪う。よって全 GEMM 呼び出しを無条件に専用プールへ通さず、`m * n * k`（総仕事量）が `GB10_AFFINITY_MAX_WORK`（初期値 `32 * 1024 * 1024` ≈ 33.5M）以下の形状に限定する。

診断対象形状（`bench-fandhe --task train/infer --size 64` の MLP: `BATCH=64・D_IN=784・D_HIDDEN=256・D_OUT=10`）の layer1 GEMM は `m*n*k = 64*256*784 ≈ 12.8M`。対して非対象の正方 GEMM 最小形状 N=512 は `512^3 ≈ 134M`。この間に十分な余裕を持つ値として上記閾値を初期値とし、実測（§6 ガードセル）で大形状の非後退を確認する。判定規則自体（5 run 中央値・checksum 完全一致・非後退 `ratio<=1.00`）は事後緩和しない。

## 6. 正しさ（bit 完全一致）契約

`gemm_blis` の並列分割はスレッド数に依らず出力が bit 完全一致であることが `#1364`／`#1312`／`#1367`／`#1318`／`#1481` 等の全 A/B で確認済みの既存不変条件。本機構はスレッドの**実行位置**のみを変え、GEMM の数学的分割（`panel_rows` 等）は `rayon::current_num_threads()`（専用プール内では専用プールのスレッド数を返す）を経由して既存ロジックがそのまま処理するため、新規の数値一致リスクは生じない（tolerance／baseline は変更しない。REQ-2 複合判定は影響を受けない）。

## 7. 性能実測（A/B）

本エージェント実行環境に GB10（DGX Spark GB10）実機への到達手段が無いため、既定 `GB10_AFFINITY_ENABLED = false` のまま実装・単体テストを完了し、性能 A/B は未実施のまま `docs/perf/cpu-gemm-gb10-affinity-ab.md` に事前登録判定規則のみを記録する。ADOPT 判断は実機実測を伴う後続セッションへ引き継ぐ（#1364 と同型のロールバック運用: 1 行差し戻すだけで無効化できる）。

## 8. スコープ外事項（`.claude/rules/out-of-scope-tracking.md` に従い別 issue で追跡）

- tolerance／baseline の変更（issue 本文の契約により対象外）
- M4 Max 側の対応（#1575 が別軸で担当）
- `GEMM_THREADING_THRESHOLD`（`gemm_blis/mod.rs`。現状 `#[cfg(test)]` 限定・未結線）との統合・置き換え（本イシューは新規の独立機構として追加し、既存の未結線メカニズムには触れない）
- ADOPT 確定後のルーティング閾値のさらなるチューニング（train/infer 以外の形状レンジへの拡大等）は、A/B 結果を踏まえて必要なら新規 issue として起票する
