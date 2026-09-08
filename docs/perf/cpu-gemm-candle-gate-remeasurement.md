# CPU GEMM N=512/1024/2048 reuse candle 比再計測と #1117 ゲート判定（イシュー #1148）

## 状態: DGX Spark（Grace CPU）・Apple M4 Max とも実機実測完了。#1117（reuse candle 超え）は両実機・全形状で未達成（DGX N=2048 は候補側 candle 無効データにより判定不能）と判定した。#1185 で正式系列 `fandhe-ai =0.7.0` を 2026-09-06 に両実機で再計測し未達成を確定（§12）。#1364 で既定スレッド数の大コア限定（#1363）on/off を両実機比較し REJECT（不採用）と確定・`BIG_CORE_LIMIT_ENABLED=false` へ差し戻し済み（§13）。#1367 で `IcDynamic` variant を両実機比較し REJECT（不採用）と確定・本番結線せず（§14）。#1337 で借用ビュー readout（既定 OFF feature）切替前後を両実機で 2026-09-07 に再計測（§15）。DGX は非後退だが未達のまま、M4 Max は達成見込み（片方向負荷差あり・確度限定的）。正式判定（§12）は不変。#1292 で reuse 計測境界のフェーズ分解を両実機で 5 回計測中央値実測し、§8.1 の「facade/autodiff 呼び出しオーバーヘッド・readout コピー・checksum の固定費」推定を内訳分解して確定した（§16）。#1305 で専有環境の RAYON_NUM_THREADS スイープを両実機で 5 回計測中央値再実測し、DGX Spark GB10 の N=1024 非単調性を taskset pin 実験で H1（異種コア由来）と確定・Apple M4 Max は専有ゲート不通過のため undetermined のまま記録した（§17）。#1312 で `TwoDDynamic`（2D 動的分配）vs `RowPanel` の両実機 A/B を実施し、DGX（専有ゲート通過）は全形状で `RowPanel` を 1.09〜1.80 倍上回ったが Apple M4 Max が専有ゲート不通過のため最終判定は undetermined（結線せず #1313 へ記録のみ引き継ぎ）・DGX 単独の非単調性（T10/T8）は「残存」（比 0.86 前後・僅かに閾値未達）と判定した（§18）。#1262 で承認済み tolerance 契約（#1241 承認・#1443/#1445 実装）下の DGX Spark GB10 N=2048 を再計測し、「判定不能」が解消して確定判定（未達・0.950 倍）へ遷移したことを確認した（§19。N=512/1024 は引き続き未達。Apple M4 Max は N=2048 が元々 0 fail のため対象外）。#1301 で出力並列ゼロ埋め（#1299）の on/off を両実機実測し、両実機・全形状で非後退（DGX N=2048 alloc_c 約 48% 削減）を確認したが、事前宣言した規則 4（candle 比の非後退）が緩和なしでは 6 セル中 3 セルで不成立だった点を PR #1448 の codex-review が指摘したため、いったん有効化した `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS = 2 << 20` を `usize::MAX`（無効化）へ差し戻し、§20.1a の改定版規則 4 を事前登録規則として固定したうえで独立の再計測（§20.6）を経て確定する方針とした（§20）。#1313 で `TwoDDynamic`（2D 動的分配）を M4 Max 専有ゲート付き Phase 0 再計測で ADOPT 確定・本番結線し、framework-compare gemm cpu 全 12 セル（両実機）が非後退（改善方向）であることを確認した（§21。candle 比ゲート自体の判定は不変）。#1321（#1283 Phase 4）で Phase 1〜3 結線後（実質 `TwoDDynamic` のみ）の両実機ゲート再判定を実施し、正式系列（`fandhe-ai =0.7.0` ピン）は両実機・全形状で未達成のまま（DGX N=2048 は §19 の確定判定を再確認）だが、参考系列（origin/main HEAD `ced4d14`）では Apple M4 Max が全 3 形状（512/1024/2048）で達成・DGX Spark GB10 が N=2048 のみ達成（1.562 倍）という、初めて candle 比ゲート達成が観測された結果を記録した（§22。ピン未更新のため正式判定は不変） #1438 で同一 facade ソース下の借用ビュー readout 既定化 before/after を両実機実測し、reuse 全 6 セル（3 サイズ×2 実機）非後退・checksum 完全一致を確認（ADOPT。§15 の交絡を解消。正式判定は §12／§19 のまま不変。fresh は参考記録で DGX 全形状改善だが M4 Max N=512/1024 は after/before 1.0115／1.0028 と僅かに後退方向であり、計測ノイズかどうかは本記録のみでは確定できない。§23）

## 1. 位置づけ

親 #1117 の受け入れ条件 2・3（CPU GEMM N=512/1024/2048 candle 比を 5 回計測中央値・
checksum 複合判定 ok で再計測しゲート表へ反映する／未達が残る場合は原因分析を
`docs/perf/` へ記録する）を、CUDA 側 #1142・Metal 側 #1147 と同一プロトコルの
`run_gemm_gate.sh`／`compare_gemm_gate.py` を CPU device 対応拡張した上で実行し
確定する。#1144（`docs/perf/cpu-gemm-candle-cpu-retune.md` §8）により CPU GEMM の
`SharedB`／`SharedBPcOuter` 候補は不採用・本番結線せずと確定済みのため、本計測は
本番既定 `RowPanel`（現行 `gemm_blis_parallel`）のまま行う（§3 で確認したとおり
CPU GEMM 本番経路は crates.io 公開版 `fandhe-ai =0.6.0` と HEAD で同一）。

## 2. 計測環境・プロトコル

- 実機: DGX Spark GB10（Grace CPU。Cortex-X925 ×10 + Cortex-A725 ×10、計 20 論理コア）・
  Apple M4 Max（P コア 12・E コア 4）。実ホスト名は
  `docs/real-hardware-verification-env.local.md` 方式のローカル管理
- worktree HEAD（origin/main 由来）: `ea19a34`
- 集計ツール: `scripts/bench/framework-compare/run_gemm_gate.sh <device> <label>`
  （#1142 の CUDA 専用実装を #1147 で Metal 対応汎用化・本 Issue で CPU 対応拡張）
  ／`run_gemm_gate_cpu.sh`（新規薄い wrapper）／`compare_gemm_gate.py --device cpu`
  （新規。既定 `cuda` で #1142/#1147 と後方互換）。`README.md`「GEMM ゲート
  5 回計測」節参照
- 対象形状は N=512/1024/2048（cuda/metal の N=1024/2048/4096 と異なる。環境 10/11
  の CPU 単発計測が対象とした形状に合わせる）
- N=512/1024/2048 それぞれ `bench-fandhe gemm cpu <N> reuse`・`bench-candle gemm
  cpu <N> fresh`（candle は reuse 非対応）を run 内で交互に 5 回起動し、run 間
  中央値で判定（coding-rust.md「ベンチは 5 回計測の中央値」）。CPU のみ
  `bench-fandhe gemm cpu <N> fresh` も同数追加起動する（環境 10/11 の単発 fresh
  計測との連続性を説明するための参考記録。**判定には使わない**。§5.4）
- 正式判定はゲートツール契約（reuse vs candle fresh）のみで行う。**単一系列**
  （`fandhe-ai =0.6.0`。正式系列のみ）で計測した。理由は §3 参照
- **DGX Spark 側の計測方式**: 共有作業ディレクトリ `~/work/rust-ai-library-run`
  （実機検証手順の既定 rsync 先。複数の並列 Issue 実装セッションが共有する
  単一ディレクトリ）に他の並列実行中セッションが直近更新したと見られる
  `.rev-stamp`・壊れた `.git` ファイルが残存していることを確認したため、
  他セッションの進行中の状態を破壊しないよう、本 Issue 専用の隔離ディレクトリ
  （`~/work/fc-1148/`。`scripts/bench/framework-compare/`・RAYON スイープに
  必要な `crates/`・ルート `Cargo.toml`／`Cargo.lock`・`scripts/bench/
  oss-gemm-compare/` のみを rsync）を新規作成して計測した。計測完了後に
  `~/work/fc-1148/` は削除済み（共有ディレクトリには一切触れていない）
- 熱・負荷状態確認: 計測前後で `uptime`（負荷平均）・`lscpu`（DGX）／`sysctl`
  の機種名・P/E コア構成（M4 Max）を記録
- 生データ:
  `scripts/bench/framework-compare/results/raw/results-dgx-cpu-gemm-gate-0.6.0.jsonl`・
  `results-m4max-cpu-gemm-gate-0.6.0.jsonl`（各 45 行）、失敗記録はいずれも空
  （`skipped-{dgx,m4max}-cpu-gemm-gate-0.6.0.log`）、manifest は
  `manifest-{dgx,m4max}-cpu-gemm-gate-0.6.0.json`
- 実行ログ:
  `scripts/bench/framework-compare/results/run_gemm_gate_cpu-{dgx,m4max}-0.6.0.log`
- RAYON スイープ（並列化分析用）:
  `docs/perf/logs/cpu-gemm-candle-gate-1148/rayon_sweep_{dgx,m4max}.log`

## 3. なぜ単一系列（正式系列のみ）か

`git diff v0.6.0..HEAD -- crates/backend-cpu/src crates/facade/src crates/autodiff/src
crates/tensor-core/src` を実装着手前に確認した結果、非コメント差分は
`crates/backend-cpu/src/gemm_blis/mod.rs`（`#[cfg(test)]` 内の A/B ハーネス整理
のみ。本番関数のコメント追記を除く実装変更なし）と `crates/backend-cpu/src/
rmsnorm.rs`（イシュー #1102 の f64 縮約精度契約。rmsnorm は GEMM と無関係な
正規化統計の計算経路であり `gemm_blis_parallel`／`gemm_blis_bias_act_parallel`
を通らない）に限られることを確認した。CPU GEMM 本番経路（`gemm_blis_parallel`。
`crates/backend-cpu/src/gemm_blis/mod.rs`）は crates.io 公開版 `fandhe-ai =0.6.0`
と HEAD で実質同一であるため、CUDA 側 #1142・Metal 側 #1147 のような「正式系列・
参考系列の 2 系列併記」は不要と判断し、正式系列（承認済みピン `=0.6.0`・registry
解決）のみで計測した。

## 4. 実測結果

### 4.1 DGX Spark GB10（Grace CPU）

| N | fandhe-ai reuse median (min–max, n=5) | candle fresh median (n=5) | candle/fandhe | GFLOP/s（fandhe） | 判定 | fandhe-ai fresh median（参考。n=5） |
|---|---|---|---|---|---|---|
| 512 | 2.376 ms (1.333–2.706 ms) | 1.805 ms | 0.760 | 112.96 | 未達 | 2.507 ms |
| 1024 | 7.085 ms (6.818–7.419 ms) | 5.604 ms | 0.791 | 303.09 | 未達 | 7.891 ms |
| 2048 | - | - | - | - | **判定不能**（candle 側要素誤差超過。§5.2） | - |

出典: `results/raw/results-dgx-cpu-gemm-gate-0.6.0.jsonl` を
`compare_gemm_gate.py --device cpu` で集計。

### 4.2 Apple M4 Max

| N | fandhe-ai reuse median (min–max, n=5) | candle fresh median (n=5) | candle/fandhe | GFLOP/s（fandhe） | 判定 | fandhe-ai fresh median（参考。n=5） |
|---|---|---|---|---|---|---|
| 512 | 744.0 µs (720.4–771.1 µs) | 699.1 µs | 0.940 | 360.80 | 未達 | 727.8 µs |
| 1024 | 3.787 ms (3.667–3.864 ms) | 2.749 ms | 0.726 | 567.02 | 未達 | 3.494 ms |
| 2048 | 24.098 ms (23.154–24.730 ms) | 17.694 ms | 0.734 | 712.92 | 未達 | 23.120 ms |

出典: `results/raw/results-m4max-cpu-gemm-gate-0.6.0.jsonl` を
`compare_gemm_gate.py --device cpu` で集計。

## 5. データ有効性

### 5.1 fandhe-ai 側の要素単位検証

両実機とも全 30 run（reuse 15 + fresh 15）で `parity_fail_count=0`・`parity_total`
が期待要素数（N=512: 262,144／N=1024: 1,048,576／N=2048: 4,194,304）と一致した。
fandhe-ai 側の parity は両実機・全 size で異常なし。

### 5.2 DGX N=2048 candle 無効データ（R3）の再現確認

**5 run すべてで完全に決定的に再現した。** DGX Spark 側の `candle/cpu N=2048
fresh` は 5 run いずれも `parity_fail_count=2, parity_total=4194304,
parity_max_abs_err=3.814697e-05, parity_max_rel_err=3.944416e-01`（run 間で
1 ビットも変わらず完全一致）。環境 10（`results/summary.md` 行 1116。2026-09-02
計測・単発 fresh）の値（`fail=2/4194304, max_abs=3.815e-05, max_rel=3.944e-01`）
とも一致し、burn/cpu も同一 N で `fail=5/4194304, max_abs=3.529e-05`（決定的な
丸め誤差挙動）であることから、DGX Spark の CPU（Grace, aarch64）で N=2048
特有の丸め誤差超過が単発計測・5 回計測いずれでも安定して再現する、候補側
（candle-core 0.11.0 の CPU GEMM カーネル。gemm crate 経由）固有の決定的な
挙動であることを確認した。原因の内部切り分け（candle-core／gemm crate 側の
丸め順序差）は本 Issue のスコープ外（fandhe-ai 側は全 run `parity_fail_count=0`
であり自作コア側に問題はない）。この結果を受け、`compare_gemm_gate.py`
（fail-closed 設計。§0）は N=2048 を「判定不能」として確定し、tolerance は
緩めていない。一方 M4 Max 側の同じ N=2048 では `parity_fail_count=0` であり
（§5.1）、この無効データはバックエンド固有（アーキテクチャ・コンパイラ最適化
差）の挙動である。**→ §19 で承認済み tolerance 契約（イシュー #1241／#1262）下の
実測により「判定不能」が解消して確定判定（未達）へ遷移したことを記録した。**

### 5.3 集計ツールの判定確認

`compare_gemm_gate.py --device cpu` は DGX 側 N=512/1024 は確定判定（未達）・
N=2048 は判定不能を返し（exit code 3）、M4 Max 側は 3 size とも確定判定（未達）
を返した（exit code 3）。いずれも「判定不能」を性能値の確定表示に混同させない
fail-closed 設計どおりの挙動。

### 5.4 環境 10/11 単発 fresh 行との対比（モード差と実力差の分解）

M4 Max fresh 参考列（§4.2。判定に使わない）と reuse 正式列を比較すると、N=512
は 744.0 µs（reuse）vs 727.8 µs（fresh）でほぼ同水準、N=1024 は 3.787 ms（reuse）
vs 3.494 ms（fresh）で reuse がやや遅い、N=2048 は 24.098 ms（reuse）vs
23.120 ms（fresh）で同様の傾向。DGX 側も N=512 で 2.376 ms（reuse）vs 2.507 ms
（fresh。reuse がやや速い）、N=1024 で 7.085 ms（reuse）vs 7.891 ms（fresh。
reuse がやや速い）、N=2048 は reuse 491.6 GFLOP/s 相当 vs fresh 478.1 GFLOP/s
相当（reuse がやや速い。N=2048 は §4.1 のとおり判定不能のため `compare_gemm_gate.py`
の表には出力されず、生 JSONL の reuse/fresh 各 5 run 中央値から算出した参考値）と、
両実機で fresh/reuse の差は数 % 程度にとどまり
GPU 系（CUDA/Metal）で見られる明確な差はない。fresh は tape 構築・`tape.var`
の N² コピー 2 回を計測窓に含む一方 reuse はそれを含まないが、CPU では
この差自体が GEMM 計算時間（数 ms〜数十 ms）に対して無視できるほど小さい
ことを両実機で確認した（M4 Max で reuse がわずかに fresh を下回る逆転が
複数 N で見られたのは、min–max の重なる範囲内のノイズと見て問題ない）。
環境 11（`results/summary.md` 行 1243〜1259。単発 fresh 計測。2026-09-02）の
M4 Max 同一 N・fresh モードの値（512: 359.4 GFLOP/s・1024: 602.7 GFLOP/s・
2048: 800.8 GFLOP/s）と本計測の fresh 参考列の中央値から算出した GFLOP/s
（512: 368.8・1024: 614.6・2048: 743.1）を比べると、N=512/1024 はほぼ同水準
（僅かに上回る）、N=2048 のみやや低い。5 回計測へ拡張しても「未達」という
結論自体は環境 10/11 単発計測から変化しない。

## 6. #1117 受入条件との突合

| # | #1117 の受入条件（親 Issue） | 本 Issue（#1148）での対応 | 結果 | 出典 |
|---|---|---|---|---|
| 2 | 5 回計測中央値・checksum 複合判定 ok で再計測しゲート表へ反映 | DGX Spark・M4 Max とも実施 | **達成**（両実機・全 size で判定確定〈N=2048@DGX は判定不能〉） | §4・§5 |
| 3 | 未達が残る場合は原因分析を `docs/perf/` へ記録 | §8 の未達原因分析を本ドキュメントへ記録 | 達成 | §8 |
| （参考）R3 | DGX N=2048 candle 無効データの再現確認 | 5 回すべてで完全決定的に再現確認 | **達成** | §5.2 |

**総合判定: DGX Spark・M4 Max とも N=512/1024 は未達成（未達 4 件）。DGX Spark
N=2048 は candle 側要素誤差超過により判定不能（tolerance は緩めていない）。
M4 Max N=2048 は未達成。#1117「reuse で candle 超え」は判定可能な全 5 件で
未達成、1 件（DGX N=2048）は判定不能。**

## 7. `results/summary.md`・`performance-targets.md` への反映

- `results/summary.md` 環境 14（DGX Spark GB10）・環境 15（Apple M4 Max）を
  新設し 5 回計測ゲート判定表・fresh 参考列・データ有効性・#1117 ゲート判定
  総括を記載した（本 PR に含む）
- `docs/performance-targets.md` §8.4「#1148 追補」（§2 段階的下限表・§3 丸め規則
  は不変）
- `docs/perf/cpu-gemm-candle-cpu-retune.md` §8「#1148 への引き継ぎ」末尾に本
  ドキュメントへの参照を追記
- `docs/perf/oss-gemm-comparison-baseline.md` §7.3 に本キャンペーンの参照行を追記
- `docs/perf/gemm-optimization-baseline.md` §6 に本ドキュメントへの参照 1 行を追記

## 8. 未達原因分析

### 8.1 計測境界固定費

§5.4 のとおり、CPU では fresh→reuse のモード差（tape 構築・N² コピー 2 回の
排除）が性能に与える影響は数 % 程度と小さく、GPU 系で観測されるような明確な
改善効果は見られない。一方 `docs/perf/cpu-gemm-candle-cpu-retune.md` §5 記入表・
§5.1〜§5.2 に記録された「カーネル単体（`RowPanel`。A/B ハーネス self 計測）」の
GFLOP/s と本計測の framework-compare 境界値を比べると:

- **M4 Max**: N=1024 で 743.8（カーネル単体）対 567.0（本計測 reuse。§4.2）、
  N=2048 で 851.8 対 712.9（約 24%・約 16% 低い）
- **DGX Spark**: N=1024 で 536.2（カーネル単体）対 303.1（本計測 reuse。§4.1）、
  N=2048 で 701.6 対 491.6（約 43%・約 30% 低い）

いずれも明確な差があり、facade/autodiff 経由の呼び出しオーバーヘッド・readout
コピー・checksum 計算が reuse 計測境界に一定量残っていることを示唆する。
この計測境界固定費は GB10 側でより顕著（環境 10 単発 fresh との対比では
1024 で 536.2 対 279.3 と約 48% もの差。retune §5 記入表）であり、facade
呼び出し境界の効率化は #1148 のスコープ外の別調査候補（§10）とする。
**この「推定」は #1292（§16）で reuse 1 反復の内訳を両実機 5 回計測中央値で
実測し、固定費の内訳（ハーネス診断コスト対本番経路固定費）・優先順位として
確定した。**

### 8.2 並列化（スレッド数スイープ）

`scripts/bench/oss-gemm-compare`（`RAYON_NUM_THREADS` を固定し各 3 回計測。
coding-rust.md の「5 回計測中央値」原則に対し、計測時間短縮のため 3 回へ
簡略化。ノイズを含む参考値として扱う）による N=1024 の `self_gemm_blis_parallel`・
gemm crate 双方の TFLOP/s 中央値:

**M4 Max（P コア 12・E コア 4。計 16 論理コア）**

| RAYON_NUM_THREADS | self_gemm_blis_parallel | gemm crate | 対 1 スレッド倍率（self） |
|---|---|---|---|
| 1 | 0.1116 | 0.1091 | 1.00 |
| 2 | 0.2154 | 0.2067 | 1.93 |
| 4 | 0.3810 | 0.3745 | 3.41 |
| 8 | 0.6969 | 0.7312 | 6.25 |
| 12 | 0.6061 | 0.6850 | 5.43（8 スレッドを下回る） |
| 16 | 0.6818 | 0.7640 | 6.11 |

**DGX Spark GB10（Cortex-X925 ×10 + Cortex-A725 ×10。計 20 論理コア）**

| RAYON_NUM_THREADS | self_gemm_blis_parallel | gemm crate | 対 1 スレッド倍率（self） |
|---|---|---|---|
| 1 | 0.1299 | 0.1253 | 1.00 |
| 2 | 0.2311 | 0.2231 | 1.78 |
| 4 | 0.4489 | 0.4093 | 3.46 |
| 8 | 0.6711 | 0.7203 | 5.17 |
| 10 | 0.3080 | 0.5732 | 2.37（8 スレッドを大きく下回る） |
| 16 | 0.4926 | 0.4971 | 3.79 |
| 20 | 0.5347 | 0.5987 | 4.12 |

両実機とも 1→8 スレッドまではほぼ線形にスケールする一方、「大コア数」
（M4 Max の P コア数 12 の近傍・DGX の X925 コア数 10）付近でスループットが
一旦大きく落ち込み、全論理コア数まで増やすと部分的に持ち直すという非単調な
挙動が観測された。特に DGX の `RAYON_NUM_THREADS=10` での落ち込みは
`self_gemm_blis_parallel` で 8 スレッド比 54% と大きい。これは静的等分割
行パネル分割（`gemm_blis_parallel` の `c.par_chunks_mut(panel_rows * n)`。
work stealing なし）が、P/E（big/little）異種コア構成のマシン上でスレッド数を
増やすほど「遅い little コアに割り当たった行パネルがクリティカルパスになる」
影響を受けやすいという仮説と整合する挙動である。ただし本計測は 3 回計測・
共有マシン上（M4 Max は他エージェント稼働中〈§8.5〉。DGX の gate 計測前後の
load average は 5.07/1.80/0.64 → 5.06/2.31/0.88〈`env_info.txt`〉で、1 分平均
5 前後は gate 計測自身の bench-fandhe/bench-candle プロセスによるもの。
RAYON スイープ実行中の load average は記録していないため、スイープ実施
時点の背景負荷の有無は不明）で行っており、この非単調性が異種コア仮説由来か
背景負荷由来かを完全には
切り分けられていない。gemm crate（`Parallelism::Rayon(0)`。同じ共有 rayon
プールを使う）でも同様の非単調性が見られる（DGX 10 スレッドで 8 スレッド比
80%）ことから、少なくとも一部は rayon の work-stealing スケジューラ自体が
背景負荷の強い環境・異種コア環境で示す挙動である可能性が高く、
`gemm_blis_parallel` 固有の静的分割のみが主因とは断定できない。

**専有環境での再実測・切り分け結果は §17（イシュー #1305）を参照**。
DGX Spark GB10 の N=1024（T=10）非単調性は taskset pin 実験により
**H1（異種コア由来）と確定**した。Apple M4 Max は専有ゲートが最後まで
通過せず undetermined のまま（§17.7）。

### 8.3 マイクロカーネル効率

`RAYON_NUM_THREADS=1` での `self_gemm_blis_parallel` 中央値は両実機とも
gemm crate とほぼ同水準（M4 Max: 0.110〜0.116 対 0.106〜0.109 TFLOP/s、DGX:
0.130 対 0.125 TFLOP/s。NEON マイクロカーネル MR=8×NR=12 の効率自体に大きな
差はない）。`docs/perf/cpu-gemm-candle-cpu-retune.md` §2 のコストモデルが
指摘する packing 重複コストは並列時（複数スレッドが同じ B パネルを重複
packing）にのみ効いてくるため、単スレッド計測ではその影響が現れにくい。

### 8.4 packing

新規計測は行わず、`docs/perf/cpu-gemm-candle-cpu-retune.md` §2 のコストモデル
と §8 の `SharedB`／`SharedBPcOuter` 非採用結果を引用する。同ドキュメントの
実測（GB10・M4 Max とも）では B packing 共有化候補が現行 `RowPanel` を
一貫して下回っており（M4 Max: 1024 で約 33〜35%・2048/4096 で約 22〜24% 低い。
GB10: 1024/2048 で約 45〜54% 低い）、packing の重複コスト自体は存在するものの、
共有化のオーバーヘッド（排他制御・キャッシュ局所性の悪化）がその削減効果を
上回っていることを示唆する。KC=256 固定・`cache_params`（実行時キャッシュ検出）
未結線（#753・#1027）も未探索の要因として残る。

### 8.5 要因の寄与順位と次候補

観測された事実から以下の優先順位で寄与を推定する:

1. **計測境界固定費（§8.1）**: framework-compare 境界とカーネル単体の差が
   最大（M4 Max で 16〜24%・GB10 で 31〜48%）で、GEMM カーネル自体の改善では
   解消できない構造要因。facade/autodiff 呼び出し境界の効率化調査が候補
   （**内訳確定は §16。ハーネス診断コスト〈host_copy／checksum〉と本番経路
   固定費〈autodiff オーバーヘッド・alloc_c〉の切り分け・削減優先順位を記載**）
2. **並列化の非単調性（§8.2）**: 両実機で観測された「大コア数付近の落ち込み」
   は P/E 異種コア構成での静的等分割行パネル分割が疑わしいが、背景負荷ノイズ
   （M4 Max は他エージェント稼働中。DGX はスイープ実行中の load average を
   記録しておらず背景負荷の有無が不明）と
   完全には分離できていない。専有環境での再計測（または work-stealing 分割
   への変更検討）が必要。**専有環境再計測は §17（イシュー #1305）で実施済み**:
   DGX Spark GB10 の N=1024（T=10）非単調性は taskset pin 実験により H1
   （異種コア由来）と確定。work-stealing 分割への変更検討（2D 動的分配）は
   #1307 が引き継ぐ
3. **マイクロカーネル効率（§8.3）**: 単スレッドでは両実機とも gemm crate と
   同水準のため優先度は低い
4. **packing（§8.4）**: `cpu-gemm-candle-cpu-retune.md` §8 で候補 1〜3（B 側
   laneq ベクトル転置化・prefetch・KC 再スイープ）として既に整理済み。本 Issue
   の分析はこの優先順位を変更しない

## 9. スコープ外事項（本 PR では対応しない）

- **facade/autodiff 呼び出し境界の効率化調査**（§8.1・§8.5 候補 1）: 別スコープ
  の設計変更が必要
- **静的等分割行パネル分割の work-stealing 化**（§8.2・§8.5 候補 2）: `gemm_blis_
  parallel` の並列分割方式自体の変更であり、性能影響・bit 完全一致契約への
  影響を含め別 Issue でのスコープ
- **`cpu-gemm-candle-cpu-retune.md` §8 の次候補 1〜3**（B 側 laneq ベクトル
  転置化・prefetch・KC 再スイープ）: 同ドキュメントで既に整理済みの追跡事項
  であり本 PR の対象外
- **RAYON スイープの専有環境での再計測**（coding-rust.md「5 回計測中央値」
  原則の完全遵守を含む）: 本計測は 3 回・背景負荷ありのため参考値にとどまる
- **DGX N=2048 candle 無効データの内部原因切り分け**（candle-core／gemm crate
  側の丸め順序差の特定。#1142 §5.3 と同じ判断で追加計装は入れない）

## 10. ユーザー判断事項

- **#1117 のクローズ可否**: 両実機で判定可能な全 5 形状が未達成、1 形状
  （DGX N=2048）が判定不能と確定した。クローズせず残課題として維持するか、
  達成条件・スコープの見直し（reuse 計測境界の再定義、転送・同期を除いた
  カーネル専有時間での判定への変更等）を検討するかはユーザー判断
- **次候補（§8.5・`cpu-gemm-candle-cpu-retune.md` §8）の Issue 化**:
  facade/autodiff 呼び出し境界の効率化・並列分割方式の見直し・B 側 laneq
  ベクトル転置化等を追跡する新規 issue を起票するかはユーザー判断
  （`out-of-scope-tracking.md` に従い、本 PR では Issue 操作を行わない）
- **N=2048 の DGX 側 candle 判定方式**: candle 側の決定的な要素誤差超過が
  確認された（§5.2）。tolerance 緩和は行わない前提で、判定方式自体の見直し
  （例: candle-core 側の既知の丸め誤差として記録した上で判定対象から除外する
  等）の要否はユーザー判断
  - **2026-09-08 更新（イシュー #1262）**: tolerance 契約変更（#1241 承認・
    #1443/#1445 実装）により、この判定不能は「判定方式の見直し」を待たずに
    解消し、確定判定（未達・0.950 倍）へ遷移した（§19）。上記のユーザー判断
    事項自体は #1241 承認の枠内で決着済みであり、本 PR では追加の判断待ちは
    ない
- **2026-09-06 更新（イシュー #1185）**: 正式系列 `fandhe-ai =0.7.0` でも両実機で
  未達成が確定した（§12）ことを受け、ユーザー指示（2026-09-06）「未達の場合は
  後継ツリーを新規起票し現 issue はクローズ」に従い、上記の残課題（§8.5 の次候補・
  DGX N=2048 の判定方式・達成条件の見直し）は後継ツリー
  #1283（CPU GEMM candle 超えトラッキング。DGX N=2048 判定不能は #1234 に依存） へ引き継ぎ、現 issue #1117 はクローズする

## 11. 関連ドキュメント

- `docs/perf/cpu-gemm-candle-cpu-retune.md`（#1144。SharedB/SharedBPcOuter
  非採用・本番結線しない判定・次候補の整理）
- `docs/perf/cuda-gemm-candle-gate-remeasurement.md`（CUDA 側の同型判定。#1142）
- `docs/perf/metal-gemm-candle-gate-remeasurement.md`（Metal 側の同型判定。#1147）
- `docs/perf/oss-gemm-comparison-baseline.md`（OSS 直接比較の再現手順・ベースライン）
- `scripts/bench/framework-compare/README.md`「GEMM ゲート 5 回計測」節
- `scripts/bench/framework-compare/results/summary.md` 環境 10/11/14/15 節
- `docs/performance-targets.md` §8/§8.4
- `docs/perf/logs/cpu-gemm-candle-gate-1148/`（実行ログ・env_info・RAYON スイープログ）
- `docs/perf/logs/gemm-candle-gate-0.7.0-1185/`（イシュー #1185。=0.7.0 正式系列
  再計測の実行ログ・env_info。CUDA / Metal / CPU 共通）
- `docs/perf/logs/cpu-gemm-rayon-sweep-1305/`（イシュー #1305。専有環境での
  `RAYON_NUM_THREADS` スイープ再実測の実行ログ・env_info・gate/uptime 記録・
  集計スクリプト）
- `docs/perf/logs/cpu-gemm-candle-gate-1262/`（イシュー #1262。承認済み tolerance 契約下の
  DGX N=2048 再計測の実行ログ・判定表出力・env_info）
- `docs/candle-parity-tolerance-contract-decision.md`（tolerance 契約変更〈候補 A-1〉の承認
  記録。イシュー #1241）
- `docs/perf/logs/cpu-gemm-candle-gate-1321/`（イシュー #1321。Phase 1〜3 結線後の両実機
  ゲート再判定の実行ログ・env_info・diff・attribution・gate/uptime ログ）

## 12. 2026-09-06 追補: 正式系列 `fandhe-ai =0.7.0` 再計測（イシュー #1185）

### 12.1 位置づけ・プロトコル

- v0.7.0 の crates.io 公開と framework-compare の承認ピン `fandhe-ai =0.7.0` 更新
  （PR #1233）を受け、**正式系列のみ**で N=512/1024/2048 reuse の 5 回計測中央値を
  DGX Spark GB10（Grace CPU）・Apple M4 Max の両実機で再取得した（CUDA 側 #1185 と
  同一プロトコルの CPU 版。§3 のとおり CPU は元より単一系列）
- プロトコルは §2 と同一（`run_gemm_gate_cpu.sh 0.7.0`〈DGX は
  `GEMM_GATE_CPU_NODE_TAG=dgx-cpu`〉・`compare_gemm_gate.py --device cpu`。manifest で
  `fandhe_ai_source=registry`・`candle_core_source=registry` を確認済み）。v0.6.0 →
  v0.7.0 の `crates/backend-cpu/src` 変更（`git log v0.6.0..v0.7.0`: #1174〈docs/
  コメントのみ。`RowPanel` 維持を確定〉・#1225〈VJP 専用 NT/TN 入口の追加〉）に NN
  正方 GEMM の reuse 経路を対象とした性能変更は含まれない
- 計測環境（`docs/perf/logs/gemm-candle-gate-0.7.0-1185/env_info.txt`）:
  - DGX Spark: rustc 1.97.0。CUDA ゲート直後に同一シェルで直列実行。計測直前の
    load average 0.04（他負荷なし）
  - M4 Max: rustc 1.96.0・macOS 26.6.2。thermal / performance warning なし。ただし
    他セッションの cargo ビルドが並走する**共有マシン状態**（load average: Metal
    ゲート完了時 7.09 → CPU ゲート完了時 7.96）で計測しており、絶対値には背景負荷の
    ノイズが乗る（0.6.0 系列〈`results/summary.md` 環境 15。計測前 7.49・計測後 9.30〉も同様に非専有）
- 生データ: `scripts/bench/framework-compare/results/raw/results-{dgx,m4max}-cpu-gemm-gate-0.7.0.jsonl`
  （各 45 行）・`skipped-{dgx,m4max}-cpu-gemm-gate-0.7.0.log`（空）・
  `manifest-{dgx,m4max}-cpu-gemm-gate-0.7.0.json`。実行ログ:
  `docs/perf/logs/gemm-candle-gate-0.7.0-1185/run_gemm_gate_cpu-{dgx,m4max}-0.7.0.log`

### 12.2 実測結果（正式系列 `0.7.0`）

DGX Spark GB10（Grace CPU）:

| N | fandhe-ai reuse median (min–max, n=5) | candle fresh median (n=5) | candle/fandhe | GFLOP/s（fandhe） | 判定 | fandhe-ai fresh median（参考。n=5） |
|---|---|---|---|---|---|---|
| 512 | 2.280 ms (2.167–2.494 ms) | 1.847 ms | 0.810 | 117.8 | 未達 | 2.402 ms |
| 1024 | 7.063 ms (6.875–7.171 ms) | 5.551 ms | 0.786 | 304.1 | 未達 | 7.757 ms |
| 2048 | - | - | - | - | **判定不能**（candle 側要素誤差超過。§12.3） | - |

Apple M4 Max:

| N | fandhe-ai reuse median (min–max, n=5) | candle fresh median (n=5) | candle/fandhe | GFLOP/s（fandhe） | 判定 | fandhe-ai fresh median（参考。n=5） |
|---|---|---|---|---|---|---|
| 512 | 741.9 µs (729.2–779.9 µs) | 684.0 µs | 0.922 | 361.8 | 未達 | 746.4 µs |
| 1024 | 3.676 ms (3.581–3.729 ms) | 2.860 ms | 0.778 | 584.2 | 未達 | 3.465 ms |
| 2048 | 24.578 ms (24.362–24.756 ms) | 20.369 ms | 0.829 | 699.0 | 未達 | 23.646 ms |

fandhe-ai 側は両実機とも全 45 run（reuse 15・candle fresh 15・fandhe fresh 15）で
`parity_fail_count=0`。M4 Max は candle 側も全 run で 0 fail。tolerance は緩めていない。

0.6.0 正式系列（§4）との対比: DGX は 0.760／0.791 → 0.810／0.786、M4 Max は
0.940／0.726／0.734 → 0.922／0.778／0.829。NN 経路を対象とした性能変更が無いため
（§12.1）、差は run 間ばらつき・背景負荷によるものと見て、コード変更への帰属は
行わない。

### 12.3 DGX N=2048 判定不能の再現

candle 側 N=2048 fresh は 5 run すべてで `parity_fail_count=2, parity_total=4194304,
parity_max_abs_err=3.814697e-05, parity_max_rel_err=3.944416e-01`（§5.2・環境 10・
0.6.0 系列と完全に同一の決定的な値。fandhe-ai 側は 0 fail）。原因は candle-core
0.11.0 の CPU GEMM カーネル側にあり（§5.2）、判定方式の変更は本追補でも実施せず
「判定不能」のまま据え置く（reuse/candle 比の参考併記も行わない）。**→ §19 で承認済み
tolerance 契約下の解消有無を記録した（イシュー #1262）。**

### 12.4 #1117 ゲート判定（確定）

| 実機 | N=512 | N=1024 | N=2048 | parity（fandhe-ai 側） |
|---|---|---|---|---|
| DGX Spark GB10（Grace） | 未達（0.810 倍） | 未達（0.786 倍） | 判定不能（candle 無効データ） | 達成（0 fail） |
| Apple M4 Max | 未達（0.922 倍） | 未達（0.778 倍） | 未達（0.829 倍） | 達成（0 fail） |

**総合判定: #1117 は正式系列 `fandhe-ai =0.7.0` においても両実機で未達成（判定可能な
5 件すべて未達・DGX N=2048 は判定不能）。#1148 の判定を 0.7.0 で確定した。** 達成条件の
見直し要否・後継ツリーへの引き継ぎは §10「2026-09-06 更新」を参照。

## 13. 2026-09-07 追補: 既定スレッド数の大コア限定 on/off 比較（イシュー #1364）

### 13.1 位置づけ

`docs/perf/cpu-gemm-default-thread-limit.md`（イシュー #1363・origin/main
`90ea1cb`）が実装した「`RAYON_NUM_THREADS` 未指定時の既定並列度を物理大コア数へ
限定する」制御について、§8.2 の `RAYON_NUM_THREADS` スイープが示した非単調性
（大コア数付近が谷）の仮説を、両実機・同一バイナリ（`GEMM_GATE_PATCH_FACADE_PATH`
の path patch。`crates/facade` HEAD＝#1363 反映後）・`RAYON_NUM_THREADS` の
on（未設定＝限定あり）/off（全論理コア数を明示＝限定なし）5 回計測中央値
比較で検証した。判定は `run_gemm_gate_cpu.sh` の出力を `compare_gemm_ab.py
--device cpu`（本イシューで N=512/1024/2048 の 6 セルに対応拡張）で突合する。

### 13.2 検出生値

| 実機 | current | detected_big_cores | effective（限定 ON 時） |
|---|---|---|---|
| Apple M4 Max | 16 | Some(12) | 12（正しく P コア数を検出） |
| DGX Spark GB10 | 20 | **Some(1)** | **1**（誤検出。§13.3 参照） |

DGX の `cpu_capacity` sysfs 値は `lscpu` が報告する「Cortex-X925 ×10 ＋
Cortex-A725 ×10」の 2 群構成と一致しない 5 段階の非一様分布
（718×5・997×5・731×5・1017×4・1024×1）であり、`big_cores_from_capacities`
の「最大値と一致するコア数」判定が cpu19 の 1 個のみを大コアとして誤検出した。

### 13.3 on/off A/B 結果（before=off・after=on・ratio=after/before）

Apple M4 Max（checksum 全セル完全一致）:

| N/mode | off median | on median | ratio | 判定 |
|---|---|---|---|---|
| 512/fresh | 739.5 us | 653.1 us | 0.8831 | 非後退（改善） |
| 512/reuse | 732.6 us | 670.2 us | 0.9149 | 非後退（改善） |
| 1024/fresh | 3.401 ms | 2.787 ms | 0.8193 | 非後退（改善） |
| 1024/reuse | 3.610 ms | 2.987 ms | 0.8275 | 非後退（改善） |
| 2048/fresh | 21.674 ms | 17.421 ms | 0.8038 | 非後退（改善） |
| 2048/reuse | 22.308 ms | 18.175 ms | 0.8147 | 非後退（改善） |

DGX Spark GB10（checksum 全セル完全一致）:

| N/mode | off median | on median | ratio | 判定 |
|---|---|---|---|---|
| 512/fresh | 2.272 ms | 3.068 ms | 1.3503 | 後退 |
| 512/reuse | 2.305 ms | 2.751 ms | 1.1936 | 後退 |
| 1024/fresh | 7.729 ms | 21.924 ms | 2.8368 | 後退 |
| 1024/reuse | 7.133 ms | 19.722 ms | 2.7648 | 後退 |
| 2048/fresh | 36.554 ms | 161.725 ms | 4.4243 | 後退 |
| 2048/reuse | 35.010 ms | 151.965 ms | 4.3406 | 後退 |

両実機とも各腕で `manifest-*.json` の `bench_fandhe_sha256` が完全一致（同一
バイナリの証拠）。fandhe-ai 側は全 run `parity_fail_count=0`。

### 13.4 §8.2 非単調性仮説との整合

M4 Max の結果は「大コア数へ限定すると改善する」という §8.2 の一部の観察
（8 スレッド→12 スレッドで谷、という非単調性のうち大コア数=12 が谷だった点とは
逆方向）と単純には一致しないが、これは §8.2 のスイープが**静的な固定値
指定**（`RAYON_NUM_THREADS=12`）である一方、本 A/B は`effective_num_threads`
の判定ロジック自体を経由した値であり、条件が異なるため直接比較はできない。
DGX の結果は「大コア数（0 判定不能）付近」ではなく「誤検出による実質 1
スレッド」という全く別の病態であり、§8.2 の非単調性仮説（little コア律速）を
検証したことにはならない。

### 13.5 #1117 ゲート（candle 比）への参考影響

on 腕（DGX, effective=1）は candle 比が著しく悪化した（例: N=1024
candle/fandhe=0.281。§12 の 0.6.0/0.7.0 系列の 0.786〜0.810 から大幅後退）。
これは thread_limit の誤検出の症状であり、#1117 の判定（§12.4）自体は
BIG_CORE_LIMIT_ENABLED=false の状態（本追補の off 腕相当）で行われている
ため §12 の結論に影響しない。

### 13.6 採否（確定）

**REJECT（不採用）**。決定規則（両実機・全判定可能セルで reuse の
ratio<=1.05 を要求）に対し、DGX の reuse 全セルが 1.19〜4.34 倍の重大な
後退を示したため、M4 Max 単独の改善（0.80〜0.91 倍）があっても総合で不採用
と判断した。`crates/backend-cpu/src/thread_limit.rs::BIG_CORE_LIMIT_ENABLED`
を `false` へ差し戻し済み（#1364 のコミット）。親 #1362 の受入条件を本追補で
充足した。

DGX（Linux 非対称コア構成）の検出手段の見直し（`cpu_capacity` 以外の指標）は
本イシューのスコープ外として記録する（ユーザー承認前提の Issue 起票は本
PR では行わない）。

### 13.7 出典

env_info・実行ログ・生データ:
`docs/perf/logs/cpu-gemm-thread-limit-1364/`（`env_info.txt`・
`thread_limit_report-{m4max,dgx}.log`・`run_gemm_gate_cpu-{m4max,dgx}-head-90ea1cb-limit-{on,off}.log`）、
`scripts/bench/framework-compare/results/raw/results-{m4max,dgx}-cpu-gemm-gate-head-90ea1cb-limit-{on,off}.jsonl`・
`manifest-{m4max,dgx}-cpu-gemm-gate-head-90ea1cb-limit-{on,off}.json`。
集計は `scripts/bench/framework-compare/results/summary.md` 環境 22 節を参照。

## 14. 2026-09-07 追補: RowPanel／IcDynamic／2D 動的の 3 variant 両実機比較（イシュー #1367）

### 14.1 位置づけ・採用ゲート（計測前に確定）

イシュー #1366（#1409）が追加した CPU GEMM 並列ドライバ候補
`GemmDriverVariant::IcDynamic`（`#[cfg(test)]` 限定・行パネルを
`AtomicUsize` で動的配布・`unsafe` なし）の実機性能を、本番既定
`RowPanel` と両実機（Apple M4 Max・DGX Spark GB10〈Grace CPU〉）で
比較し、本番結線の採否を判定する。2D 動的分配 variant（#1307／#1310／
#1311）は計画時点（2026-09-07）で設計・実装とも未着手（対応ブランチ・PR
なし）のため本イシューでは実装せず、比較表に「未実装（#1311 待ち）」列
として記録するに留める。

採用ゲート（計測前に確定。以後変更しない）:

1. **主判定（対 RowPanel。両実機とも必須）**: `IcDynamic` の 5 回中央値が
   N=1024・2048 の両方で `RowPanel` 以上（ratio ≥ 1.00）、かつ N=4096 で
   非劣化（ratio ≥ 0.95）
2. **両実機一致要件**: 1 を M4 Max・GB10 の両方で満たす場合のみ ADOPT。
   片方のみ満たす場合は REJECT
3. **参考判定（対 gemm crate）**: `oss-gemm-compare` との比はトラッキング
   （#1283／#1117）への参考値として併記する（`oss-gemm-compare` ハーネスは
   本番 `RowPanel` 経路〈`self_gemm_blis_parallel`〉のみを計測し variant
   選択に非対応のため、`IcDynamic` 単体の対 gemm crate 比は本イシューでは
   計測不能）
4. **bit 完全一致前提**: 各実機で `gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large`
   （`#[ignore]`・release）が pass することを採否判定の前提条件とする
5. 2D 動的 variant は未実装のため本ゲートの対象外

### 14.2 プロトコル

```
cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
  gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large --nocapture
cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
  gemm_blis_variant_ab_1024_2048 --nocapture   # 5 回独立プロセス
cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
  gemm_blis_variant_ab_4096 --nocapture        # 5 回独立プロセス
```

`RAYON_NUM_THREADS` は両実機とも未設定（既定で全論理コア。M4 Max 16・
DGX 20。`thread_limit::BIG_CORE_LIMIT_ENABLED=false`〈#1364 で REJECT
確定済み〉）。DGX 側は計測開始前に低負荷ゲート（1 分 load average < 6 を
2 回連続）を通過してから実行した。M4 Max 側は本マシン上で並列稼働する
他の Claude Code エージェントセッションが多数存在し、計測専有できる
低負荷窓が取れなかったため「共有負荷下」のまま実行した（計画 Step
1-1 の既定動作。ブロックしない）。詳細環境情報は
`docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/env_info.txt`。

### 14.3 bit 完全一致前提の結果

両実機とも `gemm_blis_ic_dynamic_matches_row_panel_bit_exact_large` が
pass（`docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/bit-exact-large-m4max.txt`・
`unit-test-dgx.txt`）。前提条件を満たす。

### 14.4 DGX Spark GB10 実測（5 回独立プロセス中央値）

| N | RowPanel | SharedB | SharedBPcOuter | IcDynamic | 2D 動的 | IcDynamic/RowPanel |
|---|---|---|---|---|---|---|
| 1024 | 530.338 | 215.851 | 248.020 | 332.159 | 未実装（#1311 待ち） | 0.6263 |
| 2048 | 699.913 | 359.418 | 375.050 | 594.350 | 未実装（#1311 待ち） | 0.8492 |
| 4096 | 1136.666 | 458.709 | 456.075 | 1111.147 | 未実装（#1311 待ち） | 0.9775 |

単位 GFLOP/s。生値・run 別内訳は `docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/aggregate.py`
の実行結果（生ログ `ab-1024-2048-dgx-run{1..5}.txt`・`ab-4096-dgx-run{1..5}.txt`
から再計算可能）。

### 14.5 Apple M4 Max 実測（5 回独立プロセス中央値・共有負荷下）

| N | RowPanel | SharedB | SharedBPcOuter | IcDynamic | 2D 動的 | IcDynamic/RowPanel |
|---|---|---|---|---|---|---|
| 1024 | 635.868 | 483.663 | 462.928 | 628.141 | 未実装（#1311 待ち） | 0.9878 |
| 2048 | 736.337 | 624.481 | 611.747 | 742.778 | 未実装（#1311 待ち） | 1.0087 |
| 4096 | 812.951 | 626.581 | 631.080 | 902.014 | 未実装（#1311 待ち） | 1.1096 |

単位 GFLOP/s。生ログ `ab-1024-2048-m4max-run{1..5}.txt`・`ab-4096-m4max-run{1..5}.txt`。
共有負荷（load average 22.18〜31.11。16 論理コア）によるノイズが大きい点に
留意（例: RowPanel N=2048 は run 間で 609〜786 GFLOP/s の幅がある）。

### 14.6 対 RowPanel／対 gemm crate 比の総括

| 実機 | N=1024 | N=2048 | N=4096 | ゲート判定（rule 1） |
|---|---|---|---|---|
| DGX Spark GB10 | 0.6263（未達） | 0.8492（未達） | 0.9775（達成） | **未達成**（N=1024/2048 が 1.00 を大きく下回る） |
| Apple M4 Max | 0.9878（僅かに未達） | 1.0087（達成） | 1.1096（達成） | **未達成**（N=1024 が 1.00 未満） |

対 gemm crate（参考・M4 Max のみ 1 回計測。§14.1 注記のとおり `IcDynamic`
単体は計測不能のため本番 `RowPanel` 経路の値。**対 gemm crate の 5 回計測
中央値基準線は存在しない**（本ファイル §12 は candle-core 比のみを記録して
おり gemm crate 比のデータを含まない。既存の対 gemm crate 参考値は §8.2 の
`RAYON_NUM_THREADS` スイープ〈coding-rust.md の「5 回計測中央値」原則に
対し計測時間短縮のため各 3 回へ簡略化したノイズ含みの参考値。N=1024 限定〉
のみで、`docs/perf/logs/cpu-gemm-candle-gate-1148/` にも gemm crate 比の
記録はない）: N=1024 0.7595 対
0.7946 TFLOP/s（gemm crate 上回り）・N=2048 0.7657 対 0.8176（gemm crate
上回り）・N=4096（`gemm` 側 `output_match=false`。既知の丸め差
rel_diff≈0.0036 は fail-closed 仕様の想定内）0.7687 対 0.7543（fandhe
上回り）。いずれも `RowPanel` 経路の参考値であり `IcDynamic` の採否判定
には使わない。

### 14.7 採否（確定）

**REJECT（不採用）**。DGX Spark GB10 で N=1024（比 0.6263）・N=2048（比
0.8492）とも主判定を大きく下回り（測定ノイズでは説明できない一貫した
後退。5 run すべてで RowPanel が IcDynamic を上回る）、ゲート 2「両実機で
rule 1 を満たす場合のみ ADOPT」を満たさない。Apple M4 Max も N=1024 が
0.9878 で僅かに 1.00 未満（共有負荷下のノイズの範囲内である可能性はある
が、DGX 側が非後退ではなく明確な後退のため判定に影響しない）。

DGX での後退の推定要因: `IcDynamic` は pc ごとに B を列全幅 `n` で
1 回 pack し・行パネルを動的配布する設計だが、DGX の非一様コア構成
（Cortex-X925 ×10 + Cortex-A725 ×10。`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
§13 既出）・N が小さいほど pc 同期点（`AtomicUsize` 経由の動的配布・
`Mutex` スロット）のオーバーヘッドがカーネル計算時間に対し相対的に
大きくなることが考えられる（N=4096 では非劣化まで回復している傾向と
整合）。根本原因の追加切り分けは本イシューのスコープ外とする。

本番結線（`GemmDriverVariant::IcDynamic` の `#[cfg(test)]` 解除・
`unsafe` 非導入のまま到達可能化）は **行わない**。
`crates/backend-cpu/src/gemm_blis/mod.rs` の本番並列経路
（`gemm_blis_parallel_with_transpose`）・`crates/backend-cpu/src/lib.rs`
はコード変更なし。

2D 動的分配（#1307／#1310／#1311）は未実装のため本イシューの比較対象外。
親 #1365 の受入条件のうち 2D 動的分配分は本 PR では未充足のまま
`#1311`／`#1312` へ引き継ぐ。

### 14.8 スコープ外事項（本追補では対応しない）

- 2D 動的分配 variant の実装（#1311）・両実機比較（#1312）
- N 極大時（例 65536）の `IcDynamic` B footprint の実測（`docs/perf/cpu-gemm-ic-dynamic-variant.md`
  §2.4 の見積りのみ）
- DGX での後退要因（pc 同期点オーバーヘッド仮説）の追加診断
- `oss-gemm-compare` ハーネスへの variant 選択オプション追加（`IcDynamic`
  単体の対 gemm crate 比を得るため）

### 14.9 出典

- イシュー #1366／#1409（`IcDynamic` 実装）・#1367（本追補）・#1365（親）・
  #1307／#1310／#1311（2D 動的分配・未実装）
- `docs/perf/cpu-gemm-ic-dynamic-variant.md`（設計・回帰テスト詳細）
- `docs/cpu-gemm-b-packing-sharing-decision.md` §F
- `docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/`（本追補の生ログ・env_info・集計スクリプト）
- `.claude/rules/coding-rust.md`（bit 完全一致契約）・`.claude/rules/security.md`（unsafe 非導入）

## 15. 2026-09-07 追補: 借用ビュー readout 切替前後の両実機比較（イシュー #1337）

### 15.1 位置づけ・プロトコル

- 実装本体は PR #1411（`3d5e833`。#1334 ツリー）で完了済み。`bench-fandhe` の `readout_var`
  等を cargo feature `host-view-readout`（既定 OFF）で借用ビュー readout 経路へ切替可能に
  した。本節はその効果を DGX Spark GB10（Grace CPU）・Apple M4 Max の両実機で切替前後
  5 回中央値として実測した記録（CUDA は #1360、Metal は §12 参照）
- crates.io 承認ピンには `host-view-readout` が使う API が未収録のため、正式系列（registry
  ピン）は計測せず**参考系列のみ**で行う。正式判定（§12）は不変
- 転送元 sha: `1c298ff5641b948dae3c1c65699930054af8f747`（PR #1411 より後、PR #1420 まで含む
  HEAD）。ラベル: `head-1c298ff-readout-off`／`head-1c298ff-readout-on`
- **off/on 間でソースが揃っていない（重要な限定）**: manifest 実測（
  `docs/perf/logs/gemm-candle-gate-readout-1337/run_gemm_gate_cpu-{dgx,m4max}-readout-
  {off,on}.log` の `依存元検証 OK` 行）で確認すると、両実機とも off 腕は
  `fandhe_ai_source=registry`（`GEMM_GATE_PATCH_FACADE_PATH` 未使用。crates.io
  `fandhe-ai =0.7.0`）、on 腕のみ `fandhe_ai_source=path:<facade 絶対パス>`（HEAD
  `1c298ff...`）＋ `GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout` である。off/on の
  差分には readout feature の効果に加え、v0.7.0 公開後にマージされたコード差分（HEAD と
  registry の乖離。CPU GEMM 本番経路自体は §0 のとおり v0.6.0 以降不変だが、ハーネス側
  〈`bench-fandhe`〉やその他経路の差分は排除できない）が混入しており、**以下 §15.2〜§15.5
  の off/on 比較・「readout-on で改善／達成」という記述は #1337（readout 単独）への効果と
  しては厳密には分離帰属できない**。同一 HEAD source（path 差し替え）での off 腕再計測は
  本イシューのスコープ内で追加実施していない
- プロトコルは §3 と同一（`GEMM_GATE_CPU_NODE_TAG=dgx-cpu|m4max-cpu run_gemm_gate_cpu.sh`）に
  加え、on 腕は `GEMM_GATE_PATCH_FACADE_PATH=<facade 絶対パス>
  GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout` を付与。効果分離には
  `compare_gemm_ab.py --device cpu --sizes gate`（既定 `--modes fresh,reuse`。CPU ゲートは
  fandhe fresh 参考行も発行するため既定のまま）を追加使用
- 実機構成:
  - DGX Spark GB10: Grace CPU 20 コア（aarch64）・rustc 1.97.0。計測を通じ `uptime` load
    average 1 桁台前半・他ジョブ混入なし（`docs/perf/logs/gemm-candle-gate-readout-1337/
    env_info.txt`）
  - Apple M4 Max: 16 コア（12P+4E）・macOS 26.6.2・rustc 1.96.0。**§12（Metal）と同じ共有
    マシン状態**で、off 腕より on 腕のほうが高負荷（`uptime` load average: CPU off 完了時
    21.29/14.55/11.77 → CPU on 完了時 30.73/18.73/13.62）。片方向の負荷差のため、後述の
    改善が readout 経路そのものに起因するか負荷ノイズの影響を差し引いた上での改善かは
    本節単独では完全には切り分けられない
- 生データ: `scripts/bench/framework-compare/results/raw/{results,skipped,manifest}-{dgx-cpu,
  m4max-cpu}-gemm-gate-head-1c298ff-readout-{off,on}.{jsonl,log,json}`（各 45 行・
  `skipped-*.log` 空）。実行ログ・env_info: `docs/perf/logs/gemm-candle-gate-readout-1337/`

### 15.2 実測結果（DGX Spark GB10・Grace CPU）

| N | readout-off 中央値（min–max, n=5） | readout-on 中央値（min–max, n=5） | candle fresh 中央値（n=5） | off の candle 比 | on の candle 比 | off 判定 | on 判定 |
|---|---|---|---|---|---|---|---|
| 512 | 2.360 ms（2.261–2.612 ms） | 2.362 ms（2.248–2.466 ms） | 1.757 / 1.770 ms | 0.745 | 0.750 | 未達 | 未達 |
| 1024 | 7.120 ms（6.982–7.638 ms） | 6.582 ms（6.391–6.864 ms） | 5.554 / 5.530 ms | 0.780 | 0.840 | 未達 | 未達 |
| 2048 | - | - | - | - | - | 判定不能 | 判定不能 |

N=2048 は candle 側が両腕とも 5 run 決定的に `parity_fail_count=2, max_abs_err=3.814697e-05,
max_rel_err=3.944416e-01`（§5.2・§12.3 と同一の既知事象。tolerance は変更していない）。
fandhe-ai 側は全 30 run（3 サイズ×5 run×2 腕）で `parity_fail_count=0`。off/on の checksum は
全セル完全一致（`compare_gemm_ab.py` 出力。§15.4）。

**DGX 側は N=512/1024 とも readout-on の中央値がわずかに小さい（§15.1 の限定により readout 単独の改善とは断定しない。off/on の fandhe reuse 中央値: 512 は
ほぼ同値、1024 は 7.120 ms → 6.582 ms・0.92 倍）したが、candle 比ではいずれも未達のまま**（低
負荷環境での計測にもかかわらず #1117 ゲートは達成していない）。

### 15.3 実測結果（Apple M4 Max）

| N | readout-off 中央値（min–max, n=5） | readout-on 中央値（min–max, n=5） | candle fresh 中央値（n=5） | off の candle 比 | on の candle 比 | off 判定 | on 判定 |
|---|---|---|---|---|---|---|---|
| 512 | 1.161 ms（0.913–1.303 ms） | 1.023 ms（0.881–12.547 ms） | 1.056 / 1.264 ms | 0.909 | **1.236** | 未達 | **達成** |
| 1024 | 5.024 ms（4.879–5.662 ms） | 5.748 ms（4.242–19.758 ms） | 5.055 / 6.573 ms | 1.006 | **1.144** | 達成 | **達成** |
| 2048 | 38.176 ms（35.643–76.549 ms） | 39.568 ms（36.452–53.048 ms） | 32.661 / 43.175 ms | 0.856 | **1.091** | 未達 | **達成** |

fandhe-ai 側は全 30 run で `parity_fail_count=0`。off/on の checksum は全セル完全一致。

**M4 Max 側は readout-on で 3 形状とも `candle/fandhe >= 1.0`（達成）** となった。ただし on 腕
は off 腕より高負荷な共有マシン状態下（§15.1）であり、この達成が readout 単独の効果か負荷
ノイズの影響かは切り分けられない。当初「高負荷は fandhe 側を不利にするだけなので過大評価
方向のバイアスではない」と推定していたが、これは誤りである: N=2048 では fandhe 自体も
readout-on（高負荷側）で 38.176 ms → 39.568 ms と後退している一方、candle 側は 32.661 ms →
43.175 ms とそれ以上に悪化しており、結果として `candle/fandhe` 比は 0.856 → 1.091 と改善
（達成側へ）している。これは高負荷が fandhe・candle の双方を遅くしつつ candle 側により
強く効くことで比率を押し上げうる（過大評価）ことを示しており、高負荷が過小評価にしか
働かないとは言えない。したがって本節の達成判定は**負荷ノイズによる過大評価・過小評価の
いずれの可能性も排除できない**まま報告する（低負荷環境での再確認が必要。§15.6）。一方
min–max 幅が広い run（512/reuse の max 12.547 ms・1024/reuse の max 19.758 ms・2048/fresh
の min 29.989 ms 等）が混在しており、5 run 中央値としての判定は成立するが背景負荷の影響は
無視できない

### 15.4 readout 切替効果（`compare_gemm_ab.py --device cpu --sizes gate`）

DGX（低負荷環境。信頼度が高い）:

```
| size/mode | before(off) median | after(on) median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 2.211 ms | 1.869 ms | 0.8454 | 完全一致 | 非後退 |
| 512/reuse | 2.360 ms | 2.362 ms | 1.0010 | 完全一致 | 非後退 |
| 1024/fresh | 7.749 ms | 5.403 ms | 0.6973 | 完全一致 | 非後退 |
| 1024/reuse | 7.120 ms | 6.582 ms | 0.9245 | 完全一致 | 非後退 |
| 2048/fresh | 36.531 ms | 29.285 ms | 0.8016 | 完全一致 | 非後退 |
| 2048/reuse | 35.447 ms | 30.436 ms | 0.8586 | 完全一致 | 非後退 |
```

M4 Max（片方向の負荷差あり。§15.1 参照）:

```
| size/mode | before(off) median | after(on) median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 1.028 ms | 1.624 ms | 1.5789 | 完全一致 | 後退 |
| 512/reuse | 1.161 ms | 1.023 ms | 0.8811 | 完全一致 | 非後退 |
| 1024/fresh | 5.128 ms | 10.640 ms | 2.0749 | 完全一致 | 後退（判定注意: before spread > 1.5x） |
| 1024/reuse | 5.024 ms | 5.748 ms | 1.1441 | 完全一致 | 後退 |
| 2048/fresh | 35.676 ms | 43.780 ms | 1.2272 | 完全一致 | 後退 |
| 2048/reuse | 38.176 ms | 39.568 ms | 1.0365 | 完全一致 | 非後退（判定注意: before spread > 1.5x） |
```

- **DGX（低負荷）は `reuse` 列が全 fresh/reuse セル非後退**（fresh 列も含め改善または同等）。
  DGX は低負荷環境かつ §15.1 の限定（off=registry／on=path のソース差）が残るため、この
  非後退がハーネス側の `host_copy`（memcpy）削減単独の効果であるとは断定しない（readout
  feature とソース差の複合効果である可能性を排除できない）
- **M4 Max（片方向負荷差あり）は `reuse` 判定が 512/2048 で非後退、1024 のみ「後退」**表示
  だが、§15.3 のとおり `candle/fandhe` 比では 3 形状とも達成しており、`compare_gemm_ab.py`
  の非回帰判定（before との単純比較）と candle 比ゲート判定は独立の指標であることに注意
  （前者は「同一マシン内の readout 前後」、後者は「対 candle」の比較）
- checksum は DGX・M4 Max とも全セル完全一致（数値契約は不変）

### 15.5 #1117 ゲート判定への反映

| 実機 | N | 正式系列 `0.7.0`（§12） | 参考系列 readout-off（§15.2/15.3） | 参考系列 readout-on |
|---|---|---|---|---|
| DGX | 512 | 未達（0.810 倍） | 未達（0.745 倍） | 未達（0.750 倍） |
| DGX | 1024 | 未達（0.786 倍） | 未達（0.780 倍） | 未達（0.840 倍） |
| DGX | 2048 | 判定不能 | 判定不能 | 判定不能 |
| M4 Max | 512 | 未達（0.922 倍） | 未達（0.909 倍） | **達成（1.236 倍）** |
| M4 Max | 1024 | 未達（0.778 倍） | **達成（1.006 倍）** | **達成（1.144 倍）** |
| M4 Max | 2048 | 未達（0.829 倍） | 未達（0.856 倍） | **達成（1.091 倍）** |

**正式判定（registry ピン `fandhe-ai =0.7.0` に基づく §12）: #1117 は引き続き未達成（変更なし）**。
**参考系列（次回ピン更新後の見込み値）**: DGX は readout on/off いずれも全形状未達のまま
（N=2048 は判定不能のまま）。M4 Max は readout-on で 3 形状とも達成する見込みだが、§15.1 の
off/on ソース差・片方向負荷差のいずれとも切り分けられておらず確度は限定的（低負荷・同一
ソースでの再確認が望ましい。§15.6）。負荷ノイズは過小評価・過大評価のいずれの方向にも
働きうる（§15.3 の N=2048 分析）ため、「達成見込み」は暫定値として扱う。DGX で改善が
小幅・M4 Max で改善が大きい非対称は、DGX CPU が並列度・NUMA/unified memory 構成上
`host_copy` の相対コストが元々小さい可能性を示唆するが、原因分析は本イシューのスコープ外
とする

### 15.6 公正性の論点・スコープ・ユーザー判断事項（親 #1334 受け入れ条件）

- **公正性**: candle 側ハーネスは変更していない（`to_vec2` のまま）。読み出し経路は各
  ライブラリの公開 API の一部であり、fandhe-ai 側の feature 切替はハーネスの偏向ではなく
  製品側実装の測定である
- **#1336 の非到達**: CPU バックエンドは元々 `gemm` の戻り値がホストメモリ上にあり、CUDA
  pinned host staging（#1336）に相当する概念自体が存在しない。ただし §15.1 のとおり
  off/on 間でソース（registry ピン対 HEAD path）が揃っておらず、readout feature 以外の
  コード差分が混入しうるため、**本節の効果を `#1337`（借用ビュー readout）単独へ厳密に
  帰属することはできない**（#1336 概念の非到達は追加の交絡要因が無いことのみを意味する）
- **`host-view-readout` 既定化の可否**: 本節では判断しない。DGX（低負荷）は非後退だが
  candle 比ゲート未達のまま、M4 Max は達成の見込みだが負荷ノイズと完全には切り分けられて
  いないため、既定化するには (a) M4 Max の低負荷環境での再確認、(b) CUDA 側 N=1024/2048 の
  大幅後退（#1360 §12.7）の解消、の両方が前提になる
- **M4 Max 低負荷再計測**: 本イシューのスコープ外。新規 issue 起票はユーザー判断
## 16. 2026-09-08 追補: reuse 計測境界のフェーズ分解（イシュー #1292）

### 16.1 位置づけ・プロトコル

§8.1・§8.5 は framework-compare reuse 境界の GFLOP/s がカーネル単体
（`cpu-gemm-candle-cpu-retune.md` §5 の `RowPanel` 実測値）より低いことを
「facade/autodiff 呼び出しオーバーヘッド・readout コピー・checksum の固定費」
と**推定**したまま、CPU では reuse 1 反復のどこに固定費が乗るかを分解して
いなかった（CUDA は #1182、Metal は #1189 で分解済み。本節はその CPU 版）。

本追補は依存イシュー #1290 が追加した 2 層のハーネス（`README.md`「CPU での
区間定義と Layer B（イシュー #1290）」節）を用いて両実機を実測する:

- **Layer A**（`bench-fandhe --task gemm --device cpu --mode reuse --phases`。
  framework-compare 公開 API 境界）: `matmul`／`to_tensor`／`host_copy`／
  `checksum`／`iter_total` の 5 区間
- **Layer B**（`crates/backend-cpu` 内側診断テスト
  `gemm_reuse_phase_diag_cpu`）: `alloc_c`（出力バッファ確保）／`kernel`
  （`gemm_blis_parallel` カーネル本体）／`tensor_wrap`（`Tensor` 包装）／
  `ops_gemm`（`alloc_c+kernel+tensor_wrap` の本番合成レプリカ）／
  `tape_matmul`（Layer A `matmul` のレプリカ。autodiff tape 登録込み）／
  `to_tensor`／`host_copy`／`checksum` の 8 区間

系列は Layer A が crates.io 公開版 `fandhe-ai =0.7.0`（framework-compare の
承認済みピン。両実機で `cargo tree -p bench-fandhe --depth 1` = `fandhe-ai
v0.7.0`〈registry〉を確認）、Layer B が origin/main HEAD（`308c2eb`。#1290
マージ直後で本イシューはコード変更なし）。**突合前提の確認**として
`git diff v0.7.0..HEAD -- crates/backend-cpu/src crates/autodiff/src
crates/facade/src crates/tensor-core/src` を実施し、CPU 本番経路
（`GemmDriverVariant::RowPanel`。`gemm_blis/mod.rs` の
`GemmDriverVariant::RowPanel` 呼び出し）が不変であること、
`thread_limit::BIG_CORE_LIMIT_ENABLED = false`（#1364 の差し戻し後の値）で
あること、`IcDynamic`（#1366/#1409/#1367）が本番未結線であることを確認した
（差分自体は autodiff/tensor-core/facade/backend-cpu にまたがる約 2400 行
挿入だが、いずれも新規機構の追加であり `RowPanel` 経路の呼び出し形は変更
されていない）。`RAYON_NUM_THREADS` 未設定・`host-view-readout` feature
OFF（legacy `to_vec` 経路）・`--device-checksum`（#1339/#1415）不使用
（既定 `checksum_var`）は §2 の gate プロトコルと同一。

5 回独立プロセス起動・run 間中央値（`.claude/rules/coding-rust.md`）。
Layer A は N ごとに 5 プロセス × 3 N = 15 プロセス、Layer B は 1 プロセスで
N=512/1024/2048 を順に走らせるため 5 プロセス = 5 run。失敗 run はなし
（全 run 完走・除外なし）。

**共有負荷の注記**: M4 Max は本イシュー実行中も他セッションが並走
（`uptime-m4max.log`。19 users 常駐・load average が 4.9〜7.6 の平常域から
run 終盤（Layer B run 4/5）に一時的に約 30 まで急伸）しており、§4.2・
§12.2 と同じ「共有負荷下」の実測である。DGX Spark GB10 は本イシュー専用の
隔離ディレクトリ（`~/work/fc-1292/`。§14.2 と同じ方式で `rsync` 転送・
実測後削除）で計測し、DGX 側の `uptime`（`uptime-dgx.log`）は安定していた。

生ログ・env_info は `docs/perf/logs/cpu-gemm-reuse-phase-1292/` を参照
（AC-1）。

### 16.2 Layer A 実測（両実機・N=512/1024/2048）

5 run 中央値（`iter_total` に対する比率を併記）:

**Apple M4 Max**

| N | matmul | to_tensor | host_copy | checksum | iter_total |
|---|---|---|---|---|---|
| 512 | 591.4 µs (77.8%) | 0.1 µs (0.0%) | 15.4 µs (2.0%) | 147.3 µs (19.4%) | 759.8 µs |
| 1024 | 2.887 ms (77.3%) | 0.2 µs (0.0%) | 266.4 µs (7.1%) | 573.3 µs (15.3%) | 3.735 ms |
| 2048 | 19.087 ms (84.4%) | 0.3 µs (0.0%) | 1.115 ms (4.9%) | 2.370 ms (10.5%) | 22.608 ms |

**DGX Spark GB10（Grace CPU）**

| N | matmul | to_tensor | host_copy | checksum | iter_total |
|---|---|---|---|---|---|
| 512 | 1.850 ms (77.7%) | 0.0 µs (0.0%) | 372.5 µs (15.7%) | 134.8 µs (5.7%) | 2.380 ms |
| 1024 | 5.069 ms (71.6%) | 0.0 µs (0.0%) | 1.441 ms (20.3%) | 531.1 µs (7.5%) | 7.082 ms |
| 2048 | 26.056 ms (73.8%) | 0.1 µs (0.0%) | 6.020 ms (17.1%) | 3.007 ms (8.5%) | 35.285 ms |

両実機とも `matmul` が `iter_total` の 72〜84% を占め支配的。`to_tensor`
（`Var::from_tensor` 相当の薄いラップ）は無視できる大きさ（0.0%）。
`host_copy`（readout の `to_vec` コピー）・`checksum`（ホスト f64 逐次和）
は DGX で比率が顕著に高い（host_copy が N=1024/2048 で 17〜20%。M4 Max は
5〜7%）。これは §16.3 の `alloc_c` 異常（N=2048）とは独立の傾向で、DGX の
メモリサブシステム（unified memory・aarch64）側の特性を示唆するが、本
イシューでは原因の特定までは行わない（未特定）。

**AC-2（非 `--phases` の `gemm --mode reuse` 各 N 1 run）**: checksum は
phases 版と全 N・全実機で bit 一致（例: M4 Max N=1024 = -1855.597736）、
`parity_fail_count=0`（tolerance を緩めていない）。`layerA-ac2-{dgx,m4max}.jsonl`
参照。

`python3 summarize.py --strict layerA-phases-{dgx,m4max}.jsonl` は両ファイル
とも exit 0（各 N 5 run × 5 phase の順序・件数一致）。`compare_gemm_gate.py
--device cpu` は `task=gemm_phases` 行を「fandhe-ai reuse レコード件数 0 件」
として `判定不能` 扱いにするのみで、ゲート判定へは混入しない（既存契約の
確認。両ファイルで確認済み）。

### 16.3 Layer B 実測（両実機・8 区間）

5 run 中央値。`alloc_c`〜`checksum` の各列は「区間ごとに 5 run の値を
中央値化した値」（列単位の中央値）だが、右端の `Σ` 列は列単位の中央値を
合計した値**ではない**。`Σ` は「各 run 内で 6 区間（各区間は当該 run 内
20 trials の中央値）を合算した run ごとの合計値」をまず求め、その 5 run
分の合計値をあらためて中央値化した値（= 各 run 合計の中央値）である。
中央値は加法的でない（`median(a)+median(b)+... ≠ median(a+b+...)`）ため、
「列単位の中央値を合計した値」とは一般に一致しない（例: DGX N=1024 は
列単位の中央値を合計すると 6.7801 ms だが、run ごとの合計値
[6.7707, 6.7716, 6.9936, 6.9533, 6.8525] ms の中央値は 6.8526 ms であり
表の `Σ`=6.853 ms と一致する。生ログ `layerB-dgx-run{1..5}.log` の該当行
参照）。M4 Max のみ 1 run で負荷スパイクの影響を受けた値を含むが、`Σ`
自体は run 単位の中央値方式のため代表値への影響は限定的。原ログは
`layerB-{dgx,m4max}-run{1..5}.log`）:

**Apple M4 Max**

| N | alloc_c | kernel | tensor_wrap | ops_gemm | tape_matmul | to_tensor | host_copy | checksum | Σ |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 9.2 µs | 592.5 µs | 0.3 µs | 579.2 µs | 589.0 µs | 0.1 µs | 14.9 µs | 145.5 µs | 760.2 µs |
| 1024 | 21.3 µs | 3.237 ms | 0.8 µs | 3.448 ms | 3.379 ms | 0.1 µs | 294.7 µs | 636.3 µs | 4.187 ms |
| 2048 | 78.9 µs | 23.842 ms | 1.3 µs | 25.037 ms | 25.632 ms | 0.2 µs | 1.279 ms | 2.736 ms | 27.920 ms |

**DGX Spark GB10（Grace CPU）**

| N | alloc_c | kernel | tensor_wrap | ops_gemm | tape_matmul | to_tensor | host_copy | checksum | Σ |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 5.0 µs | 1.515 ms | 2.3 µs | 1.697 ms | 1.659 ms | 0.1 µs | 367.6 µs | 141.8 µs | 2.025 ms |
| 1024 | 24.3 µs | 4.725 ms | 3.1 µs | 5.422 ms | 5.349 ms | 0.1 µs | 1.485 ms | 543.0 µs | 6.853 ms |
| 2048 | **2.881 ms** | 25.611 ms | 5.4 µs | 26.872 ms | 26.653 ms | 0.3 µs | 5.625 ms | 3.007 ms | 37.082 ms |

`kernel` を `2N³/t` で GFLOP/s 化すると: M4 Max 512=453.1・1024=663.4・
2048=720.6、DGX 512=177.2・1024=454.5・2048=670.8 GFLOP/s。

**DGX N=2048 の `alloc_c` 異常値**: `alloc_c` は `vec![0.0f32; n*n]`
（出力 C バッファの新規確保）の計時のみであり、確保した領域への実書き込み
（first-touch）は別区間の `kernel`（`gemm_blis_parallel` カーネル本体が
実際に計算結果を書き込む）で発生する。ゼロ初期化の遅延ページ（OS の
zero-fill-on-demand。物理ページの確保・ゼロ化を実際の書き込み発生まで
遅延する）が使われている場合、`alloc_c` 区間では仮想アドレス空間の確保・
ゼロクリアの要求のみが行われ、物理ページのコミット（first-touch page
fault）自体は `kernel` 区間側に計上され得る（既存診断（`cuda-large-buffer
-percall-alloc-transfer-threshold.md` 等）の一般的な説明どおり）。本追補は
`alloc_c` に first-touch が含まれるとは断定せず、**観測事実**と**未検証の
仮説**を分離して記録する:

- **観測**: N=512/1024 では `alloc_c` が数〜数十 µs にとどまるのに対し、
  N=2048（16 MiB。4194304 要素 × 4 バイト）では 5 run とも一貫して
  2.8〜3.0 ms と 2 桁跳ね上がる（`layerB-dgx-run{1..5}.log` で全 5 run
  符号一貫）。M4 Max では同じ N=2048 で `alloc_c` は 78.9 µs にとどまり、
  この跳躍は DGX（Grace CPU・aarch64・unified memory）固有の傾向である
- **仮説（未検証）**: 上記の跳躍が `alloc_c` 区間内で実際に発生した
  first-touch ページフォルトによるものか、それとも純粋な確保系コスト
  （mmap 呼び出し自体・ゼロクリア命令）によるものかは、本追補のハーネスでは
  区別できない。前者であれば `kernel` 区間側の first-touch は既に発生済み
  という帰結になり、後者であれば `kernel` 区間側で改めて first-touch が
  発生している可能性がある

本イシューでは異常値の発生を確認するにとどめ、page fault 経路（透過的
ヒュージページ閾値・glibc mmap 閾値等）や `alloc_c`／`kernel` への帰属の
特定は行わない（未特定。`cuda-large-buffer-percall-alloc-transfer
-threshold.md` の 32 MiB mmap 閾値〈CUDA D2H〉と類似の現象だが、対象・
経路とも異なるため直接の関連は主張しない）。

**autodiff 残差**（`tape_matmul − ops_gemm`。tape 登録のオーバーヘッド）は
M4 Max で N=512 +9.8 µs・N=1024 −68.4 µs・N=2048 +594.5 µs、DGX で
N=512 −37.5 µs・N=1024 −73.8 µs・N=2048 −219.6 µs と、いずれも `kernel` の
ms オーダーに対し 1〜2 桁小さく符号も安定しない。M4 Max N=2048 の +594.5 µs
は §16.1 の共有負荷スパイク（Layer B run 4/5 で load average が約 30 まで
急伸）の影響を受けた run が中央値に混入した可能性が高く、autodiff の tape
登録自体のコストは両実機とも `alloc_c`／`host_copy`／`checksum` と比べて
無視できる水準と判断する。

### 16.4 突合

**(i) Layer A `matmul` vs Layer B `tape_matmul`（HEAD レプリカ）**:
DGX は概ね近い（N=1024: 5.069 対 5.349 ms・約 5% 差、N=2048: 26.056 対
26.653 ms・約 2% 差、N=512 は 1.850 対 1.659 ms・符号が逆で Layer A の方が
高い）。M4 Max は N=512/1024 が近い（591.4 対 589.0 µs、2.887 対 3.379 ms）
一方 N=2048 は 19.087 対 25.632 ms と大きく乖離する。Layer B の N=2048
`kernel` は 5 run 全て（21.3114 / 57.4900 / 27.2153 / 23.2684 /
23.8415 ms。中央値 23.8415 ms が表の値）が Layer A `matmul` の中央値
19.087 ms を上回っており、単一の外れ値だけでは説明できない。うち
57.4900 ms（`layerB-m4max-run2.log`。当該行を run3 と誤記していたのを本
修正で訂正）は他 run の 21〜28 ms から突出しており、§16.1 の共有負荷
スパイク（Layer B run 4/5 で load average が約 30 まで急伸）と時期が近い
ことから負荷アーティファクトの疑いがあるが、残る 4 run（21.3114〜
27.2153 ms）も一様に Layer A 側より高く、これだけでは全体の乖離を
説明しきれない。したがって本追補では「負荷アーティファクトの影響を
受けた可能性がある」という**仮説**にとどめ、乖離の全量を単一の外れ値・
load average のみに帰属させる断定はしない（未特定のまま記録）。

**M4 Max N=2048 は Layer B `kernel` 自体が Layer A `iter_total` を上回る**:
Layer B の N=2048 `kernel` 中央値 23.842 ms は、同一形状の Layer A
`iter_total`（`matmul`＋`to_tensor`＋`host_copy`＋`checksum` の合計）中央値
22.608 ms を上回る。`kernel` は本来 `matmul`（ひいては `iter_total`）に
包含される部分区間であり、包含関係が保たれるなら `kernel` が全体を
超えることはない。この逆転は Layer A・Layer B が別プロセス・別時点の
計測（§16.1 のとおり系列は同一 HEAD だが実行自体は独立）であるために
生じた計測条件差（負荷変動を含む）由来と考えられるが、原因は本追補では
未特定のまま記録する。この逆転がある以上、**M4 Max N=2048 について
Layer B（`kernel`・`ops_gemm`・retune baseline との比較）から導く結論は、
Layer A 単体で直接確認できる事実（`host_copy`・`checksum` の比率等）とは
異なり、あくまで仮説として扱う**（§16.5 で区別する）。

**(ii) `ops_gemm` vs `alloc_c+kernel+tensor_wrap`**: `ops_gemm` は
イテレーションごとに `alloc_c+kernel+tensor_wrap` を直接計測した独立の
区間であり、表の `alloc_c`／`kernel`／`tensor_wrap` 列（各区間を個別に
5 run 中央値化した値）の単純合計とはおおむね近いが厳密には一致しない
（median は非線形なため「個別区間の中央値の合計」と「合成区間自体の
中央値」は一般に一致しない。例: DGX N=2048 run1 では `alloc_c`
2.8355 ms + `kernel` 25.7089 ms + `tensor_wrap` 0.0052 ms = 28.5496 ms
に対し `ops_gemm` 自体の中央値は 27.1174 ms で、約 1.4 ms の差がある。
`docs/perf/logs/cpu-gemm-reuse-phase-1292/layerB-dgx-run1.log` 参照）。

**この `ops_gemm` は、表右端の `Σ` とは計測範囲が異なり直接比較できない
点に注意する。** `Σ` は `gemm_reuse_phase_diag_tests.rs::run_size` が
出力する `alloc_c＋kernel＋tensor_wrap＋to_tensor＋host_copy＋checksum`
の 6 区間について、各 run 内でこの 6 区間（各区間は当該 run 内 20 trials
の中央値）を合算した run ごとの合計値をまず求め、その 5 run 分の合計値を
中央値化した値である（§16.3 冒頭のとおり、区間ごとに先に 5 run 中央値化
してから合計した値とは一致しない。`ops_gemm`／`tape_matmul` は含まない）。
`ops_gemm` や `tape_matmul` の重複計上は無い。例えば DGX N=2048 は
`Σ`=37.082 ms に対し `ops_gemm`=26.872 ms であり、両者には約 1.38 倍の
差（10.210 ms）があるが、これは重複計上ではなく `Σ` が `ops_gemm`
（`alloc_c+kernel+tensor_wrap` の合成計測）を含まず、代わりに `alloc_c`・
`kernel`・`tensor_wrap` を個別区間として直接合算しているために生じる差
である。この 10.210 ms は単一の原因ではなく、性質の異なる 2 つの要素へ
分けられる:

- **主因（計測範囲の違い）**: `Σ` は `host_copy`（5.625 ms）・`checksum`
  （3.007 ms）の 2 区間を含むが、`ops_gemm` はこの 2 区間を含まない
  （`alloc_c+kernel+tensor_wrap` のみを計測する独立区間のため）。この
  差分は 5.625+3.007=**8.632 ms**（約 8.63 ms）で、10.210 ms のうち
  最大の寄与を占める
- **残差（中央値の非加法性・独立計測由来）**: 表の `alloc_c`（2.881 ms）・
  `kernel`（25.611 ms）・`tensor_wrap`（0.0054 ms）を単純合計すると
  **28.497 ms**（約 28.50 ms）となり、これは `ops_gemm` 自体の中央値
  26.872 ms と厳密には一致しない（差 **1.625 ms**・約 1.63 ms）。原因は
  §16.3 冒頭で述べた中央値の非加法性（`median(a)+median(b)+... ≠
  median(a+b+...)`）に加え、`alloc_c`／`kernel`／`tensor_wrap` が
  `Σ` 側では run 内 20 trials それぞれで 3 区間を個別計測した値である
  のに対し、`ops_gemm` はイテレーションごとに 3 区間分をまとめて 1 回で
  独立計測した値であり、両者は同一コード経路を指しつつも計測手続きが
  異なる（個別区間の直接比較ではない）ことにも起因する

8.632 ms（主因）＋1.625 ms（残差）＝10.257 ms は実測差 10.210 ms と
端数の丸めの範囲でおおむね一致する。**主因は `host_copy`／`checksum`
という計測範囲の違いであり、残差（中央値の非加法性・独立計測の手続き差）
はこれよりも一桁小さい**。本番経路の実コスト指標としては `ops_gemm`
（本番合成レプリカ）を用いるべきで、`Σ` をそのまま `ops_gemm` の代替や
本番経路コストの指標として扱わない。

**(iii) Layer A ⊃ Layer B の妥当性**: Layer A `iter_total` を `2N³/t` へ
換算した値は §12.2（イシュー #1185・`fandhe-ai =0.7.0` reuse）の GFLOP/s と
近い（M4 Max: 353.3 対 361.8〈N=512〉・574.9 対 584.2〈N=1024〉・759.9 対
699.0〈N=2048。ここも上振れは共有負荷変動によるもの〉、DGX: 112.8 対
117.8〈N=512〉・303.2 対 304.1〈N=1024〉）。同一プロトコル・別イシュー実行
間でおおむね整合しており、本追補の Layer A 計測自体の再現性を確認した。

### 16.5 §8.1 推定との突合

§8.1 は「カーネル単体（retune §5 の `RowPanel`）」対「本計測 reuse
（framework-compare 境界）」の差を M4 Max で 16〜24%・DGX で 31〜48% と
記録し、原因を「facade/autodiff 呼び出しオーバーヘッド・readout コピー・
checksum の固定費」と推定していた。本追補で分解した結果は次のとおり:

| 実機 | N | kernel 単体（本追補・GFLOP/s） | retune §5 RowPanel（GFLOP/s） | 比率 |
|---|---|---|---|---|
| M4 Max | 1024 | 663.4 | 743.8 | 89.2% |
| M4 Max | 2048 | 720.6 | 851.8 | 84.6% |
| DGX | 1024 | 454.5 | 536.2 | 84.8% |
| DGX | 2048 | 670.8 | 701.6 | 95.6% |

本追補の `kernel` 区間は診断ハーネス下（毎反復 `alloc_c` で新規 C ページを
確保する構成。§16.3 のとおり first-touch がどちらの区間に計上されるかは
未特定）の値であり、retune §5 の A/B ハーネス（C バッファを反復間で
使い回す想定）とは計測条件が異なる点に注意。

**この節の結論は、条件のそろった Layer A 単体から直接確認できる事実と、
条件の異なる Layer A/Layer B・retune baseline 間比較に基づく仮説とを
分けて記述する。** §16.4 のとおり M4 Max N=2048 は Layer B `kernel`
（23.842 ms）が同一形状の Layer A `iter_total`（22.608 ms）を上回るという
包含関係の逆転があり、Layer B（`kernel`・retune baseline）を参照する比較は
少なくとも M4 Max N=2048 について確度を主張できない。DGX（`kernel`
単体が Layer A `iter_total` を上回る逆転は観測されていない）・M4 Max
N=1024 以下は Layer B・retune baseline とも大きな矛盾は見られないが、
本追補ではこの逆転が生じた条件差自体を特定していないため、逆転が
生じていない形状・実機についても Layer B 由来の結論は同じ性質の仮説
として扱う。

- **仮説（Layer B・retune baseline との比較に基づく）**: `kernel` 単体の
  GFLOP/s は retune §5 RowPanel の 85〜96%（DGX N=2048 のみ）の範囲で
  近く、この近さが実機・形状全体で成り立つなら、§8.1 が観測した「本計測
  reuse」対「カーネル単体」の 16〜48% という差の大部分はカーネル自体の
  効率差ではなく `kernel` 区間の外側（`alloc_c`・autodiff 残差・
  `to_tensor`・`host_copy`・`checksum`）に乗っていると解釈できる。ただし
  上記の逆転により、この解釈を「本追補で確定した」とは言えず、次項の
  Layer A 直接確認分で裏付けられる範囲（`host_copy`／`checksum`）を除き
  仮説にとどめる

**確認済みの事実（Layer A 単体・比較なしで直接計測）**: `iter_total`
（Layer A）に対する寄与は:

- **M4 Max**: `matmul`（`kernel` 相当を含む本番経路）が 72〜84%（既に
  カーネル自体を含む）、`host_copy` が 2.0〜7.1%、`checksum` が
  10.5〜19.4%
- **DGX**: `matmul` が 72〜78%、`host_copy` が **15.7〜20.3%**（M4 Max の
  2〜3 倍）、`checksum` が 5.7〜8.5%

DGX は `host_copy`（readout コピー）の寄与が M4 Max より顕著に大きく、
これが §8.1 が観測した「DGX の方が固定費の影響が大きい（31〜48% 対
16〜24%）」の主因の一つであることが本追補で裏付けられた。加えて DGX
N=2048 固有の `alloc_c` 異常（§16.3。2.8〜3.0 ms）は Layer B（`ops_gemm`
本番合成レプリカ）にのみ現れる区間であり、Layer A の `matmul`（`tape_matmul`
相当）には C バッファ確保コストが同様に含まれるはずだが、Layer A 側の
`matmul` 自体は §16.4(i) のとおり `tape_matmul` と近い値のため、N=2048 の
`alloc_c` 跳躍は Layer A の `matmul` 内部にも既に含まれていると解釈するのが
自然である（Layer A は `matmul` 単体でしか計測しないため `alloc_c` を
単独区間として分離できない。この点は Layer A のハーネス設計上の制約として
記録するにとどめる）。

### 16.6 固定費の帰属と削減優先順位（AC-3）

観測事実から、固定費を次の 2 種類に区分する:

**A. ハーネス診断コスト（本番経路には乗らない。#965/#970 の既存契約どおり）**

- `host_copy`（readout の `to_vec` コピー）: DGX で `iter_total` の
  15.7〜20.3%・M4 Max で 2.0〜7.1%
- `checksum`（ホスト f64 逐次和）: 両実機で 5.7〜19.4%

`host_copy` と `checksum` の合計を §16.2 の Layer A `iter_total` に対する
比率として形状（N）ごとに再計算すると次のとおり（生ログは
`layerA-phases-{dgx,m4max}.jsonl`）:

| 実機 | N=512 | N=1024 | N=2048 |
|---|---|---|---|
| DGX | 21.4% | 27.8% | 25.6% |
| M4 Max | 21.4% | 22.4% | 15.4% |

合計は `iter_total` の **21.4〜27.8%（DGX）・15.4〜22.4%（M4 Max）**の
範囲であり、`iter_total` ベースで見た framework-compare reuse 境界の
「本番外」比率として最大の寄与を持つ。ただしこれは実運用コードパスの
オーバーヘッドではなく、比較ハーネス自身の診断コストである点に注意する。

この 2 区間はさらに性質が異なる: `host_copy`（readout の `to_vec` コピー）
は CPU バックエンドが GPU 同様にホスト側へ値を読み出すためのコピーで
あり、`device-checksum-readback-ab.md` の `device_checksum` feature（CPU で
既に bit 一致確認済み）を有効化すれば checksum 用の返却値を 8 バイトへ
削減できる余地がある。一方 `checksum`（全要素和を求めるホスト側 f64
逐次和そのもの）は CPU では元々 GPU の「readback」を経由しない計算
（GPU バックエンドは `readback` してから CPU 側で逐次和を取るのに対し、
CPU バックエンドは元から同一ホスト上のメモリに対して直接逐次和を計算
する）であり、`gemm_checksum`（デバイス側 f64 reduction）を使っても
「返却値を 8 バイトへ削減できる」のは reduction 自体を GPU 側で行う場合の
話であって、CPU 側では reduction の計算量（全要素を走査して f64 で
足し込む処理そのもの）は元々ローカルで行われており readback という
形では発生していない。したがって CPU においては `host_copy`（削減候補。
readout コピーの往復コストを避けられる）と `checksum`（残存計算。逐次和
自体の計算コストは `device_checksum` 化しても CPU 側で発生し続ける）を
分けて捉える必要がある（本追補では未実施のまま記録する）。

**B. 本番経路固定費（`Sequential::predict`／`Var::matmul` 経由の実運用で
実際に発生するコスト）**

優先順位（`iter_total` に対する寄与の大きい順）:

1. **`alloc_c`（出力バッファ確保）**: DGX N=2048 で 2.8〜3.0 ms（`iter_total`
   の約 8%）と突出。N=512/1024 では無視できる水準（DGX 5〜24 µs・M4 Max
   9〜21 µs）のため、大サイズ出力バッファの確保コスト（§16.3 のとおり
   first-touch が本区間・`kernel` 区間のいずれに計上されるかは未特定）が
   DGX 固有に顕在化する形状依存の問題。プール再利用（`tensor-core::pool`。
   `device-memory-pool-design.md` の CPU 版に相当する仕組み）による
   `alloc_c` 削減が候補になりうるが、本イシューでは設計・実装まで踏み込まない
2. **autodiff 残差（tape 登録オーバーヘッド）**: §16.3 のとおり両実機とも
   `kernel` の 1〜2 桁下で無視できる水準（M4 Max N=2048 の値は共有負荷
   アーティファクトの疑いが強い）。優先度は低い
3. **`materialize`／`tensor_wrap`**: `tensor_wrap` 区間は全 N・全実機で
   1〜5 µs と無視できる水準。優先度は最も低い

**結論**: `iter_total` を候補側（gemm crate・candle 等）と比べる際の
「固定費」削減で最も効果が見込めるのは、本番経路では発生しない
**ハーネス診断コスト（`host_copy`／`checksum`。特に DGX の `host_copy`）**
であり、これは #965/#970 の契約上あえて残している比較用の計測境界の一部
である（削減しても実運用の性能改善にはならない）。**実運用コードパスで
削減効果が見込める本番経路固定費は DGX N=2048 の `alloc_c`（出力バッファ
確保）が最有力候補**であり、後続イシュー（#1294）が調査候補として引き継ぐ。
autodiff・`tensor_wrap` は優先度が低い。#1294 の設計記録は
`docs/cpu-matmul-fixed-cost-design.md` を参照（呼び出しチェーン・変更案・
bit 一致契約への影響・期待削減量の上限を記載）。

### 16.7 スコープ外

- `alloc_c`（DGX N=2048 出力バッファ確保コスト）削減の設計・実装
- `device-checksum` feature を用いた `checksum` 区間削減の framework-compare
  ハーネスへの組み込み
- N=4096 での同型分解（`gemm_reuse_phase_diag_tests.rs` の `SIZES` 拡張が
  必要）
- `host-view-readout` feature on との比較（既定 OFF のまま計測）
- 本番結線を伴うコード変更（本追補は docs／実測ログのみ）
- `alloc_c` 削減の設計自体は #1294 が引き継いだ
  （`docs/cpu-matmul-fixed-cost-design.md`）

### 16.8 出典

- イシュー #1292（本追補）・#1290（依存。Layer A/B ハーネス実装）・
  #1148／#1185（親系列。§8.1 の推定元・§12.2 の boundary 実測）
- `docs/perf/cuda-gemm-reuse-phase-breakdown.md`（CUDA 版。#1182）・
  `docs/perf/metal-gemm-reuse-phase-breakdown.md`（Metal 版。#1189）
- `docs/perf/cpu-gemm-candle-cpu-retune.md` §5（`RowPanel` カーネル単体
  実測値の出典）
- `docs/perf/logs/cpu-gemm-reuse-phase-1292/`（本追補の生ログ・env_info・
  `layerA-summarize-{dgx,m4max}.md`）
- `scripts/bench/framework-compare/README.md`「CPU での区間定義と Layer B
  （イシュー #1290）」節

## 17. 2026-09-08 追補: 専有環境での RAYON_NUM_THREADS スイープ再実測・非単調性の切り分け（イシュー #1305）

### 17.1 位置づけ・事前宣言基準

§8.2 の `RAYON_NUM_THREADS` スイープ（#1148。各 3 回計測・共有負荷下）は、
両実機とも「大コア数」付近（DGX: 10 スレッドで 8 スレッド比 54%・M4 Max:
12 スレッドで 8 スレッド比 87%。いずれも N=1024 限定）でスループットが
落ち込む非単調な挙動を観測したが、この非単調性が異種コア構成由来（H1）か
背景負荷由来（H2）かを分離できていなかった。本追補は専有状態を確認した
うえで 5 回独立プロセス中央値でスイープを再実測し、DGX 限定の taskset
pin 実験（補助軸 A）・整列制御形状 N=1920（補助軸 B）を追加して要因を
切り分ける。判定基準は計測前に以下のとおり確定し、以後変更しない（元 issue
本文・実装計画 §2 と同一）。

**非単調性の「再現」定義**（対象指標: `oss-gemm-compare` の
`self_gemm_blis_parallel` `tflops_median` を 5 独立プロセスでとった中央値）:

- 大コア数スレッド T_big: DGX = 10（Cortex-X925 ×10）・M4 Max = 12（P コア ×12）
- **再現**: `median(T_big) < 0.90 × median(T=8)` かつ run 単位の比
  （run i の T_big 値 / run i の T=8 値）が 5 回中 4 回以上 1.0 未満（符号一貫）
- **非再現**: `median(T_big) >= 0.95 × median(T=8)`
- 0.90〜0.95 の帯、または上記いずれにも当てはまらない（数値は 0.90 未満だが
  符号一貫性を満たさない等）場合: **判定不能**（undetermined。fail-closed）

**帰属の決定表**:

| 観測 | 帰属 |
|---|---|
| 専有ゲート通過下で非再現 | H2 背景負荷由来 |
| 専有下で再現・DGX taskset 大コア固定で T=10 が T=8 以上へ回復・小コア固定で顕著に劣化 | **H1 異種コア由来** |
| 専有下で再現・大コア固定でも落ち込みが残る・制御形状 N=1920 では落ち込みなし | H3 分割粒度由来 |
| 専有下で再現・大コア固定でも残り・N=1920 でも残る | H1/H3 いずれでもない要因 |
| 専有ゲートが取れなかった実機 | その実機は undetermined |

### 17.2 プロトコル・専有状態の確認結果

- ハーネス: `scripts/bench/oss-gemm-compare`（`self_gemm_blis_parallel`・
  `matrixmultiply`・`gemm` crate を同一プロトコルで計測。本番 CPU GEMM
  経路〈`RowPanel`〉のみを計測）
- **DGX Spark GB10**: 隔離ディレクトリ `~/work/sweep-1305/` へ rsync
  （`docs/real-hardware-verification-env.md` §3 の除外セット準拠）・専有
  ゲート（1 分平均 load average < 2.0 を 2 回連続）は attempt=1 で即座に
  通過（load1=0.06→0.53）。スイープ実行中の load average は 0.06〜4.72
  で推移（`env_info.txt`）。**専有状態を確保できた**
- **Apple M4 Max**: 本マシン上で並列稼働する他の Claude Code エージェント
  セッションが多数存在し、専有ゲート（1 分平均 load average < 4.0 を
  2 回連続。ループ上限は計画記載の 30 分ではなく実装上 60 分になった。
  §17.9 参照）が最大待機時間中に一度も通過せず、「共有負荷下として続行」
  した（load average 2.42〜28.74 で推移）。**専有状態は確保できなかった**
  （#1147／#1367 等の既存記録〈load average 20〜31 前後〉と比べるとやや
  低いレンジではあるが、高負荷帯を含み続けており事前宣言基準の「専有ゲート
  取れなかった実機は undetermined」に該当する）
- 主スイープ: `SIZES=1024,2048,4096`・`RUNS=5`。DGX
  `THREADS="8 10 20 1 2 4 12 16"`・M4 Max `THREADS="8 12 16 1 2 4 10 14"`
- 補助軸 A（DGX 限定）: `RAYON_NUM_THREADS=10` 固定・大コア
  （`taskset -c 5-9,15-19`。Cortex-X925。MAXMHZ 3900・`midr_el1`
  partnum=0xd85 で確認）・小コア（`taskset -c 0-4,10-14`。Cortex-A725。
  MAXMHZ 2808・partnum=0xd87）・pin なしの 3 条件 × 5 run
- 補助軸 B: 制御形状 `SIZES=1920`（8/10/12/16/20 いずれでも `m/T` が 8 の
  倍数に整列。DGX は `THREADS="8 10 20"`・M4 Max は `THREADS="8 12 16"`）
- 生ログ・env_info（内部ホスト名は含めない）: `docs/perf/logs/cpu-gemm-rayon-sweep-1305/`

### 17.3 DGX Spark GB10 主スイープ（5 回独立プロセス中央値。TFLOP/s）

| N | T=1 | T=2 | T=4 | T=8 | T=10 | T=12 | T=16 | T=20 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.1223 | 0.2215 | 0.4136 | 0.6706 | 0.3416 | 0.3610 | 0.4912 | 0.5369 |
| 2048 | 0.1206 | 0.2405 | 0.4687 | 0.9019 | 1.0438 | 0.7071 | 0.5767 | 0.6884 |
| 4096 | 0.1202 | 0.2406 | 0.4785 | 0.9313 | 1.1302 | 0.8132 | 0.9626 | 1.0941 |

対 T=8 比（T=10）: N=1024 = **0.5094**（5 run 中 4 run が 1.0 未満）・
N=2048 = **1.1573**（5 run 全て 1.0 以上）・N=4096 = **1.2136**（5 run
全て 1.0 以上）。gemm crate（対照実装。同一 rayon 共有プール）も同傾向
（N=1024 T=10 対 T=8 = 0.6286/0.6873 = 0.9146。self ほど極端ではないが
低下方向）。

**再現判定**: N=1024 のみ「再現」（0.5094 < 0.90 かつ run 単位比
below_1_count=4/5）。N=2048・N=4096 は「非再現」（比が 1.0 以上で
単調非減少方向）。#1148 §8.2 の元表も N=1024 限定の記録だったため、
本追補は元の観測と整合する形状で再現を確認できた。

### 17.4 補助軸 A: DGX taskset pin（`RAYON_NUM_THREADS=10` 固定。TFLOP/s 中央値）

| N | pin なし（対 T=8 比） | 大コア pin（対 T=8 比） | 小コア pin（対 T=8 比） |
|---|---|---|---|
| 1024 | 0.3273（0.49） | 0.8883（**1.32**） | 0.3491（0.52） |
| 2048 | 1.0428（1.16） | 1.0849（1.20） | 0.3735（0.41） |
| 4096 | 1.1325（1.22） | 1.1353（1.22） | 0.3845（0.41） |

pin なしの値は主スイープ（§17.3）とは独立の 5 run（`taskset_nopin_dgx.log`）
で、N=1024 の値（0.3273 対 0.3416）は近接しており非単調性が run 間で
安定して再現することを裏付ける。

**決定的な観測**: N=1024 で大コア pin（X925 ×10 のみ）は T=8（4 コア混在
無指定・実効的に 8 スレッド）を **上回り**（比 1.32）、小コア pin
（A725 ×10 のみ）は **一貫して劣化**（比 0.52）。行パネル分割の粒度
（`m.div_ceil(10)=103` 行／パネル。8 の倍数に非整列）は 3 条件（pin なし・
大コア pin・小コア pin）で完全に同一であるにもかかわらず、コア配置だけで
結果が劇的に変わる。これは分割粒度（H3）では説明できず、**コア異種性
（H1）が主要因**であることを直接示す。N=2048／4096 でも同じパターン
（大コア pin ≒ pin なし・小コア pin は約 0.41 まで劣化）が見られ、pin
なしのスケジューラが N=2048/4096 では（理由は本追補では未解明。§17.9）
実質的に大コア寄りの配置になっている可能性を示唆する。

### 17.5 補助軸 B: DGX 制御形状 N=1920（整列形状。TFLOP/s 中央値）

| T | self（対 T=8 比） | 符号一貫性（below_1 件数/5） |
|---|---|---|
| 8 | 0.8905（基準） | — |
| 10 | 1.0594（**1.19**） | 0/5（全て 1.0 以上） |
| 20 | 0.7213（**0.81**） | 5/5（全て 1.0 未満） |

T=10（整列済み・192 行/パネル）では非単調性は**観測されない**
（比 1.19。§17.4 の pin なし挙動〈N=2048/4096 と同様に非劣化〉と整合）。
一方 T=20（全 20 論理コア・96 行/パネルで依然整列）では **一貫した劣化**
（比 0.81。5 run 全てで 1.0 未満）が見られる。T=20 では構造的に全ての
little コア（Cortex-A725 ×10）が計算へ動員されるため pin による回避が
できない。整列形状であるにもかかわらず T=20 の劣化が残ることは、H3
（分割粒度）が主要因ではなく、H1（コア異種性。全コア使用時に遅い little
コアがクリティカルパスになる）が T=20 の劣化にも一貫して当てはまる
ことを示す追加根拠である。

### 17.6 Apple M4 Max 主スイープ（5 回独立プロセス中央値・共有負荷下。TFLOP/s）

| N | T=1 | T=2 | T=4 | T=8 | T=10 | T=12 | T=14 | T=16 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.1152 | 0.2174 | 0.4238 | 0.7018 | 0.9254 | 0.6273 | 0.5726 | 0.7805 |
| 2048 | 0.1124 | 0.2183 | 0.4175 | 0.7185 | 0.9620 | 0.7496 | 0.6405 | 0.8416 |
| 4096 | 0.1072 | 0.2134 | 0.4054 | 0.7124 | 0.9343 | 0.9074 | 0.7699 | 0.9656 |

対 T=8 比（T_big=12）: N=1024 = 0.8938（below_1_count=3/5）・N=2048 =
1.0433（below_1_count=2/5。#1429 codex-review 指摘対応: `aggregate.py` の
run_id 追跡導入後に `rayon_sweep_m4max.log` の同一 run 同士を run id で
明示的に対応付けて再集計した値。位置的な zip に基づく旧記載「0/5」は
誤りだった〈中央値比・undetermined 判定自体は不変〉）・N=4096 =
1.2737（0/5）。制御形状 N=1920 の T=12 対 T=8 比は
0.9139（below_1_count=4/5・run 間の比が 0.32〜1.46 と極端に広く、専有下
とは言えない大きなノイズを含む）。

**再現判定**: いずれの N・制御形状も §17.1 の「再現」基準（数値条件かつ
符号一貫性の両方）を満たさない。N=1024（0.8938 は 0.90 未満だが
below_1_count が 3/5 で符号一貫性条件を満たさない）・N=1920 制御形状
（0.9139 は 0.90〜0.95 の帯）はいずれも「判定不能」、N=2048・N=4096・
T=16 は「非再現」（比が 1.0 以上）。**専有ゲートが最後まで通過しなかった
ため（§17.2）、上記いずれの判定結果も §17.1 決定表の最終行「専有ゲートが
取れなかった実機は undetermined」に従い undetermined として扱う**（数値
自体は非再現方向を示すが、確定させない）。

### 17.7 再現有無と帰属結論

| 実機 | N | T_big | 再現判定（§17.1 基準） | 備考 |
|---|---|---|---|---|
| DGX Spark GB10 | 1024 | 10 | **再現** | 大コア pin で回復・小コア pin で劣化（§17.4） |
| DGX Spark GB10 | 2048 | 10 | 非再現 | 比 1.16（改善方向） |
| DGX Spark GB10 | 4096 | 10 | 非再現 | 比 1.22（改善方向） |
| DGX Spark GB10 | 1920（制御） | 10 | 非再現 | 比 1.19 |
| DGX Spark GB10 | 1920（制御） | 20（全コア） | 再現（別軸） | 全コア使用で一貫劣化。§17.5 |
| Apple M4 Max | 1024 | 12 | 判定不能 | ゲート不通過のため確定させない |
| Apple M4 Max | 2048 | 12 | 非再現（参考） | ゲート不通過のため確定させない |
| Apple M4 Max | 4096 | 12 | 非再現（参考） | ゲート不通過のため確定させない |
| Apple M4 Max | 1920（制御） | 12 | 判定不能 | ノイズ極大（run 間 0.32〜1.46） |

**帰属結論（確定分）**: DGX Spark GB10 の N=1024（T=10）非単調性は
**H1（異種コア由来）**と確定する。決定表の該当行（「専有下で再現・DGX
taskset 大コア固定で T=10 が T=8 以上へ回復・小コア固定で顕著に劣化」）に
厳密に一致する結果を得た（§17.4）。整列形状 N=1920・T=20 の劣化も H1 と
整合する追加根拠である（§17.5）。DGX の N=2048／4096 は #1148 §8.2 の
元表がそもそも N=1024 限定の記録だったため「再現しなかった」というより
「元々観測されていなかった条件を新たに計測したら非単調性は出なかった」
という位置づけであり、本追補のスコープ（元観測の再現性確認）としては
問題ない。

**Apple M4 Max は undetermined として確定する**（専有ゲート不通過。
§17.2）。数値自体は非再現方向（大コア数 T=12 でむしろ改善する傾向。
N=2048/4096 は明確に 1.0 以上）を示しており、#1148 §8.2 で観測された
M4 Max の落ち込み（12 スレッドで 8 スレッド比 87%）は本追補の低負荷化
された共有環境下では再現しなかった。ただしこの結果を「H2（背景負荷由来）
確定」とはしない（§17.1 決定表の最終行に従い、専有ゲート不通過のまま
undetermined 扱いとする。M4 Max も P/E 異種コア構成〈12P+4E〉であり、
DGX で確認された H1 のメカニズムが理論上は M4 Max にも当てはまりうる
ため、「専有環境で再計測すれば #1148 の落ち込みは再現しない」と断定する
根拠は本追補の計測だけでは不十分である）。

### 17.8 #1307 へ渡す前提

- **少なくとも 1 実機（DGX Spark GB10）で H1 が確定**したため、§2.3 の
  分岐に従い #1307（2D 動的分配の設計）は**予定どおり進める**。ただし
  対象・優先度は本追補の実測範囲に即して以下のとおり絞り込む:
  - 主対象形状: **N=1024 相当の小〜中形状**（`m.div_ceil(T)` の商が
    little コアの実効スループットで律速されやすい、計算量に対し
    同期オーバーヘッドの相対比率が高い領域）。N=2048/4096 では本追補の
    DGX 計測で非単調性が見られなかったため、2D 動的分配の優先度は
    N=1024 近傍に置くことを提案する
  - 主対象スレッド数: 大コア数（DGX=10）近傍で、異種コア混在（pin なし）
    が発生しうるスレッド数域（T=10〜16 程度。T=20〈全コア使用〉でも
    §17.5 のとおり劣化が残るため、全コア使用時の緩和も設計スコープに
    含めるべきである）
  - `IcDynamic`（#1367。行パネル単位の atomic 動的配布のみ）は同じ
    N=1024・DGX で REJECT（比 0.63）だった。2D 動的分配（列方向の
    ブロッキングも含めた配布）が行パネル単位の動的配布だけでは
    足りなかった異種コア吸収を達成できるかが #1307／#1310／#1311 の
    設計判断ポイントになる
- Apple M4 Max は undetermined のため、#1307 の設計は M4 Max 側の効果を
  過度に見積もらない（DGX で確認された機構の理論的な転用可能性のみを
  記載し、M4 Max 実機での効果確認は 2D 動的分配実装後の A/B 計測
  〈#1312 相当〉に委ねる）
- 既定スレッド数の大コア限定（#1364。`BIG_CORE_LIMIT_ENABLED=false` に
  差し戻し確定済み）は、本追補の結果（H1 確定）とは独立の判断のまま
  変更しない。大コア限定は「スレッド数を減らして big コアのみ使う」
  アプローチで DGX の `cpu_capacity` 誤検出により REJECT されたのに対し、
  2D 動的分配は「全スレッド数を使いつつ配布を動的化する」別アプローチ
  であり、#1364 の REJECT は #1307 の前提を否定しない

### 17.9 スコープ外・限界

- **M4 Max の専有状態確保**: 本イシュー実行時点で本マシン上に他の
  Claude Code エージェントセッションが多数並列稼働しており、計画時点で
  意図した「30 分待機」を実装上「60 分待機」（`m4max_orchestrate.sh` の
  ループ上限を DGX 側実装からそのまま流用したための実装齟齬）で実行しても
  ゲートを通過できなかった。単一セッション・単一実行環境の制約により
  再試行は行わない。M4 Max での H1／H2 切り分けは後続イシューへ引き継ぐ
  （低負荷な時間帯を狙った再計測、または本マシン以外の M4 Max 実機の
  利用が必要）
- **DGX N=2048/4096 で pin なしが大コア寄りに振る舞う機構**: §17.4 で
  観測した「pin なしのスケジューラが N=2048/4096 では大コア pin と同等の
  性能を出す」現象の根本機構（OS スケジューラの CPU 周波数を考慮した
  負荷分散・実行時間の長さに応じた再配置等の仮説はあるが未検証）は
  未診断のまま記録するに留める。追加診断（`perf sched` 等でのスレッド→
  コア割当ログ取得）は本追補のスコープ外
- **2D 動的分配の実装・両実機比較**: #1307／#1310／#1311／#1312 へ
  引き継ぐ（本追補は docs 限定・`crates/` 変更なし）
- **N=512 のスイープ**: 受け入れ条件外（#1148 の 3 回計測値を参考値として
  残す）
- **framework-compare 境界（candle 比）での再計測**: Phase 4（#1321）
  のスコープ

### 17.10 出典

- イシュー #1305（本追補）・親 #1303・ルート #1283
- `docs/perf/logs/cpu-gemm-rayon-sweep-1305/`（本追補の生ログ・env_info・
  taskset マスク導出根拠・集計スクリプト。内部ホスト名は含めない）
- §8.2（元の非単調性観測。#1148）・§13（#1364。既定スレッド数の大コア
  限定 REJECT）・§14（#1367。`IcDynamic` REJECT）
- `docs/cpu-gemm-b-packing-sharing-decision.md`（静的等分割行パネル分割の
  実装根拠）

## 18. 2026-09-08 追補: 2D 動的分配 vs RowPanel 両実機比較・#1305 非単調性の解消／残存（イシュー #1312）

### 18.1 位置づけ

§17（#1305）が確定した DGX Spark GB10 の N=1024・T=10 非単調性（H1・異種コア
由来）に対し、#1307/#1310/#1311 が設計・実装した (mc, nc) 2D job 動的分配
（`GemmDriverVariant::TwoDDynamic`。`#[cfg(test)]` 限定）が緩和・解消するかを
両実機（DGX Spark GB10・Apple M4 Max）5 回独立プロセス中央値で A/B 計測し、
事前宣言ゲートで採否判定した記録。詳細な実測表・判定ロジックは
`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md` を正とし、本節はその要約と
§17.3（元の非単調性実測）との突合のみを記す。

### 18.2 §17.3 との突合表（DGX・N=1024）

| 指標 | §17.3（#1305・`RowPanel` のみ） | §18（#1312。同一ハーネス内の再測定） |
|---|---|---|
| RowPanel T10/T8 比 | 0.5094（5 run 中 4 run が 1.0 未満） | 0.4795（オーダー一致・再現を確認） |
| TwoDDynamic(jpw=2) T10/T8 比 | （計測なし） | 0.8581 |
| TwoDDynamic(jpw=4) T10/T8 比 | （計測なし） | 0.8646 |

§17.3 は `RowPanel` のみを対象としていたため直接の比較対象はないが、本追補で
同一の A/B ハーネス内に `RowPanel` を再度含めたところ T10/T8 比 0.4795 と
§17.3 の 0.5094 とオーダーが一致し、非単調性自体の再現を確認した（プロセス
起動条件・実行順序の違いによる誤差の範囲内）。

### 18.3 帰結

事前宣言した判定基準（≥1.00 で「解消」・<0.90 で「残存」・その間で
「部分的緩和」）を機械的に適用すると、T10/T8 軸では `TwoDDynamic` は
**「残存」**（jpw=2: 0.8581、jpw=4: 0.8646。いずれも 0.90 未満）と判定される。
ただし `RowPanel` の 0.48 から 0.86 前後へ大幅に改善しており、閾値のすぐ外側
という結果である点は注記する。

一方、§17.5 が観測した「全コア使用時の一貫した劣化」（整列形状 N=1920・T=20
で `RowPanel` が T=8 比 0.81 倍。5 run 全て 1.0 未満）に対応する軸（本追補では
N=1024・T=20/T=8）では、`TwoDDynamic` はほぼ解消する（jpw=2: 1.0039、
jpw=4: 1.0265）のに対し `RowPanel` は引き続き 0.7527 と劣化したままである。
**「N=1024・T=10 という中間スレッド数での非単調性」（§17.3 の軸）と「全コア
使用時の劣化」（§17.5 の軸）とで、`TwoDDynamic` の効果は異なる**（前者は
残存・後者はほぼ解消）。

Apple M4 Max は §17.7 と同様に本追補でも専有ゲートを通過できなかった
（1 分 load average が計測時 30〜60 台。他セッション並走）ため、DGX 単独の
上記結果のみで採用可否を確定しない（設計 `docs/cpu-gemm-2d-dynamic-partition-design.md`
§11「M4 Max が計測中に専有できない場合の扱い」）。最終判定は **undetermined**
（`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md` の判定根拠を参照）。

### 18.4 出典

イシュー #1312・#1311/#1307/#1310・`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`・
`docs/perf/logs/cpu-gemm-2d-dynamic-ab-1312/`・§17（本ファイル。#1305 の元の
非単調性実測）。

## 19. 2026-09-08 追補: 承認済み tolerance 契約下での DGX N=2048 再計測（イシュー #1262）

### 19.1 位置づけ・プロトコル

- イシュー #1241（2026-09-08 承認）で tolerance 契約変更（候補 A-1・係数 `c=0.5`。
  `docs/candle-parity-tolerance-contract-decision.md` §8）が承認され、Phase 2 実装として
  #1443（`bench-common::parity` へ第 3 救済項 `parity_scaled_abs_bound`／
  `parity_scaled_abs_rescued` を追加。`bench-candle` は救済ありの `GemmReference::verify`・
  `bench-fandhe` は救済なしの `verify_strict` を使用）・#1445（`compare_gemm_gate.py::
  _parity_check` の判定不能条件を新契約へ更新。candle 側 `rescued>0` は許容、fandhe-ai 側
  `rescued>0` は判定不能へ倒す fail-closed）がマージ済み（`origin/main`
  145639d047f6fcd7488d0665e1488e68ee084dfc。CUDA 側同種再計測 #1260 の反映〈#1447〉を含む）。
  本追補は §5.2・§12.3 で「判定不能」と記録し続けてきた DGX Spark GB10（Grace CPU）
  N=2048 が、この契約下で実際に解消されるかを GB10 実機で確認する（CUDA 側の同型記録は
  `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §14。イシュー #1260）
- **コード変更なし**（tolerance 定数〈`PARITY_REL_TOL`/`PARITY_ABS_TOL`〉・
  `PARITY_SCALED_ABS_COEFF`／`F32_UNIT_ROUNDOFF`・判定式・`bench-common`・
  `compare_gemm_gate.py`・`crates/`・`docs/spec/` は一切変更していない。本追補は既存契約下の
  **実測確認のみ**）
- プロトコルは §2 と同一（`GEMM_GATE_CPU_NODE_TAG=dgx-cpu run_gemm_gate_cpu.sh 0.7.0-1262`・
  `compare_gemm_gate.py --device cpu`）。**正式系列のみ**を計測した（`GEMM_GATE_PATCH_
  FACADE_PATH`・`GEMM_GATE_BENCH_FANDHE_FEATURES` いずれも未指定。manifest で
  `fandhe_ai_source=registry`・`candle_core_source=registry`・`bench_fandhe_features=""` を
  確認済み）。参考系列（HEAD path 差し替え）は計測していない。Apple M4 Max は N=2048 が
  §12.2 の時点で元々 `parity_fail_count=0`（判定不能ではない）のため本追補の再計測対象外
- 本ノードは他の並列セッションが常時ベンチマーク・ビルドを実行する共有環境のため、本
  Issue 専用の隔離ディレクトリ（`~/work/fc-1262/`。計測後に削除済み）へ rsync 転送して
  計測した。**1 回目は他セッションの CPU／CUDA ゲートと並走したため負荷混入
  （load average 完了時 17.17）が明確で破棄し、競合プロセスの不在を `pgrep` で確認した
  直後に 2 回目を実行して採用した**（詳細・破棄した 1 回目の値は
  `docs/perf/logs/cpu-gemm-candle-gate-1262/env_info.txt`「実行 1 回目（並走ゲートにより
  破棄）」節）。**なお N=2048 の要素単位判定（`fail`／`rescued`／`bound`）は 1 回目・2 回目
  で完全に同一の決定的な値であり、負荷の影響を受けていない**（§19.3）
- 生データ: `scripts/bench/framework-compare/results/raw/results-dgx-cpu-gemm-gate-
  0.7.0-1262.jsonl`（45 行）・`skipped-dgx-cpu-gemm-gate-0.7.0-1262.log`（空）・
  `manifest-dgx-cpu-gemm-gate-0.7.0-1262.json`。実行ログ・判定表出力・env_info:
  `docs/perf/logs/cpu-gemm-candle-gate-1262/`（`run_gemm_gate_cpu-dgx-cpu-0.7.0-1262.log`・
  `compare_gemm_gate-0.7.0-1262.md`）

### 19.2 実測結果（正式系列 `0.7.0-1262`。DGX Spark GB10・Grace CPU）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 | fandhe-ai fresh 中央値（参考。n=5） |
|---|---|---|---|---|---|---|
| 512 | 2.198 ms（2.088–2.552 ms） | 1.894 ms | 0.862 | 122.1 | 未達 | 2.553 ms（2.263–2.576 ms） |
| 1024 | 7.037 ms（6.986–7.136 ms） | 5.476 ms | 0.778 | 305.2 | 未達 | 7.735 ms（7.545–7.873 ms） |
| 2048 | 35.223 ms（33.865–35.464 ms） | 33.475 ms | 0.950 | 487.7 | **未達（candle 救済 2 要素。判定不能ではなく確定）** | 36.080 ms（34.824–36.324 ms） |

fandhe-ai 側は全 30 run（reuse 15 + fresh 15。N=512/1024/2048）で `parity_fail_count=0`
**かつ** `parity_scaled_abs_rescued=0`（`verify_strict` 経路。救済項に依存せず従来どおり
厳密ゼロ fail）。`compare_gemm_gate.py` の終了コードは 3（N=512/1024 の「未達」判定が残る
ため。仕様どおりで失敗ではない）。

**§12.2（DGX 正式系列 `0.7.0`）と比べ N=512（0.810→0.862）・N=1024（0.786→0.778）は
誤差範囲内で再現している**（正式ピン `fandhe-ai =0.7.0` は不変。tolerance 判定側のみが
変わった）。N=2048 は今回 GB10 で初めて確定値（0.950 倍）を記録できた。CUDA 側の §14.2
（N=2048 で 0.476 倍）とは値の水準が異なるが、CPU と CUDA は別バックエンド・別カーネルの
ため直接比較しない。N=2048 の 0.950 倍は #1185 計測（§12.2。当時は「判定不能」のため
比を確定表示していなかった生データ）の median から算出した参考比 0.968 倍
（`docs/perf/logs/cpu-gemm-candle-gate-1262/env_info.txt`「N=2048 比の妥当性根拠」節）と
近い水準であり、突飛な値ではない。

### 19.3 N=2048 の要素単位判定（中核）

candle 側 N=2048 fresh の 5 run すべてで以下が完全に決定的に一致した（1 回目・2 回目とも
同一値。`docs/perf/logs/cpu-gemm-candle-gate-1262/compare_gemm_gate-0.7.0-1262.md`）:

```
parity_fail_count = 0        (旧契約〈救済なし〉では 2。§5.2・§12.3 と同一の決定的挙動)
parity_total      = 4194304
parity_max_abs_err = 3.814697e-05   (§5.2・§12.3 と完全同一)
parity_max_rel_err = 3.944416e-01   (同上)
parity_scaled_abs_bound   = 1.525878e-05
parity_scaled_abs_rescued = 2
```

`bound=1.525878e-05` は机上評価（`docs/perf/candle-parity-tolerance-candidates.md` §4:
`c=0.5, eps=2^-24` で `bound≈1.526e-05`。CUDA 側 #1260 の実測値と同値）と実測が一致し、
`rescued=2` は同ファイル §4.1 で特定した CPU 側 fail 2 要素（`idx=1372466`・
`idx=1633751`。`|actual−exact|` が 1.265e-05／1.163e-05）が両方とも新設の第 3 項
（`c・u・K・S_A・S_B ≈ 1.526e-05`）以下に収まり救済されたことと整合する（両要素とも
`|actual−exact|` は `bound` 未満）。

**重要な注意（誤読防止）**: `parity_max_abs_err`（3.814697e-05）は **pass 要素も含む全要素
中の最大値**であり `bound`（1.525878e-05）を上回ったまま変化していない。これは CUDA 側
§14.3 と同型の理由により（`_parity_check` は `fail_count`／`rescued`／`bound` の整合
〈`rescued>0` なら `bound >= CHECKSUM_ABS_TOL`〉のみを検査し、`max_abs_err` 自体を判定に
使わない設計。`compare_gemm_gate.py:384-402`）、`fail_count=0` かつ
`max_abs_err > bound` が両立することは救済ロジックの不具合ではない。

### 19.4 fandhe-ai 側の不変確認

全 30 run（reuse 15: N=512×5・N=1024×5・N=2048×5、fresh 参考 15: 同上）で
`parity_fail_count=0` **かつ** `parity_scaled_abs_rescued=0`（`bound` は
`verify_strict` 経路のため `0.000000e+00`
固定。`bench-fandhe::run_gemm` 系は `ScaledAbsTolerance::NONE` を使うため構造的に救済に
依存しない。`scripts/bench/framework-compare/README.md`「承認済み契約の実装（イシュー
#1247）」節）。`compare_gemm_gate.py::_parity_check` の fail-closed 検査
（`framework=="fandhe-ai"` かつ `rescued>0` は判定不能へ倒す）が発火しないことを実測で
確認した。

### 19.5 #1117 後継ゲート判定表

| # | #1117 の受け入れ条件 | DGX Spark GB10（正式系列 `0.7.0-1262`） | 出典 |
|---|---|---|---|
| 1 | N=512 reuse で candle 超え | 未達（0.862 倍） | §19.2 |
| 2 | N=1024 reuse で candle 超え | 未達（0.778 倍） | §19.2 |
| 3 | N=2048 reuse で candle 超え | **未達（0.950 倍。判定不能ではなく確定判定）** | §19.2・§19.3 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 30 run `parity_fail_count=0` かつ `rescued=0`） | §19.4 |

**総合判定: tolerance 契約変更（#1241 承認・#1443/#1445 実装）により DGX Spark GB10 の
N=2048「判定不能」は解消し、3 形状すべてで確定判定（いずれも未達）が得られるように
なった。** #1117 自体はすでに #1185（イシュー #1185・2026-09-06）でユーザー指示により
クローズ済みで後継ツリー #1283 へ引き継がれている。本追補は、#1283 の本文が言及する
「DGX N=2048 判定不能」（ハーネス契約変更〈#1241〉の実測確認）についての実測解消を記録
するものであり、#1117 の再判定や #1283／#1234 の状態変更を行うものではない。

### 19.6 スコープ外・ユーザー判断事項

- **tolerance 定数・判定式は不変**（`PARITY_REL_TOL`/`PARITY_ABS_TOL`/
  `PARITY_SCALED_ABS_COEFF`/`F32_UNIT_ROUNDOFF` を含め本追補で一切変更していない）
- **本体 `assert_parity`（`crates/backend-cpu/src/parity.rs`。CPU バックエンドには
  `ParityBaseline` 相当の baseline 方式は存在せず、対象は `assert_parity` のみ）は
  対象外**（#1254／#1256 は承認スコープ外〈ハーネス限定〉として NOT_PLANNED クローズ済み。
  `docs/candle-parity-tolerance-contract-decision.md` §9）
- **Apple M4 Max の再計測**: N=2048 は §12.2 の時点で元々 `parity_fail_count=0`
  （判定不能ではない）のため対象外
- **N=512/N=1024 の未達（0.862／0.778 倍）、および N=2048 が確定「未達」となったことの
  扱い**: 後継ツリー #1283 の既存スコープ（Phase 1〜4。計測境界固定費・並列分割・
  マイクロカーネル・ゲート再判定）に従い、本追補では対応しない
- **N=2048 の負荷混入（1 回目破棄）**: 本ノードは複数セッションが常時並走する共有環境
  であり、完全な専有計測は毎回保証できない。2 回目採用値（0.950 倍）は、起動直前の
  `pgrep` 確認で競合プロセスが不在だったこと・完了時 load average が 5.60 に留まり
  1 回目（17.17）のような跳ね上がりが見られなかったこと・N=512/1024 が §12.2 の正式
  系列値と誤差範囲内で一致することから妥当と判断したが、計測実行中は自ラベル限定の
  生死監視のみで他セッションのプロセスを継続監視したわけではない（詳細は
  `docs/perf/logs/cpu-gemm-candle-gate-1262/env_info.txt`）。次回の再計測（例えば次回
  crates.io 公開後の正式再計測）では隔離ディレクトリでの実行に加えて専有ゲート
  （1 分平均 load average < 4.0 を連続確認。§17.2 の基準）を計測開始前・計測中とも
  厳密適用することが望ましいかはユーザー判断
- **期待値と異なる結果が出た場合の対応方針**: 本追補では期待値どおり
  `fail=0, rescued=2, bound=1.525878e-05` を確認できたため該当なし
- **Issue 操作は行わない**（`out-of-scope-tracking.md` に従い、本 PR では #1117／#1283／
  #1234 への状態変更は行わない）

## 20. 2026-09-08 追補: 出力並列ゼロ埋め on/off 両実機比較（イシュー #1301）

### 20.1 位置づけ・事前宣言する判定規則（計測前に確定）

イシュー #1299 が実装した `CpuBackendOps::gemm` 出力バッファの並列ゼロ
書き込み分岐（`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`。本番既定
`usize::MAX` で無効化。§16.6 の「実運用経路で削減効果が見込める本番経路
固定費は DGX N=2048 の `alloc_c`」という結論を実装したもの）について、
DGX Spark GB10（Grace CPU）・Apple M4 Max の両実機で off（`usize::MAX`）
/on（`2 << 20`。設計時暫定値）を同一プロトコル・5 回独立プロセス中央値で
比較し、本番結線可否を確定する。

事前宣言する判定規則（後出しで変えない）:

1. **checksum**: 全セルで off/on 完全一致（`compare_gemm_ab.py` の複合
   判定 pass かつ `==` 列一致）。不一致なら判定不能。
2. **DGX N=2048 reuse（決定セル）**: Layer B の `alloc_c` 中央値が
   削減され、かつ `ops_gemm` 中央値の on/off 比 ≤ 1.00。Layer A
   （`run_gemm_gate_cpu.sh`）の 2048/reuse の on/off 比 ≤ 1.05（非後退）。
3. **対照セル**: 両実機の N=512/1024 の reuse/fresh すべてで on/off 比
   ≤ 1.05。
4. **candle 比**: 各腕を `compare_gemm_gate.py --device cpu` で集計し、
   on 腕の candle 比が off 腕より後退していないこと。
5. **M4 Max N=2048**: on/off 比 > 1.05 で 5 run 符号一貫の後退なら
   macOS では有効化しない（#1299 スモークの再確認）。

結線の場合分け: (a) DGX が規則 1〜4 を満たし M4 Max が規則 5 で後退 →
**ADOPT（Linux 限定 cfg gating）**。(b) DGX で削減されない・後退する →
**REJECT**（コード無変更）。(c) 両実機とも全規則を満たす → 無条件
`2 << 20`。

出典: イシュー #1301・`docs/perf/logs/cpu-matmul-fixed-cost-1301/`。

### 20.1a 規則 4 の許容幅追補（codex-review 指摘対応。計測後に追記）

§20.1 規則 4 は「on 腕の candle 比が off 腕より後退していないこと」を
数値許容幅なしで宣言していたが、§20.3 の実測では 6 セル中 3 セル
（DGX 1024・M4 Max 1024・M4 Max 2048）で candle 比がわずかに後退した
（0.4〜3.2%）。この事実を「強い後退シグナルなし」という定性表現へ
言い換えて規則 4 を「満たす」と読み替えたのは、規則自体の後出し変更に
あたり事前宣言との整合を欠く（codex-review 指摘・本追補で是正）。

ここで、規則 4 を無条件の厳密非後退（許容幅ゼロ）のまま適用すると、
candle 比は 2 個の独立計測（off 腕・on 腕）の比という間接指標であり、
各腕はそれぞれ§20.2 に記載の専有度・計測ノイズを内包する。§20.1 規則 2・
3・5 はいずれも直接の on/off 比較に対して `≤1.05`（guardrail「劣化中央値
5% 以内」の慣例値。本 doc 内で既に採用済みの基準）という数値許容幅を
明示的に与えているのに対し、規則 4 のみ許容幅を持たない設計だったことが
今回の不整合の根本原因である。よって、規則 2・3・5 と同一の慣例値を
援用し、regel 4 を次のとおり数値許容幅付きで確定する（本追補が正式な
改定であり、以後の判定はこの版を規則 4 として扱う）:

**規則 4（改定版）**: 各セルの candle 比 on/off 比（`on 腕の candle 比 /
off 腕の candle 比`）が `1/1.05 ≈ 0.9524` 以上であること（規則 2・3・5 と
同じ `±5%` 慣例許容幅を、比の比という間接指標に対して逆数側〈悪化方向〉
のみへ適用する片側検査）。

実測（§20.3 candle 比表から算出）:

| 実機 | N | candle 比 off | candle 比 on | on/off 比 | 判定（≥0.9524） |
|---|---|---|---|---|---|
| DGX | 512 | 0.703 | 0.746 | 1.0612 | 満たす |
| DGX | 1024 | 0.778 | 0.775 | 0.9961 | 満たす |
| DGX | 2048 | 0.965 | 0.972 | 1.0073 | 満たす |
| M4 Max | 512 | 0.889 | 0.894 | 1.0056 | 満たす |
| M4 Max | 1024 | 0.760 | 0.747 | 0.9829 | 満たす |
| M4 Max | 2048 | 0.809 | 0.783 | 0.9679 | 満たす |

全 6 セルが改定版規則 4 を満たす。ただし本許容幅は計測後に定義した
ものであり、事前宣言の趣旨（後出しで変えない）を厳密には満たさない
ことを明記する。§20.4 の結論はこの追補を踏まえた再評価として扱う。

**2026-09-08 追補（PR #1448 codex-review 2 回目の指摘対応）**: 上記の
「計測後に定義した許容幅を同じ系列（§20.3）へ遡及適用して満たすと
判定する」やり方自体が、計測後に緩和した基準だけで本番採用を確定する
ものであり、事前宣言との整合を欠くという指摘を codex-review から重ねて
受けた。よって規則 4（改定版）は**以後の判定に用いる事前登録規則**
としてのみ確定・固定し、§20.3 の実測系列に遡及適用した上表の判定
（「全 6 セルが満たす」）は**参考情報にとどめ、単独では本番採用の
根拠としない**。本番採用の可否は、規則 4（改定版）を事前登録した
うえで実施する独立の再計測（§20.6）の結果によって確定する。

### 20.2 計測プロトコル・専有状態

- DGX Spark GB10: 本イシュー専用の隔離ディレクトリ（`~/work/fc-1301-run`）
  へ rsync 転送。転送直前の 1 分 load average が 2 回連続 <6（1.03〜1.08）
  であることを確認してから計測を開始した。
- 初回の on 腕計測中に他セッション（イシュー #1262・#1437）の並走を検出
  （load average 15〜18・同一形状 N=2048 の CPU GEMM ベンチが同時実行され
  ていた）。この回の Layer A・Layer B は**参考値として `docs/perf/logs/
  cpu-matmul-fixed-cost-1301/` に生ログを保存したうえで正式値には採用しない**
  （contamination の可能性が高く、on/off 比が Layer A で 1.4〜1.8 倍・
  Layer B の IQR が off 腕の 2 倍超に拡大していた）。
- 他セッション終了・専有状態（1 分 load average <6 を 2 回連続）を再確認
  してから on 腕（Layer A・Layer B とも）を再計測した（ファイル名末尾
  `-clean`。以下の §20.3 表はこの再計測分を正式値とする）。
- Apple M4 Max: 共有マシン（このリポジトリのメイン worktree が動くホスト
  自身）。#1299 当時（load average 9〜11・19 users）より低負荷（off 腕
  load average 約 7・on 腕 約 5）だったが、完全な専有は確保していない。

### 20.3 実測結果

**Layer A（`bench-fandhe gemm cpu <N> reuse`。5 run 中央値の on/off 比。
`compare_gemm_ab.py --device cpu --sizes gate`）**:

| N/mode | DGX（専有確認後の clean 系列） | M4 Max |
|---|---|---|
| 512/fresh | 0.8690 | 1.0068 |
| 512/reuse | 0.8889 | 0.9509 |
| 1024/fresh | 0.9841 | 1.0019 |
| 1024/reuse | 1.0042 | 1.0269 |
| 2048/fresh | 1.0469 | 0.9728 |
| **2048/reuse（決定セル）** | **0.9990** | **0.9788** |

checksum は両実機・全セル完全一致。

**Layer B（`gemm_reuse_phase_diag_cpu`。5 run 中央値。
`docs/perf/logs/cpu-matmul-fixed-cost-1301/aggregate-dgx-clean.md`・
`aggregate-m4max.md` の全 3 サイズ〈N=512/1024/2048〉集計から本節に
関係するセルを抜粋。N=512 集計欠落バグ〈`aggregate_layer_b.py` の
`SIZE_RE` が cargo test 接頭辞付き行に一致しなかった件。codex-review
指摘〉は是正済みで両ファイルとも N=512 を含む全 3 サイズが揃っている）**:

| 実機 | N | phase | off median (ms) | on median (ms) | on/off 比 |
|---|---|---|---|---|---|
| DGX | 2048 | `alloc_c` | 3.2554 | 1.6817 | **0.5166**（約 48% 削減） |
| DGX | 2048 | `ops_gemm` | 26.8177 | 26.7420 | **0.9972** |
| DGX | 1024 | `alloc_c` | 0.0306 | 0.0269 | 0.8791 |
| DGX | 1024 | `ops_gemm` | 5.4276 | 5.3701 | 0.9894 |
| M4 Max | 2048 | `alloc_c` | 0.0709 | 0.2110 | **2.9760**（増加。並列ゼロ書き込み分岐自体のオーバーヘッドが本診断計測境界では露出） |
| M4 Max | 2048 | `ops_gemm` | 20.1551 | 21.6416 | **1.0738**（約 7.4% 後退。`scripts/bench/framework-compare/README.md` に転記済み） |
| M4 Max | 1024 | `alloc_c` | 0.0186 | 0.0187 | 1.0054 |
| M4 Max | 1024 | `ops_gemm` | 2.9457 | 2.8949 | 0.9828 |

M4 Max N=2048 の Layer B `ops_gemm` 後退は、本番相当の計測境界である
Layer A（上表。N=2048/reuse on/off 比 0.9788・改善）とは逆符号である。
Layer B は診断専用の計測（`ops_gemm` 区間の `alloc_c`＋`kernel`＋
`tensor_wrap` のみを抜き出し、本番 reuse 経路にはない反復ごとの保持・
破棄ロジックを伴う。§20.2 冒頭参照）であり、本節の判定規則（§20.1）は
Layer A を主指標として設計されているため、この Layer A/B 間の乖離は
採否判断自体には影響しない。ただし後退の事実そのものは隠さず両表に
記載する（codex-review 指摘対応）。

**candle 比（`compare_gemm_gate.py --device cpu`。#1117 は両腕とも未達成
のまま不変。参考値）**: DGX 512/1024/2048 は off 0.703/0.778/0.965 →
on 0.746/0.775/0.972。M4 Max は off 0.889/0.760/0.809 →
on 0.894/0.747/0.783。6 セル中 3 セル（DGX 1024・M4 Max 1024・M4 Max
2048）で 0.4〜3.2% の後退が実測された。§20.1a の許容幅追補（規則 4
改定版・on/off 比 ≥0.9524）を適用した結果は §20.1a の表のとおり全 6
セルが満たす。

### 20.4 判定

§20.1（§20.1a の規則 4 改定版を含む）の事前宣言規則を機械的に適用する:

1. checksum: 全セル完全一致 → **満たす**
2. DGX N=2048 決定セル: `alloc_c` 48% 削減・`ops_gemm` 0.9972（≤1.00）・
   Layer A 0.9990（≤1.05） → **満たす**
3. 対照セル（両実機 N=512/1024）: 全セル 0.869〜1.047（≤1.05） → **満たす**
4. candle 比（規則 4 当初文言・許容幅なしの厳密非後退）: 6 セル中 3 セル
   （DGX 1024・M4 Max 1024・M4 Max 2048）で 0.4〜3.2% の後退があり
   **literal には不成立**。§20.1a の許容幅追補（改定版規則 4）を本系列
   （§20.3）へ遡及適用すれば全 6 セルが満たすが、この遡及適用は計測後
   に緩和した基準だけで本番採用を確定するものであり事前宣言の趣旨を
   欠く（PR #1448 codex-review 指摘。§20.1a 追補）。よって**規則 4 は
   本系列に対しては不成立として扱う**
5. M4 Max N=2048: Layer A on/off 比 0.9788（≤1.05・改善方向） → **規則
   5 は発火しない**（Linux 限定化は不要）。なお Layer B `ops_gemm` は
   1.0738 と後退するが、規則 5 は Layer A（本番相当境界）を対象として
   設計されているため判定対象に含めない

**結論: 規則 4 が本系列（§20.3）に対して不成立のため、§20.1 の場合分け
（a）〜（c）のいずれにも該当しない。本番既定は `GEMM_OUTPUT_PARALLEL_
ZERO_MIN_ELEMS = usize::MAX`（無効化）を維持する**（`2 << 20` へは
有効化しない。いったん `2 << 20` へ有効化していた版は本追補により
`usize::MAX` へ差し戻した）。§20.1a の改定版規則 4 は今後の判定に
用いる**事前登録規則**として確定・固定し、この規則を用いた独立の
再計測（§20.6）で ADOPT／REJECT を確定する。本節（§20.1〜§20.5）の
実測結果自体（checksum 完全一致・規則 1〜3・5 の非後退・DGX の
`alloc_c` 削減効果）は破棄せず、次回計測の**参考系列**として維持する。

### 20.5 出典

イシュー #1301・`docs/perf/logs/cpu-matmul-fixed-cost-1301/`（env_info・
Layer A/B 生ログ・`aggregate_layer_b.py`・`on-arm.patch`）・
`docs/perf/cpu-matmul-fixed-cost-impl.md` §2・§6・
`docs/cpu-matmul-fixed-cost-design.md` §10。

### 20.6 独立の再計測（引き継ぎ・未実施）

PR #1448 の codex-review 指摘に基づき、§20.1a の改定版規則 4 を事前
登録規則として固定したうえで、本節（§20.1〜§20.5）とは独立な計測
セッションで以下を実施し、本番既定（`GEMM_OUTPUT_PARALLEL_ZERO_MIN_
ELEMS`）の有効化可否を確定する:

- DGX Spark GB10（専有ゲート確認済み・§20.2 と同一条件）・Apple M4 Max
  の両実機で、§20.1 規則 1〜3・5（許容幅つき数値基準）と §20.1a 改定版
  規則 4（`on/off 比 ≥ 0.9524`。追加の許容幅変更は行わない）を同一
  プロトコル（Layer A・Layer B・5 回独立プロセス起動中央値）で適用する
- 規則を計測後に緩和・読み替えない（本追補の趣旨そのもの）。全規則を
  満たせば ADOPT（`2 << 20` へ有効化）、いずれか不成立なら REJECT
  （`usize::MAX` を維持）と機械的に確定する
- 本 PR（#1448）を作成したセッションは worktree 隔離環境で動作しており
  実機（DGX Spark GB10・Apple M4 Max）への接続手段を持たないため、
  この独立再計測は本 PR の対応スコープに含めず後続イシューへ引き継ぐ
  （`out-of-scope-tracking.md` に従い、本 PR 自体での Issue 起票は
  行わない。イシュー #1301 を再オープンするか新規 issue を起票するかは
  ユーザー判断）

## 21. 2026-09-08 追補: 2D 動的分配の本番結線（イシュー #1313）

- 位置づけ: #1312（`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`）が undetermined
  （M4 Max 専有ゲート不通過）と判定した `GemmDriverVariant::TwoDDynamic`（(mc, nc) 2D job
  動的分配）について、#1313 が M4 Max 専有ゲート付き Phase 0 再計測を実施した結果、
  jpw=2・jpw=4 とも Tier 1 条件（N=1024/2048 で対 `RowPanel` 比 1.00 以上・N=4096 で 0.95
  以上・勝ち run 3/5 以上）を充足し **ADOPT 確定・本番結線済み**（詳細な Tier 1 判定・
  実測表は `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`「#1313 追記」節を正とする）
- `TWO_D_JOBS_PER_WORKER`（既定値 2。#1311 導入時点から不変）を維持したまま
  `TWO_D_DYNAMIC_PRODUCTION_ENABLED = true` で `gemm_blis_parallel_with_transpose`・
  `gemm_blis_bias_act_parallel` から結線（`crates/backend-cpu/src/gemm_blis/mod.rs`）
- framework-compare gemm cpu N=512/1024/2048 × fresh/reuse の before/after（before=
  結線前 origin/main `fddca17`・after=本ブランチ HEAD）を両実機で実行し、**全 12 セルが
  非後退（ratio 0.60〜0.96・改善方向）・checksum 完全一致**を確認した:

  | 実機 | 512/fresh | 512/reuse | 1024/fresh | 1024/reuse | 2048/fresh | 2048/reuse |
  |---|---|---|---|---|---|---|
  | Apple M4 Max | 0.8954 | 0.8789 | 0.8792 | 0.8593 | 0.8575 | 0.8385 |
  | DGX Spark GB10 | 0.9643 | 0.9003 | 0.8142 | 0.7603 | 0.6504 | 0.6015 |

- 本追補は本番既定経路の変更（`RowPanel` → `TwoDDynamic`）を記録するものであり、#1117
  ゲート（対 candle 比。§12/§13/§19）の未達成判定自体を変更するものではない（candle 比
  ではなく `RowPanel` 比の改善である点に注意。REQ-8 candle 比ゲート未達成は継続）
- 実行ログ: `docs/perf/logs/cpu-gemm-2d-dynamic-wiring-1313/`

## 22. 2026-09-08 追補: Phase 1〜3 結線後の両実機ゲート再判定（イシュー #1321）

### 22.1 位置づけ・事前宣言する判定規則（計測前に確定・計測後に変更しない）

#1283（CPU GEMM の candle〈gemm crate〉超えトラッキング）Phase 4「ゲート再判定」の唯一の
タスク。Phase 1（計測境界固定費）・Phase 2（並列分割）・Phase 3（マイクロカーネル・
ブロッキング）の結線結果を反映した状態で、`run_gemm_gate_cpu.sh`／`compare_gemm_gate.py
--device cpu` の同一プロトコル（5 回独立プロセス中央値・reuse vs candle fresh）により
N=512/1024/2048 の candle 比を DGX Spark GB10（Grace CPU）・Apple M4 Max の両実機で
再計測し、ゲート判定（達成／未達／判定不能）を確定記録する。#1448 の codex-review 指摘
（計測後に緩和した基準で判定しない）を踏まえ、以下を計測前に確定した:

1. **判定の正**: `compare_gemm_gate.py --device cpu <JSONL>` の出力（`達成`＝
   `fandhe_median_s <= candle_median_s`／`未達`／`判定不能`）のみ。閾値・tolerance・判定式
   は変更しない
2. **正式判定**: 正式系列（承認済みピン `fandhe-ai =0.7.0`・registry 解決。manifest
   `fandhe_ai_source=registry`）のみが #1283 ゲートの正式判定。参考系列（origin/main HEAD
   の `crates/facade` へ path patch）は「次回ピン更新後の見込み値」であり正式達成にはならない
3. **系列・ラベル**: 正式 `0.7.0-1321`、参考 `head-ced4d14-1321`（origin/main HEAD
   `ced4d14f8acf08499308142c19e3ea34e9a27097` の 7 桁 short sha）。参考系列は feature 無効
   （`GEMM_GATE_BENCH_FANDHE_FEATURES` 未指定）の 1 腕のみ
4. **専有ゲート**: 1 分 load average < 6.0 を 30 秒間隔で 2 回連続確認（最大 30 分待機）。
   通過後に即起動。不通過時は中断せず計測を実行し「共有負荷下」と明記する
5. **DGX N=2048**: 承認済み tolerance 契約（A-1。#1241 承認・#1443/#1445 実装。§19 で確定
   判定へ遷移済み）下で確定判定として扱う。candle 側 `parity_scaled_abs_rescued=2`・
   `bound=1.525878e-05`・`fail=0` の再現を確認し、fandhe-ai 側は全 run `fail=0` かつ
   `rescued=0`（`verify_strict`）を確認する
6. **Phase 別寄与**: 再計測ではなく既存記録（§2 相当の表）から導出する。正式系列と参考系列
   の差分（同一実機・同一日・同一負荷状態）を「結線後 HEAD 総合効果」として併記する
7. **失敗時**: run が失敗しても数値を捏造しない（`skipped-*.log` を残す）。実機に到達
   できない場合は両実機の実測が揃わない限り未完了として扱う

### 22.2 結線状態と v0.7.0 との差分帰属表

origin/main HEAD（`ced4d14`）時点で `git diff v0.7.0..HEAD --stat -- crates/backend-cpu/src
crates/facade/src crates/autodiff/src crates/tensor-core/src` を実測した結果（20 ファイル・
+7704/−946 行。全文は `docs/perf/logs/cpu-gemm-candle-gate-1321/diff_v0.7.0_ced4d14_
cpu_path.txt`）と、各施策の実コード状態を突合した帰属表:

| Phase | 施策 | 結線状態（`ced4d14` 時点で再確認） |
|---|---|---|
| 1 | 出力並列ゼロ埋め `zeroed_output`（#1299/#1301） | **無効**（`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS = usize::MAX`。`crates/backend-cpu/src/ops.rs:153`。#1448 で一度 `2 << 20` へ有効化後、codex-review 指摘により同一 PR 内で差し戻し） |
| 2 | `TwoDDynamic` 2D 動的分配（#1311/#1312/#1313） | **本番結線済み**（`TWO_D_DYNAMIC_PRODUCTION_ENABLED = true`・`TWO_D_JOBS_PER_WORKER = 2`。`crates/backend-cpu/src/gemm_blis/mod.rs:2987,3008`） |
| 2 | 既定スレッド数の大コア限定（#1363/#1364） | 無効（`BIG_CORE_LIMIT_ENABLED = false`。`crates/backend-cpu/src/thread_limit.rs:93`。REJECT） |
| 3 | 候補 3 KC 再スイープ（#1315） | REJECT（`KC=256` 維持・コード変更なし） |
| 3 | 候補 1 laneq ベクトル転置（#1317/#1318） | REJECT（`#[cfg(test)]` 維持・本番未結線） |
| 3 | 候補 2 prefetch（#1319） | docs のみ（unsafe 未着手） |
| — | 借用ビュー readout（#1337） | feature `host-view-readout` 既定 OFF（本追補は未指定のため不使用） |

**結論**: 結線後 HEAD の CPU NN GEMM reuse 本番経路と v0.7.0 の実質的な差は
**「`RowPanel` → `TwoDDynamic`」のみ**（詳細な根拠・PR タイトルと実装状態の不一致注記は
`docs/perf/logs/cpu-gemm-candle-gate-1321/attribution.md` を参照）。diff に現れる
`autodiff::optim::device_store`・`Var` 等の変更は VJP 転置入口（#1213）・デバイス常駐勾配
更新（#1212）等、backward／update フェーズの変更であり、本イシューが計測する
`Var::matmul` forward GEMM reuse 経路（`compare_gemm_gate.py` の計測対象）には到達しない。

### 22.3 プロトコル・専有状態

- 両実機とも本イシュー専用の隔離ディレクトリで計測（DGX: `~/work/fc-1321/
  rust-ai-library-run/`・計測後削除済み。M4 Max: 本 worktree を直接使用）
- DGX 専有ゲート: 08:14:23〜08:14:53 UTC に 2 回連続 load1<6.0（0.29→0.50）で通過。
  ただし直後の 1 回目の起動試行（`orchestrate.sh`）は `CARGO_TARGET_DIR` 分離に起因する
  manifest 記録失敗（`formal exit=1`／`reference exit=1`。08:14:53〜08:16:07 UTC）に終わり、
  原因判明後の 2 回目の起動試行（`orchestrate2.sh`。08:16:54 UTC 開始）で成功している
  （`gate-dgx.log`）。以下の実行順・実測時刻は成功した 2 回目の試行のもの
- M4 Max 専有ゲート: 08:18:53 UTC に load1=4.34 で 2 回連続 load1<6.0 を満たし通過（開始時
  load average 6.48〜8.26 の共有負荷から低下したタイミングで通過）。**ただし参考系列計測
  完了時点で load average が 9.27 まで再上昇しており、計測ウィンドウ中に他セッションの
  並走負荷が増加した可能性がある**（詳細・時系列は
  `docs/perf/logs/cpu-gemm-candle-gate-1321/env_info.txt`「Apple M4 Max」節）
- DGX 実行順: 正式系列（ビルド完了後の実測は 08:18:00〜08:18:48 UTC。load average は
  実測開始時 5.30・完了時 6.58 と上昇）→ 参考系列（実測は 08:18:54〜08:19:34 UTC。load
  average は実測開始時 6.06・完了時 7.87 と上昇。`run_gemm_gate_cpu-dgx-*.log` の
  `before`/`after` status）。**DGX も M4 Max と同様、専有ゲート通過後の計測ウィンドウ中に
  load average が再上昇している**
- M4 Max 実行順: 正式系列（`0.7.0-1321`。専有ゲート通過〈08:18:53 UTC〉で開始し、参考系列開始
  （`gate-m4max.log` の `08:20:01 UTC`）までに完了。macOS `uptime` は秒精度を持たないため
  `cpu status (before)/(after)` は分単位表示（before 17:19 JST=08:19 UTC・after 17:20 JST=
  08:20 UTC。load average は 5.62→10.92 と上昇）にとどまり、DGX（Linux `uptime` で秒精度
  あり）のように開始・完了を秒単位で確定できない。manifest `recorded_at`（`08:19:34 UTC`）
  はビルド・バイナリ検証完了直後に記録された時刻であり、計測完了時刻ではない
  （旧稿はこれを完了時刻と誤記していた。是正: #1321 codex-review 指摘）
  → 参考系列（08:20:01〜08:20:49 UTC）
- 4 系列とも manifest で `fandhe_ai_source`（正式=`registry`、参考=`path:<絶対パス>`）・
  `candle_core_source=registry`・`bench_fandhe_features=""` を確認済み。生データは 45 行・
  `skipped-*.log` は空

### 22.4 実測結果

#### DGX Spark GB10（Grace CPU）・正式系列 `0.7.0-1321`

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 512 | 2.471 ms（2.292–2.785 ms） | 1.721 ms | 0.696 | 108.62 | 未達 |
| 1024 | 7.154 ms（6.549–7.420 ms） | 5.510 ms | 0.770 | 300.18 | 未達 |
| 2048 | 35.429 ms（33.899–35.916 ms） | 33.241 ms | 0.938 | 484.91 | 未達（candle 救済 2 要素） |

#### DGX Spark GB10（Grace CPU）・参考系列 `head-ced4d14-1321`

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 512 | 2.233 ms（1.233–2.349 ms） | 1.792 ms | 0.802 | 120.20 | 未達 |
| 1024 | 6.254 ms（5.489–6.357 ms） | 5.470 ms | 0.875 | 343.36 | 未達 |
| 2048 | 21.417 ms（20.644–21.468 ms） | 33.462 ms | **1.562** | 802.16 | **達成（candle 救済 2 要素）** |

#### Apple M4 Max・正式系列 `0.7.0-1321`

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 512 | 764.8 µs（731.5–768.5 µs） | 750.2 µs | 0.981 | 351.01 | 未達 |
| 1024 | 3.857 ms（3.707–3.963 ms） | 3.421 ms | 0.887 | 556.73 | 未達 |
| 2048 | 23.309 ms（23.107–24.481 ms） | 20.780 ms | 0.891 | 737.04 | 未達 |

#### Apple M4 Max・参考系列 `head-ced4d14-1321`（共有負荷下。§22.3 参照）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s | 判定 |
|---|---|---|---|---|---|
| 512 | 694.6 µs（682.5–718.5 µs） | 786.8 µs | **1.133** | 386.46 | **達成** |
| 1024 | 3.281 ms（3.223–3.907 ms） | 3.837 ms | **1.169** | 654.44 | **達成** |
| 2048 | 22.678 ms（18.890–30.466 ms） | 23.837 ms | **1.051** | 757.55 | **達成** |

生データ・判定表出力・実行ログは `docs/perf/logs/cpu-gemm-candle-gate-1321/`
（`compare_gemm_gate-{dgx,m4max}-{0.7.0-1321,head-ced4d14-1321}.md`・
`run_gemm_gate_cpu-{dgx,m4max}-{0.7.0-1321,head-ced4d14-1321}.log`）・
`scripts/bench/framework-compare/results/raw/results-{dgx,m4max}-cpu-gemm-gate-
{0.7.0-1321,head-ced4d14-1321}.jsonl`（各 45 行）を参照。

### 22.5 要素単位判定（DGX N=2048・fandhe-ai 側 0 fail）

DGX 両系列とも candle 側 N=2048 fresh の全 5 run で以下が完全に決定的に一致した
（§19.3 と同一の決定的挙動）:

```
parity_fail_count = 0
parity_total      = 4194304
parity_max_abs_err = 3.814697e-05
parity_max_rel_err = 3.944416e-01
parity_scaled_abs_bound   = 1.525878e-05
parity_scaled_abs_rescued = 2
```

fandhe-ai 側は 4 系列・全 180 行中 fandhe-ai 側 120 run（各系列 reuse 15 + fresh 15 = 30 run
×4 系列。reuse 60 + fresh 60）すべてで
`parity_fail_count=0` **かつ** `parity_scaled_abs_rescued=0`（`verify_strict` 経路）を確認
した。`compare_gemm_gate.py::_parity_check` の fail-closed 検査（`rescued>0` は判定不能へ
倒す）は発火していない。

### 22.6 ゲート判定表

| 実機 | 系列 | N=512 | N=1024 | N=2048 |
|---|---|---|---|---|
| DGX Spark GB10 | 正式 `0.7.0-1321` | 未達（0.696） | 未達（0.770） | 未達（0.938。確定判定） |
| DGX Spark GB10 | 参考 `head-ced4d14-1321` | 未達（0.802） | 未達（0.875） | **達成（1.562）** |
| Apple M4 Max | 正式 `0.7.0-1321` | 未達（0.981） | 未達（0.887） | 未達（0.891） |
| Apple M4 Max | 参考 `head-ced4d14-1321` | **達成（1.133）** | **達成（1.169）** | **達成（1.051）** |

**総合判定**: #1283 の正式判定（承認済みピン `fandhe-ai =0.7.0` 基準）は
**両実機・全 3 形状で未達成のまま**（DGX N=2048 は §19 以来の確定判定を再確認）。
一方、Phase 1〜3 の結線結果（実質 `TwoDDynamic` のみ）を反映した参考系列（origin/main
HEAD・次回 crates.io 公開後の見込み値）では、**Apple M4 Max が全 3 形状で達成、
DGX Spark GB10 が N=2048 のみ達成**という、これまでの #1283 系列の記録の中で
初めて「達成」が観測されたゲート判定となった。

### 22.7 Phase 1〜3 各施策の寄与

- **Phase 2（`TwoDDynamic`）が支配的**: §22.2 の帰属表のとおり、正式↔参考系列の差は
  実質的にこの 1 施策のみ。§21 の before/after（結線前 `fddca17` → 結線後 HEAD。両実機・
  全 12 セルで非後退・改善方向、とくに 2048/reuse が M4 Max 0.8385・DGX 0.6015）と、
  本追補の正式↔参考の差（M4 Max 2048 で 23.309ms→22.678ms・約 0.97 倍、DGX 2048 で
  35.429ms→21.417ms・約 0.60 倍）はオーダーとして整合する（DGX で改善幅が大きい点も
  §21 と同方向）
- **Phase 1（出力並列ゼロ埋め）は無効のまま寄与ゼロ**: §20 の実測（DGX N=2048 の
  `alloc_c` 約 48% 削減効果を確認）は本番結線されていないため、本追補の参考系列にも
  反映されていない。§20.6（独立再計測）は未実施のまま
  （後述 22.9・22.10）
- **Phase 2（大コア限定）・Phase 3（KC 再スイープ／laneq 転置／prefetch）は全て REJECT
  または未結線**のため寄与なし
- **候補と実測の乖離に関する注記**: `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`
  「#1313 追記」・本ファイル §21 の Tier 1 判定は「対 `RowPanel` 比」であり「対 candle 比」
  ではない。本追補で初めて candle 比ゲートの達成が観測されたのは、`RowPanel` 比の改善
  （とくに N=2048 で大幅改善）が candle との差を逆転させるほど大きかったことを示す
  （§21 時点では candle 比ゲート自体は評価対象外だった）

### 22.8 tolerance 契約変更ツリーの状態

イシュー #1241（承認）→ #1443/#1445（実装）→ #1447/#1449（CUDA/CPU 反映）→ #1262
（DGX N=2048 の判定不能解消。§19）の一連のツリーは**完了済み**。本追補の計画立案時点で
懸念されていた「同ツリー未完了なら N=2048 を判定不能として記録する」というフォールバックは
**適用されない**（DGX N=2048 は §22.4/§22.5 のとおり両系列とも確定判定〈未達／達成〉と
なった）。

### 22.9 未達が残る場合の原因分析と次候補

- **正式系列（#1283 の正式判定基準）は依然として両実機・全形状で未達**。次回
  crates.io 公開（`release-all.yml`）で `TwoDDynamic` を含む HEAD がピン `fandhe-ai` へ
  反映されない限り、正式判定は §22.6 の「未達」のまま変わらない
- 未達が残る N=512/1024（両実機とも参考系列でも M4 Max のみ達成・DGX は依然未達）・
  DGX N=512/1024 の原因分析は既存の内訳分解記録（§16「計測境界固定費のフェーズ分解」・
  §17「専有環境での非単調性」・retune §8「次候補」）を参照。本追補では新規の内訳分解・
  非単調性再調査は実施していない（再計測のみがスコープ）
- **§20.6（出力並列ゼロ埋めの独立再計測）は本追補でも未実施のまま**。Phase 1 施策が
  有効化されれば DGX 側（とくに N=512/1024 の `alloc_c` 固定費）にさらなる改善余地が
  残っている可能性があるが、これは §20.6 の引き継ぎ事項でありユーザー承認待ち
- retune（`docs/perf/cpu-gemm-candle-cpu-retune.md` §8）の次候補（KC 再スイープ・laneq
  転置・prefetch）はいずれも REJECT／未着手のまま。新たな次候補の起票は本追補のスコープ外

### 22.10 スコープ外・ユーザー判断事項

- **ピン更新（次回 crates.io 公開）**: 参考系列で達成した形状（M4 Max 全 3 形状・
  DGX N=2048）を正式達成にするには、`TwoDDynamic` を含む HEAD を次回 `release-all.yml`
  で crates.io へ公開し、`fandhe-ai =0.7.0` → 新バージョンへピンを更新する必要がある。
  このリリースフローの実行はユーザー判断（`docs/crates-io-publishing-order.md`）
- **§20.6（出力並列ゼロ埋め独立再計測）**: `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS` の
  有効化可否判断は #1301 の引き継ぎ事項のまま。本追補では対応しない
- **`host-view-readout` 既定化・N=4096・CUDA／Metal 側ゲート**: 対象外
- **M4 Max 共有負荷下での参考系列計測**: §22.3 のとおり専有ゲート自体は通過したが、
  計測完了時点で load average が再上昇していた。より確度の高い再計測（真の専有環境・
  複数 attempt での再現性確認）が必要かはユーザー判断
- **Issue 操作は行わない**（`out-of-scope-tracking.md` に従い、本 PR では #1283／#1320／
  #1321 への状態変更・コメント投稿は行わない）
- **期待値と異なる結果への対応**: DGX N=2048 は期待どおり `fail=0, rescued=2,
  bound=1.525878e-05` を確認できたため該当なし。参考系列での「達成」の観測は事前規則の
  範囲内の事実（規則を事後に緩めて生じた結果ではない）

### 22.11 出典

- `docs/perf/logs/cpu-gemm-candle-gate-1321/`（実行ログ・env_info・diff・attribution・
  gate/uptime ログ）
- `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`「#1313 追記」（Tier 1 判定・ADOPT 確定）
- `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §19（DGX N=2048 tolerance 契約解消）・
  §20（出力並列ゼロ埋め）・§21（`TwoDDynamic` 本番結線）
- `docs/candle-parity-tolerance-contract-decision.md`（tolerance 契約承認記録）
- `scripts/bench/framework-compare/results/summary.md` 環境 26 節
- `docs/performance-targets.md` §8.11

## 23. 2026-09-08 追補: 同一 facade ソース下の借用ビュー readout 既定化 before/after・両実機（イシュー #1438）

### 23.0 位置づけ

§15（#1337）は片方向の負荷差を伴う registry off／path on の比較で「DGX は非後退・M4 Max は
達成見込みだが確度限定的」と記録していた。#1438 は借用ビュー readout を bench-fandhe の
既定経路とし旧 `host-view-readout` cargo feature を撤去するにあたり、**両腕を同一 facade
ソース**（本 PR HEAD の `crates/facade` への path patch）で揃えた before（legacy bench-fandhe
ソース）/after（default bench-fandhe ソース）比較を DGX Spark GB10・Apple M4 Max 両実機で
再計測し、§15 の facade ソース不一致の交絡を解消した。

### 23.1 プロトコル

`docs/perf/logs/gemm-candle-gate-readout-default-1438/env_info.txt` を参照。両腕とも
`GEMM_GATE_PATCH_FACADE_PATH` で本 PR HEAD の同一 `crates/facade` を指定し、bench-fandhe
側のソースのみを base コミット `fddca17`（legacy）／本 PR HEAD（default）で切り替えた。
実測は 2026-09-08。DGX は低負荷（`nvidia-smi`／`uptime` 実測。GPU 常駐 2 プロセスのみ・
load average 1.7〜5.1）、M4 Max は高負荷共有環境（他セッション並走。load average 9〜25）。

### 23.2 実測結果（fandhe-ai reuse・5 回計測中央値・before/after 比較。判定対象）

**DGX Spark GB10（dgx-cpu）**

| N | before 中央値（min–max） | after 中央値（min–max） | after/before | checksum |
|---|---|---|---|---|
| 512 | 2.401 ms（1.924–2.471 ms） | 2.251 ms（2.190–2.472 ms） | 0.937 | 完全一致 |
| 1024 | 6.916 ms（6.823–7.157 ms） | 6.535 ms（6.481–6.886 ms） | 0.945 | 完全一致 |
| 2048 | 35.283 ms（34.447–35.458 ms） | 30.819 ms（30.668–31.373 ms） | **0.874** | 完全一致 |

**Apple M4 Max（m4max-cpu）**

| N | before 中央値（min–max） | after 中央値（min–max） | after/before | checksum |
|---|---|---|---|---|
| 512 | 747.4 us（734.1–806.4 us） | 719.3 us（697.5–738.9 us） | 0.962 | 完全一致 |
| 1024 | 3.755 ms（3.728–3.842 ms） | 3.533 ms（3.493–3.616 ms） | 0.941 | 完全一致 |
| 2048 | 24.836 ms（23.464–25.707 ms） | 23.581 ms（22.445–24.609 ms） | 0.950 | 完全一致 |

reuse は両実機・全 3 サイズで非後退（`after/before` 0.87〜0.96）。checksum は全セル完全一致
（要素単位 `parity_fail_count=0`）。出典: `docs/perf/logs/gemm-candle-gate-readout-default-1438/
compare_gemm_ab-cpu-{dgx,m4max}.md`（`compare_gemm_ab.py --device cpu` の既定 `fresh,reuse`
出力。fresh 行を含む）

### 23.2a 参考: fresh（イシュー #1438 の受け入れ条件外。同一 raw JSONL から再集計）

イシュー #1438 の受け入れ条件・事前宣言判定木は reuse のみを対象とする。同じ raw JSONL
（`scripts/bench/framework-compare/results/raw/results-{dgx,m4max}-cpu-gemm-gate-head-{fddca17-readout-legacy,1438pr-readout-default}.jsonl`）
には fresh の 5 回計測も含まれるため、`compare_gemm_ab.py --device cpu`（標準ライブラリのみ・
5 回中央値）で再集計した結果を参考記録として掲載する（PR #1452 codex-review 指摘対応）。

**DGX Spark GB10（dgx-cpu・fresh）**

| N | before 中央値（min–max） | after 中央値（min–max） | after/before | checksum |
|---|---|---|---|---|
| 512 | 2.251 ms（2.196–2.371 ms） | 1.752 ms（1.709–2.002 ms） | 0.778 | 完全一致 |
| 1024 | 7.738 ms（7.594–7.773 ms） | 5.609 ms（5.421–5.635 ms） | 0.725 | 完全一致 |
| 2048 | 36.191 ms（34.493–36.301 ms） | 29.079 ms（28.234–29.350 ms） | 0.804 | 完全一致 |

**Apple M4 Max（m4max-cpu・fresh）**

| N | before 中央値（min–max） | after 中央値（min–max） | after/before | checksum |
|---|---|---|---|---|
| 512 | 737.9 us（731.0–771.3 us） | 746.4 us（711.3–759.7 us） | **1.0115** | 完全一致 |
| 1024 | 3.503 ms（3.469–3.535 ms） | 3.513 ms（3.424–3.608 ms） | **1.0028** | 完全一致 |
| 2048 | 24.297 ms（24.013–24.499 ms） | 23.320 ms（22.542–23.677 ms） | 0.960 | 完全一致 |

- DGX は fresh も全 3 サイズで改善方向（0.72〜0.80）
- M4 Max は N=512／1024 の fresh が 1.0115／1.0028 と**僅かに後退方向**であり、
  「fresh/reuse 両モードで全セル非後退」とは言えない。差は 0.3〜1.2% で before/after の
  min–max 範囲は重なっており（高負荷共有環境・load average 9〜25 下の計測）、**計測ノイズ
  かどうかは本記録のみでは確定できない**（専有ゲート付き再計測は本イシューのスコープ外）
- ログ表（`compare_gemm_ab-cpu-*.md`）の「判定」列は `compare_gemm_ab.py` の既定閾値
  1.05 による表示であり、判定木条件 (a)（after/before ≤ 1.00）とは異なる。本節の
  「僅かに後退方向」は後者の基準による
- 非後退の総括（§23.5）は判定対象の reuse に限定する

### 23.3 candle 比ゲート（#1117。参考記録。正式判定は §12 のまま不変）

**DGX**

| N | before candle 比 | before 判定 | after candle 比 | after 判定 |
|---|---|---|---|---|
| 512 | 0.809 | 未達 | 0.843 | 未達 |
| 1024 | 0.810 | 未達 | 0.834 | 未達 |
| 2048 | 0.955（candle 救済 2 要素） | 未達 | **1.103**（candle 救済 2 要素） | **達成** |

**M4 Max**

| N | before candle 比 | before 判定 | after candle 比 | after 判定 |
|---|---|---|---|---|
| 512 | 0.898 | 未達 | 0.968 | 未達 |
| 1024 | 0.742 | 未達 | 0.788 | 未達 |
| 2048 | 0.775 | 未達 | 0.814 | 未達 |

出典: `docs/perf/logs/gemm-candle-gate-readout-default-1438/gate_cpu-{dgx,m4max}-{legacy,default}.md`。
DGX N=2048 の after 腕が #1117 の受け入れ条件（candle 超え）を満たす一方、他の全セルは未達
のまま（正式系列 `fandhe-ai =0.7.0` ピンは本 PR で更新していないため §12／§19 の判定は不変）。

### 23.4 公正性の論点

- `bench-candle` は本 PR で一切変更していない（`.to_vec2()` による所有 `Vec` 読み出しのまま）
- CPU backend の `gemm` 本体（`gemm_blis` 系）は本 PR で変更していない。差分は readout
  （`readout_var`／`checksum_var`）のみであり、両実機とも改善方向のみが観測された

### 23.5 採否

判定対象の reuse は両実機・全 6 セル（3 サイズ×2 実機）で非後退・checksum 完全一致のため、
CPU についても既定経路化を承認（ADOPT）。イシュー #1438 の事前宣言判定木の条件 (a)「全 N で
after/before ≤ 1.00」を reuse で両実機とも満たすため、runtime `Device` 分岐によるフォールバックは
導入しない。§15 の「確度限定的」という留保は本節（同一 facade ソース比較）により解消した。
fresh（§23.2a。受け入れ条件外の参考記録）は DGX 全形状改善・M4 Max N=2048 改善だが、M4 Max
N=512／1024 は 1.0115／1.0028 と僅かに後退方向であり、ノイズかどうかは本記録のみでは確定
できない。本節の非後退の総括は reuse に限定する。
