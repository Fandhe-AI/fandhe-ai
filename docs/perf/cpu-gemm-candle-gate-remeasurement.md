# CPU GEMM N=512/1024/2048 reuse candle 比再計測と #1117 ゲート判定（イシュー #1148）

## 状態: DGX Spark（Grace CPU）・Apple M4 Max とも実機実測完了。#1117（reuse candle 超え）は両実機・全形状で未達成（DGX N=2048 は候補側 candle 無効データにより判定不能）と判定した。#1185 で正式系列 `fandhe-ai =0.7.0` を 2026-09-06 に両実機で再計測し未達成を確定（§12）

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
差）の挙動である。

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
2. **並列化の非単調性（§8.2）**: 両実機で観測された「大コア数付近の落ち込み」
   は P/E 異種コア構成での静的等分割行パネル分割が疑わしいが、背景負荷ノイズ
   （M4 Max は他エージェント稼働中。DGX はスイープ実行中の load average を
   記録しておらず背景負荷の有無が不明）と
   完全には分離できていない。専有環境での再計測（または work-stealing 分割
   への変更検討）が必要
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
「判定不能」のまま据え置く（reuse/candle 比の参考併記も行わない）。

### 12.4 #1117 ゲート判定（確定）

| 実機 | N=512 | N=1024 | N=2048 | parity（fandhe-ai 側） |
|---|---|---|---|---|
| DGX Spark GB10（Grace） | 未達（0.810 倍） | 未達（0.786 倍） | 判定不能（candle 無効データ） | 達成（0 fail） |
| Apple M4 Max | 未達（0.922 倍） | 未達（0.778 倍） | 未達（0.829 倍） | 達成（0 fail） |

**総合判定: #1117 は正式系列 `fandhe-ai =0.7.0` においても両実機で未達成（判定可能な
5 件すべて未達・DGX N=2048 は判定不能）。#1148 の判定を 0.7.0 で確定した。** 達成条件の
見直し要否・後継ツリーへの引き継ぎは §10「2026-09-06 更新」を参照。
