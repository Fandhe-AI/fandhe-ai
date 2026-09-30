# framework-compare 両機体再計測とスコアボード再生成（Phase 3 採否反映・イシュー #2120）

| 項目 | 内容 |
|---|---|
| 親 | #2099（Phase 3。負けセル 22 件への施策群）。基準は `docs/perf/loss-attribution-matrix.md` §3（`gen_1988.out`・fandhe-ai は `reuse` 行） |
| 状態 | **再計測基盤＋事前登録規則を追加済み。M4 Max・GB10 の本番実測と結果転記は未実施**（後述「申し送り」） |
| 判定規則 | `docs/perf/logs/framework-compare-phase3-remeasure-2120/RULE.txt`（実測前に固定） |
| 計測キット | `docs/perf/logs/framework-compare-phase3-remeasure-2120/`（`orchestrate.sh`・`switches.sh`・`aggregate.py`・`scoreboard/gen_2120.py`） |

## 1. 目的

Phase 3 の施策群のあと、framework-compare を両機体（Apple M4 Max・DGX Spark GB10）で再計測し、スコアボードを再生成して、
負けセル 22 件（負け 20＋僅差 2）の倍率が Phase 3 の前後でどう変わったかを定量化する。残存する負けセルの原因再分析は
Phase 4 のスコープであり本イシューでは行わない。

## 2. 前提のずれ（本キット作成時点の実測）

Phase 3 の #2100〜#2119 はクローズ済みだが、マージ済みの施策はいずれも **opt-in・既定 OFF のまま実機判定を申し送っている**。
main 上で既定 ON に結線された ADOPT 施策は 0 件であり、「ADOPT 施策の結線後に計測する」という本イシューの前提は現時点では未充足である。

このため計測キットは rev 非依存に作り、計測時点の `.rev-stamp` における各スイッチの既定値を `switches.sh` で機械抽出して
`switches-{B,C}.txt` へ証跡化する（採否スナップショット）。既定 ON のスイッチが 0 件なら、B→C の比は 1.00 近傍・checksum は
文字列一致が期待値であり、それは正当な記録結果として扱う（RULE.txt「期待値」）。各施策の実機判定・既定 ON 化の後に本キットを
再実行すれば、スナップショットと B→C 比へそのまま反映される。

### 2.1 採否スナップショット（作成時点の HEAD `0a25a9c0`。計測時に再取得して上書きする）

| スイッチ | 既定値（作成時） | 出典（file:line） | 導入 issue | 計測時の値（記入欄） |
|---|---|---|---|---|
| `HOST_ARENA_DEFAULT_ENABLED` | false | `crates/tensor-core/src/alloc.rs:25` | #2104 | |
| `REDUCTION_SEQUENTIAL_FALLBACK_ENABLED` | false | `crates/backend-cpu/src/reduction.rs:105` | #2101／#2102 | |
| `INFER_GRAPH_DEFAULT_ENABLED` | false | `crates/backend-cuda/src/graph.rs:658` | #2115 | |
| `TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` | false | `crates/backend-metal/src/train_forward_encode_runtime.rs:33` | #2113 | |
| `METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED` | false | `crates/backend-metal/src/fixed_cost_diag.rs:43` | #2114 | |
| `UNROLL_LOAD_ENABLED` | false | `crates/backend-metal/src/tile.rs:1576` | #2110／#2111 | |
| `SME_PRODUCTION_ENABLED` | false | `crates/backend-cpu/src/gemm_blis/mod.rs:3085` | #2118／#2119 | |
| `GB10_AFFINITY_ENABLED` | false | `crates/backend-cpu/src/gb10_affinity.rs:126` | #2117 | |
| `FANDHE_AI_CUDA_READBACK_DEST`（環境変数・未設定で OFF） | opt-in | `crates/backend-cuda/src/readback_policy.rs:33` | #2108 | |
| `FANDHE_AI_METAL_READBACK_DEST`（環境変数・未設定で OFF） | opt-in | `crates/backend-metal/src/readback_policy.rs:30` | #2112 | |
| `FANDHE_AI_CUDA_GRAPH_INFER`（環境変数・未設定で OFF） | opt-in | `crates/backend-cuda/src/graph.rs:684` | #2115 | |

`SME_PRODUCTION_ENABLED` と `GB10_AFFINITY_ENABLED` の切替は人間承認事項であり、本イシューでは実施しない。

## 3. 計測設計（RULE.txt の要約。正は RULE.txt）

| 腕 | fandhe-ai のソース | 役割 |
|---|---|---|
| A | registry `fandhe-ai =0.9.0`（承認ピン・patch なし） | 基準（`gen_1988.out`）と同じ実装の参考腕 |
| B | Phase 3 直前ツリー `65035979`（#2445 docs マージ。Phase 3 最初のコード変更 `7b40ed4f` の直前） | Phase 3 前 |
| C | 計測時点の main HEAD | Phase 3 後（スコアボード主表示） |

- 主比較は B→C（同一 run 内比 `C.median_s / B.median_s` の 5 run 中央値。**非後退 ⇔ 1.00 以下**）。A→C は 0.9.0 以降の非 Phase 3 変更を含む参考比。
- 5 run・プロセス独立・腕順は run ごとに反転（奇数 A→B→C・偶数 C→B→A）。candle・burn は各 run で 1 回。Python FW 行は既存値を流用。
- 専有ゲート: GB10 は load1 < 1.0 かつ GPU 使用率 0%（不通過 run も除外せず併記）。M4 Max は record_only（load1 < 8.0 を最大 30 分待機。不通過でも計測し記録）。
- checksum は各腕・各セルで 5 run 文字列完全一致、かつ腕間でも一致。不一致は是正せず記録。
- tolerance・baseline・`Cargo.toml`・`Cargo.lock`・ガードレール閾値・`docs/spec` は不変。`FANDHE_AI_*` は未設定（設定されていれば起動を拒否）。

## 4. 負けセル 22 件の Phase 3 前後倍率（実測後に記入）

倍率は `gen_1988.out` と同じ定義（最速他 FW ÷ fandhe-ai。gemm／train は所要時間比・infer は逆向き。1.00 未満が負け側）。
「基準」列は `docs/perf/loss-attribution-matrix.md` §3 の転記（2026-09-16／18 セッション）。「腕 B」「腕 C」列は
`scoreboard/gen_2120.py --tsv` の出力（腕 B を主入力にした実行と腕 C を主入力にした実行）から転記する。「B→C 比」は
`aggregate.md` の fandhe-ai セル別 `C.median_s / B.median_s` の 5 run 中央値（**時間比。infer も同じ向き**。<1 が高速化）。

| セル ID | 対象 | 基準（`gen_1988.out`） | 腕 B | 腕 C | B→C 比 | 非後退 | checksum | 備考 |
|---|---|---:|---:|---:|---:|---|---|---|
| M-MTL-G256 | M4 Max gemm Metal N=256 | 0.93 | | | | | | |
| M-MTL-G512 | M4 Max gemm Metal N=512 | 0.72 | | | | | | |
| M-MTL-G1024 | M4 Max gemm Metal N=1024 | 0.68 | | | | | | |
| M-MTL-G2048 | M4 Max gemm Metal N=2048 | 0.74 | | | | | | |
| M-MTL-G4096 | M4 Max gemm Metal N=4096 | 0.59 | | | | | | |
| M-CPU-G256 | M4 Max gemm CPU N=256 | 0.66 | | | | | | |
| M-CPU-G1024 | M4 Max gemm CPU N=1024 | 0.91 | | | | | | |
| M-CPU-G2048 | M4 Max gemm CPU N=2048 | 0.73 | | | | | | |
| M-MTL-TRN | M4 Max train Metal | 0.38 | | | | | | |
| M-MTL-INF | M4 Max infer Metal | 0.79 | | | | | | |
| M-CPU-TRN | M4 Max train CPU | 0.21 | | | | | | |
| M-CPU-INF | M4 Max infer CPU | 0.18 | | | | | | |
| G-CUDA-G256 | GB10 gemm CUDA N=256 | 0.83 | | | | | | |
| G-CUDA-G512 | GB10 gemm CUDA N=512 | 0.43 | | | | | | |
| G-CUDA-G1024 | GB10 gemm CUDA N=1024 | 0.41 | | | | | | |
| G-CUDA-G2048 | GB10 gemm CUDA N=2048 | 0.48 | | | | | | |
| G-CPU-G256 | GB10 gemm CPU N=256 | 0.72 | | | | | | |
| G-CPU-G512 | GB10 gemm CPU N=512 | 0.76 | | | | | | |
| G-CUDA-TRN | GB10 train CUDA | 0.88 | | | | | | |
| G-CUDA-INF | GB10 infer CUDA | 0.41 | | | | | | |
| G-CPU-TRN | GB10 train CPU | 0.34 | | | | | | |
| G-CPU-INF | GB10 infer CPU | 0.72 | | | | | | |

監視セル: GB10 CUDA gemm N=4096（基準は burn 比 1.03×・1 位。同一 run 内比の最速相手 burn の中央値 1.0145 と 1 に近く、
`loss-attribution-matrix.md` §3 が「回帰の兆候があれば #2108／#2120 で再確認」と定めたセル）。計測後にこのセルの腕 B／C の順位・倍率を追記する。

## 5. 再現手順（Mac／GB10 セッション向け）

```bash
# 0. 腕 B のツリーを用意（例。転送でも可）。各ツリーのルートに .rev-stamp（空不可。B は 65035979 で始まること）を書く
git worktree add --detach <pre-tree> 65035979 && git -C <pre-tree> rev-parse --short HEAD > <pre-tree>/.rev-stamp
git rev-parse --short HEAD > <head-tree>/.rev-stamp   # main HEAD のツリー（.rev-stamp はコミットしない）
# 1. 疎通確認（任意。1 run・N=256 の gemm のみ・ゲート省略・candle／burn なし。結果は判定に使わない）
SMOKE=1 HEAD_TREE=<head-tree> PRE_TREE=<pre-tree> LOGD=<scratch> bash docs/perf/logs/framework-compare-phase3-remeasure-2120/orchestrate_gb10.sh
# 2. 本計測（GB10 は orchestrate_gb10.sh・M4 Max は orchestrate_m4max.sh。LOGD は機体別の m4max／gb10 ディレクトリ）
HEAD_TREE=<head-tree> PRE_TREE=<pre-tree> LOGD=<logd> bash docs/perf/logs/framework-compare-phase3-remeasure-2120/orchestrate_<machine>.sh
# 3. 集計（派生 JSONL は <prefix>-{A,B,C}-full.jsonl）
python3 aggregate.py --self-test
python3 aggregate.py --machine <gb10|m4max> --logs <logd> --md <logd>/aggregate.md --out-prefix <logd>/results
# 4. スコアボード（腕 C を主入力・腕 B を前比入力。M4 Max と GB10 の派生 JSONL を両方渡す）
python3 scoreboard/gen_2120.py --prev-label "Phase 3 前" --main-label "Phase 3 後（HEAD）" --m4 m4max/results-C-full.jsonl --m4-prev m4max/results-B-full.jsonl \
  --gb gb10/results-C-full.jsonl --gb-extra <空ファイル> --gb-py <py JSONL> --gb-prev gb10/results-B-full.jsonl \
  --out fandhe-ai-phase3-scoreboard.html --tsv ratios-C.tsv
# 腕 B の倍率（基準との比較用）は --m4 ...-B-full.jsonl --gb ...-B-full.jsonl へ差し替えて同様に実行し --tsv ratios-B.tsv を得る
```

- `--m4-prev` は M4 Max の Python FW 転記値を `gen_2120.py` が内蔵しているため、機体別の派生 JSONL だけを渡す。
- スコアボード HTML は新規 Artifact として公開してよい（0.9.0 版・#1988 版は不変のまま履歴として残す）。
- ホスト名・ユーザー名・絶対パスは `orchestrate.sh` が `env_info.txt`／`build.log` で `<home>`／`<head-tree>`／`<pre-tree>` へマスクする。生ログ（JSONL・`err.log`）にもホスト情報が無いことを収録前に確認する。
- 所要時間の目安: 3 腕化のため 1 run は既存の run_all 系スイープの約 2 倍強（candle・burn は 1 回のみ）＋ゲート待機。

## 6. 申し送り（本 PR で未実施）

| 項目 | 状況 |
|---|---|
| M4 Max 5 run 実測（専有ゲート record_only） | 作業ホストが x86_64＋RTX 3060 のため未実施。Mac セッションへ |
| GB10 5 run 実測（専有ゲート成立下） | 同上。GB10 セッションへ |
| §2.1・§4 表の転記、`aggregate.md`／`switches-*.txt`／`env_info.txt`／生ログの収録 | 実測後 |
| スコアボード HTML の再生成・Artifact 公開 | 実測後 |
| イシュー #2120 の受入条件チェック更新 | 実測後（実機実測が済むまで #2120 を完了扱いにしない） |
| Phase 3 各施策の実機判定・既定 ON 化 | 各施策の判定 issue 側。結線後に本キットを再実行する |

## 7. 検証（作業ホストで実施したもの）

- `python3 aggregate.py --self-test`（正常系・5 run 欠け拒否・`parity_fail_count` 不正 4 種拒否・run 欠損時の空欄保持・checksum 不一致検出・比 1.00 境界〈非後退〉／1.0001〈後退〉）。
- `gen_2120.py --body body_1988.html --prev-label 0.8.0` を #1988 の元入力で実行した HTML・検証出力が `gen_1988.py` の出力と byte 同一。
- `switches.sh` を HEAD で実行し 8 定数・3 環境変数がすべて解決（`--allow-missing` は腕 B 用）。
- x86_64＋RTX 3060 で `SMOKE=1 orchestrate_gb10.sh`（3 腕ビルド〈腕 A は registry `fandhe-ai v0.9.0`・B／C は path 解決を `cargo tree` で確認〉・`Cargo.lock` sha256 前後一致・CPU gemm N=256 の 3 腕 checksum `237.546660` 一致）。この作業ホストは CUDA toolkit（NVRTC）非搭載のため CUDA セルは `skipped.log` に記録され失敗扱いで正しく分離された（結果は判定に使わず未収録）。CUDA・Metal の実行経路そのものは実機セッションでの初回実測時に確認する。

## 8. 変更していないもの

tolerance 4 定数・判定式・`ParityBaseline`／`BASELINES`・`Cargo.toml`／`Cargo.lock`・ガードレール閾値・`docs/spec`・
承認ピン（`fandhe-ai =0.9.0`・`candle-core =0.11.0`・`burn =0.21.0`）・`SME_PRODUCTION_ENABLED`／`GB10_AFFINITY_ENABLED` の既定値・
既存 REJECT／undetermined 実験（再実行なし）。新規依存・新規 `unsafe`・facade 公開面の拡張はない。
