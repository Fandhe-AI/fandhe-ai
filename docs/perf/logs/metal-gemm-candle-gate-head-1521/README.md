# イシュー #1521: Metal GEMM candle 比ゲートの参考系列（split-K 結線後 HEAD）実行手順（Mac）

本ディレクトリはイシュー #1521（Metal GEMM candle 比ゲート〈旧 #1037〉を
split-K 結線後 HEAD で再計測し `docs/perf/metal-gemm-candle-gate-
remeasurement.md` §16 正式系列との差分を帰属する）の実機実測成果物置き場。
本ラン（Linux・CI 環境）はスキャフォールド（オーケストレーション・帰属
スクリプト・事前登録判定規則・帰属テスト）のみを用意し、実測は Apple
M4 Max 実機を持つ Mac セッションへ引き継ぐ（ルート #1509 の運用方針・
メモリ `issue-1509-linux-side-policy`）。

## 位置づけ

`docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/`（イシュー #1490）は
正式系列 `fandhe-ai =0.8.0`（registry）のみを計測し、当時は
`v0.8.0 ↔ origin/main` の Metal 計測経路 diff がゼロだったため参考系列
を計測しなかった。その後 #1527（`SPLIT_K_NUMERIC_CONTRACT_APPROVED=
true`）・#1530（split-K 本番結線）で diff が非ゼロになったため、本イシュー
で初めて参考系列を計測する。ただし `attribution.md` の構造分析が示す
とおり、対象形状（NN 正方 N=1024/2048/4096）は結線後も classic 経路の
まま変わらない**期待値**であり、実測はこの期待どおりかを確認するもの。

## 事前登録判定規則

`docs/perf/metal-gemm-candle-gate-remeasurement.md` §18.1（本 PR で追記）
に固定した。要点:

1. 判定の正は `compare_gemm_gate.py --device metal` の出力のみ。
   tolerance・判定式・`BASELINES`・`SPLIT_K_*` 定数・`tile::select` 系は
   不変。
2. 正式判定（旧 #1037・§16.7）は本イシューでは更新しない。A（対照）・
   B（参考）とも「正式判定を更新しない」記録。
3. 専有ゲートは要件にしない（record_only。ルート #1509 のユーザー指示に
   従う）。負荷が高いこと自体は undetermined の理由にしない。
4. 帰属分類（N ごと）: `r = median_B / median_A`。`|r-1| <= 0.05` または
   `median_B` が A の 5 run min–max 範囲内なら「負荷差（ノイズ帯）」、
   それ以外は「構造分析と矛盾・原因未確定」（コード差確定ではない）。
5. §16 との差は「セッション間の負荷ドリフト指標」として記録のみ
   （verdict を付けない）。
6. run の差し替え禁止。失敗時は数値を捏造せず fail-closed の退避に従う。
7. 実行前に `git diff v0.8.0..HEAD --stat`（Metal 経路）を再取得し、
   想定外の差分があれば帰属表を再導出する。

## 実行手順

```sh
cd scripts/bench/framework-compare
SHORT_SHA=$(git rev-parse --short HEAD)
FACADE_PATH="$(cd ../../../crates/facade && pwd)"

# 専有ゲートは要件にしないが、負荷は記録のため確認する。
uptime

cd ../../../docs/perf/logs/metal-gemm-candle-gate-head-1521
GEMM_GATE_PATCH_FACADE_PATH="$FACADE_PATH" \
  ./orchestrate_m4max.sh "$SHORT_SHA"
```

A（対照 `0.8.0-ctrl-1521`）→ B（参考 `head-<short sha>-1521`）の順に
`run_gemm_gate_metal.sh` を 1 回ずつ実行する（`run_gemm_gate.sh` が内部で
N=1024/2048/4096 それぞれ 5 回起動するため、run 単位の interleave は
構造的に不可。系列単位の A→B 固定順のみ）。

## 集計

```sh
cd ../../../scripts/bench/framework-compare
python3 compare_gemm_gate.py --device metal \
  results/raw/results-m4max-gemm-gate-0.8.0-ctrl-1521.jsonl \
  --out ../../../docs/perf/logs/metal-gemm-candle-gate-head-1521/compare-A.md
python3 compare_gemm_gate.py --device metal \
  "results/raw/results-m4max-gemm-gate-head-${SHORT_SHA}-1521.jsonl" \
  --out "../../../docs/perf/logs/metal-gemm-candle-gate-head-1521/compare-B.md"

cd ../../../docs/perf/logs/metal-gemm-candle-gate-head-1521
python3 attribute.py \
  ../../../scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-0.8.0-ctrl-1521.jsonl \
  "../../../scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-head-${SHORT_SHA}-1521.jsonl" \
  | tee attribution-result.md
```

`attribute.py` は `compare_gemm_gate.load_rows`／`evaluate_size`（判定式・
tolerance の単一真実源）をそのまま import して使う（判定ロジックを複製
しない）。`--self-test` で合成 JSONL 3 ケース（ノイズ帯内／min–max 内
救済／帯域外）を検証できる。

## 転記先

- `docs/perf/metal-gemm-candle-gate-remeasurement.md` §18.4〜18.6（実測
  結果・データ有効性・帰属表）
- `env_info.txt`（機種・OS・rustc・base sha・負荷推移・manifest 突合・
  中断有無）
- `performance-targets.md` §8.17・`scripts/bench/framework-compare/
  results/summary.md` 環境 31 の新設は本イシューのスコープ外
  （§18.8 参照。実測後に別途反映するかはユーザー判断）

## 注意（既存成果物の上書き禁止）

`orchestrate_m4max.sh` は既存の成果物（`uptime_before.txt`・
`run_<label>.log` 等）を検出すると何も書かず非ゼロ終了する（run の
差し替え禁止）。同一実行のやり直しは成果物を手動で別名へ退避してから
行う。ロックは `scripts/bench/framework-compare/
.splitk-framework-compare-1517.lock`（#1517 と共有。framework-compare
ディレクトリ単位の排他。Cargo.lock・path patch・バイナリ等の共有ファイル
を他 issue の実行と取り合わないため）。
