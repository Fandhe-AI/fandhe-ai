# Mac mini の infer Metal reuse の二峰性の原因特定（イシュー #3183）

**状態: 診断基盤のみ実装。Mac mini の 5 run 実測・帰属判定は未実施（Mac mini セッションへ申し送り）。**
受入基準 AC1〜AC3（分布・帰属・系列 A/B の説明）は未充足で、イシューは閉じない。AC4（ライブラリのコードと既定値を変えない）は充足している。

## 1. 背景

- 0.11.0 の Mac mini 計測（`docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`）では、infer Metal reuse（784→256→ReLU→10・batch 64・`Sequential::predict_resident`）の 1 反復時間が run 内で q1 約 0.385 ms／q3 約 0.61 ms に分かれ、run 中央値が 0.382〜0.609 ms に散らばる。0.478 ms 付近にまとまった run もあり、山が 2 つとは限らない
- 系列 B の 5 run 中央値は 0.388 ms（candle 比 1.13 倍）、系列 A は 0.539 ms（0.72 倍）。fresh は 0.85〜0.91 ms で安定している
- 既存の `metal-tape-build-infer-fixedcost.md`（#2114）はセル単位の集計値しか出さず、高い側のモードがどの区間と同時に現れるかを分けられない。`infer-reuse-phase-breakdown.md` §10 は `predict_resident` が単一区間のため分解できないと報告している

## 2. 設計（事前登録規則は `logs/metal-infer-bimodal-macmini-3183/RULE.txt`）

1 反復 1 行の JSONL を出す 4 腕を、腕ごとに warmup 20 + 計測 500 反復の長い系列で回す。

| 腕 | 内容 | 判別する仮説 | 実装 |
|---|---|---|---|
| R | `predict_resident` → bench と同じ legacy 読み出し → checksum | 二峰性が main HEAD と診断の枠組みでも再現するか | `crates/facade/tests/metal_infer_bimodal_diag.rs` |
| D | provider_select → tape_new → snapshot → chain → host_copy → checksum（キャッシュ OFF = 既定） | 高モードがどの区間で起きるか | 同上 |
| Don | D をデバイス存在確認キャッシュ ON のスコープ内で実行（診断専用） | provider_select（IOKit・`MTLCopyAllDevices`）を外しても二峰が残るか | 同上 |
| G | upload → `linear_forward_device_tracked` × 2 → `synchronize_with_gpu_timestamps` → download | GPU 実行時間／ホスト完了待ち／encode のどれが増えるか | `crates/backend-metal/src/infer_fixed_cost_diag_tests.rs`（`cfg(test)`・`#[ignore]`） |

- カウンタ（encode／command_buffers／wait／probe／hit／uploads／downloads）は既存の診断面の差分を反復ごとに出す。診断カウンタ・既定値・本番コードは追加も変更もしない
- モード検出は「2 つ」を前提にせず、相対ギャップ 10% 以上（両側 5% 以上が残る位置）で分割する。時間方向（交互出現／プロセス単位の固着）、帰属（区間ごとの Δ_P/Δ_total ≥ 0.5 が多峰 run の過半数）、系列 A/B の窓判定（連続 20 反復の窓中央値）の閾値は RULE.txt に実測前に固定した
- 腕の起動順は run ごとにローテーションし、順序効果を打ち消す。集計器 `aggregate.py` は fail-closed（`--self-test` 22 項目 pass）

## 3. 結果（未実測）

| 項目 | 状態 |
|---|---|
| 反復ごとの分布（各モードの割合・中央値）（AC1） | 未実測 |
| 高モードと同時に増える区間の帰属、または「特定できなかった」の明記（AC2） | 未実測 |
| 系列 A（0.539 ms）と系列 B（0.388 ms）の差が分布で説明できるか（AC3） | 未実測 |
| #3191 への申し送り（安定させたときの比） | 実測後に記入 |

## 4. 限界

- 腕 G は facade 層（tape・snapshot・legacy 読み出し）を含まない。`linear_forward_device_tracked` は本番チェーンと同じ呼び出しを使う
- 反復単位の GPU クロック・電力状態は sudo なしでは取れない。`powermetrics` は `sudo -n` が通る場合だけ run 単位で取得し、通らなければ未取得と記録する（sudo は使わない）
- bench-fandhe は crates.io の `fandhe-ai =0.11.0`、本診断は main HEAD を使う。`v0.11.0..HEAD` には `crates/autodiff/src/optim/device_store.rs`（約 380 行）、`crates/backend-metal/src/ops.rs`（約 590 行）、`crates/facade/src/compat/sequential.rs` ほかの変更があり、infer チェーンへの影響の有無は実測側で R 腕の水準が bench 対照（0.11.0）と同程度かで確かめる。同程度でなければ原因特定の前提が崩れるため、その旨を結果に書く
- M4 Max での同じ診断は対象外

## 5. 申し送り・スコープ外

- Mac mini セッション: `logs/metal-infer-bimodal-macmini-3183/README.md` の手順で実測し、`macmini/` に生ログと `aggregate.md` を置いて本書 §3 を埋める
- 二峰性の是正そのもの（既定値や本番経路の変更）は本イシューの対象外。原因が特定できたら別イシューで A/B 後に判断する
- 関連: `infer-reuse-phase-breakdown.md` §10・`metal-tape-build-infer-fixedcost.md`（#2114）・#3191（セル別の到達値の表）
