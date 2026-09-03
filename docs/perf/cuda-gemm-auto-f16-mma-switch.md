# CudaGemmAuto::run_f16 の MatrixUnit 分岐 mma 優先化 前後比較（#1156）

イシュー #1156 の受け入れ条件 R5（ユーザー承認条件）「結線前後で同一プロトコル・
5 回計測中央値の比較を取り、後退確認時は結線しない」に対応する記録。設計の正は
`docs/dispatch-rules-design.md` §5.6。

## 状態: 未実測・本番未結線（本エージェント実行環境に CUDA 実機なし。PR #1177 codex-review P1 指摘の是正）

本 PR（#1177 是正）時点で `crates/backend-cuda/src/gemm_auto.rs::
MMA_PRIORITY_PRODUCTION_ENABLED` を `false`（wmma 優先。#1156 以前と同じ
本番挙動）へ戻した。mma 優先順位の判定ロジック自体（`select_f16_matrix_
unit_impl`）は §5.6 の設計どおり実装・単体テスト済みのまま維持しており、
`prefer_mma` 引数を渡せば有効化できる。本番結線（同定数を `true` へ）は、
下記手順による #1160 の GB10 実機実測で全対象形状の非後退を確認し、この
記録へ追記したうえでユーザー承認を得てから行う。

**数値一致（parity）の実測記録は #1158 で GB10 実機確認済み**
（`docs/perf/cuda-parity-baseline.md` §12。切替前〈本番既定・WMMA 優先〉・
切替後〈ノード側限定フリップの mma 優先〉双方で 5 回実行同値・既存
baseline 行からの後退なしを確認。フリップ自体はコミットしていない）。
**ただし PR #1178 レビュー対応（codex-review 指摘）で追加した
route-aware 受け入れテスト
（`run_f16_k4096_stress_non_regression_route_aware`）自体は、選択経路
`MmaF16` の baseline ceiling が未承認〈`None`〉のため現 HEAD 上では
fail-closed に必ず FAIL する意図的な red のままである（同 §12.4/§12.5）。
上記「実測確認済み」は実測事実の記録を指し、その記録を検証する受け入れ
テストの green を主張するものではない。** 本ドキュメントが対象とする
**性能 A/B・恒久フリップ判断**は未実測のまま #1160 へ引き継がれている。

**前バージョンの記録（実測完了・全形状非後退）は取り下げる。** 前バージョンが
「5 回計測中央値」の根拠とした計測は `bench_run`（`MeasurementConfig::default`。
warmup 20 回・計測 20 回）を各形状 1 回だけ呼び出し、その 1 回の実行内で得られた
20 サンプルの中央値を「計測中央値」として扱ったものであり、`docs/dispatch-rules-
design.md` §5.6・`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
採用し」が求める**独立した 5 回の計測それぞれの中央値**とは異なる統計量だった
（codex-review PR #1177 P1 指摘）。20 サンプルは同一プロセス内の反復であり、
プロセス起動・クロック挙動・熱条件等の独立実行間ばらつきを捕捉できないため、
この 1 回計測を根拠に非後退を確定させることはできない。

是正（1 回目）として、計測用バイナリ `examples/gemm_auto_f16_mma_switch_
bench.rs` を各形状で `bench_run` を同一プロセス内ループで**5 回**呼び出し、
5 個の run 値とその中央値（run-median）を出力するよう修正したが、これも
同一プロセス内の反復に過ぎずプロセス起動・クロック・熱条件の独立実行間
ばらつきを捕捉できない点は変わらないとの codex-review 再指摘（PR #1177
2 回目）を受け、**是正（2 回目）**として計測用バイナリはプロセス起動ごとに
各形状 1 回だけ計測して終了する設計へ変更し、独立 5 回起動・run-median の
集約は外側の `scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh`（実プロ
セスとして 5 回起動する wrapper）へ切り出した（詳細は下記「実機実行手順」
節）。しかし本エージェントの実行環境には CUDA 実機（DGX Spark GB10 等）が
存在しないため、この新プロトコルでの実測を本 PR では行えていない。前
バージョンに記載していた具体的な TFLOPS 値は上記の理由により受け入れ判定の
根拠として使えないため本ドキュメントからは削除し、実測は行われていないと
明記する。

先行根拠として、`docs/perf/cuda-wmma-f16-perf-triage.md` §3.1/§4.1（イシュー
#1123・2026-09-03 GB10 実機実測）のカーネル単体計測（`bench-harness` の
5 回計測中央値プロトコルに準拠）が `mma_sync_f16` は `wmma_f16_opt`/
`wmma_f16_basic` に対し形状依存で約 4.1〜10.8 倍高速であることを確認済みで
あり、mma 優先化の方向性自体はこの参考値が支持する。ただし `CudaGemmAuto::
run_f16` 経由の auto 経路（転送込み・§5.6 の分岐切替を経た経路）そのものの
「独立 5 回計測中央値」による正式な非後退判定は、GB10 実機を持つ後続作業
（#1160）へ引き継ぐ。

## 実機実行手順（GB10 実機保有者が本ドキュメントを更新する際の手順）

転送は rsync 方式（`docs/real-hardware-verification-env.md`。ホスト名等はローカル
管理値を使い本ドキュメントには書かない）。

計測用バイナリは `examples/gemm_auto_f16_mma_switch_bench.rs`（`bench-harness::
run`・`MeasurementConfig::default`〈warmup 20・計測 20〉による `bench_run` を
プロセス起動ごとに各形状 1 回だけ実行し、`CudaGemmAuto::run_f16` を転送込み
で計測する。`examples/gemm_mma_bench.rs` と同じ計測コア・シード）。独立 5 回
起動・run-median の集約は wrapper スクリプト `scripts/bench/
run_gemm_auto_f16_mma_switch_bench.sh` が担う（`cargo build` で 1 回ビルド
した後、生成された実行ファイルを外側のシェルから 5 回 fork+exec する。
真に独立した OS プロセスとして 5 回実行することでプロセス起動・クロック・
熱条件の独立実行間ばらつきを反映する）。

```bash
bash scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh
```

同一バイナリソースを base（`0c91218`。結線前 = `MatrixUnit` 判定時に wmma を
直接使用）／after（PR #1177 是正後の HEAD で `crates/backend-cuda/src/
gemm_auto.rs` の `MMA_PRIORITY_PRODUCTION_ENABLED` を一時的に `true` へ
書き換えたワークツリー。書き換えないまま HEAD をビルドすると本番既定
〈`false`〉のため base と同じ wmma 優先経路が計測されてしまう点に注意）
それぞれへコピーして上記 wrapper スクリプトを実行し、出力される
`size=<dim> auto_f16_tflops_runs=[<5値>] auto_f16_tflops_run_median=<値>` の
`auto_f16_tflops_run_median`（独立 5 回起動の run-median）同士を突き合わせる
（base 側は `internal-diagnostics` feature 未導入のためデフォルト feature で
ビルド。本バイナリはこの feature に依存しない）。全形状で after の
`auto_f16_tflops_run_median` が base 以上であれば非後退と判定し、この記録へ
実測値を追記したうえで `MMA_PRIORITY_PRODUCTION_ENABLED` の恒久的な `true`
化を別途ユーザー承認込みで行う（#1156 R5 の承認条件）。後退する形状があれば
`MMA_PRIORITY_PRODUCTION_ENABLED` は `false` のまま維持し、原因調査を別途
行う。

## 補足: 転送込み計測の位置づけ

`examples/gemm_mma_bench.rs::measure_mma_f16` は GPU 実行のみ（H2D/D2H を
計測区間外）を計測するのに対し、本ベンチ（`gemm_auto_f16_mma_switch_bench.rs`）
は `CudaGemmAuto::run_f16` の実利用経路（転送込み）をそのまま計測する。両者は
計測対象が異なるため TFLOPS 値を直接比較しない（`gemm_mma_bench.rs` 側の
既存注記と同じ整理）。
