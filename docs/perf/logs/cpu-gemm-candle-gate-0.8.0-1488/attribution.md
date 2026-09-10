# 差分帰属（#1321 参考系列 `ced4d14` → `v0.8.0`）

`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §22.4 が記録した参考系列
（`head-ced4d14-1321`。#1321 実測時点の origin/main HEAD `ced4d14f`）から
本イシュー（#1488）で計測する正式系列 `fandhe-ai =0.8.0` の間に入った、
CPU GEMM 計測経路（`crates/backend-cpu`・`crates/facade`・
`crates/tensor-core`・`crates/autodiff`）へのコミットは以下の 4 件のみ
（`git log --oneline ced4d14f..v0.8.0 -- crates/backend-cpu crates/facade
crates/tensor-core crates/autodiff` 実測）。

```
81411488 build(workspace): v0.8.0 へ一括バンプする（#1269 完了後の引き継ぎ改善の公開前提） (#1503)
29bfa117 docs(backend-cpu): 改定版規則 4 の判定結果に基づき GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS の本番既定を確定する (#1502)
a68c08b0 test(facade): CUDA Graph step capture の bit 同一性比較へ各 step の重み勾配生値を追加し GB10 で確認する (#1495)
93e11a71 feat(facade): resident GradStaging の重み勾配をホストへ読み出す公開 API を追加する (#1492)
40ef8906 build(bench): host-view-readout を bench-fandhe の既定経路へ取り込み feature ゲートを撤去し、3 バックエンドの candle 比ゲートを再計測する (#1452)
```

## 実コードの実値確認（grep 実測。2026-09-10）

| 定数 | 値（`v0.8.0`/origin/main で共通） | 出典 |
|---|---|---|
| `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS` | `usize::MAX`（#1502 で確定した本番既定。CPU matmul 固定費削減は無効のまま） | `crates/backend-cpu/src/ops.rs:159` |
| `TWO_D_DYNAMIC_PRODUCTION_ENABLED` | `true`（#1313 で ADOPT・本番結線済み。不変） | `crates/backend-cpu/src/gemm_blis/mod.rs:3008` |
| `TWO_D_JOBS_PER_WORKER` | `2`（不変） | `crates/backend-cpu/src/gemm_blis/mod.rs:2987` |
| `BIG_CORE_LIMIT_ENABLED` | `false`（#1364 で REJECT・差し戻し済み。不変） | `crates/backend-cpu/src/thread_limit.rs:93` |

## 結論

- #1492／#1495（GradStaging 読み出し API）は backward／update（学習 step）
  限定の追加 API であり、GEMM forward 計測経路（`Var::matmul` reuse →
  `gemm_blis` 系）には触れない
- #1502 は #1481（PR #1448 の是正）で `usize::MAX`（無効化のまま）へ
  差し戻した結果を**再確認して既定として確定**したドキュメント更新であり、
  値自体は `ced4d14` 時点から不変（#1481 の計測でも `usize::MAX` のまま
  だった）
- #1503 は v0.8.0 へのバージョン一括バンプ（`Cargo.toml` の
  `workspace.version` のみ）
- 本イシューの §0（`v0.8.0 ↔ origin/main` の CPU src 差分ゼロ。
  `diff_v0.8.0_a2a1c86_cpu_path.txt` 参照）と合わせ、**#1321 参考系列
  `head-ced4d14-1321` から本イシューの正式系列 `0.8.0-1488` までの
  CPU GEMM 計測経路の実質差分は「借用ビュー readout の既定経路化
  （#1438／#1452。旧 `host-view-readout` cargo feature 撤去）」のみ**で
  あり、`gemm_blis` 本体（2D 動的分配・並列ゼロ埋め・大コア限定スレッド数）
  はいずれも不変であることを実コードの grep で確認した
