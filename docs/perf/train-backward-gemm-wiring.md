# VJP の GEMM 系呼び出しを `BackendOps::gemm` へ切替（イシュー #1211）

## 0. 目的・スコープ

`docs/perf/train-step-phase-breakdown.md` §15 が確定した「backward が
CPU/CUDA/Metal 全バックエンド共通の支配項（75.1〜97.5%。v0.6.0 実測）」に
対する改善策の第一弾。§15.3 のコード事実が指摘したとおり、`crates/autodiff/
src/grad.rs` の VJP 経路は 3 箇所で `crates/autodiff/src/eval.rs::matmul`
（scalar 三重ループ・`rayon` なし・`BackendOps` 非経由のホスト参照実装）を
呼んでいた:

1. `matmul_vjp`（`Op::MatMul`〈`grad.rs:66`〉・`Op::LinearAct`〈`grad.rs:327`〉
   から呼ばれる。fresh 経路）
2. `Op::LinearResident` の `d_weight = xᵀ·g`（reuse 経路）

本イシューはこの 3 箇所を forward と同じ `BackendOps::gemm`（CPU は BLIS
並列 GEMM・CUDA/Metal はデバイス GEMM）経由へ切り替える。転置オペランド
（`transpose2d` の zero-copy view）はそのまま `ops.gemm` へ渡し、各
バックエンドの `gemm` 実装内の `contiguous()` による再パックを本イシューの
受け入れ範囲として許容する（NT/TN 専用の zero-copy 入口は後続イシュー
#1213〈CPU〉・#1214〈CUDA〉・#1215〈Metal〉のスコープ）。reuse 経路の grad
をデバイス常駐のまま `device_update` へ直結する変更は #1212（本イシューに
依存）のスコープであり本イシューでは扱わない。

エラーは fail-closed で `AutodiffError::Backend` として伝播し、
`eval::matmul` への暗黙フォールバックは設けない（forward と backward で
数値経路が分岐する判定迂回を作らないため。`.claude/rules/security.md`
A08）。

## 1. コード変更

| ファイル | 変更 |
|---------|------|
| `crates/autodiff/src/grad.rs` | `matmul_vjp` のシグネチャに `ops: &dyn BackendOps` を追加し、内部を `ops.gemm(g, &b_t)`／`ops.gemm(&a_t, g)` へ置換（戻り値も `Result` 化）。`Op::MatMul`・`Op::LinearAct` の呼び出し箇所を `?` 付きに追従。`Op::LinearResident` の `d_weight` を `eval::matmul(&x_t, g)` から `ops.gemm(&x_t, g).map_err(AutodiffError::Backend)?` へ置換。関連ドキュメンテーションコメント（モジュール doc・`matmul_vjp` doc・`Op::LinearResident` ブロックコメント）を更新 |
| `crates/autodiff/src/eval.rs` | `MATMUL_HOST_REPACK_COUNT` の doc コメントを更新: 本番経路は #1211 で `BackendOps::gemm` 経由になったため、本カウンタが観測するのは `NaiveOps`／`TestOps`（compat・テスト経路）経由の呼び出しに限られる旨を明記 |

`eval::matmul` 本体・`MATMUL_HOST_REPACK_COUNT` カウンタ自体は削除していない
（`NaiveOps`〈`default_ops.rs`〉・`TestOps`〈`test_support.rs`〉・forward の
参照実装・テストが引き続き使用する）。

## 2. 数値一致への影響

- autodiff クレート内の全 `BackendOps` フィクスチャ（`TestOps`・
  `tests/common/mod.rs`・`NaiveOps`）の `gemm` は `eval::matmul` へ委譲する
  ため、autodiff クレートのテストは切替前後で bit 同一（`cargo test -p
  fandhe-ai-autodiff` 全 pass。`grad::tests::matmul_grad_matches_numeric`・
  `vjp_dispatch_matmul_returns_both_inputs`・
  `matmul_vjp_does_not_repack_transposed_operands` 含む）
- facade の bit 一致テスト `compat_sequential_train::
  sequential_training_loop_matches_manual_loop_bit_exact`（本番 `CpuBackendOps`
  経由）も pass を確認した。本番 CPU 経路（BLIS 並列 GEMM）の累積順序が
  `eval::matmul` と異なっても、比較対象（手動合成 `Op::MatMul`＋`Op::Add`
  と `Sequential` の `Op::LinearAct`）が同一プロセス・同一 `CpuBackendOps::
  gemm`・同一 shape/値を通るため等式が壊れないことを実測で確認済み
- `cargo test --workspace --all-features` 全 pass（既存の複合判定・収束
  テストに非後退なし）

## 3. 計測プロトコル

- 環境: Apple M4 Max（macOS 26.6.2・arm64・rustc 1.96.0）。詳細は
  `docs/perf/logs/train-backward-gemm-wiring-1211/env_info.txt`
- ハーネス: `scripts/bench/framework-compare/bench-fandhe --task train
  --device cpu --size 64 --mode {fresh,reuse} --phases`（warmup 20・
  measured iters 80。既存プロトコル不変）
- **参考系列方式**（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` 等の
  先例と同じ）: `scripts/bench/framework-compare/bench-fandhe/Cargo.toml`
  の `fandhe-ai = "=0.6.0"` ピンをコミットせず、`--config
  patch.crates-io.fandhe-ai.path="<facade 絶対パス>"` で before（本 PR の
  変更前 = `git stash` で `grad.rs`/`eval.rs` を退避した状態）／after（本
  PR HEAD）の 2 バイナリを別々の `CARGO_TARGET_DIR` にビルドした。両
  バイナリとも `cargo tree -p bench-fandhe --depth 1` で `fandhe-ai
  (path: …)` が出ることを確認済み（patch 未適用の registry 解決ではない
  ことのハードゲート）
- 計測後 `git checkout -- scripts/bench/framework-compare/Cargo.lock` で
  復元し、`make deps-forbidden` が承認済みピン `fandhe-ai =0.6.0`
  （registry 取得元）を再度検出することを確認した（drift なし）
- 各系列・各 mode を 5 回計測し、`backward`・`step_total`（主判定軸）を
  中心に比較する。生ログは `docs/perf/logs/train-backward-gemm-wiring-1211/
  {before,after}.jsonl`
- 注意: 実行時に他 worktree での並行作業（cargo ビルド等）が走っていた
  可能性があり、値を除外せずそのまま記録する方針を取った
  （`metal-gemm-bottleneck-rediagnosis.md` 等の先例と同方針）

## 4. 実測結果（CPU）

| mode | phase | before（中央値, ms） | after（中央値, ms） | 倍率（before/after） |
|------|-------|----------------------|----------------------|------------------------|
| fresh | backward | 15.329 | 1.321 | **11.60×** |
| fresh | step_total | 15.980 | 1.835 | **8.71×** |
| reuse | backward | 7.233 | 0.817 | **8.86×** |
| reuse | step_total | 7.926 | 1.408 | **5.63×** |

全 4 系列（`before`/`after` × `fresh`/`reuse`）とも計測プロトコルどおり
厳密に 5 run の中央値（3 番目に小さい値）である（codex-review 指摘。
`before`/`fresh` に手順確認用の追加 1 run が混入し 6 run の中央値
〈16.122 ms／16.766 ms〉になっていた問題を修正。PR #1223）。混入していた
run を事後に除外して「5 run」を再構成するのではなく、`before`/`after`
の 2 バイナリ（§3 の参考系列方式でビルド）を同一セッションで
`fresh`/`reuse` とも改めて 5 回ずつ計測し直した。生ログは
`docs/perf/logs/train-backward-gemm-wiring-1211/{before,after}.jsonl`
（各 90 行 = fresh 5 run × 10 phase + reuse 5 run × 8 phase）へ全面差し替え
済み。倍率は前回計測（12.48×・9.24×）と比べ約 1〜1.5 ポイント低下する
が、いずれも大幅な改善（8×〜12× 台）であることに変わりはなく、
採否判断（§5・ADOPT）は変わらない。

その他フェーズ（`forward`／`forward_resident`／`host_sgd`／`device_update`・
`param_readout`／`tape_build`／`tape_drop`／`leaf_register`／`loss_readout`／
`apply_params`）はいずれも変化なし（µs オーダーで前後同水準）。生データの
全フェーズ内訳は
`docs/perf/logs/train-backward-gemm-wiring-1211/{before,after}.jsonl` を
参照。

## 5. 採否判断

**ADOPT**（結線を維持する）。`backward`・`step_total` はいずれも fresh・
reuse の両方で大幅な非後退（改善）を確認した。層 2（64×256×10 の小さい
GEMM）における `CpuBackendOps::gemm` の rayon 分割・packing 固定費が
scalar 参照実装を上回るリスクを事前に想定していたが（本ドキュメント§0・
`matmul-vjp-zero-copy-decision.md` §4）、実測では逆に大幅な高速化が確認
された。CPU BLIS 並列 GEMM の絶対性能が `eval::matmul`（rayon なし scalar
三重ループ）を大きく上回るため、層のサイズによらず改善方向に効くと
解釈できる。

## 6. Metal／CUDA

- **Metal**: 本ドキュメントの対象（`eval::matmul` → `BackendOps::gemm`
  への切替。#1211）**単体**の Metal 実測は引き続き未実施のまま（CPU
  主計測での ADOPT 判定が明確であったため、時間配分上見送った）。
  後続 #1215（Metal NT/TN strided 結線）の train phases フル A/B
  （`docs/perf/metal-gemm-vjp-transposed-entry.md` §3.2）は、本切替
  （#1211）が**既に適用済みの `origin/main`** を before・#1215 の結線
  を加えた HEAD を after として計測したため、#1215 の**増分のみ**を
  測っている（backward: fresh 1.649×・reuse 1.200×、step_total: fresh
  1.323×・reuse 1.109×。この結果は #1215 単体の ADOPT 根拠であり、
  #1211 自体の Metal 寄与を示すものではない）
- **CUDA**: 本セッションには CUDA 実機がないため未実測。#1214 で CUDA
  GEMM の NT/TN 転置入口自体の実装・GPU 非依存テストは完了したが、同
  セッションにも CUDA 実機がなく GB10 実測は未実施のまま
  `docs/perf/cuda-gemm-vjp-transposed-entry.md` に記入欄を残した
  （実施は別途の DGX Spark GB10 セッション）

## 7. 後続イシューへの引き継ぎ

- #1212: reuse 経路の grad をデバイス常駐のまま `device_update` へ直結
  （本イシューに依存）
- #1213: CPU BLIS の NT/TN 専用入口（`contiguous()` 再パック解消）→
  完了（`docs/perf/cpu-gemm-vjp-transposed-entry.md`・`docs/matmul-vjp-
  zero-copy-decision.md` §4.2 追補）
- #1214: CUDA GEMM の NT/TN 転置入口 → 実装・GPU 非依存テストは完了
  （`docs/perf/cuda-gemm-vjp-transposed-entry.md`・`docs/matmul-vjp-
  zero-copy-decision.md` §4.3 追補）。GB10 実機実測・本変更の CUDA
  実測は同 doc §3〜§4 に記入欄を残したまま未実施
- #1215: Metal GEMM の NT/TN strided 結線 → 完了（`docs/perf/metal-gemm-
  vjp-transposed-entry.md`・`docs/matmul-vjp-zero-copy-decision.md`
  §4.4 追補。M4 Max 実機実測で ADOPT 確定）
