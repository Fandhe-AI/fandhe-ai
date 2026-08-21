# CUDA GEMM TF32 生 `mma.sync` 経路 A/B 計測記録（#802）

イシュー #802「test(backend-cuda): TF32 mma.sync 経路の数値一致・parity・実機ベンチ確定」の
実測記録テンプレート・再開手順。`crates/backend-cuda/src/gemm_mma_tf32.rs`（`CudaMmaTf32Gemm`。
イシュー #801）が実装した TF32 生 `mma.sync.aligned.m16n8k8.tf32` 経路について、(1) バックエンド間
数値一致回帰・parity 非後退契約の実機確認、(2) 既存 `wmma_tf32`（WMMA C++ API ベース。staged/opt/
basic の 3 段選択）との A/B 実機ベンチ、(3) 本番結線の採否判断、を記録する。

## 1. 位置づけ・前提

- `CudaMmaTf32Gemm`（#801。PR #823 でコミット `09f9f98`）は **本番非結線**の直接指定 API であり、
  `ops.rs`／`gemm.rs`／`gemm_auto.rs` のディスパッチからは呼ばれない
  （`docs/cuda-tensor-core-design.md` §15 冒頭「位置づけ」参照。#803 の main 追従マージで
  §14 = warp タイル拡大設計、§15 = 本 TF32 経路へ節番号を振り直し済み）。
- 同梱の実機テスト（`crates/backend-cuda/tests/gemm_mma_tf32.rs`〈`#[ignore]` 4 本〉・
  `tests/mma_tf32_vs_wmma_tf32_staged.rs`〈`#[ignore]` 2 本〉。計 6 本）は #801 実装セッション・
  #802 本セッションのいずれも DGX Spark GB10 実機へ到達できず**未実行**のまま。
- 本ファイルは #802 の受け入れ条件 3 項（数値一致・parity・実機ベンチ）がいずれも実機実測を要する
  ため、実機到達可能なセッションが引き継いで完了させる前提の記録である。

## 2. 状態: 未計測・実機到達不可でブロック（2026-08-21・イシュー #802 実装セッション）

本セッション冒頭で実機到達性を確認したが、#792／#821（`docs/perf/cuda-gemm-swizzle-ab.md` §7.7.5）
と同型の理由でブロックされたまま変化がない:

- `docs/real-hardware-verification-env.local.md`: リポジトリルート・`docs/` 配下いずれにも存在しない
  （存在するのは Git 管理下のテンプレート `docs/real-hardware-verification-env.local.md.example`
  のみ）
- `~/.ssh/config`: 存在しない
- 参考: 本セッションの実行環境には `nvidia-smi` 到達可能な GPU（NVIDIA GeForce RTX 3060）が
  存在することを確認したが、対象実機（DGX Spark GB10・sm_121・aarch64。SSH リモート実行前提。
  `docs/real-hardware-verification-env.md` §2）とは別デバイスであり、実機実測の代替にはならない
  （推定値・未実測値の記録は禁止。`docs/perf/cuda-parity-baseline.md` §2 検査 5 項）

したがって本セッションは実装計画の前提ゲート（実機到達性確認）で fail-closed に停止し、以下のみを
実施した:

1. `crates/backend-cuda/examples/cuda_floor_bench.rs` へ `measure_mma_tf32`（既存 4 経路・
   `measure_tiled_f32`／`measure_wmma_tf32`／`measure_wmma_f16`／`measure_mma_f16` と同一の
   launch-only 計測境界）を追加し、`mma_tf32` 列を出力する。**`best_f32`（f32 候補下限の算出
   ロジック）には一切組み込まない**（実機実測・採否判断が出るまでは参考列に留める）。
2. 本ドキュメントの新設（実測テンプレート・再開手順の記録）。
3. `docs/cuda-tensor-core-design.md` §15.6 へ本状態への参照を追記。

**未実施のまま残る事項**（実機到達可能セッションへの引き継ぎ）:

- 数値一致回帰・parity 非後退契約の実機実行（§3）
- `cuda_floor_bench` の実機ベンチ 5 回計測・A/B 記録（§4）
- 採否判断（§5）・（採用時のみ）本番結線
- `docs/perf/cuda-parity-baseline.md` への実測値追記

## 3. 再開手順（実機到達可能セッション向け）: 数値一致・parity 非後退の実機実行

1. `docs/real-hardware-verification-env.md` §2/§3 に従いコード転送・PATH 設定を行う。
2. 以下を実行し、実行ログを本節へ追記する:

   ```sh
   # rust libtest の位置引数 FILTER は 1 個のみ受理する（2 個目以降は
   # unexpected argument になり実行不能）ため、`--test <file>` でテスト
   # バイナリを限定したうえで 1 呼び出し 1 FILTER に分割する。

   # crates/backend-cuda/tests/gemm_mma_tf32.rs（#[ignore] 4 本）
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     mma_tf32_matches_reference_across_shapes
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     mma_tf32_k4096_stress
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     mma_tf32_zero_dim_shape_returns_empty_without_launch
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     launch_tf32_zero_dim_shape_is_noop_or_zero_fills_without_launch

   # crates/backend-cuda/tests/mma_tf32_vs_wmma_tf32_staged.rs（#[ignore] 2 本）
   cargo test -p backend-cuda --release --test mma_tf32_vs_wmma_tf32_staged -- --ignored --nocapture \
     mma_tf32_matches_wmma_tf32_staged_across_shapes
   cargo test -p backend-cuda --release --test mma_tf32_vs_wmma_tf32_staged -- --ignored --nocapture \
     mma_tf32_matches_wmma_tf32_staged_k4096_stress

   cargo test -p backend-cuda --release --test parity_nonregression -- --ignored --nocapture \
     parity_baselines_do_not_regress
   ```

   （テスト名は `cargo test -p backend-cuda --test <file> -- --list` で実測確認済み
   〔2026-08-21〕。`--ignored` 実機テストは `parity_nonregression.rs` 内では
   `parity_baselines_do_not_regress` の 1 本のみで、ファイル名そのものをフィルタ文字列に
   使うと 0 件マッチの偽 green になるため注意。他 8 本〔`tolerance_constants_are_pinned` 等〕
   は `#[ignore]` なしの通常 CI 対象で GPU 不要）

3. **既知リスク**: TF32 系経路は CPU f32 参照との統一複合判定（相対誤差 1e-3 未満 または絶対誤差
   1e-5 未満）で恒常 fail の既知状態（`docs/perf/cuda-parity-baseline.md` §1。wmma_tf32 系で fail
   比率 15〜16%）。`mma_tf32` が大形状（512³・4096³）で CPU 参照直接照合に fail する可能性が高い。
   その場合:
   - **tolerance・判定式は一切緩和しない**（`.claude/rules/coding-rust.md`・`security.md` によりユーザー
     承認必須）
   - GPU-GPU 相互一致（vs `wmma_tf32` staged）green を数値一致の一次根拠とし、CPU 参照恒常 fail
     形状は parity 非後退契約の枠組みへ移行する（`tests/common/parity_baseline.rs` へ実測値を
     **初回記録**として追加。既存行の上方更新〔緩和〕は行わない。初回記録・下方更新はユーザー承認
     不要 — `docs/perf/cuda-parity-baseline.md` の承認記録節）。
4. 実測結果（fail_count・mean_abs_diff・出典テスト・実測環境）を `docs/perf/cuda-parity-baseline.md`
   §3 表へ追記する。

## 4. 再開手順: `cuda_floor_bench` 実機ベンチ・A/B 記録

```sh
cargo run -p backend-cuda --example cuda_floor_bench --release
```

- サイズ 512／1024／2048／4096 の `wmma_tf32_tflops`・`mma_tf32_tflops`・
  `mma_tf32_over_wmma_tf32(...)` 行を **5 回起動**して記録し、各サイズの中央値を採る
  （`CLAUDE.md`「5 回計測中央値」規約。`docs/perf/cuda-gemm-swizzle-ab.md` の運用と同型）。
- 生ログ・5 回分の値・中央値・比率を下表へ転記する。
- `mma_tf32_over_wmma_tf32` は `wmma_tf32` が **staged 経路**へ実際にルーティングされた形状
  でのみ算出される（`gemm.rs::CudaGemm::wmma_tf32_routed_path_is_staged` で判定。staged
  カーネルが未コンパイル・未整列形状で opt／basic へフォールバックした場合は `n/a` になる。
  codex-review 指摘対応。PR #826）。該当実機で `n/a` が出力された場合、staged 経路が不能な
  環境（`docs/perf/cuda-gemm-mma-tf32-ab.md` 実機の cc・cp.async 対応状況を確認）である旨を
  本節へ追記し、§5 の採否判断には使わない。

### 4.1 実測記録テンプレート（実機到達可能セッションが埋める）

計測環境: GPU 名 = （未計測）・sm = （未計測）・driver 版数 = （未計測）・実行コミット SHA = （未計測）

| size | wmma_tf32 中央値 (TFLOPS) | mma_tf32 中央値 (TFLOPS) | mma_tf32/wmma_tf32 比 |
|---|---|---|---|
| 512  | 未計測 | 未計測 | 未計測 |
| 1024 | 未計測 | 未計測 | 未計測 |
| 2048 | 未計測 | 未計測 | 未計測 |
| 4096 | 未計測 | 未計測 | 未計測 |

生ログ（5 回分。実機到達可能セッションが `cuda_floor_bench` の標準出力を追記する）: （未計測）

## 5. 採否判断（安全側に固定した採用条件）

**採用条件**: 判定対象形状（2048・4096。REQ-8 の演算律速域）で `mma_tf32` が `wmma_tf32`
（staged）を上回り、かつ 512・1024 で劣化 5% 超がないこと。満たさなければ**結線しない**
（現状維持を採否判断として記録し、部分改善のみの場合はサイズ条件付き適用〈swizzle 前例。
`docs/perf/cuda-gemm-swizzle-ab.md` §2〉の検討をフォローアップ Issue として提案するに留める）。

判断: **未計測のため判定不能（本セッションでは実施していない）**。

### 5.1 採用時の結線内容（メモ。実施は採用判断確定後）

`gemm.rs::run_wmma_tf32` の 3 段選択の最優先段として `mma_tf32` を追加する場合は、PR #678 の教訓に
従い以下を守る:

1. 実効ルーティング経路の parity 検査を新経路向けに追加する。
2. 既存 `wmma_tf32_staged` ベースライン行の検査が黙って経路すり替えにならないよう直接起動経由の
   検査へ移設する。
3. 結線後に実機で数値一致・非後退・`cuda_floor_bench` を再実行して確認する。

## 6. スコープ外（追跡）

- TF32 タイル定数拡大（Phase 4・#806。診断機構・机上候補表は整備済み。
  `docs/perf/cuda-gemm-mma-tf32-block-tile.md` 参照。実機実測・本番採用は
  実機到達可能セッションへの引き継ぎのまま）
- swizzle 変種の TF32 `mma.sync` への適用（実測を伴うため別途）
- REQ-8 下限値の再確定（候補値の記録まで。確定は人間判断・TASK-8.3 系）
- 部分改善時のサイズ条件付き適用の実装（採否判断で必要と出た場合にフォローアップ Issue を提案）

## 7. #806 との相互参照

イシュー #806（本節見出し §6 の「TF32 タイル定数拡大」）は本イシュー
（#802）と同一の実機到達不能セッション制約（§2）を引き継ぎ、Step F
フォールバックとして診断機構（`kernels_mma_tf32.rs::
mma_tf32_source_with_block_tile`）・机上候補表・`examples/
mma_tf32_ptx_dump.rs` を整備した（`docs/perf/
cuda-gemm-mma-tf32-block-tile.md`）。実機到達可能セッションでは、本
ドキュメント §3・§4（数値一致・parity・`cuda_floor_bench` A/B 計測）と
`cuda-gemm-mma-tf32-block-tile.md` §6・§8（タイル拡大候補の `ptxas -v`
実測・4096/2048 ベンチ）を同一セッションでまとめて消化できる（両者とも
DGX Spark GB10 実機・CUDA 13.0 toolkit を要求する点が共通のため）。
