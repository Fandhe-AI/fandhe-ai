# CUDA GEMM tiled f32（classic）経路のブロック実行順スウィズル A/B・本番結線可否判断（#1139）

イシュー #1139「perf(backend-cuda): ブロック実行順スウィズル（#1034）の GB10 実測に基づき本番結線可否を
判断する」の記録。**本ドキュメントは GB10 実機 A/B の完了記録ではなく、実機到達不能によりゲート 0 を
実行できず判断を保留（ブロック）した記録である**（実測値の捏造・推定記入はしない。#713 方針）。

## 1. 背景

- #1034（PR #1070）で本番既定 f32 カーネル `kernels::TILED_F32`（classic）に L2 局所性向上のための
  ブロック実行順スウィズル（`swizzle.rs::swizzled_block_idx` と同一の remap 整数式）を、
  `internal-diagnostics` feature 限定の診断用 opt-in 入口（`gemm.rs::CudaGemm::new_with_tiled_f32_swizzle`・
  `kernels.rs::tiled_f32_source_with_swizzle`・`examples/gemm_tiled_f32_swizzle_bench.rs`・
  `diagnostics::tiled_f32_swizzle_group_width`）として追加したが、GB10 実機実測が未実施のまま本番既定
  コンストラクタ `CudaGemm::new` へ未結線の状態が続いていた。
- 本イシューは同 variant を GB10 で 5 回計測中央値で A/B し、数値一致（複合判定・bit 一致）と手動境界
  チェック維持を確認したうえで、事前宣言した判定基準（下記 §3）に基づき結線可否を判断することを目的と
  した。

## 2. #1164 マージ後の到達性整理（実測前に判明した設計上の重要事実）

作業ブランチを `origin/main`（`b68590a`）から切った時点で、ローカル `main`（`1a32082`）に対し以下 2
コミットが既に取り込まれていた:

- `9383387`（PR #1163・#1136）: classic `TILED_F32` の結線前 baseline を
  `docs/perf/cuda-gemm-simt-register-blocking.md` §7 に記録。
- `b68590a`（PR #1164・#1137）: **cp.async 3 stage パイプライン（#1033）を `CudaGemm::run_tiled_f32`
  系 3 入口へ形状条件付きで本番結線済み**。

`run_tiled_f32`／`launch_tiled_f32`／`launch_tiled_f32_resident` は `select_tiled_f32_kernel`
（純粋関数 `tiled_f32_kernel_kind`）で「pipeline コンパイル成功 かつ `a_offset % 4 == 0` かつ
`n % 4 == 0 && k % 4 == 0`」なら **Pipeline**、それ以外は **Classic**（`TILED_F32`）へ分岐する。

イシューが判定形状として明記する N=1024/2048/4096 の正方形（整列形状）は、いずれも `n % 4 == 0 &&
k % 4 == 0` を満たすため、**現在の本番既定経路はこれらの形状で classic を通らない**。したがって classic
への swizzle 結線が本番で効くのは classic フォールバック形状（非整列 n/k・非整列 offset・pipeline
コンパイル失敗環境）に限られる。この事実は #1164（PR #1164 本文・`docs/perf/cuda-gemm-tiled-pipeline.md`
「スコープ外事項」節）で本イシューへ明示的に引き継がれている。

## 3. 判定基準（実測前に宣言。実測後に動かさない）

計測はすべて GB10・GPU utilization 0% 確認後・`--release --locked`・bench-harness 既定（warmup 20 /
iters 20 の中央値）を 5 回実行して run 間中央値を採る（`.claude/rules/coding-rust.md`「5 回計測の
中央値」）。

- **ゲート 0（数値一致・境界検査。最優先・必須）**
  - `tiled_f32_swizzle_variant_matches_base_bit_exact_output` PASS（bit 一致）
  - `tiled_f32_swizzle_variant_matches_cpu_reference` PASS（統一複合判定。相対誤差 1e-3 未満 または
    絶対誤差 1e-5 未満・`assert_parity` 厳密ゼロ fail）
  - `kernels.rs` の REQ-8 境界ガード実在検査 PASS
  - 1 つでも fail → 不採用
- **ゲート 1（classic 内 A/B: swizzle の純効果）**: N=4096 ≥ 1.05 かつ N=512/1024/2048 いずれも
  ≥ 0.95 で採用条件。満たさなければ不採用。
- **ゲート 2（本番ディスパッチ到達性・非後退）**: ゲート 1 通過時のみ実施。整列形状（Pipeline 経路）で
  非後退・classic 到達形状（代表 M=N=K=4098）で ≥ 1.05・非整列/非正方の参考形状で ≥ 0.95。満たさなければ
  結線せず不採用として記録する。

## 4. 実測結果: 未実施（実機到達不能）

- ゲート 0（bit 一致・複合判定・REQ-8 境界ガード実在検査）は GB10 実機（`ignore` 属性の実機テスト
  `tiled_f32_swizzle_variant_matches_base_bit_exact_output`／`_matches_cpu_reference`）を要するが、本
  イシューの実装エージェント実行環境からは GB10 ノードへネットワーク到達できない
  （`ssh <cuda-node>` → `ssh: Could not resolve hostname <cuda-node>: nodename nor servname provided,
  or not known`）。ローカル環境変更・DNS 設定変更・別ホストへの経路探索は本エージェントの隔離された
  worktree の権限・スコープ外であり、リトライしても解消しない性質の到達不能である。
- したがって §3 のゲート 0（必須・最優先）を満たすことができず、ゲート 1・ゲート 2 の実機計測も実施
  できなかった。**実測値は一切記入していない**（推定値・過去の近傍計測からの類推値の代入も行わない。
  §2 の到達性整理は静的なコードパス解析であり実測ではない）。
- **判定: ブロック（判断保留）**。#791/#792 の先例（実機到達不能時は結線せず「ブロック」として記録して
  終了する）に従い、`internal-diagnostics` feature 限定の診断用 opt-in 入口はそのまま温存し、本番既定
  コンストラクタ `CudaGemm::new` への結線は行わない。

## 5. 本 PR で実施した是正（判定結果に依らず実施。ゲート 0 の一部を静的に強化）

ゲート 0 のうち「REQ-8 境界ガード実在検査」は実機を要さないため、通常 CI（GitHub ホステッド）で常時
実行できる形で新設した:

- `crates/backend-cuda/src/kernels.rs::tests::tiled_f32_source_with_swizzle_preserves_req8_boundary_guards`:
  `tiled_f32_source_with_swizzle` が生成する各 `group_width` のソースに、A/B タイルロードの三項ガード
  （`(row < m && col < k) ? … : 0.0f;` 等）とエピローグ store の `if (row < m) { … if (col < n) { … } }`
  が残存することを機械検査する。

またグルーピング幅選択の単位不一致を是正した（実測結果には影響しないが、単一真実源を揃える是正）:

- `diagnostics::tiled_f32_swizzle_group_width`（`lib.rs`）と `gemm.rs` の bit 一致テスト・CPU 参照
  テストは、#1032（PR #1072。#1070 の 23 分後にマージ）以降 `TILED_F32` のブロックタイルが
  `TILED_F32_BM`/`BN`（64）であるにもかかわらず、旧素朴カーネル基準の `kernels::TILE`（32）を
  `select_swizzle_group_width` へ渡していた。GB10（SM 数 48）では 32 単位・64 単位いずれでも最小候補
  g8 が選ばれるため過去の実測結果自体への影響はないが、`TILED_F32_BM`/`BN` へ揃えた。
  `swizzle.rs::select_swizzle_group_width_pins_gb10_sm_count_48_tiled_f32_bm_bn_64_to_g8` で
  `select_swizzle_group_width(48, 64, 64) == 8` を機械的にピン留めした。
- `kernels.rs`／`gemm.rs` の陳腐化していたドキュメンテーションコメント（`TILED_BLOCK_DIM`／
  `kernels::TILE` 前提の記述）を `tiled_f32_launch_config`／`TILED_F32_BM`/`BN` 前提へ修正した。

いずれもコードの実行時挙動（swizzle remap 式・境界チェック・カーネルディスパッチ）は変更していない
（`git diff` は `lib.rs`/`gemm.rs`/`kernels.rs`/`swizzle.rs` のドキュメンテーションコメント・テスト
追加・定数参照の是正のみ）。

## 6. 申し送り（次にこのイシューを引き継ぐ場合）

- GB10 実機へネットワーク到達可能な環境（`docs/real-hardware-verification-env.md` §2〜§4 の rsync 転送
  手順が使える環境）で、本ドキュメント §3 の判定基準をそのまま適用し、ゲート 0〜2 を実施すること。
  判定基準自体は変更不要（実測前に宣言済みであり、今回の到達不能によって妥当性が損なわれる性質のもの
  ではない）。
- §2 の到達性整理（classic への結線が本番で効くのは classic フォールバック形状に限られる）を踏まえ、
  ゲート 2 の測定形状は判定基準どおり「整列形状（非後退確認）＋ classic 到達形状（代表 M=N=K=4098。
  効果測定）＋ 非整列/非正方の参考形状」を維持すること。
- pipeline カーネル（`kernels_tiled_pipeline.rs`）への swizzle 横展開は本イシューのスコープ外のまま
  （PR 本文の「切り出し提案」参照）。

## 7. 関連ドキュメント

- `docs/perf/cuda-gemm-swizzle-ab.md`（f16 `mma.sync`・TF32 opt-staged 経路の同型記録。§8 に本ファイルへの
  ポインタを追加）
- `docs/perf/cuda-gemm-tiled-pipeline.md`（#1137。cp.async パイプラインの本番結線記録。「スコープ外事項」
  節で #1034 結線判断を本イシューへ引き継ぎ）
- `docs/perf/cuda-gemm-simt-register-blocking.md`（#1136。classic 経路の結線前 baseline）
