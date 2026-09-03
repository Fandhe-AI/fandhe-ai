# CUDA GEMM tiled f32（classic）経路のブロック実行順スウィズル A/B・本番結線可否判断（#1139）

イシュー #1139「perf(backend-cuda): ブロック実行順スウィズル（#1034）の GB10 実測に基づき本番結線可否を
判断する」の記録。**GB10 実機でゲート 0〜1 を実施し、ゲート 1（classic 内 A/B）が N=512 で不合格
（5 回計測中央値が判定基準 ≥0.95 を下回る）となったため、事前宣言済みの判定基準（§3）に従い不採用と
判断した**（実測値の捏造・推定記入はしない。#713 方針。以前のブロック〈判断保留〉記録は本更新で置換）。

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

## 4. 実測結果（GB10 実機。2026-09-03。env_info・生ログは `docs/perf/logs/cuda-tiled-f32-swizzle-1139/`）

### 4.1 ゲート 0（数値一致・境界検査）: PASS

`--release --locked --features internal-diagnostics` で以下すべて PASS:

- `gemm::tests::tiled_f32_swizzle_variant_matches_base_bit_exact_output` … ok（bit 一致。動的幅・g8・g16
  × 3 形状〈(256,256,256)・(80,136,160)・(544,256,2048)〉全て。GB10 実機・`gate0_bitexact_parity.log`）
- `gemm::tests::tiled_f32_swizzle_variant_matches_cpu_reference` … ok（統一複合判定〈相対誤差 1e-3 未満
  または絶対誤差 1e-5 未満〉・`assert_parity` 厳密ゼロ fail。GB10 実機・`gate0_bitexact_parity.log`）
- `kernels::tests::tiled_f32_source_with_swizzle_preserves_req8_boundary_guards` … ok（REQ-8 境界ガード
  実在検査。実機を要さない静的検査のため GB10 実測ログとは別に採取・`gate0_req8_boundary_guard.log`）

### 4.2 ゲート 1（classic 内 A/B。swizzle remap の純効果）: **不合格**

`examples/gemm_tiled_f32_swizzle_bench.rs` を計測サイズ据え置き（512/1024/2048/4096・group_width ∈
{動的選択〈GB10 では 8〉, 8, 16}）のまま 5 回実行した生ログが
`swizzle_bench_gate1_run{1..5}.log`。各サイズ・各幅について 5 run の中央値（`swizzle_gN_over_base`。
動的選択幅は GB10 では g8 と同一のため列が重複する）:

| size | 判定基準 | g8（動的） | g8（明示） | g16（明示） | 判定 |
|------|---------|-----------|-----------|------------|------|
| 512  | ≥ 0.95  | **0.9447** | **0.9450** | **0.9414** | **NG（全幅で未達）** |
| 1024 | ≥ 0.95  | 0.9881 | 0.9877 | 0.9884 | OK |
| 2048 | ≥ 0.95  | 1.0045 | 1.0040 | 1.0058 | OK |
| 4096 | ≥ 1.05  | 1.1280 | 1.1275 | 1.1230 | OK |

N=512 は 3 幅 × 5 run の 15 値のうち 14 値が 0.9155〜0.9461 で判定基準 ≥0.95 を下回り、5 回中央値
（g8〈動的〉0.9447・g8〈明示〉0.9450・g16〈明示〉0.9414）はいずれも基準未達だった（中央値で基準を約
0.5〜0.9 ポイント下回る）。唯一 run1 の g8（明示）のみ 0.9557（`swizzle_bench_gate1_run1.log:5`）と基準を上回ったが
単発（5 run 中 1 run）であり、判定基準§3 が採る「5 回計測中央値」では不合格側に確定する。再現性のある
劣化であり、N=4096 の
改善（+12.3〜12.8%）と N=512 の劣化はトレードオフの関係にある（swizzle remap の追加算術オーバーヘッドが
小さい GEMM で相対的に効いてくるとみられる。ncu 等のプロファイラによる原因分解は未実施・本イシューの
スコープ外）。

**判定: ゲート 1 不合格（§3「N=512/1024/2048 いずれも ≥0.95」を N=512 が満たさない）。**

### 4.3 ゲート 2: 実施せず（ゲート 1 不合格のため §3 の手順どおり不採用判断には使わない）

判定基準 §3 は「ゲート 2 はゲート 1 通過時のみ実施」と定めており、§5 の採否判断はこの順序を守って
ゲート 1 の結果のみに基づいている。

ゲート 1 判定前の実測時点では、一時的に本番結線コード（`CudaGemm::new` へのサイズ条件付き swizzle
構築・`select_tiled_f32_kernel` classic 分岐への統合・`new_without_tiled_f32_swizzle` 診断入口）を
作業ツリー上に置き、`swizzle_bench_gate1_run{1..5}.log` の採取に付随して「本番ディスパッチ到達性」の
参考値（`dispatch_over_base`）も記録していた。しかしこの一時実装はコミットせずゲート 1 不合格を
確認した時点で `git checkout` により作業ツリーから完全に除去しており、現在の HEAD（本 PR）の
`examples/gemm_tiled_f32_swizzle_bench.rs` はゲート 1（classic 内 A/B）のみを出力する。生ログ
（`swizzle_bench_gate1_run{1..5}.log`）には当時の "gate2:" セクション出力がそのまま残存しているが、
これを生成したコード（`CudaGemm::new` への一時結線・診断入口の実装差分）はどのコミットにも保存され
ておらず、記載のコマンド（現行コード）から同じ出力を再現できない。

この状態は AGENTS.md「再利用・アセット化」および実測記録の再現性の観点（codex-review 指摘。
`docs/perf/cuda-gemm-tiled-f32-swizzle-ab.md` 旧版）で監査不能と判断された。一時実装を今から再構成して
コミットすることは、実測当時の実装と一致する保証がなく実測値の裏付けにならないため行わない。したがって
**ゲート 2 の `dispatch_over_base` 数値・「§3 のゲート 2 判定基準を満たす値だった」という説明は、再現
不能な参考値として本ドキュメントの正式な証跡から除外する**（生ログファイル自体は当時の実行結果の
一次記録として `docs/perf/logs/cuda-tiled-f32-swizzle-1139/` にそのまま残すが、本文からは正式な判断根拠
として引用しない）。§5 の採否判断はゲート 1 の結果のみに基づいており、この除外によって変わらない。

## 5. 判断

**不採用**。ブロック実行順スウィズル（#1034）の `tiled_f32`（classic）経路への本番結線は行わない。

- `internal-diagnostics` feature 限定の診断用 opt-in 入口（`gemm.rs::CudaGemm::new_with_tiled_f32_swizzle`・
  `kernels.rs::tiled_f32_source_with_swizzle`・`examples/gemm_tiled_f32_swizzle_bench.rs`・
  `diagnostics::tiled_f32_swizzle_group_width`）はそのまま温存する（コード変更なし。#1034 時点の状態を
  維持）。本番既定コンストラクタ `CudaGemm::new` への結線は行わない。
- N=512 での再現性ある劣化を許容する（判定基準を緩和する）判断はユーザー承認が要る事項であり、本イシュー
  の範囲では行わない。将来 N=512 側の劣化原因（remap 算術の追加コスト・タイル数が少なく L2 局所性向上の
  効果が算術オーバーヘッドを相殺しない可能性）を軽減する改善（例: サイズ条件付き適用〈N が大きい場合のみ
  swizzle を適用〉）を試すのであれば、それは新たな A/B（新しい判定基準の事前宣言を含む）として別イシュー
  で扱う（下記「6. 申し送り」参照）。

## 6. 前回 PR（#1165）で実施済みの是正（本 PR での変更ではない。ゲート 0 の一部の静的強化）

ゲート 0 のうち「REQ-8 境界ガード実在検査」は実機を要さないため、前回セッション（#1165。実機到達不能で
ブロック判断となった回）で通常 CI（GitHub ホステッド）で常時実行できる形として既に新設済みであり、
本 PR（#1139 再着手分）ではこれらのファイルへ変更を加えていない:

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

## 7. 申し送り（本イシューを再度引き継ぐ場合。ユーザー承認が要る事項を含む）

- 本イシュー（#1139）は §4〜§5 の実測・判定により**完了**とする。再着手する場合は新たな変更（判定基準の
  見直し・カーネル側の改善）を伴うため、新しいイシューとして起票し、新しい判定基準を実測前に宣言する
  ことをユーザーへ提案する（`.claude/rules/out-of-scope-tracking.md`）。
- N=512 の劣化を軽減する方向性の候補（実装済みではなく単なるアイデアの記録）:
  - サイズ条件付き適用（`swizzle.rs::should_apply_swizzle` 相当を classic 経路にも導入し、N が一定以上
    の場合のみ swizzle remap を適用する）。ただし §2 の到達性整理どおり、classic 到達形状自体が非整列
    正方形（大半は M=N≥4096 級）に限られるため、実際に本番へ与える影響は限定的な可能性がある。
  - remap 算術（`swizzle.rs::swizzled_block_idx` と同一の整数演算）自体のコスト削減（除算・剰余の
    削減等）。
  - いずれも新たな実装変更を伴うためユーザー承認が要る（本イシューでは着手しない）。
- pipeline カーネル（`kernels_tiled_pipeline.rs`）への swizzle 横展開は引き続きスコープ外（PR 本文の
  「切り出し提案」参照）。

## 8. 関連ドキュメント

- `docs/perf/cuda-gemm-swizzle-ab.md`（f16 `mma.sync`・TF32 opt-staged 経路の同型記録。§8 に本ファイルへの
  ポインタを追加）
- `docs/perf/cuda-gemm-tiled-pipeline.md`（#1137。cp.async パイプラインの本番結線記録。「スコープ外事項」
  節で #1034 結線判断を本イシューへ引き継ぎ）
- `docs/perf/cuda-gemm-simt-register-blocking.md`（#1136。classic 経路の結線前 baseline）
- `docs/perf/logs/cuda-tiled-f32-swizzle-1139/`（本イシューの実行ログ・env_info。ゲート 0 実機テスト
  ログ・ゲート 1 A/B ベンチ 5 回分の生ログ）
