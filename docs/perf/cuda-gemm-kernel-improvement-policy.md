# CUDA GEMM カーネル性能改善方針の検討と REQ-8 下限の据え置き判断（#938）

イシュー #938「CUDA GEMM カーネル性能改善と REQ-8 下限の見直し」の記録。ルート #920「ベンチ
実測起点の性能改善トラッキング」→ Phase 4 親 #924「カーネル性能」配下。依存 #928（PR #947、
コミット `cb153a3`、`docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md`）で完了済みの CUDA GEMM
カーネル単体性能ベースライン（vs candle / Burn）を踏まえ、(a) 差がある領域の改善方針を検討し、
(b) REQ-8 段階的下限（`docs/performance-targets.md` §2）の更新要否を判断する。

**本ドキュメントは判断案であり、他の追補（`docs/perf/performance-floor-decision.md` §8〜§11）と
同様、最終成立は本イシュー #938 の PR レビュー・マージ（人間承認）による**（先例: #158・#159・
#577 と同じ「PR マージ = 人間承認で成立」方式）。

## 0. 実行環境の制約（本ドキュメント作成セッション）

本ドキュメント作成時点で `docs/real-hardware-verification-env.local.md`（実機ホスト名のローカル
管理ファイル）が実装セッションに存在せず、DGX Spark GB10 実機へは SSH 到達不能だった。実測値の
捏造は行わない方針（PR #713 で確立、#928 でも踏襲）に従い、**本ドキュメントでも新規実機実測は
一切行わない**。以下の検討・判断は #928 が確定させた既存コミット済み一次データの転記・機械計算
（倍率換算）のみに基づく。

## 1. #928 が確定させた事実（本イシューの入力。転記のみ）

`docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md`（以下「#928 doc」）より要約する。

- 公開 API（facade → tape → backend-cuda）が実行する既定カーネルは `CudaGemm::run_tiled_f32`
  固定（`crates/backend-cuda/src/ops.rs`。`CudaBackendOps::gemm` が `context_cache::cached_gemm`
  経由で呼ぶ唯一の経路。本セッションでも実装コードを確認し、この構造に変更がないことを確認済み）
- その candle/Burn 対比（#928 doc §4 表）は **REQ-8 判定対象形状 N=2048 で約 0.56 倍・N=4096 で
  約 0.67〜0.87 倍と未満**、N=512/1024（判定対象外の参考形状）では約 1.05〜2.45 倍と上回る
  **サイズ依存**の傾向（#928 doc §4 表・§4 限定条件 4）
- 未接続の Tensor Core 候補経路 `wmma_tf32` はカーネル単体で candle/Burn の**約 3.09〜4.00 倍
  （N=2048/4096）**であり（#928 doc §4 表）、REQ-8 f32 下限 50% の根拠実測（4096 で 51.96%、
  `docs/perf/performance-floor-decision.md` §10「CUDA 限定条件の継続」節・限定条件 4 のとおり
  `wmma_tf32_staged` 経路の値。限定条件付きでユーザー承認済みの候補値）
- `CudaGemmAuto::run_f32`（`crates/backend-cuda/src/gemm_auto.rs`。本セッションで実装コードを
  確認済み）は `KernelKind::MatrixUnit`・`KernelKind::Tiled` のいずれの分岐でも
  `run_tiled_f32` へ委譲する実装であり、f32 の Tensor Core 分岐（`MatrixUnit` の実処理）は
  #62 未実装のままである。すなわち**「`CudaGemmAuto` を公開 API へ接続するだけでは `wmma_tf32`
  に到達できない」**ことが確定している（#928 doc §4 限定条件 4）
- 原因は 2 つに分離される（#928 doc §5「両論併記の確定事項」節）:
  (i) tape 経由の固定オーバーヘッド（環境 2 fresh 458〜594 ms・環境 3 reuse per-call でも
  260〜474 ms。Phase 2〈#929/#931〉・Phase 3〈#933 系〉のスコープ）
  (ii) 既定カーネル変種が Tensor Core 経路へ未接続で tiled f32 に固定されていること
- #928 doc の限定条件（計測境界差・バージョン差・TF32 使用有無未確認・クロスセッション比較で
  GB10 個体の同一性未確認）は本ドキュメントでも解消されておらず、そのまま引き継ぐ（#928 doc
  §4 限定条件 1〜3・7）

## 2. 改善方針の検討

`docs/perf/cuda-gemm-bottleneck-diagnosis.md`（#486。N=4096 のデータ再利用崩壊診断）・
`docs/perf/cuda-optimized-remeasurement.md`（#571・Phase F。tiled f32・wmma_tf32 の TFLOPS 実測）・
GEMM 性能改善ツリー（ルート #479 Phase A〜F 完了）の内容と #928 doc の確定事実を踏まえ、3 案を
比較する。

### 案 A（推奨）: 既定カーネルの Tensor Core 経路接続

`CudaGemmAuto::run_f32` に f32 `MatrixUnit` 分岐の実処理を実装し（現状は tiled への
fail-safe フォールバックのみ。`gemm_auto.rs` の `run_f32` ドキュメンテーションコメント参照）、
`select_gemm_kernel` の決定表見直し・数値一致 parity 検証とセットで公開 API を `wmma_tf32`
経路へ切り替える案。

- **期待効果**: #928 doc §4 の実測根拠により、カーネル単体で candle/Burn 比 **約 3〜4 倍**
  （N=2048: 約 3.41 倍、N=4096: 約 3.09〜4.00 倍）に達する見込みで、3 案中最大のレバー
- **課題（本イシューでは実装しない理由）**:
  1. 候補算出経路（`wmma_tf32`・`mma_f16`）は #389 §5.3 が記録した数値一致 parity の恒常 fail
     対象と一致する（`performance-floor-decision.md` §10「CUDA 限定条件の継続」節・限定条件 1）。
     TF32/f16 Tensor Core 経路の複合判定改定（REQ-2 改定）は #186 close 後も閾値定数
     （`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）自体は変更されておらず、spec リポジトリ
     側対応待ちのまま
  2. 数値一致許容誤差の変更・既定経路（公開 API が実行するカーネル）の変更はいずれもユーザー
     承認事項（`.claude/rules/security.md`「自己修復ループ固有のガードレール」節・
     `.claude/rules/coding-rust.md`「テスト・ベンチ」節）
  3. 実装後の検証（parity・性能双方の非後退確認）に GB10 実機が必要だが、本セッションは実機
     到達不能（§0）
- **結論**: 本イシューでは実装せず、ユーザー承認を経た別イシューでの追跡を**提案**として記録する
  （§4）

### 案 B: tiled f32 の大サイズ追加最適化（非推奨）

`run_tiled_f32` 自体をさらにチューニングし、N=2048/4096 での candle/Burn 対比を改善する案。

- GEMM 性能改善ツリー（ルート #479）は Phase A〜F を通じて tiled f32 を含む CPU/CUDA/Metal の
  各カーネルを既に体系的に最適化済み（`docs/perf/cuda-gemm-mma-*.md`・
  `docs/perf/cuda-gemm-boundary-predicate-ab.md` 等の一連の最適化記録）
- N=4096 でのスループット頭打ちは `docs/perf/cuda-gemm-bottleneck-diagnosis.md`（#486）が
  データ再利用構造（L2/DRAM 帯域律速）の限界として定量診断済みであり、tiled（Tensor Core 未使用）
  経路のままでの追加投資対効果は小さいと評価する
- **結論**: 非推奨。Tensor Core 経路（案 A）への切替のほうが同じ投資でレバーが大きい

### 案 C: per-call 固定オーバーヘッド削減（対象外）

tape 構築・per-call の固定オーバーヘッド（#928 doc §3.1 実測。環境 3 で 260〜474 ms／call）を
削減する案。

- #928 doc §5「原因は 2 つに分離される」の原因 (i) に対応する取り組みであり、**カーネル性能では
  なく呼び出しオーバーヘッド側の改善**である
- ルート #920 の Phase 2（#929/#931。デバイスハンドル再利用。PR #946・#949 完了済み・進行中）・
  Phase 3（#933 系）のスコープであり、本イシュー（Phase 4「カーネル性能」）の対象外
- **結論**: 本イシューでは対象外と整理する（重複起票はしない。§4 参照）

### 推奨のまとめ

3 案のうち**案 A（既定カーネルの Tensor Core 経路接続）を推奨**する。ただし数値一致許容誤差・
既定カーネル変更に関わるユーザー承認事項を含むため、本イシューでは実装せず、承認を前提とした
別イシューでの追跡を提案する。案 B は既に最適化済みの領域への追加投資であり非推奨、案 C は
本イシューのスコープ（カーネル性能）外と整理する。

## 3. REQ-8 下限の更新要否判断（判断案）: 更新不要（全行据え置き）

`docs/performance-targets.md` §2 の CUDA f32 50%・CUDA f16 35%（最適化後下限）を含む全行を
**変更しない**。根拠は以下の通り。

1. **REQ-8 下限は対 PyTorch 比のカーネル単体実測を根拠とする**（`docs/performance-targets.md`
   §4「判定対象形状」・§5「v1 確定値の位置づけ」節）。candle/Burn 相対値は同 §5 のとおり
   「参考指標」であり、v2 下限の根拠には用いない方針が既に確立している。#928・本イシュー
   （#938）のいずれも対 PyTorch 比の新規実測は発生していない（#928 doc §0「状態」節のとおり
   GB10 到達不能のため新規実測は未実施）
2. **丸め規則（`docs/performance-targets.md` §3・`bench_harness::floor_lower_bound`）を再適用
   できる新しい対 PyTorch 比データが存在しない**。GB10 実機へ到達できない本セッションでは
   新規実測を行えず（§0）、実測なしでの下限変更は行わないという先例（`performance-floor-decision.md`
   §3「確定判断（据え置き確定）」節。実機実測なしでは既存値を据え置く判断）に従う
3. **50%（f32）／35%（f16）に付随する限定条件 1〜3 は継続している**
   （`performance-floor-decision.md` §10「CUDA 限定条件の継続」節）。候補算出経路
   （`wmma_tf32`・`mma_f16`）の数値一致 parity 恒常 fail 対象との一致、REQ-2 改定待ちという
   状態は本イシューで解消も悪化もしていない。限定条件 4（`wmma_tf32_staged` 経路のベースライン
   未確立）はすでに #726（2026-08-19）で解消済みであり、これも本イシューでは変化なし
4. **§2 の改善方針検討（案 A）は候補経路の性能を変えるものではなく、公開 API の既定カーネルを
   切り替える提案にとどまる**。実装・承認・実機検証を経て初めて「既定カーネルの実測」が変わり、
   その時点で再度 §4「未確定領域と再確定手順」の手順（実測 → 丸め規則適用 → 追補作成 → ユーザー
   承認）に従い再確認すべき事項であり、本イシューの時点では下限側に反映すべき新事実はない

以上より、**`docs/performance-targets.md` は本文・値とも変更しない**（変更しないこと自体を
判断結果として記録する。受け入れ条件 (b)「更新要否の判断」はこの判断記録により充足する）。

## 4. スコープ外・未実施事項

- **案 A（既定カーネルの Tensor Core 経路接続）の実装イシュー化**: `CudaGemmAuto::run_f32` への
  f32 `MatrixUnit` 分岐実装・`select_gemm_kernel` 決定表見直し・数値一致 parity 検証・実機
  非後退確認を伴う変更であり、数値一致許容誤差・既定カーネル変更のユーザー承認事項を含む。
  本イシューでは起票せず、**ユーザー承認を得たうえでの別イシュー化を提案**する
  （`.claude/rules/out-of-scope-tracking.md` の規約に従い、承認を得てから起票する）
- **GB10 での N=256 カーネル単体実測**: #928 doc §7 が記録済みのスコープ外事項（`cuda_floor_bench`
  への 256 追加というコード変更・実機実測が必要）の再掲。#928 で既に記録済みのため本イシューでの
  重複起票はしない
- **GB10 個体同一性の確認**（#928 doc §4 限定条件 7・§7）・**GB10 での reuse per-call 新規実測**
  （#928 doc §6.2）・**reuse per-call 固定オーバーヘッドの原因切り分け**（#928 doc §7、案 C に
  対応する Phase 2/3 側の課題）: いずれも #928 で既に記録済みのスコープ外事項であり、本イシューで
  重複起票はしない

## 5. 関連ドキュメント

- `docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md`（#928・PR #947）: 本ドキュメントの入力
- `docs/perf/cuda-optimized-remeasurement.md`（#571・PR #725）: tiled f32／wmma_tf32 の TFLOPS
  実測・数値一致 parity 状態の限定条件
- `docs/perf/cuda-gemm-bottleneck-diagnosis.md`（#486）: N=4096 データ再利用崩壊の定量診断
  （案 B 非推奨の根拠）
- `docs/perf/performance-floor-decision.md` §9・§10: CUDA f32/f16 最適化後下限（50%／35%）の
  確定判断・限定条件
- `docs/performance-targets.md` §2〜§5: REQ-8 段階的下限表・丸め規則・計測プロトコル・v1 確定値
  の位置づけ（本イシューでは変更しない）
