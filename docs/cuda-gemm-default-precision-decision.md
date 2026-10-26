# CUDA GEMM の既定精度（FP32 維持）決定記録

イシュー #3262（Phase 11）。依存: #3185（cuBLAS の純カーネル時間）・#3191（27 セル表）はいずれも CLOSED。
本記録は**判断材料の整理と所有者回答の記録のみ**を扱い、コード（`.rs`）・`Cargo.toml`・`docs/spec`・
tolerance 定数・parity baseline・既定精度は変更しない。

## §0 結論

| 項目 | 確定内容 |
|---|---|
| 既定精度 | **FP32 strict のまま**（`CudaGemmPrecision::Fp32Strict`）。所有者回答（2026-10-12）で案 (A) 確定 |
| TF32 | 既存の明示 opt-in のまま（`set_cuda_gemm_precision`／`set_cuda_tf32_gemm_enabled`。`docs/cuda-tf32-optin-api-decision.md`） |
| 3×TF32 | 非推奨（確定済み。`docs/cuda-tf32x3-split-single-decision.md`。§4.3） |
| burn との比較 | 精度クラスを併記して読む（§6） |
| 案 (B)・(C) | 採らない（§3・§7） |
| 変更物 | なし（本 doc と索引・参照の追記のみ） |

## §1 位置づけ・出典

- 対象セルは G-CUDA-G2048／G4096（GB10・gemm CUDA）。最速相手は burn（cubecl）で、burn は f32 入力を TF32 へ降格し公開の無効化スイッチを持たない。fandhe-ai の既定は FP32 strict のため精度クラスが異なる。
- 出典: `docs/analysis/burn-cubecl-cuda-matmul.md` §5、`docs/analysis/candle-0.11.0-cuda-path.md`、`docs/perf/opp-kernel-gb10.md`、`docs/perf/loss-attribution-matrix.md` §10.3.3・§10.5.3・§10.7、`docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §9、`docs/cuda-tf32x3-split-single-decision.md`、`docs/cuda-tensor-core-parity-judgment-decision.md`、`docs/candle-parity-precision-class-decision.md`。
- 本記録は制約の緩和を提案するものではない。

## §2 所有者回答の転記

- 出典: イシュー #3262 のコメント <https://github.com/Fandhe-AI/fandhe-ai/issues/3262#issuecomment-6111778802>（投稿者 `aLiz-Nancy`・見出し「所有者回答（2026-10-12）」・createdAt 2026-10-11T17:36:16Z）。「全て推奨で」への回答として (A) で確定。

| 項目 | 回答 |
|---|---|
| (A) FP32 のまま・burn は精度クラス併記 | 確定 |
| (B) TF32 を既定にする | 行わない |
| tolerance・baseline・既定精度 | 変えない |

- **承認の範囲は (A) のみ**。既定精度の変更と baseline の追加は承認されていない。

## §3 選択肢の比較

| 案 | 内容 | 前提・必要な手続き | 判断 |
|---|---|---|---|
| (A) | 既定は FP32 のまま。burn との比較は精度クラスを併記して読む | なし（現状維持） | **採用（推奨）** |
| (B) | TF32 を既定にする | TF32 は strict 判定で非ゼロの不合格要素（約 13〜17%。1×1×1 は不合格ゼロ）が生じ、厳密ゼロ fail 判定を満たさない形状があるため、形状二分＋`ParityBaseline` 方式が前提。baseline の追加は実機実測値・人間承認に限られる。REQ-2 の扱いが変わるため、先に Fandhe-AI/fandhe-ai-spec への提案が必要 | 採らない |
| (C) | TF32 opt-in カーネルの最適化で差を詰める | 既定精度は変えない。別イシュー | 本件では扱わない |

(A) を推奨する理由: REQ-2 の統一複合判定（相対 1e-3 未満または絶対 1e-5 未満）を既定のまま満たす唯一の構成であり、(B) は spec 側の手続きなしには成立しない。TF32 を使いたい利用者は既存の opt-in で選べる。

## §4 判断材料の数字（転記のみ。再計算なし）

### 4.1 cuBLAS の純カーネル時間（実測）

出典: `docs/perf/opp-kernel-gb10.md`「cuBLAS（PyTorch）の純カーネル時間と fandhe」表、生ログ `docs/perf/logs/opp-kernel-gb10-3185/gb10/aggregate.md`（#3185・GB10・torch 2.14.0+cu130・2026-10-21 UTC・`.rev-stamp` `c22b8c26e4ebd171cebd5c65346dc03435430e4c`・5 run 中央値）。

| N | fp32 mm_pure [ms] | fp32 mm_event [ms] | tf32 mm_pure [ms] | tf32 mm_event [ms] | fp32 TFLOPS（対 31） | tf32 TFLOPS（対 53.3） | tf32/fp32（wall） |
|---|---|---|---|---|---|---|---|
| 256 | 0.0180 | 0.0170 | 0.0118 | 0.0103 | 1.86（0.06） | 2.84（0.05） | 0.656 |
| 512 | 0.0348 | 0.0337 | 0.0180 | 0.0164 | 7.71（0.25） | 14.89（0.28） | 0.518 |
| 1024 | 0.1372 | 0.1367 | 0.0676 | 0.0659 | 15.65（0.50） | 31.77（0.60） | 0.493 |
| 2048 | 0.9739 | 0.9846 | 0.4215 | 0.4253 | 17.64（0.57） | 40.76（0.76） | 0.433 |
| 4096 | 7.9627 | 7.9606 | 3.2933 | 3.3167 | 17.26（0.56） | 41.73（0.78） | 0.414 |

- fandhe FP32 の wall 比（対 cuBLAS fp32 mm_pure）は N>=1024 で 1.20〜1.36（実測。N=2048 は 1.201、N=4096 は 1.257）。
- 同 doc の引き継ぎ要旨: cuBLAS の FP32 は公称 31 TFLOPS の 0.56〜0.57 で頭打ちになる。FP32 SIMT のままでは、TF32 の時間（N>=1024 で FP32 の 0.41〜0.49 倍）には届かない。

### 4.2 比較の窓の値（実測。窓は純カーネルではない）

出典: `docs/perf/loss-attribution-matrix.md` §10.7（0.11.0・GB10・fresh・`GB-B`）。

| N | fandhe-ai FP32／TF32 | burn（TF32 固定） | candle FP32／TF32 |
|---|---|---|---|
| 2048 | 4.1854 ms／4.3700 ms | 4.1443 ms | 4.2211 ms／3.6952 ms |
| 4096 | 39.6519 ms／33.9267 ms | 39.6344 ms | 57.6643 ms／57.2268 ms |

- 比（burn÷fandhe-ai）は N=4096 で FP32 1.00・TF32 1.17（推定）。
- TF32 の利得が窓に見えるのは N=4096 だけで、N=2048 では TF32 opt-in のほうが遅い（4.3700 対 4.1854 ms）。

### 4.3 3×TF32（実測）

f32 SIMT に対する TFLOPS 比は 512／1024／2048／4096／256×256×4096 の順に 0.897／0.930／0.886／0.721／0.891 で、5 形状すべて遅い（出典: `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §9）。非推奨・baseline 不承認で確定済み（`docs/cuda-tf32x3-split-single-decision.md`）。

### 4.4 burn の N=2048 の到達率（推定）

2·2048³ ÷ 4.1443 ms ÷ 53.3 TFLOPS から約 7.8% になる。これは**窓ベースの推定**で、burn の純カーネル時間は未測定である。53.3 TFLOPS は第三者 1 名の投稿値。この値は合格ラインに使わない。

## §5 届かない場合の理由（3 区分）

`docs/perf/loss-attribution-matrix.md` §10.5.3 の G-CUDA-G2048／G4096 の行の要旨を、区分ごとに引く（制約の緩和は提案しない）。

| セル | ハードの上限 | 計測窓の定義 | 自分たちの制約 |
|---|---|---|---|
| G-CUDA-G2048 | cuBLAS FP32 は 0.9739 ms＝17.64 TFLOPS で公称の 0.57。cuBLAS TF32 は 0.4215 ms＝40.76 TFLOPS（公称 53.3 の 0.76）。最速相手 burn の純カーネルは未測定 | 最速相手 burn は fresh 型の窓。FP32 fresh 4.1854 ms 対 burn 4.1443 ms | 既定精度は FP32（REQ-2 の複合判定）。最速相手 burn は TF32 固定 |
| G-CUDA-G4096 | cuBLAS FP32 7.9627 ms＝17.26 TFLOPS で公称の 0.56。cuBLAS TF32 は 3.2933 ms＝41.73 TFLOPS（0.78）。burn の純カーネルは未測定 | fresh の readback 事前タッチ約 15 ms が残る（`GB-186` §14.3 判定 F） | FP32 の fandhe-ai が burn に 1.01× で並ぶ。TF32 opt-in では 1.17× |

## §6 精度クラスを併記する方針

新しい仕組みは作らない。既存の framework-compare の仕組みを読み方の正とする。

- burn の cuda 行の `tf32: true` ラベル
- `bench-common::parity::PrecisionClass`・`TF32_UNIT_ROUNDOFF`
- `summarize.py` の (a-tf32) 節
- `docs/candle-parity-precision-class-decision.md`・Fandhe-AI/fandhe-ai-spec#70

burn のセルは「TF32 相手に FP32 strict で並べた比」と読む。TF32 opt-in 同士の比は参考（判定外）として併記する。

## §7 (B) を採らない理由

TF32 は strict 判定で非ゼロの不合格要素（約 13〜17%。1×1×1 は不合格ゼロ。`docs/cuda-tensor-core-parity-judgment-decision.md` §1.1）が生じ、厳密ゼロ fail 判定を満たさない形状がある。採るなら形状二分＋`ParityBaseline` 方式（`docs/cuda-tensor-core-parity-judgment-decision.md`・spec REQ-2 2026-09-02 追記）が前提で、baseline の追加は実機実測値と人間承認に限られる。既定にするなら先に spec への提案が必要で、所有者回答（§2）も行わないとしている。

## §8 「勝てるセルは是正で届く見込み」の扱い

これは推定であり、断定しない。関連する是正の状態は `docs/perf/loss-attribution-matrix.md` §10.8.2・§10.8.3 のとおり（CUDA readback `PrefetchedPretouched`〈#3225〉は既定を反転済み・GB10 実機で確認済み）。入力常駐（#3250）の実装記録は `docs/compat-resident-train-step-decision.md`。本記録は新規の計測値を足さない。

## §9 使わない値

イシュー本文にある「窓天井 1.7〜2.3 倍」「実質天井 約 1.15／2.0 倍」はリポ内に出典がなく（`docs/perf/loss-attribution-matrix.md` §10.10 で再現不能と記録済み）、判断材料に使わない。

## §10 不変条件

tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・既定精度・`.rs`・`docs/spec`・依存を変えない。

## §11 OWASP Top 10 観点

コード・依存・CI・設定の変更はなく、A03・A06・A08 の新たな露出はない。tolerance・baseline・既定精度に触れないため判定の迂回は生じない。承認は §2 の範囲に限り、事実以上に書かない。実機の接続先・内部ホスト名・秘密情報は含めない。
