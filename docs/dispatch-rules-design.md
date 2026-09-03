# ディスパッチ規則設計（TASK-11.2a）

> 役割・参照元: 本文書は REQ-11（`docs/spec/04-requirements.md`）の受け入れ基準
> 「経路選択（ディスパッチ規則）」に対応する**設計**文書であり、TASK-11.2（#66）
> の第 1 段（TASK-11.2a・本イシュー #67）の成果物である。規則の**実装・抽象層
> 統合**は TASK-11.2b（#68）、**境界形状の実測再検証**は TASK-11.2c（#69）、
> **証跡整備**は TASK-11.3（#70）が担当し、本文書はそれらから参照される前提で
> 書く。コード変更は含まない（`docs/dispatch-rules-design.md` の新規追加のみ）。
>
> **改訂注記（#1150・2026-09-04）**: §5.6 を新設し、CUDA f16 `MatrixUnit`
> 経路内部の実装優先順位（`CudaMmaGemm` 優先・`CudaWmmaGemm` フォールバック。
> #1131 系列）を追記した。上記「コード変更は含まない」は当初 #67 時点（本文
> 初版）の記述であり、本改訂自体も設計記録のみでコード変更を伴わない
> （§5.6 冒頭に明記）。

## 1. 判断サマリ

- 行列演算ユニット（NVIDIA Tensor Core／Apple `simdgroup_matrix`）使用経路の
  選択は、**決定的な規則ベース**で行う。実行時ベンチマークによる自動選択
  （v1 の CubeCL autotune 相当の探索）は行わない。
  - 根拠 1: REQ-11 は v2 で「CubeCL の autotune ではなく自作ディスパッチ規則
    に置き換える」と書き直し済みである（`docs/spec/04-requirements.md:23`・
    `:228`・`:238`）。
  - 根拠 2: REQ-13 は、CubeCL/Burn の autotune 探索コストがコールド起動時で
    PyTorch 比約 21〜24 倍遅くなる主要因であると実測している
    （`docs/spec/04-requirements.md:258`）。探索ベースの経路選択は本ライブラリ
    が回避すべき起動コストのボトルネックそのものであり、規則ベース設計は
    この教訓を踏まえる。
- 利用者向けの明示切替 API（feature flag・環境変数・API 引数）は**提供しない**。
  - 根拠: REQ-11 受け入れ基準「行列演算ユニットの活用は、ライブラリの明示的な
    設定項目として利用者に提供しないこと」（`docs/spec/04-requirements.md:237`）。
    v1 方針の維持でもある。
  - 先行事例: `crates/backend-cpu/src/gemm_blis/microkernel.rs` は ISA
    dispatch（AVX-512／AVX2+FMA／scalar・NEON）について「環境変数等による
    dispatch 上書き機構は設けない（外部入力が `unsafe` の駆動経路に影響しない
    ため）」と明記している（同ファイル 38 行目付近）。本設計はこの CPU 側方針
    を GPU 経路選択へ拡張する位置づけで踏襲する。

## 2. HW 判定（ケイパビリティゲート）

| バックエンド | 判定材料 | 判定条件（設計値） | 根拠・出典 | 確定度 |
|---|---|---|---|---|
| CUDA | `CudaDevice::compute_capability() -> (i32, i32)`（実装済み API、`crates/backend-cuda/src/device.rs:153`） | WMMA f16 経路: `cc >= (7, 0)` ／ TF32 経路: `cc >= (8, 0)`。GB10 = `cc (12, 1)`（sm_121、Blackwell）は両対応 | WMMA は Volta（cc 7.0）以降、TF32 対応 Tensor Core は Ampere（cc 8.0）以降という一般的な NVIDIA アーキテクチャ世代対応（PoC-v2-3 が GB10 = Blackwell 世代を tensor core 搭載と記述、`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md:95`）。cc 世代境界の数値自体は本イシューでの実機再検証対象外 | 暫定（世代境界は一般値、#69 で実機 cc 分岐の要否を再確認） |
| CUDA（第 2 層。§5.6） | 同上 | `MatrixUnit` 経路内の実装優先順位として、mma.sync f16 経路（`gemm_mma.rs` の `mma.sync`/`ldmatrix`/`cp.async` 3 段パイプライン）は `cc >= (8, 0)`（LDGSTS〈cp.async/ldmatrix〉が sm_80 以降限定のため。`crates/backend-cuda/src/gemm_mma.rs:33` の独立定数）を要求し、非対応時は WMMA f16（cc>=7.0）へフォールバックする | `gemm_mma.rs:33`（`MIN_COMPUTE_CAPABILITY_MAJOR: i32 = 8`）・`gemm_mma.rs:217-229`（`check_min_compute_capability`、major のみ比較） | 確定（実装済みの独立ゲート定数。第 1 層 `CUDA_TF32_MIN_CC = (8, 0)` と値は同じだが TF32 Tensor Core 世代とは別根拠） |
| CUDA | `CudaDevice::is_available()`（`crates/backend-cuda/src/device.rs:74`） | ドライバ不在・`cudarc` 動的ロード失敗時は CUDA 経路自体が候補外 | `cudarc` は無条件依存＋動的ロード方式（CUDA toolkit 非搭載環境でもビルド成立。`deps-policy.md`）。fail-safe 設計は 2.3 節参照 | 確定（既存実装のとおり） |
| Metal | `MTLDevice.supportsFamily:MTLGPUFamilyApple7`（Apple GPU ファミリ判定）＋ `cfg(target_os = "macos")` | Apple7 以上（M1 世代以降）で `simdgroup_matrix` 対応。非対応ファミリは tiled → naive MSL 経路へフォールバック | PoC-v2-4 は M4 Max（Apple Silicon）実機で `simdgroup_matrix` を用いる MSL ソースがコンパイル・実行できることを確認済み（`docs/spec/03-poc/poc-v2-4-metal-gemm/README.md:19`・`:32`）。`supportsFamily:` による世代判定自体は本イシューでは未実装確認であり、実装時（#68）に `objc2-metal` API 突合が必要 | 暫定（API 呼び出し自体は #68 で `objc2-metal` 側の突合が必要） |
| CPU | `Isa::detect()`（実装済み・`crates/backend-cpu/src/gemm_blis/microkernel.rs:252`） | 既存規則（AVX-512 → AVX2+FMA → scalar、NEON 無条件）をそのまま踏襲 | 本文書では参照のみ。変更対象外 | 確定（既存実装のとおり） |

### 2.1 判定タイミング

HW 判定は**デバイス初期化時に 1 回**行い、ケイパビリティ構造体（`DeviceCaps`
相当、4 節参照）へキャッシュする。GEMM 呼び出しごとに FFI（`cudarc`
・`objc2-metal`）照会を繰り返さない。これは `Isa::detect()` が `OnceLock` で
検出結果をキャッシュしている既存方針（`microkernel.rs:249` 付近のコメント
「dispatch 判定を 1 箇所に固定する意図」）と整合させる。

### 2.2 fail-safe 方針

判定不能・取得失敗時は**行列演算ユニット不使用経路（tiled／naive）へ倒す**。
誤って非対応 HW で `wmma`／`simdgroup_matrix` 命令を発行しないことを優先する
（性能より正当性を優先する fail-safe）。`CudaDevice::new()` のコメントが
「`panic!`／`unwrap()` せず `is_available() == false`」を返す設計方針を既に
明記しており（`crates/backend-cuda/src/device.rs:186` 付近）、本設計はこの
既存方針と整合する。

## 3. 形状判定（暫定閾値）

判定軸は `min(M, N, K)` を基本とする。v1 PoC-8（CubeCL autotune 前提の実測、
`docs/spec/03-poc/poc-8-matrix-unit/README.md`）の実測値を**参考値**として
初期閾値を設計する。

### 3.1 参考値（v1 PoC-8、CubeCL autotune 前提）

| 実測項目 | 値 | 出典 |
|---|---|---|
| 形状境界（Metal・CUDA 共通） | M=N=K=512（256 未満は unit/accelerated 両方が候補、512 以上は accelerated のみ） | `poc-8-matrix-unit/README.md:50` |
| Metal（M4 Max）、M=N=K=256・f32 | accelerated が unit の約 20.5 倍高速 | 同上 `:75` |
| CUDA（GB10、Blackwell）、M=N=K=256・f32 | accelerated が unit の約 1.4〜1.6 倍高速（Metal より差が小さいが、小形状でも accelerated が一貫して優位） | 同上 `:126`・`:140` |
| CUDA（GB10）、M=N=K=2048/4096・f32 | TMA 系候補（`matmul_specialized_tma_mma` 等）が最速候補として選択される（TMA 選好） | 同上 `:125` |

### 3.2 v2 暫定閾値・非対称設計の提案

- **境界の初期値**: `min(M, N, K) < 512` を「小形状」、`>= 512` を「大形状」
  とする閾値を暫定的に踏襲する（v1 PoC-8 実測値、上記 3.1）。この閾値は
  **Metal 側の判定にのみ使う**。CUDA 側は下記のとおり形状閾値を設けない
  非対称設計とする。
- **CUDA**: HW 対応（`cc` ゲートを満たす）なら**形状によらず常に Tensor
  Core 経路**を選択する（形状下限を設けない）。
  - 根拠: GB10 実測では最小形状（M=N=K=256）でも accelerated 経路が unit
    経路に対し一貫して優位（約 1.4〜1.6 倍、3.1 節 `:126`・`:140`）であり、
    小形状で accelerated を避ける理由（Metal のような極端な逆転ペナルティ）
    が実測上見当たらない。したがって「HW 対応なら形状を問わず Tensor Core」
    が実測と整合する規則である。
- **Metal**: 閾値以上（`min(M,N,K) >= 512`、大形状）でのみ `simdgroup_matrix`
  経路を選択し、閾値未満（小形状）は tiled 経路とする。
  - 根拠: Metal 実測では小形状（M=N=K=256）で accelerated 経路が unit 経路の
    約 20.5 倍高速だが、これは「unit 経路が著しく遅い」ことを示すのみで
    「accelerated が常に有利」とは即断できない。v1 実測は 256/512 の 2 点
    のみで境界形状（例: 384・640）の挙動は未計測だったが、#382（Apple M4
    Max 実機・5 ラン反復）で 256〜1024 の 8 境界形状を実測済み。実測クロス
    オーバーは 384（5 ラン全てで `simdgroup_auto` が `tiled` を上回り符号
    一貫）であり、512 は保守的すぎる可能性がある。#382 では 384 への引き
    下げを**変更提案として記録**したが、コード（`METAL_SIMDGROUP_MIN_DIM`
    自体）は変更していない（実施は別レビュー・別 PR・ユーザー承認。判定の
    詳細・要調査事項は `docs/perf/dispatch-boundary-measurement.md`
    「`METAL_SIMDGROUP_MIN_DIM` の妥当性判定（#382）」節）。現行 512 のまま
    閾値未満は引き続き保守的に非 accelerated 経路（tiled、5.2 節）へ倒す。
  - **これが CUDA との非対称設計の実体**: CUDA は「HW 対応なら形状閾値なし
    で accelerated」、Metal は「閾値以上でのみ accelerated」。根拠は両者で
    実測傾向（小形状での accelerated 優位の一貫性）が異なる点にある。
- **すべての閾値は暫定値**（v1 CubeCL 前提の参考値）であり、**#69（TASK-11.2c）
  の実測（5 回計測中央値・実機・`#[ignore]` 分離）で再確定する**。本文書の
  閾値をそのまま確定値として扱わない。実測記録・採用閾値の根拠表は
  `docs/perf/dispatch-boundary-measurement.md` を参照。
- **閾値定数は 1 箇所（抽象層の定数、例: `crates/tensor-core` 内の
  `dispatch` モジュール）に集約する設計とする**（#68 で実装）。
- **TMA の扱い**: TMA（Tensor Memory Accelerator）の利用可否はカーネル
  内部の実装選択（どの WMMA/mma 系命令列を発行するか）であり、本文書が
  定義する経路選択条件（HW・形状ゲート）には含めない。GB10 実測での TMA
  選好（上記 `:125`）は Tensor Core 経路内部のチューニング材料として
  #60（TASK-11.1a WMMA 設計）の担当領域とする。

### 3.3 端数形状の扱い

WMMA fragment（16×16×16 等）・`simdgroup_matrix`（8×8 等）の整数倍でない
形状も、タイル端の手動境界チェックを維持したうえで Tensor Core／
`simdgroup_matrix` 経路の対象に含める。

- **性能下限・最適化の達成を理由に、シェーダ・カーネル側の手動境界チェック
  を省略しない**（REQ-8・`.claude/rules/coding-rust.md`「カーネル実装の境界
  検査（REQ-8）」）。境界検査を無効化する最適化（ベクトル化ロード・タイル端
  の分岐削減等）を適用する場合も、シェーダ側の手動境界チェックを維持したまま
  行う。本規約は CUDA（WMMA/mma）・Metal（`simdgroup_matrix`）の両カーネルに
  適用する。
- 端数形状を理由に Tensor Core 経路自体を除外する設計（フォールバック連鎖の
  形状条件に端数判定を含める設計）は採らない。境界検査はカーネル内部の責務
  とし、ディスパッチ規則の責務（HW・形状ゲート）とは分離する。

## 4. dtype ゲートと数値一致契約

| dtype | 経路 | HW ゲート | 数値一致閾値 | 状態 |
|---|---|---|---|---|
| f16 | CUDA mma.sync f16（優先。§5.6）／WMMA f16（フォールバック） | mma.sync: `cc >= (8, 0)` ／ WMMA: `cc >= (7, 0)` | 全ペア共通の統一複合判定（相対誤差 1e-3 未満 OR 絶対誤差 1e-5 未満、REQ-2・`.claude/rules/coding-rust.md`「バックエンド間数値一致」節。`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`＝`crates/backend-cpu/src/parity.rs:32,37`）を厳密ゼロ fail 判定として適用する。実機実測でこの厳密ゼロ fail が成立しない既知不合格形状に限り、spec REQ-2（2026-09-02 追記）の Tensor Core 経路 形状別判定方式（実測 baseline 非後退。`docs/cuda-tensor-core-parity-judgment-decision.md`・`ParityBaseline`）を承認済み baseline として適用する。PoC-v2-3 設計時点の暫定値（相対 5e-3 OR 絶対 5e-3、`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md:49`）は現行の統一複合判定へ置き換え済みであり、mma.sync 優先経路にも旧値は適用しない | 既存設計（数値一致閾値）を引用（変更なし）／実装優先順位は §5.6 で新規記録 |
| f32 | CUDA TF32 | `cc >= (8, 0)` | **既定採用を保留**。#186（TASK-11.1g）の実測再評価完了までは f32 は tiled 経路を既定とする段階案 | 保留（#186 待ち） |
| f32 | Metal `simdgroup_matrix` | Apple7 以上 | PoC-v2-4 実測（`simdgroup_matrix` f32、3.134 TFLOPS @4096、`docs/spec/03-poc/poc-v2-4-metal-gemm/README.md:93`）に基づく既存の全ペア統一複合判定（相対誤差 1e-3 未満 OR 絶対誤差 1e-5 未満、REQ-2） | 既存設計を引用（変更なし） |

**許容誤差（tolerance）の変更はユーザー承認必須**（`.claude/rules/coding-rust.md`
「バックエンド間数値一致テストの許容誤差を単独で緩和しない」・
`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は必ず
人間の承認を経る」）。本設計は TF32 経路の既定採用可否を#186 の実測・承認に
委ね、ディスパッチ規則側で閾値を先取りして確定しない。

## 5. 規則の定義形式（#68 への引き渡し仕様）

### 5.1 純関数シグネチャ

```rust
/// 形状・HW ケイパビリティ・dtype から使用カーネル経路を決定する。
///
/// 副作用なし・決定的（同一入力に対し常に同一出力）であるため、実機なしで
/// ユニットテスト可能（`DeviceCaps` をテスト用に構築するだけで検証できる）。
/// TASK-11.2b（#68）でバックエンド抽象層（TASK-1.9c のディスパッチ機構）へ
/// 統合される想定。
fn select_gemm_kernel(caps: &DeviceCaps, shape: GemmShape, dtype: DType) -> KernelKind;
```

- `DeviceCaps`: 2 節の HW 判定結果をキャッシュした構造体（`compute_capability`
  ・Metal GPU family 対応可否等）。デバイス初期化時に 1 回構築する。
- `GemmShape`: `M`・`N`・`K` を保持する形状記述（3 節の `min(M, N, K)` 判定
  に使う）。
- `DType`: `F32`／`F16`（4 節の dtype ゲートに使う）。
- `KernelKind`: フォールバック連鎖の到達先（5.2 節）を表す列挙。

### 5.2 フォールバック連鎖

- CUDA f16: `MatrixUnit(mma.sync f16) → MatrixUnit(WMMA f16) → Tiled → Naive`
  （`MatrixUnit` 経路内部の実装優先順位は §5.6 を参照。第 1 層の決定表
  〈本節・5.3 節〉は `MatrixUnit` までしか区別しない）
- CUDA f32: `Tiled → Naive`（TF32 `MatrixUnit` 経路は #186 の実測再評価まで
  既定採用を保留。4 節）
- Metal: `Simdgroup → Tiled → Naive`

PoC-v2-4 実測（`docs/spec/03-poc/poc-v2-4-metal-gemm/README.md:91-93`、
naive 1.271 TFLOPS／tiled 2.123 TFLOPS／simdgroup 3.134 TFLOPS、いずれも
M=N=K=4096・f32）が示すとおり、tiled は naive よりも一貫して高速（約 1.67
倍）である。したがって Metal の非 accelerated 経路（3.2 節「Metal は閾値
未満で非 accelerated」）は tiled を指す。naive は tiled 自体が使えない
場合（実装未完了時・境界条件等）の最終フォールバックとし、CUDA の
`TensorCore → Tiled → Naive` と対称な 3 段構成に揃える。

### 5.3 決定表（概念）

| HW ゲート | dtype | 形状 | 選択経路 |
|---|---|---|---|
| CUDA `cc >= 7.0` | f16 | 任意（形状下限なし、3.2 節） | MatrixUnit（実装優先順位は §5.6: `cc >= 8.0` かつ `n % 8 == 0 && k % 8 == 0` かつ grid 上限内 → mma.sync f16／それ以外 → WMMA f16） |
| CUDA `cc >= 8.0` | f32 | 任意 | Tiled（TF32 経路は #186 の実測再評価・ユーザー承認まで既定採用を保留。4 節） |
| CUDA `cc` 非対応 or ドライバ不在 | 任意 | 任意 | Tiled → Naive |
| Metal Apple7 以上 | f32 | `min(M,N,K) >= 512` | Simdgroup |
| Metal Apple7 以上 | f32 | `min(M,N,K) < 512` | Tiled |
| Metal Apple7 未満 | f32 | 任意 | Tiled → Naive |

- `BackendOps` v1 は f32 専用（`docs/public-api-design.md:469`）であるため、
  上表の CUDA 行は f16／f32 の両方を明記した。Metal f16 行は現時点で
  `BackendOps` が f32 のみを公開しており該当経路が存在しないため、本表には
  含めない（f16 経路の入口設計自体が TASK-1.9 実装時の未確定事項、
  `docs/public-api-design.md:469`）。

### 5.4 内部テスト用途の直接指定

テスト・証跡（#70）用途では、**内部 API（`pub(crate)` 等）で各経路を直接
指定可能**とする。バックエンド間数値一致テストが Tensor Core／
`simdgroup_matrix` 経路と tiled/naive 経路の両方を独立に検証する必要がある
ため（`select_gemm_kernel` の自動選択結果だけでは片方の経路しか実行されない
ケースがある）。この直接指定 API は**公開 API には露出しない**（REQ-11
受け入れ基準「明示切替 API を提供しない」の対象は利用者向け公開 API であり、
内部テスト用途はこの制約の対象外と整理する）。

### 5.5 #60（TASK-11.1a WMMA 設計）との境界

fragment サイズ・タイル構成等のカーネル内部構成は #60（WMMA/mma API 調査・
カーネル設計）の担当領域である。本文書は**経路選択条件（どのカーネルを
呼ぶか）のみ**を定義し、選ばれたカーネル自体の内部実装詳細には立ち入らない。
`MatrixUnit` 経路内で mma.sync／WMMA のどちらを呼ぶかという**実装優先順位**
は経路選択条件の一部として本文書 §5.6 に含める。一方、各カーネル内部の構成
（fragment・タイル・swizzle・`wmma_f16_opt`/basic 等の変種選択）は引き続き
#60 系列の担当領域であり、本文書の対象外のままとする。

### 5.6 f16 MatrixUnit 経路内の実装優先順位（#1131・#1150）

**背景**: `CudaGemmAuto::run_f16`（`crates/backend-cuda/src/gemm_auto.rs`）
は現状 `select_gemm_kernel` が `KernelKind::MatrixUnit` を返した場合に
`CudaWmmaGemm` のみを呼ぶ（WMMA f16）。`CudaMmaGemm`（`gemm_mma.rs`。
`mma.sync`/`ldmatrix`/`cp.async` 3 段パイプライン）は現状 `CudaGemmAuto` に
未結線で証跡用途のみ（テスト・ベンチから直接構築される経路。本番非到達）。
GB10 実機実測（#1123・`docs/perf/cuda-wmma-f16-perf-triage.md` §3.1・§4.1）
では `mma_sync_f16` が `wmma_f16_opt` に対し形状依存で約 4.1〜10.8 倍高速
（dim 512: 約 4.1 倍／1024: 約 4.4 倍／2048: 約 7.3 倍／4096: 約 10.8 倍）
であり、`tensor_core_tflops_record` の f16 assert（f16 カーネルが tiled を
上回ること）が GB10 で FAIL している。本節は #1131（f16 経路を mma.sync
パイプラインへ結線）の第 1 子 Issue（#1150）として、`MatrixUnit` 経路
**内部**の実装優先順位を設計として確定する。

**2 層構造（第 1 層＝本文書 §2〜§5.3 の決定表は変更しない）**:

- **第 1 層（不変）**: `select_gemm_kernel`（`fandhe_ai_tensor_core::dispatch`）
  は HW・形状・dtype から `KernelKind` を返す。CUDA f16 は `cc >= (7, 0)`
  で形状下限なしの `MatrixUnit`（§3.2・§5.3 のまま）。tensor-core クレート
  の決定表・定数は本節の変更対象外
- **第 2 層（新規）**: `backend-cuda::CudaGemmAuto` が `MatrixUnit` の
  **実装**を `CudaMmaGemm → CudaWmmaGemm → Tiled` の優先順位で選ぶ。判定
  材料（LDGSTS 要件 `cc >= 8.0`・NVRTC コンパイル成否・`cp.async` 整列
  制約・grid 上限）はいずれも `backend-cuda` 内部の事実であり `DeviceCaps`
  へ持ち上げる必要がないため、第 1 層は変更しない
- §3.2「閾値定数は 1 箇所（`tensor-core::dispatch`）に集約する」の原則は
  **第 1 層の規則定数**に適用される。第 2 層の
  `gemm_mma.rs::MIN_COMPUTE_CAPABILITY_MAJOR`（`= 8`。LDGSTS 要件の独立
  定数、理由は同ファイル該当箇所のコメントに記載済み）を `tensor-core`
  側へ集約する変更は**本 Issue のスコープ外**とする（6 節の表に将来候補
  として記録するのみ。起票はユーザー承認事項のため本 PR では行わない）
- `tests/dispatch_boundary.rs` の既存整理「mma パイプラインの優劣は経路
  選択条件でなくカーネル内部チューニング」は**第 1 層について引き続き
  真**であり、第 2 層の実装優先順位はこの整理と矛盾しない（第 1 層の決定
  表・境界テストの対象は変わらない）

**第 2 層の判定規則（決定的・fail-safe。実装は #1152（構築）・#1156（分岐切替。実装済み）が担当）**:

1. **構築時（`CudaGemmAuto::new`。#1152 で実装済み。診断用
   `mma_available`／`mma_unavailable_reason` アクセサを併設）**:
   `CudaMmaGemm::new(device)` を `CudaWmmaGemm::new(device).ok()` と同型の
   fail-soft で構築する。
   `cc < 8.0`（`TensorCoreUnsupported`）・base カーネル（`mma_f16`）の
   NVRTC コンパイル失敗は `CudaMmaGemm::new` 自体が `Err` を返すため
   `mma = None` として握り潰す（現行 `wmma` フィールドの構築方針と対称）。
   一方 **SM 数取得失敗（`device.multiprocessor_count()` が `None`）は
   base カーネルの可用性とは独立**であり、`CudaMmaGemm::new`
   （`gemm_mma.rs` 実装）は `mma_f16_swizzle`／`swizzle_group_width` のみ
   を `None` へ fail-soft に縮退させて `Ok(Self)` を返す（swizzle は L2
   再利用の性能最適化に過ぎず base カーネルの可用性とは独立であるべき、
   という既存設計判断のまま）。したがって SM 数取得失敗単独では
   `mma = Some`（swizzle 変種なしの base カーネルのみ）のまま
   `CudaGemmAuto::new` に組み込まれ、後続の #1152（`CudaMmaGemm::new(
   device).ok()` を利用する実装）はこの契約（`mma = None` になるのは
   `cc < 8.0` または base カーネルのコンパイル失敗時のみ）を前提とする。
   `wmma` フィールドはフォールバック用に維持する
2. **呼び出し時（`run_f16`）**: `KernelKind::MatrixUnit` の場合、`mma` が
   `Some` **かつ** 形状が mma 固有制約（`validate_mma_alignment(n, k)`
   ＝ `n % 8 == 0 && k % 8 == 0`・`validate_mma_grid_bounds(m)` ＝
   `m.div_ceil(MMA_BM) <= 65_535`。いずれも `gemm_mma.rs` 実装済み）を
   満たすときのみ `CudaMmaGemm::run_f16` を呼ぶ。満たさなければ `wmma`
   （`Some` なら）へ、`wmma` も `None` なら `run_tiled_f16` へフォール
   バックする
3. **形状ゲートは呼び出し前に事前判定し、エラー駆動のフォールバック
   （mma 実行の `Err` を捕捉して WMMA を再実行する方式）は採らない**。
   理由: (a) 本節冒頭の決定的規則方針との整合、(b) カーネル起動失敗・
   poison 状態を静かに別経路で覆い隠さない（`docs/backend-cuda-async-\
   execution-design.md` の同期契約と整合）、(c) 現行「tiled 自体の失敗は
   そのまま呼び出し元へ伝播する」方針との対称性。事前判定を怠ると、
   `n % 8 != 0` 等の形状で現行 WMMA では成功していた呼び出しが
   `InvalidShape` になる退行が生じる点を実装引き渡し事項として明記する
   （#1156 で実装済み: 単一真実源の純関数
   `gemm_auto::select_f16_matrix_unit_impl(mma_available, wmma_available,
   m, n, k) -> F16MatrixUnitImpl` が事前形状ゲートを含む判定を担い、
   `run_f16` はその結果に従って `match` で実装を呼び分ける）
4. no-op／退化形状（`m == 0 || n == 0 || k == 0`）は mma／WMMA／tiled の
   いずれも同一契約の早期 return で処理する。**`m == 0 || n == 0` は空
   `Vec` を返す**が、**`k == 0`（`m, n > 0`）は GEMM の数学的定義どおり
   `m * n` 個のゼロ（`vec![f16::ZERO; m * n]`）を返す**（A/B が空スライス
   になるため起動を回避し C = 全 0 とする。`gemm_mma.rs::run_f16`・
   `gemm_wmma.rs::run_f16` の早期 return と同一契約。§5.6 の実装優先順位
   〈判定規則 1〉はこれら早期 return より前の構築時契約であり、いずれの
   早期 return も `validate_mma_alignment`／`validate_mma_grid_bounds` 等の
   mma 固有ゲート評価より前に行われるため、本節の優先順位判定自体には
   影響しない
5. §3.3「端数形状を理由に Tensor Core 経路自体を除外しない」は**第 1 層で
   引き続き維持**される（`cp.async` 整列非対応の端数形状も `MatrixUnit`
   内の WMMA で処理され、tiled へは落ちない）
6. `CudaMmaGemm` 内部の swizzle 変種選択・`wmma_f16_opt`/basic のどちらを
   使うかは各構造体**内部**の責務（§5.5 の境界のまま）。`wmma_f16_opt`
   の維持・格下げ判断は #1160 が担当する
7. 利用者向けの切替 API・環境変数は本節でも新設しない（REQ-11・§1 の
   方針を維持する）

**数値一致・性能の引き渡し**:

- 数値一致: REQ-2 複合判定＋spec REQ-2（2026-09-02 追記）の形状別判定方式
  （`docs/cuda-tensor-core-parity-judgment-decision.md`。厳密ゼロ fail 判定
  が成立しない形状は実測 baseline 非後退方式）。tolerance 定数・
  `ParityBaseline` 行の追加・変更はユーザー承認必須（本節では確定しない）。
  GB10 実機実測は #1158 が担当する
- 性能: 切替前後を同一プロトコル・5 回計測中央値で比較し、後退時は結線
  しない（#1156 のユーザー承認条件）。TFLOPS 記録・`wmma_f16_opt` の扱い
  確定は #1160 が担当する

**実装 Issue 対応表**: 構築（`mma` フィールド追加。fail-soft）は #1152
（実装済み）、呼び出し分岐切替（形状ゲート込みの優先順位判定。
`select_f16_matrix_unit_impl`）は #1156（実装済み）、GB10 数値一致
非後退検証は #1158、TFLOPS 記録・`wmma_f16_opt` 扱いの確定は #1160 が
担当する。

## 6. スコープ外

以下は本イシュー（#67・TASK-11.2a）のスコープ外であり、後続イシューが担当する。

| 事項 | 担当イシュー |
|---|---|
| 規則の実装・バックエンド抽象層への統合 | #68（TASK-11.2b） |
| 境界形状の実測再検証（5 回計測中央値・実機・`#[ignore]` 分離。計測テスト・記録テンプレートは `docs/perf/dispatch-boundary-measurement.md` 参照） | #69（TASK-11.2c）。Metal 側は #382 で実測完了（変更提案あり・提案値 384。コード未変更・承認前提）。CUDA 側は #388 ツリー（#389/#390）が引き続き担当 |
| 証跡整備（カーネルソース内命令の実在＋ベンチログ） | #70（TASK-11.3） |
| TF32 経路の数値一致閾値の実測再評価・既定採用可否のユーザー承認 | #186（TASK-11.1g） |
| CUDA Tensor Core（WMMA/mma）カーネル自体の実装 | #60 系列（TASK-11.1） |
| CPU 側 ISA dispatch の変更 | 対象外（`gemm_blis/microkernel.rs` は実装済み・変更なし） |
| f16 `MatrixUnit` 経路の mma 優先実装（`CudaGemmAuto` へのフィールド追加・分岐切替） | #1152（フィールド追加は実装済み）・#1156（分岐切替。実装済み） |
| GB10 数値一致非後退（§5.6 の判定規則の実機検証） | #1158 |
| TFLOPS 記録・`wmma_f16_opt` の扱い確定 | #1160 |
| `gemm_mma.rs::MIN_COMPUTE_CAPABILITY_MAJOR` の `tensor-core::dispatch` への集約（未起票・候補） | 対象外（§5.6 参照。第 2 層定数のため現状維持） |

## 7. 出典一覧

- `docs/spec/04-requirements.md` REQ-11（226〜241 行）・REQ-2（66 行以降）・
  REQ-8（性能下限節）・REQ-13（256〜266 行）
- `docs/spec/05-tasks.md` TASK-11.1〜11.3（333〜352 行）
- `docs/spec/03-poc/poc-8-matrix-unit/README.md`（512 境界・Metal 20.5 倍・
  CUDA(GB10) 1.4〜1.6 倍・TMA 選好）
- `docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`（CUDA tiled 実測・tensor
  core 化の段階見積もり・f16 閾値設計）
- `docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`（`simdgroup_matrix` 実測
  3.134 TFLOPS）
- `crates/tensor-core/src/device.rs`・`crates/backend-cuda/src/device.rs`
  （既存 HW 判定 API）
- `crates/backend-cpu/src/gemm_blis/microkernel.rs`（トークン型 dispatch の
  先行事例）
- `docs/public-api-design.md` §4（`BackendOps`・`Device` シグネチャとの整合）
- `.claude/rules/coding-rust.md`（カーネル実装の境界検査規約・REQ-8）
- `.claude/rules/security.md`（ガードレール閾値・許容誤差変更の承認要件）
- `crates/backend-cuda/src/gemm_auto.rs`（`CudaGemmAuto`・`run_f16` 現行実装）
- `crates/backend-cuda/src/gemm_mma.rs`（`CudaMmaGemm`・cc ゲート・
  `validate_mma_alignment`／`validate_mma_grid_bounds`）
- `crates/backend-cuda/src/gemm_wmma.rs`（`CudaWmmaGemm`・cc ゲート）
- `docs/perf/cuda-wmma-f16-perf-triage.md`（#1123・GB10 実測倍率）
- `docs/perf/cuda-parity-baseline.md`（`ParityBaseline`・形状別判定方式）
- `docs/cuda-tensor-core-parity-judgment-decision.md`（Tensor Core 経路の
  受け入れ判定方式の決定記録）
- `docs/backend-cuda-async-execution-design.md`（CUDA 非同期実行の同期契約）
