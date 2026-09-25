# `log_softmax` forward の CUDA／Metal GPU カーネル化（イシュー #2155）

## 1. 背景・目的

`BackendOps::log_softmax`（イシュー #1594）は CPU 側に融合カーネル
（`fandhe_ai_backend_cpu::softmax::run_log_softmax_f32`）を持つが、
CUDA・Metal は既定の `Unsupported` のままで、`Var::log_softmax` は
両バックエンドともホスト参照実装（`fandhe_ai_autodiff::eval::
log_softmax_along`）へフォールバックしていた（`BackendOps::softmax`
は CUDA #594・Metal #604 でカーネル化済みだったが、`log_softmax`
forward だけがこの gap を残していた）。

本 doc は、この gap をイシュー #2155 で解消するにあたっての CUDA／
Metal 共通の数値契約・設計判断を記録する（Metal `min` のカーネル化
〈同一イシュー〉は `docs/backend-metal-reduce-sum-design.md` §13 を
正とし、本 doc の対象外）。

## 2. 契約の核心

### 2.1 `softmax` カーネルとの共有部分

`softmax` の online max/sum 計算（`(m, l)`。`m` は生ドメインの行 max
で NaN を無視する `fmaxf`／MSL の `max`、`l = Σ exp2((raw - m)・
log2(e))`）をそのまま再利用する。CUDA は `kernels_softmax.rs` の
1 パス／2 パス構造・境界マスク定数（`SOFTMAX_MASK_E2`）・warp 内
butterfly reduction を、Metal は `shaders/softmax.metal` の online
max/sum ループ・`SOFTMAX_NEG_FLT_MAX` センチネル・simdgroup
reduction をそれぞれ逐語で共有する。

### 2.2 `log_softmax` 固有の最終書き出し

出力は **`(raw - m) - log(l)`**（CUDA は `logf(l)`、MSL は `log(l)`）。
CPU 参照実装（`fandhe_ai_backend_cpu::softmax::run_log_softmax_f32`）
と同じ **Sterbenz 順序**（先に `raw - m` を計算し、その後で `log(l)`
を引く）を守り、`raw - (m + log(l))`（分配後に単一減算する形）には
しない。`log(l)`（自然対数。`softmax` の `exp2`／`log2(e)` とは異なり
底 2 ではなく自然対数を使う理由は CPU 参照実装の `ln` と数学的に
対応させるため）は行あたり 1 回だけ計算する。

### 2.3 数値一致判定

`exp2`（`l` の計算）と `logf`／`log`（自然対数）の丸め差・総和順序の
差があるため **bit 一致は主張せず**、REQ-2 統一複合判定（相対誤差
1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::parity::
assert_parity`）で検証する。`softmax` と同じ扱いであり、
`.claude/rules/coding-rust.md` の tolerance 契約は変更しない。

### 2.4 境界の入力

- 全要素 `-inf` の行 → 全要素 `NaN`（`softmax` と同じ理由。
  `(-inf) - (-inf) = NaN`）。
- `+inf` を含む行 → 全要素 `NaN`。
- NaN を含む行 → 全要素 `NaN`。
- `±f32::MAX` が混在する行 → 最小側の要素は `-inf`（`m` との差が
  `-f32::MAX - f32::MAX` 相当になり有限範囲を超えるため）。

いずれも CPU 参照実装と NaN クラス一致（`f32::is_nan()` のみで比較。
REQ-2 の複合判定は非有限値には適用しない）で検証する。

### 2.5 最終軸限定

`softmax`・`log_softmax_backward` と同じく最終軸限定。非最終軸は
`row_softmax_layout` が `Ok(None)` を返し、`BackendOps` 既定の
`Unsupported` をそのまま返す（ホスト参照実装へのフォールバックの
合図）。この既定・迂回経路は変更しない。

## 3. 実装構成

### 3.1 CUDA（`crates/backend-cuda/src`）

- `kernels_softmax.rs`: `LOG_SOFTMAX_F32_ONEPASS`／
  `LOG_SOFTMAX_F32_TWOPASS` を追加する。`SOFTMAX_F32_ONEPASS`／
  `SOFTMAX_F32_TWOPASS` と同じ online (m, l) 計算・`float4`
  ベクトル化・`if (base + 3 < cols)` の手動境界チェック（REQ-8）・
  `long long` ループ添字を共有し、最終書き出しのみ §2.2 の式に
  変える。
- `softmax.rs`: `CudaSoftmax` に `log_onepass_f32`／`log_twopass_f32`
  を追加し、`new` で追加の 2 PTX をコンパイル・ロードする（初回
  構築コストは 2→4 PTX に増えるが `context_cache::cached_softmax`
  により 1 回限り）。`run_softmax_f32_raw` の本体（経路選択・grid
  導出・確保・起動・readback）を private helper
  `run_row_kernel_f32_raw(x, scale, rows, cols, onepass, twopass)`
  へ抽出し、`run_softmax_f32_raw`／新設の `run_log_softmax_f32` の
  両方がこれを呼ぶ。既存の unsafe 起動ブロックは移動するだけで、
  新たな unsafe ブロックは追加しない。
- `ops.rs`: `CudaBackendOps::log_softmax` を追加する。既存
  `softmax` オーバーライドと同型（`row_softmax_layout` →
  非最終軸なら `Unsupported` → `contiguous` → `cached_softmax` →
  `run_log_softmax_f32`）。

### 3.2 Metal（`crates/backend-metal/src`）

- `shaders/softmax.metal`: `log_softmax_f32_onepass`／
  `log_softmax_f32_twopass` を追加する。buffer index 0〜4 は
  `softmax_f32_*` と同一（`encode_softmax_dispatch` を流用するため）。
  `exp(` は使わない契約（既存の source-evidence テストが検査する）。
- `softmax.rs`: `MetalSoftmax` に `log_onepass`／`log_twopass` を
  追加し、`new` で生成・`threadExecutionWidth == 32` の検証対象に
  入れる。`run_softmax_f32` の本体を private helper
  `run_row_kernel_f32(ctx, x, rows, hidden, onepass, twopass)` へ
  抽出し、`run_softmax_f32`／新設の `run_log_softmax_f32` の両方が
  これを呼ぶ。`encode_softmax_dispatch` を流用するため新規 unsafe
  なし。
- `ops.rs`: `MetalBackendOps::log_softmax` を追加する。既存
  `softmax` と同型（デバイス初期化より前に `row_softmax_layout` で
  非最終軸を判定する点を含む）。

### 3.3 trait doc（`crates/tensor-core/src/backend_ops.rs`）

`BackendOps::log_softmax` の doc コメントを更新し、CUDA・Metal も
GPU カーネルでオーバーライドしていること・数値契約（REQ-2 統一複合
判定）を明記した。シグネチャは変更しない。

## 4. スコープ外

- 非最終軸の `log_softmax` GPU 化。
- `log_softmax` の `exp`／`log` の近似化。
- `CudaSoftmax::new` の初回構築コスト増（2→4 PTX）自体の最適化。

## 5. 実測状況

本実装環境（Linux）は CUDA／Metal 実機に到達できないため、Linux
実行可能な検証（`kernels_softmax.rs` のソース文字列証跡テスト・
`rmsnorm_softmax_source_evidence.rs`・`api_surface.rs`）のみ完了
している。実機実測は Mac／DGX Spark セッションへ申し送る
（`docs/perf/logs/gpu-min-log-softmax-2155/README.md` 参照）。
