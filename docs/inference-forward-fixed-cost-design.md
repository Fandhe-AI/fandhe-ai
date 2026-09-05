# 推論 forward の固定費削減設計（イシュー #1028）

親イシュー #1008 の診断どおり、フレームワーク横並びベンチ
（`scripts/bench/framework-compare`）で観測される candle との推論
forward 速度差（CPU/CUDA/Metal いずれも約 3〜6 倍）は、GEMM カーネル
そのものの性能差ではなく **forward 1 回あたりの固定費**（tape 構築・
葉登録によるパラメータ／入力のホストコピーと再アップロード、演算
ごとのホスト実体化＝D2H 同期）に起因する。本文書は現行 3 経路の固定費
内訳、tape 不要経路（段階 A・実装済み）の設計、活性化のデバイス常駐
チェーン（段階 B・設計確定・実装は out-of-scope）を記録する。

## §1 背景・引用元

- 親イシュー #1008、`docs/perf/` 配下の各バックエンド診断
- framework-compare 推論プロトコル: batch 64・784→256→ReLU→10・
  warmup 20・iters 20・中央値（本文書 §6 のベンチもこのプロトコルに
  合わせる）
- Phase 1 先行成果: #1011〜#1013（CUDA 非同期実行）・#1015〜#1017
  （Metal コマンドバッファ共有）・#1018〜#1021（デバイスメモリ
  プールアロケータ）・#1022/#1023（`DeviceParamStore` パラメータ常駐）

## §2 現状分析（固定費の内訳）

本イシュー着手時点の推論経路は 3 つあり、いずれも forward 1 回ごとに
固定費を払っていた。

| 経路 | 場所（着手時点） | 固定費 |
|------|------|--------|
| `Sequential::forward` + 外部 `Tape` | `crates/facade/src/compat/sequential.rs`（`forward`） | 層ごとの `Linear::bind` → `tape.var(&weight)` が weight/bias を毎回 `clone()`（`crates/autodiff/src/tape.rs::Tape::var`）。GPU では毎回 H2D 再アップロード。演算ごとに戻り値の D2H（構造的な同期点） |
| `Sequential::predict`（本イシュー着手前の実装） | 旧 `predict`（現 `predict_via_tape`） | 上記に加え `Tape` 生成・入力 `tape.var`（clone）・ノード記録・`to_tensor()` |
| `Sequential::predict_resident` | `crates/autodiff/src/optim/device_store.rs::linear_forward` | weight の D2H/H2D は #1022 で排除済みだが、`gemm_resident_rhs` が返す `y` はホスト常駐 `Tensor`（`docs/backend-cuda-async-execution-design.md` §2.3「戻り値の D2H が構造的な同期点」）。中間活性化は層ごとにホスト実体化される |

`gemm_resident_rhs`（`crates/tensor-core/src/backend_ops.rs`）はホスト
常駐の `a`・戻り値を扱う契約であり、多層 MLP では層ごとに D2H
（前層の出力）→ H2D（次層の入力）が発生する。GPU 側のギャップの本丸は
「活性化のデバイス常駐化」であり、CPU 側は clone・tape 記録・
アロケーションの削減が効く。

## §3 設計

### §3.1 段階 A: tape 不要の推論 forward 経路（実装済み）

`fandhe_ai_autodiff::nn::module::Module` trait に非破壊拡張
（デフォルトメソッド追加）として `forward_host` を追加した:

```rust
fn forward_host(
    &self,
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    Err(AutodiffError::Backend(BackendError::Unsupported(...)))
}
```

`Tape`／`Var` を一切構築せず、`ops: &dyn BackendOps` を直接受け取って
ホスト常駐 `Tensor<f32>` で 1 回分の forward を計算する。`Linear`・
`Relu`・`Sigmoid`・`Tanh`（`docs/compat-api-scope.md` §1 の 3 種）は
いずれもこのデフォルトをオーバーライドする。

**bit-exactness の確保方針**: 各オーバーライドは、旧経路
（`Module::forward`。`Var` 経由）が実際に呼ぶのと**同一の演算**を
直接呼ぶ。

- `Linear::forward_host`: `ops.gemm(input, weight)` → `ops.add(&y, bias)`
  （`LinearVars::forward` が `input.matmul(&weight)` → `.add(&bias)` と
  非融合合成するのと同一。融合カーネル `gemm_bias_act` は使わない
  — 融合 epilogue はカーネル内 tiling 次第で加算順序が変わりうるため）。
  **イシュー #1137 追記**: CUDA バックエンドの `ops.gemm` は内部で
  `CudaGemm::run_tiled_f32` を経由し、#1137 以降これは cp.async
  パイプラインへ形状条件付きに分岐しうるが、新旧経路（`forward_host`／
  `Module::forward`）は**いずれも同一の `ops.gemm` 呼び出し**を通るため
  （分岐の有無自体は経路差ではなくカーネル選択の内部詳細）、本節の
  bit-exactness 契約は #1137 の影響を受けない。

- `Relu::forward_host`: `ops.relu(input)`（`Var::relu()` が
  `tape.ops().relu` を呼ぶのと同一のディスパッチ）
- `Sigmoid::forward_host` / `Tanh::forward_host`: `eval::sigmoid`/
  `eval::tanh`（`Var::sigmoid()`/`tanh()` が `BackendOps` を経由せず
  直接呼ぶホスト計算関数と同一）

`crates/facade/src/compat/sequential.rs::Sequential::predict` の内部
実装を差し替えた:

```rust
pub fn predict(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
    match self.predict_tape_free(input) {
        Err(AutodiffError::Backend(BackendError::Unsupported(_))) => self.predict_via_tape(input),
        other => other,
    }
}
```

`predict_tape_free` は CPU の `CpuBackendOps` を直接構築し（`crate::
tape()` が常に CPU バックエンドで `Tape` を構築するのと同じ「CPU 固定」
契約）、層ごとに `Module::forward_host` を呼ぶ。**公開シグネチャ・
戻り値・数値結果は不変**（`predict` の型・エラー種別は変更していない）。

**フォールバック契約（fail-closed）**: 層構成に `forward_host` 未対応の
`Module` 実装が含まれる場合（`docs/compat-api-scope.md` §1 の 3 種は
いずれも対応済みのため通常到達しない）、1 層でも `Unsupported` を返した
時点で `predict_tape_free` 全体を打ち切り（部分的な結果を使わない）、
`predict` が旧経路 `predict_via_tape`（`Tape`/`Var` 経由）へ全体
フォールバックする。黙示のホストフォールバックによる部分実行はしない。

**検証**: `crates/facade/src/compat/sequential.rs` の
`sequential_predict_tape_free_matches_via_tape_bit_exact`（Linear・
Relu・Sigmoid・Tanh 混在・複数バッチ・bias 有無混在）・
`sequential_predict_public_builder_tape_free_matches_via_tape`
（`Sequential::new()` の通常構築経路）が新旧経路の bit 完全一致を
確認する。

### §3.2 段階 B: 活性化のデバイス常駐チェーン（設計確定・実装は一部）

GPU（CUDA/Metal）で最終出力 1 回まで同期点を集約するには、活性化
チェーン自体をデバイス常駐のまま実行する必要がある。`BackendOps` に
非破壊のデフォルトメソッド（fail-safe: `BackendError::Unsupported`。
`gemm_resident_rhs` と同じ拡張様式）を追加した:

```rust
/// a[m,k]・w[k,n]（・bias[n]。デバイス常駐）から y[m,n] を
/// デバイス常駐のまま計算する。act は bias 加算後の elementwise 適用。
fn linear_forward_device(
    &self,
    a: &DeviceBuffer<f32>,
    w: DeviceBufferView<'_>,
    bias: Option<DeviceBufferView<'_>>,
    act: Activation,
) -> Result<DeviceBuffer<f32>, BackendError> {
    Err(BackendError::Unsupported(...))
}
```

**CPU 実装（実装済み・`crates/backend-cpu/src/ops.rs`）**: `a`／`w`／
`bias`／戻り値をいずれも `DeviceBuffer`（CPU では実体はホストメモリの
`Vec<f32>`）のまま扱う。bit-exactness 契約は段階 A と同じ理由で
非融合合成（`gemm_blis_parallel` → bias 行方向複製加算 → `max(x,0)`）
を採用し、融合カーネル `gemm_blis_bias_act_parallel` は使わない。
`CpuMemory::wrap_vec`（新設・`pub(crate)`）が計算済みの `Vec<f32>` を
コピーなしで `DeviceBuffer` へラップする（`AllocationTracker` への
計上は `alloc_zeroed`/`upload` と同じ計測系列を共有）。

**検証**: `crates/backend-cpu/tests/linear_forward_device_parity.rs` が
旧経路（`gemm` → `add` → `act`）と bit 完全一致することを、MR/NR/MC/
KC/NC 境界を跨ぐ複数形状・bias 有無・activation（None/Relu）の組み
合わせで確認する。境界検査（shape 不整合・bias shape 不一致）も
カーネル本体へ触れる前に拒否することを確認する。

**CUDA／Metal 実装（実装済み・イシュー #1216）**: `CudaBackendOps::
linear_forward_device`（`crates/backend-cuda/src/ops.rs`）・
`MetalBackendOps::linear_forward_device`（`crates/backend-metal/src/
ops.rs`）は、それぞれ `gemm_resident_rhs*` と同じ融合カーネル
（`CudaGemm::launch_tiled_bias_act_f32_resident`／`MetalGemm::
encode_strided_bias_act_prepared`。`dispatch_strided_bias_act_prepared`
の encode-only 版で、`ctx.synchronize()` を呼ばずコマンドバッファへ
積むのみで待たない。イシュー #1216・codex-review 指摘対応）を再利用し、
`a`・戻り値もデバイス常駐のまま扱う（`w`／`bias` のみ常駐の
`gemm_resident_rhs*` との違い）。CUDA は世代検査（poison／stale
generation）の対象に `a` 自体も追加（呼び出しを跨いで生存する常駐
バッファのため）。出力バッファは呼び出し元へ escape するため
`static_cuda_memory`／`static_metal_memory`（`memory_ops()` と同一
インスタンス）で確保し、REQ-14 の単一計測系列を維持する。Metal の
コマンドバッファ共有（#1017）の同期境界（前層出力を次層の入力として
読む順序）は M4 Max 実機の 2 層チェーンテストで確認済み（`docs/perf/
linear-forward-device-gpu.md`）。

**facade／autodiff への結線（スコープ外・引き継ぎ）**: `DeviceParamStore`
の推論ヘルパー・`Sequential::predict_resident` の内部差し替え
（結線の実施可否は before/after 実測での非後退確認が条件）は、
イシュー #1216 の Phase 2 として引き続きスコープ外とし、
`docs/perf/linear-forward-device-gpu.md` §5 に判断・引き継ぎ内容を
記録する。

### §3.3 検討結果（受け入れ条件 2 の文書化）

- **(a) 現行 3 経路の固定費内訳**: §2 に記録済み
- **(b) tape 不要経路の設計と採否判断**: §3.1 のとおり採用・実装済み。
  bit-exactness は「旧経路が実際に呼ぶのと同一の演算を直接呼ぶ」方針
  で担保し、融合カーネルへの置き換えは行わない（累積順序が変わり
  うるため）
- **(c) 活性化デバイス常駐チェーンの設計**: §3.2 のとおり確定。CPU に
  加え CUDA/Metal もイシュー #1216 で実装済み（実測は `docs/perf/
  linear-forward-device-gpu.md`）
- **(d) 不採用・保留事項**: 多層融合カーネル（GEMM+bias+act+GEMM の
  複数層またぎ融合）・CUDA Graph によるチェーン全体のグラフ化は本
  イシューでは検討していない（親イシュー #1008 の Phase 3 以降・
  candle との残差の議論へ送る）
- **(e) 改善前後の実測記録**: §6 参照

## §4 スコープ外（引き継ぎ事項）

CUDA／Metal `linear_forward_device` の実装・`#[ignore]` 実機 parity
テスト・実機実測はイシュー #1216 で完了した（`docs/perf/linear-
forward-device-gpu.md`）。以下は引き続きスコープ外とし、
`out-of-scope-tracking.md` の規約に従い追跡する:

- `facade`／`autodiff`（`DeviceParamStore` の推論ヘルパー・
  `Sequential::predict_resident` の内部差し替え）への結線（#1216
  実装計画 Step 9「Phase 2」。判断は `docs/perf/linear-forward-
  device-gpu.md` §5 を参照。#1217 では facade Phase 2 を明示的にスコープ外
  としたため、なお未着手のまま引き続き後続 Issue へ引き継ぐ）
- `Sigmoid`/`Tanh` を含む層構成のデバイス常駐対応（段階 B は現状
  `Activation::{None,Relu}` のみ対応。`gemm_resident_rhs` と同じ制約）
- ~~framework-compare の推論プロトコルへの reuse モード追加~~ →
  イシュー #1217 で実装済み（`bench-fandhe --task infer --mode reuse` /
  `--phases`。`docs/perf/infer-reuse-phase-breakdown.md`）。ただし
  facade Phase 2（上記）が未結線のため、reuse モードは `predict`
  相当の重み再構築コストを削減するのみで #1216 の中間活性化デバイス
  常駐効果は反映されていない（同ドキュメント §6）
- candle との残差ギャップ全体の解消（親イシュー #1008 Phase 3 以降）

## §5 fail-closed 検証（REQ-8・OWASP A03）

- `Module::forward_host`（`Linear`）・`BackendOps::linear_forward_device`
  （CPU／CUDA／Metal）はいずれもカーネル本体（`gemm_blis_parallel`
  等）へ触れる前に rank・shape・デバイス一致を検証する（`gemm_
  resident_rhs` と同水準）。CUDA はさらに poison／世代検査（`a` 自身
  も対象。`docs/perf/linear-forward-device-gpu.md` §2 参照）を driver
  へ触れる前に通す
- `linear_forward_device` は `act` が `Activation::{None,Relu}` 以外の
  場合（`#[non_exhaustive]` の将来 variant）を明示的に拒否する
  （`gemm_bias_act` の非融合フォールバックと同じ方針）
- フォールバック（`predict` の `Unsupported` 検出）は部分実行を伴わない
  （1 層でも失敗すれば全体を旧経路へやり直す）

## §6 実測記録

### CPU（本ラン・ローカル実測）

`cargo test -p fandhe-ai --release --test infer_fixed_cost_bench --
--nocapture` で計測（batch 64・784→256→ReLU→10・warmup 20・iters 20・
5 trial 中央値）:

```
via_tape_median_s   = 0.000270202  (q1=0.000269511, q3=0.000272357)
predict_median_s    = 0.000262453  (q1=0.000260957, q3=0.000265052)
speedup_x           = 1.030
predict_faster      = true
```

CPU は host-resident のまま同一プロセスで完結するため upload/download
のコスト自体が小さく、改善幅は小さい（`device_param_store_bench.rs`
冒頭コメントの「CPU バックエンドでは差が計測ノイズに埋もれうる」と
同じ注記が当てはまる）。段階 A の狙い（tape 構築・葉クローン・ノード
記録の固定費削減）は達成しており、退行（`predict_faster=false`）は
発生していない。

### Metal（M4 Max 実機）

`facade::predict_resident` への結線は未実施（§4・Phase 2 引き継ぎ）
のため `infer_fixed_cost_bench` 自体は未計測のまま。backend レベルの
`linear_forward_device` 単体 before/after 実測（2 層チェーン。旧経路
`gemm_resident_rhs_act` ×2 vs 新経路 `linear_forward_device` ×2）は
`docs/perf/linear-forward-device-gpu.md` §4 に記録済み（M4 Max 実機・
2026-09-06・5 回計測でいずれも speedup > 1）。

### CUDA（DGX Spark GB10 実機）

本エージェント実行環境に CUDA 実機がないため未計測。`linear_forward_
device` の CI 実行可能ユニットテスト（poison／stale generation／
DeviceMismatch）は実機なしで検証済み（`crates/backend-cuda/src/
ops.rs`）。実機実測は `docs/perf/linear-forward-device-gpu.md` §4 に
記入欄を残す。
