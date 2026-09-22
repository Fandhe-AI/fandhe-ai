# ONNX import モデルの GPU 実行（`OnnxModel::run` GPU 実行）設計判断（#2077）

イシュー #2077「ONNX import モデルの GPU 実行（`OnnxModel::run` のホスト CPU
限定解除）」（親 #2076）に対応する。`fandhe_ai::interop::onnx::OnnxModel::run`
は従来 `fandhe_ai_onnx_interop::onnx::interp::run`（素朴なホスト CPU 実装。
`BackendOps`／`Device` 非経由）へ委譲するだけだった。本イシューでグラフ実行を
`&dyn BackendOps` 経由で CUDA・Metal のネイティブカーネルへ段階的に到達させる
dispatcher を新設し、facade の opt-in スイッチ（既定 OFF）から利用可能にした。

## 承認ステータス: 自動運転モード実装（本ドキュメントが承認記録を兼ねる）

`docs/cuda-tf32-optin-api-decision.md` と同じ扱い。自動運転モードのエージェント
が実装計画に記載済みの設計判断（安全側の既定 OFF・fail-closed 方針・bit 不変
契約）をそのまま採用した。ユーザーが後から確認・差し戻し可能な形で記録する。

## 1. dispatcher の配置

イシューが例示する「`BackendOps::onnx_run` メソッド」は実装しない。
`BackendOps`（`tensor-core`）は `Graph`（`onnx-interop`）を参照できない（依存
方向が `onnx-interop → tensor-core` に固定されており、逆向きは循環依存になる）
ため trait メソッドとしては配置不能。代わりに `onnx-interop` 側へ
**`interp::run_with_ops(graph, feeds, ops: &dyn BackendOps)`** を新設した
（`crates/onnx-interop/src/onnx/interp.rs`）。`ops` が `CpuBackendOps`／
`CudaBackendOps`／`MetalBackendOps` のいずれであっても同一コードで動作する。

- 既存 `run` の本体を `run_impl(graph, feeds, dev_ops: Option<&dyn BackendOps>, report: Option<&mut DispatchReport>)` へ抽出し、
  `run` は `dev_ops = None` で呼ぶ（**挙動 bit 不変**。イシュー #2077 実装計画
  契約 (a)）。
- `run_with_ops` は `dev_ops = Some(ops)`。
- `run_with_ops_report` はさらに [`DispatchReport`]（`device_nodes`／
  `host_nodes`。ノード名ベースの可観測点）も返す。テスト・診断専用の内部
  クレート限定型で facade へは公開しない。
- device 側実行ヘルパは `crates/onnx-interop/src/onnx/interp_device.rs`
  （非公開モジュール）に集約した。

## 2. フォールバック規律（A08: 判定迂回経路を作らない）

各 device 実行ヘルパ（`interp_device::device_*`）は `try_device` を通して
`BackendOps` メソッドを 1 回呼び出し、結果を 3 分岐で扱う:

| 結果 | 扱い |
|---|---|
| `Ok(t)` | device 実行の結果を採用（ホスト実装を呼ばない） |
| `Err(BackendError::Unsupported(_))` | `Ok(None)`。呼び出し元（`compute_*`）が既存のホスト実装（`ops::*`）へフォールバックする |
| `Err(BackendError::ShapeMismatch(_))` | 同上（`Ok(None)`。§2.1 参照） |
| それ以外（`CudaUnavailable`／`DeviceAllocationFailed`／`KernelLaunchFailed`／`TransferFailed`／`DeviceMismatch`／`DeviceUnavailable` 等） | [`InterpError::Backend { node, message }`] として伝播。**ホストへの黙示フォールバックはしない**（GPU 故障を隠蔽しない） |

### 2.1 `ShapeMismatch` をフォールバック対象に含める判断

各 `device_*` は呼び出し前に必要最小限の形状検査（rank・空入力・LayerNorm の
単一正規化軸等）を行うが、境界ケースの完全な検証はホスト実装（`ops::*`）が
既に持つロジックと二重管理しない。したがって device 側カーネルが返す
shape 起因のエラーもホストへ委ねる設計とした。これにより、同じ不正入力に
対して opt-in ON／OFF で異なるエラー型（`InterpError::Backend` vs
`InterpError::Op(OpError::GemmDimMismatch)` 等）が返る事故を避ける。

## 3. op ごとの結線（実装済み・f32 のみ）

| ONNX op | device 呼び出し | 備考 |
|---|---|---|
| `MatMul`（rank 2×2） | `BackendOps::gemm_fp32_strict` | |
| `MatMul`（rank≥3 を含む） | `BackendOps::gemm_batched_fp32_strict` | 既定実装が NumPy 互換バッチブロードキャストを自前で正規化するため事前正規化不要 |
| `MatMul`（1-D オペランドを含む） | 常にホスト | ONNX の軸挿入・除去セマンティクスは `gemm_batched_fp32_strict` の契約に無い |
| `Gemm` | `BackendOps::gemm_fp32_strict` | `transA`／`transB` は `transpose(0,1).contiguous()` で実体化してから渡す。`alpha`・`beta*C` はホスト側で自前適用（GPU epilogue 融合は使わない） |
| `Add`／`Mul` | `BackendOps::add`／`mul` | 両メソッドは既に NumPy 互換ブロードキャストへ対応済みのため事前 `broadcast_with` は不要 |
| `Div` | `BackendOps::scalar_binary(ScalarBinaryOp::Div, ..)` | |
| `Sqrt` | `BackendOps::scalar_unary(ScalarUnaryOp::Sqrt, ..)` | |
| `Relu` | `BackendOps::scalar_unary(ScalarUnaryOp::Relu, ..)` | **`BackendOps::relu`（NaN を 0 に潰す）は使わない**。ONNX `Relu`・`ScalarUnaryOp::Relu` はいずれも NaN を伝播する契約のため |
| `Sigmoid` | `BackendOps::scalar_unary(ScalarUnaryOp::Sigmoid, ..)` | |
| `Softmax` | `BackendOps::softmax(x, dim)` | 最終軸限定は `BackendOps::softmax` 自身の既定契約（非最終軸は `Unsupported`）に委ねる。本イシューでは二重に軸判定しない |
| `LayerNormalization` | `BackendOps::layer_norm(x, Some(scale), bias, eps)` | `BackendOps::layer_norm` は最終軸 1 軸のみを正規化集合とする契約だが ONNX の `axis` は複数軸にまたがりうるため、正規化後の `axis == rank - 1` のときのみ device を試みる（`interp_device::device_layer_norm` の必須ガード） |

実装時点（2026-09-22）で CUDA・Metal の `scalar_unary` は `Sqrt` のみ共通実装
済み（CUDA は `Sqrt`／`Clamp`、Metal は `Sqrt`＋超越関数系。`Relu`／`Sigmoid`
は両バックエンドとも未実装）。`scalar_binary` の `Div` は両バックエンドとも
実装済み。したがって現時点で実際に GPU カーネルへ到達しうるのは
**MatMul／Gemm／Add／Mul／Div／Sqrt／Softmax（最終軸）／LayerNormalization
（最終軸）** であり、Relu／Sigmoid は結線のみ済み（現状は常にホストへ
フォールバック。将来 `#1635`／`#1636` 系でカーネルが実装されれば自動的に
到達する）。

要素数 0 のテンソル（`m`／`n`／`k` のいずれかが 0 の MatMul／Gemm・空 shape
の Add／Mul／Div／Sqrt・行数 0 の LayerNorm・空 Softmax）は device へ渡さず
常にホスト実装へ回す（ホスト側は空入力を正しく扱えるが、GPU カーネルが
`Unsupported` 以外のエラーを返しうるため、opt-in 時に偽の `Backend` エラーが
伝播する事故を防ぐ）。

## 4. facade opt-in API

`crates/facade/src/lib.rs` に既存 opt-in 群（`set_cuda_tf32_gemm_enabled` 等）
と同型の 4 関数を追加した:

- `set_cuda_onnx_gpu_execution_enabled(bool)` / `cuda_onnx_gpu_execution_enabled() -> bool`
- `set_metal_onnx_gpu_execution_enabled(bool)` / `metal_onnx_gpu_execution_enabled() -> bool`（`#[cfg(target_os = "macos")]`）

状態はプロセスワイドの `AtomicBool`（`crates/facade/src/interop/onnx.rs` の
`pub(crate)` static。`SeqCst`）で、`crate::interop::onnx` の `pub(crate)`
setter/getter へ薄く委譲する。`OnnxModel::run` は次の順で分岐する:

1. CUDA フラグ ON → `crate::resolve_ops(Device::Cuda(0))`（既存 private 関数。
   ordinal は 0 固定）→ `Err` は `OnnxError::Execution` として fail-closed
   （CPU への黙示フォールバックなし）→ `Ok` なら `interp::run_with_ops`
2. （macOS のみ）Metal フラグ ON → `resolve_ops(Device::Metal)` → 同上
3. 両方 OFF → 従来どおり `interp::run`（bit 不変）

両フラグが ON の場合は CUDA を優先する（評価順固定）。

## 5. 承認事項（列挙。安全側の仮置き値）

1. **GPU 実行の既定**: opt-in（既定 OFF・bit 不変）。「既定 ON」への変更は
   承認後の別 PR。
2. **facade 公開面の拡張**: 上記 4 関数のみ（`OnnxError` の variant 追加・
   `DispatchReport` の facade 公開は行わない）。CUDA ordinal は 0 固定。
3. 両フラグ ON 時の評価順 CUDA → Metal 固定。
4. ONNX MatMul／Gemm を `gemm_fp32_strict`（TF32 opt-in フラグ非参照）へ
   結線する。`gemm`（TF32 opt-in 追従）へ切り替える場合は
   `docs/cuda-tf32-optin-api-decision.md` の適用範囲改訂とセットで承認が
   必要。
5. parity 契約（§6）の解釈: GPU vs ホスト素朴実装は REQ-2 複合判定、bit 一致
   は「OFF 不変」「手動合成との一致」「facade vs 内部直接呼び出し」に限定。
6. dispatcher の配置（`BackendOps` trait ではなく `onnx-interop::interp::
   run_with_ops`。§1）。
7. 実機 parity の未実測分の申し送り先 `docs/perf/logs/onnx-gpu-execution-2077/`。

## 6. parity 契約

| 契約 | 判定 | 検証場所 |
|---|---|---|
| (a) opt-in OFF の `OnnxModel::run` は導入前と同一 | bit 完全一致 | CI（`crates/facade/tests/interop_onnx_gpu_execution_optin.rs`） |
| (b) `run_with_ops(ops)` の出力は、同じ `ops` を手動で合成した結果と同一 | bit 完全一致 | CI（`crates/onnx-interop/tests/onnx_interp_backend_dispatch.rs`。`CpuBackendOps`） |
| (c) GPU 実行（ON）とホスト実行（OFF）の出力 | REQ-2 統一複合判定 | CI（`CpuBackendOps` で代替）・`#[ignore]`（`interop_onnx_gpu_execution_parity.rs`。CUDA／Metal 実機） |
| (d) facade 経由（ON）と内部クレート `run_with_ops` 直接呼び出し | bit 完全一致（薄いラッパー） | 構造上自明（`OnnxModel::run` は `interp::run_with_ops` を 1 段委譲するのみ） |

GPU カーネルとホスト素朴実装（逐次 `mul_add` ループ）の bit 一致は要求しない
（`crates/onnx-interop/tests/cpu_row_kernel_naive_parity.rs` が示すとおり、同じ
CPU 上でも本番カーネルとホスト参照実装は結合順序が異なり bit 一致しない）。

## 7. 実機実測（申し送り）

CUDA・Metal 実機での §6 契約 (c) の実測は未実施。手順・保存ログの一覧は
`docs/perf/logs/onnx-gpu-execution-2077/README.md` を参照。

## 8. スコープ外（`.claude/rules/out-of-scope-tracking.md`）

- **Conv／BatchNorm／Pool 等の GPU 結線**: `interp` 自体が未対応（22 op 外）。
  import 対応は既存 open イシュー #2185／#2199／#2200 が担うため、それらの
  マージ後に本 dispatcher へ結線する後続項目として記録する（新規起票は
  ユーザー承認後）。
- Relu／Sigmoid／Exp／Tanh の GPU `scalar_unary` カーネル実装（`backend-cuda`／
  `backend-metal` 側の未実装 kind。既存の #1635／#1636 系の分担）。
- デバイス常駐チェーン（op 間の H2D／D2H 削減・`DeviceBuffer` 連鎖）・`Gemm`
  の `gemm_bias_act` 融合・ホスト側 op fusion。
- `I64`／`Bool`／`F16` 経路の device 実行（`TypedOps<f16>` 経由の f16 MatMul
  等）。
- `Device` 単位（ordinal 指定）設定・`run_on(Device)` API・`DispatchReport`
  の facade 公開。
- ONNX 仕様拡張・GraphProto 検証強化。
- autograd 接続（#2078）。
