# 学習ステップ常駐化の内部部品の設計決定記録

イシュー #3217（親 #3216）。学習 1 step を 1 同期のチェーン（#3248）へ近づけるために必要な内部部品のうち、内部 trait `BackendOps`（`crates/tensor-core/src/backend_ops.rs`）へのデフォルトメソッド追加だけで済むものを、実装の前に固める。本記録はコード変更を伴わない（古いコメントの訂正を除く）。後続の実装 issue（#3218〜#3223）は本記録のシグネチャに従う。

## 1. 目的・範囲・非範囲

- 背景: 学習 1 step の同期は Mac mini の Metal で wait 7 回、GB10 の CUDA で 8 回（実測。`docs/perf/train-step-phase-breakdown.md` §18.4・§19.6）。比較相手（candle・burn・PyTorch）は 1 回。現行の backward はホスト Tensor 専用（`crates/autodiff/src/backward.rs`）で、層ごとに同期が入る。
- 範囲: `BackendOps` への次の 4 種の追加だけ。
  - デバイス入出力の MSE（forward・backward）
  - ReLU マスク適用の常駐版
  - dW の GEMM がデバイス入力を受ける入口
  - L2 の d_input をデバイス入出力で計算する版
- 非範囲:
  - `linear_backward_device` のような一括の backward 入口、`backward_impl` でデバイスに勾配を運ぶ機構は作らない（公開形を伴うため #3249・#3251・#3252）。`backward_impl` はホスト専用のまま。
  - CPU の override は #3253 の判断に委ねる。
- 出典: `docs/backend-metal-command-batching-design.md` §7.3.3 (b)（本記録はその部品分割にあたる）、`docs/inference-chain-single-sync-design.md`（推論チェーンでの `_tracked` 契約の前例）。

## 2. 公開 API に触れない根拠

- `tensor-core` は crates.io 公開済みだが、サポートされる公開面ではない内部クレート（`docs/compat-api-scope.md` §0）。
- facade は `BackendOps` を再エクスポートしない（`crates/facade/src/lib.rs` の REQ-12 節）。デフォルト付きメソッドと `#[non_exhaustive]` enum の追加は SemVer 上も非破壊。
- facade の公開面の検査（`crates/facade/tests/api_surface.rs`）に影響しない。tolerance・FMA 契約・parity baseline も変えない。

## 3. 追加シグネチャ一覧（推奨形・未実装）

以下の名前は推奨形である。実装 issue で変える場合は、その PR で本記録を同時に更新する。

共通方針:

- 非 tracked 版の既定は `BackendError::Unsupported` を返す fail-safe とする（`linear_forward_device`・`binary_elementwise_device` と同じ）。ホストへ黙ってフォールバックすることを禁止する（A08）。
- `_tracked` 版の既定は `token` を無視して非 tracked 版へ委譲する（`linear_forward_device_tracked` と同型）。
- ROCm と未 override のバックエンドは既定（Unsupported）を継承する。
- 入力検証: `out_offset + m*n`・`bias_offset + n`・ビューの `offset + numel` は `checked_mul`／`checked_add` で、driver／encode 呼び出しより前に検査する。shape・rank・device の不一致は型付きエラーにする。カーネル側の手動境界チェックは省略しない（REQ-8。`coding-rust.md`）。

新設する型（`tensor-core`・`backend_ops.rs`）:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMatrixLayout { RowMajor, Transposed }
```

意味: ビューの `shape()` は格納形（行優先）を表す。`Transposed` は「論理オペランドが格納行列の転置」であることを示す。Metal の `MatrixLayout { transposed }` と CUDA の NT／TN 判定に 1 対 1 で写る。

| # | 部品 | シグネチャ（`_tracked` 版は末尾に `token: &DispatchFailureCell` を足した同名 `_tracked`） | 契約 |
|---|---|---|---|
| 1 | MSE forward | `fn mse_loss_device(&self, pred: &DeviceBuffer<f32>, target: &DeviceBuffer<f32>, reduction: MseReduction) -> Result<DeviceBuffer<f32>, BackendError>` | shape `[]`（numel 1）のデバイス常駐スカラーを返す。`numel == 0` は 0.0（既存 `mse_loss` 契約と同じ） |
| 2 | MSE backward | `fn mse_loss_backward_device(&self, pred: &DeviceBuffer<f32>, target: &DeviceBuffer<f32>, scale: f32) -> Result<DeviceBuffer<f32>, BackendError>` | `scale` はホストの `f32` 引数。チェーンでは loss の seed が 1.0 なので `mse_loss_scale(1.0, n, reduction)` をホストで計算でき D2H は要らない（現行 `grad.rs` の `Op::MseLoss` と同じ式で bit も同一）。戻り値は `dPred` のみ。`dTarget` は作らない。**実装での精密化（#3219、1・2 行目共通）**: Metal は `MetalMse::encode_mse_*_tracked`（待たず読み戻さない encode 本体）をホスト入出力版と共有し、同じカーネル・`num_tg`・factor・dispatch 形状で動くため bit 一致する。非 tracked 版はローカル cell で encode → `synchronize` → token 検査、`_tracked` 版は encode のみ（wait 0）。出力は `alloc_zeroed` で確保し pool 再利用では同期しない（OOM フォールバックのみ例外）。u32 上限超過・shape／device 不一致は型付きエラー、`numel == 0` は dispatch せず forward が 0.0・backward が空 shape。**実装での精密化（#3222・CUDA、1・2 行目共通）**: ホスト入出力版と同じ起動ヘルパ（`mse.rs` の `launch_partial_finalize`／`launch_backward`）を共有するため、カーネル・起動構成・縮約順序が同一で bit 一致する。forward の `partial` は `CudaMse` の永続ワークスペース（`MSE_MAX_BLOCKS` 要素。ロックは partial・finalize の 2 起動をまたぎ、プールへ返さない。readback の同期が無いとプール返却が finalize の読み出しと race するため）。出力は `alloc_zeroed`。非 tracked・tracked とも同期しない（ストリーム順序契約。`_tracked` は `token` を使わない明示 override）。**新規 `unsafe` なし**（`mse.rs` の `unsafe {` 数は変更前後で 3 のまま。ホスト版の 3 か所をヘルパへ移設）。`i32::MAX` 超過・shape／device 不一致は driver 前に型付きエラー、`numel == 0` は poison・世代検査のあと forward が 0.0・backward が空 shape |
| 3 | ReLU マスク適用 | `fn relu_backward_mask_device(&self, g: DeviceBufferView<'_>, g_layout: DeviceMatrixLayout, mask_src: DeviceBufferView<'_>) -> Result<DeviceBuffer<f32>, BackendError>` | **乗算ではなく選択**: `out[i] = if mask_src[i] > 0.0 { g[i] } else { +0.0 }`。参照は `grad.rs::elementwise_mul_mask(.., \|v\| v > 0.0)`。mask が NaN なら偽で +0.0。g の NaN／inf は mask が真なら保存する。`g * (m > 0)` 形の乗算は NaN×0 や −x×0（−0.0）で bit がずれるので禁止。出力は論理形の行優先、`mask_src` は行優先。**実装での精密化（#3218）**: `g.shape()` は rank 2 限定（`RankMismatch`）、`mask_src.shape()` は論理形と完全一致（`ShapeMismatch`）。Metal はカーネル `ew_relu_mask_select_f32`（`g`／`out` を `uint` としてビットコピーし NaN payload・−0.0 を保存）で、view 起点は `setBuffer` のバイトオフセットではなく `constant uint&` 引数で渡す（新規 `unsafe` なし）。**#3218 reopen**: Metal は float 比較で正の非正規化数を 0 へ flush する（FTZ）ため、mask を `uint` で読み `bits != 0 && bits <= 0x7F800000` で `> 0.0` と同値に判定する。そのため `offset + numel <= u32::MAX` を要求し、超過は `InvalidArgument`（fail-closed）。非 tracked 版はローカル cell で encode → `synchronize` → token 検査、`_tracked` 版は encode-only。**実装での精密化（#3221・CUDA）**: カーネル `ew_relu_mask_select_f32`（`__float_as_uint`／`__uint_as_float` でビットコピーし NaN payload・−0.0 を保存。`idx < numel` と `gi < numel` の手動境界検査を維持）。view の起点は `CudaStorage::view` のスライスで渡すため offset 引数と u32 上限は無く、`rows`・`cols`・`numel` が `i32` に収まることだけを要求する（超過は `InvalidArgument`）。範囲外で panic する `view` の前に `checked_add` で検査する。`launch_binary_resident` の `unsafe` ブロックを非公開の汎用プリミティブ `launch_binary_resident_with`（末尾に `int` 引数を足せる）へ移して再利用し、**新規 `unsafe` はない**（`elementwise.rs` の `unsafe {` 数は変更前後で同数）。非 tracked・tracked とも同期しない（ストリーム順序契約。`_tracked` は `token` を使わない明示 override。起動エラーは次の同期点で poison／世代検査から観測される）。ftz は NVRTC 既定で無効のため、正の非正規化数の mask は CPU と同じく真になる（実機の確認は #3224。GB10 で C1〈`1e-40` の正の非正規化数マスクを含む〉が 5 run 全 pass し、真になることを確認済み〈`45347ae0a`・2026-10-23〉。記録先は `docs/perf/logs/train-step-resident-parts-parity-3224/README.md`） |
| 4 | dW（bias 縮約つき） | `fn gemm_fp32_strict_into_device(&self, a: DeviceBufferView<'_>, a_layout: DeviceMatrixLayout, b: DeviceBufferView<'_>, b_layout: DeviceMatrixLayout, out: &mut DeviceBuffer<f32>, out_offset: usize, bias: Option<(usize, usize)>) -> Result<bool, BackendError>` | dW = xᵀ @ g では `a` = x の格納（Transposed）、`b` = g（RowMajor）。上書き契約、`out_offset`・`bias` の範囲検査、戻り値 `bias_filled` の意味は既存の `gemm_fp32_strict_into_with_bias_reduce_tracked` と同じ（`Err` のときは weight・bias のどちらも書かれていない）。**実装での精密化（#3220、4・5 行目共通）**: Metal はホスト版と同じ encode 本体（dW は `encode_weight_and_bias_grad_with_offsets`、d_input は `encode_strided_bias_act_prepared_with_c_offset`）と同じ `MatrixLayout`（`layout::matrix_layout_from_device_view`。格納 `[r, c]` の `RowMajor` は `{r, c, ld: c}`、`Transposed` は `{c, r, ld: c, transposed}` で、ホスト版の `classify_2d` と一致）を使うため bit 一致する。view の offset は `setBuffer` のバイトオフセットで渡るので u32 上限は無い。dW は NN/TT（`a_layout == b_layout`）を `Unsupported` で拒否する（ホスト版は `gemm_fp32_strict`〈split-K を含む〉へ落ち、classic strided カーネルとの bit 一致を保証できない。ホスト版が `Unsupported` を返さない理由〈`resident_grad_capability` の汚染防止〉は、呼び出し元が #3251 のチェーンだけで dW が常に TN になる本入口には当てはまらない）。dW の 0 次元も、wait なしで正しく書く方法が無い（ホスト版は `upload_into` で同期する）ため `Unsupported`。検査（device・rank・内側次元・`validate_gemm_bias_write_ranges`）はすべて encode より前。bias 縮約の soft-f64 bit 一致契約は不変。非 tracked 版はローカル cell で encode → `synchronize` → token 検査、`_tracked` 版は encode のみ（wait 0）。**実装での精密化（#3223・CUDA、4・5 行目共通）**: ホスト版 `launch_tiled_f32_tn_into`／`_nt_into` と同じ手順（転置で実体化 → `select_tiled_f32_kernel(0, ..)` → bias 縮約〈TN は `b` を `outer=1, axis=k, inner=n`、NT は `b` の格納を `outer=n, axis=k, inner=1`〉）を、H2D・明示 `synchronize`・プール由来の転置バッファなしで積む（`gemm.rs` の `launch_gemm_dw_device`／`launch_gemm_dinput_device`）。TN は `a_layout = Transposed`・`b_layout = RowMajor`、NT は `a_layout = RowMajor`・`b_layout = Transposed`。NN/TT・0 次元・転置カーネル不可・転置の grid.y 上限超過は `Unsupported`（ホストへ黙示フォールバックしない = A08。0 次元は CUDA なら `zero_fill` で書けるが Metal 版と trait doc の一致を優先し、ホスト版の `k == 0` 時の同期は新入口から到達しない）。**直接 GEMM へ渡すオペランドの cp.async 整列**: `tiled_f32_kernel_kind` の整列ゲートは A の offset しか見ず、既存の呼び出し元は B を常に offset 0（新規 upload か転置で実体化したバッファ）で渡してきた。新入口は任意 offset の view を直接渡す（TN dW の `b`、NT dW の `a`、d_input の RowMajor `b`）ため、offset が 4 要素（16 バイト）の倍数でなければワークスペースの staging スロットへ `copy_from_dtod`（ストリーム順序・bit 不変）してから渡す。4 の倍数なら直接渡し、カーネル選択はホスト版の offset 0 と同一になる（基底確保は 256 バイト整列の前提）。`tiled_f32_kernel_kind` に B の offset を足して classic へ落とす案は、bit 一致の根拠が「ホスト版と同一カーネル」から「classic と pipeline の実測 bit 同一」へ変わり §4 の構造的条件を外れるため採らない。d_input の `w` はホスト版も実 offset を `a_offset` に渡すので staging しない。bias 縮約の元は元の device view をそのまま読む（縮約カーネルはスカラーロードで整列制約がない）。非 tracked・tracked とも同期しない（ストリーム順序契約。`_tracked` は `token` を使わない明示 override。起動エラーは次の同期点で poison／世代検査から観測される）。検査（device・rank・内側次元・`validate_gemm_bias_write_ranges`・各 view の `offset + numel`）はすべて driver 前で、`gemm_device_model.rs` の純関数と `ops.rs` の単体テストで GPU なしでも確認する |
| 5 | d_input | `fn gemm_resident_lhs_device(&self, w: DeviceBufferView<'_>, b: DeviceBufferView<'_>, b_layout: DeviceMatrixLayout) -> Result<DeviceBuffer<f32>, BackendError>` | d_inputᵀ = W @ gᵀ を `[k, m]` で返す（既存 `gemm_resident_lhs` と同じ式・同じ出力形。ただし入出力ともデバイス）。次の部品（前層のマスク）は `g_layout = Transposed` でこの出力を読む。**実装での精密化（#3220）**: `w` は常に `RowMajor`、`b_layout` は `RowMajor`／`Transposed` のどちらも可（ホスト版も全レイアウトで classic strided カーネル）。`p == 0 || r == 0` は空、`q == 0` は全 0 の `[p, r]` を dispatch・wait なしで返す。出力は `alloc_zeroed`（`static_metal_memory`）で確保する。同期・token の扱いは行 4 と同じ。**実装での精密化（#3223・CUDA）**: `b_layout = Transposed`（チェーンの標準。格納 `[r, q]`）はホスト版 NT 分岐と同じく `b` をワークスペースへ転置して `launch_tiled_f32_resident(w, w.offset(), b_std, ..)`、`RowMajor` はホスト版フォールバックと同じカーネルへ `b` を直接渡す（offset が 4 の倍数でなければ行 4 と同じ staging）。`p == 0 \|\| r == 0` は空・`q == 0` は全 0 の `[p, r]` で、dispatch せず `begin_driver_call` で poison・世代を検査したうえで返す |

## 4. bit 一致の基準

| 部品 | bit 一致の比較先 | CPU 参照との関係 |
|---|---|---|
| ReLU マスク | `grad.rs::elementwise_mul_mask`（純粋な選択） | 全バックエンドで bit 一致（NaN はクラス一致） |
| MSE fwd／bwd | 同じバックエンドの既存 `mse_loss`／`mse_loss_backward`（同一カーネル・同一縮約方式） | 既存どおり REQ-2 統一複合判定（縮約順序がバックエンドで違うため） |
| dW | 同じバックエンドの `gemm_fp32_strict_into_with_bias_reduce_tracked`（ホスト入力版） | GEMM は既存契約のまま。bias 縮約は既存の bit 一致契約（CUDA は `double`、Metal は soft-f64）のまま |
| d_input | 同じバックエンドの `gemm_resident_lhs` | 既存契約のまま |

- bit 同一を構造的に保つ条件:
  - ホスト版と同じカーネル・同じカーネル選択入力を使う。Metal は `dispatch_strided_bias_act_prepared` 系に同じ `MatrixLayout` を渡す。CUDA は NT／TN 判定、`transpose_smem_f32` による実体化、`select_tiled_f32_kernel` の選択入力（転置実体化先は offset 0）をホスト版と揃える。
  - 転置の実体化はコピーなので bit は変わらない。
  - stride 読みの別カーネルへ差し替えると REQ-2 判定にしか落ちない（`docs/matmul-vjp-zero-copy-decision.md` §4.4）ため採らない。
- 検証方法:
  - CI で実行できる単体テスト: 既定メソッドの `Unsupported`・`_tracked` の委譲を `backend_ops.rs` の既存テスト（`linear_forward_device_default_is_unsupported` 等）と同型で書く。
  - 実機テスト（`#[ignore]`）は `to_bits()` 一致で判定し、NaN は両者 `is_nan()` のクラス一致とする。形状は m=64 の MLP 形状と端数形状、入力は NaN／inf を含める。#3224 で Mac mini と GB10 について RULE.txt・5 run・`.rev-stamp` つきで記録する。記録先は `docs/perf/logs/train-step-resident-parts-parity-3224/`（規則と手順は固定済み。Mac mini は 2026-10-23 に `e62cce21b` で実測済みで Metal M1〜M5 x 5 run 全 pass〈開発機試走で出た正の非正規化数マスクの不一致〈FTZ〉は #3218 でビット判定へ修正済み〉。GB10 は 2026-10-23 に `45347ae0a` で実測し、C1〜C4 x 5 run は全 pass、C5 は対照 assert〈ホスト版が D2H を数える前提〉が readback 先読み既定 ON〈#3227〉と食い違い 5 run とも 2/4 fail。新版の判定 assert は通過。詳細は記録先 README「不一致」）。
  - tolerance・baseline は変えない。

## 5. 失敗トークン（`_tracked`）

- Metal:
  - encode-only の dispatch は `MetalContext::encode` と同じロック区間で `token` を登録する（`sgd_step_device_tracked`・`linear_forward_device_tracked` と同じ）。
  - 新設部品の非 tracked 版は、ローカルの `DispatchFailureCell` で encode → `synchronize` → `token.is_set()` 検査を行う同期する安全版とする（`docs/backend-metal-command-batching-design.md` §7.5.1）。encode-only は `_tracked` 版だけにし、トークンなし encode-only の入口（fail-open）を新たに作らない。
- CUDA:
  - ストリーム順序契約により `token` は使わない。トレイト doc の更新漏れを防ぐため、機能的に同一な明示 override を置く（`gemm_fp32_strict_into_tracked` と同じ判断）。
  - 内部同期をなくすと launch エラーは次の同期点で既存の poison／世代検査（`context_cache`）から観測される。これは `linear_forward_device` と同じ契約。
- 呼び出し元: チェーン（#3251・#3252）は常に `_tracked` 版を呼び、`DeviceParamStore` の `failure_token` を渡す。`step()` 冒頭の検査と `check_not_poisoned` による遅延検出がそのまま効く。

## 6. CUDA の dW 内の明示同期（設計判断 A）

- 現状: `CudaGemm` の `launch_tiled_f32_nt_into`／`launch_tiled_f32_tn_into`（`crates/backend-cuda/src/gemm.rs`）は、`transpose_to_pooled` の戻り値 `PooledCudaHandle` を関数末尾で drop する。drop で即座に（ストリーム順序なしで）プールへ返すため、GEMM がまだ読んでいる間に再利用される race を避ける目的で、関数内で `stream.synchronize()` している。
- 比較した案:
  - (i) 現状維持: 1 同期の目標に反する。
  - (ii) `launch_tiled_f32_resident_nt` と同じくハンドルを呼び出し元へ返し次の同期点まで保持する: `BackendOps` のシグネチャで CUDA 固有の型を運べないため不採用。
  - (iii) **永続ワークスペース（推奨）**: ordinal 単位のオブジェクトに grow-only の転置用ワークスペースを持たせ、同じストリーム上で再利用する。次の呼び出しの転置書き込みは前の GEMM の読み出しよりストリーム順で後になるので、ホスト同期なしで race しない。
  - (iv) stride 読みの NT／TN カーネルを新設して転置を不要にする: bit 一致を保証できないため不採用。
- (iii) の成立条件（#3223 で確認する）:
  - そのオブジェクトの launch が単一ストリームに限られる。
  - スレッド間共有では既存のロックの下で使う。
  - ワークスペースをプール（`PooledCudaHandle`）へ返さない。拡張時の旧領域はストリーム順序つきの解放（cudarc の `CudaSlice` の drop）に任せる。
  - 確保は safe API（例: `alloc_zeros`）で済み、新規 `unsafe` が要らない。
- 条件のどれかが成り立たなければ明示同期を残し、理由を記録する（#3223 の受け入れ条件どおり）。
- **#3223 の採用結果: 案 (iii) が成立した**（`CudaGemm::device_workspace`。`CudaMse::partial_workspace` と同型）。条件の確認:
  1. 単一ストリーム: `CudaGemm` の起動は `self.stream` に限られる。`cached_gemm` は `ContextKey` ごとに 1 インスタンス（`device.stream()`）で、`cached_gemm_auto`／MMA 系は別インスタンスでワークスペースを共有しない。bias 縮約は `validate_bias_target` の `Arc::ptr_eq` で同一ストリームを担保済み。
  2. スレッド間共有: ワークスペースの `Mutex` を、staging コピー・転置・GEMM・bias 縮約の起動をすべて積み終えるまで保持する（poison は型付きエラー）。
  3. プールへ返さない: 確保は safe API の `stream.alloc_zeros`（プール非経由）で、grow-only。
  4. 新規 `unsafe` なし（§8 項目 1 参照）。
- 旧領域の解放: cudarc 0.19.8 の `CudaSlice::Drop` は `has_async_alloc` が真ならストリーム順序つきの `free_async`（GB10 は真）、偽なら `stream.synchronize()` のあと `free_sync` を行う。したがって**同期が出うるのは非 async-alloc 環境でワークスペースを拡張したときだけ**で、定常状態のチェーンでは確保ゼロ・同期ゼロ（受け入れ条件「残す同期の理由を記録」への回答）。graph capture 中に拡張が要る場合は確保がグラフに記録されてしまうため `Unsupported` で fail-closed にする（拡張が不要なら capture 中でも積める）。
- ホスト版 `launch_tiled_f32_{nt,tn}_into` の明示 `synchronize()` は**変更しない**（プール由来の転置バッファと H2D バッファを関数内で drop するため必要。ワークスペース化でこれを外す最適化は #3223 の範囲外）。ホスト版の同期・H2D は診断カウンタ（`diag_count_stream_sync`／`diag_count_h2d`）で計数し、新入口が 0 であることとの対比を `gemm_device_input_diag_tests_3223.rs` で実機確認する。
- `k == 0` 分岐の `zero_fill` 後の `synchronize()` もストリーム順序の memset なので同じ理由で外せる。d_input の device 版の転置実体化にも同じワークスペースを使う。

## 7. 古い記述の訂正記録

「CUDA／Metal は `gemm_fp32_strict_into` 未対応」という記述は #1212・#1479 時点のもので、現在は実態と合わない。実装の所在:

| バックエンド | 実装 |
|---|---|
| CPU | `backend-cpu::ops` の `gemm_fp32_strict_into` |
| CUDA | `backend-cuda::ops` の `gemm_fp32_strict_into`（#1559） |
| Metal | `backend-metal::ops` の `gemm_fp32_strict_into_impl`（#1555） |

- facade テスト `crates/facade/tests/device_param_store_backend_parity.rs::assert_grad_readout_contract` の期待も CUDA・Metal で `Some` になっている。
- 訂正した箇所（いずれもコメント・文書のみでロジック不変）:
  - `crates/autodiff/src/optim/device_store.rs` の `step()` ② 更新フェーズ・graph capture 分岐・`fill_resident_weight_grad` の probe 理由・テスト `resident_grads_to_host_is_unsupported_on_backend_without_gemm_into` の doc。`any_resident` はバックエンド種別ではなく、今回の backward で resident 書き込みが 1 件でも成功したかで決まる。`false` になるのは `Op::LinearResident` を経由しないモデルや、AMP・distributed の合成 `Gradients`（`resident_fingerprint = None`）など。
  - `docs/backend-cuda-graph-step-capture-design.md` の F2 行・スコープ外整理。F2 の結論（capture は update 区間に限る）は不変。§5.1 は 2026-09-09 の実測履歴のため書き換えず、時点注記を追加した。

## 8. 所有者確認が要る項目

1. CUDA の新規 `unsafe` ブロック（#3221・#3223）。**#3221（ReLU マスク）は新規 `unsafe` なしで成立した**（上記 §3 行 3）。**#3222（MSE）も新規 `unsafe` なしで成立した**（上記 §3 行 1・2）。**#3223（dW／d_input の GEMM）も新規 `unsafe` なしで成立した**（`crates/backend-cuda/src/{gemm,ops,memory,reduce}.rs` の `unsafe {` 数は変更前後で 28／0／5／12 のまま。`transpose_to_pooled` の起動ブロックを `CudaArg`／`CudaArgMut` を受ける `launch_transpose_f32_into` へ移設して再利用し、`launch_bias_reduce` の `src` を `CudaArg` へ一般化した。移設した `unsafe` 周辺は security-auditor の監査対象）。推奨形は、既存の launch ヘルパ（unary／binary resident 系・`transpose_smem_f32`・tiled GEMM の launch）の引数をホストスライスからデバイスビューへ一般化して再利用し、`unsafe` ブロックを増やさないこと。マスク選択用の新カーネルも既存の汎用 launch ヘルパへ op を足す形を第一候補にする。新しい `unsafe` が避けられないと分かった時点で実装を止め、security-auditor の監査記録をつけて所有者へ戻す。
2. `dTarget` をデバイスで出す必要が生じた場合は、公開形の問題として #3249 で扱う。
3. 範囲外の追跡: `device_store.rs` の `step_adam_impl` 内コメントは `adam_step_device` を「CUDA／Metal の既定実装（未実装のまま維持）」と書くが、実際は両バックエンドとも実装済み。主題が別のため本 issue では直さない。起票はユーザー承認後（`out-of-scope-tracking.md`）。

## 9. 補足

- 新規 `unsafe`・依存追加・公開 API の追加は本 issue にない。
- ROCm は既定の Unsupported を継承するので影響なし。
