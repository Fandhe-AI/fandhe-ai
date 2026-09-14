> 調査日 2026-09-12・対象 HEAD `097bff19`（#1556 Metal resident grad staging
> 含む）。出典: 低レイヤー診断 artifact
> （https://claude.ai/code/artifact/4e107064-a190-4861-897d-3dce44d05428）
> §2「ギャップ根拠」節。イシューツリー起票（`p1-docs`）に伴い本ドキュメントへ
> 取り込む。内容は取り込み元から変更していない（下記「#1621 追記」節を除く）。
>
> **#1621 追記（線形代数）**: 本調査時点で §2.6「線形代数」に inv／solve／
> det／qr／cholesky／svd／matrix_norm の行は存在しなかった（`matmul`／`bmm`／
> `einsum`／`transpose` の 4 行のみ）。イシュー #1621 でこれらを実装した
> （`Var::inv`／`solve`／`det`／`cholesky`／`qr`／`svd`／`matrix_norm`。
> `docs/autodiff-linalg-design.md` 参照）。§0「Var の演算メソッドは 15 個の
> み」・§1.5 の一覧・§1.7 の `BackendOps` 演算 API 一覧・§2.6 の表へ、
> 実装結果を反映する追記を各該当箇所に加える（取り込み元の記述自体は残し、
> 追記であることを明示する）。

# fandhe-ai 公開面（facade）と PyTorch/TensorFlow の機能ギャップ表

調査日: 2026-09-12。対象は `crates/facade/src/`（`fandhe_ai` crate）から到達可能な公開 API のみ。
実装コードを読んで確定した事実のみを「あり」とする（ドキュメント上の意図・将来計画は「なし」扱い）。

## 0. 総括

fandhe-ai の公開面は現時点で **MLP（全結合＋3 活性化＋MSE/CrossEntropy 損失＋SGD/AdamW）専用**の
薄いラッパーである。テンソルの汎用演算（indexing・slice・cat・stack・任意 elementwise・
縮約の複数軸同時指定・`keepdim`・除算・べき乗等）は `Var`（autodiff 公開型）レベルにすら存在せず
（単一軸の `dim: Option<usize>` 指定〈`sum`/`max`〉自体はあるが複数軸同時指定・`keepdim` はない）、
`Tensor<T>`（tensor-core 型）レベルでも shape 変形（transpose/permute/reshape/broadcast/narrow）
止まりで、CNN・RNN・Attention を組むための演算プリミティブは構造的に欠落している。

- Var の演算メソッドは **15 個のみ**（`crates/autodiff/src/var.rs`）:
  `matmul`・`matmul_checksum`・`add`・`mul`・`sum`・`max`・`mse_loss`・`mse_loss_with`・
  `cross_entropy_loss`・`relu`・`exp`・`tanh`・`sigmoid`・`reshape`・`transpose`。
  減算（`sub`）・除算（`div`）・べき乗（`pow`）・平方根（`sqrt`）・対数（`log`）・
  比較演算・`softmax`（Var メソッドとしては存在しない）は **一切ない**。
  **#1621 追記**: 上記 15 個に加え、線形代数 7 個（`inv`・`solve`・`det`・
  `cholesky`・`qr`・`svd`・`matrix_norm`）を実装した（§1.5・§1.7・§2.6 参照）。
  減算・除算・べき乗等の欠落は本追記の対象外のまま不変。
- `nn::Module` 実装は `Linear`・`Relu`・`Sigmoid`・`Tanh` の **4 種のみ**
  （`crates/autodiff/src/nn/module.rs:105,171,193,212`）。Conv・BatchNorm・LayerNorm・
  RMSNorm・Dropout・Embedding・Attention・RNN/LSTM/GRU・Pooling は **一切ない**。
- 損失は `MseLoss`・`CrossEntropyLoss` の 2 種のみ（`crates/autodiff/src/nn/loss.rs`）。
- optimizer は `Sgd`（momentum・dampening・weight_decay・nesterov 対応）・`AdamW` の 2 種
  （`crates/facade/src/optim.rs:78-81`）。Adam（無 weight-decay 版）・RMSprop・Adagrad・
  LAMB は **ない**。
- scheduler は `ConstantLr`・`StepLr` の 2 種のみ（`crates/autodiff/src/nn/optim/lr_scheduler.rs`）。
  Cosine annealing・ExponentialLR・ReduceLROnPlateau・OneCycle は **ない**。
- dtype は `f32` 演算専用。`Tensor<T>` は `Element` trait 経由で `f32`/`f64`/`i32`/`f16`/`i64`/`bool`
  をジェネリックに保持できる（`crates/tensor-core/src/element.rs`）が、
  **算術カーネル dispatch（`BackendOps`）は `f32` 固定**（同ファイル冒頭コメント「`i64`/`bool` は
  算術・backend dispatch 対象には含めない」）。`facade` の `Var`/`Tensor<f32>` は事実上 f32 のみ。
- device は CPU／CUDA（単一 ordinal）／Metal の 3 種（`crates/tensor-core/src/device.rs`）。
  多 GPU 学習（DDP・model/tensor parallel）は **ない**。
- データローディング（`DataLoader`/`Dataset`）・`state_dict` 相当の汎用シリアライズ・
  ONNX export・量子化・JIT/`compile()` は facade に **ない**（ONNX の読み込みは
  `onnx-interop` にあるが非公開クレート）。
- メモリ管理 API（`release_cached_memory`・`memory_pool_stats`）・デバイス常駐パラメータ更新
  （`DeviceParamStore`）・CUDA Graph capture opt-in・TF32/split-K opt-in 等、
  **性能インフラ層の公開 API は PyTorch/TF より充実**している箇所もある（下記 1 節参照）。

---

## 1. fandhe-ai 公開面の実装済み機能一覧

### 1.1 composition root（`crates/facade/src/lib.rs`）

| 機能 | 場所 |
|------|------|
| `tape()`（既定 CPU バックエンドで `Tape` 構築） | `crates/facade/src/lib.rs:271-275` |
| `tape_for(Device)`（CPU/CUDA/Metal を明示選択） | `crates/facade/src/lib.rs:283-286` |
| `Device`・`BackendError`・`PoolStats`・`Tensor` 再エクスポート | `crates/facade/src/lib.rs:122` |
| `release_cached_memory(Device)`（REQ-14 明示解放） | `crates/facade/src/lib.rs:334-336` |
| `memory_pool_stats(Device)`（プール統計スナップショット） | `crates/facade/src/lib.rs:345-347` |
| `set_cuda_tf32_gemm_enabled`/`cuda_tf32_gemm_enabled`（CUDA TF32 opt-in） | `crates/facade/src/lib.rs:364-372` |
| `set_cuda_gemm_precision`/`cuda_gemm_precision`（`Fp32Strict`/`Tf32`/`Tf32x3`） | `crates/facade/src/lib.rs:471-479` |
| `set_cuda_graph_step_enabled`/`cuda_graph_step_enabled`（CUDA Graph step capture opt-in） | `crates/facade/src/lib.rs:400-408` |
| `cuda_graph_step_mode`/`cuda_graph_step_stats`（診断） | `crates/facade/src/lib.rs:426-437` |
| `set_cuda_managed_memory_enabled`/`cuda_managed_memory_enabled`（managed memory opt-in） | `crates/facade/src/lib.rs:518-526` |
| `set_metal_split_k_gemm_enabled`/`metal_split_k_gemm_enabled`（Metal split-K opt-out。macOS 限定） | `crates/facade/src/lib.rs:556-565` |
| `Tape::reset`/`leaf_count`/`leaf`（tape 再利用。学習ループ最適化） | `crates/facade/src/lib.rs:169-184` |
| `Tape::step_device_param_store`/`backward_device_param_store`/`sync_device_param_store_to_host`/`resident_grads_to_host`/`param_grads_to_host`（デバイス常駐更新） | `crates/facade/src/lib.rs:194-262` |

### 1.2 `compat::array`（numpy `np.array` 慣習）

`Tensor<f32>` を 1-D（`Vec<f32>`/`&[f32]`/`[f32; N]`）・2-D（`Vec<Vec<f32>>`/`[[f32; N]; M]`）から生成。
jagged 2-D 入力は事前検証で拒否。`crates/facade/src/compat/array.rs:98-101`。

### 1.3 `compat::Sequential`（Keras `Sequential` 慣習）

| 機能 | 場所 |
|------|------|
| `add_linear`/`add_relu`/`add_sigmoid`/`add_tanh` | `crates/facade/src/compat/sequential.rs:116-144` |
| `forward`（学習用。外部 `Tape` 上、Linear→ReLU 融合結線あり） | 同 152-194 |
| `predict`（推論。tape 不要経路→フォールバックで tape 経路） | 同 221-304 |
| `bind`/`SequentialVars`（学習可能パラメータのテープ登録） | 同 318-331, 648-785 |
| `trainable_parameters`/`apply_parameters`（shape 保存・2-pass アトミック更新） | 同 339-462 |
| `init_device_param_store`/`forward_resident`/`predict_resident`（デバイス常駐パラメータ学習・推論） | 同 477-640 |

### 1.4 `optim`（`crates/facade/src/optim.rs`）

| 機能 | 場所 |
|------|------|
| `Sgd`/`SgdConfig`（momentum・dampening・weight_decay・nesterov） | `crates/autodiff/src/optim/sgd.rs:32-224` |
| `AdamW`/`AdamWConfig` | `crates/autodiff/src/nn/optim/adamw.rs:23-160` |
| `clip_grad_norm`/`global_grad_norm`/`ClipGradResult` | `crates/autodiff/src/nn/optim/clip.rs` |
| `clip_grad_value`（#1753・親 #1631。要素ごと `[-clip_value, clip_value]` クランプ） | `crates/autodiff/src/nn/optim/clip.rs` |
| `ConstantLr`/`StepLr`/`LrScheduler` | `crates/autodiff/src/nn/optim/lr_scheduler.rs` |

### 1.5 `Var` の演算メソッド一覧（`crates/autodiff/src/var.rs`）

| メソッド | 行 | 備考 |
|---------|-----|------|
| `value`/`to_tensor`/`host_view` | 98,107,135 | 借用ビュー読み出しは #1335 で追加 |
| `matmul` | 195 | GEMM。CPU BLIS／CUDA／Metal 各カーネルへ dispatch |
| `matmul_checksum` | 233 | デバイス側 f64 checksum 縮約（#1339） |
| `add`/`mul` | 339,362 | elementwise binary（融合対象） |
| `sum`/`max` | 382,402 | 縮約。`dim: Option<usize>` のみ（`amax` 均等分配なし。tie は最初の要素へ先勝ち） |
| `mse_loss`/`mse_loss_with` | 425,447 | mean/sum 縮約。単一融合カーネルあり |
| `cross_entropy_loss` | 508 | log-softmax 安定化込みの融合オペ |
| `relu`/`exp`/`tanh`/`sigmoid` | 556,568,580,600 | elementwise unary |
| `reshape`/`transpose` | 623,672 | view 系（#1080 で再計算方式・中間バッファなし） |
| `inv`/`solve`/`det`/`cholesky` | - | **#1621 追記**。線形代数（rank-2 限定）。CPU 実装先行・GPU は `Unsupported` フォールバック |
| `qr`/`svd` | - | **#1621 追記**。多出力（`QrVars`/`SvdVars`。テープは 1 ノード 1 出力のため出力ごとに別ノード） |
| `matrix_norm` | - | **#1621 追記**。`MatrixNormOrd`（`Fro`/`One`/`Inf`/`Nuc`/`Spectral`）指定 |

`Op` enum（`crates/autodiff/src/tape.rs:87-`）はこの Var メソッド集合と 1:1 対応する
（`Leaf`・`MatMul`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`Sum`・`Max`・`MseLoss`・
`CrossEntropyLoss`・`ResidentLeaf`・`LinearResident`・`LinearAct`・`Reshape`・`Transpose`・
`Inv`・`Solve`・`Det`・`Cholesky`・`QrQ`・`QrR`・`SvdU`・`SvdS`・`SvdVh`・`MatrixNorm`
〈**#1621 追記**〉）。

### 1.6 `Tensor<T>`（`crates/tensor-core/src/tensor.rs`）の shape 操作

`new`/`from_slice`/`zeros`/`ones`/`full`/`from_shape_fill`/`scalar`（生成）、
`shape`/`strides`/`offset`/`rank`/`numel`/`is_empty`/`get`/`as_slice`/`host_slice`/`as_view_slice`（アクセス）、
`transpose`/`transpose_2d`/`permute`/`narrow`/`is_contiguous`/`reshape`/`contiguous`/`broadcast_to`/`broadcast_with`（shape 変形）。
**indexing（花形インデックス）・`cat`/`stack`・`split`・`squeeze`/`unsqueeze`・`gather`/`scatter`・`expand`（`broadcast_to` はあるが `expand` API 名はない）は `Tensor<T>` レベルにも存在しない**。

### 1.7 `BackendOps` trait（`crates/tensor-core/src/backend_ops.rs`）が定義する演算 API 全量

`gemm`・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max`・`mse_loss`/`mse_loss_backward`・
`gemm_bias_act`（Linear+活性化 epilogue 融合）・`gemm_resident_rhs`/`_act`・`gemm_resident_lhs`・
`linear_forward_device`・`run_fused`（elementwise 融合実行）・`sgd_step_device`/`_tracked`・
`captured_segment_key`/`run_captured_sgd_step_segment`（CUDA Graph）・`gemm_checksum`・
`release_cached_device_memory`/`device_memory_pool_stats`・`linalg_inv`/`_solve`/`_det`/
`_cholesky`/`_qr`/`_svd`/`_matrix_norm`（**#1621 追記**。CPU 実装済み・CUDA／Metal は既定
`Unsupported` を明示オーバーライド）。**`sub`/`div`/`pow`/`sqrt`/`log`/
`sigmoid`（`BackendOps` に独立メソッドなし。`Op::Sigmoid` は `eval::sigmoid`〈`crates/autodiff/src/eval.rs:355-356`。数値安定形のホスト scalar 参照実装〉で計算し、GPU バックエンド選択時も `BackendOps` を経由しない）/`softmax`/`layer_norm`/
`conv`/`batch_norm`/`embedding`/`gather`/`scatter` はいずれも `BackendOps` に存在しない**。

なお `rmsnorm`・`softmax` の**行レベルカーネル実装自体**は CPU/CUDA/Metal 各バックエンドの
内部モジュール（`crates/backend-cpu/src/rmsnorm.rs`・`softmax.rs`、
`crates/backend-cuda/src/kernels_rmsnorm.rs`・`kernels_softmax.rs`、
`crates/backend-metal/src/rmsnorm.rs`・`softmax.rs`）に**存在する**が、`BackendOps` trait の
公開メソッドとして立っておらず、`Var`/`Tape`（autodiff）・`facade` のいずれからも到達できない
（ベンチ・parity テスト専用の内部実装。`crates/bench-harness/src/transformer_workload.rs` 等が
直接呼ぶのみ）。

### 1.8 dtype・device

- dtype: `Element` trait（`f32`/`f64`/`i32`/`half::f16`/`i64`/`bool`）は `Tensor<T>` の生成 API のみ対応
  （`crates/tensor-core/src/element.rs:26-84`）。算術は `f32` 固定。`half::f16` は GPU カーネル内部の
  中間表現としては使われる（CUDA/Metal の Tensor Core 経路）が、facade の公開型としては現れない。
- device: `Device::Cpu`/`Device::Cuda(ordinal)`/`Device::Metal`（`cfg(target_os = "macos")`）
  （`crates/tensor-core/src/device.rs`）。`to()` に相当する明示転送 API・複数デバイス間の
  自動フォールバック・デバイス列挙（`Device::available()`）は `docs/public-api-design.md` §4.1 で
  未決事項として明記され未実装（`crates/facade/src/lib.rs:75-78`）。

### 1.9 リポ内非公開（`onnx-interop`。crates.io 非公開・facade から到達不可）

ONNX opset の一部演算がホスト参照実装として存在する（`crates/onnx-interop/src/onnx/interp.rs:824-845`）:
`Gemm`・`Relu`・`Sigmoid`・`Shape`・`Gather`・`Unsqueeze`・`Concat`・`Slice`・`Add`・`Mul`・`Div`・`Mod`・
`Sqrt`・`Constant`・`Cast`・`Reshape`・`Squeeze`・`Transpose`・`MatMul`・`Softmax`・`Erf`・
`LayerNormalization`。**これらは autograd（`Tape`/`Var`）に接続されておらず推論専用のグラフ解釈器**
であり、`fandhe_ai`（facade）からは到達しない。`docs/compat-api-scope.md` の対象範囲外。

---

## 2. ギャップ表（大分類ごと）

凡例: 状態 = あり／部分／なし／リポ内非公開。難度 S=数時間〜1日, M=数日, L=1〜2週, XL=それ以上（複数バックエンド×新カーネル×VJP×parity 一式）。

### 2.1 テンソル生成

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `torch.tensor`/`from_numpy` | `tf.constant` | あり（`compat::array`・`Tensor::new`） | - | - |
| `torch.zeros`/`ones`/`full` | `tf.zeros`/`ones`/`fill` | あり（`Tensor::zeros`/`ones`/`full`。ただし `Var`/`compat` からは未再エクスポート＝`Tensor` 経由のみ） | compat 側の薄いラッパー追加 | S |
| `torch.arange`/`linspace` | `tf.range`/`linspace` | なし | `Tensor` 生成関数 1 個追加 | S |
| `torch.randn`/`rand`（乱数テンソル） | `tf.random.normal` 等 | なし（`Linear::new` 内部の重み初期化にシードベース乱数はあるが公開 API なし） | 汎用乱数テンソル生成 API（RNG 契約含む） | S〜M |
| `torch.eye` | `tf.eye` | なし | 生成関数 1 個 | S |
| `torch.sparse_coo_tensor`／`to_sparse()` | `tf.sparse.SparseTensor` | **なし（非対応。`Tensor<T>` は dense のみ。決定記録 `docs/tensor-core-sparse-complex-decision.md`〈#1633〉）** | 対象外（REQ-9 `docs/spec/04-requirements.md:233`） | - |

### 2.2 index/slice/gather/scatter

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| 基本スライス `x[a:b]` | `tf.slice`/`x[a:b]` | なし（`Tensor::narrow` は単一 dim 開始/長さのみ。`Var` レベルでは全くなし） | `Var::narrow`（VJP: 逆方向は zero-pad scatter）＋3 バックエンドカーネル or ホスト実装＋parity | M |
| 花形インデックス `x[idx]` | `tf.gather`（`gather_nd`） | なし | 新 Op（`Gather`）＋forward/backward＋3 バックエンド | L |
| `scatter`/`scatter_add`/`index_put_` | `tf.tensor_scatter_nd_*` | なし | 新 Op（`Scatter`）＋VJP＋3 バックエンド | L |
| `masked_select`/`where` | `tf.where` | なし | 新 Op（条件付き選択）＋VJP | M |
| `torch.topk`/`sort` | `tf.math.top_k`/`sort` | なし | 縮約系の拡張・非連続勾配経路の設計 | L |

### 2.3 形状操作

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `reshape`/`view` | `tf.reshape` | あり（`Var::reshape`。`Tensor::reshape`） | - | - |
| `permute`/`transpose` | `tf.transpose` | あり（`Var::transpose` は 2 軸限定 swap のみ・`Tensor::permute` は任意順だが `Var` に未接続） | `Var::permute`（VJP は逆置換）＋既存 `Tensor::permute` への接続 | S〜M |
| `cat`/`stack` | `tf.concat`/`tf.stack` | なし | 新 Op（`Concat`）＋backward（勾配を各入力へ narrow で分配）＋3 バックエンド | M〜L |
| `split`/`chunk` | `tf.split` | なし | 新 Op（`Split`）＋backward（勾配を `Concat`）＋3 バックエンド | M |
| `expand`/`broadcast_to` | `tf.broadcast_to` | 部分（`Tensor::broadcast_to`/`broadcast_with` はあるが明示 `expand`/`Var::broadcast_to` API はない。**`Var::add`/`mul` は暗黙ブロードキャストに対応済み（確定）**: `BackendOps::add`/`mul` の shape 検査 `elementwise_out_shape`〈`crates/tensor-core/src/ops_shape.rs:82-86`〉が `broadcast_shape` へ委譲しており NumPy 慣習のブロードキャストを行う） | 明示 `Var::broadcast_to`/`expand` API の追加（VJP の縮約方向勾配はブロードキャスト対応 add/mul で既に検証済みのロジックを流用可能） | S〜M |
| `squeeze`/`unsqueeze` | `tf.squeeze`/`expand_dims` | なし（`Tensor` レベルにも直接の API はない。`reshape` で代用可） | 薄いラッパー | S |

### 2.4 要素演算

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `+`/`add` | 同左 | あり（`Var::add`） | - | - |
| `*`/`mul` | 同左 | あり（`Var::mul`） | - | - |
| `-`/`sub` | 同左 | **なし** | 新 Op（`Sub`）＋VJP（片方 `-upstream`）＋`BackendOps::sub` を CPU/CUDA/Metal へ追加＋parity | M |
| `/`/`div` | 同左 | **なし** | 新 Op（`Div`）＋VJP（商の微分）＋3 バックエンドカーネル＋parity | M |
| 比較演算（`>`,`==` 等） | 同左 | なし | bool 出力の新 Op 群（非連続勾配のため VJP はゼロ扱い） | M |
| `sin`/`cos`/`tan` | 同左 | なし | elementwise unary Op×3＋VJP＋3 バックエンド | M（各） |
| `pow`/`sqrt` | 同左 | なし | elementwise unary/binary Op＋VJP＋3 バックエンド | M |
| `log`/`log2`/`log10` | 同左 | なし（`cross_entropy_loss` 内部にのみ log-sum-exp あり。汎用 log は非公開） | elementwise unary Op＋VJP（`1/x`）＋3 バックエンド | M |
| `sigmoid` | `tf.sigmoid` | あり（`Var::sigmoid`。`eval::sigmoid`〈ホスト scalar 参照実装〉で計算し `BackendOps` を経由しない。VJP は `out_value` 再利用方式で既存） | 高速化するならバックエンド専用カーネル追加＋`BackendOps::sigmoid` 新設 | S〜M（現状で機能は十分・性能改善のみ） |
| `gelu`/`silu`(swish) | 同左 | **なし**（`docs/compat-api-scope.md` が GELU を明示的スコープ外と記載） | elementwise unary Op＋VJP＋3 バックエンド。Transformer 必須 | M |
| `softmax`/`log_softmax` | 同左 | **なし（Var メソッドとしては無い）**。`cross_entropy_loss` 内部に log-softmax の融合実装があるのみ（`grad.rs`）。行カーネル自体は CPU/CUDA/Metal に既存（`softmax.rs` 系）だが `BackendOps` 未接続 | `BackendOps::softmax` を 3 バックエンドの既存行カーネルへ接続＋新 Op＋VJP（Jacobian-vector 積） | M（カーネルは既にあるため配線中心） |

### 2.5 縮約

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `sum(dim)` | `tf.reduce_sum` | あり（`Var::sum`。単一 dim または全体のみ、複数軸指定不可） | 複数軸・`keepdim` 対応への拡張 | S〜M |
| `mean(dim)` | `tf.reduce_mean` | なし（`mse_loss` 内部にのみ mean 縮約あり。汎用 `Var::mean` はない） | `sum` を分母で割るラッパー、または専用 Op | S |
| `max(dim)`/`min(dim)` | `tf.reduce_max`/`min` | 部分（`Var::max` あり・`min` なし・`argmax`/`argmin` なし・`amax`/`amin` の均等分配は明示的スコープ外） | `min`/`argmax`/`argmin` の新規 Op | M |
| `var`/`std` | `tf.math.reduce_variance`/`reduce_std` | **実装済み**（`Var::var`／`std`。イシュー #1723。`Op::Var`〈専用 Op。`sum`／`max` と同じ `dim: Option<usize>` シグネチャ＋`correction`〉・`f64` 二段計算〈平均→二乗和〉・`BackendOps::var` 既定 `Unsupported` のホストフォールバック契約） | — | — |
| `norm`（L1/L2） | `tf.norm` | **実装済み**（`Var::norm_l1`／`norm_l2`。イシュー #1723。`Op::VectorNorm`〈専用 Op。`VectorNormOrd::L1`／`L2`〉・`f64` 累積・`BackendOps::vector_norm` 既定 `Unsupported` のホストフォールバック契約。汎用 `norm(ord, dim)` の facade 公開〈`VectorNormOrd` 再エクスポート〉は Tier 1 未列挙のため見送り） | — | — |

### 2.6 線形代数

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `matmul`（2D） | `tf.matmul` | あり（`Var::matmul`。CPU BLIS／CUDA Tensor Core／Metal simdgroup 全対応・TF32/split-K opt-in 込み） | - | - |
| `bmm`（バッチ行列積） | `tf.linalg.matmul`（バッチ次元対応） | **なし（確定）**。`crates/tensor-core/src/ops_shape.rs:43-` `matmul_out_shape` が `lhs.len() != 2`/`rhs.len() != 2` を `ShapeError::RankMismatch` で拒否し、`Var::matmul`（`var.rs:195-208`）はこれを経由するため rank 2 のみ受理する | バッチ次元対応の GEMM 拡張（3 バックエンドのループ or バッチ化カーネル） | L |
| `einsum` | `tf.einsum` | **あり（#1620）**。`Var::einsum`（オペランド 1〜2 個・batch 添字を伴う縮約は対象外〈末尾追補参照〉） | - | - |
| `transpose`（線形代数用） | 同左 | あり（2.3 節参照） | - | - |
| `torch.linalg.inv` | `tf.linalg.inv` | **あり（#1621）**。`Var::inv`（rank-2 正方限定。CPU 実装・GPU は `Unsupported`） | - | - |
| `torch.linalg.solve` | `tf.linalg.solve` | **あり（#1621）**。`Var::solve` | - | - |
| `torch.linalg.det` | `tf.linalg.det` | **あり（#1621）**。`Var::det`（特異行列は `0.0`。エラーにしない） | - | - |
| `torch.linalg.cholesky` | `tf.linalg.cholesky` | **あり（#1621）**。`Var::cholesky`（下三角のみ・`upper=True` 相当は対象外） | - | - |
| `torch.linalg.qr` | `tf.linalg.qr` | **あり（#1621）**。`Var::qr`（reduced QR のみ・`m<n` backward は対象外） | - | - |
| `torch.linalg.svd` | `tf.linalg.svd` | **あり（#1621）**。`Var::svd`（reduced SVD のみ・相異なる特異値前提の backward） | - | - |
| `torch.linalg.matrix_norm` | `tf.norm` | **あり（#1621）**。`Var::matrix_norm`（`MatrixNormOrd`: Fro/One/Inf/Nuc/Spectral） | - | - |
| `torch.linalg.eigh`/`lstsq`/`pinv`/`matrix_rank`/`slogdet` | 相当 API | なし（#1621 スコープ外） | 各分解アルゴリズムの追加実装 | M〜L |

### 2.7 NN 層

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `nn.Linear` | `layers.Dense` | あり（`Linear`。bias 有無・epilogue 融合済み） | - | - |
| `nn.Conv1d`/`Conv2d` | `layers.Conv1D`/`Conv2D` | **なし** | im2col か直接畳み込みカーネル（CPU/CUDA/Metal）＋VJP（d_input は転置畳み込み・d_weight は相関）＋parity。GEMM 基盤を再利用可能だが新カーネル必須 | XL |
| `nn.BatchNorm2d` | `layers.BatchNormalization` | **なし**（rmsnorm はあるが batchnorm は統計対象軸・running stats が異なる） | 新 Op（バッチ統計・running mean/var の状態保持）＋3 バックエンド | L |
| `nn.LayerNorm` | `layers.LayerNormalization` | **なし**（`BackendOps` に layer_norm メソッドなし。onnx-interop にはホスト実装あり・非公開） | `BackendOps::layer_norm` 新設＋VJP＋3 バックエンド（rmsnorm の実装パターンを流用可能） | M〜L |
| RMSNorm | （TF に相当レイヤーなし。カスタム実装が一般的） | リポ内非公開（`backend-{cpu,cuda,metal}::rmsnorm` に行カーネルあり・`BackendOps`/`Var` 未接続） | `BackendOps::rmsnorm` 新設・`Var`/`nn::RmsNorm` 配線 | M（カーネルは既存） |
| `nn.Dropout` | `layers.Dropout` | **なし** | RNG 契約設計＋マスク適用 Op（train/eval モード分岐）＋VJP | M |
| `nn.Embedding` | `layers.Embedding` | **なし** | gather 系 Op が前提（2.2 節）＋embedding テーブル管理 | L |
| `nn.MultiheadAttention` | `layers.MultiHeadAttention` | **なし** | softmax・batched matmul・(optional) causal mask・reshape/transpose の組合せ実装。前提演算が軒並み未実装 | XL |
| RNN/LSTM/GRU | `layers.SimpleRNN`/`LSTM`/`GRU` | 内部クレート `fandhe_ai_autodiff::nn::rnn`（`RnnCell`/`LstmCell`/`GruCell`・`Rnn`/`Lstm`/`Gru`）に実装済み（3 バックエンド〈CPU・CUDA・Metal〉数値一致。Metal は実機実測完了・CUDA は本エージェント実行環境に実機なしのため未実測明記）。**facade（`fandhe_ai`）未公開**（`docs/compat-api-scope.md` §5 の範囲拡張手続きのうちユーザー承認が未取得のため。決定 10）。`forward_seq`（tape 経路）の出力は `Var::stack`〈#1598〉未実装のため `[T,B,H]` ではなく `Vec<Var>`（per-step）。設計: `docs/autodiff-rnn-cell-tape-design.md`（#1646）・実装記録: 同文書 §8（#1647） | XL（設計・内部実装は完了。facade 公開のみ残作業） |
| Pooling（Max/AvgPool） | `layers.MaxPooling2D` 等 | **なし** | Conv 同様の空間走査カーネル＋VJP（max は argmax 経路の逆伝播）。設計: `docs/pooling-ops-design.md`（#1727） | L |

### 2.8 損失

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `MSELoss` | `losses.MeanSquaredError` | あり（`Var::mse_loss`/`mse_loss_with`。mean/sum 縮約・融合カーネル） | - | - |
| `CrossEntropyLoss` | `losses.SparseCategoricalCrossentropy` | あり（`Var::cross_entropy_loss`。log-softmax 融合） | - | - |
| `BCELoss`/`BCEWithLogitsLoss` | `losses.BinaryCrossentropy` | なし | 新融合 Op（MSE/CrossEntropy と同じ設計パターン） | M |
| `NLLLoss` | - | なし（`cross_entropy_loss` が事実上兼ねる設計） | 既存 CrossEntropy から log 済み入力を受け付ける版を分離するかは要判断 | S〜M |
| `HuberLoss`/`SmoothL1Loss` | `losses.Huber` | なし | 新融合 Op（区分的関数の VJP） | M |

### 2.9 optimizer

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `SGD`（momentum/nesterov/weight_decay） | `optimizers.SGD` | あり（`optim::Sgd`。デバイス常駐版 `DeviceParamStore::step` も対応） | - | - |
| `AdamW` | `optimizers.AdamW` | あり（`optim::AdamW`） | - | - |
| `Adam`（coupled L2 weight decay） | `optimizers.Adam` | 部分（decoupled 版 `AdamW` はあり、`weight_decay=0` のときは `Adam` と完全に一致する。`weight_decay>0` の coupled L2 版〈PyTorch `Adam(weight_decay>0)`〉はなし） | `AdamW` の decay 適用箇所（勾配へ加算 vs パラメータへ直接減算）を分岐する薄い派生 | S |
| `RMSprop` | `optimizers.RMSprop` | なし | 新 optimizer 型（値型・純関数。`Sgd`/`AdamW` と同型） | S〜M |
| `Adagrad` | `optimizers.Adagrad` | なし | 同上 | S〜M |
| LAMB | - | なし | 同上（layer-wise trust ratio の追加） | M |

### 2.10 scheduler

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `StepLR` | `LearningRateSchedule`（`ExponentialDecay` 等） | あり（`StepLr`） | - | - |
| 定数 LR | 同左 | あり（`ConstantLr`） | - | - |
| `CosineAnnealingLR` | `CosineDecay` | なし | 新 `LrScheduler` 実装（純関数） | S |
| `ExponentialLR` | `ExponentialDecay` | なし | 同上 | S |
| `ReduceLROnPlateau` | `ReduceLROnPlateau`（callback） | なし | 状態保持（履歴・patience）を持つ scheduler 型 | M |
| `OneCycleLR` | - | なし | 同上 | M |

### 2.11 autograd

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| 動的テープ・`backward()` | `tf.GradientTape` | あり（`Tape::backward`。動的テープ方式。`Tape::reset` で再利用可能） | - | - |
| `no_grad()`/`torch.inference_mode()` | `tf.stop_gradient` | なし（テープに載せない選択肢は「別 `Tape` を使わない」設計自体にない。`tape_free` 推論経路〈`Sequential::predict`〉はあるが汎用 `no_grad` コンテキストではない） | 葉ノード登録をスキップする API、または `Var` を「追跡なし」でラップする型 | M |
| `detach()` | `tf.stop_gradient` | なし | 既存 `Var` から新規葉ノードへ変換する Op | S〜M |
| `retain_graph=True` | - | 該当なし（`Tape::backward` はグラフノード〈`nodes`〉を破棄せず、呼び出しごとに独立した `Gradients` を新規生成するのみで、グラフ保持は既定動作。`retain_graph` フラグ自体が不要な設計。ただし PyTorch の `.grad` 蓄積〈複数回 `backward()` の勾配加算〉に相当する契約は無く、同一 loss に対する複数回 `backward()` の勾配蓄積セマンティクスは未検証） | 設計判断が必要（複数回 backward の勾配蓄積契約） | M |
| 高階微分（`grad of grad`） | `tf.GradientTape` のネスト | なし（テープは 1 階のみを前提とした構造と推定） | Op 自体を微分可能にする再設計（VJP の VJP）。設計: `docs/autodiff-higher-order-grad-decision.md`（#1622） | XL |
| custom `autograd.Function` | `tf.custom_gradient` | なし（`Op` enum は crate 非公開の固定 variant 集合。ユーザー定義 Op を挿す口がない） | 拡張可能な Op プラグイン機構の設計（現行のクローズドな `Op` enum 設計を変更）。設計: `docs/autodiff-custom-function-decision.md`（#1623） | XL |
| `torch.utils.checkpoint`（activation checkpointing） | `tf.recompute_grad` | 部分（view 系ノード〈reshape/transpose〉は #1080 で再計算方式化済みだが、任意サブグラフの再計算チェックポイントではない） | 汎用チェックポイント機構の設計 | L |

### 2.12 dtype

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `float32` | `float32` | あり（唯一の演算 dtype） | - | - |
| `float64` | `float64` | 部分（`Tensor<f64>` は生成できるが `BackendOps`/`Var` の算術対象外） | `Element` 抽象を活かした dtype 別 dispatch の設計・全カーネルの多重化（設計はイシュー #1648・`docs/backend-dtype-dispatch-design.md` で確定済み。実装は #1649〜#1651 へ引き継ぎ） | XL |
| `float16`/`bfloat16` | 同左 | 部分（GPU カーネル内部の中間表現・Tensor Core 経路にのみ存在。公開 dtype ではない） | 公開 `Tensor<f16>` 演算経路・VJP のスケーリング契約設計（mixed precision）（dtype dispatch 方式はイシュー #1648・`docs/backend-dtype-dispatch-design.md` で確定済み。AMP スケーリング契約は #1625 の対象） | XL |
| `int32`/`int64`/`bool` | 同左 | 部分（`Tensor<T>` 生成のみ。CrossEntropy の `targets: Tensor<i32>` のように限定的に内部使用） | 汎用整数演算・型変換 API | L |
| `.to(dtype)`（型変換） | `tf.cast` | なし | dtype 変換 Op（勾配は型により打ち切り／恒等など個別設計） | M |
| AMP（自動混合精度） | `tf.keras.mixed_precision` | なし（`optim.rs` doc に「損失スケーリング（AMP）は現時点で未実装」と明記） | 損失スケーリング・unscale ステップの追加（`optim.rs` の適用順序契約に定義済みの拡張点） | L |
| `complex64`／`complex128`（`torch.fft` 含む） | `tf.complex64`／`tf.signal.fft` | **なし（非対応。`Scalar` は実数 4 型に封印。ONNX COMPLEX は `UnknownDataType` で拒否。決定記録 同上）** | 対象外（同上） | - |

### 2.13 device

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| 単一 GPU 選択（`device='cuda:0'`） | `tf.device` | あり（`Device::Cuda(ordinal)`） | - | - |
| `.to(device)`（テンソル転送） | `tf.identity` with device | なし（`tape_for` でバックエンドごと `Tape` を切替える設計。テンソル単体を明示転送する API はない） | `Tensor`/`Var` のデバイス間コピー API | M |
| 複数 GPU・`DataParallel`/`DDP` | `tf.distribute.MirroredStrategy` | なし | 勾配 all-reduce・パラメータ複製の設計（ネットワーク層から必要）。設計: `docs/facade-multi-gpu-ddp-decision.md`（#1628） | XL |
| デバイス自動列挙（`torch.cuda.device_count()`） | `tf.config.list_physical_devices` | なし（`docs/public-api-design.md` §4.1 未決事項として明記） | `Device::available()` 相当の列挙 API | S〜M |

### 2.14 データ

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `Dataset`/`DataLoader`（バッチ化・シャッフル） | `tf.data.Dataset` | なし | イテレータ・シャッフル・バッチ化の薄い層（既存 `Tensor` の上に構築可能） | M |

### 2.15 保存・相互運用

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `state_dict()`/`load_state_dict()` | `model.save_weights` | 部分（`Sequential::trainable_parameters`/`apply_parameters` で手動シリアライズは組める。汎用 `state_dict` 相当の名前付き辞書 API はない） | 層名→テンソルのマップ API | S〜M |
| safetensors 読み書き | - | リポ内非公開（`onnx-interop::st_load`/`st_save`。facade 未接続） | facade からの再エクスポート、または `Sequential` の save/load ラッパー | M |
| ONNX export | `tf2onnx` 等 | なし（`onnx-interop` は import 方向のみ・かつ非公開） | export 側の実装＋facade 公開判断 | XL |
| ONNX import | `torch.onnx`（逆方向） | リポ内非公開（`onnx-interop::onnx::interp`。autograd 未接続の推論専用グラフ解釈器） | facade への公開判断＋（学習させるなら）`Tape` への変換層 | L（公開のみなら）〜XL（学習可能化） |

### 2.16 推論・その他

| PyTorch | TF/Keras | fandhe-ai | 実装に必要なもの | 難度 |
|---|---|---|---|---|
| `torch.compile`（グラフ最適化 JIT） | `tf.function`（AutoGraph・XLA） | 部分（`run_fused`＝elementwise カーネル融合・CUDA Graph capture opt-in はあるが、汎用グラフ JIT コンパイラではない） | 既存インフラの延長線上で拡張可能（新規 JIT は不要）。設計: `docs/autodiff-graph-optimization-scope-decision.md`（#1632） | - |
| 量子化（int8 等） | `tf.lite` 量子化 | なし | 量子化 dtype・演算対応（2.12 節の dtype 拡張が前提） | XL |
| 乱数シード固定（`manual_seed`） | `tf.random.set_seed` | 部分（`Linear::new(.., seed: u64)` など個別 API にシード引数はあるが、グローバル RNG 状態を握る `manual_seed` 相当はない） | グローバル RNG 契約の設計（Dropout 等 今後追加する確率的演算との整合が前提） | M |

---

## 3. 「MLP → CNN → Transformer」に必要な最小集合（Tier 1）と長尾（Tier 2）

### Tier 1（MLP はほぼ揃っている。CNN・Transformer に必須の欠落）

| 機能 | 現状 | compat-api-scope.md の位置づけ |
|------|------|-------------------------------|
| **MLP** | ほぼ揃っている（Linear・ReLU/Sigmoid/Tanh・MSE/CrossEntropy・SGD/AdamW） | 対象範囲内（1 節） |
| `Var::sub`/`div`/`pow`/`sqrt` | なし | 記載なし（対象範囲・対象外いずれにも明記なし＝5 節の範囲拡張手続きが必要な未定義事項） |
| `softmax`（Var/nn 単体） | なし（CrossEntropy 内部にのみ融合実装） | **明示的に対象外**（`docs/compat-api-scope.md:194` 「CrossEntropy と密結合のため対象外」）。単体 softmax の公開には spec 側（対象範囲）の改定が必要 |
| GELU/SiLU | なし | **明示的に対象外**（同 :195「必要になった時点で後続イシューに切り出す」） |
| LayerNorm | なし（`BackendOps` 未接続） | 記載なし（Keras の「全レイヤー種別の網羅を対象外」に該当し得るため spec 側確認が必要） |
| Conv1d/Conv2d | なし | **明示的に対象外**（同 :192「Conv 系」） |
| MultiheadAttention | なし（softmax・batched matmul・masking いずれも未実装） | 記載なし（Keras レイヤー網羅の対象外に準ずると推定） |
| cat/stack/split（Transformer の head 分割・結合に必須） | なし | 記載なし |
| gather（Embedding の前提） | なし | 記載なし |
| Embedding | なし | **明示的に対象外**（同 :192「Embedding 等」） |
| Dropout | なし | 記載なし（Keras レイヤー網羅の対象外に準ずると推定） |
| bmm（batched matmul。Attention の QK^T に必須） | **なし（確定。`matmul_out_shape` が rank 2 以外を拒否）** | 記載なし |

**結論**: MLP は Tier 1 として成立している。CNN（Conv・Pooling・BatchNorm）・Transformer
（softmax・LayerNorm・Embedding・MultiheadAttention・cat/split・bmm）に進むには、
上表の欠落項目のうち **softmax・GELU・Conv・Embedding の 4 項目は `docs/compat-api-scope.md`
2 節が明示的にスコープ外と定めており、対象範囲を広げるには 5 節の手続き（正本 spec 側の
REQ-9 受け入れ基準改定、またはユーザー承認を得たうえでの本文書更新）を経る必要がある**。
LayerNorm・MultiheadAttention・Dropout・cat/stack・bmm・gather は現行の compat-api-scope.md に
明記がなく、範囲判断自体を要する。

### Tier 2（長尾。当面の MLP/CNN/Transformer 到達には不要）

RNN/LSTM/GRU・Pooling 各種・BatchNorm・einsum・高階微分・custom autograd Function・
AMP・量子化・DDP・多 GPU・DataLoader/Dataset・ONNX export・state_dict 汎用シリアライズ・
花形インデックス・scatter 系・比較演算・三角関数・var/std・cosine/exponential/plateau
scheduler・RMSprop/Adagrad/LAMB。

---

## 4. 既存 Op 追加のパターン（実例トレース）

**題材**: `BackendOps::mse_loss`/`mse_loss_backward` の融合実装追加（イシュー #1045、コミット
`1e13b773`「perf(autodiff): MSE loss の reduction を単一カーネルへ融合する (#1078)」）。
既存 `Op::MseLoss` 自体はホスト参照実装（`eval::mse_loss`）としてそれ以前から存在しており、
本コミットは「ホスト参照実装 → 3 バックエンド融合カーネル」への昇格パターンを示す好例
（新規 Op の追加も基本的に同じファイル群に触れる）。

| 層 | 触ったファイル | 内容 |
|----|----------------|------|
| tensor-core（trait 定義） | `crates/tensor-core/src/backend_ops.rs`（+107 行） | `BackendOps::mse_loss`/`mse_loss_backward` のデフォルト実装（`Unsupported`）と `MseReduction` enum を追加 |
| tensor-core（再エクスポート） | `crates/tensor-core/src/lib.rs` | 新規型の公開 |
| autodiff（Var/Op/VJP） | `crates/autodiff/src/var.rs`（+60）、`crates/autodiff/src/grad.rs`（+58） | `Var::mse_loss_with` が `BackendOps::mse_loss` を優先し `Unsupported` のみ `eval::mse_loss` へフォールバック。VJP も同様に `mse_loss_backward` 優先＋フォールバック |
| autodiff（テスト） | `crates/autodiff/tests/mse_loss_fusion.rs`（新規 385 行） | 融合 forward/backward の数値・フォールバック契約テスト |
| backend-cpu | `crates/backend-cpu/src/mse.rs`（新規 188 行）・`ops.rs`（+72）・`lib.rs`（+1） | CPU カーネル実装・`BackendOps` 実装への配線 |
| backend-cpu（parity） | `crates/backend-cpu/tests/mse_parity.rs`（新規 139 行） | 数値一致テスト |
| backend-cuda | `crates/backend-cuda/src/kernels_mse.rs`（新規 254 行）・`mse.rs`（新規 229 行）・`ops.rs`（+131）・`context_cache.rs`（+13）・`lib.rs`（+3） | NVRTC カーネル・コンテキストキャッシュ・`BackendOps` 配線 |
| backend-cuda（parity） | `crates/backend-cuda/tests/mse_parity.rs`（新規 145 行） | 数値一致テスト（`#[ignore]` 実機依存） |
| backend-metal | `crates/backend-metal/src/mse.rs`（新規 326 行）・`shaders/mse.metal`（新規 163 行）・`ops.rs`（+82）・`context_cache.rs`（+9）・`lib.rs`（+4） | MSL カーネル・`BackendOps` 配線 |
| backend-metal（parity・証跡） | `crates/backend-metal/tests/mse_parity.rs`（新規 111 行）・`mse_source_evidence.rs`（新規 104 行） | 数値一致テスト・融合カーネルが実際に呼ばれることのソース走査証跡 |
| ドキュメント | `docs/kernel-fusion.md`（+2） | 汎用 reduction 融合の限界注記を更新 |

**合計 23 ファイル・約 2569 行追加**（`1e13b773` の diffstat）。新規 elementwise Op
（例: `sub`/`div`/`gelu`）を素朴に追加する場合はこれよりやや小さく（VJP が単純・
struct variant 化不要）、Conv や Attention のような新規演算カテゴリを追加する場合は
これより大きくなる（新規 shader/kernel 設計・タイル戦略・reuse/fresh 両モード対応が必要）。

**再利用可能なテンプレート（この 1 コミットから読み取れる型）**:
1. `tensor-core::BackendOps` に新メソッド（デフォルト実装 `Unsupported`）を追加
2. `autodiff::Var` に新メソッド、`autodiff::grad::vjp` に対応 VJP 分岐を追加
   （`BackendOps` 優先・`Unsupported` のみホスト実装へフォールバックする二段構え）
3. `backend-cpu`/`backend-cuda`/`backend-metal` それぞれで実装・`ops.rs` へ配線
4. 各バックエンドに parity テスト（数値一致複合判定）・Metal は「実際にそのカーネルが
   呼ばれる」ことを保証する source-evidence テストを追加
5. `docs/` の関連設計ドキュメントを更新

## 追補（2026-09-12・イシュー #1594）

§2.4 の「`softmax`/`log_softmax`」行（上記表）は取り込み元スナップショット
（調査日 2026-09-12 時点）のギャップ記述のため変更していないが、同イシューで
このギャップは解消済みである: `tensor-core::BackendOps::softmax`／
`log_softmax`（デフォルト `Unsupported`）を新設し、`autodiff::tape::Op::
Softmax`／`LogSoftmax`＋VJP・`autodiff::var::Var::softmax`／`log_softmax`・
`nn::activation::Softmax`／`LogSoftmax`（`Module` 実装）を追加した（上記
「再利用可能なテンプレート」と同型のテンプレートで実装）。3 バックエンドの
既存行カーネル（CPU/CUDA/Metal の `softmax.rs`。§2.4 が「既存」と記す
カーネル）へ接続済み。**GPU（CUDA／Metal）の `log_softmax` は行カーネルを
新設せず、`BackendOps::log_softmax` の既定 `Unsupported` のままホスト参照
実装（`eval::log_softmax_along`）へフォールバックする**（CPU のみ融合
カーネル `backend-cpu::softmax::run_log_softmax_f32` で本番オーバーライド
する）。非最終軸 softmax／log_softmax も同様にホストフォールバック。
`cross_entropy_loss` 内部の log-softmax（`eval::softmax_along`。1 個の融合
オペとして解析形で forward/backward を閉じる既存実装）は本イシューで
変更しない（別実装のまま独立に存在する）。GPU `log_softmax` カーネル・
非最終軸 GPU 対応は out-of-scope として記録し、起票はユーザー承認後に限る
（`.claude/rules/out-of-scope-tracking.md`）。

## 追補（2026-09-13・イシュー #1624）

§2.11 の「`torch.utils.checkpoint`（activation checkpointing）」行（上記
表）は取り込み元スナップショットのギャップ記述のため変更していないが、
同イシューで「部分」から「あり（対象 Op 限定）」へ前進した: `Var::
checkpoint_from`（内部クレート `Tape::checkpoint` の低儀式な代替入口。
facade へは既存の `Var` 再エクスポート経由で到達）を追加し、`Op::MatMul`／
`Sigmoid`／`Sum`／`Max`（+ 既存の view 系 `Reshape`／`Transpose`）に限り
forward の中間値を解放し backward 時に再計算する（`docs/
autodiff-checkpoint-design.md`）。`MseLoss`／`CrossEntropyLoss`／
`LinearAct`／`Softmax`／`LogSoftmax`・線形代数系は非対象のまま値を保持する
（正しさ優先・エラーにはならない）。facade `Tape::checkpoint`（閉包版）
passthrough は承認未取得のため未追加（`docs/compat-api-scope.md` §1.3）。
### 追補（イシュー #1596）

§2.7 の表（`nn.LayerNorm`／RMSNorm の行）は本ドキュメント作成時点（対象 HEAD
`097bff19`）のスナップショットとして不変のまま残す。イシュー #1596 で以下を実装し、
上記ギャップを解消した（設計・実機実測状況の詳細は `docs/norm-ops-design.md` を正とする）:

- `fandhe_ai_tensor_core::BackendOps::rmsnorm`／`layer_norm` を新設（既定 `Unsupported`）。
  `rmsnorm` は既存の RMSNorm 行カーネル（`backend-{cpu,cuda,metal}::rmsnorm`）を
  `run_fused`（canonical 融合プラン限定経路）とは別の独立エントリとして接続した。
  `layer_norm` は 3 バックエンドとも新設カーネルで実装した
- `fandhe_ai_autodiff::Var::rms_norm`／`layer_norm`・`nn::RmsNorm`／`LayerNorm`
  （`RmsNormVars`／`LayerNormVars` 込み）を追加し、`Module` trait を実装した
- VJP（`grad.rs::rmsnorm_vjp_rows`／`layer_norm_vjp_rows`）をホスト側に実装し、
  数値微分・解析的性質（`Σ_row dx = 0` 等）で検証した
- facade（`crates/facade/src/`）への新規 `pub use`／`pub fn` 追加は**行っていない**。
  既存の `Var` 再エクスポート経由でユーザーへ到達する（#1594 softmax と同型の方針）
- 対象外: GPU backward カーネルの結線（VJP はホスト側実装のまま）・CUDA 既存
  RMSNorm backward カーネル（`rmsnorm_bwd_*`）への接続・多次元 `normalized_shape`・
  `Sequential::add_rms_norm`／`add_layer_norm`（#1618 のスコープ）・CPU 側 NEON
  ベクトル化・CUDA 実機実測（本エージェント実行環境に CUDA 実機への到達手段がない
  ため未実施のまま記入欄を残す）

## 追補（イシュー #1597）

§2.3「形状操作」表（上記）は取り込み元スナップショットのギャップ記述の
ため変更していないが、`permute`／`expand`（`broadcast_to`）／
`squeeze`／`unsqueeze` の各行が指す欠落は本イシューで解消済みである。

- `Var::permute`（新規 `tape::Op::Permute { input, perm }`。`Tensor::permute`
  への zero-copy 接続・VJP は逆置換 `upstream.permute(&inverse_perm)`）
- `Var::broadcast_to`（新規 `tape::Op::BroadcastTo { input }`。
  `Tensor::broadcast_to` への stride 0 view 接続・VJP は `Op::Add`/`Op::Mul`
  の暗黙ブロードキャストと同じ `reduce_to_shape` 縮約を再利用）
- `Var::expand`（`Var::broadcast_to` への薄い委譲。PyTorch 名の別名。
  負値〈-1 で軸維持〉は `shape: &[usize]` の型上表現できないため非対応）
- `Var::squeeze`／`Var::unsqueeze` は新規 `Op` を持たず、いずれも
  `Var::reshape`（既存 `Op::Reshape`）へ委譲する薄いラッパー（§4 の
  「薄いラッパー」区分どおり）。`squeeze(Some(d))` は `shape[d] != 1` の
  場合 PyTorch 準拠で no-op（numpy／TF はエラーにするが、適合する
  `ShapeError` variant が存在せず crates.io 公開クレート `tensor-core` の
  公開 enum への variant 追加は semver 可視の変更になるため見送った）
- `Var::flatten`（PyTorch `torch.flatten(start_dim, end_dim)` 相当。同じく
  `Var::reshape` への委譲。§2.3 表には行がないが同じ Tier 1 issue の
  スコープとして実装した）

いずれも `Var::reshape`／`transpose` と同じ「非 contiguous な入力への
適用は `ShapeError::NonContiguousReshape`」制約（案 A）を継承する（
`Var::contiguous()`〈明示コピー Op〉は本イシューでは追加せず、
out-of-scope として記録した）。`cat`／`stack`／`split`（§2.3 の残り 2 行）
は #1598 へ引き継ぎ。5 演算自体はホスト `Tensor` の stride 再解釈のみで
`BackendOps` を経由しないため 3 バックエンドへの個別実装は不要——
下流の融合経路（`add`）・GEMM カーネル（`matmul`）が view を正しく
消費することを `crates/facade/tests/shape_ops_backend_parity.rs` で
検証した（CPU 属性なし・Metal／CUDA `#[ignore]`）。

## 追補（イシュー #1598）

`cat`／`stack`／`split`（§2.3）・`narrow`（`docs/spec/04-requirements.md`
「index 系」・#1599 の対象だった行）を実装済み化した:

- `Var::cat(vars, dim)` — 新 Op `Op::Concat`（コピーを伴う `push_eager`
  ノード。`BackendOps::concat`〈既定 `Unsupported`〉→ `eval::concat`
  フォールバック）。3 バックエンド（CPU／CUDA／Metal）の専用カーネルは
  本イシューでは実装せず（既定実装のホストフォールバックのみ）、
  性能最適化は別イシューへ引き継ぐ（out-of-scope）。
- `Var::stack(vars, dim)` — 各要素を `unsqueeze(dim)` してから `cat`
  （PyTorch の定義そのもの）。
- `Var::narrow(dim, start, len)` — 新 Op `Op::Narrow`（zero-copy view
  ノード。`push_view`／`resolve_view` 経由。`Tensor::narrow` の再導出）。
  `docs/compat-api-scope.md` §1.2「index 系」行の narrow はこれで解消
  済み（残る where／gather／scatter は #1599 が引き続き対象）。
- `Var::split(split_size, dim)`／`split_with_sizes(sizes, dim)`／
  `chunk(chunks, dim)` — いずれも `Var::narrow` への委譲（PyTorch
  意味論。`split` の VJP は「Split の VJP は Concat」の原則で
  `Op::Narrow` の VJP が zero-pad `Concat` により入力 shape へ戻す）。

facade への到達経路は既存の `pub use fandhe_ai_autodiff::Var` 再エクス
ポートのみで、新規 `pub use`／`pub fn` は追加していない（`docs/
compat-api-scope.md` §5 の手続きは Tier 1 列挙済み機能につき再適用
不要と判断）。

## 追補（イシュー #1637）

§2.2「`masked_select`/`where`」行（スナップショット時点の記述は不変の
まま）を実装済み化した:

- `Var::where_cond(cond: &Tensor<bool>, a: &Var, b: &Var)` — 新 Op
  `Op::Where { cond: Tensor<f32>, a: NodeId, b: NodeId }`（コピーを伴う
  `push_eager` ノード。`BackendOps::where_cond`〈既定 `Unsupported`〉→
  `eval::where_cond` フォールバック）。`cond`（`&Tensor<bool>`）は
  `Var::where_cond` が `out_shape`（`a`／`b`／`cond` **3 入力**の
  broadcast 後 shape。`cond` 単独が軸を拡張するケースも含む。PR #1684
  で is-shape 契約を訂正）へ broadcast してから 1 回だけ f32 マスク
  （`{0.0, 1.0}`）へ変換し Op が保持する（`MemoryOps` の f32 専用契約
  に合わせるため）。真偽判定は 3 バックエンド共通で `c != 0.0`。
- `Var::masked_fill(&self, mask: &Tensor<bool>, value: f32)` — 新 Op
  `Op::MaskedFill { input: NodeId, mask: Tensor<f32> }`（`value` は
  forward が焼き込んだ `TapeNode::value` に含まれるため Op へ二重保持
  しない）。`BackendOps::masked_fill`〈既定 `Unsupported`〉→ `eval::
  masked_fill` フォールバック。
- **CPU／CUDA／Metal の 3 バックエンドとも専用カーネルを実装**
  （`backend-cpu::elementwise::{where_slice, masked_fill_slice}`・
  CUDA `kernels_elementwise.rs::{EW_WHERE_F32, EW_MASKED_FILL_F32}`・
  Metal `shaders/elementwise.metal::{ew_where_f32, ew_masked_fill_f32}`。
  `#1598` の cat／narrow 系とは異なりホストフォールバックのみに留めて
  いない）。選択演算は丸めを伴わないため 3 バックエンドとも bit 同一
  になることを parity テスト（`crates/backend-cuda/tests/
  where_masked_fill_parity.rs`・`crates/backend-metal/tests/
  where_masked_fill_parity.rs`。Metal は M4 Max 実機実測完了・CUDA は
  本エージェント実行環境に実機なしのため未実測明記）で確認した。
- **VJP はホスト実装**（`Op::Relu` と同型。`grad::elementwise_mul_mask`
  を再利用）。デバイス常駐 VJP（`binary_elementwise_device` 相当）は
  本イシューのスコープ外。
- facade への到達経路は既存の `pub use fandhe_ai_autodiff::Var` 再エク
  スポートのみで、新規 `pub use`／`pub fn` は追加していない（`docs/
  compat-api-scope.md` §1.2「index 系」行を参照。§5 の範囲拡張手続きは
  Tier 1 列挙済み機能につき再適用不要と判断）。
- gather／scatter／scatter_add／index_select（`docs/compat-api-scope.md`
  §1.2「index 系」行の残対象）は #1638 へ引き継ぐ。

## 追補（イシュー #1620）

§2.6「`einsum`」行（スナップショット時点の記述は不変のまま）を実装済み
化した:

- `Var::einsum(spec: &str, operands: &[&Var])` — 新 Op `Op::Contiguous
  { input: NodeId }`（`permute` 後の非 contiguous view を `reshape` へ
  渡す前段の明示実体化。eager・`push_eager`。VJP はホスト側で upstream
  パススルー）以外の新規カーネルは追加せず、既存の `Var::matmul`
  （GEMM）・`sum`（縮約）・`permute`／`reshape`（view）・`mul`
  （broadcast 乗算）への分解として実装した。`BackendOps` へのメソッド
  追加はなし——分解先の演算がすでに CPU／CUDA／Metal の 3 バックエンド
  で実装済みのため「該当バックエンドすべてに実装」は分解によって自動
  的に充足される。VJP も `einsum` 専用のものは追加せず、分解先各演算の
  VJP 合成として自動的に成立する。
- 受理範囲（v1・安全側）: 添字は ASCII 英字のみ・オペランド 1〜2 個・
  `->` 省略時は NumPy 既定（入力に 1 回だけ現れる添字を ASCII 昇順）。
  ellipsis（`...`）・同一オペランド内の添字重複（対角／trace）・出力
  添字の重複・オペランド 3 個以上は `AutodiffError::InvalidArgument`
  で拒否する。
- **batch 添字を伴う縮約（例 `"bij,bjk->bik"`）は非対応**。当初の見積
  もり（本表 §2.6「実装に必要なもの」列。スナップショット記述）は
  「バッチ次元対応の GEMM 拡張」とだけ記していたが、実装時に判明した
  正確な見積もりは以下のとおり: `einsum` 分解ドライバ自体は rank≥3
  `matmul`（`bmm`。#1600）の有無に関わらず batch 添字の分類
  （`compute_binary_plan`）まで機構として持っているため、対応は単なる
  「ガード撤去」では済まず、#1600 実装後に `einsum_matmul_path` を
  `[batch..., L, K] × [batch..., K, R]` 形状の rank≥3 `matmul` 呼び出し
  （現行の 2 次元 `[L, K] × [K, R]` から一般化）・対応する `reshape`／
  `permute` 目標形状（batch 次元を保持したまま `left`／`contract`／
  `right` を畳み込む）へ再設計する必要がある。
- **検証と `Var` 操作の分離**: `compute_binary_plan`（純関数・添字集合
  のみで分類・拒否判定を行う）がすべての検証（presum 計画・
  batch/contract/left/right 分類・内部整合性検査・batch∧contract 非空
  の拒否）を完了してから、初めて `Var::sum`（presum の実行）を呼ぶ
  設計とした。分類は添字集合のみで決まり presum の実行結果には依存
  しないため、拒否時に tape へ迷子ノードが残らない。
- **恒等最適化**: 恒等 permute（並べ替え不要）・shape 不変の reshape
  はいずれも `Var::permute`／`reshape` を呼ばずスキップする。この結果
  `"ij,jk->ik"` は `Var::matmul` 直接呼び出しと bit 同一（`MatMul`
  ノード 1 個だけを記録する）。
- facade への到達経路は既存の `pub use fandhe_ai_autodiff::Var` 再エク
  スポートのみで、新規 `pub use`／`pub fn` は追加していない（`docs/
  compat-api-scope.md` §1.3「einsum」行参照）。
- CPU（`CpuBackendOps`）・Metal（M4 Max 実機実測完了）は分解先演算の
  parity を確認済み。CUDA は本エージェント実行環境に実機がないため
  `#[ignore]` テスト（`crates/facade/tests/einsum_backend_parity.rs`）
  として未実測のまま記録し、GB10 実機セッションへ引き継ぐ。

## 追補（イシュー #1634）

§2.4「要素演算」の各行（`sub`／`div`／比較演算／`sin`/`cos`/`tan`／
`pow`/`sqrt`／`log`系／`gelu`/`silu`）が挙げる「実装に必要なもの」の
うち、**enum・dispatch 機構・CPU 参照実装・VJP は #1634 で共通基盤とし
て実装済み**（`ScalarUnaryOp`／`ScalarBinaryOp`。`tensor-core::
scalar_op`・`BackendOps::scalar_unary`／`scalar_binary`〈既定
`Unsupported`〉・`backend-cpu::scalar_elementwise`・`autodiff::Op::
ScalarUnary`／`ScalarBinary`〈汎用 VJP〉。設計記録は `docs/scalar-op-
dispatch-design.md`）。表の各行自体は変更しない（本イシューでは
`Var` 個別メソッド〈`sub`／`div`／`pow`／活性化等〉を追加しないため）。
残作業は 2 系統:

- **CUDA／Metal カーネル**（#1635／#1636）: 現状は `BackendOps::
  scalar_unary`／`scalar_binary` が既定 `Unsupported` のままのため、
  CUDA／Metal 実行時はホスト参照実装（`autodiff::eval::scalar`）への
  フォールバックで動作する（性能最適化はまだ入らない）。
- **`Var` 公開メソッド・facade 配線**（#1593／#1595）: `Var::
  scalar_unary`／`scalar_binary` は `pub(crate)` のまま（公開 API 面の
  範囲拡張は `docs/compat-api-scope.md` §5 の手続きに従い別途ユーザー
  承認を得る）。

比較演算の bool dtype 出力（§2.4 表の該当行）は #1634 の対象外のまま
（f32 の `0.0`／`1.0` 出力。`docs/scalar-op-dispatch-design.md` §10）。

## 追補（イシュー #1776）

§2.2 表の「花形インデックス `x[idx]`」（`Gather`）・「`scatter`/
`scatter_add`/`index_put_`」（`Scatter`）行の**うち `gather`／
`scatter`／`scatter_add`／`index_select` は実装済み化**（`index_put_`
自体〈複数軸の花形インデックス書き込み〉は引き続き対象外）。

- `Op::Gather { input, dim, index: Tensor<i32> }`／`Op::Scatter {
  input, dim, index: Tensor<i32>, src, reduce: ScatterReduce }`
  （`tensor-core::ScatterReduce`。`Overwrite`＝`torch.scatter`・
  `Add`＝`torch.scatter_add`）を新設し、`Var::gather`／
  `Var::index_select`（`gather` への薄い委譲。専用 `Op` は持たない）・
  `Var::scatter`／`Var::scatter_add`（共通実装 `scatter_impl` 経由）
  として実装済み。
- `BackendOps::gather`／`scatter`（既定 `Unsupported`）・CPU 参照
  実装（`backend-cpu::gather_scatter`）・ホストフォールバック
  （`autodiff::eval::gather`／`scatter`）まで実装済み。CUDA／Metal
  専用カーネルは #1777／#1778 が対象（現状はホストフォールバック
  経由で機能する。性能最適化はまだ入らない）。
- `scatter_add` は決定的集約順序契約（`index`／`src` を行優先で走査
  し `f64` アキュムレータへ逐次加算。`tensor-core::ScatterReduce`
  doc・`.claude/rules/coding-rust.md`「勾配の長軸縮約」節の先取り
  適用）を持つ。
- `Var` 公開メソッド追加のみで facade 新規公開面はない（既存 `Var`
  再エクスポート経由でそのまま到達可能。`docs/compat-api-scope.md`
  §5 手続きの再適用は不要）。
- `masked_select`（出力 shape が動的）・`index_put_`（複数軸の花形
  インデックス書き込み）・負値インデックスの wrap-around・
  `index.shape()[d] <= src.shape()[d]` のみを要求する PyTorch の緩い
  scatter 制約（本実装は `index.shape() == src.shape()` を要求する
  簡略化版）は引き続き対象外（`docs/autodiff-linalg-design.md` と
  同型の対象外整理。詳細は本文書 §2.2 表・実装計画のスコープ外節を
  参照）。

## 追補（イシュー #1724）

`§2.1`（テンソル生成）に列挙されていた「乱数生成と RNG 契約」の欠落は、
プロセスグローバルな決定的 RNG 契約（`fandhe_ai::manual_seed`。PyTorch
`torch.manual_seed` 相当）を新設したことで**基盤機構のみ実装済み**へ
更新する。設計記録は `docs/rng-global-contract-design.md`。

- 実装した契約: `manual_seed(seed: u64)`（facade 経由）・内部アクセサ
  `tensor-core::rng::with_global_rng`（`autodiff`／`facade` へは非公開。
  #1725 が `randn`／`rand`／`randint` を実装する際の消費側）。
- 既存の個別シード API（`nn::Linear::new(.., seed)` 等）とは完全に独立
  した別機構であり、本イシューはそれらのシグネチャ・挙動を変更していな
  い（独立性は `crates/autodiff/src/nn/linear.rs::tests::
  linear_new_is_unaffected_by_global_manual_seed_state` で機構的に固定）。
- 実際の乱数テンソル生成（`randn`／`rand`／`randint`）自体は #1725 で
  実装済み（下記「追補（イシュー #1725）」参照）。`arange`／`linspace`／
  `eye`／`zeros_like`／`ones_like` も #1726 で実装済み（下記「追補
  （イシュー #1726）」参照）。
- facade 新規公開面: `pub fn manual_seed`（新規）。内部型・アクセサは
  facade へ露出させない（`crates/facade/tests/api_surface.rs::
  facade_does_not_expose_rng_internal_types` で機械検査）。

## 追補（イシュー #1777）

gather／scatter／scatter_add（#1637 で where／masked_fill を実装済みへ
更新した「index 系」欄。#1776 で Op 定義・CPU 参照実装・VJP を実装済み）
の CUDA ネイティブカーネル（`CudaBackendOps::gather`／`scatter`。
`crates/backend-cuda/src/gather_scatter.rs`・`kernels_gather_scatter.rs`）
を実装した。

- **数値契約**: bit 同一（run-to-run 完全一致・CPU 参照実装と bit 完全
  一致）。`scatter_add` は `fandhe_ai_tensor_core::ScatterReduce` doc の
  決定的集約契約（row-major 走査順・`f64` アキュムレータ・最後に 1 回
  だけ `f32` downcast）を CUDA カーネル内で再現する。
- **fail-closed 検査**: `input`／`index`／`src` の shape 再検査
  （`gather_out_shape`／`scatter_out_shape`）・`index` 値の範囲検査は
  ホスト側でデバイス初期化より前に行い、CPU 実装と同一の
  `BackendError::ShapeMismatch(ShapeError::IndexOutOfRange)` を返す。
- **性能最適化は対象外**（`O(numel_out × index_shape[dim])` 走査。並列化
  方式の高度化は別イシューのスコープ）。
- **GB10 実機実測は未実施**（本エージェント実行環境に CUDA 実機なし。
  実行コマンドは `crates/backend-cuda/tests/gather_scatter_parity.rs`
  冒頭コメント参照）。
- facade 新規公開面なし（既存 `Var::gather`／`scatter`／`scatter_add`／
  `index_select` の再エクスポート経由でそのまま CUDA バックエンドへ
  到達する）。Metal 専用カーネルは #1778 が残対象。
#### #1697 の追補（`backend-cpu` の `TypedOps<f64>` 実装）

- 319 行目の `float64` 行の「部分」記載は本イシューにより CPU バックエンド限定で解消: `crates/backend-cpu` が `TypedOps<f64>`（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` の 8 演算）を実装し、`CpuBackendOps::typed_ops_f64()` accessor 経由で到達可能になった（`docs/backend-dtype-dispatch-design.md` §10）
- `Var`／`Tape`／facade は本イシューの対象外のまま不変（表本体の「XL」見積り自体は #1650／#1651〈CUDA／Metal〉・`Var`／`Tape` 昇格の残作業を含むため据え置く）。CUDA（#1650）・Metal（#1651）は未実装のまま

## 追補（イシュー #1778）

イシュー #1776 の追補で「CUDA／Metal 専用カーネルは #1777／#1778 が対象」
としていた Metal 側を実装した。

- `MetalBackendOps::gather`／`scatter`（`crates/backend-metal/src/
  gather_scatter.rs`・`shaders/gather_scatter.metal`）を新設し、
  `BackendOps::gather`／`scatter` の Metal 実装として結線済み
  （CUDA 側は #1777 で実装済み）。
- 数値契約: gather・scatter(Overwrite) は丸めを伴わない純粋コピー・
  上書きのため CPU 参照実装（`backend-cpu::gather_scatter`）と bit
  完全一致。scatter(Add) は `.claude/rules/coding-rust.md`「勾配の
  長軸縮約」節と同じ精度規律（binary64 逐次加算の 64bit 整数ソフト
  ウェアエミュレーション。`gemm.metal::bias_f64_*`／`layer_norm.metal::
  ln_f64_*` と同型の意図的複製）で CPU 参照実装と bit 完全一致（NaN
  のみクラス一致）。
- 出力定常（output-stationary）方式のカーネル（1 スレッド = 1 出力
  位置）を採用し、CPU の行優先（row-major）全走査と同じ集約順序に
  なることをホスト側逐語モデル（`gather_scatter_model.rs`）で
  Linux 実行可能に検証した。
- facade 新規公開面なし（#1776 と同じく `Var` 再エクスポート経由で
  到達）。性能最適化（並列縮約木・専用形状特化等）は対象外のまま
  （`.claude/rules/out-of-scope-tracking.md` 対象）。
- 実機（Apple M4 Max）での `tests/gather_scatter_parity.rs`
  （`#[ignore]`）実行は本エージェント実行環境に Apple Silicon 実機
  がないため未実施のまま Mac セッションへ申し送る（Linux 実行可能な
  ホスト逐語モデル bit 一致テスト・ソース証跡テスト・
  `aarch64-apple-darwin` クロス型検査は本 PR で完了済み）。

#### #1703 の追補（`backend-cuda` の `TypedOps<f64>`／`TypedOps<f16>` 実装）

- 319 行目の `float64` 行: CUDA バックエンドは本イシューでも `Unsupported` のまま（8 演算すべて driver 非接触の fail-closed。性能上の目的がないため実装対象外。`docs/backend-dtype-dispatch-design.md` §11）。CPU 限定の解消（#1697）は不変
- 320 行目の `float16` 行: CUDA バックエンドは本イシューにより一部到達可能になった——`crates/backend-cuda` が `TypedOps<half::f16>`（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` の 8 演算）を実装し、`CudaBackendOps::typed_ops_f16()` accessor 経由で到達可能。`gemm` は既存 `CudaGemmAuto::run_f16`（`mma.sync` 優先の Tensor Core 自動選択経路）への結線、残り 7 演算は f32 昇格→既存カーネル→1 回丸め（`docs/backend-dtype-dispatch-design.md` §11）
- `Var`／`Tape`／facade は本イシューの対象外のまま不変。bf16（#1704）・Metal（#1705）は未実装のまま

#### #1698 の追補

`float16` 行（319〜320 行目）のスナップショット本文は不変のまま、CPU
バックエンド限定で `TypedOps<half::f16>` が実装され `Tensor<f16>` の
`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` が CPU 経由で到達
可能になった（`crates/backend-cpu/src/typed_f16.rs`。イシュー #1698）。
CUDA（#1650）・Metal（#1651）は未実装のまま。`Var`／`Tape`／VJP・facade
公開面（`Tensor<f16>` を受け取る facade API）は引き続き未接続で、本表の
「未実装（欠落側）」列の評価（`Var` レベルの mixed precision）は変わらない。

#### #1699 の追補

`§2.12`（float64／float16・bfloat16。320 行目）の bfloat16 行に関して、
CPU バックエンド限定で `fandhe_ai_tensor_core::TypedOps<half::bf16>` が
`CpuBackendOps` に実装され、`BackendOps::typed_ops_bf16()` accessor
経由で bf16 の 8 演算（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／
`sum`／`max`）が CPU 上で実行可能になった（`crates/backend-cpu/src/
typed_bf16.rs`。設計 `docs/backend-dtype-dispatch-design.md` §11）。

- 実装方式は既存 f32 カーネルの再利用（bf16→f32 昇格 → f32 カーネル
  → f32→bf16 丸め）であり、新規カーネルは追加していない。
- `facade`（唯一の公開 API 面）への新規公開面はない。`Var`／`Tape`／
  VJP・resident 系・カーネル融合は対象外のまま。
- f16（#1698）・CUDA bf16（#1704）・Metal bf16（#1706）は本イシューで
  は触れていない。表本体のスナップショット（対象 HEAD `097bff19`）・
  必要工数見積り（XL）は変更しない（本追補は snapshot 後の部分実装差分
  の記録）。

## #1704 の追補

`float64`／`float16`／`bfloat16` 行（319〜320 行目）のスナップショット本文
は不変のまま、CUDA バックエンド限定で `TypedOps<half::bf16>` が実装され
`Tensor<bf16>` の `gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`
が CUDA 経由で到達可能になった（`crates/backend-cuda/src/typed_bf16.rs`。
cudarc 0.19.8 が `half::bf16` の `DeviceRepr`／`ValidAsZeroBits` を実装
していることを確認したうえでの実装。イシュー #1704・
`docs/backend-dtype-dispatch-design.md` §12）。CUDA `TypedOps<f64>`／
`TypedOps<f16>`（#1703）・Metal bf16（#1651）は未実装のまま（CPU bf16 は
#1699・PR #1794 で実装済み・origin/main マージ済み。上記「#1699 の追補」参照）。`Var`／`Tape`／VJP・facade 公開面（`Tensor<bf16>` を受け
取る facade API）は引き続き未接続で、本表の「未実装（欠落側）」列の評価
（`Var` レベルの mixed precision）は変わらない。

**追補（イシュー #1715）**: 上記スナップショット時点で「bmm」（#2.6）と
記載されていた rank≥3 の行列積（PyTorch `torch.matmul`／`bmm` 相当）が
実装済みへ更新された。`Var::matmul` が rank≥2（先頭 rank−2 軸を NumPy
互換ブロードキャストするバッチ次元）を受理するようになり、
`fandhe_ai_tensor_core::BackendOps` に `gemm_batched`／
`gemm_batched_fp32_strict`（既定は per-batch `gemm`/`gemm_fp32_strict`
への合成。非破壊拡張のデフォルトメソッド）を追加し、`backend-cpu` が
専用オーバーライド（`CpuBackendOps::gemm_batched`。既存 2 次元
`gemm`/`gemm_into_slice` と bit 同一）を持つ。facade 新規公開面なし
（既存 `Var` 再エクスポート経由でそのまま到達可能）。CUDA は既定合成
実装のまま（機能的に到達可能・専用バッチカーネルは #1716）。
`einsum`（rank≥3 matmul を伴う batch 添字縮約。#1600 が未実装として
いた対象）は本イシューでは対象外のまま残る。

**追補（イシュー #1717）**: Metal に専用オーバーライド
（`MetalBackendOps::gemm_batched`／`gemm_batched_fp32_strict`）が
実装済みになった。既定合成実装（バッチをほどいて `batch_len` 回
`self.gemm` を呼ぶ——各呼び出しが独自に upload・`dispatch_auto`〈内部
同期〉・download する）と異なり、正規化済みオペランドを 1 回ずつ
upload し、バッチごとの GEMM を `gemm::MetalGemm::
encode_strided_bias_act_prepared_with_c_offset`（`gemm_fp32_strict_
into`〈#1555〉・`linear_forward_device`〈#1216〉が確立した encode-only
パターン。`gemm.rs`／shader 自体は無変更）で 1 つのコマンドバッチへ
encode するだけで積み、`download` 1 回だけが GPU 完了を待つ「バッチ
ループ方式」にする。rank≥3 は classic strided カーネル
（`gemm_tiled_bias_act`）を経由するため per-batch `gemm`
（`dispatch_auto` = `gemm_simdgroup_tiled`／split-K）とは bit 同一を
主張せず、REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満）を受け入れ契約とする。本経路は `dispatch_auto`／
`tile::select_route_for_device` を経由しないため split-K 実行時
トグル（`crate::split_k_runtime`。既定 `true`）の状態に依存しない。
facade 新規公開面なし。M4 Max 実機実測は本エージェント実行環境に
Apple Silicon 実機がないため未実施のまま Mac セッションへ申し送る。

## #1636（#1707〜#1709）の追補

Metal バックエンドの `ScalarOp`（`ScalarUnaryOp`／`ScalarBinaryOp`。
`tensor-core::scalar_op`）カーネルが CUDA（#1700〜#1702）と同等の範囲まで
実装済みになった。#1707（算術系 `Sub`／`Div`／`Pow`／`Sqrt`）・#1708
（超越関数系 `Neg`／`Abs`／`Log`／`Log2`／`Log10`／`Sin`／`Cos`／`Tan`）に
加え、#1709 で比較演算 6 種（`Gt`／`Ge`／`Lt`／`Le`／`Eq`／`Ne`）＋`Clamp`
（初のペイロード付き unary kind）を実装し、親イシュー #1636 の対象 kind
をすべて実装完了した（`crates/backend-metal/src/scalar_op_source.rs`。
MSL テンプレート生成＋`context_cache` のプロセス内キャッシュ）。
facade 新規公開面はない（既存 `BackendOps::scalar_unary`／`scalar_binary`
の実装追加のみ）。`Var` 公開メソッド・facade 配線は引き続き #1593／#1595
の担当。M4 Max 実機での parity・キャッシュ非分裂テスト（`#[ignore]`）は
本イシューの実行環境に Apple Silicon 実機がないため未実施のまま Mac
セッションへ申し送り。

## #1652 の追補（ONNX import の facade 公開可否の設計判断）

§1.9・347 行目「ONNX import」のスナップショット本文（`onnx-interop::onnx::interp`
が facade から到達不可であること）は不変のまま、facade 公開の可否を設計判断として
記録した（`docs/facade-onnx-import-exposure-decision.md`。イシュー #1652）。

- 判断: facade へ公開する方針は「薄いラッパー型」（案 B。`prost`／`half::f16` を
  公開面に出さない専用型）として推奨する。DDP（#1628）・量子化（#1627）と異なり、
  ONNX import 公開は正本 spec の除外事項に従属していない。
- ただし facade は crates.io 公開クレートであり非公開クレートへの通常依存を
  持てないため、「facade から公開する」は `onnx-interop` 自体を 7 クレート目として
  crates.io へ公開することと構造的に等価になる。この publish 承認（命名確定・
  `release-all.yml` の `RELEASE_CRATES` 変更を含む）は 2026-09-12 の facade 公開面
  拡張の承認範囲には含まれない別個の事項であり、現時点では未取得。
- 現状（本追補時点）は非公開・未実装のまま変わらない。実装（facade ラッパー・
  `api_surface.rs` 拡張）は publish 承認取得後の別 issue へ引き継ぐ。
- #1775（ONNX export の facade 公開）・#1754（safetensors save／load の facade
  再公開）は同じ publish 前提を共有するため blocked のまま close しない
  （`docs/facade-onnx-import-exposure-decision.md` §6.2）。

## 追補（イシュー #1725）

上記「追補（イシュー #1724）」が未実装のまま残していた実際の乱数テンソル
生成（`randn`／`rand`／`randint`。`§2.1`「乱数生成と RNG 契約」の残対象）
を実装した。設計・実装記録は `docs/rng-global-contract-design.md` §10。

- 実装した API: `fandhe_ai::{randn(shape), rand(shape), randint(low,
  high, shape)}`（実体は `tensor-core::rng`。`autodiff` は素通しのみ）。
  `manual_seed` が設定したプロセスグローバル決定的 RNG をホスト側だけで
  消費し（`BackendOps` 非経由）、返る `Tensor` は既存の `Tape::var` で
  任意デバイスへアップロードする（#1602 本文の設計方針どおり）。
- `Op`／`BackendOps`／VJP は追加していない（乱数生成は微分不能な葉値
  であり `torch.randn` にも勾配は無いため。受け入れ条件テンプレの
  「Op／BackendOps／VJP」項は本イシューでは非適用）。
- `randint` の dtype は `i32`（PyTorch 既定の int64 とは異なる意図的な
  差異。index／targets 型契約に合わせた）。
- facade 新規公開面: `pub fn randn`／`rand`／`randint`（新規）・
  `pub use ...::RngError`（`randint` の戻り値型。1 行の再エクスポート）。
  内部型・アクセサ（`Xorshift64Star`・`with_global_rng`）は引き続き
  facade へ露出させない（`facade_does_not_expose_rng_internal_types` で
  機械検査）。
- 対象外: `arange`／`linspace`／`eye`／`zeros_like`／`ones_like`
  （#1726 で実装済み。下記「追補（イシュー #1726）」参照）・
  `randn_like`／`rand_like`／`normal`／`uniform_`／`bernoulli`／
  `multinomial`／`randperm`・非グローバル RNG（`torch.Generator` 相当）・
  CUDA／Metal デバイス側乱数カーネル・`nn::Dropout`（#1603）・
  `randint` の int64 版。

## 追補（イシュー #1726）

上記「追補（イシュー #1725）」が対象外としていた決定的テンソル生成
（`arange`／`linspace`／`eye`／`zeros_like`／`ones_like`。`§2.1`「乱数
生成と RNG 契約」の残対象）を実装した。設計・実装記録は
`docs/rng-global-contract-design.md` §11。

- 実装した API: `fandhe_ai::{arange(start, end, step), linspace(start,
  end, steps), eye(n), zeros_like(like), ones_like(like)}`（実体は
  `tensor-core::creation`。`autodiff` は素通しのみ）。`randn`／`rand`／
  `randint` と同じくホスト側だけで完結し `BackendOps` を経由しない。
- `Op`／`BackendOps`／VJP は追加していない（生成結果は微分不能な葉値
  であり `torch.arange` 等にも勾配は無いため。受け入れ条件テンプレの
  「Op／BackendOps／VJP」項は本イシューでも非適用。#1725 と同じ根拠）。
- `arange`／`linspace` の数値契約（`f64` 中間計算・PyTorch 2 分割方式）
  はプラットフォーム横断で bit 同一（`docs/rng-global-contract-design.md`
  §11 参照）。
- facade 新規公開面: `pub fn arange`／`linspace`／`eye`／`zeros_like`／
  `ones_like`（新規）・`pub use ...::CreationError`（`arange`／
  `linspace` の戻り値型。1 行の再エクスポート）。
- 対象外: 1 引数／2 引数の `arange` 便宜版・i32／i64 版 `arange`・
  長方形 `eye(rows, cols)`・`full_like`／`empty_like`／`randn_like`／
  `rand_like`・`logspace`・`Var::zeros_like` 等の `Var` 側メソッド・
  CUDA／Metal デバイス側の生成カーネル・`compat::array`／`Sequential`
  の変更。

**#1773 追記（ONNX export の op 逆マッピング）**: `onnx-interop` 内部
（`crate::onnx::export_ops`）に `interp.rs` 対応 22 op すべての逆マッピング
（`ExportOp` -> `NodeProto`。op_type・入力順・属性 name/type/既定値）を実装した
（`docs/onnx-export-op-mapping.md`）。`build_model_proto` は組み立て前に
`check_exportable`（layer B。allowlist・既定 opset の fail-closed 検査）を
経由するよう変更済み。`onnx-interop` は crates.io 非公開クレートであり facade
新規公開面はなし（#1775 の判断は本追補の対象外のまま変わらない）。

**#1774 追記（ONNX import→export→import の roundtrip 構造一致テスト）**:
`crates/onnx-interop/tests/onnx_export_roundtrip.rs` に、import -> export ->
import の総合 roundtrip（構造一致・bit 同一）・`interp::run` 結果の bit 同一・
未対応 op を含むモデルの fail-closed（`ExportError::UnsupportedOp`）を固定する
テストを追加した（`docs/onnx-export-op-mapping.md` §6）。テスト追加のみで
`onnx-interop` 本番コード（`export.rs`／`export_ops.rs`／`graph.rs`／
`interp.rs`）は無変更・facade 新規公開面はなし（#1775 の判断は本追補の対象外の
まま変わらない）。

**#1775 追記（ONNX export の facade 公開）**: 設計判断を
`docs/facade-onnx-export-exposure-decision.md` として記録した。#1652（ONNX
import 公開可否）と同じ publish 前提（`onnx-interop` の crates.io 公開という
ユーザー承認未取得の別個の事項）を共有するため、上記「ONNX export」行
（346 行目）のスナップショット本文は不変のまま、facade 公開は段階 0・
blocked のまま close しない（`docs/facade-onnx-import-exposure-decision.md`
§6.2）。本 issue の唯一のコード変更は
`crates/facade/tests/api_surface.rs` への負の guard テスト 2 件
（`facade_does_not_depend_on_unpublished_onnx_interop`／
`facade_sources_do_not_reference_onnx_interop`。facade が非公開クレート
`onnx-interop` へ依存しないことの機械的固定）であり、facade 新規公開面は
なし。

## #1705 の追補

`float64`／`float16` 行（319〜320 行目）のスナップショット本文は不変のまま、Metal バックエンド限定で以下が確定した（イシュー #1705・`docs/backend-dtype-dispatch-design.md` §14）。

- **`float64`**: Metal は `TypedOps<f64>` を実装せず（accessor `typed_ops_f64()` は既定 `None` のまま）、恒久 `Unsupported` として確定した。MSL に `double` 型が存在せず GEMM 全体を `f64` 精度で動かす手段が構造的にないため（CUDA・CPU のような「実装したうえで `Unsupported` を返す」形ではなく、accessor 自体を `None` のまま維持する capability 不在の明示）
- **`float16`**: Metal バックエンド限定で `TypedOps<half::f16>` が実装され、`Tensor<f16>` の `gemm`（既存 `MetalGemm::dispatch_f16_auto_unverified` への内部結線）・`add`／`mul`／`relu`／`exp`／`tanh` が Metal 経由で到達可能になった（`crates/backend-metal/src/typed_f16.rs`）。`sum`／`max` は Metal f32 reduction カーネル自体が未実装のため `Unsupported` を継承する（f32 版が実装されれば自動的に有効化される設計）
- CUDA `TypedOps<f64>`／`TypedOps<f16>`（#1703）・bf16（#1706）は本イシューでは触れていない。`Var`／`Tape`／VJP・facade 公開面は引き続き未接続で、本表の「未実装（欠落側）」列の評価は変わらない

## #1706 の追補

`float64`／`float16`／`bfloat16` 行（319〜320 行目）のスナップショット本文
は不変のまま、Metal バックエンド限定で `TypedOps<half::bf16>` が実装され
`Tensor<bf16>` の `gemm`／`add`／`mul`／`relu`／`exp`／`tanh` が Metal 経由で
到達可能になった（`sum`／`max` は Metal f32 `BackendOps` が GPU カーネル
未実装のため `Unsupported` のまま。`crates/backend-metal/src/
typed_bf16.rs`。イシュー #1706・`docs/backend-dtype-dispatch-design.md`
§15）。

- (a) `TypedOps<bf16>` の実装可否と (b) MSL `bfloat`／`simdgroup_bfloat8x8`
  の実機可用性は独立の問題であることが判明し、(a) はホスト側変換＋既存
  f32 経路委譲方式（CUDA #1704・CPU #1699 と同型）で MSL `bfloat` の
  可用性に依存せず実装できた。(b)（デバイス常駐ネイティブ bf16 経路の
  実現可能性）はコンパイルプローブ（`crate::typed_bf16_probe_diag_tests`。
  全 `#[ignore]`・非 gating）へ切り出し、本エージェント実行環境に
  Apple Silicon 実機がないため未実測のまま Mac セッションへ申し送る。
- CPU bf16 は #1699・CUDA bf16 は #1704 で実装済み・origin/main マージ
  済み。3 バックエンドとも bf16 の主要 6 演算（`gemm`／`add`／`mul`／
  `relu`／`exp`／`tanh`）に到達可能になった。
- facade 公開面への新規追加はない。`Var`／`Tape`／VJP は引き続き未接続で、
  本表の「未実装（欠落側）」列の評価（`Var` レベルの mixed precision）は
  変わらない。

## 追補（イシュー #1733）

§2.2「`torch.topk`/`sort`」行はスナップショット時点の記述（「なし」）の
まま不変とし、以下を追記する: `Var::sort`／`argsort`／`topk`（`torch.sort`／
`torch.argsort`／`torch.topk` 相当）を実装済み化した（`tape::Op::Sort`／
`Op::Topk`〈`push_eager` の非融合 eager 演算・`Op::Gather` と同型の最小
保持方針〉・CPU 参照実装〈`backend-cpu::sort_topk`〉・ホスト参照実装
〈`fandhe_ai_autodiff::eval::sort`／`topk`〉・scatter ベース VJP〈`values
= gather(input, dim, index)` と数学的に同一のため `Op::Gather` と同じ
`scatter_add` 式を再利用〉）。同値（ties）は `descending` の値に関わらず
元インデックス昇順・NaN は任意の非 NaN より大きい・±0 は同値、という
順序契約（`fandhe_ai_tensor_core::BackendOps::sort` doc）を CPU 参照実装・
ホストフォールバックの両方で固定し、`crates/backend-cpu/tests/
sort_topk_parity.rs` で bit 完全一致を回帰確認した。`Var::argsort` は
`Op::Sort` を記録せず（非微分演算）、`sort` と同じ検査・フォールバック
経路を通した `index` のみを返す。CUDA／Metal 専用カーネルは既定
`Unsupported`（ホストフォールバックで機能する）のまま #1741 へ引き継ぐ。
facade 新規公開面なし（既存の `Var` 再エクスポート経由）。`sorted=False`
の topk・負 `dim`・非安定ソート（本実装の同値タイブレークとは異なる
意味論）・`k` の `Var` 化は対象外のまま。

## #1710 の追補

Var 演算欠落リストの `sub`／`div`／`pow`／`sqrt` 行（スナップショット本文
は不変）が実装済みになった。`Var::sub`／`div`／`pow`（`ScalarBinaryOp`。
`add`／`mul` と同じ NumPy 互換ブロードキャスト）・`Var::sqrt`
（`ScalarUnaryOp::Sqrt`）は #1634 の汎用 dispatch 機構（`crates/autodiff/
src/var.rs::scalar_unary`／`scalar_binary`）への薄い委譲として実装され、
CPU／CUDA／Metal 3 バックエンドの `BackendOps::scalar_unary`／
`scalar_binary`（既に #1700・#1707 等で実装済み）経由で到達する
（`docs/scalar-op-dispatch-design.md`）。facade 新規公開面はない（既存
`Var` 再エクスポート経由）。CUDA／Metal 実機での facade parity 実測は
本エージェント実行環境に実機がないため未実施のまま申し送る。
`pow_scalar`（スカラー指数版）は対象外のまま（`log`／三角関数／`abs`／
`neg` は #1711、`clamp`／比較演算は #1712 で実装済み。下記参照）。

## #1711 の追補

§2.4 の「`sin`/`cos`/`tan`」「`log`/`log2`/`log10`」行（221・223 行目）の
スナップショット本文は不変のまま、以下が実装済みになった（イシュー
#1711）。

- `Var::log`／`log2`／`log10`／`sin`／`cos`／`tan`／`abs`／`neg` の 8 個の
  `pub fn` を `crates/autodiff/src/var.rs` に追加し、いずれも既存の
  `Var::scalar_unary`（`ScalarUnaryOp` 汎用 dispatch。#1634 で実装済み）
  への薄い委譲とした。3 バックエンド（CPU／CUDA／Metal）のカーネル・
  VJP 係数・CPU 参照実装は #1634／#1635／#1636（#1707〜#1709）で既に
  実装済みのため、本 issue の新規実装は `Var` 公開メソッドと facade
  到達経路の配線のみ（新規カーネル・`Op` 追加なし）。
- facade（`crates/facade/src/lib.rs`）への新規 `pub use`／`pub fn` は
  追加していない。既存の `pub use fandhe_ai_autodiff::Var` 再エクスポート
  経由でそのまま到達可能になる。
- `abs`／`neg` は §2.4 表に専用行がないが、`docs/compat-api-scope.md`
  §1.2「要素演算」行の実装 issue（#1592・#1593）の分解対象に含まれる
  ため同行の範囲として扱った（レビューで異論があれば分離して除外できる
  独立テスト単位で実装済み）。
- テストは `crates/autodiff/src/grad.rs`（Tape 経由 forward／backward・
  `log` の非正定義域〈`-inf`／`NaN`〉・`abs` の劣勾配・`neg` の `-0.0`
  符号ビット反転）と `crates/facade/tests/
  scalar_unary_transcendental_backend_parity.rs`（CPU vs NaiveOps の
  REQ-2 複合判定・`matmul → log → sum` 合成勾配・`#[ignore]` の
  Metal／CUDA 実機比較）に追加した。CUDA（DGX Spark GB10）・Metal
  （Apple Silicon）実機での facade parity テストは、本実装エージェント
  の実行環境に実機への到達手段がないため未実測のまま Mac／GB10 セッション
  へ申し送る。
- `sub`／`div`／`pow`／`sqrt`（#1710）・`clamp`／比較演算（#1712）は本
  issue の対象外のまま。

## #1712 の追補

Var 演算欠落リストの `clamp`／比較演算（`gt`／`ge`／`lt`／`le`／`eq`／
`ne`）行（スナップショット本文は不変）が実装済みになった。いずれも
#1634 の汎用 dispatch 機構（`Var::scalar_unary`／`scalar_binary`）への
薄い委譲として実装され、CPU／CUDA／Metal 3 バックエンドの
`BackendOps::scalar_unary`／`scalar_binary`（`ScalarUnaryOp::Clamp`・
`ScalarBinaryOp::{Gt,Ge,Lt,Le,Eq,Ne}`。#1634／#1635／#1636 で既に実装
済み）経由で到達する。facade 新規公開面はない（既存 `Var` 再
エクスポート経由）。

出力は f32 の `0.0`／`1.0`（bool dtype 出力・`Tensor<bool>` は #1613
〈OPEN〉の対象で本イシューの範囲外。`where_cond`／`masked_fill` の
bool 引数との直接合成も #1613 待ち）。VJP は両入力とも常にゼロ勾配
（比較演算は局所的に階段関数のため微分不可能）。CUDA／Metal 実機での
facade parity 実測は本エージェント実行環境に実機がないため未実施の
まま申し送る。


#1723 で §2.5「縮約」`var`／`std`・`norm`（L1/L2）行を実装済み化
（`Var::var`／`std`／`norm_l1`／`norm_l2`。既存 `sum`／`max` と同じ
`dim: Option<usize>` シグネチャ〈`var`／`std` は `correction` 引数を
追加〉。当初は `Op::Var`／`Op::VectorNorm` の 2 新規 Op・`BackendOps::
var`／`vector_norm`〈既定 `Unsupported`〉・CPU 参照実装〈`backend-cpu::
reduction`〉・ホストフォールバック〈`eval::var_along`／
`vector_norm_along`〉のみで `std` を `Var::var(..).sqrt()`（新規 `Op`
を追加しない合成）として実装していたが、`Op::Var` の forward 値が
分散を `f32` へ downcast してから `Op::Sqrt` に渡すため、真の分散が
`f32` の範囲を超える極端な入力（例 `[-1e20, 1e20]`）で `std` 自体は
`f32` で表現可能なはずが `inf` になる問題があったため（codex-review
P2 指摘。PR #1826 レビュー是正）、専用の `Op::Std` ノードを追加し
forward・backward とも `f64` の分散を経由してから最後に 1 回だけ
`sqrt` を計算する方式へ変更した（`eval::std_along`／`grad::std_vjp`。
現在は `Op::Var`／`Op::VectorNorm`／`Op::Std` の計 3 新規 Op。
`BackendOps::std` は設けず常にホストフォールバックを使う——`var`／
`vector_norm` とは異なり `BackendOps` 対応メソッドの非破壊拡張余地を
残したまま、現時点ではホスト参照実装のみに限定する判断）。数値契約は
`.claude/rules/coding-rust.md`「正規化統計は要素を先に `f64` へ昇格
してから二乗し、最後に 1 回だけ `f32` へ downcast する」契約に従う
（`var` は平均→二乗和の 2 段計算・`norm_l2` は二乗和→`sqrt`）。`std`
の勾配（`Op::Std` の VJP）は縮約対象が全て同値の定数列（`std == 0`）
の要素でゼロ勾配へ明示的にマスクする——PyTorch `torch.std`
backward（`std_backward`。`FunctionsManual.cpp`）の
`masked_fill_(result == 0, 0)` と一致する規約であり、`var==0` から
単純に `0.0 / 0.0 = NaN` が伝播する `Var::var(..).sqrt()` 合成とは
意図的に異なる（PR #1826 レビュー是正。`crates/autodiff/src/var.rs`
`Var::std` doc「数値規約」参照）。CUDA／Metal
専用カーネルは本 issue のスコープ外で既定 `Unsupported` のまま
ホストフォールバック経由のみ（後続 issue 候補・多軸／`keepdim` 版・
`norm(ord, dim)` の facade 公開〈Tier 1 未列挙〉も同様にスコープ外。
facade 新規公開面なし——既存 `Var` 再エクスポート経由。CUDA／Metal
実機での facade parity 実測は本エージェント実行環境に実機がないため
未実施のまま申し送る）。

## #1755 の追補

#1755 で `one_hot`（`torch.nn.functional.one_hot`／`tf.one_hot` 相当。
**非微分演算**）が実装済みになった（`Var::one_hot`・`Op::OneHot`。
VJP は明示ゼロ〈`Gradients::get` で観測可能な形で入力へゼロ勾配を流す。
寄与なしではない〉）。CPU（`backend-cpu::gather_scatter::one_hot`）・
CUDA（`kernels_gather_scatter::ONE_HOT_F32`。座標展開・ストライドを
使わない `row = idx / num_classes`・`c = idx % num_classes` の単純な
整数除算・剰余のみ）・Metal（`shaders/gather_scatter.metal::
one_hot_f32`。CUDA 版と同型の設計）の 3 バックエンドとも専用カーネルを
実装済み（既定 `Unsupported` フォールバックのホスト参照実装
`eval::one_hot` も整備済み）。facade 新規公開面はない（既存 `Var` 再
エクスポート経由）。CUDA・Metal 実機での facade parity 実測（`#[ignore]`
分離済み）は本エージェント実行環境に実機がないため未実施のまま
申し送る。

## #1734 の追補

`torch.unique(input, sorted=True)` の values のみを実装済み化した
（`Var::unique`。イシュー #1734・`docs/unique-facade-exposure-decision.md`）。
他の Var 演算と異なり非微分演算（勾配を持たない）であり、出力形状が
入力値に依存して動的に決まるため `Var`（tape ノード・静的 shape）では
なく **detached な `Tensor<f32>`** を返す（`Op` を tape に記録しない）。
CPU（参照実装）・CUDA／Metal（ビットニックソート方式。整数
compare/swap のみで浮動小数点演算を含まないため決定的）の 3
バックエンドとも bit 完全一致契約。CUDA／Metal 実機（DGX Spark
GB10・Apple Silicon）は本実装環境に到達手段がなく `#[ignore]`
テストとして未実測のまま GB10／Mac セッションへ申し送る。
`return_inverse`／`return_counts`／`dim` 指定・`sorted=false`・
`unique_consecutive`・GPU 側 prefix-sum 圧縮は対象外のまま
（decision doc §5／§6）。facade 新規公開面なし（既存 `Var`
再エクスポート経由）。


## #1713 の追補

`Var::gelu`／`gelu_tanh`／`softplus`（GELU 誤差関数版・tanh 近似版・
Softplus）を実装済み化した。`ScalarUnaryOp::Gelu`／`GeluTanh`／
`Softplus`（#1634 で enum・dispatch・CPU 参照実装・VJP まで実装済み）
への薄い委譲。CUDA（`erff`／`tanhf`／`log1pf`／`expf`）・Metal（自作
`scalar_erf_f32`〈A-S 7.1.26 の `float` 精度複製〉・
`metal::precise::tanh`・自作 `scalar_log1p_f32`〈Kahan 補正式〉・
`metal::precise::exp`）のカーネル実装まで本 issue で追加した（超越関数
のため REQ-2 統一複合判定のみで検証・bit 同一は主張しない）。
`nn::activation::Gelu`／`GeluTanh`／`Softplus`（`Module` 実装込み）も
追加。facade 新規公開面なし（既存 `Var` 再エクスポート経由のみ）。
`compat::Sequential::add_gelu`／`add_gelu_tanh`／`add_softplus` 等の
builder はユーザー承認待ちで対象外のまま。CUDA／Metal 実機での facade
parity テストは本実装エージェントの実行環境に実機への到達手段がない
ため未実測のまま Mac／GB10 セッションへ申し送る。

**#1714 追補（SiLU・LeakyReLU・ELU・Hardswish）**: 本調査時点で §2.4／
§2.7 が欠落と判定した ReLU 系派生活性化のうち、SiLU／LeakyReLU／ELU／
Hardswish を実装済み化した（GELU／Softplus は #1713 で別途実装済み。
上記「#1713 の追補」参照）。
`Var::silu`／`leaky_relu`／`elu`／`hardswish`（`ScalarUnaryOp::Silu`／
`LeakyRelu`／`Elu`／`Hardswish` への薄い委譲。`#1592`／`#1634` が敷いた
`ScalarUnaryOp` 汎用 dispatch 基盤の上）・`nn::activation::{Silu,
Hardswish, LeakyRelu, Elu}`（`Module` 実装込み）・
`compat::Sequential::add_silu`／`add_hardswish`／`add_leaky_relu`／
`add_elu`を追加した。CUDA（`kernels_scalar_op.rs`）・Metal
（`scalar_op_source.rs`）の専用カーネルも実装済み（`LeakyRelu`／
`Hardswish` は選択・算術のみで bit 同一想定、`Silu`／`Elu` は超越関数
〈`exp`〉を含むため REQ-2 統一複合判定のみ）。`LeakyRelu`／`Elu` は本
実装で初めて 1 引数ペイロード（CUDA／Metal 双方に
`UnaryPayload::One`）を持つ unary kind として追加した。Metal の `Elu`
は MSL に `expm1` 相当が存在しないため、`exp`／`log` から桁落ちなく
再構成する自作ヘルパー `fai_expm1_f32`（`u = exp(x)` を計算し
`u == 1.0` なら `expm1(x) ≈ x`、`u == 0.0`〈underflow〉なら `-1.0f`、
それ以外は `(u - 1) * x / log(u)` で再構成）を使う。単純な
`exp(x) - 1.0f` による代替は `x=-1e-8, alpha=1e8` のようなゼロ近傍・
大 `alpha` の入力で桁落ちし REQ-2 統一複合判定を満たさなかったため
不採用（PR #1825 codex-review P1 是正）。ホスト `f32::exp_m1`（正確な
libm 実装）とは一般に bit 同一にならず、超越関数系と同じく REQ-2
統一複合判定のみで検証する（`scalar_op_source.rs` モジュール doc
「`Elu` の `expm1` 非対応」参照）。facade `compat::Sequential::add_*` 4
件の新規公開面は親 #1595 コメント（2026-09-12 ユーザー承認）に基づく
`docs/compat-api-scope.md` §5 経路 2 の適用。CUDA（DGX Spark GB10）・
Metal（Apple Silicon）実機での parity テストは、本実装エージェントの
実行環境に実機への到達手段がないため未実測のまま Mac／GB10 セッションへ
申し送る。

## #1719 の追補

§2.5「縮約」の `sum(dim)`／`mean(dim)`／`max(dim)`/`min(dim)` 行
（スナップショット本文は不変）のうち、複数軸対応・`keepdim` 対応・
`mean` 新設部分が実装済みになった。

- `Var::sum_dims(&dims, keepdim)`／`max_dims(&dims, keepdim)`（複数軸・
  `keepdim` 対応。PyTorch `sum(dim=[...], keepdim=)`／`amax(dim=[...],
  keepdim=)` 相当）・`Var::mean(dim)`／`mean_dims(&dims, keepdim)`（新設。
  PyTorch `mean(dim, keepdim)` 相当）を追加した。
- 実現方式（`crates/autodiff/src/reduce_dims.rs`。新規 `pub(crate)`
  モジュール）: 縮約対象軸が単一・全軸のいずれかなら既存
  `Var::sum(Option<usize>)`／`max(Option<usize>)` へ無加工で直接委譲
  する（`sum_dims(&[d], false)` が `sum(Some(d))` と bit 同一になる
  契約はこの分岐が担う）。複数軸・非全軸の場合は `Var::permute`
  （kept 軸 → reduced 軸の順。恒等順列なら省略）→ `Var::contiguous`
  （非 contiguous のときのみ実体化）→ `Var::reshape`（reduced 軸を 1 軸
  へ併合）で単一軸縮約へ帰着させる（逐次〈軸ごと〉縮約にしない理由は
  同モジュール doc 参照。f64 アキュムレータの 1 パス蓄積・`max_dims`
  の同値タイ決定性を軸をまたいで維持するため）。新規カーネルは追加せず
  `BackendOps` も非拡張——既存 `sum`／`max`（CPU／CUDA。Metal は
  TASK-1.9c スコープ外の `Unsupported` を継承）カーネルをそのまま
  再利用する分解方式（`Var::einsum`〈#1620〉と同じ設計方針）。
- `Var::mean`（単一軸／全軸）は新 `Op::Mean`（`tape::Op`）として追加した。
  `BackendOps` に対応メソッドを持たず、forward は `self.tape.ops().sum`
  の結果をホスト側で縮約対象要素数 `n` により**1 回だけ除算**する合成
  （CPU バックエンド参照実装の「sum の後に 1 回だけ除算する」丸め規律
  と同じ）。`n == 0` は `AutodiffError::InvalidArgument` で fail-closed
  に拒否する（PyTorch は `NaN` を返すが安全側を採用）。VJP（`grad.rs::
  mean_vjp`）は `Sum` の VJP（複製）を `1/n` でスケールしたものに帰着
  する。checkpoint（`Op::is_checkpoint_eligible`）にも対応し、
  `recompute_value` が forward と同一の `ops.sum` → 同一除算で再導出
  するため checkpoint 有無で backward の値が bit 同一であることを
  統合テストで確認済み。
- `max_dims` の同値タイは `grad.rs::max_vjp` の既存「先勝ち決定的」
  規約（軸をまたいだ場合も「kept 軸〈元の順序〉→ reduced 軸〈昇順〉」
  の併合順で最初に現れる要素）をそのまま維持する。`amax`／`max` の
  勾配分配方式（先勝ち決定的 対 均等分配）を確定する #1718 は本
  イシュー時点（2026-09-14）で未解決の OPEN のまま・決定 doc も
  `docs/` に存在しないため、安全側として `max_vjp` は無変更（先勝ち
  維持）とした。#1718 が均等分配へ確定した場合は `max_dims` の期待値
  も追従して更新する必要がある。
- facade 新規公開面なし（既存 `Var` 再エクスポート経由のみ）。CUDA
  （DGX Spark GB10）実機での facade parity テスト（`crates/facade/tests/
  reduce_backend_parity.rs` の `#[ignore]` テスト）は本実装エージェント
  の実行環境に実機への到達手段がないため未実測のまま GB10 セッション
  へ申し送る。Metal は `sum`／`max` 自体が `Unsupported`（TASK-1.9c
  スコープ外）のため合成実装もその挙動を継承し対象外のまま。
- `min`／`argmax`／`argmin`（#1720）・`var`／`std`／`norm`（#1723）は
  本 issue の対象外のまま。

## 追補（イシュー #1720）

§2.5「Var 演算」の `min`／`argmax`／`argmin` 欠落行はスナップショット
不変のまま、`Var::min`／`argmax`／`argmin` を実装済み化したことを
ここに追補する。`min` は `dim: Option<usize>` を取る `torch.min(dim)`
相当（`BackendOps::min`。`sum`／`max` と異なりデフォルトメソッド。
既定 `Unsupported`）、`argmax`／`argmin` は `torch.argmax`／`argmin`
相当で非微分演算（`Tensor<i32>` を返しテープにノードを追加しない。
`Var::argsort` と同じ扱い）。CPU（`backend-cpu::reduction::min`／
`argmax`／`argmin`）はネイティブ実装済み。CUDA は `min`（`reduce::
CudaReduce::run_min_all_f32`／`run_min_axis_f32`。`fminf`・単位元
`+INFINITY`）を実装済みだが `argmax`／`argmin` は GPU カーネル未実装
（明示 `Unsupported`。`CudaBackendOps::argmax`／`argmin` が driver 非
接触で即座に返す）。Metal は `min`／`argmax`／`argmin` の 3 演算とも
明示 `Unsupported`。いずれも `Unsupported` の場合は `Var::min`／
`argmax`／`argmin` がホスト参照実装（`fandhe_ai_autodiff::eval::min`／
`argmax`／`argmin`）へフォールバックするため forward は全バックエンド
で動作する。`min` の VJP（`Op::Min`）は `Op::Max` と共有するヘルパー
`grad::extremum_first_match_vjp`（forward 記録値と `==` 一致する最初の
位置へ勾配を伝播する先勝ち決定的方式。旧 `max_vjp` を改称し `max_vjp`
自体は既存呼び出し元を壊さない薄いラッパーとして維持）を使う——#1718
（amax／amin 勾配分配方式の確定）が均等分配へ変更する場合はこのヘル
パー 1 箇所の差し替えで `Max`／`Min` 両方へ反映される。facade 新規
公開面なし（既存の `Var` 再エクスポート経由）。CUDA／Metal 実機での
facade parity テストは未実測のまま Mac／GB10 セッションへ申し送る。

## 追補（イシュー #1731）

`Var::cumsum`／`cumprod`（`torch.cumsum`／`torch.cumprod` 相当）を実装済み化した。スナップショット本体（対象 HEAD `097bff19`）は不変のまま、以下を追記する。

- `tensor-core::BackendOps::cumsum`／`cumprod`（既定 `Unsupported`）・`autodiff::Op::Cumsum`／`Op::Cumprod`・CPU 参照実装（`backend-cpu::scan`）・ホストフォールバック（`eval::cumsum_along`／`cumprod_along`）まで実装済み。
- forward は lane（縮約軸以外の全軸の組）ごとに `f64` アキュムレータを保持する逐次スキャン契約（`.claude/rules/coding-rust.md` の f64 アキュムレータ方針を forward の scan へ拡張）。VJP はホスト側のみ（`grad.rs::cumsum_vjp_along`／`cumprod_vjp_along`）で、`cumprod` は除算を用いない厳密形（排他的 prefix 積 `L` と後ろ向き Horner 型再帰 `S` の積）のため零要素を含む入力でも成り立つ。
- CUDA／Metal 専用カーネルは本イシューのスコープ外（既定 `Unsupported` フォールバックのまま）で後続イシューへ引き継ぐ。
- facade 新規公開面なし（既存 `Var` 再エクスポート経由でそのまま到達可能。`docs/compat-api-scope.md` §1.3）。

## 追補（イシュー #1718）

`amax`／`amin` 勾配分配方式（先勝ち決定的 対 均等分配）の設計判断を確定した。上記 #1719／#1720 追補が「#1718 が均等分配へ確定した場合はヘルパー 1 箇所の差し替えで反映される」と記していた前提は**採用しない**ことが確定した——`Var::max`／`min`／`max_dims`（`max(dim)`／`min(dim)` 族の意味論）は crates.io 公開全版で出荷済みの先勝ち決定的挙動を**維持**し、`grad::extremum_first_match_vjp` は差し替えない。PyTorch `torch.amax`／`amin` 相当の均等分配は、実装する場合は別 `Op`（`Op::Amax`／`Op::Amin`）・別 VJP ヘルパーとして独立に追加する方針とする（未実装・後続 issue 提案のまま。本イシューはコード変更を伴わない設計判断の確定のみ）。詳細・根拠は `docs/autodiff-amax-grad-distribution-decision.md` を参照。

## 追補（イシュー #1722）

AMP（自動混合精度。§2.12 の上記行「なし（`optim.rs` doc に「損失スケーリング（AMP）は現時点で未実装」と明記）」）を実装済み化した。スナップショット本体（対象 HEAD `097bff19`）は不変のまま、以下を追記する。

- `fandhe_ai_autodiff::nn::optim::amp`（#1721 で実装済み。`GradScaler`／`GradScalerConfig`／`UnscaleResult`／`scale_loss`／`scale_grads`／`unscale_grads`／`has_non_finite`）を `fandhe_ai::optim` から素の再エクスポート（案 A。`docs/facade-optimizer-promotion-decision.md` §4）で公開した。`crates/facade/src/optim.rs` の適用順序契約 doc を「AMP を使わない場合」（既存の `backward → clip → optimizer step`。無変更）と「AMP を使う場合」（`scale_loss → backward → unscale＋非有限検出 → 非有限なら clip・optimizer step を両方スキップ → clip → optimizer step → 必ず `GradScaler::update`）の 2 節へ更新した。
- 新規 `Op`／`BackendOps` メソッド／VJP は追加していない（`scale_loss` は既存 `Var::mul` の合成のみ）。
- 対象外事項の明記: (a) 真の混合精度（f16 forward・f32 master weight）は `docs/backend-dtype-dispatch-design.md` §8 のとおり対象外。(b) デバイス常駐更新経路（`DeviceParamStore`／`Tape::step_device_param_store`）には unscale／非有限検出が結線されておらず、AMP はホスト `Tensor<f32>` 勾配（`Gradients::get`／`SequentialVars::trainable_grads`／`Tape::param_grads_to_host` 経由）にのみ適用できる。
- facade のみ import する統合テスト（`crates/facade/tests/optim_amp_train_loop.rs`）で、収束・1 step の勾配 bit 完全一致（scale_loss→backward→unscale と非スケール backward の勾配が `f32::to_bits()` で一致すること。CPU バックエンド）・非有限勾配時の skip／backoff を固定した。

## #1627 の追補（int8 量子化の段階 0 設計判断）

スナップショット本体（対象 HEAD `097bff19`）の 354 行目「量子化（int8 等）」行（`なし・量子化 dtype・演算対応・難度 XL`）は不変のまま、以下を追記する。

- 正本 spec の除外事項「分散学習・量子化の網羅対応」（Won't・条件付き〈量子化 GEMM〉）は「spec 側で REQ として承認されるまで実装リポは量子化カーネルを起票・実装しない」と定めており、この判断は変わっていない（`docs/spec/04-requirements.md:356-364`）。
- 本イシューでは `Op`／`BackendOps`／`Var`／facade のコード実装を行わず、格上げ条件（a〜e）の充足状況の棚卸しと再開条件を `docs/backend-int8-quantization-decision.md` として記録した（#1628・#1652・#1775 と同型の段階 0）。格上げ条件のうち (a)（REQ-2 複合判定の改定）は実質充足と読めるが、(b)〜(e)（実機 MMA プローブ・Transformer 複合 WL ベースライン・量子化専用許容基準・依存追加なし設計の実装確認）は未達のまま（同 doc §2.1）。
- issue 上の承認コメント（`unsafe asm!`〈SME〉・`BackendOps` trait 拡張・facade 公開面拡張の技術的許可）は実装着手前の技術的許可事項に限られ、spec 側の除外事項ゲート自体を解除する文言ではないと整理した（同 doc §0.1）。
- facade 新規公開面なし（コード変更を伴わないため）。実装着手は本追補のスコープ外のまま引き続き #1627 として open・blocked で追跡する。

## #1741 追補

- `Var::sort`／`argsort`／`topk`（#1733 実装済み）の CUDA／Metal 専用カーネルを実装した（前段落「CUDA／Metal 専用カーネルは既定 `Unsupported`（ホストフォールバックで機能する）のまま #1741 へ引き継ぐ」の残対象を解消）。
- 両バックエンドとも 64bit 合成キー（`hi`：値を「NaN は最大・NaN 同士は同値・±0 は同値」へ正規化した `u32` totalOrder 風キー・`descending` のときのみ反転／`lo`：ライン内の元添字。`lo` を反転しないことで非安定ソートでも `key` 自体がライン内で一意になり、`docs/spec` の順序契約（同値タイブレークは `descending` に関わらず元添字昇順）が構造的に成立する）によるビットニックソート方式で実装した（CUDA: `crates/backend-cuda/src/{sort_model.rs,kernels_sort.rs,sort.rs}`。Metal: `crates/backend-metal/src/{sort_model.rs,shaders/sort.metal,sort.rs}`）。
- CPU 参照実装（`backend-cpu::sort_topk`）との bit 完全一致（`values`・`index` とも）を、実機なしで Linux 上検証できるホストモデル（`sort_model.rs::sort_lines_host_model` が実カーネルと同一アルゴリズムを意図的に複製）で網羅的に確認済み（12 形状 × `descending` × 複数 `out_len`）。
- `facade/tests/sort_topk_backend_parity.rs`（CPU vs NaiveOps の forward・backward parity。scatter ベース VJP を含め bit 完全一致確認済み）・`backend-cuda/tests/sort_topk_parity.rs`・`backend-metal/tests/{sort_topk_parity.rs,sort_topk_source_evidence.rs}` を追加した。CUDA・Metal 実機での `#[ignore]` テスト実行（形状網羅・非 contiguous 入力・`k` 網羅）は本エージェント実行環境に両実機への到達手段がないため未実施のまま GB10／Mac セッションへ申し送り。
- facade 新規公開面なし（既存の `Var` 再エクスポート経由のまま）。`sorted=False` の topk・負 `dim`・`k` の `Var` 化は引き続き対象外。
## #1753 の追補（親 #1631）

`clip_grad_value`（PyTorch `torch.nn.utils.clip_grad_value_` 相当。各
勾配要素を独立に `[-clip_value, clip_value]` へクランプする value 方式
gradient clipping）を実装済み化した。`fandhe_ai_autodiff::nn::optim::
clip::clip_grad_value`（既存 `clip_grad_norm` と同型の純関数。
`Gradients`／`Var` に非依存）・facade 到達経路は `crates/facade/src/
optim.rs` への `pub use` 1 行追加（`fandhe_ai::optim::clip_grad_value`）
のみで、新規 `Op`／`BackendOps`／VJP は追加していない（勾配マテリアラ
イズ後のホスト側後処理のため）。

`clip_grad_norm`（global L2 norm 方式）と異なりテンソル間の相関を見ず
各要素を独立にクランプするためスケーリングを伴わず、範囲内の要素は
bit 同一のまま返る。非有限（NaN／±Inf）の `clip_value` および勾配要素
はいずれも `AutodiffError::InvalidArgument` で拒否する fail-closed 契約
（`f32::clamp` の NaN 境界 panic を避けるため `max`/`min` 合成で実装し、
クランプ前に全要素の有限性を検査して NaN/Inf の静かな正規化による
隠蔽を防ぐ）。3 バックエンド専用カーネルは対象外（ホスト
`Tensor<f32>` のみを操作するため）。

## 追補（イシュー #1756）

`Var::pad`（`torch.nn.functional.pad(mode='constant')` 相当。定数
パディング）を実装済み化。`pads: &[(usize, usize)]` は先頭次元から
順に対応する（PyTorch `F.pad` の「末尾次元から逆順の平坦リスト」とは
意図的に異なる設計）。負パディング（クロップ）は非対応（`Var::narrow`
を使う）・`reflect`／`replicate` モードは対象外（本イシューのスコープ
外）。

- `BackendOps::pad`（既定 `Unsupported`。`Var::pad` は `Unsupported`
  のときのみホスト参照実装 `eval::pad` へフォールバック）・`Op::Pad`・
  CPU（`backend-cpu::constant_pad`）・CUDA（`backend-cuda::
  constant_pad`。NVRTC カーネル）・Metal（`backend-metal::
  constant_pad`。実行時コンパイルカーネル）の 3 バックエンド専用実装を
  本 issue で新規実装済み。
- VJP は新規カーネルを作らず、既存の `Tensor::narrow`（#1598 で確立した
  zero-copy view 基盤）を各軸へ連鎖適用してパディング領域を落とす
  （pad の forward ⟷ narrow の VJP・pad の VJP ⟷ narrow の forward
  という `Op::Concat`⟷`Op::Narrow` の双対性と同型の設計）。
- 出力は「入力要素のコピー」と「定数」のみで算術を含まないため、3
  バックエンド間は REQ-2 複合判定ではなく **bit 完全一致**（`value` が
  NaN の場合のみクラス一致）。
- facade 新規公開面なし（既存 `Var` 再エクスポート経由。`compat-api-
  scope.md` §5 の範囲拡張手続きは #1598／#1637／#1733 と同じ理由で
  再適用不要と判断）。
- CUDA／Metal 実機（GB10／M4 Max）での parity テストは、本実装
  エージェントの実行環境に実機への到達手段がないため未実測のまま
  Mac／GB10 セッションへ申し送る（`crates/backend-cuda/tests/
  constant_pad_parity.rs`・`crates/backend-metal/tests/
  constant_pad_parity.rs` の `#[ignore]` テストを参照）。

## #1633 の追補（sparse／complex テンソルの非対応を明文化）

§2.1「テンソル生成」・§2.12「dtype」に上記 2 行（sparse・complex）を追加した。決定記録は `docs/tensor-core-sparse-complex-decision.md`（#1633）。

- 段階 0（現時点非対応の明文化）。`Op`／`BackendOps`／`Var`／facade のコード実装・依存追加は行わない。
- spec REQ-9 の「引き続き対象外」判断（`docs/spec/04-requirements.md:233,432`）と整合しており、spec への新規提案は不要（同 doc §3）。
- ONNX complex dtype（`COMPLEX64`/`COMPLEX128`）は `GraphError::UnknownDataType` で fail-closed 拒否される一方、`GraphProto.sparse_initializer`（未宣言フィールド）は prost の仕様どおり無言でスキップされる非対称な挙動を事実として記録した（同 doc §2・§9(a)。是正は本イシューのスコープ外で、ユーザー承認を得たうえで別イシューへ引き継ぐ）。
- facade 新規公開面なし。
