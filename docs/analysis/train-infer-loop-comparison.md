# 学習／推論ループの横並び解析（candle・burn・PyTorch の 1 step 構造と fandhe phase 対応）

イシュー #2097。親 #2089（Phase 2「他ライブラリのコード取得・詳細解析」）・
祖父 #2088 系列の並列兄弟は #2090〜#2098（#2098 が集約）。#2091〜#2093 の後続。

## 1. 位置づけ・目的

framework-compare の学習（MLP 784→256→10・ReLU・batch 64・MSE・SGD lr=0.01・
100 step 中 20 warmup）と推論（同 MLP の forward + ホスト実体化）で、fandhe-ai は
candle／burn／PyTorch に対し負けセルを持つ。fandhe-ai 側は phase 分解
（`docs/perf/train-step-phase-breakdown.md` §17・`docs/perf/infer-reuse-phase-breakdown.md`
§10。区間定義の正は `scripts/bench/framework-compare/README.md`「`train --phases`」
「`infer --mode reuse` / `infer --phases`」節）が済んでいるが、**比較対象側の
1 step が「何を」「何回」行っているか（演算・alloc・同期）の構造化記録がない**。
本 doc はそれを読み取り解析で作り、fandhe の phase と 1 対 N 対応付けし、ループ構造の
差異を抽出して #2098 と Phase 3 の入力にする。

**本 doc は読み取り解析のみである。性能実測・最適化提案・数値契約の変更は行わない
（スコープ外。§10）。数値（時間）は転記しない — 構造（何が・何回）のみを扱う。**

## 2. 解析対象と版

| 対象 | 版 | 取得元・確認方法 | ライセンス |
|------|----|--------------------|-----------|
| `candle-core` | 0.11.0 | `curl -fsSL https://static.crates.io/crates/candle-core/candle-core-0.11.0.crate` を scratchpad へ取得し、`scripts/bench/framework-compare/Cargo.lock` の checksum `5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706` と一致を確認してから展開（ビルド・実行なし）。解析後に削除済み | MIT OR Apache-2.0（`Cargo.toml` 実測） |
| `burn-tensor` | 0.21.0 | ローカル cargo registry cache（`~/.cargo/registry/cache/…/burn-tensor-0.21.0.crate`）の sha256 `223ed3804bb9436e401fc57b87e44c86267070b870813520ed9f706c80c81442` が `Cargo.lock` の checksum と一致することを確認してから `~/.cargo/registry/src/…/burn-tensor-0.21.0/` を読解（ビルド・実行なし） | MIT OR Apache-2.0 |
| `burn-autodiff` | 0.21.0 | 同上（sha256 `3b93a80e43bfab909399444b29ce3894ff51f7e879720bfc75b6007ae6a01c3d` 一致確認） | MIT OR Apache-2.0 |
| `burn-ndarray` | 0.21.0 | 同上（sha256 `dceb9692e292782bab7ad0259e8e8f5ee80ba2b0329a78f0ba589a26a9633dd6` 一致確認） | MIT OR Apache-2.0 |
| `burn-wgpu` | 0.21.0 | 同上（sha256 `6eb009254af15922cb37814f3b3b61043046193ba2fde97c3879a25fbd51a92e` 一致確認） | MIT OR Apache-2.0 |
| `burn-cuda` | 0.21.0 | 同上（sha256 `a6c66b49136fc11f59773054358f401fbec7fa0d69615f55ef22c29cf241c9b7` 一致確認） | MIT OR Apache-2.0 |
| `cubecl-runtime` | 0.10.0 | 同上（sha256 `b68491bf5b3e997ae36bdc4e63b4ccd6d2f0e86b3b596a5d7a48d2b9e92622a0` 一致確認）。`burn-cubecl-cuda-matmul.md` が読解済みの CUDA dispatch 層は再解析せず引用のみ | MIT OR Apache-2.0 |
| PyTorch | 2.14.0+cu130（DGX Spark GB10 実測 JSONL の `version` フィールド。`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/rayon-sweep-pinned.jsonl` 等）。mac 側の同ディレクトリ JSONL には torch 行が含まれておらず版が未確認のため、以後は DGX 実測版（タグ `v2.14.0`）を基準に GitHub 上のソースを参照する | `raw.githubusercontent.com/pytorch/pytorch/v2.14.0/{derivatives.yaml,gen_autograd_functions.py,LICENSE}` を個別ファイル取得（リポジトリ clone はしていない。取得物は scratchpad で削除済み） | BSD-3-Clause（`LICENSE` 実測。"Redistribution and use in source and binary forms..." 節を確認） |

ハーネス側コード（`scripts/bench/framework-compare/bench-candle`・`bench-burn`・
`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`）は本リポジトリの
in-repo コードのため逐語引用可能で、取得・削除の対象外。

## 3. ハーネスのループ逐語リスト

計測窓は `Instant::now()`（Rust）／`time.perf_counter()`（Python）の対で、
記載箇所は窓の内側にある呼び出しのみ。「窓外」と明記したものはモデル初期化・
デバイス構築・入力アップロードで、いずれも計測開始前に 1 回だけ行われる。

### 3.1 candle（`bench-candle/src/main.rs`）

**窓外**: `make_device` → `Mlp::new`（`w1`／`w2` を `Xorshift64Star` で初期化し
`Var::from_tensor` へ包む。`b1`／`b2` はゼロ初期化）→ `mlp_inputs`（`x`／`y` を
`Tensor::from_vec` でアップロード。`Var` ではない）。

**train（`run_train`。L473-527）**: 100 step ループ、各 step の窓内:

1. `model.forward(&x)`（`Mlp::forward`。L454-461）:
   `x.matmul(w1) → broadcast_add(b1) → relu() → matmul(w2) → broadcast_add(b2)`
   （matmul 2 回・broadcast_add 2 回・relu 1 回。いずれもカーネル即時発行）
2. `(pred - &y)?.sqr()?.mean_all()?`（sub 1 回・sqr〈elementwise 二乗〉1 回・
   mean_all〈全要素縮約〉1 回）
3. `loss.backward()?` → `GradStore`（§4.1 で内訳を確認）
4. 4 パラメータ（`w1, b1, w2, b2`）それぞれに対し:
   `grads.get(var)` → `var.as_tensor() - (grad * LR)?` → `Var::set(&updated)`
   （`mul` 1 回・`sub` 1 回・`set` 1 回 ×4 パラメータ）
5. `loss.to_scalar::<f32>()?`（ホスト実体化 = 同期点。step を締める）

**infer（`run_infer`。L529-576）**: warmup 20 + 計測 20 回、各反復の窓内:
`model.forward(&x)` → `checksum2(&out)`（`to_vec2::<f32>()` でホスト実体化・
同期し、`flat_map` で f64 総和）。

### 3.2 burn（`bench-burn/src/main.rs`）

**窓外**: `Mlp::new`（`tensor2::<B>` で `w1/b1/w2/b2` をアップロード。`b1`／`b2`
は `[1, D]` 形。train のみさらに `require_grad()` で 4 パラメータを包む）→
`mlp_inputs`（`x`／`y` をアップロード。`require_grad()` なし）。

**train（`run_train`。L221-300）**: 100 step ループ、各 step の窓内:

1. `model.forward(x.clone())`（`Mlp::forward`。L200-203）:
   `relu(x.matmul(w1) + b1)` → `.matmul(w2) + b2`
   （`clone()` 5 回〈呼び出し側の `x.clone()` 1 回 + `Mlp::forward` 内の
   w1/b1/w2/b2 の参照カウント複製 4 回。§4.2〉・matmul 2 回・
   add 2 回・relu 1 回）
2. `let diff = pred - y.clone(); let loss = (diff.clone() * diff).mean();`
   （sub 1 回・`clone` 2 回〈`y.clone()`・`diff.clone()`〉・mul 1 回・
   mean 1 回）
3. `loss.backward()`（`Gradients` を返す。§4.2）
4. 4 パラメータそれぞれに `model.<p>.grad(&grads)` で勾配取得 → クロージャ
   `step`: `p.inner() - g.mul_scalar(LR)` → `Tensor::from_inner(..).require_grad()`
   （`mul_scalar` 1 回・`sub` 1 回・`from_inner`＋`require_grad` 1 回 ×4）
5. `loss.into_scalar().elem::<f32>()`（ホスト実体化 = 同期点）

**infer（`run_infer`。L302-355 付近）**: warmup + 計測、各反復の窓内:
`model.forward(x.clone())` → `checksum(out)`（`into_data().to_vec()` でホスト
実体化・同期し f64 総和）。

### 3.3 PyTorch（`bench_py.py` の `Torch` クラス）

**窓外**: `train_setup`/`run_infer` の `mlp_init()`（NumPy で `w1/b1/w2/b2/x/y`
生成）→ `self.p = [upload(v).requires_grad_(True) for v in (w1,b1,w2,b2)]`
（`x`／`y` は `requires_grad_` なしでアップロード）。

**train（`Torch.train_step`。L105-115）**: 100 step ループ、各 step の窓内:

1. `h = relu(x @ w1 + b1); pred = h @ w2 + b2`
   （matmul 2 回・add〈`+`〉2 回・relu 1 回）
2. `loss = ((pred - y) ** 2).mean()`（sub 1 回・`**2`〈pow〉1 回・mean 1 回）
3. `gs = torch.autograd.grad(loss, self.p)`（`self.p` の 4 パラメータのみを
   対象にした部分 backward。§4.3）
4. `with no_grad(): for p, g in zip(self.p, gs): p.sub_(g * LR)`
   （`mul` 1 回・`sub_`〈in-place〉1 回 ×4）
5. `float(loss.item())`（ホスト実体化 = 同期点）

**infer（`Torch.forward_to_host`。L116-120）**: warmup + 計測、各反復の窓内:
`with no_grad(): (relu(x @ w1 + b1) @ w2 + b2).cpu().numpy()`（`.cpu()` が
同期点。`no_grad` のためグラフを構築しない）。

## 4. 上流ソース確認結果

### 4.1 candle — backward の matmul・elementwise 数（**dX が無条件計算される**）

`candle-core-0.11.0/src/backprop.rs::sorted_nodes`（`walk` 内部関数）は、
ノードが `is_variable()`（= `Var`）であるか、祖先のいずれかが `track_grad` の
とき `track_grad = true` として `nodes` へ push する（L46-158）。この判定は
**出力ノード単位**であり、個々の演算の「どちらの入力が微分対象か」は見ない。

`Tensor::backward`（L165 以降）は `sorted_nodes()` が返したノードのみを逆順に
処理する。`Op::Matmul(lhs, rhs)` の backward 実装（L457-468）は、`lhs` 側の
勾配（`grad.matmul(&rhs.t())` 相当。dX 相当）と `rhs` 側の勾配
（`lhs.t().matmul(&grad)` 相当。dW 相当）の**両方を条件分岐なしに常時計算**
する（`Requirement`／`Option` による親の要否判定を経由しない）。

本ハーネスの層 1（`h1 = x.matmul(w1)`）は `x`（非 `Var`）と `w1`（`Var`）の
matmul で、出力 `h1` は `w1` に依存するため `track_grad = true` となり
`sorted_nodes` に含まれる。よってこの matmul の backward が実行され、
`lhs_grad`（= dX。`x` は葉ノードで以後どこにも伝播せず**使われない**）と
`rhs_grad`（= dW1）が**両方**計算される。層 2（`out = h.matmul(w2)`）は
`lhs_grad`（dH。層 1 へ伝播）・`rhs_grad`（dW2）とも必要。

**結果**: backward の matmul 回数は **4 回**（dX〈無駄〉・dW1・dH・dW2）。

その他の backward 実装（L440-720 に列挙）: `Op::Broadcast`（bias の
`broadcast_add` 逆伝播。L479-501）は `sum_keepdim` + `squeeze` ×`left_dims`
回（本ハーネスは `[BATCH,N]→[N]` で `left_dims=1` なので `squeeze` 1 回）。
`Op::Unary(_, Relu)`（L634-637）は `ge`〈比較〉→`to_dtype`〈cast〉→`mul`→`add`
の 4 演算。`Op::Unary(_, Sqr)`（L695-698）は `mul`→`affine`→`add` の 3 演算。

### 4.2 burn-autodiff — backward の matmul 数（**未追跡入力の勾配計算をスキップ**）

`burn-autodiff-0.21.0/src/ops/tensor.rs::float_matmul`（L565-621）は
`Matmul::backward` 内で `ops::backward::binary`（`backward.rs` L50-73）を呼ぶ。
`binary` は `parents: [Option<NodeRef>; 2]` を見て **`Some` の親だけ**
`func_lhs`/`func_rhs`（= matmul によるその親の勾配計算）を実行する（L64-72）。
`parents` は `Requirement::from_nodes` によって構築され、その要素はノードが
「追跡対象（= `require_grad()` 済みの祖先を持つ）」かどうかで `Some`/`None` が
決まる（`prepare::<C>` L34-46）。

本ハーネスの `x`（`mlp_inputs` で `require_grad()` を呼んでいない）は非追跡
のため、層 1（`h = x.matmul(w1)`）の `lhs`（= `x`）は `parents[0] = None` と
なり、`func_lhs`（dX 相当の matmul）は**呼ばれない**。`rhs`（= `w1`。
`require_grad()` 済み）は `parents[1] = Some` のため `func_rhs`（dW1）は実行
される。層 2（`out = h.matmul(w2)`）は `h`（`w1`/`b1` 経由で追跡対象）・
`w2`（追跡対象）とも `Some` のため両方実行。

**結果**: backward の matmul 回数は **3 回**（dW1・dH・dW2。dX 相当は未計算）。

bias（`Op::Add`。`tensor.rs` L140-179）の backward も同じ `binary` 経由で、
`broadcast_shape::<B>(grad, &shape_lhs/&shape_rhs)`（`L161-162`）が形状縮約を
行う（本ハーネスの bias は `[1,N]` 形のため縮約先は `[1,N]`。candle の `[N]`
形とは異なる縮約先形状——§6 参照）。

### 4.3 PyTorch — `torch.autograd.grad` の入力限定と `needs_input_grad`

`bench_py.py` の train は `torch.autograd.grad(loss, self.p)` を呼ぶ
（`self.p` = `[w1,b1,w2,b2]`。`x`／`y` を含まない）。`torch.autograd.grad`
は全パラメータ backward ではなく**指定した `inputs` に到達する経路のみ**を
実行するグラフ剪定を行う（`only_inputs` セマンティクス）。加えて `x` は
`requires_grad_(True)` を呼んでいないため、そもそも `mm(x, w1)` の grad_fn
（`MmBackward0`）に対する「`self`（= `x`）の勾配」は `needs_input_grad[0]`
が偽になる。

生成コード側の裏付け: PyTorch v2.14.0 の `tools/autograd/derivatives.yaml`
（L1235-1238）は `mm` の微分式を
`self: mm_mat1_backward(...)` / `mat2: mm_mat2_backward(...)` と定義し、
コード生成側 `tools/autograd/gen_autograd_functions.py`（L1009 付近）が
各出力ごとに `if (task_should_compute_output({ix}))` のガードを生成する
（`task_should_compute_output` は `needs_input_grad`／`inputs` 指定から導出
される実行要否フラグ）。`x` に対する `self` 側微分（dX 相当）はこのガードで
スキップされ、実行されない。

**結果**: backward の matmul 回数は **3 回**（dW1・dH・dW2。dX 相当は未計算。
burn と同型）。

**4.1〜4.3 のまとめ（headline diff）**: 同じ「非 `Var`／非 `requires_grad` の
入力 `x`」に対し、**candle は dX を無条件に計算して捨てる（4 matmul）**のに
対し、**burn・PyTorch はグラフ構築時点で dX を計算しない（3 matmul）**。
これは framework-compare の train 計測窓に候補として持ち込まれる構造差であり、
時間影響の実測は Phase 3 へ引き継ぐ（§9）。

### 4.4 融合・遅延（burn の fusion・autotune 無効の確認）

`bench-burn/Cargo.toml`（§2 参照済みの L14-18）は
`burn = { version = "=0.21.0", default-features = false, features = ["std", "ndarray", "autodiff"] }`
で `wgpu`/`cuda` feature は `[features]` テーブルで個別に足す構成であり、
`fusion` feature は列挙されていない。これは `burn-cubecl-cuda-matmul.md`
（イシュー #2093）が CUDA 経路について確認済みの「autotune・fusion 無効」と
同じ結論を wgpu（Metal ホスト側）経路にも及ぼす——`default-features = false`
で `fusion` を明示的に足していない以上、CUDA・wgpu の両経路で fusion は
有効化されない（`burn-cubecl-fusion` crate 自体がリンクされない）。

candle は演算ごとに即時カーネル発行（eager）で、演算融合機構を持たない
（`Tensor::matmul`/`broadcast_add`/`relu` 等はそれぞれ独立の
`Storage::…` 呼び出しに帰着する。§4.1 の `sorted_nodes` が「1 演算 = 1
グラフノード」であることからも読み取れる）。PyTorch はハーネスが
`torch.compile` を使っていない（`bench_py.py` に compile 呼び出しなし）ため
eager 実行。

### 4.5 同期点

- **candle CPU**: 演算はすべて同期実行（ホスト命令の逐次実行）。`to_scalar`/
  `to_vec2` は追加の待機を要しない（既に完了済みの計算結果を読むだけ）。
  Metal／CUDA の非同期発行・`to_scalar`/`to_vec2` での commit・wait は
  `candle-0.11.0-cuda-path.md` §6（CUDA）を引用し本 doc では再解析しない
  （Metal 経路はハーネスの `bench-candle` に `metal` feature 分岐があるのみで
  本 doc の対象外の実機解析に委ねる。未確定）。
- **burn ndarray（CPU）**: 同期実行。**burn wgpu／cuda**: `cubecl` の非同期
  キュー発行を介する（`burn-cubecl-cuda-matmul.md` が CUDA について確認済み。
  `into_scalar`/`into_data` が完了待ちの同期点になる）。wgpu 経路の同期実装
  詳細（`wgpu::Queue::submit` のポーリング方式等）は本 doc では未確認
  （未確定・理由: 時間予算内で `burn-wgpu`/`cubecl-wgpu` のポーリング実装
  まで読み切れなかった。Phase 3 or #2098 へ申し送り）。
- **PyTorch CPU**: 同期実行。`.item()`/`.cpu()` は追加待機なし。CUDA／MPS の
  ストリーム同期実装は本 doc の解析契約（ディスパッチ層まで）の範囲外で
  未確認（閉源部〈cuBLAS／MPS〉はディスパッチ層までという契約どおり、
  ディスパッチ層そのものの読解も本 doc では実施していない。未確定）。

### 4.6 メモリ確保方式

- candle・burn（cubecl）・PyTorch はいずれも独自のデバイスメモリアロケータ
  ／プールを持つ（PyTorch: caching allocator。cubecl:
  `cubecl-runtime::memory_management`。candle: `Device` ごとの
  `MetalDevice`/`CudaDevice` 内部バッファキャッシュ）。3 者とも実装詳細
  （プールの断片化戦略・再利用条件）はディスパッチ層より深く、本 doc では
  「プールを持つこと」の存在確認に留め、内部実装は未確認（未確定。理由:
  §Overview の解析契約「閉源部はディスパッチ層まで」を OSS 実装にも準用し、
  時間予算内では存在確認相当の grep に留めた）。

## 5. 1 step の構造集計表

数値（時間）は含まない。演算種別ごとの**回数**の集計。

| フレームワーク | 層 | matmul | elementwise（sub/mul/relu 等） | reduction（mean 等） | パラメータ更新の elementwise | alloc 傾向 | sync 点 |
|---|---|---|---|---|---|---|---|
| candle | forward | 2 | 2〈broadcast_add〉+ 1〈relu〉= 3 | 0 | — | 各演算が新規 `Storage` を確保（in-place API 極小） | forward 内では発生せず |
| candle | loss | 0 | sub 1・sqr 1 | mean_all 1 | — | 同上 | — |
| candle | backward | **4**（dX 含む） | Broadcast 逆伝播 4（b1・b2 各 `sum_keepdim`+`squeeze` の 2 演算）・Relu 逆伝播相当 4・Sqr 逆伝播相当 3 | mean_all 逆伝播（未読解・未確定） | — | 各 VJP が新規 `Tensor` を確保し `grads.or_insert`/`add` で蓄積 | — |
| candle | 更新 | 0 | `mul`1・`sub`1・`Var::set`1（storage 書換か差替かは未確認§4 外）×4 | — | 4 パラメータ分 | — | — |
| candle | step 末尾 | — | — | — | — | — | `to_scalar`（1 回） |
| burn | forward | 2 | add 2・relu 1（+ `clone` 5：呼び出し側 `x.clone()` 1 + w1/b1/w2/b2 の参照複製 4） | 0 | — | `clone` は Arc 参照カウントで実コピーではない可能性が高いが未確認（未確定） | forward 内では発生せず |
| burn | loss | 0 | sub 1・mul 1（`diff*diff`）+ `clone` 2（`y.clone()`・`diff.clone()`） | mean 1 | — | — | — |
| burn | backward | **3**（dX 省略） | Add 逆伝播〈broadcast_shape〉2・Relu／Mul 逆伝播（未読解・未確定） | mean 逆伝播（未読解・未確定） | — | — | — |
| burn | 更新 | 0 | `mul_scalar`1・`sub`1・`from_inner`+`require_grad`1（グラフノード新規生成）×4 | — | 4 パラメータ分 | 新規グラフノード生成が毎 step 発生（`from_inner().require_grad()`） | — |
| burn | step 末尾 | — | — | — | — | — | `into_scalar`（1 回） |
| PyTorch | forward | 2 | add 2・relu 1 | 0 | — | eager、各演算が新規 Tensor を確保（未確認・一般的挙動からの記述） | forward 内では発生せず |
| PyTorch | loss | 0 | sub 1・pow 1 | mean 1 | — | — | — |
| PyTorch | backward | **3**（dX 省略。`autograd.grad(inputs=self.p)` によるグラフ剪定 + `needs_input_grad`） | AddBackward の broadcast 縮約 2・ReluBackward／PowBackward（未読解・未確定） | MeanBackward（未読解・未確定） | — | — | — |
| PyTorch | 更新 | 0 | `mul`1・`sub_`（in-place）1 ×4 | — | 4 パラメータ分（in-place のため新規 alloc は `mul` の一時 Tensor のみ） | — | — |
| PyTorch | step 末尾 | — | — | — | — | — | `.item()`（1 回） |

融合・lazy/eager: 3 者とも eager（§4.4）。fusion は 3 者とも無効
（candle は機構自体を持たない。burn は feature 未有効。PyTorch は
`torch.compile` 未使用）。

## 6. ループ構造差の対比表

| 観点 | candle | burn | PyTorch |
|------|--------|------|---------|
| backward の dX 計算（非 `Var`/非 `requires_grad` 入力向け） | **計算する（無駄）**（§4.1） | 計算しない（§4.2） | 計算しない（§4.3） |
| bias の形状 | `[N]`（`b1: Var::from_tensor(Tensor::zeros((D_HIDDEN,)))`） | `[1, N]`（`tensor2` で `[1, D_HIDDEN]`） | `[N]`（NumPy 1 次元のままアップロード） |
| パラメータ更新後の状態 | `Var::set` で既存 `Var` を書き換え（新規グラフノード不要） | `Tensor::from_inner(..).require_grad()` で**毎 step 新規のグラフノードを生成**（`model` 構造体を作り直す） | `sub_`（in-place）で既存 Tensor を書き換え |
| backward の起動範囲 | `loss.backward()`（グラフ全体を辿るが `sorted_nodes` は track_grad ノードのみ） | `loss.backward()`（グラフ全体だが `binary` が親の `Option` で剪定） | `torch.autograd.grad(loss, self.p)`（**呼び出し時点で対象 `inputs` を明示し剪定**） |
| fusion | 機構なし（eager 固定） | feature 未有効（§4.4） | `torch.compile` 未使用 |
| lazy/eager | eager | eager（cubecl 非同期キューはあるが融合ではない） | eager |
| 同期点の数（1 step あたり） | 1（`to_scalar`） | 1（`into_scalar`） | 1（`.item()`） |
| 同期点の位置 | step 末尾（更新後の loss 読み出し） | step 末尾（同左） | step 末尾（同左） |
| CPU 経路の実行方式 | 同期実行 | 同期実行（ndarray） | 同期実行 |
| GPU 経路の非同期発行 | 未確認（§4.5・candle-cuda-path.md 参照） | あり（cubecl キュー。§4.5） | 未確認（§4.5・ディスパッチ層まで） |

## 7. fandhe-ai phase との 1 対 N 対応

fandhe の phase 名は `scripts/bench/framework-compare/README.md`「`train
--phases`」「`infer --mode reuse` / `infer --phases`」節（実装は
`bench-fandhe/src/main.rs::measure_train_phases`/`measure_train_reuse_phases`/
`measure_infer_phases`）で定義される値をそのまま使う。

### 7.1 train fresh（`tape_build`・`leaf_register`・`forward`・`loss_readout`・
`backward`・`param_readout`・`host_sgd`・`apply_params`・`tape_drop`）

| fandhe phase | 対応する比較対象側の処理 | 分類 |
|---|---|---|
| `tape_build` | 対応なし（candle/burn/PyTorch に明示的なテープ作成 API はないが、forward 中にグラフを毎 step 構築する。candle は `Var` 自体がグラフ状態を持たず forward 呼び出しごとに演算グラフが暗黙に組まれる。burn は `require_grad()` 済みテンソルの参照からグラフが暗黙に組まれる。PyTorch も同様） | ハーネス・計測境界由来（fandhe の明示的 `Tape` API 設計に起因し、3 者に対応物がない） |
| `leaf_register` | 対応なし相当（3 者とも「テープへの登録」という別呼び出しはなく、forward 内の演算呼び出し自体が入力を暗黙にグラフへ組み込む） | ハーネス・計測境界由来 |
| `forward` | §3 の forward 手順（matmul×2・broadcast/add×2・relu×1。3 者共通の演算列） | ライブラリ固有寄り（演算列自体は 1 対 1 対応するが、fandhe は matmul 即時・elementwise 遅延という独自の実行契約を持つ） |
| `loss_readout` | 対応なし相当（3 者は loss を計算するがこの時点で明示的な「実体化」呼び出しはない——host readout は step 末尾の 1 回のみ） | ハーネス・計測境界由来（fandhe は遅延 elementwise を backward 前に強制実体化する設計。§8 の差分候補） |
| `backward` | §3 の `loss.backward()`/`torch.autograd.grad(...)` 呼び出し全体 | ライブラリ固有（backward の内部構造〈dX 有無・matmul 回数〉は §4.1〜4.3 の差分） |
| `param_readout` | 対応なし（3 者ともパラメータ・勾配はデバイス常駐のままホスト往復しない） | ハーネス・計測境界由来（fandhe fresh 固有のホスト往復設計） |
| `host_sgd` | §3 の更新式（`grad*LR` → `sub`）の計算自体は 3 者にも存在するが、3 者はデバイス上で計算しホストへは出さない | ライブラリ固有＋計測境界の混合（更新の**演算そのもの**は対応するが、**ホストで行うか否か**は fandhe fresh 固有） |
| `apply_params` | 対応なし（3 者はパラメータ更新をデバイス上で完結し「適用」という別呼び出しを要しない） | ハーネス・計測境界由来 |
| `tape_drop` | 対応なし（3 者はテープ／グラフを明示的に破棄する API 呼び出しを持たない。スコープを抜けると自然に解放される） | ハーネス・計測境界由来 |
| `step_total` | 3 者それぞれの 1 step 計測窓全体（§3） | 検算用（対応関係の対象外） |

### 7.2 train reuse（`tape_build`・`leaf_register`・`forward_resident`・
`loss_readout`・`backward`・`device_update`・`tape_drop`）

fresh との差分のみ記す（`tape_build`/`leaf_register`/`loss_readout`/
`tape_drop` は 7.1 と同じ分類）。

| fandhe phase | 対応する比較対象側の処理 | 分類 |
|---|---|---|
| `forward_resident` | §3 の forward（パラメータがデバイス常駐のまま演算する点は 3 者の通常経路と一致） | ライブラリ固有寄り（fresh の `forward` より 3 者の実態に近い——3 者はそもそも常にパラメータ常駐） |
| `backward` | 7.1 と同じ | ライブラリ固有 |
| `device_update` | §3 の更新式全体（3 者ともデバイス上でパラメータ更新を完結する点が fandhe reuse と一致するのはパラメータ更新の**演算自体**に限る。fandhe reuse の `device_update`〈`tape.step_device_param_store`〉自体は「grad H2D + デバイス上 SGD 発行」と定義され（`scripts/bench/framework-compare/README.md`「`train --phases`」節 `device_update` 行）、更新演算とは別に勾配の H2D 転送が残り得る区間である。残存量はバックエンドの resident 対応状況（weight／bias 各勾配ごとの slot 対応）に依存し、`docs/device-resident-update-design.md` の対応表（CPU: weight `Some`／bias `None`、CUDA〈#1559〉: weight `Some`／bias `None`、Metal〈#1555+#1566〉: weight `Some`／bias `Some`）が正——3 バックエンド（CPU／CUDA／Metal）いずれも weight 勾配は resident のため H2D 対象は bias 等の縮約勾配のみ（CPU／CUDA）または皆無（Metal）に絞られる。framework-compare の `README.md` 側の記述（「CUDA／Metal は未対応のため全パラメータぶん H2D する」）はこの対応表と食い違っており、本 doc の時間予算では pin 済み `fandhe-ai =0.9.0` 実ソース側の突合を CUDA（`fandhe-ai-backend-cuda-0.9.0/src/ops.rs::gemm_fp32_strict_into` がオーバーライド済みで weight resident と確認）のみに留めたため、Metal 側・README 側の食い違いの原因特定は Phase 3 へ引き継ぐ未確定事項とする。3 者はそもそもパラメータ・勾配ともデバイス常駐で更新完結するため、fandhe 側に残る H2D 相当区間自体を持たない）。 | ライブラリ固有寄り（reuse は fresh より 3 者のループ構造に近いが、勾配 H2D の残存量はバックエンド resident 対応状況依存。§8・§9 参照） |

**構造的非対称の要約**: fandhe **fresh** は `leaf_register`（入力の毎 step
登録）・`param_readout`/`apply_params`（パラメータのホスト往復）という 3 者に
存在しない区間を持つのに対し、fandhe **reuse**（`forward_resident`・
`device_update`）は 3 者の「パラメータ常駐・デバイス上更新」という既定の
ループ構造に近づく。ただし reuse の `device_update` は演算（デバイス上 SGD
発行）自体は 3 者と一致する一方、**勾配の H2D 転送が更新演算とは別に残り
得る**（バックエンドの weight／bias resident 対応状況に依存。上表・
`docs/device-resident-update-design.md` 参照）ため、「パラメータ常駐・
デバイス上更新」への一致は演算のみに限られ完全な同型ではない。**`loss_readout`
が backward の前に同期を強制する**
（README「同期待ちを独立区間にできない理由」節）点は fresh・reuse 共通の
fandhe 固有構造で、3 者は同期点が step 末尾の 1 か所（loss/出力の host
readout）のみという構造（§6）と対照的である。

### 7.3 infer（fresh cpu／fresh gpu／reuse の 3 区間集合）

| fandhe phase（区間集合） | 対応する比較対象側の処理 | 分類 |
|---|---|---|
| fresh cpu: `predict` | §3 の forward（candle/burn/PyTorch はいずれも CPU では forward 呼び出しがそのまま演算列） | ライブラリ固有寄り |
| fresh gpu: `leaf_register` | 対応なし（3 者は入力をハーネス側で 1 回アップロード済みで毎反復の「登録」呼び出しを持たない。§3 のとおり 3 者の入力アップロードは計測窓外） | ハーネス・計測境界由来（fandhe fresh gpu 固有の毎反復登録） |
| fresh gpu: `forward` | §3 の forward。fandhe 固有の「`Linear::bind` の重み clone + H2D を毎反復含む」（README 該当行）という点は 3 者の「パラメータ常駐」（§3 windows outside）と対照的 | ライブラリ固有（演算列は対応するが、fandhe fresh gpu のみ毎反復パラメータ H2D を含む） |
| fresh gpu: `to_tensor` | 対応なし相当（3 者は `forward` の出力をホストへ持ち出す際、`host_copy` 相当の 1 回の呼び出し〈candle `to_vec2`／burn `into_data().to_vec`／PyTorch `.cpu().numpy()`〉が実体化とコピーを同時に行うため、fandhe の「遅延 elementwise の実体化」だけを切り出す独立区間を 3 者は持たない。区間定義は `scripts/bench/framework-compare/README.md`「`infer --phases`」節 fresh gpu 行（`out.to_tensor()`）を正とする） | ハーネス・計測境界由来（fandhe 固有の遅延 elementwise グラフを `forward` 内に保持し、`host_copy` 直前に強制実体化する設計。§7.1 の `loss_readout` と同種のハーネス境界） |
| reuse: `predict_resident` | §3 の forward（3 者は常にこの「常駐」形に近い） | ライブラリ固有寄り |
| fresh/reuse 共通: `host_copy`／`checksum`／`iter_total` | §3 のホスト実体化＋checksum 計算（candle `to_vec2`+`flat_map`、burn `into_data().to_vec`、PyTorch `.cpu().numpy()`+`np.cumsum` ※`bench_py.py::checksum`） | ライブラリ固有（実装は異なるが役割は 1 対 1 対応） |

**構造的非対称の要約**: fandhe fresh（GPU）の `forward` は毎反復
`Linear::bind` の重み clone + H2D を含む（README 記載）のに対し、3 者は
すべて重みをデバイス常駐のまま保持し毎反復のアップロードを行わない
（§3「窓外」節）。fandhe の `Tensor<f32>` はホスト常駐で GPU 演算ごとに
H2D→カーネル→D2H という前提（README 記載）を持つ一方、比較対象 3 者は
デバイス常駐 eager（PyTorch／candle）／eager+非同期キュー（burn cubecl）で
動く。この非対称は `predict_resident`（reuse）でのみ解消される。

## 8. 差分候補と既存判定のタグ

読み取り解析から見えた候補を、既存の実測記録（ADOPT／REJECT／undetermined）
と突き合わせる。**照合先に既存記録があるものは引用のみとし、再実験は提案
しない**。

| 差分候補 | 関連する既存記録 | 状態 |
|---|---|---|
| fandhe fresh のパラメータホスト往復（`param_readout`/`apply_params`）と 3 者のデバイス常駐更新の差 | `docs/perf/train-resident-grad-device-update.md`（reuse 側の対応） | 既存 ADOPT 済み（reuse 経路として実装済み。§7.2） |
| `loss_readout` が backward 前に同期を強制する構造 | `docs/inference-chain-single-sync-design.md`・`docs/perf/metal-infer-chain-single-sync.md`・`docs/perf/infer-chain-single-sync-cuda-ab.md`（推論側の単一同期化の知見） | 推論側は対応済み。学習側（train の `loss_readout` 位置）への横展開は本 doc では確認できず、Phase 3 へ引き継ぐ候補として記録するに留める |
| GEMM epilogue 融合（bias/relu を matmul に融合） | `docs/perf/train-linear-epilogue-fusion.md`・`docs/fusion-graph-design.md`・`docs/kernel-fusion.md` | 既存記録あり（引用のみ。3 者は fusion 無効〈§4.4〉のため本 doc の比較対象範囲では非対称の主因にならない） |
| elementwise 融合（relu マスク・reduce のストライド） | `docs/perf/cpu-elementwise-fusion-effect.md`・`docs/perf/gpu-elementwise-fusion-b1.md`・`docs/perf/train-reuse-relu-mask-stride.md` | 既存記録あり（引用のみ） |
| backward の GEMM 配線（dW／dX の計算経路） | `docs/perf/train-backward-gemm-wiring.md` | 既存記録あり。**candle の dX 無駄計算（§4.1）はこの既存記録が扱う fandhe 側の配線とは別軸（他ライブラリの構造）のため新規候補として #2098 へ提示する** |
| infer fresh gpu の毎反復重み H2D | `docs/perf/cpu-infer-predict-profile.md`（CPU 側の関連知見） | GPU 側の毎反復 H2D 自体への直接対応記録は本 doc の調査範囲では見当たらず、未確定のまま Phase 3 へ引き継ぐ |
| CUDA 非同期発行・同期除去 | `docs/perf/cuda-async-sync-removal-framework-compare-ab.md`・`docs/perf/cuda-async-sync-removal-rtx3060.md` | 既存記録あり（引用のみ） |
| train step hook（比較対象に相当する更新ループの抽象化） | `docs/compat-train-step-hook-decision.md` | 既存記録あり（引用のみ。facade 設計判断） |

**新規候補（既存記録に見当たらず Phase 3・#2098 へ提示するもの）**:

1. candle の backward が dX を無条件計算する構造（§4.1・§8 表内既述）。
   fandhe 側にこれに相当する「無駄な入力勾配計算」の有無は本 doc の範囲外
   （fandhe 側の配線は `train-backward-gemm-wiring.md` を参照）。
2. burn の更新ループが毎 step 新規グラフノード（`from_inner().require_grad()`）
   を生成する構造（§6 表）。fandhe reuse の `device_update` が同種のノード
   再生成を伴うかは本 doc では確認していない（未確定）。

## 9. 本 doc で読み切れなかった点（Phase 3・#2098 への申し送り）

- **burn wgpu（Metal ホスト側）の同期実装詳細**（§4.5）: `cubecl-wgpu` の
  ポーリング方式まで読み切れていない。
- **PyTorch CUDA／MPS のストリーム同期**（§4.5）: 解析契約（ディスパッチ層
  まで）の範囲内でも、本 doc では時間予算の都合でディスパッチ層自体の読解
  まで到達していない。
- **3 者のメモリプール実装の詳細**（§4.6）: 存在確認に留まる。
- **mean/relu/sqr の backward 呼び出し詳細**（§5 表の「未読解・未確定」欄）:
  candle・burn とも grep で存在は確認したが逐語の呼び出し回数までは全ては
  数えていない（`Op::Reduce`・`UnaryOp::Relu` 以外の周辺 variant）。
- **candle `Var::set` が storage 差し替えかコピーか**（§5 表）: `variable.rs`
  は本 doc の時間予算内で読めていない。
- **`device_update` の勾配 H2D 残存量の CUDA／Metal 側食い違い**（§7.2）:
  `scripts/bench/framework-compare/README.md` は「CUDA／Metal は resident
  未対応で全パラメータぶん H2D」と記すが、`docs/device-resident-update-design.md`
  の対応表は CUDA・Metal とも weight 勾配は resident（bias のみ CUDA は
  host・Metal は #1566 以降 resident）と記す。本 doc では pin 済み
  `fandhe-ai-backend-cuda =0.9.0` の実ソース（`ops.rs::gemm_fp32_strict_into`
  がオーバーライド済み）で CUDA weight resident のみ確認し、Metal 側は
  実ソース未確認（macOS 環境が本 doc の作業環境にないため）。README 側の
  記述がどの時点の状態を指すか（記述更新漏れの可能性を含む）の特定は
  行っていない。
- **burn `Tensor::clone()` の実コスト**（参照カウントのみか実データコピーか。
  §5・§6 表）: `burn-tensor` の `TensorPrimitive`／backend 実装まで確認して
  いない。
- 閉源部（cuBLAS／MPS）の到達限界: 本 doc は OSS（candle・burn・cubecl）の
  ディスパッチ層読解と、PyTorch の autograd codegen（OSS）読解に留まり、
  cuBLAS・MPS 自体のカーネル実装には触れていない（解析契約どおり）。

## 10. スコープ外

- マイクロ計測（Phase 3）・最適化提案は本 doc の対象外（§8 の新規候補提示に
  留める）。
- `bench_py.py` の `TF`（TensorFlow）・`SciPy` クラスは対象外。`SciPy` は
  BLAS 直呼び + 手書き NumPy backprop（`bench_py.py` L165-198 で読める）で
  あることのみ触れ、詳細解析は #2095 へ委ねる。
- CPU BLAS／`gemm` crate 内部は #2092／#2094 の担当（`candle-cpu-03.md` 等）。
- GEMM カーネル内部（Tensor Core・split-K 等）は #2090・#2093・#2096 の担当
  （`burn-cubecl-cuda-matmul.md`・`candle-0.11.0-cuda-path.md` 等）。
- `docs/README.md` の `analysis/` ブロック重複（並列マージ由来。L18 と L157
  付近の 2 ブロック）の解消は本 doc の対象外（§11 に申し送り）。

## 11. 出典・ライセンス

- `candle-core` 0.11.0: MIT OR Apache-2.0（§2 実測）。ソース:
  <https://static.crates.io/crates/candle-core/candle-core-0.11.0.crate>
- `burn-tensor`／`burn-autodiff`／`burn-ndarray`／`burn-wgpu`／`burn-cuda`
  各 0.21.0・`cubecl-runtime` 0.10.0: いずれも MIT OR Apache-2.0（§2 実測）。
  ソース: crates.io（ローカル registry cache 経由。sha256 一致確認済み）
- PyTorch: BSD-3-Clause（§2 実測。`LICENSE` ファイル）。ソース:
  <https://github.com/pytorch/pytorch> タグ `v2.14.0`
  （`tools/autograd/derivatives.yaml`・`tools/autograd/gen_autograd_functions.py`）
- 上記いずれもコード・派生物は本リポジトリへ持ち込んでいない（§12 参照）。

## 12. 解析契約の遵守宣言

- 上流コード（`candle-core`・`burn-*`・`cubecl-*`・PyTorch）はいずれも
  読み取りのみで、ビルド・実行は行っていない。
- 上流コードの引用ブロック（上流パスを冠したコードフェンス）は本 doc に
  含めていない。§4 の `let lhs_grad = ...` 等の短い断片は解析の根拠説明の
  ための言い換え・要約であり、上流ファイルの節を丸ごと引用したものではない
  （行番号参照で裏取り可能な形にしている）。in-repo ハーネス（§3）の逐語
  列挙は許容範囲内（対象は本リポジトリ自身のコード）。
- `Cargo.toml`／`Cargo.lock`／`docs/spec/`／tolerance／baseline／ガードレール
  閾値はいずれも変更していない。依存の追加・更新はない。
- candle・burn の取得物（`.crate` 展開物）と PyTorch の取得ファイルは
  scratchpad に置き、本 doc 作成後に削除済み（成否に関わらず削除する方針の
  とおり）。ローカル cargo registry cache（`burn-*`）はビルド用の既存
  キャッシュを読解のみに使い、削除はしていない（登録済みの通常の cargo
  キャッシュであり本 doc 固有の取得物ではないため）。
