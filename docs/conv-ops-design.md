# Conv1d／Conv2d 実装方式（im2col＋GEMM 対 直接畳み込み）設計

> 注記: 本 doc のファイル名は `conv-ops-design` とし、`docs/perf/` 配下の
> カーネル計測 doc 群（`docs/perf/cpu-gemm-*.md` 等）とは別概念（実装方式・
> VJP・数値契約の設計記録）であることを明示する。

## 0. 前提・スコープ

本 issue（#1641）は **設計 doc の作成のみ**で、コード変更・カーネル実装・
テスト追加は含まない。実装は後続 4 issue（親 #1606 配下）へ引き継ぐ:

- #1642: `backend-cpu` 実装
- #1643: `backend-cuda` 実装
- #1644: `backend-metal` 実装
- #1645: `nn::Conv` 層・parity・両実機実測

対象は `docs/compat-api-scope.md` §1.2「Conv1d／Conv2d | #1606」・
`docs/compat-feature-gap.md` §2.7「`nn.Conv1d`/`Conv2d`」（難度 XL・
「なし」）が指す欠落機能で、spec REQ-9 2026-09-12 追記（
`docs/spec/04-requirements.md:231`）の Tier 1 に列挙されている。

本 doc は HEAD `c9d7a21a906e2ad6e2e7dce706678f55c10a3ed5` 時点のコードを
根拠とする。tolerance・`BASELINES` 等の既存 baseline は変更しない。

### 0.1 承認状況

Issue #1641 のコメント（2026-09-12・ユーザー承認）により、本ツリーについて
以下は**承認済み**で着手可:

- `fandhe-ai-tensor-core` の `BackendOps` trait 拡張
- facade 公開面（`fandhe_ai`／`compat`）の `docs/compat-api-scope.md` §5
  範囲拡張

**範囲外**（別途個別承認が必要。本 doc はいずれも不要とする設計）:

- tolerance 定数・`BASELINES` 等 baseline の変更（緩和）
- 依存クレートの追加・更新
- `unsafe` 使用時の security-auditor 監査の省略

### 0.2 pooling doc（#1727）との整合確認

`docs/pooling-ops-design.md` §0.1／§12 は「NCHW レイアウト・`stride`／
`padding`／`dilation` の引数命名規約は #1641 が確定する共有規約と整合させる
べき」「#1728 着手前に整合を取り直す必要がある」と保留していた。本 doc §2
で以下のとおり確定し、この保留を解消する:

- レイアウト（NCHW／NCL）・1d→2d 併合・引数命名（`kernel_size`／`stride`／
  `padding`／`dilation`）は pooling doc と**同一規約**。
- `ceil_mode` は Conv1d／Conv2d の PyTorch API に存在しないパラメータであり
  齟齬なし（Pooling 固有の関心事）。
- `padding` の許容範囲は pooling（`padding ≤ floor(kernel/2)` 相当の上限
  あり）と Conv（**上限なし**）で意図的に異なる（§3 で根拠を示す）。

## 1. コード事実（根拠。HEAD `c9d7a21a` 時点）

後続実装（#1642〜#1645）が再利用・踏襲するパターンを、実コードの参照箇所
とともに列挙する。

- **CPU GEMM の bit 契約**: `crates/backend-cpu/src/gemm_blis/mod.rs:28-36`
  の doc comment（`gemm_blis`／`gemm_blis_parallel` は `gemm_naive` の
  p 昇順 `f32::mul_add` 連鎖と bit 完全一致。split-k・後加算方式は不採用と
  明記）。`crates/backend-cpu/src/ops.rs:592`（`CpuBackendOps::gemm` →
  `gemm_into_slice`）・VJP 専用 NT/TN zero-copy 入口
  `gemm_blis_parallel_nt`／`_tn`（`gemm_blis/mod.rs:651,670`。一般 stride
  非対応・呼び出し元が dense 転置 view を判定済みという契約）。
- **ホスト参照 `eval::matmul`**: `crates/autodiff/src/eval.rs:11-12`
  （FMA 契約の doc comment）・`eval.rs:300-322`（`pub(crate) fn matmul`。
  `i, j` の外側ループの中で `p` を 0..k 昇順に走査し `acc = a.mul_add(b,
  acc)` を積み上げる。`MatrixLayout` の `transposed`／`ld` フラグで転置
  view を再パックなしに読む）。
  → CPU では「im2col＋GEMM」と「(c, kh, kw) を p 軸として昇順走査する
  `mul_add` 連鎖の直接畳み込み」が**構造的に bit 同一**になる（§7 の
  根拠）。
- **`gemm_batched`／`gemm_batched_fp32_strict`**:
  `crates/tensor-core/src/backend_ops.rs:720`（`gemm_batched` 既定実装。
  rank≥3・NumPy 互換バッチ broadcast。既定は per-batch `gemm` 合成〈
  `gemm_batched_via_per_batch_gemm`。`backend_ops.rs:2467`〉で bit
  同一）・`backend_ops.rs:733`（`gemm_batched_fp32_strict`。TF32 opt-in に
  追従しない VJP 専用入口）・`backend_ops.rs:684`（`gemm_fp32_strict`）。
  `crates/tensor-core/src/ops_shape.rs:97`（`matmul_out_shape`）・
  `ops_shape.rs:220`（`batched_matmul_plan`）。
- **`matmul_vjp` の rank≥3 経路**: `crates/autodiff/src/grad.rs`
  （`transpose_last2` の zero-copy view → `gemm_batched_fp32_strict` →
  `reduce_batch_axes_f64` で broadcast バッチ軸を `f64` 縮約）。
  `reduce_batch_axes_f64`（`grad.rs:2477-2525`）は `target_shape` の rank
  が `g` より小さい場合に先頭へ `1` を補う NumPy 互換 prepend 方式
  （`rank_diff` 処理・`g.numel() == 0` の空勾配は `Tensor::zeros` 早期
  return。PR #1810 codex-review P2 是正）を実装する。この prepend 方式
  により、Conv の d_weight `[N, G, Cout_g, K_g] → [G, Cout_g, K_g]`
  （rank 4 → 3）は `w_mat` を `[1, G, …]` へ reshape せずそのまま
  `reduce_batch_axes_f64` を再利用できる（確認済み）。
- **bias 勾配の `f64` 規約と `Op::Add` の限界**: `.claude/rules/
  coding-rust.md`「正規化統計・勾配の長軸縮約」節（要素を伴わない単純な
  行方向和は要素を直接 `f64` へ昇格して蓄積する規約）。`grad.rs:235-236`・
  `grad.rs:312-316`・`grad.rs:702`（`Op::Add`／`Op::LinearAct`／
  `Op::LinearResident` の各 VJP が `reduce_bias_grad` を呼ぶ 3 箇所）。
  `reduce_bias_grad`（`grad.rs:2634-2645`）の適用条件は **`g` が rank-2
  `[m, n]` かつ `target_shape` が `[n]`／`[1, n]`（末尾軸一致・それ以外の
  軸がすべて 1）の行方向縮約に限る**（`is_row_axis_reduction` の判定式）。
  それ以外は `reduce_to_shape`（`grad.rs:2535`。`f32` 逐次和）へ落ちる。
  Conv の bias broadcast は `[N, Cout, Hout, Wout]` → `[1, Cout, 1, 1]`
  （rank-4・中間軸〈channel〉縮約かつ N／Hout／Wout の複数軸縮約）のため
  `reduce_bias_grad` の row-axis 判定に**該当せず**、`Op::Add` へ合成する
  と `f32` 経路（`reduce_to_shape`）になり `.claude/rules/coding-rust.md`
  の `f64` 規約に反する。→ 専用 `Op::Conv2d` を設け VJP 内で明示的に
  `f64` 縮約する根拠（§6・§8）。
- **`Op::Pad` の VJP＝narrow 連鎖**: `crates/autodiff/src/grad.rs:1239-
  1250`（`Op::Pad` の VJP が narrow の合成で表現される）・
  `crates/autodiff/src/var.rs:2211`（`pad_with_fallback`。`Unsupported`
  時のみ `eval::pad` へフォールバック）。Conv は padding をカーネル添字に
  暗黙化して扱う（im2col が padding 位置に `0.0` を書く）ため、`Var::pad`
  を明示合成しない（§5）。
- **決定的集約・`f64` 相当縮約の到達点**: `backend_ops.rs:200`
  （`enum ScatterReduce`・`Add` バリアントの決定的契約）・
  `crates/backend-metal/src/soft_f64.rs`（binary64 逐次加算の 64bit 整数
  ソフトウェアエミュレーション。ホスト `f64` 逐次和との bit 完全一致の
  逐語モデル）・`crates/backend-cpu/src/gather_scatter.rs:166`（`scatter`
  の単一スレッド逐次実装）。col2im の縮約契約（§6）はこれらと同型。
- **`Op` 配線の規約**: `crates/autodiff/src/tape.rs`（`is_checkpoint_
  eligible`・`for_each_input` 網羅 match による poison 伝播・
  `push_eager` による eager 演算の poison 伝播。`docs/autodiff-checkpoint-
  design.md` §3.4／§3.5／§3.5.1 参照）。
- **`Module` trait**: `crates/autodiff/src/nn/module.rs:30-45`
  （`forward`／`forward_host` の 2 メソッド。`forward_host` は tape を
  経由せず `ops` を直接呼ぶホスト常駐経路。イシュー #1028）。
  `crates/autodiff/src/nn/linear.rs:24-27`（`Linear` の weight
  `[in_features, out_features]` 格納。「`x.matmul(w)` 慣習・転置は
  持たない」と明記）・`linear.rs:41-72`（`uniform_init` bound
  `1/√in_features`・`in_features == 0` を `AutodiffError::InvalidArgument`
  で拒否）・`linear.rs:131-138`（`bind` が `Tape::var` で weight／bias を
  葉ノード登録）。`nn/init.rs`（`uniform_init`／`derive_seed`。
  `WEIGHT_SEED_SALT`／`BIAS_SEED_SALT` で独立導出）。
- **REQ-8 静的テストの先例**: `crates/backend-cuda/src/kernels_constant_
  pad.rs`（`kernel_includes_bounds_check` 型の静的テスト）・
  `kernels_gather_scatter.rs`。Metal は `crates/backend-metal/tests/
  constant_pad_source_evidence.rs`。
- **`Tensor` の view API**: `crates/tensor-core/src/tensor.rs:425`
  （`transpose`）・`tensor.rs:475`（`permute`）・`tensor.rs:505`
  （`narrow`）・`tensor.rs:653`（`broadcast_to`）。任意 stride を表現する
  `as_strided` 相当の API は存在しない。→ 重なりを持つ 6 次元 strided
  view として im2col を表現する「ゼロコピー im2col」は現行 `Tensor` では
  表現不能であり、im2col は実体化（コピー）カーネルとする根拠（§5）。
- **既存カーネルに Conv／im2col は存在しない**（`grep -rni "im2col|
  unfold|conv2d|conv1d|convolution" crates --include='*.rs'` が 0 件、
  `docs/onnx-export-op-mapping.md` にも ONNX `Conv` の逆マッピングは無い。
  §11 スコープ外）。

## 2. 対象演算・レイアウト（決定）

- 対象: `Conv1d`／`Conv2d`（PyTorch `nn.Conv1d`／`nn.Conv2d`・
  `F.conv1d`／`F.conv2d` の **cross-correlation**（カーネル反転なし）
  意味論）。
- レイアウト: **NCHW（2d: `[N, Cin, H, W]`）／NCL（1d: `[N, Cin, L]`）
  固定**。channels_last は対象外（§11）。pooling doc §2 と同一規約
  （§0.2 参照）。
- **weight は PyTorch 準拠 `[Cout, Cin/groups, kH, kW]`**（1d は
  `[Cout, Cin/groups, k]`）・bias は `[Cout]`。`nn::Linear` が
  `[in, out]` 格納（`linear.rs:24-27`）で PyTorch と異なる点は既存事実
  として併記し、Conv では state_dict／safetensors 相互運用（#1616）と
  im2col GEMM の自然な行列形（`w: [G, Cout_g, K_g]` への reshape が
  contiguous view でゼロコピー）を理由に PyTorch 順を採る。
- **1d は 2d へ併合**: `[N, C, L]` → `Var::reshape`（view・bit 同一）で
  `[N, C, 1, L]`、weight `[Cout, Cin_g, k]` → `[Cout, Cin_g, 1, k]`、
  H 軸パラメータは `(kernel=1, stride=1, padding=0, dilation=1)` 固定。
  `BackendOps`・カーネルは 2d 版のみ新設（pooling §2 と同型）。
- 引数命名: `kernel_size`（`Var::conv2d` の**公開シグネチャ**では weight
  shape から導出し引数としては持たない）・`stride: [usize; 2]`・
  `padding: [usize; 2]`・`dilation: [usize; 2]`・`groups: usize`（`Var`
  入口はプリミティブ引数。1d は `usize`）。pooling doc と同一命名
  （§0.2）。**ただし** `im2col`／`col2im`（`BackendOps` メソッド。§8）は
  `weight` を受け取らない契約のため、`Conv2dParams`（§8）は
  `kernel_size: [usize; 2]` を保持フィールドとして持つ（`Var::conv2d`
  が weight から導出した値を `Conv2dParams` 構築時に埋める。公開 `Var`
  シグネチャに `kernel_size` 引数を追加するものではない）。

## 3. パラメータと検査（A03。`Var` 入口で `AutodiffError::InvalidArgument`
／`Shape`、`BackendOps` 実装側でも fail-closed に再検査する）

PyTorch `aten/src/ATen/native/Convolution.cpp::check_shape_forward`
（`check_shape_forward` 内の各検査。2026-09-14 に `pytorch/pytorch` main
を確認）・`ConvUtils.h::_conv_output_size` の意味論に整合させる。

- rank: input 4（1d 併合後も 4）・weight 4・bias は `Some` なら rank 1。
- `stride ≥ 1`・`dilation ≥ 1`（PyTorch の non-positive stride 拒否
  相当）・`groups ≥ 1`（同 groups 拒否相当）。
- `padding` は **任意の非負整数**を許可する（PyTorch の `_conv_output_
  size` に `padding ≤ k/2` の制約は無い）。**pooling doc §3 の
  `padding ≤ floor(kernel/2)` 上限は Conv には適用しない**（§0.2 で
  記した意図的な差異）。`2·padding` は `checked_mul`／`checked_add` で
  overflow を拒否する。
- チャンネル整合（PyTorch `check_shape_forward` 相当）: `Cin % groups ==
  0`・`weight.shape[1] · groups == Cin`・`Cout % groups == 0`・
  `Cout ≥ groups`（`Cout_g ≥ 1`）・`bias.shape == [Cout]`。
- `Cin_g ≥ 1`・`kH, kW ≥ 1`（`Linear::new` の `in_features == 0` 拒否
  〈`linear.rs:41-45`〉と同じく `InvalidArgument`。K = 0 の GEMM を
  作らない）。
- 出力長: 各空間軸で `in + 2p ≥ d·(k−1)+1` を要求（PyTorch の "Kernel
  size can't be greater than actual input size" 相当）。実装契約は
  pooling §4 と同じ**負分子拒否ゲート**（分子 `in + 2p − d(k−1) − 1` を
  `checked_sub` で計算し負なら floor 除算せず `ShapeError`。非負なら
  整数除算＝floor）。
- **空間軸 `H`／`W`（1d 併合後は `W` 軸）`= 0` は `ShapeError` で拒否**。
  PyTorch は `in=0, p≥1` で padding だけの窓を受理し出力＝bias になり
  得るが、本設計は pooling §3 と同じく `in = 0` 自体（空間軸が存在しない
  入力）を im2col／col2im の境界契約単純化のため入口で拒否する
  （**意図的な PyTorch 非互換**として記録）。**訂正**（codex-review
  指摘）: 「すべての窓が少なくとも 1 つの有効入力を含む」は不変条件
  として**成立しない**（例: `H=W=1, kernel=1, padding=1` は `in=1 ≠ 0`
  のため本ゲートを通過するが、出力窓は padding のみで有効入力を含まな
  い）。本設計が実際に保証するのはより弱い契約——**`in = 0` の入力を
  作らせない**（負分子拒否ゲート・要素数 overflow 検査と合わせて
  im2col／col2im の座標計算〈`h + p_h − kh·d_h` 等〉が `usize` 下限を
  割らないことを保証する）——のみであり、個々の出力窓が padding のみ
  で埋まること自体は許容する（§7 の「罠」節が示すとおり、padding
  タップは `0.0` として im2col／`eval::conv2d_direct` の両方で明示的に
  扱われるため、有効入力ゼロの窓でも bit 同一性は崩れない）。`N = 0`
  （空バッチ）は受理（出力 `[0, Cout, Hout, Wout]`）。
- 要素数積（`N·G·K_g·P`・出力要素数・im2col 要素数）は `checked_mul`
  （`ElementCountOverflow`）。
- `padding_mode` は `zeros` のみ（reflect／replicate／circular は §11）。
  `padding='same'`／`'valid'` 文字列は非対応（整数のみ。§11）。

## 4. 出力 shape 関数（`tensor-core::ops_shape` へ新設予定。設計のみ）

```text
conv_out_len(in, k, s, p, d) = floor((in + 2p − d(k−1) − 1) / s) + 1
    （PyTorch ConvUtils.h::_conv_output_size と同式）
conv2d_out_shape(input, weight, stride, padding, dilation, groups)
    -> [N, Cout, Hout, Wout]
im2col_out_shape(...) -> [N, G, Cin_g·kH·kW, Hout·Wout]
```

`pool_out_len`（pooling §4）と同式だが検査規則（padding 上限の有無）が
異なるため**別関数**として定義し共有しない（誤って pooling の上限検査を
Conv に持ち込まない）。

## 5. forward: im2col＋GEMM（決定。直接畳み込みはテストオラクル限定）

### 5.1 選定理由（比較）

| 案 | 概要 | 評価 |
|---|---|---|
| A. im2col＋GEMM | im2col でコピーし既存 GEMM を再利用 | **採用**。新設カーネルは im2col（純コピー）・col2im（`f64` 縮約）の 2 種のみ。3 バックエンド既存 GEMM（CPU BLIS 2D 動的分配・CUDA cp.async pipeline／mma・Metal simdgroup／split-K）・REQ-2・FMA 契約・既存 `gemm_*_parity` テスト資産をそのまま継承 |
| B. 直接畳み込みカーネル | forward／backward 専用の積和カーネルを新規実装 | 3 バックエンド × forward で最低 3 カーネル・GPU backward 相当も必要。数値契約を GEMM と別に定義し直す必要。性能上の利点（メモリ非膨張）は kH·kW が小さい形状に限られ、v1 の受け入れ条件（機能到達・parity）に対して過大 |
| C. implicit GEMM／Winograd／FFT | CUDA 等でメモリ膨張を避けつつ GEMM 並みの効率を得る手法 | CUDA 限定・実装コスト大。将来の性能 issue（§11）へ |

**結論**: v1 は案 A を採用し、CPU の案 A を**数値参照**とする。案 B は
`eval::conv2d_direct`（ホスト）として**テストオラクル**（案 A と bit 同一
であることの回帰検証）に限定して実装し、本番経路には結線しない。

### 5.2 計算式

groups を `gemm_batched` の broadcast で吸収する（新規バッチカーネル
不要）。

1. `col = im2col(x, stride, padding, dilation, groups)`:
   `[N, G, K_g, P]`（`K_g = Cin_g·kH·kW` を **(c_in_g, kh, kw) の
   row-major** で並べる・`P = Hout·Wout` を (oh, ow) row-major で並べる）。
   padding 位置は `0.0` を書く（padded 入力を実体化しない。境界は添字で
   手動検査〈REQ-8〉）。
2. `w_mat = weight.reshape([G, Cout_g, K_g])`（contiguous 前提の view。
   ゼロコピー）。
3. `out_mat = gemm_batched(w_mat, col)` → バッチ shape `[G]` と
   `[N, G]` の NumPy 互換 broadcast で `[N, G, Cout_g, P]`。
4. `out = out_mat.reshape([N, Cout, Hout, Wout])`（`G·Cout_g = Cout`・
   contiguous のためゼロコピー）。
5. bias: `out[n, c, :, :] += bias[c]`（**GEMM 後の独立した f32 加算
   パス**。`LinearVars::forward` の `matmul → add` 合成と同じ「FMA 連鎖の
   後に 1 回加算」。`gemm_bias_act` の epilogue 融合は bias が列 `[n]`
   向けで行 `[Cout]` 向けの本ケースに合わないため v1 では使わず、将来
   融合する場合もこの合成と bit 同一であることを契約とする）。
   **実装上の注意（Bugbot 指摘の是正）**: `bias` の shape は `[Cout]`
   だが、`out`（手順 4 の結果）は `[N, Cout, Hout, Wout]` で `Cout` が
   末尾から 2 番目の軸にある。`bias`（`[Cout]`）をそのまま
   `ops.add(&out, &bias)` へ渡すと NumPy 互換の**右詰め**
   ブロードキャスト（`broadcast_shape`。`backend_ops.rs` 既定実装が
   採用する規約）により `Cout` が `out` の**末尾軸（`Wout`）**と対応
   してしまい、`Cout ≠ Wout` の形状では shape エラーに、たまたま
   `Cout == Wout` の形状では**誤った軸へ無言で加算**される（サイレント
   な誤り。テストで見逃しやすい）。正しくは `bias` を
   `Var::reshape([1, Cout, 1, 1])`（ゼロコピー view）してから
   `out.add(&bias_reshaped)` を呼ぶ——これにより `Cout` 軸が明示的に
   `out` の `Cout` 軸（axis 1）と揃い、残る軸（`N`／`Hout`／`Wout`）は
   サイズ 1 として正しくブロードキャストされる。`Op::Conv2d` の VJP
   （§6.3 d_bias）は逆に `[N, Cout, Hout, Wout]` → `[1, Cout, 1, 1]`
   （のち `[Cout]` へ reshape）の縮約であり、この reshape 済み形状が
   forward の bias 加算と対称であることを回帰テストで固定する（§13）。

### 5.3 合成の置き場所（決定。フォールバック階層の設計）

上記手順 1〜5 の**段階的合成は `autodiff` 側のヘルパ
`conv2d_with_fallback`（`pad_with_fallback`／`matmul_vjp` と同型）に
置く**。`tensor-core` の trait 既定実装として合成を書くと、`self.im2col`
が `Unsupported` を返すバックエンド（#1643／#1644 着地前の CUDA／Metal、
および将来の未対応バックエンド）では既定 `conv2d` 全体が `Unsupported`
となり `Var::conv2d` が全ホスト実装へ落ちて **GPU GEMM が完全に迂回
される**（trait 既定実装は `autodiff::eval::im2col` を参照できず段階的
フォールバックを組めない）ため採らない。

ヘルパの段階:

1. `ops.conv2d`（override フック。既定 `Unsupported`）→ `Unsupported`
   のときのみ次段へ。
2. `im2col_with_fallback`（`ops.im2col`、`Unsupported` のときのみ
   `eval::im2col`）。
3. **`ops.gemm_batched`（常にバックエンド GEMM。ホストフォールバック
   なし。`gemm` は全バックエンド必須メソッド）**。
4. `ops.add` で bias（ブロードキャスト。`Unsupported` 契約を持たない
   必須メソッド）。

`eval::conv2d`（全ホスト im2col＋`eval::matmul` 合成）は**本番
フォールバックではなくテストオラクル／全ホスト参照専用**とする。

**注意**: 手順 3 の既定 `gemm_batched` は `w_mat` を
`[N, G, Cout_g, K_g]` へ `broadcast_to → contiguous()` で N 倍実体化する
（`backend_ops.rs:720` 既定実装の挙動）。v1 は受容し、
`conv2d_with_fallback` 側で「N をチャンク分割して per-chunk に手順
2〜4 を呼ぶ」ことで col と broadcast weight の同時メモリを上限化する
（§10）。

## 6. VJP（d_input は転置畳み込み・d_weight は相関）

`Op::Conv2d { input, weight, bias: Option<NodeId>, stride, padding,
dilation, groups }`（`col` は Op に**キャッシュしない**。backward で
`im2col` を再計算する。理由: col は入力の `kH·kW` 倍のメモリで、決定的
コピーのため再計算コストは GEMM より小さい。`docs/autodiff-checkpoint-
design.md` の「再計算で実メモリを減らす」方針と整合）。

`g = upstream.reshape([N, G, Cout_g, P])`。

### 6.1 d_weight（相関）

```text
dw_full = gemm_batched_fp32_strict(g, colᵀ)   // colᵀ は transpose_last2 の zero-copy view
        -> [N, G, Cout_g, K_g]
dw = reduce_batch_axes_f64(dw_full, [G, Cout_g, K_g])   // N 軸を f64 で縮約
        .reshape([Cout, Cin_g, kH, kW])
```

`reduce_batch_axes_f64`（`grad.rs:2477`。既存ヘルパ。rank prepend
対応のため reshape 不要）で N 軸を `f64` アキュムレータで縮約する（勾配の
長軸縮約規約）。これは「入力と上流勾配の相関」の定義式そのものである。

注意: `dw_full` は `N·Cout·K_g` 要素を一時確保する（§10 で N を
チャンク分割した per-chunk 実行を許容）。**チャンク境界をまたぐ `f64`
縮約の正しさ条件（codex-review 指摘の是正）**: 「チャンクごとに独立し
た部分和を `f64` で計算し、最後にそれらの部分和同士を加算する」方式
は逐次 `f64` 和と bit 同一に**ならない**（`f64` 加算は結合則を満たさ
ないため。反例: `[2^60, 0, −2^60, 1]` を 2 要素ずつ 2 チャンクに分割
すると `(2^60 + 0) + (−2^60 + 1) = 1` だが、逐次和は
`((2^60 + 0) + (−2^60)) + 1` も同じ `1` になる一方、丸めが発生する
より一般の値では部分和の加算順序が逐次順と異なれば結果が乖離しう
る）。正しい実装契約は「**単一の `f64` アキュムレータを N 軸全体で
維持し、チャンク境界はメモリ確保の単位にすぎず縮約順序に影響しない**」
——各チャンクの処理は、直前チャンクまでの累積値を引き継いだ同一の
`f64` アキュムレータへ、そのチャンクが担当する n を昇順に 1 要素ずつ
逐次加算する（チャンク単位で独立した部分和を計算してから後で合算す
る二段階リダクションにはしない）。この契約下ではチャンク分割数に
依らず逐次 `f64` 和と bit 同一になる（§10 の「分割数に依らず bit
同一」という記述はこの単一アキュムレータ方式を前提とする）。

### 6.2 d_input（転置畳み込み）

```text
d_col = gemm_batched_fp32_strict(w_matᵀ, g)   // w_matᵀ: [G, K_g, Cout_g]・broadcast で [N, G, K_g, P]
d_input = col2im(d_col)
```

col2im（fold）は転置畳み込みそのものであることを式で示す:
`d_input[n, c, h, w] = Σ_{(kh, kw): (h, w) が窓 (kh, kw) に属する}
d_col[n, g, (c_g, kh, kw), (oh, ow)]`。**独立した conv_transpose カーネル
は作らない**（col2im が同機能を提供する。`ConvTranspose2d` 層は §11 で
col2im 再利用を前提に後続へ）。

**col2im の縮約契約（決定）**: 入力位置定常（1 スレッド＝1 入力要素・
atomic 不使用）。各 `(n, c, h, w)` について `(kh, kw)` を row-major に
走査し、`h + p_h − kh·d_h` が `s_h` で割り切れ `oh ∈ [0, Hout)`
（`ow` も同様）のときのみ `d_col[n, g, (c_g, kh, kw), (oh, ow)]` を
**`f64` へ昇格して逐次加算**、最後に 1 回 `f32` へ downcast。CUDA は
`double`・Metal は `soft_f64`（binary64 逐次加算の 64bit 整数
ソフトウェアエミュレーション）・CPU は `f64`。重なり窓（`stride <
d·(k−1)+1`）の加算順を固定し 3 バックエンド **bit 完全一致**
（`ScatterReduce::Add`／pooling §8 と同型）。padding 領域への寄与は
書き込まない（境界検査）。

### 6.3 d_bias

```text
d_bias[c] = Σ_{n, oh, ow} upstream[n, c, oh, ow]
```

(n, oh, ow) の row-major 固定順で **`f64` 逐次加算**し 1 回 downcast
（`.claude/rules/coding-rust.md`「要素積を伴わない単純な行方向和は要素を
直接 `f64` へ昇格」）。実装は
`upstream.permute([0,2,3,1]).contiguous().reshape([N·P, Cout])` →
`eval::reduce_bias_grad_rows`（既存の `f64` 行縮約。`reduce_bias_grad`
〈`grad.rs:2634-2645`〉が使う関数）の再利用で可能（Metal GPU 版は
§11）。

### 6.4 フォールバック階層

VJP は `grad.rs` ホスト側で編成し、§5.3 と同じ粒度でフォールバックする:
GEMM は**常に `ops.gemm_batched_fp32_strict`**（ホストフォールバック
なし。GPU では GPU GEMM）・im2col の再計算は `im2col_with_fallback`・
col2im は `col2im_with_fallback`（`ops.col2im`、`Unsupported` のとき
のみ `eval::col2im`）（`matmul_vjp`／`scatter_with_fallback` と同型）。
GPU backward 専用カーネルは v1 では作らない（§9）。

`bias: None` の場合は d_bias を生成しない。`requires_grad` 非学習葉への
d_input スキップは `docs/autodiff-nograd-leaf-dinput-skip-decision.md`
の方針に従う（本 doc で新規判断はしない）。

## 7. 数値契約（tolerance／baseline 不変）

- **CPU（参照実装）**: `out[n, co, oh, ow] = bias[co] + Σ_{(c, kh, kw)
  昇順} w · x` を **`acc = 0.0` から (c, kh, kw) 昇順の `f32::mul_add`
  連鎖**で求め、最後に bias を 1 回加算。im2col＋`CpuBackendOps::gemm`
  （`gemm_blis` ＝ p 昇順 `mul_add`〈`gemm_blis/mod.rs:28-36`〉）と
  `eval::conv2d_direct`（同順 `mul_add`）は **bit 完全一致**。
  **罠を明記**: 直接畳み込みオラクルは padding タップを**スキップせず
  `0.0.mul_add(w, acc)` として実行**する（スキップすると `w` が
  `inf`／`NaN` の場合や符号付きゼロで結果が変わり、im2col〈padding
  位置に `0.0` を書き GEMM で積和〉と一致しなくなる）。
- **GPU（CUDA／Metal）**: im2col は純コピー（bit 完全一致）・col2im は
  `f64` 相当縮約（bit 完全一致）・GEMM 段は既存 `matmul` parity と同じ
  **REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）**。
  よって Conv 全体の 3 バックエンド parity 判定は REQ-2 複合判定（GEMM
  由来）で行い、im2col／col2im 単体は bit 一致で判定する **2 層構成**。
  TF32 opt-in（`set_cuda_gemm_precision`）は forward の `gemm_batched` に
  追従し、VJP は `_fp32_strict` で追従しない（`matmul` と同じ扱い）。
- bias 加算は f32 1 回加算（3 バックエンドで同一順序）。
- **`tolerance`・`BASELINES` は変更しない**。GPU parity で REQ-2 複合
  判定が成立しない形状が実機で見つかった場合は #1645 で baseline 方式
  （`docs/cuda-tensor-core-parity-judgment-decision.md`）の適用可否を
  別途ユーザー承認に回す（本 doc では先取りしない）。

## 8. `BackendOps`／`Op`／`Var`／`nn` 配線案（承認済み範囲内。実装は
#1642〜#1645）

- `BackendOps`（`crates/tensor-core/src/backend_ops.rs`）へ既定実装付き
  の非破壊拡張として 3 メソッド:
  - `im2col(x, &Conv2dParams) -> Result<Tensor<f32>, _>`（既定
    `Unsupported`）
  - `col2im(d_col, input_shape, &Conv2dParams) -> Result<Tensor<f32>,
    _>`（既定 `Unsupported`）
  - `conv2d(x, w, bias: Option<&Tensor<f32>>, &Conv2dParams) ->
    Result<Tensor<f32>, _>`（**既定 `Unsupported` の純粋な override
    フック**。`linear_forward_device` と同じ fail-safe 型。§5.3 の
    段階的合成は trait 既定実装には置かず `autodiff::conv2d_with_
    fallback` に置く。v1 では 3 バックエンドとも override せず合成経路
    を使う＝**CPU 参照実装は「`CpuBackendOps::im2col` →
    `CpuBackendOps::gemm_batched`〈BLIS〉→ `add`」の段階的合成そのもの**。
    将来 GPU が direct／fused カーネルで override する場合は CPU 参照
    実装に対し REQ-2 複合判定を満たすこと・CPU 自身が override する
    場合は合成経路と bit 同一であること）。
  - `Conv2dParams`（`#[non_exhaustive]`・コンストラクタ経由・
    `kernel_size: [usize; 2]`／`stride`／`padding`／`dilation`／
    `groups`）は tensor-core 内の型で facade へ再エクスポートしない。
    **`kernel_size` を保持する理由（codex-review 指摘の是正）**:
    `im2col`／`col2im` は `weight` を引数に取らない契約（上記シグネ
    チャ）のため、`kernel_size` を `Conv2dParams` 経由で渡さない限り
    カーネル寸法を知る手段がない。`Var::conv2d` 呼び出し時に
    `weight.shape()[2..4]` から導出して `Conv2dParams` を構築する
    （公開 `Var::conv2d` シグネチャ自体には `kernel_size` 引数を追加
    しない。§2 引数命名の注記と整合）。
- `Op::Conv2d`（`tape.rs`）: `push_eager` で常時実体化・
  `is_checkpoint_eligible = false`（v1。`Op::MatMul` は `true` だが
  Conv は 3 入力・bias Option の再計算経路整備を後続へ）・
  `for_each_input` へ `input`／`weight`／`bias` を登録（poison
  伝播）。
- `Var::conv2d(&self, weight: &Var, bias: Option<&Var>, stride:
  [usize; 2], padding: [usize; 2], dilation: [usize; 2], groups:
  usize)`・`Var::conv1d(.., stride: usize, padding: usize, dilation:
  usize, groups: usize)`（1d は reshape 併合の薄いラッパー）。forward
  は `conv2d_with_fallback`（§5.3 の段階的合成。`ops.conv2d` →
  `Unsupported` のときのみ im2col 段階へ・im2col は `Unsupported` の
  ときのみ `eval::im2col`・GEMM と bias 加算は常にバックエンド）を呼び、
  `Unsupported` 以外のエラーは伝播（判定迂回経路を作らない。A08）。
  facade 到達経路は既存 `Var` 再エクスポート経由（新規 `pub use`／
  `pub fn` を facade へ追加しない）。
- `nn::Conv2d`／`nn::Conv1d`（`crates/autodiff/src/nn/conv.rs` 新設
  予定）: `weight: Tensor<f32>`（`[Cout, Cin_g, kH, kW]`）・
  `bias: Option<Tensor<f32>>`。
  `new(in_channels, out_channels, kernel_size, stride, padding,
  dilation, groups, bias, seed)` は `uniform_init` bound
  `1/√(Cin_g·kH·kW)`（PyTorch `nn.Conv2d` の `kaiming_uniform_(a=√5)` と
  同じ有効範囲）・`derive_seed` で weight／bias を独立導出（`Linear::
  new` と同型。`linear.rs:41-72`）。`Module::forward`／`forward_host`
  （**同じ `conv2d_with_fallback` ヘルパを呼ぶ**ことで `forward` と
  `forward_host` の bit 一致を構造的に保証）。`from_parameters` で
  state_dict 系との接続点を残す。
  **学習可能パラメータの収集・更新経路への接続（codex-review 指摘の
  是正）**: `crates/autodiff/src/nn/module.rs` の `Module::as_linear`／
  `as_linear_mut` フックは `Linear` 専用の明示列挙方式（`docs/
  compat-api-scope.md` §1 の閉集合維持が目的。module.rs のコメント
  「現状 `Linear` のみ」）で、`facade::compat::Sequential::bind`／
  `trainable_parameters`／`apply_parameters`（`crates/facade/src/
  compat/sequential.rs`）は全箇所 `layer.as_linear()`／
  `as_linear_mut()` でフィルタする。**`nn::Conv2d` を `Linear` と
  同型に `parameters()` を実装するだけでは、この閉集合フィルタを
  素通りして `Sequential` の学習経路（`bind`／`trainable_parameters`／
  `apply_parameters`）から不可視のまま**（optimizer が weight／bias
  を一切拾えない）になる。#1645 で本経路へ接続するには、
  `Module` trait へ `as_conv2d`／`as_conv2d_mut`（`Linear` と同じ
  明示フック方式。`Any` ダウンキャストは使わない。`docs/
  compat-api-scope.md` §1 の閉集合方針に従い型を明示列挙）を追加し、
  `Sequential::bind`／`trainable_parameters`／`apply_parameters`
  （および `forward` の融合判定 `as_relu` 呼び出し箇所と対になる各
  `filter_map` 系コード）を `as_linear()`／`as_conv2d()` の**両方**を
  見るよう拡張する必要がある（`facade` 側の非破壊拡張。トップレベル
  `Module` trait は crates.io 公開クレート `fandhe-ai-autodiff` に
  属するためデフォルト実装 `None`／`false` を維持し既存実装を壊さな
  い）。本設計 doc はこの接続方式を確定するのみで実装しない（実装は
  #1645）。
- facade: `compat::Sequential::add_conv2d`／`add_conv1d` は §5 手続きの
  承認済み（Issue #1641 コメント 2026-09-12）だが、**追加するか否かの
  最終判断と実装は #1645** に委ねる（本 doc は「承認済み・実装は
  #1645」と記録するのみ）。
- モジュール命名（既存と非衝突）:
  `crates/backend-cpu/src/im2col.rs`（im2col／col2im）・
  `crates/backend-cuda/src/{im2col.rs, kernels_im2col.rs}`・
  `crates/backend-metal/src/{im2col.rs, shaders/im2col.metal}`・
  `crates/autodiff/src/nn/conv.rs`・
  `tensor-core::ops_shape::{conv_out_len, conv2d_out_shape,
  im2col_out_shape}`。

## 9. バックエンド別実装形（#1642〜#1644 への指針）

- **CPU（#1642）**: `CpuBackendOps::im2col`／`col2im` を単一スレッド
  逐次参照実装で override（`gather_scatter.rs` の規律。並列化は §11）。
  `conv2d` は override せず（`Unsupported` のまま）`conv2d_with_
  fallback` の合成経路（`CpuBackendOps::gemm_batched` オーバーライド
  〈#1715〉が BLIS を呼ぶ）を使う。
- **CUDA（#1643）**: `im2col` は 1 スレッド＝1 col 要素（`long long`
  座標・境界手動検査 REQ-8）、`col2im` は 1 スレッド＝1 入力要素で
  `double` 累積。NVRTC 静的文字列・`kernel_includes_bounds_check` 型の
  静的テスト。`conv2d` は override しない。GEMM は既存 `gemm_batched`
  既定（per-batch `gemm`。専用バッチカーネル #1716 に依存せず動作）。
  **#1643 着地前でも** `conv2d_with_fallback` により「ホスト im2col →
  CUDA GEMM → CUDA add」で動作する（GPU GEMM が迂回されない）ことを
  記す。
- **Metal（#1644）**: 同構成で `col2im` は `soft_f64`。`shader_source_
  evidence` 型の静的テスト。`conv2d` は override しない。
- 実機（GB10／M4 Max）依存テストは `#[ignore]` 分離・未実測時は各実装
  doc に記入欄を残す。

## 10. メモリ・性能上の注意（記録のみ・判定基準にしない）

- col の要素数 `N·Cin·kH·kW·P` は入力の `kH·kW·(P/(H·W))` 倍（例:
  `N=64, Cin=64, k=3, 56×56` で約 462 MB）。`conv2d_with_fallback` は
  **N をチャンク分割**（チャンクの col バイト数上限を定数化。値は
  実装 issue で決める）。forward（im2col＋GEMM＋bias）は per-sample
  GEMM の独立性によりチャンク分割数に依らず bit 同一。**backward の
  d_weight `f64` 縮約（§6.1）はチャンク分割数に依らず bit 同一である
  ために単一 `f64` アキュムレータを N 軸全体で維持する実装契約が必須**
  （§6.1 の「正しさ条件」参照。チャンクごとに独立した部分和を計算して
  後で合算する方式は不可）。
- broadcast weight の N 倍実体化（`gemm_batched` 既定）・`dw_full` の
  一時確保は v1 受容。回避策（バッチ軸を M 軸へ畳み込む
  `[N·P, K]ᵀ` 形式等）は §11 の性能 issue へ。
- 性能目標（REQ-8）は本 doc の対象外。framework-compare への Conv
  追加は §11。

## 11. スコープ外（`.claude/rules/out-of-scope-tracking.md` に従い後続
issue で追跡。本 doc では起票しない）

直接畳み込み／implicit GEMM／Winograd／FFT カーネル・depthwise
（`groups = Cin`）専用カーネル・conv3d・ConvTranspose1d／2d（col2im
再利用前提）・channels_last・`padding_mode ≠ zeros`・
`padding='same'`／`'valid'` 文字列・GPU backward 専用カーネル（d_bias
の GPU 縮約含む）・デバイス常駐推論チェーン（`linear_forward_device`
相当）・ONNX `Conv` export／import マッピング・CPU im2col／col2im 並列化・
`gemm_bias_act` epilogue 融合の Conv 適用・framework-compare への Conv
ベンチ追加・`compat::Sequential::add_conv*` の実装判断（#1645）。

## 12. 承認事項（#1642 着手前の前提）

- **承認済み**（Issue #1641 コメント 2026-09-12）: `BackendOps` trait
  拡張（`im2col`／`col2im`／`conv2d`）・facade §5 範囲拡張。
- **範囲外**（別途承認）: tolerance／baseline 変更・依存追加・`unsafe`
  監査省略。本設計はいずれも不要（`unsafe` 追加なし・依存追加なし）。
- pooling doc §12「#1641 との整合確認」: 本 doc §0.2／§2 で解消
  （NCHW・命名同一・`ceil_mode` 非該当）。

## 13. #1642〜#1645 への受入テスト一覧

- shape／引数検査の境界: `stride=0`／`dilation=0`／`groups=0` 拒否・
  `Cin % groups ≠ 0`・`Cout % groups ≠ 0`・
  `weight.shape[1]·groups ≠ Cin`・`bias` shape 不一致・
  `in + 2p < d(k−1)+1`（負分子ゲート）・`H=0`／`W=0` 拒否と `N=0` 受理・
  `checked_mul` overflow・**`padding > k/2` が受理される**こと（pooling
  との差異の回帰）・**`H=W=1, kernel=1, padding=1` 等の「有効入力を
  含まない窓」形状が `in=0` ゲートには引っかからず受理される**こと
  （§3 訂正後の契約の回帰。padding のみの窓でも forward／VJP が
  `0.0` タップとして正しく処理し bit 同一を保つことを併せて確認）。
- 出力 shape が PyTorch `_conv_output_size` と一致（複数 stride／
  padding／dilation 組合せの表）。
- **CPU bit 一致 3 点**: `eval::conv2d_direct` ≡ `eval::conv2d`
  （ホスト im2col＋`eval::matmul`）≡ `conv2d_with_fallback
  (CpuBackendOps)`（`CpuBackendOps::im2col` → BLIS `gemm_batched` →
  `add`）・NaN／inf を含む weight で padding タップ非スキップの回帰。
- フォールバック階層の回帰: `im2col`／`col2im`／`conv2d` がすべて
  `Unsupported` のテスト用 `BackendOps`（`backend_ops.rs` テストの
  `naive_gemm_2d` 型スタブ）で、`conv2d_with_fallback` が
  **`gemm_batched` をバックエンド経由で呼ぶ**（ホスト `eval::matmul`
  へ落ちない）ことを呼び出しカウンタで固定。
- groups: `groups=1` と `groups=G` を per-group `narrow`＋`groups=1`
  の合成で突合（bit 一致）。depthwise（`groups=Cin`）を含む。
- 1d と `[N, C, 1, L]` 2d の bit 一致。`forward` と `forward_host` の
  bit 一致。`BackendOps` 実装と `eval` フォールバックの bit 一致。
- VJP: 数値微分突合（`grad.rs` 既存ハーネス。input／weight／bias
  すべて・groups>1・dilation>1・`stride < 有効カーネル` の重なり窓）・
  `d_input` が「零埋め転置畳み込み」の手計算と一致・`d_weight` が
  「相関」の手計算と一致・d_bias の `f64` 縮約（`[1e8, 1, −1e8]` 型の
  相殺列で f32 逐次和と結果が異なることを固定）。
- col2im の重なり窓での決定的順序（3 バックエンド bit 一致・
  `#[ignore]` 実機）。
- GPU parity: forward 全体は REQ-2 複合判定・im2col／col2im 単体は
  bit 一致・REQ-8 静的テスト（CUDA `kernel_includes_bounds_check`・
  Metal source evidence）。
- 学習収束スモーク（小さな CNN を `Sequential` 相当で 1 層
  conv → relu → flatten → linear。`nn_train_convergence.rs` の先例）。
  **前提として `as_conv2d`／`as_conv2d_mut`（§8 是正）を実装し
  `Sequential::bind`／`trainable_parameters`／`apply_parameters` が
  `Conv2d` の weight／bias を実際に拾えること**を単体テストで固定して
  から収束スモークへ進む（閉集合フィルタの素通りにより optimizer が
  Conv パラメータを更新しない回帰を防ぐ）。
- bias broadcast の軸回帰（§5.2 是正）: `Cout == Wout` となる形状
  （例 `Cout=8, Wout=8`）で `bias` を `[1, Cout, 1, 1]` へ明示
  reshape せず `[Cout]` のまま `out`（`[N, Cout, Hout, Wout]`）へ加算
  すると誤った軸（`Wout`）へブロードキャストされることを検出する
  回帰テスト（正しい実装は `Cout` 軸〈axis 1〉へ加算し `Wout` 軸には
  影響しないことを直接検証）。

## 14. 出典

- 親 #1606・本 #1641・兄弟 #1642〜#1645・関連 #1727（pooling）・
  #1715（`gemm_batched`）・#1213／#1214／#1215（VJP 転置入口）・
  #1566／#1659（bias 縮約 `f64`）・#1756（pad）・#1616（state_dict）。
- `docs/compat-api-scope.md` §1.2・`docs/compat-feature-gap.md` §2.7・
  `docs/spec/04-requirements.md:231`（REQ-9 2026-09-12 追記）。
- `.claude/rules/coding-rust.md`（FMA 契約・長軸縮約 `f64`・REQ-8）。
- `docs/pooling-ops-design.md`（#1727）: レイアウト・引数命名の共有
  規約・負分子拒否ゲートの先例。
- `docs/metal-grad-reduction-parity-judgment-decision.md`: Metal
  bias 縮約の `f64` 相当実装形（binary64 逐次加算エミュレーション）
  の到達点。
- PyTorch `aten/src/ATen/native/ConvUtils.h::_conv_output_size`・
  `Convolution.cpp::check_shape_forward`（2026-09-14 に
  `pytorch/pytorch` main を確認）。
- `crates/backend-cpu/src/gemm_blis/mod.rs:28-36,651,670`・
  `crates/autodiff/src/eval.rs:11-12,300-322`・
  `crates/tensor-core/src/backend_ops.rs:200,684,720,733`・
  `crates/tensor-core/src/ops_shape.rs:97,220`・
  `crates/autodiff/src/grad.rs:235-236,312-316,702,1239-1250,2477-
  2525,2535,2634-2645`・
  `crates/autodiff/src/var.rs:2211`・
  `crates/autodiff/src/nn/module.rs:30-45`・
  `crates/autodiff/src/nn/linear.rs:24-72,131-138`・
  `crates/tensor-core/src/tensor.rs:425,475,505,653`。

## 15. 実装記録（記入欄）

本 issue（#1641）はコード変更を含まない設計 doc のみ。実装記録は
#1642（CPU）・#1643（CUDA）・#1644（Metal）・#1645（`nn` 層・parity・
実機実測）の各 issue で本節へ追記する（または各 issue 側の doc へ記録し
本節から forward pointer を張る）。
