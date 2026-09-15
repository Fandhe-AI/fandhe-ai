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
  作らせない**（負分子拒否ゲート・要素数 overflow 検査）——のみであり、
  個々の出力窓が padding のみで埋まること自体は許容する（§7 の「罠」
  節が示すとおり、padding タップは `0.0` として im2col／
  `eval::conv2d_direct` の両方で明示的に扱われるため、有効入力ゼロの
  窓でも bit 同一性は崩れない）。**訂正 2**（codex-review 指摘）: 旧稿
  はここで「`in = 0` を拒否すれば im2col／col2im の座標計算〈`h + p_h
  − kh·d_h` 等〉が `usize` 下限を割らないことを保証される」と記して
  いたが、この保証自体が**誤り**であり撤回する。`in = 0` 拒否・負分子
  拒否ゲートは出力 shape の非負性（`Hout`／`Wout` ≥ 1）しか保証せず、
  個々の `(h, kh)` 組み合わせに対する中間座標 `h + p_h − kh·d_h` の
  非負性は保証しない（反例: `H=3, kernel=3, padding=0, dilation=1` で
  `h=0, kh=1` のとき `h + p_h − kh·d_h = −1`。このとき出力 shape は
  `Hout=1 ≥ 1` で上記ゲートを通過する）。座標計算が負になること自体は
  「この `kh` は入力位置 `h` に寄与しない」ことを表す正常なケースで
  あり、実装契約は **§6.2 に記す `checked_sub` ベースの符号安全な
  スキップ**とする（`usize` の通常減算で実装してはならない）。`N = 0`
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
すると、チャンク 2 の部分和 `−2^60 + 1` は `f64` の丸め（`2^60`
近傍の表現間隔は `2^(60−52) = 2^8 = 256` のため `1` は丸め落ちす
る）により `−2^60` となり、チャンク合算は
`(2^60 + 0) + (−2^60) = 0` になる。一方逐次和は
`((2^60 + 0) + (−2^60)) + 1 = 0 + 1 = 1` となり、両者は一致しない
（部分和の加算順序が逐次順と異なるため丸めの発生タイミングがずれ、
結果が乖離する）。正しい実装契約は「**単一の `f64` アキュムレータを N 軸全体で
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
走査する。**座標の非負性は保証されない**（§3「訂正 2」。`h + p_h` は
`usize` の非負和だが、`kh·d_h` を引いた `h + p_h − kh·d_h` は `kh` が
大きいと負になりうる——これは「この `kh` は入力位置 `h` に寄与しない」
という正常なケースであり異常ではない）。実装契約（`checked_sub` に
よる符号安全なスキップ。`usize` の通常減算でアンダーフローさせては
ならない）: `(h as usize + p_h).checked_sub(kh * d_h)` を計算し、
`None`（減算が負になる＝アンダーフロー）ならこの `(kh, kw)` は寄与
なしとして**この時点で**スキップする（割り切れ判定・範囲検査より
前に行う）。`Some(numerator)` のときのみ `numerator` が `s_h` で
割り切れ `oh = numerator / s_h ∈ [0, Hout)`（`w`／`kw`／`ow` も同様に
`checked_sub` で計算）のときのみ `d_col[n, g, (c_g, kh, kw), (oh, ow)]`
を**`f64` へ昇格して逐次加算**、最後に 1 回 `f32` へ downcast。CUDA は
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
- col2im の座標アンダーフロー回帰（§6.2「訂正 2」の契約固定。
  codex-review 指摘）: `H=3, kernel=3, padding=0, dilation=1` 等
  `h + p_h − kh·d_h` が負になる `(h, kh)` 組み合わせが存在する形状で、
  `usize` の通常減算では `panic`（debug）／ラップアラウンド（release）
  するところを `checked_sub` による早期スキップが正しく寄与なしと
  扱い、対応する出力位置が存在しないことを固定する。3 バックエンドで
  同一形状の bit 一致も併せて確認する。
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

### #1764（CPU 実装）

CPU 分（本 doc の実装対象 `#1642` に相当する範囲。イシュー番号は
#1764）を実装済み:

- **`tensor-core`**: `ops_shape::{conv_out_len, conv2d_out_shape,
  im2col_out_shape}`（`crates/tensor-core/src/ops_shape.rs`）・
  `backend_ops::Conv2dParams`（コンストラクタ検査付き）・
  `BackendOps::{im2col, col2im, conv2d}`（既定 `Unsupported`。
  `crates/tensor-core/src/backend_ops.rs`）。
- **`backend-cpu`**: `im2col::{im2col, col2im}`（単一スレッド逐次参照
  実装・`checked_*` による符号安全な座標計算・`f64` アキュムレータ。
  `crates/backend-cpu/src/im2col.rs`）を `CpuBackendOps::im2col`／
  `col2im` へ override 結線（`ops.rs`）。`conv2d` 自身は override せず
  段階的合成経路（`gemm_batched`〈BLIS〉）を使う。
- **`autodiff`**: `Op::Conv2d { input, weight, bias, params }`
  （`tape.rs`。非融合・`push_eager`・`is_checkpoint_eligible = false`）・
  `eval::{im2col, col2im, conv2d, conv2d_direct}`（ホスト参照実装。
  `conv2d_direct` はテストオラクル専用）・`grad::{im2col_with_fallback,
  col2im_with_fallback, conv2d_with_fallback}`（段階的フォールバック
  ヘルパ）・`Op::Conv2d` の VJP（d_input＝col2im・d_weight＝
  `reduce_batch_axes_f64` による N 軸 `f64` 縮約・d_bias＝
  `eval::reduce_bias_grad_rows` の `f64` 行縮約）・`Var::conv2d`
  （公開シグネチャ。`crates/autodiff/src/var.rs`）。
- **N チャンク分割（設計 §10）は本実装では行わない**（forward は
  per-sample GEMM の独立性によりチャンク分割数に依らず bit 同一と
  設計されており正しさには影響しない純粋なピークメモリ削減の最適化の
  ため。大規模入力でのメモリ上限は out-of-scope-tracking.md に従い
  別 issue で追跡する）。
- **テスト**: CPU bit 一致 3 点（`conv2d_direct` ≡ `eval::conv2d` ≡
  `conv2d_with_fallback(TestOps)`。`grad.rs` 内 `#[cfg(test)]`）・
  padding タップ非スキップの NaN／inf 回帰・フォールバック階層
  （`gemm_batched` がバックエンド経由で呼ばれることのカウンタ固定）・
  d_bias の `f64` アキュムレータ固定（`[1e8, 1.0, -1e8]` 相殺列）・
  `crates/backend-cpu/src/im2col.rs` 内 `#[cfg(test)]`（im2col／col2im
  の単体テスト 7 件）・`crates/autodiff/tests/conv2d.rs`（出力 shape
  表・`N=0` 受理・数値微分突合〈input／weight／bias・groups＋
  dilation・重なり窓〉・shape／引数検査の境界・groups の per-group
  `narrow` 合成との bit 一致・bias 軸回帰）・`crates/facade/tests/
  conv2d_backend_parity.rs`（CPU〈`CpuBackendOps`〉vs `NaiveOps` の
  forward／backward bit 一致・groups／depthwise。`#[ignore]`: Metal／
  CUDA を CPU と `assert_parity`〈REQ-2 複合判定〉で比較——ホスト
  im2col → GPU GEMM 経路のため。実機未実測のまま記入欄を残す）。
- **`eval::conv2d_direct` の bias 加算順序（codex-review 相当の自己
  是正）**: 当初実装は `acc` を `bias` で初期化してから `mul_add`
  連鎖を回していたが、これは im2col＋GEMM 側（GEMM 結果へ bias を
  独立した加算パスとして後から加える）と丸め順序が食い違い 1 ULP の
  bit 不一致を生んだ。`acc = 0.0` から `mul_add` 連鎖を計算し、最後に
  1 回 bias を加算する形へ修正し bit 完全一致を確認した（設計 doc §7
  の記述どおりの実装へ是正）。
- **facade 公開面**: 新規 `pub use`／`pub fn` は追加していない
  （`Var::conv2d` は既存 `Var` 再エクスポート経由で到達。
  `Conv2dParams` は facade へ再エクスポートしない）。
- **引き継ぎ**: CUDA カーネル（#1643）・Metal カーネル（#1644）・
  `Var::conv1d`（#1765）・`nn::Conv2d`／`as_conv2d`／
  `compat::Sequential::add_conv*`／GPU 実機実測（#1645）。

### #1765（`Var::conv1d`。1d を 2d の reshape 併合として実装）

§2「1d は 2d へ併合」・§8 の確定方針どおり、`Var::conv1d` を新規
`Op`／`BackendOps` メソッド／VJP／バックエンドカーネルを一切追加せず
`Var::conv2d`（#1764）への薄いラッパーとして実装済み:

- **`autodiff`**: `Var::conv1d`（`crates/autodiff/src/var.rs`。
  `conv2d` 直後）。検査順序: ①`check_same_tape`（weight／bias）→
  ②`input`／`weight` の rank 検査（rank 3 以外は
  `ShapeError::RankMismatch`）→ ③`Conv2dParams::new([1, k], [1,
  stride], [0, padding], [1, dilation], groups)`（`H` 軸固定）→
  ④合成した 4d shape に対する `conv2d_out_shape`（チャンネル整合・
  `L = 0` 拒否・負分子拒否ゲート。純粋な shape 計算で tape 非接触）→
  ⑤bias shape 検査 → ⑥ここで初めて `input`／`weight` を `[N, Cin,
  1, L]`／`[Cout, Cin_g, 1, k]` へ reshape し `conv2d` を呼ぶ →
  ⑦出力 `[N, Cout, 1, Lout]` を `[N, Cout, Lout]` へ reshape。
  `Var::reshape` は view ノードを tape へ push するため、reshape より
  前にすべての引数検査を完了させ `Err` 経路で孤児ノードを残さない
  設計。
- **contiguity の非対称解消**: `Var::reshape` は非 contiguous 入力を
  `ShapeError::NonContiguousReshape` で拒否する契約だが、`conv2d` は
  transpose 済み入力も `materialize_fallible` 経由で受理するため 1d
  だけが拒否するのは非対称になる。既存 `pub(crate) fn
  Var::contiguous`（`crate::einsum` が同じ理由で使っている非公開
  ヘルパ。イシュー #1620）を input／weight の reshape 前段に適用して
  契約を 2d と揃えた（`contiguous` の doc comment の消費者記述を
  `crate::einsum`・`Var::conv1d` へ更新）。既に contiguous なら新規
  ノードを積まない設計のため、通常経路（transpose を伴わない標準
  `[N, C, L]` 入力）に余計なコピーは生じない。
- **数値契約**: reshape は view（zero-copy）のため、1d と手動 reshape
  した `[N, C, 1, L]` 2d の forward・`d_input`／`d_weight`／`d_bias`
  すべてが **bit 完全一致**する（設計 §12 の主要件）。tolerance・
  baseline は変更しない。
- **テスト**: `crates/autodiff/tests/conv1d.rs`（19 件）—
  conv1d↔手動 reshape conv2d の bit 完全一致（groups・dilation・
  stride 重なり窓を含む 3 形状）・整数手計算オラクル（cross-
  correlation の非対称カーネル値。padding 有無 2 件）・出力 shape 表
  （PyTorch `_conv_output_size` 相当）・`N=0` 受理・数値微分突合
  （基本・groups＋dilation・重なり窓）・shape／引数検査の境界（rank・
  `stride=0`／`dilation=0`／`groups=0`・`L=0` 拒否・チャンネル不整合・
  bias shape 不一致・負分子拒否。各 `Err` 経路で `Tape::len()` が
  呼び出し前後で不変であることを固定し孤児ノードなしを保証）・
  非 contiguous 入力（transpose view）が contiguous コピーと bit
  一致することの固定。`crates/facade/tests/conv1d_backend_parity.rs`
  （`conv2d_backend_parity.rs` と同型）— CPU（`fandhe_ai::tape()`）
  vs `NaiveOps` の forward／backward bit 一致（groups／depthwise
  含む）・`#[ignore]` Metal（`cfg(target_os = "macos")`）／CUDA の
  `assert_parity`（REQ-2 複合判定）。実機未実測のまま記入欄を残す。
- **facade 公開面**: 新規 `pub use`／`pub fn` は追加していない
  （`Var::conv1d` は既存 `Var` 再エクスポート経由で到達）。
- **引き継ぎ**: `nn::Conv1d`／`nn::Conv2d` 層・`as_conv*` フック・
  `compat::Sequential::add_conv1d`・GPU 実機実測は #1645。CUDA／
  Metal 専用 im2col／col2im カーネルは #1643／#1644（`conv1d` は
  `conv2d` に委譲するため、これらのカーネルが実装され次第 1d も
  自動的に恩恵を受ける）。1d 専用の高速経路（`kH=1` の im2col
  特殊化等）は設計時点で対象外のまま。

### #1766（CUDA `im2col`／`col2im` 実装）

- **対象**: `crates/backend-cuda` に `im2col`／`col2im`（`BackendOps`
  override）を実装した。`conv2d` 自身は §9 の方針どおり override
  しない（forward は `conv2d_with_fallback` の段階的合成——CUDA
  `im2col` → `gemm_batched` → `add`〈bias〉——、backward は既存 VJP
  〈`grad.rs`〉が `im2col_with_fallback`／`col2im_with_fallback` →
  `gemm_batched_fp32_strict` ×2 → `col2im_with_fallback` の順で CUDA
  カーネルへ自動的に到達する。`autodiff` 側のコード変更は不要）。
- **カーネル設計**（`crates/backend-cuda/src/kernels_im2col.rs`）:
  `Conv2dParams` が rank 固定 4D NCHW であることを利用し、
  `kernels_gather_scatter.rs`／`kernels_constant_pad.rs` のような
  rank 可変の shape 配列 H2D を行わず、すべての形状パラメータ
  （`n_batch, cin, groups, cin_g, h_in, w_in, h_out, w_out, kh, kw,
  sh, sw, ph, pw, dh, dw, k_g, p, numel`）をスカラー `int` カーネル
  引数として渡す。`im2col_f32` は 1 出力要素 = 1 スレッドの純粋
  コピー（算術なし・**bit 完全一致**）。`col2im_f32` は 1 入力位置
  = 1 スレッドの入力位置定常走査（atomic 不使用）で、寄与する
  `(kh, kw)` を row-major に `double` アキュムレータへ逐次加算し
  最後に 1 回 `(float)` downcast する（`.claude/rules/coding-rust.md`
  の勾配長軸縮約規約・CPU 参照実装 `backend-cpu::im2col::col2im` の
  `f64` 逐次和と**bit 完全一致**。乗算を伴わない純粋な和のため FMA
  融合の余地がなく CUDA `double` はホスト側 `f64` 加算と同一の丸め
  結果になる）。座標計算（`h + p_h − kh·d_h` 等）は CPU の
  `checked_sub` 相当を符号付き `long long` 演算で表現し、剰余を取る
  **前**に符号判定する（C の負数 `%` の符号曖昧性回避。設計 §6.2
  「訂正 2」と同じ理由）。
- **起動 API**（`crates/backend-cuda/src/im2col.rs::CudaIm2col`）:
  `constant_pad.rs`／`scan.rs` と同じ構成方針（NVRTC コンパイル・
  `context_cache::cached_im2col` によるプロセス内シングルトン共有・
  `with_driver_call` 経由の CUDA Graph capture 排他参加）。
  `LaunchShape::derive` が `conv_out_len` で `h_out`／`w_out` を
  独立に再計算し（`out_shape`／`d_col` の `P` 軸だけでは
  `h_out`／`w_out` 個別の値が復元できないため）、`P` 軸との整合を
  `InvalidIm2colShape` で fail-closed 検査する。全スカラー引数は
  `i32` 範囲検査（超過は `Im2colSizeLimitExceeded`）済み。
- **エラー写像**（`ops.rs::map_im2col_error`）: `map_scan_error`／
  `map_unique_error` と同じ設計判断——
  `Im2colSizeLimitExceeded`（形状パラメータがカーネル引数 `int`
  上限を超過。col は入力の `kH·kW` 倍で現実的形状でも上限へ到達
  しうるため hard fail ではなくフォールバックが妥当）**のみ**
  `BackendError::Unsupported` へ写像し `im2col_with_fallback`／
  `col2im_with_fallback` のホスト参照実装（`eval::im2col`／
  `col2im`）へ委ねる。`InvalidIm2colShape`（内部契約違反。呼び出し元
  `ops.rs` の事前検証を通過した入力からは実質到達しない防御的経路）
  は `ShapeMismatch(ElementCountOverflow)` へ、それ以外（driver 不在
  等）は `map_cuda_error` へ委譲する（判定迂回経路を作らない。
  `.claude/rules/security.md` A08）。
- **`ops.rs` override の二重検査**: `im2col`（`im2col_out_shape` で
  再検査・出力が空〈`N`／`Cin` 系の軸が 0〉なら driver 非接触で
  早期リターン・確保前検査 `checked_f32_bytes`）・`col2im`
  （`im2col_out_shape` から導出した期待 `d_col` 形状と実形状の完全
  一致検査・`input_shape`（`d_col` とは独立に指定される戻り値
  shape）自体のバイトサイズも `checked_f32_bytes` で確保前検査。
  `backend-cpu::ops::CpuBackendOps::col2im` の PR #1862 codex-review
  是正と同じ攻撃面への対処）。
- **数値契約**: im2col は 3 バックエンド bit 完全一致（算術なし）、
  col2im は CPU `f64` 逐次和と bit 完全一致（CUDA は `double`
  ネイティブ）。conv2d 全体（GEMM 段を含む forward／backward）は
  REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
  未満）。tolerance・`BASELINES` は不変。
- **テスト**: `kernels_im2col.rs` の静的テスト（REQ-8 境界検査・
  `long long` 添字・col2im の `double` アキュムレータ／単一
  downcast・im2col の算術非含有・符号判定の順序）・`im2col.rs` の
  driver 非接触単体テスト（`validate_i32_bound`／`checked_numel`／
  `LaunchShape::derive` の `P` 軸整合検査）・`ops.rs` の driver
  非接触回帰テスト（rank 不整合・`N=0` 空出力早期リターン・`d_col`
  shape 不一致）・`crates/backend-cuda/tests/im2col_col2im_parity.rs`
  （環境適応スモーク＋GB10 実機 `#[ignore]` 形状網羅）・
  `crates/facade/tests/conv2d_backend_parity.rs::
  cuda_conv2d_backward_matches_cpu`（forward に加え backward
  〈d_input／d_weight／d_bias〉を `assert_parity` で追加）。
- **facade 公開面**: 新規 `pub use`／`pub fn` は追加していない
  （`Var::conv2d` は既存 `Var` 再エクスポート経由で到達済み）。
- **GB10 実機未実測**: 本エージェント実行環境に DGX Spark GB10
  実機への到達手段がなく、`#[ignore]` テスト群（parity・backward
  `assert_parity`）は未実行のまま `docs/perf/logs/cuda-conv2d-1766/`
  へ申し送る。
- **引き継ぎ**: Metal 専用カーネルは #1644。`conv2d` の融合／direct
  override（GEMM 段迂回等の性能最適化）・GPU im2col 出力のデバイス
  常駐化（現状はホストへ readback してから `gemm_batched` へ再
  アップロードする往復コストが残る）は既存 #1643 コメントへ追記予定
  のスコープ外事項（別 issue）。

### #1767（CUDA Conv1d。1d 形状の CUDA 経路検証・GB10 実機実測スキャフォールド）

origin/main（#1766 マージ後）時点で、CUDA の Conv1d 経路は既に
構造的に成立していた: `Var::conv1d`（#1765）は `[N, Cin, L]`／
`[Cout, Cin_g, k]` を `[N, Cin, 1, L]`／`[Cout, Cin_g, 1, k]` へ
reshape してから `Var::conv2d` へ委譲する薄いラッパーで新規 `Op`／
`BackendOps`／VJP／カーネルを持たず、CUDA `BackendOps::im2col`／
`col2im`（#1766）は `Conv2dParams` の全スカラーを引数に取る形状汎用
カーネルのため `kh=1`／`sh=1`／`ph=0`／`dh=1` の 1d 形状もそのまま
処理する。本イシューはこの「特化」（1d を 2d の特殊ケースとして
自動的に扱う）を実機で確認するためのテスト・実測スキャフォールドの
追加に限定し、**新規カーネル・新規 `Op`・facade 新規公開面は追加
しない**（設計 §2「1d は 2d へ併合」・§15 #1765「1d 専用の高速経路
（`kH=1` の im2col 特殊化等）は設計時点で対象外」の方針を維持）。

- **`crates/backend-cuda/tests/im2col_col2im_parity.rs`**: `CASES`
  に 1d 形状（`in_shape: [N, C, 1, L]`・`kernel: [1, k]`）を 6 件
  追加（基本・重なり窓〈padding〉・dilation・groups／depthwise・
  groups〈2 groups, batch>1〉・`stride > kernel extent`）。環境
  適応スモーク（属性なし）は CUDA 実機あり／なし両方の分岐で 1d
  形状を通常 CI で確認する——実機ありの分岐では 1d 代表 1 件
  （`"1d basic no pad"`）を CPU と bit 同一まで `run_case` で通し、
  実機なしの分岐でも `p_1d`（`kh=1`）を用いた有効な入力で
  `im2col_out_shape` の 1d 導出がデバイス初期化前に panic せず
  正しく完了すること（CPU 側は最後まで成功・CUDA 側は
  `CudaUnavailable` のみで停止すること）を確認する（`#[ignore]`
  側の全形状網羅テストは 1d 6 件を自動的に含む）。
- **`crates/backend-cuda/src/im2col.rs`**: `LaunchShape::derive`
  の 1d 単体テスト 2 件（driver 非接触）——非自明な
  `stride`／`padding`／`dilation` を伴う 1d 形状で `h_out=1`・
  `w_out`／`P` が正しく導出されること（`launch_shape_derive_
  handles_1d_shape`）・1d でも `P` 軸不整合が拒否されること
  （`launch_shape_derive_rejects_1d_p_axis_mismatch`）。
- **`crates/backend-cuda/src/ops.rs`**: `im2col`／`col2im` の
  `N=0` 空出力早期リターンを 1d 形状で確認する driver 非接触
  テスト各 1 件（既存 2d 版と対称）。
- **`crates/facade/tests/conv1d_backend_parity.rs`**（`#[ignore]`。
  CUDA 実機必須）:
  - `conv1d_backward_on(device)`（`conv2d_backend_parity.rs::
    conv2d_backward_on` と同型）＋`cuda_conv1d_backward_matches_
    cpu`（dx／dw／db を `assert_parity`。REQ-2 複合判定）。
  - `cuda_conv1d_matches_manual_reshape_conv2d_bit_exact`: 本
    イシューの中核契約——同一 CUDA tape 上で `conv1d` と「手動
    reshape → `conv2d` → reshape」の forward・backward（dx／dw／
    db）が **bit 完全一致**すること（`crates/autodiff/tests/
    conv1d.rs::matches_manual_reshape_conv2d_bit_exact` の CUDA
    版。同一カーネル・同一形状を通るため bit 同一が構造的に成立
    する設計）。
  - `cuda_conv1d_forward_matches_cpu_groups_dilation`: groups＋
    dilation を伴う 1d 形状の forward `assert_parity`。
- **`Var::conv1d` doc comment**（`crates/autodiff/src/var.rs`）:
  「CUDA 専用 im2col／col2im カーネルは #1766／#1767 で到達済み
  （reshape 併合のため `conv2d` 側の CUDA override へそのまま
  委譲される）」へ更新（コード変更なし・doc のみ）。
- **facade 公開面**: 新規 `pub use`／`pub fn` は追加していない。
- **GB10 実機未実測**: 本エージェント実行環境に DGX Spark GB10
  実機への到達手段がなく（`docs/real-hardware-verification-env.
  local.md` 不在・`CUDA_NODE` 未設定・`~/.ssh/config` 不在。ローカル
  `nvidia-smi` は NVML driver／library 不一致で初期化不能）、
  `#[ignore]` テスト群は未実行のまま `docs/perf/logs/cuda-conv1d-
  1767/` へ実行コマンド・保存すべきログ一覧・事前登録判定規則を
  申し送る（実測の実施先は後続 #1771「CUDA／Metal 実機 parity・
  実測」）。
- **引き継ぎ**: 1d 専用高速経路（`kh=1` 特殊化カーネル・GPU im2col
  出力のデバイス常駐化。既存 #1643 コメントの性能スコープ外事項）・
  Metal 専用カーネル（#1644）・`nn::Conv1d`／`compat::Sequential::
  add_conv1d`（#1770）・3 バックエンド実機 parity 実測（#1771）は
  対象外のまま。

### #1768（Metal `im2col`／`col2im` 実装）

- **対象**: `crates/backend-metal` に `im2col`／`col2im`（`BackendOps`
  override）を実装した。`conv2d` 自身は §9 の方針どおり override
  しない（forward は `conv2d_with_fallback` の段階的合成——Metal
  `im2col` → `gemm_batched` → `add`〈bias〉——、backward は既存 VJP
  〈`grad.rs`〉が `im2col_with_fallback`／`col2im_with_fallback` →
  `gemm_batched_fp32_strict` ×2 → `col2im_with_fallback` の順で Metal
  カーネルへ自動的に到達する。`autodiff` 側のコード変更は不要）。
- **カーネル設計**（`crates/backend-metal/src/shaders/im2col.metal`）:
  `#[repr(C)]` 構造体 `Im2colDims`（19 × `uint` = 76 バイト。
  `crate::im2col_model::Im2colDims` とレイアウト一致）を 1 回の
  `setBytes_length_atIndex`（buffer index 2）でまとめて渡す
  （`gemm.metal::Dims`／`GemmStrides` と同じ「構造体丸ごと」方式。
  CUDA 版の 19 個スカラー引数を Metal では 1 個の `constant` 参照へ
  まとめた点が CUDA との差分）。`im2col_f32` は 1 出力要素 = 1
  スレッドの純粋コピー（算術なし・**bit 完全一致**）。`col2im_f32`
  は 1 入力位置 = 1 スレッドの入力位置定常走査（atomic 不使用）で、
  寄与する `(kh, kw)` を row-major に **binary64 ソフトウェア
  エミュレーションアキュムレータ**（`im2col_f64_widen`／
  `im2col_f64_add`／`im2col_f64_narrow`。MSL は `double` 型非対応の
  ため `scan.metal::scan_f64_*`〈`mul` を除く〉の逐語複製。ホスト側
  逐語モデルは `crate::soft_f64`）へ逐次加算し最後に 1 回
  `narrow` する。CPU 参照実装（`backend-cpu::im2col::col2im` の
  `f64` 逐次和）と**bit 完全一致**（NaN のみクラス一致）。座標計算
  （`h + p_h − kh·d_h` 等）は CUDA と同じ理由で `long`（符号付き
  64bit）演算を用い、剰余を取る**前**に符号判定する（設計 §6.2
  「訂正 2」）。
- **ホストモデル**（`crates/backend-metal/src/im2col_model.rs`。
  `cfg(target_os = "macos")` を付けず Linux でも単体テストが回る。
  `scan_model.rs`／`unique_model.rs` と同じ設計判断）: `Im2colDims`
  型定義・`derive_im2col_dims`（`backend-cuda::im2col::
  LaunchShape::derive` の Metal 対応版。`conv_out_len` による
  `h_out`／`w_out` 独立再計算・`P` 軸整合検査・`u32` 収容検査）・
  `im2col_model`／`col2im_soft_f64`（両カーネル本体の逐語 Rust 移植）
  を提供する。単体テストが `fandhe_ai_backend_cpu::CpuBackendOps`
  （dev-dependency）の `im2col`／`col2im` と bit 完全一致で突合する
  （形状網羅 10 件・1d 形状含む）。
- **起動 API**（`crates/backend-metal/src/im2col.rs::MetalIm2col`）:
  `scan.rs`／`constant_pad.rs` と同じ構成方針（実行時 MSL コンパイル・
  `context_cache::cached_im2col` によるプロセス内シングルトン共有・
  `ctx.dispatch_sync` による同期ディスパッチ。呼び出し元が戻り値を
  同期消費するため encode-only 版は設けない）。`run_im2col_f32`／
  `run_col2im_f32` は呼び出し元 `ops.rs` の検査結果を信頼せず、
  `derive_im2col_dims` による独立した形状再検証とホストスライス
  実長検証を行う（多層防御。`.claude/rules/security.md` A08）。
- **エラー写像**（`error.rs::MetalError::Im2colSizeLimitExceeded`／
  `InvalidIm2colShape`・`ops.rs::map_im2col_error`）: CUDA と同じ
  2 段階分離——`derive_im2col_dims` の `SizeLimitExceeded`（`u32`
  上限超過。col は入力の `kH·kW` 倍で現実的形状でも到達しうる）
  **のみ** `BackendError::Unsupported` へ写像しホストフォールバック
  （`eval::im2col`／`col2im`）へ委ねる。`InvalidShape`（内部契約
  違反。呼び出し元の事前検証を通過した入力からは実質到達しない
  防御的経路）は `ShapeMismatch(ElementCountOverflow)` へ、それ以外
  （デバイス・パイプライン起動失敗等）は `KernelLaunchFailed` へ
  写像する（判定迂回経路を作らない）。
- **`ops.rs` override の二重検査**: `im2col`（`im2col_out_shape` で
  再検査・出力が空〈`N`／`Cin` 系の軸が 0〉なら早期リターン・確保前
  検査 `checked_bytes_for::<f32>`）・`col2im`（`im2col_out_shape` から
  導出した期待 `d_col` 形状と実形状の完全一致検査・`input_shape`
  〈`d_col` とは独立に指定される戻り値 shape〉自体のバイトサイズも
  `checked_bytes_for::<f32>` で確保前検査。CUDA 版・
  `backend-cpu::ops::CpuBackendOps::col2im` の PR #1862 codex-review
  是正と同じ攻撃面への対処）。
- **数値契約**: im2col は 3 バックエンド bit 完全一致（算術なし）、
  col2im は CPU `f64` 逐次和と bit 完全一致（Metal は binary64
  ソフトウェアエミュレーション）。conv2d 全体（GEMM 段を含む
  forward／backward）は REQ-2 統一複合判定。tolerance・`BASELINES`
  は不変。
- **テスト**: `im2col_model.rs` の Linux 実行可能単体テスト（CPU
  との bit 一致・`P` 軸不整合拒否・`u32` 上限超過拒否・1d 形状導出・
  `Im2colDims` サイズ一致・寄与なし位置の `+0.0` 契約）・
  `crates/backend-metal/tests/im2col_source_evidence.rs`（Linux 実行
  可能な MSL ソース文字列証跡: `#include` 順序・両カーネル宣言・
  `Im2colDims` 19 フィールド・REQ-8 境界検査・`im2col_f32` の算術
  非含有・`col2im_f32` の binary64 エミュレーション使用・符号判定の
  順序・`long` 添字演算）・`crates/backend-metal/tests/
  im2col_col2im_parity.rs`（macOS `#[ignore]`。CUDA 版と同一形状
  網羅・256 threadgroup 境界・NaN／±inf／−0.0・非 contiguous
  input・N=0・run-to-run bit 同一）・`crates/facade/tests/
  conv2d_backend_parity.rs::metal_conv2d_backward_matches_cpu`
  （forward に加え backward〈d_input／d_weight／d_bias〉を
  `assert_parity` で追加）。
- **facade 公開面**: 新規 `pub use`／`pub fn` は追加していない
  （`Var::conv2d` は既存 `Var` 再エクスポート経由で到達済み）。
- **M4 Max 実機未実測**: 本エージェント実行環境に Apple Silicon 実機
  への到達手段がなく、`#[ignore]` テスト群（parity・backward
  `assert_parity`）は未実行のまま `docs/perf/logs/metal-conv2d-1768/`
  へ申し送る。
- **引き継ぎ**: Metal Conv1d 経路の 1d 形状専用テスト・実機検証は
  #1769。3 バックエンド実機 parity 実測は #1771。`nn::Conv2d`
  層・`compat::Sequential::add_conv2d` は #1645。`conv2d` の
  融合／direct override（GEMM 段迂回等の性能最適化）・GPU im2col
  出力のデバイス常駐化（現状はホストへ readback してから
  `gemm_batched` へ再アップロードする往復コストが残る）・GPU 側
  d_weight／d_bias 縮約カーネル（設計 §11）は既存 #1643／#1644
  コメントへ追記予定のスコープ外事項（別 issue）。

### #1769（Metal Conv1d。1d 形状の Metal 経路検証・M4 Max 実機実測スキャフォールド）

origin/main（#1768 マージ後）時点で、Metal の Conv1d 経路は既に
構造的に成立していた: `Var::conv1d`（#1765）は `[N, Cin, L]`／
`[Cout, Cin_g, k]` を `[N, Cin, 1, L]`／`[Cout, Cin_g, 1, k]` へ
reshape してから `Var::conv2d` へ委譲する薄いラッパーで新規 `Op`／
`BackendOps`／VJP／カーネルを持たず、Metal `BackendOps::im2col`／
`col2im`（#1768）は `Conv2dParams` の全スカラーを引数に取る形状汎用
カーネルのため `kh=1`／`sh=1`／`ph=0`／`dh=1` の 1d 形状もそのまま
処理する。本イシューはこの「特化」（1d を 2d の特殊ケースとして
自動的に扱う）を実機で確認するためのテスト・実測スキャフォールドの
追加に限定し、**新規カーネル・新規 `Op`・facade 新規公開面は追加
しない**（CUDA #1767 と同じ整理。設計 §2「1d は 2d へ併合」・§15
#1765「1d 専用の高速経路（`kH=1` の im2col 特殊化等）は設計時点で
対象外」の方針を維持）。

- **`crates/backend-metal/src/im2col_model.rs`**: `CASES` へ、
  `tests/im2col_col2im_parity.rs::CASES`（#1871 で先行追加済み）と
  同一の 1d 形状 5 件（基本・重なり窓〈padding〉・groups／depthwise・
  groups〈2 groups, batch>1〉・`stride > kernel extent`）を追加し、
  既存 1 件（`"1d shape (H=1, kh=1)"`）と合わせて 1d 形状 6 件が
  `im2col_and_col2im_models_match_cpu_backend_across_shapes`（Linux
  実行可能・CPU 参照実装と bit 完全一致）で網羅される（**Linux CI
  で回る唯一の 1d 実効検証**）。`derive_im2col_dims_rejects_1d_p_
  axis_mismatch`（`backend-cuda::im2col::launch_shape_derive_
  rejects_1d_p_axis_mismatch` と同型・driver 非接触）も追加。
- **`crates/backend-metal/src/ops.rs`**: `im2col`／`col2im` の
  `N=0` 空出力早期リターンを 1d 形状で確認する driver 非接触
  テスト各 1 件（`gather`／`scatter` の rank 上限テストと同じ理由で
  `context_cache::cached_context()` より前に return するため実機
  非依存）。
- **`crates/backend-metal/tests/im2col_col2im_parity.rs`**:
  `im2col_col2im_zero_batch_1d`（N=0・1d 形状。既存 2d 版
  `im2col_col2im_zero_batch` と対称）を追加。
- **`crates/facade/tests/conv1d_backend_parity.rs`**（`#[ignore]`。
  Metal 実機必須。`cfg(target_os = "macos")`）:
  - `metal_conv1d_backward_matches_cpu`（既存 `conv1d_backward_on`
    ヘルパを共用。dx／dw／db を `assert_parity`。REQ-2 複合判定）。
  - `metal_conv1d_matches_manual_reshape_conv2d_bit_exact`: 本
    イシューの中核契約——同一 Metal tape 上で `conv1d` と「手動
    reshape → `conv2d` → reshape」の forward・backward（dx／dw／
    db）が **bit 完全一致**すること（CUDA 版
    `cuda_conv1d_matches_manual_reshape_conv2d_bit_exact` と本体
    `conv1d_matches_manual_reshape_conv2d_on` を共用。`crates/
    autodiff/tests/conv1d.rs::matches_manual_reshape_conv2d_
    bit_exact` の Metal 版）。
  - `metal_conv1d_forward_matches_cpu_groups_dilation`: groups＋
    dilation を伴う 1d 形状の forward `assert_parity`（`conv1d_
    forward_groups_dilation_on` を CUDA 版と共用）。
- **`Var::conv1d` doc comment**（`crates/autodiff/src/var.rs`）:
  「Metal 経路の 1d 形状テスト（model・ops・facade bit 一致）は
  #1769 で追加済み。CUDA／Metal 実機実測は #1771 へ申し送り」へ
  更新（コード変更なし・doc のみ）。
- **facade 公開面**: 新規 `pub use`／`pub fn` は追加していない。
- **M4 Max 実機未実測**: 本エージェント実行環境（Linux コンテナ・
  x86_64）に Apple Silicon 実機への到達手段がなく、`#[ignore]`
  テスト群は未実行のまま `docs/perf/logs/metal-conv1d-1769/` へ
  実行コマンド・保存すべきログ一覧・事前登録判定規則を申し送る
  （#1768 分の Metal Conv2d 実測も未完了のため同ディレクトリが両方
  の受け皿を兼ねる。実測の実施先は後続 #1771「CUDA／Metal 実機
  parity・実測」）。
- **引き継ぎ**: 1d 専用高速経路（`kh=1` 特殊化カーネル・GPU im2col
  出力のデバイス常駐化・GPU 側 d_weight／d_bias 縮約カーネル。既存
  #1643／#1644 コメントの性能スコープ外事項）・`nn::Conv1d`／
  `compat::Sequential::add_conv1d`（#1770）・3 バックエンド実機
  parity 実測（#1771）は対象外のまま。親 #1644 は全 sub 完了で
  close する方針のため、本 PR マージ後も実測が未完である旨を close
  判断の材料としてユーザーへ委ねる。

### #1770（`nn::Conv1d`／`Conv2d` 層・`compat::Sequential` 接続）

§8「`nn` 配線案」の実装（親 #1645）。

- **実装ファイル**: `crates/autodiff/src/nn/conv.rs`（新設。`Conv2d`／
  `Conv2dVars`／`Conv1d`／`Conv1dVars`）・`crates/autodiff/src/nn/
  module.rs`（`Module` trait への `as_conv2d`／`as_conv2d_mut`／
  `as_conv1d`／`as_conv1d_mut`〈defaulted・既定 `None`〉フック・
  `impl Module for Conv2d`／`Conv1d`）・`crates/facade/src/compat/
  sequential.rs`（`add_conv2d`／`add_conv1d`・学習経路〈`bind`／
  `trainable_parameters`／`SequentialVars::forward`／`trainable_vars`／
  `trainable_grads`／`apply_parameters`〉の Conv 対応拡張・
  `contains_conv_layer` によるデバイス常駐経路 3 入口の fail-closed
  ガード）。
- **`Conv1d` の内部表現**: 当初案（内部に `Conv2d` を保持し `weight()`
  を都度 reshape して返す）は `weight()` が所有値 `Tensor<f32>` しか
  返せず、`compat::Sequential::trainable_parameters`（`Vec<&Tensor<f32>>`
  契約）と整合しないため撤回した。最終実装は `weight`（rank 3）を
  `Conv1d` 自身が直接保持し、`forward`／`forward_host` 内部でのみ
  一時的に rank 4 へ reshape する（`Var::conv1d` と同じ演算列を再現）。
- **`nn` 配線案からの変更点**: 設計時点（§8）は `compat::Sequential`
  が `linears: Vec<LinearVars>` のみを保持する旧実装を前提としていた
  が、実装着手時点では #1759（親 #1617）により `compat::Sequential` は
  汎用 `nn::Sequential`（`inner`）への薄いラッパーへ再構成済みだった。
  このため `Sequential::forward`／`predict`（推論経路）は `Module::
  forward`／`forward_host` の多態 dispatch を通じて **無変更のまま**
  Conv 層へ対応した。変更が必要だったのは学習経路（`bind`／
  `trainable_parameters`／`SequentialVars::forward`／`trainable_vars`／
  `trainable_grads`／`apply_parameters`）のみで、いずれも「`self.inner.
  layers()` を層順に走査し、層種別ごとに対応するカーソル
  （`linears`／`conv2ds`／`conv1ds`）から 1 件ずつ消費する」という
  `Linear` 単独時と同型の設計を層種別 3 つへ一般化する形で対応した。
- **デバイス常駐経路**: `forward_from_flat_leaves`／`build_device_
  chain_steps` は `as_linear()` のみを消費する走査のため、
  `trainable_parameters()` が Conv 層を含むようになった状態で
  Conv 層を含む `Sequential` に対し `init_device_param_store` を無条件
  で許すと、後段の forward で「leaves の要件超過」という迂遠な
  エラーへ到達してしまう。`contains_conv_layer()` による入口ガード
  （`init_device_param_store`／`forward_resident`／`predict_resident`
  の 3 箇所。`BackendError::Unsupported`）で明示的に fail-closed 拒否
  する設計とした（A04「安全でない設計」対策）。
- **対象外**: Flatten 層（`nn::Flatten`／`compat::Sequential::
  add_flatten`）は §5 手続き未了のため本 PR では追加しない（facade
  経由で Conv 出力を `Linear` へ接続する経路は autodiff 側テストの
  `Var::flatten` 直接呼び出しでのみ検証）。Conv 層のデバイス常駐
  経路・`gemm_bias_act` epilogue 融合（Conv→ReLU）・
  `padding_mode='same'`・ConvTranspose・pooling との組合せは対象外の
  まま。CUDA（GB10）／Metal（M4 Max）実機 parity・実測は #1771 へ
  引き継ぐ（`crates/facade/tests/nn_conv_backend_parity.rs` に
  `#[ignore]` テストのスケルトンを用意済み）。

### #1771（CUDA／Metal 実機 parity・実測。親 #1645）

#1766〜#1770 の 4 イシューに分散していた実行手順・事前登録判定規則を
`docs/perf/logs/conv-realdevice-1771/`（実機ランブック）へ統合し、
不足していた nn 層（`compat::Sequential`）の `#[ignore]` テストを
`crates/facade/tests/nn_conv_backend_parity.rs` へ追加した。

- **追加テスト**（CUDA・Metal 各 6 件。既存 forward のみだった
  `{cuda,metal}_sequential_conv2d_matches_cpu` に加え）:
  - `{cuda,metal}_sequential_conv2d_backward_matches_cpu`: nn 層の
    backward（weight／bias／入力勾配）を CPU と REQ-2 複合判定で比較
  - `{cuda,metal}_sequential_conv1d_{forward,backward}_matches_cpu`:
    `add_conv1d` 版（`conv1d_backend_parity.rs` は `Var::conv1d` 直叩き
    のみだったため nn 層経由の版を補う）
  - `{cuda,metal}_sequential_conv1d_matches_manual_reshape_conv2d_
    bit_exact`: `conv1d_backend_parity.rs::conv1d_matches_manual_
    reshape_conv2d_on` の nn 層版（「特化」契約。同一 GPU tape 上で
    `add_conv1d` モデルと `add_conv2d`〈`[1,k]`・`[0,p]`・`[1,d]`〉
    モデルへ同一重みを注入し forward・勾配とも bit 完全一致）
  - `{cuda,metal}_sequential_conv2d_sgd_steps_record_only`:
    `compat_sequential_conv.rs::train_loop_with_sgd_reduces_loss` と
    同じ形状・SGD 設定で 5 step を GPU／CPU 双方で回し、各 step の
    loss・最終パラメータを比較する（**record-only**。GEMM 由来の差が
    step をまたいで累積しうるため ADOPT／REJECT の判定対象にしない）
  - 上記いずれも `print_fold_bits`（`mse_backward_bench.rs::fold_bits`
    と同一実装）で `<test>[<label>].fold_bits=<hex>` 形式の 1 行を
    出力し、実機ランブックの run-to-run 決定性検査（2 回起動間の
    `grep -E 'bits=|fold_bits='` 出力 `diff`）が機械的に確認できる
    ようにした
- **CPU での正しさ検証**（Linux 実行可能。実機到達不能のため本 issue
  でできる限りの検証として実施）: 上記 5 個の内部ヘルパー関数
  （`sequential_conv2d_backward_on`／`sequential_conv1d_forward_on`／
  `sequential_conv1d_backward_on`／`sequential_conv1d_matches_manual_
  reshape_conv2d_on`／`sgd_steps_on`）を一時的に `Device::Cpu` で
  2 回呼び出す自己整合性テスト（コミット対象外・スクラッチ）を実行し、
  bit 完全一致（決定性）・学習ループの loss 単調減少を確認済み
  （テストコード自体は非コミット。ロジックの妥当性確認のみが目的）
- **facade 新規公開面なし**: 既存 `Var`／`compat::Sequential` 再エクス
  ポート経由のテストのみで `pub fn`／`pub use` の追加はない
- **tolerance／`BASELINES`**: 変更なし。実機実行時に REQ-2 fail が発生
  した場合は本 issue では対応せず、`docs/conv-ops-design.md` §7 の
  baseline 方式適用可否をユーザー承認へエスカレーションする
- **実機到達可否**: 本エージェント実行環境（Linux コンテナ）には
  DGX Spark GB10・Apple Silicon いずれの実機への到達手段
  （`docs/real-hardware-verification-env.local.md`・`CUDA_NODE`
  環境変数・`~/.ssh/config` のいずれも確認できず）もないため
  **未実測**。`verdict=undetermined` のまま親 #1645 を受け皿として
  GB10／Mac 実機セッションへ申し送る
