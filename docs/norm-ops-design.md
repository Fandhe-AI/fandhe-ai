# LayerNorm 新設・RMSNorm 既存カーネル接続の設計記録（イシュー #1596）

## 0. 背景

`docs/compat-feature-gap.md` §2.7（対象 HEAD `097bff19`）が特定したギャップ:

- RMSNorm の行カーネルは CPU／CUDA／Metal に既存
  （`backend-{cpu,cuda,metal}::rmsnorm`）だが、`BackendOps::run_fused` の
  canonical プラン一致経路（`x * rsqrt(sum(x^2))`。`mean` 化・`eps`・`weight`
  なし）からしか到達できず、`BackendOps`／`Var`／`nn` に接続されていない
- LayerNorm は `onnx-interop`（`crates/onnx-interop/src/ops/layer_norm.rs`。
  推論専用・autograd 未接続・非公開）にホスト実装があるのみで、`BackendOps`・
  カーネル・VJP は存在しない

本イシューは Transformer 到達に必要な Tier 1 部品として、両者を
`BackendOps`／`Var`／`nn` へ接続する（`docs/compat-api-scope.md` §1.2）。

## 1. 最終軸限定契約

正規化軸は常に最終軸のみ（`fandhe_ai_tensor_core::row_norm_layout(shape)` が
`(rows, hidden)` を導出する。`hidden = shape[rank-1]`・`rows` は残りの先頭
次元群の要素数積）。PyTorch の多次元 `normalized_shape`（複数の末尾軸を
まとめて正規化）は対象外——利用者は `reshape` で最終軸へ畳める。

`row_norm_layout` は同種の `row_softmax_layout`（イシュー #1594）とは独立
関数とする。softmax は非最終軸を「ホストフォールバックの合図」として
`Ok(None)` で区別する契約だが、LayerNorm／RMSNorm はそもそも最終軸限定の
演算であり非最終軸という概念を持たないため `Option` を返す必要がない。

`rows` は `numel / hidden` の除算ではなく `shape[..rank-1]` の要素数積として
直接計算する（`row_softmax_layout` の `checked_div` によるゼロ除算吸収より
単純）。`rank == 0`（スカラー）は `ShapeError::RankMismatch { expected: 1,
actual: 0 }` で拒否する。

## 2. `eps` 検査

`Var::rms_norm`／`layer_norm` は forward の入口で `eps.is_finite() && eps >=
0.0` を検査し、違反時は `AutodiffError::InvalidArgument` を返す
（`Linear::new` の `in_features == 0` 拒否と同じ「構築不可能な引数を計算前に
弾く」規律）。`nn::RmsNorm::new`／`LayerNorm::new` 等のコンストラクタも同じ
検査を行い、誤った `eps` を持つ層が forward 実行まで検出されずに残ることを
防ぐ。

各バックエンドのカーネル起動前検証（`validate_rmsnorm_launch`／
`validate_layer_norm_launch`）でも同じ `eps` 検査を独立に行う（多層防御。
OWASP A03・`.claude/rules/security.md`）。

## 3. 0 サイズ契約

- `hidden == 0`: 既存 `run_rmsnorm_f32` と同じく空出力を返す
  （`rows` は `row_norm_layout` が先頭次元群の積として計算するためゼロ除算を
  経由しない）
- `rows == 0`: 空出力を返す
- rank 0: `row_norm_layout` が `ShapeError::RankMismatch` で拒否する
- **`hidden` 上限（Metal LayerNorm 限定。PR #1671 codex-review 指摘）**:
  `hidden > 2^24`（`(float)hidden` が丸め無しで表現できる上限）は
  `layer_norm.rs::validate_hidden_exact_f32` が起動前に fail-closed で
  拒否する。Metal カーネルは平均計算で `hidden` を `(float)hidden` へ
  直接変換し厳密除算するため、この上限を超えると最近接偶数丸めにより
  真の除数とのずれが `mean_lo` へ残存し出力へ伝播しうる（対応する実装
  〈整数の正確な値を保持した除算〉は行わず、この軸長を明示的に拒否
  する方針。`shaders/layer_norm.metal` 冒頭コメント参照）。CPU／CUDA
  は `f64`／`double` アキュムレータのためこの制約を持たない。

## 4. VJP（数式）

RMSNorm: `r = rstd`（行内 `rsqrt(mean(x²)+eps)`）・`x̂ = x·r`・
`dx̂ = dy·w`（`w` なしは `dy`）・
`dx = r·(dx̂ − x̂·mean(dx̂·x̂))`・`dw = Σ_rows dy·x̂`

LayerNorm: `x̂ = (x−μ)·r`（`μ` = 行内平均・`r` = 行内 `rsqrt(var+eps)`。分散は
biased ÷N）・`dx̂ = dy·w`・
`dx = r·(dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂))`・`dw = Σ_rows dy·x̂`・`db = Σ_rows dy`

forward 記録値 `out_value` だけでは（`weight` に 0 要素があると）正規化前の
`x` を逆算できないため、VJP（`crates/autodiff/src/grad.rs::rmsnorm_vjp_rows`／
`layer_norm_vjp_rows`）は `input`（と `weight` があれば `weight`）を
`materialize_fallible` で実体化し直し、行内統計（`rstd`／`mean`）を
`eval::row_rms_stats`／`row_ln_stats` で再計算する。

### 縮約精度契約

- 行内 `mean(dx̂·x̂)`（RMSNorm）・`mean(dx̂)`／`mean(dx̂·x̂)`（LayerNorm）の
  要素積は `softmax_vjp_along`（イシュー #1594 の先例）と同じ overflow 回避
  方針を踏襲する: 要素積を `f32` で確定してから `f64` へ昇格して蓄積し、
  `rstd` との最終乗算・`dy` からの減算は `f64` のまま保持して 1 回だけ
  `f32` へ downcast する
- `dw`／`db` の行方向（`rows` 軸）蓄積は `.claude/rules/coding-rust.md`
  「勾配の長軸縮約の要素積は `f32` で確定してから `f64` へ昇格して蓄積する」
  契約に厳密に従う（コメントが挙げる代表例そのもの）

数値微分テスト（`crates/autodiff/src/grad.rs::tests::rmsnorm_grad_matches_numeric_*`／
`layer_norm_grad_matches_numeric_*`）は中央差分（`H = 1e-3`）で `dx`／`dw`／`db`
を独立に検証する。

## 5. バックエンド別実装形

### CPU（`crates/backend-cpu/src/layer_norm.rs`）

`rmsnorm.rs::rmsnorm_row_scalar` と同じ縮約方式（平均・分散とも要素を `f64`
へ昇格してから計算し、`rstd` へ代入する 1 回だけ `f32` へ downcast）を
スカラー経路のみで実装する。NEON ベクトル化（`rmsnorm.rs` の `float64x2_t`
二乗和 SIMD 化と同型の最適化）は本イシューのスコープ外とし、後続の性能課題
として記録する（§9「対象外事項」参照）。`rayon` 行方向並列は既存 RMSNorm と
同じ `PARALLEL_THRESHOLD` 閾値を再利用する。

### CUDA（`crates/backend-cuda/src/kernels_layer_norm.rs`・`layer_norm.rs`）

既存 `kernels_rmsnorm.rs` の persistent block・SMEM 常駐・`float4`
ベクトル化・occupancy 予算に基づく grid 導出はいずれも採用しない。**1 CTA =
1 warp（32 レーン）が 1 行を担当し、`grid_dim = rows`（persistent block では
ない単純な 1 対 1 マッピング）**、device メモリを 2 回読む「二パス」構成
（平均 → 分散 → 書き出しの 3 段走査。`__shared__` メモリを一切使わない）と
する。理由:

- LayerNorm は平均・分散という 2 回の縮約を要し、RMSNorm の「1 縮約 + SMEM
  常駐再利用」ほど単純な 1 パス化ができない
- 受け入れ条件は「既存カーネルの接続パターンに倣った正しい新設」であり、
  persistent grid・occupancy 予算に基づくブロック数最適化は後続の性能課題
  とする

平均・分散とも `double` アキュムレータで蓄積する（CUDA は `double` 型を
ネイティブに持つため `__shfl_xor_sync` が直接対応する 5 段 warp butterfly
reduction で縮約する。`kernels_rmsnorm.rs` と同じ手法）。分散は「二パス」
（`Σ(x−μ)²/N`。`E[x²]−μ²` は使わない）。

`w`／`b` が `None` の場合のダミーバッファは `hidden` 要素のゼロ初期化
バッファを渡す（実装は `crates/backend-cuda/src/layer_norm.rs` の
`alloc_zeros::<f32>(hidden)`。`has_weight`／`has_bias == 0` により論理的には
デリファレンスされない契約だが、`hidden` 要素すべてに対する
`(has_weight != 0) ? w[i] : 1.0f` という warp 一様の三項式を nvcc が
predicated load（`i` の全域で `w[i]` の読み出し自体は無条件発行し、
書き込みのみ述語化する）へコンパイルしうるため、1 要素のダミーでは
`hidden > 1` のとき境界外読み出しになりうる〈Cursor Bugbot 指摘〉。
Metal 側 `layer_norm.rs`〈`MetalBuffer::alloc_zeroed_pooled(ctx, hidden)`。
後述〉と同じ `hidden` 要素契約へ揃えてある。既存 `rmsnorm.rs::
run_rmsnorm_f32_inner` の 1 要素ダミーはこの指摘の対象外〈スコープ外〉
として別途記録されている）。

### Metal（`crates/backend-metal/src/shaders/layer_norm.metal`・`layer_norm.rs`）

1 threadgroup = 1 simdgroup（32 スレッド）固定・persistent threadgroup 方式
（`for (row = tg_id; row < rows; row += grid_size)`。`grid_size` はホスト側
`row_kernel::derive_persistent_grid` が導出する既存の単一の真実源を再利用
する）・reduction は 5 段 butterfly（`simd_shuffle_xor` 幅 16/8/4/2/1）と、
`rmsnorm.metal` の構造イディオムを踏襲する。

MSL は `double` 型を持たないため、`mean`／`var`／`rstd` は「**64bit 整数
（`ulong`／`long`）による IEEE 754 binary64 のソフトウェアエミュレーション
（soft-f64）**」で計算する（`gemm.metal::bias_f64_*`〈イシュー #1566・
PR #1659〉と同じ手法を本ファイル独自に拡張。`ln_` 接頭辞を付けた独立関数
群として `layer_norm.metal` 内に実装する。Metal ソースは
`newLibraryWithSource` で個別ファイル単位にコンパイルされ翻訳単位を
共有できないため、`rmsnorm.metal` とは独立した複製になる）。

**当初は `rmsnorm.metal` と同じ Neumaier 改良版 Kahan 補償和 + scale/ssq
方式（LAPACK SLASSQ 系の overflow-safe な二乗和アルゴリズム）を「`f64`
アキュムレータ相当」の実装形として採用し、平均側は「行内 2 の冪
スケーリング（`row_scale`）+ 2 段補償和 + doubled-float」という設計を
経たが、いずれも次の 2 系統の反例で数値契約を満たせないことが codex-review
指摘（PR #1671）で判明し、最終的に本節冒頭の soft-f64 方式へ全面再設計した
（経緯の詳細・反例の具体的な入力値は `layer_norm.metal` 冒頭コメントを
正本とする）**:

1. **行スケール除算での微小値消失**: 同じ行に「巨大な値」と「小さいが
   非ゼロな寄与を持つ値」が混在する場合、小さい要素の比 `x_i/row_scale`
   が `f32` の正規化下限を割り込む subnormal になり、Apple GPU 実機の
   flush-to-zero（FTZ）で消える
2. **正規化係数の丸め誤差が affine の相殺で増幅される**: scale/ssq 状態を
   単一 `f32` へ丸めてから `sqrt`／逆数を取る 1 行があり、`weight` が
   極端に大きく `bias` がほぼ相殺する行でこの 1 ULP 未満の差が
   `weight` 倍に増幅され CPU（`f64` で保持）との差が数値契約を超えた

両方とも根本原因は「`f32` の限られた指数範囲・仮数精度に収まるよう値を
リスケールする」という当初のアプローチ自体にあり、soft-f64 化（`f64` は
`f32` の全域を正規化数として表現できる指数範囲を持つためリスケール自体が
不要になる）で構造的に解消した。

**平均・分散を「正しく丸めた除算」で求める理由（PR #1671 codex-review・
Cursor Bugbot 指摘・イシュー #1596 の追加是正）**: soft-f64 化した直後は
`mean`／`var` を `sum * ln_f64_recip_newton(hidden)`（Newton-Raphson 法に
よる近似逆数との積）で求めていたが、これは一様行（例
`x=[1e30f32;49]`。`weight`／`bias` なし）のような「割り切れる」ケースで
Newton 近似特有の 1 ULP 誤差が悪化し、本来 0 になるべき偏差
`x - mean` が約 `-1.407e14` という巨大な非ゼロ値になる反例が実測で
判明した。`mul_f64_bits` と対になる**正しく丸めた除算**
（`ln_f64_div`／ホスト側逐語モデル `soft_f64::div_f64_bits`。仮数を
`[2^52,2^53)` へ正規化し `numerator = ma << 55` を筆算除算〈2 進
shift-subtract〉で `mb` 除算してから 1 回だけ最近接偶数丸めする）へ
置き換えることで、この反例は解消する（`div_f64_bits` はホスト側の
ランダム・境界値・全指数組合せ網羅テストでネイティブ `f64` 除算と
bit 完全一致することを検証済み）。

平均・分散とも `mean = ln_f64_div(lane_sum, hidden_f64)`／
`var = ln_f64_div(lane_sq, hidden_f64)` として確定し、偏差計算まで
`f32` 単一値へ丸めない。`rstd = ln_f64_rsqrt_newton(var + eps)`
（Newton-Raphson 法。除算命令を使わず `mul`／`add` のみで構成し 4 回の
反復で `f64` の 52bit 精度へ収束）は引き続き近似で構わない（`rsqrt` は
「割り切れる」という特別な反例パターンを持たないため）。詳細な数式は
`layer_norm.metal` 冒頭コメントを正本とする。

`rmsnorm.metal` と異なり常に「3 パス」（device メモリ再読・threadgroup
memory 不使用。平均 → 分散 → 書き出し。行スケール導出〈`maxabs`〉の
パスは soft-f64 化により不要になったため持たない）とし、
`rmsnorm_f32_onepass` に相当する threadgroup memory キャッシュ経路も
持たない。ベクトル化ロード（`float4`）も本イシューでは適用しない。

`w`／`b` が `None` の場合のダミーバッファは `hidden` 要素のゼロ初期化
バッファを渡す（CUDA と異なり、Metal コンパイラが `(has_weight != 0) ?
w[idx] : 1.0f` を select（両辺を無条件に評価してから選択する命令）へ
最適化しうるため、1 要素ダミーでは `hidden > 1` のとき範囲外読み出しに
なりうる。既存 `rmsnorm.rs` の同型コメント参照）。

## 6. `BackendOps` 契約

`fandhe_ai_tensor_core::BackendOps::rmsnorm`／`layer_norm` は非破壊拡張
（デフォルトメソッド追加。`mse_loss` と同じパターン）で、既定は
`BackendError::Unsupported` を返す fail-safe とする。`Var::rms_norm`／
`layer_norm` は `Unsupported` のときのみホスト参照実装
（`eval::rmsnorm_rows`／`layer_norm_rows`）へフォールバックし、それ以外の
エラーは伝播する（判定迂回経路を作らない。`.claude/rules/security.md`
A08）。

`rmsnorm` は既存の `run_fused`（canonical 融合プラン一致経路）とは**別の
独立エントリ**として追加した（`run_fused` 自体は変更しない）。`layer_norm`
は `run_fused` に一致経路を追加しない——LayerNorm は本エントリ経由でのみ
到達する。

戻り値の shape 契約（入力 `x` と恒等）は `Var::rms_norm`／`layer_norm`・
`Module::forward_host` の両方が明示検証する（実装バグの黙認防止。
`.claude/rules/security.md` A08。ソフトマックス系の先例と同じ規律）。

## 7. facade 到達経路

`crates/facade/src/` へ新規 `pub use`／`pub fn` は追加していない。
`docs/compat-api-scope.md` §1.2 に「LayerNorm／RMSNorm／BatchNorm」が
Tier 1 として列挙済みのため §5 の範囲拡張手続きは不要（同節）。利用者は
既存の `Var` 再エクスポート経由で `x.rms_norm(weight, eps)`／
`x.layer_norm(weight, bias, eps)` を直接呼べる（イシュー #1594 softmax と
同型の方針）。`nn::RmsNorm`／`LayerNorm` を `Sequential` の層として追加する
`Sequential::add_rms_norm`／`add_layer_norm` は #1618（Keras 風 Sequential
の層追加）のスコープとし本イシューでは対象外。

## 8. 実機実測状況

- **Metal（Apple M4 Max）**: 実装完了・実機実測済み。
  `cargo test -p fandhe-ai-backend-metal --release --test layer_norm_parity
  -- --ignored --nocapture` の全 6 テスト・`rmsnorm_parity.rs` 追加分
  （`backend_ops_rmsnorm_*`）2 テスト・facade
  `norm_backend_parity.rs::metal_{rms_norm,layer_norm}_forward_matches_cpu`
  2 テストがいずれも green（形状網羅・極端な大きさ／`eps`・NaN 伝播・
  `BackendOps` 経由の bit 一致／CPU 直接突合を含む）
- **CUDA（DGX Spark GB10 等）**: 実装完了・**未実測**（本エージェント実行
  環境に CUDA 実機への到達手段がないため）。環境適応スモークテスト
  （`layer_norm_parity_smoke_env_adaptive`・
  `backend_ops_layer_norm_smoke_env_adaptive`）はこのマシン（CUDA 非搭載）
  で `DriverUnavailable` 分岐が green であることのみ確認済み。実機必須の
  `#[ignore]` テスト（形状網羅・極端な大きさ・CPU 直接突合・
  `BackendOps` 横断突合）は記入欄を残す

## 9. 対象外事項

- **GPU backward カーネルの結線**: 既存 backward カーネルは CUDA にしか
  存在しない（`CudaRmsNorm::run_rmsnorm_f32_train`〈rstd 保存〉＋
  `run_rmsnorm_bwd_f32`。CPU／Metal は forward のみ）。VJP は autodiff
  ホスト側（`grad.rs`）で 3 バックエンド共通に実装したため、CUDA 既存
  RMSNorm backward カーネルへの接続（`Op::RmsNorm` への rstd 保存拡張が
  必要）・LayerNorm backward カーネルの新設はいずれも対象外
- **多次元 `normalized_shape`**: 最終軸限定（§1）
- **`Sequential::add_rms_norm`／`add_layer_norm`**: #1618 のスコープ
- **CPU NEON ベクトル化**（LayerNorm）: `rmsnorm_row_neon` と同型の
  `float64x2_t` 二乗和・分散計算の SIMD 化は後続の性能課題
- **CUDA persistent grid・occupancy 予算に基づく grid 最適化**: §5「CUDA」
  節参照。単純な `grid_dim = rows` マッピングのみを実装
- **CUDA 実機実測**: §8 参照
