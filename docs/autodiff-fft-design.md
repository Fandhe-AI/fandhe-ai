# FFT（rfft・irfft・fft・ifft）の設計判断記録

イシュー #2151（親 #2131）。基準コミット `b8749d1e`（main HEAD）。
`docs/autodiff-linalg-design.md`・`docs/autodiff-einsum-batch-decision.md`
と同じ「設計判断記録 → 承認 → 実装」の第 1 段に対応する記録であり、
**本 issue はコード変更を伴わない**（`crates/**`・`Cargo.*`・
`docs/spec/`・tolerance／baseline は不変）。以下の `file_path:line`
参照はいずれも基準コミット時点の値である。

## §0 結論

PyTorch の `torch.fft.{fft, ifft, rfft, irfft}` に当たる 4 演算を、
実部・虚部を末尾次元 2 の実テンソル対で表す方式（`torch.view_as_real`
相当。§3）で設計する。VJP は複素線形演算の随伴として解析形で導出する
（§5）。**complex dtype（`Tensor<complex64>` 等）を対象とする方針は
非目標のまま変えない**（`docs/tensor-core-sparse-complex-decision.md`
§6 の案 A・非対応を覆さない）。

推奨実装方式は案 B（専用 `Op` とホスト参照実装。§4）で、CPU 参照実装を
先に作り CUDA／Metal は既定 `Unsupported` からホスト計算へフォール
バックする（`.claude/rules/coding-rust.md` の linalg §3.4 と同型）。

**本 issue（docs のみ）の承認事項は「なし」**である。ただし後続の
実装 issue に着手する前提条件として、`docs/compat-api-scope.md` §5 の
範囲拡張手続きが要る（§2・§10）。

## §1 背景

親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」）は「設計判断記録
→ 承認 → 実装」の 2 段階を方針とし、本 issue はその第 1 段として FFT
の設計だけを先行させる分割候補である。

対応する PyTorch API は `torch.fft.fft`／`torch.fft.ifft`／
`torch.fft.rfft`／`torch.fft.irfft`（複素対は `torch.view_as_real`／
`torch.view_as_complex` で実テンソルと相互変換する）。TensorFlow 側は
`tf.signal.fft`／`tf.signal.rfft` が対応する。

## §2 スコープ分類（中核判断）

FFT は `docs/spec/04-requirements.md` REQ-9 2026-09-12 追記
（`04-requirements.md:231-233`）の Tier 1（`04-requirements.md:231`）・
Tier 2（`04-requirements.md:232`）のいずれの列挙にも現れない。
`docs/compat-api-scope.md` §1.2（Tier 1・L204-256）・§1.3（Tier 2・
L257-283）の表にも FFT の行はない。イシュー #2151 が参照として挙げて
いる「§1.2（Tier 1）」は、実際には該当行を持たない（着手前に本文を
grep して確認済み）。

`docs/compat-feature-gap.md:326` は `torch.fft` を `complex64`／
`complex128` の行にまとめ「対象外」と判定している。REQ-9 の「引き続き
対象外」列挙（`04-requirements.md:233`）は sparse／complex テンソルを
挙げるが、FFT そのものを名指しでは挙げていない。

採用する整理は次のとおりである。

- **実部・虚部対表現（`[..., n, 2]` の `f32` テンソル）上の FFT** と
  **complex dtype（`Tensor<complex64>` 等）上の FFT** は別の事柄として
  扱う。前者は「実テンソル上の演算」であり、`Scalar` 実数 4 型封印
  （`docs/tensor-core-sparse-complex-decision.md` §5・§6）の対象外
  である。したがって complex dtype の非目標は不変で、同 doc §6 の
  結論（案 A・非対応）を再開しない。
- `compat-feature-gap.md:326`（`torch.fft` を complex 行へまとめた
  「対象外」判定）との関係: complex dtype 経由の FFT は引き続き対象外
  のままとする。実部・虚部対表現の FFT は、下記の §5 手続きを経た
  ときに限り、`compat-feature-gap.md` へ別行として対象範囲を追記
  できる（本 issue では追記しない。判定・難度列は変更しない）。
- ただし FFT（実部・虚部対表現であっても）は Tier 1／Tier 2 いずれ
  にも未列挙の機能である。`docs/compat-api-scope.md` §5 は「Tier 1／
  Tier 2 に列挙済みの機能の実装は本節の再適用を要しない」「1 節の
  Tier 列挙にも 2 節の『引き続き対象外』にも含まれない機能の追加は、
  従来どおり本節の手続きを要する」と定める（`compat-api-scope.md:
  514-516`）。FFT はどちらの箱にも入らない未列挙機能に当たるため、
  **後続の実装 issue に着手する前提条件として §5 の手続き（経路 1:
  正本 spec〈`Fandhe-AI/fandhe-ai-spec`〉での REQ-9 改定、または
  経路 2: 本リポジトリのユーザー承認を得たうえでの Issue 起票・
  `compat-api-scope.md` の更新）が要る**。本 issue（docs のみ）自体は
  §1 の対象範囲表を書き換えないため、§5 の再適用対象ではない。
- open の #2194（「forward-mode AD・vmap・functorch・sparse・complex・
  HTTP 非目標明記提案」）は spec への非目標明記を起草中である。本 doc
  は complex dtype 非目標を支持する側に立ち、#2194 の趣旨と矛盾しない。

## §3 テンソル表現と API 形状（PyTorch の意味論に合わせる）

複素入出力は `[..., n, 2]`（末尾が `(re, im)`。`torch.view_as_real` と
同じレイアウトで、行優先の連続配置）とする。末尾 2 次元は変換軸には
できない。

- **`fft(x, n, dim, norm)` / `ifft(x, n, dim, norm)`**: 入力
  `[..., L, 2]` → 出力 `[..., n, 2]`。`dim` は変換軸（複素次元を除いた
  実軸のインデックス）で、既定は複素次元の 1 つ前（末尾から 2 番目）
  の軸。
- **`rfft(x, n, dim, norm)`**: 実入力 `[..., L]` → `[..., n/2+1, 2]`。
- **`irfft(x, n, dim, norm)`**: 入力 `[..., m, 2]` → 実出力
  `[..., n]`。`n` の既定は `2*(m-1)`。入力が Hermitian 対称なスペクトル
  の前半分であることを仮定し、DC（bin 0）と（`n` が偶数のときの）
  Nyquist 成分（bin `n/2`）の虚部は無視する（PyTorch の C2R と同じ）。
  `m` が `n/2+1` と過不足する場合は切り詰め・ゼロ詰めで扱う。
- **`n`**: 変換前に入力を末尾変換軸方向へ切り詰めるかゼロ詰めする。
  省略時は変換軸の入力長。
- **`norm`**: `backward`（既定。順変換はスケールなし・逆変換は
  `1/n`）・`ortho`（両方向とも `1/√n`）・`forward`（順変換 `1/n`・
  逆変換はスケールなし）の 3 種。Rust 側は `#[non_exhaustive] enum
  FftNorm { Backward, Ortho, Forward }` を想定する。
- **実入力の complex 昇格は対象外**: PyTorch は `torch.fft.fft` に実
  テンソルを渡すと自動的に虚部 0 の複素テンソルへ昇格するが、本設計の
  `fft`／`ifft` は `[..., L, 2]` 形状の入力のみを受理する。実入力
  `[..., L]` から `[..., L, 2]`（虚部 0）への変換はいずれかの
  ラッパー・呼び出し側の責務とし、本 doc の対象外とする（§9）。
- 単一軸の変換に限る。`fft2`／`fftn`／`hfft`／`ihfft`／`fftshift`／
  `fftfreq` はスコープ外（§9）。

## §4 実装方式の比較と推奨

- **案 A（既存演算の合成）**: cos・sin の DFT 行列を前計算し、既存の
  `Var::matmul`／`narrow`／`cat` 等で合成する（`matrix_ops`・
  `activation_ops` と同じ型）。新規 `Op`・VJP・`BackendOps` は不要で
  GPU には gemm 経由で届く。ただし O(n²) の時間・メモリを要し、
  twiddle 因子が `f32` になる（matmul の入力精度に律速される）ため
  丸めは matmul の FMA 契約・CUDA TF32 opt-in の影響を受ける。
- **案 B（専用 `Op` とホスト参照実装。推奨）**: FFT 用の `Op` を新設
  する（4 演算を別 variant にするか、1 variant と種別 enum にするかは
  実装 issue で決める）。`BackendOps::fft_*` は既定で `Unsupported` を
  返し、`Var` は `Unsupported` のときだけ `eval::fft` 相当のホスト
  参照実装へフォールバックする（`unify_backend_error`。
  `crates/autodiff/src/var.rs:5747`。linalg §3.4 と同型）。VJP は解析
  形で書く。
- **案 C（非対応を維持）**: 何もしない。
- **案 D（`CustomFunction` で実装）**: `Op` を増やさない代わりに、
  `BackendOps` の既定フォールバック規律・`create_graph`／
  checkpoint の分類（`crates/autodiff/src/tape.rs:1311`
  `is_checkpoint_eligible`）から外れる。

**推奨は案 B**。イシュー #2151 の契約（CPU 参照実装を先に作り、GPU は
既定 `Unsupported` からホストへフォールバックさせる）に沿うこと、内部
計算を `f64` にできること、将来 O(n log n) 化へ進む余地があることが
理由である。案 A は**テストのオラクル**（DFT 行列積との突合）として
残す。

**アルゴリズムの段階**: 第 1 段は任意の `n` に使える直接 DFT を `f64`
で蓄積する O(n²) 参照実装とし、決定的で実装が小さい点を優先する。
twiddle は長さ `n` の表 `w_t = exp(-2πi t/n)`（`t = 0..n`）を
`f64::sin_cos(2π t / n)` で 1 回だけ作る。各項では `(k·j) mod n` の
添字で表を引き、大きな角度の直接評価による丸め誤差を避ける。添字の
積は `u64` の `checked_mul` で行う。

`f64::sin_cos` は `t/n` が厳密に 1/4・1/2・3/4 turn（`t ∈ {0, n/4,
n/2, 3n/4}`、`n` がそれぞれ割り切れる場合）であっても、浮動小数点の
`2π·t/n` 経由では `sin`／`cos` が厳密な `0`／`±1` を返す保証がない
（例: `sin(π)` は `libm` 実装依存で `1e-16` 程度の非零値になりうる）。
このため twiddle 表の構築時に、整数個の 1/4 turn に一致する添字
（`t % (n/4) == 0` かつ `n % 4 == 0` の場合、および `n` が偶数で
`t == n/2` の場合）は `sin_cos` の戻り値を使わず `{0.0, 1.0, -1.0}`
の厳密値へ丸めて格納する。これにより rfft の Nyquist bin（`n` が
偶数のとき `k = n/2`）の虚部が、直接 DFT の蓄積でも構造的に厳密
`0.0` になり、PyTorch の r2c 実装（Hermitian 対称性を構造的に利用し
Nyquist・DC の虚部を計算しない）と一致する（§6）。radix-2・
Bluestein 等の O(n log n) 化は性能 issue へ送る（§9）。

**共有数式の置き場所**: `tensor-core` に FFT のホスト参照実装（例
`fandhe_ai_tensor_core::fft`）を単一情報源として置き、autodiff の
eval と backend-cpu の両方から呼ぶ（`crates/tensor-core/src/
interpolate.rs` と同じ型。座標・重みの計算をクレート間で共有し複製
による乖離を防いでいる先例）。linalg（`autodiff-linalg-design.md`
§3.4〜§3.6）は `architecture_boundaries.rs` の依存方向制約のため
eval と backend-cpu で意図的にコードを複製した先例であり、本 doc は
それとの対照として挙げる。FFT は linalg と異なり `tensor-core` 側に
共有ヘルパーを置ける見込みがあるため、複製ではなく単一情報源方式を
推奨する（最終判断は実装 issue で確定する）。

## §5 VJP 方式（誤りやすいので式を明記する）

`L` を実数値の損失とし、実部・虚部対表現の勾配 `g = ∂L/∂Re + i·
∂L/∂Im`（`torch.view_as_real` の勾配と同じ整理）を用いる。以下の
`s(norm)` は各演算が順変換（`fft`／`rfft`）に対して適用するスケール
係数を表す: `s(backward) = 1`・`s(ortho) = 1/√n`・`s(forward) = 1/n`
（§3 の順変換のスケールと同じ値）。あわせて `t(norm)` を逆変換
（`ifft`／`irfft`）に適用するスケール係数とする: `t(backward) = 1/n`・
`t(ortho) = 1/√n`・`t(forward) = 1`（§3 の逆変換のスケールと同じ値）。

- **`fft`（複素線形）の VJP**: 随伴（実内積 `Re⟨a,b⟩` に関する）は
  共役転置になる。`fft` の順変換は `y = s(norm)·F·x`（`F` は DFT 行列
  `F_{kj} = exp(-2πi kj/n)`）と書け、`F` は対称（`F^T = F`）なので
  共役転置は `F^H = conj(F)` に等しく、これは `idft`（正規化なしの
  逆 DFT。`+2πi kj/n` の位相で蓄積し `1/n` を掛けない生の逆変換核）
  そのものである。スカラー倍 `s(norm)` は随伴を取っても実数のため
  そのまま係数に残る。すなわち随伴は「スケールなしの逆 DFT に、
  順変換で適用したのと同じスケール係数 `s(norm)` を掛けたもの」に
  等しく、`n` 倍は掛けない: `backward` なら
  `dx = s(backward)·idft(g) = idft(g)`、`ortho` なら
  `dx = s(ortho)·idft(g) = idft(g)/√n`、`forward` なら
  `dx = s(forward)·idft(g) = idft(g)/n`。
- **`ifft` の VJP**: `fft` と対になる形で、順変換の逆に当たる演算の
  随伴を取る。`backward` なら `dx = dft(g)/n`（正規化なし順 DFT を
  `n` で割る）、`ortho` なら `dx = dft(g)/√n`、`forward` なら
  `dx = dft(g)`（正規化なし順 DFT そのもの）。
- **`rfft` の VJP**: `g`（長さ `n/2+1`）を全スペクトル長 `n` まで
  ゼロ詰めし、c2c 随伴（スケールなしの逆 DFT に `s(norm)` を掛けた
  もの。上記 `fft` の随伴と同じ核。`n` 倍は掛けない）を適用して
  **実部を取る**。これは PyTorch の `fft_r2c_backward` と同じで、
  単純な `irfft` の VJP（下記）とは**異なる**。
- **`irfft` の VJP**: `irfft` 自身の順変換は `x = t(norm)·
  Re(idft_full(X))`（Hermitian 拡張した生の逆 DFT に `irfft` 自身の
  逆変換スケール `t(norm)` を掛ける演算）なので、その随伴は「`fft`
  の随伴と対称に、順変換用ではなく `irfft` 自身の逆変換スケール
  `t(norm)` を掛けたもの」になる（`fft` の随伴に `s(norm)` を掛けた
  のと同じ理屈で、ここでは `s(norm)` ではなく `t(norm)` を使う）。
  ここで言う `dft_half(g)`（実数 `g`、長さ `n` の実出力に対する勾配）
  は、**正規化を一切適用しない生の順 DFT**（bin `k=0..n/2` の前半
  のみを計算する r2c 変換。§3 で定義した正規化済みの `rfft` 演算
  そのものではない点に注意する。`rfft(g)` は `s(norm)` による正規化
  を内部で適用済みのため、ここで `t(norm)` をさらに掛けると
  `ortho`・`forward` では二重にスケール
  され、勾配が期待値の `s(norm)` 倍〈`ortho` なら `1/√n`
  倍・`forward` なら `1/n` 倍。`backward` は `s(backward) = 1`
  のため二重適用の影響を受けない〉になり数値契約を破る）に係数
  `c_k`（`k` は bin 添字）を掛け、`t(norm)`
  を掛ける: `dX = t(norm)·(c_k · dft_half(g))`。`t(norm)` は
  `backward` なら `t(backward) = 1/n`、`ortho` なら
  `t(ortho) = 1/√n`、`forward` なら `t(forward) = 1`（スケールなし）。
  `c_k` は「DC（`k=0`）と、`n` が偶数のときの Nyquist（`k=n/2`）では
  `1`、それ以外の内部 bin（`n` が奇数なら `k=1..(n-1)/2`、`n` が
  偶数なら `k=1..n/2-1`）では `2`」とする（PyTorch の
  `fft_c2r_backward` と同じ倍加規則）。DC と（`n` が偶数のときの）
  Nyquist の虚部方向の勾配（`Im(dX)` の該当成分）は `0` になる
  （§4 の厳密ゼロ twiddle により `dft_half(g)` 自体の該当成分が
  構造的に `0.0` であるため）。`m > n/2+1` の余剰 bin（VJP の出力側）
  の勾配はゼロとし、`m < n/2+1` の場合は切り詰める。
- `n` による切り詰め・ゼロ詰めの VJP は、逆向きのゼロ詰め・切り詰め
  になる（切り詰められた成分の勾配は破棄、ゼロ詰めされた成分に対応
  する入力位置は勾配をそのまま受け取る）。
- **高階微分（`create_graph`）**: 新しい `Op` は当初「非対象」
  （`crates/autodiff/src/create_graph.rs:383` `validate_ancestors`
  が型付きエラーで拒否する）に分類する案を採る。VJP は FFT 自身の
  線形結合で表せるため、将来は対象にできる見込みがある。
  `is_checkpoint_eligible`（`crates/autodiff/src/tape.rs:1311`）も
  当初は `false` とする。

## §6 数値契約

内部は `f64` で逐次・固定順序で計算し、出力時に 1 回だけ `f32` へ
downcast する（linalg §3.5 の先例）。同じ入力に対して run-to-run で
`bit` 決定的とする。

`f64` の `sin_cos` はプラットフォームの `libm` に依存しうる。この
ため、クレート間で `bit` 同一であることは受け入れ条件にせず、REQ-2
の統一複合判定（相対誤差 `1e-3` 未満 または 絶対誤差 `1e-5` 未満）で
判定する。ただし §4 の厳密ゼロ twiddle（1/4 turn での丸め）は `libm`
依存の丸め誤差とは別の構造的措置であり、Nyquist・DC の虚部を厳密
`0.0` にする目的に限って適用する（この措置自体は REQ-2 判定を緩める
ものではない）。

matmul 系の FMA 契約は変更しない。FFT 内の複素乗算に `f64::mul_add`
を使うかどうかは実装 issue で決めるが、方針としては「`f64` 内部精度
のため FMA の有無は REQ-2 判定の範囲内」と扱う。

将来の GPU バタフライカーネル（radix-2 等）は、直接 DFT と加算の
結合順序が異なる。そのため `.claude/rules/coding-rust.md` の「結合
順序が単一の連続 K ループと異なるカーネル」に当たり、baseline 非
後退方式を採るかどうかは GPU 専用カーネルの issue で決める。tolerance
定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）は本 doc では
変更しない。

非有限入力（`NaN`／`inf`）は事前に拒否せず、そのまま伝播させる
方針とする。

## §7 境界検査・エラー方針（REQ-8／OWASP A03）

- 複素入力の末尾次元が `2` であること、`n ≥ 1`、`dim` が範囲内
  （末尾の複素次元を除く）であること、rank ≥ 1（複素入力は
  rank ≥ 2）であることを検査する。
- 出力形状・要素数の算術は `checked_mul`／`checked_add` で行う。
  巨大な `n` は確保前に fail-closed で拒否する（open の #2264
  「`pad` の出力確保を巨大な pad 幅で fail-closed にする」と同じ
  考え方）。
- 入力の検証はすべて型付きエラー（`AutodiffError::InvalidArgument`）
  で返す。本番経路で `unwrap`／`expect` を使わず、panic しない。
  `Unsupported` は「バックエンドが未実装」の意味に限り、フォール
  バック条件から `InvalidArgument` を外す（A08: 判定の迂回経路を
  作らない）。

## §8 後続実装 issue の構成案（参考。起票はしない）

- `tensor-core`（ホスト参照実装・`BackendOps` の既定 `Unsupported`
  メソッド・`FftNorm`）、`autodiff`（`Op`・eval の呼び出し・
  `grad.rs` の VJP・内部の自由関数モジュール `fft_ops`）、
  `backend-cpu`（参照実装を呼ぶオーバーライド）、`backend-cuda`／
  `backend-metal`（明示的に `Unsupported` のまま）に分けて書く。
- テスト方針: 案 A（DFT 行列積オラクル）との REQ-2 突合、有限差分に
  よる勾配検査、解析解（デルタ関数 → 定数、`cos` → スペクトル線）、
  `fft→ifft`／`rfft→irfft` の往復、`norm` 3 種の組み合わせ、境界
  エラー、run-to-run の `bit` 決定性を含める。CUDA／Metal の実機
  parity は `#[ignore]` で分け、測っていない場合は
  `docs/perf/logs/<slug>-<issue>/` へ申し送る。
- facade 公開は einsum-batch §3 の保留実装パターンに従う。`Var`
  メソッドや facade の `pub use` は承認されるまで追加せず、内部
  クレートの自由関数だけにとどめる。

## §9 スコープ外（out-of-scope-tracking 対象として列挙のみ）

- FFT の実装そのもの（本 issue は docs のみ）
- complex dtype・complex 演算全般
- `fft2`／`fftn`／`hfft`／`ihfft`／`fftshift`／`fftfreq`
- GPU 専用 FFT カーネル（radix-2・Bluestein 等の O(n log n) 化を含む）
- ONNX `DFT` op の import／export
- STFT 等の信号処理 API
- 実入力から `[..., L, 2]`（虚部 0）への自動昇格ヘルパー（§3）

起票は承認後に行う（`.claude/rules/out-of-scope-tracking.md`）。

## §10 承認事項（すべて未承認として列挙する。本 issue では実施しない）

1. `docs/compat-api-scope.md` §5 の範囲拡張（経路 1: 正本 spec
   （`Fandhe-AI/fandhe-ai-spec`）での REQ-9 改定提案の投稿、または
   経路 2: 本リポジトリのユーザー承認と実装 issue の起票・
   `compat-api-scope.md` の更新）。
2. `BackendOps` trait の拡張（既定 `Unsupported` のメソッド追加）と
   `Op` variant の新設。
3. facade 公開面の拡張（`Var::fft` 等のメソッドや
   `fandhe_ai::FftNorm` の再エクスポート）。
4. 将来の GPU カーネルで baseline 方式を採る場合の baseline 値の
   追加（実機実測値のみ・人間承認必須）。

依存の追加・新規 `unsafe` は不要（自作の DFT で足りる見込み）。

## §11 実機実測の申し送り

docs のみの変更のため対象外。

## §12 出典一覧

| 出典 | 内容 |
|---|---|
| `docs/spec/04-requirements.md:231-233` | REQ-9 2026-09-12 追記（Tier 1・Tier 2・引き続き対象外の列挙。FFT は未列挙） |
| `docs/compat-api-scope.md` §1.2（L204-256）・§1.3（L257-283） | Tier 1／Tier 2 の実装リポ側転記（FFT の行なし） |
| `docs/compat-api-scope.md` §5（L497-516 ほか） | 範囲拡張の手続き（経路 1／経路 2）・Tier 列挙済み機能の再適用不要規定 |
| `docs/compat-feature-gap.md:326` | `torch.fft` を complex64／complex128 の行にまとめ「対象外」と判定 |
| `docs/tensor-core-sparse-complex-decision.md` §5・§6 | complex 非対応の案 A（非対応）確定・`Scalar` 実数 4 型封印・ONNX COMPLEX 拒否の根拠 |
| `docs/autodiff-linalg-design.md` §3.4〜§3.6 | `BackendOps` 既定 `Unsupported`・`f64` 内部計算・`Var` の二段フォールバックの先例 |
| `docs/autodiff-einsum-batch-decision.md` §3 | facade 未公開のまま内部自由関数限定で実装する保留パターンの先例 |
| `crates/tensor-core/src/interpolate.rs` | eval・backend-cpu・Metal モデル間で座標・重み計算を共有する単一情報源方式の先例 |
| `crates/tensor-core/src/backend_ops.rs:3444` | `linalg_inv` の既定 `Unsupported` 実装（フォールバック契約の実例） |
| `crates/autodiff/src/var.rs:5747` | `unify_backend_error`（バックエンドエラー統一のヘルパー） |
| `crates/autodiff/src/tape.rs:1311` | `is_checkpoint_eligible`（checkpoint 対象分類） |
| `crates/autodiff/src/create_graph.rs:383` | `validate_ancestors`（高階微分の対象 `Op` 分類） |
| `crates/autodiff/src/custom.rs:44` | `CustomFunction` trait（案 D の比較対象） |
| `crates/autodiff/tests/architecture_boundaries.rs` | eval／backend-cpu 間の依存方向制約（linalg の意図的複製の理由） |
| `.claude/rules/coding-rust.md` | FMA 契約統一・正規化統計の f64 アキュムレータ契約・結合順序が異なるカーネルの parity 判定方式 |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分（外部 FFT クレートは対象外） |
| #2194 | complex dtype 等の非目標を spec に明記する提案（起草中・open） |
| #2264 | `pad` の出力確保を巨大な pad 幅で fail-closed にする（境界検査の同型先例・open） |
| #2131 | Phase 5「PyTorch／TF 置き換えの API 網羅」親 issue |
