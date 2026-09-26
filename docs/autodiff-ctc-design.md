# CTC 損失 設計判断記録

イシュー #2168（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-loss-ops-decision.md`（#2166）・
`docs/autodiff-distance-poisson-loss-ops-decision.md`（#2167）と同型の
記録・同じ判断枠組みを踏襲する。

## §0 結論

PyTorch `nn.CTCLoss`（Connectionist Temporal Classification）相当の
API を、#2166・#2167 と同じく **`fandhe_ai_autodiff::loss_ops`
（facade 非公開の自由関数モジュール）** へ追加した。`Var` に inherent
の `pub fn` は追加していない。

- `ctc_loss(log_probs, targets, input_lengths, target_lengths, options,
  reduction)`: 新規 `Op::CtcLoss`（`options: &CtcLossOptions`）

`Op::CtcLoss` は `BackendOps` に対応メソッドを持たず、常にホスト参照
実装（CPU 上の `f64` アキュムレータ・対数空間 DP。`crate::eval`／
`crate::grad`）を経由し `push_eager`（実体化済み）でテープへ記録する
（#2166 の `Op::L1Loss` と同型のパターン。§2.1 参照）。CUDA／Metal
テープからは `materialize_fallible` 経由のホスト計算で到達する
（GPU 専用カーネルは未実装。§6「スコープ外」）。`nn::loss::CtcLoss`
（新規構造体）を薄いラッパーとして提供する。`nn::loss` は facade が
再エクスポートしないため、これも facade 公開面には出ない。

## §1 背景

イシュー #2168・親 #2131 のどちらにも所有者の承認コメントはない。
関連イシュー #2169（`compile()` の `Loss` enum への統合）は別 PR の
担当で、本 PR は `crates/facade/src/compat/training.rs` を変更しない。
親 #2131 が定める「設計判断記録 → 承認 → 実装」の 2 段階方針に従い、
本実装は内部クレート限定に倒す（facade 公開面の拡張は §5「承認事項」
に整理する）。

イシュー本文は対象ファイルとして `crates/autodiff/src/var.rs` を
挙げるが、**`var.rs` は変更しない**。`Var` は facade から直接
再エクスポートされるため、`Var` へ inherent メソッド `ctc_loss` を
足すとその時点で facade 公開面に出てしまい、未承認の承認事項を
実施したことになる。さらに次の既存ガードが fail-closed で落ちる:
`crates/facade/tests/api_surface.rs::
facade_does_not_reexport_or_declare_loss_ops`・同
`workspace_declares_loss_ops_fn_names_only_in_allowed_locations`。
#2166／#2167 と同様、`crates/autodiff/src/loss_ops.rs` に自由関数
`ctc_loss` として置く（`Var::ctc_loss`〈薄い委譲メソッド〉は承認後の
別作業）。

## §2 設計

### §2.1 `BackendOps` メソッドを新設しない判断

#2166・#2167 の判断をそのまま踏襲する。CTC の DP（前向き・後ろ向き
再帰）は行列積等と異なり融合カーネル化の恩恵が薄く、まずホスト参照
実装で正しさを確立する（GPU 専用カーネルは§6「スコープ外」）。

### §2.2 入力契約・検査順序

**API**:

```rust
pub fn ctc_loss<'t>(
    log_probs: &Var<'t>,          // [T, N, C]（追跡対象）
    targets: &Tensor<i32>,        // [N, S]（パディング）または [Σ target_lengths]（連結）
    input_lengths: &[usize],      // 長さ N
    target_lengths: &[usize],     // 長さ N
    options: &CtcLossOptions,
    reduction: Reduction,
) -> Result<Var<'t>, AutodiffError>
```

`CtcLossOptions` は `#[derive(Debug, Clone, Default)]`・
`#[non_exhaustive]`。フィールド非公開・builder（`blank`・
`zero_infinity`）でのみ構築する。`blank`・`zero_infinity` はいずれも
Rust の型既定値（`usize` の `0`・`bool` の `false`）が PyTorch 既定
（`blank=0`・`zero_infinity=false`）と一致するため `#[derive(Default)]`
で導出する（`CrossEntropyOptions` と同じ理由）。

`log_probs` は rank 3 のみ受け付ける。PyTorch の rank 2（unbatched
`[T, C]`）は安全側に倒して `InvalidArgument` で拒否する（§3「PyTorch
との差分」）。`targets` の形式判定: rank 2 ならパディング形式（shape
厳密に `[N, S]`）、rank 1 なら連結形式（長さ `== Σ target_lengths`。
`checked_add` で計算）。

**検査順序**（確保より前に全部終える。REQ-8・A03。本番経路で
`unwrap()`／`expect()` は使わない）:

1. `log_probs` の rank が 3・`C >= 1`
2. `checked_bytes_for::<f32>(&log_probs_shape)`（確保前上限）
3. `options.blank < C`
4. `input_lengths.len() == N` かつ `target_lengths.len() == N`
5. 各 `n` について `input_lengths[n] <= T`
6. `targets` の形式検査（パディング形式は `target_lengths[n] <= S`、
   連結形式は `checked_add` による長さ一致）
7. 使われる範囲の全 target 値が `0 <= t < C` かつ `t != blank`
8. 各 `n` について `L'_n = 2·target_lengths[n] + 1` を
   `checked_mul`／`checked_add` で計算
9. α／β バッファの確保前上限検査（`checked_bytes_for::<f64>(&[T,
   L'_max])`）
10. 実体化（`materialize_fallible`。層 1）
11. forward 計算（`eval::ctc_loss_forward`）
12. `push_eager` でノードを記録

違反時は `AutodiffError::InvalidArgument`（既存 `l1_loss`・
`cross_entropy_loss_with` と同じ使い分け）。`log_probs` の値（`NaN`・
`inf`・正値）は検査せず、そのまま伝播させる（既存損失と同じ方針）。

### §2.3 forward（数値安定性）

前向き再帰 α は対数空間・`f64` で計算する。拡張ラベル列は
`l' = [blank, l_1, blank, l_2, …, blank]`（長さ
`L' = 2·target_lengths[n] + 1`）。遷移規則は標準の CTC 再帰
（Graves 2006）に従う: `s-2` からのスキップは `l'_s != blank` かつ
`l'_s != l'_{s-2}` のときのみ許す。

対数加算ヘルパー `eval::log_add_exp_f64(a, b)`（`crate::eval`）を
新設し、次を明示的に扱う（advisor レビュー指摘。`f64::max` に
黙って丸め込ませない）:

- どちらかが `NaN` なら `NaN`
- 両方 `-inf` なら `-inf`
- どちらかが `+inf` なら `+inf`（`+inf - m` が `NaN` になるのを防ぐ。
  `log_probs` の値は検査しないため `+inf` が到達しうる）
- それ以外は `m + ln(exp(a-m) + exp(b-m))`（`m` は大きい方）

サンプル `n` の損失は
`nll_n = -logaddexp(α_{T_n-1}(L'-1), α_{T_n-1}(L'-2))`
（`L' = 1` は最後の 1 項のみ）。境界:

- `T_n = 0 ∧ tl_n = 0` → `nll_n = 0`（空系列の確率は 1）
- `T_n = 0 ∧ tl_n > 0` → `nll_n = +∞`
- 整列不能（`T_n` が短すぎる）→ 自然に `+∞` になる

`zero_infinity = true` のとき、`nll_n == +∞` のサンプルは損失 `0` と
して扱う。reduction（PyTorch 準拠）: `Mean = (1/N)·Σ_n nll_n /
max(tl_n, 1)`、`Sum = Σ_n nll_n`。蓄積は `f64`・サンプル添字昇順で
行い最後に 1 回だけ `f32` へ downcast する。`N == 0` は `Mean`／`Sum`
とも `0.0`（リポ既存の空バッチ規約。PyTorch は `NaN` を返す——§3）。
出力 shape は `[]`。

### §2.4 VJP（β の規約が最重要の設計判断）

後ろ向き再帰 β は、**現フレーム `t` の emission を含めない**規約で
計算する: `β_t(s) = log P(状態 s から frames [t+1, T_n) だけを使って
完了する確率)`。すなわち `α_t(s) + β_t(s)` は二重計上なしにそのまま
「時刻 `t` に状態 `s` を通過する全パスの確率」に一致する。

**標準 Graves 規約を採らない理由**（advisor レビュー指摘・実装計画
策定時に発見した数値上の罠）: 標準的な定式化は α・β 双方が現フレーム
の emission を含み、`lcab_{t,k} = logsumexp_{s: l'_s=k}(α_t(s) +
β_t(s))` から `lp[t,k]` を 1 回引いて二重計上を補正する
（`γ_{t,k} = exp(lcab_{t,k} - nll_n - lp[t,n,k])`）。しかし
`log_probs` の値は検証しない方針のため `lp[t,k] = -∞` が到達しうる。
このとき `α_t(s)`・`β_t(s)` は状態 `s` の実現に `lp[t,k]` を含む以上
必然的に `-∞` になり、`lcab_{t,k} = -∞`（`-∞ + -∞` は log-add-exp で
`-∞`）となって `lcab_{t,k} - lp[t,n,k] = -∞ - (-∞) = NaN` が発生する。
本実装は β を「現フレームを除く」規約に変更することでこの減算自体を
なくし、この NaN 罠を構造的に回避する。

**確立した式**: `lcab_{t,k} = logsumexp_{s: l'_s=k}(α_t(s) + β_t(s))`。
`nll_n = -log P` なので `log P = -nll_n`、
`γ_{t,k} = exp(lcab_{t,k} - log P) = exp(lcab_{t,k} + nll_n)`。
`∂nll_n/∂lp[t,n,k] = -γ_{t,k}`（値は占有事後確率 `-γ` に一致し、
標準の CTC 勾配と同じ意味論を持つ）。

**β の再帰**（現フレームを除く規約に基づく。α と対をなす鏡像）:
境界は `β_{T_n-1}(L'-1) = 0`（log 1）・`β_{T_n-1}(L'-2) = 0`
（`L' > 1` のとき）、それ以外は `-∞`。`t < T_n - 1` は
`β_t(s) = logaddexp( lp[t+1, l'_s] + β_{t+1}(s),`
`lp[t+1, l'_{s+1}] + β_{t+1}(s+1)`（`s+1 < L'`）`,`
`lp[t+1, l'_{s+2}] + β_{t+1}(s+2)`（`s+2 < L'` かつ
`l'_{s+2} != blank` かつ `l'_{s+2} != l'_s` のときのみ）`)`。

**恒等式による検証**（advisor レビュー指摘の sanity check）:
`Σ_k γ_{t,k} = 1`（`t < T_n` の各時刻で成立する。α・β の
forward-backward 恒等式 `Σ_s exp(α_t(s)+β_t(s)) = P` から従う）。
`log_softmax` 合成（`logits → log_softmax → ctc_loss`）を経由した
logits 勾配は `softmax(logits) - γ` に一致し、これは PyTorch の
`ctc_loss` backward（`softmax(logits) - γ`。Graves の式 16 相当）と
一致する（フレームごとに `Σ_k γ_{t,k} = 1`・`Σ_k softmax_k = 1` が
成り立つため）。`crates/autodiff/tests/loss_ops.rs::
ctc_loss_grad_matches_numeric_central_difference` で `log_probs` に
対する数値微分と解析 VJP の一致を固定する。

**PyTorch backward との違い**: PyTorch 本体の `ctc_loss` backward
（`LossCTC.cpp`）は `exp(lp) - exp(lcab + nll - lp)` を返すが、これは
`log_probs` が `log_softmax` 出力であることを前提にした「正規化前
logits」に対する勾配（Graves の式 16）であり、`log_probs` に対する
真の VJP ではない。本実装は `log_probs` を独立入力とみなした真の
VJP（`∂nll/∂lp[t,n,k] = -γ_{t,k}`）を採用する。理由は、数値中心差分
による勾配検査と autodiff の合成可能性（任意の上流演算と組み合わせ
ても正しいこと）を優先するため。`log_softmax` 合成後は上記のとおり
両者は一致する。

**フレーム外・整列不能・`zero_infinity`**: `t >= T_n` の勾配は `0`
（そのフレームは計算対象外のまま初期値 `0` で残す）。
`zero_infinity = true` かつ `nll_n = +∞` のサンプルは勾配も全 `0`。
`zero_infinity = false` かつ `nll_n = +∞` のとき、有効フレーム
（`t < T_n`）の勾配は `NaN` とする（数学的に未定義。PyTorch も
`NaN` を返す規約に合わせる。実装では `exp(-∞+∞)` の暗黙の `NaN` に
任せず、分岐で明示的に書き込む）。`N == 0` や `T_n == 0` など空
shape のときは、同 shape のゼロ勾配を返す（空 shape の部分積
overflow 対策として冒頭で早期 return する）。`targets`・長さ・
`options` は非追跡なので、勾配は `log_probs` の 1 系統のみ。

### §2.5 バックエンド

`BackendOps` にメソッドは新設しない（§2.1）。`Op::CtcLoss` は常に
ホスト参照実装（`crate::eval`／`crate::grad`）で計算し `push_eager`
で記録する。CUDA／Metal テープからは `materialize_fallible` 経由の
ホスト計算で到達する（イシューのいう「Unsupported フォールバック
＝ホスト計算」を満たす）。ホスト計算なので、3 バックエンドで bit
一致が期待できる。ただし受け入れ判定は REQ-2 の統一複合判定を正とし、
tolerance は変えない。

## §3 PyTorch との差分

- `log_probs` の rank 2（unbatched `[T, C]`）は非対応。安全側に倒して
  `InvalidArgument` で拒否する
- `N == 0` は損失 `0.0`（PyTorch は `NaN` を返す。既存 `mse_loss` 等
  の空バッチ規約 `n == 0 → 0.0` を踏襲する安全側の判断）
- `log_probs` に対する直接の VJP は真の勾配（`-γ`）であり、PyTorch
  本体の backward（正規化前 logits 相当の勾配）とは異なる。
  `log_softmax` 合成後は一致する（§2.4）
- `targets` の型は `Tensor<i32>`、長さは `&[usize]`（PyTorch は
  `IntTensor`／`LongTensor` を許容するが、本実装は固定型で受け取る）
- `reduction='none'`（既存公開 `Reduction` enum の拡張）は対象外

## §4 数値一致・parity

`Op::CtcLoss` はホスト計算のみで `BackendOps` を経由しないため、
3 バックエンド（CPU／CUDA／Metal）で **bit 完全一致**することが
期待される（`Op::L1Loss` と同型の性質）。CPU（`CpuBackendOps`）と
`NaiveOps` の bit 一致は `crates/facade/tests/loss_ops_backend_
parity.rs::cpu_ctc_loss_forward_and_backward_match_naive_reference`
で固定済み。CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機は本
リポの実行環境に到達手段がないため未実測
（`docs/perf/logs/ctc-loss-2168/README.md` へ申し送る）。受け入れ
判定は REQ-2 の統一複合判定を正とする。

## §5 承認事項（facade 公開面の拡張）

以下は未承認のため保留する（`crates/facade/src/lib.rs::
LossOpsHoldDoctestGuard`（#2166 で導入・本イシューで対象を 7 関数へ
拡張）・`crates/facade/tests/api_surface.rs` の対応する 4 テストで
機械的に固定）:

- `Var::ctc_loss(&self, targets, input_lengths, target_lengths,
  options, reduction)` の facade 到達

承認が得られたら、`Var` への薄い委譲メソッド（`loss_ops::ctc_loss`
を呼ぶだけ）を追加し、`LossOpsHoldDoctestGuard`・対応する
api_surface のテストのうち `ctc_loss` を撤去する（#2166／#2167 分は
引き続き保留対象のまま残る場合、ガードは該当関数名のみ縮小する）。

## §6 スコープ外（out-of-scope-tracking.md）

- `compat::Loss`（`compile()`）への CTC 追加は #2169 の担当。本 PR
  では `crates/facade/src/compat/training.rs` を変更しない
- GPU 専用 CTC カーネル（`BackendOps` の融合メソッド）は別イシュー。
  本 PR では起票しない（承認なしに Issue を作らない規約に従い、
  必要な場合は PR 上でユーザーに提案する）
- CUDA（GB10）／Metal（M4 Max）の実機 parity は未実測。§4・
  `docs/perf/logs/ctc-loss-2168/README.md` へ申し送る
- `reduction='none'`（`Reduction` enum の拡張）は既存公開 enum に
  関わるため本イシューの対象外
- `log_probs` の rank 2（unbatched）入力
- PyTorch 実機との数値突合（勾配規約の差を含む。§3）
