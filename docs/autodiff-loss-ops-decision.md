# L1 損失・CrossEntropy の label_smoothing・ignore_index・class_weight 設計判断記録

イシュー #2166（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-reduce-ops-decision.md`（#2147）と同型の記録。

## §0 結論

PyTorch `nn.L1Loss` 相当・`nn.CrossEntropyLoss(label_smoothing=,
ignore_index=, weight=)` 相当の 2 API を、**`fandhe_ai_autodiff` の
うち facade が再エクスポートしない自由関数モジュール `loss_ops`**
（`crates/autodiff/src/loss_ops.rs`）として実装した（`reduce_ops`
〈#2147〉・`matrix_ops`〈#2144〉と同じ判断枠組み）。`Var` に inherent
の `pub fn` は追加していない。

- `l1_loss(pred, target, reduction)`: 新規 `Op::L1Loss`。`BackendOps`
  に対応メソッドを持たず常にホスト参照実装（`eval::l1_loss_forward`）
  を経由する
- `cross_entropy_loss_with(logits, targets, class_dim, reduction,
  options)`: `options` が既定値（`CrossEntropyOptions::default()`）の
  ときは既存 `Var::cross_entropy_loss` へ**丸ごと委譲**し、新規 `Op`
  を構築しない（R3。既存経路は一切変更しない）。非既定オプション
  時のみ新規 `Op::CrossEntropyLossWithOptions` を構築しホスト参照
  実装（`eval::cross_entropy_loss_with_options_forward`）を経由する

`nn::loss::L1Loss`（新規構造体）・`nn::loss::CrossEntropyLoss::
forward_with`（既存構造体への追加メソッド。構造体フィールドは不変）
を薄いラッパーとして提供する。`nn::loss` は facade が再エクスポート
しない（`crates/facade/src/nn/mod.rs` は `nn::rnn` のみ再エクスポート）
ため、これらも facade 公開面には出ない。

## §1 背景

イシュー #2166・親 #2131 のどちらにも所有者の承認コメントはない
（着手前に `gh issue view --json comments` で確認済み）。関連イシュー
#2169（`compile()` の `Loss` enum への統合）は別 PR の担当で、本 PR は
`crates/facade/src/compat/training.rs` を変更しない。親 #2131 は
このツリーでの facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の
2 段階と定めているため、本実装は内部クレート限定に倒す。

## §2 各演算の設計

### §2.1 L1 損失（`Op::L1Loss`）

`|pred − target|` の `reduction` 縮約。forward
（`eval::l1_loss_forward`）は `f64` アキュムレータで index 順に蓄積し
1 回だけ `f32` へ downcast する（既存 `mse_loss`／`cross_entropy_loss`
の `f32` 蓄積は R3 のため変更していないが、新規追加の forward には
一般則〈`.claude/rules/coding-rust.md`〉を適用する判断）。
`numel == 0` は mean／sum とも `0.0`（`mse_loss` の規約を踏襲）。

VJP（`grad.rs::l1_loss_vjp`）: `dPred = scale·sign(pred − target)`
（`sign(0) = 0`・`NaN` は `NaN` を伝播。`l1_grad_sign` で手書き。
標準ライブラリの `f32::signum` は `sign(0)` 契約が異なるため使わない）。
`dTarget = −dPred`（`Op::MseLoss`／`Op::HuberLoss` と同じ符号反転
パターン。新規カーネル起動なしのホスト側 map）。

`BackendOps` は拡張しない（GPU 専用カーネルは §4 のとおりスコープ
外）ため、`Op::MseLoss`／`Op::HuberLoss` と異なり `Unsupported`
フォールバック分岐自体を持たない——常にホスト計算のみ。

### §2.2 CrossEntropy オプション（`Op::CrossEntropyLossWithOptions`）

意味論は PyTorch `aten/src/ATen/native/LossNLL.cpp` の label
smoothing 実装に準拠する。記法（サンプル `s`、クラス `c`）:

- `lp_c = x_c − lse`（log-softmax。既存 `cross_entropy_loss` と同じ
  max シフト安定化）
- `w_c` はクラス重み（`class_weight` 未指定は全クラス `1.0`）
- `W = Σ_{非 ignore} w[t_s]`（`Mean` の分母）
- `L_s = (1−ε)·w[t_s]·(−lp_{t_s}) + (ε/C)·Σ_c w_c·(−lp_c)`
  （ignore されたサンプルは寄与 0・`W` にも含めない）
- `Mean` は `(Σ_s L_s)/W`、`Sum` は `Σ_s L_s`
- `W == 0`（全サンプル ignore、または重み和が 0）は損失 `0.0`・
  勾配 `0`

勾配（`grad.rs::cross_entropy_loss_with_options_vjp`）:
`dx_c = s·[(1−ε)·w[t_s]·(p_c − 1{c==t_s}) + (ε/C)·(p_c·Σ_k w_k − w_c)]`
（`p_c = softmax(logits)[c]`。`s` は `Mean` なら `g/W`・`Sum` なら
`g`・分母 0 なら `0`）。ignore された行は勾配 0。

蓄積はいずれも `f64`・index 順で行い最後に 1 回だけ `f32` へ
downcast する。`targets`・`class_weight` は非追跡データのため勾配は
`logits` の 1 系統のみ（`Op::CrossEntropyLoss` の `targets` と同型）。

### §2.3 検査順序（REQ-8・A03）

`l1_loss`: ①`check_same_tape` → ②shape 一致（`require_same_shape`）
→ ③確保前バイト数上限検査（`checked_bytes_for::<f32>`）→ ④実体化
→ ⑤forward → ⑥ノード記録。

`cross_entropy_loss_with`（非既定オプション経路）: ①`class_dim` 範囲・
targets shape 一致（`reduce_out_shape`）→ ②確保前バイト数上限検査 →
③`label_smoothing` が有限かつ `[0, 1]` → ④`class_weight` が shape
`[C]`・全要素有限かつ非負 → ⑤targets 全添字が `ignore_index` と
一致するか `0 <= t < C` → ⑥実体化 → ⑦forward → ⑧ノード記録。

## §3 PyTorch との差分

- `ignore_index` は PyTorch の既定 `-100` を採用せず、既定 `None`
  （無効）とする。明示指定時のみ有効になる
- 全サンプル ignore、または `class_weight` の重み和が 0 のとき、
  PyTorch は `NaN` を返すが、本実装は `0.0`（損失・勾配とも）を返す
  （`Op::MseLoss` の空バッチ規約 `n == 0 → 0.0` を踏襲する安全側の
  判断）
- `class_weight` は非負値のみ許容する（PyTorch は負値も受け付けるが、
  負の重みは損失の単調性が崩れるため fail-closed に拒否する安全側の
  判断）

## §4 数値一致・parity

`Op::L1Loss`・`Op::CrossEntropyLossWithOptions` はいずれもホスト計算
のみで `BackendOps` を経由しないため、3 バックエンド（CPU／CUDA／
Metal）で **bit 完全一致**することが期待される（`Op::CrossEntropyLoss`
〈既存〉と同型の性質）。CPU（`CpuBackendOps`）と `NaiveOps` の bit
一致は `crates/facade/tests/loss_ops_backend_parity.rs` の属性なし
テストで固定済み。CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機
は本リポの実行環境に到達手段がないため未実測（`docs/perf/logs/
loss-ops-2166/README.md` へ申し送る）。受け入れ判定は REQ-2 の統一
複合判定を正とする。

## §5 承認事項（facade 公開面の拡張）

以下は未承認のため保留する（`crates/facade/src/lib.rs::
LossOpsHoldDoctestGuard`・`crates/facade/tests/api_surface.rs` の
4 テストで機械的に固定）:

- `Var::l1_loss(&self, target: &Var, reduction: Reduction)` の facade
  到達（`Var` は facade からそのまま再エクスポートされるため、
  inherent メソッド追加が即座に公開面を広げる）
- `Var::cross_entropy_loss_with(&self, targets, class_dim, reduction,
  options)` の facade 到達

承認が得られたら、`Var` への薄い委譲メソッド（`loss_ops::l1_loss`／
`cross_entropy_loss_with` を呼ぶだけ）を追加し、
`LossOpsHoldDoctestGuard`・対応する api_surface の 4 テストを撤去する。

## §6 スコープ外（out-of-scope-tracking.md）

- `compat::Loss`（`compile()`）への L1・CE オプション対応は #2169 の
  担当。本 PR では `crates/facade/src/compat/training.rs` を変更しない
- GPU 専用カーネル（`BackendOps::l1_loss` 等の融合カーネル）は別
  イシュー。本 PR では起票しない（承認なしに Issue を作らない規約に
  従い、必要な場合は PR 上でユーザーに提案する）
- CUDA（GB10）／Metal（M4 Max）の実機 parity は未実測。§4・
  `docs/perf/logs/loss-ops-2166/README.md` へ申し送る
