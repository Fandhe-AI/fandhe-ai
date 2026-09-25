# prod・logsumexp・any・all・norm_p（p-ノルム）の設計判断記録

イシュー #2147（親 #2131）。`docs/autodiff-matrix-ops-decision.md` と
同型の記録。

## §0 結論

PyTorch 互換の縮約演算 5 種（`prod`／`logsumexp`／`any`／`all`／
`norm_p`）を、**`fandhe_ai_autodiff` のうち facade が再エクスポート
しない自由関数モジュール `reduce_ops`**（`crates/autodiff/src/
reduce_ops.rs`）として実装した（`matrix_ops`〈#2144〉・
`rearrange_ops`〈#2143〉・`bool_ops`〈#2141〉と同じ判断枠組み）。
`Var` に inherent の `pub fn` は追加していない。

Issue #2147 は「facade への 5 メソッドの `pub use` 再エクスポート
（経路 2）」を承認事項として明示しているが、`facade` は `Var` を
そのまま再エクスポートしている（`crates/facade/src/lib.rs`）ため、
`Var` に inherent メソッドを 1 つ足すだけで facade の公開面が広がる。
このツリー（親 #2131）の先例に倣い、承認が取れるまで **autodiff に
新設したモジュール `reduce_ops` の自由関数 5 個**で「5 メソッド」を
満たす（承認後に追加する作業は `Var::prod` 等の薄い委譲メソッドと
facade ガードの撤去のみ）。

`prod`／`any`／`all` は既存 `Op` の合成のみ（新規 `Op` なし）。
`logsumexp`／`norm_p` は数値安定化のため専用 `Op`（`Op::LogSumExp`／
`Op::PNorm`）を追加した——`sum`／`exp`／`log` や `max`／`pow`／`sum`
の素朴な合成では、全要素が `-inf`（または `+inf`）の lane・overflow
を起こす `p` で `NaN` が出るため（`Var::std` の先例と同じ理由）。

## §1 背景

イシュー #2147・親 #2131 のどちらにも所有者の承認コメントはない
（着手前に `gh issue view --json comments` で確認済み）。親 #2131 は
このツリーでの facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の
2 段階と定めているため、本実装は内部クレート限定に倒す。

## §2 各演算の設計

### §2.1 共通方針

シグネチャは既存の `Var::sum`／`max` にそろえ、`dim: Option<usize>`
を取る（`None` は全軸縮約でスカラー `[]` を返す）。keepdim 版・複数軸
版（`*_dims`）はスコープ外。

### §2.2 `prod`: 既存 Op の合成（新規 Op なし）

`cumprod(dim) → narrow(dim, n-1, 1) → contiguous() → squeeze(dim)`。
`dim=None` のときは先に `reshape([numel])` してから `cumprod(0)` を
取る。

- `Op::Cumprod`（#1731）の forward は `f64` アキュムレータで計算し
  1 回だけ `f32` へ落とす。VJP は除算を使わない厳密形（排他 prefix
  積）のため零要素の個数に依らず正確。
- **`contiguous()` を `squeeze` の直前に挟む理由**: `narrow` は
  `dim` が末尾軸でない限り非 contiguous な stride view を返しうる。
  `squeeze`（`Var::reshape` への委譲）は contiguous な入力を要求する
  （`ShapeError::NonContiguousReshape`）。`matrix_ops::diag_2d_to_1d`
  の `narrow → gather → squeeze` は `gather` が暗黙に実体化するため
  同じ問題を踏まないが、`prod` は `gather` を経由しないため明示的に
  `contiguous()` を挟む必要がある（実装時に `dim=Some(axis)`（`axis`
  が末尾軸でない 2-D 入力）の統合テストで `NonContiguousReshape` の
  回帰を検出し是正した）。
- 空縮約（`n == 0`）は PyTorch と同じく単位元 **1.0** を返す。
  `narrow(n-1)` の underflow を避けるため合成より前に分岐する。
  **2026-09-25 追記（PR #2263 codex-review P1/P2 是正・イシュー
  #2147）**: 対応前は `out_shape.iter().product()`（`usize` 乗算の
  未検査）と `vec![1.0f32; numel]`（allocation 未検査）を実行しており、
  `[0, usize::MAX]` を `dim=Some(0)` で縮約するような shape で
  capacity overflow panic しえた（本番経路 panic 禁止規約
  `.claude/rules/coding-rust.md` 違反）。加えて `x` から独立した
  `push_leaf` 定数葉で返しており、backward で `x` への経路が失われて
  いた。是正後は確保前に `checked_bytes_for::<f32>`
  （`bool_ops.rs`。要素数積のオーバーフロー検査 + `Vec` allocation
  上限〈`isize::MAX` バイト〉検査）で拒否し、`x.sum(dim)`
  （空縮約軸は単位元 `0.0` を返す既存契約）+ 定数バイアス `1.0` の
  合成（`reduce_ops.rs::empty_reduce_identity`）で `x` への計算グラフ
  依存を保ったまま単位元を返す。
  **2026-09-25 追記 3（PR #2263 codex-review P1 是正 3・§2.6 参照）**:
  上記の検査は当初この空縮約分岐にのみ置かれていたが、非空縮約
  （`contiguous()`／`cumprod` の実体化）が未検査のまま残っていた。
  現在は空縮約分岐ではなく `prod` 関数冒頭で `ensure_alloc_fits_f32`
  により一律に検査する（分岐に依らず常に検査済み）。

### §2.3 `any`／`all`: 既存 Op の合成（新規 Op なし）

`any`: `x.ne(&zero) → max(dim)`。`all`: `x.ne(&zero) → min(dim)`。
`zero` は同じ tape 上のスカラー葉（`Tensor::scalar(0.0)` →
`Tape::push_leaf`）。

- 非ゼロを真とする。`NaN != 0` は真なので `NaN` は真として扱い、
  PyTorch と一致する。`-0.0` は偽。
- 出力値は厳密に `0.0` か `1.0` だけで縮約順序に依存しないため、
  3 バックエンドで **bit 完全一致**する。
- 勾配は `ScalarBinaryOp::Ne` の VJP がゼロを返すため、合成しただけで
  自動的に勾配ゼロの tape ノードになる。
- 空縮約: `max`／`min` は単位元を持たずエラーになるため、合成より前に
  分岐し `any(∅) = 0.0`・`all(∅) = 1.0` を返す（PyTorch と同じ）。
  **2026-09-25 追記（PR #2263。`prod` §2.2 追記と同じ是正）**:
  確保前検査（`checked_bytes_for::<f32>`）を経たうえで、`x` から
  独立した定数葉ではなく `x.ne(&zero)` を `empty_reduce_identity`
  （`x.ne(&zero)`（縮約対象要素数 0 のため 0 要素）の `sum(dim)` が
  単位元 `0.0` を返す契約 + `any` はバイアス `0.0`・`all` はバイアス
  `1.0`）へ通した合成で返す。
  **2026-09-25 追記 3（PR #2263 codex-review P1 是正 3・§2.6 参照）**:
  §2.2 の追記 3 と同じ理由で、現在は空縮約分岐ではなく `any`／`all`
  関数冒頭で `ensure_alloc_fits_f32` により一律に検査する（`x.ne(&zero)`
  の非空縮約側の実体化も併せて守られる）。
  `ne` の VJP は常にゼロを返す契約
  （非空の `any`／`all` と同じ）のため、勾配は従前どおりゼロだが
  `x` への逆伝播経路自体は保たれる。
- bool 出力版は #2141（`bool_ops`）の対象で本モジュールの対象外。

### §2.4 `logsumexp`: 専用 Op

新規: `Op::LogSumExp { input, dim }`・defaulted
`BackendOps::logsumexp`（既定 `Unsupported`）・
`eval::logsumexp_along`（ホスト参照実装）・
`grad.rs::logsumexp_vjp`。

forward（lane ごと）:
1. `m = max(x)` を `f64` で求める（`NaN` 伝播 max。`f64::max` は
   非 `NaN` 側を返してしまうため専用の `nan_propagating_max_f64` を
   使う）。`m` が非有限（±inf）なら安定化シフトを `0` に置き換える。
2. `acc = Σ exp((x_i as f64) - shift)` を `f64` で蓄積する。
3. `y = ln(acc) + shift` を計算し、`f32` へは 1 回だけ落とす。

VJP: `dx_i = g · exp(x_i - y)` を `f64` で計算して `f32` へ落とす。
`y == -inf`（全要素が `-inf`）の lane は勾配 `0` とする——PyTorch は
この lane で `NaN` を返すが、`NaN` 勾配で学習を汚染しない安全側の
判断（§4 に記録）。

空縮約は `AutodiffError::InvalidArgument`（`Var::max`・`Var::norm` と
同じく `-inf` を黙って返さない）。

backend-cpu の `reduction::logsumexp` は eval と**同じ lane ごとの
逐次 `f64` アルゴリズム**で実装し、CPU カーネルとホストフォールバックの
結果が bit 一致する構造にした。**軸指定（`dim=Some(axis)`）は lane
（出力要素）間だけ rayon で並列化**する（各 lane 内は逐次走査のため
bit 一致は保たれる）。**全軸縮約（`dim=None`）は単一 lane しかない
ため rayon 並列化を行わず、eval と同一の単一逐次 `f64` fold**を直接
使う（`crates/backend-cpu/src/reduction.rs::logsumexp_slice`／
`vector_norm_p_slice`）。

**2026-09-25 追記（PR #2263 codex-review P2 是正）**: 対応前は
`logsumexp_slice`／`vector_norm_p_slice`（全軸縮約経路）が `sum_slice`
等と同じ「CHUNK（4096）単位でチャンク内を逐次累積 → チャンク結果を
rayon 経由でチャンク番号順に結合」方式を使っており、浮動小数点加算が
結合則を満たさないため要素数が CHUNK を超えると eval の単一逐次
fold と異なる丸め結果になりえた（bit 一致契約違反。CHUNK を跨がない
小さい入力では顕在化しない）。是正後は上記のとおり全軸縮約経路のみ
`par_chunks` を使わず eval と同一の単一逐次 `f64` fold を用いる
（全軸縮約の rayon 並列性を犠牲にする。軸指定側は影響なし）。
回帰テストは `crates/facade/tests/reduce_ops_backend_parity.rs::
cpu_logsumexp_vector_norm_p_forward_bit_matches_naive_reference_across_chunk_boundary`
（`logsumexp` は `y ≈ 0` 近傍に潰す専用フィクスチャで修正前コードとの
bit 不一致を実測確認済み。`vector_norm_p` は `n = 3・CHUNK + 17`
規模では相対誤差が f32 丸め粒度に届かず修正前でも自然には bit 不一致を
再現できなかったため将来のリグレッション防止ロックとして追加）。

**2026-09-25 追記 2（PR #2263 codex-review P1 是正・非空縮約への確保前
バイト数上限検査の拡張）**: §2.2／§2.3 の空縮約（`n == 0`）向け
`checked_bytes_for::<f32>` は `out_shape` の要素数積・バイトサイズを
検査するが、`logsumexp`／`vector_norm_p` は `n != 0`（非空縮約）でも
`x` が小さなストレージを巨大な shape へ broadcast した view の場合に
同種の未検査確保が残っていた（要素数積は `usize` に収まってもバイト
数〈`f32` 換算〉が `isize::MAX` を超えうる）:

- `backend-cpu::reduction::logsumexp`／`vector_norm_p`
  （`crates/backend-cpu/src/reduction.rs`）: 軸指定
  （`dim=Some(axis)`）は `axis_reduce_logsumexp`／
  `axis_reduce_vector_norm_p` の `.collect()` が `out_shape` サイズの
  `Vec<f32>` を確保する前に `checked_alloc_numel_f32(&out_shape)` を
  追加。全縮約（`dim=None`）の非 contiguous 入力（`as_slice()` が
  `None`）は `gather_elements` が入力全体を実体化する前に
  `checked_alloc_numel_f32(a.shape())` を追加（`checked_alloc_numel_f32`
  は `checked_product` に `f32` 換算のバイトサイズ検査を足した
  `pub(crate)` ヘルパで、`backend-cuda::ops::checked_bytes_for`／
  `backend-metal::ops::checked_bytes_for`／`autodiff::bool_ops::
  checked_bytes_for` と同型の独立複製）。
- `autodiff::reduce_ops::logsumexp`／`norm_p`
  （`crates/autodiff/src/reduce_ops.rs`）: `n == 0` 検査の直後・
  `materialize_one` 呼び出し前に `checked_bytes_for::<f32>` を入力
  shape・`out_shape` の双方に適用する。これは
  `BackendOps::logsumexp`／`vector_norm_p`（CPU／CUDA／Metal いずれの
  実装も）と `Unsupported` 時のフォールバック
  `eval::logsumexp_along`／`vector_norm_p_along`
  （`crates/autodiff/src/eval.rs`。`dim=Some(axis)` の
  `vec![0f32; outer * inner]` 確保）の両方を単一の事前検査で守る
  （`eval.rs` モジュール冒頭の「shape が既に整合していることを前提とし
  `ShapeError` を返さない」契約を保つため、`eval.rs` 自体は
  `Result` 化しない。事前条件は両関数の doc に明記）。
  `grad.rs::logsumexp_vjp`／`pnorm_vjp`（backward の `dense_vec`
  実体化）は forward が拒否した shape は tape に push されないため
  追加検査不要（doc に事前条件として明記のみ）。
  `norm_p` の `p ∈ {1.0, 2.0}` 委譲先（`Var::norm_l1`／`norm_l2` →
  `Var::norm` → `eval::vector_norm_along`）自体の改修は本 PR の差分外
  （イシュー #1723 の既存経路）のため対象外（スコープ外として
  `.claude/rules/out-of-scope-tracking.md` の対象候補。§5 参照）。
  **ただし §2.6（2026-09-25 追記 3）のとおり、委譲判定自体を検査より
  後ろへ移したため、委譲先が未検査であっても本関数側の検査で巨大
  broadcast shape を委譲前に拒否できる**（委譲先自体は改修していない
  ため、委譲先を直接呼ぶ既存の `Var::norm_l1`／`norm_l2`／`Var::norm`
  経由の呼び出しには本検査は及ばない。この残存範囲は §5 のスコープ外
  候補のまま）。
- 回帰テストは `crates/backend-cpu/src/reduction.rs` の
  `logsumexp_vector_norm_p_axis_reduce_rejects_huge_broadcast_output_without_panicking`／
  `logsumexp_vector_norm_p_full_reduce_rejects_huge_broadcast_input_without_panicking`
  と、`crates/autodiff/tests/reduction_parity.rs` の
  `logsumexp_norm_p_axis_reduce_rejects_huge_broadcast_output_without_panicking`／
  `logsumexp_norm_p_full_reduce_rejects_huge_broadcast_input_without_panicking`
  （いずれも `Tensor::broadcast_to` で構築した非 contiguous な巨大
  shape view が panic せず `ShapeError::ElementCountOverflow` を返す
  ことを検証する）。

### §2.5 `norm_p`: 専用 Op

`VectorNormOrd` は拡張しない（crates.io 公開クレートで `Eq` を
derive しているため `Lp(f32)` を足すと壊れる。既存の
`eval::vector_norm_along`／`reduction::NormKind` には `_ => 0.0` の
安全側フォールバックがあり、未知 variant を静かに 0 とみなす経路が
既にあるため、新規 variant の追加ではなく別 `Op` にする方が
fail-closed）。

新規: `Op::PNorm { input, p: f32, dim }`（`Op` は
`#[derive(Debug, Clone)]` のみのため `f32` を持たせてよい）・
defaulted `BackendOps::vector_norm_p`（既定 `Unsupported`）・
`eval::vector_norm_p_along`・`grad.rs::pnorm_vjp`。

**`p` の検証（fail-closed）**: 有限かつ `p > 0` のみ受け付ける。
`NaN`・±inf・`0`・負の値は `InvalidArgument` で拒否する。PyTorch は
inf・`0`・負の `p` も受け付けるが、inf ノルム勾配の分配方式（本リポの
`max_vjp` は先勝ちのみで均等分配は別イシュー #2154 の対象）が未定の
ため見送る（§4・§5 に記録）。

`p == 1.0`／`p == 2.0` は既存の `Var::norm_l1`／`norm_l2`（`Var::norm`
`pub(crate)` 経由）へ委譲する。これにより `norm_p(x, 2.0, d)` と
`norm_l2(d)` が **bit 同一**になる（`crates/autodiff/tests/
reduction_parity.rs` で固定）。

forward は overflow を避けるスケール形で計算する:
- `mx = max|x_i|` を `f64` で求める（`NaN` 伝播）。
- `mx == 0` なら `0`、`mx` が `inf` なら `inf`、`NaN` は伝播。
- それ以外は `norm = mx · (Σ (|x_i|/mx)^p)^(1/p)` を `f64` で計算し、
  `f32` へは 1 回だけ落とす（`p` が大きい場合〈例 `p=50,
  |x|≈1e38`〉でも `f64` で overflow しない）。

VJP: `dx_i = g · sign(x_i) · (|x_i|/norm)^(p−1)` を比の形（`norm^
(p−1)` を直接求めず overflow を避ける）で `f64` 計算する。
- `norm == 0` の lane は全要素 `0`。
- `x_i == 0` の要素は `0`（`p < 1` で `0^(負)` が `inf` になるのを
  防ぐ劣勾配の選択）。

空縮約は既存の `Var::norm` と同じく `InvalidArgument`。

### §2.6 確保前バイト数上限検査の統一契約（2026-09-25 追記 3・PR
#2263 codex-review P1 是正 3）

§2.2〜§2.5 の各追記は個別に検査を足す方式だったため棚卸しの範囲が
狭く、同じ類型の指摘が 2 件残っていた:

1. `prod`／`any`／`all` は**空縮約（`n == 0`）分岐でしか**
   `checked_bytes_for` を呼んでおらず、非空縮約では `prod` の
   `contiguous()`／`cumprod`、`any`／`all` の `x.ne(&zero)` が入力
   shape 相応の `Vec` を無検査に確保しうるまま残っていた。
2. `norm_p` は `p == 1.0`／`p == 2.0` の委譲判定（`Var::norm_l1`／
   `norm_l2` へ委譲）が確保前検査より前にあり、委譲先（`Var::norm`
   `pub(crate)` 経由。本 PR の差分外・イシュー #1723 の既存経路）が
   独自に確保前検査を持たない限り、新設 API 側の検査を迂回して
   そのまま既存経路へ渡っていた。

**是正**: `crates/autodiff/src/reduce_ops.rs::ensure_alloc_fits_f32`
（唯一の共有ヘルパ。`checked_bytes_for::<f32>` を入力 shape・
`out_shape`〈既に求まっている場合〉の両方に適用する）を新設し、
[`prod`]・[`logsumexp`]・[`any`]・[`all`]・[`norm_p`] の**全 5 公開
入口が関数冒頭・あらゆる分岐（空縮約・`p` の特殊化・`dim` の有無・
`Unsupported` フォールバック）や合成演算・実体化（`contiguous`／
`cumprod`／`ne`／`materialize_one`／既存 API への委譲）よりも前**に
本ヘルパを呼ぶ形へ統一した。個別に追加していた `checked_bytes_for`
呼び出しはすべて `ensure_alloc_fits_f32` 経由に寄せ、重複を解消した。

`prod`（`dim=None`／`Some(axis)` いずれの経路も入力と同じ要素数の
中間 shape〈`reshape`／`cumprod`〉しか作らない）・`any`／`all`
（`x.ne(&zero)` は入力と同じ shape）はいずれも入力 shape の検査のみで
実体化経路全体が守られる。`norm_p` の `p ∈ {1.0, 2.0}` 委譲は検査後に
行う（委譲先自体の改修はスコープ外のまま、委譲前に巨大 broadcast
shape を拒否する）。

回帰テストは `crates/autodiff/tests/reduction_parity.rs` の
`prod_any_all_axis_reduce_rejects_huge_broadcast_output_without_panicking`／
`prod_any_all_full_reduce_rejects_huge_broadcast_input_without_panicking`／
`norm_p_one_and_two_reject_huge_broadcast_before_delegating`
（`Tensor::broadcast_to` で構築した非空・非 contiguous な巨大 shape
view が panic せず `ShapeError::ElementCountOverflow` を返すことを
検証）と、`crates/facade/tests/reduce_ops_backend_parity.rs::
cpu_norm_p_two_rejects_huge_broadcast_before_delegating`（`fandhe_ai::
tape()`〈`CpuBackendOps`〉経由でも同じ検査が dispatch 前に効くことを
確認する facade 経路の代表 1 件）。

## §3 数値契約（まとめ）

| 演算 | forward | backward |
|---|---|---|
| `prod` | `Op::Cumprod` の `f64` アキュムレータに委譲（REQ-2 統一複合判定。縮約順序がバックエンド依存） | 除算を使わない厳密形の合成 VJP（REQ-2 統一複合判定） |
| `logsumexp` | `f64` 二段計算・専用 `Op`（REQ-2 統一複合判定） | `f64` 二段計算（REQ-2 統一複合判定） |
| `any`／`all` | 厳密 `0.0`／`1.0`（**bit 完全一致**） | `ne` の VJP がゼロ（**bit 完全一致**） |
| `norm_p`（`p≠1,2`） | `f64` スケール形・専用 `Op`（REQ-2 統一複合判定） | `f64` 比の形（REQ-2 統一複合判定） |
| `norm_p`（`p=1,2`） | 既存 `norm_l1`／`norm_l2` へ委譲（**bit 同一**） | 同上へ委譲 |

## §4 PyTorch との差分

- `logsumexp` の `y == -inf`（縮約対象が全て `-inf`）lane の勾配は
  `0`（PyTorch は `NaN`）。`NaN` 勾配で学習を汚染しないための安全側
  の判断。
- `norm_p` の `p` は有限かつ正のみ許容（`0`・負・±inf・`NaN` は
  `AutodiffError::InvalidArgument`）。PyTorch は inf・`0`・負の `p`
  も受け付けるが、inf ノルムの勾配分配方式が本リポでは未定（#2154）
  のため見送る。
- keepdim 版・複数軸版（`*_dims`）は非対応。

## §5 スコープ外（out-of-scope-tracking の候補）

- facade 公開（`Var::prod` 等の委譲メソッド追加と保留ガードの撤去）:
  経路 2 の承認待ち。窓口は #2147・#2131。
- bool 出力版の `any`／`all`: #2141 に従属。
- GPU 専用カーネル（`logsumexp`・`norm_p`）: 別イシュー。
- `p ∈ {0, ±inf, 負}` のノルム: inf ノルム勾配の分配方式が #2154 の
  決定に依存するため。
- keepdim 版・複数軸版（`*_dims`）: 対象外。
- CUDA・Metal の実機 parity 実測: 申し送り（`docs/perf/logs/
  reduce-ops-2147/README.md`）。
- `Var::norm`（`norm_l1`／`norm_l2`。`var.rs`）・
  `eval::vector_norm_along`（`eval.rs:637-638` の `dense_vec`＋
  `vec![0f32; outer * inner]`）は §2.4 追記 2 と同種（小さな
  ストレージを巨大な shape へ broadcast した view で確保前検査が
  ない）だが、イシュー #1723 の既存経路であり本 PR（#2147・PR
  #2263）の差分外のため未修正。将来の是正候補として記録のみ
  （codex-review 指摘・PR #2263 レビュー時点）。

## §6 承認事項（未承認として列挙）

1. facade 公開（経路 2。上記スコープ外 1 と同じ）
2. bool 出力版の `any`／`all`（#2141 側の承認事項）
3. GPU 専用カーネル
4. `p ∈ {0, ±inf, 負}` のノルム（#2154 の決定待ち）

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/reduce-ops-2147/README.md` へ測定コマンド案・
期待結果を申し送る。

## §8 実装記録（イシュー #2147）

- `crates/tensor-core/src/backend_ops.rs`: `BackendOps::logsumexp`・
  `vector_norm_p`（defaulted・既定 `Unsupported`）＋回帰テスト
  `logsumexp_vector_norm_p_defaults_are_unsupported`
- `crates/backend-cpu/src/reduction.rs`: `ReduceError::InvalidOrder
  (f32)` variant・`logsumexp`／`vector_norm_p`（`nan_propagating_
  max_f64`・`logsumexp_slice`／`axis_reduce_logsumexp`・
  `vector_norm_p_slice`／`axis_reduce_vector_norm_p`）
- `crates/backend-cpu/src/ops.rs`: `CpuBackendOps::logsumexp`／
  `vector_norm_p` の結線・`reduce_error_to_backend_error` の
  `EmptyReduction` 写像へ `"logsumexp"`／`"norm_p"` を追加・
  `InvalidOrder` の写像を追加
- `crates/autodiff/src/tape.rs`: `Op::LogSumExp`・`Op::PNorm` の
  variant と 3 箇所の網羅 match（`is_checkpoint_eligible`・
  `for_each_input`・`supports_create_graph`。いずれも `Op::Var`／
  `Op::VectorNorm` と同じ扱い）
- `crates/autodiff/src/eval.rs`: `nan_propagating_max_f64`（`pub
  (crate)`）・`logsumexp_along`・`vector_norm_p_along`
- `crates/autodiff/src/grad.rs`: `vjp` への 2 arm 追加・
  `logsumexp_vjp`・`pnorm_vjp`
- `crates/autodiff/src/reduce_ops.rs`（新規）: `prod`・`logsumexp`・
  `any`・`all`・`norm_p`（5 自由関数）・モジュール doc・単体テスト
  34 件（forward・エッジケース・エラー系・勾配の有限差分検算・
  `NaN`／`inf`／overflow 系）
- `crates/autodiff/src/lib.rs`: `pub mod reduce_ops;`・クレート doc
  へイシュー #2147 の要約を追記
- `crates/autodiff/tests/reduction_parity.rs`（新規）: NaiveOps 上の
  閉形式・`f64` 参照値との突合・bit 同一契約（`norm_p(1)` ≡
  `norm_l1`・`norm_p(2)` ≡ `norm_l2`）・中心差分による勾配検査・
  エラー系（13 件）
- `crates/facade/src/lib.rs`: `VarReduceOpsHoldDoctestGuard`（正の
  プローブ doctest。`VarMatrixOpsHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`:
  `reduce_ops_hold_doctest_globs_all_pub_modules`・
  `reduce_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_reduce_ops`・
  `workspace_declares_reduce_ops_fn_names_only_in_allowed_locations`
  （4 テスト。**期待集合は `matrix_ops` と異なり単一ファイルに閉じ
  ない**——着手前の再 grep で `logsumexp` という関数名が
  `tensor-core::backend_ops`（trait デフォルトメソッド）・
  `backend-cpu::{reduction, ops}`（CPU 実装）にも正規に存在すること
  が判明したため、期待集合を「`autodiff/src/reduce_ops.rs` に
  5 件（各演算 1 件） + `logsumexp` が上記 3 箇所に追加で 1 件ずつ」
  という allowlist へ調整した。`prod`・`any`・`all`・`norm_p`〈完全
  一致の識別子。`vector_norm_p` とは別名〉は `reduce_ops.rs` にのみ
  存在する）
- `crates/facade/tests/reduce_ops_backend_parity.rs`（新規）: CPU と
  NaiveOps の `any`／`all` forward・勾配ゼロ性の bit 完全一致、
  `prod`／`logsumexp`／`norm_p` forward／backward の REQ-2 統一複合
  判定（属性なし 4 件）＋CUDA／Metal の `#[ignore]`（未実測。8 件。
  `docs/perf/logs/reduce-ops-2147/README.md` 参照）
- `docs/compat-api-scope.md`: §1.2 へ追補（facade 経路 2 未適用のため
  保留と明記）
- `docs/README.md`: 本 doc・perf log README の索引行を追加

承認取得後の追随（本イシューでは未実施）: `Var::prod` 等の薄い委譲
メソッド追加、facade 保留ガード（`VarReduceOpsHoldDoctestGuard`・
対応する否定ガード 4 件）の撤去。
