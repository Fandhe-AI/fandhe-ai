# eigh・lstsq・pinv・matrix_rank・slogdet の設計判断記録

イシュー #2150（親 #2131）。`docs/autodiff-reduce-ops-decision.md`
（#2147）と同型の記録。

## §0 結論・facade 非公開

PyTorch `torch.linalg` 互換の 5 演算（`eigh`／`slogdet`／`pinv`／
`matrix_rank`／`lstsq`）を、**`fandhe_ai_autodiff` のうち facade が
再エクスポートしない自由関数モジュール `linalg_ops`**（`crates/
autodiff/src/linalg_ops.rs`）として実装した（`reduce_ops`〈#2147〉・
`matrix_ops`〈#2144〉と同じ判断枠組み）。`Var` に inherent の
`pub fn` は追加していない。

イシュー #2150 本文は「facade への 5 メソッドの `pub use` 再エクス
ポート（経路 2）」を承認事項として明示しているが、`facade` は `Var`
をそのまま再エクスポートしている（`crates/facade/src/lib.rs:184`）
ため、`Var` に inherent メソッドを 1 つ足すだけで facade の公開面が
広がる。このツリー（親 #2131）の先例に倣い、承認が取れるまで
**autodiff に新設したモジュール `linalg_ops` の自由関数 5 個**で
「5 メソッド」を満たす（承認後に追加する作業は `Var::eigh` 等の薄い
委譲メソッドと facade ガード〈`VarLinalgOpsHoldDoctestGuard`〉の
撤去のみ）。

多出力の戻り値型 `EighVars`／`SlogdetVars`（`linalg_ops.rs` 内）・
値型 `EighFactors`／`SlogdetFactors`（`tensor-core::backend_ops`）も
autodiff のクレートルート／facade へは再エクスポートしない
（`Var::qr` の `QrVars` は承認済み公開だが、本イシューの 5 演算は
未承認のため対称にしない）。

## §1 PyTorch 対応表・差分

| 演算 | PyTorch 相当 | 出力 | 微分 |
|---|---|---|---|
| `eigh` | `torch.linalg.eigh(UPLO='L')` | `eigenvalues: [n]`（昇順）・`eigenvectors: [n,n]` | 可（両出力） |
| `slogdet` | `torch.linalg.slogdet` | `sign: []`・`logabsdet: []` | `logabsdet` のみ可（`sign` は勾配ゼロ） |
| `pinv` | `torch.linalg.pinv` | `[n,m]` | 可 |
| `matrix_rank` | `torch.linalg.matrix_rank` | `[]`（非負整数を表す f32） | **勾配ゼロ**（非微分） |
| `lstsq` | `torch.linalg.lstsq` | `[n,k]`（最小ノルム解のみ） | 可（`a`・`b` 両方） |

PyTorch との差分:

- `eigh`・`slogdet`・`pinv`・`lstsq` は非有限入力（`NaN`／`Inf`）を
  `AutodiffError::InvalidArgument` として拒否する（PyTorch は伝播
  またはエラーにするが挙動が演算ごとに異なる。本実装は 5 演算とも
  統一した「入口で拒否」契約とする）。
- `matrix_rank` の出力は `f32`（PyTorch は `int64`）。既存の `any`／
  `all` の 0.0／1.0 マスクと同じ「f32 でブール／整数値を表す」前例に
  合わせる。
- `lstsq` は `X`（solution）のみを返す。PyTorch の `residuals`／
  `rank`／`singular_values`・driver 選択は返さない（§8 対象外）。
- `pinv`／`lstsq`／`matrix_rank` の `rcond` 既定値は PyTorch と同じ
  `max(m,n)・f32::EPSILON`。

## §2 各演算の設計

### §2.1 配置・Op 構造

5 演算とも専用 `Op` を持つ（合成は使わない）:

| 演算 | Op |
|---|---|
| `eigh` | `Op::EighValues { input, vectors }`／`Op::EighVectors { input, values }`（`QrQ`／`QrR` と同型の多出力設計） |
| `slogdet` | `Op::SlogdetSign { input }`／`Op::SlogdetLogAbsDet { input }` |
| `pinv` | `Op::Pinv { input, rcond }` |
| `matrix_rank` | `Op::MatrixRank { input, rcond }`（VJP は明示ゼロ。`Op::OneHot` の前例） |
| `lstsq` | `Op::Lstsq { a, b, rcond }` |

`rcond` は `Op` payload では **解決済みの `f32`**（`eval::linalg::
resolve_rcond` を `linalg_ops.rs` の入口で 1 回呼んで確定させた値）
として持つ。`None` 既定と明示指定を forward 後も区別する必要はない
（backward が同じ `resolve_rcond` を再度呼んでも決定的に同じ値になる
ため、演算の意味論としては payload に `Option<f32>` を持たせる必要が
ない。実装の単純化）。

### §2.2 判断 2: 合成を採らない理由（`svd_vjp` 無条件判定の実測根拠）

既存の `eval::linalg::svd_vjp`（イシュー #1621）は
`|s_j² − s_i²| < 1e-9` を無条件に `InvalidArgument` として拒否する。
この判定は、どのコタンジェントが `Some` でも、upstream が全ゼロでも
実行される（`crates/autodiff/src/eval/linalg.rs` の該当箇所参照）。

このため `svd → gt → sum` で組んだ `matrix_rank` は、rank 落ちの入力
（σ=0 が 2 個以上）や単位行列・直交行列（σ がすべて 1）で、真の勾配が
ゼロであるにもかかわらず backward が失敗する。`svd → where → matmul`
で組んだ `pinv` も、主な用途である rank 落ちの入力で失敗する
（σ が小さいスケールでは σ 同士が異なっていても絶対閾値 `1e-9` に
引っかかる。さらに `where(s>tol, 1/s, 0)` は、選ばれない側で
`reciprocal` の VJP（`−1/s²=−inf`）とゼロ upstream の積が `NaN` に
なる）。

以上の理由により、`pinv`・`matrix_rank`・`lstsq` は既存 `svd_vjp` を
経由せず、`eval::linalg` 内に新設した `pinv_mat_f64`（`rcond` 打ち切り
の薄い SVD。テープにノードを積まない `MatrixNorm{Nuc}` の前例と同型）
を使う専用 VJP を実装した。既存 `svd_vjp` の縮退判定が無条件に働く
問題自体の是正は §8 の対象外とする。

### §2.3 eigh

- 下三角のみを読み対称化してから、古典的巡回 Jacobi 法
  （`eval::linalg::eigh_jacobi`）を適用する。収束判定は
  `off(A)_F <= EIGH_JACOBI_EPS(=1e-14) · ‖A‖_F` の相対判定のみ
  （絶対下限を持たない。`jacobi_svd_tall` と同じ設計判断——Frobenius
  ノルムは直交相似変換で不変なため、巡回中一定の `‖A‖_F` を基準に
  使える）。スイープ上限 60（`svd` と同じ）。
- 固有値は昇順（同値は `sort_by` の安定性で安定順）。固有ベクトル
  各列は `normalize_max_abs_sign`（`svd` と共有）で符号正規化する。
- VJP は PyTorch `linalg_eig_backward`（hermitian 分岐）と同形:
  `gA = V (diag(gL) + skew(Vᵀ gV) ⊘ E) Vᵀ`（`skew(X)=(X−Xᵀ)/2`・
  `E_ij=λ_j−λ_i`）。`gV` が `Some` のときのみ固有値縮退
  （`EIGH_DEGENERATE_RTOL=1e-9` の相対判定）を検査する——
  `EighValues` のみが損失に届く場合（`gV` が全て 0）は縮退していても
  成功させる（§2.2 と同じ「upstream がゼロなら無条件判定をしない」
  設計方針）。

### §2.4 slogdet

- 部分ピボット LU（既存 `lu_decompose` を再利用）。
  `logabsdet = Σ ln|u_ii|`（対角の絶対値の log を `f64` で累積してから
  最後に 1 回だけ `f32` へ変換）。`ln|det|` の合成（`det` を経由する
  素朴な実装）ではないため、`lu_diag_product` が回避しているオーバー
  フロー／アンダーフローが `slogdet` に再発しない。
- 特異（ピボット厳密 0）は forward で `(0, -inf)`（PyTorch と同じ。
  エラーにしない）。backward（`logabsdet` 側のみ）はその場合
  `InvalidArgument`（`det_vjp` と同じ fail-closed）。
- `sign` の VJP は明示ゼロ（`Op::OneHot` の前例と同型）。

### §2.5 pinv・matrix_rank・lstsq（薄い SVD 共通基盤）

`eval::linalg::pinv_mat_f64(a, rcond)` が共通の内部実装: 既存
`svd(a)`（reduced SVD、`k=min(m,n)`）を呼び、`rcond·σ_max` を上回る
特異値の個数を有効ランク `r` として `A⁺ = V diag(1/σ_i) Uᵀ`（`i<r`）
を `f64` の行列として返す。`pinv`／`matrix_rank`／`lstsq` はいずれも
この 1 関数を基盤にする。

- `pinv` の VJP（2026-09-25 是正・PR #2268 codex-review〈P1〉指摘）は、
  当初計画していた Golub–Pereyra の微分式（PyTorch `pinv_backward`
  相当。`gA = −Pᵀ G Pᵀ + (I_m − A P) Gᵀ P Pᵀ + Pᵀ P Gᵀ (I_n − P A)`。
  `P=A⁺`・`G` は upstream）を `rcond` 打ち切り後の `P` へそのまま
  適用していたが、この式は `A A⁺ A = A` 等の Moore–Penrose の 4 条件
  が「真の `A`」に対して成立することを前提に導出されており、`rcond`
  打ち切りで**非零の特異値を捨てた**場合（`rank < min(m,n)` かつ
  捨てた特異値が非零。例 `A=diag(2,1)`・`rcond=0.75`）は前提が崩れて
  有限差分と乖離する不具合があった（`A` が正方かつ全特異値が残る
  ケースは偶然一致するため、フルランクの回帰テストだけでは検出
  できなかった）。是正後は `P` を `svd(a)` の `rank` 個の特異値
  三つ組 `(U_r, S_r, V_r)` のみへ依存する関数として扱い、`P` の
  コタンジェント `G` を三つ組のコタンジェント `(dU, dS, dV)`
  （`rank` 以上の列は厳密 `0`）へ変換したうえで、`svd_vjp`
  （`Op::SvdU`／`Op::SvdS`／`Op::SvdVh` 用の Townsend 2016 汎用式）と
  同型の式（`eval::linalg::svd_vjp_rank_limited_f64`）を適用する。
  打ち切りの有無に関わらず常に正しく、かつ `svd_vjp` の結合順序
  （`X−U(UᵀX)` 型）をそのまま踏襲するため、中間行列は `m×k`・
  `n×k`・`k×k`・最終出力 `m×n`（`k=min(m,n)`）以下に収まり
  `m×m`／`n×n` を実体化しない（§8 に記録していた最適化は本是正で
  同時に達成された）。`svd_vjp` 本体（`Op::SvdU` 等の汎用契約。全域
  無条件の近接／重複特異値判定）は変更せず、`svd_vjp_rank_limited_f64`
  は打ち切りで捨てた特異値どうしのペア（寄与が定義上厳密ゼロ）に
  限り判定をスキップする専用の姉妹関数として新設した（§2.2 の
  「既存 `svd_vjp` を経由しない」判断は維持しつつ、その式の骨格は
  再利用する形）。**縮退判定の閾値（2026-09-25 是正・PR #2268
  codex-review〈P1〉指摘）**: `svd_vjp` は `|σ_j²−σ_i²| < 1e-9` を
  絶対閾値で判定するが、`svd_vjp_rank_limited_f64` はこれを流用せず
  `|σ_j²−σ_i²| < 1e-9・(σ_i²+σ_j²)`（`σ_i²+σ_j²` に対する相対閾値）で
  判定する。§2.2 で述べた「σ が小さいスケールでは絶対閾値 `1e-9` に
  引っかかる」問題（`svd → where → matmul` の合成案を退けた理由の
  ひとつ）が、専用実装である `svd_vjp_rank_limited_f64` 自身にも
  絶対閾値のまま残っていた（例 `σ=[2e-6, 1e-6]` はフルランクで
  縮退していないが `σ_j²−σ_i²=3e-12` が `1e-9` を下回り誤って
  `InvalidArgument` になっていた）ため、この関数は新規導入 API で
  `svd_vjp` の上流互換を引き継ぐ必要がないことを踏まえ相対閾値へ
  変更した（`svd_vjp` 本体は変更しない）。**重複する非零特異値
  （2026-09-25 追加是正・PR #2268 codex-review〈P1〉指摘）**:
  相対閾値化後も `svd_vjp_rank_limited_f64` は「有効ランク内
  （`i<rank && j<rank`）の特異値ペアが縮退（分母 `σ_j²−σ_i²≈0`）
  している」場合を無条件に `InvalidArgument` とする設計のままだった
  ため、`A=I₂`（`σ=[1,1]`。打ち切りなし・`rank==k==2`）のような
  「重複するが非零」の特異値では、`pinv(A)=A⁻¹` の勾配が数学的には
  常に well-defined（`dP=-A⁻¹dA A⁻¹`）であるにもかかわらず `pinv`・
  `lstsq`（内部で `pinv_vjp` を再利用）の逆伝播が無条件に失敗して
  いた。是正として、**打ち切りで非零特異値を 1 つも捨てていない場合**
  （`s[rank..k]` が全て厳密 `0.0`。`rank==k`〈打ち切りなし〉に加え、
  ランク落ち行列で切り捨てた特異値が真に `0.0` のケースも含む）に限り、
  `eval::linalg::pinv_vjp_direct_f64` が Golub–Pereyra の閉形式
  （本節冒頭の `gA` 式）を `(U_r, S_r, V_r)` の低ランク因子だけで
  （`A P = U_r U_rᵀ`・`P A = V_r V_rᵀ` の恒等式で `m×m`／`n×n` を一切
  実体化せず）直接評価する専用高速路を新設した。この場合 `P` は
  打ち切りの影響を受けない `A` 自身の真の擬似逆行列であり、Golub–
  Pereyra 式が要求する「`A A⁺ A = A` 等の 4 条件が `A` に対して成立
  する」前提が常に満たされるため、個々の `U`／`V` 列を区別する
  `svd_vjp_rank_limited_f64` の分母縮退判定を経由せずに済む。**打ち
  切りで非零特異値を実際に捨てた場合**（`rank < k` かつ `s[rank..k]`
  に非零が残る）は従来どおり `svd_vjp_rank_limited_f64` 経由の式を
  使う（この場合 `P` は `A` ではなく低ランク近似 `A_r` の擬似逆行列
  であり、Golub–Pereyra 式を `A` に直接適用できないため）。**残る
  スコープ外**: 「非零特異値を打ち切り、かつ打ち切り境界より内側で
  重複特異値が生じる」の二重発生ケース（例 `A=diag(1,1,0.5)`・
  `rcond` で `σ=0.5` のみ切り捨て、残る `σ=[1,1]` が縮退）は、`P` が
  `A_r`（≠`A`）の擬似逆行列となり Golub–Pereyra 式を直接使えず、かつ
  縮退もしているため、引き続き `svd_vjp_rank_limited_f64` 経由で
  `InvalidArgument` になる。この解消は §8 へスコープ外として申し
  送る。回帰テストは `crates/autodiff/src/linalg_ops.rs` の
  `pinv_gradient_matches_finite_difference_duplicate_nonzero_
  singular_values`・`lstsq_gradient_matches_finite_difference_
  duplicate_nonzero_singular_values`（grad-check）。
- `matrix_rank` は非微分（VJP 明示ゼロ）。
- `lstsq` は `X = V S⁻¹ (Uᵀ B)`（有効ランクまでの和。`A⁺` を陽に作ら
  ない）。VJP は `gB = A⁺ᵀ gX`、`gA = pinv_vjp(a, rcond, G=gX Bᵀ)`
  （PyTorch `linalg_lstsq_backward` 相当。`pinv_vjp` を再利用するため
  追加コストは小さい）。

## §3 数値契約

- 内部精度は `eval::linalg`（`f64` アキュムレータ・出力時に 1 回だけ
  `f32` downcast）と同一（`.claude/rules/coding-rust.md`）。
- `rcond`（`pinv`／`matrix_rank`／`lstsq`）・固有値縮退判定閾値
  （`eigh`）は**アルゴリズムの引数・内部定数であり REQ-2 の
  tolerance ではない**。`rcond` は有限かつ非負を要求し、`None` は
  `max(m,n)・f32::EPSILON`。
- 非有限入力（`NaN`／`Inf`）は `eigh`・`slogdet`・`pinv`・`lstsq` の
  入口（`linalg_ops.rs::require_finite`）で `AutodiffError::
  InvalidArgument` として拒否する。
- 確保前のバイト数上限検査（`ensure_alloc_fits_f32`。`reduce_ops` と
  同じ規律）は全 5 入口の冒頭・あらゆる分岐・実体化よりも前に行う。

## §4 VJP 式と出典

§2.3〜§2.5 参照。PyTorch の対応する `*_backward`（`FunctionsManual.cpp`
の `linalg_eig_backward`〈hermitian 分岐〉・`pinv_backward`・
`linalg_lstsq_backward`）と数式の形を照合し、grad-check（有限差分
`H=1e-3`・`REL_TOL/ABS_TOL≈1e-2`。既存 `eval::linalg` の grad-check
ヘルパと同じ定数を流用）で実測検証した
（`crates/autodiff/src/linalg_ops.rs` の `#[cfg(test)]` 内）。

## §5 テスト構成

- `crates/tensor-core/src/backend_ops.rs`: 値型 2 つ・既定
  `Unsupported` メソッド 5 本。
- `crates/autodiff/src/eval/linalg.rs`: forward・VJP の単体実装
  （`#[cfg(test)]` に既存の inv／solve 系と統合。追加の独立テストは
  `linalg_ops.rs` 側の統合テストで代替し重複を避けた）。
- `crates/autodiff/src/linalg_ops.rs`: 27 件の単体テスト（forward・
  VJP grad-check・空行列・非正方・非有限入力・rcond 検査）。
- `crates/backend-cpu/src/linalg.rs`・`ops.rs`: CPU 本番実装。
- `crates/backend-cpu/tests/linalg_parity.rs`: `BackendOps` trait 経由
  の到達性・既知値・bit 決定性（16 件追加）。
- `crates/backend-cpu/tests/backend_ops_dispatch.rs`: CUDA／Metal の
  5 メソッドが `Unsupported` を返すことの確認。
- `crates/autodiff/tests/linalg_parity.rs`（新規）: NaiveOps 経由の
  end-to-end（11 件）。
- `crates/facade/tests/linalg_ops_backend_parity.rs`（新規）: CPU vs
  NaiveOps の forward／backward parity（13 件）＋ CUDA／Metal 実機
  `#[ignore]`（4 件。§7 参照）。
- `crates/facade/src/lib.rs`・`tests/api_surface.rs`:
  `VarLinalgOpsHoldDoctestGuard`（正のプローブ doctest）＋ 4 件の
  ソース走査ガード（多層防御。§6 参照）。

## §6 承認事項・多層防御

承認待ち（未実施）:

- facade への 5 演算の公開（`pub use fandhe_ai_autodiff::linalg_ops;`
  等の経路 2）
- `Var::eigh`／`Var::slogdet`／`Var::pinv`／`Var::matrix_rank`／
  `Var::lstsq` の委譲メソッド追加
- 上記に伴う `VarLinalgOpsHoldDoctestGuard`・`api_surface.rs` の
  4 ガードテストの撤去

多層防御（`reduce_ops`〈#2147〉と同型）: ①facade 非公開（自由関数
モジュール）②`VarLinalgOpsHoldDoctestGuard`（正のプローブ doctest。
facade がどの経路で公開しても名前衝突・シグネチャ不一致でコンパイル
失敗する）③`linalg_ops_hold_doctest_globs_all_pub_modules`（doctest の
glob import 集合と実際の `pub mod` 宣言のドリフト検査）④`linalg_ops_
hold_doctest_probe_body_matches_fixed_contract`（doctest 本文の改変・
弱体化を機械検出）⑤`facade_does_not_reexport_or_declare_linalg_ops`
（facade src の `pub use`／`fn` 宣言を直接走査）⑥`workspace_declares_
linalg_ops_fn_names_only_in_allowed_locations`（workspace 全体の
定義元インベントリを固定）。

## §7 実装記録

CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機への到達手段が本
実装エージェント実行環境にないため、`linalg_ops_backend_parity.rs`
の `#[ignore]` 4 件（`eigh`／`pinv` forward × CUDA／Metal）は未実測
のまま `docs/perf/logs/linalg-ops-2150/README.md` へ申し送る。CPU 版
（属性なし・13 件）はいずれも green。

## §8 スコープ外

- GPU 専用カーネル（CUDA／Metal の `linalg_*` 実装。別イシュー）
- バッチ次元 `[..., n, n]`
- `eigh` の `UPLO='U'`
- `lstsq` の `residuals`／`rank`／`singular_values` 返却・driver 選択
- `pinv`／`matrix_rank` の `atol` 引数・`hermitian=True`
- `matrix_rank` の int 出力
- 既存 `svd_vjp`（`Op::SvdU` 等の汎用契約）の縮退判定が無条件に働く問題（upstream がゼロでも
  発火する）の是正（§2.2 参照。本 PR では変更しない）
- 「非零特異値を打ち切り、かつ打ち切り境界より内側で特異値が重複する」
  の二重発生ケースにおける `pinv`／`lstsq` の逆伝播（§2.5「重複する
  非零特異値」参照。2026-09-25 追加是正で `A=I₂` 等の単純な重複ケース
  〈打ち切りなし〉は解消済みだが、この二重発生ケースは
  `svd_vjp_rank_limited_f64` 経由のまま `InvalidArgument` になる）
- CUDA／Metal 実機計測（§7 参照）
