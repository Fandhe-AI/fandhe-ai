# tril・triu・diag・trace・outer・dot の設計判断記録

イシュー #2144（親 #2131）。`docs/autodiff-rearrange-ops-decision.md`
と同型の記録。

## §0 結論

PyTorch 互換の形状・行列演算 6 種（`tril`／`triu`／`diag`／`trace`／
`outer`／`dot`）を、**`fandhe_ai_autodiff` のうち facade が再
エクスポートしない自由関数モジュール `matrix_ops`**（`crates/
autodiff/src/matrix_ops.rs`）として実装した（案 C。§2.1 参照。
`rearrange_ops`〈#2143〉・`bool_ops`〈#2141〉と同じ判断枠組み）。
`Var` に inherent の `pub fn` は追加していない。新規 `Op`・
`BackendOps` メソッド・VJP・tape ノードは追加していない——いずれも
既存の `Var::masked_fill`（`Op::MaskedFill`）・`Var::gather`
（`Op::Gather`）・`Var::narrow`（`Op::Narrow`）・`Var::broadcast_to`
（`Op::BroadcastTo`）・`Var::transpose`（`Op::Transpose`）・
`Var::pad`（`Op::Pad`）・`Var::squeeze`（`Var::reshape` へ委譲）・
`Var::mul`（`Op::Mul`）・`Var::sum`（`Op::Sum`）の合成のみで構成した。
facade 公開（`Var` への委譲メソッド追加）は承認待ちのまま対象外とし、
`crates/facade/src/lib.rs::VarMatrixOpsHoldDoctestGuard`（正の
プローブ doctest）と `crates/facade/tests/api_surface.rs` のソース走査・
workspace インベントリ（4 テスト）で多層固定している。

## §1 背景

イシュー #2144・親 #2131 にはコメントが 0 件で、facade 公開の承認記録
はない（2026-09-24 時点。着手前に `gh issue view 2144/2131` で確認
済み）。親 #2131 はこのツリーでの facade 公開面の拡張を「設計判断
記録 → 承認 → 実装」の 2 段階と定めているため、本実装は内部クレート
限定に倒す。

## §2 設計判断

### §2.1 API 配置

- **案 A（`Var` の inherent メソッド）**: facade へ即座に到達する
  （`crates/facade/src/lib.rs` は `Var` を再エクスポートしている
  ため）。承認が無いため不採用
- **案 C（自由関数）**: 採用。承認後の撤去・委譲が単純
- モジュール名は `shape_ops` にしない（`rearrange_ops`〈#2143〉と
  同じ理由: onnx-interop の private `mod shape_ops`・既存テスト
  `crates/facade/tests/shape_ops_backend_parity.rs`〈#1597〉との
  混同回避）。`matrix_ops` という名前の衝突は着手前の grep で
  見つからなかった

### §2.2 `crates/backend-cpu/src/ops.rs` は変更しない

Issue の対象範囲には「`backend-cpu/src/ops.rs` に CPU 参照実装」が
挙がっているが、これは受け入れ条件「既存 `Var` の合成のみ」と両立
しない（`BackendOps` への新メソッド追加は crates.io 公開済みの
`tensor-core` 公開 trait を広げることにもなる）。合成に使う
`masked_fill`・`gather`・`narrow`・`broadcast_to`・`pad`・`mul`・
`sum` は CPU・CUDA・Metal すべてに実装済みのため、「`Unsupported`
フォールバックで到達可能」という契約は追加作業なしで満たされる。
「CPU 参照実装との突き合わせ」は、モジュール内 `#[cfg(test)]` の
解析的期待値の直接算出・有限差分検算で満たした。

**`crates/backend-*/**`・`crates/tensor-core/**` は変更していない。**

### §2.3 各演算の合成方式

| 演算 | 合成 |
|---|---|
| `tril`／`triu` | 末尾 2 軸 `[m, n]` の `Tensor<bool>` マスクをホストで作る（`tril` は `j - i > k` を真、`triu` は `j - i < k` を真）→ `x.masked_fill(&mask, 0.0)`（マスクは先頭バッチ軸へ自動 broadcast） |
| `diag`（2-D → 1-D） | 抽出長 `L` を先に計算 → `x.narrow(0, r0, L)`（view）→ `gather(1, idx[L,1])`（`idx[i] = i + c0`）→ `squeeze(Some(1))` |
| `diag`（1-D → 2-D） | `x.broadcast_to([n, n])`（`out[i][j] = x[j]`）→ `masked_fill(i != j, 0.0)` → `k != 0` のときだけ `pad`（`k > 0` は行 `(0,k)`・列 `(k,0)`、`k < 0` は行 `(|k|,0)`・列 `(0,|k|)`） |
| `trace` | `diag(x, 0)` → `sum(None)` |
| `outer` | `a.broadcast_to([1, n]).transpose(0, 1)`（`[n, 1]` の view）`.mul(&b.broadcast_to([1, m]))` |
| `dot` | 明示的な rank・長さ検査 → `a.mul(b).sum(None)` |

採用しなかった案:

- **0/1 マスクとの `mul`**: `0 * inf` や `0 * NaN` が `NaN` になり
  `masked_fill` の「マスク位置は常に定数」契約と合わない
- **`unsqueeze`／`reshape` での rank 拡張**: 内部で `reshape` を呼び
  非 contiguous 入力で `NonContiguousReshape` になる。`outer` と
  1-D → 2-D の `diag` は `broadcast_to`／`transpose`（いずれも view）
  で組み立てた
- **`diag`（2-D → 1-D）で `narrow` を挟む理由**: `gather_out_shape`
  は `dim` 以外の軸で index と入力の shape が厳密に一致することを
  要求するため、列だけを選ぶ `gather` の前に行方向を `narrow` で
  絞る必要がある
- **`einsum`／`matmul` 経由**: GEMM を経由すると TF32 opt-in や FMA
  契約の影響を受ける。`dot`／`outer` は `mul`＋`sum`／`transpose` に
  留めた

### §2.4 境界検査（REQ-8・`.claude/rules/security.md` A03）

- rank 検査: `tril`／`triu` は rank 2 以上（`RankMismatch { expected:
  2, .. }`。`batched_matmul_plan` と同じ「`expected: 2` は『2 以上』
  を表す」慣習）、`diag` は rank 1 か 2（`InvalidArgument`。
  `RankMismatch` は単一の `expected` しか表現できないため）、`trace`
  は rank 2 限定、`outer`／`dot` は両入力とも rank 1（`RankMismatch
  { expected: 1, .. }`）
- `dot` の長さ不一致は `mul` の暗黙 broadcast に任せず明示的に
  `ShapeError::ShapeMismatch` で拒否する
- `diagonal: isize` は `unsigned_abs()`（`isize::MIN` 対策）・
  `saturating_sub`・`checked_add`（`pad` 内部）で扱い、生の
  `usize ± isize` は使わない
- 早期リターン（ノードを積まない）: `tril` で `diagonal >= n - 1`、
  `triu` で `diagonal <= -(m - 1)`
- マスク・添字の確保前サイズ検査は `crates/autodiff/src/
  rearrange_ops.rs` の `checked_axis_len_as_i32`・
  `checked_index_alloc_len` を `pub(crate)` へ昇格して共有する
  （可視性のみの変更・両ヘルパーの挙動は不変）。`rearrange_ops`
  同様、`broadcast_to` の stride-0 view で `m`／`n` が実体を伴わず
  巨大になりうるため、`Vec<bool>`／`Vec<i32>` の確保前に必ず通す
- 本番経路で `unwrap()`／`expect()` は使わない

### §2.5 テストの配置

- 単体テスト・勾配テストは `crates/autodiff/src/matrix_ops.rs` 内の
  `#[cfg(test)] mod tests` に置いた（`rearrange_ops.rs`／`bool_ops.rs`
  と同じ配置）
- バックエンド間 parity は `crates/facade/tests/
  matrix_ops_backend_parity.rs`（`fandhe_ai_autodiff::matrix_ops::*`
  を直接 use。`autodiff` の `[dependencies]` は tensor-core のみで
  `fandhe_ai::tape()`／`Device::Cuda` に到達できないため、`crates/
  autodiff/tests/` には置けない）

## §3 数値契約

- `tril`／`triu`／`diag`（両方向）: forward はコピーまたは定数 0 の
  埋め込みのみ（算術を含まない）のため 3 バックエンド間で構造的に
  bit 完全一致する（`NaN` の payload も保存される）。backward は
  `tril`／`triu`／`diag`（2-D→1-D）が `Op::MaskedFill`／`Op::Gather`
  の VJP（fill 位置はゼロ、それ以外は素通し・scatter で寄与は各
  1 つ）を経由し、`diag`（1-D→2-D）はこれに加え `Op::Pad`（非
  パディング領域は素通し）・`Op::BroadcastTo` の VJP（軸方向の
  `reduce_to_shape` 縮約）も経由する。`BroadcastTo` の VJP は `outer`
  backward（下記・REQ-2 統一複合判定が必要）と同じ縮約だが、`diag`
  （1-D→2-D）側は縮約対象の行の非ゼロ要素が高々 1 つ（対角以外は
  `masked_fill` で 0 済み）で残りは厳密 `+0.0`（加算順序に依存しない
  exact 加算）のため bit 完全一致のまま成立する
- `trace`: `diag(x, 0)`（bit 一致）→ `sum(None)`。`sum` の縮約順序は
  バックエンドで異なりうるため REQ-2 の統一複合判定（相対誤差 1e-3
  未満 または 絶対誤差 1e-5 未満）で比較する
- `outer`: 乗算 1 回のみのため forward は bit 一致する。backward は
  `broadcast_to` の VJP（軸方向の `reduce_to_shape` 縮約）を経由する
  ため REQ-2 の統一複合判定で比較する
- `dot`: `mul` → `sum(None)`。`mul` 自体は bit 一致するが `sum` の
  縮約順序差により forward 全体は REQ-2 の統一複合判定で比較する。
  backward（`g * other`。乗算 1 回のみ）は bit 一致する

## §4 PyTorch との差異

- `trace`・`diag` の 2-D 入力は rank 2 限定（`torch.diagonal` の
  rank 3 以上・バッチ trace は非対応）
- 軸番号を持つ引数はない（`tril`／`triu`／`diag` の `diagonal` の
  みが可変パラメータ）

## §5 スコープ外

- facade 公開（`Var::tril` 等の委譲メソッド化と保留ガードの撤去）
- `trace`／`diag` の rank 3 以上（バッチ trace・`torch.diagonal`
  相当）
- GPU 専用カーネル（現状は `masked_fill`／`gather`／`pad` の既存
  カーネル・ホストフォールバックのみで到達）

## §6 承認事項（未承認として列挙）

1. facade 公開（上記スコープ外 1 と同じ）
2. `trace`／`diag` の rank 3 以上対応
3. GPU 専用カーネル

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/shape-matrix-ops-2144/README.md` へ測定
コマンド案・期待結果を申し送る。

## §8 実装記録（イシュー #2144・2026-09-24）

- `crates/autodiff/src/matrix_ops.rs`（新規）: `tril`／`triu`／
  `diag`／`trace`／`outer`／`dot`・モジュール doc（役割・facade
  非公開の理由・数値契約表・PyTorch との差分・REQ-8）・単体テスト
  58 件（forward・エッジケース・非 contiguous 入力・エラー系・勾配・
  有限差分検算・`NaN`／`inf` payload 保存・確保上限。§9 の是正時点の
  実数。着手時点の見積り「45 件」から乖離していたため §9 で実数へ
  更新した）
- `crates/autodiff/src/rearrange_ops.rs`: `checked_axis_len_as_i32`・
  `checked_index_alloc_len` を `pub(crate)` へ昇格（挙動は不変）
- `crates/autodiff/src/lib.rs`: `pub mod matrix_ops;` を追加
  （アルファベット順維持）・クレート doc にイシュー #2144 の要約を
  追記
- `crates/facade/src/lib.rs`: `VarMatrixOpsHoldDoctestGuard`（正の
  プローブ doctest。`VarRearrangeOpsHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`:
  `matrix_ops_hold_doctest_globs_all_pub_modules`・
  `matrix_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_matrix_ops`・
  `workspace_declares_matrix_ops_fn_names_only_in_autodiff_
  matrix_ops`（4 テスト。着手前の再 grep で他クレートとの名前衝突は
  見つからなかったため期待集合は `crates/autodiff/src/matrix_ops.rs`
  各 1 件のみ）
- `crates/facade/tests/matrix_ops_backend_parity.rs`（新規）: CPU と
  NaiveOps のコピー系 forward bit 一致・`NaN`／`inf` payload 保存・
  縮約系 forward の REQ-2 統一複合判定・bit 完全一致 backward・
  `outer` backward の REQ-2 統一複合判定（属性なし 5 件）＋CUDA／
  Metal の `#[ignore]`（未実測。8 件。`docs/perf/logs/
  shape-matrix-ops-2144/README.md` 参照）
- `docs/compat-api-scope.md`: §1.2 へ追補
- `docs/README.md`: 本 doc・perf log README の索引行を追加

承認取得後の追随（本イシューでは未実施）: `Var::tril` 等の薄い
委譲メソッド追加、facade 保留ガード（`VarMatrixOpsHoldDoctestGuard`・
対応する否定ガード 4 件）の撤去。

## §9 網羅契約の是正（イシュー #2144・PR #2257 codex-review 指摘・2026-09-25）

`crates/facade/tests/matrix_ops_backend_parity.rs` 冒頭が謳う「`diag`
両方向を含む各演算の forward/backward と CPU・CUDA・Metal の主要セル
を埋める」契約に対し、実際は次のセルが欠落していた（§8 時点）:

- CUDA／Metal のコピー系 forward テストが `diag` の 2-D→1-D のみを
  実行し、1-D→2-D（`pad` を通る経路）を検証していなかった
- CUDA／Metal backward テストが `trace` のみを「代表」として検証し、
  別 VJP 経路を持つ `tril`／`triu`／`diag`（両方向）／`dot` を省略
  していた
- CPU backward テストでも `diag` は 2-D→1-D のみだった

`diag` の 1-D→2-D（`broadcast_to`／`masked_fill`／`pad` の合成）と
2-D→1-D（`narrow`／`gather`／`squeeze` の合成）は別の VJP 経路を持つ
ため代表検証では代替できない。契約文は縮めず、不足セルを追加する
方針で是正した:

- CPU backward（`cpu_bit_exact_backward_matches_naive_reference`）に
  `diag`（1-D→2-D）を追加
- CUDA／Metal のコピー系 forward テストに `diag`（1-D→2-D）を追加
- CUDA／Metal backward テストを `trace` 単独代表から `tril`／
  `triu`／`diag`（両方向）／`trace`／`dot` の個別検証へ拡張（CPU 側
  `cpu_bit_exact_backward_matches_naive_reference` と同型の 6 ブロック
  構成）

併せて本 doc §8 の単体テスト件数「45 件」が実数（58 件）と乖離して
いたため実数へ更新した（同種の宣言と実体の食い違いの横展開確認）。

## §10 diagonal 分岐セルの追加是正（同 PR #2257・コーディネーター指示・2026-09-25）

§9 の是正時点でも、`tril`／`triu`／`diag`（両方向）は diagonal 値に
よってマスク・パディング・抽出の境界位置が変わるにもかかわらず、
各演算 1 diagonal 値（`tril`＝0・`triu`＝0 と 1・`diag` 2-D→1-D＝0・
`diag` 1-D→2-D＝1 のみ）でしか検証していなかった。§9 の論拠「経路が
別なら代表 1 本では代わりにならない」は diagonal の分岐にも同型で
適用されるため（`Op` 経路は同じでも境界位置が異なれば別セル）、次を
追加した:

- `DIAGONALS = [-1, 0, 1]`（負・0・正。`f32_fixture_3x3`〈`m = n = 3`〉
  に対して `tril`／`triu` の早期リターン分岐〈`diagonal >= n - 1`／
  `diagonal <= -(m - 1)`、いずれも `|diagonal| = 2` で発生〉を踏まない
  範囲）を、コピー系 forward・bit 完全一致 backward の全テスト
  （CPU・CUDA・Metal）で走査するループへ変更
- `diag`（2-D→1-D）は抽出長 `L`（`diag_2d_to_1d_extract_len`）、
  `diag`（1-D→2-D）は出力形状 `N = n + |k|`（`sequential_weight`）が
  diagonal に依存するため、backward の乗算用重み `Tensor` を diagonal
  ごとに動的生成するヘルパーを追加した
- 既存テスト関数のブロックにループを追加する形に留め、新規
  `#[test]` 関数は追加していない（コーディネーター指示）

これにより `tril`（正・負）・`triu`（負）・`diag`（1-D→2-D）の
`k<0`／`k==0` 分岐・`diag`（2-D→1-D）の `k≠0`（正・負）という、前回
是正では「同じ `Op` 経路だから」という理由でスコープ外とした diagonal
分岐セルを全て埋めた。tolerance・REQ-2 判定は変更していない。

## §11 diagonal の境界値・範囲外の走査（PR #2257 のフォローアップ・ユーザー承認
2026-09-25）

§10 の `DIAGONALS = [-1, 0, 1]` は正方形状 `f32_fixture_3x3`
（`m = n = 3`）に対する内部値の走査に留まり、`tril`／`triu` の早期
リターン境界・全ゼロ化境界そのもの（`|diagonal|` がちょうど `n - 1`
／`-(m - 1)` に一致する値、およびそれを跨いで真に範囲外となる値）と
`diag`（両方向）の抽出長 `L = 1`／`L = 0` の境界・非正方形状
（行 < 列・行 > 列）は未検証だった。本節で以下を追加した。

- `tril_triu_diagonals(m, n)`（`crates/facade/tests/
  matrix_ops_backend_parity.rs`）: 形状 `[m, n]` ごとに
  `{-1, 0, 1, -2, 2, -(m-1), -m, -m-1, -(m-1)+1, n-1, n, n+1, n-2}`
  を重複除去・昇順ソートして生成する。
  - `k = n-1／n／n+1`: `tril` の早期リターン境界（`diagonal >= n-1`
    で `build_tril_triu_mask` を経由せず `x` をそのまま返す。ノードを
    積まない）。`triu` 側ではこの範囲はマスク経路を通り全要素が
    ゼロ化される（`k >= n` で mask 全 true）。
  - `k = n-2`: `tril` がマスク経路を通る最後の値（早期リターン境界の
    直前。1 要素のみゼロ化）。
  - `k = -(m-1)／-m／-m-1`: `triu` の早期リターン境界。`tril` 側では
    マスク経路で全ゼロ化される。
  - `k = -(m-1)+1`: `triu` がマスク経路を通る最後の値（1 要素のみ
    ゼロ化）。
  - `diag`（2-D→1-D）: `k = n-1` または `-(m-1)` で抽出長 `L = 1`、
    `k >= n` または `k <= -m` で `L = 0`（範囲外）。`L = 0` は
    `crates/autodiff/src/matrix_ops.rs::diag_2d_to_1d` の
    `narrow(0, 0, 0)` → `gather(1, idx[0,1])` → `squeeze` 経路により
    空テンソル `[0]` へ収束し、**エラーにはならない**（`torch.diag`
    と異なり範囲外オフセットを許容する既存仕様どおり）。
  - **「マスク全 false（＝全要素を残す）」は `build_tril_triu_mask`
    のマスク経路自体には現れない**: この条件は早期リターン分岐の
    条件と一致し、マスクを構築する前に `x` がそのまま返るため
    （早期リターンが優先的に成立する）。
  - 走査する形状は正方形 `3×3`・非正方形状〈行 < 列〉`3×5`・
    非正方形状〈行 > 列〉`5×3` の 3 種（`TRIL_TRIU_SHAPES`）。
- `diag_1d_diagonals(n)`: `diag`（1-D→2-D）は早期リターン分岐を持たず
  `N = n + |k|` へ `pad` するだけのため、pad 幅のバリエーション
  `{-1, 0, 1, -2, 2, -n, n}` を走査する（`k == 0` は pad しない分岐・
  `k > 0`／`k < 0` は `pad` 引数〈行・列の順序〉が異なる分岐）。
  入力ベクタ長は通常長 `3` と境界値の長さ `1`（`DIAG_1D_LENGTHS`）の
  2 種。
- `diag`（2-D→1-D）backward の `L = 0` セルも、他セルと同じく勾配が
  `Some` で記録されることを契約とする（実測: CPU・NaiveOps ともに
  形状 `[m, n]` の全ゼロ勾配を `Some` で返す）。テストは
  `match (dx_cpu, dx_other) { (Some, Some) => 形状・bit 比較, (None,
  None) => panic, _ => panic }` で判定し、`None` は双方一致していても
  失敗とする（CPU・NaiveOps の比較セルに加え `#[ignore]` の CUDA／
  Metal 実機セルも同じ判定）。CPU・NaiveOps セルでは `L = 0` のとき
  勾配が形状 `[m, n]`・全要素ゼロ（符号は問わない）であることも
  明示検証する。
- forward・backward いずれのループも bit 列比較に加えて `shape()` の
  一致を明示的に検証する（`f32_bits` は連続化した要素列のみを比較し
  形状差を検出しないため、空テンソル・早期リターンの各セルで形状の
  取り違えが素通りしないようにする）。
- 既存テスト関数のブロックにループ・形状バリエーションを追加する形に
  留め、新規 `#[test]` 関数は追加していない（コーディネーター指示）。

CPU（`cargo test -p fandhe-ai --test matrix_ops_backend_parity`）は
全て green（`L = 0` セルを含む）。tolerance・REQ-2 判定・実装
（`crates/autodiff/src/matrix_ops.rs`）は変更していない。CUDA／Metal
実機セルは引き続き `#[ignore]` のまま Mac／DGX Spark GB10 実機
セッションへ申し送る（`docs/perf/logs/shape-matrix-ops-2144/
README.md`）。
