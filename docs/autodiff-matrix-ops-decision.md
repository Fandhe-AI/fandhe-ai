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
  `Op::MaskedFill`／`Op::Gather` の VJP（fill 位置はゼロ、それ以外は
  素通し・scatter で寄与は各 1 つ）のため同じく bit 一致する
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
  45 件（forward・エッジケース・非 contiguous 入力・エラー系・勾配・
  有限差分検算・`NaN`／`inf` payload 保存・確保上限）
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
