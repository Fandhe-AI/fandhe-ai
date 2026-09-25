# tril・triu・diag・trace・outer・dot（#2144）CUDA／Metal 実機未実測の申し送り

`docs/autodiff-matrix-ops-decision.md` §7・§8「実装記録」参照。
本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、
`fandhe_ai_autodiff::matrix_ops`（`tril`／`triu`／`diag`〈両方向〉／
`trace`／`outer`／`dot`）の `crates/facade/tests/
matrix_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・Metal
（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする
**計 8 テスト**（CUDA・Metal 各 4 件で対称。コピー系 4 演算（`diag`
両方向を含む）forward bit 完全一致の
`cuda_copy_ops_forward_matches_cpu_reference`／
`metal_copy_ops_forward_matches_cpu_reference`、縮約系 2 演算 forward
の REQ-2 統一複合判定
`cuda_reduce_ops_forward_matches_cpu_reference`／
`metal_reduce_ops_forward_matches_cpu_reference`、bit 完全一致
backward（`tril`／`triu`／`diag` 両方向／`trace`／`dot` の 5 演算 6 セル
をそれぞれ個別に検証。「代表 1 演算での省略」はしない。イシュー
#2144 codex-review 是正）の
`cuda_bit_exact_backward_matches_cpu_reference`／
`metal_bit_exact_backward_matches_cpu_reference`、`outer` backward の
REQ-2 統一複合判定
`cuda_outer_backward_matches_cpu_reference`／
`metal_outer_backward_matches_cpu_reference`）は `#[ignore]` のまま
未実測である。

**diagonal の分岐セル（PR #2257 での追加是正）**: `tril`／`triu`／
`diag`（両方向）は `Op` 経路が同じでも diagonal 値によって境界位置が
変わるため、コピー系 forward・bit 完全一致 backward の各テストは
`DIAGONALS = [-1, 0, 1]`（負・0・正）をループで走査する（`tril` の
正・負、`triu` の負、`diag`〈1-D→2-D〉の `k<0`／`k==0` 分岐、`diag`
〈2-D→1-D〉の `k≠0`〈正・負〉を含む。「経路が別なら代表 1 本では
代わりにならない」という上記論拠を diagonal の境界位置にも適用した。
関数の追加はしていない）。

**diagonal の境界値・範囲外の走査（PR #2257 のフォローアップ・ユーザー
承認 2026-09-25。`docs/autodiff-matrix-ops-decision.md` §11 参照）**:
上記 `DIAGONALS = [-1, 0, 1]` は正方形状 `3×3` の内部値のみで、
`tril`／`triu` の早期リターン境界そのもの（`|diagonal|` が `n-1` や
`-(m-1)` に一致する値・それを跨ぐ範囲外の値）と `diag`（両方向）の
抽出長 `L=1`／`L=0` 境界・非正方形状（行 < 列の `3×5`・行 > 列の
`5×3`）は未走査だった。`tril_triu_diagonals(m, n)`・
`diag_1d_diagonals(n)`（`crates/facade/tests/
matrix_ops_backend_parity.rs`）で形状ごとに境界値・範囲外を機械的に
生成し、コピー系 forward・bit 完全一致 backward の全テスト（CPU・
CUDA・Metal）へ横展開した。`diag`（2-D→1-D）の `L=0`（範囲外）は
`narrow(0,0,0)` → `gather` → `squeeze` 経由で空テンソルへ収束し
**エラーにはならない**。backward は他セルと同じく勾配が `Some` で
記録されることを契約とし（`None` は双方一致していても失敗とする）、
CPU・NaiveOps セルでは `L=0` のとき形状 `[m, n]`・全要素ゼロである
ことも明示検証する。forward・backward いずれも
`f32_bits` の bit 列比較に加え `shape()` の一致も明示検証する
（空テンソル・早期リターンのセルで形状差が素通りしないようにする
ため）。CPU（属性なし）実行は `L=0` セルを含め全て green
（実測: CPU・NaiveOps ともに `Some`〈全ゼロ勾配〉で一致）。CUDA／
Metal の対応セルは他セル同様 `#[ignore]` のまま本 README の申し送り
対象に含む。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test matrix_ops_backend_parity -- --ignored --nocapture cuda

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test matrix_ops_backend_parity -- --ignored --nocapture metal
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト
（`cpu_copy_ops_forward_bit_matches_naive_reference`・
`cpu_forward_masked_positions_are_zero_and_nan_bits_preserved`・
`cpu_reduce_ops_forward_matches_naive_reference_within_tolerance`・
`cpu_bit_exact_backward_matches_naive_reference`・
`cpu_outer_backward_matches_naive_reference_within_tolerance`）で既に
検証済み（green）。

## 期待結果

`tril`／`triu`／`diag`（両方向）forward（`masked_fill`／`gather`／
`narrow`／`pad` の合成。いずれも値のコピーまたは定数 0 の埋め込みの
みで算術を含まない）は CPU・CUDA・Metal 間で構造的に bit 完全一致
するはず（`NaN`／`inf` の payload も含む）。backward は `tril`／
`triu`／`diag`（2-D→1-D）が `Op::MaskedFill`／`Op::Gather` の VJP
（寄与が高々 1 つ）を経由し同じ理由で bit 一致するはず。`diag`
（1-D→2-D）はこれに加え `Op::Pad`・`Op::BroadcastTo` の VJP（軸方向
の `reduce_to_shape` 縮約）も経由するが、縮約対象の行の非ゼロ要素が
高々 1 つ（残りは厳密 `+0.0`）のため、同じ `BroadcastTo` VJP を経由
しつつ複数の非ゼロ要素を縮約しうる `outer` backward（下記・REQ-2
統一複合判定が必要）とは異なり bit 完全一致のまま成立するはず。

`trace`（`diag` → `sum`）・`dot`（`mul` → `sum`）forward は `sum` の
縮約順序がバックエンドで異なりうるため REQ-2 の統一複合判定（相対
誤差 1e-3 未満 または 絶対誤差 1e-5 未満。`fandhe_ai_backend_cpu::
parity::assert_parity`）で比較する。`outer` backward（`broadcast_to`
の VJP。軸方向の `reduce_to_shape` 縮約）も同じ判定方式を適用する。

この前提が崩れる場合（`masked_fill`／`gather`／`pad` の実装が変わった、
デバイス間で想定外の丸めが混入した等）は本 README の「期待結果」を
更新し、想定した契約を維持できない事実を型付き findings として PR へ
記録すること（tolerance の単独緩和は行わない。`.claude/rules/
coding-rust.md`）。
