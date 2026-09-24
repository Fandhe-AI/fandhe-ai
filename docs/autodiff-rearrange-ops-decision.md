# repeat・tile・flip・roll の設計判断記録

イシュー #2143（親 #2131）。`docs/autodiff-bool-ops-exposure-decision.md`
と同型の記録。

## §0 結論

PyTorch 互換の形状演算 4 種（`repeat`／`tile`／`flip`／`roll`）を、
**`fandhe_ai_autodiff` のうち facade が再エクスポートしない自由関数
モジュール `rearrange_ops`**（`crates/autodiff/src/rearrange_ops.rs`）
として実装した（案 C。§3 参照。`bool_ops`〈#2141〉と同じ判断枠組み）。
`Var` に inherent の `pub fn` は追加していない。新規 `Op`・
`BackendOps` メソッド・VJP・tape ノードは追加していない——いずれも
既存の `Var::index_select`（実体は `Var::gather` → `Op::Gather`）と
`Var::broadcast_to`（`Op::BroadcastTo`）の合成のみで構成した。facade
公開（`Var` への委譲メソッド追加）は承認待ちのまま対象外とし、
`crates/facade/src/lib.rs::VarRearrangeOpsHoldDoctestGuard`（正の
プローブ doctest）と `crates/facade/tests/api_surface.rs` のソース走査・
workspace インベントリ（4 テスト）で多層固定している。

## §1 背景

イシュー #2143・親 #2131 にはコメントが 0 件で、facade 公開の承認記録
はない（2026-09-24 時点。着手前に `gh issue view 2143/2131 --comments`
で確認済み）。親 #2131 はこのツリーでの facade 公開面の拡張を「設計
判断記録 → 承認 → 実装」の 2 段階と定めているため、本実装は内部
クレート限定に倒す。

## §2 設計判断

### §2.1 API 配置

- **案 A（`Var` の inherent メソッド）**: facade へ即座に到達する
  （`crates/facade/src/lib.rs` は `pub use fandhe_ai_autodiff::{…, Var,
  …};` で `Var` を再エクスポートしているため）。承認が無いため不採用
- **案 C（自由関数）**: 採用。承認後の撤去・委譲が単純
- モジュール名は `shape_ops` にしない（並行して動く兄弟イシュー
  #2144〈tril／triu 等〉との名前衝突回避・onnx-interop の private
  `mod shape_ops` との混同回避・既存テスト
  `crates/facade/tests/shape_ops_backend_parity.rs`〈#1597〉との混同
  回避）

### §2.2 合成の方針（新規 `Op`・`BackendOps` はゼロ）

使う既存演算は 2 つのみ。いずれも CPU・CUDA・Metal に経路がある。

- `Var::index_select`（`var.rs:4333`。実体は `Var::gather` →
  `Op::Gather`。VJP は `ScatterReduce::Add` による scatter。
  `grad.rs:1797`）
- `Var::broadcast_to`（`var.rs:2761`。view。長さ 1 の先頭軸を足すのに
  使う——`Var::reshape`／`unsqueeze` は非 contiguous 入力に対し
  `NonContiguousReshape` を返すため使わない）

各関数の合成:

- **`flip(x, dims)`**: 軸ごとに逆順添字 `[n-1, …, 0]`
  （`Tensor<i32>`）で `index_select(d, idx)` を順に適用する。`dims` が
  空なら新しいノードを積まず `x` をそのまま返す。軸番号が範囲外は
  `ShapeError::AxisOutOfRange`、重複は `ShapeError::DuplicateAxis`
  （PyTorch と同じくエラー）
- **`roll(x, shifts, dims)`**: `shifts.len() == dims.len() >= 1` を
  必須にする。`dims=None`（flatten してから roll する形）は非対応
  （`Var::expand` の「-1 非対応」注記と同じ扱い）。軸ごとに
  `s = shift.rem_euclid(n)` を計算し、`n == 0` または `s == 0` の軸は
  スキップする。それ以外は添字 `idx[j] = (j + n - s) % n` で
  `index_select` する。重複軸は PyTorch と同じく順に適用する
- **`repeat(x, repeats)`**: `repeats.len() < rank` は
  `InvalidArgument`。`repeats.len() > rank` の場合は `broadcast_to` で
  先頭に長さ 1 の軸を追加してから rank を揃える。軸ごとに `r == 1`
  ならスキップし、それ以外は添字 `idx = (0..n*r).map(|j| j % n)` で
  `index_select` する。`r == 0` または `n == 0` の軸は添字長 0 になり
  出力軸が長さ 0 になる（エラーにしない）。全軸が `r == 1` で rank も
  不変なら新しいノードを積まず `x` をそのまま返す
- **`tile(x, reps)`**: `reps.len() < rank` なら先頭を 1 で埋めて
  `repeat` に委譲する。それ以外はそのまま `repeat` に委譲する

**境界検査（REQ-8・`.claude/rules/security.md` A03）**: 軸長・添字値は
`i32` 範囲内であることを事前検査する（`checked_axis_len_as_i32`）。
添字ベクタの確保は `checked_mul` によるオーバーフロー検査・確保前
サイズ検査（`checked_index_alloc_len`）を確保前に必ず通す
（`crate::bool_ops::checked_bytes_for` と同型の独立複製）。当初は
`Vec` allocation 契約上の上限（`isize::MAX` バイト）のみを検査して
いたが、これは技術的にオーバーフローしない範囲の巨大値（例: shape
`[1]` に `repeats=[1_000_000_000]` で 4GB）を拒否できず、実確保の
失敗による abort を招きうる指摘（codex-review・PR #2256）を受け、
実用上の確保バイト数上限 `MAX_INDEX_ALLOC_BYTES`（1 GiB）による検査
へ強化した。`flip`／`roll`／`repeat`／`tile` いずれも同じ
`checked_index_alloc_len` を確保前チェックポイントとして通るため、
`broadcast_to` の stride-0 view 経由で軸長が巨大化するケース（`flip`／
`roll`）も含め一律にこの上限が適用される。

### §2.3 CPU 参照実装（`backend-cpu/src/ops.rs`）は追加しない

Issue 本文が挙げる「`backend-cpu/src/ops.rs` に CPU 参照実装」は、
受け入れ条件「新規 `Op` なし・既存演算の合成のみ」と両立しない
（`BackendOps` への新メソッド追加は crates.io 公開済みの `tensor-core`
公開 trait を広げることにもなる）。合成に使う `gather` は CPU・CUDA・
Metal すべてに実装済みのため、この差分でも「CUDA／Metal は既定
`Unsupported` のフォールバックで到達可能」という契約は追加作業なしで
満たされる。「CPU 参照実装との突き合わせ」は、テストコード内に添字
計算だけで書いた独立参照実装ではなく、有限差分検算・解析的期待値の
直接算出で満たした（`crates/autodiff/src/rearrange_ops.rs` の
`#[cfg(test)] mod tests`）。

**`crates/backend-cpu/**`・`crates/tensor-core/**`・
`crates/backend-{cuda,metal}/**` は変更していない。**

### §2.4 テストの配置

- 単体テスト・勾配テスト（NaiveOps）は `crates/autodiff/src/
  rearrange_ops.rs` 内の `#[cfg(test)] mod tests` に置いた
  （`bool_ops.rs` と同じ配置。当初計画の別ファイル
  `crates/autodiff/tests/rearrange_ops.rs` は、`bool_ops.rs` の実際の
  先例〈テストは全てモジュール内〉に倣い作成しなかった）
- バックエンド間 parity は `crates/facade/tests/
  rearrange_ops_backend_parity.rs`（`fandhe_ai_autodiff::rearrange_ops::*`
  を直接 use）

## §3 数値契約

forward は値のコピーのみ（算術を含まない）ため 3 バックエンド間で
構造的に bit 完全一致する（`NaN` の payload も保存される）。backward
（`Op::Gather` の VJP。scatter-add）は `flip`／`roll` が各入力要素への
寄与 1 つのため bit 一致し、`repeat`／`tile` は `r` 個のコピーの勾配を
合算するため REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満。`fandhe_ai_backend_cpu::parity::assert_parity`）で比較する。

## §4 PyTorch との差異

- `roll` の `dims=None`（flatten してから roll する形）は非対応
- 軸番号は非負のみ（本クレートの他の形状演算と同じ規約）

## §5 スコープ外

- facade 公開（`Var::repeat`／`tile`／`flip`／`roll` の委譲メソッド化と
  保留ガードの撤去）
- `roll` の `dims=None`（flatten 形）対応
- GPU 専用カーネル（現状は `gather` の既存カーネル・ホスト
  フォールバックのみで到達）

## §6 承認事項（未承認として列挙）

1. facade 公開（上記スコープ外 1 と同じ）
2. `roll` の `dims=None` 対応
3. GPU 専用カーネル

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md` へ
測定コマンド案・期待結果を申し送る。

## §8 実装記録（イシュー #2143・2026-09-24）

- `crates/autodiff/src/rearrange_ops.rs`（新規）: `repeat`／`tile`／
  `flip`／`roll`・境界検査ヘルパー・単体テスト 31 件（forward・
  エッジケース・非 contiguous 入力・エラー系・勾配・有限差分検算）
- `crates/autodiff/src/lib.rs`: `pub mod rearrange_ops;` を追加
  （アルファベット順維持）
- `crates/facade/src/lib.rs`: `VarRearrangeOpsHoldDoctestGuard`（正の
  プローブ doctest。`VarBoolOpsHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`:
  `rearrange_ops_hold_doctest_globs_all_pub_modules`・
  `rearrange_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_rearrange_ops`・
  `workspace_declares_rearrange_ops_fn_names_only_in_autodiff_
  rearrange_ops`（4 テスト。後者は `backend-cuda/src/gemm.rs::tile`
  〈GEMM タイル設定の無関係な inherent メソッド。2 件〉を期待集合に
  明示的に含める——`tile` という名前の衝突が実在するため
  `bool_ops` 系より 1 段複雑）。手動検証: `pub use
  fandhe_ai_autodiff::rearrange_ops;` の仮追加でソース走査が
  fail-closed に検出することを確認済み（追加 → テスト失敗を確認 →
  削除 → テスト成功を再確認。`Var` への仮 inherent メソッド追加は
  doctest 側で同型の衝突機構を持つため個別の手動確認は略した）
- `crates/facade/tests/rearrange_ops_backend_parity.rs`（新規）: CPU と
  NaiveOps の forward bit 一致・flip/roll backward bit 一致・
  repeat backward の REQ-2 統一複合判定（属性なし 4 件）＋CUDA／Metal
  の `#[ignore]`（未実測。`docs/perf/logs/
  shape-repeat-tile-flip-roll-2143/README.md` 参照）
- `docs/compat-api-scope.md`: §1.2「形状操作」行へ追補
- `docs/README.md`: 本 doc・perf log README の索引行を追加

**追記（イシュー #2143 レビュー指摘対応）**: `crates/facade/tests/
rearrange_ops_backend_parity.rs` の `#[ignore]` テストは当初 `flip`
forward のみで、`repeat`／`tile` backward の実機カバレッジが欠けて
いた（§7・本節が申し送る「期待結果」と実際のテスト内容の齟齬）。
`cuda_repeat_tile_backward_matches_cpu_reference`・
`metal_repeat_tile_backward_matches_cpu_reference`（`#[ignore]`）を
追加し、CPU 側テストも `cpu_repeat_backward_matches_naive_reference_
within_tolerance` から `cpu_repeat_tile_backward_matches_naive_
reference_within_tolerance` へ改名して tile backward の突き合わせを
追加した（属性なしテストは 4 件のまま、CUDA／Metal `#[ignore]` は
2 件 → 4 件）。`docs/perf/logs/shape-repeat-tile-flip-roll-2143/
README.md` も新テスト名に追随済み。

**追記（PR #2256 codex-review 2 巡目の指摘対応。2026-09-24）**:
1 巡目の是正後も、次の「契約と実際のテスト網羅範囲のずれ」3 件が
未解消だった（いずれも同じ類型: 関数名・doc コメント・README の
一覧が実装の網羅範囲とずれている）。網羅表（4 演算 ×
{forward, backward} × {CPU vs NaiveOps, CUDA vs CPU, Metal vs CPU}）
を作成し空セルを機械的に洗い出して一括是正した:

1. `cpu_flip_roll_backward_bit_matches_naive_reference` が `roll`
   backward を実行していなかった → 同関数に `roll` backward
   （CPU／NaiveOps 突き合わせ）を追加
2. Metal／CUDA の `#[ignore]` backward テストが `flip` を欠いていた
   → `cuda_roll_backward_matches_cpu_reference`・
   `metal_roll_backward_matches_cpu_reference` を
   `cuda_flip_roll_backward_matches_cpu_reference`・
   `metal_flip_roll_backward_matches_cpu_reference` へ改名し `flip`
   backward の比較を追加（CUDA／Metal `#[ignore]` のテスト関数数は
   6 件のまま〈1 巡目の追記時点で forward 2 件・`repeat`／`tile`
   backward 2 件・`roll` backward 2 件の計 6 件へ増えていた。本追記は
   その `roll` backward 2 件を `flip`／`roll` 両対応へ拡張しただけで
   関数の増減はない〉）
3. `docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md` の
   未実測対象カウントが「4 テスト」のままだった（実際は forward
   2 件・`flip`／`roll` backward 2 件・`repeat`／`tile` backward
   2 件の計 6 件）→ README を実装に合わせて更新
   （網羅表の一覧を追記）

網羅表で見つかった追加の同型不一致（`cpu_forward_preserves_nan_bits`
が `flip` の `NaN` payload 保存しか検証していなかった）も同時に是正
し、`roll`／`repeat`／`tile` の `NaN` payload 保存検証を追加した。

承認取得後の追随（本イシューでは未実施）: `Var::repeat`／`tile`／
`flip`／`roll` 等の薄い委譲メソッド追加、facade 保留ガード
（`VarRearrangeOpsHoldDoctestGuard`・対応する否定ガード 4 件）の撤去。

**追記（PR #2256 CI・レビュー指摘対応。2026-09-24）**: `cargo doc`
（`-D rustdoc::private-intra-doc-links` 相当。`-D warnings` 暗黙包含）が
モジュール doc・`checked_index_alloc_len` の doc コメント中の
`` [`MAX_INDEX_ALLOC_BYTES`] ``（private 定数への intra-doc リンク）を
拒否したため、非リンクのコードスパン表記（`` `MAX_INDEX_ALLOC_BYTES` ``）
へ変更した。

`repeat` の確保前検査を以下のとおり作り直した（cursor-review「Zero
axis bypasses allocation guard」・「Allocation cap rejects no-op
repeat」・codex-review「ゼロ係数より先の大きな軸が空テンソル契約を
エラーに変える」の 3 件を同一原因として一括解消）:

1. Pass 1（確保なし）: 軸ごとの `axis_total = n * r`・全軸積
   `total_out_elems` を `checked_mul` のみで確定する
   （`MAX_INDEX_ALLOC_BYTES` の判定はまだ行わない）
2. `total_out_elems == 0`（最終出力が空テンソル）なら、最初に見つかった
   `axis_total == 0` の軸だけ空添字（長さ 0）で `index_select` して
   実体を空にしてから `Var::reshape` で最終 shape へ一括変換する。
   `Tensor::is_contiguous` は `numel() == 0` を常に連続とみなす
   （NumPy 方式）ため `NonContiguousReshape` にならない。他の軸の
   `r` がどれだけ大きくても `n*r` 長の添字ベクタを確保しない
3. 非ゼロケースでは `r == 1` の軸（no-op でも確保上限チェック不要）を
   除外してから `checked_axis_len_as_i32`・`checked_index_alloc_len`
   を適用する

「新規 `Op` はゼロ」の受け入れ条件は維持する——`Var::reshape`
（`Op::Reshape`）は空テンソル最終化にのみ用いる既存演算で、CPU・
CUDA・Metal 全バックエンドに既存経路がある。

回帰テストを 4 件追加した（`repeat_large_axis_before_zero_axis_
returns_empty_ok`・`repeat_large_axis_before_zero_axis_with_n_gt_1`・
`repeat_empty_output_gradient_matches_input_shape`・
`repeat_no_op_on_huge_broadcast_view_does_not_reject`）。既存の負例
（`repeat_rejects_practically_unallocatable_size_without_panicking`
等）は変更せず全件通過を確認済み。
