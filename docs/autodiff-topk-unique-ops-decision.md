# topk・unique のオプション拡張（#2153）設計判断記録

イシュー #2153「topk・unique のオプション拡張（sorted・dim・return_* など）」（親 #2131「PyTorch／TF 置き換えの API 網羅」5-B 演算）の設計判断記録。`docs/autodiff-reduce-ops-decision.md`・`docs/autodiff-indexing-inplace-design.md` と同型の「内部モジュール＋facade 保留ガード」枠組みを踏襲する。

## 0. facade 非公開（意図的）

`Var` は facade（`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への inherent メソッド追加は即座に facade 公開面へ出てしまう。本イシューは facade 公開（`Var` への委譲メソッド追加）を承認事項として明示するため、承認が取れるまでは新 API を `Var` の外に自由関数として `fandhe_ai_autodiff::topk_unique_ops`（`crates/autodiff/src/topk_unique_ops.rs`）へ置き、facade から到達不能にする。`crates/facade/src/lib.rs::VarTopkUniqueOpsHoldDoctestGuard`（正のプローブ doctest）＋`crates/facade/tests/api_surface.rs` の 4 テストで多層防御を固定する。

## 1. API 表

| 関数 | PyTorch 相当 | 出力 | 微分 |
|---|---|---|---|
| `topk_with_options(x, k, TopkOptions)` | `torch.topk(k, dim, largest, sorted)` | `(Var, Tensor<i32>)` | 可 |
| `unique_with_options(x, UniqueOptions)` | `torch.unique(sorted=True, return_inverse, return_counts, dim)` | `UniqueOutput`（detached） | 不可 |
| `unique_consecutive(x, UniqueOptions)` | `torch.unique_consecutive(return_inverse, return_counts, dim)` | `UniqueOutput`（detached） | 不可 |

既存 `Var::topk`（`sorted=True` 固定・非負 `dim` のみ。#1733）・`Var::unique`（values のみ。#1734）のシグネチャ・意味論は不変。`topk_with_options` は `sorted=true` の場合、正規化済み `dim` で既存 `Var::topk` へそのまま委譲する（挙動・ノード構成とも完全同一）。

## 2. 契約

### 2.1 負 dim の正規化

`topk_unique_ops::normalize_dim(dim: isize, rank: usize) -> Result<usize, AutodiffError>`。許容範囲 `[-rank, rank)`。`dim >= rank`（非負）は既存 `topk_out_shape` 等と同じ `AutodiffError::Shape(ShapeError::AxisOutOfRange)`、負で範囲外は `AutodiffError::InvalidArgument`。`rank == 0` は常に拒否する（既存 `topk_out_shape` の 0-d 拒否契約に合わせる。PyTorch は 0-d を許容するため差分。§3）。

### 2.2 topk `sorted=false` の決定的契約

PyTorch は `sorted=False` の順序を未規定とするが、本リポでは 3 バックエンド bit 一致のため「既存 topk（`sorted=True` 相当）で選んだ `k` 個を、`dim` 軸上の元添字の昇順に並べ替えた順」と定義する。実装は `grad::topk_with_fallback`（`sorted=true` の内部経路と同一関数）を呼んだ後、ホスト側で各レーンの `index` を昇順ソートする置換を `values`・`index` の双方へ適用する（`topk_unique_ops::resort_topk_by_index`）。選択演算（丸めなし）のため bit 一致は構成的に保たれる。`Op::Topk` の VJP（`grad.rs` の scatter ベース実装）は index 順序に依存しないため、置換後の組をそのまま記録すれば勾配は正しい——**新規 `Op`・新規 `BackendOps` メソッドは追加しない**（facade backend parity テストで `sorted=false` の勾配が `sorted=true` の勾配と bit 同一であることを固定済み）。

### 2.3 unique 拡張の契約

新規 `fandhe_ai_tensor_core::BackendOps::unique_ext`（既定 `Unsupported`。CUDA／Metal は override しない——GPU 専用カーネルは別イシュー。§7）を追加し、常に 3 出力（`UniqueExtOutput { values, inverse, counts }`）を返す。呼び出し元（`unique_with_options`・`unique_consecutive`）が要求しなかった出力を破棄する。

- **非微分・tape 非記録・detached 出力**（既存 `Var::unique` と同型の理由。出力形状が入力値に依存して動的に決まるため）。
- `dim = None`: 平坦化 →（`consecutive` なら元順序のまま、そうでなければ `f32::total_cmp` 昇順＋元添字昇順タイブレークでソートして）隣接 IEEE `==` で群化する。`values` は既存 `Var::unique` と同じ順序キー・重複判定述語（`-0.0`／`+0.0` は同一視され `-0.0` が代表、NaN は全保持）。
- `dim = Some(d)`: 軸 `d` の各スライス（他軸を row-major で平坦化した「行」）を 1 単位として、行同士を 2 キー方式で比較する: 主キーは ±0 を `0.0` へ正規化した要素ごと `total_cmp` 辞書式（IEEE 等価なスライス、例えば `[-0.0, 5.0]` と `[+0.0, 5.0]` を必ず隣接させる）、副キーは正規化しない生の `total_cmp` 辞書式（±0 を含む行同士のタイを totalOrder で確定的に解決し、群代表を常に `-0.0` 側にする）。群化は行全体の IEEE `==`。退化ケース（スライス長 0）は「全スライスが等しい → m = 1」が二重キー比較の性質から自然に成立し、特別扱いのコードは不要（`shape[d] == 0` は `m = 0`）。
- `consecutive = true`: ソートせず元の並び順のまま隣接（`dim` 指定時は隣接スライス）を IEEE `==` で群化する（`torch.unique_consecutive` 相当）。群代表は各連続ランの**先頭出現**。
- `inverse`・`counts` は `Tensor<i32>`。対象要素数（`dim=None` なら numel、`dim` 指定時は `shape[dim]`）が `i32::MAX` を超える場合は dispatch 前に `AutodiffError::InvalidArgument` で拒否する（`topk_unique_ops::ensure_target_len_fits_i32`）。
- `unique_with_options` は `dim=None` かつ `return_inverse=return_counts=false` の場合のみ既存 `unique_with_fallback`（CUDA／Metal の既存カーネル経路をそのまま使う）へ委譲し、`values` は既存 `Var::unique` と bit 同一になる。それ以外は `unique_ext_with_fallback`（`grad.rs`）経由。
- `unique_consecutive` は常に `unique_ext_with_fallback(consecutive=true)` 経由（既存 `unique` カーネルへは委譲しない——意味論が異なるため）。

CPU 実装（`crates/backend-cpu/src/unique.rs::unique_ext`）は `fandhe_ai_autodiff::eval::unique_ext` と意図的に同一アルゴリズムを複製する（既存 `unique`／`eval::unique` の複製方針を踏襲）。バックエンド戻り値は `grad::validate_unique_ext_output` が事後検査する（shape・`counts` 総和・`inverse` 値域の不変条件）。

## 3. PyTorch との差分

- `rank == 0`（0-d）入力の `topk`／`unique` 拡張はすべて拒否する（既存 `topk_out_shape` の契約を維持。PyTorch は 0-d を許容）。`topk_with_options` は常に `dim` を要求するため `normalize_dim`（`d >= rank` が `rank == 0` で常に真になる）が自然に拒否するが、`unique_with_options`／`unique_consecutive` は `dim=None`（軸非指定・平坦化）を許す API であり `normalize_dim` を経由しない経路（`unique_with_options` の `dim=None && !return_inverse && !return_counts` 早期委譲分岐を含む）が漏れていた。codex-review P2 是正（PR #2270）で両入口の冒頭に `topk_unique_ops::reject_rank_zero`（`ShapeError::RankMismatch { expected: 1, actual: 0 }`）を追加し、`dim` の有無に関わらず 0-d を一律拒否する契約に揃えた（既存 `Var::unique`〈#1734〉自体の 0-d 許容契約は本イシューのスコープ外のため変更していない）。
- `unique` の `sorted=false` は本イシューの要件外（対応しない）。

## 4. バックエンド配線

- `tensor-core::BackendOps::unique_ext`: 新規デフォルトメソッド（既定 `Unsupported`）。
- `backend-cpu`: `unique::unique_ext` 本体 + `ops.rs` 委譲。
- `autodiff::eval::unique_ext`: ホスト参照実装（`unique_ext_flat`／`unique_ext_slices`／`unique_ext_row_cmp`／`unique_ext_row_eq`／`unique_ext_group_sorted`／`unique_ext_group_consecutive`）。
- `autodiff::grad::unique_ext_with_fallback`: `ops.unique_ext` → `Unsupported` のときのみ `eval::unique_ext` へフォールバック。
- **CUDA／Metal は `unique_ext` を override しない**（既定 `Unsupported` → ホストフォールバック到達。GPU 専用カーネルは別イシュー。§7）。`topk` は新規 `BackendOps` メソッドを追加していない（§2.2）。

## 5. 確保前検査（REQ-8）

3 つの公開入口（`topk_with_options`・`unique_with_options`・`unique_consecutive`）すべての冒頭・あらゆる分岐（`sorted=true`／`dim=None` かつ追加出力なしの既存委譲経路を含む）より前に `topk_unique_ops::ensure_alloc_fits_f32`（`crate::bool_ops::checked_bytes_for` 経由）で入力 shape の要素数積オーバーフロー・`isize::MAX` バイト超過を検査する。`unique` 系はさらに対象要素数の `i32` 上限を dispatch 前に検査する（§2.3）。

## 6. 承認事項（本 PR では実施しない）

- facade への再エクスポート（`Var` への委譲メソッド追加を含む）。窓口は #2153／#2131。

## 7. スコープ外

- CUDA／Metal の `unique_ext` 専用カーネル・topk の `sorted=false` 専用カーネル（別イシュー）。
- unique の `sorted=false`（本イシュー要件外）。
- GPU 側 prefix-sum 圧縮によるホスト往復削減（`docs/unique-facade-exposure-decision.md` §5 既存項目）。
- `inverse`／`counts` の `Tensor<i64>` 化（現行の index 型方針 `i32` に従う）。
- rank 0 入力の topk／unique 拡張特例。
- CUDA（GB10）／Metal（M4 Max）実機 parity 実測: `docs/perf/logs/topk-unique-2153/README.md` へ申し送り。

## 8. 実装・実測記録

- CPU: `cargo test -p fandhe-ai-tensor-core`・`cargo test -p fandhe-ai-backend-cpu --lib unique`・`cargo test -p fandhe-ai-autodiff --test topk_unique_parity`・`cargo test -p fandhe-ai --test topk_unique_ops_backend_parity`・`cargo test -p fandhe-ai --test api_surface` すべて green（実装 PR で実行済み）。
- CUDA（DGX Spark GB10）／Metal（Apple Silicon）実機 parity（`#[ignore]` テスト）は本実装環境に到達手段がないため未実施。`docs/perf/logs/topk-unique-2153/README.md` へ申し送る。
