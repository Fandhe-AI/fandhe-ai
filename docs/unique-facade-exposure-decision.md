# unique（torch.unique 相当）の facade 公開判断（#1734）

イシュー #1734「unique（勾配なし・3 バックエンド・facade 公開判断込み）を追加する」の facade 公開判断部分の設計記録。親: #1630（Tier 2「topk／sort／cumsum」）。

本ドキュメントは実装（`BackendOps::unique`・CPU／CUDA／Metal カーネル・`Var::unique`）と同一 PR で作成する（先例 #1775 とは異なり、実装自体がユーザー承認を要さない安全側の設計〈既存 `Var` 再エクスポート経由・新規公開面なし〉のため、決定記録と実装を分離する必要がない）。

## 1. 背景・要件

対応する PyTorch 機能: `torch.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)` の **values のみ**。`docs/spec/04-requirements.md` REQ-9 の 2026-09-12 改定（Tier 2「topk／sort／cumsum」。`docs/compat-api-scope.md` §1.3）に含まれる。

unique は他の Tier 2 演算（sort／topk／cumsum）と異なり、次の 2 点で既存の `Var`／`BackendOps` 設計から外れる:

1. **非微分演算**: 出力の各要素がどの入力位置に由来するかは一意に定まらない（重複除去により多対一の写像になる）ため、勾配（VJP）を定義できない。
2. **出力形状が入力値に依存して動的に決まる**: 出力 shape `[m]` の `m` は入力の重複度に依存し、shape だけからは事前に確定できない。`Var`（tape ノード）・`DeviceBuffer` 常駐推論チェーンはいずれも静的 shape 前提の設計であり、動的 shape を持つノードを tape へ記録する機構は現時点で存在しない。

## 2. 契約（実装の正）

`fandhe_ai_tensor_core::BackendOps::unique` のドキュメンテーションコメントを正とする。要点:

- 入力を row-major で平坦化 → totalOrder（`f32::total_cmp`）でソート → `==`（IEEE 比較）で隣接重複除去 → rank 1・contiguous な `Tensor<f32>`（shape `[m]`、`0 <= m <= numel`）を返す。
- `-0.0`／`+0.0` は同一視され、totalOrder の先頭側である `-0.0` が代表として残る。NaN は `NaN != NaN` のため全保持される。
- 選択演算（丸めなし）のため、3 バックエンドの出力は互いに bit 完全一致する契約（REQ-2 複合判定は用いない）。
- スコープ外（`.claude/rules/out-of-scope-tracking.md` 対象）: `return_inverse`／`return_counts`／`dim` 指定・`sorted=false`・`unique_consecutive`・GPU 側 prefix-sum 圧縮（3 バックエンドとも「ビットニックソート → ホスト側 `dedup_by`」方式で、圧縮自体はホストで行う）・性能最適化（radix sort 等）。

## 3. API 配置の案比較

| 案 | 内容 | 採否 |
|---|---|---|
| A. `Var::unique(&self) -> Result<Tensor<f32>, AutodiffError>` | `self` を層 1（`materialize_fallible`）で実体化した値に対し `BackendOps::unique` → `Unsupported` のときのみホストフォールバックを適用し、**detached な `Tensor<f32>`**（`Var` ではない）を返す。新規 `Op` を tape に記録しない。 | **採用** |
| B. `Var` を返す定数葉として tape に記録する | 出力を「勾配ゼロの葉」として新規 `Op::Unique` を追加し `Var` を返す。VJP は常に 0 を返す no-op として定義可能ではある。 | 不採用。動的 shape の tape ノードを許すと、`backward_impl`（固定 shape 前提の勾配蓄積バッファ）・`Tape::reset`（葉の shape 再利用契約）・checkpoint 機構（`docs/autodiff-checkpoint-design.md`）等、静的 shape を前提とする既存機構全体に波及する変更が必要になり、本 issue のスコープ（勾配なし演算 1 個の追加）を大きく超える。「勾配ゼロ」という意味論自体も、torch では `unique` の出力に対する `.grad` は単に「定義されない」（`requires_grad` を持たない新規テンソル）であり、tape 記録による「0 を流す」意味論とは一致しない。 |
| C. `Tensor<f32>::unique(&self)` を `tensor-core` に直接置く | `BackendOps` を経由せず `Tensor` の生アルゴリズム（CPU 実装）をクレート内で直接呼ぶ。 | 不採用。バックエンド切替（CPU／CUDA／Metal）の一貫した入口を `BackendOps` に統一する既存アーキテクチャ（`docs/backend-switching-design.md`）から外れ、CUDA／Metal 実装を配置する自然な場所がなくなる。 |
| D. facade 直下に `pub fn unique(x: &Tensor<f32>) -> Tensor<f32>` を新設する | `fandhe_ai::unique` として独立関数を公開する。 | 不採用（本 issue のスコープでは）。§5「範囲拡張手続き」参照。既存 `Var` 再エクスポート経由で到達可能なため、新規公開面を追加する必然性がない。 |

**採用（案 A）の帰結**: `Var::unique` は他の `Var` メソッド（`gather`・`where_cond` 等）と異なり `Result<Var<'t>, _>` ではなく `Result<Tensor<f32>, AutodiffError>` を返す非対称なシグネチャになる。これは意図的な設計判断であり、呼び出し側が「この演算の出力は以後の計算グラフに接続できない（微分不能・動的 shape）」ことを型シグネチャ自体で明示する（`.to_tensor()` を呼んで値を取り出す既存パターンと異なり、`unique` は最初から `Tensor` を返す）。

## 4. facade 公開判断

**新規公開面を追加しない**。`crates/facade/src/lib.rs` は無変更。到達経路は既存の `pub use fandhe_ai_autodiff::Var;` 経由（`fandhe_ai::Var::unique`）のみであり、これは #1620（`Var::einsum`）・#1711（`Var::log` 等の超越関数群）と同じ判断パターンである。

この判断は `docs/compat-api-scope.md` §5「範囲拡張手続き」の対象外である: 同手続きは facade **直下**への新規 `pub use`／`pub fn` 追加（`fandhe_ai::<新シンボル>` の新設）にユーザー承認を要求するものであり、`Var` の**既存**再エクスポート経由で到達可能な新規メソッド追加はこの手続きに含まれない（#1591／#1620／#1637／#1711 等の Tier 1／Tier 2 実装 issue すべてに共通する既定の扱い。`docs/compat-api-scope.md` §1.2／§1.3 の「facade 新規公開面なし」注記を参照）。

出力形状の動的性（§1 の 2 点目）についても、`Var::unique` が `Var` ではなく detached な `Tensor<f32>` を返す設計（§3 案 A）により、facade 利用者は shape が実行時に確定するテンソルとして扱う（`Tensor::shape()` を実行後に読む）契約になる。`DeviceBuffer` 常駐推論チェーン（`docs/inference-chain-single-sync-design.md`）への接続は対象外のまま（`unique` はホスト `Tensor` を返し、後続の再学習・推論チェーンへ流したい場合は改めて `tape.var(&out)` で葉として取り込む必要がある）。

## 5. ユーザー承認待ち事項（issue コメント上で列挙。本 PR では実装しない）

- facade 直下への新規 `pub fn unique(...)` の追加（§3 案 D）。動的 shape 対応の設計判断（案 A の detached `Tensor` 方式 vs 他の表現）を facade レベルで再確認する必要があるため、要否も含めユーザー判断を仰ぐ。
- `return_inverse`／`return_counts`（`torch.unique` の追加戻り値）。実装する場合は `Op::Unique`（`inverse` 出力のみ静的 shape のため tape 記録可能）の要否を含めた別設計が必要。
- `dim` 指定（軸限定の unique）。
- GPU 側 prefix-sum 圧縮によるホスト往復の削減（性能最適化。現状はビットニックソート後にホストへ 1 回読み戻してから `dedup_by` する設計）。

## 6. スコープ外事項

`.claude/rules/out-of-scope-tracking.md` の対象として本 issue のコメントへ記録する:

- `unique_consecutive`（隣接要素のみを対象とする軽量版）
- 上記§5 の全項目
- CUDA／Metal 実機実測（本実装環境には両実機への到達手段がなく、`#[ignore]` テストとして申し送る。§7）

## 7. 実装記録・実機実測状況

- CPU: `crates/backend-cpu/src/unique.rs`（参照実装）・単体テスト 6 件・`tests/unique_parity.rs`（`eval::unique` ホストフォールバックとの bit 一致。3 件）すべて Linux で green。
- CUDA: `crates/backend-cuda/src/{kernels_unique.rs, unique.rs, unique_model.rs}`。`unique_model.rs` の純関数プロパティテスト（キー変換往復・ビットニックソートの `total_cmp` 一致）は Linux で green（4 件）。実機（DGX Spark GB10）は本実装環境から到達不能のため `tests/unique_parity.rs` の `#[ignore]` テスト（`unique_matches_cpu_across_sizes`）は未実行のまま GB10 セッションへ申し送る。環境適応スモーク（`unique_parity_smoke_env_adaptive`）は CUDA 非搭載環境で `CudaUnavailable` を確認して green。
- Metal: `crates/backend-metal/src/{shaders/unique.metal, unique.rs, unique_model.rs}`。`unique_model.rs` の純関数プロパティテストは Linux で green（4 件）。`tests/unique_source_evidence.rs`（MSL ソース文字列証跡・境界検査の機械検証）も Linux で green（4 件）。`cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin` で型検査通過を確認済み。実機（Apple Silicon）は本実装環境から到達不能のため `tests/unique_parity.rs` の `#[ignore]` テスト（`unique_matches_cpu_across_shapes`）は未実行のまま Mac セッションへ申し送る。
- facade: `crates/facade/tests/unique_backend_parity.rs`。CPU vs NaiveOps の bit 一致（属性なし）は green。`api_surface.rs` は無変更で 18 件 green のまま（新規公開面がないことの間接確認）。
