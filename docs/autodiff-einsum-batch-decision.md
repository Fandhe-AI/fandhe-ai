# einsum のバッチ添字縮約（rank≥3 matmul への分解）の設計判断記録

イシュー #2149（親 #2131）。`docs/autodiff-rearrange-ops-decision.md`・
`docs/autodiff-matrix-ops-decision.md` と同型の記録。

## §0 結論

`Var::einsum`（#1620）が rank≥3 `matmul`（#1600）未実装を理由に拒否
していた batch 添字（両オペランドと出力に共通する添字。例
`"bij,bjk->bik"`）を伴う 2 項縮約を、rank≥3 `Var::matmul`（`gemm_
batched`。#1715 で実装済み）への分解として `crate::einsum` 内に実装
した。新規 `Op`・新規 VJP・新規 `BackendOps` メソッドはゼロ（受け入れ
条件）。

**facade 公開（`Var::einsum` 自体を batch 添字受理へ拡張すること）は
イシュー #2149 の承認事項であり、未承認のため実施していない**。
`Var::einsum`（facade `fandhe_ai::Var::einsum` へそのまま到達する
公開入口）は引き続き batch 添字を伴う縮約を `InvalidArgument` で拒否
する。内部クレート限定の到達経路として `fandhe_ai_autodiff::
einsum_batch::einsum_batched` を追加した（`bool_ops`〈#2141〉・
`rearrange_ops`〈#2143〉・`matrix_ops`〈#2144〉と同じ「facade 公開
承認待ち保留」の判断枠組み）。

## §1 背景

イシュー #2149・親 #2131 にはコメントが 0 件で、facade 公開の承認
記録はない（着手前に `gh issue view 2149/2131` で確認済み）。親
#2131 はこのツリーでの facade 公開面の拡張を「設計判断記録 → 承認 →
実装」の 2 段階と定めているため、本実装は内部クレート限定に倒す。

`docs/compat-api-scope.md` §5 には「Tier 1／Tier 2 に列挙済みの機能の
実装は本節の再適用を要しない」という記述があり、einsum 自体は
§1.3 Tier 2（L265）に列挙済みである。一方でイシュー #2149 本文は
`Var::einsum` の batch 添字対応（facade 公開面の拡張）を承認事項として
明示している。この 2 つの記述は矛盾しうるため、§5「承認事項の扱い」
節で判断を記録する。

## §2 設計

### §2.1 モード導入（`crates/autodiff/src/einsum.rs`）

内部 `enum BatchContraction { Reject, Allow }` を導入し、
`pub(crate) fn einsum_with(spec, operands, mode)` を実装本体にした。
既存の `pub(crate) fn einsum(spec, operands)`（`Var::einsum` から
呼ばれる唯一の呼び出し元）は `einsum_with(.., Reject)` の薄い
ラッパーとして残す。`compute_binary_plan` は `mode` 引数を受け取り、
`!contract.is_empty() && !batch.is_empty() && mode == Reject` の
ときだけ `InvalidArgument` を返す（拒否判定は presum 実行より前の
まま。既存の「検証と `Var` 操作の分離」契約は不変）。

### §2.2 batch 付き GEMM 経路（`einsum_matmul_path` の一般化）

`EinsumMatmulPlan` に `batch: &[char]` を追加した。

- **batch 空**: 既存の 2 次元 `[L,K]×[K,R]` 経路をそのまま残す（`rank
  2` は `ops.gemm` へ流れるため、`matmul_spec_matches_var_matmul_
  exactly` の bit 同一契約は不変）。
- **batch 非空**: `[batch..., left..., contract...] → [B, L, K]`
  （`apply_permute` → `contiguous()` → `apply_reshape`）・
  `[batch..., contract..., right...] → [B, K, R]`（同様）を作り、
  `a_3d.matmul(&b_3d)`（rank-3 なので `gemm_batched` へ流れる）を
  1 回呼ぶ。出力を `[batch dims..., left dims..., right dims...]` へ
  `reshape` → `permute` して戻す。

恒等 permute・shape 不変の reshape はいずれもスキップされる
（`apply_permute`／`apply_reshape` の既存の最適化）ため、
`"bij,bjk->bik"` は rank-3 `MatMul` ノード 1 個だけを記録し、
`Var::matmul` 直接呼び出しと forward・backward とも bit 同一になる
（`crates/autodiff/tests/einsum_batch_parity.rs::
single_batch_axis_matches_var_matmul_directly_forward_and_backward`
で固定）。`"abij,abjk->abik"`（複数 batch 添字）は reshape が非恒等
になるため bit 同一の主張対象外（forward 値の REQ-2 突合のみ）。

### §2.3 数値契約

batch 経路は rank≥3 `Var::matmul`（`gemm_batched`）をそのまま呼ぶ
ため、FMA 契約・CUDA TF32 opt-in 挙動・VJP・checkpoint 再計算を
含めて `Var::matmul` と完全に同一。新しい丸め経路は作っていない
（`.claude/rules/coding-rust.md` FMA 契約統一）。

### §2.4 既知の制約

- `create_graph::validate_ancestors`（`crates/autodiff/src/
  create_graph.rs`）は rank≥3 の `MatMul` を高階微分
  （`Tape::backward_create_graph`）の対象から拒否する。そのため
  batch 添字を伴う einsum は二階微分では使えない（型付きエラー
  `AutodiffError::Backward` で拒否され、panic はしない。
  `crates/autodiff/tests/einsum_batch_parity.rs::
  batch_einsum_under_create_graph_returns_typed_error_not_panic`
  で固定）。
- einsum の次元検査は完全一致のみを要求し、size-1 broadcast は行わ
  ない（`Var::einsum` の既存契約のまま不変）。
- ellipsis（`...`）・3 オペランド以上は引き続き対象外
  （`crate::einsum` モジュール doc「受理範囲」参照）。

## §3 承認事項の扱い（本記録の中核判断）

`fandhe_ai::Var` は `pub use fandhe_ai_autodiff::Var` の直接再
エクスポートである（`crates/facade/src/lib.rs`）。そのため
`Var::einsum` の挙動を変えると、それだけで facade 公開面が変わる。

`docs/compat-api-scope.md` §5 の「Tier 1／Tier 2 に列挙済みの機能は
再適用不要」は、既に列挙済みの機能をそのまま実装する場合の手続き
省略を意味すると解釈できるが、einsum の**受理範囲の拡張**（batch
添字対応）は Tier 2 行の記述（「batch 添字を伴う縮約は rank≥3
`matmul` 未実装のため対象外」）を書き換える性質の変更であり、
イシュー #2149 本文はこれを承認事項として明示している。この 2 つの
記述の優先順位について、着手時点でユーザーからの明示的な指示は
ない。

自動運転モードの「判断がつかない場合は安全側に倒す」方針に従い、
**既定の実装方針は「保留実装」（`bool_ops`／`rearrange_ops`／
`matrix_ops` と同型）とした**:

- batch 縮約の分解ロジックは `crate::einsum` にすべて実装し、
  テスト（単体・統合・3 バックエンド parity）で検証する。
- 公開入口 `Var::einsum` は従来どおり batch∧contract を
  `InvalidArgument` で拒否する。承認前は facade から観測できる
  挙動を変えない。
- 新機能の到達経路として、autodiff クレート限定の自由関数
  `einsum_batch::einsum_batched` を設ける。facade からは再
  エクスポートしない。
- 承認後の切替は「`Var::einsum` が渡すモードを `Reject` から
  `Allow` へ 1 行変更し、保留ガードを撤去する」だけで済む構造に
  した（§6「承認後の切替手順」）。

**案 A（`Var::einsum` を直接拡張）を採らなかった理由**: 非破壊
（従来 `Err` だった入力が `Ok` になるだけでシグネチャは不変）では
あるものの、イシュー本文が明示的に承認事項として列挙しており、
親 #2131 のツリー方針（facade 公開面の拡張は承認を経る）に従う方が
安全側である。案 A の非破壊性自体は§6 に記録し、承認を推奨する。

## §4 対象ファイル・変更箇所

| パス | 変更内容 |
|------|---------|
| `crates/autodiff/src/einsum.rs` | `BatchContraction` モード・`einsum_with`・`compute_binary_plan` の条件付き拒否・`einsum_matmul_path` の batch 一般化・単体テスト追加 |
| `crates/autodiff/src/einsum_batch.rs`（新規） | 内部クレート限定の到達入口 `einsum_batched` |
| `crates/autodiff/src/lib.rs` | `pub mod einsum_batch;` |
| `crates/autodiff/src/var.rs` | `Var::einsum` doc の拒否理由更新（コードは不変） |
| `crates/autodiff/tests/einsum.rs` | 拒否テストのコメント更新（アサーションは不変） |
| `crates/autodiff/tests/einsum_batch_parity.rs`（新規） | NaiveOps 上の forward／backward／bit 同一性／checkpoint／境界／create_graph の検証 |
| `crates/facade/tests/einsum_batch_backend_parity.rs`（新規） | CPU vs Naive（属性なし）、Metal／CUDA vs CPU（`#[ignore]`）、`Var::einsum` 保留の実行時ガード |
| `crates/facade/src/lib.rs` | `VarEinsumBatchHoldDoctestGuard`（`#[cfg(doctest)]`） |
| `crates/facade/tests/api_surface.rs` | 保留ガード 4 件（doctest glob ドリフト・プローブ本体固定・pub use 走査・workspace インベントリ） |
| `docs/compat-api-scope.md` | §1.3 einsum 行の更新 |
| `docs/compat-feature-gap.md` | 追補（イシュー #2149）節 |
| `docs/public-api-design.md` | #1715 段落末尾への追記 |
| `docs/perf/logs/einsum-batch-2149/README.md`（新規） | CUDA／Metal 実機実測の申し送り |
| `docs/README.md` | 本 doc・perf log README の索引行 |

変更しないもの: `crates/backend-*`・`crates/tensor-core`・
`grad.rs`・`tape.rs`（`Op`／VJP／`BackendOps` は追加しない）、
`Cargo.toml`／`Cargo.lock`、tolerance 定数、`docs/spec/`。

## §5 テストの配置

- 単体テスト（`compute_binary_plan` の Allow/Reject 分類一致・
  `einsum_with(Allow)` の bit 同一性）は `crates/autodiff/src/
  einsum.rs` 内の `#[cfg(test)] mod tests` に追加した（既存の配置
  方針と同じ）。
- autodiff 統合テスト（`tests/einsum_batch_parity.rs`）は `tests/
  einsum.rs`（#1620）と同じブルートフォース n 次元参照実装・REQ-2
  ヘルパー・中央差分ヘルパーを独立に複製した（別テストバイナリは
  非公開ヘルパーを import できないため。新しい許容誤差は導入して
  いない）。
- facade 3 バックエンド parity（`tests/einsum_batch_backend_parity.
  rs`）は `einsum_backend_parity.rs`（#1620）と同型の `VarSource`
  トレイト・`Xorshift64Star` 決定的シード・`assert_parity` 構成を
  踏襲した。

## §6 承認後の切替手順

1. `crates/autodiff/src/einsum.rs::einsum`（`Var::einsum` の実装
   本体）が渡す `BatchContraction` を `Reject` から `Allow` へ 1 行
   変更する。
2. `crates/autodiff/tests/einsum.rs::rejects_batch_axis_contraction`
   （`Var::einsum` の拒否テスト）を受理テストへ書き換える。
3. `crates/facade/src/lib.rs::VarEinsumBatchHoldDoctestGuard`
   （doctest 本体）を撤去する。
4. `crates/facade/tests/api_surface.rs` の対応する 4 テスト
   （`einsum_batch_hold_doctest_globs_all_pub_modules`・
   `einsum_batch_hold_doctest_probe_body_matches_fixed_contract`・
   `facade_does_not_reexport_or_declare_einsum_batch`・
   `workspace_declares_einsum_batched_fn_only_in_autodiff_
   einsum_batch`）を撤去する。
5. `crates/facade/tests/einsum_batch_backend_parity.rs::
   facade_var_einsum_still_rejects_batch_contraction` を撤去する
   （`Var::einsum` が受理するようになるため）。
6. `docs/compat-api-scope.md`・`docs/compat-feature-gap.md`・
   `docs/public-api-design.md` の「未承認」記述を更新する。
7. `einsum_batch.rs` 自体を撤去するか、`Var::einsum` の薄い
   ラッパーとして残すかは、その時点の互換性方針（既存の
   `fandhe_ai_autodiff::einsum_batch::einsum_batched` 呼び出し元が
   あるか）に従って判断する。

## §7 実機 parity の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/einsum-batch-2149/README.md` へ測定コマンド
案・期待結果・既知リスクを申し送る。

## §8 スコープ外

- facade 公開（`Var::einsum` の `Allow` 化と保留ガードの撤去。§3・§6
  参照）
- GPU 専用カーネル（現状は既存の `matmul`／`permute`／`reshape` の
  合成のみで到達）
- ellipsis（`...`）・3 オペランド以上（`crate::einsum` の既存対象外
  のまま）
- 複数 batch 添字（`abij,abjk->abik`）の bit 同一契約（forward 値の
  REQ-2 突合のみで、`MatMul` ノード 1 個への帰着は単一 batch 添字
  限定）

## §9 承認事項（未承認として列挙）

1. facade 公開（`Var::einsum` の batch 添字対応拡張。§3・§6 参照）
2. GPU 専用カーネル

## §10 実装記録（イシュー #2149）

- `crates/autodiff/src/einsum.rs`: `BatchContraction` モード・
  `einsum_with`・`compute_binary_plan` の条件付き拒否・
  `einsum_matmul_path` の batch 一般化・モジュール doc 更新・単体
  テスト 3 件追加（`einsum_with_allow_accepts_batch_contraction_as_
  single_matmul_node`・`compute_binary_plan_allow_and_reject_
  classify_identically`・既存拒否テストのコメント更新）
- `crates/autodiff/src/einsum_batch.rs`（新規）: `einsum_batched`
  1 関数（`einsum_with(.., Allow)` への薄い委譲）
- `crates/autodiff/src/lib.rs`: `pub mod einsum_batch;`・クレート
  doc へイシュー #2149 の要約を追記
- `crates/autodiff/src/var.rs`: `Var::einsum` doc の拒否理由更新
- `crates/autodiff/tests/einsum.rs`: `rejects_batch_axis_contraction`
  のコメント更新（アサーション不変）
- `crates/autodiff/tests/einsum_batch_parity.rs`（新規）: forward
  （ブルートフォース突合 7 件）・bit 同一性（1 件）・backward（中央
  差分突合 2 件）・checkpoint（1 件）・境界（3 件）・`Var::einsum`
  保留固定（1 件）・create_graph 型付きエラー（1 件）の計 16 テスト
- `crates/facade/src/lib.rs`: `VarEinsumBatchHoldDoctestGuard`
- `crates/facade/tests/api_surface.rs`: 保留ガード 4 テスト（着手前
  の再 grep で他クレートとの名前衝突は見つからなかったため期待集合
  は `crates/autodiff/src/einsum_batch.rs` の 1 件のみ）
- `crates/facade/tests/einsum_batch_backend_parity.rs`（新規）: CPU
  と NaiveOps の forward／backward REQ-2 突合（属性なし 2 件）＋
  `Var::einsum` 保留の実行時ガード（1 件）＋CUDA／Metal の
  `#[ignore]`（未実測。4 件。`docs/perf/logs/einsum-batch-2149/
  README.md` 参照）
- `docs/compat-api-scope.md`：§1.3 einsum 行の更新
- `docs/compat-feature-gap.md`：追補（イシュー #2149）節
- `docs/public-api-design.md`：#1715 段落末尾への追記
- `docs/README.md`：本 doc・perf log README の索引行

承認取得後の追随（本イシューでは未実施）: `Var::einsum` の `Allow`
化、facade 保留ガード（`VarEinsumBatchHoldDoctestGuard`・対応する
否定ガード 4 件・facade parity テストの保留ガード 1 件）の撤去
（§6 参照）。
