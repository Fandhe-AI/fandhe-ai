# テンソル積・距離・外積（`kron`・`tensordot`・`cdist`・`cross`）の CPU 実装記録（イシュー #2640）

親 #2625「Phase 4」・ルート #2499。実装は `crates/autodiff/src/tensor_product_ops.rs`。
**本書は実装と推奨案の記録であり、facade 公開形の承認記録ではない。**

## 0. 結論

- 4 演算（`tensordot` は dims 整数形 `tensordot` と軸リスト形 `tensordot_axes` の 2 関数。計 5 関数）を、
  **新規 `Op`・`BackendOps` メソッド・VJP なし**で既存 `Op` の合成として実装した。
  `crates/tensor-core`・`crates/backend-*`・`tape.rs`・`grad.rs` は変更していない。
- 公開面は内部クレート限定（`fandhe_ai_autodiff::tensor_product_ops`）。facade 公開は未承認で、
  承認依頼は #2677・公開自体は承認後の #2678。保留は §9 のガードで機械固定した。
- CPU 参照値は PyTorch 2.14.0 の実行値 fixture と REQ-2 統一複合判定で一致し、`f64` 総当たりオラクルの
  中心差分でも勾配が一致した。CUDA／Metal 実機の parity は未実測（§10）。

## 1. 着手時の判定（事実のみ）

- 着手前に `crates/` 配下へ `fn kron`／`tensordot`／`tensordot_axes`／`cdist`／`cross` の宣言はなく、
  モジュール名 `tensor_product_ops` も未使用だった（§9 の workspace インベントリの期待値「新モジュールの各 1 件」の根拠）。
- Phase 4 の方針は #2639（`docs/autodiff-shape-view-ops-decision.md`）と同じく「既存 Op の合成で足りる演算に
  共有カーネルを足さない」。直近の兄弟 #2631〜#2637 は `tensor-core` へカーネルを足したが、本イシューの
  4 演算は乗算・行列積・減算・ノルム・`roll` の合成で表現でき、公開済みクレート `fandhe-ai-tensor-core` の
  trait 面を無用に広げないため `crates/tensor-core` は変更しない（Issue 本文の「必要なら tensor-core」は不要と判定）。

## 2. 実装方式・合成表・バックエンド到達性

| 関数 | 合成 |
|---|---|
| `kron(a, b)` | rank を揃え（先頭へ 1 を補う）`a` を `[a0,1,a1,1,…]`・`b` を `[1,b0,1,b1,…]` へ `reshape` → `Var::mul`（broadcast）→ `[a0·b0, a1·b1, …]` へ `reshape` |
| `tensordot_axes(a, b, dims_a, dims_b)` | `a` を `permute(自由軸 ++ 縮約軸)`、`b` を `permute(縮約軸 ++ 自由軸)` → `[L,K]`・`[K,R]` へ `reshape` → `Var::matmul` → 自由軸の shape へ `reshape`（縮約なしは `K = 1`） |
| `tensordot(a, b, n)` | `dims_a = [ra-n, …, ra-1]`・`dims_b = [0, …, n-1]` を作って `tensordot_axes` の内部実装へ渡す |
| `cdist(x1, x2, p)` | `x1.unsqueeze`・`x2.unsqueeze` の差（`Var::sub`。batch 軸も broadcast）を `norm_p(…, 最終軸)` で縮約。`M == 0` は空軸の `sum` で 0 |
| `cross(a, b, dim)` | `c_i = a_{i+1} b_{i+2} − a_{i+2} b_{i+1}` を `roll` 4 回・`mul` 2 回・`sub` 1 回で組む |

恒等な `permute`／`reshape` はノードを積まない（`crate::einsum` の恒等スキップと同じ規律）。これにより rank 2 同士の
`tensordot(a, b, 1)` は `a.matmul(&b)` と tape ノード 1 個・bit 同一になる（`tests/tensor_product_parity.rs::tape_node_counts_are_fixed`・
`src/tensor_product_ops.rs` の単体テストで固定）。`tensordot` を `crate::einsum::einsum` へ委譲しないのは、`einsum` の
パーサーが rank 0 オペランド（空の添字グループ）と 52 個超の添字を受理せず、エラー文言も einsum のものになるため。

**バックエンド到達性（受入条件 2）**

| 関数 | forward が呼ぶ `BackendOps` | 既定 `Unsupported` の有無 |
|---|---|---|
| `kron` | `mul` | なし（必須メソッド。3 バックエンドとも実カーネル） |
| `tensordot` | `gemm`（backward は `gemm_fp32_strict`） | なし（同上） |
| `cdist` | `scalar_binary`（Sub）・`vector_norm`／`vector_norm_p` | あり。`vector_norm`／`vector_norm_p` を override するのは CPU のみで、CUDA／Metal 実機では「減算は GPU・ノルムはホスト参照実装」の混在になる |
| `cross` | `gather`・`mul`・`scalar_binary`（Sub）、backward で `scatter` | `gather`／`scatter`／`scalar_binary` は既定 `Unsupported` だが 3 バックエンドとも実カーネルを持つため実機では GPU が走る（#2639 の `rot90` と同じ） |

`kron`／`tensordot` は構造的にフォールバック経路を持たない。「`Unsupported` → ホスト参照実装へ到達」は `cdist` と `cross` の
モック（`ReachMock`）で固定し、`kron`／`tensordot` はフォールバック可能なメソッドを 1 回も呼ばないこと（forward・backward とも）を
固定した。加えて全関数について `Tape::new()`（`NaiveOps`。任意メソッドは既定 `Unsupported`）と CPU `BackendOps` の突合を
`crates/facade/tests/tensor_product_ops_backend_parity.rs` で行う。

**命名規律**: 新規型を持たない。`Var`／`Tape`／`Tensor` へ inherent メソッドを足さない。クレートルートからの再エクスポートもしない。

## 3. 数値契約

- `kron`: forward は乗算 1 回で bit 一致（fixture の非有限値を除く全ケースで PyTorch と bit 一致を確認）。backward は broadcast の縮約を通るため REQ-2。
- `tensordot`: `Var::matmul` と同一（FMA 契約・CUDA TF32 opt-in の挙動も同じ）。新しい丸め経路を作らない。
- `cdist`: `norm_p`（`p = 1`／`2` は `norm_l1`／`norm_l2` へ委譲）の `f64` アキュムレータ契約を継承する。
- `cross`: 乗算 2 回・減算 1 回で縮約を持たない。PyTorch 側の FMA の有無が不明なため bit 一致は主張せず REQ-2。
- tolerance・baseline は変更していない（`common::req2_close` を使用）。

## 4. 境界検査

外部から渡る軸・軸リスト・`n`・`p`・shape は、最初の `Var` 操作より前にすべて検査する（引数起因のエラーで孤児ノードを残さない。
エラー時に `tape.len()` が不変であることをテストで固定）。積は `checked_mul`、確保は `checked_bytes_for::<f32>` で事前検査し、
`broadcast_to` 由来で実体のない巨大 shape による capacity overflow panic を防ぐ。`cdist` は中間 `[…,P,R,M]` を確保するため
入力より大きなメモリを要し（O(P·R·M)）、確保前に `isize::MAX` 超過を型付きエラーにする。本番経路に `unwrap()`／`expect()`／`unsafe` はない。

## 5. PyTorch 2.14.0 との差分・実測で確定した点

fixture は `tests/fixtures/tensor-product-pytorch-reference/`（生成条件は同 README）。
意図的な差分は `tests/tensor_product_parity.rs` の `INTENDED_DIFFS` と一対一に対応する。

| 項目 | PyTorch 2.14.0 の実測 | 本実装 |
|---|---|---|
| 軸 | 負の値可 | 非負の `usize` のみ |
| `cdist` の `p = 0`・`p = inf` | 成功する（`cdist_p_zero`・`cdist_p_inf`） | `InvalidArgument`（`norm_p` の契約を継承。`p` は有限かつ正のみ。負・`NaN` は torch も例外） |
| `cdist` の `compute_mode` | 既定は P か R が 25 超で行列積分解 | 差分形の総当たりのみ（`donot_use_mm_for_euclid_dist` 相当）。引数は設けない。メモリは O(P·R·M) |
| `cdist` の `M == 0` | shape `[…,P,R]` のゼロを返す（実測） | 同じ（空軸の `sum` でゼロ。`x` への計算グラフ依存を保つ） |
| `cross` の broadcast（`cross_broadcastable`） | `[2,3]` と `[1,3]` を受理する | 同 shape のみ。`InvalidArgument`。`dim` 既定 `-1` もなく明示必須 |
| `tensordot` の size 1 縮約（`tensordot_size1_vs_n`） | size 1 と N の組を broadcast して縮約する | 完全一致のみ。`InvalidArgument` |
| `kron` の非 contiguous 入力 | transpose 済み入力で `view` エラーの例外を出す（fixture に含められない） | 受理する（`contiguous` 経由）。単体テストで確認 |

PyTorch と成否が一致するエラー（`cross` の軸長・軸範囲外・形状不一致・rank 0、`cdist` の rank 1・`M` 不一致・負の `p`・batch 非整合、
`tensordot` の軸重複・長さ不一致・軸長不一致・`n` の rank 超過・軸範囲外）は型付きエラーになる。非有限値入力は forward の値クラス
（NaN 同士・同符号の inf・有限値は REQ-2）を突合し、勾配の一致は受入条件外とした（fixture の `nonfinite` ケースは forward のみ検証）。
同一点を含む `cdist`（`identical`）の勾配は PyTorch と REQ-2 で一致した（距離 0 でも有限）。

## 6. テスト構成

- 単体: `crates/autodiff/src/tensor_product_ops.rs`（既知値・rank 不一致・rank 0・軸長 0・非 contiguous・型付きエラーとノード不増加・巨大 broadcast view の確保前拒否・cross-tape 拒否・同一点の勾配）。
- fixture 突合・`error_cases`・`f64` 総当たりオラクル（forward 突合と中心差分）・モック・ノード数・bit 決定性: `crates/autodiff/tests/tensor_product_parity.rs`。
- CPU `BackendOps` と `NaiveOps` の突合・手計算期待値・`#[ignore]` 実機テスト: `crates/facade/tests/tensor_product_ops_backend_parity.rs`。

## 7. facade 公開形の推奨案（未承認）

推奨案は 1 つ。`Var` の inherent メソッド 5 件を 1 行委譲で公開する。

- `Var::kron(&self, other)`
- `Var::tensordot(&self, other, n)`
- `Var::tensordot_axes(&self, other, dims_self, dims_other)`
- `Var::cdist(&self, other, p)`
- `Var::cross(&self, other, dim)`

モジュール `tensor_product_ops` の再エクスポートはしない。新規型がないため型の再エクスポートもない。`Sequential::add_*` は設けない。
いずれも追加のみで既存の公開 API を壊さない。

**これは推奨案の記録であり承認記録ではない。** 承認依頼は #2677、公開は承認後の #2678 で行う。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）。公開時は `TensorProductOpsHoldDoctestGuard` と否定ガードを正ガードへ反転する。
- CUDA／Metal の専用カーネルと GB10／M4 Max の実機計測。
- `cdist` の `p = 0`・`p = inf`、`compute_mode`（行列積分解）、O(P·R) メモリの専用カーネル。
- `cross` の broadcast・`dim` 既定値・旧 `torch.cross` の軸推定。
- `tensordot` の size 1 縮約 broadcast、負の軸。
- `create_graph`（高階微分）・activation checkpoint・f64／f16／bf16 自動微分経路での保証。
- 非有限入力時の勾配の PyTorch 一致。
- `docs/compat-api-scope.md` 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定・spec（REQ-9）の改定、`MIN_KNOWN_PROBE_BLOCKS` の更新。

## 9. 多層防御（保留ガード）

- 正のプローブ: `crates/facade/src/lib.rs` の `TensorProductOpsHoldDoctestGuard`（全 `pub mod` を glob import したスコープで、同名の
  ローカルモジュール・関数・メソッドを持つプローブをコンパイルする。facade が同名を公開すると名前解決の曖昧性で失敗する）。
- ソース走査: `crates/facade/tests/api_surface.rs` の `tensor_product_ops_hold_doctest_globs_all_pub_modules`・
  `tensor_product_ops_hold_doctest_probe_body_matches_fixed_contract`（固定文言 `TENSOR_PRODUCT_OPS_HOLD_PROBE_BODY`）・
  `facade_does_not_reexport_or_declare_tensor_product_ops`（と自己テスト `…_detects_each_category`）・
  `workspace_declares_tensor_product_ops_fn_names_only_in_allowed_locations`（期待値は `autodiff/src/tensor_product_ops.rs` の 5 名・各 1 件）。
- **有効性の確認（#2639 決定記録 §9 と同じ手順）**: 一時的に facade へ `pub use fandhe_ai_autodiff::tensor_product_ops;` を足すと doctest が名前解決の
  曖昧性（E0659）で、否定ガード `facade_does_not_reexport_or_declare_tensor_product_ops` が違反検出で落ちた。`Var` へ
  `pub fn kron(&self, _o: &Var<'t>)` を足すと doctest が呼び出しシグネチャ不一致（E0061・E0308）で、workspace インベントリが
  不一致で落ちた。いずれも元へ戻して green を確認した。

## 10. 実機申し送り

CUDA／Metal の `#[ignore]` テストは未実測。測定コマンド・期待結果・記入欄は `docs/perf/logs/tensor-product-ops-2640/README.md`。
REQ-2 を外れた場合は tolerance を緩めず事実を記録する。

## 11. 出典

- Issue #2640・親 #2625・ルート #2499。
- 同型の先行記録: `docs/autodiff-shape-view-ops-decision.md`（#2639）。
- fixture の出自: `crates/autodiff/tests/fixtures/tensor-product-pytorch-reference/README.md`。
- 契約: `.claude/rules/coding-rust.md`（FMA 契約・`f64` アキュムレータ・境界検査）・`.claude/rules/security.md`。
