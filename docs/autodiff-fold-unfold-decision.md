# Fold と Unfold（`F.fold`／`F.unfold`）の CPU 実装記録（イシュー #2645）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-conv-transpose3d-max-unpool-decision.md`（#2644）・
`docs/autodiff-pool3d-ops-decision.md`（#2643）と同じ「内部クレートへ CPU 参照実装を先行し、facade 公開は
記録 → 承認の 2 段」方針の実装記録であり、**承認記録ではない**（facade 公開形の承認は #2677 で依頼中。公開自体は
承認後の #2678・#2679）。`im2col`／`col2im` の意味論は `docs/conv-ops-design.md` が正で、本書は Fold／Unfold への
転用の決定・差分だけを記録する。

## 0. 結論

- `F.unfold`／`F.fold`（`nn.Unfold`／`nn.Fold`）に相当する「スライディング窓の列展開」と「列からの畳み戻し」の
  順伝播と VJP を CPU 参照実装として内部クレートへ追加した。
  - 新規 `BackendOps` メソッド・新規カーネルなし。既存の `BackendOps::im2col`／`col2im`（Conv2d 用。CPU #1764・
    CUDA #1766・Metal #1768）を `groups = 1` で再利用し、`[N, 1, C·kH·kW, L]` ⇄ `[N, C·kH·kW, L]` の reshape だけを挟む。
  - 入口は `fandhe_ai_autodiff::fold_ops::{unfold, fold}`。専用 `Op::Unfold`／`Op::Fold` と VJP（互いの随伴）は
    `tape.rs`／`grad.rs`。shape 検査は `fandhe_ai_tensor_core::fold::{unfold_out_shape, fold_out_shape}`。
- facade 公開は行わない。`FoldUnfoldHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した（§9）。
- 層化（`nn::Fold`／`nn::Unfold`・`Module` impl・`Sequential::add_fold`／`add_unfold`・保存復元フック・resident 経路）は
  #2679 の対象で、本イシューでは実装していない（§8）。イシュー題名の「層として」は、内部クレートの自由関数と
  内部 `Op` までを範囲とした。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の新規カーネルは対象外。`im2col`／`col2im` は CUDA・Metal が既に専用カーネルで override 済みのため、
  実機 tape では既存カーネルを通る。ホストフォールバック（`eval::im2col`／`col2im`）へ落ちるのは `im2col`／`col2im` を
  override しないバックエンド（`NaiveOps`・テスト用モック）である。実機テストは `#[ignore]` のまま未実測（§10）。

## 1. 着手時の判定（事実のみ）

- `Fold`／`Unfold` は `docs/conv-ops-design.md` でスコープ外のまま残っていた項目で、`BackendOps::im2col`／`col2im` は
  Conv2d の forward／VJP 用に 3 バックエンドへ揃っている。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、#2645 の
  受入条件（内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について承認済みとは記録しない
  （承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| shape 検査（tensor-core） | `tensor-core/src/fold.rs` | `unfold_out_shape`（`[N,C,H,W]` → `[N, C·kH·kW, L]`）・`fold_out_shape`（`[N, C·kH·kW, L]` と `output_size` → `[N,C,H,W]`） |
| autodiff | `autodiff/src/fold_ops.rs` | 自由関数 `unfold`・`fold`・共通ヘルパー（非公開） |
| autodiff | `tape.rs`・`grad.rs` | `Op::Unfold`／`Op::Fold` と VJP の腕 2 つ |

- 新規 `BackendOps` メソッドを足さない理由: `im2col`（純コピー）と `col2im`（`f64` アキュムレータ）がそのまま
  Unfold／Fold の forward であり、K 軸が `(c, kh, kw)` row-major・P 軸が `(oh, ow)` row-major という並びが
  PyTorch の `unfold` と同じであることを fixture で実証した。
- `Op::Conv2d` を one-hot 重みで流用する案は採らない: GEMM を通るため非有限値・符号付きゼロのビットが保存されない。
- VJP は互いの随伴: `Unfold` の VJP は `col2im`（`upstream` を `[N,1,K,L]` へ reshape）、`Fold` の VJP は
  `im2col`（`upstream.shape()` から col shape を再計算し `[N,K,L]` へ reshape）。`Fold` は `output_size` を `Op` に
  保持しない。VJP でも shape を再検証し、契約違反は panic ではなく `AutodiffError::Backward` で返す。空（`N = 0` 等）は
  バックエンドを呼ばずゼロ勾配（テストで固定）。
- shape 検査を tensor-core の自己完結モジュールに置く理由: `fold` の `output_size` は入力テンソルの大きさと独立に
  呼び出し側が指定でき、ホストフォールバックは出力バッファを無検査で確保する。確保前検査に必要な
  `checked_numel_for` は tensor-core 内 `pub(crate)` のため autodiff 側では書けない（`conv_transpose3d.rs` と同じ事情）。
  また `ops_shape.rs` へ追記しないのは兄弟イシューとの同一ファイル競合回避のため（#2643・#2644 と同じ判断）。
- 引数順は crate 内の既存規約（`Var::conv2d` 系の `stride, padding, dilation`）に揃えた。PyTorch は
  `(kernel_size, dilation, padding, stride)` 順だが、Rust にはキーワード引数がなく同型 `[usize; 2]` の取り違えは型で
  検出できないため、crate 内の一貫性を優先した（§7 の承認事項）。
- 命名規律: 素の `fn fold`／`fn unfold` は `autodiff/src/fold_ops.rs` の各 1 件のみ（workspace インベントリが固定。
  `add_fold`／`add_unfold` は 0 件）。`Var`／`Tape`／`Tensor` に inherent メソッドは足していない。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応・f64／低精度経路なし。

## 3. 数値契約

- `unfold` の forward（`im2col`）は算術を含まないコピーなので **bit 一致**（NaN／±inf／`-0.0` も保存）。`fold` の
  d_input（= `im2col`）も同様。
- `fold` の forward と `unfold` の d_input（= `col2im`）は重なる窓の加算を `f64` アキュムレータで行い 1 回だけ `f32`
  へ downcast する（`.claude/rules/coding-rust.md` の長軸縮約契約）。PyTorch は f32 累積のため、窓が重なる設定では
  bit 一致ではなく REQ-2 統一複合判定（`common::req2_close`）で突合する。窓が重ならない設定（各出力位置への寄与が
  高々 1 つ）は加算順が結果に影響しないため bit 一致で突合する。判定の区分は fixture 生成時に PyTorch 側で
  `fold(ones).max() > 1` を実測して `overlapping` として記録しており、再生成で結果が変わる厳格化を運任せで入れていない。
- CPU tape（`CpuBackendOps`）と NaiveOps tape（ホストフォールバック）の突合は bit 一致（`im2col`／`col2im` の既存
  契約。`fold_unfold_backend_parity.rs`）。

## 4. 境界検査

- 検査順（`unfold`）: `Conv2dParams::new`（kernel／stride／dilation の 0・`2·padding` オーバーフロー拒否）→
  `unfold_out_shape`（`groups == 1`・rank 4・空間軸 0・窓数 `L`・col 側の確保サイズ）→ 実体化 → `im2col` →
  戻り shape 再検証 → `push_eager`。
- 検査順（`fold`）: `Conv2dParams::new` → `fold_out_shape`（rank 3 → `groups == 1` → `K % (kH·kW) == 0` →
  `output_size` の 0 拒否 → **出力要素数・バイト数の確保前検査（`checked_numel_for`）** → `P == L`）→ 実体化 →
  `col2im` → 戻り shape 再検証 → `push_eager`。
- 検査はすべて tape 操作より前に終え、エラー時に孤児ノードを残さない（テストで `tape.len()` を固定）。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape が期待と異なる場合は `BackendError::ShapeMismatch` で拒否する。
- 確保前検査の範囲: `checked_numel_for` の上限は `isize::MAX` バイト（リポジトリ全体の既存基準）で、実メモリ量を
  超える確保までは見ない。バイト数が `usize` を超える `output_size`（例: `[2^31, 2^31]`）は
  `ElementCountOverflow` で確保前に拒否される（テストで固定）。実メモリを超えるがオーバーフローしない巨大
  `output_size` を弾く追加の上限は新しい方針判断になるため、本イシューでは導入しない（§8）。
- カーネル側の手動境界検査（既存 `im2col`／`col2im` の符号安全な座標計算）は変更・省略していない（REQ-8）。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/fold-unfold-pytorch-reference/`。f32 は
u32 ビットパターンで保存。`torch.set_num_threads(1)`・固定シード 2645 で生成。2 回生成して同一 sha256 を確認）。
unfold 10 ケース・fold 11 ケース（重なる 6・重ならない 5）・非有限 4 ケース・エラー 19 ケースで、下表の差分を除き
一致した。

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 窓数 `L` の式 | `∏ floor((size + 2·pad − dil·(k−1) − 1)/stride) + 1` | 同じ | 一致 |
| K 軸・L 軸の並び | `(c, kh, kw)`／`(oh, ow)` row-major | 同じ | 一致（fixture で bit 一致） |
| 入力 rank | バッチなし（`unfold` は `[C,H,W]`・`fold` は `[K,L]`）も受理（実測） | バッチ入力のみ | 差分（スコープ外） |
| `N = 0` | 受理（実測） | 受理（空出力。VJP はゼロ勾配） | 一致 |
| `C = 0`（`K = 0`） | 拒否（実測） | 受理（空出力） | 差分（`N = 0` と一貫させた。#2644 の MaxUnpool と同じ扱い） |
| kernel／stride／dilation の 0 | 拒否（実測） | 拒否 | 一致 |
| カーネルが padded 入力より大きい（`unfold`） | 拒否（実測） | 拒否 | 一致 |
| `K` が `kH·kW` で割り切れない／`L` 不一致／`output_size` が小さすぎる／`output_size` に 0（`fold`） | 拒否（実測） | 拒否 | 一致 |
| rank 不一致（`unfold` rank 5・`fold` rank 4） | 拒否（実測） | 拒否 | 一致 |
| 引数順 | `(kernel_size, dilation, padding, stride)` | `(kernel_size, stride, padding, dilation)`（crate 内の `conv2d` 系に揃えた） | 差分（§7 の承認事項） |
| `torch.Tensor.unfold`（次元方向の窓） | 別演算 | なし | スコープ外 |
| 累積精度（`fold`・`unfold` の d_input） | f32 累積 | `f64` | 重なる設定は REQ-2 判定で比較（重ならない設定は bit 一致） |
| 非有限値 | 伝播 | 伝播（コピー系はビット保存。NaN はクラス一致） | 一致 |

tolerance・baseline は変更していない。PyTorch との差分を埋めるために判定を緩めていない。

## 6. テスト構成

- `crates/tensor-core/src/fold.rs`（単体 10 件）: PyTorch 式の例・非等方＋dilation・`N = 0`・rank 不一致・`K` 非整除・
  `L` 不一致・空間軸 0・カーネル超過・`groups != 1`・巨大 `output_size` の確保前拒否。
- `crates/autodiff/src/tape.rs`: `Op::Unfold`／`Op::Fold` のメタ性質（非 checkpoint・非 create_graph・
  `for_each_input`）。`fold_ops.rs`: 窓の並び・重なり加算・引数エラーと孤児ノード無し・空バッチの単体。
- `crates/autodiff/tests/fold_unfold_parity.rs`（18 件）: fixture 突合（unfold・fold・非有限・エラー表）・中心差分
  （`tests/conv_transpose3d_parity.rs` と同じ `H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`。緩和なし）・随伴恒等式・
  `fold(unfold(x)) = x ⊙ fold(unfold(1))`・非連続入力・引数エラーと孤児ノード無し・`N = 0`・巨大 broadcast view と
  巨大 `output_size` の確保前拒否・テープ記録数・`create_graph` の型付きエラー・決定性・モック `BackendOps`
  （`im2col`／`col2im` の `Unsupported` フォールバック到達〈forward・VJP 両方〉・他エラー伝播・誤 shape）。
- `crates/facade/tests/fold_unfold_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし・bit 一致）。
  CUDA／Metal 実機は `#[ignore]`。
- `crates/backend-cpu/tests/backend_ops_dispatch.rs` に「`*_are_unsupported_not_panic`」型のテストは**追加しない**:
  `im2col`／`col2im` は CUDA・Metal が専用カーネルで override 済みで、GPU 非搭載環境では `CudaUnavailable` を返す
  （`Unsupported` ではない）ため、#2644 の同型テストを写すと失敗する。「`Unsupported` を返すバックエンドではホスト計算へ
  到達する」ことはモック `BackendOps` で固定した。

## 7. facade 公開形の推奨案（未承認）

> #2849 で名前と引数順を確定した（§12）。公開は #2851。本節は確定前の推奨案として履歴を残す。

推奨は 1 つ。`Var` の inherent 委譲メソッド 2 件として各自由関数への 1 行委譲で公開する（#2643・#2644 の推奨形と
同じ方式）。

- `Var::unfold(&self, kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2], dilation: [usize; 2])
  -> Result<Var<'t>, AutodiffError>`
- `Var::fold(&self, output_size: [usize; 2], kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2],
  dilation: [usize; 2]) -> Result<Var<'t>, AutodiffError>`
- `fold_ops` モジュールと `tensor_core::fold` は再エクスポートしない。新規公開型なし。inherent メソッドの追加のみで
  非破壊。
- 承認後は `FoldUnfoldHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

層化（推奨案には含めない。#2679 の対象）: 内部層型 `nn::Fold`／`nn::Unfold`、`Module` impl、保存・復元用の `as_*` フック、
`compat::Sequential::add_fold`／`add_unfold`、resident 経路の fail-closed 拒否。

承認事項（**すべて未承認**）:

- 上記 2 メソッドの公開。
- **メソッド名**: `Var::unfold` は `torch.Tensor.unfold`（次元方向のスライディング窓。別演算）と意味が異なる。
  代替名（`unfold2d`／`im2col` 等）にするか。
- **引数順**: crate 内の `conv2d` 系（`stride, padding, dilation`）に揃えるか、PyTorch（`dilation, padding, stride`）に
  揃えるか。
- **PyTorch との差分の扱い**: バッチなし入力の非対応・`C = 0` の受理（§5）。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）と層化（#2679）。
- CUDA／Metal の GPU 専用カーネルの新設と実機計測（§10）。
- バッチなし入力・1 軸／3 軸の fold／unfold・`torch.Tensor.unfold`（次元方向の窓）・channels_last・低精度（f16／bf16）・
  f64 自動微分・`create_graph`・activation checkpoint・ONNX の import／export。
- 実メモリを超えるがオーバーフローしない巨大 `output_size`（および他の出力サイズ依存演算）を弾く確保上限の方針。
  現行基準（`checked_numel_for` の `isize::MAX` バイト）はリポジトリ全体の既存挙動で、変更には別途判断が要る。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定。
- ruleset・branch protection・リポジトリ設定の変更（`ci.yml` のジョブ追加・リネームは行っていないため required
  contexts の更新も不要）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `FoldUnfoldHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>`（`fold`／`unfold`）と `compat::Sequential`（`add_fold`／`add_unfold`）に公開されるとコンパイルが失敗する正のプローブ |
| `fold_unfold_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `fold_unfold_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_fold_unfold`（＋自己テスト） | facade src の再エクスポート・`pub mod fold_ops`／`fold`・型 `Fold`／`Unfold` の独自宣言・4 名の `fn` 宣言の否定検査。検出はトークン完全一致のみで、`fold_bits`／`try_fold`／`fold_sgd_step_config_key` 等の別トークン・`.fold(` の呼び出し・コメント／文字列・非公開 `use` を誤検出しないことを自己テストで固定 |
| `workspace_declares_fold_unfold_fn_names_only_in_allowed_locations` | workspace 全体で `fn fold`／`fn unfold` の宣言が `autodiff/src/fold_ops.rs` の各 1 件のみ・`add_fold`／`add_unfold` は 0 件 |

検出範囲は上記トークン列（`pub use` の経路・型／`pub mod`／`fn` の宣言）と、プローブが名前解決で触れる位置に限る。
マクロ生成や別名経由のメソッドまでは保証しない。stable rustdoc は `compile_fail` のコードを照合しないため、
否定ガードは正のプローブ＋インベントリで組んでいる。実装中に `Var` へ同名メソッド `fold` を一時的に足し、doctest が
型不一致で失敗し workspace インベントリが不一致で失敗することを手元で確認した（確認後に元へ戻した）。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_fold_unfold_match_cpu_reference`・
`metal_fold_unfold_match_cpu_reference`）は `#[ignore]` のまま未実測。手順・期待結果は
`docs/perf/logs/fold-unfold-2645/README.md`。期待結果は「既存の `im2col`／`col2im` GPU カーネルを通る経路が CPU tape と
bit 一致する」ことの確認であり、新規 GPU カーネルの parity ではない。

## 11. 出典

- `docs/conv-ops-design.md`（`im2col`／`col2im` の設計・`col2im`〈fold〉が転置畳み込みそのものであることの式による説明）、
  `docs/autodiff-conv-transpose3d-max-unpool-decision.md`（#2644。保留ガード・決定記録の構成）、
  `docs/autodiff-pool3d-ops-decision.md`（#2643）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/fold-unfold-pytorch-reference/README.md`

## 12. #2849 決定記録（facade 公開形の確定）

- 状態: **記録のみ（コード変更なし）。** §7 で未決だった `Var::unfold` の名前と引数順を 1 案に確定する。公開（`Var::unfold`／`fold` の委譲と保留ガードの反転）は #2851 が行う。**公開までは `FoldUnfoldHoldDoctestGuard` と `api_surface.rs` の否定ガードを維持する。**
- 基準: `origin/main` `74171fb9`（2026-10-08）。
- 承認の根拠: ルート #2499 の 2026-10-08 ユーザーコメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）が明示した点は、(1) 行 12〜15 は `Var` の委譲メソッドに限って公開する、(2) `Var::unfold` は `nn.functional.unfold` に対応する名前、(3) 引数順は crate 内の `conv2d` 系、(4) 層化（`Sequential::add_*` 等）は保留継続、の 4 点。バッチなし入力の非対応・`C = 0` の受理を現行のまま維持する点は、§7 の現行形を変えない方向からの導出である。

### 12.1 確定シグネチャ

```rust
impl<'t> Var<'t> {
    pub fn unfold(
        &self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Var<'t>, AutodiffError>;

    pub fn fold(
        &self,
        output_size: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Var<'t>, AutodiffError>;
}
```

- 内部自由関数 `crates/autodiff/src/fold_ops.rs` の `unfold`／`fold` と、第 1 引数 `input` を `self` に置き換えた以外の引数順・型・戻り値が一致する。1 行委譲。新規公開型なし。`fold_ops`・`tensor_core::fold` は再エクスポートしない。

### 12.2 名前と引数順

- 名前は `Var::unfold`／`Var::fold`。代替名（`unfold2d`／`im2col`）は採らない。`torch.nn.functional.unfold`（`nn.Unfold`）に対応し、`torch.Tensor.unfold`（次元方向のスライディング窓。別演算）とは異なる。
- 引数順は crate 内の `conv2d` 系（`Var::conv2d` の `stride, padding, dilation`、`Var::max_pool2d` の `kernel_size, stride, padding, dilation`）に揃え、`kernel_size, stride, padding, dilation`（`fold` は先頭に `output_size`）とする。PyTorch の `dilation, padding, stride` 順は採らない。
- `Var::unfold` の doc コメント先頭に置く文意: 「`torch.nn.functional.unfold`（`nn.Unfold`）相当の列展開。`torch.Tensor.unfold`（次元方向のスライディング窓）とは別の演算」。#2851 がこの文意で書く。
- バッチなし入力の非対応・`C = 0` の受理は §5 の現行のまま。

### 12.3 保留を続けるもの

- 層化（`nn::Fold`／`nn::Unfold`・`Module` impl・保存復元フック・`compat::Sequential::add_fold`／`add_unfold`・resident 経路）。

### 12.4 #2851 への申し送り

- 反転するのは `Var` の委譲メソッド名（`fold`／`unfold`）のプローブだけ。`Tape`／`Tensor<f32>` 上の同名メソッド・`compat::Sequential::add_*`・モジュール再エクスポート・型名のプローブは未承認経路として維持する。
- workspace インベントリは `var.rs` の委譲 1 件ずつを許可位置に加える。
- 実機 parity は既存の `docs/perf/logs/fold-unfold-2645/README.md` を使う。
