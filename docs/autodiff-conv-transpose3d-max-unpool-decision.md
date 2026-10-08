# ConvTranspose3d と MaxUnpool（1d／2d／3d）の CPU 実装記録（イシュー #2644）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-pool3d-ops-decision.md`（#2643）と同じ「内部クレートへ
CPU 参照実装を先行し、facade 公開は記録 → 承認の 2 段」方針の実装記録であり、**承認記録ではない**（facade
公開形の承認は #2677 で依頼中。公開自体は承認後の #2678・#2679）。2D 転置畳み込みの意味論は
`docs/conv-ops-design.md` §15、MaxPool 側の索引の意味論は `docs/pooling-ops-design.md` が正で、本書は 3 軸への
一般化と MaxUnpool の決定・差分だけを記録する。

## 0. 結論

- ConvTranspose3d（`F.conv_transpose3d`／`nn.ConvTranspose3d` 相当）と MaxUnpool1d／2d／3d
  （`F.max_unpool1d/2d/3d` 相当）の順伝播と VJP を CPU 参照実装として内部クレートへ追加した。
  - ConvTranspose3d: 新規 `BackendOps` メソッド・新規カーネルなし。既存フック（`gemm_batched`・`col2im3d`・
    VJP の `im2col3d`・`gemm_batched_fp32_strict`）の合成。入口は `fandhe_ai_autodiff::conv_transpose3d_ops::
    conv_transpose3d`。専用 `Op::ConvTranspose3d` と VJP は `grad.rs`。shape 検査は
    `fandhe_ai_tensor_core::conv_transpose3d::conv_transpose3d_out_shape`。
  - MaxUnpool: 新規 `BackendOps` メソッドなし。既存の `scatter`（forward。`ScatterReduce::Overwrite`）と
    `gather`（VJP）を `(n, c)` 平面へ平坦化して再利用。入口は `fandhe_ai_autodiff::max_unpool_ops::
    max_unpool{1,2,3}d`。専用 `Op::MaxUnpool`（1 variant を 1d／2d／3d で共有）。shape 検査は
    `fandhe_ai_tensor_core::max_unpool::{max_unpool_layout, MaxUnpoolLayout}`。
- facade 公開は行わない（**→ `Var` 委譲 4 本は #2850 で公開済み。§13**）。`ConvTranspose3dMaxUnpoolHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した（§9）。
- 層化（`nn::ConvTranspose3d`／`nn::MaxUnpool*`・`Module` impl・`Sequential::add_*`・保存復元フック）は #2679 の
  対象で、本イシューでは実装していない（§8）。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。ConvTranspose3d は `gemm_batched` がデバイス上で走り `im2col3d`／
  `col2im3d` は既定 `Unsupported` → ホスト、MaxUnpool は既存の GPU `scatter`／`gather` が走る。実機テストは
  `#[ignore]` のまま未実測（§10）。

## 1. 着手時の判定（事実のみ）

- ConvTranspose3d は `docs/conv-ops-design.md` §16.8、MaxUnpool は `docs/pooling-ops-design.md` §11 で
  スコープ外として残っていた項目である。MaxPool 側は索引を `Tensor<i32>`（`(n, c)` 平面内 flat 添字）として
  返しており（`Var::max_pool1d`／`max_pool2d`・`adaptive_max_pool*`・`pool3d_ops::max_pool3d`）、MaxUnpool は
  それをそのまま受ける。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、
  #2644 の受入条件（内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について承認済みとは
  記録しない（承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| shape 検査（tensor-core） | `tensor-core/src/conv_transpose3d.rs` | `conv_transpose3d_out_shape`（`conv_transpose2d_out_shape` の 3 軸版。既存 `conv_transpose_out_len` を軸ごとに再利用） |
| shape 検査（tensor-core） | `tensor-core/src/max_unpool.rs` | `max_unpool_layout`・`MaxUnpoolLayout`（rank・索引 shape・既定出力長・`output_size` 範囲・確保サイズ。**索引値は検査しない**） |
| autodiff | `autodiff/src/conv_transpose3d_ops.rs` | 自由関数 `conv_transpose3d`・forward 合成ヘルパー（非公開） |
| autodiff | `autodiff/src/max_unpool_ops.rs` | 自由関数 `max_unpool1d`／`2d`／`3d`・共通本体（非公開） |
| autodiff | `tape.rs`・`grad.rs` | `Op::ConvTranspose3d`／`Op::MaxUnpool` と VJP の腕 2 つ |

- 新規 2 モジュールを `ops_shape.rs`・`backend_ops.rs` へ追記せず自己完結で置いた理由: 並列実行中の兄弟イシュー
  （#2645 以降）との同一ファイル競合を避けるため（#2643 と同じ判断）。`tensor-core/src/lib.rs` は `pub mod` 2 行のみ。
- ConvTranspose3d の forward 合成（`Op::ConvTranspose2d` の空間 3 軸一般化）:
  `w_mat = weight.reshape([G, Cin_g, K_g])`・`x5 = input.reshape([N, G, Cin_g, D·H·W])`・
  `d_col = gemm_batched(w_matᵀ, x5)`・`out = col2im3d_with_fallback(d_col, 出力 shape)`・bias は
  `[1, Cout, 1, 1, 1]` へ reshape して `add`。`col2im3d` の第 3 引数は「転置畳み込みの**出力** shape」（誤読注意の
  コメントを残した）。`N = 0` は GEMM／col2im3d を呼ばず空テンソルを返す。
- VJP（`col` を保持せず `im2col3d` を再計算。`gemm_batched_fp32_strict`＝TF32 非追従の既存方針。
  `conv3d_with_fallback` は直接呼ばない）:
  `d_input = gemm_batched_fp32_strict(w_mat, im2col3d(upstream))`、
  `d_weight = Σ_N gemm_batched_fp32_strict(x5, im2col3d(upstream)ᵀ)`（N 軸は `reduce_batch_axes_f64`）、
  `d_bias = eval::reduce_bias_grad_rows`（`f64` 逐次和・1 回 downcast）。`N = 0` はゼロ勾配。
- MaxUnpool の forward: 入力・索引を `[N·C, L_in]` へ平坦化し、`zeros([N·C, L_out])` へ
  `scatter_with_fallback(dim = 1, Overwrite)`。算術を含まないコピーなので値は入力要素と bit 一致（NaN／inf も保存）。
  索引の意味論は `(n, c)` 平面内 flat 添字（1d `w`／2d `h·W_out + w`／3d `d·H_out·W_out + h·W_out + w`。
  `W_out` 等は**出力**の寸法）。
- 命名規律: 素の `fn conv_transpose3d`／`fn max_unpool1d`／`fn max_unpool2d`／`fn max_unpool3d` は
  上記ファイルの各 1 件のみ（workspace インベントリが固定。`add_conv_transpose3d`／`add_max_unpool*` は 0 件）。
  `Var`／`Tape`／`Tensor` に inherent メソッドは足していない。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応・f64／低精度経路なし。
- 層型を作らない理由: 層化は #2679 に割り当てられており受入基準にも含まれない。追加すると `nn/module.rs` の型列挙
  テストが兄弟イシューと競合する。

## 3. 数値契約

- ConvTranspose3d: GEMM は forward が `gemm_batched`（CUDA TF32 opt-in に追従）・VJP が
  `gemm_batched_fp32_strict`（FMA 契約）。`col2im3d` は既存の `f64` アキュムレータ契約、bias 勾配は `f64` 逐次和・
  1 回 downcast、`d_weight` の N 軸縮約は `f64`（`.claude/rules/coding-rust.md`）。PyTorch とは総和順が異なるため
  fixture 突合は REQ-2 統一複合判定（bit 一致は求めない）。
- MaxUnpool: forward は bit 一致（コピー）。VJP は `gather`（コピー）× last-writer マスク（0／1 の選択）で算術を
  含まず bit 一致。非有限入力は拒否せず伝播する。

## 4. 境界検査

- ConvTranspose3d: 検査順は同一 tape → `weight` rank 5 → `Conv3dParams::new`（`stride`／`dilation`／`groups` の 0・
  `2·padding` オーバーフロー）→ `output_padding < stride`（各軸。違反は `AutodiffError::InvalidArgument`）→
  `conv_transpose3d_out_shape`（rank・チャンネル整合・空間軸 0・各軸出力長・確保サイズ）→ bias shape →
  実体化 → 合成 → 戻り shape 再検証 → `push_eager`。
- MaxUnpool: `max_unpool_layout`（空間 rank と引数長・入力 rank・`index.shape() == input.shape()`・kernel／stride の 0・
  空間軸 0・既定出力長〈`checked_*`・結果 1 以上〉・`output_size` 範囲・`N·C`／平面長・確保サイズ）→ **索引値の範囲検査**
  （負値・`>= 出力平面長` は `AutodiffError::InvalidArgument`。`Var::scatter` と同じ慣例）→ 実体化 →
  `scatter` → 戻り shape 再検証 → `push_eager`。VJP 側でも shape・索引範囲を再検証し、契約違反は panic ではなく
  `AutodiffError::Backward` で返す（`eval::scatter`／`gather` の手動境界検査は省略しない。REQ-8）。
- 検査はすべて tape 操作より前に終え、エラー時に孤児ノードを残さない（テストで `tape.len()` を固定）。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape が期待と異なる場合は `BackendError::ShapeMismatch` で拒否する。
- 巨大 `output_size` 相当（巨大 stride）・巨大 broadcast view は確保前に `ElementCountOverflow` で拒否される。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/conv-transpose3d-max-unpool-
pytorch-reference/`。f32 は u32 ビットパターンで保存。`torch.set_num_threads(1)` で生成）。ConvTranspose3d は
10 ケース（forward・`d_input`・`d_weight`・`d_bias`）が REQ-2 統一複合判定で、MaxUnpool は 20 ケース
（通常 12・重複 5・非有限 3）が bit 一致（NaN はクラス一致）で下表の差分を除き一致した。

ConvTranspose3d:

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| `output_padding` の上限 | `max(stride, dilation)` 未満まで受理（実測: stride=1・dilation=2・op=1 は受理。stride=1・op=1 は拒否） | **`stride` 未満のみ**（各軸） | 意図的な差分。`col2im3d` の P 軸契約（`conv_out_len(出力) = 入力長`）を保つため（`docs/conv-ops-design.md` §15 と同じ） |
| 入力 rank | rank 4（バッチなし）を受理（実測） | rank 5 のみ | 差分 |
| `N = 0` | 受理（実測） | 受理（空出力。VJP はゼロ勾配） | 一致 |
| チャンネル・groups 不整合・stride／dilation／groups の 0・過大 padding・bias shape | 拒否（実測） | 拒否 | 一致 |
| `padding_mode`・`output_size` 引数・channels_last | あり | なし | 差分（スコープ外） |
| 累積精度 | f32 累積 | GEMM FMA・`col2im3d` は `f64` | REQ-2 判定で比較 |

MaxUnpool:

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 既定出力長 | `(in − 1)·stride − 2·padding + kernel` | 同じ | 一致 |
| `output_size` の許容範囲（空間軸のみ指定） | 各軸 `default − stride < size < default + stride`（開区間。実測: default=6・stride=2 で 5・7 は受理、4・8・3 は拒否） | 同じ（`size == 0` も拒否） | 一致 |
| `output_size` の形 | 空間軸のみ、または先頭 2 軸込みの全 shape | 空間軸のみ | 差分（全 shape 形は非対応） |
| 索引範囲外・負値・索引 shape 不一致 | 拒否（実測） | 拒否（範囲外・負値は `InvalidArgument`、shape は `Shape`） | 一致 |
| 入力 rank | バッチなし（rank 2／3／4）も受理（実測） | `空間 rank + 2` のみ | 差分 |
| `kernel = 0` | 受理して空出力（実測） | 拒否 | 差分 |
| `stride = 0` | 索引不整合で拒否（実測） | 拒否 | 一致（拒否） |
| `N = 0` | 受理（実測） | 受理（空出力） | 一致 |
| `C = 0` | 拒否（実測） | 受理（空出力） | 差分（`N = 0` と一貫させた。`pool3d` の `C = 0` と同じ扱い） |
| 重複索引の forward | 複数スレッド実行では書き込みが競合し**勝者が実行ごとに変わる**（実測: 同一入力で out の勝者が 3.0／4.0 に揺れた）。単一スレッドでは最後の書き手 | `ScatterReduce::Overwrite` の決定的契約（row-major 走査で最後の書き手が残る） | 本実装の契約を維持。fixture は単一スレッド生成のため全位置が一致。重なり窓のプール由来の重複索引は同一要素由来で値が同じため全位置一致 |
| 重複索引の勾配 | `grad_input = gather(grad_output, index)`（**全書き手**へ上流を配る） | forward の真の随伴: **最後の書き手だけ**が上流を受け、敗者位置は 0 | **意図的な差分**（下記） |
| 非有限値（NaN／±inf） | ビット保存（コピー） | ビット保存 | 一致 |

重複索引の勾配（意図的な差分）: 本実装は `Op::Scatter { Overwrite }` の既存 VJP と同一規則（`gather` × last-writer
マスク）で、forward（上書き）の真の随伴であり中心差分（forward は入力の線形コピーなので厳密）とも一致する。
PyTorch は全書き手へ上流を配るため、pool → unpool の合成（重なり窓＝stride < kernel のプール由来の索引）では
同じ勝者要素へ勾配を重複計上する。**差が出るのは重複索引の「敗者」位置だけ**で、勝者位置と索引が重複しない
通常ケース（stride >= kernel のプール由来）は PyTorch と bit 一致する（`max_unpool_parity.rs` が勝者位置の bit
一致・敗者位置 0・PyTorch 側の敗者勾配が非 0 であることを固定）。この規則（真の随伴か、PyTorch 互換の単純
gather か）は §7 の未承認の設計判断として列挙する。

tolerance・baseline は変更していない。PyTorch との差分を埋めるために判定を緩めていない。

## 6. テスト構成

- `crates/tensor-core/src/conv_transpose3d.rs`（単体 8 件）・`max_unpool.rs`（単体 7 件）: PyTorch 式の例・
  非等方・groups・`op >= stride` 拒否・rank／チャンネル／groups 不整合・空間軸 0・`N = 0`・負の出力長・
  オーバーフロー（確保前拒否）、`output_size` の開区間境界・引数長不一致・kernel／stride の 0。
- `crates/autodiff/src/tape.rs`: `Op::ConvTranspose3d`／`Op::MaxUnpool` のメタ性質（非 checkpoint・非 create_graph・
  `for_each_input`）。`conv_transpose3d_ops.rs`／`max_unpool_ops.rs`: 基本動作・bias 軸・引数エラーと孤児ノード無し・
  空バッチの単体。
- `crates/autodiff/tests/conv_transpose3d_parity.rs`（19 件）: fixture 突合（10 ケース・全勾配）・エラー表
  （PyTorch 実測との一致と差分の固定）・中心差分（`tests/conv_transpose2d.rs` と同じ `H=1e-3`・相対 1e-2 または
  絶対 1e-3・`τ=1e-4`。緩和なし）・随伴恒等式（`conv3d` の VJP／forward との相互関係）・per-group `narrow`＋
  `groups=1` の合成・bias の `Cout` 軸回帰・出力 shape 式・引数エラーと孤児ノード無し・`N = 0`（ゼロ勾配）・
  巨大 broadcast view の確保前拒否・テープ記録数・`create_graph` の型付きエラー・決定性・モック `BackendOps`
  （`col2im3d`＝forward／`im2col3d`＝VJP の `Unsupported` フォールバック・他エラー伝播・誤 shape）。
- `crates/autodiff/tests/max_unpool_parity.rs`（18 件）: fixture 突合（索引が重複しない 8 ケースは forward・勾配とも
  bit 一致／重なり窓 7 ケースは forward bit 一致・勝者位置の勾配一致・敗者位置 0／手作り重複は独立オラクルと手計算／
  非有限 3 ケース）・エラー表・中心差分（重複索引を含む）・`max_pool1d`／`max_pool2d`／`max_pool3d` とのラウンド
  トリップ・非連続入力／索引・`stride = None`・引数エラーと孤児ノード無し・空バッチ／空チャンネル・巨大出力の確保前拒否・
  テープ記録数・`create_graph`・決定性・モック `BackendOps`（`scatter`／`gather` のフォールバック・他エラー・誤 shape）。
- `crates/backend-cpu/tests/backend_ops_dispatch.rs`: CUDA（macOS では Metal も）の `im2col3d`／`col2im3d` が
  `Unsupported` を返し panic しないこと。
- `crates/facade/tests/conv_transpose3d_max_unpool_backend_parity.rs`: CPU tape（`CpuBackendOps`）と NaiveOps tape の
  突合（属性なし。ConvTranspose3d は REQ-2 判定・MaxUnpool は重複索引込みで bit 一致）。CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

> #2849 で公開形と設計判断 3 件を確定した（§12）。公開は #2850。本節は確定前の推奨案として履歴を残す。

推奨は 1 つ。`Var` の inherent 委譲メソッド 4 件として各自由関数への 1 行委譲で公開する（`Var::conv_transpose2d`・
`Var::conv3d`・#2643 の推奨形と同じ方式）。

- `Var::conv_transpose3d(&self, weight: &Var<'t>, bias: Option<&Var<'t>>, stride: [usize; 3], padding: [usize; 3],
  output_padding: [usize; 3], dilation: [usize; 3], groups: usize) -> Result<Var<'t>, AutodiffError>`
- `Var::max_unpool1d(&self, indices: &Tensor<i32>, kernel_size: usize, stride: Option<usize>, padding: usize,
  output_size: Option<usize>) -> Result<Var<'t>, AutodiffError>`
- `Var::max_unpool2d(…, kernel_size: [usize; 2], stride: Option<[usize; 2]>, padding: [usize; 2],
  output_size: Option<[usize; 2]>)`・`Var::max_unpool3d(…, [usize; 3] 系)`（同形）
- `conv_transpose3d_ops`／`max_unpool_ops` モジュールと `tensor_core::{conv_transpose3d, max_unpool}`・
  `MaxUnpoolLayout` は再エクスポートしない。新規公開型なし。inherent メソッドの追加のみで非破壊。
- 承認後は `ConvTranspose3dMaxUnpoolHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ
  反転する。

層化（推奨案には含めない。#2679 の対象）: 内部層型 `nn::ConvTranspose3d`／`nn::MaxUnpool1d/2d/3d`、`Module` impl、
保存・復元用の `as_*` フック、`compat::Sequential::add_conv_transpose3d`、resident 経路の fail-closed 拒否。
`Sequential::add_max_unpool*` は直列コンテナでは対になる MaxPool の索引を受け渡せないため構造的に不向きで、
#2679 では索引の受け渡し方法（層が索引を保持する／多出力の層インターフェース等）自体の設計判断が先に要る。

承認事項（**すべて未承認**）: 上記 4 メソッドの公開、メソッド名・引数形・索引型 `Tensor<i32>`、および次の設計判断。

- **重複索引の勾配規則**: 本実装は forward の真の随伴（last-writer マスク）。PyTorch 互換の単純 `gather`（全書き手へ
  配る）へ変更するかは未承認の判断として残す（変更する場合は VJP の腕からマスクを外すだけで足りる）。
- **`output_padding >= stride` の拒否**（PyTorch 非互換。解消には `col2im3d` の P 軸契約の切り離しが必要）。
- **`C = 0` の受理**（PyTorch は拒否）・**`kernel = 0` の拒否**（PyTorch は受理）。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）と層化（#2679）。
- CUDA／Metal の GPU 専用カーネル（`im2col3d`／`col2im3d`・転置畳み込み直接カーネル・unpool 専用カーネル）と
  実機計測（§10）。
- `output_padding >= stride`・`padding_mode`・バッチなし入力・channels_last・`output_size` の全 shape 形・索引の
  `int64` 化・低精度（f16／bf16）・f64 自動微分・`create_graph`・activation checkpoint・ONNX `ConvTranspose`（3D）／
  `MaxUnpool` の import／export。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定。
- ruleset・branch protection・リポジトリ設定の変更（`ci.yml` のジョブ追加・リネームは行っていないため required
  contexts の更新も不要）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `ConvTranspose3dMaxUnpoolHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>`（`conv_transpose3d`／`max_unpool1d/2d/3d`）と `compat::Sequential`（`add_*` 4 件）に公開されるとコンパイルが失敗する正のプローブ |
| `conv_transpose3d_max_unpool_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `conv_transpose3d_max_unpool_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_conv_transpose3d_max_unpool`（＋自己テスト） | facade src の再エクスポート・保留対象 4 モジュールの `pub mod`・型の独自宣言・8 名の `fn` 宣言の否定検査 |
| `workspace_declares_conv_transpose3d_max_unpool_fn_names_only_in_allowed_locations` | workspace 全体で `conv_transpose3d`／`max_unpool1d/2d/3d` の `fn` 宣言が `autodiff/src/conv_transpose3d_ops.rs`・`max_unpool_ops.rs` の各 1 件のみ・`add_*` は 0 件 |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。
実装中に `Var` へ同名メソッド（`conv_transpose3d`・`max_unpool2d`）を一時的に足し、doctest が型不一致で失敗し
workspace インベントリが不一致で失敗することを手元で確認した（確認後に元へ戻した）。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_conv_transpose3d_max_unpool_match_cpu_reference`・
`metal_conv_transpose3d_max_unpool_match_cpu_reference`）は `#[ignore]` のまま未実測。手順・期待結果は
`docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md`。

## 11. 出典

- `docs/conv-ops-design.md` §15（転置畳み込みの設計と `output_padding` の非互換）・§16（Conv3d）、
  `docs/pooling-ops-design.md`（索引の意味論）、`docs/autodiff-pool3d-ops-decision.md`（#2643。保留ガード・決定記録の
  構成）、`docs/autodiff-cumulative-ops-decision.md`（#2636）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/conv-transpose3d-max-unpool-pytorch-reference/README.md`

## 12. #2849 決定記録（facade 公開形の確定）

- 状態: **記録のみ（コード変更なし）。** §7 の公開形と設計判断 3 件を 1 案に確定する。公開（`Var` の委譲 4 件と保留ガードの反転）は #2850 が行う。**公開までは `ConvTranspose3dMaxUnpoolHoldDoctestGuard` と `api_surface.rs` の否定ガードを維持する。**
- 基準: `origin/main` `74171fb9`（2026-10-08）。
- 承認の根拠: ルート #2499 の 2026-10-08 ユーザーコメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）が明示した点は、(1) 行 12〜15 は `Var` の委譲メソッドに限って公開する、(4) 層化（`nn::MaxUnpool*` 等・`Sequential::add_*`）は保留を継続する、の 2 点に限る。シグネチャと設計判断 3 件の確定は、§7 の現行形を変えない方向（推奨の無い点は無し）から導いた事項であり、コメントが個別に明示した文言ではない。

### 12.1 確定シグネチャ

```rust
impl<'t> Var<'t> {
    pub fn conv_transpose3d(
        &self,
        weight: &Var<'t>,
        bias: Option<&Var<'t>>,
        stride: [usize; 3],
        padding: [usize; 3],
        output_padding: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> Result<Var<'t>, AutodiffError>;

    pub fn max_unpool1d(
        &self,
        indices: &Tensor<i32>,
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        output_size: Option<usize>,
    ) -> Result<Var<'t>, AutodiffError>;

    pub fn max_unpool2d(
        &self,
        indices: &Tensor<i32>,
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
        output_size: Option<[usize; 2]>,
    ) -> Result<Var<'t>, AutodiffError>;

    pub fn max_unpool3d(
        &self,
        indices: &Tensor<i32>,
        kernel_size: [usize; 3],
        stride: Option<[usize; 3]>,
        padding: [usize; 3],
        output_size: Option<[usize; 3]>,
    ) -> Result<Var<'t>, AutodiffError>;
}
```

- 内部自由関数 `conv_transpose3d_ops.rs`／`max_unpool_ops.rs` と、第 1 引数 `input` を `self` に置き換えた以外の引数順・型・戻り値が一致する。1 行委譲。
- 新規公開型なし。`conv_transpose3d_ops`・`max_unpool_ops`・`tensor_core::{conv_transpose3d, max_unpool}`・`MaxUnpoolLayout` は再エクスポートしない。

### 12.2 設計判断 3 件（§7 の現行形のまま確定）

- (a) 重複索引の勾配は forward の真の随伴（last-writer マスク）のまま。PyTorch 互換の単純 `gather` へは変えない。
- (b) `output_padding >= stride` は各軸で拒否する（PyTorch 非互換のまま）。
- (c) `C = 0` は受理（空出力）、`kernel = 0` は拒否する。

いずれも §5・§7 の現行実装を維持する選択であり、挙動変更の承認ではない。

### 12.3 保留を続けるもの

- 層化（`nn::ConvTranspose3d`／`nn::MaxUnpool1d/2d/3d`・`Module` impl・保存復元フック・`compat::Sequential::add_conv_transpose3d`・resident 経路）。
- `Sequential::add_max_unpool*` と、対になる MaxPool の索引の受け渡し方法の設計判断。

### 12.4 #2850 への申し送り

- 反転するのは `Var` の委譲メソッド名 4 件のプローブだけ。`Tape`／`Tensor<f32>` 上の同名メソッド・`compat::Sequential::add_*`・モジュール再エクスポート・層型名のプローブは未承認経路として維持する。
- workspace インベントリは `var.rs` の委譲 1 件ずつを許可位置に加える。
- 実機 parity は既存の `docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md` を使う。

## 13. #2850 実施記録（`Var` 委譲の公開）

承認根拠: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061（明示されたのは「行 12〜15 は `Var` 委譲に限って公開」「層化は保留継続」の 2 点。シグネチャと §12.2 の設計判断 3 件は §7 の 1 案から導いた §12 の形）。§12 は書き換えていない。

- **公開した名前**: `Var::conv_transpose3d`・`Var::max_unpool1d`・`Var::max_unpool2d`・`Var::max_unpool3d`（§12.1 のシグネチャどおり。追加のみ・`fandhe-ai =0.10.0` 非破壊）。委譲本体は `crates/autodiff/src/var.rs` にあり、`crate::conv_transpose3d_ops::conv_transpose3d`／`crate::max_unpool_ops::max_unpool{1,2,3}d` への 1 行委譲を `var_phase4_ops_methods_are_thin_delegations` が固定する。`conv_transpose3d` は `&self` 込み 8 引数のため `Var::conv_transpose2d` と同じ理由で `#[allow(clippy::too_many_arguments)]` を付けた。
- **設計判断 3 件（§12.2）**: 現行挙動のまま公開した（`output_padding >= stride` 拒否・`C = 0` 受理・`kernel = 0` 拒否・索引は `Var::max_pool*` の `Tensor<i32>` をそのまま渡す）。
- **§9 のガードの現状**: `ConvTranspose3dMaxUnpoolHoldDoctestGuard` は `Var` の `impl` ブロックと `Var::` の UFCS 4 行を外した部分反転（`Tape`／`Tensor<f32>`・`compat::Sequential::add_*`・モジュール・型のプローブは維持）。固定文言 `CONV_TRANSPOSE3D_MAX_UNPOOL_HOLD_PROBE_BODY` も同じ形へ更新。`workspace_declares_conv_transpose3d_max_unpool_fn_names_only_in_allowed_locations` の期待値へ `autodiff/src/var.rs` の 4 件を追加。`facade_does_not_reexport_or_declare_conv_transpose3d_max_unpool` は不変。
- **テスト**: `crates/facade/tests/pool3d_conv_transpose3d_max_unpool_var_delegates.rs`（シグネチャ固定・自由関数との bit 一致・閉形式・往復・型付きエラー・`C = 0`）。`conv_transpose3d_max_unpool_backend_parity.rs` は `Var` メソッド経由に切り替えた（テスト名・`#[ignore]`・判定は不変）。
- **実機**: CUDA／Metal は未実測（`docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md` へ申し送り）。
- **保留継続**: 層化（`nn::ConvTranspose3d`／`nn::MaxUnpool*`・`Sequential::add_*`）、`Tape`／`Tensor<f32>` 上の同名メソッド、モジュール・型の再エクスポート。
