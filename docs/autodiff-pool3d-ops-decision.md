# 3D プーリング（`max_pool3d`・`avg_pool3d`）の CPU 実装記録（イシュー #2643）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-cumulative-ops-decision.md`（#2636）の「共有カーネルを
`tensor-core` に置く」方式を再適用した実装記録であり、**承認記録ではない**（facade 公開形の承認は
#2677 で依頼中。公開自体は承認後の #2678・#2679）。2D 版の意味論は `docs/pooling-ops-design.md`（#1727・
#1728）が正で、本書は 3 軸への一般化と差分だけを記録する。

## 0. 結論

- 3D プーリング（最大〈値と索引〉・平均）の順伝播と VJP を CPU 参照実装として内部クレートへ追加した。
  - 共有カーネル: `fandhe_ai_tensor_core::pool3d`（`Pool3dParams`・`Pool3dLayout`・`pool3d_layout`・
    `max_pool3d_host`・`avg_pool3d_host`・`avg_pool3d_vjp_host`）。走査規則・数値契約の単一情報源。
  - `BackendOps` に既定 `Unsupported` のメソッド 2 件（`pool3d_max`／`pool3d_avg`）を追加（非破壊拡張）。
    CPU は共有カーネルを呼ぶだけの override。CUDA／Metal は変更なし。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::pool3d_ops`（`max_pool3d`／`avg_pool3d`）。専用 `Op` 2
    variant（`Op::MaxPool3d`／`Op::AvgPool3d`）と VJP は `grad.rs`。
- facade 公開は行わない（**→ `Var` 委譲 2 本は #2850 で公開済み。§13**）。`Pool3dOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 層化（`nn::MaxPool3d`／`nn::AvgPool3d`・`Module` impl・`Sequential::add_*`）は #2679 の対象で、本イシューでは
  実装していない（§8）。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。既定 `Unsupported` → 共有ホストカーネルへのフォールバックで動作する。
  実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定（事実のみ）

- 3D プーリングは `docs/pooling-ops-design.md` §11 でスコープ外として残っていた項目である。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、
  #2643 の受入条件（内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について
  承認済みとは記録しない（承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| 共有カーネル | `tensor-core/src/pool3d.rs` | パラメータ（`Pool3dParams`）・レイアウト検査（`Pool3dLayout`）・forward 2 種・Avg の VJP |
| バックエンド抽象 | `tensor-core/src/backend_ops.rs` | `pool3d_max`／`pool3d_avg`（既定 `Unsupported`） |
| CPU | `backend-cpu/src/ops.rs` | 共有カーネルを呼ぶだけの override（`contiguous()` 済みスライスを渡す） |
| autodiff | `autodiff/src/pool3d_ops.rs`・`tape.rs`・`grad.rs` | 自由関数 2 件・`Op` 2 variant・VJP |

- 共有カーネル方式（FFT・累積演算と同じ）を選び、2D の「`eval.rs` と `backend-cpu` に同じアルゴリズムを複製し
  bit 一致テストで縛る」方式は採らなかった。複製による乖離とその検証コストを避けるため。
- `Pool3dParams` と出力 shape 計算は `pool3d.rs` に自己完結で置いた（`ops_shape.rs`・`backend_ops.rs` の型定義部を
  触らず、並列実行中の兄弟イシューとの競合面を減らす）。出力長は既存の `pool_out_len` を軸ごとに再利用する。
  レイアウトは検査に使ったパラメータを保持し、カーネルは `(入力スライス, レイアウト)` だけを受ける
  （パラメータとレイアウトの不整合を型で排除）。
- 命名規律: 素の `fn max_pool3d`／`fn avg_pool3d` は `autodiff/src/pool3d_ops.rs` の各 1 件のみ（workspace
  インベントリが固定。`add_max_pool3d`／`add_avg_pool3d` は 0 件）。trait メソッドは `pool3d_max`／`pool3d_avg`、
  共有カーネルは `*_host`。`Var`／`Tape`／`Tensor` に inherent メソッドは足していない。
- `Op::MaxPool2d` は VJP が rank 4 固定のため再利用せず、`Op::MaxPool3d` を新設した（VJP の腕は rank 5 版）。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応。f64 自動微分経路は対象外。
- 層型を作らない理由: 層化は #2679 に割り当てられており受入基準にも含まれない。追加すると
  `Module::is_pooling` と `nn/module.rs` の Pooling 型列挙テストの更新が必要になり、兄弟イシューと同一ファイルで
  競合する。

## 3. 数値契約

レイアウトは NCDHW 固定・rank 5 のみ。`ceil_mode = true` は `AutodiffError::InvalidArgument` で拒否、
`divisor_override` は引数に持たない。

`Pool3dParams::new`: `kernel`／`stride`／`dilation` の 0 拒否・`stride = None` は `kernel_size`・`2·padding` の
オーバーフロー拒否・**各軸 `padding <= kernel/2`**（`dilation` 非依存。2D と同じ）。

Max:

- 窓内を `kd` 外側 → `kh` → `kw` 内側の row-major で走査。更新条件は `v > best || (v.is_nan() && !best.is_nan())`
  （**タイ先勝ち・NaN 伝播・最初の NaN の索引が残る**）。padding 位置は走査対象外。値は入力要素と bit 一致。
- 索引は `(n, c)` 平面内 flat 添字 `d·H·W + h·W + w`（`Tensor<i32>`）。後続の `MaxUnpool3d`（#2644）が参照する
  定義である。`D·H·W <= i32::MAX` を実体化より前に検査し（`N`・出力が空かどうかに依存しない）、カーネルでは
  `i32::try_from` で変換する。
- VJP: `Op::MaxPool2d` の rank 5 版。`input` を `[N·C, D·H·W]`、`upstream`／`index` を `[N·C, Dout·Hout·Wout]` へ
  reshape し、全索引が `[0, D·H·W)` に収まることを事前検証（違反は `AutodiffError::Backward`）してから
  `scatter_with_fallback(…, ScatterReduce::Add, …)`。重なり窓の重複索引は `Add`（`f64` アキュムレータ・1 回
  downcast）で集約される（`Overwrite` だと誤り）。

Avg:

- 窓内（padding を除く）を row-major で `f64` へ昇格して逐次加算 → `f64` で除算 → 1 回だけ `f32` へ downcast
  （`.claude/rules/coding-rust.md` の長軸縮約契約）。divisor は `count_include_pad = true` なら `kD·kH·kW`
  （`checked_mul`）、`false` なら有効要素数。divisor 0 は型付きエラー。`dilation` は `[1, 1, 1]` 固定。
- VJP: 出力 major（`N`・`C`・`od`・`oh`・`ow`）で走査し、入力要素数ぶんの `f64` 配列へ `upstream / divisor` を
  加算し、最後に 1 回 downcast する（2D の `avg_pool2d_vjp` と同じ加算順）。

非有限入力は事前に拒否せず伝播する。`mul_add` は使わず、matmul 系 FMA 契約には触れない。

## 4. 境界検査

- `pool3d_layout` が確保・実体化より前に次の順で検査し、型付きエラー（`ShapeError`）で拒否する: rank（5）→
  空間軸 `D`／`H`／`W == 0` 拒否（`N`／`C == 0` は受理し空出力）→ 軸ごとの `pool_out_len`（負分子拒否ゲート）→
  軸ごとの空窓拒否（`kernel == 2` かつ `dilation > in_len`）→ 入出力の要素数・バイト数（`checked_numel_for`）。
- カーネルは入力スライス長を再検査し（`ElementCountMismatch`）、窓座標を `checked_mul`／`checked_add`／
  `checked_sub` で逆算して `< in_len` を手動検査する（REQ-8）。Avg VJP の `f64` アキュムレータ配列のバイト数も
  `checked_mul` で検査する。`unsafe`／`get_unchecked`／本番経路の `unwrap`／`expect` は使わない。
- 自由関数は検査を tape 操作より前に終え、エラー時に孤児ノードを残さない（テストで `tape.len()` を固定）。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape（値・索引とも）が出力 shape と異なる場合は `BackendError::ShapeMismatch` で拒否する。
- 巨大 broadcast view（`[1, 1, 1, 1, 2^61]`）は実体化前に `ElementCountOverflow` で拒否される。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/pool3d-pytorch-reference/`。f32 は
u32 ビットパターンで保存）。有限ケース 45 件（Max 17・Avg 28）で Max の値 bit 一致・索引完全一致、Avg の
forward と全勾配が REQ-2 統一複合判定で一致した。非有限ケース 18 件では forward がクラス一致した。

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 索引の型 | `int64` | `Tensor<i32>`（`max_pool2d`・`sort`・`topk` と同じ慣例） | `D·H·W` が `i32::MAX` 超は型付きエラー |
| 入力 rank | rank 4（バッチなし）も受理（実測: `[2,2,2,2]` を返す） | rank 5 のみ（`ShapeError::RankMismatch`） | 差分。受け入れ範囲外 |
| `ceil_mode=True` | 受理（実測） | `AutodiffError::InvalidArgument` | 差分（v1 は `false` のみ） |
| `divisor_override` | あり | なし | 差分 |
| `C = 0` | 拒否（実測: non-batch dimensions must be positive） | 受理（空出力） | 差分。`N = 0`（torch も受理）と一貫させた |
| `N = 0` | 受理（実測） | 受理（空出力） | 一致 |
| 空間軸 0 | 拒否 | 拒否 | 一致 |
| padding 上限超過・kernel／stride／dilation の 0・カーネル超過 | 拒否（実測） | 拒否 | 一致 |
| NaN を含む窓の索引 | 窓内で**最後**の NaN（実測: `[4, 13]` が NaN の窓で 13） | 窓内で**最初**の NaN（`[4, 13]` で 4） | 意味論の決定による差分（2D と同じ）。NaN を含む窓の索引・勾配は比較対象から除外し、「最初の NaN」であることを独立にテストで固定 |
| NaN の値 | NaN | NaN | 一致 |
| `±inf`・窓全体 `-inf` | forward・索引・勾配とも | 同じ | 一致 |
| 空窓（`kernel = 2` かつ `dilation > 入力長`） | 縮退動作（`pooling-ops-design.md` §3） | 拒否 | 差分。torch では実行せず Rust 側テストのみで固定 |
| Avg の累積精度 | f32 累積 | `f64` 累積・1 回 downcast | REQ-2 判定で比較（bit 一致は求めない） |
| `avg_pool3d` の dilation | なし | なし（`[1,1,1]` 固定） | 一致 |
| 整数 dtype・complex・`out=` 引数 | あり | 非対応 | `Var` は f32 のみ |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/pool3d.rs`（単体 12 件）: Max の値・索引・重なり窓・非等方・padding 位置の非走査・
  dilation・タイ先勝ち・NaN の最初の索引・`-inf`、Avg の `count_include_pad` 両方、Avg VJP の独立オラクル
  （基底ベクトル適用による転置）一致、`N=0`／`C=0`、各種拒否（rank・空間軸 0・負分子・空窓・巨大 shape・
  パラメータ 0・padding 上限）、`i32` 上限（`N=0` でも拒否）、スライス長不一致、run-to-run 決定性。
  `backend_ops.rs`: 既定 `Unsupported`。
- `crates/autodiff/tests/pool3d_parity.rs`（18 件）: fixture 突合（有限 45・非有限 18・エラー 24）・
  Avg／Max の勾配の中心差分（f64 の独立リファレンス窓走査）・重なり窓の勾配加算（手計算）・非連続入力・
  モック `BackendOps` による `Unsupported` フォールバックと他エラーの伝播と誤 shape（値・索引）・引数エラーと
  孤児ノード無し・空バッチ・巨大 broadcast view の確保前拒否・テープ記録数（各 1 ノード）・`create_graph` が
  型付きエラーであること（#3085 で成功する確認へ反転）・run-to-run 決定性。
- `crates/backend-cpu/tests/pool3d_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps::pool3d_*` の直接呼び出し
  （解析値・strided 入力・型付きエラー・決定性）、CUDA（macOS では Metal も）の `pool3d_*` が `Unsupported` を
  返し panic しないこと。
- `crates/facade/tests/pool3d_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。
  CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

> #2849 で公開形を確定した（§12）。公開は #2850。本節は確定前の推奨案として履歴を残す。

推奨は 1 つ。`Var` の inherent メソッド 2 件として `pool3d_ops` への 1 行委譲で公開する
（`Var::max_pool2d`／`avg_pool2d` の 3 軸版。`Var::conv3d` と同じ公開方式）。

- `Var::max_pool3d(&self, kernel_size: [usize; 3], stride: Option<[usize; 3]>, padding: [usize; 3],
  dilation: [usize; 3], ceil_mode: bool) -> Result<(Var<'t>, Tensor<i32>), AutodiffError>`
- `Var::avg_pool3d(&self, kernel_size: [usize; 3], stride: Option<[usize; 3]>, padding: [usize; 3],
  ceil_mode: bool, count_include_pad: bool) -> Result<Var<'t>, AutodiffError>`
- `pool3d_ops` モジュール・`tensor_core::pool3d`・`Pool3dParams` は再エクスポートしない。新規公開型なし。
- inherent メソッドの追加のみで非破壊。承認依頼は #2677、公開は承認後の #2678。
- 承認後は `Pool3dOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

層化（推奨案には含めない。#2679 の対象）: 内部層型 `nn::MaxPool3d`／`nn::AvgPool3d`、`Module` impl
（`is_pooling`・`forward_host`）、保存・復元用の `as_*` フック、`compat::Sequential::add_max_pool3d`／
`add_avg_pool3d`、resident 経路の fail-closed 拒否。

承認事項（**すべて未承認**）: 上記 2 メソッドの公開、およびメソッド名・引数形・索引型 `Tensor<i32>`。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）と層化（#2679）。
- CUDA／Metal の GPU 専用カーネルと実機計測（§10）。
- `ceil_mode = true`・`divisor_override`・バッチなし rank 4 入力・索引の `int64` 化・channels_last。
- `create_graph`（高階微分。#3085 で対応済み）・activation checkpoint 対象化・f64／f16／bf16 自動微分経路。
- AdaptiveAvgPool3d・AdaptiveMaxPool3d・FractionalMaxPool・LPPool（MaxUnpool は #2644）。
- ONNX export マッピング・`docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・
  spec（REQ-9）の改定。
- NaN を含む窓の索引・勾配の PyTorch 完全一致（§5）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `Pool3dOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>`（`max_pool3d`／`avg_pool3d`）と `compat::Sequential`（`add_max_pool3d`／`add_avg_pool3d`）に公開されるとコンパイルが失敗する正のプローブ |
| `pool3d_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `pool3d_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_pool3d_ops`（＋自己テスト） | facade src の再エクスポート・`pub mod pool3d_ops`／`pub mod pool3d`・型の独自宣言・4 名の `fn` 宣言の否定検査 |
| `workspace_declares_pool3d_ops_fn_names_only_in_allowed_locations` | workspace 全体で `max_pool3d`／`avg_pool3d` の `fn` 宣言が `autodiff/src/pool3d_ops.rs` の各 1 件のみ・`add_*` は 0 件 |

stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで組んでいる。
実装中に `Var` へ同名メソッドを一時的に足し、doctest と workspace インベントリが失敗することを手元で確認した
（確認後に元へ戻した）。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_pool3d_ops_match_cpu_reference`・
`metal_pool3d_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順は
`docs/perf/logs/pool3d-ops-2643/README.md`。

## 11. 出典

- `docs/pooling-ops-design.md`（2D の意味論・§3 padding 上限と空窓・§5 タイ／NaN・§6 VJP）・
  `docs/autodiff-cumulative-ops-decision.md`（#2636。共有カーネル方式）・`docs/autodiff-fft-ops-decision.md`（#2631）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/pool3d-pytorch-reference/README.md`

## 12. #2849 決定記録（facade 公開形の確定）

- 状態: **記録のみ（コード変更なし）。** §7 の公開形を 1 案に確定する。公開（`Var::max_pool3d`／`avg_pool3d` の委譲と保留ガードの反転）は #2850 が行う。**公開までは `Pool3dOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードを維持する。**
- 基準: `origin/main` `74171fb9`（2026-10-08）。
- 承認の根拠: ルート #2499 の 2026-10-08 ユーザーコメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）が明示した点は、(1) 行 12〜15 は `Var` の委譲メソッドに限って公開する、(4) 層化（`nn::MaxPool3d` 等・`Sequential::add_*`）は保留を継続する、の 2 点に限る。下記のシグネチャ確定は、§7 に代替案が無かったこと（推奨は 1 案）と内部自由関数との一致確認から導いた事項であり、コメントが個別に明示した文言ではない。

### 12.1 確定シグネチャ

```rust
impl<'t> Var<'t> {
    pub fn max_pool3d(
        &self,
        kernel_size: [usize; 3],
        stride: Option<[usize; 3]>,
        padding: [usize; 3],
        dilation: [usize; 3],
        ceil_mode: bool,
    ) -> Result<(Var<'t>, Tensor<i32>), AutodiffError>;

    pub fn avg_pool3d(
        &self,
        kernel_size: [usize; 3],
        stride: Option<[usize; 3]>,
        padding: [usize; 3],
        ceil_mode: bool,
        count_include_pad: bool,
    ) -> Result<Var<'t>, AutodiffError>;
}
```

- 内部自由関数 `crates/autodiff/src/pool3d_ops.rs` の `max_pool3d`／`avg_pool3d` と、第 1 引数 `input` を `self` に置き換えた以外の引数順・型・戻り値が一致する。`Var` メソッドは 1 行委譲。
- 新規公開型なし。`pool3d_ops`・`tensor_core::pool3d`・`Pool3dParams` は再エクスポートしない。
- 挙動は §5 のとおり現行のまま公開する（索引は `Tensor<i32>`、`ceil_mode = true` は型付きエラー）。挙動を変える案は本件の範囲外。

### 12.2 保留を続けるもの

- 層化（内部層型 `nn::MaxPool3d`／`nn::AvgPool3d`・`Module` impl・保存復元フック・`compat::Sequential::add_max_pool3d`／`add_avg_pool3d`・resident 経路）。

### 12.3 #2850 への申し送り

- 反転するのは `Var` の委譲メソッド名（`max_pool3d`／`avg_pool3d`）のプローブだけ。`Tape`／`Tensor<f32>` 上の同名メソッド・`compat::Sequential::add_*`・モジュール再エクスポート・層型名のプローブは未承認経路として維持する。
- workspace インベントリは `var.rs` の委譲 1 件ずつを許可位置に加える。
- 実機 parity は既存の `docs/perf/logs/pool3d-ops-2643/README.md` を使う。

## 13. #2850 実施記録（`Var` 委譲の公開）

承認根拠: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061（明示されたのは「行 12〜15 は `Var` 委譲に限って公開」「層化は保留継続」の 2 点。シグネチャは §7 の 1 案から導いた §12.1 の形）。§12 は書き換えていない。

- **公開した名前**: `Var::max_pool3d`・`Var::avg_pool3d`（§12.1 のシグネチャどおり。追加のみ・`fandhe-ai =0.10.0` 非破壊）。委譲本体は `crates/autodiff/src/var.rs` にあり、`crate::pool3d_ops::max_pool3d`／`avg_pool3d` への 1 行委譲を `api_surface.rs::var_phase4_ops_methods_are_thin_delegations` がトークン列で固定する。
- **§9 のガードの現状**: `Pool3dOpsHoldDoctestGuard` は `Var` の `impl` ブロックと `Var::` の UFCS 行を外した部分反転（`Tape`／`Tensor<f32>`・`compat::Sequential::add_*`・モジュール・型のプローブは維持）。`pool3d_ops_hold_doctest_probe_body_matches_fixed_contract` の固定文言も同じ形へ更新。`workspace_declares_pool3d_ops_fn_names_only_in_allowed_locations` の期待値へ `autodiff/src/var.rs::{max_pool3d, avg_pool3d}` を各 1 件追加。`facade_does_not_reexport_or_declare_pool3d_ops`（facade src の否定検査）は不変。
- **テスト**: `crates/facade/tests/pool3d_conv_transpose3d_max_unpool_var_delegates.rs`（シグネチャ固定・自由関数との bit 一致・閉形式・`ceil_mode=true` 拒否）。`pool3d_ops_backend_parity.rs` は `Var` メソッド経由に切り替えた（テスト名・`#[ignore]`・判定・tolerance は不変）。
- **実機**: CUDA／Metal は未実測（`docs/perf/logs/pool3d-ops-2643/README.md` へ申し送り）。
- **保留継続**: 層化（`nn::MaxPool3d`／`nn::AvgPool3d`・`Sequential::add_max_pool3d`／`add_avg_pool3d`）、`Tape`／`Tensor<f32>` 上の同名メソッド、モジュール・型の再エクスポート。

## 14. 2026-10-11 所有者判断の反映（#3020）

- 層化（`nn::MaxPool3d`／`AvgPool3d`・`Sequential::add_max_pool3d`／`add_avg_pool3d`）の保留継続は D-A で解除する。公開形は #3025 の記録を正とし、実装は #3026。
- 基づく判断: 2026-10-11 の所有者判断 D-A（推奨形で一括承認。#3018 ツリー起票の根拠。`compat-api-scope.md` §5 末尾の適用記録を参照）。本節は記録の反映であり新規承認ではない。
- 各 issue で本記録に公開形を書き、その形のまま実装まで進む。記録に形が無い点だけ承認依頼へ戻す。
- 依存・`unsafe`・tolerance・baseline・`docs/spec` は変更していない。既存本文は書き換えていない。

## 15. 層化の公開形（#3025）

- pool3d（MaxPool3d／AvgPool3d）の層化の公開形は `compat-sequential-3d-fold-lrn-layers-decision.md` を正とする（§3 の MaxPool3d・AvgPool3d 行）。
- 基づく判断: 2026-10-11 の所有者判断 D-A。本節は記録の参照であり新規承認ではない。既存本文は書き換えていない。

## 16. #3027 実施記録（autodiff 側の層 struct）

- `crates/autodiff/src/nn/pooling.rs` に `nn::MaxPool3d`・`nn::AvgPool3d` を追加した（公開形は #3025 の記録 §3 のとおり）。`Var::max_pool3d`／`avg_pool3d` への 1 行委譲で、新規 `Op`／`BackendOps` は無い。`ceil_mode=false` 固定、`AvgPool3d` は `dilation=[1,1,1]` 固定。`new` が `Pool3dParams::new` の検査（0 次元・padding 上限・`2*padding` オーバーフロー）を前倒しする。
- `Module` 実装は `nn/module.rs`。`is_pooling()=true`（Pooling 11 型）、`supports_forward_host()=false`（tape 経路との bit 一致が未検証の host 経路を作らない。`Mish` と同型）。`Module::forward` は `MaxPool3d` で値のみ返し、索引は固有 `forward` から得る。
- `Module::as_max_pool3d`／`as_avg_pool3d`（既定 `None`・読み取り専用）を追加した。既存フックの追加のみで非破壊。後続の facade 結線が使わない場合は撤去してよい。
- facade（`Sequential::add_*`・`LayerSpec`・`save_model` kind・保留ガード反転）は未変更で後続 issue の範囲。CUDA／Metal 実機は未実測。依存・tolerance・`docs/spec` は変更していない。既存本文は書き換えていない。
- テスト: `crates/autodiff/tests/nn_pooling3d.rs`（`Var` 直呼びとの forward・索引・入力勾配 bit 一致、不正引数拒否、孤児ノード無し）。

## 17. #3028 実施記録（facade 結線）

`Sequential::add_max_pool3d`／`add_avg_pool3d` を公開した（#3025 §3 の形・`LayerSpec::Unsupported` で保存は fail-closed。保存 kind は後続 `-c`）。`Pool3dOpsHoldDoctestGuard` は `Sequential` 受け手の分のみ反転し、`Tape`／`Tensor<f32>`・モジュール・型名の否定は維持。`nn::MaxPool3d`／`AvgPool3d` の `as_*` フックは未使用のまま維持。

#3029 で `LayerSpec` を実 variant 化し、`save_model` を `max_pool3d`／`avg_pool3d` kind で対応した（上記の「保存は fail-closed」は #3029 以降は解消済み）。

## 18. AdaptiveAvgPool3d・AdaptiveMaxPool3d の設計（保守・運用チェックリスト。イシュー #3747）

基準 sha は `6537d0925`。**設計のみ**で、コード・公開面・tolerance・baseline・依存・`docs/spec` は変えない。**承認根拠: なし（未承認。#3768 の論点 3 待ち）**。追跡表 `docs/compat-matrix-ari-tracker.md` の # 15 行（C9。イシュー本文の「項目 16」は同表では # 15）。流れは #3747（本書）→ #3768（承認）→ #3823（`Var` 演算）・#3824（`Sequential` 層）→ #3860・#3861（実機 parity）。保存・ONNX export の横断方針は #3750 に委ねる。行番号の参照は基準 sha の値で、実装 PR が直す。

### 18.0 位置づけ・推奨

- 棚卸し `docs/compat-unlisted-holds-inventory.md` の C9 は 3D 適応プーリング 2 種を「未公開」とし、本書 §8 は #2643 のスコープ外として残した。
- 現状: `crates/` に `adaptive_avg_pool3d`／`adaptive_max_pool3d`／`AdaptiveAvgPool3d`／`AdaptiveMaxPool3d` は 0 件。2D／1D の適応プーリング（`Var::adaptive_avg_pool2d`・`adaptive_max_pool2d`、`Sequential::add_adaptive_*`）と 3D 固定窓（`max_pool3d`／`avg_pool3d`）は実装済み。
- **推奨: 実装する。** 理由: (1) 2D 適応の窓式 `adaptive_window`、`Op::MaxPool3d` の VJP、`create_graph/pool.rs` の `window_scatter_structure` をそのまま流用できる。(2) 依存・`unsafe`・GPU カーネルが要らない。(3) 互換区分は「追加のみ」で `docs/api-compat-0.14.0-decision.md` の # 15 行と一致する。

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | `Var` 演算 2 件と `Sequential` 層 2 件。GPU はホストへフォールバック | 既存部品の再利用・追加のみ | GPU tape では同期とコピーが入る |
| B | CUDA／Metal の専用カーネルまで含める | GPU 常駐のまま実行 | カーネル・REQ-8 境界検査・実機検証が増え、2D の adaptive max にも GPU カーネルが無い現状と不整合 |
| C | 対象外にする | 作業なし | C9 の最優先項目が残り、2D との非対称が続く |

### 18.1 公開形とガード反転計画

- シグネチャ案（2D 版の 3 軸版）:
  - `Var::adaptive_avg_pool3d(&self, output_size: [usize; 3]) -> Result<Var<'t>, AutodiffError>`
  - `Var::adaptive_max_pool3d(&self, output_size: [usize; 3]) -> Result<(Var<'t>, Tensor<i32>), AutodiffError>`
  - `compat::Sequential::add_adaptive_avg_pool3d(mut self, output_size: [usize; 3]) -> Result<Self, AutodiffError>`
  - `compat::Sequential::add_adaptive_max_pool3d(mut self, output_size: [usize; 3]) -> Result<Self, AutodiffError>`
- 配置: `Var` の 2 件は `crates/autodiff/src/var.rs` から新モジュール `crates/autodiff/src/adaptive_pool3d_ops.rs` の自由関数へ 1 行委譲する。facade の `Var` は再エクスポートなので追加の配置は要らない。層の内部型 `nn::AdaptiveAvgPool3d`／`nn::AdaptiveMaxPool3d` は `crates/autodiff/src/nn/pooling.rs`。facade からは再エクスポートしない（層型名の公開は #3022 の個別承認に従う別判断）。`LayerSpec`（`pub(super)`）に variant を 2 件足す。
- 索引の意味: `(n, c)` 平面内の flat 添字 `d·H·W + h·W + w`（`Tensor<i32>`）。`Op::MaxPool3d` と同じで、後続の `max_unpool3d` とも整合する。
- ガード: **反転すべき既存の否定ガードは該当なし**。`AdaptiveMaxGlobalPoolHoldDoctestGuard`（`crates/facade/src/lib.rs`）がプローブするのは 2d／1d のみ、`Pool3dOpsHoldDoctestGuard` は `max_pool3d`／`avg_pool3d` のみで、両ガードと固定文言の検査（`adaptive_max_global_pool_hold_doctest_probe_body_matches_fixed_contract`・`pool3d_ops_hold_doctest_probe_body_matches_fixed_contract`）は変えない。
- #3823／#3824 が新設する正ガードの名前案:
  - `var_adaptive_pool3d_methods_are_thin_delegations`
  - `workspace_declares_adaptive_pool3d_ops_fn_names_only_in_allowed_locations`
  - `facade_does_not_reexport_or_declare_adaptive_pool3d_ops`（＋ `_detects_each_category`）
  - `add_adaptive_pool3d_signatures_match_approved_contract`（＋ `_detects_offense`）
  - `compat_sequential_exposes_adaptive_pool3d_add_methods_issue_3824_counts_declarations`
- 未承認の経路（`Tape`／`Tensor<f32>` 上の同名メソッド、モジュール・層型の再エクスポート）は否定検査として残す。実装時の確認事項: 新識別子が既存のトークン走査（`POOL3D_WORKSPACE_FN_NAMES`、adaptive_max_global のインベントリ、`facade_declares_pool3d_add_names_only_in_approved_form`）に誤って当たらないこと。トークン単位照合のため当たらない見込みだが、推測で済ませず実装時に確かめる。

### 18.2 rustdoc と doctest

- 公開 4 件それぞれに役割要約（PyTorch 相当名・窓の式・索引の意味・NaN の扱い）と doctest を置く。doctest は CPU の `fandhe_ai::tape()`／`Sequential::predict` だけで動く形にし、facade 単独で通す。GPU 例は置かない（置くなら `no_run`）。
- 失敗系（`output_size` に 0、rank 違反）は `compile_fail` に頼らず、`is_err` で確かめる通常の doctest かテストで検証する（stable rustdoc は `compile_fail` のコードを照合しないため）。

### 18.3 エラー型

- 新しい variant は足さない。検査順: rank 5（`ShapeError::RankMismatch`）→ `D`／`H`／`W` が 0 なら拒否（`ShapeMismatch`。`N`／`C` が 0 は空の出力を返す）→ `output_size` に 0 があれば拒否 → 出力要素数のオーバーフロー（`checked_numel_for`）→ Max のみ `D·H·W <= i32::MAX`（`IndexRangeOverflow`。`N` が 0 でも検査）→ 実体化 → バックエンド呼び出し → 戻り値 shape の再検証（`BackendError::ShapeMismatch`）→ `push_eager`。
- 層の `new` は `output_size` の 0 を `AutodiffError::InvalidArgument` で拒否する（2D の `AdaptiveAvgPool2d::new` と同じ）。
- ホストカーネルは入力スライス長を再検査し（`ElementCountMismatch`）、窓座標を `checked_*` で計算する。VJP の契約違反は `AutodiffError::Backward`。本番経路で `unwrap`／`expect` を使わず、エラー時に孤児ノードを残さない。

### 18.4 テスト計画

| 区分 | テスト（置き場） | 内容 |
|---|---|---|
| CI | 共有ホストカーネルの単体テスト（`tensor-core/src/adaptive_pool3d.rs`） | 窓境界・割り切れない場合の重なり窓・出力が入力より大きい場合・タイ先勝ち・最初の NaN・`-inf`・拒否系・決定性 |
| CI | `crates/autodiff/tests/adaptive_pool3d_parity.rs` | 中心差分の勾配・重なり窓での勾配加算・Unsupported 時のフォールバックと他エラーの伝播・孤児ノード無し・`create_graph` の二階有限差分 |
| CI | PyTorch 2.14.0 fixture 突合（案: `crates/autodiff/tests/fixtures/adaptive-pool3d-pytorch-reference/`） | 実行するまでは「未実測」 |
| CI | `crates/facade/tests/adaptive_pool3d_ops_backend_parity.rs::cpu_matches_naive_reference` | CPU と Naive の bit 一致 |
| CI | `crates/facade/tests/compat_sequential_adaptive_pool3d.rs` | 層経路 |
| CI | §18.1 の正ガード | 1 行委譲・宣言位置・未承認経路の否定 |
| `#[ignore]`（Phase 10） | `adaptive_pool3d_ops_backend_parity.rs::cuda_adaptive_pool3d_ops_match_cpu_reference` | #3860（`<cuda-node>`） |
| `#[ignore]`（Phase 10） | `adaptive_pool3d_ops_backend_parity.rs::metal_adaptive_pool3d_ops_match_cpu_reference`（`cfg(target_os = "macos")`） | #3861（`<metal-node>`） |
| `#[ignore]`（Phase 10） | `compat_sequential_adaptive_pool3d_backend_parity.rs::{cuda_*, metal_*}` | 層経路 |

- 実機の手順置き場の案: `docs/perf/logs/adaptive-pool3d-ops-3823/README.md`（#3823 で新設）。
- 実機列: cuda／metal は**未実測**、rocm は該当なし（facade 非公開・追跡表 # 24 保留）。cfg の都合で 0 本しか実行されなかった結果は pass の根拠にしない。

### 18.5 数値契約

- Max は純粋な選択演算なので、値と索引はバックエンド間で bit 完全一致とする。走査は `kd` → `kh` → `kw` の row-major。更新条件は `v > best || (v.is_nan() && !best.is_nan())`（タイは先勝ち、窓内で最初の NaN の索引が残る。`max_pool3d`／`adaptive_max_pool2d` と同じ）。
- Avg は窓内を `f64` へ上げて逐次に足し、`f64` で割り、最後に 1 回だけ `f32` へ戻す（`coding-rust.md` の長軸縮約契約）。divisor は実際の窓要素数。VJP は出力 major で `f64` 配列へ足し込み、最後に 1 回だけ `f32` へ戻す（`grad.rs` の 2D 版の 3 軸化）。
- PyTorch との比較は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。CPU と Naive、GPU のフォールバックは bit 一致で書く。
- FMA 契約には影響しない（`mul_add` を使わない）。tolerance 定数・baseline・ガードレール閾値は変えず、新しい baseline も作らない。

### 18.6 互換への影響

`docs/api-compat-0.14.0-decision.md` §8 の手順で判定すると**追加のみ**。`Var` は inherent メソッドの追加、`compat::Sequential` も inherent メソッドの追加（フィールド非公開）、`AutodiffError` は変更なし、`Op` は `pub(crate)`、`LayerSpec` は `pub(super)`、`BackendOps` は内部クレートの trait に既定実装付きメソッドを足すのみ。利用者が同名のトレイトメソッドを持っていても inherent が優先されるだけで、破壊的変更には数えない。

### 18.7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` の適用記録は承認後に #3823／#3824 が書く。文案: `**適用記録（イシュー #3823／#3824・承認: <#3768 回答 URL>）**: Var::adaptive_avg_pool3d／adaptive_max_pool3d と Sequential::add_adaptive_avg_pool3d／add_adaptive_max_pool3d の 4 件を追加のみで公開。新規依存なし`。
- 本 PR で更新するのは `docs/README.md` の索引と、追跡表 # 15 行の「設計記録」セルのみ。

### 18.8 ベンチと性能

既定経路への影響はない。`adaptive_window`・2D 版・`pool3d` には手を入れず新 API を足すだけ。GPU tape ではホストへのフォールバックで同期とコピーが入るが、新規経路のため前後比較は要らない。専用カーネルによる最適化はスコープ外。

### 18.9 依存

依存の追加なし。許容 12 区分・禁止リストとの照合は該当なし。新規 `unsafe` なし。

### 18.10 方式（窓境界の共通化・Op・create_graph・GPU と REQ-8）

- 共通化: 窓は軸ごとに既存の `fandhe_ai_tensor_core::adaptive_window`（`ops_shape.rs`）を呼び、forward・VJP・`create_graph` の唯一の情報源にする（関数自体は変えない）。出力 shape 検査関数 `adaptive_pool3d_out_shape` を新設する。共有ホストカーネル（`AdaptivePool3dLayout`・`adaptive_pool3d_layout`・`adaptive_avg_pool3d_host`・`adaptive_max_pool3d_host`・`adaptive_avg_pool3d_vjp_host`）は新モジュール `crates/tensor-core/src/adaptive_pool3d.rs` に置く。`pool3d.rs` に足さない理由は、兄弟の #3748／#3749 が触りうる同ファイルとの競合と、固定窓の `Pool3dLayout` との混同を避けるため（Q5）。
- `BackendOps`: `adaptive_pool3d_avg`／`adaptive_pool3d_max` を既定 `Unsupported` で足し、CPU のみ共有カーネルを呼ぶ override を持つ。名前は `pool3d_max`／`pool3d_avg` の先例にならい、素の `fn adaptive_*_pool3d` の宣言位置をインベントリで固定しやすくする。
- Op: Max は既存 `Op::MaxPool3d { input, index }` を再利用する（VJP は index による scatter-add で窓パラメータに依存しない。2D の adaptive max が `Op::MaxPool2d` を再利用する先例と同じ。重なり窓の重複索引は `Add` で集約され、`create_graph` もそのまま通る）。Avg は `Op::AvgPool3d` が固定窓の params と layout に依存するため使えず、`Op::AdaptiveAvgPool3d { input }` を新設する（融合しない・checkpoint 対象外・`supports_create_graph() == true`）。`create_graph` は `create_graph/pool.rs` の `window_scatter_structure`（窓が可変でも扱える）に 3D 適応版の窓表を足して対応する。
- GPU: 本ツリーでは専用カーネルを作らない。CUDA／Metal は既定 `Unsupported` から共有ホストカーネルへフォールバックする（`pool3d` と同じ）。#3860／#3861 はこのフォールバック経路の parity を確かめる。REQ-8 はホストカーネルの手動境界検査で満たす。将来 GPU カーネルを足す場合もシェーダ側の手動境界検査を省かないことを条件とする。#3823 本文は GPU カーネルがある前提で書かれているため Q1 で問う。
- 層（#3824 向け）: `Module` impl を持たせ `is_pooling() = true`（「Pooling 11 型」の記述は 13 型になる）、`supports_forward_host() = false`（`nn::MaxPool3d` と同じ）。`Module::forward` は Max の値のみ返し、索引は層固有の `forward` から得る。resident 経路は Pooling 層として fail-closed で拒否する。

### 18.11 PyTorch との差分（すべて未実測。ソースからの記述）

- 索引の型: PyTorch は int64、本設計は `i32`。
- PyTorch はバッチなし rank 4 入力と `output_size` の `None`（その軸は入力サイズのまま）を受け付けるが、本設計は受け付けない（Q4）。
- NaN を含む窓の索引: PyTorch は窓内で最後の NaN、本設計は最初の NaN（§5 と同方針）。
- Avg の累積精度: PyTorch は f32、本設計は f64。
- 実測は #3823 で fixture を作るときに行う。それまでは未実測。

### 18.12 所有者へ戻す論点（#3768 の論点 3 へ）・セキュリティ・スコープ外

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| Q1 GPU カーネル | (a) ホストへのフォールバックのみ (b) CUDA／Metal 専用カーネル（REQ-8 手動境界検査つき） | (a) | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| Q2 Max のタイと NaN | (a) 先勝ち・最初の NaN (b) PyTorch に合わせる | (a) | 同上 |
| Q3 `create_graph` | (a) 対応（Max は自動、Avg は V3） (b) Avg は型付きエラーで拒否 | (a) | 同上 |
| Q4 入力範囲 | (a) rank 5 のみ・`output_size` は `[usize; 3]` (b) `None` 軸と rank 4 にも対応 | (a) | 同上 |
| Q5 実体の置き場所と Op | (a) 新モジュール・Avg は新 Op・Max は `Op::MaxPool3d` 再利用 (b) `pool3d.rs` へ追記・2 つとも新 Op | (a) | 同上 |
| Q6 未承認経路の固定方法 | (a) ソース走査の正ガードと否定検査のみ (b) `AdaptivePool3dHoldDoctestGuard` も新設 | (a) | 同上 |
| Q7 保存と export | #3750 の横断方針に委ねる | #3750 に委ねる | 同上 |

- Q7 補足: `crates/facade/src/compat/model_io.rs` が kind の allowlist を 70 種に固定している。#3750 が決めるまでは `LayerSpec` に variant を置き、保存は `Unsupported` で fail-closed にする。
- セキュリティ: A03 `output_size` と shape（保存ファイル経由を含む）は外部入力として、確保と tape 操作より前に検証する。A08 判定を迂回する経路を作らず、正ガードで 1 行委譲を固定する。実機ホスト名・秘密情報は書かない。
- スコープ外: ONNX export（#3750）、GPU 専用カーネル、層型名の facade 公開（#3022）、`GlobalPool` の rank 5 対応、`ceil_mode` 等の追加引数、`docs/pooling-ops-design.md` への参照追記。新規 issue は所有者承認なしに起票しない。

## 19. LPPool の設計（保守・運用チェックリスト。イシュー #3748）

基準 sha は `a2653b4f7`。**設計のみ**で、コード・公開面・tolerance・baseline・依存・`docs/spec` は変えない。**承認根拠: なし（未承認。#3768 の論点 3 待ち）**。追跡表 `docs/compat-matrix-ari-tracker.md` の # 15 行（C9。イシュー本文の「項目 16」は同表では # 15）。流れは #3748（本書）→ #3768（承認）→ #3825（`Var` 演算と `Sequential` 層の実装）→ #3860・#3861（実機 parity）。保存・ONNX export の横断方針は #3750 に委ねる。行番号の参照は基準 sha の値で、実装 PR が直す。節番号は §18（#3747）・§20（#3749）と並ぶ取り決めで、§18 が未マージの間は欠番になる。

### 19.0 位置づけ・推奨

- 棚卸し `docs/compat-unlisted-holds-inventory.md` の C9 は LPPool（PyTorch の `nn.LPPool1d`／`2d`／`3d`）を「未公開」とし、本書 §8 は #2643 のスコープ外として残した。所有者の Won't ではない。
- 現状: `crates/` に `lp_pool`／`LpPool`／`LPPool` は 0 件（実装もガードもない）。部品はそろっている: `Var::avg_pool1d`／`avg_pool2d`／`avg_pool3d`・`Var::pow_scalar`・`Var::abs`・`Var::sign`・`Var::relu`・`Var::mul`（スカラー Var の broadcast）。層側も `Sequential::add_avg_pool1d`／`2d`／`3d` が公開済み。
- **推奨: 実装する（案 A）。** 理由: (1) 新しい `Op`・`BackendOps`・VJP・カーネルが要らず、既存演算の合成だけで書ける。(2) 依存・`unsafe` の追加がない。(3) 互換区分は「追加のみ」で `docs/api-compat-0.14.0-decision.md` の # 15 行と一致する。

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | PyTorch の `F.lp_pool*d` と同じ順で既存演算を合成する（`x.pow(p)` → `avg_poolNd`〈padding 0〉→ `sign(out)·relu(abs(out))` → `×K` → `pow(1/p)`）。1d／2d／3d の 3 種 | 部品の再利用・PyTorch と演算順が一致・追加のみ | GPU tape では `pow_scalar`／`sign` がホストへフォールバックし、同期とコピーが入る。`create_graph` は `Sign` が replay できないため既定では非対応（Q3） |
| B | 専用 `Op::LpPoolNd`（f64 で窓和を求め、f64 で `pow(1/p)`、最後に 1 回 `f32` へ戻す） | 精度が高い・GPU で 1 カーネルにできる | Op・VJP・`create_graph`・CUDA／Metal カーネル・REQ-8 境界検査・実機検証が増える。PyTorch と丸め順が異なる |
| C | 対象外にする | 作業なし | C9 の未公開が残る。利用者は合成で到達できるが、`sign·relu·abs` の勾配マスクを自分で書く必要がある |

### 19.1 公開形とガード反転計画

- シグネチャ案（PyTorch の引数順に合わせる。LPPool には padding がない）:
  - `Var::lp_pool1d(&self, norm_type: f32, kernel_size: usize, stride: Option<usize>, ceil_mode: bool) -> Result<Var<'t>, AutodiffError>`
  - `Var::lp_pool2d(&self, norm_type: f32, kernel_size: [usize; 2], stride: Option<[usize; 2]>, ceil_mode: bool) -> Result<Var<'t>, AutodiffError>`
  - `Var::lp_pool3d(&self, norm_type: f32, kernel_size: [usize; 3], stride: Option<[usize; 3]>, ceil_mode: bool) -> Result<Var<'t>, AutodiffError>`
  - `compat::Sequential::add_lp_pool1d(mut self, norm_type: f32, kernel_size: usize, stride: Option<usize>) -> Result<Self, AutodiffError>`、`add_lp_pool2d`／`add_lp_pool3d` も同型（`[usize; 2]`／`[usize; 3]`）。
- `ceil_mode`: `Var` 側は既存の `avg_pool*d` にならって引数に持ち、`true` は型付きエラーで拒否する（Q4）。`Sequential` 側は既存の `add_avg_pool*d` と同じく引数を持たず `false` 固定。
- 配置:
  - `Var` の 3 件は `crates/autodiff/src/var.rs` に置き、新モジュール `crates/autodiff/src/lp_pool_ops.rs` の自由関数 `lp_pool1d`／`lp_pool2d`／`lp_pool3d` へ 1 行で委譲する（合成の本体は自由関数側。`pool3d_ops` と同じ方式）。facade の `Var` は再エクスポートなので追加の配置は要らない。
  - 層の内部型 `nn::LpPool1d`／`LpPool2d`／`LpPool3d` は `crates/autodiff/src/nn/pooling.rs` に置き、facade からは再エクスポートしない（層型名の公開は #3022 の個別承認に従う別判断）。
  - `LayerSpec`（`pub(super)`）に variant を 3 件足す。保存は #3750 の方針が決まるまで fail-closed（Q7）。
- ガード: **反転すべき既存の否定ガードはない**（`crates/` 全体で `lp_pool` は 0 件。`Pool3dOpsHoldDoctestGuard` がプローブするのは `max_pool3d`／`avg_pool3d` だけで、変えない）。
- #3825 で新設する正ガードの名前案:
  - `var_lp_pool_methods_are_thin_delegations`（1 行委譲をトークン列で固定）
  - `workspace_declares_lp_pool_fn_names_only_in_allowed_locations`（`fn lp_pool*d` の宣言位置を `lp_pool_ops.rs` と `var.rs` の委譲各 1 件に固定）
  - `facade_does_not_reexport_or_declare_lp_pool_ops`（＋ `_detects_each_category`）
  - `compat_sequential_lp_pool_add_methods_have_approved_signatures`（＋ `_detects_offense`）
- 未承認の経路（`Tape`／`Tensor<f32>` 上の同名メソッド、`lp_pool_ops` モジュール・層型の再エクスポート）は否定検査として残す。
- 実装時の確認事項: 新しい識別子が既存のトークン走査（`POOL3D_WORKSPACE_FN_NAMES`・`facade_declares_pool3d_add_names_only_in_approved_form`・#3747 系のインベントリ）に誤って当たらないことを、推測で済ませず実際に確かめる。

### 19.2 rustdoc と doctest

- 公開 6 件それぞれに、役割要約と doctest を置く。要約に書くこと: PyTorch 相当名、式 `(Σ_window x^p)^(1/p)`、負の入力と非整数 `p` で NaN になる点、全零窓の勾配の扱い。
- doctest は CPU の `fandhe_ai::tape()`／`Sequential::predict` だけで動く形にし、facade 単独で通す。GPU の例は置かない（置くなら `no_run`）。
- 失敗系（`norm_type <= 0`・非有限、`kernel_size` 0、rank 違反、`ceil_mode=true`）は `compile_fail` に頼らない。`is_err` を確かめる通常の doctest かテストで検証する（stable rustdoc は `compile_fail` のエラーコードを照合しないため）。

### 19.3 エラー型

- 新しい variant は足さない。検査順（確保と tape 操作の前に検証を終える）:
  1. `norm_type` が有限かつ `> 0` でなければ `AutodiffError::InvalidArgument`（0・負・NaN・±inf を拒否）
  2. `ceil_mode == true` なら `InvalidArgument`（合成先の `avg_pool*d` も拒否するが、前倒しの二重化）
  3. rank の検査（1d=3、2d=4、3d=5。`ShapeError::RankMismatch`）
  4. `Pool2dParams::new`／`Pool3dParams::new` の検査（0 次元。padding は 0 固定）と出力 shape
  5. 窓の要素数 `K = Π kernel_size` を `checked_mul` で求める（オーバーフローは `InvalidArgument`）
  6. `K` を `f32` へ変換したとき厳密に表せること（`K <= 2^24`。超える場合は `InvalidArgument` で拒否するか、実装時に許容範囲を決めて記録する）
  7. 合成の実行
- 合成の各段は既存演算の `Result` を `?` で伝える。本番経路で `unwrap`／`expect` を使わない。途中でエラーになった場合に孤児ノードが残るかをテストで確かめ、残る場合は検査の前倒しで回避してその旨を記録する。
- 層の `new` は `norm_type` と `kernel_size` を同じ規則で前倒しに検査する（`AvgPool2d::new` と同型）。

### 19.4 テスト計画

| 区分 | テスト（置き場の案） | 内容 |
|------|------------------|------|
| CI | `crates/autodiff/tests/lp_pool_ops.rs` | 閉形式（`p=1` は窓和、`p=2` は二乗和の平方根）・非整数 `p` で負の入力は NaN・全零窓の勾配が 0（NaN でない）・中心差分の勾配・拒否系（`norm_type`・`ceil_mode`・rank・`kernel_size` 0）・孤児ノード無し・決定性 |
| CI | PyTorch 2.14.0 の fixture 突合（案: `crates/autodiff/tests/fixtures/lp-pool-pytorch-reference/`） | 1d／2d／3d、`p ∈ {1, 2, 3, 1.5}`、全零窓の勾配。fixture を作って実行するまでは「未実測」 |
| CI | `crates/facade/tests/lp_pool_ops_backend_parity.rs::cpu_matches_naive_reference` | CPU と Naive の bit 一致 |
| CI | `crates/facade/tests/compat_sequential_lp_pool.rs` | 層経路（`Var` 直呼びとの bit 一致・不正引数の拒否・常駐経路の fail-closed） |
| CI | §19.1 の正ガード | 1 行委譲・宣言位置・未承認経路の否定・シグネチャ固定 |
| `#[ignore]`（Phase 10） | `lp_pool_ops_backend_parity.rs::cuda_lp_pool_ops_match_cpu_reference` | #3860（`<cuda-node>`）未実測 |
| `#[ignore]`（Phase 10） | `lp_pool_ops_backend_parity.rs::metal_lp_pool_ops_match_cpu_reference`（`cfg(target_os = "macos")`） | #3861（`<metal-node>`）未実測 |
| `#[ignore]`（Phase 10） | `compat_sequential_lp_pool_backend_parity.rs::{cuda_*, metal_*}` | 層経路。未実測 |

- 実機手順の置き場の案: `docs/perf/logs/lp-pool-ops-3825/README.md`（#3825 で新設）。
- 実機列: `<cuda-node>`・`<metal-node>` は未実測。`<rocm-node>` は該当なし（facade 非公開）。cfg の都合で 0 本しか実行されなかった結果は pass の根拠にしない。

### 19.5 数値契約

- 演算順は PyTorch の `F.lp_pool*d` と同じにする（ソースからの記述で、未実測）。
  1. `s = avg_poolNd(x.pow_scalar(p), k, stride, padding=0)`。窓和は既存の `avg_pool` 契約（`f64` で蓄積し、`K` で割って 1 回 `f32` へ戻す）。padding 0・`ceil_mode=false` なので divisor は常に `K` で、`count_include_pad` は結果に影響しない（実装では `true` を渡し、その旨をコメントで固定する）。
  2. `z = sign(s) · relu(abs(s))`（forward は `s` と同値。`-0` の符号だけが異なりうる）
  3. `y = (z × K).pow_scalar(1/p)`。`×K` は `f32` で行う。
- `sign·relu·abs` の段を残す理由: 全零窓（`s = 0`）では `pow(1/p)` の勾配が `p > 1` で `inf` になる。この段を通すと勾配が 0 になり NaN を防げる。`Mul` の VJP は relu の枝へ `inf·0 = NaN` を送るが、`Op::Relu` の VJP は乗算をしない要素選択（`v > 0`）なので 0 になる。sign の枝は区分定数の VJP がゼロテンソルを直接生成するので 0 になる。
- 上の挙動は #3825 で実測し、PyTorch の fixture とも突き合わせる（未実測）。とくに `Abs` と `Mul` の VJP が `inf·0` を下流へ漏らさないことを確かめる。
- 既知の端: `0 < p < 1` で窓内に `x = 0` があると `pow_scalar(x, p)` の勾配が `0·inf = NaN` になりうる（PyTorch も同型と見込むが未実測）。fixture で確かめて記録する。
- 判定: PyTorch との比較は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。CPU と Naive は bit 一致。GPU と CPU は合成先の各演算の既存契約に従う（`pow_scalar`／`sign` はホストフォールバックで bit 一致。`avg_pool` の GPU カーネルは `docs/pooling-ops-design.md` の既存契約で、実装時にその節を引用する）。
- FMA 契約には影響しない（新しいカーネルなし・`mul_add` を使わない）。f64 長軸縮約の契約は `avg_pool` 側の既存実装に委ね、変えない。tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline・ガードレール閾値は変えず、新しい baseline も作らない。

### 19.6 互換への影響

`docs/api-compat-0.14.0-decision.md` §8 の判定手順に照らすと**追加のみ**。`Var`: inherent メソッド 3 件の追加。`compat::Sequential`: inherent メソッド 3 件の追加（フィールドは非公開）。`AutodiffError`・`Op`・`BackendOps`: 変更なし。`LayerSpec` は `pub(super)` なので公開面に出ない。利用者が同名のトレイトメソッドを持っていても inherent が優先されるだけで、破壊的変更には数えない。

### 19.7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` の適用記録は、承認後に #3825 が書く。文案: `**適用記録（イシュー #3825・承認: <#3768 回答 URL>）**: Var::lp_pool1d／lp_pool2d／lp_pool3d と Sequential::add_lp_pool1d／add_lp_pool2d／add_lp_pool3d の 6 件を追加のみで公開。新規依存なし`
- 本 PR で更新するのは `docs/README.md` の索引と追跡表 # 15 行の「設計記録」セルだけ。公開状態は「未公開」のまま。

### 19.8 ベンチと性能

- 既定経路への影響はない。`avg_pool*d`・`pow_scalar` 等の既存実装には手を入れず、新しい API を足すだけ。
- GPU tape では `pow_scalar`（CUDA／Metal カーネルが未実装）と `sign` がホストへフォールバックし、ホスト → GPU（`avg_pool`）→ ホストの同期とコピーが入る。新規経路なので前後比較は要らない。専用カーネルによる最適化（案 B）はスコープ外。

### 19.9 依存

依存の追加はない。許容 12 区分・禁止リストとの照合は該当なし。新規 `unsafe` もない。

### 19.10 方式の詳細（範囲・層・`create_graph`・GPU と REQ-8）

- 対象次元: 1d／2d／3d の 3 種。合成先の `avg_pool1d`／`2d`／`3d` がそろっていて、追加の費用は合成 1 本ずつで済むため。PyTorch 2.14.0 に `LPPool3d` があることは fixture 作成時に確かめる（それまでは未実測）。
- `norm_type`: 有限かつ `> 0` に限る（fail-closed）。`p = inf`（max pooling 相当）は `max_pool*d` を使う前提で拒否する。負の入力と非整数 `p` の組は PyTorch と同じく NaN を返し、特別扱いしない。
- スカラー `K` の乗算: `Var` に `mul_scalar` はない。`tape.var_no_grad(&Tensor::scalar(K))` との `mul`（broadcast）で行う（`create_graph.rs` の `PowScalar` の replay と同じ書き方）。実装時に既存のスカラー乗算の入口をもう一度 grep し、あればそれを使う。
- `create_graph`: `ScalarUnaryOp::Sign` は `create_graph::scalar_unary_replayable` に含まれない。PyTorch と同じ合成のままでは二階微分が型付きエラーになる（Q3）。
- 層（#3825 向け）: `Module` impl を持たせ `is_pooling() = true`、`supports_forward_host() = false`（`nn::MaxPool3d` と同じ）、resident 経路は Pooling 層として fail-closed で拒否する。「Pooling 層の型数」の記述は、マージ時点の実数を数えて直す。
- GPU と REQ-8: 本件では専用カーネルを作らない。将来カーネルを足す場合も、シェーダ側の手動境界検査を省かないことを条件とする。

### 19.11 PyTorch との差分（すべて未実測。ソースからの記述）

- `ceil_mode=true`: PyTorch は受け付けるが、本設計は型付きエラーで拒否する（`avg_pool*d` の制約を引き継ぐ。Q4）。
- `norm_type`: PyTorch は 0・負・非有限を検査しないが、本設計は拒否する（Q2）。
- バッチなし入力（1d の rank 2、2d の rank 3、3d の rank 4）: PyTorch は受け付けるが、本設計は受け付けない。
- 丸め: 窓和は本設計が `f64` で蓄積する（PyTorch は `f32`）。そのほかの段は同じ順。
- 実測は #3825 で fixture を作るときに行う。

### 19.12 所有者へ戻す論点（#3768 の論点 3 へ）・セキュリティ・スコープ外

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| Q1 方式 | (a) 既存演算の合成（案 A）(b) 専用 Op（案 B） | (a) | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| Q2 `norm_type` の範囲 | (a) 有限かつ `> 0`（他は拒否）(b) PyTorch と同じく無検査 (c) `>= 1` に限る | (a) | 同上 |
| Q3 `create_graph` | (a) 非対応（`Sign` が replay できないため型付きエラー）(b) `Sign` を replay 可能にして対応（勾配は恒等的に 0。`create_graph.rs` に触れる範囲拡大）(c) `sign·relu·abs` の段を外す（全零窓で勾配が `inf`／NaN） | (a)。二階微分の需要が出たら (b) を別 issue で扱う | 同上 |
| Q4 `ceil_mode` | (a) `Var` は引数に持って `true` を拒否、`Sequential` は持たない (b) `Var` も引数を持たない (c) `avg_pool` 側の `ceil_mode` 対応を先に行う | (a) | 同上 |
| Q5 対象次元 | (a) 1d／2d／3d (b) 1d／2d のみ | (a) | 同上 |
| Q6 未承認経路の固定方法 | (a) ソース走査の正ガードと否定検査のみ (b) `LpPoolHoldDoctestGuard` も新設 | (a)（#3747 の Q6 と揃える） | 同上 |
| Q7 保存と export | #3750 の横断方針に委ねる（それまでは `LayerSpec` に variant を置き、保存は fail-closed。ONNX には `LpPool` op があるが、対応可否は #3750 が決める） | #3750 に委ねる | 同上 |

- セキュリティ（OWASP）: A03 `norm_type`・`kernel_size`・`stride`・shape（保存ファイル経由で復元されるものを含む）は外部入力として扱い、確保と tape 操作より前に検証する（`K` の `checked_mul` を含む）。A08 判定を迂回する経路を作らない。正ガードで 1 行委譲を固定する。
- スコープ外（無断で起票しない。所有者の承認を得てから追跡する）: ONNX export・model_io の kind（#3750）／GPU 専用カーネル／層型名の facade 公開（#3022）／`ceil_mode=true` の対応／バッチなし入力／`Sign` の `create_graph` 対応／`docs/pooling-ops-design.md` への参照追記
