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
  型付きエラーであること・run-to-run 決定性。
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
- `create_graph`（高階微分）・activation checkpoint 対象化・f64／f16／bf16 自動微分経路。
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
