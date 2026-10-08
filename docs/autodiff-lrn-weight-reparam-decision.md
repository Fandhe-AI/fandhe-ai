# LocalResponseNorm・weight_norm・spectral_norm の CPU 実装記録（イシュー #2646）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-fold-unfold-decision.md`（#2645）・
`docs/autodiff-conv-transpose3d-max-unpool-decision.md`（#2644）・`docs/autodiff-pool3d-ops-decision.md`（#2643）と同じ
「内部クレートへ CPU 参照実装を先行し、facade 公開は記録 → 承認の 2 段」方針の実装記録であり、**承認記録ではない**
（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678・#2679）。正規化統計の `f64` アキュムレータ契約は
`.claude/rules/coding-rust.md` を正とし、本書は 3 機能への適用と PyTorch との差分だけを記録する。

## 0. 結論

- `F.local_response_norm`・`torch._weight_norm`／`torch.norm_except_dim`・`parametrizations.spectral_norm`
  （`_SpectralNorm.forward`＋`_power_method`）に相当する 3 機能の順伝播と VJP を CPU 参照実装として内部クレートへ
  追加した。
  - 計算は `tensor-core` の共有ホストカーネル（`lrn`・`weight_reparam`。forward／VJP の単一情報源）に置き、新規
    `BackendOps` フック 3 件（`lrn_forward`・`weight_norm_forward`・`spectral_norm_forward`。既定 `Unsupported`）を
    足した。CPU バックエンドは共有カーネルを呼ぶだけの override。CUDA／Metal は既定の `Unsupported` から
    autodiff 側が共有ホストカーネルへフォールバックする。
  - 入口は `fandhe_ai_autodiff::lrn_ops::local_response_norm`・
    `fandhe_ai_autodiff::weight_reparam_ops::{weight_norm, norm_except_dim, spectral_norm}` と内部型
    `SpectralNormState`。専用 `Op::LocalResponseNorm`／`Op::WeightNorm`／`Op::SpectralNorm` と VJP は
    `tape.rs`／`grad.rs`。
- facade 公開は行わない。`LrnWeightReparamHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した（§9）。
- 層化（`nn::LocalResponseNorm`・`Module` impl・`Sequential::add_*`・`Linear`／`Conv` の重みへの parametrization 結線・
  保存復元フック）は #2679 の対象で、本イシューでは実装していない（§8）。イシュー題名の「LocalResponseNorm」は層名だが、
  内部クレートの自由関数と内部 `Op` までを範囲とした。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の GPU 専用カーネルは対象外。実機テストは `#[ignore]` のまま未実測（§10）。

## 1. 着手時の判定（事実のみ）

- 3 機能とも `docs/norm-ops-design.md` の範囲外で、workspace に同名の宣言は 0 件だった（`nn/` にも層・`Var` メソッドなし）。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、#2646 の
  受入条件（内部実装・PyTorch 突合・決定記録・保留ガード）に限って行った。公開面の拡張について承認済みとは記録しない
  （承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| tensor-core | `src/lrn.rs` | `LrnParams`（`#[non_exhaustive]`・`new` で検査）・`lrn_layout`・`local_response_norm_host`・`local_response_norm_vjp_host` |
| tensor-core | `src/weight_reparam.rs` | `norm_except_dim_layout`／`weight_norm_layout`・`norm_except_dim_host`・`weight_norm_host`・`weight_norm_vjp_host`／`spectral_norm_layout`・`spectral_power_iterate_host`・`spectral_sigma_host`・`spectral_norm_host`・`spectral_norm_vjp_host` |
| tensor-core | `src/backend_ops.rs` | defaulted フック 3 件（既定 `Unsupported`） |
| backend-cpu | `src/ops.rs` | 3 フックの override（共有カーネルを呼ぶだけ。形状は再検査） |
| autodiff | `src/lrn_ops.rs`・`src/weight_reparam_ops.rs` | 自由関数と内部型 `SpectralNormState` |
| autodiff | `tape.rs`・`grad.rs` | `Op` 3 variant と VJP の腕 3 つ |

- **既存 Op の合成にしない理由（着手時に再確認した事実）**: `grad.rs` の broadcast 付き `Op::Mul`／`Op::Div` の VJP は
  `reduce_to_shape`（純 `f32` 逐次和）で縮約し、`f64` 縮約の `reduce_bias_grad` は `Op::Add` の bias パターン等に
  限られる。`v ⊙ (g/‖v‖)` の合成では `dg` が、`W/σ` の合成では `dσ` が `f32` 縮約になり、`x.mul(x)` → プールの合成では
  二乗が `f32` で先に確定する。いずれも `.claude/rules/coding-rust.md` の `f64` アキュムレータ契約（正規化統計は「先に
  `f64` へ昇格してから二乗」）に抵触するため、3 機能とも専用 `Op` ＋共有ホストカーネルとした。
- フック名は自由関数名と別トークンにした（`pool3d_max` と `max_pool3d` の関係と同じ）。workspace インベントリを
  「各 1 件・単一ファイル」に保つため。defaulted メソッドの追加のみで公開済み trait を破壊しない（#2643・#2637 と同じ
  拡張方式）。
- forward は「フック → `Unsupported` のときだけ共有ホストカーネル」、VJP はホスト側のみ。非融合（`push_eager`）・
  非 checkpoint・高階微分非対応・f64／低精度経路なし。`ops_shape.rs` には追記しない（兄弟イシューとの同一ファイル競合
  回避。#2643〜#2645 と同じ判断）。
- 命名規律: 素の `fn local_response_norm`・`fn weight_norm`・`fn norm_except_dim`・`fn spectral_norm` は
  `autodiff/src/lrn_ops.rs`／`autodiff/src/weight_reparam_ops.rs` の各 1 件のみ（workspace インベントリが固定。
  `add_*` は 0 件）。`Var`／`Tape`／`Tensor` に inherent メソッドは足していない。
- **`SpectralNormState`**: 最初から `#[non_exhaustive]`・フィールド非公開・アクセサのみ（後日の公開承認で形を変えずに
  済ませるため）。`from_vectors` は `u0`／`v0` を `x / max(‖x‖, eps)`（`F.normalize` と同じ。`+ eps` ではない）で
  正規化して保持する。**乱数初期化は持たない**（`Generator` の公開形が承認待ちのため結合しない）。PyTorch の
  「正規乱数 → 正規化 → 15 回の予備反復」は呼び出し側が `power_iterate(weight, 15)`
  （定数 `SPECTRAL_NORM_INIT_POWER_ITERATIONS`）で再現できる。
- **状態の不変条件**: `spectral_norm` は検査 → 実体化 → 反復（ローカル `u`／`v`）→ forward → 戻り shape 再検証 →
  状態書き戻し → `push_eager` の順で、forward が失敗（`Unsupported` 以外のバックエンドエラー・誤 shape を含む）しても
  `state` は進まない（モック `BackendOps` のテストで固定）。

## 3. 数値契約

- **LocalResponseNorm**: 窓内の二乗和は要素を先に `f64` へ昇格してから二乗し index 順に蓄積。`d = k + (α/size)·S`・
  `d^β`（`f64::powf`）・除算まで `f64` のまま、最後に 1 回だけ `f32` へ downcast。窓和はスライディング差分にせず窓ごとに
  直接加算する（非負和の引き算による桁落ちを避ける。計算量は `O(C·min(size, C))`）。VJP は
  `dx_m = g_m·d_m^{−β} − (2αβ/size)·x_m·Σ_{c∈W'(m)} g_c·x_c·d_c^{−β−1}`（`W'` は forward 窓の反転。偶数 `size` で
  forward と異なる）。要素積 `g_c·x_c` は `f32` で確定してから `f64` へ昇格する（`grad.rs::rmsnorm_vjp_rows` と同じ
  契約形）。統計 `d` は forward と同じ規則で再計算する。
- **weight_norm**: グループ（`dim` の各添字）ごとに `n = sqrt(Σ (f64)v²)`（昇格してから二乗・index 順）、
  `w = v·(g/n)` を `f64` で計算して 1 回 downcast。式順を PyTorch（`v * (g / norm)`）に揃えて `‖v‖ = 0` 時の非有限
  クラス（`0·inf = NaN`）を一致させた（拒否せず伝播）。VJP は `dot = Σ up·v`（要素積 `f32` 確定 → `f64` 蓄積）、
  `dg = dot/n`、`dv = (g/n)·up − (g·dot/n³)·v`。軸は permute コピーせず `(outer, axis_len, inner)` 分解でグループ番号を
  求める。
- **spectral_norm**: 行列ベクトル積・ノルムは `f64` 蓄積、各反復の `u`／`v` は `f32` で保持（PyTorch と同じ精度境界）。
  `σ = uᵀ W_mat v` は `f64`、除算は `f64`・1 回 downcast。VJP は `s = Σ up·W`（要素積 `f32` 確定 → `f64` 蓄積）、
  `dW_ij = up_ij/σ − (s/σ²)·u_i·v_j`。`u`／`v` は定数（PyTorch も buffer の clone で勾配を流さない）。`σ` は `Op` に
  保持せず、保持した `u`／`v` スナップショットと入力値から再計算する。
- 総和順は GPU カーネルが無いため butterfly 順ではなく index 順。`mul_add` は使わない（matmul 系 FMA 契約には触れない）。
- PyTorch は f32 累積のため bit 一致は求めず、REQ-2 統一複合判定（`common::req2_close`）で突合する。CPU tape
  （`CpuBackendOps`）と NaiveOps tape（ホストフォールバック）の突合は bit 一致（同じ共有カーネルを通る。
  `crates/facade/tests/lrn_weight_reparam_backend_parity.rs`）。

## 4. 境界検査

- LocalResponseNorm: `LrnParams::new`（`size == 0`・非有限 `alpha`／`beta`／`k` を `InvalidArgument`）→ `lrn_layout`
  （rank 3 以上・確保前の `checked_numel_for`）→ 実体化 → フック／ホスト → 戻り shape 再検証 → `push_eager`。窓端は
  `saturating_sub`／`min` で手動クリップ（REQ-8）。
- weight_norm: 同一 tape → `weight_norm_layout`（rank 1 以上・`dim < rank`・`g` shape の完全一致・確保前検査）→ 実体化 →
  フック／ホスト → 戻り shape 再検証 → `push_eager`。
- spectral_norm: `state` と `weight` の shape 整合（rank 2 以上・`dim < rank`・要素数 0 拒否・確保前検査）→ 実体化 →
  反復 → フック／ホスト → 戻り shape 再検証 → 状態書き戻し → `push_eager`。`from_vectors` は
  `n_power_iterations >= 1`・`eps` 有限かつ非負・`u0`／`v0` の長さと有限性を検査する。
- 検査はすべて tape 操作・状態更新より前に終え、エラー時に孤児ノードを残さない（テストで `tape.len()` を固定）。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、戻り値 shape が
  期待と異なる場合は `BackendError::ShapeMismatch` で拒否する。VJP でも shape を再検証し、契約違反は panic ではなく
  型付きエラー。
- `σ = 0`・`‖v‖ = 0`・非有限入力は拒否せず伝播する。

## 5. PyTorch 2.14.0 との差分

fixture は実 PyTorch 2.14.0+cpu の実行値
（`crates/autodiff/tests/fixtures/lrn-weight-reparam-pytorch-reference/`。f32 は u32 ビットパターンで保存。
`torch.set_num_threads(1)`・固定シード 2646 で生成。2 回生成して同一 sha256 を確認）。LRN 21＋非有限 5＋エラー 10、
weight_norm 17＋ゼロノルム／非有限 4＋エラー 6、spectral_norm 18（training 15・eval 3）＋エラー 6 ケースで、下表の差分を
除きすべて REQ-2 判定内で一致した。

| 項目 | PyTorch 2.14.0（実測） | 本実装 | 扱い |
|---|---|---|---|
| LRN の窓・除数・偶数 `size` の非対称性・非有限伝播 | `[c−⌊size/2⌋, c+⌊(size−1)/2⌋]`・除数は常に `size` | 同じ | 一致 |
| LRN rank 1・2 | 拒否（`ValueError`） | 拒否（`Shape`） | 一致 |
| LRN `size = 0` | 拒否（`RuntimeError`） | 拒否（`InvalidArgument`） | 一致 |
| LRN `N = 0`／`C = 0`／空間軸 0 | 受理（空出力） | 受理（空出力） | 一致 |
| LRN の非有限 `alpha`／`beta`／`k` | 受理（無検査） | 拒否（`InvalidArgument`） | 差分（意図的） |
| weight_norm の `dim` 範囲外・rank 0 | 拒否（`IndexError`） | 拒否（`Shape`） | 一致 |
| weight_norm の `g` shape | **検査しない**（`[3]`・`[2,1]`〈長さ違い〉・`[1,4]`〈別軸〉・`dim=None` に `[1,1]` を渡しても受理） | `norm_except_dim` の出力 shape と完全一致のみ受理（`Some(dim)` は keepdim 形・`None` は rank 0） | 差分（意図的。誤配線の検出を優先） |
| weight_norm の `dim = None` | `dim = -1`（`g` は 0-dim） | `Option<usize>`（`None`）。負の `dim` は非対応 | 差分（スコープ外） |
| weight_norm のゼロノルム・NaN・inf | `v·(g/norm)` が `0·inf = NaN` 等を伝播 | 同じ式順で同じクラス | 一致 |
| spectral_norm の反復順・`F.normalize`・`σ`・勾配 | `u ← normalize(W v)` → `v ← normalize(Wᵀ u)`・`weight/σ`・`u`／`v` は定数 | 同じ | 一致（`u1`／`v1`・出力・勾配とも判定内） |
| spectral_norm の rank 1 | 受理（`F.normalize` に縮退） | 拒否（`Shape`） | 差分（スコープ外） |
| spectral_norm の要素数 0 | 受理 | 拒否 | 差分（0 除算を伴う `σ` を前提とする演算のため） |
| spectral_norm の `n_power_iterations = 0`・`dim` 範囲外 | 拒否（`ValueError`／`IndexError`） | 拒否 | 一致 |
| spectral_norm の負／NaN の `eps` | 受理（NaN は出力が非有限） | 拒否 | 差分（意図的） |
| spectral_norm の負の `dim`・`u`／`v` の乱数初期化 | あり | なし（呼び出し側が `from_vectors`／`power_iterate` で再現） | 差分（スコープ外） |
| 累積精度 | f32 累積 | `f64` | REQ-2 判定で比較 |

fixture は初期化直後の `u0`／`v0` と forward 1 回後の `u1`／`v1` の両方を保存した（training では `module.weight` 相当の
forward ごとに反復が走るため、1 記録につき forward は 1 回。初期化乱数は Rust 側で再現しない）。tolerance・baseline は
変更していない。PyTorch との差分を埋めるために判定を緩めていない。

## 6. テスト構成

- `crates/tensor-core/src/{lrn,weight_reparam}.rs`（単体）: 奇数・偶数 `size`・`size = 1`・`size > C`・境界チャネル、
  `f32` 二乗で溢れる入力（`2e20`）、非有限伝播、空テンソル、`dim = Some(0)`／`Some(1)`／`None`／rank 1、往復恒等、ゼロノルム、
  2×2 既知特異値・`dim = 1`、VJP の中心差分、rank・長さ不一致、巨大 shape の確保前拒否。
- `crates/tensor-core/src/backend_ops.rs`: 3 フックの既定 `Unsupported`。`crates/backend-cpu/tests/`:
  `lrn_weight_reparam_parity.rs`（CPU override と共有カーネルの bit 一致・非連続入力・型付きエラー・決定性）・
  `backend_ops_dispatch.rs`（CUDA／Metal が `Unsupported` を返し panic しない）。
- `crates/autodiff/tests/lrn_parity.rs`（17 件）・`weight_reparam_parity.rs`（21 件）: fixture 突合（forward・勾配・
  非有限・エラー表）・中心差分（`tests/conv_transpose3d_parity.rs` と同じ `H=1e-3`・相対 1e-2 または絶対 1e-3・`τ=1e-4`。
  緩和なし）・偶数 `size` の反転窓の手計算・非連続入力・引数エラーと孤児ノード無し・巨大 broadcast view の確保前拒否・
  `create_graph` の型付きエラー・決定性・モック `BackendOps`（`Unsupported` フォールバック到達・他エラー伝播・誤 shape、
  spectral は状態不変も確認）・spectral の eval 状態不変／training 更新・十分反復後の最大特異値 1。
- `crates/facade/tests/lrn_weight_reparam_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし・bit 一致）。
  CUDA／Metal 実機は `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

> #2849 で `SpectralNormState` の位置と `norm_except_dim` の扱いを確定した（§12）。公開は #2851。本節は確定前の推奨案として履歴を残す。

推奨は 1 つ。`Var` の inherent 委譲メソッド 3 件として各自由関数への 1 行委譲で公開する（既存の `rms_norm`／`layer_norm` と
兄弟 #2643〜#2645 の推奨形との一貫性）。

- `Var::local_response_norm(&self, size: usize, alpha: f32, beta: f32, k: f32) -> Result<Var<'t>, AutodiffError>`
- `Var::weight_norm(&self /* v */, g: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError>`
- `Var::spectral_norm(&self /* weight */, state: &mut SpectralNormState, training: bool) -> Result<Var<'t>, AutodiffError>`
- この形に付随して新規公開型が必要になるのは `SpectralNormState` 1 つ（状態を持つため避けられない）と、`norm_except_dim`
  の置き場所（`Tensor` 上のメソッド／自由関数の再エクスポート／公開しない、のいずれか）。これらは承認事項として列挙する。
- モジュール再エクスポート（`lrn_ops`・`weight_reparam_ops`）は推奨しない（内部モジュール構成を公開面へ固定してしまうため）。
  `Sequential::add_*` も推奨しない（層化は #2679 の対象で、parametrization の結線方式〈重みの置換か層ラッパか〉が未決のため）。
- 承認後は `LrnWeightReparamHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

承認事項（**すべて未承認**）:

- 上記 3 メソッドの公開。
- `SpectralNormState` の公開と置き場所（`from_vectors` が乱数を持たないこと・`power_iterate` を含むか）。
- `norm_except_dim` の公開形。
- `g` の shape 規約（`norm_except_dim` の出力 shape と完全一致のみ受理。PyTorch は無検査）。
- `dim: Option<usize>`（負の `dim` なし）。
- rank 1 の `spectral_norm` 非対応。
- 非有限パラメータ（`alpha`／`beta`／`k`／`eps`）の拒否。
- 乱数初期化を持たないこと。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678）と層化（`nn::LocalResponseNorm`・`Linear`／`Conv` の重みへの weight_norm／
  spectral_norm の parametrization 結線・`Sequential::add_*`・保存復元フック・resident 経路。#2679）。
- CUDA／Metal の GPU 専用カーネルの新設と実機計測（§10）。
- spectral_norm の rank 1 入力（`F.normalize` への縮退）・`u`／`v` の乱数初期化・`remove_parametrizations` 相当・旧 API
  （`torch.nn.utils.weight_norm` の hook 方式）・負の `dim`。
- `create_graph`（高階微分）・activation checkpoint・f64 自動微分・f16／bf16・ONNX の import／export。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定・
  `MIN_KNOWN_PROBE_BLOCKS` の更新（下限検査と独立集計の一致検査のみで、新ガード追加は編集不要。兄弟も据え置き）。
- ruleset・branch protection・リポジトリ設定の変更（`ci.yml` のジョブ追加・リネームは行っていないため required
  contexts の更新も不要）。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `LrnWeightReparamHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッドが `Var`／`Tape`／`Tensor<f32>`（`local_response_norm`／`weight_norm`／`spectral_norm`／`norm_except_dim`）と `compat::Sequential`（`add_*` 3 件）に公開されるとコンパイルが失敗する正のプローブ |
| `lrn_weight_reparam_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `lrn_weight_reparam_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_lrn_weight_reparam`（＋自己テスト） | facade src の再エクスポート・`pub mod lrn_ops`／`weight_reparam_ops`／`lrn`／`weight_reparam`・型 4 件の独自宣言・7 名の `fn` 宣言の否定検査。検出はトークン完全一致のみで、`weight_norm_forward`・`spectral_norm_host`・`lrn_layout` 等の別トークン・呼び出し・コメント／文字列・非公開 `use` を誤検出しないことを自己テストで固定 |
| `workspace_declares_lrn_weight_reparam_fn_names_only_in_allowed_locations` | workspace 全体で 4 名の `fn` 宣言が承認済みの置き場所の各 1 件のみ・`add_*` は 0 件 |

検出範囲は上記トークン列（`pub use` の経路・型／`pub mod`／`fn` の宣言）と、プローブが名前解決で触れる位置に限る。
マクロ生成や別名経由のメソッドまでは保証しない。stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは
正のプローブ＋インベントリで組んでいる。実装中に `Var` へ同名メソッド `weight_norm` を一時的に足し、doctest が型不一致で
失敗し workspace インベントリが不一致で失敗することを手元で確認した（確認後に元へ戻した）。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_lrn_weight_reparam_match_cpu_reference`・
`metal_lrn_weight_reparam_match_cpu_reference`）は `#[ignore]` のまま未実測。手順・期待結果は
`docs/perf/logs/lrn-weight-reparam-2646/README.md`。期待結果は「`Unsupported` → ホストフォールバック経路が CPU tape と
bit 一致する」ことの確認であり、新規 GPU カーネルの parity ではない。

## 11. 出典

- `docs/norm-ops-design.md`（正規化系の設計）、`docs/autodiff-fold-unfold-decision.md`（#2645。保留ガード・決定記録の構成）、
  `docs/autodiff-conv-transpose3d-max-unpool-decision.md`（#2644）、`docs/autodiff-pool3d-ops-decision.md`（#2643）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・正規化統計の `f64` 契約・
  勾配の長軸縮約契約・カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/lrn-weight-reparam-pytorch-reference/README.md`

## 12. #2849 決定記録（facade 公開形の確定）

- 状態: **記録のみ（コード変更なし）。** §7 で未決だった `SpectralNormState` の公開位置と `norm_except_dim` の扱いを 1 案に確定する。公開（`Var` の委譲 3 件・`SpectralNormState` の再エクスポートと保留ガードの反転）は #2851 が行う。**公開までは `LrnWeightReparamHoldDoctestGuard` と `api_surface.rs` の否定ガードを維持する。**
- 基準: `origin/main` `74171fb9`（2026-10-08）。
- 承認の根拠: ルート #2499 の 2026-10-08 ユーザーコメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）が明示した点は、(1) 行 12〜15 は `Var` の委譲メソッドに限って公開する、(4) 層化と重み再パラメータ化の結線方式は保留継続、の 2 点に限る。`SpectralNormState` の位置と `norm_except_dim` 非公開は、コメントが明示した文言ではなく、既存の同種型の公開位置に合わせる最小限の公開という方向からの導出である。

### 12.1 確定シグネチャ

```rust
impl<'t> Var<'t> {
    pub fn local_response_norm(
        &self,
        size: usize,
        alpha: f32,
        beta: f32,
        k: f32,
    ) -> Result<Var<'t>, AutodiffError>;

    /// self が `v`
    pub fn weight_norm(
        &self,
        g: &Var<'t>,
        dim: Option<usize>,
    ) -> Result<Var<'t>, AutodiffError>;

    /// self が `weight`
    pub fn spectral_norm(
        &self,
        state: &mut SpectralNormState,
        training: bool,
    ) -> Result<Var<'t>, AutodiffError>;
}
```

- 内部自由関数 `lrn_ops.rs`／`weight_reparam_ops.rs` と、第 1 引数（`input`／`v`／`weight`）を `self` に置き換えた以外の引数順・型・戻り値が一致する。1 行委譲。`lrn_ops`・`weight_reparam_ops` モジュールは再エクスポートしない。

### 12.2 `SpectralNormState` の公開位置

- 位置は facade クレートルート `fandhe_ai::SpectralNormState`。`StftOptions`・`GradcheckOptions` など `Var` メソッドの引数に取る既存の状態・オプション型と同じ位置である。
- 経路は別名なし 1 行の `pub use fandhe_ai_autodiff::weight_reparam_ops::SpectralNormState;`（`GradcheckOptions` と同じモジュール経由パス。`weight_reparam_ops` モジュール自体は再エクスポートしない。autodiff のクレートルートは変更しない）。型の形は変えない（`#[non_exhaustive]`・フィールド非公開）。
- この `pub use` により `from_vectors`・アクセサ 6 本・`power_iterate` が到達可能になる。乱数初期化は持たない。PyTorch の予備反復は利用者が `power_iterate(weight, 15)` で再現する。

### 12.3 `norm_except_dim` と定数

- `norm_except_dim` は公開しない（`Var`／`Tape`／`Tensor` のメソッド・自由関数の再エクスポートのいずれも行わず、保留ガードの該当プローブを維持する）。理由は、公開を求める根拠が記録に無いことと、`Var::weight_norm`／`spectral_norm` の呼び出しに不要なことである。
- 使い勝手上の制約として、利用者が `g` を自前で用意する場合の shape 規約を明記する。`weight_norm` は `g` を `norm_except_dim` の出力 shape と完全一致でのみ受理する。`dim = Some(d)` は `d` 軸以外が長さ 1 の keepdim 形、`dim = None` は rank 0。値は `d` 以外の全軸にわたる L2 ノルム。
- `SPECTRAL_NORM_INIT_POWER_ITERATIONS` も公開しない（呼び出しに不要）。

### 12.4 現行のまま確定する項目

`g` の shape 完全一致・`dim: Option<usize>`（負の `dim` なし）・rank 1 の `spectral_norm` 非対応・非有限パラメータ（`alpha`／`beta`／`k`／`eps`）の拒否・乱数初期化なし。

### 12.5 保留を続けるもの

- 層化（`nn::LocalResponseNorm`・`compat::Sequential::add_*`・保存復元フック・resident 経路）。
- 重み再パラメータ化の結線方式（`Linear`／`Conv` の重みの置換か層ラッパか）。

### 12.6 #2851 への申し送り

- 反転するのは `Var` の委譲メソッド名 3 件のプローブだけ。`Tape`／`Tensor<f32>` 上の同名メソッド・`norm_except_dim`・`compat::Sequential::add_*`・モジュール再エクスポートのプローブは未承認経路として維持する。
- workspace インベントリは `var.rs` の委譲 1 件ずつと、承認済みの `SpectralNormState` 再エクスポートを許可位置に加える。非破壊性の確認（行 15 の †）は公開時に `api_surface.rs` で行う。
- 実機 parity は既存の `docs/perf/logs/lrn-weight-reparam-2646/README.md` を使う。
