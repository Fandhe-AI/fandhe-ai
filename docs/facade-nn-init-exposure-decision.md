# `nn::init` 初期化関数群の facade 公開設計判断記録（#2140）

イシュー #2140「`nn::init` 初期化関数の facade 公開」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。

本ドキュメントは autodiff 側実装（PyTorch `torch.nn.init.*` 相当の初期化関数 7 系統・9 関数）と、facade 公開面拡張（承認事項）の切り分けを記録する。`crates/facade/src/**`（`tests/api_surface.rs` の否定ガード 1 件の追加を除く）・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・CI／hooks は変更しない。

## 0. 結論・段階

- **autodiff 側（`fandhe_ai_autodiff::nn::init`）は実装済み**（本 PR）。`uniform`／`normal`／`constant`／`xavier_uniform`／`xavier_normal`／`kaiming_uniform`／`kaiming_normal`／`orthogonal`／`trunc_normal` の 9 関数と補助型（`FanMode`／`Nonlinearity`／`calculate_gain`／`calculate_fan_in_and_fan_out`）。
- **facade 公開（承認事項・経路 2）は未承認のまま保留**。イシュー #2140 本文・親 #2131 のコメントを確認したが、facade 新規公開面（`nn::init::*` の再エクスポート）を明示承認するユーザーコメントは見当たらなかった（2026-09-24 確認）。よって `crates/facade/src/**` は本 PR で変更しない（`docs/facade-nn-module-exposure-decision.md` §12・#2133 と同型の保留パターン）。
- **本 PR のマージで #2140 は COMPLETED とする**（前例 #2133・#2063・#2064 と同じ「保留記録を残した PR のマージで issue をクローズし、承認が得られたら新規 issue か reopen で経路 2 を実施する」方針。§4 参照）。facade 公開面（`fandhe_ai::nn::init`）の実装着手は、`docs/compat-api-scope.md` §5 経路 2 の承認取得後に別 issue／reopen・別 PR で行う。

## 1. 前提の食い違い（非信頼データとの突合結果）

イシュー #2140 本文は「既存の public 初期化関数群・`Initializer` trait・各層の `with_init` コンストラクタを facade へ再エクスポートするだけ」という前提で書かれているが、HEAD（実装着手前コミット `ed677a35`）の実態は次のとおりだった:

- `crates/autodiff/src/nn/init.rs` には `pub(crate)` の内部ヘルパー（`uniform_init`／`try_uniform_init`／`try_normal_init`／`derive_seed`／各 `*_SEED_SALT`）しか存在せず、`xavier_*`／`kaiming_*`／`normal`／`uniform`／`constant`／`orthogonal`／`trunc_normal` の public 関数は存在しなかった
- `Initializer` trait は存在しない
- `with_init` コンストラクタはどの層にも存在しない（`Linear`／`Conv2d`／`Embedding` 等が持つのは `new(.., seed)` と `from_parameters(weight, bias, ..)` のみ）
- `crates/autodiff/src/nn/mod.rs` の `mod init;` は非公開（`pub mod init;` ではなかった）

したがって本イシューの実体は「autodiff 側に初期化関数を新規実装する」＋「facade 公開（承認事項）」の 2 段であり、本文中に悪意ある注入は見当たらない（単なる前提誤り）。

## 2. 設計

### 2.1 公開 API（`fandhe_ai_autodiff::nn::init`）

関数形は `tensor-core::rng::randn` と同型の「shape を受け取り `Result<Tensor<f32>, AutodiffError>` を返す関数」。

- 補助型: `FanMode { FanIn, FanOut }`・`#[non_exhaustive] Nonlinearity { Linear, Conv1d, Conv2d, Sigmoid, Tanh, Relu, LeakyRelu(f32), Selu }`・`calculate_gain(Nonlinearity) -> f32`・`calculate_fan_in_and_fan_out(shape) -> Result<(usize, usize), AutodiffError>`
- 初期化関数: `uniform`／`normal`／`constant`（RNG 非消費）／`xavier_uniform`／`xavier_normal`／`kaiming_uniform`／`kaiming_normal`／`orthogonal`（`crate::eval::linalg::qr` の Householder reduced QR を再利用）／`trunc_normal`（rejection sampling。逆 CDF 法に必要な erfinv を自作しない判断）
- RNG 源: `fandhe_ai_tensor_core::rng::with_global_rng`（`randn`／`rand` と同じロック粒度。1 回の呼び出しで必要な値をまとめて引く）。決定性の範囲は Box–Muller 経由の関数（`normal`／`xavier_normal`／`kaiming_normal`／`orthogonal`／`trunc_normal`／`std==0` でない `normal` 系）は「同一プロセス・同一プラットフォーム内」、整数演算のみの `uniform`／`xavier_uniform`／`kaiming_uniform`／`constant` はプラットフォーム横断で bit 同一
- 既存の個別シード方式（`uniform_init`・`derive_seed`・`Linear::new(.., seed)` 等）は一切変更しない。`manual_seed` を呼んでもそれらの出力は不変のまま（回帰テスト `linear_new_output_is_unaffected_by_global_manual_seed_state`〈`crates/autodiff/tests/nn_init.rs`〉で固定）
- 新規 `Op`／`BackendOps`／VJP は追加しない（ホスト生成のみ。#1725 と同じ位置づけ）。CUDA／Metal 実機 parity は数値経路が存在しないため対象外

### 2.2 `nn::Linear` の重みレイアウト注意（実装時に判明した現実仕様との差異）

計画時点では PyTorch と同じ `[out_features, in_features]` を前提に `Linear::from_parameters(init::kaiming_normal(&[out, in], ..)?, ..)` という例を想定していたが、実装時に確認したところ `nn::Linear::weight`（`linear.rs`）は `y = input.matmul(weight)` の合成のため **`[in_features, out_features]`**（PyTorch とは転置の関係）だった。

`calculate_fan_in_and_fan_out` は PyTorch と同じ `[out, in, ..kernel_dims]` 規約（`shape[0]` を fan_out、`shape[1]` を fan_in とみなす）で実装したため、これを `nn::Linear::from_parameters` の shape 引数へそのまま使うと fan の意味が入れ替わる。`Conv2d`／`Embedding` の重みレイアウトは PyTorch と同じ `[out_channels, in_channels, ..]` のため本注意は生じない。この非対称性は `calculate_fan_in_and_fan_out` の doc コメントと統合テスト（`linear_from_parameters_accepts_kaiming_normal_weight_and_zero_bias`）内コメントに明記した。facade 公開時（§4 条件付き手順）にもこの注意を引き継ぐ必要がある。

### 2.3 層との互換（受入条件「既存層コンストラクタとの互換確認」の読み替え）

`with_init` は存在しないため、既存の `from_parameters` 経路で互換を確認した: `Linear::from_parameters(init::kaiming_normal(&[in, out], ..)?, Some(init::constant(&[out], 0.0)?))`・`Conv2d::from_parameters(init::kaiming_uniform(&[out_ch, in_ch, kh, kw], ..)?, None, stride, padding, dilation, groups)`。統合テスト（`crates/autodiff/tests/nn_init.rs`）で forward が成立することを確認済み。

## 3. スコープ判断

| 区分 | 内容 | 本 PR での扱い |
|------|------|----------------|
| 承認不要 | `fandhe_ai_autodiff::nn::init` を `pub mod` 化し、初期化関数・補助型を追加（内部クレートへの非破壊追加。crates.io `fandhe-ai-autodiff =0.9.0` の既存 API は不変） | 実施 |
| 承認不要 | 単体テスト・統合テスト・doc 更新・本 decision doc の新設 | 実施 |
| **承認ゲート** | facade `fandhe_ai::nn::init` の新設（`crates/facade/src/nn/init.rs`・`nn/mod.rs` への `pub mod init;`） | **未承認のため保留**（§0） |
| スコープ外 | 各層への `with_init` コンストラクタ・`Initializer` trait の新設、層の既定初期化の変更（`new(.., seed)` は不変）、カスタム初期化 hook | 実施しない |

## 4. 条件付き手順（facade 公開に明示承認が得られた場合のみ）

承認が確認できた場合に限り以下を実施し、承認根拠（コメント URL・日付）を本 doc と `docs/compat-api-scope.md` §5 に「適用記録（経路 2）」として追記する:

1. `crates/facade/src/nn/init.rs` を新設し `pub use fandhe_ai_autodiff::nn::init::{...};` の明示列挙で再エクスポートする（glob 禁止）
2. `crates/facade/src/nn/mod.rs` に `pub mod init;` を追加
3. `crates/facade/tests/api_surface.rs::nn_mod_declares_only_rnn_submodule` の期待値更新（`pub mod rnn;`＋`pub mod init;`）・`lib.rs` の hold doctest（`VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard`）への `use fandhe_ai::nn::init::*;` 追加・`nn_rnn_*` 3 点セットを鏡写しにした固定ガード追加
4. `crates/facade/tests/nn_init.rs`（facade 統合テスト）: `manual_seed` → `nn::init::*` の決定性、既存層構築経路での利用確認
5. §2.2 の重みレイアウト注意を facade 側の doc へも引き継ぐ

## 5. 出典

- `crates/autodiff/src/nn/init.rs`（実装本体）
- `crates/autodiff/tests/nn_init.rs`（統合テスト 30 件）
- `docs/compat-api-scope.md` §5（範囲拡張の手続き）
- `docs/facade-nn-module-exposure-decision.md`（同型の保留パターンの先例）
- `docs/rng-global-contract-design.md`（グローバル RNG 契約）

## 6. PR #2239 codex-review 是正（2026-09-24）

- **`kaiming_uniform`／`kaiming_normal` の `a` 引数**: 当初実装は `a` を有限性検査にのみ使い gain 計算から除外していたため、PyTorch `kaiming_uniform_(tensor, a=..., nonlinearity='leaky_relu')` と同じ `a` を指定しても初期化分散に反映されず、移行容易性契約（AGENTS.md）に反していた。`kaiming_gain`（`crates/autodiff/src/nn/init.rs`）を新設し、`nonlinearity` が `Nonlinearity::LeakyRelu(_)` の場合は `a` を負勾配として採用するよう修正した（`LeakyRelu` 以外では PyTorch と同じく `a` は無視）。§2.1 の型シグネチャ（`Nonlinearity::LeakyRelu(f32)`・`calculate_gain(Nonlinearity) -> f32`）自体は不変
- **`orthogonal` の allocation panic**: `crate::eval::linalg::qr` は内部の `Mat`（`f64` 要素）を非 fallible な確保（`vec![0.0; ..]`／`.collect()` 等）で構築するため（`crates/backend-cpu/src/linalg.rs::qr` も同型の既存実装で同じ設計前提。`eval::linalg` モジュール共通の「shape 整合性は呼び出し元契約」という前提であり本 PR のスコープ外）、`checked_mul` で `usize` オーバーフローを回避できていても `f64` 換算（8 バイト／要素）で `isize::MAX` を超える形状では capacity overflow で panic しうる懸念が codex-review で指摘された。`orthogonal`（`crates/autodiff/src/nn/init.rs`）に `qr` 呼び出し直前の事前検査（`Vec<f64>::try_reserve_exact` による f64 換算の確保可否プローブ。失敗時は `AutodiffError::InvalidArgument` で fail-closed）を追加し、末尾の `scaled` 構築も `try_alloc`（`try_reserve_exact` ベース）へ変更した。`eval::linalg::qr` 自体を fallible 化する全面的な改修（`var.rs`／`grad.rs`／`backend-cpu` 側の同型実装まで波及する）は本修正のスコープ外とした
- **P2: `xavier_uniform`／`xavier_normal` の負 `gain` 未拒否（PR #2239・`init.rs:460`。review thread 未解決分）**: `bound = gain·√(6 / (fan_in + fan_out))`（`xavier_uniform`）・`std = gain·√(2 / (fan_in + fan_out))`（`xavier_normal`）はいずれも非負区間の半幅・非負標準偏差を前提とするため、負の `gain` を素通りさせると `uniform`／`normal` の入力検証（`low > high`・`std < 0` 拒否）と矛盾する意味論になる。PyTorch の `uniform_`（`from > to` 拒否）・`normal_`（`std < 0` 拒否）と同じ設計思想で、`xavier_uniform`／`xavier_normal` の冒頭（`is_finite` 検査の直後、fan 計算・RNG 消費より前）で `gain < 0.0` を `AutodiffError::InvalidArgument` として拒否するよう是正した。テスト（`crates/autodiff/tests/nn_init.rs::xavier_uniform_rejects_negative_gain`／`xavier_normal_rejects_negative_gain`）は拒否に加え RNG を消費していないことも確認する。境界値 `gain == 0.0`（`-0.0` も含む。`-0.0 < 0.0` は偽）は受理され、`xavier_uniform_zero_gain_returns_zeros` で全要素 0 になることを固定した。
  - **同類型の点検（P2 是正時に他 5 関数へ横展開の要否を確認した結果。修正対象は xavier の 2 関数のみ）**: `uniform`（`low > high` は既存実装で拒否済み）・`normal`／`trunc_normal`（`std < 0` は既存実装で拒否済み）・`kaiming_uniform`／`kaiming_normal`（`gain` は `calculate_gain`／`kaiming_gain` から導出され常に非負。`a` が負でも二乗されるため PyTorch と同じ挙動）は問題なし。`orthogonal` は PyTorch `orthogonal_` が符号チェックをせず単なるスケールとして負の gain を受理するため、互換維持のため意図的に対象外とし doc にのみ明記した（挙動変更なし）。`calculate_gain(Nonlinearity::LeakyRelu(非有限))` は単体で呼ぶと NaN を返しうるが、`xavier_*`／`kaiming_*` 側は `gain`／`a`／`bound`／`std` の `is_finite` 検査で NaN 伝播を出力前に遮断するため実害がなく、シグネチャは変更せず doc 注記のみとした。
- **Low follow-up: `trunc_normal` 試行上限超過分岐の直接テスト欠落**: `TRUNC_NORMAL_MAX_ATTEMPTS_PER_ELEMENT` 超過による `AutodiffError::InvalidArgument` 分岐（rejection sampling の DoS 対策）を直接検査するテストがなかったため、`crates/autodiff/tests/nn_init.rs::trunc_normal_rejects_window_exceeding_attempt_cap` を追加した（`[a, b] = [20, 21]` という `N(0, 1)` の裾から大きく外れた窓で受理確率を実質ゼロにし、要素数を小さく保ってテスト時間を抑える）。
- **P1 再指摘: `orthogonal` の probe-then-drop 事前検査では `qr` 内部の同時確保失敗を防げない（PR #2239 再レビュー・`init.rs:692`）**: 上記「`orthogonal` の allocation panic」是正で追加した `Vec<f64>::try_reserve_exact(gen_numel)` プローブは、単一領域を一時的に予約して即解放するだけであり、`crate::eval::linalg::qr` が内部で `Mat::from_tensor`／`Mat::zeros`／`Vec::with_capacity`／`.collect()` 等の**非フォールブルな確保を複数同時に保持する**実行時の失敗を防げないと再指摘された。当初は「`eval::linalg` モジュール共通の設計前提」を理由に `qr` 自体の fallible 化をスコープ外としていたが、再指摘を受けて次のとおり是正した:
  - `crate::eval::linalg::qr`（`crates/autodiff/src/eval/linalg.rs`）のシグネチャを `(Tensor<f32>, Tensor<f32>)` から `Result<(Tensor<f32>, Tensor<f32>), AutodiffError>` へ変更し、内部の全 `f64` 作業領域確保（入力変換・`Q`／`R` 行列・Householder ベクトル `x`／`v`・出力変換）を `try_reserve_exact` 経由のフォールブル確保へ置き換えた。**呼び出し元の入出力・数値結果は一切変更しない**（既存の成功パスは全て従来どおり成功し、確保失敗時のみ `panic`／`abort` から `Err(AutodiffError::InvalidArgument)` へ変わる）。
  - 波及範囲を最小化するため、`Mat::zeros`／`Mat::from_tensor`／`Mat::to_tensor`（`inv`／`solve`／`cholesky`／`svd`／`qr_vjp` 等、他の `eval::linalg` 関数が使う既存の非フォールブル版）は変更せず、`qr` 専用の `Mat::try_zeros`／`Mat::try_from_tensor`／`Mat::try_to_tensor`（新設・`private`）と `try_vec_zeroed`（Householder ベクトル用）を追加した。他関数の「呼び出し元が shape の整合性を保証する契約」（`crates/autodiff/src/eval/linalg.rs` 冒頭コメント）は変更しない。
  - `crate::eval::linalg::qr` の唯一の他の呼び出し元 `var.rs::Var::qr`（既に `Result<QrVars, AutodiffError>` を返す）は `?` で伝播するだけで済み、挙動は不変。
  - `orthogonal` 内の旧プローブ（probe-then-drop）は削除し、`qr(&m)?` へ差し替えた。また `q.transpose_2d()?.contiguous()`（`rows < cols` の場合に `rows·cols` 要素の `f32` を追加で非フォールブルに確保する「テンソル変換」。codex-review 指摘の対象に含まれていた）も、`q.host_slice()` を転置インデックスで読みながら `try_alloc` 済みの `scaled` へ直接書き込む形へ書き換え、`orthogonal` 内の**shape 要素数に比例する**非フォールブル確保をゼロにした（`Tensor::new` 自体は `shape.to_vec()`／`row_major_strides` という rank 比例〈定数オーダー〉の小さな確保を行うが、これは対象外）。不要になった `Q`（`r`）・入力バッファ（`m`）は `scaled` 確保前に明示 `drop` し、ピーク時の同時確保量を減らした。
  - **残る非フォールブル確保（本 PR のスコープ外・明示）**: `Tensor::contiguous()`（非 contiguous な入力の実体化）・`Tensor::from_shape_fill`（`build_tensor` 経由の空 shape フォールバック）は `tensor-core` 全体で共有される既存実装であり、`orthogonal`・`qr` の入力（`Tensor::new` 直後の新規構築値・常に contiguous）では到達しない。`var.rs::Var::qr` 経由で非 contiguous な `Var` を渡す場合はこの経路が残るが、`tensor-core` 全体を横断する改修は本 PR（イシュー #2140）のスコープ外と判断した。**オーバーコミット環境で `try_reserve_exact` が成功を返した後に実メモリ不足で OOM killer が働く経路は、いかなる事前検査でも防げない**（codex-review 自身が明記している既知の限界。我々のコードが保証するのは「確保失敗は panic/abort ではなく `Err` になる」ことのみ）。
  - **同類型の点検（`nn::init` 内の他関数）**: `uniform`／`normal`／`constant`／`xavier_uniform`／`xavier_normal`／`kaiming_uniform`／`kaiming_normal`／`trunc_normal` は全て `try_alloc`／`fill_uniform`／`fill_normal`（`try_reserve_exact` ベースで確保した単一バッファをそのまま保持する設計）のみを使い、非フォールブルな下位関数へバッファを渡してから複数確保させる構造を持たない。`qr` を呼ぶのは `orthogonal` のみであり、probe-then-drop の不備が該当したのは `orthogonal` だけだった（先例点検の結論）。
  - **回帰テスト**: `crates/autodiff/src/eval/linalg.rs::tests::mat_try_zeros_rejects_isize_overflowing_byte_size`／`try_vec_zeroed_rejects_isize_overflowing_byte_size`（`rows*cols*size_of::<f64>()` が `isize::MAX` を超える shape で `Err` になることを、実データを伴わずレイアウト計算のみで即座に検証）、`crates/autodiff/tests/nn_init.rs::orthogonal_rejects_huge_shape_without_panicking`（`orthogonal` のエンドツーエンド呼び出しが巨大 shape で panic せず `Err` に収束することを固定。実際には `fill_normal` が `qr` 到達前に同じオーダーで先に `Err` を返すため、`qr` 内部の新規フォールブル分岐そのものへは到達しないが、パイプライン全体の non-panic 契約を担保する）。既存の `qr` 呼び出し（`crates/autodiff/src/eval/linalg.rs` のテスト・`var.rs`）は全て `?`／`.unwrap()` で `Result` 化に追従済み。
