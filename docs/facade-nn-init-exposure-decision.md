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
  - **残る非フォールブル確保（当時のスコープ外・後続の再指摘で「非 contiguous 入力」分は解消済み。下記「P1 再々指摘」参照）**: `Tensor::from_shape_fill`（`build_tensor` 経由の空 shape フォールバック）は `tensor-core` 全体で共有される既存実装であり、`orthogonal`・`qr` の各種構築経路（要素数 0 の空 shape 分岐のみ）でしか到達しない微小確保のため対象外のまま。**オーバーコミット環境で `try_reserve_exact` が成功を返した後に実メモリ不足で OOM killer が働く経路は、いかなる事前検査でも防げない**（codex-review 自身が明記している既知の限界。我々のコードが保証するのは「確保失敗は panic/abort ではなく `Err` になる」ことのみ）。
  - **同類型の点検（`nn::init` 内の他関数）**: `uniform`／`normal`／`constant`／`xavier_uniform`／`xavier_normal`／`kaiming_uniform`／`kaiming_normal`／`trunc_normal` は全て `try_alloc`／`fill_uniform`／`fill_normal`（`try_reserve_exact` ベースで確保した単一バッファをそのまま保持する設計）のみを使い、非フォールブルな下位関数へバッファを渡してから複数確保させる構造を持たない。`qr` を呼ぶのは `orthogonal` のみであり、probe-then-drop の不備が該当したのは `orthogonal` だけだった（先例点検の結論）。
  - **回帰テスト**: `crates/autodiff/src/eval/linalg.rs::tests::mat_try_zeros_rejects_isize_overflowing_byte_size`／`try_vec_zeroed_rejects_isize_overflowing_byte_size`（`rows*cols*size_of::<f64>()` が `isize::MAX` を超える shape で `Err` になることを、実データを伴わずレイアウト計算のみで即座に検証）、`crates/autodiff/tests/nn_init.rs::orthogonal_rejects_huge_shape_without_panicking`（`orthogonal` のエンドツーエンド呼び出しが巨大 shape で panic せず `Err` に収束することを固定。実際には `fill_normal` が `qr` 到達前に同じオーダーで先に `Err` を返すため、`qr` 内部の新規フォールブル分岐そのものへは到達しないが、パイプライン全体の non-panic 契約を担保する）。既存の `qr` 呼び出し（`crates/autodiff/src/eval/linalg.rs` のテスト・`var.rs`）は全て `?`／`.unwrap()` で `Result` 化に追従済み。
- **P1 再々指摘: 非 contiguous な QR 入力では確保失敗が依然 panic／abort になる（PR #2239 再々レビュー・`linalg.rs:166`）**: 直前の是正で新設した `Mat::try_from_tensor` は `Tensor::host_slice()` を使っていたが、`host_slice()` は非 contiguous な `Tensor` に対して内部で `Tensor::contiguous()`（`tensor-core` 全体で共有される既存実装。`Vec::with_capacity` ベースの非フォールブル確保）を実行してしまう。`var.rs::Var::qr` は転置・narrow 済みの `Var`（非 contiguous な view）をテープが materialize した値としてそのまま渡せる本番経路であり、直前の是正で「`qr` の確保失敗は `Err` で伝播する」とした契約を、この非 contiguous 経路だけが依然として破っていた。
  - `Mat::try_from_tensor`（`crates/autodiff/src/eval/linalg.rs`）を、`Tensor::host_slice()`／`Tensor::contiguous()` を一切使わない実装へ差し替えた: `t.as_slice()`（contiguous なら借用のみ、追加確保なし）が `Some` ならそれを使い、`None`（非 contiguous）なら `Tensor::get`（strides 経由の要素アクセス。`Option` を返すのみで確保を伴わない）で `[0, rows) × [0, cols)` を行優先順に読み、予約済みの `data: Vec<f64>` へ 1 要素ずつ push する——`Tensor::contiguous()` 自身の非 contiguous フォールバック実装（`shape` を行優先順に走査して `get` を呼ぶ）と同じロジックを、確保だけフォールブル化した版として複製した（`tensor-core` の共有実装自体は変更しない。依存方向の制約により `autodiff` → `tensor-core` の改修は避ける）。
  - `orthogonal`（`crates/autodiff/src/nn/init.rs`）側の `q.host_slice()` も `q.as_slice()`（`Err(AutodiffError::InvalidArgument)` フォールバック付き）へ差し替えた。`q` は直前の `Mat::try_to_tensor`（`Tensor::new` で `offset=0`・row-major strides の新規構築）が返す値のため必ず contiguous であり挙動は変わらないが、`Mat::try_from_tensor` で除去した「非 contiguous 時に非フォールブル確保へ迂回する」危険パターンを呼び出し元側でも一貫して避けるための対称的な修正。
  - **回帰テスト**: `crates/autodiff/src/eval/linalg.rs::tests::qr_accepts_non_contiguous_transposed_input`（`transpose(0, 1)` 直後の非 contiguous view を `.contiguous()` を呼ばずそのまま `qr` へ渡し、`.contiguous()` 済みの同値入力で計算した結果と bit 完全一致することを確認。`Mat::try_from_tensor` の `get` ループ分岐を直接運動させる）。
- **P2: 有限な一様分布境界でも幅の計算が overflow する（PR #2239・`init.rs:264`）**: `uniform` は `low`・`high` がそれぞれ有限で `low <= high` なら受理していたが、`fill_uniform` 内部で幅 `high - low` を `f32` のまま計算していたため、`low = -f32::MAX`・`high = f32::MAX` のように両端は有限でも幅自体が `f32` の表現範囲（`f32::MAX ≈ 3.4028e38`）を超えて `inf` になり、出力が `inf`（乱数値が `0` の要素は `0 * inf` で `NaN`）になっていた。
  - `fill_uniform`（`crates/autodiff/src/nn/init.rs`）を `f64` 中間計算へ変更した: 幅 `f64::from(high) - f64::from(low)` は `f64`（最大 `≈1.8e308`）の表現範囲内に必ず収まり、線形補間 `low + t·width` の結果は常に `[low, high]`（両端とも有限な `f32` として検証済み）に収まるため、最後の 1 回の `f32` ダウンキャストで overflow しない。`xavier_uniform`／`kaiming_uniform` の `fill_uniform(numel, -bound, bound)` 呼び出し（幅 `2·bound` が同じ overflow クラスに該当）も本関数経由で同時に是正される。
  - **同類型の点検（PyTorch 挙動を参考に、f64 計算で有限範囲に収める方式を優先。P2 是正時に横展開した結果）**:
    - `fill_normal`（`normal`／`xavier_normal`／`kaiming_normal` が共有）: 旧実装は Box–Muller の `z`（`r * cos(theta)` 等）を直後に `f32` へダウンキャストしてから `f32` で `std` 倍・`mean` 加算していたため、`z`・`std`・`mean` が個別に有限でも中間積 `z * std` が overflow し `inf` になりうる不具合クラスが存在した（例: `z ≈ 1.05`・`std ≈ 3.3e38`・`mean ≈ -3.3e38` は真の値 `z * std + mean ≈ 1.65e37` が `f32` に十分収まるが、`f32` の `z * std` 単体は `f32::MAX` を超えて `inf` になる）。`z` を `f32` へ早期変換せず `f64` のまま `std`・`mean` を `f64` へ昇格して演算し、最後の 1 回だけ `f32` へダウンキャストするよう是正した（`.claude/rules/coding-rust.md` の「最終書き出しのみ 1 回ダウンキャスト」方針と同型）。
    - `trunc_normal`: `fill_normal` と同じ `z * std + mean` 式を独自にインライン実装しており（rejection sampling のため）、同型の中間 overflow クラスを持っていた。旧実装では overflow した値（`inf`）は `[a, b]` 受理判定で必ず棄却される（`inf <= b` が偽になるため）ため**誤った値が出力されることはなかった**が、真の値が `[a, b]` 内に収まるはずのサンプルまで中間 overflow のせいで誤って棄却され、`std` が極端に大きい設定では試行回数上限を無駄に消費しうる欠陥だった。`mean`／`std`／`a`／`b` を `f64` へ昇格し、判定・出力とも `f64` で行ってから受理時にのみ 1 回 `f32` へダウンキャストするよう是正した。
    - `constant`: RNG も乗除算も伴わない単純な埋め込みのため対象外。
    - `orthogonal`: 直交行列への `gain` 乗算（`q_slice[..] * gain`）のみで、`gain` 自体が有限であれば積が overflow するのは真に表現域を超える場合のみ（`q` の各成分は単位ノルム由来で `|q_ij| <= 1` に収まるため `|q_ij * gain| <= |gain|`——`gain` が有限なら積も必ず有限）であり、同型の「両端は有限だが中間量だけ overflow する」構造を持たないため対象外。
  - **回帰テスト**: `crates/autodiff/tests/nn_init.rs::uniform_extreme_finite_bounds_do_not_produce_inf_or_nan`（`low = -f32::MAX`・`high = f32::MAX` で全要素が有限のまま `[low, high]` に収まることを確認）、`xavier_uniform_extreme_gain_does_not_produce_inf_or_nan`（`shape = [1, 1]` で `bound` を `f32::MAX` 近くまで押し上げる極端な `gain` でも有限であることを確認）、`trunc_normal_accepts_samples_whose_intermediate_product_would_overflow_f32`（`std ≈ 0.97 * f32::MAX`・`mean = -std` で「中間 overflow するが真の値は `[a, b]` 内」の帯域（`z ≈ 1.03〜1.18`）のサンプルが正しく受理され、受理された最大値が旧実装の頭打ち値〈約 `1.02e37`〉を明確に超えることを確認）。`normal` 自体は「`mean` で系統的に打ち消す」構成だと典型的な `z`（例えば `z ≈ 0`）でも真に非有限になる帯域が広く、全出力への一律の有限性検査を書けない（実際に試したところ正しく非有限になるはずのサンプルで誤って fail した）ため、同じ数式を共有する `trunc_normal`（`[a, b]` 受理窓により「受理された値は必ず有限」という不変量を持つ）側で固定した。
- **P1 四度目指摘: 確保失敗後のエラー生成が再度非フォールブルな確保を行う（PR #2239 四度目レビュー・`init.rs:253`）**: `try_reserve_exact` の失敗を `format!`／`Into<String>` 経由で `AutodiffError::InvalidArgument(String)` へ変換する箇所（`nn::init::try_alloc` の `format!("nn::init: 要素数 {len} の確保に失敗しました")` が代表例。同型の問題が `eval::linalg.rs` の `Mat::try_zeros`／`try_from_tensor`／`try_to_tensor`／`try_vec_zeroed` にもあり、固定文字列でも `.into()` は新規 `String` を確保する）は、確保失敗を検出した**直後**にもう一度ヒープ確保を試みる構造になっており、実メモリ枯渇時にはこのエラー文字列の構築自体が `handle_alloc_error` 経由の abort を招きうる——「確保失敗は panic／abort ではなく `Err` で伝播する」契約を、確保失敗の*報告*経路自体が破っていた。
  - **既存の非アロケーションなエラー型の調査結果（新規 variant は追加しない）**: `AutodiffError` は `#[non_exhaustive]` の公開型で `facade`（`fandhe_ai::AutodiffError`）が再エクスポートするため、新規 variant の追加は公開面の変更としてユーザー承認が必要になる。調査の結果、`fandhe_ai_tensor_core::ShapeError::ElementCountOverflow`（`tensor-core/src/error.rs`）が「shape の要素数積が `usize` の範囲でオーバーフローする、または要素型込みのバイトサイズが `Vec` の allocation 上限（`isize::MAX` バイト）を超えアロケーション不能な shape」という**本件と全く同じ意味論**を持つ非データ（unit）variant として既に存在し（`Tensor::zeros`／`ones`／`full` が既に同じ用途で使っている確立済みパターン）、`AutodiffError` は `From<ShapeError>` を既に実装しているため、新規 variant を追加せずそのまま再利用できると判断した。**AutodiffError への新規 variant 追加は不要だったため、ユーザー承認を要する変更はなかった**（`docs/facade-nn-init-exposure-decision.md` 冒頭「facade 公開は未承認」の状態自体も変えていない——`nn::init`／`eval::linalg::qr` はいずれも facade 未公開のため、本 PR での挙動変化は facade 経由では観測されない）。
  - `nn::init`（`init.rs`）・`eval::linalg`（`linalg.rs`）それぞれに非アロケーションな `alloc_failed() -> AutodiffError { AutodiffError::Shape(ShapeError::ElementCountOverflow) }` を新設した（クレート内 `private` 関数。`linalg.rs` には `use fandhe_ai_tensor_core::ShapeError;` を追加）。`ShapeError::ElementCountOverflow`（unit variant・`Clone`／`Eq` 導出のみで `String` 等のヒープ保持フィールドを持たない）の構築はスタック上のデータ移動のみで完結し、ヒープ確保を一切伴わない。
  - **類型化した適用範囲（「確保対象のサイズ計算・確保そのものが失敗した」エラーのみに適用。引数の意味論検証エラーは対象外）**: 「確保サイズ計算・確保失敗」エラーは全て `alloc_failed()` へ切り替え、`gain` の符号・`low <= high`・`std < 0`・`rank` 不足・軸が 0 等の**意味論**検証エラー（確保が一切絡まず、システムが実メモリ枯渇状態にあるとは限らないタイミングで発生するため診断メッセージを保持する価値の方が上回る）は `invalid_argument`／`invalid`（`String` メッセージ付き）のまま維持した。全該当箇所（14 件）を棚卸しした結果は下表のとおり:

    | ファイル・関数 | 元の分岐 | 種別 | 対処 |
    |---|---|---|---|
    | `init.rs::checked_numel` | 要素数積の `checked_mul` オーバーフロー | サイズ計算 | `alloc_failed()` |
    | `init.rs::try_alloc` | `try_reserve_exact` 失敗（当初の指摘箇所） | 確保失敗 | `alloc_failed()` |
    | `init.rs::calculate_fan_in_and_fan_out` | `receptive_field_size` の `checked_mul` | サイズ計算 | `alloc_failed()` |
    | 同上 | `fan_in` の `checked_mul` | サイズ計算 | `alloc_failed()` |
    | 同上 | `fan_out` の `checked_mul` | サイズ計算 | `alloc_failed()` |
    | `init.rs::orthogonal` | `cols`（`shape[1..]` 積）の `checked_mul` | サイズ計算 | `alloc_failed()` |
    | 同上 | `gen_numel` の `checked_mul` | サイズ計算 | `alloc_failed()` |
    | `init.rs::orthogonal` | `gain`／`shape` 各軸／`q` の contiguous 契約違反 | 意味論検証 | 対象外（`invalid_argument` 維持） |
    | `init.rs::uniform`／`normal`／`xavier_*`／`kaiming_*`／`trunc_normal` | `low <= high`・`std < 0`・`gain` 符号・`a` 有限性・`bound`／`std` 有限性・`fan == 0`・`a >= b` 等 | 意味論検証 | 対象外（`invalid_argument` 維持） |
    | `linalg.rs::Mat::try_zeros` | `rows.checked_mul(cols)` | サイズ計算 | `alloc_failed()` |
    | 同上 | `try_reserve_exact` 失敗 | 確保失敗 | `alloc_failed()` |
    | `linalg.rs::Mat::try_from_tensor` | `rows.checked_mul(cols)` | サイズ計算 | `alloc_failed()` |
    | 同上 | `try_reserve_exact` 失敗 | 確保失敗 | `alloc_failed()` |
    | `linalg.rs::Mat::try_to_tensor` | `try_reserve_exact` 失敗 | 確保失敗 | `alloc_failed()` |
    | `linalg.rs::try_vec_zeroed` | `try_reserve_exact` 失敗 | 確保失敗 | `alloc_failed()` |
    | `linalg.rs::qr` | `reflectors.try_reserve_exact` 失敗 | 確保失敗 | `alloc_failed()` |
    | `linalg.rs`（`inv`／`solve`／`cholesky`／`svd`／`qr_vjp` 等） | 特異・非正定値・非収束等の数値的失敗 | 意味論検証（`qr` 経路外） | 対象外（`invalid` 維持。`qr` 専用のナローな適用） |

  - **残る `format!`／`String` 化・`Box` 化の経路（`qr` 経路内で確認した限りゼロ）**: `qr`・`Mat::try_zeros`／`try_from_tensor`／`try_to_tensor`・`try_vec_zeroed`・`nn::init::try_alloc`／`checked_numel`／`calculate_fan_in_and_fan_out`／`orthogonal`（サイズ計算箇所）を全件確認し、確保失敗の報告経路に `format!`／`.into()`／`.to_string()`／`Vec` 化／`Box` 化は残っていないことを確認した。唯一残る非フォールブル確保は `Mat::try_to_tensor` が最後に呼ぶ `Tensor::new(data, shape)`（`tensor-core::Tensor::new`）内部の `shape.to_vec()`（`Vec<usize>`）・`row_major_strides(shape)`（`Vec<isize>`）・`Arc::new(Storage { data })`——いずれも要素数 `numel` ではなく `rank`（典型的に 2）に比例する O(1) 相当の小さな確保であり、`Mat::try_to_tensor` 自身が既に排除した O(numel) 確保（`data: Vec<f32>` 本体。`try_reserve_exact` 済み）とは規模が全く異なる。`tensor-core` 全体で共有される既存実装であり、依存方向の制約（`autodiff` → `tensor-core` の改修は本 PR のスコープ外）によりこれ以上のフォールブル化は行わない。`Mat::try_from_tensor` の `debug_assert!` マクロ（`{r}, {c}` を含むフォーマット文字列）は、shape 走査ロジックのバグ検出専用でありデバッグビルドでのみアサーション失敗時に評価され（release ビルドでは完全にコンパイル対象外）、正常系（OOM を含む）の実行パスには影響しない。
  - **ユーザー承認が必要な既知の同類型（本 PR ではスコープ外・報告のみ）**: `crates/autodiff/src/nn/conv.rs::checked_uniform_init`・`crates/autodiff/src/nn/rnn.rs::checked_uniform_init`／`reserve_outputs`・`crates/autodiff/src/nn/embedding.rs::Embedding::new` は、いずれも `try_uniform_init`／`try_normal_init`（`nn::init` 由来。`TryReserveError` を返す）の `Err` を `AutodiffError::InvalidArgument(format!("...{err}"))` へ変換しており、本 PR で是正した P1 と全く同じ欠陥クラスを持つ（イシュー #1604／#1647／#1770 由来の既存コードで、本 PR〈イシュー #2140〉より前から存在する）。`nn::init`（本イシューのスコープ）の外側であるため本 PR では修正しない。ユーザーに別 Issue での追跡要否を確認する必要がある（`.claude/rules/out-of-scope-tracking.md`）。
- **P2 四度目指摘: 大きな有限の LeakyReLU slope で gain が 0 に潰れる（PR #2239 四度目レビュー・`init.rs:381`）**: `calculate_gain(Nonlinearity::LeakyRelu(negative_slope))` は `negative_slope * negative_slope` を `f32` のまま計算していたため、`|negative_slope| > √f32::MAX ≈ 1.85e19` という**有限**な入力で中間値が `inf` になり、`2.0 / (1.0 + inf) == 0.0` の `sqrt` 経由で `calculate_gain` が誤って `0.0` を返していた（`negative_slope = f32::MAX` の正しい gain は約 `4.2e-39`——`f32` の subnormal 域だが表現可能）。`kaiming_uniform`／`kaiming_normal` は `gain == 0.0` だと `std == 0`／`bound == 0` になり重みが全て 0 で初期化されてしまう。
  - `calculate_gain` を `calculate_gain_f64`（新設・`private`）＋`as f32` ダウンキャストへ分離し、二乗・除算・平方根を `f64` で計算してから最後の 1 回だけ `f32` へダウンキャストするよう是正した。`kaiming_gain` も `f64` を返すよう変更し、`kaiming_uniform`／`kaiming_normal` は `std`／`bound` の計算（`gain / √fan`・`std * √3`）も `f64`（`fan as f64`）で行ってから最後にダウンキャストする。`xavier_uniform`／`xavier_normal` の `bound`／`std`（`gain * √(6/denom)`・`gain * √(2/denom)`）も同じ理由で `f64` 化した。
  - **挙動変化（1 点。doc に明記）**: `calculate_gain(LeakyRelu(f32::INFINITY))` は旧実装では `f32` の不定形経路（`1.0 + inf` の除算）を辿り `NaN` を返していたが、`f64` でも同じ極限（`negative_slope² → inf`・`2/(1+inf) → 0`）を辿るため `0.0`（数学的な極限値と一致し `NaN` より意味のある結果）を返すようになる。`negative_slope` は有限値の想定（PyTorch の `leaky_relu` も有限のスロープを前提とする）であり、呼び出し元（`kaiming_uniform`／`kaiming_normal`）は `a`（`negative_slope` の実引数）の有限性を事前検査するため実害はない。
  - **`init.rs` の全 `f32` 中間演算の棚卸し（P2 の「類型」点検。式・旧挙動・対処）**:

    | 式（関数） | 旧挙動 | 対処 |
    |---|---|---|
    | `LeakyRelu`: `negative_slope * negative_slope`（`calculate_gain`） | `\|negative_slope\| > 1.85e19` で中間値 `inf` → `gain` が誤って `0` | **f64 化（本件の主対象）** |
    | `calculate_gain` の他 4 分岐（`Linear`／`Sigmoid`／`Tanh`／`Relu`／`Selu`） | 定数（RHS に変数を含まない） | 対象外（`f64` 定数へ統一のみ実施。挙動不変） |
    | `xavier_uniform`: `gain * (6.0 / denom as f32).sqrt()` | 単発の `f32` 演算。真に表現域を超える場合のみ `is_finite` 検査で拒否（既存で安全）だが `denom`（`usize`）を `f32` へ変換する際 `2^24` 超で丸め誤差が生じうる | `f64` 統一（一貫性・精度向上目的。挙動不変の範囲を拡大） |
    | `xavier_normal`: `gain * (2.0 / denom as f32).sqrt()` | 同上 | 同上 |
    | `kaiming_uniform`／`kaiming_normal`: `gain / (fan as f32).sqrt()`・`std * 3f32.sqrt()` | `gain` 自体が `calculate_gain` 経由で `√2` 程度以下のため単体では overflow しないが、`fan as f32` は `fan > 2^24` で丸め誤差が生じうる | `f64` 統一 |
    | `calculate_fan_in_and_fan_out`: `shape[2..]`／`fan_in`／`fan_out` の `checked_mul` | `usize` の `checked_mul`（P1 側で確保失敗として扱う対象。P2 の f32 中間演算とは別軸） | P1 側で `alloc_failed()` へ是正済み（重複対応なし） |
    | `fill_uniform` の幅 `high - low` | 前回（P2・`init.rs:264`）是正済み | — |
    | `fill_normal`／`trunc_normal` の `z * std + mean` | 前回（同上）是正済み | — |
    | `orthogonal`: `q_slice[..] * gain` | `q` の各成分は単位ノルム由来で `\|q_ij\| <= 1` に収まるため `\|q_ij * gain\| <= \|gain\|`——`gain` が有限なら積も必ず有限（証明のみ。コード変更なし） | 対象外 |

  - **回帰テスト**: `crates/autodiff/tests/nn_init.rs::calculate_gain_leaky_relu_large_finite_slope_does_not_collapse_to_zero`（`LeakyRelu(f32::MAX)` の gain が有限かつ非ゼロであることを確認）、`kaiming_uniform_large_finite_a_does_not_produce_all_zero_weights`（`a = f32::MAX`・`LeakyRelu` で呼んでも全要素が `0.0` に潰れないことを確認）。既存の `calculate_gain_matches_known_values`（許容誤差付き比較のため `f64` 経路への変更後も通過）・`kaiming_uniform_uses_a_as_leaky_relu_negative_slope`（同上）は変更なしで通過することを確認した。
