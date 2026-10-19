# GPU 線形代数（CUDA／Metal）の設計判断記録（イシュー #2673・親 #2672）

コード変更なしの設計判断記録。子 issue #2674（CUDA の LU 系）・#2675（CUDA の cholesky／QR／eigh）・
#2676（Metal f32 の LU／cholesky／QR）が迷わず実装できるよう、優先順位・アルゴリズム・数値契約・
parity 判定方式・公開形の推奨を確定させる。tolerance・baseline・`Cargo.toml`・ガードレール閾値・
`docs/spec/` は変更しない。公開 API（`fandhe-ai =0.10.0`）は壊さない。

## 0. 結論（要約）

| 論点 | 決定 |
|---|---|
| 優先順位 | LU 系（solve／inv／det）> cholesky > QR > eigh（CUDA のみ）。SVD と Metal eigh は対象外（§2） |
| 並列化 | 1 行列 = 1 thread block（threadgroup）。k ステップは block 内同期で区切る。各出力要素の算術順序は CPU 参照実装の逐次ループと同一（§3） |
| 数値契約 | CUDA はネイティブ `double` 内部計算＋出力時 1 回 `f32` 丸め、FMA 縮約禁止。Metal はネイティブ f32（`precise::`・FMA 縮約なし）。soft_f64 は不採用（§4） |
| parity | 実測ベースライン方式は**現時点で不要**。統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）を厳密ゼロ fail で適用。入力集合は条件数上限で固定（§5） |
| 既定経路 | CUDA・Metal とも内部 opt-in ゲート**既定 OFF**。OFF 時は `Unsupported`（現行のホスト f64 結果と bit 同一）（§6） |
| 公開形 | facade 独自の opt-in 関数（`set_cuda_gpu_linalg_enabled`／`set_metal_gpu_linalg_enabled` 案）。既存 `Var::inv` 等から到達し新規演算名は足さない。公開は承認後に別 issue（§7） |

## 1. 背景・目的

- 現状: `tensor-core::BackendOps` の `linalg_*` 12 メソッドは既定 `Unsupported`（`crates/tensor-core/src/backend_ops.rs:4453`〜）。CUDA／Metal は先頭 7 メソッドを明示的に `Unsupported` でオーバーライド（`crates/backend-cuda/src/ops.rs:4389`・`crates/backend-metal/src/ops.rs:3464`。文言は「#1621 スコープ外」）。`Var::inv` 等は `Unsupported` のときだけホストの `eval::linalg`（f64 内部計算）へフォールバックする（`crates/autodiff/src/var.rs:6740` 付近、`docs/autodiff-linalg-design.md` §3）。VJP は常にホスト f64 のため、GPU 化の影響は forward に限られる
- 分担: 親 #2672（自作カーネルで GPU 化）／本件 #2673（設計記録）／#2674・#2675（CUDA）／#2676（Metal f32）
- cuSOLVER・MPS は許容依存 10 区分の外（`.claude/rules/deps-policy.md`）のため採らず、依存追加なしの自作カーネルとする

## 2. 対象演算と優先順位

| 優先 | 演算 | CUDA | Metal | 根拠 |
|---|---|---|---|---|
| 1 | LU 系（solve／inv／det） | #2674 | #2676 | 他演算の土台となる因数分解。逐次依存が浅い |
| 2 | cholesky | #2675 | #2676 | SPD 限定。列ごと並列で構造が最も単純 |
| 3 | QR（Householder） | #2675 | #2676 | 列ノルムの縮約を伴う |
| 4 | eigh（巡回 Jacobi） | #2675 | 対象外（子 issue なし） | 反復法。f32 では `EIGH_JACOBI_EPS = 1e-14`（`crates/backend-cpu/src/linalg.rs:1033`）の収束判定が意味を持たない |
| 5 | SVD（片側 Jacobi） | 対象外 | 対象外 | 同上。後回し |

- 「LU」は solve／inv／det が内部で共有する因数分解を指す。公開演算 `lu`／`lu_factor` は現状存在せず、新設は「CPU 参照実装を先に作る」契約の対象になるためスコープ外
- `slogdet`（LU 再利用可）・`matrix_norm`（Fro／One／Inf は縮約、Nuc／Spectral は SVD 依存）・`pinv`／`lstsq`／`matrix_rank`（SVD 依存）は派生演算または後回しとし、既定 `Unsupported`（ホストフォールバック）を維持する
- 上記スコープ外事項は未起票。起票はユーザー承認後（`.claude/rules/out-of-scope-tracking.md`）

## 3. アルゴリズムと並列化方式

対象は rank-2 限定（既存契約と同じ）。初期実装は 1 行列 = 1 thread block（threadgroup）でカーネル内で k ステップを回し、ステップ間は block 内同期で区切る。ステップごとの起動分割や blocked LU は性能最適化でスコープ外。参照は `crates/backend-cpu/src/linalg.rs`。

- **LU（部分ピボット・右方向）**: ピボット探索は block 内比較縮約。同値は最小添字を選び CPU の「より大きい」更新と一致させる（比較のみで算術順序に影響しない）。行入替・`factor = a_ik / pivot`・末尾部分行列更新 `a_ic - factor * a_kc` は要素ごと 1 スレッド。`factor == 0.0` で更新を飛ばす規則も CPU と同じ
- **solve／inv**: LU 後、右辺の各列を 1 スレッドに割り当て、前進代入（昇順）・後退代入（降順）を逐次実行。inv は B = I
- **det**: LU の対角と置換符号のみ読み戻し、積はホスト側で `lu_diag_product`（`crates/backend-cpu/src/linalg.rs:426`）と同じ演算列を複製して計算
- **cholesky**: 列 j を順に処理。`L[j][j]` 確定後、i > j の各要素を 1 スレッドが担当し内積 `Σ_k L[i][k]·L[j][k]`（k 昇順）を逐次累積。非正定値は対角の非有限または非正で判定
- **QR（Householder）**: 列ノルムは 1 スレッドで逐次計算し順序を保つ。反射適用は要素ごと並列。R 対角を非負にそろえる規約と O(mk) メモリで Q を作る方式は CPU と同じ
- **eigh（CUDA のみ）**: 回転は巡回順序（固定順）を保ち、各回転の行・列更新のみ要素並列。Brent–Luk 等の並列順序は結合順序が変わるため採らない。昇順ソート・同値時の安定順・`normalize_max_abs_sign` は読み戻し後にホスト側複製処理で適用
- **ホスト後処理の複製**: `backend-cuda`／`backend-metal` の通常依存は `tensor-core` のみ（`backend-cpu` は dev-dependency）。`lu_diag_product`・符号正規化・ソートは各 GPU クレートへ意図的に複製する（`eval::linalg` と `backend-cpu::linalg` の関係と同じ扱い）。CPU との一致は dev-dependency の `backend-cpu` を使うテストで固定する

## 4. 数値契約

- **CUDA**: ネイティブ `double` で内部計算し出力時に 1 回だけ `f32` へ落とす（`docs/autodiff-linalg-design.md` §3.5 と同契約。f64 ネイティブカーネルの先例は `crates/backend-cuda/src/typed_f64.rs`）。CPU 参照実装は `mul_add` を使わず乗算・減算を別々に行うため、カーネルは FMA 縮約を明示禁止する（`__dmul_rn`／`__dsub_rn`／`__ddiv_rn`／`__dsqrt_rn`、または NVRTC の fmad 無効化）。目標は CPU と bit 一致だが、**受け入れ判定は REQ-2 統一複合判定**とし bit 一致は受け入れ条件にしない。matmul 系 FMA 契約（`f32::mul_add`）は不変で、線形代数は「FMA なしの CPU 線形代数参照実装に合わせる」別軸として扱う
- **Metal**: ネイティブ f32（#2672／#2676 の指定）。`MathMode::Safe` と `precise::` 系（除算・sqrt）で FMA 縮約を抑止する。soft_f64（`crates/backend-metal/src/soft_f64.rs`）は不採用: 正しく丸めた sqrt がなく Cholesky／QR を bit 一致させられない／LU 更新は O(n³) でエミュレーションコストが過大。ホストの f64 フォールバックに対し**精度クラスが変わる**
- **Metal QR のノルム計算のスケール対策**: CPU の `qr` は列ノルム `norm_x` と反射ベクトルノルム `norm_v` を f64 の二乗和で求めるが、Metal のネイティブ f32 で素朴に二乗和を取ると、条件数 1 の `A = 1e20 × I₂` でも二乗が overflow して NaN（`1e-20` 倍では underflow して 0）になる。Metal の `norm_x`・`norm_v`（および Householder 係数に使う他のノルム・二乗和）は、`.claude/rules/coding-rust.md` の正規化統計と同方針で **scale/ssq 方式（LAPACK SLASSQ 系。最大絶対値を `scale` に括り出し、残りを比の二乗で蓄積して二乗を直接計算しない）**を用い、NaN／inf 入力の伝播を明示的に扱う。CUDA はネイティブ `double` のため同方式は不要だが、`1e20` 倍の二乗（1e40）は f64 範囲内であることを確認対象に含める
- **エラー写像**: カーネル検出の「ピボット厳密 0」（**solve／inv に限る**。det は特異行列で `0.0` を正常に返す CPU 契約〈`backend-cpu::linalg::det`・`docs/autodiff-linalg-design.md`〉に合わせ、ピボット 0 をエラーにせず `0.0` を返す分岐を #2676 へ申し送る）「非正定値の対角」「Jacobi 非収束」はデバイス側フラグを読み戻して `BackendError::InvalidArgument`。`Unsupported` は「未実装またはゲート OFF」に限る。非有限値は CPU と同じ（LU 素通し・Cholesky 拒否・eigh は入口で `require_finite` が拒否）。n = 0 はデバイス初期化前にホストで CPU と同じ結果を返す
- **REQ-8**: カーネル側の手動境界チェックを省略しない。確保前にバイト数上限検査と shape 検査（`require_square_2d` 相当）を行い、ゲート判定はデバイス初期化より前に置く

## 5. parity 判定方式

- 結論: **実測ベースライン方式は現時点で不要**
- 理由: §3 の方式は各出力要素の算術順序を CPU 参照実装の逐次ループと同一に保つため、`.claude/rules/coding-rust.md` の「単一の連続 K ループと異なる結合順序で部分和を求めるカーネル」に当たらない。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満、`assert_parity`）を厳密ゼロ fail でそのまま適用する
- Metal f32 と CPU f64 の差は結合順序ではなく精度クラスの差であり、baseline 方式の適用対象外。baseline で吸収しない。代わりに parity 入力集合を固定する: 対角優位行列・既知の κ を持つ SPD 行列・n ≤ 64 程度、条件数 κ ≤ 1e3 を上限とし、悪条件入力は受け入れ集合に入れない
- 条件数とは独立した**大小スケール入力**（行列全体を `1e20` 倍・`1e-20` 倍した対角優位行列・SPD 行列、`1e20 × I₂` を含む）を parity の確認対象に加える。f32 の二乗が overflow／underflow する領域であり、Metal QR（`norm_x`・`norm_v`）が NaN／0 を返さず統一複合判定を満たすことを確認する（条件数上限 κ ≤ 1e3 とは別軸）
- 将来ブロック化・並列リダクションで順序を変える場合や実機で fail が出た場合も tolerance は緩めない。カーネルを是正するか、別の決定記録とユーザー承認を経て baseline 方式（または spec への提案）を検討する
- 実機 parity は `#[ignore]` で分離し、未実測は `docs/perf/logs/<slug>-<issue>/README.md` へ申し送る（子 issue の責務）

## 6. 既定経路と opt-in ゲート

- 推奨: CUDA・Metal とも**内部 opt-in ゲート既定 OFF**。案: `backend-cuda::linalg::set_gpu_linalg_enabled`／`backend-metal::linalg::set_gpu_linalg_enabled`（`crates/backend-metal/src/split_k_runtime.rs:93` の `set_split_k_enabled` と同形）。OFF 時はデバイスに触れず `Unsupported` を返し、現行（ホスト f64）と bit 単位で同一に保つ
- 理由: Metal は精度クラスが変わる（既存メソッドの意味論を変えない）。CUDA は小行列の H2D／D2H コストと GB10 の FP64 スループットが未実測。先例（TF32・split-K・ONNX GPU 実行）も opt-in 開始
- 既定 ON への切替は実機 parity・性能の実測後に別途ユーザー承認を得て行う

## 7. 公開形の推奨（1 案）

- 推奨: **facade 独自の opt-in 関数**（`crates/facade/src/lib.rs:2353` の `set_metal_split_k_gemm_enabled`・`:2383` の `set_cuda_onnx_gpu_execution_enabled` と同形）。名前案 `fandhe_ai::set_cuda_gpu_linalg_enabled(bool)`／`set_metal_gpu_linalg_enabled(bool)`
- 演算は既存の公開 `Var::inv`／`solve`／`det`／`cholesky`／`qr`／`eigh` から到達でき、`BackendOps` オーバーライドは新規公開名を足さない。3 類型（`Var` 委譲／モジュール再エクスポート／`Sequential::add_*`）は採らない。追加のみで `=0.10.0` に非破壊
- 保留ガード（子 issue が内部ゲートを足す際に置く）: `crates/facade/src/lib.rs` に `GpuLinalgOptInHoldDoctestGuard`、`crates/facade/tests/api_surface.rs` に否定ガード（facade が当該 setter と `linalg` モジュールを再エクスポートしないことの固定）
- 公開担当 #2678／#2679 は closed のため、facade 公開は承認後に別 issue で行う。起票はユーザー承認後

## 8. 子 issue への申し送り

- `crates/backend-cpu/tests/backend_ops_dispatch.rs:246` 付近は CUDA／Metal の `linalg_*` が `Unsupported` を返すことを検査する。ゲート OFF 既定でこの検査が成り立つことを確認し、ゲート ON の挙動は別テストで扱う
- `ops.rs` の「#1621 スコープ外」文言をゲート OFF・未実装の文言へ更新する
- `BackendOps::linalg_*` の trait doc を更新する（`log_softmax` #2155、`docs/gpu-log-softmax-forward-decision.md` と同じ扱い）
- `docs/perf/framework-compare-feature-matrix-0.11.0.md` の「線形代数は CPU のみ」は次版の監査で更新する（本件では変更しない）

## 9. スコープ外

バッチ次元／blocked LU・ステップ別起動などの性能最適化／Metal の eigh・SVD／CUDA の SVD／`lu`・`lu_factor` の新規公開演算／slogdet・pinv・lstsq・matrix_rank・matrix_norm の GPU 化／GPU 側 VJP／Metal での f64／既定 ON 化／facade 公開の実施。

## 10. 実測状況

コード変更なし（本記録作成時点）。実機実測は子 issue の担当。

### #2674（CUDA LU 系）の実施内容

- ゲート `fandhe_ai_backend_cuda::linalg::set_gpu_linalg_enabled`（既定 OFF・facade 非公開。保留ガードは
  `crates/facade/tests/api_surface.rs::facade_does_not_expose_gpu_linalg_optin`）、LU／代入カーネルソース、
  NVRTC ロード、入力検証、ホスト後処理の複製を追加（`unsafe` なし）。
- **カーネル起動（新規 `unsafe` 2 ブロック）はユーザー承認が確認できず未実装**。ゲート ON でも fail-closed の
  `Unsupported`。承認依頼と残作業は `docs/perf/logs/cuda-linalg-lu-2674/README.md`。実機は未実測。

### #2675（CUDA cholesky・QR・eigh）の実施内容

- 同じゲート（`set_gpu_linalg_enabled`。既定 OFF・facade 非公開）の背後に、Cholesky・reduced Householder QR・巡回 Jacobi
  eigh の `double` カーネルソース（`kernels_linalg_factor.rs`）、サイズ検証、ホスト後処理（符号正規化・昇順ソート・QR 仕上げ）、
  NVRTC ロード（`CudaLinalgFactor`）を追加した（`unsafe` なし）。`linalg_eigh` は CUDA 初のオーバーライド。
- **カーネル起動（新規 `unsafe` 3 ブロック）はユーザー承認が確認できず未実装**。ゲート ON でも fail-closed の
  `Unsupported`。承認依頼と残作業は `docs/perf/logs/cuda-linalg-chol-qr-eigh-2675/README.md`。実機は未実測。
- 公開形は §7 の推奨を参照（本 issue では facade 公開しない）。
- **Stage B（#3136）**: D-C 承認（親 #3133）と `docs/gpu-linalg-kernel-launch-unsafe-audit.md` の CU-3〜CU-5 に基づき、3 カーネルの起動（新規 `unsafe` 3 ブロック。`CudaLinalgFactor::run_{cholesky,qr,eigh}_f32`）と `ops.rs` の結線を実装した。`status`／`refl_valid` は sentinel 初期化のうえ必ず読み戻し、非正定値と eigh 非収束は `InvalidArgument` へ写像する。ゲート OFF 既定の挙動は不変。実機での一致確認は #3138（未実施）。

### #2676（Metal LU 系・cholesky・QR）の実施内容

- 内部 opt-in ゲート（`fandhe_ai_backend_metal::linalg::set_gpu_linalg_enabled`。既定 OFF・facade 非公開）の背後に、
  部分ピボット LU・LU 代入・cholesky・reduced Householder QR の f32 MSL カーネル（`shaders/linalg.metal`。
  `#pragma clang fp contract(off)`・`precise::sqrt`／`precise::divide`・QR は scale/ssq ノルム）、サイズ検証、
  ホスト後処理の複製、パイプライン構築（`MetalLinalg::new`）、カーネルと演算順序が同一の f32 ホスト逐語モデル
  （`linalg_model.rs`）を追加した（`unsafe` なし）。Metal の eigh は §2 のとおり対象外。
- 起動前の数値契約検証: f32 ホストモデルと CPU 参照実装（f64）を統一複合判定（`assert_parity`・厳密ゼロ fail）で
  突き合わせるテストを Linux でも実行する（`tests/linalg_model_parity.rs`）。MSL の構文は M4 Max 実機で
  パイプライン構築が通ることを確認した（`docs/perf/logs/metal-linalg-lu-chol-qr-2676/README.md`）。
- **カーネル起動（新規 `unsafe`）はユーザー承認が確認できず未実装**。ゲート ON でも fail-closed の `Unsupported`。
  承認依頼と残作業は同 perf log。実機 parity は未実測。

### #3139（Metal 実機 parity・Stage C）の実施内容

- 実機テストの入力集合を §5 に揃えた（正本 `crates/backend-metal/tests/linalg_parity_inputs.rs`・κ₂ ≤ 1e3 ガード
  `linalg_parity_inputs_guard.rs`）。QR 17×17 の旧入力は κ₂ ≈ 3.5e16 で §5 に反するため是正、det は n ≤ 17 に限定
  （n=64 は f32 範囲外で `inf - inf = NaN`）、QR に 1e±20 倍と `1e20 × I₂`（finite・R 対角非 0・正規化 R parity 付き）を追加。
  tolerance・判定方式（§5）は不変。
- Mac mini 実機の結果は **未実測**（実機に到達できなかった）。事前登録と記録は `docs/perf/logs/metal-linalg-parity-3139/`。

### #3135（CUDA LU 系 Stage B）の実施内容

- `CudaLinalg::launch_lu`（CU-1）・`run_lu_solve_f32`（CU-2 を含む）・`run_lu_inv_f32`・`run_lu_det_f32` を追加し、
  `ops.rs` の `lu_launch_pending`（fail-closed の `Unsupported`）を実起動へ置換した。新規 `unsafe` は 2 ブロックのみ
  （監査記録 `docs/gpu-linalg-kernel-launch-unsafe-audit.md`。承認状態は同記録 §1 のとおり親 #3133 本文の記載に基づく。
  security-auditor の独立監査（eaf7141e・静的突合）は P0・P1 なし。P2 の対応と実機・NVRTC 未確認は同 §8.0 を参照）。
- 数値契約: H2D の前にホストで `f32`→`f64`、D2H の後に 1 回 `as f32`。`status` を必ず読み戻す。特異は solve／inv が
  `InvalidArgument`、det が `0.0`（CPU と同じ）。`CudaError` に variant は追加せず `LuSolveOutcome` で表す。
- 既定（ゲート OFF）の挙動は不変（`Unsupported` → `Var` がホスト f64 へフォールバック）。実機（GB10）での NVRTC・parity・
  特異写像の実測は #3138（本 PR では未検証）。

### #3138（CUDA 実機 parity Stage C）の実施内容

事前登録規則・測定手順・記入欄と facade 経由ゲート ON の `#[ignore]` テスト（`crates/facade/tests/linalg_gpu_optin_cuda_parity.rs`）を用意した（`docs/perf/logs/cuda-linalg-parity-3138/`）。GB10 での実測は未実施（申し送り）。判定方式（§5）・tolerance・入力集合は不変。

### #3444（ROCm linalg 6 演算の移植）の実施内容

`docs/backend-rocm-ops-coverage-plan.md` §4・§7 の R13。`linalg_inv`・`solve`・`det`・`cholesky`・`qr`・`eigh` の `double` カーネルを HIPRTC へ移植した（`linalg_svd`・`matrix_norm` は CUDA も `Unsupported` のため対象外）。

- **実装範囲**: 単体で動く実行器（`RocmLinalgLu`・`RocmLinalgFactor`。`crates/backend-rocm/src/linalg.rs`）・ソース生成（`kernels_linalg.rs`・`kernels_linalg_factor.rs`）・ゲート（`backend_rocm::linalg::{set_gpu_linalg_enabled, gpu_linalg_enabled}`。CUDA とは別の `static`・既定 OFF・`SeqCst`）・ホスト側モデル（`linalg_model.rs`）まで。`BackendOps` への結線（`RocmBackendOps` の 6 メソッド・`Unsupported` の返却・ゲート → 形状 → サイズ → `n == 0` → デバイスの検査順序）は **R0（#3431）が未マージのため未実装**で、R0 のマージ後に同じ検査順序で足す（`tests/linalg_absent.rs` にも追記）。facade は `backend-rocm` を参照しないため保留ガードは不要。
- **FMA（CUDA 版との意図的な差分）**: CUDA 版は `--fmad` 既定 ON のため `__dmul_rn` 等で縮約を避けたが、HIP-Clang の contraction 既定は異なり、`__d*_rn` が HIPRTC の暗黙ヘッダ内の通常演算子であればソース先頭の pragma はその内部に効かず縮約を保証できない。そこで各ソースの先頭に `#pragma clang fp contract(off)` を置き、素の `*`・`+`・`-`・`/` と正しく丸める `sqrt`（double）で書いた（Metal の linalg と同じ先例）。`fma(`・`__builtin_fma*`・`#include` は使わない（ソース内容テストで固定）。**issue 本文の「`__builtin_fmaf` で明示」は matmul 系（f32 SIMT）の FMA 契約向けのテンプレート文で、線形代数は §4 の別の軸（全経路 `double`・CPU 参照が `mul_add` を使わない）に従う。FMA 契約は変更していない。** HIP の double 組み込み関数の実装形・縮約の保証を一次情報で確認できていないため、`__d*_rn` は採用していない（確認できた場合に限り併用可とする判断規則）。実命令（`v_fma_f64` の不在）・`fabs`/`copysign`/`isfinite`/`atomicExch(int*, int)` が暗黙ヘッダで使えること・pragma がファイルスコープで効くことは **未確認**で、#3159 段 B の補助診断（オフラインの `clang -x hip -S`）へ申し送る。
- **f64 アキュムレータ契約**: 全経路 `double` で縮約は thread 0 が添字昇順に逐次加算する（CUDA と同じ）ため、正規化統計・勾配の長軸縮約の契約は直接は当たらない。
- **wave64**: shfl・warp 幅を使わず共有メモリの木構造縮約と `__syncthreads` のみ。block 次元 256（LU・分解系）・128（代入）は 2 のべき乗かつ 64 の倍数で、const assert で固定した。
- **`hipMalloc` は 0 初期化しない**: CUDA の `alloc_zeros` に頼っていた `perm`・`status`・`solve_err`・出力バッファを含む全バッファを起動前に upload する。`status`・`refl_valid` は `{0,1}` 以外の sentinel（-1）を書いて読み戻し、契約外（未起動を含む）は `LinalgKernelStatusInvalid` にして成功扱いにしない。
- **CUDA との差分（安全側・数値不変）**: cholesky・eigh の `total = n*n` を走るストライドループ変数を `long long` にした（QR が既にそうしている理由と同じで、`idx += BLOCK` が `INT_MAX` を超える符号付きオーバーフロー UB を避ける）。eigh のフロベニウスノルム縮約の添字も同様。
- **検証（実機なし）**: 5 カーネルを f64 で逐語に再現したホスト側モデルを、本番と同じホスト後処理へ通し、`CpuBackendOps::linalg_*` と統一複合判定（`assert_parity`）および bit 一致で照合した（n は block 境界の前後を含む 1・2・3・17・64・65・129・257、`1e±20` 倍・`1e20 × I₂`・特異・非正定値・eigh 未収束・perm の値域違反・QR の `m<n`／`m>n`／零列）。モデルは CPU と bit 一致した。tolerance・baseline・定数は変更していない。
- **実機（MI300X・gfx942）**: 未実測。`tests/linalg_real_device.rs`（`#[ignore]`）を用意し、`docs/perf/logs/rocm-parity-3159/README.md` に記入欄を追加した。実測は #3159 段 B。

## 11. #3140（Stage D）: facade 公開形の承認依頼と公開保留の記録

#3140 は GPU 線形代数の opt-in 関数を facade へ公開し、`api_surface.rs` の否定ガード（`facade_does_not_expose_gpu_linalg_optin*`）を正ガードへ反転する。
着手時点（2026-10-16・origin/main a20d036f）で公開の前提 2 条件がどちらも満たされていないため、**コード公開は行わず本節に公開形の承認依頼だけを記録した**（前例: #3115→#3116・#3095→#3096・#3118→#3119 の二段構成）。

### 11.1 ゲート状況

| ゲート | 状況 | 根拠 |
|---|---|---|
| G1: 公開形の所有者承認 | **取得済み（2026-10-17）** | §11.2 の推奨形 9 項目を所有者が承認した（#3140 コメント https://github.com/Fandhe-AI/fandhe-ai/issues/3140#issuecomment-6227907703）。既定 ON（Stage E）・tolerance／baseline・新しい `unsafe`・依存は承認範囲外。着手時点（a20d036f）では未取得で、D-C の承認対象は GPU 線形代数の `unsafe` のみだった |
| G2: #3138・#3139 の実機 parity | **未実測** | `docs/perf/logs/cuda-linalg-parity-3138/README.md`・`docs/perf/logs/metal-linalg-parity-3139/README.md` がともに未実測 |

「D-C の延長」と解釈して G1 を迂回しない（着手時点の判断。G1 はその後に個別承認で取得した）。

**所有者の判断（2026-10-17）**: G2 が未充足のため、#3140 は「公開形の記録（第 1 段）」として閉じる。公開の実装（第 2 段・§11.3）は #3018 ツリー外の #3405 に切り出し、#3138・#3139 の実機 parity 実測が pass してから着手する（#3133 コメント https://github.com/Fandhe-AI/fandhe-ai/issues/3133#issuecomment-6227908086）。

### 11.2 推奨する公開形（承認依頼）

1. CUDA: `pub fn set_cuda_gpu_linalg_enabled(enabled: bool)`／`pub fn cuda_gpu_linalg_enabled() -> bool`
2. Metal: `pub fn set_metal_gpu_linalg_enabled(enabled: bool)`／`pub fn metal_gpu_linalg_enabled() -> bool`。どちらも `#[cfg(target_os = "macos")]`（facade の `fandhe-ai-backend-metal` 依存が macOS 限定のため）
3. getter を付ける（先例 `cuda_graph_infer_enabled`・`metal_split_k_gemm_enabled` 等と同形）
4. 本体は backend のゲート関数への 1 行委譲のみ。`pub use` による再エクスポート・新しい演算名・モジュール公開は足さない
5. 既定 OFF・プロセスワイド（`Device` 単位ではない）・環境変数は足さない
6. 数値契約: ゲート OFF は変更前（ホスト f64）と bit 同一。CUDA は f64 内部計算で精度クラス不変。**Metal は f32 のため精度クラスが変わる**（判定は §5: 統一複合判定・厳密ゼロ fail・入力集合 κ ≤ 1e3）
7. 範囲外の Op（Metal の eigh・SVD・slogdet／pinv／lstsq 等）は従来どおり `Unsupported` でホスト f64 へフォールバック
8. CUDA 非搭載でも setter は呼べる（フラグのみ。driver に触れず panic しない）
9. `fandhe-ai =0.11.0` に対して追加のみ

### 11.3 承認後の作業（Phase 2。#3405 で追跡）

G1 の承認コメント URL を特定し、G2 の実測 pass を確認したうえで、(a) `lib.rs` に上記関数を追加、(b) `api_surface.rs` の否定ガード 2 件を「シグネチャ・委譲本体の固定＋allowlist 方式の違反スキャナ＋正のプローブ自己テスト＋到達性テスト」へ置換、(c) 常時実行の受入テスト `gpu_linalg_optin.rs` と実機 parity テスト（CUDA は `GateOn` を facade 経由へ、Metal は新規）、(d) backend 側 doc の「facade へ再公開しない」更新、(e) 本節への実装記録と `compat-api-scope.md` の適用記録を行う。既定 ON（Stage E）は別承認で本件の範囲外。

§9 の「facade 公開の実施」は #3405 で扱う（公開形は承認済み・実機 parity 実測待ち）。
