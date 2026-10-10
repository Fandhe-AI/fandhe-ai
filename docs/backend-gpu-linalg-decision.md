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
- **エラー写像**: カーネル検出の「ピボット厳密 0」「非正定値の対角」「Jacobi 非収束」はデバイス側フラグを読み戻して `BackendError::InvalidArgument`。`Unsupported` は「未実装またはゲート OFF」に限る。非有限値は CPU と同じ（LU 素通し・Cholesky 拒否・eigh は入口で `require_finite` が拒否）。n = 0 はデバイス初期化前にホストで CPU と同じ結果を返す
- **REQ-8**: カーネル側の手動境界チェックを省略しない。確保前にバイト数上限検査と shape 検査（`require_square_2d` 相当）を行い、ゲート判定はデバイス初期化より前に置く

## 5. parity 判定方式

- 結論: **実測ベースライン方式は現時点で不要**
- 理由: §3 の方式は各出力要素の算術順序を CPU 参照実装の逐次ループと同一に保つため、`.claude/rules/coding-rust.md` の「単一の連続 K ループと異なる結合順序で部分和を求めるカーネル」に当たらない。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満、`assert_parity`）を厳密ゼロ fail でそのまま適用する
- Metal f32 と CPU f64 の差は結合順序ではなく精度クラスの差であり、baseline 方式の適用対象外。baseline で吸収しない。代わりに parity 入力集合を固定する: 対角優位行列・既知の κ を持つ SPD 行列・n ≤ 64 程度、条件数 κ ≤ 1e3 を上限とし、悪条件入力は受け入れ集合に入れない
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

コード変更なし。実機実測は子 issue の担当。
