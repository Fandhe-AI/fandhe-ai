# GPU 線形代数カーネル起動の新規 `unsafe` の監査記録（イシュー #3134）

親 #3133（GPU 線形代数 Stage B〜D）の先頭の子 issue。後続の #3135（CUDA LU 系の起動）・#3136（CUDA cholesky・QR・eigh の起動）・#3137（Metal の起動）が本記録の不変条件を前提に `unsafe` を実装する。

## 1. 状態

| 項目 | 内容 |
|------|------|
| 承認状態 | 「GPU 線形代数の `unsafe`」を実装まで進めてよいとする D-C の承認が、親 #3133 本文に「D-C の承認済み」と記載されている（本記録はその記載に基づく）。承認コメントの URL は調査時点では確認できていない。**個々の `unsafe` ブロックの security-auditor 監査は承認とは別に必須**（`.claude/rules/security.md`） |
| 監査の種類 | **実装前の監査記録**。対象の `unsafe` ブロックは本 doc 時点（`origin/main` 266f5df5）で 1 つも存在しない（Stage A は `unsafe` なしで完了済み）。本 doc は予定ブロックの不変条件を確定する。カーネルソース・ホスト側検証は実コードを読んで突合した |
| 変更範囲 | docs のみ。コード・`Cargo.toml`／`Cargo.lock`・tolerance・baseline・ガードレール閾値・`docs/spec/` は変更しない |
| 未検証 | NVRTC コンパイルと GB10 実機実行（#3138）、MSL の Mac 実機実行（#3139）。いずれも実装前のため未実行 |
| 後続 PR の義務 | 各 `// SAFETY:` コメントに本 doc の行 ID（`CU-n`・`MT-n`）を引用する。PR ごとに security-auditor の再監査を受け、結果で本 doc を更新する |

## 2. CUDA 予定ブロック（5 件）

呼ぶ API はいずれも cudarc の `launch_builder(..).launch(cfg)`（`unsafe fn`）。

| ID | 予定の場所 | カーネル | 起動形状 | 検証方法（テスト） |
|----|-----------|----------|----------|--------------------|
| CU-1 | `CudaLinalg::run_lu_solve_f32`・`run_lu_det_f32` 内 | `linalg_lu_f64` | grid `(1,1,1)`・block は `LU_BLOCK_DIM`（256。`kernels_linalg.rs:36`）と**厳密一致** | `lu_det_matches_cpu_on_device`（`#[ignore]`・既存）。solve／inv の実機テストは未作成（#3135 で追加予定） |
| CU-2 | `run_lu_solve_f32` 内 | `linalg_lu_solve_f64` | grid `ceil(k / LU_SOLVE_BLOCK_DIM)`・block 128（`kernels_linalg.rs:151`）。`k == 0` では起動しない。CU-1 と同一ストリームで後に起動 | 同上（未作成・#3135 で追加予定） |
| CU-3 | `CudaLinalgFactor` の cholesky 実行関数内 | `linalg_cholesky_f64` | grid `(1,1,1)`・block は `FACTOR_BLOCK_DIM`（256。`kernels_linalg_factor.rs:38`）と厳密一致 | `cholesky_matches_cpu_on_device`（`#[ignore]`・既存） |
| CU-4 | 同 QR 実行関数内 | `linalg_qr_f64` | 同上 | `qr_matches_cpu_on_device`（`#[ignore]`・既存） |
| CU-5 | 同 eigh 実行関数内 | `linalg_eigh_f64` | 同上 | `eigh_matches_cpu_on_device`（`#[ignore]`・既存） |

共通の不変条件（全行。違反時の影響を併記）:

| 記号 | 不変条件 | 違反時の影響 |
|------|----------|--------------|
| (a) | ゲート OFF・形状・サイズ検証（`validate_lu_dims`〈`linalg.rs:77`〉・`validate_square_factor_dims`〈`:265`〉・`validate_qr_dims`〈`:282`〉）はデバイス初期化（`device_handle`）より前に置く | 不正入力でデバイスを触る・確保してしまう（DoS） |
| (b) | バッファ長は検証済みの次元から確保する。f32→f64 変換は H2D の前に行う | カーネルが確保範囲外を読み書きする（OOB） |
| (c) | `n`・`k`・`m` は `i32` へ切り詰めなしで変換する（`try_from` 失敗は型付きエラー） | 次元が切り詰められ、カーネルの添字が確保範囲を超える |
| (d) | `cached_linalg`／`cached_linalg_factor` の**構築**と launch・D2H を、どちらも `CudaBackendOps::with_driver_call`（`ops.rs:902`）の内側に置く | 構築を包まないと sticky エラーが観測されない fail-open の窓が残る |
| (e) | カーネル引数の順序と型をカーネルのシグネチャと一致させる | 型不一致による不定動作（引数ずれ） |
| (f) | `status`（QR は `refl_valid`）を必ず D2H で読み戻して判定し、エラーへ写像する（solve／inv のピボット 0 は `InvalidArgument`、det は `0.0`、cholesky の非正定値と eigh の非収束は `InvalidArgument`） | 未判定の不正な結果を成功として返す（A08） |
| (g) | 引数バッファは launch 完了（D2H 同期）まで生存させる | 解放済みデバイスメモリへのアクセス |
| (h) | CU-2 は CU-1 と同一ストリームで後に起動する（ストリーム順序が `perm` の整合を担う） | `perm` 未確定のまま solve が走る |

カーネルソースの観察（本 doc 作成時に実コードで確認した事実）:

- `linalg_lu_f64` は `s_val[LU_BLOCK]` の木構造縮約を前提とするため、blockDim は LU_BLOCK と一致が必須。`status[0]`／`[1]` はカーネル側（`tid == 0`）で 0 に初期化する
- `linalg_lu_solve_f64` は `col >= k` と `status[0] != 0` で return する。`b[perm[i]*k + col]` に `perm[i] < n` の検査は**ない**（Metal 版は `src < n` を検査）。`perm` を書くのは同一ストリームで先に走る LU カーネルであり、(h) が不変条件になる
- `k <= i32::MAX` のとき `ceil(k/128)*128 - 1` は INT_MAX 以内で、`col` の int は overflow しない
- cholesky・eigh の `int total = n*n` は `n <= 46340` で、`idx += 256` しても INT_MAX を超えない。QR はストライド変数を `long long` にしてある（ソースにコメントあり）
- `linalg_qr_f64` は `status` を持たず、`refl_valid` をホストが読み戻して後処理に使う
- ゲート ON で `n > 0` かつ `k == 0`（`solve` の `b: [n,0]`）は `lu_gate_prepare` が短絡しない。Stage B ではこの入力で **grid 0 の solve 起動を作らず**、LU だけ起動して特異性を判定する

## 3. Metal

### 3.1 案の比較

| 観点 | 案 A: 既存ヘルパーを共有モジュールへ移設 | 案 B: `linalg_kernel.rs` に `scan.rs` 型の `unsafe` を新設 |
|------|------|------|
| 正味の新規 `unsafe` ブロック | 0（既存の移動） | 新設（`scan.rs:203` の setBuffer・`:212` の setBytes と同型） |
| 触る既存 `unsafe` | `rnn_cell.rs` の `set_u32_arg`（`:120`）・`set_buffer_arg`（`:136`）と `sort.rs` の `set_scalar_args`（`:332`） | なし |
| 変更が及ぶ範囲 | `rnn_cell.rs` 内の参照 39 か所・`sort.rs` 内 4 か所。純粋な移設で挙動は変えない | linalg のみ |
| SAFETY 文言の一般化 | 必要。現行の `set_buffer_arg` は「呼び出し元 `ctx.dispatch_sync` の完了まで生存」、`set_scalar_args` は `shaders/sort.metal` を名指ししており、呼び出し方式・特定 MSL への依存を外す | 不要（linalg 専用の文言） |
| 移設を守るテスト | `tests/rnn_cell_source_evidence.rs`（Linux）・`tests/rnn_cell_parity.rs::rnn_cell_matches_cpu_across_shapes`（実機 `#[ignore]`）・`tests/sort_topk_parity.rs`・`tests/sort_topk_source_evidence.rs` | 該当なし |
| レビュー負荷 | 既存 2 ファイルの差分が増える | `unsafe` の重複がもう 1 組増える |

**採用案は A**（Issue・`docs/perf/logs/metal-linalg-lu-chol-qr-2676/README.md` の推奨と一致）。理由は、`unsafe` の総数が増えない・契約を 1 か所に集約できる・`scan.rs` 型の重複をこれ以上増やさないこと。コストは SAFETY 文言の一般化と既存 2 ファイルの差分で、挙動不変であることをテストで担保する。不採用: `elementwise::encode_nary_dispatch`（`elementwise.rs:757`）は引数レイアウトと起動形状が固定で、n・k の 2 スカラーを渡せない。

### 3.2 移設対象と一般化後の契約

新しい `pub(crate)` モジュール `crates/backend-metal/src/encoder_args.rs`（#3137 で確定）へ移した。

| ID | 対象 | API | 一般化後の契約 |
|----|------|-----|----------------|
| MT-1 | `set_buffer_arg`（setBuffer） | `setBuffer_offset_atIndex` | 呼び出し元が、コマンドバッファの完了までバッファの生存を保証する（`dispatch_sync`、または `encode` の `resources` で retain）。`index` は対象 MSL の `[[buffer(i)]]` と一致させる |
| MT-2 | `set_u32_arg`・`set_scalar_args`（setBytes） | `setBytes_length_atIndex` | 渡すバイト数は対象 MSL の `constant uint&` 宣言（4 バイト）と一致させる責務を呼び出し側が負う。値は `u32` へ切り詰めなしで変換済み |

### 3.3 linalg からの利用で守る不変条件

- ゲート・サイズ検証（`crates/backend-metal/src/linalg.rs:120`〈`validate_lu_dims`〉・`:133`・`:142`。`u32::MAX` と `isize::MAX`）はデバイス初期化より前に置く
- `status` バッファ（LU の `status[0]`・`[1]`、cholesky の `status[0]`、QR の `applied`）は**ホストが 0 で初期化する契約**（`shaders/linalg.metal` のカーネルコメント）。CUDA はカーネル側で 0 初期化するため非対称であり、Metal にだけ「確保時に 0 を書き込む」不変条件が付く。怠ると未初期化値で成否を誤判定する
- LU・cholesky・QR は threadgroup を **1 個だけ** dispatch し、幅は `maxTotalThreadsPerThreadgroup` で clamp する。カーネルは `threads_per_threadgroup` を使い固定長配列を持たないので幅非依存
- solve は `ceil(k / width)` 個の threadgroup で dispatch し、`k == 0` では dispatch しない
- `[[buffer(i)]]` の番号は `tests/linalg_source_evidence.rs::kernels_and_buffer_indices_are_pinned` で固定されたものと一致させる
- スカラーは `u32` へ切り詰めなしで変換する
- バッファの生存は dispatch 方式（`dispatch_sync` か `encode`＋`resources` か）を #3137 で確定し、その方式に対応する MT-1 の契約を満たす
- `status` は必ず読み戻して判定する
- 実機では `status` の判定に失敗した結果を返さない（`MetalLinalg` は `linalg_kernel.rs:27`。起動保留は `ops.rs:521 linalg_launch_pending`）

検証: Linux で走るのは `tests/linalg_model_parity.rs`（ホスト逐語モデル。`singular_matrix_is_flagged_and_det_is_zero`・`model_rejects_bad_dims_without_panic` など）と `tests/linalg_source_evidence.rs`（`bounds_checks_barriers_and_norm_are_present`・`threadgroup_width_constant_is_pinned`）。実機は `tests/linalg_parity.rs` の `#[ignore]` テスト（`pipeline_compiles_on_device`・`cholesky_matches_cpu_on_device`・`lu_family_matches_cpu_on_device`・`qr_matches_cpu_on_device`）。

## 4. REQ-8 とゲート順序

- カーネル側の手動境界チェックは**維持する**。ホスト検証だけでは、将来の呼び出し経路の追加やカーネル流用に対する多層防御にならない。`.claude/rules/coding-rust.md` は最適化を理由とした省略を禁じる。設計記録 `docs/backend-gpu-linalg-decision.md` §4 も参照
- カーネル側の検査: ストライドループの `i < n`（`idx < total` 等）、CUDA solve の `col >= k`、Metal solve の `gid >= k` と `src < n`
- 添字型: CUDA は int（`n <= 46340`・`k <= i32::MAX`・QR は `long long` ストライド）、Metal は uint（`m*n <= u32::MAX`）。Metal QR の uint ストライド（§6 の 3）は未確認
- ゲート判定はデバイス初期化より前に置き、確保前にバイト数上限と shape を検査する（`backend-gpu-linalg-decision.md` §4）。順序は「検証 → デバイス初期化 → 確保 → 起動」で固定する

## 5. 早期 return とバリアの一様性

実カーネルを読んで確認できた範囲のみを記す。

| カーネル | 確認結果 |
|----------|----------|
| Metal LU・cholesky・QR | `threadgroup_position_in_grid != 0` はどのバリアよりも前に threadgroup 単位で一様に return する。LU のピボット 0 での return は、バリア後に全スレッドが同じ `a[k*n+k]` を読んだ値で判定する（読む直前に書き込みはない）。cholesky の `fail_tg` はバリア後に全スレッドが読む。QR の `continue`（`norm_x == 0`・`norm_v == 0`）は threadgroup 変数をバリア後に読んだ値で判定し、全スレッドが一様に次の反復へ進む |
| CUDA LU | grid が 1 block なので block 間同期は不要。早期 return（ピボット 0）は `__syncthreads()` の後、全スレッドが同じ `a[k*n+k]` を読む値で判定する |
| CUDA cholesky・eigh | `s_fail`・`s_conv` は `__syncthreads()` の後に全スレッドが共有変数から読んで判定するので一様 |
| CUDA QR | 早期 return はない。`s_valid` の分岐は同期後の共有値で一様 |

eigh の回転更新（`s_skip == 0` 内）の `sym` 書き込みと `v` 書き込みの間にバリアはないが、配列が別でスレッドごとの添字が分離している。この点は実装 PR（#3136）で security-auditor が再確認する（本記録では「競合なし」と断定しない）。

#3136 実装時の実装担当による自己点検（security-auditor の最終判定ではない）: eigh の回転更新で、スレッド k（k≠p,q）が書くのは `sym[k*n+p|q]`・`sym[p|q*n+k]` と `v[k*n+p|q]` だけで、`k` ごとに添字が分離し、`sym` と `v` は別配列。tid 0 の対角書き込み（`sym[p*n+p]` 等）は直前の `__syncthreads()` の後で、同時に走る他スレッドの `v` ループとは別配列のため競合しない。回転後の `__syncthreads()` で次の (p,q) へ進む。よって競合は見つからなかった。

## 6. 申し送り（残リスク）

| # | 内容 | 担当 |
|---|------|------|
| 1 | CUDA solve の `perm[i]` に範囲検査がない（Metal は `src < n`）。REQ-8 の多層防御として `perm[i] < n` 検査の追加を検討する。カーネルソースの変更は本 PR では行わない | #3135 |
| 2 | `n > 0` かつ `k == 0` では solve の grid 0 起動を作らない | #3135・#3137（Metal は #3137 で解消: `k == 0` は LU のみ encode し solve を encode しない） |
| 3 | Metal QR は `m*n <= u32::MAX` により `m` が u32::MAX 近傍（n=1）になりうる。`for (uint i = tid; i < len; i += tg_size)` が最終要素付近でラップして終了しない（ハング）可能性を #3137 で確認する。MSL の符号なしラップは UB ではないが DoS になりうる | #3137（解消: `linalg::validate_launch_stride_headroom` で `bound + 256 <= 2^32` をホスト検証。QR は `max(m, n)`、LU・cholesky は `n*n`〈n<=65535 で 65535²+256 < 2^32 のため実質影響なし〉、solve は `max(n*n, k)`。MSL は変更していない） |
| 4 | NVRTC コンパイル・GB10 実機・Mac 実機での一致は未検証 | #3138・#3139 |
| 5 | Metal のバッファ生存契約は dispatch 方式に依存する | #3137（解消: `ctx.dispatch_sync` に固定。バッファは関数スコープで生存し、`synchronize` 後に読み戻す） |
| 6 | D-C 承認コメントの URL を確認し、本 doc §1 へ追記する | 本 issue のレビュー時 |

## 7. OWASP 対応

| 項目 | 対応 |
|------|------|
| A03 インジェクション・入力検証 | NVRTC・MSL のソースは定数文字列のみで、利用者入力を連結しない。次元は FFI の前にホストで検証する |
| A04 安全でない設計 | 要素数・バイト数・添字型・grid の上限（`i32`／`u32`／`isize::MAX`）、grid 0 の禁止、ゲート OFF・未実装時だけの fail-closed `Unsupported`、エラーの型付き写像 |
| A06 脆弱・古いコンポーネント | 依存の追加・更新なし |
| A08 データ整合性 | `status`／`refl_valid` を必ず読み戻す。`with_driver_call` による sticky エラー観測に参加する。`unsafe` の追加は security-auditor の監査を経る |
| 秘密情報 | 該当なし（ワークフロー・hooks・設定は変更しない） |

## 8. security-auditor の監査結果

**未実施**。本タスクの実行環境ではサブエージェントを起動できなかったため、security-auditor による監査は行っていない。「指摘なし」を意味しない。

実装担当による自己点検の結果は §5・§6 のとおり。PR レビューで security-auditor の監査を受け、指摘と対応をこの節へ追記する。個々の実装 PR（#3135・#3136・#3137）でも各ブロックを再監査する。実装後の再現コマンド:

```
grep -n "unsafe" crates/backend-cuda/src/linalg.rs crates/backend-metal/src/linalg_kernel.rs crates/backend-metal/src/encoder_args.rs
```

（`encoder_args.rs` は #3137 で確定するモジュール名の例。移設元の `rnn_cell.rs`・`sort.rs` も併せて確認する。）

### 8.1 Metal（#3137）

**security-auditor によるサブエージェント監査は未実施**（実行環境でサブエージェントを起動できなかった）。「指摘なし」を意味しない。以下は実装担当の自己点検で、PR レビューで security-auditor の監査を受け、指摘と対応を本節へ追記する。

- `unsafe` の差分（案 A）: 移設元の 3 ブロック（`rnn_cell.rs` の `set_u32_arg`・`set_buffer_arg`、`sort.rs` の `set_scalar_args`）を削除し、`encoder_args.rs` に 2 ブロック（MT-1: `setBuffer_offset_atIndex`、MT-2: `setBytes_length_atIndex`）を置いた。`set_u32_arg` は `set_u32_args` へ委譲するため、正味の新規 `unsafe` は 0（総数は 1 減る）。`linalg_kernel.rs` は `unsafe` を含まない。`sort.rs` の `encode_*` 内に inline で残る `setBuffer` の `unsafe` は本記録の列挙（MT-1・MT-2）の外のため触っていない。
- 生存契約: すべて `ctx.dispatch_sync` 内で呼び、バッファは関数スコープで生存する。
- `status`／`applied` は `new_zeroed_i32` で 0 初期化して起動し、`status` は必ず読み戻して判定する。特異時は det を即 `0.0` とし、未処理の対角を積に通さない。未知の非 0 status は特異（失敗側）として扱う。
- 因数分解 3 カーネルは単一 threadgroup `(1,1,1)`、幅は `maxTotalThreadsPerThreadgroup` で clamp する。solve は `ceil(k / width)` threadgroups で、`k == 0` では encode しない。
- スカラーは `u32::try_from` で切り詰めずに変換する。`run_*` の入口で長さと次元を再検証する（二重検査）。
- REQ-8: MSL のコード部分は変更しておらず、カーネル側の手動境界チェックは維持している。
- 実機（M4 Max）での数値一致・移設の非後退（`rnn_cell_parity`・`sort_topk_parity` の `--ignored`）は未検証（#3139）。

### #3136（CU-3・CU-4・CU-5 の実装）

**security-auditor による監査は未実施**（本タスクの実行環境ではサブエージェントを起動できなかった。「指摘なし」を意味しない）。PR レビューで監査を受け、指摘と対応を本節へ追記する。

実装担当による自己点検の結果（`crates/backend-cuda/src/linalg.rs` の `run_cholesky_f32`・`run_qr_f32`・`run_eigh_f32`）:

| 項目 | 結果 |
|------|------|
| `unsafe` ブロック数 | 3 つのみ（`launch` の各 1 文）。各 `// SAFETY:` に行 ID（CU-3／CU-4／CU-5）と不変条件 (a)〜(g)・block 次元・REQ-8・int 添字上限を記載 |
| (a) 順序 | `ops.rs` は「ゲート → 形状 → サイズ検証 → 空入力の短絡 → デバイス」。`run_*` も入口で検証し直す |
| (b)(c) | f32→f64 は H2D の前。長さは検証済み次元から確保。`i32::try_from` で切り詰めなし |
| (d) | ハンドル構築（`ops.rs::linalg_factor_handle`）と実行（`context_cache::with_driver_call`・`ops.rs::with_driver_call`）はいずれも包まれている |
| (e) | 引数順・型はカーネルのシグネチャと一致 |
| (f) | `status`（QR は `refl_valid`）は sentinel（-1）で初期化し必ず読み戻す。0 は成功、1 は `InvalidArgument`、それ以外は `KernelLaunchFailed` |
| (g) | バッファは readback（同期点）の完了まで関数スコープで生存 |
| 残余リスク | `validate_square_factor_dims` は n ≤ 46340 まで受け付けるが、eigh は単一 block・O(60·n³) のため大きな n で GPU を長時間占有しうる（DoS）。ゲートは内部 opt-in で facade 非公開。上限の新設は本イシューの範囲外（別 issue 候補） |
| 未検証 | NVRTC コンパイルと GB10 実機の parity は #3138 |
