# backend-rocm の `unsafe` の監査記録（イシュー #3158）

親: #3150。前提: `docs/backend-rocm-hip-ffi-decision.md`（§4 実装形の契約・§13 #3154 時点の監査）、`docs/backend-rocm-hiprtc-gemm-decision.md`（§6 簡易一覧）、`docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md`（U1〜U7 の定義）。

## 1. 状態

| 項目 | 内容 |
|------|------|
| 対象 | `crates/backend-rocm`。基準コミットは `origin/main` の 2e38c37c（#3155・PR #3422 を含む） |
| 監査の種類 | 実装担当による**自己点検**（機械的インベントリ・全ブロックの手読み・clippy の臨時実行・Linux 上の検証）。**独立した security-auditor の監査ではない**。実装環境にサブエージェントの起動手段がなかったため未実施で、PR レビューで security-auditor の監査を受け、その結果を §8 へ追記する。所有者の承認を得たとは記載しない |
| 変更範囲 | 本 doc の新設、`hiprtc.rs` の `HiprtcRuntime::_lib` の `ManuallyDrop` 化（F1）、SAFETY コメントの表記揃え。依存（`libloading =0.9.0`）・tolerance・FMA 契約・REQ-8 の境界チェック・`docs/spec/`・CI は変更しない |
| 未検証 | §6 の実機項目すべて（MI300X・gfx942。#3159）。実機テスト（`#[ignore]`）は実行していない |
| #3444 追記 | `linalg.rs` の起動 5 件（L1〜L5。§3.5）を追加し、`DeviceBuffer` を型付き（`DeviceBuffer<T>`。`f32`・`f64`・`i32`）にして M5〜M7 の前提を `elems*size_of::<T>()` へ改めた。ブロック総数は 33 → 38。独立した security-auditor の監査は PR レビューで実施し §8 へ追記する（未実施。得ていない承認は書かない） |
| 追従（R1・#3432） | `WarpGeometry` は `unsafe` を追加しない（監査表の件数は不変） |
| 追従 | #3157（facade の Device に ROCm を追加）が `backend-rocm` へ `unsafe` を足す場合は、#3157 側で本 doc の §3 を追記する。R0（#3431）は新しい `unsafe` を足さない（インベントリは 33／10／17／0 のまま。`unsafe impl Send/Sync` もなし） |

## 2. 機械的インベントリ

再現コマンド:

```
grep -n 'unsafe' crates/backend-rocm/src/*.rs crates/backend-rocm/tests/*.rs
cargo clippy -p fandhe-ai-backend-rocm --all-targets -- -D warnings -W clippy::undocumented_unsafe_blocks
```

| 種別 | hip.rs | hiprtc.rs | module.rs | gemm.rs | linalg.rs | 計 |
|------|--------|-----------|-----------|---------|-----------|----|
| `unsafe { }` ブロック | 15 | 10 | 7 | 1 | 5 | **38** |
| `unsafe fn`（`# Safety` 節つき） | 8 | 1 | 1 | 0 | 0 | **10** |
| `unsafe extern "C" fn` 型エイリアス（宣言のみ） | 12 | 5 | 0 | 0 | 0 | **17** |
| `unsafe impl`（`Send`／`Sync`） | 0 | 0 | 0 | 0 | 0 | **0** |

- `tests/` 配下に `unsafe` はない。`lib.rs` の `unsafe` はモジュール doc 内の言及で件数に含めない。
- edition 2024 の `unsafe_op_in_unsafe_fn` により、`unsafe fn` 本体の呼び出しも独立したブロックで書いており、33 に含む。
- `unsafe impl Send/Sync` が 0 件のため、生ポインタを持つ `Module`・`Function`・`DeviceBuffer<T>`・`Program` と、それを持つ `RocmGemmF32`・`RocmLinalgLu`・`RocmLinalgFactor` は自動的に `!Send`・`!Sync`（安全側の既定）。`HipRuntime`・`HiprtcRuntime` は `Library` と関数ポインタのみで自動的に `Send + Sync` となり `Arc` で共有される。HIP ランタイム API 自体のスレッド安全性は §6 の実機項目。
- `clippy::undocumented_unsafe_blocks` の臨時実行は、修正前は 5 件（`// SAFETY (Ux):` の表記が判定を満たさなかった）。`// SAFETY: (Ux)` へ揃えた後は **0 件**。`clippy::multiple_unsafe_ops_per_block` は 2 件（§3 の H3・R3。関数ポインタ解決を並べた構造体リテラル）で、構造を変えず受容する（理由は各行）。lint の恒久導入は §9 の申し送り。

## 3. 全件の表（ブロック 38）

凡例: U1 = ハンドル寿命内でのみ関数ポインタを呼ぶ／U2 = C 文字列・オプション配列の生存と個数／U3 = バッファサイズの一致／U4 = コードオブジェクトは module より長生き／U5 = `kernelParams` の個数・型・順序と生存／U7 = RAII で 1 回だけ解放。U6（cooperative 起動）は未実装で対象外。場所は関数名で示す（行番号はコミットで動くため付けない）。

### 3.1 hip.rs（15）

| # | 場所 | 操作 | 前提条件 | 破れた場合の影響 | 守る不変条件 |
|---|------|------|----------|------------------|--------------|
| H1 | `resolve` | `Library::get::<T>` | `T` が C 宣言と一致する関数ポインタ型。名前は固定の `&'static str` | 型不一致の呼び出しで UB（ABI 違反） | U1・A03 |
| H2 | `load_from` | `Library::new(name)` | `name` は固定名でパスを含まない | `LD_LIBRARY_PATH`／RUNPATH を握られると初期化子が走る（残留リスク。§7） | A03 |
| H3 | `load_from` の構造体リテラル | `resolve` を 12 回 | 各型は HIP の C 宣言と一致。ポインタは `_lib` と同じ構造に直接格納 | 型不一致で UB。構造外へ出ると解放後呼び出し | U1。`multiple_unsafe_ops_per_block` は同一契約の並びで、分割しても契約が変わらないため受容 |
| H4 | `device_count` | `hipGetDeviceCount` | `n` は有効な書き込み先 | 書き込み先不正は UB | U1 |
| H5 | `current_device` | `hipGetDevice` | `id` は有効な書き込み先 | 同上 | U1 |
| H6 | `set_device` | `hipSetDevice` | 値渡しのみ | 不正な ID は `hipError_t` で返る | U1 |
| H7 | `synchronize` | `hipDeviceSynchronize` | 引数なし | 失敗は型付きエラー | U1 |
| H8 | `malloc` | `hipMalloc` | `p` は有効な書き込み先。結果の null を拒否 | null を返して成功扱いで後続が null 参照 | U1・U7 |
| H9 | `free`（unsafe fn 内） | `hipFree` | `p` は `malloc` 由来で未解放（呼び出し元の契約） | 二重解放・不正解放 | U7 |
| H10 | `memcpy_htod` | `hipMemcpyHtoD` | `dst` は `bytes` 以上のデバイス確保済み、`src` は `bytes` 読める | OOB 読み書き | U3 |
| H11 | `memcpy_dtoh` | `hipMemcpyDtoH` | `dst` は `bytes` 書ける、`src` は `bytes` 以上確保済み | 同上 | U3 |
| H12 | `module_load_data` | `hipModuleLoadData` | `image` は有効なコードオブジェクト。`m` は有効な書き込み先。結果の null を拒否 | 不正なイメージの読み込み。null 成功の混入 | U4 |
| H13 | `module_get_function` | `hipModuleGetFunction` | `module` は未解放、`name` は NUL 終端で生存。結果の null を拒否 | 解放後 module の参照 | U2・U4 |
| H14 | `module_unload` | `hipModuleUnload` | `module` は未解放で実行中カーネルなし | 解放済み・実行中の module の破棄 | U7 |
| H15 | `launch` | `hipModuleLaunchKernel` | `func` は有効、`params` は宣言どおりの個数・型・順序で同期完了まで生存。stream・extra は null、共有メモリ 0 | 引数の型・個数不一致で OOB・ハング | U5 |

### 3.2 hiprtc.rs（10）

| # | 場所 | 操作 | 前提条件 | 破れた場合の影響 | 守る不変条件 |
|---|------|------|----------|------------------|--------------|
| R1 | `resolve` | `Library::get::<T>` | H1 と同じ | H1 と同じ | U1・A03 |
| R2 | `load_from` | `Library::new(name)` | H2 と同じ | H2 と同じ | A03 |
| R3 | `load_from` の構造体リテラル | `resolve` を 7 回 | H3 と同じ | H3 と同じ。`multiple_unsafe_ops_per_block` は受容 | U1 |
| R4 | `Program::drop` | `hiprtcDestroyProgram` | `handle` は `create` が返した未破棄のもの。`destroy` は `rt` 由来。失敗は無視 | 二重破棄・リーク | U7 |
| R5 | `compile_code_object` | `hiprtcCreateProgram` | `c_src`・`c_name` は NUL 終端で関数末尾まで生存。ヘッダ 0 件。`handle` は有効な書き込み先。null を拒否 | 解放済み文字列の読み取り | U2 |
| R6 | `compile_code_object` | `hiprtcCompileProgram` | `prog.handle` は有効。`opts` の個数 1 と numOptions=1 が一致 | OOB 読み取り | U2 |
| R7 | `compile_code_object` | `hiprtcGetCodeSize` | `size` は有効な書き込み先 | 書き込み先不正 | U3 |
| R8 | `compile_code_object` | `hiprtcGetCode` | `buf` は `GetCodeSize` のちょうどのサイズ。0 と 256 MiB 超は事前に拒否 | OOB 書き込み | U3 |
| R9 | `fetch_log` | `hiprtcGetProgramLogSize` | R7 と同じ | 同じ | U3 |
| R10 | `fetch_log` | `hiprtcGetProgramLog` | `buf` は `LogSize` のちょうどのサイズ（1 MiB 超は取得しない） | OOB 書き込み | U3 |

### 3.3 module.rs（7）・gemm.rs（1）

| # | 場所 | 操作 | 前提条件 | 破れた場合の影響 | 守る不変条件 |
|---|------|------|----------|------------------|--------------|
| M1 | `Module::load` | `module_load_data` | `code` は有効で `_code` として module と同寿命 | 解放後のイメージ参照 | U4 |
| M2 | `Module::function` | `module_get_function` | `self.handle` は未解放、`c` は NUL 終端で生存 | 解放後 module の参照 | U2・U4 |
| M3 | `Module::drop` | `module_unload` | `handle` は未解放。事前に作成デバイスを選択して同期 | 二重解放・実行中のアンロード | U7 |
| M4 | `Function::launch`（unsafe fn 内） | `HipRuntime::launch` | `self.ptr` は生存中の module の関数（`Function<'m>`）。`params` は呼び出し元の契約 | 解放後の関数ポインタ起動 | U5 |
| M5 | `DeviceBuffer<T>::upload` | `memcpy_htod` | `ptr` は `elems*size_of::<T>()` 以上確保済み、`src` 長は `elems` と一致を事前検証、`checked_mul` 済み。`T` は封印 trait `DeviceElem`（`f32`・`f64`・`i32` の POD のみ） | OOB | U3 |
| M6 | `DeviceBuffer<T>::download` | `memcpy_dtoh` | M5 と対称 | OOB | U3 |
| M7 | `DeviceBuffer<T>::drop` | `free` | `ptr` は `malloc` 由来で未解放。所有権で二重解放なし | 二重解放 | U7 |
| G1 | `RocmGemmF32::run` | `Function::launch` | `params` はカーネル署名と一致（`GEMM_F32_ARG_COUNT` とソース署名をテストで固定）。指す変数は `synchronize` まで生存。グリッドは出力全体を覆い、範囲外はカーネル内の手動境界チェック（REQ-8）が防ぐ | 引数不一致で OOB・ハング | U5 |

### 3.4 `unsafe fn`（10）と型エイリアス（17）

`unsafe fn` は全件が `# Safety` 節を持つ。`hip.rs`: `resolve`・`free`・`memcpy_htod`・`memcpy_dtoh`・`module_load_data`・`module_get_function`・`module_unload`・`launch`。`hiprtc.rs`: `resolve`。`module.rs`: `Function::launch`。契約は上表の「前提条件」と同一で、呼び出し元ブロック（H9〜H15・M1〜M7・G1・L1〜L5）が満たすことを確認した。`pub(crate)` で crate 外へは公開しない。

型エイリアス 17 件（`hip.rs` 12・`hiprtc.rs` 5）は宣言のみで呼び出しは上の各行。シグネチャは HIP／HIPRTC の C 宣言との一致を実機で確認する（§6）。

### 3.5 linalg.rs（5。#3444）

いずれも `Function::launch`（M4）の呼び出しで、1 block 起動（LU・cholesky・QR・eigh）または `k.div_ceil(128)` block 起動（代入）。ソースは `LinalgSource`（静的テンプレート＋`const u32` の `#define` のみ）で、pragma は `#pragma clang fp contract(off)`。バッファは**全て起動前に `DeviceBuffer::from_slice` で内容を書き込み**（`hipMalloc` は 0 初期化しない）、`status`・`refl_valid` は `{0,1}` 以外の sentinel（-1）を書いて読み戻し、契約外は `LinalgKernelStatusInvalid`（未起動を成功扱いにしない）。

| # | 場所 | 操作 | 前提条件 | 破れた場合の影響 | 守る不変条件 |
|---|------|------|----------|------------------|--------------|
| L1 | `RocmLinalgLu::launch_lu` | `Function::launch`（`linalg_lu_f64`） | 署名 `(double* a, int* perm, int* status, int n)` と `params` が一致（`LINALG_LU_ARG_COUNT` とソース署名をテストで固定）。`a`＝`n*n`・`perm`＝`n`・`status`＝2 要素で初期化済み（`validate_lu_dims`）。block＝`LU_BLOCK`＝256。変数は同期まで生存 | 引数不一致で OOB・ハング | U3・U5 |
| L2 | `RocmLinalgLu::run_lu_solve_f32` | `Function::launch`（`linalg_lu_solve_f64`） | 署名 8 引数と一致（`LINALG_SOLVE_ARG_COUNT`）。`b`・`x`＝`n*k`、`solve_err`＝1 要素で初期化済み。`perm` の値域はカーネルが `b` を読む前に検査（違反は `solve_err=1`・ホストが必ず読み戻して型付きエラー）。余剰スレッドは `col >= k` で何も書かない | `perm` 破損時の OOB（二重防御で遮断）・引数不一致 | U3・U5 |
| L3 | `RocmLinalgFactor::run_cholesky_f32` | `Function::launch`（`linalg_cholesky_f64`） | 署名 `(const double* a, double* l, int* status, int n)`（`LINALG_CHOLESKY_ARG_COUNT`）。`a`・`l`＝`n*n`、`status`＝1 要素。ストライド変数は `long long` | 引数不一致・符号付きオーバーフロー（UB） | U3・U5 |
| L4 | `RocmLinalgFactor::run_qr_f32` | `Function::launch`（`linalg_qr_f64`） | 署名 6 引数（`LINALG_QR_ARG_COUNT`）。`r`＝`m*n`、`q`＝`m*k`、`refl`＝`k*m`、`refl_valid`＝`k`（`validate_qr_dims`）。カーネルは `refl` の書き込み済み範囲だけを読む（ホスト側モデルで未書き込み要素を NaN にして確認） | 同上 | U3・U5 |
| L5 | `RocmLinalgFactor::run_eigh_f32` | `Function::launch`（`linalg_eigh_f64`） | 署名 6 引数（`LINALG_EIGH_ARG_COUNT`）。`a`・`sym`・`v`＝`n*n`、`evals`＝`n`、`status`＝1 要素。回転更新の書き込み先はスレッド間で重ならず、tid 0 の対角書き込みは `__syncthreads()` の後。早期 return は共有値で一様 | データ競合・同期の不一致（ハング） | U3・U5 |

`unsafe impl Send/Sync` は足さない（0 件のまま）。このため `RocmLinalg*` は `!Send`・`!Sync`で、`Tape`（`Box<dyn BackendOps + Send>`）へ載せる結線（R0 #3431 以降）は呼び出しごとに生成するか、`Send` 化を別承認にする。

### 3.5 R8（#3439）で追加した `unsafe`（4 ブロック。実装担当の自己点検。未監査）

gather/scatter・interpolate・sort・topk・unique・scan の 6 実行型は、起動を `launch.rs::KernelHost::run` の 1 か所へ、デバイスメモリ操作を `typed_buffer.rs::TypedDeviceBuffer<T>` の 3 か所へ集約した（`Function::launch` の `unsafe fn` 契約は不変で、呼び出し元が K1 に 1 つ増えただけ）。型付きバッファの要素型は封印 trait `DeviceElem`（`f32`・`i32`・`u32`・`u64` の POD のみ）。

| # | 場所 | 操作 | 前提条件 | 破れた場合の影響 | 守る不変条件 |
|---|------|------|----------|------------------|--------------|
| T1 | `TypedDeviceBuffer::upload` | `memcpy_htod` | `ptr` は `new` が `elems*SIZE` 以上確保済み（`checked_mul`）。`src.len() == elems` を事前検証。`T` は封印 POD | OOB | U3 |
| T2 | `TypedDeviceBuffer::download` | `memcpy_dtoh` | T1 と対称（`dst.len() == elems` を検証。呼び出し側は起動後に `synchronize` 済み） | OOB | U3 |
| T3 | `TypedDeviceBuffer::drop` | `free` | `ptr` は `malloc` 由来で未解放。所有権で二重解放なし | 二重解放 | U7 |
| K1 | `KernelHost::launch_dims`（`run`・`run_blocks` 共通。R6〈#3437〉で `run` から切り出した） | `Function::launch` | `params` の個数・種別を `KernelSpec`（ソース署名と同一であることを各 `kernels_*.rs` の `spec_matches_source_signature` テストで固定）と**実行時にも照合**してから構築。指す `owned` は直後の `synchronize` まで生存。グリッドは呼び出し元の `grid_1d`（`div_ceil` で覆う。`grid.x` と `grid.x*block` の `u32` 収容を checked で検証）か `wave_block_grid`（blocks は 1..=`i32::MAX`・block は wave 幅 32／64。R6〈#3437〉）が検証した値のみ。範囲外はカーネル内の手動境界チェック（REQ-8）が防ぐ。ポインタの指す要素型（`f32`／`i32`／`u64` 等）までは型で検査できないため、呼び出し側の各 `run_*` が引数順どおりにバッファを渡す（テストは非搭載環境のため起動を伴わない。実機確認は #3159） | 引数不一致・OOB・ハング | U5 |

- 起動は null stream・同期 API（`run` が `synchronize` まで待つ）。sort の `j`・`k` ループは各ステップを同期起動する（性能より単純さを優先。ステップごとの引数配列は都度構築する）。
- ホスト側の fail-closed 検証（A04／A08）: shape・スライス長・`i32`／`i64` 収容を `checked_*`／`try_from` で検査し、gather/scatter/one_hot は index の値域を全要素走査して範囲外を拒否する。interpolate は空間軸の入力サイズ 0（カーネルのクランプが `-1` 添字になる）を拒否する。
- 本節の件数は §2 の表（基準コミット時点の 33 ブロック）に含めていない。#3157 など兄弟 PR が §2 を更新している場合は、マージ順で再集計が必要。
- 実機（MI300X・gfx942）で HIPRTC のコンパイルと起動結果を確認していない（#3159）。ホスト側モデル（`*_model.rs`）は CPU 参照実装との bit 一致を検証するが、HIPRTC の出力そのものの検証ではない。

### 3.6 R6（#3437）: 新しい `unsafe` ブロックなし（実装担当の自己点検。未監査）

正規化 6 演算（`norm.rs::RocmNorm`・`kernels_norm.rs`・`norm_model.rs`〈`cfg(test)`〉）は **`unsafe` を一切追加しない**。起動は既存の K1 だけを通る。

- `launch.rs` の変更は、`KernelHost::run` の起動部分を private な `launch_dims` へ切り出し、1 block = 1 wave の `run_blocks`（検証は純関数 `wave_block_grid`）から同じ `launch_dims` を呼ぶ形にしたこと。`unsafe { func.launch(..) }` は引き続き 1 か所で、SAFETY コメントの「グリッド」の根拠だけを更新した（K1 行）。
- デバイスバッファは既存の `TypedDeviceBuffer<f32>`（T1〜T3）と `module::DeviceBuffer<f64>`（M5〜M7。`rstd`／`mean` の double スクラッチ）だけを使う。`TypedDeviceBuffer` の封印型に `f64` を足す案は T1〜T3 の前提を変えるため採らなかった。double スクラッチは起動前に `from_slice(&vec![0.0; rows])` で 0 初期化する（`hipMalloc` は 0 初期化しない。A04）。
- 省略された weight／bias／統計にも `hidden`（または `c`）要素のダミーバッファを渡す（predicated load 対策）。
- 確認方法: §2 の再現コマンドで `launch.rs` の `unsafe {` が 1 件のままであること、`norm*.rs`・`kernels_norm.rs`・`tests/norm_*.rs` に `unsafe` ブロックがないこと（doc 内の言及を除く）を確認した。§2 の件数表は R6 で増減しないが、兄弟 PR が §2 を更新している場合はマージ順で再集計が必要。
- 実機（MI300X・gfx942）で HIPRTC のコンパイル（`__shfl_xor` の double オーバーロード・`__builtin_fma` の f64 命令化）と wave64 parity を確認していない（#3159）。独立した security-auditor の監査は PR レビューで実施し §8 へ追記する（未実施。得ていない承認は書かない）。

## 4. U1 の担保方法

1. ハンドル（`Library`）と関数ポインタを同じ構造体（`HipRuntime`・`HiprtcRuntime`）に格納し、フィールドは private、getter はない。`resolve` の結果は構造体リテラルへ直接格納し、ローカルへコピーして保持しない。
2. 各ハンドル型（`Module`・`DeviceBuffer`）は `Arc<HipRuntime>` を保持するため、ライブラリがハンドルより先に解放されない。`Module` のフィールド順は handle → code → rt。
3. `HipRuntime` は `ManuallyDrop<Library>` で dlclose しない。`HiprtcRuntime` も F1 の修正で同じにした。このためポインタは `Drop` 後も無効にならない。
4. F2（呼び出し経路）: `hip.rs` は全呼び出しが `HipRuntime` のメソッド経由。`hiprtc.rs` は `compile_code_object`・`fetch_log`・`Program::drop` が private フィールド（`rt.create` 等）を直接呼ぶ。FFI 決定記録 §4-2 の「構造のメソッド経由」を、**モジュール境界（フィールドが `hiprtc.rs` の外から見えない）で満たしている**。型による担保ではなく、同一モジュール内の 3 関数を手読みで確認した。
5. F3: `Function::launch` は module の `Arc<HipRuntime>` と別に `rt` 引数を受け、同一であることを型が保証しない。呼び出しは `gemm.rs` の 1 箇所で `self.rt`（module と同じ `Arc`）を渡しており実害はない。強化案（`Function` が module の rt を参照して使う）は §9。

## 5. エラー経路での資源解放

| 経路 | 失敗点 | 解放の挙動 |
|------|--------|------------|
| `compile_code_object` | `create` 失敗 | ハンドルなし。解放不要 |
| 同上 | `create` の後（compile 失敗・ログ取得失敗・`GetCodeSize` 失敗・サイズ 0・上限超過・`GetCode` 失敗） | `Program` の RAII が 1 回だけ destroy（U7） |
| `Module::load` | `module_load_data` 失敗 | module なし。`code` は drop される |
| `run` | upload・launch・synchronize・download | `dc`・`db`・`da` の `hipFree` → `func` → `_guard` の順で drop。hipFree は作成デバイスを選択した状態で行い、その後に呼び出し元の選択を復元 |
| `RocmLinalg*::run_*`（#3444） | `from_slice`（確保＋upload）の途中失敗・launch・synchronize・download・status 解釈の失敗 | 確保済みの `DeviceBuffer<T>` が宣言の逆順で `hipFree`（全て `_guard` の後に宣言）。sentinel を upload してから読み戻す経路でも解放は同じ。status が契約外なら成功扱いにせず型付きエラー |
| `RocmGemmF32::new` | `select_device`・compile・load | `rt`（HIP）は意図的にリーク。HIPRTC も F1 修正後はリーク。`select_device` 失敗はコンパイル前に返る |
| `DeviceBuffer::new` | `malloc` 失敗 | ポインタなし。解放不要 |
| Drop 内の失敗 | `hipFree`・`hipModuleUnload`・destroy | panic せず無視（best effort） |

補足の確認事項:

- **F1（HIPRTC の dlclose）**: 修正前の `HiprtcRuntime` は素の `Library` で、`RocmGemmF32::new` のローカル変数として毎回 dlclose していた。hip.rs が dlclose しない理由（atexit フック・ワーカースレッドの残存）は comgr にも当てはまりうるため、`ManuallyDrop` 化して揃えた。同一ライブラリの再 dlopen は参照カウントが増えるだけで実質のリークは一定。挙動への影響は実機でのみ確認できる（§6）。
- **F4**: `hip::load_from` の途中の `resolve(..)?` が失敗すると、`lib` はまだ `ManuallyDrop` へ移っていないため dlclose される。この端の経路は「dlclose しない」方針と食い違うが、関数ポインタを保持した構造が作られていないので U1 には影響しない。受容する。
- **F5**: `DeviceBuffer::drop` は `hipFree` の前にデバイスを選択しない（`Module::drop` は選択する）。現状は全バッファが `run` のローカルで `_guard` の後に宣言されており、`_guard` より先に drop されるため作成デバイスが選択済みとなる。この宣言順が不変条件である。`hipFree` がデバイスに依らず動くかは実機で確認する。
- **F7**: `// SAFETY: (Ux)` の表記は §2 のとおり揃えた。

## 6. 実機なしで確認した範囲と、実機が要る確認

**実機なし（Linux・ROCm なしの CI 相当環境）で確認した**

- インベントリと全 33 ブロックの手読み、clippy の臨時実行（§2）
- private フィールドと getter がないこと（U1）。固定名の `&'static str` でパス区切りを含まないこと（`candidates_have_no_path_separators`）
- Drop 順（`Module` は handle → code → rt）と RAII によるエラー経路の解放
- 形状と grid の検証（`validate_gemm_dims`・`launch_geometry` のテスト）
- `kernelParams` の個数・順序とカーネル署名の一致（`GEMM_F32_ARG_COUNT` のテスト）
- REQ-8 の境界チェック記述がカーネルソースにあること（`kernels_gemm.rs`・`kernels_linalg*.rs` のテスト）
- 線形代数 5 カーネルの添字・同期区切り・status／補助フラグの書き込みと CPU 参照との bit 一致（`linalg_model.rs`。HIPRTC のコンパイル結果そのものの検証ではない）
- ROCm 非搭載で probe が false を返し、型付きエラーになること（`probe_absent`・`hiprtc_absent`）

**実機（MI300X・gfx942）が要る。#3159 へ申し送る**

- `hipError_t`／`hiprtcResult` を `c_int` で受ける ABI、`hipModuleLaunchKernel`・`hipMemcpyHtoD/DtoH` の実シグネチャ（型エイリアス 17 件）
- libhiprtc の `ManuallyDrop` 化（F1）が実機で問題を起こさないこと
- `hipFree` がデバイスに依らず動くこと（F5）、kernel fault 後（sticky error）の解放挙動
- コードオブジェクトの `Vec<u8>` のアラインメント要件、HIP の引数コピーの時機、grid の実際の上限
- 線形代数（#3444）: HIPRTC の暗黙ヘッダで `fabs`・`copysign`・`isfinite`・`sqrt`（double）・`atomicExch(int*, int)` が使えること、`#pragma clang fp contract(off)` がファイルスコープで効き `v_fma_f64` が出ないこと（オフラインの `clang -x hip -S` で命令数を数える補助診断）、`hipMalloc` が未初期化のため全バッファを upload している前提、wave64 での block 256／128 の正しさ
- HIP ランタイムのスレッド安全性、contraction の挙動（FMA。`docs/backend-rocm-hiprtc-gemm-decision.md` §2 の申し送りの再掲）

parity の許容誤差は `tests/gemm_real_device.rs` と REQ-2 を参照し、本 doc には書き写さない。

## 7. OWASP Top 10

- **A03**: dlopen 名・シンボル名は固定の `&'static str` のみ。HIPRTC のソースは `GemmSource`（静的テンプレートと検証済みの数値）だけを受け、オプションは `GfxTarget` の allowlist 由来の `--gpu-architecture=` 1 本。include パスは渡さない。NUL を含む文字列は `InvalidCString` の型付きエラー。**残留リスク**: 名前だけを渡す設計のため、`LD_LIBRARY_PATH`／RUNPATH を握られるとライブラリの初期化子が走る（FFI 決定記録 §4-3 で受容済み・§13 の P2 を本 doc に明記）。probe ごとの dlopen の繰り返しは、HIP 側は dlclose しないため参照カウントが増えるだけ。HIPRTC 側は F1 で同様にした。
- **A04（メモリ安全性）**: OOB は形状をホストで fail-closed に検証しカーネル側にも REQ-8 の境界チェック。解放後使用は `Arc` による寿命の延長と `Function<'m>`。二重解放は所有権と RAII。HIPRTC のログ 1 MiB・コードオブジェクト 256 MiB の上限で異常サイズによるメモリ枯渇を防ぐ。
- **A05／A09**: エラーは型付き。本番経路に panic・`unwrap`・`expect` はない。ログへ秘密情報は出さない。
- **A06**: `libloading =0.9.0` の完全固定を維持し、本変更で `Cargo.toml`／`Cargo.lock`／`deny.toml` は変更していない。
- **A08**: ディスクキャッシュを持たない。コードオブジェクトはメモリ上のみ。CI・承認ゲートは変更しない。

## 8. security-auditor の監査結果と対応

- 現時点: **security-auditor の独立監査は未実施**（§1）。PR レビューで実施し、結果をここへ追記する。
- 自己点検の指摘と対応:

| 区分 | 内容 | 対応 |
|------|------|------|
| P2 | `// SAFETY (Ux):` の表記が `clippy::undocumented_unsafe_blocks` を満たさない（5 件） | `// SAFETY: (Ux)` へ揃えた（文言は変更せず） |
| P2 | F1: `HiprtcRuntime` だけ dlclose する | `ManuallyDrop` 化（挙動は実機で確認。§6） |
| P2 | F2〜F5 | 受容として §4・§5 に記録。強化案は §9 |

- 自己点検では P0／P1 は見つからなかった。これは独立監査の代替ではない。

## 9. 残リスク・申し送り（起票はユーザー承認後）

- `clippy::undocumented_unsafe_blocks` の crate レベルでの恒久導入
- HIPRTC を `OnceLock` でキャッシュするか（F1 と関連）
- `Function` が module の rt を持つ形への強化（F3）
- 版付き soname・Windows 対応（`docs/backend-rocm-hip-ffi-decision.md` の既存の申し送りと同じ）
- #3157 で増える `unsafe` の追記（#3157 側の作業）
- 実機での確認一式（#3159）
