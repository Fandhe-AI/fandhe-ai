# tape・中間バッファの host arena 再利用による alloc 固定費削減の設計

イシュー #2103（親 #2099 Phase 3「負けセルの対処」）。後続の実装は #2104。**本 doc は設計の記録のみ**であり、コード変更・実測は行わない。

## 0. 位置づけ・スコープ

- 本 issue は docs のみ。実装の主対象は `tensor-core`（`Storage` の返却経路・arena 本体）と `backend-cpu`（出力確保の取得経路）で、`autodiff` は間接的に恩恵を受けるだけである（`Tape` 自体の構造は変えない）。
- 不変事項（本 doc・#2104 とも変更しない）: tolerance・baseline・`Cargo.toml`／`Cargo.lock`・ガードレール閾値・`docs/spec/`、REQ-2 の統一複合判定、REQ-14 のピークメモリ係数 2 倍。
- ユーザー承認が必要な事項は採用せず §7 に列挙するだけとする（依存追加・新規 `unsafe`・facade 公開面の拡張・`SME_PRODUCTION_ENABLED` の切替）。
- 対象外: CUDA／Metal への同等機構、実装そのもの（#2104）。起票はユーザー承認後（`.claude/rules/out-of-scope-tracking.md`）。

## 1. 背景・実測根拠（期待効果の上限）

#1980（train）・#1981（infer）の固定費再分解で、CPU の中間バッファを演算ごとに新規確保していることが削減候補の 1 つに挙がった。ただし既存実測は効果が小さいことを示している。

| 出典 | 内容 |
|------|------|
| `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 | 640 要素規模では「確保＋`Tensor` 構築は 165 ns 未満」 |
| `docs/cpu-matmul-fixed-cost-design.md` §0・§3.C | 出力確保が本番経路の固定費として意味を持つのは DGX Spark GB10 の N=2048 のみ（`iter_total` の約 8%）。M4 Max と N=512/1024 では無視できる水準。同 doc は「案 2（出力バッファの再利用＝プール化）」を後続候補として保留しており、本設計はその具体化である |
| 同 doc §10（#1481・#1482） | 並列ゼロ埋めは REJECT 確定。#2104 は同じ実験を再実行しない |
| `docs/perf/train-step-phase-breakdown.md` §17 | size=64 では `step_total` が約 0.3〜1.6 ms、`tape_drop` は 1 µs 未満 |

結論: size=64 の小形状では効果が小さい見込みが高い。**効果の上限は未定量であり、推定値で埋めない**（`docs/device-memory-pool-design.md` §1 と同じ立場）。#2104 には §6 の事前ゲートを引き継ぐ。

## 2. 現状整理（コードから確定できる事実）

| # | 事実 | 出典 |
|---|------|------|
| 1 | `Tensor<T>` は `Arc<Storage<T>>` と `offset`・`strides` で構成。`Storage { data: Vec<T> }` は非公開で、`Storage` 自体の `Drop` 実装は無い | `crates/tensor-core/src/tensor.rs:33`（`struct Storage`）・`:59`（`struct Tensor`） |
| 2 | `host_slice()` は非連続時に `Arc::try_unwrap` で `data` を move out する | `tensor.rs:366-383` |
| 3 | CPU バックエンドの出力確保は `vec![0.0f32; n]` が大半。集約点 `zeroed_output`／`zeroed_output_with_threshold`（#1299）がある | `crates/backend-cpu/src/ops.rs:208`・`:223` ほか |
| 4 | `Tape` のノード列は `nodes: RefCell<Vec<TapeNode>>`。`Tape::reset` が葉プレフィックスまで truncate し、そこで `TapeNode::value` を drop する | `crates/autodiff/src/tape.rs:2354`・`:2732` |
| 5 | retain_graph 契約により reset／drop までノード値は解放されない。例外は checkpoint 区間の解放 | `tape.rs:3470`（`release_checkpoint_region`） |
| 6 | 既存プール①: `PooledMemory`（opt-in の `MemoryOps` デコレータ。バイト数完全一致・LRU。本番経路には未結線） | `crates/tensor-core/src/pool.rs`・`docs/memory-pool-design.md` |
| 7 | 既存プール②: `SizeClassPool<H>`（`H: Send` のハンドル非依存コア。CUDA／Metal のデバイスバッファ用。`put` は退避ハンドルを返す） | `crates/tensor-core/src/pool_core.rs:259,318,399`・`docs/device-memory-pool-design.md` |
| 8 | 既存機構③: `DeviceParamStore`。パラメータと velocity を 1 本の連結 `DeviceBuffer` として Tape と独立の寿命で常駐させる。**プールではなく常駐ストア** | `crates/autodiff/src/optim/device_store.rs` |
| 9 | `thread_local!` の既存例 | `crates/backend-cpu/src/ops.rs:90` ほか |
| 10 | panel バッファを呼び出し単位で再利用する既存例（#556 `PanelBuffers`） | `docs/perf/cpu-gemm-packing-buffer-reuse.md` |
| 11 | `Element: Copy + Send + Sync + 'static` のため `Any` の downcast が `unsafe` なしで使える | `crates/tensor-core/src/element.rs:29` |

## 3. 設計

### 3.1 管理対象と取得・返却経路

- 管理するのは CPU ホスト側の `Storage<T>::data`（`Vec<T>`）のみ。初期は `f32` に限る。
- **返却**: `Storage` に `Drop` を実装し、arena が有効なら `std::mem::take` で `Vec` を取り出して返す。`T == f32` は `Any` の downcast で判定する（`unsafe` 不要）。
- **取得**: `zeroed_output` 系の集約点と、`vec![0.0f32; n]` の各箇所を段階的に置き換える。
- 取得した `Vec` は必ず `clear()` → `resize(len, 0.0)` でゼロ化する。`vec![0.0; n]` と同じ「全要素 0」の契約を保つため全カーネルが bit 同一になり、前の利用のデータが残らない（A02/A04。先例は `pool.rs` の `PoolZeroFill`）。
- `Storage` に `Drop` を実装すると `data` を move out できなくなる。`tensor.rs` の `Arc::try_unwrap` 直後の `unique.data` 取り出しを `mem::take` へ書き換える必要がある（bit 同一）。#2104 の作業項目とする。
- 未初期化確保（`set_len`）・`alloc_zeroed`・`mallopt` は `unsafe` またはプロセスグローバル変更にあたるため採用せず §7 に列挙する（`cpu-matmul-fixed-cost-design.md` §3.C 案 3 と同じ扱い）。

### 3.2 既存機構との棲み分け（AC1）

| 観点 | host arena（本設計） | `DeviceParamStore` | `SizeClassPool<H>` | `PooledMemory` |
|------|------|------|------|------|
| 管理メモリ | ホスト `Vec<T>`（`Storage` 内） | 連結 `DeviceBuffer`（パラメータ・velocity） | デバイスハンドル（CUDA／Metal） | `MemoryOps::alloc_zeroed` のハンドル |
| 寿命 | Tape の 1 step 内または step 間 | 学習全体（Tape と独立） | 演算単位 | デコレータの生存期間 |
| 所有者 | プロセス（§3.3） | 呼び出し元（facade） | 各 GPU バックエンド | ラップした側 |
| 入口 | `Storage::drop`／CPU 出力確保 | `new`／`step` | backend-cuda／metal の確保 | `MemoryOps` |

非競合の根拠:

- `DeviceParamStore` のバッファは `DeviceBuffer` で `Storage` を経由しないため、arena に入ることは型の上で起こり得ない。
- `SizeClassPool`・`PooledMemory` の対象はデバイス／`MemoryOps` ハンドルで、`Vec` を扱う arena と交わらない。
- GPU から読み戻した `Tensor<f32>` は arena に返り得るが、ホスト `Vec` であり二重管理にならない。GPU 中間バッファの arena 化は対象外。

実装上の禁止事項: arena を `PooledMemory` の内側に入れ子にしない。`DeviceParamStore` の staging バッファを arena の対象にしない。

### 3.3 thread-local と global の判断（AC2）

| 基準 | thread-local | global（Mutex） |
|------|------|------|
| (a) 取得側と返却側のスレッド不一致（`Tensor: Send+Sync`。Drop は別スレッドで起き得る。例: DataLoader の prefetch、Tape のスレッド間 move） | 返却スレッド側に偏って溜まり取得側は枯渇。スレッド数 × 上限でメモリが膨らむ | 問題なし |
| (b) contention（現状の CPU 演算は出力を呼び出し元スレッドで確保し rayon worker では確保しない見込み。確認は #2104） | ロックなし | 通常は非競合の Mutex 1 回。Tape の並行学習で競合し得る |
| (c) スレッド終了時の drop | `LocalKey::try_with` 失敗への対処が必要 | static は drop されない（OS が回収） |
| (d) REQ-14 の解放 API | 呼び出しスレッド分しか解放できない | プロセス全体を一括解放できる |
| (e) 統計・観測 | スレッドごとに分散 | 1 箇所に集約 |
| (f) 数値への影響 | 無し（ゼロ化で bit 同一） | 無し |

**推奨: global（プロセス共有・Mutex）を主案とする。** 理由は (a)(d)(e)。確保 1 回が 165 ns 未満の領域では Mutex の固定費も相対的に小さい。thread-local の前段キャッシュ（tcache 型）は、#2104 以降で競合が実測された場合の後続案とする。

不採用案: Tape が arena を所有する案。`BackendOps` の出力確保引数に arena を通す必要があり `tensor-core` の公開 trait の破壊的変更になる。`Drop` への返却フックはどちらの案でも必要。

保管の実体は #2104 が選ぶ。案 i: `SizeClassPool<Vec<f32>>` を再利用（`put` が退避 `Vec` を返すのでロック外で drop 可能）。案 ii: バイト数完全一致の最小構成を新設（`PooledMemory` と同じ意味論）。学習ループは毎 step 同形状のため完全一致でもヒット率は高いと見込むが**推定**であり #2104 で計測する。

### 3.4 API 配置案（最終確定は #2104）

`crates/tensor-core/src/alloc.rs`（#2104 で新設。本 issue では作らない）に、取得 `take_zeroed_f32(len) -> Vec<f32>`、`Storage::drop` から呼ぶ `pub(crate)` 返却関数、設定・解放 `set_host_arena_config`／`release_host_arena`、統計（`PoolStats` 流用）を置く案。`tensor-core` は内部クレートのため backend-cpu 向け `pub` 項目の追加は facade 公開面の拡張にあたらない。facade 公開・環境変数スイッチは §7 に列挙し #2104 では採用しない。既定は OFF（opt-in）、`max_pool_bytes` 既定は 128 MiB とし、REQ-14 の係数 2 倍および `docs/peak-memory-coefficient-decision.md` と整合させる。

## 4. FAQ・実装上の制約（AC3）

**dropper timing（いつ返却されるか）**

- `Arc` の最後の参照が消えた時点で返る。view（`transpose`／`narrow` 等）が生きている間は返らない。
- retain_graph 契約により 1 つの Tape 内のノード値は reset まで返らない。したがって再利用は主に (1) step 間（reset → 次 step）、(2) backward 中の一時値、(3) checkpoint 区間の解放 → 再計算、で起こる。
- ピークメモリが増えないとは言えない。reset 後に arena が保持するバッファと、次 step の作業バッファが同時に存在しうるため、アイドル保持分がそのままピークへ上乗せされる。§3.4 の 128 MiB 上限は上乗せ量の上限であって、ワークロードによっては REQ-14 の理論最小ワーキングセット比 2 倍（`docs/peak-memory-coefficient-decision.md` の数値契約）を超えうる。判定方法と破棄条件は §6 の事前ゲートで規定する。

**panic・unwind 中の返却**: 返却処理は panic しない。Mutex が poisoned の場合や上限超過時は通常の drop にフォールバックする（`pool.rs` と同方針）。

**再入**: 退避した `Vec` はロックの外で drop する。`Vec` の drop は `Storage::drop` を通らないため再入しない。

**contention**: 複数 Tape の並行学習・DataLoader prefetch との競合を #2104 の計測項目にする。利用者が `Tensor::new` に渡した `Vec` も arena に返り得るため、上限と LRU で抑える。

**所有権が利用者へ移る経路**: `host_slice` の `Cow::Owned` 等で利用者に移った `Vec` は arena に戻らない（正しい挙動）。

**プロセス終了時**: global の static は drop されない。リーク検出ツール使用時はアイドル保持分が「到達可能」として報告される旨を注記する。

## 5. bit 一致・セキュリティの契約

- ゼロ化を保つため全カーネルは bit 同一のまま。回帰テストは #2104 が実装する: arena の on／off で出力の bit が完全一致すること（既存 `gemm_output_alloc_bit_exact.rs`／`tape_matmul_cpu_bit_exact.rs` と同型）。
- OWASP 観点: A02/A04 再利用バッファのゼロ化（前テンソルのデータ漏えい防止）。A04/A05 アイドル保持の総量上限・LRU・明示解放 API・既定 OFF。A08 未初期化メモリ／`unsafe` は不採用、poisoned Mutex は panic せず通常 drop、tolerance・baseline・閾値は不変。A06 依存追加なし（標準ライブラリのみ）。A03 環境変数スイッチを採る場合は値の厳密検証が必要（採否自体が §7 の判断事項）。

## 6. #2104 への引き継ぎ

- 置き換え候補は §2 の事実 3 の集約点と `vec![0.0f32; n]` 各箇所。集約点から段階的に結線する。
- **ピーク判定（アイドル保持込み。必須）**: `docs/peak-memory-coefficient-decision.md` と同じ内部計測 API で、arena のアイドル保持バイト数を含めた `peak_bytes` を計測し、対理論最小ワーキングセット比が 2.0 以内であることを確認する。arena の on／off で同一ワークロードを比較し、on 側が 2.0 を超える、または off 側より悪化する場合は既定の `max_pool_bytes` を下げる。下げても超える場合は REJECT として記録する。
- **破棄条件（#2104 で設計に含める）**: 返却時に「保持バイト数 + 現在の生存バッファ量」が理論最小ワーキングセット比 2.0 相当を超えるなら、返却せず通常の drop へ落とす。加えて reset 後に一定 step 再利用されなかったバッファを解放する。`release_host_arena` による明示解放も残す。係数 2 倍自体は変更しない（ユーザー承認事項）。
- **事前ゲート案**: 実装前に 1 step あたりの確保回数・バイト数を計数し、§1 の実測と突合して効果が見込めない場合は REJECT として記録する（結果が REJECT でも記録価値がある）。
- framework-compare は registry の `fandhe-ai =0.9.0` 固定のため、未公開コードの A/B はリポジトリ内の `#[ignore]` 診断ハーネスか path patch の腕のいずれかで行う。
- 事前登録規則（5 run 中央値・checksum 完全一致・`ratio<=1.00`・専有ゲート）は #2104 の Issue 契約に従う。数値は本 doc では決めない。

## 7. ユーザー判断事項・スコープ外

- ユーザー判断事項（本設計では採用しない）: facade への公開、環境変数スイッチ、既定での有効化、`unsafe` 系高速化（未初期化確保・`alloc_zeroed`・`mallopt`）、`SME_PRODUCTION_ENABLED` の切替、依存の追加。
- スコープ外: CUDA／Metal 版、サイズクラス丸めの最適化。
