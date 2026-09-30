# Metal `tape_build` 削減（opt-in）と infer GPU 起動固定費の診断（基盤のみ。実機実測は未実施）

イシュー #2114。`docs/perf/train-step-phase-breakdown.md` §17.3・§17.4 が報告した
Metal train の `tape_build`（14.9〜15.7 µs。CPU 0.1 µs・CUDA 2.5〜3.0 µs）と、
`docs/perf/infer-reuse-phase-breakdown.md` §10.3・§10.4・§10.6.4 が報告した
Metal infer reuse（365.2 µs。CPU reuse 175.3 µs の約 2.1 倍）の固定費を、
デバイス存在確認のキャッシュ（opt-in・既定 OFF）とフェーズ分解診断で切り分ける。

## 1. 状態

| 項目 | 状態 |
|---|---|
| デバイス存在確認キャッシュ（`crates/backend-metal/src/fixed_cost_diag.rs`。**既定 OFF**） | 実装済み（状態機械の単体テストは Linux CI で実行） |
| 診断カウンタ（存在確認 probe／ヒット・ホスト⇔デバイス転送の回数とバイト数） | 実装済み |
| フェーズ分解診断（`crates/facade/tests/metal_infer_tape_build_fixedcost_diag.rs`） | 実装済み（`#[ignore]`・Metal 実機） |
| wall と GPU busy の診断（`crates/backend-metal/src/infer_fixed_cost_diag_tests.rs`） | 実装済み（`#[ignore]`・Metal 実機） |
| 事前登録の判定規則（`docs/perf/logs/metal-tape-build-infer-phase-2114/RULE.txt`） | 実測前に固定済み |
| 計測・集計スクリプト（`orchestrate.sh`／`aggregate.py`） | 実装済み（`aggregate.py --self-test` は Linux で確認） |
| Apple M4 Max での 5 run 実測と仮説判定 | **未実施（Mac 実機セッションへ申し送り。本 PR は Linux 環境で作成）** |
| 既定 ON 化・tape.rs 側の最適化 | 本イシューの対象外（§7 に起票案のみ。起票はしていない） |

本 doc に実測値は一切載せていない。

## 2. コード事実（読解）

| 箇所 | 事実 |
|---|---|
| `crates/autodiff/src/tape.rs::Tape::new_with_ops` | `AtomicU64::fetch_add` と `RefCell::new(Vec::new())`・`HashMap::new()` だけで構成され、ヒープ確保をしない。構築そのものは O(1) |
| `crates/backend-metal/src/ops.rs::MetalBackendOps` | ユニット構造体（ZST）。`new()` は無コスト |
| `crates/facade/src/lib.rs::resolve_ops`（Metal 分岐） | `tape_for(Device::Metal)` の**呼び出しごと**に `MetalDeviceProvider::select` → `probe_all()` を実行し、得た `DeviceInfo` を捨てる（存在確認にしか使わない） |
| `crates/backend-metal/src/device.rs::probe_all` | 毎回 `probe_gpu_core_count()`（IOKit: `IOServiceMatching`／`IOServiceGetMatchingService`／`IORegistryEntryCreateCFProperty`）・`MTLCopyAllDevices()`・`device.name().to_string()`・`recommendedMaxWorkingSetSize()` |
| `crates/facade/src/compat/sequential.rs::predict_resident` | 呼び出しごとに `crate::tape_for(store.device())` を実行する。同じ probe コストが infer reuse の 1 呼び出しごとにも乗る |
| `crates/backend-metal/src/context.rs` | encode／command buffer／`waitUntilCompleted` の診断カウンタ（#1099）と `synchronize_with_gpu_timestamps`（`GPUStartTime`／`GPUEndTime`）が既にある |

**仮説（事実としては扱わない）**: Metal `tape_build` の約 15 µs の大部分は `resolve_ops` の
存在確認（IOKit と `MTLCopyAllDevices`）で、`Tape::new_with_ops` は 1 µs 未満。

## 3. 設計

### 3.1 opt-in: デバイス存在確認キャッシュ（既定 OFF）

- 実体は `fixed_cost_diag.rs`（`cfg` なし・objc2 非依存。Linux CI で単体テスト）
- `METAL_DEVICE_VERIFY_CACHE_DEFAULT_ENABLED = false`（A/B の after 側ではこの定数だけを反転する。
  `tensor-core::alloc::HOST_ARENA_DEFAULT_ENABLED`〈#2104〉と同型）
- 有効化は thread-local のスコープ付きガード `override_device_verify_cache_for_scope`（`!Send`・ネスト可）のみ。
  環境変数などの隠れた有効化経路は作らない。`#[doc(hidden)] pub` で facade からは再公開しない
- `facade::resolve_ops` の Metal 分岐は `verify_device_cached` 越しに従来の `select_from` を呼ぶ。
  OFF では従来どおり毎回実行される。ON では**成功だけ**を記録し 2 回目以降の probe を省く
  （失敗はキャッシュしない。fail-closed）
- 残留リスク: `resolve_ops` を経由する `release_cached_memory`／`memory_pool_stats` も ON の間は 2 回目以降の
  probe を省く。Apple Silicon の統合 GPU は消えないが、Intel Mac の eGPU 取り外しには追従しない。
  既定 OFF であり、既定 ON 化は別イシューで判断する

### 3.2 tape.rs 側の候補（設計検討のみ。本 PR では実装しない）

- `nodes` の `with_capacity` による事前確保: 構築時点ではヒープ確保がゼロのため、確保コストが
  tape_build 側へ**移るだけ**で削減にならない
- node pooling（thread-local で `Vec<TapeNode>` の容量を再利用）: 効くのは forward 中の再確保で、
  tape_build は削減しない。フェーズ分解で forward 中の再確保が無視できない規模だと分かった場合のみ
  別イシューで検討する（`tensor-core::alloc`〈#2104・#2448〉に先例のある opt-in の型）

### 3.3 診断カウンタ（本番経路への影響の小さい順）

- (a) 既存の `__diagnostic_batch_counters_snapshot`（encode／command_buffers／wait）と、
  `fixed_cost_diag` の `verify_probe_calls`／`verify_cache_hits` の差分
- (b) `memory.rs` の実転送点（`upload_inner`・`upload_view`・`upload_into`・`download_inner`）の
  `Relaxed` の `fetch_add`（回数とバイト数）。`__diagnostic_fixed_cost_counters_snapshot()` で取得
- (c) `infer_fixed_cost_diag_tests.rs`: upload → `linear_forward_device` × 2 →
  `synchronize_with_gpu_timestamps` → download の wall と GPU busy の差（ホスト側の起動・同期固定費）。
  本番の `synchronize` にはフックしない

## 4. 事前登録仮説

| 仮説 | 内容 | 判定（RULE.txt・閾値は事後に変えない） |
|---|---|---|
| H1 | `provider_select` が tape_build の主因 | `micro.provider_select` が `fresh_off.tape_build` の 50% 以上 |
| H1a（補助） | IOKit 単独の割合 | `probe_gpu_core_count / provider_select` を記録 |
| H2 | `Tape::new_with_ops` は 1 µs 未満 | `micro.tape_new` の中央値 < 1 µs |
| H3 | reuse の 1 反復は encode 2・command_buffers 1・wait 1 | 5 run すべてで成立（#1580 実測との照合） |
| H4 | infer の wall − GPU busy がホスト固定費の主体 | (c) の wall − GPU busy − upload − download が wall の 50% 以上 |

再実行しない既存実験: #1580／#1911 の chain 単一同期 A/B・#1477 の readout 腕。

## 5. 計測プロトコル

判定規則の正は `docs/perf/logs/metal-tape-build-infer-phase-2114/RULE.txt`（実測前に固定）。要点:

- 1 run = 1 プロセスで独立 5 run。各セルは run ごとの median の中央値と min–max
- checksum（出力全要素の f32::to_bits 順序込み FNV-1a ダイジェスト）は arm 内で run 間・off と on・`reuse_decomposed` と `reuse` で完全一致（fail-closed）
- 比 `on / off`（tape_build と iter_total）の `<= 1.00` は記録のみ（ADOPT 判定はしない）
- m4max は `record_only`（共有環境）

テストは 2 本（いずれも `#[ignore]`・Metal 実機）:

- `metal_infer_decomposition_matches_public_api_bit_exact`: 分解した写しが公開 `predict_resident` と bit 一致し、
  キャッシュ OFF／ON で出力が bit 同一であること
- `metal_infer_tape_build_fixedcost_phases`（record-only）: fresh／reuse／reuse_decomposed × キャッシュ off／on と
  `tape_build_micro` を 20 反復 × 4 ラウンド（arm 順を反転）で計測し JSON 行を出力する

**窓定義の注記**: fresh の `tape_build` は計測窓の**内側**に入れる。bench-fandhe（#1217 の D4:
`make_tape` は計測窓の外）とは窓の定義が違うため、`infer-reuse-phase-breakdown.md` §10.2 の数値とは直接比較できない。

## 6. 結果（実機実測後に記入。現在は未実測）

### M4 Max

未実測。`bash docs/perf/logs/metal-tape-build-infer-phase-2114/orchestrate.sh m4max` の後
`python3 .../aggregate.py <dir>` の出力を転記する。(c) の出力も転記する。

### 仮説判定

未判定（実測後に RULE.txt の規則に従って記入する）。

## 7. 起票案（列挙のみ。起票はユーザー承認待ち）

- 存在確認キャッシュの既定 ON 化 A/B（H1 支持かつ ratio < 1.00 の場合）
- tape.rs の node pooling／pre-allocation（forward 中の再確保が支配的と分かった場合）
- `probe_all` の GPU コア数のプロセス内キャッシュ化（`MetalDeviceProvider::select` の公開挙動を保つ形）
- CUDA の `CudaDeviceProvider::select`（2.5〜3.0 µs。§17.6.3）への同型の検討
- bench-fandhe（registry `=0.9.0` ピン）への反映: 本変更は registry 版に入っていないため次のピン更新以降

## 8. 限界

- 本テストは HEAD（main）を計測する。#1980・#1981 の実測は registry 0.9.0 で、以降の変更が入っている
- Linux では実測していない（申し送り）。M4 Max の 5 run が受け入れ条件
- H1 の帰属は中央値同士の比による推定であり、因果の確定ではない
- 計測区間の `Instant` 呼び出し自体のオーバーヘッドが細かい区間（数百 ns〜数 µs）に含まれる
- キャッシュ ON の腕は同一プロセス内でキャッシュ状態が共有されるため、ON の腕でのみヒットし、OFF の腕は
  毎回 probe する（OFF 中は `verified` を読まない・書かない契約による）

出典: `docs/perf/train-step-phase-breakdown.md` §17、`docs/perf/infer-reuse-phase-breakdown.md` §10、
`docs/perf/metal-infer-chain-single-sync.md`、`docs/perf/cpu-predict-resident-fixedcost.md`（同型の診断基盤・#2105）。
