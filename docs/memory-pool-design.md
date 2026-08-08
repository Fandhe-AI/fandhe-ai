# サイズクラス別バッファプール設計（TASK-#201・REQ-14 14-3）

## 背景

v1（Burn/CubeCL）では GEMM 4096³ でピークメモリが理論値 192MiB の約 17 倍
（3235MiB）に蓄積した（`docs/spec/03-poc/` の PoC 実測。candle の Metal
プール無制限成長と同種の教訓）。`docs/spec/04-requirements.md` REQ-14
14-3 は「バッファプール等のキャッシュ機構を導入する場合、係数上限
（2 倍以内）を維持できなければプール解放 API を提供」を求める。本ドキュメント
は `crates/tensor-core/src/pool.rs` が実装するプール機構の設計判断を記録する。

## 位置付け（opt-in デコレータ）

`tensor_core::pool::PooledMemory<M>` は既存 `MemoryOps` 実装
（`CpuMemory`／`CudaMemory`／`MetalMemory`）を包むデコレータであり、
既定の確保経路（素の `MemoryOps` 実装を直接使う経路）は変更しない。
既定有効化の構成判断は PoC-v2 実測・#202 の係数維持テスト後に行う
（安全側判断）。

## サイズクラス方針: バイトサイズ完全一致

初期実装は「正確なバイトサイズをキーとするバケット」とし、再利用は
バイトサイズ完全一致時のみ行う。冪等 2 乗への切り上げ（capacity > 論理
numel）を許すと、既存 `download` 契約（handle 実長 = numel 前提。
`backend-cpu`／`backend-cuda` の `MemoryOps` 実装）を壊すため、切り上げ
クラス化は将来最適化としてスコープ外の申し送りとする（下記参照）。
GEMM 系ワークロードは同一 shape の反復が支配的であり、完全一致でも
再利用効果は得られる。

## 総量上限・LRU 破棄

`PoolConfig::max_pool_bytes`（既定 128MiB）を超えるアイドル保持は、
挿入順が最も古いエントリからグローバル LRU で自動破棄する。

- `max_pool_bytes == 0` はプール無効（全パススルー）と定義する。
- 上限より大きい単一バッファはプールに入れず即解放する。
- 既定値 128MiB の根拠: GEMM 4096³ のワーキングセット 192MiB + アイドル
  128MiB = 320MiB が REQ-14 の係数 2 倍（384MiB）を侵さない安全側の値。
  確定・調整は PoC-v2 実測に委ねる。

LRU の実装は「サイズ別バケット（`BTreeMap<u64, VecDeque<Entry>>`。バケット
内は常に FIFO）」+「全エントリの挿入順を横断的に保持する `order`
（`BTreeMap<tick, バケットキー>`）」の組合せで、プール全体の最古エントリを
O(log n) で特定する（各バケットは FIFO のためバケット内最古は常に先頭。
よってプール全体の最古は「各バケット先頭の tick の最小値」＝ `order` の
最小 tick に一致する）。

## 返却経路（RAII 維持）・透過ダウンキャスト

`MemoryOps` に明示 `free()` は無い（`buffer.rs` の RAII 一本化方針）ため、
`PooledBufferHandle` を導入し `Drop` で内部ハンドルをプールへ返却する。
内部ハンドルは `ManuallyDrop<Box<dyn BufferHandle>>` で保持する
（`Option` にして `Drop` で `take()` する設計も検討したが、`as_any`／
`as_any_mut` の実装が「フィールドの部分借用」と「`self` 全体の再借用」を
同一ライフタイムで要求する形になり borrow checker が受理しない
〈E0499〉。`ManuallyDrop` なら通常経路は常に中身へ委譲するだけで済み、
所有権を取り出す操作は `Drop::drop` の 1 箇所〈`ManuallyDrop::take`〉に
閉じ込められる）。

`PooledBufferHandle::as_any`／`as_any_mut` は内部ハンドルの `as_any`／
`as_any_mut` へ転送する。`downcast_ref::<H>()` は `Any` オブジェクトが
指す具体型の `TypeId` で判定するため、`PooledBufferHandle` 越しでも各
バックエンドの `download`／カーネルの `downcast_handle::<CpuBufferHandle>()`
等がプール経由バッファで無変更で動作する。

## ゼロ初期化契約の維持（`PoolZeroFill`）

`alloc_zeroed` の「全要素 0」契約を再利用時にも守るため、`PoolZeroFill`
トレイトを各バックエンドに実装させる:

| バックエンド | 実装 |
|---|---|
| CPU | `CpuBufferHandle::data`（`Vec<f32>`）へ `fill(0.0)`（`unsafe` なし） |
| CUDA | `CudaStream::memset_zeros`（デバイス側ゼロクリア。ホスト往復なし） |
| Metal | `MetalBuffer::zero_fill`（`StorageModeShared` の `contents()` への直接書き込み。`read_to_vec` と対になる書き込み版 FFI アクセス） |

前利用データの残留は情報漏えいリスクでもある（`.claude/rules/security.md`
A02/A04）。

`PoolZeroFill::zero_fill` は `&mut dyn BufferHandle` を受け取る。プールから
取り出した直後（まだ `PooledBufferHandle` に包まれる前・呼び出し元へ返す
前の排他所有段階）のバッファに書き込むためであり、この経路に限ることで
CPU 実装は `unsafe` な生ポインタ書き込みなしに `downcast_mut` 経由で安全に
書き換えられる（`BufferHandle::as_any_mut` を新設した理由）。

## プール対象は `alloc_zeroed` のみ（初期実装）

`upload` はパススルーとする。再利用 `upload` には `upload_into`
（既存バッファへの htod 転送）が必要で、CUDA の非同期転送同期契約・
非 contiguous 処理の再実装リスクがあるため、本イシューでは追加しない。

## 計測反映（受け入れ条件の充足機構）

`memory_stats::TrackedAllocation` は各バックエンドの具体ハンドル内に
埋まっているため、以下が追加実装なしで成立する:

- プール保持中も内部ハンドルが生存 → `allocated_bytes` に計上され続ける
- LRU 破棄で内部ハンドルが drop → `on_free` により `allocated_bytes` が
  即座に減少する（#201 受け入れ条件「上限超過時に自動破棄され、ピーク
  計測 API に反映される」の直接検証は
  `crates/backend-cpu/tests/pooled_memory_integration.rs`）

プール固有統計として `PooledMemory::pooled_bytes()`（アイドル保持量）を
公開する。

## 空テンソル契約

`numel == 0` は従来どおり FFI 非経由の空ハンドルとし、プールを介さない
（`buffer.rs` の「空テンソル（numel == 0）の契約」を `PooledMemory` でも
維持する）。

## #202 向け内部解放フック

明示解放 API の公開・係数維持テストは #202 のスコープである。本実装は
`PoolCore::clear_all`（`pub(crate)`）としてプール全保持分を即座に解放する
内部フックのみ用意し、公開 API 化（`PooledMemory` からの `pub` メソッド
追加）は #202 に委ねる。

## スコープ外・申し送り

- **サイズクラス丸め**（冪等 2 乗切り上げ等）: capacity と論理 numel の
  分離・`download` 契約の拡張が前提。
- **`upload` 経路の再利用**: `upload_into` の同期契約設計が前提。
- **プールの既定有効化**の構成判断: PoC-v2 の GEMM 4096³ 実測後。
- **明示解放 API の公開・係数維持テスト**: #202。
- **CUDA/Metal の計測反映の実機検証**: #175 完了後（本イシューでは
  `PoolZeroFill` 実装とフックのみ用意し、`#[ignore]` 実機テストを
  `crates/backend-cuda/tests/memory_real_device.rs`・
  `crates/backend-metal/tests/memory_roundtrip.rs` に追加した）。
