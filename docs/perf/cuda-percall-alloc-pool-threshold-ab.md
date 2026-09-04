# CUDA per-call アロケーション対策の A/B 計測（release threshold・cuMemAlloc 同期割当）

イシュー #1149。`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
（#1146）で確定した「32→33 MiB 帯のデバイス確保・`alloc_zeros`・H2D の
約 2.2 倍の段差」と、`docs/backend-cuda-pool-allocator-decision.md` §8
の保留事項「`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` の調整可否」を、GB10
実機での A/B 計測でデータとして決着させる。

## 状態

**未実測**（本エージェントの実行環境に CUDA 実機がないため。
`docs/perf/cuda-tf32-optin-parity.md` と同型のフォールバック方針）。
本 PR は A/B 計測テスト
（`crates/backend-cuda/tests/large_buffer_percall_alloc_ab_1149.rs`）と
実測記入欄付きの本ドキュメント骨子を成果物とする。実測は GB10 実機に
接続可能な環境で下記「6. 実行手順」のコマンドを実行し、出力（CSV 風
ログ）を本ドキュメントへ転記することで完了する。

## 1. 背景（3 つの論点）

`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
§4.3・§7 を引き継ぎ、本 A/B が決着させるべき論点を以下の 3 点に
再定義する（実装計画 §1.2 と同一）。

| # | 論点 | 根拠 | 本 A/B の役割 |
|---|---|---|---|
| (a) | 32→33 MiB 帯の P1/P2/P3（デバイス確保・`alloc_zeros`・H2D）約 2.2 倍の段差の原因が driver プール（`cuMemAllocAsync` のサイズクラス／トリム）にあるか | #1146 §4.3・§7「未特定」 | 案 A（トリム抑制）・案 B（プール迂回）で直接プローブする |
| (b) | 31→32 MiB 閾値（H-A: ホスト glibc mmap しきい値）・降順走査時の確率的スパイクが案 A／B で変化するか | #1146 §5 H-A 支持・H-B 棄却 | 期待値は「変化なし」（ホスト側要因）を null 結果として確定させる |
| (c) | `docs/backend-cuda-pool-allocator-decision.md` §8 の保留事項をデータで閉じる | 同 doc §1・§8 | 案 A の実測が保留判断の根拠になる |

## 2. 事前登録した判定基準

- **解消**: 対策条件で 32→33 MiB の P1/P2/P3 段差比（33 MiB median ÷
  32 MiB median）が 1.2 未満に縮小する、または ≥32 MiB 帯の P0/P4
  median がベースライン比 0.5 倍以下かつ `slow_count` が全ランで 0
- **緩和**: 段差比が 1.2 以上 1.8 未満、または ≥32 MiB 帯の P0/P4
  median がベースライン比 0.5〜0.8 倍、または `slow_count` 合計が
  ベースライン比半減以下
- **効果なし**: 上記いずれも満たさない（ブラケット baseline 同士の
  ばらつき範囲内）

判定は 5 回ラン median の中央値（key サイズ = 32・33 MiB・P7 dim4096）
と、ブラケット baseline（前後）の差を「ドリフト幅」として併記する。

## 3. 計測方法

### 3.1 条件

| 条件 | 確保 | ゼロ初期化 | H2D | D2H | 解放 | release threshold |
|---|---|---|---|---|---|---|
| baseline（なし） | `stream.alloc`/`alloc_zeros`/`clone_htod`（cudarc 既定 = `cuMemAllocAsync`） | cudarc 内 memset | `clone_htod` | `clone_dtoh`（新規 `Vec`） | `CudaSlice` drop（`cuMemFreeAsync`） | 既定（実測値をログへ記録） |
| A（release_threshold） | 同上 | 同上 | 同上 | 同上 | 同上 | `u64::MAX`（`ReleaseThresholdGuard` で RAII 復元） |
| B（sync_alloc） | `result::malloc_sync`（`cuMemAlloc`）→ `upgrade_device_ptr` | `memset_zeros`（cudarc safe API） | `memcpy_htod` | `clone_dtoh`（新規 `Vec`。baseline と構造的に同一） | `synchronize` → `leak()` → `result::free_sync`（`cuMemFree`） | 既定 |
| A+B（both） | B と同じ | 同上 | 同上 | 同上 | 同上 | `u64::MAX` |

B は baseline との差分を「確保／解放 API（`cuMemAlloc`/`cuMemFree` vs
`cuMemAllocAsync`/`cuMemFreeAsync`）」の 1 変数に限定する（memset・
H2D・D2H・カーネル起動は baseline と同じ cudarc safe API を
`upgrade_device_ptr` 経由の `CudaSlice` に対して呼ぶ）。A+B は理論上
B がプールを迂回するため A の影響を受けない対照として使う。

### 3.2 フェーズ

| フェーズ | 内容 | 目的 |
|---|---|---|
| P1 | 確保＋解放（`synchronize` で完了確定） | 論点 (a) |
| P2 | ゼロ初期化確保＋解放 | 論点 (a) |
| P3 | H2D のみ | 論点 (a) |
| P0 | 転送のみ合算（H2D×2 + ゼロ初期化確保 + sync + D2H + 解放 + sync。#1146 P0 と同一構造） | #1123 症状との接続・論点 (b) |
| P4 | D2H のみ・宛先が毎回新規 `Vec` | 二峰性発生箇所・論点 (b) |
| P7 | 本番経路レプリカ（`CudaMmaGemm`。`upload_f16`→`alloc_output_f16`→`launch_f16`→`download_f16`。M=N=K=4096） | dim4096 相当（#1123 の約 261〜263 ms）を条件別に直接比較 |

### 3.3 サイズ・ラン数・順序

- サイズ: `[24, 28, 31, 32, 33, 36, 48, 64]` MiB（f16 要素数換算）
- P7 形状: `(4096, 4096, 4096)`（A/B/C 各 32 MiB）
- key サイズ（32・33 MiB・P7 dim4096）は 5 ラン、それ以外は 3 ラン
  （20/20 warmup/iters の `MeasurementConfig`）
- 条件順序はブラケット方式: サイズごとに
  `baseline → A → B → A+B → baseline` の順で実行し、前後の baseline
  差を「ドリフト幅」として記録する
- 走査順: 昇順パス（全フェーズ）+ 降順パス（P0／P4 のみ）
- 各条件ブロックの前後で `RESERVED_MEM_CURRENT`／`USED_MEM_CURRENT`／
  `RELEASE_THRESHOLD`（`cuMemPoolGetAttribute`）を読み取り記録する

## 4. 環境

**未実測のため空欄**（実測時に `docs/real-hardware-verification-env.md`
§2〜§4・§6 準拠で記入する。内部ホスト名は書かない）。

| 項目 | 値 |
|---|---|
| GPU | （未実測） |
| compute capability | （未実測） |
| CUDA / nvcc / rustc 版 | （未実測） |
| has_async_alloc | （未実測） |
| 検証対象コミット | `perf/1149-cuda-percall-alloc-ab` ブランチ HEAD |
| 実行日 | （未実測） |

## 5. 事実（実測）

**未実測**。実測後、条件別・サイズ別の median・q1/q3・slow_count・
cold・pool 属性推移・P7 dim4096 の条件別 median と checksum 一致結果を
ここへ記入する。

## 6. 実行手順

```sh
# ローカル検証（Mac・CUDA 非搭載）
cargo fmt --all -- --check
cargo clippy -p fandhe-ai-backend-cuda --all-targets --all-features -- -D warnings
cargo test -p fandhe-ai-backend-cuda --test large_buffer_percall_alloc_ab_1149
cargo test -p fandhe-ai-backend-cuda --test large_buffer_percall_alloc_ab_1149 -- --ignored --list

# GB10 実機（`docs/real-hardware-verification-env.md` §2〜§4・§6 準拠）
cargo test -p fandhe-ai-backend-cuda --release \
    --test large_buffer_percall_alloc_ab_1149 \
    -- --ignored --nocapture --test-threads=1
```

`--release`・`--test-threads=1` は必須（`crates/backend-cuda/tests/large_buffer_percall_alloc_ab_1149.rs`
冒頭ドキュメンテーションコメント参照。release threshold の変更が
プロセス全体の driver プール状態を変えるため、他テストとの並行実行は
プール状態の競合を招く）。

## 7. 判定（未実測のため保留）

論点 (a)(b)(c) それぞれについて、§2 の事前登録基準に照らした
「解消／緩和／効果なし」の判定を実測後にここへ記入する。

## 8. スコープ外・引き継ぎ

- `crates/backend-cuda/src/pool.rs` 冒頭コメント 13 行目
  「既存の `stream.alloc`/`alloc_zeros` は同期 `cuMemAlloc` 系のまま」
  という記述は、cudarc 0.19.8 の実挙動（`has_async_alloc=true` の
  環境では `stream.alloc`/`alloc_zeros` も内部で `cuMemAllocAsync` を
  使う。`driver/safe/core.rs:1530〜1538`）と一致しない。
  `docs/backend-cuda-pool-allocator-decision.md` §2 は正しく
  「内部で既に `cuMemAllocAsync` を使っている」と記述済みのため、
  `pool.rs` 側コメントの是正が必要（本イシューでは本番コードの
  コメントであっても変更しない。是正イシューの起票をユーザーへ
  提案する）
- 案 A／B いずれかが有効だった場合の本番結線は #1153 のスコープ
  （D2H 宛先の事前タッチ／pinned 再利用と合わせて判断）
- 32→33 MiB 段差が driver プール起因でないと判明した場合の次段の
  切り分け（cudarc 内部・unified memory ページング等）
- 降順走査限定の確率的スパイクの根本原因特定（#1146 §7 引き継ぎのまま）

## 9. 関連ファイル

- `crates/backend-cuda/tests/large_buffer_percall_alloc_ab_1149.rs`
  （本 A/B 計測専用テスト。条件・フェーズ・ブラケット順序の実装）
- `crates/backend-cuda/tests/large_buffer_percall_alloc_transfer_triage.rs`
  （#1146。フェーズ分解・サイズスイープの元実装。ヘルパー関数の設計を
  踏襲）
- `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
  （#1146。§10 に本ドキュメントへの参照を追記）
- `docs/backend-cuda-pool-allocator-decision.md`（§8 に本 A/B の
  結果に基づく判断を追記）
- `docs/perf/cuda-wmma-f16-perf-triage.md`（§6 の #1130 項に本 A/B の
  結果と参照先を追記）
- `crates/backend-cuda/src/pool.rs`（`CudaAllocator::release_cached`。
  driver プールのトリム・属性読み取りの既存実装パターン）
- `docs/real-hardware-verification-env.md`（実機接続・転送手順）

## 10. 受け入れ条件との対応

| # | 受け入れ条件 | 対応 |
|---|---|---|
| 1 | 案 A（release threshold 引き上げ）を GB10 実機で計測 | §3.1・§6（実行手順のみ実装済み。実測は未完了） |
| 2 | 案 B（`cuMemAlloc` 同期割当）を GB10 実機で計測 | 同上 |
| 3 | なし／A／B／A+B の全組合せを計測 | §3.1（ブラケット方式で全 4 条件を実装済み） |
| 4 | 32 MiB 前後を中心に複数サイズで計測し中央値・ばらつき・二峰性を記録 | §3.3（8 サイズ・key サイズ 5 ラン） |
| 5 | どの対策が病態を解消（緩和）するか結論を明記 | §7（未実測のため保留。事前登録判定基準は §2 に記録済み） |
| 6 | `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md` へ追記 | §10「#1149 A/B 結果の要約」として追記済み（本ドキュメントへの参照込み） |
| 7 | 対策コードを本番結線しない（計測専用） | `crates/backend-cuda/src/**` 差分ゼロ（`git diff origin/main --stat -- crates/backend-cuda/src` が空であることを確認） |
