# CUDA per-call アロケーション対策の A/B 計測（release threshold・cuMemAlloc 同期割当）

イシュー #1149。`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
（#1146）で確定した「32→33 MiB 帯のデバイス確保・`alloc_zeros`・H2D の
約 2.2 倍の段差」と、`docs/backend-cuda-pool-allocator-decision.md` §8
の保留事項「`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` の調整可否」を、GB10
実機での A/B 計測でデータとして決着させる。

## 状態

**実測完了・判定確定**（DGX Spark GB10 実機。イシュー #1153 の
Phase 0 として本イシューの未完了分を完走させた）。昇順・降順パス
（24〜64 MiB の全 8 段階・4 条件・全 phase）・P7（`CudaMmaGemm` 本番
経路レプリカ・dim4096）とも完走・全量記録済み（§5）。**孤立マイクロ
ベンチマーク（P1〜P3）と現実的な呼び出しパターンのレプリカ（P7）とで
結論が相反する**という重要な知見が得られたため、判定は P7 を優先して
確定した（§7）。release threshold 引き上げ（案 A）は一度
`CudaAllocator::new` へ実装したが、P7 実測後に**差し戻した**（§7）。

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

`docs/real-hardware-verification-env.md` §2〜§4・§6 準拠（内部ホスト名
は書かない）。

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GB10（sm_121） |
| compute capability | 12.1（`nvidia-smi --query-gpu=compute_cap` 出力） |
| CUDA / rustc 版 | CUDA 13.0 系・rustc 1.97.0 |
| has_async_alloc | true（`mem_pool_attr` 行が全条件で `disabled` を出さず値を返している） |
| 検証対象コミット | `2250dce`（origin/main。イシュー #1153 作業ブランチの分岐元） |
| 実行日 | 2026-09-05 |
| GPU 占有状況 | 計測前 `nvidia-smi --query-gpu=utilization.gpu` 0% |

## 5. 事実（実測。昇順パス。5 回計測 median）

`cargo test -p fandhe-ai-backend-cuda --release --test
large_buffer_percall_alloc_ab_1149 -- --ignored --nocapture
--test-threads=1` を GB10 で実行した（生ログ: `docs/perf/logs/
cuda-gemm-mma-f16-pool-1153/ab-1149-run1.log`。内部ホスト名なし）。

### 5.1 32→33 MiB 段差比（median の比。論点 (a)）

| 条件 | P1 alloc_only (32→33 MiB, 比) | P2 alloc_zeros (比) | P3 h2d_only (比) |
|---|---|---|---|
| baseline | 1.605 → 3.522 ms（**2.19 倍**） | 1.855 → 3.725 ms（**2.01 倍**） | 2.294 → 4.182 ms（**1.82 倍**） |
| release_threshold（案 A） | 0.0018 → 0.0018 ms（**1.00 倍**） | 0.175 → 0.181 ms（**1.03 倍**） | 0.577 → 0.595 ms（**1.03 倍**） |
| sync_alloc（案 B） | 1.610 → 1.683 ms（**1.05 倍**） | 1.859 → 1.936 ms（**1.04 倍**） | 2.300 → 2.389 ms（**1.04 倍**） |
| both（案 A+B） | 1.613 → 1.850 ms（**1.15 倍**） | 1.861 → 2.011 ms（**1.08 倍**） | 2.300 → 2.465 ms（**1.07 倍**） |

### 5.2 絶対値への効果（64 MiB。段差とは独立の追加所見）

baseline 比の median（release_threshold 条件）: P1 alloc_only は
3.5 ms → 0.0018 ms（**約 2000 分の 1**）、P2 alloc_zeros は 3.9 ms →
0.34 ms（**約 11 分の 1**）、P3 h2d_only は 4.65 ms → 1.15 ms（**約 4
分の 1**）。driver プールが解放せず保持し続けるため、同一サイズの
再確保が事実上キャッシュヒットになる。sync_alloc・both は 64 MiB では
baseline とほぼ同水準（段差解消は 32→33 MiB 帯に限定され、絶対値の
削減効果は release_threshold のみに現れる）。

### 5.3 pool 属性推移（`mem_pool_attr` 行）

release_threshold 条件適用直後は `release_threshold=18446744073709551615`
（`u64::MAX`）・`reserved_bytes` がサイズに応じて増加（driver が解放
せず保持）することを確認した。baseline・sync_alloc 条件では
`release_threshold=0`・`reserved_bytes=0`（都度解放）のまま。

### 5.4 降順パス（論点 (b)）

降順パス（24〜64 MiB の逆順）も完走した。P0/P4（ホスト側 D2H）計測は
release_threshold 条件下でも `slow_count`>0 が残存しており、#1146 の
「主因はホスト側 glibc mmap しきい値」という結論と整合する（driver
プール側の対策では解消しない）。降順走査限定のスパイクの有無・強度は
昇順パスと同水準で、案 A/B いずれによっても変化しなかった。

### 5.5 P7（`CudaMmaGemm` 本番経路レプリカ・dim4096）— 重大な反証

P7 は `upload_f16`→`alloc_output_f16`→`launch_f16`→`download_f16`
（`run_f16` と同一の呼び出し列）を dim4096（f16。A/B/C とも 32 MiB）で
5 回反復した実測（5 outer run・各 run は 20 回以上の warmup/計測）。
**§5.1〜§5.2 の孤立マイクロベンチマーク（P1/P2/P3 単体）の結論と正反対
の結果が出た**:

| 条件 | median (ms) | 対 baseline |
|---|---|---|
| baseline | 26.09〜28.33（5 run） | — |
| release_threshold（案 A） | 265.21〜284.88（5 run） | **約 10 倍悪化** |
| sync_alloc（案 B） | 27.04〜27.31（5 run） | ほぼ同水準 |
| both（案 A+B） | 20.13〜22.69（5 run） | やや改善 |
| baseline（再確認ブラケット） | 25.75〜27.91（5 run） | — |

checksum は全条件で完全一致（`-132270.738122`）しており数値結果への
影響はない。release_threshold **単独**では median が baseline の
約 10 倍（27 ms → 270 ms 台）に悪化し、全 5 run で再現した（ラン間
ばらつきではない）。

**解釈**: GB10 は unified memory（`docs/real-hardware-verification-env.md`
参照。CPU/GPU が同一物理メモリを共有）である。release threshold を
`u64::MAX` へ引き上げると driver プールが解放せず保持し続ける予約量が
反復のたびに増加し（§5.3 の `reserved_bytes` 増加）、これが #1146 の
「フレッシュ `Vec` 確保（glibc mmap しきい値超過）」問題と同じ物理
メモリプールを取り合う形になり、host 側の確保コストをかえって悪化させ
たと見られる。孤立した P1/P2/P3（デバイス確保のみ・ホスト側フレッシュ
確保を伴わない）ではこの相互作用が顕在化しないため、§5.1〜§5.2 の
「解消」評価は**現実の `run_f16` 呼び出しパターンには一般化できない**。

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

## 7. 判定

§2 の事前登録基準（段差比 1.2 未満＝解消・1.2〜1.8＝緩和）に照らし、
§5.1（孤立マイクロベンチマーク）と §5.5（現実的な `run_f16` 呼び出し
パターンのレプリカ）の両方を用いて判定する。**両者が矛盾する場合は
§5.5（実際の本番呼び出しパターンに近い方）を優先する**（マイクロ
ベンチマークは実環境の相互作用を捉えられないため。§5.5「解釈」参照）。

- **論点 (a)**: 孤立 P1/P2/P3 単体では release_threshold（案 A）が段差比
  1.00〜1.03 倍で解消基準を満たす（§5.1）。**しかし § 5.5 の P7（現実的
  な `run_f16` レプリカ）では release_threshold 単独が median を約 10
  倍悪化させており、単体マイクロベンチマークの「解消」評価は現実の
  呼び出しパターンに一般化できない**。sync_alloc（案 B）は P1/P2/P3
  でも P7 でも「ほぼ変化なし」で一貫している。both（案 A+B 併用）は
  P1/P2/P3 で 1.07〜1.15 倍（解消基準内）、P7 でも改善（baseline 比
  やや高速）と、唯一 P1〜P7 の全指標で一貫して非後退〜改善だった。
- **論点 (b)**: 降順パスでも release_threshold 条件下で `slow_count`>0
  が残存し、#1146 の「主因はホスト側 glibc mmap しきい値」という結論と
  整合する（driver プール側の単独対策では解消しない。§5.4）。
- **論点 (c)**: `docs/backend-cuda-pool-allocator-decision.md` §8 の
  保留事項（release threshold 調整可否）を判定確定した。**案 A の
  単独採用は見送る**（§5.5 の重大な反証）。

**採用判断（当初の ADOPT から変更）**: release threshold の単独引き上げ
（案 A）は `CudaAllocator::new` へ**結線しない**（一度実装したが、P7
実測後に差し戻した。`git log` 参照）。案 A は unified memory 環境
（GB10）でホスト側確保と物理メモリを取り合い、実際の `run_f16` 呼び出し
パターンを悪化させるため、孤立マイクロベンチマークでの「解消」評価を
そのまま本番結線の根拠にできないという教訓を残す。both（案 A+B 併用）
は良好だったが、A 単独が有害である以上、A+B を検証なしに採用するのは
安全側でない（B 単独でも P7 は改善しない＝A+B の改善が A 由来か B 由来
かの寄与分解が未実施）ため、案 B（`cuMemAlloc` 同期割当）単独・A+B 併用
とも本 PR では見送り、後続 Issue の検討対象として引き継ぐ（§8）。
イシュー #1153 の本来の目的（`gemm_mma.rs` の per-call 確保を
`SizeClassPool` 経由へ結線）は、driver 設定に依存しないアプリ層の
明示的な貸出・返却であり、本判定の対象外として別途 before/after 計測
する（`docs/perf/cuda-gemm-mma-f16-pool-wiring.md` 参照）。

## 8. スコープ外・引き継ぎ

- `crates/backend-cuda/src/pool.rs` 冒頭コメントの
  「既存の `stream.alloc`/`alloc_zeros` は同期 `cuMemAlloc` 系のまま」
  という誤記は、イシュー #1153（本 A/B の実測完了と同一セッション）で
  是正済み（`docs/backend-cuda-pool-allocator-decision.md` §2 の記述と
  整合させた）
- **both（案 A+B 併用）が P7 で改善した理由の寄与分解**（A 単独は
  悪化・B 単独は無変化なのに A+B が改善する理由が未解明。sync_alloc
  〈cuMemAlloc 同期割当〉が release threshold 引き上げによる unified
  memory 圧迫を何らかの形で相殺している可能性があるが未検証）
- 案 B（`cuMemAlloc` 同期割当）単独・A+B 併用の本番結線可否の再検討
  （cudarc 0.19.8 の `CudaSlice::Drop` API 不一致問題〈`has_async_alloc`
  の真偽で `free_async`／`free_sync` を自身で選ぶ〉があるため所有型の
  新設が前提。`docs/perf/cuda-gemm-mma-f16-pool-wiring.md` §8 参照）
- 32→33 MiB 段差の孤立マイクロベンチマークでの原因（driver プール
  トリム挙動）は特定できたが、P7 で release threshold 引き上げが
  悪化する経路（unified memory の物理メモリ競合と推定。§5.5「解釈」）
  の定量検証（GPU counters・メモリ帯域計測等）は未実施
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
| 1 | 案 A（release threshold 引き上げ）を GB10 実機で計測 | §5.1・§5.5（実測完了。孤立ベンチでは解消・現実的レプリカでは悪化という相反する結果） |
| 2 | 案 B（`cuMemAlloc` 同期割当）を GB10 実機で計測 | §5.1・§5.5（実測完了。両ベンチとも「ほぼ変化なし」） |
| 3 | なし／A／B／A+B の全組合せを計測 | §5.1・§5.5（4 条件とも計測完了） |
| 4 | 32 MiB 前後を中心に複数サイズで計測し中央値・ばらつき・二峰性を記録 | §5.1〜§5.4（8 サイズ・昇順／降順とも完走） |
| 5 | どの対策が病態を解消（緩和）するか結論を明記 | §7（判定確定。案 A 単独は不採用〈P7 で悪化〉・案 B／A+B は本 PR では見送り） |
| 6 | `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md` へ追記 | §7・§10 に本ドキュメントへの参照を追記済み |
| 7 | 対策コードを本番結線しない（計測専用） | 判定確定後、`crates/backend-cuda/src/pool.rs` への release threshold 結線は実装後に P7 実測で差し戻し済み（本セクション上位の「採用判断」参照）。イシュー #1153 本来の `SizeClassPool` 結線は本判定と独立（`docs/perf/cuda-gemm-mma-f16-pool-wiring.md`） |
