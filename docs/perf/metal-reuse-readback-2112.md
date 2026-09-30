# Metal reuse 計測窓の readback 対処（イシュー #2112）

readback 宛先ポリシー（既定 OFF・env opt-in）と A/B 基盤の設計記録。**本 doc 時点で M4 Max
実測は未実施**（実装は Linux 環境。実測と ADOPT 時の既定値切替は Mac セッションへ申し送り。
#2104／PR #2448 のホスト arena と同じ分業）。

## 1. 背景

- `docs/perf/metal-readout-legacy-regression-four-arm-diag.md`（#1696）の 4 腕診断（プロセス分離
  single-arm・5 起動）: keep-alive の宛先 `Vec` が毎回 first-touch ページのとき `readback` が伸びる
  （N=1024／2048／4096 で BorrowedKeepAlive − LegacyToVec = +0.20／+0.70／+3.74 ms）。
  PretouchedReusedDest（既タッチ宛先の再利用）では legacy 水準へ戻る。機構の帰属は**仮説段階**（M1）。
- Issue タイトルの「N=4096 で +12 ms」は出典が確認できない。この値は `docs/perf/loss-attribution-matrix.md`
  にしか現れず、#1696 の Layer B とも `metal-gemm-reuse-phase-breakdown.md` の phase 内訳とも一致しない。
  **本作業ではこの値を基準にしない**。before は同一 A/B ラン内の `fresh` 腕の実測とする。

## 2. 候補の検討

| 候補 | 判断 |
|---|---|
| (a) 宛先の再利用（#1696 PretouchedReusedDest の本番化） | 不採用。reuse 窓の matmul 出力は tape に保持され続け（`run_gemm_reuse`）、再利用できる解放済み宛先がない。legacy の 2 本目 `to_vec` は bench 側でライブラリ外。HOST_ARENA（別系統）とも結合する |
| (b) 事前タッチ宛先（CUDA #1437 の Metal 版） | 不採用。UMA の memcpy は CPU 書き込みなので fault が fill へ移るだけで計測窓に残る（#1695 が移植しなかった判断を踏襲） |
| (c) ゼロコピー（Tensor 実体を MTLBuffer に置く） | 本 PR 対象外。tensor-core のストレージ抽象変更と新規 `unsafe` が必要で、ユーザー承認事項 |
| (d) fresh 宛先への分割並列コピー | **採用（opt-in・既定 OFF）**。派生仮説 H-par |

## 3. 実装

- `crates/backend-metal/src/readback_policy.rs`（cfg 非依存・純ロジック・`unsafe` なし・新規依存なし）
  - env `FANDHE_AI_METAL_READBACK_DEST` = `fresh`（既定）／`parallel`。完全一致のみ受理、未知値は既定へ倒す（値はエコーしない）
  - `PARALLEL_READBACK_MIN_BYTES = 8 MiB`（N=1024 の 4 MiB は対象外、N=2048 以上）／`PARALLEL_READBACK_MAX_THREADS = 8`
  - Fresh は `src.to_vec()` と同一。ParallelChunked は `vec![0.0; n]`（calloc）を chunk 分割し `std::thread::scope` + `Builder::spawn_scoped` で並列 `copy_from_slice`。spawn 失敗 chunk は呼び出し元で逐次コピー。全要素上書きのため bit 同一・古い内容の露出なし
- `buffer.rs::MetalBuffer::read_to_vec` を `copy_to_vec(unsafe { as_host_slice() })` に変更。gemm.rs・memory.rs（`download_inner`）が共通に通る 1 箇所。`unsafe` ブロック数は不変（借用の生存範囲がホストコピー呼び出しの間に延びるのみ。同期済み・借用中に GPU 書き込みを積まない契約は SAFETY コメントに根拠を記載。スコープ付きスレッドは scope 終了前に全 join）

### 期待値（実測前に表明）

H-par は「並列化で first-touch fault 処理が速くなる」という #1696 が**検証していない**仮説。legacy 既定の
判定セルでは `read_to_vec` 宛先は既タッチで readback は N=4096 で約 1 ms なので、見込める改善は
memcpy 帯域の並列化分（1 ms 未満）にとどまり、スレッド生成の固定費で後退しうる。ADOPT の見込みは高くない。
REJECT／undetermined でも機構（既定 OFF）と記録を残す（#2104 と同じ扱い）。

## 4. A/B 手順と判定規則

- `scripts/bench/framework-compare/run_ab_readback_metal.sh`（同一バイナリの env 切替。5 round・round ごとに起動順反転・プロセス独立起動・専有ゲート既定 `exclusive`）
- 判定規則は実測前固定の `docs/perf/logs/metal-reuse-readback-2112/RULE.txt` が正。手順は同ディレクトリの `README.md`

## 5. 実測記入欄（Mac セッション）

| セル | fresh 中央値 | parallel 中央値 | ratio | checksum |
|---|---|---|---|---|
| gemm reuse N=1024 | 未計測 | 未計測 | - | - |
| gemm reuse N=4096 | 未計測 | 未計測 | - | - |
| infer reuse | 未計測 | 未計測 | - | - |

## 6. 変更しないもの

`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・`docs/spec/`・facade 公開面・bench-fandhe 本体。
ADOPT 時の既定値切替（`READBACK_DEST_DEFAULT`）は後続 PR。スコープ外: ゼロコピー化、legacy が遅い根本原因。
