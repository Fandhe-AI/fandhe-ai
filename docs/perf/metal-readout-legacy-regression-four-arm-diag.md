# Metal readout legacy 後退の 4 腕診断（イシュー #1696）

## 0. 位置づけ（本 PR 時点では未実測）

`crates/backend-metal/src/readout_regression_diag_tests_1695.rs`（イシュー
#1695。PR #1806 でマージ済み）が実装した 4 腕診断ハーネスを M4 Max 実機で
実行し、機構の切り分け結果を記録するのが本ドキュメントの役割である。

**本 PR を書いた実行環境は Linux x86_64 であり Apple Silicon 実機への到達
手段がない**（`docs/real-hardware-verification-env.md` は Mac を「ローカル
直接実行」と定義しており、本エージェント実行環境はこれに該当しない）。この
ため本ドキュメントは実測プロトコル・事前登録判定規則・集計手段のみを整備し、
実測値（§6・§7・§8・§10 の記入欄）は Mac セッションへの申し送りとして
空欄のまま残す。診断専用イシューにつき、採否判定（ADOPT／REJECT）・本番
結線変更・tolerance／baseline 変更は一切行わない（#1520 の REJECT〈legacy
フォールバック維持〉は本ドキュメントによって変更されない）。

実測手順は `docs/perf/logs/metal-readout-regression-diag-1696/README.md`
を参照。

## 1. 背景

- #1520（`docs/perf/metal-gemm-candle-gate-remeasurement.md` §17）:
  bench-fandhe Metal `--readout borrowed`（`Var::host_view()` 借用）が
  `--readout legacy`（`to_tensor()` + `.to_vec()`）に対し N=1024
  fresh/reuse で 1.21／1.11 倍後退し REJECT（legacy フォールバック維持）。
- #1574 低レイヤー診断（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §5）:
  N=1024 Metal reuse の legacy→borrowed 5 回 interleave で `matmul` 区間
  1.694→2.161 ms（Δ≈+0.47 ms）、`host_copy` 0.233→0.000 ms、checksum は
  同一、`iter_total` 2.470→2.677 ms。後退は **`matmul` 区間（GPU 待ち＋
  ホストへの読み出し）に閉じ、機構は未特定**のまま残っている。
- CUDA 側の同型診断 #1436（`docs/perf/cuda-host-view-readout-small-shape-
  regression.md`）は「glibc の動的 mmap 閾値適応」を有力仮説としたが、
  同一プロセス内逐次実行の交絡により単独確証には至っていない（同ドキュメント
  §0・§8・§11）。
- 本ドキュメントは、その機構を切り分けるために #1695 が実装した 4 腕
  診断ハーネスの実測プロトコル・判定規則を整備し、実測後の機構記録の
  記入欄を用意する（実測自体は本イシュー #1696 の兄弟イシュー #1695 が
  実装したハーネスを M4 Max 実機で走らせる別セッションへ引き継ぐ）。

## 2. 経路の整理

Metal は UMA（統合メモリ）のため CUDA の D2H（`clone_dtoh`／
`memcpy_dtoh`）に相当する明示転送は存在せず、`MetalBuffer::read_to_vec()`
（`contents()` からの memcpy。`crates/backend-metal/src/buffer.rs`）が
その対応物になる（`docs/perf/metal-gemm-reuse-phase-breakdown.md` の
`readback` 区間定義と同じ整理）。

| 項目 | 実体 |
| --- | --- |
| 本番 reuse の `matmul` 区間 | `Var::matmul` → `MetalBackendOps::gemm`（`ops.rs`）→ `MetalGemm::dispatch_auto` → `alloc_uninit_pooled(C)` → encode → `synchronize`（commit + `waitUntilCompleted`）→ `read_to_vec()` → `Tensor::new` |
| legacy readout（bench-fandhe off 腕） | `to_tensor()`（`Arc` 複製）→ `.to_vec()` で 2 本目確保 → 反復末尾で free |
| borrowed readout（bench-fandhe on 腕。#1438 で既定経路） | `host_view()`（`Arc` 複製のみ）。追加確保・free なし |
| 両者の唯一の実質差 | `read_to_vec` が返す 1 本目を「2 本目へ複製して free するか」「そのまま keep-alive するか」 |

4 腕（`crate::readout_regression_diag_arms::ReadoutArm`。定義は
`crates/backend-metal/src/readout_regression_diag_arms.rs` を正とし、
本節では要約のみ記す）:

- `LegacyToVec`: legacy readout の再現（1 本目 keep-alive・2 本目確保して
  即 drop）
- `BorrowedKeepAlive`: borrowed readout の再現（追加確保・free なし）
- `BorrowedWithDummyAllocFree`: `BorrowedKeepAlive` と同じ読み出しに加え、
  読み出し後に同サイズ `Vec` を確保・全要素書き込み・即 drop する（「2 本目
  のコピー」と「アロケータへの free 副作用」を分離する対照腕）
- `PretouchedReusedDest`: 計測外で 1 回だけ確保し事前タッチ済みの宛先へ
  `MetalBuffer::read_into_slice`（確保を伴わない memcpy）する

CUDA 版の 5 腕目 `PretouchedFreshDest`（CUDA 本番の是正
`ReadbackDest::PretouchedFresh`〈#1437〉に対応する腕）は、Metal 側に
対応する本番機構が存在しないため #1695 では移植していない（意図的な
スコープ判断。#1695 冒頭コメント参照）。

## 3. 仮説と事前予測

以下は計測前に固定する中立表現の仮説であり、確定した機構主張ではない
（#1695 冒頭コメント「機構仮説は中立表現に留める」を踏襲する）。

| ID | 仮説 | Layer B での予測 |
|----|------|------------------|
| M1（主） | keep-alive により `read_to_vec` の宛先が毎回 first-touch ページとなり memcpy 中のページ確保が `readback` に乗る（legacy は free-and-reuse で既タッチページを再利用する） | `BorrowedKeepAlive` の `readback` が `LegacyToVec` より遅く、`PretouchedReusedDest` で回復する。`BorrowedWithDummyAllocFree` の挙動で「free 副作用」と「宛先の既タッチ性」を分離できる |
| M2 | GPU 書き込み後の CPU 初回読み出しにおけるキャッシュ／コヒーレンス状態 | 増分が `host_read` に出て `readback` は不変 |
| M3 | 後退がハーネスの計測範囲外（encode／`synchronize`／`alloc_uninit_pooled`）にある | 4 腕とも `readback`＋`host_read` の差が §1 の Δ≈0.47 ms（`matmul` 区間差）を説明しない |
| M4 | 共有負荷ノイズ（#1548 で N=1024 差分がノイズ帯だった前例がある） | 腕差が起動間 spread（ノイズ床）以下 |

「glibc `M_MMAP_THRESHOLD` 動的適応」は glibc 固有の機構であり、macOS
libmalloc へそのまま当てはめる根拠はない。macOS 側のアロケータ環境変数
実験（`man malloc` に文書化された変数があれば任意で試す・なければスコープ
外として記録する）は §11 のスコープ外整理を参照。

## 4. 環境・プロトコル

- 対象コマンド:
  - 主系列（単一腕・単一サイズ。プロセス分離）:
    ```
    cargo test -p fandhe-ai-backend-metal --release --lib \
      readout_regression_diag_tests_1695::readout_regression_diag_n<N>_<arm> \
      -- --ignored --nocapture --test-threads=1 --exact
    ```
    `<N>` は `1024`／`2048`／`4096`、`<arm>` は `legacy_to_vec`／
    `borrowed_keep_alive`／`borrowed_with_dummy_alloc_free`／
    `pretouched_reused_dest`。**完全修飾名（モジュール名込み）＋`--exact`
    を必ず併用する**（#1436 README「再現手順」節と同じ注意。モジュール名
    を省いた部分一致では単一腕関数と一括腕関数の両方に一致し、同一
    プロセス内でまとめて実行されてしまいアロケータ状態の交絡が生じる）。
  - 副系列（in-process 4 腕。状態引き継ぎ検出用）:
    ```
    cargo test -p fandhe-ai-backend-metal --release --lib \
      readout_regression_diag_tests_1695::readout_regression_diag_n<N> \
      -- --ignored --nocapture --test-threads=1 --exact
    ```
- release ビルド必須（debug ビルドは GEMM 自体の絶対値が意味を持たない
  ほど遅くなる。`docs/perf/metal-dinput-sync-1562/orchestrate.sh` と同じ
  注意）
- `--test-threads=1` 必須（同一 GPU 上の競合を避ける）
- `record_only` 運用（#1519 のユーザー指示「Metal は現在の環境で測れる値
  で可」を踏襲）。専有ゲートは課さず、load average を記録するのみに留める

実行手順・生成物一覧は `docs/perf/logs/metal-readout-regression-diag-1696/
README.md` を正とする。

## 5. 事前登録判定規則（record only・non-gating。計測後に緩和しない）

1. 主系列: 単一腕×単一サイズ 12 テストを**各 5 プロセス起動**し、各起動の
   テスト内中央値（20 warmup＋20 測定。ハーネス既定）の 5 起動中央値を
   代表値とする。プロセス分離により #1442 の交絡（同一プロセス内の状態
   引き継ぎ）を排除する。
2. 副系列: `*_n{1024,2048,4096}`（in-process 4 腕）を各 1 起動し、主系列
   との差で状態引き継ぎの有無を記録する（判定には使わない・§7 の記入欄）。
3. ノイズ床: 腕ごとの 5 起動 `readback` 中央値の (max−min)。腕差が
   いずれかの腕のノイズ床未満なら「判定不能（ノイズ帯）」と記す。
4. 規模照合: N=1024 で `BorrowedKeepAlive − LegacyToVec` の `readback`＋
   `host_read` 差が §1 の Δ≈0.47 ms（`matmul` 区間差）と同オーダー（目安
   0.5〜2 倍）か確認する。同オーダーなら M1 系の切り分けへ進み、ほぼ 0
   なら M3（ハーネス計測範囲外）の可能性を示すと解釈し、無理に機構を
   帰属させない。
5. 全腕・全起動で checksum が参照値（腕間で共通に求めた
   `reference_checksum`）と一致すること（ハーネスの `assert!`。REQ-2 とは
   無関係の sanity）。不一致があれば当該起動は無効として記録する。
6. `record_only` 運用（専有ゲートは課さない）。`uptime` の load average を
   記録する。
7. 内部ホスト名・内部パスをいかなる生成物にも含めない。

## 6. Layer B（単一腕・単一サイズ。5 プロセス起動）実測記入欄

**未実測**（本 PR 実行環境に Apple Silicon 実機なし）。

| N | 腕 | readback median (ms) ×5 run | host_read median (ms) ×5 run | ノイズ床 (readback max−min) | checksum 一致 |
|---|----|------------------------------|-------------------------------|------------------------------|----------------|
| 1024 | LegacyToVec | — | — | — | — |
| 1024 | BorrowedKeepAlive | — | — | — | — |
| 1024 | BorrowedWithDummyAllocFree | — | — | — | — |
| 1024 | PretouchedReusedDest | — | — | — | — |
| 2048 | LegacyToVec | — | — | — | — |
| 2048 | BorrowedKeepAlive | — | — | — | — |
| 2048 | BorrowedWithDummyAllocFree | — | — | — | — |
| 2048 | PretouchedReusedDest | — | — | — | — |
| 4096 | LegacyToVec | — | — | — | — |
| 4096 | BorrowedKeepAlive | — | — | — | — |
| 4096 | BorrowedWithDummyAllocFree | — | — | — | — |
| 4096 | PretouchedReusedDest | — | — | — | — |

## 7. in-process 4 腕（副系列。状態引き継ぎ検出）実測記入欄

**未実測**。主系列（§6）との差異（あれば）をここに記す。差異がなければ
「主系列と同水準・状態引き継ぎの影響は検出されず」と記す。

## 8. Layer A 参考（同日 Δ 再確認）記入欄

任意。`docs/perf/lowlayer-diagnosis-2026-09-12.md` §5 の Δ≈0.47 ms が
同日再現するか、`bench-fandhe --task gemm --device metal --size 1024
--mode reuse --readout <legacy|borrowed> --phases` の 1〜数回起動で
参考値を記録する（判定には使わない）。

**未実測**。

## 9. CUDA #1436 対比

- 規模: CUDA は D2H で 7〜30 倍の後退（#1436 §0）、Metal は総計で
  1.11〜1.21 倍（#1520）。オーダーが異なる。
- 機構: CUDA は明示 DMA 転送＋ページフォールトが関与しうるのに対し、
  Metal は UMA 上の `contents()` memcpy であり、転送そのものより
  ページテーブル・アロケータの挙動に起因する可能性が相対的に高いと
  推測される（仮説段階）。
- `BorrowedWithDummyAllocFree` 腕の CUDA 側 #1436 §6 注記（旧実装の
  ゼロ初期化に関する参考値）は Metal 側には適用されない（#1695 実装は
  最初から非ゼロ書き込み＋`std::hint::black_box` で最適化除去を防止済み）。
- CUDA の確証実験（`MALLOC_MMAP_THRESHOLD_` 固定）に対応する macOS 側の
  ノブは、本ドキュメントでは推定で記述しない（§11 参照）。

## 10. 機構の切り分け結果（記入欄）

**未実測のため記入なし**。実測後、§5 の規則 1〜7 に従い次を埋める:

- どの仮説（M1〜M4）が Layer B の実測と整合するか
- 規則 4（規模照合）の結果と、それに基づく機構の帰属（確定できない場合は
  「仮説段階」と明記し、次の診断（例: 計測区間を `synchronize`／pooled
  alloc まで拡張する）へ引き継ぐ）
- ノイズ床との比較結果（規則 3）

## 11. スコープ外

- `synchronize`／encode 区間を含む拡張計測（本ハーネスは `readback`／
  `host_read` のみを計測し、`matmul` 区間全体〈encode＋`synchronize` 含む〉
  とは範囲が異なる。§5 規則 4 で「ハーネス範囲外」の可能性を扱うに留める）
- `alloc_uninit_pooled` の影響
- N=2048 の bimodal 相当の有無（CUDA #1436 で観測された二峰性の Metal 版
  確認）
- GPU タイムスタンプ（`internal-diagnostics` feature）併用による
  `readback` 内訳の GPU 側／host 側分離
- macOS libmalloc の環境変数実験（`man malloc` に文書化された変数の
  有無・効果）

いずれも実測後に必要と判断されれば、`out-of-scope-tracking.md` の規約に
従いユーザー承認のうえ別イシューへ切り出す。

## 12. 参照

- `crates/backend-metal/src/readout_regression_diag_arms.rs`（4 腕定義・
  純関数ヘルパ。イシュー #1695）
- `crates/backend-metal/src/readout_regression_diag_tests_1695.rs`（診断
  テスト本体。イシュー #1695）
- `docs/perf/metal-gemm-candle-gate-remeasurement.md` §17（#1520 の REJECT
  確定記録）
- `docs/perf/lowlayer-diagnosis-2026-09-12.md` §5（Δ≈0.47 ms の局所化）
- `docs/perf/cuda-host-view-readout-small-shape-regression.md`（CUDA 側
  同型診断 #1436）
- `docs/perf/logs/metal-readout-regression-diag-1696/`（本イシューの
  実測ログ置き場・`README.md` に実行手順）
- 親イシュー #1574 → #1588（草案）→ 本イシュー #1696（兄弟イシュー #1695
  がハーネス実装）
