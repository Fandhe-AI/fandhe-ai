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

**2026-09-16 実測済み** → Mac セッションで M4 Max 実機（origin/main
565300e4・release ビルド・record_only・共有負荷下）にて `orchestrate.sh` を
実行し、§6・§7・§8・§10 の記入欄を `aggregate.md` の実値で埋めた（生成物は
上記ログ置き場。環境は同ディレクトリ `env_info.txt`）。上記の「未実測」
「申し送り」の記述は経緯として残す。診断専用の位置づけ（採否判定・本番
結線変更・tolerance／baseline 変更を行わない）は実測後も不変。

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
| M3 | 後退がハーネスの計測範囲外（encode／`synchronize`／`alloc_uninit_pooled`）にある | 4 腕とも `readback`（`matmul` 区間に対応する内訳。規則 4 参照）の差が §1 の Δ≈0.47 ms（`matmul` 区間差）を説明しない |
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
4. 規模照合: N=1024 で `BorrowedKeepAlive − LegacyToVec` の **`readback`
   のみ**の差（§2 の対応表のとおり `readback` が本番 `matmul` 区間の
   GPU 待ち＋ホストへの読み出しに対応する内訳であり、`host_read` は
   legacy の 2 本目確保・コピー等 `matmul` 区間の外側で発生する追加
   コストのため合算しない）を §1 の Δ≈0.47 ms（`matmul` 区間差）と
   同オーダー（目安 0.5〜2 倍）か確認する。同オーダーなら M1 系の切り
   分けへ進み、ほぼ 0 なら M3（ハーネス計測範囲外）の可能性を示すと
   解釈し、無理に機構を帰属させない。`host_read` の差は参考情報として
   別途報告するに留め、規模照合の判定には用いない。全起動で checksum
   が参照値と一致した run のみを対象とする（規則 5。不一致の起動は
   規模照合からも除外し、有効な run が 5 件揃わない場合は「判定不能」
   とする）。
5. 全腕・全起動で checksum が参照値（腕間で共通に求めた
   `reference_checksum`）と一致すること（ハーネスの `assert!`。REQ-2 とは
   無関係の sanity）。不一致があれば当該起動は無効として記録する。
6. `record_only` 運用（専有ゲートは課さない）。`uptime` の load average を
   記録する。
7. 内部ホスト名・内部パスをいかなる生成物にも含めない。

## 6. Layer B（単一腕・単一サイズ。5 プロセス起動）実測記入欄

**未実測**（本 PR 実行環境に Apple Silicon 実機なし）→ **2026-09-16
実測済み**（Apple M4 Max・論理 CPU 16・macOS 26.6.2・rustc 1.98.1・
origin/main 565300e4・release ビルド。record_only・専有ゲートなし。実行中
の load average〈1 分〉は 19.57 → 14.00 → 18.53 → 17.73〈5 分 20.46〜
22.03〉で、別セッションの workspace 全体テストが並走する**共有負荷下**。
`pmset -g therm` に thermal／performance warning の記録なし。出典:
`docs/perf/logs/metal-readout-regression-diag-1696/aggregate.md`・
`uptime_sampler.log`・`env_info.txt`）。

各セルは 5 プロセス起動の各起動テスト内中央値（20 warmup＋20 測定）の
5 起動中央値。ノイズ床は同 5 起動 `readback` 中央値の max−min（規則 3）。
checksum は全腕・全起動（主系列 60 起動＋副系列 3 起動）で N ごとに同一値
（N=1024: 4627.380087・N=2048: −59111.364509・N=4096: 66291.493131）を
確認し、規則 5 で除外した起動はない（有効 run は全セル 5 件）。

| N | 腕 | readback median (ms) ×5 run | host_read median (ms) ×5 run | ノイズ床 (readback max−min) | checksum 一致 |
|---|----|------------------------------|-------------------------------|------------------------------|----------------|
| 1024 | LegacyToVec | 0.0552 | 0.8450 | 0.0294 | OK（5/5） |
| 1024 | BorrowedKeepAlive | 0.2566 | 0.5887 | 0.0155 | OK（5/5） |
| 1024 | BorrowedWithDummyAllocFree | 0.0573 | 1.4460 | 0.0328 | OK（5/5） |
| 1024 | PretouchedReusedDest | 0.0536 | 0.8148 | 0.0026 | OK（5/5） |
| 2048 | LegacyToVec | 0.3030 | 3.4332 | 0.1225 | OK（5/5） |
| 2048 | BorrowedKeepAlive | 0.9989 | 2.4290 | 0.0458 | OK（5/5） |
| 2048 | BorrowedWithDummyAllocFree | 0.5000 | 5.7405 | 0.1216 | OK（5/5） |
| 2048 | PretouchedReusedDest | 0.2400 | 3.3443 | 0.0159 | OK（5/5） |
| 4096 | LegacyToVec | 1.0155 | 14.3533 | 0.1179 | OK（5/5） |
| 4096 | BorrowedKeepAlive | 4.7529 | 13.2079 | 3.4180 | OK（5/5） |
| 4096 | BorrowedWithDummyAllocFree | 2.1807 | 23.3745 | 1.3505 | OK（5/5） |
| 4096 | PretouchedReusedDest | 1.7036 | 24.8725 | 2.1348 | OK（5/5） |

### 6.1 腕差（`BorrowedKeepAlive − LegacyToVec`。`aggregate.md`「腕差」節の転記）

| N | readback 差（規模照合対象） | host_read 差（参考。規模照合対象外） |
|---|------------------------------|---------------------------------------|
| 1024 | 0.2566 − 0.0552 = **+0.2014 ms** | 0.5887 − 0.8450 = −0.2563 ms |
| 2048 | 0.9989 − 0.3030 = **+0.6959 ms** | 2.4290 − 3.4332 = −1.0042 ms |
| 4096 | 4.7529 − 1.0155 = **+3.7374 ms** | 13.2079 − 14.3533 = −1.1454 ms |

### 6.2 N=4096 の起動間ばらつき（事実の記録）

N=4096 は 5 起動の中に同一ログ上で `readback`／`host_read` がともに突出
する起動が含まれる（`layerB-n4096-borrowed_keep_alive-run3`: readback
7.5280 ms／host_read 40.7345 ms、`layerB-n4096-pretouched_reused_dest-
run3`: 3.3039／38.5740 ms、`layerB-n4096-borrowed_with_dummy_alloc_free-
run1`: 3.1570／36.3960 ms。`LegacyToVec` は 5 起動とも readback 0.99〜
1.11 ms／host_read 14.27〜15.36 ms）。これらにより N=4096 の
`BorrowedKeepAlive`／`PretouchedReusedDest`／`BorrowedWithDummyAllocFree`
のノイズ床（3.4180／2.1348／1.3505 ms）は N=1024／2048 より 1〜2 桁大きい。
共有負荷（load average 14〜22）下の計測であり、突出起動の原因は本記録では
特定しない。

## 7. in-process 4 腕（副系列。状態引き継ぎ検出）実測記入欄

**未実測**。主系列（§6）との差異（あれば）をここに記す。差異がなければ
「主系列と同水準・状態引き継ぎの影響は検出されず」と記す。

**2026-09-16 実測済み** → 各 N を 1 起動（`inprocess-n{1024,2048,4096}-
run1.log`。同一プロセス内で LegacyToVec → BorrowedKeepAlive →
BorrowedWithDummyAllocFree → PretouchedReusedDest の順に実行）。**主系列
との差異あり**。判定には使わない（規則 2）。

| N | 腕 | in-process readback / host_read (ms) | 主系列 §6 readback / host_read (ms) |
|---|----|--------------------------------------|--------------------------------------|
| 1024 | LegacyToVec | 0.0744 / 1.1267 | 0.0552 / 0.8450 |
| 1024 | BorrowedKeepAlive | 0.1387 / 0.8018 | 0.2566 / 0.5887 |
| 1024 | BorrowedWithDummyAllocFree | 0.0732 / 1.6962 | 0.0573 / 1.4460 |
| 1024 | PretouchedReusedDest | 0.0667 / 0.8125 | 0.0536 / 0.8148 |
| 2048 | LegacyToVec | 0.3640 / 3.6640 | 0.3030 / 3.4332 |
| 2048 | BorrowedKeepAlive | 0.2751 / 2.3443 | 0.9989 / 2.4290 |
| 2048 | BorrowedWithDummyAllocFree | 0.2316 / 4.9971 | 0.5000 / 5.7405 |
| 2048 | PretouchedReusedDest | 0.2464 / 2.7692 | 0.2400 / 3.3443 |
| 4096 | LegacyToVec | 4.9619 / 47.0893 | 1.0155 / 14.3533 |
| 4096 | BorrowedKeepAlive | 3.5209 / 24.3905 | 4.7529 / 13.2079 |
| 4096 | BorrowedWithDummyAllocFree | 1.1843 / 26.0715 | 2.1807 / 23.3745 |
| 4096 | PretouchedReusedDest | 1.0570 / 12.0153 | 1.7036 / 24.8725 |

観測事実:

- N=1024: in-process の `BorrowedKeepAlive` readback（0.1387 ms）は主系列
  （0.2566 ms）より小さいが、同一プロセス内の `LegacyToVec`（0.0744 ms）
  より大きい向きは主系列と同じ。
- N=2048: in-process では `BorrowedKeepAlive` readback（0.2751 ms）が
  `LegacyToVec`（0.3640 ms）より**小さく**、主系列（0.9989 対 0.3030 ms）
  と**向きが逆転**する。
- N=4096: in-process の先頭腕 `LegacyToVec`（readback 4.9619／host_read
  47.0893 ms）が主系列（1.0155／14.3533 ms）の約 4.9／3.3 倍。1 起動のみの
  ため共有負荷の影響と状態引き継ぎの影響は分離できない。
- 以上は「同一プロセス内の実行順序・アロケータ状態の引き継ぎが腕差の向き・
  大きさを変えうる」という副系列の目的（#1442 の交絡検出）と整合する
  観測であり、主系列（プロセス分離）を代表値とする規則 1 の前提を支持する
  方向の事実である。機構の帰属は行わない。

## 8. Layer A 参考（同日 Δ 再確認）記入欄

任意。`docs/perf/lowlayer-diagnosis-2026-09-12.md` §5 の Δ≈0.47 ms が
同日再現するか、`bench-fandhe --task gemm --device metal --size 1024
--mode reuse --readout <legacy|borrowed> --phases` の 1〜数回起動で
参考値を記録する（判定には使わない）。

**未実測**。→ **2026-09-16 未実施**（任意項目。`orchestrate.sh` の対象外
のため同日の bench-fandhe 起動は行っていない）。規則 4 の比較対象 Δ≈
0.47 ms は `docs/perf/lowlayer-diagnosis-2026-09-12.md` §5（2026-09-12・
別日・別負荷条件）の値をそのまま用いる。

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

**未実測のため記入なし** → **2026-09-16 実測済み（§10.1〜§10.4 に記入）**。
実測後、§5 の規則 1〜7 に従い次を埋める:

- どの仮説（M1〜M4）が Layer B の実測と整合するか
- 規則 4（規模照合）の結果と、それに基づく機構の帰属（確定できない場合は
  「仮説段階」と明記し、次の診断（例: 計測区間を `synchronize`／pooled
  alloc まで拡張する）へ引き継ぐ）
- ノイズ床との比較結果（規則 3）

### 10.1 規則 3（ノイズ床）との比較

`readback` 中央値の腕差と、差に関わる両腕のノイズ床（§6）との比較。
「超過」は差の絶対値が両腕のノイズ床のいずれをも上回ること、「ノイズ帯」は
いずれかのノイズ床未満であることを指す。

| N | 腕差（readback） | 差 (ms) | 両腕のノイズ床 (ms) | 結果 |
|---|------------------|---------|---------------------|------|
| 1024 | BorrowedKeepAlive − LegacyToVec | +0.2014 | 0.0155／0.0294 | 超過 |
| 1024 | BorrowedWithDummyAllocFree − LegacyToVec | +0.0021 | 0.0328／0.0294 | ノイズ帯 |
| 1024 | PretouchedReusedDest − LegacyToVec | −0.0016 | 0.0026／0.0294 | ノイズ帯 |
| 1024 | BorrowedKeepAlive − PretouchedReusedDest | +0.2030 | 0.0155／0.0026 | 超過 |
| 1024 | BorrowedKeepAlive − BorrowedWithDummyAllocFree | +0.1993 | 0.0155／0.0328 | 超過 |
| 2048 | BorrowedKeepAlive − LegacyToVec | +0.6959 | 0.0458／0.1225 | 超過 |
| 2048 | BorrowedWithDummyAllocFree − LegacyToVec | +0.1970 | 0.1216／0.1225 | 超過（僅差） |
| 2048 | PretouchedReusedDest − LegacyToVec | −0.0630 | 0.0159／0.1225 | ノイズ帯 |
| 2048 | BorrowedKeepAlive − PretouchedReusedDest | +0.7589 | 0.0458／0.0159 | 超過 |
| 2048 | BorrowedKeepAlive − BorrowedWithDummyAllocFree | +0.4989 | 0.0458／0.1216 | 超過 |
| 4096 | BorrowedKeepAlive − LegacyToVec | +3.7374 | 3.4180／0.1179 | 超過（僅差。差 3.7374 対 `BorrowedKeepAlive` 床 3.4180） |
| 4096 | BorrowedWithDummyAllocFree − LegacyToVec | +1.1652 | 1.3505／0.1179 | ノイズ帯 |
| 4096 | PretouchedReusedDest − LegacyToVec | +0.6881 | 2.1348／0.1179 | ノイズ帯 |
| 4096 | BorrowedKeepAlive − PretouchedReusedDest | +3.0493 | 3.4180／2.1348 | ノイズ帯 |
| 4096 | BorrowedKeepAlive − BorrowedWithDummyAllocFree | +2.5722 | 3.4180／1.3505 | ノイズ帯 |

N=1024／2048 では `BorrowedKeepAlive` と他 3 腕との差がノイズ床を明確に
超過する。N=4096 は §6.2 の突出起動によりノイズ床自体が大きく、
`BorrowedKeepAlive − LegacyToVec` 以外は「判定不能（ノイズ帯）」。

### 10.2 規則 4（規模照合）の結果

- N=1024 の `BorrowedKeepAlive − LegacyToVec` の `readback` 差は
  **+0.2014 ms**（両腕のノイズ床 0.0155／0.0294 ms を超過）。§1 の Δ≈
  0.47 ms（2026-09-12・bench-fandhe `matmul` 区間差）に対する比は
  0.2014 / 0.47 ≈ **0.43 倍**。
- 規則 4 の事前登録目安「同オーダー（0.5〜2 倍）」の**下限 0.5 倍をわずか
  に下回る**一方、「ほぼ 0」でもない。したがって規則 4 の 2 分岐（同オー
  ダー → M1 系の切り分けへ／ほぼ 0 → M3 を示す）の**いずれにも明確には
  該当しない**（**要確認**: 目安の扱い）。規則どおり無理に機構を帰属させ
  ず、次のとおり事実のみ記す:
  - `readback` 差はノイズ床を超える実在の差であり、Δ≈0.47 ms の約 4 割
    強に相当する規模を持つ（M3「全腕とも差が Δ を説明しない」とは不整合）。
  - 残り約 6 割弱（≈0.27 ms）は本ハーネスの計測範囲（`readback`／
    `host_read`）では説明されず、§11 の拡張計測（encode／`synchronize`／
    `alloc_uninit_pooled` を含む区間）へ引き継ぐ候補として記録する（M3 と
    部分的に整合）。
  - Δ≈0.47 ms は別日（2026-09-12）・別負荷条件・別ハーネス（bench-fandhe
    facade 経由）の値であり、同日 Δ 再確認（§8）は未実施のため、比 0.43
    倍自体に負荷差・経路差の交絡が含まれうる。
- `host_read` 差（参考・規模照合対象外）は全 N で `BorrowedKeepAlive` が
  `LegacyToVec` より小さい（−0.2563／−1.0042／−1.1454 ms）。`LegacyToVec`
  の `host_read` が 2 本目確保＋コピーを含むという §2 の経路整理と整合する。

### 10.3 仮説 M1〜M4 との整合（事実の対応付け。機構の断定はしない）

| ID | Layer B の観測 | 整合性 |
|----|----------------|--------|
| M1（主） | 全 N で `BorrowedKeepAlive` の `readback` が `LegacyToVec` より大きい向き（+0.2014／+0.6959／+3.7374 ms）。`PretouchedReusedDest` は `LegacyToVec` と同水準（1024: −0.0016・2048: −0.0630 ms はノイズ帯。4096: +0.6881 ms もノイズ帯）。`BorrowedWithDummyAllocFree` は N=1024 で `LegacyToVec` と同水準（+0.0021 ms・ノイズ帯）まで回復し、N=2048（0.5000 ms）／N=4096（2.1807 ms）では `LegacyToVec` と `BorrowedKeepAlive` の中間 | 予測（Borrowed が遅く Pretouched で回復）と**整合**。N=1024 では「読み出し後に同サイズを確保・書き込み・free する」対照腕が `readback` を legacy 水準へ戻す事実を示唆する（宛先の既タッチ性か free 副作用かの分離は N=2048／4096 の中間値により部分的にとどまる） |
| M2 | 増分は `readback` 側に出ており、`host_read` は全 N で `BorrowedKeepAlive` の方が小さい | 予測（増分が `host_read`・`readback` 不変）と**不整合** |
| M3 | N=1024 の `readback` 差 0.2014 ms は Δ≈0.47 ms の約 0.43 倍（§10.2） | 「差が Δ を説明しない」とは不整合だが、Δ の約 6 割弱はハーネス範囲外に残るため**部分的に整合**（仮説段階） |
| M4 | N=1024／2048 は腕差がノイズ床を大きく超過。N=4096 は突出起動によりノイズ床が大きく、`BorrowedKeepAlive − LegacyToVec` 以外はノイズ帯 | N=1024／2048 では**不整合**（ノイズでは説明されない）。N=4096 では部分的に整合。計測は load average 14〜22（論理 CPU 16）の共有負荷下 |

### 10.4 まとめ（診断専用・判定なし）

- checksum は全腕・全起動で一致（規則 5。除外起動なし）。
- N=1024／2048 で `BorrowedKeepAlive` の `readback` 後退はノイズ床を超えて
  再現し、`PretouchedReusedDest`・（N=1024 では）`BorrowedWithDummyAllocFree`
  で legacy 水準へ戻る。この観測は M1 の予測方向と整合する。
- 規模照合（N=1024）は Δ≈0.47 ms の約 0.43 倍で事前登録目安の下限を
  わずかに外れ、機構の帰属は**仮説段階**のまま。残差の所在は §11 の
  拡張計測が候補（別イシューへの切り出しはユーザー承認のうえ行う）。
- 本記録は #1520 の REJECT（legacy フォールバック維持）・tolerance／
  baseline・本番結線を一切変更しない。

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
