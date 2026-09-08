# CUDA `host-view-readout` 有効時の reuse N=1024/2048 後退のフェーズ分解診断

イシュー #1436（親 #1435）。GB10（sm_121）実機実測（2026-09-08）。
**#1437 で是正完了**（`memory::readback` の宛先確保方式を
`ReadbackDest::PretouchedFresh` へ切替。全 N で受け入れ条件〈Gate 1〉
達成。§13）。

## 0. 要約

- `bench-fandhe` cargo feature `host-view-readout`（既定 OFF・#1335/#1336/#1337）
  有効時の CUDA reuse GEMM で観測された N=1024 15.04 倍・N=2048 1.20 倍の
  後退は、**`matmul` 区間（`Var::matmul` → `readback` = `clone_dtoh` +
  `synchronize`）に完全に集中しており、`to_tensor`／`host_copy`／
  `checksum` は増分を持たない**（§5）。
- 後退の機構は **glibc malloc の動的 mmap 閾値適応の有無**が有力仮説
  である（§8・H1 **有力仮説（裏付けあり）・腕間の分離計測は未実施**。
  §4・§11）。off 腕（`readout_var` 既定経路）は反復ごとに `to_vec()`
  で確保した 2 本目のバッファを free するため、この free が
  `M_MMAP_THRESHOLD` を動的に引き上げ、以後 `clone_dtoh` の内部確保が
  ヒープ（brk）領域の既タッチページを再利用できる、という機構を想定
  している。一方 on 腕（`host-view-readout`）は free が一切発生しない
  ため `clone_dtoh` の宛先が常に未タッチの新規 mmap ページとなり、GPU
  の D2H 書き込みが初回ページフォールト処理を伴う、という説明である
  （§2・§8）。ただし Layer B（§6・§7）は 4 腕を同一プロセス内で
  `LegacyToVec` → `BorrowedKeepAlive` → `BorrowedWithDummyAllocFree` →
  `PretouchedReusedDest` の順に逐次実行しており、先行腕の free・
  `keep_alive` 一括解放が後続腕のアロケータ状態（動的 mmap 閾値）へ
  引き継がれる交絡がある。この交絡を排除する腕単体・プロセス分離計測
  （§4 に入口を用意済み）は本イシューでは未実施のため、「on 腕固有の
  free 欠如が原因」という主張は独立検証できておらず**仮説にとどまる**。
  一方、増分が `matmul`（`d2h`）区間に集中する事実（§5）、および
  アロケータ状態に依存しない別実装である `PretouchedReusedDest` が
  全 N で d2h 最速という事実（§6・§7）は、交絡の影響を受けない観測
  として確定している。
- `HostStagingCache`（#1336）はこの経路に到達しない（§2.1・§8 H4）。
  `to_tensor` 固定費・非 contiguous 実体化（H3）・小形状固定費（H5）は
  いずれも増分を説明できず棄却できる（§8）。
- N=2048（4 MiB 未満に収まらない 16 MiB バッファ）は **bimodal**（3 回中
  2 回 slow≈125 ms・1 回 fast≈4.8 ms）を実測した（§6）。N=4096（64 MiB）
  は on 腕でも matmul 自体は off 腕とほぼ同等（§6・§9）であり、on 腕が
  `to_vec()` の 2 本目確保コストを完全に免れる分だけ `iter_total` が
  改善する（既報告の 0.638 倍。#1146 の「32 MiB 以上は常に mmap」知見
  と整合）。
- `MALLOC_MMAP_THRESHOLD_` を固定（64 MiB）して動的閾値適応を無効化する
  確証実験（§7）で、off 腕も on 腕と同じ低速状態に収束することを確認し、
  機構仮説を裏付けた。
- #1437 への是正候補として、事前タッチ済み宛先を再利用する読み出し
  （腕 `PretouchedReusedDest` が全 N で **d2h が最速**）を候補 A として
  提示する。ただし N=2048 の bimodal fast 側（`BorrowedKeepAlive` 自体が
  既に高速な run）では `host_read` 増加が `d2h` 削減を上回りネットで
  約 +1.2 ms 後退する（比較元を統一した場合の再評価。§10）。

## 1. 背景

`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §12.3／§12.4／§13 が
報告した数値（正式系列 `fandhe-ai =0.7.0` に対する参考系列 HEAD
readout-on の比較）:

| N | off→on iter_total 比 | 備考 |
|---|---|---|
| 1024 | 15.04 倍（後退） | 2.382 ms → 35.832 ms（約 +33 ms） |
| 2048 | 1.20 倍（後退） | max 135 ms のスパイクを含む |
| 4096 | 0.638 倍（改善） | candle 比 0.933→1.433（#1031 達成方向） |

ユーザー判断（2026-09-08）: 既定 OFF は維持しつつ、小形状後退の原因を
確定 → 是正（#1437）→ N=4096 改善を既定経路へ取り込む（#1438）。本
イシューはその第 1 段（原因確定）。

## 2. 経路の整理

| 経路 | off（既定） | on（`host-view-readout`） |
|---|---|---|
| `readout_var` | `Var::to_tensor()`（`Arc` 複製）→ `.contiguous().as_slice().to_vec()`（新規 `Vec` 確保・memcpy）→ 反復末尾で free | `Var::host_view()`（`Arc` 複製のみ）を `Deref` で借用。追加確保・free なし |
| `Var::matmul` の出力 | `run_f32_kernel`（`crates/backend-cuda/src/gemm.rs`）→ `memory::readback`（`memory.rs:552`）→ `clone_dtoh` + `synchronize`。両腕共通 | 同左 |
| tape（reuse モード） | eager ノード（readback `Vec` を storage とする `Tensor`）が反復ごとに蓄積し free されない | 同左 |

### 2.1 `HostStagingCache` の非到達（H4 の裏付け）

`crates/backend-cuda/src/host_staging.rs`（#1336 の `MemoryOps::
with_host_view`）は `Var::matmul` → `readback` の経路から一切呼ばれない
（`docs/perf/cuda-host-view-staging-readout.md` §7・`cuda-gemm-candle-
gate-remeasurement.md` §13.3 で既に確認済み。本イシューでもコード上の
呼び出しグラフを再確認し一致）。したがって off/on の差は「`readback`
（`clone_dtoh` + `synchronize`）の宛先バッファを、呼び出し元
（`readout_var`）がどう扱うか」に限定される。

## 3. 仮説と事前予測

計測前に確定した仮説表（§8 の判定結果と対応）。

| ID | 仮説 | 予測（Layer A） | 予測（Layer B／確証実験） |
|---|---|---|---|
| H1（主） | free が無く mmap 閾値が適応しないため `clone_dtoh` 宛先が毎回未タッチ mmap ページになり、D2H 中のページ確保が `matmul` 区間に乗る | 増分は `matmul` に集中 | 腕 (b) が (a) より `d2h` で大幅遅・腕 (d) で回復・`MALLOC_MMAP_THRESHOLD_` 固定で off 腕も低速化 |
| H2 | GPU が書いたページを CPU が初めて読む際のコヒーレンス／キャッシュ状態が遅い | 増分が `checksum` に集中 | 腕 (b) で `host_read` が遅く `d2h` は不変 |
| H3 | gemm 出力 `Tensor` が非 contiguous と判定され `contiguous()` が実体化している | 増分が `to_tensor` に集中 | — |
| H4 | `HostStagingCache` の再確保・世代検査 | 非到達のため説明にならない | コード経路上棄却 |
| H5 | 小形状固定費（`VarHostView` 構築・`Arc` 操作） | `to_tensor` ≈ 数十 ns | — |

## 4. 環境・プロトコル

- GB10（sm_121）実機。driver 580.173.02・CUDA 13.0（`nvcc` V13.0.88）・
  rustc 1.97.0。計測時 GPU utilization 0%・load average 0.16〜0.42
  （`docs/perf/logs/cuda-host-view-readout-regression-1436/env_info.txt`）
- Layer A: `bench-fandhe --task gemm --device cuda --mode reuse --phases`。
  off／on の 2 バイナリはいずれも同一 HEAD の facade path patch
  （`patch.crates-io.fandhe-ai.path`）でビルド（registry／path 混在事故
  回避。§13.1 の教訓を踏襲）。各 20 warmup + 20 計測の中央値
- Layer B: `crates/backend-cuda/src/readout_regression_diag_tests_1436.rs`
  （新規診断テスト。`#[ignore]`・`--test-threads=1`・`--release`）。
  H2D・カーネル起動・同期を共通に済ませたうえで D2H 以降（`d2h`／
  `host_read`）のみを 4 腕で分解計測（20 warmup + 20 計測の中央値）。
  §6・§7 で報告する値は `ReadoutArm::ALL`（`LegacyToVec` →
  `BorrowedKeepAlive` → `BorrowedWithDummyAllocFree` →
  `PretouchedReusedDest` の順）を**同一プロセス内で逐次実行**した
  結果であり、先行腕のアロケータ状態（動的 mmap 閾値・`keep_alive`
  一括解放）が後続腕へ引き継がれる交絡を含む。腕単体・サイズ単体を
  新規プロセスとして起動し交絡を排除する入口（`readout_regression_
  diag_n{1024,2048,4096}_{legacy_to_vec,borrowed_keep_alive,
  borrowed_with_dummy_alloc_free,pretouched_reused_dest}`。同ファイル
  427〜509 行）を用意済みだが、本イシューでは実機実測の時間制約により
  **未実行**（§11 に是正候補として引き継ぐ）
- **スコープ縮小**: Layer A は各 off/on 1〜2 プロセス起動、Layer B は
  N=1024/2048 各 3 プロセス起動・N=4096 は 1 プロセス起動（計画の 5 回
  から縮小。時間制約。詳細は logs README「スコープの縮小」節）

## 5. Layer A 実測（公開 API 境界のフェーズ分解）

単位 ms（中央値）。

| N | 腕 | matmul | to_tensor | host_copy | checksum | iter_total |
|---|---|---|---|---|---|---|
| 1024 | off | 0.432 | 0.00005 | 1.488 | 0.532 | 2.459 |
| 1024 | on | 35.317 | 0.0001 | 0.0001 | 0.527 | 35.845 |
| 2048 | off | 2.108 | 0.0002 | 5.366 | 2.130 | 9.714 |
| 2048 | on（run1） | 8.836 | 0.0004 | 0.0002 | 2.144 | 10.984 |
| 2048 | on（run2） | 9.989 | 0.0005 | 0.0003 | 2.137 | 12.128 |
| 4096 | off | 29.602 | 0.00003 | 16.812 | 8.628 | 55.063 |
| 4096 | on | 29.975 | 0.0003 | 0.0004 | 8.627 | 38.625 |

checksum（要素和）は off/on・全 N で完全一致（`parity_fail_count=0`）を
確認済み（数値契約不変の裏付け）。

**観察**: N=1024 は `matmul` の増分（+34.9 ms）が `host_copy` の減少
（1.488→0.0001 ms・約 1.49 ms）に比べて 1 桁以上大きいため、`matmul`
増分がほぼそのまま `iter_total` の増分（+33.386 ms）に反映される。
一方 N=2048 は `matmul` の増分（+6.7〜7.9 ms）と `host_copy` の減少
（5.366→0.0002〜0.0003 ms・約 5.37 ms）が同程度の大きさであり、
両者はほぼ相殺し合う。`iter_total` の増分（+1.270〜2.414 ms）は
「`matmul` 増分 − `host_copy` 減少」にほぼ一致する
（run1: 6.728−5.366=1.362 ms・run2: 7.881−5.366=2.515 ms。実測の
+1.270 ms・+2.414 ms と整合）。すなわち N=2048 では `host_copy` は
「実質不変」ではなく `matmul` 増分の大半を相殺する側で明確に減少
している（`host_copy` は on 腕で `Deref` 借用のみのため定義上ほぼ 0
になる。README「借用ビュー readout」節どおり）。`to_tensor`／
`checksum` は N=1024/2048 とも off/on で実質不変。N=4096 は `matmul`
自体はほぼ不変（29.6 vs 30.0 ms）で、`iter_total` の改善
（55.1→38.6 ms）は off 腕の `host_copy`（16.8 ms。`to_vec()` の 2 本目
確保・memcpy）を on 腕が完全に免れる分そのものである。

## 6. Layer B 実測（D2H 内訳の 4 腕分解）

単位 ms（中央値）。

> **`BorrowedWithDummyAllocFree` 行は旧実装の参考値（要再計測）**:
> 下記の `BorrowedWithDummyAllocFree` 実測値は、ダミー確保を
> `vec![0.0f32; n]`（ゼロ初期化のみ）で行っていた実装（f456f10 以前）
> で取得したものであり、glibc の mmap 経由確保がゼロページ（COW）で
> 応答した場合、確保した領域全体への物理ページ書き込み・free 発生を
> 保証できない。f456f10（#1442 codex-review 指摘対応）で全要素への
> 明示的な非ゼロ値書き込み + `std::hint::black_box` 保護を追加した
> 現行実装では、この行の値が変わる可能性がある。したがって下記の
> `BorrowedWithDummyAllocFree` の値は**「ダミー確保・解放では改善
> しない」という結論の根拠にできない**（旧実装の参考値。§7 の同名行も
> 同様）。現行実装での再計測は未実施のため、対照腕としての結論
> （§10 候補 B 節）は再計測まで保留する。

### N=1024（4 MiB バッファ）

| 腕 | d2h（run1/run2/run3） | host_read |
|---|---|---|
| LegacyToVec（off 再現） | 4.043 / 3.448 / 3.605 | 2.026 / 1.846 / 1.780 |
| BorrowedKeepAlive（on 再現） | 39.027 / 30.900 / 29.946 | 0.542 / 0.533 / 0.534 |
| BorrowedWithDummyAllocFree | 38.802 / 30.579 / 30.371 | 0.549 / 0.541 / 0.539 |
| PretouchedReusedDest | 0.078 / 0.076 / 0.076 | 2.336 / 2.335 / 2.071 |

### N=2048（16 MiB バッファ。bimodal）

| 腕 | d2h（run1/run2/run3） | host_read |
|---|---|---|
| LegacyToVec | 4.175 / 4.244 / 4.057 | 7.476 / 7.088 / 7.445 |
| BorrowedKeepAlive | **125.456** / 4.768 / **125.538** | 2.157 / 2.168 / 2.157 |
| BorrowedWithDummyAllocFree | 126.042 / 159.831 / 127.149 | 2.187 / 2.217 / 2.188 |
| PretouchedReusedDest | 0.283 / 0.283 / 0.285 | 8.169 / 7.912 / 8.293 |

`BorrowedKeepAlive` の 3 回中 2 回が slow（約 125 ms）・1 回が fast（約
4.8 ms・`LegacyToVec` と同水準）という bimodal 挙動を実測した。Layer A
の N=2048（run2 が run1 より重い: matmul 8.836→9.989 ms）とも整合する
（ただし Layer A は 2 回とも比較的軽い側に留まっており、Layer B ほど
極端な slow 側は Layer A では未観測。プロセス起動間隔・直前のヒープ
状態に依存する probabilistic な挙動と考えられる。§9 参照）。

### N=4096（64 MiB バッファ）

| 腕 | d2h | host_read |
|---|---|---|
| LegacyToVec | 22.733 | 28.946 |
| BorrowedKeepAlive | 23.736 | 8.749 |
| BorrowedWithDummyAllocFree | 23.592 | 8.760 |
| PretouchedReusedDest | 1.128 | 29.820 |

N=4096 では `LegacyToVec` の `d2h` 自体も 22.7 ms まで悪化しており
（N=1024/2048 の fast 側 3〜5 ms と比べ一段重い）、`BorrowedKeepAlive`
の 23.7 ms とほぼ同水準まで収束している。64 MiB は glibc の既定
`DEFAULT_MMAP_THRESHOLD_MAX`（多くのディストリビューションで 32 MiB
前後）を超えるため、動的閾値適応が効いても mmap を回避しきれない
領域に入っていると推定される（`docs/perf/cuda-large-buffer-percall-
alloc-transfer-threshold.md` #1146 の「32 MiB 以上は常に mmap」知見と
整合。本イシューでは追加検証していない推定）。

## 7. 確証実験（`MALLOC_MMAP_THRESHOLD_` 固定）

N=1024・`MALLOC_MMAP_THRESHOLD_=67108864`（64 MiB。バッファサイズ 4 MiB
より十分大きく設定し、動的閾値適応を実質的に無効化する）:

> **`BorrowedWithDummyAllocFree` 行は旧実装の参考値（要再計測）**:
> §6 冒頭の注記と同じ理由（f456f10 以前の実装で取得）。この行の値も
> 結論の根拠にできない。

| 腕 | d2h | host_read |
|---|---|---|
| LegacyToVec | **27.587**（既定時 3.4〜4.0 ms から急変） | 1.347 |
| BorrowedKeepAlive | 30.533 | 0.534 |
| BorrowedWithDummyAllocFree | 30.301 | 0.538 |
| PretouchedReusedDest | 0.076 | 2.113 |

`LegacyToVec`（off 腕再現）が固定閾値下では `BorrowedKeepAlive`（on 腕
再現）とほぼ同じ低速状態（27.6 ms 対 30.5 ms）に収束した。既定（動的
閾値）では `LegacyToVec` が 3〜4 ms と一桁以上高速だったことと対比する
と、**off 腕の高速性は「反復ごとの free が動的 mmap 閾値を引き上げ、
以後の確保がヒープ内の既タッチ領域を再利用できる」ことに起因する**
という H1 の機構を裏付ける。

ただしこの実験でも 4 腕は同一プロセス内で `LegacyToVec` から順に
実行しており、`LegacyToVec` はその中で最初の腕である（後続腕からの
汚染は受けない）。したがって本実験が直接裏付けるのは「**off 腕
（`LegacyToVec`）自身**の高速性が動的閾値適応に依存する」ことであり、これは
off 腕単独で交絡なく確認できている。一方「on 腕（`BorrowedKeepAlive`）
が遅いのは on 腕自身に free が無いからだ」という**on 腕側の原因特定**
は、`BorrowedKeepAlive` が `LegacyToVec` の実行後（`LegacyToVec` の
free 群により閾値が動的に引き上げられた状態）で計測されているため、
この実験だけでは独立に検証できていない。§4 に用意した分離プロセス
入口（`BorrowedKeepAlive` を単独プロセスで実行し、`LegacyToVec` の
影響を受けない状態で同様に低速となるかを確認する）を実行することで
初めて on 腕側の原因も確定できる（未実施。§11）。

## 8. 仮説の支持／棄却

| ID | 判定 | 根拠 |
|---|---|---|
| H1（主） | **有力仮説（裏付けあり）・on 腕固有の原因特定は分離計測未実施** | §5: 増分が `matmul` に集中（to_tensor／host_copy／checksum は不変）— 交絡の影響を受けない確定観測。§6: `BorrowedKeepAlive`（free なし）が `LegacyToVec`（free あり）より d2h で 7〜30 倍遅い — ただし両腕は同一プロセス内で `LegacyToVec` → `BorrowedKeepAlive` の順に逐次実行されており、`LegacyToVec` の free 群が `BorrowedKeepAlive` 計測時の動的 mmap 閾値へ引き継がれる交絡がある。§6: `PretouchedReusedDest`（事前タッチ・別実装のためアロケータ状態の交絡を受けない）が全 N で d2h 最速（N=1024 で 0.076 ms・N=2048 で 0.283 ms・N=4096 で 1.128 ms）— 確定観測。§7: `MALLOC_MMAP_THRESHOLD_` 固定で `LegacyToVec`（off 腕再現。4 腕中最初に実行され交絡を受けない）が低速化することは確定。ただし同実験内の `BorrowedKeepAlive` は `LegacyToVec` 実行後の状態で計測されているため、「on 腕固有の free 欠如が原因」という主張はこれらの実測から独立に確認できておらず、腕単体・プロセス分離計測（§4 に入口あり・未実行）が必要 |
| H2 | **棄却** | §6: 増分は `d2h` に現れ `host_read` は on 系 3 腕でほぼ同水準（0.53〜0.55 ms・N=1024）。GPU 書き込み後の CPU 初回読み出しコストが支配的なら `host_read` 側に出るはずだが観測されない |
| H3 | **棄却** | Layer A の `to_tensor` は off/on とも実測で数十〜数百 ns（§5 表）であり、非 contiguous 実体化のような大きなコストは observられない |
| H4 | **コード経路で棄却（実測でも補強）** | §2.1: 呼び出しグラフ上 `with_host_view` は `readback` から到達不能。§6: `PretouchedReusedDest`（`HostStagingCache` と同じ「事前タッチ済み宛先再利用」設計方針）が別実装のまま最速を示しており、`HostStagingCache` 固有の機構（世代検査・`Pinned`/`Pageable` 分岐）を経由せずとも同じ改善効果が得られることを確認 |
| H5 | **棄却** | `to_tensor` ≈ 数百 ns（#1182 実測オーダーと一致）であり 33 ms を到底説明できない |
| 非単調性（N=2048 bimodal） | **観測事実として記録（機構は未確定）** | §6: 3 回中 2 回 slow・1 回 fast。H1 の機構（動的 mmap 閾値・ヒープ空き領域の再利用可否）はプロセス起動ごとのヒープレイアウト初期状態に依存しうるため、probabilistic な挙動として一貫する。ただし「なぜこの起動順で slow/fast が決まるか」の厳密な条件特定は本イシューのスコープ外（§11） |

## 9. N=4096 のみ改善する理由

- N=4096 の `matmul` 自体は off/on でほぼ同値（29.6 対 30.0 ms。§5）。
  §6 の Layer B でも `LegacyToVec` の `d2h` が 22.7 ms まで悪化し
  `BorrowedKeepAlive` の 23.7 ms に接近している。これは 64 MiB という
  サイズが glibc の動的 mmap 閾値適応の効果が及ぶ範囲（一般に
  `DEFAULT_MMAP_THRESHOLD_MAX` 以下）を超えており、off 腕の「free に
  よる閾値引き上げ」戦略がもはや mmap 回避に効かなくなっているためと
  推定される（#1146 の知見と整合する推定であり、本イシューで直接
  追加検証はしていない）
- 一方 `iter_total` は on 腕が明確に高速（38.6 対 55.1 ms）。これは
  `matmul` 自体の差ではなく、off 腕の `readout_var` が `to_tensor`（ほぼ
  無償）に続けて `host_copy`（`to_vec()`。64 MiB の 2 本目確保＋memcpy。
  16.8 ms）を追加で払う一方、on 腕は `host_view()` の借用のみで
  `to_tensor`／`host_copy` ともほぼ 0 になるためである。**N=4096 の
  改善は「D2H 自体が速くなった」のではなく「2 回目のホストコピーを
  省略できた」ことに由来する**、という整理が Layer A/B 双方の実測と
  整合する

## 10. #1437 への是正候補

Layer B の腕別実測（§6）から、以下の優先順位で候補を提示する。

- **候補 A（条件付き推奨）**: `memory::readback` の D2H 宛先を事前
  タッチ済みステージングバッファへ変え、tape 所有用に copy-out する。
  `PretouchedReusedDest` 腕が **全 N・全ケースで d2h が最速**（N=1024:
  0.076〜0.078 ms・N=2048: 0.283〜0.285 ms・N=4096: 1.128 ms。§6・§7）
  であり、`MALLOC_MMAP_THRESHOLD_` の probabilistic な挙動（§6 の
  bimodal・§9 の大形状での効果減衰）に依存しない決定的な解決になる。
  `host_read` 区間（copy-out のコスト）は増えるため、d2h 削減量と
  host_read 増加量は **同一比較元**（on 腕の再現である
  `BorrowedKeepAlive`）に統一して評価する必要がある（d2h 削減量を
  `BorrowedKeepAlive` 比、host_read 増加量を `LegacyToVec` 比のように
  比較元を混在させると、下記 N=2048 fast 側のネット後退が埋もれる）:
  - **N=1024**（`BorrowedKeepAlive` 比）: d2h 削減 -29.9〜-39.0 ms に
    対し host_read 増加 +1.5〜1.8 ms → 常にネット改善（約 -28〜-37 ms）
  - **N=2048**（`BorrowedKeepAlive` 比。bimodal の run1/run2/run3
    それぞれ）: d2h 削減は run1 -125.2 ms・run2 **-4.5 ms**・run3
    -125.3 ms、host_read 増加は 3 run とも +5.7〜+6.1 ms。
    `BorrowedKeepAlive` が slow 側（run1/run3。3 回中 2 回）ならネット
    改善（約 -119 ms）だが、`BorrowedKeepAlive` が既に fast 側
    （run2。動的 mmap 閾値がたまたま有利に適応済みのケース）だと
    **ネットで約 +1.2 ms 後退する**（d2h 削減 -4.5 ms < host_read
    増加 +5.7〜6.1 ms）。すなわち候補 A は「slow 側を高速化する」代わり
    に「fast 側をわずかに遅くする」トレードオフを持ち、bimodal の
    どちら側が定常状態かに依存して net の符号が変わりうる。N=2048
    bimodal の発生条件は未特定（§11）のため、#1437 での採否判断は
    このトレードオフを踏まえて行う必要がある
  `HostStagingCache`（#1336）の再利用が自然な実装先候補になる
  （既存の `Pageable` 種別・世代検査の仕組みをそのまま `readback` 経路
  へ繋げる）
- **候補 B**: `readback` 直前に宛先 `Vec` を明示的にページタッチしてから
  `memcpy_dtoh` する（毎回新規確保のまま、確保直後に触るだけ）。腕
  `BorrowedWithDummyAllocFree` は「読み出し**後**のダミー確保・解放」
  であり、候補 B が意図する「読み出し**前**の明示タッチ」とはそもそも
  異なる操作である点に注意（本イシューでは候補 B 自体（事前タッチ＋
  毎回新規確保）は直接計測していない）。加えて `BorrowedWithDummyAllocFree`
  の §6・§7 実測値自体が旧実装（f456f10 以前・物理ページ書き込み非保証）
  による**参考値**であり、「`LegacyToVec`・`BorrowedKeepAlive` とほぼ
  同水準で改善せず」という現象の記述は f456f10 で書き込み・dead store
  除去を保護した現行実装での再計測が済むまで確定した結論として扱わない
  （§6 冒頭注記参照）。候補 A の `PretouchedReusedDest` は「事前確保・
  使い回し」であり候補 B とは実装が異なる。候補 A で十分な効果が出て
  いるため、候補 B は候補 A が不採用になった場合の代替として位置づける
- **候補 C（限定的フォールバック）**: `bench-fandhe` 側の N 閾値による
  形状条件付き切替（N≥4096 のみ借用ビュー）。製品側（`crates/backend-
  cuda`）の改善にならないため、候補 A／B が #1437 で不成立の場合のみ
  検討する
- **候補外**: `MALLOC_MMAP_THRESHOLD_` 等の環境変数チューニング（診断
  専用。製品・スクリプト既定に組み込まない）・`HostStagingKind::
  Pinned` の既定化（`unsafe` 経路の既定化はユーザー承認事項のまま）

## 11. スコープ外・未確定事項

- **H1 の on 腕固有の原因特定は分離計測未実施**（本 codex-review 指摘・
  PR #1442 レビュー対応で判明）。§6・§7 の Layer B 実測は 4 腕を
  `LegacyToVec` → `BorrowedKeepAlive` → `BorrowedWithDummyAllocFree` →
  `PretouchedReusedDest` の順に同一プロセス内で逐次実行しており、
  先行腕の free・`keep_alive` 一括解放が後続腕のアロケータ状態
  （動的 mmap 閾値）へ引き継がれる交絡を排除できていない。§4 に
  用意した腕単体・プロセス分離実行エントリ（`readout_regression_
  diag_n{1024,2048,4096}_{legacy_to_vec,borrowed_keep_alive,
  borrowed_with_dummy_alloc_free,pretouched_reused_dest}`）を
  `cargo test --release -p fandhe-ai-backend-cuda --lib <test名> --
  --ignored --test-threads=1` で個別プロセス起動し、`BorrowedKeepAlive`
  単独（`LegacyToVec` の影響を受けない状態）でも同様に低速となるかを
  確認することが、H1 を「仮説」から「確定」へ格上げするために必要。
  本イシューでは実機実測の時間制約により未実施のため #1437 へ引き継ぐ。
  現時点で交絡なく確定できているのは「増分が `matmul`（`d2h`）区間に
  集中する」「`PretouchedReusedDest`（別実装で交絡を受けない）が全 N
  で d2h 最速」の 2 点のみである
- **`BorrowedWithDummyAllocFree` 腕の再計測が未実施**（本 codex-review
  指摘・PR #1442 レビュー対応で判明）。§6・§7 の同腕の値は f456f10
  以前（ダミー確保がゼロ初期化のみで物理ページ書き込み・dead store
  除去防止を保証していなかった実装）で取得したものであり、f456f10 で
  非ゼロ値書き込み + `std::hint::black_box` 保護を追加した現行実装
  での再計測が未実施のため、「ダミー確保・解放では改善しない」という
  §10 候補 B 節の記述は確定した結論として扱わない（旧実装の参考値。
  §6・§7 の注記参照）。#1437 で候補 B の採否を検討する前に、現行実装
  での再計測が必要
- N=2048 bimodal の厳密な発生条件（どの起動順・ヒープ初期状態で
  slow/fast が決まるか）は未特定。#1146 の「32→33 MiB 段差」「降順
  走査限定の確率的スパイク」と同族の可能性はあるが、本イシューでは
  推測に留め断定しない
- 4/8/12/16/24/32/64 MiB の精密なバイトサイズスイープ（非単調性・
  二峰性のより細かい可視化）は時間制約により未実施（logs README
  「スコープの縮小」節）
- `strace -c` 等の syscall 回数計測は未実施。`MALLOC_MMAP_THRESHOLD_`
  固定による確証実験（§7）で機構仮説を実測で裏付けられたため、追加の
  プロセス計装は本イシューでは行わなかった
- N=4096 D2H 二峰性の根本原因（#1169／#1146 系）と 32→33 MiB 段差は
  未解明のまま引き継ぐ（本イシューのスコープ外）
- `HostStagingKind::Pinned` 既定化・キャッシュ可能 pinned
  （`malloc_host`）はユーザー承認事項のまま（変更なし）
- N=2048 の candle 判定不能（#1258）は本イシューで扱わない
- Metal の全形状後退（1.30〜1.56 倍）は本イシューの対象外（CUDA 限定）

## 12. 参照

- 診断テスト: `crates/backend-cuda/src/readout_regression_diag_tests_1436.rs`
- 生ログ・env_info: `docs/perf/logs/cuda-host-view-readout-regression-1436/`
- 関連: `docs/perf/cuda-host-view-staging-readout.md`（#1336）・
  `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §12/§13（#1142/#1360）・
  `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`（#1146）・
  `docs/perf/cuda-gemm-reuse-phase-breakdown.md`（#1182）

## 13. #1437 是正結果（採用: 候補 B「fresh 宛先の全要素事前タッチ」・GB10 実機実測 2026-09-08）

イシュー #1437（本イシューの後続）。§10 で提示した候補 A（`PretouchedReusedDest`
系。事前タッチ済み宛先の**使い回し**＋copy-out）は N=2048 bimodal の
fast 側でネット後退しうるトレードオフを持つため、代わりに未計測だった
**候補 B の変種**（`crates/backend-cuda/src/memory.rs::ReadbackDest::
PretouchedFresh`。反復ごとに`vec![1.0f32; numel]`で新規確保した**非ゼロ
sentinel 埋め済み**宛先へ `memcpy_dtoh` する。copy-out なし・宛先の使い
回しなし）を実装し採用した。

### 13.1 設計判断の要点

- **`readback`（`memory.rs`。全 30 箇所超の呼び出し元が共有する唯一の
  D2H 同期点）自体を是正**した。これにより `readout_var`（bench 側）を
  変更せずに on 経路が改善し、feature 撤去（#1438）への引き継ぎも
  最小変更になる
- `ReadbackDest` enum（`Fresh`／`PretouchedFresh`）+ `READBACK_DEST`
  const で戦略を切り替える。`Fresh` は現行 `clone_dtoh` を bit 単位で
  維持
- **非ゼロ sentinel が必須**: `vec![0.0; n]` は `alloc_zeroed`（calloc）
  → mmap の COW ゼロページのままで物理ページが確保されず「事前タッチ」
  にならない（`ReadoutArm::PretouchedReusedDest`／`PretouchedFreshDest`
  診断腕が既に踏んだ罠と同じ。§6 参照）。`ReadbackSentinel` トレイト
  （`f32: 1.0`・`f16: half::f16::ONE`）で型ごとの非ゼロ値を強制する
- generic `T`（f32 のほか f16。`gemm_mma.rs::download_f16` が経由）
  への対応は `ReadbackSentinel` トレイトの型ごと impl で行う
- 出力は `Fresh` と bit 完全一致（`memcpy_dtoh` は全バイトを上書きする
  コピーであり、事前タッチの値は D2H 完了後には残らない。実機
  `#[ignore]` テスト `readback_pretouched_bit_match_1437.rs` で
  f32/f16 とも複数形状（0・1・37・4096・1,048,576・16,777,216 要素）を
  横断して確認済み）

### 13.2 Layer B 分離計測（単一腕プロセス分離実行。§4 の未実施事項の解消）

`readout_regression_diag_tests_1436.rs` に候補 B の先取り腕
`PretouchedFreshDest`（反復ごとに `crate::memory::pretouched_host_vec`
で新規確保・fill を d2h 区間に含めて計測）を追加し、`LegacyToVec`・
`BorrowedKeepAlive`・`PretouchedFreshDest` の 3 腕を単一腕・単一プロセス
起動（`readout_regression_diag_n{1024,2048,4096}_{legacy_to_vec,
borrowed_keep_alive,pretouched_fresh_dest}`）で計測した。これにより
§11 が引き継いだ「H1 の on 腕固有原因を交絡なく検証する」目的も同時に
満たされる。

| N | LegacyToVec（off 相当） | BorrowedKeepAlive（on 相当・後退再現） | PretouchedFreshDest（採用候補） |
|---|---|---|---|
| 1024 | d2h 3.94 ms + host_read 2.12 ms = 6.06 ms | d2h 36.61 ms + host_read 0.54 ms = **37.15 ms** | d2h 1.51 ms + host_read 0.53 ms = **2.04 ms** |
| 2048 | d2h 4.23 ms + host_read 7.91 ms = 12.15 ms | d2h 9.16 ms + host_read 2.16 ms = 11.32 ms | d2h 5.19 ms + host_read 2.15 ms = **7.34 ms** |
| 4096 | d2h 20.32 ms + host_read 28.07 ms = 48.39 ms | d2h 20.63 ms + host_read 8.63 ms = 29.26 ms | d2h 21.63 ms + host_read 8.75 ms = 30.37 ms |

（各セル: 20 warmup + 20 測定の中央値。生ログ・全 quartile は
`docs/perf/logs/cuda-host-view-readout-fix-1437/layer-b/`）

`PretouchedFreshDest` は N=1024/2048 で `LegacyToVec`（off 経路相当）・
`BorrowedKeepAlive`（on 経路・後退再現）の両方を明確に下回り、N=4096 は
`BorrowedKeepAlive`（29.26 ms）と僅差（30.37 ms・約 4% 差）に収束する。
`LegacyToVec`（48.39 ms）はこの僅差の範囲には収まらず、N=4096 では
`PretouchedFreshDest`・`BorrowedKeepAlive` の 2 腕がむしろ `LegacyToVec`
を約 1.6 倍上回る（`host_read` 側の内訳差が支配的。上表）。N=1024 の
`BorrowedKeepAlive` 37.15 ms は §0 が報告した「N=1024 15.04 倍後退」を
本診断テストの計測境界（d2h+host_read のみ。H2D／カーネル起動を含まない）
でも再現しており、H1（宛先ページ未タッチ由来）が on 腕固有の後退の
交絡なき原因であることを確認した。

### 13.3 Layer A（framework-compare 実践規模）ゲート判定

事前宣言ゲート（実装計画 §5）:

- **Gate 1（受け入れ条件）**: 全 N で `on@after / off@base ≤ 1.00`
- **Gate 2（既存経路の非後退）**: 全 N で `off@after / off@base ≤ 1.03`

`off@base` は正式系列 `fandhe-ai =0.7.0`（registry 版。#1185 で
2026-09-06 計測済みの既存ファイルを再利用。`readback_with` 内の
`Fresh` 分岐自体は本イシューで変更していないため `off@base`／
`off@after` は出力（bit 単位）で同一系列として扱えるが、**性能面では
両者は無条件に同一ではない**——`readback()` の既定 `ReadbackDest` は
`host-view-readout` feature の有効・無効を問わず `PretouchedFresh` へ
切り替わっており（`memory.rs::READBACK_DEST`）、`off@after` も
`off@base`（旧 `Fresh` 既定）とは異なる宛先確保方式を通る。この差が
下表 Gate 2 の N=1024/2048 での超過に寄与している可能性を §13.3 末尾で
再評価する）、`off@after`／`on@after` は結線後 HEAD を
`GEMM_GATE_PATCH_FACADE_PATH` で path patch し、`on@after` のみ
`GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout` を追加した。
いずれも `run_gemm_gate_cuda.sh` による N=1024/2048/4096 reuse × 5 run。

| N | off@base 中央値 | off@after 中央値 | on@after 中央値 | Gate 2（off@after/off@base） | Gate 1（on@after/off@base） |
|---|---|---|---|---|---|
| 1024 | 2.4236 ms | 2.5153 ms | 2.1291 ms | 1.0378 | **0.8785**（PASS） |
| 2048 | 9.6010 ms | 10.0509 ms | 8.6062 ms | 1.0469 | **0.8964**（PASS） |
| 4096 | 62.3005 ms | 59.4962 ms | 39.6699 ms | 0.9550 | **0.6368**（PASS） |

**Gate 1（受け入れ条件そのもの）は全 N で通過**（0.637〜0.897 倍。
`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §13 が報告した
「N=1024 15.04 倍・N=2048 1.20 倍後退」を完全に解消し、N=4096 の改善
（0.638 倍。#1360）も維持している）。

**Gate 2（自己宣言した非後退の目安。≤1.03）は N=1024/2048 でわずかに
超過**（1.038・1.047）。当初の草稿では「`off@after` は `Fresh` 分岐
限定を通るため機構的な後退要因が存在しない」としてこの超過を実機計測
ノイズと判断していたが、これは誤りである。`readback()` の既定
`ReadbackDest` は `host-view-readout` feature に連動せず**無条件に**
`PretouchedFresh` へ切り替わっている（`memory.rs:575`
`READBACK_DEST = ReadbackDest::PretouchedFresh`）。したがって
`off@after` も `off@base`（旧 `Fresh` 既定）とは異なる宛先確保方式
（`vec![SENTINEL; numel]` による事前フィル + `memcpy_dtoh`）を通っており、
「後退要因が存在しない」という前提そのものが実装（`memory.rs:575` の
無条件 `PretouchedFresh`）と矛盾していた。

再評価: `PretouchedFresh` の事前フィル費用（帯域律速。N=1024 で
4 MiB・N=2048 で 16 MiB 相当。§13.2 コメントの見積りでは概ね
0.1〜数 ms オーダー）が `off@after` の全呼び出しに一律で乗ることは、
N=1024/2048 で観測された 1.038・1.047 倍という小さな超過の説明として
機構的に整合する（N=4096 では `off@after` がむしろ `off@base` を
下回っており〈0.955〉、大形状では D2H 本体の費用が支配的でフィル費用の
相対寄与が縮小するという同じ機構と矛盾しない）。5 run の生値レンジ
（`off@base` 2.32〜2.58 ms 対 `off@after` 2.37〜2.64 ms・N=2048 も
base 9.51〜9.82 ms 対 after 9.69〜10.51 ms）はなお重なっており測定
ノイズの寄与も否定できないため、単発追加計測での寄与分離までは
行っていないが、**「機構的な後退要因が存在しない」という当初の断定は
撤回する**。

この再評価を踏まえても ADOPT 判断自体は変更しない: 受け入れ条件
そのものである Gate 1 は全 N で明確に通過しており（0.637〜0.897 倍。
上表）、Gate 2 はその判断を補助する自己宣言の目安（≤1.03 は目安であり
受け入れ条件自体には含まれない）である。N=1024/2048 の超過幅（3.78%・
4.69%）は Gate 1 の改善幅（12〜36%）に対して小さく、`PretouchedFresh`
自体は N=1024/2048 の主目的（後退是正）を達成しつつ他形状・他経路への
副作用も限定的と判断する。ただし今後 `off` 経路（`host-view-readout`
無効時）の追加最適化を検討する際は、この事前フィル費用が既に一律で
乗っていることを前提に含める。全 6 セルとも `parity_fail_count=0`・
checksum 完全一致（生ログ `docs/perf/logs/cuda-host-view-readout-fix-1437/`）。

### 13.4 契約テスト・既存テストの非後退

- `readback_pretouched_bit_match_1437.rs`（新規 `#[ignore]` 実機テスト。
  f32/f16 × 6 形状）: PASS（`Fresh`／`PretouchedFresh` が byte 単位で
  完全一致）
- `memory_real_device.rs`（5 件）・`host_view_real_device.rs`（7 件）:
  結線後 HEAD で全 PASS（非後退）
- GPU 非依存単体テスト（`memory::tests` 27 件。うち
  `readback_sentinel_{f32,f16}_is_nonzero`・
  `pretouched_host_vec_{f32,f16}_has_expected_len_and_fill`・
  `pretouched_host_vec_zero_numel_is_empty` が新規）: 全 PASS
- 既存 `#[ignore]` テスト全体を結線後 HEAD で通しで実行し、`cpu_cuda_
  mma_parity`・`cpu_cuda_wmma_parity`・`gemm_mma_tf32x3`・
  `gemm_tf32_optin`・`gemm_wmma_f16_opt`・`tensor_core_real_device`・
  `graph_capture_real_device` の FAIL を確認したが、**同一形状・同一
  シードで origin/main（未変更ツリー・イシュー #1437 着手前の
  `5480f94`）でも同一の FAIL が再現する**ことを個別実行で確認済み
  （K=4096 の f16/TF32 tail 超過・小形状計測プロトコルのタイミング
  依存・CUDA Graph capture のプロセス内テスト順序依存はいずれも本
  イシュー着手前から存在する既知の環境依存 FAIL であり、`readback`
  是正が原因ではない）。`dispatch_boundary` は同一ツリーでの再実行で
  PASS に戻ったため、単発の計測タイミング揺らぎ（他テストと同型の
  「フェーズ 1 安定性ゲート」性質の flaky）と判断する

### 13.5 結線

判定木（実装計画 §5）に従い、受け入れ条件である Gate 1（全 N で
0.637〜0.897 倍・PASS）の通過を根拠に ADOPT し、
`READBACK_DEST = ReadbackDest::PretouchedFresh` を既定として結線した
（`crates/backend-cuda/src/memory.rs`）。Gate 2（自己宣言の非後退目安）
は §13.3 のとおり N=1024/2048 でわずかに超過（1.038・1.047）しており
「Gate 1・2 とも通過」ではない点に注意——Gate 2 は受け入れ条件自体には
含まれず、超過幅（3.78%・4.69%）が Gate 1 の改善幅（12〜36%）に対して
小さいことを理由に ADOPT 判断は変更しなかった（§13.3 末尾の再評価を
参照）。候補 C（bench 側 N 閾値切替）は不要となったため実装していない。

### 13.6 スコープ外（本イシューでも未実施のまま引き継ぐ）

- N=2048 bimodal の厳密な発生条件・N=4096 D2H 二峰性・32→33 MiB 段差
  （#1146／#1169 系。§11 から不変）
- 候補 A（`HostStagingCache` 再利用経由の readback）の本実装（候補 B が
  受け入れ条件〈Gate 1〉を満たし ADOPT されたため優先度低。Gate 2 は
  §13.3 のとおり N=1024/2048 でわずかに超過している）
- feature ゲート撤去・3 バックエンド candle 比ゲート再計測（#1438）
- Metal の全形状後退（1.30〜1.56 倍。CUDA 限定のためスコープ外）
