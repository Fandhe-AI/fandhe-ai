# 推論チェーン（infer CUDA reuse）の固定費の削減表（イシュー #3205）

## 1. 目的と結論

- GB10・v0.11.0 の infer CUDA reuse は fandhe-ai 約 0.099 ms、PyTorch 約 0.040 ms、candle 約 0.042 ms
  （`infer-reuse-phase-breakdown.md` §12）。差は約 55 µs。相手も同じ演算と同期 1 回を含む。
- 本 doc は「相手が払っていない固定費」を項目ごとに分け、公開 API の追加も新規 `unsafe` も無しに
  削れるものだけを 1 項目 1 PR で削るための表である。**差を閉じるものではない**。
- **実測状況（2026-10-23。GB10・ゲート付き）**: 層の発行の下位分解（§7）・1 項目の実装と前後比較（§8）・
  bit 一致（§9）・B3 の再判定（§10）・相手との比（§11）を実測した。**採用した項目は「層ごとの出力バッファの
  再利用」1 つ**（infer CUDA reuse 99.9 → 90.3 µs。after/before 0.9038）。差約 55 µs は閉じない
  （PyTorch・candle に対し 2.5 倍 → 2.3 倍。§11）。B3 は R1(b) がまだ失敗し **undetermined**（§10）。

## 2. 削減表

出典はすべて `infer-reuse-phase-breakdown.md` §12（#3189。GB10・中央値）。「次の PR」は未着手。

| 項目 | 実測 (µs) | 扱い | 理由・行き先 |
|---|---:|---|---|
| tape・snapshot・steps | 約 0.2 | 削らない | 計測ノイズ以下 |
| facade の residual（`predict_resident` のガード・`tape_for` 等） | 2.6 | 次の PR（調査のうえ判断。小） | GB10 での個別計測後に判断 |
| 入力 H2D（h2d_upload） | 11.3 | 公開 API が要る | #3250（常駐入力ハンドル。#3249 の決定待ち） |
| 層の発行の超過分（l1 7.5 / l2 6.6 − launch 下限 1.82×2） | 約 10.5 | 削れる候補（次の PR） | 下記 3 項目に分けて GB10 で測り、大きい順に 1 項目 1 PR |
| └ 中間・出力バッファの確保と解放（alloc 3・free 3／step） | 確保 3.1＋2.7（h・out）・解放 1.0（h）＝約 6.8（§7。単独計測） | **削った（本 PR・§8）** | 内部 trait `linear_forward_device_into_tracked` で層出力を再利用（公開 API・`unsafe` なし）。実測で 9.1 µs 短縮 |
| └ 層 1 回分の driver 呼び出し境界の重複（3〜4 回）＋lookup＋起動発行 | 約 3.1（l1）・3.0（l2）の合算（導出値。個別には切れない。§7） | 次の PR | `linear_forward_device` の境界を統合。poison・世代検査の範囲は保つ。1 項目 1 PR のため本 PR では触れない |
| └ 出力の memset（`alloc_zeros`） | 確保に含まれる（§7 の alloc。memset 単独は未分解） | 再利用で結果的に省いた（`unsafe` 不要）。未初期化確保そのものは別 issue | 未初期化確保は `unsafe` か pool の結線（`pool.rs` 冒頭がスコープ外と明記）が要る |
| └ cudarc の event tracking（wait_event 6／step） | 未分解 | 削らない（本 issue） | 無効化は既存の `unsafe disable_event_tracking` と既定反転を伴い全 CUDA 利用者に効く。#3227 側 |
| D2H＋同期（download_again） | 6.1 | 削れない | tiny_roundtrip 6.14 µs とほぼ同じで下限 |
| kernel_wait | 63.2 | 範囲外（固定費ではない） | §3 |
| 推論 Graph capture（B3） | reuse で −14 µs（batch 64。#3190）。再判定は reuse 90.6 → 86.0 µs | **undetermined**（R1(b) が別の呼び出し点で失敗。§10） | §4・§10 |

## 3. kernel_wait は固定費ではない

- tiled f32 カーネルのタイルは 64×64（`crates/backend-cuda/src/kernels.rs`・`gemm.rs::tiled_f32_launch_config`）。
  m=64 の推論では L1（64×784×256）が 4 ブロック、L2（64×256×10）が 1 ブロックしか起動しない。
  占有率の問題で、K ループが逐次に効いていると推定する（推定。未検証）。
- API・`unsafe` 無しで削れる余地は層の発行の超過分（約 10.5 µs）と residual の一部だけで、
  差約 55 µs の大半は埋まらない。小 m 向けタイルの選択などは別 issue の提案とする
  （起票は所有者承認が要る。本 PR では起票しない）。

## 4. B3（推論 Graph capture）の capture 競合の是正

- 原因: 別スレッドが capture 中に同じデバイスを呼ぶと `DeviceContextCaptureInProgress` が返り、
  R1(b) が失敗して undetermined だった（`infer-chain-graphcapture-cuda-ab.md` §3、
  `optin-ab-gb10.md` が #3205 へ申し送り）。
- 是正: `DeviceParamStore::predict_device_chain` がこのエラーを受けたときだけ、
  `BackendOps::wait_device_capture_idle`（既定 no-op。CUDA は
  `context_cache::wait_foreign_capture_idle`）で待ち、チェーン全体をやり直す。**当初は上限 100 ms・再試行 1 回だったが、GB10 の R1(b) で不足し
  （§10）、全体の待ち上限 1 s の範囲で繰り返す形へ改めた**。
  - `begin_driver_call` 全体は待ち型にしない（学習 step 側はこのエラーで直接実行へ落ちる契約のため）。
  - 自スレッドが `CallToken` を保持している場合は待たない（`begin_capture_session` の drain との
    相互待ちを防ぐ）。期限切れは従来のエラー（fail-closed）。poison・Retiring・世代の検査は弱めない。
  - store を書き換えないため、やり直しで部分結果は漏れない。
- 単体テスト（GPU 不要）: `context_cache` の `wait_foreign_capture_idle_*` 5 件、
  `device_store` の `predict_device_chain_*capture*` 系 3 件、`tensor-core` の既定実装 1 件。
- 再判定（R0〜R2・6 セル A/B）は新しい label で GB10 にて行った（§10）。ADOPT でも既定は変えない（#3227）。

## 5. 実施結果（要約）

1. 層の発行の下位分解（5 run）→ §7。**済**
2. B3 再判定（RULE 継承・新 label）→ §10。**済（undetermined）**
3. 最大の「削れる候補」（確保・memset・解放）を 1 項目実装し、5 round で前後を記録 → §8・§9。**済（ADOPT）**
4. 同一セッションで PyTorch・candle との比 → §11。**済（record-only）**

## 6. 対象外

- kernel_wait（小 m のタイル占有率）、memset 単独の省略（未初期化確保は `unsafe` か pool の結線）、
  event tracking、入力 H2D（#3250）、driver 呼び出し境界の統合（次の PR）、residual の削減。
- B3 の既定反転（#3227。所有者判断）、`capture` 競合の追加是正（§10。所有者判断が要る）。

## 7. 層の発行の下位分解（GB10・5 run）

- 計測: `crates/backend-cuda/tests/infer_chain_issue_breakdown_diag_3205.rs`（公開 trait のみ・`unsafe` なし。record-only）。
  batch 64・784→256＋bias＋ReLU→10＋bias。負荷ゲート: load1 < 1.0 かつ GPU 0%（5/5 通過。`logs/infer-chain-fixed-cost-3205/gb10/breakdown/load_gate.log`）。
- 結果（中央値・µs。出典 `logs/infer-chain-fixed-cost-3205/gb10/breakdown/aggregate.md`。5 run の幅は各区間とも 0.2 µs 以内）:

| 区間 | 中央値 (µs) | 備考 |
|---|---:|---|
| l1_issue / l2_issue | 7.33 / 6.70 | 3189 の 7.5 / 6.6 と整合 |
| alloc（`[64,784]`／`[64,256]`／`[64,10]`） | 3.42 / 3.14 / 2.67 | cuMemAllocAsync＋memset の発行。GPU idle での単独計時 |
| drop（同） | 1.02 / 1.02 / 1.02 | cuMemFreeAsync＋event 記録の発行 |
| 残り（導出。l1／l2） | 3.14 / 2.98 | driver 境界・`cached_gemm` lookup・起動発行・直前層バッファの drop の合算 |

- **再利用で消える見込み**: 中間 h の確保＋解放と最終出力の確保 = 3.14＋1.02＋2.67 ＝ 約 6.8 µs（層の発行の合計 14.0 µs の約 49%）。
  入力バッファの確保・解放は対象外（入力 H2D は #3250）。残りの約 6.1 µs が driver 境界側で、次の候補。
- 限界: 単独計時はチェーン内（カーネル実行と重なる）の値と一致しない。残りは中央値差の導出値で個別には切れない。

## 8. 実装した項目と A/B（GB10・RULE.txt どおり）

- 選んだ理由: 公開 API にも `unsafe` にも触れない項目のうち最大（約 6.8 µs 対 境界側 約 6.1 µs の合算）。
- 実装: `BackendOps::linear_forward_device_into_tracked`（`tensor-core`。既定 `Unsupported` の非破壊拡張）、
  CUDA のオーバーライド（`crates/backend-cuda/src/ops.rs`。memset せず既存バッファへ同じ融合カーネルで書く。
  カーネルは論理形の全要素を無条件に代入し既存値を読まない〈`kernels.rs::gemm_tiled_bias_act_f32`〉ため
  bit 一致する。REQ-8 の手動境界検査は不変）、`DeviceParamStore::predict_device_chain` の層出力スクラッチ
  （`try_lock`・キーは batch 行数と各層の出力列数・成功時のみ保持・`Unsupported` を返すバックエンド〈CPU・Metal〉では保持しない・
  世代切れは破棄して確保経路）。Graph capture 経路には触れない。常駐メモリは層出力 1 回分だけ増える
  （batch 64 で 64×256＋64×10 要素 ≒ 66 KiB）。
- 前後（`run_ab_infer_chain_cuda.sh 3205-gb10-reuse`・5 round・起動順反転・ゲート 1 回通過〈load1 0.08・GPU 0%〉）:

| size/mode | before | after | after/before | run 内比（5 run） | checksum |
|---|---|---|---|---|---|
| 64/reuse | 99.9 µs（99.8–100.0） | 90.3 µs（90.0–90.7） | **0.9038** | 0.9018, 0.9058, 0.9037, 0.9081, 0.9001 | 完全一致 |

  出典 `logs/infer-chain-fixed-cost-3205/gb10/ab/compare-infer-3205-gb10-reuse.md`。
- **判定（RULE.txt）: ADOPT**。条件 1（中央値の比 0.9038 <= 1.00）・条件 2（5 run 中 5 run で after が速い）・
  条件 3（別バイナリ間の bit ダンプ sha256 一致。§9）をすべて満たす。短縮は 9.1 µs で、下位分解の見込み 6.8 µs より大きい
  （チェーン内ではカーネル実行と重なる解放・event 記録の待ちも減ったと推定。推定。未検証）。
- 最初の A/B（`gb10/ab-ungated-first-attempt/`。0.9076）はビルド直後で load1 が高く、ゲートを通していないため判定に使わず invalid 試行として残した。
- 腕の同定: `.rev-stamp`（before=`71321150a`・after=`071a1ddc0`）とツリー指紋の突合（`gb10/arm-identification.txt`。before は全一致、after は GB10 側に比較スクリプトが生成した `.pyc` が 1 件多いだけ）。

## 9. bit 一致

- 実機テスト（`#[ignore]`）`crates/facade/tests/predict_device_chain_cuda_fixed_cost_bit_identity_3205.rs`: batch 64／1024／4096 で、
  1 回目（確保経路）・入力を変えた 2 回目以降（再利用）・別バッチ形状を挟んだ後（作り直し）・旧経路（`forward_resident`）を
  `to_bits()` で全要素比較し、**4 件 pass**（ダンプ 1 件含む）。既存の `predict_device_chain_cuda_bit_identity.rs` も 5 件 pass。
- 別バイナリ間: before（main）と after のツリーで同じ入力のダンプ（640 要素）の sha256 が一致
  （`ac55de61917cf295a86516360489e34567176a91d908012f82e65cc0e6bd8555`。`gb10/bitdump/`）。checksum は根拠にしていない。
- GPU 不要の単体テスト: tensor-core（既定 `Unsupported`）、autodiff の mock（再利用・形状変更時の再確保・`Unsupported` 時の層単位フォールバック・
  世代切れの破棄・`try_lock` 競合時の確保経路・poison 時に保持しない）、backend-cuda（`out` のデバイス・shape 不一致を型付きエラーで拒否）。
  `out` の世代切れは driver 呼び出し内の検査（`begin_driver_call`）で拒否される経路で、実機が要るため実機側の既存検査に委ね、
  新規の実機テストは足していない（未検証の項目として明記する）。

## 10. B3（推論 Graph capture）の再判定（GB10）

- 規則: `logs/infer-chain-graphcapture-rejudge-3205/RULE.txt`（2115 の RULE を継承。**計測前に単独でコミット**）。label は `3205-b3-rejudge`。
- R0（`linear_forward_device_real_device`・`graph_capture_real_device`）: 合格。R1(a)（`INFER_CAPTURE_R1: PASS` が 5 行・SKIP 0）: 合格。
  R2（OFF/ON のダンプ diff。batch 64／1024／4096）: 3 件とも全行一致。
- **R1(b)（`predict_device_chain_cuda_bit_identity` を ON で並列実行）: 不合格（1 回目の実測は 5 件中 3 件失敗。`gb10/steps.log`）**。失敗は
  `Backend(DeviceContextCaptureInProgress { ordinal: 0 })`。bit 不一致ではない。
  - 原因 1（本 PR で是正）: 1 回目の実測で失敗は `predict_resident`（同テスト 75・267 行）で起きていた。当初の是正（100 ms・再試行 1 回）は、
    複数スレッドが続けて capture する並列実行で不足した。全体の待ち上限を 1 s にして上限内で繰り返す形へ改めたところ、
    **`predict_resident` の失敗は 8 反復で 0 件になった**（`gb10/r1b-final-code-8iters.log`。5 s では 1/6 反復が全件 pass、
    100 ms のままでは 6 反復とも 4 件失敗。調整中の試行で、判定用の記録ではない）。
  - 原因 2（本 PR の対象外。所有者判断）: 8 反復で 1 反復だけ全件 pass、残りは同テストの**旧経路側**
    （90 行 `init_device_param_store`＝重みの upload、96 行 `forward_resident`＝tape 経路）が同じエラーで失敗した
    （12 件）。これらは推論チェーンではなく、別スレッドの capture 中に driver を呼ぶ一般の経路で、
    `begin_driver_call` 全体を待ち型にしない方針（学習 step がこのエラーで直接実行へ落ちる契約）と衝突するため手を入れていない。
- 性能（6 セル・5 run・ゲート通過〈ゲートは 3 サンプル busy の後に通過〉。`logs/infer-chain-graphcapture-rejudge-3205/gb10/compare-infer-graph-3205-b3-rejudge.md`）: 全 6 セルで ratio <= 1.00
  （64/fresh 0.944・64/reuse 0.948・1024/fresh 0.968・1024/reuse 0.972・4096/fresh 0.990・4096/reuse 0.994）、checksum 完全一致。
  64/reuse は run 4 の run 内比が 0.9962 と 1.00 に近いが、全セルで 5 run すべて 1.00 未満（`符号一貫（全 run >1.00）` はいずれも「いいえ」）。
- **判定: undetermined**（RULE: ADOPT は R1 合格が条件。R1(b) の失敗は bit 不一致ではないため REJECT でもない）。`INFER_GRAPH_DEFAULT_ENABLED` は変えていない（反転は #3227）。
- 所有者判断が要る点: 別スレッドの capture 中の他の driver 呼び出し（upload・store 初期化・tape 経路の forward）を待ち型にするか、
  R1(b) の事前登録を「並列実行しない」へ変えるか。いずれも本 PR では実装・変更しない。
- 注意: A/B・R2 などの B3 の実測は待ちを 100 ms・1 回再試行の版で取った（待ちは capture 競合時にしか働かず、性能セルと R2 の経路は同一）。
  待ちを改めた版で再実測したのは R1(b) だけ（上記）。

## 11. 相手（PyTorch・candle）との比（同一セッション・record-only。合格ラインなし）

出典 `logs/infer-chain-fixed-cost-3205/gb10/vs-peers/`。infer CUDA batch 64・5 run・ゲート通過（load1 0.69・GPU 0%。90 秒待機）。fandhe-ai は reuse、candle は fresh（reuse なし）。

| 対象 | 中央値 (µs) | 5 run |
|---|---:|---|
| fandhe-ai before（main） | 99.0 | 98.9, 99.0, 99.0, 100.0, 99.0 |
| fandhe-ai after（本 PR） | 89.8 | 89.8, 90.0, 89.7, 90.8, 89.8 |
| candle 0.11.0 | 41.6 | 41.8, 40.4, 43.3, 41.4, 41.6 |
| PyTorch 2.14.0+cu130 | 39.7 | 39.5, 39.7, 40.3, 39.7, 40.4 |

- 比（fandhe-ai / 相手）: before は candle 2.38 倍・PyTorch 2.49 倍、after は candle 2.16 倍・PyTorch 2.26 倍。**差は閉じていない**。
  残りの大半は kernel_wait（§3。固定費ではない）・入力 H2D（#3250）・driver 境界側（§7 の残り約 6.1 µs）。
- 注意: before の 99.0 µs は A/B（99.9 µs）と run 間で 1 µs 動く。bench-fandhe の infer の checksum は candle・PyTorch と定義が異なる（before／after 間では一致。本件では未調査）。

## 12. 限界

- §7 の単独計時はチェーン内の値と一致しない。導出値（残り・kernel_wait）は公開 API から個別に切れない。
- A/B・相手比とも GB10 1 台・1 セッション。GB10 には他プロセスの GPU メモリ使用があった（utilization は 0〜数 %）。
- Metal・CPU では再利用は効かない（`Unsupported` を返し、保持もしない）。
