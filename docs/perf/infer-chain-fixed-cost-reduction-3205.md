# 推論チェーン（infer CUDA reuse）の固定費の削減表（イシュー #3205）

## 1. 目的と結論

- GB10・v0.11.0 の infer CUDA reuse は fandhe-ai 約 0.099 ms、PyTorch 約 0.040 ms、candle 約 0.042 ms
  （`infer-reuse-phase-breakdown.md` §12）。差は約 55 µs。相手も同じ演算と同期 1 回を含む。
- 本 doc は「相手が払っていない固定費」を項目ごとに分け、公開 API の追加も新規 `unsafe` も無しに
  削れるものだけを 1 項目 1 PR で削るための表である。**差を閉じるものではない**。
- **実測状況（本 PR 時点）: GB10 での実測は未実施（未検証）**。本 PR は削減表・事前登録規則
  （`logs/infer-chain-fixed-cost-3205/RULE.txt`）・capture 競合の是正（コードと GPU 不要の単体テスト）
  までで、層の発行の下位分解・B3 再判定・前後比較・相手との比は GB10 で取る（§5）。

## 2. 削減表

出典はすべて `infer-reuse-phase-breakdown.md` §12（#3189。GB10・中央値）。「次の PR」は未着手。

| 項目 | 実測 (µs) | 扱い | 理由・行き先 |
|---|---:|---|---|
| tape・snapshot・steps | 約 0.2 | 削らない | 計測ノイズ以下 |
| facade の residual（`predict_resident` のガード・`tape_for` 等） | 2.6 | 次の PR（調査のうえ判断。小） | GB10 での個別計測後に判断 |
| 入力 H2D（h2d_upload） | 11.3 | 公開 API が要る | #3250（常駐入力ハンドル。#3249 の決定待ち） |
| 層の発行の超過分（l1 7.5 / l2 6.6 − launch 下限 1.82×2） | 約 10.5 | 削れる候補（次の PR） | 下記 3 項目に分けて GB10 で測り、大きい順に 1 項目 1 PR |
| └ 中間・出力バッファの確保と解放（alloc 3・free 3／step） | 未測 | 削れる候補 | 内部 trait に `_into` 系を足して再利用（公開 API ではない） |
| └ 層 1 回分の driver 呼び出し境界の重複（3〜4 回） | 未測 | 削れる候補 | `linear_forward_device`（`crates/backend-cuda/src/ops.rs`）の境界を統合。poison・世代検査の範囲は保つ |
| └ 出力の memset（`alloc_zeros`） | 未測 | 別 issue | 未初期化確保は `unsafe` か pool の結線（`pool.rs` 冒頭がスコープ外と明記）が要る |
| └ cudarc の event tracking（wait_event 6／step） | 未分解 | 削らない（本 issue） | 無効化は既存の `unsafe disable_event_tracking` と既定反転を伴い全 CUDA 利用者に効く。#3227 側 |
| D2H＋同期（download_again） | 6.1 | 削れない | tiny_roundtrip 6.14 µs とほぼ同じで下限 |
| kernel_wait | 63.2 | 範囲外（固定費ではない） | §3 |
| 推論 Graph capture（B3） | reuse で −14 µs（batch 64。#3190） | 是正済み・再判定は GB10 待ち | §4 |

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
  `context_cache::wait_foreign_capture_idle`）で上限 100 ms 待ち、チェーン全体を 1 回だけやり直す。
  - `begin_driver_call` 全体は待ち型にしない（学習 step 側はこのエラーで直接実行へ落ちる契約のため）。
  - 自スレッドが `CallToken` を保持している場合は待たない（`begin_capture_session` の drain との
    相互待ちを防ぐ）。期限切れは従来のエラー（fail-closed）。poison・Retiring・世代の検査は弱めない。
  - store を書き換えないため、やり直しで部分結果は漏れない。
- 単体テスト（GPU 不要）: `context_cache` の `wait_foreign_capture_idle_*` 5 件、
  `device_store` の `predict_device_chain_*capture*` 系 3 件、`tensor-core` の既定実装 1 件。
- 再判定（R0〜R2・6 セル A/B）は新しい label で GB10 にて行う。ADOPT でも既定は変えない（#3227）。

## 5. GB10 で取る残り（未実施）

1. 層の発行の下位分解（確保＋memset／境界・lookup／launch／解放）を 5 run で測り §2 の「未測」を埋める。
2. B3 再判定（RULE.txt 継承）。
3. §2 の最大の「削れる候補」が RULE.txt の採用条件を満たす見込みのときだけ 1 項目を実装し、before/after を 5 round で記録する。
4. 同一セッションで PyTorch・candle との比を record-only で記録する。

## 6. 対象外

- kernel_wait（小 m のタイル占有率）、memset の省略（`unsafe` か pool の結線）、event tracking、
  入力 H2D（#3250）、バッファ再利用、residual の削減。
