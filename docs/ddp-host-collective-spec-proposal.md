# DDP 初期形（ホスト側 collective）の spec 改訂の根拠と記録（#3142・fandhe-ai-spec#85 反映済み）

> 本 doc は「未起票の提案」ではなく、**spec へ反映済みの改訂の根拠と記録**である。spec 改訂は Fandhe-AI/fandhe-ai-spec#85（spec コミット `aaeb62c`）でマージ済みで、`docs/spec` のポインタ更新は本リポ PR #3275 で取り込み済み。正本は `docs/spec`（編集しない）であり、本 doc は根拠の記録に徹する。

- 基準コミット: origin/main `1bc2941b`（作業時点）。`docs/spec` ポインタ `7534c9c93bd4443bc13b1a6ab91c7a2db4c2e033`
- 参照する spec の行（`docs/spec/04-requirements.md`）: `:363`（除外事項「分散学習・量子化の網羅対応」）・`:377-385`（DDP 格上げ条件表。2026-10-08 追記）・`:386-393`（初期形。2026-10-12 追記）・`:480`（改訂履歴）。行番号は submodule 更新でずれうるため、参照時に再確認すること
- コード変更なし。本 doc は後続 #3143（ホスト側 collective の設計記録）が要件として引く（§6）

## 0. 結論

- spec 改訂は #85 で反映済み。本 issue の成果物は、起票用の文案ではなく反映済み改訂の根拠と記録である
- **Could への格上げは宣言していない**（`:393`）。Won't 判断・スコープ件数（Should 8・Could 1・Won't 11）・承認時に固定する契約・REQ-2／REQ-7／REQ-8（tolerance・baseline を含む）は不変
- `docs/ddp-grade-up-conditions.md` §0／§4／§5 と `docs/facade-multi-gpu-ddp-decision.md` §5／§8 にある「未起票・未承認」「段階 0」の記述は、**初期形の範囲に限り**本 doc と spec の `:386-393` が置き換える。既存 doc は基準コミット固定の記録なので書き換えない

## 1. 位置づけ・記録の連鎖

| 時期 | 出来事 | 出典 |
|---|---|---|
| 2026-09 | 段階 0（現時点非対応）確定・案 A〜E 比較 | #1628（`docs/facade-multi-gpu-ddp-decision.md`） |
| 2026-10-08 | DDP 格上げ条件表を spec へ追記（NCCL・複数 CUDA ordinal 前提） | #2074・#2612（`docs/ddp-grade-up-conditions.md`。spec `:377-385`） |
| 2026-10-11 | 所有者判断 D-B「DDP は実装へ進める」 | #3018 issuecomment-6111643437 |
| 2026-10-12 | 所有者が提案 issue の起票・承認待ちを省き、spec を直接改訂する旨を指示 | #3142 issuecomment-6111887767 |
| 2026-10-12 | spec 改訂（初期形の追記） | Fandhe-AI/fandhe-ai-spec#85（`aaeb62c`） |
| — | 実装リポの spec ポインタ更新 | PR #3275 |
| 後続 | ホスト側 collective の設計記録 | #3143 |

## 2. 手元の機材の制約

- 実機は NVIDIA DGX Spark GB10（GPU 1 基）と Mac mini（Apple M4）の 2 台（spec `:386`）。単一ノード内に複数 CUDA ordinal を持つ機材は無い
- GB10 の NCCL 版数・ordinal 数は未プローブ（`docs/perf/logs/ddp-nccl-link-contract-2074/README.md` の申し送り）
- `docs/ddp-grade-up-conditions.md` §2.2 の RTX 3060 はエージェントの実行環境であり、「手元の機材」には数えない
- このため条件表 (c)(d) を NCCL 前提のまま満たす検証に着手できなかった。これが初期形を定めた理由である

## 3. 条件 (a)〜(g) の現状

| 条件 | 内容 | #85 以前 | #85 後（初期形での扱い） | 出典 |
|---|---|---|---|---|
| (a) | cudarc `nccl` feature が `dynamic-loading` 下で link-time の `libnccl` を要求しない | 達成済み | 変更なし | #2074 |
| (b) | facade 公開面にデバイス列挙 API | 達成済み | 変更なし | #1614（`fandhe_ai::available_devices()`） |
| (c) | ネットワーク層方式の確定と個別ユーザー承認 | 未達 | 初期形に限り案 C（std のみの TCP ring all-reduce）で確定し、承認は所有者判断で充足。**NCCL 経路を含む全体としては未達のまま** | `:387` |
| (d) | 実機側の分母（複数 CUDA ordinal・NCCL 版数のプローブ） | 未達 | 分母に「単一ノード複数プロセス（バックエンド不問）」を追加。NCCL 側の分母は未達のまま | `:388-389` |
| (e) | 依存追加なしで all-reduce 後勾配の数値一致検証が成立する設計 | 未達 | 未達。同一 GPU を共有する複数プロセスで数値一致を検証する道筋ができた（設計は #3143 以降） | `:388` |
| (f) | スケーリング効率の実測 | 未達 | 未達。複数プロセスの分母では測れない | `:388` |
| (g) | REQ-12 と整合する facade 公開面の非破壊拡張 | 未達 | 未達 | `:382` |

## 4. 案 A〜E の比較と初期形の関係

案の定義は `docs/facade-multi-gpu-ddp-decision.md` §4 を正とし、ここでは要約する。

| 案 | 概要 | 新規依存 | 難度 | 初期形との関係 |
|---|---|---|---|---|
| A | NCCL（cudarc `nccl` feature） | feature 列挙の拡張が承認事項 | XL | 据え置き（`:389`）。複数 ordinal 機の確保が条件（#2613） |
| B | 汎用分散通信ライブラリ（gloo／MPI 等） | 新規依存区分が必要 | XL | 不採用（新規区分は追加しない契約） |
| C | 自作 TCP ring all-reduce（std のみ） | なし | XL | **採用**。依存追加ゼロで (e) と整合 |
| D | 単一ノード複数 GPU 限定（P2P／host 経由） | なし | L〜XL | 不採用。GPU 1 基では検証の分母を作れず、バックエンドとも非対称 |
| E | 段階 0（非対応と明文化） | なし | — | 初期形の範囲で置き換わる |

案 C の弱点（rank 発見・障害耐性・性能）は、初期形が正しさ検証に目的を限ることで受容する。性能は (f) として未達のまま残る。

## 5. spec に入った初期形の内容（`:387-390`）

1. **(c) 方式の確定**: 初期形に限り案 C。新規クレートなし・cudarc の feature 追加なし
2. **(d) 分母の追加**: 単一ノード複数プロセス。GPU 1 基の機材では同一 GPU を共有する複数プロセスで (e) の数値一致（正しさ）を検証する
3. **NCCL 経路の据え置き**: cudarc `nccl` の追加と NCCL 呼び出しコード（#2613）は、複数 CUDA ordinal を持つ実機の確保を条件に据え置く
4. **取り決めの緩和**: 初期形（ホスト側 TCP collective・勾配平均の optimizer ステップへの結線・opt-in の公開面の非破壊な追加・単一ノード複数プロセスでの検証）は REQ 承認前でも起票・実装してよい。NCCL 呼び出しコードは従来どおり不可

## 6. 初期形の条件（`:392`。#3143 以降が要件として引く）

| # | 条件 | OWASP |
|---|---|---|
| 1 | ネットワーク待受は既定で loopback に限り、loopback 以外は明示 opt-in とする | A05 |
| 2 | 受信メッセージの長さは上限を設け、**バッファ割り当ての前に**検査する。ランク番号を検証してから処理する | A03 |
| 3 | 受信にタイムアウトを設ける | A04 |
| 4 | ランク番号の重複と想定外の接続を拒否する | A04 |
| 5 | 通信は勾配 all-reduce のための固定メッセージに限り、汎用の分散 RPC（REQ-9「引き続き対象外」）は提供しない | A04 |
| 6 | 「承認時に固定する契約」を引き継ぐ: REQ-2・REQ-7 の tolerance と REQ-8 下限の不変／手動境界チェックの維持／許容依存区分外の新規区分なし／公開 API の非破壊（追加・opt-in・`#[non_exhaustive]`）／`build-no-cuda-toolkit` 不変条件 | A08 |

## 7. 不変事項・スコープ外

- FSDP／ZeRO・tensor／pipeline／model parallel は対象外のまま（`docs/ddp-grade-up-conditions.md` §3a）。分散 RPC も対象外
- NCCL（#2613）の deps-policy 追加は承認されていない
- REQ-2 の tolerance・baseline・FMA 契約、依存ピン、`docs/spec` は変更しない
- 事実の記録: spec (c) の文言は「許容依存は現行 10 区分」だが、本リポの deps-policy は 2026-10-16 に第 11 区分（`ureq`。用途は facade の HTTPS 取得に限定）を新設した。`ureq` は collective に使えず、ネットワーク／分散通信の区分が無いという趣旨は変わらない。spec の修正要否は本 issue の範囲外

## 8. セキュリティ観点

- A03（入力検証）: 受信長を割り当て前に検査し、ランク番号を検証してから処理する（§6 の 2）
- A04（DoS 耐性）: 受信タイムアウト・重複ランクと想定外接続の拒否・固定メッセージへの限定（§6 の 3〜5）
- A05（既定値）: loopback 既定、非 loopback は明示 opt-in（§6 の 1）
- A07／A08: 初期形は認証機構を持たず、loopback 既定がその前提になる。非 loopback の opt-in 時はなりすまし対策が要る。この論点は `docs/facade-multi-gpu-ddp-decision.md` §6 のセキュリティ留意点を引き継ぐ

## 9. 出典一覧

- spec: `docs/spec/04-requirements.md:363`・`:377-385`・`:386-393`・`:480`
- `docs/ddp-grade-up-conditions.md` §2〜§4・§3a
- `docs/facade-multi-gpu-ddp-decision.md` §4・§6
- `docs/perf/logs/ddp-nccl-link-contract-2074/README.md`
- #3018 issuecomment-6111643437・#3142 issuecomment-6111887767
- Fandhe-AI/fandhe-ai-spec#85・PR #3275・#2613・#3143
