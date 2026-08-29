# ピークメモリ係数の確定（#179・TASK-14.2b）

イシュー #179「docs(backend): TASK-14.2b 係数確定・記録（超過時は再調整）」に対応する。
TASK-14.2（`docs/spec/05-tasks.md` REQ-14 対応表）のうち #179 が担当する「係数確定・記録
（超過時は再調整）」を満たすための決定記録である。判断材料は先行イシュー #178
（TASK-14.2a）の実測記録 `docs/perf/gemm-peak-memory-measurement.md`（および生データ
`docs/perf/peak-memory/cpu-run1.json`・`cpu-run2.json`）であり、同記録「スコープ外・
申し送り」節（`docs/perf/gemm-peak-memory-measurement.md:208-209`）が「兄弟イシュー #179
（TASK-14.2b）が本記録を入力として実施する」と明記している。

## 判定

**超過なし。REQ-14 の係数上限（2 倍以内 = 384MiB 以内）を再調整せず、係数 2.0 のまま確定する。**

REQ-14 の受け入れ基準（`docs/spec/04-requirements.md:275`）が定義する判定対象は「内部計測
API が返すピークメモリ」であり、`docs/perf/gemm-peak-memory-measurement.md` の CPU 実測結果
（下記「根拠」節）はこの判定対象について理論最小ワーキングセット（≈192MiB）を超過していない。

## 判定の位置づけ（承認経路）

REQ-14 の係数上限はガードレール系の設計目標値であり、値の変更（緩和・厳格化）は
`.claude/rules/security.md`「自己修復ループ固有のガードレール」が定める人間承認事項に
該当しうる。本決定は**値を変更しない（2.0 を維持）**判断であるため、単独では承認必須事項
には該当しない。ただし本イシューは自動運転モードで作成しており、本決定を含む PR の
レビュー・マージ（人間）をもって正式承認とみなす（`docs/short-lived-process-decision.md`
の運用を踏襲）。CUDA・Metal 実機実測の転記後に再判定した結果、係数超過が判明し再調整が
必要になった場合は、その時点で改めてユーザー承認を得ること。

## 根拠

### 1. 実測データ（CPU、#178 実測記録より）

`docs/perf/gemm-peak-memory-measurement.md`「CPU 実測結果」節（`:90-104`）:

| セット | peak_bytes（内部計測 API、中央値） | 理論最小ワーキングセット | 対理論比 |
|--------|-----------------------------------|--------------------------|---------|
| run1 | 201,326,592（192MiB） | 201,326,592（192MiB） | 1.000 |
| run2 | 201,326,592（192MiB） | 201,326,592（192MiB） | 1.000 |

- 内部計測 API のピーク値は 5 試行 × 2 セット（計 10 trial）すべて 201,326,592 バイト
  （192MiB）ちょうどで決定的に一致し、理論最小ワーキングセットと完全一致した（対理論比
  1.000）。384MiB（対理論比 2.0）に対し十分な余裕がある
- drop 後の `allocated_after_drop_bytes` は全 10 trial で 0（リークなし）
- 生データ: `docs/perf/peak-memory/cpu-run1.json`・`cpu-run2.json`

### 2. 判定対象の定義（`VmHWM` は参考値であり判定対象外）

REQ-14 の受け入れ基準（`docs/spec/04-requirements.md:275`）は「**内部計測 API が返す
ピークメモリ**」を判定対象と明記しており、`docs/peak-memory-measurement-methods.md`
「方針: 内部計測 API を主軸、外部計測は補助的な裏取り」節（`:107-119`）も「数値の正本は
内部計測 API の値とする」「外部計測は補助的な裏取りに留める」と定めている。

このため `VmHWM`（プロセス全体のピーク常駐セットサイズ、参考値。run1 約 330.3MiB・run2 約
330.4MiB、対理論比 約 1.72）は判定対象外である。ただし「留意点」節（後述）でこの値を
記録し、係数の実質的な余裕についての判断材料として残す。

### 3. 設計契約との一致

REQ-14 の係数上限（2 倍以内）は、PoC-v2-3／PoC-v2-4 で確認した「A・B・C の 3 バッファを
直接確保しプール・キャッシュを介さない」自作アロケータの設計から導いた設計目標値である
（`docs/spec/04-requirements.md:275`）。CPU 実測値（対理論比 1.000）は、`MemoryOps` 経由の
確保が A・B・C の 3 バッファのみであり他の中間確保を一切含まないという実装契約どおりの
結果であり（`docs/perf/gemm-peak-memory-measurement.md:100-104`）、設計時の想定と実測が
一致したことを示す。

### 4. プール導入時の係数維持テストが整備済み（#202・PR #363）

REQ-14 の受け入れ基準「最適化後（バッファプール等のキャッシュ機構を導入する場合）」
（`docs/spec/04-requirements.md:276`）に対応し、プール明示解放 API・係数維持テスト
（`crates/backend-cpu/tests/pooled_memory_integration.rs` の
`coefficient_stays_within_2x_for_repeated_same_shape_workload`）が既に整備済みである
（`docs/memory-pool-design.md:129-139`）。プール導入時も本係数（2.0）の維持が機械的に
検証可能な状態にある。

## 留意点

内部 API 値（対理論比 1.000）に対し、`VmHWM` 参考値は対理論比 約 1.72 に達している
（`docs/perf/gemm-peak-memory-measurement.md:113-151`「内部 API 値と外部参考値の乖離」節）。
このうち約 71.6MiB（対理論比 換算で約 0.37 分）は `gemm_alloc_peak_bytes`（GEMM 実行区間中
の実ヒープ確保量の純増分ピーク）として GEMM 実装内部の一時確保であることが分解済みだが、
残り約 66.7〜66.8MiB は未分解のままである。

REQ-14 の判定対象（内部計測 API 値）では対理論比 1.000 で係数上限に対し十分な余裕がある一方、
プロセス全体のピーク常駐という実態で見ると係数の余裕は内部 API 値が示すほど大きくない
可能性がある。この乖離自体は本決定の判定を変えるものではない（判定対象は内部計測 API 値と
定義されているため）が、将来 CUDA・Metal 実機実測やプール導入判断を行う際の判断材料として
記録する。

## 再判定トリガー

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機実測の転記が完了した時点で、本判定を
再確認することとしていた（`docs/short-lived-process-decision.md` と同型の再判定トリガー
方式）。**Metal は #385・CUDA は #392 でそれぞれ実機実測が完了し、3 バックエンドすべての
再判定トリガーを消化済みである**（下記「事実追記」参照）。

再判定の結果、CUDA・Metal のいずれかで対理論比が 2.0 を超えることが判明した場合は、
`.claude/rules/security.md` の承認要件に従いユーザー承認を得たうえで係数の再調整を検討する
（今回はいずれも超過なし。詳細は「事実追記」参照）。

**事実追記（イシュー #385）**: Metal（Apple Silicon）実機実測が完了し、対理論比は 1.000
（`docs/perf/gemm-peak-memory-measurement.md`「Metal 実機実測結果」節）であり超過なし。
係数 2.0 は無変更のまま維持する。

**事実追記（イシュー #392）**: CUDA（DGX Spark GB10）実機実測が完了し、対理論比は 1.000
（`docs/perf/gemm-peak-memory-measurement.md`「CUDA 実機実測結果」節。run1・run2 とも
201,326,592 バイトで理論最小と完全一致）であり超過なし。`vm_hwm_bytes` は Linux 実機の
ため全 trial で non-null が実測できたが、GB10 の統合メモリではデバイス確保が RSS に
計上されるか断定できないため参考値に留め対理論比算出には用いていない（同記録「計測境界
の限界（CUDA 固有）」節）。CPU・Metal・CUDA の全 3 バックエンドで実測が完了し、いずれも
係数超過なしとなったため、係数 2.0 は無変更のまま維持し、下記「確定の範囲と条件」を
条件付き確定から確定へ更新する（ユーザー承認必須の対象は未発生）。

## spec リポ側への申し送りメモ

`docs/spec/05-tasks.md`「Phase 4 への逆戻り提案」節（`:541`）は「TASK-14.2 の係数実測結果は
REQ-14 受け入れ基準への反映が必要」と定めている。本決定に基づく Fandhe-AI/fandhe-ai-spec
側での反映文案は以下のとおり（`docs/spec/` は本リポでは編集しない。反映は spec リポ側で
人間が実施すること）:

> REQ-14 の係数上限（初期リリース 2 倍以内 = 384MiB 以内）について、TASK-14.2（#177〜#179）
> による実測の結果、CPU バックエンドでは内部計測 API のピーク値が理論最小ワーキングセット
> と完全一致（対理論比 1.000）し、係数超過は確認されなかった。係数は 2.0 のまま変更不要と
> 判断し確定した（実装リポ `docs/peak-memory-coefficient-decision.md` 参照）。その後、Metal
> （Apple Silicon・#385）・CUDA（DGX Spark GB10・#392）の実機実測も完了し、いずれも対理論比
> 1.000 で係数超過は確認されなかった。CPU・Metal・CUDA の全バックエンドで実測に基づく確定
> となっている。

## 確定の範囲と条件

**CPU・Metal・CUDA いずれも実機実測に基づき確定した（係数 2.0 を維持。超過なし）。**

- **CPU バックエンド**: 実測に基づき確定（対理論比 1.000、係数 2.0 を維持）
- **Metal バックエンド**: イシュー #385 の実機実測に基づき確定（対理論比 1.000、係数 2.0 を維持）
- **CUDA バックエンド**: イシュー #392 の実機実測に基づき確定（対理論比 1.000、係数 2.0 を維持。
  `vm_hwm_bytes` は参考値に留め判定には用いていない）

## スコープ外

- **プールの既定有効化の構成判断**: `docs/memory-pool-design.md`「プールの既定有効化」節
  （`:137-139`）が申し送り済み（#202 系）。本決定では扱わない
- **CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機実測の実施自体**: それぞれ #392・
  #385 で実施済み（上記「事実追記」参照）
- **`VmHWM` 残差（約 66.7〜66.8MiB）の内訳分解**: `docs/perf/gemm-peak-memory-measurement.md`
  「スコープ外・申し送り」節（`:219-221`）が申し送り済み。本決定では扱わない

## 参考

- `docs/perf/gemm-peak-memory-measurement.md`（#178・TASK-14.2a 実測記録。#385・#392 で
  Metal・CUDA 実機実測結果を追記済み）
- `docs/perf/peak-memory/cpu-run1.json`・`cpu-run2.json`（CPU 生データ）
- `docs/perf/peak-memory/metal-run1.json`・`metal-run2.json`（Metal 生データ。#385）
- `docs/perf/peak-memory/cuda-run1.json`・`cuda-run2.json`（CUDA 生データ。#392）
- `docs/peak-memory-measurement-methods.md`（#180・TASK-14.3 計測手段の環境差文書化）
- `docs/memory-pool-design.md`（プール導入時の係数維持設計）
- `docs/spec/04-requirements.md`（REQ-14）
- `docs/spec/05-tasks.md`（TASK-14.2・Phase 4 への逆戻り提案節）
