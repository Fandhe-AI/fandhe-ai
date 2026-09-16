# Metal `mse_loss_backward` encode-only 化の M4 Max A/B（イシュー #1691）

親 #1582（MSE backward の Metal encode-only／CUDA ストリーム化）の
測定系 sub-issue。実装系 #1690（PR #1785。squash `38b72b1f`・親
`84490ad1`）で導入した `crates/backend-metal/src/mse.rs` の 3 箇所の
`ctx.dispatch_sync`（forward `mse_partial_f32`／`mse_finalize_f32` 2
段リダクション・backward `run_mse_backward_f32` 1 段）の `ctx.encode` +
`DispatchFailureCell` 登録化を、M4 Max 実機で REQ-2 複合判定・
非後退（5 run 中央値・checksum 完全一致・ratio<=1.00）の事前登録規則で
確認する。

## 1. コード読解で確定した事実（実測ではない）

- `git diff --stat 84490ad1 38b72b1f -- crates/` は `mse.rs` のみを
  差分として含む（他はテスト doc comment・設計文書）。
- **同期回数の内訳**（`docs/backend-metal-command-batching-design.md`
  §7.5.1・§7.5.2）:
  - forward `run_mse_loss_f32`: 旧実装は 2 段リダクション
    （`mse_partial_f32`／`mse_finalize_f32`）がそれぞれ独立に
    `dispatch_sync`（commit + wait）していたが、#1690 で両方を
    `ctx.encode`（待たない）へ変更し最後に 1 回だけ
    `ctx.synchronize()` するようになった（**2 wait → 1 wait**）。
  - backward `run_mse_backward_f32`: 呼び出し元
    `crates/autodiff/src/grad.rs`（`Op::MseLoss` の VJP）が戻り値
    `dpred` に対して直後に `dense_vec(&dpred)`（ホスト即時アクセス）
    する契約のため、本関数は encode-only 化後も関数内で**必ず自ら**
    `ctx.synchronize()` する（**1 wait → 1 wait。不変**）。
  - したがって 1 step 全体のカウンタ見積り（`mnist_scale_train_reuse_
    metal_batch_counters`）は #1566 適用後の 11/8/8 から #1690 適用後
    **11/7/7** へ変わる見積り（forward の −1 command_buffer・−1
    wait）。backward 限定窓（`mnist_scale_train_reuse_metal_backward_
    dinput_phase`）の仮説（encode_delta=5／command_buffer_delta=3／
    wait_delta=3）は**変わらない**。
- **他 issue との二重主張回避**（§7.5.3）: `Op::LinearResident` の
  d_input（#1561/#1562/#1563）・bias 勾配（#1564/#1566）側の同期境界
  は本 issue では変更しておらず、それらの待ちについて本 issue は何も
  主張しない。
- 動機となった数値（backward の loss 項が M4 Max 共有負荷下 67〜124
  µs。`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4）は
  `FANDHE_DIAG_BACKWARD=1` の**診断専用計装パッチ**由来で本番コードで
  は取れない値であり、本 issue の判定には用いない（動機の記録のみ）。

## 2. 事前登録判定規則

issue コメント
<https://github.com/Fandhe-AI/fandhe-ai/issues/1691#issuecomment-5656028993>
に固定済み。要約:

- **対象**: `crates/facade/tests/mse_backward_bench.rs::
  mse_backward_cases`（`FANDHE_BENCH_DEVICE=metal`）・framework-compare
  train Metal（`size=64 / reuse`）・カウンタテスト・parity テスト・
  既存 `#[ignore]` 群。
- **比較腕**: before=`84490ad1`・after=`38b72b1f` 以降の main。
- **計測層**（すべて record_only・専有ゲートなし）:
  - (a) REQ-2 正しさ（`mse_parity.rs::mse_matches_cpu_across_shapes`）
  - (b) bit 同一（`metal_reuse_step_grad_bit_dump`。件数・ラベル集合
    検証込み fail-closed）
  - (c) カウンタ（`mnist_scale_train_reuse_metal_batch_counters` after
    11/7/7 hard assert・before 11/8/8 再現。`backward_dinput_phase` は
    両腕とも 5/3/3 をおおむね再現・record-only）
  - (d) backward マイクロベンチ 5 round・起動順反転・非後退確認
    （事前見通し ≈1.00。改善は狙わない）
  - (e) framework-compare train Metal A/B（**主判定・Tier 1**。reuse
    `step_total` 5 run 中央値比 ≤ 1.00・checksum 完全一致。fresh は
    対照・`--phases` は診断）
  - (f) 既存 `#[ignore]` 群非後退（after 腕）
- **総合判定**: (a)(b)(f) pass かつ (c) 一致かつ (d)(e) すべて ≤ 1.00
  → ADOPT。(e) または (d) に > 1.00 → REJECT（原因帰属は判定と分けて
  記録）。実機到達不能・件数検証失敗 → undetermined。

## 3. 実測記録

**2026-09-16 実測済み → §3.1 以降に M4 Max 実機の実値を記録する（結論:
REJECT。事前登録規則 §2 をそのまま適用）。** 以下の段落はスキャフォールド
作成時（PR 時点）の記述を残したもの。

本エージェント実行環境（Linux コンテナ／worktree）に Apple Silicon
実機への到達手段がないため、**実測値は一切含まれていない**（スキャフォールド
作成時点の記述。2026-09-16 に実測済み）。本文書は
スキャフォールド（実行スクリプト・事前登録判定規則・記入欄）のみを
提供し、実測は Mac セッションへ申し送る（`docs/perf/logs/train-
resident-grad-cuda-1560/`・`docs/perf/logs/cuda-mse-backward-1692/`
と同型の運用）。

実行スクリプト・記入欄は
`docs/perf/logs/metal-mse-backward-1691/`（`README.md`・
`orchestrate.sh`・`aggregate.py`・`env_info.txt`）を参照。

### 記入欄（M4 Max 実機実測後に埋める → 2026-09-16 実測済み）

| 項目 | 値 |
|------|-----|
| 実測日 | 2026-09-16（orchestrate 01:37:07Z〜01:37:55Z・(e) A/B 01:42:25Z〜01:43:36Z。UTC） |
| before_sha | `84490ad1`（git worktree。PR #1785 マージ直前 main） |
| after_sha | `565300e4`（git worktree。origin/main。§3.3 の交絡注記参照） |
| (a) `mse_parity.rs::mse_matches_cpu_across_shapes` | pass（1 passed; 0 failed。`mse_parity_after.log`） |
| (b) bit dump 件数・ラベル集合・diff | 4462/4462 行（両腕）・ラベル集合 diff 0 行・値 diff 0 行（bit 同一） |
| (c) `batch_counters` before/after (encode/cb/wait) | before 11/8/8・after 11/7/7（見積りどおり。after は hard assert pass） |
| (c) `backward_dinput_phase` before/after (encode/cb/wait) | 両腕とも 5/3/3（5 run すべて）。backward-only median before 1.026833 ms／after 0.682667 ms（record-only・非判定） |
| (d) `train_shape/640` median_s ratio (5 run) | **0.9871**（before 0.000142500 s / after 0.000140667 s） |
| (d) `general_shape[16384/65536/1048576]` median_s ratio | **1.0109**（>1.00。before 0.000164375 / after 0.000166166）／0.9510（0.000224667 / 0.000213667）／0.9500（0.001411000 / 0.001340416） |
| (d) `grad[...].fold_bits` 完全一致 | True（全 4 セル・5 run・両腕） |
| (e) `run_ab_mse_encode_metal.sh` reuse step_total 5 run 中央値比 | **0.9618**（before 2.006 ms / after 1.929 ms。run 内比 0.9600, 0.9373, 0.9633, 0.9982, 1.3338。参考 fresh 0.9794） |
| (e) checksum 完全一致 | 完全一致（reuse・fresh とも） |
| (f) 既存 `#[ignore]` 群非後退 | 5 ファイル全 pass（mse_parity 1・mnist 3・command_batching 5・command_batching_bench 2・gemm_resident_parity 2）。`device_param_store_backend_parity` は 1 pass 1 **FAIL**（`grad_readout_contract_on_metal`。§3.4） |
| 結論（ADOPT／REJECT／undetermined） | **REJECT**（(d) 1 セル >1.00。加えて (f) に FAIL 1 件があり「(f) pass」も成立しない。§3.2） |

### 3.1 実測環境・生成物

- Apple M4 Max（論理 CPU 16）・macOS 26.6.2（25G83）・rustc 1.98.1・
  cargo 1.98.1。record_only（専有ゲート機構なし）。計測中の 1 分 load
  average は orchestrate 開始 21.17／終了 23.51・(e) A/B 開始 8.27／終了
  13.36（run 1〜5 中 11.82〜13.36）で、別セッションの workspace 全体テスト
  が並走する**共有負荷下**の計測である（`docs/perf/logs/metal-mse-
  backward-1691/env_info.txt`・`progress_1691_excerpt.txt`・
  `ab/raw/uptime-1691.log`）。pmset は thermal warning なし。
- `orchestrate.sh` は (a)〜(d)(f) を完走したのち、(f) の最終項目
  `device_param_store_backend_parity` の FAIL で `set -eu` により
  **(e) の前に中断**した（rc=101。`orchestrate_stdout.log`・
  `progress_1691_excerpt.txt`。このため `uptime_after.txt` は未生成）。
  (e) は同日 `run_ab_mse_encode_metal.sh 1691` を手動で別途実行し、
  生成物を `ab/`（`compare-train-1691.md`・`compare-train-1691-fresh-
  reference.md`・`run_ab_1691.log`・`raw/`〈JSONL・sha・cargo tree・
  uptime・pmset。バイナリは含めない〉）へ保存した。
- (d) の生ログは `before_round{1..5}.log`／`after_round{1..5}.log`・
  集計は `aggregate.md`（`aggregate.py`。5 round・起動順反転
  〈`rounds.log`〉・別プロセス）。集計値は生ログの per-run 値から手計算
  でも再現することを確認した（例: `general_shape 16384` after の 5 run
  は 0.000162583／0.000169167／0.000166166／0.000160292／0.000184042 s
  で中央値 0.000166166 s）。

### 3.2 判定（事前登録規則 §2 をそのまま適用）

- (a) pass・(b) pass（4462 行 bit 同一）・(c) 一致（before 11/8/8 →
  after 11/7/7。見積りどおり）・(d) `fold_bits` 全一致・(e) reuse 0.9618
  ≤ 1.00 かつ checksum 完全一致。
- しかし **(d) `general_shape 16384` が ratio 1.0109 > 1.00** であり、
  規則「(e) または (d) に > 1.00 → REJECT」に該当する。
- さらに **(f) は pass ではない**（`grad_readout_contract_on_metal`
  FAIL。§3.4）。規則は「(a)(b)(f) pass」を ADOPT の必要条件とするため、
  この点でも ADOPT は成立しない。
- **verdict = REJECT**（規則を事後に緩めない。tolerance／baseline 不変）。
  なお #1690（PR #1785）は既にマージ済みであり、本判定を受けた結線の
  扱い（維持／差し戻し）はユーザー判断事項として本文書では結論しない。

### 3.3 原因分析・参考（判定とは分けて記録）

- (d) の超過は 4 セル中 1 セル・幅 1.1 %（0.000164375 → 0.000166166 s、
  差 1.8 µs）で、共有負荷下（load average 8〜23）の計測ノイズ帯に
  収まる規模である。同セルの per-run 値は before 0.000139458〜
  0.000199917 s・after 0.000160292〜0.000184042 s と run 間の振れ幅
  （約 40 µs）が差分（1.8 µs）を大きく上回る。他 3 セル（0.9871／
  0.9510／0.9500）は改善方向。事前見通し（backward の wait 回数は 1 → 1
  で不変のため ≈1.00）と整合する。
- (e) は主判定の reuse が 0.9618・対照の fresh が 0.9794 でいずれも
  改善方向・checksum 完全一致。診断用フェーズ分解（単発・非判定）では
  reuse の `forward_resident` 0.887 倍・`backward` 1.015 倍・
  `step_total` 0.952 倍、fresh の `forward` 0.908 倍・`backward`
  1.026 倍・`step_total` 0.953 倍で、forward 側（2 wait → 1 wait）の
  改善が step_total の改善を説明し backward は不変という §1 の見積りと
  向きが一致する。ただし reuse の run 5 は 1.3338（他 4 run は
  0.9373〜0.9982）で符号一貫ではない。
- **交絡注記（after 腕）**: after 腕は `565300e4`（origin/main）で
  あり、`git diff --stat 84490ad1 565300e4 -- crates/*/src` は 117
  files changed（`mse.rs` 209 行に加え #1885〜#1889 等の後続マージを
  含む）。§1 の「差分は `mse.rs` のみ」は `84490ad1..38b72b1f` に
  ついての事実であり、本 A/B の差分を #1690 単独へ帰属させる分離は
  できていない。size=64 train 経路（Linear／MSE／reuse）が後続マージで
  変わったかは本文書では確認していない（要確認）。
- 以上は原因帰属の参考であり、§3.2 の verdict を変更しない。

### 3.4 (f) `grad_readout_contract_on_metal` FAIL（main 既存・本 issue 対象外）

- `crates/facade/tests/device_param_store_backend_parity.rs:239` の
  assert「param 1（bias）の resident 充填状態が期待と異なる」で FAIL
  （`ignored_after_device_param_store_backend_parity.log`。同ファイルの
  `device_resident_matches_host_sgd_on_metal_across_100_steps` は
  pass）。`--test-threads=1` の直列実行でも同じ assert で FAIL を再現
  した（`ignored_store_parity_serial.log`。01:43:36Z〜01:43:40Z）。
- 本テストは after 腕（origin/main `565300e4`）上で再現した FAIL で
  あり、`mse.rs` の encode-only 化とは無関係な bias 勾配 resident
  充填契約（#1566 系）の検査である。before 腕 `84490ad1` で同テストが
  存在・pass するかは未確認のため「main 上で再現・原因は本 issue
  対象外」とのみ記録する（要確認: 起票はユーザー承認後。
  `.claude/rules/out-of-scope-tracking.md`）。

### 3.5 隣接コミット比較による補足（2026-09-16 追記・判定は §3.2 のまま不変）

§3.3 の after 腕交絡（after=origin/main は #1690 以外の差分を含む）を切り分けるため、同日に
before=`84490ad1`／after=`fdec66a0`（#1690 マージコミット）の隣接コミット比較で (d)(e) を再実行した
（`docs/perf/logs/metal-mse-backward-1691/adj/`。record_only・1 分 load average 約 9〜14、5 分 約 17〜19 の共有負荷下）。

- (d) `mse_backward_cases` 5 round 交互（`adj/aggregate.md`）: 4 セルすべて ratio > 1.00
  （train_shape 640: 1.0657・general 16384: 1.1085・65536: 1.0250・1048576: 1.0234）。`fold_bits` は全一致
- (e) framework-compare train（`adj/compare-train-1691adj*.md`）: reuse ratio 0.9561（run 内比 0.8489〜1.2006・
  before spread > 1.5 倍でスクリプトが負荷ノイズの疑いを付記）・fresh 0.9818・checksum 完全一致
- 結論: 隣接コミット比較でも (d) は規則を満たさず、§3.2 の **REJECT** は不変。(d) と (e) の符号が
  一致しない点は共有負荷下のノイズが判定に影響している可能性を示すが、規則の事後緩和は行わない。
  専有環境での再計測が必要なら別途ユーザー判断で実施する

## 4. スコープ外事項

- CUDA 側は #1692 で分解済み・別スコープ（コード変更なし）。
- `dpred` のホスト往復残存は `BackendOps::linear_forward_device` 系の
  デバイス常駐チェーン拡張（#1216／#1673 の設計）と関係する別範囲で
  あり、本 issue のスコープ外とする
  （`.claude/rules/out-of-scope-tracking.md` に従いユーザー承認なしに
  issue 化はしない）。
- tolerance／baseline・ガードレール閾値の変更は対象外（不変）。
- §3.4 の `grad_readout_contract_on_metal` FAIL（main 既存）の原因調査・
  是正は本 issue の対象外（ユーザー承認なしに issue 化はしない）。

## 5. 出典

- `docs/backend-metal-command-batching-design.md` §7.5（#1690 実装記録・
  §7.5.4 実測記入欄。2026-09-16 実測値は本文書 §3 が正）
- `docs/perf/logs/metal-mse-backward-1691/`（2026-09-16 M4 Max 実測の
  生ログ・`aggregate.md`・`ab/`・`env_info.txt`。内部ホスト名は含めない）
- `docs/perf/cuda-mse-backward-stream-contract.md`（#1692・CUDA 側の
  同型記録）
- `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4（動機となった診断
  専用計装の実測値。本判定には用いない）
- 親 #1582 → 実装 #1690（PR #1785）→ 本 issue #1691（測定）
