# CUDA readback 宛先の再利用（イシュー #2108）

readback 宛先ポリシー（既定 OFF・env opt-in）と A/B 基盤の設計記録。**本 doc 時点で GB10 実測は未実施**
（実装は x86_64／RTX 3060 の Linux 環境。実測と ADOPT 時の既定値切替は実機セッションへ申し送り。
#2107・#2112 と同じ分業）。

## 1. 背景と前提

- `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.5: Layer A の `matmul` 区間に Σ Layer B で説明できない分が残る
  （N=1024 で 1.235 ms、N=2048 で 4.254 ms、N=4096 で約 18 ms 上界）。有力候補は本番 `readback` の既定
  `ReadbackDest::PretouchedFresh`（#1437）が毎回行う「新規 `Vec` 確保 + 非ゼロ sentinel の事前タッチ」。
- **#2107 の帰属判定はまだ実測されていない**（同 doc §13.4）。よって本 A/B は #2107 が「支持」を返した N でのみ
  ADOPT 対象になる（RULE.txt の前提ゲート）。
- Issue 本文の `readback_into` 経路は存在しない。全 D2H readback が通る `memory::readback`／`readback_with` を指すと解釈した。

## 2. 候補の検討

| 候補 | 判断 |
|---|---|
| (a) pinned staging 再利用 + copy-out（`PinnedStagingReuse`） | **採用（opt-in・既定 OFF）**。仮説 H-reuse: pageable 宛先への D2H と事前フィルを pinned への直接 DMA と CPU copy-out 1 回へ置換する。既存 `HostStaging::Pinned`（`unsafe` は `HostStaging::alloc` の既存 1 箇所）を流用し新規 `unsafe` なし |
| (b) pageable 再利用宛先（#1436 `PretouchedReusedDest`） | 本作業の対象外（宛先が pageable。#1436 で優先度低として保留。REJECT 済みではない）。本腕は宛先が pinned である点で別実験 |
| (c) managed 配置（#1352/#1353） | REJECT 済みのため扱わない |
| (d) 真の宛先再利用（copy-out なし）・ゼロコピー | 対象外。`Tensor` のストレージは `Arc<Storage{ data: Vec<T> }>` で drop フックがなく、返した `Vec` をバックエンドへ戻せない。tensor-core のストレージ抽象変更と新規 `unsafe` が必要でユーザー承認事項（承認後の別 issue 候補） |

## 3. 実装

- `crates/backend-cuda/src/readback_policy.rs`（新規・`unsafe` なし・新規依存なし）
  - env `FANDHE_AI_CUDA_READBACK_DEST` = `pretouched`（既定）／`pinned-reuse`。完全一致 allowlist、未知値は既定へ倒す（値はエコーしない）。`OnceLock` で 1 回だけ読む。`cfg(test)` の thread-local override あり
  - `ReadbackStagingPool`: ordinal ごとの `HostStagingCache`（要素数ごとに 1 バッファ・cap 256 MiB・世代不一致は fail-closed 破棄）。poison 時の取得は miss 扱い、返却は `into_inner` で回復
- `memory.rs`: `ReadbackDest::PinnedStagingReuse` を追加。`readback` は `readback_policy::current_dest()` を経由（既定 `READBACK_DEST = PretouchedFresh` は不変）。`ReadbackSentinel::readback_pinned_reuse` は f32 のみ staging 経路（`take`→未 hit なら `HostStaging::alloc`→`memcpy_dtoh`→`synchronize`→`to_vec()`→`put`）、他型は `PretouchedFresh` と同一処理。エラー時は staging を返却せず破棄。`numel == 0` は既定経路
- 診断入口（`internal-diagnostics` 限定）: `readback_f32_policy_diag`／`readback_f16_policy_diag`／`readback_staging_stats_diag`／`release_readback_staging_diag`
- 出力は D2H が全要素を上書きした後の `to_vec()` のため `PretouchedFresh` と bit 同一。tolerance・REQ-2 判定は不変

### 期待値（実測前に表明）

copy-out の memcpy が追加されるため、大形状（N=4096 で 64 MiB）では改善しないか後退しうる。ADOPT の見込みは高くない。
REJECT／undetermined でも機構（既定 OFF）と記録を残す。

## 4. A/B 手順と判定規則

- `scripts/bench/framework-compare/run_ab_readback_reuse_cuda.sh`（同一バイナリの env 切替。5 round・起動順反転・プロセス独立起動・専有ゲート既定 exclusive）
- 判定規則は実測前固定の `docs/perf/logs/cuda-gemm-readback-reuse-2108/RULE.txt` が正。手順は同ディレクトリの `README.md`
- 順序制約: #2107 の実測は本 PR 前のコミットで先に行う（本 PR は `readback`／`readback_with` を変更するため、#2107 の同一コード判定が HEAD 上では `no` になる。#2107 側は編集しない）

## 5. 実測記入欄（GB10 セッション）

実施: 2026-10-20（イシュー #3186 の一環）。HEAD `0889838c8` の facade を `AB_PATCH_FACADE_PATH` で参照、`AB_ROUNDS=5`、
専有ゲート exclusive（load1 < 1.0 かつ GPU 0% を 3 サンプル連続で通過）。前提ゲートは #2107 の全 N が「支持」
（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §13.4）。`readback_reuse_bit_match_2108` は 2 件 pass。
生ログ・manifest・比較表: `docs/perf/logs/cuda-gemm-readback-reuse-2108/`。

| セル | before 中央値 | after 中央値 | ratio | checksum |
|---|---|---|---|---|
| gemm reuse N=1024 | 2.053 ms | 2.256 ms | 1.0991 | 完全一致 |
| gemm reuse N=2048 | 8.287 ms | 9.326 ms | 1.1254 | 完全一致 |
| gemm reuse N=4096 | 38.268 ms | 42.024 ms | 1.0982 | 完全一致 |

**判定: REJECT**（RULE: 専有ゲートが成立した系列でいずれかの判定セルが ratio > 1.00）。3 セルとも後退した。
copy-out の memcpy が追加されるため後退しうるという事前表明（RULE「事前表明する期待値」）と整合する。
`READBACK_DEST` 既定は `PretouchedFresh` のまま変更しない（機構は既定 OFF のまま残す）。

### 5.1 N=256 の補遺セル（イシュー #3190・2026-10-20）

継承 RULE の判定セルは N=1024／2048／4096 で、N=256 は含まれない。#3190 の補遺規則（`logs/optin-ab-gb10-3190/RULE.txt` 9。事前登録）が
同じ判定文言を N=256 へ適用した。#2107 の前提ゲートは N=256 で測られていないため、2108 RULE の「前提ゲートで支持でない N を含む場合は
undetermined」により **ADOPT はあり得ない**（REJECT か undetermined のみ）。HEAD `d89a09370`（`origin/main` `42107efc1` に A/B 基盤の追加だけを載せたもの）の
facade、`AB_READBACK_SIZES=n256`、`AB_ROUNDS=5`、専有ゲート exclusive（load1 0.39・GPU 0% を 3 サンプル連続で通過）。
`readback_reuse_bit_match_2108` は 2 件 pass。収録: `logs/optin-ab-gb10-3190/readback-n256/`。

| セル | before 中央値 | after 中央値 | ratio | 5 run 内比 | checksum |
|---|---|---|---|---|---|
| gemm reuse N=256 | 92.1 us | 96.4 us | 1.0475 | 1.0724, 1.0500, 1.0498, 1.0472, 1.0724（5/5 後退） | 完全一致 |

なお標準出力に `他の bench-fandhe/bench-candle プロセスが実行中の可能性がある（pgrep 検出）` の警告が出た。専有ゲートは各 run の前に load1 < 1.0 かつ GPU 0% を 3 サンプル連続で要求しており（初回 load1 2.92 で待機後 0.39 で通過）、各 round 末尾の status 行は GPU 0〜1% だった。

**判定: REJECT**（補遺 RULE 9: 専有ゲートが成立した系列で ratio > 1.00）。N>=1024 の #3186 の結論と同じ向きで、全 N で pinned-reuse は後退した。
`READBACK_DEST` 既定は `PretouchedFresh` のまま。標準出力ログから JSON 行を無加工で取り出して表を再生成している（`*.reconstructed.*`。
`compare` の出力ファイルを作業中に誤って削除したため。値の改変なし。詳細は `docs/perf/optin-ab-gb10.md`）。

## 6. 検証状況（本 PR）

- 単体テスト: env allowlist・既定ドリフトガード（`READBACK_DEST` 直接検査）・override の入れ子復元・pool の ordinal 分離／世代破棄／cap／解放
- 実機 `#[ignore]` テスト（`tests/readback_reuse_bit_match_2108.rs`）: f32 の N=0〜2048² で NaN payload・±0・inf・subnormal を含む bit 一致と staging 再利用、f16 の既定経路への退避。開発ホスト（RTX 3060）で pass（driver のみ・NVRTC 不要）

## 7. 変更しないもの・スコープ外

`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・`docs/spec/`・facade 公開面・#2107 の計測成果物。
ADOPT 時の既定値切替は別 PR（ユーザー承認）。スコープ外: ゼロコピー・真の宛先再利用、並列フィル腕、fresh 行、Metal。
