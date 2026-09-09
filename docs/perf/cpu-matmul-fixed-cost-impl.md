# CPU matmul 出力確保の並列ゼロ埋め実装記録（イシュー #1299）

## §0 位置づけ

- 設計: `docs/cpu-matmul-fixed-cost-design.md`（イシュー #1294）§3.C 案 1a
- 親: #1296（要件項目 3）→ #1285（Phase 親）→ #1283（ルート）
- 後続: #1301（DGX Spark GB10 実機実測・有効化可否の最終判断）→ #1481（独立再計測・REJECT）→ #1482（確定既定）

## §1 設計 §3.x → 実装対応表

| 設計項目 | 決定 | 実装 |
|---|---|---|
| §3.A contiguous コピー | 変更なし。本番 NN 経路で発生しないことを回帰で固定 | `crates/backend-cpu/src/ops.rs::repack_count_tests::gemm_nn_does_not_increment_repack_counter`（新規）で `GEMM_HOST_REPACK_COUNT == 0` を固定 |
| §3.B tape 登録 | 変更なし | コード変更なし |
| §3.C 出力アロケーション（案 1a） | `CpuBackendOps::gemm` の `vec![0.0f32; m*n]` を、しきい値以上で rayon 並列ゼロ書き込みへ切り替える薄いヘルパーへ置換 | `crates/backend-cpu/src/ops.rs::zeroed_output`／`zeroed_output_with_threshold`（新規）。`CpuBackendOps::gemm`（`:472` 相当）を 1 行置換 |
| `GemmDriverVariant` 適用区分 | 追加しない（設計 §5.3 のとおり呼び出し元のバッファ準備方法の違いは variant の対象外） | `crate::gemm_blis::GemmDriverVariant` は無変更 |
| Layer B 診断テストの扱い | (i) 採用（`measure_one_phase_trial` 自体を変更後の確保方式へ更新） | `crates/backend-cpu/src/gemm_reuse_phase_diag_tests.rs:164` を `zeroed_output(n*n)` へ更新 |
| `facade`／`autodiff` | コード変更なし（テスト追加のみ） | `crates/facade/src/*`・`crates/autodiff/src/*` は無変更 |

## §2 本番結線の判断（2026-09-10 更新・#1481 独立再計測 REJECT により確定）: 無効化（`usize::MAX`）を確定既定とする

**#1301 が DGX Spark GB10（Grace CPU）・Apple M4 Max 両実機で on/off 比較を
実施した結果、両実機・全対象形状（N=512/1024/2048）で非後退（一部改善）を
確認した。この実測を根拠にいったんは本番既定しきい値を `usize::MAX`
（無効化）から `2 << 20`（8 MiB 相当。設計時の暫定値）へ有効化していたが、
事前宣言した判定規則（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
§20.1 規則 4・candle 比の非後退）は緩和なしでは 6 セル中 3 セルで不成立
であり、実測後に緩和した基準（同 doc §20.1a）のみを根拠に本番採用を
確定していた点が PR #1448 の codex-review 指摘（計測後に緩和した基準
だけで本番採用を確定しない）を受け、**本番既定を `usize::MAX`
（無効化）へ差し戻した****（§6・`docs/perf/logs/
cpu-matmul-fixed-cost-1301/` 参照）。#1301 の実測系列は「参考系列」
として §6 に維持しつつ、§20.1a の改定版規則 4（事前登録された規則
として以後固定）を用いた**独立の再計測**（イシュー #1481・同 doc
§20.7）を両実機で実施した結果 **verdict=REJECT** と確定した（規則 2:
DGX N=2048 の `alloc_c` が on/off で 2.1199 倍に増加／規則 3: 対照セル
8 中 3 が 1.05 超過／規則 4: M4 Max N=512 が 0.9060 < 0.9524。他は満た
す）。イシュー #1482 でこの REJECT 確定を受けて **`usize::MAX`（無効化）
を確定既定として固定**した（コード・テスト名・関連 docs の「未確定」
記述を確定状態へ整合）。再検討は同一の事前登録規則を機械適用する将来の
再計測（正式系列の新ピン更新時等）に限る。以下の §2 本文（旧版）は
#1299 実装時点の当初判断の記録として残す（内容自体は現在の判断＝無効化
維持と一致する）。

<details>
<summary>当初の判断（#1299 実装時点。参考として維持）</summary>

### （旧）§2 本番結線の判断: 無効化（`usize::MAX`）

設計 §7「判定基準」に基づき M4 Max スモーク（§4）を実施した結果、
**N=2048 で 5 run 符号一貫の後退**（`ops_gemm` 合成区間で中央値約 29%）
を観測したため、本番既定しきい値
（`crates/backend-cpu/src/ops.rs::GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`）
は設計時の暫定値 `2 << 20`（8 MiB 相当）ではなく **`usize::MAX`
（並列分岐を常に無効化）** とした。

これは「§3.C 案 1a を実装しなかった」ことを意味しない。並列ゼロ書き込み
ヘルパー自体（`zeroed_output_with_threshold`）は実装済みで、しきい値
定数 1 箇所を差し替えるだけで有効化できる状態を維持している。本番既定
だけを安全側（無効化）に倒した。

### なぜ後退したか（機構の推定）

macOS 上の `vec![0.0f32; len]`（`std::alloc::alloc_zeroed` 経由）は
OS の遅延ゼロページをそのまま返す。ページフォールト（実際にメモリが
確保される契機）は GEMM カーネルが最初にそのページへ書き込む時点まで
遅延され、`kernel` 区間へ計上される。一方 `Vec::with_capacity` +
rayon 並列書き込みは、確保した全ページへ即座に `0.0f32` を書き込む
ため、フォールトが `alloc_c` 区間内へ前倒しで発生する。

つまり本実装は「ゼロ書き込みそのものを並列化して速くする」のではなく
「本来 `kernel` 側に（ノイズに埋もれる形で）計上されていたフォールト
コストを `alloc_c` 側へ前倒しして顕在化させた」だけであり、
設計 §3.C の当初仮説（並列化で `alloc_c` が縮む）とは逆方向に働いた。

設計 §3.C はこの機構を Linux（glibc heap 経由の `calloc` が実際に
`memset` する）と macOS（VM 直接確保でフォールト遅延）の非対称として
仮説立てていた。今回の M4 Max 実測はこの仮説と整合する（macOS では
フォールト遅延の恩恵がそもそも大きく、前倒しは損にしかならない）。
DGX Spark GB10（glibc heap 経路）では逆に「`memset` の逐次実行を並列化
する」効果が働き得るため、#1301 が実機実測で判断する。

</details>

## §3 数値契約（bit 完全一致）

- **クレート内テスト**: `zeroed_output_tests`（`ops.rs`）が
  `zeroed_output`／`zeroed_output_with_threshold` 単体の契約（長さ・
  全要素ゼロ・両分岐・境界値・本番既定が無効化状態であること）を固定
- `parallel_branch_output_matches_sequential_branch_through_kernel`
  （同モジュール）が、並列分岐を明示的なしきい値 0 で強制した場合の
  GEMM 出力が逐次分岐と bit 完全一致することを確認（本番既定は
  `usize::MAX` のため通常経路では並列分岐を実走できず、このテストが
  唯一の並列分岐カバレッジ）
- **統合テスト**: `crates/backend-cpu/tests/gemm_output_alloc_bit_exact.rs`
  （新規）が `CpuBackendOps::gemm` 全体（小形状・境界形状・
  `m == 1`・`n == 0`・大形状 512/1024/2048・非正方 1 形状）で
  変更前後の bit 完全一致を検証。大形状は `#[ignore]`（release 実機）
- **facade 経由**: `crates/facade/tests/tape_matmul_cpu_bit_exact.rs`
  （新規）が `fandhe_ai::tape()` → `Var::matmul` の end-to-end 経路で
  同様の bit 完全一致を検証

いずれも本番既定が `usize::MAX` のため、通常実行時は両経路とも従来の
逐次確保を通り「差がない」ことを確認する形になる。並列分岐自体の
正しさは §3 冒頭のクレート内テストが担保する。

## §4 M4 Max スモーク実測（本セッション実行分・参考値）

`cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored
gemm_reuse_phase_diag_cpu --nocapture --test-threads=1` を、コード変更前
（`git stash` で `ops.rs`／`gemm_reuse_phase_diag_tests.rs` を
一時的に origin/main の状態へ戻して計測）・変更後（実装後の HEAD）の
双方で 5 回独立プロセス起動。生ログ・env_info は
`docs/perf/logs/cpu-matmul-fixed-cost-1299/`。

**注意**: 本セッションは共有マシン上の隔離 worktree での実行であり、
`uptime` の load average が 9〜11（19 users）と高い状態での計測。
`ops_gemm` の IQR（q1/q3）が after 側で顕著に拡大している（run4:
q1=27.4/q3=56.0 ms、run5: q1=26.5/q3=54.7 ms）ことから外部負荷の影響が
上乗せされている可能性があるが、5 run すべてで同一方向（後退）の
符号が一貫しており、中央値の差自体（約 29%）は計測ノイズ帯（設計 §7
の暫定基準・相対 5%）を大きく超えるため、ノイズのみによる誤判定とは
考えにくい。

N=2048（`m=n=2048`。4,194,304 要素 = 16 MiB。しきい値
`2 << 20` = 2,097,152 要素を上回り並列分岐の対象形状）の中央値（5 run
中央値。単位 ms）:

| 区間 | before（逐次のみ） | after（しきい値 `2 << 20` 時点の実測。本番は `usize::MAX` へ差し戻し） | 比（after/before） |
|---|---|---|---|
| `alloc_c` | 0.0733 | 0.2741 | 3.74 倍（後退） |
| `ops_gemm`（alloc_c+kernel+tensor_wrap 本番合成） | 23.7713 | 30.6585 | 1.290 倍（約 +29% 後退） |

5 run 生値（ms）:

| run | alloc_c before | alloc_c after | ops_gemm before | ops_gemm after |
|---|---|---|---|---|
| 1 | 0.0733 | 0.2387 | 23.7713 | 25.4739 |
| 2 | 0.0713 | 0.2741 | 22.9415 | 28.5781 |
| 3 | 0.0719 | 0.2634 | 23.4505 | 30.6585 |
| 4 | 0.0746 | 0.3809 | 23.9525 | 44.1277 |
| 5 | 0.0772 | 1.6112 | 26.1466 | 47.0374 |

pairwise（run 番号を揃えた before/after 比較。5/5 が同一方向）:
`ops_gemm` は 5/5 run で after > before（後退方向で符号一貫）。

N=512/1024 は設計 §0 のとおり `alloc_c` の絶対値自体が無視できる水準
（1024 でも約 0.02 ms）であり、しきい値未満のため両経路とも逐次経路の
まま差はない（本セッションのログでも変化なしを確認済み）。

**判定（設計 §7・plan 手順 11 の基準を適用）**: N=2048 で 5 run 符号
一貫かつ相対 5% を大きく超える後退（約 29%）を確認したため、本番結線
は行わず（§2）、閾値を `usize::MAX` へ差し戻した。

DGX Spark GB10 実測は本エージェント実行環境に実機接続手段がないため
未実施（#1301 が担当）。

## §5 M4 Max 実測ログの配置

`docs/perf/logs/cpu-matmul-fixed-cost-1299/`:
- `env_info.txt`: 実行環境情報（内部ホスト名は含めない）
- `before-run{1..5}.log`: コード変更前（`git stash` で一時的に
  origin/main 相当へ戻して計測）
- `after-run{1..5}.log`: コード変更後（実装後の HEAD。しきい値
  `2 << 20` を一時的に使って計測。**本番既定は §2 のとおり
  `usize::MAX`**）

## §6 DGX Spark GB10 実測結果（イシュー #1301・2026-09-08 実施済み）

DGX Spark GB10（専有ゲート確認: 1 分 load average <6 を 2 回連続確認後に
計測。1 回目の on 腕は他セッション並走〈イシュー #1262・#1437。load
average 15〜18〉により contamination を検出し、`docs/perf/logs/
cpu-matmul-fixed-cost-1301/` に記録のうえ専有確認後に再計測した「clean」
系列を正式値とする）・Apple M4 Max（このリポジトリのメイン worktree が
動くホスト自身。共有マシンだが #1299 当時より低負荷）で、§4 と同一
プロトコル（Layer B: `gemm_reuse_phase_diag_cpu`。5 回独立プロセス起動・
`RAYON_NUM_THREADS` 未設定）に加え、Layer A（`bench-fandhe gemm cpu <N>
reuse`。`run_gemm_gate_cpu.sh` 経由の 5 回独立プロセス起動）も計測した。

**Layer B（DGX。off=`usize::MAX`／on=`2 << 20`。5 run 中央値、ms）**:

| 区間 | N=2048 off | N=2048 on | on/off 比 | N=1024 off | N=1024 on | on/off 比 |
|---|---|---|---|---|---|---|
| `alloc_c` | 3.2554 | 1.6817 | **0.5166**（約 48% 削減） | 0.0306 | 0.0269 | 0.8791 |
| `ops_gemm` | 26.8177 | 26.7420 | **0.9972**（非後退） | 5.4276 | 5.3701 | 0.9894 |

**Layer A（`bench-fandhe gemm cpu <N> reuse`。5 run 中央値。
`docs/perf/logs/cpu-matmul-fixed-cost-1301/` の生ログ・`compare_gemm_ab.py`
出力）**:

| N/mode | DGX on/off 比 | M4 Max on/off 比 |
|---|---|---|
| 512/fresh | 0.8690 | 1.0068 |
| 512/reuse | 0.8889 | 0.9509 |
| 1024/fresh | 0.9841 | 1.0019 |
| 1024/reuse | 1.0042 | 1.0269 |
| 2048/fresh | 1.0469 | 0.9728 |
| **2048/reuse（決定セル）** | **0.9990** | **0.9788** |

checksum は両実機・全セル完全一致（`compare_gemm_ab.py` 判定 `完全一致`）。
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20.1 で計測前に確定した
判定規則のうち規則 1〜3・5 は満たしたが、**規則 4（candle 比の非後退）は
緩和なしでは 6 セル中 3 セルで不成立**だった（同 doc §20.3 candle 比表）。
§20.4 の判定はいったん §20.1a の事後緩和版規則 4 を適用して「全規則を
満たす（ケース (c)）」とし本番既定を無条件に `2 << 20` へ有効化していたが、
PR #1448 の codex-review 指摘（計測後に緩和した基準だけで本番採用を確定
しない）を受けて**本番既定を `usize::MAX`（無効化）へ差し戻した**。
上記の実測結果自体（Layer A/B の各表）は参考系列として本節に維持する。
§20.1a の改定版規則 4 を事前登録規則として用いた独立の再計測（同 doc
§20.6）が完了するまで、本番既定は有効化しない。

**独立再計測は #1481（同 doc §20.7）で実施済み**: 規則を計測後に緩和・
読み替えずに機械適用した結果 verdict=REJECT（規則 2・3・4 のいずれも
不成立セルを含む）と確定し、本番既定 `usize::MAX`（無効化）は変更なし。

**#1482 でこの REJECT 確定を本番既定として固定**: `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS`
の値自体は変更しない（すでに `usize::MAX`）が、`ops.rs` の定数 doc
comment・テスト名（`default_threshold_is_disabled_pending_independent_
remeasurement` → `default_threshold_is_disabled_confirmed_by_
independent_remeasurement`）・`gemm_output_alloc_bit_exact.rs`／
`tape_matmul_cpu_bit_exact.rs` の冒頭 doc・本ドキュメント・
`docs/cpu-matmul-fixed-cost-design.md` §10・CLAUDE.md の該当行を、
「独立再計測の完了・ADOPT 確定を待つ未確定状態」から「REJECT が確定し
`usize::MAX` が確定既定である状態」へ整合させた。並列ゼロ書き込み
ヘルパー自体（`zeroed_output_with_threshold`）は削除せず、しきい値
定数 1 箇所の差し替えで再有効化できる状態を維持する。

## §7 スコープ外（本 Issue では実施しない）

- `gemm_checksum`（`ops.rs:500` 相当）・`gemm_bias_act`（`:678` 相当）・
  `gemm_resident_rhs`（`:339` 相当）への横展開（設計 §3.C が `gemm` 限定
  を推奨）
- 案 2（`tensor-core` の `Storage` 返却フック実装によるバッファプール化）
- 案 3（`unsafe`: `set_len`／`alloc_zeroed`／`mallopt`／mmap 直接操作）
- `#[cfg(target_os = "linux")]` によるプラットフォーム限定 gating
  （両実機とも非後退を確認したため #1301 では不要と判明。将来他
  プラットフォームで後退が見つかった場合の予備策として設計は保持）
- machine gating（page fault 計数）: `/usr/bin/time -v` による minor page
  fault 数を DGX 実機で記録済み（`docs/perf/logs/
  cpu-matmul-fixed-cost-1301/time-v-dgx-*.log`）。alloc_c 削減が
  page fault 前倒し自体の削減によるものか厳密な定量比較は未実施
  （記入欄のみ・追加分析はスコープ外）

## §8 関連ドキュメント

- `docs/cpu-matmul-fixed-cost-design.md` §10（実装記録ポインタ）
- `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §16〜§18
  （固定費削減優先順位・専有環境実測の背景）
- `scripts/bench/framework-compare/README.md`「CPU での区間定義と
  Layer B」節
