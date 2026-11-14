# G-CUDA-TRN の stream 種別 4 腕 A/B（#4299）

#4168 の前後比較で、G-CUDA-TRN（CUDA の MLP 学習・size 64・reuse）だけ after が 5/5 run で遅かった（腕比の中央値 1.007）。コードを読むと、学習に効く差は #4167 で推論 Graph が既定 ON になり、`CudaDevice::new` の stream が legacy の NULL stream から created stream に替わったことだけである。これが原因かどうかを、同じ BASE（`1b5a8dfd4`）の同じ `bench-fandhe` ソースから作る 2 つのバイナリと、プロセスにだけ渡す 1 変数で組む 4 腕の A/B で切り分ける。規則の正本は `RULE.txt`（計測より前に固定する）。

## 状態

**実測済み・記録済み（2026-11-14・GB10〈`<cuda-node>`〉）。** 判定 R1 は「逆向き」で、#4168 の G-CUDA-TRN の後退は再現しなかった。R2 は判定しない（RULE のとおり、R1 で再現したときだけ判定する）。

- 計測: 11:44:06Z〜11:51:41Z（`gb10/hwlock.log`）。16 ラウンドを打ち切りなしで完走し、`gb10/gate.log` は 16 行とも gate=pass。runner rc=0、判定器の終了コードは 0（`gb10/judge-4299-gb10.exit`）。
- clone: BASE は `1b5a8dfd4`、ランナー側は `d3886035e`（#4300 のマージコミット）。dry-run（11:42:41Z〜11:43:29Z）は rc=0。道具の sha256 は 3 行とも `RULE.txt` と一致した。
- 収録: `gb10/` に 220 ファイル（#4302。`SHA256SUMS` は 219 件で、その sha256 は `896004a9ed38f5fa1ced8ef78b36ba0841f0625019caae0029aac3139e6073b9`）。漏れ検査と隣接語のチェックは、送信側でも統合セッションでも 0 件。

### 結果（`gb10/judge-4299-gb10.md`。reuse の median_s の時間比で、> 1 は分子の腕が遅い）

| 比 | 中央値 | > 1 のラウンド数 |
|---|---|---|
| B/A（0.15.0 相当 ÷ 0.14.0） | 0.99308 | 1/16 |
| C/A（0.15.0 相当を legacy で ÷ 0.14.0） | 0.99862 | 6/16 |
| B/C（created ÷ legacy。新しいコード） | 0.99654 | 1/16 |
| D/A（created ÷ legacy。0.14.0 のコード） | 0.99192 | 1/16 |

- hard 条件: 同じ腕の中、B と C、A と D、A と B の checksum はすべて一致した（reuse・fresh とも）。
- 読み（RULE の判定ではなく、記録としての読み）:
  - 新旧どちらのコードでも、created stream の腕（B・D）の方が legacy の腕（A・C）より約 0.3〜0.8% 速い。
  - stream をそろえた C/A は 1 をはさむ（6/16）。stream 以外のコード差は、この計測の分解能では見えない。
  - #4168 の腕比 1.007 は、5 run・腕内の振れ（0.7〜1.0%）の範囲の偶然か、#4168 の測り方（セルごとに before／after を続けて走らせる）に由来する可能性があるが、ここでは確かめていない。
- fresh（参考）: B/A の中央値は 0.951（> 1 は 0/16）、D/A は 0.949。created stream は fresh でも速い方向。
- 修正は要らない見込み（学習を legacy stream に戻すと遅くなる方向）。閉じ方は所有者が決める（#4299）。

## 腕

| 腕 | ライブラリ | プロセスにだけ渡す環境変数 | stream（コードから読んだもの） |
|----|-----------|--------------------------|------------------------------|
| A | registry の `fandhe-ai =0.14.0` | なし | legacy |
| B | `1b5a8dfd4` の path patch | なし | created（既定 ON） |
| C | B と同じバイナリ | `FANDHE_AI_CUDA_GRAPH_INFER=0` | legacy |
| D | A と同じバイナリ | `FANDHE_AI_CUDA_GRAPH_INFER=1` | created（0.14.0 の opt-in） |

腕と stream の対応はコードを読んだ結果で、実行時には観測していない（`RULE.txt` の `[腕]`）。

## 実行手順（GB10〈`<cuda-node>`〉）

1. **clone を 2 つ用意する。** BASE は `1b5a8dfd4678eaad25553cef19142c8fb63539cf` の clone で、`.rev-stamp` の 1 行目にその SHA を書く。ランナー側は本 RULE.txt を含むコミット（本 PR のマージコミット。それより後の main ではない）の clone で、`.rev-stamp` にその HEAD を書く。
   - ランナーは、ランナー側の HEAD と BASE の間に `crates/`・`bench-fandhe` の差分が 0 であることを確かめる（`RULE.txt` の `[BASE]`）。0.15.0 の作業（#4174〜）で後から main に入った変更を含む clone では止まるので、そのときは差分 0 のコミット（本 PR のマージコミットか head）を clone し直す。
   - ビルドの成果物は `--dry-run` では `mktemp -d`（`TMPDIR`）の下、本計測では `scripts/bench/framework-compare/results/raw/diag4299-gb10/` の下に置かれる。容量が足りなければ `TMPDIR` を指定する。
2. **dry-run**（ロックを取り、ビルドと事前確認だけ。ゲート・計測はしない）:

   ```
   AB_BASE_ROOT=<BASE の絶対パス> AB_RUNNER_ROOT=<ランナー側の絶対パス> bash <ランナー側>/docs/perf/logs/diag-4299-cudatrn-stream/collect.sh --dry-run
   ```

   `DRY-RUN-OK` と `dry-run rc=0` を確かめる。あわせて `AB_RUNNER_ROOT=<ランナー側> bash <ランナー側>/scripts/bench/framework-compare/run_diag_4299_cudatrn_stream.sh --print-tool-sha256` の出力が `RULE.txt` のブロックと一致することを見る。
3. **本計測**（16 ラウンド。計測中はビルドも別負荷も並走させない）:

   ```
   AB_BASE_ROOT=<BASE の絶対パス> AB_RUNNER_ROOT=<ランナー側の絶対パス> bash <ランナー側>/docs/perf/logs/diag-4299-cudatrn-stream/collect.sh
   ```

   ランナー → 判定器 → 収録（マスク・漏れ検査・`SHA256SUMS`）まで 1 回で行い、`docs/perf/logs/diag-4299-cudatrn-stream/gb10/` ができる。`収録完了` の行と `runner rc`・`judge` の値を記録する。
4. **記録 PR**: `gb10/` をコミットし、判定器の結論（R1・R2・hard）と所有者への報告事項を本 README の状態に書く。PR 本文は `Refs #4299` だけにする（閉じるのは所有者の判断を記録した後）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 腕・手順・判定式・ラテン方格・道具の sha256（計測より前に固定） |
| `collect.sh` | 起動役と収録（ロック・ランナー・判定器・マスク・漏れ検査・`SHA256SUMS`）。`--dry-run`・`--self-test` あり |
| `../../../../scripts/bench/framework-compare/run_diag_4299_cudatrn_stream.sh` | ランナー（GB10 専用。ビルド 2 つ・事前確認・ゲート・16 ラウンド）。`--dry-run`・`--print-tool-sha256` あり |
| `../../../../scripts/bench/framework-compare/judge_diag_4299.py` | 判定器（標準ライブラリのみ・`python3 -I`）。`--self-test` あり |
| `gb10/`（計測後） | `judge-4299-gb10.md`・`.exit`・`ratios.tsv`・`round1`〜`round16/`（`<腕>-<mode>.jsonl`・`launch.tsv`・`uptime_*`・`gpu_apps_*`）・`env_info.txt`・`gate.log`・`compute_apps.log`・`skipped.log`・`tree-A.txt`・`tree-B.txt`・`collect.log`・`hwlock.log`・`SHA256SUMS` |

## マスク規則

公開面には機体名を `<cuda-node>` とだけ書く。ホームの絶対パス → `<home>`、ホスト名 → `<cuda-node>`、ユーザー名 → `<user>`（語境界つき。#4168 の `recollect.sh` と同じ定義）。マスクの後の漏れ検査で 1 件でも残れば収録しない。実名は `docs/real-hardware-verification-env.local.md` を見るだけで転記しない。

## スコープ外

- #4168 の道具（`run_ab_prepost_0150_4168.sh`・`judge_prepost_0150_4168.py`・`orchestrate.sh`・`recollect.sh`・RULE.txt）の変更（sha256 で固定済み）。
- 学習を legacy stream に戻すなどの修正（結果を見て所有者が決める）。0.15.0 の公開（#4174・#4175）は本計測を待たない。
