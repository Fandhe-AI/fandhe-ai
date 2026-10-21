# CUDA train／infer の内訳と同期回数（GB10・イシュー #3189）

数値の正式記録先は `docs/perf/train-step-phase-breakdown.md` §19・`docs/perf/infer-reuse-phase-breakdown.md` §12・
`docs/perf/cuda-train-forward-resident-diagnosis.md` §6。本ディレクトリはその計測基盤（規則・計測本体・集計）と、
計測後に置く生ログ一式。**診断のみ**（tolerance・baseline・判定規則・相手側ベンチ・ライブラリの `src/` と既定値・
公開 API・`Cargo.toml`／`Cargo.lock`・framework-compare の依存ピンは不変）。規則の正は `RULE.txt`（計測前に固定）。

**現状: 計測基盤のみ。GB10 での実測は未実施**（本ディレクトリに `armA/`・`census/` 等の生ログは無い。
各 doc は「未実測」と書く）。実測は下記手順で行い、`aggregate.py` の機械出力だけを doc へ転記する。

## 構成

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（同期の定義・差分法・見込みの式・fidelity・縮退） |
| `orchestrate_gb10.sh` | 計測本体。ビルドを全部先に終え、腕 A（5 run）→ B-2（infer 上位・下位）→ B-3（固定費）→ B-0（#2116）→ B-4（属性）→ B-1（nsys。最後）の順 |
| `aggregate.py` | 集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py <dir> > aggregate.md`） |
| 計測後の出力 | `armA/run{1..5}.{jsonl,err}`・`infer-diag/run{1..5}.{facade,backend}.{jsonl,err}`・`fixed-cost/run{1..5}.log`・`census/<window>-K<k>-r<n>.*`・`unified_memory_probe.txt`・`env_info.txt`・`load_gate.log`・`rev-stamp.txt`・`tree.txt`・`aggregate.md`・`SHA256SUMS` |

2116（B-0）の出力は 2116 側の README が指定する `docs/perf/logs/cuda-train-forward-resident-2116/gb10/` へ置く。

## 新規テスト（診断のみ・`#[ignore]` の実機部分）

| テスト | 役割 |
|---|---|
| `crates/facade/tests/cuda_train_infer_sync_census_3189.rs` | 窓 {fwd, fwd_bwd, full, infer_reuse} を `CENSUS_WINDOW`／`CENSUS_STEPS`（固定集合のみ）で実行する計数用ドライバ。nsys の下で回す。CPU スモークは CI で実行 |
| `crates/facade/tests/cuda_infer_reuse_phase_diag_3189.rs` | `predict_resident` の上位固定費（tape・snapshot・steps・chain・readout・paired/residual）。CPU の bit 一致は CI で実行 |
| `crates/backend-cuda/tests/infer_chain_phase_diag_3189.rs` | infer チェーンの下位分解（h2d_upload・l1_issue・l2_issue・download・download_again・導出 kernel_wait）。公開 trait のみで `src/` 無変更 |

## 実行手順（GB10 セッション。`docs/real-hardware-verification-env.md` §3・§4・§6）

1. 計測前コミットを `git archive` で展開し `.rev-stamp`（フル SHA）を置いて rsync で転送する（`.env*`・`*.local.md` は除外）。ツリー指紋を送信側と GB10 側で突き合わせる。
2. `PATH=/usr/local/cuda/bin:$PATH` を明示し、`CARGO_TARGET_DIR` はツリーの外に置く。framework-compare は別 target（スクリプトが `-fc` を付ける）。openrsync の `--delete-excluded` で framework-compare の target が消えるので転送後に再ビルドする。
3. `setsid nohup bash docs/perf/logs/cuda-train-infer-gb10-3189/orchestrate_gb10.sh` を切り離して起動し、生存を確認する。計測中に他のビルド・ベンチを並走させない。
4. 回収後、ホスト名・`$HOME`・ユーザー名のマスクを確認してから `python3 -I aggregate.py <dir> > aggregate.md`（本件と 2116 の両方）。exit≠0 なら数値を書かず原因を記録する。

## 注意・限界

- **腕 A と腕 B のソース版が違う**。腕 A は registry `fandhe-ai =0.11.0`、腕 B は HEAD ソース。bench-fandhe に backend-cuda の直接依存を足すと依存 allowlist の変更（所有者承認）になるため行わない。腕をまたぐ比較には版差を含む。
- `tiny_roundtrip` は引数なしの空カーネルの代替（1×1×1 GEMM）。空カーネルの起動には新規 `unsafe` が要るため行わない。
- infer 下位分解の `kernel_wait` は `download − download_again` の導出値（公開 API から stream を直接同期できない）。
- nsys の計測は時間を汚すため、回数と GPU 側所要時間（record-only）だけに使う。`.nsys-rep`／`.sqlite` はコミットしない。
- checksum が一致しても bit 同一の証拠ではない。CUDA Graph の step capture（#1350 は中立）の判定は覆さない。
