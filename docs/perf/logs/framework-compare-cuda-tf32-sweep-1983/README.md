# framework-compare `run_all_cuda.sh`（(a-tf32) TF32 スイープ込み）GB10 実測記録（イシュー #1983 フォロー）

## 目的

イシュー #1983（closed。PR #2007・コミット `bd2a6057`・2026-09-17）で `bench-fandhe --tf32` を
`fandhe_ai::set_cuda_tf32_gemm_enabled` へ結線し `run_all_cuda.sh` に (a-tf32) TF32 スイープを
追加したが、実装エージェントの実行環境に DGX Spark GB10 実機への到達手段がなく、
`docs/cuda-tf32-optin-api-decision.md` 追補（イシュー #1983）「実機実測」節は
「未実施のまま GB10 セッションへ申し送り」としていた。本ディレクトリは、その申し送りを受けて
2026-09-18（UTC）に GB10 で `run_all_cuda.sh` を実行した成果物をそのまま収納したものである。

確認対象は #1983 の受け入れ条件「`summarize.py` (a-tf32) 節に burn と fandhe-ai `--tf32` の行が
並ぶこと」。**採否判定（ADOPT／REJECT）は行わず、事実の記録のみ**とする。tolerance・
判定式・baseline はいずれも変更しない。

## ディレクトリ構成

| ファイル | 内容 |
|---|---|
| `results-cuda-0.9.0-2026-09-18.jsonl` | `run_all_cuda.sh` が出力した `results/raw/results-cuda.jsonl`（112 行。全セル） |
| `skipped-cuda-0.9.0-2026-09-18.log` | `results/raw/skipped-cuda.log`（0 行。MEASURE_ERROR・失敗セルなし） |
| `summary-cuda-0.9.0-2026-09-18.md` | `python3 summarize.py results/raw/results-cuda.jsonl` の標準出力。(a-tf32) 節は 62〜84 行目 |
| `summarize.err` | 同コマンドの標準エラー（30 行。すべて `parity_fail_count > 0` セルを「無効データとして表示」する既存の warning） |
| `run_all_cuda.log` | `run_all_cuda.sh` の実行ログ（182 行。ビルド 3 行＋各セルのヘッダ行と JSONL 行） |
| `uptime_before_run_all.txt` / `uptime_after_run_all.txt` | 実行前後の `uptime`（ホストローカル時刻） |
| `env_info.txt` | 実行環境（hostname は masked） |

## 実行条件・コマンド

- 実機: DGX Spark GB10（driver 580.173.02・CUDA 13.0・Ubuntu 24.04.4 aarch64・rustc 1.97.0）。詳細は `env_info.txt`
- 転送元コミット: `536c56a8`（main）。registry ピン `fandhe-ai =0.9.0`・`candle-core =0.11.0`・`burn =0.21.0`
- 専有 1 セッション（5 回計測中央値ではない。単発 run）。開始時 load1 1.49・GPU util 0%・
  常駐 2 プロセスあり（実名は記録しない）。終了時 load1 3.13
- 計測窓（UTC）: 2026-09-18T01:44:29Z 〜 01:46:30Z（約 2 分）
- gemm セル（(a-tf32) を含む）・infer セルは warmup 20・iters 20、train セルは warmup 20・iters 80
  （`bench-fandhe`／`bench-candle`／`bench-burn` の既定。JSONL の `warmup`／`iters` フィールドどおり）
- コマンド（`<repo>/scripts/bench/framework-compare/` で実行）:

```sh
uptime > uptime_before_run_all.txt
./run_all_cuda.sh > run_all_cuda.log 2>&1      # bench-candle／bench-burn は --no-default-features --features cuda でビルド
uptime > uptime_after_run_all.txt
python3 summarize.py results/raw/results-cuda.jsonl > summary-cuda-0.9.0-2026-09-18.md 2> summarize.err
```

(a-tf32) スイープの実体は `run_all_cuda.sh` 131〜149 行目のループで、`bench-fandhe`・`bench-candle` の
`gemm cuda`（fresh・N=256〜4096）に `--tf32` を付けて起動する（`run_all_cuda.log` 162〜181 行目の
`extra=--tf32` セル 10 個）。`bench-burn` はこのループを通さない（`--tf32` を常に MEASURE_ERROR で
拒否する仕様のため。burn の cuda gemm 行は通常スイープ (a) で `tf32:true` として記録される）。

## 結果: (a-tf32) 節（`summary-cuda-0.9.0-2026-09-18.md` 62〜84 行目の転記）

`### (a-tf32) GEMM TF32（--tf32 opt-in。REQ-2 統一複合判定。CUDA Tensor Core reduced precision）` / `#### CUDA`

| N | フレームワーク | 中央値 | Q1 | Q3 | GFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 256 | fandhe-ai（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00, rescued=0, bound=0.000e+00） | 92.2 µs | 91.9 µs | 92.5 µs | - |
| 256 | candle（無効: 要素誤差超過 fail=10522/65536, max_abs=1.582e-03, max_rel=1.554e+00, rescued=0, bound=1.907e-06） | 64.7 µs | 64.4 µs | 65.7 µs | - |
| 256 | burn（無効: 要素誤差超過 fail=10538/65536, max_abs=1.581e-03, max_rel=1.554e+00, rescued=0, bound=1.907e-06） | 427.7 µs | 178.5 µs | 496.8 µs | - |
| 512 | fandhe-ai（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00, rescued=0, bound=0.000e+00） | 262.1 µs | 261.7 µs | 262.3 µs | - |
| 512 | candle（無効: 要素誤差超過 fail=42387/262144, max_abs=2.340e-03, max_rel=1.970e+00, rescued=0, bound=3.815e-06） | 223.3 µs | 222.9 µs | 223.8 µs | - |
| 512 | burn（無効: 要素誤差超過 fail=42361/262144, max_abs=2.343e-03, max_rel=1.972e+00, rescued=0, bound=3.815e-06） | 422.0 µs | 351.8 µs | 551.3 µs | - |
| 1024 | fandhe-ai（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00, rescued=0, bound=0.000e+00） | 962.8 µs | 962.0 µs | 964.0 µs | - |
| 1024 | candle（無効: 要素誤差超過 fail=169971/1048576, max_abs=3.650e-03, max_rel=1.951e+00, rescued=0, bound=7.629e-06） | 864.6 µs | 863.8 µs | 865.6 µs | - |
| 1024 | burn（無効: 要素誤差超過 fail=169929/1048576, max_abs=3.643e-03, max_rel=1.965e+00, rescued=0, bound=7.629e-06） | 1.003 ms | 989.0 µs | 1.157 ms | - |
| 2048 | fandhe-ai（無効: 要素誤差超過 fail=681454/4194304, max_abs=4.941e-03, max_rel=1.987e+00, rescued=0, bound=0.000e+00） | 4.381 ms | 4.375 ms | 4.404 ms | - |
| 2048 | candle（無効: 要素誤差超過 fail=681418/4194304, max_abs=4.930e-03, max_rel=1.984e+00, rescued=33, bound=1.526e-05） | 3.713 ms | 3.701 ms | 3.729 ms | - |
| 2048 | burn（無効: 要素誤差超過 fail=681407/4194304, max_abs=4.941e-03, max_rel=1.987e+00, rescued=47, bound=1.526e-05） | 4.208 ms | 4.157 ms | 4.276 ms | - |
| 4096 | fandhe-ai（無効: 要素誤差超過 fail=2729050/16777216, max_abs=7.117e-03, max_rel=1.997e+00, rescued=0, bound=0.000e+00） | 33.867 ms | 33.609 ms | 33.922 ms | - |
| 4096 | candle（無効: 要素誤差超過 fail=2726683/16777216, max_abs=7.167e-03, max_rel=2.000e+00, rescued=646, bound=3.052e-05） | 58.698 ms | 58.622 ms | 59.161 ms | - |
| 4096 | burn（無効: 要素誤差超過 fail=2728488/16777216, max_abs=7.117e-03, max_rel=1.997e+00, rescued=562, bound=3.052e-05） | 39.682 ms | 39.574 ms | 39.829 ms | - |

### 受け入れ条件の確認

- **3 フレームワークが並ぶ**: N=256／512／1024／2048／4096 の全 5 形状で fandhe-ai（`--tf32`）・candle（`--tf32`）・
  burn（`tf32:true`）の 3 行が (a-tf32) 節に並んだ（計 15 行）。#1983 の受け入れ条件「(a-tf32) 節に
  burn と fandhe-ai `--tf32` の行が並ぶこと」を満たすことを GB10 実機で確認した
- fandhe-ai `--tf32` の 5 行は `run_all_cuda.log` 162〜171 行目（`extra=--tf32`）で `MEASURE_ERROR` なく
  完走し、JSONL に `"tf32":true` を emit している（`results-cuda-0.9.0-2026-09-18.jsonl` 103〜107 行目）。
  `skipped-cuda-0.9.0-2026-09-18.log` は 0 行
- 全 15 行が「無効: 要素誤差超過」表示・GFLOP/s `-` である。これは TF32 が結合順序・精度とも f32 FMA
  参照実装と異なるために `parity_fail_count > 0` となる **想定内の記録事項**であり、是正対象ではない
  （`docs/cuda-tf32-optin-api-decision.md` 追補〈イシュー #1983〉「要素単位検証は不変」）。`summarize.py` が
  既存挙動どおり無効データとして表示し、`summarize.err` の warning 30 行もすべてこの表示に対応する

### fandhe-ai 行の `bound=0.000e+00`（`parity_scaled_abs_bound=0`）の意味

`bench-fandhe` の gemm 計測は `reference.verify_strict(&out)`
（`scripts/bench/framework-compare/bench-fandhe/src/main.rs` 485 行目・`--tf32` 有無を問わず同一）で
要素単位検証を行う。`verify_strict` は `ScaledAbsTolerance::NONE` を固定で渡す
（`bench-common/src/parity.rs` 744〜747 行目）ため、ハーネス限定の第 3 救済項
（スケール付き絶対誤差 `diff <= 0.5・u・K・S_A・S_B`。`PARITY_SCALED_ABS_COEFF`）を fandhe-ai 自身の
出力には**適用しない**。`bound=0` はその「救済項なし」を示す値であり、判定は既存の統一複合判定
（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）のみで行われる。救済項は比較対象（candle／burn）の
妥当性検証に限り適用する契約（`docs/candle-parity-tolerance-contract-decision.md` §8。
`verify` 経由で `bound > 0`・`rescued` が記録される）。`--tf32` 実装（#1983）はこの
`verify_strict` 呼び出しに手を加えておらず、`validate_tf32_flag` の allowlist（`--task gemm --device cuda`
の素の GEMM 限定）と `set_cuda_tf32_gemm_enabled(true)` → `cuda_tf32_gemm_enabled()` 読み戻し確認のみを追加している。

この構造は記録値からも読み取れる。fandhe-ai（`--tf32`）行と burn 行は全 5 形状で `checksum`・`max_abs`・
`max_rel` が一致し（`results-cuda-0.9.0-2026-09-18.jsonl` 27〜31 行目・103〜107 行目）、`fail_count` は
`fandhe-ai.fail = burn.fail + burn.rescued` の関係にある（256: 10538 = 10538 + 0・512: 42361 = 42361 + 0・
1024: 169929 = 169929 + 0・2048: 681454 = 681407 + 47・4096: 2729050 = 2728488 + 562）。すなわち fail_count の差は
burn 側にだけ適用される救済項の分に等しい（記録済み統計値の算術上の観察であり、出力の bit 同一を主張する
ものではない。要素ダンプは採取していない）。

### burn 行について

burn の (a-tf32) 行は `bench-burn` が `tf32: cli.device == "cuda"` で常に付与する通常スイープ (a) の行そのもの
（`results-cuda-0.9.0-2026-09-18.jsonl` 27〜31 行目）であり、(a-tf32) 用の 2 回目の計測ではない
（`run_all_cuda.sh` 137〜142 行目のコメントどおり。burn 0.21 CUDA は TF32 既定で FP32 強制不可）。

### 0.9.0 正式再計測（`docs/perf/logs/framework-compare-0.9.0-remeasure/gb10/`）の burn 行との突合

| N | 本 run burn fail／rescued | 0.9.0 再計測 burn fail／rescued | 一致 |
| --- | --- | --- | --- |
| 256 | 10538 / 0 | 10538 / 0 | 一致 |
| 512 | 42361 / 0 | 42361 / 0 | 一致 |
| 1024 | 169929 / 0 | 169929 / 0 | 一致 |
| 2048 | 681407 / 47 | 681407 / 47 | 一致 |
| 4096 | 2728488 / 562 | 2728488 / 562 | 一致 |

`checksum`・`max_abs`・`max_rel` も全 5 形状で一致した（`results-dgx-0.9.0.jsonl` の burn gemm cuda 行と突合）。
なお 0.9.0 正式再計測（2026-09-16 UTC）は #2007（2026-09-17 マージ）より前の実行であり、その
`run_all_cuda.log` に (a-tf32) ループは含まれず、`"tf32":true` 行は burn の 7 行（gemm 5・train 1・infer 1）
のみである（fandhe-ai／candle の `--tf32` 行は 0 行）。fandhe-ai・candle・burn の 3 行が (a-tf32) 節に並んだ
GB10 記録は本 run が最初となる。

### 参考: 同 run の fandhe-ai FP32（既定・`--tf32` なし）行

通常スイープ (a) の fandhe-ai cuda gemm fresh 5 行（`results-cuda-0.9.0-2026-09-18.jsonl` 1〜5 行目）は
全形状 `parity_fail_count=0`・`bound=0`（`run_all_cuda.log` 4〜13 行目）。`--tf32` なしの既定行は
`set_cuda_tf32_gemm_enabled`／`cuda_tf32_gemm_enabled` を一切呼ばない設計（結線前と bit 同一）である。

## 全体の事実

- `results-cuda-0.9.0-2026-09-18.jsonl`: 112 行（(a)・(a')・(a-tf32)・(b)・(b')・(b'')・(c)・(c')・(c'') の全セル）
- `skipped-cuda-0.9.0-2026-09-18.log`: 0 行（MEASURE_ERROR・失敗セルなし）
- `summarize.err`: 30 行。すべて `tf32:true` の 15 行（fandhe-ai 5・candle 5・burn 5）に対する
  「gemm 要素単位検証が閾値超過 — 無効データとして表示」warning で、同じ 15 行が「データ有効性」節の走査
  （ラベルなし。15 行）と (a-tf32) 節の走査（`(tf32)` ラベル付き。15 行）で 1 回ずつ警告される。
  `tf32:true` でない行（FP32 の fandhe-ai／candle 行）への warning は 0 件
- `summary-cuda-0.9.0-2026-09-18.md` 末尾の「実行時失敗（skipped*.log）」節に並ぶ
  `skipped-m4max-0.4.0.log`〜`skipped-rtx3060-train.log` の項目は、GB10 ホストの `results/raw/` に git 管理下で
  同居している過去バージョンの skipped ログを `summarize.py` が併せて読み込んだものであり、本 run
  （`skipped-cuda-0.9.0-2026-09-18.log`。0 行）の失敗ではない

## 注意

- 本 run は #1983 の受け入れ条件の確認用であり、`scripts/bench/framework-compare/results/raw/results-dgx-*.jsonl`
  および `docs/perf/logs/framework-compare-0.9.0-remeasure/gb10/` の正式系列を**置き換えない**。
  candle 比ゲート（`docs/perf/cuda-gemm-candle-gate-remeasurement.md`）の判定も更新しない
- 単発 run（5 回計測中央値ではない）のため、(a-tf32) 節の時間値同士の比較・採否判定には用いない
- 3×TF32（`CudaGemmPrecision::Tf32x3`）の `--tf32` 対応・`gemm_tf32_cuda_smoke`（`#[ignore]`）の GB10 実行は
  本 run のスコープ外（後者は引き続き未実施）

## マスク規約

収納した各ファイルについて、内部ホスト名・ログインユーザー名・ホームディレクトリ絶対パスを検出する grep
（`docs/perf/logs/framework-compare-0.9.0-remeasure/` と同じパターン。`<user>@`・`<home>/`・内部ドメイン・ホスト名接頭辞）は 0 件（`run_all_cuda.log` は
`== build …` ヘッダと JSONL 行のみで絶対パスを含まないため置換は不要だった）。hostname は `env_info.txt` で
`masked` とし、常駐プロセスの実名も記録しない。
