# CUDA GEMM `tiled_f32_outperforms_naive_at_4096` 新規 FAIL（speedup=0.235x）の切り分け

イシュー #1203。`crates/backend-cuda/tests/gemm_tiled.rs::tiled_f32_outperforms_naive_at_4096`
（性能アサーションテスト。`MIN_SPEEDUP = 1.1`）が #1162 の GB10 実機 sweep
（2026-09-05・`--test-threads=1` 直列・`--features internal-diagnostics`）で
speedup=0.235x で FAIL した事象の再現・原因切り分け記録。

## 1. 背景

- `docs/backend-cuda-real-device-testing.md` §5.1（2026-08-10）は同テストの
  FAIL を「同一バイナリ内並列実行による GPU 時間分割」と結論づけていたが、
  #1162 は `--test-threads=1` **直列**条件で FAIL しており、その説明では
  今回の事象を説明できない
- `run_naive_f32`／`run_tiled_f32` は `crates/backend-cuda/src/gemm.rs::run_f32_kernel`
  を共通ホスト経路（H2D 転送 ×2・プール確保・カーネル起動・readback）として
  共有し、差分はカーネルハンドルと `LaunchConfig` のみ
  （`crates/backend-cuda/src/gemm.rs:2746`〈`run_naive_f32`〉・
  `crates/backend-cuda/src/gemm.rs:2963`〈`run_tiled_f32`〉・
  `crates/backend-cuda/src/gemm.rs:4537`〈`run_f32_kernel`〉）
- M=N=K=4096・f32 は 1 行列 64 MiB（A・B・C 各 64 MiB）。#1130 ツリーが
  確定した「単体バッファ ≥32 MiB の D2H で、宛先が毎回新規確保・未タッチの
  `Vec` の場合に glibc mmap しきい値由来の遅延（24〜33 倍）・確率的二峰性」
  の適用範囲に入る（`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md` §11）
- 現 HEAD（2026-09-08 時点）は N=4096・K=4096 で 128×64 pipeline カーネル
  （`TILED_PIPELINE_128X64_PRODUCTION_ENABLED = true`）を選択する
  （#1162 実行時点は 64×64 pipeline。`gemm.rs:996-1018`）

## 2. 切り分けの枠組み

naive と tiled はホスト経路を byte 単位で共有するため、4.2 倍の差の
発生源は次の 2 つに限られる:

- **(a)** tiled カーネル自体が 4096 で遅い（cp.async パイプライン化・
  #1137/#1164/#1344 の結線を疑う）
- **(b)** 共有ホスト経路が「64 MiB D2H 二峰性の slow モード」に tiled 側の
  計測中だけ入った（環境要因）

## 3. 実行環境

- GB10（内部ホスト名は含めない。`docs/real-hardware-verification-env.local.md` 参照）
- HEAD: `8bfdb20778ca7bf1bde86241d073be65bb4ef12b`
- driver_version: 580.173.02
- 実行前 GPU utilization.gpu: 0 %・load average 0.02, 0.90, 1.82（アイドル）
- 詳細: `docs/perf/logs/cuda-gemm-tiled-naive-4096-1203/env_info.txt`

## 4. 再現結果

### 4.1 単発実行（本テストのみ）× 5 回

| run | 判定 | speedup | naive samples (s) | tiled samples (s) |
|---|---|---|---|---|
| 1 | PASS | 2.854x | [0.132, 0.128, 0.127, 0.126, 0.126] | [0.047, 0.045, 0.045, **0.037**, 0.039] |
| 2 | PASS | 2.973x | [**0.648**, 0.131, 0.130, 0.130, 0.132]（naive に単発スパイク） | [0.044, 0.044, 0.044, 0.045, 0.044] |
| 3 | **FAIL** | **0.189x** | [0.129, 0.128, 0.130, 0.128, 0.129] | [**0.612**, **0.696**, **0.687**, **0.680**, 0.045] |
| 4 | PASS | 1.205x | [0.656, 0.657, 0.640, 0.651, 0.648]（naive・tiled 双方が全サンプル slow） | [0.539, 0.540, 0.534, 0.545, 0.545] |
| 5 | PASS | 2.918x | [0.130, 0.131, 0.130, 0.128, 0.129] | [0.047, 0.043, 0.043, 0.045, 0.045] |

5 回中 1 回（20%）FAIL。再現した（生ログ: `logs/cuda-gemm-tiled-naive-4096-1203/single_run_{1..5}.log`）。

### 4.2 `gemm_tiled` バイナリ全体・直列（`--test-threads=1`）× 3 回

| run | 判定 | speedup | 備考 |
|---|---|---|---|
| 1 | **FAIL** | **0.208x** | tiled samples=[**0.682**, **0.642**, 0.042, **0.620**, 0.041]（3/5 が slow の二峰性） |
| 2 | PASS | 3.218x | 全サンプル fast |
| 3 | PASS | 1.188x | naive の 3/5・tiled の 4/5 が slow（双方 slow で相殺） |

3 回中 1 回（33%）FAIL。#1162 の speedup=0.235x と同水準（run 1 の 0.208x）で
再現した。**直列条件でも FAIL することを確認**し、
`docs/backend-cuda-real-device-testing.md` §5.1 の「並列実行のみ fail」と
いう帰属では説明できないことが確定した（生ログ:
`logs/cuda-gemm-tiled-naive-4096-1203/serial_binary_run_{1..3}.log`）。

### 4.3 既定並列・#1162 相当の全 sweep 直列

時間制約により未実施（下記「7. 未実施・引き継ぎ」参照）。単発・直列いずれの
条件でも既に FAIL を再現できているため、切り分けの結論に必要な追加証拠とは
判断しなかった。

## 5. 原因切り分け（診断テスト。`gemm_tiled_naive_speedup_triage_1203.rs`）

`internal-diagnostics` feature 限定の診断テスト 3 種を 1 回実行した
（生ログ: `logs/cuda-gemm-tiled-naive-4096-1203/triage_run_1.log`）。

### 5.1 classic（非 pipeline）vs 本番 pipeline（診断 (3)）

```
tiled_pipeline(production): samples=[0.036, 0.545, 0.534, 0.544, 0.037]s median=0.534s
tiled_classic(diagnostics): samples=[0.041, 0.047, 0.053, 0.050, 0.557]s median=0.050s
```

**本番 pipeline（128×64 cp.async）と classic（64×64・非 pipeline）の両方が
同程度の振幅（約 0.53〜0.56s）の slow モードを示した**。classic 版は
cp.async パイプライン化（#1137/#1164/#1344）を一切経由しないにもかかわらず
同じ slow モードに入ったことから、**pipeline カーネル自体の実装（(a)）が
FAIL の必要条件ではない**という強い傍証が得られた（1 run のみの計測かつ、
`measure` 計測点は H2D／プール確保／カーネル起動／readback を含む呼び出し
全体の壁時計時間であり、遅延がカーネル実行区間そのものではなくホスト経路の
どのフェーズで発生しているかを本診断単体では分離できていない点に注意）。

### 5.2 順序反転: tiled→naive（診断 (1)）

```
tiled(1st): samples=[0.044, 0.042, 0.041, 0.544, 0.536]s（末尾 2 サンプルが slow へ移行）
naive(2nd): samples=[0.650, 0.648, 0.652, 0.652, 0.652]s（全サンプル slow）
```

tiled の計測後半で slow モードに入り始め、直後に計測した naive は**全
サンプルが slow モードに固定**された。slow モードが単発的ではなく、
発生すると後続の計測ウィンドウへ跨って持続する傾向がある（#1146 §4.4 の
「順序依存スパイク」と整合する観測）。naive/tiled どちらが先でも影響を
受けうることを示しており、tiled 固有の劣化ではないことを補強する。

### 5.3 インターリーブ（診断 (2)）

```
pair0: naive=0.123 tiled=0.031
pair1: naive=0.126 tiled=0.039
pair2: naive=0.127 tiled=0.041
pair3: naive=0.134 tiled=0.043
pair4: naive=0.130 tiled=0.043
```

この run では slow モードが発生せず、5 ペアとも fast のまま
（speedup=3.11x）。インターリーブ自体が slow モードを必ず誘発する／
回避するわけではなく、確率的な発生という前提と矛盾しない（1 run のみの
ため確証には至らない）。

## 6. 判定

**推定される主因: (b) 環境要因（#1130 系 64 MiB D2H 二峰性と整合する挙動）。
pipeline・プール変更（#1137/#1164/#1344・#1199/#1200）との明確な相互作用は
観測されなかった。** ただし本調査の計測点（`measure` クロージャ。§1・
`crates/backend-cuda/src/gemm.rs::run_f32_kernel`）は H2D／プール確保／
カーネル起動／readback を含む呼び出し全体の壁時計時間であり、フェーズ分解
計測ではないため、遅延区間が具体的に D2H であることや pipeline との相互
作用が構造的に皆無であることを直接証明する測定ではない。以下は現時点で
入手した証拠が指し示す仮説として記録する（フェーズ分解計測による直接確認は
§8「未実施・引き継ぎ」参照）。

根拠:

1. 単発・直列いずれの条件でも FAIL を再現し（4.1・4.2）、#1162 実測値
   （0.235x）と近い水準（0.189x・0.208x）で再現した
2. tiled 側 FAIL run のサンプル分布は一貫して二峰性（3〜4/5 が約 0.5〜0.7s
   帯・残りが約 0.04s 帯）であり、`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
   が記録した 64 MiB D2H 二峰性の症状と振幅・確率的な出現パターンの点で
   整合する（前述のとおり呼び出し全体の計測のため、D2H フェーズ単体を
   分離した直接比較ではない）
3. **classic（非 pipeline）カーネルでも本番 pipeline と同振幅の slow モード
   が発生した**（5.1）。pipeline 結線を経由しない classic 版でも同じ現象が
   起きたことは、原因がカーネル実装差分（tiled 固有）ではなく naive/tiled
   が共有するホスト経路（H2D／プール確保／readback）側にある可能性を示す
   強い傍証である（1 run のみのため確証ではない）
4. naive 側にも単発スパイク（4.1 run2）・全サンプルが slow モードに入った
   ケース（4.1 run4）・大半のサンプルが slow モードに入ったケース（4.2 run3。
   naive 3/5・tiled 4/5）が観測されており、slow モードは naive/tiled の
   どちらにも対称に発生しうる。tiled 側だけが計測タイミング的に slow
   モードへ入ったときのみ speedup が閾値を割る（naive が slow でも tiled
   も同程度 slow なら speedup は保たれる。4.1 run4・4.2 run3 参照）
5. #1199（案 A 不採用・本番結線なし）・#1200（`gemm_mma.rs` の f16 経路
   限定・`run_f32_kernel` は無変更）はいずれも `run_f32_kernel` を変更
   していないため、これらとの相互作用は構造的にありえない
   （`git log -- crates/backend-cuda/src` で `run_f32_kernel` の変更履歴が
   #1130 系以降に無いことを確認済み）

`docs/perf/cuda-gemm-tiled-pipeline.md` が記録する純カーネル時間（約
10 TFLOPS・N=4096）・`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
の 64 MiB 帯二峰性の既存記録と整合するため、追加のカーネル単体計測
（ゲート C 相当）は本調査では再計測しなかった。

## 7. `docs/backend-cuda-real-device-testing.md` §5.1 の帰属再評価

2026-08-10 の記録（「並列実行のみ fail」）は誤りとまでは断定しないが、
**直列条件でも同じ症状（tiled 側の二峰性スパイク。#1130 系 D2H 二峰性と
整合すると推定されるが本調査ではフェーズ分解計測による直接確認はしていない）
で FAIL しうる**ことが本調査で確認されたため、「並列実行による GPU 時間
分割」を唯一の原因とする記述は不正確である。本ドキュメントの更新（§5.1 節）
でこの点を反映する。

## 8. 未実施・引き継ぎ（時間制約によるスコープ外。ユーザー承認が必要な後続候補）

- **P4 相当のカーネル単体純時間の再計測**（`gemm_tiled_pipeline_bench`／
  `cuda_floor_bench` 実行）は本調査では未実施。既存記録（§6 参照）との
  整合を根拠に判定したが、直接の再確認ではない
- **`MALLOC_MMAP_THRESHOLD_` 固定実験**（#1146 §4.6 の確証手法）は未実施。
  slow モードの消失／移動を確認できれば「D2H 宛先 `Vec` の glibc アロケータ
  挙動」機構をさらに直接確認できる
- **#1162 と全く同一条件（`--features internal-diagnostics --no-fail-fast`
  の全 ignored sweep 直列 × 1 回）**は時間制約により未実施。単発・直列の
  対象テスト単独実行で既に FAIL を再現できているため優先度を下げた
- **テスト硬化案**（本 PR では実施しない。ユーザー承認が必要）:
  1. naive/tiled をペアごとにインターリーブ計測へ変更し、両者を同じ
     確率で slow モードに曝すことで、片側性 FAIL を構造的に減らす
  2. #1132 前例（カーネル単体計測プロトコルへの切替）に倣い、常駐
     バッファでのカーネル単体計測へ本テストの計測境界自体を変更する
     （ホスト経路の変動を測定対象から除外する）

いずれも `MIN_SPEEDUP`・計測プロトコルの変更そのものであり、
`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を
単独で緩和しない」と同じ精神からユーザー承認が必要と判断した（本 PR の
スコープ外）。

## 9. 関連ファイル

- `crates/backend-cuda/tests/gemm_tiled.rs`（診断出力追加）
- `crates/backend-cuda/tests/gemm_tiled_naive_speedup_triage_1203.rs`（新規診断テスト）
- `docs/perf/logs/cuda-gemm-tiled-naive-4096-1203/`（実行ログ・env_info）
- `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`（64 MiB D2H 二峰性の一次記録）
- `docs/perf/cuda-fresh-gemm-n2048-overhead-diagnosis.md` §11（同族の可能性の議論）
- `docs/perf/cuda-gemm-tiled-pipeline.md`（tiled pipeline 結線の判断記録・純カーネル時間）
- `docs/backend-cuda-real-device-testing.md` §5.1（更新対象）
