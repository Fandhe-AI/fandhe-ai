# framework-compare: Rust ML フレームワーク横並びベンチマーク

fandhe-ai を candle・Burn と同一プロトコルで横並び比較する独立ベンチ workspace
（プロトコル同一性の範囲・例外は「計測プロトコル」節の `--mode fresh|reuse` の説明を参照。
Metal / CUDA の `fresh` GEMM 比較は例外にあたる）。
`scripts/bench/oss-gemm-compare/`（イシュー #755。許容依存第 9 区分）と同じく**本体 workspace 外の独立 Cargo workspace** であり、ルート `Cargo.toml` / `Cargo.lock` には一切現れない。本 workspace の `Cargo.lock` は比較対象として依存禁止リストのクレート（`candle-core`・`burn` と、その推移的依存の `cubecl`・`ndarray`・`tch` 等）を**意図的に含む**ため、`scripts/check-forbidden-deps.sh lock-all` は禁止リスト grep の代わりに専用の fail-closed 契約検査（Cargo.lock の存在・独自 `[workspace]` 宣言・承認済みピンのドリフト検出）を適用する（`.claude/rules/deps-policy.md`「第 9 区分」の適用範囲拡張、および `docs/framework-compare-harness-decision.md` を参照）。依存監査（advisories / bans / licenses / sources）は専用の `deny.toml` を対象に CI（`deps-forbidden` ジョブ）で毎回実行される。

実測記録（`results/summary.md`・raw JSONL）は「実行資産は scripts/bench・記録は docs/perf」の区分の例外として、再現に必要な生成物一式を本ディレクトリ配下で管理する（`docs/perf/` の実測記録群と同趣旨のコミット済み一次データ）。

## 比較対象

| フレームワーク | クレート | バージョン | デバイス |
| --- | --- | --- | --- |
| fandhe-ai | `fandhe-ai`（facade。crates.io 版） | =0.7.0 | CPU / Metal / CUDA（`tape_for(Device::…)`） |
| candle | `candle-core` | =0.11.0 | CPU / Metal（`metal` feature）/ CUDA（`cuda` feature） |
| Burn | `burn` | =0.21.0 | CPU（ndarray）/ Metal（wgpu）/ CUDA（cubecl） |
| tch-rs | — | 未計測 | libtorch 依存のため省略 |

計測済み環境は 3 系統（詳細・結果は `results/summary.md`）:

- 環境 1: Apple M4 Max / macOS（CPU + Metal）→ `results/raw/results.jsonl`
- 環境 2: DGX Spark（NVIDIA GB10。CUDA + ARM CPU）→ `results/raw/results-dgx.jsonl`。CUDA ホストでは `./run_all_cuda.sh` を使う（bench-candle / bench-burn は `--no-default-features --features cuda` でビルドされる。fandhe-ai は cfg + 実行時プローブのため feature 指定不要）
- 環境 3: NVIDIA GeForce RTX 3060（12 GiB）/ Linux（CUDA。デバイス/tape 再利用モードの fresh/reuse 比較用）→ `results/raw/results-rtx3060.jsonl`（イシュー #925）
- 環境 4: 環境 3 と同一機（RTX 3060 / Linux）。MLP 学習のデバイス常駐更新モード（`train --mode reuse`）の fresh/reuse 比較用 → `results/raw/results-rtx3060-train.jsonl`（イシュー #957/#958/#959。fandhe-ai 0.4.0 計測のため 0.3.0 計測の環境 3 とは別ファイル）
- 環境 5: 環境 1 と同一機（Apple M4 Max / macOS）。MLP 学習のデバイス常駐更新モード（`train --mode reuse`）の cpu/metal での fresh/reuse 比較用 → `results/raw/results-m4max-train.jsonl`（イシュー #957。fandhe-ai 0.4.0 計測のため 0.3.0 計測の環境 1 とは別ファイル。環境 1 の `results.jsonl` を上書きしないよう `run_all.sh` ではなく個別実行で取得）

## 計測タスク

すべて f32・決定的シード（xorshift64\* を `bench-common` に自前実装、全フレームワークで同一シード・同一生成式の入力）。

- **(a) GEMM**: C = A×B、N = 256 / 512 / 1024 / 2048（GPU は 4096 も）。指標: 中央値・Q1・Q3、GFLOP/s（2N³/median）
- **(b) MLP 学習**: 784→256→10（ReLU）、バッチ 64、合成データ、MSE、手動 SGD（lr 0.01）、100 ステップ。先頭 20 ステップを warmup として除外し、残り 80 ステップの 1 ステップ時間の中央値・Q1・Q3
- **(c) 推論**: 同 MLP の forward のみ、バッチ 64。1 回（= 1 バッチ）あたり時間の中央値・Q1・Q3、バッチ/秒（`throughput_per_s` = 1/median。1 バッチ = 64 件の forward であり、1 件あたりの推論/秒ではない）

## 計測プロトコル（fandhe-ai の計測規約に準拠・拡張）

- warmup 20 回 → 計測 20 回、中央値 + Q1/Q3（学習は 100 ステップ中、先頭 20 を warmup）
- **同期の統一**: 計測区間の終端で必ず結果テンソルをホストへ実体化して全要素を読み出す
  （fandhe-ai: `to_tensor()` + `contiguous().as_slice()` / candle: `to_vec2()` / Burn: `into_data()`）。
  GPU の非同期実行を計測漏れさせない。読み出した checksum は JSON に記録し、フレームワーク間の数値一致確認に使う
- 計測ごとに新しい計算グラフを作る（fandhe-ai は毎回新しい `tape()` / `tape_for(Device::…)`。
  この条件は fandhe-ai の CUDA で tape ごとの初期化コスト約 440〜460 ms を毎回計測区間に含める。`results/summary.md` 環境 2 の備考を参照）
- 重み初期化: candle / Burn は共有 RNG（同一シード）で同一の重み。fandhe-ai の `Sequential::add_linear` は
  内部初期化（シード指定）のため重みの値自体は異なるが、実行時間には影響しない（同一アーキテクチャ・同一入力）
- **`fresh` の計測範囲は fandhe-ai と candle / Burn とで非対称（イシュー #925 レビュー指摘）**:
  「計測ごとに新しい計算グラフを作る」は fandhe-ai の `tape()` / `tape_for(Device::…)` にのみ適用され、
  デバイス・入力テンソルの構築を毎回計測区間内で行う。一方 candle（`bench-candle/src/main.rs`）・
  Burn（`bench-burn/src/main.rs`）は `Device` と入力 `Tensor` をループ外（計測開始前）で 1 回だけ構築し、
  計測区間は `matmul` + ホスト実体化のみを含む。したがって CPU（`tape()` はデバイス選択コストを持たない）
  では実質的に同一プロトコルとみなせるが、**Metal / CUDA では `fresh` の GFLOP/s はフレームワーク間で
  プロトコル同一とは言えない**（fandhe-ai 側にのみ毎回のデバイス/tape 構築コストが乗る）。上記 GEMM
  比較表（Metal・CUDA 環境 2）の fandhe-ai 行はこの固定オーバーヘッドを含んだ数値であり、GEMM カーネル
  単体の速度差として解釈しない。フレームワーク間でプロトコルが完全一致する比較には次の `--mode reuse`
  （`gemm` タスクのみ）を使う

### `--mode fresh|reuse`（イシュー #925。デバイス/tape 再利用モード）

- 既定は `fresh`（上記「計測ごとに新しい計算グラフを作る」プロトコルと完全に同一。既存 JSONL・集計表との互換維持。
  ただし上記のとおり Metal / CUDA では candle / Burn とプロトコル同一ではない点に注意）
- `reuse`（`bench-fandhe` の `gemm`／`train`／`infer`〈イシュー #1217〉タスクに対応。
  `bench-candle` / `bench-burn` はデバイス再利用が API 上の既定設計のため task に
  依らず対象外で MEASURE_ERROR を返す）:
  tape/デバイスを 1 回だけ構築し、その構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの
  経過時間を `init_s`（JSONL のフィールド。初期化 1 回分のコスト）として分離記録したうえで、
  同一 tape 上で warmup 残り 19 回 → 計測 20 回を回し、`median_s`/`q1_s`/`q3_s` を
  「カーネル実行時間」として記録する。この計測区間（`matmul` + ホスト実体化のみ）は
  candle / Burn の計測区間（デバイス・入力テンソルをループ外で構築済みの `matmul` +
  ホスト実体化）と一致するため、**Metal / CUDA で fandhe-ai を candle / Burn とプロトコル
  同一で GFLOP/s 比較したい場合は `reuse` モードの `median_s`/GFLOP/s を用いる**（`fresh` の
  GFLOP/s ではない）
- **tape 上のノード蓄積に関する注意**: 葉 Var（A・B）は tape 上に 1 回だけ登録して使い回すが、
  matmul の結果ノードは呼ぶたびに tape へ蓄積される（N=2048 で約 16 MiB/回 × 40 回 ≒ 640 MiB。
  N=4096 でも約 2.6 GiB で対象 GPU メモリ内に収まる。長時間・大サイズの reuse 計測では
  メモリ使用量の増加に留意する）
- 使用例: `cargo run --release -p bench-fandhe -- --task gemm --device cuda --size 2048 --mode reuse`

### `train --mode reuse`（イシュー #958。デバイス常駐パラメータ更新）

- `run_train`（`fresh`）は各 SGD ステップでホスト経由の更新（勾配を download → ホストで
  `p - lr*g` → `apply_parameters` で書き戻し）を行っており、candle（`Var::set`）や
  Burn（デバイス上更新）と非対称なプロトコルになっている（#957 背景）。`reuse` は
  イシュー #954 で追加されたデバイス常駐パラメータ更新 API（`fandhe_ai::DeviceParamStore`）
  を使い、`p - lr*g` の更新自体をデバイス上で完結させる
- 参照実装は `crates/facade/tests/device_param_store_train.rs::train_with_device_param_store`
  （`Sequential::init_device_param_store` で全パラメータを 1 回だけ H2D upload → 以後は
  同一 `DeviceParamStore` を使い回す）であり、`run_train_reuse` はその構造に揃える
- **`init_s` の定義**: 初回 tape 構築 + `init_device_param_store`（全パラメータの 1 回限りの
  H2D upload）+ その完了を保証する明示同期点（`sync_device_param_store_to_host`）までの
  経過時間。`bench-fandhe` が依存できる公開 API 面（`fandhe-ai =0.7.0`）には「ホスト転送を
  伴わない完了待ち」が公開されていないため、この同期点は D2H 実体化コストを伴う
  （codex-review PR #998 P2 指摘。`main.rs` の `run_train_reuse` init_s コメント参照）。
  これは `gemm reuse` の `init_s` が「初回 matmul + ホスト実体化」を明示的に含めている前例
  と同じ扱いであり、`init_s` は純粋な H2D upload 時間ではなく「upload 完了を確認可能な
  最初の時点」までの時間として解釈する。以後 100 step（先頭 20 を warmup として除外、
  残り 80 を計測）は gemm 同様 `median_s`/`q1_s`/`q3_s` として記録する。各 step の計測窓は
  デバイス上 SGD 更新の完了を待たずに終える（0.5.0 の `forward_resident` は #1059 で
  D2H を伴わない `register_resident_params` に切り替わっており forward 側の D2H には
  依存しない。代わりにこの step 自身の `loss_readout`〈`loss.to_tensor().get()`〉が
  ストリーム順序保証で前 step の backward/update 完了を含めて同期点として機能し、
  定常状態では窓の境界が 1 step ずれるだけで `forward + backward + update` の総和は
  変わらない。`docs/backend-cuda-async-execution-design.md` §3 I1/I2・`main.rs` の
  ループ冒頭コメント参照）
- **tape は step ごとに新規生成する**（gemm reuse と異なり、tape 自体は使い回さない）:
  `fandhe_ai_autodiff::Tape` はノード列クリア API を持たず学習ループはステップごとに
  tape を生成・破棄する設計契約であり、単一 tape を 100 step 使い回すと `Tape::backward`
  の逆順走査コストが step 数に比例して増加し 1 step の計測時間が非定常になる。reuse で
  使い回すのは tape ではなく `DeviceParamStore`（デバイス常駐バッファ・デバイスを固定
  する側）であり、fresh/reuse の計時差は「ホスト経由 SGD vs デバイス常駐更新」に限定される
- **既知の前提（改善量の解釈範囲・codex-review PR #1104 P2 是正）**: 0.5.0 の
  `Sequential::forward_resident` が呼ぶのは `DeviceParamStore::register_resident_params`
  （D2H を伴わない。#1059 で D2H を伴う旧 `register_resident_leaves` から分離。
  `crates/autodiff/src/optim/device_store.rs` doc 参照）であり、forward 自体に毎 step
  の D2H は発生しない。各 step 中の唯一のホスト同期点は `loss_readout`
  （`loss.to_tensor().get()`）であり、これが（1 step ずれた形で）前 step の
  backward・デバイス上 SGD 更新の完了を保証する（`docs/backend-cuda-async-execution-
  design.md` §3 I1/I2）。reuse が排除するのは「毎 step のホスト経由 `p - lr*g` 計算 +
  再アップロード（H2D）」であり、パラメータの D2H（forward 用）は 0.5.0 では構造的に
  発生しない
- **数値一致確認**（受け入れ条件 5）: `cargo test --locked --release -p bench-fandhe` に
  `train_reuse_matches_fresh_final_loss_within_composite_tolerance`（cpu・実機非依存。
  fresh/reuse の最終 loss を統一複合判定で突合）を含む
- 使用例: `cargo run --release -p bench-fandhe -- --task train --device cuda --mode reuse`
- スイープ（`run_all*.sh` の (b') ループ）・集計（`summarize.py` の (b') 節）・
  cpu/cuda 実測（RTX 3060。環境 4）は #959 で実装済み。Apple Silicon 実機（cpu/metal）は
  環境 5（`results/summary.md`。イシュー #957）で実測済み。DGX Spark GB10（cuda）での実測は
  未計測（`results/summary.md` 環境 4「計測不可・未計測項目」参照。再現コマンドを記載済み）
- **(b') 節の読み方**（`results/summary.md`・`summarize.py` 出力）: `初期化(init_s)` は
  `DeviceParamStore` 構築 1 回分のコスト、`中央値/Q1/Q3` は以後 80 step の 1 step あたり
  時間（`fresh` と同一プロトコル）。`fresh 中央値（参考）` と `fresh/reuse 比` で
  ホスト経由 SGD（fresh）との速度差を確認できる。`最終 loss 突合（fresh）` は
  fresh/reuse の最終 loss（checksum）を本体の数値一致契約（相対誤差 1e-3 未満 または
  絶対誤差 1e-5 未満）で突合した結果（`一致`/`不一致`/`突合不能`）。gemm の (a') と異なり
  フレームワーク間（fandhe-ai vs candle/Burn）の checksum 突合は行わない（重み初期化が
  異なる設計のため最終 loss が一致しない。上記モジュール doc・`summarize.py` docstring 参照）

### `train --phases`（イシュー #1009。1 step のフェーズ分解）

`run_train`（fresh）/`run_train_reuse`（reuse）は 1 step の合計時間しか記録せず、
fandhe-ai の train 1 step が candle/Burn より 1 桁以上遅い理由（tape 構築・forward・
backward・パラメータ更新のどこが支配的か）を追跡できない。`bench-fandhe` の
`--task train --phases`（値なしフラグ）はこの 1 step を公開 API の呼び出し境界で
区間分解し、区間ごとの median/Q1/Q3 を `task:"train_phases"` の JSONL 行として出力する。
**`bench-fandhe`（`--task train`）専用**であり、`bench-candle`/`bench-burn` や
`--task gemm --mode fresh` との組合せは MEASURE_ERROR で fail-fast する
（`--task gemm --mode reuse --phases` は下記「`gemm --mode reuse --phases`」節、
`--task infer --phases` は下記「`infer --mode reuse` / `infer --phases`」節
〈イシュー #1217〉をそれぞれ参照）。

区間は「公開 API のどの呼び出しに時間が乗るか」を表し、GPU 内部（カーネル／転送）の
内訳ではない: fandhe-ai 0.7.0 の `Tensor<f32>` はホスト常駐で、CUDA/Metal の各演算は
演算ごとに H2D→カーネル→D2H を行う（`fandhe-ai-backend-cuda-0.7.0/src/ops.rs::gemm`）。
また `matmul` は即時実行、elementwise（relu・mse）は実体化境界（`to_tensor()`/`get()`）まで
遅延する（TASK-12.1d）。

**fresh の区間定義**（`run_train` と同一の処理順・API 呼び出しを `Instant` で分割）:

| phase | 計測対象 |
| --- | --- |
| `tape_build` | `make_tape(&cli.device)` |
| `leaf_register` | `model.bind(&tape)` + 入力/教師データの `tape.var(...)` |
| `forward` | `bound.forward(&tape, &x)` + `pred.mse_loss(&y)`（matmul 即時実行・elementwise 遅延） |
| `loss_readout` | `loss.to_tensor().get(&[])`（遅延 elementwise の実体化 = 同期点） |
| `backward` | `tape.backward(&loss)` + `bound.trainable_grads(&grads)` |
| `param_readout` | param/grad の `contiguous().as_slice().to_vec()`（D2H） |
| `host_sgd` | `p - LR*g` の計算 + `Tensor::from_slice`（ホスト計算） |
| `apply_params` | `model.apply_parameters(next)`（H2D 位置。層再構築） |
| `tape_drop` | `bound`/`param_refs` の解放 + `drop(tape)`（テンソル解放） |
| `step_total` | step 全体のウォールクロック時間（検算用。Σphase ≤ step_total） |

**reuse の区間定義**（`run_train_reuse` と同一）:

| phase | 計測対象 |
| --- | --- |
| `tape_build` | `make_tape(&cli.device)` |
| `leaf_register` | 入力/教師データの `tape.var(...)` |
| `forward_resident` | `model.forward_resident(&tape, &x, &mut store)` + `mse_loss`（`register_resident_params` 経由〈#1059〉で D2H を伴わない。`mse_loss` 自体も遅延実体化） |
| `loss_readout` | `loss.to_tensor().get(&[])` |
| `backward` | `tape.backward_device_param_store(&loss, &store)`（0.5.0 から `forward_resident` が積む `Op::LinearResident` の解決に必須。イシュー #1059） |
| `device_update` | `tape.step_device_param_store(&mut store, &grads, &config)`（grad H2D + デバイス上 SGD 発行。CUDA では非同期発行のため完了待ちは次 step の `forward_resident` に計上される。**CPU のみ #1212 以降**、`Op::LinearResident` の weight 勾配はデバイス常駐 staging へ backward 内で直接書き込み済みのため本フェーズでの H2D を含まず、bias 等の勾配のみ H2D する。CUDA／Metal は未対応のため従来どおり全パラメータぶん H2D する。`docs/perf/train-resident-grad-device-update.md` 参照） |
| `tape_drop` | `drop(tape)` |
| `step_total` | 検算用 |

**「同期待ち」を独立区間にできない理由**: `fandhe-ai =0.7.0` の公開 API 面には
ホスト転送を伴わない完了待ち（`bench-harness::sync::SyncPoint::wait_idle` 相当）が
公開されておらず（`run_train_reuse` の `init_s` コメント・PR #998 P2 と同じギャップ）、
同期は必ず `loss_readout`（実体化）の D2H を通じて発生する。そのため「同期待ち」は
独立区間にはできず `loss_readout` へ計上される。

reuse 行には `init_s`（`DeviceParamStore` 構築コスト。`run_train_reuse` と同一定義）が
乗る。`--phases` 実行時は既存の `task:"train"` 行は出さない（`step_total` 行が代替する。
計時分割つきの step 合計を通常プロトコルの値と混同させないため）。

**JSONL スキーマ**: 既存 `Record` のキー（`framework`・`version`・`task:"train_phases"`・
`device`・`size`・`median_s`/`q1_s`/`q3_s`・`checksum`・`warmup`・`iters`・`mode`・
reuse のみ `init_s`）に加え、`phase`（区間名）・`phase_index`（出力順。0 始まり）の
2 キーを末尾に追加する（`bench_common::PhaseRecord`）。

**`summarize.py` (b'') 節の読み方**: `(device, mode)` ごとに `phase_index` 昇順で表示し、
`中央値`/`Q1`/`Q3` に加え `step_total 比`（= phase 中央値 / step_total 中央値。100% に
近いほど支配的な区間）を表示する。表末尾の「フェーズ合計（中央値の和）」は参考値であり
（中央値は加法的でないため `step_total` と厳密には一致しない）、`step_total` 行の欠落・
`phase`/`phase_index` の不正や重複・phase 中央値が `step_total` を超える不整合は
「無効」表示され `--strict` の対象になる。`tape_build` 等の sub-100 ns 区間は、9 桁
固定小数シリアライズ自体は ns 単位を表現できる（41 ns なら `0.000000041`）ため丸まらない
が、計時クロックの分解能未満の間隔しか空かない標本では `Instant::now()` の連続呼び出しが
同一時刻を返し区間長が `0.000000000`（= `0.0 µs` 表示）と計測されることがある。
`step_total` 以外の phase 行に限りこれを妥当な下限として許容する（`step_total`・`init_s`
は引き続き 0 秒を不正値として扱う。イシュー #1010・`summarize.py` の
`_safe_phase_time_s` 参照）。

使用例:

```bash
cargo run --release -p bench-fandhe -- --task train --device cpu --mode fresh --phases
cargo run --release -p bench-fandhe -- --task train --device cuda --mode reuse --phases
```

Metal（M4 Max）・DGX Spark GB10 実機での計測結果は
`docs/perf/train-step-phase-breakdown.md`（イシュー #1010）を参照。
`fandhe-ai =0.6.0` ピンでの再計測は同ドキュメント §10〜（イシュー
#1145）を参照。`results/summary.md` への環境情報の統合記録は別途
#1050 に委ねる。

### `gemm --mode reuse --phases`（イシュー #1182。reuse 計測境界のフェーズ分解）

`#1142`（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §4.3・§8）は、
カーネル単体（launch-only）計測では candle を上回るのに `gemm --mode reuse` の
計測境界では candle 比未達のままである原因を「reuse の計測境界に残る H2D／D2H／
同期の固定費」と**推定**したまま確定していなかった。`--task gemm --mode reuse
--phases`（値なしフラグ。`--task train --phases` と同型）はこの推定を、
`train --phases` と同じ方法論（公開 API の呼び出し境界での区間分解）で実測確定
するための計装である。`bench-candle`/`bench-burn`・`--task gemm --mode fresh`
との組合せは MEASURE_ERROR で fail-fast する（`--task train`/`--task infer`
はそれぞれ別の `--phases` 対応を持つ。上記「`train --phases`」節・下記
「`infer --mode reuse` / `infer --phases`」節参照）。

`run_gemm_reuse` 1 反復の内側で `readout_var`（`to_tensor()` +
`contiguous().as_slice().to_vec()`）を展開し、次の 5 区間を `Instant` で計測する
（`run_gemm_reuse` 本体は変更しない。`init_s` の定義・`validate_gemm_checksum`
を全反復で実施する点・`GemmReference::verify` を計測窓外で worst 集約する点は
`run_gemm_reuse` と同一）:

| phase | 計測対象 |
| --- | --- |
| `matmul` | `a.matmul(&b)`（**H2D〈A/B アップロード〉・カーネル実行・D2H〈結果ダウンロード〉・ストリーム同期が全てこの区間の内側に閉じている**。公開 API ではこれ以上分離できない） |
| `to_tensor` | `c.to_tensor()` |
| `host_copy` | `.contiguous().as_slice().to_vec()`（ホストへのコピー） |
| `checksum` | 全要素和（f64 アキュムレータ） |
| `iter_total` | 反復全体のウォールクロック時間（検算用。Σphase ≤ iter_total） |

`matmul` 区間の内訳（H2D／カーネル専有時間／D2H の実測分解）は本節では取れない
（`fandhe-ai` 0.7.0 の公開 API 面にホスト転送を伴わない完了待ちや区間別の
カーネルタイミング API が無いため。`train --phases`「同期待ちを独立区間にできない
理由」と同じギャップ）。内訳は `crates/backend-cuda`（CUDA）・`crates/backend-metal`（Metal。
`cargo run --release -p bench-fandhe -- --task gemm --device metal --size 4096 --mode
reuse --phases`。イシュー #1189）それぞれの側の診断テスト
（`gemm_reuse_phase_diag_tests`。`#[ignore]` 実機専用）が別途取り、`matmul`
区間との突合結果を `docs/perf/cuda-gemm-reuse-phase-breakdown.md`（CUDA）・
`docs/perf/metal-gemm-reuse-phase-breakdown.md`（Metal）にそれぞれ記録する。

reuse 行には `init_s`（tape 構築 + 葉 Var 登録 + 初回 matmul + ホスト実体化までの
経過。`run_gemm_reuse` と同一定義）が乗る。

**JSONL スキーマ**: 既存 `Record` のキー（`framework`・`version`・
`task:"gemm_phases"`・`device`・`size`・`median_s`/`q1_s`/`q3_s`・`checksum`・
`warmup`・`iters`・`mode:"reuse"`・`init_s`・`parity_*`）に加え、`phase`・
`phase_index` の 2 キーを末尾に追加する（`train_phases` と同じ
`bench_common::PhaseRecord`）。`gemm_phases` 行は `(a)` GEMM 節・`--target`
目標達成ゲート・`compare_gemm_gate.py`（`_matching_rows` が `task != "gemm"` を
除外する）には一切混入しない。

**`summarize.py` (a'') 節の読み方**: `(device, mode, size)` ごとに `phase_index`
昇順で表示し、`中央値`/`Q1`/`Q3` に加え `iter_total 比`（= phase 中央値 /
`iter_total` 中央値）を表示する。検証方針（必須 phase 名の集合・順序・件数の
完全一致・`iter_total` の一意性・phase 中央値が `iter_total` を超える不整合の
検出・sub-ns 区間の 0 値許容）は `(b'')` train_phases 節と同一。

使用例:

```bash
cargo run --release -p bench-fandhe -- --task gemm --device cuda --size 4096 --mode reuse --phases
```

`gemm --mode reuse --phases` は診断専用であり、`run_all*.sh`／`run_gemm_gate*.sh`
の標準スイープには組み込まない（既存ゲート判定プロトコルを変更しないため）。
GB10 実機での計測結果・カーネル専有時間ベースの candle 比（参考値）は
`docs/perf/cuda-gemm-reuse-phase-breakdown.md`（イシュー #1182）を参照。

#### CPU での区間定義と Layer B（イシュー #1290）

CPU は上記 5 区間（`matmul`／`to_tensor`／`host_copy`／`checksum`／
`iter_total`）を **コード変更なしで** 計測できる（`--device cpu` は
`dispatch` の `("gemm","reuse",true)` 分岐に device 分岐が無いため）。
CPU gate 対象形状の下限（README「GEMM ゲート 5 回計測」節。
`cpu={512,1024,2048}`）で完走することを `gemm_reuse_phases_cpu_smoke_n512`
（`bench-fandhe/src/main.rs` テスト）が固定している:

```bash
cargo run --release -p bench-fandhe -- --task gemm --device cpu --size 512 --mode reuse --phases
```

`matmul` 区間の内訳は CPU では次のように対応する（CUDA #1182 の
H2D A/B・alloc_c（プール経由）・launch_issue・kernel_wait・d2h、Metal
#1189 の upload_a/b・alloc_c・encode・commit_wait・readback と対比）。
CPU にはホスト⇄デバイス転送・ストリーム同期が存在しない（ホスト常駐の
まま演算する）ため、`matmul` 区間は「Arc clone（`materialize_fallible`
の実体化）」＋「C 確保（`zeroed_output(n*n)`。イシュー #1299 でしきい値
以上の rayon 並列ゼロ書き込み分岐を追加したが、M4 Max スモーク実測で
N=2048 が後退したため本番既定 `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS =
usize::MAX` により無効化。#1301 が DGX 実機実測で有効化可否を判断する）」
＋「マイクロカーネル本体
（`gemm_blis_parallel`）」＋「`Tensor::new` によるラップ」＋「autodiff
ノード push（`push_eager`）」の合成である:

| Layer A `matmul` の内訳 | CPU 実体 | 対応する Layer B 区間（`crates/backend-cpu`） |
| --- | --- | --- |
| （H2D 相当なし。値渡しではなく `Arc` 共有） | `Var::matmul` 冒頭の `materialize_fallible(..).clone()`（A・B 各 1 回） | 計測対象外（診断側では呼び出し元でループ外に 1 回だけ `contiguous().as_slice()` した結果を渡し、`kernel` 計時窓から除外する） |
| C 確保 | `zeroed_output(n*n)`（#1299） | `alloc_c` |
| カーネル実行 | `gemm_blis_parallel`（本番 NN 経路。RowPanel・既定スレッド数） | `kernel` |
| （D2H 相当なし） | `Tensor::new(out, &out_shape)` | `tensor_wrap` |
| （同期相当なし） | `push_eager`（tape ノード追加） | `tape_matmul` − `ops_gemm` の残差（autodiff オーバーヘッドの近似） |

`to_tensor`／`host_copy`／`checksum` は Layer A と 1:1 で対応する
（`c_var.to_tensor()` → `.contiguous().as_slice().to_vec()` → f64 全要素
和。§下記表参照）。

**何を固定費に含めるか**（CUDA §6／Metal §6 と同じ切り分け）:

- `host_copy`＋`checksum` は **ハーネス自身の診断コスト**（#965/#970 の
  既存契約と同じ。本番経路には乗らない）。
- `alloc_c` と autodiff 残差（`tape_matmul` − `ops_gemm`）は
  **facade/autodiff 呼び出しの固定費**（削減対象候補。イシュー #1294）。
- `kernel` が実ペイロード。`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
  §8.1 の「カーネル単体 GFLOP/s」（`gemm_blis_variant_ab_*`・OSS 直接
  比較ハーネス）との突合には `kernel` 区間の値を使う。

**実行コマンド**（実機・5 回独立プロセス起動・中央値。
`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」）:

```bash
cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored \
  gemm_reuse_phase_diag_cpu --nocapture --test-threads=1
```

**忠実性の注意点**（`crates/backend-cpu/src/gemm_reuse_phase_diag_tests.rs`
冒頭コメントに同一内容あり）:

- **keep-alive**: 本番 reuse（`run_gemm_reuse`）は matmul の出力
  `Tensor` をアロケータへ返却せず毎反復新規ページに書く一方、readout
  （ホストコピー `Vec<f32>`）は反復ごとに破棄する。診断側もこれに揃え、
  `kernel`／`ops_gemm` 各パスの出力 `Tensor`（`wrapped`／`ops_out`）
  のみを保持しアロケータのページ再利用でコストが消える乖離を避け、
  readout コピーは保持しない（`tape_matmul` パスの出力は tape 自身が
  内部で保持するため追加の保持は不要）。
- **calloc／first-touch の帰属**: `zeroed_output` は本番既定
  （`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS = usize::MAX`）では常に
  `vec![0.0; n*n]` へ倒れるため、大サイズでは OS の遅延ゼロページに
  倒れうるため、初回書き込みの page-fault コストは `alloc_c` ではなく
  `kernel`（実際に書き込む側）に計上されうる。`alloc_c` を「確保コスト
  の上限」と読まない。**イシュー #1299** はしきい値以上で `alloc_c`
  区間内に並列ゼロ書き込み（first-touch を複数スレッドへ前倒しで
  分散）する分岐を追加したが、M4 Max スモーク実測で N=2048 の
  `alloc_c` が約 3〜22 倍・`ops_gemm` 合成が中央値約 29% 後退することを
  確認したため無効化した（`docs/perf/cpu-matmul-fixed-cost-impl.md`）。
  Linux（DGX Spark GB10）では帰属の曖昧さが解消される可能性が残るため
  #1301 が実機実測で有効化可否を判断する。

**突合前提**（`docs/perf/cpu-gemm-candle-gate-remeasurement.md` への
転記時に明記する）: Layer A（`gemm --mode reuse --phases`）は
crates.io 公開版 `fandhe-ai =0.7.0` で計測する一方、Layer B（本節の
診断テスト）は HEAD で計測する。突合には CPU GEMM 本番経路
（`gemm_blis_parallel`・`CpuBackendOps::gemm`・`Var::matmul`）に
`fandhe-ai =0.7.0` タグ以降の非コメント差分が無いことの確認が前提で、
`git diff v0.7.0..HEAD -- crates/backend-cpu/src crates/autodiff/src
crates/facade/src crates/tensor-core/src` を確認した結果（2026-09-07・
本イシュー #1290 時点）:
`crates/autodiff/src/var.rs`（`Var::matmul` 自体は変更なし・別メソッド
の追加のみ）・`crates/backend-cpu/src/ops.rs`（`CpuBackendOps::gemm`
自体は変更なし・`MemoryOps` へのメソッド追加のみ）・
`crates/facade/src/lib.rs`・`crates/tensor-core/src/{buffer,pool,tensor}.rs`
はいずれも既存関数の呼び出し経路を変えない**追加のみ**。
`crates/backend-cpu/src/gemm_blis/mod.rs`（627 行差分）は
`gemm_blis_parallel` を含む複数箇所の並列度算出を
`rayon::current_num_threads()` から `crate::thread_limit::
effective_num_threads(...)`（イシュー #1363）へ差し替えているが、
同関数は既定ゲート `thread_limit::BIG_CORE_LIMIT_ENABLED = false`
（#1364 の両実機実測で REJECT・差し戻し確定済み。
`docs/perf/cpu-gemm-default-thread-limit.md` §6）のときは
`rayon::current_num_threads()` をそのまま返す恒等写像であり、
`gemm_blis_parallel` の NN 経路（本診断が計測する経路）の挙動は
`fandhe-ai =0.7.0` と不変。したがって突合前提は成立する。

実測（両実機・5 回独立プロセス中央値）は
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` へ記録する
（イシュー #1292）。`gemm --mode reuse --phases`（CPU 含む）は診断専用
であり、上記のとおり `run_all*.sh`／`run_gemm_gate*.sh` の標準スイープ
には組み込まない。

#### 借用ビュー readout（既定経路。イシュー #1337・#1436・#1437・#1438）

上記 `#1182` が確定した結論（`host_copy`〈`.to_vec()` の memcpy〉が
`iter_total` の 25.6〜53.0%〈CUDA〉を占める＝ハーネス自身の診断コストが
candle 比未達の主因）を受け、`readout_var`（`to_tensor` + `host_copy` の
2 区間の実装）自体を借用ビュー（`fandhe_ai::VarHostView::host_view()`／
`Tensor::host_slice()`。イシュー #1335・#1336）へ切り替えた。当初は
`bench-fandhe` の cargo feature `host-view-readout`（既定 OFF）として導入
したが、CUDA D2H 宛先未タッチによる N=1024/2048 の後退を `#1436` が診断し
`#1437` が是正（`ReadbackDest::PretouchedFresh`）して全 N 非後退を確認した
うえで、`#1438` が feature を撤去して **bench-fandhe の既定経路**とした。

- **crates.io 公開版 `fandhe-ai =0.7.0` には該当 API が未収録**のため、
  ピン未更新の間は `managed-placement`（イシュー #1353）と同じ方式で
  `crates/facade` への path patch を CLI `--config` 経由で併用する必要が
  ある（`bench_fandhe_pin_guard.sh` が未併用時をビルド起動前に fail-closed
  で検知する）:
  ```bash
  cargo test -p bench-fandhe     --config 'patch.crates-io.fandhe-ai.path="<絶対パス>/crates/facade"'
  ```
  `[patch]` は Cargo.lock・`.cargo/config.toml` へは一切書かず invocation
  限定（deps-policy.md 第 9 区分の承認済みピン `fandhe-ai =0.7.0` を壊さない）。
  crates.io 次回公開でピンが借用ビュー API を収録した版へ更新されたら、
  更新 PR が `bench_fandhe_pin_guard.sh` と各スクリプトの呼び出し箇所・
  本節の注記を削除する。
- **区間定義**（区間名・順序・件数〈5 区間〉は旧 legacy 経路と不変）:
  - `to_tensor` 区間 = `Var::host_view()` の構築（`materialize_non_fallible
    (..).contiguous()`。contiguous な場合は `Tensor` 内部 `Arc` の複製のみ）
  - `host_copy` 区間 = 借用スライスの取得（`Deref::deref`。追加コピーなし。
    想定値 ≈0）
  - `checksum`／`matmul`／`iter_total` は無変更
- **checksum／parity 契約は不変**（`readout_var` の戻り値は `Deref<Target=
  [f32]>` で、`checksum_var`／`GemmReference::verify` の呼び出し側は旧
  legacy 経路と無変更）。`readout_var_matches_legacy_to_vec_bit_exact`
  （`main.rs` テスト）が legacy 経路（`to_tensor()` + `to_vec()`。テスト内に
  インライン展開して保持）と bit 同一であることを固定する。
- **CUDA `#1336` の pinned host staging（`MemoryOps::with_host_view`）は
  本経路に到達しない**: `Var::matmul` の出力は `gemm` バックエンド内部の
  readback で既にホスト常駐 `Tensor` になっているため（`docs/perf/
  cuda-host-view-staging-readout.md` §7）。誤帰属を避けるため実測記録
  （`docs/perf/{cuda,metal,cpu}-gemm-candle-gate-remeasurement.md`）にも
  明記する。
- **`bench-candle` は不変**（`.to_vec2()` による所有 `Vec` 読み出しのまま）。
  fandhe-ai 側だけが借用ビュー・candle 側が所有コピーという非対称は残る
  ため、candle 比の数値を読む際はこの非対称を踏まえる（公正性の論点。
  各 remeasurement doc に明記）。
- **`run_gemm_gate.sh`** は借用ビュー readout が既定経路のため feature 指定
  不要（旧 `GEMM_GATE_BENCH_FANDHE_FEATURES` は撤去済み）。ピン未更新の間は
  `GEMM_GATE_PATCH_FACADE_PATH`（HEAD ツリーへの path patch）が正式系列
  でも必須になる（下記「GEMM ゲート 5 回計測」節参照）。
- **`compare_gemm_ab.py`** は `--device cuda`・`--sizes gate`（cpu={512,1024,2048}・
  cuda/metal={1024,2048,4096}への絞り込み）・`--modes reuse`（cuda/metal の
  ゲート出力が reuse のみのため fresh 参考行をセル外扱いにする）に対応する
  （下記「GEMM ゲート 5 回計測」節参照）。
  （下記「GEMM ゲート 5 回計測」節参照）。

### `infer --mode reuse` / `infer --phases`（イシュー #1217）

`docs/perf/train-step-phase-breakdown.md` §13・§15.5 の背景: `--task infer` は
これまで CPU が `Sequential::predict`、CUDA/Metal が `make_tape` + `tape.var` +
`model.forward` の **fresh 経路のみ**で、`--task infer --phases` は
`dispatch()` が MEASURE_ERROR を返していた（infer の candle 比未達の内訳
〈H2D/D2H・forward の寄与〉が実測できていなかった）。本イシューは
`Sequential::predict_resident`（facade 公開 API。`DeviceParamStore` の
デバイス常駐重みで forward する。0.6.0 で公開済み）を使う `--mode reuse` と、
公開 API 呼び出し境界での区間分解（`--phases`。fresh/reuse 双方に対応）を
`bench-fandhe` へ追加する。

**`infer --mode reuse`**: `train --mode reuse`（イシュー #958）と同じ
「初期化コストとカーネル実行の分離」の考えを推論へ適用する。`init_s` は
`train --mode reuse` と同一定義（`make_tape` + `model.init_device_param_store`
+ `sync_device_param_store_to_host` による完了保証の同期）で、以後
`model.predict_resident(&store, &x_data)` を `WARMUP_ITERS`（20）+
`MEASURE_ITERS`（20）回呼ぶ（`predict_resident` 内部で tape を毎呼び出し
生成・破棄するため、`gemm`/`train` の reuse と異なり warmup を init 側で
消費する必要が無く、`run_infer`〈fresh〉と同じ反復数をそのまま使う）。

- 数値一致確認: `cargo test --release -p bench-fandhe`
  `infer_reuse_matches_fresh_checksum_within_composite_tolerance`（cpu・
  実機非依存。fresh/reuse の checksum を統一複合判定で突合）
- 使用例:
  `cargo run --release -p bench-fandhe -- --task infer --device metal --mode reuse`

**`infer --phases`**: `mode`（fresh/reuse）と `device`（cpu か否か）の
組合せで区間集合が異なる（公開 API の呼び出し粒度が異なるため）。

| mode | device | 区間（`phase_index` 順） | 計測対象 | 分離不能な内訳 |
| --- | --- | --- | --- | --- |
| fresh | cpu | `predict` / `host_copy` / `checksum` / `iter_total` | `model.predict(&x)` / ホストコピー / 全要素和 / 反復全体 | `predict` 内部（層ごとの `forward_host`）は公開 API 上単一呼び出しでこれ以上分離できない |
| fresh | metal・cuda | `leaf_register` / `forward` / `to_tensor` / `host_copy` / `checksum` / `iter_total` | `tape.var(&x)` / `model.forward(&tape,&x)` / `out.to_tensor()` / ホストコピー / 全要素和 / 反復全体（`make_tape` は `run_infer` と同じく計測窓外） | `forward` の内側に `Linear::bind` の重み clone + H2D・演算ごとのカーネル実行・D2H・ストリーム同期が全て閉じる（`gemm --mode reuse --phases` の `matmul` 区間と同じギャップ） |
| reuse | cpu・metal・cuda | `predict_resident` / `host_copy` / `checksum` / `iter_total` | `model.predict_resident(&store,&x)` / ホストコピー / 全要素和 / 反復全体 | `predict_resident` 内部（`tape_for`・`snapshot_resident_params`・層ごとの `gemm_resident_rhs_act`）は private ヘルパー `forward_from_flat_leaves` を経由するため公開 API からは分離できない |

`to_tensor`/`host_copy`/`checksum`/`iter_total` は `gemm --mode reuse
--phases`（イシュー #1182）の `readout_var` 展開と同一の意味（`Var`/
`Tensor` のホスト実体化 → コピー → 総和）で、`bench-fandhe/src/main.rs` 側
では同じ `PHASE_GEMM_*` 定数を再利用する（`phase` は task ごとの名前空間
ではなく区間の意味を表す値のため）。fresh・GPU の `make_tape` を計測窓外
に置く扱いは `run_infer` の既存プロトコル（`--phases` を付けない通常計測）
と同一で、`(c)` 節の既存数値の前提を変えない。

reuse 行には `init_s`（`infer --mode reuse` と同一定義）が乗る。`--phases`
実行時は既存の `task:"infer"` 行は出さない（`iter_total` 行が代替する。
`train`/`gemm` の `--phases` と同じ方針）。

**JSONL スキーマ**: 既存 `Record` のキー（`framework`・`version`・
`task:"infer_phases"`・`device`・`size`・`median_s`/`q1_s`/`q3_s`・
`checksum`・`warmup`・`iters`・`mode`・reuse のみ `init_s`）に加え、`phase`・
`phase_index` の 2 キーを末尾に追加する（`train_phases`/`gemm_phases` と
同じ `bench_common::PhaseRecord`）。`infer_phases` 行は `(c)`/`(c')` 推論節・
`--target` 目標達成ゲートには一切混入しない（`task != "infer"` の行を除外
する既存ロジックと同型）。

**`summarize.py` (c')/(c'') 節の読み方**: `(c')` は `infer --mode reuse` の
デバイス別スループット表（`(b')` train reuse と同型に加え `throughput_per_s`
〈バッチ/秒〉列を持つ）。`(c'')` は `(device, mode)` ごとに `phase_index`
昇順で表示する `infer --phases` の区間分解（`iter_total 比`・フェーズ合計
は `(a'')`/`(b'')` と同一方針）。検証方針（必須 phase 名の集合・順序・
件数の完全一致・`iter_total` の一意性・phase 中央値が `iter_total` を
超える不整合の検出）も同一だが、必須 phase 集合は `(mode, device_class)`
ごとに異なる（上表参照。`device_class` は `cpu`/`gpu`〈metal・cuda〉）。

使用例:

```bash
cargo run --release -p bench-fandhe -- --task infer --device metal --mode reuse
cargo run --release -p bench-fandhe -- --task infer --device metal --mode fresh --phases
cargo run --release -p bench-fandhe -- --task infer --device cpu --mode reuse --phases
```

`run_all*.sh` の標準スイープに `infer --mode reuse`・`infer --phases`
（fresh/reuse 双方。batch=64）を組み込み済み（`train --mode reuse`/
`train --phases` と同じ位置づけ）。M4 Max 実機での初回計測結果は
`docs/perf/infer-reuse-phase-breakdown.md`（イシュー #1217）を参照。

### 要素単位検証（イシュー #970）

`(a)` GEMM の checksum（全要素和）は、要素の入れ替わりや正負誤差の相殺で偶然一致しうる破損を
見逃す。3 バイナリ（`bench-fandhe`/`bench-candle`/`bench-burn`）は `gemm` タスクの各反復で、
結果を参照実装と**要素単位**で突合し、反復間の worst-case を JSONL の 4 フィールド
（`parity_total`・`parity_fail_count`・`parity_max_abs_err`・`parity_max_rel_err`）として記録する。

- **参照実装**: `bench-common::GemmReference`。本体 `backend-cpu::parity::matmul_reference_fma`
  と同じ FMA 契約（f32 `mul_add`・逐次 k 昇順の演算順序固定）を持つ自前 GEMM を、行ブロック分割で
  `std::thread::scope` 並列化したもの（各 `c[i][j]` の累積鎖は k 昇順のまま = 逐次実装と bit 完全
  一致。`bench-common::parity::tests::compute_is_bit_identical_to_sequential_k_ascending`）。
  fandhe-ai 0.7.0（crates.io 版）の facade は parity API を公開しておらず、candle/Burn を参照に
  すると別途バイナリ間で結果を受け渡す仕組みが要る。自前参照は各バイナリが自己完結で計算できる
  ため採用した（f64 累積の参照は「真値との差」という別の指標になり本体契約と整合しないため不採用。
  結果テンソルをファイルへダンプして summarize.py 側で突合する方式は N=4096 で 64 MiB/行になり
  コミット・転送が非現実的なため不採用）
- **閾値**: `PARITY_ABS_TOL = 1e-5`・`PARITY_REL_TOL = 1e-3`。本体の数値一致契約
  （`.claude/rules/coding-rust.md`「バックエンド構成」節）と同値であり、緩和はユーザー承認必須
- **タイミング**（x86_64・12 コアホストでの実測値。`std::thread::available_parallelism()`
  ベースで並列化されるため実効値は環境依存）: 参照 GEMM の計算は warmup 前・計測窓の外で
  1 回だけ（N=1024 で約 210 ms、N=4096 で約 12.8 s）。要素単位の比較（`compare_elementwise`。
  単スレッド）自体は毎反復（warmup 含む）行うが `start.elapsed()` の**後**（O(n²) の比較コストが
  O(n³) の GEMM 計測時間へ混入しないようにするため。checksum の計算・`validate_gemm_checksum`
  は従来どおり計測窓内のまま変更しない）で、N=4096 で 1 回あたり約 61 ms（40 反復合計で
  約 2.4 s）。合計するとバイナリ 1 回起動あたり N=4096 で参照計算 + 全反復比較の合計は
  約 15 秒程度であり、毎反復ではなくバイナリ 1 回起動（`run_all*.sh` の 1 組み合わせ）あたりの
  追加コストとしては許容範囲と判断した（GEMM 自体の計測窓には影響しない）
- **`summarize.py` の判定**: `parity_fail_count > 0`、または 4 フィールドの型・値が不正（`null` 含む）
  な行を「無効（要素誤差超過）」として表で表示し GFLOP/s を `-` にする（`parity_status`）。本フィールド
  追加前の JSONL（キー欄自体が無い）は「無効」ではなく「未検証（旧形式）」として区別する（キー欠損と
  `null` を混同しない。データ有効性節・`--strict` 対象）。イシュー #1250: 承認済み契約（#1241）下の
  第 3 救済項キー（`parity_scaled_abs_bound`／`parity_scaled_abs_rescued`）が**両方存在する場合のみ**
  追加検証する（後方互換。新 2 キーがともに欠損の 4 キー行は挙動が変わらない）。一方のみ存在・
  `bound=null`・`rescued` が値域外・`framework=="fandhe-ai"` での `rescued>0` はいずれも「無効」に
  倒す（判定式そのものは再計算しない。詳細は `parity_status` docstring 参照）
- `train`/`infer` タスクは対象外（fandhe-ai の重み初期化が candle/Burn と異なる設計のため checksum
  同様に比較不能。§「計測プロトコル」重み初期化の節を参照）

#### fail 要素ダンプ（イシュー #1183）

`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 は、N=2048 で candle-core 側 CUDA GEMM
出力が上記の複合判定で 2 要素 fail（`max_rel≈2.8e-01`）となり原因未確定のまま残っていることを
記録している。同節が検討し未実施だった「fail 要素の値（index・reference・実測値）を取得する診断
計装」を、環境変数 opt-in で追加した。

- **環境変数**: `FRAMEWORK_COMPARE_PARITY_DUMP`（`bench-common::parity::PARITY_DUMP_ENV`）。値の
  解釈は allowlist（`BenchError::InvalidMode` と同型の fail-fast 検証。security.md A03）:
  - 未設定・`""`・`"0"` → 無効（既定。**JSONL 出力・判定結果・終了コードは完全に不変**）
  - `"1"` → 有効・1 回の `verify` 呼び出しあたり既定上限 64 要素まで出力
  - 正の整数文字列（例 `"16"`） → 有効・その値を上限に出力
  - それ以外（`"abc"`・`"-1"` 等）→ 起動直後（warmup 前）に `MEASURE_ERROR:` prefix 付きの
    型付きエラーで終了（20 反復後に落ちるのではなく fail-fast）
- **出力先・形式**: **stderr 限定**（stdout は JSONL チャネルのため混入させない）。fail 要素 1 件
  につき 1 行（`PARITY_DUMP call=<verify 呼び出し番号> n=<N> idx=<flat index> row=<i> col=<j>
  ref=<10進> ref_bits=0x<f32 bit パターン> actual=<10進> actual_bits=0x<f32 bit パターン>
  abs=<絶対誤差> rel=<相対誤差>`）+ 呼び出しごとの末尾サマリ 1 行
  （`PARITY_DUMP_SUMMARY call=<k> n=<N> fail_count=<c> dumped=<d> truncated=<true|false>`）。
  bit パターンを併記するのは、10 進表記だけでは §5.3 の「0 近傍の丸め誤差」仮説の検証に不足するため
- **反復間の重複出力は仕様**: N=2048 のように決定的に同じ要素が毎反復 fail するケースでは、同じ
  index が warmup・計測の各反復で繰り返しダンプされる（反復間の非決定性も可視化するための設計であり
  バグではない）
- **`run_gemm_gate*.sh` 経由では出力が破棄される**: 同スクリプトはバイナリを `2>err.tmp` で起動し
  成功時に `rm -f err.tmp` するため、ゲートスクリプト経由では stderr のダンプが失われる。ダンプを
  見るにはバイナリを直接起動すること:

  ```bash
  cd scripts/bench/framework-compare
  FRAMEWORK_COMPARE_PARITY_DUMP=1 \
    ./target/release/bench-candle --task gemm --device cuda --size 2048 --mode fresh \
    --out /tmp/x.jsonl 2>parity-dump.txt
  ```

- **未変更事項**: `PARITY_REL_TOL`/`PARITY_ABS_TOL`・`compare_elementwise` の判定結果・3 バイナリの
  JSONL 出力・`summarize.py`/`compare_gemm_gate.py` はすべて不変（判定・閾値の変更はユーザー承認
  必須。`.claude/rules/coding-rust.md`）。**GB10 実機での N=2048 fail 要素の実際の取得・§5.3 の
  仮説検証はイシュー #1184 で実施済み**。結果は
  `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3「追記（イシュー #1184）」を参照
  （fail 要素の値は参照実装・candle 側双方の通常の累積丸め誤差であり、片側優位の実装不具合では
  ないと確定した）

**厳密真値との突合**（イシュー #1184。`scripts/bench/framework-compare/parity_dump_truth.py`）:
`PARITY_DUMP` 行は「参照実装の値」と「フレームワーク実測値」しか教えないため、どちらが
数学的真値から離れているかは別途の突合が要る。本スクリプトは `Xorshift64Star`/`fill_vec` を
Python の有理数演算（`fractions.Fraction`）で厳密再現し、fail 要素の厳密真値・f32 FMA 逐次
累積の厳密丸め再現（`ref_bits` との bit 一致検証つき）・部分和最大値からの誤差フロア見積り
（`√K·ulp(max|partial|)`）を計算する。標準ライブラリのみに依存する。

```bash
cd scripts/bench/framework-compare
python3 parity_dump_truth.py --n 2048 < ../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cuda-2048.txt
```

**候補判定の机上評価**（イシュー #1237。`scripts/bench/framework-compare/parity_tolerance_candidates.py`）:
`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5 の N=2048 fail 要素（現行複合判定を
外れる 2 要素×2 device）について、候補判定（スケール付き絶対誤差・ULP ベース）を現行複合判定へ
OR 追加した場合の fail 数を、`parity_dump_truth.py` のダンプ実値のみを入力に機械的に算出する
（`parity_dump_truth.py` 自体は importlib で再利用するのみで変更しない）。標準ライブラリのみに
依存する。CI（`ci.yml` の `deps-forbidden` ジョブ）では単体テスト
（`parity_tolerance_candidates_test.py`）のみを実行し、実ダンプに対する計算は行わない。
**契約変更（`PARITY_REL_TOL`/`PARITY_ABS_TOL` 等）自体はユーザー承認事項**であり、本スクリプトは
承認判断に使う定量根拠の算出に閉じる（`docs/perf/candle-parity-tolerance-candidates.md`）。

```bash
cd scripts/bench/framework-compare
python3 parity_tolerance_candidates.py --n 2048 \
  --dump cuda=../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cuda-2048.txt \
  --dump cpu=../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cpu-2048.txt
```

**候補判定の fandhe-ai 本体側 parity 非後退契約への影響**（イシュー #1238。
`scripts/bench/framework-compare/parity_baseline_impact.py`）: 上記と同じ
候補定義（係数のみ再利用）を、fandhe-ai 本体側の parity 非後退契約
（`crates/backend-cuda/tests/common/parity_baseline.rs::BASELINES`。45 行）
へ適用した場合の影響を机上確認する。`BASELINES` は行単位の集計値
（`fail_count`・`mean_abs_diff_ceiling`・`max_abs_diff_ceiling`）しか持た
ないため、上記スクリプトと異なり要素単位の救済可否ではなく no-op／全救済／
部分・未確定の 3 クラス分類までしか判定できない。`M`（入力規模）は
`--scale-mode`（既定 `exact`）で行ごとに `Xorshift64Star` を実際に走らせて
`max|A|・max|B|` を求める（`upper-bound` は `M=1` の即時近似）。標準
ライブラリのみに依存し、`BASELINES`・tolerance 定数は一切変更しない
（`docs/perf/candle-parity-tolerance-baseline-impact.md`）。

```bash
cd scripts/bench/framework-compare
python3 -m unittest parity_baseline_impact_test.py
python3 parity_baseline_impact.py --scale-mode exact
```

**承認済み契約の実装（イシュー #1247。スケール付き絶対誤差救済項）**: 上記の候補判定は机上評価
（`parity_dump_truth.py`/`parity_tolerance_candidates.py`/`parity_baseline_impact.py`。いずれも
`bench-common::parity` の判定式自体は変更しない）に閉じていたが、`docs/candle-parity-tolerance-
contract-decision.md` §8（イシュー #1241 承認記録・2026-09-08）で「候補 A-1・係数 `c=0.5`・
ハーネス限定（本体 `compare`/`assert_parity`/`ParityBaseline` は不変）」が承認されたのを受け、
`bench-common::parity::compare_elementwise`/`dump_parity_failures`/`GemmReference::verify` の
要素単位判定へ第 3 救済項として実装した。

- **判定式**: `pass ⇔ rel < PARITY_REL_TOL ∨ diff < PARITY_ABS_TOL ∨ diff <= c・u・K・S_A・S_B`
  （`u = 2^-24`。既存 2 条件は本体契約と bit 単位で不変・第 3 項を OR 追加するのみ）
- **定数**（`bench-common::parity::{PARITY_SCALED_ABS_COEFF, F32_UNIT_ROUNDOFF}`）:
  `PARITY_SCALED_ABS_COEFF = 0.5`・`F32_UNIT_ROUNDOFF = 2^-24`。**本項目はハーネス限定の承認済み
  契約であり、本体 `crates/backend-cpu/src/parity.rs`/`crates/backend-cuda/tests/common/
  parity_baseline.rs` には対応する定数が設計上存在しない**（本体側への反映はスコープ外・
  イシュー #1254 は承認スコープ外として対応不要クローズ済み）
- **パラメータ**（`bench-common::parity::ScaledAbsTolerance`）: `K` は内積長（正方 GEMM の一辺長
  `n`）・`S_A`/`S_B` は入力行列 A/B の絶対値の全体最大（`ScaledAbsTolerance::from_inputs` が
  `GemmReference::compute` 内で 1 回だけ導出し、以降の `verify` 呼び出しで使い回す）。
  `ScaledAbsTolerance::NONE`（`bound() == 0.0`）を渡すと第 3 項が実質無効化され、既存 2 条件のみの
  レガシー判定と完全同値になる
- **JSONL の追加キー**: `parity_scaled_abs_bound`（適用した bound。非有限は既存 2 キーと同じ
  `null` 変換規則）・`parity_scaled_abs_rescued`（既存 2 条件では fail だが第 3 項で pass に転じた
  要素数）。既存 4 キー（`parity_total`/`parity_fail_count`/`parity_max_abs_err`/
  `parity_max_rel_err`）は書式・意味とも不変で、2 キーが追加されるのみ
- **fandhe-ai 側 0 fail は救済に依存しない（構造的遮断。イシュー #1247 PR #1443 codex-review
  指摘・P1）**: 第 3 項は §7 の (b-2)「比較対象（candle/Burn）妥当性検証に限り」の承認であり、
  `GemmReference::verify`（`self.tol` を無条件適用）は 3 バイナリ（`bench-fandhe`/`bench-candle`/
  `bench-burn`）で共有される汎用経路のため、メソッド未分離のままでは fandhe-ai 自身の検証にも
  第 3 項が効いてしまい、既存複合判定に違反する自社側回帰が `scaled_abs_bound` 以下に収まる限り
  `fail_count=0` として救済されうる。`bench-common::parity::GemmReference` はこれを避けるため
  `verify`（`self.tol` 適用。`bench-candle`/`bench-burn` が使う）と `verify_strict`
  （`ScaledAbsTolerance::NONE` 固定。既存 2 条件のみ）を分離し、**`bench-fandhe::run_gemm` 系の
  全呼び出しは `verify_strict` を使う**（CPU/CUDA/Metal・全形状に一律で効く構造的な遮断であり、
  特定 backend・形状のみを対象にしたテストに依存しない）。この構造に加えて `bench-fandhe` の CPU
  GEMM（N=64/256/512/2048）は本救済項なしに引き続き 0 fail であることもテストで固定している
  （`bench-fandhe::tests::gemm_cpu_parity_zero_fail_without_scaled_rescue`。
  `scaled_abs_rescued == 0` を assert）。第 3 項は candle/Burn 側参照 GEMM のキャンセレーション由来
  丸め誤差フロア（イシュー #1184。N=2048 で決定的に発生する 2 要素）を許容するための運用であり、
  fandhe-ai 側の回帰を隠す経路にはならない
- **実装済み（イシュー #1250）**: `summarize.py`/`compare_gemm_gate.py` の判定不能条件・理由出力への
  新キー反映（`parity_status`／`compare_gemm_gate.py::_parity_check` の詳細は「GEMM ゲート 5 回計測」
  節を参照）。N=2048 の GB10 再計測・判定不能解消の確認はイシュー #1260/#1262（**CUDA は
  #1260 で実測完了**: 確定判定〈未達・0.476 倍〉へ遷移。`docs/perf/cuda-gemm-candle-gate-
  remeasurement.md` §14。**CPU は #1262 で実測完了**: 確定判定〈未達・0.950 倍〉へ遷移。
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §19）

### `--tf32`（イシュー #1042。CUDA TF32 Tensor Core opt-in 比較）

`backend-cuda` の GEMM 公開経路（`fandhe-ai::gemm`）は既定で FP32 厳密（`run_tiled_f32`）だが、
opt-in で WMMA TF32 Tensor Core 経路（`run_wmma_tf32`）へ切り替えられる公開 API
（`fandhe_ai::set_cuda_tf32_gemm_enabled`）を追加した（`docs/cuda-tf32-optin-api-decision.md`）。
一方 burn 0.21 の CUDA バックエンドは常時 TF32（既定で reduced precision accumulation へ強制降格。
メモリ `burn-cuda-tf32.md`）であり、fandhe-ai の既定 FP32 計測と条件が揃わない。`--tf32` は
`--task gemm --device cuda` 限定でこの条件差を埋め、TF32 同士の同条件比較を可能にする値なし
フラグである。

- **`bench-candle`**: `--tf32` 指定時、candle-core 0.11 の公開プロセスグローバルスイッチ
  （`candle_core::cuda_backend::set_gemm_reduced_precision_f32`。既定 `false` = FP32 厳密）を
  有効化してから計測する。`--task gemm --device cuda` 以外との組合せは `MEASURE_ERROR` で
  fail-fast する。`cuda` cargo feature を有効化したビルド（`--no-default-features --features
  cuda`）が必要（既定は `metal`）
- **`bench-fandhe`**: **`--tf32` は常に `MEASURE_ERROR` で fail-fast する**。
  承認済みピンは `fandhe-ai =0.5.0`（イシュー #1011 で `=0.4.0` から更新済み）を経て
  `fandhe-ai =0.6.0`（v0.6.0 リリースサイクルでユーザー承認済み）・`fandhe-ai =0.7.0`
  （イシュー #1185 に対するユーザー指示で承認済み）へ進んだが、
  `set_cuda_tf32_gemm_enabled` は crates.io 公開版から呼び出し可能なまま、
  `bench-fandhe`（`main.rs`）側の呼び出し結線・`run_all` の tf32 スイープ追加（C-2。
  `docs/cuda-tf32-optin-api-decision.md`）は依然スコープ外で未実施のため、fail-fast
  の挙動は変わらない
- **`bench-burn`**: `--tf32` は受理せず常に `MEASURE_ERROR` で fail-fast する。burn の CUDA
  バックエンドは FP32 厳密経路自体を持たないため、フラグに opt-in／opt-out の意味を持たせられ
  ない（既存の burn GEMM 計測が実質的に常に TF32 相当であることの明記）
- **JSONL**: `--tf32` で計測した行は `"tf32":true` を emit する（既定は emit しないキー欠損 =
  `false` の互換規約。`bench_common::Record::tf32`）
- **`summarize.py`**: `--tf32` 行は目標達成ゲート（`--target`）・(a) GEMM 節の checksum 相互突合・
  FP32 参照値算出から**既定で除外**する（fail-open 防止。FP32 目標値との混同を防ぐ）。`--tf32` 行が
  存在するファイルには専用節「`(a-tf32) GEMM TF32`」を追加表示する

### `--managed`（イシュー #1353。CUDA managed memory 配置 A/B）

CUDA managed memory 配置（`cuMemAllocManaged` 経由の `DeviceBuffer` opt-in 配置。
`fandhe_ai::set_cuda_managed_memory_enabled`。イシュー #1352・`docs/backend-cuda-managed-
placement-decision.md`）を有効化して計測する値なしフラグ。GB10（ホスト・GPU 物理統合メモリ）で
H2D/D2H 転送を消せるかを実測するための A/B 用フラグであり、既定 OFF・fail-closed 方針は不変。

**経路上の重要事実**: managed 配置が効くのは `CudaMemory::alloc_zeroed`／`upload`／`download` を
通る経路（`gemm_resident_rhs`／`gemm_resident_lhs`〈NT 分岐除く〉／`linear_forward_device`／
`sgd_step_device` = `DeviceParamStore` 系）のみである。**`bench-fandhe --task gemm` は fresh／
reuse とも `a.matmul(&b)`（`CudaBackendOps::gemm` の `clone_htod`／`alloc_zeros`／`clone_dtoh`
直呼び経路）であり managed フラグの影響を構造的に受けない**（`--managed` を付けても gemm の
計測時間自体は変化しない設計上の前提。差なしを実測で裏取りする位置づけ）。主対象は
**`train --mode reuse`**（`DeviceParamStore` の `upload(grad)`／`alloc_zeroed`／`download`）。

- **`bench-fandhe`**: `--device cuda` 以外は常に `MEASURE_ERROR`（プロセスワイドフラグが cpu
  計測で無音 no-op になるのを防ぐ）。`--device cuda` でも、`managed-placement` cargo feature
  （既定無効）を有効化したビルドでなければ `MEASURE_ERROR` になる。crates.io 公開版
  `fandhe-ai =0.7.0` ピンには `set_cuda_managed_memory_enabled` API 自体が未収録（#1352 は
  未リリースの HEAD で追加）のため、有効化するには **`managed-placement` feature ＋
  `[patch.crates-io.fandhe-ai]` による未リリース HEAD `crates/facade` への path patch**の
  両方が必要:

  ```sh
  cargo build --release -p bench-fandhe --features managed-placement \
    --config 'patch.crates-io.fandhe-ai.path="/absolute/path/to/crates/facade"'
  ```

  `[patch]`／`.cargo/config.toml` は本 workspace の `Cargo.toml`・`Cargo.lock` へコミットしない
  （deps-policy.md 第 9 区分は registry 取得元のみを許容するため、patch は CLI 引数として都度
  与える。計測後は `git checkout -- scripts/bench/framework-compare/Cargo.lock` で復元する）
- **`bench-candle`／`bench-burn`**: `--managed` は fandhe-ai 固有の CUDA managed memory opt-in
  API を指す概念であり対応する公開 API がないため、常に `MEASURE_ERROR` で fail-fast する
- **JSONL**: `--managed` で計測した行は `"managed":true` を emit する（既定は emit しないキー
  欠損 = `false` の互換規約。`bench_common::Record::managed`。`tf32` と同型）
- **`summarize.py`／`compare_gemm_gate.py`／`compare_ab.py`**: `managed:true` 行は目標達成
  ゲート・A/B 比較から**既定で除外**する（既定 device-only 配置との速度混同防止）
- **A/B 計測**: `run_ab_managed_cuda.sh`（`AB_PATCH_FACADE_PATH` 環境変数必須。上記 path patch
  先の絶対パスを指定）が同一バイナリで off/on を交互起動し、`compare_managed_ab.py` が
  `(task, device, size, mode)` セルごとに 5 回計測中央値・checksum 一致（複合判定＋完全一致）を
  集計する。実測記録・既定化可否の判定は `docs/perf/cuda-managed-placement-ab.md` を参照

### `--device-checksum`（イシュー #1339。checksum のデバイス側 f64 reduction 化）

`docs/perf/cuda-gemm-reuse-phase-breakdown.md`・`metal-gemm-reuse-phase-breakdown.md` の
実測で、gemm 計測窓のうち `host_copy`（`C` の D2H）と `checksum`（全要素和をホスト側 `f64`
逐次和で求め直す処理）がハーネス計測窓の 66〜75% を占めることが確定した（縮退検出契約自体は
維持したいがハーネスの診断コストが支配的、という問題）。`--device-checksum` は checksum を
バックエンド側の `f64` reduction（`fandhe_ai::Var::matmul_checksum`／candle 側は
`sum_all().to_dtype(F64)`）で求め、毎反復の読み戻しを「checksum のみ（8 バイト）」へ縮小する
値なしフラグ。要素単位 parity（`GemmReference::verify`）は毎反復では検証せず、計測ループ後の
未計時 1 反復（`ChecksumReadout::WithOutput`）でのみ検証する。

- **対応範囲**: `--task gemm`（`--phases` なし。fresh／reuse とも）限定。`train`／`infer`・
  `--phases` との併用は常に `MEASURE_ERROR`
- **`bench-fandhe`**: `device-checksum` cargo feature（既定無効）を有効化したビルドでなければ
  `MEASURE_ERROR` になる。crates.io 公開版 `fandhe-ai =0.7.0` ピンには `Var::matmul_checksum`／
  `ChecksumReadout`／`GemmChecksum` API 自体が未収録（本イシューは未リリースの HEAD で追加）の
  ため、`--managed` と同じく **`device-checksum` feature ＋ `[patch.crates-io.fandhe-ai]`
  による未リリース HEAD `crates/facade` への path patch**の両方が必要:

  ```sh
  cargo build --release -p bench-fandhe --features device-checksum \
    --config 'patch.crates-io.fandhe-ai.path="/absolute/path/to/crates/facade"'
  ```

  `[patch]`／`.cargo/config.toml` は本 workspace の `Cargo.toml`・`Cargo.lock` へコミットしない
  （計測後は `git checkout -- scripts/bench/framework-compare/Cargo.lock` で復元する）。
  **実装状況（本イシュー時点）**: `BackendOps::gemm_checksum` は `backend-cpu` のみ実装済み
  （`gemm` と bit 同一の `C`・checksum はホスト f64 逐次和と bit 一致）。`backend-cuda`／
  `backend-metal` はデフォルト実装（常に `BackendError::Unsupported`）のままのため、
  `--device cuda`／`--device metal --device-checksum` は現状 `MEASURE_ERROR` になる
  （GPU 側デバイス reduction カーネルの実装・実機実測は後続イシューへ引き継ぐ）
- **`bench-candle`**: `--task gemm` 限定（`--mode reuse`／`--phases` の既存拒否は不変）。
  `--device cuda`／`--device cpu` は `to_dtype(F64)` 経由で checksum を求め、
  **`--device metal` は candle 0.11 の Metal が `F64` dtype の reduction を持たないため
  `f32` のまま `sum_all()` した値を checksum とする**（8 バイトではなく 4 バイト。この非対称は
  `docs/perf/device-checksum-readback-ab.md` に明記する）
- **`bench-burn`**: 対応する結線がないため常に `MEASURE_ERROR`
- **JSONL**: `--device-checksum` で計測した行は `"device_checksum":true` を emit する
  （既定は emit しないキー欠損 = `false` の互換規約。`bench_common::Record::device_checksum`。
  `tf32`／`managed` と同型）
- **`summarize.py`／`compare_gemm_gate.py`／`compare_ab.py`／`compare_gemm_ab.py`／
  `compare_managed_ab.py`**: `device_checksum:true` 行は目標達成ゲート・A/B 比較から
  **既定で除外**する（既存プロトコル計測との速度混同防止。正式ゲート〈#1031/#1037/#1117〉の
  既定判定は device_checksum 経路へ切り替えない）
- **実測記録**: 実機（DGX Spark GB10・M4 Max）での前後比較・on/off checksum 一致確認は
  `docs/perf/device-checksum-readback-ab.md` を参照（GPU バックエンド未実装のため CUDA／Metal
  は現時点で未実測。CPU の on/off checksum 一致は `bench-fandhe` の単体テスト
  `device_checksum_matches_legacy_checksum_fresh_and_reuse`〈`device-checksum` feature 限定〉
  で自動検証済み）

### `--graph <on|stream-only>`（イシュー #1350。学習 step の CUDA Graph capture A/B）

学習 step の update 区間（`BackendOps::sgd_step_device_tracked`。`DeviceParamStore::step`）を
CUDA Graph で capture・再利用する経路（`fandhe_ai::set_cuda_graph_step_enabled`。イシュー
#1349・`docs/backend-cuda-graph-step-capture-design.md`）の launch 固定費を実測するための
3 状態（`off`／`stream-only`／`on`）A/B 用フラグ。既定 OFF・fail-closed 方針は不変。

**経路上の重要事実**: capture 対象は update 区間の SGD カーネル 1 個のみ（forward／backward は
対象外）。**対応は `--device cuda --task train` 限定**（gemm／infer は `DeviceParamStore::step`
に到達しないため常に `MEASURE_ERROR`）。`train --mode fresh` はホスト側で `p - lr*g` を計算する
経路のため `DeviceParamStore::step` へ到達せず、stream 種別変更のプロセスワイド効果のみを見る
対照計測として使う（`train --mode reuse` が主対象）。

- **3 状態の意味**（`fandhe_ai_backend_cuda::graph` モジュール冒頭コメントに詳しい）:
  - `off`（既定・`--graph` 省略）: legacy stream・capture なし
  - `stream-only`: created stream で初期化するが capture はしない（「created stream の event
    管理コストのみ」を分離計測する診断状態）
  - `on`: created stream で初期化し update 区間を capture・再利用する
- **`stream-only` は API から選べない**: `set_cuda_graph_step_enabled` を一度でも呼ぶと以後
  環境変数が無視される契約のため、`bench-fandhe` は `--graph stream-only` 起動時に API を呼ばず
  代わりに起動側が事前に `FANDHE_AI_CUDA_GRAPH_STEP=stream-only` を export していることを
  `fandhe_ai::cuda_graph_step_mode()` で確認する（未 export なら `MEASURE_ERROR`）。`--graph`
  省略時も環境変数が漏れて off 以外のモードのまま計測されないことを同様に確認する
- **`bench-fandhe`**: `--device cuda --task train` 以外は常に `MEASURE_ERROR`。`--device cuda
  --task train` でも、`graph-step` cargo feature（既定無効）を有効化したビルドでなければ
  `MEASURE_ERROR` になる。crates.io 公開版 `fandhe-ai =0.7.0` ピンには
  `cuda_graph_step_mode`/`cuda_graph_step_stats` API 自体が未収録（#1349 は未リリースの HEAD で
  追加）のため、`--managed` と同じく **`graph-step` feature ＋ `[patch.crates-io.fandhe-ai]`
  による未リリース HEAD `crates/facade` への path patch**の両方が必要:

  ```sh
  cargo build --release -p bench-fandhe --features graph-step \
    --config 'patch.crates-io.fandhe-ai.path="/absolute/path/to/crates/facade"'
  ```

  `[patch]`／`.cargo/config.toml` は本 workspace の `Cargo.toml`・`Cargo.lock` へコミットしない
  （計測後は `git checkout -- scripts/bench/framework-compare/Cargo.lock` で復元する）
- **`bench-candle`／`bench-burn`**: `--graph` は fandhe-ai 固有の CUDA Graph capture opt-in API
  を指す概念であり対応する公開 API がないため、常に `MEASURE_ERROR` で fail-fast する
- **JSONL**: `--graph on`／`--graph stream-only` で計測した行は `"graph":"on"`／
  `"graph":"stream-only"` を emit する（既定は emit しないキー欠損 = off の互換規約。
  `bench_common::Record::graph`。`tf32`／`managed` と同型）。`--graph on` で `--task train`
  計測した行は launch 固定費の診断カウンタ（`fandhe_ai::cuda_graph_step_stats()` の再公開。
  `"graph_captured"`／`"graph_replayed"`／`"graph_launches"`／`"graph_sgd_kernel_launches"`）も
  併せて emit する
- **`summarize.py`／`compare_gemm_gate.py`／`compare_ab.py`／`compare_managed_ab.py`**: `graph`
  キーを持つ行は目標達成ゲート・A/B 比較から**既定で除外**する（既定 off 計測との速度混同防止）
- **A/B 計測**: `run_ab_graph_cuda.sh`（`AB_PATCH_FACADE_PATH` 環境変数必須。上記 path patch 先の
  絶対パスを指定）が同一バイナリで off/stream-only/on を交互起動し、`compare_graph_ab.py` が
  `(task, device, size, mode, phase)` セルごとに 5 回計測中央値・比・checksum 一致（複合判定＋
  完全一致）・launch カウンタの 5 run 内一致を集計する。実測記録・既定化可否の判定は
  `docs/perf/train-step-phase-breakdown.md` §16 を参照

## 使い方

```bash
cd scripts/bench/framework-compare
./run_all.sh                 # macOS: cpu + metal 全組み合わせ（+ metal gemm reuse・train reuse・train phases スイープ）→ results/raw/results.jsonl
./run_all_cuda.sh            # CUDA ホスト: cuda + cpu 全組み合わせ（+ cuda gemm reuse・train reuse・train phases スイープ）→ results/raw/results-cuda.jsonl
# 上記 2 本・run_ab_train_cuda.sh は crates.io ピン fandhe-ai =0.7.0 に借用
# ビュー readout API（Var::host_view/Tensor::host_slice）が未収録のため
# （#1438）、registry 解決のままではビルド不能（bench_fandhe_pin_guard.sh
# が明示エラーで早期停止する）。実行するには GEMM_GATE_PATCH_FACADE_PATH
# （crates/facade への絶対パス。通常は現行 HEAD の `crates/facade`）を
# 指定して bench-fandhe のみを path patch すること（run_gemm_gate.sh と
# 同じ仕組み。正式系列〈registry ピン〉の再計測はピン更新後にのみ可能）:
#   GEMM_GATE_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" ./run_all.sh
# 個別実行:
cargo run --release -p bench-fandhe -- --task gemm --device metal --size 2048
cargo run --release -p bench-fandhe -- --task gemm --device cuda --size 2048 --mode reuse
cargo run --release -p bench-fandhe -- --task train --device cuda --mode reuse
cargo run --release -p bench-fandhe -- --task train --device cpu --mode fresh --phases  # 1 step のフェーズ分解（イシュー #1009）
# 集計（JSONL → Markdown 表。既定は results/raw/*.jsonl 全件を標準出力へ。
# gemm reuse 行が存在するファイルには (a') 節、train reuse 行が存在する
# ファイルには (b') 節（イシュー #957/#958/#959）、train_phases 行が存在する
# ファイルには (b'') 節（イシュー #1009）が追加される。
# コミット済みの results/summary.md は既定動作では上書きされない）:
python3 summarize.py
python3 summarize.py results/raw/results.jsonl --out /tmp/tables.md   # 入力・出力の明示
```

失敗した組み合わせは `results/raw/skipped.log`（CUDA は `skipped-cuda.log`）に理由付きで記録される（数値の捏造はしない）。
`summarize.py` はこの節を集計対象として渡した各入力 JSONL と同一ディレクトリの `skipped*.log` からのみ収集する（入力省略時は従来どおり `results/raw/` 配下が対象。イシュー #971）。
集計は `results/summary.md` を参照。

`summarize.py` は GEMM の checksum（全フレームワーク・全 mode で同一入力のため本来一致するはず）を
size ごとに相互突合し、参照値と外れる行を表で「（無効: checksum 不一致）」表示する
（既定では stderr へ警告のみ、`--strict` を付けると不一致 1 件以上で終了コード 2）。
これとは独立に、要素単位検証（イシュー #970。前節参照）の閾値超過も同じ表で「（無効: 要素誤差超過
fail=<k>/<total>, max_abs=<e>, max_rel=<e>）」と表示し、`--strict` の対象にする（両方に該当する行は
理由を併記する）。各バイナリ側にも `bench-common::validate_gemm_checksum` による縮退 checksum
（全ゼロ・非有限）の emit 前ガードがある（`skipped.log` に理由付きで記録される）。**既知の無効データ**:
Burn(wgpu) Metal GEMM の N>=512 は upstream 既知バグ（`docs/perf/burn-wgpu-metal-gemm-zero-result.md`。
イシュー #965）により結果テンソル全ゼロを返すため無効（`results/summary.md`「データ有効性の注記」参照）。
コミット済みの raw JSONL（`results/raw/*.jsonl`）は本フィールド追加前に計測されたものであり、
要素単位検証は「未検証（旧形式）」表示になる（本 PR で数値を捏造・再計測はしていない。
次回再計測キャンペーンから要素単位検証が有効になる）。**「未検証（旧形式）」行も要素単位検証を
一度も受けていない点では検証済みと同列に扱えないため `--strict` の対象に含まれる**（既定の
非 `--strict` 実行では引き続き警告表示のみで終了コード 0）。このため、コミット済みの旧形式
JSONL（`results/raw/*.jsonl`）に対して `--strict` を付けて実行すると終了コード 2 になる
（`run_all*.sh`・CI は `summarize.py` を `--strict` なしでのみ呼ぶため、この経路は影響を受けない。
要素単位検証つきで再計測した JSONL のみが `--strict` を通過する）。

### 目標達成ゲート（`--target`。イシュー #1051）

親 #1049「横並び再計測と目標達成ゲート」の完了判定を人間の目視に頼らず機械的に行うためのオプション。
`--target candle`（または `burn`）を付けると、**同一入力 JSONL ファイル内**（1 ファイル = 1 環境。
ファイルをまたいだ突合は環境混同になるため行わない）の `(task, device, size)` ごとに、fandhe-ai と
指定フレームワークの GEMM / 学習 / 推論の中央値を突合し、fandhe-ai が同等以上の性能
（`fandhe_median_s <= target_median_s`）かを判定して「## 目標達成ゲート」節を追加出力する。

```bash
python3 summarize.py --target candle
echo $?   # 0: 全達成 / 2: --strict の無効データ判定が優先 / 3: 未達または判定不能が 1 件以上
```

- fandhe-ai・target とも reuse 行があれば reuse を優先し、無ければ fresh を使う（infer も
  gemm/train と同様 reuse 行〈`predict_resident` 経由。イシュー #1217〉があれば reuse を
  優先する）
- checksum 不一致・要素単位検証の閾値超過・train/infer reuse の checksum 不一致等（既存の無効判定と同じ規則）
  に該当する行は「達成」と判定せず「判定不能（無効データ）」に倒す（壊れた計算の実行時間で達成判定
  しない）。target 側が未計測の組合せも「判定不能（`<target>` 未計測）」として一覧に載せる（黙って
  落とさない）
- 未達・判定不能は表の直後に「未達一覧」「判定不能一覧」として列挙され、stderr にも同じ内容が出力
  される
- `--strict` と併用し、かつ `--strict` 側の無効データ判定（終了コード 2）にも該当する場合は、データ
  無効の解消を優先して終了コード 2 を返す（ゲート結果自体は Markdown 出力に残る）
- fandhe-ai 0.6.0（crates.io 公開版）での実機再計測結果（DGX Spark GB10・Apple M4 Max。
  `results/summary.md` 環境 10/11）に対する `--target candle` は**終了コード 3**（達成 3 件
  〈DGX Spark gemm/CPU/N=256・M4 Max gemm/CPU/N=256・N=512〉・未達 21 件・判定不能 2 件）。0.5.0
  時点（環境 8/9。達成 1 件・未達 23 件・判定不能 2 件）比では DGX Spark の CPU GEMM N=256・
  M4 Max の CPU GEMM N=512 が新規達成に転じた。未達・判定不能項目の内訳・既存トラッカーとの
  対応は `results/summary.md`「目標達成ゲート総括」節を参照
- **Burn は比較対象外**: CUDA 経路が TF32 降格（#1007 系の既知制約）のため、`--target burn` で機械的に
  「達成」と判定されても性能特性の異なる経路同士の比較である点に注意する（本ツールはこの区別を自動
  判定しない。人間が判断する）

## GEMM ゲート 5 回計測（CUDA: #1031 達成判定・イシュー #1142／Metal: #1037 達成判定・
イシュー #1147／CPU: #1117 達成判定・イシュー #1148）

`summarize.py --target candle` は同一入力ファイル内の 1 レコードしか拾わない
（1 ファイル = 1 環境の単発計測が前提）ため、CUDA #1031・Metal #1037「N=1024/
2048/4096 reuse で candle 超え」・CPU #1117「N=512/1024/2048 reuse で candle
超え」（各 5 回計測の中央値）の受け入れ判定には非対応。本節の
`run_gemm_gate.sh <device> <label>`／`compare_gemm_gate.py --device
{cuda,metal,cpu}` がその 5 回計測を専用に行う（`run_ab_train_cuda.sh` /
`compare_ab.py` の GEMM 版）。本体ロジックはイシュー #1142 の CUDA 専用実装
（`run_gemm_gate_cuda.sh`）を #1147 で device 汎用化・#1148 で CPU 対応拡張
したもので、呼び出し面は device 別の薄い wrapper `run_gemm_gate_cuda.sh`
（既存呼び出しとの CLI 互換維持）／`run_gemm_gate_metal.sh`／
`run_gemm_gate_cpu.sh`（いずれも内部で `bash run_gemm_gate.sh <device> "$@"`
を呼ぶのみ）に分離している。cpu は対象形状（N=512/1024/2048。cuda/metal の
N=1024/2048/4096 と異なる）に加え、各 run で `bench-fandhe gemm cpu <N>
fresh` も交互起動する（環境 10/11 単発 fresh 計測との連続性を説明するための
参考記録。**判定〈`achieved`〉には一切使わない** — 正式契約は他 device と
同じ reuse vs candle fresh のまま）。

**2 系列の使い分け**:

- **正式系列**（#1031/#1037 のゲート判定の正）: `bench-fandhe/Cargo.toml` の承認済み
  ピン（現行 `=0.7.0`）でビルドしたまま計測する。コミット済み manifest・
  `Cargo.lock` は変更しない
- **参考系列**（次回 crates.io 公開前の見込み値）: `GEMM_GATE_PATCH_FACADE_PATH=
  <facade 絶対パス>` を指定して `run_gemm_gate_cuda.sh`／`run_gemm_gate_metal.sh`
  を呼ぶと、本体 `crates/facade`（rsync 済み HEAD ツリー、または Mac の場合は
  ローカル直接実行の worktree HEAD）への path 差し替えビルド
  （`--config 'patch.crates-io.fandhe-ai.path="<facade 絶対パス>"'`）と計測を
  スクリプト内の 1 invocation で不可分に実行する（ビルドと計測の間に別の
  `cargo` コマンドが割り込む窓を作らない設計。イシュー #1166 の事故対応。
  詳細は下記「バイナリ同一性検証」節）。**`[patch]` セクション・
  `.cargo/config.toml` は一切コミットしない**（CLI 引数のみで与える。依存
  ポリシー〈`.claude/rules/deps-policy.md` 第 9 区分〉の「承認済みピンの完全
  固定」を壊さないため）。`bench-fandhe` の `VERSION` 定数は crates.io 版の
  まま変わらないため JSONL の `framework_version` では両系列を区別できず、
  **ファイル名ラベル**（例: `head-<short sha>`）で区別する。参考系列は
  #1031/#1037 の正式達成判定には使わない（次回ピン更新後の正式再計測で確定する）

```bash
cd scripts/bench/framework-compare
# CUDA 正式系列（現行ピン）:
bash run_gemm_gate_cuda.sh 0.7.0
# Metal 正式系列（現行ピン。イシュー #1147）:
bash run_gemm_gate_metal.sh 0.7.0
# CPU 正式系列（現行ピン。DGX Spark〈Grace CPU〉／M4 Max のいずれでも実行可。
# bench-candle のビルド flag はホスト OS で自動選択される。イシュー #1148）。
# GEMM_GATE_CPU_NODE_TAG（dgx-cpu／m4max-cpu の明示指定。必須）が実行ホストの
# OS 系列と矛盾する場合は fail-closed で終了する（`uname -s` だけで正式実機の
# ファイル名を無条件確定しない。codex-review P1 PRRT_kwDOTuUCJc6fK1Pe 対応。
# イシュー #1148）。さらに OS 系列一致だけでは「同一 OS 系列の任意ホスト」を
# 排除できないため、Git 管理外のローカルファイル
# `gemm-gate-trusted-hosts.local`（`gemm-gate-trusted-hosts.local.example` を
# コピーし、`hostname` コマンドの実出力と機体識別子〈Linux:
# `/etc/machine-id`／Darwin: `IOPlatformUUID`。hostname 単独は実行者自身が
# 別ホストで同名詐称できてしまうため codex-review P1 PRRT_kwDOTuUCJc6fLNFe
# 対応で追加〉を `<hostname>|<machine-id/IOPlatformUUID>` 形式で登録して作成）
# に登録した値との照合も必須（未作成・該当タグ未登録・hostname/機体識別子
# いずれかの不一致はいずれも計測前に fail-closed で終了する。codex-review
# P1 PRRT_kwDOTuUCJc6fK_lT・PRRT_kwDOTuUCJc6fLNFe 対応）:
cp gemm-gate-trusted-hosts.local.example gemm-gate-trusted-hosts.local  # 初回のみ。hostname・機体識別子を登録する
GEMM_GATE_CPU_NODE_TAG=dgx-cpu bash run_gemm_gate_cpu.sh 0.7.0     # DGX Spark 側
GEMM_GATE_CPU_NODE_TAG=m4max-cpu bash run_gemm_gate_cpu.sh 0.7.0  # M4 Max 側

# CUDA 参考系列（#1164 結線後 HEAD。ビルド＋計測を 1 invocation で実行）:
GEMM_GATE_PATCH_FACADE_PATH="$HOME/work/rust-ai-library-run/crates/facade" \
  bash run_gemm_gate_cuda.sh head-<short sha>
# Metal 参考系列（ローカル直接実行。worktree の crates/facade をそのまま指す。
# `cd ... && pwd` で `..` セグメントを含まない正規化済み絶対パスへ解決する
# ——`$(pwd)/../../../crates/facade` のように `..` を含む生文字列を渡すと、
# `cargo tree` の表示は正規化済み絶対パスになるため record_manifest の
# 厳密文字列比較が必ず不一致になり fail-closed エラーで測定が中断する）:
GEMM_GATE_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_gemm_gate_metal.sh head-<short sha>

# 借用ビュー readout は #1438 で既定経路化済み（feature 分岐は撤去済み）。
# ピン未更新の間は正式系列でも GEMM_GATE_PATCH_FACADE_PATH の併用が必須
# （上記「gemm --mode reuse --phases」節「借用ビュー readout」小節参照）:
GEMM_GATE_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_gemm_gate_cuda.sh head-<short sha>-readout-default

# 集計（N ごとに fandhe-ai reuse vs candle fresh の 5 回計測中央値・判定）:
python3 compare_gemm_gate.py results/raw/results-dgx-gemm-gate-0.7.0.jsonl
python3 compare_gemm_gate.py --device metal results/raw/results-m4max-gemm-gate-0.7.0.jsonl
python3 compare_gemm_gate.py --device cpu results/raw/results-dgx-cpu-gemm-gate-0.7.0.jsonl
python3 compare_gemm_gate.py --device cpu results/raw/results-m4max-cpu-gemm-gate-0.7.0.jsonl
echo $?   # 0: 全 N 達成 / 3: 未達または判定不能が 1 件以上 / 2: 入力を読めない
```

- `run_gemm_gate.sh <device> <label>`（device は `cuda`／`metal`／`cpu`。通常は
  device 別 wrapper 経由で呼ぶためラベルのみを渡す）はラベル
  （`[A-Za-z0-9._-]+` のみ許可）ごとに対象形状（cuda/metal: N=1024/2048/4096、
  cpu: N=512/1024/2048）それぞれで `bench-fandhe gemm <device> <N> reuse` と
  `bench-candle gemm <device> <N> fresh`（candle は reuse 非対応）を交互に
  5 回ずつ起動し（cpu のみ `bench-fandhe gemm cpu <N> fresh` も同数追加起動。
  計 3 起動 × 3 サイズ × 5 run = 45 run。cuda/metal は従来どおり 2 起動 ×
  3 サイズ × 5 run = 30 run）`results/raw/results-<node>-gemm-gate-<label>.jsonl`
  （`<node>` は cuda=`dgx`／metal=`m4max`／cpu=`dgx-cpu`〈Linux〉・
  `m4max-cpu`〈Darwin〉。`GEMM_GATE_CPU_NODE_TAG` の明示指定必須。`uname -s`
  は指定値との OS 系列整合確認のみに使用し、不一致は fail-closed で
  終了する）へ記録する。失敗は
  `results/raw/skipped-<node>-gemm-gate-<label>.log` に記録する（数値を捏造しない）。
  Metal の熱・電源状態は `pmset -g therm`・`uptime`（`sudo` 不要。CUDA の
  `nvidia-smi` に相当する device 別スナップショット）で実行ログへ記録する
  （`docs/perf/metal-bench-noise-protocol.md`「熱・電源状態の記録」節準拠）。
  cpu は host OS 別に `nproc`・`/proc/loadavg`・`lscpu`（Linux）または
  `sysctl` の機種名・P/E コア構成・`pmset -g therm`（Darwin）に加え、両者
  共通で `RAYON_NUM_THREADS`（未設定＝両フレームワーク既定で全コア使用）を
  記録する。計測ループが完走し `ANY_FAILED == 0`（全 run 成功）の場合にのみ、
  一時ファイルから上記 2 パスへ原子的（同一ファイルシステム内 `mv`）に反映する。
  1 件でも run が失敗した場合はこの 2 パスを一切変更せず、直前の有効な計測
  結果（同一 label の過去の成功実行分）を保全したまま、不完全な計測データは
  `results/raw/results-dgx-gemm-gate-<label>.failed-<UTC タイムスタンプ>.jsonl`
  等の診断用別名ファイルへ退避する（fail-closed。#1166 codex-review 指摘
  PRRT_kwDOTuUCJc6euxgr／PRRT_kwDOTuUCJc6evCpq 対応。security.md A08）
- **借用ビュー readout の既定経路化（イシュー #1337・#1438）**: 旧
  `GEMM_GATE_BENCH_FANDHE_FEATURES`（cargo feature 切替）は撤去済み。
  bench-fandhe は借用ビュー readout を常時使うため、ピン未更新の間は
  `GEMM_GATE_PATCH_FACADE_PATH`（HEAD ツリーへの path patch）が正式系列
  でも必須になる（未指定は `bench_fandhe_pin_guard.sh` がビルド起動前に
  早期エラー。crates.io 公開版 `fandhe-ai =0.7.0` には該当 API が
  未収録なため）。manifest の `bench_fandhe_features` フィールドは JSON
  形状互換のため空文字固定で残し、`GEMM_GATE_SKIP_BUILD=1` 経路で非空値
  （旧 on 腕の manifest）を検出した場合は fail-closed で再ビルドを要求する。
- **バイナリ同一性検証（イシュー #1166。依存元照合は同イシューへの
  codex-review／Cursor Bugbot 指摘で強化。bench-candle 側の検証は同イシュー
  への追加 codex-review 指摘 PRRT_kwDOTuUCJc6evCpm 対応）**: `bench-fandhe`・
  `bench-candle` 双方をビルドした直後（他の `cargo` コマンドを挟まず）に、
  各 `target/release/<binary>` の sha256 と依存解決元（`cargo tree -p
  <package> --depth 1` の path/registry 判定。`fandhe-ai` は**ビルド時と同一の
  `--config`〈`GEMM_GATE_PATCH_FACADE_PATH` 指定時の path patch〉を付けて
  実行**し、path patch 適用ビルドでも `cargo tree` 側だけ patch なしで解決
  され registry と誤記録する事故を防ぐ。`candle-core` は patch 対象外のため
  常に registry 解決を要求する）を `results/raw/manifest-dgx-gemm-gate-
  <label>.json` へ記録し、計測ループ開始直前（`GEMM_GATE_SKIP_BUILD=1` を
  含む全経路）に再計算した sha256・依存解決元と突き合わせる。bench-candle
  側の検証がなかった旧実装では、`GEMM_GATE_SKIP_BUILD=1` 経路で candle
  binary が別バージョンへ差し替えられていても検出できず、その性能値を
  candle 0.11.0 の値として確定してしまう可能性があった。過去に、
  確認目的の素の `cargo tree` を挟んだだけで Cargo.lock が registry 解決へ
  暗黙に再ロックされ、意図しない登録版 binary へ差し替わって計測してしまった
  事故が実際に発生した（`docs/perf/logs/cuda-gemm-candle-gate-1142/env_info.txt`
  「参考系列ビルドの事故と対処」節）。この検証は fail-closed（manifest 欠落・
  sha256 不一致・依存解決元の取得失敗〈"unknown" への fail-open はしない〉・
  依存解決元の不一致ならいずれも測定を一切実行せず exit 1。security.md A08）
  - さらに、記録・検証いずれの時点でも `fandhe_ai_source` を「`GEMM_GATE_
    PATCH_FACADE_PATH` を指定した invocation なら `path:<指定パス>`、
    指定しない invocation なら `registry`」という契約に照合する。単に
    sha256 が一致しているだけでは、記録後に `GEMM_GATE_SKIP_BUILD=1` を
    使って `GEMM_GATE_PATCH_FACADE_PATH` の有無を変えて実行した場合の系列
    取り違え（正式系列のラベルで参考系列の依存解決を計測してしまう等）を
    検出できないため、README「ファイル名ラベルが唯一の系列識別手段」という
    計測契約をこの照合で担保する
  - `GEMM_GATE_SKIP_BUILD=1` は「同一 label で直前に成功した本スクリプト実行
    が残した manifest と binary が一致する場合に限り」ビルドを省略する用途
    （失敗 run の再実行等）。参考系列の外部事前ビルド＋`GEMM_GATE_SKIP_BUILD=1`
    という旧 2 段構成は、ビルドと計測の間に任意の `cargo` コマンドが割り込む
    窓を生むため廃止し、上記 `GEMM_GATE_PATCH_FACADE_PATH` に統合した
- `compare_gemm_gate.py JSONL...` は size ごとに fandhe-ai/candle 各 5 件の
  `median_s` から中央値を算出し `fandhe_median_s <= candle_median_s` を判定する。
  以下はいずれも「判定不能」として明示し性能値を確定表示しない（fail-closed。
  security.md A08）: レコードが 5 件未満、要素単位検証（`parity_*`。イシュー
  #970）が `parity_fail_count > 0` またはフィールド欠損・値域不正、checksum が
  本体の数値一致契約（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる。
  判定不能時は run ごとの `fail_count`/`max_abs`/`max_rel`/`bound`/`rescued`
  を診断表として出力する（N=2048 の candle 無効データの原因調査・再現条件
  記録に使う。イシュー #1142 R2）。`--device cpu` のみ、`bench-fandhe gemm
  cpu <N> fresh` 行がちょうど 5 件かつ要素単位検証・checksum とも正式判定
  と同じ検証を通る場合に限り「fandhe-ai fresh median（参考）」列を追加表示
  する。この列は環境 10/11 の単発 fresh 計測との連続性を説明するための参考
  記録であり `achieved` の判定には一切使わない（イシュー #1148）
  - **要素単位判定の契約（イシュー #1241 でユーザー承認・#1247 で
    `bench-common::parity` 実装済み・#1250 で本ツール〈`compare_gemm_gate.py`
    ／`summarize.py`〉が追従。`docs/candle-parity-tolerance-contract-
    decision.md` §8）**: 既存複合判定（相対誤差 1e-3 未満 または絶対誤差
    1e-5 未満）に加え、スケール付き絶対誤差の第 3 救済項（候補 A-1・係数
    `c=0.5`）が OR で追加された `parity_scaled_abs_bound`／
    `parity_scaled_abs_rescued` の 2 診断フィールド（両方存在する場合のみ
    追加検証。片方だけの JSONL は部分欠損として判定不能・両方欠損の 4 キー
    行はレガシー契約〈第 3 項なし〉で判定する）を検証する。判定式そのもの
    は再計算しない（`bench-common::parity` が単一真実源）。判定不能となる
    条件: `bound` が `null`（入力に非有限値を検出したセンチネル）または
    不正値、`rescued` が不正値・値域外、`rescued > 0` なのに `bound` が
    絶対誤差許容値未満（整合しない）、`framework == "fandhe-ai"` の行が
    `rescued > 0`（fandhe-ai は全経路 `verify_strict` のため構造的に 0 の
    はず）、`fail_count > 0`（救済後もなお fail する要素がある。framework
    を問わず判定不能）。`framework == "candle"` の `rescued > 0` は許容し
    達成／未達判定へ進める（判定不能を「達成」へ倒す経路は用意しない）。
    候補行の `max_abs_err`／`max_rel_err` は pass 要素も含む全要素の最大
    であり `bound` を超えうるため判定条件には使わない（診断表示のみ）。
    candle 側が救済に依存した size は判定列へ「（candle 救済 n 要素）」と
    注記される
- `tf32:true` の行（イシュー #1042）は本ゲートの対象外として除外する

### Metal GEMM 結線前後 A/B（`run_ab_gemm_metal.sh`／`compare_gemm_ab.py`。イシュー #1306）

`run_gemm_gate_metal.sh` が対 candle の性能ゲート判定なのに対し、本ツールは
**fandhe-ai 自身の 2 ビルド**（before=正式系列 `fandhe-ai =0.7.0`〈crates.io
registry 解決〉・after=参考系列 HEAD〈`crates/facade` への path patch〉）を
比較する。依存 #1304 が `tile::CANDIDATES`／`tile::select`／
`select_for_device` を一切変更していない（本番既定は不変。
`docs/perf/metal-gemm-n4096-kernel-gap.md` §19.1）ため、本ツールは字義通りの
「結線前後」の差分計測ではなく、**v0.7.0 → HEAD の Metal 側変更群（E2〜E8 の
function constant・候補追加等）が本番既定経路の性能を後退させていないかを
確認する 0.7.0 ↔ HEAD 非後退確認**である。`compare_ab.py` は
`framework_version` が before/after で同一だと fail-closed 拒否するため
（同一バージョンの A/B は意味を持たないという前提）、before/after とも
`fandhe-ai =0.7.0` を名乗る本用途には流用できない。`compare_managed_ab.py`
は同一バイナリのフラグ切替専用で 2 本の異なるバイナリを比較する構造を
持たないため、こちらも流用できない。

**注（#1438 以降。before 腕の構造的制約）**: before 腕は常に registry 解決
（crates.io ピン `fandhe-ai =0.7.0`）を意図する。しかし bench-fandhe の
ソース（本スクリプトが常に現行チェックアウトから同一ソースでビルドする）
が借用ビュー readout API を無条件に要求するようになった（#1438）ため、
ピンにこの API が未収録の現状では before 腕は facade への path patch では
解消できない構造的な理由で常にビルド不能（`bench_fandhe_pin_guard.sh` が
専用の note 付きで明示エラーを出す）。対処は (1) crates.io ピンが借用
ビュー readout API を収録するまで待つ、または (2) #1438 の feature 撤去
より前のコミットを別 worktree にチェックアウトし、その worktree の
`scripts/bench/framework-compare/` から本スクリプトを実行する（この場合
after 腕の `AB_PATCH_FACADE_PATH` には現行 HEAD の `crates/facade` を指定
できる）のいずれかに限られる。

- `run_ab_gemm_metal.sh <label>`（`AB_PATCH_FACADE_PATH=<HEAD の crates/facade
  絶対パス>` 必須。`AB_ROUNDS`〈既定 5〉で計測回数を調整可能だが判定は 5
  固定）は before（registry）・after（path patch）の 2 バイナリを
  `target/release/bench-fandhe-ab-before`／`-ab-after` としてビルドし、
  ビルド直後に `cargo tree` で依存解決元（before=registry・after=
  `path:<AB_PATCH_FACADE_PATH>`）を検証したうえで sha256・依存解決元を
  `results/raw/manifest-m4max-gemm-ab-<label>.json` へ記録する。N=512/1024/
  2048/4096 × fresh/reuse を 5 run（run 単位で before/after を交互起動。
  偶数 run では順序を反転し起動順序の系統誤差を均す）計測し、
  `results/raw/results-m4max-gemm-ab-before-0.7.0-<label>.jsonl`／
  `results-m4max-gemm-ab-after-<label>.jsonl` へ記録する（before 側の
  保存先も `<label>` でスコープする。別 label で再実行した際に過去の
  before データを上書きせず、当該 label の交互計測ペアを追跡できる
  ようにするため）。`[patch]` は CLI
  引数のみで与え `Cargo.lock`／`.cargo/config.toml` はコミットしない
  （`Cargo.lock` は trap で復元。deps-policy.md 第 9 区分）。全 run 成功時
  にのみ一時ファイルを正規パスへ原子的に反映し、1 件でも失敗すれば正規
  パスを変更せず `.failed-<UTC>` へ退避する（fail-closed。security.md A08）
- `compare_gemm_ab.py BEFORE.jsonl AFTER.jsonl [--threshold 1.05]` は
  `(size, mode)` セルごとに before/after 各ちょうど 5 件を要求し、
  `median_s` 中央値比（`ratio = after/before`。既定閾値 1.05 以下なら
  非後退）・checksum 複合判定（`checksum_contract.checksums_match`）＋
  完全一致列を Markdown 表で出力する。本番カーネル・選択結果は不変のため
  bit 完全一致が期待値であり、複合判定 pass のみ（完全一致でない）は
  「複合判定 ok」として区別する。`framework != "fandhe-ai"`・`tf32:true`・
  `managed:true`・`task != "gemm"`・`device != "metal"`・`parity_fail_count
  > 0` の行は判定不能として除外する。終了コード: 0=全セル非後退、
  3=後退または判定不能あり、2=入力不能
- 結線判断: 全 8 セル非後退なら現行 `select_for_device` 既定（#1304 完了
  時点の候補表）を本番既定として確定しコード変更なし。1 セル以上後退した
  場合も結線対象（unwire 対象）は存在しないためコード変更なしで、後退を
  v0.7.0 → HEAD の Metal 変更群のいずれかによる（切り分け未実施）と
  帰属して記録する。判定不能（負荷ノイズ・checksum 不一致）の場合も本番
  既定は切り替えない（安全側）。判断の記録先は
  `docs/perf/metal-gemm-n4096-kernel-gap.md` §19

### `compare_gemm_ab.py --device cpu`（既定スレッド数限定 on/off 比較・イシュー #1364）

`compare_gemm_ab.py` は既定で `--device metal`（上記 #1306 用途・8 セル）を
使うが、`--device cpu` を指定すると N=512/1024/2048 の 6 セル
（`compare_gemm_gate.py --device cpu` と同じ N 集合。CPU GEMM ゲート計測は
N=4096 を対象にしないため）に切り替わる。判定ロジック（threshold・
checksum 複合判定・parity fail-closed）は device に関わらず不変。

`crates/backend-cpu/src/thread_limit.rs`（大コア数限定。イシュー #1363）の
既定有効化 on/off を（`GEMM_GATE_PATCH_FACADE_PATH=<crates/facade 絶対
パス>` の path patch と `RAYON_NUM_THREADS` の有無を組み合わせて）比較
する用途にも本ツールを流用する。#1364 の実測当時は
`BIG_CORE_LIMIT_ENABLED = true`（限定有効）のまま HEAD 直下の同一バイナリ
で `RAYON_NUM_THREADS` の有無のみを切り替えて計測できたが、REJECT
（不採用）確定を受けて `BIG_CORE_LIMIT_ENABLED` は `false` へ差し戻し
済み（下記参照）のため、再計測には手順の変更が必要になる。

**現行 HEAD の `crates/facade`（本 checkout）は限定が無効
（`BIG_CORE_LIMIT_ENABLED = false`。イシュー #1405 実装後、DGX での重大な
後退により無効化された経緯。`crates/backend-cpu/src/thread_limit.rs`）
のため、この checkout の `crates/facade` を `GEMM_GATE_PATCH_FACADE_PATH`
に指定しても on/off で挙動が変わらない**。限定が有効な状態を再現するに
は、限定実装時点のコミット `90ea1cb`（`feat(backend-cpu): 大コア数判定
…既定スレッド数の限定を実装し…`。#1405。`BIG_CORE_LIMIT_ENABLED = true`）
を別ディレクトリへ隔離 checkout し、その `crates/facade` を指す:

```bash
# 限定が有効な隔離 checkout を用意する（初回のみ）
git worktree add /tmp/fandhe-thread-limit-on 90ea1cb
FACADE_ON="/tmp/fandhe-thread-limit-on/crates/facade"
FACADE_OFF="$(cd ../../../crates/facade && pwd)"  # 現行 HEAD（限定無効）

# 限定あり（90ea1cb の facade。RAYON_NUM_THREADS 未設定を明示）
env -u RAYON_NUM_THREADS GEMM_GATE_CPU_NODE_TAG=<node> GEMM_GATE_PATCH_FACADE_PATH="$FACADE_ON" \
  bash run_gemm_gate_cpu.sh head-limit-on
# 限定なし（現行 HEAD の facade。全論理コア数を明示し実質同一挙動で対照する）
RAYON_NUM_THREADS=<全論理コア数> GEMM_GATE_CPU_NODE_TAG=<node> GEMM_GATE_PATCH_FACADE_PATH="$FACADE_OFF" \
  bash run_gemm_gate_cpu.sh head-limit-off

# run_gemm_gate_cpu.sh の出力には fandhe-ai（reuse・fresh 参考行）と
# candle（fresh）の行が混在する。compare_gemm_ab.py は 'framework' が
# 'fandhe-ai' 以外の行を検出すると警告を出し判定不能（終了コード 2）を
# 返すため、比較にかける前に fandhe-ai の行のみを別ファイルへ抽出する
# （mode=reuse のみに絞る必要はない — mode は _cell_key に含まれ、fresh
# 参考行は期待セル集合外として無視されるだけで害はない）:
for label in head-limit-off head-limit-on; do
  jq -c 'select(.framework == "fandhe-ai")' \
    "results/raw/results-<node>-gemm-gate-${label}.jsonl" \
    > "results/raw/results-<node>-gemm-gate-${label}.fandhe-only.jsonl"
done

python3 compare_gemm_ab.py --device cpu \
  results/raw/results-<node>-gemm-gate-head-limit-off.fandhe-only.jsonl \
  results/raw/results-<node>-gemm-gate-head-limit-on.fandhe-only.jsonl
```

before=限定なし・after=限定あり（`ratio = after/before`）として渡す。
上記のとおり on/off は異なる facade checkout（HEAD／`90ea1cb`）を指すため
`manifest-*.json` の `bench_fandhe_sha256` は on/off 間で一致しない
（`BIG_CORE_LIMIT_ENABLED` の値以外はソース同一のはずだが、バイナリの
バイト同一性は保証しない）。各 invocation 内での再現性は
`manifest-*.json` の `fandhe_ai_source` が指定した `GEMM_GATE_PATCH_
FACADE_PATH` と一致していることで確認する（結果記録・採否は
`docs/perf/cpu-gemm-default-thread-limit.md` §6・
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §13 を参照）。

**イシュー #1313（(mc, nc) 2D 動的分配 `TWO_D_DYNAMIC_PRODUCTION_ENABLED` の
本番結線 on/off）** は上記 #1364 と異なり on／off いずれも現行 HEAD の
`crates/facade` 1 checkout だけで再現できない（結線 on/off は const の
値そのものであり、この checkout は既に on 固定のため）。結線前（off）の
比較には別 checkout（結線コミットの親コミット、または origin/main）を
用意する必要がある:

```bash
cd scripts/bench/framework-compare
FACADE_AFTER="$(cd ../../../crates/facade && pwd)"  # 現行 HEAD（結線後・on 固定）
git worktree add --detach /tmp/fandhe-1313-before <結線前コミット>
FACADE_BEFORE=/tmp/fandhe-1313-before/crates/facade

GEMM_GATE_CPU_NODE_TAG=<node> GEMM_GATE_PATCH_FACADE_PATH="$FACADE_BEFORE" \
  bash run_gemm_gate_cpu.sh head-1313-off
GEMM_GATE_CPU_NODE_TAG=<node> GEMM_GATE_PATCH_FACADE_PATH="$FACADE_AFTER" \
  bash run_gemm_gate_cpu.sh head-1313-on

for label in head-1313-off head-1313-on; do
  jq -c 'select(.framework == "fandhe-ai")' \
    "results/raw/results-<node>-cpu-gemm-gate-${label}.jsonl" \
    > "results/raw/results-<node>-cpu-gemm-gate-${label}.fandhe-only.jsonl"
done

python3 compare_gemm_ab.py --device cpu \
  results/raw/results-<node>-cpu-gemm-gate-head-1313-off.fandhe-only.jsonl \
  results/raw/results-<node>-cpu-gemm-gate-head-1313-on.fandhe-only.jsonl
```

DGX 側は `~/work/rust-ai-library-run`（共有作業ディレクトリ）を使わず、
本イシュー専用の隔離ディレクトリ（`rsync` で before/after 2 本を別々に
転送）を用い、計測後に削除する（#1148 と同方針）。実測結果・判定は
`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`「#1313 追記」節・
`results/summary.md` 環境 24 を参照。

### `compare_gemm_ab.py --device cuda --sizes gate --modes reuse`（借用ビュー readout・イシュー #1337・#1438）

`compare_gemm_ab.py` は `--device cuda`（新規。N=1024/2048/4096 の 6 セル。
`metal`／`cpu` と異なり cuda には 512 込みの独自 8 セル用途の前例が無いため
既定セル集合＝gate セル集合）・`--sizes gate`（`--device metal` の既定 8 セル
から 512 を除いた 6 セルへ絞り込み。`run_gemm_gate.sh` の出力形状に合わせる
用途）・`--modes`（既定 `fresh,reuse`。カンマ区切り。cuda/metal のゲート
出力は reuse のみのため `--modes reuse` で fresh 参考行をセル外扱いにできる）
に対応する。

`readout_var` 借用ビュー切替（上記「gemm --mode reuse --phases」節「借用
ビュー readout」小節）の off（旧 cargo feature 無効相当の legacy 経路）/
on（借用ビュー経路）A/B に本ツールを流用した（イシュー #1337 当時の
記録）。当時の off/on 切替は `GEMM_GATE_BENCH_FANDHE_FEATURES` cargo
feature で行っていたが、`#1438` で feature 自体を撤去し借用ビュー経路を
既定化した。

**注意（イシュー #1438 codex-review 指摘・PR #1452 P2）**: `readout_var` の
off/on 分岐は **`bench-fandhe`（本ハーネス自身のソース。`main.rs`）側**に
あり、`crates/facade`（本体ライブラリの公開 API 面）側にはない。したがって
`GEMM_GATE_PATCH_FACADE_PATH`（facade クレートの依存解決元を切り替える
機構）だけを腕ごとに変えても `bench-fandhe` 自体のソースは常に現行
チェックアウト（借用ビュー経路）のままビルドされ、legacy/default の A/B
にはならない（両腕とも default 経路を計測してしまう）。legacy/default を
再現するには **`bench-fandhe` を含むリポジトリ全体を feature 撤去前後の
2 つの worktree としてチェックアウトし**、各 worktree の
`scripts/bench/framework-compare/` からそれぞれ本スクリプトを実行する
（`GEMM_GATE_PATCH_FACADE_PATH` は**両腕とも同一の固定 `crates/facade`
ツリーを指す**ことで、facade 側の差分が比較に混入しないよう揃える。
legacy/default 2 つの worktree はあくまで `bench-fandhe`〈本ハーネス自身の
ソース〉を切り替えるためのものであり、facade 側は default worktree
〈現行 HEAD〉の 1 本に固定して両腕で使い回す。イシュー #1438 codex-review
指摘・PR #1452 P2: 旧手順は worktree ごとに `crates/facade` を別々に
指しており、「facade 側の変更が計測へ影響しない」という比較保証が
成立していなかった）:

```bash
# 2 つの worktree を用意する（同一リポジトリの異なるコミットを同時
# チェックアウトするため git worktree を使う。通常の checkout の
# 使い回しでは両腕を同時にビルド・保持できない）
git worktree add /tmp/fandhe-ai-readout-legacy <feature 撤去前コミット sha>
git worktree add /tmp/fandhe-ai-readout-default <feature 撤去後コミット sha（例: 現行 HEAD）>

# facade は default worktree（現行 HEAD）の 1 本に固定し、両腕で
# 同一パスを使い回す（facade 側の変更が両腕の計測に影響しないよう揃える）
FACADE_FIXED="$(cd /tmp/fandhe-ai-readout-default/crates/facade && pwd)"

# legacy 腕（旧 to_tensor+to_vec 経路。bench-fandhe ソースのみ旧 worktree）
cd /tmp/fandhe-ai-readout-legacy/scripts/bench/framework-compare
GEMM_GATE_PATCH_FACADE_PATH="$FACADE_FIXED" bash run_gemm_gate_cuda.sh head-<short sha>-readout-legacy

# default 腕（借用ビュー経路。#1438 で既定化）
cd /tmp/fandhe-ai-readout-default/scripts/bench/framework-compare
GEMM_GATE_PATCH_FACADE_PATH="$FACADE_FIXED" bash run_gemm_gate_cuda.sh head-<short sha>-readout-default

# 出力（results/raw/ 配下）を比較用ディレクトリへ集約してから
# fandhe-ai 行のみ抽出（cpu の thread-limit A/B と同じ手順。上記参照）
for label in head-<short sha>-readout-legacy head-<short sha>-readout-default; do
  jq -c 'select(.framework == "fandhe-ai")'     "results/raw/results-dgx-gemm-gate-${label}.jsonl"     > "results/raw/results-dgx-gemm-gate-${label}.fandhe-only.jsonl"
done

python3 compare_gemm_ab.py --device cuda --sizes gate --modes reuse   results/raw/results-dgx-gemm-gate-head-<short sha>-readout-legacy.fandhe-only.jsonl   results/raw/results-dgx-gemm-gate-head-<short sha>-readout-default.fandhe-only.jsonl
```

`metal`・`cpu` でも device を差し替えて同様に使う（`metal`／`cpu` の
`--sizes gate` は既存の gate セル集合と一致する。上記「`--device cuda`
セル集合」参照）。判定基準（`parity_fail_count=0`・checksum 完全一致・
`ratio = after/before <= 1.05`）は他 A/B ツールと同じ。結果記録は
`docs/perf/{cuda,metal,cpu}-gemm-candle-gate-remeasurement.md` を参照。

## A/B 計測（都度同期廃止・イシュー #1083）

#1011（CUDA 都度 `stream.synchronize()` 廃止）の受入条件「MLP 学習 1 step が
実測で短縮する」を、実践規模（本ハーネスの `train cuda 64`）で確認するための
before/after 比較手順。RTX 3060 トイモデルでの計測（`docs/perf/cuda-async-sync-removal-rtx3060.md`）
では非同期化の効果が 1 step 全体の短縮として顕在化しなかったため、比較対象を
本ハーネスへ広げる。

**前提**: 本比較は `fandhe-ai` ピン（`bench-fandhe/Cargo.toml`）の crates.io
バージョンで before/after を作る。ピンの更新は依存ポリシー（`.claude/rules/deps-policy.md`
第 9 区分）上ユーザー承認必須で、イシュー #1011 のユーザー承認を得て
`fandhe-ai =0.5.0`（2026-08-31 crates.io 公開・`release-all.yml` run 33388884217・
tag `v0.5.0` = `a5e465d`）へ更新した（#1011 ツリー）。**ピンはその後 v0.6.0
リリースサイクルで `=0.6.0`、v0.7.0 リリースサイクル（イシュー #1185 に対する
ユーザー指示）で `=0.7.0` へさらに更新済みであり、現在のツリー（main）の
ピンは「都度同期なし」側の延長線上（after 系列。現行 `=0.7.0`）を指すが
`after-0.5.0` の値そのものではない**。「都度同期あり」側（before = 0.4.0）・
「都度同期なし」側（after = 0.5.0）を当時のまま再現するには、それぞれ
対応するピンのコミット（`=0.4.0`・`=0.5.0`）を別 worktree で checkout して
計測する。

**注（#1438 以降）**: crates.io ピン `fandhe-ai =0.7.0` には借用ビュー readout
API が未収録のため、`GEMM_GATE_PATCH_FACADE_PATH`（`crates/facade` への絶対
パス）を指定しない限り本スクリプトは `bench_fandhe_pin_guard.sh` により
明示エラーで早期停止する（上記「使い方」節と同じ理由）。

```bash
cd scripts/bench/framework-compare
# before（現行ピン。都度同期あり）を DGX Spark 実機で計測:
GEMM_GATE_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_ab_train_cuda.sh before-0.4.0
# ピン更新（別 PR・承認後）を適用したツリーで after を計測:
GEMM_GATE_PATCH_FACADE_PATH="$(cd ../../../crates/facade && pwd)" \
  bash run_ab_train_cuda.sh after-0.5.0

# before/after の 5 回計測中央値を比較（fresh/reuse 各 mode ごとに Markdown 表）:
python3 compare_ab.py results/raw/results-dgx-ab-before-0.4.0.jsonl \
  results/raw/results-dgx-ab-after-0.5.0.jsonl
echo $?   # 0: 判定完了（性能比較が成立） / 2: 判定不能（レコード不足・version 同一・checksum 不一致等）
```

- `run_ab_train_cuda.sh <label>` はラベル（`[A-Za-z0-9._-]+` のみ許可）ごとに
  `bench-fandhe train cuda 64` を fresh/reuse それぞれ **5 回**起動し
  `results/raw/results-dgx-ab-<label>.jsonl` へ記録する（`run_all_cuda.sh` は
  fresh/reuse 各 1 回のみのため、5 回計測中央値〈coding-rust.md〉を得るには
  本スクリプトを使う）。失敗は `results/raw/skipped-dgx-ab-<label>.log` に
  記録される（`run_all_cuda.sh` と同じく数値を捏造しない）。診断用に
  `--phases`（イシュー #1009）も fresh/reuse 各 1 回追加で記録する。
- `compare_ab.py BEFORE.jsonl AFTER.jsonl` は `(mode)` ごとに 5 レコードの
  `median_s` から中央値を算出し before/after を比較する。以下はいずれも
  「判定不能」として明示され、性能値を確定表示しない（fail-closed。
  security.md A08）: レコードが 5 件未満、before/after の `framework_version`
  が同一（A/B になっていない）、最終 loss（`checksum`）が本体の数値一致契約
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる。`--phases` 行が
  両ファイルにあれば phase 別の参考表も出す（同期点の分析。単発計測のため
  5 回中央値の対象外）。
- 計測境界の注意: fandhe-ai 0.7.0 の `Tensor<f32>` はホスト常駐で、reuse
  モードでも各 step の `loss.to_tensor()` 実体化が単一 in-order ストリーム
  上の同期点として残る（`docs/backend-cuda-async-execution-design.md`）。
  定常状態では計測窓のずれ（1 step）を無視でき 1 step 総和と等価とみなす。

## 依存ポリシー上の位置づけ

- 本 workspace は許容依存第 9 区分（ベンチ比較対象）の適用範囲拡張として、`candle-core =0.11.0`・`burn =0.21.0` を**本ディレクトリ限定**で保持する（`.claude/rules/deps-policy.md`）
- 本体 workspace（ルート `Cargo.toml` / `Cargo.lock`）への混入は引き続き禁止であり、ルート `Cargo.lock` / `cargo tree` に対する `scripts/check-forbidden-deps.sh` の検査で fail-closed に検出される
- 承認記録（2026-08-28 ユーザー承認・PR #915）・ライセンス実測・統制の全体像は `docs/framework-compare-harness-decision.md` と `docs/license-matrix.md` 8b 節を参照
