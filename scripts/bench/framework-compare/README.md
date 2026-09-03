# framework-compare: Rust ML フレームワーク横並びベンチマーク

fandhe-ai を candle・Burn と同一プロトコルで横並び比較する独立ベンチ workspace
（プロトコル同一性の範囲・例外は「計測プロトコル」節の `--mode fresh|reuse` の説明を参照。
Metal / CUDA の `fresh` GEMM 比較は例外にあたる）。
`scripts/bench/oss-gemm-compare/`（イシュー #755。許容依存第 9 区分）と同じく**本体 workspace 外の独立 Cargo workspace** であり、ルート `Cargo.toml` / `Cargo.lock` には一切現れない。本 workspace の `Cargo.lock` は比較対象として依存禁止リストのクレート（`candle-core`・`burn` と、その推移的依存の `cubecl`・`ndarray`・`tch` 等）を**意図的に含む**ため、`scripts/check-forbidden-deps.sh lock-all` は禁止リスト grep の代わりに専用の fail-closed 契約検査（Cargo.lock の存在・独自 `[workspace]` 宣言・承認済みピンのドリフト検出）を適用する（`.claude/rules/deps-policy.md`「第 9 区分」の適用範囲拡張、および `docs/framework-compare-harness-decision.md` を参照）。依存監査（advisories / bans / licenses / sources）は専用の `deny.toml` を対象に CI（`deps-forbidden` ジョブ）で毎回実行される。

実測記録（`results/summary.md`・raw JSONL）は「実行資産は scripts/bench・記録は docs/perf」の区分の例外として、再現に必要な生成物一式を本ディレクトリ配下で管理する（`docs/perf/` の実測記録群と同趣旨のコミット済み一次データ）。

## 比較対象

| フレームワーク | クレート | バージョン | デバイス |
| --- | --- | --- | --- |
| fandhe-ai | `fandhe-ai`（facade。crates.io 版） | =0.6.0 | CPU / Metal / CUDA（`tape_for(Device::…)`） |
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
- `reuse`（`bench-fandhe` の `gemm`／`train` タスクに対応。`bench-candle` / `bench-burn` は
  デバイス再利用が API 上の既定設計のため task に依らず対象外で MEASURE_ERROR を返す）:
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
  経過時間。`bench-fandhe` が依存できる公開 API 面（`fandhe-ai =0.6.0`）には「ホスト転送を
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
`--task gemm`/`--task infer` との組合せは MEASURE_ERROR で fail-fast する。

区間は「公開 API のどの呼び出しに時間が乗るか」を表し、GPU 内部（カーネル／転送）の
内訳ではない: fandhe-ai 0.6.0 の `Tensor<f32>` はホスト常駐で、CUDA/Metal の各演算は
演算ごとに H2D→カーネル→D2H を行う（`fandhe-ai-backend-cuda-0.6.0/src/ops.rs::gemm`）。
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
| `device_update` | `tape.step_device_param_store(&mut store, &grads, &config)`（grad H2D + デバイス上 SGD 発行。CUDA では非同期発行のため完了待ちは次 step の `forward_resident` に計上される） |
| `tape_drop` | `drop(tape)` |
| `step_total` | 検算用 |

**「同期待ち」を独立区間にできない理由**: `fandhe-ai =0.6.0` の公開 API 面には
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
`results/summary.md` への環境情報の統合記録は別途 #1050 に委ねる。

### 要素単位検証（イシュー #970）

`(a)` GEMM の checksum（全要素和）は、要素の入れ替わりや正負誤差の相殺で偶然一致しうる破損を
見逃す。3 バイナリ（`bench-fandhe`/`bench-candle`/`bench-burn`）は `gemm` タスクの各反復で、
結果を参照実装と**要素単位**で突合し、反復間の worst-case を JSONL の 4 フィールド
（`parity_total`・`parity_fail_count`・`parity_max_abs_err`・`parity_max_rel_err`）として記録する。

- **参照実装**: `bench-common::GemmReference`。本体 `backend-cpu::parity::matmul_reference_fma`
  と同じ FMA 契約（f32 `mul_add`・逐次 k 昇順の演算順序固定）を持つ自前 GEMM を、行ブロック分割で
  `std::thread::scope` 並列化したもの（各 `c[i][j]` の累積鎖は k 昇順のまま = 逐次実装と bit 完全
  一致。`bench-common::parity::tests::compute_is_bit_identical_to_sequential_k_ascending`）。
  fandhe-ai 0.6.0（crates.io 版）の facade は parity API を公開しておらず、candle/Burn を参照に
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
  `null` を混同しない。データ有効性節・`--strict` 対象）
- `train`/`infer` タスクは対象外（fandhe-ai の重み初期化が candle/Burn と異なる設計のため checksum
  同様に比較不能。§「計測プロトコル」重み初期化の節を参照）

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
  `fandhe-ai =0.6.0`（v0.6.0 リリースサイクルでユーザー承認済み）へ進んだが、
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

## 使い方

```bash
cd scripts/bench/framework-compare
./run_all.sh                 # macOS: cpu + metal 全組み合わせ（+ metal gemm reuse・train reuse・train phases スイープ）→ results/raw/results.jsonl
./run_all_cuda.sh            # CUDA ホスト: cuda + cpu 全組み合わせ（+ cuda gemm reuse・train reuse・train phases スイープ）→ results/raw/results-cuda.jsonl
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

- fandhe-ai・target とも reuse 行があれば reuse を優先し、無ければ fresh を使う（infer には reuse
  モード自体が存在しない。モジュール docstring 参照）
- checksum 不一致・要素単位検証の閾値超過・train reuse の checksum 不一致等（既存の無効判定と同じ規則）
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
イシュー #1147）

`summarize.py --target candle` は同一入力ファイル内の 1 レコードしか拾わない
（1 ファイル = 1 環境の単発計測が前提）ため、CUDA #1031・Metal #1037「N=1024/
2048/4096 reuse で candle 超え（各 5 回計測の中央値）」の受け入れ判定には
非対応。本節の `run_gemm_gate.sh <device> <label>`／`compare_gemm_gate.py
--device {cuda,metal}` がその 5 回計測を専用に行う（`run_ab_train_cuda.sh` /
`compare_ab.py` の GEMM 版）。本体ロジックはイシュー #1142 の CUDA 専用実装
（`run_gemm_gate_cuda.sh`）を #1147 で device 汎用化したもので、呼び出し面は
device 別の薄い wrapper `run_gemm_gate_cuda.sh`（既存呼び出しとの CLI 互換
維持）／`run_gemm_gate_metal.sh`（新規）に分離している（両者とも内部で
`bash run_gemm_gate.sh <device> "$@"` を呼ぶのみ）。

**2 系列の使い分け**:

- **正式系列**（#1031/#1037 のゲート判定の正）: `bench-fandhe/Cargo.toml` の承認済み
  ピン（現行 `=0.6.0`）でビルドしたまま計測する。コミット済み manifest・
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
bash run_gemm_gate_cuda.sh 0.6.0
# Metal 正式系列（現行ピン。イシュー #1147）:
bash run_gemm_gate_metal.sh 0.6.0

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

# 集計（N ごとに fandhe-ai reuse vs candle fresh の 5 回計測中央値・判定）:
python3 compare_gemm_gate.py results/raw/results-dgx-gemm-gate-0.6.0.jsonl
python3 compare_gemm_gate.py --device metal results/raw/results-m4max-gemm-gate-0.6.0.jsonl
echo $?   # 0: 全 N 達成 / 3: 未達または判定不能が 1 件以上 / 2: 入力を読めない
```

- `run_gemm_gate.sh <device> <label>`（device は `cuda`／`metal`。通常は device
  別 wrapper 経由で呼ぶためラベルのみを渡す）はラベル（`[A-Za-z0-9._-]+` のみ
  許可）ごとに N=1024/2048/4096 それぞれで `bench-fandhe gemm <device> <N>
  reuse` と `bench-candle gemm <device> <N> fresh`（candle は reuse 非対応）を
  交互に 5 回ずつ起動し `results/raw/results-<node>-gemm-gate-<label>.jsonl`
  （`<node>` は cuda=`dgx`／metal=`m4max`）へ記録する。失敗は
  `results/raw/skipped-<node>-gemm-gate-<label>.log` に記録する（数値を捏造しない）。
  Metal の熱・電源状態は `pmset -g therm`・`uptime`（`sudo` 不要。CUDA の
  `nvidia-smi` に相当する device 別スナップショット）で実行ログへ記録する
  （`docs/perf/metal-bench-noise-protocol.md`「熱・電源状態の記録」節準拠）。
  計測ループが完走し `ANY_FAILED == 0`（全 30 run 成功）の場合にのみ、一時
  ファイルから上記 2 パスへ原子的（同一ファイルシステム内 `mv`）に反映する。
  1 件でも run が失敗した場合はこの 2 パスを一切変更せず、直前の有効な計測
  結果（同一 label の過去の成功実行分）を保全したまま、不完全な計測データは
  `results/raw/results-dgx-gemm-gate-<label>.failed-<UTC タイムスタンプ>.jsonl`
  等の診断用別名ファイルへ退避する（fail-closed。#1166 codex-review 指摘
  PRRT_kwDOTuUCJc6euxgr／PRRT_kwDOTuUCJc6evCpq 対応。security.md A08）
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
  判定不能時は run ごとの `fail_count`/`max_abs`/`max_rel` を診断表として出力
  する（N=2048 の candle 無効データの原因調査・再現条件記録に使う。イシュー
  #1142 R2）
- `tf32:true` の行（イシュー #1042）は本ゲートの対象外として除外する

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
リリースサイクルで `=0.6.0` へさらに更新済みであり、現在のツリー（main）の
ピンは「都度同期なし」側の延長線上（after 系列。現行 `=0.6.0`）を指すが
`after-0.5.0` の値そのものではない**。「都度同期あり」側（before = 0.4.0）・
「都度同期なし」側（after = 0.5.0）を当時のまま再現するには、それぞれ
対応するピンのコミット（`=0.4.0`・`=0.5.0`）を別 worktree で checkout して
計測する。

```bash
cd scripts/bench/framework-compare
# before（現行ピン。都度同期あり）を DGX Spark 実機で計測:
bash run_ab_train_cuda.sh before-0.4.0
# ピン更新（別 PR・承認後）を適用したツリーで after を計測:
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
- 計測境界の注意: fandhe-ai 0.6.0 の `Tensor<f32>` はホスト常駐で、reuse
  モードでも各 step の `loss.to_tensor()` 実体化が単一 in-order ストリーム
  上の同期点として残る（`docs/backend-cuda-async-execution-design.md`）。
  定常状態では計測窓のずれ（1 step）を無視でき 1 step 総和と等価とみなす。

## 依存ポリシー上の位置づけ

- 本 workspace は許容依存第 9 区分（ベンチ比較対象）の適用範囲拡張として、`candle-core =0.11.0`・`burn =0.21.0` を**本ディレクトリ限定**で保持する（`.claude/rules/deps-policy.md`）
- 本体 workspace（ルート `Cargo.toml` / `Cargo.lock`）への混入は引き続き禁止であり、ルート `Cargo.lock` / `cargo tree` に対する `scripts/check-forbidden-deps.sh` の検査で fail-closed に検出される
- 承認記録（2026-08-28 ユーザー承認・PR #915）・ライセンス実測・統制の全体像は `docs/framework-compare-harness-decision.md` と `docs/license-matrix.md` 8b 節を参照
