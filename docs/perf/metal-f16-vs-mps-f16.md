# Metal f16 対 PyTorch MPS f16 計測記録（#156・TASK-8.3b）

イシュー #156「test(bench-harness): TASK-8.3b Metal f16 対 PyTorch MPS f16 の実測」の実測記録テンプレート。
REQ-8 性能下限表（`docs/spec/04-requirements.md`）の「Metal f16 対 PyTorch MPS f16」行は本イシュー着手時点で
**唯一の未実測行**であり、「未設定（v2 PoC で未実測のため）。自作カーネルでの f16 実測後に本規則で設定する」と
されていた。本ファイルは v2 で新規に自作した Metal f16 カーネル（`gemm_simdgroup_f16`）の実測手順・記録
テンプレートを整備する。**下限値そのものの確定は #158（TASK-8.3d・人間担当）であり本イシューのスコープ外。**

## 状態: 数値一致は実機検証済み（イシュー #380）。TFLOPS 実測はイシュー #383 で完了

本ファイルは当初 Linux worktree で作成され、Metal 実機・PyTorch MPS 実行環境が同一セッションで使用できな
かったため計測手順・記録テンプレートのみを整備していた。イシュー #380 で Apple Silicon 実機
（M4 Max・macOS 26.6）を用いて `tests/cpu_metal_f16_parity.rs`（数値一致回帰テスト 6 件）・
`tests/shader_source_evidence.rs`（命令実在検査）を実行し、MSL 構文検証（`gemm_simdgroup_f16` を含む
`gemm.metal` 全体のコンパイル）・数値一致（複合判定）の両方を確認済み（詳細は下記「数値一致」節）。
**TFLOPS 実測（Metal f16 対 PyTorch MPS f16 の性能比較）はイシュー #383 で実施し、下記「実測結果」節に
記録済み。REQ-8 性能下限表の当該行の下限値確定はイシュー #386（人間承認）のスコープであり、本イシューでは
行わない。**

イシュー #380（Apple Silicon 実機・M4 Max・macOS 26.6・`stable-aarch64-apple-darwin`）で以下を実機検証済み:

- `crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_f16`（`gemm_naive`/`gemm_tiled`/
  `gemm_simdgroup`/`gemm_simdgroup_tiled` を含む `gemm.metal` 全体）が `MetalGemm::new` の
  `newLibraryWithSource` で実機コンパイル成功する（**MSL 構文検証は完了**。当初の懸念「実機での最初の
  実行が構文検証を兼ねる」は成立し、かつ pass した）
- `tests/cpu_metal_f16_parity.rs` 6 件全件が `cargo test -p backend-metal --release -- --ignored --nocapture`
  で PASS（数値一致は下記「数値一致」節を参照）
- `tests/shader_source_evidence.rs`（`include_str!` ベースの文字列検査のみ。`cfg(target_os = "macos")`
  不要で Linux CI でも実行可能）は実機セッションでも green

Linux 実装環境時点で検証済みだった事項（`cargo check --tests --target aarch64-apple-darwin` による型検査
のみ・`crate::pad::pad_matrix_f16`/`unpad_matrix_f16` の単体テスト）は上記の実機実行で上書き・補完された。

## 精度契約（イシュー #380 の実機検証で確定。実装計画 §3.1 の half 統一判断から変更）

`gemm_simdgroup_f16` は当初（実装計画 §3.1・Linux 実装環境時点）A・B・累算のすべてに
`simdgroup_half8x8`（half 型統一）を使っていた。理由: `apple-silicon` スキル
（`references/msl/data-types.md`・`references/msl/simdgroup-functions.md`）のいずれにも「A/B が half・累算が
float」という混在精度オーバーロードの記載がなく、Linux 実装環境では Metal コンパイラで実地検証もできない
ため、未確認のオーバーロードを推定で使わず仕様上確実に成立する単一型テンプレートを選んでいた。

イシュー #380 で Apple Silicon 実機（M4 Max・macOS 26.6）を用い、`MTLDevice.makeLibrary(source:)`
ランタイムコンパイルによる spike を実施し以下が判明した:

1. `simdgroup_multiply_accumulate(simdgroup_float8x8&, simdgroup_half8x8, simdgroup_half8x8,
   simdgroup_float8x8)` は**コンパイル成功**する（A/B=half・アキュムレータ=float の混在オーバーロードは
   実在する）。
2. ただし `simdgroup_store(simdgroup_float8x8, device half*)` は**コンパイル不可**
   （診断: `deduced conflicting types for parameter 'T' ('float' vs. 'half')`）。float アキュムレータを
   half 出力バッファへ直接 store する経路は存在しない。

この 2 点から、`simdgroup_float8x8` → `threadgroup float` へ一旦 store → `threadgroup_barrier` で同期 →
スレッド単位で `(half)` へ変換して `device half*` へ書き戻す 2 段エピローグ（変種 B）を採用し、
`ACC_T` を `simdgroup_float8x8`（f32 累算）へ変更した（`shaders/gemm.metal::gemm_simdgroup_f16` 本体
参照）。変種 B を選ぶ理由: `device float*` へ直接 store する変種 A（C 転送バイト数が 2 倍になり本ファイルの
比較手法・後続 #383 の前提を壊す）ではなく、`dispatch_f16_prepared_unverified` のシグネチャ・
`MetalHalfBuffer` の `c_buf`・C バッファの転送バイト数を変えずに済むため。

この変更により CUDA 側 WMMA f16（`crates/backend-cuda/src/kernels_wmma.rs::gemm_wmma_f16`。
`f32.f16.f16.f32` 累算）と**精度契約が整合した**。half 累算時点（実装計画時点）の実機実測では
K=4096 ストレスケースで 60155/65536 要素（max_rel_err 1.992・max_abs_diff 1.562）が REQ-2 複合判定
（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れていたが、f32 累算化後は同一形状で複合判定 green を
実機確認済み（詳細は下記「数値一致」節）。本変更は判定式・閾値・`backend_cpu::parity` を一切変更しない
「累算精度の向上」であり、「許容誤差の緩和」ではない。

**公開 API 上の扱い（PR #346 codex-review P1-2 指摘への対応）**: `MetalGemm::dispatch_f16`／
`dispatch_f16_prepared` は `dispatch_f16_unverified`／`dispatch_f16_prepared_unverified` へ改名し
`#[doc(hidden)]` を付けている（公開ドキュメントには載せず、意図的に呼び出す利用者のみが到達できるように
する）。数値一致は #380 で実機検証済みだが、`dispatch_auto`／`dispatch_backend_auto`（production 経路）への
統合はスコープ外のため `_unverified` suffix・`#[doc(hidden)]` は当面維持する。

## 計測手順（Apple Silicon 実機）

### 1. 数値一致の先行確認（受け入れ条件に必須の前提）

```sh
git fetch origin
git checkout perf/383-metal-f16-vs-mps-f16   # イシュー #383 の実装ブランチ（origin/main〈3f72039 以降〉から作成）
cargo test -p backend-metal --release -- --ignored --nocapture cpu_metal_f16_parity
```

`tests/cpu_metal_f16_parity.rs` の全ケース（8x8x8 基準・512 基準・非倍数境界形状・K=4096 ストレス・決定性）が
PASS することを確認する。K=4096 ストレスケースが FAIL した場合も緩和せず本ファイルへ FAIL 事実
（`fail_count`・`max_abs_diff`・`max_rel_err` 等）を記録し、#158 へ引き継ぐ。

### 2. Rust 側（Metal f16）実測

```sh
cargo run -p backend-metal --example gemm_f16_bench --release
```

出力形式（`examples/gemm_f16_bench.rs` 参照）: `size=<N> metal_f16_simdgroup_tflops=<値>` 行を
size=512/1024/2048/4096（正方）で出力する。

### 3. PyTorch 側（MPS f16）実測

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

出力形式: `size=<N> pytorch_mps_f16_tflops=<値>` 行を同一形状で出力する。

## 実測結果（イシュー #383・実機実測済み）

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU | Apple M4 Max（64GB） |
| OS | macOS 26.6（build 25G72） |
| rustc | 1.96.0（`stable-aarch64-apple-darwin`） |
| torch | 2.13.0（`.venv-mps-bench`。リポジトリ管理外の一時 venv。`torch.backends.mps.is_available() == True` 確認済み） |
| 計測リビジョン | `3f7203975887ef3836a003db888b56c29232ccf6`（`git rev-parse HEAD`。#380 の f32 累算エピローグ変更を含む。`git merge-base --is-ancestor 3f72039 HEAD` で確認） |
| 計測プロトコル（Rust 側） | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）を **プロセス単位で 5 回独立実行**し、size ごとに 5 個の TFLOPS 値の中央値を採用（`docs/perf/cpu-elementwise-fusion-effect.md` §0-a・§4 と同一方式） |
| 計測プロトコル（PyTorch 側） | warmup 20 回・計測 20 回・`time.perf_counter()` 中央値（`scripts/bench/gemm_bench_torch_mps_f16.py`）を同様に **5 回独立実行**し、size ごとに中央値を採用 |
| 決定的シード | `0xC0FFEE`（両スクリプト共通） |
| 同期境界 | Rust: コマンドバッファ完了待ち／PyTorch: `torch.mps.synchronize()`（ホスト転送を伴わない完了待ち。REQ-8 v2 方針） |
| GPU 排他 | Rust 側・PyTorch 側は同時実行しない。各ラン前後に `pgrep -fl "gemm_bench\|gemm_f16_bench\|gemm_bench_torch_mps_f16"` で他プロセスとの競合がないことを確認（競合検出時は破棄・取り直す運用だが、本計測では競合は検出されなかった） |

### 数値一致（`cpu_metal_f16_parity.rs`。受け入れ条件の前提。イシュー #380 実機実測。M4 Max・macOS 26.6）

累算精度契約変更（half 統一 → f32 累算。本ファイル「精度契約」節）の before/after。before は実装計画時点の
half 統一アキュムレータでの実機実測、after は #380 の f32 累算化後の実機実測。

| ケース | before（half 累算） | after（f32 累算） |
|------|------|------|
| f16_parity_baseline_8x8x8（8×8×8） | FAIL: fail_count=7/64・max_rel_err=1.171e-2・max_abs_diff=1.953e-3 | PASS |
| f16_parity_boundary_shapes_non_multiple_of_eight（17×19×23） | FAIL: fail_count=92/323・max_rel_err=4.754e-2・max_abs_diff=7.812e-3 | PASS |
| f16_parity_baseline_shape_512（512³） | FAIL: fail_count=203946/262144・max_rel_err=1.998e0・max_abs_diff=2.500e-1 | PASS |
| f16_k4096_stress（256×256×4096） | FAIL: fail_count=60155/65536・max_rel_err=1.992e0・max_abs_diff=1.562e0 | PASS |
| f16_dispatch_is_bit_deterministic_across_runs | PASS | PASS |
| f16_dispatch_prepared_rejects_undersized_and_misaligned_inputs | PASS | PASS |

6 件全件が after（f32 累算）で PASS。judgement 式・閾値・`backend_cpu::parity` は変更していない
（`git diff -- crates/backend-cpu/src/parity.rs` は空）。

**イシュー #383（計測リビジョン `3f72039`）での再確認**: TFLOPS 実測に先立ち同一 SHA で
`cargo test -p backend-metal --release -- --ignored --nocapture cpu_metal_f16_parity` を再実行し、
6 件全件（`f16_parity_baseline_8x8x8`・`f16_parity_boundary_shapes_non_multiple_of_eight`・
`f16_parity_baseline_shape_512`・`f16_k4096_stress`・`f16_dispatch_is_bit_deterministic_across_runs`・
`f16_dispatch_prepared_rejects_undersized_and_misaligned_inputs`）が PASS することを確認済み
（数値一致は #380 実機実測時と同一の f32 累算エピローグのままであり、本イシューでは緩和・変更していない）。

### TFLOPS 比較（Metal f16 対 PyTorch MPS f16）

Rust 側・PyTorch 側それぞれ 5 プロセス独立実行の結果。「中央値」は size ごとの 5 個の TFLOPS 値の中央値、
「比」は **Metal 側 5 回中央値 ÷ PyTorch 側 5 回中央値**（size ごとの比の中央値ではない。REQ-8 の比較定義に
合わせた算出方法）。

| size | Metal f16 TFLOPS（simdgroup。5 回中央値） | Metal レンジ | PyTorch MPS f16 TFLOPS（5 回中央値） | PyTorch レンジ | Metal/PyTorch 比 |
|------|------|------|------|------|------|
| 512  | 1.1554 | 0.9379〜1.1799 | 1.2055 | 0.9761〜1.5592 | 0.9584（≒ 95.8%） |
| 1024 | 2.1777 | 2.1403〜2.1899 | 5.5679 | 3.8260〜5.7458 | 0.3911（≒ 39.1%） |
| 2048 | 2.4426 | 2.3171〜2.4591 | 11.2803 | 10.6343〜12.9570 | 0.2165（≒ 21.6%） |
| 4096 | 2.2411 | 2.1379〜2.5029 | 12.0605 | 11.6667〜12.8414 | 0.1858（≒ 18.6%） |

REQ-8 の主指標は 2048/4096（512 は起動オーバーヘッド支配のため参考値。PoC-v2-4 先例）。**主指標の比は
21.6%（size=2048）・18.6%（size=4096）であり、PyTorch MPS f16 に対して Metal 自作カーネルは大きく劣後する
実測結果となった。** これはカーネル最適化の要否を示す事実であり、本イシューではカーネル最適化・下限設定・
許容誤差の変更は一切行わない（#386/#387 へ引き継ぐ）。

#### 5 回生値（付録）

| size | Metal f16 TFLOPS（run1〜5） | PyTorch MPS f16 TFLOPS（run1〜5） |
|------|------|------|
| 512  | 0.9379 / 1.1799 / 1.1639 / 1.1554 / 1.1498 | 1.5592 / 1.3484 / 0.9761 / 1.0246 / 1.2055 |
| 1024 | 2.1822 / 2.1777 / 2.1899 / 2.1403 / 2.1619 | 5.7161 / 5.7458 / 5.5679 / 3.8260 / 4.2866 |
| 2048 | 2.4552 / 2.4426 / 2.4591 / 2.4202 / 2.3171 | 12.9570 / 10.6343 / 11.3686 / 11.2326 / 11.2803 |
| 4096 | 2.5029 / 2.5027 / 2.2411 / 2.1379 / 2.1992 | 12.8414 / 12.5233 / 12.0605 / 11.7053 / 11.6667 |

生ログは実機実測時の標準出力をそのまま転記した値であり、外挿・推定値は含まない
（`.claude/rules/coding-rust.md`「テスト・ベンチ」節: 5 回計測の中央値を採用する方針）。

### 参考: PoC-v2-4 f32 実測との対比

PoC-v2-4（Apple M4 Max・size=4096）の f32 実測: simdgroup 3.134 TFLOPS 対 PyTorch MPS（f32）13.505 TFLOPS
（比 ≒ 23.2%。`docs/spec/03-poc/poc-v2-4-metal-gemm/README.md`「計測結果」節）。

**前提の更新（#383 時点）**: PoC-v2-4 時点および実装計画 §3.1 時点の f16 カーネルは half 統一アキュムレータ
だったが、現行カーネル（#380 で変更）は **`simdgroup_float8x8` による f32 累算 ＋ threadgroup 経由の 2 段
エピローグ（`threadgroup_barrier` を伴う）**である（本ファイル「精度契約」節）。したがって「half 累算が
性能に与える影響」ではなく、**f32 累算化・2 段エピローグのオーバーヘッドが f16 の理論的な速度優位（Apple GPU
は一般に f16 が f32 の約 2 倍の理論ピーク演算性能を持つ）をどれだけ相殺しているか**が論点になる。

実測結果（size=4096）を PoC-v2-4 の f32 実測と対比すると、Metal 側は f32 3.134 TFLOPS に対し f16 2.2411
TFLOPS（f32 の約 71.5%）であり、**f16 は f32 の理論上の速度優位を得られず、PoC-v2-4 の f32 実測を下回っている**
（この結論は f32 baseline に PoC-v2-4 を用いた場合のものであり、baseline の選び方に依存する。後述の追記参照）。
PyTorch 側は f32 13.505 TFLOPS に対し f16 12.0605 TFLOPS（f32 の約 89.3%）であり同様の傾向はあるが Metal 側
ほど顕著ではない。したがって Metal/PyTorch 比の低下（f32 23.2% → f16 18.6%、size=4096）は、PyTorch 側の
f16 性能低下より Metal 側の f16 性能低下（2 段エピローグの `threadgroup_barrier` 同期コスト等が疑われるが、
本イシューでは原因分析・最適化は行わず事実の記録に留める）が相対的に大きいことを示唆する。

**追記（本ファイル rebase 時点。#437 で #381 が完了）**: `docs/perf/metal-gemm-dynamic-tile.md` の
TFLOPS 実測欄は #437 で記入済みとなった。同一実機（M4 Max）の `gemm_simdgroup`（f32・simdgroup 単独。本節が
対比対象とする現行 f16 カーネルと同じ実行系列）の canonical 値は size=4096 で 1.7432 TFLOPS
（`docs/perf/metal-gemm-dynamic-tile.md` run1）であり、上記 PoC-v2-4 の 3.134 TFLOPS より低い。#437 自身が
「本実測の絶対 TFLOPS が PoC-v2-4 実測値より低い」（外部ディスプレイ接続によるコンポジタ負荷等の計測衛生条件
の差、`docs/perf/metal-gemm-dynamic-tile.md`「計測衛生」節参照）と明記しており、PoC-v2-4・#437・本ファイルの
3 実測はいずれも異なるセッション・計測衛生条件下のものであるため、絶対値同士の対比精度には限界がある
（f32 baseline をどちらに取るかで「f16 が f32 実測を下回る」という上記結論の解釈が変わりうる）。本イシュー
では f32 baseline の選定基準を新たに定義せず、上記は元々の実装計画・本イシュー着手時点で参照可能だった
PoC-v2-4 を主対比として維持する。#437 との対比を含めた要因分析は、上記のとおり **#387** のスコープとする。

## 総括・要因分析（イシュー #387）

イシュー #387（Metal 実機実測結果の総括反映）のスコープとして、Metal/PyTorch 比が主指標（2048/4096）で
2 割前後（21.6%・18.6%）に留まった実測事実（本ファイル「実測結果」節・#383）について、既存の実測記録
（本ファイル・`docs/perf/metal-gemm-dynamic-tile.md`〈#381・#437〉）のみを根拠とする定性的要因整理を行う。
**新規の実測・カーネル最適化は行わない**（本イシューはドキュメントのみの変更）。

要因として次の 3 点が挙げられる:

1. **現行 f16 カーネルは simdgroup 単独系の実装**である。上記「参考: PoC-v2-4 f32 実測との対比」節のとおり
   `gemm_simdgroup_f16` は `simdgroup_float8x8` による f32 累算 ＋ threadgroup 経由の 2 段エピローグ
   （`threadgroup_barrier` を伴う）を持つのに対し、f32 側では `docs/perf/metal-gemm-dynamic-tile.md`
   （#381 実測・run1）で dynamic-tile 経路が simdgroup 単独（1.7432 TFLOPS @ size=4096）に対して
   **約 1.74 倍**（`gemm_simdgroup_tiled`／動的タイル選択の効果。同ファイル「TFLOPS 実測」節）まで性能を
   引き上げている。f16 カーネルには dynamic-tile 相当の最適化が未適用であり、simdgroup 単独系のオーバー
   ヘッド（2 段エピローグの同期コスト等、本ファイル「参考」節で既述）がそのまま比率に反映されている
2. **比較対象の PyTorch MPS は Apple 純正の高度最適化カーネル**である。自作 1 世代目カーネル
   （simdgroup 単独＋f32 累算エピローグ）と成熟した商用実装を直接比較しているため、絶対性能差が大きく
   出ることは実装成熟度の観点から予期される範囲内であり、Metal バックエンド固有の欠陥を示すものではない
3. **セッション・計測衛生条件の異時点性による絶対値対比の限界**: 本ファイル「追記（本ファイル rebase
   時点）」節で既述のとおり、PoC-v2-4（3.134 TFLOPS）・#437（1.7432 TFLOPS）・本ファイルの実測（#383）は
   いずれも異なるセッション・計測衛生条件下のもので、外部ディスプレイ接続によるコンポジタ負荷等の要因
   （`docs/perf/metal-gemm-dynamic-tile.md`「計測衛生」節）を含む。したがって f32 baseline の取り方に
   よって比率の解釈が変わりうる点は、上記 1・2 の構造的要因とは別に留意する

上記 1 の「dynamic-tile 相当の最適化を f16 カーネルへ適用するか」は、REQ-8 最適化後段階の再確定
（`docs/perf/performance-floor-decision.md`）と一体で検討すべきカーネル最適化タスクであり、**本イシュー
では実施しない**（残存項目として `docs/backend-metal-real-device-testing.md`「将来課題」節に記録する）。

## 未実施・後続作業

- ~~本ファイルの「実測結果」節は Apple Silicon 実機での `cargo test -- --ignored`・`cargo run --release`・
  PyTorch スクリプト実行後に埋める~~ → **イシュー #383 で完了**（本ファイル「実測結果」節）
- ~~下限値の確定（REQ-8 性能下限表の当該行の更新）は **#386**（人間承認）が行う。本ファイルの実測結果を
  入力として使う。**下限確定の判断は `docs/perf/performance-floor-decision.md` を参照**（本イシューでは
  同ファイルの §2/§3/§5(b) を更新していない。「#156: 実測未実施」の記述は本イシューの実測完了により陳腐化
  したが、下限確定は #386 のスコープであるため本イシューでは書き換えない）~~ → **#386 で完了**（初期リリース
  下限 15% を確定。`docs/perf/performance-floor-decision.md` §2/§3 に参照注記・§5(b) に解消記録・§8 に
  確定記録を追加済み。`crates/bench-harness/src/threshold.rs::floor_spec` も更新済み）
- `docs/spec/04-requirements.md`（正本 submodule）の更新は本リポでは行わない。仕様変更は spec リポ側で対応する
  （`.claude/rules/out-of-scope-tracking.md`「仕様変更が必要な場合」）
- f16 の自動ディスパッチ規則への統合（`docs/dispatch-rules-design.md`）は本イシューのスコープ外
  （実装計画 §3.4「Metal f16 行は含めない」）。REQ-11 系の後続課題として別途追跡する
- K=4096 ストレスケースは #383 の再確認でも複合判定 PASS のままであり、許容誤差の再評価は不要（本ファイルは
  事実の記録のみを担い、閾値変更の判断は行わない）
- ~~Metal/PyTorch 比が主指標（2048/4096）で 2 割前後に留まった実測事実の総括・要因分析は **#387** へ引き継ぐ
  （本イシューではカーネル最適化を行わない）~~ → **#387 で完了**（本ファイル「総括・要因分析（イシュー
  #387）」節を参照。カーネル最適化は未実施のまま残存項目として記録）

## タイル化後再計測（イシュー #799）

GEMM OSS 比較ギャップ改修ツリー（#785）Phase 2（#796〜#798）で、上記「総括・要因分析（イシュー #387）」
節が要因 1 として挙げた「dynamic-tile 相当の最適化が f16 カーネルに未適用」を解消した。非タイル
`gemm_simdgroup_f16`（1 threadgroup 1 simdgroup 8x8）を、タイル化カーネル `gemm_simdgroup_tiled_f16`
（BM/BN/BK/WM/WN・ベクトル化ロード・動的タイル選択込み。#796 本体・#797 ベクトル化ロード／エピローグの
タイル粒度統合・#798 動的タイル選択統合）へ世代更新した後の再計測プロトコルを本節に追補する。

### 直近の確定実測（本節着手時点の対 MPS f16 比。#785 本文・2026-08-21 再計測）

旧経路（非タイル `gemm_simdgroup_f16`）での最新値: size=4096 で 2.27 TFLOPS 対 PyTorch MPS f16
12.26 TFLOPS ＝ **18.5%**（`docs/perf/metal-gemm-tgid-swizzle-ab.md`・`docs/perf/metal-floor-remeasurement.md`
の系譜と同一計測プロトコル）。GEMM OSS 比較ギャップ改修ツリー（#785）本文が挙げる負けカードの中で最大の
負け幅であり、本追補節が対象とする改善対象値である。

### 新経路の計測線の追加（`gemm_f16_bench.rs`）

`crates/backend-metal/examples/gemm_f16_bench.rs` へ新経路の計測線を追加した（イシュー #799）。旧経路
（`measure`・`metal_f16_simdgroup_tflops=` 行）に加え、`tile::select(m, n, k)`（本番ディスパッチ
`dispatch_auto` と同じ選択関数）が選ぶ `TileConfig` で
`MetalGemm::dispatch_f16_tiled_prepared_unverified` を計測する `measure_tiled`
（`metal_f16_tiled_tflops=` 行）を並記出力する。計測境界（プリパド済みバッファを計測外で準備し、
計測対象はエンコード＋コマンドバッファ完了待ちのみ）・`SEED`（`0xC0FFEE`）・入力分布は新旧経路で完全に
揃えてあり、同一プロセス内で新旧比較ができる。`pipeline_for_tile_f16` がデバイス上限超過等でサイレントに
`TileConfig::SINGLE_SIMDGROUP_8X8` へフォールバックしうる（`gemm.rs::dispatch_f16_tiled_prepared_unverified`
ドキュメントコメント参照）ため、resolved 構成を `tile=` として出力へ含め、実際に採用された構成を透明化
している。

### 実機計測手順（macOS・Apple Silicon。タイル化後経路を含む）

```sh
git fetch origin
git checkout perf/799-metal-f16-remeasurement   # イシュー #799 の実装ブランチ（origin/main〈f8d26c4〉以降から作成）

# 1. 数値一致の先行確認（3 系統。全 PASS が計測の前提）
cargo test -p backend-metal --release -- --ignored --nocapture cpu_metal_f16_parity
cargo test -p backend-metal --release -- --ignored --nocapture cpu_metal_f16_tiled_parity
cargo test -p backend-metal --release -- --ignored --nocapture gemm_f16_auto_parity

# 2. Rust 側（新旧 f16 経路。プロセス単位で 5 回独立実行）
cargo run -p backend-metal --example gemm_f16_bench --release

# 3. PyTorch 側（MPS f16。同一セッションで 5 回独立実行）
source .venv-mps-bench/bin/activate
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

出力形式（`gemm_f16_bench.rs` 参照）: size ごとに `metal_f16_simdgroup_tflops=`（旧経路。回帰基線として
維持）と `metal_f16_tiled_tflops=`（新経路。`tile=` で resolved `TileConfig` を併記）の 2 行を出力する。
比較手順・GPU 排他・計測衛生（AC 電源・`pmset -g therm`）は上記「実測結果（イシュー #383）」節の
「計測環境」表と同一条件を踏襲する。

### 状態: プロトコル整備済み・実測は Mac 実機セッションで消化（#799 未消化）

本実装セッションは Linux dev-box（`.claude/worktrees/wf_6c80a1fd-533-85`）であり、`docs/real-hardware-verification-env.md`
§1 が定める Metal 実機（M4 Max）到達手段は「ローカル直接実行」のみで、Linux dev-box からの SSH 経路・
`docs/real-hardware-verification-env.local.md` はいずれも本環境に存在しない（同型の Metal イシュー
#795・#814・#818・#821 と同じ実機到達不可の先例）。したがって本節は計測線の追加（上記「新経路の計測線の
追加」節）・計測手順の確立に留め、per-size（512/1024/2048/4096）の 5 回中央値・Q1/Q3・対 MPS f16 比・
改善幅（旧経路 18.5%（#785）〜18.78%（#437 系譜）→ 新経路実測値）・残ギャップの実測記入は **Mac 実機
セッションでの後続消化**へ引き継ぐ。**実測値の推定・外挿・捏造は一切行わない**
（`.claude/rules/coding-rust.md`「テスト・ベンチ」節）。

実測完了時は本節（または新設の実測結果表）へ以下を記録する: 計測環境表（GPU・OS・rustc・torch
バージョン・計測リビジョン）・数値一致 3 系統の PASS 確認・per-size 5 回生値・中央値/Q1/Q3・
新旧経路それぞれの対 MPS f16 比・改善幅（pt）。
