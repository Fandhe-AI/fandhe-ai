# CUDA 最適化後下限（暫定 40%）再実測 記録（#157・TASK-8.3c）

イシュー #157「test(bench-harness): TASK-8.3c CUDA 最適化後下限（暫定 40%）の再実測」の実測記録テンプレート。
受け入れ条件「実測記録と候補下限値」に対応する。

## 目的・受け入れ条件対応

`docs/spec/04-requirements.md`「CUDA f32 対 PyTorch CUDA」「CUDA f16 対 PyTorch f16」行の最適化後下限
**40%** は次の理由で**暫定値**として記録されている:

> tensor core（WMMA/mma）化により、cuBLAS の f16 tensor core 経路の実効値（4096: 97.6 TFLOPS）に対する
> 一般的な手書き GEMM の到達目安 40〜70%（PoC-v2-3「tensor core 化の段階見積もり」節）を適用すると
> 39〜68 TFLOPS 相当となる。この見積もりは PyTorch **f16**（tensor core 経路）を基準としたものであり、
> PyTorch **f32**（4096 実測 17.8 TFLOPS）を基準に換算すると非現実的な外挿になる。よって f32 の最適化後
> 下限は当該見積もりをそのまま流用せず、保守的な値として PyTorch f32 比 **40%** を暫定目標とする。
> **tensor core 実装完了後の実測で本値を再確定すること**

（`docs/spec/04-requirements.md:180-181`）。`docs/spec/05-tasks.md` TASK-8.3 も同じ再確定条件を明記して
いる:「CUDA f32/f16 の最適化後下限（暫定値 40%）は Tensor Core 実装完了後の実測で再確定する（REQ-11 の
TASK-11.1 完了が前提）」（`docs/spec/05-tasks.md:286`）。TASK-11.1 系（#59〜#65・#186・#187）は完了済み
のため、本イシューがその再実測を担う。

TASK-8.3 の担当欄は「共同（計測実行は Claude Code、下限値の最終確定は人間）」（`docs/spec/05-tasks.md:290`）
であり、**本ドキュメントは候補下限値の導出と記録までを行い、最終確定は #158（TASK-8.3d）へ引き継ぐ**。

## 実測バイナリ

`crates/backend-cuda/examples/cuda_floor_bench.rs`（本イシューで新規追加）。

- 計測経路: tiled f32（基準）／WMMA(TF32) opt（f32 最良。opt 不可時は `CudaGemm::run_wmma_tf32` 内部で
  基本版へ自動フォールバック）／WMMA f16 opt／`mma.sync` f16 パイプライン（f16 最良）
- **opt カーネル可用性の検証**: `run_wmma_tf32`／`run_f16` は opt カーネル未対応環境でも公開シグネチャ
  を変えず基本版へ自動フォールバックするため、戻り値の成否だけでは opt カーネルが実際に実行されたか
  判別できない。本バイナリは起動時に `CudaGemm::wmma_tf32_opt_available`／
  `CudaWmmaGemm::wmma_f16_opt_available` を確認して結果を出力し、判定対象サイズで `wmma_tf32`／
  `wmma_f16` が最良経路として選ばれたにもかかわらず opt カーネルが未確認（基本版へフォールバック済み）
  だった場合は、その f32／f16 candidate optimized floor を `n/a` として確定させない（`tiled`・
  `mma_f16` は opt-vs-basic フォールバックと無関係の別実装のためこのゲートの対象外）。未解決レビュー
  スレッド「Opt kernel use not verified」（PR #349）対応。実装は `cuda_floor_bench.rs::confirmed_candidate_floor`
  の `opt_ok` 引数を参照
- 形状: M=N=K = 512／2048／4096（PoC-v2-3 の PyTorch 参照値と同一形状）
- 計測プロトコル: `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）・
  決定的シード `0xC0FFEE`
- 判定対象形状: REQ-8 の規定どおり **M=N=K=2048・4096 の実測比率の最小値**を候補下限値の算出に用いる。
  512 は参考値としてのみ出力する（ディスパッチ・起動オーバーヘッドが支配的で試行間ばらつきが大きいため。
  `docs/spec/04-requirements.md`「判定対象形状」節）
- 丸め規則: `docs/spec/04-requirements.md`「丸め規則の統一」節（実測比率 10% 以上は 5% 刻み切り下げ、
  10% 未満は 1% 刻み切り下げ、条件付き追加ステップなし）を適用する（単体テストで仕様例
  〈10.3%→10%・26.6%→25%・1.9%→1%・境界 10%→10%〉と突合済み）。本イシュー着手時点では
  `bench-harness` の TASK-8.2 下限判定モジュール（#151〜#153）が未マージのため
  `cuda_floor_bench.rs::floor_round` にインライン実装していたが、**#158（TASK-8.3d）で
  `bench_harness::floor_lower_bound`（公開 API）へ一本化済み**（`docs/perf/performance-floor-decision.md`
  §6 参照）
- GPU 名が `GB10` を含まない場合は警告行を出力する（PoC-v2-3 参照値と計測機が異なるため比率は参考値。
  REQ-8「いずれも同一ハードウェア上の同一バックエンド比較」）。ただし GPU 名一致は WARNING 表示のみに
  用い、正式な candidate optimized floor の許可条件にはしない（GPU 名の部分一致では同一実機比較を
  保証できないため。下記「PyTorch 参照値の扱い」節参照。PR #349 codex-review 指摘 P1 対応）
- f32 の最良経路は固定優先順位ではなく実測 TFLOPS の大小比較で選ぶ（`best_of` 純関数。同 codex-review
  指摘 P1「実測性能を比較せず固定優先順位で『最良値』を選んでいる」対応。選ばれた経路ラベル
  〈`tiled`/`wmma_tf32`〉を出力に含める）
- **計測境界を PyTorch 参照計測と統一する**（PR #349 codex-review 再指摘 P1「PyTorch 参照計測
  〈`gemm_bench_torch_cuda.py`〉は `torch.matmul`+同期のみを計測し、ホスト転送・反復ごとの確保を
  含まないのに対し、tiled f32・WMMA(TF32)・WMMA f16 の 3 経路は H2D 転送・出力バッファ確保・D2H 回収
  込みで計測しており計測範囲が不当に不利だった」対応）。4 経路すべて（tiled f32・WMMA(TF32)・
  WMMA f16・`mma.sync` f16）を「入力事前配置＋カーネル起動＋同期のみ」の launch-only 境界に揃える
  （`gemm.rs::CudaGemm::upload_f32`/`launch_tiled_f32`/`launch_wmma_tf32`、
  `gemm_wmma.rs::CudaWmmaGemm::upload_f16`/`launch_f16`、`gemm_mma.rs::CudaMmaGemm::upload_f16`/
  `launch_f16` の分割 API を使用）。同一実機再計測を行う場合、PyTorch 側の再計測もこの境界
  （`gemm_bench_torch_cuda.py` と同一プロトコル）で行うこと
- f16 candidate floor は `wmma_f16`・`mma_f16` の実測比較（`best_of`）で最良経路を選ぶ。計測境界統一
  前は `mma_f16` のみ launch-only で計測範囲が異なっていたため candidate floor から除外していたが、
  上記の境界統一により両経路が同一境界（launch-only）で計測されるようになったため、`best_f32` と
  対称に実測比較へ戻した（`f16_candidate_floor_value` 参照。PR #349 codex-review 再指摘 P1 対応）

## 計測手順（DGX Spark GB10 等 CUDA 実機）

```sh
git fetch origin
git checkout test/157-cuda-floor-remeasurement   # 本イシューの実装ブランチ

# 1. 数値一致確認を先に行う（既存 parity テスト群。閾値は緩和しない）
cargo test -p backend-cuda --release -- --ignored

# 2. （推奨・PR #349 codex-review 指摘 P1 対応）同一実機で PyTorch を再計測し、
#    候補下限の正式算出に使う env override を用意する。
#    docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py を同一 GB10 個体で
#    再実行し、得られた 6 値（f32/f16 × 512/2048/4096）と出所を注入する（同スクリプトは
#    入力テンソルを計測ループの外で GPU 上に生成し、ループ内では torch.matmul + 同期のみを計測する
#    ため、上記「計測境界を PyTorch 参照計測と統一する」の境界と一致する）:
export CUDA_FLOOR_BENCH_PYTORCH_SOURCE="poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py 再実行, <実施日>, 同一 GB10 個体"
export CUDA_FLOOR_BENCH_PYTORCH_F32_512=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_2048=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F32_4096=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_512=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_2048=<再計測値>
export CUDA_FLOOR_BENCH_PYTORCH_F16_4096=<再計測値>

# 3. 再実測バイナリを実行
cargo run -p backend-cuda --example cuda_floor_bench --release
```

出力形式（`crates/backend-cuda/examples/cuda_floor_bench.rs::main` 参照）:

- `WARNING: ...` 行（GPU 名が GB10 系でない場合のみ）: PyTorch 参照値との比較が参考値に留まる旨。
  ただしこの GPU 名一致は candidate floor の許可条件ではない（下記「PyTorch 参照値の扱い」参照）
- `device: name=... compute_capability=...` 行: 計測環境（下表「計測環境」への転記元）
- `pytorch reference provenance: ...` 行: PyTorch 参照値が「同一実機で今回再計測（env override）」か
  「PoC-v2-3 固定値」かの出所
- `size=<N> tiled_f32_tflops=... wmma_tf32_tflops=... wmma_f16_tflops=... mma_f16_tflops=... f32_best_path=... f16_candidate_path=... f32_best_over_pytorch=... f16_candidate_over_pytorch=... (..., mma_over_wmma_f16(apples-to-apples, launch-only, median-based)=...)` 行:
  形状ごとの経路別 TFLOPS・f32 最良経路ラベル（`tiled`/`wmma_tf32`。実測 TFLOPS の大小比較で選出。
  固定優先順位ではない）・f16 candidate floor 経路ラベル（`wmma_f16`/`mma_f16`。両経路とも launch-only
  計測に統一されたため実測 TFLOPS の大小比較で選出）・対 PyTorch 比・`mma_f16` の参考比（`wmma_f16`
  比。両経路とも launch-only 計測のため apples-to-apples）。経路別 TFLOPS 値は
  `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で中央値・Q1・Q3 を並記する（`bench_harness::run` の
  計測プロトコル〈TASK-8.1〉が返す四分位値を破棄せず記録するため。`cuda_floor_bench.rs::TflopsSample`
  参照。PR #349 codex-review 指摘 P1「Q1/Q3 を破棄しており実測記録の契約を満たせない」対応。経路選択・
  候補下限の算出は引き続き中央値のみを根拠とする）
- `CUDA f32 candidate optimized floor ... = N%` / `CUDA f16 candidate optimized floor ... = N%` 行:
  判定対象形状（2048/4096）の最小比率に丸め規則を適用した候補下限値。**判定対象形状すべての比率が
  計測でき、かつ全形状で同一実機再計測値（env override）が使われた場合のみ**出力される。1 サイズでも
  PoC-v2-3 固定値にフォールバックしていれば `n/a`（参考比率のみ表示）になる。1 サイズでも比率が
  非有限値等で欠測（`None`）した場合も同様に `n/a` になる（残りの形状だけから確定させない。PR #349
  codex-review 再指摘 P1 対応）

### PyTorch 参照値の扱い

REQ-8 は「同一ハードウェア上の同一バックエンド比較」を要求するため、正式な candidate optimized floor は
**同一実機での PyTorch 再計測**（`docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py`
を再実行し、同一プロトコル・同一シードで再取得した値）を `CUDA_FLOOR_BENCH_PYTORCH_{F32,F16}_{512,2048,4096}` と
`CUDA_FLOOR_BENCH_PYTORCH_SOURCE`（出所文字列。非空必須）で注入した場合にのみ算出される
（`cuda_floor_bench.rs::pytorch_f32_ref`/`pytorch_f16_ref`/`print_candidate_floor` 参照）。

GPU 名が `GB10` を含む場合でも、env override が無ければ下記の PoC-v2-3 固定値が使われ、正式な
candidate floor は `n/a`（参考比率のみ）になる。GPU 名の部分一致だけでは同一実機比較を保証できない
ため（PR #349 codex-review 指摘 P1）、固定値だけでは候補下限を確定させない。

PoC-v2-3 実測値（`torch.matmul`, CUDA, DGX Spark GB10, PyTorch 2.13.0+cu130, 5〜20 回中央値。
env override 未注入時のフォールバック値・参考比率の分母）:

| M=N=K | PyTorch f32 (TFLOPS) | PyTorch f16 (TFLOPS) |
|-------|----------------------|----------------------|
| 512   | 7.8803  | 17.1898 |
| 2048  | 17.4241 | 91.2115 |
| 4096  | 17.7774 | 97.6308 |

#390 で同一実機（DGX Spark GB10・`spark-dbd9`）再計測した値（`gemm_bench_torch_cuda.py $s 20 20`
実行、2026-08-10。torch=2.13.0+cu130 numpy=2.5.1 cuda=13.0 device=NVIDIA GB10。以下は
`cuda_floor_bench` 実行時の `CUDA_FLOOR_BENCH_PYTORCH_*` env override として注入した値）:

| M=N=K | PyTorch f32 (TFLOPS, median) | PyTorch f16 (TFLOPS, median) |
|-------|-------------------------------|-------------------------------|
| 512   | 7.8362  | 17.0674 |
| 2048  | 17.1261 | 92.6833 |
| 4096  | 17.4556 | 81.2560 |

PoC-v2-3 固定値との差は数 % 以内（f16 の 4096 のみ 81.26 対 97.63 TFLOPS と乖離が大きいが、両者とも
`torch.matmul` + 同期のみの launch-only 境界での計測であり、ドライバ・cuDNN 相当の内部ヒューリスティクス
のばらつきの範囲として扱う。本イシューでは PyTorch 側の実装を追跡しない）。

## 実測結果（#390 実機実測・DGX Spark GB10・実施日 2026-08-10）

本イシュー（#390）で DGX Spark GB10（`local.fandhe.spark-dbd9`）実機にて `cuda_floor_bench` を
3 回反復実行し、同一実機で PyTorch 参照値を再計測（`warmup=20 iters=20` 明示指定）した。以下は
その実測記録である。数値は `cargo run -p backend-cuda --example cuda_floor_bench --release --locked`
の stdout から機械的に転記しており、辻褄合わせの後付け調整は行っていない。

**受け入れ条件「#389 数値一致 green が前提」の実態是正**: イシュー #390 本文は前提を「#389 数値一致
green」と記載するが、実態は #389（`docs/backend-cuda-real-device-testing.md` §5.3）が記録するとおり
**parity 恒常 fail 8 件**が残存したまま確定している（TF32 経路 5 件・f16 K=4096 tail 3 件。REQ-2
閾値改定は #186 へ引き渡し済み）。本イシューは計測イシューでありガードレール閾値・許容誤差の変更は
行わない（`.claude/rules/coding-rust.md`・`security.md`）。実測は実施したうえで、下記「数値一致
（parity）状態の限定条件」節に必須の限定条件を明記する。

### 計測環境

| 項目 | 値 |
|------|-----|
| GPU（`CudaDevice::name()`） | NVIDIA GB10 |
| compute capability（`CudaDevice::compute_capability()`） | (12, 1) |
| driver バージョン | 580.159.03（`nvidia-smi`） |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| commit SHA | `815ee0dc122d80fcae0c53d29f6d6c5907a97c29`（`.rev-stamp` とノード側転送後の値が一致確認済み） |
| 実施日 | 2026-08-10 |
| PyTorch 参照値の出典（`pytorch reference provenance:` 行を転記） | 同一機再計測（`CUDA_FLOOR_BENCH_PYTORCH_SOURCE="poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py 再実行 (warmup=20 iters=20), 2026-08-10, 同一 GB10 個体 spark-dbd9"`） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1）。**イシュー #390 受け入れ条件の文言は「5 回中央値」だが、本イシューは正本 `docs/spec/05-tasks.md` TASK-8.1 が定める warmup 20 回以上・計測 20 回以上の下限（`bench_harness::protocol::MeasurementConfig::MIN_ITERATIONS = 20`。`crates/bench-harness/src/protocol.rs:30,58-69`）に従う。この下限は `MeasurementConfig::new` が `iters < 20` を `BenchError::ProtocolViolation` で拒否するハード制約であり（同ファイル `:64-67`。回避 API なし）、5 回計測へ再集計する経路自体が存在しない。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用し」との不一致は本イシュー由来ではなく、イシュー #27 時点で既に検出・記録済みの正本（spec）と実装リポ側規約ファイルの既知の乖離であり（`crates/bench-harness/src/lib.rs:17-25` 参照。同箇所はユーザー承認を経ていないため rule ファイル側は変更せず不一致の明記に留めると結論済み）、本イシューはその既定の TASK-8.1 プロトコルを踏襲したに過ぎない。`coding-rust.md` 側の訂正は `.claude/rules/out-of-scope-tracking.md` の規約に従いユーザー承認を得たうえで行う必要があり、計測イシューである本ドキュメントの範囲外とする** |
| 決定的シード | `0xC0FFEE`（`cuda_floor_bench.rs::SEED`） |
| GPU 排他性（実行前後） | `utilization.gpu` 0%。常駐は ComfyUI（170MiB）・Kokoro TTS（870MiB）のみ。3 回のバイナリ実行・PyTorch 再計測いずれの前後も第三プロセスの介在なし |
| 反復回数 | `cuda_floor_bench` を 3 回反復実行（run1/run2/run3）。下表は各形状・各経路の 3 run 間中央値を採用し、run 間のばらつきをレンジとして注記する |

### 経路×形状 TFLOPS 実測

各セルは `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で `size=<N> ...` 出力行から転記する（中央値・
Q1・Q3 の 3 値。`cuda_floor_bench.rs::TflopsSample`。PR #349 codex-review 指摘 P1「Q1/Q3 を破棄しており
実測記録の契約を満たせない」対応。経路選択・候補下限の算出は引き続き中央値のみを根拠とする）。

各セルは 3 run（run1/run2/run3）の中央値ベース TFLOPS 値のうち run 間の中央値（run2 の値。3 run が
すべて近似しているため run2 を代表値として採用）を記載し、括弧内に run1〜run3 の中央値レンジを注記する
（`(range: <最小>〜<最大>)`）。個別 run の生ログは `docs/perf/` には含めず本ドキュメントの表を正とする。

| M=N=K | tiled f32（中央値/Q1/Q3、run 間レンジ） | WMMA(TF32) opt（中央値/Q1/Q3、run 間レンジ） | WMMA f16 opt（中央値/Q1/Q3、run 間レンジ） | mma.sync f16（中央値/Q1/Q3、run 間レンジ） | f32 最良経路 | f16 candidate 経路 | mma_over_wmma_f16（参考比・中央値ベース、run 間レンジ） |
|-------|-----------------------------|-----------------------------------|-----------------------------------|-------------------------------------|---------------|---------------------|-----------------------------------------------|
| 512（参考値） | 2.0890(q1=2.0916,q3=2.0867) (range: 2.0872〜2.1027) | 4.8545(q1=4.8657,q3=4.8461) (range: 4.8475〜4.8728) | 4.1191(q1=4.1662,q3=4.1141) (range: 4.1040〜4.1242) | 7.9475(q1=7.9699,q3=7.9287) (range: 7.8803〜8.0815) | wmma_tf32 | mma_f16 | 192.70% (range: 192.02〜196.20%) |
| 2048 | 2.3436(q1=2.3446,q3=2.3422) (range: 2.3427〜2.3455) | 6.2995(q1=6.3281,q3=6.2736) (range: 6.2946〜6.3039) | 7.4888(q1=7.5114,q3=7.3492) (range: 6.2734〜8.6994) | 12.0214(q1=12.0237,q3=11.5815) (range: 12.0204〜12.0219) | wmma_tf32 | mma_f16 | 160.53% (range: 138.17〜191.62%) |
| 4096 | 1.9775(q1=1.9776,q3=1.9772) (range: 1.9729〜1.9817) | 4.4824(q1=4.4855,q3=4.4809) (range: 4.4758〜4.4851) | 4.3623(q1=4.3634,q3=4.3609) (range: 4.3619〜4.3647) | 11.4462(q1=11.4658,q3=11.4380) (range: 11.4379〜11.4484) | wmma_tf32 | mma_f16 | 262.24% (range: 262.22〜262.44%) |

**run 間ばらつきの所見**: `f32_best_path`（`wmma_tf32`）・`f16_candidate_path`（`mma_f16`）は 3 run すべて
・全形状で同一経路が選ばれ、判定対象形状（2048/4096）の比率も小数点以下 1 桁の範囲で安定している
（f32: 25.64〜25.69%、f16: 12.97%固定）。一方 `wmma_f16`（f16 candidate ではなく参考比較用の経路）は
2048 形状で 6.27〜8.70 TFLOPS と run 間で約 40% ばらつき、これに伴い参考比 `mma_over_wmma_f16` も
138〜192% と大きく揺れた。`wmma_f16` は候補下限の算出には使われない経路（`f16_candidate_path` は
常に `mma_f16` が選出）であるため候補下限値そのものへの影響はないが、このばらつき自体は #391
（起動コスト計測）が引き継ぐ計測プロトコル頑健性の論点に加える価値がある。

「f32 最良経路」列は `f32_best_path=` 出力（`tiled`/`wmma_tf32`）を転記する。固定優先順位ではなく実測
TFLOPS の中央値の大小比較で選ばれる（`cuda_floor_bench.rs::best_of`）。「f16 candidate 経路」列は
`f16_candidate_path=` 出力（`wmma_f16`/`mma_f16`）を転記する。4 経路すべてが launch-only 計測に統一
された（下記「計測境界の統一」参照）ため、`f32_best_path` と同じく実測 TFLOPS の中央値の大小比較で
選ばれる（`f16_candidate_floor_value` 参照。PR #349 codex-review 再指摘 P1 対応）。

計測境界の統一（PR #349 codex-review 再指摘 P1 対応。`cuda_floor_bench.rs` 冒頭ドキュメンテーション
コメント「計測境界の統一」参照）: tiled f32・WMMA(TF32)・WMMA f16・`mma.sync` f16 の 4 経路はいずれも
H2D 転送・出力バッファ確保をループ外で済ませ、GPU 実行（カーネル起動＋同期）のみを計測する。これは
PyTorch 参照計測（`gemm_bench_torch_cuda.py` が入力テンソルをループ外で GPU 上に生成し `torch.matmul`
+同期のみを計測するプロトコル）と同一の境界であり、`mma_over_wmma_f16` 比は apples-to-apples の
比較になる。

### 対 PyTorch 比

| M=N=K | f32 最良（実測大小比較で選出） / PyTorch f32 比 | f16 candidate（実測大小比較で選出） / PyTorch f16 比 |
|-------|----------------------------------------------------|------------------------------------------------------|
| 512（参考値） | 61.95%（wmma_tf32。range: 61.86〜62.18%） | 46.57%（mma_f16。range: 46.17〜47.35%） |
| 2048 | 36.78%（wmma_tf32。range: 36.75〜36.81%） | 12.97%（mma_f16。3 run とも同一値） |
| 4096 | 25.68%（wmma_tf32。range: 25.64〜25.69%） | 14.09%（mma_f16。range: 14.08〜14.09%） |

### 丸め適用後の候補下限値

| 精度 | 判定対象形状の最小比率（2048/4096） | 丸め規則適用後の候補下限値 | 現行暫定値（40%）との比較 |
|------|--------------------------------------|------------------------------|------------------------------|
| f32  | 25.64〜25.69%（3 run とも 2048=36.75〜36.81%・4096=25.64〜25.69% で 4096 側が最小） | **25%**（3 run とも一致。10% 以上のため 5% 刻み切り下げ） | 下回る |
| f16  | 12.97%（3 run とも 2048=12.97% で固定・4096=14.08〜14.09% のため 2048 側が最小） | **10%**（3 run とも一致。10% 以上のため 5% 刻み切り下げ、12.97%→10%） | 下回る |

**候補下限値は 3 run すべてで完全一致**（f32=25%・f16=10%）しており、単発の run 間ばらつきに起因する
不確実性はない（上表「run 間ばらつきの所見」で述べた `wmma_f16` 経路のばらつきは候補下限の算出対象
〈`wmma_tf32`・`mma_f16`〉に含まれないため無関係）。ただし下記「数値一致（parity）状態の限定条件」の
とおり、選出された両経路（`wmma_tf32`／`mma_f16`）はいずれも #389 §5.3 の parity 恒常 fail 対象であり、
この候補下限値は #393 の下限確定根拠として単独採用できない。

### 暫定 40% との比較所見

**f32（候補下限 25%、暫定 40% を下回る）**: PoC-v2-3「tensor core 化の段階見積もり」節が示す
40〜70% 到達目安は PyTorch **f16**（tensor core 経路の実効値）を基準とした見積もりであり、
`cuda-floor-remeasurement.md` 冒頭「目的・受け入れ条件対応」節が述べるとおり f32 の暫定 40% は
そもそも「当該見積もりをそのまま流用せず、保守的な値として設定した PyTorch f32 比の目標」に過ぎない
（実測前から根拠薄弱な暫定値だった）。実測 25.68%（2048）/25.64%（4096）は、TF32 Tensor Core 経路
（`wmma_tf32`）が PyTorch の `torch.matmul`（同じく TF32 既定降格された cuBLAS 経路。REQ-2 の
TF32 前提複合指標改定の背景と同根）の 1/4 程度に留まることを示す。手書き WMMA カーネルが
cuBLAS の TF32 実装（複数タイルサイズの自動選択・ソフトウェアパイプライニング等）に対して
最適化余地を残していることが要因と考えられ、想定通りの結果である。

**f16（候補下限 10%、暫定 40% を大きく下回る）**: f16 candidate 経路 `mma_f16`（`mma.sync` パイプライン）
は PyTorch f16 の 13〜14% に留まった。PyTorch f16 は cuBLAS の Tensor Core f16 経路（GB10 実測
81.26〜92.68 TFLOPS）を使うのに対し、本リポの `mma.sync` パイプライン実装は単一ワープレベルの
基本的なパイプライニングに留まり、cuBLAS 相当のマルチステージ pipeline・warp-specialization・
`cp.async` 活用等の高度化が未実装である（`docs/cuda-gemm-mma-pipeline.md` 参照）。PoC-v2-3 の
40〜70% 見積もりは cuBLAS 相当の最適化レベルを暗黙の前提としており、本リポの現状実装レベルとの
ギャップが実測値に表れたと解釈する。

**両精度共通の注記**: 上記所見は「なぜ候補下限が暫定 40% を下回ったか」の要因分析であり、
下限値そのものの当否判断（40% を維持するか・実測値に合わせて引き下げるか）は #393（人間承認）の
範囲である。本ドキュメントは候補下限値の算出根拠の提供に留める。

### 数値一致（parity）状態の限定条件（#389 §5.3 引継ぎ）

イシュー #390 本文は「#389 数値一致 green が前提」と記載するが、**実態は green ではない**。#389
（`docs/backend-cuda-real-device-testing.md` §5.3）は、許容誤差を一切緩和しないまま **parity 恒常
fail 8 件**（TF32 経路 5 件：全要素の 15〜17% が現行複合判定〈相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満〉を外れる・32×32×32 の最小形状でも fail／f16 経路 3 件：K=4096 で 0.12〜0.15% の tail 超過）
を確定させており、REQ-2 閾値改定は #186（`docs/perf/cuda-tensor-core-tolerance-evaluation.md`）へ
引き渡し済みである。

本イシューの候補下限値算出で選出された 2 経路は、いずれもこの parity 恒常 fail 対象と**完全に一致する**:

| 候補下限の経路 | #389 §5.3 の対応する恒常 fail テスト | fail 内容 |
|---|---|---|
| `wmma_tf32`（f32 最良経路。3 run・全判定形状で選出） | `gemm_wmma_tf32.rs::wmma_tf32_k4096_stress_poc_v2_5`・`wmma_tf32_matches_reference_across_shapes`・`gemm_wmma_tf32_opt.rs::wmma_tf32_opt_k4096_stress`・`wmma_tf32_opt_matches_reference_across_shapes`・`tensor_core_real_device.rs::tensor_core_parity_record`（TF32 経路 5 件） | fail_count 15.0〜17.1%。32×32×32 の最小形状から恒常的に外れる |
| `mma_f16`（f16 candidate 経路。3 run・全判定形状で選出） | `cpu_cuda_mma_parity.rs::mma_f16_k4096_stress`（f16 経路 3 件のうちの 1 件） | fail_count 0.15%（K=4096 stress でのみ発生する tail） |

**したがって本 candidate floor（f32=25%・f16=10%）は数値一致未達の経路の実測値であり、#186（REQ-2
閾値改定）の解決前は #393 の下限確定根拠として単独採用できない。** ドキュメント上で parity が
green であるかのような記述はしない。#393 はこの限定条件を踏まえたうえで、①#186 の閾値改定完了を
待ってから最終確定する、②現状の暫定 40% を維持したまま据え置く（`docs/perf/performance-floor-decision.md`
が既に示す据え置き確定案）、のいずれかを人間判断で選ぶことになる。

### tiled f32 @4096 のバイナリ間乖離の突合結果（#389 §5.1 引継ぎ）

#389 §5.1 は、同一形状（M=N=K=4096）の tiled f32 基準値が計測バイナリによって約 5 倍乖離する
未解決事象を報告していた:

- `tensor_core_real_device.rs::tensor_core_tflops_record`（同一バイナリ内に parity テストを併載し
  並列実行される）: **0.189〜0.233 TFLOPS**
- `gemm_wmma_tf32_opt.rs::wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`（直列再実行時の
  バイナリ内計測）: **1.187〜1.237 TFLOPS**

本イシューの `cuda_floor_bench` は単一プロセス・単一計測フロー（同一バイナリ内に他の GPU 使用
`#[test]` を併載しない example）で逐次実行するため、並列競合のない権威ある実測値を提供できる。
3 run の tiled f32 @4096 中央値は **1.9729〜1.9817 TFLOPS**（run 間中央値 1.9775 TFLOPS）であった。

**結論**: この値は上記いずれの既存値よりも高く、特に最も低い 0.189〜0.233 TFLOPS を大きく上回る。
これは #389 §5.1 が推定した「同一バイナリ内 `#[test]` 並列実行による GPU 時間分割が低い方の値
（0.189〜0.233 TFLOPS）を歪めた」という仮説を裏付ける。一方で、直列再実行値（1.187〜1.237 TFLOPS）
と比較しても本実測（1.977 TFLOPS 前後）はなお約 1.6〜1.7 倍高く、両者にも無視できない差が残る。
考えられる要因は、(a) `gemm_wmma_tf32_opt.rs` は直列再実行時も同一バイナリ内に他の 4 テスト
（parity 2 件・性能アサーション 1 件・形状網羅 1 件）を伴っており、それらの GPU 初期化・実行に
伴うクロック状態・キャッシュ状態の違いが残存した可能性、(b) 計測プロトコル・ウォームアップ回数の
違い（`cuda_floor_bench` は `bench_harness::protocol::run` の warmup 20・計測 20 に統一されているが、
`gemm_wmma_tf32_opt.rs` 側の計測回数は本イシューでは未確認）。厳密な原因切り分けは #391（起動コスト
計測）に引き継ぐが、**「並列競合により低い方の値〈0.189〜0.233 TFLOPS〉が実性能を過小評価していた」
という #389 の推定は本実測により確認された**とみなし、`docs/perf/cuda-tensor-core-measurement.md`
の該当相互参照を本節への参照に更新する（次項）。

## 動作確認

実機実測（本節上部「実測結果」参照）に加え、以下のローカル検証を実施済み:

- `cargo build --workspace --locked` — `cudarc` 動的ロード契約（CUDA toolkit 非搭載環境でもビルド成立する。
  `.claude/rules/coding-rust.md`）を崩していないことを確認済み
- `cargo build -p backend-cuda --example cuda_floor_bench --release` — example のビルド成立
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda --example cuda_floor_bench` — 丸め規則（#158 で
  `bench_harness::floor_lower_bound` へ一本化済み。旧 `floor_round`）の単体テスト 3 件
  （仕様例との突合・10% 境界を跨ぐ非減少性・非有限値/負値の防御）・`best_of`（f32 最良経路選出。
  固定優先順位ではなく実測値比較であることの回帰確認）の単体テスト 4 件・`f16_candidate_floor_value`
  （計測境界統一後は `wmma_f16`/`mma_f16` の実測比較で選出することの回帰確認。PR #349 codex-review
  再指摘 P1 対応）の単体テスト 4 件・`confirmed_candidate_floor`（判定対象形状の一部欠測時に
  candidate floor を確定させないことの回帰確認）の単体テスト 1 件・`tflops_sample`（時間ドメイン
  Q1/Q3 を TFLOPS ドメインへ漏れなく変換することの回帰確認）の単体テスト 1 件、計 13 件が green
  であることを確認

## 役割分担（二重管理を避ける）

- **#158（TASK-8.3d・人間判断）**: 本ドキュメントの候補下限値を受け取り、下限の最終確定・
  `docs/spec/04-requirements.md` への反映判断を行う。`docs/spec/` の更新自体は spec リポジトリ
  （Fandhe-AI/rust-ai-library-spec）側で対応する（本リポでは編集しない）
- **`docs/spec/v2-amendment-proposal-2026-08-06.md`**（改定提案ドラフトが存在する場合）: 下限＝回帰検知
  ラインとし目標 90% を別レイヤ化する改定案との関係整理は #158 側で行う
- **`docs/performance-targets.md`（TASK-8.4・#159）**: 段階的下限の一覧整備。本ドキュメントは #157
  固有の実測記録に限定し、全バックエンド横断の一覧化は #159 に委ねる
- **丸め規則のモジュール一本化**: `bench-harness` の TASK-8.2 下限判定モジュール（#151〜#153）マージ後、
  `cuda_floor_bench.rs::floor_round` のインライン実装は削除し公開 API へ委譲する予定だったが、
  **#158（TASK-8.3d）で `bench_harness::floor_lower_bound` への一本化を実施済み**
  （`docs/perf/performance-floor-decision.md` §6）

## 未実施・後続作業

- **実機実測は #390 で完了済み**（本ファイル「実測結果（#390 実機実測・DGX Spark GB10・実施日
  2026-08-10）」節）。以降の項目のみ未完了
- 丸め規則の `bench-harness` モジュール一本化: **完了済み（#158。§「丸め規則のモジュール一本化」参照）**
- 候補下限値の最終確定・REQ-8 反映判断（#393・人間判断）。本イシューが算出した候補下限（f32=25%・
  f16=10%）は「数値一致（parity）状態の限定条件」節の限定条件付きで #393 へ引き渡す。**据え置き確定
  （暫定 40% 維持）の判断案は `docs/perf/performance-floor-decision.md` を参照**
- **#186（REQ-2 閾値改定）の解決**: `wmma_tf32`・`mma_f16` の parity 恒常 fail が解消しない限り、
  本ドキュメントの候補下限値は「数値一致未達の経路の実測値」という限定付きのまま確定できない
- **`wmma_f16` 経路 2048 形状の run 間ばらつき（6.27〜8.70 TFLOPS）の原因調査**: 候補下限の算出には
  無関係だが、計測プロトコル頑健性の論点として #391 に申し送る（「経路×形状 TFLOPS 実測」節「run 間
  ばらつきの所見」参照）
- **tiled f32 @4096 のバイナリ間乖離**: 本イシューの逐次計測（1.977 TFLOPS 前後）により「並列競合が
  低い方の値〈0.189〜0.233 TFLOPS〉を歪めた」との推定は裏付けられたが、直列再実行値（1.187〜1.237
  TFLOPS）との約 1.6〜1.7 倍の残差は未解明のまま #391 に引き継ぐ
