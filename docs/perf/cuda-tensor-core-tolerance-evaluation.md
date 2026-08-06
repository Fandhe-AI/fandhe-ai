# CUDA Tensor Core（WMMA）経路 数値一致閾値の実測評価（#186・TASK-11.1g）

イシュー #186「test(backend-cuda): TASK-11.1g Tensor Core 経路の数値一致閾値の実測再評価」の実測記録。
REQ-2 受け入れ基準「tensor core（WMMA/mma）化で TF32／f16 累算経路を導入する際は当該経路の数値一致閾値を
実測に基づき再評価する」に対応する。

**本ドキュメントは実測結果と評価結果を記録するのみで、閾値定数
（`backend_cpu::parity::RELATIVE_TOLERANCE`＝1e-3・`ABSOLUTE_RESCUE_THRESHOLD`＝1e-5）は変更していない**
（変更はユーザー承認必須。`.claude/rules/coding-rust.md`・`.claude/rules/security.md` A08）。

## 1. 計測環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GeForce RTX 3060 |
| compute capability | 8.6（Ampere。TF32〈cc 8.0+〉・f16 WMMA〈cc 7.0+〉の両経路の実測要件を満たす） |
| driver | 595.71.05（`CUDA Version: 13.2` 表示） |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| ビルド条件 | `cargo build --release -p backend-cuda --example wmma_tolerance_probe` |
| 計測バイナリ | `crates/backend-cuda/examples/wmma_tolerance_probe.rs`（TASK-11.1g で新規追加） |
| 決定的シード | 形状ごとに 5 シード（1〜5）。`bench_harness::rng::Xorshift64Star` |
| 決定性確認 | 同一シードで 2 回実行し stdout の完全一致（`diff` 差分なし）を確認済み |

### NVRTC プロビジョニング（本リポジトリの CUDA toolkit 非搭載環境での実測手順）

本実行環境には `libcuda`（driver）はあるが `libnvrtc`（NVRTC 実行時コンパイラ）を含む CUDA toolkit が
未導入だったため、以下の手順でプロセス限定のプロビジョニングを行った（システムへの `apt install`・
`/usr/local` への配置は行わない。グローバル状態を汚さない best-effort 手順）。

```bash
pip install --target <SCRATCH>/pylibs --no-deps \
  "nvidia-cuda-nvrtc==13.2.86" "nvidia-cuda-runtime==13.2.86"
# WMMA カーネルが #include する <mma.h> は "crt/mma.h" を再 include するが、
# pip 配布の nvidia-cuda-* wheel には crt/ サブディレクトリが含まれていない
# （ランタイム限定パッケージのため）。apt-get download で crt/mma.h のみを
# 取得し、pip wheel の include/ 配下に重ね合わせる（apt install はしない。
# パッケージを .deb のまま展開するのみでシステムへは導入しない）。
apt-get download cuda-crt-13-1
dpkg -x cuda-crt-13-1_*.deb <SCRATCH>/deb-extract
cp -r <SCRATCH>/deb-extract/usr/local/cuda-13.1/targets/x86_64-linux/include/crt \
  <SCRATCH>/pylibs/nvidia/cu13/include/

LD_LIBRARY_PATH=<SCRATCH>/pylibs/nvidia/cu13/lib \
CUDA_INCLUDE_PATH=<SCRATCH>/pylibs/nvidia/cu13/include \
  cargo build --release -p backend-cuda --example wmma_tolerance_probe
```

**バージョン選定の注記**: driver 595.71.05 は `CUDA Version: 13.2` までの PTX を受理する
（`nvidia-smi` 表示）。当初 `nvidia-cuda-nvrtc==13.3.33`（pypi 最新）で試したところ
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` で失敗したため、driver の対応範囲内である `13.2.86` に
ダウングレードして解決した。`crt/mma.h` は CUDA 13.1 の `cuda-crt` パッケージから取得したが、
`mma.h` の内容は minor version 間でも安定しており、compute_86 向け WMMA コンパイルに問題は
生じなかった（本ドキュメントの全実測がこの構成で成功していることが根拠）。

この手順は #64（TASK-11.1e・GB10 実機）でも再利用可能な NVRTC プロビジョニング手順として記録する。

## 2. 誤差分布実測表

各形状につき 5 シードを計測し、シード間の `fail_count` 合計・`max_abs_diff`／`max_rel_err` の
最大値を集計した（統計手法は `docs/perf/cpu-gemm-rayon-tuning.md` の記録形式に倣う）。
シードごとの生データ（`CompareReport` 全項目）は本ドキュメントに転記した計測コマンドを
再実行すれば同一内容が決定的に再現される。

閾値（変更対象外）: `RELATIVE_TOLERANCE = 1e-3`、`ABSOLUTE_RESCUE_THRESHOLD = 1e-5`。

> **注記（意図的な重複マージ）**: ハーネスの `SHAPES` 定数は 13 形状だが、「256×256×256
> (block tile x8)」と「256×256×256 (K sweep base)」は m=n=k=256 で完全に同一の形状であり
> シード導出式（`m` 由来）により同一入力・同一結果になる（
> `crates/backend-cuda/examples/wmma_tolerance_probe.rs` の `SHAPES` コメント参照）。
> そのため下記 §2.1・§2.2 の実測表はいずれも 12 行とし、重複分は「256×256×256（block tile ×8）」
> の 1 行にマージして記録している（計測が 1 件欠落しているわけではない）。

### 2.1 TF32 経路（`CudaGemm::run_wmma_tf32`）

| 形状 | fail (5 シード合計) / total | max_abs_diff | max_rel_err |
|------|------|------|------|
| 32×32×32（block tile） | 807/5120（15.8%） | 1.857e-3 | 2.556e-1 |
| 64×64×64（block tile ×2） | 3373/20480（16.5%） | 3.377e-3 | 1.046e0 |
| 128×128×128（block tile ×4） | 13140/81920（16.0%） | 4.336e-3 | 1.910e0 |
| 256×256×256（block tile ×8） | 53428/327680（16.3%） | 6.828e-3 | 1.926e0 |
| 512×512×512（block tile ×16） | 212580/1310720（16.2%） | 9.985e-3 | 1.983e0 |
| 17×23×19（非倍数エッジ） | 312/1955（16.0%） | 1.265e-3 | 8.375e-1 |
| 33×31×65（非倍数エッジ） | 827/5115（16.2%） | 2.724e-3 | 5.746e-1 |
| 100×100×100（非倍数エッジ） | 7956/50000（15.9%） | 3.624e-3 | 1.884e0 |
| 130×70×90（非倍数エッジ） | 7280/45500（16.0%） | 4.167e-3 | 1.539e0 |
| 64×96×128（非正方） | 5066/30720（16.5%） | 4.180e-3 | 1.743e0 |
| 256×256×1024（K スイープ） | 53493/327680（16.3%） | 1.296e-2 | 1.890e0 |
| 256×256×4096（K スイープ・PoC-v2-5 stress） | 53281/327680（16.3%） | 2.535e-2 | 1.981e0 |

**要旨**: TF32 経路は最小形状（32×32×32、WMMA タイル 1 個ぶん）を含む**全形状・全シードで
fail 率が約 15〜16.5% に達し、既存 `#[ignore]` テスト（`tests/gemm_wmma_tf32.rs` の
`wmma_tf32_matches_reference_across_shapes`・`wmma_tf32_k4096_stress_poc_v2_5`）が実機で
FAIL することを確認した**（本イシュー着手時の実行結果。下記 §4 参照）。fail 率は K の増加に
対してほぼ横ばい（15.8%〜16.5%）で、ブロックタイル境界・非倍数エッジによる有意差もない
（境界検査の不具合ではなく、TF32 丸め自体に起因する系統的な誤差であることを示唆する）。

### 2.2 f16 WMMA 経路（`CudaWmmaGemm::run_f16`）

| 形状 | fail (5 シード合計) / total | max_abs_diff | max_rel_err |
|------|------|------|------|
| 32×32×32（block tile） | 0/5120（0%） | 3.815e-6 | 8.621e-4 |
| 64×64×64（block tile ×2） | 0/20480（0%） | 3.906e-3 | 2.455e-3 |
| 128×128×128（block tile ×4） | 0/81920（0%） | 7.812e-3 | 1.813e-2 |
| 256×256×256（block tile ×8） | 0/327680（0%） | 7.812e-3 | 3.504e-2 |
| 512×512×512（block tile ×16） | 23/1310720（0.0018%） | 1.562e-2 | 1.303e0 |
| 17×23×19（非倍数エッジ） | 0/1955（0%） | 0.000e0 | 0.000e0 |
| 33×31×65（非倍数エッジ） | 0/5115（0%） | 1.526e-5 | 9.234e-4 |
| 100×100×100（非倍数エッジ） | 0/50000（0%） | 3.906e-3 | 1.899e-3 |
| 130×70×90（非倍数エッジ） | 0/45500（0%） | 3.906e-3 | 1.422e-2 |
| 64×96×128（非正方） | 0/30720（0%） | 3.906e-3 | 2.078e-3 |
| 256×256×1024（K スイープ） | 82/327680（0.025%） | 3.125e-2 | 2.105e-1 |
| 256×256×4096（K スイープ・PoC-v2-5 stress） | 571/327680（0.174%） | 6.250e-2 | 3.387e-1 |

**要旨**: f16 WMMA 経路は K≤256 の全形状で fail 率 0%（複合判定 PASS）。K=512×512×512（K=512）
から fail が現れ始め、K=1024・K=4096（K スイープ）で fail 率が単調増加する
（0% → 0.0018% → 0.025% → 0.174%）。**K 依存で桁落ち蓄積が増える傾向は明確だが、fail 率は
TF32 経路（15〜16%）と比べ 2〜3 桁小さい**。既存 `#[ignore]` テスト
（`tests/cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress`）が実機で FAIL することも確認した
（下記 §4）。

## 3. 閾値評価

- **TF32 経路**: 現行複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を**全形状・全シードで
  満たさない**。fail 率は約 15〜16.5% と一貫しており、統計的なノイズではなく系統的な精度不足である。
  TF32 は f32 仮数部 23bit を 10bit に丸めるため、理論上の相対誤差下限は約 2^-10 ≈ 9.8e-4 であり
  現行閾値（1e-3）と近接するが、実測される `max_rel_err` は最大 1.98（198%）に達している。この
  大きさ自体は仮数部丸め単独では説明しにくいものの、GEMM の出力要素が真値としてゼロ近傍
  （K 項の和が桁落ちでほぼ相殺）になる場合、絶対誤差はほぼ一定（TF32 丸め幅に比例する数 1e-3〜1e-2
  オーダー）でも相対誤差はゼロ除算に近づき際限なく増幅されうるため、TF32 丸め＋桁落ちキャンセレー
  ションの組み合わせで十分説明可能である。この場合 `ABSOLUTE_RESCUE_THRESHOLD`（1e-5）による救済が
  効くべきだが、実測の `max_abs_diff`（1.3e-3〜2.5e-2）は 1e-5 を 2〜3 桁上回っており救済も効いていない。
  **TF32 特有の丸め特性を前提に、絶対誤差救済閾値をより大きく設定する（例: 1e-2 台）等の複合指標の
  改定が必要と考えられる**が、具体的な改定値の決定と REQ-2 改定は本イシューのスコープ外（§5・§6 参照）。
- **f16 WMMA 経路**: K≤256 の範囲では現行閾値内に収まるが、K=512 以降で fail が発生し始め K=4096 で
  fail 率 0.174% まで増加する。fail 率自体は小さいが、既存 `#[ignore]` テスト
  （`wmma_f16_k4096_stress`）がこの領域で実際に FAIL しているため、**「K が大きい場合に現行閾値では
  不十分」という結論は TF32 ほど深刻ではないものの成立する**。K 依存の桁落ち蓄積を許容するなら
  絶対誤差救済閾値の緩やかな引き上げ、または K に応じた許容誤差スケーリングの導入が候補となるが、
  これも本イシューでは変更しない。

## 4. 結論（変更要否）

**変更が必要と判明した**（TF32 経路が全形状で著しく閾値を超過、f16 経路も大 K で閾値を超過）。

- 閾値定数（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・既存 parity テストの許容誤差は
  **本イシューでは一切変更していない**（`.claude/rules/coding-rust.md`・`security.md` A08 に従い、
  ユーザー承認なしの緩和は行わない）。
- 改定が必要な場合の候補（ユーザー承認待ち。具体案の決定はスコープ外）:
  1. TF32 経路専用の複合判定（絶対誤差救済閾値を実測ベースで引き上げる。§3 参照）を REQ-2 に追加する。
  2. f16 WMMA 経路について、K 依存のスケーリングまたは K の実用上限（本ライブラリで許容する最大 K）を
     REQ-2 側で定義し、その範囲内でのみ現行閾値を適用する。
  3. あるいは、TF32/f16 Tensor Core 経路そのものをディスパッチ規則（#66）で高精度要求時に選択しない
     方針とし、経路ごとの精度トレードオフを利用者に明示する。
- 上記いずれも **REQ-2 改定が必要**であり、正本 spec リポジトリ（Fandhe-AI/rust-ai-library-spec）側での
  対応をユーザーに提案する（`docs/spec/` は本リポでは編集しない。`.claude/rules/out-of-scope-tracking.md`）。

## 5. 制約事項

- 本実測は **compute capability 8.6（Ampere、RTX 3060）** によるものである。GB10（sm_121・Blackwell 系譜）
  実機での再確認は #64（TASK-11.1e・open）のスコープであり、Tensor Core の世代差（mantissa 丸め方式・
  累算精度）による差異が出る可能性がある。本ドキュメントの数値をそのまま sm_121 に適用しないこと。
- `mma.sync`/`cp.async` パイプライン導入（#187）後は Tensor Core 経路の実装が変わるため、本実測は
  再評価の対象になる（#187 のスコープ）。
- `tests/cpu_cuda_parity.rs` 冒頭コメントが述べる「naive f16 は複合判定対象外」という既定方針の
  一般化要否について: 本実測は WMMA f16（`CudaWmmaGemm::run_f16`）のみを対象としており、naive f16
  （`CudaGemm::run_naive_f16`）は計測していない。両者はカーネル実装（丸め・累算経路）が異なるため、
  本実測結果を naive f16 に外挿しない。naive f16 の複合判定適用要否は別途実測が必要であり、対象外と
  する（本イシューは WMMA/mma 経路の再評価のみが REQ-2 受け入れ基準の対象）。

## 6. 既存 `#[ignore]` テストの実機実行結果（参考記録）

本イシュー着手時点で `make test-ignored-cuda` 相当（§1 の NVRTC プロビジョニング環境）を実行した結果:

```
$ cargo test -p backend-cuda --test gemm_wmma_tf32 -- --ignored --test-threads=1
test wmma_tf32_zero_k_returns_all_zero ... ok
test wmma_tf32_zero_dim_shape_returns_empty_without_launch ... ok
test wmma_tf32_matches_reference_across_shapes ... FAILED（shape m=32 n=32 k=32:
  複合判定 FAIL fail_count=154/1024, max_abs_diff=1.540e-3, max_rel_err=3.239e-2）
test wmma_tf32_k4096_stress_poc_v2_5 ... FAILED（256x256x4096:
  複合判定 FAIL fail_count=10647/65536, max_abs_diff=2.312e-2, max_rel_err=1.910e0）

$ cargo test -p backend-cuda --test cpu_cuda_wmma_parity -- --ignored --test-threads=1
test wmma_f16_matches_reference_across_shapes ... ok
test wmma_f16_cross_check_against_naive_f16 ... ok
test wmma_f16_k4096_stress ... FAILED（256x256x4096:
  複合判定 FAIL fail_count=122/65536, max_abs_diff=6.250e-2, max_rel_err=3.178e-2）
```

これは §2〜§4 の結論（TF32 は全形状で FAIL、f16 は大 K で FAIL）と整合する。両テストファイルの
冒頭コメントが明記するとおり「実機で複合判定を外れた場合も緩和せず #186 へ引き渡す」方針に従い、
これらの `#[ignore]` テストの許容誤差・判定式は本イシューでも変更していない
（実機依存のため通常 CI では実行されず、`make test-ignored-cuda` 経由でのみ再現する）。
