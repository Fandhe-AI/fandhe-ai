# CUDA GEMM TF32 生 `mma.sync` 経路 A/B 計測記録（#802）

イシュー #802「test(backend-cuda): TF32 mma.sync 経路の数値一致・parity・実機ベンチ確定」の
実測記録テンプレート・再開手順。`crates/backend-cuda/src/gemm_mma_tf32.rs`（`CudaMmaTf32Gemm`。
イシュー #801）が実装した TF32 生 `mma.sync.aligned.m16n8k8.tf32` 経路について、(1) バックエンド間
数値一致回帰・parity 非後退契約の実機確認、(2) 既存 `wmma_tf32`（WMMA C++ API ベース。staged/opt/
basic の 3 段選択）との A/B 実機ベンチ、(3) 本番結線の採否判断、を記録する。

## 1. 位置づけ・前提

- `CudaMmaTf32Gemm`（#801。PR #823 でコミット `09f9f98`）は **本番非結線**の直接指定 API であり、
  `ops.rs`／`gemm.rs`／`gemm_auto.rs` のディスパッチからは呼ばれない
  （`docs/cuda-tensor-core-design.md` §15 冒頭「位置づけ」参照。#803 の main 追従マージで
  §14 = warp タイル拡大設計、§15 = 本 TF32 経路へ節番号を振り直し済み）。
- 同梱の実機テスト（`crates/backend-cuda/tests/gemm_mma_tf32.rs`〈`#[ignore]` 4 本〉・
  `tests/mma_tf32_vs_wmma_tf32_staged.rs`〈`#[ignore]` 2 本〉。計 6 本）は #801 実装セッション・
  #802 セッションのいずれも DGX Spark GB10 実機へ到達できず**未実行**のままだったが、
  この実機到達不可の状態は #838 セッションで解消し 6 本とも実行済みである（§2 参照）。
- 本ファイルは #802 の受け入れ条件 3 項（数値一致・parity・実機ベンチ）がいずれも実機実測を要する
  ため、実機到達可能なセッションが引き継いで完了させる前提の記録として作成された（#838 セッション
  で実機到達・実行を完了。結果は §2 以降）。

## 2. 状態: 実機到達成功・数値一致 FAIL（機能欠陥の疑い。2026-08-22・イシュー #838 実装セッション）

**#802 のブロック状態（実機到達不可）は解消した。** 本セッションは DGX Spark GB10 実機へ到達し
（`docs/real-hardware-verification-env.md` §2/§3 手順・`~/.ssh/config` の spark ノード定義を使用）、
`crates/backend-cuda/tests/gemm_mma_tf32.rs`・`tests/mma_tf32_vs_wmma_tf32_staged.rs` の実機テスト
（`#[ignore]` 計 6 本）を実行できた。しかし **6 本中 4 本が数値一致 FAIL**（内訳は §3）であり、その
FAIL パターンは TF32 精度差ではなく `CudaMmaTf32Gemm`（`crates/backend-cuda/src/gemm_mma_tf32.rs`）
自体の**機能欠陥（correctness bug）**を示す実測結果である。詳細な分析根拠は §3 を参照。

**したがって本イシュー（#838）の受け入れ条件 1・3（数値一致 6 本 pass・parity 初回記録）は
未達成のまま終了する。** カーネル自体を自律修正することは本イシューのスコープ外（#838 は実機計測
イシューであり、`crates/backend-cuda/src/gemm_mma_tf32.rs`・`kernels_mma_tf32.rs` の実装修正は含ま
ない）と判断し、行わなかった。受け入れ条件 2（`cuda_floor_bench` A/B 5 回計測）は実施したが、
launch-only 計測（起動可否・スループットのみで出力値の数値一致は検査しない）のため数値が測れて
いる一方、**カーネルの計算内容自体が誤っているため性能比較の有効な根拠にはならない**（§4 の
無効化注記を参照。#839 の採否判断には使わない）。

実機実測環境:

- GPU: NVIDIA GB10 / compute_capability = (12, 1)（`nvidia-smi --query-gpu=compute_cap` は
  `12.1` と表示）
- driver 版数: 580.159.03
- CUDA toolkit: 13.0.88（`nvcc --version`）
- 実行コミット SHA: `363bcdfe87dbe44f9c97c1ec17503a2527a2de2a`（origin/main HEAD。#838 実装
  ブランチはこの SHA から分岐。転送前後で `.rev-stamp` 照合済み）
- 実測日: 2026-08-22
- OS: Ubuntu 24.04.4 LTS（aarch64）
- 実機は共有ノードであり常駐サービス（`comfyui-env`・`kokoro`）が GPU メモリを小さく占有するが、
  `utilization.gpu` は各計測前後で 0% を確認済み（G4 排他性確認）

**未実施のまま残る事項**（本イシューのスコープ外・後続へ引き継ぎ）:

- `CudaMmaTf32Gemm` の機能欠陥そのものの原因調査・修正（別イシューとして起票が必要。§3 参照）
- 修正後の再実行による数値一致回帰・parity 非後退契約の完了（§3）
- （採用時のみ）本番結線。採否判断自体は #839 で確定済み（不採用・凍結。§5.1）
- `docs/perf/cuda-parity-baseline.md` への実測値追記（**意図的に行わなかった**。理由は §3 末尾）

## 3. 再開手順（実機到達可能セッション向け）: 数値一致・parity 非後退の実機実行

1. `docs/real-hardware-verification-env.md` §2/§3 に従いコード転送・PATH 設定を行う。
2. 以下を実行し、実行ログを本節へ追記する:

   ```sh
   # rust libtest の位置引数 FILTER は 1 個のみ受理する（2 個目以降は
   # unexpected argument になり実行不能）ため、`--test <file>` でテスト
   # バイナリを限定したうえで 1 呼び出し 1 FILTER に分割する。

   # crates/backend-cuda/tests/gemm_mma_tf32.rs（#[ignore] 4 本）
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     mma_tf32_matches_reference_across_shapes
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     mma_tf32_k4096_stress
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     mma_tf32_zero_dim_shape_returns_empty_without_launch
   cargo test -p backend-cuda --release --test gemm_mma_tf32 -- --ignored --nocapture \
     launch_tf32_zero_dim_shape_is_noop_or_zero_fills_without_launch

   # crates/backend-cuda/tests/mma_tf32_vs_wmma_tf32_staged.rs（#[ignore] 2 本）
   cargo test -p backend-cuda --release --test mma_tf32_vs_wmma_tf32_staged -- --ignored --nocapture \
     mma_tf32_matches_wmma_tf32_staged_across_shapes
   cargo test -p backend-cuda --release --test mma_tf32_vs_wmma_tf32_staged -- --ignored --nocapture \
     mma_tf32_matches_wmma_tf32_staged_k4096_stress

   cargo test -p backend-cuda --release --test parity_nonregression -- --ignored --nocapture \
     parity_baselines_do_not_regress
   ```

   （テスト名は `cargo test -p backend-cuda --test <file> -- --list` で実測確認済み
   〔2026-08-21〕。`--ignored` 実機テストは `parity_nonregression.rs` 内では
   `parity_baselines_do_not_regress` の 1 本のみで、ファイル名そのものをフィルタ文字列に
   使うと 0 件マッチの偽 green になるため注意。他 8 本〔`tolerance_constants_are_pinned` 等〕
   は `#[ignore]` なしの通常 CI 対象で GPU 不要）

3. **実測結果（2026-08-22。§2 実機実測環境で実行）**:

   | テスト | 結果 | 詳細 |
   |---|---|---|
   | `gemm_mma_tf32::mma_tf32_matches_reference_across_shapes` | **FAIL** | 先頭ケース `m=16 n=8 k=8`（1 `mma.sync` 呼び出しちょうど）で `fail_count=128/128, max_abs_diff=3.699e0, max_rel_err=1.948e0, mean_abs_diff=1.118e0, mean_rel_err=9.714e-1` |
   | `gemm_mma_tf32::mma_tf32_k4096_stress` | **FAIL** | `fail_count=16768000/16777216, max_abs_diff=1.148e2, max_rel_err=2.000e0, mean_abs_diff=1.702e1, mean_rel_err=8.028e-1` |
   | `gemm_mma_tf32::mma_tf32_zero_dim_shape_returns_empty_without_launch` | **FAIL** | `run_tf32` が `InvalidShape { detail: "b length mismatch: expected 16 (k*n), actual 0" }` を返し `unwrap()` で panic（テスト側の呼び出し契約と実装の不整合。数値一致とは別種の不具合） |
   | `gemm_mma_tf32::launch_tf32_zero_dim_shape_is_noop_or_zero_fills_without_launch` | pass | ゼロ次元形状で起動せず no-op になる契約は健全 |
   | `mma_tf32_vs_wmma_tf32_staged::mma_tf32_matches_wmma_tf32_staged_across_shapes` | **FAIL** | 先頭ケース `m=64 n=64 k=64`（ブロックタイルちょうど・端数なし）で `fail_count=4092/4096, max_abs_diff=9.464e0, max_rel_err=1.997e0, mean_abs_diff=2.161e0, mean_rel_err=8.057e-1` |
   | `mma_tf32_vs_wmma_tf32_staged::mma_tf32_matches_wmma_tf32_staged_k4096_stress` | **FAIL** | `fail_count=16767942/16777216, max_abs_diff=1.188e2, max_rel_err=2.000e0, mean_abs_diff=1.703e1, mean_rel_err=8.031e-1` |
   | `parity_nonregression::parity_baselines_do_not_regress` | pass | `mma_tf32` は未登録経路のため対象外。既存ベースライン（`wmma_tf32`・`wmma_tf32_opt`・`mma_f16` 系）は非後退のまま |

   合計: 6 本中 4 本 FAIL・1 本 pass（zero-dim no-op）・1 本 FAIL（zero-dim の別テストは実装契約の
   不整合で panic）。加えて `parity_nonregression` は pass（既存経路への影響なし）。

4. **本ファイル §3 既知リスク節に記載していた手順（CPU 参照恒常 fail → parity 非後退契約への移行・
   `ParityPath::MmaTf32` 初回記録）は適用しない。** 理由:

   - 上記 FAIL は「TF32 精度差」で説明できる範囲を大幅に超えている。TF32（10 bit 仮数）由来の丸め
     誤差は相対誤差でおよそ 1e-3 のオーダーに収まるはずだが、実測は `mean_rel_err` が 0.80〜0.97
     （すなわち平均で真値の約 80〜97% もずれている）。
   - **GPU-GPU 相互一致（`mma_tf32` vs `wmma_tf32` staged）も同様に大きく FAIL** している
     （`m=64 n=64 k=64` で `fail_count=4092/4096`）。この形状はブロックタイルぴったりで端数処理・
     `cp.async` アラインメント境界のいずれも踏まない、最も単純なケースである。2 つの独立した TF32
     実装が同じ入力に対して 80% 近い相対誤差で食い違うのは、丸め方針の差では説明できない。
   - `m=16 n=8 k=8`（タイリング・複数ワープ蓄積を経由しない、`mma.sync` 命令 1 回ちょうどの最小形状）
     単体でも 128 要素中 128 要素すべてが FAIL している。これはタイル境界・累積順序の問題ではなく、
     単一命令の出力そのもの（フラグメント→レジスタのレイアウト・`ldmatrix` のアドレッシング・
     アキュムレータのマッピング等）に誤りがある可能性を強く示唆する。
   - 以上から、この FAIL は「TF32 系経路の既知精度差」（`docs/perf/cuda-parity-baseline.md` §1 が
     扱う wmma_tf32 系の 15〜16% fail 比率）とは性質が異なる**機能欠陥（correctness bug）**と判断
     した。機能欠陥の実測値を「許容される精度差」として `ParityPath::MmaTf32` 初回記録に固定して
     しまうと、非後退契約（`docs/perf/cuda-parity-baseline.md` §1「承認記録」）の趣旨（既知精度差の
     上限を追跡する）を破り、壊れたカーネルの誤差を「この程度のズレは仕様」として恒久的に green 化
     してしまう（`.claude/rules/security.md` A08 ソフトウェア・データ整合性が禁じる整合性の弱体化に
     該当する）。したがって **`crates/backend-cuda/tests/common/parity_baseline.rs` への
     `ParityPath::MmaTf32` 追加、`tests/parity_nonregression.rs` への `check_mma_tf32_baseline`
     追加、`docs/perf/cuda-parity-baseline.md` §3 表への追記のいずれも行わない**（コード変更なし）。
   - カーネル自体（`crates/backend-cuda/src/gemm_mma_tf32.rs`・`kernels_mma_tf32.rs`）の原因調査・
     修正は本イシュー（#838・実機計測イシュー）のスコープ外と判断し着手しなかった。原因調査には
     PTX/SASS レベルの `mma.sync`／`ldmatrix` レイアウト検証が必要になる見込みで、実機計測とは別
     イシューとして起票すべき事項である（§6「スコープ外」参照）。

## 4. `cuda_floor_bench` 実機ベンチ・A/B 記録（2026-08-22・イシュー #838 実装セッション）

**重要な無効化注記**: `cuda_floor_bench` は launch-only 計測（起動可否とスループットのみを測り、
出力値の数値一致は検査しない）であるため実行自体は成功し、以下に生値・中央値を記録する。
しかし §2・§3 のとおり `mma_tf32` カーネルは数値的に誤った出力を返している（機能欠陥の疑い）。
**誤ったメモリアクセスパターン・演算内容で測られた性能は、修正後の正しいカーネルの性能を
代表しない可能性がある。** したがって以下の `mma_tf32` 系数値は**イシュー #839 の採否判断には
使用不可**として扱うこと（`wmma_tf32` 側の数値は既存の妥当な経路であり参考として有効）。

```sh
cargo run -p backend-cuda --example cuda_floor_bench --release
```

- サイズ 512／1024／2048／4096 の `wmma_tf32_tflops`・`mma_tf32_tflops`・
  `mma_tf32_over_wmma_tf32(...)` 行を **5 回起動**して記録し、各サイズの中央値を採る
  （`CLAUDE.md`「5 回計測中央値」規約。`docs/perf/cuda-gemm-swizzle-ab.md` の運用と同型）。
- 生ログ・5 回分の値・中央値・比率を下表へ転記する。
- `mma_tf32_over_wmma_tf32` は `wmma_tf32` が **staged 経路**へ実際にルーティングされた形状
  でのみ算出される（`gemm.rs::CudaGemm::wmma_tf32_routed_path_is_staged` で判定。staged
  カーネルが未コンパイル・未整列形状で opt／basic へフォールバックした場合は `n/a` になる。
  codex-review 指摘対応。PR #826）。該当実機で `n/a` が出力された場合、staged 経路が不能な
  環境（`docs/perf/cuda-gemm-mma-tf32-ab.md` 実機の cc・cp.async 対応状況を確認）である旨を
  本節へ追記し、§5 の採否判断には使わない。

### 4.1 実測記録（5 回計測・中央値。**mma_tf32 列は §4 冒頭の無効化注記により参考値**）

計測環境: GPU 名 = NVIDIA GB10・compute_capability = (12, 1)・driver 版数 = 580.159.03・
CUDA toolkit = 13.0.88・実行コミット SHA = `363bcdfe87dbe44f9c97c1ec17503a2527a2de2a`・
実測日 = 2026-08-22

| size | wmma_tf32 中央値 (TFLOPS) | mma_tf32 中央値 (TFLOPS)（数値不正のため参考値） | mma_tf32/wmma_tf32 比（同上） |
|---|---|---|---|
| 512  | 6.6078 | 8.3549 | 126.16% |
| 1024 | 10.5832 | 16.1086 | 151.58% |
| 2048 | 11.9983 | 19.0031 | 158.48% |
| 4096 | 8.9817 | 11.4197 | 127.08% |

すべてのサイズで `wmma_tf32` は staged 経路へルーティングされ `n/a` は発生しなかった
（`f32 optimized kernel: WMMA(TF32) staged AVAILABLE` を全 5 回で確認）。

**この表を #839 の採否判断に使わない理由**: `mma_tf32` は §2・§3 のとおり数値的に誤った出力を
返す機能欠陥状態にあるため、上表の `mma_tf32` 列・比率列は「壊れたカーネルの起動オーバーヘッド・
メモリアクセスパターンを測った参考値」に過ぎず、正しく修正された `mma_tf32` カーネルの実性能を
表さない。

### 4.2 生ログ（5 回分。`cuda_floor_bench` 標準出力の該当行を転記）

```
run1: size=512 tiled_f32_tflops=2.1013(q1=2.1037,q3=2.1005) wmma_tf32_tflops=6.6261(q1=6.6444,q3=6.6052) wmma_f16_tflops=4.1292(q1=4.1333,q3=4.1211) mma_f16_tflops=17.0663(q1=17.1196,q3=17.0154) mma_tf32_tflops=8.3594(q1=8.3760,q3=8.3218)
run1: size=1024 tiled_f32_tflops=2.3836(q1=2.3842,q3=2.3824) wmma_tf32_tflops=10.6682(q1=10.6954,q3=10.6403) wmma_f16_tflops=8.8785(q1=8.9033,q3=8.8358) mma_f16_tflops=37.6798(q1=38.5572,q3=37.2717) mma_tf32_tflops=16.1338(q1=16.2137,q3=16.0893)
run1: size=2048 tiled_f32_tflops=2.3441(q1=2.3451,q3=2.3392) wmma_tf32_tflops=12.0276(q1=12.1124,q3=12.0087) wmma_f16_tflops=7.3202(q1=7.3249,q3=7.2362) mma_f16_tflops=52.0724(q1=52.1915,q3=52.0118) mma_tf32_tflops=19.0852(q1=19.1314,q3=19.0085)
run1: size=4096 tiled_f32_tflops=1.9725(q1=1.9737,q3=1.9686) wmma_tf32_tflops=8.9690(q1=8.9742,q3=8.9597) wmma_f16_tflops=4.3633(q1=4.3639,q3=4.3617) mma_f16_tflops=55.8225(q1=55.8505,q3=53.8606) mma_tf32_tflops=11.3977(q1=11.7948,q3=11.1403)

run2: size=512 tiled_f32_tflops=2.0896(q1=2.0911,q3=2.0888) wmma_tf32_tflops=6.6156(q1=6.6259,q3=6.6078) wmma_f16_tflops=4.1080(q1=4.1130,q3=4.1050) mma_f16_tflops=17.0847(q1=17.1196,q3=17.0500) mma_tf32_tflops=8.3218(q1=8.3510,q3=8.3014)
run2: size=1024 tiled_f32_tflops=2.3835(q1=2.3844,q3=2.3829) wmma_tf32_tflops=10.6648(q1=10.6801,q3=10.6293) wmma_f16_tflops=8.8452(q1=8.8487,q3=8.7907) mma_f16_tflops=38.5572(q1=38.9934,q3=38.3808) mma_tf32_tflops=16.1086(q1=16.1648,q3=16.0201)
run2: size=2048 tiled_f32_tflops=2.3438(q1=2.3445,q3=2.3421) wmma_tf32_tflops=11.9983(q1=12.0102,q3=11.1509) wmma_f16_tflops=7.3512(q1=7.3592,q3=7.3310) mma_f16_tflops=52.1761(q1=52.2904,q3=51.9489) mma_tf32_tflops=19.0031(q1=19.0408,q3=18.9591)
run2: size=4096 tiled_f32_tflops=1.9744(q1=1.9747,q3=1.9736) wmma_tf32_tflops=8.9905(q1=9.0027,q3=8.9836) wmma_f16_tflops=4.3532(q1=4.3541,q3=4.3520) mma_f16_tflops=55.4406(q1=55.5267,q3=53.4618) mma_tf32_tflops=11.4197(q1=11.7037,q3=10.7246)

run3: size=512 tiled_f32_tflops=2.0888(q1=2.0901,q3=2.0880) wmma_tf32_tflops=6.6026(q1=6.6550,q3=6.5895) wmma_f16_tflops=4.1100(q1=4.1181,q3=4.1080) mma_f16_tflops=17.4218(q1=17.5128,q3=17.2605) mma_tf32_tflops=8.3176(q1=8.3469,q3=8.2891)
run3: size=1024 tiled_f32_tflops=2.3813(q1=2.3831,q3=2.3678) wmma_tf32_tflops=10.5816(q1=10.5974,q3=10.5657) wmma_f16_tflops=8.7994(q1=8.8551,q3=8.7919) mma_f16_tflops=38.4241(q1=38.5455,q3=38.2605) mma_tf32_tflops=16.0393(q1=16.1027,q3=16.0049)
run3: size=2048 tiled_f32_tflops=2.3414(q1=2.3439,q3=2.3400) wmma_tf32_tflops=11.9553(q1=12.0065,q3=11.8094) wmma_f16_tflops=9.5420(q1=9.5623,q3=9.5257) mma_f16_tflops=51.9893(q1=52.1330,q3=51.9037) mma_tf32_tflops=18.9468(q1=18.9756,q3=18.8868)
run3: size=4096 tiled_f32_tflops=1.9713(q1=1.9716,q3=1.9707) wmma_tf32_tflops=8.9447(q1=8.9584,q3=8.9399) wmma_f16_tflops=4.3527(q1=4.3539,q3=4.3518) mma_f16_tflops=55.2047(q1=55.2844,q3=55.1332) mma_tf32_tflops=11.6867(q1=11.9045,q3=11.3464)

run4: size=512 tiled_f32_tflops=2.0898(q1=2.0914,q3=2.0885) wmma_tf32_tflops=6.6078(q1=6.6182,q3=6.5870) wmma_f16_tflops=4.1080(q1=4.1111,q3=4.1060) mma_f16_tflops=17.3307(q1=17.4038,q3=17.2605) mma_tf32_tflops=8.3549(q1=8.5511,q3=8.3179)
run4: size=1024 tiled_f32_tflops=2.3662(q1=2.3670,q3=2.3644) wmma_tf32_tflops=10.5616(q1=10.5941,q3=10.4939) wmma_f16_tflops=8.7523(q1=8.8178,q3=8.0766) mma_f16_tflops=38.3370(q1=38.6238,q3=38.1945) mma_tf32_tflops=16.1260(q1=16.2352,q3=16.0873)
run4: size=2048 tiled_f32_tflops=2.3418(q1=2.3431,q3=2.3397) wmma_tf32_tflops=12.0170(q1=12.0274,q3=12.0037) wmma_f16_tflops=6.9672(q1=6.9831,q3=6.9488) mma_f16_tflops=51.7812(q1=51.8937,q3=51.6515) mma_tf32_tflops=19.0358(q1=19.0753,q3=18.9618)
run4: size=4096 tiled_f32_tflops=1.9651(q1=1.9658,q3=1.9641) wmma_tf32_tflops=8.9817(q1=8.9871,q3=8.9669) wmma_f16_tflops=4.3522(q1=4.3533,q3=4.3511) mma_f16_tflops=55.1789(q1=55.3086,q3=53.7347) mma_tf32_tflops=11.1614(q1=11.8156,q3=10.7417)

run5: size=512 tiled_f32_tflops=2.0880(q1=2.0888,q3=2.0872) wmma_tf32_tflops=6.5741(q1=6.5974,q3=6.5714) wmma_f16_tflops=4.1029(q1=4.1100,q3=4.0990) mma_f16_tflops=17.1898(q1=17.3318,q3=17.1185) mma_tf32_tflops=8.3633(q1=8.4991,q3=8.2850)
run5: size=1024 tiled_f32_tflops=2.3652(q1=2.3662,q3=2.3639) wmma_tf32_tflops=10.5832(q1=10.6092,q3=10.5524) wmma_f16_tflops=8.7222(q1=8.8370,q3=8.2488) mma_f16_tflops=38.5019(q1=38.6238,q3=38.2823) mma_tf32_tflops=16.1028(q1=16.1921,q3=16.0240)
run5: size=2048 tiled_f32_tflops=2.3383(q1=2.3413,q3=2.2908) wmma_tf32_tflops=11.8844(q1=11.9324,q3=11.3703) wmma_f16_tflops=7.7412(q1=7.7555,q3=7.7223) mma_f16_tflops=51.7985(q1=51.9213,q3=51.6019) mma_tf32_tflops=18.9782(q1=19.0233,q3=18.9267)
run5: size=4096 tiled_f32_tflops=1.9665(q1=1.9675,q3=1.9661) wmma_tf32_tflops=8.9925(q1=8.9989,q3=8.9711) wmma_f16_tflops=4.3504(q1=4.3513,q3=4.3496) mma_f16_tflops=55.3075(q1=55.3585,q3=53.7976) mma_tf32_tflops=11.8700(q1=12.1157,q3=11.7019)
```

## 5. 性能ベースの採用条件・#838 時点の状態（判定不能。イシュー #838 実装セッション時点の記録）

本節は #838 実装セッション（2026-08-22）時点の記録であり、性能面の採用条件と、その時点で
判定不能だった状態を**当時の記述のまま**残す。#839 セッションでの現在の判断は §5.1 を正とする
（§5 の「判定不能」は性能ベースの採用条件に対する判定であり、§5.1 の不採用確定は correctness
bug を理由とする安全側の凍結判断であって、判断の種類が異なる。両者の関係は §5.1 冒頭の注記を
参照）。

**採用条件（性能面。#838 時点）**: 判定対象形状（2048・4096。REQ-8 の演算律速域）で
`mma_tf32` が `wmma_tf32`（staged）を上回り、かつ 512・1024 で劣化 5% 超がないこと。満たさ
なければ**結線しない**（現状維持を採否判断として記録し、部分改善のみの場合はサイズ条件付き
適用〈swizzle 前例。`docs/perf/cuda-gemm-swizzle-ab.md` §2〉の検討をフォローアップ Issue と
して提案するに留める）。

**#838 時点の判断**: 判定不能（前提となる数値一致が未成立。2026-08-22・イシュー #838 実装
セッション）。

§2・§3 のとおり `CudaMmaTf32Gemm` は数値的に誤った出力を返す機能欠陥状態にあり、§4 の A/B
ベンチ数値も参考値に留まる（性能ベースの採用条件判定には使用不可）。この性能ベースの採用
条件判定はカーネルの機能欠陥が修正され数値一致（6 本 pass）が確認された後にのみ着手できる。
#838 セッションでは性能ベースの採用条件判定そのものを行っていない（充足・不充足いずれの
判定もできない状態）。

### 5.1 #839 採否判断確定: 不採用（凍結。2026-08-22）

イシュー #839「mma_tf32 の採否判断と本番ディスパッチ結線」における「採否判断」は、§5 の
性能ベースの採用条件判定（判定不能のまま）とは別の、**安全側の凍結判断**として行う。両者を
明確に区別する:

- §5 の性能ベースの採用条件判定（`wmma_tf32` staged との性能比較）は、correctness bug が
  未解消のため引き続き判定不能である（§5 の前提は変わらず未成立のまま）。
- 一方、数値的に誤った出力を返す経路を本番結線しないという判断（凍結）は、性能比較の前提
  成立を待たずに、correctness bug の実測（§2〜§4）だけを根拠として確定できる。これは
  「採用するに値するか」を判定する性能ベースの採否判断ではなく、「誤った出力を返す経路を
  結線してはならない」という安全側の契約（REQ-2・`security.md` A08）を実測に基づき適用した
  結果であり、性能条件の充足・不充足の判定を必要としない。

以上の整理に基づき、#839 の採否判断を**不採用（凍結）**として確定する。根拠はすべて §2〜§4
の実測値であり、新規の数値計測は行っていない:

1. 数値一致 6 本中 4 本 FAIL（§3 実測結果表）。最小形状 `m=16 n=8 k=8`（`mma.sync` 1 命令
   ちょうど）で `fail_count=128/128, mean_rel_err≈9.714e-1`。TF32 精度差（相対誤差 1e-3
   オーダー）では説明不能な機能欠陥である。
2. GPU-GPU 相互一致（vs `wmma_tf32` staged）も FAIL（`m=64 n=64 k=64` で
   `fail_count=4092/4096`）。端数・アラインメント境界を踏まない最も単純な形状での不一致であり、
   フラグメント／`ldmatrix` レイアウト等の correctness bug を示す（§3 の分析）。
3. §4 の A/B ベンチ値（512〜4096 で対 `wmma_tf32` 比 126〜158%）は §4 冒頭の無効化注記のとおり
   **採否判断に使用不可**。誤った演算内容で測られた launch-only 計測値は、修正後の実性能を
   代表しない。
4. 本節冒頭の採用条件（2048・4096 で `wmma_tf32`〈staged〉を上回り、512・1024 で劣化 5% 超
   なし）は、前提の数値一致が未成立のため充足を判定できない。採用条件を満たせない場合は
   結線しないと定めているとおり、結線は行わない。数値的に誤った出力を返す経路を本番結線する
   ことは、バックエンド間数値一致（`coding-rust.md`「バックエンド構成」節・REQ-2）および
   `.claude/rules/security.md` A08（ソフトウェア・データ整合性）が禁じる整合性の弱体化に
   該当するため、機能欠陥が未解消のまま結線することはできない。

**本番ディスパッチ（`gemm.rs`／`gemm_auto.rs`／`ops.rs`）へのコード変更は行わない。** #839
実装セッションで `git grep -n "MmaTf32" -- crates/backend-cuda/src/gemm.rs
crates/backend-cuda/src/gemm_auto.rs crates/backend-cuda/src/ops.rs` を再実行し、単独経路
（`wmma_tf32` 系と紛れない `MmaTf32` 型名）への参照が引き続きゼロであることを確認済み
（非結線＝凍結状態の実測根拠）。

**凍結の再評価条件**（すべて満たされて初めて #839 の採否判断を再度行える）:

- (a) `CudaMmaTf32Gemm`（`crates/backend-cuda/src/gemm_mma_tf32.rs`）・
  `kernels_mma_tf32.rs` の機能欠陥（correctness bug）が修正されること。
- (b) 実機で数値一致 6 本すべて pass し、`crates/backend-cuda/tests/common/parity_baseline.rs`
  への `ParityPath::MmaTf32` 初回登録・parity 非後退検査が成立すること。
- (c) `cuda_floor_bench` の 5 回計測中央値を再計測し、本節冒頭の採用条件（2048・4096 で
  `wmma_tf32` staged を上回り、512・1024 で劣化 5% 超なし）を満たすこと。

再評価は、上記機能欠陥を修正する別イシュー（§6 参照。本セッションでは起票せずユーザー判断に
委ねる）の完了後に行う。

### 5.2 採用時の結線内容（メモ。実施は将来の再評価で採用判断が確定した場合）

`gemm.rs::run_wmma_tf32` の 3 段選択の最優先段として `mma_tf32` を追加する場合は、PR #678 の教訓に
従い以下を守る:

1. 実効ルーティング経路の parity 検査を新経路向けに追加する。
2. 既存 `wmma_tf32_staged` ベースライン行の検査が黙って経路すり替えにならないよう直接起動経由の
   検査へ移設する。
3. 結線後に実機で数値一致・非後退・`cuda_floor_bench` を再実行して確認する。

## 6. スコープ外（追跡）

- **`CudaMmaTf32Gemm` の機能欠陥（correctness bug）の原因調査・修正（新規。イシュー #838 実装
  セッションで発見。§2・§3 参照）。TF32 タイル定数拡大（#806）はこの修正完了が前提条件となるため
  実質的にブロックされている。ユーザー承認を得たうえで別イシューとして起票する必要がある
  （`.claude/rules/out-of-scope-tracking.md` に従い、本セッションでは起票していない）**
- **#839 の採否判断は完了済み（不採用・凍結。§5.1）。再評価は上記機能欠陥修正イシューの完了後**
- TF32 タイル定数拡大（Phase 4・#806。診断機構・机上候補表は整備済み。
  `docs/perf/cuda-gemm-mma-tf32-block-tile.md` 参照。上記機能欠陥の修正が前提のためブロック）
- swizzle 変種の TF32 `mma.sync` への適用（実測を伴うため別途。同上の理由でブロック）
- REQ-8 下限値の再確定（候補値の記録まで。確定は人間判断・TASK-8.3 系）
- 部分改善時のサイズ条件付き適用の実装（採否判断で必要と出た場合にフォローアップ Issue を提案）

## 7. #806 との相互参照

イシュー #806（本節見出し §6 の「TF32 タイル定数拡大」）は、当時（#806 起票時点）
本イシュー（#802）が抱えていた実機到達不能セッション制約を引き継ぎ、Step F
フォールバックとして診断機構（`kernels_mma_tf32.rs::
mma_tf32_source_with_block_tile`）・机上候補表・`examples/
mma_tf32_ptx_dump.rs` を整備した（`docs/perf/
cuda-gemm-mma-tf32-block-tile.md`）。この実機到達不能制約自体は #838 セッションで
解消済み（§2）だが、#806 のタイル拡大候補実測は §2 で判明した `CudaMmaTf32Gemm`
自体の機能欠陥（数値一致 FAIL）の修正が前提となるため引き続きブロックされている
（§6 参照）。実機到達可能かつ上記機能欠陥が解消したセッションでは、本
ドキュメント §3・§4（数値一致・parity・`cuda_floor_bench` A/B 計測）と
`cuda-gemm-mma-tf32-block-tile.md` §6・§8（タイル拡大候補の `ptxas -v`
実測・4096/2048 ベンチ）を同一セッションでまとめて消化できる（両者とも
DGX Spark GB10 実機・CUDA 13.0 toolkit を要求する点が共通のため）。
