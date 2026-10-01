# sm_121 可用命令・アーキ固有機能の実機プローブ基盤（#2122）

## 0. 位置づけ・スコープ

- 対応イシュー #2122（親 #2121 Phase 4。「sm_121 の使える命令とアーキ固有機能のプローブ」）。本 doc は **PR-A（プローブ基盤）** の設計・使い方・結果欄を持つ。実装は `crates/backend-cuda/tests/sm121_isa_probe_*`、判定規則（実測前に固定）は `docs/perf/logs/sm121-isa-probe-2122/RULE.txt` を正とする。
- **結果欄はすべて「未実測」**である。GB10（DGX Spark・sm_121）での実測は後続 PR の範囲で、本 doc のどの結果セルにも実測値を書いていない。開発機（RTX 3060・sm_86）のスモーク結果は G0 を不成立にし全セルを判定不能にする設計で、sm_121 の結論としては一切扱わない。
- **AC3（`cp.async.bulk.tensor` の意味論: 要素座標・部分 OOB・swizzle の smem 配置・store・bulk・prefetch・multicast）は PR-B で追加予定・未実装**である。本 PR のレジストリ・RULE・aggregate に TMA のプローブ・条項はない（PR-B が `tma.*` のプローブ行と条項を足せる構造にはしてある）。
- 本 PR はテストとログ規則・ドキュメントのみで、本番経路・tolerance・baseline・閾値・facade 公開面・既存の `tests/tma_probe_real_device.rs`・`tests/setmaxnreg_*` は変更しない。

## 1. 背景（既存の記録との食い違い）

- 既存の TMA プローブ（`tests/tma_probe_real_device.rs`）の「arch ごとのコンパイル成功」は、命令が受理されたことをほぼ示さない。`compile_ptx(src, "compute_121")` は NVRTC が PTX テキストを出すだけで、inline PTX の中身を ptxas が検証するのは `cuModuleLoadData`（ドライバ JIT）の時点だからである。結論は実行段の成功から出ている。本基盤は「NVRTC の受理」と「ptxas の受理」を別の段として記録する。
- `docs/backend-cuda-tma-gemm-load-design.md` §10.8 は「意味論プローブ 3 件 ✓」と書くが、§10.4 の定義（要素座標・部分 OOB・swizzle の smem 配置ダンプ）の 3 件は未実装である（同 §10.8 末尾に訂正の追記を置いた。`cuda-sm121-gemm-candidates-design.md` §2.1 の記述が正しい）。
- 参照ガイド（`.claude/skills/nvidia-cuda/references/blackwell-tuning/streaming-multiprocessor.md`）の値は **CC 12.0**（warps/SM 48・smem 128 KB/SM・99 KB/block・portable cluster 8）で、GB10 は CC 12.1 である。12.0 の記述を 12.1 へ外挿して比較する（R-GUIDE）。スキル本体は編集しない。
- INT8/FP8 の実行・数値は別承認の `docs/int8-quant-grade-up-verification-plan.md`（§3・§8）の別プローブに委ねる。本基盤は受理段（S1〜S3）だけを記録する。

## 2. 段階モデルと判定語彙

1 つの (プローブ, target) は 7 セル（`ctl`・S1〜S6）を必ず持つ。前段が失敗した場合の後段は `not_run`、設計上不実施の段は `not_applicable_by_design` と明示する。

| 段 | 内容 | 実行場所 |
|---|---|---|
| `ctl` | 同一プロセス内で対照 `ctl.copy` を同じ target で S1〜S6 まで通した要約 | 実機 |
| S1 `nvrtc_ptx` | 仮想アーキ（`compute_121`・`compute_121a`・`compute_121f`）の NVRTC。C++ レベルの拒否だけを捕まえる | GPU 不要 |
| S2 `nvrtc_cubin` | 実アーキ（`sm_121`・`sm_121a`・`sm_121f` と、本来の対応アーキ・`sm_90a`）を NVRTC の `--gpu-architecture` に渡し、ptxas をオフラインで通す。ptxas のログ全文を記録 | GPU 不要 |
| S3 `module_load` | S1 の PTX をドライバ JIT でロード。`binary_version`・`ptx_version`・`num_regs` は参考値 | 実機 |
| S4 `launch`／S5 `sync` | 実行時エラー名（`ILLEGAL_INSTRUCTION` 等） | 実機 |
| S6 `verify` | 出力語のビット一致。最初の不一致の添字と不一致語数を記録 | 実機 |

- S2 は S1 と独立の枝で S3 を止めない（S2 拒否でも S3 の実測を取り、食い違いは判定不能にする）。
- 判定の語彙: 成立／NVRTC 拒否／ptxas 拒否（オフライン）／ロード失敗／実行時エラー／結果不一致／受理のみ（実行意味論は未検証）／判定不能（理由コード付き）／未実測。
- 方針（policy）: `verify`（S1〜S6）・`accept_only`（S1〜S3 のみ）・`record_only`（S5 の detail へ値を記録。S6 は不実施）・`attr`（デバイス属性の記録のみ）。
- 本来の対応アーキでの陽性対照（R-HOME）: 各プローブを本来の対応アーキでも S2 に通し、受理された場合に限り sm_121 側の拒否を「アーキ起因」と読む。拒否されたら「判定不能（HOME_REJECTED＝プローブ不良の疑い）」。`home=` の値は記憶に基づく仮説であり、実測の陽性対照で確かめる前提で置いている。
- 想定（`expect=reject121`: `tc5.alloc`・`tc5.ld`・`tc5.cross`）は結果ではなく事前登録した仮説で、想定外の受理は「判定不能（UNEXPECTED_ACCEPT）」とする（黙って成立にしない）。

## 3. 検出範囲（保証しないこと）

- 判定は、`aggregate.py` が読む入力（`compile.log`・`exec/*.log`・`legacy-*.log`・`env_info.txt`）に現れたレコードの完全性と、そこから導く規則に限る。ログが実際にその実行から生成されたことの証明（改竄検知）、ptxas・ドライバ内部の正しさ、sm_86 の結果の sm_121 への外挿の妥当性は保証しない。
- 重複・未知の ID／stage／status・壊れた JSON・NaN・浮動小数点・64 bit を超える整数・キー集合の過不足・連鎖の矛盾・完了記録の欠落・未マスクの `/home/<name>` 形パスは fail-closed（exit 2）。ホスト名の残存は検出しない。測定そのものの打ち切り（外部 timeout＝exit 124）・異常終了は、完全性違反ではなく判定不能（TIMEOUT／PROCESS_FAILED。compile・exec・legacy。dump は欠測として報告）として集計を続け、proc 記録の欠落や exit 0 なのに出力が欠けている場合を完全性違反にする（プロセス種別ごとの表は RULE.txt 13a）。
- 前提ゲート G0（env_info の provenance〈`git_clean=1`〉・正式 target のみ・NVRTC の存在・全 exec の cc が 12.1）が不成立なら全セルを判定不能にする。ctl.copy の成否は G0 に含めず、target ごとの R-CTL が扱う。
- レジストリ・RULE.txt・`aggregate.py` の 3 か所の突き合わせは、`sm121_isa_probe_registry`（CI で走る）が「レジストリ ↔ RULE.txt」を、`python3 aggregate.py --self-test`（CI では走らない）が「RULE.txt ↔ aggregate.py」「`orchestrate.sh --dry-run` の起動一覧 ↔ RULE.txt」を検査する。

## 4. プローブ一覧（`RULE.txt` の `PROBE:` 行の転記。正は RULE.txt）

### AC1 tcgen05 に到達できるか・マクロ（R-TC5）

| プローブ | 条項 | home | policy | layout | expect |
|---|---|---|---|---|---|
| `macro.arch` | R-TC5 | sm_80 | record_only | none | none |
| `tc5.alloc` | R-TC5 | sm_100a | verify | none | reject121 |
| `tc5.ld` | R-TC5 | sm_100a | record_only | none | reject121 |
| `tc5.cross` | R-TC5 | sm_100a | accept_only | none | reject121 |

- `tc5.alloc`: TMEM の割り当て・解放（dealloc の完了を S6 とする）。`tc5.ld`: `tcgen05.ld` の値は記録のみ。`tc5.cross`: `compute_100a` 向け PTX を GB10 でロードし S3 まで見る。
- `macro.arch`: 候補マクロ（`__CUDA_ARCH__`・`__CUDA_ARCH_SPECIFIC__`・`__CUDA_ARCH_FAMILY_SPECIFIC__`・`__CUDA_ARCH_FEAT_SM121_ALL`・`..._SM120_ALL`・`..._SM100_ALL`・`..._SM90_ALL` ほか）の存在・値を記録のみ。マクロ名は事前登録した候補で、存在を主張しない。

### AC2 ガイドとの不一致（R-GUIDE）

| プローブ | 条項 | home | policy | layout | expect |
|---|---|---|---|---|---|
| `attr.limits` | R-GUIDE | sm_80 | attr | none | none |
| `attr.cluster` | R-GUIDE | sm_80 | attr | none | none |
| `attr.misc` | R-GUIDE | sm_80 | attr | none | none |

- 突き合わせ表は `docs/perf/logs/sm121-isa-probe-2122/guide_claims.tsv`（引用元の `path:line`・原文引用・測定項目・比較方法）。引用が実在するかは `aggregate.py --self-test` が引用元の当該行と照合する。判定語彙は「一致（12.0 の記述を外挿）」「不一致」「判定不能」。

### AC4 mma 形状・SIMT 命令（R-MMA）

| プローブ | 条項 | home | policy | layout | expect |
|---|---|---|---|---|---|
| `mma.tf32.m16n8k8` | R-MMA | sm_80 | verify | verified | none |
| `mma.tf32.m16n8k4` | R-MMA | sm_80 | verify | verified | none |
| `mma.f16.m16n8k16.f32` | R-MMA | sm_80 | verify | verified | none |
| `mma.f16.m16n8k8.f32` | R-MMA | sm_80 | verify | verified | none |
| `mma.f16.m16n8k16.f16` | R-MMA | sm_80 | verify | verified | none |
| `mma.bf16.m16n8k16.f32` | R-MMA | sm_80 | verify | verified | none |
| `mma.bf16.m16n8k8.f32` | R-MMA | sm_80 | verify | verified | none |
| `mma.f64.m8n8k4` | R-MMA | sm_80 | verify | verified | none |
| `mma.f16.m8n8k4` | R-MMA | sm_80 | accept_only | none | none |
| `mma.f64.m16n8k4` | R-MMA | sm_90 | accept_only | none | none |
| `mma.f64.m16n8k8` | R-MMA | sm_90 | accept_only | none | none |
| `mma.f64.m16n8k16` | R-MMA | sm_90 | accept_only | none | none |
| `mma.s8.m16n8k32` | R-MMA | sm_80 | accept_only | none | none |
| `mma.e4m3.m16n8k32` | R-MMA | sm_89 | accept_only | none | none |
| `mma.e5m2.m16n8k32` | R-MMA | sm_89 | accept_only | none | none |
| `mma.f8f6f4.m16n8k32` | R-MMA | sm_120a | accept_only | none | none |
| `mma.block_scale.m16n8k64` | R-MMA | sm_120a | accept_only | none | none |
| `mma.ldmatrix.x1` | R-MMA | sm_80 | verify | verified | none |
| `mma.ldmatrix.x2` | R-MMA | sm_80 | verify | verified | none |
| `mma.ldmatrix.x4` | R-MMA | sm_80 | verify | verified | none |
| `mma.ldmatrix.x4_trans` | R-MMA | sm_80 | verify | verified | none |
| `mma.stmatrix.x4` | R-MMA | sm_90 | verify | unverified | none |
| `simt.fma_f32` | R-MMA | sm_80 | verify | none | none |
| `simt.fma_f64` | R-MMA | sm_80 | verify | none | none |
| `simt.fma_f16x2` | R-MMA | sm_80 | verify | none | none |
| `simt.fma_bf16x2` | R-MMA | sm_80 | verify | none | none |
| `simt.f32x2_add` | R-MMA | sm_100 | verify | unverified | none |
| `simt.f32x2_mul` | R-MMA | sm_100 | verify | unverified | none |
| `simt.f32x2_fma` | R-MMA | sm_100 | verify | unverified | none |
| `simt.cvt_tf32` | R-MMA | sm_80 | verify | none | none |
| `simt.elect_sync` | R-MMA | sm_90 | verify | none | none |
| `simt.redux_u32` | R-MMA | sm_80 | verify | none | none |
| `simt.redux_f32` | R-MMA | sm_100a | accept_only | none | none |

- 入力は f16／bf16／tf32 で正確に表せる小さな整数で、f32 累積でも結果が正確なためビット一致で判定する（tolerance は新設・変更しない）。`layout=verified` は開発機（sm_86）の実機で参照モデルを検証済み（検証記録: PR-A〈#2122 の最初の PR〉本文の実行記録〈検証コマンドと出力要約〉。生ログはコミットしない。再現手順は §6.2）。`layout=unverified`（`stmatrix`・packed `f32x2`）は開発機で実行できず、不一致は「判定不能（LAYOUT_UNVERIFIED）」とする。
- accept_only: `m8n8k4 .f16`（quad-pair のレイアウトを誤りなく記述する根拠が手元にない）・f64 `m16n8k4/k8/k16`・INT8/FP8（`s8`・`e4m3`・`e5m2`・`kind::f8f6f4`）・`block_scale`・`redux.sync` の f32 版。wmma の C++ API（`mma.h`）は含めない。

### AC5 Hopper との差分・setmaxnreg・cluster・DSMEM（R-HOPPER・R-SNR・R-CLU）

| プローブ | 条項 | home | policy | layout | expect |
|---|---|---|---|---|---|
| `wgmma.m64n8k16` | R-HOPPER | sm_90a | accept_only | none | none |
| `hop.griddepcontrol` | R-HOPPER | sm_90 | verify | none | none |
| `hop.fence_proxy_async` | R-HOPPER | sm_90 | verify | none | none |
| `snr.dec` | R-SNR | sm_90a | verify | none | none |
| `snr.incdec` | R-SNR | sm_90a | verify | none | none |
| `clu.dims1` | R-CLU | sm_90 | verify | none | none |
| `clu.dims2` | R-CLU | sm_90 | verify | none | none |
| `clu.dims4` | R-CLU | sm_90 | verify | none | none |
| `clu.dims8` | R-CLU | sm_90 | verify | none | none |
| `clu.dims16` | R-CLU | sm_90 | verify | none | none |
| `clu.dsmem` | R-CLU | sm_90 | verify | none | none |

- `wgmma.m64n8k16` は受理段のみ（S6 は設計上不実施）。`snr.*` は既存の `setmaxnreg_probe_*` 4 ファイルと同じ命令列で、既存ファイルも再実行して結論が食い違えば「判定不能（LEGACY_CONTRADICTION）」とする（R-LEGACY）。
- cluster: `__cluster_dims__(N,1,1)`（N = 1/2/4/8/16）を safe API で起動し、`occupancy_max_active_clusters` 等を S3 の参考値として記録する。N>8 は `NON_PORTABLE_CLUSTER_SIZE_ALLOWED` を設定してから起動する。起動時に cluster 次元を与える `cuLaunchKernelEx` 経路は本 PR に含めない（TMA プローブと同時に追加する）。DSMEM は `%cluster_ctarank`・`mapa.shared::cluster`・`ld.shared::cluster`・`barrier.cluster` の往復で、デッドロックは外部 `timeout` が記録する。
- Hopper 列は GB10 上の NVRTC（ptxas）による `sm_90a` 受理の実測であり、Hopper 実機での実行は未検証。PTX ISA 9.0 の節番号は、確認できているもの（warp-level MMA は 9.7.15・非同期 warpgroup MMA は 9.7.16・第 5 世代 Tensor Core は 9.7.17。出典: `.claude/skills/nvidia-cuda/references/ptx-isa/instructions-matrix-multiply.md`。ただし同ファイルは PTX ISA 9.3 の章番号）以外を「要確認」とする。

## 5. 結果表（すべて「未実測」。GB10 実測後の PR で結果セルだけを更新する）

判定の語彙は §2。表のセルは `aggregate.py` の出力（`aggregate.md`）から転記する。

### AC1（R-TC5）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `macro.arch` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `tc5.alloc` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `tc5.ld` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `tc5.cross` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

### AC2（R-GUIDE）

| claim | 出典 | 測定項目 | 判定 |
|---|---|---|---|
| C01-warps-per-sm | streaming-multiprocessor.md:23 | `max_threads_per_multiprocessor`（warps/SM 48 → 1536） | 未実測 |
| C02-blocks-per-sm | 同 :26 | `max_blocks_per_multiprocessor` | 未実測 |
| C03-smem-per-sm | 同 :27 | `max_shared_memory_per_multiprocessor` | 未実測 |
| C04-smem-per-block | 同 :28 | `max_shared_memory_per_block_optin` | 未実測 |
| C05-portable-cluster-8 | 同 :33 | `clu.dims8` | 未実測 |
| C06-tcgen05-sm100plus | instructions-matrix-multiply.md:26 | `tc5.alloc` | 未実測 |
| C07-wgmma-unavailable | cuda-tensor-core-design.md:138 | `wgmma.m64n8k16` | 未実測 |
| C08-tcgen05-unavailable | 同 :139 | `tc5.alloc` | 未実測 |
| C09-cluster-1x1x1 | 同 :140 | `clu.dims2` | 未実測 |
| C10-static-smem-48kb | memory-system.md:26 | `max_shared_memory_per_block` | 未実測 |

### AC4（R-MMA）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `mma.tf32.m16n8k8` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.tf32.m16n8k4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f16.m16n8k16.f32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f16.m16n8k8.f32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f16.m16n8k16.f16` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.bf16.m16n8k16.f32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.bf16.m16n8k8.f32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f64.m8n8k4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f16.m8n8k4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f64.m16n8k4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f64.m16n8k8` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f64.m16n8k16` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.s8.m16n8k32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.e4m3.m16n8k32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.e5m2.m16n8k32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.f8f6f4.m16n8k32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.block_scale.m16n8k64` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.ldmatrix.x1` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.ldmatrix.x2` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.ldmatrix.x4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.ldmatrix.x4_trans` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `mma.stmatrix.x4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.fma_f32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.fma_f64` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.fma_f16x2` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.fma_bf16x2` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.f32x2_add` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.f32x2_mul` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.f32x2_fma` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.cvt_tf32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.elect_sync` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.redux_u32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `simt.redux_f32` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

### AC5（R-HOPPER・R-SNR・R-CLU）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `wgmma.m64n8k16` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `hop.griddepcontrol` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `hop.fence_proxy_async` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `snr.dec` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `snr.incdec` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `clu.dims1` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `clu.dims2` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `clu.dims4` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `clu.dims8` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `clu.dims16` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `clu.dsmem` | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

### AC3（TMA の意味論）

PR-B で追加予定・未実装。結果欄なし。

## 6. 実行手順

### 6.1 GB10 実機（共有ノード。空いているときに実行する）

`docs/real-hardware-verification-env.md` §3 の手順で転送し、§4.8 の外部 timeout 運用契約に従う。GB10 ノードの作業ツリーは `.git` を持たない（rsync で `.git/` を除外するため）ので、転送元で rsync の直前に provenance 用の `.rev-stamp` を **次の 2 行形式**で作る（§3 の `git rev-parse HEAD > .rev-stamp` の 1 行形式では `dirty=` 行がなく、G0 が `G0_DIRTY_TREE` で不成立になる。転送後に転送元の `.rev-stamp` は削除する）。

```bash
sha=$(git rev-parse HEAD); dirty=$(git status --porcelain --untracked-files=normal | wc -l)
printf '%s\ndirty=%s\n' "$sha" "$dirty" > .rev-stamp
```

`orchestrate.sh` は git 作業ツリーなら git から、`.git` が無ければ `.rev-stamp` から `git_head`・`git_clean` を読み（`dirty=0` のときだけ `git_clean=1`）、どちらも無ければ開始前に停止する。正式実行（`--dev-smoke` でない実行）で `git_clean` が 1 でない場合は、**プロセスを 1 つも起動せず `exit 1` で停止する**（dirty な状態の実測は G0 が不成立で使えないため、実測前に止める）。「dirty」は `git status --porcelain --untracked-files=normal` が空でないことで、submodule ポインタの変更（`M docs/spec` など）や未追跡ファイルも含む。`docs/real-hardware-verification-env.md` §3 の既存の 1 行形式（`git rev-parse HEAD > .rev-stamp`）とは異なり、**2 行目の `dirty=0` が必須**で、1 行形式のままだと停止する。`--dev-smoke` では dirty を許容する（結果は G0 不成立）。

```bash
env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
  ./docs/perf/logs/sm121-isa-probe-2122/orchestrate.sh
python3 docs/perf/logs/sm121-isa-probe-2122/aggregate.py --out docs/perf/logs/sm121-isa-probe-2122/aggregate.md
```

`orchestrate.sh` は RULE.txt の `PROCESS:`／`PROBE:`／`TARGET:` 行から起動列を導出し、(プローブ, target) を 1 プロセスずつ外部 `timeout` 付きで実行する（意図的な不正命令を含むためプロセス分離は必須）。既存ログがある出力先では開始前に停止する。`aggregate.py` が exit 2（完全性違反）・3（G0 不成立）なら結果を反映せず原因を調べる。

### 6.2 開発機スモーク（結果は結論にしない）

```bash
LD_LIBRARY_PATH=<libnvrtc のディレクトリ> LOG_DIR=<リポジトリ外の scratch> \
  CARGO_FLAGS=--offline ./docs/perf/logs/sm121-isa-probe-2122/orchestrate.sh --dev-smoke
python3 docs/perf/logs/sm121-isa-probe-2122/aggregate.py --log-dir <同じ scratch>   # G0 不成立（exit 3）・全セル判定不能
```

NVRTC 単体の S1／S2 全行列（GPU 不要）は `cargo test -p fandhe-ai-backend-cuda --all-features --test sm121_isa_probe_compile -- --ignored --nocapture` で回せ、プローブの文法誤りを GB10 の前に見つけられる（`home` の拒否はプローブ不良の疑い）。

### 6.3 開発機で確認できた事実（sm_121 の結論ではない）

- NVRTC 13.0.88（x86_64 の開発機）で、`--gpu-architecture=sm_XX` に実アーキを渡すと NVRTC 内部で ptxas が走り、不正な命令は ptxas のログ付きでコンパイルエラーになる。成功時は PTX テキストも取得できる。したがって S2 に `nvrtcGetCUBIN` の生 FFI は不要で、本基盤の `unsafe` は `launch` のみである。GB10 側の NVRTC で同じ挙動かは、compile.log と exec の S1／S2 の突き合わせ（COMPILE_EXEC_DISAGREE）が検出する範囲でしか確認しない。
- `layout=verified` の形状（TF32 `m16n8k8`／`m16n8k4`・f16 `m16n8k16`／`m16n8k8`・f16 累積・bf16・f64 `m8n8k4`・`ldmatrix` x1/x2/x4/x4.trans）と SIMT（`fma` の f32／f64／f16x2／bf16x2・`cvt.rna.tf32`・`redux.sync` の u32 版）・対照は、RTX 3060（sm_86）の `compute_86` 実行で S6 のビット一致を確認した。これはホスト参照モデル（fragment のレイアウト）の検証であり、sm_121 の実測ではない。（出典・再現: PR-A〈#2122 の最初の PR〉本文の実行記録〈検証コマンドと出力要約〉と、§6.2 の `--dev-smoke`。生ログはコミットしない。）

## 7. §2.3（`cuda-sm121-gemm-candidates-design.md`）への反映手順

GB10 実測後の PR で、次のセルだけを更新する（RULE.txt は変更しない）。起票はしない。

1. `cuda-sm121-gemm-candidates-design.md` §2.3 の各行の「状態」を、対応するプローブ ID・条項（同表の列）の判定に更新する。§2.1 の wgmma・tcgen05・cluster・`setmaxnreg` の行、§1-1 の条件文への注記、§4・§7 の「#2122 次第」の行も同様。
2. `cuda-tensor-core-design.md` §11.1・§13 の空欄・「未了」を、判定（および `snr.*` と既存 `setmaxnreg_probe_*` の整合）で更新する。
3. `backend-cuda-tma-gemm-load-design.md` §10.8 の訂正追記はそのまま残す（TMA の意味論は PR-B）。
4. `perf/sm121-device-attributes.md` の「未実測」欄を `attr.*` の記録（`aggregate.md` の R-GUIDE 表と `exec/attr.*@*.log`）で埋める。
5. 本 doc の §5 の結果セルを更新する。

## 8. 出典・関連

- 判定規則: `docs/perf/logs/sm121-isa-probe-2122/RULE.txt`・`README.md`・`guide_claims.tsv`・`orchestrate.sh`・`aggregate.py`
- 実装: `crates/backend-cuda/tests/sm121_isa_probe_common/`・`sm121_isa_probe_registry.rs`・`sm121_isa_probe_compile.rs`・`sm121_isa_probe_exec_real_device.rs`
- 先例: `docs/perf/logs/cuda-gemm-small-launch-cost-2109/`（RULE／orchestrate／aggregate の構成）、`crates/backend-cuda/tests/setmaxnreg_common/mod.rs`（プロセス分離・対照カーネル）
- 関連: `docs/cuda-sm121-gemm-candidates-design.md`（#2130。差し込み欄 §2.3）・`docs/cuda-tensor-core-design.md` §11.1・§13・`docs/backend-cuda-tma-gemm-load-design.md` §10.4・§10.8・`docs/int8-quant-grade-up-verification-plan.md`・`docs/real-hardware-verification-env.md` §4.8
