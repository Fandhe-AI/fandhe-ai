# sm_121 可用命令・アーキ固有機能の実機プローブ基盤（#2122）

## 0. 位置づけ・スコープ

- 対応イシュー #2122（親 #2121 Phase 4。「sm_121 の使える命令とアーキ固有機能のプローブ」）。本 doc は **PR-A（プローブ基盤）** の設計・使い方・結果欄を持つ。実装は `crates/backend-cuda/tests/sm121_isa_probe_*`、判定規則（実測前に固定）は `docs/perf/logs/sm121-isa-probe-2122/RULE.txt` を正とする。
- **GB10 実測済み（PR-C・2026-10-01）**: §5 の結果セルは GB10（DGX Spark・sm_121）での正式実行 1 系列（転送元 `7b6019ae`・`aggregate.py` exit 0・G0 成立）の `aggregate.md` から転記した。PR-A・PR-B の時点では結果欄はすべて「未実測」だった（以下の PR-A・PR-B の説明はその時点の記述）。開発機（RTX 3060・sm_86）のスモーク結果は G0 を不成立にし全セルを判定不能にする設計で、sm_121 の結論としては一切扱わない。
- **AC3（`cp.async.bulk.tensor` の意味論: 要素座標・部分 OOB・swizzle の smem 配置・store・bulk・prefetch・multicast）は PR-B で実装した**（`tma.*` 15 件・条項 R-TMA-BASE／R-TMA-SEM／R-TMA-XFER・既存 `tma_probe_real_device` の再実行〈R-LEGACY の拡張〉・runtime で cluster 次元を与える `cuLaunchKernelEx` の raw 起動経路）。PR-B の時点では結果はすべて未実測で、TMA 系のカーネルは開発機（sm_86）で S2 の ptxas 拒否・S3 のロード失敗に分類される経路まで確認しただけだった。GB10 の実測結果は §5（AC3 表・R-TMA 観測）を参照。
- 本 PR はテストとログ規則・ドキュメントのみで、本番経路・tolerance・baseline・閾値・facade 公開面・既存の `tests/tma_probe_real_device.rs`・`tests/setmaxnreg_*` は変更しない。

## 1. 背景（既存の記録との食い違い）

- 既存の TMA プローブ（`tests/tma_probe_real_device.rs`）の「arch ごとのコンパイル成功」は、命令が受理されたことをほぼ示さない。`compile_ptx(src, "compute_121")` は NVRTC が PTX テキストを出すだけで、inline PTX の中身を ptxas が検証するのは `cuModuleLoadData`（ドライバ JIT）の時点だからである。結論は実行段の成功から出ている。本基盤は「NVRTC の受理」と「ptxas の受理」を別の段として記録する。
- `docs/backend-cuda-tma-gemm-load-design.md` §10.8 は「意味論プローブ 3 件 ✓」と書くが、§10.4 の定義（要素座標・部分 OOB・swizzle の smem 配置ダンプ）の 3 件は PR-A 時点で未実装だった（PR-B で `tma.coord`・`tma.oob_*`・`tma.swz64` として実装し、GB10 で実測済み〈§5 AC3〉。同 §10.8 末尾に訂正の追記を置いた。`cuda-sm121-gemm-candidates-design.md` §2.1 の記述が正しい）。
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

### AC3 `cp.async.bulk.tensor`（R-TMA-BASE・R-TMA-SEM・R-TMA-XFER。PR-B）

| プローブ | 条項 | home | policy | layout | expect |
|---|---|---|---|---|---|
| `tma.base_cta` | R-TMA-BASE | sm_90 | verify | none | none |
| `tma.base_cluster` | R-TMA-BASE | sm_90 | verify | none | none |
| `tma.coord` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.oob_none` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.oob_nan` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.oob_neg` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.swz32` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.swz64` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.swz128` | R-TMA-SEM | sm_90 | record_only | none | none |
| `tma.store` | R-TMA-XFER | sm_90 | verify | none | none |
| `tma.prefetch` | R-TMA-XFER | sm_90 | verify | none | none |
| `tma.multicast` | R-TMA-XFER | sm_90 | verify | none | none |
| `tma.bulk_cta` | R-TMA-XFER | sm_90 | verify | none | none |
| `tma.bulk_cluster` | R-TMA-XFER | sm_90 | verify | none | none |

- 共通: global は 64x96 の f32（要素値は `r * 1000 + c`）。`cuTensorMapEncodeTiled` の引数（box・swizzle・OOB fill・座標・`expect_tx`）はプローブごとに引数化したテスト側の encoder（`model_tma.rs::TmaSpec`）で与える。期待値は持たず観測値を記録し、候補モデルと一致するかで判定する。**待ちの方針（上限到達時に転送先の smem を読まない）**: mbarrier の待ちはまず上限回数（1,000,000 回）まで数える。上限に達して未完了なら、転送先の smem は読まず、到達の事実を global（`out[0]` のマーカー）へ残して `__threadfence_system()` で出したうえで、上限なしの待ちに移る。完了しなければカーネルは終了せず、外部 `timeout` がプロセスごと打ち切り、`aggregate.py` は `TIMEOUT`（判定不能）として扱う（ホストはカーネルの完了を待てないため記録上の正は外部 timeout の proc 記録で、マーカーはホストから読めない）。したがって「完了を確認する前に転送先を読む」経路と「転送中に CTA が終了する」経路はなく、CTA 終了後の smem への書き込み（未定義動作）は起こらない。`cp.async.bulk.wait_group 0`（`tma.store`）と `barrier.cluster`（`tma.multicast`・`clu.dsmem`）も上限を持たず外部 `timeout` に頼る。ダンプ・読み戻しは待ちが完了した後にだけ行う（registry テストがソース構造を静的に検査する。限界: 実行時の挙動までは保証しない）。出力ヘッダは全 TMA カーネル共通で `[polls]` の 1 語（registry テストがカーネルの格納と checker の解釈の一致を検査し、aggregate の必須キーもこれに従う）。**tx-count の規約**: `expect_tx` の総量は同じ phase で実際に届く転送量（OOB を含む box 全体・bulk の長さ・multicast では宛先 CTA ごとの受信量）と一致させ、待つ phase には必ず転送が来る（registry テストがソース構造と宣言値の一致を静的に検査する。実行時の転送量までは保証しない）。初期化は、smem 初期化の後に全スレッドが `fence.proxy.async.shared::cta`、mbarrier を init したスレッドが `fence.mbarrier_init.release.cluster` を出してから発行する（multicast は `barrier.cluster` 同期のあと。registry テストが順序を静的に検査する）。smem は転送前に番兵（`0xFEEDFACE`）で埋めてダンプする。
- R-TMA-BASE: `tma.base_cta`（`shared::cta`）・`tma.base_cluster`（`shared::cluster`・cluster 1 で起動）。box 転送が完走し、要素座標（内側次元が先）の仮説どおりにビット一致するかを判定する。
- R-TMA-SEM（観測の記録。判定語は「受理のみ」で、観測内容は S5 の detail に `k=v` で残す）: `tma.coord`（`class=ELEM_INNER_FIRST`／`TRANSPOSED`／`NONE`）・`tma.oob_none`／`oob_nan`／`oob_neg`（範囲内の一致・OOB 要素のビット列の種類〈ZERO／NAN／SENTINEL／OTHER〉）（OOB 要素が `complete_tx` のバイト数に含まれるか、の問いは `tma.oob_none`／`oob_nan` で答える。どちらも `expect_tx` を OOB を含む box 全体のバイト数にしており、完了すれば「OOB も数える」と判定し、完了しなければ上限なしの待ちのまま外部 timeout で打ち切られ TIMEOUT〈判定不能〉になる〈事前登録〉。tx-count を実際の転送量より少なく期待する構成は未定義動作になりうるため計測しない。旧 `tma.oob_tx_partial` は撤去した）・`tma.swz32`／`swz64`／`swz128`（box の内側が swizzle 幅ちょうど。無 swizzle・標準の XOR〈アドレスビット [7,7+B) を [4,4+B) へ〉・src の B64 仮説〈`tma_swizzled_chunk_a`。64B のみ〉のどれと一致するか。どれとも一致しなければダンプ全文を記録）。**本 box（タイル先頭 1024 B 整列・行 64 B）では src の B64 仮説と標準の XOR モデルは全語で同一になり区別できない**（区別にはタイル先頭をずらす必要があるが、swizzle 使用時の smem 整列要件のため作れない）。両者は常に同時に現れ、優劣は判定しない。観測の `k=v` キーはプローブごとに固定し、`class=NONE`・`inrange=MISMATCH`・OOB 要素が番兵のままのものは、判定語（受理のみ）に埋もれないよう `aggregate.md` の「R-TMA 観測」表の注意欄に明示する。
- R-TMA-XFER（成立の判定）: `tma.store`（global の読み戻しが既知パターンと一致）・`tma.bulk_cta`／`tma.bulk_cluster`（`cp.async.bulk` の 256 B コピー）・`tma.prefetch`（`prefetch.tensormap`＋`cp.async.bulk.prefetch.tensor`。効果は観測できず完走の目印のみ）・`tma.multicast`（cluster 2・両 CTA の smem が一致）。
- 依存（事前登録。自身が「成立」「受理のみ」になるときに限り、依存先が同 target で「成立」でなければ判定不能にする。評価順）: ① raw 起動（`cuLaunchKernelEx`）で動く `tma.*`・`clu.rt2`／`clu.rt4`・`ctl.rawmap` は対照 `ctl.raw` に依存（`RAW_LAUNCH_CTL_FAILED`）。tensor map 引数を使う `tma.*`（`tma.bulk_*` を除く）は `ctl.rawmap` にも依存（`RAW_TENSORMAP_CTL_FAILED`）。② `tma.base_cta` が「成立」でない target では、他の `tma.*`（`tma.base_cluster` を除く）を「判定不能（`TMA_BASE_NOT_ESTABLISHED`）」にする。③ `tma.multicast` は加えて runtime cluster 起動の `clu.rt2` が「成立」でない target で「判定不能（`CLU_RT2_NOT_ESTABLISHED`）」。自身が拒否・ロード失敗・実行時エラー・結果不一致のときは、その判定をそのまま採る。
- swizzle の候補モデルは `fandhe_ai_backend_cuda::tma_swizzled_chunk_a`（src の B64 仮説。`internal-diagnostics` feature 限定で再公開。通常ビルドの公開面は増えない）を再利用する。標準の XOR モデルとの全語一致は registry テストが検査する。
- 補助: `clu.rt2`／`clu.rt4`（`cuLaunchKernelEx` の `CLUSTER_DIMENSION` 属性で runtime に cluster 次元を与え、`clu.dims*` と同じ出力を観測）。対照 `ctl.raw`（raw 起動経路。tensor map なし。sm_86 でも S6 まで通る）・`ctl.rawmap`（tensor map 引数の ABI〈128 B 整列の値渡し・後続引数〉。`cuTensorMapEncodeTiled` が必要で、開発機の sm_86 では encode が `CUDA_ERROR_NOT_SUPPORTED`＝S4 の実行時エラーになる）。
- 既存の `tma_probe_real_device`（3 テスト）は 1 テストずつ別プロセスで再実行する（legacy 名は `tma_probe_real_device@<テスト関数>`）。`tma_execution_probe`（cluster 変種）↔ `tma.base_cluster`、`tma_execution_probe_cta` ↔ `tma.base_cta`（対象 target は legacy が選択した arch）を成否で突き合わせ、確定した失敗（不一致・全 arch のコンパイル失敗）と新プローブの成否が食い違えば「判定不能（LEGACY_CONTRADICTION）」、legacy の timeout・異常終了（一過性の失敗を含む）は RULE 13a と同じく「判定不能（LEGACY_INCONCLUSIVE）」とする。比較するのは legacy が先頭で受理して選択した arch だけで、その arch が正式 target（`compute_121`／`121a`／`121f`。既存テストの `PROBE_ARCHS` と同じ 3 値）の外なら、比較を省かず完全性違反として集計を止める（fail-closed）。

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
| `clu.rt2` | R-CLU | sm_90 | verify | none | none |
| `clu.rt4` | R-CLU | sm_90 | verify | none | none |

- `wgmma.m64n8k16` は受理段のみ（S6 は設計上不実施）。`snr.*` は既存の `setmaxnreg_probe_*` 4 ファイルと同じ命令列で、既存ファイルも再実行して結論が食い違えば「判定不能（LEGACY_CONTRADICTION）」とする（R-LEGACY）。
- cluster: `__cluster_dims__(N,1,1)`（N = 1/2/4/8/16）を safe API で起動し、`occupancy_max_active_clusters` 等を S3 の参考値として記録する。N>8 は `NON_PORTABLE_CLUSTER_SIZE_ALLOWED` を設定してから起動する。起動時に cluster 次元を与える経路は PR-B の `clu.rt2`／`clu.rt4`（raw の `cuLaunchKernelEx`）で追加した（AC3 節）。DSMEM は `%cluster_ctarank`・`mapa.shared::cluster`・`ld.shared::cluster`・`barrier.cluster` の往復で、デッドロックは外部 `timeout` が記録する。
- Hopper 列は GB10 上の NVRTC（ptxas）による `sm_90a` 受理の実測であり、Hopper 実機での実行は未検証。PTX ISA 9.0 の節番号は、確認できているもの（warp-level MMA は 9.7.15・非同期 warpgroup MMA は 9.7.16・第 5 世代 Tensor Core は 9.7.17。出典: `.claude/skills/nvidia-cuda/references/ptx-isa/instructions-matrix-multiply.md`。ただし同ファイルは PTX ISA 9.3 の章番号）以外を「要確認」とする。

## 5. 結果表（GB10 実測済み。2026-10-01）

判定の語彙は §2。表のセルは `aggregate.py` の出力（`docs/perf/logs/sm121-isa-probe-2122/aggregate.md`）から逐語で転記した。各セルの段ごとの記録（ptxas のログ全文・エラー名・参考値）は `docs/perf/logs/sm121-isa-probe-2122/exec/<プローブ>@<target>.log`、home・Hopper 列は同 `compile.log` を正とする。

- **実測環境**: 実測日 2026-10-01（`env_info.txt` の `start_utc=2026-10-01T06:33:39Z`〜`end_utc=2026-10-01T06:35:34Z`）・`<cuda-node>`（NVIDIA GB10・cc 12.1）・ドライバ 580.173.02・NVRTC 13.0（compile・exec の env レコード）・Linux 6.17.0-1031-nvidia aarch64。転送元コミット `7b6019aeb7856eb9cb2e28929a735b9938bc1503`（`git_source=rev-stamp`・`git_clean=1`）。
- **占有状況**: 常駐サービス 2 プロセスが CUDA コンテキストを保持したまま（`utilization.gpu` 0%）、ユーザー承認のうえ実行した。実行前後で他の GPU プロセスの出現はない。
- **系列の健全性**: 起動プロセス 219 件（device_attributes_dump 1・compile 1・exec 210・legacy 7）はすべて exit 0（timeout・process_failed 0 件）。`aggregate.py` は exit 0（G0 成立・完全性違反なし）。判定不能 0 件・想定外の受理（UNEXPECTED_ACCEPT）0 件・LEGACY_CONTRADICTION 0 件。home（本来の対応アーキ）の S2 は 67 件すべて ok（HOME_REJECTED なし）、対照 `ctl.copy`・`ctl.raw`・`ctl.rawmap` は 3 target すべて成立。
- 列の値: compute_121／121a／121f は判定語。home 列は `compile.log` の target=home セルの状態（括弧内は home のアーキ）。Hopper 列は `compile.log` の target=hopper（sm_90a）の S2 の状態（GB10 上の NVRTC〈ptxas〉による受理の実測で、Hopper 実機での実行は未検証）。

### AC1（R-TC5）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `macro.arch` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_80） | ok |
| `tc5.alloc` | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ok（sm_100a） | rejected |
| `tc5.ld` | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ok（sm_100a） | rejected |
| `tc5.cross` | ロード失敗 | ロード失敗 | ロード失敗 | ok（sm_100a） | rejected |

### AC2（R-GUIDE）

| claim | 出典 | 測定項目 | 判定 |
|---|---|---|---|
| C01-warps-per-sm | streaming-multiprocessor.md:23 | `max_threads_per_multiprocessor`（warps/SM 48 → 1536） | 一致（12.0 の記述を外挿）（測定値=1536,1536,1536） |
| C02-blocks-per-sm | 同 :26 | `max_blocks_per_multiprocessor` | 不一致（測定値=24,24,24） |
| C03-smem-per-sm | 同 :27 | `max_shared_memory_per_multiprocessor` | 不一致（測定値=102400,102400,102400） |
| C04-smem-per-block | 同 :28 | `max_shared_memory_per_block_optin` | 一致（12.0 の記述を外挿）（測定値=101376,101376,101376） |
| C05-portable-cluster-8 | 同 :33 | `clu.dims8` | 一致（12.0 の記述を外挿）（判定=成立,成立,成立） |
| C06-tcgen05-sm100plus | instructions-matrix-multiply.md:26 | `tc5.alloc` | 不一致（判定=ptxas 拒否（オフライン）,ptxas 拒否（オフライン）,ptxas 拒否（オフライン）） |
| C07-wgmma-unavailable | cuda-tensor-core-design.md:138 | `wgmma.m64n8k16` | 一致（12.0 の記述を外挿）（判定=ptxas 拒否（オフライン）,ptxas 拒否（オフライン）,ptxas 拒否（オフライン）） |
| C08-tcgen05-unavailable | 同 :139 | `tc5.alloc` | 一致（12.0 の記述を外挿）（判定=ptxas 拒否（オフライン）,ptxas 拒否（オフライン）,ptxas 拒否（オフライン）） |
| C09-cluster-1x1x1 | 同 :140 | `clu.dims2` | 不一致（判定=成立,成立,成立） |
| C10-static-smem-48kb | memory-system.md:26 | `max_shared_memory_per_block` | 一致（12.0 の記述を外挿）（測定値=49152,49152,49152） |

### AC4（R-MMA）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `mma.tf32.m16n8k8` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.tf32.m16n8k4` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.f16.m16n8k16.f32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.f16.m16n8k8.f32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.f16.m16n8k16.f16` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.bf16.m16n8k16.f32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.bf16.m16n8k8.f32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.f64.m8n8k4` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.f16.m8n8k4` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_80） | ok |
| `mma.f64.m16n8k4` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `mma.f64.m16n8k8` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `mma.f64.m16n8k16` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `mma.s8.m16n8k32` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_80） | ok |
| `mma.e4m3.m16n8k32` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_89） | ok |
| `mma.e5m2.m16n8k32` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_89） | ok |
| `mma.f8f6f4.m16n8k32` | ptxas 拒否（オフライン） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_120a） | rejected |
| `mma.block_scale.m16n8k64` | ptxas 拒否（オフライン） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_120a） | rejected |
| `mma.ldmatrix.x1` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.ldmatrix.x2` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.ldmatrix.x4` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.ldmatrix.x4_trans` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `mma.stmatrix.x4` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `simt.fma_f32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `simt.fma_f64` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `simt.fma_f16x2` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `simt.fma_bf16x2` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `simt.f32x2_add` | 成立 | 成立 | 成立 | ok（sm_100） | rejected |
| `simt.f32x2_mul` | 成立 | 成立 | 成立 | ok（sm_100） | rejected |
| `simt.f32x2_fma` | 成立 | 成立 | 成立 | ok（sm_100） | rejected |
| `simt.cvt_tf32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `simt.elect_sync` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `simt.redux_u32` | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `simt.redux_f32` | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ok（sm_100a） | rejected |

### AC5（R-HOPPER・R-SNR・R-CLU）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `wgmma.m64n8k16` | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ptxas 拒否（オフライン） | ok（sm_90a） | ok |
| `hop.griddepcontrol` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `hop.fence_proxy_async` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `snr.dec` | ptxas 拒否（オフライン） | 成立 | 成立 | ok（sm_90a） | ok |
| `snr.incdec` | ptxas 拒否（オフライン） | 成立 | 成立 | ok（sm_90a） | ok |
| `clu.dims1` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `clu.dims2` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `clu.dims4` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `clu.dims8` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `clu.dims16` | 実行時エラー | 実行時エラー | 実行時エラー | ok（sm_90） | ok |
| `clu.dsmem` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `clu.rt2` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `clu.rt4` | 成立 | 成立 | 成立 | ok（sm_90） | ok |

### AC3（TMA の意味論。R-TMA-*）

| プローブ | compute_121 | compute_121a | compute_121f | home（本来の対応アーキ）S2 | Hopper（sm_90a）S2 |
|---|---|---|---|---|---|
| `tma.base_cta` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `tma.base_cluster` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `tma.coord` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.oob_none` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.oob_nan` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.oob_neg` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.swz32` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.swz64` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.swz128` | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | 受理のみ（実行意味論は未検証） | ok（sm_90） | ok |
| `tma.store` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `tma.prefetch` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `tma.multicast` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `tma.bulk_cta` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `tma.bulk_cluster` | 成立 | 成立 | 成立 | ok（sm_90） | ok |
| `ctl.raw`（対照・R-CTL） | 成立 | 成立 | 成立 | ok（sm_80） | ok |
| `ctl.rawmap`（対照・R-CTL） | 成立 | 成立 | 成立 | ok（sm_90） | ok |

R-TMA-SEM の観測（`aggregate.md` の「R-TMA 観測」表の転記。判定語はいずれも「受理のみ（実行意味論は未検証）」で、3 target とも同じ観測。注意欄はすべて `-`）:

| プローブ | 観測（compute_121／121a／121f 共通） | 候補モデルとの一致 |
|---|---|---|
| `tma.coord` | `class=ELEM_INNER_FIRST` | 要素座標（内側次元が先）の仮説と一致 |
| `tma.oob_none` | `inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000` | 範囲内は一致・OOB 要素はゼロ埋め |
| `tma.oob_nan` | `inrange=MATCH oob_elems=96 oob_fill=NAN oob_distinct=0x7ff77ff7` | 範囲内は一致・OOB 要素は NaN（ビット列 `0x7ff77ff7`） |
| `tma.oob_neg` | `inrange=MATCH oob_elems=96 oob_fill=ZERO oob_distinct=0x00000000` | 負の座標の OOB 要素もゼロ埋め |
| `tma.swz32` | `class=XOR_ADDR_BITS` | 標準の XOR モデルと一致 |
| `tma.swz64` | `class=XOR_ADDR_BITS+SRC_B64_MODEL` | 標準の XOR モデルと src の B64 仮説の両方と一致（本 box では両者は全語で同一になり区別できない。優劣は判定しない） |
| `tma.swz128` | `class=XOR_ADDR_BITS` | 標準の XOR モデルと一致 |

`polls` は 13〜14（待ちの上限 1,000,000 回に対して）。`tma.oob_none`／`tma.oob_nan` は `expect_tx` を OOB を含む box 全体（512 B）にした構成で待ちが完了しており、事前登録（§4 AC3）のとおり「OOB 要素も `complete_tx` のバイト数に数える」と読む。

### 5.1 注記（判定語の読み方と記録値）

- **アーキ接尾辞（`121`／`121a`／`121f`）の差**: 3 target で判定が分かれたのは `mma.f8f6f4.m16n8k32`・`mma.block_scale.m16n8k64`（compute_121 は ptxas 拒否〈オフライン〉、121a・121f は受理のみ）と `snr.dec`・`snr.incdec`（compute_121 は ptxas 拒否〈オフライン〉、121a・121f は成立）の 4 件だけで、他は 3 target で同じ判定。ptxas のログは `Feature '.kind::f8f6f4' not supported on .target 'sm_121'`・`Instruction 'setmaxnreg.dec' not supported on .target 'sm_121'`（`exec/mma.f8f6f4.m16n8k32@compute_121.log`・`exec/snr.dec@compute_121.log`）。`aggregate.md` の R-HOPPER 表で方向列が「判定不能（target 間で不一致）」となっているのはこの 4 件で、これは方向列のラベルであり判定語（判定不能・理由コード付き）ではない。
- **tcgen05／TMEM**: `tc5.alloc`・`tc5.ld` は 3 target とも ptxas 拒否（オフライン）（`Instruction 'tcgen05.alloc' not supported on .target 'sm_121a'` ほか。`exec/tc5.alloc@compute_121a.log`）。S3（ドライバ JIT）も `CUDA_ERROR_INVALID_PTX`。home（sm_100a）の S2 は ok で、拒否はアーキ起因と読む（R-HOME）。`tc5.cross`（compute_100a の PTX を GB10 でロード）は 3 target ともロード失敗（`CUDA_ERROR_INVALID_PTX`）。事前登録の想定（`expect=reject121`）どおりで、想定外の受理はない。
- **`tc5.cross` の S2**: `tc5.cross` の S2 は固定アーキ sm_100a（FIXEDARCH）での受理であり、R-HOPPER 表の方向「sm_121 のみ受理（逆方向）」は sm_121 で受理されたことを意味しない（sm_121 系の S2 欄に sm_100a の結果が入っている）。
- **wgmma**: `wgmma.m64n8k16` は 3 target とも ptxas 拒否（オフライン）（`Instruction 'wgmma.fence' not supported on .target 'sm_121a'` ほか）。Hopper（sm_90a）の S2 は ok（R-HOPPER の方向は「Hopper のみ受理」）。
- **cluster**: `clu.dims1`〜`dims8`・`clu.dsmem`・`clu.rt2`／`rt4` は 3 target とも成立。`clu.dims16` は 3 target とも実行時エラー（S4 `CUDA_ERROR_INVALID_CLUSTER_SIZE`。S3 の参考値は `non_portable_cluster_size_allowed=set occupancy_max_active_clusters=0 occupancy_max_potential_cluster_size=12`。`exec/clu.dims16@compute_121.log`）。参考値として `clu.dims8` の `occupancy_max_active_clusters=12`、`clu.dsmem`（cluster 2）は 48。
- **Hopper との差分（sm_121 でだけ受理）**: `simt.f32x2_add`／`mul`／`fma`（home sm_100）は Hopper（sm_90a）の S2 が rejected で、sm_121 系では 3 target とも成立。`simt.redux_f32`（home sm_100a）は Hopper・sm_121 系の両方で ptxas 拒否（オフライン）（`Instruction 'redux.f32' not supported on .target 'sm_121a'`）。
- **`macro.arch` の記録値**（S5 の detail。存在を主張しない記録のみ）: 3 target とも `__CUDA_ARCH__=1210`・`__CUDA_ARCH_FEAT_SM121_ALL=0`・`__CUDA_ARCH_FEAT_SM120_ALL=0`・`__CUDA_ARCH_FEAT_SM100_ALL=0`・`__CUDA_ARCH_FEAT_SM90_ALL=0`・`__CUDACC_VER_MAJOR__=13`・`__CUDACC_VER_MINOR__=0`。`__CUDA_ARCH_SPECIFIC__` は compute_121a のみ 1（121・121f は 0）、`__CUDA_ARCH_FAMILY_SPECIFIC__` は compute_121a・121f で 1（121 は 0）。
- **R-LEGACY**: `setmaxnreg_probe_*` の base（compute_121）2 件は load_failed、accel（compute_121a）2 件は run_ok で、`snr.*` の同 target の判定（compute_121 は ptxas 拒否〈オフライン〉で S3 もロード失敗・121a は成立）と食い違わない（LEGACY_CONTRADICTION・LEGACY_INCONCLUSIVE なし）。`tma_probe_real_device` の `tma_execution_probe`（cluster 変種）と `tma_execution_probe_cta` は選択 arch compute_121 で run_ok、`tma.base_cluster`・`tma.base_cta` の成立と整合。
- **AC2（ガイドとの不一致）**: 参照ガイド（`.claude/skills/nvidia-cuda/`。CC 12.0 の記述）・設計文書（`cuda-tensor-core-design.md` §11.1）の記述と GB10（CC 12.1）の実測で「不一致」となったのは 4 件: C02（ガイドの「Maximum thread blocks per SM 32」に対し `max_blocks_per_multiprocessor` の測定値 24）・C03（ガイドの「Maximum shared memory per SM 128 KB」に対し `max_shared_memory_per_multiprocessor` の測定値 102400 B＝100 KB）・C06（ガイドの「`tcgen05.*` (Blackwell, sm_100+)」に対し、sm_121 では `tc5.alloc` が ptxas 拒否〈オフライン〉。sm_121 は「sm_100+」の範囲に含まれない）・C09（`cuda-tensor-core-design.md` の「cluster は実用上不可〈1×1×1 のみ〉」に対し、`clu.dims2` が 3 target とも成立）。C01・C04・C05・C07・C08・C10 は「一致（12.0 の記述を外挿）」（CC 12.0 の記述を 12.1 へ外挿したうえでの条件付きの一致）。各 claim の原文引用・比較方法は `docs/perf/logs/sm121-isa-probe-2122/guide_claims.tsv`、測定値は表の括弧内。`.claude/skills` は編集せず、不一致は本 doc にだけ記す（R-GUIDE）。

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

GB10 実測後の PR で、次のセルだけを更新する（RULE.txt は変更しない）。起票はしない。**2026-10-01 の実測（PR-C）で 1〜5 を反映済み**。

1. `cuda-sm121-gemm-candidates-design.md` §2.3 の各行の「状態」を、対応するプローブ ID・条項（同表の列）の判定に更新する。§2.1 の wgmma・tcgen05・cluster・`setmaxnreg` の行、§1-1 の条件文への注記、§4・§7 の「#2122 次第」の行も同様。
2. `cuda-tensor-core-design.md` §11.1・§13 の空欄・「未了」を、判定（および `snr.*` と既存 `setmaxnreg_probe_*` の整合）で更新する。
3. `backend-cuda-tma-gemm-load-design.md` §10.8 の訂正追記はそのまま残し、TMA の意味論プローブ（`tma.*`）の判定・観測を追記する。
4. `perf/sm121-device-attributes.md` の「未実測」欄を `attr.*` の記録（`aggregate.md` の R-GUIDE 表と `exec/attr.*@*.log`）で埋める。
5. 本 doc の §5 の結果セルを更新する。

## 8. 出典・関連

- 判定規則: `docs/perf/logs/sm121-isa-probe-2122/RULE.txt`・`README.md`・`guide_claims.tsv`・`orchestrate.sh`・`aggregate.py`
- 実装: `crates/backend-cuda/tests/sm121_isa_probe_common/`・`sm121_isa_probe_registry.rs`・`sm121_isa_probe_compile.rs`・`sm121_isa_probe_exec_real_device.rs`
- 先例: `docs/perf/logs/cuda-gemm-small-launch-cost-2109/`（RULE／orchestrate／aggregate の構成）、`crates/backend-cuda/tests/setmaxnreg_common/mod.rs`（プロセス分離・対照カーネル）
- 関連: `docs/cuda-sm121-gemm-candidates-design.md`（#2130。差し込み欄 §2.3）・`docs/cuda-tensor-core-design.md` §11.1・§13・`docs/backend-cuda-tma-gemm-load-design.md` §10.4・§10.8・`docs/int8-quant-grade-up-verification-plan.md`・`docs/real-hardware-verification-env.md` §4.8
