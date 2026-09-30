# Apple M4 Max GPU（Metal）の最適化考察（イシュー #2124）

対応イシュー: #2124（親 #2121「チップ情報に基づく最適化の考察」・ルート #2058）。兄弟 #2123（GB10）と対になる M4 Max 版で、
総括 #2129 から引用される前提の記録である。

- **docs 専用・コード変更ゼロ**（`crates/**`・`Cargo.toml`・本番既定値・tolerance・`docs/spec` は無変更）
- **Linux 環境で作成し新規実測はしていない**。数値はすべて既存の実測記録からの転記で、出典（`file:§`）を付ける。
  未計測の欄は「未計測」と書き、推定は「推定」と明記する
- 採否の決定はしない。既定値の切替・ユーザー判断事項は人間承認に委ねる（`.claude/rules/deps-policy.md`・`security.md`）

## 判断サマリ

| AC | 回答 |
|---|---|
| AC1 MPP（Route C） | Route C は `docs/backend-metal-mpp-tensor-decision.md`（#1326）で**実装・実測済み**。採否は同 §6 の (a)〜(c) で**ユーザー判断待ち**として記録できる。本 doc は採否を決めない |
| AC2 N=4096 頭打ち | GPU counters は取得不能。代替証跡の消去法で「タイル選択や並列度不足ではなく、カーネル本体の実行効率（ロード・同期・レジスタ／SMEM 配置）側」と**推定**する。確証は未取得 |
| AC3 「NAX／行キャッシュ競合」 | 用語が repo の記録と異なるため §2 で訂正し 2 論点に分けて再評価。NAX／MPP は M4 Max では後退（N>=2048）で再訪条件未充足。レジスタフラグメント配置は標準 API 下で制御不能 |
| AC4 UMA readback | 2 経路は「共有バッファ＋ホスト memcpy」と「ゼロコピー」。implicit lazy copy は発生しない。現行到達点と未計測欄を §7 に整理 |
| AC5 Phase 3 施策 | #2110〜#2114 はいずれも opt-in・既定 OFF・M4 Max 実測未実施。事前登録 RULE.txt の判定を条件とする**条件付き適用**。async copy／aligned load／Morton／E6〜E8 は**非目標** |

## §1 前提

- 対象機体: Apple M4 Max（GPU architecture `applegpu_g16s`・macOS 26.6.2。`docs/backend-metal-mpp-tensor-decision.md` §1.1）
- Neural Accelerator は非搭載世代（同 §5）
- 負けセル（対 candle／PyTorch。`docs/perf/loss-attribution-matrix.md`）: M-MTL-G256〜G4096 が 0.93／0.72／0.68／0.74／0.59 倍、
  M-MTL-TRN 0.38 倍、M-MTL-INF 0.79 倍
- Phase 3 で既定 ON に結線された ADOPT 施策は 0 件（`docs/perf/framework-compare-phase3-remeasure.md` §2）

## §2 用語・前提の突合（Issue 本文と repo 記録の食い違い）

Issue の表現をそのまま #2129 へ引き継ぐと誤りが伝播するため、repo 上の記録と突合して訂正する。

| Issue の表現 | repo 上の記録 | 出典 | 本 doc での扱い |
|---|---|---|---|
| NAX を「non-aligned-x」とする解釈 | NAX は MLX の `MetalPerformancePrimitives` `mpp::tensor_ops::matmul2d` 経路（Neural Accelerator 向け）。「行キャッシュ競合」の記録は repo に存在しない | `backend-metal-mlx-classic-nax-decision.md` 判断サマリ・§2 | NAX＝MPP 経路として扱う。競合論点は近い記録（Morton・tgid swizzle）に分けて再評価（§5） |
| simdgroup_matrix の NEON API | `simdgroup_matrix` は Metal（MSL）の GPU API。NEON は CPU SIMD（`backend-cpu`） | `backend-metal-morton-mapping-decision.md` 判断サマリ | GPU 側の話として扱う |
| split-K を性能限界（REJECT 扱い）とする | **本番結線済み**（#1516。`SPLIT_K_DEFAULT_ENABLED = true`、`crates/backend-metal/src/split_k_runtime.rs:84`） | `backend-metal-splitk-decision.md` §5 | 適用済み施策として扱う |
| hfrag を限界とする | E9 の候補。N=4096 のみ約 10〜12% 高速だが原因は SMEM 半減の間接効果という仮説で、無条件の opt-in 前進は非推奨 | `perf/metal-gemm-n4096-kernel-gap.md` §17 | 条件付きの結論として扱う |
| implicit lazy copy と explicit transfer の対比 | 全バッファが `StorageModeShared`。コピーの暗黙発生はなく、readback は明示的なホスト memcpy | `crates/backend-metal/src/buffer.rs:1-18`（冒頭 doc）、`perf/metal-reuse-readback-2112.md` §3 | §7 で 2 経路を再定義 |
| N=4096 で +12 ms | 出典は確認できない。#1696 の 4 腕診断では +0.20／+0.70／+3.74 ms（N=1024／2048／4096） | `perf/metal-reuse-readback-2112.md` §1 | 基準にしない（#2112 の扱いを踏襲） |
| macOS 14.4 以上限定 | repo で確認済みなのは macOS 26.6.2 で `supportsFamily(Metal4)=true` になる事実のみ | `backend-metal-mpp-tensor-decision.md` §1.1 | 最低版数は**出典未確認**。断定しない（特定 OS 版数に限る採否判断はスコープ外） |

## §3 施策の原因分類表

原因分類の軸（事前に定義）:

- (A) API 非公開・非公式 ABI
- (B) 標準 API の抽象化による制御不能
- (C) 実測で後退
- (D) 実測の安定性ゲート不成立で判定不可
- (E) REQ-8（境界検査）抵触
- (F) REQ-1（完全自作コア）の解釈待ち
- (G) 実機・計測手段の制約（Neural Accelerator 非搭載・GPU counters 非対応）
- (H) 実機実測待ち

| 施策 | 出典 | 現状 | 原因 | 判断 |
|---|---|---|---|---|
| split-K | `backend-metal-splitk-decision.md` §5 | 本番結線済み（#1516。既定 `true`） | - | 適用済み（維持） |
| E1 loop unroll | `perf/metal-gemm-n4096-kernel-gap.md` §7.7a・§7.10.6 | 結線を撤回。条件付き gating も判定不可（`UNROLL_ACC_ENABLED=false` 維持） | D | 非適用。再計測は #2110／#2111 の `UNROLL_LOAD_ENABLED` 系で扱う |
| E2〜E4（特殊化・フラグメントロード・協調ロード） | 同 §9〜§11・§18 | `tile::select` への組み込み対象なし | C／D | 非目標（再訪は #2110 系の結果次第） |
| E5 tgid swizzle／fine barrier | `perf/metal-gemm-tgid-swizzle-ab.md` 状態節・`perf/metal-gemm-fine-barrier-ab.md` | 判定不可（`SWIZZLE_ENABLED=false` 維持） | D | 非適用 |
| E6 タイルクラス分割 | 同 §12 | REJECT | C | 非目標 |
| E7／E8 タイル拡張（`CANDIDATES[9]`・`[10]`） | 同 §13〜§16 | REJECT（[9] は本番選択構成比 4.5〜7.6 倍遅い。§17 の記述による） | C | 非目標 |
| E9 hfrag | 同 §17・`perf/metal-gemm-hfrag-candidate.md` §9 | N=4096 のみ約 10〜12% 高速（仮説段階の間接効果）。前進は非推奨 | C／H | 条件付き（再評価がある場合のみ。現状は非適用） |
| `CANDIDATES[8]`（32,64,16,1,2） | `backend-metal-mlx-classic-nax-decision.md` §1・n4096 §13 | 性能面は REJECT 済み | C | 非目標 |
| NAX／MPP 経路の本番結線 | `backend-metal-mlx-classic-nax-decision.md` §3・`backend-metal-mpp-tensor-decision.md` §6 | 実験実装のみ（診断テスト限定）。採否はユーザー判断待ち | F／G／C | ユーザー判断待ち（それまで非目標扱い） |
| async copy（`simdgroup_async_copy`） | `backend-metal-async-copy-decision.md` | 不採用 | A | 非目標 |
| aligned load（align_M/N/K） | `backend-metal-aligned-load-decision.md` | 不採用（検査短絡型は REQ-8 と衝突） | E | 非目標 |
| Morton（レーンレベル） | `backend-metal-morton-mapping-decision.md` | 適用不可 | B／A | 非目標（tgid レベルは E5 の領分） |
| #2110／#2111 `UNROLL_LOAD_ENABLED` 等 steel 候補 | `perf/metal-gemm-steel-candidates.md` §8・§9 | opt-in・既定 OFF・M4 Max 実測未実施 | H | 条件付き適用 |
| #2112 readback `parallel` | `perf/metal-reuse-readback-2112.md` | opt-in・既定 OFF・実測未実施 | H | 条件付き適用 |
| #2113 train forward encode-only | `perf/framework-compare-phase3-remeasure.md` §2.1 | opt-in・既定 OFF・実測未実施 | H | 条件付き適用 |
| #2114 デバイス存在確認キャッシュ | `perf/metal-tape-build-infer-fixedcost.md` §1 | opt-in・既定 OFF・実測未実施 | H | 条件付き適用 |

## §4 AC1: MPP `matmul2d`（Route C）

- Route C（カーネル内で `device float*` から `tensor_inline` を構築する方式）は `objc2-metal =0.3.2`（`Cargo.toml:138`）から classic
  `MTLComputeCommandEncoder` で起動でき、依存追加・feature 変更を要しない（`backend-metal-mpp-tensor-decision.md` 判断サマリ・§2）
- 純カーネル時間（M4 Max・5 run 中央値。head=MPP／base=本番選択構成）: N=1024 が 1.0223 倍、N=2048 が 1.2632 倍、N=4096 が 1.9015 倍
  （同 §3 表）。REQ-2 複合判定は pass（同 §4）
- 採否は同 §6 の (a) 不採用／(b) 採用可／(c) 限定採用（cfg・環境変数の opt-in・既定 OFF）の**ユーザー判断事項**であり、
  REQ-1 の解釈変更に相当する。**本 doc は決めない**。M4 Max の実測だけでは `tile::select` への組み込みは正当化できない
- 未確認事項の再掲（同 §8）: Route A'（ホスト `MTLTensor`＋argument buffer）・Route B（`MTL4*` コマンド経路）は未実装。
  Metal Toolchain 未導入のため、オフライン `.metallib` 事前ビルド経路の可否は未確認。64×32 以外のタイル構成も未計測

## §5 AC3: 「NAX／行キャッシュ競合」の再評価

**(a) NAX／MPP 経路**: `backend-metal-mlx-classic-nax-decision.md` §3 の再訪条件は「M5 世代実機」「macOS・MPP 可用性」
「classic 経路が REQ-8 未達」の 3 点。M4 Max は Neural Accelerator 非搭載で、#1326 の差は「コンパイラ品質」のみを表し、
N>=2048 で後退している（§4）。よって条件は未充足で、ユーザー判断待ちのまま維持する。

**(b) レジスタフラグメント配置・キャッシュ局所性**: repo の記録は次の範囲にとどまる。
- レーン→要素の対応は標準 `simdgroup_matrix` が隠蔽するため制御不能（Morton 不可。`backend-metal-morton-mapping-decision.md` 判断サマリ）
- threadgroup 単位の局所性（tgid の Z オーダー／swizzle）は E5 で判定不可・結線対象なし（`perf/metal-gemm-tgid-swizzle-ab.md` 状態節）
- フラグメントロード（E3）・協調ロード（E4）は組み込み対象なし（n4096 §18）。hfrag は SMEM 半減の間接効果という仮説（n4096 §17）
- 結論: 標準 API 下でレーン単位の「行キャッシュ競合」を直接検証する手段は repo になく、検証できた範囲ではいずれも決定的な改善がない。
  競合の有無は**確証できない**（GPU counters 非対応。§6）

## §6 AC2: N=4096 頭打ちの代替推定（GPU counters なし）

GPU counters は `xctrace` の `Metal GPU Counters` が「Selected counter profile is not supported on target device」となりデータ 0 行
（`perf/metal-gemm-bottleneck-rediagnosis.md` §5.1・§5.3）。代替証跡を積み上げると次のとおり。

| 段 | 証跡 | 示唆 | 出典 |
|---|---|---|---|
| i | `synchronize_with_gpu_timestamps`（`GPUStartTime`／`GPUEndTime`。`crates/backend-metal/src/context.rs:682`）による純カーネル時間 | e2e から転送・encode を分離できる | n4096 §7〜§19 の計測方式 |
| ii | 並列度 proxy: `actual_groups` は 1024 以降 `ideal_groups` を大きく上回る（4.267 倍〜68.267 倍） | 発行 threadgroup 数の不足ではない | rediagnosis §5.2 |
| iii | N=4096 のカーネル純境界 7.10 TFLOPS に対し e2e 4.40 TFLOPS（純境界は約 1.62 倍・転送が約 38%） | e2e 差の一部は転送。残りがカーネル差 | rediagnosis §5.2・§4.3 |
| iv | タイル全探索: 最良は `CANDIDATES[3]`（32×32×16）。大きいブロックや別 K 刻みは上回らない | タイル選択では埋まらない | rediagnosis §5.2・n4096 §12〜§16 |
| v | E1〜E9 の各施策がいずれも決定的な改善を出さない（§3） | 原因は単一の局所施策では取れない | n4096 §4・§19.0 |
| vi | candle は同種の手書き steel 系カーネルで、unroll 常時 ON・境界検査なしロードを使う（fandhe-ai は REQ-8 で境界検査維持） | 差分の主因候補はカーネル本体の構造差 | `perf/loss-attribution-matrix.md`（M-MTL-G256〜G4096 行） |

**推定**: 頭打ちはタイル選択・並列度不足ではなく、カーネル本体の実行効率（ロード・unroll・同期・レジスタ／SMEM 配置）に起因する。
ただしどの要素が支配的かは GPU counters なしでは**確証できない**（消去法による推定）。

代替手段として `MTLCounterSampleBuffer`（timestamp／statistic／stageUtilization の counter set）が**未プローブの選択肢**である。
M4 Max でどの counter set・サンプリング境界が使えるかは**断定しない**。Mac での probe（`counterSets` の列挙と境界対応可否）を §9 に申し送る。
GPU counters 機構そのものの実装はスコープ外（原因分類 G）。

## §7 AC4: unified memory readback の内訳

Apple Silicon は UMA のため全バッファを `StorageModeShared` で確保しており、Private＋blit は不要と判断済み（PoC-v2-4。
`crates/backend-metal/src/buffer.rs:1-18`）。Issue の「implicit lazy copy」は発生せず、readback は明示的なホスト memcpy（`MetalBuffer::read_to_vec`）である。

| 経路 | 内容 | 現状・数値 | 出典 |
|---|---|---|---|
| (i) 共有バッファ＋ホスト memcpy | 既定は fresh な宛先 `Vec` への `to_vec`。#2112 で分割並列コピー（`FANDHE_AI_METAL_READBACK_DEST=parallel`。8 MiB 以上・最大 8 スレッド）を opt-in で追加 | 宛先が first-touch だと readback が伸びる（BorrowedKeepAlive − LegacyToVec = +0.20／+0.70／+3.74 ms: N=1024／2048／4096）。機構の帰属は仮説段階。`parallel` の効果は**未計測** | `perf/metal-reuse-readback-2112.md` §1・§3・§5 |
| (ii) ゼロコピー | Tensor の実体を MTLBuffer に置く | 未実装。tensor-core のストレージ抽象変更と新規 `unsafe` が必要でユーザー承認事項 | 同 §2 候補 (c) |

宛先再利用・事前タッチは不採用（reuse 窓の出力は tape に保持される／UMA の memcpy では fault が fill へ移るだけ。同 §2）。
推定: 内訳の支配項は first-touch のページフォルト側だが、確証は #2112 の Mac 実測待ち。

## §8 AC5: Phase 3 施策の適用判断と非目標

**条件付き適用**（いずれも各 RULE.txt の判定が ADOPT の場合のみ。結線手順は `perf/metal-gemm-steel-candidates.md` §8 ほか）:
- #2110／#2111: `LU` が ADOPT_CANDIDATE の場合のみ定数 1 個の切替。`T0U` 系は別設計 PR が必要（steel-candidates §8.2・§8.3）
- #2112: `parallel` の A/B が ADOPT の場合のみ既定値切替
- #2113・#2114: 実測仮説が支持された場合のみ既定 ON 化

**非目標**（根拠付き）:
- async copy（A）: 非公開 AIR intrinsic・ハング報告ほか（`backend-metal-async-copy-decision.md`）
- aligned load（E）: 検査短絡型は REQ-8 と衝突（`backend-metal-aligned-load-decision.md`）
- Morton（B）: 標準 API がレーン対応を隠蔽（`backend-metal-morton-mapping-decision.md`）
- E6〜E8 タイル拡張（C）: 実測で REJECT（n4096 §12〜§16）
- GPU counters 機構の実装: 本 Issue のスコープ外（G）
- NAX／MPP の本番結線（F）: ユーザー判断待ち。それまで非目標扱い

**再訪条件**: M5 世代実機の入手、Metal Toolchain の導入、Mac 実測での ADOPT 判定。

## §9 Mac 申し送り

- #2111: `docs/perf/logs/metal-gemm-candidate-ab-2111/`（`RULE.txt`・`orchestrate.sh`・`aggregate.py`）
- #2112: `docs/perf/logs/metal-reuse-readback-2112/`
- #2113: `docs/perf/logs/metal-train-forward-encodeonly-2113/`
- #2114: `docs/perf/logs/metal-tape-build-infer-phase-2114/`
- #2120: `docs/perf/logs/framework-compare-phase3-remeasure-2120/`（両機体再計測・スコアボード再生成）
- 新規案（未起票）: `MTLCounterSampleBuffer` の counter set probe（`counterSets` 列挙・サンプリング境界の対応確認。結果は推定の裏取りにのみ使う）

## §10 スコープ外・起票案（起票はユーザー承認待ち）

- MPP の直接利用・本番結線（Route A'／B・オフライン `.metallib`・他タイル構成の計測を含む）
- GPU counters 機構の代替実装
- 特定 macOS 版数に限る機能の採用判断
- ゼロコピー readback（#2112 候補 (c)）の設計・`unsafe` の承認

## §11 参照

- `docs/backend-metal-mpp-tensor-decision.md`（#1326）・`docs/backend-metal-mlx-classic-nax-decision.md`（#549）
- `docs/backend-metal-morton-mapping-decision.md`（#544）・`docs/backend-metal-async-copy-decision.md`（#546）
- `docs/backend-metal-aligned-load-decision.md`（#752／#808）・`docs/backend-metal-splitk-decision.md`（#810／#1516）
- `docs/perf/metal-gemm-n4096-kernel-gap.md`・`docs/perf/metal-gemm-bottleneck-rediagnosis.md`・`docs/perf/metal-gemm-tgid-swizzle-ab.md`
- `docs/perf/metal-gemm-steel-candidates.md`・`docs/perf/metal-reuse-readback-2112.md`・`docs/perf/metal-tape-build-infer-fixedcost.md`
- `docs/perf/framework-compare-phase3-remeasure.md`・`docs/perf/loss-attribution-matrix.md`
