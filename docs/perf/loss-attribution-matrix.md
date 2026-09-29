# 負けセル分析・原因帰属・未試行施策の対照表（Phase 3 入口）

イシュー #2098（親 #2089「他ライブラリのコード取得・詳細解析」の Phase 2 集約。後続は Phase 3 #2099〔子 #2100〜#2120〕）。
`docs/analysis/` の 8 本の解析結論を「セル × 原因 × 施策」の形へピボット集約した master doc。
**docs 専用・新規計測なし・コード変更なし**（tolerance・baseline・`Cargo.toml`・ガードレール閾値・`docs/spec/` は一切変えない）。

## 1. 位置づけ・解析契約

| 項目 | 内容 |
|---|---|
| 目的 | どの負けセルが、どの原因で、どの施策（どの Phase 3 issue）で改善しうるかを 1 か所で引けるようにし、Phase 3 の優先度付け・並列化判断の根拠にする |
| 入力 | スコアボード 2026-09-19 版（§2）・`docs/analysis/` 8 本（§9）・phase 実測 2 本（`docs/perf/train-step-phase-breakdown.md` §17・`docs/perf/infer-reuse-phase-breakdown.md` §10）・Phase 3 各 issue の本文 |
| 契約 1 | 他ライブラリのコード・シェーダ・派生物を持ち込まない。書くのは結論・出典・ライセンス名だけ |
| 契約 2 | 閉源部分（cuBLAS／MPS／tensorflow-metal／Accelerate）はディスパッチ層までが到達範囲。本 doc はこれらを `CLOSED`（到達不能）とだけ記す |
| 契約 3 | 既存の REJECT／undetermined と重複する実験は提案しない。§5 表 B は既存判定の**再掲**であり再実行の提案ではない |
| 契約 4 | **証拠レベルを格上げしない**。解析 doc が仮説・推定・未検証と書いた事項は、本 doc でも `推定`／`未確定` のまま扱う（§4.1） |
| 非目標 | Phase 3 の詳細設計・実装・実測。spec 提案は分岐として記すだけ（§5.4） |

## 2. 出典と判定規則

### 2.1 スコアボード（2026-09-19 版）

- 正本: `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.out`（生成器 `gen_1988.py`・本文 `body_1988.html`。README・`aggregate.md` は同ディレクトリの親）。claude.ai の Artifact URL はリポジトリ外のためデータ出典にしない
- 判定規則（`gen_1988.py` の行番号）:
  - `:140-150` 主表示は fandhe-ai の **`reuse` 行**（`me = data.get((…, 'reuse'))`。存在しなければ assert で停止するため、下表の 22 セルはすべて reuse 行の値）
  - `:176-186` 順位・ratio の定義。gemm／train は「最速他 FW の所要時間 ÷ fandhe-ai の所要時間」、infer は metric（1/median）の向きが逆。1 位以外で ratio ≥ 0.90 なら「僅差」
  - `:160-163`・`:237-246` 判定不能（要素検証超過）は**比較相手のセル単位で順位から除外**する
  - `:189-200` 「0.8.0 比」は `gemm`／`train` は `base.median_s / old.median_s`、`infer` は metric の比として計算される（向きがタスクで異なる）。本 doc は転記のみとし、増減の評価には使わない
- 計測条件: M4 Max は 0.9.0 系列 B の 5 run 中央値（事前登録規則は `docs/perf/logs/framework-compare-0.9.0-remeasure/README.md`）。GB10 は専有 1 セッション（burn cuda 5 行と PyTorch cpu N=4096 のみ #1988 の再計測値。`docs/perf/logs/framework-compare-precision-class-remeasure-1988/README.md`）。Python 3 FW（PyTorch・TensorFlow・SciPy）は 2026-09-12 の値の流用で、M4 の TF／SciPy は転記値（`gen_1988.py:82-86`。`docs/analysis/tensorflow-2.16-2.21-scipy-cpu-path.md` §10）

### 2.2 集計

`gen_1988.out` 末尾（28 行目）: `tally 5 2 20 invalid 1 total 27`。判定対象 27 行のうち **勝ち 5・僅差 2・負け 20**、判定不能 1。

| 区分 | セル |
|---|---|
| 勝ち 5 | M4 CPU gemm N=512／GB10 CUDA gemm N=4096／GB10 CPU gemm N=1024・2048・4096（`gen_1988.out:7`・`:18`・`:21`・`:22`・`:23`） |
| 僅差 2 | M4 Metal gemm N=256（0.93×）・M4 CPU gemm N=1024（0.91×） |
| 負け 20 | 下表 §3 の残り 20 セル |
| 判定不能 1 | GB10 の PyTorch cpu gemm N=4096（比較相手側セル。`parity_fail_count=1`・#1989 承認 T1。fandhe-ai 側の 1 位判定には影響しない。README の結果表） |

## 3. 負けセル一覧（受入 1）

本 doc は負け 20＋僅差 2 の **22 セル**を扱う。ratio・順位・最速相手・0.8.0 比は `gen_1988.out` の該当行の転記（ratio は最速他 FW ÷ fandhe-ai。1.00 未満が負け側）。fandhe-ai 側のモードはすべて `reuse`（§2.1）。

| セル ID | 機体 | 対象 | 順位 | 最速相手 | ratio | 0.8.0 比（転記） | 出典行 |
|---|---|---|---|---|---:|---:|---|
| M-MTL-G256 | M4 Max | gemm Metal N=256 | **僅差** | candle | 0.93 | 1.14 | `gen_1988.out:1` |
| M-MTL-G512 | M4 Max | gemm Metal N=512 | 2 位 | candle | 0.72 | 1.13 | `:2` |
| M-MTL-G1024 | M4 Max | gemm Metal N=1024 | 3 位 | candle | 0.68 | 1.20 | `:3` |
| M-MTL-G2048 | M4 Max | gemm Metal N=2048 | 2 位 | candle | 0.74 | 1.04 | `:4` |
| M-MTL-G4096 | M4 Max | gemm Metal N=4096 | 2 位 | candle | 0.59 | 1.21 | `:5` |
| M-CPU-G256 | M4 Max | gemm CPU N=256 | 2 位 | pytorch | 0.66 | 1.01 | `:6` |
| M-CPU-G1024 | M4 Max | gemm CPU N=1024 | **僅差** | pytorch | 0.91 | 1.04 | `:8` |
| M-CPU-G2048 | M4 Max | gemm CPU N=2048 | 3 位 | pytorch | 0.73 | 1.07 | `:9` |
| M-MTL-TRN | M4 Max | train Metal | 3 位 | pytorch | 0.38 | 0.77 | `:10` |
| M-MTL-INF | M4 Max | infer Metal | 2 位 | pytorch | 0.79 | 1.66 | `:11` |
| M-CPU-TRN | M4 Max | train CPU | 6 位 | pytorch | 0.21 | 0.97 | `:12` |
| M-CPU-INF | M4 Max | infer CPU | 3 位 | pytorch | 0.18 | 1.07 | `:13` |
| G-CUDA-G256 | GB10 | gemm CUDA N=256 | 2 位 | candle | 0.83 | 1.02 | `:14` |
| G-CUDA-G512 | GB10 | gemm CUDA N=512 | 2 位 | candle | 0.43 | 1.03 | `:15` |
| G-CUDA-G1024 | GB10 | gemm CUDA N=1024 | 3 位 | candle | 0.41 | 1.06 | `:16` |
| G-CUDA-G2048 | GB10 | gemm CUDA N=2048 | 3 位 | burn | 0.48 | 1.05 | `:17` |
| G-CPU-G256 | GB10 | gemm CPU N=256 | 3 位 | pytorch | 0.72 | 0.97 | `:19` |
| G-CPU-G512 | GB10 | gemm CPU N=512 | 2 位 | candle | 0.76 | 1.07 | `:20` |
| G-CUDA-TRN | GB10 | train CUDA | 2 位 | candle | 0.88 | 0.67 | `:24` |
| G-CUDA-INF | GB10 | infer CUDA | 3 位 | pytorch | 0.41 | 1.42 | `:25` |
| G-CPU-TRN | GB10 | train CPU | 3 位 | pytorch | 0.34 | 0.91 | `:26` |
| G-CPU-INF | GB10 | infer CPU | 2 位 | pytorch | 0.72 | 0.91 | `:27` |

内訳: M4 Max 12 セル（Metal gemm 5・CPU gemm 3・train／infer 4）＋ GB10 10 セル（CUDA gemm 4・CPU gemm 2・train／infer 4）。僅差を除く負けは 20（M4 Metal gemm 4・M4 CPU gemm 2・M4 train／infer 4・GB10 CUDA gemm 4・GB10 CPU gemm 2・GB10 train／infer 4）。

補足:

- **モード注記**: `docs/analysis/tensorflow-2.16-2.21-scipy-cpu-path.md` §10 の表 1・表 2 は fandhe-ai の `fresh` 値（0.8.0）で TF／SciPy と突合している。本表の判定は 0.9.0 の `reuse` 値であり、モードもバージョンも異なる。同 doc の「GB10 の CPU セルに負けなし」は相手が TF／SciPy の場合の結論で、0.9.0 スコアボードの G-CPU 各セルの最速相手は pytorch／candle である
- **監視セル（負けに数えない）**: GB10 CUDA gemm N=4096 は burn 比 1.03×（1 位・`gen_1988.out:18`）。ただし 2026-09-16 の fandhe-ai 行と 2026-09-18 の burn 行を比べたセッション混在比で、同一 run 内比（`aggregate.md`）では最速相手 burn の中央値が 1.0145 と 1 に近い。回帰の兆候があれば #2108／#2120 で再確認する
- **train のセル**: TF 解析 doc §10 が示すとおり、比較対象と fandhe-ai で backward の GEMM 回数が異なる（§4.3 の LOOP）。train セルの ratio は演算量が揃った比較ではない

### 3.1 件数の突合（Issue 記載の 21 との関係）

- #2098 本文は「負けセル 21 個（M4 Metal 4・M4 CPU 3・M4 infer 2・GB10 CUDA 4・GB10 CPU 2・GB10 train/infer 4・GPU 合計 4 他）」と書く。この内訳を足すと 19＋「GPU 合計 4 他」となり、21 にも `gen_1988.out` の負け 20 にも一致しない（重複の有無が本文から判別できない）
- 一致する部分: M4 Metal gemm 4・GB10 CUDA gemm 4・GB10 CPU gemm 2・GB10 train／infer 4 は本表の内訳と同数。M4 CPU 3 は N=256・1024〈僅差〉・2048 の 3 セルと数え方が合う（僅差を含めた場合）。M4 infer 2 は本表の M4 infer（Metal・CPU）の 2 セルと合う
- 方針: **`gen_1988.out` の 20＋2 を正とし、21 に合わせる調整はしない**。Issue 起票時の集計と `gen_1988.out` の差の原因は特定不能（起票時の集計元は本リポジトリに残っていない）。件数は Phase 3 の対象範囲を変えない（22 セルすべてを §4 に載せた）

## 4. 原因帰属対照表（受入 2）

### 4.1 カテゴリと証拠レベル

| カテゴリ | 意味 |
|---|---|
| `LOOP` | ループ構造差（backward の dX 計算の有無・同期点の位置〈`loss_readout`〉・ホスト往復〈`param_readout`／`apply_params`〉・毎 step の tape／ノード生成） |
| `FUSE` | 融合欠如（epilogue／elementwise 融合） |
| `FIX` | 固定費（launch・dispatch 段数・rayon fork-join・`tape_build`） |
| `ALLOC` | alloc 効率（バッファの毎回確保・readback 宛先の確保／ページフォールト） |
| `PREC` | 精度クラス差（burn の TF32 Tensor Core 対 fandhe-ai の FP32 SIMT） |
| `BOUND` | 計測境界差（fresh／reuse・readout の位置） |
| `KERN` | カーネル構成差（タイル・unroll・ロード方式・async copy） |
| `THREAD` | スレッド方針差（P コア限定・プール分離） |
| `CLOSED` | 閉源到達限界（MPS／cuBLAS／Accelerate。ディスパッチ層より内側は不明） |

帰属 1 件ごとに証拠レベルを付ける。

| タグ | 意味 |
|---|---|
| `確定（ソース根拠）` | 他 FW のソースまたは fandhe-ai の HEAD／ログ上で事実として確認できる。ただし**そのセルの負けに何 % 寄与するか**は含まない |
| `推定` | 解析 doc が仮説・推定と明記している、または phase 実測から示唆されるが検証していない |
| `未確定` | 候補として挙がっているだけで裏付けがない（解析 doc が「未検証」「未確定」とした事項を含む） |
| `到達不能` | 閉源部で確認できない |

**寄与率はどのセルについても未計測**。本表は「何が原因候補か」と「どの程度確かか」だけを示す。

### 4.2 GEMM セル

| セル ID | 主 | 副 | 根拠 | 証拠レベル |
|---|---|---|---|---|
| M-MTL-G256／G512／G1024／G2048／G4096 | `KERN` | `BOUND`（N=4096 のみ） | 最速相手 candle の Metal GEMM は cuBLAS／MPS ではなく手書き MLX steel 系カーネルで、fandhe-ai と同種の土俵（`docs/analysis/candle-metal-01.md` §2）。candle は `STEEL_PRAGMA_UNROLL` を常時 ON・`load_unsafe`／`load_safe`（境界検査なしロード）を使う一方、fandhe-ai は unroll を opt-in・既定 OFF、境界検査は REQ-8 で維持（同 §6）。タイル選択でなくカーネル本体の差とする既存実験の結論を `docs/analysis/mlx-v0.32.2-steel-gemm.md` §8 が引用。N=4096 の readback 後退（+12 ms）は `docs/perf/metal-readout-legacy-regression-four-arm-diag.md`・#1696 が「機構帰属は仮説」と結論 | `KERN`: `推定`（差分の存在は `確定（ソース根拠）`、負けへの寄与は未検証）／`BOUND`: `未確定` |
| M-CPU-G256 | `CLOSED` | `THREAD`・`FIX` | 最速相手 pytorch の M4 経路は Accelerate の Fortran `sgemm_`（ディスパッチまで `確定`・内部は閉源。`docs/analysis/pytorch-2.14-cpu-path.md` §4）。PyTorch の既定スレッド数は macOS+aarch64 で P コア数（M4 Max で 12。同 §6.1）。fandhe-ai は P コア限定 OFF（`BIG_CORE_LIMIT_ENABLED=false`。同 §8）。N=256 は最小形状で fork-join 等の固定費が相対的に大きい | `CLOSED`: `到達不能`／`THREAD`: `推定`（`docs/analysis/candle-cpu-03.md` §7.1 仮説 A も「未実測」）／`FIX`: `未確定` |
| M-CPU-G1024（僅差）・M-CPU-G2048 | `CLOSED` | `THREAD` | 同上。N=1024 は 0.91× で僅差、N=2048 は 0.73× | 同上 |
| G-CUDA-G256 | `FIX` | `CLOSED` | 相手 candle は cuBLAS 1 段呼び出し、fandhe-ai は規則選択→実装選択→NVRTC JIT の多段（`docs/analysis/candle-0.11.0-cuda-path.md` §8・§9-3）。ただし §9-3 は `launch_issue` 自体が 0.005〜0.008 ms と小さく優先度低とする | `FIX`: `未確定`／`CLOSED`: `到達不能` |
| G-CUDA-G512 | `ALLOC`／`BOUND`（readback） | `CLOSED` | `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.8 の未説明分（`iter_total` の 46〜56%）は N=1024／2048／4096 の値で、N=512 には直接の値がない。candle は readback 宛先の事前タッチをせず未同期の暗黙ブロッキング DtoH のみ（candle-cuda §8） | `未確定`（candle-cuda §9-1 が「未検証の問い」） |
| G-CUDA-G1024 | `ALLOC`／`BOUND`（readback） | `PREC`・`CLOSED` | 同上（§12.8 の 46〜56% の対象形状）。burn は TF32 Tensor Core を強制（burn-cubecl §5）で、fandhe-ai の GEMM ベンチは既定 FP32 厳密（`--tf32` 既定 false）。本セルは candle・burn の 2 者に負けている | readback: `未確定`／`PREC`: `推定`（burn doc §8(a)「優先度最高」だが実測は Phase 3 送り） |
| G-CUDA-G2048 | `PREC` | `BOUND` | 最速相手 burn。`docs/analysis/burn-cubecl-cuda-matmul.md` §8 は (a) 精度クラス差を優先度最高、(b) 計測境界差（burn は clone・matmul・readout を計測窓内で行う fresh 型）を挙げる。同一 run 内比では burn 0.4811・candle 0.5075 と近接（同 §8(d)） | `PREC`／`BOUND`: `推定`（同 doc「実測による確定は Phase 3 が引き継ぐ」）|
| G-CPU-G256 | `FIX` | `THREAD`・`CLOSED` | 最速相手 pytorch の GB10 経路は NVPL の `sgemm_` と**推定**（実行時未確認。pytorch §5・§10）。GB10 の PyTorch は論理コア数 20 を使用（同 §6.1）。fandhe-ai の GB10 affinity は既定 OFF・undetermined（`docs/perf/cpu-gemm-gb10-affinity-ab.md`） | `FIX`: `未確定`／`THREAD`: `推定`／`CLOSED`: `到達不能` |
| G-CPU-G512 | `FIX` | `THREAD` | 最速相手 candle（Rust `gemm` crate）。candle-cpu §7.4 は M4 と GB10 の非対称を参考実測として記す。分配は fandhe-ai が本番既定 `TwoDDynamic`（ADOPT 済み。candle-cpu §8） | `未確定` |

### 4.3 train／infer セル

phase 実測（`docs/perf/train-step-phase-breakdown.md` §17.2・§17.6.2、`docs/perf/infer-reuse-phase-breakdown.md` §10.2・§10.6.2）の reuse 行の支配項を併記する。scoreboard の判定は reuse 行なので、fresh 固有の区間（`param_readout`・`host_sgd`・`apply_params`）は本表の判定セルには直接寄与しない。

| セル ID | 主 | 副 | reuse の phase 実測（支配項） | 根拠 | 証拠レベル |
|---|---|---|---|---|---|
| M-CPU-TRN（0.21×・6 位） | `LOOP` | `FIX`・`CLOSED` | backward 59.0%（520.7 µs）・forward_resident 26.9%・device_update 13.9%。step_total 881.9 µs | PyTorch・burn は backward で第 1 層の入力勾配 dX を計算しない（3 GEMM）。fandhe-ai は HEAD の `Op::MatMul` VJP が `da`・`db` を両方計算してから `requires_grad == false` の寄与を捨てる形（`crates/autodiff/src/grad.rs:213`・`matmul_vjp`〈`:5569`〉と `backward.rs:347-360` の静的読取り。実測なし）。`docs/analysis/train-infer-loop-comparison.md` §4.2・§4.3・§6。PyTorch は要素ごと演算に GRAIN_SIZE の逐次フォールバックを持ち、fandhe-ai は常に rayon（pytorch §7.1・§7.3）。GEMM 本体は Accelerate（`CLOSED`） | `LOOP`（dX）: 相手側の事実は `確定（ソース根拠）`・fandhe-ai 側の無駄計算の有無は静的読取りで `推定`・寄与は `未確定`／`FIX`: `推定`／`CLOSED`: `到達不能` |
| M-CPU-INF（0.18×・3 位） | `FIX` | `CLOSED`・`ALLOC` | predict_resident が iter_total の 99.6〜99.7%（175.3 µs） | 同じ 784→256→10 の forward 2 GEMM。fandhe-ai の CPU infer reuse は GEMM 以外の区間が事実上なく、PyTorch 側との差は GEMM ディスパッチ（Accelerate）と小形状固定費に帰着しうる。ALLOC は PyTorch も毎回確保でキャッシュなし（pytorch §7.2）のため fandhe-ai 固有の差ではない | `FIX`: `未確定`／`CLOSED`: `到達不能`／`ALLOC`: `未確定` |
| M-MTL-TRN（0.38×） | `LOOP`・`FIX` | `CLOSED` | backward 49.4%（524.6 µs）・forward_resident 48.6%（516.3 µs）・tape_build 1.4%（14.9 µs）。step_total 1061.6 µs | forward の encode-only 化余地は train doc §17.4 が「要精査」と記す（近縁: `docs/backend-metal-command-batching-design.md`）。tape_build は cpu の 2 桁大（train doc §17.3）。PyTorch 側 GPU 実装は MPS（`CLOSED`） | `LOOP`／`FIX`: `推定`（train doc §17.4 が forward の encode-only 化余地を「要精査」とする）／`CLOSED`: `到達不能` |
| M-MTL-INF（0.79×） | `FIX` | `CLOSED` | predict_resident 99.8%（365.2 µs。同 CPU reuse の約 2.1 倍） | infer doc §10.3 は「GPU 起動固定費が支配的と推定される」（**推定であり検証していない**） | `推定` |
| G-CPU-TRN（0.34×） | `LOOP` | `FIX`・`CLOSED` | backward 44.7%（482.7 µs）・device_update 25.7%（277.5 µs）・forward_resident 23.8%。step_total 1080.6 µs | M-CPU-TRN と同じ dX の構造差。加えて reuse の `device_update` が Mac の約 2.3 倍（train doc §17.6.3・§17.6.4）。PyTorch の GB10 経路は NVPL と推定（`CLOSED`） | `LOOP`: 同 M-CPU-TRN／`device_update` の内訳: `未確定`（doc が原因切り分けを未実施と明記） |
| G-CPU-INF（0.72×） | `FIX` | `CLOSED` | predict_resident 195.0 µs（fresh 176.3 µs より遅い逆転。infer doc §10.6.3） | reuse が fresh より遅い原因は未切り分け（run 間の min–max 幅が大きく負荷変動の寄与も未分離。infer doc §10.6.4） | `未確定` |
| G-CUDA-TRN（0.88×・candle に僅差） | `FIX`（forward_resident） | `LOOP` | forward_resident 49.1%（155.9 µs）・backward 46.9%（149.1 µs）・device_update 2.5%。step_total 317.5 µs | 相手 candle は dX を無条件計算する（train doc §4.1）ため、本セルでは dX の構造差は fandhe-ai の負けを説明しない。forward_resident が backward と拮抗する点の内訳は未取得（train doc §17.6.4） | `未確定` |
| G-CUDA-INF（0.41×） | `FIX` | `CLOSED` | predict_resident 98.9%（98.8 µs） | 最速相手 pytorch の CUDA 経路は cuBLAS 等（`CLOSED`）。`predict_resident` の内部内訳は未取得（infer doc §10.6.5）。単一同期チェーン（`predict_device_chain`）は既に ADOPT 済み（`docs/perf/infer-chain-single-sync-cuda-ab.md`） | `FIX`: `未確定`／`CLOSED`: `到達不能` |

## 5. 未試行施策一覧と負けセル対応（受入 3）

「既存判定」の語彙は各出典の記載どおり（ADOPT／REJECT／undetermined／承認保留／本番ゲート OFF／新規）。**新規**は既存記録に見当たらない意味で、効果の保証ではない。

### 5.1 表 A: Phase 3 の既存 issue への割り当て

デバイス依存性: M4＝M4 Max のみ、GB10＝DGX Spark のみ、両＝両機体、なし＝実機不要（docs・CI）。「対象ファイル」は各 issue 本文の「対象範囲」の記載で、実在確認は §6 に記す。

| Phase 3 issue | 施策 | 対象セル | カテゴリ | 既存判定・注意 | 実機 | 対象ファイル（issue 本文） |
|---|---|---|---|---|---|---|
| #2100 | backward 非 GEMM 内訳の診断計装（`diag-instrumentation.patch` の HEAD 再適用） | M-CPU-TRN・G-CPU-TRN・M-MTL-TRN・G-CUDA-TRN | `LOOP`・`FIX` | 診断のみ（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 の再実測）。#2101 の前提 | 両 | `crates/backend-*/` の backward 経路・`bench-fandhe` |
| #2101 | 小形状 elementwise・縮約の fork-join しきい値化（bit 同一・opt-in） | M-CPU-TRN・M-CPU-INF・G-CPU-TRN・G-CPU-INF | `FIX` | `mse_loss_backward` への同型適用は REJECT 済みで禁止（`docs/perf/cpu-mse-backward-sequential-threshold.md`）。`docs/perf/elementwise-vjp-backend-ops.md` の REJECT（ルーティング・bit 後退）とは経路が別 | 両 | `crates/autodiff/src/grad.rs`・`crates/backend-cpu/src/ops.rs` |
| #2102 | #2101 の両機体 A/B・結線判断 | 同上 | `FIX` | 結線は ADOPT 判定後 | 両 | — |
| #2103 | tape・中間バッファ arena 再利用の設計 | M-CPU-TRN・G-CPU-TRN・M-CPU-INF・G-CPU-INF | `ALLOC` | GEMM パネルの alloc 固定費は REJECT 済み（`docs/perf/cpu-matmul-fixed-cost-impl.md`。#1481／#1482）。arena は tape 層で、REJECT 対象と別（issue 本文に区別の記載なし。設計 doc 側で明確化が必要） | なし | 設計のみ |
| #2104 | arena 実装＋両機体 A/B | 同上 | `ALLOC` | opt-in・既定 OFF。新規 `unsafe` の要否は設計次第（承認要） | 両 | issue 本文に `crates/tensor-core/src/alloc.rs`（新規予定）・`crates/autodiff/src/tape.rs` |
| #2105 | CPU `predict_resident` 固定費（GB10 reuse<fresh 逆転）の切り分け | G-CPU-INF・M-CPU-INF | `FIX` | 新規（infer doc §10.6.4） | 両 | issue 本文は `crates/facade/src/predict_resident.rs` だが実在せず、実体は `crates/facade/src/compat/sequential.rs` |
| #2106 | cpu reuse `device_update`（GB10 277 µs）の内訳切り分け | G-CPU-TRN・M-CPU-TRN | `LOOP`・`FIX` | 新規（train doc §17.6.4）。近縁: `docs/perf/train-resident-grad-device-update.md`（#1212） | 主に GB10 | `crates/backend-cpu/src/ops.rs`・`crates/autodiff/src/optim/sgd.rs` |
| #2107 | CUDA readback 宛先確保の帰属検証（Layer B 計装） | G-CUDA-G512・G1024・（G2048） | `ALLOC`・`BOUND` | `PretouchedFresh` は実装済み（`docs/perf/cuda-host-view-readout-small-shape-regression.md`）。未説明分の帰属確定が目的 | GB10 | `crates/backend-cuda/src/memory.rs` |
| #2108 | readback 宛先の再利用・ゼロコピー化（opt-in） | 同上 | `ALLOC` | #2107 の結果次第。ADOPT で結線 | GB10 | `crates/backend-cuda/src/memory.rs` |
| #2109 | CUDA N=256 の起動固定費（launch 回数・sync）診断 | G-CUDA-G256 | `FIX` | 新規。**issue 本文は「N=256 の 0.72×」と書くが、スコアボードの G-CUDA-G256 は 0.83×（0.72× は G-CPU-G256）。数値の転記違いの可能性（§8）** | GB10 | `crates/backend-cuda/src/gemm.rs`（`internal-diagnostics`） |
| #2110 | candle／MLX 解析の差分からの未試行 Metal GEMM 候補（opt-in 実装） | M-MTL-G256〜G4096 | `KERN` | 候補は candle-metal §6 の 1〜4（unroll 常時 ON・ロード方式・smem swizzle と E3 の組合せ・async copy）。REQ-8 の手動境界検査は維持（`docs/backend-metal-aligned-load-decision.md`）。async copy は不採用の既存決定あり（`docs/backend-metal-async-copy-decision.md`）で、差分は「candle 側の確認済み不在との突合の深掘り」のみ | M4 | `crates/backend-metal/src/gemm.rs`・`shaders/gemm.metal` |
| #2111 | 候補の kernel_gpu 5 run 実測・結線判断 | 同上 | `KERN` | 判定規則は issue 本文（正方 4 形状のうち 2 形状以上で ratio<1.00 かつ checksum 一致）。REJECT 候補の再検討はスコープ外 | M4 | `crates/backend-metal/src/tile.rs`・`gemm.rs` |
| #2112 | Metal reuse 計測窓の readback（N=4096 で +12 ms）の対処 | M-MTL-G4096・M-MTL-INF | `BOUND`・`ALLOC` | #1696 の 4 腕診断が前提（「なぜ legacy が遅いか」は別 issue） | M4 | `crates/backend-metal/src/memory.rs` |
| #2113 | Metal train forward の encode-only 化 | M-MTL-TRN | `LOOP`・`FIX` | 新規（train doc §17.4）。近縁: `docs/backend-metal-command-batching-design.md` §7（対象外の区間） | 両（issue 本文） | `crates/backend-metal/src/gemm.rs` |
| #2114 | Metal `tape_build` 削減＋infer GPU 起動固定費の診断カウンタ | M-MTL-TRN・M-MTL-INF | `FIX` | 新規。近縁: `docs/perf/metal-infer-chain-single-sync.md` | M4 | issue 本文は `crates/autodiff/src/tape.rs`（Metal 向け） |
| #2115 | CUDA 推論チェーンの CUDA Graph capture（opt-in） | G-CUDA-INF | `FIX` | update 区間 capture（#1349）とは別機構（`docs/backend-cuda-graph-step-capture-design.md`）。新規 `unsafe` の要否は承認要 | GB10 | `crates/backend-cuda/src/graph.rs` |
| #2116 | CUDA train `forward_resident` 内訳・fresh `param_readout` 削減 | G-CUDA-TRN | `FIX` | 新規（train doc §17.6.4）。**fresh の `param_readout` は scoreboard の判定行（reuse）に効かない**（§4.3） | GB10 | `crates/backend-cuda/src/ops.rs`・`crates/autodiff/src/optim/device_store.rs` |
| #2117 | GB10 大コア affinity（#1576）の正式 A/B | G-CPU-G256・G512・G-CPU-TRN・G-CPU-INF | `THREAD` | undetermined（`docs/perf/cpu-gemm-gb10-affinity-ab.md`）の決着。P コア限定そのものは GB10 で REJECT 済みの別機構（§5.2） | GB10 | `crates/backend-cpu/src/gb10_affinity.rs` |
| #2118 | SME しきい値（`SME_MIN_K` 等）の M4 Max 再実測＋GB10 非後退確認 | M-CPU-G256・G1024・G2048（と CPU train／infer の GEMM 部分） | `CLOSED` への対抗・`KERN` | `SME_PRODUCTION_ENABLED=false`（本番ゲート OFF。`crates/backend-cpu/src/gemm_blis/mod.rs:3085`）。#1979 再開条件 1・2 | 両 | `crates/backend-cpu/src/gemm_blis/mod.rs` |
| #2119 | SME `unsafe asm!` の監査記録と本番化の再承認申請 | 同上 | — | **承認要**（新規 `unsafe` の正当性判定と本番化。承認までは既定 OFF 維持） | なし | `crates/backend-cpu/src/gemm_blis/microkernel/sme.rs`（issue 本文は `sme.rs` と略記） |
| #2120 | 両機体の再計測・スコアボード再生成 | 全セル | — | 全 issue の採否決定後。負けセルの Phase 3 前後比較を記録 | 両 | `scripts/bench/framework-compare/`・`gen_*.py` |

### 5.2 表 B: 既存判定の再掲（再実行しない）

| 施策 | 既存判定 | 出典 | 関連セル |
|---|---|---|---|
| KC 再スイープ（128〜512 グリッド） | REJECT | `docs/perf/cpu-gemm-candle-cpu-retune.md` §8.1 | CPU gemm |
| B laneq ベクトル転置 | REJECT | 同 §8.2 | CPU gemm |
| P コア限定スレッド数（`BIG_CORE_LIMIT_ENABLED`） | REJECT（GB10 で重大後退。M4 Max 単体では改善方向） | `docs/perf/cpu-gemm-default-thread-limit.md` §6 | M-CPU-G* |
| 小形状 thread cap（`SMALL_SHAPE_CAP_ENABLED`） | REJECT | `docs/perf/cpu-gemm-small-shape-thread-cap.md` | CPU gemm 小形状 |
| GEMM 小形状の直列フォールバック（`GEMM_THREADING_THRESHOLD`） | 本番未結線 | `docs/perf/cpu-gemm-small-shape-serial-fallback.md` | 同上 |
| `mse_loss_backward` の逐次しきい値 | REJECT（GB10 reuse 1.0167×） | `docs/perf/cpu-mse-backward-sequential-threshold.md` | CPU train |
| GEMM PanelBuffers の alloc 固定費削減 | REJECT | `docs/perf/cpu-matmul-fixed-cost-impl.md`（#1481／#1482） | CPU gemm・train・infer |
| prefetch（`asm!` PRFM） | 承認保留（`unsafe` 新規導入との整合） | `docs/cpu-gemm-prefetch-decision.md` | CPU gemm |
| Metal E1〜E9・split-K・thread_elements・NAX／MPP・morton・serpentine 等 | REJECT／判定不可／既定 OFF（個別は出典参照） | `docs/analysis/candle-metal-01.md` §6・`docs/perf/metal-gemm-n4096-kernel-gap.md` | M-MTL-G* |
| Metal `(32,64,16,1,2)`（`CANDIDATES[8]`） | REJECT（劣後） | `docs/perf/metal-gemm-n4096-kernel-gap.md` §13・`docs/analysis/mlx-v0.32.2-steel-gemm.md` §8 | M-MTL-G* |
| CUDA Stream-K・persistent kernel・TMA（Stage 1） | REJECT／調査済み | `docs/perf/cuda-gemm-tiled-pipeline-streamk.md` ほか（`docs/analysis/burn-cubecl-cuda-matmul.md` §7 の突合表） | G-CUDA-G* |
| GB10 affinity（既定 OFF） | undetermined（決着は #2117） | `docs/perf/cpu-gemm-gb10-affinity-ab.md` | G-CPU-* |
| 2D 動的分配（`TwoDDynamic`） | ADOPT 済み | `docs/cpu-gemm-2d-dynamic-partition-design.md` | CPU gemm |
| 学習の resident 勾配・device 常駐更新 | ADOPT 済み（reuse 経路） | `docs/perf/train-resident-grad-device-update.md` | train reuse |
| 推論チェーン単一同期化 | ADOPT 済み | `docs/perf/infer-chain-single-sync-cuda-ab.md`・`docs/perf/metal-infer-chain-single-sync.md` | infer GPU |

### 5.3 表 C: 起票候補（**未起票・ユーザー承認要**）

Phase 3 のどの issue にも対応しない差分候補。起票・実施はしない（`.claude/rules/out-of-scope-tracking.md`）。

| ID | 候補 | 対象セル | 出典 | 既存判定・注意 |
|---|---|---|---|---|
| C-1 | **backward の dX（第 1 層入力勾配）GEMM の省略**。PyTorch・burn は計算せず、fandhe-ai は HEAD の `matmul_vjp` が `da`・`db` を無条件に計算し `requires_grad == false` の寄与を後から捨てる形に見える（静的読取り。実測・確認なし） | M-CPU-TRN・G-CPU-TRN（相手 pytorch）ほか train 全般 | `docs/analysis/train-infer-loop-comparison.md` §4.2・§4.3・§8 新規候補 1・`docs/analysis/tensorflow-2.16-2.21-scipy-cpu-path.md` §10・`crates/autodiff/src/grad.rs:213` | 既存記録（`docs/perf/train-backward-gemm-wiring.md`）には `dX`・`d_input`・入力勾配のいずれの語でも言及が見当たらない。bit 一致契約への影響は要確認。`grad.rs` は #2101 と対象ファイルが重複（§6） |
| C-2 | GEMM 用と要素ごと演算用のスレッドプール分離、プール共有時の競合の有無 | CPU train・infer | `docs/analysis/pytorch-2.14-cpu-path.md` §9-1・§9-2 | 未検証の構造差 |
| C-3 | 現行 `TwoDDynamic` 上での P コア限定 on/off の再計測（M4 限定の分岐候補） | M-CPU-G* | `docs/analysis/candle-cpu-03.md` §7.1・§9 | REJECT 済みの実験（#1364）は旧分配方式が対象。現行経路では未実施。GB10 では REJECT のため M4 限定 |
| C-4 | gemm crate 型のキャッシュ連想度ベースのブロックサイズ導出 | CPU gemm | `docs/analysis/candle-cpu-03.md` §7.2・§8 | 新規（未検討）。近縁: `docs/perf/cpu-gemm-runtime-cache-detect.md`（実装済み・未結線） |
| C-5 | B パネルのスレッド間共有（#565）の 2D 動的分配後の再評価 | CPU gemm | `docs/analysis/candle-cpu-03.md` §7.3・§8 | 推奨案どまり（`docs/cpu-gemm-b-packing-sharing-decision.md`）。未実装 |
| C-6 | MLX のタイル選択と `select_for_device` の対比、NAX split-K と自作 split-K の設計対比 | M-MTL-G* | `docs/analysis/mlx-v0.32.2-steel-gemm.md` §8・§10 | 直接比較は未実施。NAX 本体の持ち込みは #549 §3 の再訪条件（M5／Ultra 実機必須）のまま |
| C-7 | fandhe-ai の TF32 opt-in 有効時の GB10 N=2048 再計測（精度クラスを揃えた比較。計測のみ） | G-CUDA-G1024・G2048 | `docs/analysis/burn-cubecl-cuda-matmul.md` §8(a)・§10 | opt-in 経路は実装・実測済み（`docs/cuda-tf32-optin-api-decision.md`）。tolerance・判定式は変えない。比較条件の扱いは §5.4 |
| C-8 | 学習側 `loss_readout`（backward 前の同期強制）の位置の単一同期化 | 全 train | `docs/analysis/train-infer-loop-comparison.md` §8 | 推論側は対応済み。学習側は未確認。ただし `loss_readout` は 0.0〜0.1 µs で、phase 実測上の寄与は現状小さい（train doc §17.2） |
| C-9 | infer fresh GPU の毎反復の重み H2D | infer 系（fresh 行） | `docs/analysis/train-infer-loop-comparison.md` §8・§7.3 | scoreboard の判定行は reuse のため直接の対象外。GPU 側への直接の対応記録なし |
| C-10 | burn の毎 step ノード再生成に相当する処理の fandhe-ai 側での有無の確認 | train | `docs/analysis/train-infer-loop-comparison.md` §8 新規候補 2 | 未確認 |
| C-11 | GB10 の PyTorch の実ディスパッチ先の確認（`torch.__config__.show()`・`DNNL_VERBOSE=1`）と、M4 の TF／SciPy の元 JSONL 再収集 | G-CPU-* ・比較の前提 | `docs/analysis/pytorch-2.14-cpu-path.md` §10・`docs/analysis/tensorflow-2.16-2.21-scipy-cpu-path.md` §11 | 計測条件の確認であり最適化ではない。#2120 の前提整備として扱える |

**承認が必要な事項**（本表では実施しない）: 新規 `unsafe`（SME の `asm!`＝#2119・prefetch の `asm!`・arena／CUDA Graph 実装で生じるもの）、`SME_PRODUCTION_ENABLED` の切替、facade 公開面の拡張、依存の追加、tolerance・baseline・係数の変更。

### 5.4 spec 提案につながりうる分岐（記載のみ）

- **PREC 差の扱い**: burn cuda は常に TF32、fandhe-ai の GEMM ベンチは既定 FP32 厳密。同一精度クラスでの比較条件を framework-compare でどう定義するかは spec の判断事項になりうる。本リポジトリでは spec を編集しない。提案する場合は `Fandhe-AI/fandhe-ai-spec` 側で行う（ユーザー判断）。C-7 の計測結果が判断材料になる
- **train の演算量の不一致**: dX の有無で backward の GEMM 回数が比較相手ごとに異なる（4 対 3）。C-1 の結果しだいで、train を「勝ち負けの対象から外す」等の扱いが必要になる可能性がある

## 6. 並列化可否の判定（受入 4）

前提: #2058・#2099 は Phase 3 を `parallel: 1` と定める（実機依存のため）。**本 doc はその決定を変えない**。以下は判断根拠の整理。

### 6.1 軸 1: 実機リソース

| 分類 | issue |
|---|---|
| M4 Max のみ | #2110・#2111・#2112・#2114 |
| GB10 のみ | #2107・#2108・#2109・#2115・#2116・#2117（#2106 は主に GB10） |
| 両機体 | #2100・#2101・#2102・#2104・#2105・#2113・#2118・#2120 |
| 実機不要 | #2103・#2119 |

同一実機を使う issue は実機を専有するため同時実行できない（専有ゲート。RULE.txt の load gate）。

### 6.2 軸 2: 対象ファイルの重複（issue 本文の「対象範囲」に基づく）

| 重複グループ | issue | 補足 |
|---|---|---|
| `crates/autodiff/src/grad.rs` | #2101（と C-1 が起票された場合） | 同一ファイルの並行編集は禁止（`.claude/rules/delegation-impl.md`） |
| `crates/autodiff/src/tape.rs` | #2103・#2104・#2114 | #2103 は設計のみ |
| `crates/backend-cpu/src/ops.rs` | #2101・#2106 | |
| `crates/backend-cpu/src/gemm_blis/mod.rs`・`gb10_affinity.rs` | #2117・#2118・#2119 | #2118 は定数のみ |
| `crates/backend-cuda/src/memory.rs` | #2107・#2108 | 直列チェーン（軸 3） |
| `crates/backend-cuda/src/ops.rs`・`gemm.rs`・`graph.rs` | #2109・#2115・#2116 | ops.rs は #2116 と #2108 の近傍でも触れうる（issue 本文に明記なし） |
| `crates/backend-metal/src/gemm.rs`・`tile.rs`・`shaders/gemm.metal` | #2110・#2111・#2113 | `gemm.rs` は #2110 と #2113 が共有 |
| `crates/backend-metal/src/memory.rs` | #2112 | 単独 |
| `scripts/bench/framework-compare/bench-fandhe/src/main.rs` | 計装系（`test(bench)`／`perf(bench)` の #2100・#2105・#2106・#2107・#2114・#2116・#2117・#2118・#2120） | phase 計測（`measure_train_phases`・`measure_train_reuse_phases`・`measure_infer_phases`）が同ファイルに置かれている。issue 本文に明記のない issue は推定 |
| `scripts/bench/framework-compare/` の run／集計スクリプト | #2120 | 全 issue の後 |

### 6.3 軸 3: 依存チェーン

- #2100 → #2101 → #2102（#2100 は #2101 の前提診断）
- #2103 → #2104
- #2107 → #2108
- #2110 → #2111
- #2118 → #2119（再開条件の順）。#2119 の承認は #2118 とは独立に進められるが、本番化の判断は #2118 の結果を要する
- #2120 は #2100〜#2119 の採否決定後

### 6.4 結論

- **直列が必須**: 上記チェーン 5 本、#2120（最後）、同一実機を専有する組、同一ファイルを触る組（#2110／#2113・#2107／#2108／#2116・#2101／C-1）
- **理論上は並列にできる組**（実機も同一ファイルも共有せず、依存もない）: 実機不要の #2103・#2119 と、いずれかの実機 issue。例: #2103（設計）と #2107（GB10）／#2110（M4）、#2119（監査記録）と #2115（GB10）。M4 のみの issue と GB10 のみの issue も、別機体でファイルが重複しなければ同時に走らせられる（例: #2110〔metal〕と #2107〔cuda〕）。ただし `bench-fandhe/src/main.rs` の共有（§6.2）が最後の制約になる
- 実効的な並列度は小さく、`parallel: 1` の維持は妥当。並列化の利得が最も大きいのは「実機不要の 2 本を実機 issue と同時に進める」だけである

## 7. Phase 3 の優先度提案（受入 5・ROI ベース）

### 7.1 ROI の定義

ROI ＝ 期待効果 ÷ コスト。

- 期待効果 ＝（1−ratio で表す負けの深さ）×（改善しうるセル数）。**ただし §4 のとおり寄与率は未計測なので、期待効果は上限の目安であり見積りではない**。証拠レベルが `未確定` の帰属に基づく施策は割り引く
- コスト ＝ 実装規模・実機占有・承認の要否
- 減点: REQ-8 境界検査・bit 一致契約への抵触リスク、承認待ち（`unsafe`）、REJECT 済みの近縁（重複回避の要否）、scoreboard 判定行（reuse）に効かない施策

### 7.2 提案

| 順位 | 施策（issue） | 根拠（負けの深さ×セル数・証拠・コスト） |
|---|---|---|
| 1 | CPU train／infer の固定費・構造差（#2100→#2101→#2102、#2105、#2106。＋C-1 の起票検討） | 最も深い負け（M-CPU-TRN 0.21・M-CPU-INF 0.18・G-CPU-TRN 0.34）が集中。dX の構造差は他 FW ソースで確定済みで、コストが低い診断（#2100・#2105・#2106）から入れる。C-1 が起票されれば最大の効果候補だが未起票・要承認。**#2101 は `mse_loss_backward` を対象外とする（REJECT 済み）ため、効果は限定的になりうる** |
| 2 | CUDA GEMM readback（#2107→#2108） | 4 セル（0.41〜0.83）。未説明分 46〜56% は大きいが `未確定`。診断（#2107）が安価で、結果に応じて #2108 を実施 |
| 3 | Metal GEMM（#2110→#2111） | 5 セル（0.59〜0.93）。差分候補が具体的（candle-metal §6）で M4 のみ。ただし E1〜E9 が多数 REJECT 済みのため、成功確率を割り引く。REQ-8 の制約下 |
| 4 | Metal／CUDA の train・infer（#2113・#2114・#2115・#2116・#2112） | 深い負け（M-MTL-TRN 0.38・G-CUDA-INF 0.41）を含むが、帰属が `推定`／`未確定` で診断が先。#2116 の fresh `param_readout` は判定行に効かないため #2116 は forward_resident の診断に絞る価値が高い |
| 5 | GB10 affinity・SME（#2117・#2118→#2119） | #2117 は undetermined の決着で安価。#2118／#2119 は CLOSED（Accelerate）への対抗で、`unsafe`／本番化の承認が要る。承認結果に依存 |
| 6 | #2103→#2104（arena）・#2109（CUDA N=256） | arena は tape 層の新規機構で規模が大きく、GEMM パネルの REJECT（#1481／#1482）との区別を設計で示す必要がある。#2109 は 0.83×（一番浅い）で対象 1 セル |
| 最後 | #2120（再計測） | 全採否の後 |

### 7.3 #2099 記載順との差分

#2099 本文の順は「CPU train／infer → CUDA gemm reuse → Metal gemm → Metal／CUDA train・infer → GB10 affinity・SME 再承認」で、上記の 1〜5 と**同じ並び**。差分は次の 3 点（**提案のみ。issue 本文・sub-issue の順序は編集しない。変更はユーザー判断**）:

1. #2116 の範囲を forward_resident の診断中心に絞る（fresh `param_readout` は scoreboard の reuse 判定に効かないため）
2. #2109 の対象数値の確認（0.72× と 0.83× の不一致。§8）
3. C-1（dX の省略）を、承認が得られれば #2101 と同じ枠で先頭グループに追加する

## 8. 限界・スコープ外・既知の不整合

いずれも本 PR では直さず、記録のみ。

- **Issue の記載とファイルの不一致**: #2098 本文は `docs/analysis/candle-metal-01.md`〜`mlx-steel-07.md`・`loop-analysis-08.md` と書くが、実ファイルは `mlx-v0.32.2-steel-gemm.md`・`train-infer-loop-comparison.md`（§9）。#2105・#2119 の対象ファイル名（`predict_resident.rs`・`sme.rs`）も実在しない（実体は §5.1 に記載）。`tensor-core/src/alloc.rs`（#2103・#2104）は新規予定で現時点では存在しない
- **件数**: Issue の「21」は `gen_1988.out` の 20＋2 と一致しない（§3.1）
- **#2109 の数値**: 「N=256 の 0.72×」はスコアボードの G-CUDA-G256（0.83×）と合わない。0.72× は G-CPU-G256
- **ライセンス記述**: MLX は MIT 単独（`docs/analysis/mlx-v0.32.2-steel-gemm.md` §9）で、#2096 の「MIT／Apache-2.0 dual」とは異なる
- **H2D 残存の食い違い**: `scripts/bench/framework-compare/README.md`（CUDA／Metal は resident 未対応で全パラメータ H2D）と `docs/device-resident-update-design.md`（weight 勾配は resident）が食い違う。どの時点の状態かは未確認（`docs/analysis/train-infer-loop-comparison.md` §9）
- **M4 の TF／SciPy は転記値**（元 JSONL が未収録）。GB10 の PyTorch の実ディスパッチ先は未確認（`docs/analysis/pytorch-2.14-cpu-path.md` §10）。Python FW の値は 2026-09-12 の流用
- **モード・版の違い**: TF 解析 doc の表は 0.8.0 の `fresh`、本 doc のスコアボードは 0.9.0 の `reuse`（§3 補足）
- **`docs/README.md` の `analysis/` ブロックの重複**は既知（`docs/analysis/mlx-v0.32.2-steel-gemm.md` §10 が対象外と記録）
- **本 doc の限界**: 新規の実測・A/B・ソース解析は行っていない。寄与率は全セルで未計測。C-1 は HEAD の静的読取りに基づく観察で、実際に GEMM が実行されているかの確認（カウンタ等）は #2100 と同種の診断が必要

## 9. 出典一覧

すべてリポジトリ内のパス。他 FW の上流 URL・版・ライセンスは各解析 doc の出典・ライセンス節を参照し、本 doc へは再掲しない（他ライブラリのコード片も載せない）。

| 種別 | パス |
|---|---|
| スコアボード | `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/gen_1988.out`・`gen_1988.py`・`body_1988.html`、同 `README.md`・`aggregate.md` |
| 計測規則 | `docs/perf/logs/framework-compare-0.9.0-remeasure/README.md` |
| 解析 doc（Phase 2） | `docs/analysis/candle-metal-01.md`（2.1）・`candle-0.11.0-cuda-path.md`（2.2）・`candle-cpu-03.md`（2.3）・`burn-cubecl-cuda-matmul.md`（2.4）・`pytorch-2.14-cpu-path.md`（2.5）・`tensorflow-2.16-2.21-scipy-cpu-path.md`（2.6）・`mlx-v0.32.2-steel-gemm.md`（2.7）・`train-infer-loop-comparison.md`（2.8） |
| phase 実測 | `docs/perf/train-step-phase-breakdown.md` §17・`docs/perf/infer-reuse-phase-breakdown.md` §10、生ログは `docs/perf/logs/train-infer-phases-0.9.0-1980-1981/` |
| ベンチ環境 | `docs/oss-comparison-harness-decision.md`・`docs/perf/oss-gemm-comparison-baseline.md` |
| 既存判定 | 表 B 各行の出典（`docs/perf/*.md`・`docs/*-decision.md`） |
| Phase 3 | #2099（子 #2100〜#2120）の各 issue 本文 |
