# 負けセル分析・原因帰属・未試行施策の対照表（Phase 3 入口）

イシュー #2098（親 #2089「他ライブラリのコード取得・詳細解析」の Phase 2 集約。後続は Phase 3 #2099〔子 #2100〜#2120〕）。
`docs/analysis/` の 8 本の解析結論を「セル × 原因 × 施策」の形へピボット集約した master doc。
**docs 専用・新規計測なし・コード変更なし**（tolerance・baseline・`Cargo.toml`・ガードレール閾値・`docs/spec/` は一切変えない）。

> **位置づけの注記（#3191）**: §1〜§9 は M4 Max・0.9.0 の Phase 3 入口としての履歴であり書き換えない。Phase 10 以降の目標設定の正本は **§10「0.11.0 27 セル表」**（Mac mini 13・GB10 14 のセル別に (1)〜(5) とゲート判定を集約）。

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
| 後継（#3191） | 本節以降は Phase 3 入口（M4 Max・0.9.0）の履歴。0.11.0 の 27 セル（Mac mini 13・GB10 14）の到達値表は §10 を正本とする |

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
| G-CPU-TRN（0.34×） | `LOOP` | `FIX`・`CLOSED` | backward 44.7%（482.7 µs）・device_update 25.7%（277.5 µs）・forward_resident 23.8%。step_total 1080.6 µs | M-CPU-TRN と同じ dX の構造差。加えて reuse の `device_update` が Mac の約 2.3 倍（train doc §17.6.3・§17.6.4）。PyTorch の GB10 経路は NVPL と推定（`CLOSED`） | `LOOP`: 同 M-CPU-TRN／`device_update` の内訳: 実測済み（#3188。`docs/perf/cpu-reuse-device-update.md` §11）。head で T 169.2 µs のうち SGD ループ 27.7 µs・`flat_grad` 確保 0（構造上）・残りの約 141.5 µs は H4 に帰属。H4 は #3188 再オープン分で分解済み（同 doc §12）: prologue（検査・小 Vec・bias `clone`）は約 2.3 µs で小さく、残差の本体は学習ループ内の `sgd_step_device` が forward／backward の rayon ワーカー直後に約 122 µs 遅くなる分（`RAYON_NUM_THREADS=1` では T 26 µs で消える）。機構は LLC 追い出し・他コア書き込みの代理腕では再現せず未確定 |
| G-CPU-INF（0.72×） | `FIX` | `CLOSED` | predict_resident 195.0 µs（fresh 176.3 µs より遅い逆転。infer doc §10.6.3） | reuse が fresh より遅い原因は未切り分け（run 間の min–max 幅が大きく負荷変動の寄与も未分離。infer doc §10.6.4）。#3188 の GB10 実測（`docs/perf/cpu-predict-resident-fixedcost.md` §6）は逆転を再現するが差は 2.4〜3.5 µs と bench の約 1/5〜1/8 で、L1 カーネル差・upload／readout・chain 検証に分かれる（一部のみ説明）。bench-fandhe の pre／head 前後比較（同 §6.1）でも逆転は再現し（reuse/fresh = 1.07〜1.10）、#3196 の効果は infer reuse total で確認できない（head/pre 1.018） | 実測済み（一部帰属）。未確定部分あり |
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
| #2106 | cpu reuse `device_update`（GB10 277 µs）の内訳切り分け | G-CPU-TRN・M-CPU-TRN | `LOOP`・`FIX` | 新規（train doc §17.6.4）。近縁: `docs/perf/train-resident-grad-device-update.md`（#1212） | 主に GB10 | `crates/backend-cpu/src/ops.rs`・`crates/autodiff/src/optim/sgd.rs`（実際の経路は `optim/device_store.rs::DeviceParamStore::step`。診断基盤は `docs/perf/cpu-reuse-device-update.md`・判定は実機実測まで未確定） |
| #2107 | CUDA readback 宛先確保の帰属検証（Layer B 計装） | G-CUDA-G512・G1024・（G2048） | `ALLOC`・`BOUND` | `PretouchedFresh` は実装済み（`docs/perf/cuda-host-view-readout-small-shape-regression.md`）。未説明分の帰属確定が目的 | GB10 | `crates/backend-cuda/src/memory.rs` |
| #2108 | readback 宛先の再利用・ゼロコピー化（opt-in） | 同上 | `ALLOC` | #2107 の結果次第。ADOPT で結線。**opt-in 実装済み（pinned staging 再利用 + copy-out・既定 OFF）・GB10 A/B 未実施**（`docs/perf/cuda-gemm-readback-reuse-2108.md`）。ゼロコピーは要承認で対象外 | GB10 | `crates/backend-cuda/src/memory.rs` |
| #2109 | CUDA N=256 の起動固定費（launch 回数・sync）診断 | G-CUDA-G256 | `FIX` | 新規。**issue 本文は「N=256 の 0.72×」と書くが、スコアボードの G-CUDA-G256 は 0.83×（0.72× は G-CPU-G256）。数値の転記違いの可能性（§8）** | GB10 | `crates/backend-cuda/src/gemm.rs`（`internal-diagnostics`） |
| #2110 | candle／MLX 解析の差分からの未試行 Metal GEMM 候補（opt-in 実装） | M-MTL-G256〜G4096 | `KERN` | 候補は candle-metal §6 の 1〜4（unroll 常時 ON・ロード方式・smem swizzle と E3 の組合せ・async copy）。REQ-8 の手動境界検査は維持（`docs/backend-metal-aligned-load-decision.md`）。async copy は不採用の既存決定あり（`docs/backend-metal-async-copy-decision.md`）で、差分は「candle 側の確認済み不在との突合の深掘り」のみ | M4 | `crates/backend-metal/src/gemm.rs`・`shaders/gemm.metal` |
| #2111 | 候補の kernel_gpu 5 run 実測・結線判断 | 同上 | `KERN` | 判定規則は issue 本文（正方 4 形状のうち 2 形状以上で ratio<1.00 かつ checksum 一致）。REJECT 候補の再検討はスコープ外 | M4 | `crates/backend-metal/src/tile.rs`・`gemm.rs` |
| #2112 | Metal reuse 計測窓の readback（N=4096 で +12 ms）の対処 | M-MTL-G4096・M-MTL-INF | `BOUND`・`ALLOC` | #1696 の 4 腕診断が前提（「なぜ legacy が遅いか」は別 issue） | M4 | `crates/backend-metal/src/memory.rs` |
| #2113 | Metal train forward の encode-only 化 | M-MTL-TRN | `LOOP`・`FIX` | 新規（train doc §17.4）。近縁: `docs/backend-metal-command-batching-design.md` §7（対象外の区間） | 両（issue 本文） | `crates/backend-metal/src/gemm.rs` |
| #2114 | Metal `tape_build` 削減＋infer GPU 起動固定費の診断カウンタ | M-MTL-TRN・M-MTL-INF | `FIX` | 新規。近縁: `docs/perf/metal-infer-chain-single-sync.md`。**診断基盤・opt-in（デバイス存在確認キャッシュ・既定 OFF）実装済み・M4 実測は申し送り**（`docs/perf/metal-tape-build-infer-fixedcost.md`） | M4 | issue 本文は `crates/autodiff/src/tape.rs`（Metal 向け）。実装は `crates/facade/src/lib.rs::resolve_ops` の存在確認が対象（tape.rs の構築はヒープ確保ゼロ） |
| #2115 | CUDA 推論チェーンの CUDA Graph capture（opt-in） | G-CUDA-INF | `FIX` | update 区間 capture（#1349）とは別機構（`docs/backend-cuda-graph-step-capture-design.md`）。新規 `unsafe` は不要（cudarc の安全 API と既存 launch 関数の再利用で実装。実装済み・opt-in 既定 OFF・GB10 実測は未実施〈`docs/perf/infer-chain-graphcapture-cuda-ab.md`〉） | GB10 | `crates/backend-cuda/src/graph.rs` |
| #2116 | CUDA train `forward_resident` 内訳・fresh `param_readout` 削減 | G-CUDA-TRN | `FIX` | 新規（train doc §17.6.4）。**fresh の `param_readout` は scoreboard の判定行（reuse）に効かない**（§4.3）。診断基盤は `cuda-train-forward-resident-diagnosis.md`（#2116・GB10 実測は未実施） | GB10 | `crates/backend-cuda/src/ops.rs`・`crates/autodiff/src/optim/device_store.rs` |
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
| C-3 | 現行 `TwoDDynamic` 上での P コア限定 on/off の再計測（M4 限定の分岐候補） | M-CPU-G* | `docs/analysis/candle-cpu-03.md` §7.1・§9 | REJECT 済みの実験（#1364）は旧分配方式が対象。GB10 では REJECT のため M4 限定。現行 `TwoDDynamic` での Mac mini（P4＋E6）実測は #3178 で 2026-10-21 に実施済み（`docs/perf/cpu-thread-policy-macmini.md`）。big-core 腕は REJECT（gemm N=1024 が約 12〜14% 悪化。N=2048 の PyTorch 比は 0.50 前後のまま）で、E コア仮説は本結果の範囲では支持されない（P コアへの affinity 固定は未検証） |
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
| §10 スコアボード・窓 | `docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`（`README.md`・`scoreboard/gen_0110_macmini.out`・`macmini-series-b/results-macmini-0.11.0-median5.jsonl`・`macmini-series-b/results-macmini-py-0.11.0-median5.jsonl`）、`docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/`（`results-dgx-0.11.0-median5.jsonl`・`results-dgx-0.11.0-extra-median5.jsonl`・`results-dgx-py-0.11.0-median5.jsonl`） |
| §10 Phase 9 診断（Mac mini） | `docs/perf/opp-kernel-macmini.md`・`docs/perf/metal-gemm-reuse-phase-breakdown.md` §12・`docs/perf/metal-gemm-tile-select-macmini.md`・`docs/perf/metal-infer-bimodal-macmini.md`・`docs/perf/train-step-phase-breakdown.md` §18・`docs/perf/infer-reuse-phase-breakdown.md` §11・`docs/perf/cpu-sme-thread-scaling-macmini.md`・`docs/perf/cpu-gemm-sme-ab-macmini.md`・`docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md`・`docs/perf/cpu-thread-policy-macmini.md`・`docs/perf/optin-ab-macmini.md` |
| §10 Phase 9 診断（GB10） | `docs/perf/opp-kernel-gb10.md`・`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12〜§14・`docs/perf/cuda-gemm-small-launch-cost.md` §7.1・`docs/perf/cuda-gemm-readback-reuse-2108.md` §5・`docs/perf/cuda-gemm-boundary-predicate-ab.md`・`docs/perf/infer-chain-graphcapture-cuda-ab.md` §3・`docs/perf/optin-ab-gb10.md`・`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §6・`docs/perf/cpu-gemm-gb10-affinity-ab.md`・`docs/perf/cpu-reuse-device-update.md` §11・§12・`docs/perf/cpu-predict-resident-fixedcost.md` §6・`docs/perf/cuda-train-forward-resident-diagnosis.md` §6・`docs/perf/train-step-phase-breakdown.md` §19・`docs/perf/infer-reuse-phase-breakdown.md` §12・`docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §9・`docs/perf/metal-gemm-n4096-kernel-gap.md` |
| §10 Phase 10・11 の対応 | 親 #3197・#3207・#3210・#3213・#3216・#3229・#3233・#3237・#3240・#3244・#3248・#3258・#3263 の sub-issues（GitHub API の結果） |

## 10. 0.11.0 27 セル表（Phase 9 集約・#3191）

イシュー #3191（親 #3174「性能の診断計測」＝ Phase 9 の集約）。Phase 9 の診断 16 件（#3175〜#3190）の結論 doc を、0.11.0 の判定対象 27 行（Mac mini 13・GB10 14）へ**セル別**に引き直した表である。
**docs 専用・新規計測なし・コード変更なし**（既定値・判定規則・tolerance・baseline・`docs/spec/` は変えない）。§1〜§9 は M4 Max・0.9.0 の Phase 3 入口としての履歴であり書き換えない。
Phase 10・11 の目標設定の入口は本節（§10.9 に下流 issue との対応）。#3262（CUDA 既定精度の判断材料）と #3273（0.12.0 再計測との突き合わせ）も本節を参照する。

### 10.1 位置づけ・入力・凡例

| 項目 | 内容 |
|---|---|
| 目的 | 27 セルそれぞれについて (1) 最速の相手の純カーネル時間とハード上限比、(2) fandhe-ai の純カーネル時間と固定費、(3) 相手の構成を再現した場合に届く比、(4) 相手がやっていない工夫を足した場合に届く比、(5) 届かない理由（ハードの上限／計測窓の定義／自分たちの制約）を 1 か所に並べる |
| 基準のスコアボード | `docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/scoreboard/gen_0110_macmini.out`（`tally 6 1 20 invalid 1 noise 1 total 27`）。Mac mini 列は Mac mini 系列 B（2026-10-11・`macmini-series-b/`）、GB10 列は 0.11.0 正式系列（2026-10-10・`docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/`） |
| 契約 1 | **証拠レベルを格上げしない**。実測で置き換えられる推定は実測を採る。入力のどれかが未測定の欄は「未測定」と書き、補間しない |
| 契約 2 | 制約（REQ-1・REQ-8・既定精度・`f64` アキュムレータ）の変更に関する判断は含めない。§10.5 の「自分たちの制約」欄は数字だけを並べる |
| 契約 3 | ゲートの ADOPT／REJECT／undetermined は各結論 doc の語彙のまま転記する（§10.8）。本節が判定を出すことはない |

**タグ**: `実測`（出典の計測値そのもの）・`推定`（実測値からの算出。式と入力の出典を併記）・`未測定`（出典なし）。
**機体・版**: 数値は次の基準タグで機体・版・出典を示す。M4 Max の値は Mac mini の値として書かない。

| 基準タグ | 機体 | 版・条件 | 出典 |
|---|---|---|---|
| `MM-B` | Mac mini（Apple M4・4P＋6E・GPU 10 コア・16 GB） | fandhe-ai／candle／burn の registry 0.11.0・系列 B（5 run 中央値）・2026-10-11 | `docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`（`README.md`・`macmini-series-b/`） |
| `MM-179` | Mac mini | PyTorch 2.14.0 の純 `mm`・Python checksum と Rust 逐次総和（fandhe-ai 0.11.0）。**窓が `MM-B` の 0.52〜0.54 倍で再現しなかった系列** | `docs/perf/opp-kernel-macmini.md` |
| `MM-182` | Mac mini | Layer A は registry 0.11.0、Layer B は HEAD（版差あり）・同一セッション | `docs/perf/metal-gemm-reuse-phase-breakdown.md` §12 |
| `MM-181` | Mac mini | HEAD（`bc5b42bcf`）・candle 0.11.0 | `docs/perf/metal-gemm-tile-select-macmini.md` |
| `MM-180` | Mac mini | Layer A は registry 0.11.0、同期回数・同期固定費は HEAD（版差あり） | `docs/perf/train-step-phase-breakdown.md` §18・`docs/perf/infer-reuse-phase-breakdown.md` §11 |
| `MM-183` | Mac mini | HEAD（第 2 ラウンド）・candle は別セッションの値 | `docs/perf/metal-infer-bimodal-macmini.md` |
| `MM-17x` | Mac mini | A/B・診断（`MM-175` SME 多スレッド・`MM-176` SME ON/OFF・`MM-177` m=64 job 分割・`MM-178` スレッド方針・`MM-184` opt-in 6 ゲート）。計測時の HEAD は各 doc の記載 | `docs/perf/cpu-sme-thread-scaling-macmini.md`・`cpu-gemm-sme-ab-macmini.md`・`cpu-gemm-m64-jobgrid-diagnosis.md`・`cpu-thread-policy-macmini.md`・`optin-ab-macmini.md` |
| `GB-B` | GB10（DGX Spark・20 コア・sm_121） | registry 0.11.0・正式系列（5 run 中央値）・2026-10-10 | `docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/` |
| `GB-185` | GB10 | torch 2.14.0+cu130 の cuBLAS 純カーネル（CUDA event／wall）。fandhe-ai は HEAD の診断テスト（0.9.0 の Layer B と整合） | `docs/perf/opp-kernel-gb10.md` |
| `GB-186` | GB10 | HEAD（`0889838c8`）・5 run | `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12・§14 |
| `GB-187` | GB10 | HEAD・5 run・affinity A/B（`2117-gb10-3187`） | `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §6・`docs/perf/cpu-gemm-gb10-affinity-ab.md` |
| `GB-189` | GB10 | Layer A は registry 0.11.0（計測 SHA `ad1d106e4`）、同期・launch 回数は HEAD の nsys（版差あり） | `docs/perf/train-step-phase-breakdown.md` §19・`docs/perf/infer-reuse-phase-breakdown.md` §12 |
| `GB-188` | GB10 | HEAD の pre（`17c7767b5`）と head（`4f6716f63`）の前後比較・5 run・診断と bench-fandhe | `docs/perf/cpu-reuse-device-update.md` §11・§12、`docs/perf/cpu-predict-resident-fixedcost.md` §6 |
| `GB-190` | GB10 | HEAD（`d89a09370`）・opt-in 6 ゲート（B1〜B6） | `docs/perf/optin-ab-gb10.md` |
| `GB-215` | GB10 | 計測ツリー `31cd381f0`（セルのスクリプト修正は `977e060d1`。`crates/` は同一）・事前登録規則・純カーネル event 5 プロセス・cuBLAS 5 run・gemm セル 5 round・torch 2.14.0+cu130 | `docs/perf/logs/cuda-tma-stage2-ab-gb10-3215/`（`RULE.txt`・`aggregate.md`・`README.md`） |
| `M4M` | M4 Max 開発機 | 各表の注記のとおり（**別機体・参考。算術には混ぜない**） | 各出典 |

**算出式**（`推定` の欄に使う。(3)(4) は導出値であり出典 doc の数値ではない）:

| 式 | 内容 | 用途 |
|---|---|---|
| F1 | 再現比（カーネル置換）＝ 相手の窓 ÷（fandhe-ai の窓 − fandhe-ai の純カーネル ＋ 相手の純カーネル）。純カーネルの時計（GPU タイムスタンプ／CUDA event／wall）は fandhe-ai と相手で揃える。揃わない場合は境界を併記する | (3) |
| F1(b) | Metal 専用。fandhe-ai の `encode ＋ commit_wait` を candle の `transfer_excl`（壁時計）へ置換する境界揃え版（`MM-182` §12.6 の (b) と同じ境界） | (3) |
| F2 | 窓構成の再現比 ＝ 相手の窓 ÷ fandhe-ai **fresh** の窓（結果を毎反復解放する相手の窓構成に fandhe-ai を合わせたもの。`MM-B`／`GB-B` の同一セッション中央値） | (3) |
| F3 | 工夫比 ＝ 相手の窓 ÷（fandhe-ai の窓 − 候補が縮めた区間）。縮めた区間は A/B の実測比（after／before）または診断の区間値から得る | (4) |
| F4 | 仮想上限 ＝ 特定の区間を 0 にした場合の比。**実装も実測もない上限値**で、候補の効果の見積りではない | (4) |

**読むときの注意**（本節の全表に適用）:

1. **スコアボードの「比」は run 内比の 5 run 中央値**であり、本節の F1〜F4 は「中央値 ms からの算出」で定義が異なる（例: Mac mini の infer Metal は中央値 ms の比が 1.31、scoreboard は 1.13）。両者を同じ列に混ぜず、scoreboard の比は §10.2 に逐語で置く
2. **基準の混在**: 純カーネルの出典は系列 B と別セッションである。同一セッションで窓と内訳が揃うもの（`MM-182`・`GB-B` の fresh／reuse）を優先し、系列 B の窓との差を `MM-182` §12.6 のドリフト検査（Metal N=4096: fandhe-ai −26.7%・candle −22.3%）と同様に併記する。窓の食い違いが大きい `MM-179` は §10.10 に所有者判断待ちとして送る
3. (3)(4) の算出値は目標値の合格ラインではない（#3174 の方針）。Phase 10・11 で A/B した実測が出たら、そちらが優先する

### 10.2 27 セル一覧（区分・最速相手・窓）

区分・比・run 別比は `gen_0110_macmini.out` の値を逐語で転記した。窓は系列ごとの中央値行（`MM-B`／`GB-B`）。「相手の窓」は区分を決めた最速相手の fresh 窓、「fandhe-ai の窓」は reuse 窓（判定対象）。
Mac mini 行の最右列は、同じ行の M4 Max（0.11.0 正式系列・`M4M`）の区分と比で、`docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/README.md`「結果」節の並置表の転記（別機体・参考）。

| セル ID | 機体・版 | 区分 | 最速相手 | scoreboard の比 | run 別比 | 相手の窓 [ms] | fandhe-ai の窓 [ms] | M4 Max の同一行（別機体・参考） |
|---|---|---|---|---|---|---|---|---|
| MM-MTL-G256 | Mac mini・`MM-B` | 負け（2 位） | candle | 0.84× | 0.64 0.89 0.89 0.84 0.78 | 0.3944 | 0.4418 | 負け 0.88× |
| MM-MTL-G512 | Mac mini・`MM-B` | 負け（2 位） | candle | 0.80× | 0.80 0.80 0.79 0.81 0.80 | 1.1015 | 1.3775 | 負け 0.78× |
| MM-MTL-G1024 | Mac mini・`MM-B` | 負け（2 位） | candle | 0.69× | 0.67 0.68 0.69 0.69 0.69 | 3.2194 | 4.6815 | 負け 0.72× |
| MM-MTL-G2048 | Mac mini・`MM-B` | 負け（2 位） | candle | 0.73× | 0.74 0.70 0.73 0.69 0.74 | 14.7713 | 20.4111 | 負け 0.68× |
| MM-MTL-G4096 | Mac mini・`MM-B` | 負け（2 位） | candle | 0.58× | 0.57 0.58 0.58 0.58 0.58 | 71.1920 | 122.8210 | 負け 0.65× |
| MM-CPU-G256 | Mac mini・`MM-B` | **1 位** | PyTorch | 1.04× | 1.01 1.04 1.01 1.14 1.04 | 0.3195 | 0.3075 | 負け 0.77× |
| MM-CPU-G512 | Mac mini・`MM-B` | **1 位** | PyTorch | 1.04× | 1.03 1.26 1.05 1.04 1.03 | 1.4034 | 1.3542 | 僅差 0.95× |
| MM-CPU-G1024 | Mac mini・`MM-B` | 負け（2 位） | PyTorch | 0.89× | 1.16 0.90 0.84 0.89 0.86 | 6.9713 | 7.8709 | 負け 0.87× |
| MM-CPU-G2048 | Mac mini・`MM-B` | 負け（2 位） | PyTorch | 0.69× | 0.83 0.69 0.70 0.68 0.68 | 35.7591 | 51.7847 | 負け 0.80× |
| MM-MTL-TRN | Mac mini・`MM-B` | 負け（3 位） | PyTorch | 0.32× | 0.32 0.32 0.31 0.32 0.32 | 0.6888 | 2.1329 | 負け 0.39× |
| MM-MTL-INF | Mac mini・`MM-B` | 1 位（**ノイズ帯**。系列 A は 3 位 0.72×） | candle | 1.13× | 1.31 0.71 0.90 1.40 1.13 | 0.5075 | 0.3884 | 1 位 1.41× |
| MM-CPU-TRN | Mac mini・`MM-B` | 負け（4 位） | PyTorch | 0.20× | 0.23 0.20 0.19 0.21 0.20 | 0.2214 | 1.1059 | 負け 0.22× |
| MM-CPU-INF | Mac mini・`MM-B` | 負け（2 位） | PyTorch | 0.25× | 0.24 0.25 0.25 0.24 0.25 | 0.0535 | 0.2154 | 負け 0.20× |
| G-CUDA-G256 | GB10・`GB-B` | 負け（2 位） | candle | 0.84×（0.10.0 比 0.99×） | 0.84 0.83 0.84 0.83 0.85 | 0.0764 | 0.0900 | — |
| G-CUDA-G512 | GB10・`GB-B` | 負け（3 位） | candle | 0.42×（1.04×） | 0.42 0.42 0.41 0.42 0.43 | 0.2371 | 0.5662 | — |
| G-CUDA-G1024 | GB10・`GB-B` | 負け（3 位） | candle | 0.40×（1.00×） | 0.40 0.39 0.41 0.39 0.40 | 0.9264 | 2.2890 | — |
| G-CUDA-G2048 | GB10・`GB-B` | 負け（3 位） | burn（TF32 固定） | 0.48×（1.00×） | 0.48 0.48 0.48 0.49 0.48 | 4.1443 | 8.5947 | — |
| G-CUDA-G4096 | GB10・`GB-B` | **1 位** | burn（TF32 固定） | 1.01×（1.00×） | 1.02 1.01 1.02 1.01 1.00 | 39.6344 | 39.1144 | — |
| G-CPU-G256 | GB10・`GB-B` | 負け（3 位） | PyTorch | 0.74×（0.95×） | 0.68 0.74 0.95 0.72 1.00 | 0.2705 | 0.3670 | — |
| G-CPU-G512 | GB10・`GB-B` | 負け（2 位） | candle | 0.70×（1.03×） | 0.66 0.57 0.70 0.78 0.79 | 1.7620 | 2.5218 | — |
| G-CPU-G1024 | GB10・`GB-B` | 僅差 | candle | 0.98×（1.02×） | 0.95 0.97 0.98 1.00 1.00 | 5.4552 | 5.5976 | — |
| G-CPU-G2048 | GB10・`GB-B` | **1 位** | TensorFlow | 1.06×（0.97×） | 1.03 1.23 1.02 1.10 1.06 | 28.0825 | 26.3011 | — |
| G-CPU-G4096 | GB10・`GB-B` | **1 位** | TensorFlow | 1.21×（0.98×） | 1.19 1.23 1.20 1.23 1.21 | 147.4993 | 121.4821 | — |
| G-CUDA-TRN | GB10・`GB-B` | 負け（2 位） | candle | 0.83×（1.00×） | 0.83 0.83 0.83 0.83 0.81 | 0.2717 | 0.3309 | — |
| G-CUDA-INF | GB10・`GB-B` | 負け（3 位） | PyTorch | 0.40×（1.00×） | 0.40 0.40 0.40 0.40 0.41 | 0.0399 | 0.0990 | — |
| G-CPU-TRN | GB10・`GB-B` | 負け（3 位） | PyTorch | 0.35×（0.95×） | 0.35 0.32 0.36 0.41 0.34 | 0.3574 | 1.0302 | — |
| G-CPU-INF | GB10・`GB-B` | 負け（2 位） | PyTorch | 0.60×（1.00×） | 0.60 0.60 0.64 0.54 0.66 | 0.1127 | 0.1889 | — |

- 集計（`gen_0110_macmini.out` 末尾）: 1 位 6（Mac mini 3・GB10 3）・僅差 1・負け 20（Mac mini 10・GB10 10）・判定不能セル 1（GB10 の PyTorch CPU N=4096 の 1 要素）・ノイズ帯 1（MM-MTL-INF）
- GB10 の窓は `results-dgx-0.11.0-median5.jsonl`・`results-dgx-0.11.0-extra-median5.jsonl`・`results-dgx-py-0.11.0-median5.jsonl` の中央値行（fandhe-ai は `mode: reuse` の行）から読んだ。比を中央値 ms の除算で再計算すると scoreboard と一致するのは安定セル（例: G-CUDA-G512 0.2371÷0.5662＝0.419）で、run 間ばらつきが大きい MM-MTL-INF は一致しない（1.31 対 1.13。注意 1）
- G-CUDA-G2048・G4096 の最速相手 burn は TF32 が既定の行しかない。fandhe-ai／candle の TF32 opt-in 行は f32 の判定から除外される（`gen_0110_macmini.py` の `tf32` 除外、同 `card_tf32`）。burn の TF32 行は精度クラス TF32 の救済項で有効セル扱い（同スクリプトのコメント）

### 10.3 GEMM 19 セルの (1)〜(4)

(1) は相手の純カーネル時間とハード上限比、(2) は fandhe-ai の純カーネル時間と固定費、(3) は相手の構成の再現（F1・F2）、(4) は相手がやっていない工夫（F3・F4・A/B の実測）。各値に基準タグ（§10.1）とタグ（実測／推定／未測定）を付ける。

#### 10.3.1 Mac mini・Metal（MM-MTL-G256〜G4096。相手は candle）

(1) candle の純カーネル時間は**未測定**（GPU タイムスタンプ未取得。`MM-182` §12.10 が記録）。参考値は 2 つある。candle の `transfer_excl`（壁時計。encode と同期を含む）と、MPS の純時間（`MM-179`。参考上限であり相手ではない）。ハード上限（公称ピーク）はリポ内に出典がなく**未測定**。

| セル ID | (1) 相手の純カーネル・上限比 | (2) fandhe-ai の純カーネル・固定費 | (3) 再現で届く比 | (4) 工夫で届く比 |
|---|---|---|---|---|
| MM-MTL-G256 | candle excl 0.2046 ms＝0.164 TFLOPS（推定・`MM-182`・式 2N³÷excl）。MPS 純時間 0.136 TFLOPS は同期固定費が支配し比較にならない（`MM-179`）。上限比: 未測定 | kernel_gpu 0.0197 ms（実測・`MM-182`・窓の 4.7%）。commit_wait−kernel_gpu 0.2132 ms・upload 0.0370 ms・host_copy 0.0221 ms（実測・`MM-182`） | F1(b) 1.01（推定）。同一セッションの現状比は 0.92（推定。scoreboard 0.84）。F2 fresh 窓 0.84（推定・`MM-B`） | タイル最良は現行と同一（`MM-181` 最良/選択 1.000）で F3 変化なし 0.92（推定）。F4 upload＋host_copy を 0 にした場合 1.11（推定・仮想上限） |
| MM-MTL-G512 | excl 0.3234 ms＝0.830 TFLOPS（推定）。MPS 0.880 TFLOPS に対し 0.94 倍（推定・同期固定費が残るため参考） | kernel_gpu 0.1148 ms＝2.34 TFLOPS（推定・窓の 13.3%）。commit_wait−kernel_gpu 0.2318 ms・upload 0.1248 ms・host_copy 0.0729 ms（実測・`MM-182`） | F1(b) 0.79（推定）。現状比 0.76（同上）。F2 0.84（推定） | F3 最良タイル `[1]`（kernel ×0.946・`MM-181`）で 0.77（推定）。F4 1.06（推定・仮想上限） |
| MM-MTL-G1024 | excl 0.9503 ms＝2.26 TFLOPS（推定）。MPS 2.298 TFLOPS に対し 0.98 倍（推定） | kernel_gpu 0.8607 ms＝2.50 TFLOPS（推定・窓の 28.6%）。commit_wait−kernel_gpu 0.3912 ms・upload 0.5203 ms・host_copy 0.2406 ms（実測・`MM-182`） | F1(b) 0.83（推定）。現状比 0.75。F2 0.76（推定）。`MM-181` の主比（excl÷fandhe 壁時計合計）は 0.689、参考比（excl÷kernel_gpu）は 1.059（実測・境界が違う） | F3 最良タイル `[1]`（×0.934）で 0.77。F3 steel T0U（kernel ×0.8987・G6 は REJECT）で 0.77。F4 1.00（推定・仮想上限） |
| MM-MTL-G2048 | excl 5.4463 ms＝3.15 TFLOPS（推定）。MPS 3.237 TFLOPS に対し 0.97 倍（推定） | kernel_gpu 8.0716 ms＝2.13 TFLOPS（推定・窓の 57.5%）。commit_wait−kernel_gpu 0.5814 ms・upload 1.8262 ms・host_copy 0.9032 ms（実測・`MM-182`） | F1(a) 0.83（推定。境界は kernel_gpu 対 excl）。F1(b) 0.87（推定）。現状比 0.67。F2 0.81（推定）。`MM-181` の主比 0.608・参考比 0.676 | F3 最良タイル `[2]`（×0.812）で 0.76。F3 T0U（×0.7657）で 0.78。`MM-181` の最良候補の参考比は 0.832（実測）。F4 0.84（推定・仮想上限） |
| MM-MTL-G4096 | excl 42.4764 ms＝3.24 TFLOPS（推定）。MPS 3.429 TFLOPS に対し 0.94 倍（推定） | kernel_gpu 65.2521 ms＝2.11 TFLOPS（推定・窓の 72.4%）。commit_wait−kernel_gpu 2.2951 ms・upload 7.6416 ms・host_copy 3.7062 ms（実測・`MM-182`）。最良タイル `[2]` の kernel_gpu は 53.3344 ms＝2.58 TFLOPS（実測・`MM-181`） | F1(a) 0.82（推定）。F1(b) 0.85（推定）。現状比 0.61。F2 0.62（推定）。`MM-181` の主比 0.626・参考比 0.652 | F3 最良タイル `[2]`（×0.821）で 0.71。F3 T0U（×0.7748）で 0.73。`MM-181` の最良候補の参考比は 0.795（実測）。F4 0.70（推定・仮想上限）。F4 は upload を 0 にしても Metal N=4096 は 1.0 に届かない |

- 同一セッションの「現状比」（0.92・0.76・0.75・0.67・0.61）は `MM-182` の窓（fandhe-ai reuse 対 candle fresh）の中央値 ms の比で、scoreboard の 0.84・0.80・0.69・0.73・0.58 とは別セッション・別定義（注意 1・2）。F1・F3・F4 は同一セッションの `MM-182` を入力にした
- F3 のタイル比は `MM-181`（別セッション）の kernel_gpu の最良/選択比（0.946・0.934・0.812・0.821）を `MM-182` の kernel_gpu に掛けた。N=512 以上の最良候補は 5 run すべてで選択より速いが、Mac mini 専用表は #3206 の範囲で本節は実装しない
- steel 候補（T0U）は N=512 で 1.0378 の 5/5 run 一貫の後退があり、判定は REJECT（§10.8）。N=1024 以上の比は kernel_gpu の head/base 0.8987／0.7657／0.7748（実測・`MM-184`）。N=512 は F3 に使っていない
- candle の fresh 窓構成（`MM-182` §12.7）: fandhe-ai の reuse は fresh より N=512 以上で遅い（N=4096 で +5.19 ms）。F2 の fresh 窓比（0.84・0.84・0.76・0.81・0.62）は、reuse の窓構成差を除くと比がどこまで動くかを示す。窓側の是正は #3265

#### 10.3.2 Mac mini・CPU（MM-CPU-G256〜G2048。相手は PyTorch／Accelerate）

(1) の相手の純カーネルは PyTorch の `torch.mm`（Accelerate・閉源）。ハード上限は第三者の公開値（M4 の f32 SME ピーク約 2008 GFLOPS。リポ外・別機体）を分母にした推定。

| セル ID | (1) 相手の純カーネル・上限比 | (2) fandhe-ai の純カーネル・固定費 | (3) 再現で届く比 | (4) 工夫で届く比 |
|---|---|---|---|---|
| MM-CPU-G256 | 0.0240 ms＝1398 GFLOPS（実測・`MM-179`）。対 2008 で 0.70（推定） | 窓−Rust checksum 0.2711 ms＝124 GFLOPS（推定・`MM-B` 窓−`MM-179` の Rust 0.556 ns／要素×N²）。Rust checksum 0.0364 ms（推定） | Accelerate は閉源で再現不能。F2 fresh 窓 0.95（推定）。F1 の仮想上限 5.29（系列 B 窓）／2.76（`MM-179` の窓換算）（推定・再現ではない） | SME ON は N=256 が A/B 対象外で**未測定**（`MM-176`） |
| MM-CPU-G512 | 0.1520 ms＝1766 GFLOPS（実測・`MM-179`）。対 2008 で 0.88（推定） | 窓−Rust checksum 1.1961 ms＝224 GFLOPS（推定）。Rust checksum 0.1581 ms（推定） | F2 1.05（推定）。F1 仮想上限 4.53／2.37（推定） | SME ON 実測（`MM-176`）: after/before 0.5636（fresh）・0.5982（reuse）。同一ラウンド内の PyTorch 比 before 0.737／0.742 → after 1.307／1.251。F3 は A/B の実測比をそのまま採る |
| MM-CPU-G1024 | 1.1404 ms＝1883 GFLOPS（実測・`MM-179`）。対 2008 で 0.94（推定） | 窓−Rust checksum 7.2365 ms＝297 GFLOPS（推定）。Rust checksum 0.6344 ms（推定） | F2 0.91（推定）。F1 仮想上限 3.93／2.05（推定） | SME ON 実測（`MM-176`）: after/before 0.4687／0.4960。PyTorch 比 before 0.650／0.619 → after 1.406／1.246 |
| MM-CPU-G2048 | 9.7186 ms＝1768 GFLOPS（実測・`MM-179`）。対 2008 で 0.88（推定） | 窓−Rust checksum 49.2513 ms＝349 GFLOPS（推定）。Rust checksum 2.5334 ms（推定）。SME 経路（既定 OFF）の (2048,2048,1024) は T=1 で 355.5・T=10 で 1092.5 GFLOPS、`smstart`／`smstop` 1 組の固定費は 44.5〜89.8 ns（実測・`MM-175`） | F2 0.68（推定）。F1 仮想上限 2.92／1.59（推定） | SME ON 実測（`MM-176`）: after/before 0.4723／0.4806。PyTorch 比 before 0.525／0.494 → after 1.109／1.030 |

- **(3) の読み方**: Accelerate（閉源）は再現できないため、F1 の値は「相手の純カーネル水準に届いた仮想上限」であり再現ではない。fandhe-ai の窓から Rust checksum を引いた区間を相手の純 `mm`（`MM-179`）で置換した。「／」の左は PyTorch の窓に系列 B の値を使った場合、右は PyTorch の窓に `MM-179` の窓比 0.522／0.523／0.522／0.543 を掛けた場合（fandhe-ai の窓は不変と仮定した推定）。2 つの値の差が大きいのは `MM-179` が系列 B の約半分の窓で再現したため（§10.10 の所有者判断待ち）
- SME ON の A/B は同一ラウンド内の PyTorch 比で、系列 B の reuse 比 0.89／0.69（N=1024／2048）とは別 run の値（`MM-176` が明記）。判定は ADOPT 候補（§10.8）。本番切替は #3230、しきい値は #3231
- big-core／small-cap（`MM-178`）の gemm 4 サイズの中央値比は big-core が 1.0046〜1.1435（N=1024 が 1.1174／1.1435 で 0/5 round 改善）、small-cap が 0.9386〜1.0125 で、どちらも REJECT。E コア仮説は支持されない（並列度 4 でも N=2048 の PyTorch 比は 0.495〜0.539）が、affinity もプール縮小もない腕のため否定でもない
- MM-CPU-G256・G512 は 1 位だが、勝因の大きな部分は窓構成（PyTorch の窓に Python checksum が入る）で、ライブラリ同士の比は別に読む（§10.6）

#### 10.3.3 GB10・CUDA（G-CUDA-G256〜G4096。相手は candle／burn）

(1) の相手の純カーネルは cuBLAS FP32（PyTorch）の実測を、candle の代理として置く（candle も cuBLAS SGEMM と読める。candle 自身の純カーネルは**未測定**）。burn（cubecl）の純カーネルは**未測定**（`GB-185` が対象外と明記）。N=2048・4096 の最速相手 burn は TF32 固定。

| セル ID | (1) 相手の純カーネル・上限比 | (2) fandhe-ai の純カーネル・固定費 | (3) 再現で届く比 | (4) 工夫で届く比 |
|---|---|---|---|---|
| G-CUDA-G256 | cuBLAS FP32 0.0180 ms（wall）／0.0170 ms（event）＝1.86 TFLOPS、公称 31 の 0.06（実測・`GB-185`） | event 0.0146 ms（b2b 0.0123 ms）、event 比 0.858（実測・`GB-185`）。HEAD の再測定は event 0.0130 ms＝2.59 TFLOPS、比 0.781（実測・`GB-215`）。固定費: H2D 22.9 µs（L0 の 43.4%）・readback 25.1 µs（同 47.5%）・launch 往復の床 6.1 µs（実測・`GB-186`〔`docs/perf/cuda-gemm-small-launch-cost.md` §7.1 の有効系列〕） | F1 0.83（推定・event 時計。現状 0.85）。F2 fresh 窓 0.88（推定） | F3 readback の pinned 再利用は REJECT（比 1.0475・5/5 後退・実測・`GB-190` B6）。TF32 opt-in（fandhe-ai fresh 0.0906 ms）対 candle TF32 0.0641 ms は 0.71（推定・参考・判定外）。FP32 カーネル候補 C1（TMA 128×64）は N=256 で現行（64×64）比 0.68（`tma128_none`）・0.76（`tma128_strided`）の後退で、結線範囲（N≥1024）の外（実測・`GB-215`） |
| G-CUDA-G512 | cuBLAS FP32 0.0348 ms（wall）／0.0337 ms（event）＝7.71 TFLOPS、公称の 0.25（実測・`GB-185`） | event 0.0372 ms、event 比 1.106（実測・`GB-185`）。HEAD の再測定は event 0.0356 ms＝7.54 TFLOPS、比 1.062（実測・`GB-215`）。reuse−fresh の窓差 0.308 ms（推定・`GB-B` 中央値） | F1 0.42（推定・event 時計。現状 0.42 のまま）。F2 0.92（推定） | TF32 同士 0.85（candle TF32 0.2225 ms 対 fandhe-ai TF32 fresh 0.2632 ms。推定・参考・判定外）。FP32 カーネル候補 C1 は現行比 1.07（5/5 run 一貫。結線範囲外のため判定に使わない。実測・`GB-215`） |
| G-CUDA-G1024 | cuBLAS FP32 0.1372 ms＝15.65 TFLOPS、公称の 0.50（実測・`GB-185`）。wall と event はほぼ同値（0.1367 ms） | wall 0.1864 ms（Layer B。投入と同期を含む）＝11.5 TFLOPS（推定）。wall 比 1.358（実測・`GB-185`）。event 0.1814 ms＝11.84 TFLOPS、event 比 1.325（実測・`GB-215`）。reuse−fresh の窓差 1.333 ms（推定） | F1 0.41（推定・wall 時計。現状 0.41）。F2 0.97（推定）。F1 を fresh 窓へ適用すると 1.02（推定） | TF32 同士 0.90（candle TF32 0.8644 ms 対 fandhe-ai TF32 fresh 0.9561 ms。推定・参考・判定外）。readback pinned 再利用は REJECT（比 1.0991・`GB-186`〔`docs/perf/cuda-gemm-readback-reuse-2108.md` §5〕）。FP32 カーネル候補 C1 は純カーネル比 1.145（`tma128_none`）・1.175（`tma128_strided`）で、ゲート A・B は pass、判定は **ADOPT（結線候補。ゲート D は結線 PR で実施）**。`tma128_strided` の純カーネルは cuBLAS FP32 の 1.128 倍（実測・`GB-215`）。結線後のセル値は未測定 |
| G-CUDA-G2048 | cuBLAS FP32 0.9739 ms＝17.64 TFLOPS、公称の 0.57（実測・`GB-185`）。最速相手 burn の純カーネルは**未測定**。cuBLAS TF32 は 0.4215 ms＝40.76 TFLOPS、公称 53.3 の 0.76（実測） | wall 1.1693 ms＝14.7 TFLOPS（推定）。wall 比 1.201（実測・`GB-185`）。event 1.1854 ms＝14.49 TFLOPS、event 比 1.198（実測・`GB-215`）。reuse−fresh の窓差 4.409 ms（推定） | F1 は burn の純カーネルが未測定のため**未測定**。参考: cuBLAS FP32 で置換した仮想値 0.49（現状 0.48）。F2 fresh 窓 0.99（推定）。F2 の相手を candle FP32 にすると 1.01（推定） | TF32 opt-in（fresh 4.37 ms）対 burn（TF32）4.1443 ms は 0.95（推定）。TF32 同士の最速（candle TF32 3.6952 ms）に対しては 0.85（推定・参考・判定外）。3×TF32 は f32 SIMT より遅い（§10.7）。FP32 カーネル候補 C1 は純カーネル比 1.124（`tma128_none`）・1.133（`tma128_strided`）。ADOPT（結線候補。ゲート D は結線 PR で実施）。`tma128_strided` は cuBLAS FP32 の 1.057 倍（実測・`GB-215`）。結線後のセルは推定 0.487（現状 0.478。推定・測定値ではない） |
| G-CUDA-G4096 | cuBLAS FP32 7.9627 ms＝17.26 TFLOPS、公称の 0.56（実測・`GB-185`）。burn の純カーネルは**未測定**。cuBLAS TF32 は 3.2933 ms＝41.73 TFLOPS、公称の 0.78（実測） | wall 10.0066 ms＝13.7 TFLOPS（推定・対 31 で 0.44）。wall 比 1.257（実測・`GB-185`）。reuse と fresh は並ぶ（窓差 −0.54 ms。推定）。fresh の readback 事前タッチ約 15 ms が残る（実測・`GB-186` §14.3 の判定 F）。event 10.341 ms＝13.29 TFLOPS、event 比 1.297（実測・`GB-215`） | F1 は同上**未測定**。参考: cuBLAS FP32 置換の仮想値 1.07（現状 1.01）。F2 fresh 窓 1.00（推定） | TF32 opt-in（fresh 33.9267 ms）対 burn（TF32）39.6344 ms は **1.17**（推定。FP32 の fresh 39.6519 ms では 1.00）。TF32 同士でも最速の burn に対し 1.17。FP32 カーネル候補 C1 は純カーネル比 1.035（`tma128_none`）・1.038（`tma128_strided`）の ADOPT（結線候補。ゲート D は結線 PR で実施）。伸びは 3.5〜3.8% で、`tma128_strided` は cuBLAS FP32 の 1.250 倍が残る（実測・`GB-215`）。結線後のセルは推定 1.025（現状 1.010。推定・測定値ではない） |

- 1 位の G-CUDA-G4096 は、最速相手 burn の TF32 に既定精度（FP32）の fandhe-ai が並んでいる状態（1.01×）。TF32 opt-in では 1.17 倍に伸びる値の出典は `framework-compare-0.11.0-remeasure/gb10/results-dgx-0.11.0-median5.jsonl` の `tf32: true` 行（§10.7）
- F1 の `GB-185` の値は時計が揃うセルだけで採った（N=256・512 は event 同士、N≥1024 は wall 同士）。fandhe-ai の N≥1024 の event は #3215 で取得した（`GB-215`。上表の「event」値）。N≤512 の wall は「該当計測なし」のまま
- N=256 の窓差の大半（H2D＋readback＝L0 の 90.9%）は相手側を同じ区間で分解していないため gap への寄与は未分離（`GB-186`）。launch 往復の床は 6.1 µs

#### 10.3.4 GB10・CPU（G-CPU-G256〜G4096。相手は PyTorch／candle／TensorFlow）

(1) と F1 の根拠になる GB10 の CPU 側 BLAS 経路は**確認できなかった**（`DNNL_VERBOSE` 出力が空。`GB-185`）。全セルの (1) は**未測定**。

| セル ID | (1) 相手の純カーネル・上限比 | (2) fandhe-ai の純カーネル・固定費 | (3) 再現で届く比 | (4) 工夫で届く比 |
|---|---|---|---|---|
| G-CPU-G256 | 未測定（PyTorch の CPU 経路は未確認。`BLAS_INFO=nvpl` は設定表示のみで実行時経路の証拠ではない） | 純カーネル: 未測定。reuse／fresh 窓比 1.885（推定・`GB-B`）。診断（`GB-187`）は 1.434（fresh 179.9 µs・reuse 258.0 µs）、帰属は matmul 区間の出力確保 100%（実測） | F1 未測定。F2 fresh 窓 1.39（推定。現状 0.74）。fresh は fandhe-ai 0.1947 ms | affinity ON は reuse 0.9979・fresh 0.8352（処置セル・非後退。実測・`GB-187`）。14 セル合算の判定は REJECT（§10.8）。ほかの候補は未測定 |
| G-CPU-G512 | 未測定 | reuse／fresh 窓比 1.447（推定）。診断 1.563（794.2→1241.6 µs）、帰属 matmul 区間 100%（実測・`GB-187`） | F1 未測定。F2 1.01（推定。現状 0.70） | affinity ON は reuse 1.0341・fresh 1.0857（後退側。実測）。host arena の A/B は reuse の逆転を是正しない（reuse_arena 1292.8 µs 対 reuse 1241.6 µs。arena hit 0・実測） |
| G-CPU-G1024 | 未測定 | reuse／fresh 窓比 1.390（推定）。診断 1.495（3105.5→4641.2 µs）、帰属 matmul 区間 115%（実測・`GB-187`） | F1 未測定。F2 1.35（推定。現状 0.98） | affinity ON は reuse 1.0043・fresh 0.9539（実測） |
| G-CPU-G2048 | 未測定 | reuse／fresh 窓比 1.570（推定）。診断 1.116（16091.6→17952.4 µs）、帰属 matmul 区間 101%（実測・`GB-187`） | F1 未測定。F2 1.68（推定。現状 1.06） | affinity ON は reuse 0.9993・fresh 1.0449（実測） |
| G-CPU-G4096 | 未測定 | reuse／fresh 窓比 0.985（推定）。診断（`GB-187`）の対象外（N=256〜2048 のみ） | F1 未測定。F2 1.20（推定。現状 1.21。reuse と fresh が並ぶ） | affinity ON は reuse 0.9892・fresh 0.9915（実測） |

- F2 は fresh 窓の fandhe-ai（結果を毎反復解放）に対する、scoreboard が採った最速相手の fresh 窓の比。PyTorch（N=256）・candle（N=512・1024）・TensorFlow（N=2048・4096）の窓は、いずれも `GB-B` の同一系列の中央値行。fandhe-ai の reuse が fresh より遅い原因は、結果を保持し続けるため毎反復新しい出力バッファの first-touch が起きること（`GB-187` の因果腕。tape／autodiff 機構ではない）で、是正は #3226
- N=4096 は reuse と fresh が並ぶ（0.985）ため、現状でも最速相手に勝つ（1.21）
- GB10 の CPU gemm の純カーネル時間（fandhe-ai・相手とも）は、Phase 9 では測っていない。F1 は計算できない

### 10.4 train／infer 8 セルの (1)〜(4)

train／infer は相手のステップ内訳を Phase 9 で測っていない。(1) は全セル**未測定**で、相手の窓だけを §10.2 に置いた（`MM-B`・`GB-B`）。(3) も同じ理由で**未測定**（PyTorch・burn が backward で第 1 層の入力勾配 dX を計算しない、という §4.3 の既存分析は `推定`（`LOOP`）のまま。省略の効果は #3244 系が未実装で未測定）。
(4) の F3 は、ゲートの A/B の after／before を窓に乗じた推定で、**ゲートの判定が ADOPT でない限り採用候補ではない**（判定は §10.8）。

| セル ID | (1) 相手の純カーネル・上限比 | (2) fandhe-ai の内訳・固定費 | (3) 再現で届く比 | (4) 工夫で届く比 |
|---|---|---|---|---|
| MM-MTL-TRN | 未測定（PyTorch MPS の窓 0.6888 ms のみ。candle 1.31 ms・burn 3.44 ms は `MM-B`・実測） | reuse step_total 1885.9 µs＝forward_resident 945.4・backward 928.4・tape_build 11.4・device_update 2.2（実測・`MM-180`）。fresh 2324.4 µs。窓（系列 B）は reuse 2132.9 µs で phases の step_total より広い（原因未切り分け）。同期は 1 step で encode 11・command buffer 7・wait 7（forward 側 4・backward 3。実測・HEAD の腕 B・版差）。空 dispatch の同期 14.5 µs・実カーネル同期の `host_fixed` 208.4 µs（単発 n=40）・外挿 259.1 µs／同期（推定）。wait 7 回の説明率は空 dispatch 基準 5.4%・実カーネル同期基準 77.4%（推定・`MM-180` §18.5。GPU 計算時間との分離は未了） | 未測定 | G3 train forward encode-only は train reuse 0.9000（実測・`MM-184`・undetermined。train fresh 1.0077 のため ADOPT 不成立）。F3 0.36（推定。現状 0.32）。常駐チェーン（#3248）・入力勾配の省略（#3244）は未実装で未測定。G1 は REJECT（train reuse 1.0090）、G2 は undetermined（train reuse 0.9951） |
| MM-MTL-INF | 未測定（candle の窓 0.5075 ms は `MM-B`・実測。run 別比は 1.31／0.71／0.90／1.40／1.13 でノイズ帯） | reuse predict_resident 330.5 µs・iter_total 331.0 µs（実測・`MM-180`）。窓（系列 B）388.4 µs。GPU 実行時間 gpu_busy は run2〜5 で 90.0〜90.9 µs（run1 は 175.6 µs）。run 内の裾は commit_wait−gpu_busy が +0.1〜0.12 ms（実測・`MM-183`）。実カーネル同期の `host_fixed` 208.4 µs（`MM-180`） | 水準ごとの推定（`MM-183`・candle は別セッション）: 低モード 0.32 ms 前提で系列 B の candle 0.507 ms に対し 1.58 倍・系列 A の 0.349 ms に対し 1.09 倍。bench 対照 0.427 ms 前提（第 2 ラウンドで再現せず）で 1.19 倍・0.82 倍。run1 の低性能状態 0.55 ms 前提で 0.92 倍・0.63 倍（推定。両系列で負け側）。前提水準の選択は §10.10 | G1 REJECT（infer reuse 1.0181）・G4 REJECT（同 1.1604）・G2 undetermined（同 0.9473）・G3 record_only（同 0.9434）。ADOPT になったゲートはない（実測・`MM-184`） |
| MM-CPU-TRN | 未測定（PyTorch の窓 0.2214 ms は `MM-B`・実測） | reuse step_total 778.4 µs＝forward 206.4・backward 451.5（58.0%）・device_update 117.6（実測・`MM-180`）。fresh 719.1 µs。窓（系列 B）1105.9 µs。backward 内部（GEMM／非 GEMM）の内訳は未取得。#2106 の in-situ `device_update` は 28.0〜30.0 µs で bench の区間値と約 4 倍の差がある（原因未調査） | 未測定 | SME ON は train 64 で after/before 0.9009（fresh）・0.9117（reuse）、5/5 で ≤1.00（実測・`MM-176`）。F3 0.22（推定。現状 0.20）。small-cap は train reuse 0.9832・fresh 1.0432（悪化）、big-core は 0.9564／0.9888（どちらも REJECT・`MM-178`）。m=64 の job 分割（jpw1／cap4／cap8）の Mac mini 比は 0.832〜1.141 で採否未定（記録のみ・`MM-177`） |
| MM-CPU-INF | 未測定（PyTorch の窓 0.0535 ms は `MM-B`・実測） | reuse iter_total 194.5 µs（predict_resident 194.0）。fresh 202.2 µs（実測・`MM-180`）。窓（系列 B）215.4 µs。`predict_resident` 内部の内訳は registry 版では取得不能 | 未測定 | SME は infer が非到達で 0.9821／1.0100（実測・`MM-176`）。small-cap は infer 64 で 0.8962（fresh）・0.8488（reuse）、5/5 round 改善（実測・`MM-178`）。F3 0.29（推定。ゲート判定は REJECT のため採用候補ではない。現状 0.25）。PyTorch 比は A/B 内で fresh 0.154→0.178・reuse 0.157→0.183 と 0.2 未満（実測・`MM-178`） |
| G-CUDA-TRN | 未測定（candle の窓 0.2717 ms は `GB-B`・実測） | reuse step_total 332.3 µs＝forward_resident 154.8（46.6%）・backward 165.0（49.7%）・device_update 7.9。fresh 511.1 µs（実測・`GB-189`・0.11.0）。同期 8（forward 3・backward 5）・launch 16・H2D 10・D2H 6・alloc／free 各 14（決定的・nsys・HEAD の腕 B・版差）。同期 1 回の固定費は sync_idle 0.29 µs・tiny_roundtrip 6.14 µs・launch_issue 1.82 µs・実カーネル同期 63.18 µs（kernel_wait の導出）。説明率は 0.6%／12.9%／133.1%、launch 集約の上限は 27.4 µs（step_total の 8.2%。推定） | 未測定 | 候補（#3248 常駐チェーン・#3258 学習ステップ Graph capture）は未実装で未測定。HEAD に入った #3202（forward の ReLU を epilogue 融合）・#3203（MSE の target 再アップロード廃止）は 0.11.0 に含まれず、前後 A/B（腕 C）は未計測（`GB-189` §19.7）。what-if の融合 relu は forward_resident の 0.64〜0.79 倍（record-only・bit 一致・batch 16〜1024）。G1（B1）は undetermined（train reuse 1.0172） |
| G-CUDA-INF | 未測定（PyTorch の CUDA 経路は cuBLAS 等で閉源。窓 0.0399 ms は `GB-B`・実測。candle 0.0417 ms も同水準） | reuse iter_total 99.7 µs（predict_resident 98.7）。chain 95.9 µs（download 69.2・うち導出 kernel_wait 63.2）・h2d_upload 11.3・l1_issue 7.5・l2_issue 6.6・上位固定費 2.9 µs。1 同期・2 launch（実測・`GB-189`）。相手との差 約 55 µs は相手側を同じ区間へ分解していないため説明率を数値化できない | 未測定 | B3 推論チェーンの Graph capture は reuse の batch 64 で 0.8610・全 6 セルが ≤1.00（実測・`GB-190`・undetermined。R1(b) が複数スレッド推論で失敗）。F3 0.47（推定・ADOPT 不成立。現状 0.40）。再計測は #3205 |
| G-CPU-TRN | 未測定（PyTorch の窓 0.3574 ms は `GB-B`・実測。経路は未確認） | 0.11.0 の reuse step_total 970.7 µs・device_update 289.0 µs（実測・`GB-189` の参考表）。0.11.0 相当の pre（T 279.1 µs）では SGD ループ以外の H4 が 215.2 µs（77%）、#3193・#3194 後の head（T 169.2 µs）では 142.8 µs（84%）。H4b（in-situ カーネル超過）95.5%・RAYON_NUM_THREADS=1 で 26.2 µs の分解は head 系列（実測・`GB-188`・因果は未確定） | 未測定 | #3193・#3194 は HEAD に実装済みで 0.11.0 に含まれない（device_update 289.0 µs が pre の 293.9 µs に近い）。bench-fandhe の pre→head は train reuse step_total 0.879・device_update 0.588（実測・`GB-188`）。F3 0.39（推定・窓に 0.879 を乗じる。現状 0.35）。affinity は train fresh 1.0662・reuse 1.1115（5/5 後退・REJECT・`GB-187`）。host arena（B1）は train reuse 1.0172（undetermined） |
| G-CPU-INF | 未測定（PyTorch の窓 0.1127 ms は `GB-B`・実測。経路は未確認） | 0.11.0 の reuse iter_total 183.9 µs・fresh 175.1 µs（実測・`GB-189`）。診断の L1 `linear_forward_device` 145.8 µs・L2 39.0 µs・upload 6.5 µs・readout 0.9 µs。reuse が fresh より遅い逆転は診断で 2.4〜3.5 µs・bench で 13.0〜17.0 µs（実測・`GB-188`・一部のみ説明） | 未測定 | affinity は infer fresh 0.7470・reuse 0.7704（実測・`GB-187`。14 セル判定は REJECT）。F3 0.77（推定・ゲート判定は REJECT のため採用候補ではない。現状 0.60）。#3196（融合 epilogue）の効果は bench で未確認（head／pre 1.018・同 `GB-188`）。host arena の infer は 1.0035（undetermined） |

- train の phases の `step_total` は通常ベンチの窓より狭く、系列 B の値と一致しない（Metal reuse は 1885.9 µs 対 2132.9 µs、cpu reuse は 778.4 µs 対 1105.9 µs。`MM-180`）。(2) の内訳は phases、(1) の窓は系列 B で、両者を足し引きしない
- Mac mini の実カーネル同期 208.4 µs は単発（n=40）で 5 run 化していない。GB10 の 63.18 µs は同期待ちにカーネル実行時間が混じる導出値で、同期 1 回の純固定費とは読めない
- 同期回数の削減で step_total の大半が取れる見込みは、GB10 では実測で支持されない（説明率 0.6〜12.9%）。Mac mini では実カーネル同期基準で 77.4% まで説明しうるが、1 回減らした A/B がない（`MM-180` §18.5）

### 10.5 (5) 届かない場合の理由（ハードの上限／計測窓の定義／自分たちの制約）

3 区分で書く。**自分たちの制約は数字を並べるだけ**で、制約の変更に関する判断は含めない。1 位のセルは「届いている」とし、読み方は §10.6 に置く。「未測定」は出典に値がないことを表す。

- ハードの上限: ハード・OS・相手の閉源ライブラリのどれかが決めている部分（同期の固定費・公称性能に対する到達率を含む）
- 計測窓の定義: 窓に何が入るか（Python checksum と Rust checksum・readout・fresh／reuse の窓構成・fandhe-ai 固有の upload／host_copy）で決まる部分
- 自分たちの制約: REQ-1（完全自作コア）・REQ-8（境界検査）・既定精度・`f64` アキュムレータ契約

#### 10.5.1 Mac mini・Metal（MM-MTL-G256〜G4096・TRN・INF）

| セル ID | ハードの上限 | 計測窓の定義 | 自分たちの制約（数字のみ） |
|---|---|---|---|
| MM-MTL-G256 | kernel_gpu 0.0197 ms は iter_total の 4.7%。commit_wait−kernel_gpu は 0.2132 ms で、candle の excl 0.2046 ms と同水準の同期固定費を相手も払う（実測・推定・`MM-182`） | 窓に fandhe-ai 固有の upload（8.9%）・host_copy（5.3%）と共通の checksum（13.1%）が入る。差 G は +0.0281 ms で小さく、割合として読めない（`MM-182` §12.6） | kernel_gpu が窓の 4.7% のため、カーネル側の制約の寄与はこの割合以下（推定）。境界検査の単独コストは未測定 |
| MM-MTL-G512 | candle excl 0.830 TFLOPS は MPS の 0.94 倍（同期固定費が残る領域・推定）。fandhe-ai の kernel_gpu は 2.34 TFLOPS で窓の 13.3% | 固有の upload（14.4%）・host_copy（8.4%）・共通 checksum（18.3%）。G は +0.1688 ms で、カーネル由来の割合は負値・1 超で読めない（`MM-182` §12.6）。reuse が fresh より +0.035 ms 遅い（`MM-182` §12.7） | カーネル側の寄与は窓の 13.3% 以下（推定）。境界検査の単独コストは未測定 |
| MM-MTL-G1024 | candle excl 2.26 TFLOPS は MPS の 0.98 倍（推定）。fandhe-ai の kernel_gpu は 2.50 TFLOPS | 固有 upload（17.3%）・host_copy（8.0%）・共通 checksum（18.8%）。固有分が G のほぼ全量（99.9%・`MM-182` §12.6）。reuse が fresh より +0.194 ms 遅い | カーネル側の寄与は窓の 28.6% 以下（推定）。境界検査の単独コストは未測定 |
| MM-MTL-G2048 | candle excl 3.15 TFLOPS は MPS の 0.97 倍（推定）。fandhe-ai の kernel_gpu は 2.13 TFLOPS（MPS 純時間の 0.66 倍・推定） | 固有 upload（13.0%）・host_copy（6.4%）・共通 checksum（15.9%）。G のうちカーネル由来 57.3%（(a)）〜70.2%（(b)）、固有 59.5%（`MM-182` §12.6。(a)(b)(固有) は排他的な内訳ではない）。reuse が fresh より +0.542 ms 遅い | カーネル側の寄与は窓の 57.5% 以下（推定）。M4 Max の N=4096 では candle 13.17 TFLOPS 対 自作 9.76〜9.91 TFLOPS（到達率 74〜75%・`M4M`・`docs/perf/metal-gemm-n4096-kernel-gap.md`）。境界検査の単独コストは未測定。MPS（ベンダー製）は REQ-1 により使わない構成で、MPS 純時間は 3.237 TFLOPS（`MM-179`） |
| MM-MTL-G4096 | candle excl 3.24 TFLOPS は MPS の 0.94 倍（推定）。fandhe-ai の kernel_gpu は 2.11 TFLOPS、最良タイル `[2]` で 2.58 TFLOPS（`MM-181`）。差のうちカーネル由来は 65.5%（(a)）〜72.2%（(b)） | 固有 upload＋host_copy は 11.35 ms で G の 32.6%。共通 checksum は 8.64 ms（削れない床・`MM-182` §12.5）。reuse が fresh より +5.19 ms 遅い。系列 B との窓の乖離は fandhe-ai −26.7%・candle −22.3% | カーネル側の寄与は窓の 72.4% 以下（推定）。M4 Max の到達率 74〜75%（上記・別機体）。境界検査の単独コストは未測定。MPS 純時間は 3.429 TFLOPS（`MM-179`） |
| MM-MTL-TRN | 実カーネル同期の固定費は 208.4 µs（単発・空 dispatch の約 14 倍）、空 dispatch は 14.5 µs。wait 7 回の説明率は 5.4%〜77.4%（`MM-180`・推定）。GPU 計算速度の差と同期固定費の分離は未了 | phases の step_total（1885.9 µs）は窓（2132.9 µs）より狭い。相手（PyTorch）のステップ内訳は未測定 | f64 アキュムレータ契約（勾配の長軸縮約）の単独コストは未測定。Metal の bias 勾配は 64bit 整数のソフトウェア・エミュレーションで bit 一致（`docs/metal-grad-reduction-parity-judgment-decision.md`）。境界検査の単独コストは未測定 |
| MM-MTL-INF（1 位・ノイズ帯） | 届いている（系列 B）。run 内の裾は commit_wait−gpu_busy が +0.1〜0.12 ms で、GPU 実行時間は不変（`MM-183`）。原因は未特定 | 窓の水準が 3 通りある（低モード 0.32 ms・bench 対照 0.427 ms・低性能状態 0.55 ms）。candle は別セッションの値（`MM-183`）。読み方は §10.10 | 該当なし（記録上、制約が効く箇所は特定されていない） |

#### 10.5.2 Mac mini・CPU（MM-CPU-G256〜G2048・TRN・INF）

| セル ID | ハードの上限 | 計測窓の定義 | 自分たちの制約（数字のみ） |
|---|---|---|---|
| MM-CPU-G256（1 位） | 届いている。相手の純 `mm` は 0.0240 ms で窓の 7.5%（推定。`MM-B` 窓 0.3195 ms に対する比） | PyTorch の窓に Python checksum（2.200 ns／要素＝0.144 ms・窓の 45%・推定）が入り、fandhe-ai 側の Rust checksum は 0.556 ns／要素（`MM-179`）。勝ち幅 1.04× の読み方は §10.6 | 該当なし |
| MM-CPU-G512（1 位） | 届いている。純 `mm` 0.1520 ms は窓の 10.8%（推定） | Python checksum 2.206 ns／要素（窓の 41%・推定）対 Rust 0.603 ns／要素（`MM-179`）。読み方は §10.6 | 該当なし |
| MM-CPU-G1024 | 純 `mm` は 1883 GFLOPS（対 2008 で 0.94）。ライブラリ同士では PyTorch の純 `mm` 1.1404 ms に対し fandhe-ai の窓−Rust checksum 7.2365 ms（推定。Accelerate は閉源） | PyTorch の窓の Python checksum は 2.312 ns／要素（窓の 35%・推定）。fandhe-ai の reuse 窓は fresh 窓より +0.246 ms（`MM-B`・推定） | SME は既定 OFF（`SME_PRODUCTION_ENABLED = false`）。ON にした場合の比は §10.3.2。新規 `asm!`・unsafe の監査は #3230 |
| MM-CPU-G2048 | 純 `mm` は 1768 GFLOPS（対 2008 で 0.88）。SME 経路の fandhe-ai は 2008 GFLOPS の 54% 止まり（(2048,2048,1024) の T=10・`MM-175`）。頭打ちは観測されない | Python checksum 2.388 ns／要素（窓の 28%・推定）。`MM-179` の窓は系列 B の 0.543 倍で再現せず、約 480 GFLOPS の差は 3 軸（総和・スレッド数・ビルド）では説明できなかった（`MM-179`） | SME 既定 OFF（上記）。micro-call ごとの `smstart`／`smstop` 固定費 44.5〜89.8 ns と ZA0 1 タイルの構成（`MM-175`）。多タイル化は新規 `asm!` の承認が前提（#3234・#3238） |
| MM-CPU-TRN | 未測定（PyTorch の CPU `mm` L1 fwd 1427・L1 dW 1519・L1 dX 1041・L2 系 136〜183 GFLOPS は `MM-179`・実測。L2 系は固定費が支配） | 窓（系列 B）1105.9 µs 対 phases step_total 778.4 µs の差は未切り分け（`MM-180`） | PyTorch・burn は第 1 層の dX を計算しない（3 GEMM）一方、fandhe-ai は VJP で計算してから捨てる形（§4.3・推定・`LOOP`）。単スレッド GFLOPS は L2 の細い形状（n=10・k=10）で 29〜40（正方 512 は 117.5）で、NR=12 に満たないタイルの無駄が疑われる（`MM-177`・原因は未確認） |
| MM-CPU-INF | 未測定（相手の窓 0.0535 ms は PyTorch の 2 GEMM。純 `mm` は小形状で固定費が支配・`MM-179`） | `predict_resident` が iter_total の大半で GEMM 以外の区間が事実上ない（`MM-180`・`M4M` 0.9.0 の §4.3 と同じ形） | SME は infer（m=64 を含む非到達形状）に効かない（0.9821／1.0100・`MM-176`）。単スレッドの L2 形状 29〜40 GFLOPS（`MM-177`） |

#### 10.5.3 GB10・CUDA と GB10・CPU

| セル ID | ハードの上限 | 計測窓の定義 | 自分たちの制約（数字のみ） |
|---|---|---|---|
| G-CUDA-G256 | cuBLAS FP32 は公称 31 TFLOPS の 0.06。N=256 は launch 往復の床 6.1 µs が窓 90 µs の約 7%（実測・推定・`GB-186`）。fandhe-ai の event 純カーネルは cuBLAS の 0.86 倍（HEAD の再測定は 0.781）で、カーネルは差の原因ではない（`GB-185`・`GB-215`）。実測の到達率は cuBLAS FP32 が 0.065、現行が 0.083（2.02／2.59 TFLOPS ÷ 公称 31。公称は第三者値） | 窓の大半は H2D 22.9 µs＋readback 25.1 µs（L0 の 90.9%）。相手側の転送時間は未分離。PyTorch の窓は Python checksum を含む（`GB-186`） | 該当なし（カーネルが差の原因でないため）。境界検査の単独コストは未測定 |
| G-CUDA-G512 | cuBLAS FP32 は公称の 0.25。event 比は 1.106（fandhe-ai が遅い。HEAD の再測定は 1.062）。到達率は現行 0.243・C1 0.259（7.54／8.04 TFLOPS ÷ 公称 31。`GB-215`） | reuse が fresh より +0.308 ms 遅い（窓の差・推定）。F2 の fresh 窓で 0.92 | 同上。境界検査の単独コストは未測定 |
| G-CUDA-G1024 | cuBLAS FP32 は公称の 0.50（15.65 TFLOPS）。fandhe-ai の wall は cuBLAS の 1.358 倍（時計が wall 同士の比）。event 同士では現行 1.325 倍（11.84 TFLOPS＝公称の 0.38）、C1 `tma128_strided` 1.128 倍（13.91 TFLOPS＝0.45）。残る差は純カーネル側（`GB-215`。ハードの上限ではない。cuBLAS が 0.51 に届いているため） | reuse が fresh より +1.333 ms 遅い。#3186 は retention_share 1.067（tape 保持による未タッチ宛先）と判定（`GB-186` §14.3）。F2 の fresh 窓で 0.97 | wall 0.1864 ms（`GB-185`）と event 0.1814 ms（`GB-215`）の差 0.005 ms が投入・同期のオーバーヘッド（別 run の値同士の推定）。C1 は FP32 strict・bit 一致のまま（ゲート A）で REQ-1 により cuBLAS は呼ばない。境界検査の単独コストは未測定（検査を外す実験は行わない） |
| G-CUDA-G2048 | cuBLAS FP32 は公称の 0.57。最速相手 burn の純カーネルは未測定。cuBLAS TF32 は公称の 0.76。fandhe-ai の event は現行 14.49 TFLOPS（公称の 0.47）、C1 `tma128_strided` 16.42（0.53）。cuBLAS FP32 の 17.36 に対し C1 でも 1.057 倍（`GB-215`） | reuse が fresh より +4.409 ms 遅い（retention_share 0.984）。F2 の fresh 窓で 0.99 | 既定精度は FP32（REQ-2 の複合判定）。最速相手 burn は TF32 固定。FP32 fresh 4.1854 ms 対 burn 4.1443 ms（0.99）、TF32 opt-in fresh 4.37 ms 対 burn（0.95）。境界検査の単独コストは未測定 |
| G-CUDA-G4096（1 位） | 届いている。cuBLAS FP32 は公称の 0.56（17.26 TFLOPS）、fandhe-ai は 13.7 TFLOPS（0.44）。cuBLAS TF32 は 41.73 TFLOPS（公称 53.3 の 0.78） | reuse と fresh は並ぶ（窓差 −0.54 ms）。fresh の readback 事前タッチ約 15 ms が残る（`GB-186` §14.3 判定 F）。PyTorch の窓は GPU 純カーネル 8.0 ms に対し約 67.6 ms（`GB-185`）。fandhe-ai の gemm セル reuse 38.65 ms のうち純カーネルは 10.34 ms（27%）で、差 28.31 ms は H2D・readback・checksum 等の合算（本 run では分解していない。Rust 逐次総和 0.51 ns／要素は既存実測で、約 8.6 ms は推定）（`GB-215`） | 最速相手 burn は TF32 固定。FP32 の fandhe-ai が burn に 1.01× で並ぶ。TF32 opt-in では 1.17×（§10.7）。C1 を入れても純カーネルは 13.79 TFLOPS（公称の 0.44）で、cuBLAS FP32 の 17.24（0.56）との差 1.25 倍が残る（実測・`GB-215`。原因の切り分けは未実施） |
| G-CPU-G256 | 未測定（相手の CPU 経路を確認できなかった）。GB10 は SME 非対応で NEON のみ（`MM-175` の対象外記載） | reuse が fresh より遅い（診断 1.434。帰属は matmul 区間の出力確保 100%・`GB-187`）。F2 の fresh 窓で 1.39 | 該当なし（原因は窓構成側。#3226） |
| G-CPU-G512 | 未測定（同上） | reuse／fresh 診断 1.563（matmul 100%）。F2 で 1.01 | 該当なし |
| G-CPU-G1024（僅差 0.98×） | 未測定（同上） | reuse／fresh 診断 1.495（matmul 115%）。F2 で 1.35 | 該当なし |
| G-CPU-G2048（1 位） | 届いている。未測定（相手の純カーネル） | reuse／fresh 診断 1.116（matmul 101%）。fresh 窓なら 1.68 | 該当なし |
| G-CPU-G4096（1 位） | 届いている。未測定（相手の純カーネル） | reuse と fresh が並ぶ（0.985）。PyTorch の cpu 腕の checksum は N=512 で 5.89・N=1024 で 4.91 ns／要素とばらつきが大きい（参考値・`GB-185`） | 該当なし |
| G-CUDA-TRN | 実カーネル同期 63.18 µs（kernel_wait の導出）。launch_issue 1.82 µs・tiny_roundtrip 6.14 µs・sync_idle 0.29 µs。説明率は 0.6〜12.9%（固定費基準）で大半は未説明（`GB-189`・推定） | 系列 B の窓 0.3309 ms と phases の step_total 332.3 µs はほぼ一致する（`GB-B`・`GB-189`）。相手（candle）の内訳は未測定 | f64 アキュムレータ契約の単独コストは未測定（CUDA は `double` を直接使用）。境界検査の単独コストは未測定 |
| G-CUDA-INF | 同期・launch の固定費は小さい（tiny_roundtrip 6.14 µs）。kernel_wait 63.2 µs（導出）が chain の大半。差約 55 µs の内訳は相手側未分解（`GB-189`） | PyTorch／candle の窓は Python 側のオーバーヘッドを含みうるが分解していない（未測定） | 該当なし。cuBLAS（閉源）は使えない構成 |
| G-CPU-TRN | 未測定（相手の経路未確認）。device_update の in-situ 超過 95.5%（head 系列）はマルチコア由来の状態を支持（規則 5）。機構は未特定（`GB-188`） | 窓と phases の関係は未切り分け | PyTorch・burn は dX を計算しない（3 GEMM）一方、fandhe-ai は計算してから捨てる形（§4.3・推定）。NEON のみ（SME 非対応） |
| G-CPU-INF | 未測定（相手の経路未確認） | reuse が fresh より遅い逆転は診断 2.4〜3.5 µs・bench 13.0〜17.0 µs で、診断の窓では再現しない部分が残る（`GB-188`） | 該当なし |

### 10.6 勝ちセルの読み方（Python checksum の実測による補正）

1 位のセル（MM-CPU-G256・G512・G-CPU-G2048・G4096・G-CUDA-G4096・MM-MTL-INF）のうち、窓構成の影響が実測で見えるのは Mac mini の CPU gemm である。**Issue 本文の「約 5 倍」は使わない**。実測の Python÷Rust は 3.66〜4.17（`MM-179`）。

| N | Python checksum [ns／要素・実測] | Rust checksum [ns／要素・実測] | Python÷Rust | PyTorch 窓 [ms・`MM-B`] | fandhe-ai 窓 [ms・`MM-B`] | scoreboard の比 | ライブラリ同士の比（i） | ライブラリ同士の比（ii） |
|---|---|---|---|---|---|---|---|---|
| 256 | 2.200 | 0.556 | 3.96 | 0.3195 | 0.3075 | 1.04× | 0.65 | 0.09 |
| 512 | 2.206 | 0.603 | 3.66 | 1.4034 | 1.3542 | 1.04× | 0.69 | 0.13 |
| 1024 | 2.312 | 0.605 | 3.82 | 6.9713 | 7.8709 | 0.89× | 0.63 | 0.16 |
| 2048 | 2.388 | 0.604 | 3.95 | 35.7591 | 51.7847 | 0.69× | 0.52 | 0.20 |

- ライブラリ同士の比（推定）は、両者の窓から自分の checksum 区間を除いた時間の比（相手 ÷ fandhe-ai。1 未満で相手のライブラリ部分が速い）。(i) は系列 B の窓から `MM-179` の Python checksum（×N²）を引いたもの＝（PyTorch 窓−Python checksum）÷（fandhe-ai 窓−Rust checksum）。(ii) は `MM-179` が実測した PyTorch の純 `mm`（0.0240／0.1520／1.1404／9.7186 ms）を分子にしたもの。分母は (i) と同じ fandhe-ai の窓−Rust checksum
- (i) と (ii) の幅は、`MM-179` の窓が系列 B の 0.52〜0.54 倍でしか再現しなかった（`MM-179`）ことに起因する。どちらの時間基準が有効かを決める材料はなく、**0.09〜0.69 の範囲で確定できない**。両方を残し、選択は §10.10 の所有者判断待ちにする
- Issue 本文が書く 0.33〜0.40 は (i) とも (ii) とも一致しない（(ii) は 0.09〜0.20、(i) は 0.52〜0.69）。本節は再計算値を採り、差を記録する（本文の値は出典のない見積り）
- 勝ち幅 1.04× のうち、PyTorch の窓に入る Python checksum の寄与は、系列 B の窓の 35〜45%（(i) の計算と同じ入力・推定）。窓の構成差が勝敗区分を決めている度合いは、この表では 2 つの時間基準のどちらを採るかで変わる
- G-CPU-G2048・G4096（TensorFlow 相手）・G-CUDA-G4096 の窓構成の補正は、相手の checksum を GB10 で実測した値（PyTorch の fp32 腕 2.56〜3.25 ns／要素。TensorFlow は未測定）に限られる。Rust 側は 0.51 ns／要素（`GB-185`）。TensorFlow 側の補正は**未測定**

### 10.7 TF32・3×TF32 の実測（GB10・CUDA GEMM。既定精度の判断は #3262 の領分）

下表は値の転記で、既定精度の是非には触れない。

| N | 区分 | 値 | 出典・機体・版 |
|---|---|---|---|
| 2048 | fandhe-ai fresh FP32／TF32 | 4.1854 ms／4.3700 ms | `GB-B`（0.11.0・実測。`results-dgx-0.11.0-median5.jsonl` の `tf32` の有無） |
| 2048 | candle fresh FP32／TF32 | 4.2211 ms／3.6952 ms | 同上 |
| 2048 | burn fresh（TF32 固定） | 4.1443 ms | 同上 |
| 4096 | fandhe-ai fresh FP32／TF32 | 39.6519 ms／33.9267 ms | 同上 |
| 4096 | candle fresh FP32／TF32 | 57.6643 ms／57.2268 ms | 同上 |
| 4096 | burn fresh（TF32 固定） | 39.6344 ms | 同上 |
| 4096 | 比（burn÷fandhe-ai）FP32／TF32 | 1.00／1.17（推定） | 上の値からの算出 |
| 1024〜4096 | cuBLAS TF32／FP32 の純カーネル時間比 | 0.49（N=1024）・0.43（2048）・0.41（4096） | `GB-185`（実測） |
| 512／1024／2048／4096／256×256×4096 | 3×TF32（`mma_tf32x3`）÷ f32 SIMT（TFLOPS 比） | 0.897／0.930／0.886／0.721／0.891 | `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md` §9（GB10・別の計測系列・実測。5 形状すべて f32 SIMT より遅い） |

- 3×TF32 は構造上 f32 SIMT と bit 一致せず、`CudaGemmPrecision::Tf32x3` は既定 OFF の明示 opt-in に限る（`.claude/rules/coding-rust.md`）。上の 3×TF32 の比は純カーネルの TFLOPS 比で、窓の比ではない
- TF32 opt-in の fresh 窓は N=2048 で FP32 より遅い（4.3700 対 4.1854 ms）。この窓では TF32 の効果は N=4096 でのみ見えている（1.17×）

### 10.8 ゲートの ADOPT／REJECT 一覧（Phase 9 の診断で判定が出たもの）

判定の語彙は各結論 doc の記載のまま転記する。本表が判定を出すものではない。「GB10」と「Mac mini」の欄の機体別の値は各 doc を正とする。

| ゲート | 機体 | 判定 | 出典 |
|---|---|---|---|
| G1 host arena（2104） | Mac mini／GB10 | **REJECT**（Mac mini の正式系列で train reuse が 1.0090・5 round 全て後退。GB10 の B1 は undetermined で、2104 は一機体の 5/5 一貫後退で REJECT） | `docs/perf/optin-ab-macmini.md`・`docs/perf/optin-ab-gb10.md` |
| G2 縮約逐次化（2102。T = 16384） | Mac mini／GB10 | undetermined（Mac mini は後退 2 セル、GB10 B2 も undetermined。最終 T の候補は 16384） | 同上 |
| G3 train forward encode-only（2113） | Mac mini | undetermined（前提ゲートは成立。train fresh 1.0077 で ADOPT 不成立、train reuse は 0.9000 で記録） | `docs/perf/optin-ab-macmini.md` |
| G4 readback 宛先（2112） | Mac mini | **REJECT**（gemm reuse N=1024 1.0228・N=4096 1.0074・infer reuse 1.1604） | 同上 |
| G5 readout borrowed（§15.1） | Mac mini | 継承集計器の総合判定は ADOPT（6 セル 0.9592〜0.9773）。#1520 の REJECT は覆さず、採否は #3265 | 同上 |
| G6 steel 候補（2111） | Mac mini | LU・T0U-LU-FB は UNDETERMINED、T0U・T0U-LU は **REJECT**（N=512 で 5/5 後退） | 同上 |
| G7 device verify cache（2114） | Mac mini | record_only（H4 支持。H1〜H3 は未確定） | 同上 |
| B3 推論チェーンの Graph capture（2115） | GB10 | undetermined（R1(b) が複数スレッド推論で失敗。再計測は #3205） | `docs/perf/optin-ab-gb10.md`・`docs/perf/infer-chain-graphcapture-cuda-ab.md` |
| B4 N=256 起動固定費（2109） | GB10 | 診断（H1・H3 を支持。採否ではない） | `docs/perf/optin-ab-gb10.md`・`docs/perf/cuda-gemm-small-launch-cost.md` §7.1 |
| B5 境界述語の事前計算（#537） | GB10 | 不採用（4 形状すべて 5/5 後退。対象は `MMA_F16` で、既定の FP32 SIMT のセルには非該当） | `docs/perf/cuda-gemm-boundary-predicate-ab.md` |
| B6／2108 readback pinned 再利用 | GB10 | **REJECT**（N=256・1024・2048・4096 の全 N で後退） | `docs/perf/cuda-gemm-readback-reuse-2108.md` §5・§5.1 |
| C1 TMA 2 段 128×64（#3215） | GB10 | `tma128_none`・`tma128_strided` とも ADOPT（結線候補。ゲート D は結線 PR で実施）。N=1024／2048／4096 の純カーネル比 1.145／1.124／1.035 と 1.175／1.133／1.038（5/5 一貫。ゲート A・B pass） | `docs/perf/logs/cuda-tma-stage2-ab-gb10-3215/aggregate.md`・`docs/backend-cuda-tma-gemm-load-design.md` §11.6 |
| SME ON（#3176） | Mac mini | ADOPT 候補（本番切替は #3230、しきい値は #3231） | `docs/perf/cpu-gemm-sme-ab-macmini.md` §5 |
| big-core 並列度（#3178） | Mac mini | **REJECT** | `docs/perf/cpu-thread-policy-macmini.md` |
| small-cap（#3178） | Mac mini | **REJECT** | 同上 |
| GB10 大コア affinity（#2117／#3187） | GB10 | **REJECT**（14 セル中 6 セルで ratio > 1.00） | `docs/perf/cpu-gemm-gb10-affinity-ab.md` |
| m=64 の job 分割 jpw1／cap4／cap8（#3177） | Mac mini／GB10 | 採否未定（記録のみ。#3198・#3227 への入力） | `docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md` §4 |
| Metal タイル選択（#3181） | Mac mini | 選択ロジックは不変。Mac mini 専用表は #3206 | `docs/perf/metal-gemm-tile-select-macmini.md` |

- 結線候補として残るのは SME ON（Mac mini）と C1 TMA 2 段 128×64（GB10。ゲート D は結線 PR）。G5 は継承集計器の語彙で ADOPT だが #1520 を覆さない扱い（#3265）
- ADOPT 条件には両機体の成立を要するゲートがあり（2104・2102）、GB10 側が undetermined の間は反転できる定数がない（`docs/perf/optin-ab-gb10.md`）
- HEAD に実装済みで 0.11.0 に含まれないもの（#3193・#3194・#3196・#3202・#3203）は A/B のゲートではない。効果は §10.4 に実測を置き、0.12.0 の再計測（#3273）で確認する

### 10.9 Phase 10・11 へのつながり（診断 → 前提にする実装 → 参照するセル）

親 issue の葉は `gh api repos/Fandhe-AI/fandhe-ai/issues/<親>/sub_issues` の結果（2026-10-22 時点。すべて open）。タイトルは要約で、番号と親子関係は API の結果のとおり。
「前提にする診断」は、その葉の設計・目標値が読む本 doc の節。目標値の決定は各 Phase の issue が行い、本節は入口の対応だけを示す。

| 親 | 葉（API の結果。要約） | 前提にする診断（Phase 9） | 参照するセル |
|---|---|---|---|
| #3197 CPU の m=64 GEMM の分割戦略 | #3198 job 統合・pack 共有・重み事前 pack の設計と A/B 基盤／#3199 job 統合の実装／#3200 pack 済み panel のキャッシュ／#3201 両機体 A/B で既定を決める | #3177（job 数・再 pack 倍率・分割 A/B。§10.4 の MM-CPU・G-CPU の train／infer）・#3178（small-cap の infer 改善） | MM-CPU-TRN・MM-CPU-INF・G-CPU-TRN・G-CPU-INF |
| #3207 Metal の m=64 向けタイル選択 | #3208 m=64 向け候補を既定 OFF で追加／#3209 Mac mini で A/B | #3181（正方のみ測定。m=64 は未測定） | MM-MTL-TRN・MM-MTL-INF |
| #3210 candle（MLX steel 系）の構造へ近づける Metal GEMM | #3211 unroll とレジスタ配置の候補（既定 OFF）／#3212 Mac mini で A/B | #3181（タイル最良でも N=2048・4096 は candle の 0.83・0.80 倍）・#3182（カーネル由来の割合 65.5〜72.2%） | MM-MTL-G1024〜G4096 |
| #3213 CUDA FP32 GEMM の上限比（約 5 割）の引き上げ | #3214 sm_121 向けカーネル候補（既定 OFF）／#3215 GB10 で A/B | #3185（cuBLAS FP32 の純カーネル 0.137／0.974／7.96 ms と fandhe-ai の wall 比 1.36／1.20／1.26） | G-CUDA-G1024〜G4096 |
| #3216 学習ステップをデバイスに置いたまま進める内部部品 | #3217 設計の決定記録／#3218〜#3220 Metal の ReLU マスク・MSE・dW／dX 入口／#3221〜#3223 CUDA の同等部品／#3224 実機 parity | #3180（Metal の同期 7 wait・同期固定費）・#3189（CUDA の同期 8・launch 16・固定費） | MM-MTL-TRN・G-CUDA-TRN |
| #3229 SME の本番化 | #3230 unsafe `asm!` の監査記録更新／#3231 しきい値の確定と既定 ON／#3232 前後を両機体で確認 | #3175（固定費 44.5〜89.8 ns・頭打ちなし）・#3176（ADOPT 候補。R4 格子・PyTorch 比） | MM-CPU-G512〜G2048・MM-CPU-TRN |
| #3233 m=64 専用の SME カーネル | #3234 設計と新規 unsafe の所有者確認／#3235 既定 OFF で実装／#3236 Mac mini で A/B | #3175（(64,256,784) の micro-call 256 回・T=1 で SME 時間の 10〜20%）・#3177 | MM-CPU-TRN・MM-CPU-INF |
| #3237 SME と NEON を job ごとに使い分ける異種並列 | #3238 分配を既定 OFF で実装／#3239 Mac mini で A/B | #3175（T=4→10 の低下・SME／NEON 比の縮小）・#3178 | MM-CPU-G256〜G2048 |
| #3240 A を pack しない NEON 直接カーネル | #3241 設計と新規 unsafe の所有者確認／#3242 既定 OFF で実装／#3243 両機体で A/B | #3177（A の再 pack 倍率 11〜22）・単スレッドの L2 形状 29〜40 GFLOPS | MM-CPU-TRN・MM-CPU-INF・G-CPU-TRN・G-CPU-INF |
| #3244 学習ステップで使わない入力勾配の計算を省く | #3245 契約の決定記録／#3246 `LinearResident` の VJP で不要な d_input の GEMM を省く／#3247 両機体で A/B | §4.3 の `LOOP`（PyTorch・burn は dX を計算しない。推定） | MM-CPU-TRN・G-CPU-TRN・MM-MTL-TRN |
| #3248 学習 1 ステップを 1 回の同期で実行する常駐チェーン | #3249 公開形の決定記録と所有者確認／#3250 入力の常駐ハンドル／#3251 Metal の encode-only チェーン／#3252 CUDA の同期なしチェーン／#3253 CPU 経路／#3254 公開／#3255 loss・更新後パラメータ一致テスト／#3256 実機 parity と同期回数／#3257 両機体で計測 | #3180・#3189（同期回数と固定費。同期回数の削減だけで GB10 の step_total の大半を取れる見込みは実測で支持されない） | MM-MTL-TRN・G-CUDA-TRN・MM-CPU-TRN・G-CPU-TRN |
| #3258 学習ステップ全体を CUDA Graph に capture | #3259 設計と所有者確認／#3260 既定 OFF で実装／#3261 GB10 で A/B | #3189（launch 集約の上限 27.4 µs＝step_total の 8.2%。学習ステップの step capture は #1350 で中立と判定済みで、`GB-189` §19 はその判定を覆さない）。推論チェーンの capture は B3（§10.8） | G-CUDA-TRN |
| #3263 bench-fandhe の計測条件を相手と揃える | #3264 非対称是正の範囲の決定記録／#3265 reuse で結果を反復ごとに解放し、判定が ADOPT なら Metal の readout を借用ビューにする／#3266 是正の前後を両機体で併記計測 | #3182（reuse が fresh より遅い・固有 upload＋host_copy）・#3186（tape 保持）・#3187（出力確保） | MM-MTL-G512〜G4096・G-CUDA-G512〜G2048・G-CPU-G256〜G2048 |

親 issue 配下ではない下流 issue（本 doc の各節が名指しするもの。状態は 2026-10-22 時点）:

| issue | 状態 | 要約 | 参照するセル |
|---|---|---|---|
| #3205 | open | CUDA 推論チェーンの固定費（B3 の再計測を含む） | G-CUDA-INF |
| #3206 | open | Mac mini（M4・GPU 10 コア）向けの Metal タイル表 | MM-MTL-G512〜G4096 |
| #3225 | open | CUDA の reuse の readback の遅さの是正 | G-CUDA-G512〜G2048 |
| #3226 | open | CPU で reuse が fresh より遅い逆転の是正 | G-CPU-G256〜G2048 |
| #3227 | open | ADOPT になった opt-in の既定を ON にする（現状、反転できるゲートはない。§10.8） | 全 opt-in 対象セル |
| #3262 | open | CUDA GEMM の既定精度の判断材料（§10.7 の値を入力にする） | G-CUDA-G2048・G4096 |
| #3273 | open | 0.12.0 の 6 フレームワーク同席計測（本表との突き合わせ先） | 27 セルすべて |
| #3195 | open | 要素数の小さい MSE backward を逐次で実行する（G2 縮約逐次化の関連） | MM-MTL-TRN・MM-CPU-TRN |
| #3193・#3194・#3196・#3202・#3203 | closed | HEAD に実装済み（0.11.0 に含まれない）。効果の確認は #3273 | G-CPU-TRN・G-CPU-INF・G-CUDA-TRN |

### 10.10 所有者判断待ち・未測定・限界

**所有者判断待ち**（本節は判断を先取りしない。どちらも未判断）:

| # | 論点 | 並べた値 | 出典 |
|---|---|---|---|
| 1 | `MM-179` の PyTorch の窓が系列 B の約 0.52〜0.54 倍で再現した件。系列 B の値をそのまま (3) の基準にするか、`MM-179` の窓で更新するか | 系列 B の窓（§10.2・§10.3.2 の F1・§10.6 の (i)）と `MM-179` の純 `mm`（§10.6 の (ii)）を併記。N=256 では (i) と (ii) で約 7 倍の差（ライブラリ同士の比 0.09〜0.69 の範囲） | `docs/perf/opp-kernel-macmini.md`「#3191 への申し送り」 |
| 2 | MM-MTL-INF の前提水準（低モード約 0.32 ms／bench 対照 0.427 ms／低性能状態約 0.55 ms） | §10.4 の 3 水準の比（1.58／1.09、1.19／0.82、0.92／0.63）。candle は別セッションの値 | `docs/perf/metal-infer-bimodal-macmini.md` §3.4・§6.5 |

**未測定**（推測で埋めない）:

- candle の GPU タイムスタンプ（Metal）・candle の純カーネル（CUDA）・burn（cubecl）の純カーネル・GB10 の CPU BLAS 経路と純カーネル（全相手）
- train／infer の相手のステップ内訳（8 セルすべて）。fandhe-ai の backward 内部（GEMM／非 GEMM）
- REQ-8 の境界検査の単独コスト（無効化した測定は規約上行えない）・`f64` アキュムレータの単独コスト
- MM-CPU-G256 の SME ON（A/B の対象外）・TensorFlow の checksum の ns／要素（GB10）
- fandhe-ai の N≥1024 の CUDA event 時計・N≤512 の wall（Rust の計装追加が要る。`GB-185` の残課題）
- 判定不能セル 1（GB10 の PyTorch CPU N=4096 の 1 要素）: 同セルの最速相手は有効行の TensorFlow（§10.2）。区分は `gen_0110_macmini.out` のまま

**限界**:

- **基準の混在**: 窓は系列 B、内訳は `MM-182`（A=0.11.0・B=HEAD の版差）・`MM-180`・`GB-189`（腕 A と腕 B の版差）。Metal N=4096 は系列 B との窓の乖離が fandhe-ai −26.7%・candle −22.3%（`MM-182` §12.6。原因は切り分けていない）。同一セッション内の値だけで F1・F3・F4 を算出したが、タイル比（`MM-181`）・steel 候補（`MM-184`）は別セッションの比を掛けている
- Mac mini の infer Metal は系列 B で 1 位でも、run 別比が 0.71〜1.40 にばらつく（系列 A は 3 位 0.72×）。1 位の区分は安定していない
- Mac mini の系列 B と GB10 の 0.11.0 は計測日・ツリーが違う（Mac mini は main `624d0ee4`、GB10 は 2026-10-10）。M4 Max の値は並置のみで算術に混ぜていない
- `MM-179` は作業ディレクトリのキャッシュを seed 決定的に作り直した系列で、0.11.0 当時の作業ディレクトリそのものではない（`docs/perf/opp-kernel-macmini.md`）。Rust の checksum は N=256 で #3182 F2 より低く、N=1024 以上で 0.06〜0.09 ns 高い（経路の違い。説明していない）
- Issue 本文の見積りのうち、リポ内の実測から式を示して再現できたのは約 1.17 倍（G-CUDA-G4096 の TF32 opt-in 対 burn の TF32。上表）のみ。Mac mini Metal N=4096 の約 1.37 倍、GB10 CUDA の窓天井 1.7〜2.3 倍、現行カーネルの実質天井約 1.15 倍・2.0 倍、Python checksum 補正後の 0.33〜0.40 は、**リポ内に出典がなく再現できなかったため本表に載せない**
- (3)(4) の算出値（F1〜F4）は目標値の合格ラインではない。F4 は実装も実測もない仮想上限で、候補の効果の見積りではない
- 本節は新規の計測・A/B・ソース解析を行っていない。コード・既定値・判定規則・tolerance・baseline は変更していない
