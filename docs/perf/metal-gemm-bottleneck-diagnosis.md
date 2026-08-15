# Metal GEMM 1024 以降スループット頭打ち 定量診断（#487）

イシュー #487「test(backend-metal): 1024 以降のスループット頭打ちを定量診断（occupancy・帯域・バリア回数）」の
実測記録テンプレート。親 #480「GEMM 最適化の計測前提確定・実機プローブ・ボトルネック診断」の A-7 として、
Metal GEMM が実測（`docs/perf/metal-gemm-dynamic-tile.md`・#381 実測）で size=1024 以降スループットが
頭打ちになる要因（ロード律速か演算律速か、occupancy 不足か）を定量特定し、Phase D（#530）の 2 タスクの
優先順位を確定する:

- **D-2（#533）**: staged 協調ロードの float4 ベクトル化
- **D-7（#541/#542）**: occupancy 目標算出のタイル選択への組み込み

## 状態: 解析値算出は完了（Linux worktree）。**TFLOPS・実効帯域の実測は未実施（Mac 実機実行待ち）**

本イシューは「実装変更を伴わない調査・計測・記録タスクのみ」（親 #480 本文）のため
`crates/backend-metal/src/`・`shaders/gemm.metal` は変更していない。実行環境は Mac 実機
（`docs/real-hardware-verification-env.md` §1・§7「ローカル直接実行」）だが本セッション環境は Linux のため、
#188 → #380/#381 の先例に従い、Linux 側で完了できる範囲（診断 example の実装・解析値の算出・doc の計測
手順・記録テンプレート・判定基準の確定）のみを本 PR で行う。**§4「実測結果」・§5「結論」は Mac 実機セッション
で `cargo run -p backend-metal --example gemm_diagnosis --release -- --gpu-core-count=40 --ideal-groups-multiplier=6`
を実行してから記入する（CLI 引数は macOS 実行時必須。§1 参照）。**

## 1. 計測手段

`crates/backend-metal/examples/gemm_diagnosis.rs`（本イシューで新規作成）。

- **解析値**（`analytics` モジュール。`objc2` 系 FFI に触れない純粋関数のため非 macOS でも実行できる）:
  `tile::select(size, size, size)` の選択結果から threadgroup 数（並列度〈concurrency/saturation〉
  ヒューリスティックの一次指標。レジスタ・threadgroup memory 等の資源制約を表さないため真の occupancy
  ではない — 詳細は `gemm_diagnosis.rs` の `analytics::DeviceProfile` ドキュメント「本ヒューリスティック
  の限界」節・codex-review 指摘・PR #649 参照）・バリア回数・理論ロード/ストアトラフィック・arithmetic
  intensity を算出する
- **実測値**（macOS 限定。`macos_impl` モジュール）: `MetalGemm::dispatch_auto` を
  `bench-harness::protocol::run`（`MeasurementConfig::default()` = warmup 20 回・計測 20 回・中央値/Q1/Q3。
  TASK-8.1）で壁時計計測する

### 実測部分の設計判断（計画からの変更点）

計画（#487 実装計画 §3.1）は独自パイプライン構築による `MTLCommandBuffer::GPUStartTime`/`GPUEndTime`
直接採取を第一候補としていたが、`crate::pipeline::make_pipeline_with_constants`
（`crates/backend-metal/src/pipeline.rs:108`）が `pub(crate)` であり本 example（クレート外）から呼べない
ことを実装時に確認した。そのため計画のフォールバック節に従い、既存公開 API（`MetalGemm::dispatch_auto`）を
壁時計計測する経路を採用した。

`dispatch_auto` は呼び出しごとに A・B のアップロード・C の readback を含む（`gemm_bench.rs` と同じ計測範囲）
ため、`wall_ms` はカーネル純時間ではなく「A・B アップロード＋カーネル実行＋C readback」の end-to-end 時間
である。

#### 転送時間分離を試みて撤回した経緯（PR #649）

当初は同一 `(m, n)` で `k` を小さくした点（`k=8` 単独 → `k=8`・`k=32` の 2 点線形外挿）を「転送時間
ベースライン」として `wall_ms(size)` から差し引き `kernel_ms_approx`・`tflops_approx`・`eff_bw_gbs_approx`
を算出していたが、`tile::select` は `k < 64`（`SMALL` 閾値未満）で `SINGLE_SIMDGROUP_8X8`（`bm=bn=8`・
`staged=false`）を選ぶため、この参照点は `actual_groups = ceil(size/8)^2` という実測対象（staged 64×64
タイル）とは全く異なる大量の threadgroup をディスパッチする。この参照点の壁時計時間は A・B 転送だけでなく
直接ロード形式カーネルの演算・ディスパッチオーバーヘッドを主として含み、しかもその演算量も `k` にほぼ
比例して増えるため、2 点間の傾きを「転送レート」として `k = size` まで外挿すると staged カーネルとは無関係
な演算時間を転送時間の名目で拡大して差し引くことになる。結果として `kernel_ms_approx` がゼロに近づき
（ときに負になり `max(0.0)` 後の除算で `inf` が出力される）、`tflops_approx`・`eff_bw_gbs_approx` の基礎が
成立しなかった（イシュー #487 PR #649 への cursor[bot]・codex-review・Cursor Bugbot の各指摘。review id
4943646199 ほか）。

GPU timestamp 直接採取または演算を伴わない同量転送のみの対照経路はいずれもクレート内部（`pipeline.rs` 等）
へのアクセスを要するが、本イシュー（親 #480 本文）は「実装変更を伴わない調査・計測・記録タスクのみ」であり
`crates/backend-metal/src/`・`shaders/gemm.metal` の変更はスコープ外である。そのため転送・カーネルの分離は
**試みない**方針へ変更した。`wall_secs` を size ごとの end-to-end 指標としてそのまま報告し、そこから導出
する性能指標は「`wall_secs ≥ kernel_secs`（転送時間は非負）」という不等式のみから成立する**下限値**に
限定する:

```
tflops_lower_bound            = flops / wall_secs / 1e12
logical_load_gbs_lower_bound  = (load_bytes_total + store_bytes_total) / wall_secs / 1e9
```

`tflops_lower_bound` は実カーネル TFLOPS の健全な下限（転送時間の分だけ実際はこれより高い）。
`logical_load_gbs_lower_bound` は **DRAM 実効帯域ではない**（`load_bytes_total` はキャッシュ再利用を考慮
しない論理ロード量。キャッシュヒットにより実 DRAM トラフィックはこれより少なく、546GB/s 公称帯域との比較
には使わない。codex-review 指摘。PR #649）。分離を諦めた代わりに、ロード律速の仮説判定・並列度
〈concurrency/saturation〉ヒューリスティックによる一次観察は非 macOS でも算出できる解析値
（`arithmetic_intensity`・`actual_groups`/`ideal_groups`・`barriers_per_tg`。`ideal_groups` が真の
occupancy を表さない点は §1「解析値」節参照。§3）を主に用いる方針へ §5 の判定基準も改訂した。

計画 §3.1 が想定していた「example 内パイプライン vs `dispatch_variant` の数値照合」は、独自パイプライン構築を
行わない本設計では対象がないため実施しない（本 example は `MetalGemm::dispatch_auto` という既存の検証済み
公開 API をそのまま呼ぶのみで、新規カーネル実行経路を追加しない）。

## 2. 計測手順（Apple Silicon 実機）

```sh
git fetch origin
# 一時的な PR ブランチ test/487-metal-gemm-bottleneck-diagnosis はマージ後に
# 削除されうるため恒久参照にしない（codex-review 指摘。PR #649）。以下は
# 「現時点の main 最新」を checkout する簡便手順であり、これ自体は main の
# 更新に伴い指す commit が変わる（恒久参照ではない）。再現性を担保する恒久
# 参照は「実際に checkout された commit SHA を §4.1『計測コミット SHA』へ
# 記入する」ことで得る（下記 `git rev-parse HEAD` の出力）。マージ前に本 PR
# で先行検証する場合は origin/main の代わりに
# origin/test/487-metal-gemm-bottleneck-diagnosis を指定する
git checkout --detach origin/main
git rev-parse HEAD   # この SHA を §4.1「計測コミット SHA」へ記入する
cargo run -p backend-metal --example gemm_diagnosis --release -- \
    --gpu-core-count=40 --ideal-groups-multiplier=6
```

出力形式（1 行 1 size。`size=<N> tile=<bm>x<bn>x<bk>(<wm>x<wn>, staged=<bool>) actual_groups=<v>
ideal_groups=<v> barriers_per_tg=<v> arithmetic_intensity=<v> wall_ms=<v> wall_q1_ms=<v>
wall_q3_ms=<v> tflops_lower_bound=<v> logical_load_gbs_lower_bound=<v>`）を size=512/1024/2048/4096
で出力する。`wall_ms`（中央値）に加え `wall_q1_ms`・`wall_q3_ms`（`bench_harness::protocol::run` が
返す `Measurement::{q1,q3}_secs`）を計測手順（§1「中央値/Q1/Q3」）どおり出力し、破棄しない
（codex-review 指摘。PR #649）。

`ideal_groups` の算出（`idealGroups = gpu_core_count * ideal_groups_multiplier`）に用いる
`gpu_core_count`・`ideal_groups_multiplier` は、macOS 実行時は `--gpu-core-count`・
`--ideal-groups-multiplier` の**両方の明示指定を必須**とする。`MTLDevice` に公開の GPU コア数
取得 API は存在せず、`sysctl -n hw.model` の機種識別子（例: `Mac16,6`）だけでは同一機種内の
構成差異（binned 版等）まで保証できないため、機種識別子からの自動判定は行わない（codex-review
指摘 P1・PR #649。未指定は fail-closed でエラー終了する）。実機検証環境（M4 Max）で診断する場合は
上記コマンド例のとおり `--gpu-core-count=40 --ideal-groups-multiplier=6` を指定する。CLI 値は
正数（ゼロ拒否）・乗算オーバーフロー不可であることも検証される（codex-review 指摘 P2・PR #649）。

GPU 使用率のサンプリング（ベンチ実行と並行して別ターミナルで実行。sudo 不要）。`ioreg` は単発スナップショットの
ため、`cargo run`（4 size 分の warmup 20 回・計測 20 回を含む全実行区間。概算で数十秒〜数分）と並走させて
0.5 秒間隔で継続サンプリングし、テキストへ記録する（1 回だけの取得では計測区間内の変動・ピークを捉えられない
ため。codex-review 指摘。PR #649）:

```sh
# macOS 標準 date（BSD date）はサブ秒指定子 %N 非対応（GNU coreutils 拡張のため）。
# 秒単位のタイムスタンプで十分（0.5 秒間隔サンプリングの前後関係が分かればよい）
while true; do
    date +%H:%M:%S
    ioreg -r -d 1 -w0 -c IOAccelerator | grep "Device Utilization"
    sleep 0.5
done | tee /tmp/gpu_utilization_$(date +%Y%m%d_%H%M%S).log
```

`cargo run` 終了後に `Ctrl-C` でサンプリングを停止し、記録したログから size ごとの実行区間（stdout の
`size=<N> ...` 行のタイムスタンプ、または実行順 size=512→1024→2048→4096 と経過時間の対応）に該当する
「Device Utilization」値の最大値・中央値を §4.2 の「GPU 使用率（ioreg）」列へ記入する（単一の代表値では
なくレンジ・代表値の両方を残す）。

計測衛生: AC 電源接続。他 GPU 負荷アプリ（ブラウザ動画・Xcode ビルド・ローカル LLM 等）は終了してから計測する
（`docs/perf/metal-gemm-dynamic-tile.md`「計測環境」節と同方針）。

## 3. 解析値の事前計算（Linux worktree で算出済み。`cargo run -p backend-metal --example gemm_diagnosis` の
非 macOS stub 出力をそのまま転記）

### 3.1 出典・前提

| 項目 | 値 | 出典 |
|------|-----|------|
| GPU コア数（M4 Max） | 40 | `docs/perf/metal-gemm-dynamic-tile.md:53`（`sysctl -n hw.model` = `Mac16,6` 実測記録） |
| occupancy 判定式 | `idealGroups = coreCount * 6` | MFA（Metal FlashAttention）の FP32 系 occupancy 判定式（イシュー #487 計画「occupancy 不足の判定」節が出発点として指定した式。一次資料の直接引用は未確認のため、判定に用いる際は経験則として扱う） |
| メモリ帯域公称値 | 546 GB/s | Apple 公表スペック（M4 Max）。§5.1 の仮説生成用 machine balance point（FP32 理論ピーク ÷ 546GB/s）の算出にのみ使う。`arithmetic_intensity` はキャッシュ再利用を無視した論理ロード量ベースのためこの値との比較は実 DRAM 実効帯域比の確定算出にはならない（§1「実測部分の設計判断」節・§5.1 参照） |
| FP32 理論ピーク演算性能 | **要記入（Mac 実機セッションで一次資料を確認してから記入する）** | — |

### 3.2 size 別解析値（`tile::select(size,size,size)` は 4 サイズとも `staged=true` の 64×64 ブロック・
`bk=16`・`wm=wn=2` を選択する。`load_bytes_total`/`store_bytes_total` は threadgroup 間・K タイル間の
キャッシュ再利用を考慮しない理論下限値）

| size | tile (bm×bn×bk, wm×wn) | actual_groups | ideal_groups (=240) | actual/ideal | barriers_per_tg | load_bytes_total | store_bytes_total | flops | arithmetic_intensity (FLOP/byte) |
|------|------------------------|---------------|----------------------|---------------|------------------|-------------------|--------------------|-------|-----------------------------------|
| 512  | 64×64×16, 2×2 | 64   | 240 | 0.267 | 64  | 16,777,216    | 1,048,576   | 268,435,456     | 15.0588 |
| 1024 | 64×64×16, 2×2 | 256  | 240 | 1.067 | 128 | 134,217,728   | 4,194,304   | 2,147,483,648   | 15.5152 |
| 2048 | 64×64×16, 2×2 | 1024 | 240 | 4.267 | 256 | 1,073,741,824 | 16,777,216  | 17,179,869,184  | 15.7538 |
| 4096 | 64×64×16, 2×2 | 4096 | 240 | 17.067| 512 | 8,589,934,592 | 67,108,864  | 137,438,953,472 | 15.8760 |

（`crates/backend-metal/examples/gemm_diagnosis.rs` の非 macOS stub 実行結果をそのまま転記。算出式は
`analytics::analyze` 参照。`barriers_per_tg` は staged 経路のバリア位置〈`gemm.metal:427,441`〉が K タイル
ループ内に 2 回ずつ・エピローグにバリアなし〈ファイル末尾 498 行まで確認〉であることに基づく
`2 * ceil(size/16)`）

### 3.3 解析値からの暫定観察（Mac 実機実測前の中間所見。§5 で確定結論とする）

- **`actual_groups` は size=1024 で `ideal_groups`（240）を超え始める**（actual/ideal = 1.067）。size=512
  は 0.267 と明確に下回るが、512 は診断対象の「1024 以降の頭打ち」の範囲外であり、1024 以降は
  `actual_groups` が `ideal_groups` を大きく上回る一方（2048: 4.267 倍・4096: 17.067 倍）でも頭打ちが
  観測されている。`ideal_groups` はレジスタ・threadgroup memory 等の資源制約を表さない
  concurrency/saturation の proxy（MFA 経験式。真の occupancy ではない — §1「解析値」節・codex-review
  指摘・PR #649 参照）であるため、この比だけから「occupancy が過剰／不足である」とは確定できない点に
  注意した上で、**「発行 threadgroup 数が経験的な飽和目標を下回っている」という意味での並列度不足は
  1024 以降の頭打ちの説明にならない**（1024 以降はむしろ発行 threadgroup 数が飽和目標を大きく上回る
  領域であり、真の occupancy 不足〈資源制約由来〉が原因であれば別途レジスタ・threadgroup memory 使用率
  の実測で裏付ける必要がある）
- **arithmetic intensity は size に依らずほぼ一定（15.06〜15.88 FLOP/byte）**。GEMM の理論 AI は本来
  `size` に比例して増大するはずだが、本カーネルの staged タイルはキャッシュ再利用をせず K タイルごとに
  device メモリへ再ロードするため AI が頭打ちになっている（タイル構造由来の定数）。FP32 理論ピークとの比
  （machine balance）が確定すれば、この一定値との比較から「ロード律速の仮説」を生成できる（§5.1「ロード
  律速の『仮説』判定」参照。`arithmetic_intensity` は論理ロード量ベースのため、この比較だけでロード律速を
  **確定**はできない）。§3.1「要記入」欄が確定してから §5 で仮説として評価する
- 上記 2 点から、暫定的には **並列度不足の仮説（D-7）よりロード側の定数コスト（D-2 のベクトル化候補）が
  頭打ちの説明として有力**に見えるが、これは§4 の `wall_ms`・`tflops_lower_bound`（size 間の相対的な
  スループット傾向。転送時間分離による近似 TFLOPS・実効帯域は PR #649 で撤回済み）を補助証跡として、
  実測後に裏付けを取ってから確定する

## 4. 実測結果（Mac 実機セッションで記入）

### 4.1 計測環境

| 項目 | 値 |
|------|-----|
| チップ | 未計測 |
| OS | 未計測 |
| rustc | 未計測 |
| 計測コミット SHA | 未計測 |
| 計測プロトコル | `bench-harness::protocol::run`（`MeasurementConfig::default()` = warmup 20 回・計測 20 回・中央値/Q1/Q3。TASK-8.1） |
| 決定的シード | `0xC0FFEE`（`crates/backend-metal/examples/gemm_diagnosis.rs::SEED`） |
| 計測衛生 | 未計測 |

### 4.2 実測値

`tflops_lower_bound`・`logical_load_gbs_lower_bound` はいずれも `wall_secs`（A・B アップロード＋カーネル
実行＋C readback の end-to-end 時間）を分母とする**下限値**であり、公称帯域（546GB/s）との直接比較には
使わない（「実測部分の設計判断」節参照）。

| size | wall_ms（中央値） | wall_q1_ms | wall_q3_ms | tflops_lower_bound | logical_load_gbs_lower_bound | GPU 使用率（ioreg） |
|------|---------------------|------------|------------|----------------------|--------------------------------|------------------------|
| 512  | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |
| 1024 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |
| 2048 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |
| 4096 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 | 未計測 |

## 5. 判定基準・結論（Mac 実機実測後に確定）

### 5.1 判定基準（事前定義）

`tflops_approx`・`eff_bw_gbs_approx`（転送時間分離による近似値）は PR #649 で撤回した（「実測部分の設計
判断」節参照）。`logical_load_gbs_lower_bound` は論理ロード量ベースの下限値でありキャッシュ再利用を考慮
しないため公称帯域（546GB/s）との比較には使えない。そのため判定は非 macOS でも算出できる解析値
（§3）を主とし、`wall_ms`・`tflops_lower_bound` は size 間の相対的なスループット傾向（頭打ちの有無）の
補助証跡として扱う:

- **ロード律速の「仮説」判定（確定判定ではない）**: `arithmetic_intensity`（§3.2 表。size に依らずほぼ一定
  15.06〜15.88 FLOP/byte）は threadgroup 間・K タイル間のキャッシュ再利用を無視した論理ロード量を分母とする
  値であり、実 DRAM トラフィックではない（「実測部分の設計判断」節参照）。そのため `FP32 理論ピーク演算性能
  ÷ 546GB/s` という **DRAM 側の** machine balance point（§3.1「要記入」欄が確定してから算出）と直接比較
  しても、両者はメモリ階層が異なり（`arithmetic_intensity` はキャッシュヒット分を含めない論理値、
  machine balance point は DRAM 実効帯域が前提）本来比較できない。`arithmetic_intensity` が machine
  balance point を下回ることは「実 DRAM 帯域で律速している」ことの確定的な証拠にはならず、あくまで
  「キャッシュ再利用込みでも DRAM 律速の可能性を排除できない」という**仮説生成**に留める。この仮説を
  確定判定へ格上げするには、Metal System Trace 等による実 DRAM トラフィック・実効帯域の実測が必要
  （codex-review 指摘。PR #649）。仮説が成立する場合は **D-2 を優先候補とする**（確定ではない）
- **並列度〈concurrency/saturation〉不足の仮説判定（真の occupancy 判定ではない）**:
  `actual_groups < ideal_groups`（§3.2 表）となる size 帯が頭打ち開始点（1024 前後）と一致する場合、
  D-7（occupancy 目標算出のタイル選択への組み込み）を優先候補とする。ただし `ideal_groups` は
  レジスタ・threadgroup memory・`threads-per-threadgroup` 上限といった真の occupancy を決める資源制約
  を表さない concurrency/saturation の proxy（MFA 経験式）に過ぎないため、`actual_groups >=
  ideal_groups` が成立しても真の occupancy 不足が存在しないことの証明にはならない（codex-review 指摘。
  PR #649）。この仮説を確定判定へ格上げするには Xcode GPU counters 等によるレジスタ・threadgroup
  memory 使用率の実測が必要
- ロード律速の仮説が不成立（実測で machine balance point を上回ると判明）・並列度〈concurrency/saturation〉不足の仮説判定も不成立の
  場合、またはロード律速の仮説が実 DRAM トラフィック実測なしで確定判定に格上げできず結論が定まらない場合は、
  バリア同期コスト（`barriers_per_tg` の増加傾向。§3.2 表参照）・タイル選択自体の再検討（`tile::select` が
  1024 以降も一律 64×64 を選び続ける点）・`tflops_lower_bound` の size 間の伸び方（頭打ちしているか）を
  解釈指針として追加検討する

### 5.2 結論・D-2/D-7 優先順位（未確定。Mac 実機実測後に記入）

未計測のため未確定。§3.3「解析値からの暫定観察」の時点では並列度不足の仮説側の説明力が弱く D-2 優先を示唆するが、
これは実測（§4）による裏付けが必要な暫定所見であり、確定結論として扱わない。

## 6. 参照

- `docs/perf/metal-gemm-dynamic-tile.md`（#188・#381。頭打ち現象そのものの実測記録・出典）
- `crates/backend-metal/src/tile.rs`（`TileConfig`・`select` の実装。選択閾値は暫定値である旨がファイル
  冒頭コメントに明記されている）
- `crates/backend-metal/src/shaders/gemm.metal`（`gemm_simdgroup_tiled` の staged 協調ロード・バリア位置）
- `crates/backend-metal/examples/gemm_diagnosis.rs`（本診断の計測本体）
- 親 #480・Phase D 親 #530・D-2 #533・D-7 #541/#542
