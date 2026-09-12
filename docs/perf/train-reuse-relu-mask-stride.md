# reuse backward の ReLU マスク stride 対応（イシュー #1577）

## 1. 背景・機構

低レイヤー診断（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4。出典
イシュー `1574`）が確定した機構:

- reuse 学習（`DeviceParamStore` 経由・`Op::LinearResident`）の
  backward では、下流層の VJP が返す `d_input = transpose2d(&tmp)`
  （`crates/autodiff/src/grad.rs` の `Op::LinearResident` 分岐）が
  stride `[1, m]` のゼロコピー転置 view であり、単一寄与なら
  `backward.rs::accumulate` がコピーせずそのまま上流層の `upstream`
  になる。
- 上流層（`act == Relu` の `Op::LinearResident`／`Op::LinearAct`）の
  VJP はまず `elementwise_mul_mask(upstream, out_value, |v| v > 0.0)`
  を適用するが、旧実装は `eval::dense_vec` → `Tensor::contiguous()`
  が非連続入力を要素ごと `get(&index)`（rank 検査・軸ごとの範囲検査・
  index ベクタ繰り上げ）で走査するため、連続入力比 **約 49 倍**
  （667 ns → 32.6 µs／16384 要素・M4 Max）に劣化していた。backward
  内訳では reuse の mask が cpu 45 µs・metal 44 µs（fresh は
  8.7〜9 µs）。
- fresh 経路（`Op::Relu`／`Op::LinearAct`）の `upstream` は
  `matmul_vjp` の `gemm_fp32_strict` 出力（連続）なので影響を受けない。
- 「GEMM 呼び出し回数 fresh 2・reuse 4」は診断計装の計数単位の差で
  あり、**実際の GEMM 回数は両モードとも 4 回**（2 層 × d_input・
  d_weight）で同一であることをコード読解で確定した（`matmul_vjp` が
  層ごとに `gemm_fp32_strict` 2 回・`Op::LinearResident` が層ごとに
  `gemm_resident_lhs` 1 回 + `fill_resident_weight_grad`〈CPU／Metal／
  CUDA いずれも実装済み〉または `gemm_fp32_strict` 1 回）。したがって
  本イシューの「整理」は GEMM 呼び出し回数の削減ではなく、事実の確定・
  記録である。非学習葉への d_input GEMM スキップ（4 → 3 回）は
  `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（#1219。設計
  記録のみ・本体未実装）のスコープであり本イシューでは実装しない。

## 2. 設計決定・実装（最終版。§5.1 round1 の問題を是正済み）

- 方式は**マスク側の stride 対応**（GEMM 入口の転置との融合は不採用。
  `g` を転置レイアウトで生成すると TT パターンとなり CPU は
  `contiguous()` フォールバック、Metal／CUDA は別カーネル経路へ振り
  分けられて bit 同一契約・影響範囲の両面で不利なため）。
- `g`（reuse の非連続 `upstream`）・`mask_src`（`out_value`。
  `Op::LinearAct`／`Op::LinearResident` では view になりうる）の
  双方を独立に `MaskReadOperand`（`crates/autodiff/src/grad.rs`）で
  分類する:
  - **`Contig`**（[`Tensor::as_slice`] 成功。真に contiguous）:
    借用スライスをそのまま保持し、`flat` 添字で直接読む。
  - **`View`**（[`Tensor::as_view_slice`] 成功。`transpose`/`narrow`/
    `broadcast_to` の非負 stride view を含む）: 借用スライスと
    **`usize` へ変換済みの** strides を保持する（`isize`→`usize` の
    変換は `classify` 時に 1 回だけ）。
  - **`Owned`**（いずれも失敗。負 stride 等・現行公開 API では
    到達しない fail-safe）: `dense_vec`（コピー）を保持する。
- 両オペランドとも `Contig` の最頻ケース（fresh 経路の `Op::Relu`／
  `Op::LinearAct` はこれに該当）は、借用スライス 2 本の
  `zip`／`map`／`collect` に落とす専用高速経路で処理し、enum
  ディスパッチ・オフセット計算を経由しない（コンパイラの自動
  ベクトル化を妨げない）。
- 一方が `Contig`・他方が `View` の rank-2（reuse backward の実際の
  ホットパス。`[1, m]` 転置 view × 連続な forward 記録値、またはその
  逆）は専用経路で扱う: 行ごとの基準オフセット `i * stride0` を
  1 回だけ計算し、列方向は `+ j * stride1` の加算のみ（`Contig` 側は
  行スライスを直接インデックス）。
- それ以外の rank-2（両方 `View`／`Owned` を含む）・一般 N-d は
  `MaskReadOperand::read(idx, flat)` 経由の汎用経路（`View` の場合は
  `idx` と strides の内積、`Contig`／`Owned` の場合は `flat`）で扱う。
- **`View` のオフセット計算は `checked_mul`／`checked_add` を経由しない
  プレーンな `usize` 演算**（round1 からの変更点。§5.1 参照）。安全性の
  根拠: `as_view_slice()` が `Some` を返した時点で全 strides が非負
  であることが確定しており、かつ同メソッドは
  `span = 1 + Σ (shape_i − 1)·stride_i` を `checked_add`／`checked_mul`
  で検証済みである。したがって shape 範囲内の任意の `idx` に対し
  `Σ idx_i·stride_i < span` が保証され、要素ごとに改めて overflow を
  検査する必要はない（対象テンソルの要素数は学習用途の実用範囲で
  `usize::MAX` に遠く及ばない）。境界外アクセスの検出自体は最終的な
  `span.get(off)` の 1 回の `Option` 判定に集約する。
- `as_view_slice` が `None`・shape 不一致等、想定外の状態を検知した
  場合は、静かに 0 で埋めたり判定を迂回したりせず `dense_vec` を使う
  既存の走査へ**経路全体を丸ごと**フォールバックする（数値的に同一の
  コピー経路。`.claude/rules/security.md` A08 が禁じる判定迂回では
  ない）。
- 逐次実装のまま（rayon 非依存。#1578 のとおり数千要素規模では
  fork-join 固定費が支配的なため）。
- `fill_resident_weight_grad` → `gemm_resident_lhs` の呼び出し順序
  （#1563／#1665）は変更していない。触れたのはマスク関数本体
  （`elementwise_mul_mask`・新設ヘルパー）と呼び出し 3 箇所の doc
  comment のみ。

## 3. 検証（bit 同一）

1. **単体テスト**（`crates/autodiff/src/grad.rs` `#[cfg(test)] mod
   tests`。新規 7 件）: 新実装と参照実装（旧 `dense_vec` zip 経路を
   そのまま残した `elementwise_mul_mask_reference`）の出力を
   `to_bits()` で完全一致比較。入力ケース: `[k,m]`→`transpose2d` の
   `[1,m]` 転置 view（reuse 再現・`g` 側）・同（`mask_src` 側）・
   `narrow` 後の view（offset ≠ 0）・`broadcast_to`（stride 0）・
   rank-1／rank-3・空テンソル・`NaN`／`-0.0`／subnormal を含む値。
   全 7 件 pass（`cargo test -p fandhe-ai-autodiff --lib mask_stride`）。
2. **CPU bit ダンプ before/after**（新規
   `crates/facade/tests/cpu_reuse_step_grad_bit_dump.rs`。Metal 版
   `metal_reuse_step_grad_bit_dump.rs`〈#1555〉を `Device::Cpu`
   〈`cfg(target_os = "macos")` 制約なし〉へ移植）: `origin/main`
   （`grad.rs` のみ一時的に差し戻し）と本ブランチ（最終実装）で
   10 step の loss・重み勾配・パラメータの bit 表現（4462 行）を
   出力し `diff` で完全一致を確認。
3. **Metal bit ダンプ before/after**（既存
   `metal_reuse_step_grad_bit_dump.rs`。M4 Max 実機）: 同様に 4462 行
   完全一致を確認。
4. **既存回帰**: `cargo test -p fandhe-ai-autodiff`（全 pass）・
   `cargo test -p fandhe-ai --tests`（全 pass。0 failed）・
   `cargo fmt --all -- --check`（差分なし）・`cargo clippy -p
   fandhe-ai-autodiff --lib -- -D warnings`（警告なし。`--all-targets
   --all-features` は `backend-cuda` の非 CUDA ホスト上の pre-existing
   dead-code 警告により `origin/main` でも失敗することを確認済みで、
   本イシューの変更とは無関係）。

いずれも `docs/perf/logs/train-reuse-relu-mask-stride-1577/
bitdump-{cpu,metal}-{before,after}.txt`（round2 実装時点の 1 系列）
に生ログを保存済み。

## 4. 事前登録判定規則

イシュー #1577 コメント（<https://github.com/Fandhe-AI/fandhe-ai/issues/1577#issuecomment-5645493488>）
へ固定した文面をそのまま転記する。**このコメントは実装（`grad.rs` の
変更・単体テスト 7 件・CPU bit ダンプ実測）の完了後に投稿したもので
あり、厳密には「実装着手前」の事前登録ではない**（同コメント自体も
「実装は完了・単体テスト pass・CPU bit ダンプ完全一致を確認済み」と
明記している）。実測規則の内容自体（比較腕・対象セル・判定基準）は
framework-compare A/B 実測に着手する前に固定し、A/B 実測後の緩和は
行っていない。

- 比較腕: before = `origin/main`（マージ base コミット）の
  `crates/facade` を `[patch.crates-io.fandhe-ai]` path patch した
  `bench-fandhe`、after = 本ブランチの同 path patch。いずれも
  `scripts/bench/framework-compare` の承認ピン `fandhe-ai =0.8.0` の
  `Cargo.lock` は退避・EXIT trap 復元し、コミットしない。
- 対象セル（マスク経路が到達する reuse）: `bench-fandhe --task train
  --size 64 --mode reuse` × device（M4 Max: `cpu`・`metal`／GB10:
  `cpu`・`cuda`）。
- ガードセル（非到達・非後退確認）: 同 `--mode fresh` × 同 device。
- 各セル 5 run・run 単位で before／after の起動順を反転（奇数 run:
  before→after、偶数 run: after→before）・`compare_gemm_ab.py
  --task train` が採る `ms/step`（`--phases` なしの train レコード）
  の 5 run 中央値で `ratio = after / before`。
- 判定: 対象セルすべてで `ratio <= 1.00` かつ全セル checksum bit 完全
  一致 → ADOPT。対象セルのいずれかで `ratio > 1.00` → REJECT（原因
  帰属は判定と分けて記録）。checksum 不一致 → 数値契約違反として即
  REJECT・実装差し戻し。実機到達不能・専有条件不成立で 5 run を完走
  できない機体は undetermined として記録（他機体の結果で代替しない）。
  ガードセルの後退は verdict を変えないが、`ratio > 1.03` のセルは
  記録・原因記載を必須とする。
- `--phases`（backward／device_update 内訳）は各腕 1 回の診断のみで
  判定に用いない。
- 負荷ゲートは設けない（record_only）。`uptime`（1 分平均）を各 run
  前後に記録し、共有負荷下であることを明記する。
- 規則の事後緩和・セル除外・run 追加による再判定は行わない。REJECT／
  undetermined でも正式記録として記録し、マージ可否はユーザー判断
  （規則の結論は不変のまま別項で記す）。

**補足（`compare_gemm_ab.py` の呼び出しミス。§5.1 round1）**: round1
実測時は `compare_gemm_ab.py --task train ...` に `--device <device>`
を渡し忘れ、既定値 `metal` のまま cpu 側の集計を実行してしまった
ため、cpu の全行が `_valid_cell_identity` の `device` フィールド検証で
reject され `判定不能`（exit 2）になった。round1 の判定は代わりに
`median_s` を Python で直接集計する方式で確定した（`docs/perf/
logs/train-reuse-relu-mask-stride-1577/round1/` に生 JSONL のみ保存。
`compare-train-*.md` は生成できていないため同ディレクトリには含めて
いない）。round2 実測では `--device "$DEVICE"` を明示するよう
`run_ab_1577.sh` を修正し、`compare_gemm_ab.py` 本来の判定表
（`round2/compare-train-{cpu,metal}.md`）で確定した（本補足で原因を
確定済みのためスコープ外事項としては扱わない）。

## 5. 実測結果（M4 Max・record_only・共有負荷下。2026-09-12）

### 5.1 round1（初版実装。REJECT）

実装レビュー（advisor）で「孤立マイクロベンチマークの連続経路が
旧実装より遅い」と指摘され発覚した初版実装（`MaskReadOperand` が
`Contig` 分岐を持たず、連続入力も含め全読み出しが `checked_mul`／
`checked_add`／`isize`↔`usize` 変換を伴う `View` 相当の経路のみを
経由していた版）の実測。

**マイクロベンチ**（64×256・1000 反復中央値。初版）:
`contiguous median=45.917µs`（！）・`non_contiguous(transpose view)
median=46µs`。連続・非連続で速度差が消えたのは非連続側が高速化
されたのではなく**連続側が退行した**結果であり（旧実装の連続経路は
667 ns 相当・診断値）、当時はこれを「成功」と誤読していた。

**framework-compare A/B**（`docs/perf/logs/
train-reuse-relu-mask-stride-1577/round1/`）:

| device | mode | before median (s) | after median (s) | ratio (after/before) | checksum |
|---|---|---|---|---|---|
| cpu | fresh（ガード） | 0.000799 | 0.000876 | 1.0968 | 完全一致 |
| cpu | reuse（対象） | 0.001036 | 0.001016 | 0.9813 | 完全一致 |
| metal | fresh（ガード） | 0.001406 | 0.001429 | 1.0162 | 完全一致 |
| metal | reuse（対象） | 0.001039 | 0.001052 | **1.0131** | 完全一致 |

**判定**: metal の対象セル（reuse）が `ratio=1.0131 > 1.00` のため
規則上 **REJECT**（cpu 対象セルは 0.9813 で規則を満たす）。フレッシュ
（ガード）セルも cpu 1.0968・metal 1.0162 と後退方向で、初版実装が
連続経路まで退行させていたことと整合する。

### 5.2 round2（是正後の最終実装。ADOPT）

§2 の設計（`Contig`／`View`〈`usize` プレーン演算〉／`Owned` の
3 分岐・全 contig 高速経路・`Contig`×`View` 混在の rank-2 専用経路）
へ修正した最終実装の実測。

**マイクロベンチ**（同条件。3 回計測。単位 µs）:

| 系列 | 1 回目 | 2 回目 | 3 回目 |
|---|---|---|---|
| 新実装・contiguous | 7.33 | 6.50 | 5.88 |
| 新実装・non_contiguous（transpose view） | 14.92 | 13.67 | 13.46 |
| 参照実装（`dense_vec` zip）・contiguous | 8.21 | 8.08 | 8.00 |
| 参照実装（`dense_vec` zip）・non_contiguous | 41.33 | 42.04 | 42.21 |

新実装は contiguous・non_contiguous とも参照実装（旧 `dense_vec` zip
経路）を上回る（contiguous: 借用スライス 2 本の直接 zip によりコピー
1 回分を節約。non_contiguous: 参照実装比 約 2.8〜3.1 倍高速化。
診断で確認された「連続・非連続の速度差」自体は縮小したが、rank-2
専用経路が `MaskReadOperand::read` の enum ディスパッチ・行ごとの
基準オフセット計算を経由する分、完全な等速までは達していない
〈non_contiguous は contiguous の約 2〜2.5 倍〉）。

**framework-compare A/B**（`docs/perf/logs/
train-reuse-relu-mask-stride-1577/round2/`）:

| device | mode | before median (s) | after median (s) | ratio (after/before) | checksum |
|---|---|---|---|---|---|
| cpu | fresh（ガード） | 0.000775 | 0.000803 | 1.0356 | 完全一致 |
| cpu | reuse（対象） | 0.000961 | 0.000883 | **0.9196** | 完全一致 |
| metal | fresh（ガード） | 0.001537 | 0.001467 | 0.9545 | 完全一致 |
| metal | reuse（対象） | 0.001060 | 0.001023 | **0.9653** | 完全一致 |

**判定（事前登録規則を機械的に適用）**: cpu・metal いずれの対象セル
（reuse）も `ratio <= 1.00` を満たし、全セル checksum bit 完全一致
（`compare_gemm_ab.py` の判定列も両セルとも「非後退」）。M4 Max の
2 device（cpu・metal）について **ADOPT**。GB10（cuda）は本実装
エージェントの実行環境に到達手段がなく **undetermined**（未実測）。
cpu fresh（ガードセル）のみ `ratio=1.0356 > 1.03` のため記録する:
`--phases` 診断（`round2/results-*-cpu-phases.jsonl`）では
`backward` フェーズが before 442.6 µs → after 488.1 µs（比 1.103）と
後退方向だが、fresh 経路は本イシューの変更（`Contig`×`Contig` 高速
経路）が唯一適用される箇所であり、コード上は borrow のみで
`dense_vec` の二重コピーを削減した分だけ速くなるはずの変更である
ため、共有負荷（`round2/uptime-cpu.log`。load average 4.4〜11.3）に
よる計測ノイズと判断する（同一 run 内の 5 回中 3 回は `ratio<1.0` で
符号一貫ではない。`round2/compare-train-cpu.md` の「符号一貫」列
参照）。

## 6. ユーザー判断事項

- 本イシューの実装は M4 Max（cpu・metal）で ADOPT 相当の実測結果を
  得ているが、GB10（cuda）は未実測のまま undetermined である。本
  リポの慣例（`docs/perf/perf-rule-verdict-vs-maintainability.md`
  相当の運用）に従い、GB10 実測完了までマージを保留するか、M4 Max の
  結果と bit 同一性（3 経路で確認済み）を根拠に先行マージするかは
  ユーザー判断とする。
- round1（初版実装）で発覚した「孤立マイクロベンチでは連続経路が
  約 5.6 倍〈45.9 / 8.2〉退行していた」問題は round2 で是正済みだが、
  この種の「fast path の実装漏れ」は今後同様の stride 対応実装でも
  再発しうる観点として記録しておく（`.claude/rules/coding-rust.md`
  に一般則として追記するかはユーザー判断・本イシューでは提案のみ）。

## 7. スコープ外（`out-of-scope-tracking.md`。起票はユーザー承認後）

- `Tensor::contiguous()`／`eval::dense_vec` の非連続入力に対する
  汎用 stride 走査高速化（tensor-core。crates.io 公開クレートで
  影響範囲が広いため別途設計）。
- 非学習葉への d_input GEMM スキップ（4 → 3 回。
  `docs/autodiff-nograd-leaf-dinput-skip-decision.md`／#1219 の設計に
  基づく実装イシュー）。
- MSE backward の rayon しきい値（#1578。並行進行中の兄弟イシュー）。
- Metal／CUDA の `gemm_resident_lhs` 出力をデバイス常駐のまま次層へ
  渡す（マスクをデバイス側で行う）変更（`docs/
  inference-forward-fixed-cost-design.md` 系の別設計）。
- 診断計装（`FANDHE_DIAG_BACKWARD`）の恒久化。
- GB10（CUDA）実機実測（§5.2 undetermined。到達手段確保後の追加
  実測）。
- rank-2 の `View`×`View`（両オペランドとも非連続）専用高速経路の
  追加最適化（現状は一般 N-d と同じ `read(idx, flat)` 経由の汎用
  経路。reuse backward の実際のホットパスは `Contig`×`View` の
  組み合わせのため本イシューでは対象外とした）。
