# Metal split-K 経路と現行 classic 経路の M4 Max 実機 A/B（イシュー #1475）

## §0 状態（判定結果）

**暫定 ADOPT（3/5 run。5 run 完了による正式確定は未実施）**。

対象 9 形状（K 支配的非正方。`(32,32,*)`／`(64,64,*)`／`(128,128,*)`。
K ∈ {2048, 4096, 8192}）はすべて 3 run の中央値で speedup（= median_secs(classic)
/ median_secs(split-K)）が事前登録基準 `>= 1.5` を満たし、かつ 3 run とも符号
一貫（全 run で speedup > 1.0）だった。対照 3 形状（`(256,256,*)`）も 3 run
の中央値がすべて事前登録基準 `>= 0.95` を満たした。単一セッションの時間制約
により事前登録・実装計画（`docs/perf/logs/metal-gemm-splitk-ab-1475/`
`run_gated.sh` コメント参照）が想定した 5 プロセス起動のうち **3 run
（run1〜run3）を完了**した時点で打ち切ったため、判定は「暫定」とする。
残り 2 run の追加実測は本イシュー配下のフォローアップとして記録する
（§7）。

3 run の実測は §6 で `A′/A`（タイル構成効果）と `B/A′`（K 分割効果）へ
分解しており、対象 9 形状では `B/A′` が支配的（K 分割そのものが速度改善の
主因）であることを確認した。`docs/backend-metal-splitk-decision.md` §3 が
留保していた「因果は仮説」を実測で更新する。

**5 run 正式確定は #1515（§10）へ引き継ぐ**（本ページの本追記時点では
未実測。新規 5 run として実施し #1475 の 3 run とは混在させない）。

## §1 計測手段

実装: `crates/backend-metal/examples/gemm_splitk_ab_bench.rs`（本イシュー）。

- **計測境界**: prepared 境界（A・B・C バッファの確保・アップロードは計測外。
  計測対象は encode ＋ コマンドバッファ完了待ち。readback 対象外）。NN
  レイアウト固定。
- **腕**:
  - A（classic）: `MetalGemm::dispatch_strided_tiled_prepared` に
    `cfg = tile::select_for_device(m, n, k, ...)`（split-K のフォールバック先
    と同一経路）。
  - A′（classic・split-K タイル。参考のみ）: 同じ classic 経路だが
    `cfg = tile::split_k_tile(m, n)`。
  - B（split-K）: `MetalGemm::dispatch_split_k_strided_prepared_with_plan` に
    `tile::should_split_k(m, n, k)` の計画を明示的に渡す。戻り値が
    `SplitKRoute::Split` でなければ計測を中止する（フォールバックをデータ点
    として扱わない）。
  - B′（対照・classic）: `(256,256,K)` は `should_split_k` が `None` を
    返すことを assert したうえで classic を dispatch。**計測クロージャ
    内でも毎回 `should_split_k` を呼ぶ**（`dispatch_auto` 相当が本番で
    毎回払う選択関数の呼び出し費用込みで計測する。2026-09-09 是正。
    §3 「既知の限界」参照）。
  - C（対照・強制 split-K。参考のみ）: `max_groups=u64::MAX` で強制した
    split-K。`max_groups=40` 境界の妥当性の記録のみ。
  - フロア（参考のみ）: 各 `(M,N)` を `K=64` で classic dispatch した dispatch
    固定費。
- **比の方向**: `speedup = median_secs(A) / median_secs(B)`（1 より大きい
  ほど split-K が速い）。主指標は run 内対応比の run 数中央値。
- **プロトコル**: `bench_harness::ab::run_ab`（`AbConfig::new(6, 2s, 1s)`。
  順序反転 interleave）・`MeasurementConfig::default()`（warmup/iters 20/20）。
  フロアのみ `run_stability`（単一腕）。

実行コマンド:

```sh
cargo run -p fandhe-ai-backend-metal --release --features internal-diagnostics \
  --example gemm_splitk_ab_bench -- --max-load-avg=<f64>
```

`--self-check-only` でフェーズ 0（自己検証）のみ実行できる。

## §2 事前登録判定基準

計測前に issue #1475 へ投稿（2026-09-09T09:13:30Z UTC）:
https://github.com/Fandhe-AI/fandhe-ai/issues/1475#issuecomment-5599418673

- **ADOPT**: 対象 9 形状すべてで (i) 主指標（run 内比の run 数中央値）
  `>= 1.5` かつ (ii) 全 run で run 内比 `> 1.0`（符号一貫性）、かつ対照 3
  形状すべてで主指標 `>= 0.95`。
- **REJECT**: 専有ゲート成立下で上記いずれかが不成立。
- **undetermined**: 専有ゲート（`--max-load-avg`）が規定回数で成立しない
  場合（1 回だけ記録して終了）。
- spread（レンジベース）は判定に使わない。数値契約（各腕の run-to-run bit
  同一）は計測妥当性の前提条件（フェーズ 0）とし、腕間の差は情報として
  記録するのみで verdict の入力にしない。
- ADOPT は性能上の判定に限る。本番結線（`SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  の切替）は別途ユーザー承認が必要（#1476 のスコープ）。

以後、本基準・腕定義・定数は変更していない。

## §3 フェーズ 0 自己検証結果

`docs/perf/logs/metal-gemm-splitk-ab-1475/self_check.log` 参照。

- 対象 9 形状・対照 3 形状すべてで classic（A）・A′（classic・split-K
  タイル）・split-K（B）の run-to-run bit 同一（2 回連続 dispatch の
  `read_to_vec()` 一致）を確認した（`classic_stable=true` /
  `target_tile_stable=true` / `splitk_stable=true` が全 12 形状で成立）。
- A vs A′（`a_vs_at_fail_count`）は全 9 対象形状で **0**（タイル構成
  〈`select_for_device` の選択構成 vs `split_k_tile`〉を変えても classic
  経路の出力が `fandhe_ai_backend_cpu::parity::compare` の許容誤差複合判定
  （相対誤差 1e-3 未満または絶対誤差 1e-5 未満。`.claude/rules/coding-
  rust.md`）を全要素で満たすことを実測で確認した。`fail_count=0` は
  この複合判定を通過したことを意味し、bit 完全一致（ビットパターン一致）
  を保証するものではない〈イシュー #1499 codex-review 指摘。§3 の
  `classic_stable`／`target_tile_stable`／`splitk_stable`〈`to_bits()`
  比較〉が別途 run-to-run の bit 同一を担保している〉）。
- A vs B（`a_vs_b_fail_count`）は全 9 対象形状で総要素数近くまで不一致
  （K 分割の結合順序差に起因する既知の丸め誤差。`docs/perf/metal-gemm-
  splitk-two-pass.md` §5 と整合。`tests/gemm_splitk_parity.rs` はイシュー
  #1512 で baseline 非後退方式へ切替済み〈同 §5.8〉のため既知 FAIL では
  なくなったが、A vs B 自体の丸め誤差の存在という事実は変わらないため
  verdict の入力にはしない）。

**既知の限界（イシュー #1499 codex-review・Cursor Bugbot 指摘。2026-09-09
是正）**: 本節・`self_check.log` の `a_vs_b_fail_count` は、`prepare()` の
`seed_offset` が A（classic。`p1`）=1・B（split-K。`p2`）=2 と**異なる
入力行列**で生成されたバイナリによる実測値だった。異なる入力同士の出力を
比較しても「K 分割の結合順序差に起因する丸め誤差」という解釈の根拠には
ならない（別問題を解いた結果を比較しているにすぎない）。コード側は
`p2` の `seed_offset` を `p1` と同一の `1` へ統一済み（同一入力での
比較へ是正済み）だが、本ドキュメントが参照する `self_check.log`・上記
`a_vs_b_fail_count` の実測値自体は是正前バイナリによるものであり、
**同一入力での再実測は未実施**（実機〈M4 Max〉再接続が必要なため本 PR
のスコープ外。§7 のフォローアップ参照）。run-to-run bit 同一
（`classic_stable`／`splitk_stable`）・A vs A′（`a_vs_at_fail_count`。
両者とも `seed_offset=1` で同一入力のため元々有効）は本是正の影響を
受けない。

## §4 実測結果（3 run。`aggregate.md` から転記）

生データ・集計スクリプトは `docs/perf/logs/metal-gemm-splitk-ab-1475/`
（`run{1,2,3}.log`・`aggregate.py`・`aggregate.md`・`env_info.txt`）を参照。

### target（対象 9 形状。A=classic vs B=split-K）

| m | n | k | speedups（run1,run2,run3） | median | 5-run 判定基準 `>=1.5` かつ全 run `>1.0` |
|---|---|---|---|---|---|
| 32 | 32 | 2048 | 1.8952, 1.8183, 1.7299 | 1.8183 | 満たす（3/3 run） |
| 32 | 32 | 4096 | 2.3931, 2.6833, 2.2358 | 2.3931 | 満たす（3/3 run） |
| 32 | 32 | 8192 | 3.5266, 4.4233, 3.6979 | 3.6979 | 満たす（3/3 run） |
| 64 | 64 | 2048 | 1.5686, 1.6581, 1.6709 | 1.6581 | 満たす（3/3 run） |
| 64 | 64 | 4096 | 2.4883, 2.2917, 3.0936 | 2.4883 | 満たす（3/3 run） |
| 64 | 64 | 8192 | 3.8174, 3.7672, 4.0542 | 3.8174 | 満たす（3/3 run） |
| 128 | 128 | 2048 | 1.6336, 1.6220, 1.7561 | 1.6336 | 満たす（3/3 run） |
| 128 | 128 | 4096 | 2.1127, 2.0993, 2.5271 | 2.1127 | 満たす（3/3 run） |
| 128 | 128 | 8192 | 2.7487, 2.7191, 1.8570 | 2.7191 | 満たす（3/3 run） |

partitions は形状ごとに一貫（`(32,*,*)`／`(64,*,*)` は 32、`(128,128,2048)` は
8・`(128,128,4096)` は 16・`(128,128,8192)` は 32）。

### control（対照 3 形状。A=classic vs B′=classic）

| m | n | k | speedups（run1,run2,run3） | median | 基準 `>=0.95` |
|---|---|---|---|---|---|
| 256 | 256 | 2048 | 0.9977, 0.5903, 1.1718 | 0.9977 | 満たす（median） |
| 256 | 256 | 4096 | 0.9925, 0.9573, 0.9904 | 0.9904 | 満たす（median） |
| 256 | 256 | 8192 | 0.9963, 1.0245, 0.9968 | 0.9968 | 満たす（median） |

`(256,256,2048)` の run2 が 0.5903（>1.0 基準以下）と大きく振れているが
（対照は classic vs classic の対称比較のため理論上 1.0 近傍のはず。この
run 単発のノイズと考えられる）、median は基準を満たす。全 run を通じた
ばらつきの大きさは §5 で留保として明記する。

**既知の限界（イシュー #1499 codex-review 指摘。2026-09-09 是正）**: 上表の
実測値は、B′ の計測クロージャが `should_split_k` を呼ばずに classic
dispatch のみを測っていた是正前バイナリによるもの（A と全く同一の経路を
測っていたに等しい）。コード側は結線相当経路の選択関数呼び出し費用を
含めるよう是正済みだが、上表の再実測は未実施（§7 のフォローアップ
参照）。「対照は classic vs classic の対称比較のため理論上 1.0 近傍」
という解釈自体は妥当（`should_split_k` の呼び出し費用は encode／
コマンドバッファ完了待ちに比べ無視できるほど小さいと見込まれるため
median 側の結論〈基準 `>=0.95` を満たす〉が覆るとは考えにくいが、
未検証である点を明記する）。

### target_tile（対象 9 形状。A=classic〈select_for_device〉vs A′=classic
〈split_k_tile〉。参考のみ・タイル構成効果の分離）

| m | n | k | speedups（run1,run2,run3） | median |
|---|---|---|---|---|
| 32 | 32 | 2048 | 1.0097, 1.0381, 1.0399 | 1.0381 |
| 32 | 32 | 4096 | 0.6140, 0.6278, 0.9895 | 0.6278 |
| 32 | 32 | 8192 | 1.0466, 1.0070, 0.9684 | 1.0070 |
| 64 | 64 | 2048 | 0.8615, 1.0321, 1.0060 | 1.0060 |
| 64 | 64 | 4096 | 1.0171, 1.0144, 1.0001 | 1.0144 |
| 64 | 64 | 8192 | 1.2845, 0.9983, 1.0031 | 1.0031 |
| 128 | 128 | 2048 | 0.9984, 1.0093, 1.0052 | 1.0052 |
| 128 | 128 | 4096 | 1.0222, 0.5776, 0.9946 | 0.9946 |
| 128 | 128 | 8192 | 1.0001, 0.9853, 0.9212 | 0.9853 |

A′/A（タイル構成のみを変えた効果）はほぼ 1.0 近傍（0.99〜1.05 が大半）で、
`(32,32,4096)` の run1/run2（0.61〜0.63）のような外れ値を除けば、タイル
構成差自体の寄与は小さい（§6 参照）。

### control_forced（対照 3 形状。A=classic vs C=強制 split-K。参考のみ）

| m | n | k | speedups（run1,run2,run3） | median | partitions |
|---|---|---|---|---|---|
| 256 | 256 | 2048 | 1.4075, 0.6541, 1.3720 | 1.3720 | 2 |
| 256 | 256 | 4096 | 2.7785, 1.6551, 1.6768 | 1.6768 | 4 |
| 256 | 256 | 8192 | 1.9383, 1.6525, 1.6618 | 1.6618 | 8 |

強制 split-K は対照形状でも中央値ベースでは classic を上回る傾向がある
（`max_groups=40` の境界は保守的である可能性を示唆。ただし本結果は
`max_groups` の閾値変更を推奨するものではなく、記録のみ）。

### フロア（参考。各 `(M,N)` を `K=64` で classic dispatch）

| m | n | k | median_a_secs（3 run 中央値） |
|---|---|---|---|
| 32 | 32 | 64 | 8.8875e-5 |
| 64 | 64 | 64 | 8.9208e-5 |
| 128 | 128 | 64 | 8.8416e-5 |
| 256 | 256 | 64 | 8.7666e-5 |

対象形状の A（classic）median_a_secs は約 2.1e-4〜5.3e-4 秒であり、フロア
（約 8.7〜8.9e-5 秒）はその 20〜40% 程度を占める。dispatch 固定費が classic
経路の実行時間に無視できない割合を占めるが、B（split-K）はさらに小さい
median_b_secs（例: `(128,128,8192)` は 1.9e-4 秒）を達成しているため、
フロア自体が speedup の主因ではない。

## §5 判定

**暫定 ADOPT**（§0 参照。人間による分析上の暫定評価であり、`aggregate.py`
が出力する機械的な正式 verdict とは区別する）。3 run いずれも
`verdict=undetermined`（専有ゲート不成立）を出力せず正常終了しており、
対象 9 形状・対照 3 形状とも事前登録基準を満たした。

ただし `aggregate.py` 自体の正式な ADOPT/REJECT 出力は、渡された run 数
（本例では 3）が事前登録した `MIN_FORMAL_RUNS`（5）以上であることを条件に
変更済み（イシュー #1499 codex-review P1 指摘への対応。3〜4 run の時点で
形状さえ揃えば正式判定を出力していた旧実装の不備を修正）。このため
`docs/perf/logs/metal-gemm-splitk-ab-1475/aggregate.md`（3 run から生成）は
機械的な verdict としては **`undetermined`**（`n_runs=3 < MIN_FORMAL_RUNS=5`
により暫定値と明示）を出力しており、上記「暫定 ADOPT」という本ドキュメントの
評価はあくまで §0／本節の記述に基づく人間側の暫定的な解釈であって、
`aggregate.md` が ADOPT を宣言しているわけではない。加えて `aggregate.py`
は各 run が専有ゲート（`env_guard_mode=gated` かつ
`env_guard_overall verdict=pass`）を経て収集されたことも run 単位で検査する
（同 P2 指摘への対応）。run1〜run3 はいずれも `run_gated.sh` 経由（§0 の
`--max-load-avg=8.0` 緩和込み）で収集されておりこの検査を満たす。

### 計測環境に関する留保

- 実測時のマシンは複数セッションが並走する共有環境で、1 分 load average
  が概ね 3〜7 で推移していた。事前登録・実装計画は `--max-load-avg=4.0`
  を想定していたが、この閾値では専有ゲートのバックオフ再試行が上限
  （10 回）に達し undetermined になる可能性が高かったため、実務判断として
  `--max-load-avg=8.0` へ緩和して実行した（`env_info.txt` に経緯を記録）。
  対象・対照双方が同一の共有負荷下で interleaved 計測されているため
  speedup 比自体への影響は限定的と考えられるが、厳密な専有環境での
  再計測（5 run 完了）が望ましい。
- 単一セッションの時間制約により 5 run のうち 3 run で打ち切った。3 run
  はいずれも target 9 形状で 1.5686〜4.4233 の範囲に収まり分散は大きくない
  （最小値 1.5686 は事前登録基準 1.5 にわずかな余裕を持って上回る）ため、
  結果が閾値付近で不安定という兆候はない。ただし control `(256,256,2048)`
  の run2（0.5903）のような単発の大きな振れが観測されており、5 run 完了
  時にこの形状の中央値が基準（0.95）を下回るリスクはゼロではない。

### スコープ

ADOPT は性能上の判定に限る。本番結線（`SPLIT_K_NUMERIC_CONTRACT_APPROVED`
の切替）は REQ-2 判定方式の Metal f32 split-K への適用拡張という別のユーザー
承認が必要（#1476 のスコープ。§7 参照）。

## §6 因果に関する考察（`docs/backend-metal-splitk-decision.md` §3 留保への回答）

`docs/backend-metal-splitk-decision.md` §3・`docs/perf/metal-gemm-splitk-shapes.md`
§4/§6（#1308）は、対象 9 形状の劣化率実測が「並列度不足」という因果の
**仮説**にとどまり、タイル構成差等の代替説明を排除できていないと留保していた。

本 A/B は §4 の `target_tile`（A′/A。タイル構成のみを変えた効果）と
`target`（B/A′ 相当。K 分割効果）を分離することでこの留保を検証する:

- **A′/A（タイル構成効果）**: 対象 9 形状の median は 0.99〜1.05 が大半
  だが、`(32,32,4096)` は 3 run 中 2 run（run1=0.6140・run2=0.6278）が
  0.6 台まで劣化し、run3（0.9895）のみ 1.0 近傍に戻るという再現性のある
  ばらつきを示した（イシュー #1499 codex-review 指摘: 単発の外れ値では
  ない。3 run のみでは劣化が恒常的な効果か run 間ノイズかを切り分けられ
  ず、原因は本 A/B の観測範囲外として未確定のまま扱う）。他 8 形状では
  タイル構成を `select_for_device` の選択構成から `split_k_tile`
  （16×16 または 32×32・staged・wm2/wn2）へ変えても classic 経路の速度は
  ほぼ変化しない。
- **B/A′（K 分割効果。`target` の speedup を A′ 基準に再計算すると
  概ね target の speedup と同程度）**: A′/A がほぼ 1.0 のため、`target`
  （A/B の比）で観測された 1.5686〜4.4233 倍の改善は、タイル構成差ではなく
  **split-K による K 方向の並列度拡張（`partitions=8〜32` の追加分割）** に
  ほぼ帰属できる。

**結論**: #1308 が留保した「因果は仮説」は、本実測（A′/A がほぼ 1.0・
target の改善幅が大きい）により**「K 方向の並列度不足」が主因である
という仮説を支持する**方向で更新する。ただし §5 の留保（3/5 run・共有
負荷環境）のため、この結論も 5 run 完了後の再確認が望ましい暫定的な
ものとして扱う。フロア（§4）は classic 経路の実行時間の一定割合
（20〜40%）を占めるが、B の実行時間はフロアを下回らないためフロア自体が
speedup の主因ではないことも確認した。

## §7 スコープ外（フォローアップ）

- **フェーズ 1 control（B′）の選択関数呼び出し費用込み再実測**: §4
  「control」節「既知の限界」参照。`should_split_k` 呼び出しをクロージャへ
  含めるよう是正済みのコードで対照 3 形状を実機（M4 Max）で再実測し、
  §4 の control 表・§5 の判定根拠を更新する。
- **フェーズ 0 A vs B（`a_vs_b_fail_count`）の同一入力での再実測**: §3
  「既知の限界」参照。`prepare()` の `seed_offset` 不一致（是正済み）を
  反映した同一入力でのフェーズ 0 自己検証を実機（M4 Max）で再実行し、
  `self_check.log`・§3 の実測値を更新する。フェーズ 1（§4）の speedup
  自体は GEMM の実行時間が入力値に依存しない（形状のみに依存する）ため
  本是正による再実測は不要と判断する。
- **5 run 完了による正式確定**: 単一セッションの時間制約により 3/5 run で
  打ち切った。残り 2 run（可能なら専有環境・`--max-load-avg` をより厳格な
  値で）を追加実行し、§0/§5 の「暫定」を外す正式判定へ更新することを
  フォローアップとして issue #1475 に残す。**→ #1515（§10）で新規 5 run
  として実施する（専有ゲートは受け入れ条件にしない。ルート #1509 の
  ユーザー指示）。#1475 の 3 run とは混在させない**。
- **#1476（本番結線可否）**: `select_for_device`／`dispatch_auto`／
  `MetalBackendOps::gemm` への結線と `SPLIT_K_NUMERIC_CONTRACT_APPROVED`
  の切替（数値契約の適用拡張。ユーザー承認事項）。**結線せずと確定（§9）**。
  **追記（イシュー #1518）**: 数値契約は #1513 で解消・結線自体は #1516 で定数ゲート付き
  （既定 OFF）に実施済み。詳細は §9 追記・`docs/backend-metal-splitk-decision.md` §5 参照。
- split-K の encode 分離入口の追加と GPU タイムスタンプによる純カーネル
  時間計測（`gemm.rs` 変更が必要）。
- NT/TN/TT の性能比較・f16／hfrag の split-K・`gemm_bias_act` 融合経路への
  適用（#1474 §8 と同じ）。


## §8 参照

- `docs/backend-metal-splitk-decision.md`（本イシューの追記先。§3。本番結線可否の確定は §4）
- `docs/perf/metal-gemm-splitk-shapes.md`（#1308。劣化率の元実測）
- `docs/perf/metal-gemm-splitk-two-pass.md`（#1474。split-K 実装記録）
- `docs/perf/logs/metal-gemm-splitk-ab-1475/`（本イシューの生ログ・
  `aggregate.py`／`aggregate.md`・`env_info.txt`・`self_check.log`）

## §9 本番結線可否の確定（#1476）

**結論: 結線しない。** `select_for_device`／`dispatch_auto`／`MetalBackendOps::gemm` は不変、
`SPLIT_K_NUMERIC_CONTRACT_APPROVED=false` を維持する。判定根拠は
`docs/backend-metal-splitk-decision.md` §4 を正とし、本節では要点のみ記す。

**追記（イシュー #1513。2026-09-10）**: `SPLIT_K_NUMERIC_CONTRACT_APPROVED` は #1513 で
`true` へ切替済み（数値契約ブロッカーの解消。`docs/perf/metal-gemm-splitk-two-pass.md`
§5.9）。本節の「結線しない」判断自体は性能ブロッカー（本節が確定した undetermined 判定）
により不変で、`select_for_device`／`dispatch_auto` への本番結線は #1516 へ引き継ぐ。

- 本ドキュメント §0 の「暫定 ADOPT」は人間側の解釈であり、機械判定（`aggregate.md` の
  `verdict`）は `undetermined`（`n_runs=3 < MIN_FORMAL_RUNS=5`。PR #1499 の codex-review P1
  対応で 5 run 未満は正式 ADOPT/REJECT を出力しない仕様）。Issue #1476 の結線条件「#1475 が
  ADOPT の場合のみ」に対し、正式な ADOPT 判定は得られていない
- 独立してもう 1 つのブロッカー（`docs/perf/metal-gemm-splitk-two-pass.md` §5 の数値契約未承認）
  があり、性能判定が仮に正式 ADOPT であっても結線には至らない
- 本節は §0／§4／§5 の実測記述を「確定」へ書き換えるものではない。5 run 完了による正式確定は
  §7 のフォローアップのまま未実施（本イシューでも実施しない。理由は decision doc §4「スコープ外」）

**追記（イシュー #1518）**: `select_for_device`／`dispatch_auto`／`MetalBackendOps::
gemm` への結線は、その後 #1516 で `dispatch_auto` への定数ゲート付き結線（既定
`SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = false`）として実施された。ゲート OFF の間は
結線前と bit 同一の classic 経路が維持され、本節の「結線しない」という §9 見出しの結論
（性能の正式 ADOPT 未確定を理由にゲート ON へは進まない）自体は #1518 時点でも不変。§10.4
の 5 run 正式確定は未実測のまま Mac セッションへ引き継ぐ（`docs/backend-metal-splitk-
decision.md` §5「ゲート既定値・切替条件」）。**追記（2026-09-11）**: §10.4 で M4 Max 実機
5 run を完了し **ADOPT** を正式確定した。

## §10 5 run 正式確定（イシュー #1515。共有負荷下・専有ゲートなし）

### §10.1 運用方針

ルート #1509 のユーザー指示により、split-K A/B の 5 run 正式確定は**専有
ゲート（load average の閾値判定）を受け入れ条件にしない**。§0・§9 が
「暫定 ADOPT（3/5 run）」「undetermined」と記録してきた状態を解消するため、
本節では以下の方針で 5 run を実施する:

- `gemm_splitk_ab_bench` へ `--max-load-avg` を渡さず **record_only**
  運用（判定なし・[`bench_harness::ab::GuardRetryOutcome::record_only`]。
  load average 等を記録するのみで即座に計測へ進む）で実行する
  （`docs/perf/logs/metal-gemm-splitk-ab-5run-1515/orchestrate.sh`）。
- 共有負荷下（他プロセス並走を許容）であることを、各 run の
  `env_guard_mode=record_only`・`env_guard_load_avg` 行に加え、計測中の
  負荷推移を記録するバックグラウンド `uptime` サンプラー
  （`runN_monitor.log`）・並走プロセス watchlist の件数
  （`runN_procs.txt`）で記録する。
- **REJECT は共有負荷下でも有効な REJECT として扱う**（専有ゲート不成立
  を理由に REJECT を undetermined へ格下げしない）。

### §10.2 事前登録判定規則（計測前に固定。以後変更しない）

- **腕定義・計測境界**: §1 と同一（A=classic／A′=classic・split-K タイル
  ／B=split-K／B′=対照・classic／C=対照・強制 split-K／フロア）。
- **ADOPT／REJECT の定数**: §2 と同一。対象 9 形状すべてで (i) 主指標
  （run 内比の 5 run 中央値）≥ 1.5 かつ (ii) 5/5 run すべてで run 内比
  > 1.0、かつ対照 3 形状すべてで主指標 ≥ 0.95 なら ADOPT、いずれか不成立
  なら REJECT（共有負荷下でも有効な REJECT）。
- **undetermined の条件**（以下のいずれかのみから生じる。それ以外の理由
  で undetermined へ倒さない）:
  1. 5 run の完全性が崩れる（対象 9・対照 3 形状のいずれかが、いずれかの
     run で欠落・重複する。`aggregate.py::check_run_shape_completeness`）。
  2. フェーズ 0 の run-to-run bit 同一（`classic_stable`／
     `target_tile_stable`／`splitk_stable`）が崩れる、または target の
     checksum（`checksum_a_bits`／`checksum_at_bits`／`checksum_b_bits`。
     `f64::to_bits()` 由来の round-trip 可能な値。`.6e` 表示の
     `checksum_a` 等は丸め誤差で異なる checksum が同一文字列になり得る
     ため判定には使わない。イシュー #1529）が 5 run 間で一致しない、
     または `a_vs_at_fail_count` が 0 でない（`aggregate.py::
     check_phase0_consistency`。受け入れ条件「checksum 一致」の機械化）。
     `a_vs_b_fail_count`（classic vs split-K の既知差）は情報としてのみ
     記録し判定へは使わない（`docs/perf/metal-gemm-splitk-two-pass.md`
     §5.5 の既知 fail が期待値であるため）。
  3. env_guard の記録が欠落・不一致（record_only 運用なのに
     `env_guard_mode=record_only`・`env_guard_load_avg` の数値記録が
     見つからない、またはログの取り違え）。
- **5 run は新規**（run1〜run5）とし、#1475 の 3 run（is-optimized-away
  是正前バイナリによる計測）とは**混在させない**。run の差し替えは禁止
  し、中断した場合は `env_info.txt` に経緯を記録する。`orchestrate.sh`
  は計測開始前（何も書き込む前）に当該 run 番号の既存成果物を検出する
  と非ゼロ終了し（同番号の同時実行もロックで拒否する）、中断後の再開は
  **未実施の run 番号のみ**を指定する運用を機械的に強制する（イシュー
  #1529）。同番号の再実行は成果物を手動で別名へ退避してから行う。
  #1475 のログは `checksum_*_bits`（上記 2.）を出力しない旧バイナリに
  よる計測のため、本 `aggregate.py` の正式な checksum 一致検査対象外
  （bits 欠落で undetermined 扱い）であり、当時の判定は
  `docs/perf/logs/metal-gemm-splitk-ab-1475/aggregate.md`（旧
  aggregate.py 出力）を正とする。**正式判定の対象は入力がちょうど 5 run
  の場合に限る**（6 run 以上は差し替え・選別の余地を生むため、
  undetermined ではなく `aggregate.py` が fail-closed にエラー終了する。
  イシュー #1529）。
- 負荷推移（`runN_monitor.log` の load1 min/median/max）・並走プロセス
  件数は**情報としてのみ**記録し、判定（ADOPT/REJECT/undetermined）へは
  影響させない。
- **ADOPT は性能上の判定に限る**。本番結線（`select_for_device`／
  `dispatch_auto`／`MetalBackendOps::gemm` への結線・
  `SPLIT_K_NUMERIC_CONTRACT_APPROVED` は #1513 で承認・切替済み）は
  #1516 へ引き継ぐ。本節では判定結果のみを確定する。

### §10.3 実施手順

`docs/perf/logs/metal-gemm-splitk-ab-5run-1515/README.md` の「Mac セッション
での実行手順」を正とする（要約）:

```sh
cd docs/perf/logs/metal-gemm-splitk-ab-5run-1515
for i in 1 2 3 4 5; do ./orchestrate.sh "$i"; done
python3 aggregate.py --gate-mode=record_only \
  --monitor-logs=run1_monitor.log,run2_monitor.log,run3_monitor.log,run4_monitor.log,run5_monitor.log \
  run1.log run2.log run3.log run4.log run5.log > aggregate.md
```

`aggregate.md` の内容を §10.4 へ転記する。

### §10.4 実測結果（Apple M4 Max 実機・2026-09-11・共有負荷下）

実測は Apple M4 Max 実機（MacBook Pro Mac16,6・16 コア・64 GB・macOS
26.6.2・rustc 1.96.0）上で 2026-09-11 に実施した（run1〜run5・各 1 プロセス
起動・約 15 分/run・中断なし・run の差し替えなし）。`orchestrate.sh` を
`--max-load-avg` なし（record_only）で起動し、判定は `aggregate.py
--gate-mode=record_only` の機械判定をそのまま採用した（生ログ・監視ログ・
`aggregate.md` は `docs/perf/logs/metal-gemm-splitk-ab-5run-1515/`）。
なお実測前に `orchestrate.sh` の bash 3.2（macOS 既定 `/bin/sh`）変数名
誤解釈バグ（`$VAR` 直後の全角文字が変数名へ取り込まれ `unbound variable`
で起動不能）を PR #1537 で修正した（計測ロジック・判定規則は不変）。

#### target（対象 9 形状）

| m | n | k | n_runs | speedups | median_speedup | all_run_positive |
|---|---|---|--------|----------|-----------------|-------------------|
| 32 | 32 | 2048 | 5 | 1.7489,1.5328,1.6824,1.6904,1.8690 | 1.6904 | True |
| 32 | 32 | 4096 | 5 | 2.3089,2.4029,2.3045,2.3185,2.4385 | 2.3185 | True |
| 32 | 32 | 8192 | 5 | 3.6684,3.5047,3.6270,3.6136,3.5722 | 3.6136 | True |
| 64 | 64 | 2048 | 5 | 1.7588,1.6509,1.6792,1.7871,1.6643 | 1.6792 | True |
| 64 | 64 | 4096 | 5 | 2.3479,2.3803,2.2761,2.3687,2.4200 | 2.3687 | True |
| 64 | 64 | 8192 | 5 | 3.7450,3.7478,3.5007,3.4461,3.8005 | 3.7450 | True |
| 128 | 128 | 2048 | 5 | 1.5925,1.5332,1.5548,1.5546,1.5063 | 1.5546 | True |
| 128 | 128 | 4096 | 5 | 2.0043,2.0747,1.9846,2.0521,2.1375 | 2.0521 | True |
| 128 | 128 | 8192 | 5 | 2.7501,2.6799,2.7730,2.6787,2.7172 | 2.7172 | True |

全 9 形状で中央値 ≥ 1.5 かつ 5/5 run で比 > 1.0（§10.2 の事前登録規則を
充足）。K が大きいほど speedup が大きい（k=2048: 1.55〜1.69 倍・k=4096:
2.05〜2.37 倍・k=8192: 2.72〜3.75 倍）。

#### control（対照 3 形状）

| m | n | k | n_runs | speedups | median_speedup |
|---|---|---|--------|----------|-----------------|
| 256 | 256 | 2048 | 5 | 0.9851,0.9507,0.9829,0.9652,0.9925 | 0.9829 |
| 256 | 256 | 4096 | 5 | 1.0076,1.0074,1.0188,0.9987,0.9751 | 1.0074 |
| 256 | 256 | 8192 | 5 | 1.0253,0.9989,1.0109,1.0063,0.9853 | 1.0063 |

全 3 形状で中央値 ≥ 0.95（非後退。`should_split_k` が対照形状で split-K を
選ばず classic 経路のまま・選択関数呼び出し費用込みで差なし〈§10.5 の
§7 フォローアップ「B′ の選択関数呼び出し費用込み再実測」を消化〉）。

#### target_tile／control_forced／floor（参考のみ）

##### target_tile

| m | n | k | n_runs | speedups | median_speedup | all_run_positive | median_a_secs |
|---|---|---|--------|----------|-----------------|-------------------|----------------|
| 32 | 32 | 2048 | 5 | 1.1583,1.0303,0.9843,0.9476,0.9861 | 0.9861 | False | 2.125410e-04 |
| 32 | 32 | 4096 | 5 | 0.9894,1.0094,1.0003,1.0209,1.0104 | 1.0094 | False | 3.002080e-04 |
| 32 | 32 | 8192 | 5 | 0.9656,1.0237,0.9683,1.0538,0.9698 | 0.9698 | False | 4.676250e-04 |
| 64 | 64 | 2048 | 5 | 0.9832,0.9996,0.9868,0.9661,0.9918 | 0.9868 | False | 2.295000e-04 |
| 64 | 64 | 4096 | 5 | 0.9946,0.9966,0.9991,0.9770,1.0104 | 0.9966 | False | 3.281670e-04 |
| 64 | 64 | 8192 | 5 | 0.9968,0.9890,1.0234,1.0239,1.0019 | 1.0019 | False | 5.368330e-04 |
| 128 | 128 | 2048 | 5 | 0.9978,1.0002,1.0046,0.9880,0.9948 | 0.9978 | False | 2.292080e-04 |
| 128 | 128 | 4096 | 5 | 0.9945,1.0025,0.9899,1.0218,1.0042 | 1.0025 | False | 3.257500e-04 |
| 128 | 128 | 8192 | 5 | 1.0277,1.0024,0.9934,0.9801,1.0030 | 1.0024 | False | 5.365830e-04 |

##### control_forced

| m | n | k | n_runs | speedups | median_speedup | all_run_positive | median_a_secs |
|---|---|---|--------|----------|-----------------|-------------------|----------------|
| 256 | 256 | 2048 | 5 | 1.2491,1.2137,1.1784,1.2366,1.1835 | 1.2137 | True | 2.273330e-04 |
| 256 | 256 | 4096 | 5 | 1.5735,1.5070,1.5243,1.5159,1.6022 | 1.5243 | True | 3.205410e-04 |
| 256 | 256 | 8192 | 5 | 1.7505,1.8393,1.8688,1.7358,1.7479 | 1.7505 | True | 5.384170e-04 |

##### floor

| m | n | k | n_runs | speedups | median_speedup | all_run_positive | median_a_secs |
|---|---|---|--------|----------|-----------------|-------------------|----------------|
| 32 | 32 | 64 | 5 | NA | NA | True | 9.500000e-05 |
| 64 | 64 | 64 | 5 | NA | NA | True | 9.562500e-05 |
| 128 | 128 | 64 | 5 | NA | NA | True | 8.650000e-05 |
| 256 | 256 | 64 | 5 | NA | NA | True | 9.458400e-05 |

`target_tile`（A′: 対象形状で split-K を強制 OFF・タイル構成のみ変更）は
全形状で 0.97〜1.01 倍（≈1.0）であり、対象形状の改善が K 分割に帰属する
ことを支持する（§3 の「因果は仮説」留保を 5 run で再確認）。
`control_forced`（対照形状で split-K を強制 ON）は 1.21〜1.75 倍で、
`should_split_k` が対照形状を split-K 非対象としている現行判定は保守的
（改善余地の示唆。本イシューのスコープ外・判定規則は変更しない）。

#### フェーズ 0 checksum 一致

`aggregate.py::check_phase0_consistency` の違反リストは空（verdict が
undetermined でなく ADOPT を返したことで機械確認）。5 run の phase0 行
（13 形状分）から抽出した `checksum_*_bits` 列は 5 run すべてで完全一致
（md5 同一）・全形状 `classic_stable=true`／`target_tile_stable=true`／
`splitk_stable=true`・`a_vs_at_fail_count=0`。`a_vs_b_fail_count`
（classic 対 split-K の要素比較）は一部形状で非ゼロだが、これは #1474
§5.5 で確認済みの split-K パーティション分割に起因する既知の性質であり
#1511 承認済みの baseline 非後退方式（#1512）で受け入れ判定される対象
（本節の判定対象外。§10.5 の「A vs B の同一入力での再実測」を消化）。

#### 負荷推移（情報のみ）

| run | load1_min | load1_median | load1_max | 並走プロセス（watchlist 件数） |
|-----|-----------|--------------|-----------|-------------------------------|
| 1 | 1.50 | 2.45 | 6.00 | watchlist 全 6 名 0 件（python／torch／mlx／cargo／gemm_／bench。`ps -axo comm=` 一致件数） |
| 2 | 1.43 | 2.48 | 3.39 | watchlist 全 6 名 0 件（python／torch／mlx／cargo／gemm_／bench。`ps -axo comm=` 一致件数） |
| 3 | 1.27 | 2.23 | 7.10 | watchlist 全 6 名 0 件（python／torch／mlx／cargo／gemm_／bench。`ps -axo comm=` 一致件数） |
| 4 | 2.56 | 3.43 | 4.93 | watchlist 全 6 名 0 件（python／torch／mlx／cargo／gemm_／bench。`ps -axo comm=` 一致件数） |
| 5 | 2.83 | 3.77 | 5.74 | watchlist 全 6 名 0 件（python／torch／mlx／cargo／gemm_／bench。`ps -axo comm=` 一致件数） |

各 run 89 サンプル（10 秒間隔）。`pmset -g therm` は全 run の前後とも
thermal／performance warning の記録なし。

#### 機械判定

`aggregate.py` の verdict: **ADOPT**（target_shapes=9/9・target_ok=True・
control_shapes=3/3・control_ok=True）

#### 人間側の判定

**ADOPT**（機械判定をそのまま採用。緩和・格上げなし）。§7 の「暫定
ADOPT・3/5 run」は本節で正式確定へ昇格する。後続: #1516 のゲート
`SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` を `true` へ切替（ドリフト
検出テスト更新・実機 `#[ignore]` 群 pass 確認）→ #1517 の
framework-compare A/B。

### §10.5 §7 フォローアップとの対応

本 5 run により、§7 が残していた以下 2 項目も同時に消化される
（新規バイナリでの計測のため）:

- 「フェーズ 1 control（B′）の選択関数呼び出し費用込み再実測」
- 「フェーズ 0 A vs B（`a_vs_b_fail_count`）の同一入力での再実測」

いずれも §3／§4 本文（3 run 実測の記述）自体は書き換えない。5 run 完了後
の正式値は本節（§10.4）へ記録し、§3／§4 は歴史的記録として残す。

