# thread_elements() 方式 BlockMMA 候補 A/B（イシュー #1694）

`docs/perf/metal-gemm-thread-elements-candidate.md` §7 が引き継ぐ
「候補カーネル（イシュー #1693。`gemm_simdgroup_tiled_te`）を本番選択
構成（`tile::select_for_device`）と純カーネル時間（GPU タイムスタンプ。
`kernel_gpu`）で A/B する」計測のスキャフォールド。実測は**Apple M4 Max
実機を持つ Mac セッションで実施**する（本エージェント実行環境は Linux
のため未実施のまま `verdict=undetermined` で出荷する）。

## 位置づけ

- 前提ゲート（R0〜R3）は `docs/perf/logs/metal-gemm-thread-elements-1693/`
  へログを残す（`orchestrate.sh gate` が実行する）。R0（probe）または
  R1（parity・正しさ）が FAIL の場合は正しさ不成立として **REJECT** を
  確定し、性能 A/B（本ディレクトリ）は実施しない。R2／R3 の FAIL は
  「機構契約の不成立」として記録し、性能 A/B は実施しても参考値扱い
  とする（`verdict` は undetermined のまま）。
- 性能 A/B は `crates/backend-metal/src/gemm_te_diag_tests.rs::
  te_kernel_gpu_ab_vs_production_select` を 5 プロセス起動する
  （record_only 運用。専有ゲートなし。ルート #1519 のユーザー指示・
  #1515／#1538 と同じ運用）。

## 事前登録判定規則

イシュー #1694 の issue コメント（および
`docs/perf/metal-gemm-thread-elements-candidate.md` §7.1）を正とする
（本 README は要約のみ。計測後に緩和しない）:

1. 前提ゲート順序固定（R0→R1→R2→R3）。R0/R1 FAIL は REJECT 確定・性能
   A/B 非実施。
2. 正しさ（REQ-2）: 各 run・各 N で trial 0 の head 出力が CPU 参照
   （`matmul_reference_fma`）と複合判定 pass（診断テスト内で
   fail-closed に検証済み）。
3. checksum 完全一致（`bit_identical=true`）。1 セルでも不一致ならその
   N は比の値によらず REJECT。
4. 性能指標: `head_over_base_kernel_gpu` の N ごと 5 run 中央値。
5. N ごとの判定: 5 run 中央値 `<=1.00` かつ 5/5 run 符号一貫 →
   `ADOPT-as-opt-in-candidate`。中央値 `>1.00` かつ 5/5 run 符号一貫 →
   `REJECT`。符号反転 → `undetermined`。
6. 総合判定: 全 N が ADOPT のときのみ候補前進を推奨。1 N でも REJECT な
   ら無条件前進は推奨しない。undetermined を含めば総合も undetermined。
   いずれの場合も **本番結線は行わない**（別イシューでユーザー承認が
   必要）。
7. 負荷: record_only（判定には使わない。記録のみ）。
8. フォールバック発生 run は診断テスト側の assert で abort する
   （`resolved_cfg != cfg`）。
9. ちょうど 5 run が正式判定の対象（`aggregate.py` が完全性検査で
   fail-closed に検出する）。

## Mac セッションでの実行手順

1. リポジトリ最新化・`docs/real-hardware-verification-env.md` の実機
   接続手順に従う。
2. 前提ゲート（R0〜R3）を実行する:

   ```sh
   cd docs/perf/logs/metal-gemm-thread-elements-ab-1694
   ./orchestrate.sh gate
   ```

   R0/R1 が FAIL したら打ち切られる（性能 A/B は実施しない）。ログは
   `../metal-gemm-thread-elements-1693/`（`probe_run.log`／
   `parity_run.log`／`all_staged_candidates_run.log`／
   `bit_match_run.log`）へ保存される。**R1（parity）の N=4096 相当の
   CPU 参照計算（`matmul_reference_fma`。逐次 3 重ループ）は数十秒〜
   分単位かかりうる**（`gemm_te_parity.rs` 自体は N=2048 までのため
   直接は該当しないが、後続の性能 A/B〈本 README〉が N=4096 で毎 N
   ループごとに CPU 参照計算を 1 回行うため同程度の待ち時間が生じる）。

3. 性能 A/B を 5 回、別プロセス起動で実行する（1 run = 1 起動。
   差し替え禁止）:

   ```sh
   for i in 1 2 3 4 5; do
     ./orchestrate.sh "$i"
   done
   ```

   各 run は `kernel_gpu_te_ab_run<N>.log`（本体出力）・
   `run<N>_monitor.log`（10 秒間隔の負荷サンプラー）・
   `run<N>_procs.txt`（watchlist プロセス件数）・
   `uptime_before_run<N>.txt`・
   `pmset_therm_{before,after}_run<N>.txt` を生成する。中断した場合は
   `env_info.txt` に経緯を記録し、run を黙って差し替えない。
   `orchestrate.sh` は計測開始前に当該 run 番号の既存成果物を検出すると
   何も書かずに非ゼロ終了する（同番号の同時実行もロックで拒否する）。

4. 集計する（`--gate-dir` は手順 2 で `orchestrate.sh gate` が残した
   R0〜R3 ログのディレクトリを指す。省略・不在の場合は前提ゲート未確認
   として総合判定が必ず undetermined になる）:

   ```sh
   python3 aggregate.py --gate-dir ../metal-gemm-thread-elements-1693 \
     kernel_gpu_te_ab_run1.log kernel_gpu_te_ab_run2.log \
     kernel_gpu_te_ab_run3.log kernel_gpu_te_ab_run4.log \
     kernel_gpu_te_ab_run5.log > aggregate.md
   ```

5. `aggregate.md` の内容を `docs/perf/metal-gemm-thread-elements-
   candidate.md` §7.3 の記入欄へ転記する。判定は `aggregate.py` の機械
   判定をそのまま採用し、人間側で緩めない。
6. `env_info.txt` の記入欄（機種・OS・rustc・base sha・作業ブランチ・
   run ごとの load1 範囲・中断有無）を埋める。内部ホスト名は書かない。
7. イシュー #1694 へ結果をコメントする。いずれの判定（ADOPT-as-opt-in-
   candidate／REJECT／undetermined）でも、本番結線（`tile::select`／
   `dispatch_auto` 既定化）は本イシューのスコープ外のため行わない
   （候補前進の判断は別イシューでユーザー承認のうえ起票する）。

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `orchestrate.sh` | `gate`（前提ゲート）／`<run番号>`（性能 A/B 1 プロセス起動）の実行ラッパー（`--dry-run` あり） |
| `aggregate.py` | 集計・判定スクリプト（`--self-test` あり） |
| `env_info.txt` | 機種・OS・実行構成の記入欄（実測未実施のため未記入） |
| `kernel_gpu_te_ab_run{1..5}.log` | 各 run の `te_kernel_gpu_ab_vs_production_select` 出力（未生成） |
| `run{1..5}_monitor.log` | 各 run の負荷推移サンプラーログ（未生成） |
| `run{1..5}_procs.txt` | 各 run 開始時の並走プロセス watchlist 件数（未生成） |
| `aggregate.md` | 集計結果（未生成） |

## 対象外（本イシューのスコープ外）

- 本番結線（`tile::select`／`dispatch_auto` 既定化）・実行時トグル API
  の設計（ユーザー承認必須の別イシュー）。
- framework-compare（facade 経由）A/B: 候補は facade から到達不能のため
  対象外。
- split-K・TileClass 分割・direct-load 等との併用計測。
- MLX 型 interleaved 配置・`simdgroup_barrier(mem_none)`・fused
  epilogue（#1693 §6 を踏襲）。
- REQ-2 baseline 行の追加・tolerance 変更。

## 注意

- 内部ホスト名・ユーザー名・絶対パスを成果物（ログ・env_info・README）
  に書かない。
- 本番結線（`crates/backend-metal/src/` の変更）は本イシューのスコープ外
  （別イシューへ引き継ぐ）。`orchestrate.sh`・`aggregate.py` はいずれも
  `crates/backend-metal/src/` を変更しない。
