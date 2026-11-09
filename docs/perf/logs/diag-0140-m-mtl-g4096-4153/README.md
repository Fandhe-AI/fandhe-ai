# <metal-node> の M-MTL-G4096 で candle とカーネル時間を比べる診断（#4153）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE 雛形・集計・変換・補助集計・1 round の起動キット・xctrace の schema 検出）だけを置く。
`RULE.txt` と `env/xctrace_*`（PR-2・単独コミット）、`round{i}/`・`aggregate.md`・`aggregate.tsv`・`attrib.tsv`・結果欄（PR-3・<metal-node> での実測）は後続で追加する。
実測値・帰属・分類・確度は、実測するまで書かない（推定値で埋めない）。#4153 は実測と記録の PR-3 で閉じる（部品の PR は `Refs #4153`）。
PR-1 の担当環境（Linux）は Mac mini に到達できないため、`--self-test`・構文検査までしか行っていない。**未検証**: xctrace の有無、System Trace の表 schema・列名、GEMM 区間の抽出（本数一致・余分な ≥ 5 ms 区間の有無・開始時刻と継続時間の列が同じ単位か）、
`--target-stdout` の挙動、trace の摂動量、診断テスト出力と変換の整合、ツリー指紋の一致。

## 目的

スコアボード 0.14.0 の M-MTL-G4096 は fandhe-ai reuse 64.379 ms 対 candle 56.103 ms（比 0.88。`unreached_0140.tsv`）。fandhe 側の純カーネル時間（`kernel_gpu`）は実測済み
（`docs/perf/metal-gemm-reuse-phase-breakdown.md` §12.3）だが、candle 側は**未測定**で、手元にあるのは壁時計の `transfer_excl` だけ（同 §12.6・§12.10、`docs/perf/loss-attribution-matrix.md` §10.3.1）。
同じ時計で両者のカーネル時間と窓を測り、窓の差（fandhe − candle）が**カーネル側**か**窓側**（upload・readout・同期など）かを分ける。あわせて雛形の F1／F2 と分類を出す。

## 設計の要点（RULE 雛形が正）

- **candle は Rust 側で GPU タイムスタンプを取れない**。`candle-core` 0.11.0 の `MetalDevice` は command buffer を外へ出さず、`candle-metal-kernels` は `objc2_metal` を再エクスポートしない。
  `GPUStartTime`／`GPUEndTime` を呼ぶには `objc2-metal` を `bench-candle` の直接依存に足す必要があり、これは deps-policy 第 9 区分と `check_framework_compare` の allowlist の変更で**所有者承認が要る**。
  本 issue は依存・`bench-candle` を変えない。承認が得られた場合の代替（`objc2-metal` を直接依存にした GPUStart/EndTime 計測）は本スコープ外の選択肢。
- **採用する方式**: 外部計器 `xcrun xctrace record --template 'Metal System Trace'` で両プロセスの GPU 実行区間を取る（時計タグ `gpu_trace` が両側で一致し、雛形 F1 の条件を構成上満たす）。
  candle は既存の `gemm-transfer-split`（device 常駐入力で matmul＋synchronize）、fandhe は registry 0.14.0 の `bench-fandhe --task gemm --mode reuse`。
- **GEMM 区間の抽出**: 対象プロセスの GPU 区間のうち継続時間 ≥ 5 ms を GEMM とみなす（N=4096 の GEMM は M4 10 コアの公称上限でも 20 ms 超、ほかは 1 ms 未満）。本数が warmup＋測定反復数と完全一致しなければ INVALID。candle の `gemm-transfer-split` は転送込み(warmup+iters)→転送除外(warmup+iters)→checksum 検証 1 回の順に同一プロセスで GEMM を走らせる（JSON の task は `gemm_transfer_incl`／`gemm_transfer_excl`）ため、本数は転送込みの warmup+iters＋転送除外の warmup+iters＋1 を要求し（転送込みの本数は転送込みレコード自身の値を使う）、転送除外の測定反復だけを切り出す。warmup 分を捨てた中央値を `*_pure_kernel` にする。
  余分な ≥ 5 ms の区間の許容枠は設けない（継続時間だけでは先頭・途中・末尾のどこにあるか特定できず、許容すると測定スライスが黙ってずれる）。余分な区間が出た round は INVALID で、扱いは `RULE.v2.txt` で定め直す。同一プロセスの区間が重なる表も INVALID。件数の一致は並びの正しさまでは保証しない（GEMM の欠落と余分な区間が同数なら検出できない）。
- **照合**: 同 round 内で診断テスト `gemm_reuse_phase_diag_production_batch`（`kernel_gpu`＝GPUEnd−GPUStart）も取り、xctrace の fandhe 値との相対差が 10% 以内かを見る（超えたら確度低）。診断テストは `tree_sha` の path ソース、窓と xctrace は registry 0.14.0 のため、RULE に `git diff v0.14.0 <tree_sha> -- crates/backend-metal/src` の有無を記し、差分ありなら「版差あり」を付ける。`gemm_window_route_precheck` で `ROUTE_ACTUAL=Classic` と bit 一致を確認し、不一致なら INVALID。
- **摂動検査**: 同じ round で trace あり／なしの窓を両方取り、`trace_perturbation` が 0.10 を超えたら確度低（#4150 の `event_perturbation` と同型）。
- **窓の差の帰属**: ΔW＝fandhe_window − opp_window、ΔK＝fandhe_pure_kernel − opp_pure_kernel、`kernel_share`＝ΔK÷ΔW。≥ 0.5 でカーネル側が主、未満は窓側が主、ΔW ≤ 0 は判定不能。全 round がゲート PASS で同じ側に入り、摂動・照合が上限以内なら確度高、それ以外は低。
- **フォールバック**: probe で xctrace が使えない、または GPU 区間が 0 行なら、RULE は AUX `xctrace: unavailable`・cell 行 `formulas=F2` だけで固定する（この組は `to_values_4153.py`・`attrib_4153.py` が検査し、食い違えば止まる。`aggregate_diag.py` は AUX を読まない）。F1 は計測せず、分類は `aggregate_diag.py` が F2 だけで決める: F2 の中央値が閾値以上なら「計測窓の定義」、未満なら「分類保留」（理由は固定文言「どれも届かない」だが、F1 を測っていないので「自分たちの制約／ハードの上限」は判定していない。`attrib.tsv` の `fallback_note` に明記）。確度は雛形の規則どおりで、F1 を宣言しないので時計の条件は課さない。帰属（`kernel_share`）は判定不能、摂動・照合は unavailable。壁時計境界の F1(b)（candle `transfer_excl` 対 fandhe `encode＋commit_wait`）は参考として本欄に併記するだけで `values.tsv` には混ぜない。
- `kernel_owner=constraint`（根拠は REQ-8 の境界検査保持。candle は function constant `align_M/N/K` で境界検査なしのロード〈`docs/analysis/candle-metal-01.md` 204〜210 行・245 行〉）。所有者は PR-2 の RULE コミット時点までなら変更できる。

## 読み方の注意

- TIMEOUT／INVALID の round を含む系列は `status=partial`・表題「参考」で、確度を出さない。
- `scoreboard_ratio=0.88` は `unreached_0140.tsv` の転記。窓の中央値同士の比は 64.379 対 56.103 で 0.871 と、丸めで差がある。ドリフト検査は雛形どおり 10%。
- 窓の腕は registry 0.14.0、xctrace の腕も同じバイナリ。タイル（Mac mini の fandhe は `bm32 bn32 bk16 wm2 wn2 staged`）は記録だけにする（タイルの A/B は #4160）。
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち。#4142）。承認後に定義が変われば `RULE.v2.txt` で反映する。
- 同じ Mac mini を #4146 ほかの実測も使う。hw-lock で直列化し、計測中にビルドを並走させない。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する規則の雛形（`aggregate_diag.py --check-template` で検査）。`## AUX` が補助 PARAMS（`aggregate_diag.py` は読まない） |
| `aggregate_diag.py` | 雛形（`diag-0140-template/`）の無改変コピー。sha256 を RULE に固定 |
| `to_values_4153.py` | `round{i}/raw/` から `values.tsv`・`aux.tsv` を作る厳格変換（`--self-test`。stub の xcrun・bench で `run_round.sh`・`xctrace_probe_4153.sh` の成功・失敗経路のマスクも確かめる）。xctrace の XML は DOCTYPE／ENTITY・巨大入力を拒否。`--mask-dir` は保存物の識別子のマスク（下の「マスク」） |
| `attrib_4153.py` | `kernel_share`・摂動・照合の補助集計（`--dir`・`--check-rule`・`--self-test`）。sha256 を RULE に固定 |
| `run_round.sh` | 1 round の起動（precheck → ゲート → 窓 → xctrace → 診断テスト。ゲート後に cargo を呼ばない）。終了時に必ず round ディレクトリをマスクする |
| `xctrace_probe_4153.sh` | PR-2 の前に 1 回だけ。N=256 のスモークで `xctrace export --toc` を保存し schema 名・列名を確定する（計測ではない） |

## 手順

1. （PR-2・<metal-node>）PR-1 マージ後の main SHA を git archive で展開してビルドし、`ROOT=… OUT=<記録先>/env bash xctrace_probe_4153.sh` を実行する。`xctrace_toc.xml` から GPU 実行区間の表の schema 名・列名を確定し、AUX に書く。unavailable なら AUX の `xctrace: unavailable` と `formulas=F2` で固定する（trace 専用キー〈`xctrace_schema`〜`gemm_interval_min_ms`〉は `<…>` のまま・削除のどちらでもよい）。
2. `tree_sha`・`tree_fingerprint`・各 sha256（`aggregate_diag.py`・`to_values_4153.py`・`attrib_4153.py`・`run_round.sh`）を埋め、`git diff v0.14.0 <tree_sha> -- crates/backend-metal/src` の有無を AUX に記す。
   山括弧を外すだけの欄（issue・rounds・reach_threshold・noise_band・cell 行の prev_class）も置換する。`python3 -I aggregate_diag.py --check-template RULE.template.txt`、`python3 -I attrib_4153.py --check-rule RULE.txt` を通し、`RULE.txt` を**単独コミット**する。
3. （PR-3・<metal-node>）hw-lock を取り、ツリー指紋を RULE と照合して、`ROOT=… EXPECT_SHA=… DIAG_BIN=… XCTRACE_MODE=<RULE の AUX xctrace> XCTRACE_SCHEMA=…（unavailable なら省略可・trace の腕は取らない） HW_LOCK_OWNER_FILE=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
4. 各 round で `python3 -I to_values_4153.py --round-dir <記録先>/round<i> --rule <記録先>/RULE.txt`。
5. `python3 -I aggregate_diag.py --dir <記録先>` と `python3 -I attrib_4153.py --dir <記録先>` で集計する。
6. 結果欄と `docs/perf/loss-attribution-matrix.md` の M-MTL-G4096 行（§10.3.1 の「candle の純カーネル時間は未測定」を含む）を実測値＋タグで更新する（未測定は未測定のまま）。`Closes #4153`。保存物のマスクは下の「マスク」のとおり自動で行われるが、対象外の識別子が残っていないかは回収時に目で確かめる。

## マスク（RULE 項 7）

- `run_round.sh`・`xctrace_probe_4153.sh` は記録先を作った後、成功・ゲート不通過・腕の失敗・xctrace unavailable・シグナルのどの経路で終わっても EXIT trap で trace バンドル（`*.trace`。未マスクのバイナリ）を消し、`python3 -I to_values_4153.py --mask-dir` で記録先（`round{i}/`・`env/`）の全通常ファイルを置換する。マスクが使えない環境は記録前の `--mask-ready` で止まり、置換後に識別子が残れば非 0 で終わる（値そのものは出力しない）。
- 置換するのはホームパス（`HOME` と passwd のホーム）→ `<home>`、ホスト名（gethostname とその短縮形、`scutil` の ComputerName／LocalHostName／HostName。xctrace のログは ComputerName を出す）→ `<metal-node>`、パスの 1 要素として現れたユーザー名 → `<user>`。`.xml` はテキストと属性値を `&lt;…&gt;` で置換して XML を壊さない（CDATA 内は素の `<…>`）。
- **検出できないもの**: パス以外に現れたユーザー名（`bench-fandhe` のような正当なデータを壊さないため置換しない）、3 文字未満の名前、他人のユーザー名・ホストの別名、識別子を加工した文字列。
- スクリプトが端末へ出すエラー文（絶対パスを含みうる）と、`to_values_4153.py`・`attrib_4153.py`・`aggregate_diag.py` のエラー文はファイルへ保存しない。これらが書く `values.tsv`・`aux.tsv`・`attrib.tsv`・`aggregate.*` はパスを含まない。

## 結果

（実測後に追記。PR-1 時点は空。）
