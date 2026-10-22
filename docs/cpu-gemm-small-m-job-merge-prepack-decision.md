# m が小さい GEMM の job 統合・A の pack 共有・重みの事前 pack の設計決定記録（イシュー #3198）

## 1. 位置づけ

- 対象: MLP（784→256→10・batch 64）の CPU 学習・推論に現れる m=64 級 GEMM（6 形状）。`TwoDDynamic`（(mc, nc) 2D 動的分配。`TWO_D_DYNAMIC_PRODUCTION_ENABLED=true`・`TWO_D_JOBS_PER_WORKER=2`）が細かい job に分かれ、固定費（A の再 pack・job ごとの確保・C のコピーイン／アウト）が支配的になっている。
- 本書は設計の決定記録で、実装は #3199（job 統合と A の共有）・#3200（重みの事前 pack）、A/B は #3201、既定の反転判断は #3227。本 PR（#3198）で入れるコードは、既定 OFF の job 統合ゲート 1 個と、OFF 固定のテストだけである。
- 運用方針: 推奨形を本書に書き、後続がそのまま実装へ進む。新規 `unsafe`・依存・公開 API の追加は範囲外とする。

## 2. 入力

| 出典 | 使う内容 |
|---|---|
| `docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md` §2・§4（#3177） | 6 形状の job 数・A／B 再 pack 語数・確保回数、jpw1／cap4／cap8 の A/B（jpw=1 は GB10 で 0.82〜0.94、Mac mini はほぼ中立。Mac mini の L2 dW のみ 1.141） |
| `docs/perf/cpu-thread-policy-macmini.md`（#3178） | Mac mini のスレッド方針 A/B と負荷ゲート（load1 < 3.0） |
| `crates/backend-cpu/src/small_shape_thread_cap.rs`（#1575） | 仕事量上限 `SMALL_SHAPE_CAP_MAX_WORK = 1 << 24`（学習の最大仕事量 12,845,056 を含む最小の 2 のべき乗） |
| `docs/cpu-gemm-2d-dynamic-partition-design.md` §3・§7 | bit 一致の条件 1〜8、job 間の B 共有を REJECT（#750／#1041／#1144） |
| `docs/analysis/candle-cpu-03.md` §4.1 | 事前 pack・pack 共有の参照設計 |
| `crates/backend-cpu/src/gb10_affinity.rs`（#2117） | `GB10_AFFINITY_ENABLED` は単独 REJECT・既定 OFF |

## 3. ゲート一覧

| ゲート | 置き場 | 既定 | 追加する issue | 読み手 |
|---|---|---|---|---|
| `SMALL_M_JOB_MERGE_ENABLED` | `crates/backend-cpu/src/gemm_blis/small_m.rs` | false | 本 PR（#3198） | `small_m::jobs_per_worker_for`（`gemm_blis_parallel_with_transpose`・`gemm_blis_bias_act_parallel` の `dispatch_two_d_dynamic` 呼び出し） |
| `RESIDENT_W_PREPACK_ENABLED` | `crates/backend-cpu/src/gemm_blis/prepack.rs`（推奨） | false | #3200（読み手と同時） | `gemm_resident_rhs_impl` から呼ぶ新入口 |

- 本リポジトリは `#[allow(dead_code)]` で黙らせない。読み手のない const は `clippy -D warnings` で落ちるため、G2 の const は #3200 で読み手と同時に足す。G2 は分割方式ではなくキャッシュなので、「分割方式ごとのゲート」の例外として扱う（名前・置き場・条件は本書で確定）。
- G1 が ON のときの挙動: 対象形状の `jobs_per_worker` を 1 に下げる。OFF のときは `TWO_D_JOBS_PER_WORKER`（2）のままで、現行と同一。#3199 は同じゲートの下で A の共有と確保削減を足す。
- 既存ゲートとの関係: `TWO_D_DYNAMIC_PRODUCTION_ENABLED=false` なら G1 は効かない（行パネル経路には分岐を入れない）。`SME_PRODUCTION_ENABLED` の対象（`sme_shape_eligible`: m≥256・n≥256・k≥64）とは、A 共有の対象だけが重複せず、job 統合（G1）は重複する（例: L1 dW 784×256×64 は G1 の対象かつ `sme_shape_eligible` も満たし、`dispatch_two_d_dynamic` は SME カーネルにも G1 が選んだ jpw=1 を渡す）。G1 と SME を同時に有効にした組み合わせを、後続の実装・採否判断の検証対象に含める。`GB10_AFFINITY_ENABLED`・`SMALL_SHAPE_CAP_ENABLED` は外側のプール選択で、G1 の内側にそのまま重なる。

## 4. 対象形状の判定

- **job 統合（G1）**: `m >= 2` かつ `m * max(n, NR_CLAMP) * k < SMALL_M_MAX_WORK (= 1 << 24)`。`saturating_mul` を使い、飽和したら対象外（安全側）。6 形状はすべて対象、正方 512（1.34 億）は対象外。dW の 2 形状（L1 dW 784×256×64・L2 dW 256×10×64）は m が大きいが、仕事量では対象に入る。
- **A 共有（#3199）**: 上に加えて `m <= 128 (= MC)` かつ `align_up(m, MR) * k <= 1 << 18` 語（1 MiB）。L1 fwd・L1 dX・L2 fwd・L2 dX が対象で、dW 2 形状は job 統合だけ。const は使う側と同時に #3199 で追加する。
- dW の再 pack が重いのは B 側（#3177 §2）だが、B の job 間共有は 2D 設計 §7 で REJECT 済みのため採らない。将来の候補としてのみ記録する。
- 注意: Mac mini の L2 dW は jpw1 で 1.141 だった。悪化の可能性があり、採否は #3201 の A/B で決める。

## 5. job 統合の粒度と A の共有単位（#3199 が実装する形）

- 採用: (行帯 × 列帯) の格子を保ち jpw=1 で job を統合する。さらに行帯ごとに A を 1 回だけ事前 pack して共有する（2 段）。
  - 段 1: 共有バッファの配置は (行帯, pc, ic, ir_block)。各区画は既存の `pack_a`／`pack_a_from_transposed` が書く内容と同じで、互いに素な区画へ `par_chunks_mut` で並列に書く（`unsafe` なし）。
  - 段 2: 各 job は共有の `&[f32]` から A スライスを読む。B は従来どおり job ごとに private に pack する。
- 不採用: 全幅の行帯だけの job（col_bands=1）＋ B 共有。m=64・MR=8 では job が最大 8 個で、GB10（20 スレッド）では #3177 の cap8 が L1 fwd で 1.6 倍遅かった。B 共有は §7 に反する。
- 確保削減: `try_for_each_init` で c_local と B panel のスクラッチを worker ごとに 1 回確保して使い回す。A は共有するので job ごとの `a_panel` は不要になる。
- col_bands == 1 なら行帯の C は連続なので `chunks_mut(mc_job * n)` を ldc=n でそのまま渡す（staging なし）。
- col_bands > 1 ではコピーイン／アウトが残る（L1 fwd で 1 GEMM あたり 32768 語）。行セグメントへ直接書くには `run_with_ldc` の API 変更か raw pointer（`unsafe`）が要るため、受容する残差とする。
- `gemm_blis_ic_loop` は内部で A を pack するため、#3199 は「pack 済み A を受け取る ic ループ」を切り出す。カーネル呼び出しの列は変えない。

## 6. 事前 pack の持ち方（#3200 が実装する形）

- **持ち主**: `CpuBufferHandle`（`crates/backend-cpu/src/memory.rs`。`pub(crate)`）にキャッシュ欄（`Mutex<Vec<PrepackEntry>>`。値は `Arc<[f32]>`）を足す。buffer の破棄と一緒に解放されるので、ストアの破棄・プール返却・ABA が構造的に起きない。
- **無効化**: `data` 欄を private にし、読み取りは `data()`、書き込みは `data_mut()` だけにする。`data_mut()` はキャッシュを必ず空にする。`downcast_handle_mut::<CpuBufferHandle>` を使う箇所（現状 19 か所）は `data_mut()` へ寄せる。欄が private なので、更新経路の取りこぼしはコンパイル時に検出される。tensor-core・公開 API への epoch 受け渡しは不要。
- **キー**: (view の offset, k, n, 転置種別 Nn, カーネルの NR, blocks.kc, blocks.nc)。1 本の連結バッファを複数パラメータが共有しても衝突しない。
- **配置**: 全幅を (pc ブロック, 大域 nr ブロック) の順に並べ、各区画は `pack_b` の出力と同じにする。job は `cols.start`（nr の倍数。`partition.rs` の align_up）から区画を切り出す。使える条件は job 内の jc 境界が nr 格子と一致すること（`nc_job <= blocks.nc` または `blocks.nc % nr == 0`）で、満たさなければ現行の job 内 pack へフォールバックする。
- **上限**: `k * align_up(n, nr) <= RESIDENT_W_PREPACK_MAX_WORDS (= 1 << 20)` 語（4 MiB）。判定は `checked_mul`／`saturating_mul`。L1 の W は約 20.7 万語、L2 の W は約 3 千語。増加メモリは W とほぼ同じ大きさで、#3200 が実測を記録する。
- **入口**: `gemm_resident_rhs_impl`（`ops.rs`）から `pub(crate)` の新入口（例 `gemm_blis_bias_act_parallel_prepacked`）を呼ぶ。既存の `pub fn gemm_blis_bias_act_parallel` のシグネチャは変えない。適用範囲は forward の B のみ（backward の `gemm_resident_lhs` は範囲外）。
- **同時実行**: 引き当ては Mutex の下、構築はロックの外。同時に構築しても内容は同じなので後勝ちでよい。
- CPU バッファが `PooledBufferHandle` に包まれる経路があると #3200 の実装中に分かった場合に限り、実装を止めて所有者へ戻す。

## 7. bit 一致（FMA 連鎖・加算順序）の根拠

2D 設計 §3 の条件に照らして次を満たす。

- (a) pack は値の配置換えにすぎず、誰がいつ pack しても `pack_a`／`pack_b` が書くバイト列は同じ。
- (b) `run_with_ldc` は同じ `kc_len`・同じ ap／bp スライス内容で呼ばれる。
- (c) 全 job が pc 昇順で、pc ごとに C を load → p 昇順の FMA 連鎖 → store する。
- (d) K を分割しない（split-K なし）。1 要素は 1 job に属する。
- (e) `mc_job`／`nc_job`／jpw は性能パラメータで正しさの条件ではない（§3 条件 6）。

FMA 契約（`f32::mul_add`／`vfmaq_laneq_f32`）・tolerance・公開 API は変えない。本 PR は `small_m.rs` のテストで、6 形状と端数形状（63×257×785・65×10×9）、T=1／10／20、NaN／±inf 混入入力について、jpw=1 と本番入口の出力が `to_bits` で完全一致することを固定した。#3199・#3200 は、自分の経路について同等の bit 一致テスト（6 形状・m／k の端数・NaN／inf）を同一 PR に含める義務を負う。

## 8. 採らない案

- B の job 間共有・dW の B 側共有（§7 の REJECT。将来の候補）。
- col_bands>1 の staging 解消（API 変更または `unsafe` が必要。範囲外）。
- epoch を tensor-core へ渡す無効化方式（`data_mut()` への一本化で足りる）。

## 9. 後続への引き渡し

| issue | 内容 |
|---|---|
| #3199 | §5 の A 共有と確保削減。ゲート G1 の下に実装。pack が行帯あたり 1 回であることを確認する `#[ignore]` の report テストを用意する（実装済み。§11 参照） |
| #3200 | §6 の事前 pack と G2 の const 追加。引き当て成功を確認する report テストを用意する |
| #3201 | `docs/perf/logs/cpu-small-m-gemm-partition-3198/RULE.template.txt` を使った両機体の A/B |
| #3227 | 既定の反転判断。機体で結果が割れた場合は機体別既定の可否を併記する |

## 10. 所有者へ戻す論点

現時点でなし。§6 末尾の条件に該当した場合のみ停止して戻す。ruleset・リポジトリ設定の変更は行っていない。

## 11. #3199 実装記録

- 適格判定は `small_m::select_shared_a`（純関数）。`SMALL_M_SHARE_A_MAX_WORDS = 1 << 18`（判定は `<=`）を追加し、条件は §3 のとおり（`m <= MC`・`align_up(m, MR) * k <= 上限`・仕事量 `< 1 << 24`・`m >= 2`）。ゲートは G1 のまま。本番 2 入口が `SMALL_M_JOB_MERGE_ENABLED` を `share_a` として `dispatch_two_d_dynamic` へ渡す。テスト入口 `gemm_blis_parallel_two_d_dynamic_with_params` は常に `share_a = false`（#3198 の jpw=1・staging 腕のまま）。強制入口は `gemm_blis_parallel_two_d_dynamic_shared_a_with_params`。
- 共有 A は `gemm_blis_two_d_shared_a`（`mod.rs`）が呼び出しスレッドで 1 回確保する（`row_bands * band_stride`。`checked_mul`）。区画 (rb, pc, ir_block) の先頭は `rb * band_stride + mr_blocks * mr * pc + ir_block * kc_len * mr`。pack 段は行帯単位で並列に走り、job 段は共有読み出しのみ（新しい `unsafe` なし）。
- `gemm_blis_jr_ir_loop` の `#[cfg(test)]` を外して本番へ昇格した。適格 job では ic ループが 1 周のため ic=0 の `gemm_blis_ic_loop` と同一のタイル書き戻しになる。
- `col_bands == 1`: C を行帯で直接分割し、staging なし。`col_bands > 1`: `split_c_into_jobs` と C staging は残し（§5 の受容済み残差）、`c_local`・B パネルを rayon の init ごとに 1 回だけ確保して使い回す。残る確保は共有 A 1 回・スクラッチ（直接書き込み 1／staging 2）・`split_c_into_jobs` 側（col_bands > 1 のみ）。
- 計数は `jobgrid_probe`（`#[cfg(test)]`）に追加。受け入れは `shared_a_packs_once_per_row_band`（`a_shared_bands == row_bands`）と、small_m.rs の bit 一致テスト群。`determinism_inventory` の atomic RMW 期待値は、test 限定プローブの計数フック追加に伴い 12→15（jobgrid_probe 10→13）へ更新した。
- 報告用 `#[ignore]` テスト `small_m_shared_a_report`（本番入口・6 形状 × T）を #3201 向けに用意した。既定は OFF のまま（反転は #3227）。

## 12. #3200 実装記録

- 置き場とゲート: `crates/backend-cpu/src/gemm_blis/prepack.rs`。ゲート `RESIDENT_W_PREPACK_ENABLED`（G2）は既定 OFF。OFF のとき `ops.rs` は `PrepackSource` を渡さず、現行の job 内 pack と同一の経路になる。既定の反転は #3201（A/B）・#3227（判断）で、本 PR は行わない。実測値（時間）も本 PR には含めない。
- 入口は 2 つを結線した。§6 は `gemm_resident_rhs_impl` を例示するが、infer の CPU セルは `predict_device_chain` → `linear_forward_device`（`ops.rs`）を通るため、こちらも W を Nn の B として使う「forward の B」として結線した（`linear_forward_device_with`。trait 実装から切り出した）。範囲外（従来どおり pack）は `m == 1`・行パネル経路・backward の `gemm_resident_lhs`・Nt／Tn。
- 持ち主と無効化: `CpuBufferHandle`（`memory.rs`）がキャッシュ `PrepackCache` を持つ。`data` は private で、書き込み口 `data_mut()` が必ずキャッシュを破棄する（`Mutex::get_mut`。ロック不要）。`zero_fill`（pool 再利用）・`upload_into`・各 optimizer も同じ口を通る。ハンドルと一緒に drop されるため、ストア破棄・別 Tape・ABA は構造的に起きない。optimizer の要素ループは `data_mut()` をループ外へ出して束縛する。
- キー: `(offset, k, n, nr, kc)`。転置は Nn 固定。配置は `pc * align_up(n, nr) + jb * kc_len * nr`（各区画は `pack_b` 単独の出力とバイト一致。テストで固定）。job 側は `((cols.start + jc) / nr)` から `nr_blocks * kc_len * nr` 語を連続で切り出す。
- 使える条件（`select_prepack`）: Nn・`k * align_up(n, nr) <= 1 << 20`（`checked_mul`。溢れは不可）・`nc_job <= NC || NC % nr == 0`。満たさないときは job 内 pack へフォールバックする。NEON（NR=12）は `512 % 12 != 0` のため実質 `nc_job <= NC` だけが条件で、n > 512 で col_bands == 1 になる形状はフォールバックする。L1（n=256）・L2（n=10）は全 ISA で適格。AVX2（16）・AVX-512（32）・Scalar（4）は `512 % nr == 0` で常に通る。
- エントリ数の上限 `RESIDENT_W_PREPACK_MAX_ENTRIES = 16`（安全側の追加）。超えたら挿入せず、構築した panel をその呼び出しだけで使う。
- 構築は逐次で、並列ライブラリを使わない（`determinism_inventory` の rayon allowlist を動かさない）。W の更新ごとに 1 回（train）または初回のみ（infer）で、job ごとの B pack を 1 回の全幅 pack に置き換える。構築はロックの外で行い、同時構築は後勝ち（内容は同一）。
- 決定性の棚卸し: キャッシュの `Mutex` は引き当て専用で数値アキュムレータではない。`determinism_inventory` の lock 期待値に `gemm_blis/prepack.rs` 7 行を根拠付きで追加した。atomic は足していないため RMW 15 件は不変。
- 追加メモリ（式 `k * align_up(n, NR) * 4` バイト。`memory_footprint_matches_formula` で語数を固定）。`MemoryStats` には計上しない（キャッシュはデバイス確保ではなくハンドル付随の派生データで、計上すると既存の報告値・peak-memory の契約が変わるため。将来の候補）:

  | W | W 本体 | NEON NR=12 | AVX2 NR=16 | AVX-512 NR=32 | Scalar NR=4 |
  |---|---|---|---|---|---|
  | L1 784×256 | 200,704 語（802,816 B） | 206,976 語（827,904 B） | 200,704 語（802,816 B） | 200,704 語（802,816 B） | 200,704 語（802,816 B） |
  | L2 256×10 | 2,560 語（10,240 B） | 3,072 語（12,288 B） | 4,096 語（16,384 B） | 8,192 語（32,768 B） | 3,072 語（12,288 B） |

  ゲート OFF のときの増分は 0（ハンドルあたり Mutex と空 Vec の数十バイトのみ）。
- テスト: gemm レベルは forward 形状（64×256×784・64×10×256）・端数（63×257×785・65×10×9・2×13×7）・KC またぎ（64×256×1000）・フォールバック形状（64×1024×256）を T=1／10／20・jpw 2／1・A 共有（G1）併用・NaN／inf 混入で `to_bits` 一致。ops 層は ON／OFF 一致、3 step の更新後に古い panel を使わないこと、書き込み経路ごとの無効化、同一 `params_buf` 上の別 offset・別形状・複数ストアのキー分離。報告用 `#[ignore]` テスト `resident_w_prepack_report`（語数表示）を #3201 向けに用意した。

