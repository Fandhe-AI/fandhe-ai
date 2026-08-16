# CUDA GEMM mma.sync cp.async issue interleaving 計測記録（#496・B-5）

イシュー #496「perf(backend-cuda): cp.async 発行を warp 内 mma ループへ分散（issue interleaving）」の実測記録テンプレート。
GEMM 性能改善ツリー #479 → Phase 2 親 #490 の B-5。先行 B-4（#495・ldmatrix 先読みダブルバッファ）に続く発行レイテンシ隠蔽の最適化。

## 状態: 未実測・実機実行待ち（#502 で再計測）

本実装セッションの実行環境は CUDA **driver**（`libcuda`。compute capability 8.6・RTX 3060 実機）は存在するが NVRTC（`libnvrtc`）が存在しないため（`crates/backend-cuda/src/kernels_mma.rs` 冒頭コメント「検証状態」参照）、本ファイルが記録する変更（cp.async 発行の kstep ループ内分散）は **NVRTC による構文検証を一度も通過していない**。sm_86（この実機）・sm_121（DGX Spark GB10。設計上のターゲット）のいずれでも未検証。`docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md`（B-4 時点の同種記録）・`docs/perf/metal-gemm-dynamic-tile.md` の先例（実機での最初の実行が構文検証を兼ねる）と同じ位置づけ。

本実装セッションで検証済みの事項:

- `cargo build --workspace`（`const _: () = assert!(...)` によるコンパイル時境界検査。変更なし・§1 参照）
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda`（`kernels_mma.rs` 内 `#[cfg(test)]` の `#define` 整合検査・REQ-8 needle・issue interleaving 構造ロック〈新規〉・既存のダブルバッファ／段数可変／タイル定数 pin テスト全件・`gemm_mma.rs` の launch config div_ceil 被覆テスト）
- `tests/parity_nonregression.rs` の通常 CI 実行分（tolerance 定数 pin・fixture 自己整合。無変更で green）
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/` が無差分（§4 の bit 一致論拠の機械確認）

未検証・実機実行待ちの事項（#502「Phase B 完了時点の再計測」へ引き継ぐ）:

- NVRTC によるカーネルソースの構文検証そのもの
- B-4（#495）比の TFLOPS 改善判定（下記 §3 記録欄。B-4 自体も未実測のため #502 へ引き継ぐ）
- レジスタ予算実測（発行位置の変更のみでレジスタ配列次元・フラグメント数は不変のため #495 からの追加増分はない想定だが、実機確認は #502）
- sm_121（DGX Spark GB10）実機の SMEM/レジスタ属性（`docs/perf/sm121-device-attributes.md` は未実測のまま）

## 1. 背景

B-4（#495）は warp 内 kstep ループへ ldmatrix 先読みダブルバッファを導入し、SMEM→レジスタのロードレイテンシを Tensor Core 演算とオーバーラップさせた。一方、次段 K タイルの `cp.async`（グローバル→SMEM）発行（`LOAD_A_STAGE`/`LOAD_B_STAGE`）は K タイルループ末尾（kstep ループの外側、commit 直前）で一括発行されたままであり、発行（アドレス計算＋LSU 発行）コストが K タイル境界の 1 点に集中していた。

本イシュー（B-5）は CUTLASS `mma_multistage.h` の issue interleaving（`kAccessesPerGroupA/B = ceil(AsyncCopyIterationsPerStage / kWarpGemmIterations)` 個ずつを `warp_mma_k` ごとに分割発行する方式）と同型の技法を導入し、1 K タイル分の cp.async 発行を warp 内 kstep ループ（`BK / MMA_K` 反復）へ分散して発行レイテンシを隠蔽する。

## 2. 実装内容

- チャンク添字空間の連続分割: `K_GROUPS = BK / MMA_K`（現構成 2）個のグループへ、A は `A_CHUNKS = (BM*BK)/8 = 256` チャンク・B は `B_CHUNKS = (BK*BN)/8 = 512` チャンクをそれぞれ ceil 分割する（`A_GROUP_CHUNKS = ceil(A_CHUNKS / K_GROUPS) = 128`・`B_GROUP_CHUNKS = ceil(B_CHUNKS / K_GROUPS) = 256`）。本カーネルは per-thread 反復数が小さく（A: 0.5・B: 1）CUTLASS の per-thread 反復分割は縮退するため、チャンク添字空間の分割へ翻案した
- グループ `g` の発行レンジ `[g * GROUP_CHUNKS, min((g+1) * GROUP_CHUNKS, CHUNKS))` のみを発行する `LOAD_A_STAGE_GROUP(stage, k0, g)`/`LOAD_B_STAGE_GROUP(stage, k0, g)` マクロを新設。ceil 分割のため全チャンクが必ずいずれかのグループに含まれ、`BM`/`BN`/`BK` の将来変更にも取り零しなく追随する
- 既存の `LOAD_A_STAGE`/`LOAD_B_STAGE`（プロローグでの 1 ステージ分まとめてロード用）は、全グループについて `LOAD_*_STAGE_GROUP` を呼ぶ薄いラッパーへ再定義（発行本体は 1 箇所に集約されたまま。「ループ化・非コピペ」方針を維持）
- kstep ループ内、`g = kstep`（`K_GROUPS == BK / MMA_K` が kstep ループの反復回数と一致するため、全グループが過不足なく発行される）としてグループ発行を、ldmatrix 先読み（kstep+1 段）の後・mma.sync 発行の前に挿入
- K タイルループ末尾の一括発行（`if (next_tile < num_k_tiles) { LOAD_A_STAGE(...); LOAD_B_STAGE(...); }`）は撤去し、`cp.async.commit_group`（無条件・ループ末尾）のみ残す

境界クランプ（`gr_c`/`gc_c`・16 バイト整列切り下げ）・`valid`（範囲外チャンクの src-size 0 ゼロ充填）の REQ-8 境界検査ロジックは一切変更していない。

### 発行位置の選定理由

CUTLASS が `warp_mma()` 呼び出しと `copy_tiles_and_advance()` を交互に置く配置と同旨で、cp.async の発行が直後の Tensor Core 演算列とオーバーラップするよう、ldmatrix 先読みの後・mma.sync 発行の前に置いた。

### 同期の不変条件（追加の `__syncthreads()` が不要な理由）

発行先 SMEM ステージ（`load_stage = (t+STAGES-1) % STAGES`）は、前イテレーション末尾の `__syncthreads()` 通過後は誰にも読まれない（本イテレーションの ldmatrix は `compute_stage` を読み、`load_stage != compute_stage` は `STAGES >= 2` で常に成立する）。よって発行を kstep ループ内へ前倒ししても、書き込み（cp.async 発行）と読み出し（ldmatrix）のハザードは生じず、追加の同期は不要である。

### commit の位置を動かさなかった理由（#492 不変条件の維持）

`cp.async.commit_group` はループ末尾・無条件のまま据え置いた。「1 イテレーション = 必ず 1 commit」不変条件（#492）とループ内固定即値 `wait_group (STAGES-2)` の正しさ論証（`kernels_mma.rs::MMA_STAGES` 定数直下コメント参照）は、commit の位置を変えない限りそのまま成立する。分割後も全グループの発行が同一イテレーション内で完了してから 1 回だけ commit されることに変わりはない。

## 3. 段階的計測手順（実機・CUDA driver + NVRTC 搭載・compute capability 8.0 以上）

```sh
git fetch origin
git checkout perf/496-cp-async-issue-interleaving   # 本イシューの実装ブランチ
cargo test -p backend-cuda -- --ignored --nocapture   # parity 非後退の全行検査（数値一致確認を性能計測より先に実施）
cargo run -p backend-cuda --example gemm_mma_bench --release   # TFLOPS 計測
```

### 記録欄（実機セッションで埋める）

| 対象 | M=N=K=2048 TFLOPS（5 回中央値） | M=N=K=4096 TFLOPS（5 回中央値） | B-4（#495）比 |
|------|-------------------------------|-------------------------------|---------------|
| B-5（issue interleaving 適用後） | 未計測 | 未計測 | 未計測 |

判定基準（#496 実装計画・受け入れ基準）: 4096 TFLOPS が B-4 完了時点を下回らないこと。B-4 自身も未実測（#502 で再計測予定）であるため、B-4/B-5 の非後退比較は #502 の実機セッションへ引き継ぐ。

## 4. 数値 bit 一致の論拠（parity 非後退契約）

cp.async の**発行タイミング**の変更はコピーされるデータ・`mma.sync` の発行順序・オペランド値を一切変えない。各出力要素のアキュムレート順序は引き続き「K タイル t 順 → kstep 順」の `mma.sync` 系列のみで決まるため、出力は B-4（#495）時点と bit 一致であり:

- tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）・ベースライン fixture（TF32: 42493/262144・mean_abs_diff 1.574e-3 等）は**変更しない**
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/` が無差分であることを本実装セッションで機械確認済み（§「状態」参照）
- FMA 契約（f16 入出力・f32 アキュムレート、`mma.sync` の固定契約）は不変

## 5. リスク・安全側判断の記録

- **NVRTC 未検証**: 変更は既存 `cp.async` 発行の呼び出し箇所（ループ内の位置）とチャンク添字レンジの分割のみで、新規 PTX 命令は追加していない。B-1〜B-4 で検証済みの命令列（`cp.async.cg.shared.global`・`cp.async.commit_group`・`cp.async.wait_group`）を維持するため構文リスクは増加しない
- **性能後退リスク**: 分割によりグループあたりの発行スレッド数・チャンク数が減る（A: 128 チャンク/グループ）が、発行総量は不変で時間軸へ分散されるのみ。実測判定は #502 に委ね、後退が観測された場合の切り戻しは本 PR の差分が `kernels_mma.rs` 1 ファイルに閉じるため容易
- **数値後退リスク**: bit 一致論拠（§4）+ parity 関連ファイル無差分の機械確認により正しさリスクはない。実機セッションでは ignored parity テストを性能計測より先に実行する運用を維持する
- **クロスタイル ldmatrix 先読み（#495 の残余最適化）は本イシューのスコープ外**とし混入させていない（`docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md` の引き継ぎ記述を維持）

## 6. §4 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/
```

無差分であることを確認する（§4 の bit 一致論拠の裏付け。tolerance 定数・ベースライン fixture を変更していないことをコミット前に検査する。本実装セッションで実施済み）。
