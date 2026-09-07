# CPU `Var::matmul` 経路の固定費削減設計

## §0 要約・位置づけ

- 親: #1285（ルート #1283「CPU GEMM の candle〈gemm crate〉超えトラッキング」
  Phase 1「計測境界固定費の解消」）
- 前提: #1292（framework-compare reuse 境界のフェーズ分解実測。両実機・
  5 回独立プロセス中央値）。実測記録は
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §15、生ログは
  `docs/perf/logs/cpu-gemm-reuse-phase-1292/`
- 後続: #1299（本設計に基づく実装）・#1301（両実機 before/after 実測・
  結線可否判断）
- 本イシュー（#1294）は **docs-only**。本番コード変更・結線・実機実測は
  行わない

**結論（要約）**: `facade` → `autodiff` → `backend-cpu` の呼び出しチェーンを
`file:line` レベルで追跡した結果、

1. **contiguous コピーは本番 NN（転置なし）経路で発生していない**
   （葉テンソルは元々 contiguous のため `contiguous()` は `Arc` clone のみ）
2. **tape 登録（`push_eager`）のオーバーヘッドはノイズ水準**
   （§15.3 の `kernel` 区間比 1〜2 桁下）
3. **本番経路固定費として `iter_total` に対し実測上意味があるのは
   DGX Spark GB10 の N=2048 における出力バッファ確保（`alloc_c`）のみ**
   （`iter_total` の約 8%・`tape_matmul`〈matmul 呼び出し全体〉の約 11%。
   N=512/1024 および M4 Max では無視できる水準）

このため、**本設計単独の期待削減量の上限は DGX N=2048 で約 8〜11%、
N=512/1024・M4 Max ではほぼ 0** であり、#1041 の gate 判定基準
（DGX N=1024 で gemm crate 比 0.786 等）を本設計単独で達成する見込みは
ない。#1292 §15.6 が最大の寄与として挙げた `host_copy`／`checksum` は
ハーネス自身の診断コストであり、本番経路には乗らないため本設計の対象外
とする（§1）。

## §1 背景・実測根拠

`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §15（イシュー #1292）が
framework-compare reuse 境界の 1 反復を 8 区間（Layer B: `alloc_c` /
`tensor_wrap` / `ops_gemm` / `kernel` / `tape_matmul` / `host_copy` /
`checksum` / `iter_total`）へ分解し、両実機（DGX Spark GB10〈Grace CPU〉・
Apple M4 Max）で 5 回独立プロセス中央値実測した。

§15.6 の結論を本設計の入力として引用する:

- **ハーネス診断コスト**（比較用の計測境界の一部であり本番経路には乗らない。
  #965/#970 の既存契約どおり）: `host_copy`（readout の `to_vec`）が DGX で
  `iter_total` の 15.7〜20.3%・M4 Max で 2.0〜7.1%、`checksum`（ホスト f64
  逐次和）が両実機で 5.7〜19.4%。合計は DGX 21.4〜27.8%・M4 Max
  15.4〜22.4%
- **本番経路固定費**（`Sequential::predict`／`Var::matmul` 経由の実運用で
  実際に発生するコスト）: `alloc_c`（出力バッファ確保）が DGX N=2048 のみ
  2.8〜3.0 ms（`iter_total` の約 8%・`tape_matmul` 26.65 ms の約 11%）と
  突出。N=512/1024（DGX 5〜24 µs）・M4 Max（9〜21 µs）では無視できる水準。
  autodiff 残差（tape 登録）は `kernel` 区間の 1〜2 桁下で符号不安定（ノイズ
  水準）。`tensor_wrap` は 1〜5 µs

**未検証事項（仮説として扱う）**: #1292 §15.3 は `alloc_c`（`vec![0.0f32; …]`
によるゼロ初期化確保）の DGX/M4 Max 非対称の機構（calloc の memset か
first-touch page fault か、それがどの区間〈`alloc_c` か `kernel`〉に計上
されるか）を **未特定**と明記している。本設計でもこの点を断定せず、
§3.C で仮説として扱う。

M4 Max N=2048 は §15.4 に「Layer A と Layer B の包含関係が逆転する」
（Layer B の合算が Layer A の `iter_total` を上回る）計測アーティファクトが
記録されており、M4 Max N=2048 由来の Layer B 個別区間の絶対値は参考値
として扱う。

## §2 現状経路（呼び出しチェーン。HEAD `5a66e11` 時点で再確認）

| 段 | 場所 | 内容 |
|---|---|---|
| ハーネス | `scripts/bench/framework-compare/bench-fandhe/src/main.rs:433` `run_gemm_reuse` | `make_tape` → `tape.var(&a_data)`／`tape.var(&b_data)`（1 回）→ 反復ごとに `a.matmul(&b)` → readout（`to_tensor()` + `.contiguous().as_slice().to_vec()`）→ ホスト f64 逐次和 checksum。`--phases` 版は `main.rs:691` `measure_gemm_reuse_phases` |
| facade | `crates/facade/src/lib.rs:239` `tape()` | `CpuBackendOps::new()` を `fandhe_ai_autodiff::Tape::new_with_ops` へ結線する唯一の入口。`Var`／`Tensor`／`VarHostView` は `lib.rs:117-122` で re-export |
| autodiff（葉登録） | `crates/autodiff/src/tape.rs:579` `Tape::var` | `push_eager(Op::Leaf, tensor.clone())`。`Tensor::clone` は `Arc<Storage<T>>`（`crates/tensor-core/src/tensor.rs:53`）のポインタ複製のみ |
| autodiff（matmul） | `crates/autodiff/src/var.rs:195` `Var::matmul` | ①クロステープ検査 → ②`matmul_out_shape` → ③`materialize_fallible(..)?.clone()`（`tape.rs:1305`。実体化済みノードは参照返却 + `Arc` clone）→ ④`self.tape.ops().gemm(&lhs, &rhs)` → ⑤`push_eager(Op::MatMul, value)`（`tape.rs:702`。`shape.to_vec()` + `Vec::push` + `freeze_leaf_prefix`） |
| backend-cpu | `crates/backend-cpu/src/ops.rs:467` `CpuBackendOps::gemm` | `matmul_out_shape` 再検査 → `vec![0.0f32; m * n]`（`ops.rs:472`。出力 C のゼロ初期化確保）→ `gemm_into_slice`（`ops.rs:134`。`dense_transposed_view` 判定 → NN は `a.contiguous()`／`b.contiguous()`〈contiguous なら `Arc` clone のみ。`tensor.rs:560`〉→ `gemm_blis_parallel`）→ `Tensor::new(out, &out_shape)`（`tensor.rs:115`。`Vec` を `Arc::new(Storage { data })` へ move。コピーなし） |
| カーネル | `crates/backend-cpu/src/gemm_blis/mod.rs:463` `gemm_blis_parallel` → `mod.rs:483` `gemm_blis_parallel_with_transpose` | `validate_dims` → `m==1` 専用経路 → 行パネル数を決め rayon `par_chunks_mut` で `dispatch_region`（`GemmDriverVariant::RowPanel` 相当）。C へ FMA 累積する契約のため呼び出し元のゼロ初期化が実質の代入契約を担う |
| readout（ハーネス側） | `var.rs:107` `Var::to_tensor`（`Arc` clone）→ `tensor.rs:560` `contiguous()`（contiguous なら clone）→ `tensor.rs:299` `as_slice` → `.to_vec()`（**唯一の実コピー = `host_copy`**） | 借用版は `var.rs:135` `Var::host_view`（#1335）→ CPU `crates/backend-cpu/src/memory.rs:278` `with_host_view`（`CpuBufferHandle::data` 直接借用・コピーなし）。bench-fandhe の `host-view-readout` feature（#1337）で切替可能 |

## §3 固定費ごとの現状値と変更案

### §3.A contiguous コピー

**現状**: 本番 NN（転置なし）経路では発生していない。葉 A・B は元々
contiguous なテンソルで作られる（framework-compare の入力生成）ため
`Tensor::contiguous()`（`tensor.rs:560`）は非 contiguous 判定に入らず
`Arc` clone のみで返る。実コピーが起きる可視化点は
`GEMM_HOST_REPACK_COUNT`（`ops.rs:83`。転置入力を物理再配置した回数を
数えるスレッドローカルカウンタ）で、NN 経路ではこのカウンタは増えない。

**変更案**: なし（既にゼロ）。回帰テストで `GEMM_HOST_REPACK_COUNT == 0`
を固定する提案を §5 に記す。

ハーネス側の `host_copy`（readout の `to_vec`）は既存の `host-view-readout`
feature（#1337）で借用化・削減可能だが、これは #1117 が定義した gate
プロトコルの計測境界そのものを変える話であり、**採否はユーザー判断事項**
として §8 に回す（本設計では変更しない）。

### §3.B tape 登録

**現状**: `push_eager`（`tape.rs:702`）は `shape.to_vec()` + `Vec::push` +
`Cell` 更新の O(1) 処理で、#1292 §15.3 の autodiff 残差はノイズ水準
（`kernel` 区間の 1〜2 桁下・符号不安定）。

**変更案**: なし。

**reuse ハーネスの tape 保持特性（記録のみ）**: reuse モードは matmul の
出力ノードを tape が生存期間中保持し続けるため（N=2048 で 40 回 ×
16 MiB ≈ 640 MiB）、反復ごとに新規ページへ書き込む形になりアロケータの
ページ再利用が効かない。これは**ハーネス特性**であり本番経路の性質では
ない。`Tape::reset`（`tape.rs:679`）は `&mut self` を要求し `Var` の共有
借用と両立しないため、ハーネスから反復ごとに呼ぶことはできない（葉の
再取得 API も現状ない）。この解消は §9 のスコープ外（将来候補）とする。

本番の `Sequential::predict`（tape 不要経路）はこの特性を持たない
（`docs/inference-forward-fixed-cost-design.md` §3.1、
`docs/perf/cpu-infer-predict-profile.md`）。出力を都度 drop するため。

### §3.C 出力アロケーション（主対象）

**現状**: `CpuBackendOps::gemm`（`ops.rs:472`）が `vec![0.0f32; m * n]` で
出力 C をゼロ初期化確保する。同型の確保は他に 4 箇所ある
（`ops.rs:339`〈`gemm_resident_rhs`〉・`ops.rs:500`〈`gemm_checksum`〉・
`ops.rs:678`〈`gemm_bias_act`〉。`ops.rs:770`／`ops.rs:886` は matmul 以外の
確保）。

DGX N=2048（16 MiB）で 2.8〜3.0 ms・M4 Max で 79 µs という実機非対称の
機構は #1292 §15.3 が「calloc の memset／first-touch page fault のどちら
がどの区間〈`alloc_c`／`kernel`〉に計上されるか未特定」と明記しており、
本設計でも**仮説**として次を記す（断定しない）:

- glibc の mmap しきい値を超える確保（デフォルト 128 KiB 超）は heap
  ではなく mmap 経由になりうるが、閾値は動的調整されるため実行時履歴に
  依存する。heap 経由の場合は `memset` によるゼロ初期化 + 逐次 first-touch
  ページフォルトが `alloc_c` 区間に計上される可能性がある
- macOS の大確保（M4 Max）は VM 直接確保が起きやすく、フォルトが
  `kernel` 側（実際に書き込みが発生するタイミング）へ計上される可能性が
  ある

この仮説の検証（page fault 計数等）は #1299 の検証ステップとして提案する
（§7）。

**候補案**:

- **案 1（主候補・safe Rust）**: 並列ゼロ埋め／first-touch の分散。
  - (1a) `CpuBackendOps::gemm`（`ops.rs:472`）で `vec![0.0]` を rayon の
    並列ゼロ生成（`collect_into_vec` 等）へ置換する。カーネル本体
    （`gemm_blis_parallel`）は不変
  - (1b) `gemm_blis_parallel_with_transpose`（`mod.rs:483`）の行パネル
    タスク内で担当範囲を `fill(0.0)` してから `dispatch_region` を呼ぶ
    （呼び出し元は非ゼロ初期化バッファを渡せる契約へ変える）。
    **`m == 1`（gemv）専用分岐（`mod.rs:508`）も対象に含める**:
    同関数は行パネル分割へ入る前に `m == 1` を判定し
    `gemm_row_vector`／`gemm_row_vector_nt`（`mod.rs:289`／`mod.rs:321`）
    を呼んで早期 `return` する。両関数は既存 `c` を初期値として `+=`
    累積する契約（呼び出し元がゼロ初期化済みである前提）のため、行
    パネル内だけを `fill(0.0)` して非ゼロバッファを渡す契約へ変えると、
    この分岐では旧値が結果に残ってしまう。したがって案 1b を採る場合は
    `m == 1` 分岐の直前でも同じ担当範囲（`c` 全体）を `fill(0.0)` する
    ことを設計に含める
  - いずれも小形状（N=512・1 MiB）では rayon 起動オーバーヘッドが確保
    コストを上回りうるため、サイズ閾値を設けて小形状は既存の逐次
    `vec![0.0f32; …]` を維持する。閾値は #1299 が実測で決定し、#1301 で
    N=512 非後退を確認する
  - 同型確保 4 箇所（`ops.rs:339/472/500/678`）へ横展開しうるが、#1299 は
    まず `gemm`（`ops.rs:472`）限定の適用を推奨する（reuse ハーネスの
    測定対象と一致するため）
- **案 2（後続候補）**: 出力バッファの再利用（プール化）。`Tensor` の
  `Arc<Storage>` に返却フック（`Storage` の Drop 連携）が必要で
  `tensor-core` の内部構造へ踏み込む変更になる。reuse ハーネスは tape が
  全反復の出力を生存期間中保持するため効果を計測できず（§3.B）、
  `Sequential::predict` 型のループ（出力を都度 drop する）向けの後続候補
  として位置づける。**#1299 の対象外**
- **案 3（ユーザー承認必須。列挙のみで採用しない）**: `Vec::with_capacity`
  + `set_len`（未初期化メモリ露出のリスク）・
  `std::alloc::alloc_zeroed`（OS ゼロページ機構への直接依存）・
  `mallopt`（プロセスグローバルなアロケータ挙動変更）・mmap 直接操作
  等の unsafe 系。`.claude/rules/security.md` によりユーザー承認必須の
  ため本設計では候補として記すのみで採用判断は行わない

## §4 bit 一致契約への影響

- **縮約順序**: BLIS 5-loop の k 昇順・行パネル分割（`gemm_blis_parallel_with_transpose`）は案 1/2 いずれでも不変。出力バッファの確保・初期化方法を
  変えるだけでカーネル内部のループ構造・累積順序には触れない
- **FMA 契約**: `f32::mul_add`（`.claude/rules/coding-rust.md`）を用いる
  カーネル本体は不変
- **出力 bit 同一性の条件**: カーネル入口で C が全要素ゼロである限り、
  ゼロ埋めの実行主体（逐次／並列）・並列度は出力の bit 値に影響しない
  （FMA の被加算数が `0.0 + a*b` から始まる点は同一。並列ゼロ埋め自体は
  出力バッファの各要素に独立に `0.0` を書き込むだけで、GEMM カーネンが
  読む値は変わらない）
- **案 2（バッファ再利用）での注意**: dirty なバッファを再利用する場合も
  「カーネル入口前に対象範囲をゼロ埋めする」契約を維持すれば bit 同一
  （`gemm_fp32_strict_into`〈`crates/tensor-core/src/backend_ops.rs:555`。
  CPU 実装は #1212〉が採る `fill(0.0)` → 同一カーネルという契約と同型）
- **不採用とする案（記録のみ）**: 「累積契約を代入契約（beta=0 の
  scale-then-store）へ変えてゼロ埋め自体を省く」案は、accumulator の
  初期値が `+0.0` であることに暗黙に依存する現行契約と数学的には等価
  だが、符号付きゼロの伝播経路（`-0.0` を含む入力での挙動）を個別に
  検証する必要があり、本設計では採らない

## §5 bit 完全一致回帰テスト設計

1. **出力一致テスト**: `CpuBackendOps::gemm` の出力 vs `gemm_blis_parallel`
   を呼び出し元が確保した `vec![0.0f32; m * n]` へ直接呼んだ結果の
   `to_bits()` 全要素一致。形状は N=512/1024/2048 + 非正方 1 形状。入力は
   `Xorshift64Star` による決定的シード（`gemm_into_parity.rs` と同型の
   統合テストとして追加）
2. **contiguous コピー非回帰**: NN 経路で `GEMM_HOST_REPACK_COUNT == 0`
   を維持する検査（クレート内 `#[cfg(test)]`。既存カウンタの読み出しのみ
   で新規計装は不要）
3. **`GemmDriverVariant` 適用区分**:
   - 案 1a（呼び出し元でのゼロ埋め並列化。カーネル自体は不変）は
     `GemmDriverVariant` への追加が**不適切**。variant は「カーネル内部の
     ディスパッチ戦略の違い」を表すための機構であり、呼び出し元の
     バッファ準備方法の違いは対象外
   - 案 1b（driver 内部でのゼロ埋め。`gemm_blis_parallel_with_transpose`
     の呼び出し契約自体を変える）を採る場合のみ、既存 `RowPanel` との
     bit 一致 A/B 比較用に `GemmDriverVariant` への追加が適切
4. 既存の `fma_contract.rs`・`gemm_parity.rs`・`gemm_blis_parity.rs`
   （`crates/backend-cpu/tests/`・`crates/backend-cpu/src/gemm_blis/mod.rs`
   内の parity テスト群）の非後退を条件に含める
5. **案 1b 採用時の受入条件に `m == 1` 分岐の確認を追加する**: 非ゼロ
   初期化バッファ（例えば `f32::NAN` 以外の任意の非ゼロ値で埋めた
   `Vec`）を渡した `m == 1` 形状（gemv 相当）の出力が、既存の
   逐次ゼロ埋め経由の出力と bit 完全一致することを検証する（上記
   §3.C の注記のとおり `gemm_row_vector`／`gemm_row_vector_nt` は `c`
   への累積契約のため、この確認を欠くと `m == 1` 分岐のゼロ初期化
   漏れを検出できない）

## §6 公開 API 非破壊の確認

候補案（案 1a／案 1b／案 2）はいずれも公開 API のシグネチャ・意味論を
変えない:

- `fandhe_ai::tape()`（`facade::lib.rs:239`）・`Tape::var`・`Var::matmul`・
  `Var::to_tensor`・`Tensor::new` は不変
- 案 1a/1b は `crates/backend-cpu`（非公開クレート）に閉じる
- 案 2 は `crates/tensor-core`（非公開クレート）の内部構造変更のみで、
  `facade` が re-export する公開型（`Tensor`・`Var` 等）のフィールド・
  メソッドシグネチャは変えない設計とする

`facade` が唯一のサポートされる公開 API 面である契約
（`docs/compat-api-scope.md` §0）に照らし、本設計のいずれの候補も
追加の公開 API を要しない。

## §7 #1299／#1301 向け受入基準

### 期待削減量（上限。§0 の再掲）

- DGX N=2048: `iter_total` の約 8%・`tape_matmul`（matmul 呼び出し全体）の
  約 11%
- DGX N=512/1024・M4 Max 全形状: ほぼ 0（`alloc_c` が無視できる水準の
  ため）

### 計測方法

- **Layer B（区間別）の接続に必要な変更（#1299 実装時に必須）**:
  `gemm_reuse_phase_diag_tests.rs` の `alloc_c`／`kernel`／`tensor_wrap`
  の 3 区間（`measure_one_phase_trial`。本ファイル冒頭「対応」表参照）は
  `CpuBackendOps::gemm` を呼ばず `vec![0.0f32; n*n]` → `gemm_blis_parallel`
  → `Tensor::new` を**独立に再現**したコードであるため、案 1a／1b の
  変更（`ops.rs:472`・`mod.rs:483`）を `CpuBackendOps::gemm` 側にのみ
  適用しても、この 3 区間の実測値は旧実装のまま変わらない。したがって
  before/after 比較は次のいずれかの手順を踏む必要がある:
  - (i) `alloc_c`／`kernel`／`tensor_wrap` の 3 区間を**変更後の確保・
    ゼロ埋め方式を反映するよう `measure_one_phase_trial` 自体を
    #1299 で更新**したうえで比較する（`ops_gemm`／`tape_matmul` は
    `CpuBackendOps::gemm`／`Var::matmul` を直接呼ぶため無改修で追従
    する）、または
  - (ii) 3 区間を更新しない場合は、本番経路をそのまま呼ぶ
    `ops_gemm`（`ops.gemm(a_tensor, b_tensor)` 呼び出し 1 回。
    `alloc_c`+`kernel`+`tensor_wrap` の本番合成値に相当）のみを
    before/after 比較の基準とし、`alloc_c` 単独の増減は参考値
    （旧実装のレプリカのまま）として扱う
  #1299 は診断テストの更新方針（(i)/(ii) のどちらを採るか）を明記した
  うえで `cargo test -p fandhe-ai-backend-cpu --release -- --ignored
  gemm_reuse_phase_diag_cpu --nocapture` を before/after で実行する。
- **Layer A（実践規模）**: `--phases` は診断専用フラグであり
  `run_gemm_gate_cpu.sh`（`run_gemm_gate.sh` 経由）の標準スイープには
  組み込まれていない（`scripts/bench/framework-compare/README.md`
  「`gemm --mode reuse --phases`」節）。そのため `run_gemm_gate_cpu.sh`
  ではなく `bench-fandhe` バイナリを直接、`run_gemm_gate.sh` と同じ
  `--config patch.crates-io.fandhe-ai.path="<crates/facade 絶対パス>"`
  を付けてビルド・実行する:

  ```bash
  cargo build --release -p bench-fandhe \
    --config 'patch.crates-io.fandhe-ai.path="<crates/facade 絶対パス>"'
  cargo run --release -p bench-fandhe --config \
    'patch.crates-io.fandhe-ai.path="<crates/facade 絶対パス>"' -- \
    --task gemm --device cpu --size <N> --mode reuse --phases
  ```

  `<N>` は 512/1024/2048 を before/after で実行する。ピンは
  `fandhe-ai =0.7.0`（registry 版）を比較対象として維持する（この直接
  実行は path patch 適用時〈参考系列〉のみが対象で、registry 版
  〈正式系列〉との比較は既存 gate 判定〈#1041 指標。後述〉に委ねる）。
  両実機（DGX Spark GB10・Apple M4 Max）で実施し、`RAYON_NUM_THREADS`
  未設定・`host-view-readout` OFF・`--device-checksum` 不使用という
  §2（#1292 に同じ）の計測プロトコルを踏襲する

### 判定基準

- N=512 で非後退（rayon 起動オーバーヘッドによる劣化がないこと）
- DGX N=2048 で `alloc_c` 区間が削減されること
- **本設計の変更前後で出力が bit 完全一致すること**: §5 の回帰テスト
  （`to_bits()` 全要素比較）で検証する。`--phases` 実行時に得られる
  checksum（f64 全要素和）は decimal 表示のため桁落ちで一致しても
  bit 一致の証明にはならず、あくまで目視での粗い非回帰確認に留める
  （bit 完全一致の正は §5 のテストであり checksum ではない）
- **`parity_fail_count == 0`（gemm crate 比較）は上記の bit 一致とは
  別の指標**であり、CPU バックエンドと `gemm` crate の出力を要素単位で
  相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満の複合判定で突合する
  既存の parity テスト（`.claude/rules/coding-rust.md`「バックエンド間
  数値一致は統一複合判定」節）の合格を指す。本設計は同一実装内の
  ゼロ埋め方式変更のみで `gemm` crate 側には触れないため、この判定は
  bit 一致（本設計変更前後）が成立すれば自動的に非後退となる
- #1041 が定義する gate 指標（gemm crate 比）が非後退であること
  （本設計単独での達成は見込まないが、後退させないことを条件とする）

### ログ配置

`docs/perf/logs/cpu-matmul-fixed-cost-<issue番号>/`（内部ホスト名等の
環境固有情報は含めない。`docs/real-hardware-verification-env.md` の方針
に従う）

## §8 ユーザー判断事項

- gate プロトコル（#1117 が定義）への `host-view-readout`／
  `--device-checksum` 採用可否（ハーネス診断コスト 15〜28% の扱いを
  変えるかどうかの判断）
- 案 3（unsafe: `set_len`／`alloc_zeroed`／`mallopt`／mmap 直接操作）の
  導入可否
- 案 2（`tensor-core` の `Storage` 返却フック実装）への着手可否

## §9 スコープ外・出典

`.claude/rules/out-of-scope-tracking.md` に従い記録のみとする（起票は
ユーザー承認後に行う）。

- `alloc_c`（DGX N=2048 出力バッファ確保コスト）削減の実装・結線（#1299）
- 両実機 before/after 実測・結線可否判断（#1301）
- 出力バッファ再利用（案 2）のための `tensor-core` `Storage` 返却フック
  実装
- `Tape::reset` と葉ハンドル再取得 API（reuse ハーネスの tape 保持特性の
  解消）
- gate プロトコルへの `host-view-readout`／`--device-checksum` 採用
  （ユーザー判断事項。§8）
- DGX/M4 Max の `alloc_c` 非対称の機構特定（page fault 計数による検証。
  #1299 の検証ステップとして提案）
- `ops.rs:339/500/678`（`gemm_resident_rhs`／`gemm_checksum`／
  `gemm_bias_act`）への横展開

### 出典

- イシュー #1294（本設計記録）・親 #1285・ルート #1283
- イシュー #1292（前提。両実機フェーズ分解実測）・
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §15・
  `docs/perf/logs/cpu-gemm-reuse-phase-1292/`
- `docs/inference-forward-fixed-cost-design.md`（`Sequential::predict`
  〈tape 不要経路〉の固定費設計。本設計は `Var::matmul`〈tape 経路〉が
  対象という点で棲み分ける）
- `docs/perf/cpu-infer-predict-profile.md`（`Sequential::predict` の
  フェーズ分解）
- `docs/device-memory-pool-design.md`（`DeviceBuffer` 向けプール設計。
  案 2 の `tensor-core` 版検討時の参考）
- `.claude/rules/coding-rust.md`（FMA 契約・数値一致複合判定）
- `.claude/rules/security.md`（unsafe 導入の承認要件）
- `docs/compat-api-scope.md` §0（`facade` の公開 API サポート境界）
