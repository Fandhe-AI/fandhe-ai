# Metal タイルロード経路: `simdgroup_async_copy` 系 AIR intrinsic 不採用判断（#546）

イシュー #546「非公式 `simdgroup_async_copy` 系 API の採否判断を記録（不採用）」に対応する。
親 #530（Phase D: Metal マルチ simdgroup 化・ロード最適化）・ルート #479（GEMM 性能改善）の検討過程で、
参照実装 metal-flash-attention（MFA。以下 `philipturner/metal-flash-attention`）が採る `simdgroup_async_copy` 系
AIR intrinsic 直接バインドの採否が論点になった。本ドキュメントは `docs/backend-metal-wgpu-decision.md`（#41・TASK-1.8d）
と同型の決定記録として、判断とその根拠を残す。

対象 API の定義: Apple 公開の Metal Shading Language（MSL）仕様に存在しない `simdgroup_async_copy` / `simdgroup_event`
系の AIR（Apple Intermediate Representation）intrinsic を、Swift/MSL コード中の `__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")`
等の形式で直接バインドして呼び出す手法。

## 判断サマリ

**`backend-metal` の threadgroup メモリへのタイルロードは、現行の staged 協調ロード方式
（`crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_tiled`）を維持し、`simdgroup_async_copy` 系
AIR intrinsic は不採用とする。**

## 根拠

MFA リポジトリ（`philipturner/metal-flash-attention`、参照時点コミット `8671cddc38f19a6eadb804dee6a3ca2954b8bf32`）の
精読結果を根拠とする。

1. **非公式性**: `simdgroup_async_copy` 系 intrinsic は Apple 公開の MSL 仕様に存在せず、MFA
   `Sources/FlashAttention/GEMM/GEMMHeaders.swift:63-76` は `__metal_simdgroup_async_copy_2d` を
   `__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")` 等で LLVM/AIR ビットコードへ直接バインドしている。
   コンパイラ・OS 更新でこの非公開 ABI が壊れないことを外部が保証しない。
2. **ハング報告**: 同ファイル `:10-23` のドキュメンテーションコメントに、作者自身による M1 GPU 上のハードウェアバグ
   報告がある。「async copy を発行したら、カーネル終了前に必ず別の `threadgroup_barrier` に到達し、かつコピー結果を
   最低 1 スレッドが読む（デリファレンスする）こと」が回避策として明記されており、これを守らないと GPU がフリーズし
   再起動が必要になりうるとされている。使用制約が非自明でデバッグが困難な API である。
3. **M3/M4 以降では既定で無効化**: MFA `Sources/FlashAttention/GEMM/GEMMDescriptor/GEMMDescriptor.swift:212-216` は
   `mtlDevice.supportsFamily(.apple9)`（Apple9 ファミリー、M3/M4 以降相当）が真の場合に `preferAsyncLoad = false` を
   既定とし、そうでない場合のみ `true` とする。これは確認できる設定値の事実であり、**MFA 自身が新しい GPU 世代では
   async copy を既定で使わない**という実装選択を示す。この設定切替のみを根拠に「async copy が性能悪化を招く」と
   因果関係を断定することはできない（ベンチ実測・MFA 側のコメント等の裏付けは参照時点コミットの当該箇所に見当たらず、
   本ドキュメントも未実施）。
4. **fallback 経路の存在**: 上記のとおり MFA 自身が async copy なしの経路を（Apple9 以降の）既定として持つ。
   `backend-metal` の現行 staged 協調ロード（`gemm_simdgroup_tiled`）は同系の経路であり、非公式 API に頼らずとも
   Phase D の他施策（ベクトル化ロード・パディング・タイル選択強化等）による改善余地がある。
5. **本リポ方針との整合**: 完全自作コア・保守性を重視する方針（`.claude/rules/coding-rust.md`）の観点から、
   Apple 非公開 ABI（AIR intrinsic）への `__asm` 直接依存は、コンパイラ更新で無警告に壊れうる保守負債であり採らない。

## REQ-8（カーネル境界検査規約）との関係

`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」は、性能下限・最適化達成を理由にシェーダ・カーネル側の
手動境界チェックを省略しないことを CPU・CUDA・Metal の全カーネルに義務付ける。本判断は「タイルロード手段として非公式
intrinsic を使うか否か」という性能施策の選択であり、境界検査の省略とは無関係である。本ドキュメントの不採用判断を
「境界検査を省略してよい根拠」として引用してはならない（`docs/backend-metal-wgpu-decision.md` と同じ扱い）。

## 再検討条件

Phase D の他手段（#530 配下の兄弟イシュー。ベクトル化ロード・パディング・タイル選択強化等）を尽くしてもなお REQ-8 の性能目標に
未達の場合に限り、再検討候補としてよい。ただしその場合も、非公式 API の採用はガードレール的に人間判断が必要な事項であり
**ユーザー承認必須**とする（`.claude/rules/security.md`「自己修復ループ固有のガードレール」・イシュー #546 共通契約と同旨）。

## 出典

- MFA リポジトリ `philipturner/metal-flash-attention`（参照時点コミット `8671cddc38f19a6eadb804dee6a3ca2954b8bf32`。
  `gh api repos/philipturner/metal-flash-attention/commits/main` で解決）
  - `Sources/FlashAttention/GEMM/GEMMHeaders.swift:10-23`（M1 GPU ハング報告のドキュメンテーションコメント）
  - `Sources/FlashAttention/GEMM/GEMMHeaders.swift:63-76`（`__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")` 直接バインド）
  - `Sources/FlashAttention/GEMM/GEMMDescriptor/GEMMDescriptor.swift:212-216`（`supportsFamily(.apple9)` で
    `preferAsyncLoad` 既定値を切替）
- `crates/backend-metal/src/shaders/gemm.metal`（現行 staged 協調ロード方式 `gemm_simdgroup_tiled`）
- `.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」
- `.claude/rules/security.md`「自己修復ループ固有のガードレール」
- `docs/backend-metal-wgpu-decision.md`（同型の決定記録。フォーマットの踏襲元）
- イシュー #546・親 #530・ルート #479
