# イシュー #1478 セキュリティ監査記録（unsafe 経路の既定化）

`.claude/rules/security.md`「unsafe」節（「使用時は理由コメント＋レビュー
〈security-auditor〉必須」）・「レビュー体制」節（「依存追加・ガードレール・
CI/hooks 変更を含む PR は security-auditor の監査を並列で実施する」）に
対応する記録。本イシューは依存追加・ガードレール変更ではないが、既定で
unsafe 経路（`HostStaging::alloc` の `HostStagingKind::Pinned` 分岐）を
通すようになる変更であるため、PR レビュー段階の security-auditor 到達に
先立ち、実装フェーズで以下の機械的監査を実施した。**正式な承認は PR
レビュー（security-auditor 到達）で行う**（本記録はそれに先立つ自己監査
であり、承認そのものではない）。

## 監査項目と結果

| # | 項目 | コマンド | 結果 |
|---|------|---------|------|
| 1 | `unsafe` ブロック数が 1 箇所のまま増えていないこと | `grep -n "unsafe " crates/backend-cuda/src/host_staging.rs` | 実コードの `unsafe { .. }` ブロックは `let pinned = unsafe { ctx.alloc_pinned::<f32>(numel)? };`（line 192）の 1 箇所のみ。他のヒット（7 件中残り 6 件）はすべてドキュメンテーションコメント中の「unsafe」という語（プロース）であり実コードではない |
| 2 | `memory.rs` の `unsafe` 使用箇所数が変わっていないこと | `git diff e8cd3a2 HEAD -- crates/backend-cuda/src/memory.rs \| grep unsafe` | 差分中の全ヒットはドキュメンテーションコメントの追記・言い換え（「`unsafe` は追加しない」という記述自体）のみで、実コードの `unsafe` ブロックは追加されていない |
| 3 | `// SAFETY:` ブロックの内容が不変であること | `git diff e8cd3a2 HEAD -- crates/backend-cuda/src/host_staging.rs \| grep -A25 "SAFETY:"` | 差分に SAFETY ブロック自体の変更行は現れない（=byte 単位で不変。周辺のモジュール冒頭コメント・enum doc のみ書き換えた） |
| 4 | `HOST_STAGING_CAP_BYTES`（DoS 対策の確保上限）が不変であること | `git diff e8cd3a2 HEAD -- crates/backend-cuda/src/host_staging.rs \| grep HOST_STAGING_CAP_BYTES` | 差分に現れない（`256 * 1024 * 1024` のまま不変） |
| 5 | `release_host_staging`（REQ-14 明示解放 API）が不変であること | `git diff e8cd3a2 HEAD -- crates/backend-cuda/src/memory.rs \| grep -A10 "pub fn release_host_staging"` | 差分に現れない（実装不変） |
| 6 | `alloc_pinned` 確保失敗が `?` で fail-closed 伝播し、`Pageable` への暗黙フォールバックが追加されていないこと | `grep -n "alloc_pinned" crates/backend-cuda/src/host_staging.rs` | `let pinned = unsafe { ctx.alloc_pinned::<f32>(numel)? };` のまま `?` 伝播。`match kind { .. }` の分岐構造も不変で、`Pinned` 確保失敗時に `Pageable` へ切り替えるフォールバック分岐は追加していない |

## 結論

`HOST_STAGING_KIND` の既定を `Pinned` へ切り替えたことにより、キャッシュ
miss 時にこの unsafe ブロックへ到達する頻度は増える（切替前は
`new_with_host_staging_kind` 経由の明示選択・実機 `#[ignore]` テストの
みが通っていた）が、**unsafe ブロック自体の実装・安全性根拠
（`// SAFETY:` コメント）・確保上限・解放 API・エラー伝播契約は一切
変更していない**。安全性根拠（`memcpy_dtoh` で全域上書き後にのみ
`as_slice()` で読み出す・`f32` は全ビットパターン有効）は既定経路でも
そのまま成立する。

以上は実装フェーズでの機械的自己監査であり、`.claude/rules/security.md`
が求める security-auditor 到達（PR レビュー段階）を代替するものではない。
