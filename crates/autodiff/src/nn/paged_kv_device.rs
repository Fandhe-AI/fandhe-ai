//! デバイス常駐ページ化 KV キャッシュ `PagedKvCache`（K-3。イシュー #3111・親 #3108。
//! 設計正本 `docs/device-resident-kv-paged-attention-design.md` §2・§4・§6）。
//!
//! # 役割と責務境界
//!
//! K-1 の [`super::KvCache`]（ホスト `Tensor<f32>` を `Var::cat` で毎ステップ再確保）に
//! 対し、K／V をデバイスバッファ上のブロックプール（`[max_blocks, block_size, E]`）へ
//! 置き、追記と attention をデバイスで行う **1 層ぶんの値型**。
//!
//! - ページテーブルと空きブロック管理は #3110 の [`super::paged_kv`] の `BlockTable`
//!   （ホストロジック）を再利用する。デバイスへは「スロット写像」
//!   （`block_id · block_size + pos % block_size`）と平坦化ブロックテーブル（未使用枠は
//!   番兵 `u32::MAX`）だけを渡す
//! - 追記・attention の実体は `BackendOps::paged_kv_append_tracked`／
//!   `paged_attention_forward_tracked`（CPU・CUDA・Metal。`tensor-core`）。本型は
//!   `ops` を持たず、毎回 `&Tape` を受けて `tape.ops()` を使う（`DeviceParamStore` と同型）
//! - Tape 入口（`StatefulAttention::forward_paged` 等）と facade 公開は #3112。ここでは
//!   `Tensor`（非追跡）を受けるので `requires_grad` な `Var` は構造的に受けない
//!
//! # 同期点（1 ステップに 1 回）
//!
//! [`PagedKvCache::attention`] は `upload`（同期しない）→ `paged_attention_forward_tracked`
//! （積むだけ）→ `download`（唯一の同期点）の順で、ホストとの同期は出力の `download`
//! だけ。`append` は同期しない（同一ストリーム／コマンドバッファの順序で後続の
//! attention が追記後のプールを読む）。
//!
//! # 失敗時の状態（OWASP A08）
//!
//! - 追記は `BlockTable` を**複製**して確保とスロット計算を行い、バックエンドの書き込みが
//!   成功したときだけ複製を確定する。バックエンドが `Err` を返したらプールに書きかけが
//!   残りうるので `poisoned` を立てる（論理状態〈系列長・テーブル〉は変わらない）
//! - Metal の遅延失敗は `failure_token` に登録される。`attention` は `download` の**後**に
//!   `failure_token.is_set()` を再検査し、立っていれば `poisoned` へ移してエラーにする
//! - `poisoned` 後の `append`／`attention` は [`BackendError::StorePoisoned`] で拒否する。
//!   回復は新しい `PagedKvCache` の構築のみ（`rewind_to`／`clear` は論理状態の操作で
//!   許す。データの再書き込みは次の追記が行う）
//!
//! # 縮約順序と判定
//!
//! 数値の縮約順序は CPU 参照（`paged_attention_reference`）と同じ（系列方向・`Dh` 方向とも
//! 論理順の単一ループ）。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
//! そのまま適用する（`tensor-core::paged_kv` と各バックエンドの rustdoc 参照）。
//!
//! # 可変長
//!
//! `seq_lens` はバックエンド API が行ごとに受ける形だが、本型は #3110 の不変条件
//! 「全行が同じ長さ」を引き継ぐ。行ごとの可変長は #3104 の結論待ち。
//!
//! 新規 `unsafe` なし。

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicBool, Ordering};

use fandhe_ai_tensor_core::{
    BackendError, Device, DeviceBuffer, DispatchFailureCell, PagedAttentionArgs, PagedKvLayout,
    Tensor,
};

use super::paged_kv::{BlockTable, PagedKvCacheConfig, invalid};
use crate::error::AutodiffError;
use crate::tape::Tape;

/// 1 層ぶんのデバイス常駐ページ化 KV キャッシュ（K-3）。多層は `Vec<PagedKvCache>`。
///
/// `Clone` は実装しない（デバイスバッファの複製は明示的な設計判断が要るため。
/// K-1 のスナップショット退避に相当する用途は #3112 で扱う）。
#[derive(Debug)]
pub struct PagedKvCache {
    device: Device,
    config: PagedKvCacheConfig,
    layout: PagedKvLayout,
    k_pool: DeviceBuffer<f32>,
    v_pool: DeviceBuffer<f32>,
    table: BlockTable,
    poisoned: AtomicBool,
    failure_token: DispatchFailureCell,
}

fn backend(e: BackendError) -> AutodiffError {
    AutodiffError::Backend(e)
}

impl PagedKvCache {
    /// プールを `tape` のバックエンドへ確保して空のキャッシュを作る。
    ///
    /// `memory_ops()` が `None` なら [`BackendError::Unsupported`]、確保失敗は型付きエラー
    /// （abort しない）。`max_blocks · block_size` が `u32::MAX` を超える設定は
    /// スロット番号が `u32` に収まらないので拒否する。
    pub fn new(tape: &Tape, config: PagedKvCacheConfig) -> Result<PagedKvCache, AutodiffError> {
        let ops = tape.ops();
        let mem = ops.memory_ops().ok_or_else(|| {
            backend(BackendError::Unsupported(
                "PagedKvCache::new: backend does not implement MemoryOps".into(),
            ))
        })?;
        let rows = config
            .max_blocks()
            .checked_mul(config.block_size())
            .ok_or_else(|| invalid("max_blocks · block_size がオーバーフローした"))?;
        if u32::try_from(rows).is_err() {
            return Err(invalid(
                "max_blocks · block_size が u32::MAX を超える（スロット番号が u32）",
            ));
        }
        let layout =
            PagedKvLayout::new(config.max_blocks(), config.block_size(), config.embed_dim())
                .map_err(backend)?;
        let shape = [config.max_blocks(), config.block_size(), config.embed_dim()];
        let k_pool = mem.alloc_zeroed(&shape).map_err(backend)?;
        let v_pool = mem.alloc_zeroed(&shape).map_err(backend)?;
        Ok(PagedKvCache {
            device: ops.device(),
            config,
            layout,
            k_pool,
            v_pool,
            table: BlockTable::default(),
            poisoned: AtomicBool::new(false),
            failure_token: DispatchFailureCell::new(),
        })
    }

    /// 設定。
    pub fn config(&self) -> &PagedKvCacheConfig {
        &self.config
    }

    /// 常駐するデバイス。
    pub fn device(&self) -> Device {
        self.device
    }

    /// 有効系列長（空なら 0）。
    pub fn seq_len(&self) -> usize {
        self.table.seq_len()
    }

    /// バッチサイズ `B`（最初の追記で確定。空なら `None`）。
    pub fn batch(&self) -> Option<usize> {
        self.table.batch
    }

    /// 使用中のブロック総数（全行の合計）。
    pub fn num_used_blocks(&self) -> usize {
        self.table.tables.iter().map(Vec::len).sum()
    }

    /// 空きブロック数。
    pub fn num_free_blocks(&self) -> usize {
        self.table.free_blocks(self.config.max_blocks())
    }

    /// 以前の失敗でプールの内容が信頼できない状態か（遅延失敗トークンの検出は
    /// 次の `append`／`attention` で行う）。
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst) || self.failure_token.is_set()
    }

    /// poison 検査。`failure_token`（Metal の遅延失敗）が立っていれば `poisoned` へ移す。
    fn check_not_poisoned(&self) -> Result<(), AutodiffError> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(backend(BackendError::StorePoisoned));
        }
        if self.failure_token.is_set() {
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(backend(BackendError::StorePoisoned));
        }
        Ok(())
    }

    fn check_device(&self, tape: &Tape) -> Result<(), AutodiffError> {
        if tape.ops().device() != self.device {
            return Err(backend(BackendError::DeviceMismatch));
        }
        Ok(())
    }

    /// 新トークンの K／V（射影済み `[B, L_new, E]`）を末尾へ追記する。
    ///
    /// 検証（rank・shape 一致・`E`・`B`・長さオーバーフロー・空きブロック数）を
    /// すべて通してからデバイスへ書く。`Err` のとき論理状態は変わらない。バックエンドが
    /// 書き込みで失敗した場合は `poisoned` になる（モジュール doc）。`L_new == 0` は
    /// 検証だけして `Ok`（`B` も確定しない）。バックエンドが追記を提供しない
    /// （`Unsupported`）場合も K-1 へ暗黙フォールバックせずそのまま返す（設計 §4）。
    pub fn append(
        &mut self,
        tape: &Tape,
        k_new: &Tensor<f32>,
        v_new: &Tensor<f32>,
    ) -> Result<(), AutodiffError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        let shape = k_new.shape().to_vec();
        if shape.len() != 3 {
            return Err(invalid(format!(
                "k_new は rank 3 が必要（実際: {}）",
                shape.len()
            )));
        }
        if v_new.shape() != shape.as_slice() {
            return Err(invalid("k_new と v_new の shape が一致しない"));
        }
        let (b, l_new, e) = (shape[0], shape[1], shape[2]);
        if e != self.config.embed_dim() {
            return Err(invalid(format!(
                "最終軸 {e} が embed_dim {} と一致しない",
                self.config.embed_dim()
            )));
        }
        // 確保とスロット計算は複製に対して行い、書き込み成功後にだけ確定する。
        let mut next = self.table.clone();
        let slots =
            next.extend_for_append(self.config.max_blocks(), self.config.block_size(), b, l_new)?;
        if l_new == 0 {
            return Ok(());
        }
        let ops = tape.ops();
        let mem = ops.memory_ops().ok_or_else(|| {
            backend(BackendError::Unsupported(
                "PagedKvCache::append: backend does not implement MemoryOps".into(),
            ))
        })?;
        let k_dev = mem.upload(k_new).map_err(backend)?;
        let v_dev = mem.upload(v_new).map_err(backend)?;
        if let Err(err) = ops.paged_kv_append_tracked(
            &k_dev,
            &v_dev,
            &mut self.k_pool,
            &mut self.v_pool,
            &slots,
            &self.layout,
            &self.failure_token,
        ) {
            // 引数検証エラーはデバイスに触れる前に返るが、実行時失敗はプールに書きかけを
            // 残しうる。区別できないので fail-closed に poison する（`Unsupported` は
            // 何も書いていないので poison しない）。
            if !matches!(err, BackendError::Unsupported(_)) {
                self.poisoned.store(true, Ordering::SeqCst);
            }
            return Err(backend(err));
        }
        self.table = next;
        Ok(())
    }

    /// 追記済みのキャッシュに対する attention（head 結合後・`out_proj` 適用前の
    /// `[B, L_new, E]`）。呼び出し側は新トークンを [`PagedKvCache::append`] した**後**に
    /// 呼ぶ（`s_prev = seq_len − L_new`）。`L_new == 0` はデバイスを呼ばず `[B, 0, E]`。
    ///
    /// ホストとの同期は出力の `download` の 1 回だけ。`download` 後に遅延失敗トークンを
    /// 再検査し、立っていれば `poisoned` にしてエラーを返す。
    pub fn attention(
        &self,
        tape: &Tape,
        q: &Tensor<f32>,
        num_heads: usize,
        scale: f32,
    ) -> Result<Tensor<f32>, AutodiffError> {
        self.check_not_poisoned()?;
        self.check_device(tape)?;
        let shape = q.shape().to_vec();
        if shape.len() != 3 {
            return Err(invalid(format!(
                "q は rank 3 が必要（実際: {}）",
                shape.len()
            )));
        }
        let (b, l_new, e) = (shape[0], shape[1], shape[2]);
        if e != self.config.embed_dim() {
            return Err(invalid("q の最終軸がキャッシュの embed_dim と一致しない"));
        }
        if let Some(cb) = self.table.batch
            && cb != b
        {
            return Err(invalid("q のバッチがキャッシュと一致しない"));
        }
        let s_total = self.seq_len();
        if l_new > s_total {
            return Err(invalid(format!(
                "L_new {l_new} が有効長 {s_total} を超える（先に append すること）"
            )));
        }
        if l_new == 0 || b == 0 {
            return Tensor::new(Vec::new(), &[b, l_new, e]).map_err(AutodiffError::Shape);
        }
        let max_blocks_per_seq = self
            .table
            .tables
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(1);
        let block_tables = self.table.flatten_tables(max_blocks_per_seq)?;
        let seq_len_u32 =
            u32::try_from(s_total).map_err(|_| invalid("系列長が u32 に収まらない"))?;
        let mut seq_lens: Vec<u32> = Vec::new();
        seq_lens
            .try_reserve_exact(b)
            .map_err(|_| invalid("seq_lens の確保に失敗した"))?;
        seq_lens.resize(b, seq_len_u32);
        let args =
            PagedAttentionArgs::new(self.layout, b, l_new, num_heads, max_blocks_per_seq, scale)
                .map_err(backend)?;

        let ops = tape.ops();
        let mem = ops.memory_ops().ok_or_else(|| {
            backend(BackendError::Unsupported(
                "PagedKvCache::attention: backend does not implement MemoryOps".into(),
            ))
        })?;
        let q_dev = mem.upload(q).map_err(backend)?;
        let out_dev = ops
            .paged_attention_forward_tracked(
                &q_dev,
                &self.k_pool,
                &self.v_pool,
                &block_tables,
                &seq_lens,
                &args,
                &self.failure_token,
            )
            .map_err(|e| {
                if !matches!(e, BackendError::Unsupported(_)) {
                    self.poisoned.store(true, Ordering::SeqCst);
                }
                backend(e)
            })?;
        // 1 ステップで唯一の同期点。
        let out = mem.download(&out_dev).map_err(backend)?;
        // download 完了後に遅延失敗（Metal の token）を再検査する。
        if self.failure_token.is_set() {
            self.poisoned.store(true, Ordering::SeqCst);
            return Err(backend(BackendError::StorePoisoned));
        }
        Ok(out)
    }

    /// 有効長を `new_len` へ縮め、余る末尾ブロックを空きへ返す（#3110 の
    /// `PagedKvCacheReference::rewind_to` と同じ意味。デバイスのデータは消さず、
    /// 再利用時の追記が上書きする）。`new_len > seq_len` は `Err`（状態不変）。
    pub fn rewind_to(&mut self, new_len: usize) -> Result<(), AutodiffError> {
        let mut next = self.table.clone();
        next.rewind_to(new_len, self.config.block_size())?;
        self.table = next;
        Ok(())
    }

    /// すべてのブロックを空きへ戻し、`B` を未確定に戻す。
    pub fn clear(&mut self) {
        self.table.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::{BackendOps, BufferHandle, MemoryOps};
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct MockHandle {
        data: Vec<f32>,
    }

    impl BufferHandle for MockHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Mode {
        /// 正常（追記・attention を記録して成功する）。
        Ok,
        /// 既定どおり `Unsupported`（K-1 へフォールバックしないことの確認用）。
        Unsupported,
        /// 追記が実行時エラー。
        AppendFails,
        /// attention の計算は成功し、`token` を立てる（download 後の再検査の再現）。
        AttentionSetsToken,
    }

    #[derive(Default)]
    struct Log {
        appends: Vec<Vec<u32>>,
        tables: Vec<(Vec<u32>, Vec<u32>)>,
        downloads: usize,
    }

    struct MockOps {
        mode: Mode,
        no_memory: bool,
        log: Arc<Mutex<Log>>,
    }

    impl MockOps {
        fn new(mode: Mode) -> (Self, Arc<Mutex<Log>>) {
            let log = Arc::new(Mutex::new(Log::default()));
            (
                MockOps {
                    mode,
                    no_memory: false,
                    log: log.clone(),
                },
                log,
            )
        }

        fn buf(shape: &[usize], data: Vec<f32>) -> DeviceBuffer<f32> {
            DeviceBuffer::new(Device::Cpu, shape.to_vec(), Box::new(MockHandle { data }))
        }
    }

    impl MemoryOps for MockOps {
        fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError> {
            Ok(Self::buf(shape, vec![0.0; shape.iter().product()]))
        }
        fn upload(&self, t: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError> {
            Ok(Self::buf(
                t.shape(),
                t.contiguous().as_slice().unwrap_or(&[]).to_vec(),
            ))
        }
        fn download(&self, b: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError> {
            if let Ok(mut l) = self.log.lock() {
                l.downloads += 1;
            }
            let h = b
                .downcast_handle::<MockHandle>()
                .ok_or(BackendError::DeviceMismatch)?;
            Tensor::new(h.data.clone(), b.shape()).map_err(BackendError::ShapeMismatch)
        }
    }

    fn unused() -> BackendError {
        BackendError::Unsupported("mock: unused".into())
    }

    impl BackendOps for MockOps {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn memory_ops(&self) -> Option<&dyn MemoryOps> {
            if self.no_memory { None } else { Some(self) }
        }
        // 以下は `BackendOps` の必須メソッド。本テストでは使わない。
        fn gemm(&self, _: &Tensor<f32>, _: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn add(&self, _: &Tensor<f32>, _: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn mul(&self, _: &Tensor<f32>, _: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn relu(&self, _: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn exp(&self, _: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn tanh(&self, _: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn sum(&self, _: &Tensor<f32>, _: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn max(&self, _: &Tensor<f32>, _: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(unused())
        }
        fn paged_kv_append(
            &self,
            _k: &DeviceBuffer<f32>,
            _v: &DeviceBuffer<f32>,
            _kp: &mut DeviceBuffer<f32>,
            _vp: &mut DeviceBuffer<f32>,
            slot_mapping: &[u32],
            _layout: &PagedKvLayout,
        ) -> Result<(), BackendError> {
            match self.mode {
                Mode::Unsupported => Err(BackendError::Unsupported("mock".into())),
                Mode::AppendFails => Err(BackendError::KernelLaunchFailed("mock append".into())),
                _ => {
                    if let Ok(mut l) = self.log.lock() {
                        l.appends.push(slot_mapping.to_vec());
                    }
                    Ok(())
                }
            }
        }
        fn paged_attention_forward_tracked(
            &self,
            q: &DeviceBuffer<f32>,
            _kp: &DeviceBuffer<f32>,
            _vp: &DeviceBuffer<f32>,
            block_tables: &[u32],
            seq_lens: &[u32],
            _args: &PagedAttentionArgs,
            token: &DispatchFailureCell,
        ) -> Result<DeviceBuffer<f32>, BackendError> {
            if self.mode == Mode::Unsupported {
                return Err(BackendError::Unsupported("mock".into()));
            }
            if let Ok(mut l) = self.log.lock() {
                l.tables.push((block_tables.to_vec(), seq_lens.to_vec()));
            }
            if self.mode == Mode::AttentionSetsToken {
                token.set(BackendError::KernelLaunchFailed("mock deferred".into()));
            }
            self.alloc_zeroed(q.shape())
        }
    }

    fn tape(mode: Mode) -> (Tape, Arc<Mutex<Log>>) {
        let (ops, log) = MockOps::new(mode);
        (Tape::new_with_ops(Box::new(ops)), log)
    }

    fn cfg() -> PagedKvCacheConfig {
        PagedKvCacheConfig::new(2, 3, 4).unwrap()
    }

    fn x(b: usize, l: usize) -> Tensor<f32> {
        Tensor::new(vec![0.5f32; b * l * 4], &[b, l, 4]).unwrap()
    }

    #[test]
    fn new_requires_memory_ops() {
        let (mut ops, _) = MockOps::new(Mode::Ok);
        ops.no_memory = true;
        let t = Tape::new_with_ops(Box::new(ops));
        let r = PagedKvCache::new(&t, cfg());
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
    }

    #[test]
    fn append_builds_row_major_slot_mapping_and_commits() {
        let (t, log) = tape(Mode::Ok);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        // B=2・L_new=3: 各行 2 ブロック（block_size 2）。確保順は row0: 0,1・row1: 2（空きは 3 つ）。
        // 空きが足りない（必要 4・空き 3）ので Err。
        assert!(c.append(&t, &x(2, 3), &x(2, 3)).is_err());
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.batch(), None);
        // B=1・L_new=3 は成功（ブロック 0,1）。
        c.append(&t, &x(1, 3), &x(1, 3)).unwrap();
        assert_eq!(c.seq_len(), 3);
        assert_eq!(c.num_used_blocks(), 2);
        let appends = log.lock().unwrap().appends.clone();
        assert_eq!(appends, vec![vec![0, 1, 2]]);
        // 続けて 2 トークン: pos 3 は block 1 の 2 行目（slot 3）、pos 4 は新ブロック 2（slot 4）。
        c.append(&t, &x(1, 2), &x(1, 2)).unwrap();
        assert_eq!(log.lock().unwrap().appends[1], vec![3, 4]);
        assert_eq!(c.num_free_blocks(), 0);
    }

    #[test]
    fn append_rejects_bad_shapes_without_state_change() {
        let (t, _) = tape(Mode::Ok);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        let bad_e = Tensor::new(vec![0.0f32; 6], &[1, 2, 3]).unwrap();
        assert!(c.append(&t, &bad_e, &bad_e).is_err());
        let rank2 = Tensor::new(vec![0.0f32; 4], &[1, 4]).unwrap();
        assert!(c.append(&t, &rank2, &rank2).is_err());
        assert!(c.append(&t, &x(1, 2), &x(1, 1)).is_err());
        assert_eq!(c.seq_len(), 0);
        c.append(&t, &x(1, 1), &x(1, 1)).unwrap();
        // バッチ不一致。
        assert!(c.append(&t, &x(2, 1), &x(2, 1)).is_err());
        assert_eq!(c.seq_len(), 1);
        assert!(!c.is_poisoned());
    }

    #[test]
    fn append_failure_keeps_logical_state_and_poisons() {
        let (t, _) = tape(Mode::AppendFails);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        let r = c.append(&t, &x(1, 2), &x(1, 2));
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ));
        assert_eq!(c.seq_len(), 0);
        assert_eq!(c.num_free_blocks(), 3);
        assert_eq!(c.batch(), None);
        assert!(c.is_poisoned());
        assert!(matches!(
            c.append(&t, &x(1, 1), &x(1, 1)),
            Err(AutodiffError::Backend(BackendError::StorePoisoned))
        ));
        assert!(matches!(
            c.attention(&t, &x(1, 1), 2, 1.0),
            Err(AutodiffError::Backend(BackendError::StorePoisoned))
        ));
        // 論理状態の操作は許す。
        c.clear();
    }

    #[test]
    fn unsupported_propagates_without_fallback_or_poison() {
        let (t, _) = tape(Mode::Unsupported);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        let r = c.append(&t, &x(1, 1), &x(1, 1));
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::Unsupported(_)))
        ));
        assert!(!c.is_poisoned());
        assert_eq!(c.seq_len(), 0);
    }

    #[test]
    fn attention_flattens_tables_with_sentinel_and_downloads_once() {
        let (t, log) = tape(Mode::Ok);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        c.append(&t, &x(1, 3), &x(1, 3)).unwrap();
        let out = c.attention(&t, &x(1, 3), 2, 0.5).unwrap();
        assert_eq!(out.shape(), &[1, 3, 4]);
        let l = log.lock().unwrap();
        assert_eq!(l.downloads, 1);
        assert_eq!(l.tables, vec![(vec![0, 1], vec![3])]);
    }

    #[test]
    fn attention_validates_inputs() {
        let (t, _) = tape(Mode::Ok);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        c.append(&t, &x(1, 2), &x(1, 2)).unwrap();
        // L_new が有効長を超える。
        assert!(c.attention(&t, &x(1, 3), 2, 1.0).is_err());
        // num_heads・scale。
        assert!(c.attention(&t, &x(1, 1), 3, 1.0).is_err());
        assert!(c.attention(&t, &x(1, 1), 2, f32::NAN).is_err());
        // バッチ不一致。
        assert!(c.attention(&t, &x(2, 1), 2, 1.0).is_err());
        // L_new == 0 はデバイスを呼ばず空。
        let empty = Tensor::new(Vec::new(), &[1, 0, 4]).unwrap();
        assert_eq!(c.attention(&t, &empty, 2, 1.0).unwrap().shape(), &[1, 0, 4]);
    }

    #[test]
    fn deferred_token_after_download_poisons() {
        let (t, _) = tape(Mode::AttentionSetsToken);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        c.append(&t, &x(1, 2), &x(1, 2)).unwrap();
        let r = c.attention(&t, &x(1, 1), 2, 1.0);
        assert!(matches!(
            r,
            Err(AutodiffError::Backend(BackendError::StorePoisoned))
        ));
        assert!(c.is_poisoned());
        assert!(matches!(
            c.append(&t, &x(1, 1), &x(1, 1)),
            Err(AutodiffError::Backend(BackendError::StorePoisoned))
        ));
    }

    #[test]
    fn rewind_and_clear_follow_reference_semantics() {
        let (t, log) = tape(Mode::Ok);
        let mut c = PagedKvCache::new(&t, cfg()).unwrap();
        c.append(&t, &x(1, 5), &x(1, 5)).unwrap();
        assert_eq!(c.num_free_blocks(), 0);
        assert!(c.rewind_to(6).is_err());
        assert_eq!(c.seq_len(), 5);
        c.rewind_to(2).unwrap();
        assert_eq!(c.seq_len(), 2);
        assert_eq!(c.num_used_blocks(), 1);
        assert_eq!(c.num_free_blocks(), 2);
        // 巻き戻し後の追記は返却済みブロックを LIFO で再利用する（返却順 2,1 → 先に 1 を取る → slot 2）。
        c.append(&t, &x(1, 1), &x(1, 1)).unwrap();
        assert_eq!(log.lock().unwrap().appends.last().unwrap(), &vec![2u32]);
        c.rewind_to(0).unwrap();
        assert_eq!(c.batch(), None);
        c.clear();
        assert_eq!(c.num_free_blocks(), 3);
        assert_eq!(c.seq_len(), 0);
    }

    #[test]
    fn config_with_u32_overflowing_rows_is_rejected() {
        let (t, _) = tape(Mode::Ok);
        let cfg = PagedKvCacheConfig::new(256, u32::MAX as usize / 2, 1);
        // 設定自体は通っても、スロット番号が u32 に収まらない構成は new が拒否する。
        if let Ok(cfg) = cfg {
            assert!(PagedKvCache::new(&t, cfg).is_err());
        }
    }
}
