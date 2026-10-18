//! `RocmBackendOps` のデバイス常駐 KV 2 メソッド（`paged_kv_append`・`paged_attention_forward`）の結線
//! （イシュー #3443・分解案 R12。設計 `docs/device-resident-kv-paged-attention-design.md`）。
//!
//! 役割: `ops.rs` の `BackendOps` impl から 1 行で呼ばれ、(1) `DeviceBuffer` のハンドルを
//! `RocmBufferHandle` へ downcast（失敗は `DeviceMismatch`）、(2) tensor-core 共通検証
//! （`validate_paged_kv_append`／`validate_paged_attention`。範囲・重複スロット・番兵・`i32` 収容）、
//! (3) ハンドル長の再検査、(4) 空入力の早期リターン（HIP に触れない）、(5) engine（Hip／テストの
//! HostModel）の分岐、(6) 出力 `DeviceBuffer` の組み立て、を担う。カーネル起動そのものは `paged_kv.rs` の
//! `RocmPagedKv`。
//!
//! 検証は pool を書き換える前にすべて終える（検証違反で部分書き込みを残さない）。ROCm 非搭載は
//! `DeviceUnavailable` で返し、`Unsupported` へは落とさない（autodiff の暗黙フォールバックを作らない。
//! security.md A08）。`_tracked` 版は trait 既定の委譲のまま上書きしない。
//!
//! # 到達性
//! `memory_ops()` は R0 の方針（既定継承の `None`）のままなので、crate の外から ROCm の `DeviceBuffer` を
//! 作れない。判別は `downcast_handle::<RocmBufferHandle>()` だけで行う（`device()` は #3157 まで仮置きの
//! `Device::Cpu` で、CPU 由来のバッファと区別できない）。到達性は #3157 待ち。
//!
//! 結線を子モジュールに置くのは、兄弟 issue の `ops.rs` 変更との衝突面を小さくするため。thread_local の
//! キャッシュもここで宣言し、親の宣言群には触れない。

use std::cell::RefCell;

use fandhe_ai_tensor_core::{
    BackendError, BackendOps, DeviceBuffer, PagedAttentionArgs, PagedKvLayout, ShapeError,
    validate_paged_attention, validate_paged_kv_append,
};

use super::{Cache, Engine, RocmBackendOps, cached};
use crate::buffer_handle::RocmBufferHandle;
use crate::error::to_backend_error;
use crate::paged_kv::RocmPagedKv;

thread_local! {
    /// paged KV 2 カーネル（R12・#3443）。
    static PAGED_KV_CACHE: Cache<RocmPagedKv> = const { RefCell::new(Vec::new()) };
}

fn handle_of(buf: &DeviceBuffer<f32>) -> Result<&RocmBufferHandle, BackendError> {
    buf.downcast_handle::<RocmBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)
}

/// ハンドルの実要素数が `DeviceBuffer` の shape 由来の要素数と一致することを確かめる。
fn check_handle_len(h: &RocmBufferHandle, numel: usize) -> Result<(), BackendError> {
    if h.len() != numel {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: numel,
                actual: h.len(),
            },
        ));
    }
    Ok(())
}

impl RocmBackendOps {
    /// `paged_kv_append` の本体。順序は (1) 読み取り 2 本の downcast → (2) 共通検証 → (3) ハンドル長 →
    /// (4) 行数 0 の早期リターン → (5) pool の可変 downcast と engine 分岐。
    pub(super) fn paged_kv_append_impl(
        &self,
        k_new: &DeviceBuffer<f32>,
        v_new: &DeviceBuffer<f32>,
        k_pool: &mut DeviceBuffer<f32>,
        v_pool: &mut DeviceBuffer<f32>,
        slot_mapping: &[u32],
        layout: &PagedKvLayout,
    ) -> Result<(), BackendError> {
        handle_of(k_new)?;
        handle_of(v_new)?;
        handle_of(k_pool)?;
        handle_of(v_pool)?;
        let rows = validate_paged_kv_append(
            k_new.numel(),
            v_new.numel(),
            k_pool.numel(),
            v_pool.numel(),
            slot_mapping,
            layout,
        )?;
        check_handle_len(handle_of(k_new)?, k_new.numel())?;
        check_handle_len(handle_of(v_new)?, v_new.numel())?;
        check_handle_len(handle_of(k_pool)?, k_pool.numel())?;
        check_handle_len(handle_of(v_pool)?, v_pool.numel())?;
        if rows == 0 {
            return Ok(());
        }
        let kn = handle_of(k_new)?;
        let vn = handle_of(v_new)?;
        let kp = k_pool
            .downcast_handle_mut::<RocmBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        let vp = v_pool
            .downcast_handle_mut::<RocmBufferHandle>()
            .ok_or(BackendError::DeviceMismatch)?;
        match (self.engine, kn, vn, kp, vp) {
            (
                Engine::Hip,
                RocmBufferHandle::Hip(a),
                RocmBufferHandle::Hip(b),
                RocmBufferHandle::Hip(c),
                RocmBufferHandle::Hip(d),
            ) => {
                let pk = cached(&PAGED_KV_CACHE, self.target, RocmPagedKv::new)
                    .map_err(to_backend_error)?;
                pk.append_resident(a, b, c, d, slot_mapping, layout)
                    .map_err(to_backend_error)
            }
            #[cfg(test)]
            (
                Engine::HostModel,
                RocmBufferHandle::Host(a),
                RocmBufferHandle::Host(b),
                RocmBufferHandle::Host(c),
                RocmBufferHandle::Host(d),
            ) => {
                let (rows_i, e_i, pool_rows, blocks) = crate::paged_kv::append_launch_params(
                    a.len(),
                    b.len(),
                    c.len(),
                    d.len(),
                    rows,
                    layout,
                )
                .map_err(to_backend_error)?;
                crate::paged_kv_model::append(
                    a,
                    b,
                    c,
                    d,
                    slot_mapping,
                    rows_i,
                    e_i,
                    pool_rows,
                    blocks,
                );
                Ok(())
            }
            // engine とハンドル種別の取り違え（Hip engine に Host ハンドル、Empty 混在等）は fail-closed。
            _ => Err(BackendError::DeviceMismatch),
        }
    }

    /// `paged_attention_forward` の本体。出力は `[B, L_new, E]` の新規 `DeviceBuffer`。
    pub(super) fn paged_attention_impl(
        &self,
        q: &DeviceBuffer<f32>,
        k_pool: &DeviceBuffer<f32>,
        v_pool: &DeviceBuffer<f32>,
        block_tables: &[u32],
        seq_lens: &[u32],
        args: &PagedAttentionArgs,
    ) -> Result<DeviceBuffer<f32>, BackendError> {
        let qh = handle_of(q)?;
        let kh = handle_of(k_pool)?;
        let vh = handle_of(v_pool)?;
        let plan = validate_paged_attention(
            q.numel(),
            k_pool.numel(),
            v_pool.numel(),
            block_tables,
            seq_lens,
            args,
        )?;
        check_handle_len(qh, q.numel())?;
        check_handle_len(kh, k_pool.numel())?;
        check_handle_len(vh, v_pool.numel())?;
        let shape = vec![args.batch, args.l_new, args.layout.embed_dim];
        if plan.out_numel == 0 {
            // `E = H · Dh` なので `query_rows == 0` と同値。HIP に触れず空バッファを返す。
            return Ok(DeviceBuffer::new(
                self.device(),
                shape,
                Box::new(RocmBufferHandle::Empty),
            ));
        }
        let out = match (self.engine, qh, kh, vh) {
            (
                Engine::Hip,
                RocmBufferHandle::Hip(a),
                RocmBufferHandle::Hip(b),
                RocmBufferHandle::Hip(c),
            ) => {
                let pk = cached(&PAGED_KV_CACHE, self.target, RocmPagedKv::new)
                    .map_err(to_backend_error)?;
                RocmBufferHandle::from(
                    pk.attention_resident(a, b, c, block_tables, seq_lens, args, &plan)
                        .map_err(to_backend_error)?,
                )
            }
            #[cfg(test)]
            (
                Engine::HostModel,
                RocmBufferHandle::Host(a),
                RocmBufferHandle::Host(b),
                RocmBufferHandle::Host(c),
            ) => {
                let p = crate::paged_kv::attention_launch_params(args, &plan)
                    .map_err(to_backend_error)?;
                crate::paged_kv::check_attention_lens(
                    a.len(),
                    b.len(),
                    c.len(),
                    block_tables.len(),
                    seq_lens.len(),
                    args,
                    &plan,
                )
                .map_err(to_backend_error)?;
                RocmBufferHandle::Host(crate::paged_kv_model::attention(
                    a,
                    b,
                    c,
                    block_tables,
                    seq_lens,
                    &p,
                ))
            }
            _ => return Err(BackendError::DeviceMismatch),
        };
        Ok(DeviceBuffer::new(self.device(), shape, Box::new(out)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::GfxTarget;
    use fandhe_ai_backend_cpu::{CpuBackendOps, assert_parity};
    use fandhe_ai_tensor_core::Tensor;

    fn rocm() -> RocmBackendOps {
        RocmBackendOps::with_host_model(GfxTarget::Gfx942)
    }

    /// 決定的な擬似乱数（xorshift）。値域は約 [-1, 1)。
    fn data(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn tensor(shape: &[usize], seed: u64) -> Tensor<f32> {
        Tensor::new(data(shape.iter().product(), seed), shape).unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// REQ-2 の統一複合判定（許容誤差は不変）に加え、bit 一致も確認する。ホストモデルは CPU 参照と
    /// 演算順が同一なので bit 一致するはず（モデルの逐語性の確認。GPU の `expf` の bit 一致を
    /// 主張するものではない）。
    fn check(name: &str, got: &[f32], want: &[f32]) {
        assert_parity(name, got, want);
        assert_eq!(bits(got), bits(want), "{name}: bit 一致");
    }

    struct Case {
        block_size: usize,
        batch: usize,
        embed: usize,
        heads: usize,
        /// 各ステップの追記トークン数（先頭が prefill）。
        steps: &'static [usize],
    }

    /// ROCm 側（HostModel）と CPU 側の両方へ同じ append → attention を流し、pool 全体と出力を照合する。
    /// ブロック ID は番号をずらした非連続割り当てで、未使用枠は番兵 `u32::MAX`。
    fn run_case(c: &Case) {
        let max_tokens: usize = c.steps.iter().sum();
        let max_blocks = max_tokens.div_ceil(c.block_size);
        let num_blocks = c.batch * max_blocks + 3;
        let layout = PagedKvLayout::new(num_blocks, c.block_size, c.embed).unwrap();
        // (b, 枠) -> ブロック ID。素数ストライドで散らす（num_blocks と互いに素になる値を選ぶ）。
        let stride = (1..num_blocks)
            .rev()
            .find(|s| gcd(*s, num_blocks) == 1)
            .unwrap_or(1);
        let id_of = |b: usize, slot: usize| ((b * max_blocks + slot) * stride % num_blocks) as u32;

        let cpu = CpuBackendOps::new();
        let mem = cpu.memory_ops().unwrap();
        let ro = rocm();
        let pool_shape = [num_blocks, c.block_size, c.embed];
        let mut cpu_k = mem.alloc_zeroed(&pool_shape).unwrap();
        let mut cpu_v = mem.alloc_zeroed(&pool_shape).unwrap();
        let zeros = Tensor::new(vec![0.0f32; layout.pool_elems().unwrap()], &pool_shape).unwrap();
        let mut ro_k = ro.upload_resident(&zeros).unwrap();
        let mut ro_v = ro.upload_resident(&zeros).unwrap();

        let scale = 1.0 / ((c.embed / c.heads) as f32).sqrt();
        let mut seq = 0usize;
        for (si, &l) in c.steps.iter().enumerate() {
            let seed = 10 + si as u64;
            let k = tensor(&[c.batch, l, c.embed], seed);
            let v = tensor(&[c.batch, l, c.embed], seed + 1000);
            let q = tensor(&[c.batch, l, c.embed], seed + 2000);
            // スロット写像（行は [b, i] の順）とブロックテーブル。
            let mut slots = Vec::new();
            for b in 0..c.batch {
                for i in 0..l {
                    let pos = seq + i;
                    slots.push(
                        id_of(b, pos / c.block_size) * c.block_size as u32
                            + (pos % c.block_size) as u32,
                    );
                }
            }
            seq += l;
            let used = seq.div_ceil(c.block_size);
            let mut tables = Vec::new();
            for b in 0..c.batch {
                for slot in 0..max_blocks {
                    tables.push(if slot < used {
                        id_of(b, slot)
                    } else {
                        u32::MAX
                    });
                }
            }
            let seq_lens = vec![seq as u32; c.batch];
            let args =
                PagedAttentionArgs::new(layout, c.batch, l, c.heads, max_blocks, scale).unwrap();

            let (ck, cv, cq) = (
                mem.upload(&k).unwrap(),
                mem.upload(&v).unwrap(),
                mem.upload(&q).unwrap(),
            );
            cpu.paged_kv_append(&ck, &cv, &mut cpu_k, &mut cpu_v, &slots, &layout)
                .unwrap();
            let want = cpu
                .paged_attention_forward(&cq, &cpu_k, &cpu_v, &tables, &seq_lens, &args)
                .unwrap();

            let (rk, rv, rq) = (
                ro.upload_resident(&k).unwrap(),
                ro.upload_resident(&v).unwrap(),
                ro.upload_resident(&q).unwrap(),
            );
            ro.paged_kv_append(&rk, &rv, &mut ro_k, &mut ro_v, &slots, &layout)
                .unwrap();
            let got = ro
                .paged_attention_forward(&rq, &ro_k, &ro_v, &tables, &seq_lens, &args)
                .unwrap();

            let name = format!(
                "bs={} B={} E={} H={} step={si} L_new={l}",
                c.block_size, c.batch, c.embed, c.heads
            );
            // append: pool 全体が bit 一致（コピーのみ）。
            let flat = |t: Tensor<f32>| t.contiguous().as_slice().unwrap().to_vec();
            assert_eq!(
                bits(&flat(ro.download_resident(&ro_k).unwrap())),
                bits(&flat(mem.download(&cpu_k).unwrap())),
                "{name}: K pool"
            );
            assert_eq!(
                bits(&flat(ro.download_resident(&ro_v).unwrap())),
                bits(&flat(mem.download(&cpu_v).unwrap())),
                "{name}: V pool"
            );
            assert_eq!(got.shape(), want.shape(), "{name}");
            check(
                &name,
                &flat(ro.download_resident(&got).unwrap()),
                &flat(mem.download(&want).unwrap()),
            );
        }
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    #[test]
    fn matches_cpu_across_block_boundaries() {
        // ブロックサイズ 1/4/16・複数ブロックにまたがる長さ・prefill 後の L_new = 1 の連続。
        for &(bs, steps) in &[
            (1usize, &[3usize, 1, 1][..]),
            (4, &[3, 1, 1, 1, 1, 5]),
            (4, &[4, 1]),
            (4, &[5, 1]),
            (16, &[15, 1, 1, 17]),
        ] {
            for &(batch, embed, heads) in &[(1usize, 8usize, 2usize), (2, 12, 3), (2, 16, 4)] {
                run_case(&Case {
                    block_size: bs,
                    batch,
                    embed,
                    heads,
                    steps,
                });
            }
        }
    }

    fn resident(o: &RocmBackendOps, shape: &[usize], seed: u64) -> DeviceBuffer<f32> {
        o.upload_resident(&tensor(shape, seed)).unwrap()
    }

    fn cpu_buf(shape: &[usize], seed: u64) -> DeviceBuffer<f32> {
        let cpu = CpuBackendOps::new();
        cpu.memory_ops()
            .unwrap()
            .upload(&tensor(shape, seed))
            .unwrap()
    }

    #[test]
    fn append_rejects_violations_without_touching_pool() {
        let o = rocm();
        let layout = PagedKvLayout::new(2, 2, 4).unwrap();
        let k = resident(&o, &[1, 2, 4], 1);
        let v = resident(&o, &[1, 2, 4], 2);
        let mut kp = resident(&o, &[2, 2, 4], 3);
        let mut vp = resident(&o, &[2, 2, 4], 4);
        let before = o.download_resident(&kp).unwrap();
        for bad in [&[0u32, 4][..], &[0, u32::MAX], &[1, 1], &[0]] {
            let r = o.paged_kv_append(&k, &v, &mut kp, &mut vp, bad, &layout);
            assert!(
                matches!(r, Err(BackendError::InvalidArgument(_))),
                "{bad:?}"
            );
        }
        assert_eq!(
            bits(before.as_slice().unwrap()),
            bits(o.download_resident(&kp).unwrap().as_slice().unwrap())
        );
        // 正しい入力は通る。
        o.paged_kv_append(&k, &v, &mut kp, &mut vp, &[0, 3], &layout)
            .unwrap();
        // pool の shape がレイアウトと一致しない。
        let mut small = resident(&o, &[1, 2, 4], 5);
        assert!(
            o.paged_kv_append(&k, &v, &mut small, &mut vp, &[0, 3], &layout)
                .is_err()
        );
    }

    #[test]
    fn attention_rejects_bad_tables_and_sizes() {
        let o = rocm();
        let layout = PagedKvLayout::new(2, 2, 4).unwrap();
        let args = PagedAttentionArgs::new(layout, 1, 1, 2, 2, 0.5).unwrap();
        let q = resident(&o, &[1, 1, 4], 1);
        let kp = resident(&o, &[2, 2, 4], 2);
        let vp = resident(&o, &[2, 2, 4], 3);
        assert!(
            o.paged_attention_forward(&q, &kp, &vp, &[0, 1], &[3], &args)
                .is_ok()
        );
        for (tables, lens) in [
            (&[0u32, 2][..], &[3u32][..]),
            (&[0, u32::MAX], &[3]),
            (&[0, 1], &[5]),
            (&[0, 1], &[0]),
            (&[0], &[3]),
            (&[0, 1], &[3, 3]),
        ] {
            let r = o.paged_attention_forward(&q, &kp, &vp, tables, lens, &args);
            assert!(
                matches!(r, Err(BackendError::InvalidArgument(_))),
                "{tables:?} {lens:?}"
            );
        }
    }

    #[test]
    fn cpu_origin_buffers_are_device_mismatch() {
        let o = rocm();
        let layout = PagedKvLayout::new(2, 2, 4).unwrap();
        let args = PagedAttentionArgs::new(layout, 1, 1, 2, 2, 0.5).unwrap();
        let (rq, rkp, rvp) = (
            resident(&o, &[1, 1, 4], 1),
            resident(&o, &[2, 2, 4], 2),
            resident(&o, &[2, 2, 4], 3),
        );
        let (cq, ckp, cvp) = (
            cpu_buf(&[1, 1, 4], 1),
            cpu_buf(&[2, 2, 4], 2),
            cpu_buf(&[2, 2, 4], 3),
        );
        for (q, k, v) in [(&cq, &rkp, &rvp), (&rq, &ckp, &rvp), (&rq, &rkp, &cvp)] {
            assert!(matches!(
                o.paged_attention_forward(q, k, v, &[0, 1], &[3], &args),
                Err(BackendError::DeviceMismatch)
            ));
        }
        // append: 4 引数のどれが CPU 由来でも DeviceMismatch（pool は書き換わらない）。
        let (rk, rv) = (resident(&o, &[1, 2, 4], 4), resident(&o, &[1, 2, 4], 5));
        let (ck, cv) = (cpu_buf(&[1, 2, 4], 4), cpu_buf(&[1, 2, 4], 5));
        let mut kp = resident(&o, &[2, 2, 4], 6);
        let mut vp = resident(&o, &[2, 2, 4], 7);
        let mut ckp = cpu_buf(&[2, 2, 4], 6);
        let mut cvp = cpu_buf(&[2, 2, 4], 7);
        assert!(matches!(
            o.paged_kv_append(&ck, &rv, &mut kp, &mut vp, &[0, 3], &layout),
            Err(BackendError::DeviceMismatch)
        ));
        assert!(matches!(
            o.paged_kv_append(&rk, &cv, &mut kp, &mut vp, &[0, 3], &layout),
            Err(BackendError::DeviceMismatch)
        ));
        assert!(matches!(
            o.paged_kv_append(&rk, &rv, &mut ckp, &mut vp, &[0, 3], &layout),
            Err(BackendError::DeviceMismatch)
        ));
        assert!(matches!(
            o.paged_kv_append(&rk, &rv, &mut kp, &mut cvp, &[0, 3], &layout),
            Err(BackendError::DeviceMismatch)
        ));
    }

    #[test]
    fn empty_inputs_succeed_without_hip() {
        let o = rocm();
        let layout = PagedKvLayout::new(2, 2, 4).unwrap();
        // 追記 0 行。
        let k = resident(&o, &[1, 0, 4], 1);
        let v = resident(&o, &[1, 0, 4], 2);
        let mut kp = resident(&o, &[2, 2, 4], 3);
        let mut vp = resident(&o, &[2, 2, 4], 4);
        o.paged_kv_append(&k, &v, &mut kp, &mut vp, &[], &layout)
            .unwrap();
        // batch = 0 の attention は空の Empty バッファ（shape [0, L_new, E]）。
        let args = PagedAttentionArgs::new(layout, 0, 1, 2, 2, 0.5).unwrap();
        let q = resident(&o, &[0, 1, 4], 5);
        let out = o
            .paged_attention_forward(&q, &kp, &vp, &[], &[], &args)
            .unwrap();
        assert_eq!(out.shape(), &[0, 1, 4]);
        assert!(matches!(handle_of(&out).unwrap(), RocmBufferHandle::Empty));
    }

    /// ROCm 非搭載の Hip engine では、Host ハンドルは engine と合わず `DeviceMismatch`（fail-closed）。
    /// HIP 非搭載のとき Hip ハンドルを作る経路がないため、取り違えの拒否で確認する。
    #[test]
    fn hip_engine_rejects_host_model_handles() {
        let host = rocm();
        let hip = RocmBackendOps::new(GfxTarget::Gfx942);
        let layout = PagedKvLayout::new(2, 2, 4).unwrap();
        let args = PagedAttentionArgs::new(layout, 1, 1, 2, 2, 0.5).unwrap();
        let q = resident(&host, &[1, 1, 4], 1);
        let kp = resident(&host, &[2, 2, 4], 2);
        let vp = resident(&host, &[2, 2, 4], 3);
        assert!(matches!(
            hip.paged_attention_forward(&q, &kp, &vp, &[0, 1], &[3], &args),
            Err(BackendError::DeviceMismatch)
        ));
    }
}
