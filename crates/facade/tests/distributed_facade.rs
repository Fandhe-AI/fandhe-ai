//! イシュー #3147（親 #3143）: `fandhe_ai::distributed`（分散学習の公開面）の受け入れ条件を
//! 公開 API だけで検証する統合テスト。
//!
//! 検証する契約（`docs/facade-distributed-host-collective-decision.md` §16）:
//! - 設定の構築時に world_size・rank・アドレス・タイムアウトを検証し、範囲外は型付きエラーで
//!   拒否する（OWASP A03）。非 loopback は明示 opt-in だけで許す。
//! - world_size 1 は無通信で、既存の `Tape::step_device_param_store*` と bit 一致する。
//! - 2 ランク（同一プロセス内の 2 スレッド・`127.0.0.1:0` 限定）の分散ステップは、全バッチの単一
//!   プロセス更新と統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致し、
//!   2 ランクのパラメータどうしは bit 一致する。
//! - all-reduce が失敗したらパラメータを更新しない（fail-closed）。
//!
//! 実機非依存（CPU・loopback のみ）のため `#[ignore]` 分離はしない。新しい tolerance・baseline は
//! 作らない。

use std::net::SocketAddr;
use std::thread;
use std::time::Duration;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::distributed::{
    Communicator, DistributedConfig, DistributedError, Rendezvous, step_adam, step_adamw, step_sgd,
};
use fandhe_ai::optim::{AdamConfig, AdamWConfig};
use fandhe_ai::{SgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;
const HALF: usize = 2;
const STEPS: usize = 5;

fn loopback0() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap()
}

/// 全バッチ（`2 * HALF` 行）の入力と目標を `Vec` で返す（行優先）。
fn full_data() -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    (
        rng.fill_vec(2 * HALF * D_IN),
        rng.fill_vec(2 * HALF * D_OUT),
    )
}

/// ランク `r`（0／1）が受け持つ半分のバッチ。
fn half_data(r: usize) -> (Vec<f32>, Vec<f32>) {
    let (x, y) = full_data();
    (
        x[r * HALF * D_IN..(r + 1) * HALF * D_IN].to_vec(),
        y[r * HALF * D_OUT..(r + 1) * HALF * D_OUT].to_vec(),
    )
}

#[derive(Clone, Copy)]
enum Opt {
    Sgd,
    Adam,
    AdamW,
}

fn flatten(params: &[Tensor<f32>]) -> Vec<f32> {
    params
        .iter()
        .flat_map(|p| p.contiguous().as_slice().unwrap().to_vec())
        .collect()
}

/// `steps` 回の学習。`comm` が `None` なら既存の単一プロセス入口、`Some` なら分散ステップ。
/// 最終パラメータ（ホスト）を返す。
fn train(
    opt: Opt,
    x: &[f32],
    y: &[f32],
    rows: usize,
    steps: usize,
    mut comm: Option<&mut Communicator>,
) -> Vec<Tensor<f32>> {
    let model = build_model();
    let x_t = tensor(x.to_vec(), &[rows, D_IN]);
    let y_t = tensor(y.to_vec(), &[rows, D_OUT]);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    for _ in 0..steps {
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x_t);
        let yv = tape.var(&y_t);
        let pred = model.forward_resident(&tape, &xv, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &yv).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        match (opt, comm.as_deref_mut()) {
            (Opt::Sgd, None) => tape
                .step_device_param_store(&mut store, &grads, &SgdConfig::new(0.05))
                .unwrap(),
            (Opt::Adam, None) => tape
                .step_device_param_store_adam(&mut store, &grads, &AdamConfig::default())
                .unwrap(),
            (Opt::AdamW, None) => tape
                .step_device_param_store_adamw(&mut store, &grads, &AdamWConfig::default())
                .unwrap(),
            (Opt::Sgd, Some(c)) => {
                step_sgd(&tape, &mut store, &grads, &SgdConfig::new(0.05), c).unwrap()
            }
            (Opt::Adam, Some(c)) => {
                step_adam(&tape, &mut store, &grads, &AdamConfig::default(), c).unwrap()
            }
            (Opt::AdamW, Some(c)) => {
                step_adamw(&tape, &mut store, &grads, &AdamWConfig::default(), c).unwrap()
            }
        }
    }
    let tape = fandhe_ai::tape();
    tape.sync_device_param_store_to_host(&store).unwrap()
}

/// 2 ランクの communicator を作る（ランク 0 が `bind` → `local_addr` → ランク 1 へ渡す → `establish`）。
fn establish_pair(session: u64) -> (Communicator, Communicator) {
    let cfg = |rank: usize, addr: SocketAddr| {
        DistributedConfig::builder(2, rank, addr)
            .session_id(session)
            .establish_timeout(Duration::from_secs(20))
            .io_timeout(Duration::from_secs(10))
            .op_timeout(Duration::from_secs(20))
            .build()
            .unwrap()
    };
    let rdv0 = Rendezvous::bind(&cfg(0, loopback0())).unwrap();
    let addr = rdv0.local_addr().expect("rank 0 は実アドレスを持つ");
    let cfg1 = cfg(1, addr);
    let t1 = thread::spawn(move || Rendezvous::bind(&cfg1).unwrap().establish().unwrap());
    let c0 = rdv0.establish().unwrap();
    let c1 = t1.join().unwrap();
    assert_eq!((c0.rank(), c0.world_size()), (0, 2));
    assert_eq!((c1.rank(), c1.world_size()), (1, 2));
    (c0, c1)
}

/// 統一複合判定は共通定義 `tensor-core::tolerance::composite_close_f32` へ委譲する（閾値の分散定義を避ける）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    assert!(
        fandhe_ai_tensor_core::tolerance::composite_close_f32(actual, expected),
        "{ctx}: actual={actual} expected={expected}"
    );
}

#[test]
fn invalid_configs_are_rejected_with_typed_errors() {
    let ok: SocketAddr = "127.0.0.1:29500".parse().unwrap();
    let build = |w, r, a: SocketAddr| DistributedConfig::builder(w, r, a).build();

    assert!(matches!(
        build(0, 0, ok),
        Err(DistributedError::WorldSizeOutOfRange { world_size: 0, .. })
    ));
    assert!(matches!(
        build(65, 0, ok),
        Err(DistributedError::WorldSizeOutOfRange { world_size: 65, .. })
    ));
    assert!(matches!(
        build(2, 2, ok),
        Err(DistributedError::RankOutOfRange {
            rank: 2,
            world_size: 2
        })
    ));
    for bad in ["0.0.0.0:29500", "224.0.0.1:29500"] {
        assert!(
            matches!(
                build(2, 0, bad.parse().unwrap()),
                Err(DistributedError::InvalidConfig { .. })
            ),
            "{bad}"
        );
    }
    assert!(matches!(
        DistributedConfig::builder(2, 0, ok)
            .io_timeout(Duration::ZERO)
            .build(),
        Err(DistributedError::InvalidConfig { .. })
    ));
    assert!(matches!(
        DistributedConfig::builder(2, 0, ok)
            .op_timeout(Duration::MAX)
            .build(),
        Err(DistributedError::InvalidConfig { .. })
    ));
    assert!(matches!(
        DistributedConfig::builder(2, 0, ok)
            .establish_timeout(Duration::MAX)
            .build(),
        Err(DistributedError::InvalidConfig { .. })
    ));
    // ランク 1 以降はポート 0 では接続先が定まらない。
    assert!(matches!(
        build(2, 1, loopback0()),
        Err(DistributedError::InvalidConfig { .. })
    ));
}

#[test]
fn non_loopback_requires_explicit_opt_in() {
    let remote: SocketAddr = "192.0.2.1:1234".parse().unwrap();
    assert!(matches!(
        DistributedConfig::builder(2, 1, remote).build(),
        Err(DistributedError::NonLoopbackNotAllowed { .. })
    ));
    // opt-in 時の `build` はソケットを開かない。
    let cfg = DistributedConfig::builder(2, 1, remote)
        .allow_non_loopback(true)
        .build()
        .unwrap();
    assert!(cfg.is_non_loopback_allowed());
    assert_eq!(cfg.rendezvous_addr(), remote);
    assert_eq!((cfg.world_size(), cfg.rank()), (2, 1));
    // 既定は loopback のみ。
    let def = DistributedConfig::builder(1, 0, loopback0())
        .build()
        .unwrap();
    assert!(!def.is_non_loopback_allowed());
}

#[test]
fn init_rejects_rank0_port0_when_world_size_is_two_or_more() {
    let cfg = DistributedConfig::builder(2, 0, loopback0())
        .build()
        .unwrap();
    assert!(matches!(
        Communicator::init(&cfg),
        Err(DistributedError::InvalidConfig { .. })
    ));
}

#[test]
fn world_size_one_init_is_local_and_matches_single_process_bit_for_bit() {
    let cfg = DistributedConfig::builder(1, 0, loopback0())
        .build()
        .unwrap();
    let (x, y) = full_data();
    for opt in [Opt::Sgd, Opt::Adam, Opt::AdamW] {
        let mut comm = Communicator::init(&cfg).unwrap();
        assert_eq!((comm.rank(), comm.world_size()), (0, 1));
        assert!(!comm.is_poisoned());
        let dist = flatten(&train(opt, &x, &y, 2 * HALF, STEPS, Some(&mut comm)));
        let single = flatten(&train(opt, &x, &y, 2 * HALF, STEPS, None));
        assert_eq!(dist.len(), single.len());
        for (i, (d, s)) in dist.iter().zip(&single).enumerate() {
            assert_eq!(d.to_bits(), s.to_bits(), "param element {i}");
        }
    }
}

#[test]
fn two_ranks_match_full_batch_single_process_update() {
    for (n, opt) in [Opt::Sgd, Opt::Adam, Opt::AdamW].into_iter().enumerate() {
        let (mut c0, mut c1) = establish_pair(0xD157_0000 + n as u64);
        let (x0, y0) = half_data(0);
        let (x1, y1) = half_data(1);
        let t1 = thread::spawn(move || flatten(&train(opt, &x1, &y1, HALF, STEPS, Some(&mut c1))));
        let p0 = flatten(&train(opt, &x0, &y0, HALF, STEPS, Some(&mut c0)));
        let p1 = t1.join().unwrap();

        // 2 ランクのパラメータどうしは bit 一致する。
        assert_eq!(p0.len(), p1.len());
        for (i, (a, b)) in p0.iter().zip(&p1).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "rank 間の param element {i}");
        }
        // 全バッチを使った単一プロセス更新と統一複合判定で一致する。
        let (x, y) = full_data();
        let single = flatten(&train(opt, &x, &y, 2 * HALF, STEPS, None));
        assert_eq!(p0.len(), single.len());
        for (i, (d, s)) in p0.iter().zip(&single).enumerate() {
            assert_close(*d, *s, &format!("param element {i}"));
        }
    }
}

#[test]
fn failed_all_reduce_does_not_update_parameters_and_poisons_communicator() {
    let (mut c0, c1) = establish_pair(0xD157_00FF);
    // ランク 1 がステップ前に communicator を閉じる。
    drop(c1);

    let model = build_model();
    let (x, y) = half_data(0);
    let x_t = tensor(x, &[HALF, D_IN]);
    let y_t = tensor(y, &[HALF, D_OUT]);
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let before = flatten(&init_tape.sync_device_param_store_to_host(&store).unwrap());
    drop(init_tape);

    let tape = fandhe_ai::tape();
    let xv = tape.var(&x_t);
    let yv = tape.var(&y_t);
    let pred = model.forward_resident(&tape, &xv, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &yv).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();

    let err = step_sgd(&tape, &mut store, &grads, &SgdConfig::new(0.05), &mut c0).unwrap_err();
    assert!(
        matches!(
            err,
            DistributedError::PeerDisconnected
                | DistributedError::Timeout { .. }
                | DistributedError::Io(_)
        ),
        "想定外のエラー: {err:?}"
    );
    assert!(c0.is_poisoned());
    // poisoned 後の再利用は拒否される。
    let again = step_sgd(&tape, &mut store, &grads, &SgdConfig::new(0.05), &mut c0).unwrap_err();
    assert!(matches!(again, DistributedError::Poisoned), "{again:?}");

    // pending は残るので明示的にクリアしてから、パラメータがステップ前と bit 一致することを確認する。
    assert!(store.abandon_pending_forward());
    let after = flatten(&tape.sync_device_param_store_to_host(&store).unwrap());
    assert_eq!(before.len(), after.len());
    for (i, (b, a)) in before.iter().zip(&after).enumerate() {
        assert_eq!(
            b.to_bits(),
            a.to_bits(),
            "失敗後に param element {i} が変化"
        );
    }
}
