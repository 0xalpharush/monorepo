//! Crash/restart campaigns that run each aggregation validator as a simulated host.
//!
//! Validators crash at random times (one at a time or the whole cluster at once) and from
//! within their journal writes and syncs, with torn unsynchronized writes, slow disks, pauses,
//! and skewed clocks. Each validator keeps its reporter across restarts, which checks every
//! certificate and that certificates never conflict; once faults stop, every validator must keep
//! certifying new heights.

use super::*;
use commonware_runtime::deterministic::{
    ClockOffset, FaultConfig, HostConfig, Hosts, PartialWriteMode, WriteConfig,
};
use commonware_utils::Probability;
use rand::{SeedableRng as _, rngs::StdRng};

#[allow(clippy::too_many_arguments)]
fn start_engine<S: Scheme<Sha256Digest, PublicKey = PublicKey>>(
    context: Context,
    fixture: &Fixture<S>,
    idx: usize,
    epoch: Epoch,
    oracle: &Oracle<PublicKey, deterministic::Context>,
    reporter: mocks::ReporterMailbox<S, Sha256Digest>,
    registration: (Sender<PublicKey, deterministic::Context>, Receiver<PublicKey>),
) {
    let participant = &fixture.participants[idx];
    let provider = mocks::Provider::new();
    assert!(provider.register(epoch, fixture.schemes[idx].clone()));
    let engine = Engine::new(
        context.child("engine"),
        Config {
            monitor: mocks::Monitor::new(epoch),
            provider,
            automaton: mocks::Application::new(mocks::Strategy::Correct),
            reporter,
            blocker: oracle.control(participant.clone()),
            priority_acks: false,
            rebroadcast_timeout: NonZeroDuration::new_panic(Duration::from_millis(100)),
            epoch_bounds: (EpochDelta::new(1), EpochDelta::new(1)),
            window: std::num::NonZeroU64::new(10).unwrap(),
            activity_timeout: HeightDelta::new(1_024),
            journal_partition: "aggregation".into(),
            journal_write_buffer: NZUsize!(4096),
            journal_replay_buffer: NZUsize!(4096),
            journal_heights_per_section: std::num::NonZeroU64::new(6).unwrap(),
            journal_compression: Some(3),
            journal_page_cache: CacheRef::from_pooler(&context, PAGE_SIZE, PAGE_CACHE_SIZE),
            strategy: Sequential,
        },
    );
    engine.start(registration);
}

async fn tips<S: Scheme<Sha256Digest, PublicKey = PublicKey>>(
    reporters: &mut [mocks::ReporterMailbox<S, Sha256Digest>],
) -> Vec<Height> {
    let mut tips = Vec::new();
    for reporter in reporters.iter_mut() {
        tips.push(reporter.get_tip().await.map_or(Height::zero(), |(h, _)| h));
    }
    tips
}

fn host_campaign<S, F>(
    seed: u64,
    rounds: usize,
    crash_rate: Probability,
    retention_rate: Probability,
    mode: PartialWriteMode,
    fixture: F,
) -> String
where
    S: Scheme<Sha256Digest, PublicKey = PublicKey>,
    F: FnOnce(&mut deterministic::Context, &[u8], u32) -> Fixture<S>,
{
    let cfg = deterministic::Config::new()
        .with_seed(seed)
        .with_timeout(Some(Duration::from_secs(3_600)))
        .with_storage_fault_config(
            FaultConfig::default()
                .write(WriteConfig {
                    failure_rate: probability!(0.0),
                    retention_rate,
                    mode,
                })
                .latency(Duration::ZERO..Duration::from_millis(3)),
        );
    deterministic::Runner::new(cfg).start(|mut context| async move {
        let n = 4;
        let fixture = fixture(&mut context, TEST_NAMESPACE, n);
        let epoch = Epoch::new(111);
        let (oracle, mut registrations) =
            initialize_simulation(context.child("simulation"), &fixture, RELIABLE_LINK).await;
        let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        let mut hosts = Hosts::new(&context);
        let mut reporters = Vec::new();
        let mut contexts = Vec::new();
        for (idx, participant) in fixture.participants.iter().enumerate() {
            let (reporter, mailbox) =
                mocks::Reporter::new(context.child("reporter"), fixture.verifier.clone());
            reporter.start();
            reporters.push(mailbox.clone());
            let host = hosts.start(&names[idx], HostConfig::new());
            contexts.push(host.child("probe"));
            start_engine(
                host,
                &fixture,
                idx,
                epoch,
                &oracle,
                mailbox,
                registrations.remove(participant).unwrap(),
            );
        }
        context.sleep(Duration::from_secs(2)).await;
        context.storage_fault_config().write().crash_rate = Some(crash_rate);

        let mut rng = StdRng::seed_from_u64(seed);
        macro_rules! restart {
            ($idx:expr) => {{
                let idx = $idx;
                let host = hosts.restart(&names[idx]);
                contexts[idx] = host.child("probe");
                let registration = oracle
                    .control(fixture.participants[idx].clone())
                    .register(0, TEST_QUOTA)
                    .await
                    .unwrap();
                start_engine(
                    host,
                    &fixture,
                    idx,
                    epoch,
                    &oracle,
                    reporters[idx].clone(),
                    registration,
                );
            }};
        }
        for _ in 0..rounds {
            context
                .sleep(Duration::from_millis(rng.random_range(100..2_000)))
                .await;
            let idx = rng.random_range(0..names.len());
            match rng.random_range(0..5u8) {
                0 => hosts.crash_all(),
                1 | 2 => hosts.crash(&names[idx]),
                3 => {
                    if let Some(process) = hosts.get(&names[idx]) {
                        process.pause();
                        context
                            .sleep(Duration::from_millis(rng.random_range(100..3_000)))
                            .await;
                        if let Some(process) = hosts.get(&names[idx]) {
                            process.resume();
                        }
                    }
                }
                _ => {
                    if let Some(process) = hosts.get(&names[idx]) {
                        process.set_clock_drift(rng.random_range(-50_000..50_000));
                        process.set_clock_offset(ClockOffset::Behind(Duration::from_millis(
                            rng.random_range(0..2_000),
                        )));
                        let slow = Duration::from_millis(rng.random_range(1..30));
                        process.set_storage_latency(Some(slow / 2..slow));
                    }
                }
            }
            for idx in 0..names.len() {
                if !hosts.is_running(&names[idx]) {
                    context
                        .sleep(Duration::from_millis(rng.random_range(0..3_000)))
                        .await;
                    restart!(idx);
                }
            }
        }

        // Stop faults and check every validator keeps certifying
        context.storage_fault_config().write().crash_rate = None;
        for idx in 0..names.len() {
            if !hosts.is_running(&names[idx]) {
                restart!(idx);
            }
        }
        let before = tips(&mut reporters).await;
        let target = before.iter().max().unwrap().saturating_add(HeightDelta::new(20));
        let deadline = context.current() + Duration::from_secs(300);
        loop {
            let now = tips(&mut reporters).await;
            if now.iter().all(|tip| *tip >= target) {
                break;
            }
            assert!(
                context.current() < deadline,
                "no liveness after recovery: before {before:?}, now {now:?}, target {target}, engine tips {:?}, running {:?}, starts {:?}",
                contexts
                    .iter()
                    .map(|context| context
                        .encode()
                        .lines()
                        .find(|line| line.starts_with("engine_tip "))
                        .map(str::to_string))
                    .collect::<Vec<_>>(),
                hosts.running(),
                names.iter().map(|name| hosts.starts(name)).collect::<Vec<_>>()
            );
            context.sleep(Duration::from_millis(100)).await;
        }
        context.auditor().state()
    })
}

#[test_traced("WARN")]
fn test_host_campaign_smoke() {
    let run = |seed| {
        host_campaign(
            seed,
            4,
            Probability::new(1, 500).unwrap(),
            probability!(0.5),
            PartialWriteMode::Prefix,
            ed25519::fixture,
        )
    };
    assert_eq!(run(0), run(0), "runs are deterministic");
}

/// Seed sweep; `AGG_DST_SEEDS=a..b`, `AGG_DST_MODE=subset`.
#[test_group("slow")]
#[test_traced("WARN")]
fn test_host_campaign_sweep() {
    let seeds = std::env::var("AGG_DST_SEEDS").unwrap_or_else(|_| "0..8".into());
    let (a, b) = seeds.split_once("..").unwrap();
    let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
    let mode = match std::env::var("AGG_DST_MODE").as_deref() {
        Ok("subset") => PartialWriteMode::Subset,
        _ => PartialWriteMode::Prefix,
    };
    for seed in a..b {
        eprintln!("aggregation host campaign seed {seed}");
        host_campaign(
            seed,
            10,
            Probability::new(1 + seed % 5, 500).unwrap(),
            Probability::new(seed % 101, 100).unwrap(),
            mode,
            ed25519::fixture,
        );
    }
}

/// A crash between journaling certificates and syncing the tip they advance can leave a
/// validator's journal holding certificates for every height of its window above its last
/// journaled tip. Replay restores the tip from the last `Tip` record and puts the certified
/// heights in `confirmed`, but nothing advances the tip over them: the window is full (no new
/// digest is requested) and no further certificate arrives for those heights. When a quorum of
/// validators recovers into this state (as after a whole-cluster power loss), every validator
/// that could advance them stays below, and the cluster stops certifying.
///
/// Builds that journal state directly (certificates for heights `0..window`, no `Tip` record)
/// for every validator, then starts the engines and expects progress.
#[test_traced("WARN")]
#[ignore = "finding: replay does not advance the tip over journaled certificates"]
fn test_replay_advances_tip_over_journaled_certificates() {
    use crate::aggregation::types::{Ack, Activity, Certificate, Item};
    use commonware_storage::journal::segmented::variable::{Config as JConfig, Journal};
    use commonware_utils::iter::NonEmpty;

    let window = 10u64;
    deterministic::Runner::timed(Duration::from_secs(120)).start(|mut context| async move {
        let fixture = ed25519::fixture(&mut context, TEST_NAMESPACE, 4);
        let epoch = Epoch::new(111);
        let certificates: Vec<_> = (0..window)
            .map(|height| {
                let item = Item {
                    height: Height::new(height),
                    digest: <commonware_cryptography::Sha256 as commonware_cryptography::Hasher>::hash(&[format!("data for height {height}").as_bytes()]),
                };
                let acks: Vec<_> = fixture
                    .schemes
                    .iter()
                    .map(|scheme| Ack::sign(scheme, epoch, item.clone()).unwrap())
                    .collect();
                Certificate::from_acks(
                    &fixture.schemes[0],
                    NonEmpty::new(&acks[0], acks[1..].iter()),
                    &Sequential,
                )
                .unwrap()
            })
            .collect();

        let names: Vec<String> = (0..4).map(|i| format!("v{i}")).collect();
        let mut hosts = Hosts::new(&context);
        for name in &names {
            let host = hosts.start(name, HostConfig::new());
            let mut journal = Journal::init(
                host.child("journal"),
                JConfig {
                    partition: "aggregation".into(),
                    compression: Some(3),
                    codec_config: <ed25519::Scheme as commonware_cryptography::certificate::Verifier>::certificate_codec_config_unbounded(),
                    page_cache: CacheRef::from_pooler(&host, PAGE_SIZE, PAGE_CACHE_SIZE),
                    write_buffer: NZUsize!(4096),
                },
            )
            .await
            .unwrap();
            for certificate in &certificates {
                let section = certificate.item.height.get() / 6;
                (journal, _, _) = journal
                    .append(section, &Activity::<ed25519::Scheme, Sha256Digest>::Certified(certificate.clone()))
                    .await
                    .unwrap();
            }
            drop(journal.sync_all().await.unwrap());
            hosts.crash(name);
        }

        let (oracle, mut registrations) =
            initialize_simulation(context.child("simulation"), &fixture, RELIABLE_LINK).await;
        let mut reporters = Vec::new();
        for (idx, participant) in fixture.participants.iter().enumerate() {
            let (reporter, mailbox) =
                mocks::Reporter::new(context.child("reporter"), fixture.verifier.clone());
            reporter.start();
            reporters.push(mailbox.clone());
            let host = hosts.restart(&names[idx]);
            start_engine(
                host,
                &fixture,
                idx,
                epoch,
                &oracle,
                mailbox,
                registrations.remove(participant).unwrap(),
            );
        }

        let target = Height::new(window + 10);
        let deadline = context.current() + Duration::from_secs(60);
        loop {
            let now = tips(&mut reporters).await;
            if now.iter().all(|tip| *tip >= target) {
                break;
            }
            assert!(
                context.current() < deadline,
                "no progress past the journaled certificates: tips {now:?}, target {target}"
            );
            context.sleep(Duration::from_millis(100)).await;
        }
    });
}

/// Campaign seed 18 (sweep parameters): after a validator's tip is fast-forwarded to the safe
/// tip of its peers onto a height it had already certified, nothing advances it further, and once
/// a quorum is stuck that way the cluster stops certifying.
#[test_traced("WARN")]
#[ignore = "finding: fast-forwarding the tip onto a certified height stalls it"]
fn test_host_campaign_seed_18_fast_forward_onto_certified_height() {
    host_campaign(
        18,
        10,
        Probability::new(4, 500).unwrap(),
        Probability::new(18, 100).unwrap(),
        PartialWriteMode::Prefix,
        ed25519::fixture,
    );
}
