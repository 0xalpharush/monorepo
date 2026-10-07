//! Crash/restart campaigns that run each validator as a simulated host.
//!
//! Every validator runs in its own [deterministic::Hosts] host (own storage namespace, metrics,
//! IP, clock), so a crash affects exactly its storage. Campaigns crash validators at random
//! times (including the whole cluster at once), crash them from within storage writes and syncs
//! with torn unsynchronized writes, pause them, skew their clocks, and slow their disks. No
//! validator ever loses synced data, so no validator may ever equivocate.

use super::*;
use commonware_runtime::deterministic::{FaultConfig, HostConfig, Hosts, PartialWriteMode, WriteConfig};
use commonware_utils::Probability;

/// Faults a campaign injects.
#[derive(Clone, Copy, Debug)]
pub(super) struct Campaign {
    /// Number of fault rounds.
    pub rounds: usize,
    /// Probability that a write or sync crashes its host from within.
    pub crash_rate: Probability,
    /// Probability that a byte of an unsynchronized write survives a crash.
    pub retention_rate: Probability,
    /// How unsynchronized writes tear.
    pub mode: PartialWriteMode,
    /// Upper bound of the storage latency.
    pub latency: Duration,
    /// Upper bound of the downtime of a crashed host.
    pub downtime: Duration,
}

impl Default for Campaign {
    fn default() -> Self {
        Self {
            rounds: 6,
            crash_rate: Probability::new(1, 2_000).unwrap(),
            retention_rate: probability!(0.5),
            mode: PartialWriteMode::Prefix,
            latency: Duration::from_millis(3),
            downtime: Duration::from_secs(5),
        }
    }
}

type R<S, L> = mocks::reporter::Reporter<deterministic::Context, S, L, Sha256Digest>;

/// Checks that no signer ever voted for two payloads (or both nullified and finalized) in one
/// view, across everything every reporter observed, and that finalizations agree.
fn assert_safe<S, L>(reporters: &[R<S, L>])
where
    S: Scheme<Sha256Digest, PublicKey = PublicKey>,
    L: elector::Config<S>,
{
    let mut finalized = BTreeMap::new();
    let mut notarized: BTreeMap<(View, PublicKey), Sha256Digest> = BTreeMap::new();
    let mut finalizes: BTreeMap<(View, PublicKey), Sha256Digest> = BTreeMap::new();
    let mut nullifies: HashSet<(View, PublicKey)> = HashSet::new();
    for reporter in reporters {
        reporter.assert_no_invalid();
        reporter.assert_no_faults();
        for (view, (finalization, _)) in reporter.finalizations.lock().iter() {
            let digest = finalization.proposal.payload;
            assert_eq!(
                *finalized.entry(*view).or_insert(digest),
                digest,
                "conflicting finalizations at view {view}"
            );
        }
        for (view, payloads) in reporter.notarizes.lock().iter() {
            for (digest, signers) in payloads {
                for signer in signers {
                    let previous = *notarized.entry((*view, signer.clone())).or_insert(*digest);
                    assert_eq!(previous, *digest, "{signer} notarized twice at view {view}");
                }
            }
        }
        for (view, payloads) in reporter.finalizes.lock().iter() {
            for (digest, signers) in payloads {
                for signer in signers {
                    let previous = *finalizes.entry((*view, signer.clone())).or_insert(*digest);
                    assert_eq!(previous, *digest, "{signer} finalized twice at view {view}");
                }
            }
        }
        for (view, signers) in reporter.nullifies.lock().iter() {
            for signer in signers {
                nullifies.insert((*view, signer.clone()));
            }
        }
    }
    for (view, signer) in finalizes.keys() {
        assert!(
            !nullifies.contains(&(*view, signer.clone())),
            "{signer} both finalized and nullified view {view}"
        );
    }
}

/// The highest finalized view any reporter has observed.
fn highest<S, L>(reporters: &[R<S, L>]) -> View
where
    S: Scheme<Sha256Digest, PublicKey = PublicKey>,
    L: elector::Config<S>,
{
    reporters
        .iter()
        .filter_map(|reporter| reporter.finalizations.lock().keys().max().copied())
        .max()
        .unwrap_or(View::zero())
}

/// Runs `n` validators as hosts through `campaign` and returns the auditor state.
pub(super) fn host_campaign<S, F, L>(
    seed: u64,
    n: u32,
    campaign: Campaign,
    elector: L,
    mut fixture: F,
) -> String
where
    S: Scheme<Sha256Digest, PublicKey = PublicKey>,
    F: FnMut(&mut deterministic::Context, &[u8], u32) -> Fixture<S>,
    L: elector::Config<S>,
{
    let namespace = b"consensus".to_vec();
    let cfg = deterministic::Config::new()
        .with_seed(seed)
        .with_timeout(Some(Duration::from_secs(3_600)))
        .with_storage_fault_config(
            FaultConfig::default()
                .write(WriteConfig {
                    failure_rate: probability!(0.0),
                    retention_rate: campaign.retention_rate,
                    mode: campaign.mode,
                })
                .latency(Duration::ZERO..campaign.latency),
        );
    deterministic::Runner::new(cfg).start(|mut context| async move {
        let Fixture {
            participants,
            schemes,
            ..
        } = fixture(&mut context, &namespace, n);
        let mut oracle =
            start_test_network_with_peers(context.child("network"), participants.clone(), true)
                .await;
        let link = Link {
            latency: Duration::from_millis(10),
            jitter: Duration::from_millis(3),
            success_rate: probability!(1.0),
        };
        link_validators(&mut oracle, &participants, Action::Link(link), None).await;

        let relay = Arc::new(mocks::relay::Relay::<Sha256Digest, _>::new());
        let mut hosts = Hosts::new(&context);
        let names: Vec<String> = (0..participants.len()).map(|i| format!("v{i}")).collect();
        let mut reporters = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);
        for (idx, validator) in participants.iter().enumerate() {
            let reporter = mocks::reporter::Reporter::new(
                context.child("reporter"),
                mocks::reporter::Config {
                    participants: participants.clone().try_into().unwrap(),
                    scheme: schemes[idx].clone(),
                    elector: elector.clone(),
                },
            );
            reporters.push(reporter.clone());
            let host = hosts.start(&names[idx], HostConfig::new());
            let registration = register_validator(&mut oracle, validator.clone()).await;
            start_nemesis_validator(
                &host,
                &oracle,
                &relay,
                validator,
                schemes[idx].clone(),
                elector.clone(),
                reporter,
                registration,
            );
        }
        await_view(&context, &mut reporters, View::new(5)).await;

        // Crash from within storage operations while the campaign runs
        context.storage_fault_config().write().crash_rate = Some(campaign.crash_rate);

        macro_rules! restart {
            ($idx:expr) => {{
                let idx = $idx;
                let host = hosts.restart(&names[idx]);
                let registration = register_validator(&mut oracle, participants[idx].clone()).await;
                start_nemesis_validator(
                    &host,
                    &oracle,
                    &relay,
                    &participants[idx],
                    schemes[idx].clone(),
                    elector.clone(),
                    reporters[idx].clone(),
                    registration,
                );
            }};
        }

        for round in 0..campaign.rounds {
            context
                .sleep(Duration::from_millis(rng.random_range(200..3_000)))
                .await;
            let action = rng.random_range(0..6u8);
            info!(round, action, "fault");
            match action {
                // Whole-cluster power loss
                0 => hosts.crash_all(),
                // Crash one validator
                1 | 2 => {
                    let idx = rng.random_range(0..names.len());
                    hosts.crash(&names[idx]);
                }
                // Pause one validator
                3 => {
                    let idx = rng.random_range(0..names.len());
                    if let Some(process) = hosts.get(&names[idx]) {
                        process.pause();
                        context
                            .sleep(Duration::from_millis(rng.random_range(100..4_000)))
                            .await;
                        if let Some(process) = hosts.get(&names[idx]) {
                            process.resume();
                        }
                    }
                }
                // Skew one validator's clock
                4 => {
                    let idx = rng.random_range(0..names.len());
                    if let Some(process) = hosts.get(&names[idx]) {
                        process.set_clock_drift(rng.random_range(-50_000..50_000));
                        let skew = Duration::from_millis(rng.random_range(0..2_000));
                        process.set_clock_offset(if rng.random_bool(0.5) {
                            deterministic::ClockOffset::Ahead(skew)
                        } else {
                            deterministic::ClockOffset::Behind(skew)
                        });
                    }
                }
                // Slow one validator's disk
                _ => {
                    let idx = rng.random_range(0..names.len());
                    if let Some(process) = hosts.get(&names[idx]) {
                        let slow = Duration::from_millis(rng.random_range(1..50));
                        process.set_storage_latency(Some(slow / 2..slow));
                    }
                }
            }

            // Restart crashed validators after a downtime (sometimes right away, during
            // another's recovery)
            let down: Vec<usize> = (0..names.len())
                .filter(|idx| !hosts.is_running(&names[*idx]))
                .collect();
            for idx in down {
                let downtime = rng.random_range(0..campaign.downtime.as_millis() as u64);
                context.sleep(Duration::from_millis(downtime)).await;
                restart!(idx);
            }
        }

        // Stop crashing, restart anyone still down, and check liveness
        context.storage_fault_config().write().crash_rate = None;
        for idx in 0..names.len() {
            if !hosts.is_running(&names[idx]) {
                restart!(idx);
            }
        }
        let target = highest(&reporters).saturating_add(ViewDelta::new(10));
        let start = context.current();
        select! {
            _ = await_view(&context, &mut reporters, target) => {},
            _ = context.sleep(Duration::from_secs(300)) => {
                panic!("no liveness after recovery: target {target}, highest {}", highest(&reporters));
            },
        }
        warn!(
            seed,
            elapsed = ?context.current().duration_since(start).unwrap(),
            starts = ?names.iter().map(|name| hosts.starts(name)).collect::<Vec<_>>(),
            "recovered"
        );
        assert_safe(&reporters);
        context.auditor().state()
    })
}

#[test_traced("WARN")]
fn test_host_campaign_smoke() {
    let run = |seed| {
        host_campaign::<_, _, RoundRobin>(
            seed,
            4,
            Campaign {
                rounds: 3,
                ..Campaign::default()
            },
            RoundRobin::default(),
            ed25519::fixture,
        )
    };
    assert_eq!(run(0), run(0), "runs are deterministic");
}

/// Seed sweep; set `SIMPLEX_DST_SEEDS=a..b` to choose seeds.
#[test_group("slow")]
#[test_traced("WARN")]
fn test_host_campaign_sweep() {
    let seeds = std::env::var("SIMPLEX_DST_SEEDS").unwrap_or_else(|_| "0..8".into());
    let (a, b) = seeds.split_once("..").unwrap();
    let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
    let mode = match std::env::var("SIMPLEX_DST_MODE").as_deref() {
        Ok("subset") => PartialWriteMode::Subset,
        _ => PartialWriteMode::Prefix,
    };
    for seed in a..b {
        info!(seed, "campaign");
        eprintln!("simplex host campaign seed {seed}");
        host_campaign::<_, _, RoundRobin>(
            seed,
            4 + (seed % 3) as u32,
            Campaign {
                rounds: 10,
                crash_rate: Probability::new(1 + seed % 4, 1_000).unwrap(),
                retention_rate: Probability::new(seed % 101, 100).unwrap(),
                mode,
                ..Campaign::default()
            },
            RoundRobin::default(),
            ed25519::fixture,
        );
    }
}
