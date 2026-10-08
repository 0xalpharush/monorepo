//! Fault campaigns without crashes: every validator runs for the whole campaign while links
//! are cut, partitioned, slowed, and made lossy, validators are paused, and the automaton
//! answers slowly. A reporter wrapper records each time a validator's tip moves onto a height it
//! had already certified (which no further certificate advances past); once faults stop, every
//! validator must keep advancing its tip.

use super::*;
use crate::{Automaton, aggregation::types::Activity};
use commonware_runtime::deterministic::{HostConfig, Hosts};
use commonware_utils::{Probability, sync::Mutex};
use rand::{SeedableRng as _, rngs::StdRng};
use std::{collections::BTreeSet, sync::Arc, time::SystemTime};

/// An automaton that answers each proposal after a random delay below `max_delay`.
#[derive(Clone)]
struct SlowAutomaton {
    context: Arc<deterministic::Context>,
    rng: Arc<Mutex<StdRng>>,
    inner: mocks::Application,
    max_delay: Duration,
}

impl Automaton for SlowAutomaton {
    type Context = Height;
    type Digest = Sha256Digest;

    async fn propose(&mut self, context: Height) -> oneshot::Receiver<Sha256Digest> {
        let digest = self.inner.propose(context).await;
        if self.max_delay.is_zero() {
            return digest;
        }
        let delay = Duration::from_micros(
            self.rng
                .lock()
                .random_range(0..self.max_delay.as_micros() as u64),
        );
        let (sender, receiver) = oneshot::channel();
        self.context
            .child("slow_propose")
            .spawn(move |context| async move {
                context.sleep(delay).await;
                if let Ok(digest) = digest.await {
                    sender.send_lossy(digest);
                }
            });
        receiver
    }

    async fn verify(&mut self, context: Height, payload: Sha256Digest) -> oneshot::Receiver<bool> {
        self.inner.verify(context, payload).await
    }
}

/// What one validator's reporter observed.
#[derive(Default, Debug)]
struct Watch {
    /// Heights the validator certified (pruned with nothing; heights are small).
    certified: BTreeSet<Height>,
    /// The latest tip the validator reported.
    tip: Height,
    /// Tips the validator moved onto that it had already certified.
    landings: Vec<Height>,
    /// When the validator's current tip landed on a certified height (if it did).
    landed_at: Option<SystemTime>,
    /// How long each landed tip stayed in place.
    stuck: Vec<Duration>,
}

/// Forwards to the checking reporter and records tips that land on certified heights.
#[derive(Clone)]
struct Watcher<S: Scheme<Sha256Digest, PublicKey = PublicKey>> {
    inner: mocks::ReporterMailbox<S, Sha256Digest>,
    watch: Arc<Mutex<Watch>>,
    clock: Arc<deterministic::Context>,
}

impl<S: Scheme<Sha256Digest, PublicKey = PublicKey>> crate::Reporter for Watcher<S> {
    type Activity = Activity<S, Sha256Digest>;

    fn report(&mut self, activity: Self::Activity) -> commonware_actor::Feedback {
        {
            let mut watch = self.watch.lock();
            match &activity {
                Activity::Certified(certificate) => {
                    watch.certified.insert(certificate.item.height);
                }
                Activity::Tip(height) => {
                    let now = self.clock.current();
                    if let Some(at) = watch.landed_at.take() {
                        watch.stuck.push(now.duration_since(at).unwrap_or_default());
                    }
                    if watch.certified.contains(height) {
                        watch.landings.push(*height);
                        watch.landed_at = Some(now);
                    }
                    watch.tip = *height;
                }
                Activity::Ack(_) => {}
            }
        }
        self.inner.report(activity)
    }
}

/// The outcome of a campaign.
#[derive(Debug)]
pub(super) struct Outcome {
    /// Per validator, the tips it moved onto that it had already certified.
    pub landings: Vec<Vec<Height>>,
    /// The longest a landed tip stayed in place (including one still in place at the end).
    pub max_stuck: Duration,
    /// Whether every validator kept advancing its tip after faults stopped.
    pub live: bool,
    /// Per validator, the tip after the liveness check.
    pub tips: Vec<Height>,
}

/// Run a campaign of `rounds` faults on `n` validators. `window` and `max_automaton_delay`
/// configure every engine.
pub(super) fn live_campaign(
    seed: u64,
    n: u32,
    rounds: usize,
    window: u64,
    max_automaton_delay: Duration,
) -> Outcome {
    let cfg = deterministic::Config::new()
        .with_seed(seed)
        .with_timeout(Some(Duration::from_secs(7_200)));
    deterministic::Runner::new(cfg).start(|mut context| async move {
        let fixture = ed25519::fixture(&mut context, TEST_NAMESPACE, n);
        let participants = fixture.participants.clone();
        let epoch = Epoch::new(111);
        let (oracle, mut registrations) =
            initialize_simulation(context.child("simulation"), &fixture, RELIABLE_LINK).await;
        let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
        let mut hosts = Hosts::new(&context);
        let mut watches = Vec::new();
        for (idx, participant) in participants.iter().enumerate() {
            let (reporter, mailbox) =
                mocks::Reporter::new(context.child("reporter"), fixture.verifier.clone());
            reporter.start();
            let watch = Arc::new(Mutex::new(Watch::default()));
            watches.push(watch.clone());
            let host = hosts.start(&names[idx], HostConfig::new());
            let provider = mocks::Provider::new();
            assert!(provider.register(epoch, fixture.schemes[idx].clone()));
            let engine = Engine::new(
                host.child("engine"),
                Config {
                    monitor: mocks::Monitor::new(epoch),
                    provider,
                    automaton: SlowAutomaton {
                        context: Arc::new(host.child("automaton")),
                        rng: Arc::new(Mutex::new(StdRng::seed_from_u64(seed ^ idx as u64))),
                        inner: mocks::Application::new(mocks::Strategy::Correct),
                        max_delay: max_automaton_delay,
                    },
                    reporter: Watcher {
                        inner: mailbox,
                        watch,
                        clock: Arc::new(host.child("clock")),
                    },
                    blocker: oracle.control(participant.clone()),
                    priority_acks: false,
                    rebroadcast_timeout: NonZeroDuration::new_panic(Duration::from_millis(100)),
                    epoch_bounds: (EpochDelta::new(1), EpochDelta::new(1)),
                    window: std::num::NonZeroU64::new(window).unwrap(),
                    activity_timeout: HeightDelta::new(1_024),
                    journal_partition: "aggregation".into(),
                    journal_write_buffer: NZUsize!(4096),
                    journal_replay_buffer: NZUsize!(4096),
                    journal_heights_per_section: std::num::NonZeroU64::new(6).unwrap(),
                    journal_compression: Some(3),
                    journal_page_cache: CacheRef::from_pooler(&host, PAGE_SIZE, PAGE_CACHE_SIZE),
                    strategy: Sequential,
                },
            );
            engine.start(registrations.remove(participant).unwrap());
        }
        context.sleep(Duration::from_secs(2)).await;

        // Every directed link is either up (with some configuration) or cut
        let pairs: Vec<(usize, usize)> = (0..n as usize)
            .flat_map(|a| {
                (0..n as usize)
                    .filter(move |&b| b != a)
                    .map(move |b| (a, b))
            })
            .collect();
        let set_link = |a: usize, b: usize, link: Option<Link>| {
            let oracle = oracle.clone();
            let (pa, pb) = (participants[a].clone(), participants[b].clone());
            async move {
                let _ = oracle.remove_link(pa.clone(), pb.clone()).await;
                if let Some(link) = link {
                    oracle.add_link(pa, pb, link).await.unwrap();
                }
            }
        };
        let heal = || async {
            for &(a, b) in &pairs {
                set_link(a, b, Some(RELIABLE_LINK)).await;
            }
        };

        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..rounds {
            let idx = rng.random_range(0..n as usize);
            let hold = Duration::from_millis(rng.random_range(100..5_000));
            match rng.random_range(0..6u8) {
                // Partition into two random groups
                0 => {
                    let side: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
                    for &(a, b) in &pairs {
                        if side[a] != side[b] {
                            set_link(a, b, None).await;
                        }
                    }
                    context.sleep(hold).await;
                    heal().await;
                }
                // Cut a random subset of directed links (asymmetric loss)
                1 => {
                    for &(a, b) in &pairs {
                        if rng.random_bool(0.3) {
                            set_link(a, b, None).await;
                        }
                    }
                    context.sleep(hold).await;
                    heal().await;
                }
                // Slow and lossy links everywhere
                2 => {
                    let link = Link {
                        latency: Duration::from_millis(rng.random_range(10..1_000)),
                        jitter: Duration::from_millis(rng.random_range(0..500)),
                        success_rate: Probability::new(rng.random_range(30..100), 100).unwrap(),
                    };
                    for &(a, b) in &pairs {
                        set_link(a, b, Some(link.clone())).await;
                    }
                    context.sleep(hold).await;
                    heal().await;
                }
                // Isolate (or slow down) one validator
                3 => {
                    let link = if rng.random_bool(0.5) {
                        None
                    } else {
                        Some(Link {
                            latency: Duration::from_millis(rng.random_range(200..3_000)),
                            jitter: Duration::from_millis(rng.random_range(0..500)),
                            success_rate: Probability::new(rng.random_range(50..100), 100).unwrap(),
                        })
                    };
                    for &(a, b) in &pairs {
                        if a == idx || b == idx {
                            set_link(a, b, link.clone()).await;
                        }
                    }
                    context.sleep(hold).await;
                    heal().await;
                }
                // Pause a validator
                4 => {
                    hosts.get(&names[idx]).unwrap().pause();
                    context.sleep(hold).await;
                    hosts.get(&names[idx]).unwrap().resume();
                }
                // Let it run
                _ => context.sleep(hold).await,
            }
            context
                .sleep(Duration::from_millis(rng.random_range(0..1_000)))
                .await;
        }

        // Stop faults and check every validator keeps advancing its tip
        heal().await;
        let tips = |watches: &[Arc<Mutex<Watch>>]| -> Vec<Height> {
            watches.iter().map(|w| w.lock().tip).collect()
        };
        let before = tips(&watches);
        let target = before
            .iter()
            .max()
            .unwrap()
            .saturating_add(HeightDelta::new(20));
        let deadline = context.current() + Duration::from_secs(300);
        let live = loop {
            if tips(&watches).iter().all(|tip| *tip >= target) {
                break true;
            }
            if context.current() >= deadline {
                break false;
            }
            context.sleep(Duration::from_millis(100)).await;
        };
        Outcome {
            max_stuck: watches
                .iter()
                .flat_map(|w| {
                    let w = w.lock();
                    let open = w
                        .landed_at
                        .map(|at| context.current().duration_since(at).unwrap_or_default());
                    w.stuck.iter().copied().chain(open).collect::<Vec<_>>()
                })
                .max()
                .unwrap_or_default(),
            landings: watches.iter().map(|w| w.lock().landings.clone()).collect(),
            live,
            tips: tips(&watches),
        }
    })
}

/// Seed sweep without crashes; `AGG_LIVE_SEEDS=a..b`, `AGG_LIVE_N`, `AGG_LIVE_WINDOW`,
/// `AGG_LIVE_DELAY_MS` (max automaton delay). Prints how often a tip lands on a certified
/// height and fails if the cluster stops certifying.
#[test_traced("WARN")]
#[ignore]
fn test_live_campaign_sweep() {
    let env = |name: &str, default: &str| std::env::var(name).unwrap_or_else(|_| default.into());
    let seeds = env("AGG_LIVE_SEEDS", "0..8");
    let (a, b) = seeds.split_once("..").unwrap();
    let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
    let n: u32 = env("AGG_LIVE_N", "4").parse().unwrap();
    let window: u64 = env("AGG_LIVE_WINDOW", "10").parse().unwrap();
    let delay: u64 = env("AGG_LIVE_DELAY_MS", "200").parse().unwrap();
    let mut stalled = Vec::new();
    let mut landed = 0;
    for seed in a..b {
        let outcome = live_campaign(seed, n, 20, window, Duration::from_millis(delay));
        let landings: usize = outcome.landings.iter().map(Vec::len).sum();
        landed += usize::from(landings > 0);
        eprintln!(
            "seed {seed}: live={} landings={landings} max_stuck={:?} tips={:?}",
            outcome.live, outcome.max_stuck, outcome.tips
        );
        if !outcome.live {
            stalled.push(seed);
        }
    }
    eprintln!(
        "seeds with landings: {landed}/{}; stalled: {stalled:?}",
        b - a
    );
    assert!(stalled.is_empty(), "stalled seeds: {stalled:?}");
}

/// Without any crash or restart, a validator's tip is fast-forwarded (to the safe tip of its
/// peers) onto a height it had already certified: it certified that height out of order while
/// missing the certificate for its tip, and its peers then reported that height as their tip.
/// No further certificate arrives for a certified height, so the validator stays there until
/// enough peers move past it. (In these campaigns its peers always do, so the stall is
/// transient; a crash campaign is needed to stall a quorum this way.)
#[test_traced("WARN")]
#[ignore = "finding: fast-forwarding the tip onto a certified height stalls it"]
fn test_live_campaign_tip_never_lands_on_certified_height() {
    let outcome = live_campaign(0, 4, 20, 10, Duration::from_millis(200));
    assert!(outcome.live, "cluster stalled: {outcome:?}");
    assert!(
        outcome.landings.iter().all(Vec::is_empty),
        "tips landed on certified heights without crashes: {:?} (longest stay {:?})",
        outcome.landings,
        outcome.max_stuck
    );
}
