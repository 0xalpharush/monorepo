//! Crash/restart campaigns for QMDB variants on simulated hosts.
//!
//! Each campaign runs a database on a [deterministic::Hosts] host and repeatedly crashes it (at
//! random simulated times and from within its own storage operations, tearing unsynchronized
//! writes) before restarting it. After every restart, the recovered database must equal one of
//! the states applied since the last acknowledged durability point (commit, sync, or awaited
//! start_sync), and it must remain usable.

use crate::{
    merkle::{Family, Location},
    qmdb::{
        Error,
        any::traits::{DbAny, UnmerkleizedBatch as _},
        floor::Proportional,
    },
};
use commonware_cryptography::{Hasher as _, Sha256, sha256::Digest};
use commonware_runtime::{
    Clock as _, Handle, Runner as _, Supervisor as _,
    deterministic::{
        self, Context, FaultConfig, HostConfig, Hosts, MetadataConfig, PartialWriteMode,
        WriteConfig,
    },
};
use commonware_utils::{Probability, TestRng, sync::Mutex};
use rand::RngExt as _;
use std::{collections::BTreeMap, fmt::Debug, future::Future, sync::Arc, time::Duration};

/// How a campaign injects faults.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Faults {
    /// Crash/restart cycles before the final, fault-free recovery.
    pub cycles: usize,
    /// Upper bound of the simulated time a host runs before it is crashed.
    pub max_uptime: Duration,
    /// Probability that a write or sync crashes the host from within.
    pub crash_rate: Probability,
    /// Probability that an open or remove crashes the host from within.
    pub metadata_crash_rate: Probability,
    /// Probability that each unsynchronized byte (or prefix) survives a crash.
    pub retention_rate: Probability,
    /// Arrangement of retained bytes.
    pub mode: PartialWriteMode,
    /// Upper bound of each storage operation's latency.
    pub max_latency: Duration,
    /// Number of distinct keys the workload writes.
    pub keys: u64,
}

impl Faults {
    /// Moderate crash rates with prefix tearing.
    pub(crate) fn prefix(cycles: usize) -> Self {
        Self {
            cycles,
            max_uptime: Duration::from_millis(1_500),
            crash_rate: Probability::new(1, 200).unwrap(),
            metadata_crash_rate: Probability::new(1, 100).unwrap(),
            retention_rate: Probability::new(1, 2).unwrap(),
            mode: PartialWriteMode::Prefix,
            max_latency: Duration::from_millis(1),
            keys: 40,
        }
    }

    /// Subset tearing (stronger than real sector-granular tearing).
    pub(crate) fn subset(cycles: usize) -> Self {
        Self {
            mode: PartialWriteMode::Subset,
            retention_rate: Probability::new(9, 10).unwrap(),
            ..Self::prefix(cycles)
        }
    }

    /// High crash rates, so recovery itself is frequently interrupted.
    pub(crate) fn recovery(cycles: usize) -> Self {
        Self {
            crash_rate: Probability::new(1, 15).unwrap(),
            metadata_crash_rate: Probability::new(1, 10).unwrap(),
            ..Self::prefix(cycles)
        }
    }

    fn storage(&self) -> FaultConfig {
        FaultConfig::default()
            .write(WriteConfig {
                failure_rate: Probability::new(0, 1).unwrap(),
                retention_rate: self.retention_rate,
                mode: self.mode,
            })
            .crash(self.crash_rate)
            .metadata(MetadataConfig {
                crash_rate: self.metadata_crash_rate,
                retention_rate: Probability::new(1, 2).unwrap(),
            })
            .latency(Duration::ZERO..self.max_latency)
    }
}

/// A state the workload applied.
#[derive(Clone)]
struct Applied<F: Family, V> {
    size: Location<F>,
    root: Digest,
    state: BTreeMap<u64, V>,
    metadata: Option<V>,
    /// What produced the state (for diagnostics).
    label: &'static str,
}

/// What the workload applied and what it knows to be durable.
pub(crate) struct Model<F: Family, V> {
    applied: Vec<Applied<F, V>>,
    /// Index into `applied` of the latest state acknowledged as durable.
    acked: Option<usize>,
    /// Every recovery: (cycle, recovered index, acked index, issued index).
    recoveries: Vec<(usize, usize, Option<usize>, usize)>,
    /// Batches applied in total.
    batches: u64,
}

fn key(i: u64) -> Digest {
    Sha256::hash(&[&i.to_be_bytes()])
}

/// The outcome of a campaign run.
#[derive(Debug)]
pub(crate) struct Summary {
    /// The auditor state at the end of the run.
    pub state: String,
    /// Recoveries checked.
    pub recoveries: usize,
    /// Recoveries to a state applied after the acknowledged one (unsynced data survived).
    pub beyond: usize,
    /// Batches applied.
    pub batches: u64,
}

/// One run of the workload on a (re)started host.
pub(crate) struct Task<F: Family, V> {
    pub cycle: usize,
    pub seed: u64,
    pub keys: u64,
    pub model: Arc<Mutex<Model<F, V>>>,
    /// In the final round, the number of durable batches after which the workload returns.
    pub final_batches: Option<usize>,
}

/// Recover the database, check it against the model, and run batches until crashed (or, in the
/// final round, until `final_batches` batches are durable).
pub(crate) async fn workload<F, D, V, O, Fut>(
    context: Context,
    task: Task<F, V>,
    open: O,
    make_value: fn(u64) -> V,
) where
    F: Family,
    D: DbAny<F, Key = Digest, Value = V, Digest = Digest>,
    V: Clone + Eq + Debug + Send + Sync + 'static,
    O: Fn(Context) -> Fut,
    Fut: Future<Output = Result<D, Error<F>>>,
{
    let Task {
        cycle,
        seed,
        keys,
        model,
        final_batches,
    } = task;
    let mut db = match open(context.child("db")).await {
        Ok(db) => db,
        Err(err) => {
            let model = model.lock();
            panic!(
                "seed={seed} cycle={cycle}: recovery failed: {err:?} (acked={:?}, issued={}, recoveries={:?})",
                model.acked,
                model.applied.len().saturating_sub(1),
                model.recoveries
            );
        }
    };

    // The recovered database must equal a state applied since the last acknowledged one
    let candidate = {
        let mut model = model.lock();
        if model.applied.is_empty() {
            model.applied.push(Applied {
                size: db.size(),
                root: db.root(),
                state: BTreeMap::new(),
                metadata: None,
                label: "init",
            });
            model.acked = Some(0);
            None
        } else {
            let lo = model.acked.unwrap_or(0);
            let issued = model.applied.len() - 1;
            let found = (lo..=issued)
                .rev()
                .find(|&j| model.applied[j].size == db.size() && model.applied[j].root == db.root());
            let Some(j) = found else {
                let sizes: Vec<_> = model.applied[lo..]
                    .iter()
                    .map(|a| (*a.size, a.label))
                    .collect();
                panic!(
                    "seed={seed} cycle={cycle}: recovered size {} root {:?} matches no state applied since the acked one (acked={:?}, issued={issued}, candidates={sizes:?}, recoveries={:?})",
                    *db.size(),
                    db.root(),
                    model.acked,
                    model.recoveries
                );
            };
            let acked = model.acked;
            model.recoveries.push((cycle, j, acked, issued));
            model.applied.truncate(j + 1);
            Some(model.applied[j].clone())
        }
    };
    if let Some(expected) = candidate {
        for i in 0..keys {
            let got = db.get(&key(i)).await.unwrap();
            assert_eq!(
                got.as_ref(),
                expected.state.get(&i),
                "seed={seed} cycle={cycle}: key {i} differs after recovery to size {}",
                *expected.size
            );
        }
        assert_eq!(
            db.get_metadata().await.unwrap(),
            expected.metadata,
            "seed={seed} cycle={cycle}: metadata differs after recovery"
        );
        // Recovery persists the recovered state
        let mut model = model.lock();
        model.acked = Some(model.applied.len() - 1);
    }

    let mut rng = TestRng::new(seed ^ ((cycle as u64) << 32));
    let mut committed = 0;
    loop {
        let (mut state, metadata) = {
            let model = model.lock();
            let last = model.applied.last().unwrap();
            (last.state.clone(), last.metadata.clone())
        };
        let mut batch = db.new_batch();
        for _ in 0..rng.random_range(1..8) {
            let k = rng.random_range(0..keys);
            if rng.random_bool(0.2) {
                batch = batch.write(key(k), None);
                state.remove(&k);
            } else {
                let value = make_value(rng.random());
                batch = batch.write(key(k), Some(value.clone()));
                state.insert(k, value);
            }
        }
        let metadata = if rng.random_bool(0.3) {
            let value = make_value(rng.random());
            Some(value)
        } else {
            drop(metadata);
            None
        };
        let merkleized = batch
            .merkleize(&db, metadata.clone(), &mut Proportional)
            .await
            .unwrap();
        let (next, _) = db.apply_batch(merkleized).await.unwrap();
        db = next;
        let root = db.root();
        {
            let mut model = model.lock();
            model.batches += 1;
            model.applied.push(Applied {
                size: db.size(),
                root,
                state,
                metadata,
                label: "apply",
            });
        }

        // Make the applied states durable in one of several ways
        let index = model.lock().applied.len() - 1;
        match rng.random_range(0..10) {
            0..=3 => {
                db = db.commit().await.unwrap();
                model.lock().acked = Some(index);
                committed += 1;
            }
            4 => {
                db = db.sync().await.unwrap();
                model.lock().acked = Some(index);
                committed += 1;
            }
            5 => {
                let (next, handle) = db.start_sync().await.unwrap();
                db = next;
                handle.await.unwrap();
                model.lock().acked = Some(index);
                committed += 1;
            }
            6 => {
                let boundary = db.sync_boundary();
                db = db.prune(boundary).await.unwrap();
            }
            _ => {}
        }
        if final_batches.is_some_and(|target| committed >= target) {
            db = db.commit().await.unwrap();
            let mut model = model.lock();
            model.acked = Some(model.applied.len() - 1);
            drop(db);
            return;
        }
    }
}

/// Run a crash/restart campaign, where `start` spawns [workload] for a [Task] on a host's
/// context (so the spawned future is checked for `Send` with a concrete database type),
/// returning the auditor state, the number of recoveries, and the number of applied batches.
pub(crate) fn run<F, V, S>(seed: u64, faults: Faults, start: S) -> Summary
where
    F: Family,
    V: Clone + Eq + Debug + Send + Sync + 'static,
    S: Fn(Context, Task<F, V>) -> Handle<()> + Send + 'static,
{
    let cfg = deterministic::Config::default()
        .with_seed(seed)
        .with_timeout(Some(Duration::from_secs(36_000)))
        .with_storage_fault_config(faults.storage());
    deterministic::Runner::new(cfg).start(|context| async move {
        let model = Arc::new(Mutex::new(Model::<F, V> {
            applied: Vec::new(),
            acked: None,
            recoveries: Vec::new(),
            batches: 0,
        }));
        let mut hosts = Hosts::new(&context);
        let mut rng = TestRng::new(seed);
        let task = |cycle, final_batches| Task {
            cycle,
            seed,
            keys: faults.keys,
            model: model.clone(),
            final_batches,
        };
        for cycle in 0..faults.cycles {
            let host = if cycle == 0 {
                hosts.start("db", HostConfig::new())
            } else {
                hosts.restart("db")
            };
            drop(start(host, task(cycle, None)));
            let uptime = rng.random_range(0..=faults.max_uptime.as_millis() as u64);
            context.sleep(Duration::from_millis(uptime)).await;
            hosts.crash("db");
        }

        // Recover without faults: the database must recover and keep working
        *context.storage_fault_config().write() = FaultConfig::default();
        let host = hosts.restart("db");
        start(host, task(faults.cycles, Some(3)))
            .await
            .expect("final round failed");
        hosts.crash("db");
        let model = model.lock();
        let beyond = model
            .recoveries
            .iter()
            .filter(|(_, j, acked, _)| Some(*j) > *acked)
            .count();
        Summary {
            state: context.auditor().state(),
            recoveries: model.recoveries.len(),
            beyond,
            batches: model.batches,
        }
    })
}

/// A [run] start function for a database opened by the async function `$open` with values made
/// by `$value`.
macro_rules! starter {
    ($open:expr, $value:expr) => {
        |host: commonware_runtime::deterministic::Context, task| {
            commonware_runtime::Spawner::spawn(host, move |context| {
                $crate::qmdb::crash_campaign::workload(context, task, $open, $value)
            })
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        merkle::mmr,
        qmdb::any::{
            test::fixed_db_config,
            unordered::fixed::test::AnyTest as AnyFixed,
        },
        translator::TwoCap,
    };
    use commonware_macros::test_group;

    fn digest_value(i: u64) -> Digest {
        Sha256::hash(&[&i.to_le_bytes()])
    }

    async fn open_any_fixed(context: Context) -> Result<AnyFixed, Error<mmr::Family>> {
        let cfg = fixed_db_config::<TwoCap>("c", &context);
        AnyFixed::init(context, cfg, None).await
    }

    fn any_fixed(seed: u64, faults: Faults) -> Summary {
        run(seed, faults, starter!(open_any_fixed, digest_value))
    }

    #[test]
    fn test_any_fixed_crash_smoke() {
        let summary = any_fixed(0, Faults::prefix(8));
        assert!(summary.recoveries > 0 && summary.batches > 0, "{summary:?}");
        assert_eq!(summary.state, any_fixed(0, Faults::prefix(8)).state);
    }

    fn sweep(name: &str, seeds: std::ops::Range<u64>, f: impl Fn(u64) -> Summary) {
        let seeds = std::env::var("CRASH_SEEDS")
            .ok()
            .and_then(|s| {
                let (a, b) = s.split_once("..")?;
                Some(a.parse().ok()?..b.parse().ok()?)
            })
            .unwrap_or(seeds);
        for seed in seeds {
            let Summary {
                recoveries,
                beyond,
                batches,
                ..
            } = f(seed);
            eprintln!(
                "{name} seed={seed} recoveries={recoveries} beyond_acked={beyond} batches={batches}"
            );
        }
    }

    #[test_group("slow")]
    #[test]
    fn test_any_fixed_crash_campaign() {
        sweep("any_fixed/prefix", 0..50, |s| any_fixed(s, Faults::prefix(30)));
        sweep("any_fixed/subset", 0..50, |s| any_fixed(s, Faults::subset(30)));
        sweep("any_fixed/recovery", 0..50, |s| any_fixed(s, Faults::recovery(30)));
    }
}
