//! Code-level fault points ([crate::buggify!]) for the deterministic runtime.

use crate::Site;
use commonware_utils::{Probability, sync::Mutex};
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest as _, Sha256};
use std::{collections::HashMap, sync::Arc};

/// Domain separator for [Seeded] draws.
const NAMESPACE: &[u8] = b"_COMMONWARE_RUNTIME_BUGGIFY";

/// Decides whether each [crate::buggify!] site is enabled and when an enabled site fires.
///
/// The runtime asks [Self::enabled] once per site, on the site's first evaluation, and remembers
/// the answer for the rest of the run (including runs recovered with
/// [super::Runner::start_and_recover]). Every evaluation of an enabled site then asks
/// [Self::fires] with the site's evaluation index, so a policy can derive, record, replay, or
/// override each decision independently of all other randomness in the runtime. Sites that are
/// not enabled never consult [Self::fires].
pub trait BuggifyPolicy: Send + Sync {
    /// Whether `site` is enabled for the rest of the run.
    fn enabled(&self, site: &Site) -> bool;

    /// Whether enabled `site` fires on its `evaluation`-th evaluation (starting at zero).
    ///
    /// `rate` is the probability requested by the site, or `None` for the policy's default.
    fn fires(&self, site: &Site, evaluation: u64, rate: Option<Probability>) -> bool;
}

/// The default [BuggifyPolicy]: decisions are hashes of a seed, the site, and the evaluation
/// index, so they never draw from (or perturb) the runtime's RNG.
#[derive(Clone, Copy, Debug)]
pub struct Seeded {
    seed: u64,
    enable_rate: Probability,
    fire_rate: Probability,
}

impl Seeded {
    /// Enable each site with probability `enable_rate` and fire an enabled site with probability
    /// `fire_rate` (unless the site requests its own rate).
    pub const fn new(seed: u64, enable_rate: Probability, fire_rate: Probability) -> Self {
        Self {
            seed,
            enable_rate,
            fire_rate,
        }
    }

    /// Sample `rate` with a draw derived from the seed, `site`, and `index`.
    fn sample(&self, site: &Site, index: Option<u64>, rate: Probability) -> bool {
        if rate.is_zero() || rate.is_one() {
            return rate.is_one();
        }
        let mut hasher = Sha256::new();
        hasher.update(NAMESPACE);
        hasher.update(self.seed.to_be_bytes());
        hasher.update(site.file.as_bytes());
        hasher.update(site.line.to_be_bytes());
        hasher.update(site.column.to_be_bytes());
        match index {
            None => hasher.update([0]),
            Some(index) => {
                hasher.update([1]);
                hasher.update(index.to_be_bytes());
            }
        }
        rate.sample(&mut StdRng::from_seed(hasher.finalize().into()))
    }
}

impl BuggifyPolicy for Seeded {
    fn enabled(&self, site: &Site) -> bool {
        self.sample(site, None, self.enable_rate)
    }

    fn fires(&self, site: &Site, evaluation: u64, rate: Option<Probability>) -> bool {
        self.sample(site, Some(evaluation), rate.unwrap_or(self.fire_rate))
    }
}

/// Per-site state remembered for the run.
#[derive(Clone, Copy)]
struct Entry {
    enabled: bool,
    evaluations: u64,
}

/// Remembers each site's enablement and evaluation count for a run.
pub(super) struct Buggify {
    policy: Arc<dyn BuggifyPolicy>,
    sites: Mutex<HashMap<Site, Entry>>,
}

impl Buggify {
    pub(super) fn new(policy: Arc<dyn BuggifyPolicy>) -> Self {
        Self {
            policy,
            sites: Mutex::new(HashMap::new()),
        }
    }

    /// Evaluate `site` once.
    pub(super) fn evaluate(&self, site: &Site, rate: Option<Probability>) -> bool {
        // Consult the policy without holding the lock, so a policy may evaluate sites itself.
        let known = self.sites.lock().get(site).map(|entry| entry.enabled);
        let enabled = known.unwrap_or_else(|| {
            let enabled = self.policy.enabled(site);
            self.sites
                .lock()
                .entry(*site)
                .or_insert(Entry {
                    enabled,
                    evaluations: 0,
                })
                .enabled
        });
        if !enabled {
            return false;
        }
        let evaluation = {
            let mut sites = self.sites.lock();
            let entry = sites.get_mut(site).expect("site was recorded");
            entry.evaluations += 1;
            entry.evaluations - 1
        };
        self.policy.fires(site, evaluation, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Runner as _, deterministic};
    use commonware_utils::probability;
    use rand::Rng as _;

    /// Evaluate eight distinct sites `rounds` times, returning each round's outcomes.
    fn evaluate(context: &deterministic::Context, rounds: usize) -> Vec<[bool; 8]> {
        (0..rounds)
            .map(|_| {
                [
                    buggify!(context),
                    buggify!(context),
                    buggify!(context),
                    buggify!(context),
                    buggify!(context),
                    buggify!(context),
                    buggify!(context),
                    buggify!(context),
                ]
            })
            .collect()
    }

    fn run(cfg: deterministic::Config, rounds: usize) -> (Vec<[bool; 8]>, u64) {
        deterministic::Runner::new(cfg).start(|mut context| async move {
            let outcomes = evaluate(&context, rounds);
            (outcomes, context.next_u64())
        })
    }

    #[test]
    fn test_off_by_default() {
        let (outcomes, draw) = run(deterministic::Config::default(), 100);
        assert!(outcomes.iter().flatten().all(|fired| !fired));

        // Evaluating sites consumes no randomness.
        let (_, baseline) = run(deterministic::Config::default(), 0);
        assert_eq!(draw, baseline);
    }

    #[test]
    fn test_decisions_do_not_perturb_rng() {
        let cfg = || {
            deterministic::Config::default().with_buggify(7, probability!(0.5), probability!(0.5))
        };
        let (outcomes, draw) = run(cfg(), 100);
        assert!(outcomes.iter().flatten().any(|fired| *fired));
        let (_, baseline) = run(deterministic::Config::default(), 0);
        assert_eq!(draw, baseline);
    }

    #[test]
    fn test_enablement_is_stable_within_run() {
        for seed in 0..16 {
            let cfg = deterministic::Config::default().with_buggify(
                seed,
                probability!(0.5),
                probability!(1.0),
            );
            let (outcomes, _) = run(cfg, 50);
            // With fire rate one, a site fires on every evaluation or on none.
            for site in 0..8 {
                let first = outcomes[0][site];
                assert!(outcomes.iter().all(|round| round[site] == first));
            }
        }
    }

    #[test]
    fn test_deterministic_across_runs() {
        let cfg = |seed| {
            deterministic::Config::default().with_buggify(
                seed,
                probability!(0.5),
                probability!(0.25),
            )
        };
        let (first, _) = run(cfg(1), 100);
        let (second, _) = run(cfg(1), 100);
        assert_eq!(first, second);
        let (other, _) = run(cfg(2), 100);
        assert_ne!(first, other);
    }

    #[test]
    fn test_site_rate_overrides_default() {
        let cfg =
            deterministic::Config::default().with_buggify(3, probability!(1.0), probability!(0.0));
        deterministic::Runner::new(cfg).start(|context| async move {
            for _ in 0..10 {
                assert!(!buggify!(context));
                assert!(buggify!(context, probability!(1.0)));
            }
        });
    }

    /// Enables only sites on `line` and fires them on `evaluation`.
    struct Force {
        line: u32,
        evaluation: u64,
        seen: Mutex<Vec<(Site, u64)>>,
    }

    impl BuggifyPolicy for Force {
        fn enabled(&self, site: &Site) -> bool {
            site.line == self.line
        }

        fn fires(&self, site: &Site, evaluation: u64, rate: Option<Probability>) -> bool {
            assert!(rate.is_none());
            self.seen.lock().push((*site, evaluation));
            evaluation == self.evaluation
        }
    }

    #[test]
    fn test_policy_forces_site() {
        let target = line!() + 10;
        let policy = Arc::new(Force {
            line: target,
            evaluation: 2,
            seen: Mutex::new(Vec::new()),
        });
        let cfg = deterministic::Config::default().with_buggify_policy(policy.clone());
        let fired = deterministic::Runner::new(cfg).start(|context| async move {
            (0..5)
                .map(|_| {
                    let forced = buggify!(context);
                    let other = buggify!(context);
                    (forced, other)
                })
                .collect::<Vec<_>>()
        });
        let forced: Vec<_> = fired.iter().map(|(forced, _)| *forced).collect();
        let others: Vec<_> = fired.iter().map(|(_, other)| *other).collect();
        assert_eq!(forced, vec![false, false, true, false, false]);
        assert_eq!(others, vec![false; 5]);

        // Only the enabled site consults the policy, once per evaluation.
        let seen = policy.seen.lock();
        assert!(
            seen.iter()
                .all(|(site, _)| site.line == target && site.file == file!())
        );
        assert_eq!(
            seen.iter()
                .map(|(_, evaluation)| *evaluation)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn test_enablement_survives_recovery() {
        let policy = Arc::new(Force {
            line: line!() + 5,
            evaluation: 1,
            seen: Mutex::new(Vec::new()),
        });
        let cfg = deterministic::Config::default().with_buggify_policy(policy);
        let site = |context: deterministic::Context| async move { buggify!(context) };
        let (fired, checkpoint) = deterministic::Runner::new(cfg).start_and_recover(site);
        assert!(!fired);
        // The evaluation count continues in the recovered run.
        let fired = deterministic::Runner::from(checkpoint).start(site);
        assert!(fired);
    }
}
