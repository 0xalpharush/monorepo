//! Scheduling strategies for the `deterministic` runtime.
//!
//! Each strategy is a [SchedulingPolicy] with its own seeded randomness, so the schedules it
//! explores are independent of every other use of the runtime's RNG. Strategies follow the
//! controlled concurrency testing literature, adapted to cooperative batch scheduling: every
//! iteration of the event loop polls a batch of runnable tasks, so a strategy can reorder the
//! batch and (with [Pausing]) hold tasks back across iterations.

use super::{RunnableTask, SchedulingPolicy};
use commonware_utils::Probability;
use rand::{RngExt as _, SeedableRng, prelude::SliceRandom, rngs::StdRng};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::{Duration, SystemTime},
};

/// Draws `count` distinct points uniformly from `0..steps`.
fn points(rng: &mut StdRng, count: usize, steps: u64) -> BTreeSet<u64> {
    let mut points = BTreeSet::new();
    let count = count.min(usize::try_from(steps).unwrap_or(usize::MAX));
    while points.len() < count {
        points.insert(rng.random_range(0..steps));
    }
    points
}

/// Polls each batch in a uniformly random order: a random walk over schedules.
pub struct RandomWalk {
    rng: StdRng,
}

impl RandomWalk {
    /// A random walk seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl SchedulingPolicy for RandomWalk {
    fn order(&mut self, _: SystemTime, ready: &mut [RunnableTask]) {
        ready.shuffle(&mut self.rng);
    }
}

/// Probabilistic concurrency testing (PCT; Burckhardt et al., ASPLOS 2010).
///
/// Each task receives a random priority when first seen, and every batch is polled in
/// descending priority order, so the same tasks consistently run first. At `depth - 1` change
/// points, chosen uniformly among the first `steps` polls, the task polled at that point drops
/// below every other task for the rest of the run. A bug that needs `depth` specific ordering
/// constraints is found with probability at least `1 / (n * steps^(depth - 1))` for `n` tasks.
///
/// Strict priorities can starve low-priority tasks of any ordering advantage indefinitely, which
/// can stall progress checks. [Self::fair_after] switches to a random walk after a number of
/// polls (fair PCT), so liveness can still be asserted once faults heal.
pub struct Pct {
    rng: StdRng,
    depth: u64,
    priorities: BTreeMap<u128, u64>,
    change_points: BTreeSet<u64>,
    lowered: u64,
    step: u64,
    fair_after: Option<u64>,
}

impl Pct {
    /// PCT with bug depth `depth` (at least 1) over an estimated `steps` polls.
    pub fn new(seed: u64, depth: usize, steps: u64) -> Self {
        assert!(depth >= 1, "PCT depth must be at least 1");
        assert!(steps >= 1, "PCT needs at least one step");
        let mut rng = StdRng::seed_from_u64(seed);
        let change_points = points(&mut rng, depth - 1, steps);
        Self {
            rng,
            depth: u64::try_from(depth).expect("bounded depth"),
            priorities: BTreeMap::new(),
            change_points,
            lowered: 0,
            step: 0,
            fair_after: None,
        }
    }

    /// Poll batches in a uniformly random order once `steps` polls have elapsed.
    pub const fn fair_after(mut self, steps: u64) -> Self {
        self.fair_after = Some(steps);
        self
    }

    fn priority(&mut self, id: u128) -> u64 {
        let depth = self.depth;
        let rng = &mut self.rng;
        *self
            .priorities
            .entry(id)
            .or_insert_with(|| rng.random_range(depth..u64::MAX))
    }
}

impl SchedulingPolicy for Pct {
    fn order(&mut self, _: SystemTime, ready: &mut [RunnableTask]) {
        let polls = u64::try_from(ready.len()).expect("bounded batch");
        if self.fair_after.is_some_and(|after| self.step >= after) {
            ready.shuffle(&mut self.rng);
            self.step = self.step.saturating_add(polls);
            return;
        }
        for task in ready.iter() {
            self.priority(task.id());
        }
        ready.sort_by_key(|task| {
            (
                std::cmp::Reverse(self.priorities[&task.id()]),
                task.id(),
                task.occurrence(),
            )
        });

        // A change point demotes the task polled there below every task demoted before it
        for (offset, task) in (0u64..).zip(ready.iter()) {
            if self
                .change_points
                .contains(&self.step.saturating_add(offset))
            {
                self.lowered += 1;
                let low = self.depth.saturating_sub(self.lowered);
                self.priorities.insert(task.id(), low);
            }
        }
        self.step = self.step.saturating_add(polls);
    }
}

/// Delay bounding (Emmi, Qadeer, and Rakamaric, POPL 2011).
///
/// Batches are polled in a deterministic base order (task spawn order) that deviates at most
/// `delays` times: at each delay point, chosen uniformly among the first `steps` polls, the task
/// about to be polled is moved to the end of its batch. Few deviations from a deterministic
/// schedule expose many ordering bugs.
pub struct DelayBounded {
    delay_points: BTreeSet<u64>,
    step: u64,
}

impl DelayBounded {
    /// Delay bounding with at most `delays` deviations over an estimated `steps` polls.
    pub fn new(seed: u64, delays: usize, steps: u64) -> Self {
        assert!(steps >= 1, "delay bounding needs at least one step");
        let mut rng = StdRng::seed_from_u64(seed);
        Self {
            delay_points: points(&mut rng, delays, steps),
            step: 0,
        }
    }
}

impl SchedulingPolicy for DelayBounded {
    fn order(&mut self, _: SystemTime, ready: &mut [RunnableTask]) {
        ready.sort_by_key(|task| (task.id(), task.occurrence()));
        let polls = u64::try_from(ready.len()).expect("bounded batch");
        let mut position = 0;
        let mut remaining = ready.len();
        for offset in 0..polls {
            if remaining == 0 {
                break;
            }
            if self
                .delay_points
                .contains(&self.step.saturating_add(offset))
            {
                ready[position..].rotate_left(1);
            } else {
                position += 1;
            }
            remaining -= 1;
        }
        self.step = self.step.saturating_add(polls);
    }
}

/// Thread pausing, as Antithesis injects it: tasks are descheduled for short periods.
///
/// Wraps another policy, which orders each batch. Before each poll, with probability `rate`, the
/// task is instead held back for a uniform duration up to `max` while other tasks and time keep
/// moving, so it runs late relative to everything else. [Self::only] restricts pauses to tasks
/// whose label contains a pattern (for example one validator's tasks).
pub struct Pausing<P> {
    inner: P,
    rng: StdRng,
    rate: Probability,
    max: Duration,
    only: Option<String>,
}

impl<P: SchedulingPolicy> Pausing<P> {
    /// Pause tasks ordered by `inner` with probability `rate`, each for up to `max`.
    pub fn new(inner: P, seed: u64, rate: Probability, max: Duration) -> Self {
        Self {
            inner,
            rng: StdRng::seed_from_u64(seed),
            rate,
            max,
            only: None,
        }
    }

    /// Only pause tasks whose label contains `pattern`.
    pub fn only(mut self, pattern: impl Into<String>) -> Self {
        self.only = Some(pattern.into());
        self
    }
}

impl<P: SchedulingPolicy> SchedulingPolicy for Pausing<P> {
    fn order(&mut self, time: SystemTime, ready: &mut [RunnableTask]) {
        self.inner.order(time, ready);
    }

    fn hold(&mut self, time: SystemTime, task: &RunnableTask) -> Duration {
        let inner = self.inner.hold(time, task);
        if !inner.is_zero() {
            return inner;
        }
        if self
            .only
            .as_ref()
            .is_some_and(|pattern| !task.label().contains(pattern.as_str()))
        {
            return Duration::ZERO;
        }
        if self.max.is_zero() || !self.rate.sample(&mut self.rng) {
            return Duration::ZERO;
        }
        let max = u64::try_from(self.max.as_nanos()).expect("bounded pause");
        Duration::from_nanos(self.rng.random_range(1..=max))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Clock as _, Runner as _, Spawner as _, Supervisor as _, deterministic, reschedule,
    };
    use commonware_utils::{probability, sync::Mutex};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    /// Runs `tasks` tasks that each log their index once per round for `rounds` rounds, and
    /// returns the log split into rounds.
    fn rounds(policy: impl SchedulingPolicy, tasks: usize, rounds: usize) -> Vec<Vec<usize>> {
        let config = deterministic::Config::new().with_scheduling_policy(policy);
        let log = deterministic::Runner::new(config).start(|ctx| async move {
            let log = Arc::new(Mutex::new(Vec::new()));
            let handles: Vec<_> = (0..tasks)
                .map(|index| {
                    let log = log.clone();
                    ctx.child("worker").spawn(move |_| async move {
                        for _ in 0..rounds {
                            log.lock().push(index);
                            reschedule().await;
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.await.unwrap();
            }
            log.lock().clone()
        });
        log.chunks(tasks).map(<[usize]>::to_vec).collect()
    }

    #[test]
    fn test_random_walk_is_seeded() {
        assert_eq!(
            rounds(RandomWalk::new(1), 5, 4),
            rounds(RandomWalk::new(1), 5, 4)
        );
        assert_ne!(
            rounds(RandomWalk::new(1), 5, 4),
            rounds(RandomWalk::new(2), 5, 4)
        );
    }

    #[test]
    fn test_pct_without_change_points_keeps_one_order() {
        let rounds = rounds(Pct::new(7, 1, 100), 5, 6);
        assert!(
            rounds.windows(2).all(|pair| pair[0] == pair[1]),
            "{rounds:?}"
        );
    }

    #[test]
    fn test_pct_change_point_demotes_a_task() {
        // Some seed demotes a task that ran first so it runs last from then on.
        let demoted = (0..32).any(|seed| {
            let rounds = rounds(Pct::new(seed, 2, 20), 4, 8);
            let first = &rounds[0];
            let last = rounds.last().unwrap();
            first != last && last.last() == Some(&first[0])
        });
        assert!(demoted);
    }

    #[test]
    fn test_pct_fair_after_shuffles() {
        let rounds = rounds(Pct::new(3, 1, 10).fair_after(10), 5, 12);
        assert!(rounds[..2].windows(2).all(|pair| pair[0] == pair[1]));
        assert!(rounds[2..].windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn test_delay_bounded_without_delays_is_spawn_order() {
        let rounds = rounds(DelayBounded::new(0, 0, 100), 4, 3);
        assert!(
            rounds.iter().all(|round| round == &[0, 1, 2, 3]),
            "{rounds:?}"
        );
    }

    #[test]
    fn test_delay_bounded_deviates_at_most_delays_times() {
        for seed in 0..16 {
            let rounds = rounds(DelayBounded::new(seed, 2, 12), 4, 3);
            let deviations = rounds
                .iter()
                .filter(|round| round.as_slice() != [0, 1, 2, 3])
                .count();
            assert!(deviations <= 2, "seed {seed}: {rounds:?}");
        }
    }

    #[test]
    fn test_pausing_delays_only_matching_tasks() {
        let policy = Pausing::new(
            RandomWalk::new(0),
            0,
            probability!(0.5),
            Duration::from_millis(20),
        )
        .only("slow");
        let config = deterministic::Config::new().with_scheduling_policy(policy);
        let (fast, slow) = deterministic::Runner::new(config).start(|ctx| async move {
            let counters = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
            for (label, counter) in ["fast", "slow"].into_iter().zip(counters.iter().cloned()) {
                ctx.child(label).spawn(move |ctx| async move {
                    loop {
                        counter.fetch_add(1, Ordering::SeqCst);
                        ctx.sleep(Duration::from_millis(1)).await;
                    }
                });
            }
            ctx.sleep(Duration::from_secs(1)).await;
            (
                counters[0].load(Ordering::SeqCst),
                counters[1].load(Ordering::SeqCst),
            )
        });
        assert!(fast >= 990, "fast task was paused: {fast}");
        assert!(
            slow < fast / 2,
            "slow task was not paused: {slow} vs {fast}"
        );
    }

    #[test]
    fn test_held_task_ignores_wakeups_and_does_not_stall() {
        /// Holds every task labeled `held` for 100ms on each poll.
        struct HoldAll;

        impl SchedulingPolicy for HoldAll {
            fn order(&mut self, _: SystemTime, _: &mut [RunnableTask]) {}

            fn hold(&mut self, _: SystemTime, task: &RunnableTask) -> Duration {
                if task.label().contains("held") {
                    Duration::from_millis(100)
                } else {
                    Duration::ZERO
                }
            }
        }

        let config = deterministic::Config::new().with_scheduling_policy(HoldAll);
        deterministic::Runner::new(config).start(|ctx| async move {
            let (sender, mut receiver) = commonware_utils::channel::mpsc::unbounded_channel();
            let start = ctx.current();
            let received = ctx.child("held").spawn(move |ctx| async move {
                receiver.recv().await.unwrap();
                ctx.current()
            });
            ctx.sleep(Duration::from_millis(10)).await;
            sender.send(()).unwrap();
            let at = received.await.unwrap();
            // The first poll was held 100ms, and the wakeup at 10ms did not cut it short.
            assert!(
                at.duration_since(start).unwrap() >= Duration::from_millis(100),
                "{:?}",
                at.duration_since(start)
            );
        });
    }

    #[test]
    fn test_strategies_are_deterministic() {
        let run = || {
            let policy = Pausing::new(
                Pct::new(5, 3, 200).fair_after(500),
                9,
                probability!(0.1),
                Duration::from_millis(5),
            );
            let config = deterministic::Config::new().with_scheduling_policy(policy);
            deterministic::Runner::new(config).start(|ctx| async move {
                for _ in 0..4 {
                    ctx.child("worker").spawn(|ctx| async move {
                        for _ in 0..50 {
                            ctx.sleep(Duration::from_millis(1)).await;
                        }
                    });
                }
                ctx.sleep(Duration::from_millis(100)).await;
                ctx.auditor().state()
            })
        };
        assert_eq!(run(), run());
    }
}
