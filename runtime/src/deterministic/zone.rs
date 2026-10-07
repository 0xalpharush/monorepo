//! Correlated process failures and total data loss.
//!
//! A [Zone] groups the [Process]es that fail together (a rack, an availability zone, a power
//! domain) so a test can crash, pause, or resume all of them at the same instant.
//! [Process::wipe] (and [Zone::wipe]) model losing a host's disk entirely: the process crashes and
//! every partition it owned is erased, so a process restarted in its place starts empty and must
//! recover its state from its peers.

use super::Process;

impl Process {
    /// Crash the process (unless it already crashed) and erase every partition it owned, as
    /// replacing a failed host's disk would.
    ///
    /// Handles opened before the crash can no longer publish (see [Self::crash]), so a process
    /// restarted on the same partitions opens them empty. Call this before restarting the
    /// process: erasing a partition a live process has opened since the crash panics.
    pub fn wipe(self) {
        self.lose();
    }

    /// Crash the process unless it already crashed.
    fn halt(&self) {
        self.executor().crash_process(&self.host);
    }

    /// Crash the process and erase its partitions.
    fn lose(&self) {
        self.halt();
        let erased = self
            .host
            .storage
            .inner()
            .inner()
            .inner()
            .erase_partitions(&|partition| (self.host.partitions)(partition));
        self.executor().auditor.event(b"wipe_process", |hasher| {
            for partition in &erased {
                hasher.update(partition.as_bytes());
            }
        });
    }
}

/// A group of [Process]es that fail together.
///
/// Every operation applies to all members at the same instant: no task of any member runs
/// between the first member's fault and the last's. Members stay in the zone after they crash
/// (crashing or wiping a crashed member only erases what it still owns), so a zone can be
/// crashed and then wiped. Start replacements for crashed members as new processes, in a new
/// zone or added to this one.
#[derive(Default)]
pub struct Zone {
    members: Vec<Process>,
}

impl Zone {
    /// An empty zone.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `process` to the zone.
    pub fn add(&mut self, process: Process) {
        self.members.push(process);
    }

    /// The number of processes in the zone.
    pub const fn len(&self) -> usize {
        self.members.len()
    }

    /// Whether the zone has no processes.
    pub const fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// The zone's processes, in the order they were added, to fault individually (for example,
    /// to [Process::wipe] one member of a crashed zone).
    pub fn into_processes(self) -> Vec<Process> {
        self.members
    }

    /// Crash every live process of the zone at once, as a power loss across the zone would.
    ///
    /// Every member is crashed (see [Process::crash]) before this returns, so none of their tasks
    /// runs again, whatever order the runtime would have polled them in.
    pub fn crash(&self) {
        for member in &self.members {
            member.halt();
        }
    }

    /// Whether every process of the zone has crashed.
    pub fn crashed(&self) -> bool {
        self.members.iter().all(Process::crashed)
    }

    /// Pause every process of the zone at once (see [Process::pause]).
    pub fn pause(&self) {
        for member in &self.members {
            member.pause();
        }
    }

    /// Resume every process of the zone at once (see [Process::resume]).
    pub fn resume(&self) {
        for member in &self.members {
            member.resume();
        }
    }

    /// Crash every process of the zone at once and erase every partition each owned (see
    /// [Process::wipe]).
    pub fn wipe(&self) {
        self.crash();
        for member in &self.members {
            member.lose();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Blob as _, Clock as _, Runner as _, Spawner as _, Storage as _, WriteOptions,
        deterministic,
    };
    use commonware_utils::sync::Mutex;
    use std::{sync::Arc, time::Duration};

    /// Spawns a task in each of `count` processes of a zone that counts how often each one ticks.
    fn start_zone(
        context: &deterministic::Context,
        count: usize,
    ) -> (Zone, Arc<Mutex<Vec<u64>>>) {
        let ticks = Arc::new(Mutex::new(vec![0u64; count]));
        let mut zone = Zone::new();
        for i in 0..count {
            let (process_ctx, process) = context.process("member", |_| false);
            let ticks = ticks.clone();
            process_ctx.spawn(move |context| async move {
                loop {
                    context.sleep(Duration::from_millis(1)).await;
                    ticks.lock()[i] += 1;
                }
            });
            zone.add(process);
        }
        (zone, ticks)
    }

    #[test]
    fn test_zone_crash_is_atomic() {
        // Whatever order the runtime polls tasks in, no member ticks after the zone crashes
        for seed in 0..32 {
            deterministic::Runner::seeded(seed).start(|context| async move {
                let (zone, ticks) = start_zone(&context, 4);
                context.sleep(Duration::from_millis(10)).await;
                zone.crash();
                assert!(zone.crashed());
                let at_crash = ticks.lock().clone();
                context.sleep(Duration::from_millis(10)).await;
                assert_eq!(*ticks.lock(), at_crash, "seed {seed}");
                assert!(at_crash.iter().all(|ticks| *ticks > 0));
            });
        }
    }

    #[test]
    fn test_zone_pause_and_resume() {
        deterministic::Runner::default().start(|context| async move {
            let (zone, ticks) = start_zone(&context, 3);
            context.sleep(Duration::from_millis(10)).await;
            zone.pause();
            let at_pause = ticks.lock().clone();
            context.sleep(Duration::from_millis(50)).await;
            assert_eq!(*ticks.lock(), at_pause);
            zone.resume();
            context.sleep(Duration::from_millis(10)).await;
            assert!(
                ticks
                    .lock()
                    .iter()
                    .zip(&at_pause)
                    .all(|(now, then)| now > then)
            );
            assert!(!zone.crashed());
        });
    }

    #[test]
    fn test_wipe_loses_synced_and_unsynced_data() {
        let run = |seed| {
            deterministic::Runner::seeded(seed).start(|context| async move {
                // A node writes synced and unsynced data to two partitions; a bystander writes too
                let owned = |partition: &str| partition.starts_with("node_");
                let (node, process) = context.process("node", owned);
                for partition in ["node_a", "node_b"] {
                    let (blob, _) = node.open(partition, b"blob").await.unwrap();
                    blob.write_at(0, b"synced".to_vec(), WriteOptions::default()).await.unwrap();
                    blob.sync().await.unwrap();
                    blob.write_at(6, b"unsynced".to_vec(), WriteOptions::default()).await.unwrap();
                    drop(blob);
                }
                let (other, _bystander) = context.process("other", |p| p == "other");
                let (blob, _) = other.open("other", b"blob").await.unwrap();
                blob.write_at(0, b"kept".to_vec(), WriteOptions::default()).await.unwrap();
                blob.sync().await.unwrap();
                drop(blob);

                process.wipe();

                // The restarted node starts empty; the bystander keeps its data
                let (node, _process) = context.process("node", owned);
                for partition in ["node_a", "node_b"] {
                    assert!(node.scan(partition).await.is_err());
                    let (_, len) = node.open(partition, b"blob").await.unwrap();
                    assert_eq!(len, 0);
                }
                let (blob, len) = other.open("other", b"blob").await.unwrap();
                assert_eq!(len, 4);
                drop(blob);
                context.auditor().state()
            })
        };
        assert_eq!(run(3), run(3));
    }

    #[test]
    fn test_zone_wipe_after_crash() {
        deterministic::Runner::default().start(|context| async move {
            let mut zone = Zone::new();
            for partition in ["zone_0", "zone_1"] {
                let (node, process) = context.process("node", move |p| p == partition);
                let (blob, _) = node.open(partition, b"blob").await.unwrap();
                blob.write_at(0, b"data".to_vec(), WriteOptions::default()).await.unwrap();
                blob.sync().await.unwrap();
                drop(blob);
                zone.add(process);
            }
            zone.crash();
            assert_eq!(context.scan("zone_0").await.unwrap().len(), 1);
            zone.wipe();
            assert!(context.scan("zone_0").await.is_err());
            assert!(context.scan("zone_1").await.is_err());
        });
    }

    #[test]
    #[should_panic(expected = "was opened since its owner crashed")]
    fn test_wipe_rejects_reopened_partition() {
        deterministic::Runner::default().start(|context| async move {
            let (node, process) = context.process("node", |p| p == "data");
            let (blob, _) = node.open("data", b"blob").await.unwrap();
            blob.write_at(0, b"data".to_vec(), WriteOptions::default()).await.unwrap();
            blob.sync().await.unwrap();
            drop(blob);
            let mut zone = Zone::new();
            zone.add(process);
            zone.crash();

            // A restart reopens the partition before the zone is wiped
            let (restarted, _process) = context.process("node", |p| p == "data");
            let _blob = restarted.open("data", b"blob").await.unwrap();
            zone.wipe();
        });
    }
}
