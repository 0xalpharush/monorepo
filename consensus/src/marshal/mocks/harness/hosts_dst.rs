//! Marshal hailstorm with validators as simulated hosts.
//!
//! Like [super::hailstorm], but each validator runs in its own [deterministic::Hosts] host and
//! a shutdown is a power loss (its unsynchronized writes tear per the storage fault
//! configuration) rather than an abort, storage operations have latency (so a crash lands in
//! the middle of writes and syncs), and some shutdowns take down the whole cluster.
//!
//! Durability oracle: a restarted validator recovers a processed height no greater than the
//! last height it processed, has every finalized block and finalization up to the recovered
//! height locally (before it is told about any), then delivers every later height to its
//! application exactly in order (no gaps) and converges to the canonical chain.

use super::*;
use commonware_runtime::deterministic::{
    FaultConfig, HostConfig, Hosts, PartialWriteMode, WriteConfig,
};
use commonware_utils::Probability;

/// Faults a host hailstorm injects.
#[derive(Clone, Copy, Debug)]
pub struct HostStorm {
    pub shutdowns: usize,
    pub interval: u64,
    pub max_down: usize,
    pub retention_rate: Probability,
    pub mode: PartialWriteMode,
    pub latency: Duration,
    /// Probability (in percent) that a shutdown takes down every validator.
    pub power_loss_pct: u8,
}

impl Default for HostStorm {
    fn default() -> Self {
        Self {
            shutdowns: 4,
            interval: 4,
            max_down: 2,
            retention_rate: probability!(0.5),
            mode: PartialWriteMode::Prefix,
            latency: Duration::from_millis(5),
            power_loss_pct: 25,
        }
    }
}

fn phase(label: &str) {
    if std::env::var("MARSHAL_DST_VERBOSE").is_ok() {
        eprintln!("phase {label}");
    }
}

fn names() -> Vec<String> {
    (0..NUM_VALIDATORS).map(|i| format!("v{i}")).collect()
}

pub fn host_hailstorm<H: TestHarness>(seed: u64, storm: HostStorm, link: Link) -> String {
    let runner = deterministic::Runner::new(
        deterministic::Config::new()
            .with_seed(seed)
            .with_timeout(Some(H::finalize_timeout()))
            .with_storage_fault_config(
                FaultConfig::default()
                    .write(WriteConfig {
                        failure_rate: probability!(0.0),
                        retention_rate: storm.retention_rate,
                        mode: storm.mode,
                    })
                    .latency(Duration::ZERO..storm.latency),
            ),
    );
    runner.start(|mut context| async move {
        let Fixture {
            participants,
            schemes,
            ..
        } = bls12381_threshold_vrf::fixture::<V, _>(&mut context, NAMESPACE, NUM_VALIDATORS);
        let mut oracle = setup_network_with_participants(
            context.child("network"),
            NZUsize!(3),
            participants.clone(),
        )
        .await;
        setup_network_links(&mut oracle, &participants, link.clone()).await;

        let names = names();
        let mut hosts = Hosts::new(&context);
        let mut validators = Vec::new();
        for (idx, validator) in participants.iter().enumerate() {
            let host = hosts.start(&names[idx], HostConfig::new());
            let setup = H::setup_validator(
                host,
                &mut oracle,
                validator.clone(),
                ConstantProvider::new(schemes[idx].clone()),
            )
            .await;
            validators.push(Some(HailstormValidator::<H> {
                application: setup.application,
                handle: ValidatorHandle {
                    mailbox: setup.mailbox,
                    extra: setup.extra,
                },
                actor_handle: setup.actor_handle,
            }));
        }

        let mut canonical = CanonicalChain::<H>::new();
        let mut parent = Sha256::hash(&[b""]);
        let mut parent_commitment = H::genesis_parent_commitment(participants.len() as u16);
        let mut target_height = 0u64;
        let max_interval = storm.interval.max(1);

        macro_rules! state {
            () => {
                HailstormState {
                    validators: &mut validators,
                    canonical: &mut canonical,
                    parent: &mut parent,
                    parent_commitment: &mut parent_commitment,
                    participants: &participants,
                    schemes: &schemes,
                }
            };
        }

        for shutdown_idx in 0..storm.shutdowns {
            let leadup = context.random_range(1..=max_interval);
            target_height += leadup;
            let active_pre = active_validator_indices(&validators);
            let power_loss = context.random_range(0..100u8) < storm.power_loss_pct;
            let mut selected: Vec<usize> = if power_loss {
                active_pre.clone()
            } else {
                let down_limit = usize::min(storm.max_down, active_pre.len().saturating_sub(1));
                let down_count = context.random_range(1..=down_limit.max(1));
                active_pre.iter().copied().sample(&mut context, down_count)
            };
            selected.sort_unstable();
            let crash_after = context.random_range(0..=leadup);
            let persisted_height = target_height - leadup + crash_after;
            phase("advance_pre");
            advance_hailstorm_to(persisted_height, &mut context, &mut state!()).await;

            // Drive the next height up to verification, then lose power mid-flight (possibly
            // after a short delay, so in-flight writes and syncs are interrupted)
            let pending = if persisted_height < target_height {
                Some(
                    drive_hailstorm_height_up_to_verify(
                        persisted_height + 1,
                        &mut context,
                        &mut state!(),
                    )
                    .await,
                )
            } else {
                None
            };
            // Usually tell everyone (including the validators about to crash) about the
            // pending finalization first, so the crash interrupts its processing
            if let Some(pending) = &pending
                && context.random_bool(0.75)
            {
                for idx in active_validator_indices(&validators) {
                    let validator = validators[idx].as_mut().unwrap();
                    H::report_finalization(
                        &mut validator.handle.mailbox,
                        pending.finalization.clone(),
                    )
                    .await;
                }
            }
            let jitter = context.random_range(0..storm.latency.as_micros() as u64 * 8);
            context.sleep(Duration::from_micros(jitter)).await;
            let mut processed = BTreeMap::new();
            for idx in selected.iter().copied() {
                let crashed = validators[idx]
                    .take()
                    .expect("selected validator should be active");
                processed.insert(idx, crashed.application.tip().map(|(height, _)| height));
            }
            if power_loss {
                hosts.crash_all();
            } else {
                for idx in selected.iter().copied() {
                    hosts.crash(&names[idx]);
                }
            }
            if let Some(pending) = pending {
                phase("finalize_pending");
                finalize_hailstorm_height(pending, &mut context, &mut state!()).await;
            }
            info!(
                seed,
                shutdown_idx,
                ?selected,
                power_loss,
                persisted_height,
                "marshal host hailstorm shutdown"
            );

            // With every validator down, the chain cannot advance until they restart
            let downtime = if power_loss {
                0
            } else {
                context.random_range(1..=max_interval)
            };
            target_height = canonical.len() as u64 + downtime;
            phase("advance_down");
            advance_hailstorm_to(target_height, &mut context, &mut state!()).await;

            for idx in selected.iter().copied() {
                phase("restart");
                let host = hosts.restart(&names[idx]);
                let restarted = H::setup_validator(
                    host,
                    &mut oracle,
                    participants[idx].clone(),
                    ConstantProvider::new(schemes[idx].clone()),
                )
                .await;
                let recovered = restarted.height.unwrap_or(Height::zero());
                let tip = processed[&idx].unwrap_or(Height::zero());
                assert!(
                    recovered <= tip,
                    "validator {idx} recovered processed height {} beyond its tip {}",
                    recovered.get(),
                    tip.get()
                );
                if std::env::var("MARSHAL_DST_VERBOSE").is_ok() {
                    eprintln!(
                        "validator {idx}: tip {} recovered {} power_loss {power_loss}",
                        tip.get(),
                        recovered.get()
                    );
                }
                // Everything processed and recovered is durable locally
                for (height, digest, _) in canonical.iter().take(recovered.get() as usize) {
                    let block = restarted.mailbox.get_block(*height).await.unwrap_or_else(|| {
                        panic!(
                            "validator {idx} recovered height {} but lost block {}",
                            recovered.get(),
                            height.get()
                        )
                    });
                    assert_eq!(block.digest(), *digest);
                    assert!(
                        restarted.mailbox.get_finalization(*height).await.is_some(),
                        "validator {idx} recovered height {} but lost finalization {}",
                        recovered.get(),
                        height.get()
                    );
                }
                let mut restarted = HailstormValidator::<H> {
                    application: restarted.application,
                    handle: ValidatorHandle {
                        mailbox: restarted.mailbox,
                        extra: restarted.extra,
                    },
                    actor_handle: restarted.actor_handle,
                };
                for (_, _, finalization) in canonical.iter().skip(recovered.get() as usize) {
                    H::report_finalization(&mut restarted.handle.mailbox, finalization.clone())
                        .await;
                }
                processed.insert(idx, Some(recovered));
                validators[idx] = Some(restarted);
            }

            for idx in selected.iter().copied() {
                let validator = validators[idx]
                    .as_ref()
                    .expect("restarted validator should be active");
                for (height, digest, finalization) in canonical.iter() {
                    let mut waited = 0u64;
                    loop {
                        let block = validator.handle.mailbox.get_block(*height).await;
                        let stored = validator.handle.mailbox.get_finalization(*height).await;
                        if let (Some(block), Some(stored)) = (&block, &stored) {
                            assert_eq!(block.digest(), *digest);
                            assert_eq!(stored.round(), finalization.round());
                            break;
                        }
                        waited += 1;
                        if waited == 3_000 {
                            let round = finalization.round();
                            let mut holders = Vec::new();
                            for (other, entry) in validators.iter().enumerate() {
                                if let Some(entry) = entry {
                                    holders.push((
                                        other,
                                        entry.handle.mailbox.get_verified(round).await.is_some(),
                                        entry.handle.mailbox.get_block(*height).await.is_some(),
                                    ));
                                }
                            }
                            panic!(
                                "validator {idx} stuck at height {} (block {}, finalization {}); (validator, verified, finalized block): {holders:?}; tip {:?}",
                                height.get(),
                                block.is_some(),
                                stored.is_some(),
                                validator.application.tip(),
                            );
                        }
                        context.sleep(Duration::from_millis(10)).await;
                    }
                }
                // Wait for the application to process the tip, then check it saw every height
                // after the recovered one, in order
                let tip = canonical.last().map(|(height, _, _)| *height);
                phase("app_tip");
                loop {
                    if validator.application.tip().map(|(height, _)| height) == tip {
                        break;
                    }
                    context.sleep(Duration::from_millis(10)).await;
                }
                let recovered = processed[&idx].unwrap().get();
                let delivered: Vec<u64> = validator
                    .application
                    .blocks()
                    .keys()
                    .map(|height| height.get())
                    .filter(|height| *height > recovered)
                    .collect();
                let expected: Vec<u64> = ((recovered + 1)..=canonical.len() as u64).collect();
                assert_eq!(
                    delivered, expected,
                    "validator {idx} skipped heights after recovering {recovered}"
                );
            }
            assert_active_validators_match_canonical(
                &validators,
                &canonical,
                participants.len() as u16,
            )
            .await;
        }
        context.auditor().state()
    })
}

fn sweep<H: TestHarness>(default: &str) {
    let seeds = std::env::var("MARSHAL_DST_SEEDS").unwrap_or_else(|_| default.into());
    let (a, b) = seeds.split_once("..").unwrap();
    let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
    let mode = match std::env::var("MARSHAL_DST_MODE").as_deref() {
        Ok("subset") => PartialWriteMode::Subset,
        _ => PartialWriteMode::Prefix,
    };
    for seed in a..b {
        eprintln!("marshal host hailstorm seed {seed}");
        host_hailstorm::<H>(
            seed,
            HostStorm {
                shutdowns: 6,
                interval: 3 + seed % 8,
                retention_rate: Probability::new(seed % 101, 100).unwrap(),
                mode,
                ..HostStorm::default()
            },
            LINK,
        );
    }
}

#[test]
fn test_host_hailstorm_smoke() {
    let run = |seed| host_hailstorm::<InlineHarness>(seed, HostStorm::default(), LINK);
    assert_eq!(run(0), run(0), "runs are deterministic");
    host_hailstorm::<CodingHarness>(1, HostStorm::default(), LINK);
}

#[test]
#[ignore]
fn test_host_hailstorm_sweep_inline() {
    sweep::<InlineHarness>("0..8");
}

#[test]
#[ignore]
fn test_host_hailstorm_sweep_deferred() {
    sweep::<DeferredHarness>("0..8");
}

#[test]
#[ignore]
fn test_host_hailstorm_sweep_standard() {
    sweep::<StandardHarness>("0..8");
}

#[test]
#[ignore]
fn test_host_hailstorm_sweep_coding() {
    sweep::<CodingHarness>("0..8");
}

/// Crashes the process owning a partition with `needle` in its name at the first write or
/// sync of it once armed, and makes no other fault.
struct CrashOnce {
    needle: &'static str,
    armed: std::sync::atomic::AtomicBool,
}

impl commonware_runtime::deterministic::FaultPolicy for CrashOnce {
    fn occurs(
        &self,
        decision: &commonware_runtime::deterministic::FaultDecision<'_>,
        _: Probability,
    ) -> bool {
        use commonware_runtime::deterministic::{FaultDraw, StorageOp};
        matches!(decision.draw, FaultDraw::Crash)
            && matches!(decision.op, StorageOp::Write | StorageOp::Sync)
            && decision.partition.starts_with("v0-")
            && decision.partition.contains(self.needle)
            && self
                .armed
                .swap(false, std::sync::atomic::Ordering::SeqCst)
    }

    fn between(
        &self,
        _: &commonware_runtime::deterministic::FaultDecision<'_>,
        range: std::ops::Range<u64>,
    ) -> u64 {
        range.start
    }
}

/// A crash while marshal syncs its two finalization archives in parallel can leave a finalized
/// block durable without its finalization. After restart, marshal re-dispatches the block from
/// the finalized-blocks archive (advancing the processed height past it) and then drops the
/// finalization for that height as "at or below the processed height", so the finalization is
/// never stored: `get_finalization` stays `None` and the latest finalized height stays behind.
fn finalization_lost_behind_block<H: TestHarness>() {
    let policy = Arc::new(CrashOnce {
        needle: "finalizations-by-height-metadata",
        armed: std::sync::atomic::AtomicBool::new(false),
    });
    let runner = deterministic::Runner::new(
        deterministic::Config::new()
            .with_seed(0)
            .with_timeout(Some(Duration::from_secs(120)))
            .with_storage_fault_config(FaultConfig::default().crash(probability!(1.0)))
            .with_storage_fault_policy(policy.clone()),
    );
    runner.start(|mut context| async move {
        let Fixture {
            participants,
            schemes,
            ..
        } = bls12381_threshold_vrf::fixture::<V, _>(&mut context, NAMESPACE, NUM_VALIDATORS);
        let mut oracle = setup_network_with_participants(
            context.child("network"),
            NZUsize!(3),
            participants.clone(),
        )
        .await;
        setup_network_links(&mut oracle, &participants, LINK).await;
        let names = names();
        let mut hosts = Hosts::new(&context);
        let mut validators = Vec::new();
        for (idx, validator) in participants.iter().enumerate() {
            let host = hosts.start(&names[idx], HostConfig::new());
            let setup = H::setup_validator(
                host,
                &mut oracle,
                validator.clone(),
                ConstantProvider::new(schemes[idx].clone()),
            )
            .await;
            validators.push(Some(HailstormValidator::<H> {
                application: setup.application,
                handle: ValidatorHandle {
                    mailbox: setup.mailbox,
                    extra: setup.extra,
                },
                actor_handle: setup.actor_handle,
            }));
        }
        let mut canonical = CanonicalChain::<H>::new();
        let mut parent = Sha256::hash(&[b""]);
        let mut parent_commitment = H::genesis_parent_commitment(participants.len() as u16);
        let mut state = HailstormState {
            validators: &mut validators,
            canonical: &mut canonical,
            parent: &mut parent,
            parent_commitment: &mut parent_commitment,
            participants: &participants,
            schemes: &schemes,
        };
        advance_hailstorm_to(4, &mut context, &mut state).await;

        // Validator 0 crashes while making height 5's finalization durable
        let pending = drive_hailstorm_height_up_to_verify(5, &mut context, &mut state).await;
        let finalization = pending.finalization.clone();
        policy.armed.store(true, std::sync::atomic::Ordering::SeqCst);
        let v0 = state.validators[0].as_mut().unwrap();
        H::report_finalization(&mut v0.handle.mailbox, finalization.clone()).await;
        while hosts.is_running(&names[0]) {
            context.sleep(Duration::from_millis(1)).await;
        }
        drop(state.validators[0].take());
        finalize_hailstorm_height(pending, &mut context, &mut state).await;

        // Restart validator 0 and tell it about the finalization again
        let host = hosts.restart(&names[0]);
        let mut restarted = H::setup_validator(
            host,
            &mut oracle,
            participants[0].clone(),
            ConstantProvider::new(schemes[0].clone()),
        )
        .await;
        H::report_finalization(&mut restarted.mailbox, finalization.clone()).await;
        context.sleep(Duration::from_secs(10)).await;
        let block = restarted.mailbox.get_block(Height::new(5)).await;
        let stored = restarted.mailbox.get_finalization(Height::new(5)).await;
        let latest = restarted.mailbox.get_info(Identifier::Latest).await;
        assert!(block.is_some(), "block 5 should be durable or recovered");
        assert!(
            stored.is_some(),
            "finalization 5 was reported again after restart but never stored \
             (block present: {}, latest: {latest:?})",
            block.is_some()
        );
    });
}

#[test]
#[ignore = "finding: a re-reported finalization at or below the processed height is never stored"]
fn test_finalization_lost_behind_block_inline() {
    finalization_lost_behind_block::<InlineHarness>();
}

#[test]
#[ignore = "finding: a re-reported finalization at or below the processed height is never stored"]
fn test_finalization_lost_behind_block_standard() {
    finalization_lost_behind_block::<StandardHarness>();
}
