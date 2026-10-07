# Deterministic runtime testing

Use the deterministic runtime for async protocol tests. It makes scheduling, time, failure injection, and state recovery reproducible. Use `commonware_utils::test_rng()` for test data; for independent streams use `TestRng::new(seed)`.

## Basic async test

```rust
#[test]
fn test_async_behavior() {
    let runner = deterministic::Runner::seeded(42);
    runner.start(|context| async move {
        let handle = context.child("worker").spawn(|context| async move {
            context.sleep(Duration::from_secs(1)).await;
        });

        context.sleep(Duration::from_millis(100)).await;

        select! {
            result = handle => { /* handle result */ },
            _ = context.sleep(Duration::from_secs(5)) => panic!("timeout"),
        }
    });
}
```

Label actors with `context.child("role")`. Use a seeded runner for repeatability and a timeout when testing a bounded operation:

```rust
let cfg = deterministic::Config::new()
    .with_seed(seed)
    .with_timeout(Some(Duration::from_secs(30)));
let runner = deterministic::Runner::new(cfg);
```

## Recovery

Use `start_and_recover` to exercise unclean shutdown and restart paths:

```rust
let mut checkpoint = None;
loop {
    let runner = if let Some(checkpoint) = checkpoint.take() {
        deterministic::Runner::from(checkpoint)
    } else {
        deterministic::Runner::timed(Duration::from_secs(30))
    };

    let (complete, next_checkpoint) = runner.start_and_recover(f);
    if complete {
        break;
    }
    checkpoint = Some(next_checkpoint);
}
```

## Harness faults

Run each simulated node as a host: `context.host(name, HostConfig)` (or a `deterministic::Hosts` manager, which restarts hosts by name) gives it its own storage namespace (applications keep plain partition names), metrics registry, IP, clock, and storage latency, and crashing it affects exactly its storage. `context.process(label, owned_partitions)` starts a process that owns the partitions a selector picks, for finer-grained faults (such as one component of a host). Group processes that fail together in a `deterministic::Zone` to crash, pause, or resume them at the same instant, and call `Process::wipe` (or `Zone::wipe`) to model total data loss before restarting a node. For network faults, install a `deterministic::Partitions` (with `Config::with_network_policy` for sockets, giving each process an IP with `Process::set_ip`, or `Oracle::set_policy` for `p2p::simulated`) and change it mid-run, or drive it with `deterministic::Swizzle`.

## Verification checklist

- Check determinism with `context.auditor().state()` when relevant.
- Monitor progress with supervisors or metrics rather than time alone.
- For shutdown, assert the task-prefix count becomes non-zero before shutdown and zero afterward.
- Run a scenario twice with the same seed when its state is meant to be deterministic.
- Include recovery cases when the changed component has those boundaries.
