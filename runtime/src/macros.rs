//! Macros shared across runtime implementations.

/// Prepare metrics for a spawned task.
///
/// Returns a `(Label, MetricHandle)` pair for tracking spawned tasks.
///
/// The `Label` identifies the task in the metrics registry and the
/// `MetricHandle` immediately increments the `tasks_running` gauge for that
/// label. Call `MetricHandle::finish` once the task completes to decrement the
/// gauge.
#[cfg(not(any(
    commonware_stability_GAMMA,
    commonware_stability_DELTA,
    commonware_stability_EPSILON,
    commonware_stability_RESERVED
)))] // BETA
#[macro_export]
macro_rules! spawn_metrics {
    // Handle future tasks
    ($ctx:ident) => {
        $crate::spawn_metrics!(
            $crate::telemetry::metrics::task::Label::task(
                $ctx.name.clone(),
                $ctx.execution,
            ),
            @make $ctx
        )
    };

    // Increment the number of spawned tasks and return a metrics tracker that
    // keeps the running tasks gauge accurate
    ($label:expr, @make $ctx:ident) => {{
        let label = $label;
        let metrics = $ctx.metrics();
        metrics.tasks_spawned.get_or_create(&label).inc();
        let metric =
            $crate::utils::MetricHandle::new(metrics.tasks_running.get_or_create(&label).clone());
        (label, metric)
    }};
}

/// Whether a code-level fault point fires, in the style of FoundationDB's `BUGGIFY`.
///
/// `buggify!(context)` evaluates [`Supervisor::buggify`](crate::Supervisor::buggify) for the
/// invocation's source location, and `buggify!(context, rate)` additionally requests the
/// [`Probability`](commonware_utils::Probability) `rate` with which an enabled site fires. Use a
/// site to take a rare but legal path, such as flushing early, forcing a slow path, or
/// shortening a timeout. Forcing the path must never violate the surrounding code's contract.
///
/// Outside simulation, a site never fires. In the deterministic runtime, sites are off unless
/// configured with `deterministic::Config::with_buggify`.
///
/// # Examples
///
/// ```rust
/// use commonware_runtime::{buggify, deterministic, Runner};
///
/// deterministic::Runner::default().start(|context| async move {
///     // Sites are off by default.
///     assert!(!buggify!(context));
/// });
/// ```
#[cfg(not(any(
    commonware_stability_GAMMA,
    commonware_stability_DELTA,
    commonware_stability_EPSILON,
    commonware_stability_RESERVED
)))] // BETA
#[macro_export]
macro_rules! buggify {
    ($ctx:expr) => {
        $crate::buggify!($ctx, @rate ::core::option::Option::None)
    };
    ($ctx:expr, @rate $rate:expr) => {{
        use $crate::Supervisor as _;
        const SITE: $crate::Site = $crate::Site {
            file: ::core::file!(),
            line: ::core::line!(),
            column: ::core::column!(),
        };
        ($ctx).buggify(&SITE, $rate)
    }};
    ($ctx:expr, $rate:expr) => {
        $crate::buggify!($ctx, @rate ::core::option::Option::Some($rate))
    };
}
