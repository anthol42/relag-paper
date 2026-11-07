//! Temporary build tracing, for diagnosing parallel-build recall.
//!
//! Enabled when the `RELAG_TRACE` environment variable is set at the start of a build;
//! the events are then written to that path as JSON lines when the build finishes.
//!
//! Events:
//! - `{"ev":"insert","node","layer","t0","t1","entry":[node,layer]|null,
//!    "l0_start":[nodes],"visits":[[node,distance]],"selected":[nodes]}`
//! - `{"ev":"prune","node","by","t","removed":[nodes]}` (layer 0 only)
//!
//! `t0`, `t1` and `t` are ticks of one global counter shared by all threads.

use std::cell::RefCell;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static CLOCK: AtomicU64 = AtomicU64::new(0);
static EVENTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

thread_local! {
    static VISITS: RefCell<Vec<(u32, f32)>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

pub(super) fn tick() -> u64 {
    CLOCK.fetch_add(1, Ordering::SeqCst)
}

/// Record a distance computed by the current thread's layer-0 search.
pub(super) fn record_visit(node: u32, distance: f32) {
    VISITS.with(|v| v.borrow_mut().push((node, distance)));
}

/// Return and clear the current thread's recorded visits.
pub(super) fn take_visits() -> Vec<(u32, f32)> {
    VISITS.with(|v| std::mem::take(&mut *v.borrow_mut()))
}

pub(super) fn record_insert(
    node: u32, layer: usize, t0: u64, entry: Option<(u32, usize)>,
    l0_start: &[u32], visits: &[(u32, f32)], selected: &[u32],
) {
    let entry = entry.map_or("null".to_string(), |(n, l)| format!("[{n},{l}]"));
    let visits: Vec<String> = visits.iter().map(|(n, d)| format!("[{n},{d}]")).collect();
    let line = format!(
        r#"{{"ev":"insert","node":{node},"layer":{layer},"t0":{t0},"t1":{},"entry":{entry},"l0_start":{l0_start:?},"visits":[{}],"selected":{selected:?}}}"#,
        tick(), visits.join(","),
    );
    EVENTS.lock().unwrap().push(line);
}

pub(super) fn record_prune(node: u32, by: u32, removed: &[u32]) {
    let line = format!(
        r#"{{"ev":"prune","node":{node},"by":{by},"t":{},"removed":{removed:?}}}"#,
        tick(),
    );
    EVENTS.lock().unwrap().push(line);
}

/// Enable tracing for the coming build if `RELAG_TRACE` is set.
pub(super) fn start() {
    if std::env::var_os("RELAG_TRACE").is_some() {
        CLOCK.store(0, Ordering::SeqCst);
        EVENTS.lock().unwrap().clear();
        ENABLED.store(true, Ordering::SeqCst);
    }
}

/// Write the recorded events to `RELAG_TRACE` and disable tracing.
pub(super) fn finish() {
    if !ENABLED.swap(false, Ordering::SeqCst) {
        return;
    }
    let path = std::env::var_os("RELAG_TRACE").expect("RELAG_TRACE was set at start()");
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).expect("cannot create trace file"));
    for line in EVENTS.lock().unwrap().drain(..) {
        writeln!(file, "{line}").expect("cannot write trace file");
    }
}
