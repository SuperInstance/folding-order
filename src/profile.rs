use crate::types::*;
use std::collections::HashMap;
use std::time::Instant;

/// Auto-profile hardware by running microbenchmarks
pub fn auto_profile() -> HardwareProfile {
    eprintln!("Profiling hardware...");

    let mut baseline_cycles = HashMap::new();
    let mut thermal_coefficients = HashMap::new();
    let mut utilization_baselines = HashMap::new();

    for op in ALL_OPERATIONS {
        let prec = op.default_precision();
        let key = format!("{}/{}", op, prec);

        eprintln!("  Benchmarking {}...", op);

        // Phase 1: Quick baseline (100ms warmup + measurement)
        let (baseline, samples) = benchmark_operation(op, 100);
        baseline_cycles.insert(key.clone(), baseline);

        // Phase 2: Compute utilization baseline (mean, std_dev of cycles_per_op)
        if !samples.is_empty() {
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let variance = samples
                .iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>()
                / samples.len() as f64;
            let std_dev = variance.sqrt();
            utilization_baselines.insert(key.clone(), (0.0, if std_dev == 0.0 { 0.01 } else { std_dev / mean }));
        } else {
            utilization_baselines.insert(key.clone(), (0.0, 0.01));
        }

        // Phase 3: Thermal calibration (5s sustained load, measure drift)
        let thermal_coeff = calibrate_thermal(op);
        thermal_coefficients.insert(key, thermal_coeff);
    }

    let cpu_model = get_cpu_model();

    HardwareProfile {
        cpu_model,
        baseline_cycles,
        thermal_coefficients,
        utilization_baselines,
        calibrated_at: chrono_now(),
    }
}

fn get_cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown CPU".into())
}

fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s since epoch", d.as_secs())
}

/// Benchmark a single operation, returns (baseline_cycles_per_op, sample_distribution)
fn benchmark_operation(op: Operation, duration_ms: u64) -> (f64, Vec<f64>) {
    let target = std::time::Duration::from_millis(duration_ms);
    let mut total_ops = 0u64;
    let mut total_cycles = 0u64;
    let mut samples = Vec::new();

    let start = Instant::now();
    while start.elapsed() < target {
        let (ops, cycles) = run_op_batch(op, 10000);
        total_ops += ops;
        total_cycles += cycles;
        if ops > 0 {
            samples.push(cycles as f64 / ops as f64);
        }
    }

    let baseline = if total_ops > 0 {
        total_cycles as f64 / total_ops as f64
    } else {
        1.0
    };

    (baseline, samples)
}

/// Calibrate thermal coefficient by running sustained load and measuring drift
fn calibrate_thermal(op: Operation) -> f64 {
    let duration = std::time::Duration::from_secs(3);
    let start = Instant::now();

    let mut early_cycles_per_op = 0.0;
    let mut late_cycles_per_op = 0.0;
    let mut phase = 0;
    let mut count = 0u64;
    let mut total_cycles = 0u64;

    while start.elapsed() < duration {
        let (ops, cycles) = run_op_batch(op, 50000);
        total_cycles += cycles;
        count += ops;

        if start.elapsed() < duration / 2 {
            // First half
        } else if phase == 0 {
            early_cycles_per_op = if count > 0 {
                total_cycles as f64 / count as f64
            } else {
                1.0
            };
            count = 0;
            total_cycles = 0;
            phase = 1;
        }
    }

    late_cycles_per_op = if count > 0 {
        total_cycles as f64 / count as f64
    } else {
        early_cycles_per_op
    };

    // Thermal coefficient = relative slowdown over the period
    if early_cycles_per_op > 0.0 {
        let drift = (late_cycles_per_op - early_cycles_per_op) / early_cycles_per_op;
        drift.abs() / duration.as_secs_f64()
    } else {
        0.001
    }
}

/// Run a batch of operations and return (op_count, cycles)
fn run_op_batch(op: Operation, batch_size: u64) -> (u64, u64) {
    let start = rdtsc();
    match op {
        Operation::Int8PackedVnni => run_int8_batch(batch_size),
        Operation::Int32Scalar => run_int32_batch(batch_size),
        Operation::Fp64Norm => run_fp64_batch(batch_size),
        Operation::EisensteinMultiply => run_eisenstein_batch(batch_size),
    }
    let end = rdtsc();
    (batch_size, end.saturating_sub(start))
}

#[inline]
fn rdtsc() -> u64 {
    // Use std::time fallback since rdtsc via asm may not be available on all platforms
    let now = std::time::SystemTime::now();
    let dur = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    // Convert to a cycle-like approximation (nanoseconds * approximate freq)
    dur.as_nanos() as u64
}

fn run_int8_batch(n: u64) {
    let mut acc: i8 = 0;
    for i in 0..n {
        acc = acc.wrapping_add((i & 0xFF) as i8);
        acc = acc.wrapping_mul(3i8);
    }
    // Prevent optimization
    std::hint::black_box(acc);
}

fn run_int32_batch(n: u64) {
    let mut acc: i32 = 0;
    for i in 0..n {
        acc = acc.wrapping_add(i as i32);
        acc = acc.wrapping_mul(7);
    }
    std::hint::black_box(acc);
}

fn run_fp64_batch(n: u64) {
    let mut acc: f64 = 1.0;
    for i in 0..n {
        let x = i as f64;
        acc += (x * x + 1.0).sqrt().recip();
    }
    std::hint::black_box(acc);
}

/// Eisenstein integer multiply: (a + bω)(c + dω) where ω = e^(2πi/3)
/// Result: (ac - bd) + (bc + ad - bd)ω
fn run_eisenstein_batch(n: u64) {
    let mut re = 1i64;
    let mut im = 1i64;
    for i in 0..n {
        let a = re;
        let b = im;
        let c = (i as i64).wrapping_rem(97);
        let d = (i as i64).wrapping_rem(31);
        // (a + bω)(c + dω) = (ac - bd) + (bc + ad - bd)ω
        re = a * c - b * d;
        im = b * c + a * d - b * d;
    }
    std::hint::black_box((re, im));
}
