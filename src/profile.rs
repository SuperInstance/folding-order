use crate::types::*;
use std::collections::HashMap;
use std::time::Instant;

/// Auto-profile hardware by running microbenchmarks.
///
/// Calibrates:
/// 1. Baseline cycles/op for each operation
/// 2. Thermal coefficient from sustained load drift
/// 3. Utilization baseline (mean, std_dev) using unbiased sample variance
pub fn auto_profile() -> HardwareProfile {
    eprintln!("Profiling hardware...");

    let mut baseline_cycles = HashMap::new();
    let mut thermal_coefficients = HashMap::new();
    let mut utilization_baselines = HashMap::new();

    let reference_temp = read_cpu_temp_mc();

    for op in ALL_OPERATIONS {
        let prec = op.default_precision();
        let key = format!("{}/{}", op, prec);

        eprintln!("  Benchmarking {}...", op);

        // Phase 1: Baseline measurement
        let (baseline, samples) = benchmark_operation(op, 100);
        baseline_cycles.insert(key.clone(), baseline);

        // Phase 2: Utilization baseline using unbiased sample variance
        // of normalized deviations (replicating stages 1-2 on calibration data)
        if samples.len() > 1 {
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            // Unbiased sample variance (Bessel's correction)
            let variance = samples
                .iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>()
                / (samples.len() - 1) as f64;
            let std_dev = variance.sqrt().max(0.001);
            // Store as coefficient of variation (relative std dev)
            utilization_baselines.insert(key.clone(), (0.0, std_dev / mean));
        } else {
            utilization_baselines.insert(key.clone(), (0.0, 0.01));
        }

        // Phase 3: Thermal calibration
        let thermal_coeff = calibrate_thermal(op);
        thermal_coefficients.insert(key, thermal_coeff);
    }

    let cpu_model = get_cpu_model();

    HardwareProfile {
        cpu_model,
        baseline_cycles,
        thermal_coefficients,
        utilization_baselines,
        reference_temp_mc: reference_temp,
        calibrated_at: chrono_now(),
    }
}

/// Read CPU temperature from /sys/class/thermal/ on Linux.
/// Returns None if unavailable (non-Linux, no thermal zone, etc.)
fn read_cpu_temp_mc() -> Option<i64> {
    // Try common thermal zone paths
    for i in 0..10 {
        let path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
        if let Ok(data) = std::fs::read_to_string(&path) {
            if let Ok(temp) = data.trim().parse::<i64>() {
                return Some(temp);
            }
        }
    }
    None
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
    format!("{}", d.as_secs())
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

    let mut early_cpo = 0.0;
    let mut late_cpo = 0.0;
    let mut phase = 0;
    let mut count = 0u64;
    let mut total_cycles = 0u64;

    while start.elapsed() < duration {
        let (ops, cycles) = run_op_batch(op, 50000);
        total_cycles += cycles;
        count += ops;

        if start.elapsed() >= duration / 2 && phase == 0 {
            early_cpo = if count > 0 {
                total_cycles as f64 / count as f64
            } else {
                1.0
            };
            count = 0;
            total_cycles = 0;
            phase = 1;
        }
    }

    late_cpo = if count > 0 {
        total_cycles as f64 / count as f64
    } else {
        early_cpo
    };

    // Thermal coefficient = relative slowdown per second
    if early_cpo > 0.0 {
        let drift = (late_cpo - early_cpo) / early_cpo;
        drift.abs() / duration.as_secs_f64()
    } else {
        0.001
    }
}

/// Run a batch of operations and return (op_count, elapsed_ns)
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

/// Cycle counter — uses RDTSC on x86_64, nanoseconds as fallback.
/// NOTE: On x86_64 this gives actual CPU cycles.
/// On other architectures, this gives nanoseconds (different unit!).
#[inline]
fn rdtsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // RDTSC for cycle counter
        // Note: for precise serialization, use `core::arch::x86_64::_mm_mfence`
        // or serialize with a dummy volatile read before this call.
        unsafe { std::arch::x86_64::_rdtsc() }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let now = std::time::Instant::now();
        // Approximate: use nanoseconds as a stand-in for cycles
        // Caller should be aware this is not cycle-accurate on non-x86
        now.elapsed().as_nanos() as u64
    }
}

fn run_int8_batch(n: u64) {
    let mut acc: i8 = 0;
    for i in 0..n {
        acc = acc.wrapping_add((i & 0xFF) as i8);
        acc = acc.wrapping_mul(3i8);
    }
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
        re = a * c - b * d;
        im = b * c + a * d - b * d;
    }
    std::hint::black_box((re, im));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_benchmark_runs() {
        let (baseline, samples) = benchmark_operation(Operation::Int32Scalar, 50);
        assert!(baseline > 0.0, "Baseline should be positive");
        assert!(!samples.is_empty(), "Should have samples");
    }

    #[test]
    fn test_rdtsc_is_monotonic() {
        let a = rdtsc();
        // Small spin to ensure time passes
        for _ in 0..1000 {
            std::hint::black_box(42u64);
        }
        let b = rdtsc();
        assert!(b >= a, "Cycle counter should be monotonic");
    }

    #[test]
    fn test_read_cpu_temp() {
        // This may return None on non-Linux or in containers — that's fine
        let _ = read_cpu_temp_mc();
    }

    #[test]
    fn test_profile_completes() {
        let profile = auto_profile();
        assert!(!profile.cpu_model.is_empty() || profile.cpu_model == "Unknown CPU");
        assert_eq!(profile.baseline_cycles.len(), ALL_OPERATIONS.len());
        assert_eq!(profile.thermal_coefficients.len(), ALL_OPERATIONS.len());
        assert_eq!(profile.utilization_baselines.len(), ALL_OPERATIONS.len());
    }
}
