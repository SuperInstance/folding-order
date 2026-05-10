#![allow(dead_code)]
mod types;
mod fold;
mod profile;
mod detector;

use types::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "profile" => cmd_profile(),
        "monitor" => cmd_monitor(),
        "analyze" => {
            if args.len() < 3 {
                eprintln!("Usage: folding-order analyze <file.json>");
                return;
            }
            cmd_analyze(&args[2]);
        }
        "benchmark" => cmd_benchmark(),
        "calibrate" => cmd_calibrate(),
        "inject-anomaly" => cmd_inject_anomaly(),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("folding-order — 5-stage RG flow for temporal anomaly detection");
    println!();
    println!("Commands:");
    println!("  profile          Auto-profile hardware, save to ~/.folding-order/");
    println!("  monitor          Live constraint monitoring with anomaly detection");
    println!("  analyze FILE     Analyze a JSON log of measurements");
    println!("  benchmark        Run folding pipeline benchmark");
    println!("  calibrate        Recalibrate thermal coefficients");
    println!("  inject-anomaly   Inject simulated anomaly for testing");
}

fn cmd_profile() {
    eprintln!("=== Folding Order: Hardware Profiling ===");
    let profile = profile::auto_profile();
    let path = HardwareProfile::profile_path();
    match profile.save(path.to_str().unwrap()) {
        Ok(()) => {
            println!("Profile saved to {}", path.display());
            println!("CPU: {}", profile.cpu_model);
            if let Some(temp) = profile.reference_temp_mc {
                println!("Reference temp: {:.1}°C", temp as f64 / 1000.0);
            }
            println!("Operations profiled: {}", profile.baseline_cycles.len());
            for (key, cycles) in &profile.baseline_cycles {
                println!("  {}: {:.2} cycles/op", key, cycles);
            }
        }
        Err(e) => eprintln!("Error saving profile: {}", e),
    }
}

fn cmd_monitor() {
    let profile_path = HardwareProfile::profile_path();
    let profile = if profile_path.exists() {
        match HardwareProfile::load(profile_path.to_str().unwrap()) {
            Ok(p) => {
                println!("Loaded profile from {}", profile_path.display());
                p
            }
            Err(e) => {
                eprintln!("Error loading profile: {}. Re-profiling...", e);
                let p = profile::auto_profile();
                let _ = p.save(profile_path.to_str().unwrap());
                p
            }
        }
    } else {
        println!("No profile found. Running auto-profile...");
        let p = profile::auto_profile();
        let _ = p.save(profile_path.to_str().unwrap());
        p
    };

    println!("=== Folding Order: Live Monitor ===");
    println!("Monitoring constraint operations... (Ctrl+C to stop)");
    println!();

    let mut detector = detector::AnomalyDetector::new(profile);
    let mut last_stats = Instant::now();

    loop {
        for op in ALL_OPERATIONS {
            let prec = op.default_precision();
            let (ops, cycles) = run_monitoring_batch(op);

            let temp_mc = read_cpu_temp_inline();

            let measurement = RawMeasurement {
                timestamp_ns: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64,
                operation: op,
                cycles,
                precision: prec,
                value: ops as i64,
                op_count: ops,
                temp_mc,
            };

            if let Some(anomaly) = detector.feed(measurement) {
                println!(
                    "⚠ ANOMALY: {} score={:.4} — {}",
                    anomaly.measurement.operation,
                    anomaly.anomaly_score,
                    anomaly.description
                );
            }

            if last_stats.elapsed() > std::time::Duration::from_secs(2) {
                let stats = detector.stats();
                print!(
                    "\r[{}] {} samples, {} anomalies ({:.2}%)   ",
                    op,
                    stats.total_measurements,
                    stats.anomalies_detected,
                    stats.anomaly_rate * 100.0
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
                last_stats = Instant::now();
            }
        }
    }
}

fn cmd_analyze(path: &str) {
    println!("=== Folding Order: Analyzing {} ===", path);
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return;
        }
    };

    let measurements: Vec<RawMeasurement> = match serde_json::from_str(&data) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing JSON: {}", e);
            return;
        }
    };

    println!("Loaded {} measurements", measurements.len());

    let profile_path = HardwareProfile::profile_path();
    let profile = if profile_path.exists() {
        HardwareProfile::load(profile_path.to_str().unwrap()).unwrap_or_else(|_| profile::auto_profile())
    } else {
        profile::auto_profile()
    };

    let results = fold::fold_verbose(&measurements, &profile);
    let anomalies = results.iter().filter(|r| r.decision.is_anomalous()).count();

    println!("\nResults: {} total, {} anomalies ({:.1}%)",
        results.len(),
        anomalies,
        if results.is_empty() { 0.0 } else { anomalies as f64 / results.len() as f64 * 100.0 }
    );

    println!("\n{:<20} {:>12} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "Operation", "Cyc/Op", "Deviation", "Norm.Dev", "Z-Score", "Score", "Decision");
    println!("{}", "─".repeat(80));

    for r in &results {
        println!("{:<20} {:>12.2} {:>10.4} {:>10.4} {:>8.2} {:>8.4} {:>8}",
            format!("{}", r.measurement.operation),
            r.cycles_per_op,
            r.deviation,
            r.normalized_deviation,
            r.z_score,
            r.anomaly_score,
            r.decision,
        );
    }
}

fn cmd_benchmark() {
    println!("=== Folding Order: Pipeline Benchmark ===");
    let profile = profile::auto_profile();

    let measurements: Vec<RawMeasurement> = (0..10000)
        .map(|i| {
            let op = ALL_OPERATIONS[i % ALL_OPERATIONS.len()];
            let prec = op.default_precision();
            let baseline = profile.get_baseline_cycles(op, prec);
            let cycles = (baseline * 1000.0) as u64 + (i % 10) as u64;

            RawMeasurement {
                timestamp_ns: i as u64 * 1_000_000,
                operation: op,
                cycles,
                precision: prec,
                value: 1000,
                op_count: 1000,
                temp_mc: None,
            }
        })
        .collect();

    let start = Instant::now();
    let decisions = fold::fold(&measurements, &profile);
    let elapsed = start.elapsed();

    let anomalies = decisions.iter().filter(|d| d.is_anomalous()).count();

    println!("Folded {} measurements in {:?}", measurements.len(), elapsed);
    println!(
        "Throughput: {:.0} measurements/sec",
        measurements.len() as f64 / elapsed.as_secs_f64()
    );
    println!("Anomalies: {}/{}", anomalies, decisions.len());

    // Benchmark detector
    let mut det = detector::AnomalyDetector::new(profile);
    let start = Instant::now();
    for m in &measurements {
        det.feed(m.clone());
    }
    let elapsed = start.elapsed();
    let stats = det.stats();

    println!("\nDetector: {} measurements in {:?}", stats.total_measurements, elapsed);
    println!(
        "Detector throughput: {:.0} measurements/sec",
        stats.total_measurements as f64 / elapsed.as_secs_f64()
    );
}

fn cmd_calibrate() {
    println!("=== Folding Order: Thermal Calibration ===");
    println!("Running sustained load to measure thermal drift...");
    println!("(This takes ~15 seconds)");

    let profile = profile::auto_profile();
    let path = HardwareProfile::profile_path();
    match profile.save(path.to_str().unwrap()) {
        Ok(()) => {
            println!("Calibrated profile saved to {}", path.display());
            if let Some(temp) = profile.reference_temp_mc {
                println!("Reference temperature: {:.1}°C", temp as f64 / 1000.0);
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn cmd_inject_anomaly() {
    println!("=== Folding Order: Anomaly Injection Test ===");

    let profile_path = HardwareProfile::profile_path();
    let profile = if profile_path.exists() {
        HardwareProfile::load(profile_path.to_str().unwrap()).unwrap_or_else(|_| profile::auto_profile())
    } else {
        let p = profile::auto_profile();
        let _ = p.save(profile_path.to_str().unwrap());
        p
    };

    let mut detector = detector::AnomalyDetector::new(profile.clone());

    // Phase 1: Normal measurements
    println!("\nPhase 1: Feeding normal measurements...");
    for i in 0..100 {
        let op = Operation::Int32Scalar;
        let baseline = profile.get_baseline_cycles(op, Precision::Int32);
        let m = RawMeasurement {
            timestamp_ns: i * 1_000_000,
            operation: op,
            cycles: (baseline * 1000.0) as u64,
            precision: Precision::Int32,
            value: 1000,
            op_count: 1000,
            temp_mc: Some(45000),
        };
        detector.feed(m);
    }
    let stats = detector.stats();
    println!(
        "After 100 normal: {} anomalies ({:.1}%)",
        stats.anomalies_detected,
        stats.anomaly_rate * 100.0
    );

    // Phase 2: Inject 10x slowdown
    println!("\nPhase 2: Injecting anomalous measurements (10x slowdown)...");
    let mut detected = false;
    for i in 0..50 {
        let op = Operation::Int32Scalar;
        let baseline = profile.get_baseline_cycles(op, Precision::Int32);
        let m = RawMeasurement {
            timestamp_ns: (100 + i) * 1_000_000,
            operation: op,
            cycles: (baseline * 10000.0) as u64,
            precision: Precision::Int32,
            value: 1000,
            op_count: 1000,
            temp_mc: Some(45000),
        };
        if let Some(anomaly) = detector.feed(m) {
            println!(
                "  ⚠ Detected at sample {}: score={:.4}",
                100 + i,
                anomaly.anomaly_score
            );
            detected = true;
            break;
        }
    }

    if !detected {
        println!("  ⚠ Failed to detect 10x slowdown anomaly!");
    }

    let stats = detector.stats();
    println!(
        "\nFinal: {} total, {} anomalies ({:.2}%)",
        stats.total_measurements,
        stats.anomalies_detected,
        stats.anomaly_rate * 100.0
    );
}

/// Run a monitoring batch and return (op_count, elapsed_cycles)
fn run_monitoring_batch(op: Operation) -> (u64, u64) {
    let batch = 5000;

    #[cfg(target_arch = "x86_64")]
    let start = unsafe { std::arch::x86_64::_rdtsc() };

    #[cfg(not(target_arch = "x86_64"))]
    let start = std::time::Instant::now();

    match op {
        Operation::Int8PackedVnni => {
            let mut acc: i8 = 0;
            for i in 0..batch {
                acc = acc.wrapping_add((i & 0xFF) as i8);
                acc = acc.wrapping_mul(3);
            }
            std::hint::black_box(acc);
        }
        Operation::Int32Scalar => {
            let mut acc: i32 = 0;
            for i in 0..batch {
                acc = acc.wrapping_add(i as i32);
                acc = acc.wrapping_mul(7);
            }
            std::hint::black_box(acc);
        }
        Operation::Fp64Norm => {
            let mut acc: f64 = 1.0;
            for i in 0..batch {
                let x = i as f64;
                acc += (x * x + 1.0).sqrt().recip();
            }
            std::hint::black_box(acc);
        }
        Operation::EisensteinMultiply => {
            let mut re: i64 = 1;
            let mut im: i64 = 1;
            for i in 0..batch {
                let a = re;
                let b = im;
                let c = (i as i64) % 97;
                let d = (i as i64) % 31;
                re = a * c - b * d;
                im = b * c + a * d - b * d;
            }
            std::hint::black_box((re, im));
        }
    }

    #[cfg(target_arch = "x86_64")]
    let end = unsafe { std::arch::x86_64::_rdtsc() };

    #[cfg(not(target_arch = "x86_64"))]
    let end_nanos = start.elapsed().as_nanos() as u64;

    #[cfg(target_arch = "x86_64")]
    return (batch, end.saturating_sub(start));

    #[cfg(not(target_arch = "x86_64"))]
    return (batch, end_nanos);
}

/// Inline CPU temperature reading for live monitoring
fn read_cpu_temp_inline() -> Option<i64> {
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
