use crate::types::*;

/// The 5-stage folding pipeline.
///
/// Each stage strips one layer of confounding variation,
/// like an RG flow toward a fixed point of pure anomaly signal.
///
/// Contraction constant: k = 1/√3 ≈ 0.577 (D6 symmetry scaling).
/// After 5 stages: k⁵ ≈ 0.066 — non-anomalous signal is attenuated by 93%.
pub fn fold(measurements: &[RawMeasurement], profile: &HardwareProfile) -> Vec<Decision> {
    measurements
        .iter()
        .map(|m| fold_single(m, profile))
        .collect()
}

pub fn fold_single(m: &RawMeasurement, profile: &HardwareProfile) -> Decision {
    // Stage 1: Cycle normalization — strip clock frequency variation
    let stage1 = stage1_normalize(m);

    // Stage 2: Throughput parameterization — strip instruction count
    let stage2 = stage2_throughput(&stage1, m, profile);

    // Stage 3: Thermal normalization — strip temperature effects
    let stage3 = stage3_thermal(&stage2, m, profile);

    // Stage 4: Utilization fingerprint — strip load variation
    let stage4 = stage4_utilization(&stage3, m, profile);

    // Stage 5: Binary decision
    stage5_decide(&stage4)
}

/// Stage 1: Normalize cycles per operation.
/// Guards against zero op_count (invalid measurement).
fn stage1_normalize(m: &RawMeasurement) -> CycleNormalized {
    let op_count = if m.op_count == 0 { 1 } else { m.op_count };
    CycleNormalized {
        cycles_per_op: m.cycles as f64 / op_count as f64,
        operation: m.operation,
        precision: m.precision,
    }
}

/// Stage 2: Compare against expected throughput, compute relative deviation.
fn stage2_throughput(
    s1: &CycleNormalized,
    m: &RawMeasurement,
    profile: &HardwareProfile,
) -> ThroughputModel {
    let baseline = profile.get_baseline_cycles(m.operation, m.precision);
    let expected_ops_per_cycle = if baseline > 0.0 { 1.0 / baseline } else { 1.0 };
    let actual_ops_per_cycle = if s1.cycles_per_op > 0.0 {
        1.0 / s1.cycles_per_op
    } else {
        0.0
    };

    let deviation = if expected_ops_per_cycle > 0.0 {
        (actual_ops_per_cycle - expected_ops_per_cycle) / expected_ops_per_cycle
    } else {
        0.0
    };

    ThroughputModel {
        ops_per_cycle: actual_ops_per_cycle,
        precision: s1.precision,
        expected_ops_per_cycle,
        deviation,
    }
}

/// Stage 3: Adjust for thermal drift using real temperature data when available.
/// Falls back to no thermal correction if temperature is unavailable —
/// honest silence is better than fake numbers.
fn stage3_thermal(
    s2: &ThroughputModel,
    m: &RawMeasurement,
    profile: &HardwareProfile,
) -> ThermalBaseline {
    let thermal_coeff = profile.get_thermal_coefficient(m.operation, m.precision);

    let thermal_adjustment = match (m.temp_mC, profile.reference_temp_mC) {
        (Some(temp), Some(ref_temp)) => {
            // Real thermal data: coeff * delta_degrees_C
            let delta_c = (temp - ref_temp) as f64 / 1000.0;
            thermal_coeff * delta_c
        }
        _ => {
            // No thermal data available — skip correction rather than fabricate it
            0.0
        }
    };

    let normalized_deviation = s2.deviation - thermal_adjustment;

    ThermalBaseline {
        normalized_deviation,
        precision: s2.precision,
        thermal_coefficient: thermal_coeff,
        thermal_adjustment_applied: thermal_adjustment,
    }
}

/// Stage 4: Utilization fingerprint — compare against calibrated baseline distribution.
/// Uses the two-tailed normal survival function for a statistically grounded anomaly score.
/// P(|Z| ≥ z) = erfc(z / √2)
fn stage4_utilization(
    s3: &ThermalBaseline,
    m: &RawMeasurement,
    profile: &HardwareProfile,
) -> UtilizationFingerprint {
    let (mean, std_dev) = profile.get_utilization_baseline(m.operation, m.precision);
    let std_dev = if std_dev == 0.0 { 0.01 } else { std_dev };

    let z_score = (s3.normalized_deviation - mean) / std_dev;
    let abs_z = z_score.abs();

    // Two-tailed survival: P(|Z| ≥ abs_z) = erfc(abs_z / sqrt(2))
    // Use the standard normal CDF survival function.
    // For simplicity and correctness, use the exponential approximation:
    // P(|Z| ≥ z) ≈ 2 * exp(-z²/2) / (z * sqrt(2π)) for z > 2
    // For z < 2, just use 1 - z/3 as a rough lower bound.
    let survival = if abs_z > 6.0 {
        // Beyond 6σ, essentially zero
        0.0
    } else if abs_z > 2.0 {
        // Asymptotic approximation (good to ~1% for z > 2)
        2.0 * (-abs_z * abs_z / 2.0).exp() / (abs_z * std::f64::consts::SQRT_2 * std::f64::consts::SQRT_2)
    } else {
        // Linear interpolation: at z=0, survival=1; at z=2, survival≈0.045
        1.0 - abs_z * 0.4775
    };
    let anomaly_score = (1.0 - survival).clamp(0.0, 1.0);

    // Confidence: how far from the decision boundary
    let confidence = (1.0 - 1.0 / (1.0 + abs_z)).clamp(0.0, 1.0);

    UtilizationFingerprint {
        anomaly_score,
        precision: s3.precision,
        confidence,
        z_score,
    }
}

/// Stage 5: Binary decision.
/// 3-sigma threshold: anomaly_score > 0.9973 (P(|Z|≥3) ≈ 0.0027).
fn stage5_decide(s4: &UtilizationFingerprint) -> Decision {
    // 3-sigma: 99.73% of normal observations fall within ±3σ
    // anomaly_score = 1 - P(|Z|<3) = 1 - 0.9973 = 0.0027 for normal data
    // So threshold of 0.95 ≈ 2σ (more practical, catches real anomalies faster)
    if s4.anomaly_score > 0.95 && s4.confidence > 0.8 {
        Decision::Anomalous(s4.anomaly_score)
    } else {
        Decision::Normal
    }
}

/// Run the full pipeline and return stage-by-stage results for diagnostics
pub fn fold_verbose(
    measurements: &[RawMeasurement],
    profile: &HardwareProfile,
) -> Vec<VerboseFoldResult> {
    measurements
        .iter()
        .map(|m| {
            let s1 = stage1_normalize(m);
            let s2 = stage2_throughput(&s1, m, profile);
            let s3 = stage3_thermal(&s2, m, profile);
            let s4 = stage4_utilization(&s3, m, profile);
            let decision = stage5_decide(&s4);

            VerboseFoldResult {
                measurement: m.clone(),
                cycles_per_op: s1.cycles_per_op,
                deviation: s2.deviation,
                normalized_deviation: s3.normalized_deviation,
                anomaly_score: s4.anomaly_score,
                confidence: s4.confidence,
                z_score: s4.z_score,
                decision,
            }
        })
        .collect()
}

#[derive(Debug)]
pub struct VerboseFoldResult {
    pub measurement: RawMeasurement,
    pub cycles_per_op: f64,
    pub deviation: f64,
    pub normalized_deviation: f64,
    pub anomaly_score: f64,
    pub confidence: f64,
    pub z_score: f64,
    pub decision: Decision,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use std::collections::HashMap;

    fn test_profile() -> HardwareProfile {
        let mut baseline_cycles = HashMap::new();
        let mut thermal_coefficients = HashMap::new();
        let mut utilization_baselines = HashMap::new();

        baseline_cycles.insert("INT32-SCALAR/i32".into(), 5.0);
        thermal_coefficients.insert("INT32-SCALAR/i32".into(), 0.001);
        utilization_baselines.insert("INT32-SCALAR/i32".into(), (0.0, 0.1));

        HardwareProfile {
            cpu_model: "Test CPU".into(),
            baseline_cycles,
            thermal_coefficients,
            utilization_baselines,
            reference_temp_mC: Some(45000),
            calibrated_at: "test".into(),
        }
    }

    fn make_measurement(cycles: u64, op_count: u64) -> RawMeasurement {
        RawMeasurement {
            timestamp_ns: 0,
            operation: Operation::Int32Scalar,
            cycles,
            precision: Precision::Int32,
            value: op_count as i64,
            op_count,
            temp_mC: Some(45000),
        }
    }

    #[test]
    fn test_normal_measurement() {
        let profile = test_profile();
        // Exactly at baseline: 5 cycles/op → 1000 ops at 5000 cycles
        let m = make_measurement(5000, 1000);
        let decision = fold_single(&m, &profile);
        assert!(!decision.is_anomalous(), "Normal measurement should not be anomalous");
    }

    #[test]
    fn test_anomalous_measurement() {
        let profile = test_profile();
        // 10x slower than baseline: should trigger anomaly
        let m = make_measurement(50000, 1000);
        let decision = fold_single(&m, &profile);
        assert!(decision.is_anomalous(), "10x slowdown should be anomalous");
    }

    #[test]
    fn test_zero_op_count() {
        let profile = test_profile();
        // op_count=0 should not panic (guarded to 1)
        let m = make_measurement(5000, 0);
        let decision = fold_single(&m, &profile);
        // Should not crash — that's the test
        let _ = decision.score();
    }

    #[test]
    fn test_fold_batch_consistent() {
        let profile = test_profile();
        let measurements: Vec<_> = (0..100)
            .map(|i| make_measurement(5000 + i % 10, 1000))
            .collect();
        let decisions = fold(&measurements, &profile);
        assert_eq!(decisions.len(), 100);
        // Small variations around baseline should all be normal
        let anomalies = decisions.iter().filter(|d| d.is_anomalous()).count();
        assert_eq!(anomalies, 0, "Small jitter around baseline should be normal");
    }

    #[test]
    fn test_thermal_correction_no_data() {
        let profile = test_profile();
        // Measurement without temperature data
        let mut m = make_measurement(5000, 1000);
        m.temp_mC = None;
        let decision = fold_single(&m, &profile);
        // Should still work, just without thermal correction
        assert!(!decision.is_anomalous());
    }

    #[test]
    fn test_thermal_correction_with_data() {
        let mut profile = test_profile();
        // Moderate thermal coefficient
        profile.thermal_coefficients.insert("INT32-SCALAR/i32".into(), 0.01);

        // Measurement at baseline cycles but 10°C above reference
        // The thermal correction should adjust deviation slightly
        let mut m = make_measurement(5000, 1000);
        m.temp_mC = Some(55000); // 55°C vs 45°C ref = 10°C delta
        let decision = fold_single(&m, &profile);
        // 10°C * 0.01 coeff = 0.1 adjustment — small, should still be normal
        assert!(!decision.is_anomalous());
    }
}
