use crate::types::*;
use std::collections::HashMap;

/// The 5-stage folding pipeline.
///
/// Each stage strips one layer of confounding variation,
/// like an RG flow toward a fixed point of pure anomaly signal.
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

/// Stage 1: Normalize cycles per operation
fn stage1_normalize(m: &RawMeasurement) -> CycleNormalized {
    let op_count = if m.op_count == 0 { 1 } else { m.op_count };
    CycleNormalized {
        cycles_per_op: m.cycles as f64 / op_count as f64,
        operation: m.operation,
        precision: m.precision,
    }
}

/// Stage 2: Compare against expected throughput, compute deviation
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

/// Stage 3: Adjust for thermal drift
fn stage3_thermal(
    s2: &ThroughputModel,
    m: &RawMeasurement,
    profile: &HardwareProfile,
) -> ThermalBaseline {
    let thermal_coeff = profile.get_thermal_coefficient(m.operation, m.precision);
    // Simulated temperature based on timestamp spread (in real impl, read from HW)
    let simulated_temp_delta = ((m.timestamp_ns % 1_000_000_000) as f64) / 1e9 * 10.0;
    let thermal_adjustment = thermal_coeff * simulated_temp_delta;
    let normalized_deviation = s2.deviation - thermal_adjustment;

    ThermalBaseline {
        normalized_deviation,
        precision: s2.precision,
        thermal_coefficient: thermal_coeff,
    }
}

/// Stage 4: Utilization fingerprint — compare against baseline distribution
fn stage4_utilization(
    s3: &ThermalBaseline,
    m: &RawMeasurement,
    profile: &HardwareProfile,
) -> UtilizationFingerprint {
    let (mean, std_dev) = profile.get_utilization_baseline(m.operation, m.precision);
    let std_dev = if std_dev == 0.0 { 0.01 } else { std_dev };

    let z_score = (s3.normalized_deviation - mean).abs() / std_dev;
    let anomaly_score = 1.0 - (-z_score * 0.5).exp();
    let confidence = 1.0 - (1.0 / (1.0 + z_score));

    UtilizationFingerprint {
        anomaly_score: anomaly_score.min(1.0).max(0.0),
        precision: s3.precision,
        confidence: confidence.min(1.0).max(0.0),
    }
}

/// Stage 5: Binary decision with 3-sigma threshold
fn stage5_decide(s4: &UtilizationFingerprint) -> Decision {
    // 3-sigma threshold ≈ anomaly_score > 0.95
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
    pub decision: Decision,
}
