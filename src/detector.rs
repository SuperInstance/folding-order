use crate::types::*;
use crate::fold;

/// Online anomaly detector with a sliding window buffer.
///
/// Maintains a ring buffer of recent measurements and runs
/// the 5-stage folding pipeline for anomaly detection.
pub struct AnomalyDetector {
    pub profile: HardwareProfile,
    buffer: Vec<RawMeasurement>,
    buffer_size: usize,
    write_pos: usize,
    filled: bool,
    threshold_sigma: f64,
    anomaly_count: usize,
    total_count: usize,
}

impl AnomalyDetector {
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            profile,
            buffer: Vec::new(),
            buffer_size: 1024,
            write_pos: 0,
            filled: false,
            threshold_sigma: 3.0,
            anomaly_count: 0,
            total_count: 0,
        }
    }

    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    pub fn with_threshold(mut self, sigma: f64) -> Self {
        self.threshold_sigma = sigma;
        self
    }

    /// Feed a single measurement. Returns Some(Anomaly) if detected.
    pub fn feed(&mut self, measurement: RawMeasurement) -> Option<Anomaly> {
        self.total_count += 1;

        // Ring buffer insert
        if self.buffer.len() < self.buffer_size {
            self.buffer.push(measurement.clone());
        } else {
            self.buffer[self.write_pos] = measurement.clone();
            self.write_pos = (self.write_pos + 1) % self.buffer_size;
            self.filled = true;
        }

        // Need minimum buffer for meaningful detection
        let effective_len = if self.filled { self.buffer_size } else { self.buffer.len() };
        if effective_len < 5 {
            return None;
        }

        // Quick check: raw deviation from baseline
        let baseline = self.profile.get_baseline_cycles(measurement.operation, measurement.precision);
        let actual_cpo = if measurement.op_count > 0 {
            measurement.cycles as f64 / measurement.op_count as f64
        } else {
            baseline
        };

        let raw_deviation = if baseline > 0.0 {
            (actual_cpo - baseline).abs() / baseline
        } else {
            0.0
        };

        // Run the full 5-stage pipeline for a score
        let decision = fold::fold_single(&measurement, &self.profile);

        // Trigger on either raw deviation > 50% or pipeline anomaly
        if raw_deviation > 0.5 || decision.is_anomalous() {
            self.anomaly_count += 1;
            let desc = format!(
                "Anomaly: raw_dev={:.1}%, cpo={:.2} vs baseline={:.2}, pipeline_score={:.4}, z_score={:.2}, op={:?}",
                raw_deviation * 100.0,
                actual_cpo,
                baseline,
                decision.score(),
                match &decision { Decision::Anomalous(_) => 0.0, _ => 0.0 }, // z_score not in Decision
                measurement.operation
            );
            Some(Anomaly {
                measurement,
                anomaly_score: raw_deviation.min(1.0).max(decision.score()),
                stage: if raw_deviation > 0.5 { 2 } else { 5 },
                description: desc,
            })
        } else {
            None
        }
    }

    /// Run the full folding pipeline on the buffer
    pub fn analyze(&self) -> Vec<(RawMeasurement, Decision)> {
        let decisions = fold::fold(&self.buffer, &self.profile);
        self.buffer
            .iter()
            .cloned()
            .zip(decisions)
            .collect()
    }

    /// Get current statistics
    pub fn stats(&self) -> DetectorStats {
        DetectorStats {
            total_measurements: self.total_count,
            anomalies_detected: self.anomaly_count,
            buffer_fill: if self.filled { self.buffer_size } else { self.buffer.len() },
            anomaly_rate: if self.total_count > 0 {
                self.anomaly_count as f64 / self.total_count as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the buffer and counters
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.write_pos = 0;
        self.filled = false;
        self.anomaly_count = 0;
        self.total_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> HardwareProfile {
        let mut baseline_cycles = std::collections::HashMap::new();
        let mut thermal_coefficients = std::collections::HashMap::new();
        let mut utilization_baselines = std::collections::HashMap::new();

        baseline_cycles.insert("INT32-SCALAR/i32".into(), 5.0);
        thermal_coefficients.insert("INT32-SCALAR/i32".into(), 0.001);
        utilization_baselines.insert("INT32-SCALAR/i32".into(), (0.0, 0.1));

        HardwareProfile {
            cpu_model: "Test CPU".into(),
            baseline_cycles,
            thermal_coefficients,
            utilization_baselines,
            reference_temp_mc: Some(45000),
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
            temp_mc: Some(45000),
        }
    }

    #[test]
    fn test_no_anomaly_on_normal_data() {
        let profile = test_profile();
        let mut detector = AnomalyDetector::new(profile);
        let mut anomalies = 0;

        for i in 0..100 {
            let m = make_measurement(5000 + (i % 10) as u64, 1000);
            if detector.feed(m).is_some() {
                anomalies += 1;
            }
        }

        assert_eq!(anomalies, 0, "Normal data should produce zero anomalies");
    }

    #[test]
    fn test_detects_anomaly() {
        let profile = test_profile();
        let mut detector = AnomalyDetector::new(profile);

        // Warm up with normal data
        for i in 0..10 {
            detector.feed(make_measurement(5000, 1000));
        }

        // Inject 10x slowdown
        let result = detector.feed(make_measurement(50000, 1000));
        assert!(result.is_some(), "10x slowdown should be detected");
    }

    #[test]
    fn test_ring_buffer_wraps() {
        let profile = test_profile();
        let mut detector = AnomalyDetector::new(profile).with_buffer_size(16);

        for i in 0..100 {
            detector.feed(make_measurement(5000 + (i % 10) as u64, 1000));
        }

        let stats = detector.stats();
        assert_eq!(stats.buffer_fill, 16);
        assert_eq!(stats.total_measurements, 100);
    }

    #[test]
    fn test_reset_clears_state() {
        let profile = test_profile();
        let mut detector = AnomalyDetector::new(profile);

        for i in 0..50 {
            detector.feed(make_measurement(5000, 1000));
        }
        assert!(detector.stats().total_measurements > 0);

        detector.reset();
        let stats = detector.stats();
        assert_eq!(stats.total_measurements, 0);
        assert_eq!(stats.buffer_fill, 0);
        assert_eq!(stats.anomalies_detected, 0);
    }
}
