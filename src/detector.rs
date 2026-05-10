use crate::types::*;
use crate::fold;

/// Online anomaly detector with a sliding window buffer
pub struct AnomalyDetector {
    pub profile: HardwareProfile,
    buffer: Vec<RawMeasurement>,
    buffer_size: usize,
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

        self.buffer.push(measurement.clone());
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }

        // Need minimum buffer size for meaningful detection
        if self.buffer.len() < 5 {
            return None;
        }

        // Online detection: compare cycles/op against calibrated baseline
        let baseline = self.profile.get_baseline_cycles(measurement.operation, measurement.precision);
        let actual_cpo = if measurement.op_count > 0 {
            measurement.cycles as f64 / measurement.op_count as f64
        } else {
            baseline
        };

        let deviation = if baseline > 0.0 {
            (actual_cpo - baseline).abs() / baseline
        } else {
            0.0
        };

        // Also run the full 5-stage pipeline for a score
        let decision = fold::fold_single(&measurement, &self.profile);

        // Trigger on either raw deviation > 50% or pipeline anomaly
        if deviation > 0.5 || decision.is_anomalous() {
            self.anomaly_count += 1;
            let desc = format!(
                "Anomaly: dev={:.1}%, cpo={:.2} vs baseline={:.2}, pipeline_score={:.4}, op={:?}",
                deviation * 100.0,
                actual_cpo,
                baseline,
                decision.score(),
                measurement.operation
            );
            Some(Anomaly {
                measurement,
                anomaly_score: deviation.min(1.0).max(decision.score()),
                stage: if deviation > 0.5 { 2 } else { 5 },
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
            buffer_fill: self.buffer.len(),
            anomaly_rate: if self.total_count > 0 {
                self.anomaly_count as f64 / self.total_count as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the buffer
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.anomaly_count = 0;
        self.total_count = 0;
    }
}
