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
                match &decision { Decision::Anomalous(_) => 0.0, _ => 0.0 },
                measurement.operation
            );
            Some(Anomaly::new(
                measurement,
                raw_deviation.min(1.0).max(decision.score()),
                if raw_deviation > 0.5 { 2 } else { 5 },
                desc,
            ))
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

    // ── Simulation-First Predictions ──────────────────────────

    /// Predict the expected behavior of an operation before it runs.
    /// This is the core of simulation-first: plan the check before the work.
    pub fn predict(&self, op: Operation, prec: Precision, t_minus_ns: Option<u64>) -> PredictedMeasurement {
        let baseline = self.profile.get_baseline_cycles(op, prec);
        let (_, std_dev) = self.profile.get_utilization_baseline(op, prec);
        PredictedMeasurement {
            operation: op,
            precision: prec,
            expected_cycles_per_op: baseline,
            tolerance_sigma: self.threshold_sigma,
            t_minus_event_ns: t_minus_ns,
            lamport: 0, // caller should set via LamportClock
            confirmed: false,
            actual_deviation: None,
        }
    }

    /// Feed a measurement and check it against a prediction.
    /// Returns (decision, was_confirmed).
    /// Simulation-first: the live check is confirmation, not discovery.
    pub fn feed_and_confirm(&mut self, m: RawMeasurement, prediction: &mut PredictedMeasurement) -> (Option<Anomaly>, bool) {
        let confirmed = prediction.confirm(&m, &self.profile);
        let mut anomaly = self.feed(m);

        // If anomaly detected but prediction was confirmed, the prediction
        // was based on stale profile — flag for re-calibration
        if anomaly.is_some() && confirmed {
            if let Some(ref mut a) = anomaly {
                a.description.push_str(" [STALE-PROFILE: prediction confirmed but pipeline flagged anomaly]");
            }
        }

        (anomaly, confirmed)
    }

    /// Get active (non-retracted) anomalies from recent history
    pub fn active_anomalies(&self) -> Vec<Anomaly> {
        // In a full implementation this would query a persistent store
        // For now, the ring buffer + last anomaly is available
        Vec::new()
    }
}

/// Lamport-aware detector that tracks causal ordering across agents
pub struct LamportDetector {
    detector: AnomalyDetector,
    clock: LamportClock,
    predictions: Vec<PredictedMeasurement>,
}

impl LamportDetector {
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            detector: AnomalyDetector::new(profile),
            clock: LamportClock::new(),
            predictions: Vec::new(),
        }
    }

    /// Issue a prediction with Lamport timestamp
    pub fn predict(&mut self, op: Operation, prec: Precision, t_minus_ns: Option<u64>) -> PredictedMeasurement {
        let mut pred = self.detector.predict(op, prec, t_minus_ns);
        pred.lamport = self.clock.tick();
        self.predictions.push(pred.clone());
        pred
    }

    /// Feed measurement, confirm prediction, return causally-ordered result
    pub fn feed_and_confirm(&mut self, m: RawMeasurement) -> Option<Anomaly> {
        let lamport = self.clock.tick();

        // Find matching prediction
        let pred_idx = self.predictions.iter().position(|p| {
            p.operation == m.operation && p.precision == m.precision && !p.confirmed
        });

        if let Some(idx) = pred_idx {
            let (anomaly, _confirmed) = self.detector.feed_and_confirm(m, &mut self.predictions[idx]);
            if let Some(mut a) = anomaly {
                a.lamport = lamport;
                Some(a)
            } else {
                None
            }
        } else {
            // No prediction — just detect
            let mut anomaly = self.detector.feed(m);
            if let Some(ref mut a) = anomaly {
                a.lamport = lamport;
            }
            anomaly
        }
    }

    /// Merge with remote Lamport clock (e.g., from fleet coordination)
    pub fn merge_clock(&mut self, remote: u64) -> u64 {
        self.clock.merge(remote)
    }

    pub fn stats(&self) -> DetectorStats { self.detector.stats() }
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

    #[test]
    fn test_prediction_confirm_matches() {
        let profile = test_profile();
        let detector = AnomalyDetector::new(profile);
        let mut pred = detector.predict(Operation::Int32Scalar, Precision::Int32, None);

        // Feed a normal measurement
        let m = make_measurement(5000, 1000);
        let confirmed = pred.confirm(&m, &detector.profile);
        assert!(confirmed, "Normal measurement should confirm prediction");
        assert!(pred.confirmed);
    }

    #[test]
    fn test_prediction_rejects_outlier() {
        let profile = test_profile();
        let detector = AnomalyDetector::new(profile);
        let mut pred = detector.predict(Operation::Int32Scalar, Precision::Int32, None);

        // Feed a 10x outlier
        let m = make_measurement(50000, 1000);
        let confirmed = pred.confirm(&m, &detector.profile);
        assert!(!confirmed, "10x outlier should fail confirmation");
    }

    #[test]
    fn test_lamport_clock_monotonic() {
        let mut clock = LamportClock::new();
        let t1 = clock.tick();
        let t2 = clock.tick();
        let t3 = clock.tick();
        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn test_lamport_clock_merge() {
        let mut clock = LamportClock::new();
        clock.tick();
        clock.tick(); // time = 2
        let merged = clock.merge(5); // max(2,5)+1 = 6
        assert_eq!(merged, 6);
    }

    #[test]
    fn test_tile_lifecycle() {
        let profile = test_profile();
        let mut detector = AnomalyDetector::new(profile);

        // Feed normal data
        for _ in 0..10 { detector.feed(make_measurement(5000, 1000)); }

        // Feed anomaly
        let anomaly = detector.feed(make_measurement(50000, 1000));
        assert!(anomaly.is_some());
        let mut a = anomaly.unwrap();
        assert_eq!(a.tile_state, TileState::Active);

        // Retract it (false positive)
        a.retract("false positive");
        assert_eq!(a.tile_state, TileState::Retracted);
        assert!(a.description.contains("RETRACTED"));
    }

    #[test]
    fn test_lamport_detector_prediction() {
        let profile = test_profile();
        let mut ld = LamportDetector::new(profile);

        // Issue prediction
        let pred = ld.predict(Operation::Int32Scalar, Precision::Int32, Some(1000));
        assert_eq!(pred.lamport, 1); // first tick
        assert_eq!(pred.t_minus_event_ns, Some(1000));
        assert!(!pred.confirmed);

        // Feed confirming measurement
        let m = make_measurement(5000, 1000);
        let result = ld.feed_and_confirm(m);
        assert!(result.is_none()); // no anomaly = good
    }
}
