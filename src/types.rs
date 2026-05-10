use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Operation {
    Int8PackedVnni,
    Int32Scalar,
    Fp64Norm,
    EisensteinMultiply,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Int8PackedVnni => write!(f, "INT8-PACKED-VNNI"),
            Operation::Int32Scalar => write!(f, "INT32-SCALAR"),
            Operation::Fp64Norm => write!(f, "FP64-NORM"),
            Operation::EisensteinMultiply => write!(f, "EISENSTEIN-MUL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    Int8,
    Int32,
    Fp64,
    Eisenstein,
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Precision::Int8 => write!(f, "i8"),
            Precision::Int32 => write!(f, "i32"),
            Precision::Fp64 => write!(f, "f64"),
            Precision::Eisenstein => write!(f, "eis"),
        }
    }
}

impl Operation {
    pub fn default_precision(&self) -> Precision {
        match self {
            Operation::Int8PackedVnni => Precision::Int8,
            Operation::Int32Scalar => Precision::Int32,
            Operation::Fp64Norm => Precision::Fp64,
            Operation::EisensteinMultiply => Precision::Eisenstein,
        }
    }
}

pub const ALL_OPERATIONS: [Operation; 4] = [
    Operation::Int8PackedVnni,
    Operation::Int32Scalar,
    Operation::Fp64Norm,
    Operation::EisensteinMultiply,
];

/// Stage 0: Raw measurement from hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawMeasurement {
    pub timestamp_ns: u64,
    pub operation: Operation,
    pub cycles: u64,
    pub precision: Precision,
    pub value: i64,
    pub op_count: u64,
    /// Optional CPU temperature in millidegrees C (from /sys/class/thermal/)
    #[serde(default)]
    pub temp_mC: Option<i64>,
}

impl RawMeasurement {
    /// Returns temperature delta in degrees C relative to a reference,
    /// or None if temperature data is unavailable.
    pub fn temp_delta_c(&self, reference_mC: i64) -> Option<f64> {
        self.temp_mC.map(|t| (t - reference_mC) as f64 / 1000.0)
    }
}

/// Stage 1: Cycle-normalized (strip clock frequency variation)
#[derive(Debug, Clone)]
pub struct CycleNormalized {
    pub cycles_per_op: f64,
    pub operation: Operation,
    pub precision: Precision,
}

/// Stage 2: Throughput-parameterized (strip instruction count)
#[derive(Debug, Clone)]
pub struct ThroughputModel {
    pub ops_per_cycle: f64,
    pub precision: Precision,
    pub expected_ops_per_cycle: f64,
    pub deviation: f64,
}

/// Stage 3: Thermal-normalized (strip temperature effects)
#[derive(Debug, Clone)]
pub struct ThermalBaseline {
    pub normalized_deviation: f64,
    pub precision: Precision,
    pub thermal_coefficient: f64,
    pub thermal_adjustment_applied: f64,
}

/// Stage 4: Utilization fingerprint (strip load variation)
#[derive(Debug, Clone)]
pub struct UtilizationFingerprint {
    pub anomaly_score: f64,
    pub precision: Precision,
    pub confidence: f64,
    pub z_score: f64,
}

/// Stage 5: Binary decision
#[derive(Debug, Clone)]
pub enum Decision {
    Normal,
    Anomalous(f64),
}

impl Decision {
    pub fn is_anomalous(&self) -> bool {
        matches!(self, Decision::Anomalous(_))
    }

    pub fn score(&self) -> f64 {
        match self {
            Decision::Normal => 0.0,
            Decision::Anomalous(s) => *s,
        }
    }
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Decision::Normal => write!(f, "NORMAL"),
            Decision::Anomalous(score) => write!(f, "ANOMALOUS({:.4})", score),
        }
    }
}

/// Hardware profile from calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu_model: String,
    pub baseline_cycles: HashMap<String, f64>,
    pub thermal_coefficients: HashMap<String, f64>,
    /// (mean, std_dev) of normalized deviations from calibration
    pub utilization_baselines: HashMap<String, (f64, f64)>,
    /// Reference temperature in millidegrees C from calibration
    #[serde(default)]
    pub reference_temp_mC: Option<i64>,
    pub calibrated_at: String,
}

impl HardwareProfile {
    pub fn profile_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        std::path::PathBuf::from(home).join(".folding-order")
    }

    pub fn profile_path() -> std::path::PathBuf {
        Self::profile_dir().join("profile.json")
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&data).map_err(|e| format!("Parse error: {}", e))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("Mkdir error: {}", e))?;
        }
        let data = serde_json::to_string_pretty(self).map_err(|e| format!("Serialize error: {}", e))?;
        std::fs::write(path, data).map_err(|e| format!("Write error: {}", e))
    }

    fn op_key(op: Operation, prec: Precision) -> String {
        format!("{}/{}", op, prec)
    }

    pub fn get_baseline_cycles(&self, op: Operation, prec: Precision) -> f64 {
        let key = Self::op_key(op, prec);
        self.baseline_cycles.get(&key).copied().unwrap_or(1.0)
    }

    pub fn get_thermal_coefficient(&self, op: Operation, prec: Precision) -> f64 {
        let key = Self::op_key(op, prec);
        self.thermal_coefficients.get(&key).copied().unwrap_or(0.001)
    }

    /// Returns (mean, std_dev) of normalized deviations for this operation.
    /// Default: mean=0.0 (no systematic bias), std_dev=0.1 (10% typical variation).
    pub fn get_utilization_baseline(&self, op: Operation, prec: Precision) -> (f64, f64) {
        let key = Self::op_key(op, prec);
        self.utilization_baselines.get(&key).copied().unwrap_or((0.0, 0.1))
    }
}

/// Anomaly report
#[derive(Debug, Clone)]
pub struct Anomaly {
    pub measurement: RawMeasurement,
    pub anomaly_score: f64,
    pub stage: usize,
    pub description: String,
}

/// Detector statistics
#[derive(Debug, Clone)]
pub struct DetectorStats {
    pub total_measurements: usize,
    pub anomalies_detected: usize,
    pub buffer_fill: usize,
    pub anomaly_rate: f64,
}
