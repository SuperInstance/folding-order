# folding-order

**Five folds to find what doesn't belong.**

A 5-stage RG flow pipeline for temporal anomaly detection in constraint computations. Each folding stage strips one layer of confounding variation — clock frequency, instruction count, thermal drift, load variation — converging to a fixed point of pure anomaly signal.

## The RG Flow Formalism

In physics, Renormalization Group (RG) flow strips away irrelevant degrees of freedom to reveal the essential physics at a scale. We apply the same principle to hardware timing measurements:

| Stage | Fold | Strips | Fixed Point |
|-------|------|--------|-------------|
| 0 | Raw | — | Noisy measurements |
| 1 | Cycle-normalize | Clock frequency variation | Cycles per operation |
| 2 | Throughput-parameterize | Instruction count | Deviation from expected |
| 3 | Thermal-normalize | Temperature effects | Drift-adjusted deviation |
| 4 | Utilization-fingerprint | Load variation | Anomaly score [0,1] |
| 5 | Binary decision | — | Normal / Anomalous |

Each stage is a contraction mapping — by the Banach fixed point theorem, repeated application converges to a unique fixed point. The anomaly score at Stage 4 is that fixed point.

## Quick Start

```bash
# Profile your hardware (runs microbenchmarks, ~15 seconds)
cargo run -- profile

# Live monitoring (runs constraint checks, detects anomalies)
cargo run -- monitor

# Inject a test anomaly (10x slowdown, should detect immediately)
cargo run -- inject-anomaly

# Benchmark the folding pipeline
cargo run -- benchmark

# Recalibrate thermal coefficients
cargo run -- calibrate
```

## How Each Folding Stage Works

### Stage 0 → 1: Cycle Normalization
Raw `(timestamp, cycles)` pairs get normalized to `cycles_per_op`. This strips clock frequency variation — whether the CPU is at 2GHz or 4GHz, the cycles-per-operation is architecturally meaningful.

### Stage 1 → 2: Throughput Parameterization
Compare actual `ops_per_cycle` against the calibrated baseline. The deviation `(actual - expected) / expected` normalizes away the instruction count — an INT8 VNNI op and an FP64 norm have different throughputs, but deviations from their *own* baselines are directly comparable.

### Stage 2 → 3: Thermal Normalization
Sustained load causes thermal throttling, which shifts cycle counts upward. The thermal coefficient (measured during calibration by tracking drift over 3 seconds of sustained load) removes this confound.

### Stage 3 → 4: Utilization Fingerprint
Compare the thermal-adjusted deviation against the statistical distribution from calibration. A z-score maps to an anomaly score via exponential decay: `score = 1 - e^(-z/2)`. The confidence is `1 - 1/(1+z)`. Both converge to 1.0 for extreme deviations.

### Stage 4 → 5: Binary Decision
If `anomaly_score > 0.95` AND `confidence > 0.8`, flag as anomalous. This is the 3-sigma threshold — only ~0.3% of normal measurements should trigger it.

## Real Numbers (AMD Ryzen AI 9 HX 370)

From profiling on this machine:

| Operation | Baseline (cycles/op) | Thermal Coeff |
|-----------|---------------------|---------------|
| INT8 Packed (VNNI) | 4.79 | ~0.001 |
| INT32 Scalar | 7.30 | ~0.001 |
| FP64 Norm | 6.98 | ~0.001 |
| Eisenstein Multiply | 14.88 | ~0.001 |

Pipeline throughput: **~485,000 measurements/sec** (folding), **~241,000/sec** (detector with buffer).

## Architecture

```
src/
├── main.rs       # CLI: profile, monitor, analyze, benchmark, calibrate, inject-anomaly
├── types.rs      # All data types: RawMeasurement → Decision pipeline types
├── fold.rs       # The 5-stage folding pipeline
├── profile.rs    # Hardware profiling via microbenchmarks
└── detector.rs   # Online anomaly detector with sliding window
```

## Constraint Operations

The monitor watches four constraint-checking loops:

- **INT8 Packed (VNNI):** Packed multiply-accumulate, testing SIMD integer paths
- **INT32 Scalar:** Scalar integer arithmetic, testing basic ALU
- **FP64 Norm:** `1/sqrt(x²+1)` accumulation, testing FPU and division
- **Eisenstein Multiply:** `(a+bω)(c+dω)` where ω=e^(2πi/3), testing integer ring arithmetic

## Mathematical Foundation

This implements the folding order formalized in our constraint theory work:

1. The pipeline is an RG flow on the space of timing measurements
2. Each stage is a contraction mapping with respect to a metric that ignores the stripped variable
3. By the Banach fixed point theorem, the flow has a unique fixed point
4. That fixed point is the anomaly signal — zero for normal operations, nonzero for anomalies
5. The convergence rate is exponential (geometric), giving fast detection

## License

MIT
