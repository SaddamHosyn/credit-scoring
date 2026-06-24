import numpy as np
import pandas as pd

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between two distributions.
    
    Formula:
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        
    Interpretation:
        PSI < 0.1: No significant change / stable.
        0.1 <= PSI < 0.25: Moderate change / warrants monitoring.
        PSI >= 0.25: Significant change / requires action (retrain).
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin edges based on expected (reference) distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    
    # Adjust boundaries to handle duplicate percentiles
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # If expected is constant
        return 0.0
        
    # Ensure outer edges cover all values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Calculate counts in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    # Convert to proportions
    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)
    
    # Handle zero proportions by adding a small epsilon to prevent log(0) or div by 0
    eps = 1e-4
    expected_props = np.where(expected_props == 0, eps, expected_props)
    actual_props = np.where(actual_props == 0, eps, actual_props)
    
    # Calculate PSI
    psi_value = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    return float(psi_value)

def analyze_drift(reference_pd: np.ndarray, target_pd: np.ndarray) -> dict:
    """
    Analyzes drift in the Probability of Default (PD) distribution.
    """
    ref_mean = np.mean(reference_pd)
    tgt_mean = np.mean(target_pd)
    
    pd_diff = tgt_mean - ref_mean
    pd_pct_change = (pd_diff / ref_mean) * 100 if ref_mean != 0 else 0
    
    psi = calculate_psi(reference_pd, target_pd, num_bins=10)
    
    # Determine Status
    if psi < 0.1:
        status = "STABLE"
    elif psi < 0.25:
        status = "WARNING"
    else:
        status = "DRIFT_DETECTED"
        
    return {
        "psi": psi,
        "reference_mean_pd": ref_mean,
        "target_mean_pd": tgt_mean,
        "pd_absolute_shift": pd_diff,
        "pd_percentage_shift": pd_pct_change,
        "status": status,
    }

def main():
    print("=" * 60)
    print("RUNNING MODEL DRIFT AND PERFORMANCE MONITORING")
    print("=" * 60)
    
    # Mock data demonstration (Reference vs Target)
    # Simulate a baseline distribution of credit probabilities (e.g. mean 0.08 default rate)
    np.random.seed(42)
    reference = np.random.beta(a=2, b=20, size=10000) # Baseline default probs
    
    # Scenario A: Stable production predictions (no drift)
    stable_target = np.random.beta(a=2, b=20, size=5000)
    
    # Scenario B: Shifted production predictions (significant drift, e.g. economic downturn)
    drifted_target = np.random.beta(a=3, b=15, size=5000) # Higher default rate
    
    print("\nScenario A: Stable Production Period")
    metrics_a = analyze_drift(reference, stable_target)
    print(f"  PSI: {metrics_a['psi']:.4f}")
    print(f"  Ref Mean PD: {metrics_a['reference_mean_pd']:.4f} | Target Mean PD: {metrics_a['target_mean_pd']:.4f}")
    print(f"  PD Relative Shift: {metrics_a['pd_percentage_shift']:.2f}%")
    print(f"  Status: {metrics_a['status']}")
    
    print("\nScenario B: Shifted Production Period (Economic Downturn)")
    metrics_b = analyze_drift(reference, drifted_target)
    print(f"  PSI: {metrics_b['psi']:.4f}")
    print(f"  Ref Mean PD: {metrics_b['reference_mean_pd']:.4f} | Target Mean PD: {metrics_b['target_mean_pd']:.4f}")
    print(f"  PD Relative Shift: {metrics_b['pd_percentage_shift']:.2f}%")
    print(f"  Status: {metrics_b['status']}")
    
    # Return 1 if drift detected, so it can fail a CI build or trigger notifications
    if metrics_b['status'] == "DRIFT_DETECTED":
        print("\nDrift monitoring run completed successfully. (Detected drift correctly in simulated downturn).")

if __name__ == "__main__":
    main()
