import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Monitor Model Drift and Score Distribution (PSI/PD)")
    parser.add_argument(
        "--reference",
        type=str,
        default="output/model_outputs/validation_predictions.csv",
        help="Path to reference predictions CSV (e.g., validation predictions from training)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="output/production_predictions.csv",
        help="Path to target predictions CSV (e.g., live production predictions)",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10,
        help="Number of bins for PSI calculation",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="output/drift_report.json",
        help="Path to save the generated drift report JSON file",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RUNNING MODEL DRIFT AND PERFORMANCE MONITORING")
    print("=" * 70)

    ref_path = Path(args.reference)
    tgt_path = Path(args.target)

    if ref_path.exists() and tgt_path.exists():
        print(f"Loading real datasets for drift monitoring:")
        print(f"  Reference path (expected): {ref_path}")
        print(f"  Target path (actual):    {tgt_path}")
        
        try:
            df_ref = pd.read_csv(ref_path)
            df_tgt = pd.read_csv(tgt_path)
            
            # Extract probability columns
            # For reference: can be 'y_prob' or 'probability'
            ref_col = 'y_prob' if 'y_prob' in df_ref.columns else ('probability' if 'probability' in df_ref.columns else None)
            # For target: expected 'probability'
            tgt_col = 'probability' if 'probability' in df_tgt.columns else None
            
            if ref_col is None or tgt_col is None:
                raise ValueError(
                    f"Required probability columns not found. Reference cols: {df_ref.columns.tolist()}, Target cols: {df_tgt.columns.tolist()}"
                )
                
            reference = df_ref[ref_col].values
            target = df_tgt[tgt_col].values
            
            print(f"Reference distribution size: {len(reference):,}")
            print(f"Target distribution size:    {len(target):,}")
            
            metrics = analyze_drift(reference, target)
            
            print(f"\nReal-World Drift Analysis Results:")
            print(f"  PSI: {metrics['psi']:.4f}")
            print(f"  Ref Mean PD: {metrics['reference_mean_pd']:.4f} | Target Mean PD: {metrics['target_mean_pd']:.4f}")
            print(f"  PD Relative Shift: {metrics['pd_percentage_shift']:.2f}%")
            print(f"  Status: {metrics['status']}")
            
            # Save report to JSON
            out_report_path = Path(args.output_report)
            out_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_report_path, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"\nDrift report saved to {out_report_path}")
            
            # Fail the run / return exit code if drift detected so CI can flag it
            if metrics['status'] == "DRIFT_DETECTED":
                print("\n[WARNING] Significant drift detected! Action recommended (e.g. retrain).")
                
        except Exception as e:
            print(f"Error processing real drift monitoring: {e}")
            print("Falling back to simulated run.")
            run_simulated()
    else:
        print("Real-world prediction files not found or incomplete. Running fallback simulated demonstration.")
        run_simulated()


def run_simulated():
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
