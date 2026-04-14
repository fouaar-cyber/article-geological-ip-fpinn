# analyze_significance_final.py - CORRECTED for new JSON format
import json
import numpy as np
from scipy import stats
from pathlib import Path
import sys

def load_results_from_combined(filename):
    """Load results from combined ablation JSON file"""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # New format: list of 3 configurations
        if isinstance(data, list) and len(data) >= 2:
            print(f"✓ Loaded {len(data)} configurations from {file_path.name}")
            return data
        else:
            print(f"⚠️ Unexpected format in {filename}")
            return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def analyze_pair(baseline_config, test_config, test_name):
    """Statistical comparison between two configurations"""
    
    baseline_errors = np.array(baseline_config["raw_errors"])
    test_errors = np.array(test_config["raw_errors"])
    
    n1, n2 = len(baseline_errors), len(test_errors)
    
    # Descriptive statistics
    baseline_mean = np.mean(baseline_errors)
    baseline_std = np.std(baseline_errors, ddof=1)
    baseline_cv = baseline_std / baseline_mean * 100
    
    test_mean = np.mean(test_errors)
    test_std = np.std(test_errors, ddof=1)
    test_cv = test_std / test_mean * 100
    
    # Improvement
    abs_improvement = baseline_mean - test_mean
    rel_improvement = (abs_improvement / baseline_mean) * 100
    
    # Welch's t-test (one-tailed: baseline > test)
    t_stat, p_two_tailed = stats.ttest_ind(baseline_errors, test_errors, equal_var=False)
    p_one_tailed = p_two_tailed / 2
    
    # Mann-Whitney U (one-tailed)
    u_stat, p_mw = stats.mannwhitneyu(baseline_errors, test_errors, alternative='greater')
    
    # Cohen's d (pooled std)
    pooled_std = np.sqrt(((n1-1)*baseline_std**2 + (n2-1)*test_std**2) / (n1+n2-2))
    cohens_d = abs_improvement / pooled_std
    
    # Bootstrap 95% CI for difference
    n_bootstrap = 10000
    differences = []
    for _ in range(n_bootstrap):
        bs_baseline = np.random.choice(baseline_errors, size=n1, replace=True)
        bs_test = np.random.choice(test_errors, size=n2, replace=True)
        differences.append(np.mean(bs_baseline) - np.mean(bs_test))
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)
    
    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_cv": baseline_cv,
        "test_mean": test_mean,
        "test_std": test_std,
        "test_cv": test_cv,
        "abs_improvement": abs_improvement,
        "rel_improvement": rel_improvement,
        "p_ttest": p_one_tailed,
        "p_mw": p_mw,
        "cohens_d": cohens_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

def print_results_table(results):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS (n=15, 1500 epochs, α=0.5)")
    print("="*80)
    print(f"{'Configuration':<20} {'L₂ Error':<15} {'Std':<12} {'CV%':<8} {'Improvement':<12} {'p-value':<10}")
    print("-"*80)
    
    baseline = results["baseline"]
    print(f"{'Baseline PINN':<20} {baseline['mean']:<15.4e} {baseline['std']:<12.4e} {baseline['cv']:<8.1f} {'—':<12} {'—':<10}")
    
    for name in ["Feature Only", "Full Enhanced"]:
        if name in results:
            r = results[name]
            sig = "***" if r['p_ttest'] < 0.001 else "**" if r['p_ttest'] < 0.01 else "*" if r['p_ttest'] < 0.05 else "ns"
            print(f"{name:<20} {r['test_mean']:<15.4e} {r['test_std']:<12.4e} {r['test_cv']:<8.1f} {r['rel_improvement']:>10.1f}% {r['p_ttest']:<10.4f}{sig}")
    
    print("="*80)

def print_detailed_stats(results):
    """Print detailed statistical analysis"""
    print("\n" + "="*80)
    print("DETAILED STATISTICAL ANALYSIS")
    print("="*80)
    
    for name in ["Feature Only", "Full Enhanced"]:
        if name not in results:
            continue
            
        r = results[name]
        print(f"\n{name} vs Baseline PINN:")
        print("-"*40)
        print(f"  Baseline:     {r['baseline_mean']:.4e} ± {r['baseline_std']:.4e} (CV={r['baseline_cv']:.1f}%)")
        print(f"  {name}: {r['test_mean']:.4e} ± {r['test_std']:.4e} (CV={r['test_cv']:.1f}%)")
        print(f"  Absolute Δ:   {r['abs_improvement']:.4e}")
        print(f"  Relative Δ:   {r['rel_improvement']:.1f}%")
        print(f"\n  Statistical Tests:")
        print(f"    Welch's t-test (one-tailed): p = {r['p_ttest']:.6f}")
        print(f"    Mann-Whitney U (one-tailed): p = {r['p_mw']:.6f}")
        print(f"    Cohen's d: {r['cohens_d']:.3f} ({'Large' if r['cohens_d'] > 0.8 else 'Medium' if r['cohens_d'] > 0.5 else 'Small'} effect)")
        print(f"    95% Bootstrap CI: [{r['ci_lower']:.4e}, {r['ci_upper']:.4e}]")
        
        # Significance interpretation
        if r['p_ttest'] < 0.001 and r['cohens_d'] > 0.8:
            print(f"    ✓✓✓ HIGHLY SIGNIFICANT with LARGE EFFECT SIZE")
        elif r['p_ttest'] < 0.05:
            print(f"    ✓ SIGNIFICANT at α = 0.05")
        else:
            print(f"    ~ Not significant")

def main():
    # Find the most recent combined results file
    results_dir = Path("results_enhanced")
    combined_files = list(results_dir.glob("full_ablation_alpha*.json"))
    
    if not combined_files:
        print("❌ No combined ablation files found in results_enhanced/")
        print("Looking for individual files...")
        
        # Fallback: look for individual files
        baseline_files = list(results_dir.glob("ablation_baseline*.json"))
        feature_files = list(results_dir.glob("ablation_feature_only*.json"))
        full_files = list(results_dir.glob("ablation_full_enhanced*.json"))
        
        if not baseline_files or not feature_files:
            print("❌ Cannot find required result files")
            sys.exit(1)
        
        # Load individual files (older format)
        print("⚠️ Using individual files (manual comparison needed)")
        return
    
    # Use most recent combined file
    latest_file = sorted(combined_files)[-1]
    print(f"Analyzing: {latest_file}")
    
    # Load data
    data = load_results_from_combined(latest_file)
    if not data:
        sys.exit(1)
    
    # Extract configurations
    configs = {item["configuration"]: item for item in data}
    
    if "Baseline PINN" not in configs:
        print("❌ Baseline PINN not found in results")
        sys.exit(1)
    
    # Analyze comparisons
    results = {"baseline": {
        "mean": configs["Baseline PINN"]["mean_l2_error"],
        "std": configs["Baseline PINN"]["std_l2_error"],
        "cv": configs["Baseline PINN"]["cv_percent"]
    }}
    
    baseline_config = configs["Baseline PINN"]
    
    if "Feature Only" in configs:
        results["Feature Only"] = analyze_pair(baseline_config, configs["Feature Only"], "Feature Only")
    
    if "Full Enhanced" in configs:
        results["Full Enhanced"] = analyze_pair(baseline_config, configs["Full Enhanced"], "Full Enhanced")
    
    # Print results
    print_results_table(results)
    print_detailed_stats(results)
    
    # Save summary
    output_file = results_dir / "statistical_analysis_final.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()