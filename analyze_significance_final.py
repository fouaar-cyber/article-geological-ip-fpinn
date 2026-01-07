# analyze_significance_final.py
import json
import numpy as np
from scipy import stats
from pathlib import Path

def load_raw_errors(filename, model_type="pinn_baseline"):
    """Load raw error list from JSON results file"""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            return None
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract raw errors
        if model_type in data[0] and "raw_errors" in data[0][model_type]:
            errors = data[0][model_type]["raw_errors"]
            print(f"✓ Loaded {len(errors)} errors from {file_path.name}")
            return errors
        else:
            print(f"⚠️  {filename} missing 'raw_errors' key. Available keys: {list(data[0].keys())}")
            if model_type in data[0]:
                print(f"  Available in '{model_type}': {list(data[0][model_type].keys())}")
            return None
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

# Exact file paths from your runs
baseline_file = "results_enhanced/ablation_baseline_20260102_181443.json"
feature_file = "results_enhanced/ablation_feature_only_20260102_182308.json"

print("="*80)
print("FINAL STATISTICAL ANALYSIS: Feature Engineering vs Baseline PINN")
print("="*80)
print(f"Baseline: n=15 ({baseline_file})")
print(f"Feature:  n=15 ({feature_file})")
print("="*80)

# Load raw errors
baseline_errors = load_raw_errors(baseline_file, "pinn_baseline")
feature_errors = load_raw_errors(feature_file, "ipfpinn")

# Verify data
if baseline_errors is None or feature_errors is None:
    print("\n❌ Cannot proceed without both datasets. Exiting.")
    exit(1)

if len(baseline_errors) != 15 or len(feature_errors) != 15:
    print(f"\n⚠️  Warning: Expected n=15 for both, got n={len(baseline_errors)} baseline, n={len(feature_errors)} feature")

# Verify ranges
print(f"\nBaseline errors range: [{min(baseline_errors):.4e}, {max(baseline_errors):.4e}]")
print(f"Feature errors range:  [{min(feature_errors):.4e}, {max(feature_errors):.4e}]")

# Descriptive statistics
baseline_mean = np.mean(baseline_errors)
baseline_std = np.std(baseline_errors, ddof=1)
baseline_cv = baseline_std / baseline_mean * 100

feature_mean = np.mean(feature_errors)
feature_std = np.std(feature_errors, ddof=1)
feature_cv = feature_std / feature_mean * 100

# Welch's t-test (unequal variances, two-tailed)
t_stat, p_value_two_tailed = stats.ttest_ind(baseline_errors, feature_errors, equal_var=False)

# One-tailed p-value (appropriate for directional hypothesis: baseline > feature)
p_value_one_tailed = p_value_two_tailed / 2

# Effect size (Cohen's d) - using pooled standard deviation
n1, n2 = len(baseline_errors), len(feature_errors)
pooled_std = np.sqrt(((n1-1)*baseline_std**2 + (n2-1)*feature_std**2) / (n1+n2-2))
cohens_d = (baseline_mean - feature_mean) / pooled_std

# Mann-Whitney U test (non-parametric, one-tailed)
u_stat, p_value_mw = stats.mannwhitneyu(baseline_errors, feature_errors, alternative='greater')

# Bootstrap 95% CI for the mean difference
n_bootstrap = 10000
differences = []
for _ in range(n_bootstrap):
    bs_baseline = np.random.choice(baseline_errors, size=n1, replace=True)
    bs_feature = np.random.choice(feature_errors, size=n2, replace=True)
    differences.append(np.mean(bs_baseline) - np.mean(bs_feature))
ci_lower = np.percentile(differences, 2.5)
ci_upper = np.percentile(differences, 97.5)

print("\n" + "-"*80)
print("DESCRIPTIVE STATISTICS")
print("-"*80)
print(f"Baseline PINN (n={n1}): {baseline_mean:.4e} ± {baseline_std:.4e} (CV = {baseline_cv:.1f}%)")
print(f"Feature Only (n={n2}):  {feature_mean:.4e} ± {feature_std:.4e} (CV = {feature_cv:.1f}%)")
print(f"Absolute improvement: {baseline_mean - feature_mean:.4e}")
print(f"Relative improvement: {(baseline_mean - feature_mean)/baseline_mean*100:.1f}%")

print("\n" + "-"*80)
print("STATISTICAL TESTS")
print("-"*80)

test_results = []
# Welch's t-test (parametric)
if p_value_one_tailed < 0.001:
    sig_level = "***"
elif p_value_one_tailed < 0.01:
    sig_level = "**"
elif p_value_one_tailed < 0.05:
    sig_level = "*"
else:
    sig_level = "ns"
print(f"Welch's t-test (one-tailed): p = {p_value_one_tailed:.4f} {sig_level}")
test_results.append(("Welch's t-test", p_value_one_tailed))

# Mann-Whitney U test (non-parametric)
if p_value_mw < 0.001:
    sig_level = "***"
elif p_value_mw < 0.01:
    sig_level = "**"
elif p_value_mw < 0.05:
    sig_level = "*"
else:
    sig_level = "ns"
print(f"Mann-Whitney U (one-tailed): p = {p_value_mw:.4f} {sig_level}")
test_results.append(("Mann-Whitney U", p_value_mw))

# Effect size
effect_size = 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
print(f"Cohen's d (effect size): {cohens_d:.3f} ({effect_size})")

print("\n" + "-"*80)
print("INTERPRETATION")
print("-"*80)

# Overall significance (conservative: require both tests to be significant)
significant_count = sum(1 for _, p in test_results if p < 0.05)
if significant_count == len(test_results):
    print("✓✓✓ BOTH TESTS SIGNIFICANT - Strong evidence for improvement")
elif any(p < 0.05 for _, p in test_results):
    print("✓✓ AT LEAST ONE TEST SIGNIFICANT - Moderate evidence for improvement")
else:
    print("✗ NO TESTS SIGNIFICANT - Insufficient evidence")

# Evidence strength
if p_value_one_tailed < 0.01:
    print("✓✓ HIGHLY SIGNIFICANT: p < 0.01 (one-tailed)")
elif p_value_one_tailed < 0.05:
    print("✓ SIGNIFICANT: p < 0.05 (one-tailed)")
else:
    print("~ Not significant at α = 0.05")

if abs(cohens_d) > 0.8:
    print("✓ LARGE EFFECT SIZE: Cohen's d > 0.8")
elif abs(cohens_d) > 0.5:
    print("~ Medium effect size")
else:
    print("~ Small effect size")

# Check confidence interval
if ci_lower > 0:
    print(f"✓ Confidence interval entirely positive: improvement is robust")
else:
    print(f"~ Confidence interval includes zero: some uncertainty")

print("\n" + "-"*80)
print(f"95% Bootstrap CI for improvement: [{ci_lower:.4e}, {ci_upper:.4e}]")
print("="*80)

# Save results to file
results_summary = {
    "baseline": {
        "mean": baseline_mean,
        "std": baseline_std,
        "cv": baseline_cv,
        "n": n1
    },
    "feature": {
        "mean": feature_mean,
        "std": feature_std,
        "cv": feature_cv,
        "n": n2
    },
    "improvement": {
        "absolute": baseline_mean - feature_mean,
        "relative_percent": (baseline_mean - feature_mean)/baseline_mean * 100
    },
    "statistical_tests": {
        "welch_ttest_one_tailed": {
            "t_statistic": t_stat,
            "p_value": p_value_one_tailed
        },
        "mann_whitney_u": {
            "u_statistic": u_stat,
            "p_value": p_value_mw
        },
        "cohens_d": cohens_d,
        "bootstrap_ci_95": [ci_lower, ci_upper]
    }
}

output_file = "results_enhanced/statistical_analysis_summary.json"
with open(output_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✓ Statistical summary saved to: {output_file}")
print("="*80)