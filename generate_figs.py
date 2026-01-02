import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories
os.makedirs('figs', exist_ok=True)
os.makedirs('paper', exist_ok=True)

# Load complete results
with open('results_enhanced/q1_all_results.json', 'r') as f:
    data = json.load(f)

# Convert to arrays
alphas = [d['alpha'] for d in data]
ipfpinn_errors = [d['ipfpinn']['mean_l2'] for d in data]
pinn_errors = [d['pinn']['mean_l2'] for d in data]
ipfpinn_std = [d['ipfpinn']['std_l2'] for d in data]
pinn_std = [d['pinn']['std_l2'] for d in data]
ipfpinn_cv = [d['ipfpinn']['cv'] for d in data]
pinn_cv = [d['pinn']['cv'] for d in data]
times_ip = [d['ipfpinn']['time'] for d in data]
times_pinn = [d['pinn']['time'] for d in data]

# Set publication quality style
plt.style.use('default')
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5
})

# Figure 1: Error Comparison with Error Bars
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(alphas))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, ipfpinn_errors, width, 
               label='IP-FPINN', color='#2E86AB', 
               yerr=ipfpinn_std, capsize=5, alpha=0.85,
               edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, pinn_errors, width, 
               label='PINN Baseline', color='#A23B72',
               yerr=pinn_std, capsize=5, alpha=0.85,
               edgecolor='black', linewidth=0.5)

# Formatting
ax.set_xlabel('Geological Formation (α)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative $L^2$ Error', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Fractured\n(α=0.3)', 'Heterogeneous\n(α=0.5)', 'Layered\n(α=0.7)', 'Homogeneous\n(α=1.0)'])
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (ip, pn) in enumerate(zip(ipfpinn_errors, pinn_errors)):
    ax.text(i - width/2, ip + ipfpinn_std[i] + 0.002, f'{ip:.3f}', 
            ha='center', va='bottom', fontsize=10)
    ax.text(i + width/2, pn + pinn_std[i] + 0.002, f'{pn:.3f}', 
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figs/error_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: CV (Stability) Comparison
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(x - width/2, ipfpinn_cv, width, label='IP-FPINN', 
       color='#2E86AB', alpha=0.85, edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, pinn_cv, width, label='PINN', 
       color='#A23B72', alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Geological Formation (α)', fontsize=12, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Fractured\n(α=0.3)', 'Heterogeneous\n(α=0.5)', 'Layered\n(α=0.7)', 'Homogeneous\n(α=1.0)'])
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for i, (ip_cv, pn_cv) in enumerate(zip(ipfpinn_cv, pinn_cv)):
    ax.text(i - width/2, ip_cv + 0.1, f'{ip_cv:.2f}%', 
            ha='center', va='bottom', fontsize=10)
    ax.text(i + width/2, pn_cv + 0.1, f'{pn_cv:.2f}%', 
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figs/cv_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Computational Cost vs Accuracy Trade-off
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(times_pinn, pinn_errors, s=100, label='PINN', color='#A23B72', 
           marker='s', alpha=0.8, edgecolors='black', linewidth=0.5)
ax.scatter(times_ip, ipfpinn_errors, s=100, label='IP-FPINN', color='#2E86AB', 
           marker='o', alpha=0.8, edgecolors='black', linewidth=0.5)

ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative $L^2$ Error', fontsize=12, fontweight='bold')
ax.set_title('Accuracy vs Computational Cost', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')

# Annotate points
for i, alpha in enumerate(alphas):
    ax.annotate(f'α={alpha}', (times_pinn[i], pinn_errors[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.annotate(f'α={alpha}', (times_ip[i], ipfpinn_errors[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig('figs/cost_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Figures generated successfully!")
print("📁 Files created:")
print("   - figs/error_comparison.png")
print("   - figs/cv_comparison.png") 
print("   - figs/cost_accuracy_tradeoff.png")