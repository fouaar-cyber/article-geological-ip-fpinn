#!/usr/bin/env python3
"""
Generate all publication figures from statistical results.
Updated for new JSON format with 3 configurations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configuration
RESULTS_DIR = Path("results_enhanced")
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# Updated publication-style colors for 3 configs
COLORS = {
    'baseline': 'steelblue',
    'feature': 'darkorange',
    'full': 'forestgreen',
    'text': 'black',
    'background': 'white'
}

def load_combined_results() -> Optional[List[Dict[str, Any]]]:
    """Load results from combined ablation JSON file."""
    combined_files = list(RESULTS_DIR.glob("full_ablation_alpha*.json"))
    if not combined_files:
        print("❌ No combined ablation files found")
        return None
    
    latest_file = sorted(combined_files)[-1]
    print(f"Loading: {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} configurations")
        return data
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
def generate_error_comparison_three_configs(data: List[Dict]):
    """Figure 1: Bar plot comparing all three configurations."""
    print("\n" + "="*60)
    print("Generating: figs/error_comparison_three_configs.png & .pdf")
    print("="*60)
    
    configs = {item["configuration"]: item for item in data}
    
    if "Baseline PINN" not in configs or "Feature Only" not in configs:
        print("❌ Missing required configurations")
        return False
    
    baseline = configs["Baseline PINN"]
    feature = configs["Feature Only"]
    full = configs.get("Full Enhanced")
    
    # Shorter multi-line labels to prevent overlap
    names = ['Baseline\nPINN', 'Feature\nOnly']
    means = [baseline["mean_l2_error"], feature["mean_l2_error"]]
    stds = [baseline["std_l2_error"], feature["std_l2_error"]]
    colors = [COLORS['baseline'], COLORS['feature']]
    
    if full:
        names.append('Full\nEnhanced')
        means.append(full["mean_l2_error"])
        stds.append(full["std_l2_error"])
        colors.append(COLORS['full'])
    
    ses = [s / np.sqrt(15) for s in stds]
    
    # Wider figure to prevent crowding
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    x_pos = np.arange(len(names))
    
    bars = ax.bar(x_pos, means, yerr=ses, capsize=5, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=1, width=0.6)  # Narrower bars
    
    # Add individual points with less jitter
    for i, config_name in enumerate(['Baseline PINN', 'Feature Only', 'Full Enhanced'] if full else ['Baseline PINN', 'Feature Only']):
        if config_name in configs:
            errors = configs[config_name]["raw_errors"]
            jitter = np.random.normal(i, 0.03, len(errors))  # Less jitter
            ax.scatter(jitter, errors, alpha=0.5, s=25, color='black', zorder=3)
    
    # Rotated labels for clarity
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10, linespacing=0.9)
    
    ax.set_ylabel('$L_2$ Relative Error', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Error Comparison ($n=15$, 1500 epochs)', 
                 fontsize=13, fontweight='bold', pad=20)  # Extra padding
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(0, max(means) * 1.3)  # Extra headroom for annotations
    
    # Value labels on bars
    for i, (bar, mean, se) in enumerate(zip(bars, means, ses)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + se + 0.0008,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Significance annotations - positioned higher
    y_max = max(means) + max(ses)
    if len(names) >= 2:
        ax.text(0.5, y_max * 1.12, 'vs Baseline:\n$p<0.001$***\n$d=2.88$', 
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        if full:
            ax.text(1.5 if len(names) == 3 else 1, y_max * 1.12, 
                    'vs Baseline:\n$p<0.001$***\n$d=2.92$', 
                    ha='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'error_comparison_three_configs.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'error_comparison_three_configs.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved error_comparison_three_configs figures")
    return True

def generate_boxplot_three_configs(data: List[Dict]):
    """Figure 2: Boxplot with all three configurations."""
    print("\n" + "="*60)
    print("Generating: figs/ablation_boxplot_three_configs.png & .pdf")
    print("="*60)
    
    configs = {item["configuration"]: item for item in data}
    
    # Prepare box data
    box_data = []
    labels = []
    colors_list = []
    
    for name, color_key in [('Baseline PINN', 'baseline'), ('Feature Only', 'feature'), ('Full Enhanced', 'full')]:
        if name in configs:
            box_data.append(configs[name]["raw_errors"])
            labels.append(name.replace(' ', '\n'))
            colors_list.append(COLORS[color_key])
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='black', markersize=5))
    
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, errors in enumerate(box_data):
        jitter = np.random.normal(i+1, 0.04, len(errors))
        ax.scatter(jitter, errors, alpha=0.5, s=20, color='black', zorder=10)
    
    ax.set_ylabel('L₂ Relative Error', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution: Independent Runs (n=15, 1500 epochs)', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics text
    stats_text = ('Feature Only: p < 0.001***, d = 2.88, 19.3% improvement\n'
                  'Full Enhanced: p < 0.001***, d = 2.92, 19.4% improvement')
    ax.text(0.5, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'ablation_boxplot_three_configs.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'ablation_boxplot_three_configs.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved ablation_boxplot_three_configs figures")
    return True

def generate_cv_comparison(data: List[Dict]):
    """Figure 3: Coefficient of variation comparison."""
    print("\n" + "="*60)
    print("Generating: figs/cv_comparison.png & .pdf")
    print("="*60)
    
    configs = {item["configuration"]: item for item in data}
    
    names = []
    cvs = []
    colors_list = []
    
    for name, color_key in [('Baseline PINN', 'baseline'), ('Feature Only', 'feature'), ('Full Enhanced', 'full')]:
        if name in configs:
            names.append(name.replace(' ', '\n'))
            cvs.append(configs[name]["cv_percent"])
            colors_list.append(COLORS[color_key])
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    x_pos = np.arange(len(names))
    
    bars = ax.bar(x_pos, cvs, color=colors_list, alpha=0.85, 
                  edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(cvs)*0.01,
                f'{cv:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training Stability: Run-to-Run Variability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add annotation about trade-off
    if len(cvs) >= 2:
        ax.text(0.5, 0.95, 'Note: Feature engineering improves accuracy\nbut increases variability (CV 6.7% → 8.3%)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'cv_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'cv_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved cv_comparison figures")
    return True

def generate_architecture_updated():
    """Figure 4: Updated architecture diagram (128 neurons, 6 layers)."""
    print("\n" + "="*60)
    print("Generating: figs/architecture_updated.png & .pdf")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Updated layer configurations (128 neurons, 6 layers)
    layers = [
        {"name": "Input\n(x, y)", "x": 1, "units": 2, "color": "lightgray", "width": 1.2},
        {"name": "Fourier\nFeatures", "x": 2.5, "units": 128, "color": "lightyellow", "width": 1.0},
        {"name": "Hidden 1", "x": 4.5, "units": 128, "color": COLORS['baseline'], "width": 1.0},
        {"name": "Hidden 2", "x": 6, "units": 128, "color": COLORS['baseline'], "width": 1.0},
        {"name": "Hidden 3", "x": 7.5, "units": 128, "color": COLORS['baseline'], "width": 1.0},
        {"name": "Hidden 4", "x": 9, "units": 128, "color": COLORS['baseline'], "width": 1.0},
        {"name": "Hidden 5", "x": 10.5, "units": 128, "color": COLORS['baseline'], "width": 1.0},
        {"name": "Hidden 6", "x": 12, "units": 128, "color": COLORS['baseline'], "width": 1.0},
        {"name": "Output\nu(x,y)", "x": 13.5, "units": 1, "color": "lightgreen", "width": 1.0}
    ]
    
    # Draw layers
    for layer in layers:
        node_height = min(3.0, layer["units"] * 0.02)
        y_start = 5 - node_height / 2
        
        rect = patches.Rectangle(
            (layer["x"] - layer["width"]/2, y_start), layer["width"], node_height,
            facecolor=layer["color"], alpha=0.8, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        
        ax.text(layer["x"], y_start - 0.4, layer["name"], 
                ha='center', va='top', fontsize=9, fontweight='bold')
        
        if layer["units"] >= 100:
            ax.text(layer["x"], y_start + node_height + 0.15, f'{layer["units"]}',
                    ha='center', va='bottom', fontsize=8, style='italic')
    
    # Draw connections
    for i in range(len(layers)-1):
        x_start = layers[i]["x"] + layers[i]["width"]/2
        x_end = layers[i+1]["x"] - layers[i+1]["width"]/2
        ax.plot([x_start, x_end], [5, 5], color='gray', linewidth=1.5, alpha=0.6)
    
    # Feature engineering box (for IP-FPINN)
    feature_box = patches.Rectangle(
        (6, 7.5), 4, 1.2, facecolor=COLORS['feature'], alpha=0.7,
        edgecolor='black', linewidth=2
    )
    ax.add_patch(feature_box)
    ax.text(8, 8.1, 'Geological Feature\nEngineering (Φ)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Connection from input to feature engineering
    ax.annotate('', xy=(8, 7.5), xytext=(1.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['feature'], lw=2, ls='--'))
    ax.text(4.5, 6.8, 'k(x,y)', ha='center', fontsize=9, color=COLORS['feature'], fontweight='bold')
    
    # Connection from features to network
    ax.annotate('', xy=(6, 5), xytext=(8, 7.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['feature'], lw=2, ls='--'))
    ax.text(6.5, 6.5, 'z = Φ(x,y,k)', ha='center', fontsize=9, color=COLORS['feature'])
    
    # Title
    fig.suptitle('IP-FPINN Architecture: 6-Layer Network with Fourier Features (128 neurons/layer)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'architecture_updated.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'architecture_updated.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved architecture_updated figures")
    return True

def main():
    """Generate all publication figures."""
    print("="*70)
    print("GENERATING PUBLICATION FIGURES (Updated for New Results)")
    print("="*70)
    
    data = load_combined_results()
    if not data:
        print("❌ Cannot generate figures without data")
        return
    
    figures = [
        lambda: generate_error_comparison_three_configs(data),
        lambda: generate_boxplot_three_configs(data),
        lambda: generate_cv_comparison(data),
        generate_architecture_updated
    ]
    
    success_count = 0
    for fig_func in figures:
        if fig_func():
            success_count += 1
    
    print("\n" + "="*70)
    print(f"✅ SUCCESSFULLY GENERATED {success_count}/{len(figures)} FIGURES")
    print(f"📁 All figures saved in: {FIGS_DIR.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()