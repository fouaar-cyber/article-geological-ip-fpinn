#!/usr/bin/env python3
"""
Generate all publication figures from statistical results and neural network architecture.
Run this after ipfpinn_enhanced_final.py and analyze_significance_final.py

Expected files:
- results_enhanced/ablation_baseline_*.json
- results_enhanced/ablation_feature_only_*.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Optional, Dict, Any

# Configuration
RESULTS_DIR = Path("results_enhanced")
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(exist_ok=True)

# Publication-style colors
COLORS = {
    'baseline': 'steelblue',
    'feature': 'darkorange',
    'text': 'black',
    'background': 'white'
}

def load_errors(filename: str, model_type: str = "pinn_baseline") -> Optional[List[float]]:
    """Load raw error list from JSON results file with robust error handling."""
    try:
        filepath = RESULTS_DIR / filename
        with open(filepath, 'r') as f:
            data: List[Dict[str, Any]] = json.load(f)
        
        # Navigate nested structure safely
        if data and model_type in data[0] and "raw_errors" in data[0][model_type]:
            errors = data[0][model_type]["raw_errors"]
            print(f"✓ Loaded {len(errors)} errors from {filename}")
            return errors
        else:
            print(f"❌ Missing 'raw_errors' key structure in {filename}")
            return None
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in {filename}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error loading {filename}: {e}")
        return None

def generate_error_comparison():
    """Figure 1: Bar plot of mean errors with standard error bars and individual points."""
    print("\n" + "="*60)
    print("Generating: figs/error_comparison.png & .pdf")
    print("="*60)
    
    # Find latest result files dynamically
    baseline_files = list(RESULTS_DIR.glob("ablation_baseline_*.json"))
    feature_files = list(RESULTS_DIR.glob("ablation_feature_only_*.json"))
    
    if not baseline_files or not feature_files:
        print("❌ Cannot find result files in results_enhanced/")
        return False
    
    # Use most recent files
    baseline_file = sorted(baseline_files)[-1]
    feature_file = sorted(feature_files)[-1]
    
    baseline_errors = load_errors(baseline_file.name, "pinn_baseline")
    feature_errors = load_errors(feature_file.name, "ipfpinn")
    
    if baseline_errors is None or feature_errors is None:
        return False
    
    # Calculate statistics
    baseline_mean = np.mean(baseline_errors)
    baseline_se = np.std(baseline_errors, ddof=1) / np.sqrt(len(baseline_errors))
    feature_mean = np.mean(feature_errors)
    feature_se = np.std(feature_errors, ddof=1) / np.sqrt(len(feature_errors))
    
    # Create publication-quality plot
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    models = ['Baseline PINN', 'IPFPINN\n(Feature Only)']
    errors = [baseline_mean, feature_mean]
    std_errors = [baseline_se, feature_se]
    
    bars = ax.bar(models, errors, yerr=std_errors, capsize=5,
                  color=[COLORS['baseline'], COLORS['feature']], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    
    # Add individual data points with jitter
    for i, errors_list in enumerate([baseline_errors, feature_errors]):
        x = np.random.normal(i, 0.05, len(errors_list))
        ax.scatter(x, errors_list, alpha=0.6, s=25, color='black', zorder=3)
    
    # Formatting
    ax.set_ylabel('L₂ Relative Error', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy: Baseline vs Feature Engineering', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add significance annotation
    y_max = max(errors) + max(std_errors)
    ax.text(0.5, y_max * 1.15, 'p = 0.020*', ha='center', fontsize=11, 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'error_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'error_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved error_comparison figures")
    return True

def generate_cv_comparison():
    """Figure 2: Coefficient of variation comparison with detailed annotations."""
    print("\n" + "="*60)
    print("Generating: figs/cv_comparison.png & .pdf")
    print("="*60)
    
    baseline_file = sorted(RESULTS_DIR.glob("ablation_baseline_*.json"))[-1]
    feature_file = sorted(RESULTS_DIR.glob("ablation_feature_only_*.json"))[-1]
    
    baseline_errors = load_errors(baseline_file.name, "pinn_baseline")
    feature_errors = load_errors(feature_file.name, "ipfpinn")
    
    if baseline_errors is None or feature_errors is None:
        return False
    
    # Calculate CV
    baseline_cv = np.std(baseline_errors, ddof=1) / np.mean(baseline_errors) * 100
    feature_cv = np.std(feature_errors, ddof=1) / np.mean(feature_errors) * 100
    
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    models = ['Baseline PINN', 'IPFPINN\n(Feature Only)']
    cvs = [baseline_cv, feature_cv]
    
    bars = ax.bar(models, cvs, color=[COLORS['baseline'], COLORS['feature']], 
                  alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, cv in zip(bars, cvs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(cvs)*0.02,
                f'{cv:.1f}%', ha='center', va='bottom', fontsize=11, 
                fontweight='bold')
    
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Stability: Training Run Variability', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'cv_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'cv_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved cv_comparison figures")
    return True

def generate_ablation_boxplot():
    """Figure 3: Boxplot distribution with detailed statistics overlay."""
    print("\n" + "="*60)
    print("Generating: figs/ablation_boxplot.png & .pdf")
    print("="*60)
    
    baseline_file = sorted(RESULTS_DIR.glob("ablation_baseline_*.json"))[-1]
    feature_file = sorted(RESULTS_DIR.glob("ablation_feature_only_*.json"))[-1]
    
    baseline_errors = load_errors(baseline_file.name, "pinn_baseline")
    feature_errors = load_errors(feature_file.name, "ipfpinn")
    
    if baseline_errors is None or feature_errors is None:
        return False
    
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    box_data = [baseline_errors, feature_errors]
    labels = ['Baseline PINN', 'IPFPINN\n(Feature Only)']
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    boxprops=dict(linewidth=1.2),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    
    bp['boxes'][0].set_facecolor(COLORS['baseline'])
    bp['boxes'][1].set_facecolor(COLORS['feature'])
    
    # Add individual data points
    for i, errors_list in enumerate(box_data):
        x = np.random.normal(i+1, 0.05, len(errors_list))
        ax.scatter(x, errors_list, alpha=0.6, s=25, color='black', zorder=10)
    
    ax.set_ylabel('L₂ Relative Error', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution: Independent Runs (n=15)', 
                 fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add statistics annotation
    max_y = max(max(baseline_errors), max(feature_errors))
    y_range = max_y - min(min(baseline_errors), min(feature_errors))
    offset = y_range * 0.15
    
    ax.text(1.5, max_y + offset, 
            'p = 0.020* (one-tailed Welch\'s t-test)\nCohen\'s d = 0.82 (large effect)',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.25", facecolor="lightgray", alpha=0.8, 
                     linewidth=0.5))
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'ablation_boxplot.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'ablation_boxplot.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved ablation_boxplot figures")
    return True

def generate_architecture_diagram():
    """Figure 4: Publication-quality neural network architecture diagram."""
    print("\n" + "="*60)
    print("Generating: figs/architecture.png & .pdf")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Layer configurations
    layers = [
        {"name": "Input\n(x, y, t)", "x": 1, "units": 3, "color": "lightgray"},
        {"name": "Hidden 1", "x": 3, "units": 50, "color": COLORS['baseline']},
        {"name": "Hidden 2", "x": 5, "units": 50, "color": COLORS['baseline']},
        {"name": "Hidden 3", "x": 7, "units": 50, "color": COLORS['baseline']},
        {"name": "Output\nu(x,y,t)", "x": 9, "units": 1, "color": "lightgreen"}
    ]
    
    # Draw layers
    layer_patches = []
    for i, layer in enumerate(layers):
        # Create vertical stack of nodes
        node_height = min(2.0, layer["units"] * 0.15)  # Scale with units
        y_start = 4 - node_height / 2
        
        # Draw layer box
        rect = patches.Rectangle(
            (layer["x"] - 0.5, y_start), 1, node_height,
            facecolor=layer["color"], alpha=0.7, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        layer_patches.append(rect)
        
        # Add layer label
        ax.text(layer["x"], y_start - 0.5, layer["name"], 
                ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Add unit count
        if i > 0 and i < len(layers)-1:
            ax.text(layer["x"], y_start + node_height + 0.2, f'{layer["units"]} neurons',
                    ha='center', va='bottom', fontsize=9, style='italic')
    
    # Draw connections
    for i in range(len(layers)-1):
        x_start = layers[i]["x"] + 0.5
        x_end = layers[i+1]["x"] - 0.5
        y_start = 4
        y_end = 4
        
        # Draw connecting line
        ax.plot([x_start, x_end], [y_start, y_end], 
                color='gray', linewidth=1.5, alpha=0.6)
        
        # Draw arrows
        if i == len(layers)-2:  # Last connection
            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Add feature engineering component
    feature_box = patches.Rectangle(
        (5.5, 6), 3, 1, facecolor=COLORS['feature'], alpha=0.7,
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(feature_box)
    ax.text(7, 6.5, 'Intrusive Feature\nEngineering', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Connect feature to input
    ax.plot([1.5, 7], [4.5, 6], color=COLORS['feature'], linewidth=2, 
            linestyle='--', alpha=0.8)
    ax.text(4, 5.5, 'Physics-informed\nfeatures', ha='center', va='center',
            fontsize=9, color=COLORS['feature'], fontweight='bold')
    
    # Add title
    fig.suptitle('IPFPINN Architecture: Physics-Informed Feature Enhancement', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'architecture.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Saved architecture figures")
    return True

def main():
    """Generate all publication figures."""
    print("="*70)
    print("GENERATING PUBLICATION FIGURES")
    print("="*70)
    
    figures = [
        generate_error_comparison,
        generate_cv_comparison,
        generate_ablation_boxplot,
        generate_architecture_diagram
    ]
    
    success_count = 0
    for fig_func in figures:
        if fig_func():
            success_count += 1
    
    print("\n" + "="*70)
    print(f"✅ SUCCESSFULLY GENERATED {success_count}/{len(figures)} FIGURES")
    print(f"📁 All figures saved in: {FIGS_DIR.absolute()}")
    for fig_name in ['error_comparison', 'cv_comparison', 'ablation_boxplot', 'architecture']:
        png_path = FIGS_DIR / f'{fig_name}.png'
        pdf_path = FIGS_DIR / f'{fig_name}.pdf'
        if png_path.exists():
            print(f"   - {fig_name}.png & .pdf")
    print("="*70)

if __name__ == "__main__":
    main()