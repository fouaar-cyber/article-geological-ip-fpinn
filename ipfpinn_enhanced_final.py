import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results_enhanced")
PLOT_DIR = Path("debug_plots_enhanced")
RESULTS_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# ==================== Data Generation ====================
def generate_formation_data(alpha: float, grid_size: int = 64, save_plot: bool = False) -> Dict:
    """Generate synthetic geological formation data with heterogeneity parameter alpha."""
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Permeability field with heterogeneity
    k_x = 1e-13 * (1 + alpha * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y))
    k_y = 0.5e-13 * (1 + alpha * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y))
    
    # Pressure solution (synthetic)
    P = np.sin(np.pi * X) * np.sin(np.pi * Y) * (1 + 0.2 * alpha * X * Y)
    
    # Derivatives for physics-informed loss
    dP_dx = np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y) * (1 + 0.2 * alpha * X * Y) + \
            np.sin(np.pi * X) * np.sin(np.pi * Y) * (0.2 * alpha * Y)
    dP_dy = np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y) * (1 + 0.2 * alpha * X * Y) + \
            np.sin(np.pi * X) * np.sin(np.pi * Y) * (0.2 * alpha * X)
    
    data = {
        "X": X, "Y": Y, "P": P,
        "k_x": k_x, "k_y": k_y,
        "dP_dx": dP_dx, "dP_dy": dP_dy,
        "alpha": alpha,
        "grid_size": grid_size
    }
    
    # Validation
    sol_std = np.std(P)
    perm_range = [np.min(k_x), np.max(k_x)]
    
    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im0 = axes[0].imshow(k_x, cmap='viridis')
        axes[0].set_title(f'Permeability Field (α={alpha})')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(P, cmap='jet')
        axes[1].set_title('Pressure Solution')
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(dP_dx, cmap='coolwarm')
        axes[2].set_title('dP/dx')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"formation_alpha_{alpha}.png", dpi=150)
        plt.close()
        
    print(f"  Data verified: α={alpha} | sol_std={sol_std:.3e} | perm_range=[{perm_range[0]:.1e}, {perm_range[1]:.1e}]")
    return data

# ==================== Neural Networks ====================
class PINN(nn.Module):
    """Baseline Physics-Informed Neural Network"""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        return self.net(inputs)

class SpatialAttentionIPFPINN(nn.Module):
    """Enhanced IP-FPINN with optional feature engineering and intrusive physics"""
    def __init__(self, hidden_dim: int = 64, grid_size: int = 64, 
                 use_feature: bool = True, use_intrusive: bool = True):
        super().__init__()
        self.use_feature = use_feature
        self.use_intrusive = use_intrusive
        
        # Feature extraction (optional)
        if use_feature:
            self.feature_net = nn.Sequential(
                nn.Linear(4, hidden_dim),  # x, y, k_x, k_y
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2)  # output enhanced features
            )
            input_dim = 4  # x, y, features
        else:
            input_dim = 2  # x, y only
        
        # Main prediction network
        self.main_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y, k_x=None, k_y=None):
        if self.use_feature and k_x is not None and k_y is not None:
            # Feature engineering mode
            geo_features = torch.cat([x, y, k_x, k_y], dim=1)
            enhanced_features = self.feature_net(geo_features)
            inputs = torch.cat([x, y, enhanced_features], dim=1)
        else:
            # Standard mode
            inputs = torch.cat([x, y], dim=1)
        
        return self.main_net(inputs)

# ==================== Physics-Informed Loss ====================
def physics_informed_loss(model, x, y, k_x, k_y, dP_dx_true, dP_dy_true, 
                         use_intrusive: bool):
    """Compute physics-informed loss based on Darcy's law"""
    x.requires_grad_(True)
    y.requires_grad_(True)
    
    # Forward pass
    if use_intrusive and hasattr(model, 'use_feature') and model.use_feature:
        P_pred = model(x, y, k_x, k_y)
    else:
        P_pred = model(x, y)
    
    # Compute gradients
    dP_dx = torch.autograd.grad(
        P_pred, x, 
        grad_outputs=torch.ones_like(P_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    dP_dy = torch.autograd.grad(
        P_pred, y, 
        grad_outputs=torch.ones_like(P_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Darcy's law residual
    residual_x = dP_dx + (k_x * dP_dx_true)
    residual_y = dP_dy + (k_y * dP_dy_true)
    
    physics_loss = torch.mean(residual_x**2) + torch.mean(residual_y**2)
    
    return physics_loss

# ==================== Training ====================
def train_model(model, data, epochs: int, device: torch.device, 
                model_name: str, use_feature: bool, use_intrusive: bool) -> Tuple[float, float]:
    """Train a single model with proper physics-informed loss"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Convert data to tensors
    x = torch.from_numpy(data["X"].flatten()).float().unsqueeze(1).to(device)
    y = torch.from_numpy(data["Y"].flatten()).float().unsqueeze(1).to(device)
    p_true = torch.from_numpy(data["P"].flatten()).float().unsqueeze(1).to(device)
    k_x = torch.from_numpy(data["k_x"].flatten()).float().unsqueeze(1).to(device)
    k_y = torch.from_numpy(data["k_y"].flatten()).float().unsqueeze(1).to(device)
    dP_dx_true = torch.from_numpy(data["dP_dx"].flatten()).float().unsqueeze(1).to(device)
    dP_dy_true = torch.from_numpy(data["dP_dy"].flatten()).float().unsqueeze(1).to(device)
    
    start_time = time.time()
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass
        if use_feature:
            p_pred = model(x, y, k_x, k_y)
        else:
            p_pred = model(x, y)
        
        # Data loss
        data_loss = torch.mean((p_pred - p_true)**2)
        
        # Physics-informed loss
        if use_intrusive:
            physics_loss = physics_informed_loss(
                model, x, y, k_x, k_y, dP_dx_true, dP_dy_true, use_intrusive
            )
        else:
            physics_loss = torch.tensor(0.0, device=device)
        
        # Combined loss (you can adjust weights here)
        total_loss = data_loss + physics_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if epoch % 20 == 0:
            print(f"  [{model_name}] Epoch {epoch:4d} | Loss: {total_loss.item():.3e} "
                  f"(Data: {data_loss.item():.3e}, PDE: {physics_loss.item():.3e})")
    
    train_time = time.time() - start_time
    
    # Compute final L2 error
    with torch.no_grad():
        if use_feature:
            p_pred_final = model(x, y, k_x, k_y)
        else:
            p_pred_final = model(x, y)
        
        l2_error = torch.sqrt(torch.mean((p_pred_final - p_true)**2)).item()
    
    return l2_error, train_time

# ==================== Experiment Runner (CORRECTED) ====================
def run_experiment(alpha: float, n_runs: int, epochs: int, 
                   device: torch.device, use_feature: bool, use_intrusive: bool) -> Dict:
    """
    FIXED: Run ONE model type based on flags, compare to saved baseline
    """
    print(f"\n{'='*60}")
    print(f"[Formation α={alpha}]")
    print(f"Mode: use_feature={use_feature}, use_intrusive={use_intrusive}")
    print(f"{'='*60}")
    
    data = generate_formation_data(alpha, save_plot=True)
    
    # ===== CORRECT LOGIC: Determine which model to train =====
    if use_feature or use_intrusive:
        # ========== Train IP-FPINN (Enhanced) ==========
        model_name = "IP-FPINN (Enhanced)"
        errors, times = [], []
        
        for run in range(1, n_runs + 1):
            print(f"\n  {'─'*50}")
            print(f"  Run {run}/{n_runs} - {model_name}")
            print(f"  {'─'*50}")
            
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            model = SpatialAttentionIPFPINN(
                hidden_dim=64, 
                grid_size=64,
                use_feature=use_feature,
                use_intrusive=use_intrusive
            )
            
            error, train_time = train_model(
                model, data, epochs, device, model_name,
                use_feature=use_feature, use_intrusive=use_intrusive
            )
            
            errors.append(error)
            times.append(train_time)
            print(f"    L2={error:.3e}, Time={train_time:.1f}s")
        
        # Compute stats for IP-FPINN
        mean_l2 = np.mean(errors)
        std_l2 = np.std(errors, ddof=1) if n_runs > 1 else 0.0
        cv = std_l2 / mean_l2 * 100 if mean_l2 > 0 else 0.0
        avg_time = np.mean(times)
        
        # Load or generate baseline PINN for comparison
        baseline_file = RESULTS_DIR / "baseline_reference.json"
        if not baseline_file.exists():
            print("\n  Generating baseline PINN for comparison...")
            baseline_errors = []
            for run in range(1, n_runs + 1):
                torch.manual_seed(42 + run)
                np.random.seed(42 + run)
                pinn = PINN(hidden_dim=64)
                pinn_error, _ = train_model(
                    pinn, data, epochs, device, "PINN-Baseline (Reference)",
                    use_feature=False, use_intrusive=False
                )
                baseline_errors.append(pinn_error)
                print(f"    PINN Run {run}: L2={pinn_error:.3e}")
            
            baseline_mean = np.mean(baseline_errors)
            # Save baseline for future experiments
            with open(baseline_file, "w") as f:
                json.dump({"baseline_l2": baseline_mean}, f, indent=2)
        else:
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
                baseline_mean = baseline_data["baseline_l2"]
            print(f"  Loaded baseline PINN L2 error: {baseline_mean:.3e}")
        
        # Compute improvement
        improvement = (baseline_mean - mean_l2) / baseline_mean * 100
        
        return {
            "alpha": alpha,
            "use_feature": use_feature,
            "use_intrusive": use_intrusive,
            "ipfpinn": {
                "mean_l2": mean_l2,
                "std_l2": std_l2,
                "cv": cv,
                "time": avg_time,
                "raw_errors": errors  # ✅ ADDED: Store individual run errors
            },
            "pinn_baseline": {
                "mean_l2": baseline_mean,
                "std_l2": 0.0,
                "cv": 0.0,
                "time": avg_time
            },
            "improvement_percent": improvement,
            "status": "enhanced_mode"
        }
    
    else:
        # ========== Train Baseline PINN Only ==========
        model_name = "PINN-Baseline (Only)"
        errors, times = [], []
        
        for run in range(1, n_runs + 1):
            print(f"\n  {'─'*50}")
            print(f"  Run {run}/{n_runs} - {model_name}")
            print(f"  {'─'*50}")
            
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            model = PINN(hidden_dim=64)
            
            error, train_time = train_model(
                model, data, epochs, device, model_name,
                use_feature=False, use_intrusive=False
            )
            
            errors.append(error)
            times.append(train_time)
            print(f"    L2={error:.3e}, Time={train_time:.1f}s")
        
        mean_l2 = np.mean(errors)
        std_l2 = np.std(errors, ddof=1) if n_runs > 1 else 0.0
        cv = std_l2 / mean_l2 * 100 if mean_l2 > 0 else 0.0
        avg_time = np.mean(times)
        
        return {
            "alpha": alpha,
            "use_feature": False,
            "use_intrusive": False,
            "pinn_baseline": {
                "mean_l2": mean_l2,
                "std_l2": std_l2,
                "cv": cv,
                "time": avg_time,
                "raw_errors": errors  # ✅ ADDED: Store individual run errors
            },
            "status": "baseline_only_mode"
        }

# ==================== Main Execution ====================
def main():
    parser = argparse.ArgumentParser(description="Enhanced IP-FPINN with Correct Ablation Logic")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5], help="Heterogeneity parameters")
    parser.add_argument("--use_feature", type=str, choices=["true", "false"], default="true", 
                       help="Enable feature engineering")
    parser.add_argument("--use_intrusive", type=str, choices=["true", "false"], default="true", 
                       help="Enable intrusive physics")
    
    args = parser.parse_args()
    
    # Parse boolean flags
    use_feature = args.use_feature.lower() == "true"
    use_intrusive = args.use_intrusive.lower() == "true"
    
    print("\n" + "="*80)
    print("ENHANCED IP-FPINN - CORRECTED ABLATION VERSION")
    print("="*80)
    print(f"✓ n_runs: {args.n_runs} | epochs: {args.epochs} | alphas: {args.alphas}")
    print(f"✓ use_feature: {use_feature} | use_intrusive: {use_intrusive}")
    print("="*80)
    print(f"\nUsing device: {DEVICE}")
    
    # Run experiment
    results = []
    for alpha in args.alphas:
        result = run_experiment(
            alpha=alpha,
            n_runs=args.n_runs,
            epochs=args.epochs,
            device=DEVICE,
            use_feature=use_feature,
            use_intrusive=use_intrusive
        )
        results.append(result)
        
        # Print summary
        if result["status"] == "enhanced_mode":
            print(f"\n{'='*60}")
            print(f"SUMMARY α={alpha}:")
            print(f"  IP-FPINN L2: {result['ipfpinn']['mean_l2']:.3e} ± {result['ipfpinn']['std_l2']:.3e}")
            print(f"  PINN Baseline L2: {result['pinn_baseline']['mean_l2']:.3e}")
            print(f"  Improvement: {result['improvement_percent']:.1f}%")
            print(f"  Runtime: {result['ipfpinn']['time']:.1f}s")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"BASELINE ONLY α={alpha}:")
            print(f"  PINN L2: {result['pinn_baseline']['mean_l2']:.3e} ± {result['pinn_baseline']['std_l2']:.3e}")
            print(f"  Runtime: {result['pinn_baseline']['time']:.1f}s")
            print(f"{'='*60}")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if use_feature and use_intrusive:
        filename = RESULTS_DIR / f"q1_ultimate_results_{timestamp}.json"
    elif not use_feature and not use_intrusive:
        filename = RESULTS_DIR / f"ablation_baseline_{timestamp}.json"
    elif use_feature and not use_intrusive:
        filename = RESULTS_DIR / f"ablation_feature_only_{timestamp}.json"
    else:
        filename = RESULTS_DIR / f"ablation_intrusive_only_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {filename}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()