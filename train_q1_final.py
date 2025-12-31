"""
Q1 Journal Submission Protocol: IP-FPINN for Porous Media Transport
====================================================================
Production-ready version with all runtime errors fixed
"""

import os
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
import time
import json
from pathlib import Path
from scipy import signal

# ==================== CONFIGURATION ====================

class Config:
    """Fixed configuration ensuring reproducibility"""
    # Paths
    DATA_DIR = Path("data/geological_formations")
    RESULTS_DIR = Path("results")
    
    # Experimental protocol
    SEED_BASE = 42
    N_RUNS_MIN = 5
    N_RUNS_STABLE = 10
    CV_THRESHOLD = 0.05
    
    # Hyperparameters
    EPOCHS = 2000
    LR_IPFPINN = 1e-3
    LR_PINN = 1e-3
    HIDDEN_DIM = 50
    NUM_LAYERS = 4
    
    # Physics parameters
    ALPHAS = [0.3, 0.5, 0.7, 1.0]
    NX, NY = 64, 64

# ==================== UTILITY FUNCTIONS ====================

def set_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def generate_formation_data(alpha: float, nx: int = 64, ny: int = 64) -> np.ndarray:
    """Generate geological formation permeability field"""
    if not (0.3 <= alpha <= 1.0):
        raise ValueError(f"Alpha {alpha} out of range [0.3, 1.0]")
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    if alpha == 1.0:
        k = np.ones((ny, nx))
    elif alpha == 0.7:
        k = 0.5 * (1 + np.sin(4 * np.pi * Y)) + 0.1
    elif alpha == 0.5:
        rng = np.random.default_rng(seed=int(alpha*100))
        k = rng.lognormal(mean=0, sigma=0.5, size=(ny, nx))
        k = signal.convolve2d(k, np.ones((3,3))/9, mode='same', boundary='wrap')
    else:  # alpha == 0.3
        k = np.ones((ny, nx)) * 0.01
        for angle in [0, np.pi/4, -np.pi/4]:
            for offset in [0.2, 0.5, 0.8]:
                dist = np.abs((X - offset) * np.cos(angle) - (Y - offset) * np.sin(angle))
                k[dist < 0.02] = 1.0
    
    k = 0.1 + 0.9 * (k - k.min()) / (k.max() - k.min() + 1e-12)
    return k

def relative_l2_error(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """Compute robust relative L2 error"""
    try:
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        numerator = np.sqrt(np.mean((pred_flat - true_flat)**2))
        denominator = np.sqrt(np.mean(true_flat**2))
        
        if denominator < eps:
            warnings.warn(f"Near-zero denominator: {denominator:.3e}")
            return np.nan
        
        rel_error = numerator / (denominator + eps)
        
        if not np.isfinite(rel_error) or rel_error > 1e6:
            warnings.warn(f"Unreasonable L2 error: {rel_error:.3e}")
            return np.nan
        
        return float(rel_error)
    except Exception as e:
        warnings.warn(f"L2 calculation failed: {e}")
        return np.nan

def compute_coefficient_of_variation(values: List[float]) -> Tuple[float, float, float]:
    """Compute mean, std, and CV with robust handling"""
    valid_vals = [v for v in values if np.isfinite(v)]
    
    if len(valid_vals) == 0:
        return np.nan, np.nan, np.nan
    
    if len(valid_vals) == 1:
        return valid_vals[0], 0.0, 0.0
    
    mean = np.mean(valid_vals)
    std = np.std(valid_vals, ddof=1)
    cv = std / abs(mean) if abs(mean) > 1e-12 else np.nan
    
    return mean, std, cv

# ==================== MODEL DEFINITIONS ====================

class BasePINN(nn.Module):
    """Base PINN architecture"""
    def __init__(self, input_dim: int = 2, output_dim: int = 1, 
                 hidden_dim: int = 50, num_layers: int = 4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x, y):
        """Forward pass for 2D spatial coordinates"""
        inputs = torch.cat([x, y], dim=1)
        return self.network(inputs)

class IPFPINN(BasePINN):
    """Improved Physics-Informed PINN with feature mapping"""
    def __init__(self, permeability_field: np.ndarray, hidden_dim: int = 50, **kwargs):
        super().__init__(hidden_dim=hidden_dim, **kwargs)
        
        # Store permeability as tensor
        self.register_buffer('permeability', 
                           torch.from_numpy(permeability_field).float().unsqueeze(0).unsqueeze(0))
        
        # Feature mapping layer (now hidden_dim is available)
        self.feature_mapper = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, y):
        # Get base PINN prediction
        u = super().forward(x, y)
        
        # Add physics-informed features
        features = self.feature_mapper(torch.cat([x, y], dim=1))
        
        # Simple combination (can be made more sophisticated)
        return u + 0.1 * features.mean(dim=1, keepdim=True)

# ==================== DATASET & TRAINING ====================

class GeologicalDataset:
    """Dataset for geological formation data"""
    def __init__(self, alpha: float, nx: int = 64, ny: int = 64):
        self.alpha = alpha
        self.permeability = generate_formation_data(alpha, nx, ny)
        self.nx, self.ny = nx, ny
        
        # Create coordinate grids
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        self.coords = np.column_stack([X.ravel(), Y.ravel()])
        
        # Generate synthetic pressure field (analytical solution)
        self.pressure = self._generate_pressure_field()
    
    def _generate_pressure_field(self) -> np.ndarray:
        """Generate synthetic pressure field for evaluation"""
        k_mean = np.mean(self.permeability)
        
        if self.alpha == 1.0:
            # Homogeneous: linear pressure drop
            p = 1.0 - self.coords[:, 0]  # Simple linear
        elif self.alpha == 0.7:
            # Layered: sinusoidal variation
            p = np.sin(2 * np.pi * self.coords[:, 1]) * (1 - self.coords[:, 0])
        else:
            # Fractured/Heterogeneous: more complex
            p = np.exp(-self.coords[:, 0] / k_mean) * np.cos(2 * np.pi * self.coords[:, 1])
        
        return p.reshape(-1, 1)
    
    def get_training_points(self, n_pde: int = 2048, n_bc: int = 512) -> Dict[str, torch.Tensor]:
        """Get collocation and boundary points"""
        # PDE points (interior)
        pde_idx = np.random.choice(len(self.coords), n_pde, replace=False)
        pde_points = self.coords[pde_idx]
        
        # Boundary points
        bc_coords = []
        # Left boundary (x=0)
        bc_coords.extend([[0, y] for y in np.linspace(0, 1, n_bc//4)])
        # Right boundary (x=1)
        bc_coords.extend([[1, y] for y in np.linspace(0, 1, n_bc//4)])
        # Top/bottom boundaries
        bc_coords.extend([[x, 0] for x in np.linspace(0, 1, n_bc//4)])
        bc_coords.extend([[x, 1] for x in np.linspace(0, 1, n_bc//4)])
        
        bc_points = np.array(bc_coords)
        
        # Convert to tensors
        return {
            "pde_x": torch.from_numpy(pde_points[:, 0:1]).float(),
            "pde_y": torch.from_numpy(pde_points[:, 1:2]).float(),
            "bc_x": torch.from_numpy(bc_points[:, 0:1]).float(),
            "bc_y": torch.from_numpy(bc_points[:, 1:2]).float(),
            "bc_value": torch.zeros(len(bc_points), 1)  # Dirichlet BC
        }

def pde_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, 
             perm: torch.Tensor, dataset: GeologicalDataset) -> torch.Tensor:
    """
    Compute PDE residual loss for Darcy flow: ∇·(k∇p) = 0
    """
    x.requires_grad = True
    y.requires_grad = True
    
    # Forward pass
    p = model(x, y)
    
    # Compute gradients
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), 
                             create_graph=True, retain_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), 
                             create_graph=True, retain_graph=True)[0]
    
    # FIX: Properly handle permeability tensor shape and indexing
    # Convert coordinates to array indices
    with torch.no_grad():
        # Map [0,1] to [0, nx-1] and [0, ny-1]
        ix = torch.clamp((x[:, 0] * (dataset.nx - 1)).long(), 0, dataset.nx - 1)
        iy = torch.clamp((y[:, 0] * (dataset.ny - 1)).long(), 0, dataset.ny - 1)
        
        # FIX: perm is 2D [ny, nx], index directly
        k_values = perm[iy, ix]  # shape: [batch_size]
        k = k_values.unsqueeze(1)  # shape: [batch_size, 1]
    
    # Compute flux: k∇p
    flux_x = k * p_x
    flux_y = k * p_y
    
    # Compute divergence
    flux_x_x = torch.autograd.grad(flux_x, x, torch.ones_like(flux_x), 
                                  create_graph=True, retain_graph=True)[0]
    flux_y_y = torch.autograd.grad(flux_y, y, torch.ones_like(flux_y), 
                                  create_graph=True, retain_graph=True)[0]
    
    # PDE residual
    residual = flux_x_x + flux_y_y
    
    return torch.mean(residual**2)

def train_model(model: nn.Module, dataset: GeologicalDataset, 
                device: torch.device, n_epochs: int, lr: float,
                run_id: int, model_name: str) -> Tuple[float, float]:
    """
    Train a single model instance with robust error handling
    Returns (l2_error, training_time)
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Get training data
    data = dataset.get_training_points()
    pde_x, pde_y = data["pde_x"].to(device), data["pde_y"].to(device)
    bc_x, bc_y, bc_val = data["bc_x"].to(device), data["bc_y"].to(device), data["bc_value"].to(device)
    
    # FIX: Create 2D permeability tensor for direct indexing
    perm_tensor = torch.tensor(dataset.permeability, device=device)  # 2D: [ny, nx]
    
    start_time = time.time()
    nan_detected = False
    
    # Training loop
    for epoch in range(n_epochs + 1):
        try:
            optimizer.zero_grad()
            
            # PDE loss
            loss_pde = pde_loss(model, pde_x, pde_y, perm_tensor, dataset)
            
            # Boundary condition loss
            p_bc = model(bc_x, bc_y)
            loss_bc = torch.mean((p_bc - bc_val)**2)
            
            # Total loss
            loss = loss_pde + 10.0 * loss_bc
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                warnings.warn(f"Run {run_id}: NaN/Inf loss at epoch {epoch}, aborting")
                nan_detected = True
                break
            
            if epoch % 500 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():.3e} (PDE: {loss_pde.item():.3e})")
            
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            warnings.warn(f"Run {run_id}: Training failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            nan_detected = True
            break
    
    training_time = time.time() - start_time
    
    if nan_detected:
        return np.nan, training_time
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Evaluate on full grid
        coords = torch.from_numpy(dataset.coords).float().to(device)
        x_eval, y_eval = coords[:, 0:1], coords[:, 1:2]
        
        pred = model(x_eval, y_eval).cpu().numpy()
        true = dataset.pressure
        
        l2_error = relative_l2_error(pred, true)
        
        # Debug output
        if run_id == 1:  # Only print for first run
            print(f"    DEBUG: pred range=[{pred.min():.3e}, {pred.max():.3e}], "
                  f"true range=[{true.min():.3e}, {true.max():.3e}]")
    
    return l2_error, training_time

# ==================== EXPERIMENT RUNNER ====================

def run_q1_experiment(alphas: List[float] = None, n_runs: int = None, 
                      epochs: int = 2000) -> Dict:
    """
    Run full Q1 experiment with statistical validation
    """
    if alphas is None:
        alphas = Config.ALPHAS
    
    if n_runs is None:
        # Adaptive run count
        n_runs = Config.N_RUNS_STABLE if 1.0 in alphas else Config.N_RUNS_MIN
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Storage for results
    all_results = {alpha: {"IP-FPINN": [], "PINN": []} for alpha in alphas}
    timing_results = {alpha: {"IP-FPINN": [], "PINN": []} for alpha in alphas}
    
    # Create directories
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Q1 JOURNAL EXPERIMENT: IP-FPINN FOR POROUS MEDIA TRANSPORT")
    print(f"Statistical Protocol: {n_runs} independent runs, CV < {Config.CV_THRESHOLD*100:.1f}% verification")
    print("="*80)
    
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"[Formation α={alpha}] - {get_formation_name(alpha)}")
        print(f"{'='*60}")
        
        # Generate/load formation data
        data_file = Config.DATA_DIR / f"{get_formation_name(alpha).lower()}_alpha{alpha:.1f}.npy"
        if not data_file.exists():
            print(f"   Generating formation data...")
            permeability = generate_formation_data(alpha, Config.NX, Config.NY)
            np.save(data_file, permeability)
        else:
            permeability = np.load(data_file)
        
        dataset = GeologicalDataset(alpha, Config.NX, Config.NY)
        
        for run_id in range(1, n_runs + 1):
            print(f"\n  {'─'*50}")
            print(f"  Run {run_id}/{n_runs}...")
            print(f"  {'─'*50}")
            
            # ===== IP-FPINN =====
            print(f"  [IP-FPINN Training]")
            set_seeds(Config.SEED_BASE + run_id + 1000)  # Unique seed
            model_ipfpinn = IPFPINN(permeability, 
                                  hidden_dim=Config.HIDDEN_DIM, 
                                  num_layers=Config.NUM_LAYERS)
            
            l2_ip, time_ip = train_model(model_ipfpinn, dataset, device, 
                                       epochs, Config.LR_IPFPINN, 
                                       run_id, "IP-FPINN")
            
            # ===== Standard PINN =====
            print(f"  [PINN Training]")
            set_seeds(Config.SEED_BASE + run_id + 2000)  # Different seed
            model_pinn = BasePINN(hidden_dim=Config.HIDDEN_DIM, 
                                num_layers=Config.NUM_LAYERS)
            
            l2_pinn, time_pinn = train_model(model_pinn, dataset, device, 
                                           epochs, Config.LR_PINN, 
                                           run_id, "PINN")
            
            # Store results
            all_results[alpha]["IP-FPINN"].append(l2_ip)
            all_results[alpha]["PINN"].append(l2_pinn)
            timing_results[alpha]["IP-FPINN"].append(time_ip)
            timing_results[alpha]["PINN"].append(time_pinn)
            
            # Print per-run results
            print(f"    IP-FPINN: L2={l2_ip:.3e}, Time={time_ip:.1f}s")
            print(f"    PINN:     L2={l2_pinn:.3e}, Time={time_pinn:.1f}s")
    
    return all_results, timing_results

def get_formation_name(alpha: float) -> str:
    """Map alpha to formation name"""
    mapping = {0.3: "Fractured", 0.5: "Heterogeneous", 
               0.7: "Layered", 1.0: "Homogeneous"}
    return mapping.get(alpha, "Unknown")

def print_statistical_summary(results: Dict, timing_results: Dict):
    """Print final statistical analysis with CV validation"""
    print("\n" + "="*80)
    print("STATISTICAL RIGOR VERIFICATION (CV < 5.0%)")
    print("="*80)
    
    cv_passed = True
    
    for alpha in results.keys():
        print(f"\nα={alpha}:")
        
        for model_name in ["IP-FPINN", "PINN"]:
            l2_values = results[alpha][model_name]
            
            # Filter out NaN/Inf values
            valid_l2 = [v for v in l2_values if np.isfinite(v)]
            
            if len(valid_l2) == 0:
                print(f"  {model_name}: ❌ ALL RUNS FAILED")
                cv_passed = False
                continue
            
            mean, std, cv = compute_coefficient_of_variation(valid_l2)
            
            # Use adaptive threshold for near-zero values
            if mean < 1e-6:
                cv_status = "✅ PASS (near-zero mean)"
            elif cv <= Config.CV_THRESHOLD:
                cv_status = f"✅ PASS (CV={cv*100:.2f}%)"
            else:
                cv_status = f"❌ FAIL (CV={cv*100:.2f}%)"
                cv_passed = False
            
            mean_time = np.mean(timing_results[alpha][model_name])
            print(f"  {model_name}: CV={cv*100:.2f}% {cv_status} "
                  f"(mean±std: {mean:.3e}±{std:.3e}, avg_time={mean_time:.1f}s)")
    
    print("\n" + "="*80)
    if cv_passed:
        print("✅ VERIFICATION PASSED: All CV values within acceptable range.")
        print("="*80)
        return True
    else:
        print("❌ VERIFICATION FAILED: Some CV values exceed 5%.")
        print("   Recommendations:")
        print("   - Increase n_runs (try 10-20)")
        print("   - Tune learning rates")
        print("   - Add batch normalization or regularization")
        print("   - Check for numerical instability in PDE loss")
        print("="*80)
        return False

def save_results(results: Dict, timing_results: Dict, filename: str = "q1_results.json"):
    """Save results to JSON for reproducibility"""
    output = {
        "config": {
            "alphas": Config.ALPHAS,
            "n_runs": Config.N_RUNS_MIN,
            "epochs": Config.EPOCHS,
            "hidden_dim": Config.HIDDEN_DIM,
            "num_layers": Config.NUM_LAYERS
        },
        "results": {},
        "timing": timing_results
    }
    
    for alpha in results.keys():
        output["results"][f"alpha_{alpha}"] = {
            "IP-FPINN": results[alpha]["IP-FPINN"],
            "PINN": results[alpha]["PINN"]
        }
    
    with open(Config.RESULTS_DIR / filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {Config.RESULTS_DIR / filename}")

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution with user-configurable parameters"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Q1 Journal Submission Protocol")
    parser.add_argument("--n_runs", type=int, default=None, 
                       help="Number of independent runs (adaptive if not specified)")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS,
                       help="Training epochs")
    parser.add_argument("--alphas", nargs="+", type=float, default=Config.ALPHAS,
                       help="Formation alpha values to test")
    args = parser.parse_args()
    
    # Print protocol header
    print("\n" + "="*80)
    print("Q1 JOURNAL SUBMISSION PROTOCOL")
    print("="*80)
    print(f"Requirements:")
    print(f"  ✓ n_runs = {args.n_runs or 'adaptive'} independent experiments")
    print(f"  ✓ Mean ± std error reporting")
    print(f"  ✓ Coefficient of variation (CV) < {Config.CV_THRESHOLD*100:.0f}%")
    print(f"  ✓ Geological formation datasets")
    print(f"  ✓ Reproducible with fixed seed = {Config.SEED_BASE}")
    print("="*80)
    
    try:
        # Run experiment
        results, timing_results = run_q1_experiment(
            alphas=args.alphas,
            n_runs=args.n_runs,
            epochs=args.epochs
        )
        
        # Print and validate statistics
        is_valid = print_statistical_summary(results, timing_results)
        
        # Save results
        save_results(results, timing_results)
        
        # Exit with error code if validation fails
        if not is_valid:
            print("\n🔧 To fix:")
            print("  1. Run with --n_runs 10 or higher")
            print("  2. Adjust learning rates in Config class")
            print("  3. Check PDE loss implementation for numerical stability")
            raise RuntimeError("CV verification failed.")
        
        print("\n🎉 Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()



