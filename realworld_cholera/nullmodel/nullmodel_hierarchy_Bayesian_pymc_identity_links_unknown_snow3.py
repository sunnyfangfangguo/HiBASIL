# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 17:10:32 2025

@author: sunny
"""

# -*- coding: utf-8 -*-
"""
Multi-Row ZOIB Bayesian Model for Disease Severity Analysis
Fits all rows together with shared dispersal parameters but row-specific foci
"""

import os
cache_dir = "/share/snb2023/fguo7/pytensor_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['PYTENSOR_FLAGS'] = f'base_compiledir={cache_dir}'

import pytensor
import pytensor.tensor as pt
import pymc as pm

import numpy as np
import pandas as pd

import arviz as az
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import warnings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

print(f"✓ PyTensor cache: {pytensor.config.base_compiledir}")


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

"Because the true slope is 0, 0 * kernel will multiply out kernel. The kernel parameters should be the true value. "




def zoib_logpdf(value, p_param, q_param, alphas, betas):
    """ZOIB log-likelihood function"""
    
    eps = 1e-12

    # Clip parameters
    p_param = pt.clip(p_param, eps, 1 - eps)
    q_param = pt.clip(q_param, eps, 1 - eps)
    alphas = pt.clip(alphas, eps, 1000)
    betas = pt.clip(betas, eps, 1000)
    
    # Masks
    is_zero = pt.eq(value, 0.0)
    is_one = pt.eq(value, 1.0)
    
    # Safe masking for Beta evaluation
    safe_value = pt.where(is_zero, 0.5, pt.where(is_one, 0.5, value))
    
    # Components
    logp_zero = pt.log(p_param)
    logp_one = pt.log((1 - p_param) * q_param)
    logp_continuous = pt.log((1 - p_param) * (1 - q_param)) + pm.Beta.logp(safe_value, alphas, betas)
    
    # Clean selection
    result = pt.switch(is_zero, logp_zero, pt.switch(is_one, logp_one, logp_continuous))
     
    return result



def zoib_random_fn(p_param, q_param, alpha_vals, beta_vals, rng=None, size=None):
    """
    Random number generator for ZOIB using categorical component selection.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Ensure inputs are arrays
    p_param = np.asarray(p_param)
    q_param = np.asarray(q_param)
    alpha_vals = np.asarray(alpha_vals)
    beta_vals = np.asarray(beta_vals)
    
    # Get output shape from spatially-varying parameters
    output_shape = alpha_vals.shape
    n_obs = np.prod(output_shape)
       
    # Calculate component probabilities for ZOIB
    prob_zero = p_param
    prob_one = (1 - p_param) * q_param
    prob_beta = (1 - p_param) * (1 - q_param)
    
    # Stack probabilities for categorical choice (shape: [n_obs, 3])
    probs = np.stack([prob_zero.flatten(), 
                      prob_one.flatten(), 
                      prob_beta.flatten()], axis=-1)
    
    # Draw component membership for each observation
    components = np.array([rng.choice(3, p=probs[i]) for i in range(n_obs)])
    components = components.reshape(output_shape)
    
    # Initialize output
    out = np.zeros(output_shape)
    
    # Fill in values based on component
    zero_mask = components == 0
    one_mask = components == 1
    beta_mask = components == 2
    
    out[zero_mask] = 0.0
    out[one_mask] = 1.0
    
    # Generate beta values for continuous component
    if np.any(beta_mask):
        out[beta_mask] = rng.beta(alpha_vals[beta_mask], beta_vals[beta_mask])
    
    return out



def create_multi_row_zoib_model(obs_data):
    """
    Create ZOIB model using all rows with shared dispersal parameters
    """
    eps = 1e-12
    # Prepare data arrays
    X_data = np.array(obs_data['COORD_X'].values, dtype=np.float64)
    Y_data = np.array(obs_data['COORD_Y'].values, dtype=np.float64)
    severity_data = np.array(obs_data['death_proportion'].values, dtype=np.float64)
    
    n_obs = len(severity_data)

    
    print("\nMulti-row model preparation:")
    print(f"- Total observations: {n_obs}")

    # Check for edge cases
    n_zeros = np.sum(severity_data == 0.0)
    n_ones = np.sum(severity_data == 1.0)
    n_continuous = np.sum((severity_data > 0) & (severity_data < 1))
    
    print(f"- Exact zeros: {n_zeros} ({n_zeros/n_obs*100:.1f}%)")
    print(f"- Exact ones: {n_ones} ({n_ones/n_obs*100:.1f}%)")
    print(f"- Continuous: {n_continuous} ({n_continuous/n_obs*100:.1f}%)")
                     
    with pm.Model() as model:
        # Convert to tensor variables
        X = pt.as_tensor_variable(X_data)
        Y = pt.as_tensor_variable(Y_data)

        "weakly informed priors"
        # Shared dispersal parameters (same across all rows)
        fz1 = pm.Beta("fz1", alpha=1.0, beta=1.0)  # Base intensity for focus 1
                        
        # Store data for later use
        pm.Data("X_data", X_data)
        pm.Data("Y_data", Y_data)
        pm.Data("severity_data", severity_data)
                 
        # ZOIB mixture parameters (shared across rows)
        q_param = pm.Beta("q_param", alpha=1, beta=10)  # Probability of exact 1 given not 0, constant in space
        pi0 = pm.Beta("pi0", alpha=2, beta=2)
        
        # Precision parameter for Beta distribution
        phi = pm.Gamma("phi", alpha=2, beta=0.1) # constant in space
                
        mu = pm.Deterministic("mu", pt.clip(pt.full(n_obs, fz1), eps, 1-eps))    
        p_param = pm.Deterministic("p_param", pt.clip(pt.full(n_obs, pi0), eps, 1-eps))
        
        # Beta distribution parameters
        alpha_vals = mu * phi
        beta_vals = (1 - mu) * phi
                
        # Create observation model
        obs = pm.CustomDist(
            "obs", 
            p_param, q_param, alpha_vals, beta_vals,
            logp=zoib_logpdf,
            random=zoib_random_fn,
            observed=severity_data,
            shape=n_obs
        )
    
    return model



def convergence_diagnostics(trace):
    """
    Extract R-hat and divergence statistics from MCMC trace
    """
    diagnostics = {
        'max_rhat': None,
        'n_divergent': None,
        'rhat_converged': False,
        'divergence_acceptable': False,
        'rhat_error': None,
        'divergence_error': None
    }
    
    "R-hat calculation"    
    r_hat = az.rhat(trace)
    
    if hasattr(r_hat, 'to_array'):
        r_hat_array = r_hat.to_array()
        max_r_hat = float(r_hat_array.max().values)
    elif hasattr(r_hat, 'max'):
        max_val = r_hat.max()
        max_r_hat = float(max_val.values if hasattr(max_val, 'values') else max_val)
    else:
        max_values = []
        for var_name in r_hat.data_vars:
            var_data = r_hat[var_name]
            if hasattr(var_data, 'values'):
                max_values.append(float(var_data.values.max()))
            else:
                max_values.append(float(var_data.max()))
        max_r_hat = max(max_values)
    
    diagnostics['max_rhat'] = max_r_hat
    diagnostics['rhat_converged'] = max_r_hat <= 1.1
            
    "divergence calculation"    
    diverging = trace.sample_stats.diverging
    if hasattr(diverging, 'sum'):
        if hasattr(diverging.sum(), 'values'):
            n_divergent = int(diverging.sum().values)
        else:
            n_divergent = int(diverging.sum())
    else:
        n_divergent = int(diverging)
    
    diagnostics['n_divergent'] = n_divergent
    diagnostics['divergence_acceptable'] = n_divergent < 10
    
    diagnostics_df = pd.DataFrame(data=diagnostics, index=[0])
    
    
    "Basic summary"
    summary = az.summary(trace, hdi_prob=0.95)
    print("✓ Parameter summary computed")
    print(f"Parameters estimated: {len(summary)}")

    # Print parameter estimates
    print("\nShared dispersal parameters (mean ± std):")
    shared_params = ['fz1_base', 'scale1', 'p', 'q', 'phi']
    for param in shared_params:
        if param in summary.index:
            mean_val = summary.loc[param, 'mean']
            std_val = summary.loc[param, 'sd']
            print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Print row effects
    print("\nRow-specific effects:")
    row_params = [p for p in summary.index if 'row_effect' in p or 'sigma_row' in p]
    for param in row_params:
        mean_val = summary.loc[param, 'mean']
        std_val = summary.loc[param, 'sd']
        print(f"  {param}: {mean_val:.4f} ± {std_val:.4f}")
                        
    return diagnostics_df, summary



def calculate_all_ess(trace):
    """Calculate ESS for all parameters and return as DataFrame"""
    
    param_names = list(trace.posterior.data_vars.keys())    
    results = []    
    for param in param_names:
        param_data = trace.posterior[param]
        
        if param_data.ndim == 2:  # Scalar parameter (chains, draws)            
            ess_value = az.ess(param_data, method='tail').values
            results.append({
                'parameter': param,
                'element': None,
                'ess': ess_value
            })

                
        elif param_data.ndim == 3:  # Array parameter (chains, draws, dim)
            dim_size = param_data.shape[2]
            for i in range(dim_size):                
                param_element = param_data.isel({f'{param}_dim_0': i})
                ess_value = az.ess(param_element, method='tail').values
                results.append({
                    'parameter': param,
                    'element': i,
                    'ess': ess_value
                })
    
    return pd.DataFrame(results)



def plot_multi_row_results(trace, obs_data, y_pred, y_low, y_high, model_type):
    """Create visualization of the multi-row results"""    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Get data for this row
    row_x = obs_data['COORD_X'].values
    row_y = obs_data['COORD_Y'].values
    row_sev = obs_data['death_proportion'].values
    
    # Row-specific intensities
    fz1_est = summary.loc['fz1', 'mean']

    "calculate R2 from posterior predictive"
    r2 = r2_score(row_sev, y_pred) # y_pred is from posterior predicitve
    # Plot observed vs predicted
    sort_idx = np.argsort(row_y)
    ax.scatter(row_y, row_sev, label='Observed', color='black', s=10, alpha=0.5)
    ax.fill_between(row_y[sort_idx], y_low[sort_idx], y_high[sort_idx], color='skyblue', alpha=0.3, label='95% CI')
    ax.plot(row_y[sort_idx], y_pred[sort_idx], label='Posterior Mean', color='navy', linewidth=2)
    
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_ylim(top=1.1)
    ax.set_xlabel('Distance (y)')
    ax.set_ylabel('Deaths Proportion')
    ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.3), fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_multi_row_results.png', dpi=600, bbox_inches='tight')
    


def plot_3d_rows_subplots(trace, obs_data, y_pred, model_type):
    """Create 2x2 grid of 3D subplots, one for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Create figure with 2x2 3D subplots
    fig, ax = plt.subplots(1,1, figsize=(8, 6), subplot_kw={'projection': '3d'})
    
    # Get row-specific data
    row_x = obs_data['COORD_X'].values
    row_y = obs_data['COORD_Y'].values
    row_sev = obs_data['death_proportion'].values

    # Row-specific intensities
    fz1_est = summary.loc['fz1', 'mean']
    
    # Create grid for this row's area
    row_x_min, row_x_max = row_x.min() - 1, row_x.max() + 1
    row_y_min, row_y_max = row_y.min() - 5, row_y.max() + 5
    
    X_row, Y_row = np.meshgrid(
        np.linspace(row_x_min-5, row_x_max+5, 50),
        np.linspace(row_y_min-5, row_y_max+5, 50)
    )
    
    def predict_row(X, Y):
        kernel = np.full_like(X, fz1_est, dtype=float)
        
        return kernel
    
    # Plot row-specific surface
    Z_row = predict_row(X_row, Y_row)
    surf = ax.plot_surface(X_row, Y_row, Z_row, 
                         cmap='viridis', alpha=0.7, linewidth=0)
    
    # Plot observed data points
    scatter = ax.scatter(row_x, row_y, row_sev, 
                       c=row_sev, s=60, cmap='viridis',
                       edgecolors='black', linewidth=0.5)
        
    "calculate R2 from posterior predictive"
    r2 = r2_score(row_sev, y_pred)
    
    # Set labels and title
    ax.set_xlabel('Coord X', fontsize=10)
    ax.set_ylabel('Coord Y', fontsize=10)
    ax.set_zlabel('Death Proportion', fontsize=10)
    ax.set_title(f'R² = {r2:.3f} (fz1={fz1_est:.3f}')
    
    # Set consistent viewing angle
    ax.view_init(elev=45, azim=45)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.savefig(output_dir + f'{model_type}_rows_3d_subplots.png', dpi=600, bbox_inches='tight')    


def plot_row_comparison_2d(trace, obs_data, y_pred, model_type):
    """Create 2D comparison plots showing model fit for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Create figure with contour plots for each row
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Get row-specific data
    row_x = obs_data['COORD_X'].values
    row_y = obs_data['COORD_Y'].values
    row_sev = obs_data['death_proportion'].values
    
    # Row-specific intensities
    fz1_est = summary.loc['fz1', 'mean']
   
    def predict_row(X, Y):
        kernel = np.full_like(X, fz1_est, dtype=float)
        
        return kernel
    
    "calculate R² for this row from posterior predictive"
    r2 = r2_score(row_sev, y_pred)

    # Create grid for contour
    x_min, x_max = row_x.min() - 2, row_x.max() + 2
    y_min, y_max = row_y.min() - 10, row_y.max() + 10
    
    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min-5, x_max+5, 50),
        np.linspace(y_min-5, y_max+5, 50)
    )
    
    # Create contour plot
    Z_pred = predict_row(X_grid, Y_grid)
    contour = ax.contourf(X_grid, Y_grid, Z_pred, levels=15, cmap='viridis', alpha=0.7)
    
    # Plot observed data
    scatter = ax.scatter(row_x, row_y, c=row_sev, s=80, cmap='viridis', 
                       edgecolors='black', linewidth=0.8, zorder=5)
    
    ax.set_xlabel('Coord X')
    ax.set_ylabel('Coord Y')
    ax.set_title(f'R² = {r2:.3f} (fz1={fz1_est:.3f}')
    ax.legend()
    
    # Add colorbar for this subplot
    plt.colorbar(contour, ax=ax, label='Predicted Proportion')
    
    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_row_comparison_2d.png', dpi=600, bbox_inches='tight')
    

        

def plot_surfaces(trace, obs_data, y_pred, model_type):
    """Create all surface plots - focused on row-specific analysis"""
    print(f"\nCreating surface plots for {model_type} model...")
    
    # Create 2D row comparison plots (returns R² values)
    print("Creating 2D row comparison plots...")
    plot_row_comparison_2d(trace, obs_data, y_pred, model_type)
    
    # Create comprehensive 3D plot of all rows
    print("Creating 3D plot of all rows...")
    plot_3d_rows_subplots(trace, obs_data, y_pred, model_type)



def model_performance(traces, obs_data, y_pred):
    """
    model selection based on loo, waic
    """    
    "compare with loo"
    loo_comparison = az.compare(traces,ic='loo')
    
    comparisons = {}
    for model_name, trace in traces.items():
        # effective number of parameters
        waic_result = az.waic(trace)
        p_waic = float(waic_result.p_waic)
        n_obs = int(waic_result.n_data_points)
                       
        row = loo_comparison.loc[model_name]
        elpd_loo = row['elpd_loo'] # Like 'log_likelihood' but better
        p_loo = row['p_loo'] # Effective parameters 
        rank_loo = row['rank']  # 0=best, 1=second, etc.
        se_d_loo = row['dse'] # Standard error of difference
        warning = row['warning'] # Any computational warnings
        
        model_loo = az.loo(traces[model_name])
        pareto_k_bad = int(np.sum(model_loo.pareto_k > 0.7)) # Problematic observations
                

                
        comparisons[model_name] = {'n_obs':n_obs, 
                               'elpd_loo': elpd_loo, 
                               'p_loo': p_loo, 'rank_loo':rank_loo,
                               'se_d_loo':se_d_loo, 'warning':warning, 
                               'pareto_k_bad':pareto_k_bad}
        
    comparisons_df = pd.DataFrame(comparisons)
    
    return comparisons_df
        



def plot_posterior_predictive(y_sim, y_obs, coord_ys, model_type):
    """
    Plot posterior predictive checks for ZOIB model.
    
    Parameters:
    -----------
    y_sim : array (chains, draws, n_obs)
        Posterior predictive samples
    y_obs : array (n_obs,)
        Observed severity values
    distances : array (n_obs,)
        Distance from focus for spatial plot
    model_type : str
        Model name for title
    sim : int
        Simulation ID for filename
    """
    
    # Flatten simulations
    y_sim_flat = y_sim.reshape(-1, len(y_obs))  # (n_samples, n_obs)
    
    # Calculate overall metrics
    y_pred = y_sim_flat.mean(axis=0)
    
    r2 = r2_score(y_obs, y_pred)
    rmse = np.sqrt(mean_squared_error(y_obs, y_pred))
    
    # Create PPC plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Overall distribution
    ax1 = axes[0, 0]
    ax1.hist(y_obs, bins=50, alpha=0.5, label='Observed', density=True, color='red')
    for i in range(min(100, y_sim_flat.shape[0])):
        ax1.hist(y_sim_flat[i], bins=50, alpha=0.01, color='blue', density=True)
    ax1.set_xlabel('Severity', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'Overall Distribution (R² = {r2:.3f})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # 2. Zero/One inflation
    ax2 = axes[0, 1]
    prop_zeros_obs = np.mean(y_obs == 0)
    prop_ones_obs = np.mean(y_obs == 1)
    prop_zeros_sim = np.mean(y_sim_flat == 0, axis=1)
    prop_ones_sim = np.mean(y_sim_flat == 1, axis=1)
    
    ax2.axvline(prop_zeros_obs, color='red', linewidth=2, 
                label=f'Obs. zeros ({prop_zeros_obs:.1%})', linestyle='--')
    ax2.hist(prop_zeros_sim, bins=30, alpha=0.5, color='red', 
             label=f'Sim. zeros (μ={prop_zeros_sim.mean():.1%})')
    ax2.axvline(prop_ones_obs, color='blue', linewidth=2, 
                label=f'Obs. ones ({prop_ones_obs:.1%})', linestyle='--')
    ax2.hist(prop_ones_sim, bins=30, alpha=0.5, color='blue',
             label=f'Sim. ones (μ={prop_ones_sim.mean():.1%})')
    ax2.set_xlabel('Proportion', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Zero/One Inflation Check', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Spatial gradient
    ax3 = axes[1, 0]
    ax3.scatter(coord_ys, y_obs, alpha=0.5, label='Observed', s=30, color='red')
    
    # Plot mean and uncertainty
    y_sim_mean = y_sim_flat.mean(axis=0)
    y_sim_lower = np.percentile(y_sim_flat, 2.5, axis=0)
    y_sim_upper = np.percentile(y_sim_flat, 97.5, axis=0)
    
    sort_idx = np.argsort(coord_ys)
    ax3.plot(coord_ys[sort_idx], y_sim_mean[sort_idx], 
             'b-', linewidth=2, label='Predicted mean', alpha=0.8)
    ax3.fill_between(coord_ys[sort_idx], 
                      y_sim_lower[sort_idx], 
                      y_sim_upper[sort_idx],
                      alpha=0.3, color='blue', label='95% PI')
    ax3.set_xlabel('Coord Y', fontsize=11)
    ax3.set_ylabel('deaths_proportion', fontsize=11)
    ax3.set_title(f'Spatial Gradient (RMSE = {rmse:.4f})', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    
    # 4. Continuous values only
    ax4 = axes[1, 1]
    mask_cont = (y_obs > 0) & (y_obs < 1)
    if mask_cont.sum() > 5:
        y_obs_cont = y_obs[mask_cont]
        y_sim_cont = y_sim_flat[:, mask_cont]
        
        ax4.hist(y_obs_cont, bins=30, alpha=0.5, label='Observed', 
                 density=True, color='red')
        for i in range(min(100, y_sim_cont.shape[0])):
            ax4.hist(y_sim_cont[i], bins=30, alpha=0.01, color='blue', density=True)
        
        # R² for continuous only
        y_pred_cont = y_sim_cont.mean(axis=0)
        r2_cont = r2_score(y_obs_cont, y_pred_cont)
        
        ax4.set_xlabel('Severity (continuous only)', fontsize=11)
        ax4.set_ylabel('Density', fontsize=11)
        ax4.set_title(f'Continuous Values (R² = {r2_cont:.3f})', 
                     fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.3)
    
    plt.suptitle(f'{model_type.upper()} Model - Posterior Predictive Check',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    filename = f'posterior_predictive_check_{model_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')



       

if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()
    
    # # HPC environment
    # n_cores = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    n_cores = min(2, os.cpu_count())

    "simulate parameters"
    output_dir = "./output/"   
    eps = 1e-12

    """Main execution function for multi-row analysis"""        
    print("Loading multi-row disease severity data...")
    
    output_file = output_dir + 'ppc_metrics_all_sims.csv'
    pd.DataFrame(columns=['model_type', 'overall_r2', 'overall_rmse', 'overall_mae',
                          'obs_zeros', 'pred_zeros', 'obs_ones', 'pred_ones', 'continuous_r2', 
                          'continuous_rmse']).to_csv(output_file, index=False) # store all ppc metrics
    
    obs_data = pd.read_csv("deaths_proportion.csv")
    print("\nDataset summary:")
    print(f"Total observations: {len(obs_data)}")
    
    # Test different models
    model_type = "null_model"
    print(f"\n{'='*60}")
    print(f"FITTING {model_type.upper()} MODEL (MULTI-ROW)")
    print(f"{'='*60}")
                
    model = create_multi_row_zoib_model(obs_data)            
    with model:
        initial_point = model.initial_point()
        print("✓ Initial point computed successfully")
        
        logp_fn = model.compile_logp()
        initial_logp = logp_fn(initial_point)
        print(f"✓ Initial log-probability: {initial_logp:.4f}")
        
        if not np.isfinite(initial_logp):
            print("⚠ WARNING: Initial log-probability is not finite")
            
        "sample from posterior"
        print("\n--- Starting MCMC sampling ---")                    
        trace = pm.sample(
            draws=8000,
            tune=4000,
            chains=4,
            cores=4,
            target_accept=0.85,
            return_inferencedata=True,
            random_seed=42,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
            init='jitter+adapt_diag'  # Better initialization
            )
        print("✓ Sampling completed!")
        
        "save trace file for later plotting"
        trace.to_netcdf(output_dir + f'{model_type}_trace.nc')
        
        "posterior predictive check"
        # generate posterior predictive samples
        ppc = pm.sample_posterior_predictive(trace, var_names=['obs'], random_seed=42)
        
        # extract generated data
        y_sim = ppc.posterior_predictive['obs'].values                
        y_obs = obs_data['death_proportion'].values
        coord_xs = obs_data['COORD_X'].values
        coord_ys = obs_data['COORD_Y'].values            
        
        # plot posterior predictive check
        plot_posterior_predictive(y_sim, y_obs, coord_ys, model_type)                
        y_pred = ppc.posterior_predictive['obs'].mean(dim=['chain', 'draw']).values
        y_low = ppc.posterior_predictive['obs'].quantile(0.025, dim=['chain', 'draw']).values
        y_high = ppc.posterior_predictive['obs'].quantile(0.975, dim=['chain', 'draw']).values           
        y_std = ppc.posterior_predictive['obs'].std(dim=['chain', 'draw']).values# Standard Deviation for each observation
        
        posterior_df = pd.DataFrame({'coord_x':coord_xs, 'coord_y':coord_ys, 'y_obs':y_obs, 'y_pred':y_pred,
                                     'y_low':y_low, 'y_high':y_high, 'y_std':y_std})
        
        posterior_df.to_csv(f"posterior_predictive_{model_type}.csv")
    
        "calculate ppc metrics"
        mask_cont = (y_obs > 0) & (y_obs < 1)
        y_obs_cont = y_obs[mask_cont]
        y_pred_cont = y_pred[mask_cont]
        
        metrics = {
                'model_type': model_type,
                'overall_r2': r2_score(y_obs, y_pred),
                'overall_rmse': np.sqrt(mean_squared_error(y_obs, y_pred)),
                'overall_mae': mean_absolute_error(y_obs, y_pred),
                
                # Zero/One inflation
                'obs_zeros': np.mean(y_obs == 0),
                'pred_zeros': np.mean(y_sim == 0),
                'obs_ones': np.mean(y_obs == 1),
                'pred_ones': np.mean(y_sim == 1),
                
                # Continuous values only
                'continuous_r2': r2_score(y_obs_cont, y_pred_cont) if mask_cont.sum() > 5 else np.nan,
                'continuous_rmse': np.sqrt(mean_squared_error(y_obs_cont, y_pred_cont)) if mask_cont.sum() > 5 else np.nan,
            }
            
        pd.DataFrame([metrics]).to_csv(output_file, mode='a', header=False, index=False)
        
        "check convergence"
        diagnostics, summary = convergence_diagnostics(trace)
        print(diagnostics)
        diagnostics.to_csv(output_dir + f'{model_type}_disgnostic.csv')
        summary.to_csv(output_dir + f'{model_type}_trace_summary.csv')
                            
        "create plots if diagnostics worked"                
        print("\nCreating result plots...")
        plot_multi_row_results(trace, obs_data, y_pred, y_low, y_high, model_type)
        print("✓ Plots created successfully")
        
        # Add surface plots with row-specific R²
        plot_surfaces(trace, obs_data, y_pred, model_type)
        print("✓ Surface plots created successfully")
                
        "calculate effective sample sizes"
        ess_df = calculate_all_ess(trace)
        ess_df['ess_num'] = ess_df['ess'].astype(str).str.extract(r'(\d+\.\d+e[+-]\d+)>')
        ess_df['ess_num'] = pd.to_numeric(ess_df['ess_num'])
        ess_df.to_csv(output_dir + f"{model_type}_effective_sample_size.csv")



                    
        
