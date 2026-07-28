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




def create_multi_row_zoib_model(obs_data, focus_coords, model_type='exponential'):
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
    print(f"- Severity range: [{severity_data.min():.1f}, {severity_data.max():.1f}]")
    
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
        
        focus1_coords = focus_coords['focus1']
        focus2_coords = focus_coords['focus2']

        "weakly informed priors"
        # Focus 1: Broad Street
        fx1 = pm.Normal('fx1', mu=focus1_coords[0], sigma=2) # Tight sigma
        fy1 = pm.Normal('fy1', mu=focus1_coords[1], sigma=2)
        
        # Focus 2: Little Marlborough Street
        fx2 = pm.Normal('fx2', mu=focus2_coords[0], sigma=2)
        fy2 = pm.Normal('fy2', mu=focus2_coords[1], sigma=2)
        
        # Shared dispersal parameters (same across all rows)
        fz1 = pm.Beta("fz1", alpha=1.0, beta=1.0)  # Base intensity for focus 1
        fz2 = pm.Beta("fz2", alpha=1.0, beta=1.0)  # Base intensity for focus 1
        
        # Store data for later use
        pm.Data("X_data", X_data)
        pm.Data("Y_data", Y_data)
        pm.Data("severity_data", severity_data)
        
        if model_type == 'exponential':
            scale1 = pm.Gamma("scale1", alpha=2, beta=0.04)
            scale2 = pm.Gamma("scale2", alpha=2, beta=0.04)
            
        elif model_type == 'gaussian': 
            scale1 = pm.Gamma("scale1", alpha=2, beta=0.04)
            scale2 = pm.Gamma("scale2", alpha=2, beta=0.04) 
            
        elif model_type == 'power_law':
            scale1 = pm.Gamma("scale1", alpha=2, beta=0.04)
            scale2 = pm.Gamma("scale2", alpha=2, beta=0.04)
            
            exponent1 = pm.Gamma("exponent1", alpha=2, beta=1) # !ensure mean > 2
            exponent2 = pm.Gamma("exponent2", alpha=2, beta=1) # !ensure mean > 2
         
        # ZOIB mixture parameters (shared across rows)
        q_param = pm.Beta("q_param", alpha=1, beta=10)  # Probability of exact 1 given not 0, constant in space
        pi0 = pm.Beta("pi0", alpha=2, beta=2)
        
        # Precision parameter for Beta distribution
        phi = pm.Gamma("phi", alpha=2, beta=0.1)
        weight = pm.Beta("weight", alpha=1, beta=1) # weight = 1 -> All Broad St; weight = 0 -> All Little Marl.
               
        distance1 = pt.sqrt((X - fx1)**2 + (Y - fy1)**2)
        distance2 = pt.sqrt((X - fx2)**2 + (Y - fy2)**2)
        
        "Kernel functions with shared parameters"
        if model_type == 'exponential':
            kernel1 = fz1 * pt.exp(-distance1 / pt.maximum(scale1, 0.01))
            kernel2 = fz2 * pt.exp(-distance2 / pt.maximum(scale2, 0.01))
            
        elif model_type == 'gaussian':
            kernel1 = fz1 * pt.exp(-(distance1**2) / pt.maximum(2 * scale1**2, 0.01))
            kernel2 = fz2 * pt.exp(-(distance2**2) / pt.maximum(2 * scale2**2, 0.01))
            
        elif model_type == 'power_law':
            kernel1 = fz1 / pt.pow(1 + distance1 / pt.maximum(scale1, 0.01), exponent1)
            kernel2 = fz2 / pt.pow(1 + distance2 / pt.maximum(scale2, 0.01), exponent2)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        kernel = weight * kernel1 + (1 - weight) * kernel2
    
        mu = pm.Deterministic("mu", pt.clip(kernel, eps, 1-eps))    
        p_param = pm.Deterministic("p_param", pt.clip(pi0 * (1 - kernel), eps, 1-eps))
        
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
    
    
    diagnostics = {
        'max_rhat': None,
        'n_divergent': None,
        'rhat_converged': False,
        'divergence_acceptable': False,
        'rhat_error': None,
        'divergence_error': None
    }
    
    "R-hat calculation"    
    max_rhat = summary['r_hat'].max()
    
    diagnostics['max_rhat'] = max_rhat
    diagnostics['rhat_converged'] = max_rhat <= 1.1
            
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



def plot_multi_row_results(trace, obs_data, focus_coords, y_pred, y_low, y_high, model_type):
    """Create visualization of the multi-row results"""    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']
    scale2_est = summary.loc['scale2', 'mean']     
    weight_est = summary.loc['weight', 'mean']
    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
        exponent2_est = summary.loc['exponent2', 'mean']
    
    # Create subplots for each row
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    row_x = obs_data['COORD_X'].values
    row_y = obs_data['COORD_Y'].values
    row_sev = obs_data['death_proportion'].values
    
    # Get true focus coordinates for this row
    focus1_coords = focus_coords['focus1']
    focus2_coords = focus_coords['focus2']
            
    # Row-specific intensities
    fz1_est = summary.loc['fz1', 'mean']
    fx1_est = summary.loc['fx1', 'mean']
    fy1_est = summary.loc['fy1', 'mean']
    
    fz2_est = summary.loc['fz2', 'mean']
    fx2_est = summary.loc['fx2', 'mean']
    fy2_est = summary.loc['fy2', 'mean']
    
    
    # Row-specific prediction function
    def predict_row(X, Y):
        distance1 = np.sqrt((X - fx1_est)**2 + (Y - fy1_est)**2)
        distance2 = np.sqrt((X - fx2_est)**2 + (Y - fy2_est)**2)
        
        if model_type == 'exponential':
            kernel1 = fz1_est * np.exp(-distance1 / scale1_est)
            kernel2 = fz2_est * np.exp(-distance2 / scale2_est)

        elif model_type == 'gaussian':
            kernel1 = fz1_est * np.exp(-(distance1**2) / (2 * scale1_est**2))
            kernel2 = fz2_est * np.exp(-(distance2**2) / (2 * scale2_est**2))

        elif model_type == 'power_law':
            kernel1 = fz1_est / np.power(1 + distance1 / scale1_est, exponent1_est)
            kernel2 = fz2_est / np.power(1 + distance2 / scale2_est, exponent2_est)
        
        kernel = weight_est * kernel1 + (1-weight_est)*kernel2
        
        return kernel
    
            
    "calculate R2 from posterior predictive"        
    r2 = r2_score(row_sev, y_pred) # y_pred is from posterior predicitve
    
    # Plot observed vs predicted
    sort_idx = np.argsort(row_y)
    ax.scatter(row_y, row_sev, label='Observed', color='black', s=10, alpha=0.5)
    ax.fill_between(row_y[sort_idx], y_low[sort_idx], y_high[sort_idx], color='skyblue', alpha=0.3, label='95% CI')
    ax.plot(row_y[sort_idx], y_pred[sort_idx], label='Posterior Mean', color='navy', linewidth=2)
    
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Mark foci
    ax.axvline(focus1_coords[1], color='black', linestyle='--', alpha=0.7, 
              label=f'Observed Focus 1 (Y={focus1_coords[1]})')
    ax.axvline(focus2_coords[1], color='black', linestyle='--', alpha=0.7, 
              label=f'Observed Focus 2 (Y={focus2_coords[1]})')
    
    ax.axvline(fy1_est, color='green', linestyle='--', alpha=0.7, 
              label=f'Estimated Focus 1 (Y={fy1_est})')
    ax.axvline(fy2_est, color='green', linestyle='--', alpha=0.7, 
              label=f'Estimated Focus 2 (Y={fy2_est})')
    
    ax.set_ylim(top=1.1)
    ax.set_xlabel('Coord Y')
    ax.set_ylabel('Death Proportion')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    
    "Small 2D Panel (Inset)"
    ax_ins = inset_axes(ax, width="30%", height="35%", loc='upper right', bbox_to_anchor=(0, -0.05, 1, 1), 
                        bbox_transform=ax.transAxes, borderpad=0)
    
    # Scatter plot of the 2D field
    ax_ins.scatter(row_x, row_y, c=row_sev, cmap='YlOrRd', s=2)
    
    # Create grid for contour
    x_min, x_max = row_x.min() - 5, row_x.max() + 5
    y_min, y_max = row_y.min() - 10, row_y.max() + 10
    
    X_grid, Y_grid = np.meshgrid(
        np.linspace(x_min-5, x_max+5, 50),
        np.linspace(y_min-5, y_max+5, 50)
    )
    
    # Create contour plot
    Z_pred = predict_row(X_grid, Y_grid)
    contour = ax_ins.contourf(X_grid, Y_grid, Z_pred, levels=15, cmap='YlOrRd', alpha=0.7)
    
    
    # Mark the Estimated Focus
    ax_ins.scatter(fx1_est, fy1_est, marker='*', color='cyan', s=50, label='Estimated')
    ax_ins.text(fx1_est + 4, fy1_est, 'Estimated', color='darkcyan', 
        fontsize=7, fontweight='bold', va='center')
    
    ax_ins.scatter(fx2_est, fy2_est, marker='*', color='cyan', s=50, label='Estimated')
    ax_ins.text(fx2_est + 4, fy2_est, 'Estimated', color='darkcyan', 
        fontsize=7, fontweight='bold', va='center')
    
    
    ax_ins.scatter(focus1_coords[0], focus1_coords[1], marker='x', color='black', s=40, label='True Focus', zorder=5)
    ax_ins.text(focus1_coords[0]+2, focus1_coords[1]+4, 'True', fontsize=7, color='black', fontweight='bold')
    
    ax_ins.scatter(focus2_coords[0], focus2_coords[1], marker='x', color='black', s=40, label='True Focus', zorder=5)
    ax_ins.text(focus2_coords[0]+2, focus2_coords[1]+4, 'True', fontsize=7, color='black', fontweight='bold')
    

    # Clean up inset axes
    ax_ins.set_title("Spatial View", fontsize=8, pad=3, fontweight='bold')
    ax_ins.set_xticks([]); ax_ins.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_multi_row_results.png', dpi=600, bbox_inches='tight')
    


def plot_3d_rows_subplots(trace, obs_data, y_pred, model_type):
    """Create 2x2 grid of 3D subplots, one for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)

    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']
    scale2_est = summary.loc['scale2', 'mean']     
    weight_est = summary.loc['weight', 'mean']
    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
        exponent2_est = summary.loc['exponent2', 'mean']
    
    # Create figure with 2x2 3D subplots
    fig, ax = plt.subplots(1,1, figsize=(8, 6), subplot_kw={'projection': '3d'})
    
    row_x = obs_data['COORD_X'].values
    row_y = obs_data['COORD_Y'].values
    row_sev = obs_data['death_proportion'].values
    
    # Row-specific intensities
    fz1_est = summary.loc['fz1', 'mean']
    fx1_est = summary.loc['fx1', 'mean']
    fy1_est = summary.loc['fy1', 'mean']
    
    fz2_est = summary.loc['fz2', 'mean']
    fx2_est = summary.loc['fx2', 'mean']
    fy2_est = summary.loc['fy2', 'mean']
    
    # Create grid for this row's area
    row_x_min, row_x_max = row_x.min() - 1, row_x.max() + 1
    row_y_min, row_y_max = row_y.min() - 5, row_y.max() + 5
    
    X_row, Y_row = np.meshgrid(
        np.linspace(row_x_min-5, row_x_max+5, 50),
        np.linspace(row_y_min-5, row_y_max+5, 50)
    )
    
    # Row-specific prediction function
    def predict_row(X, Y):
        distance1 = np.sqrt((X - fx1_est)**2 + (Y - fy1_est)**2)
        distance2 = np.sqrt((X - fx2_est)**2 + (Y - fy2_est)**2)
        
        if model_type == 'exponential':
            kernel1 = fz1_est * np.exp(-distance1 / scale1_est)
            kernel2 = fz2_est * np.exp(-distance2 / scale2_est)

        elif model_type == 'gaussian':
            kernel1 = fz1_est * np.exp(-(distance1**2) / (2 * scale1_est**2))
            kernel2 = fz2_est * np.exp(-(distance2**2) / (2 * scale2_est**2))

        elif model_type == 'power_law':
            kernel1 = fz1_est / np.power(1 + distance1 / scale1_est, exponent1_est)
            kernel2 = fz2_est / np.power(1 + distance2 / scale2_est, exponent2_est)
        
        kernel = weight_est * kernel1 + (1-weight_est)*kernel2
        
        return kernel
    
    # Plot row-specific surface
    Z_row = predict_row(X_row, Y_row)
    surf = ax.plot_surface(X_row, Y_row, Z_row, cmap='viridis', alpha=0.7, linewidth=0)
    
    # Plot observed data points
    scatter = ax.scatter(row_x, row_y, row_sev, 
                       c=row_sev, s=60, cmap='viridis',
                       edgecolors='black', linewidth=0.5)
    
    # Mark foci with stars and vertical lines
    # Calculate predicted severity at focus locations using the model
    focus1_pred = predict_row(fx1_est, fy1_est)   
    
    ax.scatter([fx1_est], [fy1_est], [focus1_pred], 
              marker='*', s=150, color='red', 
              edgecolors='darkred', linewidth=2, label='Focus 1')
    ax.plot([fx1_est, fx1_est], 
           [fy1_est, fy1_est], 
           [0, focus1_pred], 'r-', linewidth=3, alpha=0.8)
    
    
    focus2_pred = predict_row(fx2_est, fy2_est)   
    
    ax.scatter([fx2_est], [fy2_est], [focus2_pred], 
              marker='*', s=150, color='red', 
              edgecolors='darkred', linewidth=2, label='Focus 1')
    ax.plot([fx2_est, fx2_est], 
           [fy2_est, fy2_est], 
           [0, focus2_pred], 'r-', linewidth=3, alpha=0.8)
    
    "calculate R2 from posterior predictive"
    r2_row = r2_score(row_sev, y_pred)
    
    # Set labels and title
    ax.set_xlabel('Coord X', fontsize=10)
    ax.set_ylabel('Coord Y', fontsize=10)
    ax.set_zlabel('Death Proportion', fontsize=10)
    ax.set_title(f'R² = {r2_row:.3f}\nfz={fz1_est:.3f}', fontsize=12)
    
    # Set consistent viewing angle
    ax.view_init(elev=45, azim=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.savefig(output_dir + f'{model_type}_rows_3d_subplots.png', dpi=600, bbox_inches='tight')    


def plot_row_comparison_2d(trace, obs_data, y_pred, model_type):
    """Create 2D comparison plots showing model fit for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']
    scale2_est = summary.loc['scale2', 'mean']     
    weight_est = summary.loc['weight', 'mean']
    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
        exponent2_est = summary.loc['exponent2', 'mean']
    
    # Create figure with contour plots for each row
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    row_x = obs_data['COORD_X'].values
    row_y = obs_data['COORD_Y'].values
    row_sev = obs_data['death_proportion'].values
    
    # Row-specific intensities
    fz1_est = summary.loc['fz1', 'mean']
    fx1_est = summary.loc['fx1', 'mean']
    fy1_est = summary.loc['fy1', 'mean']
    
    fz2_est = summary.loc['fz2', 'mean']
    fx2_est = summary.loc['fx2', 'mean']
    fy2_est = summary.loc['fy2', 'mean']
   
    # Row-specific prediction function
    def predict_row(X, Y):
        distance1 = np.sqrt((X - fx1_est)**2 + (Y - fy1_est)**2)
        distance2 = np.sqrt((X - fx2_est)**2 + (Y - fy2_est)**2)
        
        if model_type == 'exponential':
            kernel1 = fz1_est * np.exp(-distance1 / scale1_est)
            kernel2 = fz2_est * np.exp(-distance2 / scale2_est)

        elif model_type == 'gaussian':
            kernel1 = fz1_est * np.exp(-(distance1**2) / (2 * scale1_est**2))
            kernel2 = fz2_est * np.exp(-(distance2**2) / (2 * scale2_est**2))

        elif model_type == 'power_law':
            kernel1 = fz1_est / np.power(1 + distance1 / scale1_est, exponent1_est)
            kernel2 = fz2_est / np.power(1 + distance2 / scale2_est, exponent2_est)
        
        kernel = weight_est * kernel1 + (1-weight_est)*kernel2
        
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
    
    # Mark foci
    ax.scatter([fx1_est], [fy1_est], 
              marker='*', s=200, color='red', 
              edgecolors='darkred', linewidth=2, zorder=6, label='Focus 1')
    ax.scatter([fx2_est], [fy2_est], 
              marker='*', s=200, color='red', 
              edgecolors='darkred', linewidth=2, zorder=6, label='Focus 1')
    
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
        


def plot_posterior_predictive(y_sim, y_obs, distances, model_type):
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
    ax3.scatter(distances, y_obs, alpha=0.5, label='Observed', s=30, color='red')
    
    # Plot mean and uncertainty
    y_sim_mean = y_sim_flat.mean(axis=0)
    y_sim_lower = np.percentile(y_sim_flat, 2.5, axis=0)
    y_sim_upper = np.percentile(y_sim_flat, 97.5, axis=0)
    
    sort_idx = np.argsort(distances)
    ax3.plot(distances[sort_idx], y_sim_mean[sort_idx], 
             'b-', linewidth=2, label='Predicted mean', alpha=0.8)
    ax3.fill_between(distances[sort_idx], 
                      y_sim_lower[sort_idx], 
                      y_sim_upper[sort_idx],
                      alpha=0.3, color='blue', label='95% PI')
    ax3.set_xlabel('Distance along transect', fontsize=11)
    ax3.set_ylabel('Severity', fontsize=11)
    ax3.set_title(f'Spatial pattern (RMSE = {rmse:.4f})', fontsize=12, fontweight='bold')
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
    
    plt.tight_layout()
    
    # Save
    filename = f'posterior_predictive_check_{model_type}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')



        

if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()
    
    # # HPC environment
    # n_cores = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    n_cores = min(2, os.cpu_count())
    output_dir = "./output/"
    eps = 1e-12
    
    focus_coords = {'focus1':[529396.5394, 181025.063], 'focus2':[529192.5379, 181079.3914]}

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
    models_to_test = ['exponential', 'gaussian', 'power_law']
    traces = {}
    
    for model_type in models_to_test:
        print(f"\n{'='*60}")
        print(f"FITTING {model_type.upper()} MODEL (MULTI-ROW)")
        print(f"{'='*60}")
        
        model = create_multi_row_zoib_model(obs_data, focus_coords, model_type=model_type)            
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
                init='advi+adapt_diag'  # Better initialization
                )
            print("✓ Sampling completed!")
            
            "store trace to dict"                        
            traces[model_type] = trace
            
            "save trace file for later plotting"
            trace.to_netcdf(output_dir + f'{model_type}_trace.nc')
            
            "check convergence"
            diagnostics, summary = convergence_diagnostics(trace)
            print(diagnostics)
            diagnostics.to_csv(output_dir + f'{model_type}_disgnostic.csv')
            summary.to_csv(output_dir + f'{model_type}_trace_summary.csv')
            
                           
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
            y_std = ppc.posterior_predictive['obs'].std(dim=['chain', 'draw']).values # Standard Deviation for each observation
            
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
            
                                
            "create plots if diagnostics worked"                
            print("\nCreating result plots...")
            plot_multi_row_results(trace, obs_data, focus_coords, y_pred, y_low, y_high, model_type)
            print("✓ Plots created successfully")
            
            # Add surface plots with row-specific R²
            plot_surfaces(trace, obs_data, y_pred, model_type)
            print("✓ Surface plots created successfully")
                    
            "calculate effective sample sizes"
            ess_df = calculate_all_ess(trace)
            ess_df['ess_num'] = ess_df['ess'].astype(str).str.extract(r'(\d+\.\d+e[+-]\d+)>')
            ess_df['ess_num'] = pd.to_numeric(ess_df['ess_num'])
            ess_df.to_csv(output_dir + f"{model_type}_effective_sample_size.csv")
    
    "compare models if multiple successful"
    if len(traces) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        loo_comparison = az.compare(traces, ic='loo')
        comp_df = model_performance(traces, obs_data, y_pred)
        comp_df.to_csv(output_dir + "models_performances.csv")
        
        print("\nLOO comparison:")
        print(loo_comparison)
        # Get the best model
        best_model = loo_comparison.index[0]  # First row is best model
        best_loo = loo_comparison.loc[best_model, 'elpd_loo']            
        print(f"\n Best model: {best_model}")
        print(f"   LOO score: {best_loo:.1f}")
        
        # Show differences from best model
        print("\nModel ranking by predictive performance:")
        for i, (model, row) in enumerate(loo_comparison.iterrows()):
            rank = i + 1
            loo_val = row['elpd_loo']
            elpd_diff = row['elpd_diff']
            se = row['dse']
            
            if rank == 1:
                print(f"  {rank}. {model}: LOO = {loo_val:.1f} (Best model)")
            else:
                print(f"  {rank}. {model}: LOO = {loo_val:.1f} (Δ = {elpd_diff:.1f} ± {se:.1f})")
                
                # Interpret the difference
                if elpd_diff > 2 * se:
                    strength = "Strong evidence against"
                elif elpd_diff > se:
                    strength = "Moderate evidence against"
                else:
                    strength = "Weak evidence against"
                print(f"      → {strength} this model")
        
        "additional LOO diagnostics"
        print("\nLOO Diagnostics:")
        for model_name, trace in traces.items():
            loo_result = az.loo(trace, pointwise=True)
            n_high_pareto = np.sum(loo_result.pareto_k > 0.7)
            n_total = len(loo_result.pareto_k)
            
            print(f"  {model_name}: {n_high_pareto}/{n_total} high Pareto-k values")
            if n_high_pareto > 0:
                print(f"    → {n_high_pareto/n_total*100:.1f}% potentially problematic points")

            # show which models completed successfully
            print(f"\nSuccessfully fitted models: {list(traces.keys())}")
            
        "try WAIC if LOO fails"
        print(f"\n{'='*60}")
        print("WAIC COMPARISON (Alternative Information Criterion)")
        print(f"{'='*60}")
        
        waic_comparison = az.compare(traces, ic='waic')
        print("\nWAIC Comparison (lower = better):")
        print(waic_comparison)

