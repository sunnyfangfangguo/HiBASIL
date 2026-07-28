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
import simulation

print(f"✓ PyTensor cache: {pytensor.config.base_compiledir}")


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

"Because the true slope is 0, 0 * kernel will multiply out kernel. The kernel parameters should be the true value. "

def load_and_process_multi_row_data(df, fz1_value=0.3):
    """Load and process the complete multi-row dataset"""
    # Separate foci from observations
    foci_data = df[df['Plant'] == 'focus'].copy()
    obs_data = df[df['Plant'] != 'focus'].copy() 
    
    print(f"Foci locations: {len(foci_data)} points")
    print(f"Observations: {len(obs_data)} points")
    print(f"Unique rows: {sorted(obs_data['Row'].unique())}")
    
    # Extract focus coordinates for each row
    focus_coords = {}
    for row_id in obs_data['Row'].unique():
        row_foci = foci_data[foci_data['Row'] == row_id]
        if len(row_foci) == 1:
            # Sort by Distance to get consistent focus1 (Y=0) and focus2 (Y=~100)
            row_foci_sorted = row_foci.sort_values('Distance')
            focus1 = row_foci_sorted.iloc[0]
            
            focus_coords[row_id] = {
                'focus1': (focus1['InterrowDistance'], focus1['Distance'], fz1_value)
            }
            
            print(f"Row {row_id}: Focus1=({focus1['InterrowDistance']}, {focus1['Distance']}, fz1_value)")
    
    return obs_data, focus_coords


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



def create_multi_row_zoib_model(obs_data, model_type='exponential'):
    """
    Create ZOIB model using all rows with shared dispersal parameters
    """
    eps = 1e-12
    # Prepare data arrays
    X_data = np.array(obs_data['InterrowDistance'].values, dtype=np.float64)
    Y_data = np.array(obs_data['Distance'].values, dtype=np.float64)
    severity_data = np.array(obs_data['Severity'].values, dtype=np.float64)
    row_ids = np.array(obs_data['Row'].values, dtype=int)
    
    n_obs = len(severity_data)
    unique_rows = sorted(obs_data['Row'].unique())
    n_rows = len(unique_rows)
    row_to_idx = {row: idx for idx, row in enumerate(unique_rows)}
    row_indices = np.array([row_to_idx[row] for row in row_ids])
    
    print("\nMulti-row model preparation:")
    print(f"- Total observations: {n_obs}")
    print(f"- Number of rows: {n_rows}")
    print(f"- Rows: {unique_rows}")
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
        row_idx = pt.as_tensor_variable(row_indices)
        
        "weakly informed priors"
        # weakly informed priors for focus coordinates based on observed data
        fx1 = pm.Normal("fx1", mu=5.0, sigma=5.0, shape=n_rows)
        fy1 = pm.Normal("fy1", mu=25.0, sigma=20.0, shape=n_rows)
        
        # Shared dispersal parameters (same across all rows)
        fz1_base = pm.Beta("fz1_base", alpha=1.0, beta=1.0)  # Base intensity for focus 1
        
        # Row-specific random effects on intensity
        sigma_row = pm.HalfNormal("sigma_row", sigma=0.15)
        fz1_row_effect = pm.Normal("fz1_row_effect", mu=0, sigma=sigma_row, shape=n_rows)
        
        # Row-specific intensities
        fz1 = pm.Deterministic("fz1", pt.clip(fz1_base + fz1_row_effect, 0.001, 0.99))
                
        # Store data for later use
        pm.Data("X_data", X_data)
        pm.Data("Y_data", Y_data)
        pm.Data("severity_data", severity_data)
        pm.Data("row_indices", row_indices)
        
        if model_type == 'exponential':
            scale1 = pm.Gamma("scale1", alpha=2, beta=0.4)  
            
        elif model_type == 'gaussian': 
            scale1 = pm.Gamma("scale1", alpha=2, beta=0.4) 
            
        elif model_type == 'power_law':
            scale1 = pm.Gamma("scale1", alpha=2, beta=0.4) 
            exponent1 = pm.Gamma("exponent1", alpha=3.6, beta=2.0) # !ensure mean > 2
         
        # ZOIB mixture parameters (shared across rows)
        q_param = pm.Beta("q_param", alpha=1, beta=10)  # Probability of exact 1 given not 0, constant in space
        pi0 = pm.Beta("pi0", alpha=2, beta=2)
        
        # Precision parameter for Beta distribution
        phi = pm.Gamma("phi", alpha=2, beta=0.1) # constant in space
        
        # Calculate distances from each observation to its row's foci
        # Use advanced indexing to get row-specific focus coordinates
        fx1_row = fx1[row_idx]  # Focus 1 X for each row
        fy1_row = fy1[row_idx]  # Focus 1 Y for each row
        fz1_row = fz1[row_idx]  # Focus 1 intensity for each row
        
        distance = pt.sqrt((X - fx1_row)**2 + (Y - fy1_row)**2)
        
        "Kernel functions with shared parameters"
        if model_type == 'exponential':
            kernel = fz1_row * pt.exp(-distance / pt.maximum(scale1, 0.01))
            
        elif model_type == 'gaussian':
            kernel = fz1_row * pt.exp(-(distance**2) / pt.maximum(2 * scale1**2, 0.01))
            
        elif model_type == 'power_law':
            kernel = fz1_row / pt.pow(1 + distance / pt.maximum(scale1, 0.01), exponent1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
                
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



def plot_multi_row_results(trace, obs_data, focus_coords, y_pred, model_type, sim):
    """Create visualization of the multi-row results"""    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']

    # Get row effects
    unique_rows = sorted(obs_data['Row'].unique())
    
    # Create subplots for each row
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, row in enumerate(unique_rows):
        ax = axes[idx]
        
        # Get data for this row
        row_data = obs_data[obs_data['Row'] == row]
        # row_data = row_data.sort_values(by=['Distance', 'InterrowDistance'])
        row_x = row_data['InterrowDistance'].values
        row_y = row_data['Distance'].values
        row_sev = row_data['Severity'].values
        
        # Get focus coordinates for this row
        focus1_coords = focus_coords[row]['focus1']
        
        # Row-specific intensities
        fz1_row_est = summary.loc[f'fz1[{idx}]', 'mean']
        fx1_row_est = summary.loc[f'fx1[{idx}]', 'mean']
        fy1_row_est = summary.loc[f'fy1[{idx}]', 'mean']
        
        # Create prediction function for this row
        def predict_row(x, y):
            distance = np.sqrt((x - fx1_row_est)**2 + (y - fy1_row_est)**2)
            
            if model_type == 'exponential':
                kernel = fz1_row_est * np.exp(-distance / scale1_est)
            
            elif model_type == 'gaussian':
                kernel = fz1_row_est * np.exp(-(distance**2) / (2 * scale1_est**2))
                
            elif model_type == 'power_law':
                kernel = fz1_row_est / np.power(1 + distance / scale1_est, exponent1_est)
                            
            return kernel
        
        # predicted_row_severity = predict_row(row_x, row_y)
        predicted_severity = predict_row(row_x, row_y)
        
        "calculate R2 from posterior predictive"
        row_mask = obs_data['Row'] == row
        y_pred_row = y_pred[row_mask.values]
        r2 = r2_score(row_sev, y_pred_row) # y_pred is from posterior predicitve
        
        # Plot observed vs predicted
        ax.scatter(row_y, row_sev, label='Observed', color='red', s=60, alpha=0.7)
        ax.plot(row_y, predicted_severity, label='Predicted', color='blue', linewidth=2)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Mark foci
        ax.axvline(focus1_coords[1], color='black', linestyle='--', alpha=0.7, 
                  label=f'Observed Focus (Y={focus1_coords[1]})')
        ax.axvline(fy1_row_est, color='green', linestyle='--', alpha=0.7, 
                  label=f'Estimated Focus (Y={fy1_row_est})')
        
        ax.set_xlabel('Distance along transect')
        ax.set_ylabel('Disease Severity')
        ax.set_title(f'Row {row}: X={focus1_coords[0]:.1f} (fz1={fz1_row_est:.3f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_type.title()} Model', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_multi_row_results_sim{sim}.png', dpi=300, bbox_inches='tight')
    


def plot_3d_rows_subplots(trace, obs_data, y_pred, model_type, sim):
    """Create 2x2 grid of 3D subplots, one for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
    
    # Get row effects
    unique_rows = sorted(obs_data['Row'].unique())

    # Create figure with 2x2 3D subplots
    fig = plt.figure(figsize=(16, 12))
    
    for idx, row in enumerate(unique_rows):
        # Create 3D subplot
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        
        # Get row-specific data
        row_data = obs_data[obs_data['Row'] == row]
        row_x = row_data['InterrowDistance'].values
        row_y = row_data['Distance'].values
        row_sev = row_data['Severity'].values

        # Row-specific intensities
        fz1_row_est = summary.loc[f'fz1[{idx}]', 'mean']
        fx1_row_est = summary.loc[f'fx1[{idx}]', 'mean']
        fy1_row_est = summary.loc[f'fy1[{idx}]', 'mean']
        
        # Create grid for this row's area
        row_x_min, row_x_max = row_x.min() - 1, row_x.max() + 1
        row_y_min, row_y_max = row_y.min() - 5, row_y.max() + 5
        
        X_row, Y_row = np.meshgrid(
            np.linspace(-15, 15, 50),
            np.linspace(row_y_min-5, row_y_max+5, 50)
        )
        
        # Row-specific prediction function
        def predict_row(X, Y):
            distance = np.sqrt((X - fx1_row_est)**2 + (Y - fy1_row_est)**2)
            
            if model_type == 'exponential':
                kernel = fz1_row_est * np.exp(-distance / scale1_est)

            elif model_type == 'gaussian':
                kernel = fz1_row_est * np.exp(-(distance**2) / (2 * scale1_est**2))

            elif model_type == 'power_law':
                kernel = fz1_row_est / np.power(1 + distance / scale1_est, exponent1_est)
            
            return kernel
        
        # Plot row-specific surface
        Z_row = predict_row(X_row, Y_row)
        surf = ax.plot_surface(X_row, Y_row, Z_row, 
                             cmap='viridis', alpha=0.7, linewidth=0)
        
        # Plot observed data points
        scatter = ax.scatter(row_x, row_y, row_sev, 
                           c=row_sev, s=60, cmap='viridis',
                           edgecolors='black', linewidth=0.5)
        
        # Mark foci with stars and vertical lines
        # Calculate predicted severity at focus locations using the model
        focus1_pred = predict_row(fx1_row_est, fy1_row_est)   
        
        ax.scatter([fx1_row_est], [fy1_row_est], [focus1_pred], 
                  marker='*', s=150, color='red', 
                  edgecolors='darkred', linewidth=2, label='Estimated Focus')
        ax.plot([fx1_row_est, fx1_row_est], 
               [fy1_row_est, fy1_row_est], 
               [0, focus1_pred], 'r-', linewidth=3, alpha=0.8)
        
        "calculate R2 from posterior predictive"
        row_mask = obs_data['Row'] == row
        y_pred_row = y_pred[row_mask.values]
        r2_row = r2_score(row_sev, y_pred_row)
        
        # Set labels and title
        ax.set_xlabel('InterrowDistance', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)
        ax.set_zlabel('Severity', fontsize=10)
        ax.set_title(f'Row {row}: R² = {r2_row:.3f} (fz1={fz1_row_est:.3f}')
        
        # Set consistent viewing angle
        ax.view_init(elev=45, azim=45)
        
        # Only show legend on first subplot to avoid clutter
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Add overall title
    fig.suptitle(f'{model_type.title()} Model', 
                 fontsize=16, y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for suptitle
    plt.savefig(output_dir + f'{model_type}_rows_3d_subplots_sim{sim}.png', dpi=300, bbox_inches='tight')    
    print("✓ 3D subplots (2x2 grid) created successfully")


def plot_row_comparison_2d(trace, obs_data, y_pred, model_type, sim):
    """Create 2D comparison plots showing model fit for each row"""
    
    # Extract parameter estimates
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Get shared parameters
    scale1_est = summary.loc['scale1', 'mean']    
    if model_type == "power_law":
        exponent1_est = summary.loc['exponent1', 'mean']
    
    # Get row effects
    unique_rows = sorted(obs_data['Row'].unique())
    
    # Create figure with contour plots for each row
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Store R² values for summary
    row_r2_values = {}    
    for idx, row in enumerate(unique_rows):
        ax = axes[idx]
        
        # Get row-specific data
        row_data = obs_data[obs_data['Row'] == row]
        row_x = row_data['InterrowDistance'].values
        row_y = row_data['Distance'].values
        row_sev = row_data['Severity'].values
        
        # Row-specific intensities
        fz1_row_est = summary.loc[f'fz1[{idx}]', 'mean']
        fx1_row_est = summary.loc[f'fx1[{idx}]', 'mean']
        fy1_row_est = summary.loc[f'fy1[{idx}]', 'mean']
       
        # Row-specific prediction function
        def predict_row(X, Y):
            distance = np.sqrt((X - fx1_row_est)**2 + (Y - fy1_row_est)**2)
            
            if model_type == 'exponential':
                kernel = fz1_row_est * np.exp(-distance / scale1_est)

            elif model_type == 'gaussian':
                kernel = fz1_row_est * np.exp(-(distance**2) / (2 * scale1_est**2))

            elif model_type == 'power_law':
                kernel = fz1_row_est / np.power(1 + distance / scale1_est, exponent1_est)
            
            return kernel
        
        "calculate R² for this row from posterior predictive"
        row_mask = obs_data['Row'] == row
        y_pred_row = y_pred[row_mask.values]
        r2_row = r2_score(row_sev, y_pred_row)
        row_r2_values[row] = r2_row
        
        # Create grid for contour
        x_min, x_max = row_x.min() - 2, row_x.max() + 2
        y_min, y_max = row_y.min() - 10, row_y.max() + 10
        
        X_grid, Y_grid = np.meshgrid(
            np.linspace(-15, 15, 50),
            np.linspace(y_min-5, y_max+5, 50)
        )
        
        # Create contour plot
        Z_pred = predict_row(X_grid, Y_grid)
        contour = ax.contourf(X_grid, Y_grid, Z_pred, levels=15, cmap='viridis', alpha=0.7)
        
        # Plot observed data
        scatter = ax.scatter(row_x, row_y, c=row_sev, s=80, cmap='viridis', 
                           edgecolors='black', linewidth=0.8, zorder=5)
        
        # Mark foci
        ax.scatter([fx1_row_est], [fy1_row_est], 
                  marker='*', s=200, color='red', 
                  edgecolors='darkred', linewidth=2, zorder=6, label='Focus 1')
        
        ax.set_xlabel('InterrowDistance')
        ax.set_ylabel('Distance')
        ax.set_title(f'Row {row}: R² = {r2_row:.3f} (fz1={fz1_row_est:.3f}')
        ax.legend()
        
        # Add colorbar for this subplot
        plt.colorbar(contour, ax=ax, label='Predicted Severity')
    
    fig.suptitle(f'{model_type.title()} Model', 
                 fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_dir + f'{model_type}_row_comparison_2d_sim{sim}.png', dpi=300, bbox_inches='tight')
    
    # Print R² summary
    print(f"\nRow-specific R² values for {model_type} model:")
    for row, r2 in row_r2_values.items():
        print(f"  Row {row}: R² = {r2:.3f}")

        

def plot_surfaces(trace, obs_data, y_pred, model_type, sim):
    """Create all surface plots - focused on row-specific analysis"""
    print(f"\nCreating surface plots for {model_type} model...")
    
    # Create 2D row comparison plots (returns R² values)
    print("Creating 2D row comparison plots...")
    plot_row_comparison_2d(trace, obs_data, y_pred, model_type, sim)
    
    # Create comprehensive 3D plot of all rows
    print("Creating 3D plot of all rows...")
    plot_3d_rows_subplots(trace, obs_data, y_pred, model_type, sim)



def model_performance(traces, obs_data, y_pred, focus_coords):
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
                
        "Calculate comprehensive model performance metrics for each row and overall data"            
        # Extract parameter estimates
        summary = az.summary(trace, hdi_prob=0.95)
        
        # Get shared parameters
        scale1_est = summary.loc['scale1', 'mean']    
        if model_name == "power_law":
            exponent1_est = summary.loc['exponent1', 'mean']
        
        # Get row effects
        unique_rows = sorted(obs_data['Row'].unique())
        
        # Calculate performance metrics for each row
        performance_metrics = {}
        all_observed = []  # Keep as Python list
        
        for idx, row in enumerate(unique_rows):
            # Get row-specific data
            row_data = obs_data[obs_data['Row'] == row]
            row_x = row_data['InterrowDistance'].values
            row_y = row_data['Distance'].values
            row_sev = row_data['Severity'].values
            
            # Row-specific focus coordinates and intensities
            fz1_row_est = summary.loc[f'fz1[{idx}]', 'mean']
            fx1_row_est = summary.loc[f'fx1[{idx}]', 'mean']
            fy1_row_est = summary.loc[f'fy1[{idx}]', 'mean']
                        
            # Calculate predictions and metrics
            row_mask = obs_data['Row'] == row
            y_pred_row = y_pred[row_mask.values]            
            predicted_sev = y_pred_row # posterior predictive
            
            # Store for overall calculation (extend lists, don't append arrays)
            all_observed.extend(row_sev)
            
            # R²
            r2 = r2_score(row_sev, predicted_sev)
            
            # RMSE
            rmse = np.sqrt(mean_squared_error(row_sev, predicted_sev))
            
            # MAE
            mae = mean_absolute_error(row_sev, predicted_sev)
            
            # Correlation - handle edge cases
            if len(row_sev) > 1 and np.var(row_sev) > 1e-10 and np.var(predicted_sev) > 1e-10:
                correlation = np.corrcoef(row_sev, predicted_sev)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
            
            performance_metrics[row] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Correlation': correlation,
                'n_points': len(row_sev),
                'fz1_intensity': fz1_row_est
            }
        
        # Calculate overall metrics after processing all rows
        all_observed = np.array(all_observed)
        all_predicted = y_pred
        

        overall_R2 = r2_score(all_observed, all_predicted)
        overall_RMSE = np.sqrt(mean_squared_error(all_observed, all_predicted))
        overall_MAE = mean_absolute_error(all_observed, all_predicted)
        overall_n_points = len(all_observed)

        
        # Calculate overall correlation
        if len(all_observed) > 1 and np.var(all_observed) > 1e-10 and np.var(all_predicted) > 1e-10:
            overall_correlation = np.corrcoef(all_observed, all_predicted)[0, 1]
            if np.isnan(overall_correlation):
                overall_correlation = 0
        else:
            overall_correlation = 0
                
        comparisons[model_name] = {'n_obs':n_obs, 
                               'elpd_loo': elpd_loo, 
                               'p_loo': p_loo, 'rank_loo':rank_loo,
                               'se_d_loo':se_d_loo, 'warning':warning, 
                               'pareto_k_bad':pareto_k_bad, 'overall_R2':overall_R2,
                               'overall_RMSE':overall_RMSE, 'overall_MAE':overall_MAE,
                               'overall_correlation':overall_correlation, 
                               'overall_n_points':overall_n_points,
                               'rows':performance_metrics}
        
    comparisons_df = pd.DataFrame(comparisons)
    
    return comparisons_df
        



def plot_posterior_predictive(y_sim, y_obs, distances, model_type, sim):
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
    ax3.set_xlabel('Distance from Focus', fontsize=11)
    ax3.set_ylabel('Severity', fontsize=11)
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
    
    plt.suptitle(f'{model_type.upper()} Model - Posterior Predictive Check (Sim {sim})',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    filename = f'posterior_predictive_check_{model_type}_sim{sim}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')



       

if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()
    
    # # HPC environment
    # n_cores = min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    n_cores = min(2, os.cpu_count())

    "simulate parameters"
    n_sims = 100
    sim_dir = "./sim/"
    output_dir = "./output/"
    sample_size = 50
    
    eps = 1e-08
    p0 = 0.3
    q_param = 0.01 #p1 is a constant value
    phi = 18
    model_type = 'power_law'
    scale1 = 5.0
    exponent1 = 2.0
    fx1_obs = 0  # Focus 1 X for each observation
    fy1_obs = 0  # Focus 1 Y for each observation
    fz1_obs = 0.6  # Focus 1 intensity for each observation
    focus1 = [fx1_obs, fy1_obs, fz1_obs]
    
    "define the sampling coordinates"
    y_coords = np.linspace(start=-2, stop=50, num=sample_size) # 10 points from Y=0 to Y=12
    y_coords = np.unique(np.concatenate([[fx1_obs], y_coords]))
    x_coords = np.full(y_coords.shape, 0.0) # X is always 0.0
    sample_points = np.column_stack((x_coords, y_coords))  # np.column_stack puts the arrays side-by-side
    
    "sampling from simulation"
    sample_kernel = simulation.single_focus_kernel(sample_points[:,0], sample_points[:, 1], focus1, scale1, model_type=model_type, exponent=exponent1)
    
    "p0 and mu related to the dispersal kernel"    
    p_param = np.clip(p0 * (1- sample_kernel), eps, 1- eps)
    mu = np.clip(sample_kernel, eps, 1- eps)
    
    alpha_samples = mu * phi
    beta_samples = (1 - mu) * phi
    
    prob_zero = p_param
    prob_one = (1.0 - p_param) * q_param
    prob_continuous = (1.0 - p_param) * (1.0 - q_param)   
    
    "simulate 100 datasets"
    df_id_current = 0
    rows_df = pd.DataFrame(columns=['InterrowDistance', 'Distance', 'Severity', 'Row', 'Simulation', 'Plant'])
    for run_id in range(n_sims * 4):
        np.random.seed(run_id)
        
        y_obs_run = simulation.sampling(sample_points, focus1, prob_zero, prob_one, prob_continuous, alpha_samples, beta_samples)
        row_val = run_id%4
        row_label = np.full(len(sample_points), row_val)
        sim_label = np.full(len(sample_points), run_id)
        y_obs_run['Row'] = row_label
        y_obs_run['Simulation'] = sim_label
                
        "add row label to the simulation for hierachical Bayesian model"
        df_id = int(run_id / 4)
        if df_id != df_id_current:
            rows_df.to_csv(sim_dir + "simulated_disease_data_" + str(df_id_current) + ".csv")
            
            df_id_current = df_id
            rows_df = pd.DataFrame(columns=['InterrowDistance', 'Distance', 'Severity', 'Row', 'Simulation', 'Plant'])

        rows_df = pd.concat([rows_df, y_obs_run])
            
    rows_df.to_csv(sim_dir + "simulated_disease_data_" + str(df_id_current) + ".csv")


    """Main execution function for multi-row analysis"""        
    print("Loading multi-row disease severity data...")
    
    output_file = output_dir + 'ppc_metrics_all_sims.csv'
    pd.DataFrame(columns=['simulation', 'model_type', 'overall_r2', 'overall_rmse', 'overall_mae',
                          'obs_zeros', 'pred_zeros', 'obs_ones', 'pred_ones', 'continuous_r2', 
                          'continuous_rmse']).to_csv(output_file, index=False) # store all ppc metrics
    
    for sim in range(n_sims):
        df = pd.read_csv(sim_dir + "simulated_disease_data_" + str(sim) + ".csv")
    
        obs_data, focus_coords = load_and_process_multi_row_data(df, fz1_value=fz1_obs)
        full_obs_data = df
        
        print("\nDataset summary:")
        print(f"Total observations: {len(full_obs_data)}")
        print(f"Rows analyzed: {sorted(full_obs_data['Row'].unique())}")
        print(full_obs_data.groupby('Row')['Severity'].describe())
        
        # Test different models
        models_to_test = ['exponential', 'gaussian', 'power_law']
        traces = {}
        
        for model_type in models_to_test:
            print(f"\n{'='*60}")
            print(f"FITTING {model_type.upper()} MODEL (MULTI-ROW)")
            print(f"{'='*60}")
                        
            model = create_multi_row_zoib_model(full_obs_data, model_type=model_type)            
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
                trace.to_netcdf(output_dir + f'{model_type}_trace_{sim}.nc')
                
                "posterior predictive check"
                # generate posterior predictive samples
                ppc = pm.sample_posterior_predictive(trace, var_names=['obs'], random_seed=42+sim)
                
                # extract generated data
                y_sim = ppc.posterior_predictive['obs'].values                
                y_obs = full_obs_data['Severity'].values
                distances = full_obs_data['Distance'].values
                
                # plot posterior predictive check
                plot_posterior_predictive(y_sim, y_obs, distances, model_type, sim)                
                y_pred = ppc.posterior_predictive['obs'].mean(dim=['chain', 'draw']).values
                
                "calculate ppc metrics"
                mask_cont = (y_obs > 0) & (y_obs < 1)
                y_obs_cont = y_obs[mask_cont]
                y_pred_cont = y_pred[mask_cont]
                
                metrics = {
                        'simulation': sim,
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
                diagnostics.to_csv(output_dir + f'{model_type}_disgnostic_sim{sim}.csv')
                summary.to_csv(output_dir + f'{model_type}_trace_summary_sim{sim}.csv')
                                    
                "create plots if diagnostics worked"                
                print("\nCreating result plots...")
                plot_multi_row_results(trace, full_obs_data, focus_coords, y_pred, model_type, sim)
                print("✓ Plots created successfully")
                
                # Add surface plots with row-specific R²
                plot_surfaces(trace, full_obs_data, y_pred, model_type, sim)
                print("✓ Surface plots created successfully")
                        
                "calculate effective sample sizes"
                ess_df = calculate_all_ess(trace)
                ess_df['ess_num'] = ess_df['ess'].astype(str).str.extract(r'(\d+\.\d+e[+-]\d+)>')
                ess_df['ess_num'] = pd.to_numeric(ess_df['ess_num'])
                ess_df.to_csv(output_dir + f"{model_type}_effective_sample_size_sim{sim}.csv")



                    
        
        "compare models if multiple successful"
        if len(traces) > 1:
            print(f"\n{'='*60}")
            print("MODEL COMPARISON")
            print(f"{'='*60}")
            
            loo_comparison = az.compare(traces, ic='loo')
            comp_df = model_performance(traces, full_obs_data, y_pred, focus_coords)
            comp_df.to_csv(output_dir + f"models_performances_sim{sim}.csv")
            
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
    
