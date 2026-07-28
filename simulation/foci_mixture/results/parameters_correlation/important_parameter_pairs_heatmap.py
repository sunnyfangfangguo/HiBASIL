#!/usr/bin/env python3
"""
CUSTOMIZABLE HEATMAP SCRIPT

Modify the 'selected_pairs' list below to choose which correlations to plot
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# MODIFY THIS LIST - Choose your parameter pairs!
# ============================================================

selected_pairs = [
    # Copy any pairs from PARAMETER_PAIRS_LIST.md
    # Format: ('parameter1', 'parameter2')
    
    ('scale1', 'exponent1'),      # Scale-exponent confounding
    ('scale2', 'exponent2'),      
    ('fx1', 'scale1'),            # Location-dispersal
    ('fx2', 'scale2'),
    ('fz1', 'fz2'),               # Intensity coupling
    ('fx2', 'fy2'),               # Secondary location
    ('fz1', 'weight'),            # Intensity-weight
    ('fz2', 'weight'),
    ('fx1', 'fz1'),               # Location-intensity
    ('fx2', 'fz2')
]

# ============================================================
# DATA LOADING AND PLOTTING (don't need to modify below)
# ============================================================

# File paths
scenario_files = {
    'Balanced\n[0.5,0.5]': './power_law_ALL_correlations_complete_w5050.csv',
    'Slight\n[0.6,0.4]': './power_law_ALL_correlations_complete_w6040.csv',
    'Heavy\n[0.8,0.2]': './power_law_ALL_correlations_complete_w8020.csv',
    'Extreme\n[0.95,0.05]': './power_law_ALL_correlations_complete_w9505.csv'
}

print(f"Loading data for {len(selected_pairs)} parameter pairs...")

# Extract correlations
results = []

for scenario_name, filepath in scenario_files.items():
    df = pd.read_csv(filepath)
    scenario_data = {'Scenario': scenario_name}
    
    for param1, param2 in selected_pairs:
        # Find correlation (checks both directions)
        row = df[((df['Parameter 1'] == param1) & (df['Parameter 2'] == param2)) |
                 ((df['Parameter 1'] == param2) & (df['Parameter 2'] == param1))]
        
        if len(row) > 0:
            label = f"{param1}↔{param2}"
            r_value = float(row['Pearson r'].values[0])
            scenario_data[label] = r_value
        else:
            print(f"  Warning: {param1}-{param2} not found in {scenario_name}")
            scenario_data[f"{param1}↔{param2}"] = float('nan')
    
    results.append(scenario_data)

# Create DataFrame
comparison_df = pd.DataFrame(results)
print("\n✓ Loaded data successfully")
print(f"  Scenarios: {len(comparison_df)}")
print(f"  Parameters: {len(selected_pairs)}")

# Prepare for heatmap (transpose)
data = comparison_df.set_index('Scenario').T

# Create heatmap
fig_height = max(8, len(selected_pairs) * 0.5)
plt.figure(figsize=(10, fig_height))

sns.heatmap(data, 
            annot=True,              # Show values
            fmt='.3f',               # 3 decimal places
            cmap='RdBu_r',           # Red-Blue reversed
            center=0,                # White at zero
            vmin=-1,                 # Min correlation
            vmax=1,                  # Max correlation
            linewidths=0.5,          # Grid lines
            linecolor='white',
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.7})

plt.xlabel('Weight Mixtures', fontsize=12, fontweight='bold')
plt.ylabel('Parameter Pair', fontsize=12, fontweight='bold')
plt.tight_layout()

# Save
output_file = './foci_imbalance_important_paramters_pair_heatmap.png'
plt.savefig(output_file, dpi=600, bbox_inches='tight')
print(f"\n✓ Heatmap saved: {output_file}")

# Print summary
print("\n" + "="*60)
print("CORRELATION SUMMARY")
print("="*60)

for col in data.columns:
    values = data[col].values
    print(f"\n{col}:")
    for idx, param_pair in enumerate(data.index):
        param_pair_clean = param_pair.replace('↔', ' ↔ ')
        print(f"  {param_pair_clean:<25} {values[idx]:>+.3f}")

print("\n" + "="*60)
print("Done!")
print("="*60)

plt.show()
