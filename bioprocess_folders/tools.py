import keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def remove_nan_rows(data_dict, length = None):
    """
    Removes all values at positions (indices) where any key in the dictionary
    has a NaN value. Assumes all values in the dictionary are lists of equal length.
    
    Args:
        data_dict (dict): Dictionary with list values.
        
    Returns:
        dict: A new dictionary with NaN-containing rows removed across all keys.
    """
    # Convert everything to np.array for consistency
    data_dict = {k: np.atleast_2d(np.array(v)) for k, v in data_dict.items()}

    # Infer length (rows)
    if length is None:
        length = next(iter(data_dict.values())).shape[0]

    # Build a mask of valid (non-NaN) rows
    valid_mask = np.ones(length, dtype=bool)
    for key, array in data_dict.items():
        if array.shape[0] != length:
            raise ValueError(f"Key '{key}' has inconsistent length: {array.shape[0]} != {length}")
        valid_mask &= ~np.isnan(array).any(axis=1)

    # Apply mask and ensure 2D shape
    cleaned_dict = {key: array[valid_mask] for key, array in data_dict.items()}

    return cleaned_dict

def check_for_nan_inf(data):
    flat_data = keras.tree.flatten(data)
    for i, tensor in enumerate(flat_data):
        arr = keras.ops.convert_to_numpy(tensor)
        if np.isnan(arr).any():
            print(f"Tensor {i} has NaNs.")
        if np.isinf(arr).any():
            print(f"Tensor {i} has Infs.")
    print("No INF")


def plot_sampling_results(sampling, size= int, obseravble_variable = ['GLC', 'ACE_env', 'X'], save_path="data/sampled_dataset_"):
    """
    Visualize time-series data and parameter distributions from sampling results.
    Fixed version with dimension checks and error handling.
    """
    save_path = save_path + str(size) + ".png"
    try:
        # Convert time-series data to numpy arrays with dimension validation
        time_series = {}
        for var in obseravble_variable:
            data = keras.ops.convert_to_numpy(sampling[var])
            if data.ndim == 3:  # (batch_size, time_steps, 1)
                data = data.squeeze(-1)  # Remove singleton dimension if exists
            time_series[var] = data

        # Get time dimension from first variable
        n_time_points = next(iter(time_series.values())).shape[1]
        # n_time_points = time_series[key].shape[1]
        time_points = np.arange(n_time_points)  # Simple 0-based time index

        # --- Parameter Extraction ---
        param_keys = [k for k in sampling.keys() if k not in obseravble_variable]
        n_params = len(param_keys)  

        # Create figure with two columns
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.3)

        # --- Time Series Plot ---
        ts_gs = gs[0].subgridspec(3, 1)
        ts_axes = [fig.add_subplot(ts_gs[i]) for i in range(3)]

        for ax, (var_name, data) in zip(ts_axes, time_series.items()):
            # Validate dimensions
            if data.shape[1] != n_time_points:
                raise ValueError(f"Time dimension mismatch in {var_name}: "
                               f"expected {n_time_points}, got {data.shape[1]}")

            # Calculate statistics
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            q05, q95 = np.percentile(data, [5, 95], axis=0)

            # Plotting
            ax.plot(time_points, mean, 'o-', color='blue', label='Mean', markersize=3)
            ax.fill_between(time_points, mean - std, mean + std,
                          alpha=0.3, color='blue', label='±1 SD')
            ax.fill_between(time_points, q05, q95,
                          alpha=0.15, color='blue', label='90% CI')
            
            # Formatting
            ax.set_title(f'{var_name} over Time', fontsize=10)
            ax.set_xlabel('Time Step', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # --- Parameter Distributions ---
        max_cols = 2
        n_cols = min(max_cols, n_params)
        n_rows = int(np.ceil(n_params / n_cols))
        param_gs = gs[1].subgridspec(n_rows, n_cols)
        
        for i, key in enumerate(param_keys):
            row = i // n_cols
            col = i % n_cols
            ax = fig.add_subplot(param_gs[row, col])
            data = keras.ops.convert_to_numpy(sampling[key]).flatten()
            
            sns.histplot(data, ax=ax, kde=True, bins=15,
                        color='green', alpha=0.7,
                        edgecolor='darkgreen', linewidth=0.5)
            sns.rugplot(data, ax=ax, color='darkgreen', height=0.05)
            ax.axvline(np.mean(data), color='red', linestyle='--', linewidth=1)
            
            ax.set_title(f'{key}', fontsize=9)
            ax.set_xlabel('Parameter Value', fontsize=7)
            ax.grid(True, alpha=0.2)

        plt.suptitle(f'Sampling Results (n={size})', y=1.02, fontsize=12)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Plot saved to {save_path}")

    except Exception as e:
        print(f"❌ Plotting failed: {str(e)}")
        # Debug information
        print("\nDebug Info:")
        print(f"Time series shapes: { {k: v.shape for k, v in time_series.items()} }")
        if 'param_keys' in locals():
            print(f"Parameter keys: {param_keys}")
        raise  # Re-raise after showing debug info

from typing import Dict, Any

def load_sampled_data(file_path: str) -> Dict[str, Any]:
    """Load pre-sampled data from torch.save file"""
    try:
        data = torch.load(file_path)
        # Convert all tensors to numpy arrays
        return {k: v.numpy() if torch.is_tensor(v) else np.array(v) 
                for k, v in data.items()}
    except Exception as e:
        raise IOError(f"Error loading {file_path}: {str(e)}")

from typing import Dict, Tuple, Optional

def plot_results(sim_data_dict, ode_data):
    """
    Plot simulation results vs experimental data.

    Parameters:
    - sim_data_dict: dict with keys ['GLC', 'ACE_env', 'X', 'ACCOA', 'ACP', 'ACE_cell']
    - data_file_path: path to folder containing CSV file
    - data_filename: name of CSV file (default 'data_1mM.csv')
    """

    # Time points from data (excluding header)
    data_t = ode_data[1:, 0]


# Plot configuration
    plot_configs = [
        ('GLC',      0, 0, 3, 'GLC (mM)',      'b'),
        ('ACE_env',  0, 1, 1, 'ACE_env (mM)',  'g'),
        ('X',        0, 2, 2, 'X (g_DW/L)',    'r'),
        # ('ACCOA',    1, 0, None, 'ACCOA',     'm'),
        # ('ACP',      1, 1, None, 'ACP',       'c'),
        # ('ACE_cell', 1, 2, None, 'ACE_cell',  'y')
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 3))

    for key, _, col, data_col, ylabel, color in plot_configs:
        ax = axs[col]
        sim_values = sim_data_dict.get(key)
        if sim_values is None or len(sim_values) == 0:
            print(f"[Warning] No simulation data found for '{key}'. Skipping plot.")
            continue

        ax.plot(data_t, sim_values, label=f'{key}_gen', color=color)

        # Plot data if available
        if data_col is not None:
            ax.plot(data_t, ode_data[1:, data_col], 'o', label=f'{key}_data', color=color)

        ax.set_title(f'{key} 1mM')
        ax.set_xlabel('t (h)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_results_grid(all_sim_data, ode_data):
    """
    Plot multiple simulation results vs experimental data in a grid.
    Each row shows one simulation result with experimental data for comparison.
    Columns show different variables.
    """
    # Time points from data (excluding header)
    data_t = ode_data[1:, 0]

    # Variables to plot and their configuration
    plot_configs = [
        ('GLC', 3, 'GLC (mM)', 'b'),
        ('ACE_env', 1, 'ACE_env (mM)', 'g'),
        ('X', 2, 'X (g_DW/L)', 'r'),
        ('ACCOA',4, 'ACCOA 1mM', 'm'),
        ('ACP',5,'ACP 1mM','c'),
        ('ACE_cell',6,'ACE_cell 1mM','y')
    ]
    
    n_simulations = len(all_sim_data)
    n_variables = len(plot_configs)
    
    # Create figure with one row per simulation, columns for each variable
    fig, axs = plt.subplots(n_simulations, n_variables, 
                            figsize=(4*n_variables, 3*n_simulations),
                            squeeze=False)
    
    # Plot each simulation in its own row
    for row_idx, sim_data_dict in enumerate(all_sim_data):
        for col_idx, (key, data_col, ylabel, color) in enumerate(plot_configs):
            ax = axs[row_idx, col_idx]
            
            sim_values = sim_data_dict.get(key)
            if sim_values is None or len(sim_values) == 0:
                print(f"[Warning] No simulation data found for '{key}' in simulation {row_idx}. Skipping plot.")
                continue

            if data_col >= ode_data.shape[1]:
                print(f"[Info] Experimental data column {data_col} is out of bounds. Replacing with NaNs.")
                exp_values = np.full_like(data_t, np.nan, dtype=np.float64)
            else:
                exp_values = ode_data[1:, data_col]

            # Check if experimental data is valid (not all NaNs or zeros)
            exp_data_valid = not (np.all(np.isnan(exp_values)) or np.all(exp_values == 0))

            # Ensure matching length
            if len(sim_values) != len(exp_values) or not exp_data_valid:
                if len(sim_values) != len(exp_values):
                    print(f"[Warning] Length mismatch for '{key}' in simulation {row_idx}. Skipping SSR.")
                elif not exp_data_valid:
                    print(f"[Info] No valid experimental data for '{key}' in simulation {row_idx}. Skipping exp plot.")
                ssr = np.nan
            else:
                # Calculate SSR (Sum of Squared Residuals)
                ssr = np.sum((sim_values - exp_values)**2)

            # Plot simulation (line)
            ax.plot(data_t, sim_values, color=color, label=f'{key}_sim')

            # Only plot experimental data if valid
            if exp_data_valid:
                ax.plot(data_t, exp_values, 'o', 
                        color=color, label=f'{key}_data', markersize=4)
            
            # X-axis label only on bottom row
            if row_idx == n_simulations - 1:
                ax.set_xlabel('t (h)')
            else:
                ax.set_xticklabels([])
            
            # Y-axis label
            ax.set_ylabel(ylabel)
            
            # Title with SSR
            ax.set_title(f'{key} SSR = {ssr:.2f}')
            
            ax.legend(fontsize='small')
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()