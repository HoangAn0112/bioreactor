import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from bioprocess_utils import *
import pandas as pd


fp = Path("bioreactor_18_751_5000_50_inference_log.npz")
npz = np.load(fp, allow_pickle=True)

# as numpy arrays
posterior = {k: npz[k] for k in npz.files}
print("keys:", list(posterior.keys()))
for k, v in posterior.items():
    print(k, type(v), v.shape)

num_reactor = 18
num_simulation = 5000
num_samples = 15

name = f"bioreactor_{num_reactor}_{num_simulation}_{num_samples}_inference_log"
data = pd.read_csv(f"experimental_dataset/bioreactor_{num_reactor}_750.csv")
initial_observables = data.iloc[0,1]

inputs_dir = "./experimental_inputs"
outputs_dir = "./experimental_outputs"
params_to_infer = ["mu_max", "K_subs", "K_L_a", "Y_Sub"]
observables = ["Biomass"]
loaded = load_model_inputs(inputs_dir, basename="ambr_run1_140323_13-18__R1__inputs")
mecanistic_solution= run_from_inputs(loaded)
param_distributions = {
    "mu_max":   {"dist": "uniform", "low": 1e-4, "high": 5e-2},
    "K_subs":   {"dist": "uniform",  "low": 1e-2, "high": 10},
    "K_L_a":    {"dist": "uniform", "low": 0.05, "high": 3},
    "Y_Sub":    {"dist": "normal", "loc": 0.35, "scale": 0.05},
}

time_min = data["time"].min()
time_max = data["time"].max()
loaded["t_span"] = [time_min, time_max]
loaded["t_eval"] = data["time"].values
loaded["initials"]["Biomass"]["values"] = [initial_observables]

def prior_sample(batch_size=1, param_distributions=param_distributions):
    """Sample n_samples from specified distributions for each parameter."""
    if param_distributions is None:
        raise ValueError("param_distributions must be provided.")

    prior_sample = {k: [] for k in params_to_infer}
    for _ in range(batch_size):
        sample = []
        for k in params_to_infer:
            dist_info = param_distributions[k]
            
            if dist_info["dist"] == "uniform":
                v = np.random.uniform(dist_info["low"], dist_info["high"])
            elif dist_info["dist"] == "normal":
                v = np.random.normal(dist_info["loc"], dist_info["scale"])
            elif dist_info["dist"] == "lognormal":
                v = np.random.lognormal(dist_info["mean"], dist_info["sigma"])
            else:
                raise ValueError(f"Unknown distribution: {dist_info['dist']}")
            # Scale by loaded value if needed
            # v = v * loaded["params"][k]["values"][0]
            prior_sample[k].append(v)
    for k in prior_sample:
        prior_sample[k] = np.array(prior_sample[k])
    return prior_sample


def mechanistic_solver(**prior_sample):
    """Run the mechanistic model with given 1 set of parameter and return biomass predictions

    """
    full_inputs = loaded.copy()
    for k, v in prior_sample.items():
        full_inputs["params"][k]["values"] = v

    mecanistic_solution = run_from_inputs(full_inputs)
    result = {}
    for obs in observables:
        if obs in mecanistic_solution.state_names:
            idx = mecanistic_solution.state_names.index(obs)
            result[obs] = np.asarray(mecanistic_solution.y[idx]).flatten()
        else:
            raise ValueError(f"Observable '{obs}' not found in state_names: {mecanistic_solution.state_names}")
    # result.update(prior_sample)
    return result


def mechanistic_solver_array(**kwargs):
    first_key = next(iter(kwargs))
    first_val = kwargs[first_key][0]
    n = len(first_val)
    results = []

    for idx in range(n):
        single_kwargs = {
            k: (v[0][idx] if isinstance(v, (np.ndarray, list)) else v)
            for k, v in kwargs.items()
        }
        results.append(mechanistic_solver(**single_kwargs))

    combined = {}
    for key in results[0].keys():
        # Get all arrays for this key
        arrays = [np.array(d[key]) for d in results if key in d]

        # Determine the maximum length
        max_len = max(len(a) for a in arrays if a.ndim > 0)

        # Keep only arrays matching the max length
        filtered = [a for a in arrays if len(a) == max_len]

        # Stack them safely
        try:
            arr = np.stack(filtered)
        except ValueError:
            # fallback if they have different shapes in other dims
            arr = np.array(filtered, dtype=object)

        if arr.ndim > 2:
            arr = arr.squeeze()

        combined[key] = arr

    return combined

biomass_result = mechanistic_solver_array(**posterior)
j_list = list([31])

for i, obs in enumerate(observables):
    plt.figure(figsize=(8, 5))
    h = 0
    for j in j_list:
        res = biomass_result[obs][j]
        plt.plot(data["time"], biomass_result[obs][j],linewidth=2.5, label=f"ODE solution {h+1}")
        h += 1
    plt.scatter(data["time"], data[obs], color="C0", label="Experimental data")
    from sklearn.metrics import r2_score
    r2 = r2_score(data[obs], biomass_result[obs][j])
    # plt.text(f"R² Score: {r2:.4f}")
    plt.xlabel("Time (minutes)")
    plt.ylabel(obs)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"outputs/bioreactor_{num_reactor}_{num_simulation}_{num_samples}_inference_log.png")
    plt.show()  


print("done")
    # plt.close()  

