import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if not os.environ.get("KERAS_BACKEND"):
    os.environ["KERAS_BACKEND"] = "jax"

from bioprocess_utils import *
import pandas as pd
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.integrate import odeint
import keras
import bayesflow as bf
from tools import *

num_reactor = 18
data = pd.read_csv(f"experimental_dataset/bioreactor_{num_reactor}.csv")
inputs_dir = "./experimental_inputs"
outputs_dir = "./experimental_outputs"
params_to_infer = ["mu_max", "K_subs", "K_L_a"]
observables = ["Biomass"]
loaded = load_model_inputs(inputs_dir, basename="ambr_run1_140323_13-18__R1__inputs")
mecanistic_solution= run_from_inputs(loaded)

param_distributions = {
    "mu_max":   {"dist": "uniform", "low": 1e-4, "high": 1},
    "K_subs":   {"dist": "uniform",  "low": 1e-2, "high": 10},
    "K_L_a":    {"dist": "uniform", "low": 0.05, "high": 3}
}



time_min = data["time"].min()
time_max = data["time"].max()
loaded["t_span"] = [time_min, time_max]
loaded["t_eval"] = data["time"].values


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
            v = v * loaded["params"][k]["values"][0]
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
    first_val = kwargs[first_key]
    n = len(first_val)
    results = []
    for idx in range(n):
        single_kwargs = {k: (v[idx] if isinstance(v, (np.ndarray, list)) else v) for k, v in kwargs.items()}
        results.append(mechanistic_solver(**single_kwargs))
        # for k, v in single_kwargs.items():
        #     results[k] = np.array(v)
        # results.append(results)
    combined = {}
    for key in results[0].keys():
        arr = np.stack(np.array([d[key] for d in results]))
        if arr.ndim > 2:
            arr = arr.squeeze()
        combined[key] = arr
    return combined


simulator = bf.make_simulator([prior_sample, mechanistic_solver])

# sampled_params = prior_sample(low=0, high=2, batch_size=5)
# samples = mechanistic_solver_array(**sampled_params)
data_keys = params_to_infer + observables

adapter = (
    bf.adapters.Adapter()
    .convert_dtype("float64", "float32")
    .concatenate(params_to_infer, into="inference_variables")
    # .concatenate(observables, into="summary_variables")
    .as_time_series(observables)
)

optimizer = keras.optimizers.Adam(
    learning_rate=0.0001,
    global_clipnorm=1.0
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

inference_network = bf.networks.CouplingFlow()

workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_network,
        optimizer=optimizer,
        callbacks=lr_scheduler,
        standardize=["inference_variables", "summary_variables"]
    )

epochs = 200
batch_size = 64
num_simulation = 1000
num_samples=5
samples = simulator.sample(num_simulation)
sampling = remove_nan_rows(samples, num_simulation)
filename = f"simulated_dataset/bioractor_{num_reactor}_{num_simulation}.pth"
torch.save(sampling, filename)

plot_sampling_results(sampling, 
                      obseravble_variable = observables, 
                      size = num_simulation, 
                      save_path = f"simulated_dataset/bioractor_{num_reactor}_")

# grid = bf.diagnostics.plots.pairs_samples(
#     samples, variable_keys=observables
# )

history = workflow.fit_offline(
        samples,  
        epochs=epochs,
        batch_size=batch_size,
)
f = bf.diagnostics.plots.loss(history)
plt.show()

data_infer = {i:np.array(data[i]).astype("float32").reshape(1,-1) for i in observables}
posterior = workflow.sample(conditions=data_infer, num_samples=num_samples)
biomass_result = mechanistic_solver_array(**posterior)

for i, obs in enumerate(observables):
    plt.figure(figsize=(8, 5))
    for j in range(num_samples):
        plt.plot(data["time"], biomass_result[obs][j], alpha=0.3, label=f"ODE solution {j+1}")
    plt.scatter(data["time"], data[obs], color="C0", label="Experimental data")
    plt.xlabel("Time (h)")
    plt.ylabel(obs)
    plt.legend()
    plt.show()