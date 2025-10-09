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
import bayesflow as bf

data = pd.read_csv("experimental_dataset/bioreactor_18.csv")
inputs_dir = "./experimental_inputs"
outputs_dir = "./experimental_outputs"
params_to_infer = ["mu_max", "K_subs", "K_L_a"]
observables = ["Biomass"]
loaded = load_model_inputs(inputs_dir, basename="inputs")
mecanistic_solution= run_from_inputs(loaded)

time_min = data["time"].min()
time_max = data["time"].max()
loaded["t_span"] = [time_min, time_max]
loaded["t_eval"] = data["time"].values


def prior_sample(low=0, high=3):
    """Sample n_samples from a uniform distribution [low, high)."""
    arr = np.random.uniform(low, high, size=len(params_to_infer))
    # Scale by loaded["values"][0] for each parameter
    scaled = arr * np.array([loaded["params"][k]["values"][0] for k in params_to_infer])
    return {k: [v] for k, v in zip(params_to_infer, scaled)}

example_prior = prior_sample()

def simulator(parameter):
    """Run the mechanistic model with given parameters and return biomass predictions

    """
    full_inputs = loaded.copy()
    for k, v in parameter.items():
        full_inputs["params"][k]["values"] = v

    mecanistic_solution = run_from_inputs(full_inputs)
    result = {}
    for obs in observables:
        if obs in mecanistic_solution.state_names:
            idx = mecanistic_solution.state_names.index(obs)
            result[obs] = mecanistic_solution.y[idx]
        else:
            result[obs] = None

    result.update(parameter)

    return result

biomass = simulator(example_prior)

simulator = bf.make_simulator([prior_sample, simulator])
num_trajectories = 1000
samples = simulator.sample(num_trajectories)
print("Example prior sample:", example_prior)
