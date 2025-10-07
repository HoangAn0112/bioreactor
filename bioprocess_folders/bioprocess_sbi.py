from bioprocess_utils import *
# from ipywidgets import interact, FloatSlider

inputs_dir = "./inputs"
outputs_dir = "./outputs"
loaded = load_model_inputs(inputs_dir, basename="inputs")
sol = run_from_inputs(loaded)
written = save_solution(sol, loaded["params"], loaded["initials"],
                        out_dir=outputs_dir, basename="solution")
vars_to_show = ["Biomass", "Sub"]
# plot_timeseries(
#     sol,
#     variables=vars_to_show,
#     out_dir=outputs_dir,
#     basename="solution__"+"_".join(vars_to_show),
#     show=True,   # set False for headless runs
#     save=True
# )

print("Inputs saved to:", inputs_dir)
print("Solution files:", written)