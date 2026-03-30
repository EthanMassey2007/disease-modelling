import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

adjacency_matrix = os.path.join(os.path.dirname(__file__),"data","adjacency_matrix_correct.parquet")
aero_anac = os.path.join(os.path.dirname(__file__), "data", "aero_anac_2017_2023.parquet")
fluvi_road = os.path.join(os.path.dirname(__file__),"data","fluvi_road_ibge.parquet")

cases = os.path.join(os.path.dirname(__file__),"data","cases.csv")
humidity = os.path.join(os.path.dirname(__file__), "data", "humidity.csv")
idhm = os.path.join(os.path.dirname(__file__),"data","idhm.csv")
rainfall = os.path.join(os.path.dirname(__file__),"data","rainfall.csv")
temperature = os.path.join(os.path.dirname(__file__), "data", "temperature.csv")


# Simulated data
np.random.seed(42)
n_groups = 5
n_per_group = 20

true_mu_group = np.random.normal(0, 2, n_groups)
data = [np.random.normal(mu, 1, n_per_group) for mu in true_mu_group]

# Flatten data
y = np.concatenate(data)
group_idx = np.concatenate([[i]*n_per_group for i in range(n_groups)])

# Model
with pm.Model() as model:

    # Hyperpriors (global level)
    mu_global = pm.Normal("mu_global", mu=0, sigma=5)
    sigma_global = pm.HalfNormal("sigma_global", sigma=5)

    # Group-level parameters
    mu_group = pm.Normal("mu_group", mu=mu_global, sigma=sigma_global, shape=n_groups)

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=5)

    # Likelihood
    y_obs = pm.Normal("y_obs", mu=mu_group[group_idx], sigma=sigma, observed=y)

    # MCMC sampling
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Posterior summary
print(az.summary(trace, var_names=["mu_global", "sigma_global", "mu_group", "sigma"]))

# Optional: visualize
az.plot_trace(trace)

#Maybe bring in our own adjacency matrix from last project and compare to this, and run both to get posteriors with MCMC
#Need to do more research on background behind MCMC and Bayesian model connectedness 