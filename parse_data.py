import os
import pandas as pd
import numpy as np
import pymc as pm

adjacency_matrix = os.path.join(os.path.dirname(__file__),"data","adjacency_matrix_correct.parquet")
aero_anac = os.path.join(os.path.dirname(__file__), "data", "aero_anac_2017_2023.parquet")
fluvi_road = os.path.join(os.path.dirname(__file__),"data","fluvi_road_ibge.parquet")

df_adj = pd.read_parquet(adjacency_matrix)

# dummy safe data (prevents key errors)
n = 100
k = 2
n_regions = 10

y = np.random.poisson(3, size=n)
X = np.random.randn(n, k)
region_idx = np.random.randint(0, n_regions, size=n)

# safe adjacency → Laplacian
W = np.eye(n_regions)
L = np.diag(W.sum(1)) - W

with pm.Model() as model:
    beta = pm.Normal("beta", 0, 1, shape=k)
    region = pm.Normal("region", 0, 1, shape=n_regions)
    spatial = pm.MvNormal("spatial", mu=np.zeros(n_regions), tau=L + np.eye(n_regions)*1e-5)

    mu = X @ beta + region[region_idx] + spatial[region_idx]
    pm.Poisson("y_obs", mu=pm.math.exp(mu), observed=y)

    trace = pm.sample(500, tune=500, progressbar=False)

#incorporate a bayesian hierarchical model, with inputs of
#our datatypes, as well as cases from our last project
#need to check the granularity of data, pivot if annual
#feed in cases.csv for testing on our own data
#maybe try a monte carlo markov chain simulation