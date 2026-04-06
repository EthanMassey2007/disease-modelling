import os
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================
# Helpers
# =========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip().str.lower()
    return df


def normalize_municipio_name(name):
    if pd.isna(name):
        return np.nan

    name = str(name).strip().lower()

    # Remove trailing /UF, e.g. "São Paulo/SP" -> "São Paulo"
    name = re.sub(r"/[a-z]{2}$", "", name)

    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name


def iso_week_to_month(year_series, week_series):
    dt = pd.to_datetime(
        year_series.astype(str)
        + "-W"
        + week_series.astype(int).astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    return dt, dt.dt.month


# =========================================================
# Config
# =========================================================
USE_FULL_COVARIATE_SET = False
MAKE_PLOTS = True

# Start with smaller covariate set for better convergence
BASE_COVARIATES = [
    "rainfall",
    "humidity",
    "air_pass_in",
    "idhm",
]

FULL_COVARIATES = [
    "rainfall",
    "humidity",
    "temperature",
    "idhm",
    "air_pass_in",
    "road_conec_in",
    "fluv_conec_in",
]

# =========================================================
# File paths
# =========================================================
base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")

cases_file = os.path.join(data_dir, "cases.csv")
temperature_file = os.path.join(data_dir, "temperature.csv")
humidity_file = os.path.join(data_dir, "humidity.csv")
rainfall_file = os.path.join(data_dir, "rainfall.csv")
idhm_file = os.path.join(data_dir, "idhm.csv")

municipios_file = os.path.join(data_dir, "municipios.csv")
aero_file = os.path.join(data_dir, "aero_anac_2017_2023.parquet")
fluvi_file = os.path.join(data_dir, "fluvi_road_ibge.parquet")


# =========================================================
# Load data
# =========================================================
cases_df = clean_columns(pd.read_csv(cases_file))
temp_df = clean_columns(pd.read_csv(temperature_file))
hum_df = clean_columns(pd.read_csv(humidity_file))
rain_df = clean_columns(pd.read_csv(rainfall_file))
idhm_df = clean_columns(pd.read_csv(idhm_file))

municipios_df = clean_columns(pd.read_csv(municipios_file))
aero_df = clean_columns(pd.read_parquet(aero_file))
fluvi_df = clean_columns(pd.read_parquet(fluvi_file))


# =========================================================
# Normalize municipio names in the main CSVs
# =========================================================
for d in [cases_df, temp_df, hum_df, rain_df, idhm_df]:
    d["municipio"] = d["municipio"].apply(normalize_municipio_name)


# =========================================================
# Build RJ-only city -> IBGE lookup from municipios.csv
# This fixes ambiguous names like "valenca"
# =========================================================
municipios_df["state_code"] = municipios_df["city"].astype(str).str.extract(r"/([A-Z]{2})$")
municipios_df["city_clean"] = municipios_df["city"].apply(normalize_municipio_name)
municipios_df["ibgeid"] = pd.to_numeric(municipios_df["ibgeid"], errors="coerce").astype("Int64")
# Restrict to RJ only
municipios_df = municipios_df[
    (municipios_df["state_code"] == "RJ") 
].copy()
print(municipios_df.head())
lookup_df = (
    municipios_df[["city_clean", "ibgeid"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={"city_clean": "municipio", "ibgeid": "ibge_code"})
)

dup_counts = lookup_df.groupby("municipio")["ibge_code"].nunique()
ambiguous_after_filter = dup_counts[dup_counts > 1]

print("Ambiguous names after RJ filter:")
print(ambiguous_after_filter)


# =========================================================
# Numeric conversion
# =========================================================
for d in [cases_df, temp_df, hum_df, rain_df, idhm_df]:
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d["week"] = pd.to_numeric(d["week"], errors="coerce").astype("Int64")

cases_df["cases"] = pd.to_numeric(cases_df["cases"], errors="coerce")
temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")
hum_df["humidity"] = pd.to_numeric(hum_df["humidity"], errors="coerce")
rain_df["rainfall"] = pd.to_numeric(rain_df["rainfall"], errors="coerce")
idhm_df["idhm"] = pd.to_numeric(idhm_df["idhm"], errors="coerce")


# =========================================================
# Aggregate each input BEFORE merging
# =========================================================
cases_df = (
    cases_df.groupby(["municipio", "year", "week"], as_index=False)
    .agg({"cases": "sum"})
)

temp_df = (
    temp_df.groupby(["municipio", "year", "week"], as_index=False)
    .agg({"temperature": "mean"})
)

hum_df = (
    hum_df.groupby(["municipio", "year", "week"], as_index=False)
    .agg({"humidity": "mean"})
)

rain_df = (
    rain_df.groupby(["municipio", "year", "week"], as_index=False)
    .agg({"rainfall": "mean"})
)

idhm_df = (
    idhm_df.groupby(["municipio", "year", "week"], as_index=False)
    .agg({"idhm": "mean"})
)

print("cases_df columns:", cases_df.columns.tolist())
print("temp_df columns:", temp_df.columns.tolist())
print("hum_df columns:", hum_df.columns.tolist())
print("rain_df columns:", rain_df.columns.tolist())
print("idhm_df columns:", idhm_df.columns.tolist())


# =========================================================
# Merge main weekly data
# =========================================================
df = (
    cases_df
    .merge(temp_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
    .merge(hum_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
    .merge(rain_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
    .merge(idhm_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
)

print("Merged df columns:", df.columns.tolist())
print("Merged row count:", len(df))


# =========================================================
# Add IBGE code
# =========================================================
df = df.merge(
    lookup_df,
    on="municipio",
    how="left",
    validate="many_to_one",
)

missing_ibge = df[df["ibge_code"].isna()]["municipio"].drop_duplicates().sort_values()
print(f"Municipios with no IBGE match: {len(missing_ibge)}")
if len(missing_ibge) > 0:
    print(missing_ibge.tolist()[:50])

print("Valenca mapping:")
print(df[df["municipio"] == "valenca"][["municipio", "ibge_code"]].drop_duplicates())


# =========================================================
# Keep usable rows
# =========================================================
df = df.dropna(
    subset=["cases", "temperature", "humidity", "rainfall", "idhm", "ibge_code"]
).copy()
df["ibge_code"] = df["ibge_code"].astype(int)

print(f"Number of rows after dropping NaNs / missing IBGE: {len(df)}")


# =========================================================
# Add month from ISO week
# =========================================================
df["date"], df["month"] = iso_week_to_month(df["year"], df["week"])
df = df.dropna(subset=["month"]).copy()
df["month"] = df["month"].astype(int)


# =========================================================
# Only care about municipios in the main CSVs
# =========================================================
valid_ibge = set(df["ibge_code"].unique())


# =========================================================
# Prepare air travel features
# =========================================================
aero_df = aero_df.rename(columns={"ano": "year", "mes": "month"})

for col in [
    "year",
    "month",
    "co_muni_ori",
    "co_muni_des",
    "aero_pass",
    "aero_pass_week",
    "aero_conec",
]:
    aero_df[col] = pd.to_numeric(aero_df[col], errors="coerce")

aero_df = aero_df.dropna(subset=["year", "month", "co_muni_ori", "co_muni_des"]).copy()
aero_df["year"] = aero_df["year"].astype(int)
aero_df["month"] = aero_df["month"].astype(int)
aero_df["co_muni_ori"] = aero_df["co_muni_ori"].astype(int)
aero_df["co_muni_des"] = aero_df["co_muni_des"].astype(int)

aero_out = (
    aero_df.groupby(["co_muni_ori", "year", "month"], as_index=False)
    .agg({
        "aero_pass": "sum",
        "aero_pass_week": "sum",
        "aero_conec": "sum",
        "co_muni_des": "nunique",
    })
    .rename(columns={
        "co_muni_ori": "ibge_code",
        "aero_pass": "air_pass_out",
        "aero_pass_week": "air_pass_week_out",
        "aero_conec": "air_conec_out",
        "co_muni_des": "air_destinations_n",
    })
)

aero_in = (
    aero_df.groupby(["co_muni_des", "year", "month"], as_index=False)
    .agg({
        "aero_pass": "sum",
        "aero_pass_week": "sum",
        "aero_conec": "sum",
        "co_muni_ori": "nunique",
    })
    .rename(columns={
        "co_muni_des": "ibge_code",
        "aero_pass": "air_pass_in",
        "aero_pass_week": "air_pass_week_in",
        "aero_conec": "air_conec_in",
        "co_muni_ori": "air_origins_n",
    })
)

aero_features = aero_out.merge(
    aero_in,
    on=["ibge_code", "year", "month"],
    how="outer",
).fillna(0)

aero_features = aero_features[aero_features["ibge_code"].isin(valid_ibge)].copy()


# =========================================================
# Prepare road/fluvial features
# =========================================================
for col in [
    "co_muni_ori",
    "co_muni_des",
    "fluv_conec",
    "road_conec",
    "tot_conec",
    "irregular_conec",
]:
    fluvi_df[col] = pd.to_numeric(fluvi_df[col], errors="coerce")

fluvi_df = fluvi_df.dropna(subset=["co_muni_ori", "co_muni_des"]).copy()
fluvi_df["co_muni_ori"] = fluvi_df["co_muni_ori"].astype(int)
fluvi_df["co_muni_des"] = fluvi_df["co_muni_des"].astype(int)

fluvi_out = (
    fluvi_df.groupby("co_muni_ori", as_index=False)
    .agg({
        "fluv_conec": "sum",
        "road_conec": "sum",
        "tot_conec": "sum",
        "irregular_conec": "sum",
        "co_muni_des": "nunique",
    })
    .rename(columns={
        "co_muni_ori": "ibge_code",
        "fluv_conec": "fluv_conec_out",
        "road_conec": "road_conec_out",
        "tot_conec": "tot_conec_out",
        "irregular_conec": "irregular_conec_out",
        "co_muni_des": "network_destinations_n",
    })
)

fluvi_in = (
    fluvi_df.groupby("co_muni_des", as_index=False)
    .agg({
        "fluv_conec": "sum",
        "road_conec": "sum",
        "tot_conec": "sum",
        "irregular_conec": "sum",
        "co_muni_ori": "nunique",
    })
    .rename(columns={
        "co_muni_des": "ibge_code",
        "fluv_conec": "fluv_conec_in",
        "road_conec": "road_conec_in",
        "tot_conec": "tot_conec_in",
        "irregular_conec": "irregular_conec_in",
        "co_muni_ori": "network_origins_n",
    })
)

fluvi_features = fluvi_out.merge(
    fluvi_in,
    on="ibge_code",
    how="outer",
).fillna(0)

fluvi_features = fluvi_features[fluvi_features["ibge_code"].isin(valid_ibge)].copy()


# =========================================================
# Merge spatial features
# =========================================================
df = (
    df
    .merge(aero_features, on=["ibge_code", "year", "month"], how="left", validate="many_to_one")
    .merge(fluvi_features, on="ibge_code", how="left", validate="many_to_one")
)

spatial_cols = [
    "air_pass_out", "air_pass_week_out", "air_conec_out", "air_destinations_n",
    "air_pass_in", "air_pass_week_in", "air_conec_in", "air_origins_n",
    "fluv_conec_out", "road_conec_out", "tot_conec_out", "irregular_conec_out", "network_destinations_n",
    "fluv_conec_in", "road_conec_in", "tot_conec_in", "irregular_conec_in", "network_origins_n",
]

for col in spatial_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print("Final modeling row count:", len(df))


# =========================================================
# Encode municipios
# =========================================================
df["municipio_idx"], municipios = pd.factorize(df["municipio"], sort=True)
n_groups = df["municipio_idx"].nunique()
print(f"Number of municipios: {n_groups}")


# =========================================================
# Covariates
# =========================================================
covariates = FULL_COVARIATES if USE_FULL_COVARIATE_SET else BASE_COVARIATES
correlation_matrix = df[covariates].corr()
print(correlation_matrix)
print("Using covariates:", covariates)

missing_covs = [c for c in covariates if c not in df.columns]
if missing_covs:
    raise ValueError(f"Missing covariates in dataframe: {missing_covs}")

scaler = StandardScaler()
df[covariates] = scaler.fit_transform(df[covariates])


# =========================================================
# Arrays
# =========================================================
y = df["cases"].to_numpy(dtype=np.int64)
group_idx = df["municipio_idx"].to_numpy(dtype=np.int32)
X = df[covariates].to_numpy(dtype=np.float64)

print("Mean of cases:", float(df["cases"].mean()))
print("Variance of cases:", float(df["cases"].var()))


# =========================================================
# Model
# =========================================================
coords = {
    "municipio": municipios,
    "covariate": covariates,
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as model:
    # Data for covariates and group IDs
    X_data = pm.Data("X_data", X, dims=("obs_id", "covariate"))
    X_data.set_value(df[covariates])
    group_data = pm.Data("group_data", group_idx, dims="obs_id")
    import pymc as pm

with model:

    # Global and group-specific intercepts
    alpha_global = pm.Normal("alpha_global", mu=0.0, sigma=0.7)
    sigma_group = pm.HalfNormal("sigma_group", sigma=1)

    z_group_raw = pm.Normal("z_group_raw", mu=0.0, sigma=1.0, dims="municipio")
    z_group = pm.Deterministic(
        "z_group",
        z_group_raw - pm.math.mean(z_group_raw),
        dims="municipio",
    )

    alpha_group = pm.Deterministic(
        "alpha_group",
        alpha_global + sigma_group * z_group,
        dims="municipio",
    )

    # Coefficients for covariates
    beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims="covariate")

    # Linear predictor (eta)
    eta = alpha_group[group_data] + pm.math.dot(X_data, beta)
    mu = pm.math.exp(eta)

    # Over-dispersion parameter for Negative Binomial
    alpha_nb = pm.Exponential("alpha_nb", lam=1.0)

    # Observed data (Negative Binomial likelihood)
    y_obs = pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha_nb, observed=y, dims="obs_id")

    # Compute log-likelihood for Negative Binomial


    # Add log-likelihood to the trace
    trace = pm.sample(
        draws=1000,
        tune=2000,
        chains=2,
        cores=2,
        target_accept=0.95,
        init="adapt_diag",
        random_seed=42,
        return_inferencedata=True,
        progressbar=True,
        idata_kwargs={"log_likelihood": True},
    )
    posterior_predictive = pm.sample_posterior_predictive(
        trace,
        var_names=["y_obs"], 
        random_seed=42
    )


    # Add log-likelihood to posterior for WAIC/BIC
# =========================================================
# Diagnostics
# =========================================================
summary_main = az.summary(
    trace,
    var_names=["alpha_global", "sigma_group", "alpha_nb", "beta"],
    round_to=4,
)
print(summary_main)

summary_all = az.summary(trace, round_to=4)

# Assign weight to each parameter based on its importance
# Function to calculate weighted averages
def weighted_average(values, weights):
    """Calculate the weighted average for a list of values."""
    return np.average(values, weights=weights)

# Get the Rhat values for each parameter
rhat = az.rhat(trace)

# Get ESS values (bulk and tail)
ess_bulk = az.ess(trace, method="bulk")
ess_tail = az.ess(trace, method="tail")

# Calculate the weighted averages for Rhat, ESS bulk, and ESS tail
rhat_values = [rhat["beta"].mean(), rhat["alpha_group"].mean(), rhat["sigma_group"].mean(), rhat["z_group"].mean()]
ess_bulk_values = [ess_bulk["beta"].mean(), ess_bulk["alpha_group"].mean(), ess_bulk["sigma_group"].mean(), ess_bulk["z_group"].mean()]
ess_tail_values = [ess_tail["beta"].mean(), ess_tail["alpha_group"].mean(), ess_tail["sigma_group"].mean(), ess_tail["z_group"].mean()]

# Define the weights for each parameter
weights = {
    "beta": 2,          # Covariates (temperature, rainfall, idhm, etc.)
    "alpha_group": 1,   # Global intercept (base level disease count)
    "sigma_group": 1,   # Variance of municipio intercepts
    "z_group": 3,       # Municipio-specific intercepts (most important)
}

# Compute weighted averages
weighted_rhat = weighted_average(rhat_values, weights=list(weights.values()))
weighted_ess_bulk = weighted_average(ess_bulk_values, weights=list(weights.values()))
weighted_ess_tail = weighted_average(ess_tail_values, weights=list(weights.values()))

# Print the results
print(f"Weighted Rhat: {weighted_rhat}")
print(f"Weighted ESS Bulk: {weighted_ess_bulk}")
print(f"Weighted ESS Tail: {weighted_ess_tail}")
import arviz as az
import numpy as np

# =============================
# WAIC and LOO
# =============================

# Compute WAIC
waic_result = az.waic(trace)
print("WAIC Result:")
print(waic_result)
print(f"WAIC (elpd_waic): {waic_result.elpd_waic:.2f}, p_WAIC (effective params): {waic_result.p_waic:.2f}")

# Compute LOO
loo_result = az.loo(trace)
print("\nLOO Result:")
print(loo_result)
print(f"LOO (elpd_loo): {loo_result.elpd_loo:.2f}, p_LOO (effective params): {loo_result.p_loo:.2f}")
# Extract Pareto k values
k_vals = loo_result.pareto_k

# Find "problematic" points (k > 0.7)
high_k_idx = np.where(k_vals > 0.7)[0]

# Optional: compare WAIC and LOO across multiple models later
# lower elpd_waic or elpd_loo indicates worse predictive fit


# Delta ELPD
delta_elpd = loo_result.elpd_loo
print(f"ΔELPD (humidity): {delta_elpd:.2f}")
beta_means = summary_main["mean"]

percent_effect = np.exp(beta_means) - 1

for i, val in enumerate(percent_effect):
    print(f"Variable {i}: {val*100:.2f}% change in cases")

# -------------------------
# 1️⃣ Posterior coefficients
# -------------------------
posterior_means = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
posterior_stds = trace.posterior["beta"].std(dim=["chain", "draw"]).values
num_vars = len(posterior_means)
ppc_mean = posterior_predictive.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values.flatten()
ppc_std  = posterior_predictive.posterior_predictive["y_obs"].std(dim=["chain", "draw"]).values.flatten()
plt.figure(figsize=(14, 6))

# Subplot 1: Posterior coefficients
plt.subplot(1, 2, 1)
plt.errorbar(range(num_vars), posterior_means, yerr=posterior_stds, fmt='o', capsize=5)
plt.xticks(range(num_vars), [f"Var {i}" for i in range(num_vars)], rotation=45)
plt.axhline(0, color='red', linestyle='--', label='No effect')
plt.xlabel("Covariates")
plt.ylabel("Posterior mean coefficient ± SD")
plt.title("Posterior Coefficients of Covariates")
plt.legend()

# -------------------------
# 2️⃣ Predicted vs Actual
# -------------------------
y_actual = y  # your observed data

plt.subplot(1, 2, 2)
plt.errorbar(y_actual, ppc_mean, yerr=ppc_std, fmt='o', alpha=0.3, capsize=2)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label="y = y_pred")
plt.xlabel("Actual cases")
plt.ylabel("Predicted cases")
plt.title("Predicted vs Actual Cases")
plt.legend()

plt.tight_layout()
plt.show()

# Posterior mean of group intercepts
alpha_group_mean = trace.posterior["alpha_group"].mean(dim=["chain", "draw"]).values  # shape: (num_groups,)

# Posterior mean contribution of covariates
beta_mean = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
X_array = X_data.get_value()
eta_covariate = X_array @ beta_mean

# Print ranges to compare
print("Alpha_group range:", alpha_group_mean.min(), "to", alpha_group_mean.max())
print("Covariate contribution (X_data @ beta) range:", eta_covariate.min(), "to", eta_covariate.max())
print("Mean alpha_group:", alpha_group_mean.mean())
print("Mean covariate contribution:", eta_covariate.mean())


if MAKE_PLOTS:
    az.plot_trace(trace, var_names=["alpha_global", "sigma_group", "alpha_nb", "beta"])
    az.plot_posterior(trace, var_names=["alpha_global", "sigma_group", "alpha_nb", "beta"])