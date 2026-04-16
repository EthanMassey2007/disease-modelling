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
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def normalize_municipio_name(name):
    if pd.isna(name):
        return np.nan

    name = str(name).strip().lower()
    name = re.sub(r"/[a-z]{2}$", "", name)

    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    name = re.sub(r"\s+", " ", name).strip()
    return name


def iso_week_to_date(year_series, week_series):
    dt = pd.to_datetime(
        year_series.astype(str)
        + "-W"
        + week_series.astype(int).astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    return dt


def weighted_average(values, weights):
    return np.average(values, weights=weights)


def da_mean(x):
    return float(np.asarray(x).mean())


# =========================================================
# Config
# =========================================================
TARGET_STATE = "RJ"
START_YEAR = 2017
END_YEAR = 2022

MAKE_PLOTS = True
APPLY_LOG1P_TO_SKEWED_FEATURES = True

SKEWED_FEATURES = [
    "air_pass_in",
    "road_conec_in",
    "fluv_conec_in",
]

BASE_COVARIATES = [
    "rainfall",
    "humidity",
    "temperature",
    "idhm",
    "air_pass_in",
    "log_cases_lag1",
]

FULL_COVARIATES = [
    "rainfall",
    "humidity",
    "temperature",
    "idhm",
    "air_pass_in",
    "road_conec_in",
    "fluv_conec_in",
    "log_cases_lag1",
]

USE_FULL_COVARIATE_SET = True


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
# Normalize municipio names
# =========================================================
for d in [cases_df, temp_df, hum_df, rain_df, idhm_df]:
    d["municipio"] = d["municipio"].apply(normalize_municipio_name)


# =========================================================
# Build RJ-only city -> IBGE lookup
# =========================================================
municipios_df["state_code"] = municipios_df["city"].astype(str).str.extract(
    r"/([A-Z]{2})$",
    expand=False,
)
municipios_df["city_clean"] = municipios_df["city"].apply(normalize_municipio_name)
municipios_df["ibgeid"] = pd.to_numeric(municipios_df["ibgeid"], errors="coerce").astype("Int64")

municipios_df = municipios_df[municipios_df["state_code"] == TARGET_STATE].copy()

lookup_df = (
    municipios_df[["city_clean", "ibgeid"]]
    .dropna()
    .drop_duplicates()
    .rename(columns={"city_clean": "municipio", "ibgeid": "ibge_code"})
)

print("RJ municipios in lookup:", len(lookup_df))


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
# Restrict weekly core datasets to 2017-2022
# =========================================================
def restrict_years(d):
    d = d.dropna(subset=["year", "week"]).copy()
    d = d[(d["year"] >= START_YEAR) & (d["year"] <= END_YEAR)].copy()
    return d

cases_df = restrict_years(cases_df)
temp_df = restrict_years(temp_df)
hum_df = restrict_years(hum_df)
rain_df = restrict_years(rain_df)
idhm_df = restrict_years(idhm_df)

print("Core years after restriction:")
print("cases:", sorted(cases_df["year"].dropna().astype(int).unique().tolist()))
print("temp:", sorted(temp_df["year"].dropna().astype(int).unique().tolist()))
print("humidity:", sorted(hum_df["year"].dropna().astype(int).unique().tolist()))
print("rain:", sorted(rain_df["year"].dropna().astype(int).unique().tolist()))
print("idhm:", sorted(idhm_df["year"].dropna().astype(int).unique().tolist()))


# =========================================================
# Aggregate BEFORE merging
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


# =========================================================
# Merge weekly core data
# =========================================================
df = (
    cases_df
    .merge(temp_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
    .merge(hum_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
    .merge(rain_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
    .merge(idhm_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
)

print("Merged weekly row count:", len(df))


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
print("Municipios with no IBGE match:", len(missing_ibge))
if len(missing_ibge) > 0:
    print(missing_ibge.tolist()[:50])

df = df.dropna(
    subset=["cases", "temperature", "humidity", "rainfall", "idhm", "ibge_code"]
).copy()

df["ibge_code"] = df["ibge_code"].astype(int)


# =========================================================
# Add date + month
# =========================================================
df["date"] = iso_week_to_date(df["year"], df["week"])
df = df.dropna(subset=["date"]).copy()
df["month"] = df["date"].dt.month.astype(int)

print("Weekly years used:", sorted(df["year"].dropna().astype(int).unique().tolist()))


# =========================================================
# Valid IBGE codes from weekly data
# =========================================================
valid_ibge = set(df["ibge_code"].unique())


# =========================================================
# Prepare air travel features (time-varying)
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

aero_df = aero_df[
    (aero_df["year"] >= START_YEAR) & (aero_df["year"] <= END_YEAR)
].copy()

print("Air years used:", sorted(aero_df["year"].unique().tolist()))

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
# Prepare road/fluvial features (STATIC)
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

print("Row count after spatial merges:", len(df))


# =========================================================
# Add lagged cases
# =========================================================
df = df.sort_values(["municipio", "date"]).copy()

df["cases_lag1"] = df.groupby("municipio")["cases"].shift(1)
df["log_cases_lag1"] = np.log1p(df["cases_lag1"])

# Drop first observed week per municipio
df = df.dropna(subset=["cases_lag1"]).copy()

print("Row count after lag creation:", len(df))


# =========================================================
# Encode categorical indices
# =========================================================
df["municipio_idx"], municipios = pd.factorize(df["municipio"], sort=True)
n_groups = len(municipios)

df["week_of_year"] = df["week"].astype(int)
week_levels = sorted(df["week_of_year"].unique().tolist())
week_to_idx = {w: i for i, w in enumerate(week_levels)}
df["week_idx"] = df["week_of_year"].map(week_to_idx).astype(int)

year_levels = sorted(df["year"].astype(int).unique().tolist())
year_to_idx = {y: i for i, y in enumerate(year_levels)}
df["year_idx"] = df["year"].astype(int).map(year_to_idx).astype(int)

n_weeks = len(week_levels)
n_years = len(year_levels)

print("Municipios:", n_groups)
print("Week levels:", n_weeks)
print("Year levels:", year_levels)

if n_groups < 2:
    raise ValueError("Too few municipios after preprocessing.")


# =========================================================
# Covariates
# =========================================================
covariates = FULL_COVARIATES if USE_FULL_COVARIATE_SET else BASE_COVARIATES

missing_covs = [c for c in covariates if c not in df.columns]
if missing_covs:
    raise ValueError(f"Missing covariates: {missing_covs}")

if APPLY_LOG1P_TO_SKEWED_FEATURES:
    for col in SKEWED_FEATURES:
        if col in df.columns:
            df[col] = np.log1p(df[col])

correlation_matrix = df[covariates].corr()
print("Covariate correlation matrix:")
print(correlation_matrix)

scaler = StandardScaler()
df[covariates] = scaler.fit_transform(df[covariates])


# =========================================================
# Arrays
# =========================================================
y = df["cases"].to_numpy(dtype=np.int64)
X = df[covariates].to_numpy(dtype=np.float64)
group_idx = df["municipio_idx"].to_numpy(dtype=np.int32)
week_idx = df["week_idx"].to_numpy(dtype=np.int32)
year_idx = df["year_idx"].to_numpy(dtype=np.int32)

print("Mean of cases:", float(df["cases"].mean()))
print("Variance of cases:", float(df["cases"].var()))
print("Min/Max of cases:", int(df["cases"].min()), int(df["cases"].max()))


# =========================================================
# Model
# =========================================================
coords = {
    "municipio": municipios,
    "covariate": covariates,
    "week_level": np.array(week_levels),
    "year_level": np.array(year_levels),
    "obs_id": np.arange(len(df)),
}

with pm.Model(coords=coords) as model:
    X_data = pm.Data("X_data", X, dims=("obs_id", "covariate"))
    group_data = pm.Data("group_data", group_idx, dims="obs_id")
    week_data = pm.Data("week_data", week_idx, dims="obs_id")
    year_data = pm.Data("year_data", year_idx, dims="obs_id")

    # Municipality intercepts
    alpha_global = pm.Normal(
        "alpha_global",
        mu=np.log(np.maximum(y.mean(), 1.0)),
        sigma=3.0,
    )

    sigma_group = pm.HalfNormal("sigma_group", sigma=3.0)

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

    # Week-of-year effect
    sigma_week = pm.HalfNormal("sigma_week", sigma=1.0)
    week_raw = pm.Normal("week_raw", mu=0.0, sigma=1.0, dims="week_level")
    week_effect = pm.Deterministic(
        "week_effect",
        sigma_week * (week_raw - pm.math.mean(week_raw)),
        dims="week_level",
    )

    # Year effect
    sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
    year_raw = pm.Normal("year_raw", mu=0.0, sigma=1.0, dims="year_level")
    year_effect = pm.Deterministic(
        "year_effect",
        sigma_year * (year_raw - pm.math.mean(year_raw)),
        dims="year_level",
    )

    # Covariate coefficients
    beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims="covariate")

    eta = (
        alpha_group[group_data]
        + week_effect[week_data]
        + year_effect[year_data]
        + pm.math.dot(X_data, beta)
    )

    mu = pm.Deterministic("mu", pm.math.exp(eta), dims="obs_id")

    # Overdispersion
    alpha_nb = pm.Exponential("alpha_nb", lam=1.0)

    # Zero-inflation
    zi_intercept = pm.Normal("zi_intercept", mu=0.0, sigma=1.5)
    psi = pm.Deterministic(
        "psi",
        pm.math.sigmoid(zi_intercept),
        dims=(),
    )

    pm.ZeroInflatedNegativeBinomial(
        "y_obs",
        psi=psi,
        mu=mu,
        alpha=alpha_nb,
        observed=y,
        dims="obs_id",
    )

    trace = pm.sample(
        draws=400,
        tune=600,
        chains=4,
        cores=4,
        target_accept=0.98,
        init="jitter+adapt_diag",
        random_seed=42,
        return_inferencedata=True,
        progressbar=True,
        idata_kwargs={"log_likelihood": True},
    )

    posterior_predictive = pm.sample_posterior_predictive(
        trace,
        var_names=["mu", "y_obs"],
        random_seed=42,
        progressbar=True,
    )


# =========================================================
# Diagnostics
# =========================================================
summary_main = az.summary(
    trace,
    var_names=[
        "alpha_global",
        "sigma_group",
        "sigma_week",
        "sigma_year",
        "alpha_nb",
        "zi_intercept",
        "beta",
    ],
    round_to=4,
)
print(summary_main)

rhat = az.rhat(trace)
ess_bulk = az.ess(trace, method="bulk")
ess_tail = az.ess(trace, method="tail")

rhat_values = [
    da_mean(rhat["beta"]),
    da_mean(rhat["alpha_group"]),
    da_mean(rhat["sigma_group"]),
]

ess_bulk_values = [
    da_mean(ess_bulk["beta"]),
    da_mean(ess_bulk["alpha_group"]),
    da_mean(ess_bulk["sigma_group"]),
]

ess_tail_values = [
    da_mean(ess_tail["beta"]),
    da_mean(ess_tail["alpha_group"]),
    da_mean(ess_tail["sigma_group"]),
]

weights = [2, 2, 1]

print(f"Weighted Rhat: {weighted_average(rhat_values, weights):.4f}")
print(f"Weighted ESS Bulk: {weighted_average(ess_bulk_values, weights):.2f}")
print(f"Weighted ESS Tail: {weighted_average(ess_tail_values, weights):.2f}")


# =========================================================
# WAIC / LOO
# =========================================================
waic_result = az.waic(trace)
loo_result = az.loo(trace)

print("WAIC:")
print(waic_result)
print("LOO:")
print(loo_result)


# =========================================================
# Fitted values
# =========================================================
fitted_mu_mean = (
    posterior_predictive.posterior_predictive["mu"]
    .mean(dim=["chain", "draw"])
    .values
)

fitted_mu_std = (
    posterior_predictive.posterior_predictive["mu"]
    .std(dim=["chain", "draw"])
    .values
)

ppc_y_mean = (
    posterior_predictive.posterior_predictive["y_obs"]
    .mean(dim=["chain", "draw"])
    .values
)

ppc_y_std = (
    posterior_predictive.posterior_predictive["y_obs"]
    .std(dim=["chain", "draw"])
    .values
)

print("Observed range:", int(y.min()), "to", int(y.max()))
print("Fitted mu range:", float(fitted_mu_mean.min()), "to", float(fitted_mu_mean.max()))
print("Predicted y range:", float(ppc_y_mean.min()), "to", float(ppc_y_mean.max()))
print("Std of fitted mu:", float(np.std(fitted_mu_mean)))


# =========================================================
# Effect summaries
# =========================================================
beta_summary = az.summary(trace, var_names=["beta"], round_to=4)
beta_means = beta_summary["mean"].to_numpy()
percent_effect = np.exp(beta_means) - 1

print("\nApprox multiplicative effect of +1 SD in each covariate:")
for name, val in zip(covariates, percent_effect):
    print(f"{name}: {val * 100:.2f}% change in expected cases")


# =========================================================
# Prediction helper
# For new rows, supply:
# municipio, year, week, and all covariates including log_cases_lag1
# =========================================================
posterior_samples = trace.posterior.stack(sample=("chain", "draw"))

beta_draws = posterior_samples["beta"].values
alpha_global_draws = posterior_samples["alpha_global"].values
alpha_group_draws = posterior_samples["alpha_group"].values
week_effect_draws = posterior_samples["week_effect"].values
year_effect_draws = posterior_samples["year_effect"].values

municipio_to_idx = {m: i for i, m in enumerate(municipios)}


def predict_expected_cases(new_df: pd.DataFrame) -> np.ndarray:
    new_df = new_df.copy()

    required_cols = ["municipio", "year", "week"] + covariates
    missing = [c for c in required_cols if c not in new_df.columns]
    if missing:
        raise ValueError(f"Missing columns in new_df: {missing}")

    new_df["municipio"] = new_df["municipio"].apply(normalize_municipio_name)

    if APPLY_LOG1P_TO_SKEWED_FEATURES:
        for col in SKEWED_FEATURES:
            if col in new_df.columns:
                new_df[col] = np.log1p(new_df[col])

    X_new = scaler.transform(new_df[covariates])

    n_new = len(new_df)
    n_draws = alpha_global_draws.shape[0]

    intercept_draws = np.zeros((n_new, n_draws))
    week_draws = np.zeros((n_new, n_draws))
    year_draws = np.zeros((n_new, n_draws))

    for i, row in new_df.iterrows():
        muni = row["municipio"]
        year_val = int(row["year"])
        week_val = int(row["week"])

        if muni in municipio_to_idx:
            intercept_draws[i, :] = alpha_group_draws[municipio_to_idx[muni], :]
        else:
            intercept_draws[i, :] = alpha_global_draws

        if week_val in week_to_idx:
            week_draws[i, :] = week_effect_draws[week_to_idx[week_val], :]
        else:
            week_draws[i, :] = 0.0

        if year_val in year_to_idx:
            year_draws[i, :] = year_effect_draws[year_to_idx[year_val], :]
        else:
            year_draws[i, :] = 0.0

    eta_new = intercept_draws + week_draws + year_draws + X_new @ beta_draws
    mu_new = np.exp(eta_new)
    return mu_new.mean(axis=1)


# =========================================================
# Quick sanity check
# =========================================================
sample_cols = ["municipio", "year", "week"] + covariates + ["cases"]
example_rows = df[sample_cols].sample(min(10, len(df)), random_state=42).copy()
example_pred = predict_expected_cases(example_rows[["municipio", "year", "week"] + covariates])

print("\nExample predictions:")
for muni, actual, pred in zip(example_rows["municipio"], example_rows["cases"], example_pred):
    print(f"{muni}: actual={actual}, predicted_mean={pred:.2f}")


# =========================================================
# Plots
# =========================================================
if MAKE_PLOTS:
    posterior_means = trace.posterior["beta"].mean(dim=["chain", "draw"]).values
    posterior_stds = trace.posterior["beta"].std(dim=["chain", "draw"]).values

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.errorbar(
        range(len(covariates)),
        posterior_means,
        yerr=posterior_stds,
        fmt="o",
        capsize=5,
    )
    plt.xticks(range(len(covariates)), covariates, rotation=45, ha="right")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Covariates")
    plt.ylabel("Posterior mean coefficient ± SD")
    plt.title("Posterior Coefficients")

    plt.subplot(1, 2, 2)
    plt.errorbar(
        y,
        fitted_mu_mean,
        yerr=fitted_mu_std,
        fmt="o",
        alpha=0.2,
        capsize=2,
    )
    line_min = min(float(y.min()), float(fitted_mu_mean.min()))
    line_max = max(float(y.max()), float(fitted_mu_mean.max()))
    plt.plot([line_min, line_max], [line_min, line_max], "r--")
    plt.xlabel("Actual cases")
    plt.ylabel("Fitted expected cases")
    plt.title("Fitted Expected Cases vs Actual Cases")

    plt.tight_layout()
    plt.show()

    az.plot_trace(
        trace,
        var_names=[
            "alpha_global",
            "sigma_group",
            "sigma_week",
            "sigma_year",
            "alpha_nb",
            "zi_intercept",
            "beta",
        ],
    )
    plt.tight_layout()
    plt.show()