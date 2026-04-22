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

def to_entity_sample_array(x, entity_size):
   """
   Convert posterior object to numpy array with shape:
   (entity_size, n_samples) if entity dimension exists
   or (n_samples,) for scalar arrays.
   """
   arr = np.asarray(x)


   if arr.ndim == 1:
       return arr


   if arr.shape[0] == entity_size:
       return arr


   if arr.shape[-1] == entity_size:
       return np.moveaxis(arr, -1, 0)


   raise ValueError(
       f"Could not align array with entity_size={entity_size}. "
       f"Observed shape: {arr.shape}"
   )

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
   y_true = np.asarray(y_true, dtype=float)
   y_pred = np.asarray(y_pred, dtype=float)


   mae = float(np.mean(np.abs(y_true - y_pred)))
   rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

   total_actual = float(np.sum(np.abs(y_true)))
   wape = float(np.sum(np.abs(y_true - y_pred)) / max(total_actual, 1e-9))
   accuracy_pct = max(0.0, 100.0 * (1.0 - wape))

   sst = float(np.sum((y_true - y_true.mean()) ** 2))
   sse = float(np.sum((y_true - y_pred) ** 2))
   r2 = float(1.0 - sse / sst) if sst > 0 else np.nan

   return {
       "mae": mae,
       "rmse": rmse,
       "wape": wape,
       "accuracy_pct": accuracy_pct,
       "r2": r2,
   }

# =========================================================
# Config
# =========================================================
TARGET_STATE = "RJ"
START_YEAR = 2017
END_YEAR = 2022

START_LAG = 2
END_LAG = 2
LAG_VALUES = list(range(START_LAG, END_LAG + 1))

MAKE_PLOTS = True
APPLY_LOG1P_TO_SKEWED_FEATURES = True
USE_FULL_COVARIATE_SET = True
SAVE_RESULTS_CSV = True

# This accuracy is based on WAPE:
# accuracy_pct = 100 * (1 - sum(|actual - predicted|) / sum(actual))
# Higher is better. 100% is perfect.
ACCURACY_LABEL = "WAPE-based accuracy (%)"

# Keep the same model settings as your original script.
# If the lag sweep is too slow, reduce these.
DRAWS = 1000
TUNE = 2000
CHAINS = 4
CORES = 4
TARGET_ACCEPT = 0.98
RANDOM_SEED = 42

SKEWED_FEATURES = [
   "air_pass_in",
   "road_conec_in",
   "fluv_conec_in",
]

BASE_COVARIATES_NO_LAG = [
   "rainfall",
   "humidity",
   "temperature",
   "idhm",
   "air_pass_in",
]

FULL_COVARIATES_NO_LAG = [
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
# Shared preprocessing (done once)
# =========================================================
def restrict_years(d: pd.DataFrame) -> pd.DataFrame:
   d = d.dropna(subset=["year", "week"]).copy()
   d = d[(d["year"] >= START_YEAR) & (d["year"] <= END_YEAR)].copy()
   return d

def build_base_dataframe() -> pd.DataFrame:
   # -----------------------------------------------------
   # Load data
   # -----------------------------------------------------
   cases_df = clean_columns(pd.read_csv(cases_file))
   temp_df = clean_columns(pd.read_csv(temperature_file))
   hum_df = clean_columns(pd.read_csv(humidity_file))
   rain_df = clean_columns(pd.read_csv(rainfall_file))
   idhm_df = clean_columns(pd.read_csv(idhm_file))

   municipios_df = clean_columns(pd.read_csv(municipios_file))
   aero_df = clean_columns(pd.read_parquet(aero_file))
   fluvi_df = clean_columns(pd.read_parquet(fluvi_file))

   # -----------------------------------------------------
   # Normalize municipio names
   # -----------------------------------------------------
   for d in [cases_df, temp_df, hum_df, rain_df, idhm_df]:
       d["municipio"] = d["municipio"].apply(normalize_municipio_name)

   # -----------------------------------------------------
   # Build RJ-only city -> IBGE lookup
   # -----------------------------------------------------
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

   dup_counts = lookup_df.groupby("municipio")["ibge_code"].nunique()
   ambiguous_after_filter = dup_counts[dup_counts > 1]

   print("RJ municipios in lookup:", len(lookup_df))
   print("Ambiguous names after state filter:")
   print(ambiguous_after_filter)

   # -----------------------------------------------------
   # Numeric conversion
   # -----------------------------------------------------
   for d in [cases_df, temp_df, hum_df, rain_df, idhm_df]:
       d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
       d["week"] = pd.to_numeric(d["week"], errors="coerce").astype("Int64")

   cases_df["cases"] = pd.to_numeric(cases_df["cases"], errors="coerce")
   temp_df["temperature"] = pd.to_numeric(temp_df["temperature"], errors="coerce")
   hum_df["humidity"] = pd.to_numeric(hum_df["humidity"], errors="coerce")
   rain_df["rainfall"] = pd.to_numeric(rain_df["rainfall"], errors="coerce")
   idhm_df["idhm"] = pd.to_numeric(idhm_df["idhm"], errors="coerce")

   # -----------------------------------------------------
   # Restrict weekly core datasets to 2017-2022
   # -----------------------------------------------------
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

   # -----------------------------------------------------
   # Aggregate BEFORE merging
   # -----------------------------------------------------
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

   # -----------------------------------------------------
   # Merge weekly core data
   # -----------------------------------------------------
   df = (
       cases_df
       .merge(temp_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
       .merge(hum_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
       .merge(rain_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
       .merge(idhm_df, on=["municipio", "year", "week"], how="left", validate="one_to_one")
   )

   print("Merged weekly row count:", len(df))

   # -----------------------------------------------------
   # Add IBGE code
   # -----------------------------------------------------
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


   # -----------------------------------------------------
   # Add date + month
   # -----------------------------------------------------
   df["date"] = iso_week_to_date(df["year"], df["week"])
   df = df.dropna(subset=["date"]).copy()
   df["month"] = df["date"].dt.month.astype(int)


   print("Weekly years used:", sorted(df["year"].dropna().astype(int).unique().tolist()))


   # -----------------------------------------------------
   # Valid IBGE codes from weekly data
   # -----------------------------------------------------
   valid_ibge = set(df["ibge_code"].unique())


   # -----------------------------------------------------
   # Prepare air travel features (time-varying)
   # -----------------------------------------------------
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


   # -----------------------------------------------------
   # Prepare road/fluvial features (STATIC)
   # -----------------------------------------------------
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


   # -----------------------------------------------------
   # Merge spatial features
   # -----------------------------------------------------
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


   df = df.sort_values(["municipio", "date"]).reset_index(drop=True)
   return df




# =========================================================
# Lag-specific prep
# =========================================================
def prepare_lagged_dataframe(df_base: pd.DataFrame, lag_weeks: int):
   df = df_base.copy()


   lag_cases_col = f"cases_lag{lag_weeks}"
   lag_log_col = f"log_cases_lag{lag_weeks}"


   df[lag_cases_col] = df.groupby("municipio")["cases"].shift(lag_weeks)
   df[lag_log_col] = np.log1p(df[lag_cases_col])
   df = df.dropna(subset=[lag_cases_col]).copy()


   print(f"Using case lag of {lag_weeks} week(s)")
   print("Row count after lag creation:", len(df))


   # -----------------------------------------------------
   # Encode categorical indices
   # -----------------------------------------------------
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


   # -----------------------------------------------------
   # Covariates
   # -----------------------------------------------------
   covariates_no_lag = FULL_COVARIATES_NO_LAG if USE_FULL_COVARIATE_SET else BASE_COVARIATES_NO_LAG
   covariates = covariates_no_lag + [lag_log_col]


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


   # -----------------------------------------------------
   # Arrays
   # -----------------------------------------------------
   y = df["cases"].to_numpy(dtype=np.int64)
   X = df[covariates].to_numpy(dtype=np.float64)
   group_idx = df["municipio_idx"].to_numpy(dtype=np.int32)
   week_idx = df["week_idx"].to_numpy(dtype=np.int32)
   year_idx = df["year_idx"].to_numpy(dtype=np.int32)


   print("Mean of cases:", float(df["cases"].mean()))
   print("Variance of cases:", float(df["cases"].var()))
   print("Min/Max of cases:", int(df["cases"].min()), int(df["cases"].max()))


   return {
       "df": df,
       "y": y,
       "X": X,
       "group_idx": group_idx,
       "week_idx": week_idx,
       "year_idx": year_idx,
       "covariates": covariates,
       "lag_log_col": lag_log_col,
       "municipios": municipios,
       "week_levels": week_levels,
       "year_levels": year_levels,
       "scaler": scaler,
   }




# =========================================================
# Model fit for one lag
# =========================================================
def fit_one_lag(df_base: pd.DataFrame, lag_weeks: int):
   prepared = prepare_lagged_dataframe(df_base, lag_weeks)


   df = prepared["df"]
   y = prepared["y"]
   X = prepared["X"]
   group_idx = prepared["group_idx"]
   week_idx = prepared["week_idx"]
   year_idx = prepared["year_idx"]
   covariates = prepared["covariates"]
   lag_log_col = prepared["lag_log_col"]
   municipios = prepared["municipios"]
   week_levels = prepared["week_levels"]
   year_levels = prepared["year_levels"]


   coords = {
       "municipio": municipios,
       "covariate": covariates,
       "week_level": np.array(week_levels),
       "year_level": np.array(year_levels),
       "obs_id": np.arange(len(df)),
   }


   lag_col_idx = covariates.index(lag_log_col)


   with pm.Model(coords=coords) as model:
       X_data = pm.Data("X_data", X, dims=("obs_id", "covariate"))
       group_data = pm.Data("group_data", group_idx, dims="obs_id")
       week_data = pm.Data("week_data", week_idx, dims="obs_id")
       year_data = pm.Data("year_data", year_idx, dims="obs_id")


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


       sigma_week = pm.HalfNormal("sigma_week", sigma=1.0)
       week_raw = pm.Normal("week_raw", mu=0.0, sigma=1.0, dims="week_level")
       week_effect = pm.Deterministic(
           "week_effect",
           sigma_week * (week_raw - pm.math.mean(week_raw)),
           dims="week_level",
       )


       sigma_year = pm.HalfNormal("sigma_year", sigma=1.0)
       year_raw = pm.Normal("year_raw", mu=0.0, sigma=1.0, dims="year_level")
       year_effect = pm.Deterministic(
           "year_effect",
           sigma_year * (year_raw - pm.math.mean(year_raw)),
           dims="year_level",
       )


       beta = pm.Normal("beta", mu=0.0, sigma=1.5, dims="covariate")


       eta = (
           alpha_group[group_data]
           + week_effect[week_data]
           + year_effect[year_data]
           + pm.math.dot(X_data, beta)
       )


       mu = pm.Deterministic("mu", pm.math.exp(eta), dims="obs_id")


       alpha_nb = pm.Exponential("alpha_nb", lam=1.0)


       # Reversed sign on lag impact
       zi_intercept = pm.Normal("zi_intercept", mu=-1.5, sigma=1.0)
       zi_beta_lag = pm.Normal("zi_beta_lag", mu=1.0, sigma=0.75)  # Positive impact

       logit_psi = zi_intercept + zi_beta_lag * X_data[:, lag_col_idx]  # Positive effect of lag
       psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi), dims="obs_id")


       pm.ZeroInflatedNegativeBinomial(
           "y_obs",
           psi=psi,
           mu=mu,
           alpha=alpha_nb,
           observed=y,
           dims="obs_id",
       )


       trace = pm.sample(
           draws=DRAWS,
           tune=TUNE,
           chains=CHAINS,
           cores=CORES,
           target_accept=TARGET_ACCEPT,
           init="jitter+adapt_diag",
           random_seed=RANDOM_SEED,
           return_inferencedata=True,
           progressbar=True,
           idata_kwargs={"log_likelihood": True},
       )


       posterior_predictive = pm.sample_posterior_predictive(
           trace,
           var_names=["y_obs"],
           random_seed=RANDOM_SEED,
           progressbar=True,
       )


   fitted_mu_mean = posterior_predictive.posterior_predictive["y_obs"].mean(dim=["chain", "draw"]).values


   fitted_mu_std = posterior_predictive.posterior_predictive["y_obs"].std(dim=["chain", "draw"]).values


   metrics = compute_metrics(y, fitted_mu_mean)


   print(f"Lag {lag_weeks} metrics:")
   print(
       f"  accuracy_pct={metrics['accuracy_pct']:.2f}, "
       f"rmse={metrics['rmse']:.4f}, "
       f"mae={metrics['mae']:.4f}, "
       f"r2={metrics['r2']:.4f}, "
       f"wape={metrics['wape']:.4f}"
   )


   summary_main = az.summary(
       trace,
       var_names=[
           "alpha_global",
           "sigma_group",
           "sigma_week",
           "sigma_year",
           "alpha_nb",
           "zi_intercept",
           "zi_beta_lag",
           "beta",
       ],
       round_to=4,
   )
   print(summary_main)


   rhat = az.rhat(trace)
   ess_bulk = az.ess(trace, method="bulk")
   ess_tail = az.ess(trace, method="tail")

   summary = az.summary(trace)
   print(summary[["r_hat", "ess_bulk", "ess_tail"]])


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


   diag_summary = {
       "weighted_rhat": weighted_average(rhat_values, weights),
       "weighted_ess_bulk": weighted_average(ess_bulk_values, weights),
       "weighted_ess_tail": weighted_average(ess_tail_values, weights),
   }


   print(f"Weighted Rhat: {diag_summary['weighted_rhat']:.4f}")
   print(f"Weighted ESS Bulk: {diag_summary['weighted_ess_bulk']:.2f}")
   print(f"Weighted ESS Tail: {diag_summary['weighted_ess_tail']:.2f}")


   return {
       "lag": lag_weeks,
       "df": df,
       "trace": trace,
       "fitted_mu_mean": fitted_mu_mean,
       "fitted_mu_std": fitted_mu_std,
       "metrics": metrics,
       "diag_summary": diag_summary,
   }




# =========================================================
# Main loop over lags
# =========================================================
def main():
   df_base = build_base_dataframe()


   lag_results = []
   best_result = None


   for lag in LAG_VALUES:
       print("\n" + "=" * 80)
       print(f"RUNNING MODEL FOR LAG = {lag}")
       print("=" * 80)


       result = fit_one_lag(df_base, lag)
       lag_results.append({
           "lag": result["lag"],
           "accuracy_pct": result["metrics"]["accuracy_pct"],
           "mae": result["metrics"]["mae"],
           "rmse": result["metrics"]["rmse"],
           "wape": result["metrics"]["wape"],
           "r2": result["metrics"]["r2"],
           "weighted_rhat": result["diag_summary"]["weighted_rhat"],
           "weighted_ess_bulk": result["diag_summary"]["weighted_ess_bulk"],
           "weighted_ess_tail": result["diag_summary"]["weighted_ess_tail"],
           "n_rows": len(result["df"]),
       })


       if best_result is None or result["metrics"]["accuracy_pct"] > best_result["metrics"]["accuracy_pct"]:
           best_result = result


   results_df = pd.DataFrame(lag_results).sort_values("lag").reset_index(drop=True)


   print("\nFinal lag comparison:")
   print(results_df)


   best_idx = results_df["accuracy_pct"].idxmax()
   best_row = results_df.loc[best_idx]


   print(
       f"\nBest lag based on {ACCURACY_LABEL}: "
       f"lag={int(best_row['lag'])}, accuracy={best_row['accuracy_pct']:.2f}%"
   )


   if SAVE_RESULTS_CSV:
       results_path = os.path.join(base_dir, "lag_sweep_results.csv")
       results_df.to_csv(results_path, index=False)
       print(f"Saved lag comparison table to: {results_path}")

   zero_percentage = np.mean(y == 0) * 100
   print(f"Percentage of zeros in target variable: {zero_percentage:.2f}%")
   
   if MAKE_PLOTS:
       # -------------------------------------------------
       # Plot 1: accuracy by lag
       # -------------------------------------------------
       plt.figure(figsize=(10, 6))
       plt.plot(results_df["lag"], results_df["accuracy_pct"], marker="o", linewidth=2)
       plt.xticks(results_df["lag"])
       plt.xlabel("Lag (weeks)")
       plt.ylabel(ACCURACY_LABEL)
       plt.title("Model accuracy vs lag")
       plt.grid(True, alpha=0.3)


       plt.scatter([best_row["lag"]], [best_row["accuracy_pct"]], s=100)
       plt.annotate(
           f"best lag = {int(best_row['lag'])}\n{best_row['accuracy_pct']:.2f}%",
           xy=(best_row["lag"], best_row["accuracy_pct"]),
           xytext=(10, 10),
           textcoords="offset points",
       )
       plt.tight_layout()
       plt.show()


       # -------------------------------------------------
       # Plot 2: RMSE by lag
       # -------------------------------------------------
       plt.figure(figsize=(10, 6))
       plt.plot(results_df["lag"], results_df["rmse"], marker="o", linewidth=2)
       plt.xticks(results_df["lag"])
       plt.xlabel("Lag (weeks)")
       plt.ylabel("RMSE")
       plt.title("Prediction RMSE vs lag")
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.show()


       # -------------------------------------------------
       # Plot 3: actual vs fitted for the best lag
       # -------------------------------------------------
       y_best = best_result["df"]["cases"].to_numpy(dtype=float)
       fitted_best = best_result["fitted_mu_mean"]
       fitted_std_best = best_result["fitted_mu_std"]


       plt.figure(figsize=(8, 8))
       plt.errorbar(
           y_best,
           fitted_best,
           yerr=fitted_std_best,
           fmt="o",
           alpha=0.15,
           capsize=2,
       )
       line_min = min(float(y_best.min()), float(fitted_best.min()))
       line_max = max(float(y_best.max()), float(fitted_best.max()))
       plt.plot([line_min, line_max], [line_min, line_max], "r--")
       plt.xlabel("Actual cases")
       plt.ylabel("Predicted expected cases")
       plt.title(f"Best lag fit: lag = {best_result['lag']}")
       plt.tight_layout()
       plt.show()




if __name__ == "__main__":
   main()



