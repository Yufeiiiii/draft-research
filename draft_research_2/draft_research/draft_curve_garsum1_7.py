import os

os.environ["PYTENSOR_FLAGS"] = "base_compiledir=/Users/yufeizou/draft_research_2/.python_cache/pytensor,linker=py,cxx="
print(os.environ["PYTENSOR_FLAGS"])


import pytensor

print("PYTENSOR_FLAGS =", os.environ.get("PYTENSOR_FLAGS"))
print("pytensor.config.cxx =", repr(pytensor.config.cxx))
print("pytensor.config.linker =", pytensor.config.linker)


import warnings
from datetime import date
from pathlib import Path


local_cache = Path.cwd() / '.python_cache'
local_cache.mkdir(parents=True, exist_ok=True)
posix_cache = local_cache.as_posix()

os.environ['LOCALAPPDATA'] = posix_cache
os.environ['PYTENSOR_FLAGS'] = f'base_compiledir={posix_cache}/pytensor'
os.environ['NUMBA_CACHE_DIR'] = f'{posix_cache}/numba'
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_BOUNDSCHECK'] = '0'
os.environ['JAX_PLATFORMS'] = 'cpu'

import numba
numba.config.CACHE_DIR = f'{posix_cache}/numba'

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from patsy import build_design_matrices, dmatrix


try:
    arviz_cache_dir = local_cache / 'arviz' / 'arviz' / 'Cache'
    arviz_cache_dir.mkdir(parents=True, exist_ok=True)
    (arviz_cache_dir / 'daily_warning').write_text(date.today().isoformat())
except Exception:
    pass

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

print('Libraries loaded.')


import sys
print(sys.executable)

import arviz as az
print("arviz file:", az.__file__)
print("arviz version:", getattr(az, "__version__", "no version"))
print("has InferenceData:", hasattr(az, "InferenceData"))


print('=' * 70)
print('SECTION 1: LOAD AND VALIDATE DATA')
print('=' * 70)

data_path = Path.cwd() / 'draft_pick_wide_GAR_adjusted_2009_2018_GARsum1_7.csv'
export_dir = Path.cwd() / 'results' / 'garsum1_7_curve'
export_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)
print(f'Loaded {len(df):,} rows from {data_path.name}')
print(f'Columns: {df.columns.tolist()}')

required_cols = ['DraftYr', 'DraftOv', 'GARsum1_7']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f'Missing required columns: {missing_cols}')

df = df[required_cols].copy()
df['DraftYr'] = pd.to_numeric(df['DraftYr'], errors='coerce')
df['DraftOv'] = pd.to_numeric(df['DraftOv'], errors='coerce')
df['GARsum1_7'] = pd.to_numeric(df['GARsum1_7'], errors='coerce')
df = df.dropna(subset=['DraftOv', 'GARsum1_7']).copy()

df['DraftOv'] = df['DraftOv'].astype(int)
df = df.sort_values(['DraftOv', 'DraftYr']).reset_index(drop=True)

print(f'Draft years: {int(df["DraftYr"].min())}-{int(df["DraftYr"].max())}')
print(f'Pick range: {df["DraftOv"].min()}-{df["DraftOv"].max()}')
print(df.head())


print('=' * 70)
print('SECTION 2: PREPARE RESPONSE')
print('=' * 70)

target_col = 'GARsum1_7'
df['response_raw'] = df[target_col]

target_min = df['response_raw'].min()
target_max = df['response_raw'].max()

# normalize
def normalize_to_1_100(value, min_val, max_val):
    if pd.isna(value):
        return np.nan
    if max_val == min_val:
        return 50.0
    return 1 + ((value - min_val) / (max_val - min_val)) * 99

df['response_normalized'] = df['response_raw'].apply(lambda x: normalize_to_1_100(x, target_min, target_max))

print(f'Raw target range: {target_min:.3f} to {target_max:.3f}')
print(f'Normalized target range: {df["response_normalized"].min():.3f} to {df["response_normalized"].max():.3f}')
print(df[['DraftYr', 'DraftOv', 'response_raw', 'response_normalized']].head(10))


print('=' * 70)
print('SECTION 3: TRAINING DATA AND SPLINE BASIS')
print('=' * 70)

train_max_pick = 210
df_train = df[df['DraftOv'] <= train_max_pick].copy()
df_train = df_train.sort_values('DraftOv').reset_index(drop=True)

X = df_train['DraftOv'].values.astype(float)
y = df_train['response_normalized'].values.astype(float)

print(f'Training rows: {len(df_train):,}')
print(f'Excluded picks > {train_max_pick}: {(df["DraftOv"] > train_max_pick).sum():,}')
print(f'X range: {X.min():.0f} to {X.max():.0f}')
print(f'y mean/std: {y.mean():.3f} / {y.std():.3f}')
#internal knots
internal_knots = [5, 15, 30, 50, 100, 150, 200]
boundary_knots = [1, train_max_pick]

# build B-spline basis
B_matrix = dmatrix(
    f"bs(x, knots={internal_knots}, degree=3, include_intercept=True, lower_bound={boundary_knots[0]}, upper_bound={boundary_knots[1]}) - 1",
    {'x': X}
)
design_info = B_matrix.design_info
B = np.asarray(B_matrix)

print(f'Basis matrix shape: {B.shape}')


print('=' * 70)
print('SECTION 4: BAYESIAN MONOTONIC B-SPLINE MODEL')
print('=' * 70)

# pirors
prior_mu_start = np.max(y) if len(y) else 100.0
prior_sigma = np.std(y) if len(y) else 1.0

n_basis = B.shape[1]
n_steps = n_basis - 1
#earlier steps are allowed to vary more
step_priors = np.zeros(n_steps)
step_priors[:4] = prior_sigma / 2
step_priors[4:] = prior_sigma / 20

print(f'prior_mu_start: {prior_mu_start:.3f}')
print(f'prior_sigma: {prior_sigma:.3f}')
print(f'n_basis: {n_basis}, n_steps: {n_steps}')

with pm.Model() as garsum1_7_spline_model:
    sigma = pm.HalfNormal('sigma', sigma=prior_sigma)
    beta_0 = pm.HalfNormal('beta_0', sigma=prior_mu_start) #first spline coefficient
    steps = pm.HalfNormal('steps', sigma=step_priors, shape=n_steps) #step size coefficient

    cumulative_steps = pm.math.concatenate([pt.zeros(1), pm.math.cumsum(steps)])
    betas_full = beta_0 - cumulative_steps
    betas = pm.Deterministic('betas', pm.math.maximum(betas_full, 0))
    mu = pm.Deterministic('mu', pm.math.dot(B, betas))

    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print('Model defined.')


print('=' * 70)
print('SECTION 5: POSTERIOR SAMPLING')
print('=' * 70)

with garsum1_7_spline_model:
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        nuts_sampler='nutpie',
        target_accept=0.90,
        random_seed=42,
        progressbar=True
    )

rhat = az.rhat(idata)
rhat_max = max(v.max().item() for v in rhat.data_vars.values())
print(f'Max R-hat: {rhat_max:.4f}')


print('=' * 70)
print('SECTION 6: EXTRACT PREDICTIONS AND BUILD CURVE')
print('=' * 70)

pick_grid_train = np.arange(1, train_max_pick + 1, dtype=float)
B_grid_train = build_design_matrices([design_info], {'x': pick_grid_train})[0]
B_grid_train = np.asarray(B_grid_train)

posterior_betas = idata.posterior['betas'].values
n_chains, n_draws, n_basis = posterior_betas.shape
posterior_betas_flat = posterior_betas.reshape(n_chains * n_draws, n_basis)

mu_grid_samples = np.dot(B_grid_train, posterior_betas_flat.T).T
mu_mean = mu_grid_samples.mean(axis=0)
mu_hdi_3pct = np.percentile(mu_grid_samples, 3, axis=0)
mu_hdi_97pct = np.percentile(mu_grid_samples, 97, axis=0)

pick_grid = np.arange(1, int(df['DraftOv'].max()) + 1, dtype=float)
mu_mean_extended = np.zeros(len(pick_grid))
mu_hdi_3_extended = np.zeros(len(pick_grid))
mu_hdi_97_extended = np.zeros(len(pick_grid))

mu_mean_extended[:train_max_pick] = mu_mean
mu_hdi_3_extended[:train_max_pick] = mu_hdi_3pct
mu_hdi_97_extended[:train_max_pick] = mu_hdi_97pct

tail_value = mu_mean[-1]
tail_floor = mu_hdi_3pct[-1]
tail_ceiling = mu_hdi_97pct[-1]

if len(pick_grid) > train_max_pick:
    mu_mean_extended[train_max_pick:] = tail_value
    mu_hdi_3_extended[train_max_pick:] = tail_floor
    mu_hdi_97_extended[train_max_pick:] = tail_ceiling

pick_1_value = mu_mean[0]
if pick_1_value > 0:
    expected_value = 100 * mu_mean_extended / pick_1_value
    value_floor_3pct = 100 * mu_hdi_3_extended / pick_1_value
    value_ceiling_97pct = 100 * mu_hdi_97_extended / pick_1_value
else:
    expected_value = mu_mean_extended
    value_floor_3pct = mu_hdi_3_extended
    value_ceiling_97pct = mu_hdi_97_extended

master_curve = pd.DataFrame({
    'overall': pick_grid.astype(int),
    'expected_draft_value': expected_value,
    'value_ceiling_97pct': value_ceiling_97pct,
    'value_floor_3pct': value_floor_3pct,
    'raw_model_output': mu_mean_extended,
    'raw_floor_3pct': mu_hdi_3_extended,
    'raw_ceiling_97pct': mu_hdi_97_extended,
})

curve_path = export_dir / 'draft_curve_garsum1_7_lookup.csv'
master_curve.to_csv(curve_path, index=False)
print(master_curve.head(15))
print(f'Saved curve: {curve_path}')


print('=' * 70)
print('SECTION 7: VISUALIZATION')
print('=' * 70)

plt.figure(figsize=(14, 8))
plt.scatter(df['DraftOv'], df['response_normalized'], alpha=0.25, s=18, color='gray', label='Observed normalized GARsum1_7')
plt.plot(master_curve['overall'], master_curve['raw_model_output'], color='red', linewidth=3, label='Expected value (mean)')
plt.fill_between(master_curve['overall'], master_curve['raw_floor_3pct'], master_curve['raw_ceiling_97pct'], color='red', alpha=0.15, label='94% credible interval')
plt.title('Draft Curve for GARsum1_7')
plt.xlabel('Draft Pick (Overall)')
plt.ylabel('Model Output (Normalized Scale)')
plt.xlim(0, int(df['DraftOv'].max()) + 5)
plt.axhline(0, color='black', linewidth=1, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = export_dir / 'draft_curve_garsum1_7.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Saved plot: {plot_path}')


# LOYO CV for the monotonic Bayesian spline model.
# This is Leave-one-year-out CV: we hold out one entire draft year at a time.
# That is more appropriate than random K-fold here because it tests how well the
# curve generalizes across draft classes rather than across randomly mixed picks.

from patsy import build_design_matrices, dmatrix

def sigma_weight(y_value: float) -> float:
    """
    Weight function for weighted RMSE.
    Customize this later if you want higher-value picks/outcomes to matter more.
    Current default: uniform weights.
    """
    return 1.0


def compute_weighted_rmse(y_true, y_pred, weight_fn=sigma_weight):
    """
    Weighted RMSE as requested:
        sqrt(sum_i w_i * (y_i - yhat_i)^2)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray([weight_fn(y) for y in y_true], dtype=float)
    return np.sqrt(np.sum(weights * (y_true - y_pred) ** 2))


def fit_monotone_spline_fold(
    train_df,
    test_df,
    spline_formula,
    draws=500,
    tune=500,
    chains=2,
    cores=2,
    target_accept=0.90,
    random_seed=42,
):
    """
    Fit one LOYO fold and return fold metrics plus optional predictions.
    """
    # Training design matrix: fit spline basis on training data only
    X_train = train_df["DraftOv"].values.astype(float)
    y_train = train_df["response_normalized"].values.astype(float)

    B_train_matrix = dmatrix(spline_formula, {"x": X_train})
    design_info = B_train_matrix.design_info
    B_train = np.asarray(B_train_matrix)

    # Test design matrix: reuse training design_info to avoid leakage
    X_test = test_df["DraftOv"].values.astype(float)
    y_test = test_df["response_normalized"].values.astype(float)

    B_test = build_design_matrices([design_info], {"x": X_test})[0]
    B_test = np.asarray(B_test)

    # Priors computed from training data only
    prior_mu_start = np.max(y_train) if len(y_train) else 100.0
    prior_sigma = np.std(y_train) if len(y_train) else 1.0
    if prior_sigma == 0:
        prior_sigma = 1.0

    n_basis = B_train.shape[1]
    n_steps = n_basis - 1

    step_priors = np.zeros(n_steps, dtype=float)
    step_priors[:4] = prior_sigma / 2
    step_priors[4:] = prior_sigma / 20

    with pm.Model() as fold_model:
        sigma = pm.HalfNormal("sigma", sigma=prior_sigma)
        beta_0 = pm.HalfNormal("beta_0", sigma=prior_mu_start)
        steps = pm.HalfNormal("steps", sigma=step_priors, shape=n_steps)

        cumulative_steps = pm.math.concatenate([pt.zeros(1), pm.math.cumsum(steps)])
        betas_full = beta_0 - cumulative_steps
        betas = pm.Deterministic("betas", pm.math.maximum(betas_full, 0))

        mu = pm.math.dot(B_train, betas)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    posterior_betas = idata.posterior["betas"].values
    posterior_betas_flat = posterior_betas.reshape(-1, posterior_betas.shape[-1])
    beta_mean = posterior_betas_flat.mean(axis=0)

    yhat_test = B_test @ beta_mean

    rmse = np.sqrt(np.mean((y_test - yhat_test) ** 2))
    weighted_rmse = compute_weighted_rmse(y_test, yhat_test)

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "rmse": rmse,
        "weighted_rmse": weighted_rmse,
        "y_true": y_test,
        "y_pred": yhat_test,
        "beta_mean": beta_mean,
    }


def run_loyo_cv(
    df,
    train_max_pick,
    draws=500,
    tune=500,
    chains=2,
    cores=2,
    target_accept=0.90,
    random_seed=42,
):
    """
    Run leave-one-year-out CV across all unique DraftYr values.
    """
    cv_df = df.loc[df["DraftOv"] <= train_max_pick, ["DraftYr", "DraftOv", "response_normalized"]].copy()
    cv_df = cv_df.dropna(subset=["DraftYr", "DraftOv", "response_normalized"]).copy()
    cv_df["DraftYr"] = cv_df["DraftYr"].astype(int)

    spline_formula = (
        "bs(x, knots=(5, 15, 30, 50, 100, 150, 200), "
        "degree=3, include_intercept=True, lower_bound=1, upper_bound=210) - 1"
    )

    years = sorted(cv_df["DraftYr"].unique())
    fold_results = []
    prediction_rows = []

    for test_year in years:
        print(f"\nRunning LOYO fold for held-out year: {test_year}")

        train_df = cv_df[cv_df["DraftYr"] != test_year].copy()
        test_df = cv_df[cv_df["DraftYr"] == test_year].copy()

        fold_out = fit_monotone_spline_fold(
            train_df=train_df,
            test_df=test_df,
            spline_formula=spline_formula,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
        )

        fold_results.append({
            "test_year": test_year,
            "n_train": fold_out["n_train"],
            "n_test": fold_out["n_test"],
            "rmse": fold_out["rmse"],
            "weighted_rmse": fold_out["weighted_rmse"],
        })

        # Optional: store predictions for inspection by held-out year
        fold_pred_df = test_df[["DraftYr", "DraftOv", "response_normalized"]].copy()
        fold_pred_df = fold_pred_df.rename(columns={"response_normalized": "y_true"})
        fold_pred_df["y_pred"] = fold_out["y_pred"]
        fold_pred_df["test_year"] = test_year
        prediction_rows.append(fold_pred_df)

    results_df = pd.DataFrame(fold_results).sort_values("test_year").reset_index(drop=True)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    return results_df, predictions_df


# Run LOYO CV with lighter sampling settings for a first pass
loyo_results_df, loyo_predictions_df = run_loyo_cv(
    df=df,
    train_max_pick=train_max_pick,
    draws=500,
    tune=500,
    chains=2,
    cores=2,
    target_accept=0.90,
    random_seed=42,
)


# Fold-level results and summary metrics
print("\nFold-level LOYO CV results:")
print(loyo_results_df.to_string(index=False))

print("\nSummary:")
print(f"Mean RMSE across years: {loyo_results_df['rmse'].mean():.4f}")
print(f"SD of RMSE across years: {loyo_results_df['rmse'].std(ddof=1):.4f}")
print(f"Mean weighted RMSE across years: {loyo_results_df['weighted_rmse'].mean():.4f}")
print(f"SD of weighted RMSE across years: {loyo_results_df['weighted_rmse'].std(ddof=1):.4f}")
