import os
import warnings
from datetime import date
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from patsy import build_design_matrices, dmatrix


DATA_PATH = Path.cwd() / "NHL_2009_2018_combined_cleaned.csv"
TARGET_COLUMN = "PS_sum"
PREDICTOR_COLUMN = "Selection"
GROUP_COLUMN = "Year"
TRAIN_MAX_PICK = 210
INTERNAL_KNOTS = (5, 15, 30, 50, 100, 150, 200)


def configure_environment():
    local_cache = Path.cwd() / ".python_cache"
    local_cache.mkdir(parents=True, exist_ok=True)
    posix_cache = local_cache.as_posix()

    os.environ["LOCALAPPDATA"] = posix_cache
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={posix_cache}/pytensor"
    os.environ["NUMBA_CACHE_DIR"] = f"{posix_cache}/numba"
    os.environ["NUMBA_DISABLE_JIT"] = "0"
    os.environ["NUMBA_BOUNDSCHECK"] = "0"
    os.environ["JAX_PLATFORMS"] = "cpu"

    numba.config.CACHE_DIR = f"{posix_cache}/numba"

    try:
        arviz_cache_dir = local_cache / "arviz" / "arviz" / "Cache"
        arviz_cache_dir.mkdir(parents=True, exist_ok=True)
        (arviz_cache_dir / "daily_warning").write_text(date.today().isoformat())
    except Exception:
        pass

    warnings.filterwarnings("ignore")
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)


def normalize_to_1_100(value, min_val, max_val):
    if pd.isna(value):
        return np.nan
    if max_val == min_val:
        return 50.0
    return 1 + ((value - min_val) / (max_val - min_val)) * 99


def invert_1_100_scale(value, min_val, max_val):
    if pd.isna(value):
        return np.nan
    if max_val == min_val:
        return min_val
    return min_val + ((value - 1) / 99) * (max_val - min_val)


def sigma_weight(y_value: float) -> float:
    """Default weight for weighted RMSE; customize here if needed later."""
    return 1.0


def compute_weighted_rmse(y_true, y_pred, weight_fn=sigma_weight):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray([weight_fn(y) for y in y_true], dtype=float)
    return np.sqrt(np.mean(weights * (y_true - y_pred) ** 2))



def load_and_prepare_data(data_path: Path, target_column: str):
    df = pd.read_csv(data_path)

    required_cols = [GROUP_COLUMN, PREDICTOR_COLUMN, TARGET_COLUMN]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()
    df[GROUP_COLUMN] = pd.to_numeric(df[GROUP_COLUMN], errors="coerce")
    df[PREDICTOR_COLUMN] = pd.to_numeric(df[PREDICTOR_COLUMN], errors="coerce")
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=required_cols).copy()

    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype(int)
    df[PREDICTOR_COLUMN] = df[PREDICTOR_COLUMN].astype(int)
    df["response_raw"] = df[target_column]

    target_min = df["response_raw"].min()
    target_max = df["response_raw"].max()
    df["response_normalized"] = df["response_raw"].apply(
        lambda x: normalize_to_1_100(x, target_min, target_max)
    )

    df = df.sort_values([PREDICTOR_COLUMN, GROUP_COLUMN]).reset_index(drop=True)
    return df, target_min, target_max


def get_spline_formula(train_max_pick: int):
    return (
        f"bs(x, knots={INTERNAL_KNOTS}, degree=3, include_intercept=True, "
        f"lower_bound=1, upper_bound={train_max_pick}) - 1"
    )


def fit_monotone_spline_model(
    X_train,
    y_train,
    spline_formula,
    draws=2000,
    tune=1000,
    chains=4,
    cores=4,
    target_accept=0.90,
    random_seed=42,
):
    B_train_matrix = dmatrix(spline_formula, {"x": X_train})
    design_info = B_train_matrix.design_info
    B_train = np.asarray(B_train_matrix)

    prior_mu_start = np.max(y_train) if len(y_train) else 100.0
    prior_sigma = np.std(y_train) if len(y_train) else 1.0
    if prior_sigma == 0:
        prior_sigma = 1.0

    n_basis = B_train.shape[1]
    n_steps = n_basis - 1
    step_priors = np.zeros(n_steps, dtype=float)
    step_priors[:4] = prior_sigma / 2
    step_priors[4:] = prior_sigma / 20

    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", sigma=prior_sigma)
        beta_0 = pm.HalfNormal("beta_0", sigma=prior_mu_start)
        steps = pm.HalfNormal("steps", sigma=step_priors, shape=n_steps)

        cumulative_steps = pm.math.concatenate([pt.zeros(1), pm.math.cumsum(steps)])
        betas_full = beta_0 - cumulative_steps
        betas = pm.Deterministic("betas", pm.math.maximum(betas_full, 0))
        mu = pm.Deterministic("mu", pm.math.dot(B_train, betas))

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            nuts_sampler="nutpie",
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    return model, idata, design_info, B_train


def summarize_rhat(idata):
    rhat = az.rhat(idata)
    return max(v.max().item() for v in rhat.data_vars.values())


def build_curve_from_idata(idata, design_info, max_pick):
    pick_grid = np.arange(1, max_pick + 1, dtype=float)
    B_grid = build_design_matrices([design_info], {"x": pick_grid})[0]
    B_grid = np.asarray(B_grid)

    posterior_betas = idata.posterior["betas"].values
    posterior_betas_flat = posterior_betas.reshape(-1, posterior_betas.shape[-1])

    mu_grid_samples = np.dot(B_grid, posterior_betas_flat.T).T
    mu_mean = mu_grid_samples.mean(axis=0)
    mu_hdi_3pct = np.percentile(mu_grid_samples, 3, axis=0)
    mu_hdi_97pct = np.percentile(mu_grid_samples, 97, axis=0)

    pick_1_value = mu_mean[0]
    if pick_1_value > 0:
        expected_value = 100 * mu_mean / pick_1_value
        value_floor_3pct = 100 * mu_hdi_3pct / pick_1_value
        value_ceiling_97pct = 100 * mu_hdi_97pct / pick_1_value
    else:
        expected_value = mu_mean
        value_floor_3pct = mu_hdi_3pct
        value_ceiling_97pct = mu_hdi_97pct

    return pd.DataFrame(
        {
            "overall": pick_grid.astype(int),
            "expected_draft_value": expected_value,
            "value_ceiling_97pct": value_ceiling_97pct,
            "value_floor_3pct": value_floor_3pct,
            "raw_model_output": mu_mean,
            "raw_floor_3pct": mu_hdi_3pct,
            "raw_ceiling_97pct": mu_hdi_97pct,
        }
    )


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
    X_train = train_df[PREDICTOR_COLUMN].values.astype(float)
    y_train_raw = train_df["response_raw"].values.astype(float)
    y_test_raw = test_df["response_raw"].values.astype(float)

    target_min = np.min(y_train_raw)
    target_max = np.max(y_train_raw)

    y_train = np.asarray(
        [normalize_to_1_100(y, target_min, target_max) for y in y_train_raw],
        dtype=float,
    )

    B_train_matrix = dmatrix(spline_formula, {"x": X_train})
    design_info = B_train_matrix.design_info
    B_train = np.asarray(B_train_matrix)

    X_test = test_df[PREDICTOR_COLUMN].values.astype(float)
    y_test = np.asarray(
        [normalize_to_1_100(y, target_min, target_max) for y in y_test_raw],
        dtype=float,
    )
    B_test = build_design_matrices([design_info], {"x": X_test})[0]
    B_test = np.asarray(B_test)

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
            nuts_sampler="nutpie",
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    posterior_betas = idata.posterior["betas"].values
    posterior_betas_flat = posterior_betas.reshape(-1, posterior_betas.shape[-1])
    beta_mean = posterior_betas_flat.mean(axis=0)
    y_pred = B_test @ beta_mean

    y_pred_raw = np.asarray(
        [invert_1_100_scale(y, target_min, target_max) for y in y_pred],
        dtype=float,
    )

    rmse_normalized = np.sqrt(np.mean((y_test - y_pred) ** 2))
    weighted_rmse_normalized = compute_weighted_rmse(y_test, y_pred)
    rmse_raw = np.sqrt(np.mean((y_test_raw - y_pred_raw) ** 2))
    weighted_rmse_raw = compute_weighted_rmse(y_test_raw, y_pred_raw)

    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "target_min": target_min,
        "target_max": target_max,
        "rmse_normalized": rmse_normalized,
        "weighted_rmse_normalized": weighted_rmse_normalized,
        "rmse_raw": rmse_raw,
        "weighted_rmse_raw": weighted_rmse_raw,
        "y_true_normalized": y_test,
        "y_pred_normalized": y_pred,
        "y_true_raw": y_test_raw,
        "y_pred_raw": y_pred_raw,
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
    # Leave-one-year-out CV is more appropriate than random K-fold here because
    # it evaluates generalization to an entirely unseen draft class.
    cv_df = df.loc[
        df[PREDICTOR_COLUMN] <= train_max_pick,
        [GROUP_COLUMN, PREDICTOR_COLUMN, "response_raw"],
    ].copy()
    cv_df = cv_df.dropna().copy()
    cv_df[GROUP_COLUMN] = cv_df[GROUP_COLUMN].astype(int)

    spline_formula = get_spline_formula(train_max_pick)
    years = sorted(cv_df[GROUP_COLUMN].unique())

    fold_rows = []
    prediction_rows = []

    for test_year in years:
        print(f"\nRunning LOYO fold for held-out year: {test_year}")
        train_df = cv_df[cv_df[GROUP_COLUMN] != test_year].copy()
        test_df = cv_df[cv_df[GROUP_COLUMN] == test_year].copy()

        fold_result = fit_monotone_spline_fold(
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

        fold_rows.append(
            {
                "test_year": test_year,
                "n_train": fold_result["n_train"],
                "n_test": fold_result["n_test"],
                "rmse_normalized": fold_result["rmse_normalized"],
                "weighted_rmse_normalized": fold_result["weighted_rmse_normalized"],
                "rmse_raw": fold_result["rmse_raw"],
                "weighted_rmse_raw": fold_result["weighted_rmse_raw"],
            }
        )

        pred_df = test_df[[GROUP_COLUMN, PREDICTOR_COLUMN]].copy()
        pred_df["y_true_normalized"] = fold_result["y_true_normalized"]
        pred_df["y_pred_normalized"] = fold_result["y_pred_normalized"]
        pred_df["y_true_raw"] = fold_result["y_true_raw"]
        pred_df["y_pred_raw"] = fold_result["y_pred_raw"]
        pred_df["test_year"] = test_year
        prediction_rows.append(pred_df)

    results_df = pd.DataFrame(fold_rows).sort_values("test_year").reset_index(drop=True)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    return results_df, predictions_df


def save_curve_plot(df_train, master_curve, export_dir, target_column):
    plt.figure(figsize=(14, 8))
    plt.scatter(
        df_train[PREDICTOR_COLUMN],
        df_train["response_normalized"],
        alpha=0.25,
        s=18,
        color="gray",
        label=f"Observed normalized {target_column}",
    )
    plt.plot(
        master_curve["overall"],
        master_curve["raw_model_output"],
        color="red",
        linewidth=3,
        label="Expected value (mean)",
    )
    plt.fill_between(
        master_curve["overall"],
        master_curve["raw_floor_3pct"],
        master_curve["raw_ceiling_97pct"],
        color="red",
        alpha=0.15,
        label="94% credible interval",
    )
    plt.title(f"Draft Curve for {target_column}")
    plt.xlabel("Draft Pick (Selection)")
    plt.ylabel("Model Output (Normalized Scale)")
    plt.xlim(0, TRAIN_MAX_PICK + 5)
    plt.axhline(0, color="black", linewidth=1, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = export_dir / f"draft_curve_{target_column.lower()}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_path


def main():
    configure_environment()

    export_dir = Path.cwd() / "results" / TARGET_COLUMN.lower()
    export_dir.mkdir(parents=True, exist_ok=True)

    df, target_min, target_max = load_and_prepare_data(DATA_PATH, TARGET_COLUMN)
    print(f"Loaded {len(df):,} rows for {TARGET_COLUMN}")
    print(f"Raw target range: {target_min:.3f} to {target_max:.3f}")
    print(f"Draft years: {df[GROUP_COLUMN].min()}-{df[GROUP_COLUMN].max()}")
    print(f"Pick range: {df[PREDICTOR_COLUMN].min()}-{df[PREDICTOR_COLUMN].max()}")

    df_train = df[df[PREDICTOR_COLUMN] <= TRAIN_MAX_PICK].copy()
    X_train = df_train[PREDICTOR_COLUMN].values.astype(float)
    y_train = df_train["response_normalized"].values.astype(float)
    spline_formula = get_spline_formula(TRAIN_MAX_PICK)

    _, idata, design_info, _ = fit_monotone_spline_model(
        X_train=X_train,
        y_train=y_train,
        spline_formula=spline_formula,
    )

    rhat_max = summarize_rhat(idata)
    print(f"Max R-hat: {rhat_max:.4f}")

    master_curve = build_curve_from_idata(idata, design_info, TRAIN_MAX_PICK)
    curve_path = export_dir / f"draft_curve_{TARGET_COLUMN.lower()}_lookup.csv"
    master_curve.to_csv(curve_path, index=False)

    plot_path = save_curve_plot(df_train, master_curve, export_dir, TARGET_COLUMN)

    loyo_results_df, loyo_predictions_df = run_loyo_cv(
        df=df,
        train_max_pick=TRAIN_MAX_PICK,
        draws=500,
        tune=500,
        chains=2,
        cores=2,
        target_accept=0.90,
        random_seed=42,
    )

    cv_results_path = export_dir / f"{TARGET_COLUMN.lower()}_loyo_results.csv"
    cv_predictions_path = export_dir / f"{TARGET_COLUMN.lower()}_loyo_predictions.csv"
    loyo_results_df.to_csv(cv_results_path, index=False)
    loyo_predictions_df.to_csv(cv_predictions_path, index=False)

    print("\nFold-level LOYO CV results:")
    print(loyo_results_df.to_string(index=False))
    print("\nSummary:")
    print(
        f"Mean RMSE across years (normalized scale): "
        f"{loyo_results_df['rmse_normalized'].mean():.4f}"
    )
    print(
        f"SD of RMSE across years (normalized scale): "
        f"{loyo_results_df['rmse_normalized'].std(ddof=1):.4f}"
    )
    print(
        f"Mean weighted RMSE across years (normalized scale): "
        f"{loyo_results_df['weighted_rmse_normalized'].mean():.4f}"
    )
    print(
        f"SD of weighted RMSE across years (normalized scale): "
        f"{loyo_results_df['weighted_rmse_normalized'].std(ddof=1):.4f}"
    )
    print(
        f"Mean RMSE across years (raw scale): "
        f"{loyo_results_df['rmse_raw'].mean():.4f}"
    )
    print(
        f"SD of RMSE across years (raw scale): "
        f"{loyo_results_df['rmse_raw'].std(ddof=1):.4f}"
    )
    print(
        f"Mean weighted RMSE across years (raw scale): "
        f"{loyo_results_df['weighted_rmse_raw'].mean():.4f}"
    )
    print(
        f"SD of weighted RMSE across years (raw scale): "
        f"{loyo_results_df['weighted_rmse_raw'].std(ddof=1):.4f}"
    )

    print(f"\nSaved curve lookup: {curve_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved LOYO results: {cv_results_path}")
    print(f"Saved LOYO predictions: {cv_predictions_path}")


if __name__ == "__main__":
    main()
