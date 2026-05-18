from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path("/Users/yufeizou/draft_research_2/draft_research")
RESULTS_DIR = PROJECT_DIR / "results"


def scale_curve_sum1000(curve_df, label, epsilon=1e-8):
    raw_values = curve_df["raw_model_output"].astype(float)
    min_value = raw_values.min()
    raw_sum = raw_values.sum()
    shift_value = 0.0
    epsilon_added = 0.0

    adjusted_values = raw_values.copy()

    if min_value < 0:
        shift_value = -min_value
        adjusted_values = adjusted_values + shift_value

    adjusted_sum = adjusted_values.sum()

    if adjusted_sum <= 0:
        epsilon_added = epsilon
        adjusted_values = adjusted_values + epsilon
        adjusted_sum = adjusted_values.sum()

    if adjusted_sum <= 0:
        raise ValueError(
            f"[{label}] Unable to scale fitted curve because the adjusted sum is nonpositive."
        )

    scaled_values = adjusted_values / adjusted_sum * 1000.0

    scaled_df = curve_df.copy()
    scaled_df["scaled_curve"] = scaled_values
    scaled_df["shift_value"] = shift_value
    scaled_df["epsilon_added"] = epsilon_added
    scaled_df["raw_sum"] = raw_sum
    scaled_df["adjusted_sum"] = adjusted_sum
    return scaled_df


def load_scaled_curve(label, curve_path):
    curve_df = pd.read_csv(curve_path)
    curve_df = curve_df[["overall", "raw_model_output"]].copy()
    curve_df["overall"] = pd.to_numeric(curve_df["overall"], errors="coerce")
    curve_df["raw_model_output"] = pd.to_numeric(
        curve_df["raw_model_output"], errors="coerce"
    )
    curve_df = curve_df.dropna(subset=["overall", "raw_model_output"]).copy()
    curve_df = curve_df.groupby("overall", as_index=False)["raw_model_output"].mean()
    curve_df = curve_df.sort_values("overall").reset_index(drop=True)

    scaled_df = scale_curve_sum1000(curve_df, label)
    scaled_df["metric"] = label
    return scaled_df


def main():
    sns.set_style("whitegrid")

    curve_specs = [
        (
            "GP",
            PROJECT_DIR / "results" / "gp_sum" / "draft_curve_gp_sum_lookup_extended_224.csv",
            "#d95f02",
        ),
        (
            "PS",
            PROJECT_DIR / "results" / "ps_sum" / "draft_curve_ps_sum_lookup_extended_224.csv",
            "#1b9e77",
        ),
        (
            "TOI",
            PROJECT_DIR
            / "results"
            / "toi_sum_min"
            / "draft_curve_toi_sum_min_lookup_extended_224.csv",
            "#7570b3",
        ),
    ]

    scaled_curves = []
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, curve_path, color in curve_specs:
        scaled_df = load_scaled_curve(label, curve_path)
        scaled_curves.append(scaled_df)
        ax.plot(
            scaled_df["overall"],
            scaled_df["scaled_curve"],
            color=color,
            linewidth=3,
            label=label,
        )

    combined_df = pd.concat(scaled_curves, ignore_index=True)

    ax.set_title("B-spline Comparison (Curve Sum = 1000)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Draft Pick (Selection)")
    ax.set_ylabel("Scaled Fitted Value")
    ax.set_xlim(1, 224)
    ax.legend(loc="upper right", frameon=False, title="Metric")

    plt.tight_layout()

    csv_output_path = RESULTS_DIR / "draft_curve_bspline_scaled_sum1000_single_extended_224.csv"
    plot_output_path = RESULTS_DIR / "draft_curve_bspline_scaled_sum1000_single_extended_224.png"

    combined_df.to_csv(csv_output_path, index=False)
    plt.savefig(plot_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved scaled B-spline CSV: {csv_output_path}")
    print(f"Saved scaled B-spline plot: {plot_output_path}")


if __name__ == "__main__":
    main()
