from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path("/Users/yufeizou/draft_research_2/draft_research")
ROOT_DIR = Path("/Users/yufeizou/draft_research_2")


def normalize_to_1_100(value, min_val, max_val):
    if pd.isna(value):
        return None
    if max_val == min_val:
        return 50.0
    return 1 + ((value - min_val) / (max_val - min_val)) * 99


def load_nhl_scatter(target_column):
    data_path = PROJECT_DIR / "NHL_2009_2018_combined_cleaned.csv"
    df = pd.read_csv(data_path)
    df = df[["Selection", target_column]].copy()
    df["Selection"] = pd.to_numeric(df["Selection"], errors="coerce")
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=["Selection", target_column]).copy()
    df["Selection"] = df["Selection"].astype(int)

    target_min = df[target_column].min()
    target_max = df[target_column].max()
    df["response_normalized"] = df[target_column].apply(
        lambda x: normalize_to_1_100(x, target_min, target_max)
    )
    return df


def plot_panel(ax, scatter_df, x_col, curve_path, title, xlabel, xlim_max):
    curve_df = pd.read_csv(curve_path)

    ax.scatter(
        scatter_df[x_col],
        scatter_df["response_normalized"],
        alpha=0.25,
        s=18,
        color="gray",
    )
    ax.plot(
        curve_df["overall"],
        curve_df["raw_model_output"],
        color="red",
        linewidth=3,
        label="Posterior mean",
    )
    ax.fill_between(
        curve_df["overall"],
        curve_df["raw_floor_5pct"],
        curve_df["raw_ceiling_95pct"],
        color="red",
        alpha=0.2,
        label="90% credible interval",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Model Output (Normalized Scale)")
    ax.set_xlim(0, xlim_max)
    ax.axhline(0, color="black", linewidth=1, alpha=0.3)


def main():
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (14, 8)

    gp_scatter = load_nhl_scatter("GP_sum")
    ps_scatter = load_nhl_scatter("PS_sum")
    toi_scatter = load_nhl_scatter("TOI_sum_min")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    plot_panel(
        ax=axes[0],
        scatter_df=gp_scatter,
        x_col="Selection",
        curve_path=PROJECT_DIR / "results" / "gp_sum" / "draft_curve_gp_sum_lookup.csv",
        title="GP",
        xlabel="Draft Pick (Selection)",
        xlim_max=215,
    )
    plot_panel(
        ax=axes[1],
        scatter_df=ps_scatter,
        x_col="Selection",
        curve_path=PROJECT_DIR / "results" / "ps_sum" / "draft_curve_ps_sum_lookup.csv",
        title="PS",
        xlabel="Draft Pick (Selection)",
        xlim_max=215,
    )
    plot_panel(
        ax=axes[2],
        scatter_df=toi_scatter,
        x_col="Selection",
        curve_path=PROJECT_DIR / "results" / "toi_sum_min" / "draft_curve_toi_sum_min_lookup.csv",
        title="TOI",
        xlabel="Draft Pick (Selection)",
        xlim_max=215,
    )
    fig.delaxes(axes[3])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle("B-spline Comparison", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    output_path = PROJECT_DIR / "results" / "draft_curve_faceted.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved faceted plot: {output_path}")


if __name__ == "__main__":
    main()
