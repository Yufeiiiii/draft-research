from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path("/Users/yufeizou/draft_research_2")
DATA_PATH = PROJECT_DIR / "NHL_2009_2018_combined_cleaned.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "observed_values_faceted_scatter.png"
RESPONSES = ["TOI_sum_min", "GP_sum", "PS_sum"]


def load_plot_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, usecols=["Selection", *RESPONSES]).copy()
    df["Selection"] = pd.to_numeric(df["Selection"], errors="coerce")

    for column in RESPONSES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Selection"]).copy()

    long_df = df.melt(
        id_vars="Selection",
        value_vars=RESPONSES,
        var_name="response",
        value_name="observed_value",
    )
    long_df = long_df.dropna(subset=["observed_value"]).copy()

    return long_df


def make_plot(plot_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

    for ax, response in zip(axes, RESPONSES):
        panel_df = plot_df.loc[plot_df["response"] == response]
        ax.scatter(
            panel_df["Selection"],
            panel_df["observed_value"],
            alpha=0.35,
            s=18,
            color="#4c78a8",
            edgecolors="none",
        )
        ax.set_title(response)
        ax.set_xlabel("Selection")
        ax.set_ylabel("Observed Value")
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Observed Values by Response", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_df = load_plot_data()
    make_plot(plot_df)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
