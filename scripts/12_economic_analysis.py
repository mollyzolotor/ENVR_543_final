"""
This script: 
    1. Reads in the finalized csv file for redcross and microsoft building points (all buildings with damage classifications, both observed and predicted). 
    3. Loads the trained random forest models for both binary damage prediction and severity prediction.
    4. Applies the binary damage model to the Microsoft data to predict damaged vs not damaged
    5. For the Microsoft rows predicted as damaged, applies the severity model to predict low/moderate/severe damage.
    6. Standardizes the predicted damage classes for Microsoft and the observed damage classes for Red Cross into a common scheme of no damage / low / moderate / severe / unknown.
    7. Combines the Microsoft and Red Cross datasets into one final output file with a consistent set of columns and a source label indicating whether each row is a Microsoft prediction or a Red Cross observation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/")   # Update for GitHub
OUTPUT_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/outputs/")  # match your folder name

EVI_PATH = DATA_DIR / "county_economic_vitality_index_2010_2024.csv"
DAMAGE_PATH = OUTPUT_DIR / "all_points_real_and_predicted.csv"

OUTPUT_CSV = OUTPUT_DIR / "county_resilience_summary.csv"
OUTPUT_FIGURE = OUTPUT_DIR / "resilience_quadrant.png"


TARGET_COUNTIES = [
    "Alexander", "Ashe", "Watauga", "Avery", "Mitchell",
    "Burke", "Yancey", "McDowell", "Rutherford", "Buncombe",
    "Henderson", "Haywood", "Jackson"
]


def load_data(evi_path, damage_path):
    """Load EVI and predicted damage datasets."""
    evi = pd.read_csv(evi_path)
    damage = pd.read_csv(damage_path)

    return evi, damage


def clean_damage_data(damage):
    """Clean damage dataset and standardize county names."""
    damage = damage.dropna(subset=["predicted_severity_raw", "county"]).copy()
    damage["county"] = damage["county"].str.title()

    return damage


def calculate_economic_impact(evi, target_counties):
    """
    Calculate county-level economic impact using change in EVI composite score.

    Economic impact is defined as:
        EVI composite score in 2023 - EVI composite score in 2024

    A larger positive value indicates a larger decline from 2023 to 2024.
    """
    evi_subset = evi[evi["county_name"].isin(target_counties)].copy()

    evi_2023 = (
        evi_subset[evi_subset["year"] == 2023]
        [["county_name", "composite_score"]]
        .rename(columns={"composite_score": "score_2023"})
    )

    evi_2024 = (
        evi_subset[evi_subset["year"] == 2024]
        [["county_name", "composite_score"]]
        .rename(columns={"composite_score": "score_2024"})
    )

    evi_combined = pd.merge(evi_2023, evi_2024, on="county_name", how="inner")
    evi_combined["economic_impact"] = (
        evi_combined["score_2023"] - evi_combined["score_2024"]
    )

    return evi_combined


def aggregate_damage_by_county(damage):
    """Calculate mean predicted damage severity for each county."""
    damage_by_county = (
        damage.groupby("county")["predicted_severity_raw"]
        .mean()
        .reset_index()
        .rename(columns={
            "county": "county_name",
            "predicted_severity_raw": "avg_damage"
        })
    )

    return damage_by_county


def merge_damage_and_economic_data(evi_combined, damage_by_county):
    """Merge county-level damage and economic impact data."""
    combined = pd.merge(
        evi_combined,
        damage_by_county,
        on="county_name",
        how="inner"
    )

    return combined


def make_resilience_quadrant_plot(combined, output_figure):
    """Create and save a quadrant plot of physical damage vs. economic impact."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        combined["avg_damage"],
        combined["economic_impact"],
        s=100,
        zorder=5
    )

    for _, row in combined.iterrows():
        ax.annotate(
            row["county_name"],
            xy=(row["avg_damage"], row["economic_impact"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9
        )

    x_mid = combined["avg_damage"].median()
    y_mid = combined["economic_impact"].median()

    ax.axvline(x=x_mid, linestyle="--", linewidth=1)
    ax.axhline(y=y_mid, linestyle="--", linewidth=1)

    xmin = combined["avg_damage"].min()
    xmax = combined["avg_damage"].max()
    ymin = combined["economic_impact"].min()
    ymax = combined["economic_impact"].max()

    ax.text(
        xmax, ymax,
        "High Damage\nHigh Economic Impact\n(Needs Most Aid)",
        fontsize=9,
        ha="right",
        va="top"
    )

    ax.text(
        xmin, ymax,
        "Low Damage\nHigh Economic Impact",
        fontsize=9,
        ha="left",
        va="top"
    )

    ax.text(
        xmax, ymin,
        "High Damage\nLow Economic Impact\n(Most Resilient)",
        fontsize=9,
        ha="right",
        va="bottom"
    )

    ax.text(
        xmin, ymin,
        "Low Damage\nLow Economic Impact",
        fontsize=9,
        ha="left",
        va="bottom"
    )

    ax.set_xlabel("Average Predicted Damage Score (0–3)", fontsize=12)
    ax.set_ylabel("Economic Impact (EVI 2023 − EVI 2024)", fontsize=12)
    ax.set_title(
        "County Resilience: Physical Damage vs. Economic Impact\n"
        "Hurricane Helene, Western North Carolina",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(output_figure, dpi=150)
    plt.show()


def main():
    """Run the full economic resilience analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    evi, damage = load_data(EVI_PATH, DAMAGE_PATH)

    damage = clean_damage_data(damage)
    evi_combined = calculate_economic_impact(evi, TARGET_COUNTIES)
    damage_by_county = aggregate_damage_by_county(damage)

    combined = merge_damage_and_economic_data(evi_combined, damage_by_county)

    combined.to_csv(OUTPUT_CSV, index=False)

    print("\nCounty resilience summary:")
    print(combined[["county_name", "avg_damage", "economic_impact"]])

    make_resilience_quadrant_plot(combined, OUTPUT_FIGURE)

    print(f"\nSaved summary CSV to: {OUTPUT_CSV}")
    print(f"Saved figure to: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()