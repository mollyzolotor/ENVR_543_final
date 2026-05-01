"""
This script:
    1. Reads in the finalized csv file for redcross building points.
    2. Reads in the Microsoft building points csv file.
    3. Loads the trained random forest models for both binary damage prediction and severity prediction.
    4. Applies the binary damage model to the Microsoft data to predict damaged vs not damaged.
    5. For the Microsoft rows predicted as damaged, applies the severity model to predict low/moderate/severe damage.
    6. Standardizes the predicted damage classes for Microsoft and the observed damage classes for Red Cross into a common scheme of no damage / low / moderate / severe / unknown.
    7. Combines the Microsoft and Red Cross datasets into one final output file with a consistent set of columns and a source label indicating whether each row is a Microsoft prediction or a Red Cross observation.

Notes:
        -The csv files for microsoft and redcross building points are included in the carolina digital repository as "microsoft_points_processed.csv" and "redcross_points_processed.csv". These are the outputs from the previous script 7_elevation.py but then combined with distance to nearest stream ("distance") which was processed in QGIS (more details on this in the README).
"""


from click import Path
import pandas as pd
import joblib

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored
MODEL_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/models/") #For Github: update this to the actual path where you want to save your model and feature columns
OUTPUT_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/output/") #For Github: update this to the actual path where you want to save your output files

MICROSOFT_PATH = DATA_DIR / "microsoft_points_H.csv"
REDCROSS_PATH = DATA_DIR / "redcross_points_H.csv"

BINARY_MODEL_PATH = MODEL_DIR / "damage_model_binary.pkl"
BINARY_FEATURES_PATH = MODEL_DIR / "feature_cols_binary.pkl"

SEVERITY_MODEL_PATH = MODEL_DIR / "damage_model_classes.pkl"
SEVERITY_FEATURES_PATH = MODEL_DIR / "feature_cols_classes.pkl"

MICROSOFT_OUTPUT_PATH = OUTPUT_DIR / "microsoft_points_predicted.csv"
COMBINED_OUTPUT_PATH = OUTPUT_DIR / "all_points_real_and_predicted.csv"


NO_DAMAGE_BINARY_VALUE = 0
DAMAGE_BINARY_VALUE = 1


SEVERITY_MAP = {
    1: "low",
    2: "moderate",
    3: "severe"
}

REDCROSS_TRUE_DAMAGE_COL = "classification"

# Standardize Red Cross labels into the same final scheme
REDCROSS_DAMAGE_MAP = {
    "NVD": "no damage",
    "Affected": "low",
    "Minor": "moderate",
    "Major": "severe",
    "Destroyed": "severe",
    "Inaccessible": "unknown",
    "Revisit": "unknown"
}

MICROSOFT_COLUMNS_TO_KEEP_AND_RENAME = {
    "County": "county",
    "lat": "lat",
    "lon": "lon",
    "HAND_m": "HAND_m",
    "slope": "slope",
    "precipitation": "precipitation",
    "precipitation_72H": "precipitation_72H",
    "impervious_surface": "impervious",
    "elevation": "elevation",
    "elevation_sq": "elevation_sq",
    "elev_minus_HAND": "elev_minus_HAND",
    "precip_ratio": "precip_ratio",
    "precip_diff": "precip_diff",
    "predicted_damage": "predicted_damage_binary",
    "predicted_damage_prob": "predicted_damage_probability",
    "predicted_severity": "predicted_severity_raw",
    "final_damage_class": "damage_class_final"
}

REDCROSS_COLUMNS_TO_KEEP_AND_RENAME = {
    "county": "county",
    "y": "lat",
    "x": "lon",
    "HAND_m": "HAND_m",
    "slope": "slope",
    "precipitation": "precipitation",
    "precipitation_72H": "precipitation_72H",
    "impervious_surface": "impervious",
    "elevation": "elevation",
    "elevation_sq": "elevation_sq",
    "elev_minus_HAND": "elev_minus_HAND",
    "precip_ratio": "precip_ratio",
    "precip_diff": "precip_diff",
    "true_damage_class": "damage_class_final"
}

MICROSOFT_SOURCE_LABEL = "microsoft_predicted"
REDCROSS_SOURCE_LABEL = "redcross_observed"


def load_csv(path, label):
    df = pd.read_csv(path)
    print(f"Loaded {label}: {len(df)} rows")
    return df


def add_engineered_features(df, dataset_label):
    """
    Add engineered columns used by your models.
    """
    df = df.copy()

    required_for_engineering = ["elevation", "HAND_m", "precipitation", "precipitation_72H"]
    missing = [col for col in required_for_engineering if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_label} is missing columns needed for engineered features: {missing}")

    df["elevation_sq"] = df["elevation"] ** 2
    df["elev_minus_HAND"] = df["elevation"] - df["HAND_m"]
    df["precip_ratio"] = df["precipitation"] / (df["precipitation_72H"] + 1)
    df["precip_diff"] = df["precipitation_72H"] - df["precipitation"]

    return df


def prepare_features(df, feature_cols):
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    return df[feature_cols].copy()


def print_missing_feature_summary(df, feature_cols, label):
    print(f"\nMissing values for {label} feature columns:")
    for col in feature_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].isna().sum()}")
        else:
            print(f"  {col}: COLUMN MISSING")


def predict_binary_damage(buildings, binary_model, binary_feature_cols):
    X = prepare_features(buildings, binary_feature_cols)
    valid_mask = X.notna().all(axis=1)

    buildings = buildings.copy()
    buildings["predicted_damage"] = pd.NA
    buildings["predicted_damage_prob"] = pd.NA

    if valid_mask.sum() == 0:
        print("No Microsoft rows had complete binary predictor data.")
        return buildings

    X_valid = X.loc[valid_mask]

    buildings.loc[valid_mask, "predicted_damage"] = binary_model.predict(X_valid)

    if hasattr(binary_model, "predict_proba"):
        buildings.loc[valid_mask, "predicted_damage_prob"] = binary_model.predict_proba(X_valid)[:, 1]

    print(f"Binary predictions made for {valid_mask.sum()} Microsoft buildings.")
    print(f"Microsoft rows skipped for binary prediction: {(~valid_mask).sum()}")
    return buildings


def convert_severity_predictions_to_classes(y_pred):
    out = []
    for val in y_pred:
        if val >= 2.3:
            out.append(3)
        elif val >= 1.6:
            out.append(2)
        else:
            out.append(1)
    return out


def predict_severity(buildings, severity_model, severity_feature_cols):
    buildings = buildings.copy()
    buildings["predicted_severity"] = pd.NA
    buildings["predicted_severity_continuous"] = pd.NA

    damaged_mask = buildings["predicted_damage"] == DAMAGE_BINARY_VALUE

    if damaged_mask.sum() == 0:
        print("No Microsoft buildings were predicted damaged, so no severity predictions were made.")
        return buildings

    damaged_buildings = buildings.loc[damaged_mask].copy()
    X = prepare_features(damaged_buildings, severity_feature_cols)
    valid_mask = X.notna().all(axis=1)

    if valid_mask.sum() == 0:
        print("No damaged Microsoft buildings had complete severity predictor data.")
        return buildings

    X_valid = X.loc[valid_mask]

    severity_pred_continuous = severity_model.predict(X_valid)
    severity_pred_class = convert_severity_predictions_to_classes(severity_pred_continuous)

    buildings.loc[X_valid.index, "predicted_severity_continuous"] = severity_pred_continuous
    buildings.loc[X_valid.index, "predicted_severity"] = severity_pred_class

    print(f"Severity predictions made for {len(X_valid)} Microsoft buildings.")
    print(f"Predicted-damaged Microsoft rows skipped for severity prediction: {(~valid_mask).sum()}")
    print("\nUnique severity class predictions:")
    print(pd.Series(severity_pred_class).value_counts().sort_index())

    return buildings

def build_final_damage_class_microsoft(df):
    """
    Create one clean final damage column for Microsoft rows.
    """
    df = df.copy()

    def combine_row(row):
        if pd.isna(row["predicted_damage"]):
            return "unknown"
        if row["predicted_damage"] == NO_DAMAGE_BINARY_VALUE:
            return "no damage"
        if pd.isna(row["predicted_severity"]):
            return "unknown"
        return SEVERITY_MAP.get(row["predicted_severity"], "unknown")

    df["final_damage_class"] = df.apply(combine_row, axis=1)
    return df


def prepare_redcross_truth(df):
    """
    Standardize observed Red Cross classes into:
    no damage / low / moderate / severe / unknown
    """
    df = df.copy()

    if REDCROSS_TRUE_DAMAGE_COL not in df.columns:
        raise ValueError(
            f"Red Cross damage column '{REDCROSS_TRUE_DAMAGE_COL}' not found. "
            f"Available columns are: {df.columns.tolist()}"
        )

    df["true_damage_class"] = df[REDCROSS_TRUE_DAMAGE_COL].map(REDCROSS_DAMAGE_MAP)

    unmapped = df.loc[df["true_damage_class"].isna(), REDCROSS_TRUE_DAMAGE_COL].dropna().unique()
    if len(unmapped) > 0:
        print("\nWarning: unmapped Red Cross damage classes found:")
        print(unmapped)

    
    df["true_damage_class"] = df["true_damage_class"].fillna("unknown")

    return df


def select_and_rename_columns(df, column_map, dataset_name):
    df = df.copy()

    missing_cols = [col for col in column_map.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing columns required for final output: {missing_cols}"
        )

    df = df[list(column_map.keys())].rename(columns=column_map)
    return df


def add_source_column(df, source_label):
    df = df.copy()
    df["data_source"] = source_label
    return df


def align_columns_for_concat(df1, df2):
    all_cols = sorted(set(df1.columns).union(set(df2.columns)))

    for col in all_cols:
        if col not in df1.columns:
            df1[col] = pd.NA
        if col not in df2.columns:
            df2[col] = pd.NA

    df1 = df1[all_cols].copy()
    df2 = df2[all_cols].copy()

    return df1, df2


def main():

    microsoft = load_csv(MICROSOFT_PATH, "Microsoft buildings")
    redcross = load_csv(REDCROSS_PATH, "Red Cross training set")

    microsoft = add_engineered_features(microsoft, "Microsoft")
    redcross = add_engineered_features(redcross, "Red Cross")

    binary_model = joblib.load(BINARY_MODEL_PATH)
    binary_feature_cols = joblib.load(BINARY_FEATURES_PATH)

    print("\nBinary feature columns:")
    print(binary_feature_cols)
    print_missing_feature_summary(microsoft, binary_feature_cols, "Microsoft binary")


    microsoft = predict_binary_damage(microsoft, binary_model, binary_feature_cols)

    try:
        severity_model = joblib.load(SEVERITY_MODEL_PATH)
        severity_feature_cols = joblib.load(SEVERITY_FEATURES_PATH)

        print("\nSeverity feature columns:")
        print(severity_feature_cols)
        print_missing_feature_summary(
            microsoft.loc[microsoft["predicted_damage"] == DAMAGE_BINARY_VALUE].copy(),
            severity_feature_cols,
            "Microsoft severity"
        )

        microsoft = predict_severity(microsoft, severity_model, severity_feature_cols)

    except FileNotFoundError:
        print("Severity model files not found. Microsoft output will only have binary damage.")
        microsoft["predicted_severity"] = pd.NA
    except Exception as e:
        print(f"Severity prediction skipped due to error: {e}")
        microsoft["predicted_severity"] = pd.NA


    microsoft = build_final_damage_class_microsoft(microsoft)

    print("\nMicrosoft final damage counts:")
    print(microsoft["final_damage_class"].value_counts(dropna=False))
    print("\nMissing binary predictions:")
    print(microsoft["predicted_damage"].isna().sum())
    print("\nMissing severity predictions among predicted damaged:")
    print(((microsoft["predicted_damage"] == DAMAGE_BINARY_VALUE) & (microsoft["predicted_severity"].isna())).sum())


    redcross = prepare_redcross_truth(redcross)

    print("\nRed Cross standardized damage counts:")
    print(redcross["true_damage_class"].value_counts(dropna=False))

    microsoft_final = select_and_rename_columns(
        microsoft,
        MICROSOFT_COLUMNS_TO_KEEP_AND_RENAME,
        "Microsoft output"
    )
    microsoft_final = add_source_column(microsoft_final, MICROSOFT_SOURCE_LABEL)

    redcross_final = select_and_rename_columns(
        redcross,
        REDCROSS_COLUMNS_TO_KEEP_AND_RENAME,
        "Red Cross output"
    )
    redcross_final = add_source_column(redcross_final, REDCROSS_SOURCE_LABEL)


    microsoft_final.to_csv(MICROSOFT_OUTPUT_PATH, index=False)
    print(f"\nSaved Microsoft prediction file to:\n{MICROSOFT_OUTPUT_PATH}")

    microsoft_final, redcross_final = align_columns_for_concat(microsoft_final, redcross_final)
    combined = pd.concat([redcross_final, microsoft_final], ignore_index=True)


    combined.to_csv(COMBINED_OUTPUT_PATH, index=False)
    print(f"\nSaved combined real + predicted file to:\n{COMBINED_OUTPUT_PATH}")


    print("\nCombined damage class counts:")
    if "damage_class_final" in combined.columns:
        print(combined["damage_class_final"].value_counts(dropna=False))
    else:
        print("damage_class_final column not found in combined output.")


if __name__ == "__main__":
    main()