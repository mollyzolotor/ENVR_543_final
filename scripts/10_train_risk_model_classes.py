"""
This script:
    1. Reads in the finalized csv file for redcross building points.
    2. Trains a random forest model to predict the damage classification (ordinal) using the various predictors in the dataset (elevation, slope, HAND, distance to stream, precipitation, impervious surface) as well as some engineered features (elevation squared, ratio of precipitation to 72H precipitation)
    3. Evaluates model performance using accuracy, mean absolute error, off-by-one accuracy, confusion matrix, classification report, and variable importance.
    4. Saves the trained model and variable importance table for future use.

Notes:
    -The csv files for microsoft and redcross building points are included in the carolina digital repository as "microsoft_points_processed.csv" and "redcross_points_processed.csv". These are the outputs from the previous script 7_elevation.py but then combined with distance to nearest stream ("distance") which was processed in QGIS (more details on this in the README).
    - The random forest model is trained on data that has already been filtered to only include rows with damage classifications of Affected, Minor, Major, or Destroyed. The "Revisit" and blank classifications were removed before modeling. The target variable is treated as ordinal, where Affected/Minor = 1 (low damage), Major = 2 (moderate damage), and Destroyed = 3 (severe damage). The model predicts a continuous value which is then converted to the nearest class for evaluation.
"""


from joblib.numpy_pickle import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.utils import compute_sample_weight
import joblib

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored
MODEL_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/models/") #For Github: update this to the actual path where you want to save your model and feature columns

def load_data():
    file_path = DATA_DIR / "redcross_points_H.csv"
    df = pd.read_csv(file_path)
    return df


def prepare_data(df):
    damage_col = "classification"

    df[damage_col] = df[damage_col].astype(str).str.strip().str.lower()

    print("\nOriginal class counts:")
    print(df[damage_col].value_counts())

    def collapse_damage(x):
        if x in ["affected", "minor"]:
            return "low"
        elif x == "major":
            return "moderate"
        elif x == "destroyed":
            return "severe"
        else:
            return np.nan

    df["damage_group"] = df[damage_col].apply(collapse_damage)
    df = df[df["damage_group"].notnull()].copy()

    print("\nCollapsed class counts:")
    print(df["damage_group"].value_counts())

    damage_map = {
        "low": 1,
        "moderate": 2,
        "severe": 3
    }

    reverse_damage_map = {v: k for k, v in damage_map.items()}
    df["damage_ordinal"] = df["damage_group"].map(damage_map)

    return df, reverse_damage_map


def select_features(df):
    feature_cols = [
        "precipitation",
        "impervious_surface",
        "HAND_m",
        #"landcover_group",
        "elevation",
        "slope",
        "distance",
        "precipitation_72H",
        #"elevation_sq",
        # "precip_ratio",
        # "precip_diff"
        #"HAND_x_precip72",
        #"distance_x_precip72"
        #"low_HAND_flag"
        #"elevation_sq",
        #"HAND_log",
        #"distance_log",
        "elev_minus_HAND",
        "precip_ratio",
        "precip_diff"
    ]

    feature_cols = [col for col in feature_cols if col in df.columns]


    X = df[feature_cols].copy()
    X = pd.get_dummies(X, drop_first=False)

    y = df["damage_ordinal"].copy()

    valid_idx = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X, y, feature_cols


def convert_predictions_to_classes(y_pred):
    out = []
    for val in y_pred:
        if val >= 2.3:        # We added a lower threshold for severe to improve recall for that class
            out.append(3)
        elif val >= 1.6:
            out.append(2)
        else:
            out.append(1)
    return np.array(out)


def evaluate_model(y_test, y_pred_ord, reverse_damage_map):
    accuracy = accuracy_score(y_test, y_pred_ord)
    mae = mean_absolute_error(y_test, y_pred_ord)
    off_by_one_accuracy = np.mean(np.abs(y_test - y_pred_ord) <= 1)

    labels = [1, 2, 3]
    label_names = [reverse_damage_map[i] for i in labels]

    cm = confusion_matrix(y_test, y_pred_ord, labels=labels)
    report = classification_report(
        y_test,
        y_pred_ord,
        labels=labels,
        target_names=label_names,
        zero_division=0
    )

    print("\n--- ORDINAL MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Off-by-one accuracy: {off_by_one_accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=label_names, columns=label_names))

    print("\nClassification Report:")
    print(report)


def print_feature_importance(model, X):
    importance_df = pd.DataFrame({
        "variable": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n--- VARIABLE IMPORTANCE ---")
    print(importance_df)

    return importance_df


def main():
    df = load_data()

     #Possible feature engineering ideas
    #df["HAND_x_precip72"] = df["HAND_m"] * df["precipitation_72H"]
    #df["distance_x_precip72"] = df["distance"] * df["precipitation_72H"]
    #df["low_HAND_flag"] = (df["HAND_m"] < 3).astype(int)
    #df["elevation_sq"] = df["elevation"] ** 2
    #df["HAND_log"] = np.log1p(df["HAND_m"])
    #df["distance_log"] = np.log1p(df["distance"])
    df["elev_minus_HAND"] = df["elevation"] - df["HAND_m"]
    # df["high_rain_flag"] = (df["precipitation_72H"] > 200).astype(int)
    # df["near_stream_flag"] = (df["distance"] < 50).astype(int)
    # df["steep_slope_flag"] = (df["slope"] > threshold).astype(int)
    df["precip_ratio"] = df["precipitation"] / (df["precipitation_72H"] + 1)
    df["precip_diff"] = df["precipitation_72H"] - df["precipitation"]

    df, reverse_damage_map = prepare_data(df)

    print(f"\nRows after filtering to damaged classes only: {len(df)}")
    print("\nOriginal 4-class counts still present in filtered data:")
    print(df["classification"].value_counts())

    print("\nCollapsed class counts used for modeling:")
    print(df["damage_group"].value_counts())

    X, y, feature_cols = select_features(df)

    print(f"\nNumber of observations used in model: {len(X)}")
    print(f"Number of predictor columns after encoding: {X.shape[1]}")

    print("\nOrdinal class counts used in model:")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Compute class weights to handle class imbalance because our classes are pretty imbalanced (especially severe damage class)
    from sklearn.utils.class_weight import compute_sample_weight 

    weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )
    model.fit(X_train, y_train, sample_weight=weights)

    joblib.dump(model, MODEL_DIR + "/damage_model_classes.pkl")
    joblib.dump(feature_cols, MODEL_DIR + "/feature_cols_classes.pkl")

    y_pred_continuous = model.predict(X_test)
    y_pred_ord = convert_predictions_to_classes(y_pred_continuous)

    evaluate_model(y_test, y_pred_ord, reverse_damage_map)
    print_feature_importance(model, X)



if __name__ == "__main__":
    main()