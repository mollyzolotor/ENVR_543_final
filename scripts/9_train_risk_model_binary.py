"""
This script:
    1. Reads in the finalized csv file for redcross building points.
    2. Trains a random forest classifier to predict the presence of damage (binary) using the various predictors in the dataset (elevation, slope, HAND, distance to stream, precipitation, impervious surface) as well as some engineered features (e.g. elevation squared, ratio of precipitation to 72H precipitation)
    3. Evaluates model performance using accuracy, confusion matrix, classification report, and variable importance.
    4. Saves the trained model and variable importance table for future use.

Notes:
    -The csv files for microsoft and redcross building points are included in the github repository as "microsoft_points_processed.csv" and "redcross_points_processed.csv". These are the outputs from the previous script (7_elevation.py) but then combined with distance to nearest stream ("distance") which was processed in QGIS (more details on this in the README).
    - The random forest model is trained on the Red Cross data, which has labels for damage presence. The Microsoft data does not have damage labels and is not used for training, but could be used for prediction after the model is trained.
"""


from click import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
import joblib

DATA_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/data/") #For Github: update this to the actual path where your data is stored
MODEL_DIR = Path("/Users/mzolotor/ENVR_543_Final_Project/models/") #For Github: update this to the actual path where you want to save your model and feature columns


def read_redcross_data():
    file_path = DATA_DIR / "redcross_points_H.csv"
    df = pd.read_csv(file_path)
    return df


def make_binary_damage_column(df, damage_col):
    #converting classification labels into binary, where 0 = no damage and 1 = any damage present. Rows labeled Revisit or blank are removed before modeling.
    df = df.copy()
    cleaned = df[damage_col].astype(str).str.strip().str.lower()
    bad_labels = ["revisit", "", "nan"]
    df = df[~cleaned.isin(bad_labels)].copy()
    cleaned = df[damage_col].astype(str).str.strip().str.lower()

    no_damage_labels = ["nvd"]

    # Create binary target
    df["damage_present"] = cleaned.apply(
        lambda x: 0 if x in no_damage_labels else 1
    )

    return df


def prepare_model_data(df, predictor_cols, target_col="damage_present"):
    #Keep only predictors + target, drop rows with missing values

    needed_cols = predictor_cols + [target_col]
    model_df = df[needed_cols].copy()

    print("Rows before dropping NA:", len(model_df))
    model_df = model_df.dropna()
    print("Rows after dropping NA:", len(model_df))

    X = model_df[predictor_cols].copy()
    y = model_df[target_col].copy()

    print("\nPredictor columns after encoding:")
    print(X.columns.tolist())

    print("\nPredictor dtypes:")
    print(X.dtypes)

    return X, y, model_df


def train_random_forest(X, y):
    #Split data into training and validation sets, then fit a random forest classifier. Class weights are set to "balanced" to help address class imbalance in the target variable. You can experiment with different hyperparameters (e.g. n_estimators, max_depth) to see if you can improve performance.
    """
    Split Red Cross data into training and validation sets,
    then fit a random forest classifier.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)

    return rf, X_train, X_val, y_train, y_val


def evaluate_model(model, X_val, y_val):
    """
    Print basic validation metrics.
    """
    y_prob = model.predict_proba(X_val)[:,1]
    y_pred = (y_prob > 0.4).astype(int)

    print("\n--- MODEL PERFORMANCE ---")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    try:
        auc = roc_auc_score(y_val, y_prob)
        print("ROC AUC:", auc)
    except ValueError:
        print("ROC AUC could not be calculated.")

    importance_df = pd.DataFrame({
        "variable": X_val.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n--- VARIABLE IMPORTANCE ---")
    print(importance_df)

    return importance_df


def main():
    df = read_redcross_data()

    #Possible feature engineering ideas to add to dataset before modeling (The commented out ones are just examples we experimented with)
    # df["HAND_x_precip72"] = df["HAND_m"] * df["precipitation_72H"]
    # df["distance_x_precip72"] = df["distance"] * df["precipitation_72H"]
    # df["low_HAND_flag"] = (df["HAND_m"] < 5).astype(int)
    df["elevation_sq"] = df["elevation"] ** 2
    # df["HAND_log"] = np.log1p(df["HAND_m"])
    # df["distance_log"] = np.log1p(df["distance"])
    # df["elev_minus_HAND"] = df["elevation"] - df["HAND_m"]
    # df["high_rain_flag"] = (df["precipitation_72H"] > 200).astype(int)
    # df["near_stream_flag"] = (df["distance"] < 50).astype(int)
    # df["steep_slope_flag"] = (df["slope"] > threshold).astype(int)
    df["precip_ratio"] = df["precipitation"] / (df["precipitation_72H"] + 1)
    df["precip_diff"] = df["precipitation_72H"] - df["precipitation"]


    print("Columns in dataset:")
    print(df.columns.tolist())

    damage_col = "classification"

    df = make_binary_damage_column(df, damage_col)

    print("\nDamage present counts:")
    print(df["damage_present"].value_counts(dropna=False))

    predictor_cols = [
        "precipitation",
        "impervious_surface",
        "HAND_m",
        #"landcover_group",
        "elevation",
        "slope",
        "distance",
        "precipitation_72H",
        "elevation_sq",
        "precip_ratio",
        "precip_diff"
    ]

   
    X, y, model_df = prepare_model_data(
        df,
        predictor_cols=predictor_cols,
        target_col="damage_present"
    )

    model, X_train, X_val, y_train, y_val = train_random_forest(X, y)

    joblib.dump(model, MODEL_DIR + "/damage_model_binary.pkl")
    joblib.dump(predictor_cols, MODEL_DIR + "/feature_cols_binary.pkl")

    importance_df = evaluate_model(model, X_val, y_val)

   
    model_df.to_csv(
        MODEL_DIR + "/redcross_binary_model_table.csv",
        index=False
    )

    importance_df.to_csv(
        MODEL_DIR + "/random_forest_variable_importance.csv",
        index=False
    )

    print("\nDone.")


if __name__ == "__main__":
    main()