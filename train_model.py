"""
Train and evaluate the Telugu Movie Worldwide Gross Predictor.

Expected dataset:
    telugu_movie_prediction_dataset.xlsx

Required columns:
    Hero, Director, Budget_₹_crore, Worldwide_Gross_₹_crore

Run:
    python train_model.py
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

DATA_PATH = Path(__file__).parent / "telugu_movie_prediction_dataset.xlsx"
TARGET = "Worldwide_Gross_₹_crore"
BUDGET = "Budget_₹_crore"
REQUIRED = ["Hero", "Director", BUDGET, TARGET]


def split_names(value):
    if pd.isna(value):
        return []
    return [
        name.strip()
        for name in re.split(r";|,|\band\b", str(value), flags=re.IGNORECASE)
        if name.strip()
    ]


def normalize_hero(value):
    value = str(value).strip()
    aliases = {"ntr": "Jr NTR", "jr ntr": "Jr NTR", "junior ntr": "Jr NTR"}
    return aliases.get(value.lower(), value)


def load_dataset(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Keep the Excel file beside train_model.py."
        )
    workbook = pd.ExcelFile(path)
    sheet = "Movie_Dataset" if "Movie_Dataset" in workbook.sheet_names else workbook.sheet_names[0]
    data = pd.read_excel(path, sheet_name=sheet)
    data.columns = [str(column).strip() for column in data.columns]

    missing = [column for column in REQUIRED if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data[BUDGET] = pd.to_numeric(data[BUDGET], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")

    valid = (
        data[BUDGET].notna()
        & (data[BUDGET] > 0)
        & data[TARGET].notna()
        & (data[TARGET] >= 0)
        & data["Hero"].notna()
        & data["Hero"].astype(str).str.strip().ne("")
        & data["Director"].notna()
        & data["Director"].astype(str).str.strip().ne("")
    )
    usable = data.loc[valid].copy()
    if len(usable) < 8:
        raise ValueError(f"Only {len(usable)} usable rows; at least 8 are needed for this prototype.")
    return usable


def make_features(data, hero_encoder=None, feature_columns=None):
    hero_lists = data["Hero"].apply(
        lambda value: [normalize_hero(hero) for hero in split_names(value)]
    )

    if hero_encoder is None:
        hero_encoder = MultiLabelBinarizer()
        hero_matrix = hero_encoder.fit_transform(hero_lists)
    else:
        hero_matrix = hero_encoder.transform(hero_lists)

    hero_features = pd.DataFrame(
        hero_matrix,
        columns=[f"hero__{hero}" for hero in hero_encoder.classes_],
        index=data.index,
    )

    director_features = pd.get_dummies(
        data["Director"].astype(str).str.strip(), prefix="director", dtype=int
    )
    features = pd.concat(
        [data[[BUDGET]].rename(columns={BUDGET: "budget_crore"}), hero_features, director_features],
        axis=1,
    )

    if feature_columns is not None:
        features = features.reindex(columns=feature_columns, fill_value=0)
    return features, hero_encoder


def build_model():
    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=42,
    )


def main():
    data = load_dataset()
    print(f"Usable historical rows: {len(data)}")

    # Hold-out evaluation gives an estimate of generalization, but a tiny dataset
    # makes metrics unstable. These metrics should not be treated as a guarantee.
    train_rows, test_rows = train_test_split(data, test_size=0.25, random_state=42)
    X_train, encoder = make_features(train_rows)
    X_test, _ = make_features(test_rows, hero_encoder=encoder, feature_columns=X_train.columns)
    y_train = train_rows[TARGET].astype(float)
    y_test = test_rows[TARGET].astype(float)

    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    test_predictions = evaluation_model.predict(X_test)

    rmse = mean_squared_error(y_test, test_predictions) ** 0.5
    print("\nHeld-out evaluation (₹ crore)")
    print(f"MAE : {mean_absolute_error(y_test, test_predictions):,.3f}")
    print(f"RMSE: {rmse:,.3f}")
    print(f"R²  : {r2_score(y_test, test_predictions):,.3f}")
    print(f"Training rows: {len(train_rows)}")
    print(f"Test rows    : {len(test_rows)}")

    # Refit on all usable rows for a final model that can be used for predictions.
    X_all, final_encoder = make_features(data)
    final_model = build_model()
    final_model.fit(X_all, data[TARGET].astype(float))

    # Save a transparent training-data summary. The Streamlit app currently
    # trains its own model from the Excel file, so this script is independently
    # runnable and does not alter the live website.
    summary = pd.DataFrame(
        [{
            "usable_rows": len(data),
            "mae_crore": mean_absolute_error(y_test, test_predictions),
            "rmse_crore": rmse,
            "r2": r2_score(y_test, test_predictions),
            "training_rows": len(train_rows),
            "test_rows": len(test_rows),
            "model": "RandomForestRegressor",
        }]
    )
    output_path = Path(__file__).parent / "model_evaluation.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nEvaluation summary saved to: {output_path}")
    print("Note: This script evaluates and refits the model; the Streamlit app still trains from the Excel dataset.")


if __name__ == "__main__":
    main()
