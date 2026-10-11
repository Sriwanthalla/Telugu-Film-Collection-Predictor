
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(page_title="Telugu Movie Gross Predictor", page_icon="🎬", layout="wide")

DATA_PATH = Path(__file__).parent / "telugu_movie_prediction_dataset.xlsx"
REQUIRED = ["Movie", "Hero", "Director", "Budget_₹_crore", "Worldwide_Gross_₹_crore"]

st.title("🎬 Telugu Movie Worldwide Gross Predictor")
st.caption("Predict worldwide gross using the selected hero, director, and movie budget.")

@st.cache_data
def load_data(path: str, modified_time: float):
    book = pd.ExcelFile(path)
    sheet = "Movie_Dataset" if "Movie_Dataset" in book.sheet_names else book.sheet_names[0]
    data = pd.read_excel(path, sheet_name=sheet)
    data.columns = [str(c).strip() for c in data.columns]
    return data

def split_names(value):
    if pd.isna(value):
        return []
    return [x.strip() for x in re.split(r";|,|\band\b", str(value), flags=re.I) if x.strip()]

def normalize_hero(value):
    value = str(value).strip()
    aliases = {"ntr": "Jr NTR", "jr ntr": "Jr NTR", "junior ntr": "Jr NTR"}
    return aliases.get(value.lower(), value)

@st.cache_resource
def train_model(data_fingerprint: str, data: pd.DataFrame):
    # Release_Status deliberately does not filter rows; use every row with usable supervised-learning values.
    d = data.copy()
    for c in ["Budget_₹_crore", "Worldwide_Gross_₹_crore"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    valid = (
        d["Budget_₹_crore"].notna() & (d["Budget_₹_crore"] > 0) &
        d["Worldwide_Gross_₹_crore"].notna() & (d["Worldwide_Gross_₹_crore"] >= 0) &
        d["Hero"].notna() & d["Hero"].astype(str).str.strip().ne("") &
        d["Director"].notna() & d["Director"].astype(str).str.strip().ne("")
    )
    train = d.loc[valid].copy()
    if len(train) < 8:
        raise ValueError(f"Only {len(train)} usable training rows. At least 8 are needed for this prototype.")
    train["Hero_list"] = train["Hero"].apply(lambda x: [normalize_hero(h) for h in split_names(x)])
    train["Director_clean"] = train["Director"].astype(str).str.strip()
    encoder = MultiLabelBinarizer()
    hero_mat = encoder.fit_transform(train["Hero_list"])
    hero_features = pd.DataFrame(hero_mat, columns=[f"hero__{h}" for h in encoder.classes_], index=train.index)
    director_categories = sorted(train["Director_clean"].unique())
    director_features = pd.get_dummies(train["Director_clean"], prefix="director", dtype=int)
    X = pd.concat([
        train[["Budget_₹_crore"]].rename(columns={"Budget_₹_crore": "budget_crore"}),
        hero_features,
        director_features
    ], axis=1)
    y = train["Worldwide_Gross_₹_crore"].astype(float)
    model = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features=0.9, random_state=42
    )
    # Fit the model on all usable historical rows for predictions.
    model.fit(X, y)
    return model, encoder, list(X.columns), train, director_categories

if not DATA_PATH.exists():
    st.error("Dataset file not found. Keep `telugu_movie_prediction_dataset.xlsx` in the same folder as `app.py`.")
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()
    fingerprint = str(DATA_PATH.stat().st_mtime_ns) + str(DATA_PATH.stat().st_size)
    model, hero_encoder, feature_columns, training_data, directors = train_model(fingerprint, df)
except Exception as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.stop()

st.subheader("Predict an upcoming movie")
# Build dropdown choices from the usable historical records in the dataset.
hero_options = sorted({
    normalize_hero(hero)
    for value in training_data["Hero"].dropna()
    for hero in split_names(value)
    if str(hero).strip()
})
director_options = sorted({
    str(value).strip()
    for value in training_data["Director"].dropna()
    if str(value).strip()
})

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    hero_input = c1.selectbox("Select hero", options=hero_options, index=None, placeholder="Choose a hero...")
    director_input = c2.selectbox("Select director", options=director_options, index=None, placeholder="Choose a director...")
    budget_input = st.number_input("Upcoming movie budget (₹ crore)", min_value=1.0, value=500.0, step=10.0)
    submitted = st.form_submit_button("Predict worldwide gross", type="primary")

if submitted:
    if not hero_input or not director_input:
        st.warning("Please select both a hero and a director.")
    else:
        heroes = [normalize_hero(h) for h in split_names(hero_input)]
        row = {col: 0 for col in feature_columns}
        row["budget_crore"] = float(budget_input)
        unknown_heroes = []
        for hero in heroes:
            col = f"hero__{hero}"
            if col in row:
                row[col] = 1
            else:
                unknown_heroes.append(hero)
        director_col = f"director_{director_input.strip()}"
        known_directors = set(training_data["Director_clean"])
        if director_col in row:
            row[director_col] = 1
        prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=feature_columns))[0]))

        st.success("Prediction complete")
        a, b = st.columns(2)
        a.metric("Predicted worldwide gross", f"₹{prediction:,.1f} crore")
        b.metric("Budget entered", f"₹{budget_input:,.1f} crore")
        st.caption("This is a model estimate, not a guaranteed collection or a producer-profit/recovery figure.")
        if unknown_heroes:
            st.warning("Hero(s) not represented in training data: " + ", ".join(unknown_heroes) + ". Their individual effect was not learned.")
        if director_input.strip() not in known_directors:
            st.warning(f"Director '{director_input.strip()}' was not represented in training data. The model cannot learn a specific effect for this director.")

with st.expander("Historical data used"):
    cols = [c for c in ["Movie", "Hero", "Director", "Budget_₹_crore", "Worldwide_Gross_₹_crore", "Release_Status"] if c in training_data.columns]
    st.dataframe(training_data[cols].reset_index(drop=True), use_container_width=True)

st.divider()
st.subheader("🎥 About TFI")
st.write(
    "The finest Indian film industry, producing some of the biggest blockbusters in Indian cinema "
    "and introducing Indian cinema to the world."
)
st.caption("Predictions are estimates based on the historical dataset and are not guaranteed collections.")
