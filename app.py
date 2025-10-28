import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Seattle Weather Explorer", layout="wide")

st.title("Weather Explorer - Streamlit App")
st.markdown("Upload a CSV/XLSX from the left or place the file in the same folder as this app. The app will load, show EDA, train a GaussianNB model and allow interactive prediction.")

DATA_FILES = [
    "seattle-weather.csv",
    "seattle-weather.xlsx",
    "seattle-weather.xls",
]

@st.cache_data
def load_local_or_uploaded(uploaded_file):
    # If user uploaded a file via Streamlit uploader, use it first
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
                return pd.read_excel(uploaded_file)
            else:
                return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return None

    # Look for a file in the app folder
    cwd = Path.cwd()
    for name in DATA_FILES:
        p = cwd / name
        if p.exists():
            try:
                if p.suffix.lower() in [".xls", ".xlsx"]:
                    return pd.read_excel(p)
                else:
                    return pd.read_csv(p)
            except Exception as e:
                st.error(f"Error reading {p}: {e}")
                return None

    # Fallback: search for files with 'seattle' in name
    found = next(cwd.glob("*seattle*.csv"), None) or next(cwd.glob("*seattle*.xls*"), None)
    if found:
        try:
            if found.suffix.lower() in [".xls", ".xlsx"]:
                return pd.read_excel(found)
            else:
                return pd.read_csv(found)
        except Exception as e:
            st.error(f"Error reading {found}: {e}")
            return None

    return None

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel (optional)", type=["csv", "xls", "xlsx"])

with st.spinner("Loading dataset..."):
    df = load_local_or_uploaded(uploaded_file)

if df is None:
    st.warning("No dataset found. Place `seattle-weather.csv/xlsx` in the app folder or upload a file.")
    st.stop()

# Basic cleaning and parsing
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])    
    except Exception:
        pass

# show dataset
st.subheader("Dataset preview")
st.dataframe(df.head(200))

# quick summary
with st.expander("Summary statistics and info"):
    st.write(df.describe(include='all'))
    buf = []
    try:
        df.info(buf=buf)
        st.text("\n".join(buf))
    except Exception:
        st.text("Could not display df.info() output.")

# EDA plots
st.subheader("Exploratory plots")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Temperature distributions")
    fig, ax = plt.subplots(figsize=(6,3))
    if "temp_max" in df.columns:
        sns.histplot(data=df, x="temp_max", bins=20, ax=ax, color='tomato')
        ax.set_title('temp_max')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(6,3))
    if "temp_min" in df.columns:
        sns.histplot(data=df, x="temp_min", bins=20, ax=ax, color='steelblue')
        ax.set_title('temp_min')
    st.pyplot(fig)

with col2:
    st.markdown("### Weather type counts")
    if "weather" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x="weather", order=df["weather"].value_counts().index, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.markdown("### Precipitation vs Wind")
    if "precipitation" in df.columns and "wind" in df.columns:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=df, x="precipitation", y="wind", ax=ax)
        st.pyplot(fig)

# Prepare features for modeling
required_features = ["temp_min", "temp_max", "precipitation", "wind", "weather"]
missing = [c for c in required_features if c not in df.columns]

if missing:
    st.warning(f"Required columns missing for modeling: {missing}. The model UI will be disabled until these columns are available.")

# Training
st.subheader("Train Gaussian Naive Bayes model")
if not missing:
    if "year" not in df.columns and "date" in df.columns:
        try:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
        except Exception:
            pass

    le = LabelEncoder()
    try:
        y = le.fit_transform(df['weather'].astype(str))
    except Exception as e:
        st.error(f"Label encoding failed: {e}")
        st.stop()

    X = df[["temp_min", "temp_max", "precipitation", "wind"]].copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()

    # Align y with X after dropping na rows
    y_aligned = le.transform(df.loc[X.index, 'weather'].astype(str))

    test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2)
    random_state = st.sidebar.number_input("Random seed", value=42, step=1)

    if st.button("Train model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y_aligned, test_size=test_size, random_state=int(random_state))
        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained — accuracy: {acc:.3f}")

        st.write("Confusion matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification report:")
        st.text(classification_report(y_test, y_pred, zero_division=1))

        # store model and encoder in session_state
        st.session_state['model'] = model
        st.session_state['le'] = le

    # if a model already trained in session
    if 'model' in st.session_state:
        st.info("A trained model is available in this session.")

    # Prediction UI
    st.subheader("Make a prediction")
    with st.form(key='predict_form'):
        temp_min_input = st.number_input("Minimum temperature (°C)", value=float(df['temp_min'].median()))
        temp_max_input = st.number_input("Maximum temperature (°C)", value=float(df['temp_max'].median()))
        precipitation_input = st.number_input("Precipitation (mm)", value=float(df['precipitation'].median()))
        wind_input = st.number_input("Wind speed (km/h)", value=float(df['wind'].median()))
        submit = st.form_submit_button("Predict")

    if submit:
        if 'model' not in st.session_state:
            st.warning("Model not trained in this session. Click 'Train model' first.")
        else:
            model = st.session_state['model']
            le = st.session_state['le']
            user_X = pd.DataFrame([[temp_min_input, temp_max_input, precipitation_input, wind_input]],
                                  columns=["temp_min","temp_max","precipitation","wind"])
            try:
                pred_enc = model.predict(user_X)
                pred = le.inverse_transform(pred_enc.astype(int))
                st.success(f"Predicted weather: {pred[0]}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:
    st.info("Modeling disabled — add the required columns and reload.")

st.markdown("---")
