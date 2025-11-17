import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# ======================================================================================
# 1. ADVANCED DATA PROCESSING
# ======================================================================================

@st.cache_data
def process_data(df_raw):
    """
    Cleans the raw laptop data, performs feature engineering, and returns the 
    processed DataFrame and the label encoders.
    """
    df = df_raw.copy()
    
    # CORRECTED: Drop unnecessary columns including 'id' at the beginning
    df = df.drop(columns=["id", "Unnamed: 0", "Product"], errors='ignore')
    df.dropna(inplace=True)

    # --- Feature Engineering ---
    df['Ram'] = df['Ram'].str.replace("GB", "").astype(float)
    df['Weight'] = df['Weight'].str.replace("kg", "").astype(float)

    def extract_memory_components(mem):
        mem = mem.replace('TB', '000GB').replace('.0', '')
        components = {'SSD': 0, 'HDD': 0}
        parts = mem.split('+')
        for part in parts:
            part = part.strip()
            if 'SSD' in part:
                components['SSD'] += int(re.sub(r'\D', '', part))
            elif 'HDD' in part:
                components['HDD'] += int(re.sub(r'\D', '', part))
        return pd.Series(components)

    memory_components = df['Memory'].apply(extract_memory_components)
    df = pd.concat([df.drop(columns=['Memory']), memory_components], axis=1)

    df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
    df['IPS'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)
    
    try:
        resolutions = df['ScreenResolution'].str.extract(r'(\d+)x(\d+)').astype(float)
        resolutions = resolutions.fillna(0)
        df['PPI'] = ((resolutions[0]**2 + resolutions[1]**2)**0.5) / df['Inches']
        df['PPI'] = df['PPI'].fillna(df['PPI'].median())
    except Exception:
        df['PPI'] = 96.0

    df = df.drop(columns=['ScreenResolution', 'Inches'])

    df['Cpu_Brand'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
    df['Gpu_Brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df = df.drop(columns=['Cpu', 'Gpu'])

    def simplify_cpu(name):
        if name.startswith('Intel Core i'): return 'Intel Core i-Series'
        if name.startswith('Intel'): return 'Other Intel'
        if name.startswith('AMD'): return 'AMD'
        return 'Other'
    df['Cpu_Brand'] = df['Cpu_Brand'].apply(simplify_cpu)

    def simplify_os(name):
        if 'Windows' in name: return 'Windows'
        if 'Mac' in name or 'macOS' in name: return 'macOS'
        if 'Linux' in name: return 'Linux'
        return 'Other/No OS'
    df['OpSys'] = df['OpSys'].apply(simplify_os)

    # --- Encoding ---
    categorical_cols = ['Company', 'TypeName', 'OpSys', 'Cpu_Brand', 'Gpu_Brand']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders

# ======================================================================================
# 2. MODEL TRAINING & EVALUATION
# ======================================================================================

@st.cache_resource
def train_and_evaluate_models(df_processed):
    """Trains and evaluates multiple regressor models."""
    X = df_processed.drop(columns=['Price_euros'])
    y = np.log(df_processed['Price_euros'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'SVR': SVR(kernel='rbf')
    }

    trained_models = {}
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        trained_models[name] = model
        model_scores[name] = score
        
    return trained_models, model_scores

# ======================================================================================
# 3. STREAMLIT UI
# ======================================================================================

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Advanced Laptop Price Predictor")
st.write("This tool compares multiple models to predict laptop prices based on their specifications.")

try:
    df_raw = pd.read_csv('laptops.csv', encoding='latin1')
    df_processed, label_encoders = process_data(df_raw)
    trained_models, model_scores = train_and_evaluate_models(df_processed)

    st.success("Models trained successfully on the `laptops.csv` dataset!")

    # --- Model Performance Comparison ---
    st.header("📊 Model Performance Comparison")
    scores_df = pd.DataFrame(list(model_scores.items()), columns=['Model', 'R-squared Score']).sort_values('R-squared Score', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='R-squared Score', y='Model', data=scores_df, ax=ax, palette='viridis')
    ax.set_xlabel("R-squared (R²) Score - Higher is Better")
    ax.set_ylabel("Algorithm")
    ax.set_title("Comparison of Model Accuracy")
    ax.set_xlim(0, 1.0)
    st.pyplot(fig)

    # --- Display Best Model ---
    best_model_name = scores_df.iloc[0]['Model']
    best_model_score = scores_df.iloc[0]['R-squared Score']
    st.info(f"🏆 Best Performing Model: **{best_model_name}** with an R-squared score of **{best_model_score:.2f}**")


    # --- Prediction Interface ---
    st.header("🔮 Predict a Laptop's Price")
    col1, col2, col3 = st.columns(3)

    with col1:
        company = st.selectbox("Brand", label_encoders['Company'].classes_)
        type_name = st.selectbox("Type", label_encoders['TypeName'].classes_)
        ram = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64])
        opsys = st.selectbox("Operating System", label_encoders['OpSys'].classes_)

    with col2:
        weight = st.slider("Weight (kg)", 0.5, 4.5, 1.8)
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
        ips = st.selectbox("IPS Display", ["No", "Yes"])
        ppi = st.slider("Screen PPI (Pixels Per Inch)", 90, 350, 150)
        
    with col3:
        cpu_brand = st.selectbox("CPU Brand", label_encoders['Cpu_Brand'].classes_)
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1000, 2000])
        ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024])
        gpu_brand = st.selectbox("GPU Brand", label_encoders['Gpu_Brand'].classes_)
        
    # --- Model Selection and Prediction Button ---
    model_choice = st.selectbox("Select Model for Prediction", scores_df['Model'].tolist())

    if st.button("Predict Price", key="predict_button"):
        company_enc = label_encoders['Company'].transform([company])[0]
        type_name_enc = label_encoders['TypeName'].transform([type_name])[0]
        opsys_enc = label_encoders['OpSys'].transform([opsys])[0]
        cpu_brand_enc = label_encoders['Cpu_Brand'].transform([cpu_brand])[0]
        gpu_brand_enc = label_encoders['Gpu_Brand'].transform([gpu_brand])[0]
        touchscreen_enc = 1 if touchscreen == 'Yes' else 0
        ips_enc = 1 if ips == 'Yes' else 0

        # Define the columns in the exact order the model was trained on
        X_cols = df_processed.drop(columns=['Price_euros']).columns
        
        input_data = pd.DataFrame([[
            company_enc, type_name_enc, ram, weight, opsys_enc, hdd, ssd,
            touchscreen_enc, ips_enc, ppi, cpu_brand_enc, gpu_brand_enc
        ]], columns=X_cols)

        selected_model = trained_models[model_choice]
        predicted_log_price = selected_model.predict(input_data)[0]
        predicted_price = np.exp(predicted_log_price)

        st.success(f"💰 Predicted Price (using {model_choice}): **€{predicted_price:,.2f}**")

except FileNotFoundError:
    st.error("Error: `laptops.csv` not found. Please upload the correct dataset to enable the prediction tool.")
except Exception as e:
    st.error(f"An error occurred: {e}")
