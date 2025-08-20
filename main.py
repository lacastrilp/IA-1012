import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.title("🔎 Clasificación o Regresión Automática")

# Subir archivo CSV
uploaded_file = st.file_uploader("📂 Sube un archivo CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Datos cargados:")
    st.dataframe(df.head())

    # Seleccionar variable objetivo y predictoras
    target = st.selectbox("🎯 Selecciona la variable objetivo (target)", df.columns)
    features = st.multiselect("📊 Selecciona las variables predictoras", [col for col in df.columns if col != target])

    if target and features:
        X = df[features]
        y = df[target]

        # Detectar si es clasificación o regresión
        n_unique = y.nunique()
        is_classification = (y.dtype == 'object') or (n_unique < 20)

        test_size = st.slider("📏 Tamaño del conjunto de prueba (%)", 10, 50, 30)

        # Dividir datos
        try:
            if is_classification:
                st.info("🔵 Se detectó un problema de **clasificación**.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
            else:
                st.info("🟢 Se detectó un problema de **regresión**.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )
        except ValueError as e:
            st.error(f"⚠️ Error al dividir los datos: {e}")
            st.stop()

        # Escalado solo para modelos que lo requieran
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.subheader("⚙️ Entrenando modelo...")

        if is_classification:
            # Modelos de clasificación
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()
            }
        else:
            # Modelos de regresión
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor()
            }

        results = {}

        for name, model in models.items():
            if "Regression" in name or "Regressor" in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[name] = {"MSE": mse, "R2": r2}
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)
                results[name] = {"Accuracy": acc}

        st.subheader("📊 Resultados")

        if is_classification:
            st.write(pd.DataFrame(results).T)
        else:
            st.write(pd.DataFrame(results).T)
