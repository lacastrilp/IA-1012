import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ===============================
# 1. CARGA DE DATOS
# ===============================
st.title("📊 Clasificación Interactiva con Varios Modelos")

st.sidebar.header("Carga de datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
url = st.sidebar.text_input("O pega la URL de un CSV en GitHub/Cloud:")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif url:
    df = pd.read_csv(url)
else:
    st.warning("Por favor sube un archivo o pega un enlace de un CSV.")
    st.stop()

st.write("### Vista previa de los datos")
st.dataframe(df.head())

# ===============================
# 2. EDA
# ===============================
st.write("## 🔍 Análisis Exploratorio de Datos (EDA)")

st.write("**Información general:**")
st.write(df.describe())

st.write("**Valores nulos:**")
st.write(df.isnull().sum())

st.write("**Distribución de las variables numéricas:**")
st.bar_chart(df.select_dtypes(include=np.number).iloc[:, :5])  # primeras 5

# Heatmap de correlación
if df.select_dtypes(include=np.number).shape[1] > 1:
    st.write("**Mapa de correlación**")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===============================
# 3. SELECCIÓN DE VARIABLES
# ===============================
st.sidebar.header("Selección de variables")
target_col = st.sidebar.selectbox("Selecciona la variable objetivo", df.columns)
feature_cols = st.sidebar.multiselect(
    "Selecciona las variables predictoras",
    [col for col in df.columns if col != target_col]
)

if not feature_cols:
    st.warning("Debes seleccionar al menos una variable predictora.")
    st.stop()

X = df[feature_cols]
y = df[target_col]

# ===============================
# 4. DIVISIÓN DE DATOS
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ===============================
# 5. MODELOS DISPONIBLES
# ===============================
st.sidebar.header("Modelos de clasificación")
model_choice = st.sidebar.selectbox(
    "Selecciona un modelo",
    ["Árbol de Decisión", "Naive Bayes", "K-Vecinos Cercanos (KNN)", "Máquina de Vectores de Soporte (SVC)", "Regresión Logística"]
)

# ===============================
# 6. ENTRENAMIENTO SEGÚN MODELO
# ===============================
if model_choice == "Árbol de Decisión":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 3)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write("## 🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, feature_names=feature_cols, class_names=[str(c) for c in y.unique()],
              filled=True, rounded=True, fontsize=8, ax=ax)
    st.pyplot(fig)

elif model_choice == "Naive Bayes":
    clf = GaussianNB(var_smoothing=1e-9)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

elif model_choice == "K-Vecinos Cercanos (KNN)":
    k = st.sidebar.slider("Número de vecinos (k)", 1, 20, 5)
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

elif model_choice == "Máquina de Vectores de Soporte (SVC)":
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = st.sidebar.slider("Parámetro C", 0.01, 10.0, 1.0)
    clf = SVC(kernel=kernel, C=C)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

elif model_choice == "Regresión Logística":
    max_iter = st.sidebar.slider("Número máximo de iteraciones", 50, 500, 100)
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

# ===============================
# 7. RESULTADOS
# ===============================
st.write("## 📈 Resultados del modelo")
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)
