import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ===============================
# 📌 Función para cargar dataset
# ===============================
st.title("🔍 Clasificación Interactiva con ML")

st.sidebar.header("Carga de Datos")
option = st.sidebar.radio("Selecciona fuente de datos:", ["📂 Subir CSV", "🌐 Desde URL/GitHub"])

if option == "📂 Subir CSV":
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    url = st.sidebar.text_input("Ingresa URL de un CSV (GitHub/raw o nube)")
    if url:
        try:
            df = pd.read_csv(url)
        except Exception as e:
            st.error(f"Error al cargar: {e}")
            st.stop()
    else:
        st.stop()

st.subheader("📊 Vista previa de los datos")
st.dataframe(df.head())

# ===============================
# 📌 Preprocesamiento de Datos
# ===============================
st.subheader("🧹 Preprocesamiento de Datos")

# Eliminar duplicados
df = df.drop_duplicates()

# Rellenar valores nulos
df = df.fillna(df.mean(numeric_only=True))   # media en numéricas
df = df.fillna("desconocido")                # "desconocido" en categóricas

# Detección y tratamiento de atípicos en columnas numéricas
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
              np.where(df[col] > upper, upper, df[col]))

st.write("✅ Datos preprocesados (sin nulos, sin duplicados, atípicos tratados):")
st.dataframe(df.head())

# ===============================
# 📌 EDA rápido
# ===============================
st.subheader("📈 Análisis Exploratorio de Datos (EDA)")

if st.checkbox("Mostrar información general"):
    st.write(df.describe())
    st.write("Valores nulos por columna:", df.isnull().sum())

if st.checkbox("Distribución de variables"):
    col = st.selectbox("Selecciona columna para graficar", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

if st.checkbox("Matriz de correlación"):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===============================
# 📌 Selección de variables
# ===============================
st.sidebar.header("Configuración del modelo")

target = st.sidebar.selectbox("Selecciona la variable objetivo (target)", df.columns)
features = st.sidebar.multiselect("Selecciona las variables predictoras", [c for c in df.columns if c != target])

if not features:
    st.warning("⚠️ Selecciona al menos una variable predictora")
    st.stop()

# Convertir categóricas a numéricas (OneHot Encoding)
X = pd.get_dummies(df[features], drop_first=True)

# Asegurarse de que y sea categórica
y = df[target].astype(str)

# ===============================
# 📌 División de datos
# ===============================
test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 50, 30, step=5)

if y.nunique() < 20:
    stratify = y
else:
    stratify = None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=stratify
)

# ===============================
# 📌 Selección de modelo
# ===============================
st.sidebar.subheader("Modelo de Clasificación")
model_choice = st.sidebar.selectbox(
    "¿Qué modelo quieres usar?",
    ["Árbol de Decisión", "Naive Bayes", "K-Vecinos Cercanos (KNN)", "Máquina de Vectores de Soporte (SVC)", "Regresión Logística"]
)

# ===============================
# 📌 Hiperparámetros dinámicos
# ===============================
if model_choice == "Árbol de Decisión":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
    criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy", "log_loss"])
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

elif model_choice == "Naive Bayes":
    var_smoothing = st.sidebar.slider("Var smoothing (escala log)", -12, -3, -9)
    model = GaussianNB(var_smoothing=10**var_smoothing)

elif model_choice == "K-Vecinos Cercanos (KNN)":
    n_neighbors = st.sidebar.slider("Número de vecinos (k)", 1, 20, 5)
    weights = st.sidebar.selectbox("Ponderación", ["uniform", "distance"])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

elif model_choice == "Máquina de Vectores de Soporte (SVC)":
    C = st.sidebar.slider("Parámetro C", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    model = SVC(C=C, kernel=kernel)

elif model_choice == "Regresión Logística":
    C = st.sidebar.slider("Parámetro C (regularización inversa)", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("Iteraciones máximas", 100, 1000, 300)
    model = LogisticRegression(C=C, max_iter=max_iter)

# ===============================
# 📌 Entrenamiento
# ===============================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# 📌 Resultados
# ===============================
st.subheader("📌 Resultados del Modelo")

st.write("**Exactitud (Accuracy):**", accuracy_score(y_test, y_pred))
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
st.pyplot(fig)

# ===============================
# 📌 Visualización del Árbol
# ===============================
if model_choice == "Árbol de Decisión":
    st.subheader("🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in y.unique()], filled=True, ax=ax)
    st.pyplot(fig)
