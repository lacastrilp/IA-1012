import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clasificadores ML", layout="wide")

st.title("🔎 Clasificación con varios algoritmos")

# ===============================
# Subida de dataset
# ===============================
st.sidebar.header("📂 Cargar Datos")

opcion_datos = st.sidebar.radio(
    "Elige cómo cargar los datos:",
    ("Dataset de ejemplo", "Subir archivo CSV", "Desde URL")
)

if opcion_datos == "Dataset de ejemplo":
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=300, n_features=6, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

elif opcion_datos == "Subir archivo CSV":
    archivo = st.sidebar.file_uploader("Sube un CSV", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo)
    else:
        st.stop()

else:  # Desde URL
    url = st.sidebar.text_input("Ingresa la URL del CSV (GitHub/raw o nube)")
    if url:
        try:
            df = pd.read_csv(url)
        except Exception as e:
            st.error(f"Error cargando dataset desde URL: {e}")
            st.stop()
    else:
        st.stop()

st.write("### Vista previa de los datos")
st.dataframe(df.head())

# ===============================
# Selección de variables
# ===============================
st.sidebar.header("⚙️ Configuración del modelo")

col_target = st.sidebar.selectbox("Selecciona la variable objetivo (target)", df.columns)

# Validación de variable objetivo
y = df[col_target].values
n_clases = len(np.unique(y))

if n_clases < 2:
    st.error("❌ El target debe tener al menos 2 clases.")
    st.stop()

if pd.api.types.is_numeric_dtype(y) and n_clases > 20:
    st.warning("⚠️ El target parece numérico con muchos valores distintos. Revisa si es realmente clasificación o regresión.")

clase_counts = pd.Series(y).value_counts()
if clase_counts.min() < 5:
    st.warning(f"⚠️ La clase con menos muestras tiene solo {clase_counts.min()} registros. Esto puede afectar el entrenamiento.")

# Features disponibles (excluyendo target)
features = st.sidebar.multiselect(
    "Selecciona las variables predictoras",
    [col for col in df.columns if col != col_target],
    default=[col for col in df.columns if col != col_target][:5]
)

if len(features) == 0:
    st.error("❌ Debes seleccionar al menos una variable predictora.")
    st.stop()

X = df[features].values

# ===============================
# División train/test
# ===============================
test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 50, 30, step=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y
)

# ===============================
# Selección del modelo
# ===============================
clasificador = st.sidebar.selectbox(
    "Elige el algoritmo",
    ["Naive Bayes", "Árbol de Decisión", "KNN", "SVC", "Regresión Logística"]
)

# ===============================
# Hiperparámetros por modelo
# ===============================
if clasificador == "Naive Bayes":
    var_smoothing = st.sidebar.number_input("Var smoothing", 1e-12, 1e-6, 1e-9, format="%.1e")
    model = GaussianNB(var_smoothing=var_smoothing)

elif clasificador == "Árbol de Decisión":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
    criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy"])
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

elif clasificador == "KNN":
    n_neighbors = st.sidebar.slider("Número de vecinos", 1, 20, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

elif clasificador == "SVC":
    C = st.sidebar.slider("Parámetro C", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    model = SVC(C=C, kernel=kernel, probability=True, random_state=42)

elif clasificador == "Regresión Logística":
    C = st.sidebar.slider("Parámetro C", 0.01, 10.0, 1.0)
    max_iter = st.sidebar.slider("Máx. iteraciones", 100, 1000, 200, step=50)
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

# ===============================
# Entrenamiento y evaluación
# ===============================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader("📊 Resultados")
st.write("**Exactitud:**", accuracy_score(y_test, y_pred))
st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))

# Matriz de confusión
st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)
