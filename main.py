import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

# Clasificadores
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="Clasificadores Interactivos", layout="wide")
st.title("🧠 Clasificadores Interactivos con Streamlit")

# ===============================
# Cargar dataset
# ===============================
st.sidebar.header("📂 Datos")
uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
github_url = st.sidebar.text_input("O pega URL de un CSV en GitHub/Drive")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ CSV cargado desde tu PC")
elif github_url:
    try:
        data = pd.read_csv(github_url)
        st.success("✅ CSV cargado desde URL")
    except:
        st.error("❌ Error al cargar desde la URL")
        data = None
else:
    # Dataset simulado por defecto
    X, y = make_classification(
        n_samples=300, n_features=6, n_classes=2,
        n_informative=4, random_state=42
    )
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
    data["target"] = y
    st.info("📊 Usando dataset simulado (300x6)")

st.write("### Vista previa de los datos")
st.dataframe(data.head())

# ===============================
# Selección de variables
# ===============================
features = st.multiselect(
    "Selecciona las variables predictoras:",
    options=data.columns[:-1].tolist(),
    default=data.columns[:-1].tolist()
)

target = st.selectbox("Selecciona la variable objetivo:", options=data.columns)

X = data[features].values
y = data[target].values

# ===============================
# Train-test split
# ===============================
test_size = st.sidebar.slider("Proporción de Test (%)", 10, 50, 30, step=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y
)

# Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# Selección de modelo
# ===============================
st.sidebar.header("⚙️ Modelo")
model_choice = st.sidebar.selectbox(
    "Elige un clasificador:",
    ["Naive Bayes", "Árbol de Decisión", "KNN", "SVM", "Regresión Logística"]
)

if model_choice == "Naive Bayes":
    var_smoothing = st.sidebar.slider("var_smoothing", 1e-12, 1e-2, 1e-9, step=1e-12, format="%.0e")
    model = GaussianNB(var_smoothing=var_smoothing)

elif model_choice == "Árbol de Decisión":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
    criterion = st.sidebar.selectbox("Criterio", ["gini", "entropy", "log_loss"])
    model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

elif model_choice == "KNN":
    n_neighbors = st.sidebar.slider("Número de vecinos (k)", 1, 20, 5)
    weights = st.sidebar.selectbox("Pesos", ["uniform", "distance"])
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

elif model_choice == "SVM":
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    model = SVC(C=C, kernel=kernel, probability=True, random_state=42)

elif model_choice == "Regresión Logística":
    C = st.sidebar.slider("C", 0.01, 10.0, 1.0)
    penalty = st.sidebar.selectbox("Penalización", ["l2", "none"])
    solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)

# ===============================
# Entrenar y evaluar
# ===============================
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{acc:.2%}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
st.pyplot(fig)

# ===============================
# Visualización 2D con PCA
# ===============================
if len(features) > 2:
    st.write("### Reducción PCA para visualización 2D")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=y_pred, cmap="coolwarm", alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
    ax.add_artist(legend1)
    st.pyplot(fig)
