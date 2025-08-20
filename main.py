import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Crear dataset simulado
# ----------------------------
X, y = make_classification(
    n_samples=300,
    n_features=6,
    n_informative=4,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(1, 7)])
df["Target"] = y

# ----------------------------
# Interfaz Streamlit
# ----------------------------
st.title("Comparación de Modelos de Clasificación")
st.write("Dataset simulado con 300 muestras y 6 columnas.")

# Mostrar dataset
if st.checkbox("Mostrar datos simulados"):
    st.dataframe(df.head(10))

# Selección de modelo
modelo = st.selectbox(
    "Seleccione un modelo de clasificación:",
    ["Naive Bayes", "Árbol de Decisión", "K-Vecinos Cercanos (KNN)", 
     "Máquina de Vectores de Soporte (SVC)", "Regresión Logística"]
)

# División en train/test
test_size = st.slider("Porcentaje de datos para prueba (%)", 10, 50, 30, step=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42, stratify=y
)

# ----------------------------
# Selección de hiperparámetros
# ----------------------------
st.subheader("Configuración de Hiperparámetros")

if modelo == "Naive Bayes":
    var_smoothing = st.number_input("Var smoothing (escala de suavizado)", 
                                     value=1e-9, format="%.1e")
    clf = GaussianNB(var_smoothing=var_smoothing)

elif modelo == "Árbol de Decisión":
    max_depth = st.slider("Profundidad máxima", 1, 20, 5)
    criterion = st.selectbox("Criterio de división", ["gini", "entropy", "log_loss"])
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)

elif modelo == "K-Vecinos Cercanos (KNN)":
    n_neighbors = st.slider("Número de vecinos (k)", 1, 20, 5)
    weights = st.selectbox("Tipo de ponderación", ["uniform", "distance"])
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

elif modelo == "Máquina de Vectores de Soporte (SVC)":
    kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
    C = st.slider("Parámetro C", 0.01, 10.0, 1.0)
    gamma = st.selectbox("Gamma", ["scale", "auto"])
    clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)

elif modelo == "Regresión Logística":
    penalty = st.selectbox("Tipo de regularización", ["l2", "l1", "elasticnet", "none"])
    C = st.slider("Inverso de la regularización (C)", 0.01, 10.0, 1.0)
    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga", "newton-cg"])
    clf = LogisticRegression(penalty=penalty, C=C, solver=solver,
                             max_iter=1000, random_state=42)

# ----------------------------
# Entrenamiento y resultados
# ----------------------------
if st.button("Entrenar modelo"):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Resultados")
    st.write(f"**Exactitud (Accuracy):** {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
