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
st.write("Se utiliza un dataset simulado con 300 muestras y 6 columnas.")

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------
# Selección del clasificador
# ----------------------------
if modelo == "Naive Bayes":
    clf = GaussianNB()
elif modelo == "Árbol de Decisión":
    clf = DecisionTreeClassifier(random_state=42)
elif modelo == "K-Vecinos Cercanos (KNN)":
    clf = KNeighborsClassifier()
elif modelo == "Máquina de Vectores de Soporte (SVC)":
    clf = SVC(probability=True, random_state=42)
elif modelo == "Regresión Logística":
    clf = LogisticRegression(max_iter=1000, random_state=42)

# Entrenar
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ----------------------------
# Resultados
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Resultados")
st.write(f"**Exactitud (Accuracy):** {accuracy:.2f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues")
st.pyplot(fig)
