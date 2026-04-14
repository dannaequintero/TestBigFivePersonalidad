import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(layout="wide")

st.title("Test de Personalidad Big Five")
st.write("Por favor, responde a las siguientes preguntas según tu nivel de acuerdo o desacuerdo.")

# Escala Likert
likert_options = {
    1: "Totalmente en desacuerdo",
    2: "En desacuerdo",
    3: "Ni de acuerdo ni en desacuerdo",
    4: "De acuerdo",
    5: "Totalmente de acuerdo"
}

# FORMULARIO
with st.form("big_five_form"):
    st.header("Preguntas")

    respuestas = []

    for i in range(1, 51):
        response = st.radio(
            f"Pregunta {i}",
            options=list(likert_options.keys()),
            format_func=lambda x: likert_options[x],
            key=f"q_{i}"
        )
        respuestas.append(response)

    submitted = st.form_submit_button("Ver resultado")

# RESULTADO
if submitted:

    st.write("### ¡Tus respuestas han sido enviadas!")

    try:
        # Cargar modelo
        with open('kmeans_model_bigfive_gm.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)

        # Cargar scaler
        with open('scaler_st_bigfive_gm.pkl', 'rb') as f:
            scaler = pickle.load(f)

    # Convertir respuestas en 5 grupos (10 preguntas cada uno)
ext = sum(respuestas[0:10])
neu = sum(respuestas[10:20])
agr = sum(respuestas[20:30])
con = sum(respuestas[30:40])
opn = sum(respuestas[40:50])

# Crear arreglo con 5 rasgos
datos = np.array([[ext, neu, agr, con, opn]])

# Escalar
datos_escalados = scaler.transform(datos)

        # Predecir cluster
        cluster = kmeans_model.predict(datos_escalados)[0]

        st.write(f"### Perteneces al Cluster: {cluster}")

        # Descripción simple de clusters
        descripciones = {
            0: "Personas introvertidas, analíticas y reservadas.",
            1: "Personas sociables, expresivas y extrovertidas.",
            2: "Personas organizadas, responsables y disciplinadas.",
            3: "Personas creativas, abiertas a nuevas experiencias.",
            4: "Personas emocionales y sensibles."
        }

        st.write("### Descripción de tu personalidad:")
        st.write(descripciones.get(cluster, "Descripción no disponible."))

    except Exception as e:
        st.error(f"Error al cargar el modelo o procesar datos: {e}")
