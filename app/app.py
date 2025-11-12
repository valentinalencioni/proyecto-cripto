import streamlit as st

st.set_page_config(page_title="CriptoPredictor", page_icon=":material/wallet:", layout="wide")

# --- Definir las páginas usando rutas RELATIVAS ---
home = st.Page(
    "pages/home.py",
    title="Home",
    icon=":material/home:"
)

exploracion = st.Page(
    "pages/exploracion_datos.py",
    title="Exploración de datos",
    icon=":material/insights:"
)

prediccion = st.Page(
    "pages/prediccion.py",
    title="Predicción de valor",
    icon=":material/trending_up:"
)

metricas = st.Page(
    "pages/metricas.py",
    title="Métricas relevantes",
    icon=":material/leaderboard:"
)

# --- Registrar la navegación ---
pg = st.navigation({
    "App": [home, exploracion, prediccion, metricas]
})

pg.run()