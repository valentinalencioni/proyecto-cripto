import streamlit as st
import pandas as pd
import altair as alt

st.title("📊 Exploración de datos")
st.write("Explorá la evolución histórica de precios por criptomoneda y acotá el rango temporal que quieras analizar.")

# --- Usar el CSV ya preprocesado por tu EDA (sin reprocesar acá) ---
df = pd.read_csv("criptos_final.csv")

# Asegurar columna temporal utilizable
if "open_time_local" in df.columns:
    df["open_time_local"] = pd.to_datetime(df["open_time_local"])
elif "open_time_utc" in df.columns:
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    df["open_time_local"] = df["open_time_utc"].dt.tz_convert("America/Argentina/Buenos_Aires")
else:
    st.error("No se encontró 'open_time_local' ni 'open_time_utc' en el dataset.")
    st.stop()

# --- 1) Selector de criptomoneda ---
symbols = sorted(df["symbol"].unique())
symbol = st.selectbox("Elegí la criptomoneda", symbols)
df_symbol = df[df["symbol"] == symbol].copy()

# --- 2) Gráfico histórico general (Altair + hover) ---
hover = alt.selection_single(fields=["open_time_local"], nearest=True, on="mouseover", empty="none")
base = alt.Chart(df_symbol).encode(
    x=alt.X("open_time_local:T", title="Fecha (hora local Argentina)"),
    y=alt.Y("close:Q", title="Precio de cierre (USDT)")
)
hist_chart = (
    base.mark_line(color="#1f77b4").interactive()
    + base.mark_circle(size=60, color="#ff7f0e").encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
      ).add_selection(hover)
    + base.mark_rule(color="gray").encode(
        opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
        tooltip=[alt.Tooltip("open_time_local:T", title="Fecha"),
                 alt.Tooltip("close:Q", title="Cierre (USDT)", format=".2f")]
      )
).properties(height=420)

st.subheader(f"Evolución histórica de {symbol}")
st.altair_chart(hist_chart, use_container_width=True)

# --- 3) Selector de rango de fechas ---
min_date = df_symbol["open_time_local"].min().date()
max_date = df_symbol["open_time_local"].max().date()
rango = st.date_input("Seleccioná un rango de fechas",
                      value=(min_date, max_date),
                      min_value=min_date, max_value=max_date)
if isinstance(rango, (list, tuple)) and len(rango) == 2:
    start_date, end_date = rango
else:
    start_date, end_date = min_date, max_date

# --- 4) Métricas y gráfico del rango (sin HTML; color con sintaxis de Streamlit) ---
df_rango = df_symbol[
    (df_symbol["open_time_local"].dt.date >= start_date) &
    (df_symbol["open_time_local"].dt.date <= end_date)
].copy()

st.subheader(f"📈 Análisis de {symbol} entre {start_date} y {end_date}")

if df_rango.empty:
    st.warning("No hay datos en el rango seleccionado.")
else:
    precio_inicial = float(df_rango["close"].iloc[0])
    precio_final = float(df_rango["close"].iloc[-1])
    variacion_pct = (precio_final / precio_inicial - 1) * 100
    variacion_decimal= round(variacion_pct, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precio inicial", f"{precio_inicial:,.2f} USDT")
    c2.metric("Precio final", f"{precio_final:,.2f} USDT", variacion_decimal, delta_color="normal")

    color_token = "green" if variacion_pct >= 0 else "red"


    # Gráfico del rango con color según signo
    line_color = "#16a34a" if variacion_pct >= 0 else "#dc2626"
    rango_chart = (
        alt.Chart(df_rango)
        .mark_line(color=line_color, point=True)
        .encode(
            x=alt.X("open_time_local:T", title="Fecha"),
            y=alt.Y("close:Q", title="Precio de cierre (USDT)"),
            tooltip=[alt.Tooltip("open_time_local:T", title="Fecha"),
                     alt.Tooltip("close:Q", title="Cierre (USDT)", format=".2f")],
        )
        .properties(height=350, title=f"Comportamiento de {symbol} en el rango seleccionado")
        .interactive()
    )
    st.altair_chart(rango_chart, use_container_width=True)