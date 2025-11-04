# pages/exploracion_datos.py
# Arriba: último año completo (1h) desde Binance (UTC -> AR)
# Abajo: subrango siempre dentro de ese año

import streamlit as st
import pandas as pd
import altair as alt
import requests
import time
from typing import Tuple

st.title(":material/finance: Exploración de datos")
st.write(
    "Se descarga el **último año** de datos (velas de 1h) desde la API de Binance en UTC y se convierten a "
    "hora local de Argentina. Abajo podés elegir un **subrango** dentro de ese año."
)

# --------------------------- Utilidades ---------------------------

@st.cache_data(show_spinner=True, ttl=300)
def fetch_binance_klines_last_year(symbol: str) -> pd.DataFrame:
    """
    Descarga velas 1h del último año para 'symbol' desde /api/v3/klines.
    Pagina en lotes de 1000. Convierte UTC -> America/Argentina/Buenos_Aires.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    interval = "1h"

    now_ar = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires")
    end_utc = now_ar.tz_convert("UTC")
    start_utc = (now_ar - pd.Timedelta(days=365)).tz_convert("UTC")

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms   = int(end_utc.timestamp() * 1000)

    all_rows, cur = [], start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "startTime": cur,
            "endTime": end_ms
        }
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)

        last_close = batch[-1][6]   # close_time ms
        cur = last_close + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    cols_raw = [
        "open_time_ms","open","high","low","close","volume",
        "close_time_ms","quote_volume","trades","taker_base","taker_quote","ignore"
    ]
    df = pd.DataFrame(all_rows, columns=cols_raw)

    # tipos
    for c in ["open","high","low","close","volume","quote_volume","taker_base","taker_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)

    # tiempos
    df["open_time_utc"]  = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    df["open_time_local"]  = df["open_time_utc"].dt.tz_convert("America/Argentina/Buenos_Aires")
    df["close_time_local"] = df["close_time_utc"].dt.tz_convert("America/Argentina/Buenos_Aires")

    df["symbol"] = symbol
    df = df.sort_values("open_time_local").reset_index(drop=True)

    return df[[
        "symbol",
        "open_time_local","open_time_utc",
        "open","high","low","close","volume",
        "close_time_local","close_time_utc",
        "quote_volume","trades","taker_base","taker_quote"
    ]]

# --------------------------- Controles ---------------------------

symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
symbol = st.selectbox("Criptomoneda", symbols, index=0)

# --------------------------- Descarga (último año) ---------------------------

with st.spinner("Descargando último año (1h) desde Binance..."):
    df_all = fetch_binance_klines_last_year(symbol)

if df_all.empty:
    st.warning("La API no devolvió datos.")
    st.stop()

# --------------------------- Gráfico superior: histórico (año completo) ---------------------------

st.subheader(f"Histórico · {symbol} · 1h (últimos 12 meses)")
hover_all = alt.selection_single(fields=["open_time_local"], nearest=True, on="mouseover", empty="none")

base_all = alt.Chart(df_all).encode(
    x=alt.X("open_time_local:T", title="Fecha (hora local AR)"),
    y=alt.Y("close:Q", title="Precio de cierre (USDT)")
)

chart_all = (
    base_all.mark_line(color="#1f77b4").interactive()
    + base_all.mark_circle(size=50, color="#ff7f0e").encode(
        opacity=alt.condition(hover_all, alt.value(1), alt.value(0))
      ).add_selection(hover_all)
    + base_all.mark_rule(color="gray").encode(
        opacity=alt.condition(hover_all, alt.value(0.3), alt.value(0)),
        tooltip=[
            alt.Tooltip("open_time_local:T", title="Fecha"),
            alt.Tooltip("close:Q", title="Cierre (USDT)", format=".2f")
        ]
      )
).properties(height=420)

st.altair_chart(chart_all, use_container_width=True)

# --------------------------- Subrango (siempre dentro del año) ---------------------------

min_date = df_all["open_time_local"].min().date()
max_date = df_all["open_time_local"].max().date()

rango_sub = st.date_input(
    "Elegí un subrango dentro del último año",
    value=(max(min_date, max_date - pd.Timedelta(days=30)), max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(rango_sub, (list, tuple)) and len(rango_sub) == 2:
    sub_start, sub_end = rango_sub
else:
    sub_start, sub_end = min_date, max_date

df_sub = df_all[
    (df_all["open_time_local"].dt.date >= sub_start) &
    (df_all["open_time_local"].dt.date <= sub_end)
].copy()

st.subheader(f"Subrango seleccionado · {symbol} · {sub_start} → {sub_end}")

if df_sub.empty:
    st.warning("No hay datos en el subrango seleccionado.")
    st.stop()

precio_inicial = float(df_sub["close"].iloc[0])
precio_final   = float(df_sub["close"].iloc[-1])
variacion_pct  = (precio_final / precio_inicial - 1) * 100
variacion_decimal = round(variacion_pct, 2)

c1, c2, _ = st.columns(3)
c1.metric("Precio inicial", f"{precio_inicial:,.2f} USDT")
c2.metric("Precio final",   f"{precio_final:,.2f} USDT", variacion_decimal, delta_color="normal")

line_color = "#16a34a" if variacion_pct >= 0 else "#dc2626"

hover_sub = alt.selection_single(fields=["open_time_local"], nearest=True, on="mouseover", empty="none")
chart_sub = (
    alt.Chart(df_sub)
    .mark_line(color=line_color, point=True)
    .encode(
        x=alt.X("open_time_local:T", title="Fecha (hora local AR)"),
        y=alt.Y("close:Q", title="Precio de cierre (USDT)"),
        tooltip=[
            alt.Tooltip("open_time_local:T", title="Fecha"),
            alt.Tooltip("close:Q", title="Cierre (USDT)", format=".2f")
        ],
    )
    .add_selection(hover_sub)
    .properties(height=360, title="Comportamiento en el subrango")
    .interactive()
)
st.altair_chart(chart_sub, use_container_width=True)