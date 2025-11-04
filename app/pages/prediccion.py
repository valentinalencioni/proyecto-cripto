
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

st.title(":material/price_check: Predicción del próximo cierre")
st.write(
    "Elegí la **moneda** y el **horizonte (en horas)**. "
    "Mostramos el **próximo precio de cierre estimado** y la **precisión (R²)** del modelo, "
    "siguiendo los mismos métodos y variables del notebook."
)

# --- Dataset ya preprocesado (no se reprocesa el EDA aquí) ---
DF_PATH = "criptos_final.csv"
df = pd.read_csv(DF_PATH)

# Columna temporal utilizable
if "open_time_local" in df.columns:
    df["open_time_local"] = pd.to_datetime(df["open_time_local"])
    time_col = "open_time_local"
elif "open_time_utc" in df.columns:
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
    time_col = "open_time_utc"
else:
    st.error("No se encontró columna temporal en el dataset.")
    st.stop()

# --- Controles ---
symbols = sorted(df["symbol"].unique())
c1, c2 = st.columns(2)
with c1:
    symbol = st.selectbox("Moneda", symbols, index=0)
with c2:
    horas = st.selectbox("Horizonte (horas)", [1, 4, 24, 48], index=0)

# --- Serie por símbolo y resample en horas (último close del bloque) ---
df_sym = (
    df[df["symbol"] == symbol]
    .sort_values(time_col)
    .set_index(time_col)
)
if not isinstance(df_sym.index, pd.DatetimeIndex):
    df_sym.index = pd.to_datetime(df_sym.index)

rule = f"{int(horas)}H"
serie = df_sym["close"].resample(rule).last().dropna()

if len(serie) < 100:
    st.warning("No hay suficientes datos para este horizonte.")
    st.stop()

# --- Construcción de features (mismas del notebook) ---
feat = pd.DataFrame({"close": serie})
feat["ret_1"]   = feat["close"].pct_change(1)
feat["ret_3"]   = feat["close"].pct_change(3)
feat["ret_6"]   = feat["close"].pct_change(6)
feat["ma_5"]    = feat["close"].rolling(5,  min_periods=3).mean()
feat["ma_10"]   = feat["close"].rolling(10, min_periods=5).mean()
feat["ma_diff"] = feat["ma_5"] - feat["ma_10"]
feat["vol_10"]  = feat["ret_1"].rolling(10, min_periods=5).std()

# Mejora para 48h: features de ventana larga
if int(horas) == 48:
    feat["ret_12"] = feat["close"].pct_change(12)
    feat["ret_24"] = feat["close"].pct_change(24)
    feat["ret_48"] = feat["close"].pct_change(48)

    feat["ma_24"]  = feat["close"].rolling(24, min_periods=12).mean()
    feat["ma_48"]  = feat["close"].rolling(48, min_periods=24).mean()
    feat["ma_d_24_48"] = feat["ma_24"] - feat["ma_48"]

    feat["vol_24"] = feat["ret_1"].rolling(24, min_periods=12).std()

    feat["min_48"] = feat["close"].rolling(48, min_periods=24).min()
    feat["max_48"] = feat["close"].rolling(48, min_periods=24).max()

# Target de referencia (nivel)
feat["close_next"] = feat["close"].shift(-1)

# Selección de columnas de entrada
base_cols = ["ret_1","ret_3","ret_6","ma_5","ma_10","ma_diff","vol_10"]
extra_48  = ["ret_12","ret_24","ret_48","ma_24","ma_48","ma_d_24_48","vol_24","min_48","max_48"]
if int(horas) == 48:
    cols = base_cols + [c for c in extra_48 if c in feat.columns]
else:
    cols = base_cols

# --- Preparación de X, y según horizonte ---
if int(horas) == 48:
    # Mejorar 48h: predecir log-retorno y reconstruir a precio
    feat["y_ret"] = np.log(feat["close_next"] / feat["close"])
    X_all = feat[cols].dropna()
    y_all = feat.loc[X_all.index, "y_ret"].dropna()
    X_all = X_all.loc[y_all.index]
else:
    # Otros horizontes: nivel directo
    X_all = feat[cols].dropna()
    y_all = feat.loc[X_all.index, "close_next"].dropna()
    X_all = X_all.loc[y_all.index]

if len(X_all) < 100:
    st.warning("No hay suficientes puntos tras construir las variables.")
    st.stop()

# --- Elección de modelo y alpha (TimeSeriesSplit simple) ---
def pick_alpha(model_class, alphas, X, y, splits=3):
    tscv = TimeSeriesSplit(n_splits=splits)
    best_a, best_r2 = None, -1e9
    for a in alphas:
        r2s = []
        for tr_idx, te_idx in tscv.split(X):
            m = model_class(alpha=a)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            yhat = m.predict(X.iloc[te_idx])
            if int(horas) == 48:
                # Reconstruir a precio para evaluar R² sobre nivel (consistente con notebook)
                close_now = feat.loc[X.iloc[te_idx].index, "close"]
                y_true_price = feat.loc[X.iloc[te_idx].index, "close_next"]
                y_pred_price = close_now * np.exp(yhat)
                r2s.append(r2_score(y_true_price, y_pred_price))
            else:
                r2s.append(r2_score(y.iloc[te_idx], yhat))
        avg = float(np.mean(r2s)) if len(r2s) else -1e9
        if avg > best_r2:
            best_r2, best_a = avg, a
    return best_a

if int(horas) in [1, 4]:
    alpha = pick_alpha(Ridge, [0.01, 0.1, 1.0, 10.0], X_all, y_all)
    model = Ridge(alpha=alpha or 1.0)
    nombre_modelo = f"Ridge (α={model.alpha})"
else:
    alpha = pick_alpha(Lasso, [0.0001, 0.001, 0.01, 0.1], X_all, y_all)
    model = Lasso(alpha=alpha or 0.001, max_iter=10000)
    nombre_modelo = f"Lasso (α={model.alpha})"

# --- Holdout temporal (20%) para precisión R² ---
split = int(len(X_all) * 0.8)
X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

if int(horas) == 48:
    close_now_test = feat.loc[X_test.index, "close"]
    y_true_price_test = feat.loc[X_test.index, "close_next"]
    y_pred_price_test = close_now_test * np.exp(y_pred_test)
    r2 = r2_score(y_true_price_test, y_pred_price_test)
else:
    y_true_price_test = y_test
    y_pred_price_test = y_pred_test
    r2 = r2_score(y_test, y_pred_test)

r2_pct = max(min(r2 * 100.0, 100.0), -100.0)

# --- Reentrenar con todo y predecir próximo cierre ---
model.fit(X_all, y_all)
X_last = X_all.iloc[[-1]]
last_close = float(feat["close"].iloc[-1])

if int(horas) == 48:
    y_hat_ret = float(model.predict(X_last)[0])
    y_hat = float(last_close * np.exp(y_hat_ret))
else:
    y_hat = float(model.predict(X_last)[0])

delta_pct = (y_hat / last_close - 1.0) * 100.0

# --- Métricas principales (UX) ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Modelo", nombre_modelo)
m2.metric("Horizonte", f"{int(horas)} h")
m3.metric("Último cierre", f"{last_close:,.2f} USDT")
m4.metric("Próximo cierre estimado", f"{y_hat:,.2f} USDT", f"{delta_pct:.2f}%")

st.metric("Precisión (R²)", f"{r2_pct:.2f} %")
st.caption("La precisión (R²) se calcula con validación temporal (20% final). La predicción corresponde al **próximo bloque** del horizonte elegido.")

# --- Gráfico 1: histórico + línea del último cierre + punto de predicción ---
tail_n = min(200, len(serie))
hist = serie.tail(tail_n).reset_index()
hist.columns = ["timestamp", "close"]
next_ts = hist["timestamp"].iloc[-1] + pd.Timedelta(hours=int(horas))

base = alt.Chart(hist).encode(
    x=alt.X("timestamp:T", title="Fecha"),
    y=alt.Y("close:Q", title="Precio (USDT)")
)
line_hist = base.mark_line(point=True)

rule_last = alt.Chart(pd.DataFrame({"y": [last_close]})).mark_rule(
    color="#9ca3af", strokeDash=[6,4]
).encode(y="y:Q")

pred_df = pd.DataFrame({"timestamp": [next_ts], "close": [y_hat]})
point_pred = alt.Chart(pred_df).mark_point(size=120, filled=True, color="#ff7f0e").encode(
    x="timestamp:T",
    y="close:Q",
    tooltip=[alt.Tooltip("timestamp:T", title="Próximo bloque"),
             alt.Tooltip("close:Q", title="Cierre estimado", format=".2f")]
)

st.altair_chart((line_hist + rule_last + point_pred).properties(
    height=320, title=f"{symbol} · Horizonte {int(horas)}h"
), use_container_width=True)

# --- Gráfico 2: Real vs. Predicción en el holdout temporal ---
# Para cada índice de prueba, la etiqueta real es el cierre del siguiente bloque.
# Construimos la serie temporal del "siguiente bloque" para graficar correctamente.
ts_next = pd.Series(X_test.index + pd.Timedelta(hours=int(horas)), index=X_test.index)

df_eval = pd.DataFrame({
    "timestamp": ts_next.values,
    "Real": y_true_price_test.values,
    "Predicción": y_pred_price_test
}).sort_values("timestamp")

df_eval_melt = df_eval.melt("timestamp", var_name="Serie", value_name="Precio")

chart_eval = (
    alt.Chart(df_eval_melt)
    .mark_line(point=True)
    .encode(
        x=alt.X("timestamp:T", title="Fecha (siguiente bloque)"),
        y=alt.Y("Precio:Q", title="Precio (USDT)"),
        color=alt.Color("Serie:N", scale=alt.Scale(range=["#1f77b4", "#ff7f0e"])),
        tooltip=[alt.Tooltip("timestamp:T", title="Fecha"),
                 alt.Tooltip("Serie:N", title="Serie"),
                 alt.Tooltip("Precio:Q", title="Precio", format=".2f")]
    )
    .properties(height=320, title="Real vs. Predicción (holdout)")
    .interactive()
)
st.altair_chart(chart_eval, use_container_width=True)