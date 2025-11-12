# app/pages/metricas.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests, time
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

st.title(":material/leaderboard: Métricas relevantes")
st.write(
    "Mostramos **qué variables pesan más** en la predicción del próximo cierre. "
    "Usamos **importancia por permutación** (caída de R² al desordenar una variable), "
    "**coeficientes estandarizados (β)** para comparar pesos y **correlación** como referencia."
)

st.info(
    "**Permutación**: cuánto cae R² si desordeno una variable (más caída ⇒ más importante).  \n"
    "**Coeficientes estandarizados (β)**: impacto por 1 desvío estándar (+/− indica dirección).  \n"
    "**Correlación**: referencia intuitiva (no equivale a importancia del modelo)."
)

# -------------------- Helpers (idénticos a tus otras pages) --------------------
@st.cache_data(show_spinner=True, ttl=300)
def fetch_binance_klines_last_year(symbol: str) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    interval = "1h"
    now_ar = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires")
    end_utc = now_ar.tz_convert("UTC")
    start_utc = (now_ar - pd.Timedelta(days=365)).tz_convert("UTC")
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms   = int(end_utc.timestamp() * 1000)

    all_rows, cur = [], start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "interval": interval, "limit": 1000,
                  "startTime": cur, "endTime": end_ms}
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch: break
        all_rows.extend(batch)
        cur = batch[-1][6] + 1
        if len(batch) < 1000: break
        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    cols = ["open_time_ms","open","high","low","close","volume",
            "close_time_ms","quote_volume","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","taker_base","taker_quote"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["open_time_utc"]  = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    df["open_time_local"]  = df["open_time_utc"].dt.tz_convert("America/Argentina/Buenos_Aires")
    df["close_time_local"] = df["close_time_utc"].dt.tz_convert("America/Argentina/Buenos_Aires")
    df = df.sort_values("open_time_local").reset_index(drop=True)
    return df[["open_time_local","open_time_utc","open","high","low","close","volume",
               "close_time_local","close_time_utc","quote_volume","trades","taker_base","taker_quote"]]

def build_features(serie_close: pd.Series, horas: int) -> pd.DataFrame:
    feat = pd.DataFrame({"close": serie_close})
    feat["ret_1"]   = feat["close"].pct_change(1)
    feat["ret_3"]   = feat["close"].pct_change(3)
    feat["ret_6"]   = feat["close"].pct_change(6)
    feat["ma_5"]    = feat["close"].rolling(5,  min_periods=3).mean()
    feat["ma_10"]   = feat["close"].rolling(10, min_periods=5).mean()
    feat["ma_diff"] = feat["ma_5"] - feat["ma_10"]
    feat["vol_10"]  = feat["ret_1"].rolling(10, min_periods=5).std()
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
    feat["close_next"] = feat["close"].shift(-1)
    return feat

def pick_alpha(model_class, alphas, X, y, horas: int, feat_full: pd.DataFrame):
    tscv = TimeSeriesSplit(n_splits=3)
    best_a, best_r2 = None, -1e9
    for a in alphas:
        r2s = []
        for tr, te in tscv.split(X):
            m = model_class(alpha=a) if model_class is Ridge else model_class(alpha=a, max_iter=10000)
            m.fit(X.iloc[tr], y.iloc[tr])
            yhat = m.predict(X.iloc[te])
            if int(horas) == 48:
                close_now = feat_full.loc[X.iloc[te].index, "close"]
                y_true_p  = feat_full.loc[X.iloc[te].index, "close_next"]
                y_pred_p  = close_now * np.exp(yhat)
                r2s.append(r2_score(y_true_p, y_pred_p))
            else:
                r2s.append(r2_score(y.iloc[te], yhat))
        avg = float(np.mean(r2s)) if len(r2s) else -1e9
        if avg > best_r2:
            best_r2, best_a = avg, a
    return best_a, best_r2

# -------------------- Controles --------------------
symbols = ["BTCUSDT","ETHUSDT","SOLUSDT"]
c1, c2 = st.columns(2)
with c1:
    symbol = st.selectbox("Moneda", symbols, index=0)
with c2:
    horas = st.selectbox("Horizonte (horas)", [1, 4, 24, 48], index=0)

with st.spinner("Descargando último año (1h) desde Binance..."):
    df_api = fetch_binance_klines_last_year(symbol)
if df_api.empty:
    st.error("La API no devolvió datos."); st.stop()

df_sym = df_api.sort_values("open_time_local").set_index("open_time_local")
if not isinstance(df_sym.index, pd.DatetimeIndex):
    df_sym.index = pd.to_datetime(df_sym.index)

serie = df_sym["close"].resample(f"{int(horas)}H").last().dropna()
if len(serie) < 100:
    st.warning("No hay suficientes datos para este horizonte."); st.stop()

feat = build_features(serie, int(horas))
base_cols = ["ret_1","ret_3","ret_6","ma_5","ma_10","ma_diff","vol_10"]
extra_48  = ["ret_12","ret_24","ret_48","ma_24","ma_48","ma_d_24_48","vol_24","min_48","max_48"]
cols = base_cols + ([c for c in extra_48 if c in feat.columns] if int(horas) == 48 else [])

# target y matrices (idéntico criterio que en predicción)
if int(horas) == 48:
    feat["y_ret"] = np.log(feat["close_next"] / feat["close"])
    X_all = feat[cols].dropna()
    y_all = feat.loc[X_all.index, "y_ret"].dropna()
    X_all = X_all.loc[y_all.index]
else:
    X_all = feat[cols].dropna()
    y_all = feat.loc[X_all.index, "close_next"].dropna()
    X_all = X_all.loc[y_all.index]

if len(X_all) < 100:
    st.warning("No hay suficientes puntos tras construir variables."); st.stop()

# -------------------- Entrenamiento (cacheado en sesión) --------------------
key = f"exp_{symbol}_{horas}"
if key not in st.session_state:
    st.session_state[key] = {}
ss = st.session_state[key]

if "fitted" not in ss:
    # modelo igual que en predicción
    if int(horas) in [1, 4]:
        alpha, cv_r2 = pick_alpha(Ridge, [0.01, 0.1, 1.0, 10.0], X_all, y_all, int(horas), feat)
        model = Ridge(alpha=alpha or 1.0)
        model_txt = f"Ridge (α={model.alpha})"
    else:
        alpha, cv_r2 = pick_alpha(Lasso, [0.0001, 0.001, 0.01, 0.1], X_all, y_all, int(horas), feat)
        model = Lasso(alpha=alpha or 0.001, max_iter=10000)
        model_txt = f"Lasso (α={model.alpha})"

    split = int(len(X_all) * 0.8)
    X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
    y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if int(horas) == 48:
        close_now = feat.loc[X_test.index, "close"]
        y_true_p  = feat.loc[X_test.index, "close_next"]
        y_pred_p  = close_now * np.exp(y_pred)
        r2_holdout = r2_score(y_true_p, y_pred_p)
    else:
        r2_holdout = r2_score(y_test, y_pred)

    # guardo todo en sesión
    ss["fitted"] = True
    ss["model"] = model
    ss["model_txt"] = model_txt
    ss["cv_r2"] = cv_r2
    ss["X_train"], ss["X_test"] = X_train, X_test
    ss["y_train"], ss["y_test"] = y_train, y_test
    ss["r2_holdout"] = r2_holdout
    # permutación precomputada
    ss["perm"] = permutation_importance(model, X_test, y_test, scoring="r2", n_repeats=12, random_state=42)

model = ss["model"]
model_txt = ss["model_txt"]
cv_r2 = ss["cv_r2"]
X_train, X_test = ss["X_train"], ss["X_test"]
y_train, y_test = ss["y_train"], ss["y_test"]
r2_holdout = ss["r2_holdout"]
imp = ss["perm"]

st.caption(f"Modelo: **{model_txt}** · R² (holdout): **{r2_holdout*100:.2f}%** · CV R²: **{cv_r2*100:.2f}%**")

tab_perm, tab_coef, tab_corr = st.tabs(
    ["Importancia por permutación", "Coeficientes (β)", "Correlación con el target"]
)

# -------------------- 1) Permutación --------------------
with tab_perm:
    st.subheader("Importancia por permutación")
    order = np.argsort(imp.importances_mean)[::-1]
    df_imp = pd.DataFrame({
        "feature": np.array(cols)[order],
        "r2_drop": imp.importances_mean[order]
    })
    # % de caída relativo al R² base
    base = max(r2_holdout, 1e-9)
    df_imp["drop_pct"] = 100 * df_imp["r2_drop"] / base

    top_n = st.slider("Top N", 5, min(20, len(cols)), value=min(10, len(cols)))
    view = df_imp.head(top_n)

    chart = alt.Chart(view).mark_bar().encode(
        x=alt.X("r2_drop:Q", title="ΔR² (caída al permutar)"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=[
            "feature",
            alt.Tooltip("r2_drop:Q", title="ΔR²", format=".4f"),
            alt.Tooltip("drop_pct:Q", title="ΔR² (%)", format=".1f")
        ],
    ).properties(height=28*len(view))
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(view, use_container_width=True)

    st.download_button(
        "⬇️ Descargar importancias (CSV)",
        df_imp.to_csv(index=False).encode("utf-8"),
        file_name=f"importancias_{symbol}_{horas}h.csv",
        mime="text/csv"
    )

# -------------------- 2) Coeficientes estandarizados (β) --------------------
with tab_coef:
    st.subheader("Coeficientes estandarizados (β)")
    # β = coef * σx / σy  (para hacer comparables las escalas)
    sigma_x = X_train.std(numeric_only=True)
    sigma_y = float(np.std(y_train.values, ddof=1)) if len(y_train) > 1 else 1.0
    betas = pd.Series(model.coef_, index=X_train.columns) * (sigma_x / max(sigma_y, 1e-12))

    df_beta = (pd.DataFrame({"feature": betas.index, "beta_std": betas.values})
               .assign(abs_beta=lambda d: d["beta_std"].abs())
               .sort_values("abs_beta", ascending=False))

    top_n2 = st.slider("Top N coeficientes", 5, min(20, len(cols)), value=min(10, len(cols)), key="coefN")

    chart2 = alt.Chart(df_beta.head(top_n2)).mark_bar().encode(
        x=alt.X("beta_std:Q", title="β estandarizado (±)"),
        y=alt.Y("feature:N", sort='-x', title="Feature"),
        color=alt.condition("datum.beta_std >= 0", alt.value("#16a34a"), alt.value("#dc2626")),
        tooltip=["feature", alt.Tooltip("beta_std:Q", title="β", format=".4f")]
    ).properties(height=28*min(top_n2, len(df_beta)))
    st.altair_chart(chart2, use_container_width=True)
    st.dataframe(df_beta.head(top_n2)[["feature","beta_std"]], use_container_width=True)

# -------------------- 3) Correlación --------------------
with tab_corr:
    st.subheader("Correlación simple con el target")
    y_target = y_all.copy()
    df_c = pd.DataFrame({"target": y_target})
    for c in cols:
        df_c[c] = X_all[c].values
    corrs = df_c.corr(numeric_only=True)["target"].drop("target").sort_values(key=np.abs, ascending=False)
    df_corr = corrs.reset_index().rename(columns={"index": "feature", "target": "corr_target"})

    top_n3 = st.slider("Top N correlaciones", 5, min(20, len(cols)), value=min(10, len(cols)), key="corrN")
    chart3 = alt.Chart(df_corr.head(top_n3)).mark_bar().encode(
        x=alt.X("corr_target:Q", title="Correlación con el target"),
        y=alt.Y("feature:N", sort='-x', title="Feature"),
        tooltip=["feature", alt.Tooltip("corr_target:Q", format=".4f")]
    ).properties(height=28*min(top_n3, len(df_corr)))
    st.altair_chart(chart3, use_container_width=True)
    st.dataframe(df_corr.head(top_n3), use_container_width=True)

st.caption("Grupo 2 – Ciencia de Datos · UTN FRM 2025")