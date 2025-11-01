import pandas as pd
df= pd.read_csv("criptos_final.csv")
df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)
df["open_time_local"] = df["open_time_utc"].dt.tz_convert("America/Argentina/Buenos_Aires")
df = df.sort_values("open_time_local")

# --- Rango temporal ---
rango_temporal = pd.DataFrame({
    "Fecha mínima (local)": [df["open_time_local"].min()],
    "Fecha máxima (local)": [df["open_time_local"].max()]
})

# --- Limpieza de columnas irrelevantes ---
cols_to_drop = [
    'open_time_ms', 'close_time_ms',
    'quote_volume', 'trades', 'taker_base', 'taker_quote'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 4) Resumen por símbolo
symbols_present = df['symbol'].unique().tolist()
counts = df['symbol'].value_counts()
ranges = df.groupby('symbol').agg(['min','max'])

return df, counts, ranges