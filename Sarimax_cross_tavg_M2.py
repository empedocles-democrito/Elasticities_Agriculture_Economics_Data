import pandas as pd
import numpy as np
import statsmodels.api as sm
from pmdarima import auto_arima

# ============================
# Cargar datos
# ============================
df = pd.read_csv(
    "/Users/rgarrido/Documents/Centro de Data Science/Difusion/Elasticidades/Modelo Armington/data/Base_de_Calibracion_q_en_ton_por_usd_utf8.csv"
)

df.columns = [c.lower().strip() for c in df.columns]

# Asegurar fecha correcta
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["country", "date"])

countries = df["country"].unique()
print("===============================================")
print("Países detectados:", countries)
print("===============================================")

# =============================================================
# Construcción del índice competidor
# =============================================================
df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")

competitor_idx = (
    df.groupby("date")
      .apply(lambda g: np.exp(np.log(g["log_price"]).mean()))
      .rename("comp_price_index")
)

df = df.merge(competitor_idx, on="date", how="left")

# Fix: crear log del índice competidor
df["log_price_competitor_index"] = np.log(df["comp_price_index"])

# precio relativo: log(P_own / P_comp)
df["log_price_relative"] = df["log_price"] - df["log_price_competitor_index"]

# =============================
# Dummies mensuales NUMÉRICAS
# =============================
df["month"] = df["date"].dt.month
dummies = pd.get_dummies(df["month"], prefix="M").astype(float)
df = pd.concat([df, dummies], axis=1)

# =============================
# Función para estimar un país
# =============================
def run_country(c):
    print("\n===============================================")
    print(f" Estimando para: {c}")
    print("===============================================")

    d = df[df["country"] == c].copy()

    # Endógena
    y = pd.to_numeric(d["log_share"], errors="coerce")

    # =============================
    # Regresores por país
    # =============================
    if c == "mexico":
        print("   ⚠ México → modelo M2 (precio relativo + dummies)")
        ex_cols = ["log_price_relative"] + [f"M_{i}" for i in range(2, 13)]
    else:
        ex_cols = ["log_price", "log_price_competitor_index", "tavg"] + [f"M_{i}" for i in range(2, 13)]

    # Asegurar columnas
    for col in ex_cols:
        if col not in d.columns:
            print(f"⚠ WARNING: columna faltante: {col}, se crea")
            d[col] = 0.0

    X = d[ex_cols].copy()
    X["const"] = 1.0

    print("✔ Regresores usados:", list(X.columns))

    # =============================
    # Conversión y limpieza NUMÉRICA
    # =============================
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = pd.to_numeric(y, errors="coerce").astype(float)

    # rellenar NA
    X = X.ffill().bfill()
    y = y.ffill().bfill()

    # Verificar que no quedan objetos
    for col in X.columns:
        if X[col].dtype == "object":
            raise ValueError(f"ERROR: columna {col} quedó como object")

    # =============================
    # Modelo ARIMA
    # =============================
    try:
        model_auto = auto_arima(
            y,
            exogenous=X,
            seasonal=True,
            m=12,
            max_p=2, max_q=2,
            max_P=2, max_Q=2,
            d=None, D=None,
            trace=False,
            suppress_warnings=True
        )
        order = model_auto.order
        seasonal_order = model_auto.seasonal_order
    except:
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, 12)

    print(f"   ➤ Orden ARIMA elegido: {order}, estacional: {seasonal_order}")

    model = sm.tsa.SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fit = model.fit(disp=False)
    resid = fit.resid
    rmse = np.sqrt(np.mean(resid**2))

    return {
        "country": c,
        "beta_own": fit.params.get("log_price", np.nan) if c != "mexico" else np.nan,
        "beta_competitor": fit.params.get("log_price_competitor_index", np.nan) if c != "mexico" else np.nan,
        "beta_relative_mex": fit.params.get("log_price_relative", np.nan) if c == "mexico" else np.nan,
        "rmse": rmse
    }


# =============================
# Ejecutar todos los países
# =============================
results = [run_country(c) for c in countries]
out = pd.DataFrame(results)

# =============================
# Guardar salida
# =============================
out_path = "/Users/rgarrido/Documents/Centro de Data Science/Difusion/Elasticidades/Modelo Armington/data/Elasticidades_logQ_SARIMAX_M2.csv"
out.to_csv(out_path, index=False)

print("\n Archivo generado:", out_path)
