import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm
from itertools import product
import os

OUTPUT_DIR = "/Users/rgarrido/Documents/Centro de Data Science/Difusion/Elasticidades/Modelo Armington/Resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# =====================================================
#  CARGA DE DATOS
# =====================================================

DATA_PATH = "/Users/rgarrido/Documents/Centro de Data Science/Difusion/Elasticidades/Modelo Armington/data/Base_de_Calibracion_q_en_ton_por_usd_utf8.csv"
df = pd.read_csv(DATA_PATH)

required_cols = ["Country", "Date", "log_share", "log_price", "TAVG", "share", "q_imported"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas requeridas: {missing}")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Country", "Date"])

countries = sorted(df["Country"].unique())
print("Países detectados:", countries)

# =====================================================
#  FUNCIÓN PARA AJUSTAR SARIMAX (BY COUNTRY)
# =====================================================

def fit_sarimax(dcountry):
    """
    Ajusta SARIMAX por país para obtener elasticidad propia.
    """
    d = df[df["Country"] == dcountry].copy()
    d = d.dropna(subset=["log_price", "q_imported", "TAVG"])

    y = np.log(d["q_imported"])
    X = pd.DataFrame({
        "log_price": d["log_price"],
        "TAVG": d["TAVG"]
    })
    X = sm.add_constant(X)

    try:
        model = sm.tsa.SARIMAX(
            y, exog=X, order=(2,1,2), seasonal_order=(1,0,1,12),
            enforce_stationarity=False, enforce_invertibility=False
        )
        res = model.fit(disp=False)

        params = res.params

        beta_own = params.get("log_price", np.nan)
        beta_tavg = params.get("TAVG", np.nan)

        return {
            "Country": dcountry,
            "Own_Elasticity": beta_own,
            "Temp_Effect": beta_tavg,
            "RMSE": np.sqrt(np.mean(res.resid**2)),
            "Converged": res.mle_retvals["converged"],
        }

    except Exception as e:
        print(f"Error en {dcountry}: {e}")
        return {
            "Country": dcountry,
            "Own_Elasticity": np.nan,
            "Temp_Effect": np.nan,
            "RMSE": np.nan,
            "Converged": False,
        }

# =====================================================
#  OBTENER ELASTICIDADES PROPIAS
# =====================================================

print("\n=== Ajustando elasticidades propias ===")
own_results = []

for c in countries:
    print(f"Ajustando {c}...")
    own_results.append(fit_sarimax(c))

df_own = pd.DataFrame(own_results)
print(df_own)

# =====================================================
#  ELASTICIDADES CRUZADAS
# =====================================================

# Fórmula económica basada en Armington:
# ε_ij = - (elasticidad propia del país j) * share_j_promedio

avg_shares = df.groupby("Country")["share"].mean().to_dict()

rows = []
for i, j in product(countries, countries):
    if i == j:
        continue

    if np.isnan(df_own[df_own["Country"] == j]["Own_Elasticity"].values[0]):
        cross = np.nan
    else:
        beta_j = df_own[df_own["Country"] == j]["Own_Elasticity"].values[0]
        s_j = avg_shares.get(j, np.nan)
        cross = - beta_j * s_j

    rows.append({
        "Importer": i,
        "Competitor": j,
        "Cross_Elasticity": cross
    })

df_cross = pd.DataFrame(rows)

# =====================================================
#  EXPORTAR RESULTADOS
# =====================================================

OUT1 = os.path.join(OUTPUT_DIR, "Elasticidades_SARIMAX_OWN.csv")
OUT2 = os.path.join(OUTPUT_DIR, "Elasticidades_SARIMAX_CROSS.csv")

df_own.to_csv(OUT1, index=False)
df_cross.to_csv(OUT2, index=False)

print("\nArchivos generados:")
print(" -", OUT1)
print(" -", OUT2)
print("\nProceso completado.")
