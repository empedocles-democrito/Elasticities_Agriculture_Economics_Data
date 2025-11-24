import pandas as pd
import os

# === RUTAS ===
BASE_DIR = "/Users/rgarrido/Documents/Centro de Data Science/Difusion/Elasticidades/Modelo Armington"

OWN_PATH = f"{BASE_DIR}/Resultados/Elasticidades_SARIMAX_OWN.csv"
CROSS_PATH = f"{BASE_DIR}/Resultados/Elasticidades_SARIMAX_CROSS.csv"
OUT_LATEX = f"{BASE_DIR}/Resultados/Elasticidades_Tabla_Final.tex"

print(f"Leyendo OWN:  {OWN_PATH}")
print(f"Leyendo CROSS:{CROSS_PATH}")

own = pd.read_csv(OWN_PATH)
cross = pd.read_csv(CROSS_PATH)

print("\nColumnas OWN:", list(own.columns))
print("Columnas CROSS:", list(cross.columns))

# ==========================================================
#   NORMALIZAR NOMBRES
# ==========================================================
own["Country"] = own["Country"].str.title()
cross["Importer"] = cross["Importer"].str.title()
cross["Competitor"] = cross["Competitor"].str.title()

# ==========================================================
#   REGLA PARA MÉXICO EN CRUZADAS
# ==========================================================

cross = cross[cross["Importer"] != "Mexico"]


# ==========================================================
#   TABLA FINAL (OWN + CROSS EN UNA SOLA TABLA)
# ==========================================================

tabla_own = own[["Country", "Own_Elasticity", "Temp_Effect"]].copy()
tabla_own.columns = ["País", "Elasticidad Propia", "Efecto Temp"]

tabla_cross = cross.rename(columns={
    "Importer": "Importador",
    "Competitor": "Competidor",
    "Cross_Elasticity": "Elasticidad Cruzada"
})

# Generar LaTeX
latex = []
latex.append("\\begin{table}[H]")
latex.append("\\centering")
latex.append("\\footnotesize")
latex.append("\\caption{Elasticidades Armington (Propias y Cruzadas)}")
latex.append("\\begin{tabular}{lccc}")
latex.append("\\hline")
latex.append("País & Elasticidad Propia & Efecto Temp &  \\\\")
latex.append("\\hline")

for _, r in tabla_own.iterrows():
    latex.append(f"{r['País']} & {r['Elasticidad Propia']:.3f} & {r['Efecto Temp']:.3f} & \\\\")

latex.append("\\hline")
latex.append("\\end{tabular}")
latex.append("\\vspace{0.4cm}")

latex.append("% ==============================")
latex.append("% TABLA DE ELASTICIDADES CRUZADAS")
latex.append("% ==============================")

latex.append("\\begin{tabular}{lcc}")
latex.append("\\hline")
latex.append("Importador & Competidor & Elasticidad Cruzada \\\\")
latex.append("\\hline")

for _, r in tabla_cross.iterrows():
    latex.append(
        f"{r['Importador']} & {r['Competidor']} & {r['Elasticidad Cruzada']:.3f} \\\\"
    )

latex.append("\\hline")
latex.append("\\end{tabular}")

latex.append("\\end{table}")
latex.append(nota_mex)

# Guardar archivo
with open(OUT_LATEX, "w") as f:
    f.write("\n".join(latex))

print("\nArchivo LaTeX generado en:")
print(OUT_LATEX)
