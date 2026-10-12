# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ruta al archivo multisemilla que acabamos de generar
multisemilla_path = ("/home/laura/...(modificar).../DMEyF/competencia01/pred_junio_exp4.parquet")

ganancia_acierto = 1_072_500
costo_estimulo = 27_500
semillas = [
    777781, 500041, 600011, 800077, 200023, 
    388890, 1555565, 790126, 250020, 1000085, 
    512386, 300005, 1200025, 612356, 400038, 
    1600157, 812422, 100011, 400049, 212368
]


# 1. Cargar el archivo
print("Cargando predicciones multisemilla...")
df = pd.read_parquet(multisemilla_path)

plt.figure(figsize=(12, 7))

# Diccionario para guardar los picos óptimos de cada semilla
picos_optimos = {}

# 2. Iterar por cada semilla, calcular su curva y graficarla
for seed in semillas:
    col_pred = f"pred_seed_{seed}"

    # Ordenar de mayor a menor probabilidad para esta semilla
    df_sorted = df.sort_values(by=col_pred, ascending=False).reset_index(
        drop=True
    )

    # Ganancia acumulada
    df_sorted["ganancia_ind"] = np.where(
        df_sorted["clase_binaria"] == 1, ganancia_acierto, -costo_estimulo
    )
    df_sorted["ganancia_acum"] = df_sorted["ganancia_ind"].cumsum()

    # Encontrar pico
    pico_idx = df_sorted["ganancia_acum"].idxmax()
    n_optimo = pico_idx + 1
    ganancia_max = df_sorted.loc[pico_idx, "ganancia_acum"]
    picos_optimos[seed] = (n_optimo, ganancia_max)

    print(
        f"Semilla {seed}: N* óptimo = {n_optimo:,} envíos | Ganancia Máxima ="
        f" ${ganancia_max:,.0f}"
    )

    # Graficar curva tenue para esta semilla
    plt.plot(
        df_sorted.index + 1,
        df_sorted["ganancia_acum"] / 1e6,
        alpha=0.4,
        linewidth=1.2,
        label=f"Semilla {seed} (N*={n_optimo:,})",
    )

# 3. Calcular y graficar el PROMEDIO de las predicciones (estrategia de ensembling)
cols_preds = [f"pred_seed_{s}" for s in semillas]
df["pred_promedio"] = df[cols_preds].mean(axis=1)

df_prom_sorted = df.sort_values(by="pred_promedio", ascending=False).reset_index(
    drop=True
)
df_prom_sorted["ganancia_ind"] = np.where(
    df_prom_sorted["clase_binaria"] == 1, ganancia_acierto, -costo_estimulo
)
df_prom_sorted["ganancia_acum"] = df_prom_sorted["ganancia_ind"].cumsum()

pico_prom_idx = df_prom_sorted["ganancia_acum"].idxmax()
n_optimo_prom = pico_prom_idx + 1
ganancia_max_prom = df_prom_sorted.loc[pico_prom_idx, "ganancia_acum"]

print("=" * 60)
print(
    f" PROMEDIO DE SEMILLAS: N* óptimo = {n_optimo_prom:,} envíos | Ganancia"
    f" Máxima = ${ganancia_max_prom:,.0f}"
)
print("=" * 60)

# Graficar la curva del promedio bien marcada en negro/rojo
plt.plot(
    df_prom_sorted.index + 1,
    df_prom_sorted["ganancia_acum"] / 1e6,
    color="black",
    linewidth=2.5,
    linestyle="--",
    label=f"PROMEDIO (N*={n_optimo_prom:,})",
)

# 4. Formato del gráfico con zoom en el Eje Y
plt.title(
    "Curvas de Ganancia Acumulada - Múltiples Semillas vs Promedio (Junio"
    " 2021)",
    fontsize=14,
    pad=12,
)
plt.xlabel("Cantidad de Clientes Estimulados (Ordenados por Score)", fontsize=11)
plt.ylabel("Ganancia Acumulada en Millones de $", fontsize=11)

# --- ACÁ AJUSTAMOS LAS ESCALAS PARA QUE SE VEA PERFECTO ---
plt.xlim(0, 25000)  # Zoom en el eje X (de 0 a 25.000 clientes)
plt.ylim(
    -50, 500
)  # Zoom en el eje Y (de -50M a 450M para apreciar el pico de las curvas)
# ---------------------------------------------------------

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=9, loc="lower right")
plt.tight_layout()
# Definir marcas en el eje X cada 2,500 clientes para ver más detalle
plt.xticks(np.arange(0, 26000, 2500))

plt.show()
