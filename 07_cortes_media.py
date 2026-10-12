# -*- coding: utf-8 -*-
import os
import pandas as pd

# --- RUTAS ---
pred_agosto_path = "/home/laura/...(modificar).../DMEyF/competencia01/predicciones_agosto_exp4.parquet"
csv_salida_path = ("/home/laura/...(modificar).../DMEyF/competencia01/exp4/media/agosto_media_70_sem_10k.csv")

#ir cambiando el valor de corte
CORTE = 1000

# Creamos la carpeta de salida si no existe
os.makedirs(os.path.dirname(csv_salida_path), exist_ok=True)

print("Cargando predicciones individuales de las 70 semillas (.parquet)...")
df_preds = pd.read_parquet(pred_agosto_path)

# Identificamos todas las columnas de semillas
cols_semillas = [c for c in df_preds.columns if c.startswith("pred_seed_")]
print(f"Cantidad de semillas encontradas para ensamble: {len(cols_semillas)}")

if len(cols_semillas) == 0:
  raise ValueError(
      "No se encontraron columnas que empiecen con 'pred_seed_' en el archivo."
  )

# --- PROMEDIAR LAS PROBABILIDADES (Ensemble) ---
print("Calculando la media de las probabilidades de todas las semillas...")
df_preds["prob_ensemble"] = df_preds[cols_semillas].mean(axis=1)

# --- ORDENAR Y APLICAR EL CORTE ---
print(f"Ordenando y seleccionando los mejores {CORTE} clientes...")

df_sorted = df_preds.sort_values(by="prob_ensemble", ascending=False).reset_index(
    drop=True
)

# Nos quedamos estrictamente con los primeros CORTE clientes
df_envio = df_sorted.head(CORTE)[["numero_de_cliente"]].copy()

# --- GUARDAR ARCHIVO ---
df_envio.to_csv(csv_salida_path, index=False, header=False)

print("\n" + "=" * 50)
print(f"¡CSV de envío final generado en:")
print(f"{csv_salida_path}")
print(f"Total de clientes seleccionados: {len(df_envio)}")
print("=" * 50)
