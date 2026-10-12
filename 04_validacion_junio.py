# -*- coding: utf-8 -*-

import gc
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

# --- CONFIGURACIÓN de paths---
dataset_path = (
    "/home/laura/----(modificar).../DMEyF/datos/competencia_01_exp4.parquet"
)
output_path = "/home/laura/----(modificar).../DMEyF/competencia01/pred_junio_exp4.parquet"
db_path = "/home/laura/----(modificar).../DMEyF/db/"


# 20 semillas (5 asignadas + 15 más)
semillas = [
    777781,
    500041,
    600011,
    800077,
    200023,
    388890,
    1555565,
    790126,
    250020,
    1000085,
    512386,
    300005,
    1200025,
    612356,
    400038,
    1600157,
    812422,
    100011,
    400049,
    212368,
]

cols_a_excluir = [
    "clase_ternaria",
    "clase_binaria",
    "numero_de_cliente",
    "foto_mes",
]

# --- RECUPERAR EL MEJOR MODELO Y PARÁMETROS DESDE OPTUNA ---
storage_name = "sqlite:///" + db_path + "optimization_lgbm_4.db"
study_name = "exp_4_lgbm_sss"

try:
  study = optuna.load_study(study_name=study_name, storage=storage_name)
  best_params_optuna = study.best_params
  # Recuperamos la mejor iteración guardada en los atributos del mejor trial
  best_iter_optuna = study.best_trial.user_attrs.get("best_iter", 1000)

  print(f"\n[Optuna] ¡Configuración óptima recuperada con éxito!")
  print(f" -> Mejor ganancia en CV: ${study.best_value:,.0f}")
  print(f" -> Mejores hiperparámetros: {best_params_optuna}")
  print(f" -> Mejor cantidad de árboles (best_iter): {best_iter_optuna}")

except Exception as e:
  print(
      f"\n[Aviso] No se pudo cargar el estudio de Optuna ({e}). Usando valores"
      " por defecto."
  )
  best_params_optuna = {
      "num_leaves": 31,
      "learning_rate": 0.02,
      "min_data_in_leaf": 200,
      "feature_fraction": 0.5,
      "bagging_fraction": 1.0,
  }
  best_iter_optuna = 500

# --- 1. CARGA OPTIMIZADA DE DATOS (Marzo, Abril y Junio) ---
print("\n[1/2] Cargando datos de validación (Marzo, Abril y Junio)...")
data = pd.read_parquet(
    dataset_path, filters=[("foto_mes", "in", [202103, 202104, 202106])]
)
data["clase_binaria"] = np.where(data["clase_ternaria"] == "BAJA+2", 1, 0)

# Selección exclusiva de Rankings + Zeros (excluyendo préstamos personales)
cols_features = [
    c
    for c in data.columns
    if (c.endswith("_rank") or c.endswith("_zero"))
    and not c.startswith("mprestamos_personales")
    and not c.startswith("cprestamos_personales")
]

cols_meta = ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase_binaria"]
print(
    f"Cantidad de variables seleccionadas (Rankings + Zeros sin préstamos):"
    f" {len(cols_features)}"
)

train_mask = data["foto_mes"].isin([202103, 202104])
test_mask = data["foto_mes"] == 202106

train_data = data.loc[train_mask, cols_meta + cols_features]
test_data = data.loc[test_mask, cols_meta + cols_features]

X_train = train_data[cols_features]
y_train = train_data["clase_binaria"]
X_junio = test_data[cols_features]

df_resultados = test_data[
    ["numero_de_cliente", "clase_ternaria", "clase_binaria"]
].copy()

del data, train_data, test_data
gc.collect()

# --- 2. BUCLE SEGURO POR CADA SEMILLA ---
print(
    f"\n[2/2] Entrenando modelo y prediciendo para {len(semillas)} semillas..."
)

ganancia_acierto = 1_072_500
costo_estimulo = 27_500


def calcular_ganancia_array(y_true, y_pred, prop=0.3):
  df_temp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
  df_temp = df_temp.sort_values(by="y_pred", ascending=False)
  n_estimulos = int(len(df_temp) * prop)
  enviados = df_temp.iloc[:n_estimulos]
  ganancia_total = np.where(
      enviados["y_true"] == 1, ganancia_acierto, -costo_estimulo
  ).sum()
  return ganancia_total


ganancias_por_semilla = []

for i, seed in enumerate(semillas, 1):
  print(f" -> [{i}/{len(semillas)}] Entrenando con semilla: {seed}...")

  # Armamos el diccionario combinando la estructura fija con los hiperparámetros de Optuna
  params = {
      "objective": "binary",
      "metric": "auc",
      "boosting_type": "gbdt",
      "first_metric_only": True,
      "boost_from_average": True,
      "feature_pre_filter": False,
      "max_bin": 31,
      # Hiperparámetros optimizados dinámicamente por Optuna:
      "num_leaves": int(best_params_optuna.get("num_leaves", 31)),
      "learning_rate": float(best_params_optuna.get("learning_rate", 0.02)),
      "min_data_in_leaf": int(best_params_optuna.get("min_data_in_leaf", 200)),
      "feature_fraction": float(best_params_optuna.get("feature_fraction", 0.5)),
      "bagging_fraction": float(best_params_optuna.get("bagging_fraction", 1.0)),
      "seed": seed,
      "verbose": -1,
      "n_jobs": 6,
  }

  train_data_lgb = lgb.Dataset(X_train, label=y_train, free_raw_data=True)

  # Usamos el best_iter exacto que arrojó el estudio de Optuna
  model_val = lgb.train(params, train_data_lgb, num_boost_round=best_iter_optuna)

  preds = model_val.predict(X_junio)
  df_resultados[f"pred_seed_{seed}"] = preds

  # Calculamos ganancia individual por semilla
  ganancia_semilla = calcular_ganancia_array(
      df_resultados["clase_binaria"].values, preds, prop=0.3
  )
  ganancias_por_semilla.append(ganancia_semilla)

  del model_val, train_data_lgb
  gc.collect()

# --- 3. GUARDAR RESULTADOS A DISCO ---
df_resultados.to_parquet(output_path, index=False)
print("\n" + "=" * 50)
print("¡PValidacion multisemilla de Junio finalizado!")
print(f"Archivo guardado en: {output_path}")
print("=" * 50)
