# -*- coding: utf-8 -*-
"""
Optimizacion con Optuna + LightGBM 
"""

import gc
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import (
    plot_contour,
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde, ks_2samp
from sklearn.model_selection import StratifiedShuffleSplit

# 1. Leés tu archivo parquet actual
db_path = "/home/laura/----(modificar).../DMEyF/db/"
data = pd.read_parquet(
    "/home/laura/----(modificar).../DMEyF/datos/competencia_01_exp4.parquet"
)

ganancia_acierto = 1_072_500
costo_estimulo = 27_500

meses_train = [202103, 202104]
mes_test = 202106

# agregue sus semillas
semilla = 777781
np.random.seed(semilla)
semillas = np.random.choice(1000000, size=50, replace=False).tolist()

data["clase_binaria"] = np.where(data["clase_ternaria"] == "BAJA+2", 1, 0)

# ==============================================================================
# SELECCIÓN DE COLUMNAS: RANKINGS + FLAGS_ZEROS (Y EXCLUSIÓN DE COL.PRÉSTAMOS)
# ==============================================================================
# Identificar columnas que terminan en _rank o en _zero
cols_features = [
    c
    for c in data.columns
    if (c.endswith("_rank") or c.endswith("_zero"))
    and not c.startswith("mprestamos_personales")
    and not c.startswith("cprestamos_personales")
]

# Columnas de control necesarias
cols_meta = ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase_binaria"]

print(
    f"Cantidad de variables seleccionadas (Rankings + Flags_Zeros):"
    f" {len(cols_features)}"
)

# Filtrar train y test quedándonos solo con meta + las features elegidas
train_data = data[data["foto_mes"].isin(meses_train)][cols_meta + cols_features]
test_data = data[data["foto_mes"] == mes_test][cols_meta + cols_features]

# ¡PARA LA RAM: Borramos el dataframe gigante original y liberamos memoria
del data
gc.collect()

# --- ENTRENAMIENTO ---
X_train = train_data[cols_features]
y_train_binaria = train_data["clase_binaria"]

# --- TEST / EVALUACIÓN ---
X_test = test_data[cols_features]
y_test_binaria = test_data["clase_binaria"]
y_test_class = test_data["clase_ternaria"]

# Crear Datasets de LightGBM (free_raw_data=True ahorra un 50% de RAM)
train_data_lgb = lgb.Dataset(
    X_train, label=y_train_binaria, free_raw_data=True
)
test_data_lgb = lgb.Dataset(
    X_test, label=y_test_binaria, reference=train_data_lgb, free_raw_data=True
)

# ==============================================================================
# EVALUACIÓN Y OPTUNA
# ==============================================================================
sss_opt = StratifiedShuffleSplit(
    n_splits=5, test_size=0.3, random_state=semillas[0]
)


def ganancia_prob_lgm(y_pred, data, prop=0.3):
  y_true = data.get_label()
  ganancia = (
      np.where(y_true == 1, ganancia_acierto, 0)
      - np.where(y_true == 0, costo_estimulo, 0)
  )
  return "gan_prob", ganancia[y_pred >= 0.025].sum() / prop, True


# hiperparametros para LGBM
def objective(trial):
  num_leaves = trial.suggest_int("num_leaves", 8, 100)
  learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
  min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 1000)
  feature_fraction = trial.suggest_float("feature_fraction", 0.1, 1.0)
  bagging_fraction = trial.suggest_float("bagging_fraction", 0.1, 1.0)

  params = {
      "objective": "binary",
      "metric": "custom",
      "boosting_type": "gbdt",
      "first_metric_only": True,
      "boost_from_average": True,
      "feature_pre_filter": False,
      "max_bin": 31,
      "num_leaves": num_leaves,
      "learning_rate": learning_rate,
      "min_data_in_leaf": min_data_in_leaf,
      "feature_fraction": feature_fraction,
      "bagging_fraction": bagging_fraction,
      "seed": semillas[0],
      "verbose": -1,
      "n_jobs": -1,
  }

  cv_results = lgb.cv(
      params,
      train_data_lgb,
      num_boost_round=3000,
      callbacks=[
          lgb.early_stopping(
              stopping_rounds=int(50 + 5 / learning_rate), verbose=False
          )
      ],
      feval=ganancia_prob_lgm,
      folds=sss_opt.split(X_train, y_train_binaria),
      seed=semillas[0],
  )

  max_gan = max(cv_results["valid gan_prob-mean"])
  best_iter = cv_results["valid gan_prob-mean"].index(max_gan) + 1

  # Guardamos cual es la mejor iteración del modelo
  trial.set_user_attr("best_iter", best_iter)

  return max_gan


storage_name = "sqlite:///" + db_path + "optimization_lgbm_4.db"
study_name = "exp_4_lgbm_sss"

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

# 1. Guardamos el tiempo de inicio
start_time = time.time()

# optimizo usando optuna
study.optimize(objective, n_trials=60)

elapsed_time = time.time() - start_time
minutos = elapsed_time / 60
horas = minutos / 60

# 4. Cálculo del tiempo de ejecución
print(f"\n--- TIEMPO DE EJECUCIÓN ---")
print(f"Segundos: {elapsed_time:.2f}")
print(f"Minutos:  {minutos:.2f}")
if horas >= 1:
  print(f"Horas:    {horas:.2f}")

