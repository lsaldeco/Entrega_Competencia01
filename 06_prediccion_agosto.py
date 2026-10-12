# -*- coding: utf-8 -*-
import gc
import time
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

# --- CONFIGURACIÓN DE RUTAS Y CONSTANTES ---
db_path = "/home/laura/...(modificar).../DMEyF/db/"
dataset_path = (
    "/home/laura/...(modificar).../DMEyF/datos/competencia_01_exp4.parquet"
)
pred_agosto_path = "/home/laura/----(modificar).../DMEyF/competencia01/predicciones_agosto_exp4.parquet"

study_name = "exp_4_lgbm_sss"
storage_name = "sqlite:///" + db_path + "optimization_lgbm_4.db"

# 20 semillas (5 asignadas + 15 más)
semillas = [
    # --- Tus 20 semillas originales ---
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
    111111,
    222222,
    333333,
    444444,
    555555,
    666666,
    777777,
    888888,
    999999,
    123789,
    456123,
    789456,
    321654,
    987321,
    654987,
    112233,
    445566,
    778899,
    102030,
    405060,
    708090,
    152637,
    485960,
    718293,
    142536,
    475869,
    708192,
    132435,
    465768,
    798091,
    246810,
    135792,
    864209,
    975318,
    192837,
    465870,
    791827,
    364554,
    829103,
    506172,
    617283,
    728394,
    839405,
    940516,
    151617,
    262728,
    373839,
    484950,
    596071,
    607182,
]


# --- 1. CARGA DE OPTUNA & RECUPERACIÓN DEL MEJOR TRIAL ---
print("Conectando a Optuna para recuperar el mejor trial...")
study = optuna.load_study(study_name=study_name, storage=storage_name)

completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
completed_trials.sort(key=lambda t: t.value, reverse=True)
best_trial = completed_trials[0]

best_iter = best_trial.user_attrs.get("best_iter", None)
if best_iter is None:
  raise ValueError(
      f"El trial #{best_trial.number} no tiene 'best_iter' guardado en"
      " user_attrs."
  )

print(f" -> Mejor trial ID: {best_trial.number}")
print(f" -> Ganancia en CV del mejor trial: ${best_trial.value:,.0f}")
print(f" -> Mejor cantidad de iteraciones (best_iter): {best_iter}")

params_base = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "first_metric_only": True,
    "boost_from_average": True,
    "feature_pre_filter": False,
    "max_bin": 31,
    "verbose": -1,
    "n_jobs": 6,
}
params_base.update(best_trial.params)

# --- 2. CARGA FRACCIONADA MES A MES (Con préstamos incluidos) ---
print(
    "\nCargando datos de entrenamiento mes a mes con préstamos (Marzo a"
    " Junio)..."
)
meses_train = [202103, 202104, 202105, 202106]
X_train_list = []
y_train_list = []
cols_features = None

for m in meses_train:
  print(f" -> Leyendo mes {m}...")
  df_m = pd.read_parquet(dataset_path, filters=[("foto_mes", "==", m)])

  if cols_features is None:
    # Seleccionamos TODOS los _rank y _zero SIN excluir préstamos
    cols_features = [
        c for c in df_m.columns if c.endswith("_rank") or c.endswith("_zero")
    ]
    print(
        f"    Cantidad total de variables seleccionadas (Rankings + Flags_Zeros e incluyo"
        f" préstamos): {len(cols_features)}"
    )

  clase_bin = np.where(df_m["clase_ternaria"] == "BAJA+2", 1, 0).astype(
      np.float32
  )

  X_train_list.append(df_m[cols_features].to_numpy(dtype=np.float32))
  y_train_list.append(clase_bin)

  del df_m
  gc.collect()

print("Concatenando matrices de entrenamiento...")
X_train_final = np.vstack(X_train_list)
y_train_final = np.concatenate(y_train_list)
del X_train_list, y_train_list
gc.collect()

print(f"Shape final X_train: {X_train_final.shape}")

print("\nCargando datos de predicción de Agosto...")
agosto_raw = pd.read_parquet(
    dataset_path,
    filters=[("foto_mes", "==", 202108)],
)
cols_meta = ["numero_de_cliente", "foto_mes", "clase_ternaria"]
df_agosto = agosto_raw[
    [c for c in cols_meta if c in agosto_raw.columns] + cols_features
].copy()
X_agosto = df_agosto[cols_features].to_numpy(dtype=np.float32)

del agosto_raw
gc.collect()

# DataFrame base para almacenar el cliente y las predicciones individuales
df_preds_individuales = pd.DataFrame(
    {"numero_de_cliente": df_agosto["numero_de_cliente"]}
)

# --- 3. BUCLE MULTISEMILLA: ENTRENAMIENTO Y GUARDADO INDIVIDUAL ---
print(
    f"\nIniciando entrenamiento de {len(semillas)} modelos con train completo..."
)
tiempo_inicio_total = time.time()

for i, seed in enumerate(semillas, 1):
  tiempo_inicio_semilla = time.time()
  print(f" -> [{i}/{len(semillas)}] Entrenando con Semilla: {seed}...")

  params_seed = params_base.copy()
  params_seed["seed"] = seed

  train_data_final_lgb = lgb.Dataset(
      X_train_final, label=y_train_final, free_raw_data=True
  )
  final_model = lgb.train(
      params_seed, train_data_final_lgb, num_boost_round=best_iter
  )

  df_preds_individuales[f"pred_seed_{seed}"] = final_model.predict(X_agosto)

  tiempo_semilla = time.time() - tiempo_inicio_semilla
  print(f"    [OK] Semilla {seed} finalizada en {tiempo_semilla:.1f} segundos.")

  del final_model, train_data_final_lgb
  gc.collect()

# --- 4. GUARDADO DE RESULTADOS ---
print(
    "\nGuardando archivo Parquet con las predicciones de Agosto ..."
)
df_preds_individuales.to_parquet(pred_agosto_path, index=False)

tiempo_total = time.time() - tiempo_inicio_total
print("\n" + "=" * 50)
print("¡Proceso de predicción de Agosto finalizado!")
print(f"Tiempo total transcurrido: {tiempo_total / 60:.2f} minutos.")
print(f"Archivo guardado en: {pred_agosto_path}")
print("=" * 50)
