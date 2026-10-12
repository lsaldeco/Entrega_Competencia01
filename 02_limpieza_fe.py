# -*- coding: utf-8 -*-
"""Limpieza + Lags/Deltas + Ranking + imputacion faltantes + flags para ceros"""

import gc
import numpy as np
import pandas as pd

dataset_path = "/home/laura/----(modificar).../DMEyF/datos/"
dataset_file = "competencia_01.csv"
output_path = ("/home/laura/----(modificar).../DMEyF/datos/competencia_01_exp4.parquet")
data = pd.read_csv(dataset_path + dataset_file)

# 1. NaN en la columna ccajas_depositos de Mayo
data_1 = data.copy()
data_1.loc[data_1["foto_mes"] == 202105, "ccajas_depositos"] = np.nan


# 2. Imputación general con promedio entre mes siguiente y mes previo
def imputacion_general_dataset(df, cols_excluir=None):
  if cols_excluir is None:
    cols_excluir = [
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
        "cliente_vip",
        "internet",
    ]

  df_clean = df.copy()
  df_clean = df_clean.sort_values(
      by=["numero_de_cliente", "foto_mes"]
  ).reset_index(drop=True)
  cols_a_imputar = [c for c in df_clean.columns if c not in cols_excluir]

  print(f"Procesando {len(cols_a_imputar)} variables para interpolación...")
  for col in cols_a_imputar:
    lag = df_clean.groupby("numero_de_cliente")[col].shift(1)
    lead = df_clean.groupby("numero_de_cliente")[col].shift(-1)
    mask_nan = df_clean[col].isnull() & lag.notnull() & lead.notnull()
    if mask_nan.sum() > 0:
      df_clean.loc[mask_nan, col] = (lag[mask_nan] + lead[mask_nan]) / 2.0

  print("¡Limpieza completa!")
  return df_clean


data_limpia = imputacion_general_dataset(data_1)


# 3. Creación de columnas con deltas y lags por bloques
def agregar_lags_y_deltas_por_bloques(
    df, cols_excluir=None, batch_size=20, eps=1e-5
):
  if cols_excluir is None:
    cols_excluir = [
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
        "cliente_vip",
        "internet",
    ]

  df_base = df.sort_values(by=["numero_de_cliente", "foto_mes"]).reset_index(
      drop=True
  )
  cols_float = df_base.select_dtypes(include=["float64"]).columns
  df_base[cols_float] = df_base[cols_float].astype("float32")

  cols_features = [c for c in df_base.columns if c not in cols_excluir]
  print(
      f"Calculando lags y deltas para {len(cols_features)} variables en bloques"
      f" de {batch_size}..."
  )

  lista_df_resultados = [df_base]

  for i in range(0, len(cols_features), batch_size):
    batch_cols = cols_features[i : i + batch_size]
    dict_batch = {}

    for col in batch_cols:
      dict_batch[f"{col}_lag1"] = df_base.groupby("numero_de_cliente")[
          col
      ].shift(1)
      dict_batch[f"{col}_lag2"] = df_base.groupby("numero_de_cliente")[
          col
      ].shift(2)

    for col in batch_cols:
      lag1 = dict_batch[f"{col}_lag1"]
      dict_batch[f"{col}_delta1"] = df_base[col] - lag1
      dict_batch[f"{col}_ratio1"] = (df_base[col] - lag1) / (
          np.abs(lag1) + eps
      )

    df_batch = pd.DataFrame(dict_batch, index=df_base.index).astype("float32")
    lista_df_resultados.append(df_batch)
    del dict_batch, df_batch
    gc.collect()

  df_final = pd.concat(lista_df_resultados, axis=1)
  del lista_df_resultados
  gc.collect()
  print(f"¡Lags y deltas listos! Forma actual: {df_final.shape}")
  return df_final


data_completa = agregar_lags_y_deltas_por_bloques(data_limpia, batch_size=20)


# 4. Creación del ranking de cero fijo mes a mes
def agregar_ranking_cero_fijo_por_bloques(df, cols_excluir=None, batch_size=20):
  if cols_excluir is None:
    cols_excluir = [
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
        "cliente_vip",
        "internet",
    ]

  cols_float = df.select_dtypes(include=["float64"]).columns
  if len(cols_float) > 0:
    df[cols_float] = df[cols_float].astype("float32")

  cols_features = [
      c
      for c in df.columns
      if c not in cols_excluir
      and not c.endswith("_zero")
      and pd.api.types.is_numeric_dtype(df[c])
  ]
  print(
      f"Calculando ranking de cero fijo para {len(cols_features)} variables en"
      f" bloques de {batch_size}..."
  )

  lista_df_resultados = [df]

  for i in range(0, len(cols_features), batch_size):
    batch_cols = cols_features[i : i + batch_size]
    dict_batch = {}

    for col in batch_cols:
      serie_rank = pd.Series(0.0, index=df.index, dtype="float32")
      for mes, idx_mes in df.groupby("foto_mes").groups.items():
        sub_datos = df.loc[idx_mes, col]

        mask_pos = sub_datos > 0
        if mask_pos.any():
          serie_rank.loc[idx_mes[mask_pos]] = (
              sub_datos[mask_pos].rank(pct=True).astype("float32")
          )

        mask_neg = sub_datos < 0
        if mask_neg.any():
          serie_rank.loc[idx_mes[mask_neg]] = (
              -(sub_datos[mask_neg].abs().rank(pct=True))
          ).astype("float32")

      dict_batch[f"{col}_rank"] = serie_rank

    df_batch = pd.DataFrame(dict_batch, index=df.index).astype("float32")
    lista_df_resultados.append(df_batch)
    del dict_batch, df_batch
    gc.collect()

  df_final = pd.concat(lista_df_resultados, axis=1)
  del lista_df_resultados
  gc.collect()
  print(f"¡Ranking completo! shape actual: {df_final.shape}")
  return df_final


data_con_ranks = agregar_ranking_cero_fijo_por_bloques(
    data_completa, batch_size=20
)


# 5. Agregar las banderas _zero SOLAMENTE sobre las variables numéricas originales
def agregar_variables_zero_final(df, cols_excluir=None):
  if cols_excluir is None:
    cols_excluir = [
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
        "cliente_vip",
        "internet",
    ]

  cols_base_numericas = [
      c
      for c in data_limpia.columns
      if c not in cols_excluir and pd.api.types.is_numeric_dtype(data_limpia[c])
  ]
  print(
      f"Agregando banderas _zero estáticas solo para"
      f" {len(cols_base_numericas)} variables base numéricas..."
  )

  dict_zero = {}
  for col in cols_base_numericas:
    s = data_limpia[col]
    zero_col = pd.Series(0.0, index=df.index, dtype="float32")
    zero_col[s.isna()] = np.nan
    zero_col[s == 0] = 1.0
    dict_zero[f"{col}_zero"] = zero_col

  df_final = pd.concat([df, pd.DataFrame(dict_zero, index=df.index)], axis=1)
  del dict_zero
  gc.collect()
  print(f"¡Feature Engineering completo! Forma final: {df_final.shape}")
  return df_final


data_final = agregar_variables_zero_final(data_con_ranks)

# Guardar dataset final
data_final.to_parquet(output_path, index=False)
print(f"¡Dataset guardado exitosamente en: {output_path}")
