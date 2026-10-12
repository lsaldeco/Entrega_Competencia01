# Entrega_Competencia01
Repo con scripts que se usaron para generar los submits 
Los scripts deben ejecutarse en este orden:

01_target.py (genera la columna ternaria a partir del archivo competencia_01_crudo.csv)

02_limpieza_fe.py (limpeza del dataset + feature enginering)

03_optuna_LGBM.py (optimización con Optuna+LightGBM, 60 trials, 1 semilla, se usa marzo+abril para predecir junio)

04_validacion_junio.py (valida junio con un total de 20 semillas)

05_curvas_ganancia.py (grafica las curvas de ganancia de la validacion de junio con todas sus semillas)

06_prediccion_agosto.py (usa marzo, abril, mayo y junio para predecir agosto, busca el mejor modelo de la optimizacion, se usaron 70 semillas)

07_cortes_media.py (promedia las probabilidades de las semillas y genera el csv para subir al bot)
