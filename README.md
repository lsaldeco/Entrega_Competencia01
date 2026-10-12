# Entrega_Competencia01
Repo con scripts que se usaron para generar los submits 
Los scripts deben ejecutarse en este orden:

01_target.py (genera la columna ternaria a partir del archivo competencia_01_crudo.csv)

02_limpieza_fe.py (limpeza del dataset + feature enginering)

03_optuna_LGBM.py (optimización con Optuna+LightGBM, 60 trials, 1 semilla, se usa marzo+abril para predecir junio)

04_validacion_junio.py (valida junio con un total de 20 semillas)

05_curvas_ganancia.py (grafica las curvas de ganancia de la validacion de junio con todas sus semillas)

06_prediccion_agosto.py (usa marzo, abril, mayo y junio para predecir agosto, usa el mejor modelo de la optimizacion)

07_cortes_media.py (promedia las probabilidades de las semillas y genera el csv para subir al bot)

# Algunas aclaraciones
Respecto de los faltantes se los imputó con el promedio del mes siguiente y el mes previo, a la columna ccajas_depositos de mayo se la imputó con NaN por recomendación de la cátedra, se crearon columnas de deltas y lags de 1 y 2 meses, se aplicó ranking de 0 fijo a todas las variables numéricas, se crearon columnas con flags para cuando las variables son nulas (recomendación de la cátedra)

Respecto del entrenamiento se usó marzo y abril para predecir el mes de junio. Se tomaron sólo las columnas del ranking + las columnas de flags, se dejaron afueran del entrenamiento las columnas con variables originales y las asociadas a prestamos personales (ccprestamos_personales y mprestamos_personales).

Respecto a la validacion se utilizaron 20 semillas para validar el mes de junio con el mejor modelo obtenido y luego obtener la curva de ganancia acumulada que se utilizó para estimar posibles valores de corte para el submit.

Para la predicción de agosto se utilizó el mejor modelo + 70 semillas y se definió el submit final de acuerdo al public gain del bot. Por recomendacion de la cátedra se tuvieron en cuenta las columnas asociadas a préstamos_personales que no se habían tenido en cuenta para el entrenamiento.




