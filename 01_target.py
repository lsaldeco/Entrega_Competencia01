import os
import duckdb
import pandas as pd

# 1. Definir rutas (modificar path)
dataset_path = "/home/laura/(modificar)/DMEyF/datos/"
dataset_filename = "competencia_01_crudo.csv"
output_filename = "competencia_01.csv"

file_path = os.path.join(dataset_path, dataset_filename)
output_path = os.path.join(dataset_path, output_filename)

# 2. Inicializar conexión a DuckDB
con = duckdb.connect(database=":memory:")

print("Cargando dataset crudo y calculando la clase ternaria...")

# 3. Ejecutar la query para crear la tabla con la clase ternaria
con.execute(f"""
    CREATE OR REPLACE TABLE competencia_01 AS
    WITH periodos AS (
        SELECT DISTINCT foto_mes FROM read_csv_auto('{file_path}')
    ),
    clientes AS (
        SELECT DISTINCT numero_de_cliente FROM read_csv_auto('{file_path}')
    ),
    todo AS (
        SELECT numero_de_cliente, foto_mes FROM clientes CROSS JOIN periodos
    ),
    clase_ternaria AS (
        SELECT
            c.*
            , IF(c.numero_de_cliente IS NULL, 0, 1) AS mes_0
            , LEAD(mes_0, 1) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_1
            , LEAD(mes_0, 2) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_2
            , CASE
                WHEN mes_0 = 1 AND mes_1 = 0 THEN 'BAJA+1'
                WHEN mes_0 = 1 AND mes_1 = 1 AND mes_2 = 0 THEN 'BAJA+2'
                WHEN mes_0 = 1 AND mes_1 = 1 AND mes_2 = 1 THEN 'CONTINUA'
                ELSE NULL
              END AS clase_ternaria
        FROM todo t
        LEFT JOIN read_csv_auto('{file_path}') c USING (numero_de_cliente, foto_mes)
    )
    SELECT * EXCLUDE (mes_0, mes_1, mes_2)
    FROM clase_ternaria
    WHERE mes_0 = 1
""")

print(f"Guardando el resultado procesado en '{output_path}'...")

# 4. Exportar la tabla resultante a un nuevo archivo CSV
con.execute(f"""
    COPY competencia_01 TO '{output_path}' (FORMAT CSV, HEADER)
""")

# 5. Generar y mostrar por pantalla la tabla pivoteada de control
print("\n--- Distribución de la Clase Ternaria por Mes ---")
df_pivot = con.execute("""

PIVOT (
    SELECT *
    FROM competencia_01
    WHERE foto_mes != 202108
)
ON clase_ternaria
USING count(numero_de_cliente)
GROUP BY foto_mes
ORDER BY foto_mes;
""").fetchdf()

print(df_pivot)
print("\n¡Listo!")
