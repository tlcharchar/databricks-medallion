# Databricks notebook source
# MAGIC %md
# MAGIC # 🥈 Silver Layer — Limpeza e Transformação
# MAGIC
# MAGIC **Objetivo:** Transformar os dados brutos da Bronze em dados limpos e confiáveis.
# MAGIC
# MAGIC Transformações aplicadas:
# MAGIC - Remoção de registros com valores nulos em campos críticos
# MAGIC - Filtro de corridas inválidas (distância <= 0, tarifa <= 0)
# MAGIC - Tipagem correta de colunas
# MAGIC - Colunas calculadas (duração da corrida, velocidade média)
# MAGIC - Remoção de duplicatas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, round as spark_round,
    unix_timestamp, hour, dayofweek,
    date_format, current_timestamp
)

CATALOG = "workspace"
SCHEMA_BRONZE = "medallion_bronze"
SCHEMA_SILVER = "medallion_silver"

# Criar schema Silver se não existir
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_SILVER}")

BRONZE_TABLE = f"{CATALOG}.{SCHEMA_BRONZE}.taxi_trips_raw"
SILVER_TABLE = f"{CATALOG}.{SCHEMA_SILVER}.taxi_trips_cleaned"

print(f"Lendo de: {BRONZE_TABLE}")
print(f"Gravando em: {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura da Bronze

# COMMAND ----------

df_bronze = spark.table(BRONZE_TABLE)
total_bronze = df_bronze.count()
print(f"Registros na Bronze: {total_bronze:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Análise de qualidade antes da limpeza

# COMMAND ----------

# Verificar nulos por coluna
# Nota: isnan() só funciona com DOUBLE/FLOAT, não com TIMESTAMP/STRING
from pyspark.sql.functions import count, when, isnan, isnull
from pyspark.sql.types import DoubleType, FloatType

numeric_types = (DoubleType, FloatType)

null_exprs = []
for field in df_bronze.schema.fields:
    if field.name in ["_ingestion_timestamp", "_source"]:
        continue
    if isinstance(field.dataType, numeric_types):
        # Para colunas numéricas: verificar null E NaN
        null_exprs.append(count(when(isnull(field.name) | isnan(field.name), field.name)).alias(field.name))
    else:
        # Para demais tipos: verificar apenas null
        null_exprs.append(count(when(isnull(field.name), field.name)).alias(field.name))

null_counts = df_bronze.select(null_exprs)
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformações Silver

# COMMAND ----------

df_silver = (
    df_bronze
    # 1. Remover nulos em campos críticos
    .filter(
        col("tpep_pickup_datetime").isNotNull() &
        col("tpep_dropoff_datetime").isNotNull() &
        col("PULocationID").isNotNull() &
        col("DOLocationID").isNotNull()
    )
    # 2. Filtrar corridas inválidas
    .filter(
        (col("trip_distance") > 0) &
        (col("fare_amount") > 0) &
        (col("passenger_count") > 0)
    )
    # 3. Calcular duração da corrida em minutos
    .withColumn(
        "trip_duration_min",
        spark_round(
            (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60,
            2
        )
    )
    # 4. Filtrar durações absurdas (< 1 min ou > 300 min / 5h)
    .filter(
        (col("trip_duration_min") >= 1) &
        (col("trip_duration_min") <= 300)
    )
    # 5. Calcular velocidade média (mph)
    .withColumn(
        "avg_speed_mph",
        spark_round(col("trip_distance") / (col("trip_duration_min") / 60), 2)
    )
    # 6. Filtrar velocidades absurdas (> 100 mph)
    .filter(col("avg_speed_mph") <= 100)
    # 7. Extrair features temporais
    .withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    .withColumn("pickup_day_of_week", dayofweek("tpep_pickup_datetime"))
    .withColumn("pickup_date", date_format("tpep_pickup_datetime", "yyyy-MM-dd"))
    # 8. Calcular custo por milha
    .withColumn(
        "cost_per_mile",
        spark_round(col("fare_amount") / col("trip_distance"), 2)
    )
    # 9. Tip percentage
    .withColumn(
        "tip_percentage",
        spark_round((col("tip_amount") / col("fare_amount")) * 100, 2)
    )
    # 10. Remover colunas de metadados da Bronze
    .drop("_ingestion_timestamp", "_source")
    # 11. Adicionar metadados da Silver
    .withColumn("_silver_timestamp", current_timestamp())
    # 12. Remover duplicatas
    .dropDuplicates()
)

# COMMAND ----------

total_silver = df_silver.count()
removed = total_bronze - total_silver
pct_removed = (removed / total_bronze) * 100

print(f"Bronze:  {total_bronze:,}")
print(f"Silver:  {total_silver:,}")
print(f"Removidos: {removed:,} ({pct_removed:.1f}%)")

# COMMAND ----------

# Preview dos dados limpos
display(df_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravar na Silver

# COMMAND ----------

(
    df_silver
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

print(f"Tabela {SILVER_TABLE} criada com sucesso!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação

# COMMAND ----------

df_check = spark.table(SILVER_TABLE)
print(f"Registros na Silver: {df_check.count():,}")
df_check.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Silver completa! Próximo: `03_gold_aggregation`
