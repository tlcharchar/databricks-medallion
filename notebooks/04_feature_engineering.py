# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering -- Silver para ML Feature Table
# MAGIC
# MAGIC **Tutorial 2: Machine Learning no Databricks Free Edition**
# MAGIC
# MAGIC **Objetivo:** Transformar os dados da Silver em features prontas para treinamento de modelos de ML.
# MAGIC
# MAGIC Caso de uso: **Prever o valor da gorjeta (tip_amount)** de corridas de taxi em NYC.
# MAGIC
# MAGIC Features criadas:
# MAGIC - Temporais (hora, dia da semana, fim de semana, periodo do dia)
# MAGIC - Geograficas (zona de pickup/dropoff)
# MAGIC - De viagem (distancia, duracao, velocidade, custo/milha)
# MAGIC - De pagamento (tipo de pagamento, tarifa base)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql.functions import (
    col, when, round as spark_round, log1p,
    hour, dayofweek, month, year,
    current_timestamp
)

CATALOG = "workspace"
SCHEMA_SILVER = "medallion_silver"
SCHEMA_ML = "medallion_ml"

# Criar schema para ML
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_ML}")

SILVER_TABLE = f"{CATALOG}.{SCHEMA_SILVER}.taxi_trips_cleaned"
FEATURE_TABLE = f"{CATALOG}.{SCHEMA_ML}.taxi_tip_features"

print(f"Lendo de: {SILVER_TABLE}")
print(f"Gravando em: {FEATURE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura da Silver

# COMMAND ----------

df_silver = spark.table(SILVER_TABLE)
print(f"Registros na Silver: {df_silver.count():,}")
df_silver.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analise exploratoria do target (tip_amount)

# COMMAND ----------

# Distribuicao do tip_amount
display(
    df_silver.select("tip_amount", "tip_percentage", "fare_amount", "payment_type")
    .summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Construcao das Features
# MAGIC
# MAGIC Vamos criar features em 4 categorias:
# MAGIC 1. **Temporais** -- quando a corrida aconteceu
# MAGIC 2. **Geograficas** -- de onde para onde
# MAGIC 3. **De viagem** -- caracteristicas da corrida
# MAGIC 4. **Categoricas** -- tipo de pagamento, vendor

# COMMAND ----------

df_features = (
    df_silver
    # --- FEATURES TEMPORAIS ---
    # Periodo do dia (madrugada, manha, tarde, noite)
    .withColumn(
        "period_of_day",
        when((col("pickup_hour") >= 0) & (col("pickup_hour") < 6), 0)   # madrugada
        .when((col("pickup_hour") >= 6) & (col("pickup_hour") < 12), 1)  # manha
        .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 18), 2) # tarde
        .otherwise(3)  # noite
    )
    # Fim de semana (1=domingo, 7=sabado no Spark)
    .withColumn(
        "is_weekend",
        when(col("pickup_day_of_week").isin(1, 7), 1).otherwise(0)
    )
    # Horario de pico (7-9h e 17-19h em dias uteis)
    .withColumn(
        "is_rush_hour",
        when(
            (col("is_weekend") == 0) &
            (
                ((col("pickup_hour") >= 7) & (col("pickup_hour") <= 9)) |
                ((col("pickup_hour") >= 17) & (col("pickup_hour") <= 19))
            ),
            1
        ).otherwise(0)
    )

    # --- FEATURES GEOGRAFICAS ---
    # Zona de pickup (arredondada para 2 decimais ~ 1.1km)
    .withColumn("pickup_zone_lat", spark_round(col("pickup_latitude"), 2))
    .withColumn("pickup_zone_lon", spark_round(col("pickup_longitude"), 2))
    # Zona de dropoff
    .withColumn("dropoff_zone_lat", spark_round(col("dropoff_latitude"), 2))
    .withColumn("dropoff_zone_lon", spark_round(col("dropoff_longitude"), 2))

    # --- FEATURES DE VIAGEM ---
    # Log da distancia (normalizar distribuicao)
    .withColumn("log_trip_distance", spark_round(log1p(col("trip_distance")), 4))
    # Log da duracao
    .withColumn("log_trip_duration", spark_round(log1p(col("trip_duration_min")), 4))
    # Ratio distancia/duracao (proxy de trafego)
    .withColumn(
        "distance_duration_ratio",
        spark_round(col("trip_distance") / col("trip_duration_min"), 4)
    )

    # --- FEATURES CATEGORICAS (codificadas) ---
    # Payment type: 1=Credit card, 2=Cash, etc.
    # Gorjetas em cash nao sao registradas -> filtrar apenas cartao
    .withColumn(
        "payment_is_credit",
        when(col("payment_type") == "1", 1)
        .when(col("payment_type") == "CRD", 1)
        .when(col("payment_type") == "Credit", 1)
        .otherwise(0)
    )

    # --- TARGET ---
    .withColumn("target_tip_amount", col("tip_amount"))

    # Selecionar apenas as colunas relevantes para ML
    .select(
        # Features temporais
        "pickup_hour",
        "pickup_day_of_week",
        "period_of_day",
        "is_weekend",
        "is_rush_hour",
        # Features geograficas
        "pickup_zone_lat",
        "pickup_zone_lon",
        "dropoff_zone_lat",
        "dropoff_zone_lon",
        # Features de viagem
        "trip_distance",
        "log_trip_distance",
        "trip_duration_min",
        "log_trip_duration",
        "avg_speed_mph",
        "cost_per_mile",
        "distance_duration_ratio",
        "fare_amount",
        "passenger_count",
        # Features categoricas
        "payment_is_credit",
        # Target
        "target_tip_amount",
    )

    # Filtrar apenas corridas com cartao de credito
    # (gorjetas em dinheiro nao sao registradas no dataset)
    .filter(col("payment_is_credit") == 1)

    # Remover outliers extremos de gorjeta (> $100 ou < 0)
    .filter(
        (col("target_tip_amount") >= 0) &
        (col("target_tip_amount") <= 100)
    )

    # Remover nulos
    .dropna()
)

# COMMAND ----------

print(f"Registros para ML: {df_features.count():,}")
print(f"Features: {len(df_features.columns) - 1}")  # -1 pelo target
print(f"\nColunas:")
for c in df_features.columns:
    print(f"  {'[TARGET]' if 'target' in c else '[FEATURE]'} {c}")

# COMMAND ----------

# Preview
display(df_features.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Estatisticas das Features

# COMMAND ----------

display(df_features.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlacao com o Target

# COMMAND ----------

# Correlacao de Pearson de cada feature com o target
import pyspark.sql.functions as F

feature_cols = [c for c in df_features.columns if c != "target_tip_amount"]
correlations = []

for feat in feature_cols:
    corr_val = df_features.stat.corr(feat, "target_tip_amount")
    correlations.append((feat, round(corr_val, 4)))

corr_df = spark.createDataFrame(correlations, ["feature", "correlation_with_tip"])
display(corr_df.orderBy(F.abs(col("correlation_with_tip")).desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gravar Feature Table

# COMMAND ----------

(
    df_features
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FEATURE_TABLE)
)

print(f"Feature Table {FEATURE_TABLE} criada com sucesso!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar Train/Test Split e salvar
# MAGIC
# MAGIC Salvamos splits separados para garantir reprodutibilidade.

# COMMAND ----------

# Split 80/20 com seed fixa para reprodutibilidade
df_train, df_test = df_features.randomSplit([0.8, 0.2], seed=42)

TRAIN_TABLE = f"{CATALOG}.{SCHEMA_ML}.taxi_tip_train"
TEST_TABLE = f"{CATALOG}.{SCHEMA_ML}.taxi_tip_test"

df_train.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TRAIN_TABLE)
df_test.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TEST_TABLE)

print(f"Train: {df_train.count():,} registros -> {TRAIN_TABLE}")
print(f"Test:  {df_test.count():,} registros -> {TEST_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pronto! Proximo: `05_automl_baseline` ou `06_mlflow_training`
