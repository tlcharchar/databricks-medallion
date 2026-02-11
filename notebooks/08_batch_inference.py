# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Inference -- Predicoes em Escala + Gold ML Table
# MAGIC
# MAGIC **Tutorial 2: Machine Learning no Databricks Free Edition**
# MAGIC
# MAGIC **Objetivo:** Usar o modelo Champion do Registry para gerar predicoes
# MAGIC em batch sobre os dados, criando uma nova tabela Gold com insights de ML.
# MAGIC
# MAGIC O que voce aprende:
# MAGIC - Carregar modelo do Unity Catalog Registry
# MAGIC - Aplicar modelo em batch com Spark UDF
# MAGIC - Criar tabela Gold com predicoes para analytics
# MAGIC - Analise de erro e confiabilidade do modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
from pyspark.sql.functions import (
    col, abs as spark_abs, round as spark_round,
    avg, count, when, current_timestamp,
)
import pandas as pd
import numpy as np
import os

os.environ["MLFLOW_TRACKING_URI"] = "databricks"

CATALOG = "workspace"
SCHEMA_ML = "medallion_ml"
SCHEMA_GOLD = "medallion_gold"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar melhor modelo do MLflow

# COMMAND ----------

# Recuperar model_uri salvo no notebook 07
ref = spark.table(f"{CATALOG}.{SCHEMA_ML}.best_model_ref").collect()[0]
model_uri = ref["model_uri"]
print(f"Carregando modelo de: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)
print(f"Modelo carregado: {type(model).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar dados de teste para inferencia

# COMMAND ----------

# Amostrar para caber na memoria do serverless Free Edition
MAX_INFERENCE_ROWS = 50_000
df_test_full = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_test")
test_count = df_test_full.count()
sample_frac = min(1.0, MAX_INFERENCE_ROWS / test_count)
df_test = df_test_full.sample(fraction=sample_frac, seed=42)
print(f"Registros para inferencia: {df_test.count():,} (amostrado de {test_count:,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aplicar modelo em batch
# MAGIC
# MAGIC Usamos `pandas_udf` para aplicar o modelo sklearn
# MAGIC de forma distribuida no Spark.

# COMMAND ----------

TARGET = "target_tip_amount"
FEATURE_COLS = [c for c in df_test.columns if c != TARGET]

# Converter para Pandas para predicao
# (No Free Edition serverless, funciona bem para datasets moderados)
df_test_pd = df_test.toPandas()

# Gerar predicoes
df_test_pd["predicted_tip"] = model.predict(df_test_pd[FEATURE_COLS].values)
df_test_pd["prediction_error"] = df_test_pd["predicted_tip"] - df_test_pd["target_tip_amount"]
df_test_pd["absolute_error"] = abs(df_test_pd["prediction_error"])

# Converter de volta para Spark
df_predictions = spark.createDataFrame(df_test_pd)

print(f"Predicoes geradas: {df_predictions.count():,}")

# COMMAND ----------

# Preview
display(
    df_predictions
    .select("target_tip_amount", "predicted_tip", "prediction_error", "absolute_error",
            "trip_distance", "fare_amount", "pickup_hour", "is_weekend")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analise de Erro do Modelo

# COMMAND ----------

# Metricas gerais
error_stats = df_predictions.agg(
    spark_round(avg("absolute_error"), 4).alias("mae"),
    spark_round(avg("prediction_error"), 4).alias("mean_error_bias"),
    count("*").alias("total_predictions"),
    spark_round(
        (count(when(col("absolute_error") < 1, True)) / count("*") * 100), 2
    ).alias("pct_within_1_dollar"),
    spark_round(
        (count(when(col("absolute_error") < 2, True)) / count("*") * 100), 2
    ).alias("pct_within_2_dollars"),
).collect()[0]

print("Metricas de Erro do Modelo:")
print(f"  MAE: ${error_stats['mae']}")
print(f"  Bias medio: ${error_stats['mean_error_bias']}")
print(f"  Total predicoes: {error_stats['total_predictions']:,}")
print(f"  Dentro de $1: {error_stats['pct_within_1_dollar']}%")
print(f"  Dentro de $2: {error_stats['pct_within_2_dollars']}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Erro por faixa horaria

# COMMAND ----------

error_by_hour = (
    df_predictions
    .groupBy("pickup_hour")
    .agg(
        spark_round(avg("absolute_error"), 4).alias("avg_error"),
        spark_round(avg("target_tip_amount"), 4).alias("avg_actual_tip"),
        spark_round(avg("predicted_tip"), 4).alias("avg_predicted_tip"),
        count("*").alias("n_trips")
    )
    .orderBy("pickup_hour")
)
display(error_by_hour)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Criar tabela Gold com predicoes
# MAGIC
# MAGIC Esta tabela combina dados reais + predicoes para uso
# MAGIC em dashboards e Genie Spaces.

# COMMAND ----------

# Agregar predicoes por dia, hora e zona
df_gold_ml = (
    df_predictions
    .withColumn("pickup_zone_lat_str", col("pickup_zone_lat").cast("string"))
    .withColumn("pickup_zone_lon_str", col("pickup_zone_lon").cast("string"))
    .groupBy("pickup_hour", "pickup_day_of_week", "is_weekend", "period_of_day")
    .agg(
        count("*").alias("total_trips"),
        spark_round(avg("target_tip_amount"), 2).alias("avg_actual_tip"),
        spark_round(avg("predicted_tip"), 2).alias("avg_predicted_tip"),
        spark_round(avg("absolute_error"), 2).alias("avg_prediction_error"),
        spark_round(avg("fare_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance"),
        spark_round(
            (count(when(col("absolute_error") < 1, True)) / count("*") * 100), 1
        ).alias("pct_accurate_within_1usd"),
    )
    .withColumn(
        "day_name",
        when(col("pickup_day_of_week") == 1, "Sunday")
        .when(col("pickup_day_of_week") == 2, "Monday")
        .when(col("pickup_day_of_week") == 3, "Tuesday")
        .when(col("pickup_day_of_week") == 4, "Wednesday")
        .when(col("pickup_day_of_week") == 5, "Thursday")
        .when(col("pickup_day_of_week") == 6, "Friday")
        .when(col("pickup_day_of_week") == 7, "Saturday")
    )
    .withColumn(
        "period_name",
        when(col("period_of_day") == 0, "Madrugada (0-6h)")
        .when(col("period_of_day") == 1, "Manha (6-12h)")
        .when(col("period_of_day") == 2, "Tarde (12-18h)")
        .when(col("period_of_day") == 3, "Noite (18-24h)")
    )
    .withColumn("_gold_timestamp", current_timestamp())
    .orderBy("pickup_day_of_week", "pickup_hour")
)

GOLD_ML_TABLE = f"{CATALOG}.{SCHEMA_GOLD}.taxi_tip_predictions"
df_gold_ml.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_ML_TABLE)
print(f"Tabela {GOLD_ML_TABLE} criada com {df_gold_ml.count()} registros")

# COMMAND ----------

display(df_gold_ml)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Queries de exemplo sobre as predicoes

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Quando o modelo acerta mais?
# MAGIC SELECT
# MAGIC   period_name,
# MAGIC   ROUND(AVG(avg_prediction_error), 2) as avg_error,
# MAGIC   ROUND(AVG(pct_accurate_within_1usd), 1) as pct_accurate,
# MAGIC   SUM(total_trips) as total_trips
# MAGIC FROM workspace.medallion_gold.taxi_tip_predictions
# MAGIC GROUP BY period_name
# MAGIC ORDER BY avg_error

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Gorjeta prevista: fim de semana vs dia util
# MAGIC SELECT
# MAGIC   CASE WHEN is_weekend = 1 THEN 'Fim de semana' ELSE 'Dia util' END as periodo,
# MAGIC   ROUND(AVG(avg_actual_tip), 2) as gorjeta_real,
# MAGIC   ROUND(AVG(avg_predicted_tip), 2) as gorjeta_prevista,
# MAGIC   ROUND(AVG(avg_prediction_error), 2) as erro_medio
# MAGIC FROM workspace.medallion_gold.taxi_tip_predictions
# MAGIC GROUP BY is_weekend

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo do Pipeline ML Completo
# MAGIC
# MAGIC ```
# MAGIC Silver (taxi_trips_cleaned)
# MAGIC    |
# MAGIC    v
# MAGIC Feature Engineering (04) --> taxi_tip_features, _train, _test
# MAGIC    |
# MAGIC    v
# MAGIC AutoML Baseline (05) --> MLflow Experiments (4 modelos)
# MAGIC    |
# MAGIC    v
# MAGIC Hyperparameter Tuning (06) --> MLflow nested runs, best_model_ref
# MAGIC    |
# MAGIC    v
# MAGIC Model Registry (07) --> UC: nyc_taxi_tip_model @Champion
# MAGIC    |
# MAGIC    v
# MAGIC Batch Inference (08) --> Gold: taxi_tip_predictions
# MAGIC    |
# MAGIC    v
# MAGIC Genie Space --> "Qual a gorjeta prevista para corridas noturnas?"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Proximo passo: Atualizar o Genie Space
# MAGIC
# MAGIC Adicione a tabela `workspace.medallion_gold.taxi_tip_predictions`
# MAGIC ao seu Genie Space e adicione estas instrucoes:
# MAGIC
# MAGIC ```
# MAGIC A tabela taxi_tip_predictions contem predicoes de ML sobre gorjetas.
# MAGIC - avg_actual_tip = gorjeta real media
# MAGIC - avg_predicted_tip = gorjeta prevista pelo modelo
# MAGIC - avg_prediction_error = erro medio do modelo
# MAGIC - pct_accurate_within_1usd = % de predicoes com erro < $1
# MAGIC - period_name = periodo do dia (Madrugada, Manha, Tarde, Noite)
# MAGIC ```
