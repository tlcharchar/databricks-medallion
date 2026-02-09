# Databricks notebook source
# MAGIC %md
# MAGIC # 🥇 Gold Layer — Agregações de Negócio
# MAGIC
# MAGIC **Objetivo:** Criar tabelas agregadas prontas para consumo por dashboards, SQL Analytics e Genie Spaces.
# MAGIC
# MAGIC Tabelas criadas:
# MAGIC - `taxi_daily_metrics` — Métricas diárias (receita, corridas, médias)
# MAGIC - `taxi_zone_metrics` — Performance por zona (pickup/dropoff)
# MAGIC - `taxi_hourly_metrics` — Padrões por hora do dia e dia da semana

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, min as spark_min, max as spark_max,
    round as spark_round, percentile_approx, current_timestamp, when
)

CATALOG = "workspace"
SCHEMA_SILVER = "medallion_silver"
SCHEMA_GOLD = "medallion_gold"

# Criar schema Gold se não existir
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_GOLD}")

SILVER_TABLE = f"{CATALOG}.{SCHEMA_SILVER}.taxi_trips_cleaned"

print(f"Lendo de: {SILVER_TABLE}")

# COMMAND ----------

df_silver = spark.table(SILVER_TABLE)
print(f"Registros na Silver: {df_silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 1 — Métricas Diárias

# COMMAND ----------

df_daily = (
    df_silver
    .groupBy("pickup_date")
    .agg(
        count("*").alias("total_trips"),
        spark_round(spark_sum("fare_amount"), 2).alias("total_revenue"),
        spark_round(spark_sum("tip_amount"), 2).alias("total_tips"),
        spark_round(avg("fare_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance_miles"),
        spark_round(avg("trip_duration_min"), 2).alias("avg_duration_min"),
        spark_round(avg("tip_percentage"), 2).alias("avg_tip_pct"),
        spark_round(avg("avg_speed_mph"), 2).alias("avg_speed_mph"),
        spark_round(avg("passenger_count"), 1).alias("avg_passengers"),
        spark_round(spark_sum("total_amount"), 2).alias("total_amount_collected")
    )
    .withColumn("_gold_timestamp", current_timestamp())
    .orderBy("pickup_date")
)

GOLD_DAILY = f"{CATALOG}.{SCHEMA_GOLD}.taxi_daily_metrics"
df_daily.write.format("delta").mode("overwrite").saveAsTable(GOLD_DAILY)
print(f"Tabela {GOLD_DAILY} criada — {df_daily.count()} registros")

display(df_daily.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 2 — Métricas por Zona Geográfica
# MAGIC
# MAGIC Como o dataset usa lat/long (não Location IDs), criamos zonas
# MAGIC arredondando coordenadas para 2 decimais (~1.1 km de precisão).

# COMMAND ----------

df_zone = (
    df_silver
    # Criar zona geográfica arredondando lat/long (2 decimais ≈ 1.1km)
    .withColumn("zone_lat", spark_round(col("pickup_latitude"), 2))
    .withColumn("zone_lon", spark_round(col("pickup_longitude"), 2))
    .groupBy("zone_lat", "zone_lon")
    .agg(
        count("*").alias("total_trips"),
        spark_round(spark_sum("fare_amount"), 2).alias("total_revenue"),
        spark_round(avg("fare_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance_miles"),
        spark_round(avg("tip_percentage"), 2).alias("avg_tip_pct"),
        spark_round(avg("trip_duration_min"), 2).alias("avg_duration_min")
    )
    .withColumn("_gold_timestamp", current_timestamp())
    .orderBy(col("total_trips").desc())
)

GOLD_ZONE = f"{CATALOG}.{SCHEMA_GOLD}.taxi_zone_metrics"
df_zone.write.format("delta").mode("overwrite").saveAsTable(GOLD_ZONE)
print(f"Tabela {GOLD_ZONE} criada — {df_zone.count()} registros")

display(df_zone.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 3 — Métricas por Hora e Dia da Semana

# COMMAND ----------

df_hourly = (
    df_silver
    .groupBy("pickup_hour", "pickup_day_of_week")
    .agg(
        count("*").alias("total_trips"),
        spark_round(avg("fare_amount"), 2).alias("avg_fare"),
        spark_round(avg("trip_distance"), 2).alias("avg_distance_miles"),
        spark_round(avg("tip_percentage"), 2).alias("avg_tip_pct"),
        spark_round(avg("trip_duration_min"), 2).alias("avg_duration_min"),
        spark_round(avg("avg_speed_mph"), 2).alias("avg_speed_mph")
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
    .withColumn("_gold_timestamp", current_timestamp())
    .orderBy("pickup_day_of_week", "pickup_hour")
)

GOLD_HOURLY = f"{CATALOG}.{SCHEMA_GOLD}.taxi_hourly_metrics"
df_hourly.write.format("delta").mode("overwrite").saveAsTable(GOLD_HOURLY)
print(f"Tabela {GOLD_HOURLY} criada — {df_hourly.count()} registros")

display(df_hourly.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resumo das tabelas Gold

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN workspace.medallion_gold

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Gold completa!
# MAGIC
# MAGIC Próximos passos:
# MAGIC - Criar queries no **SQL Editor**
# MAGIC - Configurar **Genie Space** para "fale com seus dados"
