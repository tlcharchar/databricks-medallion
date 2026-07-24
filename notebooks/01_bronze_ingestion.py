# Databricks notebook source
# MAGIC %md
# MAGIC # 🥉 Bronze Layer — Raw Ingestion
# MAGIC
# MAGIC **Objetivo:** Ingerir dados brutos do NYC Taxi dataset para a camada Bronze (Delta Table).
# MAGIC
# MAGIC - Sem transformações
# MAGIC - Dados como vieram da fonte
# MAGIC - Adição de metadados de ingestão (timestamp, source)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup — Configuração do Catalog e Schema

# COMMAND ----------

# Configuração
CATALOG = "urban_mobility"
SCHEMA_BRONZE = "medallion_bronze"

# Criar schema Bronze se não existir
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA_BRONZE}")
print(f"Schema {CATALOG}.{SCHEMA_BRONZE} pronto.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explorar dados de origem
# MAGIC
# MAGIC O Databricks já vem com o dataset NYC Taxi disponível em `/databricks-datasets/`.

# COMMAND ----------

# Listar arquivos disponíveis do NYC Taxi
dbutils.fs.ls("/databricks-datasets/nyctaxi/tables/nyctaxi_yellow")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingestão — Leitura e gravação na Bronze

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, lit

# Ler dados brutos
SOURCE_PATH = "/databricks-datasets/nyctaxi/tables/nyctaxi_yellow"

df_raw = spark.read.format("delta").load(SOURCE_PATH)

print(f"Total de registros na fonte: {df_raw.count():,}")
print(f"Colunas: {df_raw.columns}")

# COMMAND ----------

# Preview dos dados
display(df_raw.limit(10))

# COMMAND ----------

# Adicionar metadados de ingestão
df_bronze = (
    df_raw
    .withColumn("_ingestion_timestamp", current_timestamp())
    .withColumn("_source", lit("nyctaxi_yellow"))
)

# COMMAND ----------

# Gravar na camada Bronze como Delta Table
BRONZE_TABLE = f"{CATALOG}.{SCHEMA_BRONZE}.taxi_trips_raw"

(
    df_bronze
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

print(f"Tabela {BRONZE_TABLE} criada com sucesso!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validação

# COMMAND ----------

# Verificar a tabela criada
df_check = spark.table(BRONZE_TABLE)
print(f"Registros na Bronze: {df_check.count():,}")
print(f"Schema:")
df_check.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Bronze completa! Próximo: `02_silver_transformation`
