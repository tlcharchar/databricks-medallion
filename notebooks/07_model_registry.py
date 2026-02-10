# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registry -- Registrar e Versionar Modelos
# MAGIC
# MAGIC **Tutorial 2: Machine Learning no Databricks Free Edition**
# MAGIC
# MAGIC **Objetivo:** Registrar o melhor modelo no Unity Catalog Model Registry,
# MAGIC criar versoes e gerenciar o ciclo de vida do modelo.
# MAGIC
# MAGIC O que voce aprende:
# MAGIC - Registrar modelos no Unity Catalog (nao no legacy Workspace Registry)
# MAGIC - Criar aliases (Champion, Challenger)
# MAGIC - Carregar modelos do registry para inferencia
# MAGIC - Versionamento automatico

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Configurar para usar Unity Catalog como model registry
mlflow.set_registry_uri("databricks-uc")

client = MlflowClient()

CATALOG = "workspace"
SCHEMA_ML = "medallion_ml"
MODEL_NAME = f"{CATALOG}.{SCHEMA_ML}.nyc_taxi_tip_model"

print(f"Model Registry: Unity Catalog")
print(f"Model Name: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recuperar referencia do melhor modelo

# COMMAND ----------

# Carregar referencia salva no notebook anterior
ref = spark.table(f"{CATALOG}.{SCHEMA_ML}.best_model_ref").collect()[0]
best_run_id = ref["run_id"]
best_mae = ref["best_mae"]

print(f"Melhor Run ID: {best_run_id}")
print(f"Melhor MAE: ${best_mae:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registrar modelo no Unity Catalog

# COMMAND ----------

# Registrar o modelo do melhor run
model_uri = f"runs:/{best_run_id}/model"
print(f"Registrando modelo de: {model_uri}")

result = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print(f"\nModelo registrado com sucesso!")
print(f"  Nome: {result.name}")
print(f"  Versao: {result.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definir alias "Champion"
# MAGIC
# MAGIC No Unity Catalog, usamos **aliases** em vez de stages (Staging/Production).
# MAGIC - `Champion`: modelo em producao
# MAGIC - `Challenger`: candidato a substituir o Champion

# COMMAND ----------

# Definir o alias "Champion" para esta versao
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=result.version
)

print(f"Alias 'Champion' definido para versao {result.version}")

# COMMAND ----------

# Adicionar descricao ao modelo
client.update_registered_model(
    name=MODEL_NAME,
    description="Modelo de previsao de gorjeta para corridas de taxi NYC. "
                "Treinado com Gradient Boosting sobre features temporais, "
                "geograficas e de viagem. Target: tip_amount (USD)."
)

# Adicionar descricao a versao
client.update_model_version(
    name=MODEL_NAME,
    version=result.version,
    description=f"Baseline GradientBoosting. MAE: ${best_mae:.4f}. "
                f"Treinado com dados da Silver (taxi_trips_cleaned)."
)

print("Descricoes adicionadas ao modelo e versao.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verificar modelo registrado

# COMMAND ----------

# Listar versoes do modelo
model_info = client.get_registered_model(MODEL_NAME)
print(f"Modelo: {model_info.name}")
print(f"Descricao: {model_info.description}")
print(f"\nVersoes:")

for alias in model_info.aliases:
    print(f"  Alias '{alias}' -> versao {model_info.aliases[alias]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar modelo do Registry (teste)
# MAGIC
# MAGIC Demonstra como carregar o modelo Champion para uso em inferencia.

# COMMAND ----------

# Carregar modelo usando alias
champion_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@Champion")
print(f"Modelo Champion carregado: {type(champion_model).__name__}")

# Teste rapido com dados de exemplo
import pandas as pd

df_test = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_test").limit(5).toPandas()
TARGET = "target_tip_amount"
FEATURE_COLS = [c for c in df_test.columns if c != TARGET]

X_sample = df_test[FEATURE_COLS].values
y_actual = df_test[TARGET].values
y_pred = champion_model.predict(X_sample)

print(f"\nTeste do modelo Champion:")
print(f"{'Previsto':>10} {'Real':>10} {'Erro':>10}")
print("-" * 32)
for pred, actual in zip(y_pred, y_actual):
    print(f"${pred:>8.2f}  ${actual:>8.2f}  ${abs(pred-actual):>8.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acesse o Model Registry no UI
# MAGIC
# MAGIC Va em: **Catalog** -> **workspace** -> **medallion_ml** -> **Models**
# MAGIC
# MAGIC Ou: menu lateral **Models**
# MAGIC
# MAGIC Voce vera:
# MAGIC - O modelo `nyc_taxi_tip_model` registrado
# MAGIC - Versao 1 com alias "Champion"
# MAGIC - Link para o run do MLflow que gerou o modelo
# MAGIC - Descricao e metadados
# MAGIC
# MAGIC Proximo: `08_batch_inference`
