# Databricks notebook source
# MAGIC %md
# MAGIC # Model Registry -- Salvar e Versionar o Melhor Modelo
# MAGIC
# MAGIC **Tutorial 2: Machine Learning no Databricks Free Edition**
# MAGIC
# MAGIC **Objetivo:** Selecionar o melhor modelo do MLflow e prepara-lo
# MAGIC para uso em inferencia.
# MAGIC
# MAGIC **Nota sobre Free Edition:** O Unity Catalog Model Registry requer
# MAGIC permissoes de storage (S3 PutObject) que nao estao disponiveis
# MAGIC no Free Edition. Por isso, usamos o MLflow Tracking diretamente
# MAGIC para gerenciar nossos modelos -- que e exatamente como funciona
# MAGIC em muitas empresas com MLflow open-source.
# MAGIC
# MAGIC O que voce aprende:
# MAGIC - Consultar runs do MLflow programaticamente
# MAGIC - Selecionar o melhor modelo por metrica
# MAGIC - Carregar modelo salvo para inferencia
# MAGIC - Testar predicoes do modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os

os.environ["MLFLOW_TRACKING_URI"] = "databricks"

client = MlflowClient()

CATALOG = "workspace"
SCHEMA_ML = "medallion_ml"
EXPERIMENT_NAME = "/Users/thiagocharchar@gmail.com/nyc-taxi-tip-prediction"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recuperar referencia do melhor modelo

# COMMAND ----------

ref = spark.table(f"{CATALOG}.{SCHEMA_ML}.best_model_ref").collect()[0]
best_run_id = ref["run_id"]
best_mae = ref["best_mae"]
experiment_name = ref["experiment_name"]

print(f"Melhor Run ID: {best_run_id}")
print(f"Melhor MAE: ${best_mae:.4f}")
print(f"Experimento: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consultar detalhes do run no MLflow

# COMMAND ----------

run = client.get_run(best_run_id)

print("Detalhes do melhor run:")
print(f"  Run ID:    {run.info.run_id}")
print(f"  Run Name:  {run.data.tags.get('mlflow.runName', 'N/A')}")
print(f"  Status:    {run.info.status}")
print(f"\nParametros:")
for k, v in sorted(run.data.params.items()):
    if k not in ["feature_names", "random_state"]:
        print(f"  {k}: {v}")
print(f"\nMetricas:")
for k, v in sorted(run.data.metrics.items()):
    print(f"  {k}: {v:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Listar todos os runs do experimento
# MAGIC
# MAGIC Comparacao de todos os modelos treinados.

# COMMAND ----------

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.mae ASC"],
    max_results=20
)

print(f"Total de runs: {len(runs)}\n")
print(f"{'Run Name':<45} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
print("-" * 75)
for r in runs:
    name = r.data.tags.get("mlflow.runName", "?")
    mae = r.data.metrics.get("mae", float("nan"))
    rmse = r.data.metrics.get("rmse", float("nan"))
    r2 = r.data.metrics.get("r2", float("nan"))
    marker = " <-- BEST" if r.info.run_id == best_run_id else ""
    print(f"{name:<45} ${mae:>7.4f} ${rmse:>7.4f} {r2:>7.4f}{marker}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar o melhor modelo do MLflow

# COMMAND ----------

# Carregar modelo diretamente dos artefatos do MLflow
model_uri = f"runs:/{best_run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

print(f"Modelo carregado: {type(model).__name__}")
print(f"URI: {model_uri}")
print(f"\nParametros do modelo:")
for k, v in model.get_params().items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testar o modelo com dados reais

# COMMAND ----------

# Carregar amostra de teste
df_test = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_test").limit(100).toPandas()
TARGET = "target_tip_amount"
FEATURE_COLS = [c for c in df_test.columns if c != TARGET]

X_sample = df_test[FEATURE_COLS].values
y_actual = df_test[TARGET].values
y_pred = model.predict(X_sample)

mae = mean_absolute_error(y_actual, y_pred)
within_1 = np.mean(np.abs(y_pred - y_actual) < 1.0) * 100
within_2 = np.mean(np.abs(y_pred - y_actual) < 2.0) * 100

print(f"Teste com {len(y_actual)} amostras:")
print(f"  MAE: ${mae:.2f}")
print(f"  Dentro de $1: {within_1:.1f}%")
print(f"  Dentro de $2: {within_2:.1f}%")

print(f"\n{'Previsto':>10} {'Real':>10} {'Erro':>10}")
print("-" * 32)
for pred, actual in zip(y_pred[:10], y_actual[:10]):
    print(f"${pred:>8.2f}  ${actual:>8.2f}  ${abs(pred-actual):>8.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Salvar referencia do modelo para o notebook 08
# MAGIC
# MAGIC O model_uri (runs:/RUN_ID/model) e tudo que o proximo notebook
# MAGIC precisa para carregar o modelo.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_ML}.best_model_ref AS
    SELECT
        '{best_run_id}' as run_id,
        '{model_uri}' as model_uri,
        '{EXPERIMENT_NAME}' as experiment_name,
        '{type(model).__name__}' as model_type,
        {best_mae} as best_mae,
        current_timestamp() as registered_at
""")

print(f"Referencia atualizada em {CATALOG}.{SCHEMA_ML}.best_model_ref")
print(f"Model URI: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acesse o MLflow UI
# MAGIC
# MAGIC Va em: **Experiments** -> **nyc-taxi-tip-prediction**
# MAGIC
# MAGIC Voce pode:
# MAGIC - Ver todos os runs e comparar metricas
# MAGIC - Clicar em um run para ver artefatos (modelo, feature importance)
# MAGIC - Usar o Chart view para visualizar comparacoes
# MAGIC
# MAGIC ### Sobre o Unity Catalog Model Registry
# MAGIC
# MAGIC No Free Edition, o upload de modelos para o UC Model Registry
# MAGIC esta restrito por politicas de storage (S3). Em edicoes pagas,
# MAGIC voce usaria:
# MAGIC ```python
# MAGIC mlflow.set_registry_uri("databricks-uc")
# MAGIC mlflow.register_model(model_uri, "catalog.schema.model_name")
# MAGIC client.set_registered_model_alias(name, "Champion", version)
# MAGIC ```
# MAGIC
# MAGIC Por ora, o MLflow Tracking com `runs:/RUN_ID/model` funciona
# MAGIC perfeitamente para salvar, versionar e carregar modelos.
# MAGIC
# MAGIC Proximo: `08_batch_inference`
