# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Training -- Treinamento Manual com Hyperparameter Tuning
# MAGIC
# MAGIC **Tutorial 2: Machine Learning no Databricks Free Edition**
# MAGIC
# MAGIC **Objetivo:** Treinar o melhor modelo (Gradient Boosting) com tuning
# MAGIC de hiperparametros, usando MLflow para tracking completo.
# MAGIC
# MAGIC O que voce aprende:
# MAGIC - Nested runs no MLflow (parent run + child runs)
# MAGIC - Grid search com logging automatico
# MAGIC - Comparacao visual no MLflow UI
# MAGIC - Selecao do melhor modelo

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

# Configurar MLflow para usar Unity Catalog como registry
mlflow.set_registry_uri("databricks-uc")
os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# Configurar experimento
EXPERIMENT_NAME = "/Users/thiagocharchar@gmail.com/nyc-taxi-tip-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar dados

# COMMAND ----------

CATALOG = "workspace"
SCHEMA_ML = "medallion_ml"

# Amostrar para caber na memoria do serverless Free Edition
MAX_TRAIN_ROWS = 50_000
MAX_TEST_ROWS = 10_000

df_train_spark = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_train")
df_test_spark = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_test")

train_count = df_train_spark.count()
test_count = df_test_spark.count()

train_frac = min(1.0, MAX_TRAIN_ROWS / train_count)
test_frac = min(1.0, MAX_TEST_ROWS / test_count)

df_train_pd = df_train_spark.sample(fraction=train_frac, seed=42).toPandas()
df_test_pd = df_test_spark.sample(fraction=test_frac, seed=42).toPandas()

TARGET = "target_tip_amount"
FEATURE_COLS = [c for c in df_train_pd.columns if c != TARGET]

X_train = df_train_pd[FEATURE_COLS].values
y_train = df_train_pd[TARGET].values
X_test = df_test_pd[FEATURE_COLS].values
y_test = df_test_pd[TARGET].values

print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Features: {X_train.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definir grid de hiperparametros
# MAGIC
# MAGIC Vamos testar combinacoes de:
# MAGIC - `n_estimators`: numero de arvores
# MAGIC - `max_depth`: profundidade maxima
# MAGIC - `learning_rate`: taxa de aprendizado
# MAGIC - `subsample`: fracao de dados por arvore

# COMMAND ----------

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
}

grid = list(ParameterGrid(param_grid))
print(f"Total de combinacoes: {len(grid)}")
for i, params in enumerate(grid):
    print(f"  [{i+1}] {params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinamento com Nested Runs
# MAGIC
# MAGIC Usamos um **parent run** que contem todos os **child runs** (uma por combinacao).
# MAGIC Isso organiza o experimento no MLflow UI.

# COMMAND ----------

best_mae = float("inf")
best_run_id = None
best_params = None
all_results = []

# Parent run
with mlflow.start_run(run_name="hyperparameter_tuning_gb") as parent_run:
    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_combinations", len(grid))
    mlflow.log_param("n_train_samples", len(X_train))
    mlflow.log_param("n_test_samples", len(X_test))

    for i, params in enumerate(grid):
        run_name = f"gb_lr{params['learning_rate']}_d{params['max_depth']}_n{params['n_estimators']}"

        # Child run
        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            print(f"\n[{i+1}/{len(grid)}] {run_name}")
            start_time = time.time()

            # Logar parametros
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("random_state", 42)

            # Treinar
            model = GradientBoostingRegressor(**params, random_state=42)
            model.fit(X_train, y_train)

            # Avaliar
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            train_time = time.time() - start_time

            # Logar metricas
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("training_time_sec", train_time)

            # Logar modelo
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Feature importance
            importance = pd.DataFrame({
                "feature": FEATURE_COLS,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            mlflow.log_table(importance, artifact_file="feature_importance.json")

            print(f"  MAE: ${mae:.4f} | RMSE: ${rmse:.4f} | R2: {r2:.4f} | Time: {train_time:.1f}s")

            # Track best
            result = {"run_id": child_run.info.run_id, "run_name": run_name, **params, "mae": mae, "rmse": rmse, "r2": r2, "time_sec": train_time}
            all_results.append(result)

            if mae < best_mae:
                best_mae = mae
                best_run_id = child_run.info.run_id
                best_params = params.copy()
                best_model = model

    # Logar resultado final no parent
    mlflow.log_metric("best_mae", best_mae)
    mlflow.log_param("best_run_id", best_run_id)
    for k, v in best_params.items():
        mlflow.log_param(f"best_{k}", v)

    # Logar tabela de resultados
    results_df = pd.DataFrame(all_results).sort_values("mae")
    mlflow.log_table(results_df, artifact_file="tuning_results.json")

    parent_run_id = parent_run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultados do Tuning

# COMMAND ----------

print("Ranking dos modelos (por MAE):\n")
display(spark.createDataFrame(results_df.drop(columns=["run_id"])))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Melhor Modelo

# COMMAND ----------

print(f"Melhor modelo encontrado:")
print(f"  Run ID: {best_run_id}")
print(f"  Parametros: {best_params}")
print(f"  MAE:  ${best_mae:.4f}")
print(f"\nTop 5 features mais importantes:")
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": best_model.feature_importances_
}).sort_values("importance", ascending=False)

for _, row in importance.head(5).iterrows():
    bar = "#" * int(row["importance"] * 100)
    print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Salvar referencia do melhor modelo
# MAGIC
# MAGIC Salvamos o run_id para uso no proximo notebook (Model Registry).

# COMMAND ----------

# Salvar run_id como widget/param para o proximo notebook
spark.sql(f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_ML}.best_model_ref AS
    SELECT
        '{best_run_id}' as run_id,
        '{parent_run_id}' as parent_run_id,
        '{EXPERIMENT_NAME}' as experiment_name,
        {best_mae} as best_mae,
        current_timestamp() as registered_at
""")

print(f"Referencia salva em {CATALOG}.{SCHEMA_ML}.best_model_ref")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize no MLflow UI
# MAGIC
# MAGIC Va em: **Experiments** -> **nyc-taxi-tip-prediction**
# MAGIC
# MAGIC Voce vera:
# MAGIC - O **parent run** `hyperparameter_tuning_gb` com todos os child runs
# MAGIC - Clique em "Chart" para comparar metricas visualmente
# MAGIC - Cada child run tem o modelo treinado como artefato
# MAGIC
# MAGIC Proximo: `07_model_registry`
