# Databricks notebook source
# MAGIC %md
# MAGIC # AutoML Baseline -- Benchmark Rapido de Modelos
# MAGIC
# MAGIC **Tutorial 2: Machine Learning no Databricks Free Edition**
# MAGIC
# MAGIC **Objetivo:** Criar um baseline automatizado testando multiplos algoritmos
# MAGIC e logar tudo no MLflow. Simula a experiencia do Databricks AutoML
# MAGIC usando sklearn + MLflow.
# MAGIC
# MAGIC Modelos testados:
# MAGIC - Linear Regression (baseline simples)
# MAGIC - Decision Tree
# MAGIC - Random Forest
# MAGIC - Gradient Boosting (XGBoost-like)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# Configurar MLflow para usar Unity Catalog como registry
# (necessario no serverless do Free Edition)
mlflow.set_registry_uri("databricks-uc")
os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# Configurar experimento MLflow
EXPERIMENT_NAME = "/Users/thiagocharchar@gmail.com/nyc-taxi-tip-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"Experimento MLflow: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Carregar dados de treino e teste

# COMMAND ----------

CATALOG = "workspace"
SCHEMA_ML = "medallion_ml"

# Carregar como Spark DataFrames
df_train_spark = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_train")
df_test_spark = spark.table(f"{CATALOG}.{SCHEMA_ML}.taxi_tip_test")

print(f"Train: {df_train_spark.count():,} registros")
print(f"Test:  {df_test_spark.count():,} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparar dados para sklearn
# MAGIC
# MAGIC Como estamos no serverless (sem cluster ML dedicado),
# MAGIC usamos sklearn com dados em pandas. Para datasets grandes,
# MAGIC vamos amostrar para caber em memoria.

# COMMAND ----------

# Amostrar se necessario (sklearn precisa caber em memoria)
MAX_TRAIN_ROWS = 500_000
MAX_TEST_ROWS = 100_000

train_count = df_train_spark.count()
test_count = df_test_spark.count()

if train_count > MAX_TRAIN_ROWS:
    sample_fraction = MAX_TRAIN_ROWS / train_count
    df_train_pd = df_train_spark.sample(fraction=sample_fraction, seed=42).toPandas()
    print(f"Train amostrado: {len(df_train_pd):,} de {train_count:,} ({sample_fraction:.1%})")
else:
    df_train_pd = df_train_spark.toPandas()
    print(f"Train completo: {len(df_train_pd):,}")

if test_count > MAX_TEST_ROWS:
    sample_fraction = MAX_TEST_ROWS / test_count
    df_test_pd = df_test_spark.sample(fraction=sample_fraction, seed=42).toPandas()
    print(f"Test amostrado: {len(df_test_pd):,} de {test_count:,}")
else:
    df_test_pd = df_test_spark.toPandas()
    print(f"Test completo: {len(df_test_pd):,}")

# COMMAND ----------

# Separar features e target
TARGET = "target_tip_amount"
FEATURE_COLS = [c for c in df_train_pd.columns if c != TARGET]

X_train = df_train_pd[FEATURE_COLS].values
y_train = df_train_pd[TARGET].values
X_test = df_test_pd[FEATURE_COLS].values
y_test = df_test_pd[TARGET].values

print(f"Features: {len(FEATURE_COLS)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Target mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definir modelos para o benchmark

# COMMAND ----------

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=20,
        random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=20,
        n_jobs=-1,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
}

print(f"Modelos a testar: {list(models.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Treinar e logar no MLflow
# MAGIC
# MAGIC Para cada modelo:
# MAGIC 1. Treina com os dados de treino
# MAGIC 2. Avalia no conjunto de teste
# MAGIC 3. Loga parametros, metricas e artefatos no MLflow

# COMMAND ----------

def evaluate_model(model, X_test, y_test):
    """Calcula metricas de regressao."""
    y_pred = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mape": np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.01))) * 100,
    }

# COMMAND ----------

results = []

for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Treinando: {model_name}")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=f"automl_{model_name}"):
        # Logar parametros
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("feature_names", str(FEATURE_COLS))

        # Logar hiperparametros do modelo
        params = model.get_params()
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass  # Ignorar parametros nao-serializaveis

        # Treinar
        model.fit(X_train, y_train)

        # Avaliar
        metrics = evaluate_model(model, X_test, y_test)

        # Logar metricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")

        # Logar modelo
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_test[:5],
        )

        # Feature importance (se disponivel)
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": FEATURE_COLS,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            mlflow.log_table(importance, artifact_file="feature_importance.json")
            print(f"\n  Top 5 features:")
            for _, row in importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")

        results.append({
            "model": model_name,
            **metrics
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparacao dos Modelos

# COMMAND ----------

results_df = pd.DataFrame(results).sort_values("mae")
print("\nRanking dos Modelos (por MAE - menor e melhor):\n")
print(results_df.to_string(index=False))

# COMMAND ----------

# Visualizar como Spark DataFrame para display interativo
display(spark.createDataFrame(results_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Melhor modelo

# COMMAND ----------

best = results_df.iloc[0]
print(f"Melhor modelo: {best['model']}")
print(f"  MAE:  ${best['mae']:.2f} (erro medio absoluto)")
print(f"  RMSE: ${best['rmse']:.2f} (erro medio quadratico)")
print(f"  R2:   {best['r2']:.4f} (variancia explicada)")
print(f"\nInterpretacao: o modelo erra em media ${best['mae']:.2f} na previsao da gorjeta.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Acesse o MLflow UI
# MAGIC
# MAGIC Para ver todos os experimentos, va no menu lateral:
# MAGIC **Experiments** -> **nyc-taxi-tip-prediction**
# MAGIC
# MAGIC La voce pode:
# MAGIC - Comparar runs lado a lado
# MAGIC - Ver graficos de metricas
# MAGIC - Inspecionar artefatos (modelos, feature importance)
# MAGIC - Escolher o melhor modelo para registro

# COMMAND ----------

# MAGIC %md
# MAGIC ## Proximo: `06_mlflow_training` (treinamento manual com tuning)
