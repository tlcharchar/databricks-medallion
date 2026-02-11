# Databricks Medallion Architecture + ML Pipeline

Pipeline de dados e Machine Learning end-to-end no **Databricks Free Edition**, usando NYC Taxi Dataset.

## O que este projeto cobre

**Tutorial 1 -- Medallion Architecture** (Bronze, Silver, Gold, SQL Analytics, Genie Spaces)

**Tutorial 2 -- Machine Learning** (Feature Engineering, MLflow Tracking, Hyperparameter Tuning, Batch Inference)

Os tutoriais completos em PDF estao na raiz do repositorio:
- `Tutorial_Medallion_Databricks.pdf`
- `Tutorial_02_ML_Databricks.pdf`

---

## Arquitetura

```
NYC Taxi Dataset (/databricks-datasets)
        |
  +-------------+
  |   BRONZE    |  Dados brutos + metadados de ingestao
  +------+------+
         |
  +-------------+
  |   SILVER    |  Dados limpos, tipados, enriquecidos
  +------+------+
         |
   +-----+------------------+
   |                        |
  +--------+         +-------------+
  |  GOLD  |         |   ML        |
  +--------+         +-------------+
  Metricas           Feature Eng.
  agregadas          MLflow Training
  por dia/           Hyperparameter Tuning
  zona/hora          Batch Inference
   |                        |
   +-----+------------------+
         |
  +------------------+
  |  SQL + Genie     |  Dashboards + "Fale com seus dados"
  +------------------+
```

---

## Estrutura do Projeto

```
databricks-medallion/
|-- notebooks/
|   |-- 01_bronze_ingestion.py         # Ingestao raw -> Delta
|   |-- 02_silver_transformation.py    # Limpeza e transformacoes
|   |-- 03_gold_aggregation.py         # Agregacoes de negocio
|   |-- 04_feature_engineering.py      # Silver -> ML features
|   |-- 05_automl_baseline.py          # 4 modelos sklearn + MLflow
|   |-- 06_mlflow_training.py          # Hyperparameter tuning (Grid Search)
|   |-- 07_model_registry.py           # Selecao do melhor modelo
|   |-- 08_batch_inference.py          # Predicoes em batch -> Gold
|-- sql/
|   |-- gold_queries.sql               # Queries analiticas
|-- config/
|   |-- pipeline_config.yaml           # Configuracao do pipeline
|-- Tutorial_Medallion_Databricks.pdf  # Tutorial 1 (PDF)
|-- Tutorial_02_ML_Databricks.pdf      # Tutorial 2 (PDF)
|-- README.md
```

---

## Tabelas

### Tutorial 1 -- Medallion

| Camada | Tabela | Descricao |
|--------|--------|-----------|
| Bronze | `medallion_bronze.taxi_trips_raw` | Dados brutos do NYC Taxi |
| Silver | `medallion_silver.taxi_trips_cleaned` | Dados limpos e enriquecidos |
| Gold | `medallion_gold.taxi_daily_metrics` | Metricas diarias |
| Gold | `medallion_gold.taxi_zone_metrics` | Metricas por zona geografica |
| Gold | `medallion_gold.taxi_hourly_metrics` | Metricas por hora/dia da semana |

### Tutorial 2 -- Machine Learning

| Camada | Tabela | Descricao |
|--------|--------|-----------|
| ML | `medallion_ml.taxi_tip_features` | Feature table completa (19 features) |
| ML | `medallion_ml.taxi_tip_train` | Dados de treinamento (80%) |
| ML | `medallion_ml.taxi_tip_test` | Dados de teste (20%) |
| ML | `medallion_ml.best_model_ref` | Referencia ao melhor modelo (run_id + model_uri) |
| Gold | `medallion_gold.taxi_tip_predictions` | Predicoes agregadas por hora/periodo |

---

## Stack

- **Databricks Free Edition** (AWS, serverless compute)
- **PySpark / Spark SQL**
- **Delta Lake** (Unity Catalog)
- **MLflow** (Experiment Tracking, Model Artifacts)
- **scikit-learn** (GradientBoosting, RandomForest, etc.)
- **Databricks SQL** + **Genie Spaces**
- **GitHub** para versionamento

---

## Como executar

### Pre-requisitos

- Conta Databricks Free Edition
- Repositorio conectado via **Databricks Repos**
- SQL Warehouse ativo (Serverless Starter Warehouse)

### Tutorial 1 -- Medallion Architecture

1. Execute os notebooks em ordem: `01 -> 02 -> 03`
2. Use as queries de `sql/gold_queries.sql` no **SQL Editor**
3. Configure o **Genie Space** com as tabelas Gold

### Tutorial 2 -- Machine Learning

1. Execute os notebooks em ordem: `04 -> 05 -> 06 -> 07 -> 08`
2. Acompanhe os experimentos em **Experiments** (MLflow UI)
3. Adicione `taxi_tip_predictions` ao **Genie Space**

---

## Notas sobre o Free Edition

O Databricks Free Edition tem limitacoes importantes para ML:

- **Memoria**: ~2GB RAM no serverless. Datasets devem ser amostrados (50K rows)
- **UC Model Registry**: Upload de modelos bloqueado (S3 AccessDenied). Usamos MLflow artifacts diretamente (`runs:/{RUN_ID}/model`)
- **Sem clusters**: Apenas serverless compute. sklearn (single-node) em vez de SparkML
- **Model Signature**: Obrigatoria ao logar modelos. Usar `infer_signature()`

Detalhes completos no troubleshooting dos PDFs.

---

*Projeto criado por Thiago Charchar & Claude AI -- Fevereiro 2026*
