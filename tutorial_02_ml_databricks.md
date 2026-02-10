# Tutorial 2: Machine Learning no Databricks Free Edition

## De Feature Engineering a Batch Inference com MLflow

---

**Autor:** Thiago Charchar & Claude AI
**Data:** Fevereiro 2026
**Repositorio:** github.com/tlcharchar/databricks-medallion
**Nivel:** Intermediario
**Pre-requisito:** Tutorial 1 (Medallion Architecture) completo

---

## Sumario

1. Visao Geral
2. Arquitetura ML
3. Capacidades ML no Free Edition
4. Fase 1 -- Feature Engineering
5. Fase 2 -- AutoML Baseline
6. Fase 3 -- Hyperparameter Tuning com MLflow
7. Fase 4 -- Model Registry (Unity Catalog)
8. Fase 5 -- Batch Inference
9. Fase 6 -- Genie Space com predicoes ML
10. Troubleshooting
11. Proximos Passos

---

## 1. Visao Geral

Este tutorial e a continuacao do Tutorial 1 (Medallion Architecture). Usamos a tabela **Silver** (`taxi_trips_cleaned`) como base para construir um pipeline de Machine Learning completo.

### Caso de Uso

**Prever o valor da gorjeta (tip_amount)** de corridas de taxi em NYC.

Pergunta de negocio: "Dado uma corrida de taxi (distancia, hora, dia, zona), qual sera a gorjeta?"

### Por que este caso de uso?

- E um problema de **regressao** (prever valor continuo)
- Usa dados que ja existem na Silver (sem nova ingestao)
- Tem features ricas (temporais, geograficas, de viagem)
- Resultado e intuitivo e verificavel

---

## 2. Arquitetura ML

```
Silver (taxi_trips_cleaned)
   |
   v
Feature Engineering (04)
   |  --> medallion_ml.taxi_tip_features
   |  --> medallion_ml.taxi_tip_train
   |  --> medallion_ml.taxi_tip_test
   v
AutoML Baseline (05)
   |  --> MLflow: 4 modelos (LR, DT, RF, GB)
   v
Hyperparameter Tuning (06)
   |  --> MLflow: nested runs, grid search
   |  --> medallion_ml.best_model_ref
   v
Model Registry (07)
   |  --> UC: nyc_taxi_tip_model @Champion
   v
Batch Inference (08)
   |  --> medallion_gold.taxi_tip_predictions
   v
Genie Space
   --> "Qual a gorjeta prevista para corridas noturnas?"
```

---

## 3. Capacidades ML no Free Edition

### O que esta disponivel

| Recurso | Status | Como usamos |
|---|---|---|
| MLflow Experiments | Disponivel | Tracking de parametros e metricas |
| MLflow Model Logging | Disponivel | Salvar modelos como artefatos |
| Unity Catalog Models | Disponivel | Model Registry (nao legacy) |
| Serverless Compute | Disponivel | Executar notebooks de ML |
| Genie Spaces | Disponivel | NLP sobre predicoes |

### O que NAO esta disponivel

| Recurso | Status | Alternativa |
|---|---|---|
| Databricks AutoML | Nao disponivel | sklearn + MLflow manual |
| ML Runtime clusters | Nao disponivel | Serverless com pip install |
| Feature Store API | Legacy desabilitada | Delta Tables no Unity Catalog |
| Model Serving endpoints | Nao disponivel | Batch inference |

---

## 4. Fase 1 -- Feature Engineering

### Notebook: `04_feature_engineering.py`

Transforma dados da Silver em features prontas para ML.

### Categorias de Features

**Temporais** -- quando a corrida aconteceu:

```python
# Periodo do dia
.withColumn(
    "period_of_day",
    when((col("pickup_hour") >= 0) & (col("pickup_hour") < 6), 0)
    .when((col("pickup_hour") >= 6) & (col("pickup_hour") < 12), 1)
    .when((col("pickup_hour") >= 12) & (col("pickup_hour") < 18), 2)
    .otherwise(3)
)
# Fim de semana
.withColumn(
    "is_weekend",
    when(col("pickup_day_of_week").isin(1, 7), 1).otherwise(0)
)
# Horario de pico
.withColumn(
    "is_rush_hour",
    when(
        (col("is_weekend") == 0) &
        (col("pickup_hour").between(7, 9) |
         col("pickup_hour").between(17, 19)),
        1
    ).otherwise(0)
)
```

**Geograficas** -- de onde para onde (zonas arredondadas):

```python
.withColumn("pickup_zone_lat",
            spark_round(col("pickup_latitude"), 2))
.withColumn("pickup_zone_lon",
            spark_round(col("pickup_longitude"), 2))
```

**De viagem** -- caracteristicas da corrida:

```python
# Log-transform para normalizar distribuicoes
.withColumn("log_trip_distance", log1p(col("trip_distance")))
.withColumn("log_trip_duration", log1p(col("trip_duration_min")))
# Proxy de trafego
.withColumn("distance_duration_ratio",
            col("trip_distance") / col("trip_duration_min"))
```

### Filtro importante: apenas cartao de credito

Gorjetas em dinheiro **nao sao registradas** no dataset. Filtramos apenas corridas pagas com cartao:

```python
.filter(col("payment_is_credit") == 1)
```

### Train/Test Split

```python
df_train, df_test = df_features.randomSplit([0.8, 0.2], seed=42)
```

### Tabelas criadas

| Tabela | Descricao |
|---|---|
| medallion_ml.taxi_tip_features | Feature table completa |
| medallion_ml.taxi_tip_train | 80% para treinamento |
| medallion_ml.taxi_tip_test | 20% para avaliacao |

---

## 5. Fase 2 -- AutoML Baseline

### Notebook: `05_automl_baseline.py`

Testa 4 algoritmos automaticamente e loga tudo no MLflow.

### Modelos testados

| Modelo | Tipo | Complexidade |
|---|---|---|
| LinearRegression | Linear | Baixa (baseline) |
| DecisionTree | Arvore | Media |
| RandomForest | Ensemble | Alta |
| GradientBoosting | Ensemble | Alta |

### Configuracao do MLflow

```python
import mlflow

EXPERIMENT_NAME = "/Users/seu@email.com/nyc-taxi-tip-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)
```

### Loop de treinamento

```python
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"automl_{model_name}"):
        # Logar parametros
        mlflow.log_param("model_type", model_name)

        # Treinar
        model.fit(X_train, y_train)

        # Avaliar
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        # Logar metricas
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Logar modelo
        mlflow.sklearn.log_model(model, "model")
```

### Metricas de avaliacao

| Metrica | Significado |
|---|---|
| MAE | Erro medio absoluto (em dolares) |
| RMSE | Raiz do erro quadratico medio |
| R2 | Variancia explicada (0 a 1) |

### Nota sobre amostragem

Como o Free Edition usa serverless (sem cluster ML dedicado), precisamos amostrar o dataset para caber em memoria do pandas:

```python
MAX_TRAIN_ROWS = 500_000
if train_count > MAX_TRAIN_ROWS:
    sample_fraction = MAX_TRAIN_ROWS / train_count
    df_train_pd = df_train_spark.sample(
        fraction=sample_fraction, seed=42
    ).toPandas()
```

---

## 6. Fase 3 -- Hyperparameter Tuning com MLflow

### Notebook: `06_mlflow_training.py`

Otimiza o melhor modelo (GradientBoosting) com grid search.

### Grid de hiperparametros

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
}
# Total: 12 combinacoes
```

### Nested Runs no MLflow

Usamos um **parent run** que organiza todos os **child runs**:

```python
with mlflow.start_run(run_name="hyperparameter_tuning_gb") as parent:
    for params in grid:
        with mlflow.start_run(nested=True) as child:
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(model, "model")
```

### Visualizacao no MLflow UI

No menu lateral do Databricks, va em **Experiments** e abra `nyc-taxi-tip-prediction`. Voce vera:

- Parent run com 12 child runs
- Graficos comparativos de metricas
- Tabela com todos os hiperparametros

---

## 7. Fase 4 -- Model Registry (Unity Catalog)

### Notebook: `07_model_registry.py`

Registra o melhor modelo no Unity Catalog.

### Configuracao importante

```python
# Usar Unity Catalog (NAO o legacy workspace registry)
mlflow.set_registry_uri("databricks-uc")
```

### Registrar modelo

```python
MODEL_NAME = "workspace.medallion_ml.nyc_taxi_tip_model"

result = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=MODEL_NAME
)
```

### Aliases (substituem Stages)

No Unity Catalog, usamos **aliases** em vez de stages (Staging/Production):

```python
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="Champion",
    version=result.version
)
```

| Alias | Significado |
|---|---|
| Champion | Modelo em producao |
| Challenger | Candidato a substituir o Champion |

### Carregar modelo do Registry

```python
# Carregar pelo alias
model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@Champion"
)
```

---

## 8. Fase 5 -- Batch Inference

### Notebook: `08_batch_inference.py`

Aplica o modelo Champion em batch para gerar predicoes.

### Fluxo

```
Carregar modelo Champion
    |
    v
Carregar dados de teste
    |
    v
Gerar predicoes
    |
    v
Calcular erros
    |
    v
Agregar em tabela Gold
    |
    v
taxi_tip_predictions (Gold)
```

### Analise de erro

O notebook calcula:
- **MAE** geral do modelo
- **% de predicoes dentro de $1 e $2** de erro
- **Erro por hora do dia** (quando o modelo acerta mais?)
- **Erro por periodo** (madrugada, manha, tarde, noite)

### Tabela Gold ML

```python
GOLD_ML_TABLE = "workspace.medallion_gold.taxi_tip_predictions"
```

Colunas principais:

| Coluna | Descricao |
|---|---|
| avg_actual_tip | Gorjeta real media |
| avg_predicted_tip | Gorjeta prevista pelo modelo |
| avg_prediction_error | Erro medio do modelo |
| pct_accurate_within_1usd | % de predicoes com erro menor que $1 |
| period_name | Periodo do dia (Madrugada, Manha, Tarde, Noite) |
| day_name | Nome do dia da semana |

---

## 9. Fase 6 -- Genie Space com predicoes ML

### Atualizar o Genie Space

No Genie Space criado no Tutorial 1, adicione a nova tabela:

1. Va em **Genie** no menu lateral
2. Edite o Space **NYC Taxi Analytics**
3. Adicione a tabela: `workspace.medallion_gold.taxi_tip_predictions`
4. Atualize as **General instructions** adicionando:

```
A tabela taxi_tip_predictions contem predicoes de ML
sobre gorjetas feitas por um modelo GradientBoosting.

Colunas de ML:
- avg_actual_tip = gorjeta real media
- avg_predicted_tip = gorjeta prevista pelo modelo
- avg_prediction_error = erro medio da previsao
- pct_accurate_within_1usd = % de previsoes precisas
- period_name = Madrugada/Manha/Tarde/Noite

O modelo foi treinado com features temporais, geograficas
e de viagem. Use estas colunas para responder perguntas
sobre predicoes e acuracia do modelo.
```

### Perguntas para testar

- "Qual a gorjeta prevista para corridas noturnas?"
- "Em que horario o modelo erra mais?"
- "Qual o percentual de acerto do modelo nos fins de semana?"
- "Compare a gorjeta real vs prevista por periodo do dia"
- "Quando as gorjetas sao maiores: manha ou noite?"

---

## 10. Troubleshooting

### Erro: MemoryError ao converter para Pandas

**Causa:** Dataset muito grande para caber em memoria do serverless.
**Solucao:** Amostrar antes de converter:

```python
MAX_ROWS = 500_000
sample_frac = min(1.0, MAX_ROWS / total_count)
df_pd = df_spark.sample(fraction=sample_frac, seed=42).toPandas()
```

### Erro: legacy workspace model registry is disabled

**Causa:** Free Edition usa Unity Catalog, nao o registry antigo.
**Solucao:** Configurar MLflow para usar UC:

```python
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = "catalog.schema.model_name"  # formato 3 niveis
```

### Erro: Only serverless compute is supported

**Causa:** Free Edition nao permite clusters customizados.
**Solucao:** Usar sklearn (single-node) em vez de SparkML (distribuido). O serverless suporta bem com datasets amostrados.

### Erro: ModuleNotFoundError (sklearn, xgboost, etc.)

**Causa:** Biblioteca nao disponivel no runtime serverless.
**Solucao:** Instalar no notebook:

```python
%pip install scikit-learn xgboost
```

---

## 11. Proximos Passos

Apos completar este tutorial, considere:

1. **SparkML** -- Substituir sklearn por SparkML para treinar de forma distribuida
2. **Feature Store** -- Quando disponivel, migrar features para o Feature Store nativo
3. **Model Serving** -- Em edicoes pagas, criar endpoints REST para inferencia online
4. **A/B Testing** -- Usar aliases Champion/Challenger para A/B testing de modelos
5. **Retraining** -- Automatizar retraining periodico com Workflows
6. **Monitoring** -- Implementar data drift e model drift detection
7. **Deep Learning** -- Explorar PyTorch/TensorFlow para modelos mais complexos
8. **LLMs** -- Usar os modelos do Playground/Mosaic AI para tarefas de NLP

---

## Resumo: Estrutura completa do Projeto

```
databricks-medallion/
|-- notebooks/
|   |-- 01_bronze_ingestion.py         # Tutorial 1
|   |-- 02_silver_transformation.py    # Tutorial 1
|   |-- 03_gold_aggregation.py         # Tutorial 1
|   |-- 04_feature_engineering.py      # Tutorial 2
|   |-- 05_automl_baseline.py          # Tutorial 2
|   |-- 06_mlflow_training.py          # Tutorial 2
|   |-- 07_model_registry.py           # Tutorial 2
|   |-- 08_batch_inference.py          # Tutorial 2
|-- sql/
|   |-- gold_queries.sql
|-- config/
|   |-- pipeline_config.yaml
|-- README.md
```

### Tabelas criadas (Tutorial 2)

| Camada | Tabela | Descricao |
|---|---|---|
| ML | medallion_ml.taxi_tip_features | Feature table completa |
| ML | medallion_ml.taxi_tip_train | Dados de treinamento (80%) |
| ML | medallion_ml.taxi_tip_test | Dados de teste (20%) |
| ML | medallion_ml.best_model_ref | Referencia ao melhor modelo |
| Gold | medallion_gold.taxi_tip_predictions | Predicoes agregadas |
| UC | nyc_taxi_tip_model @Champion | Modelo registrado |

---

*Tutorial criado como parte do projeto databricks-medallion.*
*Repositorio: github.com/tlcharchar/databricks-medallion*
