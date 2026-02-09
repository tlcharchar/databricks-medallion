# Databricks Medallion Architecture — NYC Taxi

Pipeline de dados end-to-end usando **Medallion Architecture** (Bronze → Silver → Gold) no Databricks Free Edition, com **SQL Analytics** e **Genie Spaces** para consultas em linguagem natural.

## Arquitetura

```
NYC Taxi Dataset (/databricks-datasets)
        ↓
  ┌─────────────┐
  │   BRONZE    │  Dados brutos + metadados de ingestão
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │   SILVER    │  Dados limpos, tipados, enriquecidos
  └──────┬──────┘
         ↓
  ┌─────────────┐
  │    GOLD     │  Métricas agregadas por dia/zona/hora
  └──────┬──────┘
         ↓
  ┌──────────────────┐
  │  SQL + Genie     │  Dashboards + "Fale com seus dados"
  └──────────────────┘
```

## Estrutura

```
├── notebooks/
│   ├── 01_bronze_ingestion.py      # Ingestão raw → Delta
│   ├── 02_silver_transformation.py # Limpeza e transformações
│   └── 03_gold_aggregation.py      # Agregações de negócio
├── sql/
│   └── gold_queries.sql            # Queries analíticas
├── config/
│   └── pipeline_config.yaml        # Configuração do pipeline
└── README.md
```

## Tabelas

| Camada | Tabela | Descrição |
|--------|--------|-----------|
| Bronze | `medallion_bronze.taxi_trips_raw` | Dados brutos do NYC Taxi |
| Silver | `medallion_silver.taxi_trips_cleaned` | Dados limpos e enriquecidos |
| Gold | `medallion_gold.taxi_daily_metrics` | Métricas diárias |
| Gold | `medallion_gold.taxi_zone_metrics` | Métricas por zona |
| Gold | `medallion_gold.taxi_hourly_metrics` | Métricas por hora/dia |

## Stack

- **Databricks Free Edition** (AWS)
- **PySpark / Spark SQL**
- **Delta Lake** (Unity Catalog)
- **Databricks SQL** + **Genie Spaces**
- **GitHub** para versionamento

## Como executar

1. Conecte este repo ao Databricks via **Repos**
2. Execute os notebooks em ordem: `01 → 02 → 03`
3. Use as queries SQL no **SQL Editor**
4. Configure o **Genie Space** com as tabelas Gold
