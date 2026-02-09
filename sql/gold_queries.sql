-- ============================================
-- Gold Layer — SQL Queries para Databricks SQL
-- Use estas queries no SQL Editor ou como base
-- para dashboards e Genie Spaces
-- ============================================

-- -----------------------------------------------
-- 1. Top 10 dias com maior receita
-- -----------------------------------------------
SELECT
    pickup_date,
    total_trips,
    total_revenue,
    total_tips,
    avg_fare,
    avg_distance_miles
FROM workspace.medallion_gold.taxi_daily_metrics
ORDER BY total_revenue DESC
LIMIT 10;

-- -----------------------------------------------
-- 2. Evolução diária de corridas e receita
-- -----------------------------------------------
SELECT
    pickup_date,
    total_trips,
    total_revenue,
    total_amount_collected,
    avg_fare
FROM workspace.medallion_gold.taxi_daily_metrics
ORDER BY pickup_date;

-- -----------------------------------------------
-- 3. Top 20 zonas mais movimentadas
-- -----------------------------------------------
SELECT
    zone_id,
    total_trips,
    total_revenue,
    avg_fare,
    avg_distance_miles,
    avg_tip_pct
FROM workspace.medallion_gold.taxi_zone_metrics
ORDER BY total_trips DESC
LIMIT 20;

-- -----------------------------------------------
-- 4. Padrão de corridas por hora do dia
-- -----------------------------------------------
SELECT
    pickup_hour,
    SUM(total_trips) AS total_trips,
    ROUND(AVG(avg_fare), 2) AS avg_fare,
    ROUND(AVG(avg_speed_mph), 2) AS avg_speed
FROM workspace.medallion_gold.taxi_hourly_metrics
GROUP BY pickup_hour
ORDER BY pickup_hour;

-- -----------------------------------------------
-- 5. Comparação dias úteis vs fim de semana
-- -----------------------------------------------
SELECT
    CASE
        WHEN pickup_day_of_week IN (1, 7) THEN 'Weekend'
        ELSE 'Weekday'
    END AS period,
    SUM(total_trips) AS total_trips,
    ROUND(AVG(avg_fare), 2) AS avg_fare,
    ROUND(AVG(avg_tip_pct), 2) AS avg_tip_pct,
    ROUND(AVG(avg_speed_mph), 2) AS avg_speed
FROM workspace.medallion_gold.taxi_hourly_metrics
GROUP BY
    CASE
        WHEN pickup_day_of_week IN (1, 7) THEN 'Weekend'
        ELSE 'Weekday'
    END;

-- -----------------------------------------------
-- 6. Heatmap — Corridas por hora e dia da semana
-- -----------------------------------------------
SELECT
    day_name,
    pickup_hour,
    total_trips,
    avg_fare,
    avg_speed_mph
FROM workspace.medallion_gold.taxi_hourly_metrics
ORDER BY pickup_day_of_week, pickup_hour;
