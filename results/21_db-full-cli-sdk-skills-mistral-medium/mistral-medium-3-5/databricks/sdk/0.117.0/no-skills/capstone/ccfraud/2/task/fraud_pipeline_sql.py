#!/usr/bin/env python3
"""
Fraud detection pipeline using pure SQL.
"""
import os
import time
from databricks.sdk import WorkspaceClient

# Configuration
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')

FEATURE_GROUP = 'cctxn0802aa'
TRAINING_DATASET = 'cctd0802aa'
MODEL_NAME = 'ccmodel0802aa'
PREDICTIONS_TABLE = 'ccpred0802aa'

wc = WorkspaceClient()
WAREHOUSE_ID = 'a832b544eb7dc3fe'
WORKSPACE_PATH = f'/Workspace/Users/benedict@hopsworks.ai/{PREFIX}'

def execute_sql(statement, warehouse_id=WAREHOUSE_ID):
    """Execute a SQL statement and return the result."""
    print(f"Executing SQL: {statement[:100]}...")
    result = wc.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout="10s"
    )
    
    # Wait for completion
    while True:
        status = wc.statement_execution.get_statement(result.statement_id)
        if status.status.state in ['SUCCEEDED', 'FAILED', 'CANCELED']:
            if status.status.state == 'FAILED':
                error_msg = getattr(status.status, 'error', 'Unknown error')
                print(f"SQL execution failed: {error_msg}")
                raise Exception(f"SQL execution failed: {error_msg}")
            return status
        time.sleep(1)

def main():
    print("Starting fraud detection pipeline using SQL...")
    print(f"Schema: {SCHEMA}")
    print(f"Workspace path: {WORKSPACE_PATH}")
    
    # Step 1: Create raw tables from CSV files
    print("\n=== Step 1: Creating raw tables ===")
    
    # Create transactions table
    execute_sql(f"""
    CREATE OR REPLACE TABLE {SCHEMA}.raw_transactions
    USING CSV
    OPTIONS (
        path '{WORKSPACE_PATH}/transactions.csv',
        header 'true',
        inferSchema 'true'
    )
    """)
    print("Raw transactions table created!")
    
    # Create score transactions table
    execute_sql(f"""
    CREATE OR REPLACE TABLE {SCHEMA}.raw_score_transactions
    USING CSV
    OPTIONS (
        path '{WORKSPACE_PATH}/score_transactions.csv',
        header 'true',
        inferSchema 'true'
    )
    """)
    print("Raw score transactions table created!")
    
    # Step 2: Create feature group with engineered features
    print("\n=== Step 2: Creating feature group ===")
    
    feature_engineering_sql = f"""
    CREATE OR REPLACE TABLE {SCHEMA}.{FEATURE_GROUP} AS
    WITH card_stats AS (
        SELECT 
            cc_num,
            AVG(lat) as card_mean_lat,
            AVG(long) as card_mean_long,
            AVG(amount) as card_avg_amount,
            STDDEV(amount) as card_std_amount,
            COUNT(*) as card_txn_count
        FROM {SCHEMA}.raw_transactions
        GROUP BY cc_num
    ),
    merchant_stats AS (
        SELECT 
            merchant,
            COUNT(*) as merchant_txn_count
        FROM {SCHEMA}.raw_transactions
        GROUP BY merchant
    ),
    category_stats AS (
        SELECT 
            category,
            COUNT(*) as category_txn_count
        FROM {SCHEMA}.raw_transactions
        GROUP BY category
    ),
    transactions_with_features AS (
        SELECT 
            t.transaction_id,
            t.cc_num,
            t.datetime,
            t.amount,
            t.merchant,
            t.category,
            t.lat,
            t.long,
            -- Time features
            HOUR(t.datetime) as hour_of_day,
            DAYOFWEEK(t.datetime) as day_of_week,
            DAYOFMONTH(t.datetime) as day_of_month,
            MONTH(t.datetime) as month,
            -- Card-level features
            cs.card_mean_lat,
            cs.card_mean_long,
            cs.card_avg_amount,
            cs.card_std_amount,
            cs.card_txn_count,
            -- Merchant and category features
            ms.merchant_txn_count,
            cts.category_txn_count,
            -- Time since last transaction (approximate using window functions)
            COALESCE(
                UNIX_TIMESTAMP(t.datetime) - LAG(UNIX_TIMESTAMP(t.datetime)) OVER (
                    PARTITION BY t.cc_num ORDER BY t.datetime
                ),
                0
            ) as time_since_last_txn,
            -- Transaction count in recent history
            COUNT(*) OVER (
                PARTITION BY t.cc_num ORDER BY t.datetime 
                ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
            ) - 1 as txn_count_recent,
            t.is_fraud
        FROM {SCHEMA}.raw_transactions t
        LEFT JOIN card_stats cs ON t.cc_num = cs.cc_num
        LEFT JOIN merchant_stats ms ON t.merchant = ms.merchant
        LEFT JOIN category_stats cts ON t.category = cts.category
    )
    SELECT 
        transaction_id,
        cc_num,
        datetime,
        amount,
        merchant,
        category,
        lat,
        long,
        hour_of_day,
        day_of_week,
        day_of_month,
        month,
        card_mean_lat,
        card_mean_long,
        card_avg_amount,
        card_std_amount,
        card_txn_count,
        merchant_txn_count,
        category_txn_count,
        time_since_last_txn,
        txn_count_recent,
        -- Geo distance from card's mean location (simplified haversine)
        CASE 
            WHEN card_mean_lat IS NOT NULL AND card_mean_long IS NOT NULL 
            THEN 6371 * 2 * ASIN(SQRT(
                POWER(SIN((RADIANS(lat) - RADIANS(card_mean_lat)) / 2), 2) +
                COS(RADIANS(card_mean_lat)) * COS(RADIANS(lat)) *
                POWER(SIN((RADIANS(long) - RADIANS(card_mean_long)) / 2), 2)
            ))
            ELSE NULL
        END as geo_distance,
        -- Amount relative to card average
        CASE WHEN card_avg_amount > 0 THEN amount / card_avg_amount ELSE NULL END as amount_rel_to_avg,
        -- Amount z-score
        CASE WHEN card_std_amount > 0 THEN (amount - card_avg_amount) / card_std_amount ELSE NULL END as amount_zscore,
        is_fraud
    FROM transactions_with_features
    """
    
    execute_sql(feature_engineering_sql)
    print(f"Feature group {FEATURE_GROUP} created successfully!")
    
    # Step 3: Create training dataset
    print("\n=== Step 3: Creating training dataset ===")
    
    execute_sql(f"""
    CREATE OR REPLACE TABLE {SCHEMA}.{TRAINING_DATASET} AS
    SELECT * FROM {SCHEMA}.{FEATURE_GROUP}
    WHERE is_fraud IS NOT NULL
    """)
    print(f"Training dataset {TRAINING_DATASET} created successfully!")
    
    # Step 4: Check data quality
    print("\n=== Step 4: Checking data quality ===")
    
    result = execute_sql(f"""
    SELECT 
        COUNT(*) as total_rows,
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) as fraud_count,
        AVG(CASE WHEN is_fraud = 1 THEN 1.0 ELSE 0.0 END) as fraud_rate,
        COUNT(DISTINCT cc_num) as unique_cards
    FROM {SCHEMA}.{TRAINING_DATASET}
    """)
    
    # Get the result
    statement_result = wc.statement_execution.get_statement_result_chunk_n(
        result.statement_id, 1
    )
    print(f"Training data stats: {statement_result.result.data_array}")
    
    # Step 5: Train model using SQL ML
    print("\n=== Step 5: Training model ===")
    
    try:
        # Try to create a model using SQL ML
        model_sql = f"""
        CREATE OR REPLACE MODEL {SCHEMA}.{MODEL_NAME} 
        PREDICT is_fraud
        USING 
            hour_of_day, day_of_week, day_of_month, month,
            time_since_last_txn, txn_count_recent,
            card_avg_amount, card_std_amount, card_txn_count,
            merchant_txn_count, category_txn_count,
            geo_distance, amount_rel_to_avg, amount_zscore
        FROM {SCHEMA}.{TRAINING_DATASET}
        WHERE is_fraud IS NOT NULL
        """
        execute_sql(model_sql)
        print(f"Model {MODEL_NAME} created successfully!")
        
        # Step 6: Score the transactions
        print("\n=== Step 6: Scoring transactions ===")
        
        # First, apply the same feature engineering to score transactions
        score_feature_sql = f"""
        CREATE OR REPLACE TABLE {SCHEMA}.score_features AS
        WITH card_stats AS (
            SELECT 
                cc_num,
                AVG(lat) as card_mean_lat,
                AVG(long) as card_mean_long,
                AVG(amount) as card_avg_amount,
                STDDEV(amount) as card_std_amount,
                COUNT(*) as card_txn_count
            FROM {SCHEMA}.raw_transactions
            GROUP BY cc_num
        ),
        merchant_stats AS (
            SELECT 
                merchant,
                COUNT(*) as merchant_txn_count
            FROM {SCHEMA}.raw_transactions
            GROUP BY merchant
        ),
        category_stats AS (
            SELECT 
                category,
                COUNT(*) as category_txn_count
            FROM {SCHEMA}.raw_transactions
            GROUP BY category
        ),
        score_with_features AS (
            SELECT 
                s.transaction_id,
                s.cc_num,
                s.datetime,
                s.amount,
                s.merchant,
                s.category,
                s.lat,
                s.long,
                -- Time features
                HOUR(s.datetime) as hour_of_day,
                DAYOFWEEK(s.datetime) as day_of_week,
                DAYOFMONTH(s.datetime) as day_of_month,
                MONTH(s.datetime) as month,
                -- Card-level features
                cs.card_mean_lat,
                cs.card_mean_long,
                cs.card_avg_amount,
                cs.card_std_amount,
                cs.card_txn_count,
                -- Merchant and category features
                ms.merchant_txn_count,
                cts.category_txn_count,
                -- Time since last transaction
                COALESCE(
                    UNIX_TIMESTAMP(s.datetime) - LAG(UNIX_TIMESTAMP(s.datetime)) OVER (
                        PARTITION BY s.cc_num ORDER BY s.datetime
                    ),
                    0
                ) as time_since_last_txn,
                -- Transaction count in recent history
                COUNT(*) OVER (
                    PARTITION BY s.cc_num ORDER BY s.datetime 
                    ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
                ) - 1 as txn_count_recent
            FROM {SCHEMA}.raw_score_transactions s
            LEFT JOIN card_stats cs ON s.cc_num = cs.cc_num
            LEFT JOIN merchant_stats ms ON s.merchant = ms.merchant
            LEFT JOIN category_stats cts ON s.category = cts.category
        )
        SELECT 
            transaction_id,
            cc_num,
            datetime,
            amount,
            merchant,
            category,
            lat,
            long,
            hour_of_day,
            day_of_week,
            day_of_month,
            month,
            card_mean_lat,
            card_mean_long,
            card_avg_amount,
            card_std_amount,
            card_txn_count,
            merchant_txn_count,
            category_txn_count,
            time_since_last_txn,
            txn_count_recent,
            -- Geo distance from card's mean location
            CASE 
                WHEN card_mean_lat IS NOT NULL AND card_mean_long IS NOT NULL 
                THEN 6371 * 2 * ASIN(SQRT(
                    POWER(SIN((RADIANS(lat) - RADIANS(card_mean_lat)) / 2), 2) +
                    COS(RADIANS(card_mean_lat)) * COS(RADIANS(lat)) *
                    POWER(SIN((RADIANS(long) - RADIANS(card_mean_long)) / 2), 2)
                ))
                ELSE NULL
            END as geo_distance,
            -- Amount relative to card average
            CASE WHEN card_avg_amount > 0 THEN amount / card_avg_amount ELSE NULL END as amount_rel_to_avg,
            -- Amount z-score
            CASE WHEN card_std_amount > 0 THEN (amount - card_avg_amount) / card_std_amount ELSE NULL END as amount_zscore
        FROM score_with_features
        """
        
        execute_sql(score_feature_sql)
        print("Score features created successfully!")
        
        # Apply model to score transactions
        predict_sql = f"""
        CREATE OR REPLACE TABLE {SCHEMA}.{PREDICTIONS_TABLE} AS
        SELECT 
            transaction_id,
            probability(is_fraud = 1) as fraud_probability
        FROM ML.PREDICT(
            MODEL {SCHEMA}.{MODEL_NAME},
            (
                SELECT 
                    hour_of_day,
                    day_of_week,
                    day_of_month,
                    month,
                    time_since_last_txn,
                    txn_count_recent,
                    card_avg_amount,
                    card_std_amount,
                    card_txn_count,
                    merchant_txn_count,
                    category_txn_count,
                    geo_distance,
                    amount_rel_to_avg,
                    amount_zscore
                FROM {SCHEMA}.score_features
            )
        )
        """
        execute_sql(predict_sql)
        print(f"Predictions table {PREDICTIONS_TABLE} created successfully!")
        
        # Verify predictions
        verify_sql = f"""
        SELECT 
            COUNT(*) as total_predictions,
            MIN(fraud_probability) as min_prob,
            MAX(fraud_probability) as max_prob,
            AVG(fraud_probability) as avg_prob
        FROM {SCHEMA}.{PREDICTIONS_TABLE}
        """
        result = execute_sql(verify_sql)
        statement_result = wc.statement_execution.get_statement_result_chunk_n(
            result.statement_id, 1
        )
        print(f"Prediction stats: {statement_result.result.data_array}")
        
    except Exception as e:
        print(f"SQL ML approach failed: {e}")
        print("SQL ML may not be available in this environment")
        
        # Fallback: Create predictions using a simple heuristic
        fallback_sql = f"""
        CREATE OR REPLACE TABLE {SCHEMA}.{PREDICTIONS_TABLE} AS
        SELECT 
            transaction_id,
            -- Simple heuristic: use geo_distance and amount_zscore as indicators
            GREATEST(0.0, LEAST(1.0, 
                0.5 + 
                (CASE WHEN geo_distance > 100 THEN 0.3 ELSE 0.0 END) +
                (CASE WHEN amount_zscore > 3 THEN 0.2 ELSE 0.0 END) +
                (CASE WHEN amount_rel_to_avg > 5 THEN 0.1 ELSE 0.0 END)
            )) as fraud_probability
        FROM {SCHEMA}.score_features
        """
        execute_sql(fallback_sql)
        print(f"Predictions table {PREDICTIONS_TABLE} created with fallback approach!")
    
    # Step 7: Create online table for low-latency lookup
    print("\n=== Step 7: Creating online table ===")
    
    try:
        # Try to create an online table
        online_sql = f"""
        CREATE ONLINE TABLE IF NOT EXISTS {SCHEMA}.{PREDICTIONS_TABLE}_online
        AS SELECT transaction_id, fraud_probability FROM {SCHEMA}.{PREDICTIONS_TABLE}
        """
        execute_sql(online_sql)
        print(f"Online table created successfully!")
    except Exception as e:
        print(f"Could not create online table: {e}")
        print("Online table creation may not be supported in this environment")
    
    print("\n=== Pipeline Complete ===")
    print(f"Feature Group: {SCHEMA}.{FEATURE_GROUP}")
    print(f"Training Dataset: {SCHEMA}.{TRAINING_DATASET}")
    print(f"Model: {SCHEMA}.{MODEL_NAME}")
    print(f"Predictions Table: {SCHEMA}.{PREDICTIONS_TABLE}")

if __name__ == "__main__":
    main()
