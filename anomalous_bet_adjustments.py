import os
import sys
import logging
from pathlib import Path

import redshift_connector  # Use the native Redshift driver instead of SQLAlchemy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from dotenv import load_dotenv

# --- 1. SETUP AND CONFIGURATION ---

def setup_logging():
    """Configures a logger to print messages to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# --- DATABASE CONNECTION HELPERS ---

def get_db_connection_and_table():
    """
    Loads database credentials and table name from .env and establishes a direct
    connection to Amazon Redshift using the redshift_connector library.

    Returns
    -------
    conn : redshift_connector.Connection
        An open Redshift connection that adheres to the Python DB-API 2.0 spec.
    table_name : str
        Fully-qualified table name (without schema) as provided in .env.
    """
    load_dotenv()

    # Accept both REDSHIFT_DB and REDSHIFT_DATABASE for convenience
    db_name_env = os.getenv("REDSHIFT_DB") or os.getenv("REDSHIFT_DATABASE")

    required_vars = {
        "REDSHIFT_HOST": os.getenv("REDSHIFT_HOST"),
        "REDSHIFT_PORT": os.getenv("REDSHIFT_PORT", "5439"),
        "REDSHIFT_DATABASE": db_name_env,
        "REDSHIFT_USER": os.getenv("REDSHIFT_USER"),
        "REDSHIFT_PASSWORD": os.getenv("REDSHIFT_PASSWORD"),
        "REDSHIFT_TABLE": os.getenv("REDSHIFT_TABLE"),
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        logging.error(
            "Missing the following required environment variables: %s",
            ", ".join(missing),
        )
        sys.exit(1)

    try:
        conn = redshift_connector.connect(
            host=required_vars["REDSHIFT_HOST"],
            port=int(required_vars["REDSHIFT_PORT"]),
            database=db_name_env,
            user=required_vars["REDSHIFT_USER"],
            password=required_vars["REDSHIFT_PASSWORD"],
        )
        logging.info("Database connection successful via redshift_connector.")
        return conn, required_vars["REDSHIFT_TABLE"]
    except Exception as exc:
        logging.error("Failed to connect to Redshift: %s", exc)
        sys.exit(1)

# --- 2. DATA LOADING (STREAMLINED FOR COUNT ANALYSIS) ---

def load_adjustment_data(conn, table_name: str, schema: str) -> pd.DataFrame:
    """
    Loads only the necessary data (id, created_at) for count analysis.
    This is highly efficient as it ignores all value-based columns.
    """
    logging.info("Loading adjustment data from %s.%s for count analysis.", schema, table_name)

    # Simplified query to only fetch essential columns for counting
    query = f'''
        SELECT
            id,
            created_at
        FROM {schema}."{table_name}"
    '''

    try:
        df = pd.read_sql_query(
            query,
            conn,
            parse_dates={'created_at': {'dayfirst': True, 'errors': 'coerce'}},
        )
        df.dropna(subset=['created_at'], inplace=True)

        logging.info(f"Successfully loaded {len(df):,} adjustment records.")

        if df.empty:
            logging.error("Query returned no data. Please check table name and schema.")
            sys.exit(1)
        return df
    except Exception as e:
        logging.error(
            "FATAL: Data loading failed. Check that column names in the SQL query exactly match your database schema. Error: %s",
            e,
        )
        sys.exit(1)

# --- 3. COUNT-BASED ANALYSIS MODULES ---

def run_temporal_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes adjustment counts by hour of day and day of week."""
    logging.info("--- Running Temporal Analysis (Counts by Hour/Weekday) ---")
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()

    activity_pivot = df.pivot_table(index='hour', columns='day_of_week', values='id', aggfunc='count')
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    activity_pivot = activity_pivot.reindex(columns=weekdays)

    plt.figure(figsize=(14, 8))
    sns.heatmap(activity_pivot, cmap="viridis", annot=True, fmt=".0f", linewidths=.5)
    plt.title('Adjustment Heatmap: Hour of Day vs. Day of Week', fontsize=16)
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day (24H)')
    plt.savefig(output_dir / "temporal_heatmap_counts.png")
    plt.close()
    logging.info("Saved hourly/weekday activity heatmap.")


def run_aggregate_count_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes aggregate counts over time to set platform-wide alert thresholds."""
    logging.info("--- Running Aggregate Count Analysis for Platform-Wide Alerting ---")

    # --- 1. Analysis of Absolute Counts per Time Window ---
    time_windows = {
        '5-Minute': '5min',
        '15-Minute': '15min',
        'Hourly': 'H',
        'Daily': 'D',
    }

    for name, freq in time_windows.items():
        logging.info(f"--- Analyzing {name} Adjustment Counts ---")

        counts_per_period = df.set_index('created_at').resample(freq)['id'].count()
        count_stats = counts_per_period.describe(percentiles=[.75, .90, .95, .99, .999]).round(2)

        logging.warning(f"Statistics for {name} Adjustment Counts:\n{count_stats.to_string()}")
        count_stats.to_csv(output_dir / f"aggregate_{name.lower()}_count_stats.csv")

        plt.figure(figsize=(15, 7))
        counts_per_period.plot(marker='.', linestyle='-', markersize=4)
        p95 = count_stats.loc['95%']
        p99 = count_stats.loc['99%']
        plt.axhline(y=p95, color='orange', linestyle='--', label=f'95th Percentile ({p95:.0f})')
        plt.axhline(y=p99, color='red', linestyle='--', label=f'99th Percentile ({p99:.0f})')
        plt.title(f'Total Bet Adjustments per {name}', fontsize=16)
        plt.ylabel('Number of Adjustments')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig(output_dir / f"timeseries_{name.lower()}_counts.png")
        plt.close()
        logging.info(f"Saved {name} counts time series plot.")

    # --- 2. Analysis of Rate of Change (Spike Detection) ---
    logging.info("--- Analyzing Hourly Rate of Change (Spike Detection) ---")
    hourly_counts = df.set_index('created_at').resample('H')['id'].count()
    hourly_pct_change = hourly_counts.pct_change().mul(100).replace([np.inf, -np.inf], np.nan).dropna()

    change_stats = hourly_pct_change.describe(percentiles=[.90, .95, .99, .999]).round(2)
    logging.warning(f"Statistics for Hour-over-Hour Percentage Change:\n{change_stats.to_string()}")
    change_stats.to_csv(output_dir / "aggregate_hourly_pct_change_stats.csv")

# --- MAIN ORCHESTRATOR ---

def main():
    """Main script to orchestrate the count-based EDA pipeline."""

    # --- DEFAULT CONFIGURATION ---
    # The table name is now loaded from the .env file (as REDSHIFT_TABLE)
    SCHEMA_NAME = "public"
    OUTPUT_DIR = "count_analysis_reports"
    # -----------------------------

    setup_logging()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Reports will be saved to: {output_dir.resolve()}")

    # Get the database connection and table name from the environment variables
    conn, table_name = get_db_connection_and_table()

    # Load data using the retrieved table name
    df = load_adjustment_data(conn, table_name, SCHEMA_NAME)

    # Execute all count-based analysis modules
    run_temporal_analysis(df, output_dir)
    run_aggregate_count_analysis(df, output_dir)

    logging.info("--- Count Analysis EDA COMPLETE ---")

    # Ensure we close the database connection when we're done
    try:
        conn.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()