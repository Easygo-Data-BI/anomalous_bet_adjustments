import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
from sqlalchemy import create_engine, text
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

def get_db_config():
    """
    Loads database credentials, table, and schema from .env and creates a SQLAlchemy engine.
    Returns the engine, table name, and schema name.
    """
    load_dotenv()
    try:
        host = os.environ["REDSHIFT_HOST"].split('#')[0].strip()
        port = os.environ.get("REDSHIFT_PORT", "5439").split('#')[0].strip()
        db = os.environ["REDSHIFT_DB"].split('#')[0].strip()
        user = os.environ["REDSHIFT_USER"].split('#')[0].strip()
        password = os.environ["REDSHIFT_PASSWORD"].split('#')[0].strip()
        table_name = os.environ["REDSHIFT_TABLE"].split('#')[0].strip()
        # Load schema from .env, default to 'public' if not found
        schema_name = os.environ.get("REDSHIFT_SCHEMA", "public").split('#')[0].strip()
    except KeyError as e:
        logging.error(f"Missing required environment variable in .env file: {e}")
        sys.exit(1)

    conn_str = f"redshift+psycopg2://{user}:{password}@{host}:{port}/{db}"
    logging.info("Creating SQLAlchemy engine for Redshift...")
    try:
        engine = create_engine(
            conn_str,
            connect_args={'connect_timeout': 10},
            pool_pre_ping=False
        )
        with engine.connect() as connection:
            logging.info("Database connection successful.")
        return engine, table_name, schema_name
    except Exception as e:
        logging.error(f"Failed to create database engine: {e}")
        sys.exit(1)

# --- 2. DATA LOADING (FILTERED FOR COUNT ANALYSIS) ---

def load_adjustment_data(engine, table_name: str, schema: str) -> pd.DataFrame:
    """
    Loads necessary data for count analysis, excluding the high-volume Jade Rabbit game.
    """
    game_to_exclude = '617bd322-c947-43f2-b1cd-b44c1a0a8611'
    logging.info(f"Loading adjustment data from {schema}.{table_name}, excluding game_id {game_to_exclude}.")

    # Modified query to filter out the specific game on the database side
    query = text(f'''
        SELECT
            "id",
            "created_at" 
        FROM {schema}."{table_name}"
        WHERE "game_id" != :game_to_exclude
    ''')

    try:
        df = pd.read_sql_query(
            query,
            engine,
            params={'game_to_exclude': game_to_exclude},
            parse_dates={'created_at': {'dayfirst': True, 'errors': 'coerce'}}
        )
        df.dropna(subset=['created_at'], inplace=True)
        
        logging.info(f"Successfully loaded {len(df):,} adjustment records after filtering.")

        if df.empty:
            logging.error("Query returned no data after filtering.")
            sys.exit(1)
        return df
    except Exception as e:
        logging.error(f"FATAL: Data loading failed. Check schema and table names. Error: {e}")
        sys.exit(1)

# --- 3. COUNT-BASED ANALYSIS MODULES ---

def run_temporal_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes adjustment counts by hour of day and day of week."""
    logging.info("--- Running Temporal Analysis (Filtered Counts by Hour/Weekday) ---")
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()
    
    activity_pivot = df.pivot_table(index='hour', columns='day_of_week', values='id', aggfunc='count')
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    activity_pivot = activity_pivot.reindex(columns=weekdays).fillna(0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(activity_pivot, cmap="viridis", annot=True, fmt=".0f", linewidths=.5)
    plt.title('Adjustment Heatmap (Excluding Jade Rabbit Game)', fontsize=16)
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day (24H)')
    plt.savefig(output_dir / "temporal_heatmap_counts_filtered.png")
    plt.close()
    logging.info("Saved filtered hourly/weekday activity heatmap.")


def run_aggregate_count_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyzes aggregate counts over time to set platform-wide alert thresholds."""
    logging.info("--- Running Aggregate Count Analysis (Filtered Data) ---")

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
        
        logging.warning(f"Statistics for {name} Adjustment Counts (Filtered):\n{count_stats.to_string()}")
        count_stats.to_csv(output_dir / f"aggregate_{name.lower()}_count_stats_filtered.csv")

        plt.figure(figsize=(15, 7))
        counts_per_period.plot(marker='.', linestyle='-', markersize=4)
        p95 = count_stats.loc['95%']
        p99 = count_stats.loc['99%']
        plt.axhline(y=p95, color='orange', linestyle='--', label=f'95th Percentile ({p95:.0f})')
        plt.axhline(y=p99, color='red', linestyle='--', label=f'99th Percentile ({p99:.0f})')
        plt.title(f'Total Bet Adjustments per {name} (Excluding Jade Rabbit Game)', fontsize=16)
        plt.ylabel('Number of Adjustments')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig(output_dir / f"timeseries_{name.lower()}_counts_filtered.png")
        plt.close()
        logging.info(f"Saved {name} counts time series plot.")

    # --- 2. Analysis of Rate of Change (Spike Detection) ---
    logging.info("--- Analyzing Hourly Rate of Change (Filtered Data) ---")
    hourly_counts = df.set_index('created_at').resample('H')['id'].count()
    hourly_pct_change = hourly_counts.pct_change().mul(100).replace([np.inf, -np.inf], np.nan).dropna()

    change_stats = hourly_pct_change.describe(percentiles=[.90, .95, .99, .999]).round(2)
    logging.warning(f"Statistics for Hour-over-Hour Percentage Change (Filtered):\n{change_stats.to_string()}")
    change_stats.to_csv(output_dir / "aggregate_hourly_pct_change_stats_filtered.csv")

def run_comparative_pattern_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Creates comparative plots of hourly and daily adjustment counts.
    """
    logging.info("--- Analyzing and Comparing Daily and Hourly Patterns (Filtered) ---")
    
    # Ensure 'hour' and 'day_of_week' columns exist
    if 'hour' not in df.columns:
        df['hour'] = df['created_at'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['created_at'].dt.day_name()

    # --- 1. Create Pivot Table for Comparative Line Chart ---
    daily_hourly_pivot = df.pivot_table(
        index='hour',
        columns='day_of_week',
        values='id',
        aggfunc='count'
    ).fillna(0)

    # Ensure the columns (days) are in the correct order
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_hourly_pivot = daily_hourly_pivot.reindex(columns=weekdays)

    # --- 2. Generate the Comparative Hourly Line Chart ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 9))
    
    daily_hourly_pivot.plot(kind='line', marker='o', ax=ax, markersize=5)
        
    ax.set_title('Hourly Bet Adjustment Volume by Day (Excluding High-Volume Game)', fontsize=18, weight='bold')
    ax.set_xlabel('Hour of Day (UTC)', fontsize=12)
    ax.set_ylabel('Total Number of Adjustments', fontsize=12)
    ax.set_xticks(range(0, 24))
    ax.legend(title='Day of Week')
    ax.grid(True, which='both', linestyle='--')
    
    plt.savefig(output_dir / "comparative_hourly_patterns_filtered.png", bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved comparative hourly patterns plot (filtered).")

    # --- 3. Generate Daily Volume Over Time Chart ---
    logging.info("--- Analyzing Daily Volume Over Time (Filtered) ---")
    
    daily_counts = df.set_index('created_at').resample('D')['id'].count()
    daily_counts_df = daily_counts.reset_index()
    daily_counts_df['day_of_week'] = daily_counts_df['created_at'].dt.day_name()
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    sns.lineplot(data=daily_counts_df, x='created_at', y='id', hue='day_of_week', 
                 palette='viridis', marker='o', markersize=4, ax=ax)
    
    ax.set_title('Daily Bet Adjustment Volume Over Time (Excluding High-Volume Game)', fontsize=18, weight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Number of Adjustments', fontsize=12)
    ax.legend(title='Day of Week')
    ax.grid(True, which='both', linestyle='--')
    
    plt.savefig(output_dir / "daily_volume_over_time_filtered.png", bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved daily volume over time plot (filtered).")


# --- MAIN ORCHESTRATOR ---

def main():
    """Main script to orchestrate the count-based EDA pipeline."""
    # --- DEFAULT CONFIGURATION ---
    # All database config is now loaded from the .env file.
    OUTPUT_DIR = "filtered_count_reports"
    # -----------------------------

    setup_logging()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Reports will be saved to: {output_dir.resolve()}")

    # Get the database engine, table, and schema from the environment variables
    engine, table_name, schema_name = get_db_config()

    # Load data using the retrieved configuration
    df = load_adjustment_data(engine, table_name, schema_name)

    # Execute all count-based analysis modules
    run_temporal_analysis(df, output_dir)
    run_aggregate_count_analysis(df, output_dir)
    run_comparative_pattern_analysis(df, output_dir)
    
    logging.info("--- Filtered Count Analysis EDA COMPLETE ---")

if __name__ == "__main__":
    main()