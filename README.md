# Anomalous Bet Adjustments Analysis

This project provides tools for analyzing bet adjustment data from a Redshift database. It offers two main analysis scripts:

1. **anomalous_bet_adjustments.py** - Performs count-based analysis of all bet adjustments.
2. **hotspot_analysis.py** - Performs filtered analysis excluding a high-volume game (Jade Rabbit).

## Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) (Modern Python package manager)
- Access to an AWS Redshift cluster with the required table

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd anomalous_bet_adjustments
   ```

2. Install dependencies using uv:

   ```bash
   # Install dependencies from pyproject.toml
   uv sync
   
   # Or install individual packages if needed
   # uv add <package-name>
   ```

   The project will automatically create and manage a virtual environment for you.

## Configuration

Create a `.env` file in the project root with the following variables:

```
REDSHIFT_HOST=<your-redshift-host>
REDSHIFT_PORT=5439
REDSHIFT_DB=<your-database>
REDSHIFT_USER=<your-username>
REDSHIFT_PASSWORD=<your-password>
REDSHIFT_TABLE=<your-schema>.<your-table>
REDSHIFT_SCHEMA=public
```

## Usage

### Basic Analysis

Run the main analysis script for all bet adjustments:

```bash
uv run python anomalous_bet_adjustments.py
```

This will:
- Load adjustment data from the Redshift table
- Compute count-based analyses for different time periods (5-min, hourly, daily)
- Generate temporal analysis visualizations (hour of day vs. day of week)
- Calculate statistics for aggregate count analysis and rate of change
- Save all plots and statistics in the `count_analysis_reports` directory

### Filtered Analysis

To analyze the data excluding the high-volume Jade Rabbit game:

```bash
uv run python hotspot_analysis.py
```

This will:
- Load adjustment data excluding the specified high-volume game
- Perform similar temporal and aggregate count analyses
- Generate additional comparative pattern analyses
- Create visualizations showing patterns by hour of day and day of week
- Save all filtered analysis outputs in the `filtered_count_reports` directory

## Output Files

### From anomalous_bet_adjustments.py:
- `temporal_heatmap_counts.png` - Heatmap of adjustment activity by hour and weekday
- `timeseries_5-minute_counts.png` - Time series of 5-minute adjustment counts
- `timeseries_15-minute_counts.png` - Time series of 15-minute adjustment counts
- `timeseries_hourly_counts.png` - Time series of hourly adjustment counts
- `timeseries_daily_counts.png` - Time series of daily adjustment counts
- Various CSV files with count statistics at different time granularities

### From hotspot_analysis.py:
- `temporal_heatmap_counts_filtered.png` - Heatmap excluding the high-volume game
- `comparative_hourly_patterns_filtered.png` - Line chart of hourly patterns by day of week
- `daily_volume_over_time_filtered.png` - Time series of daily volumes with weekday highlighted
- Various filtered CSV statistics files

## Package Management with uv

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management:

- **Install dependencies**: `uv sync` (installs from pyproject.toml and uv.lock)
- **Add new packages**: `uv add <package-name>`
- **Remove packages**: `uv remove <package-name>`
- **Run scripts**: `uv run python <script-name>`
- **Update dependencies**: `uv lock --upgrade`

## Next Steps

Use the insight from these visualizations to:
1. Establish normal baseline activity patterns
2. Determine threshold values for alerting on anomalous activity
3. Implement automated monitoring using the identified thresholds

---