# AI Coding Guidelines for Trading System

## Architecture Overview
This is a Python-based trading system that computes technical indicators on stock price data stored in Azure SQL Database. The system follows a modular structure with clear separation between data ingestion, processing, and computation.

**Key Components:**
- `trading_system/`: Main package
  - `ingestion/`: Fetches data from yfinance, stores in staging tables
  - `engine/`: Core processing (runner.py computes indicators, db.py handles DB ops, logger.py for logging)
  - `indicators/`: Technical indicator implementations using pandas/numpy
  - `config/`: Database configuration loader
- `run_master_upgraded.py`: Main entry point for indicator computation
- `tests/`: Custom test suite for indicator validation

## Data Flow
1. **Ingestion**: `ingestion_1m_range_staging_day.py` fetches 1-minute price data via yfinance → inserts into `tblRawPrices` staging
2. **Merging**: `merger.py` moves staging data to production tables
3. **Computation**: `runner.py` loads active indicators from `tblIndicatorDefinitions`, computes them in parallel using ThreadPoolExecutor, stores results in `tblIndicators`

## Dependencies & Environment
- **Core**: pandas, numpy, pyodbc (Azure SQL), yfinance
- **DB**: Azure SQL with ODBC Driver 18; config in `config/db_config.txt` (SERVER, DATABASE, USERNAME, PASSWORD)
- **No virtual env required** - uses system Python with pip installs
- **Timezone**: All timestamps use `America/Toronto` zone

## Critical Workflows
- **Run full pipeline**: `python run_master_upgraded.py` (computes all indicators, max_workers=8)
- **Dry run**: Set `dry_run=True` in master_run() to skip DB inserts
- **Test indicators**: `python tests/test_indicators.py` (custom test runner, logs to `logs/tests/`)
- **Ingest data**: Run `trading_system/ingestion/ingestion_1m_range_staging_day.py` directly
- **Merge staging**: Run `trading_system/ingestion/merger.py` directly

## Project Conventions
- **Imports**: Use relative imports within `trading_system` package; tests use absolute path insertion (`sys.path.insert(0, r"E:\ProjectTrading\trading_system")`)
- **Logging**: Custom logger creates timestamped files in `Logs/` (e.g., `master_run_20260103_004431.txt`); use `log()` function, set prefix with `set_log_prefix()`
- **Indicators**: Registry in `runner.py` INDICATOR_MAP; functions take (df, period) → pandas Series; handle NaN with `.dropna()`
- **DB Schema**: Use square brackets for column names `[ColumnName]`; parameterized inserts with `?` placeholders
- **Parallel Processing**: Use ThreadPoolExecutor for symbol/indicator parallelism; max_workers=8 default
- **Error Handling**: Log exceptions but continue processing; no retries implemented
- **Testing**: Custom assertion-style tests with traceback printing; not using pytest/unittest

## Code Examples
**Computing an indicator:**
```python
# From runner.py
func = INDICATOR_MAP.get(name_upper)
values = func(df_symbol, param)
result_df = pd.DataFrame({
    "Symbol": symbol, "Date": df_symbol["Date"], 
    "IndicatorValue": values
}).dropna(subset=["IndicatorValue"])
```

**DB Query:**
```python
# From db.py
rows = fetch_all("SELECT [Symbol], [ClosePrice] FROM [tblRawPrices]")
df = pd.DataFrame.from_records(rows, columns=["Symbol", "Close"])
```

**Logging:**
```python
# From any module
from trading_system.engine.logger import log
log(f"Processing {len(df)} rows")
```</content>
<parameter name="filePath">e:\ProjectTrading\.github\copilot-instructions.md