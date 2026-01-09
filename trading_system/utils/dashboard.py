import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

LOG_PATH = r"E:\ProjectTrading\trading_system\logs\orchestrator_run_20260108_193605.txt"  # Update if needed

def extract_timestamp(line):
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S") if match else None

def extract_float(line):
    matches = re.findall(r"\d+\.\d+", line)
    return float(matches[-1]) if matches else None

def extract_int(line):
    nums = re.findall(r"\d+", line)
    return int(nums[-1]) if nums else None

def dashboard():
    log_file = Path(LOG_PATH)
    if not log_file.exists():
        print(f"Log file not found: {LOG_PATH}")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metrics = {
        "ingest_rows": 0,
        "ingest_symbols": 0,
        "ingest_runtime": None,
        "merge_rows": 0,
        "merge_runtime": None,
        "indicator_rows": 0,
        "indicator_runtime": None,
        "feature_rows": 0,
        "feature_runtime": None,
        "total_runtime": None
    }

    start_time = None
    end_time = None
    ingest_times = {}
    indicator_times = defaultdict(list)
    raw_before = None
    raw_after = None

    current_symbol = None
    current_indicator = None
    current_indicator_start = None

    for line in lines:
        ts = extract_timestamp(line)

        if "MASTER ORCHESTRATOR STARTED" in line:
            start_time = ts

        if "Completed: merge_feature.py" in line:
            end_time = ts

        if "Total rows inserted:" in line:
            metrics["ingest_rows"] = extract_int(line)

        if "Total symbols processed:" in line:
            metrics["ingest_symbols"] = extract_int(line)

        if "INGESTION COMPLETED in" in line:
            metrics["ingest_runtime"] = extract_float(line)

        if "Inserted" in line and "tblRawPrices" in line:
            metrics["merge_rows"] = extract_int(line)

        if "MERGE COMPLETED in" in line:
            metrics["merge_runtime"] = extract_float(line)

        if "Total computed indicator rows" in line:
            metrics["indicator_rows"] = extract_int(line)

        if "MASTER RUN COMPLETED in" in line:
            metrics["indicator_runtime"] = extract_float(line)

        if "Merged dataset:" in line:
            metrics["feature_rows"] = extract_int(line)

        if "Insert step completed in" in line and "tblMergedFeatures" not in line:
            metrics["feature_runtime"] = extract_float(line)

        if "Total rows in RawPrices after merge" in line:
            raw_after = extract_int(line)

        if "Loaded 31049 price rows" in line:
            raw_before = extract_int(line)

        # Per-symbol ingestion
        if "Processing" in line and "(Backfill=" in line:
            current_symbol = re.findall(r"Processing (\S+)", line)[0]
            ingest_times[current_symbol] = {"start": ts}

        if "Ingestion completed in" in line and current_symbol:
            ingest_times[current_symbol]["duration"] = extract_float(line)
            current_symbol = None

        # Indicator runtimes
        if "Starting indicator" in line:
            current_indicator = re.findall(r"indicator (\S+)", line)[0]
            current_indicator_start = ts

        if "Completed indicator" in line and current_indicator:
            duration = extract_float(line)
            indicator_times[current_indicator].append(duration)
            current_indicator = None

    if start_time and end_time:
        metrics["total_runtime"] = round((end_time - start_time).total_seconds(), 2)

    # Print dashboard
    print("\n" + "="*70)
    print("                 TRADING SYSTEM DAILY DASHBOARD")
    print("="*70)

    print(f"\n📌 INGESTION")
    print(f"   Symbols processed:     {metrics['ingest_symbols']}")
    print(f"   Rows inserted:         {metrics['ingest_rows']:,}")
    print(f"   Runtime:               {metrics['ingest_runtime']} sec")

    print(f"\n📌 MERGE (Staging → RawPrices)")
    print(f"   Rows merged:           {metrics['merge_rows']:,}")
    print(f"   RawPrices before:      {raw_before:,}")
    print(f"   RawPrices after:       {raw_after:,}")
    print(f"   Runtime:               {metrics['merge_runtime']} sec")

    print(f"\n📌 INDICATORS")
    print(f"   Indicator rows:        {metrics['indicator_rows']:,}")
    print(f"   Runtime:               {metrics['indicator_runtime']} sec")

    print(f"\n📌 MERGED FEATURES")
    print(f"   Final rows:            {metrics['feature_rows']:,}")
    print(f"   Runtime:               {metrics['feature_runtime']} sec")

    print(f"\n📌 TOTAL PIPELINE RUNTIME")
    print(f"   Total runtime:         {metrics['total_runtime']} sec")

    print("\n" + "="*70)
    print("                 PER-SYMBOL INGESTION TIMES")
    print("="*70)
    for sym, data in ingest_times.items():
        dur = data.get("duration", "N/A")
        print(f"   {sym:<8} → {dur} sec")

    print("\n" + "="*70)
    print("                 INDICATOR RUNTIMES (avg per type)")
    print("="*70)
    for ind, durations in indicator_times.items():
        avg = round(sum(durations)/len(durations), 2)
        print(f"   {ind:<15} → {avg} sec")

    print("\n" + "="*70)
    print("                     DASHBOARD COMPLETE")
    print("="*70)

if __name__ == "__main__":
    dashboard()