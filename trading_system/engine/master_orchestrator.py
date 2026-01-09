import argparse
import subprocess
import sys
import os

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from trading_system.engine.logger import log, set_log_prefix

# ============================================================
# INTERNAL FLAGS — FULL BOOLEAN CONTROL
# ============================================================

RUN_INGEST = True
RUN_MERGE = True
RUN_INDICATORS = True
RUN_FEATURES = True
RUN_LABELS = False          # <-- NEW FLAG
DRY_RUN = False            # Applies ONLY to indicators + features + labels

# Script paths
INGEST_SCRIPT = os.path.join(PROJECT_ROOT, "trading_system", "ingestion", "ingestion_1m_range_staging_day.py")
MERGER_SCRIPT = os.path.join(PROJECT_ROOT, "trading_system", "ingestion", "merger.py")
INDICATOR_SCRIPT = os.path.join(PROJECT_ROOT, "run_master_upgraded.py")
FEATURE_SCRIPT = os.path.join(PROJECT_ROOT, "merge_feature.py")
LABEL_SCRIPT = os.path.join(PROJECT_ROOT, "generate_labeled_features.py")   # <-- NEW SCRIPT PATH

# ============================================================
# RUN A PYTHON SCRIPT AS A SUBPROCESS
# ============================================================

def run_script(path, args=None):
    cmd = [sys.executable, path]
    if args:
        cmd.extend(args)

    log(f"Running: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    log("----- STDOUT -----")
    if result.stdout.strip():
        log(result.stdout.strip())

    if result.stderr.strip():
        log("----- STDERR -----")
        log(result.stderr.strip())

    if result.returncode != 0:
        log(f"ERROR: Script failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    log(f"Completed: {os.path.basename(path)}")

# ============================================================
# MASTER ORCHESTRATOR
# ============================================================

def main():
    set_log_prefix("orchestrator_run")
    log("=== MASTER ORCHESTRATOR STARTED ===")

    # Track what ran for summary
    summary = {
        "ingest": RUN_INGEST,
        "merge": RUN_MERGE,
        "indicators": RUN_INDICATORS,
        "features": RUN_FEATURES,
        "labels": RUN_LABELS,
        "dry_run": DRY_RUN
    }

    # --------------------------------------------------------
    # 1. INGESTION
    # --------------------------------------------------------
    if RUN_INGEST:
        log("STEP 1: Ingestion → Staging")
        run_script(INGEST_SCRIPT)
    else:
        log("STEP 1 SKIPPED")

    # --------------------------------------------------------
    # 2. MERGE STAGING → RAW
    # --------------------------------------------------------
    if RUN_MERGE:
        log("STEP 2: Merge Staging → RawPrices")
        run_script(MERGER_SCRIPT)
    else:
        log("STEP 2 SKIPPED")

    # --------------------------------------------------------
    # 3. INDICATOR COMPUTATION
    # --------------------------------------------------------
    if RUN_INDICATORS:
        log("STEP 3: Indicator Computation")
        args = ["--dry-run"] if DRY_RUN else []
        run_script(INDICATOR_SCRIPT, args)
    else:
        log("STEP 3 SKIPPED")

    # --------------------------------------------------------
    # 4. FEATURE MERGE
    # --------------------------------------------------------
    if RUN_FEATURES:
        log("STEP 4: Merge Raw + Indicators → tblMergedFeatures")
        args = ["--dry-run"] if DRY_RUN else []
        run_script(FEATURE_SCRIPT, args)
    else:
        log("STEP 4 SKIPPED")

    # --------------------------------------------------------
    # 5. LABEL GENERATION  <-- NEW STEP
    # --------------------------------------------------------
    if RUN_LABELS:
        log("STEP 5: Generate Labels → tblLabeledFeatures")
        args = ["--dry-run"] if DRY_RUN else []
        run_script(LABEL_SCRIPT, args)
    else:
        log("STEP 5 SKIPPED")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------
    log("=== RUN SUMMARY ===")
    log(f"INGESTION:   {'RUN' if summary['ingest'] else 'SKIPPED'}")
    log(f"MERGE:       {'RUN' if summary['merge'] else 'SKIPPED'}")
    log(f"INDICATORS:  {'RUN' if summary['indicators'] else 'SKIPPED'}")
    log(f"FEATURES:    {'RUN' if summary['features'] else 'SKIPPED'}")
    log(f"LABELS:      {'RUN' if summary['labels'] else 'SKIPPED'}")
    log(f"DRY_RUN:     {summary['dry_run']}")
    log("====================")

    log("=== MASTER ORCHESTRATOR COMPLETED ===")


if __name__ == "__main__":
    main()