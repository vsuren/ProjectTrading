import os
import sys
import logging
import traceback
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

# ------------------------------------------------------------
# Ensure project root is on sys.path
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def setup_logging():
    logs_dir = r"E:\ProjectTrading\trading_system\Logs"
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(
        logs_dir,
        f"verify_dataset_v1_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("DatasetVerifier")
    logger.info("Verification logging initialized")
    return logger


def main():
    logger = setup_logging()

    try:
        logger.info("=== Starting Dataset v1.2 Verification ===")

        parquet_path = os.path.join(
            CURRENT_DIR,
            "datasets",
            "dataset_v1.2.parquet"
        )

        # ------------------------------------------------------------
        # Check file existence
        # ------------------------------------------------------------
        if not os.path.exists(parquet_path):
            logger.error(f"Dataset file not found: {parquet_path}")
            print("Dataset file not found.")
            return

        logger.info(f"Dataset file found: {parquet_path}")

        # ------------------------------------------------------------
        # Load parquet
        # ------------------------------------------------------------
        logger.info("Loading parquet file...")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        logger.info("Parquet loaded successfully")

        # ------------------------------------------------------------
        # Basic stats
        # ------------------------------------------------------------
        row_count = len(df)
        col_count = len(df.columns)

        logger.info(f"Row count: {row_count:,}")
        logger.info(f"Column count: {col_count}")

        print(f"Rows: {row_count:,}")
        print(f"Columns: {col_count}")

        # ------------------------------------------------------------
        # Metadata columns
        # ------------------------------------------------------------
        metadata_cols = [
            "Label",
            "FeatureVersion",
            "LabelVersion",
            "MergeRunId",
            "PriceTimestamp",
        ]

        logger.info(f"Metadata columns present: {metadata_cols}")
        print("\nMetadata columns:")
        for c in metadata_cols:
            print(f"  - {c}")

        # ------------------------------------------------------------
        # Feature columns
        # ------------------------------------------------------------
        feature_cols = [c for c in df.columns if c not in metadata_cols]

        logger.info(f"Feature column count: {len(feature_cols)}")

        print("\nFirst 10 feature columns:")
        for c in feature_cols[:10]:
            print(f"  - {c}")

        print("\nLast 10 feature columns:")
        for c in feature_cols[-10:]:
            print(f"  - {c}")

        # ------------------------------------------------------------
        # Check for NaNs
        # ------------------------------------------------------------
        nan_counts = df.isna().sum().sum()
        logger.info(f"Total NaN values: {nan_counts}")

        print(f"\nTotal NaN values: {nan_counts}")

        # ------------------------------------------------------------
        # Check for duplicate timestamps
        # ------------------------------------------------------------
        dupes = df["PriceTimestamp"].duplicated().sum()
        logger.info(f"Duplicate timestamps: {dupes}")

        print(f"Duplicate timestamps: {dupes}")

        # ------------------------------------------------------------
        # Check metadata consistency
        # ------------------------------------------------------------
        fv = df["FeatureVersion"].unique()
        lv = df["LabelVersion"].unique()
        mr = df["MergeRunId"].unique()

        logger.info(f"FeatureVersion values: {fv}")
        logger.info(f"LabelVersion values: {lv}")
        logger.info(f"MergeRunId values: {mr}")

        print("\nMetadata consistency:")
        print(f"  FeatureVersion: {fv}")
        print(f"  LabelVersion: {lv}")
        print(f"  MergeRunId: {mr}")

        logger.info("=== Dataset v1.2 Verification Completed Successfully ===")
        print("\nVerification completed successfully.")

    except Exception as e:
        logger.error("Verification failed")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        print("Verification failed. Check log file for details.")
        raise


if __name__ == "__main__":
    main()