import os
import sys
import logging
import traceback
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
import sqlalchemy

# ------------------------------------------------------------
# Ensure project root is on sys.path
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from trading_system.config.config_loader import load_db_config
from trading_system.model_training.train_model_v1_2 import build_engine


def setup_logging():
    logs_dir = r"E:\ProjectTrading\trading_system\Logs"
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(
        logs_dir,
        f"deep_dataset_integrity_v1_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("DeepDatasetIntegrity")
    logger.info("Deep integrity logging initialized")
    return logger


def load_feature_set(engine):
    query = """
        SELECT FeatureName
        FROM dbo.tblFeatureSetMembers
        WHERE FeatureSetID = 4
        ORDER BY Ordinal;
    """
    df = pd.read_sql(query, engine)
    return df["FeatureName"].tolist()


def main():
    logger = setup_logging()

    try:
        logger.info("=== Starting Deep Dataset Integrity Check v1.2 ===")

        parquet_path = os.path.join(
            CURRENT_DIR,
            "datasets",
            "dataset_v1.2.parquet"
        )

        if not os.path.exists(parquet_path):
            logger.error(f"Dataset file not found: {parquet_path}")
            print("Dataset file not found.")
            return

        logger.info(f"Dataset file found: {parquet_path}")

        # Load parquet
        logger.info("Loading parquet file...")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        logger.info("Parquet loaded successfully")

        # Load feature set from SQL
        logger.info("Loading FeatureSetID = 4 from SQL...")
        cfg = load_db_config()
        engine = build_engine(cfg)
        feature_set = load_feature_set(engine)
        logger.info(f"FeatureSetID = 4 contains {len(feature_set)} features")

        # ------------------------------------------------------------
        # 1. Check (Symbol, PriceTimestamp) duplicates
        # ------------------------------------------------------------
        if "Symbol" in df.columns:
            dupes = df.duplicated(subset=["Symbol", "PriceTimestamp"]).sum()
            logger.info(f"Duplicate (Symbol, PriceTimestamp) rows: {dupes}")
            print(f"Duplicate (Symbol, PriceTimestamp): {dupes}")
        else:
            logger.warning("Column 'Symbol' not found in dataset")
            print("WARNING: No Symbol column found")

        # ------------------------------------------------------------
        # 2. NaN % per feature
        # ------------------------------------------------------------
        logger.info("Computing NaN percentages per feature...")
        nan_report = df.isna().mean().sort_values(ascending=False)

        high_nan = nan_report[nan_report > 0.50]
        full_nan = nan_report[nan_report == 1.0]

        logger.info(f"Features with >50% NaN: {list(high_nan.index)}")
        logger.info(f"Features with 100% NaN: {list(full_nan.index)}")

        print("\nFeatures with >50% NaN:")
        for f, pct in high_nan.items():
            print(f"  {f}: {pct:.2%}")

        print("\nFeatures with 100% NaN:")
        for f in full_nan.index:
            print(f"  {f}")

        # ------------------------------------------------------------
        # 3. Zero-variance features
        # ------------------------------------------------------------
        logger.info("Checking for zero-variance features...")
        zero_var = [c for c in df.columns if df[c].nunique() <= 1]

        logger.info(f"Zero-variance features: {zero_var}")
        print("\nZero-variance features:")
        for f in zero_var:
            print(f"  {f}")

        # ------------------------------------------------------------
        # 4. Compare parquet columns vs FeatureSetID = 4
        # ------------------------------------------------------------
        parquet_cols = set(df.columns)
        feature_set_cols = set(feature_set)

        missing_from_parquet = feature_set_cols - parquet_cols
        extra_in_parquet = parquet_cols - feature_set_cols - {
            "Label", "FeatureVersion", "LabelVersion", "MergeRunId", "PriceTimestamp"
        }

        logger.info(f"Missing features from parquet: {missing_from_parquet}")
        logger.info(f"Extra features in parquet: {extra_in_parquet}")

        print("\nMissing features from parquet:")
        for f in missing_from_parquet:
            print(f"  {f}")

        print("\nExtra features in parquet:")
        for f in extra_in_parquet:
            print(f"  {f}")

        logger.info("=== Deep Dataset Integrity Check Completed Successfully ===")
        print("\nDeep integrity check completed successfully.")

    except Exception as e:
        logger.error("Integrity check failed")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        print("Integrity check failed. Check log file for details.")
        raise


if __name__ == "__main__":
    main()