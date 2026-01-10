import os
import pandas as pd
import sqlalchemy
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa
import logging
import traceback


class DatasetBuilderV12:
    """
    Dataset builder for Model v1.2.
    Extracts engineered features + labels from tblLabeledFeatures,
    filters trainable rows, and freezes a reproducible dataset snapshot.
    """

    def __init__(self, engine: sqlalchemy.engine.Engine):
        self.engine = engine
        self.dataset_version = "1.2"

        # Save inside model_training/datasets/
        base_dir = os.path.dirname(__file__)
        self.output_path = os.path.join(
            base_dir,
            "datasets",
            f"dataset_v{self.dataset_version}.parquet"
        )

        # ------------------------------------------------------------
        # Logging setup
        # ------------------------------------------------------------
        logs_dir = r"E:\ProjectTrading\trading_system\Logs"
        os.makedirs(logs_dir, exist_ok=True)

        log_file = os.path.join(
            logs_dir,
            f"dataset_builder_v{self.dataset_version}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        self.logger = logging.getLogger("DatasetBuilderV12")
        self.logger.info("Initialized DatasetBuilderV12")

    # ------------------------------------------------------------
    # Load FeatureSetID = 4 (ordered)
    # ------------------------------------------------------------
    def load_feature_set(self) -> list:
        self.logger.info("Loading FeatureSetID = 4 from tblFeatureSetMembers")

        query = """
            SELECT FeatureName
            FROM dbo.tblFeatureSetMembers
            WHERE FeatureSetID = 4
            ORDER BY Ordinal;
        """

        df = pd.read_sql(query, self.engine)
        feature_list = df["FeatureName"].tolist()

        self.logger.info(f"Loaded {len(feature_list)} features from FeatureSetID = 4")
        return feature_list

    # ------------------------------------------------------------
    # Load labeled features from SQL
    # ------------------------------------------------------------
    def load_from_sql(self) -> pd.DataFrame:
        self.logger.info("Loading trainable rows from tblLabeledFeatures")

        query = """
            SELECT *
            FROM dbo.tblLabeledFeatures
            WHERE IsTrainableRow = 1
              AND IsOutlier = 0
            ORDER BY PriceTimestamp ASC;
        """

        df = pd.read_sql(query, self.engine)
        self.logger.info(f"Loaded {len(df):,} rows from SQL")
        return df

    # ------------------------------------------------------------
    # Select metadata + ordered feature columns
    # ------------------------------------------------------------
    def select_feature_columns(self, df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
        self.logger.info("Selecting metadata + ordered feature columns")

        metadata_cols = [
	    "Symbol",
            "Label",
            "FeatureVersion",
            "LabelVersion",
            "MergeRunId",
            "PriceTimestamp",
        ]

        missing = [c for c in feature_list if c not in df.columns]
        if missing:
            self.logger.warning(f"Missing features in dataset: {missing}")

        available_features = [c for c in feature_list if c in df.columns]

        final_cols = metadata_cols + available_features
        self.logger.info(f"Final dataset will contain {len(final_cols)} columns")
        return df[final_cols]

    # ------------------------------------------------------------
    # Save parquet
    # ------------------------------------------------------------
    def save_parquet(self, df: pd.DataFrame):
        self.logger.info(f"Saving dataset to {self.output_path}")

        table = pa.Table.from_pandas(df)
        pq.write_table(table, self.output_path)

        self.logger.info("Dataset saved successfully")

    # ------------------------------------------------------------
    # Full build pipeline
    # ------------------------------------------------------------
    def build(self):
        try:
            self.logger.info("=== Starting Dataset Build v1.2 ===")

            feature_list = self.load_feature_set()
            df = self.load_from_sql()
            df_final = self.select_feature_columns(df, feature_list)
            self.save_parquet(df_final)

            metadata = {
                "dataset_version": self.dataset_version,
                "row_count": len(df_final),
                "feature_count": len(df_final.columns) - 5,
                "timestamp": datetime.utcnow().isoformat(),
                "feature_version": df_final["FeatureVersion"].iloc[-1],
                "label_version": df_final["LabelVersion"].iloc[-1],
                "merge_run_id": df_final["MergeRunId"].iloc[-1],
            }

            self.logger.info("Dataset v1.2 built successfully")
            self.logger.info(f"Metadata: {metadata}")

            return metadata

        except Exception as e:
            self.logger.error("Dataset build failed")
            self.logger.error(str(e))
            self.logger.error(traceback.format_exc())
            raise