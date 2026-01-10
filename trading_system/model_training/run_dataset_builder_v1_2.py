import os
import sys
import logging
from datetime import datetime
import traceback

# ------------------------------------------------------------
# Ensure project root is on sys.path
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------
# Imports from your project
# ------------------------------------------------------------
from trading_system.config.config_loader import load_db_config
from trading_system.model_training.train_model_v1_2 import build_engine
from trading_system.model_training.dataset_builder_v1_2 import DatasetBuilderV12


def setup_runner_logging():
    """Configure logging for the runner script."""
    logs_dir = r"E:\ProjectTrading\trading_system\Logs"
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(
        logs_dir,
        f"run_dataset_builder_v1_2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger = logging.getLogger("DatasetBuilderRunner")
    logger.info("Runner logging initialized")
    return logger


def main():
    logger = setup_runner_logging()

    try:
        logger.info("=== Starting Dataset Builder v1.2 Runner ===")

        # Load DB credentials
        logger.info("Loading DB config...")
        cfg = load_db_config()

        # Create SQLAlchemy engine
        logger.info("Building SQL engine...")
        engine = build_engine(cfg)

        # Run dataset builder
        logger.info("Instantiating DatasetBuilderV12...")
        builder = DatasetBuilderV12(engine)

        logger.info("Running dataset build...")
        metadata = builder.build()

        logger.info("=== Dataset Builder v1.2 Completed Successfully ===")
        logger.info(f"Metadata: {metadata}")

        print("Dataset Builder v1.2 completed successfully.")
        print(metadata)

    except Exception as e:
        logger.error("Runner failed with an exception")
        logger.error(str(e))
        logger.error(traceback.format_exc())
        print("Runner failed. Check log file for details.")
        raise


if __name__ == "__main__":
    main()