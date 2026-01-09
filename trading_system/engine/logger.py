import os
import datetime as dt

# ============================================================
# LOG DIRECTORY (DYNAMIC, UNDER PROJECT ROOT)
# ============================================================

# logger.py lives in:
#   E:\ProjectTrading\trading_system\engine\logger.py
#
# We want logs in:
#   E:\ProjectTrading\trading_system\Logs\

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "Logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# INTERNAL: Build log filename
# ============================================================

def _get_log_file(prefix="master_run"):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(LOG_DIR, f"{prefix}_{ts}.txt")

# Default log file (used by master_run)
LOG_FILE = _get_log_file()

# ============================================================
# PUBLIC: Allow scripts to change the prefix
# ============================================================

def set_log_prefix(prefix):
    """
    Switches the log file prefix.
    Example: set_log_prefix("ingestion_run")
    Produces: ingestion_run_YYYYMMDD_HHMMSS.txt
    """
    global LOG_FILE
    LOG_FILE = _get_log_file(prefix)

# ============================================================
# PUBLIC: Logging function
# ============================================================

def log(message):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} [INFO] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")