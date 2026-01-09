import os

def load_db_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "db_config.txt")

    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()

    required = ["SERVER", "DATABASE", "USERNAME", "PASSWORD"]
    for r in required:
        if r not in config:
            raise Exception(f"Missing '{r}' in db_config.txt")

    return config