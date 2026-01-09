from trading_system.engine.runner import master_run

if __name__ == "__main__":
    master_run(dry_run=False, max_workers=8)