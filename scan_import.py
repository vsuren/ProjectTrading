import os
import re

PROJECT_ROOT = "E:\\ProjectTrading"
PACKAGE_NAME = "trading_system"

# Regex patterns for import statements
IMPORT_PATTERN = re.compile(r"^\s*import\s+([a-zA-Z0-9_\.]+)")
FROM_PATTERN = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import")

def scan_file(filepath):
    bad_imports = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments
            if line.startswith("#"):
                continue

            # Match "import X"
            m1 = IMPORT_PATTERN.match(line)
            if m1:
                module = m1.group(1)
                if not module.startswith(PACKAGE_NAME) and not module.startswith("python"):
                    bad_imports.append((line, filepath))

            # Match "from X import Y"
            m2 = FROM_PATTERN.match(line)
            if m2:
                module = m2.group(1)
                if not module.startswith(PACKAGE_NAME) and not module.startswith("python"):
                    bad_imports.append((line, filepath))

    return bad_imports


def scan_project():
    print("\n=== SCANNING PROJECT FOR BAD IMPORTS ===\n")
    all_bad = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                bad = scan_file(full_path)
                all_bad.extend(bad)

    if not all_bad:
        print("No bad imports found. All imports look clean.")
        return

    print("⚠ Found imports that are NOT using absolute package paths:\n")
    for line, path in all_bad:
        print(f"[{path}]  -->  {line}")

    print("\nFix these by converting them to:")
    print("    from trading_system.<folder>.<file> import <function>")
    print("\n=== SCAN COMPLETE ===\n")


if __name__ == "__main__":
    scan_project()