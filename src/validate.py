import subprocess
import sys

print("Starting preprocessing pipeline...")

subprocess.run(
    [sys.executable, "src/validate_data.py"],
    check=True
)

subprocess.run(
    [sys.executable, "src/preprocess.py"],
    check=True
)

print("Preprocessing pipeline completed.")