import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================", flush=True)
    print("[INFO] Starting Lab 5: Crop Recommendation Validation Pipeline", flush=True)
    print("[INFO] =========================================", flush=True)

    scripts = [
        "src/validate_data.py",
        "src/validate.py",
        "src/validate_outputs.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...", flush=True)

        result = subprocess.run([sys.executable, script])

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.", flush=True)
            sys.exit(1)

    print("\n[SUCCESS] Lab 5 Pipeline fully executed!", flush=True)

if __name__ == "__main__":
    execute_pipeline()
