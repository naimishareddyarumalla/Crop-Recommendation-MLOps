import subprocess
import sys

def execute_pipeline():
    print("[INFO] =========================================", flush=True)
    print("[INFO] Starting Lab 6: Hyperparameter Tuning and Prediction", flush=True)
    print("[INFO] =========================================", flush=True)

    scripts = [
        "src/tune.py",
        "src/predict.py",
        "src/validate_outputs.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...", flush=True)

        result = subprocess.run([sys.executable, script])

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} failed.", flush=True)
            sys.exit(1)

    print("\n[SUCCESS] Lab 6 Pipeline fully executed!", flush=True)

if __name__ == "__main__":
    execute_pipeline()
