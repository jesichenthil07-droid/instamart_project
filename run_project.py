
import subprocess
import sys
from pathlib import Path

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# All project stages in execution order
steps = [
    ("Generating datasets", "src/generate_data.py"),
    ("Cleaning datasets", "src/data_cleaning.py"),
    ("Creating SQLite database", "src/database.py"),
    ("Building RFM segmentation", "src/rfm_segmentation.py"),
    ("Training churn prediction model", "src/churn_prediction.py"),
    ("Processing support tickets", "src/triage.py"),
    ("Running RAG pipeline", "src/rag_pipeline.py")
]


def run_project():

    print("\n================================")
    print("   INSTAMART PROJECT STARTED")
    print("================================\n")

    for step_name, script_path in steps:

        print(f"\n>>> {step_name}")

        try:
            subprocess.run(
                [sys.executable, str(BASE_DIR / script_path)],
                cwd=BASE_DIR,
                check=True
            )

            print(f">>> {step_name} completed!")

        except subprocess.CalledProcessError as error:

            print(f"\nERROR: {step_name} failed!")
            print("Exit code:", error.returncode)
            print("Please check the error above.")

            sys.exit(error.returncode)

    print("\n================================")
    print(" INSTAMART PROJECT COMPLETED!")
    print("================================")

    print("\nCheck the reports and models folders.")


if __name__ == "__main__":
    run_project()
