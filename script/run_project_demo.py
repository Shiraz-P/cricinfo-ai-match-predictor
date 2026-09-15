import subprocess
import sys
from pathlib import Path


ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

SCRIPT_DIR = ROOT / "script"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"


def run_script(script_name):

    script_path = SCRIPT_DIR / script_name

    print("\n========================================")
    print(f"RUNNING: {script_name}")
    print("========================================\n")

    if not script_path.exists():

        print(
            f"ERROR: Script not found:\n"
            f"{script_path}"
        )

        input("\nPress Enter to return to menu...")

        return

    try:

        subprocess.run(
            [
                sys.executable,
                str(script_path)
            ],
            cwd=str(ROOT),
            check=False
        )

    except Exception as error:

        print("\nERROR while running script:")
        print(error)

    input("\nPress Enter to return to menu...")


def project_status():

    print("\n========================================")
    print("PROJECT STATUS")
    print("========================================")

    files = {

        "Core dataset":
            DATA_DIR / "matches.csv",

        "Candidate dataset v2":
            DATA_DIR / "matches_candidate_v2.csv",

        "Champion features":
            DATA_DIR / "matches_features.csv",

        "Candidate features":
            DATA_DIR / "matches_candidate_features.csv",

        "Champion model v1.0":
            MODEL_DIR / "cricket_model.pkl",

        "Challenger model v1.1":
            MODEL_DIR / "cricket_model_challenger_v11.pkl",

        "Candidate model v1.2":
            MODEL_DIR / "cricket_model_ablation_9feature.pkl",

        "Production prediction log":
            DATA_DIR / "prediction_log.csv",

        "Candidate v1.2 shadow log":
            DATA_DIR / "shadow_v12_log.csv"
    }

    for name, path in files.items():

        status = "OK" if path.exists() else "MISSING"

        print(
            f"{name:<32} : {status}"
        )

    print("\n----------------------------------------")
    print("MODEL GOVERNANCE")
    print("----------------------------------------")

    print("Production Champion : v1.0")
    print("Challenger v1.1     : REJECTED")
    print("Candidate v1.2      : SHADOW")
    print("Production changed  : NO")

    print("\n----------------------------------------")
    print("VALIDATION SUMMARY")
    print("----------------------------------------")

    print(
        "Champion v1.0 common-set accuracy : 71.82%"
    )

    print(
        "Candidate v1.2 common-set accuracy: 72.26%"
    )

    print(
        "Observed improvement              : +0.44 pp"
    )

    print(
        "McNemar p-value                   : 0.742829"
    )

    print(
        "Statistically significant         : NO"
    )

    print("\nDecision:")

    print(
        "Champion v1.0 remains production."
    )

    print(
        "Candidate v1.2 continues in shadow mode."
    )

    input("\nPress Enter to return to menu...")


def show_menu():

    print("\n")
    print("=" * 62)
    print("CRICINFO AI PROJECT - MAIN MENU")
    print("=" * 62)

    print(" 1. Project Status")
    print(" 2. Dataset Analysis")
    print(" 3. Predict Match - Auto Model Selection")
    print(" 4. Champion vs Challenger")
    print(" 5. Model Monitoring")
    print(" 6. Mean-Based Drift Analysis")
    print(" 7. Statistical Drift Analysis")
    print(" 8. Calibration Analysis")
    print(" 9. Temporal Error Analysis")
    print("10. Feature Reliability Analysis")
    print("11. Shadow Model Monitoring")
    print("12. Retraining Readiness")
    print("13. Reliability Ablation Results")
    print("14. Validate v1.1 Candidate")
    print("15. Update Production Prediction Result")
    print("16. Update Legacy Shadow Result")

    print("-" * 62)

    print("17. Candidate v1.2 Shadow Prediction")
    print("18. Candidate v1.2 Shadow Monitoring")
    print("19. Update Candidate v1.2 Result")

    print("-" * 62)

    print(" 0. Exit")

    print("=" * 62)


while True:

    show_menu()

    choice = input(
        "\nSelect option: "
    ).strip()


    if choice == "1":

        project_status()


    elif choice == "2":

        run_script(
            "analyze_dataset.py"
        )


    elif choice == "3":

        run_script(
            "predict_match.py"
        )


    elif choice == "4":

        run_script(
            "compare_models_live.py"
        )


    elif choice == "5":

        run_script(
            "monitor_model.py"
        )


    elif choice == "6":

        run_script(
            "drift_analysis.py"
        )


    elif choice == "7":

        run_script(
            "statistical_drift.py"
        )


    elif choice == "8":

        run_script(
            "calibration_analysis.py"
        )


    elif choice == "9":

        run_script(
            "temporal_error_analysis.py"
        )


    elif choice == "10":

        run_script(
            "feature_reliability.py"
        )


    elif choice == "11":

        run_script(
            "shadow_monitor.py"
        )


    elif choice == "12":

        run_script(
            "retraining_check.py"
        )


    elif choice == "13":

        run_script(
            "reliability_ablation.py"
        )


    elif choice == "14":

        run_script(
            "validate_reliability_candidate.py"
        )


    elif choice == "15":

        run_script(
            "update_result.py"
        )


    elif choice == "16":

        run_script(
            "update_shadow_result.py"
        )


    elif choice == "17":

        run_script(
            "shadow_predict_v12.py"
        )


    elif choice == "18":

        run_script(
            "monitor_shadow_v12.py"
        )


    elif choice == "19":

        run_script(
            "update_shadow_v12_result.py"
        )


    elif choice == "0":

        print("\n========================================")
        print("PROJECT DEMO CLOSED")
        print("========================================")

        print(
            "Champion v1.0 remains production."
        )

        print(
            "Candidate v1.2 remains shadow."
        )

        break


    else:

        print(
            "\nInvalid option."
        )

        print(
            "Please select a number from 0 to 19."
        )