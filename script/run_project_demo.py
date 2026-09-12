import subprocess
import sys

from pathlib import Path


# -------------------------------------------------
# Project Paths
# -------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"

MODEL_DIR = PROJECT_DIR / "model"


# -------------------------------------------------
# Helper: Run another Python script
# -------------------------------------------------

def run_script(script_name):

    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():

        print(
            "\nERROR:"
        )

        print(
            "Script not found:"
        )

        print(
            script_path
        )

        return


    print(
        "\n========================================"
    )

    print(
        "RUNNING:",
        script_name
    )

    print(
        "========================================\n"
    )


    try:

        subprocess.run(
            [
                sys.executable,
                str(script_path)
            ],
            check=False
        )

    except Exception as error:

        print(
            "\nERROR while running script:"
        )

        print(
            error
        )


# -------------------------------------------------
# Helper: Pause before returning to menu
# -------------------------------------------------

def pause():

    input(
        "\nPress ENTER to return to main menu..."
    )


# -------------------------------------------------
# Project Status
# -------------------------------------------------

def show_project_status():

    print(
        "\n========================================"
    )

    print(
        "CRICINFO AI - PROJECT STATUS"
    )

    print(
        "========================================"
    )


    champion_model = (
        MODEL_DIR
        /
        "cricket_model.pkl"
    )


    challenger_model = (
        MODEL_DIR
        /
        "cricket_model_v1_1_candidate.pkl"
    )


    feature_data = (
        DATA_DIR
        /
        "matches_features.csv"
    )


    shadow_log = (
        DATA_DIR
        /
        "shadow_comparison_log.csv"
    )


    print(
        "\nProject Directory:"
    )

    print(
        PROJECT_DIR
    )


    print(
        "\n--- DATASET ---"
    )

    print(
        "Feature Dataset:",
        (
            "AVAILABLE"
            if feature_data.exists()
            else "MISSING"
        )
    )


    print(
        "\n--- MODEL STATUS ---"
    )

    print(
        "Champion v1.0:",
        (
            "AVAILABLE"
            if champion_model.exists()
            else "MISSING"
        )
    )

    print(
        "Challenger v1.1:",
        (
            "AVAILABLE"
            if challenger_model.exists()
            else "MISSING"
        )
    )


    print(
        "\n--- CURRENT MODEL ROLES ---"
    )

    print(
        "Champion:"
    )

    print(
        "Logistic Regression v1.0"
    )

    print(
        "Features: 9"
    )

    print(
        "Holdout Accuracy: 71.82%"
    )


    print(
        "\nChallenger:"
    )

    print(
        "Logistic Regression v1.1-candidate"
    )

    print(
        "Features: 11"
    )

    print(
        "Holdout Accuracy: 72.41%"
    )

    print(
        "Brier Score: 0.1928"
    )


    print(
        "\n--- SHADOW TESTING ---"
    )

    print(
        "Shadow Log:",
        (
            "AVAILABLE"
            if shadow_log.exists()
            else "NOT CREATED"
        )
    )


    print(
        "\nProduction Model:"
    )

    print(
        "v1.0 remains Champion."
    )

    print(
        "v1.1 remains Challenger / Shadow."
    )


    print(
        "\nPromotion Status:"
    )

    print(
        "NOT READY FOR PROMOTION"
    )

    print(
        "Live ground-truth collection is still in progress."
    )


# -------------------------------------------------
# Show Main Menu
# -------------------------------------------------

def show_menu():

    print(
        "\n"
        "========================================"
    )

    print(
        "CRICINFO AI - PROJECT DEMONSTRATION"
    )

    print(
        "========================================"
    )

    print(
        "\n1. Project Status"
    )

    print(
        "2. Dataset Analysis"
    )

    print(
        "3. Predict Match - Champion v1.0"
    )

    print(
        "4. Champion vs Challenger"
    )

    print(
        "5. Model Monitoring"
    )

    print(
        "6. Mean-Based Drift Analysis"
    )

    print(
        "7. Statistical Drift Analysis"
    )

    print(
        "8. Calibration Analysis"
    )

    print(
        "9. Temporal Error Analysis"
    )

    print(
        "10. Feature Reliability Analysis"
    )

    print(
        "11. Shadow Model Monitoring"
    )

    print(
        "12. Retraining Readiness"
    )

    print(
        "13. Reliability Ablation Results"
    )

    print(
        "14. Validate v1.1 Candidate"
    )

    print(
        "15. Update Production Prediction Result"
    )

    print(
        "16. Update Shadow Result"
    )

    print(
        "\n0. Exit"
    )


# -------------------------------------------------
# Main Program
# -------------------------------------------------

def main():

    while True:

        show_menu()

        choice = input(
            "\nSelect option: "
        ).strip()


        # -----------------------------------------
        # Exit
        # -----------------------------------------

        if choice == "0":

            print(
                "\nExiting Cricinfo AI Demo."
            )

            print(
                "Thank you."
            )

            break


        # -----------------------------------------
        # Project Status
        # -----------------------------------------

        elif choice == "1":

            show_project_status()

            pause()


        # -----------------------------------------
        # Dataset Analysis
        # -----------------------------------------

        elif choice == "2":

            run_script(
                "analyze_dataset.py"
            )

            pause()


        # -----------------------------------------
        # Champion Prediction
        # -----------------------------------------

        elif choice == "3":

            run_script(
                "predict_match.py"
            )

            pause()


        # -----------------------------------------
        # Champion vs Challenger
        # -----------------------------------------

        elif choice == "4":

            run_script(
                "compare_models_live.py"
            )

            pause()


        # -----------------------------------------
        # Model Monitoring
        # -----------------------------------------

        elif choice == "5":

            run_script(
                "monitor_model.py"
            )

            pause()


        # -----------------------------------------
        # Mean-Based Drift
        # -----------------------------------------

        elif choice == "6":

            run_script(
                "drift_analysis.py"
            )

            pause()


        # -----------------------------------------
        # Statistical Drift
        # -----------------------------------------

        elif choice == "7":

            run_script(
                "statistical_drift.py"
            )

            pause()


        # -----------------------------------------
        # Calibration
        # -----------------------------------------

        elif choice == "8":

            run_script(
                "calibration_analysis.py"
            )

            pause()


        # -----------------------------------------
        # Temporal Error Analysis
        # -----------------------------------------

        elif choice == "9":

            run_script(
                "temporal_error_analysis.py"
            )

            pause()


        # -----------------------------------------
        # Feature Reliability
        # -----------------------------------------

        elif choice == "10":

            run_script(
                "feature_reliability.py"
            )

            pause()


        # -----------------------------------------
        # Shadow Monitoring
        # -----------------------------------------

        elif choice == "11":

            run_script(
                "shadow_monitor.py"
            )

            pause()


        # -----------------------------------------
        # Retraining Check
        # -----------------------------------------

        elif choice == "12":

            run_script(
                "retraining_check.py"
            )

            pause()


        # -----------------------------------------
        # Reliability Ablation
        # -----------------------------------------

        elif choice == "13":

            run_script(
                "reliability_ablation.py"
            )

            pause()


        # -----------------------------------------
        # Candidate Validation
        # -----------------------------------------

        elif choice == "14":

            run_script(
                "validate_reliability_candidate.py"
            )

            pause()


        # -----------------------------------------
        # Update Production Result
        # -----------------------------------------

        elif choice == "15":

            run_script(
                "update_result.py"
            )

            pause()


        # -----------------------------------------
        # Update Shadow Result
        # -----------------------------------------

        elif choice == "16":

            run_script(
                "update_shadow_result.py"
            )

            pause()


        # -----------------------------------------
        # Invalid Choice
        # -----------------------------------------

        else:

            print(
                "\nInvalid option."
            )

            print(
                "Please select a number from 0 to 16."
            )


# -------------------------------------------------
# Program Entry Point
# -------------------------------------------------

if __name__ == "__main__":

    main()