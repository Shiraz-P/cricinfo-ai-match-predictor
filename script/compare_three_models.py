import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

CHAMPION_MODEL = ROOT / "model" / "cricket_model.pkl"
ABLATION_MODEL = ROOT / "model" / "cricket_model_ablation_9feature.pkl"
CHALLENGER_MODEL = ROOT / "model" / "cricket_model_challenger_v11.pkl"

CHAMPION_DATA = ROOT / "data" / "matches_features.csv"
CANDIDATE_DATA = ROOT / "data" / "matches_candidate_features.csv"

OUTPUT_FILE = (
    ROOT
    / "data"
    / "three_model_comparison.csv"
)

FEATURES_9 = [
    "toss_winner_is_team1",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate"
]

FEATURES_10 = [
    "toss_winner_is_team1",
    "toss_data_known",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate"
]


def extract_model(package):

    if isinstance(package, dict):
        return package["model"]

    return package


print("========================================")
print("THREE-MODEL FAIR COMPARISON")
print("========================================")


# -------------------------------------------------
# Load models
# -------------------------------------------------

champion = extract_model(
    joblib.load(CHAMPION_MODEL)
)

ablation = extract_model(
    joblib.load(ABLATION_MODEL)
)

challenger = extract_model(
    joblib.load(CHALLENGER_MODEL)
)


# -------------------------------------------------
# Load datasets
# -------------------------------------------------

champion_df = pd.read_csv(
    CHAMPION_DATA
)

candidate_df = pd.read_csv(
    CANDIDATE_DATA
)


for df in [champion_df, candidate_df]:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


champion_df = champion_df.sort_values(
    "date",
    kind="stable"
).reset_index(drop=True)

candidate_df = candidate_df.sort_values(
    "date",
    kind="stable"
).reset_index(drop=True)


# -------------------------------------------------
# Champion 80/20 test set
# -------------------------------------------------

split = int(
    len(champion_df) * 0.80
)

champion_test = champion_df.iloc[
    split:
].copy()


print(
    "\nChampion total rows:",
    len(champion_df)
)

print(
    "Champion test rows:",
    len(champion_test)
)

print(
    "Test period:",
    champion_test["date"].min(),
    "to",
    champion_test["date"].max()
)


# -------------------------------------------------
# Exact historical match key
# -------------------------------------------------

def create_key(df):

    result = df.copy()

    result["_base_key"] = (
        result["date"]
        .dt.strftime("%Y-%m-%d")
        + "|"
        + result["team1"].astype(str)
        + "|"
        + result["team2"].astype(str)
        + "|"
        + result["winner"].astype(str)
    )

    result["_occurrence"] = (
        result.groupby(
            "_base_key"
        ).cumcount()
    )

    result["_key"] = (
        result["_base_key"]
        + "|"
        + result["_occurrence"].astype(str)
    )

    return result


champion_test = create_key(
    champion_test
)

candidate_df = create_key(
    candidate_df
)


# -------------------------------------------------
# Match exact Champion test records
# -------------------------------------------------

test_keys = set(
    champion_test["_key"]
)

candidate_test = candidate_df[
    candidate_df["_key"].isin(
        test_keys
    )
].copy()


candidate_keys = set(
    candidate_test["_key"]
)


champion_test = champion_test[
    champion_test["_key"].isin(
        candidate_keys
    )
].copy()


champion_test = (
    champion_test
    .sort_values("_key")
    .reset_index(drop=True)
)

candidate_test = (
    candidate_test
    .sort_values("_key")
    .reset_index(drop=True)
)


print("\n========================================")
print("ALIGNMENT VALIDATION")
print("========================================")

print(
    "Successfully matched:",
    len(champion_test)
)


if len(champion_test) != len(
    candidate_test
):

    raise ValueError(
        "Row counts do not match."
    )


key_alignment = (
    champion_test["_key"].values
    ==
    candidate_test["_key"].values
).all()


team1_alignment = (
    champion_test["team1"].values
    ==
    candidate_test["team1"].values
).all()


team2_alignment = (
    champion_test["team2"].values
    ==
    candidate_test["team2"].values
).all()


winner_alignment = (
    champion_test["winner"].values
    ==
    candidate_test["winner"].values
).all()


print(
    "Key alignment:",
    key_alignment
)

print(
    "Team1 alignment:",
    team1_alignment
)

print(
    "Team2 alignment:",
    team2_alignment
)

print(
    "Winner alignment:",
    winner_alignment
)


if not all([
    key_alignment,
    team1_alignment,
    team2_alignment,
    winner_alignment
]):

    raise ValueError(
        "Datasets are not safely aligned."
    )


# -------------------------------------------------
# Actual result
# -------------------------------------------------

y_actual = (
    champion_test["winner"]
    ==
    champion_test["team1"]
).astype(int)


# -------------------------------------------------
# Predictions
# -------------------------------------------------

champion_pred = champion.predict(
    champion_test[FEATURES_9]
)


ablation_pred = ablation.predict(
    candidate_test[FEATURES_9]
)


challenger_pred = challenger.predict(
    candidate_test[FEATURES_10]
)


# -------------------------------------------------
# Accuracy
# -------------------------------------------------

champion_accuracy = accuracy_score(
    y_actual,
    champion_pred
)

ablation_accuracy = accuracy_score(
    y_actual,
    ablation_pred
)

challenger_accuracy = accuracy_score(
    y_actual,
    challenger_pred
)


print("\n========================================")
print("FAIR HEAD-TO-HEAD RESULTS")
print("========================================")

print(
    "Common matches:",
    len(y_actual)
)

print(
    "Champion v1.0:",
    f"{champion_accuracy:.4f}",
    f"({champion_accuracy * 100:.2f}%)"
)

print(
    "Ablation 9-feature:",
    f"{ablation_accuracy:.4f}",
    f"({ablation_accuracy * 100:.2f}%)"
)

print(
    "Challenger v1.1:",
    f"{challenger_accuracy:.4f}",
    f"({challenger_accuracy * 100:.2f}%)"
)


# -------------------------------------------------
# Difference from Champion
# -------------------------------------------------

ablation_difference = (
    ablation_accuracy
    -
    champion_accuracy
)

challenger_difference = (
    challenger_accuracy
    -
    champion_accuracy
)


print("\n========================================")
print("DIFFERENCE FROM CHAMPION")
print("========================================")

print(
    "Ablation:",
    f"{ablation_difference * 100:+.2f}",
    "percentage points"
)

print(
    "Challenger:",
    f"{challenger_difference * 100:+.2f}",
    "percentage points"
)


# -------------------------------------------------
# Correct prediction counts
# -------------------------------------------------

champion_correct = (
    champion_pred
    ==
    y_actual.values
)

ablation_correct = (
    ablation_pred
    ==
    y_actual.values
)

challenger_correct = (
    challenger_pred
    ==
    y_actual.values
)


print("\n========================================")
print("CORRECT PREDICTIONS")
print("========================================")

print(
    "Champion:",
    int(champion_correct.sum())
)

print(
    "Ablation:",
    int(ablation_correct.sum())
)

print(
    "Challenger:",
    int(challenger_correct.sum())
)


# -------------------------------------------------
# Ablation vs Champion
# -------------------------------------------------

ablation_improvements = (
    (~champion_correct)
    &
    ablation_correct
)

ablation_regressions = (
    champion_correct
    &
    (~ablation_correct)
)


print("\n========================================")
print("ABLATION vs CHAMPION")
print("========================================")

print(
    "Ablation improvements:",
    int(ablation_improvements.sum())
)

print(
    "Ablation regressions:",
    int(ablation_regressions.sum())
)

print(
    "Net:",
    int(
        ablation_improvements.sum()
        -
        ablation_regressions.sum()
    )
)


# -------------------------------------------------
# Challenger vs Ablation
# -------------------------------------------------

challenger_over_ablation = (
    (~ablation_correct)
    &
    challenger_correct
)

challenger_under_ablation = (
    ablation_correct
    &
    (~challenger_correct)
)


print("\n========================================")
print("10 FEATURES vs 9 FEATURES")
print("========================================")

print(
    "10-feature improvements:",
    int(
        challenger_over_ablation.sum()
    )
)

print(
    "10-feature regressions:",
    int(
        challenger_under_ablation.sum()
    )
)

print(
    "Net effect of 10-feature model:",
    int(
        challenger_over_ablation.sum()
        -
        challenger_under_ablation.sum()
    )
)


# -------------------------------------------------
# Save detailed results
# -------------------------------------------------

result = pd.DataFrame({

    "date":
        champion_test["date"].values,

    "team1":
        champion_test["team1"].values,

    "team2":
        champion_test["team2"].values,

    "winner":
        champion_test["winner"].values,

    "actual_team1_won":
        y_actual.values,

    "champion_prediction":
        champion_pred,

    "ablation_prediction":
        ablation_pred,

    "challenger_prediction":
        challenger_pred,

    "champion_correct":
        champion_correct,

    "ablation_correct":
        ablation_correct,

    "challenger_correct":
        challenger_correct
})


result.to_csv(
    OUTPUT_FILE,
    index=False
)


# -------------------------------------------------
# Final decision
# -------------------------------------------------

print("\n========================================")
print("CURRENT DECISION")
print("========================================")


scores = {
    "Champion v1.0":
        champion_accuracy,

    "Ablation 9-feature":
        ablation_accuracy,

    "Challenger v1.1":
        challenger_accuracy
}


winner_model = max(
    scores,
    key=scores.get
)


print(
    "Best model on common test set:",
    winner_model
)

print(
    "Best accuracy:",
    f"{scores[winner_model] * 100:.2f}%"
)


if winner_model == "Champion v1.0":

    print(
        "Decision: KEEP Champion v1.0."
    )

else:

    print(
        "Decision: Candidate requires further "
        "validation before promotion."
    )


print("\n========================================")
print("RESULT SAVED")
print("========================================")

print(
    "File:",
    OUTPUT_FILE
)

print(
    "\nNo production model was modified."
)