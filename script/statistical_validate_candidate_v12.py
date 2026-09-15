import pandas as pd
import math
from pathlib import Path

ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

INPUT_FILE = (
    ROOT
    / "data"
    / "three_model_comparison.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "statistical_validation_candidate_v12.csv"
)

print("========================================")
print("STATISTICAL VALIDATION - CANDIDATE v1.2")
print("========================================")

# -------------------------------------------------
# Load common-test comparison
# -------------------------------------------------

df = pd.read_csv(INPUT_FILE)

# Make sure correctness columns are Boolean
for col in [
    "champion_correct",
    "ablation_correct"
]:
    if df[col].dtype != bool:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False
            })
        )

if df[
    [
        "champion_correct",
        "ablation_correct"
    ]
].isna().any().any():

    raise ValueError(
        "Could not safely parse correctness columns."
    )

total = len(df)

# -------------------------------------------------
# Accuracy
# -------------------------------------------------

champion_correct_total = int(
    df["champion_correct"].sum()
)

candidate_correct_total = int(
    df["ablation_correct"].sum()
)

champion_accuracy = (
    champion_correct_total / total
)

candidate_accuracy = (
    candidate_correct_total / total
)

print("\nTotal common matches:", total)

print(
    "Champion correct:",
    champion_correct_total
)

print(
    "Candidate correct:",
    candidate_correct_total
)

print(
    "Champion accuracy:",
    f"{champion_accuracy * 100:.2f}%"
)

print(
    "Candidate accuracy:",
    f"{candidate_accuracy * 100:.2f}%"
)

print(
    "Difference:",
    f"{(candidate_accuracy - champion_accuracy) * 100:+.2f}",
    "percentage points"
)

# -------------------------------------------------
# McNemar disagreement table
# -------------------------------------------------

both_correct = int(
    (
        df["champion_correct"]
        &
        df["ablation_correct"]
    ).sum()
)

both_wrong = int(
    (
        (~df["champion_correct"])
        &
        (~df["ablation_correct"])
    ).sum()
)

# Champion wrong, Candidate correct
candidate_improvements = int(
    (
        (~df["champion_correct"])
        &
        df["ablation_correct"]
    ).sum()
)

# Champion correct, Candidate wrong
candidate_regressions = int(
    (
        df["champion_correct"]
        &
        (~df["ablation_correct"])
    ).sum()
)

print("\n========================================")
print("PAIRED OUTCOME TABLE")
print("========================================")

print(
    "Both correct:",
    both_correct
)

print(
    "Both wrong:",
    both_wrong
)

print(
    "Candidate improvement:",
    candidate_improvements
)

print(
    "Candidate regression:",
    candidate_regressions
)

print(
    "Net candidate gain:",
    candidate_improvements
    -
    candidate_regressions
)

# -------------------------------------------------
# Exact McNemar test
#
# Under H0:
# among disagreement cases, either model should
# be correct with probability 0.5.
#
# Exact two-sided binomial test.
# -------------------------------------------------

n = (
    candidate_improvements
    +
    candidate_regressions
)

k = min(
    candidate_improvements,
    candidate_regressions
)

if n == 0:

    p_value = 1.0

else:

    lower_tail = sum(
        math.comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    p_value = min(
        1.0,
        2 * lower_tail
    )

print("\n========================================")
print("MCNEMAR EXACT TEST")
print("========================================")

print(
    "Discordant predictions:",
    n
)

print(
    "Candidate wins among disagreements:",
    candidate_improvements
)

print(
    "Champion wins among disagreements:",
    candidate_regressions
)

print(
    "Exact two-sided p-value:",
    f"{p_value:.6f}"
)

# -------------------------------------------------
# Interpretation
# -------------------------------------------------

alpha = 0.05

print("\n========================================")
print("INTERPRETATION")
print("========================================")

print(
    "Significance threshold (alpha):",
    alpha
)

if p_value < alpha:

    print(
        "Result: STATISTICALLY SIGNIFICANT."
    )

    print(
        "Evidence suggests the two models "
        "have different error rates."
    )

else:

    print(
        "Result: NOT STATISTICALLY SIGNIFICANT."
    )

    print(
        "The observed difference could "
        "reasonably occur by chance."
    )

# -------------------------------------------------
# Promotion decision
# -------------------------------------------------

print("\n========================================")
print("PROMOTION ASSESSMENT")
print("========================================")

if (
    candidate_accuracy > champion_accuracy
    and
    p_value < alpha
):

    status = (
        "STRONG CANDIDATE FOR FURTHER "
        "PROMOTION VALIDATION"
    )

    print(
        "Candidate beats Champion and the "
        "paired difference is statistically significant."
    )

elif candidate_accuracy > champion_accuracy:

    status = (
        "KEEP AS SHADOW CANDIDATE"
    )

    print(
        "Candidate accuracy is higher, but the "
        "improvement is not statistically significant."
    )

    print(
        "Do NOT replace Champion based on "
        "this test alone."
    )

else:

    status = "KEEP CHAMPION"

    print(
        "Candidate does not outperform Champion."
    )

# -------------------------------------------------
# Save summary
# -------------------------------------------------

summary = pd.DataFrame([{

    "total_matches":
        total,

    "champion_correct":
        champion_correct_total,

    "candidate_correct":
        candidate_correct_total,

    "champion_accuracy":
        champion_accuracy,

    "candidate_accuracy":
        candidate_accuracy,

    "difference_percentage_points":
        (
            candidate_accuracy
            -
            champion_accuracy
        ) * 100,

    "both_correct":
        both_correct,

    "both_wrong":
        both_wrong,

    "candidate_improvements":
        candidate_improvements,

    "candidate_regressions":
        candidate_regressions,

    "discordant_predictions":
        n,

    "mcnemar_exact_p_value":
        p_value,

    "alpha":
        alpha,

    "statistically_significant":
        p_value < alpha,

    "status":
        status
}])

summary.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========================================")
print("RESULT SAVED")
print("========================================")

print(
    "File:",
    OUTPUT_FILE
)

print(
    "\nChampion v1.0 was NOT modified."
)

print(
    "Candidate v1.2 was NOT promoted."
)