import pandas as pd

from scipy.stats import ks_2samp


# -------------------------------------------------
# Load feature dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Continuous features for KS test
# -------------------------------------------------

features = [
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate"
]


# -------------------------------------------------
# Historical / Recent split
# -------------------------------------------------

split_index = int(len(df) * 0.8)

historical = df.iloc[:split_index]
recent = df.iloc[split_index:]


print("\n================================")
print("STATISTICAL DRIFT ANALYSIS")
print("KS TEST")
print("================================")

print(
    "Historical matches:",
    len(historical)
)

print(
    "Recent matches:",
    len(recent)
)


# -------------------------------------------------
# Significance threshold
# -------------------------------------------------

alpha = 0.05


# -------------------------------------------------
# Run KS test
# -------------------------------------------------

results = []

for feature in features:

    historical_values = (
        historical[feature]
        .dropna()
    )

    recent_values = (
        recent[feature]
        .dropna()
    )

    ks_statistic, p_value = ks_2samp(
        historical_values,
        recent_values
    )


    # -------------------------------------------------
    # Statistical significance
    # -------------------------------------------------

    if p_value < alpha:

        statistical_status = "SIGNIFICANT"

    else:

        statistical_status = "NOT SIGNIFICANT"


    # -------------------------------------------------
    # Practical severity based on KS statistic
    # -------------------------------------------------

    if ks_statistic < 0.05:

        severity = "LOW"

    elif ks_statistic < 0.10:

        severity = "MODERATE"

    else:

        severity = "HIGH"


    # -------------------------------------------------
    # Combined interpretation
    # -------------------------------------------------

    if p_value >= alpha:

        status = "STABLE"

    elif severity == "LOW":

        status = "SIGNIFICANT BUT SMALL"

    elif severity == "MODERATE":

        status = "WATCH"

    else:

        status = "INVESTIGATE"


    results.append({
        "feature": feature,
        "ks_statistic": ks_statistic,
        "p_value": p_value,
        "statistical_status": statistical_status,
        "severity": severity,
        "status": status
    })


# -------------------------------------------------
# Results dataframe
# -------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="ks_statistic",
    ascending=False
)


# -------------------------------------------------
# Display results
# -------------------------------------------------

print(
    "\n--- KS TEST RESULTS ---"
)

display_df = results_df.copy()

display_df[
    "ks_statistic"
] = display_df[
    "ks_statistic"
].round(4)

display_df[
    "p_value"
] = display_df[
    "p_value"
].round(6)

print(
    display_df.to_string(
        index=False
    )
)


# -------------------------------------------------
# Summary Counts
# -------------------------------------------------

significant_count = len(
    results_df[
        results_df["p_value"] < alpha
    ]
)

non_significant_count = len(
    results_df[
        results_df["p_value"] >= alpha
    ]
)

low_count = len(
    results_df[
        results_df["severity"] == "LOW"
    ]
)

moderate_count = len(
    results_df[
        results_df["severity"] == "MODERATE"
    ]
)

high_count = len(
    results_df[
        results_df["severity"] == "HIGH"
    ]
)


# -------------------------------------------------
# Operational Status Counts
# -------------------------------------------------

stable_count = len(
    results_df[
        results_df["status"] == "STABLE"
    ]
)

small_count = len(
    results_df[
        results_df["status"] == "SIGNIFICANT BUT SMALL"
    ]
)

watch_count = len(
    results_df[
        results_df["status"] == "WATCH"
    ]
)

investigate_count = len(
    results_df[
        results_df["status"] == "INVESTIGATE"
    ]
)


# -------------------------------------------------
# Statistical Drift Summary
# -------------------------------------------------

print(
    "\n--- STATISTICAL DRIFT SUMMARY ---"
)

print(
    "Significance threshold:",
    alpha
)

print(
    "Features with significant change:",
    significant_count
)

print(
    "Features without significant change:",
    non_significant_count
)


# -------------------------------------------------
# Practical Severity Summary
# -------------------------------------------------

print(
    "\n--- PRACTICAL SEVERITY SUMMARY ---"
)

print(
    "LOW severity features:",
    low_count
)

print(
    "MODERATE severity features:",
    moderate_count
)

print(
    "HIGH severity features:",
    high_count
)


# -------------------------------------------------
# Operational Interpretation
# -------------------------------------------------

print(
    "\n--- OPERATIONAL INTERPRETATION ---"
)

print(
    "STABLE:",
    stable_count
)

print(
    "SIGNIFICANT BUT SMALL:",
    small_count
)

print(
    "WATCH:",
    watch_count
)

print(
    "INVESTIGATE:",
    investigate_count
)


# -------------------------------------------------
# Overall Status
# -------------------------------------------------

if investigate_count > 0:

    overall_status = "INVESTIGATE"

elif watch_count > 0:

    overall_status = "WATCH"

elif small_count > 0:

    overall_status = "MINOR CHANGE"

else:

    overall_status = "STABLE"


print(
    "\nOverall Statistical Drift Status:",
    overall_status
)


# -------------------------------------------------
# Threshold Explanation
# -------------------------------------------------

print(
    "\n--- KS SEVERITY RULES ---"
)

print(
    "LOW:"
)

print(
    "KS statistic < 0.05"
)

print(
    "\nMODERATE:"
)

print(
    "KS statistic from 0.05 to < 0.10"
)

print(
    "\nHIGH:"
)

print(
    "KS statistic >= 0.10"
)


# -------------------------------------------------
# Important Notes
# -------------------------------------------------

print(
    "\nImportant:"
)

print(
    "A low p-value indicates evidence that "
    "historical and recent distributions differ."
)

print(
    "The KS statistic indicates how large the "
    "distributional difference is."
)

print(
    "Statistical significance does not automatically "
    "mean the model must be retrained."
)

print(
    "The LOW/MODERATE/HIGH KS thresholds used here "
    "are project monitoring rules, not universal "
    "scientific thresholds."
)

print(
    "Retraining decisions should also consider "
    "live model accuracy, completed predictions, "
    "error analysis, and business impact."
)