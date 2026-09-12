import pandas as pd


# -------------------------------------------------
# Load feature dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Features to monitor
# -------------------------------------------------

features = [
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


# -------------------------------------------------
# Split historical vs recent data
# Same chronological 80/20 idea
# -------------------------------------------------

split_index = int(len(df) * 0.8)

historical = df.iloc[:split_index]
recent = df.iloc[split_index:]


print("\n================================")
print("FEATURE DRIFT ANALYSIS")
print("================================")

print("Historical matches:", len(historical))
print("Recent matches:", len(recent))

print(
    "Historical period:",
    historical["date"].min(),
    "to",
    historical["date"].max()
)

print(
    "Recent period:",
    recent["date"].min(),
    "to",
    recent["date"].max()
)


# -------------------------------------------------
# Compare feature averages
# -------------------------------------------------

results = []

for feature in features:

    historical_mean = historical[
        feature
    ].mean()

    recent_mean = recent[
        feature
    ].mean()

    difference = (
        recent_mean
        -
        historical_mean
    )

    absolute_difference = abs(
        difference
    )


    # ---------------------------------------------
    # Drift Severity Classification
    # ---------------------------------------------

    if absolute_difference >= 0.10:

        severity = "HIGH / POSSIBLE DRIFT"

    elif absolute_difference >= 0.05:

        severity = "MODERATE / WATCH"

    else:

        severity = "LOW / STABLE"


    results.append({
        "feature": feature,
        "historical_mean": historical_mean,
        "recent_mean": recent_mean,
        "difference": difference,
        "absolute_difference": absolute_difference,
        "severity": severity
    })


drift_df = pd.DataFrame(
    results
)


# -------------------------------------------------
# Sort strongest change first
# -------------------------------------------------

drift_df = drift_df.sort_values(
    by="absolute_difference",
    ascending=False
)


# -------------------------------------------------
# Display results
# -------------------------------------------------

print(
    "\n--- FEATURE DISTRIBUTION CHANGE ---"
)

display_df = drift_df.copy()

display_df[
    "historical_mean"
] = (
    display_df["historical_mean"]
    * 100
).round(2)

display_df[
    "recent_mean"
] = (
    display_df["recent_mean"]
    * 100
).round(2)

display_df[
    "difference"
] = (
    display_df["difference"]
    * 100
).round(2)

display_df[
    "absolute_difference"
] = (
    display_df["absolute_difference"]
    * 100
).round(2)

print(
    display_df.to_string(
        index=False
    )
)


# -------------------------------------------------
# Drift Severity Summary
# -------------------------------------------------

print(
    "\n--- DRIFT SEVERITY ---"
)

for _, row in drift_df.iterrows():

    print(
        row["feature"],
        ":",
        round(
            row["absolute_difference"] * 100,
            2
        ),
        "percentage points",
        "->",
        row["severity"]
    )


# -------------------------------------------------
# Count severity levels
# -------------------------------------------------

low_count = len(
    drift_df[
        drift_df["severity"]
        ==
        "LOW / STABLE"
    ]
)

moderate_count = len(
    drift_df[
        drift_df["severity"]
        ==
        "MODERATE / WATCH"
    ]
)

high_count = len(
    drift_df[
        drift_df["severity"]
        ==
        "HIGH / POSSIBLE DRIFT"
    ]
)


# -------------------------------------------------
# Overall Drift Status
# -------------------------------------------------

print(
    "\n--- OVERALL DRIFT STATUS ---"
)

print(
    "LOW / STABLE features:",
    low_count
)

print(
    "MODERATE / WATCH features:",
    moderate_count
)

print(
    "HIGH / POSSIBLE DRIFT features:",
    high_count
)


if high_count > 0:

    overall_status = "HIGH / INVESTIGATE"

elif moderate_count > 0:

    overall_status = "MODERATE / WATCH"

else:

    overall_status = "LOW / STABLE"


print(
    "Overall Status:",
    overall_status
)


# -------------------------------------------------
# Summary
# -------------------------------------------------

print(
    "\n--- DRIFT SUMMARY ---"
)

print(
    "LOW / STABLE:"
)

print(
    "Less than 5 percentage points"
)

print(
    "\nMODERATE / WATCH:"
)

print(
    "5 to less than 10 percentage points"
)

print(
    "\nHIGH / POSSIBLE DRIFT:"
)

print(
    "10 percentage points or more"
)

print(
    "\nImportant:"
)

print(
    "This is still a simple mean-based drift check."
)

print(
    "It compares feature averages between "
    "historical and recent periods."
)

print(
    "It does not compare the complete "
    "feature distributions."
)

print(
    "It also does not prove or rule out "
    "concept drift."
)