import pandas as pd


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Containers for historical sample counts
# -------------------------------------------------

team_match_count = {}

h2h_match_count = {}

venue_match_count = {}


team1_history_counts = []
team2_history_counts = []

team1_h2h_counts = []
team2_h2h_counts = []

team1_venue_counts = []
team2_venue_counts = []


# -------------------------------------------------
# Process matches chronologically
# -------------------------------------------------

for _, row in df.iterrows():

    team1 = row["team1"]
    team2 = row["team2"]
    venue = row["venue"]

    # ---------------------------------------------
    # Overall historical match counts
    # BEFORE current match
    # ---------------------------------------------

    team1_count = team_match_count.get(
        team1,
        0
    )

    team2_count = team_match_count.get(
        team2,
        0
    )

    team1_history_counts.append(
        team1_count
    )

    team2_history_counts.append(
        team2_count
    )


    # ---------------------------------------------
    # Head-to-head count
    # Use sorted pair so Team A vs Team B
    # equals Team B vs Team A
    # ---------------------------------------------

    h2h_key = tuple(
        sorted(
            [
                team1,
                team2
            ]
        )
    )

    h2h_count = h2h_match_count.get(
        h2h_key,
        0
    )

    team1_h2h_counts.append(
        h2h_count
    )

    team2_h2h_counts.append(
        h2h_count
    )


    # ---------------------------------------------
    # Venue sample count
    # ---------------------------------------------

    team1_venue_key = (
        team1,
        venue
    )

    team2_venue_key = (
        team2,
        venue
    )

    team1_venue_count = venue_match_count.get(
        team1_venue_key,
        0
    )

    team2_venue_count = venue_match_count.get(
        team2_venue_key,
        0
    )

    team1_venue_counts.append(
        team1_venue_count
    )

    team2_venue_counts.append(
        team2_venue_count
    )


    # ---------------------------------------------
    # Update counts AFTER feature collection
    # Prevent future-data leakage
    # ---------------------------------------------

    team_match_count[
        team1
    ] = team1_count + 1

    team_match_count[
        team2
    ] = team2_count + 1

    h2h_match_count[
        h2h_key
    ] = h2h_count + 1

    venue_match_count[
        team1_venue_key
    ] = team1_venue_count + 1

    venue_match_count[
        team2_venue_key
    ] = team2_venue_count + 1


# -------------------------------------------------
# Add reliability count columns
# -------------------------------------------------

df[
    "team1_historical_sample_size"
] = team1_history_counts

df[
    "team2_historical_sample_size"
] = team2_history_counts

df[
    "team1_h2h_sample_size"
] = team1_h2h_counts

df[
    "team2_h2h_sample_size"
] = team2_h2h_counts

df[
    "team1_venue_sample_size"
] = team1_venue_counts

df[
    "team2_venue_sample_size"
] = team2_venue_counts


# -------------------------------------------------
# Reliability Classification
# -------------------------------------------------

def reliability_level(sample_size):

    if sample_size >= 20:
        return "HIGH"

    elif sample_size >= 5:
        return "MEDIUM"

    else:
        return "LOW"


df[
    "team1_h2h_reliability"
] = df[
    "team1_h2h_sample_size"
].apply(
    reliability_level
)

df[
    "team2_h2h_reliability"
] = df[
    "team2_h2h_sample_size"
].apply(
    reliability_level
)

df[
    "team1_venue_reliability"
] = df[
    "team1_venue_sample_size"
].apply(
    reliability_level
)

df[
    "team2_venue_reliability"
] = df[
    "team2_venue_sample_size"
].apply(
    reliability_level
)


# -------------------------------------------------
# Overall Summary
# -------------------------------------------------

print("\n================================")
print("FEATURE RELIABILITY ANALYSIS")
print("================================")

print(
    "Total matches:",
    len(df)
)


# -------------------------------------------------
# Sample Size Summary
# -------------------------------------------------

sample_columns = [
    "team1_historical_sample_size",
    "team2_historical_sample_size",
    "team1_h2h_sample_size",
    "team2_h2h_sample_size",
    "team1_venue_sample_size",
    "team2_venue_sample_size"
]

print(
    "\n--- SAMPLE SIZE SUMMARY ---"
)

print(
    df[
        sample_columns
    ]
    .describe()
    .round(2)
    .to_string()
)


# -------------------------------------------------
# H2H Reliability
# -------------------------------------------------

print(
    "\n--- TEAM 1 H2H RELIABILITY ---"
)

print(
    df[
        "team1_h2h_reliability"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n--- TEAM 2 H2H RELIABILITY ---"
)

print(
    df[
        "team2_h2h_reliability"
    ]
    .value_counts()
    .to_string()
)


# -------------------------------------------------
# Venue Reliability
# -------------------------------------------------

print(
    "\n--- TEAM 1 VENUE RELIABILITY ---"
)

print(
    df[
        "team1_venue_reliability"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n--- TEAM 2 VENUE RELIABILITY ---"
)

print(
    df[
        "team2_venue_reliability"
    ]
    .value_counts()
    .to_string()
)


# -------------------------------------------------
# Extreme Venue Rates With Small Samples
# -------------------------------------------------

print(
    "\n--- EXTREME VENUE RATES WITH LOW SAMPLE SIZE ---"
)

extreme_venue = df[
    (
        (
            df["team1_venue_win_rate"]
            >= 0.90
        )
        |
        (
            df["team1_venue_win_rate"]
            <= 0.10
        )
    )
    &
    (
        df["team1_venue_sample_size"]
        < 5
    )
]

print(
    extreme_venue[
        [
            "date",
            "team1",
            "venue",
            "team1_venue_win_rate",
            "team1_venue_sample_size"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# -------------------------------------------------
# Extreme H2H Rates With Small Samples
# -------------------------------------------------

print(
    "\n--- EXTREME H2H RATES WITH LOW SAMPLE SIZE ---"
)

extreme_h2h = df[
    (
        (
            df["team1_h2h_win_rate"]
            >= 0.90
        )
        |
        (
            df["team1_h2h_win_rate"]
            <= 0.10
        )
    )
    &
    (
        df["team1_h2h_sample_size"]
        < 5
    )
]

print(
    extreme_h2h[
        [
            "date",
            "team1",
            "team2",
            "team1_h2h_win_rate",
            "team1_h2h_sample_size"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# -------------------------------------------------
# Save Reliability Dataset
# -------------------------------------------------

output_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\feature_reliability_analysis.csv"
)

df.to_csv(
    output_file,
    index=False
)


print(
    "\nFeature reliability analysis saved to:"
)

print(
    output_file
)