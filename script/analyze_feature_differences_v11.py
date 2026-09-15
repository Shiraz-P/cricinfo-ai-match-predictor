import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"K:\Python\Cricinfo_AI_Project")

CHAMPION_FILE = (
    PROJECT_ROOT / "data" / "matches_features.csv"
)

CHALLENGER_FILE = (
    PROJECT_ROOT / "data" / "matches_candidate_features.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT / "data" / "champion_challenger_v11_comparison.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "feature_difference_analysis_v11.csv"
)

# -------------------------------------------------
# Features common to both models
# -------------------------------------------------

COMMON_FEATURES = [
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


print("========================================")
print("FEATURE DIFFERENCE ANALYSIS v1.1")
print("========================================")


# -------------------------------------------------
# Load data
# -------------------------------------------------

champion = pd.read_csv(
    CHAMPION_FILE
)

challenger = pd.read_csv(
    CHALLENGER_FILE
)

comparison = pd.read_csv(
    COMPARISON_FILE
)


for df in [
    champion,
    challenger,
    comparison
]:

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )


# -------------------------------------------------
# Keep only disagreement matches
# -------------------------------------------------

disagreements = comparison[
    comparison["models_disagree"] == True
].copy()


print(
    "\nTotal comparison matches:",
    len(comparison)
)

print(
    "Disagreement matches:",
    len(disagreements)
)


# -------------------------------------------------
# Create exact historical record key
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


champion = create_key(champion)

challenger = create_key(challenger)

disagreements = create_key(disagreements)


# -------------------------------------------------
# Build lookup tables
# -------------------------------------------------

champion_lookup = (
    champion
    .set_index("_key")
)

challenger_lookup = (
    challenger
    .set_index("_key")
)


# -------------------------------------------------
# Analyze each disagreement
# -------------------------------------------------

rows = []

missing = 0


for _, match in disagreements.iterrows():

    key = match["_key"]

    if (
        key not in champion_lookup.index
        or
        key not in challenger_lookup.index
    ):

        missing += 1
        continue


    c = champion_lookup.loc[key]

    h = challenger_lookup.loc[key]


    # Safety if duplicate index exists
    if isinstance(c, pd.DataFrame):

        c = c.iloc[0]


    if isinstance(h, pd.DataFrame):

        h = h.iloc[0]


    row = {

        "date":
            match["date"],

        "team1":
            match["team1"],

        "team2":
            match["team2"],

        "winner":
            match["winner"],

        "champion_prediction":
            match["champion_prediction"],

        "challenger_prediction":
            match["challenger_prediction"],

        "champion_correct":
            match["champion_correct"],

        "challenger_correct":
            match["challenger_correct"]
    }


    # ---------------------------------------------
    # Category
    # ---------------------------------------------

    if (
        match["challenger_correct"]
        and
        not match["champion_correct"]
    ):

        row["category"] = (
            "CHALLENGER_IMPROVEMENT"
        )

    elif (
        match["champion_correct"]
        and
        not match["challenger_correct"]
    ):

        row["category"] = (
            "CHALLENGER_REGRESSION"
        )

    else:

        row["category"] = "OTHER"


    # ---------------------------------------------
    # Compare common features
    # ---------------------------------------------

    changed_features = []

    total_abs_change = 0.0


    for feature in COMMON_FEATURES:

        champion_value = float(
            c[feature]
        )

        challenger_value = float(
            h[feature]
        )

        difference = (
            challenger_value
            -
            champion_value
        )


        row[
            f"champion_{feature}"
        ] = champion_value


        row[
            f"challenger_{feature}"
        ] = challenger_value


        row[
            f"diff_{feature}"
        ] = difference


        if abs(difference) > 0.000001:

            changed_features.append(
                feature
            )

            total_abs_change += abs(
                difference
            )


    # ---------------------------------------------
    # Challenger-only feature
    # ---------------------------------------------

    row["challenger_toss_data_known"] = (
        h["toss_data_known"]
    )


    row["changed_feature_count"] = (
        len(changed_features)
    )


    row["changed_features"] = (
        ", ".join(
            changed_features
        )
        if changed_features
        else "NONE"
    )


    row["total_absolute_feature_change"] = (
        total_abs_change
    )


    rows.append(row)


# -------------------------------------------------
# Create analysis dataframe
# -------------------------------------------------

result = pd.DataFrame(rows)


print(
    "\nSuccessfully analyzed:",
    len(result)
)

print(
    "Missing disagreement records:",
    missing
)


# -------------------------------------------------
# Count feature changes
# -------------------------------------------------

print("\n========================================")
print("HOW OFTEN EACH FEATURE CHANGED")
print("========================================")


feature_change_summary = []


for feature in COMMON_FEATURES:

    diff_column = (
        f"diff_{feature}"
    )

    changed = (
        result[
            diff_column
        ].abs()
        >
        0.000001
    )


    count = int(
        changed.sum()
    )


    mean_change = (
        result.loc[
            changed,
            diff_column
        ].abs().mean()
        if count > 0
        else 0
    )


    feature_change_summary.append({

        "feature":
            feature,

        "matches_changed":
            count,

        "mean_absolute_change":
            mean_change
    })


summary_df = pd.DataFrame(
    feature_change_summary
).sort_values(
    "matches_changed",
    ascending=False
)


print(
    summary_df.to_string(
        index=False
    )
)


# -------------------------------------------------
# Improvements vs regressions
# -------------------------------------------------

print("\n========================================")
print("FEATURE CHANGES: IMPROVEMENTS")
print("========================================")


improvements = result[
    result["category"]
    ==
    "CHALLENGER_IMPROVEMENT"
]


for feature in COMMON_FEATURES:

    count = int(
        (
            improvements[
                f"diff_{feature}"
            ].abs()
            >
            0.000001
        ).sum()
    )

    if count > 0:

        print(
            f"{feature}: {count}"
        )


print("\n========================================")
print("FEATURE CHANGES: REGRESSIONS")
print("========================================")


regressions = result[
    result["category"]
    ==
    "CHALLENGER_REGRESSION"
]


for feature in COMMON_FEATURES:

    count = int(
        (
            regressions[
                f"diff_{feature}"
            ].abs()
            >
            0.000001
        ).sum()
    )

    if count > 0:

        print(
            f"{feature}: {count}"
        )


# -------------------------------------------------
# Toss-data-known analysis
# -------------------------------------------------

print("\n========================================")
print("TOSS DATA KNOWN")
print("========================================")


print(
    result[
        "challenger_toss_data_known"
    ].value_counts(
        dropna=False
    )
)


# -------------------------------------------------
# Matches with largest feature movement
# -------------------------------------------------

print("\n========================================")
print("LARGEST FEATURE MOVEMENTS")
print("========================================")


display_columns = [
    "date",
    "team1",
    "team2",
    "winner",
    "category",
    "changed_feature_count",
    "total_absolute_feature_change",
    "changed_features"
]


print(
    result[
        display_columns
    ]
    .sort_values(
        "total_absolute_feature_change",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# -------------------------------------------------
# Special check:
# disagreements where NO common feature changed
#
# These are especially interesting because the
# model itself / toss_data_known may explain flip.
# -------------------------------------------------

no_common_change = result[
    result[
        "changed_feature_count"
    ] == 0
]


print("\n========================================")
print("NO COMMON FEATURE CHANGED")
print("========================================")


print(
    "Matches:",
    len(no_common_change)
)


if len(no_common_change) > 0:

    print(
        no_common_change[
            [
                "date",
                "team1",
                "team2",
                "winner",
                "category",
                "challenger_toss_data_known"
            ]
        ]
        .to_string(
            index=False
        )
    )


# -------------------------------------------------
# Save
# -------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n========================================")
print("ANALYSIS SAVED")
print("========================================")


print(
    "Rows:",
    len(result)
)

print(
    "File:",
    OUTPUT_FILE
)

print(
    "\nNo model was modified."
)