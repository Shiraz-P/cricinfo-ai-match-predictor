import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss
)


# -------------------------------------------------
# Load base feature dataset
# -------------------------------------------------

feature_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\matches_features.csv"
)

df = pd.read_csv(feature_file)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Load reliability analysis dataset
# -------------------------------------------------

reliability_file = (
    r"K:\Python\Cricinfo_AI_Project\data"
    r"\feature_reliability_analysis.csv"
)

reliability_df = pd.read_csv(
    reliability_file
)

reliability_df["date"] = pd.to_datetime(
    reliability_df["date"]
)

reliability_df = reliability_df.sort_values(
    "date"
).reset_index(
    drop=True
)


# -------------------------------------------------
# Safety check
# -------------------------------------------------

if len(df) != len(reliability_df):

    raise ValueError(
        "Dataset row counts do not match."
    )


# -------------------------------------------------
# Original 9 features
# -------------------------------------------------

baseline_features = [
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
# Add sample-size columns
# -------------------------------------------------

df[
    "team1_historical_sample_size"
] = reliability_df[
    "team1_historical_sample_size"
]

df[
    "team2_historical_sample_size"
] = reliability_df[
    "team2_historical_sample_size"
]

df[
    "h2h_sample_size"
] = reliability_df[
    "team1_h2h_sample_size"
]

df[
    "team1_venue_sample_size"
] = reliability_df[
    "team1_venue_sample_size"
]

df[
    "team2_venue_sample_size"
] = reliability_df[
    "team2_venue_sample_size"
]


# -------------------------------------------------
# Log-transform reliability counts
# -------------------------------------------------

df[
    "team1_historical_sample_log"
] = np.log1p(
    df[
        "team1_historical_sample_size"
    ]
)

df[
    "team2_historical_sample_log"
] = np.log1p(
    df[
        "team2_historical_sample_size"
    ]
)

df[
    "h2h_sample_log"
] = np.log1p(
    df[
        "h2h_sample_size"
    ]
)

df[
    "team1_venue_sample_log"
] = np.log1p(
    df[
        "team1_venue_sample_size"
    ]
)

df[
    "team2_venue_sample_log"
] = np.log1p(
    df[
        "team2_venue_sample_size"
    ]
)


# -------------------------------------------------
# Reliability feature groups
# -------------------------------------------------

historical_reliability = [
    "team1_historical_sample_log",
    "team2_historical_sample_log"
]

h2h_reliability = [
    "h2h_sample_log"
]

venue_reliability = [
    "team1_venue_sample_log",
    "team2_venue_sample_log"
]

all_reliability = (
    historical_reliability
    +
    h2h_reliability
    +
    venue_reliability
)


# -------------------------------------------------
# Candidate feature sets
# -------------------------------------------------

feature_sets = {
    "Baseline 9":
        baseline_features,

    "Baseline + Historical Reliability":
        baseline_features
        +
        historical_reliability,

    "Baseline + H2H Reliability":
        baseline_features
        +
        h2h_reliability,

    "Baseline + Venue Reliability":
        baseline_features
        +
        venue_reliability,

    "Baseline + All Reliability":
        baseline_features
        +
        all_reliability
}


# -------------------------------------------------
# Target
# -------------------------------------------------

y = df[
    "team1_won"
]


# -------------------------------------------------
# Chronological 80 / 20 split
# -------------------------------------------------

split_index = int(
    len(df) * 0.8
)

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print("\n================================")
print("RELIABILITY FEATURE ABLATION")
print("================================")

print(
    "Total matches:",
    len(df)
)

print(
    "Training matches:",
    len(y_train)
)

print(
    "Testing matches:",
    len(y_test)
)

print(
    "Training period:",
    df.iloc[
        :split_index
    ]["date"].min(),
    "to",
    df.iloc[
        :split_index
    ]["date"].max()
)

print(
    "Testing period:",
    df.iloc[
        split_index:
    ]["date"].min(),
    "to",
    df.iloc[
        split_index:
    ]["date"].max()
)


# -------------------------------------------------
# Train and evaluate every candidate
# -------------------------------------------------

results = []

for model_name, features in feature_sets.items():

    X = df[
        features
    ]

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]


    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )


    results.append({
        "model":
            model_name,

        "feature_count":
            len(features),

        "accuracy":
            accuracy,

        "brier_score":
            brier
    })


# -------------------------------------------------
# Results dataframe
# -------------------------------------------------

results_df = pd.DataFrame(
    results
)


# -------------------------------------------------
# Baseline values
# -------------------------------------------------

baseline_row = results_df[
    results_df["model"]
    ==
    "Baseline 9"
].iloc[0]

baseline_accuracy = baseline_row[
    "accuracy"
]

baseline_brier = baseline_row[
    "brier_score"
]


# -------------------------------------------------
# Differences vs baseline
# -------------------------------------------------

results_df[
    "accuracy_change_pp"
] = (
    results_df[
        "accuracy"
    ]
    -
    baseline_accuracy
) * 100


results_df[
    "brier_improvement"
] = (
    baseline_brier
    -
    results_df[
        "brier_score"
    ]
)


# -------------------------------------------------
# Display formatting
# -------------------------------------------------

display_df = results_df.copy()

display_df[
    "accuracy"
] = (
    display_df[
        "accuracy"
    ]
    * 100
).round(2)

display_df[
    "brier_score"
] = display_df[
    "brier_score"
].round(4)

display_df[
    "accuracy_change_pp"
] = display_df[
    "accuracy_change_pp"
].round(2)

display_df[
    "brier_improvement"
] = display_df[
    "brier_improvement"
].round(4)


# -------------------------------------------------
# Print comparison
# -------------------------------------------------

print(
    "\n--- ABLATION RESULTS ---"
)

print(
    display_df[
        [
            "model",
            "feature_count",
            "accuracy",
            "accuracy_change_pp",
            "brier_score",
            "brier_improvement"
        ]
    ].to_string(
        index=False
    )
)


# -------------------------------------------------
# Best model by accuracy
# -------------------------------------------------

best_accuracy_row = results_df.loc[
    results_df[
        "accuracy"
    ].idxmax()
]


print(
    "\n--- BEST MODEL BY ACCURACY ---"
)

print(
    "Model:",
    best_accuracy_row[
        "model"
    ]
)

print(
    "Accuracy:",
    round(
        best_accuracy_row[
            "accuracy"
        ]
        * 100,
        2
    ),
    "%"
)

print(
    "Brier Score:",
    round(
        best_accuracy_row[
            "brier_score"
        ],
        4
    )
)


# -------------------------------------------------
# Best model by Brier Score
# -------------------------------------------------

best_brier_row = results_df.loc[
    results_df[
        "brier_score"
    ].idxmin()
]


print(
    "\n--- BEST MODEL BY BRIER SCORE ---"
)

print(
    "Model:",
    best_brier_row[
        "model"
    ]
)

print(
    "Accuracy:",
    round(
        best_brier_row[
            "accuracy"
        ]
        * 100,
        2
    ),
    "%"
)

print(
    "Brier Score:",
    round(
        best_brier_row[
            "brier_score"
        ],
        4
    )
)


# -------------------------------------------------
# Interpretation
# -------------------------------------------------

print(
    "\n--- INTERPRETATION ---"
)

print(
    "Ablation testing shows which reliability "
    "feature groups add useful information."
)

print(
    "Positive accuracy_change_pp means the "
    "candidate improved accuracy over baseline."
)

print(
    "Positive brier_improvement means the "
    "candidate improved probability quality."
)

print(
    "A candidate should not be promoted simply "
    "because it improves one metric."
)

print(
    "We should prefer feature groups that improve "
    "generalization consistently."
)


# -------------------------------------------------
# Important
# -------------------------------------------------

print(
    "\nImportant:"
)

print(
    "This experiment does not change "
    "cricket_model.pkl."
)

print(
    "The current production model remains "
    "the 9-feature Logistic Regression."
)