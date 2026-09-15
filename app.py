# ================================================================
# CRICINFO AI MATCH PREDICTOR
# Streamlit Educational AI / ML Application
#
# Author:
# Syed Shiraz Hussain
#
# Model Governance:
# Champion v1.0    -> PRODUCTION
# Challenger v1.1 -> REJECTED
# Candidate v1.2  -> SHADOW
#
# AI:
# Predictive AI
# Supervised Machine Learning
# Binary Classification
# Logistic Regression
# ================================================================

from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Cricinfo AI Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# PATHS
# ================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "model"

CHAMPION_FEATURE_FILE = DATA_DIR / "matches_features.csv"

CANDIDATE_FEATURE_FILE = (
    DATA_DIR / "matches_candidate_features.csv"
)

CHAMPION_MODEL_FILE = (
    MODEL_DIR / "cricket_model.pkl"
)

CANDIDATE_MODEL_FILE = (
    MODEL_DIR / "cricket_model_ablation_9feature.pkl"
)

PRODUCTION_LOG_FILE = (
    DATA_DIR / "prediction_log.csv"
)

SHADOW_LOG_FILE = (
    DATA_DIR / "shadow_v12_log.csv"
)


# ================================================================
# MODEL FEATURES
# ================================================================

MODEL_FEATURES = [
    "toss_winner_is_team1",
    "team1_historical_win_rate",
    "team2_historical_win_rate",
    "team1_recent_win_rate",
    "team2_recent_win_rate",
    "team1_h2h_win_rate",
    "team2_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
]


# ================================================================
# LOAD DATA
# ================================================================

@st.cache_data
def load_dataset(file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = (
        df
        .sort_values(
            "date",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return df


# ================================================================
# LOAD MODEL
# ================================================================

@st.cache_resource
def load_model_package(model_file):

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    return joblib.load(model_file)


# ================================================================
# MODEL HELPERS
# ================================================================

def extract_model(package):

    if isinstance(package, dict):

        if "model" in package:
            return package["model"]

    return package


def extract_features(package):

    if isinstance(package, dict):

        if "features" in package:
            return list(
                package["features"]
            )

    return MODEL_FEATURES.copy()


# ================================================================
# LOAD PROJECT
# ================================================================

try:

    champion_history = load_dataset(
        CHAMPION_FEATURE_FILE
    )

    candidate_history = load_dataset(
        CANDIDATE_FEATURE_FILE
    )

    champion_package = load_model_package(
        CHAMPION_MODEL_FILE
    )

    candidate_package = load_model_package(
        CANDIDATE_MODEL_FILE
    )

    champion_model = extract_model(
        champion_package
    )

    candidate_model = extract_model(
        candidate_package
    )

    champion_features = extract_features(
        champion_package
    )

    candidate_features = extract_features(
        candidate_package
    )

except Exception as error:

    st.error(
        "Project resources could not be loaded."
    )

    st.exception(error)

    st.stop()


# ================================================================
# FEATURE SCHEMA VALIDATION
# ================================================================

if champion_features != MODEL_FEATURES:

    st.error(
        "Champion model feature schema mismatch."
    )

    st.write(champion_features)

    st.stop()


if candidate_features != MODEL_FEATURES:

    st.error(
        "Candidate v1.2 feature schema mismatch."
    )

    st.write(candidate_features)

    st.stop()


# ================================================================
# BUILD TEAM LIST
# ================================================================

def build_team_list(history):

    teams = set(
        history["team1"]
        .dropna()
        .astype(str)
    )

    teams.update(
        history["team2"]
        .dropna()
        .astype(str)
    )

    return sorted(teams)


champion_teams = build_team_list(
    champion_history
)

candidate_teams = build_team_list(
    candidate_history
)


# ================================================================
# VENUE VALIDATION
# ================================================================

def is_valid_venue_string(venue):

    text = str(venue).strip()
    lower = text.lower()

    if not text:
        return False

    if len(text) > 100:
        return False

    if lower.startswith("v "):
        return False

    if lower.startswith("/"):
        return False

    bad_phrases = [
        "t20i series",
        "1st t20i-",
        "2nd t20i-",
        "3rd t20i-",
        "4th t20i-",
        "5th t20i-",
        "world cup /",
        "match, group",
        "icc men's t20 world cup",
    ]

    for phrase in bad_phrases:

        if phrase in lower:
            return False

    return True


# ================================================================
# CLEAN VENUE LIST
# ================================================================

def build_venue_list(history):

    venues = (
        history["venue"]
        .dropna()
        .astype(str)
        .unique()
    )

    clean = [

        venue.strip()

        for venue in venues

        if is_valid_venue_string(
            venue
        )
    ]

    return sorted(
        set(clean)
    )


# ================================================================
# MODEL ROUTER
# ================================================================

def select_model(
    team1,
    team2,
):

    # Both teams covered by original production history.
    if (
        team1 in champion_teams
        and
        team2 in champion_teams
    ):

        return {
            "model": champion_model,
            "history": champion_history,
            "features": champion_features,
            "model_name": "Champion v1.0",
            "model_version": "champion_v1.0",
            "mode": "PRODUCTION",
            "reason":
                "Both teams are covered by the original "
                "Champion training dataset.",
        }

    # Expanded candidate coverage.
    if (
        team1 in candidate_teams
        and
        team2 in candidate_teams
    ):

        return {
            "model": candidate_model,
            "history": candidate_history,
            "features": candidate_features,
            "model_name": "Candidate v1.2",
            "model_version": "candidate_v1.2",
            "mode": "SHADOW",
            "reason":
                "This match requires expanded historical "
                "coverage that is not available in the "
                "original Champion dataset.",
        }

    raise ValueError(
        "This team combination is not supported "
        "by the available models."
    )


# ================================================================
# TEAM MATCHES
# ================================================================

def get_team_matches(
    history,
    team,
):

    return history[

        (history["team1"] == team)

        |

        (history["team2"] == team)

    ]


# ================================================================
# HISTORICAL WIN RATE
# ================================================================

def historical_win_rate(
    history,
    team,
):

    matches = get_team_matches(
        history,
        team,
    )

    if len(matches) == 0:
        return 0.5

    wins = (
        matches["winner"]
        ==
        team
    ).sum()

    return wins / len(matches)


# ================================================================
# HISTORICAL SAMPLE SIZE
# ================================================================

def historical_sample_size(
    history,
    team,
):

    return len(
        get_team_matches(
            history,
            team,
        )
    )


# ================================================================
# RECENT WIN RATE
# ================================================================

def recent_win_rate(
    history,
    team,
):

    matches = (
        get_team_matches(
            history,
            team,
        )
        .sort_values("date")
        .tail(5)
    )

    if len(matches) == 0:
        return 0.5

    wins = (
        matches["winner"]
        ==
        team
    ).sum()

    return wins / len(matches)


# ================================================================
# HEAD-TO-HEAD MATCHES
# ================================================================

def h2h_matches(
    history,
    team1,
    team2,
):

    return history[

        (
            (history["team1"] == team1)
            &
            (history["team2"] == team2)
        )

        |

        (
            (history["team1"] == team2)
            &
            (history["team2"] == team1)
        )

    ]


# ================================================================
# HEAD-TO-HEAD WIN RATE
# ================================================================

def h2h_win_rate(
    history,
    team,
    opponent,
):

    matches = h2h_matches(
        history,
        team,
        opponent,
    )

    if len(matches) == 0:
        return 0.5

    wins = (
        matches["winner"]
        ==
        team
    ).sum()

    return wins / len(matches)


# ================================================================
# HEAD-TO-HEAD SAMPLE SIZE
# ================================================================

def h2h_sample_size(
    history,
    team1,
    team2,
):

    return len(
        h2h_matches(
            history,
            team1,
            team2,
        )
    )


# ================================================================
# VENUE WIN RATE
# ================================================================

def venue_win_rate(
    history,
    team,
    venue,
):

    if venue == "NEW / Unseen Venue":
        return 0.5

    matches = history[

        (history["venue"] == venue)

        &

        (
            (history["team1"] == team)

            |

            (history["team2"] == team)
        )

    ]

    if len(matches) == 0:
        return 0.5

    wins = (
        matches["winner"]
        ==
        team
    ).sum()

    return wins / len(matches)


# ================================================================
# VENUE SAMPLE SIZE
# ================================================================

def venue_sample_size(
    history,
    team,
    venue,
):

    if venue == "NEW / Unseen Venue":
        return 0

    matches = history[

        (history["venue"] == venue)

        &

        (
            (history["team1"] == team)

            |

            (history["team2"] == team)
        )

    ]

    return len(matches)


# ================================================================
# CALCULATE MODEL FEATURES
# ================================================================

def calculate_match_features(
    history,
    team1,
    team2,
    venue,
    toss_winner,
):

    # Toss encoding:
    #
    # Team1 wins = 1.0
    # Team2 wins = 0.0
    # Unknown    = 0.5

    if toss_winner == "Unknown":

        toss_feature = 0.5

    elif toss_winner == team1:

        toss_feature = 1.0

    else:

        toss_feature = 0.0


    values = {

        "toss_winner_is_team1":
            toss_feature,

        "team1_historical_win_rate":
            historical_win_rate(
                history,
                team1,
            ),

        "team2_historical_win_rate":
            historical_win_rate(
                history,
                team2,
            ),

        "team1_recent_win_rate":
            recent_win_rate(
                history,
                team1,
            ),

        "team2_recent_win_rate":
            recent_win_rate(
                history,
                team2,
            ),

        "team1_h2h_win_rate":
            h2h_win_rate(
                history,
                team1,
                team2,
            ),

        "team2_h2h_win_rate":
            h2h_win_rate(
                history,
                team2,
                team1,
            ),

        "team1_venue_win_rate":
            venue_win_rate(
                history,
                team1,
                venue,
            ),

        "team2_venue_win_rate":
            venue_win_rate(
                history,
                team2,
                venue,
            ),
    }


    reliability = {

        "team1_historical_sample_size":
            historical_sample_size(
                history,
                team1,
            ),

        "team2_historical_sample_size":
            historical_sample_size(
                history,
                team2,
            ),

        "h2h_sample_size":
            h2h_sample_size(
                history,
                team1,
                team2,
            ),

        "team1_venue_sample_size":
            venue_sample_size(
                history,
                team1,
                venue,
            ),

        "team2_venue_sample_size":
            venue_sample_size(
                history,
                team2,
                venue,
            ),
    }


    return values, reliability


# ================================================================
# BUILD MODEL INPUT
# ================================================================

def build_model_input(
    feature_values,
    schema,
):

    missing = [

        feature

        for feature in schema

        if feature not in feature_values
    ]

    if missing:

        raise ValueError(
            f"Missing model features: {missing}"
        )

    row = {

        feature:
            feature_values[feature]

        for feature in schema
    }

    return pd.DataFrame(
        [row]
    )


# ================================================================
# MAKE PREDICTION
# ================================================================

def make_prediction(
    model,
    schema,
    feature_values,
    team1,
    team2,
):

    model_input = build_model_input(
        feature_values,
        schema,
    )

    prediction = int(
        model.predict(
            model_input
        )[0]
    )

    probabilities = (
        model.predict_proba(
            model_input
        )[0]
    )

    classes = list(
        model.classes_
    )

    zero_index = classes.index(0)
    one_index = classes.index(1)

    team1_probability = float(
        probabilities[one_index]
    )

    team2_probability = float(
        probabilities[zero_index]
    )

    predicted_winner = (
        team1
        if prediction == 1
        else team2
    )

    confidence = max(
        team1_probability,
        team2_probability,
    )

    if confidence >= 0.70:

        confidence_level = "HIGH"

    elif confidence >= 0.60:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"

    return {

        "predicted_winner":
            predicted_winner,

        "team1_probability":
            team1_probability,

        "team2_probability":
            team2_probability,

        "confidence":
            confidence,

        "confidence_level":
            confidence_level,
    }


# ================================================================
# PREDICTION LOG
# ================================================================

def append_prediction_log(
    file_path,
    row,
):

    new_df = pd.DataFrame(
        [row]
    )

    if file_path.exists():

        old_df = pd.read_csv(
            file_path
        )

        final_df = pd.concat(
            [
                old_df,
                new_df,
            ],
            ignore_index=True,
        )

    else:

        final_df = new_df

    final_df.to_csv(
        file_path,
        index=False,
    )


# ================================================================
# FEATURE DISPLAY
# ================================================================

def display_feature_values(
    feature_values,
    team1,
    team2,
):

    table = pd.DataFrame({

        "Feature": [
            "Historical Win Rate",
            "Recent Win Rate",
            "Head-to-Head Win Rate",
            "Venue Win Rate",
        ],

        team1: [

            feature_values[
                "team1_historical_win_rate"
            ],

            feature_values[
                "team1_recent_win_rate"
            ],

            feature_values[
                "team1_h2h_win_rate"
            ],

            feature_values[
                "team1_venue_win_rate"
            ],
        ],

        team2: [

            feature_values[
                "team2_historical_win_rate"
            ],

            feature_values[
                "team2_recent_win_rate"
            ],

            feature_values[
                "team2_h2h_win_rate"
            ],

            feature_values[
                "team2_venue_win_rate"
            ],
        ],
    })

    table[team1] = (
        table[team1]
        * 100
    ).round(2)

    table[team2] = (
        table[team2]
        * 100
    ).round(2)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )


# ================================================================
# TEAM INPUT
# ================================================================

def team_input_form(
    key_prefix,
    team_options,
):

    col1, col2 = st.columns(2)

    with col1:

        default_team1 = (

            team_options.index(
                "Pakistan"
            )

            if "Pakistan" in team_options

            else 0
        )

        team1 = st.selectbox(
            "Team 1",
            team_options,
            index=default_team1,
            key=f"{key_prefix}_team1",
        )

    with col2:

        team2_options = [

            team

            for team in team_options

            if team != team1
        ]

        default_team2 = (

            team2_options.index(
                "India"
            )

            if "India" in team2_options

            else 0
        )

        team2 = st.selectbox(
            "Team 2",
            team2_options,
            index=default_team2,
            key=f"{key_prefix}_team2",
        )

    return team1, team2


# ================================================================
# VENUE INPUT
#
# User types city / venue.
# If multiple clean venues match, user chooses exact venue.
# ================================================================

def venue_input_form(
    key_prefix,
    history,
):

    st.write(
        "**Venue / City**"
    )

    search_text = st.text_input(
        "Search Venue",
        placeholder=(
            "Example: Delhi, Mumbai, Dubai, Lahore"
        ),
        label_visibility="collapsed",
        key=f"{key_prefix}_venue_search",
    )

    if not search_text.strip():

        st.caption(
            "Start typing a city or stadium name."
        )

        return None


    clean_venues = build_venue_list(
        history
    )

    search_lower = (
        search_text
        .strip()
        .lower()
    )

    matches = [

        venue

        for venue in clean_venues

        if search_lower
        in venue.lower()
    ]


    # ------------------------------------------------------------
    # No venue found
    # ------------------------------------------------------------

    if not matches:

        st.warning(
            f"No known historical venue found "
            f"for '{search_text}'."
        )

        use_new = st.checkbox(
            "Use as new / unseen venue",
            key=f"{key_prefix}_new_venue",
        )

        if use_new:

            st.info(
                "No venue history will be available. "
                "The model will use the neutral venue value 0.5."
            )

            return "NEW / Unseen Venue"

        return None


    # ------------------------------------------------------------
    # Rank by historical frequency
    # ------------------------------------------------------------

    venue_counts = (
        history["venue"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    matches = sorted(

        matches,

        key=lambda venue: (
            venue_counts.get(
                venue,
                0
            ),
            len(venue),
        ),

        reverse=True,
    )


    # ------------------------------------------------------------
    # One venue
    # ------------------------------------------------------------

    if len(matches) == 1:

        selected = matches[0]

        st.success(
            f"Selected Venue: {selected}"
        )

        count = int(
            venue_counts.get(
                selected,
                0
            )
        )

        st.caption(
            f"Historical records at this venue: {count}"
        )

        return selected


    # ------------------------------------------------------------
    # Multiple venues
    # ------------------------------------------------------------

    st.caption(
        f"{len(matches)} matching historical venues found."
    )

    selected = st.selectbox(
        "Select Exact Venue",
        matches,
        key=f"{key_prefix}_venue_choice",
    )

    count = int(
        venue_counts.get(
            selected,
            0
        )
    )

    st.success(
        f"Selected Venue: {selected}"
    )

    st.caption(
        f"Historical records at this venue: {count}"
    )

    return selected


# ================================================================
# TOSS INPUT
# ================================================================

def toss_input_form(
    key_prefix,
    team1,
    team2,
):

    return st.selectbox(
        "Toss Winner",
        [
            "Unknown",
            team1,
            team2,
        ],
        key=f"{key_prefix}_toss",
    )


# ================================================================
# AI / ML LEARNING PANEL
# ================================================================

def display_learning_panel():

    st.subheader(
        "🎓 AI / ML Learning"
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "AI Type",
        "Predictive AI",
    )

    col2.metric(
        "Learning",
        "Supervised",
    )

    col3.metric(
        "ML Task",
        "Classification",
    )

    col4.metric(
        "Algorithm",
        "Logistic Regression",
    )

    col5.metric(
        "Features",
        "9",
    )

    with st.expander(
        "📘 What do these terms mean?"
    ):

        st.markdown(
            """
**Predictive AI**  
Uses historical data to estimate a future or unknown outcome.

**Supervised Learning**  
The model learns from historical examples where the correct
answer is already known. In our dataset, the historical
`winner` provides the label.

**Binary Classification**  
The model chooses between two classes. In this project:

- `team1_won = 1` → Team 1 won
- `team1_won = 0` → Team 2 won

**Logistic Regression**  
A supervised classification algorithm that learns relationships
between the input features and the probability of the target
class.

**Feature**  
A measurable input supplied to the model. This project uses
nine features based on toss, historical performance, recent
form, head-to-head history and venue performance.
"""
        )


# ================================================================
# HEADER
# ================================================================

st.title(
    "🏏 Cricinfo AI Match Predictor"
)

st.markdown(
    """
### Cricket is a game of skill, strategy, uncertainty and chance.

Cricinfo AI uses historical men's T20 cricket data and
**Predictive Machine Learning** to estimate match win
probabilities.

The goal of this application is not only to predict matches,
but also to demonstrate **how an end-to-end AI/ML system works**.
"""
)

st.info(
    """
**Educational Disclaimer**

Predictions are statistical estimates and not guaranteed
match outcomes.

This project is for learning, education, experimentation,
research practice, demonstration and fun. It does not provide
betting, gambling, financial or professional sports advice.
"""
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title(
    "🏏 Cricinfo AI"
)

st.sidebar.caption(
    "Predictive AI / ML Learning Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏏 Match Predictor",
        "⚖️ Champion vs Candidate",
        "📊 Model Performance",
        "🎓 AI Learning",
        "ℹ️ About Project",
    ],
)

st.sidebar.divider()

st.sidebar.subheader(
    "Model Governance"
)

st.sidebar.write(
    "**Production:** Champion v1.0"
)

st.sidebar.write(
    "**Rejected:** Challenger v1.1"
)

st.sidebar.write(
    "**Shadow:** Candidate v1.2"
)

st.sidebar.divider()

st.sidebar.subheader(
    "AI / ML"
)

st.sidebar.write(
    "**AI Type:** Predictive AI"
)

st.sidebar.write(
    "**Learning:** Supervised"
)

st.sidebar.write(
    "**Task:** Binary Classification"
)

st.sidebar.write(
    "**Algorithm:** Logistic Regression"
)

st.sidebar.write(
    "**Features:** 9"
)


# ================================================================
# PAGE 1
# MATCH PREDICTOR
# ================================================================

if page == "🏏 Match Predictor":

    st.header(
        "Match Prediction"
    )

    st.write(
        "Enter match information below. "
        "The application automatically selects the correct model."
    )


    # ------------------------------------------------------------
    # AI LEARNING PANEL
    # ------------------------------------------------------------

    display_learning_panel()

    st.divider()


    # ------------------------------------------------------------
    # MATCH INPUT
    # ------------------------------------------------------------

    st.subheader(
        "1️⃣ Match Input"
    )

    team1, team2 = team_input_form(
        "predict",
        candidate_teams,
    )


    # ------------------------------------------------------------
    # MODEL ROUTING
    # ------------------------------------------------------------

    try:

        route = select_model(
            team1,
            team2,
        )

    except Exception as error:

        st.error(
            str(error)
        )

        st.stop()


    # ------------------------------------------------------------
    # MODEL INFORMATION
    # ------------------------------------------------------------

    st.subheader(
        "2️⃣ Automatic Model Selection"
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Selected Model",
        route["model_name"],
    )

    col2.metric(
        "Deployment Mode",
        route["mode"],
    )

    st.write(
        "**Why was this model selected?**"
    )

    st.write(
        route["reason"]
    )


    if route["mode"] == "SHADOW":

        st.warning(
            """
**SHADOW means:** Candidate v1.2 is allowed to make predictions
for evaluation, but it has not been approved to replace the
production Champion.
"""
        )

    else:

        st.success(
            """
**PRODUCTION means:** Champion v1.0 is the currently approved
production model.
"""
        )


    # ------------------------------------------------------------
    # VENUE
    # ------------------------------------------------------------

    st.subheader(
        "3️⃣ Match Context"
    )

    venue = venue_input_form(
        "predict",
        route["history"],
    )

    toss_winner = toss_input_form(
        "predict",
        team1,
        team2,
    )


    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------

    predict_button = st.button(
        "🏏 Predict Match",
        type="primary",
        use_container_width=True,
    )


    if predict_button:

        if venue is None:

            st.error(
                "Please enter and select a valid venue."
            )

            st.stop()


        (
            feature_values,
            reliability,
        ) = calculate_match_features(
            route["history"],
            team1,
            team2,
            venue,
            toss_winner,
        )


        result = make_prediction(
            route["model"],
            route["features"],
            feature_values,
            team1,
            team2,
        )


        # ========================================================
        # RESULT
        # ========================================================

        st.divider()

        st.subheader(
            "4️⃣ Prediction Result"
        )

        st.success(
            "🏆 Predicted Winner: "
            +
            result["predicted_winner"]
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        col1.metric(
            team1,
            f"{result['team1_probability'] * 100:.2f}%",
        )

        col2.metric(
            team2,
            f"{result['team2_probability'] * 100:.2f}%",
        )

        col3.metric(
            "Confidence",
            result["confidence_level"],
        )

        col4.metric(
            "Model",
            route["model_name"],
        )


        st.write(
            f"**{team1} probability**"
        )

        st.progress(
            result["team1_probability"]
        )

        st.write(
            f"**{team2} probability**"
        )

        st.progress(
            result["team2_probability"]
        )


        # ========================================================
        # WHAT MODEL DID
        # ========================================================

        st.divider()

        st.subheader(
            "🎓 What did the model just do?"
        )

        st.markdown(
            f"""
The application used a **supervised Machine Learning
classification model** to estimate the outcome of
**{team1} vs {team2}**.

The process was:

**1. Historical data**  
The application selected the historical dataset appropriate
for these teams.

**2. Feature engineering**  
It converted cricket history into **9 numerical features**.

**3. Model inference**  
Those nine values were passed to the trained
**Logistic Regression** model.

**4. Probability calculation**  
The model calculated:

- **{team1}: {result['team1_probability'] * 100:.2f}%**
- **{team2}: {result['team2_probability'] * 100:.2f}%**

**5. Classification**  
The higher model probability resulted in:

### Predicted Winner: {result['predicted_winner']}

This step is called **inference** because the trained model
is being used to make a prediction on new input.
"""
        )


        # ========================================================
        # FEATURE ENGINEERING
        # ========================================================

        st.subheader(
            "🔬 Feature Engineering Used for This Prediction"
        )

        display_feature_values(
            feature_values,
            team1,
            team2,
        )

        toss_table = pd.DataFrame({

            "Feature": [
                "Toss Winner",
                "Encoded Toss Value",
            ],

            "Value": [
                toss_winner,
                feature_values[
                    "toss_winner_is_team1"
                ],
            ],
        })

        st.dataframe(
            toss_table,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            "Unknown toss = 0.5, "
            "Team 1 wins toss = 1.0, "
            "Team 2 wins toss = 0.0."
        )


        # ========================================================
        # RELIABILITY
        # ========================================================

        st.subheader(
            "📚 Historical Evidence Behind the Features"
        )

        evidence = pd.DataFrame({

            "Evidence": [
                f"{team1} historical matches",
                f"{team2} historical matches",
                "Head-to-head matches",
                f"{team1} matches at selected venue",
                f"{team2} matches at selected venue",
            ],

            "Count": [
                reliability[
                    "team1_historical_sample_size"
                ],
                reliability[
                    "team2_historical_sample_size"
                ],
                reliability[
                    "h2h_sample_size"
                ],
                reliability[
                    "team1_venue_sample_size"
                ],
                reliability[
                    "team2_venue_sample_size"
                ],
            ],
        })

        st.dataframe(
            evidence,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            """
**Why sample size matters:** a feature based on many historical
matches generally has more historical evidence behind it than a
feature based on only one or two matches. This does not guarantee
that the prediction is correct, but it helps us understand the
reliability of the evidence supplied to the model.
"""
        )


        # ========================================================
        # LOGGING
        # ========================================================

        log_row = {

            "prediction_timestamp":
                datetime.now().isoformat(
                    timespec="seconds"
                ),

            "team1":
                team1,

            "team2":
                team2,

            "venue":
                venue,

            "toss_winner":
                toss_winner,

            "model_version":
                route["model_version"],

            "predicted_winner":
                result["predicted_winner"],

            "team1_probability":
                result["team1_probability"],

            "team2_probability":
                result["team2_probability"],

            "actual_winner":
                "",

            "correct_prediction":
                "",

            "result_status":
                "PENDING",
        }


        if route["mode"] == "PRODUCTION":

            append_prediction_log(
                PRODUCTION_LOG_FILE,
                log_row,
            )

            st.caption(
                "Prediction saved to production prediction log."
            )

        else:

            append_prediction_log(
                SHADOW_LOG_FILE,
                log_row,
            )

            st.caption(
                "Prediction saved to Candidate v1.2 shadow log."
            )


        st.warning(
            "The prediction is a statistical estimate, "
            "not a guaranteed cricket result."
        )


# ================================================================
# PAGE 2
# CHAMPION VS CANDIDATE
# ================================================================

elif page == "⚖️ Champion vs Candidate":

    st.header(
        "Champion v1.0 vs Candidate v1.2"
    )

    st.write(
        """
This page demonstrates **model comparison**, an important
Machine Learning and MLOps concept.

We run the same input through two trained models and compare
their predictions.
"""
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🏆 Champion v1.0"
        )

        st.write(
            "**Dataset:** Original"
        )

        st.write(
            "**Features:** 9"
        )

        st.write(
            "**Accuracy:** 71.82%"
        )

        st.write(
            "**Status:** PRODUCTION"
        )

    with col2:

        st.subheader(
            "🧪 Candidate v1.2"
        )

        st.write(
            "**Dataset:** Expanded"
        )

        st.write(
            "**Features:** 9"
        )

        st.write(
            "**Accuracy:** 72.26%"
        )

        st.write(
            "**Status:** SHADOW"
        )


    st.divider()


    team1, team2 = team_input_form(
        "compare",
        champion_teams,
    )

    venue = venue_input_form(
        "compare",
        candidate_history,
    )

    toss_winner = toss_input_form(
        "compare",
        team1,
        team2,
    )


    if st.button(
        "⚖️ Compare Models",
        type="primary",
        use_container_width=True,
    ):

        if venue is None:

            st.error(
                "Please select a venue."
            )

            st.stop()


        champion_values, _ = (
            calculate_match_features(
                champion_history,
                team1,
                team2,
                venue,
                toss_winner,
            )
        )


        candidate_values, _ = (
            calculate_match_features(
                candidate_history,
                team1,
                team2,
                venue,
                toss_winner,
            )
        )


        champion_result = make_prediction(
            champion_model,
            champion_features,
            champion_values,
            team1,
            team2,
        )


        candidate_result = make_prediction(
            candidate_model,
            candidate_features,
            candidate_values,
            team1,
            team2,
        )


        st.divider()


        left, right = st.columns(2)


        with left:

            st.subheader(
                "🏆 Champion"
            )

            st.metric(
                "Predicted Winner",
                champion_result[
                    "predicted_winner"
                ],
            )

            st.metric(
                team1,
                f"{champion_result['team1_probability'] * 100:.2f}%",
            )

            st.metric(
                team2,
                f"{champion_result['team2_probability'] * 100:.2f}%",
            )


        with right:

            st.subheader(
                "🧪 Candidate"
            )

            st.metric(
                "Predicted Winner",
                candidate_result[
                    "predicted_winner"
                ],
            )

            st.metric(
                team1,
                f"{candidate_result['team1_probability'] * 100:.2f}%",
            )

            st.metric(
                team2,
                f"{candidate_result['team2_probability'] * 100:.2f}%",
            )


        agree = (
            champion_result[
                "predicted_winner"
            ]
            ==
            candidate_result[
                "predicted_winner"
            ]
        )


        difference = abs(
            champion_result[
                "team1_probability"
            ]
            -
            candidate_result[
                "team1_probability"
            ]
        ) * 100


        col1, col2 = st.columns(2)

        col1.metric(
            "Models Agree",
            "YES" if agree else "NO",
        )

        col2.metric(
            "Probability Difference",
            f"{difference:.2f} pp",
        )


        st.info(
            """
**Learning concept — Champion vs Candidate**

A Champion model is the currently approved model.

A Candidate is a newer model being evaluated.

A Candidate should not replace the Champion simply because one
accuracy result is slightly higher. It should be validated using
fair testing, temporal performance, statistical significance and
future/shadow evidence.
"""
        )


# ================================================================
# PAGE 3
# MODEL PERFORMANCE
# ================================================================

elif page == "📊 Model Performance":

    st.header(
        "Model Performance & Governance"
    )

    st.write(
        """
This page demonstrates **model evaluation** — measuring whether
a Machine Learning model actually performs well on data it was
not trained on.
"""
    )


    summary = pd.DataFrame({

        "Model": [
            "Champion v1.0",
            "Challenger v1.1",
            "Candidate v1.2",
        ],

        "Features": [
            9,
            10,
            9,
        ],

        "Common Test Accuracy": [
            "71.82%",
            "71.24%",
            "72.26%",
        ],

        "Status": [
            "PRODUCTION",
            "REJECTED",
            "SHADOW",
        ],
    })


    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    st.subheader(
        "Candidate v1.2 Validation"
    )


    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Champion",
        "71.82%",
    )

    col2.metric(
        "Candidate",
        "72.26%",
    )

    col3.metric(
        "Difference",
        "+0.44 pp",
    )

    col4.metric(
        "p-value",
        "0.742829",
    )


    st.warning(
        """
Candidate v1.2 had slightly higher accuracy, but its improvement
was **not statistically significant**.

Therefore it was not promoted to production.
"""
    )


    st.subheader(
        "Paired Test Results"
    )


    comparison = pd.DataFrame({

        "Metric": [
            "Common test matches",
            "Champion correct",
            "Candidate correct",
            "Both correct",
            "Both wrong",
            "Candidate improvements",
            "Candidate regressions",
            "Net improvement",
        ],

        "Value": [
            685,
            492,
            495,
            475,
            173,
            20,
            17,
            3,
        ],
    })


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


    st.info(
        """
**Learning concept — Statistical significance**

The Candidate was correct on three more matches than the Champion,
but the paired statistical test returned **p = 0.742829**.

Because this is much greater than 0.05, we do not have strong
evidence that Candidate v1.2 is genuinely better rather than the
difference being explainable by normal sampling variation.
"""
    )


# ================================================================
# PAGE 4
# AI LEARNING
# ================================================================

elif page == "🎓 AI Learning":

    st.header(
        "🎓 AI / Machine Learning Concepts"
    )

    st.write(
        """
This section connects the Cricinfo project directly to the AI/ML
concepts being applied.
"""
    )


    st.subheader(
        "1. Predictive AI"
    )

    st.write(
        """
Predictive AI uses historical data to estimate a future or unknown
outcome.

In this project:

**Historical T20 matches → predicted match winner**
"""
    )


    st.subheader(
        "2. Supervised Learning"
    )

    st.write(
        """
Supervised Learning trains a model using examples where the
correct answer is already known.

Our historical matches contain the actual **winner**.

The model therefore learns relationships between match features
and known historical outcomes.
"""
    )


    st.subheader(
        "3. Classification"
    )

    st.write(
        """
Classification predicts a category rather than a continuous
number.

Our target is binary:

`team1_won = 1`

or

`team1_won = 0`
"""
    )


    st.subheader(
        "4. Feature Engineering"
    )

    st.write(
        """
Raw cricket match information cannot simply be handed to the
algorithm in its original form.

Feature engineering converts historical information into useful
numerical inputs.
"""
    )


    feature_learning = pd.DataFrame({

        "Feature": [
            "Toss",
            "Team 1 Historical Win Rate",
            "Team 2 Historical Win Rate",
            "Team 1 Recent Win Rate",
            "Team 2 Recent Win Rate",
            "Team 1 H2H Win Rate",
            "Team 2 H2H Win Rate",
            "Team 1 Venue Win Rate",
            "Team 2 Venue Win Rate",
        ],

        "What it represents": [
            "Toss information",
            "Long-term Team 1 strength",
            "Long-term Team 2 strength",
            "Recent Team 1 form",
            "Recent Team 2 form",
            "Team 1 direct matchup history",
            "Team 2 direct matchup history",
            "Team 1 venue history",
            "Team 2 venue history",
        ],
    })


    st.dataframe(
        feature_learning,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "5. Logistic Regression"
    )

    st.write(
        """
Logistic Regression is a supervised classification algorithm.

Despite the word **regression** in its name, Logistic Regression
is commonly used for classification.

It estimates the probability of a binary outcome.
"""
    )


    st.subheader(
        "6. Training vs Inference"
    )


    training_table = pd.DataFrame({

        "Stage": [
            "Training",
            "Inference",
        ],

        "Meaning": [
            "Model learns patterns from historical labeled data",
            "Trained model predicts a new match",
        ],

        "Project Example": [
            "train_model.py",
            "app.py / predict_match.py",
        ],
    })


    st.dataframe(
        training_table,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "7. Overfitting"
    )

    st.write(
        """
Overfitting occurs when a model learns the training data too
specifically and performs poorly on unseen data.

This is one reason we evaluate models on separate chronological
test data.
"""
    )


    st.subheader(
        "8. Underfitting"
    )

    st.write(
        """
Underfitting occurs when a model is too simple to capture useful
patterns in the data and performs poorly even on the underlying
problem.
"""
    )


    st.subheader(
        "9. Ablation Testing"
    )

    st.write(
        """
Ablation testing removes or changes part of a model design to
measure its effect.

In our project, the 10-feature Challenger v1.1 was compared with
a 9-feature version trained on the expanded dataset.

The 9-feature Candidate performed better.

This taught us:

**More features do not automatically mean a better model.**
"""
    )


    st.subheader(
        "10. Shadow Model"
    )

    st.write(
        """
A shadow model receives real prediction inputs but does not
replace the production model.

Its predictions can later be compared with actual outcomes.

Candidate v1.2 is currently our shadow model.
"""
    )


    st.subheader(
        "11. Model Drift"
    )

    st.write(
        """
Drift occurs when the data or relationships seen by the model
change over time.

The project includes drift monitoring to help determine whether
the production model may eventually require retraining.
"""
    )


    st.subheader(
        "12. Model Governance"
    )

    st.write(
        """
Model governance defines how models are evaluated, approved,
monitored and promoted.

Our current decision is:

**Champion v1.0 → Production**

**Challenger v1.1 → Rejected**

**Candidate v1.2 → Shadow**

Candidate v1.2 is not promoted because its small historical
accuracy improvement was not statistically significant.
"""
    )


# ================================================================
# PAGE 5
# ABOUT
# ================================================================

elif page == "ℹ️ About Project":

    st.header(
        "About Cricinfo AI"
    )


    st.subheader(
        "Project Objective"
    )

    st.write(
        """
Build an end-to-end Predictive AI / Machine Learning solution
for men's T20 cricket match prediction while learning the
complete ML lifecycle step by step.
"""
    )


    st.subheader(
        "Dataset"
    )


    dataset_table = pd.DataFrame({

        "Dataset": [
            "Original Core Dataset",
            "Supplemental Afghanistan",
            "Expanded Candidate Dataset",
        ],

        "Matches": [
            3539,
            164,
            3703,
        ],
    })


    st.dataframe(
        dataset_table,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "End-to-End ML Lifecycle"
    )


    st.code(
        """
Data Acquisition
       |
       v
Data Validation / Cleaning
       |
       v
Pandas / NumPy Processing
       |
       v
Feature Engineering
       |
       v
Train / Test Split
       |
       v
Scikit-Learn Model Training
       |
       v
Model Validation
       |
       v
Champion / Candidate Comparison
       |
       v
Statistical Validation
       |
       v
Prediction / Inference
       |
       v
Shadow Testing
       |
       v
Monitoring / Drift
       |
       v
Retraining Decision
"""
    )


    st.subheader(
        "Technologies"
    )


    technology_table = pd.DataFrame({

        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-Learn",
            "SciPy",
            "Joblib",
            "Streamlit",
            "Git",
            "GitHub",
        ],

        "Purpose": [
            "Programming",
            "Data processing",
            "Numerical processing",
            "Machine Learning",
            "Statistical testing",
            "Model persistence",
            "Web GUI",
            "Version control",
            "Project repository",
        ],
    })


    st.dataframe(
        technology_table,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Project Author"
    )


    st.markdown(
        """
### Syed Shiraz Hussain

**Technical Manager - Support**  
**AI-Enabled IT & Telecom Operations**  
**PMP | ITIL 4 | Senior Member IEEE**

This project was developed as a practical AI-learning initiative
covering Python, data acquisition, Pandas, NumPy, feature
engineering, Scikit-Learn, Predictive AI, model validation,
statistical testing, Streamlit, Git/GitHub and basic MLOps.
"""
    )


    st.warning(
        """
Cricinfo AI is an educational, experimental, demonstration and
fun project.

Its predictions are statistical estimates and should not be
interpreted as guaranteed cricket outcomes or betting advice.
"""
    )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.caption(
    "Cricinfo AI Match Predictor | "
    "Predictive AI • Supervised Learning • Classification • "
    "Logistic Regression • MLOps | "
    "Developed by Syed Shiraz Hussain"
)