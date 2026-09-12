import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# -------------------------------------------------
# Load Feature-Engineered Dataset
# -------------------------------------------------

file_path = r"K:\Python\Cricinfo_AI_Project\data\matches_features.csv"

df = pd.read_csv(file_path)

# Convert date to datetime and enforce chronological order
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# -------------------------------------------------
# Select 9 Features
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

X = df[features]
y = df["team1_won"]

print("X - Features:")
print(X.head())

print("\ny - Target:")
print(y.head())

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# -------------------------------------------------
# Train / Test Split
# Chronological split: first 80% train, last 20% test
# -------------------------------------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n--- TRAIN / TEST SPLIT ---")

print("Total matches:", len(df))
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

print(
    "\nTraining percentage:",
    round(len(X_train) / len(df) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(df) * 100, 2),
    "%"
)

training_start_date = df.iloc[:split_index]["date"].min()
training_end_date = df.iloc[:split_index]["date"].max()

testing_start_date = df.iloc[split_index:]["date"].min()
testing_end_date = df.iloc[split_index:]["date"].max()

print("\nTraining date range:")
print(
    training_start_date,
    "to",
    training_end_date
)

print("\nTesting date range:")
print(
    testing_start_date,
    "to",
    testing_end_date
)


# -------------------------------------------------
# Logistic Regression Model Training
# -------------------------------------------------

print("\n--- LOGISTIC REGRESSION MODEL TRAINING ---")

model = LogisticRegression()

model.fit(
    X_train,
    y_train
)

print("Logistic Regression trained successfully.")


# -------------------------------------------------
# Logistic Regression Evaluation
# -------------------------------------------------

print("\n--- LOGISTIC REGRESSION EVALUATION ---")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "Logistic Regression Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -------------------------------------------------
# Sample Match Prediction
# -------------------------------------------------

print("\n--- SAMPLE MATCH PREDICTION ---")

new_match = pd.DataFrame([{
    "toss_winner_is_team1": 1,
    "team1_historical_win_rate": 0.65,
    "team2_historical_win_rate": 0.50,
    "team1_recent_win_rate": 0.60,
    "team2_recent_win_rate": 0.40,
    "team1_h2h_win_rate": 0.60,
    "team2_h2h_win_rate": 0.40,
    "team1_venue_win_rate": 0.55,
    "team2_venue_win_rate": 0.45
}])

prediction = model.predict(
    new_match
)

probability = model.predict_proba(
    new_match
)

print(
    "Prediction:",
    prediction[0]
)

print(
    "Team 1 win probability:",
    round(probability[0][1] * 100, 2),
    "%"
)

print(
    "Team 2 win probability:",
    round(probability[0][0] * 100, 2),
    "%"
)


# -------------------------------------------------
# Random Forest Model Training
# -------------------------------------------------

print("\n--- RANDOM FOREST MODEL ---")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully.")


# -------------------------------------------------
# Random Forest Evaluation
# -------------------------------------------------

rf_pred = rf_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(
    "Random Forest Accuracy:",
    round(rf_accuracy * 100, 2),
    "%"
)

print("\nRandom Forest Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_pred
    )
)

print("\nRandom Forest Classification Report:")

print(
    classification_report(
        y_test,
        rf_pred
    )
)


# -------------------------------------------------
# Train vs Test Scores
# -------------------------------------------------

print("\n--- TRAIN VS TEST SCORE ---")

lr_train_score = model.score(
    X_train,
    y_train
)

lr_test_score = model.score(
    X_test,
    y_test
)

rf_train_score = rf_model.score(
    X_train,
    y_train
)

rf_test_score = rf_model.score(
    X_test,
    y_test
)

print(
    "Logistic Regression Train Score:",
    round(lr_train_score * 100, 2),
    "%"
)

print(
    "Logistic Regression Test Score:",
    round(lr_test_score * 100, 2),
    "%"
)

print(
    "\nRandom Forest Train Score:",
    round(rf_train_score * 100, 2),
    "%"
)

print(
    "Random Forest Test Score:",
    round(rf_test_score * 100, 2),
    "%"
)


# -------------------------------------------------
# Logistic Regression Feature Importance
# -------------------------------------------------

print("\n--- LOGISTIC REGRESSION FEATURE IMPORTANCE ---")

feature_importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

feature_importance["Absolute_Importance"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Importance",
    ascending=False
)

print(
    feature_importance.to_string(
        index=False
    )
)


# -------------------------------------------------
# Random Forest Feature Importance
# -------------------------------------------------

print("\n--- RANDOM FOREST FEATURE IMPORTANCE ---")

rf_feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

rf_feature_importance = rf_feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    rf_feature_importance.to_string(
        index=False
    )
)


# -------------------------------------------------
# Final Model Comparison
# -------------------------------------------------

print("\n--- FINAL MODEL COMPARISON ---")

print(
    "Logistic Regression Test Accuracy:",
    round(lr_test_score * 100, 2),
    "%"
)

print(
    "Random Forest Test Accuracy:",
    round(rf_test_score * 100, 2),
    "%"
)

if lr_test_score > rf_test_score:

    best_model_name = "Logistic Regression"
    best_model = model
    best_test_score = lr_test_score

elif rf_test_score > lr_test_score:

    best_model_name = "Random Forest"
    best_model = rf_model
    best_test_score = rf_test_score

else:

    best_model_name = "Logistic Regression"
    best_model = model
    best_test_score = lr_test_score


print(
    "Current Best Model:",
    best_model_name
)


# -------------------------------------------------
# Save Best Trained Model + Metadata
# -------------------------------------------------

model_path = r"K:\Python\Cricinfo_AI_Project\model\cricket_model.pkl"

model_package = {
    "model": best_model,
    "features": features,
    "model_version": "1.0",
    "model_type": best_model_name,
    "feature_count": len(features),
    "total_matches": len(df),
    "training_matches": len(X_train),
    "testing_matches": len(X_test),
    "training_start_date": str(training_start_date.date()),
    "training_end_date": str(training_end_date.date()),
    "testing_start_date": str(testing_start_date.date()),
    "testing_end_date": str(testing_end_date.date()),
    "test_accuracy": round(best_test_score * 100, 2)
}

joblib.dump(
    model_package,
    model_path
)

print("\n--- MODEL SAVED ---")

print(
    "Model package saved successfully."
)

print(
    "Model location:",
    model_path
)

print(
    "Model version:",
    model_package["model_version"]
)

print(
    "Model type:",
    model_package["model_type"]
)

print(
    "Features saved:",
    model_package["feature_count"]
)

print(
    "Total matches:",
    model_package["total_matches"]
)

print(
    "Training matches:",
    model_package["training_matches"]
)

print(
    "Testing matches:",
    model_package["testing_matches"]
)

print(
    "Test accuracy saved:",
    model_package["test_accuracy"],
    "%"
)