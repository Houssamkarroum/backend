import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, log_loss
from sklearn.preprocessing import LabelEncoder

# Load your dataset
matches = pd.read_csv("C:/Users/LEGEND/Desktop/final_merged_seasons.csv")

# Initialize LabelEncoders for categorical features
home_team_encoder = LabelEncoder()
away_team_encoder = LabelEncoder()
division_encoder = LabelEncoder()
half_time_result_encoder = LabelEncoder()

# Fit encoders
home_team_encoder.fit(matches["HomeTeam"])
away_team_encoder.fit(matches["AwayTeam"])
division_encoder.fit(matches["Div"])
half_time_result_encoder.fit(matches["HTR"])

# Save the encoders
joblib.dump(home_team_encoder, 'home_team_encoder.pkl')
joblib.dump(away_team_encoder, 'away_team_encoder.pkl')
joblib.dump(division_encoder, 'division_encoder.pkl')
joblib.dump(half_time_result_encoder, 'half_time_result_encoder.pkl')

# Map target variable 'FTR' to numerical codes
result_mapping = {'H': 2, 'D': 1, 'A': 0}
matches['FTR'] = matches['FTR'].map(result_mapping)

# Convert categorical variables to numerical codes using encoders
matches["HomeTeam"] = home_team_encoder.transform(matches["HomeTeam"])
matches["AwayTeam"] = away_team_encoder.transform(matches["AwayTeam"])
matches["Div"] = division_encoder.transform(matches["Div"])
matches["HTR"] = half_time_result_encoder.transform(matches["HTR"])

# Drop rows with missing values
matches = matches.dropna()

# Define predictors and target variables
predictors = [
    "HomeTeam", "AwayTeam", "Div",
    "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"
]
outcome_target = "FTR"

# Ensure all predictors are present in the DataFrame
available_predictors = [col for col in predictors if col in matches.columns]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(matches[available_predictors], matches[outcome_target], test_size=0.2, random_state=42)

# Train the final model with the best parameters
best_params = {
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 10,
    'n_estimators': 200
}

final_rf_model = RandomForestClassifier(random_state=1, **best_params)
final_rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(final_rf_model, 'final_rf_model.pkl')

# Optionally, evaluate the model
final_predictions = final_rf_model.predict(X_test)
final_pred_proba = final_rf_model.predict_proba(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
final_precision = precision_score(y_test, final_predictions, average='weighted')
final_log_loss = log_loss(y_test, final_pred_proba)

print("Final Random Forest Model:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Log Loss: {final_log_loss:.4f}")

# Save the model and encoders
joblib.dump(final_rf_model, 'final_rf_model.pkl')
