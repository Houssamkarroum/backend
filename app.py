from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your model and encoders
model = joblib.load('./final_rf_model.pkl')
home_team_encoder = joblib.load('./home_team_encoder.pkl')
away_team_encoder = joblib.load('./away_team_encoder.pkl')
division_encoder = joblib.load('./division_encoder.pkl')
half_time_result_encoder = joblib.load('./half_time_result_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    home_team = data['home_team']
    away_team = data['away_team']
    # Define a mapping for the prediction results
    result_mapping = {0: away_team + ' wins', 1: 'Draw', 2: home_team + ' wins'}
    # Encode team names using the same encoders
    try:
        home_team_encoded = home_team_encoder.transform([home_team])[0]
        away_team_encoded = away_team_encoder.transform([away_team])[0]
    except ValueError as e:
        return jsonify({'error': 'Invalid team name'}), 400

    # Prepare features (make sure this matches the format the model expects)
    features = [
        home_team_encoded,
        away_team_encoded,
        # Add other features as needed, for now we'll use dummy values for the remaining features
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    
    prediction = model.predict([features])
    probability = model.predict_proba([features])

    # Convert prediction and probability to Python native types and format probabilities as percentages
    result = {
        'home_team': home_team,
        'away_team': away_team,
        'winning_team': result_mapping[int(prediction[0])],  # Convert int64 to int and map to result
        'probability': [f"{prob * 100:.2f}" for prob in probability[0]]  # Format each probability as percentage
    }
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
