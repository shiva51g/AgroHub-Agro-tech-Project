from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data_df = pd.DataFrame(data)
    data_scaled = scaler.transform(data_df)
    prediction = model.predict(data_scaled)
    prediction_labels = label_encoder.inverse_transform(prediction)
    return jsonify(prediction_labels.tolist())

if __name__ == '__main__':
    app.run(debug=True)
