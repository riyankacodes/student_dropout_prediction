from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        attendance = float(data['attendance'])
        study_hours = float(data['study_hours'])
        gpa = float(data['gpa'])
        classroom_interaction = float(data['classroom_interaction'])
        assignment_completion = float(data['assignment_completion'])

        features = pd.DataFrame([[attendance, study_hours, gpa,
                                   classroom_interaction, assignment_completion]],
                                 columns=['attendance', 'study_hours', 'gpa',
                                          'classroom_interaction', 'assignment_completion'])

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        result = {
            'prediction': int(prediction),
            'risk_label': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': round(float(probability), 4),
            'probability_percent': round(float(probability) * 100, 1)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
