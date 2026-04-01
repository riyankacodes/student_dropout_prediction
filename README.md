# AI Student Dropout Risk Predictor

A machine learning web application that predicts student dropout risk based on academic performance indicators.

## Features
- Predicts **Low Risk** or **High Risk** of dropout
- Shows exact dropout probability score
- Clean, responsive web interface
- Trained on 10,000 synthetic student records
- **84.4% model accuracy**

## Input Parameters
| Parameter | Description |
|---|---|
| Attendance (%) | Student's class attendance percentage |
| Study Hours/Day | Average daily study hours |
| Assignment Completion (%) | Percentage of assignments submitted |
| GPA (4–10) | Grade Point Average |
| Classroom Interaction (1–10) | Level of class participation |

## Tech Stack
- **ML Model**: Logistic Regression (scikit-learn)
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render

## How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/riyankacodes/student_dropout_prediction
cd student_dropout_prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (generates model.pkl):
```bash
python train_model.py
```

4. Run the Flask app:
```bash
python app.py
```

5. Open browser at: `http://localhost:5000`

## Project Structure
```
student_dropout_prediction/
├── app.py                  # Flask backend
├── train_model.py          # Model training script
├── model.pkl               # Trained ML model
├── requirements.txt        # Python dependencies
├── Procfile                # For Render deployment
├── templates/
│   └── index.html          # Frontend web page
└── README.md
```

## Team
- Baishali Mishra (24215603)
- Riyanka Bhattacharyya (24215609)
- Sejal Pandey (24215610)
- Sritirupa Dey (24215612)

**Institution**: Christ (Deemed to be University), Delhi NCR  
**Course**: Intel Unnati Generative AI · 2026
