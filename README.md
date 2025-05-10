

# AI Disease Prediction
This is a Flask web application that predicts the most likely disease based on the symptoms selected by the user. It uses a machine learning model (Decision Tree Classifier) trained on a realistic synthetic dataset with 100 diseases and multiple symptoms.

## Features
- Input form for patient details  
- Symptom checklist dynamically generated from dataset  
- Disease prediction based on selected symptoms  
- Displays prediction result and model accuracy  
- Responsive and styled UI  

## Dataset Used
The model is trained on a synthetic dataset containing realistic combinations of symptoms and diseases.

## Technologies Used
- Python  
- Flask  
- Pandas  
- scikit-learn  
- HTML/CSS  

## Getting Started
Step 1: Clone the Repository  
'''bash
git clone https://github.com/reenasoundar/disease_predictor_web.git

cd disease_predictor_web.git

Step 2: Set Up a Virtual Environment

python -m venv venv

Step 3: Activate the Virtual Environment
For Windows:

venv\Scripts\activate

For macOS/Linux:

source venv/bin/activate

Step 4: Install Required Packages

pip install -r requirements.txt

Step 5: Run the Application

python app.py

Open your browser and go to: http://127.0.0.1:5000

Folder Structure

/static/
  - style.css
  - bg2.jpg
  - result_background.jpg
  -img1
  -img2
  -img3
  -img4
  -logo
  -home_backgroung
  -script.js
  -style
/model/
  -accuracy
  -disease_model
  -symptoms
  

/templates/
  - index.html
  - symptoms.html
  - result.html

app.py
synthetic_symptoms_dataset_realistic_100_diseases(3).xlsx



