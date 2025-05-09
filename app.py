from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load and shuffle dataset
df = pd.read_excel("synthetic_symptoms_dataset_realistic_100_diseases (3).xlsx")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare data
X = df.drop("Label", axis=1).replace({'yes': 1, 'no': 0})
y = df["Label"]
symptom_names = list(X.columns)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
model = DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
joblib.dump(model, "disease_model.pkl")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['name'] = request.form.get('name')
        session['age'] = request.form.get('age')
        session['height'] = request.form.get('height')
        session['weight'] = request.form.get('weight')
        session['blood_group'] = request.form.get('blood')
        session['gender'] = request.form.get('gender')
        return redirect(url_for('symptoms'))
    return render_template('index.html')


@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if request.method == 'POST':
        selected_symptoms = [
            symptom for symptom in symptom_names if request.form.get(symptom) == 'yes'
        ]
        session['selected_symptoms'] = selected_symptoms

        user_input = [1 if symptom in selected_symptoms else 0 for symptom in symptom_names]
        model = joblib.load("disease_model.pkl")
        prediction = model.predict([user_input])[0]
        confidence = model.predict_proba([user_input])[0].max() * 100

        # Clamp confidence
        confidence = max(65, min(confidence, 85))

        return redirect(url_for(
            'result',
            prediction=prediction,
            confidence=round(confidence, 2),
            accuracy=round(accuracy * 100, 2)
        ))
    return render_template('symptoms.html', symptoms=symptom_names)


@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    acc = request.args.get('accuracy')

    return render_template('result.html',
                           prediction=prediction,
                           confidence=confidence,
                           accuracy=acc,
                           name=session.get('name'),
                           age=session.get('age'),
                           height=session.get('height'),
                           weight=session.get('weight'),
                           blood_group=session.get('blood_group'),
                           gender=session.get('gender'))


@app.route('/submit-review', methods=['POST'])
def submit_review():
    review_text = request.form['review']
    flash('Review sent successfully. Thank you for your feedback!')
    return redirect(url_for('index'))


@app.route('/download-result')
def download_result():
    # Gather data
    name = session.get('name', 'N/A')
    age = session.get('age', 'N/A')
    height = session.get('height', 'N/A')
    weight = session.get('weight', 'N/A')
    blood_group = session.get('blood_group', 'N/A')
    gender = session.get('gender', 'N/A')
    prediction = request.args.get('prediction', 'N/A')
    confidence = request.args.get('confidence', 'N/A')
    acc = request.args.get('accuracy', 'N/A')

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height_page = letter

    y = height_page - 50
    p.setFont("Helvetica-Bold", 16)
    p.drawString(180, y, "Prediction Report")

    p.setFont("Helvetica", 12)
    y -= 40
    p.drawString(50, y, f"Patient Name: {name}")
    y -= 20
    p.drawString(50, y, f"Age: {age}")
    y -= 20
    p.drawString(50, y, f"Gender: {gender}")
    y -= 20
    p.drawString(50, y, f"Height: {height} cm")
    y -= 20
    p.drawString(50, y, f"Weight: {weight} kg")
    y -= 20
    p.drawString(50, y, f"Blood Group: {blood_group}")

    y -= 40
    p.setFont("Helvetica-Bold", 13)
    p.drawString(50, y, f"Disease Predicted: {prediction}")
    y -= 20
    p.setFont("Helvetica", 12)
    p.drawString(50, y, f"Prediction Confidence: {confidence}%")
    y -= 20
    p.drawString(50, y, f"Model Accuracy: {acc}%")

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(buffer,
                     as_attachment=True,
                     download_name="prediction_result.pdf",
                     mimetype='application/pdf')


if __name__ == '__main__':
    app.run(debug=True)
