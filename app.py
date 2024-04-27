from flask import Flask, render_template, request
import pandas as pd
import joblib

model = joblib.load('diabetes_model.py')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()
        input_features = pd.DataFrame([input_data])
        prediction = model.predict(input_features)
        if prediction[0] == 0:
            result = 'Not Diabetic'
        else:
            result = 'Diabetic'
        return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)