from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('model/heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = 'Heart Disease' if prediction == 1 else 'No Heart Disease'
        return render_template('index.html', prediction_text=f'Result: {output}')
    
if __name__ == "__main__":
    app.run(debug=True)
