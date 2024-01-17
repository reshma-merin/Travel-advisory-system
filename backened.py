from flask import Flask, render_template, request, jsonify
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

app = Flask(__name__)

# Dummy data for demonstration
data = {
    'Age': [25, 30, 35, 40, 45],
    'Budget': [1000, 1500, 2000, 2500, 3000],
    'Safety': [1, 2, 3, 4, 5],
    'Destination': ['Beach', 'Mountain', 'City', 'Beach', 'Mountain']
}

df = pd.DataFrame(data)

# Machine learning model
X = df[['Age', 'Budget', 'Safety']]
y = df['Destination']

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Age': int(request.form['age']),
            'Budget': int(request.form['budget']),
            'Safety': int(request.form['safety'])
        }

        input_array = np.array([input_data['Age'], input_data['Budget'], input_data['Safety']]).reshape(1, -1)
        prediction = model.predict(input_array)

        return render_template('result.html', destination=prediction[0])

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
