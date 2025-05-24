#project-folder/
#├── app.py
#├── logistic_model.pkl
#└── (optional) scaler.pkl

from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np

# Load the trained model
model = joblib.load(open('API/model.pkl','rb'))

# Optional: load scaler if used during training
scaler = joblib.load(open('API/scaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():


 try:
        if request.is_json:
            data = request.get_json()
        else:
            # Handle form data
            import json
            data = json.loads(request.form['json'])

        input_values = np.array([list(data.values())])
        prediction = model.predict(input_values)[0]
        result = 'Yes' if prediction == 1 else 'No'
        return render_template('index.html',Churn_Prediction='{}'.format(result))
        #return jsonify({'Churn Prediction': result})

 except Exception as e:
        return jsonify({'error': str(e)})






#try:
        # Convert input to NumPy array
        #input_values = np.array([list(data)])

        # Optional: scale data if used during training
        #input_values = scaler.transform(input_values)

        # Predict
        #prediction = model.predict(input_values)[0]

        #result = 'Yes' if prediction == 1 else 'No'
       # return render_template('index.html',predication_result='{}'.format(result))

#except Exception as e:
 #   return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)