from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('ridge.pkl', 'rb'))
stand_scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_data():
    if request.method == "POST":
        # Get form data
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("WS"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        # Arrange features into numpy array
        input_data = np.array([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region]])

        # Scale data
        scaled_data = stand_scaler.transform(input_data)

        # Predict
        prediction = ridge_model.predict(scaled_data)[0]

        # Return to template with prediction
        return render_template("home.html", result=prediction)

    else:
        return render_template("home.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0')
