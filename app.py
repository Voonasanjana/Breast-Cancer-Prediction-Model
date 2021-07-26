from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('cancer.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        MeanRadius = float(request.form['mean_radius'])
        MeanTexture = float(request.form['mean_texture'])
        MeanPerimeter = float(request.form['mean_perimeter'])
        MeanArea = float(request.form['mean_area'])
        MeanSmoothing = float(request.form['mean_smoothness'])

        values = np.array([[MeanRadius,MeanTexture,MeanPerimeter,MeanArea,MeanSmoothing]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)