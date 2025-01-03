from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

# Initialize the flask application
application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle GET and POST requests for the '/predictdata' route.

    This function serves two purposes:
    1. For GET requests, it renders the 'home.html' template.
    2. For POST requests, it processes form data, makes a prediction, and renders the result.

    Parameters:
    None (implicitly uses Flask's request object)

    Returns:
    str: Rendered HTML template. For GET requests, it's 'home.html'.
         For POST requests, it's 'home.html' with prediction results.

    Note:
    - The function expects form data for various student attributes when handling POST requests.
    - It uses CustomData and PredictPipeline classes for data processing and prediction.
    """
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        