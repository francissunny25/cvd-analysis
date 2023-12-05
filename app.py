from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            General_Health = request.form.get('health'),
            Checkup = request.form.get('checkup'),
            Age_Category = request.form.get('age'),
            Exercise = request.form.get('exercise'),
            Skin_Cancer = request.form.get('skin_cancer'),
            Other_Cancer = request.form.get('other_cancer'),
            Depression = request.form.get('depression'),
            Diabetes = request.form.get('diabetes'),
            Arthritis = request.form.get('arthritis'),
            Sex = request.form.get('sex'),
            Smoking_History = request.form.get('smoking'),
            Height_cm = float(request.form.get('height')),
            Weight_kg = float(request.form.get('weight')),
            BMI = float(request.form.get('bmi')),
            Alcohol_Consumption = float(request.form.get('alcohol_units')),
            Fruit_Consumption = float(request.form.get('fruit_units')),
            Green_Vegetables_Consumption = float(request.form.get('veg_units')),
            FriedPotato_Consumption = float(request.form.get('potato_units'))
        )
    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predictor(pred_df)
    return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)