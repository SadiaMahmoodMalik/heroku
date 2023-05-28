import numpy as np
from flask import Flask, request, jsonify
import pickle

model = pickle.load(open('heartDiseaseModel.pk1', 'rb'))
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    HighBP = float(request.form.get('HighBP'))
    HighChol = float(request.form.get('HighChol'))
    CholCheck = float(request.form.get('CholCheck'))
    BMI = float(request.form.get('BMI'))
    Smoker = float(request.form.get('Smoker'))
    Stroke = float(request.form.get('Stroke'))
    Diabetes = float(request.form.get('Diabetes'))
    PhysActivity = float(request.form.get('PhysActivity'))
    Fruits = float(request.form.get('Fruits'))
    Veggies = float(request.form.get('Veggies'))
    HvyAlcoholConsump = float(request.form.get('HvyAlcoholConsump'))
    AnyHealthcare = float(request.form.get('AnyHealthcare'))
    NoDocbcCost = float(request.form.get('NoDocbcCost'))
    GenHlth = float(request.form.get('GenHlth'))
    MentHlth = float(request.form.get('MentHlth'))
    PhysHlth = float(request.form.get('PhysHlth'))
    DiffWalk = float(request.form.get('DiffWalk'))
    Sex = float(request.form.get('Sex'))
    Age = float(request.form.get('Age'))
    Education = float(request.form.get('Education'))
    Income = float(request.form.get('Income'))

    input_query = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity,
                             Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
                             GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income]])

    result = model.predict(input_query)  # Adjust the prediction method based on your model

    return jsonify({'HeartDiseaseorAttack': result[0]})  # Assuming result is a single value


if __name__ == '__main__':
    app.run(debug=True)
