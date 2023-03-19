from flask import Flask,render_template,request
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
app = Flask(__name__)
car = pd.read_csv('CarPrice_Assignment_Assignment_3.csv')
pipe = pickle.load(open("Randomforestregressor.pkl","rb"))


# [carbody,'fueltype', 'aspiration', 'doornumber', 'drivewheel',
#        'enginelocation', 'wheelbase', 'carwidth', 'enginetype', 'enginesize',
#        'fuelsystem', 'boreratio', 'horsepower', 'highwaympg']

@app.route('/')
def index():
    return render_template('file.html',new1=car['carbody'].unique(),new2=car['fueltype'].unique(),
                           new3=car['aspiration'].unique(),new4=car['doornumber'].unique(),new5=car['drivewheel'].unique()
                           ,new6=car['enginelocation'].unique(),new7=car['wheelbase'].unique()
                           ,new8=car['carwidth'].unique(),new9=car['enginetype'].unique()
                           ,new10=car['enginesize'].unique(),new11=car['fuelsystem'].unique()
                           ,new12=car['boreratio'].unique(),new13=car['horsepower'].unique()
                           ,new14=car['highwaympg'].unique())


@app.route('/predict',methods=['POST'])
def predict():
    carbody = (request.form['carbody'])
    fueltype = (request.form['fueltype'])
    aspiration = (request.form['aspiration'])
    doornumber = (request.form['doornumber'])
    drivewheel = (request.form['drivewheel'])
    enginelocation = (request.form['enginelocation'])
    wheelbase = (request.form['wheelbase'])
    carwidth = (request.form['carwidth'])
    enginetype = (request.form['enginetype'])
    enginesize = (request.form['enginesize'])
    fuelsystem = (request.form['fuelsystem'])
    boreratio = (request.form['boreratio'])
    horsepower = (request.form['horsepower'])
    highwaympg = (request.form['highwaympg'])

    input =  pd.DataFrame([[carbody,fueltype, aspiration, doornumber, drivewheel,
       enginelocation, wheelbase, carwidth, enginetype, enginesize,
       fuelsystem, boreratio, horsepower, highwaympg]],columns=['carbody','fueltype', 'aspiration', 'doornumber', 'drivewheel',
       'enginelocation', 'wheelbase', 'carwidth', 'enginetype', 'enginesize',
       'fuelsystem', 'boreratio', 'horsepower', 'highwaympg'])
    prediction = pipe.predict(input)
    return prediction


if __name__== '__main__':
    app.run(debug=True)



