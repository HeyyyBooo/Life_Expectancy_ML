# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 18:31:06 2023

@author: narze Nishant
"""



import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

afg = joblib.load("afg.pkl")
arg = joblib.load("arg.pkl")
ban = joblib.load("ban.pkl")
bhu = joblib.load("bhu.pkl")
china = joblib.load("china.pkl")
fra = joblib.load("fra.pkl")
ger = joblib.load("ger.pkl")
india = joblib.load("india.pkl")
jp = joblib.load("jp.pkl")
kor = joblib.load("kor.pkl")
nepal = joblib.load("nepal.pkl")
nz = joblib.load("nz.pkl")
pak = joblib.load("pak.pkl")
qtr = joblib.load("qtr.pkl")
rs= joblib.load("rs.pkl")
sa= joblib.load("sa.pkl")
saudi= joblib.load("saudi.pkl")
sl=joblib.load("sl.pkl")
uae=joblib.load("uae.pkl")
uk=joblib.load("uk.pkl")
usa=joblib.load("usa.pkl")
zim = joblib.load("zim.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/data.html')
def data():
    return render_template('data.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features = [x for x in request.form.values()]
    
    
    for i in [1,2,3,4,5] :
        input_features[i]=int(input_features[i])
    
    #for BMI Optimization
    if(input_features[3]>25):
        input_features[3]=35-input_features[3]
    
    print(input_features)
    features_value = np.array(input_features[1:])
    
    print(input_features[0])
    
    if(input_features[0]=="Afghanistan"):
        output_arr = afg.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='Argentina'):
        output_arr = arg.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='Bangladesh'):
        output_arr = ban.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='Bhutan'):
        output_arr = bhu.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='C'):
        output_arr = china.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='F'):
        output_arr = fra.predict([features_value])
        output=output_arr[0].round(2)    
    elif(input_features[0]=='G'):
        output_arr = ger.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='India'):
        output_arr = india.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='J'):
        output_arr = jp.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='N'):
        output_arr = nepal.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='NZ'):
        output_arr = nz.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='P'):
        output_arr = pak.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='Q'):
        output_arr = qtr.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='RK'):
        output_arr = kor.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='R'):
        output_arr = rs.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='S'):
        output_arr = saudi.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='SA'):
        output_arr = sa.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='SL'):
        output_arr = sl.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='UAE'):
        output_arr = uae.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='UK'):
        output_arr = uk.predict([features_value])
        output=output_arr[0].round(2)   
    elif(input_features[0]=='USA'):
        output_arr = usa.predict([features_value])
        output=output_arr[0].round(2)
    elif(input_features[0]=='Z'):
        output_arr = zim.predict([features_value])
        output=output_arr[0].round(2)
        
    output=output_arr[0].round(2)
   
    return render_template('index.html', prediction_text='Your Predicted age {} '.format(output))
    
       

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    