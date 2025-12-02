import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd 

#flask is used to create light weight web applications
app=Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling_v1.pkl','rb'))

#with below command if some one enter app it will redirected to below defined html
@app.route('/')
def home():
    return render_template('home.html')

#below command is used to create api which can be used by tools like postman
#this works like giving input from our side to predict_api app 
@app.route('/predict_api',methods=['Post'])

def predict_api():
    #below command says what ever request we give to predict_api
    #it will be stored in below variable data in json format
    data=request.json['data']
    vals = np.array(list(data.values()))  # convert dict values → array
    vals = vals.reshape(1, -1)            # reshape properly

    print(vals)

    new_data = scaler.transform(vals)
    output = regmodel.predict(new_data)
    return jsonify(float(output[0]))

    new_data=scaler.transform(np.array(list(data.values().reshape(1,-1))))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)