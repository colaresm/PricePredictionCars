from flask import Flask
from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np
import os
from flask_cors import CORS
 

APP = Flask(__name__)
API = Api(APP)
app = Flask("_name_")
CORS(app)

@app.route('/cars', methods = ['GET'])
def returnprine():
    lambdav=-0.26894417210667504
    reg =joblib.load('modelprinceprediction.mdl')
    drivewheel= normalizedata(float(request.args['drivewheel']),1.326829,0.556171)
    wheelbase = normalizedata(float(request.args['wheelbase']),98.756585,6.021776)
    carwidth = normalizedata(float(request.args['carwidth']),65.907805,2.145204)
    carheight = normalizedata(float(request.args['carheight']),53.724878,2.443522)
    curbweight = normalizedata(float(request.args['curbweight']),2555.565854,520.680204)
    enginesize=normalizedata(float(request.args['enginesize']),126.907317,41.642693)
    fuelsystem=normalizedata(float(request.args['fuelsystem']),3.253659,2.013204)
    boreratio=normalizedata(float(request.args['boreratio']),3.329756,0.270844)
    horsepower=normalizedata(float(request.args['horsepower']),104.117073,39.544167)
    citympg=normalizedata(float(request.args['citympg']),25.219512,6.542142)
    highwaympg=normalizedata(float(request.args['highwaympg']),30.751220,6.886443)
    y_r =reg.predict( [[drivewheel,  wheelbase,  carwidth,  carheight,
    curbweight,  enginesize, fuelsystem, boreratio,
    horsepower, citympg,highwaympg]])
    y_predicted = np.power(y_r*lambdav+1,1/lambdav)-1
    y_predicted = y_predicted*7969.3435057200+13276.710570731706
    people = [{'price': round(y_predicted[0],2)}]  
    return jsonify(people[0]) 
         
def normalizedata(data,mean,std):
    return (data-mean)/std

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)