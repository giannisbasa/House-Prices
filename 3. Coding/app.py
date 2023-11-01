import pickle
from flask import Flask, request, app, jsonify, url_for, render_template, escape
import pandas
import numpy 

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    
 
