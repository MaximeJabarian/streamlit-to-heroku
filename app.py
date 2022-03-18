# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import requests
import pickle
# import imblearn

##  Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
# Create an instance of the class
app = Flask(__name__)

# route test hello world
@app.route("/")
def hello():
    return "Hello World!"

# noms des fichiers
model = 'RandomForest.pkl'
fic_data = 'DATA/test_init2.csv'

# chargement du modele
pickle_in = open(model,'rb')
classifier = pickle.load(pickle_in)

# chargement des donnees
df = pd.read_csv(fic_data)
# X = df.drop(columns=['TARGET']) 195146

# y = df['TARGET']

# calcul du score

# @app.route('/post', methods=["POST"])
# def testpost():
#      input_json = request.get_json(force=True)
#      dictToReturn = {'text':input_json['text']}
#      return jsonify(dictToReturn)

# @app.route('/api')
@app.route('/api/<int:post_id>')
# @app.route('/')
def mon_api():

    post_id = int(request.args.get('SK_ID_CURR'))
    val = classifier.predict_proba(df)

    dictionnaire = {
        'type': 'Prévision defaut client',
        'valeurs': [val[post_id].tolist()],
        'SK_ID_CURR': post_id
    }
    return jsonify(dictionnaire)

# is a special variable in Python which takes the value of the script name.
# This line ensures that our Flask app runs only when it is executed in the main file
# and not when it is imported in some other file
if __name__ == "__main__":

    app.run(debug=True) #  Run the Flask application
