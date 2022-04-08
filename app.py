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
# @app.route("/dashboard/")
@app.route("/")
def hello():
    return "Bienvenue sur le <i>dashboard</i>"
    # return render_template("dashboard.html")
# noms des fichiers
model_name = 'RandomForest.pkl'
fic_data = 'DATA2/test_init2.csv'

# chargement du modele
pickle_in = open(model_name,'rb')
model = pickle.load(pickle_in)

# chargement des donnees
data = pd.read_csv(fic_data)


# calcul du score

# @app.route('/api')
@app.route('/api/<int:SK_ID_CURR>')
def mon_api(SK_ID_CURR):

    # SK_ID = request.args.get('SK_ID_CURR') 221792
    idx = data[data['SK_ID_CURR']==SK_ID_CURR].index
    val = model.predict_proba(data.iloc[idx, 1:])

    dictionnaire = {
        'type': 'Prevision defaut client',
        'valeurs': [val[0].tolist()],
        # 'ID client': idx,
        'SK_ID_CURR': SK_ID_CURR
    }
    return jsonify(dictionnaire)

# is a special variable in Python which takes the value of the script name.
# This line ensures that our Flask app runs only when it is executed in the main file
# and not when it is imported in some other file
if __name__ == "__main__":

    app.run(debug=True) #port=5000, debug=True) #  Run the Flask application
