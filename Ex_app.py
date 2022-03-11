# -*- coding: utf-8 -*-
from flask import Flask, render_template, jsonify
import pandas as pd
import lightgbm as lgb
import json
import requests
from flask import request
import pickle
import imblearn
from imblearn.under_sampling  import RandomUnderSampler
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# route test hello world
@app.route("/")
def hello():
    return "Hello World!"

# noms des fichiers
# fic_model = 'modele/model_file_forest.p'
fic_model = 'modele/model_file.pkl'
fic_data = 'donnee_pretraitre/donnee_train_pretraitrement_1000.csv'

# chargement du modele
pickle_in = open(fic_model,'rb')
classifier = pickle.load(pickle_in)

# chargement des donnees
app_train = pd.read_csv(fic_data)
X = app_train.drop(columns=['TARGET'])
y = app_train['TARGET']

# calcul du score
@app.route('/api/<int:post_id>')
def mon_api(post_id):

    val = classifier['model'].predict_proba(X)
   
    dictionnaire = {
        'type': 'Prévision defaut client',
        'valeurs': [val[post_id].tolist()],
        'post_id': post_id
    }
    return jsonify(dictionnaire)

if __name__ == "__main__":
    application.run(debug=True)