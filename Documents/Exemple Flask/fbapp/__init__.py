from flask import Flask
from .views import app
from . import models
#from models import init_db
import os
models.db.init_app(app) # connection à la base de donnée
#@app.cli.command() # décorateur

def init_db():
    models.init_db()
