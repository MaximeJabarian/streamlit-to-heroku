from flask_sqlalchemy import SQLAlchemy
from .views import app
import logging as lg
import enum

# Create database connection object
# Connection de l'application à la base de donnée
db = SQLAlchemy(app) # modèle pour créer une table

class Gender(enum.Enum):
    female = 0
    male = 1
    other = 2

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True) # l'iD (1ere colonne) est un entier (integer)
    description = db.Column(db.String(200), nullable=False) # 2eme col
    gender = db.Column(db.Enum(Gender), nullable=False) # 3eme col

    def __init__(self, description, gender):
        self.description = description
        self.gender = gender


def init_db():
    db.drop_all() # suppression toutes les données de la database
    db.create_all() # création d'une nouvelle database
    db.session.add(Content("THIS IS SPARTAAAAAAA!!!", 1)) # ajout d'un nouvel élément
    db.session.add(Content("What's your favorite scary movie?", 0)) # idem
    db.session.commit() # envoie les transactions
    lg.warning('Database initialized!') # indication que la base de donnée a bien étée réinitialisée

#db.create_all()
