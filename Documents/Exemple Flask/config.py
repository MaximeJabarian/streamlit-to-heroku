
import os

# To generate a new secret key:
# >>> import random, string
# >>> "".join([random.choice(string.printable) for _ in range(24)])
SECRET_KEY = "#d#JCqTTW\nilK\\7m\x0bp#\tj~#H"

FB_APP_ID = 257105156571667

# On donne le chemin de notre base de donnée
basedir = os.path.abspath(os.path.dirname(__file__))

# On indique qu'on va utiliser sqlite pour récup les data, et que notre base s'appelle app.db
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
