SE CONNECTER À UN ENVIRONNEMENT VIRTUEL:
	1° python -m virtualenv venv => Créer un environnement virtuel avec un dossier 	"venv" dans le dossier en question
	2° . venv/bin/activate => Activer l'environnement virtuel

Si conflit d'urls locaux avec Flask:
	1° ps -fA | grep python => liste les urls locaux utilisés
	2° kill -9 $(ps -A | grep python | awk '{print $1}') => supprime les urls actifs

CRÉER UNE APP AVEC HEROKU:
I) Heroku cherche deux documents qu'il faut d'abord créer:
	1° Procfile : donne les instructions à exécuter au démarrage de l'application.
	2° requirements.txt : liste les librairies à installer.

II) heroku login
	1° heroku create credit-validation-api
	2° git push heroku main > création du lien web sur heroku
	3° coller le lien_heroku dans Chrome
	4° Si erreur d'affichage: heroku logs --app credit-validation-api

