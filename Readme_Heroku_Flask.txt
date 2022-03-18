Créer une app avec Heroku:
heroku login
heroku create credit-validation-app
git push heroku main

Flask:
ps -fA | grep python => liste les urls locaux utilisés
kill -9 $(ps -A | grep python | awk '{print $1}') => supprime les urls actifs