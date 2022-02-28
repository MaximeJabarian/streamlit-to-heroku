Lignes de commande pour lancer un environnement virtuel et l'utiliser:

-$ . venv/bin/activate
-$ python run.py
-$ desactivate

Si conflit de chemin d'env virtuel: 
-$ ps -fA | grep python 
-$ kill "deuxième numéro affiché avant views.py"
ici fbapp est un module dans lequel on place des fonctions et des paramètres pour notre application web.