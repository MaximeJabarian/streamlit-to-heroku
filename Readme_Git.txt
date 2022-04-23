LIGNES DE COMMANDE POUR PUSHER SUR GITHUB:

SE CONNECTER SUR GITHUB VIA SSH UNE PREMIÈRE FOIS (ET NON PAS HTTPS)
OR CREATE A NEW REPOSITORY ON THE COMMAND LINE
echo "# Openclassrooms2" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:MaximeJabarian/Openclassrooms2.git
git push -u origin main


OR PUSH AN EXISTING REPOSITORY FROM THE COMMAND LINE
git remote add origin git@github.com:MaximeJabarian/Openclassrooms2.git
git branch -M main
git push -u origin main


POUR EXCLURE CERTAINS DOSSIERS/FICHIERS DU PUSH, IL FAUT CRÉER UN FICHIER .GITIGNORE DANS LE DOSSIER RACINE:
touch .gitignore
vi .gitignore ( écrire le nom des dossiers/fichiers à exclure )


git config --global user.name "Maxime Jabarian"
git config --global user.email maximejabarian@gmail.com
git config --global merge.tool vimdiff

git checkout main
git reset --hard upstream/main
git push --force
git push --set-upstream origin main

POUR ANNULER UN PUSH DE FICHIER TROP LOURD:
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch file" --prune-empty --tag-name-filter cat -- --all