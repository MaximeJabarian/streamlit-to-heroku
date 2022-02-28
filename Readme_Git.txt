Lignes de commande pour pusher sur GitHub:

Se connecter sur GitHub via SSH une première fois (et non pas https)
Or create a new repository on the command line
echo "# Openclassrooms2" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:MaximeJabarian/Openclassrooms2.git
git push -u origin main


Or push an existing repository from the command line
git remote add origin git@github.com:MaximeJabarian/Openclassrooms2.git
git branch -M main
git push -u origin main


Pour exclure certains dossiers/fichiers du push, il faut créer un fichier .gitignore dans le dossier racine:
touch .gitignore
vi .gitignore ( écrire le nom des dossiers/fichiers à exclure )


git config --global user.name "Maxime Jabarian"
git config --global user.email maximejabarian@gmail.com
git config --global merge.tool vimdiff

git checkout main
git reset --hard upstream/main
git push --force
git push --set-upstream origin main