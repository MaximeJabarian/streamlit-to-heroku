from flask import Flask, render_template

app = Flask(__name__)

app.config.from_object('config')

@app.route('/')
def index():
    return render_template('index.html') # nom tu template à renvoyer à partir de notre dossier template.

# if __name__ == "__main__":
#    app.run()
