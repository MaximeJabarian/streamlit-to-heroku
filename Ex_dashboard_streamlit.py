from pydoc import cli
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

from urllib.error import URLError

# entete
st.title('Home Bank')
st.sidebar.image('logo_bank.jpg')

# variables choisies dans les features importances principales
ListeVar = [ 
    'EXT_SOURCE_3',
    'EXT_SOURCE_2',
    'EXT_SOURCE_1',
    'DAYS_EMPLOYED',
    'AMT_GOODS_PRICE',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'AMT_INCOME_TOTAL']

ListeDef = [
    'score 3 (calcul externe)',
    'score 2 (calcul externe)',
    'score 1 (calcul externe)',
    'nombre de jours en emploi',
    'montant du bien',
    'montant du crédit',
    'annuité',
    'ressources du client']

fic_donnee = 'donnee/donnee_train_pretraitrement_1000.csv'

@st.cache
def get_my_data():
    # chargement des donnees
    app_train = pd.read_csv(fic_donnee)
    X = app_train.drop(columns=['TARGET'])
    return X

try:
    
    df_mydata = get_my_data()

    st.sidebar.write("# Choix d´un identifiant client")

    st.sidebar.info('**Veuillez sélectionner un identifiant \
    client dans le menu déroulant.**')


    client_id = st.sidebar.selectbox( 
        "", list(df_mydata.index),index=1
    )
    
    st.sidebar.write("# Client id:  ", client_id)

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )


# url web 
name_url = 'https://app-api-projet-p7.herokuapp.com/api/'+ str(client_id)

# requete et construction du dataframe
r = requests.get(name_url, timeout=3)
df = pd.read_json(r.content.decode('utf-8'))

# IHM
st.markdown("### Estimation du risque de crédit")
st.markdown("\n\n\n\n")

id_client = df['post_id'].to_list()[0]
val_nondefaut = round(100*df['valeurs'].to_list()[0][0],2)
val_defaut = round(100*df['valeurs'].to_list()[0][1],2)

# recommandation
seuil1 = 45 # seuil de non defaut en pourcentage
seuil2 = 80
if val_nondefaut<seuil1:
    st.error("**Recommandation : dossier refusé**")
elif (val_nondefaut>=seuil1) & (val_nondefaut<seuil2):
    st.warning("**Recommandation : dossier à examiner**")
else:
    st.success("**Recommandation : dossier validé**")

# info client et probabilité
st.write("- **Identifiant client**",id_client)
st.write( 
    "- **Risque estimé: ** ",
     val_defaut," %" )

# Graphique subplot de la liste des variables choisis
fig = plt.figure(figsize=(16, 16))
st.write("\n\n\n### Positionnement du client dans la population\n")
st.markdown("\n\n\n\n")

ind=0
for feat in ListeVar:
    ind += 1
    # Graphique de toute la population
    ax = fig.add_subplot(4,3, ind )
    var = df_mydata.loc[ :, feat ].sort_values()
    var.index = range(len(var))
    plt.scatter(range(len(var)),var)
    
    # Graphique positionnement du client
    var_client = df_mydata.loc[ client_id, feat ]
    idx = var[ var==var_client ].first_valid_index()
    plt.scatter(idx,var_client,color='r',marker = 'o',s = 200)
    ax.set_title(feat, fontsize=14)

st.write(fig)

st.write("\n\n\n### Valeurs descriptives du client")
st.markdown("\n\n\n\n")

# tableau des indicateurs du clients
tab = df_mydata.loc[client_id,ListeVar]
tab = tab.reset_index()
tab.index = ListeDef
tab.columns = ['Nom abrégé de l´indicateur','Client Id: '+str(client_id)]

st.write(tab)
