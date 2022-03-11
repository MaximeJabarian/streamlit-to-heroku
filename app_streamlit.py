import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import shap

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from bokeh.plotting import figure


# message_text = st.text_input("Entrez votre age")
class open_file:
    def file_selector(self):
       file = st.sidebar.file_uploader("Choisir un ficher CSV", type="csv")
       if file is not None:
          data = pd.read_csv(file)
          return data
       else:
          st.text("Veuillez télécharger un ficher CSV")

def preprocessing(data):

    # Create a label encoder object
    encoding = LabelEncoder()
    encoding_count = 0

    # Iterate through the columns
    for col in data:

        if data[col].dtype == 'object':

            # If 2 or fewer unique categories

            if len(list(data[col].unique())) <= 2:
                print(col)
                # Train on the training data
                encoding.fit(data[col])
                # Transform both training and testing data
                data[col] = encoding.transform(data[col])

                # Keep track of how many columns were label encoded
                encoding_count += 1

    print('%d colonnes ont été encodées.' % encoding_count)

    # one-hot encoding of categorical variables
    data = pd.get_dummies(data)

    return(data)

def classify_people(model, data, idx):

  label = model.predict(data)
  label_prob = model.predict_proba(data)[:, 1]
  data_res = pd.DataFrame([idx, label, label_prob]).T
  data_res.columns = ['idx', 'TARGET', 'TARGET_probability']

  return(data_res)

## Titre app
st.title("Demande de crédit")
menu = ["Général", "Profil Client"]
choice = st.sidebar.selectbox("Menu",menu)
data_file = st.file_uploader("Veuillez télécharger un ficher CSV",type=["csv"])

if data_file is not None:

    df = pd.read_csv(data_file, sep=';')
    df = df.set_index('Unnamed: 0')
    df_cols = df.columns
    clients_ids = df['SK_ID_CURR'].astype(int)
    df_original = df.copy()

if data_file is not None:

    df = df.sample(frac=1) # frac=1 : shuffe 100%
    np.random.seed(seed=3)
    size = np.random.rand(len(df)) < 0.8
    train_init = df[size]
    test_init = df[~size]
    y_train_init = train_init.iloc[:,-1]
    y_test_init = test_init.iloc[:,-1]
    df = test_init.iloc[:,:-1]
    df_idx = df.index
    df['SK_ID_CURR'] = clients_ids.astype(int)
    df = df.dropna()


    ## Dataset_init: Population d'entrainement
    data_app_train = df_original.copy()
    data_app_train = pd.get_dummies(data_app_train)
    data_app_train['DAYS_BIRTH'] = abs(data_app_train['DAYS_BIRTH'])


    ## Distribution de CREDIT_INCOME_PERCENT
    data_app_train['CREDIT_INCOME_PERCENT'] = data_app_train['AMT_CREDIT'] / data_app_train['AMT_INCOME_TOTAL']
    data_app_train['ANNUITY_INCOME_PERCENT'] = data_app_train['AMT_ANNUITY'] / data_app_train['AMT_INCOME_TOTAL']
    data_app_train['CREDIT_TERM'] = data_app_train['AMT_ANNUITY'] / data_app_train['AMT_CREDIT']
    data_app_train['DAYS_EMPLOYED_PERCENT'] = data_app_train['DAYS_EMPLOYED'] / data_app_train['DAYS_BIRTH']



if choice == "Général" and data_file is not None:

        ################# Plots généraux version st
        # ## Courbe
        # st.line_chart(data['AMT_INCOME_TOTAL'])
        # ## Histo
        # st.bar_chart(data[['AMT_CREDIT', 'AMT_INCOME_TOTAL']])
        # ## Courbe
        # p = figure(title='simple line example', x_axis_label='x', y_axis_label='y')
        # p.line(data['AMT_CREDIT'], data['AMT_ANNUITY'], legend_label='Trend', line_width=2)
        # st.bokeh_chart(p, use_container_width=True)

        ## Distribution de l'age en fonction des classes
        # hist_data = [data_app_train.loc[data_app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365,
        # data_app_train.loc[data_app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365]
        # group_labels = ['target == 0', 'target == 1']
        # fig = ff.create_distplot(hist_data, group_labels, bin_size=[2, 2])
        # st.plotly_chart(fig, use_container_width=True)
        fig = plt.figure(figsize=(12, 20))

        for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):

            # create a new subplot for each source
            plt.subplot(4, 1, i + 1)
            # plot repaid loans
            sns.kdeplot(data_app_train.loc[data_app_train['TARGET'] == 0, feature], label = 'target == 0')
            # plot loans that were not repaid
            sns.kdeplot(data_app_train.loc[data_app_train['TARGET'] == 1, feature], label = 'target == 1')
            # plt.scatter(data.loc[idx][feature])
            # Label the plots
            plt.title('Distribution of %s by Target Value' % feature)
            plt.xlabel('%s' % feature); plt.ylabel('Density')
            plt.legend()

        plt.tight_layout(h_pad = 2.5)
        st.pyplot(fig)
        # fig = plt.figure(figsize=(10, 4))
        #
        # plt.subplot(1, 2, 1)
        # plt.scatter(data['AMT_CREDIT'], data['AMT_INCOME_TOTAL'])
        # plt.scatter(data.loc[idx]['AMT_CREDIT'], data.loc[idx]['AMT_INCOME_TOTAL'])
        #
        # # fig = plt.figure(figsize = (10, 5))
        # plt.subplot(1, 2, 2)
        # plt.bar(data['AMT_CREDIT'], data['AMT_INCOME_TOTAL'])
        # plt.xlabel("Programming Environment")
        # plt.ylabel("Number of Students")
        # plt.title("Students enrolled in different courses")
        # st.pyplot(fig)

        # plt.pie(data['AMT_INCOME_TOTAL'], labels = mylabels)
        # st.pyplot(fig)


        # X1 = data_app_train.loc[data_app_train['TARGET'] == 0, 'CREDIT_INCOME_PERCENT']
        # X2 = data_app_train.loc[data_app_train['TARGET'] == 1, 'CREDIT_INCOME_PERCENT']
        # p = figure(title='simple line example', x_axis_label='x', y_axis_label='y')
        # XX1 = np.linspace(1, len(X1), len(X1))
        # XX2 = np.linspace(1, len(X2), len(X2))
        # p.line(XX1, X1, legend_label='Trend', line_width=2)
        # p.line(XX2, X2, legend_label='Trend', line_width=2)
        # st.bokeh_chart(p, use_container_width=True)

        ############### Plots classiques
        # Distribution de l'age en fonction des remboursements des crédits
        fig = plt.figure(figsize=(10, 4))
        age_data = data_app_train[['TARGET', 'DAYS_BIRTH']]
        age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

        # Bin the age data
        age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
        age_groups  = age_data.groupby('YEARS_BINNED').mean()
        plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
        plt.xticks(rotation = 75)
        plt.xlabel('Groupe d'' âge(année)')
        plt.ylabel('Echec de remboursement (%)')
        plt.title('Echec de remboursement par groupe d'' âge')
        st.pyplot(fig)

elif choice == "Profil Client" and data_file is not None:

        client_id = df['SK_ID_CURR']
        choice = st.sidebar.selectbox("Client ID",client_id)
        idx =  df[df['SK_ID_CURR']==choice].index
        features = ["AMT_INCOME_TOTAL","AMT_CREDIT"]
        feature_choice = st.sidebar.selectbox("Features",features)

        ## Prédiction
        models = ["Regression_Logisitique.pkl","RandomForest.pkl", "XGBOOST.pkl"]
        model_choice = st.sidebar.selectbox("Modèle",models)
        model = pickle.load(open(model_choice, 'rb'))
        df_client = pd.DataFrame(df.loc[idx].values.reshape(1, len(df.columns)), columns=df.columns)
        results = classify_people(model, df_client, idx)

        st.subheader("Crédit accordé au client: \n")

        if results['TARGET_probability'][0] < 0.45:
             res = 'Non'
             st.error(res)

        elif 0.45 < results['TARGET_probability'][0] < 0.55:
            res = 'Incertain'
            st.warning(res)

        else:
            res = 'Oui'
            st.success(res)

        ## Jauge

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = results['TARGET_probability'][0],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilité de remboursement", 'font': {'size': 18}},
            delta = {'reference': 1, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 250], 'color': 'cyan'},
                    {'range': [250, 400], 'color': 'royalblue'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': results['TARGET_probability'][0]}}))

        fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"})
        # fig.show()
        st.plotly_chart(fig)
        st.subheader("Probabilité de remboursement du client: %.1f%%"%(results['TARGET_probability'][0]*1e2))

        fig = plt.figure(figsize=(10, 4))

        idx_target1 = data_app_train[data_app_train['TARGET']==0].index
        idx_target2 = data_app_train[data_app_train['TARGET']==1].index
        plt.scatter(abs(data_app_train.loc[idx_target1, 'DAYS_BIRTH']/365),
         data_app_train.loc[idx_target1, feature_choice], label='TARGET = 0')
        plt.scatter(abs(data_app_train.loc[idx_target2, 'DAYS_BIRTH']/365),
         data_app_train.loc[idx_target2, feature_choice], label='TARGET = 1')
        plt.scatter(abs(data_app_train.loc[idx, 'DAYS_BIRTH']/365),
        data_app_train.loc[idx, feature_choice], color='k', label='Client ID:%d'%int(idx.values))
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure(figsize=(10, 4))
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        i = df.index[0]
        shap.force_plot(explainer.expected_value, shap_values[i], features=df.loc[i], feature_names=df.columns)
        st.pyplot(fig)
