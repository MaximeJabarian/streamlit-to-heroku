import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


# message_text = st.text_input("Entrez votre age")
class open_file:
    def file_selector(self):
       file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
       if file is not None:
          data = pd.read_csv(file)
          return data
       else:
          st.text("Please upload a csv file")

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

def classify_people(model, data):

  label = model.predict(data)
  label_prob = model.predict_proba(data)[:, 1]
  data_res = pd.DataFrame([label, label_prob]).T
  data_res.columns = ['TARGET', 'TARGET_probability']

  return(data_res)
  # return {'label': label, 'label_probability': label_prob}

## Titre app
st.title("Demande de crédit accordé ou refusé")
menu = ["Image","Dataset","DocumentFiles","About"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Dataset":

    st.subheader("Dataset")

elif choice == "DocumentFiles":

    st.subheader("DocumentFiles")

if choice == "Dataset":

    # st.subheader("Dataset")
    data_file = st.file_uploader("Upload CSV",type=["csv"])

    if data_file is not None:

        file_details = {"filename":data_file.name, "filetype":data_file.type, "filesize":data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        data = df.set_index('Unnamed: 0')
        # df = df.dropna()
        # data = preprocessing(df)#.drop(columns='ORGANIZATION_TYPE_Trade: type 5')
        # print(data)

# def set_features(self):
#    self.features = st.multiselect('Please choose\
#     the features including target variable that go into the model', self.data.columns )
#

    if data_file is not None:

        client_id = data['SK_ID_CURR']
        choice = st.sidebar.selectbox("Client ID",client_id)
        idx =  data[data['SK_ID_CURR']==choice].index
        st.dataframe([idx])
        st.dataframe(data.loc[idx])

        model = pickle.load(open('log_reg.pkl', 'rb'))
        results = classify_people(model, data.loc[idx].values.reshape(1, len(data.columns)))
        # st.write(""" # Simple Stock Price App ### Shown are the stock **closing price** and **volume** of _**Tesla !**_ ####""")
        st.write(results)
        st.line_chart(data['AMT_INCOME_TOTAL'])

        fig = plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.scatter(data['AMT_CREDIT'], data['AMT_INCOME_TOTAL'])
        plt.scatter(data.loc[idx]['AMT_CREDIT'], data.loc[idx]['AMT_INCOME_TOTAL'])

        # fig = plt.figure(figsize = (10, 5))
        plt.subplot(1, 2, 2)
        plt.bar(data['AMT_CREDIT'], data['AMT_INCOME_TOTAL'])
        plt.xlabel("Programming Environment")
        plt.ylabel("Number of Students")
        plt.title("Students enrolled in different courses")
        st.pyplot(fig)

        # plt.pie(data['AMT_INCOME_TOTAL'], labels = mylabels)
        # st.pyplot(fig)
