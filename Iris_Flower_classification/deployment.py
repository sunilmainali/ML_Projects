import streamlit as st
import numpy as np
import pickle
import os
Labels = {
    0: 'Iris-setosa',
    1 : 'Iris-versicolor', 
    2 : 'Iris-virginica'
}
# Get the full paths to the pickle files
scaler_path = os.path.join(os.path.dirname(__file__), 'standard_scaler.pkl')
iris_model_path = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')

 #Load both the scaler and the model
with open(scaler_path, 'rb') as f1, open(iris_model_path, 'rb') as f2:
    scaler, model = pickle.load(f1), pickle.load(f2)

#scaler=pickle.load(open("standard_scaler.pkl","rb"))
#model=pickle.load(opem("model.pkl","rb"))

def prediction(user_input:str):
    output=model.predict(scaler.transform(user_input))[0]
    return Labels[output]

if __name__=="__main__":
    st.title("IRIS FLOWER CLASSIFICATION")
    sl=st.number_input("Enter sepal length: ")
    sw=st.number_input("Enter sepal width: ")
    pl=st.number_input("Enter petal length: ")
    pw=st.number_input("Enter petal width: ")

    user_input=np.array([[sl,sw,pl,pw]])

    if st.button("Predict"):
        st.write(
            f" Given flower is {prediction(user_input)}"
        )

