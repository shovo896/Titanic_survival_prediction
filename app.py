import streamlit as st
import pickle
import pandas as pd

with open('catboost_model_package.pkl','rb') as f:
    model_package = pickle.load(f)

model=model_package['model']
st_title("Titanic Survival Prediction")
st.write("Enter the details of the passenger to predict survival probability.")

Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 30)
SibSp= st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
Parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
Fare = st.slider("Fare", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

FamilySize = SibSp + Parch+1 
IsAlone= 1 if FamilySize==1 else 0
FarePerPerson = Fare / FamilySize if FamilySize > 0 else Fare

Title= "Mr"
if Sex=="female":
    if Age<14:
        Title="Miss"
    else:
        Title="Mrs"
else:
    if Age<14:
        Title="Master"
    else:
        Title="Mr"
input_data = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [Sex],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare],
    'Embarked': [Embarked],
    'FamilySize': [FamilySize],
    'IsAlone': [IsAlone],
    'FarePerPerson': [FarePerPerson],
    'Title': [Title]
})

if st.button("Predict"):
    prediction = model.predict_proba(input_data)[:, 1][0]
    st.write(f"Predicted Survival Probability: {prediction:.2f}")
    if prediction >= model_package['threshold']:
        st.success("The passenger is likely to survive.")
    else:        st.error("The passenger is unlikely to survive.")

