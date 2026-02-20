import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open('catboost_model_package.pkl','rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
st.title("Titanic Survival Prediction")
st.write("Enter the details of the passenger to predict survival probability.")

Pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 80, 30)
SibSp= st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
Parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)
Fare = st.slider("Fare", 0.0, 500.0, 32.0)
Embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"])

FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0
FarePerson = Fare / FamilySize if FamilySize > 0 else Fare
LogFare = float(np.log1p(Fare))

Title = "Mr"
if Sex == "female":
    if Age < 14:
        Title = "Miss"
    else:
        Title = "Mrs"
else:
    if Age < 14:
        Title = "Master"
    else:
        Title = "Mr"

AgeGroup = pd.cut(
    pd.Series([Age]),
    bins=[0, 12, 18, 35, 60, 80],
    labels=["Child", "Teenager", "Adult", "MiddleAged", "Senior"],
    include_lowest=True,
)[0]

cabin = "U"
Deck = str(cabin)[0] if cabin else "U"
input_data = pd.DataFrame({
    "Pclass": [Pclass],
    "Sex": [Sex],
    "Age": [Age],
    "SibSp": [SibSp],
    "Parch": [Parch],
    "Fare": [Fare],
    "Embarked": [Embarked],
    "Title": [Title],
    "FamilySize": [FamilySize],
    "IsAlone": [IsAlone],
    "FarePerson": [FarePerson],
    "LogFare": [LogFare],
    "AgeGroup": [AgeGroup],
    "cabin": [cabin],
    "Deck": [Deck],
})

expected_features = getattr(model, "feature_names_", None)
if expected_features:
    input_data = input_data.reindex(columns=expected_features)

cat_cols = ["Sex", "Embarked", "Title", "AgeGroup", "cabin", "Deck"]
for col in cat_cols:
    if col in input_data.columns:
        input_data[col] = (
            input_data[col]
            .astype("object")
            .where(input_data[col].notna(), "Missing")
            .astype(str)
        )

if st.button("Predict"):
    prediction = model.predict_proba(input_data)[:, 1][0]
    st.write(f"Predicted Survival Probability: {prediction:.2f}")
    if prediction >= model_package['threshold']:
        st.success("The passenger is likely to survive.")
    else:
        st.error("The passenger is unlikely to survive.")


    
