https://titanicsurvivalprediction-5spjwhwhnawjqrjuptsy5i.streamlit.app/



https://huggingface.co/spaces/shovo896/titanic_survival_prediction1




# 🚢 Titanic Survival Prediction (CatBoost + Streamlit + Docker)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange)
![Framework](https://img.shields.io/badge/UI-Streamlit-red)
![Deployment](https://img.shields.io/badge/Deploy-HuggingFace-yellow)
![Docker](https://img.shields.io/badge/Container-Docker-blue)

An end-to-end Machine Learning project that predicts Titanic passenger survival using **CatBoost**, deployed with **Streamlit**, containerized using **Docker**, and hosted on **Hugging Face Spaces**.

---

# 📌 Project Overview

This project demonstrates a complete production-ready ML workflow:

* Advanced feature engineering
* Stratified K-Fold Cross Validation
* Out-of-Fold (OOF) predictions
* Precision-Recall threshold optimization
* Model serialization (.pkl)
* Streamlit web application
* Docker containerization
* Hugging Face deployment

---

# 🧠 Model Details

| Component   | Description            |
| ----------- | ---------------------- |
| Algorithm   | CatBoostClassifier     |
| Task        | Binary Classification  |
| Metric      | ROC-AUC, Accuracy, F1  |
| Threshold   | Optimized via PR curve |
| CV Strategy | 5-Fold Stratified      |

---

# 📊 Feature Engineering

* Title extraction from Name
* FamilySize calculation
* IsAlone feature
* FarePerPerson
* Title-based Age imputation
* Native categorical encoding via CatBoost

---

# 🏗️ Project Architecture

```
User Input (Streamlit UI)
        ↓
Preprocessing
        ↓
Trained CatBoost Model (.pkl)
        ↓
Probability Output
        ↓
Threshold Optimization
        ↓
Final Prediction
```

---

# 📁 Project Structure

```
titanic-hf-app/
│
├── app.py
├── titanic_catboost_full.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 🐳 Docker Setup (Production Ready)

## 1️⃣ Create `Dockerfile`

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 2️⃣ Build Docker Image

```bash
docker build -t titanic-app .
```

---

## 3️⃣ Run Container

```bash
docker run -p 8501:8501 titanic-app
```

Now open:

```
http://localhost:8501
```

---

# 🚀 Run Locally (Without Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 🌍 Deploy to Hugging Face Spaces

1. Create new Space
2. Select **Streamlit**
3. Upload:

   * app.py
   * model file (.pkl)
   * requirements.txt
4. Hugging Face auto-build করবে

---

# 📦 Requirements

```
streamlit
pandas
catboost
scikit-learn
numpy
```

---

# 📈 Model Performance (Expected)

| Metric   | Score      |
| -------- | ---------- |
| Accuracy | ~0.86–0.87 |
| ROC-AUC  | ~0.88–0.89 |
| F1       | Optimized  |

---

# 🔐 Production Best Practices Followed

* No data leakage
* Proper train-test split
* CV-based threshold tuning
* Model serialization
* Containerized deployment
* Reproducible environment

---


# ⭐ If You Like This Project

Give it a ⭐ and feel free to fork!

---
